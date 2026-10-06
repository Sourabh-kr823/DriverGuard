"""
modules/dms/driver_monitor.py
──────────────────────────────
DriverMonitor — real-time per-frame DMS pipeline.

Usage
-----
    from modules.dms.driver_monitor import DriverMonitor

    dms = DriverMonitor(cfg["dms"])
    dms.start()

    while True:
        frame = cam.read()
        result = dms.process(frame)
        # result.risk  ->  "low" | "moderate" | "high"

Fix history
-----------
    v2.2 — Five false-trigger bugs fixed (identified from June-17 session logs):
        BUG 1  DISTRACTED had zero frame counter — any instant yaw spike fired it.
               Added _head_counter requiring CONSEC_HEAD sustained frames (default 8).
        BUG 2  EAR calibration baseline was polluted by blink frames, pulling it
               low (0.217 vs typical 0.277-0.287). Added top-75% percentile filter
               to exclude blink frames from the mean.
        BUG 3  "Drowsy watch" FATIGUED fired on slow normal blinks (333ms).
               Raised threshold from ear_consec//2 (10fr) to ear_consec*3//4 (15fr).
        BUG 4  MP_MOUTH had duplicate landmark indices 61 & 291, and used wrong
               points for the denominator — giving MAR 1.3-4.2 for a closed mouth.
               Fixed in ear_mar.py: proper 6-point lips [61,82,312,291,317,87].
        BUG 5  MAR calibration included open-mouth / speech frames, skewing the
               baseline high (1.746 vs typical 1.294-1.346 in old scale; now
               ~0.05-0.15 in corrected scale). Added bottom-50% percentile filter.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from .ear_mar   import compute_ear_mar
from .head_pose import (estimate_head_pose_dlib,
                        estimate_head_pose_mediapipe,
                        draw_head_pose_axes)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class DMSResult:
    ear: float = 0.0
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    eye_closed_frames: int = 0
    yawn_frames: int = 0
    head_frames: int = 0          # NEW: consecutive frames of head deviation
    risk: str = "low"             # "low" | "moderate" | "high"
    driver_state: str = "ALERT"   # "ALERT" | "FATIGUED" | "DROWSY" | "DISTRACTED"
    alert_info: str = ""
    face_detected: bool = False
    annotated_frame: Optional[np.ndarray] = None


# ─── DriverMonitor ────────────────────────────────────────────────────────────

class DriverMonitor:
    """
    Orchestrates face detection, landmark extraction, EAR/MAR, and head pose.

    Parameters
    ----------
    cfg : dict  — the `dms` section from config.yaml
    """

    def __init__(self, cfg: dict, draw_overlays: bool = True):
        self.cfg = cfg
        self.backend = cfg.get("backend", "both")
        self._lock   = threading.Lock()
        # PERF: when False, skip frame.copy(), HUD text, landmark dots, pose axes
        self.draw_overlays = draw_overlays

        # ── Frame counters ────────────────────────────────────────────────────
        self._eye_counter  = 0
        self._yawn_counter = 0
        self._head_counter = 0   # BUG 1 FIX: new sustained head-deviation counter

        # ── Thresholds — loaded from config, overridden by auto-calibration ──
        self.ear_thresh  = cfg["ear"]["threshold"]
        self.ear_consec  = cfg["ear"]["consec_frames"]
        self.mar_thresh  = cfg["mar"]["threshold"]
        self.mar_consec  = cfg["mar"]["consec_frames"]
        self.yaw_limit   = cfg["head_pose"]["yaw_limit"]
        self.pitch_limit = cfg["head_pose"]["pitch_limit"]

        # BUG 1 FIX: frames of sustained head deviation required for DISTRACTED
        # Configurable via head_pose.consec_frames (default 8 = ~267ms at 30fps)
        self._CONSEC_HEAD = cfg.get("head_pose", {}).get("consec_frames", 8)

        # ── Models (lazy-loaded in start()) ───────────────────────────────────
        self._dlib_detector  = None
        self._dlib_predictor = None
        self._mp_face_mesh   = None

        self._last_result: DMSResult = DMSResult()

        # ── Auto-calibration ──────────────────────────────────────────────────
        # Collects EAR/MAR readings for first 90 frames (~3 s at 30 fps)
        # while the driver sits normally (eyes open, mouth closed), then sets
        # personal thresholds.  Works for any face — bearded, glasses, etc.
        self._ear_samples   = []
        self._mar_samples   = []
        self._ear_baseline  = None   # set after calibration
        self._mar_baseline  = None   # set after calibration
        self._calibrated    = False  # True once both are done

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Load models. Call once before process()."""
        logger.info("[DMS] Loading models …")

        # Reset calibration on every start so a new driver gets re-calibrated
        self._ear_samples  = []
        self._mar_samples  = []
        self._ear_baseline = None
        self._mar_baseline = None
        self._calibrated   = False
        self._eye_counter  = 0
        self._yawn_counter = 0
        self._head_counter = 0

        if self.backend in ("dlib", "both"):
            try:
                import dlib
                self._dlib_detector  = dlib.get_frontal_face_detector()
                self._dlib_predictor = dlib.shape_predictor(
                    self.cfg["dlib_model_path"])
                logger.success("[DMS] Dlib 68-pt model loaded ✓")
            except Exception as e:
                logger.error(f"[DMS] Dlib load failed: {e}")

        if self.backend in ("mediapipe", "both"):
            try:
                import mediapipe as mp
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,   # PERF: iris tracking not needed → saves ~15-20ms/frame
                    min_detection_confidence=self.cfg.get("mediapipe_confidence", 0.5),
                    min_tracking_confidence=0.5,
                )
                logger.success("[DMS] MediaPipe 468-pt model loaded ✓")
            except Exception as e:
                logger.error(f"[DMS] MediaPipe load failed: {e}")

    def stop(self):
        if self._mp_face_mesh:
            self._mp_face_mesh.close()

    def recalibrate(self):
        """
        Trigger fresh calibration for a new driver without restarting the app.
        Models stay loaded — only calibration state and counters are reset.
        Thresholds revert to config defaults until the new 90-frame window
        completes (~3 s at 30 fps).
        """
        with self._lock:
            self._ear_samples  = []
            self._mar_samples  = []
            self._ear_baseline = None
            self._mar_baseline = None
            self._calibrated   = False
            self._eye_counter  = 0
            self._yawn_counter = 0
            self._head_counter = 0
            # Restore config defaults until new calibration completes
            self.ear_thresh = self.cfg["ear"]["threshold"]
            self.mar_thresh = self.cfg["mar"]["threshold"]
        logger.info("[DMS] Recalibration triggered — new driver detected")

    # ── Main process loop ─────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> DMSResult:
        """
        Run the full DMS pipeline on one frame.

        Parameters
        ----------
        frame : np.ndarray  BGR image from OpenCV

        Returns
        -------
        DMSResult
        """
        # PERF: only copy the frame if we will actually draw on it
        annotated = frame.copy() if self.draw_overlays else frame
        result = DMSResult(annotated_frame=annotated)
        h, w = frame.shape[:2]

        # PERF: grayscale is only needed by dlib; skip it in mediapipe-only mode
        gray = None
        if self._dlib_detector and self._dlib_predictor:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ear, mar = 0.0, 0.0
        yaw, pitch, roll = 0.0, 0.0, 0.0
        face_found = False

        # ── Try dlib ──────────────────────────────────────────────────────────
        if self._dlib_detector and self._dlib_predictor and gray is not None:
            faces = self._dlib_detector(gray, 0)
            if faces:
                face_found = True
                shape = self._dlib_predictor(gray, faces[0])
                d_ear, d_mar = compute_ear_mar(shape, "dlib")
                d_yaw, d_pitch, d_roll = estimate_head_pose_dlib(shape, h, w)
                ear, mar = d_ear, d_mar
                yaw, pitch, roll = d_yaw, d_pitch, d_roll
                if self.draw_overlays:
                    self._annotate_dlib(result.annotated_frame, shape, faces[0])

        # ── Try mediapipe ─────────────────────────────────────────────────────
        if self._mp_face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = self._mp_face_mesh.process(rgb)
            if mp_result.multi_face_landmarks:
                face_found = True
                fl = mp_result.multi_face_landmarks[0]
                mp_ear, mp_mar = compute_ear_mar(fl, "mediapipe", w, h)
                mp_yaw, mp_pitch, mp_roll = estimate_head_pose_mediapipe(fl, h, w)

                if self.backend == "both" and ear != 0.0:
                    ear   = (ear   + mp_ear)   / 2
                    mar   = (mar   + mp_mar)   / 2
                    yaw   = (yaw   + mp_yaw)   / 2
                    pitch = (pitch + mp_pitch) / 2
                else:
                    ear, mar   = mp_ear, mp_mar
                    yaw, pitch = mp_yaw, mp_pitch
                    roll       = mp_roll

                if self.draw_overlays:
                    draw_head_pose_axes(result.annotated_frame, fl,
                                        "mediapipe", yaw, pitch, roll)

        # ── Auto-calibration ──────────────────────────────────────────────────
        # Runs silently for first 90 frames (~3 s) while face is detected.
        # Sit normally (eyes open, mouth closed) during this period.
        #
        # BUG 2 FIX — EAR: use top-75% of samples to exclude blink frames.
        #   A normal blink lasts 5-12 frames and gives EAR ~0.10-0.18.
        #   Sorting and taking the upper 75% removes these low outliers so the
        #   baseline reflects true open-eye EAR, not a blink-polluted average.
        #
        # BUG 5 FIX — MAR: use bottom-50% of samples to exclude open-mouth /
        #   speech frames.  Closed-mouth frames have the LOWEST MAR values, so
        #   the bottom half of the sorted distribution is the cleanest baseline.
        if face_found and not self._calibrated:
            if ear > 0.10:
                self._ear_samples.append(ear)
            if mar > 0.02:   # threshold adjusted for new proper MAR scale
                self._mar_samples.append(mar)

            # EAR calibration — top-75% percentile filter (BUG 2 fix)
            if len(self._ear_samples) >= 90 and self._ear_baseline is None:
                sorted_ear = sorted(self._ear_samples)
                # Discard bottom 25% (blink frames) and average the rest
                open_eye_samples = sorted_ear[len(sorted_ear) // 4:]
                self._ear_baseline = sum(open_eye_samples) / len(open_eye_samples)
                # threshold = baseline - 0.08
                # Normal blink: <400 ms (<12 fr) — won't sustain past ear_consec
                # Drowsy closure: >500 ms (>15 fr) — will trigger
                self.ear_thresh = max(0.15, self._ear_baseline - 0.08)
                logger.info(
                    f"[DMS] EAR calibrated — "
                    f"baseline: {self._ear_baseline:.3f}  "
                    f"threshold: {self.ear_thresh:.3f}  "
                    f"(from {len(open_eye_samples)} open-eye samples)"
                )

            # MAR calibration — bottom-50% percentile filter (BUG 5 fix)
            if len(self._mar_samples) >= 90 and self._mar_baseline is None:
                sorted_mar = sorted(self._mar_samples)
                # Take the lower half = frames where mouth is most closed
                closed_mouth_samples = sorted_mar[:len(sorted_mar) // 2]
                self._mar_baseline = sum(closed_mouth_samples) / len(closed_mouth_samples)
                # threshold = baseline + 0.38
                # With proper 6-pt MAR, closed mouth ~0.05-0.12, yawn ~0.50-0.80
                # So threshold lands at ~0.43-0.50, well below a genuine yawn
                self.mar_thresh = self._mar_baseline + 0.38
                logger.info(
                    f"[DMS] MAR calibrated — "
                    f"baseline: {self._mar_baseline:.3f}  "
                    f"threshold: {self.mar_thresh:.3f}  "
                    f"(from {len(closed_mouth_samples)} closed-mouth samples)"
                )

            if self._ear_baseline is not None and self._mar_baseline is not None:
                self._calibrated = True
                logger.success("[DMS] Auto-calibration complete ✓")

        # ── Update frame counters ─────────────────────────────────────────────
        if face_found:
            # EAR counter — only count eye closure when head is roughly forward.
            # Head rotation naturally compresses the 2-D EAR projection; that
            # is NOT real drowsiness and should not count.
            head_facing_forward = abs(yaw) < 25 and abs(pitch) < 18

            if ear < self.ear_thresh and head_facing_forward:
                self._eye_counter += 1
            else:
                self._eye_counter = 0   # instant reset: correct for EAR

            # Yawn counter — instant reset when mouth closes
            if mar > self.mar_thresh:
                self._yawn_counter += 1
            else:
                self._yawn_counter = 0

            # BUG 1 FIX — Head counter with gradual decay.
            # OLD behaviour: _classify_risk checked abs(yaw) inline → ANY single
            #   frame with yaw > yaw_limit fired DISTRACTED immediately.
            # NEW behaviour: _head_counter increments while head is deviated and
            #   decays by 3/frame when centred, requiring _CONSEC_HEAD sustained
            #   frames (~267 ms at 30 fps) before DISTRACTED is reported.
            if abs(yaw) > self.yaw_limit or abs(pitch) > self.pitch_limit:
                self._head_counter = min(self._head_counter + 1,
                                         self._CONSEC_HEAD + 10)  # cap to avoid overflow
            else:
                self._head_counter = max(0, self._head_counter - 3)  # fast decay when centred

        else:
            # No face in frame — reset all counters immediately
            self._eye_counter  = 0
            self._yawn_counter = 0
            self._head_counter = max(0, self._head_counter - 3)

        # ── Risk classification ───────────────────────────────────────────────
        risk, state, info = self._classify_risk(
            ear, mar, yaw, pitch,
            self._eye_counter, self._yawn_counter, self._head_counter
        )

        # ── Build result ──────────────────────────────────────────────────────
        result.ear               = round(ear,   3)
        result.mar               = round(mar,   3)
        result.yaw               = round(yaw,   1)
        result.pitch             = round(pitch, 1)
        result.roll              = round(roll,  1)
        result.eye_closed_frames = self._eye_counter
        result.yawn_frames       = self._yawn_counter
        result.head_frames       = self._head_counter   # NEW field
        result.risk              = risk
        result.driver_state      = state
        result.alert_info        = info
        result.face_detected     = face_found

        if self.draw_overlays:
            self._overlay_hud(result.annotated_frame, result)
        with self._lock:
            self._last_result = result
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _classify_risk(
        self, ear, mar, yaw, pitch, ecf, yf, hcf
    ) -> Tuple[str, str, str]:
        """
        Map current counters to (risk_level, driver_state, info_string).

        Parameters
        ----------
        ecf : eye_closed_frames counter
        yf  : yawn_frames counter
        hcf : head_frames counter  (BUG 1 fix — replaces inline yaw/pitch check)
        """
        # Priority 1 — DROWSY (eye closure sustained >= ear_consec frames)
        if ecf >= self.ear_consec:
            return "high", "DROWSY", f"Eye closure {ecf} fr"

        # Priority 2 — FATIGUED from yawning
        if yf >= self.mar_consec:
            return "moderate", "FATIGUED", f"Yawning {yf} fr"

        # Priority 3 — DISTRACTED from sustained head deviation
        # BUG 1 FIX: require hcf >= _CONSEC_HEAD instead of instant inline check.
        # Prevents flickering from momentary mirror-check / GPS glance.
        if hcf >= self._CONSEC_HEAD:
            return "moderate", "DISTRACTED", f"Head deviation {hcf} fr"

        # Priority 4 — early drowsiness warning ("drowsy watch")
        # BUG 3 FIX: raised threshold from ear_consec//2 (10 fr, 333 ms) to
        # ear_consec*3//4 (15 fr, 500 ms) to avoid false positives from slow
        # normal blinks, which typically peak at ~12 frames (400 ms).
        if ecf > self.ear_consec * 3 // 4:
            return "moderate", "FATIGUED", "Drowsy watch"

        return "low", "ALERT", ""

    def _annotate_dlib(self, frame, shape, face_rect):
        import dlib
        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 160), 1)
        for i in range(68):
            pt = shape.part(i)
            cv2.circle(frame, (pt.x, pt.y), 1, (0, 180, 120), -1)

    def _overlay_hud(self, frame, r: DMSResult):
        colour = {"low": (0, 229, 160), "moderate": (245, 158, 11),
                  "high": (239, 68, 68)}[r.risk]
        lines = [
            f"EAR: {r.ear:.2f}  MAR: {r.mar:.2f}",
            f"Yaw: {r.yaw:+.0f}  Pit: {r.pitch:+.0f}",
            f"State: {r.driver_state}  Risk: {r.risk.upper()}",
            f"EyeFr:{r.eye_closed_frames:02d}/{self.ear_consec}"
            f"  YawnFr:{r.yawn_frames:02d}/{self.mar_consec}"
            f"  HdFr:{r.head_frames:02d}/{self._CONSEC_HEAD}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (6, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

        # Calibration status indicator — top right corner
        if self._calibrated:
            calib_text  = "CAL OK"
            calib_color = (0, 200, 0)      # green
        else:
            samples     = min(len(self._ear_samples), len(self._mar_samples))
            pct         = int(samples / 90 * 100)
            calib_text  = f"CAL {pct}%"
            calib_color = (0, 180, 255)    # cyan

        cv2.putText(frame, calib_text, (frame.shape[1] - 70, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, calib_color, 1, cv2.LINE_AA)

    @property
    def latest(self) -> DMSResult:
        with self._lock:
            return self._last_result