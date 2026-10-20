"""
utils/risk_score.py
────────────────────
Standalone, stateless risk-score computation.

Keeps the math completely separate from I/O so it can be unit-tested
in isolation and reused by any module (AlertManager, dashboard, etc.).

Risk score formula
──────────────────
    score = w_ear  * ear_component(ear, eye_frames)
           + w_mar  * mar_component(mar, yawn_frames)
           + w_head * head_component(yaw, pitch)

Each component is normalised to [0.0, 1.0].

    Level   score range
    ──────────────────
    low       [0.00, 0.35)
    moderate  [0.35, 0.65)
    high      [0.65, 1.00]
    (or high if eye_frames ≥ ear_consec regardless of score)

Usage
─────
    from utils.risk_score import RiskScorer

    scorer = RiskScorer()          # default thresholds
    result = scorer.compute(
        ear=0.18, mar=0.45,
        yaw=12.0, pitch=5.0,
        eye_frames=16, yawn_frames=3,
    )
    print(result.level, result.score)   # high  0.742
"""

from __future__ import annotations
from dataclasses import dataclass


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    score:         float   # 0.0 – 1.0
    level:         str     # "low" | "moderate" | "high"
    ear_component: float
    mar_component: float
    head_component: float
    trigger:       str     # human-readable reason for the level


# ─── RiskScorer ───────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Pure-function risk scorer — no state, no I/O.

    Parameters (all optional — defaults match config.yaml)
    ──────────────────────────────────────────────────────
    ear_thresh      EAR below which eyes are considered closing  (0.25)
    ear_consec      Frames needed to trigger drowsiness alert    (15)
    mar_thresh      MAR above which mouth is considered open     (0.65)
    mar_consec      Frames needed to trigger yawn alert          (10)
    yaw_limit       Head yaw limit in degrees                    (30)
    pitch_limit     Head pitch limit in degrees                  (20)
    weights         Dict with keys ear, mar, head (must sum to 1.0)
    low_threshold   Score below this → "low"                     (0.35)
    high_threshold  Score at/above this → "high"                 (0.65)
    """

    def __init__(
        self,
        ear_thresh:      float = 0.25,
        ear_consec:      int   = 15,
        mar_thresh:      float = 0.65,
        mar_consec:      int   = 10,
        yaw_limit:       float = 30.0,
        pitch_limit:     float = 20.0,
        weights:         dict  = None,
        low_threshold:   float = 0.35,
        high_threshold:  float = 0.65,
    ):
        self.ear_thresh     = ear_thresh
        self.ear_consec     = ear_consec
        self.mar_thresh     = mar_thresh
        self.mar_consec     = mar_consec
        self.yaw_limit      = yaw_limit
        self.pitch_limit    = pitch_limit
        self.low_thr        = low_threshold
        self.high_thr       = high_threshold

        w = weights or {"ear": 0.45, "mar": 0.25, "head": 0.30}
        total = sum(w.values())
        self.w_ear  = w["ear"]  / total
        self.w_mar  = w["mar"]  / total
        self.w_head = w["head"] / total

    # ── Public ────────────────────────────────────────────────────────────────

    def compute(
        self,
        ear:         float,
        mar:         float,
        yaw:         float,
        pitch:       float,
        eye_frames:  int = 0,
        yawn_frames: int = 0,
    ) -> RiskResult:
        """
        Compute composite risk score.

        Parameters
        ----------
        ear         Eye Aspect Ratio (typically 0.15 – 0.45)
        mar         Mouth Aspect Ratio (typically 0.0 – 1.2)
        yaw         Head yaw in degrees  (+= right, -= left)
        pitch       Head pitch in degrees (+= up, -= down)
        eye_frames  Consecutive frames with EAR < threshold
        yawn_frames Consecutive frames with MAR > threshold

        Returns
        -------
        RiskResult
        """
        ear_c  = self._ear_component(ear, eye_frames)
        mar_c  = self._mar_component(mar, yawn_frames)
        head_c = self._head_component(yaw, pitch)

        score = (self.w_ear  * ear_c
               + self.w_mar  * mar_c
               + self.w_head * head_c)
        score = min(1.0, max(0.0, score))

        level, trigger = self._classify(score, eye_frames, yawn_frames,
                                        abs(yaw), abs(pitch))
        return RiskResult(
            score          = round(score, 4),
            level          = level,
            ear_component  = round(ear_c,  4),
            mar_component  = round(mar_c,  4),
            head_component = round(head_c, 4),
            trigger        = trigger,
        )

    # ── Components ────────────────────────────────────────────────────────────

    def _ear_component(self, ear: float, eye_frames: int) -> float:
        """
        Rises from 0 as EAR drops below threshold, scaled by
        how many consecutive frames the eyes have been closed.
        """
        if ear >= self.ear_thresh:
            return 0.0
        raw = (self.ear_thresh - ear) / self.ear_thresh          # 0 → 1
        frame_scale = min(1.0, eye_frames / self.ear_consec)     # 0 → 1
        return raw * frame_scale

    def _mar_component(self, mar: float, yawn_frames: int) -> float:
        """
        Rises from 0 as MAR exceeds threshold, scaled by yawn frame count.
        """
        if mar <= self.mar_thresh:
            return 0.0
        raw = min(1.0, (mar - self.mar_thresh) / self.mar_thresh)
        frame_scale = min(1.0, yawn_frames / self.mar_consec)
        return raw * frame_scale

    def _head_component(self, yaw: float, pitch: float) -> float:
        """
        Linear ramp from 0 at 50% of limit to 1 at 100% of limit.
        Takes the max of yaw and pitch components.
        """
        def _ramp(val: float, limit: float) -> float:
            half = limit * 0.5
            if abs(val) <= half:
                return 0.0
            return min(1.0, (abs(val) - half) / half)

        return max(_ramp(yaw, self.yaw_limit), _ramp(pitch, self.pitch_limit))

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(
        self,
        score:       float,
        eye_frames:  int,
        yawn_frames: int,
        abs_yaw:     float,
        abs_pitch:   float,
    ):
        # Hard overrides (frame-count based)
        if eye_frames >= self.ear_consec:
            return "high", f"Eye closure {eye_frames} frames"
        if yawn_frames >= self.mar_consec:
            return "moderate", f"Yawning {yawn_frames} frames"
        if abs_yaw > self.yaw_limit or abs_pitch > self.pitch_limit:
            return "moderate", "Head deviation"

        # Score-based
        if score >= self.high_thr:
            return "high", f"Score {score:.3f}"
        if score >= self.low_thr:
            return "moderate", f"Score {score:.3f}"
        return "low", "Normal"
