"""
modules/alert/alert_manager.py
───────────────────────────────
Alert Manager — central hub that:

1. Computes a composite risk score from DMS + road signals.
2. Triggers audio / GPIO buzzer alerts.
3. Emits real-time state to the web dashboard via a shared dict.

Risk score formula (weighted, 0.0 – 1.0)
-----------------------------------------
    score = w_ear * ear_component
           + w_mar * mar_component
           + w_head * head_component

    Low      score < 0.35
    Moderate score < 0.65
    High     score ≥ 0.65  (or eye_frames ≥ threshold)

Usage
-----
    alert = AlertManager(cfg)
    alert.start()

    alert.update(dms_result, road_detections, gps_fix)
    state = alert.state
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


# ─── Shared state ─────────────────────────────────────────────────────────────

@dataclass
class AlertState:
    risk_score:    float = 0.0          # 0.0 – 1.0
    risk_level:    str   = "low"        # "low" | "moderate" | "high"
    driver_state:  str   = "ALERT"
    buzzer_active: bool  = False
    buzzer_reason: str   = ""
    ear:  float = 0.0
    mar:  float = 0.0
    yaw:  float = 0.0
    pitch: float = 0.0
    eye_frames: int = 0
    yawn_frames: int = 0
    road_detections: list = field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_kmh: float = 0.0
    proximity_alert: dict = field(default_factory=dict)  # from ProximityAlertManager
    updated_at: float = field(default_factory=time.time)


# ─── AlertManager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Parameters
    ----------
    cfg : dict  — full config dict (reads `dms.weights`, `alert`)
    """

    def __init__(self, cfg: dict):
        self.weights     = cfg["dms"].get("weights", {"ear": 0.45, "mar": 0.25, "head": 0.30})
        self.alert_cfg   = cfg.get("alert", {})
        self.gpio_pin    = self.alert_cfg.get("buzzer_gpio_pin")
        self.audio_en    = self.alert_cfg.get("audio_alert", False)
        self.audio_file  = self.alert_cfg.get("audio_file", "assets/alert.wav")
        self.escalation  = self.alert_cfg.get("escalation_frames", 5)

        self._state      = AlertState()
        self._lock       = threading.Lock()
        self._buzz_lock  = threading.Lock()
        self._buzzing    = False
        self._stop       = threading.Event()
        self._gpio       = None

        # Debounce: don't re-trigger audio within N seconds
        self._last_audio_ts = 0.0
        self._audio_cooldown = 4.0

        # Proximity alert (set externally by ProximityAlertManager)
        self._proximity_alert: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._init_gpio()
        logger.success("[Alert] AlertManager started ✓")

    def stop(self):
        self._stop.set()
        self._set_buzzer(False)
        if self._gpio:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
            except Exception:
                pass

    def _init_gpio(self):
        if self.gpio_pin is None:
            logger.info("[Alert] GPIO buzzer disabled (no pin configured)")
            return
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            GPIO.output(self.gpio_pin, GPIO.LOW)
            self._gpio = GPIO
            logger.success(f"[Alert] GPIO buzzer ready on BCM pin {self.gpio_pin}")
        except ImportError:
            logger.warning("[Alert] RPi.GPIO not available — GPIO buzzer disabled")
        except Exception as e:
            logger.warning(f"[Alert] GPIO init failed: {e}")

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, dms_result, road_detections, gps_fix=None):
        """
        Called once per frame with latest DMS + road data.

        Parameters
        ----------
        dms_result      : DMSResult
        road_detections : List[Detection]
        gps_fix         : GPSFix | None
        """
        score = self._compute_risk_score(dms_result)
        level = self._score_to_level(score, dms_result)

        # Road: severe pothole overrides to at least moderate
        has_severe = any(d.severity == "severe" for d in road_detections)
        if has_severe and level == "low":
            level = "moderate"
            score = max(score, 0.45)

        should_buzz = (level == "high") or has_severe
        buzzer_reason = ""
        if level == "high":
            buzzer_reason = f"Drowsiness ({dms_result.driver_state})"
        elif has_severe:
            buzzer_reason = "Severe road damage"

        # GPIO + audio
        self._set_buzzer(should_buzz)
        if should_buzz and (time.time() - self._last_audio_ts) > self._audio_cooldown:
            self._play_audio()
            self._last_audio_ts = time.time()

        # Build road summary for state
        road_summary = [
            {"class": d.class_name, "severity": d.severity, "conf": d.confidence}
            for d in road_detections
        ]

        with self._lock:
            self._state = AlertState(
                risk_score     = round(score, 3),
                risk_level     = level,
                driver_state   = dms_result.driver_state,
                buzzer_active  = should_buzz,
                buzzer_reason  = buzzer_reason,
                ear            = dms_result.ear,
                mar            = dms_result.mar,
                yaw            = dms_result.yaw,
                pitch          = dms_result.pitch,
                eye_frames     = dms_result.eye_closed_frames,
                yawn_frames    = dms_result.yawn_frames,
                road_detections= road_summary,
                lat            = gps_fix.lat  if gps_fix else None,
                lon            = gps_fix.lon  if gps_fix else None,
                speed_kmh      = gps_fix.speed_kmh if gps_fix else 0.0,
                proximity_alert= self._proximity_alert,
                updated_at     = time.time(),
            )

    # ── Risk scoring ──────────────────────────────────────────────────────────

    def _compute_risk_score(self, r) -> float:
        # EAR component: 0 when normal, 1 when fully closed for threshold frames
        ear_c  = max(0.0, (0.35 - r.ear) / 0.35) if r.ear < 0.35 else 0.0
        ear_c  = min(1.0, ear_c * (r.eye_closed_frames / 15))

        # MAR component
        mar_c  = max(0.0, (r.mar - 0.50) / 0.50) if r.mar > 0.50 else 0.0
        mar_c  = min(1.0, mar_c * (r.yawn_frames / 10))

        # Head pose component
        yaw_c  = min(1.0, max(0.0, (abs(r.yaw)   - 15) / 15))
        pit_c  = min(1.0, max(0.0, (abs(r.pitch)  - 10) / 10))
        head_c = max(yaw_c, pit_c)

        w  = self.weights
        return (w["ear"] * ear_c + w["mar"] * mar_c + w["head"] * head_c)

    def _score_to_level(self, score: float, r) -> str:
        if r.eye_closed_frames >= 15 or score >= 0.65:
            return "high"
        if r.yawn_frames >= 10 or score >= 0.35:
            return "moderate"
        return "low"

    # ── Buzzer / audio ────────────────────────────────────────────────────────

    def _set_buzzer(self, on: bool):
        if self._gpio is None:
            return
        import RPi.GPIO as GPIO
        try:
            GPIO.output(self.gpio_pin, GPIO.HIGH if on else GPIO.LOW)
        except Exception as e:
            logger.debug(f"[Alert] GPIO write error: {e}")

    def _play_audio(self):
        if not self.audio_en:
            return
        def _play():
            try:
                import playsound
                playsound.playsound(self.audio_file, block=True)
            except Exception as e:
                logger.debug(f"[Alert] Audio error: {e}")
        threading.Thread(target=_play, daemon=True).start()

    # ── Public ────────────────────────────────────────────────────────────────

    def set_proximity_alert(self, alert: dict):
        """Called by ProximityAlertManager to inject the current proximity alert."""
        with self._lock:
            self._proximity_alert = alert or {}

    @property
    def state(self) -> AlertState:
        with self._lock:
            return self._state