"""
tests/test_robustness_dms.py
DMS behaviour under low light and degraded frames (no models loaded -> no face).
"""
import numpy as np
from backend.dms.driver_monitor import DriverMonitor

CFG = {
    "backend": "mediapipe",
    "ear": {"threshold": 0.2, "consec_frames": 20},
    "mar": {"threshold": 0.5, "consec_frames": 12},
    "head_pose": {"yaw_limit": 30, "pitch_limit": 20, "consec_frames": 8},
    "low_light": {"enabled": True, "brightness_threshold": 60},
}


def _dm():
    # No start() -> no face mesh -> every frame reports no face, which is what we want
    return DriverMonitor(CFG, draw_overlays=False)


def _frame(value):
    return np.full((480, 640, 3), value, dtype=np.uint8)


class TestLowLight:
    def test_dark_frame_flagged_low_light(self):
        assert _dm().process(_frame(20)).visibility == "low_light"

    def test_dark_frame_no_face_reports_low_visibility(self):
        assert _dm().process(_frame(20)).driver_state == "LOW VISIBILITY"

    def test_bright_frame_is_good(self):
        r = _dm().process(_frame(200))
        assert r.visibility == "good"
        assert r.driver_state != "LOW VISIBILITY"


class TestBadFrames:
    def test_black_frame_does_not_crash(self):
        r = _dm().process(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.face_detected is False
