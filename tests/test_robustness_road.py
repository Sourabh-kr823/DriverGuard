"""
tests/test_robustness_road.py
Road-detector edge cases: temporal confirmation + graceful behaviour with no model.
"""
import numpy as np
from backend.road.detector import RoadDamageDetector, Detection

BASE = {
    "model_path": "models/yolov8n_rdd_india.pt",
    "temporal_confirmation": True,
    "confirm_frames": 3,
    "confirm_window": 5,
}


def _pothole():
    return [Detection(class_id=0, class_name="pothole", confidence=0.6, bbox=(0, 0, 10, 10))]


class TestTemporalConfirmation:
    def test_single_frame_blip_is_filtered(self):
        det = RoadDamageDetector(BASE)
        assert det._apply_temporal(_pothole()) == []        # 1 of 5 — not enough
        for _ in range(4):
            assert det._apply_temporal([]) == []            # never reaches 3 of 5

    def test_sustained_hazard_is_confirmed(self):
        det = RoadDamageDetector(BASE)
        assert det._apply_temporal(_pothole()) == []        # frame 1: 1/5
        assert det._apply_temporal(_pothole()) == []        # frame 2: 2/5
        assert len(det._apply_temporal(_pothole())) == 1    # frame 3: 3/5 -> confirmed

    def test_disabled_passes_through_immediately(self):
        det = RoadDamageDetector({**BASE, "temporal_confirmation": False})
        assert len(det._apply_temporal(_pothole())) == 1

    def test_confirm_frames_clamped_to_window(self):
        det = RoadDamageDetector({**BASE, "confirm_frames": 99, "confirm_window": 5})
        assert det.confirm_frames == 5


class TestNoModel:
    def test_process_without_model_returns_empty(self):
        det = RoadDamageDetector(BASE)                       # start() not called -> no model
        out = det.process(np.zeros((640, 640, 3), dtype=np.uint8))
        assert out == []
