"""
tests/test_road_severity.py
────────────────────────────
Unit tests for RoadDamageDetector severity classification.

These tests exercise only the _classify_severity() method — no
YOLOv8 model or camera needed.

Run:  pytest tests/test_road_severity.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.road.detector import RoadDamageDetector, Detection


@pytest.fixture
def detector():
    cfg = {
        "model_path": "models/yolov8n_rdd_india.pt",
        "confidence_threshold": 0.50,
        "iou_threshold": 0.45,
        "severity": {
            "minor":    {"area_px2_max": 3000},
            "moderate": {"area_px2_max": 10000},
        },
    }
    return RoadDamageDetector(cfg)


class TestSeverityClassification:

    def test_tiny_area_is_minor(self, detector):
        assert detector._classify_severity(area=500, conf=0.60) == "minor"

    def test_boundary_minor_to_moderate(self, detector):
        assert detector._classify_severity(area=2999, conf=0.60) == "minor"
        assert detector._classify_severity(area=3000, conf=0.60) == "moderate"

    def test_boundary_moderate_to_severe(self, detector):
        assert detector._classify_severity(area=9999, conf=0.60) == "moderate"
        assert detector._classify_severity(area=10000, conf=0.60) == "severe"

    def test_large_area_is_severe(self, detector):
        assert detector._classify_severity(area=50000, conf=0.90) == "severe"

    def test_zero_area_is_minor(self, detector):
        assert detector._classify_severity(area=0, conf=0.55) == "minor"


class TestDetectionDataclass:

    def test_detection_fields(self):
        d = Detection(
            class_id=0, class_name="pothole",
            confidence=0.87, bbox=(10, 20, 110, 170),
            severity="moderate", area_px2=10000,
        )
        assert d.class_name == "pothole"
        assert d.severity == "moderate"
        assert d.confidence == 0.87
        assert d.lat is None  # geo-tag optional
        assert d.lon is None

    def test_class_names_map(self, detector):
        assert detector.CLASS_NAMES[0] == "pothole"
        assert detector.CLASS_NAMES[1] == "crack"
        assert detector.CLASS_NAMES[2] == "rutting"
