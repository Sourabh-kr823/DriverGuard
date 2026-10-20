"""
tests/test_ear_mar.py
──────────────────────
Unit tests for EAR and MAR computation.

Tests verify the mathematical formulae directly using synthetic
landmark arrays — no camera or model needed.

Run:  pytest tests/test_ear_mar.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.dms.ear_mar import (
    _eye_aspect_ratio,
    _mouth_aspect_ratio,
    compute_ear_mar,
)


# ─── EAR tests ────────────────────────────────────────────────────────────────

class TestEAR:
    """Eye Aspect Ratio formula: (A + B) / (2 * C)"""

    def _eye(self, vertical: float = 1.0, horizontal: float = 4.0) -> np.ndarray:
        """
        Build a synthetic 6-point eye where A=B=vertical/2 and C=horizontal.
        Expected EAR = (v/2 + v/2) / (2 * horizontal) = v / (2 * h)
        """
        return np.array([
            [0.0,             0.0       ],   # p1 left corner
            [horizontal/3,    vertical/2],   # p2 upper-lid left
            [2*horizontal/3,  vertical/2],   # p3 upper-lid right
            [horizontal,      0.0       ],   # p4 right corner
            [2*horizontal/3, -vertical/2],   # p5 lower-lid right
            [horizontal/3,   -vertical/2],   # p6 lower-lid left
        ], dtype=np.float64)

    def test_open_eye_ear_above_threshold(self):
        eye = self._eye(vertical=1.0, horizontal=2.0)
        ear = _eye_aspect_ratio(eye)
        # EAR ≈ 1.0 / (2 * 2.0) = 0.25 exactly at threshold, open should be higher
        # With v=1.0, h=2.0: EAR = 1.0/4.0 = 0.25
        assert ear == pytest.approx(0.25, abs=1e-4)

    def test_wide_open_eye(self):
        eye = self._eye(vertical=2.0, horizontal=4.0)
        ear = _eye_aspect_ratio(eye)
        assert ear == pytest.approx(0.25, abs=1e-4)

    def test_closed_eye_near_zero(self):
        """Closed eye → vertical distance ≈ 0 → EAR ≈ 0"""
        eye = self._eye(vertical=0.01, horizontal=4.0)
        ear = _eye_aspect_ratio(eye)
        assert ear < 0.01

    def test_ear_alert_threshold(self):
        """EAR < 0.25 should signal potential closure"""
        eye = self._eye(vertical=0.5, horizontal=4.0)
        ear = _eye_aspect_ratio(eye)
        assert ear < 0.25

    def test_ear_non_negative(self):
        """EAR must always be non-negative"""
        for v in [0.0, 0.1, 0.5, 1.0, 2.0]:
            eye = self._eye(vertical=v, horizontal=4.0)
            assert _eye_aspect_ratio(eye) >= 0.0

    def test_ear_symmetry(self):
        """Mirroring the eye horizontally should not change EAR"""
        eye = self._eye(vertical=1.0, horizontal=4.0)
        mirrored = eye.copy()
        mirrored[:, 0] = -mirrored[:, 0]
        assert _eye_aspect_ratio(eye) == pytest.approx(
            _eye_aspect_ratio(mirrored), abs=1e-6)


# ─── MAR tests ────────────────────────────────────────────────────────────────

class TestMAR:
    """Mouth Aspect Ratio: (A + B + C) / (2 * D) — similar to EAR"""

    def _mouth(self, vertical: float = 1.0, horizontal: float = 3.0) -> np.ndarray:
        """Synthetic 8-point mouth."""
        return np.array([
            [0.0,              0.0       ],   # p0 left corner
            [horizontal * 0.3, vertical/2],   # p1 upper-lip left
            [horizontal * 0.5, vertical  ],   # p2 upper-lip centre
            [horizontal * 0.7, vertical/2],   # p3 upper-lip right
            [horizontal,       0.0       ],   # p4 right corner
            [horizontal * 0.7,-vertical/2],   # p5 lower-lip right
            [horizontal * 0.5,-vertical  ],   # p6 lower-lip centre
            [horizontal * 0.3,-vertical/2],   # p7 lower-lip left
        ], dtype=np.float64)

    def test_closed_mouth_below_threshold(self):
        mouth = self._mouth(vertical=0.1, horizontal=3.0)
        mar = _mouth_aspect_ratio(mouth)
        assert mar < 0.65

    def test_yawn_above_threshold(self):
        mouth = self._mouth(vertical=2.5, horizontal=3.0)
        mar = _mouth_aspect_ratio(mouth)
        assert mar > 0.65

    def test_mar_non_negative(self):
        for v in [0.0, 0.5, 1.0, 2.0]:
            mouth = self._mouth(vertical=v, horizontal=3.0)
            assert _mouth_aspect_ratio(mouth) >= 0.0


# ─── compute_ear_mar dispatcher tests ────────────────────────────────────────

class TestComputeDispatcher:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            compute_ear_mar(None, "faceapi")
