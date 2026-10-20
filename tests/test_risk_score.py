"""
tests/test_risk_score.py
─────────────────────────
Unit tests for the RiskScorer in utils/risk_score.py.

Run:  pytest tests/test_risk_score.py -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.risk_score import RiskScorer, RiskResult


@pytest.fixture
def scorer():
    return RiskScorer()


class TestRiskScoreLevels:

    def test_all_normal_is_low(self, scorer):
        r = scorer.compute(ear=0.35, mar=0.30, yaw=5, pitch=3)
        assert r.level == "low"
        assert r.score < 0.35

    def test_ear_below_threshold_no_frames_is_low(self, scorer):
        """EAR below threshold but 0 consecutive frames → still low"""
        r = scorer.compute(ear=0.20, mar=0.30, yaw=0, pitch=0, eye_frames=0)
        assert r.level == "low"

    def test_ear_closed_full_frames_is_high(self, scorer):
        """15 consecutive frames with low EAR → high"""
        r = scorer.compute(ear=0.18, mar=0.30, yaw=0, pitch=0, eye_frames=15)
        assert r.level == "high"
        assert r.trigger == "Eye closure 15 frames"

    def test_yawning_full_frames_is_moderate(self, scorer):
        r = scorer.compute(ear=0.35, mar=0.80, yaw=0, pitch=0, yawn_frames=10)
        assert r.level == "moderate"
        assert "Yawning" in r.trigger

    def test_head_deviation_yaw_is_moderate(self, scorer):
        r = scorer.compute(ear=0.35, mar=0.30, yaw=31, pitch=0)
        assert r.level == "moderate"
        assert "Head" in r.trigger

    def test_head_deviation_pitch_is_moderate(self, scorer):
        r = scorer.compute(ear=0.35, mar=0.30, yaw=0, pitch=21)
        assert r.level == "moderate"

    def test_within_head_limits_is_low(self, scorer):
        r = scorer.compute(ear=0.35, mar=0.30, yaw=29, pitch=19)
        assert r.level == "low"

    def test_score_range(self, scorer):
        """Score must always be in [0.0, 1.0]"""
        for ear in [0.10, 0.25, 0.40]:
            for mar in [0.20, 0.65, 1.20]:
                for frames in [0, 8, 15]:
                    r = scorer.compute(ear=ear, mar=mar, yaw=0,
                                       pitch=0, eye_frames=frames)
                    assert 0.0 <= r.score <= 1.0, (
                        f"Score out of range: {r.score} "
                        f"(ear={ear}, mar={mar}, frames={frames})")

    def test_components_non_negative(self, scorer):
        r = scorer.compute(ear=0.35, mar=0.30, yaw=0, pitch=0)
        assert r.ear_component  >= 0.0
        assert r.mar_component  >= 0.0
        assert r.head_component >= 0.0

    def test_result_is_risk_result(self, scorer):
        r = scorer.compute(ear=0.30, mar=0.40, yaw=10, pitch=5)
        assert isinstance(r, RiskResult)
        assert r.level in ("low", "moderate", "high")


class TestRiskScorerCustomThresholds:

    def test_custom_ear_threshold(self):
        scorer = RiskScorer(ear_thresh=0.30, ear_consec=5)
        r = scorer.compute(ear=0.28, mar=0.30, yaw=0, pitch=0, eye_frames=5)
        assert r.level == "high"

    def test_custom_weights_sum_normalised(self):
        """Weights that don't sum to 1.0 should be normalised automatically."""
        scorer = RiskScorer(weights={"ear": 9, "mar": 3, "head": 3})
        r = scorer.compute(ear=0.35, mar=0.30, yaw=0, pitch=0)
        assert 0.0 <= r.score <= 1.0

    def test_high_head_weight(self):
        """With all weight on head, big yaw should give high score."""
        scorer = RiskScorer(
            weights={"ear": 0.01, "mar": 0.01, "head": 0.98},
            high_threshold=0.65,
        )
        r = scorer.compute(ear=0.35, mar=0.30, yaw=45, pitch=0)
        assert r.score > 0.5


class TestEdgeCases:

    def test_perfect_alert_driver(self, scorer):
        """Wide-open eyes, closed mouth, straight head → score near 0"""
        r = scorer.compute(ear=0.45, mar=0.10, yaw=0, pitch=0,
                           eye_frames=0, yawn_frames=0)
        assert r.score < 0.05
        assert r.level == "low"

    def test_worst_case_all_signals(self, scorer):
        r = scorer.compute(ear=0.10, mar=1.20, yaw=45, pitch=30,
                           eye_frames=20, yawn_frames=15)
        assert r.level == "high"
        assert r.score > 0.5
