
from types import SimpleNamespace
from backend.alert.alert_manager import AlertManager

CFG = {"dms": {"weights": {"ear": 0.45, "mar": 0.25, "head": 0.30}}, "alert": {}}

def _am():
    return AlertManager(CFG, voice_alert=None)


def _dms(ear=0.3, mar=0.1, yaw=0, pitch=0, ecf=0, yf=0, state="ALERT", vis="good"):
    return SimpleNamespace(
        ear=ear, mar=mar, yaw=yaw, pitch=pitch,
        eye_closed_frames=ecf, yawn_frames=yf,
        driver_state=state, visibility=vis,
    )

class TestRiskBoundaries:
    def test_score_within_unit_range_for_extremes(self):
        extreme = _dms(ear=0.0, mar=1.0, yaw=90, pitch=90, ecf=100, yf=100)
        s = _am()._compute_risk_score(extreme)
        assert 0.0 <= s <= 1.0

    def test_calm_driver_is_low(self):
        am = _am()
        r = _dms()
        assert am._score_to_level(am._compute_risk_score(r), r) == "low"

    def test_sustained_eye_closure_is_high(self):
        am = _am()
        r = _dms(ear=0.1, ecf=20)
        assert am._score_to_level(am._compute_risk_score(r), r) == "high"

class TestMissingGps:
    def test_none_gps_does_not_crash(self):
        am = _am()
        am.update(dms_result=_dms(), road_detections=[], gps_fix=None)
        assert am.state.lat is None
        assert am.state.speed_kmh == 0.0

class TestSevereRoadOverride:
    def test_severe_hazard_bumps_level(self):
        am = _am()
        severe = SimpleNamespace(severity="severe", class_name="pothole", confidence=0.9)
        am.update(dms_result=_dms(), road_detections=[severe], gps_fix=None)
        assert am.state.risk_level in ("moderate", "high")