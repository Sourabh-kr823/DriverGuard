"""
tests/test_gps_reader.py
─────────────────────────
Unit tests for GPSReader in simulation mode (no hardware needed).

Run:  pytest tests/test_gps_reader.py -v
"""

import sys
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.gps.gps_reader import GPSReader, GPSFix


@pytest.fixture
def gps_sim():
    cfg = {"simulation": True, "port": "/dev/ttyUSB0",
           "baudrate": 9600, "timeout": 1.0}
    reader = GPSReader(cfg)
    reader.start()
    time.sleep(0.5)   # Let sim loop produce a few fixes
    yield reader
    reader.stop()


class TestGPSSimulation:

    def test_latest_returns_gps_fix(self, gps_sim):
        fix = gps_sim.latest
        assert isinstance(fix, GPSFix)

    def test_fix_is_valid(self, gps_sim):
        fix = gps_sim.latest
        assert fix.valid is True

    def test_coordinates_near_bengaluru(self, gps_sim):
        fix = gps_sim.latest
        # Bengaluru bounds roughly: lat 12.8–13.1, lon 77.4–77.8
        assert 12.8 <= fix.lat <= 13.2, f"Lat {fix.lat} out of range"
        assert 77.3 <= fix.lon <= 77.9, f"Lon {fix.lon} out of range"

    def test_speed_is_positive(self, gps_sim):
        fix = gps_sim.latest
        assert fix.speed_kmh >= 0.0

    def test_satellites_reported(self, gps_sim):
        fix = gps_sim.latest
        assert fix.satellites > 0

    def test_coordinates_change_over_time(self, gps_sim):
        fix1 = gps_sim.latest
        time.sleep(0.6)
        fix2 = gps_sim.latest
        # Position must have moved
        assert fix1.lat != fix2.lat or fix1.lon != fix2.lon

    def test_latest_returns_copy(self, gps_sim):
        """Mutating the returned fix must not affect internal state."""
        fix = gps_sim.latest
        original_lat = fix.lat
        fix.lat = 0.0
        assert gps_sim.latest.lat == pytest.approx(original_lat, abs=0.01)


class TestGPSFix:

    def test_default_fix_invalid(self):
        fix = GPSFix()
        assert fix.valid is False
        assert fix.lat is None
        assert fix.lon is None

    def test_fix_fields(self):
        fix = GPSFix(lat=12.97, lon=77.59, speed_kmh=45.0,
                     satellites=9, fix_quality=1, valid=True)
        assert fix.lat == 12.97
        assert fix.speed_kmh == 45.0
        assert fix.valid is True
