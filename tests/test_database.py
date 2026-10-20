"""
tests/test_database.py
───────────────────────
Unit tests for DatabaseManager using a temporary SQLite file.

Run:  pytest tests/test_database.py -v
"""

import sys
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.db_manager import DatabaseManager


@pytest.fixture
def db(tmp_path):
    cfg = {
        "path":               str(tmp_path / "test_vsm.db"),
        "flush_interval_sec": 0.1,   # Fast flush for testing
        "max_rows_per_table": 1000,
    }
    manager = DatabaseManager(cfg)
    manager.start()
    yield manager
    manager.stop()


class TestDriverEvents:

    def test_log_and_query_driver_event(self, db):
        db.log_driver_event(
            ear=0.18, mar=0.42, yaw=12.5, pitch=5.0,
            risk="high", state="DROWSY",
            eye_frames=16, yawn_frames=2,
            lat=12.9716, lon=77.5946,
        )
        time.sleep(0.3)  # Wait for background flush
        rows = db.query_recent_driver_events(10)
        assert len(rows) == 1
        row = rows[0]
        assert row["risk"]  == "high"
        assert row["state"] == "DROWSY"
        assert abs(row["ear"] - 0.18) < 0.001
        assert abs(row["lat"] - 12.9716) < 0.0001

    def test_multiple_driver_events_ordered(self, db):
        for i in range(5):
            db.log_driver_event(
                ear=0.30 - i * 0.02, mar=0.40,
                yaw=0, pitch=0,
                risk="low", state="ALERT",
            )
            time.sleep(0.05)
        time.sleep(0.3)
        rows = db.query_recent_driver_events(10)
        assert len(rows) == 5
        # Most recent first
        assert rows[0]["ts"] >= rows[-1]["ts"]

    def test_driver_event_without_gps(self, db):
        db.log_driver_event(
            ear=0.25, mar=0.50, yaw=0, pitch=0,
            risk="moderate", state="FATIGUED",
        )
        time.sleep(0.3)
        rows = db.query_recent_driver_events(5)
        assert rows[0]["lat"] is None
        assert rows[0]["lon"] is None


class TestRoadEvents:

    def test_log_and_query_road_event(self, db):
        db.log_road_event(
            class_name="pothole", confidence=0.87,
            severity="severe", area_px2=12000,
            lat=12.9716, lon=77.5946,
        )
        time.sleep(0.3)
        rows = db.query_recent_road_events(5)
        assert len(rows) == 1
        assert rows[0]["class_name"] == "pothole"
        assert rows[0]["severity"]   == "severe"
        assert abs(rows[0]["confidence"] - 0.87) < 0.001

    def test_multiple_road_classes(self, db):
        db.log_road_event("pothole", 0.80, "moderate", 5000)
        db.log_road_event("crack",   0.65, "minor",    1000)
        db.log_road_event("rutting", 0.90, "severe",   15000)
        time.sleep(0.3)
        rows = db.query_recent_road_events(10)
        names = [r["class_name"] for r in rows]
        assert "pothole" in names
        assert "crack"   in names
        assert "rutting" in names


class TestStats:

    def test_stats_counts(self, db):
        db.log_driver_event(0.18, 0.40, 0, 0, "high", "DROWSY")
        db.log_driver_event(0.30, 0.40, 0, 0, "low",  "ALERT")
        db.log_road_event("pothole", 0.90, "severe", 12000)
        db.log_road_event("crack",   0.70, "minor",  500)
        time.sleep(0.3)
        stats = db.query_stats()
        assert stats["total_driver_events"] == 2
        assert stats["total_road_events"]   == 2
        assert stats["high_risk_events"]    == 1
        assert stats["severe_road_events"]  == 1

    def test_empty_db_stats_are_zero(self, db):
        stats = db.query_stats()
        assert stats["total_driver_events"] == 0
        assert stats["total_road_events"]   == 0


class TestSystemLog:

    def test_system_log_written(self, db):
        db.log_system("INFO", "Test message from unit test")
        time.sleep(0.3)
        import sqlite3
        con = sqlite3.connect(db.db_path)
        rows = con.execute("SELECT * FROM system_log").fetchall()
        con.close()
        messages = [r[3] for r in rows]
        assert any("Test message from unit test" in m for m in messages)
