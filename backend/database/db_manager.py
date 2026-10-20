"""
backend/database/db_manager.py
───────────────────────────────
SQLite event logger for VSM.

Tables
------
driver_events   — drowsiness / distraction alerts with risk level
road_events     — road damage detections with geo-tag and severity
system_log      — boot/shutdown and system health entries

Usage
-----
    db = DatabaseManager(cfg["database"])
    db.start()

    db.log_driver_event(ear=0.18, mar=0.42, yaw=12, pitch=5,
                        risk="high", state="DROWSY",
                        lat=12.9716, lon=77.5946)

    db.log_road_event(class_name="pothole", confidence=0.87,
                      severity="moderate", area_px2=4500,
                      lat=12.9716, lon=77.5946)

    rows = db.query_recent_driver_events(limit=20)
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger


# ─── DatabaseManager ──────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Thread-safe SQLite wrapper.

    All writes are queued in a deque and flushed by a background
    thread every `flush_interval_sec` seconds to avoid I/O blocking
    the camera pipeline.
    """

    _CREATE_DRIVER = """
    CREATE TABLE IF NOT EXISTS driver_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        ts_iso      TEXT,
        ear         REAL,
        mar         REAL,
        yaw         REAL,
        pitch       REAL,
        risk        TEXT,
        state       TEXT,
        eye_frames  INTEGER,
        yawn_frames INTEGER,
        lat         REAL,
        lon         REAL
    );
    CREATE INDEX IF NOT EXISTS idx_de_ts ON driver_events(ts);
    """

    _CREATE_ROAD = """
    CREATE TABLE IF NOT EXISTS road_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        ts_iso      TEXT,
        class_name  TEXT,
        confidence  REAL,
        severity    TEXT,
        area_px2    INTEGER,
        lat         REAL,
        lon         REAL
    );
    CREATE INDEX IF NOT EXISTS idx_re_ts ON road_events(ts);
    """

    _CREATE_SYSLOG = """
    CREATE TABLE IF NOT EXISTS system_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        level       TEXT,
        message     TEXT
    );
    """

    def __init__(self, cfg: dict):
        self.db_path        = Path(cfg["path"])
        self.flush_interval = cfg.get("flush_interval_sec", 5)
        self.max_rows       = cfg.get("max_rows_per_table", 50_000)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._queue: List[Tuple] = []
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._init_db()
        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="DBFlusher")
        self._thread.start()
        self.log_system("INFO", "VSM database started")
        logger.success(f"[DB] SQLite opened: {self.db_path}")

    def stop(self):
        self.log_system("INFO", "VSM database stopping")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._flush_now()

    def _init_db(self):
        with self._connect() as con:
            con.executescript(self._CREATE_DRIVER)
            con.executescript(self._CREATE_ROAD)
            con.executescript(self._CREATE_SYSLOG)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    # ── Public write API ──────────────────────────────────────────────────────

    def log_driver_event(self, ear: float, mar: float,
                         yaw: float, pitch: float,
                         risk: str, state: str,
                         eye_frames: int = 0, yawn_frames: int = 0,
                         lat: Optional[float] = None,
                         lon: Optional[float] = None):
        now = time.time()
        row = ("driver_events",
               now, time.strftime("%Y-%m-%dT%H:%M:%S"),
               round(ear, 4), round(mar, 4),
               round(yaw, 2), round(pitch, 2),
               risk, state, eye_frames, yawn_frames,
               lat, lon)
        with self._lock:
            self._queue.append(row)

    def log_road_event(self, class_name: str, confidence: float,
                       severity: str, area_px2: int = 0,
                       lat: Optional[float] = None,
                       lon: Optional[float] = None):
        now = time.time()
        row = ("road_events",
               now, time.strftime("%Y-%m-%dT%H:%M:%S"),
               class_name, round(confidence, 4),
               severity, area_px2, lat, lon)
        with self._lock:
            self._queue.append(row)

    def log_system(self, level: str, message: str):
        now = time.time()
        row = ("system_log", now, level, message)
        with self._lock:
            self._queue.append(row)

    # ── Public read API ───────────────────────────────────────────────────────

    def query_recent_driver_events(self, limit: int = 50) -> List[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT * FROM driver_events ORDER BY ts DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def query_recent_road_events(self, limit: int = 50) -> List[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT * FROM road_events ORDER BY ts DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def query_stats(self) -> dict:
        with self._connect() as con:
            de = con.execute("SELECT COUNT(*) FROM driver_events").fetchone()[0]
            re = con.execute("SELECT COUNT(*) FROM road_events").fetchone()[0]
            hs = con.execute(
                "SELECT COUNT(*) FROM driver_events WHERE risk='high'").fetchone()[0]
            sv = con.execute(
                "SELECT COUNT(*) FROM road_events WHERE severity='severe'").fetchone()[0]
        return {
            "total_driver_events": de,
            "total_road_events":   re,
            "high_risk_events":    hs,
            "severe_road_events":  sv,
        }

    # ── Background flush ──────────────────────────────────────────────────────

    def _flush_loop(self):
        while not self._stop.is_set():
            time.sleep(self.flush_interval)
            self._flush_now()

    def _flush_now(self):
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()

        if not batch:
            return

        driver_rows, road_rows, sys_rows = [], [], []
        for row in batch:
            table = row[0]
            data  = row[1:]
            if table == "driver_events":
                driver_rows.append(data)
            elif table == "road_events":
                road_rows.append(data)
            elif table == "system_log":
                sys_rows.append(data)

        try:
            with self._connect() as con:
                if driver_rows:
                    con.executemany(
                        "INSERT INTO driver_events "
                        "(ts,ts_iso,ear,mar,yaw,pitch,risk,state,"
                        " eye_frames,yawn_frames,lat,lon) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", driver_rows)
                if road_rows:
                    con.executemany(
                        "INSERT INTO road_events "
                        "(ts,ts_iso,class_name,confidence,severity,area_px2,lat,lon) "
                        "VALUES (?,?,?,?,?,?,?,?)", road_rows)
                if sys_rows:
                    con.executemany(
                        "INSERT INTO system_log (ts,level,message) VALUES (?,?,?)",
                        sys_rows)
                con.commit()
                logger.debug(f"[DB] Flushed {len(batch)} rows")
        except sqlite3.Error as e:
            logger.error(f"[DB] Flush error: {e}")
            # Put back failed batch
            with self._lock:
                self._queue = batch + self._queue
