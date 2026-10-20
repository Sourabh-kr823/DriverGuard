"""
backend/gps/gps_reader.py
──────────────────────────
Serial reader for u-blox NEO-6M GPS module.

Reads NMEA sentences from /dev/ttyUSB0 (or configured port), parses
GGA/RMC sentences with pynmea2, and exposes the latest fix in a
thread-safe dataclass.

Simulation mode
---------------
When cfg["simulation"] == True the reader generates a synthetic GPS
trace around Bengaluru for testing without hardware.

Usage
-----
    gps = GPSReader(cfg["gps"])
    gps.start()           # launches background thread

    fix = gps.latest
    print(fix.lat, fix.lon, fix.speed_kmh)
"""

import math
import time
import threading
from dataclasses import dataclass
from typing import Optional

import serial
import pynmea2
from loguru import logger


# ─── Fix dataclass ────────────────────────────────────────────────────────────

@dataclass
class GPSFix:
    lat:       Optional[float] = None    # degrees North
    lon:       Optional[float] = None    # degrees East
    speed_kmh: float = 0.0
    altitude_m: float = 0.0
    heading:   float = 0.0              # degrees (0 = North)
    satellites: int  = 0
    fix_quality: int = 0               # 0 = no fix, 1 = GPS, 2 = DGPS
    timestamp:  Optional[str] = None
    valid:      bool = False


# ─── GPSReader ────────────────────────────────────────────────────────────────

class GPSReader:
    """
    Background serial reader for NEO-6M.

    Parameters
    ----------
    cfg : dict  — the `gps` section from config.yaml
    """

    # Simulation: Bengaluru route trace (lat, lon) pairs
    _SIM_WAYPOINTS = [
        (12.9716, 77.5946),  # Bengaluru city centre
        (12.9776, 77.6000),
        (12.9820, 77.6080),
        (12.9870, 77.6120),
        (12.9920, 77.6060),
        (12.9880, 77.5990),
        (12.9830, 77.5940),
        (12.9760, 77.5900),
    ]

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.port       = cfg.get("port", "/dev/ttyUSB0")
        self.baudrate   = cfg.get("baudrate", 9600)
        self.timeout    = cfg.get("timeout", 1.0)
        self.simulation = cfg.get("simulation", False)

        self._fix   = GPSFix()
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the background reader thread."""
        target = self._sim_loop if self.simulation else self._serial_loop
        self._thread = threading.Thread(target=target, daemon=True, name="GPSReader")
        self._thread.start()
        logger.info(f"[GPS] Reader started  "
                    f"({'simulation' if self.simulation else self.port})")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    # ── Serial loop ───────────────────────────────────────────────────────────

    def _serial_loop(self):
        while not self._stop.is_set():
            try:
                with serial.Serial(self.port, self.baudrate,
                                   timeout=self.timeout) as ser:
                    logger.success(f"[GPS] Connected: {self.port} @ {self.baudrate}baud")
                    while not self._stop.is_set():
                        raw = ser.readline().decode("ascii", errors="replace").strip()
                        self._parse_sentence(raw)
            except serial.SerialException as e:
                logger.warning(f"[GPS] Serial error: {e} — retrying in 5s")
                time.sleep(5)

    def _parse_sentence(self, sentence: str):
        if not sentence.startswith("$"):
            return
        try:
            msg = pynmea2.parse(sentence)
        except pynmea2.ParseError:
            return

        with self._lock:
            # GGA — fix quality, satellites, altitude
            if isinstance(msg, pynmea2.types.talker.GGA):
                if msg.latitude and msg.longitude:
                    self._fix.lat         = msg.latitude
                    self._fix.lon         = msg.longitude
                    self._fix.altitude_m  = float(msg.altitude or 0)
                    self._fix.satellites  = int(msg.num_sats or 0)
                    self._fix.fix_quality = int(msg.gps_qual or 0)
                    self._fix.timestamp   = str(msg.timestamp)
                    self._fix.valid       = msg.gps_qual > 0

            # RMC — speed, heading
            elif isinstance(msg, pynmea2.types.talker.RMC):
                if msg.status == "A":
                    self._fix.speed_kmh = float(msg.spd_over_grnd or 0) * 1.852
                    self._fix.heading   = float(msg.true_course or 0)
                    if msg.latitude and msg.longitude:
                        self._fix.lat   = msg.latitude
                        self._fix.lon   = msg.longitude
                        self._fix.valid = True

    # ── Simulation loop ───────────────────────────────────────────────────────

    def _sim_loop(self):
        wp  = self._SIM_WAYPOINTS
        idx = 0
        t   = 0.0
        while not self._stop.is_set():
            a  = wp[idx % len(wp)]
            b  = wp[(idx + 1) % len(wp)]
            lat = a[0] + (b[0] - a[0]) * t
            lon = a[1] + (b[1] - a[1]) * t
            # Jitter ± 0.00005 degrees (≈ 5 m)
            import random
            lat += random.uniform(-0.00005, 0.00005)
            lon += random.uniform(-0.00005, 0.00005)
            speed = 30 + random.uniform(-5, 15)
            with self._lock:
                self._fix = GPSFix(
                    lat=lat, lon=lon,
                    speed_kmh=speed,
                    altitude_m=920.0,
                    heading=math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])) % 360,
                    satellites=8,
                    fix_quality=1,
                    timestamp=time.strftime("%H:%M:%S"),
                    valid=True,
                )
            t += 0.05
            if t >= 1.0:
                t = 0.0
                idx += 1
            time.sleep(0.2)

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def latest(self) -> GPSFix:
        with self._lock:
            return GPSFix(**self._fix.__dict__)
