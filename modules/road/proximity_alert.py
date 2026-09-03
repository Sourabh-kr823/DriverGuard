"""
modules/road/proximity_alert.py
────────────────────────────────
Predictive Road Hazard Alert System

Layer 1 — Local History (offline, no internet needed)
    Queries past road_events from SQLite. When the vehicle approaches
    within ALERT_RADIUS_METRES of a previously detected hazard that is
    roughly ahead, emits a dashboard alert: "POTHOLE — 54m ahead".

Layer 2 — OpenStreetMap Road Surface (internet needed, gracefully skipped)
    Queries Overpass API for road surface tags (bad/unpaved/dirt) within
    LOOK_AHEAD_METRES ahead of current position. Warns driver about rough
    road surface on roads they've never driven.

Usage
-----
    from modules.road.proximity_alert import ProximityAlertManager

    prox = ProximityAlertManager(db_manager, cfg=cfg)
    prox.start()

    # in your main loop:
    prox.update_gps(gps_fix)

    # read latest alert:
    alert = prox.latest_alert   # None or dict
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
import requests
from loguru import logger


# ── Config defaults ──────────────────────────────────────────────────────────
ALERT_RADIUS_METRES  = 80     # warn when hazard is within this distance
LOOK_AHEAD_DEGREES   = 70     # ±70° of vehicle heading counts as "ahead"
CHECK_INTERVAL_S     = 0.5    # how often to check proximity (seconds)
ALERT_COOLDOWN_S     = 4.0    # min seconds between same-hazard alerts
ALERT_DISPLAY_S      = 6.0    # how long alert stays visible on dashboard
OSM_RADIUS_M         = 150    # OSM query radius ahead of vehicle
OSM_TIMEOUT_S        = 5.0    # max wait for OSM response
OSM_COOLDOWN_S       = 30.0   # only query OSM every N seconds


# ── Maths helpers ─────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two GPS coordinates."""
    R = 6_371_000  # Earth radius metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_to(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return bearing in degrees (0=North, 90=East) from point 1 to point 2."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    x  = math.sin(dλ) * math.cos(φ2)
    y  = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a: float, b: float) -> float:
    """Smallest angle between two bearings (0–180°)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def is_ahead(vehicle_heading: float, hazard_bearing: float,
             tolerance: float = LOOK_AHEAD_DEGREES) -> bool:
    """Return True if hazard bearing is within ±tolerance of vehicle heading."""
    return angle_diff(vehicle_heading, hazard_bearing) <= tolerance


# ── Alert dataclass ───────────────────────────────────────────────────────────

@dataclass
class ProximityAlert:
    source:       str   = ""       # "history" | "osm"
    hazard_type:  str   = ""       # "pothole" | "crack_longitudinal" | "unpaved" …
    distance_m:   float = 0.0      # metres to hazard
    severity:     str   = ""       # "minor" | "moderate" | "severe" | "unknown"
    message:      str   = ""       # human-readable alert string
    expires_at:   float = 0.0      # time.time() when alert should clear
    lat:          Optional[float] = None
    lon:          Optional[float] = None


# ── OSM Layer 2 ───────────────────────────────────────────────────────────────

_BAD_SURFACES = {"bad", "very_bad", "unpaved", "dirt", "gravel", "ground",
                 "sand", "mud", "earth", "grass"}

def _query_osm_road_surface(lat: float, lon: float, radius_m: int = OSM_RADIUS_M) -> Optional[dict]:
    """
    Query Overpass API for road surface conditions near `lat`, `lon`.
    Returns a dict with hazard info or None if road is fine / unreachable.
    """
    query = f"""
    [out:json][timeout:{int(OSM_TIMEOUT_S)}];
    (
      way["highway"]["surface"~"bad|very_bad|unpaved|dirt|gravel|ground|sand|mud|earth"]
        (around:{radius_m},{lat},{lon});
      way["highway"]["smoothness"~"bad|very_bad|horrible|very_horrible|impassable"]
        (around:{radius_m},{lat},{lon});
    );
    out tags 1;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=OSM_TIMEOUT_S,
        )
        if resp.status_code != 200:
            return None
        elements = resp.json().get("elements", [])
        if not elements:
            return None
        tags  = elements[0].get("tags", {})
        surf  = tags.get("surface", "")
        smooth = tags.get("smoothness", "")
        name  = tags.get("name", tags.get("ref", "road ahead"))
        label = surf or smooth or "rough surface"
        return {
            "hazard_type": f"road surface: {label}",
            "severity":    "moderate" if smooth in {"bad","very_bad"} else "minor",
            "detail":      f"{name} — surface: {surf or 'unknown'}, smoothness: {smooth or 'unknown'}",
        }
    except Exception as e:
        logger.debug(f"[Proximity] OSM query failed: {e}")
        return None


# ── Main class ────────────────────────────────────────────────────────────────

class ProximityAlertManager:
    """
    Runs a background thread that checks proximity to known hazards
    and optionally queries OpenStreetMap for road surface conditions.

    Parameters
    ----------
    db_manager : DatabaseManager   — to query past road_events
    cfg        : dict              — full config dict (reads proximity section)
    """

    def __init__(self, db_manager, cfg: dict = None, voice_alert=None):
        self._db          = db_manager
        cfg               = cfg or {}
        prox_cfg          = cfg.get("proximity_alert", {})

        self.radius_m     = prox_cfg.get("radius_metres",   ALERT_RADIUS_METRES)
        self.ahead_deg    = prox_cfg.get("ahead_degrees",   LOOK_AHEAD_DEGREES)
        self.cooldown_s   = prox_cfg.get("cooldown_seconds", ALERT_COOLDOWN_S)
        self.display_s    = prox_cfg.get("display_seconds",  ALERT_DISPLAY_S)
        self.osm_enabled  = prox_cfg.get("osm_enabled",     True)
        self.osm_cooldown = prox_cfg.get("osm_cooldown_s",  OSM_COOLDOWN_S)
        self.min_speed    = prox_cfg.get("min_speed_kmh",   3.0)  # don't alert if parked

        self._lock        = threading.Lock()
        self._stop        = threading.Event()
        self._current_gps = None          # latest GPSFix

        self._active_alert: Optional[ProximityAlert] = None
        self._alerted_ids: dict = {}      # hazard_id → last alert timestamp
        self._last_osm_ts = 0.0
        self._voice = voice_alert

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        t = threading.Thread(target=self._run, daemon=True, name="ProximityAlert")
        t.start()
        logger.info(f"[Proximity] Alert monitor started  "
                    f"(radius={self.radius_m}m, OSM={'on' if self.osm_enabled else 'off'})")

    def stop(self):
        self._stop.set()

    # ── GPS feed ───────────────────────────────────────────────────────────

    def update_gps(self, gps_fix):
        """Call this every time you get a new GPS fix (from your main loop)."""
        with self._lock:
            self._current_gps = gps_fix

    # ── Background loop ────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            try:
                self._check()
            except Exception as e:
                logger.debug(f"[Proximity] Check error: {e}")
            time.sleep(CHECK_INTERVAL_S)

    def _check(self):
        with self._lock:
            gps = self._current_gps

        # Need a valid GPS fix
        if gps is None or not gps.valid or gps.lat is None or gps.lon is None:
            return

        # Don't alert if vehicle is essentially parked
        if gps.speed_kmh < self.min_speed:
            return

        now = time.time()

        # ── Clear expired alert ─────────────────────────────────────────
        with self._lock:
            if self._active_alert and now > self._active_alert.expires_at:
                self._active_alert = None

        # ── Layer 1: Local history from SQLite ──────────────────────────
        history_alert = self._check_history(gps, now)
        if history_alert:
            with self._lock:
                self._active_alert = history_alert
            return

        # ── Layer 2: OSM road surface (only if no history alert) ────────
        if self.osm_enabled and (now - self._last_osm_ts) >= self.osm_cooldown:
            self._last_osm_ts = now
            osm_result = _query_osm_road_surface(gps.lat, gps.lon)
            if osm_result:
                alert = ProximityAlert(
                    source      = "osm",
                    hazard_type = osm_result["hazard_type"],
                    distance_m  = 0,   # OSM doesn't give precise distance
                    severity    = osm_result["severity"],
                    message     = f"⚠ Road Ahead: {osm_result['hazard_type'].upper()}",
                    expires_at  = now + self.display_s,
                )
                with self._lock:
                    self._active_alert = alert
                logger.info(f"[Proximity] OSM road alert: {osm_result['detail']}")

    def _check_history(self, gps, now: float) -> Optional[ProximityAlert]:
        """
        Layer 1: Check all stored road events within ALERT_RADIUS_METRES.
        Returns the closest valid ahead-hazard alert, or None.
        """
        try:
            # Fetch recent hazards from DB (last 1000 events)
            events = self._db.query_recent_road_events(1000)
        except Exception as e:
            logger.debug(f"[Proximity] DB query failed: {e}")
            return None

        best: Optional[ProximityAlert] = None
        best_dist = float("inf")

        for ev in events:
            lat = ev.get("lat")
            lon = ev.get("lon")
            if lat is None or lon is None:
                continue

            dist = haversine_m(gps.lat, gps.lon, lat, lon)
            if dist > self.radius_m:
                continue

            # Check if hazard is ahead of us
            bearing = bearing_to(gps.lat, gps.lon, lat, lon)
            if not is_ahead(gps.heading, bearing, self.ahead_deg):
                continue

            # Cooldown: skip if we just alerted about this hazard
            ev_id = ev.get("id") or f"{lat:.5f},{lon:.5f}"
            last_alerted = self._alerted_ids.get(ev_id, 0)
            if (now - last_alerted) < self.cooldown_s:
                continue

            if dist < best_dist:
                best_dist = dist
                raw_name   = ev.get("class_name", "hazard")
                # Map integer class IDs to names (old DB events store ints)
                _id_map = {"0": "pothole", "1": "crack_longitudinal",
                           "2": "crack_transverse", "3": "rutting", "4": "repair"}
                class_name = _id_map.get(str(raw_name), str(raw_name))
                severity   = ev.get("severity",   "unknown")
                dist_int   = int(round(dist / 5) * 5)   # round to nearest 5m

                # Build message
                icons = {
                    "pothole":            "🕳",
                    "crack_longitudinal": "🔱",
                    "crack_transverse":   "⚡",
                    "rutting":            "〰",
                    "repair":             "🔧",
                }
                icon = icons.get(class_name, "⚠")
                msg  = f"{icon} {class_name.replace('_',' ').upper()} — {dist_int}m ahead"

                best = ProximityAlert(
                    source      = "history",
                    hazard_type = class_name,
                    distance_m  = dist,
                    severity    = severity,
                    message     = msg,
                    expires_at  = now + self.display_s,
                    lat         = lat,
                    lon         = lon,
                )
                self._alerted_ids[ev_id] = now

        if best:
            logger.info(f"[Proximity] {best.message}")
            if self._voice is not None and best.distance_m > 5:
                dist_rounded = max(10, int(round(best.distance_m / 10) * 10))
                name = best.hazard_type.replace("_"," ")
                self._voice.speak(
                    f"Caution! {name} {dist_rounded} metres ahead.",
                    cooldown=self.cooldown_s
                )
        return best

    # ── Public read ────────────────────────────────────────────────────────

    @property
    def latest_alert(self) -> Optional[dict]:
        """
        Returns None or a dict:
        {
            "source":      "history" | "osm",
            "hazard_type": "pothole",
            "distance_m":  54.0,
            "severity":    "severe",
            "message":     "🕳 POTHOLE — 55m ahead",
        }
        """
        with self._lock:
            a = self._active_alert
        if a is None or time.time() > a.expires_at:
            return None
        return {
            "source":      a.source,
            "hazard_type": a.hazard_type,
            "distance_m":  round(a.distance_m, 1),
            "severity":    a.severity,
            "message":     a.message,
        }