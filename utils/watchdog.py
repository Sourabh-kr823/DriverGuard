"""
utils/watchdog.py
──────────────────
Thread watchdog — monitors registered threads and restarts them
if they die unexpectedly.

Each watched thread must call `watchdog.heartbeat(name)` regularly.
If a thread misses heartbeats for longer than `timeout_sec`, the
watchdog calls the registered restart_fn and logs the event.

Usage
─────
    from utils.watchdog import Watchdog

    wd = Watchdog(timeout_sec=10)
    wd.register("DMS", restart_fn=start_dms_thread)
    wd.start()

    # Inside the DMS thread loop:
    wd.heartbeat("DMS")
"""

import threading
import time
from typing import Callable, Dict, Optional
from dataclasses import dataclass, field

from loguru import logger


# ─── Heartbeat record ─────────────────────────────────────────────────────────

@dataclass
class _WatchEntry:
    name:        str
    restart_fn:  Optional[Callable] = None
    timeout_sec: float = 10.0
    last_beat:   float = field(default_factory=time.time)
    restart_count: int = 0
    max_restarts:  int = 5


# ─── Watchdog ─────────────────────────────────────────────────────────────────

class Watchdog:
    """
    Parameters
    ----------
    timeout_sec     Default seconds without heartbeat before restart.
    check_interval  How often the watchdog polls (seconds).
    """

    def __init__(self, timeout_sec: float = 10.0, check_interval: float = 2.0):
        self.default_timeout = timeout_sec
        self.check_interval  = check_interval
        self._entries: Dict[str, _WatchEntry] = {}
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        name:         str,
        restart_fn:   Optional[Callable] = None,
        timeout_sec:  Optional[float]    = None,
        max_restarts: int                = 5,
    ):
        """
        Register a thread to watch.

        Parameters
        ----------
        name         Unique thread name (same string used in heartbeat calls).
        restart_fn   Callable that re-launches the thread (called on timeout).
        timeout_sec  Override the default timeout for this thread.
        max_restarts Stop restarting after this many attempts.
        """
        with self._lock:
            self._entries[name] = _WatchEntry(
                name         = name,
                restart_fn   = restart_fn,
                timeout_sec  = timeout_sec or self.default_timeout,
                last_beat    = time.time(),
                max_restarts = max_restarts,
            )
        logger.debug(f"[Watchdog] Registered '{name}' "
                     f"(timeout={timeout_sec or self.default_timeout}s)")

    def unregister(self, name: str):
        with self._lock:
            self._entries.pop(name, None)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def heartbeat(self, name: str):
        """
        Called by a watched thread to signal it is alive.
        Should be called at least once per timeout_sec inside the thread loop.
        """
        with self._lock:
            if name in self._entries:
                self._entries[name].last_beat = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="Watchdog")
        self._thread.start()
        logger.info("[Watchdog] Started")

    def stop(self):
        self._stop.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _watch_loop(self):
        while not self._stop.is_set():
            time.sleep(self.check_interval)
            now = time.time()
            with self._lock:
                entries = list(self._entries.values())

            for e in entries:
                elapsed = now - e.last_beat
                if elapsed > e.timeout_sec:
                    logger.warning(
                        f"[Watchdog] '{e.name}' missed heartbeat "
                        f"for {elapsed:.1f}s (timeout={e.timeout_sec}s)"
                    )
                    if e.restart_fn and e.restart_count < e.max_restarts:
                        e.restart_count += 1
                        logger.warning(
                            f"[Watchdog] Restarting '{e.name}' "
                            f"(attempt {e.restart_count}/{e.max_restarts})"
                        )
                        try:
                            e.restart_fn()
                            e.last_beat = time.time()   # Reset timer
                        except Exception as ex:
                            logger.error(
                                f"[Watchdog] Restart of '{e.name}' failed: {ex}")
                    elif e.restart_count >= e.max_restarts:
                        logger.error(
                            f"[Watchdog] '{e.name}' exceeded max restarts "
                            f"({e.max_restarts}) — giving up. "
                            "Manual intervention required."
                        )
                        with self._lock:
                            self._entries.pop(e.name, None)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return health status of all watched threads."""
        now = time.time()
        with self._lock:
            return {
                name: {
                    "last_heartbeat_sec_ago": round(now - e.last_beat, 1),
                    "restart_count":          e.restart_count,
                    "healthy":                (now - e.last_beat) < e.timeout_sec,
                }
                for name, e in self._entries.items()
            }
