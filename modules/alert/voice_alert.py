"""
modules/alert/voice_alert.py
─────────────────────────────
Offline TTS for Windows using pyttsx3 + PowerShell fallback.

FIX: Engine is recreated for every message — this is the only
     reliable way to use pyttsx3 on Windows from a background thread
     (SAPI5 COM objects are not thread-persistent).
"""
import threading
import queue
import time
import subprocess
from loguru import logger

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False
    logger.warning("[Voice] pyttsx3 not installed — using PowerShell TTS only")


class VoiceAlert:
    def __init__(self, cfg: dict = None):
        cfg            = cfg or {}
        va             = cfg.get("voice_alert", {})
        self._rate     = va.get("rate",     150)
        self._volume   = va.get("volume",   0.9)
        self._cooldown = va.get("cooldown", 8.0)
        self._enabled  = va.get("enabled",  True)
        self._q        = queue.Queue(maxsize=8)
        self._last_ts  : dict = {}

    def start(self):
        if not self._enabled:
            logger.info("[Voice] TTS disabled in config")
            return
        t = threading.Thread(target=self._worker, daemon=True, name="VoiceAlert")
        t.start()
        logger.info(f"[Voice] TTS started  (rate={self._rate}, cooldown={self._cooldown}s)")

    def speak(self, text: str, cooldown: float = None):
        """Queue a voice alert — dropped if in cooldown or queue full."""
        if not self._enabled:
            return
        cd  = cooldown if cooldown is not None else self._cooldown
        now = time.time()
        if now - self._last_ts.get(text, 0) < cd:
            return
        self._last_ts[text] = now
        try:
            self._q.put_nowait(text)
            logger.debug(f"[Voice] Queued: {text!r}")
        except queue.Full:
            logger.debug("[Voice] Queue full — alert dropped")

    def _worker(self):
        """Background thread — speaks each queued message in order."""
        logger.info("[Voice] Worker thread started")
        while True:
            text = self._q.get()
            self._speak_once(text)

    def _speak_once(self, text: str):
        """
        Speak one message. Tries pyttsx3 first, falls back to PowerShell.
        Engine is created fresh every call — required for Windows SAPI5.
        """
        # ── Method 1: pyttsx3 (preferred) ────────────────────────────────
        if _PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate",   self._rate)
                engine.setProperty("volume", self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
                logger.debug(f"[Voice] Spoke (pyttsx3): {text!r}")
                return
            except Exception as e:
                logger.debug(f"[Voice] pyttsx3 failed: {e} — trying PowerShell")

        # ── Method 2: PowerShell SAPI5 (Windows fallback) ─────────────────
        try:
            safe = text.replace('"', "'").replace(';', ',')
            rate = max(-10, min(10, int((self._rate - 150) / 15)))
            subprocess.run(
                [
                    "powershell", "-WindowStyle", "Hidden", "-NonInteractive",
                    "-Command",
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$s.Rate = {rate}; '
                    f'$s.Volume = {int(self._volume * 100)}; '
                    f'$s.Speak("{safe}")',
                ],
                timeout=15,
                capture_output=True,
            )
            logger.debug(f"[Voice] Spoke (PowerShell): {text!r}")
        except Exception as e:
            logger.debug(f"[Voice] PowerShell TTS failed: {e}")