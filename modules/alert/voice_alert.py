"""
modules/alert/voice_alert.py
─────────────────────────────
Offline text-to-speech using pyttsx3 (Windows SAPI5).
No internet required. Engine runs in its own thread.

Usage in main.py:
    from modules.alert.voice_alert import VoiceAlert
    voice = VoiceAlert(cfg=cfg)
    voice.start()
    # pass to alert_manager and proximity_alert
"""
import pyttsx3, threading, queue, time
from loguru import logger


class VoiceAlert:
    def __init__(self, cfg: dict = None):
        cfg    = cfg or {}
        va     = cfg.get("voice_alert", {})
        self._rate     = va.get("rate",     150)
        self._volume   = va.get("volume",   0.9)
        self._cooldown = va.get("cooldown", 8.0)
        self._enabled  = va.get("enabled",  True)
        self._q        = queue.Queue(maxsize=5)
        self._last_ts  : dict = {}

    def start(self):
        if not self._enabled:
            logger.info("[Voice] TTS disabled in config"); return
        threading.Thread(target=self._worker, daemon=True, name="VoiceAlert").start()
        logger.info(f"[Voice] TTS started (rate={self._rate} wpm, cooldown={self._cooldown}s)")

    def speak(self, text: str, cooldown: float = None):
        """Queue a voice alert — dropped if in cooldown or queue full."""
        if not self._enabled: return
        cd  = cooldown if cooldown is not None else self._cooldown
        now = time.time()
        if now - self._last_ts.get(text, 0) < cd: return
        self._last_ts[text] = now
        try:   self._q.put_nowait(text)
        except queue.Full: pass

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            logger.info("[Voice] TTS engine initialised")
        except Exception as e:
            logger.error(f"[Voice] Engine init failed: {e}"); return
        while True:
            text = self._q.get()
            try:   engine.say(text); engine.runAndWait()
            except Exception as e: logger.debug(f"[Voice] TTS error: {e}")
