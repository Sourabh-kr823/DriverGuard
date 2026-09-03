"""
modules/alert/voice_alert.py
─────────────────────────────
Dual-channel offline TTS:
  • Driver channel  → Female voice (Microsoft Zira) — driver monitoring alerts
  • Road channel    → Male voice   (Microsoft David) — road/pothole alerts
  • Severe tier     → Faster rate + louder + emphasis on words

pyttsx3 engine is re-created per message (required for Windows SAPI5 threading).
PowerShell fallback if pyttsx3 fails.
"""
import threading
import queue
import time
import subprocess
from loguru import logger

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False
    logger.warning("[Voice] pyttsx3 not found — using PowerShell TTS only")


# ── Default Windows voice names ───────────────────────────────────────────────
VOICE_FEMALE = "Microsoft Zira Desktop"    # Driver alerts (higher pitch naturally)
VOICE_MALE   = "Microsoft David Desktop"   # Road/pothole alerts


class VoiceAlert:
    """
    Two-channel voice alert system.

    Usage:
        voice = VoiceAlert(cfg=cfg)
        voice.start()

        voice.speak_driver("Drowsiness detected", severe=False)
        voice.speak_driver("SEVERE ALERT! Pull over!", severe=True)
        voice.speak_road("Pothole 70 metres ahead")
    """

    def __init__(self, cfg: dict = None):
        cfg    = cfg or {}
        va     = cfg.get("voice_alert", {})

        # Normal settings
        self._rate_normal   = va.get("rate_normal",   148)
        self._rate_severe   = va.get("rate_severe",   205)  # faster = more urgent
        self._vol_normal    = va.get("volume_normal",  0.9)
        self._vol_severe    = va.get("volume_severe",  1.0)
        self._cd_driver     = va.get("cooldown_driver", 10.0)
        self._cd_road       = va.get("cooldown_road",    6.0)
        self._cd_severe     = va.get("cooldown_severe",  7.0)
        self._enabled       = va.get("enabled", True)

        # Separate queues per channel so road and driver don't block each other
        self._q_driver = queue.Queue(maxsize=4)
        self._q_road   = queue.Queue(maxsize=4)

        # Cooldown tracker: key → last spoken timestamp
        self._last_ts: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self._enabled:
            logger.info("[Voice] TTS disabled in config"); return
        threading.Thread(target=self._worker, args=(self._q_driver, "driver"),
                         daemon=True, name="Voice-Driver").start()
        threading.Thread(target=self._worker, args=(self._q_road, "road"),
                         daemon=True, name="Voice-Road").start()
        logger.info("[Voice] Dual-channel TTS started "
                    f"(normal={self._rate_normal}wpm, severe={self._rate_severe}wpm)")

    # ── Public API ────────────────────────────────────────────────────────────

    def speak_driver(self, text: str, severe: bool = False):
        """
        Driver monitoring alert.
        severe=True  → fast rate, full volume, female voice (sounds urgent)
        severe=False → normal rate, female voice
        """
        cd = self._cd_severe if severe else self._cd_driver
        self._enqueue(self._q_driver, text, cd, severe)

    def speak_road(self, text: str):
        """Road/pothole proximity alert — male voice, normal rate."""
        self._enqueue(self._q_road, text, self._cd_road, False)

    def speak(self, text: str, cooldown: float = None):
        """Generic fallback — routes to driver channel."""
        self._enqueue(self._q_driver, text, cooldown or self._cd_driver, False)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enqueue(self, q: queue.Queue, text: str, cooldown: float, severe: bool):
        if not self._enabled: return
        key = text
        now = time.time()
        if now - self._last_ts.get(key, 0) < cooldown: return
        self._last_ts[key] = now
        try:
            q.put_nowait((text, severe))
            logger.debug(f"[Voice] Queued ({'severe' if severe else 'normal'}): {text!r}")
        except queue.Full:
            logger.debug("[Voice] Queue full — dropped")

    def _worker(self, q: queue.Queue, channel: str):
        logger.info(f"[Voice] {channel} worker started")
        while True:
            text, severe = q.get()
            rate   = self._rate_severe   if severe else self._rate_normal
            volume = self._vol_severe    if severe else self._vol_normal
            voice  = VOICE_FEMALE        if channel == "driver" else VOICE_MALE
            self._speak_once(text, rate, volume, voice)

    def _speak_once(self, text: str, rate: int, volume: float, voice_name: str):
        """Speak one message — tries pyttsx3 first, falls back to PowerShell."""

        # ── pyttsx3 ───────────────────────────────────────────────────────────
        if _HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate",   rate)
                engine.setProperty("volume", volume)
                # Select voice by name (partial match)
                voices = engine.getProperty("voices")
                for v in voices:
                    if voice_name.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
                logger.debug(f"[Voice] Spoke (pyttsx3 {voice_name}): {text!r}")
                return
            except Exception as e:
                logger.debug(f"[Voice] pyttsx3 error: {e}")

        # ── PowerShell fallback ───────────────────────────────────────────────
        try:
            safe = text.replace('"', "'").replace(';', ',').replace('!', ',')
            ps_rate = max(-10, min(10, int((rate - 150) / 15)))
            ps_vol  = int(volume * 100)
            # Select voice by name in PowerShell
            ps_cmd = (
                f'Add-Type -AssemblyName System.Speech; '
                f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$s.Rate = {ps_rate}; $s.Volume = {ps_vol}; '
                f'try {{ $s.SelectVoice("{voice_name}") }} catch {{}}; '
                f'$s.Speak("{safe}")'
            )
            subprocess.run(
                ["powershell", "-WindowStyle", "Hidden", "-NonInteractive", "-Command", ps_cmd],
                timeout=15, capture_output=True,
            )
            logger.debug(f"[Voice] Spoke (PowerShell {voice_name}): {text!r}")
        except Exception as e:
            logger.debug(f"[Voice] PowerShell error: {e}")