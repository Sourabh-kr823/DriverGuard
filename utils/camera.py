"""
utils/camera.py
────────────────
Opens a camera source and returns an object with the cv2.VideoCapture
interface (isOpened / read / release).

Supported sources
    "picamera" / "picamera:N"  → Raspberry Pi CSI camera N via picamera2
    /dev/videoN, /dev/v4l/...  → V4L2 device (USB webcam)
    int                        → OpenCV device index (V4L2 on Linux)
    anything else              → passed to cv2.VideoCapture (file / RTSP URL)
"""
import os
import re
import sys

import cv2
from loguru import logger


class PiCamera:
    """Minimal cv2.VideoCapture-compatible wrapper around picamera2."""

    def __init__(self, index: int, width: int, height: int, fps: int):
        self._cam = None
        try:
            from picamera2 import Picamera2
            self._cam = Picamera2(index)
            # libcamera "RGB888" is stored B,G,R — exactly what OpenCV expects
            config = self._cam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            self._cam.configure(config)
            self._cam.start()
        except Exception as e:
            logger.warning(f"[Camera] picamera2 camera {index} failed: {e}")
            self.release()

    def isOpened(self) -> bool:
        return self._cam is not None

    def read(self):
        if self._cam is None:
            return False, None
        try:
            return True, self._cam.capture_array("main")
        except Exception as e:
            logger.debug(f"[Camera] picamera2 read error: {e}")
            return False, None

    def release(self):
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass
        self._cam = None


def open_capture(source, width: int = 640, height: int = 480, fps: int = 30):
    """Open `source` and apply the requested resolution / frame rate."""
    if isinstance(source, str) and source.startswith("picamera"):
        _, _, idx = source.partition(":")
        return PiCamera(int(idx or 0), width, height, fps)

    if isinstance(source, str) and source.startswith("/dev/"):
        # pip OpenCV can't open V4L2 devices by name — resolve the
        # /dev/v4l/by-id symlink to /dev/videoN and use the index
        node = os.path.realpath(source)
        m = re.fullmatch(r"/dev/video(\d+)", node)
        if m:
            source = int(m.group(1))
        else:
            logger.warning(f"[Camera] {source} not found (resolved to {node})")

    if isinstance(source, int) and sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        # MJPG lets USB webcams reach full frame rate over USB 2.0
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap
