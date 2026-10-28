"""
backend/dms/mp_face_mesh.py
────────────────────────────
Drop-in replacement for mp.solutions.face_mesh.FaceMesh built on the
MediaPipe Tasks FaceLandmarker.

MediaPipe >= 0.10.30 (the only builds for Python 3.13 / aarch64, i.e. the
Raspberry Pi) removed the legacy `mp.solutions` API.  This adapter exposes the
same `.process(rgb)` → `.multi_face_landmarks[i].landmark[j].x/.y` shape, so
ear_mar.py and head_pose.py work unchanged (same 468-point mesh topology).

Requires models/face_landmarker.task:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import time
from types import SimpleNamespace

DEFAULT_MODEL_PATH = "models/face_landmarker.task"


class TasksFaceMesh:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, rgb):
        # VIDEO mode requires strictly increasing timestamps
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        image  = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        faces  = [SimpleNamespace(landmark=lms) for lms in result.face_landmarks]
        return SimpleNamespace(multi_face_landmarks=faces or None)

    def close(self):
        self._landmarker.close()
