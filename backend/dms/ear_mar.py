
import numpy as np
from scipy.spatial import distance as dist

DLIB_LEFT_EYE  = list(range(36, 42))
DLIB_RIGHT_EYE = list(range(42, 48))

DLIB_MOUTH_6 = [48, 50, 52, 54, 56, 58]

MP_LEFT_EYE  = [362, 385, 387, 263, 373, 380]   # p1…p6 order
MP_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

MP_MOUTH_6 = [61, 82, 312, 291, 317, 87]


# ─── Core geometry ───────────────────────────────────────────────────────────

def _eye_aspect_ratio(eye: np.ndarray) -> float:
    A = dist.euclidean(eye[1], eye[5])   # upper-inner <-> lower-inner
    B = dist.euclidean(eye[2], eye[4])   # upper-outer <-> lower-outer
    C = dist.euclidean(eye[0], eye[3])   # left-corner <-> right-corner (width)
    return (A + B) / (2.0 * C + 1e-6)


def _mouth_aspect_ratio(mouth: np.ndarray) -> float:
    A = dist.euclidean(mouth[1], mouth[5])   # upper-left  <-> lower-left  (vertical)
    B = dist.euclidean(mouth[2], mouth[4])   # upper-right <-> lower-right (vertical)
    C = dist.euclidean(mouth[0], mouth[3])   # left-corner <-> right-corner (width)
    return (A + B) / (2.0 * C + 1e-6)


# ─── Dlib ────────────────────────────────────────────────────────────────────

def compute_ear_dlib(landmarks) -> float:
    def pts(indices):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y)
                         for i in indices], dtype=np.float64)
    return (_eye_aspect_ratio(pts(DLIB_LEFT_EYE)) +
            _eye_aspect_ratio(pts(DLIB_RIGHT_EYE))) / 2.0


def compute_mar_dlib(landmarks) -> float:
    mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                      for i in DLIB_MOUTH_6], dtype=np.float64)
    return _mouth_aspect_ratio(mouth)


# ─── MediaPipe ───────────────────────────────────────────────────────────────

def compute_ear_mediapipe(face_landmarks, img_w: int, img_h: int) -> float:
    def pts(indices):
        return np.array([
            (face_landmarks.landmark[i].x * img_w,
             face_landmarks.landmark[i].y * img_h)
            for i in indices
        ], dtype=np.float64)
    return (_eye_aspect_ratio(pts(MP_LEFT_EYE)) +
            _eye_aspect_ratio(pts(MP_RIGHT_EYE))) / 2.0


def compute_mar_mediapipe(face_landmarks, img_w: int, img_h: int) -> float:
    mouth = np.array([
        (face_landmarks.landmark[i].x * img_w,
         face_landmarks.landmark[i].y * img_h)
        for i in MP_MOUTH_6
    ], dtype=np.float64)
    return _mouth_aspect_ratio(mouth)


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_ear_mar(landmarks, backend: str, img_w: int = 0, img_h: int = 0):
    if backend == "dlib":
        return compute_ear_dlib(landmarks), compute_mar_dlib(landmarks)
    elif backend == "mediapipe":
        return (compute_ear_mediapipe(landmarks, img_w, img_h),
                compute_mar_mediapipe(landmarks, img_w, img_h))
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'dlib' or 'mediapipe'.")