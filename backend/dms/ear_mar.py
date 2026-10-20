"""
backend/dms/ear_mar.py
──────────────────────
Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) computation.

EAR formula (Soukupová & Čech, 2016):
    EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 · ‖p1-p4‖)
    6 eye landmarks: p1=left-corner, p4=right-corner,
                     p2,p3=upper-lid, p5,p6=lower-lid
    Range: ~0.25–0.40 (open), <0.20 (closed)

MAR formula (6-point EAR-style, same geometry):
    MAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 · ‖p1-p4‖)
    6 lip landmarks: p1=left-corner, p2=upper-left, p3=upper-right,
                     p4=right-corner, p5=lower-right, p6=lower-left
    Range: ~0.05–0.15 (closed mouth), ~0.50–0.80 (yawn)

Fix history:
    v2.2 — Replaced bogus 26-point MP_MOUTH list (had duplicate indices 61 and
            291, used wrong points for the denominator giving MAR > 1.0 for a
            closed mouth) with correct 6-point definitions for both backends.
            Updated _mouth_aspect_ratio() to proper vertical / width formula.
"""

import numpy as np
from scipy.spatial import distance as dist


# ─── Dlib 68-point landmark indices ──────────────────────────────────────────
# Left eye:  36-41    Right eye: 42-47
DLIB_LEFT_EYE  = list(range(36, 42))
DLIB_RIGHT_EYE = list(range(42, 48))

# 6 outer lip points: left-corner, upper-left, upper-right,
#                     right-corner, lower-right, lower-left
DLIB_MOUTH_6 = [48, 50, 52, 54, 56, 58]


# ─── MediaPipe 468-point landmark indices ────────────────────────────────────
# Reference: mediapipe/python/solutions/face_mesh_connections.py
MP_LEFT_EYE  = [362, 385, 387, 263, 373, 380]   # p1…p6 order
MP_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# 6 outer lip points (same semantic order as DLIB_MOUTH_6):
#   p1=left-corner(61), p2=upper-left(82), p3=upper-right(312),
#   p4=right-corner(291), p5=lower-right(317), p6=lower-left(87)
MP_MOUTH_6 = [61, 82, 312, 291, 317, 87]


# ─── Core geometry ───────────────────────────────────────────────────────────

def _eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    EAR for a single eye given 6 (x,y) landmark points [p1..p6].
    Returns float typically 0.25–0.40 (open), <0.20 (closed).
    """
    A = dist.euclidean(eye[1], eye[5])   # upper-inner <-> lower-inner
    B = dist.euclidean(eye[2], eye[4])   # upper-outer <-> lower-outer
    C = dist.euclidean(eye[0], eye[3])   # left-corner <-> right-corner (width)
    return (A + B) / (2.0 * C + 1e-6)


def _mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """
    MAR for 6 mouth landmark points.
    Ordered [left-corner, upper-left, upper-right, right-corner, lower-right, lower-left].
    Returns float: ~0.05-0.15 closed, ~0.50-0.80 yawn.
    """
    A = dist.euclidean(mouth[1], mouth[5])   # upper-left  <-> lower-left  (vertical)
    B = dist.euclidean(mouth[2], mouth[4])   # upper-right <-> lower-right (vertical)
    C = dist.euclidean(mouth[0], mouth[3])   # left-corner <-> right-corner (width)
    return (A + B) / (2.0 * C + 1e-6)


# ─── Dlib ────────────────────────────────────────────────────────────────────

def compute_ear_dlib(landmarks) -> float:
    """EAR from a dlib 68-point shape object (averaged left + right)."""
    def pts(indices):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y)
                         for i in indices], dtype=np.float64)
    return (_eye_aspect_ratio(pts(DLIB_LEFT_EYE)) +
            _eye_aspect_ratio(pts(DLIB_RIGHT_EYE))) / 2.0


def compute_mar_dlib(landmarks) -> float:
    """MAR from a dlib 68-point shape object using 6 outer lip points."""
    mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                      for i in DLIB_MOUTH_6], dtype=np.float64)
    return _mouth_aspect_ratio(mouth)


# ─── MediaPipe ───────────────────────────────────────────────────────────────

def compute_ear_mediapipe(face_landmarks, img_w: int, img_h: int) -> float:
    """EAR from a MediaPipe NormalizedLandmarkList (averaged left + right)."""
    def pts(indices):
        return np.array([
            (face_landmarks.landmark[i].x * img_w,
             face_landmarks.landmark[i].y * img_h)
            for i in indices
        ], dtype=np.float64)
    return (_eye_aspect_ratio(pts(MP_LEFT_EYE)) +
            _eye_aspect_ratio(pts(MP_RIGHT_EYE))) / 2.0


def compute_mar_mediapipe(face_landmarks, img_w: int, img_h: int) -> float:
    """MAR from a MediaPipe NormalizedLandmarkList using 6 outer lip points."""
    mouth = np.array([
        (face_landmarks.landmark[i].x * img_w,
         face_landmarks.landmark[i].y * img_h)
        for i in MP_MOUTH_6
    ], dtype=np.float64)
    return _mouth_aspect_ratio(mouth)


# ─── Public API ──────────────────────────────────────────────────────────────

def compute_ear_mar(landmarks, backend: str, img_w: int = 0, img_h: int = 0):
    """
    Unified EAR + MAR dispatcher.

    Parameters
    ----------
    landmarks : dlib full_object_detection | mediapipe NormalizedLandmarkList
    backend   : "dlib" | "mediapipe"
    img_w, img_h : required when backend == "mediapipe"

    Returns
    -------
    tuple[float, float]  (ear, mar)
    """
    if backend == "dlib":
        return compute_ear_dlib(landmarks), compute_mar_dlib(landmarks)
    elif backend == "mediapipe":
        return (compute_ear_mediapipe(landmarks, img_w, img_h),
                compute_mar_mediapipe(landmarks, img_w, img_h))
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'dlib' or 'mediapipe'.")