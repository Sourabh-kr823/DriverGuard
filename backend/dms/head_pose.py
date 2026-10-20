"""
backend/dms/head_pose.py
────────────────────────
Head pose estimation (Yaw, Pitch, Roll) using solvePnP.

Uses 6 canonical 3-D face model points (nose tip, chin, eye corners,
mouth corners) that correspond to dlib landmark indices.  The same
mapping is used whether landmarks come from dlib or MediaPipe.

Algorithm
---------
1. Extract 2-D image points from detected landmarks.
2. Solve the perspective-n-point (PnP) problem with cv2.solvePnP.
3. Convert the rotation vector → Euler angles (Yaw, Pitch, Roll).

References
----------
- Gudi et al. (2015) – Real-time estimation of head pose
- OpenCV solvePnP docs
"""

import cv2
import numpy as np


# ─── Canonical 3-D face model (metres, face-centred coords) ──────────────────
# Points: nose-tip, chin, left-eye-corner, right-eye-corner,
#         left-mouth-corner, right-mouth-corner
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,     0.0),    # Nose tip
    (0.0,   -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0),   # Left eye outer corner
    (225.0,  170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0,  -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# Corresponding dlib 68-point indices
DLIB_INDICES = [30, 8, 36, 45, 48, 54]

# Corresponding MediaPipe 468-point indices
MP_INDICES   = [1, 152, 226, 446, 57, 287]


def _build_camera_matrix(img_h: int, img_w: int) -> np.ndarray:
    """Estimate camera intrinsics assuming no distortion."""
    focal = img_w
    cx, cy = img_w / 2, img_h / 2
    return np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ],
    ], dtype=np.float64)


def _rotation_vec_to_euler(rvec: np.ndarray):
    """
    Convert OpenCV rotation vector → (yaw, pitch, roll) in degrees.

    Convention:
        yaw   > 0  → face turned right
        pitch > 0  → face tilted up
        roll  > 0  → face rolled counter-clockwise
    """
    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose with QR-like approach
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = 0.0
    return float(yaw), float(pitch), float(roll)


# ─── Public API ──────────────────────────────────────────────────────────────

def estimate_head_pose_dlib(landmarks, img_h: int, img_w: int):
    """
    Compute head pose from dlib 68-point landmarks.

    Returns
    -------
    tuple[float, float, float]  (yaw, pitch, roll) in degrees
                                 or (0, 0, 0) on failure.
    """
    image_pts = np.array([
        (landmarks.part(i).x, landmarks.part(i).y)
        for i in DLIB_INDICES
    ], dtype=np.float64)

    cam_matrix = _build_camera_matrix(img_h, img_w)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_pts,
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0
    return _rotation_vec_to_euler(rvec)


def estimate_head_pose_mediapipe(face_landmarks, img_h: int, img_w: int):
    """
    Compute head pose from MediaPipe 468-point NormalizedLandmarkList.

    Returns
    -------
    tuple[float, float, float]  (yaw, pitch, roll) in degrees
    """
    image_pts = np.array([
        (face_landmarks.landmark[i].x * img_w,
         face_landmarks.landmark[i].y * img_h)
        for i in MP_INDICES
    ], dtype=np.float64)

    cam_matrix = _build_camera_matrix(img_h, img_w)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_pts,
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0
    return _rotation_vec_to_euler(rvec)


def draw_head_pose_axes(frame: np.ndarray, landmarks, backend: str,
                        yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Overlay small 3-D axes on the frame to visualise head orientation.
    """
    h, w = frame.shape[:2]
    if backend == "dlib":
        nose = (landmarks.part(30).x, landmarks.part(30).y)
    else:  # mediapipe
        lm = landmarks.landmark[1]
        nose = (int(lm.x * w), int(lm.y * h))

    length = 50
    yaw_r, pitch_r, roll_r = np.radians(yaw), np.radians(pitch), np.radians(roll)

    # Simplified 2-D projection of axes
    x_end = (int(nose[0] + length * np.cos(yaw_r)),
              int(nose[1] - length * np.sin(pitch_r)))
    y_end = (int(nose[0] - length * np.sin(roll_r)),
              int(nose[1] - length * np.cos(roll_r)))

    cv2.arrowedLine(frame, nose, x_end, (0, 0, 255),   2, tipLength=0.3)  # Red  = Yaw
    cv2.arrowedLine(frame, nose, y_end, (0, 255, 0),   2, tipLength=0.3)  # Green = Pitch
    cv2.putText(frame, f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    return frame
