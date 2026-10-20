"""
scripts/calibrate_camera.py
────────────────────────────
Camera calibration using a printed checkerboard pattern.

Why calibrate?
──────────────
solvePnP (used in head_pose.py) relies on accurate camera intrinsics
(focal length, principal point) and distortion coefficients.  Without
calibration, head-pose yaw/pitch estimates can drift by 5–15°, making
the ±30° / ±20° thresholds unreliable.

Steps
─────
1. Print a checkerboard:
   - US Letter or A4 paper
   - Default: 9×6 inner corners (10×7 squares)
   - Measure the actual square size in mm — update --square-mm

2. Run this script with your DMS IR camera:
       python scripts/calibrate_camera.py --source 0

3. Hold the board at varying angles / distances.
   Press SPACE to capture (need at least 15 good frames).
   Press Q to finish and compute calibration.

4. Calibration is saved to:
       models/camera_dms_calib.npz  (for DMS camera)
       models/camera_road_calib.npz (for road camera, pass --source 1)

5. head_pose.py will auto-load the calibration if the file exists.

Usage
─────
    python scripts/calibrate_camera.py --source 0 --name dms
    python scripts/calibrate_camera.py --source 1 --name road --cols 9 --rows 6
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def calibrate(source, name: str, cols: int, rows: int,
              square_mm: float, min_frames: int, out_dir: Path):
    """Run interactive calibration session."""

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3-D object points for one board position
    obj_pts_single = np.zeros((cols * rows, 3), np.float32)
    obj_pts_single[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_pts_single *= square_mm

    obj_points = []   # 3-D points in real world
    img_points = []   # 2-D points in image plane

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"✗ Cannot open camera {source}")
        return

    print(f"\n── Camera Calibration ── {name} camera (source={source})")
    print(f"  Board: {cols}×{rows} inner corners, {square_mm} mm squares")
    print(f"  SPACE → capture  |  Q → finish & compute\n")

    captured = 0
    frame_h, frame_w = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        display = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (cols, rows),
                                      corners_refined, found)
            cv2.putText(display, "Board detected — SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 230, 100), 2)
        else:
            cv2.putText(display, "No board detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 80, 220), 2)

        cv2.putText(display, f"Captured: {captured}/{min_frames}",
                    (10, frame_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 0), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            obj_points.append(obj_pts_single)
            img_points.append(corners_refined)
            captured += 1
            print(f"  Captured frame {captured}")
            # Flash green feedback
            flash = np.zeros_like(frame)
            flash[:] = (0, 200, 80)
            cv2.addWeighted(frame, 0.5, flash, 0.5, 0, display)
            cv2.imshow("Calibration", display)
            cv2.waitKey(300)
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured < min_frames:
        print(f"\n✗ Not enough frames ({captured} < {min_frames}). Aborting.")
        return

    print(f"\n  Computing calibration from {captured} frames …")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points,
        (frame_w, frame_h), None, None)

    # Reprojection error (lower is better — aim for < 1.0 px)
    total_err = 0.0
    for i, obj in enumerate(obj_points):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], mtx, dist)
        total_err += cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total_err / len(obj_points)

    print(f"\n  ── Results ──────────────────────────────")
    print(f"  RMS reprojection error : {ret:.4f} px")
    print(f"  Mean reprojection error: {mean_err:.4f} px")
    print(f"  Camera matrix:\n{mtx}")
    print(f"  Distortion coeffs: {dist.ravel()}")

    if mean_err > 2.0:
        print("\n  ⚠  High reprojection error — consider recapturing with")
        print("     better board coverage (angles, distances, corners).")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"camera_{name}_calib.npz"
    np.savez(str(out_path),
             camera_matrix=mtx,
             dist_coeffs=dist,
             img_size=(frame_w, frame_h),
             rms_error=np.array([ret]),
             mean_error=np.array([mean_err]))
    print(f"\n  ✓ Saved calibration to: {out_path}")
    print(f"\n  head_pose.py will auto-load this file on next run.\n")


def load_calibration(name: str, models_dir: Path = Path("models")):
    """
    Load a previously saved calibration.
    Returns (camera_matrix, dist_coeffs) or (None, None) if not found.
    """
    path = models_dir / f"camera_{name}_calib.npz"
    if not path.exists():
        return None, None
    data = np.load(str(path))
    return data["camera_matrix"], data["dist_coeffs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration for VSM")
    parser.add_argument("--source",    type=int,   default=0,
                        help="Camera device index (0=DMS, 1=Road)")
    parser.add_argument("--name",      type=str,   default="dms",
                        help="Camera name suffix for output file")
    parser.add_argument("--cols",      type=int,   default=9,
                        help="Inner corners horizontally")
    parser.add_argument("--rows",      type=int,   default=6,
                        help="Inner corners vertically")
    parser.add_argument("--square-mm", type=float, default=25.0,
                        help="Physical size of each square in mm")
    parser.add_argument("--min-frames",type=int,   default=15,
                        help="Minimum captures before computing")
    parser.add_argument("--out-dir",   type=str,   default="models",
                        help="Output directory for .npz calibration file")
    args = parser.parse_args()

    calibrate(
        source     = args.source,
        name       = args.name,
        cols       = args.cols,
        rows       = args.rows,
        square_mm  = args.square_mm,
        min_frames = args.min_frames,
        out_dir    = Path(args.out_dir),
    )
