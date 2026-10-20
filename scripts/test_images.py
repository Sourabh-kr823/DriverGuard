"""
test_images.py
──────────────
Run road damage detection on a folder of images or a single image.
Use this to demo the model to your guide without a road test.

Usage:
    python test_images.py --source path/to/image.jpg
    python test_images.py --source path/to/folder/
    python test_images.py --source 0          # webcam
    python test_images.py --source video.mp4  # video file

Controls:
    Q or ESC → quit
    S        → save current frame
    SPACE    → pause/resume (video only)
"""

import argparse
from unicodedata import name
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH  = "models/yolov8n_rdd_india.pt"
CONF_THRESH = 0.35
SAVE_DIR    = Path("data/test_results")

CLASS_COLORS = {
    "pothole"           : (0,   0,   255),   # Red
    "crack_longitudinal": (0,   0,   0  ),   # Black
    "crack_transverse"  : (0,   165, 255),   # Orange
    "rutting"           : (42,  42,  165),   # Brown
    "repair"            : (128, 0,   128),   # Purple
}

TEXT_COLORS = {
    "pothole"           : (0,   0,   0  ),   # black text on red
    "crack_longitudinal": (255, 255, 255),   # white text on black
    "crack_transverse"  : (0,   0,   0  ),   # black text on orange
    "rutting"           : (255, 255, 255),   # white text on brown
    "repair"            : (255, 255, 255),   # white text on purple
}


def draw_detections(frame, results):
    """Draw bounding boxes with class name and confidence."""
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = r.names[cls_id]
            color      = CLASS_COLORS.get(name, (255, 255, 255))
            text_color = TEXT_COLORS.get(name, (0, 0, 0))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return frame


def run_on_images(model, folder: Path):
    """Run detection on all images in a folder — press any key to advance."""
    imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + \
           list(folder.glob("*.jpeg"))
    if not imgs:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(imgs)} images. Press any key to advance, Q to quit.")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model(frame, conf=CONF_THRESH, verbose=False)
        frame   = draw_detections(frame, results)

        n_dets = sum(len(r.boxes) for r in results)
        cv2.putText(frame, f"[{i+1}/{len(imgs)}] {img_path.name} — {n_dets} detections",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

        cv2.imshow("DriverGuard — Road Detection Test", frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            out = SAVE_DIR / f"result_{img_path.stem}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"Saved → {out}")

    cv2.destroyAllWindows()


def run_on_video(model, source):
    """Run detection on video file or webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    paused  = False
    frame_n = 0
    print("Controls: Q=quit  S=save frame  SPACE=pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

            results = model(frame, conf=CONF_THRESH, verbose=False)
            frame   = draw_detections(frame, results)

            n_dets = sum(len(r.boxes) for r in results)
            cv2.putText(frame, f"Frame {frame_n} | Detections: {n_dets}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

        cv2.imshow("DriverGuard — Road Detection Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            out = SAVE_DIR / f"frame_{frame_n:05d}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"Saved → {out}")
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Image file, folder of images, video file, or 0 for webcam")
    parser.add_argument("--conf",   type=float, default=CONF_THRESH,
                        help=f"Confidence threshold (default: {CONF_THRESH})")
    args = parser.parse_args()

    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded ✓\n")

    source = args.source
    p = Path(source)

    if p.is_dir():
        run_on_images(model, p)
    elif p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        # Single image
        frame   = cv2.imread(str(p))
        results = model(frame, conf=args.conf, verbose=False)
        frame   = draw_detections(frame, results)
        n_dets  = sum(len(r.boxes) for r in results)
        print(f"Detections: {n_dets}")
        cv2.imshow("DriverGuard — Road Detection Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Video or webcam
        src = int(source) if source.isdigit() else source
        run_on_video(model, src)


if __name__ == "__main__":
    main()
