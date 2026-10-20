"""
check_system.py
───────────────
Pre-flight check for DriverGuard.
Run before main.py to verify all components are working.

Usage:
    python check_system.py
    python check_system.py --config config.yaml
"""

import argparse
import importlib
import os
import sys
import time

import yaml

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"
INFO = "\033[94m  INFO\033[0m"

results = []

def check(label, passed, detail="", warn=False):
    tag  = WARN if warn else (PASS if passed else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    results.append(passed or warn)
    return passed


def section(title):
    print(f"\n\033[1m{'─'*50}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'─'*50}\033[0m")


# ── Load config ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

print("\n\033[1m  DriverGuard — System Check\033[0m")
print(f"  Config: {args.config}\n")

try:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"{PASS}  Config loaded: {args.config}")
except Exception as e:
    print(f"{FAIL}  Config load failed: {e}")
    sys.exit(1)

# ── 1. Python & Dependencies ──────────────────────────────────────────────────
section("1. Python & Dependencies")

py_ver = sys.version_info
check("Python version",
      py_ver >= (3, 10),
      f"{py_ver.major}.{py_ver.minor}.{py_ver.micro} {'(OK)' if py_ver>=(3,10) else '(needs 3.10+)'}")

packages = [
    ("cv2",          "opencv-python"),
    ("mediapipe",    "mediapipe"),
    ("ultralytics",  "ultralytics"),
    ("flask",        "flask"),
    ("flask_socketio","flask-socketio"),
    ("loguru",       "loguru"),
    ("yaml",         "pyyaml"),
    ("scipy",        "scipy"),
    ("numpy",        "numpy"),
    ("pyttsx3",      "pyttsx3"),
    ("dotenv",       "python-dotenv"),
    ("requests",     "requests"),
]

for module, pkg in packages:
    try:
        importlib.import_module(module)
        check(f"Package: {pkg}", True)
    except ImportError:
        check(f"Package: {pkg}", False, f"pip install {pkg}")

# ── 2. Model Files ────────────────────────────────────────────────────────────
section("2. Model Files")

model_path = cfg.get("road", {}).get("model_path", "models/yolov8n_rdd_india.pt")
model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
check("Road model (yolov8n_rdd_india.pt)",
      os.path.exists(model_path),
      f"{model_size/1e6:.1f} MB" if model_size else "NOT FOUND")

dlib_path = cfg.get("dms", {}).get("dlib_model_path",
            "models/shape_predictor_68_face_landmarks.dat")
dlib_size = os.path.getsize(dlib_path) if os.path.exists(dlib_path) else 0
backend   = cfg.get("dms", {}).get("backend", "mediapipe")
if backend == "mediapipe":
    check("Dlib model (shape_predictor_68.dat)",
          True, "not needed — backend is mediapipe", warn=True)
else:
    check("Dlib model (shape_predictor_68.dat)",
          os.path.exists(dlib_path),
          f"{dlib_size/1e6:.1f} MB" if dlib_size else "NOT FOUND")

# ── 3. Cameras ────────────────────────────────────────────────────────────────
section("3. Cameras")

import cv2

def test_camera(source, name):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        check(f"Camera {source} ({name})", False, "not detected")
        return
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        check(f"Camera {source} ({name})", True, f"{w}x{h} frame captured")
    else:
        check(f"Camera {source} ({name})", False, "opened but no frame")

dms_src  = cfg.get("cameras", {}).get("dms",  {}).get("source", 1)
road_src = cfg.get("cameras", {}).get("road", {}).get("source", 0)
test_camera(dms_src,  "DMS / driver-facing")
test_camera(road_src, "Road / forward-facing")

# ── 4. MediaPipe ─────────────────────────────────────────────────────────────
section("4. MediaPipe Face Detection")

try:
    import mediapipe as mp
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    check("MediaPipe FaceMesh init", True)

    # Try on a blank frame
    blank = __import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8)
    rgb   = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
    res   = mesh.process(rgb)
    check("MediaPipe inference test", True,
          "face detected" if res.multi_face_landmarks else "no face (blank frame — OK)")
    mesh.close()
except Exception as e:
    check("MediaPipe FaceMesh", False, str(e))

# ── 5. YOLOv8 Road Model ─────────────────────────────────────────────────────
section("5. YOLOv8 Road Model Inference")

try:
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(model_path)
    load_ms = int((time.time() - t0) * 1000)
    check("YOLOv8 model load", True, f"{load_ms}ms")

    import numpy as np
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    t0 = time.time()
    model(blank, verbose=False)
    inf_ms = int((time.time() - t0) * 1000)
    fps_est = round(1000 / inf_ms) if inf_ms > 0 else 0

    check("YOLOv8 inference test", True,
          f"{inf_ms}ms/frame  (~{fps_est} FPS road pipeline)")

    if inf_ms > 100:
        check("YOLOv8 speed",
              False, f"{inf_ms}ms is slow — GPU not used or Pi 5 expected", warn=True)
    else:
        check("YOLOv8 speed", True, "fast enough for real-time")
except Exception as e:
    check("YOLOv8", False, str(e))

# ── 6. Database ───────────────────────────────────────────────────────────────
section("6. SQLite Database")

import sqlite3
db_path = cfg.get("database", {}).get("path", "data/logs/driver_guard_events.db")
db_dir  = os.path.dirname(db_path)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

try:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS _test_check (id INTEGER PRIMARY KEY)")
    con.execute("DROP TABLE _test_check")
    con.commit()
    con.close()
    size_kb = round(os.path.getsize(db_path)/1024) if os.path.exists(db_path) else 0
    check("SQLite WAL mode", True, f"{db_path} ({size_kb} KB)")
except Exception as e:
    check("SQLite", False, str(e))

# ── 7. GPS ────────────────────────────────────────────────────────────────────
section("7. GPS Module")

simulate = cfg.get("gps", {}).get("simulation", True)
if simulate:
    check("GPS mode", True, "simulation mode — NEO-6M not needed", warn=True)
else:
    port = cfg.get("gps", {}).get("port", "COM3")
    try:
        import serial
        s = serial.Serial(port, timeout=1)
        s.close()
        check(f"GPS port {port}", True, "opened successfully")
    except Exception as e:
        check(f"GPS port {port}", False, str(e))

# ── 8. Voice TTS ─────────────────────────────────────────────────────────────
section("8. Voice Alerts (TTS)")

voice_enabled = cfg.get("voice_alert", {}).get("enabled", True)
if not voice_enabled:
    check("Voice alerts", True, "disabled in config", warn=True)
else:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        voice_names = [v.name for v in (voices or [])]
        has_zira  = any("Zira"  in n for n in voice_names)
        has_david = any("David" in n for n in voice_names)
        engine.stop()
        del engine
        check("pyttsx3 engine", True,
              f"{len(voice_names)} voices found")
        check("Female voice (Zira — driver alerts)",
              has_zira, "Microsoft Zira Desktop" if has_zira else "not found — will use default")
        check("Male voice (David — road alerts)",
              has_david, "Microsoft David Desktop" if has_david else "not found — will use default")
    except Exception as e:
        check("pyttsx3", False, str(e))

# ── 9. Environment / API Keys ─────────────────────────────────────────────────
section("9. Environment & API Keys")

from dotenv import load_dotenv
load_dotenv()

weather_key = os.environ.get("OPENWEATHER_API_KEY", "")
check(".env loaded", os.path.exists(".env"), ".env file found" if os.path.exists(".env") else "missing — create .env file")
check("OPENWEATHER_API_KEY",
      bool(weather_key),
      f"{'set (' + weather_key[:4] + '...)'}" if weather_key else "not set — weather widget will be disabled",
      warn=not bool(weather_key))

# ── 10. Config values ─────────────────────────────────────────────────────────
section("10. Key Config Values")

ear_c  = cfg.get("dms", {}).get("ear", {}).get("consec_frames", 0)
mar_c  = cfg.get("dms", {}).get("mar", {}).get("consec_frames", 0)
head_c = cfg.get("dms", {}).get("head_pose", {}).get("consec_frames", 0)
iou    = cfg.get("road", {}).get("iou_threshold", 0.45)
radius = cfg.get("proximity_alert", {}).get("radius_metres", 80)

check("EAR consec_frames", ear_c == 20, f"{ear_c} frames (expected 20)")
check("MAR consec_frames", mar_c == 12, f"{mar_c} frames (expected 12)")
check("Head consec_frames", head_c == 8, f"{head_c} frames (expected 8)")
check("Road IOU threshold", iou <= 0.40, f"{iou} (≤0.40 = tighter NMS)")
check("Proximity radius", radius == 80, f"{radius}m")
check("DMS backend", backend == "mediapipe",
      f"{backend} ({'fast' if backend=='mediapipe' else 'slow — consider mediapipe'})")

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")

total   = len(results)
passed  = sum(results)
failed  = total - passed

if failed == 0:
    print(f"\n\033[92m  ✓  All {total} checks passed — system ready to run\033[0m")
    print("  Run:  python main.py --simulate --preview\n")
else:
    print(f"\n\033[91m  ✗  {failed} check(s) failed — fix above issues before running\033[0m\n")
    sys.exit(1)
