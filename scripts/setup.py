"""
scripts/setup.py
─────────────────
One-shot setup script.

Steps
-----
1. Create directory layout
2. Download dlib shape_predictor_68_face_landmarks.dat
3. Download YOLOv8-nano base weights (fine-tune separately on RDD2022)
4. Create placeholder alert audio
5. Create dummy SQLite DB to verify write permissions

Run
---
    python scripts/setup.py
"""

import os
import sys
import urllib.request
import bz2
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = [
    "models", "data/logs", "assets",
    "backend/dms", "backend/road",
    "backend/gps", "backend/database",
    "backend/alert", "dashboard", "utils",
]

DLIB_URL  = ("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
YOLO_URL  = ("https://github.com/ultralytics/assets/releases/download/"
             "v0.0.0/yolov8n.pt")


def make_dirs():
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        (ROOT / d / "__init__.py").touch()
    print("✓ Directories created")


def download_dlib():
    dest_bz2 = ROOT / "models" / "shape_predictor_68_face_landmarks.dat.bz2"
    dest_dat = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
    if dest_dat.exists():
        print("✓ Dlib model already present — skipping")
        return
    print("  Downloading dlib shape predictor …")
    urllib.request.urlretrieve(DLIB_URL, dest_bz2,
        reporthook=lambda b, bs, ts: print(
            f"\r  {min(100, int(b*bs/ts*100))}%", end="", flush=True))
    print()
    with bz2.open(dest_bz2) as src, open(dest_dat, "wb") as dst:
        shutil.copyfileobj(src, dst)
    dest_bz2.unlink()
    print("✓ Dlib model ready")


def download_yolo():
    dest = ROOT / "models" / "yolov8n.pt"
    rdd_dest = ROOT / "models" / "yolov8n_rdd_india.pt"
    if rdd_dest.exists():
        print("✓ YOLOv8 RDD model already present — skipping")
        return
    if not dest.exists():
        print("  Downloading YOLOv8-nano base weights …")
        urllib.request.urlretrieve(YOLO_URL, dest,
            reporthook=lambda b, bs, ts: print(
                f"\r  {min(100, int(b*bs/ts*100))}%", end="", flush=True))
        print()
        print("✓ YOLOv8-nano base weights downloaded")

    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  NEXT STEP: Fine-tune on RDD2022 + your 1 200 India frames  │")
    print("  │                                                              │")
    print("  │  See scripts/train_road.py for training commands.           │")
    print("  │  Then copy the best.pt to:                                  │")
    print("  │    models/yolov8n_rdd_india.pt                              │")
    print("  └─────────────────────────────────────────────────────────────┘")
    # For now, point config to base weights for testing
    shutil.copy(dest, rdd_dest)
    print("  (Copied base weights as placeholder — replace with fine-tuned)")


def create_audio():
    wav = ROOT / "assets" / "alert.wav"
    if wav.exists():
        return
    # Write minimal silent WAV (44 bytes)
    # fmt: 1 = PCM, 1 ch, 8000 Hz, 8-bit — 0.1 s of silence
    header = bytes([
        0x52,0x49,0x46,0x46, 0x24,0x03,0x00,0x00,  # RIFF + size
        0x57,0x41,0x56,0x45,                         # WAVE
        0x66,0x6D,0x74,0x20, 0x10,0x00,0x00,0x00,  # fmt  + size=16
        0x01,0x00,                                   # PCM
        0x01,0x00,                                   # 1 channel
        0x40,0x1F,0x00,0x00,                         # 8000 Hz
        0x40,0x1F,0x00,0x00,                         # byte rate
        0x01,0x00, 0x08,0x00,                        # block align, bits/sample
        0x64,0x61,0x74,0x61, 0x00,0x03,0x00,0x00,  # data + size=768
    ])
    payload = bytes([128] * 768)  # silence
    wav.write_bytes(header + payload)
    print("✓ Placeholder alert.wav written (replace with real alert audio)")


def verify_db():
    import sqlite3
    db_path = ROOT / "data" / "logs" / "vsm_events.db"
    try:
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE IF NOT EXISTS _test (x INTEGER)")
        con.execute("DROP TABLE _test")
        con.close()
        print(f"✓ SQLite writable at {db_path}")
    except Exception as e:
        print(f"✗ SQLite error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n── VSM Setup ─────────────────────────────────────")
    make_dirs()
    download_dlib()
    download_yolo()
    create_audio()
    verify_db()
    print("\n✓ Setup complete — run:  python main.py --simulate --preview\n")
