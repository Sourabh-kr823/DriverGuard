"""
scripts/export_ncnn.py
One-time: convert the trained .pt road model to NCNN (ARM-optimised, CPU).

NCNN runs YOLO fast on the Raspberry Pi's CPU with no accuracy loss. The
exported files are platform-independent, so you can run this on your LAPTOP
(where Ultralytics already works) and copy the folder to the Pi.

Run from the repo root:
    python scripts/export_ncnn.py
    python scripts/export_ncnn.py --model models/yolov8n_rdd_india.pt --imgsz 640

Produces:  models/yolov8n_rdd_india_ncnn_model/
Then set  road.model_path  in config.yaml (on the Pi) to that folder.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Export the road model to NCNN")
    ap.add_argument("--model", default=str(ROOT / "models" / "yolov8n_rdd_india.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        return

    from ultralytics import YOLO
    print(f"Exporting {args.model}  ->  NCNN  (imgsz={args.imgsz}) …")
    out = YOLO(args.model).export(format="ncnn", imgsz=args.imgsz)

    print(f"\nDone. NCNN model folder:\n   {out}")
    print("\nNext:")
    print("  1. Copy that folder to the Pi (or commit it to the repo).")
    print("  2. On the Pi, set in config.yaml:")
    print("       road:")
    print("         model_path: \"models/yolov8n_rdd_india_ncnn_model\"")
    print("  3. On the Pi: pip install ncnn")


if __name__ == "__main__":
    main()
