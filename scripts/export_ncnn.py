"""
Export YOLOv8 .pt model to NCNN format for fast ARM inference.

Usage
─────
    python scripts/export_ncnn.py                # default 320x320
    python scripts/export_ncnn.py --imgsz 640    # export at 640x640

Ultralytics requires the folder to end with _ncnn_model.
When exporting at a non-default size, the folder is named:
    models/yolov8n_rdd_india_640_ncnn_model
so both 320 and 640 exports can coexist.
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to NCNN")
    parser.add_argument("--model", default="models/yolov8n_rdd_india.pt",
                        help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Input image size for the exported model (default 320)")
    args = parser.parse_args()

    model_path = args.model
    imgsz = args.imgsz
    stem = Path(model_path).stem

    print(f"Loading {model_path} ...")
    model = YOLO(model_path)

    # Ultralytics names the output after the .pt stem, so for non-default
    # sizes export from a temp copy named <stem>_<size>.pt. This yields
    # <stem>_<size>_ncnn_model without touching the default 320 folder.
    tmp_pt = None
    if imgsz != 320:
        tmp_pt = Path(model_path).with_name(f"{stem}_{imgsz}.pt")
        shutil.copy2(model_path, tmp_pt)
        sized_path = tmp_pt.with_name(f"{tmp_pt.stem}_ncnn_model")
        if sized_path.exists():
            shutil.rmtree(sized_path)
        model = YOLO(str(tmp_pt))

    print(f"Exporting to NCNN @ imgsz={imgsz} ...")
    try:
        out = model.export(format="ncnn", imgsz=imgsz, half=False)
    finally:
        if tmp_pt is not None:
            tmp_pt.unlink(missing_ok=True)
    out_path = Path(out)  # e.g. models/yolov8n_rdd_india_ncnn_model

    # Save metadata so evaluate.py knows the export size
    meta = {"imgsz": imgsz, "source": model_path, "stem": stem}
    (out_path / "export_meta.json").write_text(json.dumps(meta))

    print(f"Done → {out_path}")
    print(f"\nTo use in config.yaml:")
    print(f'  model_path: "{out_path}"')
    if imgsz != 320:
        print(f"\nTo compare 320 vs {imgsz}:")
        print(f"  python evaluate.py --compare 320 {imgsz}")


if __name__ == "__main__":
    main()
