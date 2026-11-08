"""
Export a YOLO .pt model to NCNN format for fast ARM inference.

Usage
─────
    python scripts/export_ncnn.py                # pothole_yolo11n.pt @ 416 (config.yaml road.imgsz)
    python scripts/export_ncnn.py --imgsz 320    # export at 320x320

Ultralytics requires the folder to end with _ncnn_model. The output is
always named with its size so several exports can coexist:
    models/pothole_yolo11n_416_ncnn_model
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def _default_imgsz() -> int:
    try:
        with open(ROOT / "config.yaml") as f:
            return int(yaml.safe_load(f)["road"].get("imgsz", 416))
    except Exception:
        return 416


def main():
    default_imgsz = _default_imgsz()
    parser = argparse.ArgumentParser(description="Export YOLO to NCNN")
    parser.add_argument("--model", default="models/pothole_yolo11n.pt",
                        help="Path to .pt model file (default: models/pothole_yolo11n.pt)")
    parser.add_argument("--imgsz", type=int, default=default_imgsz,
                        help=f"Input image size for the exported model (default {default_imgsz} "
                             f"from config.yaml)")
    args = parser.parse_args()

    model_path = args.model
    imgsz = args.imgsz
    stem = Path(model_path).stem

    # Ultralytics names the output after the .pt stem, so export from a temp
    # copy named <stem>_<size>.pt. This yields <stem>_<size>_ncnn_model.
    tmp_pt = Path(model_path).with_name(f"{stem}_{imgsz}.pt")
    shutil.copy2(model_path, tmp_pt)
    sized_path = tmp_pt.with_name(f"{tmp_pt.stem}_ncnn_model")
    if sized_path.exists():
        shutil.rmtree(sized_path)

    print(f"Loading {model_path} ...")
    model = YOLO(str(tmp_pt))
    print(f"Classes: {model.names}")

    print(f"Exporting to NCNN @ imgsz={imgsz} ...")
    try:
        out = model.export(format="ncnn", imgsz=imgsz, half=False)
    finally:
        tmp_pt.unlink(missing_ok=True)
    out_path = Path(out)  # e.g. models/pothole_yolo11n_416_ncnn_model

    # Save metadata so evaluate.py knows the export size
    meta = {"imgsz": imgsz, "source": model_path, "stem": stem}
    (out_path / "export_meta.json").write_text(json.dumps(meta))

    print(f"Done → {out_path}")
    print(f"\nTo use in config.yaml:")
    print(f'  model_path: "{out_path}"')
    print(f"  imgsz: {imgsz}")


if __name__ == "__main__":
    main()
