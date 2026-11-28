

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Dataset YAML (written at runtime) ────────────────────────────────────────

DATASET_YAML = """\
path: {data_root}
train: images/train
val:   images/val

nc: 3
names:
  0: pothole
  1: crack
  2: rutting
"""


def prepare_dataset(data_root: Path):
    for split in ("train", "val"):
        (data_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    yaml_path = data_root / "dataset.yaml"
    yaml_path.write_text(DATASET_YAML.format(data_root=str(data_root)))
    print(f"Dataset YAML written: {yaml_path}")
    return yaml_path


def train(epochs, batch, imgsz, device):
    from ultralytics import YOLO

    data_root = ROOT / "data" / "road_dataset"
    yaml_path = prepare_dataset(data_root)

    print(f"\nImages in train: {len(list((data_root/'images'/'train').glob('*.*')))}")
    print(f"Images in val:   {len(list((data_root/'images'/'val').glob('*.*')))}\n")

    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))

    results = model.train(
        data     = str(yaml_path),
        epochs   = epochs,
        batch    = batch,
        imgsz    = imgsz,
        device   = device,
        project  = str(ROOT / "runs" / "road"),
        name     = "rdd_india",
        patience = 20,
        save     = True,
        plots    = True,
        # Augmentation — useful for varied Indian road conditions
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        fliplr   = 0.5,
        mosaic   = 1.0,
        degrees  = 5.0,
    )

    # Copy best weights to models/
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = ROOT / "models" / "yolov8n_rdd_india.pt"
    if best.exists():
        shutil.copy(best, dest)
        print(f"\n✓ Best weights saved to: {dest}")
    else:
        print(f"\n⚠ best.pt not found at {best} — check training output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=50)
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--device", type=str,   default="0",
                        help="CUDA device index or 'cpu'")
    args = parser.parse_args()

    print(f"Training YOLOv8-nano | epochs={args.epochs} "
          f"batch={args.batch} imgsz={args.imgsz} device={args.device}")
    train(args.epochs, args.batch, args.imgsz, args.device)
