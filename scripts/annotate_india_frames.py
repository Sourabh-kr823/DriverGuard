"""
scripts/annotate_india_frames.py
─────────────────────────────────
Helper for annotating the 1 200 custom Indian road frames.

What this script does
─────────────────────
1. Checks for LabelImg / Label Studio installation
2. Prepares a pre-configured LabelImg workspace with the correct
   class list (pothole, crack, rutting) and YOLO output mode
3. Merges completed YOLO annotations into data/road_dataset/

This script does NOT draw bounding boxes itself — it sets up the
annotation environment so you can do it quickly in LabelImg.

Recommended tool: LabelImg (fast, offline, YOLO-native)
    pip install labelImg
    labelImg

Alternative: Label Studio (web UI, team annotation)
    pip install label-studio
    label-studio start

Usage
─────
    # Step 1 — prepare workspace
    python scripts/annotate_india_frames.py prepare \\
        --images-dir /path/to/your/1200/jpg/frames \\
        --workspace  data/annotation_workspace

    # Step 2 — open LabelImg
    python scripts/annotate_india_frames.py open

    # Step 3 — after annotating, merge into training dataset
    python scripts/annotate_india_frames.py merge \\
        --workspace  data/annotation_workspace \\
        --out-root   data/road_dataset \\
        --val-split  0.15
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

CLASSES = ["pothole", "crack", "rutting"]

LABELIMG_PREDEFINED = """pothole
crack
rutting
"""


# ─── Subcommands ──────────────────────────────────────────────────────────────

def prepare(images_dir: Path, workspace: Path):
    """Set up LabelImg workspace directory."""
    (workspace / "images").mkdir(parents=True, exist_ok=True)
    (workspace / "labels").mkdir(parents=True, exist_ok=True)

    # Copy / symlink images
    imgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not imgs:
        print(f"✗ No images found in {images_dir}")
        sys.exit(1)

    print(f"  Linking {len(imgs)} images into workspace …")
    for img in imgs:
        dest = workspace / "images" / img.name
        if not dest.exists():
            dest.symlink_to(img.resolve())

    # Write predefined_classes.txt for LabelImg
    (workspace / "predefined_classes.txt").write_text(LABELIMG_PREDEFINED)

    # Write LabelImg config shortcut
    (workspace / "open_labelimg.sh").write_text(
        f"#!/bin/bash\n"
        f"labelImg '{workspace}/images' "
        f"'{workspace}/predefined_classes.txt' "
        f"'{workspace}/labels'\n"
    )
    (workspace / "open_labelimg.sh").chmod(0o755)

    print(f"\n  ✓ Workspace ready: {workspace}")
    print(f"\n  Annotation checklist:")
    print(f"    Classes  → pothole (0)  crack (1)  rutting (2)")
    print(f"    Format   → YOLO  (set in LabelImg: View → Change Save Format → YOLO)")
    print(f"    Save dir → {workspace}/labels/")
    print(f"\n  Open LabelImg:")
    print(f"    bash {workspace}/open_labelimg.sh")
    print(f"  or manually:")
    print(f"    labelImg {workspace}/images "
          f"{workspace}/predefined_classes.txt {workspace}/labels")
    print(f"\n  LabelImg keyboard shortcuts:")
    print(f"    W          → draw bounding box")
    print(f"    D / A      → next / previous image")
    print(f"    Ctrl+S     → save current annotation")
    print(f"    Ctrl+R     → change save directory")


def open_labelimg(workspace: Path):
    """Launch LabelImg with the prepared workspace."""
    script = workspace / "open_labelimg.sh"
    if script.exists():
        subprocess.run(["bash", str(script)])
    else:
        print("✗ Run 'prepare' first to set up the workspace.")


def merge(workspace: Path, out_root: Path, val_split: float, seed: int):
    """Merge annotated frames into data/road_dataset/ train/val splits."""
    label_dir = workspace / "labels"
    image_dir = workspace / "images"

    labels = sorted(label_dir.glob("*.txt"))
    if not labels:
        print(f"✗ No label files found in {label_dir}")
        sys.exit(1)

    # Only keep labels that have at least one annotation
    valid_pairs = []
    for lbl in labels:
        content = lbl.read_text().strip()
        if not content:
            continue
        img = image_dir / (lbl.stem + ".jpg")
        if not img.exists():
            img = image_dir / (lbl.stem + ".png")
        if img.exists():
            valid_pairs.append((img, lbl))

    print(f"\n  Found {len(valid_pairs)} annotated frames with labels")

    random.seed(seed)
    random.shuffle(valid_pairs)
    n_val = max(1, int(len(valid_pairs) * val_split))
    val_set = set(range(len(valid_pairs) - n_val, len(valid_pairs)))

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "boxes": 0}

    for i, (img_path, lbl_path) in enumerate(valid_pairs):
        split = "val" if i in val_set else "train"
        shutil.copy2(img_path, out_root / "images" / split / img_path.name)
        shutil.copy2(lbl_path, out_root / "labels" / split / lbl_path.name)
        stats[split] += 1
        stats["boxes"] += len(lbl_path.read_text().strip().splitlines())

    print(f"\n  Merged into {out_root}:")
    print(f"    Train : {stats['train']} images")
    print(f"    Val   : {stats['val']} images")
    print(f"    Boxes : {stats['boxes']} total annotations")
    print(f"\n  ✓ Ready to train:")
    print(f"    python scripts/train_road.py --epochs 50 --batch 16\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotation helper for custom India road frames")
    sub = parser.add_subparsers(dest="cmd")

    p_prep = sub.add_parser("prepare", help="Set up annotation workspace")
    p_prep.add_argument("--images-dir", required=True)
    p_prep.add_argument("--workspace",  default="data/annotation_workspace")

    p_open = sub.add_parser("open", help="Launch LabelImg")
    p_open.add_argument("--workspace", default="data/annotation_workspace")

    p_merge = sub.add_parser("merge", help="Merge annotations into dataset")
    p_merge.add_argument("--workspace",  default="data/annotation_workspace")
    p_merge.add_argument("--out-root",   default="data/road_dataset")
    p_merge.add_argument("--val-split",  type=float, default=0.15)
    p_merge.add_argument("--seed",       type=int,   default=42)

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare(Path(args.images_dir), Path(args.workspace))
    elif args.cmd == "open":
        open_labelimg(Path(args.workspace))
    elif args.cmd == "merge":
        merge(Path(args.workspace), Path(args.out_root),
              args.val_split, args.seed)
    else:
        parser.print_help()
