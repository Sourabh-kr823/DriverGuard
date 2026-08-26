"""
merge_datasets.py
──────────────────────────────────────────────────────────────────
Merges downloaded road damage datasets into existing
D:/We4/train/ folder, remapping class IDs to match your
5-class DriverGuard structure.

Your target classes:
    0: pothole
    1: crack_longitudinal
    2: crack_transverse
    3: rutting
    4: repair

Usage
─────
    python merge_datasets.py --verify   # check mappings first
    python merge_datasets.py            # run full merge
"""

import argparse
import shutil
import os
from pathlib import Path

# ─── Your existing training folder ───────────────────────────────────────────
TRAIN_IMAGES = Path("D:/We4/train/images")
TRAIN_LABELS = Path("D:/We4/train/labels")

# ─── Dataset configurations ──────────────────────────────────────────────────
# PATHS FIXED — pointing to where check_datasets.py extracted them
# DATASET 3 FIXED — uses labels-YOLO/ instead of labels/ (confirmed from output)

DATASET_CONFIGS = [

    # ── Dataset 1: Indian Roads Dataset ──────────────────────────────────────
    # Classes: pothole(0), speed_breaker(1), unpaved_road(2)
    {
        "name"      : "Indian Roads Dataset",
        "path"      : Path("D:/We4/datasets/indian_road/Dataset3Class"),  # FIXED PATH
        "remap" : {
                0: -1,   # speed_breaker → skip
                1: 0,    # pothole → pothole ✓
                2: -1,   # unpaved_road → skip
            },
        "split"     : "",       # images at path/images/
        "label_dir" : "",
    },

    # ── Dataset 2: Potholes YOLOv8 Dataset ───────────────────────────────────
    # Classes: pothole(0) only — nc=1 confirmed
    {
        "name"      : "Potholes YOLOv8 Dataset",
        "path"      : Path("D:/We4/datasets/yv8"),  # FIXED PATH
        "remap"     : {
            0: 0,    # pothole → pothole ✓
        },
        "split"     : "train",  # images at path/train/images/
        "label_dir" : "labels",
    },

    # ── Dataset 3: Potholes + Cracks + Manholes ───────────────────────────────
    # Classes confirmed from COCO json:
    #   ID 0 = pothole, ID 1 = crack, ID 2 = manhole
    # Labels are in labels-YOLO/ (NOT labels/) — confirmed from output
    {
        "name"      : "Potholes+Cracks+Manholes",
        "path"      : Path("D:/We4/datasets/potholes_and_manhole"),  # FIXED PATH
        "remap"     : {
            0: 0,    # pothole → pothole ✓
            1: 1,    # crack   → crack_longitudinal ✓
            2: -1,   # manhole → skip
        },
        "split"     : "data",        # images at path/data/images/
        "label_dir" : "labels-YOLO", # FIXED — use labels-YOLO not labels
    },
]


# ─── Merge logic ─────────────────────────────────────────────────────────────

def find_images_and_labels(dataset_path: Path, split: str, label_dir: str):
    """Find image and label directories inside dataset folder."""

    # Build base path
    base = dataset_path / split if split else dataset_path

    # Image directory
    img_dir = base / "images"
    if not img_dir.exists():
        img_dir = base  # images directly in base (flat structure)

    # Label directory
    if label_dir == "":
        # Flat structure — labels are in same folder as images
        lbl_dir = img_dir
    else:
        lbl_dir = base / label_dir
        if not lbl_dir.exists():
            lbl_dir = base / "labels"  # fallback

    if not img_dir.exists():
        print(f"   ✗ Images not found at: {img_dir}")
        return None, None

    return img_dir, lbl_dir


def remap_label_file(src_label: Path, remap: dict) -> list:
    """
    Read a YOLO label file and remap class IDs.
    Returns remapped lines. Lines with class -1 are discarded.
    """
    if not src_label.exists():
        return []

    remapped = []
    with open(src_label, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                orig_cls = int(parts[0])
            except ValueError:
                continue
            new_cls = remap.get(orig_cls, -1)
            if new_cls == -1:
                continue
            parts[0] = str(new_cls)
            remapped.append(" ".join(parts))
    return remapped


def verify_dataset(config: dict):
    """Print class ID distribution to confirm remapping is correct."""
    path      = config["path"]
    split     = config["split"]
    label_dir = config["label_dir"]
    remap     = config["remap"]

    if not path.exists():
        print(f"   ✗ Not found: {path}")
        return

    base = path / split if split else path
    if label_dir == "":
        lbl = base  # flat structure
    else:
        lbl = base / label_dir
        if not lbl.exists():
            lbl = base / "labels"

    class_counts = {}
    files = list(lbl.rglob("*.txt"))[:200]
    for f in files:
        try:
            with open(f) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        c = int(parts[0])
                        class_counts[c] = class_counts.get(c, 0) + 1
        except Exception:
            pass

    print(f"   Labels dir: {lbl}")
    print(f"   Class mapping:")
    TARGET = {0:"pothole",1:"crack_longitudinal",2:"crack_transverse",3:"rutting",4:"repair"}
    for cls_id, count in sorted(class_counts.items()):
        mapped = remap.get(cls_id, -1)
        mapped_name = TARGET.get(mapped, "SKIP") if mapped != -1 else "SKIP"
        arrow = "✓" if mapped != -1 else "✗ skip"
        print(f"     Class {cls_id}: {count:5d} annotations → {mapped_name} {arrow}")


def merge_dataset(config: dict, existing_images: set):
    """Copy images + remapped labels to training folder."""
    name      = config["name"]
    path      = config["path"]
    remap     = config["remap"]
    split     = config["split"]
    label_dir = config["label_dir"]

    print(f"\n── {name} ──────────────────────────────────")

    if not path.exists():
        print(f"   ✗ Folder not found: {path}")
        return 0, 0

    img_dir, lbl_dir = find_images_and_labels(path, split, label_dir)
    if img_dir is None:
        return 0, 0

    images = list(img_dir.glob("*.jpg")) + \
             list(img_dir.glob("*.png")) + \
             list(img_dir.glob("*.jpeg"))
    print(f"   Images found : {len(images)}")
    print(f"   Labels dir   : {lbl_dir}")

    copied = skipped = empty = 0

    for img_path in images:
        stem = img_path.stem

        if stem in existing_images:
            skipped += 1
            continue

        lbl_path = lbl_dir / (stem + ".txt") \
                   if lbl_dir and lbl_dir.exists() \
                   else Path("__none__")

        remapped_lines = remap_label_file(lbl_path, remap)

        if not remapped_lines:
            empty += 1
            continue

        dst_img = TRAIN_IMAGES / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        dst_lbl = TRAIN_LABELS / (stem + ".txt")
        dst_lbl.write_text("\n".join(remapped_lines))

        existing_images.add(stem)
        copied += 1

    print(f"   Added   : {copied}")
    print(f"   Skipped : {skipped} duplicates")
    print(f"   Dropped : {empty} (no valid annotations after remapping)")
    return copied, skipped


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Check class mappings only — no files copied")
    args = parser.parse_args()

    print("\n── DriverGuard Dataset Merger ────────────────────────────────")
    print(f"   Target: {TRAIN_IMAGES}")

    if args.verify:
        print("\n── VERIFY MODE (no files copied) ──────────────────────────────")
        for cfg in DATASET_CONFIGS:
            print(f"\n  {cfg['name']}:")
            verify_dataset(cfg)
        print("\n   If mappings look correct → python merge_datasets.py")
        print("   If something looks wrong → tell me and I'll fix the remap\n")
        exit(0)

    # ── Full merge ────────────────────────────────────────────────────────────
    if not TRAIN_IMAGES.exists():
        print(f"\n✗  Not found: {TRAIN_IMAGES}")
        exit(1)

    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    print("\n   Scanning existing dataset...")
    existing = {p.stem for p in TRAIN_IMAGES.glob("*.*")}
    print(f"   Existing images: {len(existing)}")

    print("\n── Merging ────────────────────────────────────────────────────")
    total_added = 0
    for cfg in DATASET_CONFIGS:
        added, _ = merge_dataset(cfg, existing)
        total_added += added

    final_count = len(list(TRAIN_IMAGES.glob("*.*")))
    print(f"\n── Summary ────────────────────────────────────────────────────")
    print(f"   Added this run  : {total_added}")
    print(f"   Total images    : {final_count}")
    print(f"\n   Next → python finetune_india.py\n")