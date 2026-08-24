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
    python merge_datasets.py

Edit the DATASET_CONFIGS section below to point to your
downloaded dataset folders and define the class remapping.

What it does
────────────
    1. Reads each dataset's images + labels
    2. Remaps class IDs in label files
    3. Copies images + remapped labels to D:/We4/train/
    4. Skips duplicates (checks filename)
    5. Prints a summary of what was added
"""

import shutil
import os
from pathlib import Path

# ─── Your existing training folder ───────────────────────────────────────────
TRAIN_IMAGES = Path("D:/We4/train/images")
TRAIN_LABELS = Path("D:/We4/train/labels")

# ─── Dataset configurations ──────────────────────────────────────────────────
# For each dataset, define:
#   "path"    : root folder of downloaded dataset
#   "remap"   : {original_class_id → your_class_id}
#               Use -1 to skip/discard a class
#
# HOW TO FILL IN "remap":
#   Open the dataset's data.yaml and check "names" list.
#   The index in the list = the class ID in label files.
#   Map each index to your target class ID (0-4) or -1 to skip.
#
# Example:
#   If dataset yaml says: names: [pothole, speed_breaker, crack]
#   Then: {0: 0, 1: -1, 2: 1}
#   = pothole(0)→your pothole(0), speed_breaker(1)→skip, crack(2)→crack_long(1)

DATASET_CONFIGS = [

    # ── Dataset 1: Indian Roads Dataset ──────────────────────────────────────
    # kaggle.com/datasets/mitangshu11/indian-roads-dataset
    # Classes: pothole, speed_breaker, unpaved_road (check yaml to confirm order)
    {
        "name"  : "Indian Roads Dataset",
        "path"  : Path("D:/Downloads/indian-roads-dataset"),  # ← change this path
        "remap" : {
            0: 0,   # pothole → pothole
            1: -1,  # speed_breaker → skip (not in your classes)
            2: -1,  # unpaved_road → skip
        },
        # Folder structure inside the dataset:
        # Set to "" if images are directly in path/images/
        # Set to "train" if images are in path/train/images/
        "split" : "",
    },

    # ── Dataset 2: Potholes Detection YOLOv8 ─────────────────────────────────
    # kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8
    # Classes: pothole only (nc=1)
    {
        "name"  : "Potholes YOLOv8 Dataset",
        "path"  : Path("D:/Downloads/potholes-detection-yolov8"),  # ← change path
        "remap" : {
            0: 0,   # pothole → pothole
        },
        "split" : "train",  # images are in path/train/images/
    },

    # ── Dataset 3: Potholes + Cracks + Manholes ───────────────────────────────
    # kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes
    # Classes: pothole, crack, maintenance_hole (check yaml to confirm)
    {
        "name"  : "Potholes+Cracks+Manholes Dataset",
        "path"  : Path("D:/Downloads/road-damage-dataset-potholes-cracks-and-manholes"),  # ← change path
        "remap" : {
            0: 0,   # pothole → pothole
            1: 1,   # crack → crack_longitudinal
            2: -1,  # maintenance_hole → skip
        },
        "split" : "",
    },
]


# ─── Merge logic ─────────────────────────────────────────────────────────────

def find_images_and_labels(dataset_path: Path, split: str):
    """Find image and label directories inside dataset folder."""
    if split:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
    else:
        img_dir = dataset_path / "images"
        lbl_dir = dataset_path / "labels"

    # Fallback — try without subdirectory
    if not img_dir.exists():
        img_dir = dataset_path / "images"
        lbl_dir = dataset_path / "labels"

    if not img_dir.exists():
        print(f"   ✗ Images folder not found at: {img_dir}")
        print(f"     Check the 'path' and 'split' settings for this dataset")
        return None, None

    return img_dir, lbl_dir


def remap_label_file(src_label: Path, remap: dict) -> list:
    """
    Read a YOLO label file and remap class IDs.
    Returns list of remapped lines (empty lines filtered out).
    Lines with class ID mapped to -1 are discarded.
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
            orig_cls = int(parts[0])
            new_cls  = remap.get(orig_cls, -1)
            if new_cls == -1:
                continue   # skip this annotation
            parts[0] = str(new_cls)
            remapped.append(" ".join(parts))
    return remapped


def merge_dataset(config: dict, existing_images: set):
    """Process one dataset config and copy files to training folder."""
    name  = config["name"]
    path  = config["path"]
    remap = config["remap"]
    split = config["split"]

    print(f"\n── {name} ──────────────────────────────────")

    if not path.exists():
        print(f"   ✗ Dataset folder not found: {path}")
        print(f"     Update the 'path' in DATASET_CONFIGS and re-run")
        return 0, 0

    img_dir, lbl_dir = find_images_and_labels(path, split)
    if img_dir is None:
        return 0, 0

    # Count existing images to avoid duplicates
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + \
             list(img_dir.glob("*.jpeg"))
    print(f"   Found {len(images)} images")

    copied = skipped = empty = 0

    for img_path in images:
        stem = img_path.stem

        # Skip if already in training set (by filename)
        if stem in existing_images:
            skipped += 1
            continue

        # Find corresponding label
        lbl_path = lbl_dir / (stem + ".txt") if lbl_dir and lbl_dir.exists() \
                   else Path("__nonexistent__")

        # Remap label
        remapped_lines = remap_label_file(lbl_path, remap)

        # Skip images with no valid annotations after remapping
        if not remapped_lines:
            empty += 1
            continue

        # Copy image
        dst_img = TRAIN_IMAGES / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Write remapped label
        dst_lbl = TRAIN_LABELS / (stem + ".txt")
        dst_lbl.write_text("\n".join(remapped_lines))

        existing_images.add(stem)
        copied += 1

    print(f"   Added  : {copied} images with labels")
    print(f"   Skipped: {skipped} duplicates")
    print(f"   Dropped: {empty} (no valid annotations after remapping)")
    return copied, skipped


def check_yaml(config: dict):
    """Print dataset yaml so user can verify class order."""
    path  = config["path"]
    name  = config["name"]
    split = config["split"]

    if not path.exists():
        return

    # Look for yaml files
    yamls = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
    if yamls:
        print(f"\n   {name} — class structure:")
        with open(yamls[0]) as f:
            for line in f:
                if "names" in line or "nc" in line:
                    print(f"     {line.strip()}")
    else:
        print(f"\n   {name} — no yaml found, check folder manually")


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── DriverGuard Dataset Merger ────────────────────────────────")
    print(f"   Target train images : {TRAIN_IMAGES}")
    print(f"   Target train labels : {TRAIN_LABELS}")

    # Verify target folders exist
    if not TRAIN_IMAGES.exists():
        print(f"\n✗  Training images folder not found: {TRAIN_IMAGES}")
        print("   Make sure D:/We4/train/images/ exists")
        exit(1)

    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    # Get existing image stems to avoid duplicates
    print("\n   Scanning existing dataset...")
    existing = {p.stem for p in TRAIN_IMAGES.glob("*.*")}
    print(f"   Existing images: {len(existing)}")

    # First — print yaml info so user can verify class mapping
    print("\n── Verifying dataset class structures ─────────────────────────")
    for config in DATASET_CONFIGS:
        check_yaml(config)

    print("\n── Merging datasets ───────────────────────────────────────────")
    total_added = 0
    for config in DATASET_CONFIGS:
        added, _ = merge_dataset(config, existing)
        total_added += added

    # Final count
    final_count = len(list(TRAIN_IMAGES.glob("*.*")))
    print(f"\n── Summary ────────────────────────────────────────────────────")
    print(f"   Images added this run : {total_added}")
    print(f"   Total training images : {final_count}")
    print(f"\n   Next step: run python finetune_india.py")
    print("──────────────────────────────────────────────────────────────\n")
