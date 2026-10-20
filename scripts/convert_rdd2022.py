"""
scripts/convert_rdd2022.py
───────────────────────────
Convert RDD2022 dataset (Pascal VOC XML) → YOLO format (.txt).

RDD2022 structure (after download)
────────────────────────────────────
RDD2022/
  country/
    India/
      train/
        images/   ← .jpg frames
        annotations/xmls/  ← Pascal VOC .xml
      test/
        images/

RDD2022 class names → VSM class IDs
─────────────────────────────────────
    D00 Longitudinal crack   → 1 (crack)
    D10 Transverse crack     → 1 (crack)
    D20 Alligator crack      → 1 (crack)
    D40 Pothole              → 0 (pothole)
    (all other classes       → skipped unless --keep-all)

Output layout
─────────────
    data/road_dataset/
      images/
        train/   ← .jpg files (symlinked or copied)
        val/     ← held-out split (default 15 %)
      labels/
        train/   ← .txt YOLO annotations
        val/

Usage
─────
    # Convert India split only
    python scripts/convert_rdd2022.py \\
        --rdd-root /path/to/RDD2022 \\
        --out-root data/road_dataset \\
        --countries India \\
        --val-split 0.15

    # All countries, keep all damage classes mapped to crack/pothole
    python scripts/convert_rdd2022.py \\
        --rdd-root /path/to/RDD2022 \\
        --out-root data/road_dataset \\
        --val-split 0.15
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ─── Class mapping ────────────────────────────────────────────────────────────
# RDD2022 label → VSM class id
#   0 = pothole  1 = crack  (2 = rutting — not in RDD2022, add manually)

RDD_TO_VSM = {
    "D00": 1,   # Longitudinal crack
    "D01": 1,
    "D10": 1,   # Transverse crack
    "D11": 1,
    "D20": 1,   # Alligator crack
    "D40": 0,   # Pothole
    "D43": 0,
    "D44": 0,
}

ALL_COUNTRIES = ["India", "Japan", "Czech", "Norway", "United_States", "China"]


# ─── XML parser ───────────────────────────────────────────────────────────────

def parse_xml(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse a Pascal VOC XML file and return YOLO-format annotation lines.

    Each line: <class_id> <cx> <cy> <w> <h>   (all normalised 0–1)

    Returns [] if no relevant objects are found.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    lines = []

    # Prefer size from XML if image dims not passed
    size_el = root.find("size")
    if size_el is not None:
        img_w = int(size_el.findtext("width",  default=str(img_w)))
        img_h = int(size_el.findtext("height", default=str(img_h)))

    if img_w == 0 or img_h == 0:
        return []

    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        cls_id = RDD_TO_VSM.get(name)
        if cls_id is None:
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        # Clamp to image bounds
        xmin, xmax = max(0, xmin), min(img_w, xmax)
        ymin, ymax = max(0, ymin), min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


# ─── Main conversion ──────────────────────────────────────────────────────────

def convert(rdd_root: Path, out_root: Path,
            countries: list[str], val_split: float,
            copy_images: bool):

    # Collect (image_path, xml_path) pairs
    pairs = []
    for country in countries:
        for split in ("train",):          # RDD2022 test has no labels
            img_dir = rdd_root / country / split / "images"
            xml_dir = rdd_root / country / split / "annotations" / "xmls"
            if not img_dir.exists():
                print(f"  ⚠  Not found: {img_dir} — skipping")
                continue
            for img in sorted(img_dir.glob("*.jpg")):
                xml = xml_dir / (img.stem + ".xml")
                if xml.exists():
                    pairs.append((img, xml))

    print(f"\n  Found {len(pairs)} labelled images across: {countries}")

    # Shuffle + split
    random.shuffle(pairs)
    n_val  = max(1, int(len(pairs) * val_split))
    val    = set(range(len(pairs) - n_val, len(pairs)))

    # Output directories
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = empty = converted = 0

    for i, (img_path, xml_path) in enumerate(pairs):
        split = "val" if i in val else "train"

        # Get image dimensions without loading full image
        try:
            from PIL import Image as _Img
            with _Img.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        yolo_lines = parse_xml(xml_path, w, h)

        if not yolo_lines:
            empty += 1
            continue

        # Write label file
        label_out = out_root / "labels" / split / (img_path.stem + ".txt")
        label_out.write_text("\n".join(yolo_lines))

        # Image: copy or symlink
        img_out = out_root / "images" / split / img_path.name
        if not img_out.exists():
            if copy_images:
                shutil.copy2(img_path, img_out)
            else:
                img_out.symlink_to(img_path.resolve())

        converted += 1

    print(f"\n  Converted : {converted}")
    print(f"  Empty XML : {empty}  (no relevant classes)")
    print(f"  Train     : {converted - sum(1 for i in val if i < len(pairs))}")
    print(f"  Val       : {n_val}")

    # Write dataset YAML
    yaml_out = out_root / "dataset.yaml"
    yaml_out.write_text(
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: pothole\n"
        "  1: crack\n"
        "  2: rutting\n"
    )
    print(f"\n  Dataset YAML: {yaml_out}")
    print("\n  ✓ Conversion complete.\n")
    print("  Next step:")
    print("    python scripts/train_road.py --epochs 50 --batch 16 --device 0\n")


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RDD2022 Pascal VOC → YOLO format")
    parser.add_argument("--rdd-root",  required=True,
                        help="Path to extracted RDD2022 root directory")
    parser.add_argument("--out-root",  default="data/road_dataset",
                        help="Output directory for images/ and labels/")
    parser.add_argument("--countries", nargs="+", default=["India"],
                        choices=ALL_COUNTRIES + ["all"],
                        help="Which RDD2022 country subsets to include")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction held out for validation (default 0.15)")
    parser.add_argument("--copy",      action="store_true",
                        help="Copy images (default: symlink to save disk)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    countries = ALL_COUNTRIES if "all" in args.countries else args.countries
    rdd_root  = Path(args.rdd_root)
    out_root  = Path(args.out_root)

    if not rdd_root.exists():
        print(f"✗ RDD root not found: {rdd_root}")
        raise SystemExit(1)

    print(f"\n── RDD2022 → YOLO Converter ──────────────────────────")
    print(f"  Source : {rdd_root}")
    print(f"  Output : {out_root}")
    print(f"  Countries: {countries}")
    print(f"  Val split: {args.val_split * 100:.0f}%")

    convert(rdd_root, out_root, countries, args.val_split, args.copy)
