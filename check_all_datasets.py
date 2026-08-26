"""
check_all_datasets.py
──────────────────────
Checks all 3 datasets thoroughly even without yaml files.
Reads label files directly to find class IDs,
checks README and COCO json for class names.

Run this and share the full output.

Usage:
    python check_all_datasets.py
"""

import json
import os
from pathlib import Path

DATASETS = [
    {
        "name": "Dataset 1 — Indian Roads",
        "root": Path("D:/We4/datasets/indian_road"),
    },
    {
        "name": "Dataset 2 — Potholes YOLOv8",
        "root": Path("D:/We4/datasets/yv8"),
    },
    {
        "name": "Dataset 3 — Potholes + Cracks + Manholes",
        "root": Path("D:/We4/datasets/potholes_and_manhole"),
    },
]


def scan_class_ids(label_dir: Path, max_files=100):
    """Read up to max_files label .txt files and collect all class IDs."""
    class_ids = set()
    files = list(label_dir.rglob("*.txt"))[:max_files]
    for f in files:
        try:
            with open(f) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_ids.add(int(parts[0]))
        except Exception:
            pass
    return sorted(class_ids), len(list(label_dir.rglob("*.txt")))


def print_file_content(path: Path, max_lines=30):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    print("      ... (truncated)")
                    break
                print(f"      {line.rstrip()}")
    except Exception as e:
        print(f"      Error reading: {e}")


def count_images(root: Path):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    total = 0
    for ext in exts:
        total += len(list(root.rglob(ext)))
    return total


print("\n══════════════════════════════════════════════════════════════")
print("  Dataset Class Checker — Full Analysis")
print("══════════════════════════════════════════════════════════════")

for ds in DATASETS:
    root = ds["root"]
    print(f"\n\n{'─'*60}")
    print(f"  {ds['name']}")
    print(f"{'─'*60}")

    if not root.exists():
        print(f"  ✗ Folder not found: {root}")
        continue

    # ── Full folder tree (3 levels) ───────────────────────────────────────────
    print("\n  📁 Folder structure:")
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth <= 3:
            indent = "  " + "  " * depth
            if p.is_dir():
                n_files = len(list(p.glob("*")))
                print(f"{indent}{p.name}/  ({n_files} items)")
            else:
                size = os.path.getsize(p)
                if size < 1024*1024:
                    sz = f"{size//1024}KB"
                else:
                    sz = f"{size//1024//1024}MB"
                print(f"{indent}{p.name}  [{sz}]")

    # ── Print all yaml files ──────────────────────────────────────────────────
    yamls = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    if yamls:
        for yf in yamls[:2]:
            print(f"\n  📄 {yf.name}:")
            print_file_content(yf)
    else:
        print("\n  ⚠  No yaml found")

    # ── Print README if exists ────────────────────────────────────────────────
    readmes = list(root.rglob("README*"))
    if readmes:
        print(f"\n  📄 {readmes[0].name}:")
        print_file_content(readmes[0], max_lines=20)

    # ── COCO json categories ──────────────────────────────────────────────────
    coco_jsons = list(root.rglob("*.json"))
    if coco_jsons:
        for cj in coco_jsons[:1]:
            try:
                with open(cj) as f:
                    coco = json.load(f)
                cats = coco.get("categories", [])
                if cats:
                    print(f"\n  🏷  Classes from {cj.name}:")
                    for cat in cats:
                        print(f"     ID: {cat.get('id')}  Name: {cat.get('name')}")
            except Exception as e:
                print(f"\n  ⚠  Could not read {cj.name}: {e}")

    # ── Scan label files for class IDs ────────────────────────────────────────
    label_dirs = [p for p in root.rglob("*")
                  if p.is_dir() and "label" in p.name.lower()]
    if label_dirs:
        for ld in label_dirs:
            ids, total = scan_class_ids(ld)
            if total > 0:
                print(f"\n  🔢 Class IDs in {ld.name}/: {ids}")
                print(f"     Total label files: {total}")
    else:
        # Try scanning all txt files from root
        ids, total = scan_class_ids(root)
        print(f"\n  🔢 Class IDs found in label files: {ids}")
        print(f"     Total label files: {total}")

    # ── Image count ───────────────────────────────────────────────────────────
    n_imgs = count_images(root)
    print(f"\n  🖼  Total images: {n_imgs}")

print("\n\n══════════════════════════════════════════════════════════════")
print("  Share the full output above")
print("══════════════════════════════════════════════════════════════\n")
