"""
mk_manifest.py — Generate a PointNet-compatible JSON manifest from raw mesh directories.

Scans a directory tree (e.g. HKS-CNN3/) containing process-named subdirectories
with .mat mesh files and optional paired .jpg images.  Writes a JSON array that
the gmdl.py ManufacturingDataset reads directly — no geometry feature computation
required.

For PointNet training (--encoder pointnet), the dataset reads raw vertices from
the .mat files at runtime.  This script only creates the index that maps files
to process labels.

Usage:
  python mk_manifest.py --root HKS-CNN3 --output data/train_hks3.json

  # Then split and train:
  python data-helpers/datasplit.py data/train_hks3.json
  python gmdl.py --encoder pointnet --contrastive_epochs 0 --phase2_epochs 50

Output format (one entry per mesh):
  {
    "mesh_file":       "InjectionMolding/42.mat",
    "image":           "InjectionMolding/42.jpg",
    "process_label":   2
  }

Label mapping (override with --labels):
  InjectionMolding → 2   Machining → 0   Turning → 5

If --data_root is set, mesh_file/image paths are made relative to that root
instead of --root, so the manifest works when the data lives elsewhere at
training time.
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter

# Allow running from data-helpers/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gmdl import MANUFACTURING_PROCESSES

# Default mapping: subdirectory name → gmdl.py process label index
DEFAULT_LABELS = {
    "InjectionMolding": 2,   # Injection molding
    "Machining":        0,   # 3-axis CNC machining
    "Turning":          5,   # Lathing/Turning
    "LaserCut":         0,   # no direct match → default to 3-axis CNC
    "Casting":          3,
    "Forging":          4,
    "SheetMetal":       6,
    "Printing":         7,
    "Sintering":        8,
}


def scan_directory(root, label_map, data_root=None):
    """Walk process subdirectories and collect mesh + image pairs.

    Returns a list of dicts ready to be serialised as the training JSON.
    """
    path_root = data_root if data_root else root
    samples = []

    for dirname, label in sorted(label_map.items()):
        dirpath = os.path.join(root, dirname)
        if not os.path.isdir(dirpath):
            print(f"  Skipping {dirname} (not found)")
            continue

        # Find all .mat files in this subdirectory
        mat_files = sorted(glob.glob(os.path.join(dirpath, "*.mat")))

        n_paired = 0
        for mat_path in mat_files:
            stem = os.path.splitext(os.path.basename(mat_path))[0]
            rel_mesh = os.path.relpath(mat_path, path_root)

            # Look for a paired image with the same stem
            img_path = os.path.join(dirpath, f"{stem}.jpg")
            rel_image = os.path.relpath(img_path, path_root) if os.path.exists(img_path) else ""

            if rel_image:
                n_paired += 1

            samples.append({
                "mesh_file":     rel_mesh,
                "image":         rel_image,
                "process_label": label,
            })

        print(f"  {dirname} ({MANUFACTURING_PROCESSES[label]}): "
              f"{len(mat_files)} meshes, {n_paired} paired images")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate a PointNet-compatible JSON manifest from raw mesh directories")
    parser.add_argument("--root", required=True,
                        help="Root directory containing process subdirectories "
                             "(e.g. HKS-CNN3/)")
    parser.add_argument("--output", default="data/train.json",
                        help="Output JSON path (default: data/train.json)")
    parser.add_argument("--data_root", default=None,
                        help="Make paths relative to this root instead of --root "
                             "(use when data will live elsewhere at training time)")
    parser.add_argument("--labels", default=None,
                        help='Override label mapping as JSON dict, e.g. '
                             '\'{"InjectionMolding": 2, "Machining": 0}\'')
    args = parser.parse_args()

    label_map = dict(DEFAULT_LABELS)
    if args.labels:
        label_map.update(json.loads(args.labels))

    print(f"Scanning {args.root}")
    samples = scan_directory(args.root, label_map, data_root=args.data_root)

    if not samples:
        print("Error: no samples found")
        return

    # Write manifest
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    # Summary
    labels = [s["process_label"] for s in samples]
    n_paired = sum(1 for s in samples if s["image"])
    print(f"\nWrote {len(samples)} samples to {args.output}")
    print(f"With paired images: {n_paired}/{len(samples)}")
    print("Label distribution:")
    for idx, count in sorted(Counter(labels).items()):
        print(f"  {idx}: {MANUFACTURING_PROCESSES[idx]} → {count}")


if __name__ == "__main__":
    main()
