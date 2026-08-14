#!/usr/bin/env python3
"""
Cross-modal alignment statistics and visualization.

Computes cosine similarity between geometry and image embeddings for all
paired samples and produces summary stats + plots.

Usage:
  python alignment_stats.py [--ckpt train_log/default/model/latest.pt]
                            [--output alignment_stats.png]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gmdl import (
    GeometryEncoder, PointNetEncoder, ImageEncoder, ManufacturingDataset,
    ConfigGeometryEncoder, ConfigProcessPredictor,
    GEOM_FEATURE_DIM, LATENT_DIM, N_PROCESSES,
)

MANUFACTURING_PROCESSES = [
    "3-axis CNC machining", "5-axis CNC machining", "Injection molding",
    "Casting", "Forging", "Lathing/Turning",
    "Sheet metal fabrication", "3D printing", "Sintering",
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("cfg", {})
    geom_cfg = ConfigGeometryEncoder.from_dict(saved_cfg.get("geom_encoder", {}))
    use_pointnet = getattr(geom_cfg, "encoder_type", "mlp") == "pointnet"
    if use_pointnet:
        geom_encoder = PointNetEncoder(geom_cfg).to(device)
    else:
        geom_encoder = GeometryEncoder(geom_cfg).to(device)
    geom_encoder.load_state_dict(ckpt["geom_encoder"])
    geom_encoder.eval()

    image_encoder = None
    if "image_encoder" in ckpt:
        image_encoder = ImageEncoder(geom_cfg).to(device)
        image_encoder.load_state_dict(ckpt["image_encoder"])
        image_encoder.eval()

    return geom_encoder, image_encoder, saved_cfg, use_pointnet


def extract_pairs(dataset, geom_encoder, image_encoder, device, use_pointnet=False):
    """Extract geometry/image latent pairs for samples that have images."""
    pairs = []  # list of (z_geom, z_img, label)

    for i in range(len(dataset)):
        sample = dataset[i]
        has_img = sample["has_image"].item() if hasattr(sample["has_image"], "item") else sample["has_image"]
        if not has_img or image_encoder is None:
            continue

        label = sample["process_label"].item() if hasattr(sample["process_label"], "item") else sample["process_label"]

        if use_pointnet:
            geom_input = sample["vertices"].unsqueeze(0).to(device)
            n_valid = torch.tensor([sample["n_valid_points"]], device=device)
        else:
            geom_input = sample["geom_features"].unsqueeze(0).to(device)
            n_valid = None

        img_tensor = sample["image_tensor"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_geom = geom_encoder(geom_input, n_valid_points=n_valid).squeeze(0)
            z_img = image_encoder(img_tensor).squeeze(0)

        pairs.append((z_geom, z_img, label))

    return pairs


def compute_stats(pairs):
    """Compute cosine similarity stats."""
    cos_sims = []
    by_class = {}

    for z_geom, z_img, label in pairs:
        cos = F.cosine_similarity(z_geom.unsqueeze(0), z_img.unsqueeze(0)).item()
        cos_sims.append(cos)
        by_class.setdefault(label, []).append(cos)

    cos_sims = np.array(cos_sims)
    stats = {
        "n_pairs": len(cos_sims),
        "mean": float(cos_sims.mean()),
        "std": float(cos_sims.std()),
        "min": float(cos_sims.min()),
        "max": float(cos_sims.max()),
        "p25": float(np.percentile(cos_sims, 25)),
        "p50": float(np.percentile(cos_sims, 50)),
        "p75": float(np.percentile(cos_sims, 75)),
    }
    per_class = {}
    for label, sims in sorted(by_class.items()):
        sims = np.array(sims)
        per_class[int(label)] = {
            "name": MANUFACTURING_PROCESSES[label],
            "n": len(sims),
            "mean": float(sims.mean()),
            "std": float(sims.std()),
            "p50": float(np.percentile(sims, 50)),
        }
    return stats, per_class, cos_sims, by_class


def print_stats(stats, per_class):
    print(f"\n{'='*55}")
    print(f"  Cross-Modal Alignment Statistics")
    print(f"{'='*55}")
    print(f"  Paired samples:  {stats['n_pairs']}")
    print(f"  Mean cosine sim: {stats['mean']:.4f}")
    print(f"  Std:             {stats['std']:.4f}")
    print(f"  Min:             {stats['min']:.4f}")
    print(f"  Max:             {stats['max']:.4f}")
    print(f"  Percentiles:     p25={stats['p25']:.4f}  p50={stats['p50']:.4f}  p75={stats['p75']:.4f}")
    print(f"\n  Per-class breakdown:")
    for label, c in per_class.items():
        print(f"    [{label}] {c['name']:30s}  n={c['n']:4d}  mean={c['mean']:.4f}  std={c['std']:.4f}  p50={c['p50']:.4f}")
    print(f"{'='*55}\n")


def plot(stats, per_class, cos_sims, by_class, output_path):
    cmap = {0: "#1f77b4", 2: "#ff7f0e"}
    label_names = {k: v["name"] for k, v in per_class.items()}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Overall histogram ---
    ax = axes[0]
    ax.hist(cos_sims, bins=40, color="#555555", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(stats["mean"], color="red", linestyle="--", linewidth=1.5, label=f'Mean = {stats["mean"]:.3f}')
    ax.axvline(stats["p50"], color="orange", linestyle=":", linewidth=1.5, label=f'Median = {stats["p50"]:.3f}')
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Overall Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.1, 1.05)

    # --- Panel 2: Per-class histogram overlay ---
    ax = axes[1]
    for label, sims in by_class.items():
        ax.hist(sims, bins=30, color=cmap.get(label, "gray"), edgecolor="white",
                linewidth=0.5, alpha=0.6, label=f'{label_names[label]} (n={len(sims)})')
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Per-Class Distribution", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.1, 1.05)

    # --- Panel 3: Per-class box plot ---
    ax = axes[2]
    labels_sorted = sorted(by_class.keys())
    data = [np.array(by_class[l]) for l in labels_sorted]
    names = [label_names[l] for l in labels_sorted]
    colors = [cmap.get(l, "gray") for l in labels_sorted]

    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Per-Class Box Plot", fontsize=14)
    ax.set_ylim(-0.1, 1.05)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cross-modal alignment statistics")
    parser.add_argument("--ckpt", type=str, default="train_log/default/model/latest.pt")
    parser.add_argument("--output", type=str, default="alignment_stats.png")
    parser.add_argument("--stats_json", type=str, default=None,
                        help="Optional path to save stats as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    geom_encoder, image_encoder, saved_cfg, use_pointnet = load_model(args.ckpt, device)
    print(f"Encoder: {'PointNet' if use_pointnet else 'MLP'}")

    if image_encoder is None:
        print("ERROR: No image encoder found in checkpoint. Need paired training with images.")
        sys.exit(1)

    data_root = saved_cfg.get("data_root", "data")
    image_root = saved_cfg.get("image_root", None)
    max_points = saved_cfg.get("geom_encoder", {}).get("max_points", 512)

    all_pairs = []
    for split in ["train", "validation"]:
        ds = ManufacturingDataset(data_root, split=split, image_root=image_root,
                                  use_pointnet=use_pointnet, max_points=max_points)
        print(f"  {split}: {len(ds)} samples")
        pairs = extract_pairs(ds, geom_encoder, image_encoder, device, use_pointnet=use_pointnet)
        print(f"    {len(pairs)} paired (have image)")
        all_pairs.extend(pairs)

    if len(all_pairs) == 0:
        print("ERROR: No paired samples found.")
        sys.exit(1)

    stats, per_class, cos_sims, by_class = compute_stats(all_pairs)
    print_stats(stats, per_class)
    plot(stats, per_class, cos_sims, by_class, args.output)

    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump({"overall": stats, "per_class": per_class}, f, indent=2)
        print(f"Saved stats to {args.stats_json}")


if __name__ == "__main__":
    main()
