#!/usr/bin/env python3
"""
UMAP visualization of the learned latent space.

2-panel output:
  Left  — Geometry encoder latents for all train+val samples, colored by class
  Right — Cross-modal alignment: geometry vs image embeddings for paired samples

Usage:
  python umap_viz.py [--ckpt train_log/default/model/latest.pt] [--output umap_latent.png]
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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


def extract_latents(dataset, geom_encoder, image_encoder, device, use_pointnet=False):
    geom_latents = []
    img_latents = []
    labels = []
    has_images = []

    for i in range(len(dataset)):
        sample = dataset[i]
        if use_pointnet:
            geom_input = sample["vertices"].unsqueeze(0).to(device)
            n_valid = torch.tensor([sample["n_valid_points"]], device=device)
        else:
            geom_input = sample["geom_features"].unsqueeze(0).to(device)
            n_valid = None
        label = sample["process_label"].item() if hasattr(sample["process_label"], "item") else sample["process_label"]
        has_img = sample["has_image"].item() if hasattr(sample["has_image"], "item") else sample["has_image"]

        with torch.no_grad():
            z_geom = geom_encoder(geom_input, n_valid_points=n_valid).squeeze(0).cpu().numpy()
        geom_latents.append(z_geom)

        if has_img and image_encoder is not None:
            img_tensor = sample["image_tensor"].unsqueeze(0).to(device)
            with torch.no_grad():
                z_img = image_encoder(img_tensor).squeeze(0).cpu().numpy()
            img_latents.append(z_img)
        else:
            img_latents.append(None)

        labels.append(label)
        has_images.append(has_img)

    return (
        np.array(geom_latents),
        img_latents,
        np.array(labels),
        np.array(has_images),
    )


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of latent space")
    parser.add_argument("--ckpt", type=str, default="train_log/default/model/latest.pt")
    parser.add_argument("--output", type=str, default="umap_latent.png")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    geom_encoder, image_encoder, saved_cfg, use_pointnet = load_model(args.ckpt, device)
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"  Geometry encoder: {'PointNet' if use_pointnet else 'MLP'}")
    print(f"  Image encoder: {'loaded' if image_encoder else 'not found in checkpoint'}")

    data_root = saved_cfg.get("data_root", "data")
    image_root = saved_cfg.get("image_root", None)
    max_points = saved_cfg.get("geom_encoder", {}).get("max_points", 512)

    train_ds = ManufacturingDataset(data_root, split="train", image_root=image_root,
                                    use_pointnet=use_pointnet, max_points=max_points)
    val_ds = ManufacturingDataset(data_root, split="validation", image_root=image_root,
                                 use_pointnet=use_pointnet, max_points=max_points)
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    print("Extracting geometry latents...")
    geom_train, img_train, labels_train, has_img_train = extract_latents(
        train_ds, geom_encoder, image_encoder, device, use_pointnet=use_pointnet
    )
    print("Extracting validation latents...")
    geom_val, img_val, labels_val, has_img_val = extract_latents(
        val_ds, geom_encoder, image_encoder, device, use_pointnet=use_pointnet
    )

    geom_all = np.concatenate([geom_train, geom_val], axis=0)
    labels_all = np.concatenate([labels_train, labels_val], axis=0)
    split_all = np.array([0] * len(train_ds) + [1] * len(val_ds))
    has_img_all = np.concatenate([has_img_train, has_img_val])
    n_train = len(train_ds)

    print(f"Running UMAP on {len(geom_all)} samples...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
    )
    embedding = reducer.fit_transform(geom_all)

    cmap = {0: "#1f77b4", 2: "#ff7f0e"}
    label_names = {0: "3-axis CNC", 2: "Injection molding"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: geometry latents by class ---
    ax = axes[0]
    for cls in [0, 2]:
        for sp, marker in [(0, "o"), (1, "^")]:
            mask = (labels_all == cls) & (split_all == sp)
            split_label = "Train" if sp == 0 else "Val"
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=cmap[cls], marker=marker, s=12, alpha=0.7,
                edgecolors="none",
                label=f"{label_names[cls]} ({split_label})",
            )
    ax.set_title("Geometry Encoder Latent Space", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=9, markerscale=1.5, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Right panel: cross-modal alignment ---
    ax = axes[1]
    paired_mask = has_img_all == 1
    paired_indices = np.where(paired_mask)[0]
    print(f"Paired samples (with image): {len(paired_indices)}")

    img_all_latents = []
    for entry in img_train:
        img_all_latents.append(entry if entry is not None else np.full(LATENT_DIM, np.nan))
    for entry in img_val:
        img_all_latents.append(entry if entry is not None else np.full(LATENT_DIM, np.nan))
    img_all_latents = np.array(img_all_latents)

    if len(paired_indices) > 0 and image_encoder is not None:
        geom_paired = embedding[paired_indices]
        labels_paired = labels_all[paired_indices]

        reducer_img = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            metric="euclidean",
            random_state=42,
        )
        combined = np.concatenate([
            geom_all[paired_indices],
            img_all_latents[paired_indices],
        ], axis=0)
        combined_emb = reducer_img.fit_transform(combined)
        n_p = len(paired_indices)
        geom_emb = combined_emb[:n_p]
        img_emb = combined_emb[n_p:]

        for cls in [0, 2]:
            mask = labels_paired == cls
            ax.scatter(
                geom_emb[mask, 0], geom_emb[mask, 1],
                c=cmap[cls], marker="o", s=14, alpha=0.7,
                edgecolors="none",
                label=f"{label_names[cls]} (geom)",
            )
            ax.scatter(
                img_emb[mask, 0], img_emb[mask, 1],
                c=cmap[cls], marker="^", s=14, alpha=0.5,
                edgecolors="white", linewidths=0.3,
                label=f"{label_names[cls]} (image)",
            )

        step = max(1, len(paired_indices) // 200)
        for idx in range(0, len(paired_indices), step):
            ax.plot(
                [geom_emb[idx, 0], img_emb[idx, 0]],
                [geom_emb[idx, 1], img_emb[idx, 1]],
                c="gray", alpha=0.15, linewidth=0.5,
            )
    else:
        ax.text(0.5, 0.5, "No paired samples or image encoder unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.set_title("Cross-Modal Alignment", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=9, markerscale=1.5, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
