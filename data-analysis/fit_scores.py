#!/usr/bin/env python3
"""
Fit score visualization: compute and plot fit scores across a dataset.

======================================================================
The fit score for a part-process pair combines:
  - Softmax probability P(process | latent)         (weight 0.6)
  - Cosine similarity to the class centroid in z    (weight 0.4)

A score near 1.0 means the part is a strong fit; near 0.0 it is a
poor fit.

Output (2-panel figure saved to --output):
  Left  — Heatmap: fit scores for every process across dataset samples.
           Rows are dataset samples, columns are the 9 processes.
           Darker = higher score.  Diagonal blocks indicate good
           process-part alignment.
  Right — Scatter: current fit score (x) vs optimised score (y) for
           each part toward its own process.  Points above the diagonal
           show improvement.  Color = initial score.

Usage:
  python fit_scores.py --ckpt train_log/default/model/latest.pt \
      --decoders train_log/default/model/decoders.pt \
      --data-root HKS-CNN3 --output fit_scores.png
"""

import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from gmdl import (
    ManufacturingProcessPredictor, ProcessFitScorer,
    MANUFACTURING_PROCESSES, N_PROCESSES,
)
from gmdl.decoder_training import DecoderDataset


def main():
    parser = argparse.ArgumentParser(description="Fit score visualization")
    parser.add_argument("--ckpt", type=str, default="train_log/default/model/latest.pt")
    parser.add_argument("--decoders", type=str, default=None)
    parser.add_argument("--centroids", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="HKS-CNN3")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=50, help="Samples to analyze")
    parser.add_argument("--output", type=str, default="fit_scores.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load predictor
    predictor = ManufacturingProcessPredictor()
    if os.path.exists(args.ckpt):
        predictor.load_weights(args.ckpt)
    if args.decoders:
        predictor.load_decoders(args.decoders)
    if args.centroids:
        predictor.load_centroids(args.centroids)

    # Load data
    manifest = args.manifest or os.path.join(args.data_root, "train.json")
    ds = DecoderDataset(args.data_root, manifest_file=manifest)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    # Compute centroids if not loaded
    if not predictor.class_centroids:
        print("Computing centroids...")
        # Need labels — use manifest to build label map
        with open(manifest) as f:
            samples = json.load(f)
        labels_map = {}
        for i, s in enumerate(samples):
            labels_map[i] = s.get("process_label", 0)

        latents_by_label = {}
        predictor.model.eval()
        for i, batch in enumerate(loader):
            vertices, _, n_valid = batch
            vertices = vertices.to(device)
            n_valid = n_valid.to(device)
            with torch.no_grad():
                z = predictor.model.geom_encoder(vertices, n_valid_points=n_valid)
            lbl = labels_map[i]
            if lbl not in latents_by_label:
                latents_by_label[lbl] = []
            latents_by_label[lbl].append(z.cpu().numpy())

        predictor.class_centroids = {}
        for lbl, vecs in latents_by_label.items():
            predictor.class_centroids[lbl] = np.mean(np.concatenate(vecs, axis=0), axis=0)
        print(f"Computed {len(predictor.class_centroids)} centroids")

    # Score samples
    n = min(args.n_samples, len(ds))
    print(f"Scoring {n} samples...")

    all_scores = []  # [n, N_PROCESSES]
    loader_iter = iter(loader)

    for i in range(n):
        batch = next(loader_iter)
        vertices, _, n_valid = batch
        vertices = vertices.to(device)
        n_valid = n_valid.to(device)
        with torch.no_grad():
            z = predictor.model.geom_encoder(vertices, n_valid_points=n_valid)
        results = ProcessFitScorer(predictor.model, predictor.class_centroids).score(z)
        all_scores.append([r.fit_score for r in results])

    scores = np.array(all_scores)  # [n, N_PROCESSES]

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(scores.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(N_PROCESSES))
    ax.set_yticklabels(MANUFACTURING_PROCESSES, fontsize=8)
    ax.set_xlabel("Sample index")
    ax.set_title("Per-Process Fit Scores (greener = better fit)")
    plt.colorbar(im, ax=ax, label="Fit score")

    # Right: best process per sample
    ax = axes[1]
    best_idx = scores.argmax(axis=1)
    best_score = scores.max(axis=1)
    colors = plt.cm.tab10(best_idx / max(N_PROCESSES, 1))
    ax.scatter(range(n), best_score, c=colors, s=30, alpha=0.7)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Best fit score")
    ax.set_title("Best Process per Sample (colored by process class)")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
