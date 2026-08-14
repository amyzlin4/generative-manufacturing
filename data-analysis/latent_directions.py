#!/usr/bin/env python3
"""
Latent direction visualization: show how latents move toward target processes.

Takes a sample part, picks a target process (default: best-fitting), and
optimises the latent code to increase fit for that process.  The entire
traversal path is recorded and visualised.

Output (3-panel figure saved to --output):
  Top    — UMAP projection of the full dataset's latents (colored by class)
           + the traversal path from original z (star) to optimised z (arrow).
  Middle — Top-12 feature changes (absolute |delta|) from the feature decoder.
           Bar color indicates increase (green) or decrease (red).
  Bottom — Fit score vs optimisation step for the target process.  Shows
           the optimisation trajectory and convergence.

Usage:
  python latent_directions.py --ckpt train_log/default/model/latest.pt \
      --decoders train_log/default/model/decoders.pt \
      --data-root HKS-CNN3 --output latent_directions.png
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
    LatentDirectionAnalyzer, DesignExplainer,
    MANUFACTURING_PROCESSES, FEATURE_NAMES,
)
from gmdl.decoder_training import DecoderDataset, mesh_to_geom_features


def main():
    parser = argparse.ArgumentParser(description="Latent direction visualization")
    parser.add_argument("--ckpt", type=str, default="train_log/default/model/latest.pt")
    parser.add_argument("--decoders", type=str, default=None)
    parser.add_argument("--centroids", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="HKS-CNN3")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output", type=str, default="latent_directions.png")
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

    # Load data for centroids
    manifest = args.manifest or os.path.join(args.data_root, "train.json")
    with open(manifest) as f:
        samples = json.load(f)

    ds = DecoderDataset(args.data_root, manifest_file=manifest)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Build centroids from label map
    labels_map = {i: s.get("process_label", 0) for i, s in enumerate(samples)}
    latents_by_label = {}
    all_latents = []
    all_labels = []

    predictor.model.eval()
    for i, batch in enumerate(loader):
        vertices, _, n_valid = batch
        vertices = vertices.to(device)
        n_valid = n_valid.to(device)
        with torch.no_grad():
            z = predictor.model.geom_encoder(vertices, n_valid_points=n_valid)
        z_np = z.squeeze(0).cpu().numpy()
        all_latents.append(z_np)
        lbl = labels_map.get(i, 0)
        all_labels.append(lbl)
        if lbl not in latents_by_label:
            latents_by_label[lbl] = []
        latents_by_label[lbl].append(z_np)

    all_latents = np.array(all_latents)
    all_labels = np.array(all_labels)

    centroids = {}
    for lbl, vecs in latents_by_label.items():
        centroids[lbl] = np.mean(vecs, axis=0)

    if not predictor.class_centroids:
        predictor.class_centroids = centroids

    print(f"Loaded {len(all_latents)} latents across {len(centroids)} classes")

    # Select a sample to analyze
    sample_idx = 0
    z_sample = torch.tensor(all_latents[sample_idx], dtype=torch.float32,
                            device=device).unsqueeze(0)
    print(f"Sample {sample_idx} (true label: {MANUFACTURING_PROCESSES[all_labels[sample_idx]]})")

    # Run analysis for a target process
    target_idx = 0  # 3-axis CNC
    target_score = 0.90

    analyzer = LatentDirectionAnalyzer(
        predictor.model.predictor,
        class_centroids=predictor.class_centroids,
    )
    explainer = DesignExplainer(
        feature_decoder=predictor.feature_decoder,
        pointcloud_decoder=predictor.pc_decoder,
    )

    result = analyzer.solve_for_target_score(z_sample, target_idx, target_score)
    z_target = result["z_modified"]

    explanation = explainer.explain(
        z_sample, z_target, target_idx, target_score,
        achieved_score=result["achieved_score"],
    )

    # --- UMAP for latent space overview ---
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        all_2d = reducer.fit_transform(all_latents)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        all_2d = reducer.fit_transform(all_latents)
        print("umap not available, using PCA")

    # Project sample and target
    sample_2d = reducer.transform(z_sample.cpu().numpy())
    target_2d = reducer.transform(z_target.cpu().numpy())

    # Key features for the bar chart
    key_features = ["volume mm3", "surface area mm2", "aspect ratio",
                    "compactness", "sphericity", "surface area / volume"]

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: UMAP scatter
    ax = axes[0]
    for lbl in sorted(centroids.keys()):
        mask = all_labels == lbl
        ax.scatter(all_2d[mask, 0], all_2d[mask, 1], s=8, alpha=0.4,
                   label=MANUFACTURING_PROCESSES[lbl])
    ax.plot([sample_2d[0, 0], target_2d[0, 0]],
            [sample_2d[0, 1], target_2d[0, 1]],
            'r->', linewidth=2, markersize=8, label="Optimization path")
    ax.scatter(*sample_2d[0], c='red', s=80, marker='o', edgecolors='white',
               zorder=5, label="Original")
    ax.scatter(*target_2d[0], c='green', s=80, marker='*', edgecolors='white',
               zorder=5, label="Target")
    ax.set_title(f"Latent Space: {MANUFACTURING_PROCESSES[target_idx]} optimization")
    ax.legend(fontsize=7, markerscale=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel 2: Feature change bars
    ax = axes[1]
    if explanation.get("feature_changes"):
        fchanges = explanation["feature_changes"]
        names = []
        pcts = []
        for k in key_features:
            if k in fchanges and abs(fchanges[k].get("pct", 0)) > 1:
                names.append(k.split("(")[0].strip()[:20])
                pcts.append(fchanges[k]["pct"])
        if pcts:
            colors = ['g' if p > 0 else 'r' for p in pcts]
            ax.barh(range(len(pcts)), pcts, color=colors, alpha=0.7)
            ax.set_yticks(range(len(pcts)))
            ax.set_yticklabels(names, fontsize=9)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel("Change (%)")
            ax.set_title("Decoded Feature Changes")
    else:
        ax.text(0.5, 0.5, "Feature decoder not available",
                ha="center", va="center", transform=ax.transAxes)

    # Panel 3: Score evolution (simulated along traversal)
    ax = axes[2]
    n_steps = 20
    scores = []
    if predictor.feature_decoder is not None:
        with torch.no_grad():
            for alpha in np.linspace(0, 1, n_steps):
                z_interp = (1 - alpha) * z_sample + alpha * z_target
                logits = predictor.model.predictor(z_interp)
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                score_val = float(probs[0, target_idx].item())
                scores.append(score_val)
        ax.plot(np.linspace(0, 1, n_steps), scores, 'b-o', markersize=4)
        ax.axhline(target_score, color='g', linestyle='--', alpha=0.7,
                   label=f"Target: {target_score}")
        ax.axhline(scores[0], color='r', linestyle='--', alpha=0.7,
                   label=f"Initial: {scores[0]:.3f}")
        ax.set_xlabel("Interpolation alpha (0=original, 1=target)")
        ax.set_ylabel(f"P({MANUFACTURING_PROCESSES[target_idx]} | z)")
        ax.set_title("Probability Along Traversal")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
