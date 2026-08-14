#!/usr/bin/env python3
"""
Standalone training loss visualization.

Reads losses.json from a training run and generates a multi-panel plot
showing contrastive loss, prediction loss, and validation metrics.

Usage:
  python plot_losses.py [--exp default] [--output loss_plot.png]
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


def load_losses(path):
    with open(path) as f:
        return json.load(f)


def group_by_epoch(records):
    """Group records by (phase, epoch, split), returning mean values per epoch."""
    buckets = defaultdict(list)
    for r in records:
        phase = r.get("phase", 0)
        epoch = r.get("epoch", 0)
        split = r.get("split", "train")
        key = (phase, epoch, split)
        buckets[key].append(r)

    # Compute means per key
    result = {}
    for key, recs in buckets.items():
        phase, epoch, split = key
        loss_keys = [k for k in recs[0] if k not in ("phase", "epoch", "step", "split")]
        means = {}
        for lk in loss_keys:
            vals = [r[lk] for r in recs if lk in r and isinstance(r[lk], (int, float))]
            if vals:
                means[lk] = sum(vals) / len(vals)
        result[key] = means
    return result


def plot_losses(log_dir, output_path):
    loss_path = os.path.join(log_dir, "losses.json")
    if not os.path.exists(loss_path):
        print(f"No losses.json found at {loss_path}")
        return

    records = load_losses(loss_path)
    epoch_means = group_by_epoch(records)

    # Collect all loss keys that exist
    all_keys = set()
    for means in epoch_means.values():
        all_keys.update(means.keys())

    # Determine which panels we need
    has_phase1 = any(k[0] == 1 for k in epoch_means)
    has_phase2 = any(k[0] == 2 for k in epoch_means)

    panels = []
    if has_phase1:
        panels.append(("Phase 1: Contrastive Pre-training", 1, ["contrastive_loss"]))
    if has_phase2:
        panels.append(("Phase 2: Prediction Training", 2, ["loss", "ce_loss", "contrastive_loss", "raw_align_loss"]))

    # Check for validation losses
    val_keys_available = set()
    for key, means in epoch_means.items():
        if key[2] == "val":
            val_keys_available.update(means.keys())

    n_panels = len(panels)
    if val_keys_available:
        n_panels += 1

    if n_panels == 0:
        print("No loss data to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    for title, phase, loss_keys in panels:
        ax = axes[panel_idx]
        for lk in loss_keys:
            epochs = []
            vals = []
            for (p, ep, sp), means in sorted(epoch_means.items()):
                if p == phase and sp == "train" and lk in means:
                    epochs.append(ep)
                    vals.append(means[lk])
            if epochs:
                ax.plot(epochs, vals, label=lk, linewidth=1.5)

        # Validation losses for this phase
        val_map = {
            "contrastive_loss": "val_contrastive_loss",
            "loss": "val_loss",
        }
        for lk in loss_keys:
            val_lk = val_map.get(lk)
            if val_lk and val_lk in val_keys_available:
                epochs = []
                vals = []
                for (p, ep, sp), means in sorted(epoch_means.items()):
                    if p == phase and sp == "val" and val_lk in means:
                        epochs.append(ep)
                        vals.append(means[val_lk])
                if epochs:
                    ax.plot(epochs, vals, label=val_lk, linewidth=1.5, linestyle="--")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    # Validation summary panel
    if val_keys_available:
        ax = axes[panel_idx]
        for vk in sorted(val_keys_available):
            epochs = []
            vals = []
            for (p, ep, sp), means in sorted(epoch_means.items()):
                if sp == "val" and vk in means:
                    epochs.append(ep)
                    vals.append(means[vk])
            if epochs:
                ax.plot(epochs, vals, label=vk, linewidth=1.5)
        ax.set_title("Validation Losses", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training losses")
    parser.add_argument("--exp", type=str, default="default",
                        help="Experiment name under train_log/ (default: default)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: train_log/<exp>/loss_plot.png)")
    args = parser.parse_args()

    log_dir = os.path.join("train_log", args.exp, "log")
    output = args.output or os.path.join("train_log", args.exp, "loss_plot.png")
    plot_losses(log_dir, output)


if __name__ == "__main__":
    main()
