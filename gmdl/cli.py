# =============================================================================
# gmdl/cli.py — Command-Line Interface (Training, Prediction, Analysis)
# =============================================================================
#
# Entry points for the CLI modes dispatched by gmdl.py:
#
#   Training (default):
#       python gmdl.py                    # Phase 1 (contrastive) + Phase 2 (prediction)
#       python gmdl.py --contrastive_epochs 0 --phase2_epochs 50  # geometry-only
#
#   Prediction:
#       python gmdl.py --predict --step part.step --weights path/to/model.pt
#       python gmdl.py --predict --mesh part.mat  --weights path/to/model.pt
#       python gmdl.py --predict --image photo.jpg --weights path/to/model.pt
#
#   Analysis (fit scoring + latent optimisation + design explanation):
#       python gmdl.py --analyze --mesh part.mat --weights path/to/model.pt
#
#   Decoder training (required before --analyze explanations work):
#       python gmdl.py --train-decoders --weights path/to/model.pt
#
# This module contains main(), predict_cli(), analyze_cli(), and
# train_decoders_cli(), which are called by the gmdl.py entry-point script.
# They are defined here so the heavy torch imports only happen when these
# functions are actually called.
# =============================================================================

import os
import json

import torch

from gmdl.config import ConfigExperiment
from gmdl.training import Trainer
from gmdl.datasets import get_dataloader
from gmdl.utils import cycle


def main():
    """Training entry point.

    Orchestrates the two-phase training loop:
        Phase 1 — Contrastive pre-training (image + geometry alignment)
        Phase 2 — Prediction fine-tuning (with contrastive regulariser)

    Configuration is parsed from command-line arguments.  Checkpoints are
    saved to ``train_log/<exp_name>/model/``.  Loss logs are written to
    ``train_log/<exp_name>/log/losses.json``.

    Data format: ``data/train.json`` and ``data/validation.json`` — each a
    JSON array of objects with keys ``geom_features`` (or ``mesh_file``),
    ``process_label``, and optionally ``image``.
    """
    cfg = ConfigExperiment(phase="train")
    cfg.parse_args()
    print(f"Experiment: {cfg.exp_name}")
    print(f"Encoder: {cfg.encoder_type.upper()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    from tqdm import tqdm

    trainer = Trainer(cfg)
    if getattr(cfg, "cont", False):
        trainer.load_ckpt(cfg.ckpt)

    train_loader = get_dataloader("train", cfg)
    val_loader = get_dataloader("validation", cfg)
    val_loader_iter = cycle(val_loader)
    trainer.show_parameters()
    loss_log_path = os.path.join(cfg.log_dir, "losses.json")

    # Initialise loss log (persisted across both phases)
    all_loss_records = []
    if getattr(cfg, "cont", False) and os.path.exists(loss_log_path):
        with open(loss_log_path) as f:
            all_loss_records = json.load(f)

    # --- Phase 1: Contrastive pre-training (image + geometry alignment) ---
    if cfg.contrastive_epochs > 0:
        trainer.optimizer = trainer._build_optimizer_phase1()
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=cfg.contrastive_epochs
        )

        start_epoch = trainer.clock.epoch if getattr(cfg, "cont", False) else 0
        print(f"\n=== Phase 1: Contrastive pre-training ({cfg.contrastive_epochs} epochs) ===")

        for e in range(start_epoch, cfg.contrastive_epochs):
            epoch_losses = []
            pbar = tqdm(train_loader)
            for b, batch in enumerate(pbar):
                losses = trainer.contrastive_train_func(batch)
                pbar.set_description(f"CONTRASTIVE[{e}][{b}]")
                pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

                # Record training losses
                record = {"phase": 1, "epoch": e, "step": trainer.clock.step}
                for k, v in losses.items():
                    record[k] = v.item()
                epoch_losses.append(record)

                # Periodic validation
                if trainer.clock.step % cfg.val_frequency == 0:
                    val_losses = trainer.contrastive_val_func(next(val_loader_iter))
                    vrec = {"phase": 1, "epoch": e, "step": trainer.clock.step, "split": "val"}
                    for k, v in val_losses.items():
                        vrec[k] = v.item()
                    epoch_losses.append(vrec)
                trainer.clock.tick()

            # Compute epoch-level means across training steps
            train_keys = [k for k in epoch_losses[0]
                          if k not in ("phase", "epoch", "step", "split")]
            mean_rec = {"phase": 1, "epoch": e, "split": "train_mean"}
            for k in train_keys:
                vals = [r[k] for r in epoch_losses
                        if r.get("split") != "val" and k in r]
                if vals:
                    mean_rec[k] = sum(vals) / len(vals)

            # Persist loss log
            all_loss_records.extend(epoch_losses)
            all_loss_records.append(mean_rec)
            with open(loss_log_path, "w") as f:
                json.dump(all_loss_records, f)

            trainer.scheduler.step()
            trainer.clock.tock()
            if trainer.clock.epoch % cfg.save_frequency == 0:
                trainer.save_ckpt()
            trainer.save_ckpt("latest")

        print("Phase 1 complete.")

    # --- Phase 2: Prediction fine-tuning (with contrastive regulariser) ---
    if cfg.phase2_epochs > 0:
        trainer.optimizer = trainer._build_optimizer_phase2()
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=cfg.phase2_epochs
        )
        start_epoch = (max(0, trainer.clock.epoch - cfg.contrastive_epochs)
                       if getattr(cfg, "cont", False) else 0)
        print(f"\n=== Phase 2: Prediction fine-tuning ({cfg.phase2_epochs} epochs) ===")

        for e in range(start_epoch, cfg.phase2_epochs):
            epoch_losses = []
            pbar = tqdm(train_loader)
            for b, batch in enumerate(pbar):
                _, losses = trainer.train_func(batch)
                pbar.set_description(f"PREDICT[{e}][{b}]")
                pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

                # Record training losses
                record = {"phase": 2, "epoch": e, "step": trainer.clock.step}
                for k, v in losses.items():
                    record[k] = v.item()
                epoch_losses.append(record)

                # Periodic validation
                if trainer.clock.step % cfg.val_frequency == 0:
                    _, val_losses = trainer.val_func(next(val_loader_iter))
                    vrec = {"phase": 2, "epoch": e, "step": trainer.clock.step, "split": "val"}
                    for k, v in val_losses.items():
                        vrec[k] = v.item()
                    epoch_losses.append(vrec)
                trainer.clock.tick()

            # Compute epoch-level means
            train_keys = [k for k in epoch_losses[0]
                          if k not in ("phase", "epoch", "step", "split")]
            mean_rec = {"phase": 2, "epoch": e, "split": "train_mean"}
            for k in train_keys:
                vals = [r[k] for r in epoch_losses
                        if r.get("split") != "val" and k in r]
                if vals:
                    mean_rec[k] = sum(vals) / len(vals)

            # Persist loss log
            all_loss_records.extend(epoch_losses)
            all_loss_records.append(mean_rec)
            with open(loss_log_path, "w") as f:
                json.dump(all_loss_records, f)

            trainer.scheduler.step()
            trainer.clock.tock()
            if trainer.clock.epoch % cfg.save_frequency == 0:
                trainer.save_ckpt()
            trainer.save_ckpt("latest")

        print("Phase 2 complete.")


def predict_cli():
    """CLI entry point for prediction.

    Usage:
        python gmdl.py --predict --step part.step --weights train_log/default/model/latest.pt
        python gmdl.py --predict --mesh part.mat  --weights train_log/default/model/latest.pt
        python gmdl.py --predict --image photo.jpg --weights train_log/default/model/latest.pt

    Use ``--config`` to specify a custom config.json and ``--top_k`` to
    control how many results are printed.
    """
    import argparse
    from gmdl.inference import ManufacturingProcessPredictor

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--mesh", type=str, default=None, help="Path to .mat mesh file (PointNet)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--weights", type=str, default="train_log/default/model/latest.pt")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predictor = ManufacturingProcessPredictor(cfg_path=args.config)
    if os.path.exists(args.weights):
        predictor.load_weights(args.weights)

    results = predictor.predict(
        step_path=args.step, image_path=args.image, mesh_path=args.mesh
    )
    print(f"\nTop-{args.top_k} recommended manufacturing processes:")
    for i, (name, conf) in enumerate(results[:args.top_k]):
        print(f"  {i + 1}. {name} (confidence: {conf:.3f})")


def analyze_cli():
    """CLI entry point for fit scoring and design analysis.

    Usage:
        # Show fit scores for all processes
        python gmdl.py --analyze --mesh part.mat --weights model.pt

        # Reach a target fit score
        python gmdl.py --analyze --mesh part.mat --weights model.pt \\
            --target "3-axis CNC machining" --target-score 0.90

        # Find best achievable score
        python gmdl.py --analyze --mesh part.mat --weights model.pt \\
            --target "3-axis CNC machining" --find-best

        # With decoder explanations (requires trained decoders)
        python gmdl.py --analyze --mesh part.mat --weights model.pt \\
            --decoders train_log/default/model/decoders.pt \\
            --centroids centroids.npz \\
            --target "Injection molding" --target-score 0.85
    """
    import json
    import argparse
    from gmdl.inference import ManufacturingProcessPredictor

    parser = argparse.ArgumentParser(description="Fit scoring and design analysis")
    parser.add_argument("--step", type=str, default=None, help="Path to STEP file")
    parser.add_argument("--mesh", type=str, default=None, help="Path to .mat mesh file")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--weights", type=str, default="train_log/default/model/latest.pt",
                        help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--decoders", type=str, default=None,
                        help="Decoder checkpoint path (.pt)")
    parser.add_argument("--centroids", type=str, default=None,
                        help="Pre-computed centroids (.npz)")
    parser.add_argument("--latent-refs", type=str, default=None,
                        help="Pre-computed KNN latent references (.npz)")
    parser.add_argument("--target", type=str, default=None,
                        help="Target process name (e.g. '3-axis CNC machining')")
    parser.add_argument("--target-score", type=float, default=None,
                        help="Target fit score (0.0-1.0)")
    parser.add_argument("--find-best", action="store_true",
                        help="Find max achievable score for target")
    parser.add_argument("--processes", type=str, default=None,
                        help="Comma-separated list of active process indices "
                             "(overrides auto-detection, e.g. '0,2')")
    parser.add_argument("--save-centroids", type=str, default=None,
                        help="Save computed centroids to this .npz path")
    parser.add_argument("--save-latent-refs", type=str, default=None,
                        help="Save computed KNN latent references to this .npz path")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print JSON output")
    parser.add_argument("--output-viz", type=str, default=None,
                        help="Output stem for displacement visualization "
                             "(saves <stem>.ply, <stem>_mesh.ply, <stem>.png)")
    args = parser.parse_args()

    predictor = ManufacturingProcessPredictor(cfg_path=args.config)
    if os.path.exists(args.weights):
        predictor.load_weights(args.weights)

    # Load decoders if provided
    if args.decoders:
        predictor.load_decoders(args.decoders)
        print(f"Loaded decoders from {args.decoders}")

    # Derive a default centroids path alongside the weights checkpoint
    default_centroids_path = os.path.join(
        os.path.dirname(os.path.abspath(args.weights)),
        "centroids.npz",
    )
    default_latent_refs_path = os.path.join(
        os.path.dirname(os.path.abspath(args.weights)),
        "latent_refs.npz",
    )

    # Prefer KNN latent references for manufacturability fit scoring.
    if args.latent_refs:
        predictor.load_latent_refs(args.latent_refs)
    elif os.path.exists(default_latent_refs_path):
        print(f"Loading cached KNN latent refs from {default_latent_refs_path}")
        predictor.load_latent_refs(default_latent_refs_path)

    if predictor.knn_scorer is None:
        data_root = getattr(predictor, 'data_root', 'data')
        manifest = os.path.join(data_root, 'train.json')
        if os.path.exists(manifest):
            print(f"Computing KNN latent refs from {manifest}...")
            predictor.compute_latent_refs(manifest_path=manifest)
            predictor.save_latent_refs(default_latent_refs_path)

    # Centroids are retained only as a fallback when KNN refs are unavailable.
    if predictor.knn_scorer is None:
        if args.centroids:
            predictor.load_centroids(args.centroids)
        elif os.path.exists(default_centroids_path):
            print(f"Loading cached centroids from {default_centroids_path}")
            predictor.load_centroids(default_centroids_path)

        if not predictor.class_centroids:
            data_root = getattr(predictor, 'data_root', 'data')
            manifest = os.path.join(data_root, 'train.json')
            if os.path.exists(manifest):
                print(f"Computing centroids from {manifest}...")
                predictor.compute_centroids(manifest_path=manifest)
                predictor.save_centroids(default_centroids_path)

    # Override active indices from CLI if provided
    if args.processes is not None:
        predictor.active_indices = [int(x.strip()) for x in args.processes.split(",")]
        print(f"Using active process indices: {predictor.active_indices}")

    # Save centroids to a custom path if explicitly requested
    if args.save_centroids and predictor.class_centroids:
        predictor.save_centroids(args.save_centroids)
    if args.save_latent_refs and predictor.knn_scorer is not None:
        predictor.save_latent_refs(args.save_latent_refs)

    result = predictor.predict_with_analysis(
        step_path=args.step,
        mesh_path=args.mesh,
        image_path=args.image,
        target_process=args.target,
        target_score=args.target_score,
        find_best=args.find_best,
        output_viz=args.output_viz,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))


def train_decoders_cli():
    """CLI entry point for training the feature + point cloud decoders.

    Usage:
        python gmdl.py --train-decoders --weights train_log/default/model/latest.pt
    """
    import argparse
    from gmdl.encoders import PointNetEncoder, GeometryEncoder
    from gmdl.config import ConfigGeometryEncoder
    from gmdl.decoder_training import get_decoder_dataloader, train_decoders

    parser = argparse.ArgumentParser(description="Train decoders for analysis")
    parser.add_argument("--weights", type=str, default="train_log/default/model/latest.pt",
                        help="Trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Decoder training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Decoder learning rate")
    parser.add_argument("--lr-patience", type=int, default=3,
                        help="Epochs of plateau before LR halves (default: 3)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data root (default: from checkpoint config)")
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("cfg", {})

    data_root = args.data_root or saved_cfg.get("data_root", "data")
    use_pointnet = saved_cfg.get("geom_encoder", {}).get("encoder_type", "mlp") == "pointnet"
    max_points = saved_cfg.get("geom_encoder", {}).get("max_points", 512)

    geom_cfg = ConfigGeometryEncoder()
    geom_cfg.encoder_type = "pointnet" if use_pointnet else "mlp"
    geom_cfg.max_points = max_points

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build encoder and load weights
    if use_pointnet:
        encoder = PointNetEncoder(geom_cfg).to(device)
    else:
        encoder = GeometryEncoder(geom_cfg).to(device)
    encoder.load_state_dict(ckpt["geom_encoder"])
    encoder.eval()
    print(f"Loaded encoder from {args.weights}")

    # Build minimal config for decoder training
    class DecoderCfg:
        pass

    cfg = DecoderCfg()
    cfg.data_root = data_root
    cfg.decoder_epochs = args.epochs
    cfg.decoder_lr = args.lr
    cfg.decoder_lr_patience = args.lr_patience
    cfg.decoder_batch_size = 32
    cfg.decoder_hidden = 256
    cfg.max_points = max_points
    cfg.num_workers = 4
    cfg.geom_encoder = geom_cfg

    train_loader = get_decoder_dataloader(cfg, "train")
    val_loader = get_decoder_dataloader(cfg, "validation")

    feature_decoder, pc_decoder, standardizer = train_decoders(
        encoder, train_loader, cfg, device=device, val_loader=val_loader
    )

    exp_dir = os.path.dirname(os.path.dirname(args.weights))
    decoder_path = os.path.join(exp_dir, "model", "decoders.pt")
    torch.save({
        "feature_decoder": feature_decoder.state_dict(),
        "pointcloud_decoder": pc_decoder.state_dict(),
        "standardizer_mean": standardizer.mean,
        "standardizer_std": standardizer.std,
    }, decoder_path)
    print(f"Decoders saved to {decoder_path}")
