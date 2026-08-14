# =============================================================================
# gmdl.py — Manufacturing Process Predictor (Deep Learning) — Entry Point
# =============================================================================
#
# This file is a thin entry point that delegates to the organised ``gmdl``
# package.  All implementation lives under ``gmdl/``:
#
#   gmdl/constants.py        — process labels, dimension constants
#   gmdl/utils.py            — directory helpers, infinite iterator
#   gmdl/config.py           — configuration dataclasses + CLI argument parsing
#   gmdl/datasets.py         — PyTorch datasets and DataLoader factory
#   gmdl/encoders.py         — GeometryEncoder, ImageEncoder, PointNetEncoder
#   gmdl/losses.py           — InfoNCE contrastive loss
#   gmdl/predictor.py        — ProcessPredictor, ProcessPredictorHybrid
#   gmdl/training.py         — Clock, Trainer (two-phase training loop)
#   gmdl/inference.py        — ManufacturingProcessPredictor, extract_image_features
#   gmdl/cli.py              — CLI entry points (train, predict, analyze, train-decoders)
#   gmdl/decoders.py         — LinearDecoder, PointCloudDecoder (analysis decoders)
#   gmdl/decoder_training.py — DecoderDataset, train_decoders (training loop + metrics)
#   gmdl/analysis.py         — ProcessFitScorer, LatentDirectionAnalyzer, DesignExplainer
#
# The public API is fully backward-compatible:
#   from gmdl import ManufacturingProcessPredictor
#   from gmdl import GeometryEncoder, PointNetEncoder, ImageEncoder
#   from gmdl import GEOM_FEATURE_DIM, MANUFACTURING_PROCESSES
#
# ============================================================================
# QUICK START
# ============================================================================
#
# ── Training ─────────────────────────────────────────────────────────────
# Two-phase (requires paired image+geometry data):
#   python gmdl.py --contrastive_epochs 20 --phase2_epochs 30
#
# Geometry-only training (no images):
#   python gmdl.py --contrastive_epochs 0 --phase2_epochs 50
#
# ── Prediction ───────────────────────────────────────────────────────────
# From a CAD .step file (MLP encoder):
#   python gmdl.py --predict --step part.step \
#                --weights train_log/default/model/latest.pt
#
# From an image (CNN encoder):
#   python gmdl.py --predict --image photo.jpg \
#                --weights train_log/default/model/latest.pt
#
# From a .mat mesh (PointNet encoder):
#   python gmdl.py --predict --mesh part.mat \
#                --weights train_log/default/model/latest.pt
#
# ── Decoder training (for --analyze) ─────────────────────────────────────
#   python gmdl.py --train-decoders --weights train_log/default/model/latest.pt
#
# ── Fit scoring and design analysis ──────────────────────────────────────
# Score a part against all processes:
#   python gmdl.py --analyze --mesh part.mat --weights train_log/default/model/latest.pt
#
# Optimise toward a target fit score for a specific process:
#   python gmdl.py --analyze --mesh part.mat --weights train_log/default/model/latest.pt \
#       --target "3-axis CNC machining" --target-score 0.90
#
# Find best achievable score for a process:
#   python gmdl.py --analyze --mesh part.mat --weights train_log/default/model/latest.pt \
#       --target "3-axis CNC machining" --find-best
#
# With decoder explanations (requires trained decoders + centroids):
#   python gmdl.py --analyze --mesh part.mat --weights train_log/default/model/latest.pt \
#       --decoders train_log/default/model/decoders.pt \
#       --centroids train_log/default/model/centroids.npz \
#       --target "Injection molding" --target-score 0.85
#
# ── Python API ───────────────────────────────────────────────────────────
#   from gmdl import ManufacturingProcessPredictor
#   predictor = ManufacturingProcessPredictor()
#   predictor.load_weights("train_log/default/model/latest.pt")
#   results = predictor.predict(step_path="part.step")
#   # results: [(process_name, confidence), ...] sorted descending
#
# ============================================================================
# DATA FORMAT
# ============================================================================
#
# data/train.json and data/validation.json are JSON arrays of objects:
#
# For --encoder mlp (default):
#   {
#     "image":           "relative/path/to/part.jpg"  (optional, for contrastive)
#     "geom_features":   [float x 64]                 (from hks2gmdl.py or cadhandler)
#     "process_label":   int                           (0-8, index into MANUFACTURING_PROCESSES)
#   }
#
# For --encoder pointnet:
#   {
#     "mesh_file":       "relative/path/to/part.mat"  (MATLAB .mat with point3D variable)
#     "image":           "relative/path/to/part.jpg"  (optional, for contrastive)
#     "process_label":   int                           (0-8, index into MANUFACTURING_PROCESSES)
#   }
#
# ============================================================================
# CHECKPOINTING
# ============================================================================
#
# Main model checkpoints:  train_log/<exp_name>/model/<tag>.pt
# Decoder checkpoints:     train_log/<exp_name>/model/decoders.pt
# Loss logs:               train_log/<exp_name>/log/losses.json
#
# Resume training:   python gmdl.py --continue --ckpt latest
# Reset experiment:  rm -rf train_log/<exp_name>/ (and re-run without --continue)
# =============================================================================

if __name__ == "__main__":
    import sys
    from gmdl.cli import main, predict_cli, analyze_cli, train_decoders_cli

    if "--predict" in sys.argv:
        sys.argv.remove("--predict")
        predict_cli()
    elif "--analyze" in sys.argv:
        sys.argv.remove("--analyze")
        analyze_cli()
    elif "--train-decoders" in sys.argv:
        sys.argv.remove("--train-decoders")
        train_decoders_cli()
    else:
        main()
