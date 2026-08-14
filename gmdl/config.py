# =============================================================================
# gmdl/config.py — Configuration Classes for Training & Inference
# =============================================================================
#
# Three configuration dataclasses control the behaviour of the deep-learning
# pipeline.  None require PyTorch to define; they are plain Python objects
# populated from command-line arguments or from a saved config.json.
#
#   ConfigGeometryEncoder  — hyper-parameters for the geometry / image encoders
#   ConfigProcessPredictor — hyper-parameters for the transformer predictor head
#   ConfigExperiment       — top-level experiment settings (paths, LR, phases)
# =============================================================================

import os
import json
import shutil
import argparse

from gmdl.constants import (
    GEOM_FEATURE_DIM, LATENT_DIM, N_PROCESSES, MAX_SEQ_LEN, MAX_POINTS,
)
from gmdl.utils import ensure_dirs

# ---------------------------------------------------------------------------
# Check whether PyTorch is available (for CUDA setup in ConfigExperiment).
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ConfigGeometryEncoder:
    """Hyper-parameters shared by both the MLP and PointNet geometry encoders,
    as well as the CNN image encoder (``img_size``, ``input_channels``)."""

    def __init__(self):
        self.input_channels = 3          # RGB channels for image encoder
        self.feature_dim = GEOM_FEATURE_DIM  # 64-dim hand-crafted features
        self.latent_dim = LATENT_DIM     # 128-dim shared latent space
        self.n_conv_layers = 4           # number of conv layers in image encoder
        self.img_size = 224              # spatial size of input images
        self.encoder_type = "mlp"        # "mlp" or "pointnet"
        self.max_points = MAX_POINTS     # max vertices per mesh for PointNet

    @classmethod
    def from_dict(cls, d):
        """Construct an instance from a dictionary, ignoring unknown keys."""
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class ConfigProcessPredictor:
    """Hyper-parameters for the transformer-based process prediction head."""

    def __init__(self):
        self.n_processes = N_PROCESSES       # number of output classes (9)
        self.latent_dim = LATENT_DIM         # input / model dimension
        self.n_heads = 4                     # multi-head attention heads
        self.n_layers = 4                    # transformer encoder layers
        self.dim_feedforward = 256           # FFN intermediate dimension
        self.dropout = 0.1                   # dropout probability
        self.max_seq_len = MAX_SEQ_LEN       # positional embedding length

    @classmethod
    def from_dict(cls, d):
        """Construct an instance from a dictionary, ignoring unknown keys."""
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class ConfigExperiment:
    """Top-level experiment configuration.  Combines sub-configs for the
    encoder and predictor, plus training schedule, paths, and two-phase
    contrastive / prediction settings.

    Call ``parse_args()`` to populate from ``sys.argv``; this also creates
    the experiment directories and persists the config to disk.
    """

    def __init__(self, phase="train"):
        self.is_train = phase == "train"
        # Sub-configs
        self.geom_encoder = ConfigGeometryEncoder()
        self.process_predictor = ConfigProcessPredictor()
        # Paths
        self.proj_dir = "train_log"
        self.exp_name = "default"
        self.data_root = "data"
        self.image_root = None  # defaults to data_root if None
        # DataLoader
        self.batch_size = 16
        self.num_workers = 4
        # Optimiser
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.warmup_step = 2000
        # Checkpointing
        self.save_frequency = 5
        self.val_frequency = 50
        self.gpu_ids = "0"
        self.nr_epochs = 50  # legacy; unused in two-phase mode
        # Phase 1 — Contrastive pre-training
        self.contrastive_epochs = 20
        self.contrastive_lr = 3e-4
        self.contrastive_temperature = 0.07
        # Phase 2 — Prediction fine-tuning
        self.phase2_epochs = 30
        self.phase2_lr = 1e-4
        self.phase2_encoder_lr = 1e-5
        # Loss weights
        self.contrastive_weight = 3.0
        self.prediction_weight = 1.0
        self.kl_weight = 0.0          # disabled for discriminative tasks
        self.raw_align_weight = 1.5
        self.center_loss_weight = 0.05    # weight for center loss (0 = disabled)
        self.center_loss_momentum = 0.9   # EMA momentum for running centroids
        # ── Phase 3 — Decoder training ──────────────────────────────────
        # Hyper-parameters for training the feature and point-cloud
        # decoders used by the analysis pipeline (--analyze).
        # These are only relevant after the main model is fully trained.
        self.decoder_epochs = 20           # epochs for decoder training
        self.decoder_lr = 1e-3             # learning rate for decoder training
        self.decoder_lr_patience = 3       # epochs of plateau before LR halves
        self.decoder_hidden = 256          # hidden dimension for decoders
        self.decoder_batch_size = 32       # batch size for decoder training

        # Encoder selection
        self.encoder_type = "mlp"     # "mlp" or "pointnet"
        self.max_points = MAX_POINTS

    # ------------------------------------------------------------------
    # CLI argument parsing
    # ------------------------------------------------------------------

    def parse_args(self):
        """Parse command-line arguments, update self in-place, create
        experiment directories, and persist config to ``config.json``."""
        parser = argparse.ArgumentParser()
        # Paths
        parser.add_argument("--proj_dir", type=str, default=self.proj_dir)
        parser.add_argument("--data_root", type=str, default=self.data_root)
        parser.add_argument("--exp_name", type=str, default=self.exp_name)
        parser.add_argument("--image_root", type=str, default=None,
                            help="Root dir for images (default: same as data_root)")
        # DataLoader
        parser.add_argument("--batch_size", type=int, default=self.batch_size)
        parser.add_argument("--num_workers", type=int, default=self.num_workers)
        # Optimiser / schedule
        parser.add_argument("--nr_epochs", type=int, default=self.nr_epochs)
        parser.add_argument("--lr", type=float, default=self.lr)
        parser.add_argument("--weight_decay", type=float, default=self.weight_decay)
        parser.add_argument("--grad_clip", type=float, default=self.grad_clip)
        parser.add_argument("--warmup_step", type=int, default=self.warmup_step)
        # Checkpointing
        parser.add_argument("--save_frequency", type=int, default=self.save_frequency)
        parser.add_argument("--val_frequency", type=int, default=self.val_frequency)
        parser.add_argument("-g", "--gpu_ids", type=str, default=self.gpu_ids)
        parser.add_argument("--continue", dest="cont", action="store_true")
        parser.add_argument("--ckpt", type=str, default="latest")
        # Phase 1 — Contrastive
        parser.add_argument("--contrastive_epochs", type=int, default=self.contrastive_epochs)
        parser.add_argument("--contrastive_lr", type=float, default=self.contrastive_lr)
        parser.add_argument("--contrastive_temperature", type=float, default=self.contrastive_temperature)
        # Phase 2 — Prediction
        parser.add_argument("--phase2_epochs", type=int, default=self.phase2_epochs)
        parser.add_argument("--phase2_lr", type=float, default=self.phase2_lr)
        parser.add_argument("--phase2_encoder_lr", type=float, default=self.phase2_encoder_lr)
        # Encoder
        parser.add_argument("--encoder", type=str, choices=["mlp", "pointnet"],
                            default=self.encoder_type, dest="encoder_type",
                            help="Geometry encoder: mlp (hand-crafted) or pointnet (raw vertices)")
        parser.add_argument("--max_points", type=int, default=self.max_points,
                            help="Max vertices per mesh for PointNet (default: 512)")
        # Loss weights
        parser.add_argument("--kl_weight", type=float, default=self.kl_weight,
                            help="KL divergence weight (default: 0.0)")
        parser.add_argument("--raw_align_weight", type=float, default=self.raw_align_weight,
                            help="Weight for raw latent cosine alignment loss (default: 1.5)")
        parser.add_argument("--center_loss_weight", type=float, default=self.center_loss_weight,
                            help="Weight for center loss (default: 0.05, 0=disabled)")
        parser.add_argument("--center_loss_momentum", type=float, default=self.center_loss_momentum,
                            help="EMA momentum for running centroids (default: 0.9)")

        # Inference-only arguments (not used during training)
        if not self.is_train:
            parser.add_argument("--mode", type=str, choices=["predict", "encode"])
            parser.add_argument("--input", type=str, default=None)
            parser.add_argument("--output", type=str, default=None)

        args = parser.parse_args()

        # Copy parsed args onto self
        for k, v in vars(args).items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Default image_root to data_root
        if self.image_root is None:
            self.image_root = self.data_root

        # Sync encoder settings into the sub-config
        self.geom_encoder.encoder_type = self.encoder_type
        self.geom_encoder.max_points = self.max_points

        # Build experiment directory structure
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if self.is_train and not getattr(self, "cont", False) and os.path.exists(self.exp_dir):
            response = input("Experiment log/model already exists, overwrite? (y/n) ")
            if response != "y":
                exit()
            shutil.rmtree(self.exp_dir)
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        ensure_dirs([self.log_dir, self.model_dir])

        # Set CUDA_VISIBLE_DEVICES if torch is available
        if self.gpu_ids is not None and _TORCH_AVAILABLE:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)

        # Persist config to disk for reproducibility
        if self.is_train:
            with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
                json.dump(self._to_dict(), f, indent=2)

        return args

    def _to_dict(self):
        """Serialize to a plain dict suitable for JSON storage."""
        d = {}
        for k, v in self.__dict__.items():
            if k in ("geom_encoder", "process_predictor"):
                d[k] = v.__dict__
            else:
                d[k] = v
        return d
