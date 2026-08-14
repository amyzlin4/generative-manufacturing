# =============================================================================
# gmdl/training.py — Training Loop, Checkpointing, and Prediction Helpers
# =============================================================================
#
# Contains the core training infrastructure:
#
#   Clock — simple epoch/step counter with state_dict serialisation.
#
#   Trainer — orchestrates the full two-phase training pipeline:
#       Phase 1 (contrastive):  aligns image and geometry embeddings via
#           InfoNCE + raw latent cosine alignment.  Both encoders are
#           trained; the predictor is frozen.
#       Phase 2 (prediction):   trains the predictor on geometry features
#           with a contrastive regulariser so the latent space stays aligned.
#           Encoder LR is reduced to prevent forgetting.
#
#   The Trainer also provides:
#       - predict() / predict_image() for single-sample inference
#       - save_ckpt() / load_ckpt() for checkpointing
#       - show_parameters() for model summary
# =============================================================================

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmdl.constants import LATENT_DIM, MAX_POINTS
from gmdl.encoders import GeometryEncoder, PointNetEncoder, ImageEncoder
from gmdl.predictor import ProcessPredictor
from gmdl.losses import InfoNCELoss


class Clock:
    """Simple counter tracking the current epoch and training step.

    Provides ``state_dict()`` / ``load_state_dict()`` for checkpoint
    serialisation alongside the model and optimizer.
    """

    def __init__(self):
        self.epoch = 0
        self.step = 0

    def tick(self):
        """Advance by one training step."""
        self.step += 1

    def tock(self):
        """Advance by one epoch."""
        self.epoch += 1

    def state_dict(self):
        return {"epoch": self.epoch, "step": self.step}

    def load_state_dict(self, d):
        self.epoch = d["epoch"]
        self.step = d["step"]


class Trainer:
    """Full training orchestrator.

    To reset weights:  delete checkpoint .pt files and re-run without --continue.
    To retrain:        answer "y" to overwrite prompt (or delete exp dir).
    To resume:         python gmdl.py --continue --ckpt latest
    To predict:        trainer.predict(geom_features_np)

    Attributes:
        geom_encoder:  GeometryEncoder or PointNetEncoder
        image_encoder: ImageEncoder (CNN)
        predictor:     ProcessPredictor (transformer head)
        info_nce:      InfoNCELoss for contrastive alignment
        criterion:     CrossEntropyLoss for classification
        clock:         epoch/step counter
        optimizer:     current AdamW optimiser (changes between phases)
        scheduler:     cosine annealing LR scheduler (changes between phases)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_pointnet = getattr(cfg, "encoder_type", "mlp") == "pointnet"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate geometry encoder (MLP or PointNet based on config)
        if self.use_pointnet:
            self.geom_encoder = PointNetEncoder(cfg.geom_encoder).to(self.device)
        else:
            self.geom_encoder = GeometryEncoder(cfg.geom_encoder).to(self.device)

        # Image encoder (CNN) and process predictor (transformer)
        self.image_encoder = ImageEncoder(cfg.geom_encoder).to(self.device)
        self.predictor = ProcessPredictor(cfg.process_predictor).to(self.device)

        # Loss functions
        self.info_nce = InfoNCELoss(temperature=cfg.contrastive_temperature)
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.clock = Clock()
        self.best_val_loss = float("inf")
        # Running centroids for center loss (lazy-initialised)
        self.center_centroids = None
        self.n_processes = getattr(getattr(cfg, 'process_predictor', None),
                                   'n_processes', 9)

        # Start with Phase 1 optimizer (contrastive pre-training)
        self.optimizer = self._build_optimizer_phase1()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.contrastive_epochs
        )

    # ------------------------------------------------------------------
    # Geometry input extraction (handles both MLP and PointNet formats)
    # ------------------------------------------------------------------

    def _get_geom_input(self, batch, indices=None):
        """Extract geometry input from a batch dict.

        For PointNet: returns (vertices_tensor, n_valid_points_tensor)
        For MLP:      returns (geom_features_tensor, None)

        If *indices* is provided, the tensors are gathered to those indices
        (used for selecting only paired samples in contrastive training).
        """
        if self.use_pointnet:
            geom = batch["vertices"].to(self.device)
            n_valid = batch["n_valid_points"]
            if indices is not None:
                geom = geom[indices]
                n_valid = n_valid[indices]
            return geom, n_valid.to(self.device)
        else:
            geom = batch["geom_features"].to(self.device)
            if indices is not None:
                geom = geom[indices]
            return geom, None

    # ------------------------------------------------------------------
    # Optimizer builders (one per training phase)
    # ------------------------------------------------------------------

    def _build_optimizer_phase1(self):
        """Phase 1 optimizer: jointly trains both encoders (geometry + image)."""
        return torch.optim.AdamW(
            list(self.geom_encoder.parameters()) + list(self.image_encoder.parameters()),
            lr=self.cfg.contrastive_lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _build_optimizer_phase2(self):
        """Phase 2 optimizer: trains the predictor with a lower LR for the
        geometry encoder to prevent catastrophic forgetting."""
        return torch.optim.AdamW([
            {"params": self.geom_encoder.parameters(), "lr": self.cfg.phase2_encoder_lr},
            {"params": self.predictor.parameters(), "lr": self.cfg.phase2_lr},
        ], weight_decay=self.cfg.weight_decay)

    # ------------------------------------------------------------------
    # Phase 1 — Contrastive training
    # ------------------------------------------------------------------

    def contrastive_train_func(self, batch):
        """One step of contrastive pre-training.

        Only samples with paired images contribute to the loss.  The loss
        combines InfoNCE (on projected embeddings) and a raw latent cosine
        alignment term to keep the un-projected encoder outputs aligned.

        Returns:
            dict with ``contrastive_loss`` and ``raw_align_loss`` scalars
        """
        self.geom_encoder.train()
        self.image_encoder.train()

        # Select only samples that have paired images
        has_image = batch["has_image"].to(self.device)
        paired = has_image.nonzero(as_tuple=False).squeeze(1)
        if paired.numel() < 2:
            return {"contrastive_loss": torch.tensor(0.0, device=self.device)}

        geom, n_valid = self._get_geom_input(batch, paired)
        images = batch["image_tensor"].to(self.device)[paired]

        # Encode both modalities
        z_geom = self.geom_encoder(geom, n_valid_points=n_valid)
        z_geom_proj = self.geom_encoder.project(z_geom)
        z_img = self.image_encoder(images)
        z_img_proj = self.image_encoder.project(z_img)

        # InfoNCE on projected embeddings
        info_nce = self.info_nce(z_img_proj, z_geom_proj)

        # Raw latent alignment: force encoder outputs to be directionally similar
        raw_align = 1.0 - F.cosine_similarity(
            F.normalize(z_geom, dim=-1), F.normalize(z_img, dim=-1)
        ).mean()

        loss = info_nce + self.cfg.raw_align_weight * raw_align

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.geom_encoder.parameters()) + list(self.image_encoder.parameters()),
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        return {"contrastive_loss": info_nce, "raw_align_loss": raw_align}

    def contrastive_val_func(self, batch):
        """One step of contrastive validation (no gradient computation).

        Returns:
            dict with ``val_contrastive_loss`` and ``val_raw_align_loss``
        """
        self.geom_encoder.eval()
        self.image_encoder.eval()
        with torch.no_grad():
            has_image = batch["has_image"].to(self.device)
            paired = has_image.nonzero(as_tuple=False).squeeze(1)
            if paired.numel() < 2:
                return {
                    "val_contrastive_loss": torch.tensor(0.0, device=self.device),
                    "val_raw_align_loss": torch.tensor(0.0, device=self.device),
                }
            geom, n_valid = self._get_geom_input(batch, paired)
            images = batch["image_tensor"].to(self.device)[paired]

            z_geom = self.geom_encoder(geom, n_valid_points=n_valid)
            z_geom_proj = self.geom_encoder.project(z_geom)
            z_img = self.image_encoder(images)
            z_img_proj = self.image_encoder.project(z_img)

            info_nce = self.info_nce(z_img_proj, z_geom_proj)
            raw_align = 1.0 - F.cosine_similarity(
                F.normalize(z_geom, dim=-1), F.normalize(z_img, dim=-1)
            ).mean()

        return {"val_contrastive_loss": info_nce, "val_raw_align_loss": raw_align}

    # ------------------------------------------------------------------
    # Phase 2 — Prediction fine-tuning
    # ------------------------------------------------------------------

    def train_func(self, batch):
        """One step of prediction fine-tuning.

        The geometry encoder feeds into the predictor for classification.
        An optional contrastive regulariser (on paired samples only) keeps
        the latent space aligned with the frozen image encoder.

        Returns:
            (logits, losses_dict) — logits shape [B, seq_len, n_processes]
        """
        self.geom_encoder.train()
        self.image_encoder.eval()
        self.predictor.train()

        geom, n_valid = self._get_geom_input(batch)
        labels = batch["process_label"].to(self.device)
        images = batch["image_tensor"].to(self.device)
        has_image = batch["has_image"].to(self.device)

        # Prediction path: geometry encoder → predictor
        z, kl = self.geom_encoder(geom, n_valid_points=n_valid, return_kl=True)
        logits = self.predictor(z)
        B, S, C = logits.shape
        labels_exp = labels.unsqueeze(1).expand(-1, S)
        ce_loss = self.criterion(logits.reshape(-1, C), labels_exp.reshape(-1))

        # Contrastive regulariser (only on paired samples; image encoder frozen)
        img_indices = has_image.nonzero(as_tuple=False).squeeze(1)
        if img_indices.numel() > 1:
            z_geom_proj = self.geom_encoder.project(z[img_indices])
            with torch.no_grad():
                z_img = self.image_encoder(images[img_indices])
                z_img_proj = self.image_encoder.project(z_img)
            contrastive_loss = self.info_nce(z_img_proj, z_geom_proj)
            raw_align = 1.0 - F.cosine_similarity(
                F.normalize(z[img_indices], dim=-1), F.normalize(z_img, dim=-1)
            ).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
            raw_align = torch.tensor(0.0, device=self.device)

        # Center loss (tightens same-class latent clusters)
        cw = getattr(self.cfg, 'center_loss_weight', 0.0)
        if cw > 0.0:
            latent_dim = z.shape[-1]
            # Lazy-initialise running centroids
            if self.center_centroids is None:
                self.center_centroids = torch.zeros(
                    self.n_processes, latent_dim, device=self.device
                )
            # Compute per-class batch means and update EMA
            with torch.no_grad():
                for k in range(self.n_processes):
                    mask = labels == k
                    if mask.any():
                        batch_mean = z[mask].mean(dim=0)
                        momentum = getattr(self.cfg, 'center_loss_momentum', 0.9)
                        self.center_centroids[k] = (
                            momentum * self.center_centroids[k]
                            + (1 - momentum) * batch_mean
                        )
            # Squared distance to each sample's class centroid
            cent = self.center_centroids[labels]  # [B, latent_dim]
            center_loss = (z - cent.detach()).pow(2).sum(dim=-1).mean()
        else:
            center_loss = torch.tensor(0.0, device=self.device)

        # Weighted combination of all loss terms
        loss = (
            self.cfg.prediction_weight * ce_loss
            + self.cfg.kl_weight * kl
            + self.cfg.contrastive_weight * contrastive_loss
            + self.cfg.raw_align_weight * raw_align
            + cw * center_loss
        )

        # Backward pass with gradient clipping (only encoder + predictor)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.geom_encoder.parameters()) + list(self.predictor.parameters()),
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        return logits, {
            "loss": loss,
            "ce_loss": ce_loss,
            "kl_loss": kl,
            "contrastive_loss": contrastive_loss,
            "raw_align_loss": raw_align,
            "center_loss": center_loss,
        }

    def val_func(self, batch):
        """One step of prediction validation (no gradient computation).

        Returns:
            (logits, val_losses_dict)
        """
        self.geom_encoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            geom, n_valid = self._get_geom_input(batch)
            labels = batch["process_label"].to(self.device)
            z = self.geom_encoder(geom, n_valid_points=n_valid)
            logits = self.predictor(z)
            B, S, C = logits.shape
            labels = labels.unsqueeze(1).expand(-1, S)
            loss = self.criterion(logits.reshape(-1, C), labels.reshape(-1))

            # Center loss on validation (using running centroids from training)
            cw = getattr(self.cfg, 'center_loss_weight', 0.0)
            val_center_loss = torch.tensor(0.0, device=self.device)
            if cw > 0.0 and self.center_centroids is not None:
                cent = self.center_centroids[labels[:, 0]]  # [B, latent_dim]
                val_center_loss = (z - cent).pow(2).sum(dim=-1).mean()

        return logits, {"val_loss": loss, "val_center_loss": val_center_loss}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, geom_features):
        """Run inference on geometry features.

        For MLP:   geom_features is a numpy array of length GEOM_FEATURE_DIM.
        For PointNet: geom_features is a numpy array of shape [N, 3] (vertices).

        Returns:
            (preds, probs) — shape [seq_len] and [seq_len, n_processes]
        """
        self.geom_encoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            arr = np.asarray(geom_features, dtype=np.float32)
            n_valid = None

            if self.use_pointnet:
                if arr.ndim == 2:
                    # [N, 3] vertices → pad to [1, MAX_POINTS, 3]
                    max_pts = self.cfg.geom_encoder.max_points
                    pts = np.zeros((max_pts, 3), dtype=np.float32)
                    n = min(len(arr), max_pts)
                    pts[:n] = arr[:n]
                    geom = torch.from_numpy(pts).unsqueeze(0).to(self.device)
                    n_valid = torch.tensor([n], dtype=torch.long, device=self.device)
                else:
                    geom = torch.from_numpy(arr).unsqueeze(0).to(self.device)
                    n_valid = torch.tensor(
                        [arr.shape[0]], dtype=torch.long, device=self.device
                    )
            else:
                geom = torch.from_numpy(arr).unsqueeze(0).to(self.device)

            z = self.geom_encoder(geom, n_valid_points=n_valid)
            logits = self.predictor(z)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        return preds.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()

    def predict_image(self, image_tensor):
        """Run inference from a preprocessed image tensor [B, 3, 224, 224].

        Returns:
            (preds, probs) — shape [seq_len] and [seq_len, n_processes]
        """
        self.image_encoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            img = image_tensor.to(self.device)
            z = self.image_encoder(img)
            logits = self.predictor(z)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_ckpt(self, tag="latest"):
        """Save all model weights, optimizer state, and training clock."""
        path = os.path.join(self.cfg.model_dir, f"{tag}.pt")
        torch.save({
            "geom_encoder": self.geom_encoder.state_dict(),
            "image_encoder": self.image_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "clock": self.clock.state_dict(),
            "cfg": self.cfg._to_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_ckpt(self, tag="latest"):
        """Restore model weights, optimizer state, and training clock.

        Raises FileNotFoundError if the checkpoint does not exist.
        """
        path = os.path.join(self.cfg.model_dir, f"{tag}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.geom_encoder.load_state_dict(ckpt["geom_encoder"])
        if "image_encoder" in ckpt:
            self.image_encoder.load_state_dict(ckpt["image_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.clock.load_state_dict(ckpt["clock"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

    def show_parameters(self):
        """Print parameter counts for each module."""
        n_geom = sum(p.numel() for p in self.geom_encoder.parameters())
        n_img = sum(p.numel() for p in self.image_encoder.parameters())
        n_pred = sum(p.numel() for p in self.predictor.parameters())
        enc_type = "PointNet" if self.use_pointnet else "MLP"
        print(f"GeometryEncoder ({enc_type}) params: {n_geom:,}")
        print(f"ImageEncoder params:    {n_img:,}")
        print(f"ProcessPredictor params: {n_pred:,}")
        print(f"Total params: {n_geom + n_img + n_pred:,}")
