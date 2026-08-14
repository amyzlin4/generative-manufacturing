# =============================================================================
# gmdl/predictor.py — Transformer-Based Process Prediction Heads
# =============================================================================
#
# Two predictor modules:
#
#   ProcessPredictor:
#       Takes a single 128-dim latent vector, projects it, expands to
#       max_seq_len positions via learned positional embeddings, and feeds
#       through a transformer encoder.  The output is [B, seq_len, n_processes]
#       logits — one per sequence position.  At inference, only the first
#       position's prediction is typically used, but multi-position output
#       enables ensemble-like behaviour during training.
#
#   ProcessPredictorHybrid:
#       Combines a geometry encoder (MLP or PointNet) with the
#       ProcessPredictor into a single module.  Used by the high-level
#       ManufacturingProcessPredictor inference API for convenient
#       weight loading and forward passes.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmdl.config import ConfigGeometryEncoder, ConfigProcessPredictor


class ProcessPredictor(nn.Module):
    """Transformer-based prediction head that maps a latent vector to
    manufacturing process logits.

    Architecture:
        L2-normalise z → Linear projection → expand to [B, max_seq_len, latent_dim]
        → add learned positional embeddings → transformer encoder layers
        → Linear output projection → [B, max_seq_len, n_processes] logits

    The positional embedding expansion effectively creates a small "sequence"
    from a single vector, allowing the transformer to attend across positions
    and produce multiple predictions per sample.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Projection from latent_dim → latent_dim (identity-sized, but learned)
        self.project = nn.Linear(cfg.latent_dim, cfg.latent_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )

        # Learned positional embeddings for the sequence expansion
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.max_seq_len, cfg.latent_dim)
        )

        # Output head: latent_dim → n_processes
        self.out_proj = nn.Linear(cfg.latent_dim, cfg.n_processes)

    def forward(self, z):
        """
        Args:
            z: [B, latent_dim] latent vector from any encoder

        Returns:
            logits: [B, max_seq_len, n_processes] raw prediction scores
        """
        # Normalise and project
        z = F.normalize(z, dim=-1)
        z = self.project(z).unsqueeze(1)  # [B, 1, latent_dim]

        # Expand to max_seq_len positions and add positional embeddings
        z_expanded = (
            z.expand(-1, self.cfg.max_seq_len, -1)
            + self.pos_embed[:, :self.cfg.max_seq_len, :]
        )

        # Transformer processes the sequence
        out = self.transformer(z_expanded)  # [B, max_seq_len, latent_dim]

        # Project to class logits
        return self.out_proj(out)  # [B, max_seq_len, n_processes]


class ProcessPredictorHybrid(nn.Module):
    """Convenience wrapper that pairs a geometry encoder with the predictor.

    Selects PointNetEncoder or GeometryEncoder based on ``geom_cfg.encoder_type``.
    Used by ManufacturingProcessPredictor for clean weight loading and inference.

    forward() returns a dict with keys:
        ``logits``       — [B, max_seq_len, n_processes]
        ``latent``       — [B, latent_dim] raw encoder output
        ``kl_loss``      — scalar (only if return_kl=True)
    """

    def __init__(self, geom_cfg, proc_cfg):
        super().__init__()

        # Import here to avoid circular imports at module level
        from gmdl.encoders import GeometryEncoder, PointNetEncoder

        if getattr(geom_cfg, "encoder_type", "mlp") == "pointnet":
            self.geom_encoder = PointNetEncoder(geom_cfg)
        else:
            self.geom_encoder = GeometryEncoder(geom_cfg)

        self.predictor = ProcessPredictor(proc_cfg)

    def forward(self, geom_input, n_valid_points=None, return_kl=False):
        """
        Args:
            geom_input:      [B, feature_dim] (MLP) or [B, N, 3] (PointNet)
            n_valid_points:  [B] int tensor (PointNet only)
            return_kl:       if True, include KL divergence in output

        Returns:
            dict with ``logits``, ``latent``, and optionally ``kl_loss``
        """
        if return_kl:
            z, kl = self.geom_encoder(
                geom_input, n_valid_points=n_valid_points, return_kl=True
            )
        else:
            z = self.geom_encoder(geom_input, n_valid_points=n_valid_points)

        logits = self.predictor(z)

        res = {"logits": logits, "latent": z}
        if return_kl:
            res["kl_loss"] = kl
        return res
