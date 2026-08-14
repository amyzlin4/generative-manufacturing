# =============================================================================
# gmdl/encoders.py — Neural Network Encoders for Geometry and Images
# =============================================================================
#
# Three encoder modules, all mapping to a shared 128-dim latent space:
#
#   GeometryEncoder (MLP path):
#       64-dim hand-crafted feature vector → VAE-style 128-dim latent.
#       Two-layer MLP with LeakyReLU + BatchNorm, then mu/logvar heads
#       for reparameterisation during training.
#
#   PointNetEncoder (PointNet path):
#       Raw vertex point cloud [B, N, 3] → VAE-style 128-dim latent.
#       Shared Conv1d MLP applied per-point, followed by max-pooling to
#       obtain a global shape descriptor.  Learns its own feature
#       representation directly from point positions — no hand-crafted
#       features required.  Masks padded vertices during max-pool.
#
#   ImageEncoder (CNN path):
#       224×224×3 RGB image → 128-dim latent via strided convolutions.
#       Strided Conv2d stack (kernel 4, stride 2) with BatchNorm + LeakyReLU,
#       then a linear projection to latent_dim.
#
# All encoders expose:
#   forward(x, ...) → z          (encoder output)
#   project(z) → z_proj          (L2-normalised projection for InfoNCE)
#
# During training, the mu/logvar reparameterisation trick is used (VAE-style)
# so that the latent space is smooth and well-regularised.  During evaluation,
# only the mean (mu) is used.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
    """MLP-based encoder for 64-dim hand-crafted geometry features.

    Architecture:
        Linear(64 → 128) → LeakyReLU → BN → Linear(128 → 256) → LeakyReLU → BN
        → mu head (256 → 128)
        → logvar head (256 → 128)
        → projection head (128 → 128, for InfoNCE contrastive loss)

    The ``projection`` head is used only during contrastive pre-training to
    compute InfoNCE loss; the raw latent ``z`` is fed to the predictor.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Shared trunk: feature_dim → 256
        self.fc_in = nn.Sequential(
            nn.Linear(cfg.feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
        )

        # VAE heads: 256 → latent_dim
        self.fc_mu = nn.Linear(256, cfg.latent_dim)
        self.fc_logvar = nn.Linear(256, cfg.latent_dim)

        # Projection head for contrastive loss (not used at inference)
        self.projection = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )

    def forward(self, geom_features, n_valid_points=None, return_kl=False):
        """
        Args:
            geom_features: [B, feature_dim] tensor of geometry features
            n_valid_points: ignored (present for API compatibility with PointNetEncoder)
            return_kl:     if True, also compute and return the KL divergence

        Returns:
            z:      [B, latent_dim] latent vector
            kl:     scalar KL divergence (only if return_kl=True)
        """
        h = self.fc_in(geom_features)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterisation trick during training
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        if return_kl:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            return z, kl
        return z

    def project(self, z):
        """L2-normalise the projection of z for contrastive learning."""
        return F.normalize(self.projection(z), dim=-1)


class ImageEncoder(nn.Module):
    """CNN encoder for 224×224 RGB images.

    Architecture:
        Strided Conv2d stack: 3→32→64→128→256 (kernel=4, stride=2, pad=1)
        with BatchNorm2d + LeakyReLU between layers (except after the last).
        Final spatial size = 224 / 2^4 = 14.
        Linear(256 * 14 * 14 → latent_dim)
        Projection head for InfoNCE (not used at inference).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Build strided convolution stack
        channels = [cfg.input_channels, 32, 64, 128, 256]
        convs = []
        for i in range(len(channels) - 1):
            convs.append(
                nn.Conv2d(channels[i], channels[i + 1],
                          kernel_size=4, stride=2, padding=1, bias=False)
            )
            # Add BN + activation for all but the last conv
            if i < len(channels) - 2:
                convs.append(nn.BatchNorm2d(channels[i + 1]))
                convs.append(nn.LeakyReLU(0.2, inplace=True))
        self.convs = nn.Sequential(*convs)

        # Compute the flattened size after all strided convolutions
        final_size = cfg.img_size // (2 ** (len(channels) - 1))
        self.fc = nn.Linear(channels[-1] * final_size * final_size, cfg.latent_dim)

        # Projection head for contrastive loss
        self.projection = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] image tensor (normalised with ImageNet stats)

        Returns:
            z: [B, latent_dim] latent vector
        """
        x = self.convs(x)
        x = x.view(x.size(0), -1)   # flatten spatial dims
        return self.fc(x)

    def project(self, z):
        """L2-normalise the projection of z for contrastive learning."""
        return F.normalize(self.projection(z), dim=-1)


class PointNetEncoder(nn.Module):
    """PointNet encoder: processes raw vertex point clouds [B, N, 3] into
    a latent embedding.

    Learns features directly from point positions via shared MLPs + max
    pooling, analogous to how the CNN learns from images.  The output
    interface matches GeometryEncoder for seamless swapping.

    Architecture:
        Shared MLP (via Conv1d): 3→64→128→latent_dim, each with BN + ReLU.
        Max-pool over N points → global feature vector [B, latent_dim].
        VAE heads: mu, logvar (for reparameterisation).
        Projection head for InfoNCE.

    Padding mask support:
        When ``n_valid_points`` is provided, padded (zero) points are masked
        out with ``-inf`` before max-pooling so that only real vertices
        contribute to the global feature.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        latent_dim = cfg.latent_dim

        # Shared MLP applied to each point independently (Conv1d for efficiency)
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        # Global feature via max pooling → VAE mu / logvar heads
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # Projection head for contrastive loss
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, vertices, n_valid_points=None, return_kl=False):
        """
        Args:
            vertices:        [B, N, 3] tensor of mesh vertices
            n_valid_points:  [B] int tensor — number of real vertices per sample
            return_kl:       if True, also return the KL divergence

        Returns:
            z:   [B, latent_dim] latent vector
            kl:  scalar KL divergence (only if return_kl=True)
        """
        # [B, N, 3] → [B, 3, N] for Conv1d
        x = vertices.permute(0, 2, 1)
        x = self.mlp(x)  # [B, latent_dim, N]

        # Mask out padded points so max-pool only sees real vertices
        if n_valid_points is not None:
            mask = (
                torch.arange(x.shape[2], device=x.device).unsqueeze(0)
                < n_valid_points.unsqueeze(1)
            )
            x = x.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # Global feature via max pooling
        x = x.max(dim=2)[0]  # [B, latent_dim]

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterisation trick during training
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        if return_kl:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            return z, kl
        return z

    def project(self, z):
        """L2-normalise the projection of z for contrastive learning."""
        return F.normalize(self.projection(z), dim=-1)
