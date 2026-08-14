# =============================================================================
# gmdl/decoders.py — Latent Decoder Networks for Analysis
# =============================================================================
#
# Two lightweight MLP decoders that map the 128-dim latent space back to
# interpretable physical quantities:
#
#   LinearDecoder (128 → 64):
#       Reconstructs the 64 hand-crafted geometry features (volume, aspect
#       ratio, PCA eigenvalues, histogram bins, etc.).  During analysis, the
#       original and modified latents are decoded and their feature deltas
#       are reported as "what changed" in physical terms.
#
#   PointCloudDecoder (128 → [512, 3]):
#       Reconstructs the input vertex positions.  Only usable when the
#       original input was a mesh (PointNet encoder) — image and STEP inputs
#       do not have a vertex ground-truth to train against.  The reconstructed
#       point cloud shows "what the shape would look like" after a latent edit.
#
# Both decoders are trained simultaneously (summed MSE) in decoder_training.py
# with the encoder frozen.  They are not used during the main model training.
# =============================================================================

import torch
import torch.nn as nn


class LinearDecoder(nn.Module):
    """Linear decoder: 128-dim latent → 64-dim hand-crafted features.

    Trained with MSE loss on (encoder_output, geom_features) pairs.
    During analysis, both the original and modified latent vectors
    are decoded.  The per-feature deltas (absolute and percentage) are
    compared to explain which physical properties a latent edit changed.

    Architecture:
        Linear(128 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 64)

    The two hidden layers give enough capacity to learn the mapping from
    the abstract latent space to the heterogeneous feature space
    (volumes in mm\u00b3, angles in rad, histogram bins, etc.).
    """

    def __init__(self, latent_dim=128, feature_dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, z):
        """Decode a latent vector to 64-dim feature space.

        Args:
            z: [B, latent_dim] latent vector(s), typically the output of
               the geometry encoder (PointNetEncoder or GeometryEncoder).

        Returns:
            [B, feature_dim] decoded geometry features.
        """
        return self.net(z)


class PointCloudDecoder(nn.Module):
    """MLP decoder: 128-dim latent → [max_points, 3] vertices.

    Trained with MSE loss on (encoder_output, original_vertices) pairs.
    Only meaningful when the original input was a mesh — image and STEP
    inputs do not have vertex ground-truth and should set this to None.

    Architecture:
        Linear(128 → 256) → ReLU → Linear(256 → 256) → ReLU →
        Linear(256 → max_points × 3) → reshape to [B, max_points, 3]

    Note:
        The output vertices are *unordered* — the MSE loss operates on
        position-wise correspondence, not shape matching.  For a more
        shape-aware evaluation, use Chamfer distance (see
        decoder_training.chamfer_distance).
    """

    def __init__(self, latent_dim=128, max_points=512, hidden=256):
        super().__init__()
        self.max_points = max_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max_points * 3),
        )

    def forward(self, z):
        """Decode a latent vector to a point cloud.

        Args:
            z: [B, latent_dim] latent vector(s).

        Returns:
            [B, max_points, 3] decoded vertex positions.
        """
        out = self.net(z)
        return out.view(-1, self.max_points, 3)
