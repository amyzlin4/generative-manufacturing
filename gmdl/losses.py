# =============================================================================
# gmdl/losses.py — Contrastive Loss Functions
# =============================================================================
#
# Contains the InfoNCE loss used to align image and geometry embeddings in
# the shared latent space during Phase 1 (contrastive pre-training).
#
# InfoNCE (Noise Contrastive Estimation):
#   Treats each (image, geometry) pair in the batch as a positive, and
#   all other geometry vectors as negatives (and vice versa).  The loss
#   is the symmetric average of cross-entropy in both directions
#   (image→geometry and geometry→image).
#
#   temperature controls the sharpness of the softmax over similarities;
#   lower values produce harder negatives.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss for aligning two sets of embeddings.

    Given a batch of L2-normalised image embeddings ``img_z`` and geometry
    embeddings ``geom_z``, the loss encourages the i-th image to be most
    similar to the i-th geometry vector (the positive pair), while all
    off-diagonal entries are negatives.
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: softmax temperature (smaller = sharper distribution)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, img_z, geom_z):
        """
        Args:
            img_z:  [B, D] L2-normalised image embeddings
            geom_z: [B, D] L2-normalised geometry embeddings

        Returns:
            Scalar contrastive loss (symmetric InfoNCE).
        """
        # Cosine similarity matrix scaled by temperature: [B, B]
        logits = img_z @ geom_z.T / self.temperature

        # Diagonal entries are the positives (i-th image ↔ i-th geometry)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Symmetric: image→geometry + geometry→image
        loss_i2g = F.cross_entropy(logits, labels)
        loss_g2i = F.cross_entropy(logits.T, labels)
        return (loss_i2g + loss_g2i) / 2
