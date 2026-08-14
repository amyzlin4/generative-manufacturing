# =============================================================================
# gmdl/decoder_training.py — Decoder Training for Latent-Space Analysis
# =============================================================================
#
# Trains two decoder heads on top of a frozen geometry encoder:
#   1. LinearDecoder (128 → 64)   — reconstructs hand-crafted geometry features
#   2. PointCloudDecoder (128 → [512, 3]) — reconstructs the input point cloud
#
# The decoders are trained simultaneously (summed MSE loss) so that the latent
# space encodes enough information for both.  Once trained, they enable the
# analysis pipeline to translate latent-direction edits back into physical
# property changes (feature decoder) and shape changes (point cloud decoder).
#
# Data source:
#   Same JSON manifests (train.json / validation.json) and .mat mesh files
#   used by the main model.  The 64-dim feature targets are computed
#   on-the-fly from the mesh geometry via mesh_to_geom_features().
#
# Validation metrics tracked per epoch:
#   - feature_mse   : MSE on the 64-dim feature vector
#   - pc_mse        : MSE on the 512×3 vertex array
#   - feature_cosine: mean cosine similarity (scale-invariant)
#   - chamfer_dist  : Chamfer distance for point cloud quality
#   - per_feature_nmse[64]: per-feature normalised MSE (identifies weak dims)
# =============================================================================

import os
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gmdl.constants import GEOM_FEATURE_DIM, MAX_POINTS, LATENT_DIM
from gmdl.decoders import LinearDecoder, PointCloudDecoder


# ---------------------------------------------------------------------------
# Mesh helpers (adapted from data-helpers/hks2gmdl.py)
# ---------------------------------------------------------------------------

def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def mesh_surface_area(pts, faces):
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    total = 0.0
    for f in faces:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        total += triangle_area(a, b, c)
    return total


def mesh_volume(pts, faces):
    vol = 0.0
    for f in faces:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        vol += np.dot(a, np.cross(b, c))
    return abs(vol) / 6.0


def mesh_bbox(pts):
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return maxs - mins


def load_mat_mesh(path):
    """Load a .mat file and return (vertices, faces) as numpy arrays.
    Handles both v5 (scipy.io.loadmat) and v7.3 (h5py) formats.
    """
    pts = faces = None
    try:
        import scipy.io as sio
        data = sio.loadmat(str(path))
        pts = data.get("point3D")
        faces = data.get("face3D")
        if pts is not None and faces is not None:
            faces = np.asarray(faces, dtype=int) - 1
            return pts, faces
    except Exception:
        pass
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            pts = f["point3D"][:]
            faces = f["face3D"][:]
            if pts.shape[0] == 3 and pts.shape[1] != 3:
                pts = pts.T
            faces = faces - 1
            return pts, faces
    except Exception:
        raise ValueError(f"Could not load {path}")


def mesh_to_geom_features(pts, faces):
    """Compute the 64-dim geometric feature vector from mesh data.
    Matches the ordering used in data-helpers/hks2gmdl.py.
    """
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    faces = np.asarray(faces, dtype=int)

    xlen, ylen, zlen = mesh_bbox(pts)
    vol = mesh_volume(pts, faces)
    sa = mesh_surface_area(pts, faces)
    bb_vol = xlen * ylen * zlen
    aspect_ratio = max(xlen, ylen, zlen) / min(xlen, ylen, zlen) if min(xlen, ylen, zlen) > 0 else 1.0
    compactness = vol / bb_vol if bb_vol > 0 else 0.0

    n_verts = len(pts)
    n_faces = len(faces)
    vertex_density = n_verts / sa if sa > 0 else 0.0

    centroid = pts.mean(axis=0)
    v_std = pts.std(axis=0)

    centered = pts - centroid
    cov = np.cov(centered, rowvar=False) if centered.shape[0] > 1 else np.zeros((3, 3))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    pca_ratio = eigvals[0] / eigvals[2] if eigvals[2] > 1e-12 else 1.0

    v0 = pts[faces[:, 0]]
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]
    edge_lengths = np.concatenate([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    edge_mean = edge_lengths.mean()
    edge_std = edge_lengths.std()
    edge_min = edge_lengths.min()
    edge_max = edge_lengths.max()

    face_areas = np.array([triangle_area(pts[f[0]], pts[f[1]], pts[f[2]]) for f in faces])
    face_area_mean = face_areas.mean()
    face_area_std = face_areas.std()
    top3_faces = np.sort(face_areas)[::-1][:3]
    while len(top3_faces) < 3:
        top3_faces = np.append(top3_faces, 0.0)

    fn = np.cross(v1 - v0, v2 - v0)
    fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_norms[fn_norms == 0] = 1.0
    fn = fn / fn_norms
    fn_mean = fn.mean(axis=0)
    fn_std = fn.std(axis=0)

    n_edges = (3 * n_faces + 3) // 2
    euler = n_verts - n_edges + n_faces

    try:
        from scipy.spatial import ConvexHull
        hull_vol = ConvexHull(pts).volume if len(pts) >= 4 else vol
        convex_ratio = vol / hull_vol if hull_vol > 0 else 0.0
    except Exception:
        convex_ratio = 0.0

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    sa_vol_ratio = sa / vol if vol > 0 else 0.0
    sphericity = (np.pi ** (1.0 / 3.0)) * ((6.0 * vol) ** (2.0 / 3.0)) / sa if sa > 0 else 0.0
    diameter = np.sqrt(xlen**2 + ylen**2 + zlen**2)

    vtc_dist = np.linalg.norm(pts - centroid, axis=1)
    vtc_mean = vtc_dist.mean()
    vtc_std = vtc_dist.std()

    face_area_hist, _ = np.histogram(face_areas, bins=7)
    face_area_hist = face_area_hist.astype(np.float32)
    fa_sum = face_area_hist.sum()
    if fa_sum > 0:
        face_area_hist /= fa_sum

    edge_hist, _ = np.histogram(edge_lengths, bins=7)
    edge_hist = edge_hist.astype(np.float32)
    e_sum = edge_hist.sum()
    if e_sum > 0:
        edge_hist /= e_sum

    dih_angles = []
    edge_to_face = {}
    for fi, f in enumerate(faces):
        for j in range(3):
            e = tuple(sorted((int(f[j]), int(f[(j + 1) % 3]))))
            if e in edge_to_face:
                fi2 = edge_to_face[e]
                dot = np.clip(np.dot(fn[fi], fn[fi2]), -1.0, 1.0)
                dih_angles.append(np.arccos(dot))
            else:
                edge_to_face[e] = fi
    if dih_angles:
        dih_mean = float(np.mean(dih_angles))
        dih_std = float(np.std(dih_angles))
    else:
        dih_mean = 0.0
        dih_std = 0.0

    geom = np.array([
        vol, sa, aspect_ratio, compactness, xlen, ylen, zlen,
        n_verts, n_faces, vertex_density,
        centroid[0], centroid[1], centroid[2],
        v_std[0], v_std[1], v_std[2],
        eigvals[0], eigvals[1], eigvals[2],
        pca_ratio,
        edge_mean, edge_std, edge_min, edge_max,
        face_area_mean, face_area_std,
        fn_mean[0], fn_mean[1], fn_mean[2],
        fn_std[0], fn_std[1], fn_std[2],
        euler, convex_ratio,
        mins[0], mins[1], mins[2],
        maxs[0], maxs[1], maxs[2],
        sa_vol_ratio,
        dih_mean, dih_std,
        *face_area_hist,
        *edge_hist,
        sphericity, diameter,
        vtc_mean, vtc_std,
        top3_faces[0], top3_faces[1], top3_faces[2],
    ], dtype=np.float32)

    assert len(geom) == GEOM_FEATURE_DIM, f"Expected {GEOM_FEATURE_DIM} features, got {len(geom)}"
    return geom


# ---------------------------------------------------------------------------
# Feature standardization
# ---------------------------------------------------------------------------
# The 64 geometry features span very different magnitudes (e.g. volume ~10^2,
# surface area ~10^3, histogram bins ~10^-2).  Without standardization the
# MSE loss is dominated by large-scale features and the decoder never learns
# to reconstruct small ones, producing low cosine similarity scores.
# ---------------------------------------------------------------------------

class FeatureStandardizer:
    """Per-feature z-score standardization for decoder training.

    Usage:
        stdz = FeatureStandardizer()
        stdz.fit(dataset)            # compute mean + std over all samples
        feats = stdz.transform(raw)  # (x - mean) / std
        raw = stdz.inverse(feats)    # un-standardize for display
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        """Compute per-feature mean and std over all non-trivial samples."""
        all_feats = []
        for i in range(len(dataset)):
            _, feat, n_valid = dataset[i]
            if n_valid > 0:
                all_feats.append(feat.numpy())
        if not all_feats:
            all_feats = [np.zeros(GEOM_FEATURE_DIM, dtype=np.float32)]
        all_feats = np.stack(all_feats, axis=0)
        self.mean = np.mean(all_feats, axis=0).astype(np.float32)
        self.std = np.std(all_feats, axis=0, ddof=0).astype(np.float32)
        # Prevent division by zero for constant features
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, features):
        """Z-score standardize: (x - mean) / std."""
        if isinstance(features, torch.Tensor):
            mean_t = torch.tensor(self.mean, device=features.device, dtype=features.dtype)
            std_t = torch.tensor(self.std, device=features.device, dtype=features.dtype)
            return (features - mean_t) / std_t
        return (features - self.mean) / self.std

    def inverse_transform(self, features):
        """Un-standardize back to original scale: x * std + mean."""
        if isinstance(features, torch.Tensor):
            mean_t = torch.tensor(self.mean, device=features.device, dtype=features.dtype)
            std_t = torch.tensor(self.std, device=features.device, dtype=features.dtype)
        else:
            mean_t = self.mean
            std_t = self.std
        return features * std_t + mean_t


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def chamfer_distance(pred, target):
    """Chamfer distance between two point cloud tensors.

    For each point in set A, finds the nearest point in set B (and vice
    versa), then returns the mean squared distance summed in both directions.

    Args:
        pred:   [B, N, 3] predicted vertices.
        target: [B, N, 3] ground-truth vertices.

    Returns:
        [B] per-sample Chamfer distance.
    """
    B, N, _ = pred.shape
    pred_expand = pred.unsqueeze(2)    # [B, N, 1, 3]
    target_expand = target.unsqueeze(1)  # [B, 1, N, 3]
    dist = torch.sum((pred_expand - target_expand) ** 2, dim=-1)  # [B, N, N]
    min_dist_pred = dist.min(dim=2)[0]    # [B, N] — nearest target for each pred
    min_dist_target = dist.min(dim=1)[0]  # [B, N] — nearest pred for each target
    return min_dist_pred.mean(dim=1) + min_dist_target.mean(dim=1)


def per_feature_nmse(pred, target, eps=1e-8):
    """Per-feature normalised MSE.

    Each feature's MSE is normalised by its variance in the target, so
    high-error features that matter most stand out regardless of scale.

    Args:
        pred:   [B, F] predicted features.
        target: [B, F] ground-truth features.
        eps:    small constant to avoid division by zero.

    Returns:
        [F] per-feature normalised MSE.
    """
    mse_per_feat = ((pred - target) ** 2).mean(dim=0)
    var_per_feat = target.var(dim=0, unbiased=False)
    return mse_per_feat / (var_per_feat + eps)


# ---------------------------------------------------------------------------
# Decoder dataset
# ---------------------------------------------------------------------------

class DecoderDataset(Dataset):
    """Dataset that yields (vertices, geom_features) pairs for decoder training.

    Loads .mat mesh files from the same JSON manifest used for the main
    training, then computes the 64-dim feature vector from the full mesh
    (vertices + faces) on the fly.
    """

    def __init__(self, data_root, split="train", max_points=MAX_POINTS, manifest_file=None):
        self.data_root = Path(data_root)
        self.max_points = max_points
        self.samples = []

        split_path = Path(manifest_file) if manifest_file else self.data_root / f"{split}.json"
        if split_path.exists():
            with open(split_path) as f:
                self.samples = json.load(f)
        else:
            warnings.warn(f"No manifest found at {split_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        mesh_path = self.data_root / sample.get("mesh_file", "")
        if not mesh_path.exists():
            # Return dummy data if mesh not found
            vertices = torch.zeros((self.max_points, 3), dtype=torch.float32)
            return vertices, torch.zeros(GEOM_FEATURE_DIM, dtype=torch.float32), 0

        try:
            pts, faces = load_mat_mesh(mesh_path)
        except Exception:
            return torch.zeros((self.max_points, 3), dtype=torch.float32), \
                   torch.zeros(GEOM_FEATURE_DIM, dtype=torch.float32), 0

        # --- Vertices (for PointNet encoder) ---
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T

        n_valid = min(len(pts), self.max_points)
        vertices = np.zeros((self.max_points, 3), dtype=np.float32)
        vertices[:n_valid] = pts[:n_valid]

        # --- 64-dim features (decoder target) ---
        try:
            geom = mesh_to_geom_features(pts, faces)
        except Exception:
            geom = np.zeros(GEOM_FEATURE_DIM, dtype=np.float32)

        return (
            torch.from_numpy(vertices),
            torch.from_numpy(geom),
            n_valid,
        )


def get_decoder_dataloader(cfg, split="train"):
    """Build a DataLoader for decoder training."""
    ds = DecoderDataset(
        cfg.data_root,
        split=split,
        max_points=getattr(cfg, "max_points", MAX_POINTS),
        manifest_file=getattr(cfg, "decoder_manifest", None),
    )
    return DataLoader(
        ds,
        batch_size=getattr(cfg, "decoder_batch_size", 32),
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Decoder training
# ---------------------------------------------------------------------------

def train_decoders(encoder, train_loader, cfg, device="cpu", val_loader=None):
    """Train the feature decoder and point cloud decoder.

    The encoder is frozen — only decoder weights are updated.
    Both decoders are trained simultaneously with summed MSE losses.

    When val_loader is provided, validation metrics (feature_mse, pc_mse,
    feature_cosine, chamfer_dist, per_feature_nmse) are computed after
    every epoch and the best epoch is tracked.

    Args:
        encoder:      trained and frozen PointNetEncoder (or GeometryEncoder).
        train_loader: DataLoader yielding (vertices, geom_features, n_valid).
        cfg:          object with decoder hyper-params
                      (geom_encoder, decoder_hidden, decoder_lr, decoder_epochs,
                       decoder_batch_size, etc.).
        device:       torch device.
        val_loader:   optional validation DataLoader.

    Returns:
        (feature_decoder, pointcloud_decoder) with trained weights.
    """
    latent_dim = getattr(cfg.geom_encoder, "latent_dim", LATENT_DIM)
    feature_dim = getattr(cfg.geom_encoder, "feature_dim", GEOM_FEATURE_DIM)
    max_points = getattr(cfg.geom_encoder, "max_points", MAX_POINTS)
    hidden = getattr(cfg, "decoder_hidden", 256)
    lr = getattr(cfg, "decoder_lr", 1e-3)
    epochs = getattr(cfg, "decoder_epochs", 20)

    feature_decoder = LinearDecoder(latent_dim, feature_dim, hidden).to(device)
    pc_decoder = PointCloudDecoder(latent_dim, max_points, hidden).to(device)

    # Fit feature standardizer on the training set
    standardizer = FeatureStandardizer()
    standardizer.fit(train_loader.dataset)
    stdz_mean = torch.tensor(standardizer.mean, device=device, dtype=torch.float32)
    stdz_std = torch.tensor(standardizer.std, device=device, dtype=torch.float32)

    encoder.eval()
    optim = torch.optim.AdamW(
        list(feature_decoder.parameters()) + list(pc_decoder.parameters()),
        lr=lr,
    )
    mse = nn.MSELoss()

    # EMA trackers for auto-balancing PC loss against feature loss
    feat_loss_ema = None
    pc_loss_ema = None

    # ReduceLROnPlateau on validation feature MSE
    lr_patience = getattr(cfg, "decoder_lr_patience", 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", patience=lr_patience, factor=0.5,
    )

    best_val_feat_mse = float("inf")
    best_state = None
    best_epoch = -1
    cos_sim = nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        # --- Training ---
        feature_decoder.train()
        pc_decoder.train()
        total_feat_loss = 0.0
        total_pc_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            vertices, geom_features, n_valid = batch
            vertices = vertices.to(device)
            n_valid = n_valid.to(device)
            # Standardize feature targets to equalise contribution of each dim
            geom_features = ((geom_features.to(device) - stdz_mean) / stdz_std)

            with torch.no_grad():
                z = encoder(vertices, n_valid_points=n_valid)

            pred_features = feature_decoder(z)
            feat_loss = mse(pred_features, geom_features)

            pred_vertices = pc_decoder(z)
            pc_loss = mse(pred_vertices, vertices)

            # Auto-balancing PC loss weight via EMA
            fl = feat_loss.item()
            pl = pc_loss.item()
            if feat_loss_ema is None:
                feat_loss_ema = fl
                pc_loss_ema = pl
            else:
                feat_loss_ema = 0.99 * feat_loss_ema + 0.01 * fl
                pc_loss_ema = 0.99 * pc_loss_ema + 0.01 * pl
            pc_weight = feat_loss_ema / (pc_loss_ema + 1e-8)

            loss = feat_loss + pc_weight * pc_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_feat_loss += fl
            total_pc_loss += pl
            n_batches += 1

        avg_train_feat_mse = total_feat_loss / n_batches
        avg_train_pc_mse = total_pc_loss / n_batches

        # --- Validation ---
        if val_loader is not None:
            feature_decoder.eval()
            pc_decoder.eval()
            val_feat_loss = 0.0
            val_pc_loss = 0.0
            val_cos = 0.0
            val_chamfer = 0.0
            val_n_feat = 0
            val_n_pc = 0
            all_pred_feat, all_target_feat = [], []

            with torch.no_grad():
                for batch in val_loader:
                    vertices, geom_features, n_valid = batch
                    vertices = vertices.to(device)
                    n_valid = n_valid.to(device)
                    geom_features = ((geom_features.to(device) - stdz_mean) / stdz_std)
                    z = encoder(vertices, n_valid_points=n_valid)

                    pred_features = feature_decoder(z)
                    pred_vertices = pc_decoder(z)

                    valid_mask = n_valid > 0
                    if valid_mask.any():
                        v_idx = valid_mask
                        val_feat_loss += mse(pred_features[v_idx],
                                             geom_features[v_idx]).item() * v_idx.sum().item()
                        val_n_feat += v_idx.sum().item()
                        all_pred_feat.append(pred_features[v_idx])
                        all_target_feat.append(geom_features[v_idx])

                    val_cos += cos_sim(pred_features, geom_features).sum().item()
                    val_n_feat += geom_features.shape[0]

                    val_pc_loss += mse(pred_vertices, vertices).item() * vertices.shape[0]
                    val_n_pc += vertices.shape[0]

                    if valid_mask.any():
                        val_chamfer += chamfer_distance(
                            pred_vertices[v_idx], vertices[v_idx]
                        ).sum().item()

            avg_val_feat_mse = val_feat_loss / max(val_n_feat, 1)
            avg_val_pc_mse = val_pc_loss / max(val_n_pc, 1)
            avg_val_cos = val_cos / max(val_n_feat, 1)
            avg_val_chamfer = val_chamfer / max(val_n_feat, 1)

            feat_nmse = None
            if all_pred_feat:
                all_pred_feat = torch.cat(all_pred_feat, dim=0)
                all_target_feat = torch.cat(all_target_feat, dim=0)
                feat_nmse = per_feature_nmse(all_pred_feat, all_target_feat)

            # Step scheduler and track best epoch
            scheduler.step(avg_val_feat_mse)

            if avg_val_feat_mse < best_val_feat_mse:
                best_val_feat_mse = avg_val_feat_mse
                best_state = {
                    "feature_decoder": feature_decoder.state_dict(),
                    "pc_decoder": pc_decoder.state_dict(),
                }
                best_epoch = epoch

            current_lr = optim.param_groups[0]["lr"]
            summary = (f"  Epoch {epoch+1:3d}/{epochs}: "
                       f"train feat={avg_train_feat_mse:.5f}  pc={avg_train_pc_mse:.5f} | "
                       f"val  feat={avg_val_feat_mse:.5f}  pc={avg_val_pc_mse:.5f}  "
                       f"cos={avg_val_cos:.4f}  chamfer={avg_val_chamfer:.5f}  "
                       f"w={pc_weight:.3f}  lr={current_lr:.1e}")
            if feat_nmse is not None:
                worst_idx = feat_nmse.argsort(descending=True)[:3]
                from gmdl.analysis import FEATURE_NAMES
                worst_str = "  worst_feat: " + ", ".join(
                    f"{FEATURE_NAMES[i]}({feat_nmse[i]:.3f})" for i in worst_idx
                )
                summary += worst_str
            print(summary)

        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"train feat_mse={avg_train_feat_mse:.6f}  "
                      f"pc_mse={avg_train_pc_mse:.6f}")

    if best_state is not None:
        feature_decoder.load_state_dict(best_state["feature_decoder"])
        pc_decoder.load_state_dict(best_state["pc_decoder"])
        print(f"  Best epoch: {best_epoch+1} "
              f"(val_feat_mse={best_val_feat_mse:.6f})")

    # Attach standardizer so the caller can save + use it later
    return feature_decoder, pc_decoder, standardizer
