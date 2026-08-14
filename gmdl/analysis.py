# =============================================================================
# gmdl/analysis.py — Fit Scoring, Latent Optimisation, and Design Explanation
# =============================================================================
#
# Three composable analysis components built on top of a trained model:
#
# 1. ProcessFitScorer
#    Computes per-process fit scores from a latent vector.  The fit score
#    combines softmax probability (0.6 weight) and a centroid-based silhouette
#    score in latent space (0.4 weight).  Centroids are computed by running the
#    encoder over the training set.
#
# 2. LatentDirectionAnalyzer
#    Uses Adam to optimise a latent vector toward (or away from) a target
#    fit score for a given process.  This is *not* gradient-based model
#    editing — the predictor weights are frozen; only the latent is updated.
#    The optimiser naturally handles both increasing and decreasing fit.
#
# 3. DesignExplainer
#    Decodes the original and modified latents through trained decoders,
#    compares the 64-dim feature vectors, and maps the largest deltas
#    to process-specific design suggestions via a rule-based lookup table.
#
# Together these enable the `--analyze` CLI mode:
#   "How well does this part fit process X?"
#   "What would the design need to look like for a better fit?"
# =============================================================================

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gmdl.constants import (
    GEOM_FEATURE_DIM, LATENT_DIM, MANUFACTURING_PROCESSES,
)
from gmdl.decoders import LinearDecoder, PointCloudDecoder
from gmdl.mesh_viz import (
    displacement_to_color, export_colored_pointcloud_ply, export_gray_mesh_ply,
    render_pointcloud_mpl,
)

# ---------------------------------------------------------------------------
# Human-readable names for the 64 geometry features
# ---------------------------------------------------------------------------
# Order matches the index layout of mesh_to_geom_features() in
# decoder_training.py.  See that function for the exact computation.
# The first 40-odd are scalar / per-face stats; the rest are histogram
# bins, sphericity, and derived quantities.
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "volume mm3", "surface area mm2", "aspect ratio", "compactness",
    "bbox x mm", "bbox y mm", "bbox z mm",
    "vertex count", "face count", "vertex density",
    "centroid x mm", "centroid y mm", "centroid z mm",
    "vertex std x", "vertex std y", "vertex std z",
    "PCA eigenvalue 1", "PCA eigenvalue 2", "PCA eigenvalue 3",
    "PCA elongation ratio",
    "edge length mean mm", "edge length std mm",
    "edge length min mm", "edge length max mm",
    "face area mean mm2", "face area std mm2",
    "face normal mean x", "face normal mean y", "face normal mean z",
    "face normal std x", "face normal std y", "face normal std z",
    "Euler characteristic", "convex hull volume ratio",
    "bbox min x mm", "bbox min y mm", "bbox min z mm",
    "bbox max x mm", "bbox max y mm", "bbox max z mm",
    "surface area / volume",
    "dihedral angle mean rad", "dihedral angle std rad",
    "face area histogram bin 1", "face area histogram bin 2",
    "face area histogram bin 3", "face area histogram bin 4",
    "face area histogram bin 5", "face area histogram bin 6",
    "face area histogram bin 7",
    "edge length histogram bin 1", "edge length histogram bin 2",
    "edge length histogram bin 3", "edge length histogram bin 4",
    "edge length histogram bin 5", "edge length histogram bin 6",
    "edge length histogram bin 7",
    "sphericity", "diameter mm",
    "vertex-to-centroid dist mean mm", "vertex-to-centroid dist std mm",
    "largest face area mm2", "2nd largest face area mm2",
    "3rd largest face area mm2",
]

# Weights for fit score computation
FIT_ALPHA = 0.6  # weight for softmax probability
FIT_BETA = 0.4   # weight for normalized silhouette score


class LatentTransformer:
    """LDA projection + RBF similarity for amplified class separation.

    Fits a linear discriminant analysis (LDA) projection from the original
    latent space to a lower-dimensional space (n_classes - 1 dimensions),
    then computes similarity using a Gaussian RBF kernel on distances
    in the projected space.  This preserves magnitude information (unlike
    cosine similarity in low dimensions) and provides sharper separation.
    """

    def __init__(self):
        self.W = None         # ndarray [latent_dim, lda_dim]
        self.W_t = None       # cached torch tensor
        self.rbf_gamma = 1.0  # RBF kernel width

    def fit(self, X, y):
        """Fit LDA projection + RBF kernel width from training latents.

        Args:
            X: ndarray [N, latent_dim] of latent vectors.
            y: ndarray [N] of integer class labels.
        """
        classes = np.unique(y)
        K = len(classes)
        d = X.shape[1]

        global_mean = X.mean(axis=0)
        S_w = np.zeros((d, d), dtype=np.float64)
        S_b = np.zeros((d, d), dtype=np.float64)

        for k in classes:
            Xk = X[y == k]
            nk = Xk.shape[0]
            mk = Xk.mean(axis=0)
            centered = Xk - mk
            S_w += centered.T @ centered
            diff = (mk - global_mean).reshape(-1, 1)
            S_b += nk * (diff @ diff.T)

        # Regularise S_w to handle ill-conditioned / low-rank cases
        reg = 1e-4 * np.trace(S_w) / d
        Sw_inv_Sb = np.linalg.solve(S_w + reg * np.eye(d, dtype=np.float64), S_b)
        eigvals, eigvecs = np.linalg.eigh(Sw_inv_Sb)

        # Take top K-1 eigenvectors (real eigenvalues, descending)
        idx = np.argsort(eigvals)[::-1][:K - 1]
        self.W = eigvecs[:, idx].real.astype(np.float32)
        self.W_t = None

        # Fit RBF kernel width from median within-class distance in LDA space
        X_proj = X @ self.W
        dists = []
        for k in classes:
            Xk = X_proj[y == k]
            mk = Xk.mean(axis=0)
            dists.extend(np.linalg.norm(Xk - mk, axis=1).tolist())
        median_d = np.median(dists) if dists else 1.0
        self.rbf_gamma = float(1.0 / (2 * max(median_d, 1e-8) ** 2))

    def transform(self, z):
        """Project latent vector(s) through LDA subspace.

        Args:
            z: [batch, latent_dim] tensor or ndarray.

        Returns:
            Same type and batch dim, projected to [batch, lda_dim].
        """
        if isinstance(z, torch.Tensor):
            if self.W_t is None or self.W_t.device != z.device:
                self.W_t = torch.tensor(self.W, dtype=torch.float32, device=z.device)
            return z @ self.W_t
        return z @ self.W

    def similarity(self, z, centroid):
        """RBF similarity in LDA projected space.

        Args:
            z:        [1, latent_dim] tensor on the model device.
            centroid: [1, latent_dim] tensor on the same device.

        Returns:
            scalar tensor in (0, 1], differentiable for gradient-based
            optimisation.
        """
        z_p = self.transform(z)
        c_p = self.transform(centroid)
        dist2 = ((z_p - c_p) ** 2).sum(dim=-1)
        return torch.exp(-self.rbf_gamma * dist2)

# Process-specific design suggestions
# Each process has two buckets ("high" / "low") keyed by the direction
# of the most-changed feature in the decoded feature delta.  During
# explanation, the DesignExplainer picks the bucket based on whether the
# largest feature delta (e.g. aspect_ratio, compactness) increased or
# decreased.  The text within each bucket is cycled to avoid repetition.
# These are qualitative heuristics, not model-derived rules.
PROCESS_SUGGESTIONS = {
    0: {  # 3-axis CNC machining
        "high": ["Reducing aspect ratio improves fixturing stability for 3-axis CNC",
                 "Increasing compactness reduces the need for multiple setups",
                 "Lower surface-area-to-volume ratio speeds up roughing passes",
                 "Adding fillets to sharp internal corners improves tool access"],
        "low":  ["The aspect ratio is high — consider splitting into multiple setups or using 5-axis",
                 "Low compactness suggests deep cavities that may be hard to reach",
                 "High surface-area-to-volume ratio increases machining time",
                 "Large face area variations suggest non-uniform stock removal"],
    },
    1: {  # 5-axis CNC machining
        "high": ["Moderate aspect ratio is well-suited for 5-axis simultaneous machining",
                 "Good compactness allows efficient 5-axis tool paths",
                 "Face normal variation indicates freeform surfaces suitable for 5-axis"],
        "low":  ["Consider if this part can be simplified into 3-axis operations to reduce cost",
                 "Excessive surface complexity may require custom fixturing"],
    },
    2: {  # Injection molding
        "high": ["Uniform wall thickness promotes even cooling and reduces warpage",
                 "High compactness reduces material usage per part",
                 "Low aspect ratio helps with mold filling and ejection",
                 "Good sphericity indicates balanced shrinkage during cooling",
                 "Adding draft angles (1-3 degrees) would improve ejection"],
        "low":  ["High aspect ratio may cause filling difficulties — consider adjusting part orientation",
                 "Low compactness suggests thick sections that increase cycle time",
                 "High surface-area-to-volume ratio may cause premature freezing",
                 "Consider adding ribs for stiffness instead of increasing wall thickness"],
    },
    3: {  # Casting
        "high": ["High compactness promotes directional solidification",
                 "Low aspect ratio reduces hot tearing risk",
                 "Increasing corner radii would improve metal flow"],
        "low":  ["High surface-area-to-volume ratio increases risk of cold shuts",
                 "Consider adding feeders/risers to address thick sections",
                 "Low compactness suggests uneven solidification — adjust gating system"],
    },
    4: {  # Forging
        "high": ["Good compactness suggests material will flow evenly in the die",
                 "Low aspect ratio reduces the number of preform stages needed",
                 "Adding draft angles to vertical walls improves die release"],
        "low":  ["High aspect ratio may require multiple preform steps",
                 "Consider if the geometry can be simplified for a single forging blow"],
    },
    5: {  # Lathing/Turning
        "high": ["Near-cylindrical shape is ideal for turning operations",
                 "Good compactness minimizes waste material",
                 "Low surface-area-to-volume ratio suggests efficient material removal",
                 "Centroid along the axis of rotation would improve balance"],
        "low":  ["Non-axisymmetric features suggest a mill-turn or secondary operation",
                 "Increase roundness and concentricity for better turning results",
                 "Consider adding a center-drill hole for live-center support"],
    },
    6: {  # Sheet metal fabrication
        "high": ["Low volume and high aspect ratio typical of sheet metal parts",
                 "Planar-like geometry bends predictably",
                 "High compactness of the unfolded blank reduces scrap"],
        "low":  ["Deep-draw depth exceeds recommended 1:1 depth-to-width ratio",
                 "Consider reducing flange width to avoid buckling",
                 "Generous corner radii improve formability"],
    },
    7: {  # 3D printing
        "high": ["Complex geometries are handled naturally by additive manufacturing",
                 "High surface-area-to-volume ratio is not a concern for layer-based processes",
                 "No tool-access constraints simplify otherwise complex parts"],
        "low":  ["Consider consolidating assemblies into single printed parts",
                 "Generous internal fillets reduce the need for support structures",
                 "Orient the part to minimize overhangs below 45 degrees"],
    },
    8: {  # Sintering
        "high": ["Moderate complexity is well-suited for powder-based sintering",
                 "Uniform cross-sections promote even densification"],
        "low":  ["Thin walls may distort during sintering — consider thickening",
                 "Large flat faces can cause warpage — add corrugation or reduce area"],
    },
}


class FitResult:
    """Container for a single process fit score."""

    def __init__(self, idx, process_name, probability, silhouette_score,
                 fit_score, rank, silhouette_score_raw=None):
        self.idx = idx
        self.process_name = process_name
        self.probability = probability
        self.silhouette_score = silhouette_score
        self.silhouette_score_raw = (
            silhouette_score_raw if silhouette_score_raw is not None
            else silhouette_score
        )
        self.fit_score = fit_score
        self.rank = rank

    def to_dict(self):
        return {
            "process_name": self.process_name,
            "probability": round(float(self.probability), 4),
            "silhouette_score": round(float(self.silhouette_score), 4),
            "fit_score": round(float(self.fit_score), 4),
            "rank": self.rank,
        }

    def __repr__(self):
        return (f"FitResult({self.process_name}, "
                f"score={self.fit_score:.3f}, "
                f"prob={self.probability:.3f}, "
                f"silhouette={self.silhouette_score:.3f})")


class KNNFitScorer:
    """Class-support fit scorer based on nearest training latents.

    This scorer answers a narrower question than classifier confidence:
    "does this latent look like it belongs anywhere inside the target class's
    training support?"  It does not privilege the centroid, so points near a
    valid edge of an irregular cluster can still receive a high fit score.
    """

    def __init__(self, class_embeddings=None, latent_mean=None, latent_std=None,
                 train_knn_distances=None, k=5, good_percentile=90,
                 bad_percentile=99, bad_radius_scale=1.5,
                 tail_alpha=1.0, active_indices=None):
        self.class_embeddings = class_embeddings if class_embeddings is not None else {}
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.train_knn_distances = train_knn_distances if train_knn_distances is not None else {}
        self.k = int(k)
        self.good_percentile = good_percentile
        self.bad_percentile = bad_percentile
        self.bad_radius_scale = bad_radius_scale
        self.tail_alpha = tail_alpha
        self.active_indices = active_indices

    def fit(self, latents, labels):
        """Build class reference sets and leave-one-out KNN distance baselines."""
        latents = np.asarray(latents, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        if latents.size == 0:
            return self

        self.latent_mean = latents.mean(axis=0).astype(np.float32)
        self.latent_std = latents.std(axis=0).astype(np.float32)
        self.latent_std[self.latent_std < 1e-8] = 1.0

        self.class_embeddings = {}
        self.train_knn_distances = {}
        for lbl in sorted(np.unique(labels)):
            emb = latents[labels == lbl].astype(np.float32)
            self.class_embeddings[int(lbl)] = emb
            self.train_knn_distances[int(lbl)] = self._leave_one_out_distances(emb)
        return self

    def score(self, z, target_idx=None, probabilities=None):
        """Score one target class or all active classes.

        Args:
            z: torch tensor or ndarray with shape [latent_dim] or [1, latent_dim].
            target_idx: optional process index. If None, scores all classes with
                reference embeddings.
            probabilities: optional classifier probabilities for reporting only.
        """
        z_np = self._to_numpy(z).reshape(-1)
        probs_np = None if probabilities is None else self._to_numpy(probabilities).reshape(-1)

        if target_idx is not None:
            return self._score_one(z_np, int(target_idx), probs_np)

        indices = self.active_indices if self.active_indices is not None else sorted(self.class_embeddings.keys())
        results = [self._score_one(z_np, int(idx), probs_np) for idx in indices
                   if int(idx) in self.class_embeddings]
        results.sort(key=lambda r: r["fit_score"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        return results

    def suggest_target(self, z, target_idx):
        """Return a local target latent and direction toward nearest class support."""
        z_np = self._to_numpy(z).reshape(-1).astype(np.float32)
        emb = self.class_embeddings.get(int(target_idx))
        if emb is None or len(emb) == 0:
            return z_np, np.zeros_like(z_np), []

        dists = self._distances_to_class(z_np, emb)
        k_eff = min(max(1, self.k), len(emb))
        nn_idx = np.argsort(dists)[:k_eff]
        nn_dists = dists[nn_idx]
        weights = 1.0 / (nn_dists + 1e-8)
        weights = weights / weights.sum()
        target = (emb[nn_idx] * weights[:, None]).sum(axis=0).astype(np.float32)
        direction = target - z_np
        return target, direction, nn_dists.astype(np.float32)

    def _score_one(self, z_np, idx, probs_np=None):
        emb = self.class_embeddings.get(idx)
        name = MANUFACTURING_PROCESSES[idx] if idx < len(MANUFACTURING_PROCESSES) else f"Process {idx}"
        prob = float(probs_np[idx]) if probs_np is not None and idx < len(probs_np) else None
        if emb is None or len(emb) == 0:
            return {
                "process_name": name,
                "probability": prob,
                "fit_score": 0.0,
                "knn_distance": None,
                "class_good_radius": None,
                "class_bad_radius": None,
                "in_distribution": False,
                "nearest_neighbor_distances": [],
                "rank": 0,
            }

        dists = self._distances_to_class(z_np, emb)
        k_eff = min(max(1, self.k), len(emb))
        nn_dists = np.sort(dists)[:k_eff]
        d_test = float(nn_dists.mean())
        good_radius, bad_radius = self._class_radii(idx)

        if d_test <= good_radius:
            score = 1.0
        else:
            # Soft long-tail decay: outside support is penalized, but not
            # flattened to zero unless the distance is extremely large.
            t = (d_test - good_radius) / max(bad_radius - good_radius, 1e-8)
            score = float(np.exp(-self.tail_alpha * max(t, 0.0)))
            score = max(score, 1e-6)

        if d_test <= good_radius:
            support_status = "inside"
        elif d_test <= bad_radius:
            support_status = "near_boundary"
        elif d_test <= 2.0 * bad_radius:
            support_status = "outside_but_near"
        else:
            support_status = "far_out_of_distribution"

        return {
            "process_name": name,
            "probability": None if prob is None else round(prob, 4),
            "fit_score": round(float(score), 6),
            "fit_method": "knn_class_support",
            "knn_distance": round(d_test, 6),
            "class_good_radius": round(float(good_radius), 6),
            "class_bad_radius": round(float(bad_radius), 6),
            "in_distribution": bool(d_test <= good_radius),
            "support_status": support_status,
            "distance_over_good_radius": round(float(d_test / max(good_radius, 1e-8)), 6),
            "distance_over_bad_radius": round(float(d_test / max(bad_radius, 1e-8)), 6),
            "nearest_neighbor_distances": [round(float(x), 6) for x in nn_dists],
            "rank": 0,
        }

    def _leave_one_out_distances(self, emb):
        if len(emb) <= 1:
            return np.array([0.0], dtype=np.float32)
        emb_z = self._standardize(emb)
        diff = emb_z[:, None, :] - emb_z[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dmat, np.inf)
        k_eff = min(max(1, self.k), len(emb) - 1)
        return np.sort(dmat, axis=1)[:, :k_eff].mean(axis=1).astype(np.float32)

    def _class_radii(self, idx):
        ref = self.train_knn_distances.get(int(idx))
        if ref is None or len(ref) == 0:
            return 0.0, 1.0
        good = float(np.percentile(ref, self.good_percentile))
        bad = float(np.percentile(ref, self.bad_percentile) * self.bad_radius_scale)
        if bad <= good:
            bad = good + max(abs(good) * 0.25, 1e-6)
        return good, bad

    def _distances_to_class(self, z_np, emb):
        z_s = self._standardize(z_np.reshape(1, -1))[0]
        emb_s = self._standardize(emb)
        return np.linalg.norm(emb_s - z_s[None, :], axis=1)

    def _standardize(self, x):
        if self.latent_mean is None or self.latent_std is None:
            return np.asarray(x, dtype=np.float32)
        return (np.asarray(x, dtype=np.float32) - self.latent_mean) / self.latent_std

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


class ProcessFitScorer:
    """Computes per-process fit scores for a given latent vector.

    The fit score combines:
    - Softmax probability P(process | z) from the classifier
    - Normalized silhouette score using process centroids in latent space

    Usage:
        scorer = ProcessFitScorer(model)
        scorer.compute_centroids(dataloader)
        results = scorer.score(z)  # scores for all processes
    """

    def __init__(self, model, class_centroids=None, active_indices=None,
                 transformer=None):
        """Args:
            model: frozen ProcessPredictorHybrid whose .predictor is used
                   to compute probabilities. (Can also be a ProcessPredictor
                   if only the predictor head is needed.)
            class_centroids: optional dict {process_idx: np.ndarray[latent_dim]}
                             of pre-computed class means.
            active_indices: optional list of int — process indices that had
                            training data. Logits for other indices are masked
                            to -inf before softmax so they don't dilute
                            probabilities. If None, all indices are used.
            transformer: optional LatentTransformer for LDA projection before
                         silhouette distance computation.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.centroids = class_centroids if class_centroids is not None else {}
        self.active_indices = active_indices
        self.transformer = transformer

    def compute_centroids(self, dataloader):
        """Compute class centroids by running the encoder over a dataset.

        Also fits an LDA transformer on the collected latents to amplify
        class separation before silhouette distance computation.

        Args:
            dataloader: iterable yielding batches with keys
                        'vertices' and 'process_label' (for PointNet) or
                        'geom_features' and 'process_label' (for MLP).

        Populates self.centroids: {process_idx: np.ndarray[latent_dim]}.
        Populates self.transformer: LatentTransformer (fitted).
        """
        all_latents = []
        all_labels = []
        latents_by_label = {}
        self.model.eval()

        # Detect encoder type
        use_pointnet = hasattr(self.model, 'geom_encoder') and \
                       hasattr(self.model.geom_encoder, 'mlp')

        with torch.no_grad():
            for batch in dataloader:
                if use_pointnet:
                    vertices = batch["vertices"].to(self.device)
                    n_valid = batch.get("n_valid_points", None)
                    if n_valid is not None:
                        n_valid = n_valid.to(self.device)
                    z = self.model.geom_encoder(vertices, n_valid_points=n_valid)
                else:
                    geom = batch["geom_features"].to(self.device)
                    z = self.model.geom_encoder(geom)

                labels = batch["process_label"].to(self.device)

                for i in range(z.shape[0]):
                    lbl = int(labels[i].item())
                    vec = z[i].cpu().numpy()
                    all_latents.append(vec)
                    all_labels.append(lbl)
                    if lbl not in latents_by_label:
                        latents_by_label[lbl] = []
                    latents_by_label[lbl].append(vec)

        self.centroids = {}
        for lbl, vecs in latents_by_label.items():
            self.centroids[lbl] = np.mean(vecs, axis=0)

        # Fit LDA transformer to amplify class separation
        n_classes = len(latents_by_label)
        if len(all_latents) > 0 and n_classes > 1:
            X_all = np.array(all_latents, dtype=np.float32)
            y_all = np.array(all_labels, dtype=np.int32)
            self.transformer = LatentTransformer()
            self.transformer.fit(X_all, y_all)
            print(f"Fitted LDA transformer "
                  f"(latent {X_all.shape[1]} → {self.transformer.W.shape[1]} dim)")
        else:
            self.transformer = None

        print(f"Computed centroids for {n_classes} process classes")
        return self.centroids

    def _centroid_tensor(self, idx):
        """Return centroid for process *idx* as a 2D tensor on the correct device."""
        if idx not in self.centroids:
            return torch.zeros(1, LATENT_DIM, device=self.device)
        return torch.tensor(self.centroids[idx], dtype=torch.float32,
                            device=self.device).unsqueeze(0)

    def score(self, z, target_idx=None):
        """Compute fit scores for one or all processes.

        Args:
            z: [1, latent_dim] tensor on the model's device.
            target_idx: if provided, score only this process.
                        If None, score all processes (filtered by
                        active_indices if set).

        Returns:
            If target_idx is provided: FitResult for that process.
            Otherwise: list of FitResult sorted by fit_score descending.
        """
        self.model.eval()

        probs = self._compute_probs(z)       # [1, n_classes]
        n_classes = probs.shape[1]

        if target_idx is not None:
            results = [self._result_for(z, probs, target_idx)]
        else:
            indices = self.active_indices if self.active_indices is not None else range(n_classes)
            results = [self._result_for(z, probs, i) for i in indices]

            for r in results:
                r.fit_score = FIT_ALPHA * r.probability + FIT_BETA * r.silhouette_score

            results.sort(key=lambda r: r.fit_score, reverse=True)
            for rank, r in enumerate(results, 1):
                r.rank = rank

        return results

    def _compute_probs(self, z):
        """Compute softmax probabilities, masking untrained processes."""
        with torch.no_grad():
            logits = self.model.predictor(z)  # [1, seq_len, n_classes]
            if self.active_indices is not None:
                mask = torch.full_like(logits, -float('inf'))
                for idx in self.active_indices:
                    mask[..., idx] = 0.0
                logits = logits + mask
            probs = F.softmax(logits, dim=-1).mean(dim=1)  # [1, n_classes]
        return probs

    def _result_for(self, z, probs, idx, raw_silhouette=None):
        """Build a FitResult for process *idx*.

        Args:
            z: latent tensor.
            probs: softmax probabilities [1, n_classes].
            idx: target process index.
            raw_silhouette: optional pre-computed raw silhouette. If None,
                            computed here.
        """
        if idx >= probs.shape[1]:
            return FitResult(idx, f"Process {idx}", 0.0, 0.0, 0.0, rank=0)
        prob = float(probs[0, idx].item())
        if raw_silhouette is None:
            silhouette_score, raw_silhouette = self._silhouette_score(z, idx)
        else:
            silhouette_score = float(np.clip((raw_silhouette + 1.0) / 2.0, 0.0, 1.0))
        score = FIT_ALPHA * prob + FIT_BETA * silhouette_score
        name = MANUFACTURING_PROCESSES[idx] if idx < len(MANUFACTURING_PROCESSES) else f"Process {idx}"
        return FitResult(idx, name, prob, silhouette_score, score, rank=0,
                         silhouette_score_raw=raw_silhouette)

    def _silhouette_score(self, z, target_idx):
        """Return normalized and raw silhouette scores for one latent vector.

        Raw silhouette is (b - a) / max(a, b), where a is distance to the
        target process centroid and b is distance to the nearest other active
        process centroid.  The normalized score maps [-1, 1] to [0, 1].
        """
        score_t, raw_t = self._silhouette_score_tensor(z, target_idx)
        return float(score_t.squeeze().item()), float(raw_t.squeeze().item())

    def _silhouette_score_tensor(self, z, target_idx):
        if target_idx not in self.centroids or self.centroids[target_idx] is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        active = self.active_indices if self.active_indices is not None else self.centroids.keys()
        other_indices = [
            idx for idx in active
            if idx != target_idx and idx in self.centroids and self.centroids[idx] is not None
        ]
        if not other_indices:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        target_centroid = torch.tensor(
            self.centroids[target_idx], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        other_centroids = torch.tensor(
            np.stack([self.centroids[idx] for idx in other_indices]),
            dtype=torch.float32,
            device=self.device,
        )

        if self.transformer is not None:
            z_dist = self.transformer.transform(z)
            target_centroid = self.transformer.transform(target_centroid)
            other_centroids = self.transformer.transform(other_centroids)
        else:
            z_dist = z

        a = torch.linalg.norm(z_dist - target_centroid, dim=-1).squeeze()
        b = torch.linalg.norm(z_dist - other_centroids, dim=-1).min()
        raw = (b - a) / torch.clamp(torch.maximum(a, b), min=1e-8)
        normalized = torch.clamp((raw + 1.0) / 2.0, min=0.0, max=1.0)
        return normalized, raw


class LatentDirectionAnalyzer:
    """Finds the latent vector that achieves a target fit score for a process.

    Uses Adam optimization on the latent vector (not model weights) to
    directly minimize (fit_score(z) - target_score)^2.

    Usage:
        analyzer = LatentDirectionAnalyzer(predictor, centroids)
        result = analyzer.solve_for_target_score(z, target_idx=0, target_score=0.90)
    """

    def __init__(self, predictor, class_centroids=None, active_indices=None,
                 transformer=None):
        """Args:
            predictor: frozen ProcessPredictor (transformer head).
            class_centroids: dict {process_idx: np.ndarray[latent_dim]}.
            active_indices: optional list of int — only these process
                            indices are unmasked in the softmax.
            transformer: optional LatentTransformer for LDA projection before
                         silhouette distance computation.
        """
        self.predictor = predictor
        self.device = next(predictor.parameters()).device
        self.centroids = class_centroids if class_centroids is not None else {}
        self.active_indices = active_indices
        self.transformer = transformer

    def solve_for_target_score(self, z, target_idx, target_score,
                               max_iters=100, lr=0.1, tol=0.005):
        """Optimize z to achieve *target_score* for process *target_idx*.

        Handles both increasing (current < target) and decreasing
        (current > target) cases automatically via gradient descent.

        Args:
            z: [1, latent_dim] tensor — the original latent.
            target_idx: int — index of the target process.
            target_score: float between 0 and 1.
            max_iters: max Adam iterations.
            lr: learning rate for the optimizer.
            tol: convergence tolerance.

        Returns:
            dict with keys:
                z_modified: [1, latent_dim] tensor.
                achieved_score: float.
                converged: bool.
                n_iters: int.
        """
        use_silhouette = self._has_silhouette_reference(target_idx)

        # Initialize optimizable latent
        z_opt = nn.Parameter(z.clone().detach())
        optimizer = torch.optim.Adam([z_opt], lr=lr)

        with torch.no_grad():
            prev_score_val = self._compute_score(z, target_idx, use_silhouette).item()

        converged = False
        n_iters = 0
        curr_score = prev_score_val

        for i in range(max_iters):
            optimizer.zero_grad()
            score = self._compute_score(z_opt, target_idx, use_silhouette)
            loss = (score - target_score) ** 2
            loss.backward()
            optimizer.step()

            n_iters = i + 1
            with torch.no_grad():
                curr_score = score.item()
                if abs(curr_score - target_score) < tol:
                    converged = True
                    break

        achieved = curr_score if converged else None
        if not converged:
            with torch.no_grad():
                achieved = self._compute_score(
                    z_opt, target_idx, use_silhouette
                ).item()

        return {
            "z_modified": z_opt.detach(),
            "achieved_score": round(achieved, 4),
            "initial_score": round(prev_score_val, 4),
            "converged": converged,
            "n_iters": n_iters,
        }

    def find_achievable_range(self, z, target_idx, max_iters=100, lr=0.1):
        """Estimate the max and min achievable fit scores for a process.

        Optimizes z toward high and low fit scores under the configured score
        definition.

        Returns:
            dict with keys: min_score, max_score, z_min, z_max.
        """
        # We use a high target to push as far as possible
        res_up = self.solve_for_target_score(
            z, target_idx, target_score=1.0,
            max_iters=max_iters, lr=lr, tol=1e-6,
        )

        res_down = self.solve_for_target_score(
            z, target_idx, target_score=0.0,
            max_iters=max_iters, lr=lr, tol=1e-6,
        )

        return {
            "max_score": res_down["initial_score"]
                if res_down["achieved_score"] < res_down["initial_score"]
                else res_up["achieved_score"],
            "min_score": res_up["initial_score"]
                if res_up["achieved_score"] < res_up["initial_score"]
                else res_down["achieved_score"],
            "z_max": res_up["z_modified"],
            "z_min": res_down["z_modified"],
        }

    def _centroid_tensor(self, idx):
        if idx not in self.centroids:
            return torch.zeros(1, LATENT_DIM, device=self.device)
        return torch.tensor(self.centroids[idx], dtype=torch.float32,
                            device=self.device).unsqueeze(0)

    def _has_silhouette_reference(self, target_idx):
        if target_idx not in self.centroids or self.centroids[target_idx] is None:
            return False
        active = self.active_indices if self.active_indices is not None else self.centroids.keys()
        return any(
            idx != target_idx and idx in self.centroids and self.centroids[idx] is not None
            for idx in active
        )

    def _compute_score(self, z, target_idx, use_silhouette=True):
        """Compute fit score at z for process *target_idx*.

        The silhouette term is normalized from [-1, 1] to [0, 1] so it can be
        combined with softmax probability on the existing score scale.
        """
        logits = self.predictor(z)
        if self.active_indices is not None:
            mask = torch.full_like(logits, -float('inf'))
            for idx in self.active_indices:
                mask[..., idx] = 0.0
            logits = logits + mask
        probs = F.softmax(logits, dim=-1).mean(dim=1)
        prob = probs[0, target_idx]

        if use_silhouette:
            silhouette_score = self._silhouette_score(z, target_idx)
        else:
            silhouette_score = torch.tensor(0.0, device=self.device)

        return FIT_ALPHA * prob + FIT_BETA * silhouette_score

    def _silhouette_score(self, z, target_idx):
        target_centroid = self._centroid_tensor(target_idx)
        active = self.active_indices if self.active_indices is not None else self.centroids.keys()
        other_indices = [
            idx for idx in active
            if idx != target_idx and idx in self.centroids and self.centroids[idx] is not None
        ]
        if not other_indices:
            return torch.tensor(0.0, device=self.device)

        other_centroids = torch.tensor(
            np.stack([self.centroids[idx] for idx in other_indices]),
            dtype=torch.float32,
            device=self.device,
        )

        if self.transformer is not None:
            z_dist = self.transformer.transform(z)
            target_centroid = self.transformer.transform(target_centroid)
            other_centroids = self.transformer.transform(other_centroids)
        else:
            z_dist = z

        a = torch.linalg.norm(z_dist - target_centroid, dim=-1).squeeze()
        b = torch.linalg.norm(z_dist - other_centroids, dim=-1).min()
        raw = (b - a) / torch.clamp(torch.maximum(a, b), min=1e-8)
        return torch.clamp((raw + 1.0) / 2.0, min=0.0, max=1.0)


class DesignExplainer:
    """Decodes latent perturbations into physical design suggestions.

    Takes the original and modified latents, runs them through the
    trained decoders, and produces human-readable explanations.

    Usage:
        explainer = DesignExplainer(feature_decoder, pc_decoder)
        report = explainer.explain(z_orig, z_mod, target_idx=0, target_score=0.90)

    If the decoders were trained with feature standardisation (see
    decoder_training.FeatureStandardizer), pass the standardizer so decoded
    outputs are mapped back to the original physical scale.
    """

    def __init__(self, feature_decoder=None, pointcloud_decoder=None,
                 feature_standardizer=None):
        """Args:
            feature_decoder: trained LinearDecoder (128 → 64 features).
            pointcloud_decoder: trained PointCloudDecoder (128 → [512, 3]).
                                Mesh-only; set to None for image inputs.
            feature_standardizer: object with .mean (ndarray[64]) and
                                  .std (ndarray[64]) attributes, or None.
        """
        self.feature_decoder = feature_decoder
        self.pointcloud_decoder = pointcloud_decoder
        self.standardizer = feature_standardizer
        self.device = None
        if feature_decoder is not None:
            self.device = next(feature_decoder.parameters()).device
        elif pointcloud_decoder is not None:
            self.device = next(pointcloud_decoder.parameters()).device

    def decode_features(self, z):
        """Decode a latent vector to 64-dim features.

        If a feature_standardizer was provided, the decoder output is
        un-standardised back to the original physical scale.

        Returns np.ndarray of shape [64] or None if no decoder.
        """
        if self.feature_decoder is None:
            return None
        self.feature_decoder.eval()
        with torch.no_grad():
            z_t = z if isinstance(z, torch.Tensor) else torch.from_numpy(z).float()
            if z_t.device != self.device:
                z_t = z_t.to(self.device)
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
            feat = self.feature_decoder(z_t).squeeze(0).cpu().numpy()
        # Un-standardize to physical scale if standardizer was provided
        if self.standardizer is not None and feat is not None:
            feat = feat * self.standardizer.std + self.standardizer.mean
        return feat

    def decode_vertices(self, z):
        """Decode a latent vector to a point cloud.

        Returns np.ndarray of shape [max_points, 3] or None.
        Only meaningful from mesh input.
        """
        if self.pointcloud_decoder is None:
            return None
        self.pointcloud_decoder.eval()
        with torch.no_grad():
            z_t = z if isinstance(z, torch.Tensor) else torch.from_numpy(z).float()
            if z_t.device != self.device:
                z_t = z_t.to(self.device)
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
            verts = self.pointcloud_decoder(z_t).squeeze(0).cpu().numpy()
        return verts

    def visualize_pointcloud(self, z_original, z_modified, output_stem=None, mesh_path=None):
        if self.pointcloud_decoder is None:
            return None
        verts_orig = self.decode_vertices(z_original)
        verts_mod = self.decode_vertices(z_modified)
        if verts_orig is None or verts_mod is None:
            return None
        displacement = np.linalg.norm(verts_mod - verts_orig, axis=1)
        colors = displacement_to_color(displacement)

        mesh_vertices = None
        mesh_faces = None
        if mesh_path:
            try:
                from gmdl.decoder_training import load_mat_mesh
                mesh_vertices, mesh_faces = load_mat_mesh(str(mesh_path))
                if mesh_vertices.shape[0] == 3 and mesh_vertices.shape[1] != 3:
                    mesh_vertices = mesh_vertices.T
            except Exception:
                pass

        if output_stem:
            export_colored_pointcloud_ply(verts_mod, colors, output_stem + ".ply")
            if mesh_vertices is not None and mesh_faces is not None:
                export_gray_mesh_ply(mesh_vertices, mesh_faces, output_stem + "_mesh.ply")
            render_pointcloud_mpl(verts_mod, colors, displacement, output_stem + ".png",
                                  mesh_vertices=mesh_vertices, mesh_faces=mesh_faces)
        return {
            "displacements": displacement,
            "colors": colors,
            "points_modified": verts_mod,
        }

    def explain(self, z_original, z_modified, target_idx, target_score,
                achieved_score=None):
        """Generate a structured explanation of latent-space changes.

        Args:
            z_original: original latent vector.
            z_modified: optimized latent vector (from solve_for_target_score).
            target_idx: int — target process index.
            target_score: float — user's target fit score.
            achieved_score: optional — actual achieved score.

        Returns:
            dict with keys: target_process, target_score, achieved_score,
                            current_score, status, feature_changes,
                            vertex_changes, suggestions.
        """
        target_name = MANUFACTURING_PROCESSES[target_idx] \
            if target_idx < len(MANUFACTURING_PROCESSES) else f"Process {target_idx}"

        # Decode features for original and modified
        feat_orig = self.decode_features(z_original)
        feat_mod = self.decode_features(z_modified)

        feature_changes = {}
        if feat_orig is not None and feat_mod is not None:
            for i in range(min(len(FEATURE_NAMES), len(feat_orig))):
                old_val = float(feat_orig[i])
                new_val = float(feat_mod[i])
                delta = new_val - old_val
                pct = (delta / abs(old_val) * 100) if abs(old_val) > 1e-8 else 0.0
                feature_changes[FEATURE_NAMES[i]] = {
                    "old": round(old_val, 4),
                    "new": round(new_val, 4),
                    "delta": round(delta, 4),
                    "pct": round(pct, 1),
                }

        # Decode vertices (point cloud, mesh-only)
        verts_orig = self.decode_vertices(z_original)
        verts_mod = self.decode_vertices(z_modified)
        vertex_changes = None
        if verts_orig is not None and verts_mod is not None:
            # Summary stats on vertex changes
            displacement = np.linalg.norm(verts_mod - verts_orig, axis=1)
            centroid_shift = verts_mod.mean(axis=0) - verts_orig.mean(axis=0)
            vertex_changes = {
                "centroid_shift_mm": {
                    "x": round(float(centroid_shift[0]), 3),
                    "y": round(float(centroid_shift[1]), 3),
                    "z": round(float(centroid_shift[2]), 3),
                },
                "max_vertex_displacement_mm": round(float(displacement.max()), 3),
                "mean_vertex_displacement_mm": round(float(displacement.mean()), 3),
            }

        # Generate suggestions from process-specific rules
        # For each "key" feature (volume, aspect ratio, compactness, etc.),
        # check whether the decoded delta exceeds a 5 % threshold.  If so,
        # look up the process-specific suggestion bucket for the direction
        # of change ("high" / "low") and pick one suggestion from it.
        suggestions = []
        if self.feature_decoder is not None and feat_orig is not None and feat_mod is not None:
            # Key features (indices into the 64-dim vector):
            # 0=volume, 1=surface_area, 2=aspect_ratio, 3=compactness,
            # 57=sphericity, 40=SA/volume_ratio
            key_indices = [0, 1, 2, 3, 57, 40]
            for ki in key_indices:
                pct = feature_changes.get(FEATURE_NAMES[ki], {}).get("pct", 0)
                if abs(pct) < 5:     # skip changes smaller than 5 %
                    continue
                suggestion_key = "high" if pct > 0 else "low"
                rule_bucket = PROCESS_SUGGESTIONS.get(target_idx, {}).get(suggestion_key, [])
                if rule_bucket:
                    # Cycle through bucket entries to avoid repeating the same text
                    suggestions.append(rule_bucket[len(suggestions) % len(rule_bucket)])

            # Deduplicate while preserving order, limit to 5 suggestions
            suggestions = list(OrderedDict.fromkeys(suggestions))[:5]

        # Current score from feature_changes isn't stored directly;
        # the caller provides it via achieved_score
        return {
            "target_process": target_name,
            "target_score": target_score,
            "achieved_score": achieved_score,
            "feature_changes": feature_changes,
            "vertex_changes": vertex_changes,
            "suggestions": suggestions,
        }
