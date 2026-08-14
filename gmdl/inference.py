# =============================================================================
# gmdl/inference.py — High-Level Inference API
# =============================================================================
#
# Provides the public-facing ManufacturingProcessPredictor class and the
# standalone extract_image_features() helper.
#
# ManufacturingProcessPredictor:
#     High-level API for predicting manufacturing process from CAD .step
#     files, .mat mesh files, or images.  Handles config loading, weight
#     loading, feature extraction, and ranked output generation.
#
#     Modes:
#         .predict()               — standard ranked prediction
#         .predict_with_analysis() — prediction + fit scoring + latent
#                                    optimisation + design explanation
#
#     The analysis modes require additional trained components:
#         load_decoders()          — feature & point-cloud decoders
#         load_centroids()         — pre-computed class centroids
#         compute_centroids()      — compute centroids from a dataset
#
#     Usage:
#         predictor = ManufacturingProcessPredictor()
#         predictor.load_weights("train_log/default/model/latest.pt")
#         results = predictor.predict(step_path="part.step")
#         # results: [(process_name, confidence), ...] sorted descending
#
# extract_image_features():
#     Standalone helper that loads an image, runs it through a fresh
#     ImageEncoder, and returns the raw latent vector as a numpy array.
#     Useful for feature extraction without loading the full model.
# =============================================================================

import os
import json
import warnings
import numpy as np

import torch
import torch.nn.functional as F

from gmdl.constants import (
    GEOM_FEATURE_DIM, LATENT_DIM, N_PROCESSES,
    MANUFACTURING_PROCESSES, MAX_POINTS,
)
from gmdl.config import ConfigGeometryEncoder, ConfigProcessPredictor
from gmdl.encoders import ImageEncoder
from gmdl.predictor import ProcessPredictorHybrid
from gmdl.utils import get_active_processes
from gmdl.analysis import (
    ProcessFitScorer, LatentDirectionAnalyzer, DesignExplainer, LatentTransformer,
    KNNFitScorer,
)
from gmdl.decoders import LinearDecoder, PointCloudDecoder


class ManufacturingProcessPredictor:
    """High-level inference interface for manufacturing process prediction.

    Supports three input modalities:
        1. CAD .step files   → geometry features extracted via cadhandler
        2. .mat mesh files   → raw vertices for PointNet encoder
        3. Image files       → encoded via the CNN image encoder

    The appropriate encoder is selected automatically based on the input
    type and the encoder_type stored in the checkpoint config.

    Attributes:
        device:            torch.device (cuda or cpu)
        model:             ProcessPredictorHybrid (encoder + predictor)
        image_encoder:     ImageEncoder (loaded from checkpoint if available)
        geom_cfg:          ConfigGeometryEncoder
        proc_cfg:          ConfigProcessPredictor
        use_pointnet:      bool — True if the loaded model uses PointNet
        n_processes:       int — number of active process classes
        feature_decoder:   LinearDecoder (loaded via load_decoders, optional)
        pc_decoder:        PointCloudDecoder (loaded via load_decoders, optional)
        class_centroids:   dict {process_idx: np.ndarray[latent_dim]}
                           of class-mean latent vectors (optional)
    """

    def __init__(self, cfg_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine active process list (from options.json if available)
        self.active_processes = get_active_processes()
        self.n_processes = len(self.active_processes)

        # Load or default config
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg_dict = json.load(f)
            self.geom_cfg = ConfigGeometryEncoder.from_dict(
                cfg_dict.get("geom_encoder", {})
            )
            self.proc_cfg = ConfigProcessPredictor.from_dict(
                cfg_dict.get("process_predictor", {})
            )
        else:
            self.geom_cfg = ConfigGeometryEncoder()
            self.proc_cfg = ConfigProcessPredictor()

        self.proc_cfg.n_processes = self.n_processes
        self.use_pointnet = getattr(self.geom_cfg, "encoder_type", "mlp") == "pointnet"

        # Build the combined encoder + predictor model
        self.model = ProcessPredictorHybrid(self.geom_cfg, self.proc_cfg).to(self.device)
        self.model.eval()

        # Image encoder is loaded separately (optional, from checkpoint)
        self.image_encoder = None

        # Analysis components (loaded lazily)
        self.feature_decoder = None
        self.pc_decoder = None
        self.class_centroids = {}
        self.knn_scorer = None

    def load_weights(self, path):
        """Load model weights from a checkpoint file.

        Automatically rebuilds the encoder/predictor if the saved encoder type
        or number of processes differs from the current config (e.g. when
        loading a PointNet checkpoint into an MLP-configured instance).

        Also loads the image encoder if it was saved during training.
        """
        if not os.path.exists(path):
            warnings.warn(f"Weights file not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_cfg = ckpt.get("cfg", {})
        saved_geom_cfg_dict = saved_cfg.get("geom_encoder", {})
        saved_encoder_type = saved_geom_cfg_dict.get("encoder_type", "mlp")
        saved_n_proc = saved_cfg.get("process_predictor", {}).get("n_processes", N_PROCESSES)

        # Rebuild encoder/predictor if encoder type or n_processes changed
        needs_rebuild = False
        if saved_encoder_type != getattr(self.geom_cfg, "encoder_type", "mlp"):
            self.geom_cfg = (
                ConfigGeometryEncoder.from_dict(saved_geom_cfg_dict)
                if saved_geom_cfg_dict else self.geom_cfg
            )
            self.geom_cfg.encoder_type = saved_encoder_type
            self.use_pointnet = saved_encoder_type == "pointnet"
            needs_rebuild = True
        if saved_n_proc != self.proc_cfg.n_processes:
            self.proc_cfg.n_processes = saved_n_proc
            self.n_processes = saved_n_proc
            needs_rebuild = True
        if needs_rebuild:
            self.model = ProcessPredictorHybrid(self.geom_cfg, self.proc_cfg).to(self.device)

        # Load encoder and predictor weights
        self.model.geom_encoder.load_state_dict(ckpt["geom_encoder"])
        self.model.predictor.load_state_dict(ckpt["predictor"])

        # Load image encoder if available in checkpoint
        if "image_encoder" in ckpt:
            self.image_encoder = ImageEncoder(self.geom_cfg).to(self.device)
            self.image_encoder.load_state_dict(ckpt["image_encoder"])
            self.image_encoder.eval()
        else:
            self.image_encoder = None

        # Detect which process labels actually had training data
        self.data_root = saved_cfg.get("data_root", "data")
        self.active_indices = self._detect_active_indices(self.data_root)

    def _detect_active_indices(self, data_root):
        """Scan training manifest for unique process labels.

        Returns a sorted list of ints representing process indices that
        appeared in the training data. Returns None if the manifest can't
        be read (in which case all indices are treated as active).
        """
        manifest_path = os.path.join(data_root, "train.json")
        if not os.path.exists(manifest_path):
            warnings.warn(f"Training manifest not found: {manifest_path}, "
                          "all process indices will be used")
            return None
        try:
            with open(manifest_path) as f:
                samples = json.load(f)
            labels = sorted({s["process_label"] for s in samples})
            if labels:
                print(f"Detected {len(labels)} process classes "
                      f"with training data: {labels}")
            return labels
        except Exception as e:
            warnings.warn(f"Failed to scan {manifest_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Feature extraction helpers (private)
    # ------------------------------------------------------------------

    def _extract_cad_features(self, step_path):
        """Extract geometry features from a .step file via cadhandler.

        Returns a numpy array of shape [GEOM_FEATURE_DIM] with the first
        7 entries being volume, surface area, aspect ratio, compactness,
        and bounding box dimensions (x, y, z), zero-padded to 64 dims.
        """
        from cadhandler import process_step
        features = json.loads(process_step(step_path))
        geom = np.array([
            features.get("volume mm3", 0),
            features.get("surface area mm2", 0),
            features.get("aspect ratio", 0),
            features.get("compactness", 0),
            features.get("dimensions mm", {}).get("x", 0),
            features.get("dimensions mm", {}).get("y", 0),
            features.get("dimensions mm", {}).get("z", 0),
        ], dtype=np.float32)
        geom = np.pad(
            geom, (0, max(0, GEOM_FEATURE_DIM - len(geom))),
            mode="constant",
        )[:GEOM_FEATURE_DIM]
        return geom

    def _extract_mesh_features(self, mesh_path):
        """Extract 64-dim geometry features from a .mat mesh for MLP encoder.

        Uses the same mesh-to-feature pipeline as decoder_training so the
        feature vector is consistent with what the MLP encoder was trained on.
        """
        from gmdl.decoder_training import load_mat_mesh, mesh_to_geom_features
        try:
            pts, faces = load_mat_mesh(str(mesh_path))
            return mesh_to_geom_features(pts, faces)
        except Exception as e:
            warnings.warn(f"Failed to extract features from {mesh_path}: {e}")
            return np.zeros(GEOM_FEATURE_DIM, dtype=np.float32)

    def _extract_vertices(self, mesh_path):
        """Extract vertices from a .mat file for PointNet inference.

        Returns a numpy array of shape [max_points, 3], zero-padded.
        """
        max_pts = getattr(self.geom_cfg, "max_points", MAX_POINTS)
        vertices = np.zeros((max_pts, 3), dtype=np.float32)
        try:
            import scipy.io as sio
            data = sio.loadmat(str(mesh_path))
            pts = data.get("point3D")
            if pts is not None:
                pts = np.asarray(pts, dtype=np.float32)
                # Normalise MATLAB (3, N) layout to (N, 3)
                if pts.shape[0] == 3 and pts.shape[1] != 3:
                    pts = pts.T
                n = min(len(pts), max_pts)
                vertices[:n] = pts[:n]
        except Exception as e:
            warnings.warn(f"Failed to load vertices from {mesh_path}: {e}")
        return vertices

    def _extract_image_features(self, image_path):
        """Encode an image file into a latent vector.

        Returns a numpy array of shape [latent_dim], or None on failure.
        Uses the loaded image encoder if available; otherwise creates a
        temporary one (untrained weights — for architecture testing only).
        """
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(image_path).convert("RGB")
            transform = T.Compose([
                T.Resize((self.geom_cfg.img_size, self.geom_cfg.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.224]),
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            if self.image_encoder is not None:
                self.image_encoder.eval()
                with torch.no_grad():
                    return self.image_encoder(img_tensor).squeeze(0).cpu().numpy()
            else:
                # Fallback: use a fresh (untrained) encoder
                img_encoder = ImageEncoder(self.geom_cfg).to(self.device)
                return img_encoder(img_tensor).squeeze(0).detach().cpu().numpy()
        except ImportError:
            warnings.warn("PIL/torchvision not available for image processing")
        return None

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(self, step_path=None, image_path=None, mesh_path=None):
        """Predict the manufacturing process for a given input.

        Exactly one of ``step_path``, ``mesh_path``, or ``image_path``
        must be provided.

        Args:
            step_path:  path to a CAD .step file (MLP encoder)
            mesh_path:  path to a .mat mesh file (PointNet encoder)
            image_path: path to an image file (.jpg, .png)

        Returns:
            list of (process_name, confidence) tuples, sorted by confidence
            descending.  Up to ``self.n_processes`` unique entries.

        Example:
            predictor = ManufacturingProcessPredictor()
            predictor.load_weights("train_log/default/model/latest.pt")
            results = predictor.predict(step_path="part.step")
            # results: [("3-axis CNC machining", 0.82), ("Casting", 0.11), ...]
        """
        n_valid = None
        use_geom_encoder = False

        if mesh_path:
            if self.use_pointnet:
                geom = self._extract_vertices(mesh_path)
                max_pts = getattr(self.geom_cfg, "max_points", MAX_POINTS)
                n_valid_val = int(np.count_nonzero(geom.any(axis=1)))
                n_valid = torch.tensor(
                    [min(n_valid_val, max_pts)], dtype=torch.long, device=self.device
                )
                geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                geom = self._extract_mesh_features(mesh_path)
                geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).to(self.device)
            use_geom_encoder = True

        elif step_path:
            geom = self._extract_cad_features(step_path)
            geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).to(self.device)
            use_geom_encoder = True

        elif image_path:
            latent = self._extract_image_features(image_path)
            z = (torch.tensor(latent, dtype=torch.float32).unsqueeze(0).to(self.device)
                 if latent is not None else None)
            use_geom_encoder = False

        else:
            raise ValueError("Provide one of step_path, mesh_path, or image_path")

        # Run inference
        self.model.eval()
        with torch.no_grad():
            if use_geom_encoder:
                z = self.model.geom_encoder(geom_tensor, n_valid_points=n_valid)
            logits = self.model.predictor(z)
            # Mask untrained processes so they don't dilute probabilities
            active_idx = getattr(self, 'active_indices', None)
            if active_idx is not None:
                mask = torch.full_like(logits, -float('inf'))
                for idx in active_idx:
                    mask[..., idx] = 0.0
                logits = logits + mask
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Rank predictions across all sequence positions, deduplicating
        ranked = []
        seen = set()
        for step_idx in range(probs.shape[0]):
            for p_idx in np.argsort(probs[step_idx])[::-1]:
                p_name = MANUFACTURING_PROCESSES[p_idx]
                if p_name not in seen:
                    ranked.append((p_name, float(probs[step_idx][p_idx])))
                    seen.add(p_name)
            if len(ranked) >= (len(active_idx) if active_idx else self.n_processes):
                break
        return ranked

    # ------------------------------------------------------------------
    # Decoder loading
    # ------------------------------------------------------------------

    def load_decoders(self, path):
        """Load trained decoder weights from a separate checkpoint file.

        The checkpoint should contain 'feature_decoder' and optionally
        'pointcloud_decoder' state dicts, plus 'standardizer_mean' and
        'standardizer_std' arrays, as saved by train_decoders().
        """
        if not os.path.exists(path):
            warnings.warn(f"Decoder checkpoint not found: {path}")
            return

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        latent_dim = getattr(self.geom_cfg, "latent_dim", LATENT_DIM)
        feature_dim = getattr(self.geom_cfg, "feature_dim", GEOM_FEATURE_DIM)
        max_points = getattr(self.geom_cfg, "max_points", MAX_POINTS)

        self.feature_decoder = LinearDecoder(latent_dim, feature_dim).to(self.device)
        if "feature_decoder" in ckpt:
            self.feature_decoder.load_state_dict(ckpt["feature_decoder"])
        self.feature_decoder.eval()

        if "pointcloud_decoder" in ckpt:
            self.pc_decoder = PointCloudDecoder(latent_dim, max_points).to(self.device)
            self.pc_decoder.load_state_dict(ckpt["pointcloud_decoder"])
            self.pc_decoder.eval()
        else:
            self.pc_decoder = None

        # Load feature standardizer (used to un-standardize decoder outputs)
        if "standardizer_mean" in ckpt and "standardizer_std" in ckpt:
            self.feature_standardizer = type("stdz", (), {})()
            self.feature_standardizer.mean = ckpt["standardizer_mean"]
            self.feature_standardizer.std = ckpt["standardizer_std"]
            print(f"Loaded feature standardizer "
                  f"(mean={self.feature_standardizer.mean[:3].round(1)}...)")
        else:
            self.feature_standardizer = None

    # ------------------------------------------------------------------
    # Centroid management
    # ------------------------------------------------------------------

    def load_centroids(self, path):
        """Load pre-computed class centroids + LDA projection from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self.class_centroids = {}
        lda_W = None
        rbf_gamma = None
        for k, v in data.items():
            if k == "lda_projection":
                lda_W = v
            elif k == "rbf_gamma":
                rbf_gamma = float(v.item())
            else:
                self.class_centroids[int(k)] = v
        if lda_W is not None:
            t = LatentTransformer()
            t.W = lda_W
            if rbf_gamma is not None:
                t.rbf_gamma = rbf_gamma
            self.latent_transformer = t
            print(f"Loaded LDA projection ({lda_W.shape[0]} → {lda_W.shape[1]} dim)")
        print(f"Loaded centroids for {len(self.class_centroids)} classes")

    def save_centroids(self, path):
        """Save computed class centroids + LDA projection to a .npz file."""
        if hasattr(self, 'class_centroids') and self.class_centroids:
            save_dict = {str(k): v for k, v in self.class_centroids.items()}
            transformer = getattr(self, 'latent_transformer', None)
            if transformer is not None and transformer.W is not None:
                save_dict["lda_projection"] = transformer.W
                save_dict["rbf_gamma"] = np.array([transformer.rbf_gamma])
            np.savez_compressed(path, **save_dict)
            print(f"Saved centroids + LDA to {path}")

    def load_latent_refs(self, path):
        """Load KNN latent reference sets from a .npz file."""
        data = np.load(path, allow_pickle=True)
        class_embeddings = {}
        train_knn_distances = {}
        latent_mean = None
        latent_std = None
        k = 5
        tail_alpha = 1.0
        for key, value in data.items():
            if key.startswith("embeddings_"):
                class_embeddings[int(key.split("_", 1)[1])] = value.astype(np.float32)
            elif key.startswith("train_knn_distances_"):
                train_knn_distances[int(key.rsplit("_", 1)[1])] = value.astype(np.float32)
            elif key == "latent_mean":
                latent_mean = value.astype(np.float32)
            elif key == "latent_std":
                latent_std = value.astype(np.float32)
            elif key == "k":
                k = int(value.item())
            elif key == "tail_alpha":
                tail_alpha = float(value.item())

        self.knn_scorer = KNNFitScorer(
            class_embeddings=class_embeddings,
            latent_mean=latent_mean,
            latent_std=latent_std,
            train_knn_distances=train_knn_distances,
            k=k,
            tail_alpha=tail_alpha,
            active_indices=getattr(self, 'active_indices', None),
        )
        print(f"Loaded KNN latent refs for {len(class_embeddings)} classes")

    def save_latent_refs(self, path):
        """Save KNN latent reference sets to a .npz file."""
        if self.knn_scorer is None or not self.knn_scorer.class_embeddings:
            return
        save_dict = {
            "latent_mean": self.knn_scorer.latent_mean,
            "latent_std": self.knn_scorer.latent_std,
            "k": np.array([self.knn_scorer.k], dtype=np.int32),
            "tail_alpha": np.array([self.knn_scorer.tail_alpha], dtype=np.float32),
        }
        for idx, emb in self.knn_scorer.class_embeddings.items():
            save_dict[f"embeddings_{idx}"] = emb
        for idx, dists in self.knn_scorer.train_knn_distances.items():
            save_dict[f"train_knn_distances_{idx}"] = dists
        np.savez_compressed(path, **save_dict)
        print(f"Saved KNN latent refs to {path}")

    def compute_latent_refs(self, dataloader=None, manifest_path=None, k=5):
        """Compute KNN latent reference sets from training data."""
        if dataloader is None and manifest_path is not None:
            from gmdl.datasets import ManufacturingDataset
            from torch.utils.data import DataLoader
            data_root = getattr(self, 'data_root', 'data')
            ds = ManufacturingDataset(
                data_root,
                split="train",
                use_pointnet=self.use_pointnet,
                max_points=getattr(self.geom_cfg, "max_points", MAX_POINTS),
            )
            dataloader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        if dataloader is None:
            warnings.warn("No data source for latent reference computation")
            return

        all_latents = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if self.use_pointnet:
                    geom = batch["vertices"].to(self.device)
                    n_valid = batch.get("n_valid_points", None)
                    if n_valid is not None:
                        n_valid = n_valid.to(self.device)
                    z = self.model.geom_encoder(geom, n_valid_points=n_valid)
                else:
                    geom = batch["geom_features"].to(self.device)
                    z = self.model.geom_encoder(geom)
                labels = batch["process_label"].to(self.device)
                all_latents.append(z.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_latents:
            warnings.warn("No samples available for latent reference computation")
            return

        latents = np.concatenate(all_latents, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        self.knn_scorer = KNNFitScorer(
            k=k,
            active_indices=getattr(self, 'active_indices', None),
        ).fit(latents, labels)
        print(f"Computed KNN latent refs for {len(self.knn_scorer.class_embeddings)} classes")

    def compute_centroids(self, dataloader=None, manifest_path=None):
        """Compute class centroids from a data source.

        Centroids are the per-class mean of the latent vectors produced by
        the geometry encoder.  They are used by ProcessFitScorer to compute
        the silhouette component of the fit score.

        Two ways to call:
            1. Provide an existing ``dataloader`` (e.g. from train set).
            2. Provide ``manifest_path`` pointing to a JSON manifest file;
               the method will create its own DecoderDataset + DataLoader.

        Args:
            dataloader:     existing DataLoader yielding
                            (vertices, geom_features, n_valid).
            manifest_path:  path to JSON manifest (e.g. "data/train.json").
        """
        if dataloader is None and manifest_path is not None:
            from gmdl.datasets import ManufacturingDataset
            from torch.utils.data import DataLoader
            data_root = getattr(self, 'data_root', 'data')
            ds = ManufacturingDataset(
                data_root,
                split="train",
                use_pointnet=self.use_pointnet,
                max_points=getattr(self.geom_cfg, "max_points", MAX_POINTS),
            )
            dataloader = DataLoader(
                ds, batch_size=32, shuffle=False, num_workers=4
            )

        if dataloader is None:
            warnings.warn("No data source for centroid computation")
            return

        scorer = ProcessFitScorer(
            self.model,
            active_indices=getattr(self, 'active_indices', None),
        )
        scorer.compute_centroids(dataloader)
        self.class_centroids = scorer.centroids
        if scorer.transformer is not None:
            self.latent_transformer = scorer.transformer

    # ------------------------------------------------------------------
    # Analysis API
    # ------------------------------------------------------------------

    def predict_with_analysis(self, step_path=None, mesh_path=None, image_path=None,
                              target_process=None, target_score=None, find_best=False,
                              output_viz=None):
        """Predict with detailed fit scoring and optional improvement analysis.

        Args:
            step_path:      CAD .step file (MLP encoder).
            mesh_path:      .mat mesh file (PointNet encoder).
            image_path:     image file (CNN encoder).
            target_process: process name or index to analyze
                            (e.g. "3-axis CNC machining").
            target_score:   target fit score to reach (0.0 to 1.0).
            find_best:      if True, find the max achievable score for
                            target_process instead of optimising to a
                            specific value.

        Returns:
            dict with keys:

            ``ranked_processes``
                list of {process_name, probability, silhouette_score,
                fit_score, rank} for all processes, sorted by fit_score
                descending.

            ``latent``
                [latent_dim] numpy array of the original latent code.

            ``input_type``
                "mesh", "step", or "image".

            ``analysis`` (only if target_process was provided)
                dict with:
                    target_process        : str
                    target_score          : float (if ``target_score`` given)
                    achieved_score        : float
                    initial_score         : float
                    converged             : bool
                    n_optimization_iters  : int
                    feature_changes       : dict {name: {old, new, delta, pct}}
                    vertex_changes        : dict (mesh-only) or None
                    suggestions           : list[str]
                    achievable_range      : {min_score, max_score}
                                           (only if ``find_best``)
                    best_achievable_score : float (only if ``find_best``)
        """
        # --- Get the latent vector ---
        n_valid = None
        use_geom_encoder = False
        input_type = None

        if mesh_path:
            if self.use_pointnet:
                geom = self._extract_vertices(mesh_path)
                max_pts = getattr(self.geom_cfg, "max_points", MAX_POINTS)
                n_valid_val = int(np.count_nonzero(geom.any(axis=1)))
                n_valid = torch.tensor(
                    [min(n_valid_val, max_pts)], dtype=torch.long, device=self.device
                )
                geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                geom = self._extract_mesh_features(mesh_path)
                geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).to(self.device)
            use_geom_encoder = True
            input_type = "mesh"
        elif step_path:
            geom = self._extract_cad_features(step_path)
            geom_tensor = torch.tensor(geom, dtype=torch.float32).unsqueeze(0).to(self.device)
            use_geom_encoder = True
            input_type = "step"
        elif image_path:
            latent = self._extract_image_features(image_path)
            z = (torch.tensor(latent, dtype=torch.float32).unsqueeze(0).to(self.device)
                 if latent is not None else None)
            use_geom_encoder = False
            input_type = "image"
        else:
            raise ValueError("Provide one of step_path, mesh_path, or image_path")

        self.model.eval()
        with torch.no_grad():
            if use_geom_encoder:
                z = self.model.geom_encoder(geom_tensor, n_valid_points=n_valid)

        # --- Classifier probabilities for reporting ---
        active_idx = getattr(self, 'active_indices', None)
        with torch.no_grad():
            logits = self.model.predictor(z)
            if active_idx is not None:
                mask = torch.full_like(logits, -float('inf'))
                for idx in active_idx:
                    mask[..., idx] = 0.0
                logits = logits + mask
            probabilities = F.softmax(logits, dim=-1).mean(dim=1).squeeze(0)

        # --- Fit scoring ---
        if self.knn_scorer is not None and self.knn_scorer.class_embeddings:
            self.knn_scorer.active_indices = active_idx
            all_scores = self.knn_scorer.score(z, probabilities=probabilities)
            fit_method = "knn_class_support"
        else:
            centroids = getattr(self, 'class_centroids', {})
            transformer = getattr(self, 'latent_transformer', None)
            scorer = ProcessFitScorer(self.model, class_centroids=centroids,
                                      active_indices=active_idx,
                                      transformer=transformer)
            all_scores = [r.to_dict() for r in scorer.score(z)]
            fit_method = "centroid_silhouette_fallback"

        # --- Resolve target process ---
        target_idx = None
        if target_process is not None:
            if isinstance(target_process, str):
                try:
                    target_idx = MANUFACTURING_PROCESSES.index(target_process)
                except ValueError:
                    # Try to match by case-insensitive prefix
                    for i, name in enumerate(MANUFACTURING_PROCESSES):
                        if target_process.lower() in name.lower():
                            target_idx = i
                            break
                    if target_idx is None:
                        warnings.warn(f"Unknown process '{target_process}'")
            else:
                target_idx = int(target_process)

        result = {
            "ranked_processes": all_scores,
            "latent": z.squeeze(0).cpu().numpy(),
            "input_type": input_type,
            "fit_method": fit_method,
        }

        # --- Optional target-score analysis ---
        if target_idx is not None and (
            self.knn_scorer is not None or target_score is not None or find_best
        ):
            stdz = getattr(self, 'feature_standardizer', None)
            explainer = DesignExplainer(
                feature_decoder=getattr(self, 'feature_decoder', None),
                pointcloud_decoder=getattr(self, 'pc_decoder', None)
                if input_type == "mesh" else None,
                feature_standardizer=stdz,
            )

            analysis = {"target_process": MANUFACTURING_PROCESSES[target_idx]}

            if self.knn_scorer is not None and self.knn_scorer.class_embeddings:
                target_fit = self.knn_scorer.score(z, target_idx=target_idx,
                                                   probabilities=probabilities)
                z_target_np, direction_np, nn_dists = self.knn_scorer.suggest_target(z, target_idx)
                z_target = torch.tensor(z_target_np, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                target_achieved = self.knn_scorer.score(z_target, target_idx=target_idx,
                                                        probabilities=probabilities)
                change_needed = target_fit["fit_score"] < 0.95
                analysis.update(target_fit)
                analysis.update({
                    "change_needed": change_needed,
                    "latent_direction_norm": round(float(np.linalg.norm(direction_np)), 6),
                })

                if find_best:
                    analysis["best_achievable_score"] = 1.0
                    analysis["achievable_range"] = {
                        "min_score": 0.0,
                        "max_score": 1.0,
                    }

                if target_score is not None:
                    analysis["target_score"] = target_score
                    analysis["achieved_score"] = (
                        target_achieved["fit_score"] if change_needed else target_fit["fit_score"]
                    )
                    analysis["initial_score"] = target_fit["fit_score"]
                    analysis["converged"] = bool(analysis["achieved_score"] >= target_score)
                    analysis["n_optimization_iters"] = 0

                if change_needed:
                    explanation = explainer.explain(
                        z, z_target, target_idx,
                        target_score=target_score,
                        achieved_score=analysis.get("achieved_score"),
                    )
                    analysis.update(explanation)
                else:
                    z_target = z
                    analysis.update({
                        "feature_changes": {},
                        "vertex_changes": None,
                        "suggestions": ["No design change recommended; latent is inside the target class support."],
                    })
            else:
                centroids = getattr(self, 'class_centroids', {})
                transformer = getattr(self, 'latent_transformer', None)
                analyzer = LatentDirectionAnalyzer(
                    self.model.predictor,
                    class_centroids=centroids,
                    active_indices=active_idx,
                    transformer=transformer,
                )

                if find_best:
                    range_info = analyzer.find_achievable_range(z, target_idx)
                    analysis["achievable_range"] = {
                        "min_score": round(range_info["min_score"], 4),
                        "max_score": round(range_info["max_score"], 4),
                    }
                    z_target = range_info["z_max"]
                    analysis["best_achievable_score"] = round(range_info["max_score"], 4)

                if target_score is not None:
                    solve_result = analyzer.solve_for_target_score(z, target_idx, target_score)
                    z_target = solve_result["z_modified"]
                    analysis["target_score"] = target_score
                    analysis.update({
                        "target_score": target_score,
                        "achieved_score": solve_result["achieved_score"],
                        "initial_score": solve_result["initial_score"],
                        "converged": solve_result["converged"],
                        "n_optimization_iters": solve_result["n_iters"],
                    })

                    explanation = explainer.explain(
                        z, z_target, target_idx,
                        target_score=target_score,
                        achieved_score=solve_result["achieved_score"],
                    )
                    analysis.update(explanation)

            if output_viz is not None and explainer.pointcloud_decoder is not None:
                try:
                    explainer.visualize_pointcloud(z, z_target, output_viz,
                                                   mesh_path=mesh_path)
                    print(f"Visualization saved to {output_viz}.ply, "
                          f"{output_viz}_mesh.ply, and {output_viz}.png")
                except Exception as e:
                    warnings.warn(f"Visualization failed: {e}")

            result["analysis"] = analysis

        return result


def extract_image_features(image_path, img_size=224, device=None):
    """Standalone helper: encode an image into a latent vector.

    Creates a fresh ImageEncoder (untrained) and returns the raw latent
    as a numpy array of shape [latent_dim].

    This is useful for feature extraction or architecture testing without
    loading a full trained model.

    Args:
        image_path: path to the image file
        img_size:   spatial resolution (default 224)
        device:     torch device (default: auto-detect CUDA)

    Returns:
        numpy array of shape [latent_dim]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    cfg = ConfigGeometryEncoder()
    cfg.img_size = img_size
    img_encoder = ImageEncoder(cfg).to(device)
    return img_encoder(img_tensor).squeeze(0).detach().cpu().numpy()
