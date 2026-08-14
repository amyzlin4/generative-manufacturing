# =============================================================================
# gmdl/datasets.py — PyTorch Dataset & DataLoader for Manufacturing Data
# =============================================================================
#
# Provides two dataset classes and a factory function:
#
#   ManufacturingDataset — main training / validation dataset.  Reads JSON
#       split files and yields geometry feature vectors (MLP path), raw vertex
#       point clouds (PointNet path), and optional images for contrastive
#       learning.
#
#   CADFeatureDataset — lightweight dataset for CAD .step files.  Extracts
#       geometry features on-the-fly via cadhandler.process_step().
#
#   get_dataloader() — convenience wrapper that constructs a DataLoader with
#       the correct batch size, shuffling, and worker count from a Config.
#
# Image transforms (resize, augmentation, normalisation) are applied
# automatically when torchvision is available.
# =============================================================================

import json
import warnings
from pathlib import Path

import numpy as np

from gmdl.constants import GEOM_FEATURE_DIM, MAX_POINTS

# ---------------------------------------------------------------------------
# PyTorch availability guard
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import DataLoader as _TorchDataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TorchDataset = object
    _TorchDataLoader = object
    _TORCH_AVAILABLE = False


class ManufacturingDataset(_TorchDataset):
    """Dataset that reads samples from ``data_root/<split>.json``.

    Each JSON entry is a dict with keys:

    For ``--encoder mlp`` (default):
        ``image``         — relative path to image (optional, for contrastive)
        ``geom_features`` — list of floats (length GEOM_FEATURE_DIM)
        ``process_label`` — integer 0-8, index into MANUFACTURING_PROCESSES

    For ``--encoder pointnet``:
        ``mesh_file``     — relative path to .mat file with ``point3D`` variable
        ``image``         — relative path to image (optional, for contrastive)
        ``process_label`` — integer 0-8, index into MANUFACTURING_PROCESSES

    Image augmentations (random flip, affine, colour jitter) are applied only
    when ``split == "train"`` and torchvision is installed.
    """

    def __init__(self, data_root, split="train", transform=None,
                 image_root=None, use_pointnet=False, max_points=MAX_POINTS):
        self.data_root = Path(data_root)
        self.image_root = Path(image_root) if image_root else self.data_root
        self.split = split
        self.transform = transform
        self.use_pointnet = use_pointnet
        self.max_points = max_points
        self.samples = []

        # Load split file (e.g. data/train.json)
        split_file = self.data_root / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                self.samples = json.load(f)
        else:
            warnings.warn(f"No split file found at {split_file}")

        # Build image transform pipeline (requires torchvision)
        self._img_transform = None
        try:
            import torchvision.transforms as T
            common = [T.Resize((224, 224))]
            if self.split == "train":
                common += [
                    T.RandomHorizontalFlip(),
                    T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            common += [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.224]),
            ]
            self._img_transform = T.Compose(common)
        except ImportError:
            pass

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path):
        """Load and transform an image, returning a [3, 224, 224] tensor.
        Returns a zero tensor on failure or if transforms are unavailable."""
        if self._img_transform is None:
            return torch.zeros(3, 224, 224)
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            return self._img_transform(img)
        except Exception:
            return torch.zeros(3, 224, 224)

    def _load_vertices(self, mesh_path):
        """Load vertices from a .mat file and pad/truncate to self.max_points.

        MATLAB .mat files store the point cloud as the ``point3D`` variable,
        which may be shaped (3, N) or (N, 3).  We normalise to (N, 3).

        Returns:
            vertices:   torch.Tensor of shape [max_points, 3]
            n_valid:    int — number of real (non-padded) vertices
        """
        vertices = np.zeros((self.max_points, 3), dtype=np.float32)
        n_valid = 0
        try:
            import scipy.io as sio
            data = sio.loadmat(str(mesh_path))
            pts = data.get("point3D")
            if pts is None:
                return torch.from_numpy(vertices), n_valid
            pts = np.asarray(pts, dtype=np.float32)
            # MATLAB stores as (3, N) or (N, 3) — normalise to (N, 3)
            if pts.shape[0] == 3 and pts.shape[1] != 3:
                pts = pts.T
            n_valid = min(len(pts), self.max_points)
            vertices[:n_valid] = pts[:n_valid]
        except Exception:
            pass
        return torch.from_numpy(vertices), n_valid

    def __getitem__(self, idx):
        sample = self.samples[idx]
        process_label = sample.get("process_label", 0)

        # --- Image (optional, used for contrastive training) ---
        image_path = self.image_root / sample.get("image", "")
        has_image = (image_path.exists()
                     and image_path.suffix.lower() in (".jpg", ".jpeg", ".png"))
        image_tensor = (self._load_image(image_path) if has_image
                        else torch.zeros(3, 224, 224))

        # --- Geometry modality ---
        if self.use_pointnet:
            mesh_path = self.data_root / sample.get("mesh_file", "")
            vertices, n_valid = self._load_vertices(mesh_path)
            return {
                "vertices": vertices,
                "n_valid_points": n_valid,
                "process_label": torch.tensor(process_label, dtype=torch.long),
                "image_tensor": image_tensor,
                "has_image": has_image,
            }
        else:
            geom_features = torch.tensor(
                sample.get("geom_features", [0.0] * GEOM_FEATURE_DIM),
                dtype=torch.float32,
            )
            return {
                "geom_features": geom_features,
                "process_label": torch.tensor(process_label, dtype=torch.long),
                "image_tensor": image_tensor,
                "has_image": has_image,
            }


class CADFeatureDataset(_TorchDataset):
    """Lightweight dataset for CAD .step files.

    Each item extracts geometry features on-the-fly via
    ``cadhandler.process_step()``.  If the cadhandler module is unavailable or
    processing fails, a zero vector is returned with a warning.
    """

    def __init__(self, step_paths, labels=None, transform=None):
        self.step_paths = step_paths
        self.labels = labels if labels is not None else [0] * len(step_paths)
        self.transform = transform

    def __len__(self):
        return len(self.step_paths)

    def __getitem__(self, idx):
        step_file = self.step_paths[idx]
        label = self.labels[idx]
        try:
            from cadhandler import process_step
            features = json.loads(process_step(step_file))
            geom_vec = [
                features.get("volume mm3", 0),
                features.get("surface area mm2", 0),
                features.get("aspect ratio", 0),
                features.get("compactness", 0),
                features.get("dimensions mm", {}).get("x", 0),
                features.get("dimensions mm", {}).get("y", 0),
                features.get("dimensions mm", {}).get("z", 0),
            ]
        except Exception as e:
            geom_vec = [0.0] * 7
            warnings.warn(f"Failed to process {step_file}: {e}")

        # Pad / truncate to GEOM_FEATURE_DIM
        geom_vec = np.pad(
            geom_vec,
            (0, max(0, GEOM_FEATURE_DIM - len(geom_vec))),
            mode="constant",
        )[:GEOM_FEATURE_DIM]

        return {
            "geom_features": torch.tensor(geom_vec, dtype=torch.float32),
            "process_label": torch.tensor(label, dtype=torch.long),
            "step_path": str(step_file),
        }


def get_dataloader(split, cfg, dataset_cls=ManufacturingDataset):
    """Build a ``DataLoader`` for the given *split* using settings from *cfg*.

    Automatically selects PointNet mode (raw vertices) or MLP mode (hand-crafted
    features) based on ``cfg.encoder_type``.
    """
    use_pointnet = getattr(cfg, "encoder_type", "mlp") == "pointnet"
    max_points = getattr(cfg, "max_points", MAX_POINTS)
    ds = dataset_cls(
        cfg.data_root,
        split=split,
        image_root=getattr(cfg, "image_root", None),
        use_pointnet=use_pointnet,
        max_points=max_points,
    )
    return _TorchDataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        drop_last=(split == "train"),
    )
