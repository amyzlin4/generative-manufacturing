# =============================================================================
# gmdl/__init__.py — Package Initialisation and Public API Re-exports
# =============================================================================
#
# The ``gmdl`` package provides a modular reimplementation of the manufacturing
# process prediction system originally contained in a single gmdl.py file.
#
# Module layout:
#   gmdl.constants          — process labels, dimension constants
#   gmdl.utils              — directory helpers, infinite iterator, options loader
#   gmdl.config             — ConfigGeometryEncoder, ConfigProcessPredictor, ConfigExperiment
#   gmdl.datasets           — ManufacturingDataset, CADFeatureDataset, get_dataloader
#   gmdl.encoders           — GeometryEncoder, ImageEncoder, PointNetEncoder
#   gmdl.losses             — InfoNCELoss
#   gmdl.predictor          — ProcessPredictor, ProcessPredictorHybrid
#   gmdl.training           — Clock, Trainer
#   gmdl.inference          — ManufacturingProcessPredictor, extract_image_features
#   gmdl.cli                — main(), predict_cli(), analyze_cli(), train_decoders_cli()
#   gmdl.decoders           — LinearDecoder, PointCloudDecoder (analysis decoders)
#   gmdl.decoder_training   — DecoderDataset, train_decoders (training loop + metrics)
#   gmdl.analysis           — ProcessFitScorer, LatentDirectionAnalyzer, DesignExplainer
#
# This __init__.py re-exports every public symbol so that existing code like
# ``from gmdl import GeometryEncoder, GEOM_FEATURE_DIM`` continues to work
# unchanged.
#
# When PyTorch is not installed, the torch-dependent symbols are replaced by
# stub classes/functions that raise ImportError on use, preserving the
# importability of the constants and config modules.
# =============================================================================

# ---------------------------------------------------------------------------
# Always-available symbols (no PyTorch required)
# ---------------------------------------------------------------------------
from gmdl.constants import (
    MANUFACTURING_PROCESSES,
    N_PROCESSES,
    GEOM_FEATURE_DIM,
    LATENT_DIM,
    MAX_SEQ_LEN,
    MAX_POINTS,
)

from gmdl.utils import (
    ensure_dir,
    ensure_dirs,
    cycle,
    load_options,
    get_active_processes,
)

from gmdl.config import (
    ConfigGeometryEncoder,
    ConfigProcessPredictor,
    ConfigExperiment,
)

# Datasets use a torch-availability guard internally and are always importable
from gmdl.datasets import (
    ManufacturingDataset,
    CADFeatureDataset,
    get_dataloader,
)

# ---------------------------------------------------------------------------
# Torch-dependent symbols (graceful fallback if torch is not installed)
# ---------------------------------------------------------------------------
try:
    from gmdl.encoders import GeometryEncoder, ImageEncoder, PointNetEncoder
    from gmdl.losses import InfoNCELoss
    from gmdl.predictor import ProcessPredictor, ProcessPredictorHybrid
    from gmdl.training import Clock, Trainer
    from gmdl.inference import ManufacturingProcessPredictor, extract_image_features
    from gmdl.cli import main, predict_cli, analyze_cli, train_decoders_cli
    from gmdl.decoders import LinearDecoder, PointCloudDecoder
    from gmdl.decoder_training import FeatureStandardizer
    from gmdl.analysis import (
        ProcessFitScorer, LatentDirectionAnalyzer, DesignExplainer,
        FitResult, FEATURE_NAMES,
    )
except ImportError:
    # PyTorch not available — provide stubs that raise on use
    def main():
        print("PyTorch is required. Install with: pip install torch")

    def predict_cli():
        print("PyTorch is required. Install with: pip install torch")

    def analyze_cli():
        print("PyTorch is required. Install with: pip install torch")

    def train_decoders_cli():
        print("PyTorch is required. Install with: pip install torch")

    def extract_image_features(*args, **kwargs):
        raise ImportError("PyTorch is required. Install with: pip install torch")

    class ManufacturingProcessPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")
        def load_weights(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")
        def predict(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")
