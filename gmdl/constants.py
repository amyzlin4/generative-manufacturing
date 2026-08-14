# =============================================================================
# gmdl/constants.py — Global Constants for Manufacturing Process Prediction
# =============================================================================
#
# Central definition of all magic numbers, dimension sizes, and the canonical
# list of manufacturing process labels.  Every other module in the gmdl/ package
# imports from here, so changing a dimension or adding a process only requires
# editing this one file.
#
# Process label indices are used throughout the codebase:
#   - JSON data files encode process_label as an integer 0-8.
#   - The model's output head produces N_PROCESSES logits.
#   - MANUFACTURING_PROCESSES[i] maps index i back to a human-readable name.
# =============================================================================

# ---------------------------------------------------------------------------
# Manufacturing process labels
# ---------------------------------------------------------------------------

MANUFACTURING_PROCESSES = [
    "3-axis CNC machining",
    "5-axis CNC machining",
    "Injection molding",
    "Casting",
    "Forging",
    "Lathing/Turning",
    "Sheet metal fabrication",
    "3D printing",
    "Sintering",
]

# Number of distinct manufacturing process classes
N_PROCESSES = len(MANUFACTURING_PROCESSES)

# ---------------------------------------------------------------------------
# Feature / tensor dimension constants
# ---------------------------------------------------------------------------

# Dimensionality of the hand-crafted geometry feature vectors produced by
# hks2gmdl.py or cadhandler.py.  The MLP encoder accepts this as input.
GEOM_FEATURE_DIM = 64

# Dimensionality of the shared latent space that both the geometry and image
# encoders map into.  All downstream heads (predictor, projection layers) use
# this size.
LATENT_DIM = 128

# Maximum sequence length for the transformer-based ProcessPredictor.  The
# predictor expands the single latent vector to this many positions via
# learned positional embeddings before feeding into the transformer encoder.
MAX_SEQ_LEN = 3

# Maximum number of vertices per mesh for PointNet.  Meshes with more points
# are truncated; meshes with fewer are zero-padded.  This determines the
# second dimension of the [B, MAX_POINTS, 3] input tensor.
MAX_POINTS = 512
