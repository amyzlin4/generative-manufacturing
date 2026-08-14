# =============================================================================
# gmdl/utils.py — Miscellaneous Utility Functions
# =============================================================================
#
# Small helpers used across the gmdl/ package for directory creation, infinite
# iteration, and loading process-configuration options from disk.  None of
# these depend on PyTorch.
# =============================================================================

import os
import json


def ensure_dir(path):
    """Create directory *path* (and parents) if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """Create one or more directories.  *paths* may be a single string or a
    list of strings."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def cycle(iterable):
    """Yield items from *iterable* indefinitely, restarting when exhausted.
    Useful for creating an infinite validation iterator that the training loop
    can sample from with ``next()``."""
    while True:
        for x in iterable:
            yield x


def load_options(path="options.json"):
    """Load and return the JSON contents of *path* (default ``options.json``)."""
    with open(path, "r") as f:
        return json.load(f)


def get_active_processes(options=None):
    """Return a list of process names whose ``status`` field is ``True`` in the
    given *options* dict (or loaded from ``options.json`` if *None*)."""
    if options is None:
        options = load_options()
    return [p for p, v in options.items() if v.get("status", False)]
