"""Central path configuration for the RA-VP reproducibility package.

All data, weights, and feature caches live under ``ra-vp/assets/`` (populated by
``download_assets.py``). Override the repo root with the ``RAVP_ROOT`` env var.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = the ra-vp/ directory (parent of this package), or RAVP_ROOT.
ROOT = Path(os.environ.get("RAVP_ROOT", Path(__file__).resolve().parents[1]))

ASSETS_DIR = ROOT / "assets"
DATA_DIR = ASSETS_DIR / "data"          # dataset/<DS>/{processed_data,derived_data}
WEIGHTS_DIR = ASSETS_DIR / "weights"    # forecaster, head, window caches, baseline caches
FEATURES_DIR = ASSETS_DIR / "features"  # RA-VP exact-checker risk features
SDF_DIR = WEIGHTS_DIR / "sdf"           # cached signed-distance fields

RESULTS_DIR = ROOT / "results"          # cells.json + per-cell run JSONs (in-repo)

# Window caches (per dataset) shipped under weights/cache_<DS>/{train,val,test}.pt
CACHE_ROOT = WEIGHTS_DIR
