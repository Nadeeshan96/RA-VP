"""Geofence geometry + the deterministic safe/unsafe rule.

Thin, clean public API over the ported helpers:
  - ``resolve_dataset_paths`` maps a dataset name to its files under ``DATA_DIR``.
  - ``load_zones`` / ``load_scene_context`` / ``ZoneBundle`` come from the tabular
    feature module (shared with the tabular baseline).
  - the safe/unsafe rule and signal lookups come from ``jaywalk_geofence``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ravp.paths import DATA_DIR
from ravp.tabular_features import ZoneBundle, load_scene_context, load_zones  # noqa: F401
from ravp.jaywalk_geofence import (  # noqa: F401
    check_jaywalk_status,
    infer_signal_path,
    load_signal_intervals_by_dir_turn,
    lookup_state,
    normalize_state,
)


def resolve_dataset_paths(dataset: str) -> Tuple[Path, Path, Path, Path]:
    """Return (dataset_root, gt_path, zones_csv, traj_dir) under DATA_DIR.

    The traj_dir need not contain trajectory files; it is only used to derive the
    per-scene signal CSV path (see ``infer_signal_path``)."""
    root = DATA_DIR / dataset
    gt = root / "processed_data" / "jaywalk_ground_truth.csv"
    zones = root / "derived_data" / "map" / f"{dataset}_int_all_zones.csv"
    traj = root / "derived_data" / "traj"
    return root, gt, zones, traj
