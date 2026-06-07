#!/usr/bin/env python3
"""Deterministic geofence checker: turn predicted future trajectories into a
violation probability using the SAME rule as label generation
(utils.jaywalk_geofence.check_jaywalk_status).

A window is predicted-violating if ANY predicted future frame is unsafe (target
in road core, or in a crosswalk under a non-permissive signal). With K sampled
futures the violation PROBABILITY is the fraction of samples that violate — this
gives a graded score for PR-AUC / ROC-AUC and threshold tuning.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from shapely.geometry import Point

from ravp.geofence import load_zones, resolve_dataset_paths
from ravp.jaywalk_geofence import (
    check_jaywalk_status, infer_signal_path, load_signal_intervals_by_dir_turn,
)

FPS = 10.0


class GeofenceChecker:
    """Caches zones (per dataset) and pedestrian signal intervals (per scene).

    Loads signals straight from the per-scene <scene>_signal.csv (via
    infer_signal_path) rather than the multi-GB trajectory CSVs, so this only
    needs dataset/<DS>/derived_data/{signal,map} on disk."""

    def __init__(self, dataset: str):
        root, _gt, zones_csv, traj_dir = resolve_dataset_paths(dataset)
        self.zones = load_zones(zones_csv)
        self.traj_dir = traj_dir
        self._sig: Dict[str, dict] = {}

    def _sig_for(self, scene: str) -> dict:
        if scene not in self._sig:
            sig_by_dir = {}
            sp = infer_signal_path(self.traj_dir / f"{scene}_Traj.csv")
            if sp is not None and sp.is_file():
                full = load_signal_intervals_by_dir_turn(sp)
                sig_by_dir = {d: iv for (d, tr), iv in full.items() if tr == "p"}
            self._sig[scene] = sig_by_dir
        return self._sig[scene]

    def violates(self, abs_xy: np.ndarray, times: List[float], scene: str) -> bool:
        """True if any (position, time) is unsafe. abs_xy: (Tf, 2) absolute metres."""
        sig = self._sig_for(scene)
        wa, cw = self.zones.walking_areas, self.zones.crosswalks
        for (x, y), t in zip(abs_xy, times):
            is_jay, _ = check_jaywalk_status(Point(float(x), float(y)), float(t), wa, cw, sig)
            if is_jay:
                return True
        return False

    def score_samples(self, pred_rel: np.ndarray, origin, times: List[float],
                      scene: str) -> float:
        """pred_rel: (S, Tf, 2) relative to origin. Returns fraction of the S
        sampled futures that violate."""
        ox, oy = origin
        hits = 0
        for s in range(pred_rel.shape[0]):
            abs_xy = pred_rel[s] + np.array([ox, oy], dtype=np.float32)
            if self.violates(abs_xy, times, scene):
                hits += 1
        return hits / max(pred_rel.shape[0], 1)
