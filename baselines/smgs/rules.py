"""Pure-function rule bodies R1..R7 for the SMGS baseline.

Each rule takes a SMGSSample + SMGSConfig and returns (flag, diagnostics).
The flag is 0/1; diagnostics is a small dict of intermediate quantities that
the orchestrator copies into SMGSResult.diagnostics for inspection.

Rules never mutate the sample or the config. All geometry lookups reuse the
legal_region / crosswalks stored on sample.map_info (resolved once per sample
in context.py) so rule bodies stay fast and signal-aware.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from shapely.geometry import Point
from shapely.ops import nearest_points

from .config import SMGSConfig
from .types import SMGSSample

_EPS = 1e-9
_MAX_DIST = 50.0


# -------------------------------------------------------------------------
# Shared geometry helpers
# -------------------------------------------------------------------------

def _nearest_illegal_point(sample: SMGSSample) -> Tuple[float, float, float]:
    """(nx, ny, dist) of the nearest point on the legal region's boundary.

    For a pedestrian inside the legal region, this is the shortest-path
    exit. If the legal region is missing (no zones), returns (x, y, 0.0)
    so downstream rules fall through safely.
    """
    p = sample.target
    legal = sample.map_info.legal_region
    if legal is None:
        return p.x, p.y, 0.0
    # `nearest_points(legal.boundary, point)` returns (on_boundary, point)
    on_bdry, _ = nearest_points(legal.boundary, Point(p.x, p.y))
    return float(on_bdry.x), float(on_bdry.y), float(math.hypot(on_bdry.x - p.x,
                                                                 on_bdry.y - p.y))


def _nearest_legal_crosswalk_entry(sample: SMGSSample) -> Tuple[float, float, str]:
    """Pick the closest (by centroid distance) legal crosswalk entry point.

    Preference order:
      1. a crosswalk whose signal is currently G/y, if signal is available;
      2. the nearest crosswalk of any direction (fallback).
    Returns (cx, cy, direction). If no crosswalks exist at all, returns the
    pedestrian's own position (rule becomes a no-op).
    """
    p = sample.target
    cws = sample.map_info.crosswalks
    if not cws:
        return p.x, p.y, ""

    legal_dirs = sample.signal_info.legal_crosswalks
    candidates = [d for d in legal_dirs if d in cws] if sample.signal_info.available else list(cws.keys())
    if not candidates:
        candidates = list(cws.keys())

    best_dir = min(
        candidates,
        key=lambda d: (sample.map_info.crosswalk_centroids[d][0] - p.x) ** 2
                    + (sample.map_info.crosswalk_centroids[d][1] - p.y) ** 2,
    )
    cx, cy = sample.map_info.crosswalk_centroids[best_dir]
    return float(cx), float(cy), best_dir


def _cos_alignment(ux: float, uy: float, vx: float, vy: float) -> float:
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu < _EPS or nv < _EPS:
        return 0.0
    return float((ux * vx + uy * vy) / (nu * nv))


# -------------------------------------------------------------------------
# R1 — Projected illegal entry (constant-velocity extrapolation)
# -------------------------------------------------------------------------

def rule_R1(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    """Extrapolate position under constant velocity over [0, horizon_s] and
    flag if any projected point leaves the legal region.

    When the legal region is missing (e.g. zones file empty), R1 returns 0
    rather than faulting — this is logged upstream by the sample builder.
    """
    diag: Dict[str, float] = {}
    p = sample.target
    legal = sample.map_info.legal_region
    if legal is None or (p.vx == 0.0 and p.vy == 0.0):
        diag["R1_first_violation_t"] = -1.0
        return 0, diag

    steps = max(1, int(math.ceil(cfg.horizon_s / cfg.proj_dt_s)))
    for s in range(1, steps + 1):
        t = s * cfg.proj_dt_s
        proj = Point(p.x + p.vx * t, p.y + p.vy * t)
        if not legal.contains(proj):
            diag["R1_first_violation_t"] = float(t)
            return 1, diag
    diag["R1_first_violation_t"] = -1.0
    return 0, diag


# -------------------------------------------------------------------------
# R2 — Velocity component toward illegal area
# -------------------------------------------------------------------------

def rule_R2(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    p = sample.target
    diag: Dict[str, float] = {}
    nx, ny, d = _nearest_illegal_point(sample)
    if d < _EPS or (p.vx == 0.0 and p.vy == 0.0):
        diag["v_toward_illegal"] = 0.0
        return 0, diag
    ux = (nx - p.x) / d
    uy = (ny - p.y) / d
    v_toward = p.vx * ux + p.vy * uy     # m/s
    diag["v_toward_illegal"] = float(v_toward)
    return (1 if v_toward > cfg.v_toward_illegal_thresh else 0), diag


# -------------------------------------------------------------------------
# R3 — Heading prefers illegal path over legal crosswalk
# -------------------------------------------------------------------------

def rule_R3(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    p = sample.target
    diag: Dict[str, float] = {"align_illegal": 0.0, "align_legal": 0.0}
    if p.heading is None:
        return 0, diag

    nx, ny, di = _nearest_illegal_point(sample)
    cx, cy, _ = _nearest_legal_crosswalk_entry(sample)
    if di < _EPS:
        return 0, diag

    align_illegal = _cos_alignment(nx - p.x, ny - p.y, p.vx, p.vy)
    align_legal = _cos_alignment(cx - p.x, cy - p.y, p.vx, p.vy)
    diag["align_illegal"] = float(align_illegal)
    diag["align_legal"] = float(align_legal)
    return (1 if align_illegal > align_legal + cfg.heading_margin else 0), diag


# -------------------------------------------------------------------------
# R4 — Signal / waiting pressure
# -------------------------------------------------------------------------

def rule_R4(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    diag: Dict[str, float] = {"waiting_s": float(sample.target.waiting_time_s)}
    if not sample.signal_info.available:
        diag["signal_available"] = 0.0
        return 0, diag
    diag["signal_available"] = 1.0
    pressure = (not sample.signal_info.legal_now) or (
        sample.target.waiting_time_s > cfg.wait_time_thresh
    )
    return (1 if pressure else 0), diag


# -------------------------------------------------------------------------
# R5 — Safe vehicle gap (min TTC above threshold => safe => attempt)
# -------------------------------------------------------------------------

def rule_R5(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    diag: Dict[str, float] = {}
    if not sample.nearby_vehicles:
        diag["min_ttc"] = float("inf")
        return 1, diag
    min_ttc = min(v.ttc for v in sample.nearby_vehicles)
    diag["min_ttc"] = float(min_ttc)
    return (1 if min_ttc > cfg.safe_ttc_thresh else 0), diag


# -------------------------------------------------------------------------
# R6 — Social following (nearby peds already illegally crossing)
# -------------------------------------------------------------------------

def rule_R6(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    social_count = sum(
        1 for q in sample.nearby_peds
        if q.inside_prohibited and q.dist <= cfg.social_radius
    )
    return (1 if social_count >= 1 else 0), {"social_count": float(social_count)}


# -------------------------------------------------------------------------
# R7 — Near-boundary urgency
# -------------------------------------------------------------------------

def rule_R7(sample: SMGSSample, cfg: SMGSConfig) -> Tuple[int, Dict[str, float]]:
    _, _, d = _nearest_illegal_point(sample)
    diag = {"dist_to_illegal": float(d)}
    return (1 if (d > _EPS and d < cfg.near_boundary_dist) else 0), diag


# -------------------------------------------------------------------------
# Dispatcher
# -------------------------------------------------------------------------

RULE_FUNCS = {
    "R1": rule_R1,
    "R2": rule_R2,
    "R3": rule_R3,
    "R4": rule_R4,
    "R5": rule_R5,
    "R6": rule_R6,
    "R7": rule_R7,
}
