"""Build SMGSSample objects from a GT row + SceneContext + ZoneBundle.

This module does the heavy lifting of turning the 20-frame observation window
stored in jaywalk_ground_truth.csv into the clean data structures that rules.py
consumes. It reuses v2_tabular_engineered/features.py helpers for
zone loading, per-frame agent arrays, and waiting-time scanning so the
rule baseline stays consistent with the rest of the pipeline.
"""
from __future__ import annotations

import ast
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from models.anticipatory_classifier.versions.v2_tabular_engineered.features import (  # type: ignore
    SceneContext,
    ZoneBundle,
    _frame_arrays,
    _norm_sig_char,
)

from .config import SMGSConfig
from .types import (
    MapInfo,
    NearbyPed,
    NearbyVehicle,
    PedState,
    SMGSSample,
    SignalInfo,
)

SIGNAL_DIRS = ("N", "S", "E", "W")
SMOOTHING_FRAMES = 5
WAIT_SPEED_THRESH_M_PER_FRAME = 0.02  # same as v2 features; ~0.2 m/s at 10 FPS
_EPS = 1e-9


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _parse_list(cell) -> list:
    if isinstance(cell, (list, tuple)):
        return list(cell)
    if isinstance(cell, str):
        try:
            return list(json.loads(cell))
        except Exception:
            try:
                return list(ast.literal_eval(cell))
            except Exception:
                return []
    return []


def _finite_velocity(cx: List[float], cy: List[float], fps: float) -> Tuple[float, float]:
    """Estimate velocity in m/s from the last SMOOTHING_FRAMES+1 frames."""
    if len(cx) < 2:
        return 0.0, 0.0
    sf = min(SMOOTHING_FRAMES, len(cx) - 1)
    try:
        vx_fr = (cx[-1] - cx[-(sf + 1)]) / sf       # m/frame
        vy_fr = (cy[-1] - cy[-(sf + 1)]) / sf
    except (TypeError, ValueError):
        return 0.0, 0.0
    return float(vx_fr * fps), float(vy_fr * fps)   # -> m/s


def _waiting_time_s(ctx: SceneContext, ped_id: int, frame_current: int,
                    cap_s: float = 60.0) -> float:
    """Consecutive seconds the pedestrian has been (near-)stationary before
    frame_current. Uses the scene's per-pedestrian speed map (m/frame)."""
    spd = ctx.ped_speed_by_id.get(int(ped_id), {})
    if not spd or ctx.fps <= 0:
        return 0.0
    cap_frames = int(cap_s * ctx.fps)
    waits = 0
    f = int(frame_current)
    while f >= 0 and waits < cap_frames:
        s = spd.get(f)
        if s is None or s >= WAIT_SPEED_THRESH_M_PER_FRAME:
            break
        waits += 1
        f -= 1
    return float(waits / ctx.fps)


def _build_signal_info(row: pd.Series) -> SignalInfo:
    """Construct SignalInfo from the ground-truth row's signal columns.

    A missing / empty signal value is normalised to '?'. Signal is considered
    'available' for this sample if at least one direction has a known state.
    """
    ped_sig: Dict[str, str] = {}
    for d in SIGNAL_DIRS:
        raw = row.get(f"signal_{d}", "")
        ped_sig[d] = _norm_sig_char(raw)
    known = [v for v in ped_sig.values() if v != "?"]
    legal_dirs = tuple(d for d, v in ped_sig.items() if v in ("G", "y"))
    return SignalInfo(
        available=bool(known),
        ped_signal=ped_sig,
        legal_crosswalks=legal_dirs,
        legal_now=bool(legal_dirs),
    )


def _build_map_info(zones: ZoneBundle, legal_dirs: Tuple[str, ...],
                    signal_available: bool) -> MapInfo:
    """Resolve the per-sample legal region from static walking areas and
    the crosswalks whose pedestrian signal is currently G or y.

    When signal info is not available we conservatively treat **every**
    crosswalk polygon as legal — matching the constant-velocity baseline's
    semantics (`check_projected_jaywalk` defaults to legal when signal is
    unknown). This prevents rules R1/R7 from over-firing on scenes without
    signal CSVs.
    """
    legal_polys = list(zones.walking_areas)
    if not signal_available:
        legal_polys.extend(zones.crosswalks.values())
    else:
        for d in legal_dirs:
            if d in zones.crosswalks:
                legal_polys.append(zones.crosswalks[d])

    if legal_polys:
        legal_region = unary_union(legal_polys)
        legal_boundary = legal_region.boundary
    else:
        legal_region = None
        legal_boundary = None

    return MapInfo(
        walking_areas=tuple(zones.walking_areas),
        crosswalks=dict(zones.crosswalks),
        crosswalk_centroids=dict(zones.crosswalk_centroids),
        legal_region=legal_region,
        legal_boundary=legal_boundary,
    )


def _ttc_to_point(veh_xy: np.ndarray, veh_v: np.ndarray,
                  ped_xy: np.ndarray, max_ttc: float = 10.0) -> float:
    """Time-to-minimum-separation between a moving vehicle and a stationary
    point (the pedestrian's current position). Returns inf-proxy (max_ttc)
    if the vehicle is not approaching, or if relative speed ~= 0."""
    speed = float(np.hypot(veh_v[0], veh_v[1]))
    if speed < _EPS:
        return max_ttc
    rel = ped_xy - veh_xy            # vector from vehicle to pedestrian
    closing = float(rel @ veh_v) / speed   # projection of rel onto velocity
    if closing <= 0:
        return max_ttc
    return float(min(closing / speed, max_ttc))


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def build_sample(ctx: SceneContext, zones: ZoneBundle, row: pd.Series,
                 cfg: SMGSConfig) -> Optional[SMGSSample]:
    """Assemble a SMGSSample from a single ground-truth row and its scene.

    Returns None if the row's observation history cannot be parsed (e.g.
    empty cx/cy lists) — the caller should skip such rows.
    """
    cx = _parse_list(row["cx_m"])
    cy = _parse_list(row["cy_m"])
    if not cx or not cy:
        return None

    fps = float(ctx.fps) if ctx.fps else 10.0
    x = float(cx[-1])
    y = float(cy[-1])
    vx, vy = _finite_velocity(cx, cy, fps)
    speed = float(math.hypot(vx, vy))
    heading = math.atan2(vy, vx) if speed > _EPS else None

    frame = int(row["frame_current"])
    time_s = frame / fps if fps > 0 else 0.0
    waiting_s = _waiting_time_s(ctx, int(row["ped_id"]), frame)

    target = PedState(
        scene=str(row.get("scene_name", ctx.scene_name)),
        ped_id=int(row["ped_id"]),
        frame=frame,
        time_s=time_s,
        x=x, y=y, vx=vx, vy=vy, speed=speed, heading=heading,
        waiting_time_s=waiting_s,
    )

    signal_info = _build_signal_info(row)
    map_info = _build_map_info(zones, signal_info.legal_crosswalks,
                               signal_info.available)

    nearby_peds = _build_nearby_peds(ctx, frame, x, y, target.ped_id,
                                     cfg.ped_context_radius, map_info)
    nearby_vehicles = _build_nearby_vehicles(ctx, frame, x, y, fps,
                                             cfg.veh_context_radius)

    label = int(row["label"]) if "label" in row.index else None

    return SMGSSample(
        target=target,
        map_info=map_info,
        signal_info=signal_info,
        nearby_peds=tuple(nearby_peds),
        nearby_vehicles=tuple(nearby_vehicles),
        label=label,
    )


def _build_nearby_peds(ctx: SceneContext, frame: int, x: float, y: float,
                       self_id: int, radius: float,
                       map_info: MapInfo) -> List[NearbyPed]:
    peds = _frame_arrays(ctx, frame, vehicle=False)
    if len(peds) == 0:
        return []
    dx = peds[:, 0] - x
    dy = peds[:, 1] - y
    d = np.hypot(dx, dy)
    mask = (peds[:, 2].astype(int) != self_id) & (d <= radius)
    if not np.any(mask):
        return []
    fps = float(ctx.fps) if ctx.fps else 10.0
    out: List[NearbyPed] = []
    legal = map_info.legal_region
    for i in np.where(mask)[0]:
        px = float(peds[i, 0])
        py = float(peds[i, 1])
        inside_legal = bool(legal.contains(Point(px, py))) if legal is not None else True
        out.append(NearbyPed(
            dx=float(dx[i]),
            dy=float(dy[i]),
            dist=float(d[i]),
            vx=float(peds[i, 3]) * fps,   # m/frame -> m/s
            vy=float(peds[i, 4]) * fps,
            inside_prohibited=not inside_legal,
        ))
    return out


def _build_nearby_vehicles(ctx: SceneContext, frame: int, x: float, y: float,
                           fps: float, radius: float) -> List[NearbyVehicle]:
    vehs = _frame_arrays(ctx, frame, vehicle=True)
    if len(vehs) == 0:
        return []
    dx = vehs[:, 0] - x
    dy = vehs[:, 1] - y
    d = np.hypot(dx, dy)
    mask = d <= radius
    if not np.any(mask):
        return []
    ped_xy = np.array([x, y], dtype=np.float64)
    out: List[NearbyVehicle] = []
    for i in np.where(mask)[0]:
        veh_xy = vehs[i, 0:2].astype(np.float64)
        veh_v_ms = np.array([vehs[i, 3] * fps, vehs[i, 4] * fps], dtype=np.float64)
        speed = float(np.hypot(veh_v_ms[0], veh_v_ms[1]))
        ttc = _ttc_to_point(veh_xy, veh_v_ms, ped_xy, max_ttc=10.0)
        out.append(NearbyVehicle(
            dx=float(dx[i]),
            dy=float(dy[i]),
            dist=float(d[i]),
            vx=float(veh_v_ms[0]),
            vy=float(veh_v_ms[1]),
            speed=speed,
            ttc=ttc,
        ))
    return out
