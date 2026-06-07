"""Per-timestep + static feature extraction for v4 sequence classifiers.

Emits, for each GT row:
    dyn    : np.ndarray (T, D_dyn)   one feature vector per past timestep
    static : np.ndarray (D_static,)  scalar features at the decision frame tau

D_dyn depends on the feature set:
    - seq_core      : 22
    - seq_core_ctx  : 30  (adds 8 compact interaction scalars per timestep)

D_static is always 14 (see STATIC_FEATURE_NAMES below).

This module deliberately reuses the v2_tabular_engineered helpers so the map /
signal / interaction math stays consistent with the tabular baseline. We only
add a thin per-timestep wrapper — no new geometry code.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Point

# Reuse v2 helpers verbatim — do NOT re-implement geometry / scene I/O.
from models.anticipatory_classifier.versions.v2_tabular_engineered import features as v2feat
from models.anticipatory_classifier.versions.v2_tabular_engineered.features import (
    DEFAULT_FPS,
    DENSITY_RADIUS_M,
    HORIZON_FRAMES,
    MAX_DIST_M,
    MAX_GAP_M,
    MAX_TTC_S,
    NEIGHBOR_RADIUS_M,
    SIGNAL_DIRS,
    SMOOTHING_FRAMES,
    SceneContext,
    ZoneBundle,
    _frame_arrays,
    _last_transition_time,
    _norm_sig_char,
    _point_in_any,
    _safe_literal_list,
    _signed_dist,
    load_scene_context,
    load_zones,
)

T_WINDOW = 20            # fixed by rolling-window generator (t1 = 2 s @ 10 Hz)
_SIG_ENC = {"G": 1.0, "g": 1.0, "y": 0.0, "Y": 0.0, "r": -1.0, "R": -1.0, "?": -0.5}
_DECISION_ZONE_CW_DIST_M = 2.0    # "near a crosswalk boundary" threshold
_IS_MOVING_SPEED = 0.2            # m/s


# ---------------------------------------------------------------------------
# Name / shape manifests
# ---------------------------------------------------------------------------


DYN_FEATURE_NAMES_CORE: List[str] = [
    # Target kinematics (10)
    "dx", "dy", "vx", "vy", "ax", "ay", "speed",
    "heading_sin", "heading_cos",
    "inside_walking_area",       # 10 (last of 10 -> actually starts map group; see below)
]
# NOTE: keep names explicit and grouped for clarity — the actual list is
# assembled below so we can also emit feature_groups.


def _dyn_feature_names(include_context: bool) -> Tuple[List[str], List[str]]:
    """Return (names, groups) for a single timestep. Groups: A/B/C for map/sig,
    D for interaction context."""
    names: List[str] = []
    groups: List[str] = []

    def add(n, g):
        names.append(n)
        groups.append(g)

    # Kinematics (9) — ax/ay/heading already covers the 10th position.
    for n in ("dx", "dy", "vx", "vy", "ax", "ay", "speed",
              "heading_sin", "heading_cos"):
        add(n, "A")

    # Map / legal (6)
    add("inside_walking_area", "B")
    add("inside_crosswalk", "B")
    add("inside_waiting_zone", "B")
    add("signed_dist_walkarea", "B")
    add("dist_to_nearest_crosswalk", "B")
    add("signed_dist_nearest_noped", "B")

    # Signal (7)
    for d in SIGNAL_DIRS:
        add(f"signal_{d}", "C")       # 4
    add("any_pedestrian_green", "C")  # 5
    add("time_since_phase_change_min", "C")  # 6
    add("legal_crossing_allowed", "C")       # 7

    # Context / interaction (8)  — only in seq_core_ctx
    if include_context:
        for n in ("nearest_vehicle_dist", "nearest_vehicle_speed",
                  "nearest_vehicle_ttc", "nearest_ped_dist",
                  "n_neighbors_3m", "n_neighbors_crossing_3m",
                  "local_density", "gap_size"):
            add(n, "D")

    return names, groups


STATIC_FEATURE_NAMES: List[str] = [
    "signed_dist_cw_N", "signed_dist_cw_S", "signed_dist_cw_E", "signed_dist_cw_W",
    "heading_alignment_to_nearest_cw",
    "heading_alignment_to_nearest_noped",
    "ttc_road_boundary_cv",
    "ped_waiting_time_s",
    "time_since_intersection_phase_change_s",
    "decision_zone",
    "any_conflict_vehicle_green",
    "heading_change_cum",
    "speed_at_tau",
    "is_moving",
]


def feature_shapes(include_context: bool) -> Tuple[int, int, int]:
    """(T, D_dyn, D_static) for the selected feature set."""
    names, _ = _dyn_feature_names(include_context)
    return T_WINDOW, len(names), len(STATIC_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Per-timestep interaction features — a free-function port of
# v2feat.TabularFeatureBuilder._interaction_features so we can call it per
# timestep without building a full tabular builder.
# ---------------------------------------------------------------------------


def _interaction_scalars_at(
    ctx: SceneContext, zones: ZoneBundle,
    frame: int, x: float, y: float, vx: float, vy: float,
    target_id: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return (nearest_vehicle_dist, nearest_vehicle_speed, nearest_vehicle_ttc,
    nearest_ped_dist, n_nbr_3m, n_crossing_3m, local_density, gap_size)."""
    peds = _frame_arrays(ctx, frame, vehicle=False)
    vehs = _frame_arrays(ctx, frame, vehicle=True)

    if len(peds):
        mask = peds[:, 2].astype(int) != int(target_id)
        peds_others = peds[mask]
    else:
        peds_others = peds

    if len(peds_others):
        d2 = (peds_others[:, 0] - x) ** 2 + (peds_others[:, 1] - y) ** 2
        d = np.sqrt(d2)
        nearest_ped = float(min(d.min(), MAX_DIST_M))
        nbr_mask = d <= NEIGHBOR_RADIUS_M
        n_nbr = int(nbr_mask.sum())
        n_crossing = 0
        for px, py in peds_others[nbr_mask, 0:2]:
            p = Point(float(px), float(py))
            if _point_in_any(p, list(zones.crosswalks.values())) or \
                    _point_in_any(p, zones.no_ped_areas):
                n_crossing += 1
        n_dens = int((d <= DENSITY_RADIUS_M).sum())
        local_density = float(n_dens / (math.pi * DENSITY_RADIUS_M ** 2))
    else:
        nearest_ped = float(MAX_DIST_M)
        n_nbr = 0
        n_crossing = 0
        local_density = 0.0

    if len(vehs):
        d2 = (vehs[:, 0] - x) ** 2 + (vehs[:, 1] - y) ** 2
        d = np.sqrt(d2)
        idx_near = int(np.argmin(d))
        nearest_veh = float(min(d[idx_near], MAX_DIST_M))
        vx_v = float(vehs[idx_near, 3])
        vy_v = float(vehs[idx_near, 4])
        veh_speed = float(math.hypot(vx_v, vy_v) * ctx.fps)
        tx = x + vx * ctx.fps
        ty = y + vy * ctx.fps
        vx_vs = vx_v * ctx.fps
        vy_vs = vy_v * ctx.fps
        rel_x = float(vehs[idx_near, 0]) - tx
        rel_y = float(vehs[idx_near, 1]) - ty
        rel_speed = math.hypot(vx_vs, vy_vs)
        if rel_speed > 1e-6:
            dot = (-rel_x) * vx_vs + (-rel_y) * vy_vs
            if dot > 0:
                ttc_veh = float(min(math.hypot(rel_x, rel_y) / rel_speed, MAX_TTC_S))
            else:
                ttc_veh = float(MAX_TTC_S)
        else:
            ttc_veh = float(MAX_TTC_S)
        vxn, vyn = vx, vy
        n = math.hypot(vxn, vyn)
        if n < 1e-6:
            vxn, vyn = 1.0, 0.0
        else:
            vxn /= n
            vyn /= n
        projs = (vehs[:, 0] - x) * vxn + (vehs[:, 1] - y) * vyn
        front = projs[projs > 0]
        back = projs[projs < 0]
        front_min = float(front.min()) if len(front) else float(MAX_GAP_M) / 2
        back_min = float(-back.max()) if len(back) else float(MAX_GAP_M) / 2
        gap = float(min(front_min + back_min, MAX_GAP_M))
    else:
        nearest_veh = float(MAX_DIST_M)
        veh_speed = 0.0
        ttc_veh = float(MAX_TTC_S)
        gap = float(MAX_GAP_M)

    return (nearest_veh, veh_speed, ttc_veh, nearest_ped,
            float(n_nbr), float(n_crossing), local_density, gap)


# ---------------------------------------------------------------------------
# Signal state at an arbitrary time t
# ---------------------------------------------------------------------------


def _signal_state_at(intervals, t: float) -> str:
    if not intervals:
        return "?"
    begins = [x[0] for x in intervals]
    i = bisect_right(begins, t) - 1
    if i < 0:
        return "?"
    b, e, st = intervals[i]
    if b <= t < e:
        return _norm_sig_char(st)
    return "?"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


@dataclass
class _Counts:
    ok: int = 0
    failed: int = 0


class SeqFeatureBuilder:
    """Per-scene stateful builder for v4 sequence features.

    Usage:
        zones = load_zones(zones_csv)
        builder = SeqFeatureBuilder(zones, fps=10.0, include_context=True)
        for scene_name, gt_group in gt_df.groupby('scene_name'):
            ctx = load_scene_context(scene_name, traj_dir, fps=10.0)
            builder.set_scene(ctx)
            for _, row in gt_group.iterrows():
                dyn, static = builder.compute_row(row)
                # dyn.shape == (20, D_dyn); static.shape == (D_static,)
    """

    def __init__(self, zones: ZoneBundle, fps: float = DEFAULT_FPS,
                 include_context: bool = False):
        self.zones = zones
        self.fps = fps
        self.include_context = include_context
        self._ctx: Optional[SceneContext] = None
        self.dyn_names, self.dyn_groups = _dyn_feature_names(include_context)
        self.static_names = list(STATIC_FEATURE_NAMES)
        self.counts = _Counts()

    # ------------------------------------------------------------------
    def set_scene(self, ctx: SceneContext) -> None:
        self._ctx = ctx

    # ------------------------------------------------------------------
    def compute_row(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        ctx = self._ctx
        assert ctx is not None, "call set_scene() before compute_row()"

        cx = _safe_literal_list(row["cx_m"])
        cy = _safe_literal_list(row["cy_m"])
        spd = _safe_literal_list(row["speed_smooth"])
        T = T_WINDOW
        D_dyn = len(self.dyn_names)
        D_static = len(self.static_names)

        if not cx or not cy or len(cx) < T or len(cy) < T:
            # Shouldn't happen — 1_generate_rolling_windows drops short
            # histories — but fall back to NaN so downstream can skip.
            self.counts.failed += 1
            return (
                np.full((T, D_dyn), np.nan, dtype=np.float32),
                np.full((D_static,), np.nan, dtype=np.float32),
            )

        cx = np.asarray(cx, dtype=np.float64)[-T:]
        cy = np.asarray(cy, dtype=np.float64)[-T:]
        spd = (np.asarray(spd, dtype=np.float64)[-T:]
               if len(spd) >= T else np.zeros(T, dtype=np.float64))

        frame_current = int(row["frame_current"])
        ped_id = int(row["ped_id"])
        # Timestep i in [0..T-1] corresponds to frame = frame_current - (T-1-i)
        # and t = frame / fps.
        frames = frame_current - (T - 1) + np.arange(T, dtype=np.int64)
        times = frames.astype(np.float64) / self.fps

        x_tau = float(cx[-1])
        y_tau = float(cy[-1])

        dyn = np.empty((T, D_dyn), dtype=np.float32)

        # Per-timestep kinematics: finite-diff velocity over a causal
        # smoothing window of up to SMOOTHING_FRAMES.
        for i in range(T):
            x_i = float(cx[i])
            y_i = float(cy[i])
            sf = min(SMOOTHING_FRAMES, i)
            if sf > 0:
                vx_i = (cx[i] - cx[i - sf]) / sf
                vy_i = (cy[i] - cy[i - sf]) / sf
            else:
                vx_i = vy_i = 0.0
            # Acceleration (diff of vel at [i, i-sf] vs prior window).
            if i >= 2 * SMOOTHING_FRAMES:
                vx_prev = (cx[i - SMOOTHING_FRAMES] - cx[i - 2 * SMOOTHING_FRAMES]) / SMOOTHING_FRAMES
                vy_prev = (cy[i - SMOOTHING_FRAMES] - cy[i - 2 * SMOOTHING_FRAMES]) / SMOOTHING_FRAMES
                ax_i = (vx_i - vx_prev) / float(SMOOTHING_FRAMES)
                ay_i = (vy_i - vy_prev) / float(SMOOTHING_FRAMES)
            else:
                ax_i = 0.0
                ay_i = 0.0

            speed_i = float(spd[i]) if spd[i] >= 0 else 0.0
            if vx_i == 0.0 and vy_i == 0.0:
                heading_sin = 0.0
                heading_cos = 1.0
            else:
                h = math.atan2(vy_i, vx_i)
                heading_sin = math.sin(h)
                heading_cos = math.cos(h)

            # Kinematics (target-centered positions: dx, dy)
            k_feats = [
                x_i - x_tau, y_i - y_tau, vx_i, vy_i, ax_i, ay_i, speed_i,
                heading_sin, heading_cos,
            ]

            # Map (6)
            pt = Point(x_i, y_i)
            inside_walk = _point_in_any(pt, self.zones.walking_areas)
            inside_cw = _point_in_any(pt, list(self.zones.crosswalks.values()))
            inside_wait = _point_in_any(pt, self.zones.waiting_zones)
            wa_dists = [_signed_dist(wa, pt) for wa in self.zones.walking_areas]
            min_wa = float(min(wa_dists)) if wa_dists else float(MAX_DIST_M)
            cw_dists = []
            for d in SIGNAL_DIRS:
                if d in self.zones.crosswalks:
                    cw_dists.append(_signed_dist(self.zones.crosswalks[d], pt))
                else:
                    cw_dists.append(float(MAX_DIST_M))
            dist_to_nearest_cw = float(min(abs(x) for x in cw_dists))
            if self.zones.no_ped_areas:
                dist_to_nearest_noped = float(min(
                    _signed_dist(p, pt) for p in self.zones.no_ped_areas))
            else:
                dist_to_nearest_noped = float(MAX_DIST_M)

            m_feats = [
                float(inside_walk), float(inside_cw), float(inside_wait),
                min_wa, dist_to_nearest_cw, dist_to_nearest_noped,
            ]

            # Signal (7)
            t_i = float(times[i])
            sig_vals = [
                _SIG_ENC.get(_signal_state_at(ctx.sig_by_dir_ped.get(d, []), t_i),
                             -0.5)
                for d in SIGNAL_DIRS
            ]
            any_ped_green = float(1.0 if any(v > 0.5 for v in sig_vals) else 0.0)
            tspc_list = [
                _last_transition_time(ctx.sig_by_dir_ped.get(d, []), t_i)
                for d in SIGNAL_DIRS
                if ctx.sig_by_dir_ped.get(d)
            ]
            tspc_min = float(min(tspc_list)) if tspc_list else 60.0
            legal_cross = any_ped_green

            s_feats = [*sig_vals, any_ped_green, tspc_min, legal_cross]

            row_feats = k_feats + m_feats + s_feats

            # Context (optional, 8)
            if self.include_context:
                ctx_feats = list(_interaction_scalars_at(
                    ctx, self.zones, int(frames[i]), x_i, y_i, vx_i, vy_i,
                    target_id=ped_id,
                ))
                row_feats += ctx_feats

            dyn[i, :] = row_feats

        # ---- static features at tau -------------------------------------
        pt_tau = Point(x_tau, y_tau)

        # Velocity at tau using same smoothing as per-timestep loop.
        sf = min(SMOOTHING_FRAMES, T - 1)
        vx_tau = float((cx[-1] - cx[-(sf + 1)]) / sf) if sf > 0 else 0.0
        vy_tau = float((cy[-1] - cy[-(sf + 1)]) / sf) if sf > 0 else 0.0
        speed_tau = float(spd[-1]) if spd[-1] >= 0 else 0.0

        # signed_dist_cw_{N,S,E,W}
        cw_dists_signed = []
        for d in SIGNAL_DIRS:
            if d in self.zones.crosswalks:
                cw_dists_signed.append(_signed_dist(self.zones.crosswalks[d], pt_tau))
            else:
                cw_dists_signed.append(float(MAX_DIST_M))

        # Heading alignment helpers
        def _align(cx_c, cy_c):
            dx = cx_c - x_tau
            dy = cy_c - y_tau
            n = math.hypot(dx, dy)
            vn = math.hypot(vx_tau, vy_tau)
            if n < 1e-6 or vn < 1e-6:
                return 0.0
            return float((vx_tau * dx + vy_tau * dy) / (n * vn))

        # Nearest crosswalk direction by absolute signed dist
        nearest_cw_dir = min(
            SIGNAL_DIRS, key=lambda d: abs(cw_dists_signed[SIGNAL_DIRS.index(d)])
        )
        cw_c = self.zones.crosswalk_centroids.get(nearest_cw_dir, (x_tau, y_tau))
        align_cw = _align(cw_c[0], cw_c[1])

        if self.zones.no_ped_centroids:
            noped_c = min(self.zones.no_ped_centroids,
                          key=lambda c: (c[0] - x_tau) ** 2 + (c[1] - y_tau) ** 2)
            align_noped = _align(noped_c[0], noped_c[1])
        else:
            align_noped = 0.0

        # TTC road boundary (CV projection exits all walking areas)
        ttc_road = float(HORIZON_FRAMES)
        inside_walk_tau = _point_in_any(pt_tau, self.zones.walking_areas)
        if inside_walk_tau and (vx_tau != 0 or vy_tau != 0):
            for step in range(1, HORIZON_FRAMES + 1):
                p_f = Point(x_tau + step * vx_tau, y_tau + step * vy_tau)
                still_in = any(wa.contains(p_f) for wa in self.zones.walking_areas)
                if not still_in:
                    ttc_road = float(step)
                    break

        # ped_waiting_time_s — scan backward using ped_speed_by_id.
        spd_map = ctx.ped_speed_by_id.get(ped_id, {})
        waits = 0
        cap = int(30.0 * self.fps)
        f = int(frame_current)
        while f >= 0 and waits < cap:
            s = spd_map.get(f)
            if s is None or s >= 0.02:   # same thresh as v2 (m/frame)
                break
            waits += 1
            f -= 1
        ped_wait_s = float(waits / self.fps if self.fps > 0 else waits)

        # time_since_intersection_phase_change_s — reuse v2 helper
        cur_tup, _prev_tup, t_since_intx = v2feat._current_and_previous_phase(
            ctx.phase_timeline, float(frame_current) / self.fps)

        # Decision zone: inside walking area, < _DECISION_ZONE_CW_DIST_M from
        # nearest crosswalk boundary, and NOT currently on a crosswalk.
        dist_cw_tau = float(min(abs(d) for d in cw_dists_signed))
        inside_cw_tau = _point_in_any(pt_tau, list(self.zones.crosswalks.values()))
        decision_zone = 1.0 if (inside_walk_tau and not inside_cw_tau and
                                dist_cw_tau < _DECISION_ZONE_CW_DIST_M) else 0.0

        # any_conflict_vehicle_green at tau
        any_veh_green = 0.0
        t_tau = float(frame_current) / self.fps
        if ctx.sig_by_dir_vehicle:
            for d, intervals in ctx.sig_by_dir_vehicle.items():
                if _signal_state_at(intervals, t_tau) == "G":
                    any_veh_green = 1.0
                    break

        # heading_change_cum: heading at tau vs heading at tau-SMOOTHING_FRAMES
        if T - 1 >= 2 * SMOOTHING_FRAMES:
            vx_prev = (cx[-(SMOOTHING_FRAMES + 1)] - cx[-(2 * SMOOTHING_FRAMES + 1)]) / SMOOTHING_FRAMES
            vy_prev = (cy[-(SMOOTHING_FRAMES + 1)] - cy[-(2 * SMOOTHING_FRAMES + 1)]) / SMOOTHING_FRAMES
            h_tau = math.atan2(vy_tau, vx_tau) if (vx_tau != 0 or vy_tau != 0) else 0.0
            h_prev = math.atan2(vy_prev, vx_prev) if (vx_prev != 0 or vy_prev != 0) else 0.0
            heading_change = float((h_tau - h_prev + math.pi) % (2 * math.pi) - math.pi)
        else:
            heading_change = 0.0

        is_moving = 1.0 if (math.hypot(vx_tau, vy_tau) * self.fps) > _IS_MOVING_SPEED else 0.0

        static = np.array([
            *cw_dists_signed,                        # 4
            align_cw,                                # 5
            align_noped,                             # 6
            ttc_road,                                # 7
            ped_wait_s,                              # 8
            float(t_since_intx),                     # 9
            decision_zone,                           # 10
            any_veh_green,                           # 11
            heading_change,                          # 12
            speed_tau,                               # 13
            is_moving,                               # 14
        ], dtype=np.float32)

        self.counts.ok += 1
        return dyn.astype(np.float32), static
