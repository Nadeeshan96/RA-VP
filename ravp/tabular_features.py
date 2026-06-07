"""Extended engineered feature builder for v2_tabular_engineered.

Produces ~60 features per GT sample across 5 ablation groups (A–E):

  A — target kinematics (10)
  B — map/legal geometry (13)
  C — instantaneous signal (9)
  D — interaction (8)   [requires full scene trajectory]
  E — pedestrian waiting time + intersection phase history (2K+4; K=8 => 20)

The builder is *stateful* per scene: call `prepare_scene()` once per scene
to load the trajectory CSV + signal CSV + zone polygons + per-frame KD-trees
used by the interaction and waiting-time features. Then call `compute_row(row)`
for every GT row in that scene.

The top-K phase vocabulary is *not* learned here. Use
`learn_phase_vocabulary(gt_train_df)` first on the train split and pass the
resulting dict to the builder constructor.
"""
from __future__ import annotations

import ast
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely import wkt as shp_wkt
from shapely.geometry import Point

from ravp.jaywalk_geofence import (
    infer_signal_path,
    load_signal_intervals_by_dir_turn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_DIRS = ["N", "S", "E", "W"]
# Same encoding as HcFeatureBuilder: G->1, y->0, r->-1, unknown/missing -> -0.5.
_SIG_ENC = {"G": 1.0, "g": 1.0, "y": 0.0, "Y": 0.0, "r": -1.0, "R": -1.0}

SMOOTHING_FRAMES = 5
HORIZON_FRAMES = 20  # 2 s @ 10 FPS

DEFAULT_FPS = 10.0
WAIT_SPEED_THRESH_M_PER_FRAME = 0.02  # ~ 0.2 m/s at 10 FPS
WAIT_TIME_CAP_S = 30.0
INTERSECTION_PHASE_TIME_CAP_S = 120.0
NEIGHBOR_RADIUS_M = 3.0
DENSITY_RADIUS_M = 5.0
MAX_DIST_M = 50.0
MAX_TTC_S = 10.0
MAX_GAP_M = 60.0

PHASE_VOCAB_K = 8  # top-K intersection phase tuples learned on train


def _wrap_pi(a: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _signed_dist(poly, pt: Point) -> float:
    """Signed distance to polygon boundary: negative inside, positive outside."""
    d = float(poly.boundary.distance(pt))
    return -d if poly.contains(pt) else d


def _safe_literal_list(cell) -> list:
    if isinstance(cell, str):
        try:
            return list(ast.literal_eval(cell))
        except Exception:
            try:
                return list(json.loads(cell))
            except Exception:
                return []
    if isinstance(cell, (list, tuple)):
        return list(cell)
    return []


# ---------------------------------------------------------------------------
# Zone loader (extends HcFeatureBuilder's loader to also expose no_ped_area
# polygons, which the existing builder ignores)
# ---------------------------------------------------------------------------


@dataclass
class ZoneBundle:
    walking_areas: List[Any] = field(default_factory=list)
    crosswalks: Dict[str, Any] = field(default_factory=dict)
    no_ped_areas: List[Any] = field(default_factory=list)
    crosswalk_centroids: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    no_ped_centroids: List[Tuple[float, float]] = field(default_factory=list)
    waiting_zones: List[Any] = field(default_factory=list)


def load_zones(zones_csv: Path) -> ZoneBundle:
    zb = ZoneBundle()
    df = pd.read_csv(zones_csv)
    for _, row in df.iterrows():
        name = str(row["zone_name"])
        poly = shp_wkt.loads(row["WKT"])
        if "WalkingArea" in name:
            zb.walking_areas.append(poly)
        elif "Crosswalk" in name:
            if "North" in name:
                zb.crosswalks["N"] = poly
            elif "South" in name:
                zb.crosswalks["S"] = poly
            elif "East" in name:
                zb.crosswalks["E"] = poly
            elif "West" in name:
                zb.crosswalks["W"] = poly
        elif "no_ped_area" in name:
            zb.no_ped_areas.append(poly)

    for d, poly in zb.crosswalks.items():
        c = poly.centroid
        zb.crosswalk_centroids[d] = (float(c.x), float(c.y))
    for poly in zb.no_ped_areas:
        c = poly.centroid
        zb.no_ped_centroids.append((float(c.x), float(c.y)))

    # Synthesise "waiting zones" as 1 m buffer of crosswalk boundary
    # intersected with walking areas. Fail-soft: if anything errors, leave
    # zb.waiting_zones empty and the feature will be 0 for all samples.
    try:
        from shapely.geometry import MultiPolygon
        from shapely.ops import unary_union

        wa_union = unary_union(zb.walking_areas) if zb.walking_areas else None
        for poly in zb.crosswalks.values():
            buf = poly.boundary.buffer(1.0)
            if wa_union is not None:
                buf = buf.intersection(wa_union)
            if buf.is_empty:
                continue
            if isinstance(buf, MultiPolygon):
                for p in buf.geoms:
                    zb.waiting_zones.append(p)
            else:
                zb.waiting_zones.append(buf)
    except Exception as e:  # noqa: BLE001
        print(f"  [zones] waiting-zone synthesis failed ({e}); feature set to 0")

    return zb


def _point_in_any(pt: Point, polys) -> int:
    for p in polys:
        if p.contains(pt):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Phase vocabulary (top-K intersection phase tuples, learned on train split)
# ---------------------------------------------------------------------------


@dataclass
class PhaseVocab:
    tuples: List[Tuple[str, str, str, str]]  # canonical list of top-K tuples
    k: int

    def size(self) -> int:
        return len(self.tuples) + 1  # +1 for "other"

    def encode(self, tup: Tuple[str, str, str, str]) -> np.ndarray:
        v = np.zeros(self.size(), dtype=np.float32)
        try:
            idx = self.tuples.index(tup)
            v[idx] = 1.0
        except ValueError:
            v[-1] = 1.0  # "other"
        return v

    def to_dict(self) -> Dict:
        return {"tuples": [list(t) for t in self.tuples], "k": self.k}

    @classmethod
    def from_dict(cls, d: Dict) -> "PhaseVocab":
        return cls(tuples=[tuple(t) for t in d["tuples"]], k=int(d["k"]))


def _norm_sig_char(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "?"
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return "?"
    u = s.upper()
    if u == "G":
        return "G"
    if s.lower() == "y":
        return "y"
    if s.lower() == "r":
        return "r"
    return "?"


def _row_phase_tuple(row: pd.Series) -> Tuple[str, str, str, str]:
    return tuple(_norm_sig_char(row.get(f"signal_{d}", "?")) for d in SIGNAL_DIRS)  # type: ignore[return-value]


def learn_phase_vocabulary(gt_train: pd.DataFrame, k: int = PHASE_VOCAB_K) -> PhaseVocab:
    cnt: Counter = Counter()
    for _, row in gt_train.iterrows():
        cnt[_row_phase_tuple(row)] += 1
    tuples = [t for t, _ in cnt.most_common(k)]
    return PhaseVocab(tuples=tuples, k=k)


# ---------------------------------------------------------------------------
# Per-scene context (trajectory CSV, KD-trees per frame, signal intervals,
# per-ped frame timeline for waiting time)
# ---------------------------------------------------------------------------


@dataclass
class SceneContext:
    scene_name: str
    traj_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sig_by_dir_ped: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)
    sig_by_dir_vehicle: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)
    # Intersection-wide phase transitions: list of (time_s, tuple_NSEW).
    phase_timeline: List[Tuple[float, Tuple[str, str, str, str]]] = field(default_factory=list)
    # Per-ped frame -> speed_smooth (for waiting-time scan).
    ped_speed_by_id: Dict[int, Dict[int, float]] = field(default_factory=dict)
    fps: float = DEFAULT_FPS
    signal_missing: bool = False
    # Per-frame KD-trees + row indices (lazily built).
    _frames_ped: Dict[int, np.ndarray] = field(default_factory=dict)
    _frames_veh: Dict[int, np.ndarray] = field(default_factory=dict)


def _build_phase_timeline(
    sig_by_dir_ped: Dict[str, List[Tuple[float, float, str]]]
) -> List[Tuple[float, Tuple[str, str, str, str]]]:
    """Sample the intersection-wide pedestrian phase at every interval begin
    in any direction, producing a time-ordered [(t, tuple_NSEW), ...]."""
    # All boundary times.
    times: List[float] = []
    for d in SIGNAL_DIRS:
        for b, _e, _st in sig_by_dir_ped.get(d, []):
            times.append(b)
    times = sorted(set(times))
    if not times:
        return []

    def _state_at(d: str, t: float) -> str:
        intervals = sig_by_dir_ped.get(d, [])
        begins = [x[0] for x in intervals]
        i = bisect_right(begins, t) - 1
        if i < 0:
            return "?"
        b, e, st = intervals[i]
        if b <= t < e:
            return _norm_sig_char(st)
        return "?"

    out: List[Tuple[float, Tuple[str, str, str, str]]] = []
    prev_tuple: Optional[Tuple[str, str, str, str]] = None
    for t in times:
        tup = tuple(_state_at(d, t) for d in SIGNAL_DIRS)  # type: ignore[assignment]
        if tup != prev_tuple:
            out.append((float(t), tup))  # type: ignore[arg-type]
            prev_tuple = tup  # type: ignore[assignment]
    return out


def _current_and_previous_phase(
    timeline: List[Tuple[float, Tuple[str, str, str, str]]], t: float
) -> Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str], float]:
    """Return (current_tuple, previous_tuple, seconds_since_last_transition)."""
    if not timeline:
        empty = ("?", "?", "?", "?")
        return empty, empty, INTERSECTION_PHASE_TIME_CAP_S
    begins = [x[0] for x in timeline]
    i = bisect_right(begins, t) - 1
    if i < 0:
        cur = timeline[0][1]
        return cur, cur, INTERSECTION_PHASE_TIME_CAP_S
    cur = timeline[i][1]
    prev = timeline[i - 1][1] if i - 1 >= 0 else cur
    dt = float(t - timeline[i][0])
    return cur, prev, max(0.0, min(dt, INTERSECTION_PHASE_TIME_CAP_S))


def _last_transition_time(
    intervals: List[Tuple[float, float, str]], t: float
) -> float:
    """Seconds since the last state transition on this direction's timeline."""
    if not intervals:
        return 60.0
    begins = [x[0] for x in intervals]
    i = bisect_right(begins, t) - 1
    if i < 0:
        return 60.0
    b, _e, _st = intervals[i]
    return float(max(0.0, min(t - b, 60.0)))


# ---------------------------------------------------------------------------
# Scene loader
# ---------------------------------------------------------------------------


def load_scene_context(
    scene_name: str,
    traj_dir: Path,
    fps: float = DEFAULT_FPS,
) -> SceneContext:
    """Load a single scene's trajectory + signal CSVs and pre-index them."""
    ctx = SceneContext(scene_name=scene_name, fps=fps)

    traj_path = traj_dir / f"{scene_name}_Traj.csv"
    if not traj_path.is_file():
        # No trajectory -> interaction + waiting-time features will be NaN.
        ctx.signal_missing = True
        return ctx

    df = pd.read_csv(traj_path)
    need_cols = {"frame", "id", "cx_m", "cy_m", "type"}
    if not need_cols.issubset(df.columns):
        ctx.signal_missing = True
        return ctx

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    df["type_lc"] = df["type"].astype(str).str.lower().str.strip()
    if "speed_smooth" not in df.columns:
        df["speed_smooth"] = 0.0
    ctx.traj_df = df

    # Per-pedestrian (frame -> speed_smooth) for waiting-time scan.
    ped = df[df["type_lc"] == "pedestrian"]
    for pid, g in ped.groupby("id", sort=False):
        ctx.ped_speed_by_id[int(pid)] = {
            int(f): float(s) for f, s in zip(g["frame"], g["speed_smooth"])
        }

    # Signals
    signal_path = infer_signal_path(traj_path)
    if signal_path is None or not signal_path.is_file():
        ctx.signal_missing = True
        return ctx

    try:
        sig_full = load_signal_intervals_by_dir_turn(signal_path)
    except Exception:
        ctx.signal_missing = True
        return ctx

    # Split by (turn == 'p' for pedestrians, others for vehicles).
    for (d, turn), intervals in sig_full.items():
        if turn == "p":
            ctx.sig_by_dir_ped.setdefault(d, []).extend(intervals)
        else:
            ctx.sig_by_dir_vehicle.setdefault(d, []).extend(intervals)

    # Sort intervals.
    for d in list(ctx.sig_by_dir_ped.keys()):
        ctx.sig_by_dir_ped[d].sort(key=lambda x: x[0])
    for d in list(ctx.sig_by_dir_vehicle.keys()):
        ctx.sig_by_dir_vehicle[d].sort(key=lambda x: x[0])

    ctx.phase_timeline = _build_phase_timeline(ctx.sig_by_dir_ped)
    return ctx


def _frame_arrays(ctx: SceneContext, frame: int, vehicle: bool) -> np.ndarray:
    """Return (N, 5) array [x, y, id, vx, vy] for peds (vehicle=False) or
    vehicles (vehicle=True) at the given frame. Cached per frame per kind."""
    cache = ctx._frames_veh if vehicle else ctx._frames_ped
    if frame in cache:
        return cache[frame]
    if ctx.traj_df.empty:
        cache[frame] = np.zeros((0, 5), dtype=np.float64)
        return cache[frame]

    df = ctx.traj_df
    if vehicle:
        sub = df[(df["frame"] == frame) & (df["type_lc"] != "pedestrian")]
    else:
        sub = df[(df["frame"] == frame) & (df["type_lc"] == "pedestrian")]

    cols: List[np.ndarray] = []
    cols.append(sub["cx_m"].to_numpy(dtype=np.float64))
    cols.append(sub["cy_m"].to_numpy(dtype=np.float64))
    cols.append(sub["id"].to_numpy(dtype=np.float64))
    vx = sub["vx"].to_numpy(dtype=np.float64) if "vx" in sub.columns else np.zeros(len(sub))
    vy = sub["vy"].to_numpy(dtype=np.float64) if "vy" in sub.columns else np.zeros(len(sub))
    cols.append(vx)
    cols.append(vy)
    arr = np.stack(cols, axis=1) if len(sub) else np.zeros((0, 5), dtype=np.float64)
    cache[frame] = arr
    return arr


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------


class TabularFeatureBuilder:
    """Compute the ~60-dim tabular feature vector for a GT row.

    Call `set_scene(ctx)` before `compute_row(row)` for each scene's rows.
    Stateless across scenes except for the phase vocabulary + zone bundle
    which live on the builder.
    """

    def __init__(
        self,
        zones: ZoneBundle,
        phase_vocab: PhaseVocab,
        fps: float = DEFAULT_FPS,
    ):
        self.zones = zones
        self.phase_vocab = phase_vocab
        self.fps = fps
        self._ctx: Optional[SceneContext] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_groups: Optional[List[str]] = None

    def set_scene(self, ctx: SceneContext) -> None:
        self._ctx = ctx

    # ---- column definitions -------------------------------------------------

    def feature_names(self) -> List[str]:
        if self._feature_names is not None:
            return self._feature_names
        names: List[str] = []
        groups: List[str] = []

        def add(n, g):
            names.append(n)
            groups.append(g)

        # Group A — kinematics (10)
        for n in ("x", "y", "vx", "vy", "speed", "ax", "ay",
                  "heading_sin", "heading_cos", "heading_change"):
            add(n, "A")

        # Group B — map/legal (13)
        add("inside_walking_area", "B")
        add("inside_crosswalk", "B")
        add("inside_waiting_zone", "B")
        for d in SIGNAL_DIRS:
            add(f"signed_dist_cw_{d}", "B")
        add("signed_dist_walkarea", "B")
        add("dist_to_nearest_crosswalk", "B")
        add("dist_to_nearest_no_ped_area", "B")
        add("heading_alignment_to_nearest_cw", "B")
        add("heading_alignment_to_nearest_noped", "B")
        add("ttc_road_boundary_cv", "B")

        # Group C — instantaneous signal (9)
        for d in SIGNAL_DIRS:
            add(f"signal_{d}", "C")
        for d in SIGNAL_DIRS:
            add(f"time_since_phase_change_{d}", "C")
        add("any_conflict_vehicle_green", "C")

        # Group D — interaction (8)
        for n in ("n_neighbors_3m", "n_neighbors_crossing_3m", "nearest_ped_dist",
                  "nearest_vehicle_dist", "nearest_vehicle_speed",
                  "nearest_vehicle_ttc", "gap_size", "local_density"):
            add(n, "D")

        # Group E — waiting time + phase history (2K+4)
        add("ped_waiting_time_s", "E")
        for i in range(self.phase_vocab.size() - 1):
            add(f"phase_current_oh_{i}", "E")
        add("phase_current_oh_other", "E")
        for i in range(self.phase_vocab.size() - 1):
            add(f"phase_previous_oh_{i}", "E")
        add("phase_previous_oh_other", "E")
        add("time_since_intersection_phase_change_s", "E")

        self._feature_names = names
        self._feature_groups = groups
        return names

    def feature_groups(self) -> List[str]:
        if self._feature_groups is None:
            self.feature_names()
        return self._feature_groups  # type: ignore[return-value]

    # ---- main compute -------------------------------------------------------

    def compute_row(self, row: pd.Series) -> np.ndarray:
        ctx = self._ctx
        assert ctx is not None, "Call set_scene() before compute_row()."
        names = self.feature_names()

        cx = _safe_literal_list(row["cx_m"])
        cy = _safe_literal_list(row["cy_m"])
        spd = _safe_literal_list(row["speed_smooth"])
        if not cx or not cy:
            return np.full(len(names), np.nan, dtype=np.float32)

        x_cur = float(cx[-1])
        y_cur = float(cy[-1])
        speed_cur = float(spd[-1]) if spd else 0.0

        sf = min(SMOOTHING_FRAMES, len(cx) - 1)
        if sf > 0:
            vx = (cx[-1] - cx[-(sf + 1)]) / sf
            vy = (cy[-1] - cy[-(sf + 1)]) / sf
        else:
            vx = vy = 0.0

        # Acceleration: diff of velocities at tau and tau-5.
        if len(cx) >= 11:
            vx_prev = (cx[-6] - cx[-11]) / 5
            vy_prev = (cy[-6] - cy[-11]) / 5
            ax = (vx - vx_prev) / 5.0
            ay = (vy - vy_prev) / 5.0
        else:
            ax = ay = float("nan")

        heading = math.atan2(vy, vx) if (vx != 0 or vy != 0) else 0.0
        heading_sin = math.sin(heading)
        heading_cos = math.cos(heading)

        if len(cx) >= 11 and not (np.isnan(ax) or np.isnan(ay)):
            vx_prev = (cx[-6] - cx[-11]) / 5
            vy_prev = (cy[-6] - cy[-11]) / 5
            h_prev = math.atan2(vy_prev, vx_prev) if (vx_prev != 0 or vy_prev != 0) else 0.0
            heading_change = _wrap_pi(heading - h_prev)
        else:
            heading_change = 0.0

        # ---- Group B -------------------------------------------------------
        pt = Point(x_cur, y_cur)
        inside_walk = _point_in_any(pt, self.zones.walking_areas)
        inside_cw = _point_in_any(pt, list(self.zones.crosswalks.values()))
        inside_wait = _point_in_any(pt, self.zones.waiting_zones)

        cw_dists = {}
        for d in SIGNAL_DIRS:
            if d in self.zones.crosswalks:
                cw_dists[d] = _signed_dist(self.zones.crosswalks[d], pt)
            else:
                cw_dists[d] = float(MAX_DIST_M)

        wa_dists = [_signed_dist(wa, pt) for wa in self.zones.walking_areas]
        min_wa = min(wa_dists) if wa_dists else float(MAX_DIST_M)

        nearest_cw_dir = min(SIGNAL_DIRS, key=lambda d: abs(cw_dists[d]))
        dist_to_nearest_cw = abs(cw_dists[nearest_cw_dir])

        if self.zones.no_ped_areas:
            np_dists = [_signed_dist(p, pt) for p in self.zones.no_ped_areas]
            dist_to_nearest_noped = min(np_dists)
        else:
            dist_to_nearest_noped = float(MAX_DIST_M)

        # Heading alignments
        def _align(cx_c, cy_c):
            dx = cx_c - x_cur
            dy = cy_c - y_cur
            n = math.hypot(dx, dy)
            if n < 1e-6 or math.hypot(vx, vy) < 1e-6:
                return 0.0
            return float((vx * dx + vy * dy) / (n * math.hypot(vx, vy)))

        cw_c = self.zones.crosswalk_centroids.get(nearest_cw_dir, (x_cur, y_cur))
        align_cw = _align(cw_c[0], cw_c[1])

        if self.zones.no_ped_centroids:
            noped_c = min(self.zones.no_ped_centroids,
                          key=lambda c: (c[0] - x_cur) ** 2 + (c[1] - y_cur) ** 2)
            align_noped = _align(noped_c[0], noped_c[1])
        else:
            align_noped = 0.0

        # TTC to road boundary (CV projection exits all walking areas)
        ttc_road = float(HORIZON_FRAMES)
        if inside_walk and (vx != 0 or vy != 0):
            for step in range(1, HORIZON_FRAMES + 1):
                pt_f = Point(x_cur + step * vx, y_cur + step * vy)
                still_in = any(wa.contains(pt_f) for wa in self.zones.walking_areas)
                if not still_in:
                    ttc_road = float(step)
                    break

        # ---- Group C — instantaneous signal --------------------------------
        sigs = [_SIG_ENC.get(_norm_sig_char(row.get(f"signal_{d}", "?")), -0.5)
                for d in SIGNAL_DIRS]

        # Time since phase change (per direction, in seconds).
        frame_current = int(row["frame_current"])
        t_current = frame_current / self.fps if self.fps > 0 else 0.0
        tspc = []
        for d in SIGNAL_DIRS:
            iv = ctx.sig_by_dir_ped.get(d, [])
            tspc.append(_last_transition_time(iv, t_current) if iv else float("nan"))

        any_veh_green = 0
        if ctx.sig_by_dir_vehicle:
            for d, intervals in ctx.sig_by_dir_vehicle.items():
                begins = [x[0] for x in intervals]
                i = bisect_right(begins, t_current) - 1
                if i < 0:
                    continue
                b, e, st = intervals[i]
                if b <= t_current < e and _norm_sig_char(st) == "G":
                    any_veh_green = 1
                    break
        else:
            any_veh_green = 0 if not ctx.signal_missing else int(-1)  # keep binary; unknown -> 0
            if ctx.signal_missing:
                any_veh_green = 0  # fall-through

        # ---- Group D — interaction -----------------------------------------
        feats_D = self._interaction_features(ctx, frame_current, x_cur, y_cur, vx, vy, row)

        # ---- Group E — waiting time + phase history -----------------------
        wait_s = self._waiting_time(ctx, int(row["ped_id"]), frame_current)
        cur_tup, prev_tup, t_since_intersection_change = _current_and_previous_phase(
            ctx.phase_timeline, t_current
        )
        cur_oh = self.phase_vocab.encode(cur_tup)
        prev_oh = self.phase_vocab.encode(prev_tup)

        # ---- assemble ------------------------------------------------------
        feats: List[float] = []
        feats += [x_cur, y_cur, vx, vy, speed_cur, ax, ay,
                  heading_sin, heading_cos, heading_change]  # A (10)
        feats += [float(inside_walk), float(inside_cw), float(inside_wait)]  # B1-B3
        feats += [cw_dists[d] for d in SIGNAL_DIRS]                          # B4-B7
        feats += [min_wa, dist_to_nearest_cw, dist_to_nearest_noped,
                  align_cw, align_noped, ttc_road]                           # B8-B13
        feats += sigs + tspc + [float(any_veh_green)]                        # C (9)
        feats += list(feats_D)                                               # D (8)
        feats += [wait_s]
        feats += cur_oh.tolist()
        feats += prev_oh.tolist()
        feats += [t_since_intersection_change]                               # E
        arr = np.array(feats, dtype=np.float32)
        expected = len(self.feature_names())
        if arr.shape[0] != expected:
            raise AssertionError(
                f"Feature length mismatch: got {arr.shape[0]} expected {expected}"
            )
        return arr

    # ---- helpers ------------------------------------------------------------

    def _interaction_features(
        self,
        ctx: SceneContext,
        frame: int,
        x: float,
        y: float,
        vx: float,
        vy: float,
        row: pd.Series,
    ) -> Tuple[float, ...]:
        target_id = int(row["ped_id"])

        peds = _frame_arrays(ctx, frame, vehicle=False)
        vehs = _frame_arrays(ctx, frame, vehicle=True)

        # Peds: drop self.
        if len(peds):
            mask = peds[:, 2].astype(int) != target_id
            peds_others = peds[mask]
        else:
            peds_others = peds

        if len(peds_others):
            d2 = (peds_others[:, 0] - x) ** 2 + (peds_others[:, 1] - y) ** 2
            d = np.sqrt(d2)
            nearest_ped = float(min(d.min(), MAX_DIST_M))
            n_nbr = int((d <= NEIGHBOR_RADIUS_M).sum())
            # Neighbours that are inside a crosswalk or no-ped area.
            nbr_mask = d <= NEIGHBOR_RADIUS_M
            n_crossing = 0
            for px, py in peds_others[nbr_mask, 0:2]:
                p = Point(float(px), float(py))
                if _point_in_any(p, list(self.zones.crosswalks.values())) or \
                        _point_in_any(p, self.zones.no_ped_areas):
                    n_crossing += 1
            # Local density in 5 m disc.
            n_dens = int((d <= DENSITY_RADIUS_M).sum())
            local_density = float(n_dens / (math.pi * DENSITY_RADIUS_M ** 2))
        else:
            nearest_ped = float(MAX_DIST_M)
            n_nbr = 0
            n_crossing = 0
            local_density = 0.0

        # Vehicles
        if len(vehs):
            d2 = (vehs[:, 0] - x) ** 2 + (vehs[:, 1] - y) ** 2
            d = np.sqrt(d2)
            idx_near = int(np.argmin(d))
            nearest_veh = float(min(d[idx_near], MAX_DIST_M))
            vx_v = float(vehs[idx_near, 3])
            vy_v = float(vehs[idx_near, 4])
            veh_speed = float(math.hypot(vx_v, vy_v) * ctx.fps)  # m/s
            # TTC: time for nearest vehicle on CV path to reach ped's CV
            # position at tau+1s. Use 1D approximation along connecting line.
            tx = x + vx * ctx.fps  # ped pos in ~1 s (fps frames later, vx in m/frame)
            ty = y + vy * ctx.fps
            vx_vs = vx_v * ctx.fps  # veh velocity m/s
            vy_vs = vy_v * ctx.fps
            rel_x = float(vehs[idx_near, 0]) - tx
            rel_y = float(vehs[idx_near, 1]) - ty
            rel_speed = math.hypot(vx_vs, vy_vs)
            if rel_speed > 1e-6:
                # Approaching iff rel_pos . -velocity > 0
                dot = (-rel_x) * vx_vs + (-rel_y) * vy_vs
                if dot > 0:
                    ttc_veh = float(min(math.hypot(rel_x, rel_y) / rel_speed, MAX_TTC_S))
                else:
                    ttc_veh = float(MAX_TTC_S)
            else:
                ttc_veh = float(MAX_TTC_S)

            # Gap size: distance between front and behind vehicles along
            # pedestrian velocity direction. If ped has no velocity, use
            # x-axis as arbitrary road direction.
            vxn = vx
            vyn = vy
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

        return (
            float(n_nbr),
            float(n_crossing),
            nearest_ped,
            nearest_veh,
            veh_speed,
            ttc_veh,
            gap,
            local_density,
        )

    def _waiting_time(self, ctx: SceneContext, ped_id: int, frame_current: int) -> float:
        spd = ctx.ped_speed_by_id.get(int(ped_id), {})
        if not spd:
            return 0.0
        thresh = WAIT_SPEED_THRESH_M_PER_FRAME
        # Scan backward: count consecutive frames with speed below threshold.
        waits = 0
        f = int(frame_current)
        cap_frames = int(WAIT_TIME_CAP_S * ctx.fps)
        while f >= 0 and waits < cap_frames:
            s = spd.get(f)
            if s is None or s >= thresh:
                break
            waits += 1
            f -= 1
        return float(waits / ctx.fps if ctx.fps > 0 else waits)
