#!/usr/bin/env python3
"""Risk features for SMI-VP-Forecaster Step 1.

Given K sampled futures from the frozen Trajectron++ forecaster (plus the window
origin, future times and scene), compute three groups of features:

  checker  (B) : hard-checker statistics over the K samples.
  softdist (C) : soft signed-distance / near-boundary risk over predicted points.
  signal   (D) : portable pedestrian-signal-timing features at tau and over the
                 horizon (schedule-based: uses the recorded signal intervals).

Every feature is named; FEATURE_GROUPS lets the ablations select subsets. All
features are robust to missing polygons/signals (value 0 + a *_missing flag).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from ravp.checker import GeofenceChecker, FPS
from ravp.jaywalk_geofence import check_jaywalk_status, lookup_state, normalize_state

DT = 1.0 / FPS
HORIZON_S = 20 * DT           # 2.0 s
MARGINS = (0.25, 0.5, 1.0)
PERMISSIVE = {"G", "y"}

# bump when the feature set changes so caches don't mix (v3 adds the 'agg' group)
FEATURE_VERSION = 2

CHECKER_NAMES = ["p_check_mean", "p_check_count", "p_check_max", "p_check_majority",
                 "earliest_viol_time_mean", "earliest_viol_time_min"]
SOFTDIST_NAMES = ["min_sd_unsafe", "mean_persample_min_sd", "sd_p05", "sd_p10",
                  "frac_within_0p25", "frac_within_0p5", "frac_within_1p0",
                  "min_dist_boundary", "softdist_missing"]
SIGNAL_NAMES = ["sig_green", "sig_yellow", "sig_red", "sig_unknown",
                "time_since_change", "time_to_change", "changes_in_horizon",
                "frac_permissive", "frac_nonpermissive", "veh_green", "signal_missing"]
# v3 tail-aggregation over the K samples (free: numpy over per-sample arrays).
# Surfaces rare risky futures the *mean* checker score misses -> higher recall.
AGG_NAMES = ["frac_samples_near_0p5", "frac_samples_near_1p0",
             "persample_min_sd_p10", "topk_mean_risk", "earliest_viol_time_p10"]
# portable static map features at the decision frame (SMI-VP-style)
MAP_NAMES = ["map_d_safe", "map_d_nearest_cw", "map_in_safe", "map_missing"]
FEATURE_GROUPS = {"checker": CHECKER_NAMES, "softdist": SOFTDIST_NAMES,
                  "signal": SIGNAL_NAMES, "agg": AGG_NAMES, "map": MAP_NAMES}
ALL_NAMES = CHECKER_NAMES + SOFTDIST_NAMES + SIGNAL_NAMES + AGG_NAMES


def _interval_edges(intervals):
    """Sorted unique begin times (phase-change boundaries)."""
    return sorted({b for b, _e, _s in intervals})


class RiskExtractor:
    def __init__(self, dataset: str):
        self.ck = GeofenceChecker(dataset)
        wa = list(self.ck.zones.walking_areas)
        cw = list(self.ck.zones.crosswalks.values())
        self.safe_union = unary_union(wa + cw) if (wa or cw) else None
        self.crosswalks = self.ck.zones.crosswalks
        self._vsig: Dict[str, dict] = {}   # vehicle-turn signal per scene

    # ---- signed distance to the unsafe region (negative inside unsafe) -------
    def _signed_dist_unsafe(self, x: float, y: float) -> float:
        """+ inside safe (margin to leaving), - inside unsafe (depth)."""
        if self.safe_union is None:
            return 0.0
        p = Point(x, y)
        if self.safe_union.contains(p):
            return float(p.distance(self.safe_union.boundary))
        return float(-p.distance(self.safe_union))

    def _veh_sig_for(self, scene: str) -> dict:
        if scene not in self._vsig:
            from ravp.jaywalk_geofence import infer_signal_path, load_signal_intervals_by_dir_turn
            d = {}
            sp = infer_signal_path(self.ck.traj_dir / f"{scene}_Traj.csv")
            if sp is not None and sp.is_file():
                full = load_signal_intervals_by_dir_turn(sp)
                d = {k: v for k, v in full.items() if k[1] != "p"}
            self._vsig[scene] = d
        return self._vsig[scene]

    def _nearest_cw_dir(self, ox: float, oy: float):
        if not self.crosswalks:
            return None
        p = Point(ox, oy)
        return min(self.crosswalks, key=lambda d: self.crosswalks[d].distance(p))

    def nonpermissive_future(self, origin, times: List[float], scene: str) -> np.ndarray:
        """Per future-frame indicator (len == len(times)) that the nearest crosswalk's
        pedestrian signal is NON-permissive (red/unknown) at that time. Used by the
        VP-5 signal-aware differentiable checker. 1.0 = non-permissive."""
        sig = self.ck._sig_for(scene)
        d = self._nearest_cw_dir(origin[0], origin[1])
        intervals = sig.get(d, []) if d is not None else []
        out = np.zeros(len(times), dtype=np.float32)
        for i, t in enumerate(times):
            st = lookup_state(intervals, float(t)) if intervals else None
            n = normalize_state(st) if st is not None else None
            out[i] = 0.0 if n in ("G", "y") else 1.0  # red/unknown/missing -> non-permissive
        return out

    def map_features(self, origin) -> Dict[str, float]:
        """Portable static map features at the decision frame (no signal, no samples)."""
        ox, oy = origin
        if self.safe_union is None:
            return {"map_d_safe": 0.0, "map_d_nearest_cw": 0.0, "map_in_safe": 0.0, "map_missing": 1.0}
        p = Point(ox, oy)
        d_safe = self._signed_dist_unsafe(ox, oy)  # + inside safe, - in road core
        if self.crosswalks:
            d_cw = float(min(self.crosswalks[d].distance(p) for d in self.crosswalks))
        else:
            d_cw = 50.0
        return {"map_d_safe": float(d_safe), "map_d_nearest_cw": d_cw,
                "map_in_safe": 1.0 if self.safe_union.contains(p) else 0.0, "map_missing": 0.0}

    def signal_features(self, origin, times: List[float], scene: str) -> Dict[str, float]:
        """The 11 SIGNAL_NAMES features only (no sampling needed). Used by v4."""
        sig = self.ck._sig_for(scene)
        ox, oy = origin
        f: Dict[str, float] = {}
        tau_t = float(times[0]) - DT
        d = self._nearest_cw_dir(ox, oy)
        intervals = sig.get(d, []) if d is not None else []
        st0 = lookup_state(intervals, tau_t) if intervals else None
        n0 = normalize_state(st0) if st0 is not None else None
        f["sig_green"] = 1.0 if n0 == "G" else 0.0
        f["sig_yellow"] = 1.0 if n0 == "y" else 0.0
        f["sig_red"] = 1.0 if n0 == "r" else 0.0
        f["sig_unknown"] = 1.0 if n0 not in ("G", "y", "r") else 0.0
        edges = _interval_edges(intervals)
        past = [e for e in edges if e <= tau_t]
        future = [e for e in edges if tau_t < e <= tau_t + HORIZON_S]
        nxt = [e for e in edges if e > tau_t]
        f["time_since_change"] = float(min(tau_t - past[-1], 60.0)) if past else 60.0
        f["time_to_change"] = float(min(nxt[0] - tau_t, 60.0)) if nxt else 60.0
        f["changes_in_horizon"] = 1.0 if future else 0.0
        states = [normalize_state(lookup_state(intervals, float(t)) or "?") for t in times] if intervals else []
        if states:
            perm = sum(1 for s in states if s in PERMISSIVE) / len(states)
            f["frac_permissive"] = float(perm); f["frac_nonpermissive"] = float(1.0 - perm)
        else:
            f["frac_permissive"] = 0.0; f["frac_nonpermissive"] = 0.0
        vsig = self._veh_sig_for(scene)
        vg = 0.0
        for (vd, vt), iv in vsig.items():
            vs = lookup_state(iv, tau_t)
            if vs is not None and normalize_state(vs) == "G":
                vg = 1.0; break
        f["veh_green"] = vg
        f["signal_missing"] = 1.0 if not intervals else 0.0
        return f

    # ---- per-window features --------------------------------------------------
    def extract(self, pred_rel: np.ndarray, origin, times: List[float],
                scene: str) -> Dict[str, float]:
        S, Tf, _ = pred_rel.shape
        ox, oy = origin
        sig = self.ck._sig_for(scene)

        # ---- B: hard checker over samples ----
        viol = np.zeros(S, dtype=bool)
        earliest = np.full(S, HORIZON_S, dtype=np.float32)
        # ---- C: signed distances over all predicted points ----
        persample_min = np.full(S, np.nan, dtype=np.float32)
        all_sd = []
        wa, cw = self.ck.zones.walking_areas, self.crosswalks
        for s in range(S):
            sd_s = []
            first = None
            for k in range(Tf):
                x = pred_rel[s, k, 0] + ox; y = pred_rel[s, k, 1] + oy
                is_jay, _ = check_jaywalk_status(Point(x, y), float(times[k]), wa, cw, sig)
                if is_jay and first is None:
                    first = k
                sd_s.append(self._signed_dist_unsafe(x, y))
            if first is not None:
                viol[s] = True
                earliest[s] = first * DT
            persample_min[s] = float(np.min(sd_s)) if sd_s else 0.0
            all_sd.extend(sd_s)

        f: Dict[str, float] = {}
        nviol = int(viol.sum())
        f["p_check_mean"] = nviol / S
        f["p_check_count"] = float(nviol)
        f["p_check_max"] = 1.0 if nviol > 0 else 0.0
        f["p_check_majority"] = 1.0 if nviol > S / 2 else 0.0
        f["earliest_viol_time_mean"] = float(earliest[viol].mean()) if nviol else HORIZON_S
        f["earliest_viol_time_min"] = float(earliest[viol].min()) if nviol else HORIZON_S

        all_sd = np.array(all_sd, dtype=np.float32) if all_sd else np.zeros(1, np.float32)
        miss = 1.0 if self.safe_union is None else 0.0
        f["min_sd_unsafe"] = float(all_sd.min())
        f["mean_persample_min_sd"] = float(np.nanmean(persample_min))
        f["sd_p05"] = float(np.percentile(all_sd, 5))
        f["sd_p10"] = float(np.percentile(all_sd, 10))
        for m, nm in zip(MARGINS, ["frac_within_0p25", "frac_within_0p5", "frac_within_1p0"]):
            f[nm] = float((all_sd < m).mean())
        f["min_dist_boundary"] = float(np.abs(all_sd).min())
        f["softdist_missing"] = miss

        # ---- v3 tail aggregation over the K samples ----
        f["frac_samples_near_0p5"] = float((persample_min < 0.5).mean())
        f["frac_samples_near_1p0"] = float((persample_min < 1.0).mean())
        f["persample_min_sd_p10"] = float(np.percentile(persample_min, 10))
        kk = max(1, int(np.ceil(0.25 * S)))
        f["topk_mean_risk"] = float(np.sort(persample_min)[:kk].mean())
        f["earliest_viol_time_p10"] = float(np.percentile(earliest, 10))

        # ---- D: signal timing at tau / over horizon ----
        tau_t = float(times[0]) - DT
        d = self._nearest_cw_dir(ox, oy)
        intervals = sig.get(d, []) if d is not None else []
        st0 = lookup_state(intervals, tau_t) if intervals else None
        n0 = normalize_state(st0) if st0 is not None else None
        f["sig_green"] = 1.0 if n0 == "G" else 0.0
        f["sig_yellow"] = 1.0 if n0 == "y" else 0.0
        f["sig_red"] = 1.0 if n0 == "r" else 0.0
        f["sig_unknown"] = 1.0 if n0 not in ("G", "y", "r") else 0.0
        edges = _interval_edges(intervals)
        past = [e for e in edges if e <= tau_t]
        future = [e for e in edges if tau_t < e <= tau_t + HORIZON_S]
        nxt = [e for e in edges if e > tau_t]
        f["time_since_change"] = float(min(tau_t - past[-1], 60.0)) if past else 60.0
        f["time_to_change"] = float(min(nxt[0] - tau_t, 60.0)) if nxt else 60.0
        f["changes_in_horizon"] = 1.0 if future else 0.0
        states = [normalize_state(lookup_state(intervals, float(t)) or "?") for t in times] if intervals else []
        if states:
            perm = sum(1 for s in states if s in PERMISSIVE) / len(states)
            f["frac_permissive"] = float(perm)
            f["frac_nonpermissive"] = float(1.0 - perm)
        else:
            f["frac_permissive"] = 0.0
            f["frac_nonpermissive"] = 0.0
        # conflicting vehicle green at tau (any vehicle-turn direction green)
        vsig = self._veh_sig_for(scene)
        vg = 0.0
        for (vd, vt), iv in vsig.items():
            vs = lookup_state(iv, tau_t)
            if vs is not None and normalize_state(vs) == "G":
                vg = 1.0; break
        f["veh_green"] = vg
        f["signal_missing"] = 1.0 if not intervals else 0.0
        return f
