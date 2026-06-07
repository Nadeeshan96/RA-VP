#!/usr/bin/env python3
"""Build per-window trajectory caches for the trajectory-prediction baselines.

For every ground-truth window (scene, ped_id, frame_current=tau, label) we store
the target's history and TRUE future positions, a snapshot of nearby agents, and
enough metadata for the geofence checker to evaluate predicted futures. All
coordinates are relative to the target position at tau (origin stored in meta so
the checker can map predictions back to absolute metres).

Cache: logs/traj_baselines_mac/cache_<DS>/<split>.pt
  hist   (N, Th, 4)  target [dx,dy,vx,vy] relative to tau
  fut    (N, Tf, 2)  target TRUE future [dx,dy] relative to tau
  neigh  (N, K, 4)   nearest-K agents at tau [dx,dy,vx,vy] (zero-padded)
  y      (N,)        GT violation label
  meta   list[dict]  scene, ped_id, frame_current, origin (x0,y0),
                     future_frames, future_times
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ravp.paths import CACHE_ROOT
from ravp.split_utils import load_split
from ravp.geofence import load_scene_context, resolve_dataset_paths

TH, TF, FPS, K = 20, 20, 10.0, 8


def _vel(xy: np.ndarray) -> np.ndarray:
    """Per-step velocity (m/s) from positions; first step repeats."""
    v = np.zeros_like(xy)
    if len(xy) > 1:
        v[1:] = (xy[1:] - xy[:-1]) * FPS
        v[0] = v[1]
    return v


def build_cache(dataset: str, split: str, force: bool = False) -> Path:
    out_dir = CACHE_ROOT / f"cache_{dataset}"
    out = out_dir / f"{split}.pt"
    if out.exists() and not force:
        print(f"[{dataset}/{split}] exists -> {out}")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)

    root, gt_path, zones_csv, traj_dir = resolve_dataset_paths(dataset)
    gt = pd.read_csv(gt_path)
    scenes = set(load_split(root)[split])
    gt = gt[gt["scene_name"].isin(scenes)].reset_index(drop=True)

    H, F, NB, Y, META = [], [], [], [], []
    for scene, rows in gt.groupby("scene_name"):
        ctx = load_scene_context(scene, traj_dir, fps=FPS)
        df = ctx.traj_df
        if df.empty:
            continue
        peds = df[df["type_lc"] == "pedestrian"] if "type_lc" in df.columns else df
        # frame -> DataFrame of agents present (for neighbour snapshot)
        by_frame = {int(f): g for f, g in df.groupby("frame")}
        # id -> {frame: (x,y)} (built once; used for target + neighbour velocity)
        id2fr2xy: Dict[int, Dict[int, tuple]] = {}
        for pid_i, g_i in peds.groupby("id"):
            id2fr2xy[int(pid_i)] = {int(fr): (float(x), float(y))
                                    for fr, x, y in zip(g_i["frame"], g_i["cx_m"], g_i["cy_m"])}
        gtime = {}
        has_time = "time" in df.columns
        if has_time:
            for pid_i, g_i in peds.groupby("id"):
                gtime[int(pid_i)] = {int(fr): float(t) for fr, t in zip(g_i["frame"], g_i["time"])}
        for _, r in rows.iterrows():
            pid = int(r["ped_id"]); tau = int(r["frame_current"])
            fr2xy = id2fr2xy.get(pid, {})
            need = list(range(tau - TH + 1, tau + TF + 1))
            if any(fr not in fr2xy for fr in need):
                continue
            xy = np.array([fr2xy[fr] for fr in need], dtype=np.float32)  # (Th+Tf,2)
            x0, y0 = xy[TH - 1]
            rel = xy - np.array([x0, y0], dtype=np.float32)
            hist_xy = rel[:TH]; fut_xy = rel[TH:]
            hist = np.concatenate([hist_xy, _vel(hist_xy)], axis=1)  # (Th,4)

            # neighbour snapshot at tau (nearest K non-target agents)
            fa = by_frame.get(tau)
            neigh = np.zeros((K, 4), dtype=np.float32)
            if fa is not None:
                others = fa[fa["id"] != pid]
                if len(others):
                    pxy = others[["cx_m", "cy_m"]].to_numpy(np.float32)
                    d = np.linalg.norm(pxy - np.array([x0, y0]), axis=1)
                    order = np.argsort(d)[:K]
                    for i, j in enumerate(order):
                        ox, oy = pxy[j]
                        # neighbour velocity from its own prev frame if available
                        oid = int(others.iloc[j]["id"])
                        prev = id2fr2xy.get(oid, {}).get(tau - 1)
                        if prev is not None:
                            nvx = (ox - prev[0]) * FPS; nvy = (oy - prev[1]) * FPS
                        else:
                            nvx = nvy = 0.0
                        neigh[i] = [ox - x0, oy - y0, nvx, nvy]

            if has_time:
                tser = gtime.get(pid, {})
                fut_times = [tser.get(fr, (fr - 1) / FPS) for fr in need[TH:]]
            else:
                fut_times = [(fr - 1) / FPS for fr in need[TH:]]

            H.append(hist); F.append(fut_xy); NB.append(neigh); Y.append(int(r["label"]))
            META.append({"scene": scene, "ped_id": pid, "frame_current": tau,
                         "origin": [float(x0), float(y0)],
                         "future_frames": need[TH:], "future_times": fut_times})
        print(f"  {scene}: kept {sum(1 for m in META if m['scene']==scene)}")

    payload = {
        "hist": torch.tensor(np.array(H), dtype=torch.float32),
        "fut": torch.tensor(np.array(F), dtype=torch.float32),
        "neigh": torch.tensor(np.array(NB), dtype=torch.float32),
        "y": torch.tensor(np.array(Y), dtype=torch.long),
        "meta": META, "dataset": dataset, "split": split,
        "Th": TH, "Tf": TF, "fps": FPS,
    }
    torch.save(payload, out)
    print(f"[{dataset}/{split}] wrote {out}  N={len(Y)} pos={int(np.sum(Y))}")
    return out


def fr2xy_other(peds: pd.DataFrame, oid: int, fr: int):
    g = peds[(peds["id"] == oid) & (peds["frame"] == fr)]
    if len(g):
        return float(g.iloc[0]["cx_m"]), float(g.iloc[0]["cy_m"])
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for sp in a.splits:
        build_cache(a.dataset, sp, force=a.force)
