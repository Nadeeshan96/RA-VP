# -*- coding: utf-8 -*-
"""
Shared spatio-temporal jaywalking rules (no OpenCV).

Used by ``demo/visualization/pedestrian_spatial_vis.py`` and analysis scripts.

Logic:
1. If pedestrian (cx_m, cy_m) is in a WalkingArea -> NOT jaywalking.
2. If in a Crosswalk -> Check synchronized signal (pedestrian turn ``p``). If 'G' or 'y' -> NOT jaywalking. If 'r' -> Jaywalking.
3. If not in WalkingArea AND not in Crosswalk (i.e., in the road) -> Jaywalking.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd
import shapely.wkt
from shapely.geometry import Point


def infer_signal_path(traj_csv: Path) -> Path | None:
    """
    Map .../derived_data/traj/<name>_Traj.csv -> .../derived_data/signal/<name>_signal.csv
    """
    traj_csv = traj_csv.resolve()
    if traj_csv.parent.name != "traj":
        return None
    signal_dir = traj_csv.parent.parent / "signal"
    stem = traj_csv.name
    if not stem.endswith("_Traj.csv"):
        return None
    base = stem[: -len("_Traj.csv")]
    candidate = signal_dir / f"{base}_signal.csv"
    return candidate if candidate.is_file() else None


def load_signal_intervals_by_dir_turn(
    signal_path: Path,
) -> dict[tuple[str, str], list[tuple[float, float, str]]]:
    df = pd.read_csv(signal_path)
    need = {"direction", "turn", "begin_time", "end_time", "state"}
    if not need.issubset(df.columns):
        raise ValueError(f"Signal CSV missing columns: {sorted(need - set(df.columns))}")
    by_key: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        d = str(row["direction"]).strip()
        turn = str(row["turn"]).strip().lower()
        b = float(row["begin_time"])
        e = float(row["end_time"])
        st = str(row["state"]).strip()
        by_key[(d, turn)].append((b, e, st))
    for k in by_key:
        by_key[k].sort(key=lambda x: x[0])
    return dict(by_key)


def lookup_state(intervals: list[tuple[float, float, str]], t: float) -> str | None:
    if not intervals:
        return None
    begins = [x[0] for x in intervals]
    i = bisect_right(begins, t) - 1
    if i < 0:
        return None
    b, e, st = intervals[i]
    if b <= t < e:
        return st
    return None


def normalize_state(state: str) -> str:
    s = state.strip()
    if s.upper() == "G":
        return "G"
    if s.lower() == "r":
        return "r"
    if s.lower() == "y":
        return "y"
    return s


def load_spatial_zones(csv_path: Path):
    """Loads the WKT CSV and separates it into walking areas and crosswalks."""
    walking_areas = []
    crosswalks = {}  # Map 'N', 'S', 'E', 'W' to their respective polygons

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        name = str(row["zone_name"])
        poly = shapely.wkt.loads(row["WKT"])

        if "WalkingArea" in name:
            walking_areas.append(poly)
        elif "Crosswalk" in name:
            if "North" in name:
                crosswalks["N"] = poly
            elif "South" in name:
                crosswalks["S"] = poly
            elif "East" in name:
                crosswalks["E"] = poly
            elif "West" in name:
                crosswalks["W"] = poly

    return walking_areas, crosswalks


def check_jaywalk_status(
    pt: Point, t: float, walking_areas: list, crosswalks: dict, sig_by_dir: dict
) -> tuple[bool, str]:
    """
    Evaluates the strict spatio-temporal logic for a pedestrian.
    Returns: (is_jaywalking: bool, status_text: str)
    """
    for wa in walking_areas:
        if wa.contains(pt):
            return False, "Safe (Sidewalk)"

    for direction, cw_poly in crosswalks.items():
        if cw_poly.contains(pt):
            intervals = sig_by_dir.get(direction, [])
            st = lookup_state(intervals, t)

            if st is None:
                return False, f"CW_{direction} (No Sig)"

            n_st = normalize_state(st)
            if n_st in ["G", "y"]:
                return False, f"CW_{direction} ({n_st})"

    return True, "Road Core (Jaywalk)"
