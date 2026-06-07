"""Immutable data structures describing a single SMGS evaluation sample.

A SMGSSample bundles everything the rule set needs to reach a prediction for
one pedestrian at one time tau. The builder in context.py is responsible for
producing these from a ground-truth row + scene trajectory + map zones;
rules.py only consumes SMGSSample, never raw dataframes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PedState:
    scene: str
    ped_id: int
    frame: int
    time_s: float          # frame / fps (seconds, same clock as signal CSV)
    x: float               # metres, world frame
    y: float
    vx: float              # m/s
    vy: float
    speed: float           # m/s (== hypot(vx, vy))
    heading: Optional[float]  # radians; None if speed ~= 0
    waiting_time_s: float  # consecutive seconds near-stationary in obs history


@dataclass(frozen=True)
class MapInfo:
    # Raw zone polys (shapely geometries); kept by reference — do not mutate.
    walking_areas: Tuple[Any, ...]
    crosswalks: Dict[str, Any]            # {'N','S','E','W'} -> polygon (not all always present)
    crosswalk_centroids: Dict[str, Tuple[float, float]]
    # Legal / prohibited regions resolved for this sample's signal state.
    # legal_region = walking_areas ∪ crosswalks-currently-green-or-yellow
    legal_region: Any                     # shapely (Multi)Polygon or None
    legal_boundary: Any                   # legal_region.boundary or None


@dataclass(frozen=True)
class SignalInfo:
    available: bool
    ped_signal: Dict[str, str]            # {'N','S','E','W'} -> 'G'/'y'/'r'/'?'
    legal_crosswalks: Tuple[str, ...]     # directions with 'G' or 'y' right now
    legal_now: bool                       # any adjacent-crosswalk ped signal green or yellow


@dataclass(frozen=True)
class NearbyPed:
    dx: float
    dy: float
    dist: float
    vx: float                 # m/s
    vy: float
    inside_prohibited: bool   # at frame tau, w.r.t. current legal_region


@dataclass(frozen=True)
class NearbyVehicle:
    dx: float
    dy: float
    dist: float
    vx: float                 # m/s
    vy: float
    speed: float              # m/s
    ttc: float                # seconds until closest approach; inf if diverging


@dataclass(frozen=True)
class SMGSSample:
    target: PedState
    map_info: MapInfo
    signal_info: SignalInfo
    nearby_peds: Tuple[NearbyPed, ...]
    nearby_vehicles: Tuple[NearbyVehicle, ...]
    # Optional passthroughs used by diagnostics / debugging.
    label: Optional[int] = None           # ground-truth label (0/1) or None


@dataclass
class SMGSResult:
    eligible: bool
    rule_flags: Dict[str, int] = field(default_factory=dict)       # R1..R7 -> 0/1
    score_raw: float = 0.0                 # weighted sum
    score_norm: float = 0.0                # score_raw / sum(weights) in [0, 1]
    pred: int = 0                          # 0/1 after theta
    diagnostics: Dict[str, float] = field(default_factory=dict)
