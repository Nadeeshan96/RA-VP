"""SMGS rule baseline — Signal-, Map-, Gap-, Social-aware handcrafted rules
for anticipatory jaywalking prediction on FLUID-style rolling windows."""
from .config import (
    SMGSConfig,
    ABLATIONS,
    RULE_NAMES,
)
from .types import (
    PedState,
    MapInfo,
    SignalInfo,
    NearbyPed,
    NearbyVehicle,
    SMGSSample,
    SMGSResult,
)
from .smgs import evaluate_sample, evaluate_batch, score_from_flags

__all__ = [
    "SMGSConfig",
    "ABLATIONS",
    "RULE_NAMES",
    "PedState",
    "MapInfo",
    "SignalInfo",
    "NearbyPed",
    "NearbyVehicle",
    "SMGSSample",
    "SMGSResult",
    "evaluate_sample",
    "evaluate_batch",
    "score_from_flags",
]
