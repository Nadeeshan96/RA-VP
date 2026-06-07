"""SMGS rule weights, thresholds, and ablation presets.

All numeric defaults live here so the rules stay data-structure-agnostic and
so ablations are 5-line dict overrides rather than code forks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict


RULE_NAMES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


@dataclass(frozen=True)
class SMGSConfig:
    # ---- weights ---------------------------------------------------------
    w1: float = 3.0   # R1 projected illegal entry
    w2: float = 1.5   # R2 velocity toward illegal
    w3: float = 1.5   # R3 heading prefers illegal over legal
    w4: float = 1.0   # R4 signal / waiting pressure
    w5: float = 1.0   # R5 safe vehicle gap
    w6: float = 1.0   # R6 social following
    w7: float = 0.5   # R7 near-boundary urgency

    # ---- thresholds ------------------------------------------------------
    horizon_s: float = 2.0            # R1 prediction horizon (same as label t2)
    proj_dt_s: float = 0.1            # R1 projection step
    v_toward_illegal_thresh: float = 0.3   # m/s; R2
    heading_margin: float = 0.1       # R3 cos-alignment margin
    wait_time_thresh: float = 12.0    # seconds; R4
    safe_ttc_thresh: float = 3.5      # seconds; R5
    social_radius: float = 4.0        # metres; R6
    near_boundary_dist: float = 1.0   # metres; R7

    # ---- scan radii for nearby agents (used by context builder) ----------
    ped_context_radius: float = 10.0  # metres; cap for nearby peds list
    veh_context_radius: float = 25.0  # metres; cap for nearby vehicles list

    # ---- default decision threshold (normalized score) -------------------
    theta: float = 4.0 / 9.5   # default raw theta 4.0 / sum(weights) = 9.5

    def weight_for(self, rule: str) -> float:
        return getattr(self, "w" + rule[1:])

    def sum_weights(self) -> float:
        return sum(getattr(self, f"w{i}") for i in range(1, 8))

    def to_dict(self) -> Dict:
        return asdict(self)


# -------------------------------------------------------------------------
# Ablation presets: zero out the disabled rule weight; threshold stays
# default — the driver re-tunes theta on val for each ablation.
# -------------------------------------------------------------------------

def _replace_weights(**kw) -> SMGSConfig:
    base = SMGSConfig().to_dict()
    base.update(kw)
    return SMGSConfig(**base)


ABLATIONS: Dict[str, SMGSConfig] = {
    "FULL":        SMGSConfig(),
    "R1_ONLY":     _replace_weights(w2=0.0, w3=0.0, w4=0.0, w5=0.0, w6=0.0, w7=0.0),
    "NO_SIGNAL":   _replace_weights(w4=0.0),
    "NO_GAP":      _replace_weights(w5=0.0),
    "NO_SOCIAL":   _replace_weights(w6=0.0),
    "NO_HEADING":  _replace_weights(w3=0.0),
}
