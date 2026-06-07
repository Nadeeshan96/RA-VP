"""Score and predict on SMGS samples.

The core call is `evaluate_sample(sample, cfg) -> SMGSResult`. `evaluate_batch`
is a thin loop for the CLI driver. Score normalization (raw / sum_weights) is
done here so downstream threshold tuning can use a fixed [0, 1] range.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .config import RULE_NAMES, SMGSConfig
from .rules import RULE_FUNCS
from .types import SMGSResult, SMGSSample


def score_from_flags(flags: Dict[str, int], cfg: SMGSConfig) -> Tuple[float, float]:
    """Return (raw_score, normalized_score) from per-rule 0/1 flags."""
    raw = sum(cfg.weight_for(r) * int(flags.get(r, 0)) for r in RULE_NAMES)
    denom = cfg.sum_weights()
    norm = raw / denom if denom > 0 else 0.0
    return float(raw), float(norm)


def evaluate_sample(sample: SMGSSample, cfg: SMGSConfig,
                    eligibility_ok: bool = True) -> SMGSResult:
    """Apply every rule, accumulate the weighted score, emit diagnostics.

    ``eligibility_ok`` lets the caller mark a sample ineligible without
    running the rules (e.g. a ped that's already in the prohibited area at
    tau, which should be filtered upstream). Ineligible samples get pred=0
    and score=0 so they never contribute spurious positives.
    """
    if not eligibility_ok:
        return SMGSResult(eligible=False, rule_flags={r: 0 for r in RULE_NAMES})

    flags: Dict[str, int] = {}
    diagnostics: Dict[str, float] = {}
    for r in RULE_NAMES:
        flag, d = RULE_FUNCS[r](sample, cfg)
        flags[r] = int(flag)
        diagnostics.update(d)

    raw, norm = score_from_flags(flags, cfg)
    pred = 1 if norm >= cfg.theta else 0
    return SMGSResult(
        eligible=True,
        rule_flags=flags,
        score_raw=raw,
        score_norm=norm,
        pred=pred,
        diagnostics=diagnostics,
    )


def evaluate_batch(samples: Iterable[SMGSSample], cfg: SMGSConfig
                   ) -> List[SMGSResult]:
    return [evaluate_sample(s, cfg) for s in samples]
