"""Dataset + normalisation helpers for v4 sequence classifiers.

The cache files are small (~26 MB per split). We load the full tensor into
memory and expose a standard torch Dataset. A mask tensor is carried per
sample so encoders can length-mask — but since the rolling-window generator
only emits full 20-frame histories, mask is all-ones in practice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureStats:
    """Per-column mean/std for dyn and static features, learned on train."""
    dyn_mean: torch.Tensor      # (D_dyn,)
    dyn_std: torch.Tensor       # (D_dyn,)
    static_mean: torch.Tensor   # (D_static,)
    static_std: torch.Tensor    # (D_static,)

    def to_dict(self) -> Dict:
        return {
            "dyn_mean": self.dyn_mean.tolist(),
            "dyn_std": self.dyn_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureStats":
        return cls(
            dyn_mean=torch.tensor(d["dyn_mean"], dtype=torch.float32),
            dyn_std=torch.tensor(d["dyn_std"], dtype=torch.float32),
            static_mean=torch.tensor(d["static_mean"], dtype=torch.float32),
            static_std=torch.tensor(d["static_std"], dtype=torch.float32),
        )


def compute_stats(dyn: torch.Tensor, static: torch.Tensor,
                  eps: float = 1e-4) -> FeatureStats:
    """Learn per-feature mean/std over the train split.

    NaNs are replaced with 0 before stats computation (they'll be re-imputed
    with the learned mean at normalise time).
    """
    dyn_f = torch.where(torch.isnan(dyn), torch.zeros_like(dyn), dyn)
    stat_f = torch.where(torch.isnan(static), torch.zeros_like(static), static)

    # Flatten the time axis for dyn stats: treat every (sample, t) as a row.
    dyn_flat = dyn_f.reshape(-1, dyn_f.shape[-1])
    return FeatureStats(
        dyn_mean=dyn_flat.mean(dim=0),
        dyn_std=dyn_flat.std(dim=0).clamp(min=eps),
        static_mean=stat_f.mean(dim=0),
        static_std=stat_f.std(dim=0).clamp(min=eps),
    )


def normalise(dyn: torch.Tensor, static: torch.Tensor,
              stats: FeatureStats) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the learned z-score. NaNs map to 0 post-centering."""
    dyn_z = (dyn - stats.dyn_mean) / stats.dyn_std
    stat_z = (static - stats.static_mean) / stats.static_std
    dyn_z = torch.where(torch.isnan(dyn_z), torch.zeros_like(dyn_z), dyn_z)
    stat_z = torch.where(torch.isnan(stat_z), torch.zeros_like(stat_z), stat_z)
    return dyn_z, stat_z


# ---------------------------------------------------------------------------


class SequenceWindowDataset(Dataset):
    def __init__(self, dyn: torch.Tensor, static: torch.Tensor,
                 y: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert dyn.shape[0] == static.shape[0] == y.shape[0], \
            f"mismatched first dim: dyn={dyn.shape} static={static.shape} y={y.shape}"
        self.dyn = dyn.float()
        self.static = static.float()
        self.y = y.float()
        if mask is None:
            mask = torch.ones(dyn.shape[:2], dtype=torch.float32)
        self.mask = mask.float()

    def __len__(self) -> int:
        return int(self.dyn.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "dyn": self.dyn[idx],
            "static": self.static[idx],
            "mask": self.mask[idx],
            "y": self.y[idx],
        }


def load_cached_split(cache_dir: Path, split: str, feature_set: str) -> Dict:
    path = Path(cache_dir) / f"{split}_seq_{feature_set}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run "
            f"'python -m models.anticipatory_classifier.cli precompute-seq "
            f"--splits {split} --feature-set {feature_set}' first."
        )
    return torch.load(str(path))


def build_datasets(cache_dir: Path, feature_set: str,
                   splits: List[str] = ("train", "val", "test"),
                   ) -> Tuple[Dict[str, SequenceWindowDataset], FeatureStats, Dict]:
    """Load caches, fit stats on train, return normalised datasets + metadata."""
    caches = {s: load_cached_split(cache_dir, s, feature_set) for s in splits}

    stats = compute_stats(caches["train"]["dyn"], caches["train"]["static"])

    datasets: Dict[str, SequenceWindowDataset] = {}
    for s in splits:
        dyn_z, stat_z = normalise(caches[s]["dyn"], caches[s]["static"], stats)
        datasets[s] = SequenceWindowDataset(
            dyn=dyn_z, static=stat_z, y=caches[s]["y"],
        )

    # Carry through feature metadata from train cache.
    meta = {
        "dyn_names": caches["train"]["dyn_names"],
        "dyn_groups": caches["train"]["dyn_groups"],
        "static_names": caches["train"]["static_names"],
        "feature_set": caches["train"]["feature_set"],
        "signal_missing_scenes": {s: caches[s].get("signal_missing_scenes", [])
                                  for s in splits},
        "windows_total": {s: caches[s].get("windows_total",
                                           int(caches[s]["y"].shape[0]))
                          for s in splits},
        "skipped_no_hist": {s: caches[s].get("skipped_no_hist", 0) for s in splits},
        "meta": {s: caches[s].get("meta", []) for s in splits},
    }
    return datasets, stats, meta
