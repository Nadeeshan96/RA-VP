#!/usr/bin/env python3
"""Reproduce the SMGS rule baseline, all-six pooled test (live).

SMGS is parameter-free apart from one decision threshold. We load the cached
per-window rule scores (assets/weights/smgs_scores/<DS>.npz: scores, label,
split), pool the six datasets, tune the threshold on the pooled val split
(Macro-F1-optimal), and evaluate on the pooled test split.

Usage:  python scripts/reproduce_smgs.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid macOS libomp clash

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ravp.paths import WEIGHTS_DIR
from ravp.metrics import sweep_threshold, compute_split_metrics

ALL6 = ["FI", "FIDRT", "SInDTianjin", "SInDChongqing", "SInDXian", "SInDChangchun"]
SCORES = WEIGHTS_DIR / "smgs_scores"


def pool(split):
    s, y = [], []
    for d in ALL6:
        z = np.load(SCORES / f"{d}.npz", allow_pickle=True)
        mask = z["split"] == split
        s.append(z["scores"][mask]); y.append(z["y"][mask])
    return np.concatenate(s), np.concatenate(y)


def main():
    if not SCORES.exists():
        sys.exit(f"missing {SCORES} (run: python download_assets.py)")
    s_val, y_val = pool("val")
    theta, _ = sweep_threshold(s_val, y_val)
    s_te, y_te = pool("test")
    m = compute_split_metrics(split="test", probs=s_te, y_true=y_te, threshold=theta,
                              windows_total=len(y_te), skipped_no_hist=0)
    print("=== SMGS rule, all-six pooled (test) ===")
    print(f"  Macro-F1      : {m.macro_f1 * 100:.1f}")
    print(f"  Violation Rec : {m.recall_jaywalk * 100:.1f}")
    print(f"  PR-AUC        : {m.pr_auc * 100:.1f}")
    print(f"  (paper: 79.4 / 54.8 / 51.9; theta={theta:.3f})")


if __name__ == "__main__":
    main()
