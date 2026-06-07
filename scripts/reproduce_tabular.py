#!/usr/bin/env python3
"""Reproduce the tabular baselines (Logistic Regression / Random Forest / XGBoost),
all-six pooled test (live).

Loads the engineered tabular feature caches (assets/weights/tabular_cache/
cache_<DS>/{train,val,test}_tab.pt), restricts to the portable feature set shared
by all intersections, fits each classifier on the pooled train split, tunes the
threshold on the pooled val split, and evaluates on the pooled test split.

Usage:  python scripts/reproduce_tabular.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid macOS libomp clash

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "tabular"))

from ravp.paths import WEIGHTS_DIR
import train_eval  # ra-vp/baselines/tabular/train_eval.py

ALL6 = ["FI", "FIDRT", "SInDTianjin", "SInDChongqing", "SInDXian", "SInDChangchun"]
PAPER = {"logreg": (66.3, 36.5, 37.2), "rf": (82.4, 58.4, 74.8), "xgb": (82.3, 58.8, 74.2)}
NAME = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}


def main():
    if not (WEIGHTS_DIR / "tabular_cache").exists():
        sys.exit("missing assets/weights/tabular_cache (run: python download_assets.py)")
    out_root = Path(tempfile.mkdtemp(prefix="ravp_tabular_"))
    paths = train_eval.run(ALL6, ALL6, "all6", out_root=out_root)
    print("\n=== Tabular baselines, all-six pooled (test) ===")
    print(f"{'Model':22s}{'Macro-F1':>10s}{'V-Recall':>10s}{'PR-AUC':>9s}")
    for key, p in paths.items():
        rec = json.loads(Path(p).read_text())
        te = [s for s in rec["splits"] if s["split"] == "test"][0]
        mf1, rv, pr = te["macro_f1"] * 100, te["recall_jaywalk"] * 100, te["pr_auc"] * 100
        pm, pv, pp = PAPER[key]
        print(f"{NAME[key]:22s}{mf1:10.1f}{rv:10.1f}{pr:9.1f}   (paper {pm}/{pv}/{pp})")


if __name__ == "__main__":
    main()
