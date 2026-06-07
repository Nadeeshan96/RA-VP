#!/usr/bin/env python3
"""Multi-domain tabular baselines (LogReg / RF / XGBoost).

Reads the per-dataset v2 tabular caches ({split}_tab.pt, key "x") built by
`cli.py precompute-tabular`, pools the requested training datasets, fits each
model, tunes a Macro-F1 threshold on the pooled training-side val split, and
evaluates on the eval datasets' test split. Same {"splits": [...]} metrics
shape as the SMI-VP trainer.

Imbalance handling: class_weight='balanced' (LR/RF) and scale_pos_weight=neg/pos
(XGB). Decision threshold tuned on val (mirrors SMI-VP).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ravp.metrics import (
    compute_split_metrics, sweep_threshold, print_split_report,
)
from ravp.paths import WEIGHTS_DIR

CACHE_ROOT = WEIGHTS_DIR / "tabular_cache"
MODEL_KEYS = ("logreg", "rf", "xgb")


def _cache_path(dataset: str, split: str) -> Path:
    return CACHE_ROOT / f"cache_{dataset}" / f"{split}_tab.pt"


def _all_cached_datasets() -> List[str]:
    return sorted(p.name[len("cache_"):] for p in CACHE_ROOT.glob("cache_*")
                  if (p / "train_tab.pt").exists())


def global_feature_names() -> List[str]:
    """Intersection of tabular feature names across ALL cached datasets, in the
    order of the first dataset. The v2 tabular feature set includes a
    dataset-specific phase-tuple one-hot whose width varies per intersection;
    restricting to the shared (kinematic/map/gap/social) columns gives a single
    portable feature set usable in-domain, pooled, and leave-one-out."""
    order, sets = None, []
    for d in _all_cached_datasets():
        c = torch.load(str(_cache_path(d, "train")), map_location="cpu", weights_only=False)
        fn = list(c["feature_names"])
        if order is None:
            order = fn
        sets.append(set(fn))
    return [n for n in order if all(n in s for s in sets)]


def load_pooled(datasets: List[str], split: str,
                feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for d in datasets:
        c = torch.load(str(_cache_path(d, split)), map_location="cpu", weights_only=False)
        names = list(c["feature_names"])
        idx = [names.index(n) for n in feature_names]
        xs.append(c["x"].numpy().astype(np.float32)[:, idx])
        ys.append(c["y"].numpy().astype(np.int64))
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def build_model(key: str, pos_weight: float):
    if key == "logreg":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ])
    if key == "rf":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
        ])
    if key == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=pos_weight, tree_method="hist",
            eval_metric="logloss", n_jobs=-1, random_state=42)
    raise ValueError(key)


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def run(train_datasets: List[str], eval_datasets: List[str], tag: str,
        out_root: Path, models: List[str] = None) -> Dict[str, Path]:
    models = models or list(MODEL_KEYS)
    feat = global_feature_names()
    Xtr, ytr = load_pooled(train_datasets, "train", feat)
    Xva_t, yva_t = load_pooled(train_datasets, "val", feat)   # threshold-tuning split
    pos = max(int(ytr.sum()), 1)
    neg = len(ytr) - int(ytr.sum())
    pos_weight = neg / pos

    ts = time.strftime("%Y-%m-%d_%H%M%S")
    out_paths: Dict[str, Path] = {}
    for key in models:
        t0 = time.time()
        model = build_model(key, pos_weight)
        model.fit(Xtr, ytr)
        theta, best = sweep_threshold(_predict_proba(model, Xva_t), yva_t)
        print(f"[{key}] train={len(ytr)} pos={pos} theta={theta:.3f} "
              f"(val macroF1={best:.4f}) fit={time.time()-t0:.1f}s")

        splits_out = []
        for split in ("val", "test"):
            Xs, ys = load_pooled(eval_datasets, split, feat)
            if len(ys) == 0:
                continue
            m = compute_split_metrics(split=split, probs=_predict_proba(model, Xs),
                                      y_true=ys, threshold=theta,
                                      windows_total=len(ys), skipped_no_hist=0)
            print_split_report(m)
            splits_out.append(m.to_dict())

        run_dir = out_root / f"{ts}_{tag}_{key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "method": key, "model": key, "tag": tag,
            "train_datasets": train_datasets, "eval_datasets": eval_datasets,
            "pos_weight": pos_weight, "theta": float(theta),
            "splits": splits_out,
        }
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        out_paths[key] = run_dir / "metrics.json"
        print(f"[{key}] wrote {out_paths[key]}")
    return out_paths
