#!/usr/bin/env python3
"""Reproduce RA-VP's all-six pooled test numbers (Macro-F1 / Violation-Recall / PR-AUC).

Pipeline (Stage 2 of RA-VP, using the shipped fine-tuned forecaster):
  load all6_vp4_ft.pt  ->  for each window draw K=16 future samples  ->  run the
  EXACT geofence checker over the samples  ->  build the portable feature vector
  (checker / signal / soft-distance / aggregate groups)  ->  train the residual
  head (focal loss, recall-oriented F_beta=1.5 threshold)  ->  evaluate on test.

Extracted features are cached under assets/features/all6_vp4/ so re-runs are fast.

Requires (via download_assets.py):
  assets/weights/forecaster_ckpt/all6_vp4_ft.pt
  assets/weights/cache_<DS>/{train,val,test}.pt      (window caches)
  assets/data/<DS>/derived_data/{map,signal}/        (zones + signals for the checker)

Usage:
  python scripts/reproduce_ravp.py                 # uses/creates the feature cache
  python scripts/reproduce_ravp.py --from-forecaster   # force re-extraction
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid macOS libomp clash

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ravp.paths import WEIGHTS_DIR, FEATURES_DIR
from ravp.forecaster import load_forecaster, deltas_from_pos
from ravp.risk_features import RiskExtractor, FEATURE_GROUPS, ALL_NAMES
from ravp import risk_head as RH
from ravp.metrics import compute_split_metrics

ALL6 = ["FI", "FIDRT", "SInDTianjin", "SInDChongqing", "SInDXian", "SInDChangchun"]
GROUPS = ["checker", "signal", "softdist", "agg"]   # RA-VP residual-head features
CKPT = WEIGHTS_DIR / "forecaster_ckpt" / "all6_vp4_ft.pt"
FEAT = FEATURES_DIR / "all6_vp4"


def _select(X_all, names):
    cols = [names.index(n) for g in GROUPS for n in FEATURE_GROUPS[g]]
    return X_all[:, cols].astype(np.float32)


@torch.no_grad()
def extract(fc, dataset, split, K, device, force=False):
    """Forecaster sampling + exact checker -> feature matrix. Cached to FEAT/."""
    cache = FEAT / f"{dataset}__{split}__K{K}.pt"
    if cache.exists() and not force:
        return torch.load(str(cache), weights_only=False)
    c = torch.load(str(WEIGHTS_DIR / f"cache_{dataset}" / f"{split}.pt"), weights_only=False)
    hist_d = deltas_from_pos(c["hist"][..., :2].float())
    neigh = c["neigh"].float(); y = c["y"].numpy().astype(np.int64); meta = c["meta"]
    ext = RiskExtractor(dataset)
    N = len(y); bs = 512
    X = np.zeros((N, len(ALL_NAMES)), dtype=np.float32)
    for i in range(0, N, bs):
        a = hist_d[i:i + bs].to(device); b = neigh[i:i + bs].to(device)
        pred = fc.sample(a, b, K=K).cpu().numpy()                  # (B,K,Tf,2)
        for j in range(pred.shape[0]):
            m = meta[i + j]
            fd = ext.extract(pred[j], m["origin"], m["future_times"], m["scene"])
            X[i + j] = [fd[n] for n in ALL_NAMES]
        print(f"  [{dataset}/{split}] {min(i + bs, N)}/{N}", end="\r")
    print()
    out = {"X": X, "names": ALL_NAMES, "y": y,
           "p_check_mean": X[:, ALL_NAMES.index("p_check_mean")].copy()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(cache))
    return out


def pool(fc, split, K, device, force):
    Xs, ys, cl = [], [], []
    for d in ALL6:
        f = extract(fc, d, split, K, device, force)
        Xs.append(_select(f["X"], f["names"])); ys.append(f["y"])
        cl.append(RH.checker_logit(f["p_check_mean"]))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--from-forecaster", action="store_true", help="force feature re-extraction")
    a = ap.parse_args()
    dev = torch.device(a.device); torch.manual_seed(42); np.random.seed(42)
    if not CKPT.exists():
        sys.exit(f"missing {CKPT} (run: python download_assets.py)")
    fc = load_forecaster(CKPT, device=a.device)
    t0 = time.time()
    Xtr, ytr, cltr = pool(fc, "train", a.K, dev, a.from_forecaster)
    Xva, yva, clva = pool(fc, "val", a.K, dev, a.from_forecaster)
    head, std, _ = RH.train_risk_head(Xtr, ytr, cltr, Xva, yva, clva, mode="residual",
                                      device=a.device, focal=True, early_stop_metric="pr_auc")
    pv = RH.predict_proba(head, std, Xva, clva, "residual", a.device)
    thr = RH.select_threshold(pv, yva, objective="fbeta", beta=a.beta)
    Xte, yte, clte = pool(fc, "test", a.K, dev, a.from_forecaster)
    pte = RH.predict_proba(head, std, Xte, clte, "residual", a.device)
    m = compute_split_metrics(split="test", probs=pte, y_true=yte, threshold=thr,
                              windows_total=len(yte), skipped_no_hist=0)
    print("\n=== RA-VP, all-six pooled (test) ===")
    print(f"  Macro-F1      : {m.macro_f1 * 100:.1f}")
    print(f"  Violation Rec : {m.recall_jaywalk * 100:.1f}")
    print(f"  PR-AUC        : {m.pr_auc * 100:.1f}")
    print(f"  (paper: 91.1 / 86.2 / 90.2;  elapsed {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
