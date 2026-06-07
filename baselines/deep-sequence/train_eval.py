#!/usr/bin/env python3
"""Multi-domain sequence baselines (LSTM / GRU / TCN).

Reads the per-dataset v4 sequence caches ({split}_seq_<fs>.pt with dyn [N,T,D],
static [N,Ds], y [N]) built by `cli.py precompute-seq`, pools the training
datasets, z-score normalises (train stats, NaN->0, clip +/-5), trains with
BCEWithLogitsLoss + sqrt_ratio pos_weight, early-stops on val Macro-F1 with a
per-epoch threshold sweep, then evaluates the eval datasets' test split.
Same {"splits": [...]} metrics shape as the SMI-VP trainer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.anticipatory_classifier.common.metrics import (
    compute_split_metrics, sweep_threshold, print_split_report,
)
from models.anticipatory_classifier.versions.v4_sequence_classifiers.models import (
    build_model,
)

_REPO = Path(__file__).resolve().parents[4]
CACHE_ROOT = _REPO / "logs" / "anticipatory_classifier" / "v12_sequence_paper"
MODEL_KEYS = ("lstm", "gru", "tcn")
FEATURE_SET = "seq_core_ctx"


def _cache_path(dataset: str, split: str) -> Path:
    return CACHE_ROOT / f"cache_{dataset}" / f"{split}_seq_{FEATURE_SET}.pt"


def load_pooled(datasets: List[str], split: str):
    dyn, st, y = [], [], []
    for d in datasets:
        c = torch.load(str(_cache_path(d, split)), map_location="cpu", weights_only=False)
        dyn.append(c["dyn"].float())
        st.append(c["static"].float())
        y.append(c["y"].long())
    return torch.cat(dyn), torch.cat(st), torch.cat(y)


class Standardizer:
    def __init__(self, dyn: torch.Tensor, st: torch.Tensor):
        d = dyn.reshape(-1, dyn.shape[-1]).numpy()
        self.dmean = np.nanmean(d, 0); self.dstd = np.nanstd(d, 0) + 1e-6
        s = st.numpy()
        self.smean = np.nanmean(s, 0); self.sstd = np.nanstd(s, 0) + 1e-6

    def apply(self, dyn: torch.Tensor, st: torch.Tensor):
        dn = (dyn.numpy() - self.dmean) / self.dstd
        sn = (st.numpy() - self.smean) / self.sstd
        dn = np.clip(np.nan_to_num(dn, nan=0.0), -5, 5)
        sn = np.clip(np.nan_to_num(sn, nan=0.0), -5, 5)
        return torch.from_numpy(dn).float(), torch.from_numpy(sn).float()


def _loader(dyn, st, y, bs, shuffle):
    return DataLoader(TensorDataset(dyn, st, y), batch_size=bs, shuffle=shuffle)


def _infer(model, dyn, st, dev, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(dyn), bs):
            d = dyn[i:i+bs].to(dev); s = st[i:i+bs].to(dev)
            mask = torch.ones(d.shape[0], d.shape[1], device=dev)
            logits = model(d, s, mask).squeeze(-1)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_one(key, dyn_tr, st_tr, y_tr, dyn_va, st_va, y_va, dev,
              epochs, patience, bs, lr):
    d_in = dyn_tr.shape[-1]; d_static = st_tr.shape[-1]
    model = build_model(key, d_in, d_static, {}).to(dev)
    pos = max(int(y_tr.sum()), 1); neg = len(y_tr) - int(y_tr.sum())
    pos_weight = torch.tensor([np.sqrt(neg / pos)], dtype=torch.float32, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = _loader(dyn_tr, st_tr, y_tr, bs, True)

    best_macro, best_state, best_thr, best_ep, wait = -1.0, None, 0.5, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for d, s, yb in dl:
            d, s, yb = d.to(dev), s.to(dev), yb.float().to(dev)
            mask = torch.ones(d.shape[0], d.shape[1], device=dev)
            opt.zero_grad()
            logits = model(d, s, mask).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
        pva = _infer(model, dyn_va, st_va, dev)
        thr, macro = sweep_threshold(pva, y_va.numpy())
        if macro > best_macro:
            best_macro, best_thr, best_ep, wait = macro, thr, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    print(f"[{key}] best_val_macroF1={best_macro:.4f} @ep{best_ep} thr={best_thr:.3f}")
    return model, best_thr


def run(train_datasets: List[str], eval_datasets: List[str], tag: str,
        out_root: Path, dev_str: str = "mps", models: List[str] = None,
        epochs: int = 60, patience: int = 10, bs: int = 256, lr: float = 1e-3):
    models = models or list(MODEL_KEYS)
    dev = torch.device(dev_str)
    torch.manual_seed(42); np.random.seed(42)

    dyn_tr_r, st_tr_r, y_tr = load_pooled(train_datasets, "train")
    std = Standardizer(dyn_tr_r, st_tr_r)
    dyn_tr, st_tr = std.apply(dyn_tr_r, st_tr_r)
    dyn_va_r, st_va_r, y_va_tr = load_pooled(train_datasets, "val")
    dyn_va, st_va = std.apply(dyn_va_r, st_va_r)

    ts = time.strftime("%Y-%m-%d_%H%M%S")
    out_paths = {}
    for key in models:
        t0 = time.time()
        model, thr = train_one(key, dyn_tr, st_tr, y_tr, dyn_va, st_va, y_va_tr,
                               dev, epochs, patience, bs, lr)
        splits_out = []
        for split in ("val", "test"):
            dyn_s_r, st_s_r, ys = load_pooled(eval_datasets, split)
            dyn_s, st_s = std.apply(dyn_s_r, st_s_r)
            probs = _infer(model, dyn_s, st_s, dev)
            m = compute_split_metrics(split=split, probs=probs, y_true=ys.numpy(),
                                      threshold=thr, windows_total=len(ys),
                                      skipped_no_hist=0)
            print_split_report(m)
            splits_out.append(m.to_dict())
        run_dir = out_root / f"{ts}_{tag}_{key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "method": key, "model": key, "tag": tag,
            "train_datasets": train_datasets, "eval_datasets": eval_datasets,
            "theta": float(thr), "device": str(dev),
            "elapsed_seconds": round(time.time() - t0, 1),
            "splits": splits_out,
        }
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        out_paths[key] = run_dir / "metrics.json"
        print(f"[{key}] wrote {out_paths[key]} ({time.time()-t0:.1f}s)")
    return out_paths
