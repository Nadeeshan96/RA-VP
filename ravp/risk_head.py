#!/usr/bin/env python3
"""RiskHead — small MLP violation classifier on top of frozen-forecaster risk
features. Two scoring modes:

  replacement : final_logit = head_logit
  residual    : final_logit = checker_logit + head_logit   (preferred default;
                preserves the strong checker and learns a correction)

checker_logit = logit(clip(p_check_mean, eps, 1-eps)).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score
from ravp.metrics import sweep_threshold, compute_split_metrics

EPS = 1e-4


def select_threshold(probs, y, objective="macro_f1", beta=2.0,
                     grid=(0.01, 0.99, 0.01)):
    """Pick a decision threshold. macro_f1 = balanced; fbeta (beta>1) = recall-
    oriented (maximises F_beta of the violation class)."""
    if objective == "macro_f1":
        return sweep_threshold(probs, y)[0]
    y = np.asarray(y)
    best_t, best = float(grid[0]), -1.0
    t = grid[0]
    while t <= grid[1] + 1e-9:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        b2 = beta * beta
        fb = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) else 0.0
        if fb > best:
            best, best_t = fb, t
        t += grid[2]
    return float(best_t)


def checker_logit(p_check_mean: np.ndarray) -> np.ndarray:
    p = np.clip(p_check_mean, EPS, 1 - EPS)
    return np.log(p / (1 - p)).astype(np.float32)


class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mean = np.nanmean(X, 0); self.std = np.nanstd(X, 0) + 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.clip(np.nan_to_num(Z, nan=0.0), -5, 5).astype(np.float32)


class RiskHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _final_logit(head_logit, clogit, mode):
    return head_logit + clogit if mode == "residual" else head_logit


def train_risk_head(Xtr, ytr, clog_tr, Xva, yva, clog_va, mode="residual",
                    device="cpu", epochs=200, patience=20, lr=1e-3,
                    pos_cap=10.0, focal=False, threshold_objective="macro_f1",
                    beta=2.0, early_stop_metric="macro_f1") -> Tuple[RiskHead, Standardizer, float]:
    dev = torch.device(device)
    torch.manual_seed(42); np.random.seed(42)
    std = Standardizer(Xtr)
    Xtr_n = torch.tensor(std(Xtr), device=dev)
    Xva_n = torch.tensor(std(Xva), device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    clog_tr_t = torch.tensor(clog_tr, dtype=torch.float32, device=dev)
    clog_va_t = torch.tensor(clog_va, dtype=torch.float32, device=dev)

    pos = max(int(ytr.sum()), 1); neg = len(ytr) - int(ytr.sum())
    pw = float(min(np.sqrt(neg / pos), pos_cap))
    model = RiskHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))

    def loss_fn(logits, y):
        if not focal:
            return bce(logits, y)
        p = torch.sigmoid(logits); g = 2.0
        w = (1 - p) ** g * y + p ** g * (1 - y)
        return (w * nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor(pw, device=dev), reduction="none")).mean()

    best_mf1, best_state, best_thr, wait = -1.0, None, 0.5, 0
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        fl = _final_logit(model(Xtr_n), clog_tr_t, mode)
        loss = loss_fn(fl, ytr_t)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(_final_logit(model(Xva_n), clog_va_t, mode)).cpu().numpy()
        thr, mf1 = sweep_threshold(pva, yva)
        score = mf1 if early_stop_metric == "macro_f1" else float(average_precision_score(yva, pva))
        if score > best_mf1:
            best_mf1, best_thr, wait = score, thr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    # final operating point: re-select threshold on val by the chosen objective
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(_final_logit(model(Xva_n), clog_va_t, mode)).cpu().numpy()
    final_thr = select_threshold(pva, yva, objective=threshold_objective, beta=beta)
    return model, std, final_thr


@torch.no_grad()
def predict_proba(model, std, X, clog, mode, device="cpu"):
    dev = torch.device(device)
    Xn = torch.tensor(std(X), device=dev)
    clog_t = torch.tensor(clog, dtype=torch.float32, device=dev)
    return torch.sigmoid(_final_logit(model(Xn), clog_t, mode)).cpu().numpy()
