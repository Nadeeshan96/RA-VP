"""Metrics + threshold sweep shared across versions."""
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class SplitMetrics:
    split: str
    windows_total: int
    windows_valid: int
    jaywalk_count: int
    skipped_no_hist: int
    # class 0 (non-violation / compliant)
    precision_nonviolation: float
    recall_nonviolation: float
    f1_nonviolation: float
    # class 1 (violation / jaywalk)
    precision_jaywalk: float
    recall_jaywalk: float
    f1_jaywalk: float
    # ranking + summary
    macro_f1: float
    roc_auc: float
    pr_auc: float
    threshold: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict:
        return asdict(self)


def sweep_threshold(probs: np.ndarray, y_true: np.ndarray,
                    grid: Tuple[float, float, float] = (0.05, 0.95, 0.01)
                    ) -> Tuple[float, float]:
    """Return (best_threshold, best_macro_f1) by grid search for Macro-F1."""
    lo, hi, step = grid
    thresholds = np.arange(lo, hi + step / 2, step)
    best_t, best_f1 = float(thresholds[0]), -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def compute_split_metrics(split: str,
                          probs: np.ndarray,
                          y_true: np.ndarray,
                          threshold: float,
                          windows_total: int,
                          skipped_no_hist: int) -> SplitMetrics:
    """Compute the full metric bundle for one split at a fixed threshold."""
    # Cast to int so classification_report's output_dict keys are "0" / "1",
    # not "0.0" / "1.0" (which would break the per-class lookups below).
    y_true = np.asarray(y_true).astype(int)
    y_pred = (probs >= threshold).astype(int)
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    nv = cr.get("0", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    j = cr.get("1", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except ValueError:
        pr_auc = 0.0
    try:
        # roc_auc_score requires both classes present in y_true.
        roc_auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        roc_auc = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return SplitMetrics(
        split=split,
        windows_total=int(windows_total),
        windows_valid=int(len(y_true)),
        jaywalk_count=int(y_true.sum()),
        skipped_no_hist=int(skipped_no_hist),
        precision_nonviolation=round(float(nv["precision"]), 4),
        recall_nonviolation=round(float(nv["recall"]), 4),
        f1_nonviolation=round(float(nv["f1-score"]), 4),
        precision_jaywalk=round(float(j["precision"]), 4),
        recall_jaywalk=round(float(j["recall"]), 4),
        f1_jaywalk=round(float(j["f1-score"]), 4),
        macro_f1=round(float(macro_f1), 4),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        threshold=round(float(threshold), 4),
        confusion_matrix=cm,
    )


def print_split_report(m: SplitMetrics) -> None:
    print()
    print("=" * 65)
    print(f"ANTICIPATORY CLASSIFIER  —  {m.split.upper()}")
    print("=" * 65)
    print(f"  Windows  : {m.windows_valid}/{m.windows_total} valid  "
          f"({m.jaywalk_count} jaywalk / {m.windows_valid - m.jaywalk_count} compliant)")
    if m.skipped_no_hist:
        print(f"  Skipped  : {m.skipped_no_hist} insufficient-history")
    print(f"  Threshold: {m.threshold:.4f}")
    print(f"  Non-viol : P={m.precision_nonviolation:.4f}  R={m.recall_nonviolation:.4f}  F1={m.f1_nonviolation:.4f}")
    print(f"  Jaywalk  : P={m.precision_jaywalk:.4f}  R={m.recall_jaywalk:.4f}  F1={m.f1_jaywalk:.4f}")
    print(f"  Macro-F1 : {m.macro_f1:.4f}   ROC-AUC: {m.roc_auc:.4f}   PR-AUC: {m.pr_auc:.4f}")
    cm = m.confusion_matrix
    print(f"  Confusion: [[{cm[0][0]:d},{cm[0][1]:d}],[{cm[1][0]:d},{cm[1][1]:d}]]")
    print("=" * 65)
