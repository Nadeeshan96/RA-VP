"""Helper: print a method's precomputed test metrics from results/runs/."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
INTERS = ["FI", "FIDRT", "Tianjin", "Chongqing", "Xian", "Changchun"]
POOLED = ["all6", "sind4", "fi_fidrt"]


def _test(method, kind, key):
    p = RUNS / f"{method}__{kind}__{key}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    te = [s for s in rec["splits"] if s["split"] == "test"]
    return te[0] if te else None


def show(method: str, display: str, retrain_note: str):
    print(f"=== {display}: precomputed test metrics (Macro-F1 / V-Recall / PR-AUC, %) ===")
    print(f"{'Scenario':18s}{'Macro-F1':>10s}{'V-Recall':>10s}{'PR-AUC':>9s}")

    def row(label, kind, key):
        m = _test(method, kind, key)
        if m is None:
            print(f"{label:18s}{'--':>10s}{'--':>10s}{'--':>9s}"); return
        print(f"{label:18s}{m['macro_f1']*100:10.1f}{m['recall_jaywalk']*100:10.1f}{m['pr_auc']*100:9.1f}")

    for k in INTERS:
        row(f"in-domain {k}", "indomain", k)
    for k in POOLED:
        row(f"pooled {k}", "pooled", k)
    for k in INTERS:
        row(f"LOO {k}", "loo", k)
    print(f"\nNote: these numbers are provided precomputed. {retrain_note}")
