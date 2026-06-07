#!/usr/bin/env python3
"""Render the three main paper tables (Violation Recall, PR-AUC, Macro-F1) from
the shipped ``results/cells.json``. This reproduces the paper tables exactly and
needs NO downloaded assets.

Each table has 15 scenario columns (6 in-domain, 3 pooled, 6 leave-one-out) for
the 10 methods. Values are shown as percentages.

Usage:
  python scripts/render_tables.py            # print + write results/expected_main_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "results" / "cells.json"
OUT = ROOT / "results" / "expected_main_tables.md"

METHODS = [
    ("smgs", "SMGS Rule"),
    ("logreg", "Logistic Reg."),
    ("rf", "Random Forest"),
    ("xgb", "XGBoost"),
    ("gru", "GRU"),
    ("lstm", "LSTM"),
    ("tcn", "TCN"),
    ("trajectron", "Trajectron++ + Checker"),
    ("donut", "DONUT + Checker"),
    ("smivp_vp4", "RA-VP (Ours)"),
]
INTERSECTIONS = [("FI", "FI"), ("FIDRT", "FIDRT"), ("Tianjin", "Tianjin"),
                 ("Chongqing", "Chongqing"), ("Xian", "Xi'an"), ("Changchun", "Changchun")]
POOLED = [("all6", "All-6"), ("sind4", "SInD-4"), ("fi_fidrt", "FI+FIDRT")]
METRICS = [("rec_v", "Violation Recall"), ("pr_auc", "PR-AUC"), ("macro_f1", "Macro-F1")]


def _cols():
    cols = [(disp, ("indomain", k)) for k, disp in INTERSECTIONS]
    cols += [(disp, ("pooled", k)) for k, disp in POOLED]
    cols += [(f"{disp} (LOO)", ("loo", k)) for k, disp in INTERSECTIONS]
    return cols


def _fmt(v):
    return "--" if v is None else f"{v * 100:.1f}"


def render(cells, metric_key):
    cols = _cols()
    head = "| Method | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    rows = [head, sep]
    for m, disp in METHODS:
        vals = []
        for _, (kind, key) in cols:
            md = cells.get(kind, {}).get(key, {}).get(m, {})
            vals.append(_fmt(md.get(metric_key)))
        rows.append(f"| {disp} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main():
    if not CELLS.exists():
        sys.exit(f"missing {CELLS}")
    cells = json.loads(CELLS.read_text())
    blocks = []
    for key, title in METRICS:
        blocks.append(f"### Test {title} (%)\n\n" + render(cells, key) + "\n")
    text = ("# RA-VP main results (reproduced from results/cells.json)\n\n"
            "Columns: 6 in-domain intersections, 3 pooled sets, 6 leave-one-"
            "intersection-out (LOO). Values are percentages.\n\n" + "\n".join(blocks))
    OUT.write_text(text)
    print(text)
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
