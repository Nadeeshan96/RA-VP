#!/usr/bin/env python3
"""Deep sequence baselines (GRU / LSTM / TCN): precomputed test metrics.

These RNN/TCN classifiers consume the full N_h-step history. Their numbers are
provided precomputed; retraining uses the per-window sequence feature cache (~357
MB, not in the default asset bundle). See baselines/deep-sequence/README.md.

Usage:  python scripts/reproduce_deep-sequence.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _precomputed import show

NOTE = ("Retraining: baselines/deep-sequence/ (needs the sequence feature cache; "
        "see that folder's README.md).")
for method, disp in [("gru", "GRU"), ("lstm", "LSTM"), ("tcn", "TCN")]:
    show(method, disp, NOTE)
    print()
