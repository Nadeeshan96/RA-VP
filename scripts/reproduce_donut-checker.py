#!/usr/bin/env python3
"""DONUT + geofence checker: precomputed test metrics.

A DONUT (decoder-only autoregressive) forecaster whose K sampled futures are
passed through the exact geofence checker. Numbers are provided precomputed;
retraining needs a GPU and the raw trajectory CSVs (~3.5 GB, not shipped).
See baselines/donut-checker/README.md.

Usage:  python scripts/reproduce_donut-checker.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _precomputed import show

show("donut", "DONUT + Checker",
     "Retraining needs a GPU + raw trajectories; see baselines/donut-checker/README.md.")
