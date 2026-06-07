#!/usr/bin/env python3
"""Trajectron++ + geofence checker: precomputed test metrics.

A Trajectron++ CVAE forecaster whose K sampled futures are passed through the
exact geofence checker (same rule as labelling). Numbers are provided precomputed;
retraining needs a GPU and the raw trajectory CSVs (~3.5 GB, not shipped).
See baselines/trajectronpp-checker/README.md.

Usage:  python scripts/reproduce_trajectronpp-checker.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _precomputed import show

show("trajectron", "Trajectron++ + Checker",
     "Retraining needs a GPU + raw trajectories; see baselines/trajectronpp-checker/README.md.")
