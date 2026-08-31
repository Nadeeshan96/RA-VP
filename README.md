# RA-VP: Risk-Aware Anticipatory Pedestrian Traffic-Violation Prediction

Code for the paper *"RA-VP: Risk-Aware Pedestrian Traffic
Violation Prediction at Signalized Intersections."*

RA-VP predicts whether a currently-safe pedestrian will enter an unsafe state
(road core, or a crosswalk under a non-permissive signal) within a 2 s horizon.
It builds on the strongest baseline family, forecast-and-check (sample future
trajectories, then apply an exact geofence rule), and improves its recall by
fine-tuning the forecaster with a differentiable geometric risk while keeping the
exact checker in the inference loop, plus a residual head over portable
signal/map features.

This repo reproduces the three main tables: Violation Recall, PR-AUC, and
Macro-F1 - across 15 scenarios (6 in-domain intersections, 3 pooled sets,
6 leave-one-intersection-out) for 10 methods.

## Layout

```
ravp/                 # the RA-VP method (forecaster, checker, risk features, residual head, metrics)
baselines/
  smgs/               # rule baseline (SMGS)
  tabular/            # Logistic Regression / Random Forest / XGBoost
  deep-sequence/      # GRU / LSTM / TCN
  trajectronpp-checker/  # Trajectron++ + exact checker
  donut-checker/         # DONUT + exact checker
scripts/
  render_tables.py            # render the 3 paper tables from results/cells.json (no download)
  reproduce_ravp.py           # RA-VP all-6 pooled, live from weights
  reproduce_smgs.py           # SMGS all-6, live
  reproduce_tabular.py        # LogReg/RF/XGB all-6, live
  reproduce_deep-sequence.py        # GRU/LSTM/TCN (precomputed)
  reproduce_trajectronpp-checker.py # Trajectron++ (precomputed)
  reproduce_donut-checker.py        # DONUT (precomputed)
results/
  cells.json                  # all method x scenario numbers (the source for the tables)
  runs/                       # per-cell result JSONs
  expected_main_tables.md     # rendered tables (ground truth to compare against)
download_assets.py            # fetch + verify + extract the large asset bundle
assets_manifest.json          # URL + sha256 of the asset bundle (set the URL after upload)
environment.yml
assets/                       # created by download_assets.py (data, weights, features)
```

## Install

```bash
conda env create -f environment.yml
conda activate ravp
```

## Quick start (no download needed)

Render the three main paper tables straight from the shipped results:

```bash
python scripts/render_tables.py          # prints + writes results/expected_main_tables.md
```

## Full reproduction

Download the asset bundle (data + weights + feature caches), then run the
per-method scripts.

```bash
python download_assets.py                          # populates assets/
python scripts/reproduce_ravp.py                   # RA-VP all-6 pooled
python scripts/reproduce_smgs.py                   # SMGS all-6 pooled
python scripts/reproduce_tabular.py                # LogReg / RF / XGB all-6 pooled
python scripts/reproduce_deep-sequence.py          # GRU / LSTM / TCN (precomputed)
python scripts/reproduce_trajectronpp-checker.py   # Trajectron++ + checker (precomputed)
python scripts/reproduce_donut-checker.py          # DONUT + checker (precomputed)
```

RA-VP is recomputed end-to-end from the shipped fine-tuned forecaster: it draws
K=16 future samples, runs the exact geofence checker, builds the residual-head
features, trains the head, and evaluates. 

## Large files

All large files: processed ground-truth CSVs, the fine-tuned forecaster, window/feature/baseline caches -
ship as a single tarball `assets_bundle.tar.gz` (~80 MB compressed, ~200 MB
extracted) hosted on Google Drive.

`download_assets.py` fetches it with `gdown`.

```bash
python download_assets.py
```

If you already have the tarball, skip the download:

```bash
python download_assets.py --bundle /path/to/assets_bundle.tar.gz
```

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{dissanayake2026ravp,
  title     = {RA-VP: Risk-Aware Pedestrian Traffic Violation Prediction at Signalized Intersections},
  author    = {Dissanayake, Nadeeshan and Borovica-Gajic, Renata and Tanin, Egemen and Karunasekera, Shanika},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26)},
  year      = {2026},
  doi       = {10.1145/3841645.3842969}
}
```
