# DONUT + geofence checker

Same forecast-and-check pipeline as `trajectronpp-checker`, but the forecaster is
DONUT (a decoder-only, autoregressive trajectory predictor). K sampled futures are
scored by the exact geofence checker (`ravp/checker.py`), with the threshold tuned
on validation.

Numbers are provided **precomputed**:
`python ../../scripts/reproduce_donut-checker.py`

Retraining is GPU-only and needs the raw trajectory CSVs (~3.5 GB, not shipped).
