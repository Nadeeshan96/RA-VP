# Trajectron++ + geofence checker

A Trajectron++-style CVAE forecaster: for each window it draws K future
trajectory samples, and the **exact** geofence checker (the same rule used for
labelling, `ravp/checker.py`) flags any sample that enters an unsafe region. The
violation score is the fraction of samples that violate; the threshold is tuned on
validation. This is the strongest baseline family and the basis RA-VP builds on.

Numbers are provided **precomputed**:
`python ../../scripts/reproduce_trajectronpp-checker.py`

Retraining is GPU-only and needs the raw trajectory CSVs (~3.5 GB, not shipped):
train `ravp.forecaster.TrajectronPP` with the trajectory ELBO, sample K futures,
and score them with `ravp.checker.GeofenceChecker`.
