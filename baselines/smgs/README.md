# SMGS rule baseline

SMGS (Signal-, Map-, Gap-, and Social-aware) is a training-free, interpretable
rule baseline. It scores each window from seven binary indicators (constant-
velocity entry into a prohibited region, velocity/heading toward it, signal /
waiting-time pressure, vehicle-gap availability, a neighbour already entering,
and near-boundary urgency), normalises the weighted score to [0,1], and applies a
single decision threshold tuned on the validation split.

Reproduce (all-six pooled, live): `python ../../scripts/reproduce_smgs.py`
It loads the cached per-window rule scores (`assets/weights/smgs_scores/<DS>.npz`),
pools the six datasets, tunes the threshold on val, and evaluates on test.

The rule implementation in this folder (`smgs.py`, `rules.py`, `config.py`,
`context.py`) is the scorer used to produce those cached scores from the
ground-truth windows + zone/signal data.
