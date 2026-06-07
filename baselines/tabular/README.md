# Tabular baselines (Logistic Regression / Random Forest / XGBoost)

Classical classifiers on engineered per-window features (pedestrian kinematics,
map containment + signed distances, signal context, local interaction summaries).
Trained on the pooled train split with class balancing, threshold tuned on val.

Reproduce (all-six pooled, live): `python ../../scripts/reproduce_tabular.py`
`train_eval.py` loads the cached feature tensors
(`assets/weights/tabular_cache/cache_<DS>/{train,val,test}_tab.pt`), restricts to
the portable feature set shared across intersections, fits the three models, and
evaluates. RF/XGB carry mild nondeterminism across library versions.
