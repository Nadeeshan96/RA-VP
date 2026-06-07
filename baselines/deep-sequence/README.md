# Deep sequence baselines (GRU / LSTM / TCN)

Sequence classifiers over the full N_h-step history (kinematics + compact map,
signal, and interaction channels), with a small static-feature MLP. LSTM/GRU are
2-layer unidirectional (hidden 128); TCN uses causal dilated residual blocks.
Trained with class-weighted BCE, early stopping + threshold on validation Macro-F1.

Numbers are provided **precomputed**:
`python ../../scripts/reproduce_deep-sequence.py`

Retraining: the model/feature code is in this folder (`train_eval.py`,
`v4_sequence_classifiers/`). It needs the per-window sequence feature cache
(~357 MB), which is not in the default asset bundle to keep downloads small; it
can be regenerated from the processed data with the original feature builder.
