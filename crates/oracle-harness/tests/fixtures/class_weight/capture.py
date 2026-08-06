#!/usr/bin/env python3
"""Capture the `lightgbm==4.6.0` golden for the class-weight parameters
`is_unbalance` and `scale_pos_weight`.

# Why this fixture exists

Both parameters were parsed, range-validated and then DROPPED: `lgbm::Binary`
hard-coded `label_weight = 1.0` for both classes, so `is_unbalance=true` and
`scale_pos_weight=5` trained exactly the balanced model. Nothing in the suite
noticed, because every committed corpus used the balanced default — the same
class of defect as the `feature_fraction` gap (see that fixture's header).

C++ `BinaryLogloss::Init` (`binary_objective.hpp:85-100`) derives:

    label_weights_ = [1.0, 1.0]                          # [negative, positive]
    if is_unbalance and cnt_pos > 0 and cnt_neg > 0:
        if cnt_pos > cnt_neg: label_weights_[0] = cnt_pos / cnt_neg
        else:                 label_weights_[1] = cnt_neg / cnt_pos
    label_weights_[1] *= scale_pos_weight                # unconditionally

and `GetGradients` multiplies BOTH the gradient and the hessian by
`label_weights_[is_pos]`. `MulticlassOVA` (`multiclass_objective.hpp:190-193`)
builds one `BinaryLogloss` per class from the same Config, so both parameters
apply per one-vs-all split with that class's own counts.

The label vector is deliberately IMBALANCED (~20% positive) — at a 50/50 split
`is_unbalance` derives `[1.0, 1.0]` and the cell would pass against the unwired
code.

Run:
  .venv/bin/python crates/oracle-harness/tests/fixtures/class_weight/capture.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np

import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent
OUT = HERE / "class_weight_golden.json"

N, D, ITERS = 600, 12, 8
RNG = np.random.default_rng(20260806)
X = RNG.standard_normal((N, D))
_z = X @ RNG.standard_normal(D)
# ~20% positive: imbalanced enough that `is_unbalance` derives a weight far from 1.
BINARY_Y = (_z > np.quantile(_z, 0.8)).astype(float)
# Skewed 3-class labels, so every one-vs-all split is imbalanced too.
MULTI_Y = np.digitize(_z, np.quantile(_z, [0.6, 0.85])).astype(float)

LABELS = {"binary": BINARY_Y, "multi": MULTI_Y}

# (objective, num_class, is_unbalance, scale_pos_weight)
#
# `is_unbalance` and `scale_pos_weight` are never combined: C++ `Log::Fatal`s on
# "Cannot set is_unbalance and scale_pos_weight at the same time"
# (binary_objective.hpp:31-33). That rejection is asserted separately, in the Rust
# test, against the typed error.
CELLS = [
    ("binary", 1, False, 1.0),  # control — must stay bit-identical to the old path
    ("binary", 1, True, 1.0),
    ("binary", 1, False, 0.25),
    ("binary", 1, False, 2.0),
    ("binary", 1, False, 5.0),
    ("multiclassova", 3, False, 1.0),  # control
    ("multiclassova", 3, True, 1.0),
    ("multiclassova", 3, False, 3.0),
]


def params_for(objective, num_class, is_unbalance, scale_pos_weight):
    p = dict(
        objective=objective,
        num_class=num_class,
        num_iterations=ITERS,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=5,
        max_bin=63,
        seed=1,
        deterministic=True,
        force_row_wise=True,
        num_threads=1,
        verbose=-1,
        is_unbalance=is_unbalance,
        scale_pos_weight=scale_pos_weight,
    )
    if num_class == 1:
        p.pop("num_class")
    return p


def main():
    out = {
        "lightgbm_version": lgb.__version__,
        "num_rows": N,
        "num_cols": D,
        "num_iterations": ITERS,
        "features_bits": [f"{v.view(np.uint64):016x}" for v in X.reshape(-1)],
        "labels_bits": {
            k: [f"{v.view(np.uint64):016x}" for v in y] for k, y in LABELS.items()
        },
        "cells": {},
    }
    for objective, num_class, is_unbalance, spw in CELLS:
        p = params_for(objective, num_class, is_unbalance, spw)
        y = LABELS["binary" if num_class == 1 else "multi"]
        ds = lgb.Dataset(X, label=y, params=p, free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=ITERS)
        raw = np.asarray(b.predict(X, raw_score=True), dtype=np.float64).reshape(-1)
        key = f"{objective}:{is_unbalance}:{spw}"
        out["cells"][key] = {
            "objective": objective,
            "num_class": num_class,
            "is_unbalance": is_unbalance,
            "scale_pos_weight": spw,
            "labels": "binary" if num_class == 1 else "multi",
            "pred_bits": [f"{v.view(np.uint64):016x}" for v in raw],
        }
        print(f"  ok {key}: {raw.size} preds", file=sys.stderr)

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {OUT}: {len(out['cells'])} cells", file=sys.stderr)


if __name__ == "__main__":
    main()
