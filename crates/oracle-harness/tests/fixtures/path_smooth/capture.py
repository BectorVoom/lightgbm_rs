#!/usr/bin/env python3
"""Capture the `lightgbm==4.6.0` golden for `path_smooth` and `max_delta_step`.

# Why this fixture exists

Both parameters were REJECTED by the split kernels ("only the default 0.0 path is
transcribed") until they were implemented. They are the C++ `USE_SMOOTHING` and
`USE_MAX_OUTPUT` template axes of `FeatureHistogram`:

- `USE_MAX_OUTPUT = config->max_delta_step > 0` (feature_histogram.hpp:248-259):
  every computed leaf output is clamped to `Sign(ret) * max_delta_step`.
- `USE_SMOOTHING = config->path_smooth > kEpsilon` (feature_histogram.hpp:264-270):
  every leaf output is blended toward the leaf's `parent_output` with weight
  `num_data / (num_data + path_smooth)`.

Both axes ALSO switch the gain FORM: with either on, `GetLeafGain` stops using the
closed form `sg²/(h+λ)` and instead evaluates `GetLeafGainGivenOutput` at the
computed output. Those are equal in exact arithmetic but differ in floating point,
so a cell where the clamp never actually fires (`max_delta_step` larger than every
output) is STILL a distinct code path — `max_delta_step_switches_the_gain_form`
in the Rust test pins exactly that.

# What makes these cells load-bearing

`path_smooth` chains through depth: a leaf's `parent_output` is the (already
smoothed) output its parent split assigned it, so an error at depth 1 compounds
downward. The cells therefore use `num_leaves=16` over 8 iterations rather than
stumps, and the golden carries the full model text so the Rust test can compare
every leaf value, not just predictions.

Run:
  .venv/bin/python crates/oracle-harness/tests/fixtures/path_smooth/capture.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np

import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent
OUT = HERE / "path_smooth_golden.json"

N, D, ITERS = 800, 10, 8
RNG = np.random.default_rng(20260806)
X = RNG.standard_normal((N, D))
Y = (X @ RNG.standard_normal(D) > 0).astype(float)

# (path_smooth, max_delta_step, lambda_l1, num_leaves)
#
# `lambda_l1 > 0` rows matter because the new axes compose with the C++ `USE_L1`
# axis: `ThresholdL1` feeds the base output that then gets clamped/blended.
CELLS = [
    (0.0, 0.0, 0.0, 16),      # control — must stay bit-identical to the old path
    (0.1, 0.0, 0.0, 16),
    (1.0, 0.0, 0.0, 16),
    (10.0, 0.0, 0.0, 16),
    (100.0, 0.0, 0.0, 16),
    (1.0, 0.0, 0.0, 4),       # shallow: fewer parent_output hops
    (1.0, 0.0, 0.5, 16),      # smoothing x L1
    (0.0, 0.5, 0.0, 16),      # clamp active
    (0.0, 0.05, 0.0, 16),     # clamp active and BINDING on most leaves
    (0.0, 1e3, 0.0, 16),      # clamp never fires: gain FORM switch only
    (0.0, 0.5, 0.5, 16),      # clamp x L1
    (1.0, 0.5, 0.0, 16),      # both axes
    (10.0, 0.05, 0.5, 16),    # both axes x L1
]


def params_for(path_smooth, max_delta_step, lambda_l1, num_leaves):
    return dict(
        objective="binary",
        num_iterations=ITERS,
        learning_rate=0.1,
        num_leaves=num_leaves,
        min_data_in_leaf=5,
        max_bin=63,
        seed=1,
        deterministic=True,
        force_row_wise=True,
        num_threads=1,
        verbose=-1,
        path_smooth=path_smooth,
        max_delta_step=max_delta_step,
        lambda_l1=lambda_l1,
    )


def leaf_values(model_text):
    """Every `leaf_value=` entry, tree by tree — the direct witness of the blend."""
    out = []
    for chunk in model_text.split("Tree=")[1:]:
        vals = []
        for line in chunk.splitlines():
            if line.startswith("leaf_value="):
                vals.extend(float(v) for v in line.split("=", 1)[1].split())
        out.append(vals)
    return out


def main():
    out = {
        "lightgbm_version": lgb.__version__,
        "num_rows": N,
        "num_cols": D,
        "num_iterations": ITERS,
        "features_bits": [f"{v.view(np.uint64):016x}" for v in X.reshape(-1)],
        "labels_bits": [f"{v.view(np.uint64):016x}" for v in Y],
        "cells": {},
    }
    for ps, mds, l1, nl in CELLS:
        p = params_for(ps, mds, l1, nl)
        ds = lgb.Dataset(X, label=Y, params=p, free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=ITERS)
        raw = np.asarray(b.predict(X, raw_score=True), dtype=np.float64)
        leaves = leaf_values(b.model_to_string())
        key = f"{ps}:{mds}:{l1}:{nl}"
        out["cells"][key] = {
            "path_smooth": ps,
            "max_delta_step": mds,
            "lambda_l1": l1,
            "num_leaves": nl,
            "pred_bits": [f"{v.view(np.uint64):016x}" for v in raw],
            "leaf_values": leaves,
        }
        flat = [v for t in leaves for v in t]
        print(
            f"  ok ps={ps} mds={mds} l1={l1} nl={nl}: "
            f"{len(leaves)} trees, |leaf| max {max(abs(v) for v in flat):.4f}",
            file=sys.stderr,
        )

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {OUT}: {len(out['cells'])} cells", file=sys.stderr)


if __name__ == "__main__":
    main()
