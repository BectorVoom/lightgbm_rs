#!/usr/bin/env python3
"""Capture the `lightgbm==4.6.0` golden for the `feature_fraction` family.

# Why this fixture exists

Before 2026-08-06 the committed `learner/col_sampler.txt` golden covered exactly
ONE cell — `feature_fraction=1.0, feature_fraction_bynode=0.5` — so the per-TREE
`feature_fraction` draw had NO C++ coverage. Three defects hid behind that gap:

1. `feature_fraction` never reached the tree learner from the public API
   (`with_feature_fraction` was called only from tests), so the port trained an
   UNSAMPLED model.
2. `ColSampler` was rebuilt per tree, re-seeding its PRNG, so every tree selected
   the SAME feature subset instead of advancing the stream.
3. The first tree consumed the Init-time draw, shifting every tree's subset by one
   (C++ draws once in `SetTrainingData` and once per tree in `BeforeTrain`, so C++
   tree N uses draw N+2).

Each is invisible to a single-tree, `feature_fraction=1.0` golden. This capture
spans MULTIPLE trees and several fractions, which is what makes it a real gate.

Run:
  .venv/bin/python crates/oracle-harness/tests/fixtures/feature_fraction/capture.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np

import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent
OUT = HERE / "feature_fraction_golden.json"

# Small but non-degenerate: enough features that a fraction actually excludes some,
# enough trees that a per-tree redraw is observable.
N, D, ITERS = 600, 20, 8
RNG = np.random.default_rng(20260806)
X = RNG.standard_normal((N, D))
Y = (X @ RNG.standard_normal(D) > 0).astype(float)

# (feature_fraction, feature_fraction_bynode)
#
# The `bynode` rows are load-bearing for a FOURTH defect, fixed 2026-08-06: the port
# used the per-node mask to SKIP THE SCAN, whereas C++ scans every bytree-selected
# feature (setting `is_splittable_` from real data) and applies the per-node mask only
# to the split argmax. The Rust skip left `is_splittable_ = false`, which propagates to
# both children through `parent_splittable` and permanently removes features C++ would
# still consider deeper in the tree — so this only shows up several levels down, which
# is why multi-tree, multi-level cells are required to catch it.
CELLS = [
    (1.0, 1.0),
    (0.75, 1.0),
    (0.5, 1.0),
    (0.25, 1.0),
    (0.1, 1.0),
    (1.0, 0.5),
    (1.0, 0.25),
    (0.5, 0.5),
    (0.75, 0.25),
]


def base_params(ff, ffn):
    return dict(
        objective="binary",
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
        feature_fraction=ff,
        feature_fraction_bynode=ffn,
        feature_fraction_seed=2,
    )


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
    for ff, ffn in CELLS:
        p = base_params(ff, ffn)
        ds = lgb.Dataset(X, label=Y, params=p, free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=ITERS)
        raw = np.asarray(b.predict(X, raw_score=True), dtype=np.float64)
        # Per-tree split features: the DIRECT witness that each tree draws its own
        # subset. A per-tree redraw bug shows up here even when predictions are close.
        per_tree = []
        for chunk in b.model_to_string().split("Tree=")[1:]:
            feats = set()
            for line in chunk.splitlines():
                if line.startswith("split_feature="):
                    feats.update(int(v) for v in line.split("=", 1)[1].split())
            per_tree.append(sorted(feats))
        out["cells"][f"{ff}:{ffn}"] = {
            "feature_fraction": ff,
            "feature_fraction_bynode": ffn,
            "pred_bits": [f"{v.view(np.uint64):016x}" for v in raw],
            "per_tree_split_features": per_tree,
        }
        print(f"  ok ff={ff} bynode={ffn}: trees={len(per_tree)}", file=sys.stderr)

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {OUT}: {len(out['cells'])} cells", file=sys.stderr)


if __name__ == "__main__":
    main()
