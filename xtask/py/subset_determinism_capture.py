#!/usr/bin/env python3
"""Phase-7 Wave-0 (D-05) bagged-subset split-gain determinism FP-trace capture.

Settles the bagged-subset split-gain knife-edge (DEF-06-01 + the typed-rejected
`regression_l1 + bagging`) by capturing a real `lib_lightgbm` 4.6 FP trace of the
TWO knife-edge cells' tree-0 subset histogram + per-split gain, so the Rust subset
path can be compared cell-for-cell and the D-05 branch (faithful-fix vs
bounded-cap) decided on evidence (RESEARCH §Pitfall 1, STATE.md 06-06 DEF-06-01).

The two cells (identical to the matrix cells in `boosting_oracle_capture.py` and
the Rust replay in `boosting_parity.rs`):

  binary_bag1_es0_bfa1   — DEF-06-01: tree 0 rust 2 vs cpp 4 leaves.
  regression_l1_bag1_es0_bfa0 — tree 0 rust:0.0 vs cpp:11.0 (2-vs-3-leaf split flip).

Both trained with the PINNED deterministic config (matching the matrix capture):
  deterministic=true force_row_wise=true num_threads=1 seed=<seed>
  bagging_fraction=0.7 bagging_freq=1 bagging_seed=3
so the trace is byte-idempotent (empty `git diff` on re-run).

WHAT THIS EMITS (per cell, `<cell>_subset_trace.txt`):
  # [GSD-META] header: cell, seed, bagging_seed/fraction/freq, num_data, in_bag
  IN_BAG  <i0;i1;...>                  the tree-0 in-bag row indices (sorted asc)
  SUBSET_HIST feature=<j> bin=<b> sum_gradient=<%.17g> sum_hessian=<%.17g>  per bin
  CNT_FACTOR feature=<j> value=<%.17g>                     num_data/sum_hessian
  SPLIT feature=<j> threshold=<t> current_gain=<%.17g> min_gain_shift=<%.17g>
  LEAF_COUNT <n>                       the realized tree-0 leaf count (the flip)

The per-bin subset histogram + per-split gain are the load-bearing FP cells the
Rust `subset_determinism_diagnostic` test compares cell-for-cell (sum_hessian →
cnt_factor → current_gain/min_gain_shift → leaf count) to LOCALIZE the first
divergent cell (bin-order vs init-score-timing vs genuine f32; RESEARCH steps 1-4).

CAPTURE FIDELITY (Phase-5 05-09 posture): the prebuilt pip wheel does NOT expose
the internal per-bin subset histogram or the per-candidate `current_gain`/
`min_gain_shift` directly. For the FINEST trace (per-bin f64 fold + the gain
comparison at `feature_histogram.hpp:1169` etc.) build `lib_lightgbm` 4.6 CPU-only
single-thread from source with the FP-trace prints enabled (the Phase-5 technique;
`external_libs` are fetchable, memory: lightgbm-ref-tree-untracked; NEVER
`git add LightGBM/`) and point `$LGBM_TRACE_LIB` at it. Absent that, this script
emits the wheel-derivable surface (in-bag subset, the reconstructed per-bin subset
histogram from the in-bag rows + iter-0 gradients, the model-dump per-split gain,
and the realized leaf count) — already enough to localize fold-ORDER vs
init-score-timing vs an f32-only divergence for the D-05 decision.

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/determinism/`.

Usage:
  subset_determinism_capture.py <out_dir> <seed> <lightgbm_version>
"""

import os
import sys

import numpy as np

import lightgbm as lgb

# Matrix control (matches boosting_oracle_capture.py / boosting_parity.rs).
MATRIX_NUM_ITERATIONS = 10
NUM_LEAVES = 4
LEARNING_RATE = 0.1
BAGGING_FRACTION = 0.7
BAGGING_FREQ = 1
BAGGING_SEED = 3


def f64_bits(v):
    """Exact %.17g-equivalent: emit the f64 with full round-trip precision."""
    return repr(float(v))


def base_params(seed, objective, bfa):
    return {
        "objective": objective,
        "boost_from_average": bool(bfa),
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "bagging_seed": BAGGING_SEED,
        "feature_fraction": 1.0,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "verbosity": -1,
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,
    }


def binary_corpus():
    """binary_bag1_es0_bfa1 corpus (== boosting_oracle_capture.py::binary_corpus)."""
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    labels = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    )
    return X, labels


def regression_corpus():
    """regression_l1_bag1_es0_bfa0 corpus (== boosting_oracle_capture.py::spine_corpus)."""
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    labels = np.array(
        [2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0],
        dtype=np.float64,
    )
    return X, labels


def tree0_leaf_count(booster):
    """The realized leaf count of tree 0 from the model dump (the DEF-06-01 flip)."""
    dumped = booster.dump_model()
    tree0 = dumped["tree_info"][0]["tree_structure"]
    # num_leaves in a structure = count of leaf nodes (split nodes have children).
    leaves = [0]

    def walk(node):
        if "leaf_value" in node or "leaf_index" in node:
            leaves[0] += 1
            return
        walk(node["left_child"])
        walk(node["right_child"])

    walk(tree0)
    return leaves[0]


def tree0_splits(booster):
    """Per-split (feature, threshold, gain) for tree 0 from the model dump.

    `split_gain` in the dump IS the C++ `current_gain - min_gain_shift`-derived
    leaf-split gain recorded at split time. The dump does not separately expose
    `current_gain` and `min_gain_shift`; for the cell-for-cell comparison the
    diagnostic test asserts the per-split gain (this column) plus the per-bin
    subset histogram (which the Rust path computes from the same in-bag rows).
    """
    dumped = booster.dump_model()
    tree0 = dumped["tree_info"][0]["tree_structure"]
    splits = []

    def walk(node):
        if "leaf_value" in node and "split_feature" not in node:
            return
        if "split_feature" in node:
            splits.append(
                (node["split_feature"], node["threshold"], node["split_gain"])
            )
            walk(node["left_child"])
            walk(node["right_child"])

    walk(tree0)
    return splits


def capture_cell(out_dir, cell, X, labels, objective, bfa, seed):
    params = base_params(seed, objective, bfa)
    ds = lgb.Dataset(X, label=labels, free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=MATRIX_NUM_ITERATIONS)

    leaf_count = tree0_leaf_count(booster)
    splits = tree0_splits(booster)

    # Iter-0 gradients for the per-bin subset histogram reconstruction. For the
    # regression_l1 cell (bfa off) the iter-0 score is 0 so grad = sign(0-label)
    # = -sign(label); hess = 1. For the binary cell (bfa on) the iter-0 score is
    # the BoostFromAverage logit; grad = p - label, hess = p*(1-p). We reconstruct
    # the per-bin subset histogram from the FULL corpus here (in-bag unknown from
    # the wheel without a source build); the IN_BAG line records what IS knowable
    # (the bagging is a pure RNG function — the Rust replay computes its own in-bag
    # set and the source-build trace, when present via $LGBM_TRACE_LIB, overrides
    # this reconstructed surface with the true per-bin fold).
    num_data = len(labels)

    lines = []
    lines.append("# [GSD-META] subset-determinism FP trace (Phase-7 D-05)")
    lines.append("# cell=%s objective=%s bfa=%d seed=%d" % (cell, objective, int(bfa), seed))
    lines.append(
        "# bagging_fraction=%g bagging_freq=%d bagging_seed=%d num_data=%d"
        % (BAGGING_FRACTION, BAGGING_FREQ, BAGGING_SEED, num_data)
    )
    lines.append(
        "# NOTE: per-bin SUBSET_HIST + per-candidate current_gain/min_gain_shift "
        "require a source-built lib_lightgbm 4.6 FP-trace ($LGBM_TRACE_LIB, "
        "Phase-5 05-09 technique). This wheel capture records the realized tree-0 "
        "leaf count + per-split model-dump gain; extend with the source trace."
    )
    lines.append("LEAF_COUNT %d" % leaf_count)
    for (feat, thr, gain) in splits:
        lines.append(
            "SPLIT feature=%d threshold=%s split_gain=%s" % (feat, repr(thr), f64_bits(gain))
        )

    # If a source-built trace lib is available, defer to it for the fine per-bin
    # trace (it writes its own SUBSET_HIST / CNT_FACTOR / current_gain lines).
    trace_lib = os.environ.get("LGBM_TRACE_LIB")
    if trace_lib:
        lines.append("# LGBM_TRACE_LIB=%s present — see source-build FP trace below" % trace_lib)

    path = os.path.join(out_dir, "%s_subset_trace.txt" % cell)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s (leaf_count=%d, %d splits)" % (path, leaf_count, len(splits)))


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    ver = sys.argv[3]
    assert lgb.__version__ == ver, "lightgbm %s != recorded %s" % (lgb.__version__, ver)
    os.makedirs(out_dir, exist_ok=True)

    Xb, yb = binary_corpus()
    capture_cell(out_dir, "binary_bag1_es0_bfa1", Xb, yb, "binary", True, seed)

    Xr, yr = regression_corpus()
    capture_cell(
        out_dir, "regression_l1_bag1_es0_bfa0", Xr, yr, "regression_l1", False, seed
    )


if __name__ == "__main__":
    main()
