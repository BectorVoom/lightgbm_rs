#!/usr/bin/env python3
"""Phase-7 W6 REAL lib_lightgbm Random Forest (BST-06) oracle capture (plan 07-07).

Captures the RF MODEL PARITY cells — `rf_{single,multi}_bag_model.txt` + `_pred.txt`.
Trains the `boosting=rf` averaged-tree variant with MANDATORY bagging
(bagging_fraction=0.7, bagging_freq=1) over the single-output (regression) and
multiclass axes on the real prebuilt `lib_lightgbm` 4.6 pip wheel and dumps the
authoritative `%.17g` model text + per-row predictions.

RF semantics the goldens validate (rf.hpp):
  - averaged (not accumulated) trees: the stored leaf values are the RAW per-tree
    outputs and the model carries `average_output`; predict() divides the tree sum by
    num_iteration (gbdt_prediction.cpp:57-59).
  - mandatory randomization: RF requires bagging (or feature_fraction<1). We use
    bagging (the 07-01 bit-exact bagging RNG golden carries over).
  - per-tree leaf renewal to the mean residual `label - init_score` and NO
    learning-rate shrinkage (shrinkage_rate_=1.0).

The averaged-tree leaf structure on the bagged subset inherits the 07-01 D-05
posture (faithful-fix: the bagged-subset split-gain knife-edge was a min_gain_shift
operand bug, now fixed — so RF reaches faithful parity on the bagged subset, not a
bounded-cap). A wrong RF branch (accumulation instead of averaging, missing renew,
or a shrinkage) shifts the leaves and FAILS the parity replay.

DETERMINISM / IDEMPOTENCY: trained with `deterministic=true force_row_wise=true
num_threads=1 seed=<seed>`; the goldens are pure functions of the pinned inputs, so
re-running produces byte-identical files (empty `git diff`).

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/rf/`.

Usage:
  rf_oracle_capture.py <out_dir> <seed> <bagging_seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb

# ---- RF capture control (mirrors boosting_parity.rs RF_* constants) ----
NUM_ITERATIONS = 12
NUM_LEAVES = 4
# RF ignores learning_rate (shrinkage 1.0); set it to the matrix default anyway so
# the param dict is identical in shape to the other variants.
LEARNING_RATE = 0.1
# Mandatory bagging (matches the boosting matrix MATRIX_BAGGING_* constants).
BAGGING_FRACTION = 0.7
BAGGING_FREQ = 1
NUM_CLASS = 3


def f64_bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def f64_bits_line(values):
    return " ".join(str(f64_bits(v)) for v in np.ravel(values))


def base_params(seed, bagging_seed):
    return {
        "boosting": "rf",
        "boost_from_average": True,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "verbosity": -1,
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,
        # RF's mandatory randomization (bagging).
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "bagging_seed": bagging_seed,
    }


def single_corpus():
    """The single-output (regression) spine corpus (D-08): 2 features, 12 rows,
    identity-binned, monotone-ish labels so splits are found."""
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    labels = np.array(
        [2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0],
        dtype=np.float64,
    )
    return X, labels


def multi_corpus():
    """The multiclass spine corpus (D-08): 2 features, 12 rows, identity-binned,
    3-class integer labels, all classes present (class_need_train true)."""
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    labels = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0],
        dtype=np.float64,
    )
    return X, labels


def capture_single(out_dir, seed, bagging_seed):
    X, labels = single_corpus()
    p = base_params(seed, bagging_seed)
    p["objective"] = "regression"
    p["metric"] = ["l2"]
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    booster = lgb.train(p, dtrain, num_boost_round=NUM_ITERATIONS)
    booster.save_model(os.path.join(out_dir, "rf_single_bag_model.txt"))
    preds = booster.predict(X)
    with open(os.path.join(out_dir, "rf_single_bag_pred.txt"), "w") as fh:
        fh.write("# RF single-output (regression) per-row averaged predict() "
                 "(f64 bit patterns).\n")
        fh.write(f64_bits_line(preds) + "\n")


def capture_multi(out_dir, seed, bagging_seed):
    X, labels = multi_corpus()
    p = base_params(seed, bagging_seed)
    p["objective"] = "multiclass"
    p["num_class"] = NUM_CLASS
    p["metric"] = ["multi_logloss"]
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    booster = lgb.train(p, dtrain, num_boost_round=NUM_ITERATIONS)
    booster.save_model(os.path.join(out_dir, "rf_multi_bag_model.txt"))
    # Per-row per-class transformed predictions; dump class-major (column-major) so
    # the layout matches the Rust score_[num_data*k + i] convention.
    preds = booster.predict(X)  # shape (num_data, num_class)
    with open(os.path.join(out_dir, "rf_multi_bag_pred.txt"), "w") as fh:
        fh.write("# RF multiclass per-row averaged predict() (f64 bit patterns), "
                 "class-major.\n")
        fh.write(f64_bits_line(np.asarray(preds).reshape(-1, order="F")) + "\n")


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: rf_oracle_capture.py <out_dir> <seed> <bagging_seed> "
                 "<lightgbm_version>")
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    bagging_seed = int(sys.argv[3])
    expected_version = sys.argv[4]
    if lgb.__version__ != expected_version:
        sys.exit(f"ABORT: lightgbm {lgb.__version__} != recorded {expected_version}")
    os.makedirs(out_dir, exist_ok=True)

    capture_single(out_dir, seed, bagging_seed)
    capture_multi(out_dir, seed, bagging_seed)
    print("rf_oracle_capture: wrote RF single + multiclass model + pred cells to",
          out_dir)


if __name__ == "__main__":
    main()
