#!/usr/bin/env python3
"""Phase-6 REAL lib_lightgbm GBDT-spine oracle capture (plan 06-02, D-06..D-13).

Trains the minimal end-to-end regression-L2 SPINE on the real prebuilt
`lib_lightgbm` 4.6 pip wheel (the same authoritative binary the Phase-3/5 captures
use) and dumps the layered L1-L5 goldens. This is the FIRST full
builder->train->score-accumulate->predict->metric loop validated against the real
binary, on ONE objective (regression), before any axis widens (D-14/D-15).

The spine cell is the allowed D-07 collapse: bagging OFF, early-stopping OFF,
boost_from_average ON (the C++ regression default) — RESEARCH Cross-Product
Collapse Analysis.

DETERMINISM / IDEMPOTENCY (Phase 1-5 discipline): trained with
`deterministic=true force_row_wise=true num_threads=1 seed=<seed>
bagging_fraction=1.0 feature_fraction=1.0`, NO subsampling, so re-running produces
byte-identical goldens (empty `git diff`).

IDENTITY BINNING (the single highest execution risk — a binning mismatch would
surface as a misleading parity FAILURE unrelated to the loop). Raw feature values
are distinct consecutive integers 0..K-1 == the intended bin indices; with
`max_bin>=K`, `min_data_in_bin=1`, `bin_construct_sample_cnt >> n_rows`,
`feature_pre_filter=false`, LightGBM's chosen bin upper-bounds are the
half-integers between consecutive raw values, so `binned_value == raw_value` and
the Rust learner (fed the same integer bins) grows a bit-comparable tree.

Layered goldens (RESEARCH Validation Architecture L1-L5):
  L1 regression_gh_iter1.txt / regression_gh_iterN.txt
       per-row grad/hess at iter 1 and a later iter. For regression-L2,
       grad = score_{k-1} - label, hess = 1, so g/h is DERIVED from the
       per-iteration raw score (RESEARCH L1 recommendation: score-derivation
       route) — exact, no custom-fobj interception needed.
  L2 regression_scores.txt
       per-iteration accumulated raw score: predict(raw_score=True,
       num_iteration=k) for k=1..N (f64 bit pattern).
  L3 regression_metrics.txt
       per-round l2 + rmse from evals_result (record_evaluation).
  L5 regression_spine_model.txt   save_model() authoritative %.17g model text.
     regression_spine_pred.txt    predict() transformed output (== raw for
                                  regression identity ConvertOutput) (f64 bits).

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/boosting/`.

Usage:
  boosting_oracle_capture.py <out_dir> <seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb

# The spine training control (D-09: modest depth, small trees).
NUM_ITERATIONS = 10
NUM_LEAVES = 4
LEARNING_RATE = 0.1
# "A later iteration" for the D-10 L1 golden (scores no longer zero).
LATER_ITER = 5


def f64_bits_line(values):
    """Serialize a list of f64 as space-separated u64 bit patterns (exact)."""
    return " ".join(str(struct.unpack("<Q", struct.pack("<d", float(v))[:8])[0])
                     for v in values)


def f32_bits_line(values):
    """Serialize a list of f32 as space-separated u32 bit patterns (exact)."""
    return " ".join(str(struct.unpack("<I", struct.pack("<f", float(v)))[0])
                    for v in values)


def base_params(seed):
    return {
        "objective": "regression",        # L2
        "metric": ["l2", "rmse"],
        "boost_from_average": True,        # the C++ regression DEFAULT (D-15)
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "verbosity": -1,
        # identity binning: bin index == raw value
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,
    }


def assert_identity_binning(X, expected_most_freq):
    """ABORT unless each feature's raw values are 0..K-1 and the modal bin matches
    the intended most_freq_bin (the load-bearing offset property)."""
    for j in range(X.shape[1]):
        col = X[:, j]
        distinct = np.unique(col)
        k = len(distinct)
        if not np.array_equal(distinct, np.arange(k, dtype=col.dtype)):
            sys.exit(
                "ABORT: feature %d raw values %s are not consecutive 0..%d-1"
                % (j, distinct.tolist(), k)
            )
        counts = np.bincount(col.astype(np.int64), minlength=k)
        modal = int(np.argmax(counts))
        if modal != expected_most_freq[j]:
            sys.exit(
                "ABORT: feature %d realized most_freq_bin %d != intended %d (counts=%s)"
                % (j, modal, expected_most_freq[j], counts.tolist())
            )


def spine_corpus():
    """The regression spine corpus: 2 features, 12 rows, identity-binned.

    Feature 0: 6 distinct values 0..5 each twice (no dominant bin => mfb 0).
    Feature 1: a second informative column (also mfb 0).
    Labels: a clean monotone spread so boost_from_average (label mean) is non-zero
    (|init| > kEpsilon) AND the trees find real splits.
    """
    #          row:  0  1  2  3  4  5  6  7  8  9 10 11
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    # Labels with a non-zero mean (so BoostFromAverage init is significant) and a
    # monotone-ish relation to f0 so splits are found.
    labels = np.array(
        [2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0],
        dtype=np.float64,
    )
    # f0 modal bin = 0 (all tie at count 2 => lowest wins); f1 modal bin = 0.
    expected_most_freq = [0, 0]
    return X, labels, expected_most_freq


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s" % (lgb.__version__, version)
    )

    os.makedirs(out_dir, exist_ok=True)
    X, labels, expected_most_freq = spine_corpus()

    p = base_params(seed)
    dtrain = lgb.Dataset(X, label=labels, free_raw_data=False)
    dtrain.construct()
    assert_identity_binning(X, expected_most_freq)

    evals_result = {}
    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=NUM_ITERATIONS,
        valid_sets=[dtrain],
        valid_names=["training"],
        callbacks=[lgb.record_evaluation(evals_result)],
    )

    # ---- L5: model text + transformed predict ----
    booster.save_model(os.path.join(out_dir, "regression_spine_model.txt"))
    pred = booster.predict(X)  # transformed (identity for regression)
    with open(os.path.join(out_dir, "regression_spine_pred.txt"), "w") as fh:
        fh.write("# regression_spine_pred — transformed predict() f64 bits\n")
        fh.write(f64_bits_line(pred) + "\n")

    # ---- L2: per-iteration accumulated raw score ----
    # predict(raw_score=True, num_iteration=k) == the internal score_ after k iters
    # (RESEARCH Open-Q2/A4: this cell verifies score_ == predict(raw,k)).
    with open(os.path.join(out_dir, "regression_scores.txt"), "w") as fh:
        fh.write("# regression_scores — per-iter raw score; one line per k=1..N; f64 bits\n")
        fh.write(f"# num_iterations={NUM_ITERATIONS} num_data={len(labels)}\n")
        for k in range(1, NUM_ITERATIONS + 1):
            raw_k = booster.predict(X, raw_score=True, num_iteration=k)
            fh.write(f64_bits_line(raw_k) + "\n")

    # ---- L1: per-row grad/hess (DERIVED from per-iter scores) ----
    # regression-L2: grad_k[i] = score_{k-1}[i] - label[i], hess = 1.
    # iter 1: score_0 = init (label mean, added by BoostFromAverage). With
    # boost_from_average=true the iter-1 input score IS the init score for all rows.
    init = float(np.mean(labels))
    score0 = np.full(len(labels), init, dtype=np.float64)
    grad1 = (score0 - labels).astype(np.float32)
    hess1 = np.ones(len(labels), dtype=np.float32)
    with open(os.path.join(out_dir, "regression_gh_iter1.txt"), "w") as fh:
        fh.write("# regression_gh_iter1 — iter-1 per-row grad/hess; f32 bits; "
                 "GRAD then HESS\n")
        fh.write("GRAD " + f32_bits_line(grad1) + "\n")
        fh.write("HESS " + f32_bits_line(hess1) + "\n")

    score_prev = booster.predict(X, raw_score=True, num_iteration=LATER_ITER - 1)
    gradN = (np.asarray(score_prev) - labels).astype(np.float32)
    hessN = np.ones(len(labels), dtype=np.float32)
    with open(os.path.join(out_dir, "regression_gh_iterN.txt"), "w") as fh:
        fh.write(f"# regression_gh_iterN — iter-{LATER_ITER} per-row grad/hess; "
                 "f32 bits; GRAD then HESS\n")
        fh.write(f"# iter={LATER_ITER} init={init!r}\n")
        fh.write("GRAD " + f32_bits_line(gradN) + "\n")
        fh.write("HESS " + f32_bits_line(hessN) + "\n")

    # ---- L3: per-round metrics ----
    with open(os.path.join(out_dir, "regression_metrics.txt"), "w") as fh:
        fh.write("# regression_metrics — per-round l2 + rmse (record_evaluation); f64 bits\n")
        for metric in ("l2", "rmse"):
            vals = evals_result["training"][metric]
            fh.write(f"{metric} " + f64_bits_line(vals) + "\n")

    print("boosting_oracle_capture: wrote L1/L2/L3/L5 goldens to %s" % out_dir)


if __name__ == "__main__":
    main()
