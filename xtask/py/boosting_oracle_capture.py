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
# The multiclass cells run for FEWER iterations than the single-output spine: the
# redundant-form softmax `exp` (Rust libm vs the C++ wheel's std::exp) diverges at
# the ~1-ULP level and flips a knife-edge split at iter ~5-6 on this corpus. Capping
# the multiclass horizon at 5 iters (15 trees over 3 classes) keeps EVERY grown tree
# BIT-EXACT to the real binary (the L2/L5 contract holds), proving the per-class
# class-major layout end-to-end. The flip itself is the documented exp-libm residual
# (see 06-04-SUMMARY); 06-05+ may revisit a libm-matched longer horizon.
MULTICLASS_NUM_ITERATIONS = 5
MULTICLASS_LATER_ITER = 4

# OBJ-04/05 exp/log family (poisson/gamma/tweedie/cross_entropy/cross_entropy_lambda):
# every grad/hess + BoostFromScore + ConvertOutput uses a TRANSCENDENTAL (exp / log /
# log1p / expm1). The Rust system libm and the C++ wheel's std::{exp,log} differ at the
# ~1-ULP level (Pitfall 5, carried from the multiclass exp-libm horizon, 06-04). At a
# knife-edge split gain a sub-ULP g/h difference can flip a split at a deep iteration.
# We CAP the iteration horizon at EXP_LOG_NUM_ITERATIONS so EVERY grown tree stays
# BIT-EXACT to the real binary (the leaf-value parity contract holds end-to-end),
# rather than weakening the assertion. The only ORACLE_TOL surface is the predict-side
# ConvertOutput (one exp/sigmoid/log1p per row). Mirrors the multiclass 5-iter
# precedent.
EXP_LOG_NUM_ITERATIONS = 5
EXP_LOG_LATER_ITER = 4


def f64_bits_line(values):
    """Serialize a list of f64 as space-separated u64 bit patterns (exact)."""
    return " ".join(str(struct.unpack("<Q", struct.pack("<d", float(v))[:8])[0])
                     for v in values)


def f32_bits_line(values):
    """Serialize a list of f32 as space-separated u32 bit patterns (exact)."""
    return " ".join(str(struct.unpack("<I", struct.pack("<f", float(v)))[0])
                    for v in values)


def base_params(seed, objective="regression", metric=None):
    if metric is None:
        metric = ["l2", "rmse"]
    return {
        "objective": objective,
        "metric": metric,
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


def binary_corpus():
    """The binary spine corpus: 2 features, 12 rows, identity-binned, 0/1 labels
    with BOTH classes present (so ClassNeedTrain == true and pavg != 0/1)."""
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    # 5 positives of 12 -> pavg = 5/12 (|init| > kEps); separable-ish by f0.
    labels = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    )
    expected_most_freq = [0, 0]
    return X, labels, expected_most_freq


def multiclass_corpus():
    """The multiclass spine corpus (D-08): 2 features, 12 rows, identity-binned,
    3-class integer labels in [0,3), ALL classes present (class_need_train true)
    and class probs distinct from 0/1.

    f0 separates the classes reasonably so the per-class one-vs-rest trees find
    real splits."""
    f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    X = np.array([f0, f1], dtype=np.float64).T
    # 3 classes, balanced (4 rows each), all present (class_need_train true). The
    # redundant-form softmax `exp` is a TRANSCENDENTAL — the Rust system libm `exp`
    # and the C++ wheel's `std::exp` differ at the ~1-ULP level, so the multiclass
    # g/h (and the Newton leaf outputs / scores it drives) match the real binary to
    # ~1e-6 (the project's CPU contract for transcendental-bearing paths) but NOT
    # bit-exact — and at a knife-edge split gain a sub-ULP g/h difference can flip a
    # split. This is the documented exp-libm residual (CLAUDE.md "bit-exact where
    # the algorithm permits"); the multiclass replay asserts ~1e-6, the layout /
    # tree-count / class-major stride are exact. See 06-04-SUMMARY.
    labels = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0],
        dtype=np.float64,
    )
    expected_most_freq = [0, 0]
    return X, labels, expected_most_freq


def _class_major(arr2d):
    """Flatten a (num_data, num_class) row-major array to a class-major 1-D vector
    (class 0's rows, then class 1's, ...), matching the Rust score_ layout
    score_[num_data*k + i]. numpy `order='F'` on a (num_data, num_class) array does
    exactly this (column-major)."""
    return np.asarray(arr2d).reshape(-1, order="F")


def softmax_rows(raw2d):
    """Common::Softmax per row (max-subtraction), on a (num_data, num_class) raw
    score matrix. Returns the same-shape probability matrix."""
    raw = np.asarray(raw2d, dtype=np.float64)
    wmax = raw.max(axis=1, keepdims=True)
    e = np.exp(raw - wmax)
    return e / e.sum(axis=1, keepdims=True)


def multiclass_gh(raw2d, labels, num_class):
    """Softmax GetGradients (multiclass_objective.hpp:86-130), CLASS-MAJOR output.

    raw2d is (num_data, num_class) raw scores; returns (grad, hess) as class-major
    f32 vectors of length num_data*num_class:
      p = softmax(raw); grad[k] = p[k]-1 if label==k else p[k]; hess = factor*p*(1-p).
    """
    num_data = len(labels)
    factor = num_class / (num_class - np.float32(1.0))
    p = softmax_rows(raw2d)  # (num_data, num_class)
    lab = np.asarray(labels).astype(int)
    grad = p.copy()
    grad[np.arange(num_data), lab] -= np.float32(1.0)
    hess = factor * p * (np.float32(1.0) - p)
    return _class_major(grad).astype(np.float32), _class_major(hess).astype(np.float32)


def multiclassova_gh(raw2d, labels, num_class, sigmoid=1.0):
    """MulticlassOVA GetGradients (multiclass_objective.hpp:228-233), CLASS-MAJOR.

    Per class i: binary g/h with is_pos = (label == i), over the class-major slice.
    raw2d is (num_data, num_class) raw scores."""
    raw = np.asarray(raw2d, dtype=np.float64)
    grads = np.empty_like(raw)
    hesss = np.empty_like(raw)
    lab = np.asarray(labels).astype(int)
    for i in range(num_class):
        is_pos = (lab == i)
        label_val = np.where(is_pos, 1.0, -1.0)
        s = raw[:, i]
        response = -label_val * sigmoid / (1.0 + np.exp(label_val * sigmoid * s))
        abs_resp = np.abs(response)
        grads[:, i] = response
        hesss[:, i] = abs_resp * (sigmoid - abs_resp)
    return _class_major(grads).astype(np.float32), _class_major(hesss).astype(np.float32)


def write_layered_multiclass(out_dir, prefix, booster, X, labels, evals_result,
                             metric_names, gh_fn, inits, num_class, num_data,
                             num_iterations=NUM_ITERATIONS, later_iter=LATER_ITER):
    """Write the L1-L5 goldens for a multiclass cell (class-major layout, K trees
    per iteration).

    `inits` is the per-class init score vector (length num_class); `gh_fn(raw2d,
    labels, num_class) -> (grad, hess)` derives the class-major g/h from a
    (num_data, num_class) raw-score matrix."""
    # L5: model text + transformed predict (probabilities, class-major flattened).
    booster.save_model(os.path.join(out_dir, f"{prefix}_spine_model.txt"))
    pred_prob = booster.predict(X)  # (num_data, num_class) probabilities
    with open(os.path.join(out_dir, f"{prefix}_spine_pred.txt"), "w") as fh:
        fh.write(f"# {prefix}_spine_pred — transformed predict() probabilities; "
                 f"class-major f64 bits (class 0 rows, then class 1, ...)\n")
        fh.write(f64_bits_line(_class_major(pred_prob)) + "\n")

    # L2: per-iter raw score (class-major reshape of the (num_data, num_class)
    # raw_score predict).
    with open(os.path.join(out_dir, f"{prefix}_scores.txt"), "w") as fh:
        fh.write(f"# {prefix}_scores — per-iter raw score; one line per k=1..N; "
                 f"class-major f64 bits; num_class={num_class}\n")
        fh.write(f"# num_iterations={num_iterations} num_data={num_data}\n")
        for k in range(1, num_iterations + 1):
            raw_k = booster.predict(X, raw_score=True, num_iteration=k)
            fh.write(f64_bits_line(_class_major(raw_k)) + "\n")

    # L1 iter 1: score_0 = the per-class init for every row (BoostFromAverage).
    raw0 = np.tile(np.asarray(inits, dtype=np.float64), (num_data, 1))  # (nd, K)
    grad1, hess1 = gh_fn(raw0, labels, num_class)
    with open(os.path.join(out_dir, f"{prefix}_gh_iter1.txt"), "w") as fh:
        fh.write(f"# {prefix}_gh_iter1 — iter-1 per-row/per-class grad/hess; "
                 f"class-major f32 bits; GRAD then HESS\n")
        fh.write("GRAD " + f32_bits_line(grad1) + "\n")
        fh.write("HESS " + f32_bits_line(hess1) + "\n")

    # L1 later iter: raw score AFTER later_iter-1 trees per class.
    raw_prev = np.asarray(booster.predict(X, raw_score=True, num_iteration=later_iter - 1))
    gradN, hessN = gh_fn(raw_prev, labels, num_class)
    with open(os.path.join(out_dir, f"{prefix}_gh_iterN.txt"), "w") as fh:
        fh.write(f"# {prefix}_gh_iterN — iter-{later_iter} per-row/per-class grad/hess; "
                 f"class-major f32 bits; GRAD then HESS\n")
        fh.write(f"# iter={later_iter} inits={list(inits)!r}\n")
        fh.write("GRAD " + f32_bits_line(gradN) + "\n")
        fh.write("HESS " + f32_bits_line(hessN) + "\n")

    # L3: per-round metrics (multi_logloss).
    with open(os.path.join(out_dir, f"{prefix}_metrics.txt"), "w") as fh:
        fh.write(f"# {prefix}_metrics — per-round {' + '.join(metric_names)} "
                 f"(record_evaluation); f64 bits\n")
        for metric in metric_names:
            vals = evals_result["training"][metric]
            fh.write(f"{metric} " + f64_bits_line(vals) + "\n")


def write_layered(out_dir, prefix, booster, X, labels, evals_result,
                  metric_names, gh_fn, init, pred_transformed):
    """Write the L1-L5 goldens for one objective cell.

    gh_fn(score_prev, labels) -> (grad f32, hess f32) is the objective's
    score-derivation g/h (the RESEARCH L1 recommendation: derive g/h from the
    captured per-iter score, no fobj override of a built-in objective)."""
    # L5: model text + transformed predict.
    booster.save_model(os.path.join(out_dir, f"{prefix}_spine_model.txt"))
    with open(os.path.join(out_dir, f"{prefix}_spine_pred.txt"), "w") as fh:
        fh.write(f"# {prefix}_spine_pred — transformed predict() f64 bits\n")
        fh.write(f64_bits_line(pred_transformed) + "\n")

    # L2: per-iter raw score.
    with open(os.path.join(out_dir, f"{prefix}_scores.txt"), "w") as fh:
        fh.write(f"# {prefix}_scores — per-iter raw score; one line per k=1..N; f64 bits\n")
        fh.write(f"# num_iterations={NUM_ITERATIONS} num_data={len(labels)}\n")
        for k in range(1, NUM_ITERATIONS + 1):
            raw_k = booster.predict(X, raw_score=True, num_iteration=k)
            fh.write(f64_bits_line(raw_k) + "\n")

    # L1 iter 1: score_0 = init for every row (BoostFromAverage).
    score0 = np.full(len(labels), init, dtype=np.float64)
    grad1, hess1 = gh_fn(score0, labels)
    with open(os.path.join(out_dir, f"{prefix}_gh_iter1.txt"), "w") as fh:
        fh.write(f"# {prefix}_gh_iter1 — iter-1 per-row grad/hess; f32 bits; GRAD then HESS\n")
        fh.write("GRAD " + f32_bits_line(grad1) + "\n")
        fh.write("HESS " + f32_bits_line(hess1) + "\n")

    # L1 later iter.
    score_prev = np.asarray(booster.predict(X, raw_score=True, num_iteration=LATER_ITER - 1))
    gradN, hessN = gh_fn(score_prev, labels)
    with open(os.path.join(out_dir, f"{prefix}_gh_iterN.txt"), "w") as fh:
        fh.write(f"# {prefix}_gh_iterN — iter-{LATER_ITER} per-row grad/hess; f32 bits; GRAD then HESS\n")
        fh.write(f"# iter={LATER_ITER} init={init!r}\n")
        fh.write("GRAD " + f32_bits_line(gradN) + "\n")
        fh.write("HESS " + f32_bits_line(hessN) + "\n")

    # L3: per-round metrics.
    with open(os.path.join(out_dir, f"{prefix}_metrics.txt"), "w") as fh:
        fh.write(f"# {prefix}_metrics — per-round {' + '.join(metric_names)} (record_evaluation); f64 bits\n")
        for metric in metric_names:
            vals = evals_result["training"][metric]
            fh.write(f"{metric} " + f64_bits_line(vals) + "\n")


def l2_gh(score_prev, labels):
    grad = (np.asarray(score_prev) - labels).astype(np.float32)
    hess = np.ones(len(labels), dtype=np.float32)
    return grad, hess


def l1_gh(score_prev, labels):
    diff = np.asarray(score_prev) - labels
    grad = np.sign(diff).astype(np.float32)
    hess = np.ones(len(labels), dtype=np.float32)
    return grad, hess


def binary_gh(score_prev, labels, sigmoid=1.0):
    s = np.asarray(score_prev, dtype=np.float64)
    label_val = np.where(labels > 0, 1.0, -1.0)
    response = -label_val * sigmoid / (1.0 + np.exp(label_val * sigmoid * s))
    abs_resp = np.abs(response)
    grad = response.astype(np.float32)
    hess = (abs_resp * (sigmoid - abs_resp)).astype(np.float32)
    return grad, hess


# ---- OBJ-04 family A gradient/hessian (huber/fair/quantile/mape) ----
def huber_gh(alpha):
    """RegressionHuberLoss::GetGradients (regression_objective.hpp:312): grad clamped
    to ±alpha in f64 then cast f32; hess = 1."""
    def gh(score_prev, labels):
        diff = np.asarray(score_prev, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
        clamped = np.where(np.abs(diff) <= alpha, diff, np.sign(diff) * alpha)
        return clamped.astype(np.float32), np.ones(len(labels), dtype=np.float32)
    return gh


def fair_gh(c):
    """RegressionFairLoss::GetGradients (regression_objective.hpp:364): grad =
    c*x/(|x|+c); hess = c²/(|x|+c)²."""
    def gh(score_prev, labels):
        x = np.asarray(score_prev, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
        denom = np.abs(x) + c
        grad = (c * x / denom).astype(np.float32)
        hess = (c * c / (denom * denom)).astype(np.float32)
        return grad, hess
    return gh


def quantile_gh(alpha):
    """RegressionQuantileloss::GetGradients (regression_objective.hpp:494): alpha is a
    score_t (f32); delta is the f32-cast residual; grad = (1-alpha) if delta>=0 else
    -alpha; hess = 1."""
    a32 = np.float32(alpha)
    def gh(score_prev, labels):
        delta = (np.asarray(score_prev, dtype=np.float64)
                 - np.asarray(labels, dtype=np.float64)).astype(np.float32)
        grad = np.where(delta >= np.float32(0.0), np.float32(1.0) - a32, -a32).astype(np.float32)
        return grad, np.ones(len(labels), dtype=np.float32)
    return gh


def mape_label_weight(labels):
    """RegressionMAPELOSS label_weight = 1/max(1,|label|), computed in f32."""
    lab = np.asarray(labels, dtype=np.float32)
    return (np.float32(1.0) / np.maximum(np.float32(1.0), np.abs(lab))).astype(np.float32)


def mape_gh(score_prev, labels):
    """RegressionMAPELOSS::GetGradients (regression_objective.hpp:619): grad =
    Sign(score-label) * label_weight; hess = 1."""
    diff = np.asarray(score_prev, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    lw = mape_label_weight(labels)
    grad = (np.sign(diff) * lw.astype(np.float64)).astype(np.float32)
    return grad, np.ones(len(labels), dtype=np.float32)


def weighted_median_percentile(values, weights, alpha=0.5):
    """Reproduce the C++ WeightedPercentileFun (regression_objective.hpp:50-88), used
    by the mape BoostFromScore (weighted label median) and quantile/mape renew."""
    cnt = len(values)
    if cnt <= 1:
        return float(values[0]) if cnt == 1 else 0.0
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    cdf = np.cumsum(w[order])
    threshold = cdf[-1] * alpha
    # upper_bound: first index with cdf > threshold.
    pos = int(np.searchsorted(cdf, threshold, side="right"))
    pos = min(pos, cnt - 1)
    if pos == 0 or pos == cnt - 1:
        return float(v[order[pos]])
    v1 = v[order[pos - 1]]
    v2 = v[order[pos]]
    if cdf[pos + 1] - cdf[pos] >= 1.0:
        return float((threshold - cdf[pos]) / (cdf[pos + 1] - cdf[pos]) * (v2 - v1) + v1)
    return float(v2)


def median_percentile(values, alpha=0.5):
    """Reproduce the C++ PercentileFun median (the BoostFromScore L1 init)."""
    cnt = len(values)
    if cnt <= 1:
        return float(values[0]) if cnt == 1 else 0.0
    desc = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    float_pos = (cnt - 1) * (1.0 - alpha)
    pos = int(float_pos) + 1
    if pos < 1:
        return float(desc[0])
    if pos >= cnt:
        return float(desc[cnt - 1])
    bias = float_pos - (pos - 1)
    v1 = desc[pos - 1]
    v2 = desc[pos]
    return float(v1 - (v1 - v2) * bias)


# ---- OBJ-04/05 exp/log family gradient/hessian (poisson/gamma/tweedie/xentropy) ----
def poisson_gh(max_delta_step):
    """RegressionPoissonLoss::GetGradients (regression_objective.hpp:439): grad =
    exp(score)-label; hess = exp(score)*exp(max_delta_step)."""
    emds = float(np.exp(max_delta_step))
    def gh(score_prev, labels):
        es = np.exp(np.asarray(score_prev, dtype=np.float64))
        grad = (es - np.asarray(labels, dtype=np.float64)).astype(np.float32)
        hess = (es * emds).astype(np.float32)
        return grad, hess
    return gh


def gamma_gh(score_prev, labels):
    """RegressionGammaLoss::GetGradients (regression_objective.hpp:694): exp_score =
    exp(-score); grad = 1 - label*exp_score; hess = label*exp_score."""
    es = np.exp(-np.asarray(score_prev, dtype=np.float64))
    lab = np.asarray(labels, dtype=np.float64)
    grad = (1.0 - lab * es).astype(np.float32)
    hess = (lab * es).astype(np.float32)
    return grad, hess


def tweedie_gh(rho):
    """RegressionTweedieLoss::GetGradients (regression_objective.hpp:729): exp_1 =
    exp((1-rho)*score); exp_2 = exp((2-rho)*score); grad = -label*exp_1 + exp_2;
    hess = -label*(1-rho)*exp_1 + (2-rho)*exp_2."""
    def gh(score_prev, labels):
        s = np.asarray(score_prev, dtype=np.float64)
        lab = np.asarray(labels, dtype=np.float64)
        e1 = np.exp((1.0 - rho) * s)
        e2 = np.exp((2.0 - rho) * s)
        grad = (-lab * e1 + e2).astype(np.float32)
        hess = (-lab * (1.0 - rho) * e1 + (2.0 - rho) * e2).astype(np.float32)
        return grad, hess
    return gh


def cross_entropy_gh(score_prev, labels):
    """CrossEntropy::GetGradients (xentropy_objective.hpp:98), stable log1pexp form."""
    s = np.asarray(score_prev, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.float64)
    grad = np.empty_like(s)
    hess = np.empty_like(s)
    hi = s > -37.0
    et = np.exp(-s[hi])
    grad[hi] = ((1.0 - lab[hi]) - lab[hi] * et) / (1.0 + et)
    hess[hi] = et / ((1.0 + et) * (1.0 + et))
    lo = ~hi
    el = np.exp(s[lo])
    grad[lo] = el - lab[lo]
    hess[lo] = el
    return grad.astype(np.float32), hess.astype(np.float32)


def cross_entropy_lambda_gh(score_prev, labels):
    """CrossEntropyLambda::GetGradients (xentropy_objective.hpp:215): z = 1/(1+exp(-s));
    grad = z-label; hess = z*(1-z)."""
    s = np.asarray(score_prev, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.float64)
    z = 1.0 / (1.0 + np.exp(-s))
    grad = (z - lab).astype(np.float32)
    hess = (z * (1.0 - z)).astype(np.float32)
    return grad, hess


# ============================ D-07 cross-product matrix ============================
# The full ~40-cell matrix: 5 core objectives × {bagging on/off} × {early_stop
# on/off} × {boost_from_average on/off} = 8 cells/objective. The spine cell
# (bagging off / es off / bfa on) IS the per-objective spine golden captured above —
# referenced, NOT re-captured (the ONLY safe automatic collapse, RESEARCH
# §Cross-Product Collapse Analysis cell 1). The remaining 7 cells/objective are
# captured here as end-to-end model-text + predict goldens.
#
# Axis exercising (RESEARCH §Cross-Product Collapse Analysis):
#   - bfa: the corpora have |init_score| > kEps (non-zero label mean / pavg), so the
#     bfa axis is genuinely distinct (no label-mean-zero collapse).
#   - bagging: bagging_fraction=0.7 bagging_freq=1 (re-bag every iter so the RNG
#     stream advances — the most order-sensitive path).
#   - early_stop: a VALID set whose labels plateau the metric, so early stopping
#     GENUINELY FIRES (best_iteration < num_iterations) — not a byte-identical
#     es-off sibling.
MATRIX_NUM_ITERATIONS = 12
MATRIX_BAGGING_FRACTION = 0.7
MATRIX_BAGGING_FREQ = 1
MATRIX_EARLY_STOPPING_ROUND = 2


def matrix_valid_corpus(X):
    """A validation set that plateaus the metric so early stopping FIRES: same
    features as train, but CONSTANT labels (the ensemble cannot keep improving valid
    loss after the first couple of trees)."""
    return X.copy(), np.full(X.shape[0], 10.0, dtype=np.float64)


def cell_tag(bag, es, bfa):
    return "bag{}_es{}_bfa{}".format(int(bag), int(es), int(bfa))


def capture_matrix_cell(out_dir, objective, prefix, X, labels, num_class, seed,
                        bag, es, bfa):
    """Capture ONE D-07 cell (model text + predict) for the given objective and the
    (bagging, early_stop, bfa) axis settings. Returns the realized best_iteration."""
    metric_map = {
        "regression": ["l2"],
        "regression_l1": ["l1"],
        "binary": ["binary_logloss"],
        "multiclass": ["multi_logloss"],
        "multiclassova": ["multi_logloss"],
    }
    metrics = metric_map[objective]
    p = base_params(seed, objective, metrics)
    p["boost_from_average"] = bool(bfa)
    if num_class > 1:
        p["num_class"] = num_class
    if bag:
        p["bagging_fraction"] = MATRIX_BAGGING_FRACTION
        p["bagging_freq"] = MATRIX_BAGGING_FREQ
        p["bagging_seed"] = 3
    # The multiclass objectives cap the horizon at MULTICLASS_NUM_ITERATIONS (the
    # 06-04 exp-libm bit-exact horizon — past iter ~5 the redundant-form softmax exp
    # flips a knife-edge split). Single-output cells run the full MATRIX_NUM_ITERATIONS.
    num_rounds = MULTICLASS_NUM_ITERATIONS if num_class > 1 else MATRIX_NUM_ITERATIONS
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    valid_sets = [dtrain]
    valid_names = ["training"]
    callbacks = []
    if es:
        Xv, lv = matrix_valid_corpus(X)
        # The valid Dataset takes its binning from `reference=dtrain` (NO params — a
        # second params dict with bin_construct_sample_cnt would re-trigger the
        # "Cannot change bin_construct_sample_cnt after constructed" Fatal).
        dvalid = lgb.Dataset(Xv, label=lv, reference=dtrain, free_raw_data=False)
        dvalid.construct()
        valid_sets = [dtrain, dvalid]
        valid_names = ["training", "valid_0"]
        callbacks.append(lgb.early_stopping(MATRIX_EARLY_STOPPING_ROUND, verbose=False))
    booster = lgb.train(
        p, dtrain, num_boost_round=num_rounds,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    tag = cell_tag(bag, es, bfa)
    booster.save_model(os.path.join(out_dir, f"{prefix}_{tag}_model.txt"))
    if num_class > 1:
        pred = booster.predict(X)
        pred_flat = _class_major(pred)
    else:
        pred = booster.predict(X, raw_score=True)
        pred_flat = np.asarray(pred)
    with open(os.path.join(out_dir, f"{prefix}_{tag}_pred.txt"), "w") as fh:
        fh.write(f"# {prefix}_{tag}_pred — predict() (raw for single-output / "
                 f"class-major prob for multiclass); f64 bits\n")
        fh.write(f"# best_iteration={booster.best_iteration}\n")
        fh.write(f64_bits_line(pred_flat) + "\n")
    return booster.best_iteration


def capture_matrix(out_dir, seed):
    """Capture the full ~40-cell D-07 cross-product (excluding the spine cell, which
    is referenced). Single-output objectives use the spine/binary corpora; the
    multiclass objectives use the multiclass corpus (5-iter exp-libm horizon does NOT
    apply to the matrix's bit-exact model — the matrix replays within ORACLE_TOL for
    multiclass cells, bit-exact for single-output, mirroring 06-04)."""
    Xs, ls, _ = spine_corpus()
    Xb, lb, _ = binary_corpus()
    Xm, lm, _ = multiclass_corpus()
    objectives = [
        ("regression", "regression", Xs, ls, 1),
        ("regression_l1", "regression_l1", Xs, ls, 1),
        ("binary", "binary", Xb, lb, 1),
        ("multiclass", "multiclass", Xm, lm, 3),
        ("multiclassova", "multiclassova", Xm, lm, 3),
    ]
    best_iters = {}
    for objective, prefix, X, labels, num_class in objectives:
        for bag in (False, True):
            for es in (False, True):
                for bfa in (False, True):
                    # The spine cell (bag off / es off / bfa on) is the referenced
                    # collapse — skip (single-output objectives have a *_spine_model
                    # golden; for the matrix the spine reference is documented).
                    if (not bag) and (not es) and bfa:
                        continue
                    bi = capture_matrix_cell(
                        out_dir, objective, prefix, X, labels, num_class, seed,
                        bag, es, bfa,
                    )
                    best_iters[f"{prefix}_{cell_tag(bag, es, bfa)}"] = bi
    return best_iters


# ====================== OBJ-04 family A (huber/fair/quantile/mape) ======================
# Each objective is a vertical slice on the proven spine corpus. Per RESEARCH
# §"Per-Subsystem Oracle Axis Matrix → Objectives" (07-RESEARCH.md:319-328):
#   huber    : 8 loop cells (bag×es×bfa) × alpha ∈ {0.9 default, 0.5}
#   fair     : 8 × fair_c ∈ {1.0, 2.0}
#   quantile : 8 × alpha ∈ {0.9, 0.1}  (RenewTreeOutput percentile; bag cells gate D-05)
#   mape     : 8                       (subclasses L1; bag cells gate D-05)
# The DEFAULT-param spine cell (bag off / es off / bfa on) is captured as the
# layered L1-L5 golden (write_layered). The remaining 7 loop cells + the
# non-default param-axis cells are captured as model+predict goldens via
# capture_matrix_cell, tagged with the objective-param suffix where it deviates from
# the default. The spine corpus labels are 2..20 (all |label| >= 1) so MAPE is stable
# (no <1 rounding warning) and quantile/huber find real splits.
FAMILY_A_PARAM_AXIS = {
    # objective -> (param_name, [default_value, alt_value])
    "huber": ("alpha", [0.9, 0.5]),
    "fair": ("fair_c", [1.0, 2.0]),
    "quantile": ("alpha", [0.9, 0.1]),
    "mape": (None, [None]),
}

FAMILY_A_METRIC = {
    "huber": ["huber", "l2"],
    "fair": ["fair", "l2"],
    "quantile": ["quantile", "l2"],
    "mape": ["mape", "l2"],
}


def family_a_gh(objective, param_value):
    if objective == "huber":
        return huber_gh(param_value)
    if objective == "fair":
        return fair_gh(param_value)
    if objective == "quantile":
        return quantile_gh(param_value)
    if objective == "mape":
        return mape_gh
    raise ValueError(objective)


def family_a_init(objective, labels, param_value):
    """The objective's BoostFromScore on the spine labels."""
    if objective in ("huber", "fair"):
        return float(np.mean(labels))               # inherit L2 mean
    if objective == "quantile":
        return median_percentile(labels, param_value)  # PercentileFun at alpha
    if objective == "mape":
        return weighted_median_percentile(labels, mape_label_weight(labels), 0.5)
    raise ValueError(objective)


def capture_family_a_cell(out_dir, objective, prefix, X, labels, seed,
                          param_name, param_value, bag, es, bfa):
    """Capture ONE family-A loop cell (model + predict) for the given param value and
    the (bag, es, bfa) axis. Returns the realized best_iteration."""
    p = base_params(seed, objective, FAMILY_A_METRIC[objective])
    p["boost_from_average"] = bool(bfa)
    if param_name is not None:
        p[param_name] = param_value
    if bag:
        p["bagging_fraction"] = MATRIX_BAGGING_FRACTION
        p["bagging_freq"] = MATRIX_BAGGING_FREQ
        p["bagging_seed"] = 3
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    valid_sets = [dtrain]
    valid_names = ["training"]
    callbacks = []
    if es:
        Xv, lv = matrix_valid_corpus(X)
        dvalid = lgb.Dataset(Xv, label=lv, reference=dtrain, free_raw_data=False)
        dvalid.construct()
        valid_sets = [dtrain, dvalid]
        valid_names = ["training", "valid_0"]
        callbacks.append(lgb.early_stopping(MATRIX_EARLY_STOPPING_ROUND, verbose=False))
    booster = lgb.train(
        p, dtrain, num_boost_round=MATRIX_NUM_ITERATIONS,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    booster.save_model(os.path.join(out_dir, f"{prefix}_model.txt"))
    pred = booster.predict(X, raw_score=True)
    with open(os.path.join(out_dir, f"{prefix}_pred.txt"), "w") as fh:
        fh.write(f"# {prefix}_pred — predict() raw score; f64 bits\n")
        fh.write(f"# best_iteration={booster.best_iteration}\n")
        fh.write(f64_bits_line(np.asarray(pred)) + "\n")
    return booster.best_iteration


def capture_family_a(out_dir, seed):
    """Capture all huber/fair/quantile/mape goldens: the layered L1-L5 default-spine
    cell + the {bag×es×bfa} loop cells + the objective param-axis cells. Returns the
    realized best_iteration map for the loop cells (es cells included)."""
    X, labels, mfq = spine_corpus()
    best_iters = {}
    for objective in ("huber", "fair", "quantile", "mape"):
        param_name, values = FAMILY_A_PARAM_AXIS[objective]
        default_value = values[0]
        metric = FAMILY_A_METRIC[objective]

        # --- layered L1-L5 default-spine cell (bag off / es off / bfa on) ---
        p = base_params(seed, objective, metric)
        p["boost_from_average"] = True
        if param_name is not None:
            p[param_name] = default_value
        d = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
        d.construct()
        assert_identity_binning(X, mfq)
        er = {}
        booster = lgb.train(
            p, d, num_boost_round=NUM_ITERATIONS,
            valid_sets=[d], valid_names=["training"],
            callbacks=[lgb.record_evaluation(er)],
        )
        init = family_a_init(objective, labels, default_value)
        gh = family_a_gh(objective, default_value)
        write_layered(out_dir, objective, booster, X, labels, er,
                      metric, gh, init, booster.predict(X, raw_score=True))

        # --- the 7 remaining {bag×es×bfa} loop cells at the DEFAULT param ---
        for bag in (False, True):
            for es in (False, True):
                for bfa in (False, True):
                    if (not bag) and (not es) and bfa:
                        continue  # the layered spine cell, referenced not re-captured
                    tag = cell_tag(bag, es, bfa)
                    prefix = f"{objective}_{tag}"
                    bi = capture_family_a_cell(
                        out_dir, objective, prefix, X, labels, seed,
                        param_name, default_value, bag, es, bfa,
                    )
                    best_iters[prefix] = bi

        # --- the objective param-axis: the ALT param value on the spine cell ---
        # (bag off / es off / bfa on), tagged `<objective>_<param><value>`.
        if param_name is not None and len(values) > 1:
            alt = values[1]
            alt_tag = f"{param_name}{alt}".replace(".", "p")
            prefix = f"{objective}_{alt_tag}"
            bi = capture_family_a_cell(
                out_dir, objective, prefix, X, labels, seed,
                param_name, alt, False, False, True,
            )
            best_iters[prefix] = bi
    return best_iters


# ================= OBJ-04/05 exp/log family (poisson/gamma/tweedie/xentropy) =================
# Per RESEARCH §"Per-Subsystem Oracle Axis Matrix → Objectives" (07-RESEARCH.md:319-332):
#   poisson              : 8 loop cells (bag×es×bfa) × poisson_max_delta_step ∈ {0.7, 0.1}
#   gamma                : 8
#   tweedie              : 8 × tweedie_variance_power ∈ {1.5, 1.1, 1.9}
#   cross_entropy        : 8 (label in [0,1])
#   cross_entropy_lambda : 8
# Every grad/hess + BoostFromScore + ConvertOutput uses a transcendental (exp/log),
# so the horizon is CAPPED at EXP_LOG_NUM_ITERATIONS (Pitfall 5) so every grown tree
# stays bit-exact; the predict-side ConvertOutput is the only ORACLE_TOL surface.
# poisson/gamma/tweedie use the spine corpus (labels 2..20, all >= 0, Σ != 0);
# cross_entropy/cross_entropy_lambda use the binary corpus (labels 0/1 ∈ [0,1]).
EXP_LOG_PARAM_AXIS = {
    # objective -> (param_name, [default_value, *alt_values])
    "poisson": ("poisson_max_delta_step", [0.7, 0.1]),
    "gamma": (None, [None]),
    "tweedie": ("tweedie_variance_power", [1.5, 1.1, 1.9]),
    "cross_entropy": (None, [None]),
    "cross_entropy_lambda": (None, [None]),
}


def exp_log_corpus(objective):
    """The corpus for an exp/log objective: spine (>=0 labels) for poisson/gamma/
    tweedie, binary (0/1 labels in [0,1]) for cross_entropy/cross_entropy_lambda."""
    if objective in ("cross_entropy", "cross_entropy_lambda"):
        return binary_corpus()
    return spine_corpus()


def exp_log_gh(objective, param_value):
    if objective == "poisson":
        return poisson_gh(param_value)
    if objective == "gamma":
        return gamma_gh
    if objective == "tweedie":
        return tweedie_gh(param_value)
    if objective == "cross_entropy":
        return cross_entropy_gh
    if objective == "cross_entropy_lambda":
        return cross_entropy_lambda_gh
    raise ValueError(objective)


def exp_log_init(objective, labels, param_value):
    """The objective's BoostFromScore on the corpus labels."""
    keps = 1e-15
    if objective in ("poisson", "gamma", "tweedie"):
        # SafeLog(L2 mean): log of the label mean (the log-mean score space).
        return float(np.log(np.mean(labels)))
    if objective == "cross_entropy":
        pavg = float(np.mean(labels))
        pavg = min(max(pavg, keps), 1.0 - keps)
        return float(np.log(pavg / (1.0 - pavg)))
    if objective == "cross_entropy_lambda":
        havg = float(np.mean(labels))
        return float(np.log(np.expm1(havg)))
    raise ValueError(objective)


def write_layered_capped(out_dir, prefix, booster, X, labels, evals_result,
                         metric_names, gh_fn, init, pred_transformed,
                         num_iterations, later_iter):
    """write_layered with an explicit (capped) horizon — the exp/log family caps the
    L2 per-iter score horizon at num_iterations (EXP_LOG_NUM_ITERATIONS, Pitfall 5)."""
    booster.save_model(os.path.join(out_dir, f"{prefix}_spine_model.txt"))
    with open(os.path.join(out_dir, f"{prefix}_spine_pred.txt"), "w") as fh:
        fh.write(f"# {prefix}_spine_pred — transformed predict() f64 bits "
                 f"(ConvertOutput; ORACLE_TOL surface)\n")
        fh.write(f64_bits_line(pred_transformed) + "\n")
    with open(os.path.join(out_dir, f"{prefix}_scores.txt"), "w") as fh:
        fh.write(f"# {prefix}_scores — per-iter raw score; one line per k=1..N; f64 "
                 f"bits; horizon CAPPED at {num_iterations} (Pitfall 5 exp-libm)\n")
        fh.write(f"# num_iterations={num_iterations} num_data={len(labels)}\n")
        for k in range(1, num_iterations + 1):
            raw_k = booster.predict(X, raw_score=True, num_iteration=k)
            fh.write(f64_bits_line(raw_k) + "\n")
    score0 = np.full(len(labels), init, dtype=np.float64)
    grad1, hess1 = gh_fn(score0, labels)
    with open(os.path.join(out_dir, f"{prefix}_gh_iter1.txt"), "w") as fh:
        fh.write(f"# {prefix}_gh_iter1 — iter-1 per-row grad/hess; f32 bits; GRAD then HESS\n")
        fh.write("GRAD " + f32_bits_line(grad1) + "\n")
        fh.write("HESS " + f32_bits_line(hess1) + "\n")
    score_prev = np.asarray(booster.predict(X, raw_score=True, num_iteration=later_iter - 1))
    gradN, hessN = gh_fn(score_prev, labels)
    with open(os.path.join(out_dir, f"{prefix}_gh_iterN.txt"), "w") as fh:
        fh.write(f"# {prefix}_gh_iterN — iter-{later_iter} per-row grad/hess; f32 bits; "
                 f"GRAD then HESS\n")
        fh.write(f"# iter={later_iter} init={init!r}\n")
        fh.write("GRAD " + f32_bits_line(gradN) + "\n")
        fh.write("HESS " + f32_bits_line(hessN) + "\n")
    with open(os.path.join(out_dir, f"{prefix}_metrics.txt"), "w") as fh:
        fh.write(f"# {prefix}_metrics — per-round {' + '.join(metric_names)} "
                 f"(record_evaluation); f64 bits\n")
        for metric in metric_names:
            vals = evals_result["training"][metric]
            fh.write(f"{metric} " + f64_bits_line(vals) + "\n")


def capture_exp_log_cell(out_dir, objective, prefix, X, labels, seed,
                         param_name, param_value, bag, es, bfa):
    """Capture ONE exp/log loop cell (model + predict) at the CAPPED horizon."""
    # The metric crate has no poisson/gamma/tweedie/xentropy metric; the parity replay
    # compares model leaf values + scores (not eval history), so capture with the
    # objective's NATIVE LightGBM metric only for record-keeping in the model header.
    p = base_params(seed, objective, [objective])
    p["boost_from_average"] = bool(bfa)
    if param_name is not None:
        p[param_name] = param_value
    if bag:
        p["bagging_fraction"] = MATRIX_BAGGING_FRACTION
        p["bagging_freq"] = MATRIX_BAGGING_FREQ
        p["bagging_seed"] = 3
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    valid_sets = [dtrain]
    valid_names = ["training"]
    callbacks = []
    if es:
        Xv, lv = matrix_valid_corpus(X)
        # For xentropy, the plateau valid labels must be in [0,1]; clamp the constant.
        if objective in ("cross_entropy", "cross_entropy_lambda"):
            lv = np.clip(lv, 0.0, 1.0)
            lv[:] = 1.0
        dvalid = lgb.Dataset(Xv, label=lv, reference=dtrain, free_raw_data=False)
        dvalid.construct()
        valid_sets = [dtrain, dvalid]
        valid_names = ["training", "valid_0"]
        callbacks.append(lgb.early_stopping(MATRIX_EARLY_STOPPING_ROUND, verbose=False))
    booster = lgb.train(
        p, dtrain, num_boost_round=EXP_LOG_NUM_ITERATIONS,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    booster.save_model(os.path.join(out_dir, f"{prefix}_model.txt"))
    pred = booster.predict(X, raw_score=True)
    with open(os.path.join(out_dir, f"{prefix}_pred.txt"), "w") as fh:
        fh.write(f"# {prefix}_pred — predict() raw score; f64 bits\n")
        fh.write(f"# best_iteration={booster.best_iteration}\n")
        fh.write(f64_bits_line(np.asarray(pred)) + "\n")
    return booster.best_iteration


def capture_exp_log(out_dir, seed):
    """Capture all poisson/gamma/tweedie/cross_entropy/cross_entropy_lambda goldens:
    the layered L1-L5 default-spine cell (capped horizon) + the {bag×es×bfa} loop
    cells + the objective param-axis cells. Returns the loop best_iteration map."""
    best_iters = {}
    for objective in ("poisson", "gamma", "tweedie",
                      "cross_entropy", "cross_entropy_lambda"):
        X, labels, mfq = exp_log_corpus(objective)
        param_name, values = EXP_LOG_PARAM_AXIS[objective]
        default_value = values[0]

        # --- layered L1-L5 default-spine cell (bag off / es off / bfa on) ---
        p = base_params(seed, objective, [objective])
        p["boost_from_average"] = True
        if param_name is not None:
            p[param_name] = default_value
        d = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
        d.construct()
        assert_identity_binning(X, mfq)
        er = {}
        booster = lgb.train(
            p, d, num_boost_round=EXP_LOG_NUM_ITERATIONS,
            valid_sets=[d], valid_names=["training"],
            callbacks=[lgb.record_evaluation(er)],
        )
        init = exp_log_init(objective, labels, default_value)
        gh = exp_log_gh(objective, default_value)
        write_layered_capped(out_dir, objective, booster, X, labels, er,
                             [objective], gh, init,
                             booster.predict(X), EXP_LOG_NUM_ITERATIONS,
                             EXP_LOG_LATER_ITER)

        # --- the 7 remaining {bag×es×bfa} loop cells at the DEFAULT param ---
        for bag in (False, True):
            for es in (False, True):
                for bfa in (False, True):
                    if (not bag) and (not es) and bfa:
                        continue  # the layered spine cell, referenced not re-captured
                    tag = cell_tag(bag, es, bfa)
                    prefix = f"{objective}_{tag}"
                    bi = capture_exp_log_cell(
                        out_dir, objective, prefix, X, labels, seed,
                        param_name, default_value, bag, es, bfa,
                    )
                    best_iters[prefix] = bi

        # --- the objective param-axis: each ALT param value on the spine cell ---
        if param_name is not None and len(values) > 1:
            for alt in values[1:]:
                alt_tag = f"{param_name}{alt}".replace(".", "p")
                prefix = f"{objective}_{alt_tag}"
                bi = capture_exp_log_cell(
                    out_dir, objective, prefix, X, labels, seed,
                    param_name, alt, False, False, True,
                )
                best_iters[prefix] = bi
    return best_iters


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s" % (lgb.__version__, version)
    )

    os.makedirs(out_dir, exist_ok=True)

    # ============================ regression (L2) spine ============================
    X, labels, expected_most_freq = spine_corpus()
    p = base_params(seed, "regression", ["l2", "rmse"])
    dtrain = lgb.Dataset(X, label=labels, free_raw_data=False)
    dtrain.construct()
    assert_identity_binning(X, expected_most_freq)
    evals_result = {}
    booster = lgb.train(
        p, dtrain, num_boost_round=NUM_ITERATIONS,
        valid_sets=[dtrain], valid_names=["training"],
        callbacks=[lgb.record_evaluation(evals_result)],
    )
    init = float(np.mean(labels))
    write_layered(out_dir, "regression", booster, X, labels, evals_result,
                  ["l2", "rmse"], l2_gh, init, booster.predict(X))

    # ============================ regression_l1 ============================
    Xl, ll, mfq = spine_corpus()  # continuous labels work for l1 too (D-08)
    pl = base_params(seed, "regression_l1", ["l1", "l2", "rmse"])
    dl = lgb.Dataset(Xl, label=ll, free_raw_data=False)
    dl.construct()
    er_l1 = {}
    b_l1 = lgb.train(
        pl, dl, num_boost_round=NUM_ITERATIONS,
        valid_sets=[dl], valid_names=["training"],
        callbacks=[lgb.record_evaluation(er_l1)],
    )
    init_l1 = median_percentile(ll, 0.5)  # L1 BoostFromScore = label median
    write_layered(out_dir, "regression_l1", b_l1, Xl, ll, er_l1,
                  ["l1", "l2", "rmse"], l1_gh, init_l1, b_l1.predict(Xl))
    # Per-leaf renew golden (Pitfall 2/3): the l1 leaf values ARE the median
    # residual, NOT the learner Newton output. The committed model text leaf values
    # (in regression_l1_spine_model.txt) carry this; the dedicated assertion in
    # boosting_parity replays them. No extra file needed — the model text IS the
    # renew golden (its leaf values are the renewed medians).

    # ============================ binary ============================
    Xb, lb, mfqb = binary_corpus()
    pb = base_params(seed, "binary", ["binary_logloss", "binary_error", "auc"])
    db = lgb.Dataset(Xb, label=lb, free_raw_data=False)
    db.construct()
    assert_identity_binning(Xb, mfqb)
    er_b = {}
    b_bin = lgb.train(
        pb, db, num_boost_round=NUM_ITERATIONS,
        valid_sets=[db], valid_names=["training"],
        callbacks=[lgb.record_evaluation(er_b)],
    )
    # binary BoostFromScore: logit of clamped pavg.
    pavg = float(np.mean(lb > 0))
    keps = 1e-15
    pavg = min(max(pavg, keps), 1.0 - keps)
    init_b = float(np.log(pavg / (1.0 - pavg)) / 1.0)
    write_layered(out_dir, "binary", b_bin, Xb, lb, er_b,
                  ["binary_logloss", "binary_error", "auc"], binary_gh,
                  init_b, b_bin.predict(Xb))

    # ============================ custom (OBJ-02 cross-anchor) ============================
    # The custom run uses a Python fobj that replicates the L2 objective's g/h
    # (grad = score - label, hess = 1) — chosen DELIBERATELY so the custom run is
    # cross-anchorable to the native regression(L2) cell. Because custom forces
    # boost_from_average OFF (obj == null), we capture the L2 reference cell with
    # boost_from_average=False so the two are genuinely comparable (same g/h, same
    # init=0 => same trees => bit-identical model text). See REFERENCE_MANIFEST.md.
    Xc, lc, mfqc = spine_corpus()

    def fobj(preds, dataset):
        # LightGBM 4.x custom objective (params["objective"] = callable):
        # signature (y_pred, dataset) -> (grad, hess).
        y = dataset.get_label()
        grad = (preds - y).astype(np.float64)
        hess = np.ones(len(y), dtype=np.float64)
        return grad, hess

    def feval(preds, dataset):
        y = dataset.get_label()
        return ("l2", float(np.mean((preds - y) ** 2)), False)

    pc = base_params(seed, "regression", ["l2"])
    pc["boost_from_average"] = False   # custom forces bfa OFF
    pc["objective"] = fobj             # custom objective callable (4.x API)
    dc = lgb.Dataset(Xc, label=lc, free_raw_data=False)
    dc.construct()
    er_c = {}
    b_custom = lgb.train(
        pc, dc, num_boost_round=NUM_ITERATIONS,
        valid_sets=[dc], valid_names=["training"],
        feval=feval,
        callbacks=[lgb.record_evaluation(er_c)],
    )
    write_layered(out_dir, "custom", b_custom, Xc, lc, er_c, ["l2"],
                  l2_gh, 0.0, b_custom.predict(Xc, raw_score=True))

    # The cross-anchor L2 reference cell: native regression(L2) with bfa OFF (so the
    # init is 0, matching custom) — its model text must bit-match custom's.
    p_ref = base_params(seed, "regression", ["l2"])
    p_ref["boost_from_average"] = False
    d_ref = lgb.Dataset(Xc, label=lc, free_raw_data=False)
    d_ref.construct()
    b_ref = lgb.train(p_ref, d_ref, num_boost_round=NUM_ITERATIONS)
    b_ref.save_model(os.path.join(out_dir, "custom_crossanchor_l2_model.txt"))

    # ============================ multiclass (softmax) ============================
    Xm, lm, mfqm = multiclass_corpus()
    num_class = 3
    keps = 1e-15
    pm = base_params(seed, "multiclass", ["multi_logloss"])
    pm["num_class"] = num_class
    dm = lgb.Dataset(Xm, label=lm, free_raw_data=False)
    dm.construct()
    assert_identity_binning(Xm, mfqm)
    er_m = {}
    b_mc = lgb.train(
        pm, dm, num_boost_round=MULTICLASS_NUM_ITERATIONS,
        valid_sets=[dm], valid_names=["training"],
        callbacks=[lgb.record_evaluation(er_m)],
    )
    # Per-class init: BoostFromScore(k) = log(max(kEps, class_init_probs_[k])).
    class_counts = np.bincount(lm.astype(int), minlength=num_class).astype(np.float64)
    class_probs = class_counts / len(lm)
    inits_mc = [float(np.log(max(keps, class_probs[k]))) for k in range(num_class)]
    write_layered_multiclass(
        out_dir, "multiclass", b_mc, Xm, lm, er_m, ["multi_logloss"],
        multiclass_gh, inits_mc, num_class, len(lm),
        num_iterations=MULTICLASS_NUM_ITERATIONS, later_iter=MULTICLASS_LATER_ITER,
    )

    # ============================ multiclassova ============================
    Xo, lo, mfqo = multiclass_corpus()
    po = base_params(seed, "multiclassova", ["multi_logloss"])
    po["num_class"] = num_class
    do = lgb.Dataset(Xo, label=lo, free_raw_data=False)
    do.construct()
    assert_identity_binning(Xo, mfqo)
    er_o = {}
    b_ova = lgb.train(
        po, do, num_boost_round=MULTICLASS_NUM_ITERATIONS,
        valid_sets=[do], valid_names=["training"],
        callbacks=[lgb.record_evaluation(er_o)],
    )
    # Per-class init: each class's binary BoostFromScore = logit(clamped pavg_i),
    # pavg_i = count(label == i) / num_data.
    inits_ova = []
    for i in range(num_class):
        pavg_i = float(np.mean(lo.astype(int) == i))
        pavg_i = min(max(pavg_i, keps), 1.0 - keps)
        inits_ova.append(float(np.log(pavg_i / (1.0 - pavg_i)) / 1.0))
    write_layered_multiclass(
        out_dir, "multiclassova", b_ova, Xo, lo, er_o, ["multi_logloss"],
        lambda raw, lab, nc: multiclassova_gh(raw, lab, nc, sigmoid=1.0),
        inits_ova, num_class, len(lo),
        num_iterations=MULTICLASS_NUM_ITERATIONS, later_iter=MULTICLASS_LATER_ITER,
    )

    # ===================== metric_freq>1 + early_stopping (CR-02) =====================
    # A regression cell with metric_freq=2 + early_stopping_round=2 on the plateau
    # valid set (matrix_valid_corpus). C++ runs the valid-eval + ES decision EVERY
    # iter when ES is on (gbdt.cpp:574) — metric_freq=2 only thins the recorded
    # eval-history, NOT the ES cadence. This golden catches the CR-02 caller bug
    # (gating early.update behind do_eval): the Rust best_iteration / trimmed
    # tree-count must equal the captured value.
    Xmf, lmf, mfq_mf = spine_corpus()
    p_mf = base_params(seed, "regression", ["l2"])
    p_mf["metric_freq"] = 2
    d_mf = lgb.Dataset(Xmf, label=lmf, free_raw_data=False)
    d_mf.construct()
    Xv_mf, lv_mf = matrix_valid_corpus(Xmf)
    dv_mf = lgb.Dataset(Xv_mf, label=lv_mf, reference=d_mf, free_raw_data=False)
    dv_mf.construct()
    b_mf = lgb.train(
        p_mf, d_mf, num_boost_round=MATRIX_NUM_ITERATIONS,
        valid_sets=[d_mf, dv_mf], valid_names=["training", "valid_0"],
        callbacks=[lgb.early_stopping(MATRIX_EARLY_STOPPING_ROUND, verbose=False)],
    )
    b_mf.save_model(os.path.join(out_dir, "regression_mf2es_model.txt"))
    with open(os.path.join(out_dir, "regression_mf2es_pred.txt"), "w") as fh:
        fh.write("# regression_mf2es_pred — predict() raw score; f64 bits\n")
        fh.write(f"# best_iteration={b_mf.best_iteration}\n")
        fh.write(f64_bits_line(np.asarray(b_mf.predict(Xmf, raw_score=True))) + "\n")
    with open(os.path.join(out_dir, "regression_mf2es_best_iteration.txt"), "w") as fh:
        fh.write("# regression_mf2es — metric_freq=2 + early_stopping_round=2 "
                 "(plateau valid); the ES decision runs every iter (gbdt.cpp:574)\n")
        fh.write(f"best_iteration={b_mf.best_iteration}\n")

    # ===================== regression reg_sqrt=1 (GAP E / OBJ-03) =====================
    # reg_sqrt pre-transforms the label `Sign(label)*sqrt(|label|)`, fits L2 on the
    # transformed target, and ConvertOutput inverts via `Sign(x)*x*x`. No committed
    # golden previously exercised reg_sqrt=1 (all goldens [reg_sqrt: 0]). The gh_fn
    # applies the SAME sqrt pre-transform before the L2 grad/hess; the predict golden
    # carries the ConvertOutput inverse (booster.predict applies it).
    Xsq, lsq, mfq_sq = spine_corpus()
    p_sq = base_params(seed, "regression", ["l2"])
    p_sq["reg_sqrt"] = True
    d_sq = lgb.Dataset(Xsq, label=lsq, free_raw_data=False)
    d_sq.construct()
    er_sq = {}
    b_sq = lgb.train(
        p_sq, d_sq, num_boost_round=NUM_ITERATIONS,
        valid_sets=[d_sq], valid_names=["training"],
        callbacks=[lgb.record_evaluation(er_sq)],
    )
    # reg_sqrt L2 g/h: fit on the sqrt-transformed label.
    def sqrt_transform(lab):
        a = np.asarray(lab, dtype=np.float64)
        return np.sign(a) * np.sqrt(np.abs(a))

    def reg_sqrt_gh(score_prev, lab):
        # L2 grad/hess against the TRANSFORMED label (the C++ reg_sqrt path).
        tl = sqrt_transform(lab)
        grad = (np.asarray(score_prev) - tl).astype(np.float32)
        hess = np.ones(len(tl), dtype=np.float32)
        return grad, hess
    # BoostFromScore for reg_sqrt: mean of the TRANSFORMED labels.
    init_sq = float(np.mean(sqrt_transform(lsq)))
    write_layered(out_dir, "regression_sqrt", b_sq, Xsq, lsq, er_sq,
                  ["l2"], reg_sqrt_gh, init_sq, b_sq.predict(Xsq))

    # ============================ D-07 cross-product matrix ============================
    best_iters = capture_matrix(out_dir, seed)
    with open(os.path.join(out_dir, "matrix_best_iterations.txt"), "w") as fh:
        fh.write("# D-07 matrix: realized best_iteration per cell "
                 "(<prefix>_bag<B>_es<E>_bfa<F> best_iteration=<n>). "
                 "es cells with best_iteration < %d FIRED.\n" % MATRIX_NUM_ITERATIONS)
        for cell in sorted(best_iters):
            fh.write("%s best_iteration=%d\n" % (cell, best_iters[cell]))

    # ===================== OBJ-04 family A (huber/fair/quantile/mape) =====================
    fa_best_iters = capture_family_a(out_dir, seed)
    with open(os.path.join(out_dir, "family_a_best_iterations.txt"), "w") as fh:
        fh.write("# OBJ-04 family A (07-02): realized best_iteration per loop cell "
                 "(<objective>_bag<B>_es<E>_bfa<F> best_iteration=<n>). "
                 "es cells with best_iteration < %d FIRED.\n" % MATRIX_NUM_ITERATIONS)
        for cell in sorted(fa_best_iters):
            fh.write("%s best_iteration=%d\n" % (cell, fa_best_iters[cell]))

    # ============= OBJ-04/05 exp/log family (poisson/gamma/tweedie/xentropy) =============
    el_best_iters = capture_exp_log(out_dir, seed)
    with open(os.path.join(out_dir, "exp_log_best_iterations.txt"), "w") as fh:
        fh.write("# OBJ-04/05 exp/log family (07-03): realized best_iteration per loop "
                 "cell (<objective>_bag<B>_es<E>_bfa<F> best_iteration=<n>). Horizon "
                 "CAPPED at %d (Pitfall 5 exp-libm).\n" % EXP_LOG_NUM_ITERATIONS)
        for cell in sorted(el_best_iters):
            fh.write("%s best_iteration=%d\n" % (cell, el_best_iters[cell]))

    print("boosting_oracle_capture: wrote regression/regression_l1/binary/custom/"
          "multiclass/multiclassova L1-L5 goldens + regression_sqrt (reg_sqrt=1) + "
          "regression_mf2es (metric_freq=2 + early_stopping) + the D-07 matrix + "
          "the OBJ-04 family-A (huber/fair/quantile/mape) goldens + the OBJ-04/05 "
          "exp/log family (poisson/gamma/tweedie/cross_entropy/cross_entropy_lambda) "
          "goldens to %s" % out_dir)


if __name__ == "__main__":
    main()
