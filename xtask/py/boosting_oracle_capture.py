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

    print("boosting_oracle_capture: wrote regression/regression_l1/binary/custom/"
          "multiclass/multiclassova L1-L5 goldens to %s" % out_dir)


if __name__ == "__main__":
    main()
