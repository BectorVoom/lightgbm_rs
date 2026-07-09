#!/usr/bin/env python3
"""Phase-7 W3 (MET-03) REAL lib_lightgbm extended-metric oracle capture (plan 07-04).

Captures real-binary golden values for the extended evaluation metrics added in
07-04 — regression/xentropy: quantile/huber/fair/poisson/mape/gamma/gamma_deviance/
tweedie/cross_entropy/cross_entropy_lambda/kullback_leibler; multiclass:
multi_error/auc_mu; binary: average_precision — so the Rust `Metric::eval` /
`XentropyMetric::eval` / `MultiError::eval` / `AucMu::eval` / `BinaryMetric::eval`
can be replayed against the authoritative `lib_lightgbm` 4.6 metric value.

Metrics are pure evaluation functions (score + label -> number); they do NOT train.
So for each metric we:
  1. Train a tiny model with a COMPATIBLE objective (so the score is on the right
     scale and the metric's interval/positivity preconditions hold).
  2. Dump the final-iteration RAW score (`predict(raw_score=True)`) — class-major
     for the multiclass metrics — as the input the Rust eval consumes.
  3. Dump the labels.
  4. Dump the real-binary metric value from `record_evaluation` at the final round.

The Rust parity test reads (scores, labels), runs the Rust eval, and asserts it
matches the captured metric value within ORACLE_TOL (bit-exact where the algorithm
permits). Because the metric is independent of the learner-level split knife-edge
(DEF-07-02), even the fair/gamma/tweedie metrics reach faithful parity: the capture
supplies the scores; the metric just evaluates them.

DETERMINISM / IDEMPOTENCY: trained `deterministic=true force_row_wise=true
num_threads=1 seed=<seed>` with NO subsampling, so re-running produces
byte-identical goldens (empty `git diff`).

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/metric/`.

Usage:
  metric_oracle_capture.py <out_dir> <seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb

NUM_ITERATIONS = 5
NUM_LEAVES = 4
LEARNING_RATE = 0.1


def f64_bits_line(values):
    """Serialize a list of f64 as space-separated u64 bit patterns (exact)."""
    return " ".join(
        str(struct.unpack("<Q", struct.pack("<d", float(v))[:8])[0]) for v in values
    )


def f32_bits_line(values):
    """Serialize a list of f32 as space-separated u32 bit patterns (exact)."""
    return " ".join(
        str(struct.unpack("<I", struct.pack("<f", float(v)))[0]) for v in values
    )


def base_params(seed, objective, metric, num_class=None):
    p = {
        "objective": objective,
        "metric": metric,
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
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,
    }
    if num_class is not None:
        p["num_class"] = num_class
    return p


# ---------------------------------------------------------------------------
# Corpora. Features are 2-column identity-binned integers; labels obey each
# metric family's domain (positive for gamma/poisson/tweedie, [0,1] for xentropy).
# ---------------------------------------------------------------------------
F0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
F1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


def X():
    return np.array([F0, F1], dtype=np.float64).T


def regression_labels():
    # continuous spread (works for quantile/huber/fair/mape).
    return np.array(
        [2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0],
        dtype=np.float64,
    )


def positive_labels():
    # strictly positive (gamma/poisson/tweedie require label > 0).
    return np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        dtype=np.float64,
    )


def unit_interval_labels():
    # labels in [0, 1] (xentropy family preconditions).
    return np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.5],
        dtype=np.float64,
    )


def binary_labels():
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    )


def multiclass_labels():
    return np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0],
        dtype=np.float64,
    )


def _class_major(arr2d):
    """(num_data, num_class) row-major -> class-major 1-D (class0 rows, then class1...)."""
    return arr2d.T.reshape(-1)


def train_and_capture(out_dir, name, seed, objective, metric, labels,
                      num_class=None, multiclass=False):
    """Train, then dump <name>_scores.txt, <name>_labels.txt, <name>_value.txt."""
    Xd = X()
    dtrain = lgb.Dataset(Xd, label=labels, free_raw_data=False)
    evals = {}
    params = base_params(seed, objective, metric, num_class)
    num_round = NUM_ITERATIONS
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        valid_sets=[dtrain],
        valid_names=["training"],
        callbacks=[lgb.record_evaluation(evals)],
    )
    # Raw scores at the final iteration.
    raw = booster.predict(Xd, raw_score=True)
    if multiclass:
        # raw shape (num_data, num_class) -> class-major.
        scores = _class_major(np.asarray(raw, dtype=np.float64))
    else:
        scores = np.asarray(raw, dtype=np.float64).reshape(-1)

    # The captured metric value at the final round. The record_evaluation key is
    # the metric's canonical name as LightGBM reports it.
    metric_key = metric if isinstance(metric, str) else metric[0]
    train_hist = evals["training"]
    if metric_key not in train_hist:
        # LightGBM may report under a slightly different key; take the sole entry.
        keys = list(train_hist.keys())
        if len(keys) != 1:
            sys.exit(
                "ABORT: metric %s not found in eval history keys %s" % (metric_key, keys)
            )
        metric_key = keys[0]
    value = float(train_hist[metric_key][-1])

    with open(os.path.join(out_dir, f"{name}_scores.txt"), "w") as fh:
        fh.write(f"# {name} final-iter raw scores (class-major if multiclass); f64 bits\n")
        fh.write(f64_bits_line(scores) + "\n")
    with open(os.path.join(out_dir, f"{name}_labels.txt"), "w") as fh:
        fh.write(f"# {name} labels; f32 bits\n")
        fh.write(f32_bits_line(np.asarray(labels, dtype=np.float32)) + "\n")
    with open(os.path.join(out_dir, f"{name}_value.txt"), "w") as fh:
        fh.write(f"# {name} real-binary {metric_key} value at round {num_round}; f64 bits\n")
        fh.write(f64_bits_line([value]) + "\n")
    print(f"  {name}: {metric_key} = {value!r}")


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: metric_oracle_capture.py <out_dir> <seed> <lightgbm_version>")
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    expected_version = sys.argv[3]
    if lgb.__version__ != expected_version:
        sys.exit(
            "ABORT: lightgbm %s != recorded %s" % (lgb.__version__, expected_version)
        )
    os.makedirs(out_dir, exist_ok=True)

    reg = regression_labels()
    pos = positive_labels()
    unit = unit_interval_labels()

    # base regression losses (Pitfall 1 gap — these 4 device-supported goldens had
    # never been captured; the discriminator marks rmse/l2/l1/binary_logloss as
    # on-device supported so they MUST have a real lib_lightgbm anchor). Objective
    # `regression` (identity ConvertOutput) with the matching metric name; the raw
    # score IS the metric input for the pointwise regression losses.
    train_and_capture(out_dir, "rmse", seed, "regression", "rmse", reg)
    train_and_capture(out_dir, "l2", seed, "regression", "l2", reg)
    train_and_capture(out_dir, "l1", seed, "regression", "l1", reg)
    # binary_logloss uses objective `binary` (NOT regression) so its captured raw
    # score is on the pre-sigmoid scale the binary metric's inverse-link consumes.
    train_and_capture(out_dir, "binary_logloss", seed, "binary", "binary_logloss",
                      binary_labels())

    # regression-family metrics (identity-ConvertOutput objective: regression).
    train_and_capture(out_dir, "quantile", seed, "regression", "quantile", reg)
    train_and_capture(out_dir, "huber", seed, "regression", "huber", reg)
    train_and_capture(out_dir, "fair", seed, "regression", "fair", reg)
    train_and_capture(out_dir, "mape", seed, "regression", "mape", reg)

    # exp-ConvertOutput regression metrics (objective == metric family so the
    # captured raw score is on the log scale the metric exp()s back).
    train_and_capture(out_dir, "poisson", seed, "poisson", "poisson", pos)
    train_and_capture(out_dir, "gamma", seed, "gamma", "gamma", pos)
    train_and_capture(out_dir, "gamma_deviance", seed, "gamma", "gamma_deviance", pos)
    train_and_capture(out_dir, "tweedie", seed, "tweedie", "tweedie", pos)

    # xentropy family ([0,1] labels).
    train_and_capture(out_dir, "cross_entropy", seed, "cross_entropy",
                      "cross_entropy", unit)
    train_and_capture(out_dir, "cross_entropy_lambda", seed, "cross_entropy_lambda",
                      "cross_entropy_lambda", unit)
    train_and_capture(out_dir, "kullback_leibler", seed, "cross_entropy",
                      "kullback_leibler", unit)

    # binary: average_precision.
    train_and_capture(out_dir, "average_precision", seed, "binary",
                      "average_precision", binary_labels())

    # multiclass: multi_error, auc_mu.
    train_and_capture(out_dir, "multi_error", seed, "multiclass", "multi_error",
                      multiclass_labels(), num_class=3, multiclass=True)
    train_and_capture(out_dir, "auc_mu", seed, "multiclass", "auc_mu",
                      multiclass_labels(), num_class=3, multiclass=True)


if __name__ == "__main__":
    main()
