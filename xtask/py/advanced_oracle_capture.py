#!/usr/bin/env python3
"""Phase-7 (plan 07-12, ADV-06 refit / ADV-07 importance) REAL lib_lightgbm
advanced-model-ops oracle capture.

Models the Phase-5/07-08/07-11 learner-oracle posture (the real pip `lib_lightgbm`
4.6 `save_model()` IS the authoritative v4 model text). It trains a small
deterministic MULTI-tree regression model on a shared numeric corpus and dumps,
under `crates/oracle-harness/tests/fixtures/advanced/`:

  - `base_model.txt`        — the v4 base model text (Rust loads it as the start),
  - `refit_decay09.txt`     — `Booster.refit(X, y, decay_rate=0.9)` model text,
  - `refit_decay00.txt`     — `Booster.refit(X, y, decay_rate=0.0)` model text,
  - `continue_model.txt`    — model after continuing training from base (init_model),
  - `importance.json`       — per-feature split-count + gain-sum vectors,
  - `advanced.json`         — shared sidecar (per-feature bins + bin_upper_bound +
                              num_bin + most_freq_bin + per-row label) so the Rust
                              replay routes rows + recomputes grad/hess identically.

OBJECTIVE = regression-l2, `boost_from_average=false`: at the refit's iteration 1
the score is 0 so `grad = -label`, `hess = 1` — matching the Rust `Gbdt::refit`
clean-accumulation start.

DETERMINISM / IDEMPOTENCY: every model trains with `deterministic=true
force_row_wise=true num_threads=1 seed=<seed> bagging_fraction=1.0
feature_fraction=1.0` and no subsampling => byte-identical re-runs.

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/advanced/`.

Usage:
  advanced_oracle_capture.py <out_dir> <seed> <lightgbm_version>
"""

import json
import os
import struct
import sys

import numpy as np

import lightgbm as lgb


def next_up(x):
    """The next f64 after x toward +inf (LightGBM `GetDoubleUpperBound` nudges the
    bin midpoint up by one ULP)."""
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return struct.unpack("<d", struct.pack("<Q", bits + 1))[0]


ZERO_SENTINEL = float(np.float64(np.float32(1e-35)))


def dataset_params():
    return dict(
        max_bin=255,
        min_data_in_bin=1,
        bin_construct_sample_cnt=1_000_000,
        feature_pre_filter=False,
    )


def base_train_params(seed, num_iterations):
    return dict(
        objective="regression",
        boost_from_average=False,
        deterministic=True,
        force_row_wise=True,
        num_threads=1,
        seed=seed,
        bagging_freq=0,
        bagging_fraction=1.0,
        feature_fraction=1.0,
        verbosity=-1,
        min_data_in_leaf=1,
        min_sum_hessian_in_leaf=1e-3,
        num_leaves=4,
        learning_rate=0.1,
        num_iterations=num_iterations,
    )


def feature_sidecar(X):
    """Per-feature identity-binning layout (bins + bin_upper_bound + num_bin +
    most_freq_bin), matching lib_lightgbm's realized dataset for the corpus."""
    num_feat = X.shape[1]
    feats = []
    for fi in range(num_feat):
        col = X[:, fi]
        distinct = sorted(set(float(v) for v in col))
        rank = {v: i for i, v in enumerate(distinct)}
        bins = [rank[float(v)] for v in col]
        num_bin = len(distinct)
        counts = np.bincount(np.asarray(bins, dtype=np.int64), minlength=num_bin)
        most_freq_bin = int(np.argmax(counts))
        ub = []
        for i in range(num_bin):
            if i == 0 and most_freq_bin == 0:
                ub.append(ZERO_SENTINEL)
            elif i + 1 < num_bin:
                ub.append(next_up((distinct[i] + distinct[i + 1]) / 2.0))
            else:
                ub.append(next_up(distinct[i] + 0.5))
        feats.append(
            dict(
                bins=bins,
                bin_upper_bound=ub,
                num_bin=num_bin,
                most_freq_bin=most_freq_bin,
            )
        )
    return feats


def make_corpus(seed):
    """A small deterministic numeric corpus: 24 rows, 3 integer-valued features
    (identity-binned), a smooth-ish target separable by features 0 and 1 (so the
    grown trees split on more than one feature -> non-trivial importance)."""
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    n = 24
    f0 = rng.randint(0, 4, size=n).astype(np.float64)
    f1 = rng.randint(0, 3, size=n).astype(np.float64)
    f2 = rng.randint(0, 2, size=n).astype(np.float64)
    X = np.stack([f0, f1, f2], axis=1)
    y = (2.0 * f0 + 3.0 * f1 - 1.0 * f2 + 0.25 * rng.randn(n)).astype(np.float64)
    return X, y


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]
    assert lgb.__version__ == version, f"lightgbm {lgb.__version__} != recorded {version}"
    os.makedirs(out_dir, exist_ok=True)

    X, y = make_corpus(seed)
    n_iter_base = 5

    # ---- base model ----
    d = lgb.Dataset(X, label=y, free_raw_data=False, params=dataset_params())
    base = lgb.train(base_train_params(seed, n_iter_base), d)
    base.save_model(os.path.join(out_dir, "base_model.txt"))

    # ---- refit (leaf-refit, ADV-06) ----
    # Booster.refit re-fits the leaf OUTPUTS on (X, y) without changing structure.
    # Reload the base from disk each time so refit starts from an identical model.
    for decay, fname in [(0.9, "refit_decay09.txt"), (0.0, "refit_decay00.txt")]:
        b = lgb.Booster(model_file=os.path.join(out_dir, "base_model.txt"))
        refit = b.refit(X, y, decay_rate=decay, dataset_params=dataset_params())
        refit.save_model(os.path.join(out_dir, fname))

    # ---- continue-training (input_model, ADV-06) ----
    # Continue boosting from the base for 3 more iterations via init_model.
    d2 = lgb.Dataset(X, label=y, free_raw_data=False, params=dataset_params())
    cont = lgb.train(
        base_train_params(seed, 3),
        d2,
        init_model=os.path.join(out_dir, "base_model.txt"),
    )
    cont.save_model(os.path.join(out_dir, "continue_model.txt"))

    # ---- feature importance (ADV-07) ----
    imp = dict(
        split=[int(v) for v in base.feature_importance(importance_type="split")],
        gain=[float(v) for v in base.feature_importance(importance_type="gain")],
    )
    with open(os.path.join(out_dir, "importance.json"), "w") as f:
        json.dump(imp, f, sort_keys=True, indent=2)
        f.write("\n")

    # ---- shared sidecar ----
    sidecar = dict(
        features=feature_sidecar(X),
        label=[float(v) for v in y],
        learning_rate=0.1,
        num_iterations_base=n_iter_base,
    )
    with open(os.path.join(out_dir, "advanced.json"), "w") as f:
        json.dump(sidecar, f, sort_keys=True, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
