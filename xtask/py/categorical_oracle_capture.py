#!/usr/bin/env python3
"""Phase-7 (plan 07-08, TRL-06) REAL lib_lightgbm CATEGORICAL learner-oracle capture.

Models exactly the Phase-5 `learner_oracle_capture.py` posture (the real pip
`lib_lightgbm` 4.6 `save_model()` IS the authoritative v4 model text), extended to
CATEGORICAL features. It trains single-tree regression models on synthetic
categorical corpora across the D-07 axis matrix and dumps:

  - the authoritative v4 model text (`<name>.txt`), loaded by the Rust test via
    `lgbm_model::load` and compared against the Rust-grown categorical tree
    field-for-field through the shared %.17g formatter; AND
  - a small JSON sidecar (`<name>.bins.json`) carrying the EXACT per-row bins,
    the `bin_2_categorical` map, `num_bin`, and `most_freq_bin` the Rust learner
    must consume so the bit-exact comparison can ONLY falsify the categorical
    split/gain logic — never the (separately-validated Phase-2) categorical
    binning. The sidecar is derived deterministically from the corpus: lib_lightgbm
    categorical binning assigns bin 0 = the NaN dummy (category -1) and bins
    1..K to the distinct categories in ASCENDING category-value order (confirmed
    via `dump_model()["feature_infos"]`), so category `c` (0..K-1) -> bin `c+1`.

OBJECTIVE = regression-l2, `boost_from_average=false`: at iteration 1 the score is
0 so `grad = -label`, `hess = 1`. The Rust test supplies `grad[i] = -label[i]`,
`hess = 1`, the SAME per-row bins, and the SAME cat_* config, so the tree
comparison isolates the categorical split finding.

DETERMINISM / IDEMPOTENCY: each model trains with `deterministic=true
force_row_wise=true num_threads=1 seed=<seed> bagging_fraction=1.0
feature_fraction=1.0` and no subsampling => byte-identical re-runs.

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/categorical/`.

Usage:
  categorical_oracle_capture.py <out_dir> <seed> <lightgbm_version>
"""

import json
import os
import sys

import numpy as np

import lightgbm as lgb


def dataset_params():
    """Identity-binning dataset params (so bin == category+1, NaN dummy at 0)."""
    return dict(
        max_bin=255,
        min_data_in_bin=1,
        bin_construct_sample_cnt=1_000_000,
        feature_pre_filter=False,
    )


def train_params(seed, axis):
    """Deterministic single-tree regression params + the categorical axis cell."""
    p = dict(
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
    )
    p.update(axis)
    return p


def bins_and_map(cats):
    """Derive the per-row bins + bin_2_categorical from the corpus categories.

    lib_lightgbm categorical binning: bin 0 = NaN dummy (category -1), bins 1..K
    for the distinct categories in ASCENDING value order. With categories that are
    the consecutive integers 0..K-1, category `c` -> bin `c+1`."""
    distinct = sorted(set(int(c) for c in cats))
    if distinct != list(range(len(distinct))):
        sys.exit(
            "ABORT: categorical corpus categories %s are not consecutive integers "
            "0..K-1 (the identity-bin precondition)" % distinct
        )
    bin_2_categorical = [-1] + distinct  # bin 0 = NaN dummy
    bins = [int(c) + 1 for c in cats]
    num_bin = len(distinct) + 1
    counts = np.bincount(np.asarray(bins, dtype=np.int64), minlength=num_bin)
    # most_freq_bin = modal bin (ties -> lowest), matching the Rust corpus builder.
    most_freq_bin = int(np.argmax(counts))
    return bins, bin_2_categorical, num_bin, most_freq_bin


def assert_feature_info(booster, bin_2_categorical, label):
    """ABORT unless lib_lightgbm's realized category->bin layout matches the
    deterministic mapping the sidecar records (the load-bearing pin)."""
    dm = booster.dump_model()
    infos = dm.get("feature_infos", {})
    # The single feature is "Column_0".
    info = infos.get("Column_0")
    if info is None:
        sys.exit("ABORT [%s]: feature_infos missing Column_0: %s" % (label, infos))
    values = info.get("values")
    if values != bin_2_categorical:
        sys.exit(
            "ABORT [%s]: lib_lightgbm feature_infos values %s != derived "
            "bin_2_categorical %s — a golden trained on a different categorical "
            "bin layout than the Rust learner consumes would MISLEAD the gate."
            % (label, values, bin_2_categorical)
        )


def train_and_dump(out_dir, name, cats, labels, axis, seed):
    """Train ONE deterministic single-tree categorical model + dump model + sidecar."""
    X = np.asarray(cats, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels, dtype=np.float64)
    d = lgb.Dataset(
        X, label=y, free_raw_data=False, categorical_feature=[0], params=dataset_params()
    )
    d.construct()
    booster = lgb.train(train_params(seed, axis), d, num_boost_round=1)

    bins, bin_2_categorical, num_bin, most_freq_bin = bins_and_map(cats)
    assert_feature_info(booster, bin_2_categorical, name)

    model_path = os.path.join(out_dir, name + ".txt")
    sidecar_path = os.path.join(out_dir, name + ".bins.json")
    booster.save_model(model_path)
    sidecar = dict(
        name=name,
        bins=bins,
        bin_2_categorical=bin_2_categorical,
        num_bin=num_bin,
        most_freq_bin=most_freq_bin,
        grad=[float(-v) for v in labels],  # grad = -label (iter-1, score 0)
        num_leaves=int(axis.get("num_leaves", 4)),
        cat_l2=float(axis.get("cat_l2", 10.0)),
        cat_smooth=float(axis.get("cat_smooth", 10.0)),
        min_data_per_group=int(axis.get("min_data_per_group", 100)),
        max_cat_threshold=int(axis.get("max_cat_threshold", 32)),
        max_cat_to_onehot=int(axis.get("max_cat_to_onehot", 4)),
        shrinkage=0.1,
    )
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("categorical_oracle_capture: wrote %s + %s" % (model_path, sidecar_path))


def build_corpus(num_cats, per_cat, high_cat, high_label, low_label):
    """A balanced categorical corpus: `num_cats` categories, `per_cat` rows each,
    category `high_cat` gets `high_label`, the rest `low_label`."""
    cats = []
    labels = []
    for c in range(num_cats):
        for _ in range(per_cat):
            cats.append(c)
            labels.append(high_label if c == high_cat else low_label)
    return cats, labels


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update "
        "CATEGORICAL_ORACLE_LIGHTGBM_VERSION" % (lgb.__version__, version)
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- one-hot cell: 4 categories (<= max_cat_to_onehot=4); category 3 is the
    #     odd one out, so the one-vs-rest split isolates {3}. ---
    cats, labels = build_corpus(4, 10, high_cat=3, high_label=10.0, low_label=0.0)
    train_and_dump(
        out_dir, "cat_onehot", cats, labels,
        axis=dict(
            num_leaves=4, cat_l2=10.0, cat_smooth=10.0,
            min_data_per_group=1, max_cat_threshold=32, max_cat_to_onehot=4,
        ),
        seed=seed,
    )

    # --- many-vs-many cell: 6 categories (> max_cat_to_onehot=4 forces the
    #     sorted-by-ctr group scan); a monotone label-per-category spread so the
    #     scan accumulates a non-trivial category group on one side. ---
    cats = []
    labels = []
    spread = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]
    for c in range(6):
        for _ in range(10):
            cats.append(c)
            labels.append(spread[c])
    train_and_dump(
        out_dir, "cat_manyvsmany", cats, labels,
        axis=dict(
            num_leaves=4, cat_l2=10.0, cat_smooth=0.0,
            min_data_per_group=1, max_cat_threshold=32, max_cat_to_onehot=4,
        ),
        seed=seed,
    )


if __name__ == "__main__":
    main()
