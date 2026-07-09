#!/usr/bin/env python3
"""Phase-7 (plan 07-11, ADV-01..05) REAL lib_lightgbm advanced-constraints
learner-oracle capture.

Models the Phase-5/07-08 learner-oracle posture (the real pip `lib_lightgbm` 4.6
`save_model()` IS the authoritative v4 model text), extended to the W10 advanced
learner constraints. It trains single-tree regression models on a shared numeric
corpus across the per-axis matrix and dumps, per cell:

  - the authoritative v4 model text (`<name>.txt`), loaded by the Rust test via
    `lgbm_model::load` and compared against the Rust-grown constrained tree
    field-for-field through the shared %.17g formatter; AND
  - a small JSON sidecar (`<name>.json`) carrying the EXACT per-row bins,
    `bin_upper_bound`, `num_bin`, `most_freq_bin`, the per-row `grad`, and the
    constraint axis the Rust learner must consume — so the bit-exact comparison
    can ONLY falsify the constraint gate, never the (Phase-2) numeric binning.

OBJECTIVE = regression-l2, `boost_from_average=false`: at iteration 1 the score is
0 so `grad = -label`, `hess = 1`.

DETERMINISM / IDEMPOTENCY: each model trains with `deterministic=true
force_row_wise=true num_threads=1 seed=<seed> bagging_fraction=1.0
feature_fraction=1.0` and no subsampling => byte-identical re-runs.

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/constraints/`.

Usage:
  constraints_oracle_capture.py <out_dir> <seed> <lightgbm_version>
"""

import json
import math
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


# LightGBM bin-0 zero sentinel when most_freq_bin == 0 (offset==1): the threshold
# stored for a split at bin 0 is `(float)1e-35` widened to f64.
ZERO_SENTINEL = float(np.float64(np.float32(1e-35)))


def dataset_params():
    """Identity-binning dataset params: each distinct value gets its own bin, so the
    Rust sidecar's per-row bins == lib_lightgbm's realized bins."""
    return dict(
        max_bin=255,
        min_data_in_bin=1,
        bin_construct_sample_cnt=1_000_000,
        feature_pre_filter=False,
    )


def base_train_params(seed):
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
    )


def feature_sidecar(X, booster):
    """Derive the per-feature bins + bin_upper_bound from lib_lightgbm's realized
    dataset, so the Rust replay consumes the IDENTICAL bin layout."""
    dm = booster.dump_model()
    infos = dm.get("feature_infos", {})
    num_feat = X.shape[1]
    feats = []
    for fi in range(num_feat):
        col = X[:, fi]
        distinct = sorted(set(float(v) for v in col))
        # Identity binning: bin index = rank of the value among distinct values.
        # lib_lightgbm numeric binning maps value -> the smallest bin whose upper
        # bound >= value; with min_data_in_bin=1 and distinct consecutive values
        # each value gets its own bin. bin_upper_bound[b] = midpoint to next value
        # (or value for the last bin). We pin the realized layout from the model.
        rank = {v: i for i, v in enumerate(distinct)}
        bins = [rank[float(v)] for v in col]
        num_bin = len(distinct)
        counts = np.bincount(np.asarray(bins, dtype=np.int64), minlength=num_bin)
        most_freq_bin = int(np.argmax(counts))
        # bin_upper_bound matching LightGBM `GetDoubleUpperBound` EXACTLY: the
        # midpoint to the next bin nudged up one ULP. When most_freq_bin == 0 the
        # offset==1 path stores the zero-sentinel for bin 0 (the Rust learner reads
        # bin_upper_bound[threshold]). For the corpus's consecutive integer values
        # the midpoint is `b + 0.5`.
        ub = []
        for i in range(num_bin):
            if i == 0 and most_freq_bin == 0:
                ub.append(ZERO_SENTINEL)
            elif i + 1 < num_bin:
                ub.append(next_up((distinct[i] + distinct[i + 1]) / 2.0))
            else:
                # last bin upper bound = +inf in LightGBM; Rust never records it as
                # a threshold (a split AT the last bin is invalid), so a finite
                # placeholder is fine. Use next_up(value + 0.5) for determinism.
                ub.append(next_up(distinct[i] + 0.5))
        _ = math
        feats.append(
            dict(
                bins=bins,
                bin_upper_bound=ub,
                num_bin=num_bin,
                most_freq_bin=most_freq_bin,
            )
        )
    _ = infos
    return feats


def train_and_dump(out_dir, name, X, y, axis, seed):
    """Train ONE deterministic single-tree model under `axis` + dump model + sidecar."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = lgb.Dataset(X, label=y, free_raw_data=False, params=dataset_params())
    d.construct()
    params = base_train_params(seed)
    params.update(axis)
    booster = lgb.train(params, d, num_boost_round=1)

    feats = feature_sidecar(X, booster)
    model_path = os.path.join(out_dir, name + ".txt")
    sidecar_path = os.path.join(out_dir, name + ".json")
    booster.save_model(model_path)
    # Record the axis in the sidecar WITHOUT machine-absolute paths: the Rust test
    # reads the forced JSON from `<name>.forced.json` (relative to the fixtures
    # dir), so store only the basename — keeping the committed sidecar portable +
    # byte-idempotent across machines.
    axis_recorded = dict(axis)
    if "forcedsplits_filename" in axis_recorded:
        axis_recorded["forcedsplits_filename"] = os.path.basename(
            axis_recorded["forcedsplits_filename"]
        )
    sidecar = dict(
        name=name,
        features=feats,
        grad=[float(-v) for v in y],  # grad = -label (iter-1, score 0)
        num_leaves=int(params.get("num_leaves", 4)),
        axis=axis_recorded,
        shrinkage=float(params.get("learning_rate", 0.1)),
    )
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("constraints_oracle_capture: wrote %s + %s" % (model_path, sidecar_path))


def numeric_corpus():
    """A 2-feature numeric corpus (8 rows, 4 distinct values/feature). f0 increasing
    (clean split), f1 arranged so the unconstrained best is 'decreasing' — so a +1
    monotone constraint changes the chosen tree."""
    f0 = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    f1 = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
    X = np.column_stack([f0, f1])
    # low rows positive label (=> negative leaf output), high rows negative.
    y = [6.0, 6.0, 5.0, 5.0, -5.0, -5.0, -6.0, -6.0]
    return X, y


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update "
        "CONSTRAINTS_ORACLE_LIGHTGBM_VERSION" % (lgb.__version__, version)
    )
    os.makedirs(out_dir, exist_ok=True)

    X, y = numeric_corpus()

    # --- ADV-01 monotone: method×penalty×vector axes ---
    for method in ("basic", "intermediate", "advanced"):
        for penalty in (0.0, 5.0):
            tag = "mono_%s_p%g" % (method, penalty)
            train_and_dump(
                out_dir, tag, X, y,
                axis=dict(
                    monotone_constraints=[1, 1],
                    monotone_constraints_method=method,
                    monotone_penalty=penalty,
                ),
                seed=seed,
            )
    # mixed +1/-1 vector (basic).
    train_and_dump(
        out_dir, "mono_mixed", X, y,
        axis=dict(monotone_constraints=[1, -1], monotone_constraints_method="basic"),
        seed=seed,
    )

    # --- ADV-02 interaction: one group, two groups ---
    train_and_dump(
        out_dir, "interaction_one", X, y,
        axis=dict(interaction_constraints=[[0]]),
        seed=seed,
    )
    train_and_dump(
        out_dir, "interaction_two", X, y,
        axis=dict(interaction_constraints=[[0], [1]]),
        seed=seed,
    )

    # --- ADV-04 extra-trees: extra_seed RNG-replay axis ---
    for es in (6, 9):
        train_and_dump(
            out_dir, "extra_trees_seed%d" % es, X, y,
            axis=dict(extra_trees=True, extra_seed=es),
            seed=seed,
        )

    # --- ADV-05 CEGB: tradeoff×penalty axes ---
    for tradeoff in (1.0, 0.5):
        train_and_dump(
            out_dir, "cegb_t%g_psplit" % tradeoff, X, y,
            axis=dict(cegb_tradeoff=tradeoff, cegb_penalty_split=0.1),
            seed=seed,
        )
    train_and_dump(
        out_dir, "cegb_coupled", X, y,
        axis=dict(cegb_tradeoff=1.0, cegb_penalty_feature_coupled=[1.0, 5.0]),
        seed=seed,
    )

    # --- ADV-03 forced splits: single + nested left/right ---
    # forced split written as a JSON file lib_lightgbm reads via forcedsplits_filename.
    single = {"feature": 1, "threshold": 1.5}
    nested = {
        "feature": 0,
        "threshold": 1.5,
        "left": {"feature": 1, "threshold": 0.5},
        "right": {"feature": 1, "threshold": 2.5},
    }
    for tag, spec in (("forced_single", single), ("forced_nested", nested)):
        fs_path = os.path.join(out_dir, tag + ".forced.json")
        with open(fs_path, "w") as fh:
            json.dump(spec, fh)
        train_and_dump(
            out_dir, tag, X, y,
            axis=dict(forcedsplits_filename=fs_path),
            seed=seed,
        )


if __name__ == "__main__":
    main()
