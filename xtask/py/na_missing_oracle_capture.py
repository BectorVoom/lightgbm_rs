#!/usr/bin/env python3
"""T-G4-4 (SPEC-G4-4) REAL lib_lightgbm `na_as_missing` capture.

Trains ONE deterministic single-split-tree regression model on a real
`lib_lightgbm` 4.6 binary (the prebuilt pip wheel — see `learner_oracle_capture.py`
for why the in-repo `LightGBM/` tree cannot be built here) over a corpus with
genuine NaN values on a `num_bin > 2` feature with `use_missing=True` (the C++
default; `zero_as_missing=False`), and dumps the authoritative v4 model text. The
Rust `na_missing_parity.rs` test loads this golden, trains the SAME (bin, grad,
hess) corpus through `lgbm_treelearner::SerialTreeLearner`, and asserts the grown
tree matches bit-exact (threshold / decision_type / default_left / leaf output
modulo shrinkage) — proving the T-G4-1 histogram-scan fold and T-G4-2 partition
routing reproduce the real NA_AS_MISSING forward-branch decision.

DETERMINISM / IDEMPOTENCY: `deterministic=true force_row_wise=true
num_threads=1 seed=<NA_MISSING_ORACLE_SEED>` (the SAME
`LEARNER_ORACLE_SEED` other captures use, `xtask/src/main.rs`), NO bagging/
subsampling — re-running produces byte-identical model text.

BINNING (verified interactively before writing this script, T-G4-4 session):
with 5 distinct non-NaN raw values `0.0..4.0` (each row's value IS its intended
bin index — "identity binning", the SAME technique `learner_oracle_capture.py`
uses) and `max_bin=255 min_data_in_bin=1 bin_construct_sample_cnt>=n_rows
feature_pre_filter=false`, LightGBM assigns bin `i` to raw value `i` for
`i in 0..4`, and (per `BinMapper::ValueToBin`'s NaN routing) NaN gets the TOP bin
`num_bin-1 == 5` — so `num_bin == 6`. Every real bin's row count is tied at 2
(identical tie structure to `learner_oracle_capture.py`'s spine corpus), which
resolves to `most_freq_bin == 0` (verified: the captured `threshold` field reads
`2.5000000000000004`, LightGBM's real per-bin upper bound for an IDENTITY,
`most_freq_bin==0` (offset==1) binned feature — see `real_upper_bounds` in
`learner_parity.rs`). This is the offset==1 case, i.e. it exercises the T-G4-1
FORWARD `na_preamble` (bin 0 reconstructed via subtraction), NOT just the
REVERSE top-bin exclusion.

Usage:
  na_missing_oracle_capture.py <out_model.txt> <oracle_seed> <lightgbm_version>
"""

import sys

import numpy as np

import lightgbm as lgb


def main():
    out_path = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update NA_MISSING_ORACLE_LIGHTGBM_VERSION"
        % (lgb.__version__, version)
    )

    # 12 rows, one feature: 5 distinct identity-binned values 0..4 (each twice)
    # PLUS 2 genuine NaN rows (`na_as_missing` requires num_bin > 2 -- num_bin=6
    # here). grad[i] = -label[i] (boost_from_average=False, iter-1 score==0), so
    # picking integer labels makes the captured gradients match the Rust corpus
    # EXACTLY (the same D-03 trick `learner_oracle_capture.py` uses).
    #               row:   0    1    2    3    4    5    6    7    8    9   10   11
    f0 = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, float("nan"), float("nan")]
    grad = [-6.0, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]
    label = [-g for g in grad]

    X = np.array(f0, dtype=np.float64).reshape(-1, 1)
    y = np.array(label, dtype=np.float64)

    params = {
        "objective": "regression",
        "boost_from_average": False,  # iter-1 score==0 => grad=-label, hess=1
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "verbosity": -1,
        "num_leaves": 2,  # ONE split — the cleanest possible na_as_missing golden
        "num_iterations": 1,
        # --- identity binning (bin index == raw value) ---
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,  # <= smallest leaf hessian sum (=1.0)
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_gain_to_split": 0.0,
        # --- the T-G4-1..3 surface under test ---
        "use_missing": True,
        "zero_as_missing": False,
    }

    dtrain = lgb.Dataset(X, label=y, params=params, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=1)

    # Sanity-check the tree shape BEFORE writing the golden, so a future
    # LightGBM version drift ABORTS loudly instead of silently emitting a
    # misleading golden (mirrors `learner_oracle_capture.py`'s
    # `assert_identity_binning` discipline): the root must actually be a
    # SPLIT node (not a bare leaf) whose `missing_type` is NaN.
    model = booster.dump_model()
    tree0 = model["tree_info"][0]["tree_structure"]
    assert "left_child" in tree0 and "right_child" in tree0, (
        "expected a single-split (2-leaf) root node, got a bare leaf — "
        "the corpus/params no longer produce a splittable root"
    )
    assert tree0.get("missing_type") == "NaN", (
        "expected the root split's missing_type to be NaN (na_as_missing), got %r"
        % tree0.get("missing_type")
    )

    booster.save_model(out_path)
    print("na_missing_oracle_capture: wrote %s" % out_path)


if __name__ == "__main__":
    main()
