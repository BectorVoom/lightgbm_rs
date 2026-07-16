#!/usr/bin/env python3
"""T-G5-4 (SPEC-G5-4) REAL lib_lightgbm gain-param captures.

Trains THREE independent deterministic single-split-tree regression models on
a real `lib_lightgbm` 4.6 binary (the prebuilt pip wheel — see
`learner_oracle_capture.py` for why the in-repo `LightGBM/` tree cannot be
built here), each with exactly ONE of the G5 gain params set to a non-default
value (others at their config.h default), and dumps the authoritative v4 model
text for each. The Rust `gain_params_parity.rs` test loads these goldens,
trains the IDENTICAL (bin, grad, hess) corpus through
`lgbm_treelearner::SerialTreeLearner`, and asserts the grown tree matches
bit-exact — proving G5-1 (feature_contri penalty), G5-2 (max_delta_step
clamp), and G5-3 (path_smooth blend, INCLUDING the scan-level gain dispatch
that can change the winning threshold) reproduce the real gain-param math.

DETERMINISM / IDEMPOTENCY: `deterministic=true force_row_wise=true
num_threads=1 seed=<LEARNER_ORACLE_SEED>` (the SAME seed other captures use,
`xtask/src/main.rs`), NO bagging/subsampling — re-running produces
byte-identical model text.

BINNING (verified interactively before writing this script, T-G5-4 session):
a single feature with 5 distinct identity-binned values `0.0..4.0` (each row's
value IS its intended bin index — the SAME technique `learner_oracle_capture.py`
/ `na_missing_oracle_capture.py` use), `max_bin=255 min_data_in_bin=1
bin_construct_sample_cnt>=n_rows feature_pre_filter=false`. UNLIKE the tied
corpus those scripts use, THIS corpus's per-bin counts are 2/2/2/2/2 but the
gradient values are chosen asymmetric enough that `most_freq_bin==0` still
holds (verified: `threshold=2.5000000000000004` for the default/penalty/
max_delta_step goldens, the `offset==1` identity-binning convention) — EXCEPT
the `path_smooth` golden, whose smoothing blend deliberately picks a
DIFFERENT threshold (`1.5000000000000002`), proving the smoothing changes the
scan's argmax, not merely the leaf output.

Usage:
  gain_params_oracle_capture.py <which> <out_model.txt> <oracle_seed> <lightgbm_version>
  <which> in {penalty, max_delta_step, path_smooth}
"""

import sys

import numpy as np

import lightgbm as lgb


# Single feature, 5 identity-binned distinct values 0..4 (each twice), no NaN.
# grad[i] = -label[i] (boost_from_average=False, iter-1 score==0), so picking
# integer labels makes the captured gradients match the Rust corpus EXACTLY
# (the same D-03 trick `learner_oracle_capture.py` uses).
F0 = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
GRAD = [-6.0, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 6.0, 6.0]

EXTRA_PARAMS = {
    "penalty": {"feature_contri": [0.5]},
    "max_delta_step": {"max_delta_step": 0.7},
    "path_smooth": {"path_smooth": 2.0},
}

# The root split's expected `missing_type`-free `decision_type` value differs
# only by the threshold picked; both goldens are single-split (2-leaf) trees.
EXPECTED_THRESHOLD = {
    "penalty": 2.5000000000000004,
    "max_delta_step": 2.5000000000000004,
    "path_smooth": 1.5000000000000002,
}


def main():
    which = sys.argv[1]
    out_path = sys.argv[2]
    seed = int(sys.argv[3])
    version = sys.argv[4]

    assert which in EXTRA_PARAMS, "unknown gain-param golden %r" % which
    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update the recorded version"
        % (lgb.__version__, version)
    )

    label = [-g for g in GRAD]
    X = np.array(F0, dtype=np.float64).reshape(-1, 1)
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
        "num_leaves": 2,  # ONE split — the cleanest possible gain-param golden
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
        # --- the G5-1/2/3 surface under test: exactly ONE non-default here ---
        **EXTRA_PARAMS[which],
    }

    dtrain = lgb.Dataset(X, label=y, params=params, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=1)

    # Sanity-check the tree shape BEFORE writing the golden, so a future
    # LightGBM version drift ABORTS loudly instead of silently emitting a
    # misleading golden (mirrors `na_missing_oracle_capture.py`'s discipline).
    model = booster.dump_model()
    tree0 = model["tree_info"][0]["tree_structure"]
    assert "left_child" in tree0 and "right_child" in tree0, (
        "expected a single-split (2-leaf) root node, got a bare leaf — "
        "the corpus/params no longer produce a splittable root"
    )
    got_threshold = tree0.get("threshold")
    expected = EXPECTED_THRESHOLD[which]
    assert got_threshold == expected, (
        "expected root threshold %r for %r, got %r — the corpus/params no "
        "longer produce the expected split (a LightGBM version drift?)"
        % (expected, which, got_threshold)
    )

    booster.save_model(out_path)
    print("gain_params_oracle_capture[%s]: wrote %s" % (which, out_path))


if __name__ == "__main__":
    main()
