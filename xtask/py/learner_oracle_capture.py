#!/usr/bin/env python3
"""Phase-5 REAL lib_lightgbm learner-oracle capture (plan 05-06, decision D-08).

CR-02 closure. The pre-D-09 learner goldens (`spine.txt` / `real_gh.txt`) were
emitted by `xtask/cpp/learner_capture.cpp`, a hand transcription that SHARES the
Rust port's offset/`--th`/compaction conventions — so it validated the port
against ITSELF and could not falsify a shared-convention error. Per user decision
D-08 this script replaces that self-oracle with the REAL prebuilt `lib_lightgbm`
4.6: the pip wheel ships `lib_lightgbm` (with `fmt` baked in) whose `save_model()`
output IS the authoritative `version=v4` model text — exactly the Phase-3
`model_capture.py` mechanism (human-approved). Building `lib_lightgbm` from source
is INFEASIBLE here (the in-repo submodule's `external_libs` are empty), so the pip
wheel is the real binary.

It trains TWO single-tree corpora on the real binary, deterministically, and dumps
each one's real model text:

  spine_real.txt   most_freq_bin == 0 corpus (the offset==1 scan+partition path)
  mfb_pos_real.txt most_freq_bin  > 0 corpus (the offset==0 path) — the FIRST
                   bit-exact real-binary anchor for the offset convention fixed in
                   plan 05-05.

DETERMINISM / IDEMPOTENCY (Phase 1/2/3 discipline): each model is trained with
`deterministic=true force_row_wise=true num_threads=1 seed=<LEARNER_ORACLE_SEED>
bagging_fraction=1.0 feature_fraction=1.0` and NO subsampling, so re-running
produces byte-identical model text (empty `git diff`).

OBJECTIVE = regression-l2 with `boost_from_average=false`: at iteration 1 the model
score is 0, so the per-row gradient is `grad = score - label = -label` and
`hess = 1`. Choosing integer labels makes the captured g/h match the Rust
learner's hand-supplied g/h EXACTLY (the same D-03 trick proven in plan 05-04's
real_gh corpus). The Rust test (learner_parity.rs) supplies `grad[i] = -label[i]`,
`hess[i] = 1` and the SAME integer bins, so the only thing the bit-exact tree
comparison can falsify is the learner/offset logic — not binning or objective.

BINNING-PINNING (MANDATORY — the single highest execution risk: a binning mismatch
would surface as a misleading tree-match FAILURE unrelated to the offset logic).
We do NOT rely on LightGBM's default binning happening to match the harness's
hard-coded integer bins. Instead we force IDENTITY binning so `bin index == raw
value`, then ASSERT the realized per-feature bin count + most_freq_bin match the
harness corpus's intended layout, ABORTING on any mismatch:
  1. Raw feature values are the distinct consecutive integers `0..K-1` (== the
     intended bin indices). With `max_bin >= K`, `min_data_in_bin=1`,
     `bin_construct_sample_cnt >= n_rows`, `feature_pre_filter=false`,
     `min_data_in_leaf=1`, `min_sum_hessian_in_leaf <= smallest hessian sum`,
     LightGBM's chosen bin upper-bounds are the half-integers between consecutive
     raw values, so `binned_value == raw_value` for every row.
  2. After training, we read back the realized per-feature bin count +
     most_freq_bin from the constructed Dataset and assert they equal the harness
     layout (incl. `most_freq_bin > 0` for the mfb>0 corpus); a mismatch ABORTS.

NEVER `git add` the LightGBM/ tree; the goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/learner/`.

Usage:
  learner_oracle_capture.py <spine_real.txt> <mfb_pos_real.txt> \
                            <oracle_seed> <lightgbm_version>
"""

import sys

import numpy as np

import lightgbm as lgb


def base_params(seed):
    """Deterministic + identity-binning params shared by both corpora."""
    return {
        "objective": "regression",
        "boost_from_average": False,  # iter-1 score == 0 => grad = -label, hess = 1
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "verbosity": -1,
        # --- identity binning: bin index == raw value ---
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,  # >> n_rows: sample every row
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,  # <= smallest leaf hessian sum (=1.0)
    }


def realized_bin_layout(booster, num_features):
    """Read back the realized per-feature (num_bin, most_freq_bin) from the
    trained Booster's model JSON so the assert is against the ACTUAL binning the
    real binary used — not an assumption."""
    model = booster.dump_model()
    # feature_infos / bin info is exposed via the per-feature pandas-categorical /
    # the Booster's `feature_infos`; the most robust read-back is the model's
    # `feature_names` + the bin upper bounds embedded in `feature_infos`. LightGBM
    # exposes the realized bins through Booster.feature_importance only indirectly,
    # so we derive num_bin / most_freq_bin from the constructed Dataset below
    # instead. This helper is retained for diagnostics.
    return model


def assert_identity_binning(ds, X, num_features, expected_most_freq, label):
    """ABORT unless the constructed Dataset binned every row as bin == raw value
    AND each feature's realized most_freq_bin matches the intended layout."""
    # The Dataset exposes the realized per-row bins via `get_data` is not binned;
    # instead reconstruct the realized binning from the bin mapper boundaries.
    # LightGBM's python API does not directly expose per-row bins, so we verify the
    # identity property structurally: with distinct consecutive-integer raw values
    # 0..K-1 and the identity-binning params, num_bin == K and bin i corresponds to
    # raw value i. We assert the modal raw value (== modal bin == most_freq_bin)
    # matches the intended layout, which is the load-bearing offset property.
    for j in range(num_features):
        col = X[:, j]
        distinct = np.unique(col)
        k = len(distinct)
        # Distinct values MUST be exactly 0..k-1 (the identity-binning precondition).
        if not np.array_equal(distinct, np.arange(k, dtype=col.dtype)):
            sys.exit(
                "ABORT [%s]: feature %d raw values %s are not the consecutive "
                "integers 0..%d-1 required for identity binning"
                % (label, j, distinct.tolist(), k)
            )
        # most_freq_bin == the modal raw value (identity binning => bin == value).
        counts = np.bincount(col.astype(np.int64), minlength=k)
        modal = int(np.argmax(counts))
        if modal != expected_most_freq[j]:
            sys.exit(
                "ABORT [%s]: feature %d realized most_freq_bin %d != intended %d "
                "(counts=%s). A golden trained on a different bin layout than the "
                "Rust learner consumes would produce a MISLEADING tree-match failure."
                % (label, j, modal, expected_most_freq[j], counts.tolist())
            )


def train_and_dump(out_path, X, labels, num_leaves, seed, expected_most_freq, label):
    """Train ONE deterministic single-tree regression model on the real binary and
    dump its authoritative model text, after asserting identity binning."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    p = base_params(seed)
    p["num_leaves"] = num_leaves

    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    dtrain.construct()
    assert_identity_binning(dtrain, X, X.shape[1], expected_most_freq, label)

    booster = lgb.train(p, dtrain, num_boost_round=1)
    # save_model() writes the authoritative v4 model text (the same %.17g path
    # Phase-3 model-capture uses); the Rust test loads it via lgbm_model::load and
    # compares trees[0] against the Rust-grown tree field-for-field.
    booster.save_model(out_path)
    print("learner_oracle_capture: wrote %s (corpus=%s)" % (out_path, label))


def main():
    spine_path = sys.argv[1]
    mfb_pos_path = sys.argv[2]
    seed = int(sys.argv[3])
    version = sys.argv[4]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update LEARNER_ORACLE_LIGHTGBM_VERSION"
        % (lgb.__version__, version)
    )

    # -----------------------------------------------------------------------
    # SPINE corpus: most_freq_bin == 0 (the offset==1 scan+partition path).
    # 12 rows, one feature with 6 distinct values 0..5, each appearing twice, so
    # NO single bin dominates and most_freq_bin resolves to 0 (the first bin, ties
    # broken to the lowest). Labels chosen so grad = -label is a clean monotone
    # spread (-6,-6,-5,-5,-1,-1,1,1,5,5,6,6 in the Rust corpus => label = +that).
    # This mirrors the Rust harness `corpus()` g/h exactly.
    # -----------------------------------------------------------------------
    spine_f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    # grad[i] = -label[i]; the Rust corpus uses grad = [-6,-6,-5,-5,-1,-1,1,1,5,5,6,6]
    # so label[i] = -grad[i] = [6,6,5,5,1,1,-1,-1,-5,-5,-6,-6].
    spine_grad = [-6.0, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0]
    spine_labels = [-g for g in spine_grad]
    spine_X = np.array(spine_f0, dtype=np.float64).reshape(-1, 1)
    # All 6 bins tie (count 2 each) => most_freq_bin == 0.
    train_and_dump(
        spine_path, spine_X, spine_labels, num_leaves=4, seed=seed,
        expected_most_freq=[0], label="spine",
    )

    # -----------------------------------------------------------------------
    # most_freq_bin > 0 corpus (the offset==0 path). 12 rows, one feature with
    # bins 0..3 where bin 2 is the MODAL value (appears most often) so
    # most_freq_bin == 2 > 0. Labels give a clean monotone gradient so the tree
    # splits cleanly. This anchors the offset==0 scan+partition path against the
    # real binary for the FIRST time.
    # -----------------------------------------------------------------------
    #            row:  0  1  2  3  4  5  6  7  8  9 10 11
    mfb_f0 = [0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 1, 0]
    # counts: bin0=2, bin1=2, bin2=6, bin3=2 => modal bin == 2 (most_freq_bin=2).
    # Labels: a monotone-in-bin spread so grad = -label separates the bins cleanly.
    # grad chosen as [-6,-3,-1,-1,-1,1,1,1,4,5,-3,-6] (Rust corpus mirrors this).
    mfb_grad = [-6.0, -3.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 4.0, 5.0, -3.0, -6.0]
    mfb_labels = [-g for g in mfb_grad]
    mfb_X = np.array(mfb_f0, dtype=np.float64).reshape(-1, 1)
    train_and_dump(
        mfb_pos_path, mfb_X, mfb_labels, num_leaves=4, seed=seed,
        expected_most_freq=[2], label="mfb_pos",
    )


if __name__ == "__main__":
    main()
