//! Split-kernel gain-param parity vs real `lib_lightgbm` 4.6 (G5 / SPEC-G5-4,
//! T-G5-4).
//!
//! Loads the THREE real-binary goldens captured by
//! `cargo run -p xtask -- gain-params-oracle-capture`
//! (`tests/fixtures/gain_params/{penalty,max_delta_step,path_smooth}/model_real.txt`,
//! produced by `xtask/py/gain_params_oracle_capture.py`), trains the IDENTICAL
//! (bin, grad, hess) corpus through `lgbm_treelearner::SerialTreeLearner` with
//! the matching ONE non-default gain param, and asserts the grown tree
//! matches bit-exact on `split_feature`/`threshold`/`split_gain` and the raw
//! Newton `leaf_value` modulo the GBDT learning-rate shrinkage the golden
//! carries — the CPU f64-fold anchor for G5-1 (`feature_contri` penalty),
//! G5-2 (`max_delta_step` clamp), and G5-3 (`path_smooth` blend, INCLUDING the
//! scan-level gain dispatch, not merely the final leaf output). All three
//! goldens land on the same `threshold=2.5` for this corpus — see
//! `gain_params_oracle_capture.py`'s module doc for why an earlier draft's
//! claimed `1.5` for `path_smooth` did not reproduce against a real
//! lightgbm==4.6.0 and was corrected.
//!
//! Idiom mirrors `na_missing_parity.rs`: a `CARGO_MANIFEST_DIR`-rooted
//! fixture path (NEVER the untracked `LightGBM/` tree) and a graceful SKIP
//! when a golden is absent.

use std::path::PathBuf;

use lgbm_compute::gain::GainConfig;
use lgbm_compute::{runtime::cpu_client, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_model::Tree;
use lgbm_treelearner::learner::{FeatureColumn, LearnerConstraints, SerialTreeLearner};
use lgbm_treelearner::BinColumn;

fn gain_params_dir(which: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gain_params")
        .join(which)
}

fn model_real_fixture(which: &str) -> PathBuf {
    gain_params_dir(which).join("model_real.txt")
}

/// Load a real-binary golden's single grown `trees[0]`, or SKIP gracefully
/// (`eprintln!`, not fail) when the fixture is absent — mirroring
/// `na_missing_parity.rs::load_real_tree`.
fn load_real_tree(which: &str) -> Option<Tree> {
    let path = model_real_fixture(which);
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "gain_params_parity[{which}]: SKIP — real-binary golden {} not found. Run \
             `LGBM_CAPTURE_PYTHON=<py-with-lightgbm-4.6> cargo run -p xtask -- \
             gain-params-oracle-capture` and commit the golden.",
            path.display()
        );
        return None;
    };
    let model = lgbm_model::model_text::load(&text)
        .unwrap_or_else(|e| panic!("real golden {} failed to load: {e:?}", path.display()));
    assert_eq!(
        model.trees.len(),
        1,
        "real golden {} must contain exactly one grown tree",
        path.display()
    );
    Some(model.trees[0].clone())
}

/// Join f64s through the SHARED `lgbm-model` `%.17g` formatter — the same
/// arbiter `learner_parity.rs`/`na_missing_parity.rs` use.
fn join_g17(v: &[f64]) -> String {
    v.iter()
        .map(|&x| lgbm_model::format::format_g17(x))
        .collect::<Vec<_>>()
        .join(" ")
}

/// LightGBM's real per-bin upper bound for an identity-binned integer feature
/// (`midpoint(i-1,i) + 1 ULP`) — see `na_missing_parity.rs::real_upper_bounds`
/// for the derivation; duplicated here (test binaries do not share code).
fn real_upper_bounds(num_bin: u32) -> Vec<f64> {
    (0..num_bin)
        .map(|b| {
            let mid = b as f64 + 0.5;
            f64::from_bits(mid.to_bits() + 1) // mid + 1 ULP (mid > 0 here)
        })
        .collect()
}

/// The gain-param corpus `gain_params_oracle_capture.py` trained on: 10 rows,
/// one feature, 5 identity-binned distinct values `0..4` (each twice), no
/// NaN. All 5 real bins tie at count 2 ⇒ `most_freq_bin == 0` (`offset==1`
/// identity-binning convention, matching `na_missing_corpus`'s sibling).
fn gain_params_corpus() -> (FeatureColumn, Vec<f32>, Vec<f32>) {
    let num_bin = 5u32;
    let bins = vec![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let grad: Vec<f32> = vec![-6.0, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 6.0, 6.0];
    let hess = vec![1.0f32; grad.len()];
    let f0 = FeatureColumn {
        bins: BinColumn::new(bins, num_bin),
        num_bin,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: num_bin - 1,
        default_bin: num_bin, // out of range -> never the skip target
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: real_upper_bounds(num_bin),
        real_feature_index: 0,
        ..Default::default()
    };
    (f0, grad, hess)
}

fn relaxed_gain_cfg() -> GainConfig {
    GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 1e-3, // == the python min_sum_hessian_in_leaf
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    }
}

/// Assert the Rust-grown `rust` tree matches the real-binary `golden` tree on
/// `split_feature`/`threshold`/`split_gain` (bit-exact) and the raw Newton
/// `leaf_value` modulo the GBDT shrinkage the golden carries. Mirrors
/// `na_missing_parity.rs::assert_real_tree_parity`.
fn assert_real_tree_parity(which: &str, rust: &Tree, golden: &Tree, shrinkage: f64) {
    assert_eq!(
        rust.num_leaves, golden.num_leaves,
        "[{which}] num_leaves {} != real golden {}",
        rust.num_leaves, golden.num_leaves
    );
    assert_eq!(
        rust.split_feature, golden.split_feature,
        "[{which}] split_feature != real golden"
    );
    assert_eq!(
        join_g17(&rust.threshold),
        join_g17(&golden.threshold),
        "[{which}] threshold (%.17g) != real golden"
    );
    // split_gain is f32 in the Tree model (matches the C++ `float` field); the
    // golden's %g-formatted text round-trips through the SAME f32 width, so a
    // plain `==` on the parsed f32 is the bit-exact comparison.
    assert_eq!(
        rust.split_gain, golden.split_gain,
        "[{which}] split_gain != real golden — the G5 gain-param formula diverges"
    );
    let mut shrunk = rust.clone();
    for v in shrunk.leaf_value.iter_mut() {
        *v *= shrinkage;
    }
    assert_eq!(
        join_g17(&shrunk.leaf_value),
        join_g17(&golden.leaf_value),
        "[{which}] shrinkage-applied leaf_value (%.17g) != real golden — the learner's raw \
         Newton leaf output diverges from lib_lightgbm"
    );
}

/// T-G5-4 (SPEC-G5-4, AS-4), penalty (G5-1): `feature_contri = [0.5]` on the
/// sole feature must reproduce lib_lightgbm's split_gain (exactly half the
/// unpenalized gain) and leaf outputs (unchanged — a single feature has no
/// argmax to perturb).
#[test]
fn gain_params_parity_penalty_real_binary() {
    let Some(golden) = load_real_tree("penalty") else {
        return;
    };
    let backend = CpuBackend;
    let client = cpu_client();
    let (f0, g, h) = gain_params_corpus();
    let cfg = relaxed_gain_cfg();
    let tree = SerialTreeLearner::new(&backend, &client, cfg, 2, -1)
        .with_features(vec![f0])
        .with_constraints(LearnerConstraints {
            feature_contri: vec![0.5],
            ..Default::default()
        })
        .train(&g, &h, true)
        .expect("feature_contri train ok (T-G5-1)");
    assert_real_tree_parity("penalty", &tree, &golden, 0.1);
}

/// T-G5-4 (SPEC-G5-4, AS-4), max_delta_step (G5-2): `max_delta_step = 0.7`
/// must reproduce lib_lightgbm's clamped leaf outputs (both leaves clamp to
/// `±0.7` pre-shrinkage here) and the resulting clamped split_gain.
#[test]
fn gain_params_parity_max_delta_step_real_binary() {
    let Some(golden) = load_real_tree("max_delta_step") else {
        return;
    };
    let backend = CpuBackend;
    let client = cpu_client();
    let (f0, g, h) = gain_params_corpus();
    let mut cfg = relaxed_gain_cfg();
    cfg.max_delta_step = 0.7;
    let tree = SerialTreeLearner::new(&backend, &client, cfg, 2, -1)
        .with_features(vec![f0])
        .train(&g, &h, true)
        .expect("max_delta_step train ok (T-G5-2)");
    assert_real_tree_parity("max_delta_step", &tree, &golden, 0.1);
}

/// T-G5-4 (SPEC-G5-4, AS-4), path_smooth (G5-3): `path_smooth = 2.0` must
/// reproduce lib_lightgbm's smoothed leaf outputs. Same winning threshold
/// (`2.5`) as the other two goldens for this corpus — see the module doc.
#[test]
fn gain_params_parity_path_smooth_real_binary() {
    let Some(golden) = load_real_tree("path_smooth") else {
        return;
    };
    let backend = CpuBackend;
    let client = cpu_client();
    let (f0, g, h) = gain_params_corpus();
    let mut cfg = relaxed_gain_cfg();
    cfg.path_smooth = 2.0;
    let tree = SerialTreeLearner::new(&backend, &client, cfg, 2, -1)
        .with_features(vec![f0])
        .train(&g, &h, true)
        .expect("path_smooth train ok (T-G5-3)");
    assert_real_tree_parity("path_smooth", &tree, &golden, 0.1);
}
