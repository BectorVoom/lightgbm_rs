//! `na_as_missing` NA-forward-routing parity vs real `lib_lightgbm` 4.6
//! (G4 / SPEC-G4-4, T-G4-4).
//!
//! Loads the real-binary golden captured by
//! `cargo run -p xtask -- na-missing-oracle-capture`
//! (`tests/fixtures/na_missing/model_real.txt`, produced by
//! `xtask/py/na_missing_oracle_capture.py`), trains the IDENTICAL (bin, grad,
//! hess) corpus through `lgbm_treelearner::SerialTreeLearner`, and asserts the
//! grown tree matches bit-exact on every learner-authoritative field
//! (`split_feature`, `decision_type` — which encodes the NA_AS_MISSING
//! `missing_type`/`default_left` bits — child topology, `leaf_count`/
//! `internal_count`, and the real-valued `threshold`), plus the raw Newton
//! `leaf_value` modulo the GBDT learning-rate shrinkage the golden carries.
//!
//! This is the T-G4-1 (histogram-scan NA fold) + T-G4-2 (partition NA routing)
//! + T-G4-3 (gate removal) end-to-end anchor: a wrong fold, a wrong routing
//! direction, or a stale gate would each surface here as a `decision_type` /
//! `threshold` / `leaf_count` mismatch against the real binary.
//!
//! Idiom mirrors `learner_parity.rs`'s real-binary tests / `predict_parity.rs`'s
//! `read_golden`: a `CARGO_MANIFEST_DIR`-rooted fixture path (NEVER the
//! untracked `LightGBM/` tree) and a graceful SKIP when the golden is absent.

use std::path::PathBuf;

use lgbm_compute::gain::GainConfig;
use lgbm_compute::{runtime::cpu_client, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_model::Tree;
use lgbm_treelearner::learner::{FeatureColumn, SerialTreeLearner};
use lgbm_treelearner::BinColumn;

fn na_missing_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/na_missing")
}

fn model_real_fixture() -> PathBuf {
    na_missing_dir().join("model_real.txt")
}

/// Load the real-binary golden's single grown `trees[0]`, or SKIP gracefully
/// (`eprintln!`, not fail) when the fixture is absent — mirroring
/// `learner_parity.rs::load_real_tree` / `predict_parity.rs::read_golden`.
fn load_real_tree(path: &PathBuf) -> Option<Tree> {
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!(
            "na_missing_parity: SKIP — real-binary golden {} not found. Run \
             `LGBM_CAPTURE_PYTHON=<py-with-lightgbm-4.6> cargo run -p xtask -- \
             na-missing-oracle-capture` and commit the golden.",
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
/// arbiter `learner_parity.rs`/`json_dump_parity.rs` use.
fn join_g17(v: &[f64]) -> String {
    v.iter()
        .map(|&x| lgbm_model::format::format_g17(x))
        .collect::<Vec<_>>()
        .join(" ")
}

/// LightGBM's real per-bin upper bound for an identity-binned integer feature
/// (`midpoint(i-1,i) + 1 ULP`) — see `learner_parity.rs::real_upper_bounds` for
/// the derivation; duplicated here (test binaries do not share code) since this
/// is the SAME `most_freq_bin==0` (offset==1) identity-binning convention the
/// capture script's docstring verifies (`threshold=2.5000000000000004` in the
/// captured golden IS `real_upper_bounds(6)[2]`).
fn real_upper_bounds(num_bin: u32) -> Vec<f64> {
    (0..num_bin)
        .map(|b| {
            let mid = b as f64 + 0.5;
            f64::from_bits(mid.to_bits() + 1) // mid + 1 ULP (mid > 0 here)
        })
        .collect()
}

/// C++ `Tree::MaybeRoundToZero` (tree.h:255-260) — see
/// `learner_parity.rs::maybe_round_to_zero` for the full derivation. Not
/// currently exercised (this corpus's leaf outputs are far from zero) but kept
/// for parity with the shrinkage-replay idiom in case a future corpus needs it.
#[allow(dead_code)]
fn maybe_round_to_zero(fval: f64) -> f64 {
    const K_ZERO_THRESHOLD_F64: f64 = 1e-35f32 as f64;
    if fval >= -K_ZERO_THRESHOLD_F64 && fval <= K_ZERO_THRESHOLD_F64 {
        0.0
    } else {
        fval
    }
}

/// The na_as_missing corpus `na_missing_oracle_capture.py` trained on: 12 rows,
/// one feature, 5 identity-binned distinct values `0..4` (each twice) PLUS 2
/// genuine NaN rows. `na_as_missing()` requires `num_bin > 2 && missing_type ==
/// NaN`; NaN routes to the TOP bin (`num_bin-1 == 5`, `BinMapper::ValueToBin`'s
/// NaN convention). All 5 real bins tie at count 2 ⇒ `most_freq_bin == 0`
/// (verified against the real golden's `threshold=2.5000000000000004`, which is
/// only reachable under the `offset==1` identity-binning convention) — so this
/// corpus exercises the T-G4-1 FORWARD `na_preamble` (bin 0 reconstructed via
/// subtraction), not merely the REVERSE top-bin exclusion.
fn na_missing_corpus() -> (Vec<FeatureColumn>, Vec<f32>, Vec<f32>, GainConfig, i32, i32) {
    let num_bin = 6u32;
    let bins = vec![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]; // NaN rows carry bin 5 (num_bin-1)
    let grad: Vec<f32> = vec![-6.0, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0];
    let hess = vec![1.0f32; grad.len()];
    let f0 = FeatureColumn {
        bins: BinColumn::new(bins, num_bin),
        num_bin,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: num_bin - 1,
        default_bin: num_bin, // out of range -> never the skip target (irrelevant: NaN, not Zero)
        most_freq_bin: 0,
        missing_type: MissingType::NaN,
        bin_upper_bound: real_upper_bounds(num_bin),
        real_feature_index: 0,
        ..Default::default()
    };
    let cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 1e-3, // == the python min_sum_hessian_in_leaf
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    };
    (vec![f0], grad, hess, cfg, 2, -1)
}

/// Assert the Rust-grown `rust` tree matches the real-binary `golden` tree on
/// every LEARNER-AUTHORITATIVE field, bit-exact, and on the raw leaf output
/// modulo the GBDT shrinkage the golden carries. Mirrors
/// `learner_parity.rs::assert_real_tree_parity`.
fn assert_real_tree_parity(rust: &Tree, golden: &Tree, shrinkage: f64) {
    assert_eq!(
        rust.num_leaves, golden.num_leaves,
        "num_leaves {} != real golden {}",
        rust.num_leaves, golden.num_leaves
    );
    assert_eq!(rust.split_feature, golden.split_feature, "split_feature != real golden");
    assert_eq!(
        rust.decision_type, golden.decision_type,
        "decision_type (missing-direction: missing_type + default_left bits) != real golden \
         — a T-G4-1 histogram-scan fold error or a T-G4-2 partition-routing error would \
         surface here"
    );
    assert_eq!(rust.left_child, golden.left_child, "left_child topology != real golden");
    assert_eq!(rust.right_child, golden.right_child, "right_child topology != real golden");
    assert_eq!(
        rust.leaf_count, golden.leaf_count,
        "leaf_count (data-partition) != real golden — the T-G4-2 NA row-routing anchor"
    );
    assert_eq!(
        rust.internal_count, golden.internal_count,
        "internal_count (data-partition) != real golden"
    );
    assert_eq!(
        join_g17(&rust.threshold),
        join_g17(&golden.threshold),
        "threshold (%.17g) != real golden — T-G4-1 histogram-scan fold divergence"
    );
    let mut shrunk = rust.clone();
    for v in shrunk.leaf_value.iter_mut() {
        *v = maybe_round_to_zero(*v * shrinkage);
    }
    assert_eq!(
        join_g17(&shrunk.leaf_value),
        join_g17(&golden.leaf_value),
        "shrinkage-applied leaf_value (%.17g) != real golden — the learner's raw Newton leaf \
         output (which folds the na_as_missing rows into whichever child won) diverges from \
         lib_lightgbm"
    );
}

/// T-G4-4 (SPEC-G4-4, AS-3): the Rust `SerialTreeLearner` grows the SAME tree
/// as the REAL `lib_lightgbm` 4.6 binary on an `na_as_missing` (genuine NaN,
/// `use_missing=true`) corpus, bit-exact on every learner-authoritative field —
/// the CPU f64-fold anchor. SKIPs gracefully when the golden is absent.
#[test]
fn na_missing_parity_real_binary() {
    let Some(golden) = load_real_tree(&model_real_fixture()) else {
        return;
    };

    let backend = CpuBackend;
    let client = cpu_client();
    let (features, g, h, cfg, num_leaves, max_depth) = na_missing_corpus();
    let mut learner = SerialTreeLearner::new(&backend, &client, cfg, num_leaves, max_depth)
        .with_features(features);
    let tree = learner.train(&g, &h, true).expect("na_as_missing train ok (T-G4-3)");

    // shrinkage=0.1, matching `na_missing_oracle_capture.py`'s params (implicit
    // LightGBM default `learning_rate=0.1`, unset by the capture script).
    assert_real_tree_parity(&tree, &golden, 0.1);
}
