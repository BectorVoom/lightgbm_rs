//! GBDT spine end-to-end boosting parity replay (Phase 6, Wave-0 scaffold).
//!
//! This is the **Nyquist scaffold** (01-config `nyquist_validation = true`): the
//! failing/`#[ignore]`d end-to-end test that names the spine golden it WILL replay
//! once the boosting loop lands. It exists now so the test surface is sampled
//! before the implementation — the slice in 06-02 fills in the body and removes
//! the `#[ignore]`.
//!
//! Mirrors the `learner_parity.rs` idioms: a `CARGO_MANIFEST_DIR`-rooted fixture
//! path (NEVER the untracked `LightGBM/` tree), the comparator precision contract
//! (`compare_exact_f64_bits` for per-iter scores / model-text leaf values,
//! `compare_within(.., ORACLE_TOL)` for g/h and metrics), and a localizing assert.
//!
//! ## Validation layers (RESEARCH §Validation Architecture, L1–L5)
//! - L1 `gradients`        — per-row g/h from the objective (~1e-6).
//! - L2 `score_accumulation` — per-iter raw scores (bit-exact f64).
//! - L3 `early_stopping`   — eval-history / best-iteration (D-12).
//! - L4 `bagging_rng`      — bagged row indices via RNG-replay (D-13 Option A,
//!   exact u32) — DEFERRED to 06-05; named here as the seam.
//! - L5 `spine_end_to_end` — `save_model()` text + `predict()` (D-13/L5).
//! - `custom_objective`    — the D-04 closure objective path.
//!
//! Allowed D-07 collapse: the spine == bagging-off / early-stopping-off /
//! boost-from-average-on cell (RESEARCH §Cross-Product Collapse Analysis).

use std::path::PathBuf;

use lgbm::{train, Booster, DenseCorpus, TrainingBuilder};
use oracle_harness::comparator::{
    compare_exact_f64_bits, compare_within, ORACLE_TOL,
};

/// The committed boosting golden directory — TRACKED under the oracle-harness
/// crate, NEVER the untracked C++/LightGBM reference tree. Populated by
/// `cargo run -p xtask -- boosting-oracle-capture`.
fn boosting_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boosting")
}

/// The recorded capture seed (xtask `BOOSTING_ORACLE_SEED`) — the spine goldens
/// were trained with this seed; the Rust replay must use the SAME seed.
const SPINE_SEED: i32 = 0x6005_7000;

/// The spine corpus, identical to `xtask/py/boosting_oracle_capture.py::spine_corpus`
/// (identity-binned: bin == raw value).
fn spine_corpus() -> DenseCorpus {
    let f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
    let f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let features: Vec<Vec<f64>> = (0..12).map(|i| vec![f0[i] as f64, f1[i] as f64]).collect();
    let labels = vec![
        2.0f32, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0,
    ];
    DenseCorpus { features, labels }
}

/// Train the spine the SAME way the capture did (regression L2, 10 iters, lr 0.1,
/// num_leaves 4, boost_from_average=true, deterministic, identity binning).
fn train_spine() -> (Booster, DenseCorpus) {
    let corpus = spine_corpus();
    let cfg = TrainingBuilder::new()
        .objective("regression")
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(true)
        .seed(SPINE_SEED)
        .deterministic(true)
        .build()
        .expect("valid spine config");
    let booster = train(&cfg, &corpus).expect("spine train ok");
    (booster, corpus)
}

/// Parse a whitespace-separated line of u64 bit patterns into `Vec<f64>` (the L2
/// score / pred golden encoding). Lines starting with `#` are comments.
fn parse_f64_bits_line(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|t| f64::from_bits(t.parse::<u64>().expect("u64 bits")))
        .collect()
}

/// Parse a whitespace-separated line of u32 bit patterns into `Vec<f32>` (the L1
/// g/h golden encoding).
fn parse_f32_bits_line(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .map(|t| f32::from_bits(t.parse::<u32>().expect("u32 bits")))
        .collect()
}

/// SKIP gracefully (returning `None`) when a golden file is absent — matching the
/// learner_parity idiom (a fresh checkout without the capture run still builds).
fn read_golden(name: &str) -> Option<String> {
    let path = boosting_dir().join(name);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "boosting_parity: SKIP — golden {} not found. Run \
                 `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- boosting-oracle-capture`.",
                path.display()
            );
            None
        }
    }
}

#[test]
fn spine_end_to_end() {
    // L5: train the regression L2 spine and assert the grown ensemble's model-text
    // leaf values replay BIT-EXACT (compare_exact_f64_bits) and predict() matches
    // `regression_spine_pred.txt` within ORACLE_TOL, both vs the real lib_lightgbm
    // 4.6 golden.
    let Some(model_text) = read_golden("regression_spine_model.txt") else {
        return;
    };
    let (booster, corpus) = train_spine();
    let golden = lgbm_model::model_text::load(&model_text).expect("parse golden model");
    let rust = booster.model();

    assert_eq!(
        rust.trees.len(),
        golden.trees.len(),
        "tree count: rust {} != golden {}",
        rust.trees.len(),
        golden.trees.len()
    );
    // Per-tree leaf values BIT-EXACT (the L5 bit-exact contract).
    for (i, (rt, gt)) in rust.trees.iter().zip(golden.trees.iter()).enumerate() {
        compare_exact_f64_bits(&rt.leaf_value, &gt.leaf_value)
            .unwrap_or_else(|m| panic!("tree {i} leaf_value not bit-exact vs golden: {m:?}"));
    }

    // predict() within ORACLE_TOL vs the captured transformed predictions.
    if let Some(pred_text) = read_golden("regression_spine_pred.txt") {
        let golden_pred: Vec<f32> = pred_text
            .lines()
            .find(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .map(|l| parse_f64_bits_line(l).into_iter().map(|v| v as f32).collect())
            .expect("pred line");
        let rust_pred: Vec<f32> = corpus
            .features
            .iter()
            .map(|row| booster.predict_row(row)[0])
            .collect();
        compare_within(&rust_pred, &golden_pred, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("predict() not within ORACLE_TOL vs golden: {m:?}"));
    }
}

#[test]
fn score_accumulation() {
    // L2: the internal per-iter score_ (== predict(raw_score=True, num_iteration=k))
    // replays BIT-EXACT f64 vs `regression_scores.txt` for every k=1..N and every
    // row. This GATE resolves the phase-wide L2 precision contract: BIT-EXACT.
    let Some(scores_text) = read_golden("regression_scores.txt") else {
        return;
    };
    let (booster, _corpus) = train_spine();
    let golden_lines: Vec<Vec<f64>> = scores_text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(parse_f64_bits_line)
        .collect();
    assert_eq!(
        golden_lines.len(),
        booster.iter_scores.len(),
        "iteration count: golden {} != rust {}",
        golden_lines.len(),
        booster.iter_scores.len()
    );
    for (k, (rust_k, golden_k)) in booster
        .iter_scores
        .iter()
        .zip(golden_lines.iter())
        .enumerate()
    {
        compare_exact_f64_bits(rust_k, golden_k)
            .unwrap_or_else(|m| panic!("iter {} per-row score not bit-exact: {m:?}", k + 1));
    }
}

#[test]
fn gradients() {
    // L1: per-row grad/hess at iter 1 and a later iter, within ORACLE_TOL vs the
    // captured reference (`regression_gh_iter1.txt` / `regression_gh_iterN.txt`).
    let (booster, _corpus) = train_spine();

    // iter 1.
    if let Some(gh1) = read_golden("regression_gh_iter1.txt") {
        let (g_golden, h_golden) = parse_gh(&gh1);
        let (g_rust, h_rust) = &booster.iter_grad_hess[0];
        compare_within(g_rust, &g_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("iter-1 grad not within ORACLE_TOL: {m:?}"));
        compare_within(h_rust, &h_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("iter-1 hess not within ORACLE_TOL: {m:?}"));
    }

    // a later iter (the capture's LATER_ITER = 5; iter index 4).
    if let Some(ghn) = read_golden("regression_gh_iterN.txt") {
        let (g_golden, h_golden) = parse_gh(&ghn);
        let later = 5usize;
        let (g_rust, h_rust) = &booster.iter_grad_hess[later - 1];
        compare_within(g_rust, &g_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("iter-{later} grad not within ORACLE_TOL: {m:?}"));
        compare_within(h_rust, &h_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("iter-{later} hess not within ORACLE_TOL: {m:?}"));
    }
}

/// Parse a `GRAD …\nHESS …` g/h golden (u32 bit patterns) into `(grad, hess)`.
fn parse_gh(text: &str) -> (Vec<f32>, Vec<f32>) {
    let mut grad = Vec::new();
    let mut hess = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("GRAD ") {
            grad = parse_f32_bits_line(rest);
        } else if let Some(rest) = line.strip_prefix("HESS ") {
            hess = parse_f32_bits_line(rest);
        }
    }
    (grad, hess)
}

#[test]
#[ignore = "MISSING — implemented in wave 4 (06-05): early-stopping / eval-history parity"]
fn early_stopping() {
    // L3: eval-history + best_iteration vs `record_evaluation` (D-12).
    panic!("MISSING — implemented in wave 4 (06-05)");
}

#[test]
#[ignore = "MISSING — implemented in wave 3 (06-03+): custom (D-04 closure) objective parity"]
fn custom_objective() {
    // The D-04 custom-objective closure path produces the same g/h + tree.
    panic!("MISSING — implemented in wave 3 (06-03)");
}

#[test]
#[ignore = "MISSING — implemented in wave 4 (06-05): bagging RNG-replay (D-13 Option A) bagged-index parity"]
fn bagging_rng() {
    // L4: bagged row indices derived in-Rust from the replayed RNG sequence,
    // asserted exact (compare_exact_u32) against the captured bag.
    panic!("MISSING — implemented in wave 4 (06-05)");
}

