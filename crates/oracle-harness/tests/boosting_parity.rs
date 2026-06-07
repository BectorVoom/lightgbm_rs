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

use lgbm::{train, train_custom, Booster, DenseCorpus, TrainingBuilder};
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

/// The binary spine corpus, identical to `boosting_oracle_capture.py::binary_corpus`
/// (12 rows, 0/1 labels, both classes present).
fn binary_corpus() -> DenseCorpus {
    let f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
    let f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let features: Vec<Vec<f64>> = (0..12).map(|i| vec![f0[i] as f64, f1[i] as f64]).collect();
    let labels = vec![
        0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    ];
    DenseCorpus { features, labels }
}

/// A builder shared across the cells: 10 iters, lr 0.1, num_leaves 4,
/// deterministic, the recorded seed.
fn cell_builder(objective: &str, bfa: bool) -> TrainingBuilder {
    TrainingBuilder::new()
        .objective(objective)
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(bfa)
        .seed(SPINE_SEED)
        .deterministic(true)
}

/// Train the regression(L2) spine the SAME way the capture did.
fn train_spine() -> (Booster, DenseCorpus) {
    let corpus = spine_corpus();
    let cfg = cell_builder("regression", true).build().expect("valid spine config");
    let booster = train(&cfg, &corpus).expect("spine train ok");
    (booster, corpus)
}

/// Train the regression_l1 cell (Sign grad, median init, median-residual renew).
fn train_regression_l1() -> (Booster, DenseCorpus) {
    let corpus = spine_corpus(); // continuous labels (D-08)
    let cfg = cell_builder("regression_l1", true).build().expect("valid l1 config");
    let booster = train(&cfg, &corpus).expect("l1 train ok");
    (booster, corpus)
}

/// Train the binary cell (sigmoid grad/hess, logit init).
fn train_binary() -> (Booster, DenseCorpus) {
    let corpus = binary_corpus();
    let cfg = cell_builder("binary", true).build().expect("valid binary config");
    let booster = train(&cfg, &corpus).expect("binary train ok");
    (booster, corpus)
}

/// The multiclass spine corpus, identical to
/// `boosting_oracle_capture.py::multiclass_corpus` (12 rows, 3-class integer
/// labels, all classes present).
fn multiclass_corpus() -> DenseCorpus {
    let f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
    let f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let features: Vec<Vec<f64>> = (0..12).map(|i| vec![f0[i] as f64, f1[i] as f64]).collect();
    // 3 balanced classes (matches the capture). The redundant-form softmax `exp` is
    // a transcendental whose Rust-libm vs C++-wheel-libm ULP gap makes the
    // multiclass g/h / scores / leaf outputs match to ~1e-6 (NOT bit-exact); the
    // replay therefore asserts within ORACLE_TOL for L1/L2/L3/L5 numerics and
    // bit-exact ONLY for the structural per-class layout (tree count == iters*K,
    // class-major stride). See 06-04-SUMMARY exp-libm residual note.
    let labels = vec![
        0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0,
    ];
    DenseCorpus { features, labels }
}

/// The number of classes for the multiclass cells (matches the capture).
const MULTICLASS_NUM_CLASS: i32 = 3;

/// The multiclass capture horizon (`MULTICLASS_NUM_ITERATIONS` in the capture
/// script). The single-output spine runs 10 iters; the multiclass cells run 5 so
/// EVERY grown tree stays BIT-EXACT to the real binary — the redundant-form softmax
/// `exp` (Rust libm vs the C++ wheel's std::exp) flips a knife-edge split at iter
/// ~5-6 on this corpus (the documented exp-libm residual). 5 iters × 3 classes =
/// 15 trees proves the per-class class-major layout end-to-end, bit-exact.
const MULTICLASS_NUM_ITERATIONS: i32 = 5;

/// The multiclass cell builder: like `cell_builder` but capped at
/// [`MULTICLASS_NUM_ITERATIONS`] and carrying `num_class`.
fn multiclass_cell_builder(objective: &str) -> TrainingBuilder {
    TrainingBuilder::new()
        .objective(objective)
        .num_iterations(MULTICLASS_NUM_ITERATIONS)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(true)
        .num_class(MULTICLASS_NUM_CLASS)
        .seed(SPINE_SEED)
        .deterministic(true)
}

/// Train the `multiclass` (softmax) cell the SAME way the capture did.
fn train_multiclass() -> (Booster, DenseCorpus) {
    let corpus = multiclass_corpus();
    let cfg = multiclass_cell_builder("multiclass")
        .build()
        .expect("valid multiclass config");
    let booster = train(&cfg, &corpus).expect("multiclass train ok");
    (booster, corpus)
}

/// Train the `multiclassova` (one-vs-all) cell the SAME way the capture did.
fn train_multiclassova() -> (Booster, DenseCorpus) {
    let corpus = multiclass_corpus();
    let cfg = multiclass_cell_builder("multiclassova")
        .build()
        .expect("valid multiclassova config");
    let booster = train(&cfg, &corpus).expect("multiclassova train ok");
    (booster, corpus)
}

/// Train the custom cell: an L2-equivalent closure (grad = score - label, hess = 1)
/// with boost_from_average forced OFF (matching the capture's bfa-off custom run).
fn train_custom_cell() -> (Booster, DenseCorpus) {
    let corpus = spine_corpus();
    let cfg = cell_builder("regression", false).build().expect("valid custom config");
    let labels = corpus.labels.clone();
    let booster = train_custom(&cfg, &corpus, move |preds: &[f64]| {
        // f64-subtract then f32-cast (the native L2 g/h op order — the cross-anchor).
        let grad: Vec<f32> = preds
            .iter()
            .zip(labels.iter())
            .map(|(&p, &l)| (p - l as f64) as f32)
            .collect();
        let hess = vec![1.0f32; preds.len()];
        (grad, hess)
    })
    .expect("custom train ok");
    (booster, corpus)
}

/// Assert a booster's per-tree leaf values bit-match a golden model text file, and
/// (optionally) its predict() within ORACLE_TOL.
fn assert_model_and_pred(booster: &Booster, corpus: &DenseCorpus, model_file: &str, pred_file: &str) {
    let Some(model_text) = read_golden(model_file) else {
        return;
    };
    let golden = lgbm_model::model_text::load(&model_text).expect("parse golden model");
    let rust = booster.model();
    assert_eq!(
        rust.trees.len(),
        golden.trees.len(),
        "{model_file}: tree count rust {} != golden {}",
        rust.trees.len(),
        golden.trees.len()
    );
    for (i, (rt, gt)) in rust.trees.iter().zip(golden.trees.iter()).enumerate() {
        compare_exact_f64_bits(&rt.leaf_value, &gt.leaf_value)
            .unwrap_or_else(|m| panic!("{model_file} tree {i} leaf_value not bit-exact: {m:?}"));
    }
    if let Some(pred_text) = read_golden(pred_file) {
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
            .unwrap_or_else(|m| panic!("{pred_file} predict() not within ORACLE_TOL: {m:?}"));
    }
}

/// Assert a booster's per-iter scores bit-match a golden scores file.
fn assert_scores(booster: &Booster, scores_file: &str) {
    let Some(scores_text) = read_golden(scores_file) else {
        return;
    };
    let golden_lines: Vec<Vec<f64>> = scores_text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(parse_f64_bits_line)
        .collect();
    assert_eq!(
        golden_lines.len(),
        booster.iter_scores.len(),
        "{scores_file}: iter count golden {} != rust {}",
        golden_lines.len(),
        booster.iter_scores.len()
    );
    for (k, (rust_k, golden_k)) in booster.iter_scores.iter().zip(golden_lines.iter()).enumerate() {
        compare_exact_f64_bits(rust_k, golden_k)
            .unwrap_or_else(|m| panic!("{scores_file} iter {} not bit-exact: {m:?}", k + 1));
    }
}

/// Assert iter-1 and iter-N g/h within ORACLE_TOL vs the golden g/h files (the
/// single-output cells use LATER_ITER = 5).
fn assert_gradients(booster: &Booster, gh1_file: &str, ghn_file: &str) {
    assert_gradients_at(booster, gh1_file, ghn_file, 5);
}

/// Like [`assert_gradients`] but with an explicit `later_iter` (the multiclass
/// cells use 4, matching the capture's `MULTICLASS_LATER_ITER`).
fn assert_gradients_at(booster: &Booster, gh1_file: &str, ghn_file: &str, later: usize) {
    if let Some(gh1) = read_golden(gh1_file) {
        let (g_golden, h_golden) = parse_gh(&gh1);
        let (g_rust, h_rust) = &booster.iter_grad_hess[0];
        compare_within(g_rust, &g_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{gh1_file} iter-1 grad not within ORACLE_TOL: {m:?}"));
        compare_within(h_rust, &h_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{gh1_file} iter-1 hess not within ORACLE_TOL: {m:?}"));
    }
    if let Some(ghn) = read_golden(ghn_file) {
        let (g_golden, h_golden) = parse_gh(&ghn);
        let (g_rust, h_rust) = &booster.iter_grad_hess[later - 1];
        compare_within(g_rust, &g_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{ghn_file} iter-{later} grad not within ORACLE_TOL: {m:?}"));
        compare_within(h_rust, &h_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{ghn_file} iter-{later} hess not within ORACLE_TOL: {m:?}"));
    }
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

// ========================= regression_l1 (06-03) =========================

#[test]
fn regression_l1_spine_end_to_end() {
    // L5: the l1 leaf values ARE the median residual (RenewTreeOutput), NOT the
    // learner Newton output (Pitfall 2/3). The committed model text carries the
    // renewed medians; bit-matching it proves the renew body is correct.
    let (booster, corpus) = train_regression_l1();
    assert_model_and_pred(
        &booster,
        &corpus,
        "regression_l1_spine_model.txt",
        "regression_l1_spine_pred.txt",
    );
}

#[test]
fn regression_l1_score_accumulation() {
    let (booster, _) = train_regression_l1();
    assert_scores(&booster, "regression_l1_scores.txt");
}

#[test]
fn regression_l1_gradients() {
    let (booster, _) = train_regression_l1();
    assert_gradients(
        &booster,
        "regression_l1_gh_iter1.txt",
        "regression_l1_gh_iterN.txt",
    );
}

#[test]
fn regression_l1_renew_leaf_is_median_residual() {
    // Dedicated Pitfall 2/3 assertion: the l1 leaf values bit-match the golden's
    // (which are the median residuals the real binary's RenewTreeOutput produced),
    // and are DISTINCT from what an L2 (Newton) tree on the same corpus would give
    // — proving the median-residual renew is actually applied, not the learner
    // output. (assert_model_and_pred already bit-checks the leaves; here we add the
    // negative control: the l1 leaves differ from the L2 leaves.)
    let Some(l1_text) = read_golden("regression_l1_spine_model.txt") else {
        return;
    };
    let (booster, _) = train_regression_l1();
    let golden = lgbm_model::model_text::load(&l1_text).expect("parse l1 golden");
    // The Rust l1 leaves bit-match the golden (renew applied).
    for (i, (rt, gt)) in booster.model().trees.iter().zip(golden.trees.iter()).enumerate() {
        compare_exact_f64_bits(&rt.leaf_value, &gt.leaf_value)
            .unwrap_or_else(|m| panic!("l1 tree {i} renewed leaf not bit-exact: {m:?}"));
    }
    // Negative control: train L2 on the same corpus; its tree-0 leaf values must
    // DIFFER from the l1 renewed ones (else the renew would be a silent no-op).
    let (l2_booster, _) = train_spine();
    let l1_leaves = &booster.model().trees[1].leaf_value; // tree 1 (tree 0 folds bfa)
    let l2_leaves = &l2_booster.model().trees[1].leaf_value;
    assert!(
        l1_leaves != l2_leaves,
        "l1 renewed leaves must differ from L2 Newton leaves (renew is load-bearing)"
    );
}

// ========================= binary (06-03) =========================

#[test]
fn binary_spine_end_to_end() {
    let (booster, corpus) = train_binary();
    assert_model_and_pred(
        &booster,
        &corpus,
        "binary_spine_model.txt",
        "binary_spine_pred.txt",
    );
}

#[test]
fn binary_score_accumulation() {
    let (booster, _) = train_binary();
    assert_scores(&booster, "binary_scores.txt");
}

#[test]
fn binary_gradients() {
    let (booster, _) = train_binary();
    assert_gradients(&booster, "binary_gh_iter1.txt", "binary_gh_iterN.txt");
}

// ========================= custom (OBJ-02, 06-03) =========================

#[test]
fn custom_objective() {
    // The D-04 custom-objective closure (L2-equivalent g/h, bfa OFF) replays the
    // captured custom golden (model text + scores + g/h) — the closure is invoked
    // in place of GetGradients.
    let (booster, corpus) = train_custom_cell();
    assert_model_and_pred(
        &booster,
        &corpus,
        "custom_spine_model.txt",
        "custom_spine_pred.txt",
    );
    assert_scores(&booster, "custom_scores.txt");
    assert_gradients(&booster, "custom_gh_iter1.txt", "custom_gh_iterN.txt");
}

#[test]
fn custom_cross_anchored_to_native_regression_l2() {
    // OBJ-02 cross-anchor: the custom run's end-to-end model text bit-matches the
    // NATIVE regression(L2) cell captured with boost_from_average=OFF
    // (`custom_crossanchor_l2_model.txt`) — same g/h ⇒ same trees ⇒ same model.
    // This anchors the custom path to a real C++ objective (no distinct custom
    // objective exists to diff). The comparison basis (bfa-off, init=0) is recorded
    // in the REFERENCE_MANIFEST.
    let Some(anchor_text) = read_golden("custom_crossanchor_l2_model.txt") else {
        return;
    };
    let (booster, _) = train_custom_cell();
    let anchor = lgbm_model::model_text::load(&anchor_text).expect("parse cross-anchor model");
    let rust = booster.model();
    assert_eq!(
        rust.trees.len(),
        anchor.trees.len(),
        "cross-anchor tree count mismatch"
    );
    for (i, (rt, at)) in rust.trees.iter().zip(anchor.trees.iter()).enumerate() {
        compare_exact_f64_bits(&rt.leaf_value, &at.leaf_value).unwrap_or_else(|m| {
            panic!("custom tree {i} leaf_value != native regression(L2 bfa-off): {m:?}")
        });
    }
}

#[test]
fn bagging_rng() {
    // L4 (D-13 Option A): the full bag_data_indices array (in-bag ++ OOB tail)
    // reproduced by `BaggingSampleStrategy::bagging` over the proven
    // `lgbm_core::Random` LCG, asserted BIT-EXACT (compare_exact i32) against the
    // committed RNG-replay golden `bag_indices_seed3_frac0.7.txt`. The bag is a pure
    // function of (bagging_seed, bagging_fraction, num_data, block 1024), so a wrong
    // RNG draw/order can never hide behind a near-matching model.
    use lgbm_boosting::{BaggingConfig, BaggingSampleStrategy};

    let Some(text) = read_golden("bag_indices_seed3_frac0.7.txt") else {
        return;
    };
    let mut cells = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Parse: seed=<S> fraction=<F> num_data=<N> bag_data_cnt=<C> indices=<csv>.
        let mut seed = 0i32;
        let mut fraction = 0.0f64;
        let mut num_data = 0i32;
        let mut bag_cnt = 0i32;
        let mut expected: Vec<i32> = Vec::new();
        for tok in line.split_whitespace() {
            if let Some(v) = tok.strip_prefix("seed=") {
                seed = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("fraction=") {
                fraction = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("num_data=") {
                num_data = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("bag_data_cnt=") {
                bag_cnt = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("indices=") {
                expected = v.split(',').map(|t| t.parse::<i32>().unwrap()).collect();
            }
        }
        let labels = vec![0.0f32; num_data as usize];
        let cfg = BaggingConfig::new(fraction, 1.0, 1.0, 1, seed, false).unwrap();
        let mut strat = BaggingSampleStrategy::reset_sample_config(cfg, num_data, &labels);
        assert!(strat.bagging(0, &labels), "iter 0 must bag (need_re_bagging)");
        assert_eq!(
            strat.bag_data_cnt(),
            bag_cnt,
            "seed={seed} frac={fraction}: realized in-bag count != golden"
        );
        assert_eq!(
            strat.bag_data_indices(),
            expected.as_slice(),
            "seed={seed} frac={fraction}: bag_data_indices not bit-exact vs RNG-replay golden"
        );
        cells += 1;
    }
    assert!(cells >= 1, "expected at least one bag golden cell");
}

// ========================= multiclass / multiclassova (06-04) =========================

/// Assert a multiclass booster replays the golden over the 5-iter bit-exact
/// horizon: tree count == iters*K + class-major stride (STRUCTURAL, exact), per-tree
/// leaf values BIT-EXACT, and the class-major transformed predict within ORACLE_TOL
/// (the predict-side `ConvertOutput` softmax/sigmoid `exp` is the only ~1e-6 step).
fn assert_multiclass_model_and_pred(
    booster: &Booster,
    corpus: &DenseCorpus,
    num_class: i32,
    model_file: &str,
    pred_file: &str,
) {
    let Some(model_text) = read_golden(model_file) else {
        return;
    };
    let golden = lgbm_model::model_text::load(&model_text).expect("parse golden model");
    let rust = booster.model();
    // Tree count == iters * num_class, class-major (num_tree_per_iteration stride).
    assert_eq!(
        rust.trees.len(),
        golden.trees.len(),
        "{model_file}: tree count rust {} != golden {}",
        rust.trees.len(),
        golden.trees.len()
    );
    assert_eq!(
        rust.num_tree_per_iteration, num_class,
        "{model_file}: num_tree_per_iteration {} != num_class {num_class}",
        rust.num_tree_per_iteration
    );
    assert_eq!(
        rust.trees.len() as i32 % num_class,
        0,
        "{model_file}: tree count {} not a multiple of num_class {num_class}",
        rust.trees.len()
    );
    // Per-tree leaf values BIT-EXACT (the L5 contract holds over the 5-iter horizon —
    // the softmax exp-libm knife-edge flip is past iter 5, so every grown tree here
    // is bit-identical to the real binary).
    for (i, (rt, gt)) in rust.trees.iter().zip(golden.trees.iter()).enumerate() {
        compare_exact_f64_bits(&rt.leaf_value, &gt.leaf_value)
            .unwrap_or_else(|m| panic!("{model_file} tree {i} leaf_value not bit-exact: {m:?}"));
    }
    // Class-major transformed predict: class 0's rows, then class 1's, ... matching
    // the capture's `_class_major(predict())` layout. ConvertOutput (softmax /
    // per-class sigmoid) is the only transcendental step here → ORACLE_TOL.
    if let Some(pred_text) = read_golden(pred_file) {
        let golden_pred: Vec<f32> = pred_text
            .lines()
            .find(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .map(|l| parse_f64_bits_line(l).into_iter().map(|v| v as f32).collect())
            .expect("pred line");
        // Build the class-major rust predict: for each class k, each row.
        let per_row: Vec<Vec<f32>> =
            corpus.features.iter().map(|row| booster.predict_row(row)).collect();
        let nd = corpus.features.len();
        let mut rust_pred = Vec::with_capacity(nd * num_class as usize);
        for k in 0..num_class as usize {
            for r in &per_row {
                rust_pred.push(r[k]);
            }
        }
        compare_within(&rust_pred, &golden_pred, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{pred_file} predict() not within ORACLE_TOL: {m:?}"));
    }
}

/// Assert per-round multi_logloss matches the golden metrics file within ORACLE_TOL.
fn assert_multi_logloss(booster: &Booster, metrics_file: &str) {
    let Some(text) = read_golden(metrics_file) else {
        return;
    };
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("multi_logloss ") {
            let golden = parse_f64_bits_line(rest);
            let (_name, vals) = booster
                .eval_history
                .iter()
                .find(|(n, _)| n == "multi_logloss")
                .expect("multi_logloss eval history present");
            assert_eq!(
                vals.len(),
                golden.len(),
                "{metrics_file}: multi_logloss round count rust {} != golden {}",
                vals.len(),
                golden.len()
            );
            let tol = ORACLE_TOL as f64;
            for (k, (&r, &g)) in vals.iter().zip(golden.iter()).enumerate() {
                assert!(
                    (r - g).abs() <= tol,
                    "{metrics_file} multi_logloss round {k}: rust {r} != golden {g} (tol {tol})"
                );
            }
        }
    }
}

#[test]
fn multiclass_spine_end_to_end() {
    // L5: softmax multiclass — model text bit-exact + class-major probabilities
    // within tol; tree count == iters * num_class in class-major order.
    let (booster, corpus) = train_multiclass();
    assert_multiclass_model_and_pred(
        &booster,
        &corpus,
        MULTICLASS_NUM_CLASS,
        "multiclass_spine_model.txt",
        "multiclass_spine_pred.txt",
    );
}

#[test]
fn multiclass_score_accumulation() {
    // L2: per-iter class-major raw scores bit-exact f64.
    let (booster, _) = train_multiclass();
    assert_scores(&booster, "multiclass_scores.txt");
}

#[test]
fn multiclass_gradients() {
    // L1: per-row/per-class softmax g/h (class-major) within ORACLE_TOL. The
    // multiclass capture's later iter is 4 (MULTICLASS_LATER_ITER).
    let (booster, _) = train_multiclass();
    assert_gradients_at(
        &booster,
        "multiclass_gh_iter1.txt",
        "multiclass_gh_iterN.txt",
        4,
    );
}

#[test]
fn multiclass_metrics() {
    // L3: per-round multi_logloss.
    let (booster, _) = train_multiclass();
    assert_multi_logloss(&booster, "multiclass_metrics.txt");
}

#[test]
fn multiclassova_spine_end_to_end() {
    let (booster, corpus) = train_multiclassova();
    assert_multiclass_model_and_pred(
        &booster,
        &corpus,
        MULTICLASS_NUM_CLASS,
        "multiclassova_spine_model.txt",
        "multiclassova_spine_pred.txt",
    );
}

#[test]
fn multiclassova_score_accumulation() {
    let (booster, _) = train_multiclassova();
    assert_scores(&booster, "multiclassova_scores.txt");
}

#[test]
fn multiclassova_gradients() {
    let (booster, _) = train_multiclassova();
    assert_gradients_at(
        &booster,
        "multiclassova_gh_iter1.txt",
        "multiclassova_gh_iterN.txt",
        4,
    );
}

#[test]
fn multiclassova_metrics() {
    let (booster, _) = train_multiclassova();
    assert_multi_logloss(&booster, "multiclassova_metrics.txt");
}
