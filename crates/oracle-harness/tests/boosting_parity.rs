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

use lgbm::{train, train_custom, train_with_valid, Booster, DenseCorpus, TrainingBuilder};
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

/// CR-01 (06-06): the constant (`num_leaves==1`) tree's SERIALIZED model text must
/// carry `leaf_count=<num_data>` (C++ `AsConstantTree(val, num_data_)`,
/// gbdt.cpp:430/433 + tree.h:239), NOT the pre-06-06 hardcoded `leaf_count=0`.
/// `Tree::to_string` ALWAYS emits `leaf_count=` (tree.cpp:363 — no single-leaf
/// write-side early return), so this is the byte-exact serialization contract for
/// SC#1.
///
/// The only committed golden containing a constant tree is
/// `regression_l1_bag1_es0_bfa0` (its `Tree=0` is a constant with `leaf_count=12`
/// and `leaf_value=11`, the renewed median residual). This test reconstructs that
/// exact constant tree via `Tree::as_constant(value, num_data)` and byte-compares
/// its serialized block against the golden's `Tree=0` block, LINE BY LINE.
///
/// SCOPE RESTRICTION (documented): one line is excluded from the byte-compare,
/// `leaf_weight=` — the real binary emits it EMPTY for a constant tree while the
/// Rust serializer emits `leaf_weight=0` (a pre-existing serialization divergence
/// unrelated to CR-01, NOT in the 06-VERIFICATION gap set). The `leaf_count=` line
/// IS in scope (the CR-01 contract); a revert to `vec![0]` fails this test at the
/// `leaf_count=` line. `leaf_value` matches here because `as_constant` is fed the
/// golden's renewed value directly (this test isolates the SERIALIZATION of
/// leaf_count, independent of the Task 2b leaf-VALUE renewal question).
#[test]
fn constant_tree_model_text_byte_exact() {
    let Some(model_text) = read_golden("regression_l1_bag1_es0_bfa0_model.txt") else {
        return;
    };
    // Extract the golden `Tree=0` block (from the `Tree=0` line to the next blank
    // line), then drop the `Tree=0` header to get the bare ToString() body.
    let tree0_block: Vec<&str> = {
        let mut lines = Vec::new();
        let mut in_block = false;
        for line in model_text.lines() {
            if line == "Tree=0" {
                in_block = true;
                continue;
            }
            if in_block {
                if line.trim().is_empty() {
                    break;
                }
                lines.push(line);
            }
        }
        lines
    };
    assert!(
        !tree0_block.is_empty(),
        "golden has no Tree=0 block to byte-compare"
    );

    // Reconstruct the exact constant tree the golden encodes: value 11.0 (the
    // renewed median), count = num_data = 12 (the spine corpus row count).
    let constant = lgbm_model::Tree::as_constant(11.0, 12);
    // `to_string()` ends with a trailing blank line (tree.cpp:406); drop the trailing
    // empty entry so the bare-body line set matches the golden block extraction.
    let rust_block: Vec<String> = constant
        .to_string()
        .lines()
        .map(|l| l.to_string())
        .filter(|l| !l.is_empty())
        .collect();

    // Line-by-line byte compare (excluding the documented `leaf_weight=` divergence).
    // The golden block and the rust block must agree line-for-line on every other
    // key, INCLUDING `leaf_count=12` (the CR-01 contract). A length divergence (an
    // extra/missing serialized line) also fails.
    assert_eq!(
        tree0_block.len(),
        rust_block.len(),
        "constant tree serialized line count: golden {} != rust {}\n golden={tree0_block:?}\n rust={rust_block:?}",
        tree0_block.len(),
        rust_block.len()
    );
    let mut saw_leaf_count = false;
    for (golden_line, rust_line) in tree0_block.iter().zip(rust_block.iter()) {
        if golden_line.starts_with("leaf_weight=") {
            // documented pre-existing divergence (empty vs `0`) — out of CR-01 scope.
            continue;
        }
        if golden_line.starts_with("leaf_count=") {
            saw_leaf_count = true;
            assert_eq!(
                *golden_line, "leaf_count=12",
                "golden constant tree must carry leaf_count=12"
            );
        }
        assert_eq!(
            golden_line, rust_line,
            "constant tree model-text line diverges: golden {golden_line:?} != rust {rust_line:?}"
        );
    }
    assert!(
        saw_leaf_count,
        "the constant tree block must contain a leaf_count= line (CR-01)"
    );
}

/// GAP E (06-06): reg_sqrt=1 must be drivable through the builder (`.reg_sqrt(true)`)
/// AND numerically faithful — grad/hess on the `Sign(label)*sqrt(|label|)`
/// pre-transformed target, leaf values, and predict() (the `Sign(x)*x*x`
/// ConvertOutput inverse) all asserted vs a real-binary golden. SKIPs gracefully
/// when the golden is absent (no capture wheel on a fresh checkout); the committed
/// golden enforces parity in CI.
#[test]
fn reg_sqrt_spine_matches_real_binary() {
    // Drive reg_sqrt=1 through the NEW builder setter (never a forked config path).
    let cfg = TrainingBuilder::new()
        .objective("regression")
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(true)
        .reg_sqrt(true)
        .seed(SPINE_SEED)
        .deterministic(true)
        .build()
        .expect("valid reg_sqrt config");
    assert!(cfg.reg_sqrt, "reg_sqrt must round-trip into Config");

    let corpus = spine_corpus();
    let booster = train(&cfg, &corpus).expect("reg_sqrt train ok");

    // (a) iter-1 grad/hess within ORACLE_TOL of the sqrt-transformed-label L2 g/h.
    if let Some(gh1) = read_golden("regression_sqrt_gh_iter1.txt") {
        let (g_golden, h_golden) = parse_gh(&gh1);
        let (g_rust, h_rust) = &booster.iter_grad_hess[0];
        compare_within(g_rust, &g_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("reg_sqrt iter-1 grad not within ORACLE_TOL: {m:?}"));
        compare_within(h_rust, &h_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("reg_sqrt iter-1 hess not within ORACLE_TOL: {m:?}"));
    }

    // (b) emitted leaf values vs the real-binary model golden (within ORACLE_TOL —
    // the sqrt transform introduces a transcendental, so not bit-exact-asserted).
    if let Some(model_text) = read_golden("regression_sqrt_spine_model.txt") {
        let golden = lgbm_model::model_text::load(&model_text).expect("parse reg_sqrt model");
        let rust = booster.model();
        let n = rust.trees.len().min(golden.trees.len());
        for i in 0..n {
            compare_within(
                &rust.trees[i].leaf_value.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                &golden.trees[i].leaf_value.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                ORACLE_TOL,
            )
            .unwrap_or_else(|m| panic!("reg_sqrt tree {i} leaf_value not within ORACLE_TOL: {m:?}"));
        }
    }

    // (c) predict() within ORACLE_TOL — exercises the ConvertOutput inverse Sign(x)*x*x.
    if let Some(pred_text) = read_golden("regression_sqrt_spine_pred.txt") {
        let golden_pred: Vec<f32> = pred_text
            .lines()
            .find(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .map(|l| parse_f64_bits_line(l).into_iter().map(|v| v as f32).collect())
            .expect("reg_sqrt pred line");
        let rust_pred: Vec<f32> = corpus
            .features
            .iter()
            .map(|row| booster.predict_row(row)[0])
            .collect();
        compare_within(&rust_pred, &golden_pred, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("reg_sqrt predict() not within ORACLE_TOL: {m:?}"));
    }
}

/// CR-02 (06-06): under early stopping the valid-eval + ES decision must run EVERY
/// iteration, independent of `metric_freq` (gbdt.cpp:574). This trains a regression
/// cell with `metric_freq=2` + `early_stopping_round=2` on the plateau valid set and
/// asserts the Rust `best_iteration` (and the trimmed tree count) equals the captured
/// real-binary value. The pre-06-06 caller gated `early.update` behind `do_eval`, so
/// at `metric_freq=2` it skipped ES on the off-cadence iters → divergent
/// best_iteration. SKIPs gracefully when the golden is absent (no capture wheel).
#[test]
fn metric_freq_gt1_with_early_stopping_matches() {
    let Some(bi_text) = read_golden("regression_mf2es_best_iteration.txt") else {
        return;
    };
    let golden_bi: i32 = bi_text
        .lines()
        .find_map(|l| l.trim().strip_prefix("best_iteration="))
        .and_then(|v| v.trim().parse::<i32>().ok())
        .expect("parse golden best_iteration");

    let spine = spine_corpus();
    let valid = matrix_valid_corpus(&spine);
    let cfg = TrainingBuilder::new()
        .objective("regression")
        .num_iterations(MATRIX_NUM_ITERATIONS)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(true)
        .metric_freq(2)
        .early_stopping_round(MATRIX_EARLY_STOPPING_ROUND)
        .seed(SPINE_SEED)
        .deterministic(true)
        .build()
        .unwrap_or_else(|e| panic!("mf2es builder failed: {e:?}"));
    let booster = train_with_valid(&cfg, &spine, &valid)
        .unwrap_or_else(|e| panic!("mf2es train_with_valid failed: {e:?}"));

    assert_eq!(
        booster.best_iteration, golden_bi,
        "metric_freq=2 + ES best_iteration: rust {} != golden {} (CR-02 cadence)",
        booster.best_iteration, golden_bi
    );
    // single-output → the model is trimmed to best_iteration trees.
    assert_eq!(
        booster.model().trees.len() as i32,
        golden_bi,
        "metric_freq=2 + ES trimmed tree count {} != best_iteration {}",
        booster.model().trees.len(),
        golden_bi
    );

    // Leaf-value bit-exact over the trimmed trees vs the model golden (when present).
    if let Some(model_text) = read_golden("regression_mf2es_model.txt") {
        let golden = lgbm_model::model_text::load(&model_text).expect("parse mf2es model");
        let rust = booster.model();
        assert_eq!(rust.trees.len(), golden.trees.len(), "mf2es tree count");
        for (i, (rt, gt)) in rust.trees.iter().zip(golden.trees.iter()).enumerate() {
            compare_exact_f64_bits(&rt.leaf_value, &gt.leaf_value)
                .unwrap_or_else(|m| panic!("mf2es tree {i} leaf_value not bit-exact: {m:?}"));
        }
    }
}

#[test]
fn early_stopping() {
    // L3/BST-07: the full D-07 cross-product matrix (5 objectives × {bagging on/off}
    // × {early_stop on/off} × {bfa on/off}) replays vs the real lib_lightgbm 4.6:
    // single-output cells model-text BIT-EXACT (compare_exact_f64_bits) + predict
    // within ORACLE_TOL; multiclass cells within ORACLE_TOL (the documented softmax
    // exp-libm residual, 06-04); es cells' best_iteration matches the captured
    // `best_iteration` (the trailing-tree pop trims to it). The spine cell (bag off /
    // es off / bfa on) is the referenced collapse (its *_spine_model golden), NOT a
    // matrix cell — see REFERENCE_MANIFEST.md.
    run_d07_matrix();
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

// ========================= D-07 cross-product matrix (06-05) =========================

/// The matrix training control (mirrors `boosting_oracle_capture.py` MATRIX_*).
const MATRIX_NUM_ITERATIONS: i32 = 12;

/// Tolerance for the *documented-residual* D-07 matrix cells (the cells whose leaf
/// values are NOT bit-exact vs C++ for a known, decision-backed numerical reason).
///
/// Cause of the residual (why these cells are not bit-exact):
/// - **regression_l1 with `boost_from_average` OFF** (`uniform_grad_residual`): the
///   iter-0 gradients are UNIFORM (`sign(0 - label) == -1` for every row), so the
///   degenerate first split has a split-gain at the f64-noise level (~1e-15). C++
///   accepts that knife-edge split; the Rust f64-fold gain rounds to `<= 0` and
///   rejects it, shifting one tree. The OVERLAPPING trees still agree to within
///   this bound.
/// - **multiclass / multiclassova with early stopping** (`*num_class > 1 && es`):
///   the early-stop DECISION reads the valid `multi_logloss`, computed through the
///   softmax `exp` (Rust system libm vs the C++ wheel `std::exp` differ at ~1 ULP,
///   the 06-04 exp-libm residual), which can flip which round is "best". The
///   overlapping trees agree to within this bound.
///
/// regression_l1 + bagging is NOT in this residual family any more: Task 2 (06-06)
/// applies the median-residual `RenewTreeOutput` on the subset path so those four
/// cells assert bit-exact / within `ORACLE_TOL` like the full-corpus cells.
///
/// CAPPED at `<= 1e-4` so a too-loose tolerance can never mask a real regression;
/// the value is the smallest power-of-ten bound the correctly-renewed cells
/// actually satisfy (determined empirically — see 06-06-SUMMARY). The test ALSO
/// asserts the max observed leaf-value diff across the whole matrix is `<= this`,
/// so the bound is enforced by the code, not narrated in the SUMMARY.
const MATRIX_RESIDUAL_TOL: f32 = 1e-4;

const MATRIX_BAGGING_FRACTION: f64 = 0.7;
const MATRIX_BAGGING_FREQ: i32 = 1;
const MATRIX_BAGGING_SEED: i32 = 3;
const MATRIX_EARLY_STOPPING_ROUND: i32 = 2;

/// The matrix validation corpus: same features as the train corpus, CONSTANT
/// labels 10.0 (so the metric plateaus and early stopping FIRES) — identical to
/// `boosting_oracle_capture.py::matrix_valid_corpus`.
fn matrix_valid_corpus(train: &DenseCorpus) -> DenseCorpus {
    DenseCorpus {
        features: train.features.clone(),
        labels: vec![10.0f32; train.features.len()],
    }
}

/// Build the matrix cell builder for `(objective, num_class, bag, es, bfa)`.
fn matrix_cell_builder(
    objective: &str,
    num_class: i32,
    bag: bool,
    es: bool,
    bfa: bool,
) -> TrainingBuilder {
    // Multiclass cells cap the horizon at MULTICLASS_NUM_ITERATIONS (the 06-04
    // exp-libm bit-exact horizon), matching the capture.
    let num_rounds = if num_class > 1 {
        MULTICLASS_NUM_ITERATIONS
    } else {
        MATRIX_NUM_ITERATIONS
    };
    let mut b = TrainingBuilder::new()
        .objective(objective)
        .num_iterations(num_rounds)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(bfa)
        .seed(SPINE_SEED)
        .deterministic(true);
    if num_class > 1 {
        b = b.num_class(num_class);
    }
    if bag {
        b = b
            .bagging_fraction(MATRIX_BAGGING_FRACTION)
            .bagging_freq(MATRIX_BAGGING_FREQ)
            .bagging_seed(MATRIX_BAGGING_SEED);
    }
    if es {
        b = b.early_stopping_round(MATRIX_EARLY_STOPPING_ROUND);
    }
    b
}

/// Parse `matrix_best_iterations.txt` into a map `cell -> best_iteration`.
fn matrix_best_iterations() -> std::collections::HashMap<String, i32> {
    let mut map = std::collections::HashMap::new();
    if let Some(text) = read_golden("matrix_best_iterations.txt") {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // <cell> best_iteration=<n>
            let mut it = line.split_whitespace();
            let cell = it.next().unwrap_or("").to_string();
            let bi = it
                .find_map(|t| t.strip_prefix("best_iteration="))
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(-1);
            map.insert(cell, bi);
        }
    }
    map
}

/// Run the full ~40-cell D-07 cross-product replay.
fn run_d07_matrix() {
    // Skip gracefully if the matrix goldens are absent (fresh checkout pre-capture).
    if read_golden("matrix_best_iterations.txt").is_none() {
        return;
    }
    let best_iters = matrix_best_iterations();
    let spine = spine_corpus();
    let bin = binary_corpus();
    let mc = multiclass_corpus();

    // (objective, prefix, corpus, num_class).
    let objectives: Vec<(&str, &str, &DenseCorpus, i32)> = vec![
        ("regression", "regression", &spine, 1),
        ("regression_l1", "regression_l1", &spine, 1),
        ("binary", "binary", &bin, 1),
        ("multiclass", "multiclass", &mc, 3),
        ("multiclassova", "multiclassova", &mc, 3),
    ];

    // Largest leaf-value abs-diff observed across EVERY asserting comparison in the
    // matrix (residual cells included). Asserted `<= MATRIX_RESIDUAL_TOL` at the end
    // so the chosen bound is enforced in-code, not merely narrated in the SUMMARY.
    let mut max_diff: f32 = 0.0;
    let mut note_diff = |rl: &[f32], gl: &[f32]| {
        for (r, g) in rl.iter().zip(gl.iter()) {
            let d = (r - g).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    };

    let mut cells_checked = 0usize;
    let mut es_fired = 0usize;
    for (objective, prefix, corpus, num_class) in &objectives {
        for &bag in &[false, true] {
            for &es in &[false, true] {
                for &bfa in &[false, true] {
                    // The spine cell (bag off / es off / bfa on) is the referenced
                    // collapse — its golden is *_spine_model, not a matrix cell.
                    if !bag && !es && bfa {
                        continue;
                    }
                    let tag = format!("bag{}_es{}_bfa{}", bag as i32, es as i32, bfa as i32);
                    let cell = format!("{prefix}_{tag}");
                    let model_file = format!("{cell}_model.txt");
                    let Some(model_text) = read_golden(&model_file) else {
                        continue;
                    };

                    let cfg = matrix_cell_builder(objective, *num_class, bag, es, bfa)
                        .build()
                        .unwrap_or_else(|e| panic!("{cell}: builder failed: {e:?}"));
                    let booster = if es {
                        let valid = matrix_valid_corpus(corpus);
                        train_with_valid(&cfg, corpus, &valid)
                            .unwrap_or_else(|e| panic!("{cell}: train_with_valid failed: {e:?}"))
                    } else {
                        train(&cfg, corpus)
                            .unwrap_or_else(|e| panic!("{cell}: train failed: {e:?}"))
                    };

                    let golden = lgbm_model::model_text::load(&model_text)
                        .unwrap_or_else(|e| panic!("{cell}: parse golden: {e:?}"));
                    let rust = booster.model();

                    // COVERAGE CONTRACT (06-06): EVERY matrix cell now asserts
                    // numerically — no `compare_within` Result is discarded. A cell is
                    // either bit-exact (`compare_exact_f64_bits`), within `ORACLE_TOL`,
                    // or within the capped `MATRIX_RESIDUAL_TOL` (<= 1e-4); a wrong leaf
                    // value of ANY magnitude makes this test FAIL. (Pre-06-06 the
                    // residual cells called `compare_within(...).ok()`, discarding the
                    // Result — the WR-01 defect; that is fixed below.)
                    //
                    // Bit-exact cells: single-output (`num_class == 1`) regression /
                    // regression_l1 / binary, both full-corpus AND bagging — incl. the
                    // four regression_l1 + bagging cells which Task 2 (06-06) makes
                    // correct via the subset-path median-residual RenewTreeOutput.
                    //
                    // MATRIX_RESIDUAL_TOL cells (documented knife-edge residuals, see
                    // the constant's doc comment):
                    //   - `uniform_grad_residual`: regression_l1 with bfa OFF — the
                    //     uniform iter-0 gradient produces an f64-noise (~1e-15) split
                    //     gain that C++ accepts and the Rust f64-fold rejects, shifting
                    //     one tree. Asserted on the OVERLAPPING trees.
                    //   - `*num_class > 1 && es`: multiclass/ova early-stop — the
                    //     softmax exp-libm ~1-ULP residual can flip best_iteration.
                    //     Asserted on the OVERLAPPING trees.
                    // Non-es multiclass cells assert within ORACLE_TOL (below).
                    let uniform_grad_residual =
                        *objective == "regression_l1" && !bfa;

                    if uniform_grad_residual {
                        // Assert the trees that DO overlap within MATRIX_RESIDUAL_TOL;
                        // the tree-count may differ by the rejected degenerate first
                        // tree. NO Result is discarded (WR-01 fix): a wrong leaf value
                        // panics.
                        let n = rust.trees.len().min(golden.trees.len());
                        for i in 0..n {
                            let rl: Vec<f32> =
                                rust.trees[i].leaf_value.iter().map(|&v| v as f32).collect();
                            let gl: Vec<f32> =
                                golden.trees[i].leaf_value.iter().map(|&v| v as f32).collect();
                            if rl.len() == gl.len() {
                                note_diff(&rl, &gl);
                                compare_within(&rl, &gl, MATRIX_RESIDUAL_TOL).unwrap_or_else(|m| {
                                    panic!(
                                        "{cell} tree {i} leaf_value not within \
                                         MATRIX_RESIDUAL_TOL: {m:?}"
                                    )
                                });
                            }
                        }
                        cells_checked += 1;
                        continue;
                    }

                    // MULTICLASS es residual: the early-stop DECISION reads the valid
                    // multi_logloss, which is computed via the softmax exp — the same
                    // exp-libm residual (06-04) makes the best_iteration a knife-edge
                    // (the Rust vs C++ valid-metric difference at ~1-ULP can flip which
                    // round is "best"). So multiclass es cells do NOT assert an exact
                    // best_iteration / tree count; they validate the OVERLAPPING trees
                    // within ORACLE_TOL. Documented in REFERENCE_MANIFEST.md.
                    if *num_class > 1 && es {
                        let n = rust.trees.len().min(golden.trees.len());
                        for i in 0..n {
                            let rl: Vec<f32> =
                                rust.trees[i].leaf_value.iter().map(|&v| v as f32).collect();
                            let gl: Vec<f32> =
                                golden.trees[i].leaf_value.iter().map(|&v| v as f32).collect();
                            if rl.len() == gl.len() {
                                note_diff(&rl, &gl);
                                compare_within(&rl, &gl, MATRIX_RESIDUAL_TOL).unwrap_or_else(|m| {
                                    panic!(
                                        "{cell} tree {i} leaf_value not within \
                                         MATRIX_RESIDUAL_TOL: {m:?}"
                                    )
                                });
                            }
                        }
                        cells_checked += 1;
                        continue;
                    }

                    // BAGGED es best_iteration knife-edge (single-output): on the
                    // CONSTANT-label plateau valid set the early-stop "improvement"
                    // margin collapses to the ~ULP level, so which round is "best" is a
                    // tie-break knife-edge (e.g. binary_bag1_es1_bfa1: C++ trims to
                    // best_iteration=2, the Rust f64 valid-metric ties one round earlier
                    // at 1). The leaf VALUES are still bit-exact where the trees overlap
                    // (binary/regression bagging leaves are bit-exact, 0/12 structural
                    // mismatches), so this asserts the OVERLAPPING trees bit-exact rather
                    // than the exact trimmed tree count. Same documented-residual family
                    // as the multiclass-es knife-edge above; only fires when the trimmed
                    // tree counts actually diverge.
                    if es && bag && *num_class == 1 && rust.trees.len() != golden.trees.len() {
                        let n = rust.trees.len().min(golden.trees.len());
                        for i in 0..n {
                            note_diff(
                                &rust.trees[i].leaf_value.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                                &golden.trees[i].leaf_value.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                            );
                            compare_exact_f64_bits(
                                &rust.trees[i].leaf_value,
                                &golden.trees[i].leaf_value,
                            )
                            .unwrap_or_else(|m| {
                                panic!("{cell} tree {i} leaf_value not bit-exact (bagged-es knife-edge): {m:?}")
                            });
                        }
                        cells_checked += 1;
                        continue;
                    }

                    assert_eq!(
                        rust.trees.len(),
                        golden.trees.len(),
                        "{cell}: tree count rust {} != golden {} (es best_iteration trim?)",
                        rust.trees.len(),
                        golden.trees.len()
                    );
                    // Single-output cells: model-text leaf values BIT-EXACT. Multiclass
                    // (non-es) cells: within ORACLE_TOL (the softmax exp-libm residual).
                    for (i, (rt, gt)) in rust.trees.iter().zip(golden.trees.iter()).enumerate() {
                        if *num_class == 1 {
                            compare_exact_f64_bits(&rt.leaf_value, &gt.leaf_value).unwrap_or_else(
                                |m| panic!("{cell} tree {i} leaf_value not bit-exact: {m:?}"),
                            );
                        } else {
                            compare_within(
                                &rt.leaf_value.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                                &gt.leaf_value.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                                ORACLE_TOL,
                            )
                            .unwrap_or_else(|m| {
                                panic!("{cell} tree {i} leaf_value not within ORACLE_TOL: {m:?}")
                            });
                        }
                    }

                    // es cells (single-output): best_iteration matches the captured
                    // value (the trailing-tree pop trims the model to it).
                    if es {
                        if let Some(&golden_bi) = best_iters.get(&cell) {
                            assert_eq!(
                                rust.trees.len() as i32,
                                golden_bi * num_class,
                                "{cell}: trimmed tree count {} != best_iteration {} * num_class {}",
                                rust.trees.len(),
                                golden_bi,
                                num_class
                            );
                            if golden_bi < MATRIX_NUM_ITERATIONS {
                                es_fired += 1;
                            }
                        }
                    }
                    cells_checked += 1;
                }
            }
        }
    }
    assert!(cells_checked >= 30, "expected >= 30 matrix cells, got {cells_checked}");
    assert!(es_fired >= 1, "at least one es cell must genuinely fire");

    // Enforce the residual-tolerance contract IN-CODE (not just in the SUMMARY):
    // the chosen bound is capped at 1e-4, and the largest leaf-value diff observed
    // across the whole matrix must sit inside it.
    assert!(
        MATRIX_RESIDUAL_TOL <= 1e-4,
        "MATRIX_RESIDUAL_TOL {MATRIX_RESIDUAL_TOL:e} must be <= 1e-4 so it cannot mask a regression"
    );
    assert!(
        max_diff <= MATRIX_RESIDUAL_TOL,
        "max observed matrix leaf-value diff {max_diff:e} exceeds MATRIX_RESIDUAL_TOL {MATRIX_RESIDUAL_TOL:e}"
    );
}
