//! Predict-mode parity replay (Phase 7, Wave 9 — plan 07-10, PRD-04 + PRD-05).
//!
//! Two NEW prediction modes are validated against real `lib_lightgbm` 4.6:
//!
//! - **PRD-04 TreeSHAP `predict_contrib`** — `contrib_*` cells load the captured
//!   model + input matrix, run [`lgbm_model::predict::predict_contrib_mat`], and
//!   assert (1) the per-class block (`[per-feature; expected-value base]`) matches
//!   the real-binary `pred_contrib=True` golden within `ORACLE_TOL`, AND (2) the
//!   load-bearing INVARIANT `Σ block == raw margin` holds (sum of each class block
//!   equals [`predict_raw_mat`]). Covers numeric, categorical (TreeSHAP over
//!   `find_in_bitset` decision nodes), and multiclass (per-class stride) trees.
//!
//! - **PRD-05 prediction early stopping** — `early_stop_*` cells replay the
//!   captured freq×margin axis through [`GbdtModel::predict_raw_early_stop`] and
//!   assert the frozen raw score matches the real-binary `pred_early_stop` golden
//!   within `ORACLE_TOL`.
//!
//! Idioms mirror `boosting_parity.rs` / `learner_parity.rs`: a
//! `CARGO_MANIFEST_DIR`-rooted fixture path (NEVER the untracked `LightGBM/`
//! tree), the `compare_within(.., ORACLE_TOL)` precision contract, and a graceful
//! SKIP (return) when a golden file is absent so a fresh checkout without the
//! capture run still builds + passes. Populate the goldens with
//! `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- predict-mode-oracle-capture`.

use std::path::PathBuf;

use lgbm_model::GbdtModel;
use oracle_harness::comparator::{compare_within, ORACLE_TOL};

/// The committed predict-mode golden directory — TRACKED under the oracle-harness
/// crate, NEVER the untracked C++/LightGBM reference tree.
fn predict_modes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/predict_modes")
}

/// SKIP gracefully (returning `None`) when a golden file is absent — matching the
/// boosting_parity / learner_parity idiom.
fn read_golden(corpus: &str, file: &str) -> Option<String> {
    let path = predict_modes_dir().join(corpus).join(file);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "predict_parity: SKIP — golden {} not found. Run \
                 `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- predict-mode-oracle-capture`.",
                path.display()
            );
            None
        }
    }
}

/// Parse a `# rows=<n> <key>=<m>` header + body of space-separated f64 bit
/// patterns into `(rows, cols, flat f64 row-major)`.
fn parse_matrix(text: &str) -> (usize, usize, Vec<f64>) {
    let mut lines = text.lines();
    let header = lines.next().expect("matrix header");
    let (rows, cols) = parse_dims(header);
    let mut flat = Vec::with_capacity(rows * cols);
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for tok in line.split_whitespace() {
            flat.push(f64::from_bits(tok.parse::<u64>().expect("u64 bits")));
        }
    }
    assert_eq!(flat.len(), rows * cols, "matrix body length != rows*cols");
    (rows, cols, flat)
}

/// Parse `rows=<n>` and the second `=<m>` (cols/width) out of a header line.
fn parse_dims(header: &str) -> (usize, usize) {
    let mut rows = 0usize;
    let mut second = 0usize;
    for tok in header.split_whitespace() {
        if let Some(v) = tok.strip_prefix("rows=") {
            rows = v.parse().expect("rows");
        } else if let Some(v) = tok.strip_prefix("cols=") {
            second = v.parse().expect("cols");
        } else if let Some(v) = tok.strip_prefix("width=") {
            second = v.parse().expect("width");
        }
    }
    (rows, second)
}

/// Load the corpus model + input matrix; returns `None` to SKIP when absent.
fn load_corpus(corpus: &str) -> Option<(GbdtModel, usize, usize, Vec<f32>)> {
    let model_text = read_golden(corpus, "model.txt")?;
    let x_text = read_golden(corpus, "X.txt")?;
    let model = lgbm_model::model_text::load(&model_text).expect("parse golden model");
    let (rows, cols, x_f64) = parse_matrix(&x_text);
    let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
    Some((model, rows, cols, x_f32))
}

/// Run a contrib parity cell for `corpus`: compare `predict_contrib_mat` to the
/// real-binary golden within ORACLE_TOL AND assert the sum+base==raw invariant.
fn run_contrib_cell(corpus: &str) {
    let Some((model, rows, cols, x)) = load_corpus(corpus) else {
        return;
    };
    let Some(golden_text) = read_golden(corpus, "contrib.txt") else {
        return;
    };
    let (g_rows, g_width, golden) = parse_matrix(&golden_text);
    assert_eq!(g_rows, rows, "{corpus}: contrib golden row count mismatch");

    let rust = lgbm_model::predict::predict_contrib_mat(&model, &x, rows as i32, cols as i32)
        .expect("predict_contrib_mat");
    let per_row = rust.len() / rows;
    assert_eq!(
        per_row, g_width,
        "{corpus}: contrib width rust {per_row} != golden {g_width}"
    );

    // (1) Within ORACLE_TOL vs the real-binary golden (f32 contract).
    let rust_f32: Vec<f32> = rust.iter().map(|&v| v as f32).collect();
    let golden_f32: Vec<f32> = golden.iter().map(|&v| v as f32).collect();
    compare_within(&rust_f32, &golden_f32, ORACLE_TOL)
        .unwrap_or_else(|m| panic!("{corpus}: contrib not within ORACLE_TOL vs golden: {m:?}"));

    // (2) The load-bearing INVARIANT: per class block sum == raw margin.
    let raw = lgbm_model::predict::predict_raw_mat(&model, &x, rows as i32, cols as i32)
        .expect("predict_raw_mat");
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let nf = (model.max_feature_idx + 1).max(0) as usize;
    let block = nf + 1;
    for r in 0..rows {
        for k in 0..ntpi {
            let off = r * per_row + k * block;
            let sum: f64 = rust[off..off + block].iter().sum();
            let raw_k = raw[r * ntpi + k] as f64;
            assert!(
                (sum - raw_k).abs() <= 1e-5,
                "{corpus}: row {r} class {k} SHAP invariant: sum {sum} != raw {raw_k}"
            );
        }
    }
}

#[test]
fn contrib_numeric() {
    run_contrib_cell("numeric");
}

#[test]
fn contrib_categorical() {
    run_contrib_cell("categorical");
}

#[test]
fn contrib_multiclass() {
    run_contrib_cell("multiclass");
}

/// Run the early-stop freq×margin axis for `corpus`: replay each captured cell
/// through `predict_raw_early_stop` and compare the frozen raw score within
/// ORACLE_TOL.
fn run_early_stop_cell(corpus: &str) {
    let Some((model, rows, cols, x)) = load_corpus(corpus) else {
        return;
    };
    let Some(golden_text) = read_golden(corpus, "early_stop.txt") else {
        return;
    };

    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let width = (model.max_feature_idx + 1).max(0) as usize;

    // Parse the CELL blocks: `CELL freq=<f> margin=<g> rows=<n> width=<w>` then rows.
    let mut lines = golden_text.lines().peekable();
    let mut any_cell = false;
    while let Some(header) = lines.next() {
        let header = header.trim();
        if !header.starts_with("CELL") {
            continue;
        }
        any_cell = true;
        let mut freq = 0i32;
        let mut margin = 0.0f64;
        let (g_rows, _g_width) = parse_dims(header);
        for tok in header.split_whitespace() {
            if let Some(v) = tok.strip_prefix("freq=") {
                freq = v.parse().expect("freq");
            } else if let Some(v) = tok.strip_prefix("margin=") {
                margin = v.parse().expect("margin");
            }
        }
        assert_eq!(g_rows, rows, "{corpus}: early_stop golden row count mismatch");

        // Read `rows` body lines into the golden score matrix.
        let mut golden = Vec::with_capacity(rows * ntpi);
        for _ in 0..rows {
            let body = lines.next().expect("early_stop body line").trim();
            for tok in body.split_whitespace() {
                golden.push(f64::from_bits(tok.parse::<u64>().expect("u64 bits")));
            }
        }

        // Replay each row through predict_raw_early_stop.
        let mut rust = Vec::with_capacity(rows * ntpi);
        let mut row_buf = vec![0.0f64; width];
        for r in 0..rows {
            for (c, slot) in row_buf.iter_mut().enumerate() {
                *slot = x[r * cols + c] as f64;
            }
            let (scores, _iters) =
                model.predict_raw_early_stop(&row_buf, 0, -1, freq, margin);
            rust.extend(scores);
        }

        let rust_f32: Vec<f32> = rust.iter().map(|&v| v as f32).collect();
        let golden_f32: Vec<f32> = golden.iter().map(|&v| v as f32).collect();
        compare_within(&rust_f32, &golden_f32, ORACLE_TOL).unwrap_or_else(|m| {
            panic!(
                "{corpus}: early_stop freq={freq} margin={margin} not within ORACLE_TOL: {m:?}"
            )
        });
    }
    assert!(any_cell, "{corpus}: early_stop.txt had no CELL blocks");
}

#[test]
fn early_stop_numeric() {
    run_early_stop_cell("numeric");
}

#[test]
fn early_stop_multiclass() {
    run_early_stop_cell("multiclass");
}
