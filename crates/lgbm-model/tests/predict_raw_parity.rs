//! D-06 layer 2 (PRD-01): raw-score prediction matches the C++ reference within
//! `ORACLE_TOL` (~1e-6 f32), on DENSE + CSR + CSC inputs.
//!
//! Load the committed regression `model.txt`, predict raw scores on the example
//! input matrix the model was trained + predicted on (`regression.train`), and
//! compare each input form against the committed `raw.txt` golden (per-row f64
//! bit patterns). Accumulation is f64 internally (in `predict_raw`); the f32 cast
//! happens only at the output boundary.
//!
//! Fixture provenance (project memory `lightgbm-ref-tree-untracked`): resolved
//! via `CARGO_MANIFEST_DIR`; graceful SKIP pre-capture. Never references a path
//! under `LightGBM/`.

mod golden;

use oracle_harness::{compare_within, ORACLE_TOL};

use golden::{
    corpus_dir, dense_to_csc, dense_to_csr, example_train_path, load_svm_like, parse_f64_bits_row,
};

/// Load `raw.txt` (one `;`-row of f64 bits per data row) and flatten to a single
/// `f32` vector in row-major `[row0_class0.., row1_class0..]` order — matching the
/// predict driver's output layout.
fn load_raw_golden(corpus: &str) -> Option<Vec<f32>> {
    let path = corpus_dir(corpus).join("raw.txt");
    let text = std::fs::read_to_string(&path).ok()?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        for v in parse_f64_bits_row(line) {
            out.push(v as f32);
        }
    }
    Some(out)
}

#[test]
fn regression_raw_parity_dense_csr_csc() {
    let corpus = "regression";
    let model_path = corpus_dir(corpus).join("model.txt");
    let train_path = example_train_path("regression.train");

    let (Ok(model_text), true) = (std::fs::read_to_string(&model_path), train_path.exists()) else {
        eprintln!(
            "predict_raw_parity[{corpus}]: SKIP — model {} or input {} not found. \
             Run `cargo run -p xtask -- model-capture`.",
            model_path.display(),
            train_path.display()
        );
        return;
    };
    let Some(golden) = load_raw_golden(corpus) else {
        eprintln!("predict_raw_parity[{corpus}]: SKIP — raw.txt golden not found.");
        return;
    };

    let model = lgbm_model::model_text::load(&model_text)
        .unwrap_or_else(|e| panic!("[{corpus}] load model: {e}"));
    let (data, num_rows, num_cols) = load_svm_like(&train_path);
    assert_eq!(
        num_cols,
        model.max_feature_idx + 1,
        "[{corpus}] input has {num_cols} features, model expects {}",
        model.max_feature_idx + 1
    );
    assert_eq!(
        golden.len(),
        num_rows as usize * model.num_tree_per_iteration as usize,
        "[{corpus}] raw.txt golden length mismatch"
    );

    // --- dense ---
    let dense = lgbm_model::predict::predict_raw_mat(&model, &data, num_rows, num_cols)
        .unwrap_or_else(|e| panic!("[{corpus}] dense predict: {e}"));
    if let Err(m) = compare_within(&dense, &golden, ORACLE_TOL) {
        panic!("[{corpus}] DENSE raw-score parity failed vs raw.txt: {m}");
    }

    // --- CSR (every entry stored explicitly) ---
    let (rp, ri, rv) = dense_to_csr(&data, num_rows, num_cols);
    let csr = lgbm_model::predict::predict_raw_csr(&model, &rp, &ri, &rv, num_rows, num_cols)
        .unwrap_or_else(|e| panic!("[{corpus}] csr predict: {e}"));
    if let Err(m) = compare_within(&csr, &golden, ORACLE_TOL) {
        panic!("[{corpus}] CSR raw-score parity failed vs raw.txt: {m}");
    }

    // --- CSC ---
    let (cp, ci, cv) = dense_to_csc(&data, num_rows, num_cols);
    let csc = lgbm_model::predict::predict_raw_csc(&model, &cp, &ci, &cv, num_rows, num_cols)
        .unwrap_or_else(|e| panic!("[{corpus}] csc predict: {e}"));
    if let Err(m) = compare_within(&csc, &golden, ORACLE_TOL) {
        panic!("[{corpus}] CSC raw-score parity failed vs raw.txt: {m}");
    }
}
