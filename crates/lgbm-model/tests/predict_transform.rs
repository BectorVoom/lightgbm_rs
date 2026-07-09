//! D-06 layer 3 (PRD-02): transformed prediction matches the C++ reference.
//!
//! Load the committed C++ `model.txt`, run the TRANSFORMED batch driver
//! ([`lgbm_model::predict::predict_mat`]) on the committed input matrix, and
//! assert the per-row per-class output is within `ORACLE_TOL` of the committed
//! `transformed.txt` golden (`compare_within`).
//!
//! Covered objectives:
//! - `regression`  (identity)            — `regression/`
//! - `binary`      (sigmoid)             — `binary/`
//! - `multiclass`  (softmax, 3 classes)  — `multiclass/`, exercises the per-class
//!   `num_tree_per_iteration==num_class` stride + the softmax max-subtraction.
//!
//! Golden format (`golden::parse_f64_bits_row`): one line per row, `;`-separated
//! f64 bit patterns; 1 value for `num_class==1`, `num_class` values otherwise.
//!
//! Fixture provenance (project memory `lightgbm-ref-tree-untracked`): resolved via
//! `CARGO_MANIFEST_DIR`; graceful SKIP pre-capture; never a path under `LightGBM/`.

mod golden;

use oracle_harness::{compare_within, ORACLE_TOL};

use golden::{corpus_dir, example_train_path, load_svm_like, parse_f64_bits_row};

/// The example `.train` table each corpus was trained + predicted on. Binary
/// reuses `binary.train`; everything else reuses `regression.train`.
fn train_file(corpus: &str) -> &'static str {
    match corpus {
        "binary" | "categorical" => "binary.train",
        _ => "regression.train",
    }
}

fn transform_corpus(corpus: &str) {
    let model_path = corpus_dir(corpus).join("model.txt");
    let golden_path = corpus_dir(corpus).join("transformed.txt");
    let input_path = example_train_path(train_file(corpus));

    let (Ok(model_text), Ok(golden_text)) = (
        std::fs::read_to_string(&model_path),
        std::fs::read_to_string(&golden_path),
    ) else {
        eprintln!(
            "predict_transform[{corpus}]: SKIP — golden(s) not found ({} / {}). \
             Run `cargo run -p xtask -- model-capture`.",
            model_path.display(),
            golden_path.display()
        );
        return;
    };

    let model = lgbm_model::model_text::load(&model_text)
        .unwrap_or_else(|e| panic!("[{corpus}] load model: {e}"));

    let (data, num_rows, num_cols) = load_svm_like(&input_path);

    // Flatten the golden into row-major f32 (the predict_mat output layout).
    let mut cpp: Vec<f32> = Vec::new();
    let mut golden_rows = 0i32;
    let mut width = 0usize;
    for line in golden_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = parse_f64_bits_row(line);
        if width == 0 {
            width = row.len();
        } else {
            assert_eq!(row.len(), width, "[{corpus}] ragged golden row");
        }
        for v in row {
            cpp.push(v as f32);
        }
        golden_rows += 1;
    }
    assert_eq!(
        golden_rows, num_rows,
        "[{corpus}] golden rows {golden_rows} != input rows {num_rows}"
    );

    let rust = lgbm_model::predict::predict_mat(&model, &data, num_rows, num_cols)
        .unwrap_or_else(|e| panic!("[{corpus}] predict_mat: {e}"));

    match compare_within(&rust, &cpp, ORACLE_TOL) {
        Ok(()) => {}
        Err(m) => {
            // Localize: width is num outputs per row.
            let idx_hint = match &m {
                oracle_harness::comparator::Mismatch::ValueMismatch { index, .. } => {
                    format!(" (row {}, class {})", index / width, index % width)
                }
                _ => String::new(),
            };
            panic!("[{corpus}] transformed predict diverged{idx_hint}: {m}");
        }
    }
}

#[test]
fn regression_transformed_matches_cpp() {
    transform_corpus("regression");
}

#[test]
fn binary_transformed_matches_cpp() {
    transform_corpus("binary");
}

#[test]
fn multiclass_transformed_matches_cpp() {
    transform_corpus("multiclass");
}
