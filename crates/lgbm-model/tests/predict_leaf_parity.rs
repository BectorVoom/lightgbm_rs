//! D-06 layer 4 (PRD-03): leaf-index prediction matches the C++ reference EXACTLY.
//!
//! Load the committed C++ `model.txt`, run
//! [`lgbm_model::predict::predict_leaf_index_mat`] on the committed input matrix,
//! and assert the per-(row×iter×class) leaf-id layout equals the committed
//! `leaf.txt` golden bit-for-bit (`compare_exact_u32`).
//!
//! - `regression/` — `num_tree_per_iteration==1`: per row, `num_iteration`
//!   leaf ids `[iter0, iter1, ...]`.
//! - `multiclass/` — `num_tree_per_iteration==num_class==3`: per row,
//!   `num_iteration * num_class` leaf ids in the
//!   `[iter0_class0, iter0_class1, iter0_class2, iter1_class0, ...]` stride
//!   (Pitfall 8) — the layer that proves the stride layout.
//!
//! `compare_exact_u32` is NOT re-exported at the crate root (PATTERNS lines
//! 345-351) — import it via the full module path.
//!
//! Fixture provenance: `CARGO_MANIFEST_DIR`; graceful SKIP pre-capture; never a
//! path under `LightGBM/`.

mod golden;

use oracle_harness::comparator::{compare_exact_u32, Mismatch};

use golden::{corpus_dir, example_train_path, load_svm_like, parse_u32_row};

fn train_file(corpus: &str) -> &'static str {
    match corpus {
        "binary" | "categorical" => "binary.train",
        _ => "regression.train",
    }
}

fn leaf_corpus(corpus: &str) {
    let model_path = corpus_dir(corpus).join("model.txt");
    let golden_path = corpus_dir(corpus).join("leaf.txt");
    let input_path = example_train_path(train_file(corpus));

    let (Ok(model_text), Ok(golden_text)) = (
        std::fs::read_to_string(&model_path),
        std::fs::read_to_string(&golden_path),
    ) else {
        eprintln!(
            "predict_leaf_parity[{corpus}]: SKIP — golden(s) not found ({} / {}). \
             Run `cargo run -p xtask -- model-capture`.",
            model_path.display(),
            golden_path.display()
        );
        return;
    };

    let model = lgbm_model::model_text::load(&model_text)
        .unwrap_or_else(|e| panic!("[{corpus}] load model: {e}"));

    let (data, num_rows, num_cols) = load_svm_like(&input_path);

    let mut cpp: Vec<u32> = Vec::new();
    let mut golden_rows = 0i32;
    let mut width = 0usize;
    for line in golden_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = parse_u32_row(line);
        if width == 0 {
            width = row.len();
        } else {
            assert_eq!(row.len(), width, "[{corpus}] ragged leaf golden row");
        }
        cpp.extend_from_slice(&row);
        golden_rows += 1;
    }
    assert_eq!(
        golden_rows, num_rows,
        "[{corpus}] golden rows {golden_rows} != input rows {num_rows}"
    );

    let rust = lgbm_model::predict::predict_leaf_index_mat(&model, &data, num_rows, num_cols)
        .unwrap_or_else(|e| panic!("[{corpus}] predict_leaf_index_mat: {e}"));

    match compare_exact_u32(&rust, &cpp) {
        Ok(()) => {}
        Err(m) => {
            let hint = match &m {
                Mismatch::ExactMismatch { index, .. } => {
                    format!(" (row {}, slot {} of {width})", index / width, index % width)
                }
                _ => String::new(),
            };
            panic!("[{corpus}] leaf-index diverged{hint}: {m}");
        }
    }
}

#[test]
fn regression_leaf_index_matches_cpp() {
    leaf_corpus("regression");
}

#[test]
fn multiclass_leaf_index_matches_cpp() {
    leaf_corpus("multiclass");
}
