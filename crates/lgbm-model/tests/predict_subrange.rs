//! D-06 layer 5 (PRD-06): sub-range raw-score prediction matches the C++
//! reference within `ORACLE_TOL` (~1e-6 f32) for representative
//! `(start_iteration, num_iteration)` slices — including `num_iteration == -1`
//! (== all), a bounded count, and a non-zero start.
//!
//! Load the committed `subrange` `model.txt`, predict raw scores on the example
//! input matrix the model was trained + predicted on (`regression.train`) for
//! each `(start, num)` slice recorded in the `subrange.txt` golden, and compare
//! against the committed slice golden (per-row f64 bit patterns). The slice math
//! (`init_predict`, `gbdt.h:426-435`) is the only behavior under test here —
//! input form is orthogonal (CSR/CSC raw parity is covered in 03-02), so this
//! exercises the DENSE driver only.
//!
//! Golden format (emitted by `xtask model-capture`, byte-idempotent — 03-01):
//! the file is a flat sequence of `SLICE start=<s> num=<n>` header lines, each
//! immediately followed by ONE data line of `;`-separated f64 bit patterns
//! (one entry per data row, since `num_class == 1`).
//!
//! Fixture provenance (project memory `lightgbm-ref-tree-untracked`): resolved
//! via `CARGO_MANIFEST_DIR`; graceful SKIP pre-capture. Never references a path
//! under `LightGBM/`.

mod golden;

use oracle_harness::{compare_within, ORACLE_TOL};

use golden::{corpus_dir, example_train_path, load_svm_like, parse_f64_bits_row};

/// One `(start_iteration, num_iteration)` slice + its committed C++ raw golden
/// (flattened to `f32` in row-major order, matching the driver's output layout).
struct Slice {
    start: i32,
    num: i32,
    golden: Vec<f32>,
}

/// Parse `subrange.txt`: pairs of `SLICE start=<s> num=<n>` headers each followed
/// by one `;`-separated f64-bits data line. Returns the recorded slices in order.
fn load_subrange_golden(corpus: &str) -> Option<Vec<Slice>> {
    let path = corpus_dir(corpus).join("subrange.txt");
    let text = std::fs::read_to_string(&path).ok()?;
    let mut lines = text.lines();
    let mut slices = Vec::new();
    while let Some(header) = lines.next() {
        let header = header.trim();
        if header.is_empty() {
            continue;
        }
        assert!(
            header.starts_with("SLICE "),
            "[{corpus}] subrange.txt: expected `SLICE ...` header, got {header:?}"
        );
        // `SLICE start=<s> num=<n>`
        let mut start: Option<i32> = None;
        let mut num: Option<i32> = None;
        for tok in header.split_whitespace().skip(1) {
            if let Some(v) = tok.strip_prefix("start=") {
                start = Some(v.parse().unwrap_or_else(|_| panic!("bad start {tok:?}")));
            } else if let Some(v) = tok.strip_prefix("num=") {
                num = Some(v.parse().unwrap_or_else(|_| panic!("bad num {tok:?}")));
            }
        }
        let start = start.unwrap_or_else(|| panic!("[{corpus}] missing start= in {header:?}"));
        let num = num.unwrap_or_else(|| panic!("[{corpus}] missing num= in {header:?}"));

        let data = lines
            .next()
            .unwrap_or_else(|| panic!("[{corpus}] subrange.txt: header {header:?} has no data line"));
        let golden: Vec<f32> = parse_f64_bits_row(data).into_iter().map(|v| v as f32).collect();
        slices.push(Slice { start, num, golden });
    }
    Some(slices)
}

#[test]
fn subrange_raw_parity_dense() {
    let corpus = "subrange";
    let model_path = corpus_dir(corpus).join("model.txt");
    let train_path = example_train_path("regression.train");

    let (Ok(model_text), true) = (std::fs::read_to_string(&model_path), train_path.exists()) else {
        eprintln!(
            "predict_subrange[{corpus}]: SKIP — model {} or input {} not found. \
             Run `cargo run -p xtask -- model-capture`.",
            model_path.display(),
            train_path.display()
        );
        return;
    };
    let Some(slices) = load_subrange_golden(corpus) else {
        eprintln!(
            "predict_subrange[{corpus}]: SKIP — subrange.txt golden not found. \
             Run `cargo run -p xtask -- model-capture`."
        );
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

    // The golden must record at least 3 distinct (start, num) pairs incl. -1==all,
    // a bounded count, and a non-zero start (PRD-06 representative coverage).
    assert!(
        slices.len() >= 3,
        "[{corpus}] subrange.txt must record >= 3 slices, found {}",
        slices.len()
    );
    let has_all = slices.iter().any(|s| s.num <= 0); // -1 (or 0) == all-from-start
    let has_bounded = slices.iter().any(|s| s.num > 0);
    let has_nonzero_start = slices.iter().any(|s| s.start > 0);
    assert!(has_all, "[{corpus}] no `-1 == all` slice in golden");
    assert!(has_bounded, "[{corpus}] no bounded-count slice in golden");
    assert!(has_nonzero_start, "[{corpus}] no non-zero-start slice in golden");

    let ntpi = model.num_tree_per_iteration as usize;
    for s in &slices {
        assert_eq!(
            s.golden.len(),
            num_rows as usize * ntpi,
            "[{corpus}] slice (start={}, num={}) golden length {} != num_rows*ntpi {}",
            s.start,
            s.num,
            s.golden.len(),
            num_rows as usize * ntpi
        );

        let rust = lgbm_model::predict::predict_raw_mat_range(
            &model, &data, num_rows, num_cols, s.start, s.num,
        )
        .unwrap_or_else(|e| panic!("[{corpus}] (start={}, num={}) predict: {e}", s.start, s.num));

        if let Err(m) = compare_within(&rust, &s.golden, ORACLE_TOL) {
            panic!(
                "[{corpus}] DENSE sub-range raw parity failed for (start={}, num={}) vs subrange.txt: {m}",
                s.start, s.num
            );
        }
    }
}
