//! Shared test-golden loaders for the `lgbm-model` parity suites — mirrors the
//! `lgbm-dataset` crate's `tests/golden/mod.rs` keyed/`;`-separated idiom.
//!
//! Golden formats (committed by `xtask model-capture`, RESEARCH D-05/D-06):
//! - `raw.txt` / `transformed.txt`: one line per row; each line is `;`-separated
//!   f64 bit patterns (decimal `u64`) — one entry for `num_class==1`, `num_class`
//!   entries otherwise. Decoded bit-exact via [`parse_f64_bits_row`].
//! - `leaf.txt`: one line per row; `;`-separated `u32` leaf ids.
//! - The example INPUT matrix is the whitespace-delimited
//!   `<label> <f1> ... <fN>` LightGBM `.train` table (the model was trained AND
//!   predicted on it); load via [`load_svm_like`].
//!
//! Fixture provenance (project memory `lightgbm-ref-tree-untracked`): resolved
//! via `CARGO_MANIFEST_DIR`, NEVER an absolute path / a path under `LightGBM/`.

#![allow(dead_code)]

use std::path::PathBuf;

/// `crates/lgbm-model/tests/fixtures`.
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// `crates/lgbm-model/tests/fixtures/models/<corpus>`.
pub fn corpus_dir(corpus: &str) -> PathBuf {
    fixtures_dir().join("models").join(corpus)
}

/// The example INPUT `.train` table for a corpus (the matrix the model was
/// trained + predicted on). Regression/multiclass/subrange reuse
/// `examples/regression.train`; binary reuses `examples/binary.train`.
pub fn example_train_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../lgbm-dataset/tests/fixtures/examples")
        .join(file)
}

/// Parse a `;`-separated row of f64 bit patterns (decimal `u64`) into `f64`s,
/// bit-exact (`f64::from_bits`).
pub fn parse_f64_bits_row(line: &str) -> Vec<f64> {
    line.split(';')
        .filter(|t| !t.is_empty())
        .map(|t| {
            let bits: u64 = t
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("bad f64-bits token {t:?}"));
            f64::from_bits(bits)
        })
        .collect()
}

/// Parse a `;`-separated row of `u32` leaf ids.
pub fn parse_u32_row(line: &str) -> Vec<u32> {
    line.split(';')
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().parse().unwrap_or_else(|_| panic!("bad u32 token {t:?}")))
        .collect()
}

/// Load a whitespace-delimited `<label> <f1> ... <fN>` table (LightGBM `.train`).
/// Returns `(data_row_major_f32, num_rows, num_cols)` — the feature matrix only
/// (the label column is dropped), matching the `predict` dense-input shape.
pub fn load_svm_like(path: &PathBuf) -> (Vec<f32>, i32, i32) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut data: Vec<f32> = Vec::new();
    let mut num_rows = 0i32;
    let mut num_cols = -1i32;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let _label = fields.next();
        let row: Vec<f32> = fields
            .map(|t| t.parse::<f32>().unwrap_or_else(|_| panic!("bad feature {t:?}")))
            .collect();
        if num_cols < 0 {
            num_cols = row.len() as i32;
        } else {
            assert_eq!(row.len() as i32, num_cols, "ragged row in {}", path.display());
        }
        data.extend_from_slice(&row);
        num_rows += 1;
    }
    (data, num_rows, num_cols)
}

/// Convert a dense row-major `f32` matrix into CSR `(indptr, indices, values)`.
/// Every entry is stored explicitly (the raw predict path treats absent == 0.0,
/// so a dense CSR with all entries is equivalent and exercises the sparse path).
pub fn dense_to_csr(data: &[f32], num_rows: i32, num_cols: i32) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let nc = num_cols as usize;
    let mut indptr = Vec::with_capacity(num_rows as usize + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0i64);
    for r in 0..num_rows as usize {
        for c in 0..nc {
            indices.push(c as i32);
            values.push(data[r * nc + c]);
        }
        indptr.push((indices.len()) as i64);
    }
    (indptr, indices, values)
}

/// Convert a dense row-major `f32` matrix into CSC `(indptr, indices, values)`.
pub fn dense_to_csc(data: &[f32], num_rows: i32, num_cols: i32) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let nc = num_cols as usize;
    let nr = num_rows as usize;
    let mut indptr = Vec::with_capacity(nc + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0i64);
    for c in 0..nc {
        for r in 0..nr {
            indices.push(r as i32);
            values.push(data[r * nc + c]);
        }
        indptr.push((indices.len()) as i64);
    }
    (indptr, indices, values)
}
