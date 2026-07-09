//! Cross-representation ingestion equivalence (SC#2 / DAT-06): the SAME logical
//! matrix, ingested via `from_mat` (dense), `from_csr`, and `from_csc`, must
//! produce BIT-IDENTICAL Datasets — same per-feature `bin_upper_bound_` and same
//! per-row stored bin indices. This needs no C++ golden: it is a pure internal
//! invariant (the three representations describe identical data, so binning must
//! agree to the last bit). Comparison is exact `assert_eq` / f64-bit equality,
//! never a tolerance.
//!
//! The matrix deliberately includes a ZERO-HEAVY column so the sparse gather's
//! "absent entry == 0.0" path (Open Q2) is exercised and proven equivalent to the
//! dense form.
//!
//! Also asserts malformed CSR/CSC/dense input returns a typed `DatasetError`,
//! never a panic (Security V5).

use lgbm_core::config::Config;
use lgbm_dataset::dataset::FinishedDataset;
use lgbm_dataset::{from_csc, from_csr, from_mat, DatasetError, Metadata};

fn cfg() -> Config {
    let mut c = Config::default();
    c.max_bin = 16;
    c.min_data_in_bin = 1;
    c.bin_construct_sample_cnt = 100000; // sample all rows -> deterministic
    c.feature_pre_filter = false;
    c
}

fn meta(n: i32) -> Metadata {
    Metadata::new(vec![0.0f32; n as usize], Vec::new(), Vec::new(), Vec::new()).unwrap()
}

/// Extract, per feature, the `(bin_upper_bound_, per-row stored bin index)` pair
/// from a finished dataset (one-feature-per-group default).
fn fingerprint(ds: &FinishedDataset) -> Vec<(Vec<f64>, Vec<u32>)> {
    let n = ds.num_data();
    let mut out = Vec::new();
    for g in 0..ds.num_groups() as usize {
        let fg = ds.feature_group(g);
        let bub = fg.bin_mapper(0).bin_upper_bound_.clone();
        let bd = fg.bin_data().expect("dense single-value bin store");
        let row_idx: Vec<u32> = (0..n).map(|r| bd.data(r)).collect();
        out.push((bub, row_idx));
    }
    out
}

fn assert_fingerprints_eq(label: &str, a: &[(Vec<f64>, Vec<u32>)], b: &[(Vec<f64>, Vec<u32>)]) {
    assert_eq!(a.len(), b.len(), "{label}: feature count mismatch");
    for (f, (fa, fb)) in a.iter().zip(b.iter()).enumerate() {
        // bin_upper_bound_ bit-exact (f64::to_bits per element).
        assert_eq!(fa.0.len(), fb.0.len(), "{label}: feature {f} bound count");
        for (i, (&x, &y)) in fa.0.iter().zip(fb.0.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: feature {f} bin_upper_bound_[{i}] mismatch: {x} vs {y}"
            );
        }
        // per-row stored bin index exact.
        assert_eq!(
            fa.1, fb.1,
            "{label}: feature {f} per-row bin index vector mismatch"
        );
    }
}

/// Build dense/CSR/CSC of the SAME logical matrix and assert bin-equivalence.
#[test]
fn ingest_equivalence_dense_csr_csc_bit_identical() {
    // 6 rows x 3 cols. Column 1 is ZERO-HEAVY (only two nonzero entries).
    // Row-major dense layout:
    let num_rows = 6i32;
    let num_cols = 3i32;
    #[rustfmt::skip]
    let dense: Vec<f32> = vec![
        // c0     c1     c2
        1.0,    0.0,   3.5,
        2.0,    0.0,   1.5,
        3.0,    7.0,   2.5,   // c1 nonzero here
        4.0,    0.0,   9.5,
        5.0,    0.0,   0.5,
        6.0,    4.0,   8.5,   // c1 nonzero here
    ];

    // CSR: per-row nonzero (col, value) pairs. We store EXACTLY the nonzeros
    // (zeros are absent -> the gather fills 0.0), matching the dense data.
    let mut csr_indptr: Vec<i64> = vec![0];
    let mut csr_indices: Vec<i32> = Vec::new();
    let mut csr_values: Vec<f32> = Vec::new();
    for row in 0..num_rows as usize {
        for col in 0..num_cols as usize {
            let v = dense[row * num_cols as usize + col];
            if v != 0.0 {
                csr_indices.push(col as i32);
                csr_values.push(v);
            }
        }
        csr_indptr.push(csr_values.len() as i64);
    }

    // CSC: per-column nonzero (row, value) pairs.
    let mut csc_indptr: Vec<i64> = vec![0];
    let mut csc_indices: Vec<i32> = Vec::new();
    let mut csc_values: Vec<f32> = Vec::new();
    for col in 0..num_cols as usize {
        for row in 0..num_rows as usize {
            let v = dense[row * num_cols as usize + col];
            if v != 0.0 {
                csc_indices.push(row as i32);
                csc_values.push(v);
            }
        }
        csc_indptr.push(csc_values.len() as i64);
    }

    let c = cfg();
    let (ds_mat, _) = from_mat(&dense, num_rows, num_cols, &c, meta(num_rows)).unwrap();
    let (ds_csr, _) = from_csr(
        &csr_indptr,
        &csr_indices,
        &csr_values,
        num_rows,
        num_cols,
        &c,
        meta(num_rows),
    )
    .unwrap();
    let (ds_csc, _) = from_csc(
        &csc_indptr,
        &csc_indices,
        &csc_values,
        num_rows,
        num_cols,
        &c,
        meta(num_rows),
    )
    .unwrap();

    let fp_mat = fingerprint(&ds_mat);
    let fp_csr = fingerprint(&ds_csr);
    let fp_csc = fingerprint(&ds_csc);

    assert_fingerprints_eq("dense vs CSR", &fp_mat, &fp_csr);
    assert_fingerprints_eq("dense vs CSC", &fp_mat, &fp_csc);

    // Sanity: the zero-heavy column (feature 1) actually has nonzero data routed.
    let zero_heavy = &fp_mat[1].1;
    assert!(
        zero_heavy.iter().any(|&b| b != 0),
        "zero-heavy column must have at least one non-default stored bin"
    );
}

#[test]
fn from_mat_shape_mismatch_never_panics() {
    let c = cfg();
    let err = from_mat(&[1.0, 2.0, 3.0], 2, 2, &c, meta(2)).unwrap_err();
    assert!(matches!(err, DatasetError::ShapeMismatch { .. }));
}

#[test]
fn from_csr_malformed_indptr_never_panics() {
    let c = cfg();
    // indptr last (2) != nnz (1).
    let err = from_csr(&[0, 2], &[0], &[1.0], 1, 2, &c, meta(1)).unwrap_err();
    assert!(matches!(err, DatasetError::MalformedSparse { .. }));
}

#[test]
fn from_csc_out_of_range_row_never_panics() {
    let c = cfg();
    let err = from_csc(&[0, 1, 1], &[99], &[1.0], 2, 2, &c, meta(2)).unwrap_err();
    assert!(matches!(err, DatasetError::MalformedSparse { .. }));
}
