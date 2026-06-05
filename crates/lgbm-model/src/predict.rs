//! Batch raw-score prediction driver (PRD-01, D-02a) — dense / CSR / CSC inputs.
//!
//! Mirrors C++ `Predictor` (`src/application/predictor.hpp`): predictions run on
//! the model's stored REAL thresholds / categorical bitsets over RAW feature
//! values — there is NO re-binning through a `BinMapper` (D-02a). For each row we
//! materialize a dense `f64` row buffer of width `max_feature_idx + 1` (absent /
//! sparse-zero entries == 0.0, mirroring `Predictor::CopyToPredictBuffer`), feed
//! [`crate::ensemble::GbdtModel::predict_raw`], and collect the per-row per-class
//! raw scores.
//!
//! # Accumulation precision
//! The per-class accumulator is `f64` (in `predict_raw`); we cast to `f32` ONLY
//! at the output boundary (RESEARCH anti-pattern line 264) so the ~1e-6 contract
//! does not drift on deep ensembles.
//!
//! # Input shapes (mirror `lgbm-dataset` ingest)
//! - dense: `(&[f32] row-major, num_rows, num_cols)`.
//! - CSR: `(indptr[num_rows+1], indices, values, num_rows, num_cols)`.
//! - CSC: `(indptr[num_cols+1], indices, values, num_rows, num_cols)`.
//!
//! # Validation (Security V5 / T-03-07)
//! Each entry validates caller input at the boundary FIRST: `num_cols >=
//! max_feature_idx + 1`, shape/length consistency, sparse-index range. On failure
//! returns [`ModelError::ShapeMismatch`] / [`ModelError::MalformedModel`], never a
//! panic.
//!
//! Output layout is row-major `[row0_class0, row0_class1, ..., row1_class0, ...]`
//! of length `num_rows * num_tree_per_iteration`.

use crate::ensemble::GbdtModel;
use crate::error::ModelError;

/// Width of the materialized raw row buffer = `max_feature_idx + 1`.
#[inline]
fn row_width(model: &GbdtModel) -> usize {
    (model.max_feature_idx + 1).max(0) as usize
}

/// Validate that the caller's `num_cols` can supply every feature the model
/// references (T-03-07). The model needs `max_feature_idx + 1` columns.
fn check_cols(model: &GbdtModel, num_cols: i32) -> Result<(), ModelError> {
    let need = model.max_feature_idx + 1;
    if num_cols < need {
        return Err(ModelError::ShapeMismatch {
            detail: format!(
                "input has {num_cols} columns, model needs max_feature_idx+1 = {need}"
            ),
        });
    }
    Ok(())
}

/// Run the resolved sub-range raw predict over one materialized row, appending
/// the `num_tree_per_iteration` f32 outputs to `out`.
#[inline]
fn predict_row(model: &GbdtModel, row: &[f64], out: &mut Vec<f32>) {
    let scores = model.predict_raw(row, 0, -1);
    for s in scores {
        out.push(s as f32);
    }
}

/// Dense raw-score prediction. `data` is row-major `num_rows * num_cols` `f32`.
pub fn predict_raw_mat(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    if num_rows < 0 || num_cols < 0 {
        return Err(ModelError::ShapeMismatch {
            detail: format!("num_rows={num_rows} / num_cols={num_cols} must be >= 0"),
        });
    }
    let expected = (num_rows as i64) * (num_cols as i64);
    if data.len() as i64 != expected {
        return Err(ModelError::ShapeMismatch {
            detail: format!(
                "data.len()={} must equal num_rows*num_cols={expected}",
                data.len()
            ),
        });
    }
    check_cols(model, num_cols)?;

    let width = row_width(model);
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let mut out = Vec::with_capacity(num_rows as usize * ntpi);
    let mut row = vec![0.0f64; width];
    let ncols = num_cols as usize;
    for r in 0..num_rows as usize {
        // Materialize the raw f64 row buffer (only the first `width` columns are
        // referenced by the model; extra trailing columns are ignored).
        let base = r * ncols;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = data[base + c] as f64;
        }
        predict_row(model, &row, &mut out);
    }
    Ok(out)
}

/// CSR raw-score prediction. `indptr` has `num_rows + 1` entries; row `r`'s
/// `(col, value)` pairs are `indices[indptr[r]..indptr[r+1]]` / `values[...]`.
pub fn predict_raw_csr(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    if num_rows < 0 || num_cols < 0 {
        return Err(ModelError::ShapeMismatch {
            detail: format!("num_rows={num_rows} / num_cols={num_cols} must be >= 0"),
        });
    }
    if indices.len() != values.len() {
        return Err(ModelError::MalformedModel {
            detail: format!(
                "indices.len()={} must equal values.len()={}",
                indices.len(),
                values.len()
            ),
        });
    }
    if indptr.len() != num_rows as usize + 1 {
        return Err(ModelError::ShapeMismatch {
            detail: format!(
                "csr indptr.len()={} must equal num_rows+1={}",
                indptr.len(),
                num_rows as usize + 1
            ),
        });
    }
    check_cols(model, num_cols)?;

    let width = row_width(model);
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let mut out = Vec::with_capacity(num_rows as usize * ntpi);
    let mut row = vec![0.0f64; width];
    for r in 0..num_rows as usize {
        for slot in row.iter_mut() {
            *slot = 0.0;
        }
        let start = validate_indptr_pair(indptr, r, values.len())?;
        let end = indptr[r + 1] as usize;
        for k in start..end {
            let col = indices[k];
            if col < 0 || col >= num_cols {
                return Err(ModelError::ShapeMismatch {
                    detail: format!("csr column index {col} out of range [0, {num_cols})"),
                });
            }
            let c = col as usize;
            if c < width {
                row[c] = values[k] as f64;
            }
        }
        predict_row(model, &row, &mut out);
    }
    Ok(out)
}

/// CSC raw-score prediction. `indptr` has `num_cols + 1` entries; column `c`'s
/// `(row, value)` pairs are `indices[indptr[c]..indptr[c+1]]` / `values[...]`.
pub fn predict_raw_csc(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    if num_rows < 0 || num_cols < 0 {
        return Err(ModelError::ShapeMismatch {
            detail: format!("num_rows={num_rows} / num_cols={num_cols} must be >= 0"),
        });
    }
    if indices.len() != values.len() {
        return Err(ModelError::MalformedModel {
            detail: format!(
                "indices.len()={} must equal values.len()={}",
                indices.len(),
                values.len()
            ),
        });
    }
    if indptr.len() != num_cols as usize + 1 {
        return Err(ModelError::ShapeMismatch {
            detail: format!(
                "csc indptr.len()={} must equal num_cols+1={}",
                indptr.len(),
                num_cols as usize + 1
            ),
        });
    }
    check_cols(model, num_cols)?;

    let width = row_width(model);
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let nrows = num_rows as usize;

    // Materialize the full dense f64 matrix once (column-driven scatter), then
    // predict row by row. CSC is column-major so a per-row stream isn't natural.
    let mut dense = vec![0.0f64; nrows * width];
    for c in 0..num_cols as usize {
        let start = validate_indptr_pair(indptr, c, values.len())?;
        let end = indptr[c + 1] as usize;
        for k in start..end {
            let r = indices[k];
            if r < 0 || r >= num_rows {
                return Err(ModelError::ShapeMismatch {
                    detail: format!("csc row index {r} out of range [0, {num_rows})"),
                });
            }
            if c < width {
                dense[r as usize * width + c] = values[k] as f64;
            }
        }
    }

    let mut out = Vec::with_capacity(nrows * ntpi);
    for r in 0..nrows {
        let row = &dense[r * width..(r + 1) * width];
        predict_row(model, row, &mut out);
    }
    Ok(out)
}

/// Validate `indptr[i] <= indptr[i+1]`, both in `[0, nnz]`, returning `indptr[i]`
/// as a `usize`. Rejects negative / decreasing / out-of-range pointers.
fn validate_indptr_pair(indptr: &[i64], i: usize, nnz: usize) -> Result<usize, ModelError> {
    let lo = indptr[i];
    let hi = indptr[i + 1];
    if lo < 0 || hi < lo || hi as usize > nnz {
        return Err(ModelError::MalformedModel {
            detail: format!(
                "malformed indptr at {i}: [{lo}, {hi}] not within [0, {nnz}]"
            ),
        });
    }
    Ok(lo as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    fn stump(v0: f64, v1: f64, feat: i32, thr: f64) -> Tree {
        Tree {
            num_leaves: 2,
            num_cat: 0,
            left_child: vec![-1],
            right_child: vec![-2],
            split_feature: vec![feat],
            threshold: vec![thr],
            decision_type: vec![2],
            split_gain: vec![0.0],
            leaf_value: vec![v0, v1],
            leaf_weight: vec![1.0, 1.0],
            leaf_count: vec![1, 1],
            internal_value: vec![0.0],
            internal_weight: vec![0.0],
            internal_count: vec![2],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
        }
    }

    fn model() -> GbdtModel {
        GbdtModel {
            trees: vec![stump(1.0, 2.0, 0, 0.5), stump(0.1, 0.2, 1, 0.5)],
            num_class: 1,
            num_tree_per_iteration: 1,
            label_index: 0,
            max_feature_idx: 1,
            average_output: false,
            objective_string: Some("regression".to_string()),
            feature_names: "Column_0 Column_1".to_string(),
            feature_infos: "[0:1] [0:1]".to_string(),
            monotone_constraints: None,
            trailer: None,
        }
    }

    #[test]
    fn dense_matches_hand_computed() {
        let m = model();
        // row0: f0=1.0(>0.5 ->2.0) f1=0.0(<=0.5 ->0.1) => 2.1
        // row1: f0=0.0(<=0.5->1.0) f1=1.0(>0.5 ->0.2) => 1.2
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let out = predict_raw_mat(&m, &data, 2, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 2.1).abs() < 1e-6);
        assert!((out[1] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn csr_matches_dense() {
        let m = model();
        // Same two rows in CSR. row0: f0=1.0 (f1=0 omitted). row1: f1=1.0 (f0=0 omitted).
        let indptr = vec![0i64, 1, 2];
        let indices = vec![0i32, 1];
        let values = vec![1.0f32, 1.0];
        let out = predict_raw_csr(&m, &indptr, &indices, &values, 2, 2).unwrap();
        assert!((out[0] - 2.1).abs() < 1e-6);
        assert!((out[1] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn csc_matches_dense() {
        let m = model();
        // column-major: col0 has row0=1.0; col1 has row1=1.0.
        let indptr = vec![0i64, 1, 2];
        let indices = vec![0i32, 1]; // col0 -> row0; col1 -> row1
        let values = vec![1.0f32, 1.0];
        let out = predict_raw_csc(&m, &indptr, &indices, &values, 2, 2).unwrap();
        assert!((out[0] - 2.1).abs() < 1e-6);
        assert!((out[1] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn too_few_columns_is_shape_mismatch() {
        let m = model(); // needs max_feature_idx+1 = 2 columns
        let data = vec![1.0f32, 0.0];
        let err = predict_raw_mat(&m, &data, 2, 1).unwrap_err();
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
    }

    #[test]
    fn dense_wrong_len_is_shape_mismatch() {
        let m = model();
        let data = vec![1.0f32, 0.0, 0.0]; // not 2*2
        let err = predict_raw_mat(&m, &data, 2, 2).unwrap_err();
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
    }
}
