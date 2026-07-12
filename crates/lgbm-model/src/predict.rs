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
use crate::objective::ObjectiveKind;

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
/// the `num_tree_per_iteration` f32 outputs to `out`. `start_iteration` /
/// `num_iteration` are threaded straight into `GbdtModel::predict_raw`, which
/// clamps them via `init_predict` (`gbdt.h:426-435`) — `0`/`-1` = full range,
/// `num_iteration == -1` (== all from start), extreme values never panic.
#[inline]
fn predict_row(
    model: &GbdtModel,
    row: &[f64],
    start_iteration: i32,
    num_iteration: i32,
    out: &mut Vec<f32>,
) {
    let scores = model.predict_raw(row, start_iteration, num_iteration);
    for s in scores {
        out.push(s as f32);
    }
}

/// Dense raw-score prediction over the full ensemble (`start_iteration=0`,
/// `num_iteration=-1`). Thin wrapper over [`predict_raw_mat_range`].
pub fn predict_raw_mat(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    predict_raw_mat_range(model, data, num_rows, num_cols, 0, -1)
}

/// Dense raw-score prediction over the sub-range `(start_iteration,
/// num_iteration)` (PRD-06). `data` is row-major `num_rows * num_cols` `f32`.
/// `num_iteration == -1` (or `<= 0`) means "all iterations from `start_iteration`";
/// `start_iteration` is clamped into `[0, num_iteration()]` — extreme values
/// produce an empty slice (zero scores), never a panic or OOB (T-03-12).
pub fn predict_raw_mat_range(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
    start_iteration: i32,
    num_iteration: i32,
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
        predict_row(model, &row, start_iteration, num_iteration, &mut out);
    }
    Ok(out)
}

/// CSR raw-score prediction over the full ensemble. Thin wrapper over
/// [`predict_raw_csr_range`].
pub fn predict_raw_csr(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    predict_raw_csr_range(model, indptr, indices, values, num_rows, num_cols, 0, -1)
}

/// CSR raw-score prediction over the sub-range `(start_iteration, num_iteration)`
/// (PRD-06). `indptr` has `num_rows + 1` entries; row `r`'s `(col, value)` pairs
/// are `indices[indptr[r]..indptr[r+1]]` / `values[...]`.
pub fn predict_raw_csr_range(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
    start_iteration: i32,
    num_iteration: i32,
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
        predict_row(model, &row, start_iteration, num_iteration, &mut out);
    }
    Ok(out)
}

/// CSC raw-score prediction over the full ensemble. Thin wrapper over
/// [`predict_raw_csc_range`].
pub fn predict_raw_csc(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    predict_raw_csc_range(model, indptr, indices, values, num_rows, num_cols, 0, -1)
}

/// CSC raw-score prediction over the sub-range `(start_iteration, num_iteration)`
/// (PRD-06). `indptr` has `num_cols + 1` entries; column `c`'s `(row, value)`
/// pairs are `indices[indptr[c]..indptr[c+1]]` / `values[...]`.
pub fn predict_raw_csc_range(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
    start_iteration: i32,
    num_iteration: i32,
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
        predict_row(model, row, start_iteration, num_iteration, &mut out);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Transformed prediction (PRD-02) + leaf-index prediction (PRD-03).
//
// These reuse the SAME raw dense-row materialization as the raw driver above:
// each public entry materializes the row buffer (mat: per-row slice; csr: per-row
// scatter; csc: full dense scatter), then either
//   - applies `GbdtModel::predict_raw` (full range) + ObjectiveKind::convert
//     (transformed, `gbdt_prediction.cpp:55` Predict), or
//   - walks `trees[i*ntpi+k].get_leaf(row)` into the per-(iter×class) stride
//     (leaf, `gbdt_prediction.cpp:79-86` PredictLeafIndex + Pitfall 8 layout).
// ---------------------------------------------------------------------------

/// Resolve + validate the model's objective once per transformed-predict call.
/// The transformed output width must match `num_tree_per_iteration` (the raw
/// per-class width) — a mismatch is a malformed model.
fn resolve_objective(model: &GbdtModel) -> Result<ObjectiveKind, ModelError> {
    let line = model
        .objective_string
        .as_deref()
        .ok_or_else(|| ModelError::MalformedModel {
            detail: "model has no objective= line; cannot apply ConvertOutput".to_string(),
        })?;
    let kind = ObjectiveKind::parse(line)?;
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    if kind.num_output() != ntpi {
        return Err(ModelError::MalformedModel {
            detail: format!(
                "objective output width {} != num_tree_per_iteration {ntpi}",
                kind.num_output()
            ),
        });
    }
    Ok(kind)
}

/// Apply the transformed pipeline to one materialized row: raw per-class scores
/// (f64) → `ObjectiveKind::convert` → push f32 outputs.
#[inline]
fn predict_row_transformed(
    model: &GbdtModel,
    kind: &ObjectiveKind,
    row: &[f64],
    raw_buf: &mut Vec<f64>,
    conv_buf: &mut [f64],
    out: &mut Vec<f32>,
) {
    raw_buf.clear();
    raw_buf.extend(model.predict_raw(row, 0, -1));
    // RF (average_output) divides the per-tree SUM by num_iteration BEFORE the
    // ConvertOutput, matching C++ `GBDT::Predict` (gbdt_prediction.cpp:57-61:
    // the `average_output_` block runs after `PredictRaw` and before
    // `ConvertOutput`). Mirrors `Booster::predict_row`. The raw path
    // (`predict_raw`) must NOT divide — it matches C++ `PredictRaw`.
    if model.average_output {
        let (_start, num) = model.init_predict(0, -1);
        if num > 0 {
            for v in raw_buf.iter_mut() {
                *v /= num as f64;
            }
        }
    }
    kind.convert(raw_buf, conv_buf);
    for &v in conv_buf.iter() {
        out.push(v as f32);
    }
}

/// Walk one materialized row's leaf ids into the per-(iter×class) stride
/// (`gbdt_prediction.cpp:79-86`, Pitfall 8): output length
/// `num_tree_per_iteration * num_iteration_for_pred`, ordered
/// `[iter0_class0, iter0_class1, ..., iter1_class0, ...]`. Leaf ids are
/// non-negative (`!node`), emitted as `u32`.
#[inline]
fn predict_row_leaf(model: &GbdtModel, row: &[f64], out: &mut Vec<u32>) {
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let num_iter = model.num_iteration().max(0) as usize;
    for i in 0..num_iter {
        for k in 0..ntpi {
            let idx = i * ntpi + k;
            let leaf = model.trees[idx].predict_leaf_index(row);
            out.push(leaf as u32);
        }
    }
}

/// Transformed dense prediction (PRD-02). Output is row-major
/// `[row0_out0, .., row0_out{m-1}, row1_out0, ..]` where `m =
/// num_tree_per_iteration` (1 for regression/binary, `num_class` for multiclass).
pub fn predict_mat(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    validate_dense_shape(model, data, num_rows, num_cols)?;
    let kind = resolve_objective(model)?;

    let width = row_width(model);
    let m = kind.num_output();
    let ncols = num_cols as usize;
    let mut out = Vec::with_capacity(num_rows as usize * m);
    let mut row = vec![0.0f64; width];
    let mut raw_buf: Vec<f64> = Vec::with_capacity(m);
    let mut conv_buf = vec![0.0f64; m];
    for r in 0..num_rows as usize {
        let base = r * ncols;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = data[base + c] as f64;
        }
        predict_row_transformed(model, &kind, &row, &mut raw_buf, &mut conv_buf, &mut out);
    }
    Ok(out)
}

/// Transformed CSR prediction (PRD-02).
pub fn predict_csr(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    validate_csr_shape(model, indptr, indices, values, num_rows, num_cols)?;
    let kind = resolve_objective(model)?;

    let width = row_width(model);
    let m = kind.num_output();
    let mut out = Vec::with_capacity(num_rows as usize * m);
    let mut row = vec![0.0f64; width];
    let mut raw_buf: Vec<f64> = Vec::with_capacity(m);
    let mut conv_buf = vec![0.0f64; m];
    for r in 0..num_rows as usize {
        for slot in row.iter_mut() {
            *slot = 0.0;
        }
        scatter_csr_row(indptr, indices, values, r, num_cols, width, &mut row)?;
        predict_row_transformed(model, &kind, &row, &mut raw_buf, &mut conv_buf, &mut out);
    }
    Ok(out)
}

/// Transformed CSC prediction (PRD-02).
pub fn predict_csc(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f32>, ModelError> {
    validate_csc_shape(model, indptr, indices, values, num_rows, num_cols)?;
    let kind = resolve_objective(model)?;

    let width = row_width(model);
    let nrows = num_rows as usize;
    let m = kind.num_output();
    let dense = scatter_csc_dense(indptr, indices, values, num_rows, num_cols, width)?;

    let mut out = Vec::with_capacity(nrows * m);
    let mut raw_buf: Vec<f64> = Vec::with_capacity(m);
    let mut conv_buf = vec![0.0f64; m];
    for r in 0..nrows {
        let row = &dense[r * width..(r + 1) * width];
        predict_row_transformed(model, &kind, row, &mut raw_buf, &mut conv_buf, &mut out);
    }
    Ok(out)
}

/// Leaf-index dense prediction (PRD-03). Output is row-major; each row contributes
/// `num_tree_per_iteration * num_iteration` `u32` leaf ids in the per-(iter×class)
/// layout (Pitfall 8).
pub fn predict_leaf_index_mat(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<u32>, ModelError> {
    validate_dense_shape(model, data, num_rows, num_cols)?;
    let width = row_width(model);
    let per_row = leaf_width(model);
    let ncols = num_cols as usize;
    let mut out = Vec::with_capacity(num_rows as usize * per_row);
    let mut row = vec![0.0f64; width];
    for r in 0..num_rows as usize {
        let base = r * ncols;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = data[base + c] as f64;
        }
        predict_row_leaf(model, &row, &mut out);
    }
    Ok(out)
}

/// Leaf-index CSR prediction (PRD-03).
pub fn predict_leaf_index_csr(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<u32>, ModelError> {
    validate_csr_shape(model, indptr, indices, values, num_rows, num_cols)?;
    let width = row_width(model);
    let per_row = leaf_width(model);
    let mut out = Vec::with_capacity(num_rows as usize * per_row);
    let mut row = vec![0.0f64; width];
    for r in 0..num_rows as usize {
        for slot in row.iter_mut() {
            *slot = 0.0;
        }
        scatter_csr_row(indptr, indices, values, r, num_cols, width, &mut row)?;
        predict_row_leaf(model, &row, &mut out);
    }
    Ok(out)
}

/// Leaf-index CSC prediction (PRD-03).
pub fn predict_leaf_index_csc(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<u32>, ModelError> {
    validate_csc_shape(model, indptr, indices, values, num_rows, num_cols)?;
    let width = row_width(model);
    let nrows = num_rows as usize;
    let per_row = leaf_width(model);
    let dense = scatter_csc_dense(indptr, indices, values, num_rows, num_cols, width)?;
    let mut out = Vec::with_capacity(nrows * per_row);
    for r in 0..nrows {
        let row = &dense[r * width..(r + 1) * width];
        predict_row_leaf(model, row, &mut out);
    }
    Ok(out)
}

/// Per-row leaf-index output width = `num_tree_per_iteration * num_iteration`
/// (`NumPredictOneRow` for leaf, `gbdt.h:281-291`).
#[inline]
fn leaf_width(model: &GbdtModel) -> usize {
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let num_iter = model.num_iteration().max(0) as usize;
    ntpi * num_iter
}

// ---------------------------------------------------------------------------
// Prediction early stopping (PRD-05) — model-aware driver.
//
// Mirrors C++ `Predictor` (`predictor.hpp:41-59`): the early-stop margin hook is
// installed ONLY when `early_stop && !boosting->NeedAccuratePrediction()`. For an
// accurate-prediction objective (regression / poisson / cross-entropy / …) the
// request is silently ignored and the FULL ensemble is evaluated — so the result
// is byte-identical to `predict_raw`. The hook is active only for binary /
// multiclass margins (`GbdtModel::predict_raw_early_stop`).
// ---------------------------------------------------------------------------

/// Dense early-stop raw prediction (PRD-05). Returns row-major raw scores of width
/// `num_tree_per_iteration` plus a per-row `iterations_evaluated` count.
///
/// `freq` / `margin` come from `Config.pred_early_stop_freq` /
/// `pred_early_stop_margin`. When the model's objective `need_accurate_prediction()`
/// (regression-like), the hook is disabled and every row evaluates the full
/// ensemble (`iterations_evaluated == num_iteration()`), matching C++.
pub fn predict_raw_early_stop_mat(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
    freq: i32,
    margin: f64,
) -> Result<(Vec<f32>, Vec<i32>), ModelError> {
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

    // GATE: early stop is only honored for objectives that do NOT need accurate
    // prediction (binary / multiclass). Otherwise force freq=0 (disabled).
    let kind = resolve_objective(model)?;
    let effective_freq = if kind.need_accurate_prediction() { 0 } else { freq };

    let width = row_width(model);
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let ncols = num_cols as usize;
    let mut out = Vec::with_capacity(num_rows as usize * ntpi);
    let mut iters = Vec::with_capacity(num_rows as usize);
    let mut row = vec![0.0f64; width];
    for r in 0..num_rows as usize {
        let base = r * ncols;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = data[base + c] as f64;
        }
        let (scores, n) =
            model.predict_raw_early_stop(&row, 0, -1, effective_freq, margin);
        for s in scores {
            out.push(s as f32);
        }
        iters.push(n);
    }
    Ok((out, iters))
}

// ---------------------------------------------------------------------------
// TreeSHAP feature-contribution prediction (PRD-04, predict_contrib).
//
// Mirrors C++ `GBDT::PredictContrib` (`gbdt.cpp:640-651`): for each class `k`,
// the per-row output sub-block `[k*(num_features+1) .. (k+1)*(num_features+1))`
// holds per-feature contributions `[0..num_features]` plus the expected-value
// base `[num_features]`, accumulated across the resolved iteration sub-range.
//
// INVARIANT (PRD-04): for class `k`, the sum of that sub-block equals the raw
// margin `predict_raw(row)[k]` (each tree contributes its own per-feature SHAP
// values + its ExpectedValue base; the sum telescopes to the tree's predict).
// ---------------------------------------------------------------------------

/// C++ never wires up `PredictContrib` for `linear_tree=true` models
/// (`predictor.hpp:89-90` builds the contrib lambda only for non-linear
/// boosters) — `Tree::tree_shap`/`expected_value` read the constant
/// `leaf_value` directly and have no linear-leaf-aware counterpart in C++ to
/// port. Rather than inventing a SHAP variant with no C++ reference to stay in
/// parity with, refuse the combination outright, matching C++.
fn check_no_linear_trees(model: &GbdtModel) -> Result<(), ModelError> {
    if model.trees.iter().any(|t| t.is_linear) {
        return Err(ModelError::Unsupported {
            detail: "predict_contrib (SHAP feature contributions) is not supported for \
                     linear_tree=true models — matches C++ LightGBM, which never wires up \
                     PredictContrib for linear trees"
                .to_string(),
        });
    }
    Ok(())
}

/// Per-row contribution output width = `num_tree_per_iteration * (num_features+1)`
/// (`NumPredictOneRow` for contrib, `gbdt.h`).
#[inline]
fn contrib_width(model: &GbdtModel) -> usize {
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let nf = row_width(model);
    ntpi * (nf + 1)
}

/// Run TreeSHAP for one materialized row over the full ensemble, appending the
/// `num_tree_per_iteration * (num_features + 1)` f64 contributions to `out`.
/// `num_features = max_feature_idx + 1` (the SHAP feature axis is the model's
/// feature width, NOT the caller's `num_cols`).
#[inline]
fn predict_row_contrib(model: &GbdtModel, row: &[f64], out: &mut Vec<f64>) {
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let nf = row_width(model);
    let block = nf + 1;
    let base = out.len();
    // Zero the per-row slab, then accumulate each tree into its class sub-block.
    out.resize(base + ntpi * block, 0.0);
    let num_iter = model.num_iteration().max(0) as usize;
    for i in 0..num_iter {
        for k in 0..ntpi {
            let idx = i * ntpi + k;
            let off = base + k * block;
            model.trees[idx].predict_contrib(row, nf, &mut out[off..off + block]);
        }
    }
}

/// TreeSHAP dense feature-contribution prediction (PRD-04). Output is row-major;
/// each row contributes `num_tree_per_iteration * (max_feature_idx + 2)` f64
/// values: per class `k`, `[per-feature contributions; expected-value base]`.
///
/// INVARIANT: for each row and class, the sum of that class's sub-block equals
/// the raw margin (`predict_raw_mat`). Callers (and the parity gate) assert it.
pub fn predict_contrib_mat(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f64>, ModelError> {
    check_no_linear_trees(model)?;
    validate_dense_shape(model, data, num_rows, num_cols)?;
    let width = row_width(model);
    let per_row = contrib_width(model);
    let ncols = num_cols as usize;
    let mut out = Vec::with_capacity(num_rows as usize * per_row);
    let mut row = vec![0.0f64; width];
    for r in 0..num_rows as usize {
        let base = r * ncols;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = data[base + c] as f64;
        }
        predict_row_contrib(model, &row, &mut out);
    }
    Ok(out)
}

/// TreeSHAP CSR feature-contribution prediction (PRD-04).
pub fn predict_contrib_csr(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f64>, ModelError> {
    check_no_linear_trees(model)?;
    validate_csr_shape(model, indptr, indices, values, num_rows, num_cols)?;
    let width = row_width(model);
    let per_row = contrib_width(model);
    let mut out = Vec::with_capacity(num_rows as usize * per_row);
    let mut row = vec![0.0f64; width];
    for r in 0..num_rows as usize {
        for slot in row.iter_mut() {
            *slot = 0.0;
        }
        scatter_csr_row(indptr, indices, values, r, num_cols, width, &mut row)?;
        predict_row_contrib(model, &row, &mut out);
    }
    Ok(out)
}

/// TreeSHAP CSC feature-contribution prediction (PRD-04).
pub fn predict_contrib_csc(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<Vec<f64>, ModelError> {
    check_no_linear_trees(model)?;
    validate_csc_shape(model, indptr, indices, values, num_rows, num_cols)?;
    let width = row_width(model);
    let nrows = num_rows as usize;
    let per_row = contrib_width(model);
    let dense = scatter_csc_dense(indptr, indices, values, num_rows, num_cols, width)?;
    let mut out = Vec::with_capacity(nrows * per_row);
    for r in 0..nrows {
        let row = &dense[r * width..(r + 1) * width];
        predict_row_contrib(model, row, &mut out);
    }
    Ok(out)
}

// --- shared shape validators + materializers (extracted so raw/transformed/leaf
//     entry points apply the SAME boundary checks) ---

fn validate_dense_shape(
    model: &GbdtModel,
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<(), ModelError> {
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
    check_cols(model, num_cols)
}

fn validate_csr_shape(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<(), ModelError> {
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
    check_cols(model, num_cols)
}

fn validate_csc_shape(
    model: &GbdtModel,
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
) -> Result<(), ModelError> {
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
    check_cols(model, num_cols)
}

/// Scatter CSR row `r`'s `(col, value)` pairs into the (pre-zeroed) `row` buffer.
fn scatter_csr_row(
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    r: usize,
    num_cols: i32,
    width: usize,
    row: &mut [f64],
) -> Result<(), ModelError> {
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
    Ok(())
}

/// Materialize the full dense `f64` matrix (`num_rows * width`) from CSC input.
fn scatter_csc_dense(
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
    width: usize,
) -> Result<Vec<f64>, ModelError> {
    let nrows = num_rows as usize;
    let mut dense = vec![0.0f64; nrows * width];
    for c in 0..num_cols as usize {
        let start = validate_indptr_pair(indptr, c, values.len())?;
        let end = indptr[c + 1] as usize;
        for k in start..end {
            let row = indices[k];
            if row < 0 || row >= num_rows {
                return Err(ModelError::ShapeMismatch {
                    detail: format!("csc row index {row} out of range [0, {num_rows})"),
                });
            }
            if c < width {
                dense[row as usize * width + c] = values[k] as f64;
            }
        }
    }
    Ok(dense)
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
            linear: None,
            leaf_depth: vec![1, 1],
            leaf_parent: vec![0, 0],
            split_feature_inner: vec![-1],
            threshold_in_bin: vec![0],
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

    #[test]
    fn transformed_regression_is_identity_of_raw() {
        let m = model(); // objective=regression -> identity
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let raw = predict_raw_mat(&m, &data, 2, 2).unwrap();
        let tf = predict_mat(&m, &data, 2, 2).unwrap();
        assert_eq!(raw, tf, "regression transform must be identity");
    }

    #[test]
    fn transformed_rf_average_output_divides_by_num_iteration() {
        // CR-01: a Random Forest model (`average_output = true`, 2 trees =
        // 2 iterations at ntpi=1) must return the per-tree AVERAGE through the
        // transformed batch-predict API, mirroring C++ `GBDT::Predict`
        // (gbdt_prediction.cpp:57-61) and the in-crate `Booster::predict_row`,
        // NOT the raw SUM.
        let mut m = model();
        m.average_output = true;
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        // Raw SUM is unaffected by average_output (matches C++ PredictRaw).
        let raw = predict_raw_mat(&m, &data, 2, 2).unwrap();
        assert!((raw[0] - 2.1).abs() < 1e-6);
        assert!((raw[1] - 1.2).abs() < 1e-6);
        // Transformed output must be the SUM / num_iteration (=2), and the
        // regression convert is the identity.
        let tf = predict_mat(&m, &data, 2, 2).unwrap();
        assert!(
            (tf[0] - 2.1 / 2.0).abs() < 1e-6,
            "RF transformed output must be per-tree average, got {}",
            tf[0]
        );
        assert!((tf[1] - 1.2 / 2.0).abs() < 1e-6);
        // CSR/CSC must agree with the dense transformed path.
        let indptr = vec![0i64, 1, 2];
        let indices = vec![0i32, 1];
        let values = vec![1.0f32, 1.0];
        let csr = predict_csr(&m, &indptr, &indices, &values, 2, 2).unwrap();
        assert_eq!(csr, tf, "CSR RF transform must match dense");
        let csc = predict_csc(&m, &indptr, &indices, &values, 2, 2).unwrap();
        assert_eq!(csc, tf, "CSC RF transform must match dense");
    }

    /// A multiclass model: num_tree_per_iteration=2, one iteration. Tree for
    /// class 0 outputs raw 0.0/feature-routed; class 1 outputs the other.
    fn multiclass_model() -> GbdtModel {
        // iter0 class0: stump on feature 0 (->1.0 or 2.0)
        // iter0 class1: stump on feature 1 (->0.1 or 0.2)
        GbdtModel {
            trees: vec![stump(1.0, 2.0, 0, 0.5), stump(0.1, 0.2, 1, 0.5)],
            num_class: 2,
            num_tree_per_iteration: 2,
            label_index: 0,
            max_feature_idx: 1,
            average_output: false,
            objective_string: Some("multiclass num_class:2".to_string()),
            feature_names: "Column_0 Column_1".to_string(),
            feature_infos: "[0:1] [0:1]".to_string(),
            monotone_constraints: None,
            trailer: None,
        }
    }

    #[test]
    fn transformed_multiclass_softmax_sums_to_one() {
        let m = multiclass_model();
        let data = vec![1.0f32, 0.0]; // 1 row: f0=1.0->class0=2.0, f1=0.0->class1=0.1
        let tf = predict_mat(&m, &data, 1, 2).unwrap();
        assert_eq!(tf.len(), 2);
        let sum = tf[0] + tf[1];
        assert!((sum - 1.0).abs() < 1e-6, "softmax row sums to 1, got {sum}");
        // class0 raw (2.0) > class1 raw (0.1) -> class0 prob larger.
        assert!(tf[0] > tf[1]);
    }

    #[test]
    fn leaf_index_regression_layout() {
        let m = model(); // 2 trees, ntpi=1, 2 iterations.
        let data = vec![1.0f32, 0.0]; // 1 row
        let leaf = predict_leaf_index_mat(&m, &data, 1, 2).unwrap();
        // per_row = ntpi(1) * num_iter(2) = 2.
        assert_eq!(leaf.len(), 2);
        // tree0 stump feat0: 1.0>0.5 -> leaf 1. tree1 stump feat1: 0.0<=0.5 -> leaf 0.
        assert_eq!(leaf, vec![1, 0]);
    }

    #[test]
    fn leaf_index_multiclass_per_iter_class_stride() {
        let m = multiclass_model(); // ntpi=2, 1 iteration.
        let data = vec![1.0f32, 1.0]; // 1 row: f0=1.0->leaf1, f1=1.0->leaf1
        let leaf = predict_leaf_index_mat(&m, &data, 1, 2).unwrap();
        // layout [iter0_class0, iter0_class1].
        assert_eq!(leaf, vec![1, 1]);
    }

    #[test]
    fn transformed_leaf_csr_csc_match_dense() {
        let m = multiclass_model();
        let data = vec![1.0f32, 0.0];
        let dense_tf = predict_mat(&m, &data, 1, 2).unwrap();
        let dense_leaf = predict_leaf_index_mat(&m, &data, 1, 2).unwrap();

        let (ip, ix, v) = (vec![0i64, 1], vec![0i32], vec![1.0f32]); // only f0=1.0
        let csr_tf = predict_csr(&m, &ip, &ix, &v, 1, 2).unwrap();
        let csr_leaf = predict_leaf_index_csr(&m, &ip, &ix, &v, 1, 2).unwrap();
        assert_eq!(dense_tf, csr_tf);
        assert_eq!(dense_leaf, csr_leaf);

        // CSC: col0 has row0=1.0; col1 empty.
        let (cip, cix, cv) = (vec![0i64, 1, 1], vec![0i32], vec![1.0f32]);
        let csc_tf = predict_csc(&m, &cip, &cix, &cv, 1, 2).unwrap();
        let csc_leaf = predict_leaf_index_csc(&m, &cip, &cix, &cv, 1, 2).unwrap();
        assert_eq!(dense_tf, csc_tf);
        assert_eq!(dense_leaf, csc_leaf);
    }

    #[test]
    fn transformed_out_of_scope_objective_is_err() {
        let mut m = model();
        m.objective_string = Some("lambdarank".to_string());
        let data = vec![1.0f32, 0.0];
        let err = predict_mat(&m, &data, 1, 2).unwrap_err();
        assert!(matches!(err, ModelError::MalformedModel { .. }));
    }

    #[test]
    fn transformed_too_few_cols_is_shape_mismatch() {
        let m = model();
        let data = vec![1.0f32];
        let err = predict_mat(&m, &data, 1, 1).unwrap_err();
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
    }

    // --- predict_contrib (PRD-04) ---

    #[test]
    fn contrib_sum_plus_base_equals_raw_dense() {
        // Two 2-leaf stumps over features 0 and 1; the per-class contrib sub-block
        // (per-feature + base) must sum to the raw margin for every row.
        let m = model(); // ntpi=1, nf=2 -> block width 3
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let raw = predict_raw_mat(&m, &data, 2, 2).unwrap();
        let contrib = predict_contrib_mat(&m, &data, 2, 2).unwrap();
        let block = 3; // num_features+1
        assert_eq!(contrib.len(), 2 * block);
        for r in 0..2 {
            let sum: f64 = contrib[r * block..(r + 1) * block].iter().sum();
            assert!(
                (sum - raw[r] as f64).abs() < 1e-6,
                "row {r}: sum(contrib)+base {sum} != raw {}",
                raw[r]
            );
        }
    }

    #[test]
    fn contrib_multiclass_per_class_block_sums_to_raw() {
        let m = multiclass_model(); // ntpi=2, nf=2 -> 2 blocks of width 3
        let data = vec![1.0f32, 0.0];
        let raw = predict_raw_mat(&m, &data, 1, 2).unwrap(); // 2 raw class margins
        let contrib = predict_contrib_mat(&m, &data, 1, 2).unwrap();
        let block = 3;
        assert_eq!(contrib.len(), 2 * block);
        for k in 0..2 {
            let sum: f64 = contrib[k * block..(k + 1) * block].iter().sum();
            assert!(
                (sum - raw[k] as f64).abs() < 1e-6,
                "class {k}: block sum {sum} != raw {}",
                raw[k]
            );
        }
    }

    #[test]
    fn contrib_csr_csc_match_dense() {
        let m = model();
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let dense = predict_contrib_mat(&m, &data, 2, 2).unwrap();

        let (ip, ix, v) = (vec![0i64, 1, 2], vec![0i32, 1], vec![1.0f32, 1.0]);
        let csr = predict_contrib_csr(&m, &ip, &ix, &v, 2, 2).unwrap();
        assert_eq!(dense, csr);

        let (cip, cix, cv) = (vec![0i64, 1, 2], vec![0i32, 1], vec![1.0f32, 1.0]);
        let csc = predict_contrib_csc(&m, &cip, &cix, &cv, 2, 2).unwrap();
        assert_eq!(dense, csc);
    }

    #[test]
    fn contrib_too_few_cols_is_shape_mismatch() {
        let m = model();
        let data = vec![1.0f32];
        let err = predict_contrib_mat(&m, &data, 1, 1).unwrap_err();
        assert!(matches!(err, ModelError::ShapeMismatch { .. }));
    }

    #[test]
    fn contrib_rejects_linear_tree_model() {
        // C++ never wires up PredictContrib for linear_tree=true models
        // (predictor.hpp:89-90) -- the Rust port must refuse it too, on all 3
        // entry points, rather than silently returning wrong (non-linear-aware)
        // SHAP values.
        let mut m = model();
        m.trees[0].is_linear = true;
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        assert!(matches!(
            predict_contrib_mat(&m, &data, 2, 2).unwrap_err(),
            ModelError::Unsupported { .. }
        ));
        let (indptr, indices, values) = (vec![0i64, 1, 2], vec![0i32, 1], vec![1.0f32, 1.0]);
        assert!(matches!(
            predict_contrib_csr(&m, &indptr, &indices, &values, 2, 2).unwrap_err(),
            ModelError::Unsupported { .. }
        ));
        assert!(matches!(
            predict_contrib_csc(&m, &indptr, &indices, &values, 2, 2).unwrap_err(),
            ModelError::Unsupported { .. }
        ));

        // A model with ONLY non-linear trees is unaffected (regression guard).
        let m2 = model();
        assert!(predict_contrib_mat(&m2, &data, 2, 2).is_ok());
    }

    // --- early-stop driver gating (PRD-05) ---

    #[test]
    fn early_stop_driver_disabled_for_regression() {
        // regression need_accurate_prediction()==true -> the hook is ignored even
        // with an aggressive freq/margin; result == full raw predict, all iters.
        let m = model(); // objective=regression, ntpi=1, 2 iters
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let raw = predict_raw_mat(&m, &data, 2, 2).unwrap();
        let (es, iters) =
            predict_raw_early_stop_mat(&m, &data, 2, 2, 1, 0.0).unwrap();
        assert_eq!(es, raw, "regression must ignore early stop (full predict)");
        assert_eq!(iters, vec![2, 2], "all iterations evaluated for regression");
    }

    #[test]
    fn early_stop_driver_active_for_binary() {
        // A binary model: ntpi=1, objective=binary. With a tiny margin the binary
        // 2*|score| check fires after the first iteration.
        let mut m = model();
        m.objective_string = Some("binary sigmoid:1".to_string());
        let data = vec![1.0f32, 0.0]; // f0=1.0->tree0 leaf1=2.0
        let (_es, iters) =
            predict_raw_early_stop_mat(&m, &data, 1, 2, 1, 1.0).unwrap();
        // tree0 contributes 2.0; 2*|2.0|=4.0 > 1.0 -> stop after 1 iteration.
        assert_eq!(iters, vec![1], "binary early stop fires after first iter");
    }
}
