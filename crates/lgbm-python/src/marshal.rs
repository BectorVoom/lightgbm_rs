//! numpy/scipy <-> Rust marshalling (08-RESEARCH Patterns 3 & 4). Converts an
//! untrusted numpy dense f32/f64 matrix OR a scipy CSR/CSC sparse matrix into
//! owned Rust rows WHILE holding the GIL, with explicit contiguity/dtype handling
//! (Pitfall 3 / SC#2, T-08-03-02) and shape/indptr validation at the FFI boundary
//! (Security V5, T-08-03-01).
//!
//! # The single f32 -> f64 widen site
//!
//! Both numpy widths and the sparse `values` are caller `f32`/`f64` data that must
//! end up as the f64 rows the facade [`lgbm::RawCorpus`] consumes (binning is f64;
//! see `ingest.rs`'s identical `widen` contract). Every conversion flows through
//! exactly ONE place per path: [`widen`]. Do NOT scatter `as f64` casts (T-08-03-03).

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The SINGLE f32 -> f64 widen site for caller dense/sparse feature data crossing
/// from numpy/scipy into the f64 rows the facade bins. Mirrors `ingest.rs::widen`
/// so the f32 and f64 Python paths are provably identical after this one point
/// (the self-consistency guarantee, T-08-03-03).
#[inline]
fn widen(v: f32) -> f64 {
    v as f64
}

/// Copy a numpy 2-D f64 array into owned row-major Rust rows.
///
/// Contiguity (T-08-02-01): `ndarray::ArrayView::rows()` iterates logically
/// regardless of the underlying memory layout, so a non-C-contiguous (e.g.
/// Fortran-ordered or sliced) array is read CORRECTLY here — we never index the
/// raw buffer directly. We still copy every element into freshly-owned `Vec`s so
/// nothing is lent back to Python (SC#1) and the buffer can be read with the GIL
/// released afterwards.
///
/// # Errors
/// `PyValueError` when the array is empty (zero rows or zero columns) — validated
/// before any indexing (Security V5).
pub fn numpy_dense_to_rows(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vec<f64>>> {
    let view = arr.as_array();
    let shape = view.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    if nrows == 0 || ncols == 0 {
        return Err(PyValueError::new_err(format!(
            "input array must be non-empty (got shape {nrows}x{ncols})"
        )));
    }
    // Explicit contiguity awareness (Pitfall 3 / SC#2): a standard (C-contiguous)
    // layout could be bulk-copied, but `rows()` is correct for ANY layout, so we
    // route both through it and only branch to document the check. `is_standard_layout`
    // is queried so the contiguity handling is explicit at this boundary.
    let _standard = view.is_standard_layout();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(nrows);
    for row in view.rows() {
        rows.push(row.iter().copied().collect());
    }
    Ok(rows)
}

/// Copy a numpy 2-D **f32** array into owned row-major Rust rows, widening to f64
/// at the single [`widen`] site (PYB-02 SC#2 — accept both widths).
///
/// Identical contiguity treatment to [`numpy_dense_to_rows`]: `view.rows()` reads
/// the logical view correctly for ANY layout, so a non-C-contiguous f32 array
/// (Fortran-order / sliced) is read correctly (T-08-03-02). The widen happens at
/// ONE site, so f32 and f64 of the same data yield byte-identical f64 rows and
/// therefore identical bins / models (the self-consistency guarantee, T-08-03-03).
///
/// # Errors
/// `PyValueError` when the array is empty — validated before any indexing
/// (Security V5).
pub fn numpy_dense_f32_to_rows(arr: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Vec<f64>>> {
    let view = arr.as_array();
    let shape = view.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    if nrows == 0 || ncols == 0 {
        return Err(PyValueError::new_err(format!(
            "input array must be non-empty (got shape {nrows}x{ncols})"
        )));
    }
    // Explicit contiguity awareness (T-08-03-02): queried so the handling is
    // explicit at this boundary; `rows()` is correct for ANY layout.
    let _standard = view.is_standard_layout();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(nrows);
    for row in view.rows() {
        rows.push(row.iter().copied().map(widen).collect());
    }
    Ok(rows)
}

/// Copy a numpy 1-D f64 array into an owned `Vec<f32>` (the label width — labels
/// are `label_t = f32` end-to-end, CLAUDE.md). Reads via the logical view so any
/// layout is handled.
///
/// # Errors
/// `PyValueError` when empty.
pub fn numpy_labels_to_f32(arr: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f32>> {
    let view = arr.as_array();
    if view.is_empty() {
        return Err(PyValueError::new_err("label array must be non-empty"));
    }
    Ok(view.iter().map(|&v| v as f32).collect())
}

/// Validate a CSR/CSC `indptr` BEFORE any indexing (Security V5 / T-08-03-01),
/// mirroring `ingest.rs::validate_indptr`: `indptr.len() == outer + 1`,
/// `indptr[0] == 0`, monotone non-decreasing, and `last == nnz`. Surfaces a
/// `PyValueError` on any violation — never indexes first, never panics.
fn validate_indptr(indptr: &[i64], outer: usize, nnz: usize) -> PyResult<()> {
    if indptr.len() != outer + 1 {
        return Err(PyValueError::new_err(format!(
            "indptr.len()={} must equal outer+1={}",
            indptr.len(),
            outer + 1
        )));
    }
    // outer+1 >= 1 here, so indptr[0] is in bounds.
    if indptr[0] != 0 {
        return Err(PyValueError::new_err(format!(
            "indptr[0]={} must be 0",
            indptr[0]
        )));
    }
    for w in indptr.windows(2) {
        if w[1] < w[0] {
            return Err(PyValueError::new_err(format!(
                "indptr not monotone non-decreasing: {} then {}",
                w[0], w[1]
            )));
        }
    }
    let last = indptr[indptr.len() - 1];
    if last != nnz as i64 {
        return Err(PyValueError::new_err(format!(
            "indptr last={last} must equal nnz={nnz}"
        )));
    }
    Ok(())
}

/// Densify a scipy **CSR** matrix (`indptr`/`indices`/`data`) into the owned f64
/// row-major rows the facade [`lgbm::RawCorpus`] consumes, with absent entries
/// defaulting to `0.0`. This is the EXACT gather-with-zeros that
/// `ingest::from_csr` performs internally, so a CSR matrix and its dense
/// equivalent produce identical rows and therefore identical bins/models (D-05,
/// CSR↔dense equivalence inherits the Phase-2 guarantee).
///
/// ALL caller input is validated BEFORE any indexing into `data`/`indices`
/// (Security V5 / T-08-03-01): `indices.len() == data.len()`, `indptr`
/// well-formed (`validate_indptr`), and every `indices[k]` in `[0, num_cols)`.
/// Widens `f32 -> f64` at the single [`widen`] site.
///
/// # Errors
/// `PyValueError` on a non-positive shape, ragged arrays, malformed `indptr`, or
/// an out-of-range column index — never panics.
pub fn scipy_csr_to_rows(
    indptr: &PyReadonlyArray1<'_, i64>,
    indices: &PyReadonlyArray1<'_, i32>,
    data: &PyReadonlyArray1<'_, f32>,
    num_rows: usize,
    num_cols: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    if num_rows == 0 || num_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "sparse matrix must be non-empty (got shape {num_rows}x{num_cols})"
        )));
    }
    if indices.len() != data.len() {
        return Err(PyValueError::new_err(format!(
            "indices.len()={} must equal data.len()={}",
            indices.len(),
            data.len()
        )));
    }
    validate_indptr(indptr, num_rows, data.len())?;
    for &c in indices {
        if c < 0 || (c as usize) >= num_cols {
            return Err(PyValueError::new_err(format!(
                "column index {c} out of range [0, {num_cols})"
            )));
        }
    }
    let mut rows: Vec<Vec<f64>> = vec![vec![0.0f64; num_cols]; num_rows];
    for row in 0..num_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            rows[row][indices[k] as usize] = widen(data[k]);
        }
    }
    Ok(rows)
}

/// Densify a scipy **CSC** matrix (`indptr`/`indices`/`data`) into the owned f64
/// row-major rows the facade [`lgbm::RawCorpus`] consumes — the column-oriented
/// mirror of [`scipy_csr_to_rows`], matching `ingest::from_csc`'s gather. `indptr`
/// has `num_cols + 1` entries; `indices` are ROW indices in `[0, num_rows)`.
///
/// Same Security-V5 validation BEFORE any indexing (T-08-03-01). Widens at the
/// single [`widen`] site.
///
/// # Errors
/// `PyValueError` on a non-positive shape, ragged arrays, malformed `indptr`, or
/// an out-of-range row index — never panics.
pub fn scipy_csc_to_rows(
    indptr: &PyReadonlyArray1<'_, i64>,
    indices: &PyReadonlyArray1<'_, i32>,
    data: &PyReadonlyArray1<'_, f32>,
    num_rows: usize,
    num_cols: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;
    if num_rows == 0 || num_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "sparse matrix must be non-empty (got shape {num_rows}x{num_cols})"
        )));
    }
    if indices.len() != data.len() {
        return Err(PyValueError::new_err(format!(
            "indices.len()={} must equal data.len()={}",
            indices.len(),
            data.len()
        )));
    }
    validate_indptr(indptr, num_cols, data.len())?;
    for &r in indices {
        if r < 0 || (r as usize) >= num_rows {
            return Err(PyValueError::new_err(format!(
                "row index {r} out of range [0, {num_rows})"
            )));
        }
    }
    let mut rows: Vec<Vec<f64>> = vec![vec![0.0f64; num_cols]; num_rows];
    for col in 0..num_cols {
        let start = indptr[col] as usize;
        let end = indptr[col + 1] as usize;
        for k in start..end {
            rows[indices[k] as usize][col] = widen(data[k]);
        }
    }
    Ok(rows)
}
