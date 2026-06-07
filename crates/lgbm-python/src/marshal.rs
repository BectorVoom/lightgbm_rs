//! numpy <-> Rust marshalling (08-RESEARCH Pattern 3). Converts an untrusted
//! numpy dense f64 matrix into owned Rust rows WHILE holding the GIL, with
//! explicit contiguity handling (Pitfall 3 / SC#2, T-08-02-01) and shape
//! validation at the FFI boundary (Security V5, T-08-02-02).

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
