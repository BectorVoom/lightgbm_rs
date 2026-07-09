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

use std::collections::HashSet;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use polars::prelude::{Categories, DataType, Series};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

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

// ---------------------------------------------------------------------------
// polars DataFrame ingest (08-04, D-03/D-04)
// ---------------------------------------------------------------------------

/// Extract a polars [`Series`] as an owned `Vec<f64>`, null -> `NaN` (the facade
/// `BinMapper` treats `NaN` as missing). Numeric columns are cast to f64; for a
/// categorical/enum series this receives its already-physical-code repr.
///
/// Two ingest fast-paths, both byte-identical to the prior
/// `cast().f64().into_iter().map(unwrap_or(NAN)).collect()`:
/// - **O3:** skip the `cast(Float64)` (a whole extra `Series` allocation) when the
///   column is ALREADY `Float64` — the common numeric case — borrowing it instead.
/// - **O5:** build the `Vec` by walking the underlying Arrow CHUNKS
///   (`downcast_iter`) rather than forcing a single materialized/rechunked buffer.
///   A chunk with no validity bitmap (`null_count == 0`) is copied wholesale from
///   its contiguous values slice; a chunk with nulls falls back to the per-value
///   null -> NaN map. (The contiguous-copy step is what makes O5 a real win over
///   `into_iter()`, which already walks chunks — it necessarily overlaps the O4
///   no-null bulk-copy idea.)
fn series_to_f64_vec(s: &Series, name: &str) -> PyResult<Vec<f64>> {
    // O3: only allocate a casted Series when the dtype is not already Float64.
    let casted: std::borrow::Cow<'_, Series> = if matches!(s.dtype(), DataType::Float64) {
        std::borrow::Cow::Borrowed(s)
    } else {
        std::borrow::Cow::Owned(s.cast(&DataType::Float64).map_err(|e| {
            PyValueError::new_err(format!(
                "column '{name}': cannot convert dtype {:?} to numeric f64: {e}",
                s.dtype()
            ))
        })?)
    };
    let ca = casted
        .f64()
        .map_err(|e| PyValueError::new_err(format!("column '{name}': {e}")))?;

    // O5: chunk-wise extraction — no rechunk, no materialized round-trip.
    let mut out: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        match arr.validity() {
            // No validity bitmap ⇒ every value present ⇒ copy the contiguous slice.
            None => out.extend_from_slice(arr.values().as_slice()),
            // Has a validity bitmap ⇒ map per value, null -> NaN (present NaN bits
            // are preserved, identical to the prior into_iter().unwrap_or(NAN)).
            Some(_) => {
                for opt in arr.iter() {
                    out.push(opt.copied().unwrap_or(f64::NAN));
                }
            }
        }
    }
    Ok(out)
}

/// Extract one DataFrame column's raw f64 feature values.
///
/// When `as_categorical` the values are the category CODES the facade
/// `find_bin_categorical` path consumes (D-04):
/// - `Categorical`/`Enum` -> the physical integer codes (`to_physical_repr`),
///   widened to f64 (handles any physical width: u8/u16/u32/u64).
/// - `String` -> cast to a global `Categorical` first, then its physical codes.
/// - numeric (an explicit override of a numeric column) -> the values ARE the
///   codes; taken verbatim as f64.
///
/// Otherwise the column is a numeric feature cast straight to f64.
fn column_feature_values(s: &Series, name: &str, as_categorical: bool) -> PyResult<Vec<f64>> {
    if !as_categorical {
        return series_to_f64_vec(s, name);
    }
    match s.dtype() {
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let phys = s.to_physical_repr().into_owned();
            series_to_f64_vec(&phys, name)
        }
        DataType::String => {
            let cat = s
                .cast(&DataType::from_categories(Categories::global()))
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "column '{name}': cannot cast string to categorical: {e}"
                    ))
                })?;
            let phys = cat.to_physical_repr().into_owned();
            series_to_f64_vec(&phys, name)
        }
        // A numeric column explicitly overridden to categorical: its integer-valued
        // entries are the category codes already.
        _ => series_to_f64_vec(s, name),
    }
}

/// The owned parts a polars DataFrame marshals into: COLUMN-major f64 feature
/// columns (`columns[j]` = feature `j`'s per-row values), the column (feature)
/// names, and the indices routed to categorical features.
pub type CorpusParts = (Vec<Vec<f64>>, Vec<String>, Vec<usize>);

/// Marshal a polars DataFrame (consumed Arrow-side in Rust via pyo3-polars — NO
/// numpy round-trip, which would erase the Categorical/Enum dtype) into the owned
/// COLUMN-major f64 columns + feature names + categorical-feature index set that
/// build a [`lgbm::RawCorpus`] (via `RawCorpus::from_columns`, D-03/D-04). The
/// columns are returned in the source's natural Arrow layout — NO transpose (O1).
///
/// Routing (D-04): with `categorical_override = None` (the default `'auto'`),
/// `Categorical`/`Enum`/`String` columns are categorical and numeric columns are
/// numeric. When `categorical_override` is `Some(names)`, that list FULLY decides
/// routing (official-package precedence over dtype) — only the named columns are
/// categorical, everything else numeric.
///
/// # Errors
/// `PyValueError` for an empty frame, an unknown override name, or a column that
/// cannot be coerced — never panics.
pub fn polars_df_to_corpus(
    pydf: PyDataFrame,
    categorical_override: Option<Vec<String>>,
) -> PyResult<CorpusParts> {
    let df = pydf.0;
    let columns = df.columns();
    let ncols = columns.len();
    if ncols == 0 {
        return Err(PyValueError::new_err("DataFrame has no columns"));
    }
    let nrows = df.height();
    if nrows == 0 {
        return Err(PyValueError::new_err("DataFrame has no rows"));
    }

    let names: Vec<String> = columns.iter().map(|c| c.name().to_string()).collect();

    // Validate override names against the actual columns BEFORE routing.
    let override_set: Option<HashSet<String>> = match categorical_override {
        Some(list) => {
            let known: HashSet<&str> = names.iter().map(String::as_str).collect();
            for n in &list {
                if !known.contains(n.as_str()) {
                    return Err(PyValueError::new_err(format!(
                        "categorical_feature '{n}' is not a column of the DataFrame"
                    )));
                }
            }
            Some(list.into_iter().collect())
        }
        None => None,
    };

    let mut col_data: Vec<Vec<f64>> = Vec::with_capacity(ncols);
    let mut cat_indices: Vec<usize> = Vec::new();

    for (j, col) in columns.iter().enumerate() {
        let s = col.as_materialized_series();
        let name = &names[j];
        let is_cat = match &override_set {
            // Override fully decides routing (D-04 precedence).
            Some(set) => set.contains(name),
            // Default 'auto' = dtype routing.
            None => matches!(
                s.dtype(),
                DataType::Categorical(_, _) | DataType::Enum(_, _) | DataType::String
            ),
        };
        if is_cat {
            cat_indices.push(j);
        }
        let values = column_feature_values(s, name, is_cat)?;
        if values.len() != nrows {
            return Err(PyValueError::new_err(format!(
                "column '{name}' length {} != frame height {nrows}",
                values.len()
            )));
        }
        col_data.push(values);
    }

    // O1: return the COLUMN-major data verbatim. The Arrow source is already
    // columnar and the facade's `RawCorpus::from_columns` stores it column-major,
    // so the old col→row transpose here (followed by a row→col transpose inside
    // `build_feature_columns_from_raw`) is eliminated — binning reads each feature
    // as a contiguous slice with no round trip.
    Ok((col_data, names, cat_indices))
}
