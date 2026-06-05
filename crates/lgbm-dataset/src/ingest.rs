//! The minimal internal-facing ingestion API (D-05): `from_mat` / `from_csr` /
//! `from_csc`. Wires the full pipeline VERBATIM from the C-API in-memory path
//! (`LightGBM/src/c_api.cpp` `SampleCount`/`CreateSampleIndices` 974-982,
//! `LGBM_DatasetCreateFromMat`/`FromCSR`/`FromCSC`):
//!
//! ```text
//! sample row indices (Phase-1 RNG, data_random_seed)
//!   -> gather per-column sampled values (f32 -> f64 at ONE widen site)
//!   -> find_bin per column (sequential, D-03)
//!   -> Dataset::construct
//!   -> push each row's features (FeatureGroup::PushData)
//!   -> finish_load (the immutability boundary)
//! ```
//!
//! Follows `lgbm-core/src/config/set.rs`'s validated-entry-point discipline:
//! each `from_*` is a SINGLE public entry that VALIDATES all caller input at the
//! boundary FIRST (Security V5 — typed [`DatasetError`], never a panic), then
//! runs the fixed-order pipeline. No `unsafe` indexing on caller data.
//!
//! # The single f32 -> f64 widen site
//!
//! Caller feature data is `&[f32]` (the public dense/sparse value type). Binning
//! arithmetic is entirely f64 (`bin_mapper.rs` contract). The widen happens at
//! exactly ONE documented point: [`widen`]. Do NOT scatter `as f64` casts —
//! every value flows through `widen` so the conversion site is auditable.
//!
//! # Sampling routes through the Phase-1 RNG
//!
//! `create_sample_indices` builds `Random::new(cfg.data_random_seed)` and calls
//! `.sample(total_nrow, min(total_nrow, bin_construct_sample_cnt))` — the exact
//! C-API path. There is NO new RNG here (the function is re-exported from
//! `bin_mapper`).

use lgbm_core::config::Config;

pub use crate::bin_mapper::{create_sample_indices, sample_count};
use crate::bin_mapper::BinMapper;
use crate::dataset::FinishedDataset;
use crate::error::DatasetError;
use crate::metadata::Metadata;

/// The SINGLE f32 -> f64 widen site for caller feature data. Every feature value
/// that crosses from the caller's `f32` buffers into the f64 binning arithmetic
/// passes through here (audit point; do NOT scatter `as f64`).
#[inline]
fn widen(v: f32) -> f64 {
    v as f64
}

/// Validate the shared binning config knobs (Security V5 / T-02-12): reject a
/// non-positive `max_bin` (C++ `CHECK_GT(max_bin, 0)`) and a negative
/// `min_data_in_bin` / `bin_construct_sample_cnt` BEFORE allocating.
fn validate_binning_config(cfg: &Config) -> Result<(), DatasetError> {
    if cfg.max_bin <= 0 {
        return Err(DatasetError::InvalidConfig {
            detail: format!("max_bin must be > 0, got {}", cfg.max_bin),
        });
    }
    if cfg.min_data_in_bin < 0 {
        return Err(DatasetError::InvalidConfig {
            detail: format!("min_data_in_bin must be >= 0, got {}", cfg.min_data_in_bin),
        });
    }
    if cfg.bin_construct_sample_cnt < 0 {
        return Err(DatasetError::InvalidConfig {
            detail: format!(
                "bin_construct_sample_cnt must be >= 0, got {}",
                cfg.bin_construct_sample_cnt
            ),
        });
    }
    Ok(())
}

/// Build the numeric `BinMapper` for one column from its full per-row values.
///
/// Mirrors the C-API path: sample row indices via the Phase-1 RNG, gather the
/// sampled rows' values, then `find_bin_numeric` with `total_sample_cnt = k`.
/// `column` is already f64-widened. This is the per-feature constructor (D-03):
/// no shared mutable state, so a later `par_iter` over columns is a one-line swap.
fn build_mapper(column: &[f64], cfg: &Config) -> BinMapper {
    let total_nrow = column.len() as i32;
    let indices = create_sample_indices(
        total_nrow,
        cfg.bin_construct_sample_cnt,
        cfg.data_random_seed,
    );
    let sampled: Vec<f64> = indices.iter().map(|&i| column[i as usize]).collect();
    let total_sample_cnt = sampled.len();
    BinMapper::find_bin_numeric(
        sampled,
        cfg.max_bin,
        cfg.min_data_in_bin,
        cfg.min_data_in_leaf,
        cfg.feature_pre_filter,
        cfg.use_missing,
        cfg.zero_as_missing,
        total_sample_cnt,
        &[],
    )
}

/// Run the shared tail of every ingestion path: build one mapper per column,
/// `Dataset::construct`, push every row, attach the validated metadata, and
/// `finish_load` (the immutability boundary).
///
/// `columns[c]` is the full f64-widened per-row column for feature `c`; every
/// column has the same length `num_rows`.
fn finish_from_columns(
    columns: &[Vec<f64>],
    num_rows: i32,
    cfg: &Config,
    mut metadata: Metadata,
) -> Result<(FinishedDataset, Metadata), DatasetError> {
    let mappers: Vec<BinMapper> = columns.iter().map(|col| build_mapper(col, cfg)).collect();
    let mut ds = crate::dataset::Dataset::construct(mappers, num_rows)?;
    for row in 0..num_rows {
        for (feature, col) in columns.iter().enumerate() {
            ds.push_value(feature, row, col[row as usize]);
        }
    }
    let finished = ds.finish_load();
    metadata.finish_load();
    Ok((finished, metadata))
}

/// Ingest a DENSE row-major matrix (`data[row*num_cols + col]`) + metadata into
/// an immutable [`FinishedDataset`].
///
/// VALIDATES first (Security V5 / T-02-11): `num_rows`/`num_cols` non-negative,
/// `data.len() == num_rows * num_cols`, and the binning config. Returns a typed
/// [`DatasetError`] on any violation — never a panic. The returned [`Metadata`]
/// has run `finish_load` (query weights derived).
pub fn from_mat(
    data: &[f32],
    num_rows: i32,
    num_cols: i32,
    cfg: &Config,
    metadata: Metadata,
) -> Result<(FinishedDataset, Metadata), DatasetError> {
    if num_rows < 0 || num_cols < 0 {
        return Err(DatasetError::ShapeMismatch {
            detail: format!("num_rows={num_rows} / num_cols={num_cols} must be >= 0"),
        });
    }
    let expected = (num_rows as i64) * (num_cols as i64);
    if data.len() as i64 != expected {
        return Err(DatasetError::ShapeMismatch {
            detail: format!(
                "data.len()={} must equal num_rows*num_cols={expected}",
                data.len()
            ),
        });
    }
    validate_binning_config(cfg)?;
    if metadata.num_data() != num_rows {
        return Err(DatasetError::ShapeMismatch {
            detail: format!(
                "metadata num_rows={} must equal num_rows={num_rows}",
                metadata.num_data()
            ),
        });
    }

    // Gather columns (widen f32 -> f64 at the single widen site).
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(num_rows as usize); num_cols as usize];
    for row in 0..num_rows as usize {
        let base = row * num_cols as usize;
        for col in 0..num_cols as usize {
            columns[col].push(widen(data[base + col]));
        }
    }

    finish_from_columns(&columns, num_rows, cfg, metadata)
}

/// Validate a CSR/CSC `indptr` (monotone non-decreasing, `indptr[0] == 0`,
/// `indptr.len() == outer + 1`, `last == nnz`) — Security V5 / T-02-10. Returns
/// the nnz on success.
fn validate_indptr(indptr: &[i64], outer: i64, nnz: usize) -> Result<(), DatasetError> {
    if indptr.len() as i64 != outer + 1 {
        return Err(DatasetError::MalformedSparse {
            detail: format!(
                "indptr.len()={} must equal outer+1={}",
                indptr.len(),
                outer + 1
            ),
        });
    }
    if indptr[0] != 0 {
        return Err(DatasetError::MalformedSparse {
            detail: format!("indptr[0]={} must be 0", indptr[0]),
        });
    }
    for w in indptr.windows(2) {
        if w[1] < w[0] {
            return Err(DatasetError::MalformedSparse {
                detail: format!("indptr not monotone non-decreasing: {} then {}", w[0], w[1]),
            });
        }
    }
    let last = *indptr.last().unwrap();
    if last != nnz as i64 {
        return Err(DatasetError::MalformedSparse {
            detail: format!("indptr last={last} must equal nnz={nnz}"),
        });
    }
    Ok(())
}

/// Ingest a CSR (Compressed Sparse Row) matrix + metadata. `indptr` has
/// `num_rows + 1` entries; for row `r`, the stored `(col, value)` pairs are
/// `indices[indptr[r]..indptr[r+1]]` / `values[...]`. Absent entries are zero.
///
/// VALIDATES (Security V5 / T-02-10): `indices.len() == values.len() == nnz`,
/// `indptr` well-formed, and every `indices[k] < num_cols` — BEFORE any indexing
/// into `values`/columns. Returns a typed [`DatasetError`], never panics.
pub fn from_csr(
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
    cfg: &Config,
    metadata: Metadata,
) -> Result<(FinishedDataset, Metadata), DatasetError> {
    if num_rows < 0 || num_cols < 0 {
        return Err(DatasetError::ShapeMismatch {
            detail: format!("num_rows={num_rows} / num_cols={num_cols} must be >= 0"),
        });
    }
    if indices.len() != values.len() {
        return Err(DatasetError::MalformedSparse {
            detail: format!(
                "indices.len()={} must equal values.len()={}",
                indices.len(),
                values.len()
            ),
        });
    }
    validate_indptr(indptr, num_rows as i64, values.len())?;
    for &c in indices {
        if c < 0 || c >= num_cols {
            return Err(DatasetError::MalformedSparse {
                detail: format!("column index {c} out of range [0, {num_cols})"),
            });
        }
    }
    validate_binning_config(cfg)?;
    if metadata.num_data() != num_rows {
        return Err(DatasetError::ShapeMismatch {
            detail: format!(
                "metadata num_rows={} must equal num_rows={num_rows}",
                metadata.num_data()
            ),
        });
    }

    // Dense-by-column gather: absent entries default to 0.0 (zero_cnt-aware,
    // Open Q2). Widen f32 -> f64 at the single widen site.
    let mut columns: Vec<Vec<f64>> = vec![vec![0.0f64; num_rows as usize]; num_cols as usize];
    for row in 0..num_rows as usize {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            columns[col][row] = widen(values[k]);
        }
    }

    finish_from_columns(&columns, num_rows, cfg, metadata)
}

/// Ingest a CSC (Compressed Sparse Column) matrix + metadata. `indptr` has
/// `num_cols + 1` entries; for column `c`, the stored `(row, value)` pairs are
/// `indices[indptr[c]..indptr[c+1]]` / `values[...]`. Absent entries are zero.
///
/// VALIDATES (Security V5 / T-02-10): `indices.len() == values.len() == nnz`,
/// `indptr` well-formed, and every `indices[k] < num_rows` — BEFORE any indexing.
/// Returns a typed [`DatasetError`], never panics.
pub fn from_csc(
    indptr: &[i64],
    indices: &[i32],
    values: &[f32],
    num_rows: i32,
    num_cols: i32,
    cfg: &Config,
    metadata: Metadata,
) -> Result<(FinishedDataset, Metadata), DatasetError> {
    if num_rows < 0 || num_cols < 0 {
        return Err(DatasetError::ShapeMismatch {
            detail: format!("num_rows={num_rows} / num_cols={num_cols} must be >= 0"),
        });
    }
    if indices.len() != values.len() {
        return Err(DatasetError::MalformedSparse {
            detail: format!(
                "indices.len()={} must equal values.len()={}",
                indices.len(),
                values.len()
            ),
        });
    }
    validate_indptr(indptr, num_cols as i64, values.len())?;
    for &r in indices {
        if r < 0 || r >= num_rows {
            return Err(DatasetError::MalformedSparse {
                detail: format!("row index {r} out of range [0, {num_rows})"),
            });
        }
    }
    validate_binning_config(cfg)?;
    if metadata.num_data() != num_rows {
        return Err(DatasetError::ShapeMismatch {
            detail: format!(
                "metadata num_rows={} must equal num_rows={num_rows}",
                metadata.num_data()
            ),
        });
    }

    // Dense-by-column gather: absent entries default to 0.0 (Open Q2). Widen at
    // the single widen site.
    let mut columns: Vec<Vec<f64>> = vec![vec![0.0f64; num_rows as usize]; num_cols as usize];
    for col in 0..num_cols as usize {
        let start = indptr[col] as usize;
        let end = indptr[col + 1] as usize;
        for k in start..end {
            let row = indices[k] as usize;
            columns[col][row] = widen(values[k]);
        }
    }

    finish_from_columns(&columns, num_rows, cfg, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        // Use a small sample cnt so sampling is exercised; defaults otherwise.
        let mut c = Config::default();
        c.max_bin = 16;
        c.min_data_in_bin = 1;
        c.bin_construct_sample_cnt = 100000; // sample all rows for determinism
        c.feature_pre_filter = false;
        c
    }

    fn meta(n: i32) -> Metadata {
        Metadata::new(vec![0.0f32; n as usize], Vec::new(), Vec::new(), Vec::new()).unwrap()
    }

    #[test]
    fn from_mat_shape_mismatch_is_typed_error() {
        let c = cfg();
        let err = from_mat(&[1.0, 2.0, 3.0], 2, 2, &c, meta(2)).unwrap_err();
        assert!(matches!(err, DatasetError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_mat_invalid_max_bin_is_typed_error() {
        let mut c = cfg();
        c.max_bin = 0;
        let err = from_mat(&[1.0, 2.0, 3.0, 4.0], 2, 2, &c, meta(2)).unwrap_err();
        assert!(matches!(err, DatasetError::InvalidConfig { .. }));
    }

    #[test]
    fn from_csr_non_monotone_indptr_is_typed_error() {
        let c = cfg();
        // indptr 0,2,1 — not monotone.
        let err = from_csr(
            &[0, 2, 1],
            &[0, 1, 0],
            &[1.0, 2.0, 3.0],
            2,
            2,
            &c,
            meta(2),
        )
        .unwrap_err();
        assert!(matches!(err, DatasetError::MalformedSparse { .. }));
    }

    #[test]
    fn from_csr_out_of_range_index_is_typed_error() {
        let c = cfg();
        // column index 5 >= num_cols 2.
        let err = from_csr(&[0, 1], &[5], &[1.0], 1, 2, &c, meta(1)).unwrap_err();
        assert!(matches!(err, DatasetError::MalformedSparse { .. }));
    }

    #[test]
    fn from_csc_out_of_range_row_is_typed_error() {
        let c = cfg();
        // row index 9 >= num_rows 2.
        let err = from_csc(&[0, 1, 1], &[9], &[1.0], 2, 2, &c, meta(2)).unwrap_err();
        assert!(matches!(err, DatasetError::MalformedSparse { .. }));
    }

    #[test]
    fn from_mat_builds_finished_dataset() {
        let c = cfg();
        // 3 rows, 2 cols, row-major.
        let data = vec![
            0.0, 10.0, //
            1.0, 20.0, //
            2.0, 30.0, //
        ];
        let (ds, _md) = from_mat(&data, 3, 2, &c, meta(3)).unwrap();
        assert_eq!(ds.num_data(), 3);
        assert_eq!(ds.num_features(), 2);
    }
}
