---
quick_id: 260609-9nu
title: Investigate dataset for arrow-rs optimisation opportunities
type: investigation
status: complete
date: 2026-06-08
---

# Findings: Arrow-rs optimisation opportunities in the dataset path

## TL;DR

1. **`lgbm-dataset` uses no Apache Arrow / arrow-rs at all.** The only two "arrow"
   grep hits in `crates/lgbm-dataset/src` are the English word **"narrow"** in
   comments (`bin/mod.rs:29`, `multi_val_bin.rs:84`). There is nothing to
   optimise *in that crate* with respect to Arrow.

2. **The real Arrow boundary lives in `crates/lgbm-python`**, where polars
   DataFrames arrive over the Arrow C-stream via `pyo3-polars` (`polars-arrow`).
   That path — `marshal.rs::polars_df_to_corpus` → `RawCorpus` →
   `build_feature_columns_from_raw` — has concrete, measurable waste.

3. **Headline finding — a double transpose + nested-Vec layout.** Arrow data is
   *already columnar* (exactly what binning wants), but the marshal layer
   transposes it to **row-major `Vec<Vec<f64>>`** (one heap allocation per row),
   and the core then transposes it **back** to columns to bin. Two transposes and
   `num_rows` allocations are pure overhead on the Arrow path.

## Where Arrow actually is

| Concern | Location |
|---|---|
| Arrow C-stream import (no numpy round-trip) | `crates/lgbm-python/src/dataset.rs:126` `from_polars` |
| Series → `Vec<f64>` (cast + null→NaN) | `crates/lgbm-python/src/marshal.rs:260` `series_to_f64_vec` |
| Categorical/Enum/String → physical codes | `crates/lgbm-python/src/marshal.rs:284` `column_feature_values` |
| Column gather + **transpose to row-major** | `crates/lgbm-python/src/marshal.rs:361-396` `polars_df_to_corpus` |
| Row-major store | `crates/lgbm/src/booster.rs:256` `RawCorpus.features: Vec<Vec<f64>>` |
| **Transpose back to columns** for binning | `crates/lgbm/src/booster.rs:160-180` `build_feature_columns_from_raw` |
| Per-value bin lookup (downstream hot loop) | `crates/lgbm-dataset/src/bin_mapper.rs:148` `value_to_bin` |

## Optimisation opportunities (ranked)

### O1 — Eliminate the column→row→column round trip (HIGH, biggest win)
`polars_df_to_corpus` builds `col_data: Vec<Vec<f64>>` column-major (the natural
Arrow layout), then immediately transposes into `rows: Vec<Vec<f64>>`
(`marshal.rs:389-395`). Downstream, `build_feature_columns_from_raw`
(`booster.rs:175-180`) transposes back to columns because binning is
column-at-a-time (`ingest.rs::finish_from_columns` operates per column).

- The transpose touches every cell twice and scatters writes across `num_rows`
  separate heap allocations (cache-hostile, allocator-heavy).
- Fix direction: let the polars path hand its **already-columnar** `col_data`
  straight to a column-oriented corpus/ingest entry, skipping both transposes.
  This is a marshal/`RawCorpus` shape change, not an algorithm change — parity
  neutral (same f64 values, same order).

### O2 — Flatten `Vec<Vec<f64>>` to a single contiguous buffer (HIGH)
Both `RawCorpus.features` and the marshal `rows`/`col_data` are nested `Vec<Vec<f64>>`.
A flat `Vec<f64>` + stride (column-major: `col*num_rows + row`) removes per-row
allocations, improves prefetch, and makes a future `rayon` per-column `par_iter`
trivial. Pairs naturally with O1.

### O3 — Avoid the extra f64 cast copy per column (MEDIUM)
`series_to_f64_vec` does `s.cast(&DataType::Float64)` (allocates a casted Series)
**then** `ca.into_iter().map(...).collect()` (allocates the `Vec<f64>`). For an
already-f64 column the cast is a wasted copy; for other numeric dtypes the cast +
collect is two passes. Consider casting in place / collecting directly from the
typed chunk, and short-circuiting when `dtype == Float64`.

### O4 — Bulk null handling instead of per-value `Option::unwrap_or` (LOW–MEDIUM)
`ca.into_iter().map(|opt| opt.unwrap_or(f64::NAN))` (`marshal.rs:270`) branches per
value. When a chunk has **no** validity bitmap (no nulls — common case), the
values buffer can be copied wholesale (`memcpy`/slice clone) with no per-element
branch. Check `null_count == 0` per chunk and take the fast path.

### O5 — Iterate Arrow chunks instead of `as_materialized_series()` (LOW)
`col.as_materialized_series()` (`marshal.rs:365`) forces a copy when the column is
multi-chunk. Iterating chunks directly (or only materialising when chunk_count > 1)
avoids that copy. Lower priority — O1/O2 dominate.

## Notes / non-issues
- The dense-numpy and scipy CSR/CSC paths share the same `Vec<Vec<f64>>` row-major
  corpus, so O1/O2 benefit **all** Python ingest paths, not just polars/Arrow.
- `value_to_bin`'s per-value `is_nan()` + binary search (`bin_mapper.rs:148-173`)
  is inherent to histogram binning and parity-locked; not an Arrow concern.
- Any change here must preserve the f32→f64 widen discipline and the exact
  sampled-set / EFB ordering documented in `ingest.rs` — these are parity gates.

## Recommended next step
If pursued, scope a focused quick task: **"column-major ingest for the Python/Arrow
path"** covering O1+O2 together (shared change), with the existing
`bench_categorical_bin.rs` example extended to an A/B ingest benchmark and a
parity assertion (polars frame vs equivalent numpy array → identical model).
