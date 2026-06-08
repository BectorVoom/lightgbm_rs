---
quick_id: 260609-9nu
title: Implement O3+O5 — polars ingest fast paths (skip redundant cast + chunk-wise extract)
status: complete
date: 2026-06-08
type: implementation
---

# Summary — O3 + O5 (polars ingest fast paths)

Single-file change in `crates/lgbm-python/src/marshal.rs::series_to_f64_vec` (the
polars/Arrow column → `Vec<f64>` extractor). Does NOT touch the core `lgbm` crate or
its bit-exact path. Byte-identical output to the prior
`cast().f64().into_iter().map(unwrap_or(NAN)).collect()`.

## What changed
- **O3 — skip the redundant `cast(Float64)`.** The old code unconditionally allocated
  a casted `Series` even for already-`Float64` columns (the common numeric case). Now
  borrows via `Cow::Borrowed` when `dtype == Float64`; only allocates a casted Series
  for other numeric dtypes. Removes one full `Series` allocation per f64 column.
- **O5 — chunk-wise extraction (`downcast_iter`), no rechunk/materialize.** Builds the
  `Vec<f64>` by walking the Arrow chunks: a chunk with no validity bitmap is copied
  from its contiguous values slice (`extend_from_slice`); a chunk with nulls falls back
  to the per-value `null -> NaN` map. Avoids any forced single-buffer rechunk.

## Honest scope note (O4 overlap)
O5's only real win over the existing `into_iter().collect()` is the no-null contiguous
slice copy — `ChunkedArray::into_iter()` already walks chunks without rechunking, and
`as_materialized_series()` is free for Series-backed columns, so "iterate chunks" alone
would be a no-op. The contiguous-copy step therefore necessarily overlaps the O4 no-null
bulk-copy idea (which was nominally out of scope). It is applied ONLY per-chunk when
that chunk has no validity bitmap, so present values (incl. present-NaN bit patterns)
are preserved exactly; null-bearing chunks use the unchanged per-value path. No
behavioural change, so parity is intact.

## Verification
- `cargo build -p lgbm-python` ✓ (used inherent `PrimitiveArray::validity()` —
  `polars_arrow` is not a direct dep, so no trait import needed).
- Python suite (debug `maturin develop`): **55 passed, 3 skipped** — incl.
  `test_polars_input`, `test_numpy_sparse_parity`, `test_booster_parity`,
  `test_smoke`, `test_sklearn_parity` (A/B vs lightgbm 4.6).
- Core bit-exact gate unaffected (no core file changed).

## Not done: O4 as a standalone whole-array `cont_slice` memcpy guard (subsumed at chunk
granularity by O5 above). No wall-clock benchmark — allocation/branch reduction on the
polars ingest path, not a measured speedup claim.
