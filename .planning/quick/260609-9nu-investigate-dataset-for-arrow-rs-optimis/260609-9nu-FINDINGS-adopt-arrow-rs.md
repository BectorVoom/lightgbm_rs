---
quick_id: 260609-9nu
title: Can the dataset path be optimised BY ADOPTING arrow-rs?
type: investigation
status: complete
date: 2026-06-08
note: Companion to 260609-9nu-FINDINGS.md (inverse question — adopt arrow-rs vs optimise existing arrow usage)
---

# Findings: would adopting `arrow-rs` optimise the dataset path?

This is the inverse of the first investigation. There, the question was "optimise our
arrow-rs usage" (answer: the dataset crate has none). Here: **would pulling in
`arrow-rs` (the `apache/arrow-rs` crate family) as a new dependency make the dataset
path faster?**

## Verdict (short)

- **Internal binned storage + histogram hot path → NO.** Already optimal *and*
  parity-locked. arrow-rs would add overhead and break the bit-exact contract.
- **Python/FFI ingest copies → marginal / not the right tool.** The real waste is
  fixable with plain column-major Rust (O1/O2 in the companion doc), and the tree
  already has a columnar Arrow source (`polars-arrow`). Adding `arrow-rs` as a
  *second* Arrow impl is duplicative.
- **Future file ingestion (Parquet / CSV / IPC) → YES, this is the one real case.**
  Currently v2-deferred (`ING-01/02/03`). When built, arrow-rs is the idiomatic,
  high-performance choice.

## Why NOT for internal storage/compute

1. **Bin storage is already an Arrow-shaped columnar store, and better-suited.**
   `DenseBin` (`crates/lgbm-dataset/src/bin/dense_bin.rs`) holds packed integer
   columns — `u8`/`u16`/`u32`, plus a **4-bit nibble-packed** mode (two bins per
   byte). Arrow's `UInt8Array` etc. are byte-per-element with a separate validity
   bitmap and offset metadata — *less* compact than the existing 4-bit packing, and
   they carry machinery (nulls, offsets) the bin store doesn't need.

2. **The byte layout is a hard parity gate.** `dense_bin.rs:5-21` documents that
   `data_` is "the EXACT memory the Phase 4 histogram kernels read against. Any
   deviation in packing breaks parity." Storage-layout golden tests
   (`tests/bin_storage_layout.rs`) assert the exact bytes vs the C++ reference.
   Swapping in arrow-rs arrays would change the memory image → fails the bit-exact
   CPU merge gate. No upside, direct contract violation.

3. **Binning arithmetic must stay scalar deterministic f64.** arrow-rs's value is
   partly its SIMD compute kernels (`arrow-arith`, `arrow-ord`). But binning is a
   custom histogram algorithm held to f64-fold bit-exactness vs C++; vectorised
   kernels reorder/parallelise float ops and cannot reproduce the reference
   sums. The hot loop (`bin_mapper.rs:148` `value_to_bin`) is parity-locked.

4. **Null model mismatch.** The project uses NaN-as-missing (parity semantics).
   arrow-rs uses validity bitmaps. Routing through Arrow would just reintroduce the
   bitmap→NaN translation the Python layer already does (`marshal.rs:270`).

## Why it's NOT the right lever for the ingest copies either

The genuine ingest waste (companion doc O1/O2: a column→row→column double transpose
over nested `Vec<Vec<f64>>`) is a *layout* problem, fixed by feeding the
already-columnar data straight into column-at-a-time binning. You do **not** need
arrow-rs for that — plain `Vec<f64>` columns (or one flat strided buffer) do it.

And critically: the tree already depends on **`polars-arrow`** (via `pyo3-polars`),
not `arrow-rs` (confirmed in `Cargo.lock`: `polars-arrow` present, no `arrow`/
`arrow-array`). polars-arrow already exposes contiguous typed chunk buffers
(`ChunkedArray::downcast_iter()` → `&[f64]`), which is the same zero-copy access
arrow-rs gives via `array.as_primitive::<Float64Type>().values()`. Adopting
arrow-rs *in addition* means two Arrow runtimes in the build graph for no new
capability on the polars path. The only scenario where arrow-rs would specifically
win at the FFI boundary is if the project **dropped polars** and imported
pandas/pyarrow tables directly over the Arrow C Data Interface — a much larger
change, not an optimisation.

## The one place arrow-rs is genuinely additive: file ingestion (v2)

`ING-01/02/03` (text-file / binary-cache / Arrow ingestion) are **deferred to v2**
(`STATE.md:521`). When that work is picked up, arrow-rs is a strong fit:

- **`parquet`** (in the arrow-rs family) is the de-facto Rust Parquet reader; it
  yields columnar `RecordBatch`es that map *directly* onto the dataset's
  column-at-a-time binning — no transpose, contiguous typed buffers, projection
  pushdown to read only needed columns.
- **`arrow-csv`** gives a fast typed CSV reader (LightGBM's text path).
- **`arrow-ipc` / C Data Interface** gives zero-copy interchange with the broader
  Arrow ecosystem.

This is *new functionality*, not an optimisation of the current path — so it belongs
in the v2 ingestion design, where "arrow-rs vs polars readers" should be the
explicit decision (likely polars for consistency, since it's already a dependency;
arrow-rs only if a polars-free core ingest is wanted).

## Recommendation

1. **Do not adopt arrow-rs to optimise the existing dataset storage or binning** —
   no win, breaks the bit-exact parity gate.
2. **For the ingest-copy waste, pursue O1/O2 from the companion doc** (column-major
   ingest + flat buffer) using plain Rust + the columnar buffers polars-arrow
   already provides. arrow-rs adds nothing there.
3. **Earmark arrow-rs (specifically `parquet` + `arrow-csv`) for the v2 `ING-*`
   file-ingestion work** — that's the only place it's a real performance lever, and
   it's additive rather than parity-risking.
