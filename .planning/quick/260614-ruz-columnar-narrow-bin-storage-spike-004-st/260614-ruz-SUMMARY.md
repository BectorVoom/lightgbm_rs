---
phase: quick-260614-ruz
plan: 01
subsystem: treelearner / compute (histogram build hot path)
tags: [performance, cpu, histogram, cache, bin-width, columnar, spike-004]
requires: [spike-003, quick-260614-r4o]
provides: [BinColumn, columnar-narrow-bins]
affects: [lgbm-compute, lgbm-treelearner, lgbm-boosting, lgbm]
tech-stack:
  added: []
  patterns: [enum-narrow-storage, monomorphic-per-width-fold, widening-accessor]
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-treelearner/src/data_partition.rs
    - crates/lgbm-treelearner/src/lib.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm/src/booster.rs
    - crates/oracle-harness/tests/learner_parity.rs
    - crates/oracle-harness/tests/advanced_parity.rs
decisions:
  - "BinColumn lives in lgbm-compute (lowest crate, owns Backend+fold) — no dependency cycle"
  - "Single &[&BinColumn] trait seam for build_leaf_histograms_raw (CPU default + Rocm override share it)"
  - "Gate hot per-row reads behind monomorphic per-width scans (first_ge), never the boxed iter_u32"
  - "Skip the per-tree u32 widening on CpuBackend via wants_resident_bins() (no-op upload)"
metrics:
  duration: ~1.5h
  completed: 2026-06-15
  commits: 2
  files_modified: 8
  lines: "+431 / -74"
---

# Phase quick-260614-ruz: Columnar Narrow Bin Storage (Spike 004) Summary

**One-liner:** Stored each `FeatureColumn`'s bins in the narrowest unsigned type for
its `num_bin` (a `BinColumn { U8/U16/U32 }` enum in `lgbm-compute`), with a
monomorphic per-width CPU histogram fold — bit-exact, and **−49% large-row train /
−53% build** from L2 cache density (faithful to C++ `DenseBin<uint8_t>`), with **no
small-row regression**.

## What shipped

- **`BinColumn { U8(Vec<u8>), U16(Vec<u16>), U32(Vec<u32>) }`** defined in
  `lgbm-compute` (the lowest crate, which owns the `Backend` trait + the hot fold —
  putting it in `lgbm-treelearner` would be a dependency cycle), re-exported from
  `lgbm-treelearner`. Methods: `new(Vec<u32>, num_bin)` (width by `num_bin`: u8 ≤256,
  u16 ≤65536, else u32), `bin(row)->u32` (widening), `len`/`is_empty`,
  `gather(&[u32])` (re-narrows preserving width), `to_u32_vec`, `iter_u32` (cold,
  boxed), and **`first_ge(bound)`** (the allocation-free monomorphic per-width scan
  for the bin-range gate). `#[derive(Clone, Debug, PartialEq)]`.
- **`FeatureColumn.bins: Vec<u32>` → `BinColumn`**; `Default` = `BinColumn::U32(empty)`.
- **CPU `build_leaf_histograms_raw`**: the single `&[&BinColumn]` trait seam (CPU
  default + Rocm override share one signature). The CPU fold dispatches on the column
  width **once per feature** (a `match` OUTSIDE the row loop), so each arm is a
  monomorphic tight loop reading the narrow element directly — the cache-density win.
  Fold order + `bin*2(+1)` index arithmetic are byte-identical across arms ⇒ bit-exact.
- **Rocm override** widens to u32 internally (defensive fallback) / uses its existing
  resident u32 upload path → GPU kernel input byte-unchanged.
- **Blanket cold-reader migration** to the widening accessor: bagging subset gather
  (`gather`, preserves width), once-per-train bin-range gate (`first_ge`, VALUE check
  preserved verbatim), GPU upload (`to_u32_vec`, gated, cold), `data_partition`
  `split`/`split_categorical` (`&BinColumn` + `.bin`), `gbdt.rs` DART/RF/predict
  scatter (`.bin`, lines 761/1177/1308), and every test ctor (`BinColumn::new`).
  **Zero residual `.bins[..]` index reads; zero `Vec<u32>`-typed `.bins`.**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Perf bug] Small-row regression from the boxed `iter_u32` gate + per-tree no-op u32 widening**
- **Found during:** Task 3 (the interleaved A/B measurement — the standing
  measure-before-shipping lesson did its job).
- **Issue:** The first BinColumn cut regressed small train ~7% (consistent across 5
  rounds, outside the noise band) even though the per-width FOLD itself was flat.
  Two narrow-storage hot-path costs that the fold did not have: (a) the once-per-train
  bin-range gate read via `BinColumn::iter_u32()`, which **boxes** the per-variant
  iterator (`Box<dyn Iterator>` ⇒ heap alloc + dynamic dispatch over every bin every
  tree — 2.4M boxed reads/train at small); (b) the CpuBackend **widened every column
  to u32 per tree** for a no-op `upload_resident_bins`.
- **Fix:** (a) Added `BinColumn::first_ge(num_bin)` — a monomorphic per-width slice
  scan that returns the first offending VALUE (so the `BinIndexOutOfRange` rejection
  still carries the exact index); the gate uses it instead of the boxed iterator.
  (b) Added `Backend::wants_resident_bins()` (default `false`/CpuBackend skips the
  widening; RocmBackend `true` so its resident u32 upload is byte-unchanged).
- **Files modified:** `crates/lgbm-compute/src/lib.rs`, `crates/lgbm-treelearner/src/learner.rs`
- **Commit:** `67c7c7c`
- **Result:** small regression eliminated (within noise); large win grew (the gate runs
  O(rows) per tree at large too, so removing the boxed iterator helped both scales).

**2. [Rule 1 - Plan-vs-behavior reconciliation] `BinColumn::new` debug_assert relaxed to fits-type, not `bin < num_bin`**
- **Found during:** Task 1 (the plan's stated behavior `new(vec![0,300], 300).bin(1)==300`
  feeds a value equal to `num_bin`).
- **Issue:** A `debug_assert!(b < num_bin)` in `new` (the plan's literal suggestion)
  fired on the plan's own widening-test inputs.
- **Fix:** The assert guards the truncation / memory-safety concern (`b` FITS the chosen
  narrow type) — which is the real T-ruz-01 mitigation at the cast — NOT `bin < num_bin`.
  The authoritative `bin < num_bin` VALUE check is the once-per-train gate (preserved
  verbatim, now via `first_ge`). This keeps the threat register intact: the gate runs
  upstream of any tree growth, and `new` is allowed an edge value equal to `num_bin`.
- **Files modified:** `crates/lgbm-compute/src/lib.rs`

## Interleaved A/B measurement (the gate)

Baseline = HEAD `34095b0` (pre-change), built as a separate release binary; POST = the
shipped change. Interleaved (alternated POST/BASE each round to cancel thermal drift),
`LGBM_PHASE_PROF=1`. **Bench corpora bin widths: small bins=32 → U8; large bins=64 → U8**
(u8 carries the win at the default `max_bin`; u16/u32 are exercised by the unit tests).

### SMALL — `BENCH_SIZES=small:2000:12:32 BENCH_ITERS=100 BENCH_REPS=9` (must NOT regress)

| round | POST train_median | BASE train_median |
|------:|------------------:|------------------:|
| 1 | 26.22 ms | 26.31 ms |
| 2 | 30.96 ms | 30.69 ms (paired thermal spike) |
| 3 | 26.36 ms | 26.12 ms |
| 4 | 31.98 ms | 29.21 ms (paired thermal spike) |
| 5 | 26.30 ms | 26.15 ms |

POST median ≈ **26.3 ms** vs BASE ≈ **26.2 ms** — **within the ±2-3% noise band, NO
regression.** phase_prof BUILD is flat (~92 µs/iter both): a 2k-row u8 column already
fits L1, so narrowing cannot help and its cost is negligible (exactly the spike-004
small-row prediction).

### LARGE — `BENCH_SIZES=large:200000:32:64 BENCH_ITERS=50 BENCH_REPS=5` (must improve)

| round | metric | POST | BASE | delta |
|------:|--------|-----:|-----:|------:|
| 1 | train_median | **1.39 s** | 2.74 s | **−49%** |
| 1 | BUILD phase (cum µs) | 4448.1 | 9753.6 | **−54%** |
| 2 | train_median | **1.41 s** | 2.76 s | **−49%** |
| 2 | BUILD phase (cum µs) | 4580.9 | 9750.5 | **−53%** |

The histogram BUILD phase — ~90% of large train at HEAD — **more than halved** from L2
cache density (the u8 column at 64 bins fits L2; the u32 column overflowed to L3). This
is the spike-004 lever, realized end-to-end and bit-exactly. Build's share of train
drops from 89.4% (BASE) to 81.4% (POST).

## Bit-exact HARD GATE (fold order frozen) — GREEN

- `cargo test -p lgbm-compute` — **green** (21 prior + 8 new BinColumn tests, incl.
  width-selection boundaries 256→U8 / 257→U16 / 65536→U16 / 65537→U32, width-by-num_bin
  not observed-max, widening `bin()` per variant, `gather` preserves width, `to_u32_vec`
  round-trip, `iter_u32`, `first_ge`).
- `cargo test -p lgbm-treelearner` — **66/0** (incl. the relocated out-of-range
  rejection test, now via `first_ge`).
- `cargo test -p lgbm-boosting` — **55/0**.
- `cargo test -p oracle-harness --test learner_parity` — **29 passed / 0 failed
  BIT-EXACT** (the merge gate).
- `cargo test -p oracle-harness` (full) — **0 failed** across every binary
  (advanced 5, boosting 75, kernel 6, learner 29, metric 15, predict 5, rank 4,
  raw_bin_train 2, rng 1, comparator 5, config_drift 3, lib 3). No DEF-07-02-class
  regression; no new failures.
- clippy clean on edited code (the only remaining `too_many_arguments (8/7)` warning is
  pre-existing on `find_best_splits_batched`, confirmed present at HEAD `34095b0`).
- `LightGBM/`, `target/`, `.venv/`, `.serena/`, `cuml-main/` never git-added.

## Out of scope (untouched, as planned)

GPU/ROCm/cubecl kernels (RocmBackend stays on u32 internally, widens at upload), split
scan, partition LOGIC (only its per-row bin READ uses the accessor), feature-parallelism.

## Commits

- `0d8bdd5` — feat: columnar narrow bin storage (BinColumn u8/u16/u32) + full blanket
  migration + monomorphic per-width CPU fold (Tasks 1+2; committed atomically — the
  build only compiles with the type, migration, and fold seam together).
- `67c7c7c` — perf: kill the small-row regression (monomorphic `first_ge` gate +
  `wants_resident_bins`-gated upload) — the Task-3 A/B-driven fix.

## Self-Check: PASSED

- `crates/lgbm-compute/src/lib.rs` — FOUND (contains `pub enum BinColumn`).
- Commit `0d8bdd5` — FOUND in git log.
- Commit `67c7c7c` — FOUND in git log.
- Zero residual `.bins[..]` indexing / `Vec<u32>`-typed `.bins` — VERIFIED by grep.
- learner_parity 29/0 BIT-EXACT — VERIFIED.
