# Phase 2: Dataset + Binning (determinism root) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 2-dataset-binning-determinism-root
**Areas discussed:** Bin storage representation, Binning parallelism vs determinism, Dataset crate boundary & API shape, Parity fixture strategy

---

## Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Bin storage representation | Model C++'s templated DenseBin/SparseBin value-widths in Rust | ✓ |
| Binning parallelism vs determinism | Single-threaded vs C++ OpenMP work-partitioning | ✓ |
| Dataset crate boundary & API shape | New crate + how much public API now | ✓ |
| Parity fixture strategy | What datasets + granularity prove bit-identical binning | ✓ |

**User's choice:** All four areas selected for discussion.

---

## Bin Storage Representation

| Option | Description | Selected |
|--------|-------------|----------|
| Generic + dyn Bin trait (mirror C++) | DenseBin<T: BinValue>/SparseBin<T> behind a `Bin` trait, Box<dyn Bin> per group, factory selects width from num_bins; 1:1 mirror of C++ template-storage + virtual-base | ✓ |
| Enum-of-typed-vecs dispatch | Single enum { U8/U16/U32/Packed4 } matched at access; one concrete type, no monomorphization, but per-element match in the hot loop | |
| Single u32 width for Phase 2, defer narrowing | Store all bins as Vec<u32>; defer width selection + 4-bit packing to Phase 4 since Phase 2 SC only checks integer indices | |

**User's choice:** Generic + dyn Bin trait (mirror C++).
**Notes:** Faithful C++ mirror preferred — width selected from num_bins (u8<256, u16<65536, else u32).

### 4-bit packing sub-decision

| Option | Description | Selected |
|--------|-------------|----------|
| Include 4-bit packing now | Implement DenseBin<u8, FOUR_BIT=true> two-bins-per-byte packing in Phase 2, golden-tested | ✓ |
| All widths, 4-bit deferred to Phase 4 | Ship u8/u16/u32 + Sparse now; ≤16-bin features use non-packed u8 path; true packing in Phase 4 | |

**User's choice:** Include 4-bit packing now.
**Notes:** Maximally faithful byte layout from the start; histogram phase inherits a complete, parity-proven storage layer.

---

## Binning Parallelism vs Determinism

| Option | Description | Selected |
|--------|-------------|----------|
| Single-threaded first (determinism root) | Straight sequential port matching num_threads=1 reference; parallelism is a later optimization | |
| Parallel from the start (match C++ OpenMP) | Reproduce C++'s OpenMP work-partitioning with rayon now | |
| Sequential core, parallel-ready seams | Sequential execution (deterministic, matches num_threads=1) but per-feature independent units so rayon drops in later without restructuring | ✓ |

**User's choice:** Sequential core, parallel-ready seams.
**Notes:** Determinism now, cheap parallelization later; no shared mutable accumulation across features.

---

## Dataset Crate Boundary & API Shape

| Option | Description | Selected |
|--------|-------------|----------|
| One lgbm-dataset crate | Single crate: BinMapper + Dense/Sparse store + encoding + EFB + metadata + ingestion (mirrors C++ src/io/) | ✓ |
| Split lgbm-bin + lgbm-dataset | Pure binning primitives in lgbm-bin; store/metadata/ingestion/EFB in lgbm-dataset | |
| Extend lgbm-core | Add dataset modules into existing lgbm-core | |

**User's choice:** One lgbm-dataset crate.

### API surface sub-decision

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal internal-facing constructors | Just enough for dense+CSR/CSC + metadata ingestion + parity tests; polished API in Phase 6 | ✓ |
| Fuller Rust-native Dataset builder now | Ergonomic builder in Phase 2; Phase 6 wires Booster onto it | |
| You decide | Let research/planning pick the thinnest API | |

**User's choice:** Minimal internal-facing constructors.
**Notes:** Smallest stable-surface commitment; Dataset::from_mat/from_csr/from_csc, immutable after finish-load.

---

## Parity Fixture Strategy

### Input corpus (multi-select)

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic randomized distributions | Master-seed-derived columns across varied distributions, sweeping max_bin/min_data_in_bin/sample_cnt | ✓ |
| Curated edge-case battery | NaN per MissingType, ±0.0, on-boundary, out-of-range categorical, all-missing, single-value, sparse | ✓ |
| LightGBM bundled example datasets | Real datasets under LightGBM/examples/ as realistic end-to-end fixtures | ✓ |
| Categorical + EFB-specific corpus | Multi-categorical with rare levels (DAT-04) + mutually-exclusive sparse sets (DAT-05) | ✓ |

**User's choice:** All four sources.

### Golden granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Boundaries + per-row assignment + mapper internals | Three layers: BinMapper internals, full per-row assignment vector, categorical maps + EFB layout | ✓ |
| Boundaries + per-row assignment only | bin_upper_bound_ + per-row indices, skip categorical/EFB snapshots | |
| Final bin-assignment matrix only | Just the final per-row×per-feature index matrix | |

**User's choice:** Boundaries + per-row assignment + mapper internals.
**Notes:** Maximally diagnostic — a mismatch localizes to the exact stage/feature, fitting a determinism-root phase.

---

## Claude's Discretion

- Exact fixture file formats/serialization.
- The precise `BinValue` trait bound set.
- Internal module layout within `lgbm-dataset`.
- Sparse-vs-dense selection threshold details.
- Precise category→bin folding/ordering implementation.

(All bounded by "faithful C++ mirror, exact integer bin-index parity"; C++ source is authoritative over inferred defaults.)

## Deferred Ideas

- Parallel (rayon) binning — later, separately-validated optimization producing byte-identical bins.
- Polished public Rust-native Dataset API — Phase 6 (API-01).
- Text-file / binary-cache / Arrow ingestion — v2 (ING-01/02/03).
- 4-bit packing perf tuning — downstream of the parity-correct packed path.
