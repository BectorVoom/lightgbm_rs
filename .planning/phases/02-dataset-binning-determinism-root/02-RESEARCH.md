# Phase 2: Dataset + Binning (determinism root) - Research

**Researched:** 2026-06-05
**Domain:** Deterministic feature binning + immutable columnar dataset (faithful C++ LightGBM port, exact integer bin-index parity)
**Confidence:** HIGH (the entire spec is the in-repo C++ source, read directly this session; no external-library uncertainty)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Faithful 1:1 mirror of C++ value-width-templated storage: a `Bin` trait with generic `DenseBin<T: BinValue>` and `SparseBin<T>`, stored as `Box<dyn Bin>` (or equivalent) per feature group, with a factory selecting value width (u8/u16/u32) from `num_bins` exactly as C++ does (uint8 <256, uint16 <65536, else uint32). Zero-cost element access in the hot path; byte layout is the one Phase 4/5 histogram path reads against. (Chosen over enum-of-typed-vecs or single-u32-deferred representation.)
- **D-02:** Include the 4-bit packed DenseBin variant now (`IS_4BIT` path, `num_bins ≤ 16`): two bins per byte with exact C++ packing, golden-tested in Phase 2.
- **D-03:** Sequential single-threaded core matching pinned `num_threads=1` C++, structured into per-feature independent units so rayon can be dropped in later as a separately-validated optimization producing byte-identical bins. Binning is CPU pre-processing, off the CubeCL backend.
- **D-04:** One new `lgbm-dataset` crate holds the entire subsystem (BinMapper, Bin trait + Dense/Sparse store, MissingType/categorical encoding, EFB/MultiValBin, metadata, ingestion). Mirrors C++ `src/io/` cohesion. Consistent with Phase 1 D-09 ("add a crate per subsystem") and `lgbm-*` naming (D-08).
- **D-05:** Minimal internal-facing ingestion API only — `Dataset::from_mat` / `from_csr` / `from_csc` returning a store immutable after finish-load. No ergonomic builder, no sklearn surface; polished public API is Phase 6 (API-01).
- **D-06:** Four-source parity corpus, all under the D-14 randomized-at-capture / committed-master-seed discipline: (1) synthetic randomized distributions sweeping `max_bin`/`min_data_in_bin`/`bin_construct_sample_cnt`/`data_random_seed`; (2) curated edge-case battery (NaN per MissingType, +0.0/-0.0, on-boundary, out-of-range categorical, all-missing, single-value, sparse/zero-heavy); (3) LightGBM bundled example datasets under `LightGBM/examples/`; (4) categorical + EFB-specific corpus (rare levels, mutually-exclusive sparse feature sets).
- **D-07:** Per-stage golden granularity = three layers: (1) BinMapper internals — `bin_upper_bound_` array, `bin_type`, `missing_type`, `default_bin`, `most_freq_bin`, `num_bin`; (2) full per-row bin-index assignment vector per feature; (3) categorical category→bin maps + EFB bundle/offset layout. Reuses Phase 1's committed-fixtures + idempotent-regen pattern.

### Claude's Discretion
- Exact fixture file formats/serialization, precise `BinValue` trait bound set, internal module layout within `lgbm-dataset`, sparse-vs-dense selection threshold details, precise category→bin low-frequency folding/ordering implementation — all bounded by "faithful C++ mirror, exact integer bin-index parity." When C++ behavior is the spec, the C++ source is authoritative over any inferred default.

### Deferred Ideas (OUT OF SCOPE)
- Parallel (rayon) binning — ships sequential; rayon is later, separately-validated, must produce byte-identical bins.
- Polished public Rust-native `Dataset` API — Phase 6 (API-01).
- Text-file / binary-cache / Arrow ingestion — v2 (ING-01/02/03).
- 4-bit packing perf tuning — byte-layout parity now, perf tuning downstream.
- Histogram construction, split finding, tree learning (Phase 5); prediction / model text I/O (Phase 3); CubeCL kernels (Phase 4).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DAT-01 | `BinMapper` continuous→bin mapping (`FindBin`) producing bit-identical bin boundaries vs C++ (`max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, `data_random_seed`) | Full `FindBin` + `GreedyFindBin` + `FindBinWithZeroAsOneBin` + `FindBinWithPredefinedBin` algorithms transcribed below (Code Examples §1–4). `ValueToBin` literal `(r+l-1)/2`+`<=` search (§5). Sampling via `Random(data_random_seed).Sample(total_nrow, sample_cnt)` — reuses Phase 1 `lgbm_core::Random`. |
| DAT-02 | Binned columnar store (DenseBin + SparseBin) immutable after finish-load | `Bin` trait + `DenseBin<VAL_T,IS_4BIT>` (incl. 4-bit packing) + `SparseBin<VAL_T>` delta-encoding layouts transcribed (§6–8). `FeatureGroup` bin-offset packing + `CreateBinData` sparse/dense selection (§9–10). Immutability = `FinishLoad()` boundary (Architecture). |
| DAT-03 | Missing-value handling (`use_missing`, `zero_as_missing`, `MissingType`) with C++-matching default-direction routing | `FindBin` missing-type derivation (§1 lines 322–333) + NaN-bin append for `MissingType::NaN` + `ValueToBin` NaN routing (§5). `MissingType` enum {None, Zero, NaN}. |
| DAT-04 | Categorical encoding (category→bin, low-frequency folding) | Categorical branch of `FindBin` (§2): int conversion, descending count sort (`SortForPair`), 99% cut-off (`RoundInt(...*0.99)`), `min_data_in_bin` folding, NaN dummy bin 0. `ValueToBin` categorical path (§5). |
| DAT-05 | Exclusive Feature Bundling (`enable_bundle`) reproducing C++ feature grouping | `FastFeatureBundling` + `FindGroups` + `GetConflictCount` + `FixSampleIndices` (§11) — two-pass grouping, RNG-driven group search & shuffle (`Random(num_data)`, `NextShort`), dense/sparse split. `MultiValBin` dense/sparse layout + offsets. |
| DAT-06 | Metadata (labels, weights, init_score, query/group boundaries) | `Metadata` struct fields + `FinishLoad`/`CalculateQueryWeights` (Architecture §Metadata). f32 `label_t`/`weights_`, `double init_score_`, `int32 query_boundaries_`. |
| DAT-07 | In-memory dense + CSR/CSC ingestion via Rust API | C-API ingestion path (`CreateSampleIndices` → per-column sample → `FindBin` → `Construct` → row-by-row `PushData` → `FinishLoad`) transcribed as the model for `from_mat`/`from_csr`/`from_csc` (Architecture §Data Flow). |
| ORA-03 | Per-stage parity tests (bin granularity) | Three-layer golden granularity (D-07) maps to the Validation Architecture section; localizes divergence to binning before histograms exist (SC#5). |
</phase_requirements>

## Summary

Phase 2's entire specification is the in-repo C++ source under `LightGBM/` — read directly this session. There is **no external-library research dimension** and almost no training-knowledge dependence: the deliverable is a line-faithful Rust transcription of `BinMapper::FindBin` (+ its three helpers), `BinMapper::ValueToBin`, the `DenseBin`/`SparseBin` storage layouts, the `FeatureGroup` bin-offset packing, and the `FastFeatureBundling`/`FindGroups` EFB pipeline. Every numeric decision (bin boundary, bin index, group assignment) is fully determined by code paths captured below.

The single highest-risk area is **floating-point boundary determinism**: bin boundaries are computed in `double` (f64) and rounded with `std::nextafter(a, INFINITY)` via `Common::GetDoubleUpperBound`, and de-duplicated with `CheckDoubleEqualOrdered` (`b <= nextafter(a, INF)`). Rust must use `f64::next_up()` (stable since Rust 1.86; the workspace pins rust 1.95) or `libm`-equivalent `nextafter` semantics — getting this wrong silently shifts boundaries by 1 ULP and breaks parity. Sampling for `FindBin` and all EFB group decisions route through the **already-ported Phase 1 `lgbm_core::Random`** — RNG parity is a precondition, not new work.

The second-highest risk is **fixture capture feasibility**: unlike Phase 1's header-only `Random` capture, `BinMapper::FindBin` lives in `src/io/bin.cpp` and transitively includes `common.h` → `fast_double_parser.h` + `fmt/format.h` from `external_libs/`. **Verified this session:** those external_libs ARE physically present on disk (`external_libs/{fast_double_parser,fmt,eigen,compute}/`), just git-untracked — so a focused C++ capture harness compiling `bin.cpp` + the `Dataset`/`FeatureGroup`/EFB sources is buildable without vendoring. The capture program emits the three golden layers; normal `cargo test` replays committed fixtures (D-06, no C++ toolchain).

**Primary recommendation:** Build `lgbm-dataset` as a direct transcription in vertical slices — start with `BinMapper` numeric `FindBin` + `ValueToBin` (the determinism kernel, golden layer 1+2 for one numeric feature), then DenseBin storage (incl. 4-bit), then SparseBin, then categorical, then metadata + ingestion, then EFB last. Mirror C++ control flow line-for-line; resist idiomatic refactors that could reorder FP operations or change tie-breaks.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Bin boundary computation (`FindBin`) | CPU pre-processing (`lgbm-dataset`) | — | Pure CPU sequential pass; off the CubeCL backend per D-03. f64 math, deterministic. |
| Value→bin mapping (`ValueToBin`) | CPU pre-processing (`lgbm-dataset`) | Phase 4/5 read this for predict/histogram | Hot path; called per row at ingestion and per row at predict. Zero-cost. |
| Columnar bin storage (Dense/Sparse) | CPU memory (`lgbm-dataset`) | Phase 4 GPU reads byte layout | Byte layout is the contract Phase 4 histogram kernels and Phase 5 split reads against. |
| Sampling for bin construction | CPU (`lgbm-core::Random`) | — | Routes through ported Phase 1 LCG; `data_random_seed`-seeded. RNG-parity critical. |
| EFB feature grouping | CPU (`lgbm-dataset`) | `lgbm-core::Random` for group search/shuffle | Sequential greedy; RNG-driven. Produces feature→group→offset layout Phase 4/5 inherit. |
| Metadata (labels/weights/query) | CPU memory (`lgbm-dataset`) | Phase 6 objectives/metrics consume | Plain owned vectors; query-weight derivation at FinishLoad. |
| Ingestion (dense/CSR/CSC) | `lgbm-dataset` public-internal API | `lgbm-core` types/errors | Minimal constructors (D-05); the only externally-callable surface this phase. |

## Standard Stack

This phase introduces **no new third-party runtime dependencies.** It is a pure-Rust transcription depending only on the existing workspace crate `lgbm-core` and the Rust standard library. The "stack" is therefore the C++ reference source plus the existing workspace.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `lgbm-core` (workspace) | path dep | `Random` LCG, `Config`, f32 types, `thiserror` errors | Phase 1 deliverable; the dependency root. `[VERIFIED: crates/lgbm-core/src/ read this session]` |
| Rust `std` | edition 2024 / rustc 1.95 | `f64::next_up()`, `Vec`, sort, `unordered_map`→`HashMap`/`BTreeMap` | `f64::next_up`/`next_down` stabilized in Rust 1.86 (`< 1.95` pin) — exactly mirrors C++ `std::nextafter(a, +INF)`. `[VERIFIED: Cargo.toml rust-version = "1.95"]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `thiserror` | 2.0.18 (workspace) | Domain errors at `lgbm-dataset` boundary (FND-04 pattern) | Ingestion validation errors (shape mismatch, bad CSR, etc.) |
| `anyhow` | 1.0.102 (workspace) | Ergonomic propagation in tests/xtask harness | Fixture regen + parity test plumbing only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `f64::next_up()` (std) | `libm::nextafter` crate | std method is exact-equivalent, zero-dep, stable on the pinned toolchain — no reason to add a crate. Only fall back to `libm` if a `next_up` edge case (NaN/subnormal) diverges from C++ `nextafter` in testing (it should not for finite positive args). |
| `std::collections::HashMap` for `categorical_2_bin_` | `BTreeMap` / `FxHashMap` | C++ uses `std::unordered_map<int,unsigned int>` but **iteration order never affects output** — the bin assignment order is set by the descending-count sort, not map iteration. A plain `HashMap` is parity-safe; `BTreeMap` is fine too. (Contrast Phase 1's alias-collision determinism issue — that does NOT recur here because folding order is sort-driven, not map-driven.) |
| `Box<dyn Bin>` trait objects | enum dispatch (`enum BinStore { U8(..), U16(..) }`) | D-01 locked `Box<dyn Bin>` for faithful C++ virtual-dispatch mirror. Enum dispatch would be marginally faster but breaks the 1:1 mapping and the factory pattern. Honor D-01. |

**Installation:** No new crates. Add to root `Cargo.toml` `members`:
```toml
members = [
    "crates/lgbm-core",
    "crates/lgbm-compute",
    "crates/lgbm-dataset",   # NEW
    "crates/oracle-harness",
    "xtask",
]
```
`crates/lgbm-dataset/Cargo.toml` deps: `lgbm-core = { path = "../lgbm-core" }`, `thiserror.workspace = true`.

## Package Legitimacy Audit

**Not applicable — this phase installs zero external packages.** `lgbm-dataset` depends only on the in-workspace `lgbm-core` (path dependency, already audited in Phase 1) and `thiserror`/`anyhow` (already workspace dependencies, in `Cargo.lock` since Phase 1). slopcheck/registry verification is moot: there are no new registry packages to verify.

| Package | Registry | Disposition |
|---------|----------|-------------|
| `lgbm-core` | path (workspace) | Approved — Phase 1 deliverable |
| `thiserror` 2.0.18 | crates.io (already locked) | Approved — Phase 1 dependency, unchanged |
| `anyhow` 1.0.102 | crates.io (already locked) | Approved — Phase 1 dependency, unchanged |

**Packages removed due to slopcheck [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                         INGESTION (D-05 minimal API)
  from_mat(dense) ─┐
  from_csr ────────┼──► raw columns (per-feature value arrays)
  from_csc ────────┘            │
                                ▼
              ┌─────────────────────────────────────┐
              │  SAMPLE for bin construction          │
              │  Random(data_random_seed)             │   ← lgbm-core::Random (Phase 1)
              │  .Sample(num_rows, bin_construct_     │
              │           sample_cnt)                 │
              └─────────────────────────────────────┘
                                │ sampled values per column
                                ▼
        ┌──────────────────── per feature (sequential, D-03) ──────────────────┐
        │  BinMapper::FindBin(values, max_bin, min_data_in_bin, bin_type,        │
        │                     use_missing, zero_as_missing, forced_bounds)       │
        │   ├─ numeric  → FindBinWithZeroAsOneBin / FindBinWithPredefinedBin     │
        │   │              → GreedyFindBin  (f64 + nextafter rounding)           │
        │   │              → bin_upper_bound_[], num_bin_, missing_type_,         │
        │   │                default_bin_, most_freq_bin_                         │   ◄── GOLDEN LAYER 1
        │   └─ categorical → descending-count fold, 99% cut, categorical_2_bin_  │
        └───────────────────────────────────────────────────────────────────────┘
                                │ vector<BinMapper>
                                ▼
              ┌─────────────────────────────────────┐
              │  Dataset::Construct                   │
              │   OneFeaturePerGroup (default) OR     │
              │   FastFeatureBundling (enable_bundle) │   ← Random(num_data) group search+shuffle
              │     → FindGroups → GetConflictCount   │   ◄── GOLDEN LAYER 3 (groups/offsets)
              │   build FeatureGroup[]:               │
              │     bin_offsets_[], num_total_bin_,   │
              │     CreateBinData (sparse vs dense)   │
              └─────────────────────────────────────┘
                                │
                                ▼  for each row, each feature:
              ┌─────────────────────────────────────┐
              │  FeatureGroup::PushData(sub_fidx,     │
              │      row, value)                      │
              │   bin = ValueToBin(value)             │   ◄── GOLDEN LAYER 2 (per-row bin index)
              │   if bin==most_freq_bin: skip         │      ValueToBin: (r+l-1)/2 + <= search
              │   if most_freq_bin==0: bin -= 1       │
              │   single-val: bin += bin_offsets_;    │
              │     bin_data_->Push(row, bin)          │
              │   multi-val: multi_bin_data_[f]->      │
              │     Push(row, bin+1)                   │
              └─────────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │  Dataset::FinishLoad()                │   ◄── IMMUTABILITY BOUNDARY
              │   FeatureGroup::FinishLoad (per group)│       (DenseBin 4-bit merge buf_→data_;
              │     Bin::FinishLoad                    │        SparseBin sort+delta-encode+fast_index)
              │   Metadata::FinishLoad                 │
              └─────────────────────────────────────┘
```

File-to-implementation mapping is in the Component Responsibilities table (Don't Hand-Roll / below), not the diagram.

### Recommended Project Structure
Mirror C++ `src/io/` cohesion (D-04 single crate). Suggested module layout (Claude's discretion — bounded by faithful mirror):
```
crates/lgbm-dataset/
├── src/
│   ├── lib.rs                # re-exports, crate-level error type
│   ├── error.rs              # thiserror DatasetError (shape, bad-csr, etc.)
│   ├── bin_mapper.rs         # BinMapper: FindBin + helpers + ValueToBin   (DAT-01, DAT-03, DAT-04)
│   ├── bin/
│   │   ├── mod.rs            # Bin trait + BinValue trait + CreateDenseBin/CreateSparseBin factories (D-01)
│   │   ├── dense_bin.rs      # DenseBin<T, const IS_4BIT: bool>           (DAT-02, D-02)
│   │   └── sparse_bin.rs     # SparseBin<T>: delta + fast_index           (DAT-02)
│   ├── multi_val_bin.rs      # MultiValBin dense/sparse (EFB store)        (DAT-05)
│   ├── feature_group.rs      # FeatureGroup: bin_offsets_, PushData, CreateBinData (DAT-02, DAT-05)
│   ├── efb.rs                # FastFeatureBundling, FindGroups, GetConflictCount (DAT-05)
│   ├── metadata.rs           # Metadata: label/weights/init_score/query   (DAT-06)
│   ├── dataset.rs            # Dataset::Construct + FinishLoad + immutability (DAT-02)
│   └── ingest.rs             # from_mat / from_csr / from_csc + sampling   (DAT-07)
└── tests/                    # parity tests vs committed goldens (ORA-03)
```

### Pattern 1: f64 boundary math with `next_up` (the determinism kernel)
**What:** All bin boundaries are `double`, midpoints are `(a+b)/2.0`, and each candidate boundary is pushed through `Common::GetDoubleUpperBound(x) = std::nextafter(x, INFINITY)`. De-duplication uses `CheckDoubleEqualOrdered(back, val) == (val <= nextafter(back, INFINITY))`.
**When to use:** Everywhere `bin_upper_bound_` is built (`GreedyFindBin`, `FindBinWithZeroAsOneBin`, `FindBinWithPredefinedBin`).
**Example:**
```rust
// Source: LightGBM/include/LightGBM/utils/common.h:845-852
#[inline] fn get_double_upper_bound(a: f64) -> f64 { a.next_up() }      // std, Rust >=1.86
#[inline] fn check_double_equal_ordered(a: f64, b: f64) -> bool { b <= a.next_up() }
// de-dup guard, mirrors C++ exactly:
//   if bin_upper_bound.empty() || !CheckDoubleEqualOrdered(bin_upper_bound.back(), val) { push(val) }
```
**Note:** C++ does midpoint as `(distinct_values[i] + distinct_values[i+1]) / 2.0` in f64 (the sample values arrive as `double` even though the feature data is f32 — the sampled values are widened to `double` at ingestion). Keep intermediate math in `f64`; do NOT compute midpoints in f32.

### Pattern 2: most_freq_bin / default_bin and the "bin 0" offset trick
**What:** `default_bin_ = ValueToBin(0)`. `most_freq_bin_ = argmax(cnt_in_bin)` but collapses to `default_bin_` unless the feature is sparse enough (`max_sparse_rate >= kSparseThreshold = 0.7`). When `most_freq_bin_ == 0`, the feature group drops one bin (the "store most_freq in bin 0" optimization) and `PushData` subtracts 1. This permeates `FeatureGroup` offset arithmetic and the multi-val `+1` push.
**When to use:** `FindBin` tail (§1 lines 490–505), `FeatureGroup` constructor offset loop, `PushData`.
**Example:** see Code Examples §9 (offset packing) and §10 (PushData).

### Pattern 3: Sequential per-feature seams (D-03)
**What:** Each feature's `FindBin` and each feature's bin push are independent — no shared mutable accumulator across features. C++ uses `#pragma omp parallel for ... schedule(guided)` over features in `ConstructBinMappersFromTextData`, but with `num_threads=1` it is a plain sequential loop. Structure the Rust as `for fidx in 0..num_features { find_bin(...) }` so a later `par_iter` is a one-line swap.
**When to use:** bin-mapper construction loop, push loop.

### Anti-Patterns to Avoid
- **Idiomatic binary search in `ValueToBin`.** The literal `int m = (r + l - 1) / 2;` with `value <= bin_upper_bound_[m]` is NOT the same as `slice::partition_point` or `binary_search_by` — the `-1` in the midpoint and the `<=` tie direction change which bin on-boundary values land in. Transcribe the loop verbatim (§5).
- **Computing midpoints / boundaries in f32.** The sample values are f32 feature data but C++ widens to `double` for binning. Mixing precision shifts ULPs.
- **Using `partition_point`/`std::lower_bound` semantics for the de-dup.** `CheckDoubleEqualOrdered` is an asymmetric `b <= nextafter(a)` test, not `a == b`.
- **`std::swap` on `vector<bool>` analog.** C++ explicitly notes `std::swap` on `vector<bool>` is wrong and swaps `group_is_multi_val` element-by-element in the EFB shuffle — in Rust `Vec<bool>`/`Vec<i8>` swap is fine, but preserve the *element-wise* swap of the two parallel vectors in the same loop iteration (§11).
- **Reordering the categorical descending-count sort.** `SortForPair(..., is_reverse=true)` is a `std::stable_sort` on `(count, value)` by count descending. Use a **stable** sort and the same comparator (count desc; ties keep input order) or category→bin assignment diverges (§2).
- **Skipping `most_freq_bin == default_bin` collapse.** Forgetting the `kSparseThreshold` collapse changes `num_total_bin_`, every downstream offset, and the per-row push.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Next-representable-double | Custom bit-manipulation `nextafter` | `f64::next_up()` (std) | Exact `std::nextafter(a, +INF)` equivalent on the pinned toolchain; bit-twiddling risks subnormal/zero edge cases. |
| Random sampling for bin construction | New RNG / `rand` crate | `lgbm_core::Random` (Phase 1) | RNG parity is the whole point; `data_random_seed` seeds the ported LCG. C-API uses `Random(data_random_seed).Sample(n,k)`. |
| Group-search/shuffle randomness in EFB | `rand::shuffle` | `lgbm_core::Random` (`Sample`, `NextShort`) | EFB seeds `Random(num_data)` and calls `Sample` + `NextShort(i+1, num_group)`; any other RNG breaks group layout parity. |
| Stable sort with parallel arrays | Manual index juggling | `slice::sort_by` (stable) on `(key, value)` pairs, mirroring `SortForPair` | C++ uses `std::stable_sort`; Rust `sort_by` is stable. Match the comparator and stability exactly. |
| f32↔f64 widening at ingest | Ad-hoc casts scattered through | One explicit widen point (sampled f32 → f64 for FindBin) | C++ binning math is f64 throughout; centralize the widen so precision is consistent. |

**Key insight:** In this phase, "hand-rolling" almost always means *accidentally writing idiomatic Rust that reorders or re-rounds a computation the C++ does in a specific order.* The safest implementation is the least clever one: transcribe control flow, keep types (`double`→`f64`, `int`→`i32`, `uint32_t`→`u32`, `uint8_t`→`u8`), and verify each stage against goldens before moving up a layer.

## Runtime State Inventory

> This is a greenfield crate (new `lgbm-dataset`), not a rename/refactor of existing runtime state. The only "state" considerations are integration points and fixture provenance, covered below.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore. Bins live in-process, immutable after FinishLoad. | None. |
| Live service config | None. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | None. | None. |
| Build artifacts | New crate dir + workspace `members` entry; `Cargo.lock` gains `lgbm-dataset` path entry (no registry churn). Fixture binaries from xtask C++ capture land in `target/` (gitignored). | Add crate to `members`; commit fixtures (not the build dir). |

**Fixture provenance (critical, from project memory `lightgbm-ref-tree-untracked`):** The `LightGBM/` reference tree and `external_libs/` are **git-untracked** — never `git add` them. EFB/example-dataset fixtures (D-06 source #3, `LightGBM/examples/`) and any captured goldens must be **copied into the committed fixtures dir** (`crates/oracle-harness/fixtures/` or a new `crates/lgbm-dataset/tests/fixtures/`), never referenced from the untracked `LightGBM/` path at test time. Verified this session: `LightGBM/examples/{binary_classification,regression,multiclass_classification,lambdarank,xendcg,...}` and `external_libs/{fast_double_parser,fmt,eigen,compute}` are all present on disk.

## Common Pitfalls

### Pitfall 1: 1-ULP boundary drift from `nextafter` mismatch
**What goes wrong:** A bin boundary differs from C++ by one ULP, so a handful of on-boundary sample values land in adjacent bins; `num_bin` may even differ, cascading into every offset and per-row index.
**Why it happens:** Using `f32` midpoint math, a custom `nextafter`, or an `==`-based de-dup instead of the asymmetric `b <= nextafter(a)` test.
**How to avoid:** Use `f64::next_up()`, keep all boundary math in `f64`, and transcribe `CheckDoubleEqualOrdered` verbatim. Golden layer 1 (`bin_upper_bound_` array) catches this immediately.
**Warning signs:** `bin_upper_bound_` matches for most features but one feature has an off-by-one `num_bin`, or per-row indices diverge only near boundaries.

### Pitfall 2: `ValueToBin` tie direction
**What goes wrong:** Values exactly equal to a boundary (`value == bin_upper_bound_[m]`) route to the wrong bin.
**Why it happens:** Idiomatic binary search uses `<` or `partition_point`; the C++ midpoint is `(r+l-1)/2` (note the `-1`) with `value <= bound` choosing the left branch.
**How to avoid:** Copy the loop in §5 character-for-character. Test with the curated on-boundary battery (D-06 source #2).
**Warning signs:** Per-row parity fails only for rows whose feature value equals a `bin_upper_bound_` entry.

### Pitfall 3: `most_freq_bin` collapse and the bin-0 offset
**What goes wrong:** `num_total_bin_`, `bin_offsets_`, and every pushed bin index are off because the `most_freq_bin == default_bin` collapse (`kSparseThreshold = 0.7`) or the `most_freq_bin == 0 → bin -= 1` adjustment was missed.
**Why it happens:** These are easy-to-skip tail branches in `FindBin` and `FeatureGroup`.
**How to avoid:** Transcribe `FindBin` lines 490–505 and the `FeatureGroup` offset loop (§9) and `PushData` (§10) exactly. Golden layer 3 (offsets) + golden layer 2 (per-row index) jointly catch it.
**Warning signs:** Bin-mapper internals (layer 1) match but per-row indices (layer 2) are uniformly shifted by 1, or `num_total_bin_` per group is off.

### Pitfall 4: Categorical folding order and the 99% cut
**What goes wrong:** Category→bin assignment differs, so categorical features bin differently.
**Why it happens:** Non-stable sort, wrong cut-off arithmetic (`RoundInt((total - na)*0.99f)` — note `0.99f` is *float*), or mishandling the NaN dummy bin (bin 0 = category -1) and the `min_data_in_bin && cur_cat_idx > 1` break.
**How to avoid:** Transcribe §2 exactly, including `RoundInt(x) = (int)(x + 0.5)` semantics and the `0.99f` float literal. Use a stable descending sort by count.
**Warning signs:** Categorical `categorical_2_bin_` map mismatches for low-frequency categories.

### Pitfall 5: EFB non-determinism from RNG or sort drift
**What goes wrong:** Feature groups differ from C++, so bundled layout and offsets diverge — a hard failure for DAT-05.
**Why it happens:** EFB runs `FindGroups` twice (natural order + count-sorted order), picks the smaller, then shuffles with `Random(num_data)`. Any deviation in the `Sample`/`NextShort` sequence, the `stable_sort` by non-zero count, or the conflict-count comparisons changes grouping.
**How to avoid:** Route all randomness through `lgbm_core::Random` seeded exactly as C++ (`Random(num_data)`), use stable sorts, and transcribe `GetConflictCount`/`FindGroups`/the shuffle loop verbatim (§11). Build EFB **last**, after numeric+categorical binning is golden-proven, so a group mismatch is unambiguously an EFB bug.
**Warning signs:** Single-feature-per-group (bundle disabled) parity passes but `enable_bundle=true` groups differ.

### Pitfall 6: Sample-count edge cases
**What goes wrong:** `bin_construct_sample_cnt > num_rows` or tiny datasets produce a different sample set, so bins differ.
**Why it happens:** C++ clamps `sample_cnt = min(total_nrow, bin_construct_sample_cnt)` (`SampleCount`) before `Random(data_random_seed).Sample(total_nrow, sample_cnt)`. Off-by-one or unclamped sampling diverges.
**How to avoid:** Mirror `SampleCount` + `CreateSampleIndices` (Code Examples §12) exactly.
**Warning signs:** Parity fails only on small fixtures or when `bin_construct_sample_cnt` is swept high.

## Code Examples

Verified patterns transcribed from the in-repo C++ source (read this session). Line numbers are stable against the checked-in `LightGBM/` tree.

### §1 — Numeric `FindBin` skeleton (missing-type derivation + dispatch)
```cpp
// Source: LightGBM/src/io/bin.cpp:311-409 (BinMapper::FindBin, numeric branch)
// 1. strip NaNs in-place, count na_cnt
// 2. derive missing_type_:
//    if (!use_missing) None
//    else if (zero_as_missing) Zero
//    else if (non_na_cnt == num_sample_values) None  else NaN (na_cnt = total-non_na)
// 3. stable_sort(values); build distinct_values[] + counts[], inserting a zero
//    "distinct value" with count=zero_cnt at the correct position (front/middle/back)
//    using CheckDoubleEqualOrdered for distinct-equality and the sign-crossing rule.
// 4. dispatch:
//    Zero    -> FindBinWithZeroAsOneBin(...max_bin...); if result size==2 -> missing_type_=None
//    None    -> FindBinWithZeroAsOneBin(...max_bin...)
//    NaN     -> FindBinWithZeroAsOneBin(...max_bin-1, total-na_cnt...); push_back(NaN)
//    num_bin_ = bin_upper_bound_.size()
// 5. cnt_in_bin[]: assign each distinct value to its bin via the same
//    "while value > bound[i_bin] && i_bin<num_bin-1: ++i_bin" walk; NaN bin gets na_cnt.
```
Transcribe the distinct-value/zero-insertion block (lines 343–375) carefully — the zero pseudo-value placement (front if all positive, middle at sign change, back if all negative) is parity-critical.

### §2 — Categorical `FindBin` branch (folding)
```cpp
// Source: LightGBM/src/io/bin.cpp:410-476
// convert distinct_values -> int (negatives => NaN, accumulate into na_cnt, warn)
// rest_cnt = total - na_cnt
// SortForPair(&counts_int, &distinct_values_int, 0, /*is_reverse=*/true)  // count DESC, stable
// cut_cnt = RoundInt((total - na_cnt) * 0.99f)                            // NOTE 0.99f float
// bin_2_categorical_ = [-1]; categorical_2_bin_[-1] = 0; cnt_in_bin=[0]; num_bin_=1
// while cur_cat_idx < n && (used_cnt < cut_cnt || num_bin_ < max_bin):
//     if counts_int[cur] < min_data_in_bin && cur_cat_idx > 1: break
//     bin_2_categorical_.push(value); categorical_2_bin_[value]=num_bin_
//     used_cnt += counts_int[cur]; cnt_in_bin.push(counts_int[cur]); ++num_bin_; ++cur
// missing_type_ = (consumed all && na_cnt==0) ? None : NaN
// cnt_in_bin[0] = total - used_cnt
```

### §3 — `GreedyFindBin` (the core numeric boundary builder)
```cpp
// Source: LightGBM/src/io/bin.cpp:78-155  (transcribe in full — two branches)
// Branch A (num_distinct <= max_bin): walk distinct values, open a new bound when
//   cur_cnt_inbin >= min_data_in_bin, bound = GetDoubleUpperBound((v[i]+v[i+1])/2.0),
//   de-dup with CheckDoubleEqualOrdered; final bound = +inf.
// Branch B (num_distinct > max_bin): clamp max_bin to total_cnt/min_data_in_bin;
//   mark "big count" values (counts[i] >= mean_bin_size) as their own bins;
//   greedily accumulate to mean_bin_size with the 0.5*mean lookahead rule;
//   bounds = GetDoubleUpperBound((upper[i]+lower[i+1])/2.0), de-dup, final = +inf.
```
This is ~75 lines; reproduce arithmetic exactly (the `mean_bin_size * 0.5f` lookahead uses a float literal; `rest_bin_cnt`/`rest_sample_cnt` are `int`).

### §4 — `FindBinWithZeroAsOneBin` and `FindBinWithPredefinedBin`
```cpp
// Source: LightGBM/src/io/bin.cpp:242-309 (zero-as-one) and 157-240 (predefined/forced)
// Zero-as-one: split distinct values at +/- kZeroThreshold (1e-35), allocate left/right
//   bin budgets proportionally, GreedyFindBin each side, stitch with a zero bin.
// Predefined: insert +/- kZeroThreshold bounds + forced_upper_bounds (|x|>kZeroThreshold),
//   stable_sort, then GreedyFindBin the gaps. Used only when forced_upper_bounds non-empty.
```
`kZeroThreshold = 1e-35` (double, from `meta.h:56`), `kEpsilon = 1e-15f`.

### §5 — `ValueToBin` (THE hot path — transcribe verbatim)
```cpp
// Source: LightGBM/include/LightGBM/bin.h:612-650
inline uint32_t BinMapper::ValueToBin(double value) const {
  if (std::isnan(value)) {
    if (bin_type_ == CategoricalBin) return 0;
    else if (missing_type_ == NaN) return num_bin_ - 1;
    else value = 0.0f;                       // None/Zero: treat NaN as 0
  }
  if (bin_type_ == NumericalBin) {
    int l = 0, r = num_bin_ - 1;
    if (missing_type_ == NaN) r -= 1;        // NaN occupies the top bin
    while (l < r) {
      int m = (r + l - 1) / 2;               // <-- the literal midpoint, note -1
      if (value <= bin_upper_bound_[m]) r = m;
      else l = m + 1;
    }
    return l;
  } else {                                   // categorical
    int int_value = static_cast<int>(value);
    if (int_value < 0) return 0;             // negative -> NaN bin
    auto it = categorical_2_bin_.find(int_value);
    return it != end ? it->second : 0;
  }
}
```
Rust: `int_value = value as i32` (C++ `static_cast<int>` truncates toward zero — Rust `as i32` matches for in-range finite values).

### §6 — DenseBin storage + 4-bit packing (D-02)
```cpp
// Source: LightGBM/src/io/dense_bin.hpp:56-82, 510-565
// ctor: if IS_4BIT { data_.resize((n+1)/2, 0); buf_.resize((n+1)/2, 0); } else data_.resize(n,0)
// Push(idx, value): if IS_4BIT { i1=idx>>1; i2=(idx&1)<<2; v=(u8)value<<i2;
//                                if i2==0 data_[i1]=v else buf_[i1]=v }
//                   else data_[idx] = (VAL_T)value
// FinishLoad (4-bit): for i in 0..(n+1)/2: data_[i] |= buf_[i]; buf_.clear()
// data(idx): IS_4BIT ? (data_[idx>>1] >> ((idx&1)<<2)) & 0xf : data_[idx]
```
Rust: `DenseBin<T, const IS_4BIT: bool>` with const generics, or two structs. The even/odd `buf_` split then OR-merge at FinishLoad is the exact byte layout Phase 4 reads — golden-test the raw `data_` bytes.

### §7 — SparseBin delta encoding (`FinishLoad` → `LoadFromPair`)
```cpp
// Source: LightGBM/src/io/sparse_bin.hpp:598-659
// Push(idx,value): if value!=0 push_buffers_[tid].emplace_back(idx, (VAL_T)value)
// FinishLoad: concat push_buffers_, std::sort by .first (data index), LoadFromPair:
//   last_idx=0; for (idx,bin) in pairs:
//     cur_delta = idx - last_idx; if (i>0 && cur_delta==0) continue;   // one val per row
//     while cur_delta>=256 { deltas_.push(255); vals_.push(0); cur_delta-=255; }
//     deltas_.push((u8)cur_delta); vals_.push(bin); last_idx=idx;
//   deltas_.push(0); num_vals_=vals_.size(); GetFastIndex();
```
`std::sort` (NOT stable) by index is fine because indices are unique post the `cur_delta==0` skip. `GetFastIndex` (lines 661–687) builds a power-of-two-strided lookup — reproduce for layout parity, though it is a derived acceleration index.

### §8 — Sparse-vs-dense bin selection
```cpp
// Source: LightGBM/include/LightGBM/feature_group.h:586-612 (CreateBinData)
// single-feature group: sparse if (force_sparse || (!force_dense && num_feature==1
//                       && bin_mappers_[0]->sparse_rate() >= kSparseThreshold(=0.7)))
// In Dataset::Construct path, FeatureGroup is built with force_dense=true (line 75),
// so single/bundled groups are DENSE unless explicitly multi-val/sparse.
// multi-val group: per sub-feature, sparse if sparse_rate() >= 0.7 else dense.
```
Note: the primary `Dataset::Construct` constructor calls `CreateBinData(num_data, is_multi_val_, /*force_dense=*/true, /*force_sparse=*/false)` — so for the common MVP path, single-value groups are **dense**. SparseBin is exercised mainly via multi-val sub-features and the copy/binary-load constructors.

### §9 — FeatureGroup bin-offset packing
```cpp
// Source: LightGBM/include/LightGBM/feature_group.h:39-76
// offset = 1; if (sum_sparse_rate < 0.25 && is_multi_val) { offset=0; is_dense_multi_val=true }
// num_total_bin_ = offset;
// if (group_id==0 && is_dense_multi_val && bin_mappers_[0]->most_freq_bin>0) num_total_bin_=1;
// bin_offsets_=[num_total_bin_]
// for each feature: num_bin = bin_mappers_[i]->num_bin();
//                   if most_freq_bin==0: num_bin -= offset;
//                   num_total_bin_ += num_bin; bin_offsets_.push(num_total_bin_)
```
`multi_val_bin_sparse_threshold = 0.25` (bin.h:599), `kSparseThreshold = 0.7` (bin.h:43).

### §10 — PushData (per-row, per-feature bin emission)
```cpp
// Source: LightGBM/include/LightGBM/feature_group.h:253-267
uint32_t bin = bin_mappers_[sub]->ValueToBin(value);
if (bin == bin_mappers_[sub]->GetMostFreqBin()) return;          // skip most-freq (implicit)
if (bin_mappers_[sub]->GetMostFreqBin() == 0) bin -= 1;
if (is_multi_val_) multi_bin_data_[sub]->Push(tid, row, bin + 1);
else { bin += bin_offsets_[sub]; bin_data_->Push(tid, row, bin); }
```

### §11 — EFB: GetConflictCount + FindGroups + shuffle
```cpp
// Source: LightGBM/src/io/dataset.cpp:60-323
// GetConflictCount(mark, indices, n, max): count marked; early-return -1 if > max.
// FindGroups: single_val_max_conflict_cnt = total_sample_cnt/10000; Random rand(num_data);
//   greedy: for each feature in find_order, gather available groups (capacity + bin checks),
//   sample up to 99 of them (rand.Sample) + always the last group, pick first group whose
//   conflict cnt <= max && <= non_zero/2, else open a new group. Second pass splits
//   dense (>=0.4 used-row-rate) vs sparse-bundle (multi-val) groups.
// FastFeatureBundling: run FindGroups twice (used_features order, and feature_order_by_cnt
//   via stable_sort on non-zero count DESC); keep the fewer-group result; then:
//     Random tmp_rand(num_data);
//     for i in 0..num_group-1: j = tmp_rand.NextShort(i+1, num_group);
//       swap(features_in_group[i], features_in_group[j]);
//       swap(group_is_multi_val[i], group_is_multi_val[j]);   // element-wise, same iter
```
`FixSampleIndices` (lines 82–105) pre-filters sample indices to only rows whose bin != most_freq_bin, *only when default_bin != most_freq_bin*. Transcribe it — it changes the conflict counts EFB sees.

### §12 — Sampling for bin construction (ingestion entry)
```cpp
// Source: LightGBM/src/c_api.cpp:974-982
static int SampleCount(int32_t total_nrow, const Config& c) {
  return total_nrow < c.bin_construct_sample_cnt ? total_nrow : c.bin_construct_sample_cnt;
}
static std::vector<int32_t> CreateSampleIndices(int32_t total_nrow, const Config& c) {
  Random rand(c.data_random_seed);          // <-- lgbm_core::Random::new(seed)
  return rand.Sample(total_nrow, SampleCount(total_nrow, c));
}
// Then per column: gather sample_values[col] = feature value at each sampled row
// (nonzero-aware for sparse), widened f32->f64, passed to FindBin.
```

## State of the Art

Binning algorithm is stable LightGBM core logic; there is no "newer approach" to chase — the contract is *this exact code*. Notes on what NOT to modernize:

| C++ Approach | Keep As-Is | Why |
|--------------|-----------|-----|
| `(r+l-1)/2` integer-midpoint search | Verbatim transcription | Idiomatic binary search rounds ties differently — breaks parity. |
| f64 boundaries + `std::nextafter` | `f64::next_up()` | Exact equivalent; not an upgrade, a translation. |
| `unordered_map` for categorical | `HashMap` (order-irrelevant here) | Folding order is sort-driven, not map-driven — safe (unlike Phase 1's alias map). |
| OpenMP `parallel for` over features | Sequential `for` (D-03) | `num_threads=1` reference; per-feature seams keep rayon a later drop-in. |

**Deprecated/outdated:** none relevant. The 4-bit packed bin (`DenseBin<uint8_t,true>`) is current LightGBM (used for `num_bin<=16`), not legacy — include it (D-02).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `f64::next_up()` is bit-exact-equivalent to C++ `std::nextafter(a, +INFINITY)` for the finite positive/negative args binning produces | Pattern 1, Pitfall 1 | If a subnormal/zero/sign-edge case diverges, a boundary drifts 1 ULP. Mitigation: golden layer 1 catches it; fall back to `libm::nextafter` if a specific value mismatches. LOW risk — IEEE-754 nextUp is well-specified and Rust's `next_up` follows it. |
| A2 | C++ `static_cast<int>(double)` truncation toward zero matches Rust `value as i32` for in-range finite categorical values | §5 ValueToBin categorical | Out-of-range/NaN already special-cased in C++ (negative→0); finite in-range truncation is identical. LOW risk. |
| A3 | The `external_libs/{fast_double_parser,fmt}` headers present on disk are sufficient to compile a focused `bin.cpp`+`dataset.cpp` capture harness without building full `lib_lightgbm` | Validation Architecture, Environment | If `dataset.cpp` transitively needs more of the lib (network, threading, io), the harness may need more sources or a fuller build. MEDIUM risk — mitigated by capturing BinMapper goldens (layer 1+2) via a minimal harness first; EFB goldens (layer 3) may need a fuller build or CLI-dump fallback. |
| A4 | Plain `HashMap` iteration order for `categorical_2_bin_` cannot affect any committed output | Stack alternatives, State of the Art | If some downstream consumer iterates the map in order (it should not — lookups are by key), order could leak. LOW risk — verified `ValueToBin` does point lookups only. |
| A5 | The C-API `from_mat` ingestion path (sample→FindBin→Construct→PushData→FinishLoad) is the correct model for the Rust `from_mat`/`from_csr`/`from_csc`, vs the dataset_loader text path | DAT-07, Architecture | The text-file path adds parsing not in scope; the C-API path is the right in-memory analog. LOW risk — both converge on `Dataset::Construct`. |

## Open Questions

1. **EFB golden capture mechanism (layer 3).**
   - What we know: `external_libs` headers are on disk; a `bin.cpp`-only harness can produce layer 1+2 goldens.
   - What's unclear: whether `dataset.cpp` (EFB) compiles in a focused harness or needs a fuller `lib_lightgbm` build (which Phase 1 avoided because external_libs were unvendored — but they ARE present now).
   - Recommendation: Plan a Wave that (a) first delivers BinMapper goldens via a minimal harness, then (b) attempts a focused `dataset.cpp` capture; if it pulls too much of the lib, fall back to driving the full build with `enable_bundle=true` and dumping group/offset layout via the CLI or a tiny linked program. De-risk by sequencing EFB last.

2. **Sample-value gathering for sparse ingest.**
   - What we know: dense gathering is trivial (value at sampled row); the C-API sparse path gathers nonzeros per column.
   - What's unclear: exact handling when a sampled row is zero in a sparse column (counts toward `zero_cnt`, not `sample_values`).
   - Recommendation: Mirror the C-API `LGBM_DatasetCreateFromCSR`/`FromMat` sample-column construction (read `c_api.cpp` `PushDataToBin`/sample gather in the planning read-set); golden-test sparse ingest against dense-equivalent matrices.

3. **`min_split_data`/`feature_pre_filter` (`NeedFilter`) in MVP scope.**
   - What we know: `FindBin` takes `min_split_data` (= `filter_cnt` derived from `min_data_in_leaf`) and `pre_filter`; `NeedFilter` can mark a feature trivial.
   - What's unclear: whether MVP fixtures exercise `feature_pre_filter` (default true) enough to need parity now.
   - Recommendation: Include `NeedFilter` in the transcription (it is small and on the `FindBin` tail path) and include at least one fixture where pre-filtering triggers, so the `is_trivial_` path is golden-covered.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (edition 2024) | crate build | ✓ (assumed, Phase 1 built) | 1.95 pin | — |
| `f64::next_up` (std) | boundary math | ✓ | stable since 1.86 ≤ 1.95 | `libm::nextafter` |
| C++ toolchain + CMake | fixture regen only (xtask) | ✓ (Phase 1 used it) | per Phase 1 manifest | — |
| `LightGBM/` source tree | fixture regen (read-only ref) | ✓ on disk (untracked) | checked-in submodule rev | — |
| `external_libs/{fast_double_parser,fmt,eigen,compute}` | compiling `bin.cpp`/`dataset.cpp` capture harness | ✓ on disk (untracked) | submodule revs (see `git submodule status`) | vendor the 2 needed headers into xtask/cpp, or full lib build |
| `LightGBM/examples/*` datasets | D-06 fixture source #3 | ✓ on disk | — | copy into committed fixtures (never reference untracked path at test time) |
| `lgbm_core::Random` | sampling + EFB RNG | ✓ | Phase 1 | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** EFB capture harness build (A3/Q1) has a CLI-dump fallback if `dataset.cpp` won't compile in isolation.

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`). ORA-03 explicitly demands per-stage parity at bin granularity, so this section is mandatory and central to the phase.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + the existing `oracle-harness` comparator (`crates/oracle-harness/src/comparator.rs`) |
| Config file | none (cargo native); fixtures under `crates/oracle-harness/fixtures/` + new binning fixtures (committed) |
| Quick run command | `cargo test -p lgbm-dataset` |
| Full suite command | `cargo test --workspace` |
| Fixture regen (C++) | `cargo run -p xtask -- regen` (extend with a `bin-capture` step; idempotent, master-seed-derived per D-14) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DAT-01 | `bin_upper_bound_`/`num_bin`/missing/default/most_freq match C++ (golden layer 1) across synthetic+edge corpus | unit (golden) | `cargo test -p lgbm-dataset bin_mapper_internals` | ❌ Wave 0 |
| DAT-01 | per-row numeric bin index matches (golden layer 2) | unit (golden) | `cargo test -p lgbm-dataset numeric_assignment` | ❌ Wave 0 |
| DAT-03 | NaN/+0/-0/on-boundary/all-missing routing matches | unit (golden) | `cargo test -p lgbm-dataset missing_edge_cases` | ❌ Wave 0 |
| DAT-04 | categorical `categorical_2_bin_` + per-row index matches (layer 1+3) | unit (golden) | `cargo test -p lgbm-dataset categorical_folding` | ❌ Wave 0 |
| DAT-02 | DenseBin raw bytes (incl. 4-bit `data_`) match expected layout; SparseBin deltas/vals/fast_index match | unit (golden) | `cargo test -p lgbm-dataset bin_storage_layout` | ❌ Wave 0 |
| DAT-05 | EFB group membership + `bin_offsets_`/`num_total_bin_` per group match (layer 3) | unit (golden) | `cargo test -p lgbm-dataset efb_grouping` | ❌ Wave 0 |
| DAT-06 | metadata round-trips (labels/weights/init_score/query) + query-weight derivation | unit | `cargo test -p lgbm-dataset metadata` | ❌ Wave 0 |
| DAT-07 | dense vs CSR vs CSC of the same logical matrix produce identical bins | unit (cross-rep) | `cargo test -p lgbm-dataset ingest_equivalence` | ❌ Wave 0 |
| DAT-07 | end-to-end ingest of a `LightGBM/examples/` dataset matches C++ goldens (layer 1+2) | integration (golden) | `cargo test -p lgbm-dataset example_dataset_parity` | ❌ Wave 0 |
| ORA-03 | three-layer goldens are wired into the oracle comparator + REFERENCE_MANIFEST records binning master seed/tolerance | harness | `cargo test -p lgbm-dataset` (replays committed) + `cargo run -p xtask -- regen` idempotent | ❌ Wave 0 |

All comparisons for binning are **exact integer / exact-bytes** (bin indices, group offsets, storage bytes) — NOT the f32 ~1e-6 tolerance. The `bin_upper_bound_` array is f64 and should compare **bit-exact** (it is computed deterministically), so use exact `==` on the raw f64 bits for layer 1, not a tolerance.

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-dataset` (the stage's golden tests; sub-30s).
- **Per wave merge:** `cargo test --workspace` (binning goldens + Phase 1 RNG/config regressions).
- **Phase gate:** full suite green + `cargo run -p xtask -- regen` produces byte-identical committed fixtures (idempotency proof, D-06) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/lgbm-dataset/Cargo.toml` + crate skeleton + `members` entry
- [ ] `crates/lgbm-dataset/tests/` golden fixtures dir (or extend `oracle-harness/fixtures/`) — three layers, committed
- [ ] `xtask` `bin-capture` subcommand: focused C++ harness compiling `bin.cpp` (+ later `dataset.cpp`) against on-disk `external_libs`, emitting layer-1/2/3 goldens from the master seed
- [ ] Extend `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` with binning master seed, corpus parameters (max_bin/min_data_in_bin/sample_cnt sweeps), and exact-comparison tolerance note
- [ ] Comparator helpers for exact bin-index vectors, exact f64-bit boundary arrays, exact byte-layout (the existing comparator is f32-tolerance oriented — add exact-equality variants)
- [ ] Copy chosen `LightGBM/examples/*` datasets into committed fixtures (never reference untracked path at test time)

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1` in config.json. This phase has no network, auth, session, or secrets surface — it is an in-process data-transformation library reading caller-provided in-memory matrices. The relevant ASVS category is input validation at the ingestion boundary.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (library, no auth) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `thiserror` `DatasetError` at `from_mat`/`from_csr`/`from_csc`: validate matrix dims, CSR/CSC `indptr`/`indices` monotonicity & bounds, label/weight length == num_rows, query boundaries sum == num_rows, non-negative `max_bin`/`min_data_in_bin`. Return typed errors, never panic on bad caller input. |
| V6 Cryptography | no | — (no crypto; the "Random" LCG is a deterministic binning seed, not a security primitive — never represent it as one) |

### Known Threat Patterns for this stack (Rust in-process library)

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds read from malformed CSR/CSC (`indices[k] >= ncols`, non-monotone `indptr`) | Tampering / DoS | Validate `indptr` monotone & in-range and `indices` bounded before indexing; return `DatasetError`, no `unsafe` indexing on caller data. |
| Integer overflow in offset/`num_total_bin_` accumulation (large feature counts × bins) | DoS | C++ uses `uint64_t num_total_bin`; mirror with `u64` accumulator (matches dataset.cpp:383). Use checked/`as u64` widening for offset math. |
| Panic on degenerate input (0 rows, all-NaN feature, 0 features) | DoS | C++ handles these (trivial features, empty `used_features` warning); mirror the guards (`is_trivial_`, the empty-`used_features` warning) and return/produce empty-but-valid datasets rather than panicking. |
| Slopsquat / supply-chain | — | N/A — zero new external packages this phase. |

## Sources

### Primary (HIGH confidence — read directly this session)
- `LightGBM/include/LightGBM/bin.h` — `BinMapper`, `Bin`/`BinIterator`/`MultiValBin` interfaces, `BinType`/`MissingType` enums, `ValueToBin` inline (lines 612-650), thresholds (`kSparseThreshold=0.7` L43, `multi_val_bin_sparse_threshold=0.25` L599).
- `LightGBM/src/io/bin.cpp` — `FindBin` (311-506), `GreedyFindBin` (78-155), `FindBinWithZeroAsOneBin` (242-309), `FindBinWithPredefinedBin` (157-240), `NeedFilter` (54-76), `CreateDenseBin`/`CreateSparseBin` factories (613-633), `CreateMultiValBin` (635-706), Copy/Save serialization (508-598).
- `LightGBM/src/io/dense_bin.hpp` — DenseBin storage, 4-bit Push/FinishLoad/data (56-82, 510-565), CopySubrow (567-592).
- `LightGBM/src/io/sparse_bin.hpp` — SparseBin Push/FinishLoad/LoadFromPair delta-encode (92-97, 598-659), GetFastIndex (661-687).
- `LightGBM/include/LightGBM/feature_group.h` — FeatureGroup ctors + offset packing (39-111), AllocateBins (212-229), PushData (253-267), CreateBinData (586-612), FinishLoad.
- `LightGBM/src/io/dataset.cpp` — `Construct` (325-441), `FinishLoad` (443-463), `GetConflictCount`/`MarkUsed`/`FixSampleIndices` (60-105), `FindGroups` (107-244), `FastFeatureBundling` (246-323), `PushDataToMultiValBin` (465-516).
- `LightGBM/src/io/dataset_loader.cpp` — `ConstructBinMappersFromTextData` FindBin call site + missing/categorical/forced wiring (594-752), sampling (1009-1064).
- `LightGBM/src/c_api.cpp` — `SampleCount`/`CreateSampleIndices` in-memory ingestion sampling (974-982).
- `LightGBM/include/LightGBM/utils/common.h` — `GetDoubleUpperBound`/`CheckDoubleEqualOrdered` (845-852), `SortForPair` (611-632), `RoundInt` (904).
- `LightGBM/include/LightGBM/meta.h` — `kEpsilon=1e-15f` (L54), `kZeroThreshold=1e-35f` (L56).
- `LightGBM/include/LightGBM/dataset.h` — `Dataset`/`FeatureGroup`/`Metadata` interfaces, immutability.
- Workspace: `crates/lgbm-core/src/random.rs` (Random API: `new`, `next_short`, `next_int`, `next_float`, `sample`), `Cargo.toml` (rust 1.95, deps), `crates/oracle-harness/` (comparator + fixtures + xtask regen pattern).
- `.planning/` CONTEXT (Phase 1 + Phase 2), REQUIREMENTS, STATE, config.json — read this session.

### Secondary (MEDIUM confidence)
- `f64::next_up`/`next_down` stabilization — Rust 1.86 (training knowledge, A1; the pinned 1.95 toolchain has it). Verify in `cargo doc` / build if any boundary mismatch appears.

### Tertiary (LOW confidence)
- None — this research has no WebSearch-only claims. The spec is the in-repo source.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new external deps; only `lgbm-core` (Phase 1) + std, both verified on disk.
- Architecture: HIGH — entire pipeline transcribed from the in-repo C++ source read line-by-line this session.
- Pitfalls: HIGH — derived directly from the exact FP/RNG/tie-break code paths, not inferred.
- Fixture-capture feasibility: MEDIUM — BinMapper capture is clearly buildable (external_libs present); EFB-stage capture (`dataset.cpp` in isolation) is the one open feasibility question (Q1/A3), with a CLI-dump fallback.

**Research date:** 2026-06-05
**Valid until:** Indefinite for the algorithm spec (pinned to the checked-in `LightGBM/` source — only changes if the submodule rev changes). 30 days for the `f64::next_up`/toolchain claim.
