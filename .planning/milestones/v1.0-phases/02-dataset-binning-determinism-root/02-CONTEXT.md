# Phase 2: Dataset + Binning (determinism root) - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 delivers a **binned, immutable columnar dataset whose bin boundaries and per-row bin assignments are bit-identical to C++** — the determinism root every downstream split inherits. Binning targets **exact integer bin-index reproduction** (the f32/~1e-6 score contract from Phase 1 does not relax the integer binning requirement).

In scope: `BinMapper` continuous→bin mapping with literal `(r+l-1)/2` + `<=` search (DAT-01); Dense/Sparse columnar bin store immutable after finish-load (DAT-02); missing-value handling `use_missing`/`zero_as_missing`/`MissingType` with C++-matching default-direction routing (DAT-03); categorical encoding category→bin + low-frequency folding (DAT-04); Exclusive Feature Bundling `enable_bundle` / `MultiValBin` (DAT-05); metadata — labels, weights, init_score, query/group boundaries (DAT-06); in-memory dense + CSR/CSC sparse ingestion via the Rust API (DAT-07); per-stage parity tests at bin granularity (ORA-03).

Out of scope: histogram construction, split finding, tree learning (Phase 5); prediction / model text I/O (Phase 3); CubeCL compute kernels (Phase 4); the polished public Rust-native `Dataset`/`Booster` API (Phase 6, API-01 — Phase 2 exposes only minimal internal-facing ingestion constructors); text-file / binary-cache / Arrow ingestion (v2: ING-01/02/03).

</domain>

<decisions>
## Implementation Decisions

### Bin Storage Representation
- **D-01:** Model C++'s value-width-templated storage as a **faithful 1:1 mirror**: a `Bin` trait with generic `DenseBin<T: BinValue>` and `SparseBin<T>` implementations, stored as `Box<dyn Bin>` (or equivalent) per feature group, with a **factory that selects the value width (u8/u16/u32) from `num_bins`** exactly as C++ does (uint8 <256, uint16 <65536, else uint32). Zero-cost element access in the hot path; the byte layout is the one the Phase 4/5 histogram path will read against. (Chose the C++ mirror over an enum-of-typed-vecs or a single-u32-deferred representation.)
- **D-02:** **Include the 4-bit packed DenseBin variant now** (the `IS_4BIT` path, used when `num_bins ≤ 16`): two bins per byte with the exact C++ packing, golden-tested in Phase 2. The histogram phase inherits a complete, parity-proven storage layer rather than a stub. (Bin *indices* are identical with or without packing, so this is about faithful byte layout, not Phase 2 index parity.)

### Parallelism vs Determinism
- **D-03:** **Sequential core with parallel-ready seams.** Phase 2 binning + dataset construction execute **single-threaded**, matching the pinned `num_threads=1` C++ reference exactly — zero reduction-ordering or chunk-boundary ambiguity in the subsystem that must be the deterministic anchor. But the code is **structured into per-feature independent units** (each feature bins independently, no shared mutable accumulation across features) so rayon can be dropped in later as a separately-validated optimization that must produce byte-identical bins. Binning is CPU pre-processing, off the CubeCL backend — correctness-first matches the phase's purpose.

### Crate Boundary & API Shape
- **D-04:** **One new `lgbm-dataset` crate** holds the entire subsystem: `BinMapper`, the `Bin` trait + Dense/Sparse store, `MissingType`/categorical encoding, EFB/`MultiValBin`, metadata, and ingestion. Mirrors C++'s `src/io/` cohesion and keeps the determinism root in one place. (Chose single crate over an `lgbm-bin` + `lgbm-dataset` split or extending `lgbm-core`; consistent with Phase 1 D-09 "add a crate per subsystem in the phase that introduces it" and `lgbm-*` naming from D-08.)
- **D-05:** **Minimal internal-facing ingestion API only.** Expose just enough to ingest dense + CSR/CSC matrices and metadata and drive parity tests — e.g. `Dataset::from_mat` / `from_csr` / `from_csc` returning a store that is **immutable after finish-load**. No ergonomic builder, no sklearn-ish surface; the polished public Rust-native API is designed in Phase 6 (API-01) once `Booster` exists. Smallest stable-surface commitment now.

### Parity Fixture Strategy (DAT-01 / ORA-03)
- **D-06:** **Four-source parity corpus**, all captured under the D-14 randomized-at-capture / committed-master-seed discipline:
  1. **Synthetic randomized distributions** — many feature columns derived from the committed master seed across uniform / gaussian / skewed / heavy-tailed / discrete / constant / high-cardinality shapes, sweeping `max_bin` / `min_data_in_bin` / `bin_construct_sample_cnt` (and `data_random_seed`).
  2. **Curated edge-case battery** — hand-built columns hitting the branch points: NaN per `MissingType` (None/Zero/NaN), `+0.0`/`-0.0`, on-boundary values (exact `bin_upper_bound_`), out-of-range categorical, all-missing, single-value, sparse/zero-heavy.
  3. **LightGBM bundled example datasets** — real data under `LightGBM/examples/` (binary_classification, regression, …) as realistic end-to-end binning fixtures.
  4. **Categorical + EFB-specific corpus** — multi-categorical columns with rare levels (DAT-04 low-frequency folding) and mutually-exclusive sparse feature sets (DAT-05 bundling), since these have their own bit-exact grouping/ordering logic separate from numeric binning.
- **D-07:** **Per-stage golden granularity = three layers** (maximally diagnostic, befitting a determinism-root phase): (1) **BinMapper internals** — `bin_upper_bound_` array, `bin_type`, `missing_type`, `default_bin`, `most_freq_bin`, `num_bin`; (2) the **full per-row bin-index assignment vector** per feature; (3) **categorical category→bin maps + EFB bundle/offset layout**. A mismatch points at the exact stage and feature, localizing divergence to binning before histograms exist (SC#5). Reuses Phase 1's committed-fixtures + idempotent-regen pattern (D-06): goldens generated once from the in-repo C++ build, committed, replayed with no C++ toolchain at normal test time.

### Claude's Discretion
- Exact fixture file formats/serialization, the precise `BinValue` trait bound set, the internal module layout within `lgbm-dataset`, the sparse-vs-dense selection threshold details, and the precise category→bin folding/ordering implementation are left to research/planning — bounded by "faithful C++ mirror, exact integer bin-index parity" above. When C++ behavior is the spec, the C++ source (below) is authoritative over any inferred default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ reference source (read-only port target — authoritative for all binning behavior)
- `LightGBM/include/LightGBM/bin.h` — `BinMapper`, `Bin`/`BinIterator` virtual interface, `DenseBin`/`SparseBin`/`MultiValBin` declarations, `BinType`/`MissingType` enums, `BinMapper::ValueToBin` inline (the literal `(r+l-1)/2` + `<=` search and missing/NaN/categorical routing)
- `LightGBM/src/io/bin.cpp` — `BinMapper::FindBin` (bin boundary construction, `max_bin`/`min_data_in_bin`/`bin_construct_sample_cnt`/`data_random_seed`, categorical low-frequency folding), `most_freq_bin`/`default_bin` derivation
- `LightGBM/src/io/dense_bin.hpp` — `DenseBin<VAL_T, IS_4BIT>` storage incl. the 4-bit packing (two bins/byte, `(idx+1)/2` layout) — the D-02 reference
- `LightGBM/src/io/sparse_bin.hpp` — `SparseBin<VAL_T>` storage + iterator
- `LightGBM/include/LightGBM/dataset.h` — `Dataset` / `FeatureGroup` / `Metadata` interfaces, immutability after `FinishLoad`
- `LightGBM/src/io/dataset.cpp` — dataset construction, feature grouping, EFB (`enable_bundle`) bundling, `CreateBin` width/4-bit selection, metadata wiring
- `LightGBM/include/LightGBM/config.h` — `max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, `data_random_seed`, `use_missing`, `zero_as_missing`, `enable_bundle`, `max_cat_to_onehot`, `cat_l2`, etc. (already ported into `lgbm-core::Config` in Phase 1)

### Foundations to build on (Phase 1 deliverables)
- `crates/lgbm-core/src/random.rs` — the bit-exact `Random` LCG; binning's sampling (`bin_construct_sample_cnt` / `data_random_seed`) MUST route through it for RNG parity
- `crates/lgbm-core/src/config/` — `Config` bag (params already modeled); read the binning-relevant fields above
- `crates/lgbm-core/src/types.rs`, `src/error.rs` — f32 types + `thiserror` domain errors at the crate boundary (FND-03/FND-04 pattern to extend into `lgbm-dataset`)
- `crates/oracle-harness/` — the comparator + golden-replay seam every parity test plugs into; `REFERENCE_MANIFEST.md` records master seed + tolerance (extend for binning fixtures per D-06/D-14)

### Project-level contract
- `.planning/PROJECT.md` — Core Value, Constraints, Key Decisions (f32/~1e-6; standard f32 accumulations)
- `.planning/REQUIREMENTS.md` — DAT-01..07, ORA-03 (Phase 2 requirements)
- `.planning/ROADMAP.md` §"Phase 2" — goal + 5 success criteria
- `.planning/phases/01-oracle-contract-foundations/01-CONTEXT.md` — Phase 1 decisions carried forward (D-05/D-06/D-08/D-09/D-11/D-12/D-14 referenced above)

### Codebase maps (reference C++ architecture)
- `.planning/codebase/STRUCTURE.md`, `.planning/codebase/STACK.md`, `.planning/codebase/CONVENTIONS.md` — C++ layout, stack, conventions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `lgbm-core::Random` (`crates/lgbm-core/src/random.rs`) — bit-exact LCG; binning's value sampling routes through it (parity-critical for `bin_construct_sample_cnt`/`data_random_seed`).
- `lgbm-core::Config` (`crates/lgbm-core/src/config/`) — all binning hyperparameters already modeled and validated.
- `lgbm-core` f32 types + `thiserror` error pattern (`src/types.rs`, `src/error.rs`) — extend the same boundary-error idiom into `lgbm-dataset`.
- `oracle-harness` comparator + committed-golden + idempotent-regen harness (`crates/oracle-harness/`) — the validation seam; binning goldens plug in here, extending `REFERENCE_MANIFEST.md`.

### Established Patterns
- Hand-port mirroring C++ 1:1, flat structs, guarded by a drift/parity test (Phase 1 D-11/D-12) — applies to `BinMapper` and the `Bin` hierarchy.
- Committed fixtures + idempotent C++-regen script; no C++ toolchain at normal test time (D-06). Randomized-at-capture from a committed master seed (D-14).
- C++ constants that begin to matter here: `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f` (live in C++ headers; relevant to binning thresholds/zero handling).
- C++ reference built from the in-repo `LightGBM/` submodule with `deterministic=true force_row_wise=true num_threads=1` (D-05) — matches the D-03 single-threaded Rust path.

### Integration Points
- New `lgbm-dataset` crate depends on `lgbm-core` (types/errors/RNG/config); it becomes the dependency root for Phase 3 (predict reads the binned store + bin mappers), Phase 4 (histogram kernels read `Bin`/`MultiValBin` column data), and Phase 5 (tree learner).
- Add `crates/lgbm-dataset` to the workspace `members` in the root `Cargo.toml` (currently lists lgbm-core, lgbm-compute, oracle-harness, xtask).
- The `LightGBM/` reference tree is **untracked** (never `git add` it); EFB/example-dataset fixtures must be copied/derived, not referenced from an untracked submodule at test time (see memory: lightgbm-ref-tree-untracked).

</code_context>

<specifics>
## Specific Ideas

- Faithfulness over idiom: every gray area resolved toward the closest C++ mirror (templated bin widths incl. 4-bit packing, single-threaded determinism, src/io/-style crate cohesion). The user consistently prioritized "match the original exactly" — when in doubt, reproduce C++ behavior rather than choose a cleaner Rust design.
- `BinMapper::ValueToBin` must preserve the literal `(r+l-1)/2` midpoint + `<=` boundary search, not an idiomatic binary search that could round differently on ties/boundaries.
- The three-layer golden capture (mapper internals + per-row assignment + categorical/EFB layout) is deliberately maximally diagnostic so a future histogram-phase divergence can be definitively ruled out of binning.

</specifics>

<deferred>
## Deferred Ideas

- **Parallel (rayon) binning** — Phase 2 ships sequential; rayon parallelization is a later, separately-validated optimization that must produce byte-identical bins (enabled by the per-feature seams from D-03).
- **Polished public Rust-native `Dataset` API** — minimal internal constructors only in Phase 2; the ergonomic/sklearn-style surface is Phase 6 (API-01) once `Booster` exists.
- **Text-file / binary-cache / Arrow ingestion** — v2 (ING-01/02/03), already roadmapped out.
- **4-bit packing optimization tuning** — included for byte-layout parity now (D-02), but any perf tuning of the packed path is downstream.

None other — discussion stayed within Phase 2 scope.

</deferred>

---

*Phase: 2-dataset-binning-determinism-root*
*Context gathered: 2026-06-05*
