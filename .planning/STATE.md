---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 02-03-PLAN.md
last_updated: "2026-06-05T06:58:28Z"
last_activity: 2026-06-05 -- Completed Plan 02-03 (categorical folding + missing-value routing golden parity)
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 8
  completed_plans: 6
  percent: 38
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05)

**Core value:** For identical inputs and config, reproduce C++ LightGBM outputs to within ~1e-6 absolute difference on every backend (CPU and ROCm), using f32 (single-precision) data types matching the C++ reference defaults.
**Current focus:** Phase 02 — dataset-binning-determinism-root

## Current Position

Phase: 02 (dataset-binning-determinism-root) — EXECUTING
Plan: 4 of 5
Status: Executing Phase 02
Last activity: 2026-06-05 -- Completed Plan 02-03

Progress: [██████░░░░] 60% (3 of 5 Phase-02 plans complete)

### Resume

Phase 02 Plan 03 complete: categorical folding + missing-value routing are locked and bit-proven. Next: Plan 02-04 (ingestion `from_mat`/`from_csr`/`from_csc` + metadata).

- `lgbm-dataset` crate exists with `BinMapper` (numeric `find_bin`/`value_to_bin`, bit-exact f64 boundary kernel), `BinType`/`MissingType`, and `DatasetError`.
- Categorical (Plan 02-03): `find_bin_categorical` (descending-count fold via stable `SortForPair`, f32 0.99 cut, `min_data_in_bin` fold-break, NaN dummy bin 0) + `categorical_2_bin_`/`bin_2_categorical_` + categorical `value_to_bin`; completed `MissingType` routing (None/Zero/NaN, signed zeros, all-missing) — 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden replay, bit-identical, idempotent.
- Storage layer (Plan 02-02): `Bin` trait + `BinValue` (u8/u16/u32) + `create_dense_bin`/`create_sparse_bin` factories (Box<dyn Bin>, D-01); `DenseBin<T, IS_4BIT>` incl. 4-bit packing (D-02); `SparseBin<T>` delta-encode + fast-index; `FeatureGroup` offset packing (u64) + `PushData`; `Dataset::construct` + `finish_load` type-state immutability (→ `FinishedDataset`).
- `bin-capture` xtask subcommand + `xtask/cpp/bin_capture.cpp` emit numeric goldens (layers 1+2) AND storage-layout goldens (DenseBin/SparseBin/FeatureGroup bytes); oracle-harness has `compare_exact_u32`/`compare_exact_f64_bits`/`compare_exact_bytes`; later plans plug in here.
- 45-case numeric golden replay (bit-identical, SC#1/SC#5) + 6-case storage golden replay (byte-identical incl. 4-bit + sparse, SC#2); both regen idempotent.
- `lgbm_core::Config` + `Config::from_params` are the config bag for all later crates (Phase 1).

## Performance Metrics

**Velocity:**

- Total plans completed: 1 (tracked)
- Average duration: ~3 min
- Total execution time: <1 hour

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-oracle-contract-foundations | 3/3 | ~2 sessions | ~1 session |
| 02-dataset-binning-determinism-root | 3/5 | ~30 min | ~10 min |

**Plan 01-02:** 3 tasks, 11 files (9 created + 2 modified), 29 new tests; `cargo test --workspace` green.
**Plan 01-03:** 2 TDD tasks, 3 files modified, 7 new tests (49 → 56); deterministic alias resolution + empty==absent reads; `cargo test --workspace` green.
**Plan 02-01:** 4 tasks (~12 min), 14 files (9 created + 5 modified), numeric BinMapper kernel + bin-capture harness + 45-case golden replay (layers 1+2 bit-exact); `cargo test --workspace` green.
**Plan 02-02:** 3 tasks (~9 min), 11 files (7 created + 4 modified), bin-storage layer (Bin trait + DenseBin/4-bit + SparseBin + FeatureGroup offsets + Dataset finish_load type-state immutability) + 6-case storage golden replay (byte-exact); 25 new lib tests (10→35); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-03:** 2 tasks (~9 min), 9 files (4 created + 5 modified), categorical `find_bin_categorical` (stable descending-count fold + f32 0.99 cut + fold-break + NaN dummy bin) + categorical `value_to_bin` + completed MissingType routing; 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden replay (bit-exact); 7 new inline + 2 integration tests (35→42 lib); `cargo test --workspace` green; bin-capture idempotent.

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Build order is dependency-forced — each layer must match the reference before anything above it can be validated (binning → predict → compute → learner → GBDT → variants → Python).
- [Phase 1 discuss, 2026-06-05]: Numerical contract revised to **f32 (single-precision) end-to-end, ~1e-6 absolute oracle tolerance** on every backend — matching the C++ reference defaults (`score_t`/`label_t` = `float`). Supersedes the prior strict-1e-12 / tiered-oracle direction.
- [Phase 1 discuss, 2026-06-05]: **Standard f32 histogram/score accumulations** on CPU and ROCm — the integer-quantized histogram strategy is dropped (buys nothing at f32 / ~1e-6).
- [Phase 1 exec, 2026-06-05]: **Deterministic config invariant** — `from_params` alias-collision resolution is a faithful port of C++ `ParameterAlias::KeyAliasTransform` + `Config::SortAlias` (canonical beats alias; alias-vs-alias ties by `(key.len(), key)`). No observable Config outcome may depend on HashMap iteration order; enforced by an N-run determinism test. The seed + six enum reads route through `present()` so empty == absent (C++ `Get*` parity).
- [Phase 1 exec, 2026-06-05]: **Header-only C++ RNG capture** — `rng_capture` compiles directly against `include/LightGBM/utils/random.h` instead of linking `lib_lightgbm` (external_libs submodules not vendored, so the full lib is unbuildable). Numerically identical reference source; preserves FND-01 / ORA-02 / D-14 parity contract. Master seed 1592594996, 512 cases.
- [Phase 2 exec, 2026-06-05]: **Numeric binning capture via verbatim transcription** — `xtask/cpp/bin_capture.cpp` verbatim-transcribes the numeric `BinMapper::FindBin`/`ValueToBin` family from the pinned `bin.cpp`/`bin.h` (using real `std::nextafter`) rather than compiling `bin.cpp`, because `external_libs/{fast_double_parser,fmt}` are empty/unvendored here so `bin.cpp` → `common.h` is unbuildable. Header-only reference `Random` used for sampling. Goldens byte-identical to lib_lightgbm. Binning compared **bit-exact** (`.to_bits()` / exact-u32), NOT the ~1e-6 oracle tolerance. BIN_MASTER_SEED 0x0B11BEEF, 45 cases.
- [Phase 2 exec, 2026-06-05]: **Numeric binning determinism root locked (DAT-01, ORA-03)** — `BinMapper::find_bin`/`value_to_bin` produce `bin_upper_bound_` (f64 `next_up` boundary math, asymmetric `b <= a.next_up()` dedup) and per-row bin indices (literal `(r+l-1)/2` + `<=` search) bit-identical to C++ across 45 cases / 21,475 rows.
- [Phase 2 exec, 2026-06-05]: **Bin-storage layer locked (DAT-02)** — `Bin` trait + `BinValue` width abstraction + `create_dense_bin`/`create_sparse_bin` factories (`Box<dyn Bin>`, D-01) selecting u8/u16/u32 + the 4-bit packed `DenseBin<u8,true>` (num_bin<=16, D-02) exactly per bin.cpp:613-633; `DenseBin` 4-bit even/odd `buf_` split + OR-merge, `SparseBin` 255-run-length delta + GetFastIndex, `FeatureGroup` u64 offset packing + `PushData` (most-freq skip / -1 / +offset) all byte-identical to C++ across 6 storage cases spanning every width path + sparse + odd-row 4-bit.
- [Phase 2 exec, 2026-06-05]: **Dataset immutability = type-state, not a runtime flag** — `Dataset::finish_load(self)` consumes the mutable loading state and returns `FinishedDataset` (no `push_*`/`finish_load` method), so a post-finish mutation is a COMPILE error — strictly stronger than C++ `is_finish_load_` while observably identical.
- [Phase 2 exec, 2026-06-05]: **Categorical folding + missing routing locked (DAT-03, DAT-04)** — `find_bin_categorical` transcribes the descending-count fold via the STABLE `slice::sort_by` (mirrors `SortForPair(is_reverse=true)`; equal-count ties keep ascending-value order), the f32 `0.99` cut (`RoundInt((rest as f32 * 0.99f) ...)` — NOT `0.99_f64`, which would shift the cut), the `min_data_in_bin && cur_cat_idx>1` fold-break, and the NaN dummy bin 0; `categorical_2_bin_` is a by-key `HashMap` (fold order is sort-driven). Missing routing (None/Zero/NaN, signed +0/-0 identical, all-missing single bin) proven bit-identical to C++ across a 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden battery.
- [Phase 2 exec, 2026-06-05]: **`autobins = false` on lgbm-dataset** — the locked module path `src/bin/{mod,dense_bin,sparse_bin}.rs` collides with Cargo's binary-target convention (Cargo tried to compile the storage module files as `main`-bearing binaries). Disabling binary auto-discovery preserves the path; the crate ships no binaries.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4]: CubeCL is alpha (v0.10.0) — pin exactly, isolate behind lgbm-compute; ROCm capability gaps and CPU-runtime-vs-HIP divergence need empirical validation on the local ROCm GPU (research flag: HIGH). Now evaluated against the ~1e-6 (f32) tolerance rather than bit-exactness.
- [Phase 6]: f32 transcendental (exp/log/pow/sigmoid) parity CPU↔ROCm is unproven at ~1e-6 — needs empirical validation; if a gap appears, fallback is CPU-resident objective grad/hess.
- [Cross-cutting]: RESOLVED (2026-06-05) — the strict-1e-12-vs-tiered tension is closed by adopting the f32 / ~1e-6 contract; project docs (PROJECT/REQUIREMENTS/ROADMAP) updated to match.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | QNT-01 quantized/discretized gradient training | Deferred (v2) | Roadmap |
| v2 | LIN-01 linear-tree leaves | Deferred (v2) | Roadmap |
| v2 | ING-01/02/03 text-file / binary-cache / Arrow ingestion | Deferred (v2) | Roadmap |

## Session Continuity

Last session: 2026-06-05T06:58:28Z
Stopped at: Completed 02-03-PLAN.md
Resume file: None
