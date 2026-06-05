---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 02-01-PLAN.md
last_updated: "2026-06-05T06:30:19Z"
last_activity: 2026-06-05 -- Completed Plan 02-01 (numeric binning determinism root)
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 8
  completed_plans: 4
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05)

**Core value:** For identical inputs and config, reproduce C++ LightGBM outputs to within ~1e-6 absolute difference on every backend (CPU and ROCm), using f32 (single-precision) data types matching the C++ reference defaults.
**Current focus:** Phase 02 — dataset-binning-determinism-root

## Current Position

Phase: 02 (dataset-binning-determinism-root) — EXECUTING
Plan: 2 of 5
Status: Executing Phase 02
Last activity: 2026-06-05 -- Completed Plan 02-01

Progress: [██░░░░░░░░] 20% (1 of 5 Phase-02 plans complete)

### Resume

Phase 02 Plan 01 complete: the numeric binning determinism root is locked. Next: Plan 02-02 (Bin trait + DenseBin/SparseBin storage + FeatureGroup offsets + Dataset finish_load immutability).

- `lgbm-dataset` crate exists with `BinMapper` (numeric `find_bin`/`value_to_bin`, bit-exact f64 boundary kernel), `BinType`/`MissingType`, and `DatasetError`.
- `bin-capture` xtask subcommand + `xtask/cpp/bin_capture.cpp` emit numeric goldens (layers 1+2); oracle-harness has `compare_exact_u32`/`compare_exact_f64_bits`/`compare_exact_bytes`; later plans plug in here.
- 45-case golden replay proves `bin_upper_bound_` + per-row indices bit-identical to C++ (SC#1, SC#5); regen idempotent.
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
| 02-dataset-binning-determinism-root | 1/5 | ~12 min | ~12 min |

**Plan 01-02:** 3 tasks, 11 files (9 created + 2 modified), 29 new tests; `cargo test --workspace` green.
**Plan 01-03:** 2 TDD tasks, 3 files modified, 7 new tests (49 → 56); deterministic alias resolution + empty==absent reads; `cargo test --workspace` green.
**Plan 02-01:** 4 tasks (~12 min), 14 files (9 created + 5 modified), numeric BinMapper kernel + bin-capture harness + 45-case golden replay (layers 1+2 bit-exact); `cargo test --workspace` green.

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

Last session: 2026-06-05T06:30:19Z
Stopped at: Completed 02-01-PLAN.md
Resume file: None
