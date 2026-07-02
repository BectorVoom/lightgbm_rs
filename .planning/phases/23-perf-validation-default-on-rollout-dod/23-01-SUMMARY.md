---
phase: 23-perf-validation-default-on-rollout-dod
plan: 01
subsystem: infra
tags: [routing, feature-flags, env-toggle, cuda, on-device, tri-state]

# Dependency graph
requires:
  - phase: 20-score-updater-metrics
    provides: "on_device_growth_supported() discriminator returning cuda_on_device_enabled() on both CpuBackend and GpuBackend"
  - phase: 14-foundation
    provides: "learner on_device_eligible AND-gate + duplicate cuda_on_device_env() parse"
provides:
  - "Tri-state LGBM_CUDA_ON_DEVICE resolver in lgbm-compute (unset=>default, \"0\"=>force-off, \"1\"=>force-on)"
  - "on_device_default() compile-time helper (returns false this plan; 23-04 flips to cfg!(feature=cuda))"
  - "Single source of truth for on-device eligibility — learner duplicate parse removed (L-2)"
  - "Pure cuda_on_device_override_from() mapping helper + unit tests"
affects: [23-04-verdict-gated-flip, 23-02, 23-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tri-state env toggle: OnceLock<Option<bool>> override cache + compile-time device-default helper, resolved as override.unwrap_or_else(default)"
    - "Pure mapping helper (#[doc(hidden)] pub) as the OnceLock initializer's core so the closed-enum mapping is unit-testable without fighting read-once caching (P-1)"

key-files:
  created:
    - crates/lgbm-compute/tests/cuda_on_device.rs
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs

key-decisions:
  - "on_device_default() returns literal false this plan (D-09 pre-verdict safe state) — byte-unchanged on every backend; 23-04 owns the verdict-gated flip"
  - "cuda_on_device_enabled() name/signature kept unchanged so score_updater + both on_device_growth_supported() impls inherit tri-state semantics with zero call-site edits (D-02, Pattern 3)"
  - "Resolver NOT cpu-feature-gated (unlike split_2lane_enabled) because cuda/rocm builds resolve it too"

patterns-established:
  - "Tri-state closed-enum env parse: exact-string match only (\"1\"/\"0\"/else), no eval/path/format interpretation (ASVS V5, T-23-01)"
  - "Single-source-of-truth env resolution: the compute-layer resolver is the only parse; learner/boosting read through on_device_growth_supported()/cuda_on_device_enabled()"

requirements-completed: [ODL-21]

# Metrics
duration: 4min
completed: 2026-07-02
status: complete
---

# Phase 23 Plan 01: Tri-state routing + single-source on-device eligibility Summary

**Converted `LGBM_CUDA_ON_DEVICE` from a binary `"1"`-only parse to a tri-state resolver (unset=>device default, `"0"`=>force-off, `"1"`=>force-on) with a compile-time `on_device_default()` stub returning `false`, and collapsed the learner's duplicate env parse so all three toggle sites share one source of truth — every backend byte-unchanged.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-07-02T20:32:55Z
- **Completed:** 2026-07-02T20:37:00Z
- **Tasks:** 3
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- Tri-state resolver: `cuda_on_device_override()` (OnceLock<Option<bool>>) + `on_device_default()` (false stub, D-09) + `cuda_on_device_enabled()` = `override.unwrap_or_else(default)`, name/signature preserved
- `"0"` is now a first-class force-off (the off-switch fallback), distinct from unset — the old binary parse silently treated `"0"` the same as unset
- Learner duplicate `cuda_on_device_env()` deleted; both call sites (`on_device_eligible` ctor + `refresh_on_device_eligibility` mirror) now gate on `backend.on_device_growth_supported()` alone (L-2)
- Pure `cuda_on_device_override_from()` mapping helper makes the closed-enum contract unit-testable without fighting OnceLock read-once semantics (P-1)
- Full workspace suite green (969 tests) with the env unset — SC-4 byte-unchanged merge gate upheld

## Task Commits

Each task was committed atomically:

1. **Task 1: Tri-state resolver + on_device_default() stub in lgbm-compute** - `2020a1a` (feat)
2. **Task 2: Unit tests for tri-state resolver + on_device_default per-feature** - `f671230` (test)
3. **Task 3: Reconcile the learner's duplicate parse (L-2, single source of truth)** - `699874c` (refactor)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `crates/lgbm-compute/src/lib.rs` - Replaced binary `cuda_on_device_enabled()` body with tri-state resolver; added `cuda_on_device_override_from()` (pure mapping, `#[doc(hidden)] pub`), `cuda_on_device_override()` (OnceLock cache), `on_device_default()` (false stub with D-09/P-3 doc)
- `crates/lgbm-compute/tests/cuda_on_device.rs` - New: tri-state mapping unit tests (`"1"`/`"0"`/unset/empty/`"2"`/`"true"`) + `"0"` force-off distinctness + `#[cfg(not(feature="cuda"))]` cpu-build default-off anchor
- `crates/lgbm-treelearner/src/learner.rs` - Deleted `fn cuda_on_device_env()`; dropped `&& cuda_on_device_env()` at both eligibility sites; updated struct doc comment referencing the removed parse

## Decisions Made
- None beyond the plan — followed D-01/D-02/D-09 as specified. `on_device_default()` intentionally returns literal `false` this plan; the verdict-gated flip to `cfg!(feature="cuda")` is 23-04's responsibility (P-3 dual/mono-feature caveat documented inline for that flip).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `cargo test -p lgbm-compute cuda_on_device` as a name-filter selected 0 tests in the `cuda_on_device` binary (filter matched other binaries' 0-test runs); confirmed the new tests pass via `cargo test -p lgbm-compute --test cuda_on_device` (3 passed). No code change needed — the acceptance intent (tests run and pass, selectable by the file) is met.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Routing machinery for the default-on rollout (ODL-21) is landed and byte-unchanged. Single source of truth (`lgbm_compute::cuda_on_device_enabled()`) is in place for the compute discriminator, learner eligibility gate, and boosting `boosting_on_cuda`.
- 23-04 flips `on_device_default()` to `cfg!(feature="cuda")` after the Kaggle A/B PASS verdict; the `"0"` off-switch is the documented escape hatch for that flip.
- No blockers.

## Self-Check: PASSED

All 3 files present (2 modified, 1 created) and all 3 task commits (`2020a1a`, `f671230`, `699874c`) exist in git history.

---
*Phase: 23-perf-validation-default-on-rollout-dod*
*Completed: 2026-07-02*
