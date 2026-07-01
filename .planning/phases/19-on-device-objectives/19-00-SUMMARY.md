---
phase: 19-on-device-objectives
plan: 00
subsystem: infra
tags: [cubecl, objective, lambdarank, oracle-harness, kernel-scaffold, cuda-parity]

# Dependency graph
requires:
  - phase: 18-on-device-data-partition-tree-predict
    provides: ungated kernel-module stub convention + LGBM_CUDA_ON_DEVICE seam
  - phase: 07-ranking-objectives-metrics
    provides: score-derivation oracle-capture route + rank.rs lambdarank host math
provides:
  - Four ungated compiling objective kernel stub modules (objective_{regression,binary,multiclass,rank})
  - device_objective.rs — DeviceObjectiveKind enum + device_objective_supported() host-fallback discriminator (SC #5)
  - Shared objective parity harness (objective_common/mod.rs) with capture-gated skip-pass readers
  - lambdarank_gh_iter{1,N}.txt real-lib_lightgbm grad/hess goldens (the one Wave-0 capture gap)
affects: [19-01, 19-02, 19-03, 19-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-1 mod.rs + owned-stub scaffold so parallel family plans have zero mod.rs contention"
    - "Pure host-fallback classifier (from_name -> Option<Kind>) as the on/off-device routing discriminator"
    - "Subdir tests/*/mod.rs helper module (not a test binary) shared by family test files via `mod objective_common;`"
    - "Score-derivation golden capture: re-derive grad/hess from real-lib_lightgbm raw scores via a 1:1 host-math port"

key-files:
  created:
    - crates/lgbm-compute/src/kernels/objective_regression.rs
    - crates/lgbm-compute/src/kernels/objective_binary.rs
    - crates/lgbm-compute/src/kernels/objective_multiclass.rs
    - crates/lgbm-compute/src/kernels/objective_rank.rs
    - crates/lgbm-compute/src/device_objective.rs
    - crates/oracle-harness/tests/objective_common/mod.rs
    - crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter1.txt
    - crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iterN.txt
  modified:
    - crates/lgbm-compute/src/kernels/mod.rs
    - crates/lgbm-compute/src/lib.rs
    - xtask/py/rank_oracle_capture.py
    - xtask/src/main.rs

key-decisions:
  - "device_objective_supported is a pure classifier — it does NOT enable on-device growth; on_device_growth_supported() stays false (D-02/D-06)"
  - "Kernel stub modules ungated (NOT #[cfg(feature=gpu)]) so the default cpu f64 anchor exercises them (D-08)"
  - "lambdarank grad/hess captured via the score-derivation route (RESEARCH A1), a 1:1 f32-accumulation port of rank.rs get_gradients — not the fobj fallback"

patterns-established:
  - "Pattern: one owned stub file per downstream family plan, declared together in Wave-1 to eliminate mod.rs merge contention"
  - "Pattern: capture-gated golden readers return None+SKIP so a fresh checkout builds and skip-passes without the capture run"

requirements-completed: [ODL-05, ODL-06, ODL-07, ODL-08]

# Metrics
duration: 15min
completed: 2026-07-01
status: complete
---

# Phase 19 Plan 00: On-Device Objectives Foundation Summary

**Ungated `objective_{regression,binary,multiclass,rank}` kernel stubs + the `device_objective_supported()` 11-true/7-false host-fallback discriminator (SC #5) + a shared `objective_common` parity harness + the real-lib_lightgbm `lambdarank_gh` grad/hess golden — unblocking the four Wave-2 family plans with zero mod.rs contention.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-07-01T14:39:10Z
- **Completed:** 2026-07-01T14:54:25Z
- **Tasks:** 3
- **Files created/modified:** 12 (8 created, 4 modified)

## Accomplishments
- Four ungated compiling stub kernel modules (`objective_regression/binary/multiclass/rank`) declared together in `kernels/mod.rs`, each owned by exactly one Wave-2 plan (19-01..04) — no `mod.rs` contention.
- `device_objective.rs`: `DeviceObjectiveKind` (the eleven CUDA §5 objective kinds) + `device_objective_supported()`, the SC #5 discriminator that rejects all seven CUDA-unsupported objectives (`mape`, `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`, `cross_entropy_lambda`, `map`/`rank_map`) and accepts the eleven supported. Re-exported from `lib.rs`. `on_device_growth_supported()` untouched (stays `false`).
- `objective_common/mod.rs`: shared parity harness (`parse_gh`, `read_boosting_golden`, `read_rank_golden`) with capture-gated skip-pass semantics.
- `lambdarank_gh_iter{1,N}.txt`: real-lib_lightgbm 4.6 grad/hess goldens in the boosting `*_gh` bit format, closing the one Wave-0 capture gap.

## Task Commits

Each task was committed atomically:

1. **Task 1: objective kernel stubs + mod.rs exports + device_objective support-set (SC #5)** - `f675d0c` (feat)
2. **Task 2: shared objective parity harness (objective_common)** - `fcc5317` (test)
3. **Task 3: capture lambdarank grad/hess golden (real lib_lightgbm 4.6)** - `e129350` (test)

**Plan metadata:** _(final docs commit)_

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/objective_regression.rs` - compiling stub for ODL-05 (filled by 19-01)
- `crates/lgbm-compute/src/kernels/objective_binary.rs` - compiling stub for ODL-06 (filled by 19-02)
- `crates/lgbm-compute/src/kernels/objective_multiclass.rs` - compiling stub for ODL-07 (filled by 19-03)
- `crates/lgbm-compute/src/kernels/objective_rank.rs` - compiling stub for ODL-08 (filled by 19-04)
- `crates/lgbm-compute/src/device_objective.rs` - DeviceObjectiveKind + device_objective_supported() (SC #5)
- `crates/lgbm-compute/src/kernels/mod.rs` - appended ungated Phase-19 `pub mod` block
- `crates/lgbm-compute/src/lib.rs` - `pub mod device_objective;` + re-export
- `crates/oracle-harness/tests/objective_common/mod.rs` - shared parity harness (subdir helper module)
- `crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter{1,N}.txt` - lambdarank grad/hess goldens
- `xtask/py/rank_oracle_capture.py` - lambdarank_gh score-derivation emit path
- `xtask/src/main.rs` - added the two gh fixtures to the rank-capture verification list

## Decisions Made
- `device_objective_supported` is a pure classifier that never flips routing; the phase stays OFF by default behind `LGBM_CUDA_ON_DEVICE` and `on_device_growth_supported()` stays `false` (D-02/D-06).
- Stub modules are ungated (not `#[cfg(feature="gpu")]`) so the default cubecl-cpu f64 anchor exercises them (D-08).
- lambdarank grad/hess captured via the score-derivation route (a 1:1 f32-accumulation port of `rank.rs`), not the `fobj` fallback — the derivation was clean.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added the two `lambdarank_gh` fixtures to the xtask rank-capture verification list**
- **Found during:** Task 3 (lambdarank golden capture)
- **Issue:** `rank_oracle_capture()` in `xtask/src/main.rs` asserts a set of expected output files after running the capture, but the two new gh goldens were not in that list — so a future capture run that silently failed to emit them would not be caught. `xtask/src/main.rs` was not in the plan's `files_modified`.
- **Fix:** Added `lambdarank_gh_iter1.txt` and `lambdarank_gh_iterN.txt` to the post-capture existence check. No behavior change to any other crate; `cargo build -p xtask` passes.
- **Files modified:** xtask/src/main.rs
- **Verification:** `cargo build -p xtask` succeeds; capture command re-runs byte-idempotently.
- **Committed in:** `e129350` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** The one deviation keeps the capture command self-verifying. No scope creep; no cross-plan file contention (`main.rs` is not owned by any 19-01..04 family plan).

## Issues Encountered
- **Wrong seed on first capture run.** The first `rank_oracle_capture.py` run used seed `1611157504` (a miscalculation of `0x60057000`), which regenerated the 8 committed lambdarank/rank_xendcg model fixtures with different derived sub-seeds (visible as `[seed: …]`/`feature_fraction_seed` drift). Resolved by `git checkout`-restoring the fixtures and re-running with the correct seed `0x60057000 = 1610969088`; the re-run then regenerated every existing fixture byte-identically (empty diff) and emitted only the two new gh files. Byte-idempotency re-confirmed by a second run (identical sha1).
- **Workspace test parallel-link race.** `cargo test --workspace` at full parallelism produced transient `ld`/`could not write output: No such file or directory` errors across unrelated crates (polars-core) plus a pre-existing `getrandom`/`regex_automata` rlib quirk in the `bench_crossover` example. Re-running `cargo test --workspace --lib --tests -j 4` (reduced link concurrency, excluding examples/benches) was fully green: 64 test binaries ok, zero failures. Not caused by this plan's changes (no new deps, no build-config change); `lgbm-compute --lib` alone passes 100 tests.

## User Setup Required
None - no external service configuration required. (Task 3 used the existing repo-root uv `.venv` with `lightgbm==4.6.0`; no new dependency installed.)

## Next Phase Readiness
- Wave-2 family plans (19-01..04) are unblocked: each owns exactly one kernel stub file (`objective_*.rs`) and one parity test file, all declared/scaffolded here with no shared-file contention.
- `device_objective_supported()` is available (re-exported from `lgbm_compute`) as the SC #5 routing discriminator for the family plans and any future boosting-seam wiring.
- Default host boosting path is byte-unchanged; `LGBM_CUDA_ON_DEVICE` stays OFF.

## Self-Check: PASSED

All 8 created files exist on disk; all 3 task commits (`f675d0c`, `fcc5317`, `e129350`) present in git history.

---
*Phase: 19-on-device-objectives*
*Completed: 2026-07-01*
