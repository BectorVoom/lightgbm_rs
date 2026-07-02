---
phase: 23-perf-validation-default-on-rollout-dod
plan: 02
subsystem: infra
tags: [profiling, phase_prof, on-device, cuda, launch-count, instrumentation, atomics]

# Dependency graph
requires:
  - phase: 21-end-to-end-driver-integration
    provides: grow_tree_on_device_driver — the on-device per-leaf best-first grow driver whose real build/subtract/scan dispatch sites are instrumented here
  - phase: 23-perf-validation-default-on-rollout-dod (plan 01)
    provides: cuda_on_device_enabled() single-source routing gate that decides when the on-device path (and thus this counter) runs
provides:
  - Compute-owned ON_DEVICE_LAUNCH_CNT AtomicU64 + on_device_launch_count_take() accessor in lgbm-compute (no crate cycle)
  - bump_launch() at the on-device driver's real build/subtract/scan device dispatches, gated inert on LGBM_PHASE_PROF
  - phase_prof COUNTS line folds on_device into the device_launches= total + an on_device= sub-field (unsuppresses the line for on-device trains, P-2)
  - Local instrumentation test proving a non-zero, sub-8570-baseline launch count from an on-device grow
affects: [23-03-kaggle-ab-harness, 23-04-verdict-gated-flip, ODL-20, SC-2]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cross-crate profiling counter: a compute-owned AtomicU64 + swap-to-0 take accessor lets lgbm-treelearner's phase_prof read on-device launch counts without a treelearner→compute→treelearner cycle"
    - "Read-once LGBM_PHASE_PROF OnceLock gate mirrored locally in lgbm-compute so the bump is inert/zero-overhead in the default merge gate (parity-neutral)"

key-files:
  created:
    - crates/lgbm-compute/tests/on_device_launch_count.rs
  modified:
    - crates/lgbm-compute/src/kernels/grow_driver.rs
    - crates/lgbm-treelearner/src/phase_prof.rs

key-decisions:
  - "Counter lives in lgbm-compute (owner of the dispatch sites); phase_prof reads it via a public take() accessor — respects the crate DAG (phase_prof is ABOVE lgbm-compute)"
  - "on_device= sub-field placed INSIDE the parenthesized breakdown so the 23-03 harness short regex device_launches=(?P<launches>\\d+) keeps capturing the total unchanged (Open-Q2)"
  - "Bumps count actual device dispatches (per-feature build/scan, per-child subtract), not logical leaves, so the number is comparable to the host 8,570/100-trees baseline denominator (D-08)"

patterns-established:
  - "Out-of-band launch counting: the driver return signature Result<(Tree, LeafPartitionLayout), ComputeError> is untouched; the count flows via the compute-owned static — no ripple into Backend::grow_tree_on_device"

requirements-completed: [ODL-20]

# Metrics
duration: ~16min
completed: 2026-07-02
status: complete
---

# Phase 23 Plan 02: On-Device Launch-Count Instrumentation Summary

**A compute-owned, phase-prof-gated launch counter bumped at the on-device driver's real build/subtract/scan dispatches, folded into the phase_prof `device_launches=` COUNTS line so an on-device train reports a visible non-zero launch count (was suppressed at 0) — making ODL-20/SC-2's launch collapse measurable for the Kaggle A/B harness.**

## Performance

- **Duration:** ~16 min
- **Started:** 2026-07-02T20:28Z (approx)
- **Completed:** 2026-07-02T20:44Z
- **Tasks:** 3
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- Added `ON_DEVICE_LAUNCH_CNT: AtomicU64` + `on_device_launch_count_take()` (swap-to-0) + inline `bump_launch()` in lgbm-compute's `grow_driver.rs`, gated on a read-once `LGBM_PHASE_PROF=="1"` OnceLock (inert/zero-overhead by default).
- Instrumented the three real on-device device dispatches: histogram build (`construct_histograms_f64_on`), subtraction-trick derive (`subtract_histograms_f64_on`), and per-leaf scan (`find_best_split_f64_on`).
- Extended `phase_prof::dump` to fold the on-device count into the `device_launches=` total (guard now fires on `... + on_dev > 0`) plus an `on_device=` sub-field inside the parenthesized breakdown — keeping the harness's short total-capture regex stable.
- Added `crates/lgbm-compute/tests/on_device_launch_count.rs` proving an on-device grow bumps a non-zero count far below the 8,570/100-trees host baseline.

## Task Commits

Each task was committed atomically:

1. **Task 1: Compute-owned launch counter + driver instrumentation (L-1)** - `53adcca` (feat)
2. **Task 2: Surface the on-device launch count in the COUNTS line (phase_prof)** - `5ae7f85` (feat)
3. **Task 3: Local test — on-device grow bumps a non-zero launch count** - `ee3858f` (test)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/grow_driver.rs` - Compute-owned launch counter (static + take accessor + gated bump_launch) and bumps at the build/subtract/scan dispatch sites.
- `crates/lgbm-treelearner/src/phase_prof.rs` - COUNTS `dump()` reads `lgbm_compute::kernels::grow_driver::on_device_launch_count_take()`, folds it into the `device_launches=` total, and emits an `on_device=` sub-field.
- `crates/lgbm-compute/tests/on_device_launch_count.rs` - cubecl-cpu instrumentation test: non-zero, sub-baseline launch count from a tiny on-device grow.

## Decisions Made
- Placed the counter in lgbm-compute rather than importing phase_prof (which would form a crate cycle); the learner (above compute) reads it via the public `take()` accessor. This is the ONLY workable direction given the DAG.
- Kept `device_launches=<total>` as the first COUNTS field and put `on_device=` inside the parens, per the W3 producer/consumer contract: the 23-03 consumer uses the short total-only regex, not the full paren-terminated one.
- Counted actual dispatches (per-feature build/scan) rather than logical leaves so the value is comparable to the host baseline denominator.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. Edition 2024 requires `unsafe { std::env::set_var(...) }`; the test sets `LGBM_PHASE_PROF=1` at the very top of the single-test process before the first OnceLock read (P-1), documented with a SAFETY note.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ODL-20 instrumentation is in place and green: an on-device train now emits a non-zero `device_launches=` line with an `on_device=` breakdown. 23-03 (Kaggle A/B harness) can capture the real magnitude via the short `device_launches=(?P<launches>\d+)` regex and compare against the 8,570/100-trees host baseline for SC-2.
- Merge gate green with prof off — `cargo test --workspace` passes, counter inert/byte-unchanged (SC-4).
- Threat T-23-02 mitigated: counter is inert unless `LGBM_PHASE_PROF=="1"`, no effect on tree structure/values.

## Self-Check: PASSED

All created/modified files present; all task commits (53adcca, 5ae7f85, ee3858f) in git history.

---
*Phase: 23-perf-validation-default-on-rollout-dod*
*Completed: 2026-07-02*
