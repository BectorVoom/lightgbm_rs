---
status: complete
phase: 23-perf-validation-default-on-rollout-dod
source: [23-01-SUMMARY.md, 23-02-SUMMARY.md, 23-03-SUMMARY.md, 23-04-SUMMARY.md]
started: 2026-07-03T05:43:55Z
updated: 2026-07-03T05:48:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Tri-state on-device env toggle
expected: `LGBM_CUDA_ON_DEVICE` is tri-state (unset=>default, "0"=>force-off, "1"=>force-on), single source of truth. `cargo test -p lgbm-compute --test cuda_on_device` → 3 passed.
result: pass

### 2. On-device launch-count instrumentation
expected: An on-device grow bumps a non-zero, sub-8570-baseline `device_launches=` count, surfaced in the phase_prof COUNTS line (with `on_device=` sub-field) only under `LGBM_PHASE_PROF=1`; inert/zero-overhead otherwise. `cargo test -p lgbm-compute --test on_device_launch_count` → passes.
result: pass

### 3. Kaggle A/B harness parse + verdict logic
expected: The A/B harness's pure parse/median/verdict/parity-envelope functions work without Kaggle or a GPU — SHORT `device_launches` regex, launches/tree, median, both verdict branches (PASS, FAIL-on-slowdown, FAIL-on-parity), near-tie envelope. `python3 test_ab_harness_parse.py` → 8/8 assertions pass, exit 0.
result: pass
note: Ran by Claude — 8/8 passed, exit 0.

### 4. DoD evidence artifact + no-flip decision
expected: `results.md` and `results.json` exist recording the real-CUDA A/B verdict **FAIL** with the numbers-not-captured caveat; the behavior-changing default-on flip was WITHHELD — `on_device_default()` stays `false`, on-device stays opt-in via `LGBM_CUDA_ON_DEVICE=1`. `crates/lgbm-compute/` is unchanged by plan 04 (flip not made).
result: pass
note: Ran by Claude — results.{md,json} verdict FAIL, flip_performed=false, on_device_default()=false stub, dod_complete=true.

### 5. Byte-unchanged merge gate (SC-4, default off)
expected: With the env unset, the full workspace suite is green and every backend is byte-unchanged — the routing + instrumentation are parity-neutral and the default remains host-CUDA / cpu-default-off. `cargo test --workspace` → all pass.
result: pass

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
