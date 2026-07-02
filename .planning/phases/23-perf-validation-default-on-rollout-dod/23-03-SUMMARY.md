---
phase: 23-perf-validation-default-on-rollout-dod
plan: 03
subsystem: infra
tags: [kaggle, ab-harness, cuda, on-device, benchmark, parity, device-launches, dod-evidence]

# Dependency graph
requires:
  - phase: 23-perf-validation-default-on-rollout-dod (plan 01)
    provides: LGBM_CUDA_ON_DEVICE tri-state off-switch — the host arm sets "0", on-device arm sets "1"
  - phase: 23-perf-validation-default-on-rollout-dod (plan 02)
    provides: non-zero on-device device_launches= COUNTS line (on_device= folded inside parens) — the harness parse target
provides:
  - Committed, credential-free Kaggle A/B harness (ab_harness.py) — 3x2x2 matrix, medians, ratios, device_launches parse, D-11 parity, unconditional results.{md,json} emit, non-zero exit on FAIL
  - Importable pure parse/median/verdict/parity-envelope functions (locally testable without Kaggle/GPU)
  - Captured phase_prof stderr fixture (host 8570 + on-device collapsed) as a faithful local parse target
  - Local dry-run test proving the COUNTS regex + median + BOTH verdict branches + near-tie envelope
affects: [23-04-verdict-gated-flip, ODL-20, ODL-21, SC-2]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-arm subprocess with fresh env (LGBM_CUDA_ON_DEVICE + LGBM_PHASE_PROF) so the OnceLock tri-state resolver is set BEFORE process start (P-1) and each arm's phase_prof COUNTS is captured on its own stderr"
    - "Crash-safe evidence emit: matrix/parity flow wrapped in try/finally with a pessimistic FAIL default so results.{md,json} land unconditionally (D-09) even on a run/parity crash"
    - "Parity as a captured boolean (parity_ok), never a bare inline assert — folded into the verdict AFTER results are stored, under a documented near-tie envelope with the cpu-f64 anchor as primary proof (W1/W2, def-f8u-01)"
    - "SHORT total-only device_launches regex (W3 producer/consumer contract) — captures device_launches=<total> regardless of the parenthesized on_device= breakdown"

key-files:
  created:
    - .planning/phases/23-perf-validation-default-on-rollout-dod/ab_harness.py
    - .planning/phases/23-perf-validation-default-on-rollout-dod/fixtures/sample_phase_prof.stderr
    - .planning/phases/23-perf-validation-default-on-rollout-dod/test_ab_harness_parse.py
  modified: []

key-decisions:
  - "Heavy numpy/lightgbm imports deferred into the Kaggle-only run path; only stdlib at module scope so ab_harness.py imports cleanly on any machine and the local test can import the pure functions"
  - "Each of the 6 arms per (shape,run) is a separate subprocess (P-1) — the tri-state env is process-frozen by OnceLock, so per-arm env cannot be flipped in-process"
  - "Parity worst-case folded across all runs/shapes into ONE gate (max over |p_on - p_host|); official lightgbm reported as context-only max(|p_on - p_official|), never a same-tol gate (f32 reference)"
  - "Fixture uses the EXACT phase_prof.rs post-23-02 format (on_device= inside parens) so the SHORT regex matches the fixture identically to real Kaggle stderr"

patterns-established:
  - "DoD evidence harness pattern: importable pure logic (parse/median/verdict/envelope) + a Kaggle-only orchestrator + a committed stderr fixture + a local dry-run test = the measurement is de-risked and the parser cannot silently regress without Kaggle"

requirements-completed: [ODL-20]

# Metrics
duration: ~12min
completed: 2026-07-03
status: complete
---

# Phase 23 Plan 03: Kaggle A/B Harness (device_launches + wall-clock + parity DoD evidence) Summary

**A committed, credential-free Kaggle A/B harness (`ab_harness.py`) that runs the D-05 3x2x2 matrix (3 runs x {500kx50, 100kx500} x {host-cuda, on-device}) with the 23-01 tri-state off-switch/on-device arms, parses `device_launches` via the SHORT total-only regex, computes the D-04 (<=5%) / D-05 (both-shapes) verdict folding in the D-11 real-CUDA parity gate under a near-tie envelope (cpu-f64 anchor primary), and emits `results.{md,json}` UNCONDITIONALLY on a finally path (D-09) then exits non-zero on FAIL — plus a captured stderr fixture and a local dry-run test that prove the parse + verdict logic without Kaggle or a GPU.**

## Performance
- **Duration:** ~12 min
- **Completed:** 2026-07-03
- **Tasks:** 3
- **Files created:** 3 (0 modified)

## Accomplishments
- Authored `ab_harness.py` extending the `continue_benchmark.py` build flow (pinned clone of `BectorVoom/lightgbm_rs`, `maturin build --release -F cuda`, wheel install, official lightgbm `--no-binary` USE_CUDA=ON) — with NO credential literal (T-23-03) and from-source builds only (T-23-03-SC).
- Runs the 3x2x2 matrix as per-arm subprocesses with fresh env (`LGBM_CUDA_ON_DEVICE=0` host / `=1` on-device, both `LGBM_PHASE_PROF=1`) so the OnceLock tri-state resolver is frozen correctly per arm (P-1); captures stdout wall-clocks + stderr COUNTS.
- Parses `device_launches` with the SHORT total-only regex `device_launches=(?P<launches>\d+)` (W3), optionally capturing the `on_device=` sub-field; computes `device_launches/tree = launches/100` vs the 85.7 host baseline (SC-2).
- Computes the verdict via importable pure functions: median of 3 wall-clocks, PASS iff both shapes `median(on_device) <= 1.05 * median(host_cuda)` AND `parity_ok`.
- D-11 parity is CAPTURED into `max_abs_on_host` and folded into a boolean `parity_ok` (never a bare inline assert — W1) under the documented envelope `atol=1e-6, rtol=1e-6` (W2, def-f8u-01), with a `results.md` note stating the cpu-f64 structural anchor is the primary parity proof and the real-CUDA number is corroboration.
- Emits `results.{md,json}` on a `finally`/before-exit path with a pessimistic FAIL default so the committed evidence artifact ALWAYS lands (D-09), then exits non-zero on FAIL.
- Added `fixtures/sample_phase_prof.stderr` (host `device_launches=8570 on_device=0`, on-device `device_launches=2900 on_device=2900`) matching the exact phase_prof.rs post-23-02 format.
- Added `test_ab_harness_parse.py` — imports the real functions, proves both fixture parses, launches/tree, the median helper, BOTH verdict branches (PASS, FAIL-on-slowdown, FAIL-on-parity), and the near-tie envelope (boundary true / divergence false). 8/8 green, no pytest needed.

## Task Commits
1. **Task 1: Committed Kaggle A/B harness (matrix + parse + parity + results emit)** - `23ba101` (feat)
2. **Task 2: Capture a phase_prof stderr fixture for local parser verification** - `8691988` (test)
3. **Task 3: Local dry-run parse + verdict test over the fixture** - `00dcb9d` (test)

## Files Created/Modified
- `.planning/phases/23-.../ab_harness.py` - the Kaggle A/B harness: importable parse/median/verdict/parity-envelope pure functions + a Kaggle-only orchestrator (setup build, run matrix, unconditional results emit, non-zero exit on FAIL).
- `.planning/phases/23-.../fixtures/sample_phase_prof.stderr` - captured phase_prof COUNTS fixture (host 8570 / on-device collapsed 2900), faithful post-23-02 format target for the SHORT regex.
- `.planning/phases/23-.../test_ab_harness_parse.py` - local dry-run test (8 assertions) proving the parser + verdict + envelope without Kaggle/GPU.

## Decisions Made
- Deferred heavy numpy/lightgbm imports into the run path so the module imports cleanly anywhere and the local test imports only the pure logic.
- One subprocess per arm (P-1) — the tri-state env is OnceLock-frozen per process, so per-arm env cannot be flipped in-process.
- Parity worst-case folded across all runs/shapes into ONE gate; official lightgbm reported context-only (never a same-tol gate — f32 reference).

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None. The harness itself is Kaggle-only (real discrete CUDA); the local GPU is a spoofed 8-CU APU, so the parse/verdict logic is de-risked via the committed fixture + local test instead.

## Manual inter-wave step (Wave 2 -> Wave 3, D-07 evidence commit)
The real-CUDA A/B run is an explicit out-of-band hand-off before the 23-04 checkpoint (not auto-runnable locally). A human must: (1) push/run `ab_harness.py` on Kaggle as `boomvector`, (2) poll with `poll_kaggle.sh` and pull the output, (3) extract the emitted `results.{md,json}` into the phase dir, (4) commit them as the D-07 first-class DoD evidence artifact — even on a FAIL verdict (D-09).

## Next Phase Readiness
- ODL-20 harness is committed and locally proven: `ab_harness.py` + fixture + `test_ab_harness_parse.py` (8/8 green). The parser/verdict logic cannot silently regress without Kaggle.
- 23-04 (verdict-gated default-on flip) can gate on the committed `results.{md,json}` once the manual Kaggle run lands them; the harness exits non-zero on FAIL to make the verdict machine-checkable.

## Self-Check: PASSED

All three created files present; all task commits (23ba101, 8691988, 00dcb9d) in git history; `test_ab_harness_parse.py` runs 8/8 green (exit 0); `ast.parse` on `ab_harness.py` is `parse-ok`; credential grep empty.

---
*Phase: 23-perf-validation-default-on-rollout-dod*
*Completed: 2026-07-03*
