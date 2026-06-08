---
phase: 260608-s2b-fuse-gpu-build-kernels-fold-f32-f64-wide
plan: 01
subsystem: infra
tags: [cubecl, rocm, gpu, histogram, gfx1100, perf, device-resident, f32-f64-widen]

# Dependency graph
requires:
  - phase: 260608-p90
    provides: device-resident per-leaf histogram pool (build→fix→compact→subtract→scan resident) + resident_eligible gate
  - phase: 260608-oib
    provides: on-GPU fix_compact_kernel (L3) + the bit-exact fix_compact oracle
provides:
  - "Folded widen+fix+compact GPU kernel — resident per-leaf build is 2 launches (construct + folded fix), not 3"
  - "num_data size-gate on resident_eligible (RESIDENT_MIN_NUM_DATA=12000) with a LGBM_RESIDENT_FORCE bench knob"
  - "Measured resident-vs-host crossover on gfx1100 (host wins ≤8000 rows, resident wins ≥20000)"
affects: [device-resident histogram pool, future GPU perf work, rocm parity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fold an adjacent device kernel's work into a one-cube-per-feature kernel's first pass to drop a GPU launch while staying bit-identical (same cast, same fold order)"
    - "Data-driven size-gated dispatch between two proven-equivalent paths, with an env override for benching both from one binary"

key-files:
  created:
    - .planning/quick/260608-s2b-fuse-gpu-build-kernels-fold-f32-f64-wide/T0-baseline.md
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-treelearner/src/resident_pool.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/kernel_parity.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "Lever A folds the widen as the fix kernel's inline first pass (f64::cast_from per feature region) — bit-identical to the prior 3-launch widen-then-fix, proven by compare_exact_f64_bits oracle"
  - "RESIDENT_MIN_NUM_DATA=12000 set from the measured (8000, 20000] crossover bracket middle — host wins small+medium, resident wins large"
  - "learner_parity_resident_equals_host_tree_on_hip forces LGBM_RESIDENT_FORCE=1 so its 3000-row corpus still exercises the resident chain past the new size gate"

patterns-established:
  - "Launch-fold pattern: widen kernel removed; its cast becomes the consumer kernel's first per-region pass"
  - "Perf-knob size gate after fail-safe correctness checks; env override bypasses ONLY the size threshold, never a correctness gate"

requirements-completed: [PERF-GPU-LAUNCH]

# Metrics
duration: 38min
completed: 2026-06-08
---

# Phase 260608-s2b: Fuse GPU build kernels — fold f32→f64 widen + size-gate resident Summary

**Folded the standalone f32→f64 widen launch into the on-GPU fix_compact kernel (resident per-leaf build 3 launches → 2, bit-identical) and added a measured num_data size-gate (12000) so the launch-bound small/medium workloads take the host path and only large takes the resident path.**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-06-08
- **Completed:** 2026-06-08
- **Tasks:** 4 (T0 baseline, T1 Lever A, T2 Lever B, T3 sweep+write-up)
- **Files modified:** 5

## Accomplishments

- **Lever A (widen-fold, 3→2 launches):** `fix_compact_kernel` now takes the f32 RAW histogram as INPUT plus a zeroed f64 OUTPUT, widens each feature region inline via `f64::cast_from` (the identical cast the standalone `widen_f32_to_f64_kernel` performed) as its first pass, then runs the unchanged in-place f64 FixHistogram + compact. The standalone widen kernel + its launch are removed. The resident per-leaf build (`build_fix_compact_resident_f64_on`) now issues exactly 2 launches: `construct_leaf_hist_resident_kernel` + the folded `fix_compact_kernel`.
- **Bit-identity proven:** `kernel_parity_fix_compact_equals_host_on_hip` (compare_exact_f64_bits) GREEN for the folded kernel; `kernel_parity_resident_build_fix_compact_equals_host_on_hip` GREEN (within ~1e-6). Same cast + same ascending fold order ⇒ identical f64 bits.
- **Lever B (data-driven size-gate):** `resident_eligible` gained a `num_data` parameter and `RESIDENT_MIN_NUM_DATA = 12000`. Below the threshold the workload takes the byte-unchanged host path; at/above it takes the resident path. A `LGBM_RESIDENT_FORCE` env knob (0=host, 1=resident) overrides ONLY the size threshold (after every fail-safe correctness check), letting the bench measure both paths from one binary.
- **Crossover measured on gfx1100** (after Lever A, bench_train both ways): host wins at 2000 and 8000 rows, resident wins at 20000 rows; threshold set in the bracket middle.

## Task Commits

1. **T0 — capture BEFORE baseline** — no source commit (scratch note `T0-baseline.md`).
2. **T1 — fold f32→f64 widen into fix_compact (3→2 launches)** — `94312e4` (perf)
3. **T2 — size-gate resident_eligible on num_data** — `43b256c` (perf)
4. **T3 — final gate sweep + write-up** — no source fixup needed; SUMMARY only (orchestrator commits docs).

## Files Created/Modified

- `crates/lgbm-compute/src/kernels/histogram.rs` — folded `fix_compact_kernel` (f32 RAW in + f64 out, inline widen first pass); removed `widen_f32_to_f64_kernel` + its launch from `build_fix_compact_resident_f64_on`; updated `fix_compact_f64_on` launcher to feed f32 RAW. **All rocm-gated.**
- `crates/lgbm-treelearner/src/resident_pool.rs` — `RESIDENT_MIN_NUM_DATA` const (with measured provenance table), `num_data` param + size gate + `LGBM_RESIDENT_FORCE` override in `resident_eligible`.
- `crates/lgbm-treelearner/src/learner.rs` — pass `num_data` into the `resident_eligible` call.
- `crates/oracle-harness/tests/kernel_parity.rs` — `kernel_parity_fix_compact_equals_host_on_hip` drives the folded kernel with an f32 RAW buffer + f64-widened host reference (kept compare_exact_f64_bits teeth).
- `crates/oracle-harness/tests/learner_parity.rs` — `learner_parity_resident_equals_host_tree_on_hip` forces `LGBM_RESIDENT_FORCE=1` around the resident train so the 3000-row corpus still exercises the resident path past the 12000 gate.

## Launch count 3 → 2 (evidence)

Confirmed via **code inspection** (cheap, no extra profiling hook — a per-tree profiling call inflates timings, so none was left in). `build_fix_compact_resident_f64_on` now contains exactly two `::launch` sites:
- `construct_leaf_hist_resident_kernel::launch` (stage 1, f32-atomic RAW build) — UNCHANGED.
- `fix_compact_kernel::launch` (stage 3, folded widen+fix+compact) — folded.

The prior stage-2 `widen_f32_to_f64_kernel::launch` is removed; the `widen_f32_to_f64_kernel` definition is deleted (dead after the fold). Per-leaf spine launches: 3 → 2.

## Chosen threshold + both-ways crossover numbers

`RESIDENT_MIN_NUM_DATA = 12000`. Measured AFTER Lever A on the local gfx1100, `bench_train` (iters 100, leaves 31, train_median of 5, two runs each), both ways via `LGBM_RESIDENT_FORCE`:

| rows  | FORCE_HOST (=0) | FORCE_RESIDENT (=1) | winner   |
|-------|-----------------|---------------------|----------|
| 2000  | 1.50 / 1.43s    | 1.64 / 1.66s        | HOST     |
| 8000  | 4.33 / 4.18s    | 4.68 / 4.89s        | HOST     |
| 20000 | 11.95 / 12.11s  | 11.50 / 11.73s      | RESIDENT |

Crossover lies in the (8000, 20000] bracket. The threshold is set at the bracket middle (12000) so small+medium route to the host winner and large routes to the resident winner. The knob is also retained as a safety valve (a future tiny-row regression falls back to host without a code change).

## Before/after train_median (small no-regression; medium/large not harmed)

| size   | rows  | T0 baseline (resident, p90) | Lever A (resident) | s2b default (size-gated) |
|--------|-------|-----------------------------|--------------------|--------------------------|
| small  | 2000  | 1.60s                       | 1.46 / 1.42s       | 1.55 / 1.51s (host path) |
| medium | 8000  | 4.98s                       | 4.89 / 4.67s       | 4.25 / 4.17s (host path) |
| large  | 20000 | 11.14s                      | 11.54 / 11.55s     | 11.51 / 11.65s (resident)|

- **small: no regression** — 1.51-1.55s default vs 1.60s T0 (and vs 1.64-1.66s if forced resident). The p90 −13% small regression is gone.
- **medium: improved** — 4.17-4.25s default vs 4.98s T0 (host path is the measured winner here too).
- **large: not harmed** — 11.51-11.65s on the resident path (its own measured winner; FORCE_HOST was 11.95-12.11s). The +3% vs the single noisy T0 reading (11.14s) is within the ±0.4s run-to-run GPU noise; the gate correctly routes large to its faster path.

## Bit-exact gate status

- **CPU bit-exact (default `cargo test --workspace`):** GREEN — 59 ok suites, 0 failed. kernel_parity 6/6, learner_parity 29/29, boosting_parity 75/75 (incl. `mfb_zero_offset_histogram_contract`, `goss_parity_matrix`).
- **hip oracles (`--features rocm`):**
  - `kernel_parity_fix_compact_equals_host_on_hip` — GREEN (BIT-EXACT, folded kernel).
  - `kernel_parity_resident_build_fix_compact_equals_host_on_hip` — GREEN (within ~1e-6).
  - `find_best_splits_batched_from_handle_equals_host_buf_on_hip` + resident_gather + partition + subtract/histogram within-tol — GREEN.
  - `learner_parity_resident_equals_host_tree_on_hip` — passes intermittently (see Known Issues; pre-existing flaky borderline, NOT worsened).
- **CPU anchor byte-unchanged:** `git diff b8a7080 HEAD` touches only the rocm-gated kernel, resident_pool, the learner call site, and two oracles. NO hunks in `lgbm-treelearner/src/fix_histogram.rs`, host `compact_histogram`, CpuBackend, or host-scan. Verified.

## Decisions Made

- **Folded-widen as the fix kernel's inline first pass** (vs. an in-place re-widen): cleanest path to bit-identity. Each cube widens its own feature region from f32 RAW into the zeroed f64 output, then the existing in-place f64 fix+compact runs over that output. Because `fix_feats` enumerates every feature and the regions tile `[0, slot_len)` contiguously, the per-region inline widen covers the whole buffer exactly as the prior full-buffer widen did.
- **Threshold at 12000** (bracket middle), not at the smallest size: the measured data shows resident LOSES at both 2000 and 8000 even after Lever A — it only wins at 20000. The initial guess of 1024 would have wrongly routed medium (8000) to the resident loser; the measured numbers corrected it to 12000.
- **Oracle env-force** (Lever B side-effect): the size gate would have degenerated `learner_parity_resident_equals_host_tree_on_hip` into a vacuous host-vs-host comparison (its 3000-row corpus is below 12000). Forcing `LGBM_RESIDENT_FORCE=1` for the resident learner keeps the oracle's teeth.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Oracle would have degenerated under the new size gate**
- **Found during:** Task 3 (T2 — adding the size gate)
- **Issue:** `learner_parity_resident_equals_host_tree_on_hip` trains a 3000-row corpus via `RocmBackend::with_resident(true)`; after Lever B's 12000 threshold, `resident_eligible` would return false for it → the "resident" learner silently takes the host path → the oracle compares host-vs-host and passes vacuously, losing its ability to detect a resident-chain divergence.
- **Fix:** The test now sets `LGBM_RESIDENT_FORCE=1` around the resident train (and removes it before the host train) so the resident chain is genuinely exercised past the gate. The override bypasses ONLY the size threshold — every correctness check still runs.
- **Files modified:** `crates/oracle-harness/tests/learner_parity.rs`
- **Verification:** Re-ran 5×; the oracle genuinely exercises the resident path (leaf-11 abs_diff fluctuates around 1e-6 exactly as the pre-existing flaky baseline), confirming the resident chain is driven.
- **Committed in:** `43b256c` (T2 commit)

**2. [Documentation correction] T0 "deterministic failure" reading was actually flaky**
- **Found during:** Task 2 (T1)
- **Issue:** T0 recorded `learner_parity_resident_equals_host_tree_on_hip` as a deterministic baseline failure (3 fails in a row). Re-running post-Lever-A revealed it is FLAKY (pass/fail varies, leaf-11 value varies run-to-run).
- **Fix:** Corrected the T0 note — the root cause is GPU f32-atomic non-determinism in the construct + host build (layers s2b does not touch), with the abs_diff hovering at the 1e-6 threshold. Documented honestly; no code change.
- **Files modified:** `T0-baseline.md`
- **Committed in:** docs (orchestrator commits).

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking) + 1 doc correction.
**Impact on plan:** The Rule 3 fix preserves an oracle's teeth that the size gate would otherwise have neutered — necessary for correctness coverage, no scope creep. Both levers landed as planned.

## Issues Encountered

### Pre-existing failures carried over (NOT introduced by s2b)

1. **`kernel_parity_split_within_tol_on_hip` — FAIL (D-03a, OUT OF SCOPE).** The split-scan f32-vs-f64 accumulation gap (abs_diff ~3.8e-6 / 7.6e-6 > 1e-6 + a knife-edge `default_left` flip). Labeled in-test as the documented `04-ROCM-GAPS.md / D-03a` gap. Present identically at T0 baseline (HEAD `b8a7080`); the s2b change does not touch the split scan.

2. **`learner_parity_resident_equals_host_tree_on_hip` — FLAKY around 1e-6 (pre-existing).** Structural fields (topology/split_feature/decision_type/threshold/counts) are BIT-EXACT every run; the ONLY divergence is a leaf-VALUE on leaf 11 whose abs_diff fluctuates around the test's 1e-6 tolerance (observed 1.6e-6–2.5e-6), passing intermittently. Root cause: GPU f32-ATOMIC build non-determinism in the construct kernel and the host f32-atomic build — BOTH unchanged by s2b. The p90 SUMMARY's "resident==host trees GREEN" does not hold at this exact tolerance on the committed corpus at HEAD. **Pre-existing at HEAD `b8a7080`.** Lever A is bit-identical for a given f32 RAW (proven by the compare_exact_f64_bits oracle), so it neither causes nor cures this; Lever B only routes. The acceptance bar — "does not WORSEN the baseline" — holds: the leaf-11 value distribution is unchanged.

   _Follow-up (out of scope here):_ closing this borderline cell would require a deterministic GPU histogram reduction (e.g. a non-atomic ordered fold) to remove the f32-atomic ordering variance — a separate D-03a-class effort.

## Threat Flags

None — no new network/auth/file/schema surface. Pure GPU-kernel + dispatch-knob change, no new dependencies (T-s2b-SC satisfied).

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/histogram.rs` — FOUND (modified).
- `crates/lgbm-treelearner/src/resident_pool.rs` — FOUND (modified, RESIDENT_MIN_NUM_DATA present).
- `crates/lgbm-treelearner/src/learner.rs` — FOUND (num_data passed).
- `crates/oracle-harness/tests/kernel_parity.rs` — FOUND (modified).
- `crates/oracle-harness/tests/learner_parity.rs` — FOUND (modified).
- Commit `94312e4` (T1) — FOUND in git log.
- Commit `43b256c` (T2) — FOUND in git log.

## Next Phase Readiness

- The resident per-leaf build is now 2 launches and routed only where it wins; the launch-bound small-workload regression from p90 is resolved.
- A clean future target remains: the pre-existing flaky `learner_parity_resident_equals_host_tree_on_hip` (f32-atomic GPU non-determinism at ~1e-6) and the D-03a split gap — both documented, both out of s2b scope.

---
*Phase: 260608-s2b-fuse-gpu-build-kernels-fold-f32-f64-wide*
*Completed: 2026-06-08*
