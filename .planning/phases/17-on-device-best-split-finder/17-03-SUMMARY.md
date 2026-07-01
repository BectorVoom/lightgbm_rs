---
phase: 17-on-device-best-split-finder
plan: 03
subsystem: compute
tags: [cubecl, best-split, stage-1, cpu-f64-anchor, hip-f32-mirror, lds-scan, parity, gsd-wave-2]

# Dependency graph
requires:
  - phase: 17-01
    provides: "SplitFindTask + Stage1Scalars + round_ties_even + best_split_parity.rs harness + best_split.txt scaffold (the stub this plan fills)"
  - phase: 17-02
    provides: "get_leaf_gain_smoothed(+_f32) / calculate_splitted_leaf_output_smoothed(+_f32) — the USE_SMOOTHING gain path"
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: "split_info::SplitScalars record + random::draw_rand_int32_on (CUDARandom LCG) + primitives plane/SharedMemory idiom"
provides:
  - "split_eval_body #[cube] stage-1 numerical core (fwd+rev, f64) — the cpu single-owner fold anchor"
  - "find_best_splits_stage1_on cpu f64 launcher (fills the 17-01 stub; best_split.txt goldens GREEN)"
  - "round_ties_even_cube / round_ties_even_f32_cube — branch-free even-round via floor (no round-half-up), #[cube]-lowerable"
  - "validate_stage1_inputs — V5 launch-boundary check (num_bin/hist-length)"
  - "split_eval_body_f32 / split_eval_kernel_f32 / find_best_splits_stage1_f32_on — cpu-testable f32 mirror (structure-exact, ~1e-5)"
  - "stage1_block_scan (net-new two-level LDS scan, D-03) + reduce_best_gain (strict-> argmax) + split_eval_block_kernel_f32 — the gpu-gated hip block-parallel path"
  - "best_split.txt goldens FINALIZED by independent §8.1 hand-transcription (11 records)"
affects: [17-04, 17-05, on-device-best-split-finder]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Independent-calculator provenance: a throwaway §8.1 reference calculator (transcribing cuda_best_split_finder.cu, NOT the cpu-fold kernel) finalizes the goldens, so the anchor cannot self-certify"
    - "Body + f32-mirror duplication (the shipped split.rs precedent), NOT one generic body: cubecl gain fns are type-specific (f64 vs f32), so split_eval_body (f64) + split_eval_body_f32 (f32) mirror each other rather than a single Numeric generic"
    - "cpu = single-owner serial fold (runs on cubecl-cpu); gpu = block-parallel plane/LDS kernel #[cfg(feature=gpu)] — the Phase 14-16 convention (cubecl-cpu has NO plane support, primitives.rs:1182)"
    - "Two-phase kEpsilon + round-ties-even count recovery reproduced bit-exact from §8.1"

key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/best_split.rs — the stage-1 numerical core (f64 fold + f32 mirror + gpu block path + validation)"
    - "crates/oracle-harness/tests/best_split_parity.rs — un-ignored stage-1 test + added f32-within-tol test"
    - "crates/oracle-harness/tests/fixtures/kernels/best_split.txt — FINALIZED 11 SWIN goldens (+ count_ties_even case; fixed empty-case num_bin/hist length)"

key-decisions:
  - "Followed the shipped split.rs body+mirror precedent instead of a single `split_eval_body::<N: Numeric>` generic: the gain fns (get_split_gains vs get_split_gains_f32) are type-specific, so a true generic would need a trait; split_eval_body (f64) + split_eval_body_f32 (f32) is the faithful, lower-risk mirror (Rule 3)"
  - "The block-parallel two-level LDS scan + ReduceBestGain (D-03) are gpu-gated (#[cfg(feature=gpu)]) exactly like every rocm kernel in histogram.rs/primitives.rs — cubecl-cpu has NO plane support, so the cpu-testable f32 mirror is the single-owner fold and the block path's on-device rocm assertion is 17-05 (as Task 2 itself states)"
  - "Goldens finalized by an INDEPENDENT Python §8.1 calculator (hand-transcription of cu:146-320 + cuda_leaf_splits.hpp), NOT copied from the cpu-fold output (anti-circularity provenance guard honored)"
  - "count recovery uses round-ties-even via a floor-based branch-free identity inside #[cube] (Assumption A1 resolved conservatively: f64::round_ties_even is NOT relied on inside #[cube])"

patterns-established:
  - "split_eval_body/_f32 dispatch use_smoothing at runtime via select(sm, gain_sm, gain_ns) — both computed, one selected (no comptime fan-out)"
  - "reduce_best_gain: SharedMemory stage + unit-0 serial strict-> argmax (block_size sentinel = no winner) — deterministic lowest-index tie-break matching the cpu fold"

requirements-completed: [ODL-11]

# Metrics
duration: 95min
completed: 2026-07-01
status: complete
---

# Phase 17 Plan 03: On-Device Best-Split Finder — Stage-1 Numerical Core Summary

**The Phase-17 §8.1 numerical heart: `split_eval_body` — the cpu f64 fold that reproduces `cuda_best_split_finder.cu`'s per-`(leaf,feature)` split evaluation (prefix-sum → complement-from-parent → two-phase kEpsilon count recovery → guards → gain → strict-`>` argmax → `CUDASplitInfo`) bit-exact to the C++-transcribed goldens, plus the f32 hip mirror (cpu-testable single-owner fold + gpu-gated two-level LDS block scan + `ReduceBestGain`) — all five parity landmines reproduced.**

## Performance
- **Duration:** ~95 min
- **Completed:** 2026-07-01
- **Tasks:** 2/2

## What Was Built

### Task 1 — `split_eval_body` cpu f64 fold + finalized goldens (commit 42627e1)
- `split_eval_body` (`#[cube]`, f64) — a verbatim transcription of `FindBestSplitsForLeafKernelInner` (`cu:146-320`) driven single-owner (`CubeDim(1)`) as the deterministic anchor. Serial inclusive prefix-sum → cumulative scanned side → complement-from-parent → **two-phase kEpsilon** (thread-0 adds `kEpsilon` once; guard recovers count from the kEpsilon-included hessian, the record subtracts it back and re-recovers) → guards (fixed order) → gain (runtime `use_smoothing` dispatch: `get_split_gains` / the 17-02 `get_leaf_gain_smoothed`) → **strict-`>` argmax** (lowest bin wins ties) → the winning `CUDASplitInfo` record with `default_left = assume_out_default_left`.
- `round_ties_even_cube` — branch-free round-ties-**even** using only `f64::floor` (a cubecl `Float` intrinsic), so it lowers on cubecl-cpu AND hip without relying on `f64::round_ties_even` (Assumption A1 resolved). Diverges from `split.rs::round_int`'s round-half-up (the D-01 landmine).
- `find_best_splits_stage1_on` — the cpu f64 launcher: `validate_stage1_inputs` (V5), the categorical Phase-22 dispatch seam (returns `is_valid=false` sentinel, no eval), the USE_RAND `NextInt(0,num_bin-2)` draw via `random.rs` `draw_rand_int32_on` (RandInt32, seeded `extra_seed+task_index`), launch, decode `SplitScalars`.
- **Goldens finalized:** the 11 `best_split.txt` SWIN records were re-derived by an **independent Python §8.1 calculator** (hand-transcription of `cu:146-320` + `cuda_leaf_splits.hpp`, NOT the cpu-fold output). Added a `count_ties_even` case (`left_count=2` where round-half-up would give 3) and fixed the Wave-0 empty-case `num_bin`/histogram-length mismatch.
- Un-ignored `best_split_parity_stage1_bit_exact_on_cpu` — now **1 passed** (was `#[ignore]`d), every field `compare_exact_f64_bits`.
- Unit test `validate_stage1_inputs_rejects_bad_shapes`.

### Task 2 — hip f32 mirror (commit 91b3544)
- `split_eval_body_f32` + `split_eval_kernel_f32` + `find_best_splits_stage1_f32_on` — the **cpu-testable** f32 single-owner mirror (all literals f32-pinned, the `*_f32` gain fns, `round_ties_even_f32_cube`). Drives through cubecl-cpu, anchored to the Task-1 f64 fold: structure bit-exact, per-side sums/value/gain within ~1e-5 (def-f8u-01, never GPU-vs-GPU).
- `stage1_block_scan` — the **net-new two-level within-block inclusive scan** (D-03): `plane_inclusive_sum` intra-plane + a `SharedMemory` cross-plane carry under `sync_cube()` (the idiom borrowed from `primitives.rs`, NOT the generic `block_scan`). `#[cfg(feature="gpu")]`.
- `reduce_best_gain` — the `ReduceBestGain` block argmax over `(gain, found, thread_index)` with strict `>` (lowest index wins ties); a `block_size` sentinel signals no winner. `#[cfg(feature="gpu")]`.
- `split_eval_block_kernel_f32` — the block-parallel hip stage-1 kernel wiring the scan + argmax, byte-for-byte the §8.1 math of `split_eval_body_f32` parallelized. `#[cfg(feature="gpu")]`, compile-verified with `--features gpu`; the on-device rocm parity assertion lands in **17-05** (as the plan states).
- Test `best_split_parity_stage1_f32_within_tol_on_cpu` (default-template + USE_L1).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Body + f32-mirror instead of one `split_eval_body::<N: Numeric>` generic**
- **Found during:** Task 1/2
- **Issue:** The plan's artifact names `split_eval_body::<N: Numeric>` as one shared generic calling the gain fns. But `crate::gain`'s fns are type-specific (`get_split_gains` f64 vs `get_split_gains_f32` f32) — a true generic body would need a trait abstraction over them.
- **Fix:** Followed the SHIPPED `split.rs` precedent (which the plan's `read_first` cites as "the SHAPE to follow"): `split_eval_body` (f64, the anchor) + `split_eval_body_f32` (f32 mirror), each calling its typed gain fns. Same math, faithful, lower-risk. The `split_eval_body` symbol + `contains: split_eval_body` + the `get_(split_gains|leaf_gain)` key-links are all satisfied.
- **Files:** `best_split.rs`
- **Commits:** 42627e1, 91b3544

**2. [Rule 3 - Blocking] Block-parallel two-level LDS scan is gpu-gated; cpu f32 verify uses the single-owner mirror**
- **Found during:** Task 2
- **Issue:** Task 2 asks the f32 launcher (using `stage1_block_scan`) to "drive through cubecl-cpu". But cubecl-cpu has NO plane support (`primitives.rs:1182`), and EVERY multi-unit LDS/plane kernel in this codebase (`histogram.rs`, `primitives.rs`) is `#[cfg(feature="gpu")]`. A block-parallel plane scan cannot run on cubecl-cpu.
- **Fix:** Split the f32 path per the established Phase-14-16 convention (CLAUDE.md: codebase conventions govern): the cpu-testable f32 anchor is the SINGLE-OWNER `split_eval_kernel_f32` fold (runs on cubecl-cpu, structure-exact + ~1e-5); the `stage1_block_scan` (D-03 two-level LDS) + `reduce_best_gain` + `split_eval_block_kernel_f32` are the gpu-gated hip path, compile-verified with `--features gpu`, whose on-device rocm assertion the plan itself defers to 17-05 ("the actual on-device rocm assertion is added in 17-05"). Both `split_eval_kernel_f32` and `stage1_block_scan` symbols exist; the scan does not call the generic `block_scan`; the argmax uses strict `>`.
- **Files:** `best_split.rs`, `best_split_parity.rs`
- **Commit:** 91b3544

**3. [Rule 1 - Bug] Wave-0 empty-case histogram length mismatch**
- **Found during:** Task 1 (first parity run)
- **Issue:** The 17-01 `empty_no_valid_split` golden declared `num_bin=5` but its SHIST had only 8 cells (4 bins). `validate_stage1_inputs` (correctly) rejected `hist.len() != 2*num_bin`.
- **Fix:** Extended the empty-case SHIST to 10 cells (5 bins). Still `is_valid=0`.
- **Files:** `best_split.txt`
- **Commit:** 42627e1

## Acceptance-Criteria Note (grep clarification)
The Task-1 acceptance `grep -nE 'round_int|x + 0\.5' best_split.rs returns nothing` still matches lines — but all matches are (a) rustdoc intra-links naming the divergent host fn `super::split::round_int` for explanation and (b) the pre-existing `count_recovery_ties_even` unit test's round-half-up **contrast** closure that PROVES the divergence. Neither is the count-recovery code path — the numerical core uses only `round_ties_even_cube` / `round_ties_even_f32_cube`. Intent (round-half-up not used in the core) is met.

## Verification
- `cargo test -p oracle-harness --test best_split_parity` — **2 passed** (`stage1_bit_exact_on_cpu` all-fields bit-exact on 11 goldens incl. count_ties_even; `stage1_f32_within_tol_on_cpu` structure-exact + ~1e-5).
- `cargo test -p lgbm-compute --lib best_split` — 3 passed (count_recovery_ties_even, assume_out_default_left_table, validate_stage1_inputs_rejects_bad_shapes).
- `cargo test --workspace` — GREEN (0 failures; kernel_parity 10/10, learner_parity, raw_bin_train unregressed).
- `cargo build -p lgbm-compute --features gpu` — the gpu-gated block path (`stage1_block_scan` + `reduce_best_gain` + `split_eval_block_kernel_f32`) compiles.
- clippy clean on `best_split.rs` (default + gpu).
- Landmine greps: no generic `block_scan(` call; `reduce_best_gain` uses strict `>`; no un-pinned f64 literal in the f32 mirror body.

## Known Stubs
Stages 2 (cross-feature reduce) and 3 (cross-leaf argmax + 8-int export) remain 17-01 stubs — filled by 17-04/17-05 (not this plan's scope). The gpu-gated `split_eval_block_kernel_f32` has no on-device rocm parity assertion yet — that is 17-05's scope (the plan's own deferral, not an accidental stub).

## Self-Check: PASSED
- Commits 42627e1, 91b3544 present in git log.
- `best_split.rs` exists (1480 lines) with `split_eval_body` + `stage1_block_scan` symbols.
- Both parity tests + workspace merge gate green.
