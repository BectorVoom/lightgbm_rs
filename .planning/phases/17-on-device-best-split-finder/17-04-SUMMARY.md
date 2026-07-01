---
phase: 17-on-device-best-split-finder
plan: 04
subsystem: compute
tags: [cubecl, best-split, stage-1, globalmem-spill, strided-scan, alloc-once, cpu-f64-anchor, hip-f32-mirror, parity, gsd-wave-3]

# Dependency graph
requires:
  - phase: 17-03
    provides: "split_eval_body cpu f64 fold + split_eval_body_f32 / split_eval_block_kernel_f32 hip mirror + stage1_block_scan + reduce_best_gain + round_ties_even_f32_cube + validate_stage1_inputs + the D-07 fixture matrix (incl. globalmem_spill num_bin=300 golden)"
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: "DeviceSplitInfo::new alloc-once idiom (counted alloc closure + device_allocations counter) + SplitScalars + random.rs draw_rand_int32_on"
provides:
  - "split_eval_globalmem_kernel_f32 (#[cube(launch_unchecked)], gpu-gated) — the hip f32 strided >256-bin _GlobalMemory variant (D-05)"
  - "global_memory_prefix_sum (#[cube], gpu-gated) — the chunked two-level in-place GlobalMemoryPrefixSum scan over global scratch"
  - "find_best_splits_stage1_globalmem_f32_on — the gpu-gated spill launcher consuming the pre-allocated scratch"
  - "Stage1GlobalMemScratch::new — the alloc-once (D-11) feature_hist_{grad,hess,stat,index} scratch + device_allocations counter"
  - "validate_globalmem_scratch (V5 checked_mul overflow guard) + stage1_needs_globalmem (dispatch boundary) + STAGE1_BLOCK_THREADS / NUM_STAGE1_SCRATCH_BUFFERS consts"
affects: [17-05, on-device-best-split-finder]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "cpu f64 fold handles >256 bins by a larger serial loop bound (A4) — no separate cpu spill implementation; the net-new part is the hip strided kernel only"
    - "Body+mirror precedent extended to the spill path: the strided variant is f32 (gpu-gated), anchored to the 17-03 cpu f64 fold (never GPU-vs-GPU, def-f8u-01), NOT a Numeric generic — same rationale as 17-03 (type-specific gain fns)"
    - "GlobalMemoryPrefixSum = chunked two-level in-place scan: per-thread contiguous chunk sum → exclusive block-scan of chunk sums (stage1_block_scan(sum)-sum) → add base to chunk head → serial within-chunk propagate"
    - "alloc-once scratch via the DeviceSplitInfo counted-closure idiom; device_allocations proves 'allocated exactly once' (D-11)"

key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/best_split.rs — the _GlobalMemory strided f32 kernel + GlobalMemoryPrefixSum + alloc-once scratch + V5 validation + dispatch helper + 3 unit tests"

key-decisions:
  - "The >256-bin cpu parity was ALREADY satisfied by the 17-03 serial fold (split_eval_body loops scan_count=num_bin with no register/LDS cap, A4) — the globalmem_spill num_bin=300 golden was already bit-exact and saw_globalmem already asserted. This plan adds the NET-NEW hip strided f32 kernel (compile-verified; on-device rocm assertion deferred to 17-05, the same deferral 17-03 used for the block kernel)."
  - "Named the scratch overflow guard validate_globalmem_scratch (a companion fn called by Stage1GlobalMemScratch::new) rather than mutating validate_stage1_inputs's signature — extending the existing fn would break its 3 launcher call-sites + unit test; the DeviceSplitInfo::new precedent (checked_mul inside new) is the codebase convention the plan itself cites (Rule 3, mirrors 17-03's artifact-naming deviation)."
  - "Faithfully reproduced the C++ _GlobalMemory within-thread last-beating overwrite (cu:1199-1204) then cross-thread ReduceBestGain — the reference behavior, not the serial fold's lowest-index scan (parity-relevant only on-device where 17-05 asserts it)."
  - "Reserved all 4 scratch buffers (grad/hess used by the continuous kernel; stat/index reserved for the discretized/categorical _GlobalMemory variants that are C++ TODO / Phase-22 / v2) — mirrors how DeviceSplitInfo reserves its categorical slabs."

patterns-established:
  - "stage1_needs_globalmem(num_bin, block_threads) = num_bin > block_threads: the ungated, cpu-testable spill dispatch boundary (the gpu block/globalmem on-device routing is wired in 17-05)"
  - "GlobalMemoryPrefixSum exclusive base = stage1_block_scan(thread_sum) - thread_sum (reuse the inclusive two-level LDS scan for the per-chunk-sum exclusive prefix)"

requirements-completed: [ODL-11]

# Metrics
duration: 45min
completed: 2026-07-01
status: complete
---

# Phase 17 Plan 04: On-Device Best-Split Finder — `_GlobalMemory` >256-bin Spill Variant Summary

**The `_GlobalMemory` stage-1 spill path (D-05): a faithful strided hip f32 port of `FindBestSplitsForLeafKernelInner_GlobalMemory` + `GlobalMemoryPrefixSum` over pre-allocated global scratch (allocated ONCE, D-11), running the SAME gain/count/guard/argmax math as 17-03 — with the cpu f64 fold already covering >256 bins by a larger serial loop bound (A4), so the `num_bin=300` global-memory golden is bit-exact and all 6 D-07 categories are green.**

## Performance
- **Duration:** ~45 min
- **Completed:** 2026-07-01
- **Tasks:** 1/1

## What Was Built

### Task 1 — `_GlobalMemory` strided variant + alloc-once scratch (commit 8faec40)

**The net-new hip strided kernel (gpu-gated, compile-verified):**
- `split_eval_globalmem_kernel_f32` — a verbatim strided port of `FindBestSplitsForLeafKernelInner_GlobalMemory` (`cuda_best_split_finder.cu:1051-1273`, the continuous forward/reverse branches). Three phases: (A) each unit STRIDES over bins `t, t+blockDim, …` linearising the skip/reverse-adjusted per-bin sums into the pre-allocated `grad_buf`/`hess_buf` scratch, thread-0 seeds `kEpsilon` at the scan origin; (B) `global_memory_prefix_sum` scans both in place; (C) a second strided pass evaluates guards + gain (runtime smoothing dispatch) exactly as `split_eval_body_f32`, `reduce_best_gain` picks the winner (strict `>`, lowest index), and the winning unit writes the kEpsilon-subtracted `CUDASplitInfo` record. Same gain/count/guard/argmax math as 17-03 — only the scan carrier (global scratch vs LDS) and the strided iteration differ (A4). `#[cfg(feature="gpu")]`, `--features gpu` compile-verified; the on-device rocm parity assertion lands in 17-05 (the same deferral 17-03 used for the block kernel).
- `global_memory_prefix_sum` — the `GlobalMemoryPrefixSum` (`cuda_algorithms.hpp:169-185`) chunked two-level in-place inclusive scan: per-thread contiguous chunk sum → exclusive block-scan of the per-chunk sums (`stage1_block_scan(sum) - sum`) → add base to the chunk head → serial within-chunk propagate. NO f64 (D-10, WR-05).
- `find_best_splits_stage1_globalmem_f32_on` — the gpu-gated spill launcher: `validate_stage1_inputs` (V5), scratch-slab fit check, USE_RAND draw, `launch_unchecked` → single `read_one_unchecked` (SC#2), `SplitScalars` decode. Consumes the PRE-ALLOCATED `Stage1GlobalMemScratch` handles (never allocates the scan scratch).

**The alloc-once scratch + dispatch + validation (ungated, cpu-testable):**
- `Stage1GlobalMemScratch::new` — pre-allocates the 4 `feature_hist_{grad,hess,stat,index}_buffer` handles ONCE via the counted `alloc` closure (the `DeviceSplitInfo::new` idiom, `split_info.rs:289-293`), sized `largest_feature_bin_count × num_concurrent_blocks`; `device_allocations()` returns `NUM_STAGE1_SCRATCH_BUFFERS = 4`, proving the D-11 alloc-once invariant. `grad`/`hess` carry the strided scan; `stat`/`index` are reserved for the discretized/categorical `_GlobalMemory` variants (C++ TODO / Phase-22 / v2), mirroring how `DeviceSplitInfo` reserves its categorical slabs.
- `validate_globalmem_scratch` — V5 launch-boundary guard: rejects zero/overflowing `largest_feature_bin_count × num_concurrent_blocks` with a typed `ComputeError::Runtime` via `checked_mul` (T-17-01/T-17-02), called by `new`.
- `stage1_needs_globalmem(num_bin, block_threads)` — the `num_bin > block_threads` spill dispatch boundary + `STAGE1_BLOCK_THREADS = 256`.
- 3 unit tests: `stage1_dispatch_globalmem_boundary`, `validate_globalmem_scratch_rejects_overflow`, `globalmem_scratch_allocated_exactly_once` (device_allocations == 4 on the cubecl-cpu client).

**The cpu f64 fold needed no change (A4):** `split_eval_body` is serial (`for t in 0..scan_count` with `scan_count = num_bin`) and has no register/LDS cap, so the `globalmem_spill` num_bin=300 golden was already bit-exact and `saw_globalmem` already asserted in 17-03. This plan is the hip strided net-new part + the D-11 scratch + V5 guard.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `validate_globalmem_scratch` companion fn instead of extending `validate_stage1_inputs`**
- **Found during:** Task 1
- **Issue:** The acceptance names `validate_stage1_inputs` as the scratch-slab guard, but that fn's signature (`num_bin, hist_len`) is consumed by 3 launcher call-sites + a unit test; adding scratch params would break them.
- **Fix:** Added a companion `validate_globalmem_scratch(largest_feature_bin_count, num_concurrent_blocks)` (the `checked_mul` overflow guard, T-17-01) called by `Stage1GlobalMemScratch::new` — the `DeviceSplitInfo::new` precedent the plan itself cites (checked_mul inside `new`). Same intent, lower risk. Directly unit-tested (`validate_globalmem_scratch_rejects_overflow`). Mirrors 17-03's artifact-naming deviation (codebase convention governs, CLAUDE.md).
- **Files:** `best_split.rs`
- **Commit:** 8faec40

**2. [Rule 3 - Blocking] f32 strided variant (gpu-gated) instead of a `split_eval_body_globalmem::<N: Numeric>` generic**
- **Found during:** Task 1
- **Issue:** The plan's artifact names `split_eval_body_globalmem::<N: Numeric>`, but (a) the gain fns are type-specific (f64 vs f32, no Numeric generic — the same wall 17-03 hit) and (b) the strided scan uses plane intrinsics with NO cubecl-cpu support (`primitives.rs:1182`), so it must be `#[cfg(feature="gpu")]` like every rocm kernel.
- **Fix:** Followed the 17-03 precedent exactly — the strided variant is the f32 `split_eval_globalmem_kernel_f32` (gpu-gated, compile-verified), anchored to the cpu f64 fold (which handles >256 by A4, never GPU-vs-GPU). The `globalmem` `contains:` substring is satisfied (`split_eval_globalmem_kernel_f32`, `find_best_splits_stage1_globalmem_f32_on`, `global_memory_prefix_sum`). On-device rocm assertion is 17-05's scope (the plan's own deferral for the hip path).
- **Files:** `best_split.rs`
- **Commit:** 8faec40

## Acceptance-Criteria Note (client.empty grep clarification)
The acceptance says "`client.empty` appears only in the constructor." `grep client.empty best_split.rs` shows 4 code occurrences: **line 1173 is the scratch constructor** (`Stage1GlobalMemScratch::new`, where ALL 4 `feature_hist_*` scratch buffers are allocated once, D-11). The other 3 (lines 662, 971, 1774) are the tiny **14-cell `h_out` output packet** in the three per-launch stage-1 launchers — the established SC#2 single-readback pattern (`random.rs:177`), NOT scan scratch. Two of those pre-date this plan (the f64 + f32 single-owner launchers, 17-03); the third is the new globalmem launcher's own output packet. The D-11 intent (the resident scan scratch is pre-allocated once, never per-split / in-kernel) is met: no `client.empty` inside any `#[cube]` kernel body, and the `feature_hist_*` scratch is constructor-only.

## Threat Flags
None. New surface (host `largest_feature_bin_count` → scratch sizing + strided launch) is exactly the plan's `<threat_model>` T-17-01/T-17-02, mitigated by `validate_globalmem_scratch` (`checked_mul`) + the scratch-fit check before `launch_unchecked`.

## Known Stubs / Limitations
- The `na_as_missing && mfb_offset == 1` special reduction subcase of `_GlobalMemory` (`cu:1095-1114`, a `ShuffleReduceSum` of the non-default bins) is NOT ported — it is not exercised by the D-07 fixture (the `globalmem_spill` golden is forward, `mfb=0`, `na=0`). Documented in the kernel doc as a `_GlobalMemory` sub-branch to complete when a golden needs it, not a silent stub (the continuous forward/reverse branches the fixture drives ARE ported).
- The gpu-gated `split_eval_globalmem_kernel_f32` has no on-device rocm parity assertion yet — that is 17-05's scope (the plan's own deferral, matching the 17-03 block kernel).
- Stages 2 (cross-feature reduce) and 3 (cross-leaf argmax + 8-int export) remain 17-01 stubs — 17-05's scope.

## Verification
- `cargo test -p oracle-harness --test best_split_parity` — **2 passed** (`stage1_bit_exact_on_cpu`: globalmem_spill num_bin=300 bit-exact on the cpu f64 fold, `saw_globalmem` asserted, all 6 D-07 categories green; `stage1_f32_within_tol_on_cpu` unregressed).
- `cargo test -p lgbm-compute --lib best_split` — **10 passed** (incl. the 3 new: dispatch boundary, validate-overflow, alloc-once counter `device_allocations == 4`).
- `cargo build -p lgbm-compute --features gpu` — the gpu-gated strided kernel + `global_memory_prefix_sum` + launcher compile.
- `cargo test --workspace` — **GREEN** (73 result blocks, 0 failures; merge gate intact, default path byte-unchanged, D-09).
- clippy clean on `best_split.rs` (default + `--features gpu`).
- D-10 audit: no un-pinned f64 in the f32 spill kernels (only doc-comment mentions).

## Self-Check: PASSED
- Commit 8faec40 present in git log.
- `crates/lgbm-compute/src/kernels/best_split.rs` exists with `split_eval_globalmem_kernel_f32`, `global_memory_prefix_sum`, `Stage1GlobalMemScratch`, `validate_globalmem_scratch`, `stage1_needs_globalmem` symbols.
- best_split parity + lib tests + workspace merge gate all green; gpu build compiles.
