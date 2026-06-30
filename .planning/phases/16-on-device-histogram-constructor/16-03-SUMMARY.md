---
phase: 16-on-device-histogram-constructor
plan: 03
subsystem: compute
tags: [cubecl, histogram, build-kernel, u64-fixed-point, partition-geometry, rocm, cpu-anchor, ODL-09]

# Dependency graph
requires:
  - phase: 16-on-device-histogram-constructor
    plan: 01
    provides: "Wave-0 cpu f64-anchor scaffold (cpu_anchor_columns, assert_close, interleave_layout) + sparse/large-bin partition-aware case stubs"
  - phase: 15-on-device-device-dataset-row-subset-gather
    provides: "FeaturePartitionLayout / divide_cuda_feature_groups §13 partition geometry; CudaRowData dense + CSR re-lay"
  - phase: 11-gpu-fixedpoint-int-atomics
    provides: "u64 fixed-point LDS Atomic<u64> accumulation body (S=2^30) + SCALE_F32; fix_compact dequant logic"
provides:
  - "construct_leaf_hist_partition_u64<B: Int> — two-tier §13-geometry build kernel (dense+sparse comptime), gpu-gated"
  - "construct_leaf_hist_partition_global_u64<B: Int> — _GlobalMemory spill twin for large-bin partitions"
  - "dequant_leaf_hist / dequant_leaf_hist_f32 — raw u64 → hist_t at 2^30 (separate de-quant pass)"
  - "spill_cells — checked_mul spill-size guard"
  - "construct_leaf_hist_on_device — V5-checked host launcher deriving §13 geometry, caller-zeroed accumulator"
  - "rocm end-to-end build cases (dense/sparse/shared/forced-global/large-bin-spill/out-zeroed) anchored to the cpu f64 fold"
affects: [16-04-fix-subtract, 17-best-split-finder, on-device-histogram]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-tier §13 build geometry: CUBE_POS_X=partition, UNIT_POS_X=column, UNIT_POS_Y×CUBE_POS_Y=row stripe; LDS Atomic<u64> then cross-block global atomic merge (no grid barrier in cubecl 0.10)"
    - "Dense/sparse as ONE #[cube] generic with a #[comptime] is_sparse branch; shared/global as a sibling #[cube] specialization sharing the body"
    - "De-quant kept a SEPARATE pass (RESEARCH Pattern 3): BUILD stays a clean u64-only accumulator; the cpu f64 anchor split is unentangled"
    - "Anchor-pinned to construct_histograms_f64_on (never GPU-vs-GPU, def-f8u-01); shipped per-feature ROCm kernel left byte-unchanged (480 insertions / 0 deletions)"

key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/histogram.rs"
    - "crates/lgbm-compute/tests/rocm_cuda_mirror.rs"

key-decisions:
  - "Shared-LDS cap stays HIST_LDS_MAX (256 bins); a partition whose span·2 exceeds the cap (or num_large_bin_partition>0, or force_global) routes to the _GlobalMemory sibling — a parity-neutral capacity choice (§17), not the full §13 3072-bin budget"
  - "Kernel kept generic over the bin width <B: Int> only (matching the plan's stated signature); the sparse CSR row_ptr is read at u32 — the {16,32,64} widths are validated through the cpu anchor, not a kernel monomorphization, since the §13 re-lay is fully-dense-as-CSR"
  - "grad/hess read FULL-CORPUS at the gathered row idx (§7.2 cuda_gradients[idx]), matching cpu_anchor_columns exactly"
  - "Marked ODL-09 complete: this plan lands the on-device histogram BUILD (dense+sparse × shared+global, u64 fixed-point, anchor-pinned) — the exact ODL-09 scope; ODL-10 (subtraction trick) remains 16-04"

patterns-established:
  - "Global-bin out layout (partition order + column_hist_offsets) ALIGNS cell-for-cell with cpu_anchor_columns' per-feature concatenation (prefix[col]·2), so the de-quanted raw u64 histogram compares directly to the f64 anchor"
  - "out accumulator uploaded from an explicit zero slice (never client.empty pooled-uninitialized); two back-to-back builds return identical histograms (no stale fold)"

requirements-completed: [ODL-09]

# Metrics
duration: 45min
completed: 2026-07-01
status: complete
---

# Phase 16 Plan 03: On-Device Histogram Constructor — Two-Tier BUILD Kernel Summary

**The hot-path ODL-09 BUILD: a NET-NEW two-tier §13-feature-partition u64 fixed-point histogram kernel (dense + sparse × shared-LDS + `_GlobalMemory` spill) with a de-quant-once pass and a V5-guarded host launcher — anchor-pinned to the cpu f64 fold (18/18 green on the ROCm APU within ~1e-6), with the shipped per-feature ROCm build left byte-unchanged.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-01
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- **Task 1 — two-tier build kernel (`construct_leaf_hist_partition_u64<B: Int>`):** lifted the shipped Phase-11 u64 LDS body onto the §7.1/§13 partition geometry (`CUBE_POS_X`=partition, `UNIT_POS_X`=column, `UNIT_POS_Y×CUBE_POS_Y`=row stripe). LDS block-local `Atomic<u64>` fixed-point (S=2^30) accumulation → cross-block GLOBAL atomic merge (cubecl 0.10 has no grid barrier). Dense-vs-sparse is ONE `#[cube]` generic with a `#[comptime] is_sparse` branch (dense = row-major partition store + `column_hist_offsets`; sparse = per-partition CSR `row_ptr` lookup, partition-local). u64-only hot loop (no f64 scatter, D-08).
- **Task 2 — `_GlobalMemory` spill twin (`construct_leaf_hist_partition_global_u64<B: Int>`):** identical geometry/gather/quantize/merge, but each y-block's partial histogram lives in a pre-allocated global `Array<Atomic<u64>>` at `(CUBE_POS_Y·num_total_bin + phs)·2` (§7.2) instead of LDS — so large-bin partitions build for real. Shared-vs-global is parity-neutral (§17).
- **Task 3 — de-quant pass + V5 launcher + Wave-0 cases:** `dequant_leaf_hist` (raw u64 → hist_t at 2^30, a SEPARATE pass per RESEARCH Pattern 3) + `dequant_leaf_hist_f32` mirror; `spill_cells` `checked_mul` guard (D-09); `construct_leaf_hist_on_device` derives the §13 geometry from `FeaturePartitionLayout`, zeroes the accumulator from an explicit zero slice, runs the V5 bounds ladder (num_total_bin>0, 2·num_total_bin overflow, grad/hess/data/row_ptr length+range) returning typed `ComputeError` BEFORE any `launch_unchecked`, and selects dense/sparse + shared/global. Filled the Wave-0 mirror cases end to end: build → de-quant → `assert_close` vs the cpu f64 anchor.
- **Validation on the real ROCm APU:** `cargo test --features rocm --test rocm_cuda_mirror` → **18 passed**, including the 5 new partition-build cases (dense, sparse, shared, forced-global, large-bin spill) and the out-zeroed regression, all within the ABS 5e-6 / REL 1e-5 envelope of the cpu f64 anchor. The 4 shipped CUDA-mirror tests still pass (no regression).

## Task Commits

1. **Task 1: two-tier §13-geometry u64 build kernel (D-03/D-06/D-08)** — `1f95065` (feat)
2. **Task 2: _GlobalMemory spill build variant (D-04/D-09)** — `1527920` (feat)
3. **Task 3: de-quant-once pass + V5 launcher + Wave-0 mirror cases (D-01/D-05/D-08)** — `24bc317` (feat)

## Files Created/Modified

- `crates/lgbm-compute/src/kernels/histogram.rs` — +480 lines, **0 deletions** (purely additive; the shipped `construct_leaf_hist_resident_lds_kernel_u64` is byte-unchanged, D-03/D-07). Added the two build kernels, the two de-quant fns, `spill_cells`, and `construct_leaf_hist_on_device`.
- `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` — added `mod build_partition_host` (4 gpu-gated host tests: de-quant round-trip, spill overflow guard, V5 rejects, empty-layout no-launch) + the §13 store builders and 5 rocm-gated end-to-end build tests in `mod hip`; updated the two Wave-0 scaffold TODOs to point at the new GPU tests.

## Decisions Made

- **Shared-LDS cap is `HIST_LDS_MAX` (256 bins), not the §13 3072-bin budget.** A partition whose `span·2` exceeds the cap (or `num_large_bin_partition>0`, or `force_global`) routes to the `_GlobalMemory` sibling. This is a parity-neutral capacity choice (§17) that simply routes more partitions to global — it never changes the float result (every rocm equivalence test confirms shared == global == anchor).
- **Kernel generic over `<B: Int>` only** (the plan's stated signature); the sparse CSR `row_ptr` is read at u32. The {16,32,64} CSR widths are validated through the cpu anchor (the §13 re-lay here is fully-dense-as-CSR), not a kernel monomorphization — keeping the build a single generic.
- **De-quant is a SEPARATE pass** (RESEARCH Pattern 3 / Open Q1) so BUILD stays a clean u64-only accumulator; 16-04 Fix then operates on the durable `hist_t`.
- **ODL-09 marked complete** — this plan delivers exactly the ODL-09 scope (on-device build, dense+sparse × shared+global, u64 fixed-point, no f64 hot loop, anchor-pinned). ODL-10 (subtraction trick) remains 16-04.

## Deviations from Plan

None requiring approval. Three pragmatic in-scope choices, all documented above and consistent with the plan's must-haves:
- The shared/global route uses the LDS cap (`HIST_LDS_MAX`) as the spill threshold in addition to `num_large_bin_partition>0` (the plan's stated gate) — necessary because the cubecl LDS size is a comptime `HIST_LDS_MAX`, so any partition exceeding it MUST spill. Parity-neutral (§17).
- The strict TDD RED-first cycle was applied only to the runnable host tests (de-quant, V5, spill); the GPU kernels are not launchable on the cpu CI runtime, so their "test" is compilation under `--features gpu` + the rocm-gated end-to-end (which DID run and pass on the local APU). The cpu f64-anchor merge-gate scaffold (16-01) was already green on first commit, as designed.

## Known Stubs

None. The build kernels, de-quant pass, and launcher are fully wired and exercised end-to-end on the real ROCm device against the cpu f64 anchor. (The launcher dispatches the bin width at `B=u32`; narrower-width `CudaRowData` dispatch is an existing future-wiring detail, not a stub in this plan's scope.)

## Issues Encountered

None blocking. One field-name correction during authoring (`ComputeError::BinIndexOutOfRange{row,bin,num_bin}`), caught and fixed before the Task-3 commit.

## Self-Check: PASSED

- Files: FOUND `crates/lgbm-compute/src/kernels/histogram.rs`, FOUND `crates/lgbm-compute/tests/rocm_cuda_mirror.rs`
- Commits: FOUND `1f95065`, FOUND `1527920`, FOUND `24bc317`
- Gates: `cargo build -p lgbm-compute --features gpu` exit 0; `cargo test -p lgbm-compute --lib` → 62 passed; `cargo test -p lgbm-compute --features gpu --test rocm_cuda_mirror` → 9 passed; `cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror` → **18 passed** (5 new partition-build + 4 shipped mirror + 9 cpu-anchor/host); `cargo test --workspace` (default) → all green (D-07). No `Atomic<i64>` in code (doc-only); no f64 in the build scatter loops; shipped kernel byte-unchanged (480/0 diff).

## Next Phase Readiness

- 16-04 (Fix + Subtract, ODL-10) consumes the raw u64 histogram this build produces: `dequant_leaf_hist` → `hist_t`, then `fix_compact_kernel` (FixHistogram + compact) and the `hist_t**` rotation / `SubtractHistogram` already staged by 16-02. The de-quant scale (2^30) and the `[2b]/[2b+1]` interleave are shared and verified.
- The `construct_leaf_hist_on_device` launcher is the seam Phase 17 (best-split finder) reads from once the per-leaf build is wired into the growth loop.

---
*Phase: 16-on-device-histogram-constructor*
*Completed: 2026-07-01*
