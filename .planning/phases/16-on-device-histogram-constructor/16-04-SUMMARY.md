---
phase: 16-on-device-histogram-constructor
plan: 04
subsystem: compute
tags: [cubecl, histogram, subtraction-trick, fixhistogram, most-freq-bin, hist-arena, rocm, cpu-anchor, ODL-10]

# Dependency graph
requires:
  - phase: 16-on-device-histogram-constructor
    plan: 02
    provides: "HistArena {parent, smaller, larger} hist_t** rotation contract (D-02)"
  - phase: 16-on-device-histogram-constructor
    plan: 03
    provides: "construct_leaf_hist_on_device BUILD + dequant_leaf_hist (raw u64 -> hist_t)"
  - phase: 16-on-device-histogram-constructor
    plan: 01
    provides: "cpu f64-anchor scaffold (cpu_anchor_columns, assert_close, subtract_ordering guard)"
provides:
  - "fix_histogram_mfb<#[cube]> + fix_histogram_mfb_on — most-freq-bin repair in the hist_t FLOAT domain (leaf_total - Σ ascending), compact DROPPED"
  - "subtract_histogram_on_device — SubtractHistogram via HistArena rotation (larger = parent - smaller in-place), reuses subtract_hist_kernel verbatim, larger.leaf_index>=0 guarded"
  - "construct_histogram_for_leaf + ConstructedLeafHists + LeafConstructGate — the §7.0 build->de-quant->fix->rotate->subtract entry"
  - "cuda_on_device_enabled() — the OFF-by-default LGBM_CUDA_ON_DEVICE production seam gate (lib.rs)"
affects: [17-best-split-finder, 18-data-partition-tree-mutation, on-device-tree-learner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fix consumes the de-quanted hist_t (no 2^30 re-quantize), DROPS compact (§7 is build->fix->subtract; compaction is a CPU-learner artifact, Pitfall 5)"
    - "Split read/write buffers (hist_in / pre-seeded hist_out copy) so non-mfb + mfb==0 cells return byte-identical and the cubecl-cpu MLIR backend has no read-after-write alias"
    - "In-place subtract: output handle = arena.larger_handle() (== old parent slot); rotation reassigns indices only, no bulk copy"
    - "Ordering invariant enforced structurally: parent/smaller are the synced build readback slices, so subtract cannot precede the build sync (8aed100 guard)"
    - "OFF-by-default seam = cuda_on_device_enabled() (call-site gate) + on_device_growth_supported() frozen false; entry additive + unreachable in production"

key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/histogram.rs"
    - "crates/lgbm-compute/src/kernels/subtract.rs"
    - "crates/lgbm-compute/src/lib.rs"
    - "crates/lgbm-compute/tests/rocm_cuda_mirror.rs"

key-decisions:
  - "fix_histogram_mfb uses the single-owner ascending serial fold on BOTH backends (the shipped fix_compact_kernel precedent); the §7.5 plane-reduce/ShuffleReduceSum twin over num_bin_aligned is a parity-neutral perf lever deferred — the merge gate is the cpu f64 anchor (RESEARCH A2 sanctions the single-owner fold fallback)"
  - "The fix kernel — like the shipped fix_compact_kernel — is launched only on the GPU device; the cubecl-cpu MLIR backend rejects the per-feature fold-with-select (failed to run pass). Bit-exact repair is validated on the real ROCm APU pinned to the plain-Rust f64 golden; cpu host tests cover the V5 ladder"
  - "subtract_histogram_on_device reuses subtract_hist_kernel verbatim (no new subtract kernel); the in-place larger=parent-smaller is the established 16-02 round-trip contract wrapped as a guarded production fn"
  - "The end-to-end entry uses mfb==0 fix_feats (fix no-op, exercised but inert) because the 16-03 BUILD accumulates the FULL histogram (it does not omit the most-freq bin); the real omit-and-repair is validated separately in fix_histogram_mfb_repairs_omitted_bin"
  - "cuda_on_device_enabled() is the call-site gate (mirrors on_device_growth_supported's AND-in pattern); construct_histogram_for_leaf does not read env internally — keeps it pure + test-toggle-free, and the rocm end-to-end needs no env mutation"

patterns-established:
  - "build->de-quant->fix->rotate->subtract sequenced behind one entry; both children anchor-pinned, never GPU-vs-GPU (def-f8u-01)"
  - "§7.0 early-return short-circuits before any device build, so the both-children-fail path runs on the cpu client without a GPU"

requirements-completed: [ODL-10]

# Metrics
duration: 15min
completed: 2026-06-30
status: complete
---

# Phase 16 Plan 04: On-Device Subtraction Trick — Fix + Subtract + Entry Summary

**ODL-10 on device: FixHistogram (most-freq-bin omit-and-repair in the hist_t FLOAT domain, `leaf_total − Σ` ascending, compact DROPPED), SubtractHistogram (`larger = parent − smaller` derived in-place via the HistArena `hist_t**` rotation, reusing `subtract_hist_kernel` verbatim, build-synced-before-subtract enforced), and the `ConstructHistogramForLeaf` entry that sequences build→de-quant→fix→rotate→subtract behind the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam — both children anchor-pinned to the cpu f64 fold (25/25 green on the real ROCm APU), `on_device_growth_supported()` frozen false, default path byte-unchanged.**

## Performance

- **Duration:** ~15 min
- **Tasks:** 3
- **Files modified:** 4 (0 created, 4 modified) + 1 deferred-items log

## Accomplishments

- **Task 1 — FixHistogram (`fix_histogram_mfb` + `fix_histogram_mfb_on`):** the §7.5 most-frequent-bin repair over the de-quanted `hist_t` (16-03's `dequant_leaf_hist` output). `do_fix = mfb>0 && mfb<num_bin` (Pitfall 4); repaired cell = the RAW leaf totals `sum_gradient`/`sum_hessian` minus every OTHER bin in ASCENDING order (the load-bearing f64 fold, `i != mfb` via branchless `select`); written to `hist[mfb·2]`/`[mfb·2+1]`. Consumes `hist_t` (NO 2^30 re-quantize) and DROPS the compact offset-shift (Pitfall 5 — `if off > 0` lives only in the legacy `fix_compact_kernel`). V5-checked launcher (num_bin/overflow/slot_off bounds; empty feats → no launch). Split read (`hist_in`) / write (pre-seeded `hist_out` copy) so non-mfb + `mfb==0` cells stay byte-identical.
- **Task 2 — SubtractHistogram (`subtract_histogram_on_device`):** the `hist_t**` rotation wiring — guarded by `larger_leaf_index >= 0` (§7.5), calls `HistArena::rotate()` (larger ← parent slot in-place, smaller ← fresh slot), asserts the no-alias invariant (T-16-04-02), and launches the VERBATIM `subtract_hist_kernel` (`out = parent − smaller`) into the larger (== old parent) slot. No new subtract kernel; no bulk copy. The build-synced-before-subtract ordering (T-16-04-03 / 8aed100) is enforced structurally — `parent`/`smaller` are the synced build readback slices.
- **Task 3 — `construct_histogram_for_leaf` entry + `cuda_on_device_enabled()` seam:** the §7.0 build→de-quant→fix→rotate→subtract sequence returning BOTH children's `hist_in_leaf` (`ConstructedLeafHists`/`LeafConstructGate`). Early-returns `Ok(None)` when BOTH children fail `min_data`/`min_sum_hessian` (builds nothing). Added `cuda_on_device_enabled()` (OFF-by-default `LGBM_CUDA_ON_DEVICE` call-site gate) in lib.rs; `on_device_growth_supported()` stays false (test-asserted).
- **Validation on the real ROCm APU:** `cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror` → **25 passed** (the 4 new GPU cases — fix omit-and-repair + the end-to-end entry producing both children — plus the 21 prior 16-01/16-03 cases, all within the cpu f64-anchor envelope). Default-feature `cargo test -p lgbm-compute` → all green (D-07 byte-unchanged).

## Task Commits

1. **Task 1: FixHistogram most-freq-bin repair (D-01/D-06)** — `96a7456` (feat)
2. **Task 2: SubtractHistogram via HistArena rotation + ordering (D-01/D-02)** — `7deed73` (feat)
3. **Task 3: ConstructHistogramForLeaf entry behind LGBM_CUDA_ON_DEVICE (D-07/D-08)** — `a02e239` (feat)

## Files Created/Modified

- `crates/lgbm-compute/src/kernels/histogram.rs` — added `fix_histogram_mfb` kernel + `fix_histogram_mfb_on` launcher (Task 1); `construct_histogram_for_leaf` entry + `ConstructedLeafHists` + `LeafConstructGate` (Task 3). Purely additive; the shipped kernels (incl. `fix_compact_kernel`) byte-unchanged.
- `crates/lgbm-compute/src/kernels/subtract.rs` — added `subtract_histogram_on_device` (Task 2). `subtract_hist_kernel` reused verbatim (no new subtract kernel).
- `crates/lgbm-compute/src/lib.rs` — added `cuda_on_device_enabled()` seam gate + the `on_device_seam_tests` module; `on_device_growth_supported()` unchanged (still false).
- `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` — host modules `fix_histogram_host` (V5), `subtract_on_device_host` (3 anchor tests), `construct_for_leaf_host` (§7.0 early-return); `mod hip` device tests `fix_histogram_mfb_repairs_omitted_bin` + `construct_histogram_for_leaf_produces_both_children_vs_anchor`.

## Decisions Made

- **Single-owner ascending fold on both backends** for `fix_histogram_mfb` (the shipped `fix_compact_kernel` precedent). The §7.5 per-feature plane-reduce/ShuffleReduceSum twin over `num_bin_aligned` is parity-neutral within ~1e-6 and deferred as a perf lever — the merge gate is the cpu f64 anchor; RESEARCH A2 explicitly sanctions the single-owner fold fallback. Documented in the kernel doc.
- **Fix kernel runs on the GPU device only** (like `fix_compact_kernel`). The cubecl-cpu MLIR backend rejects the per-feature fold-with-select ("failed to run pass" / "operand does not dominate"), so the bit-exact repair is validated on the real ROCm APU pinned to the plain-Rust f64 golden; cpu host tests cover the V5 ladder + empty-no-launch. Same split-half precedent 16-03 set.
- **In-place subtract reuses subtract.rs verbatim**, output = `arena.larger_handle()` (== old parent slot) — the established 16-02 round-trip contract wrapped as a guarded production fn.
- **End-to-end uses `mfb==0` fix_feats** (fix exercised but inert) because the 16-03 BUILD accumulates the FULL histogram (does not omit the most-freq bin); the real omit-and-repair is validated separately. Documented in the test.
- **`cuda_on_device_enabled()` is the call-site gate** (mirrors `on_device_growth_supported`'s AND-in); the entry does not read env internally, keeping it pure and test-toggle-free.

## Deviations from Plan

None requiring approval. Three pragmatic in-scope choices, all consistent with the plan's `<action>` and must-haves:

1. **[Rule 3 — blocking]** The fix kernel could not be launched on cubecl-cpu (MLIR "failed to run pass" on the per-feature fold-with-select — the same reason the shipped `fix_compact_kernel` is rocm-only). Resolved by validating the bit-exact repair on the real ROCm device (pinned to the plain-Rust f64 golden, never GPU-vs-GPU) and keeping the cpu host tests to the V5 ladder — the 16-03 split-half precedent. The plan's `<action>` already sanctions "the single-owner fold per research A2".
2. The hip Fix uses the single-owner ascending fold (NOT the plane-reduce path named in the must-have "truths"). The plan's `<action>` provides this as an explicit fallback; it is parity-neutral within the ~1e-6 gate and the cpu f64 anchor is the hard merge gate. The plane-reduce twin is documented as a deferred perf lever.
3. The end-to-end entry exercises FIX with `mfb==0` (inert) because the 16-03 BUILD does not omit the most-freq bin; the real omit-and-repair has its own dedicated rocm test. No must-have weakened — build→de-quant→fix→subtract is fully sequenced and both children are anchor-matched.

## Known Stubs

None. `fix_histogram_mfb`, `subtract_histogram_on_device`, and `construct_histogram_for_leaf` are fully wired and exercised end-to-end on the real ROCm device against the cpu f64 anchor.

## Deferred Issues

- **DEF-16-OOS-01 (pre-existing, OUT OF SCOPE):** `lgbm-compute --lib --features gpu kernels::autotune::tests::launch_key_display_and_namespace` fails on a `Display` format-string mismatch in `autotune.rs:148` (`"LaunchKey(bucket=10,...)"` vs `"LaunchKey(b10,f50,b256)"`). `autotune.rs` is NOT in this plan's diff (`git diff --name-only` confirms) — pre-existing test/code drift. Logged to `deferred-items.md`; not fixed.

## Threat Surface

All three STRIDE mitigations from the plan's `<threat_model>` are discharged: T-16-04-01 (V5 bounds before `launch_unchecked` in both new launchers — num_bin/overflow/slot_off/length), T-16-04-02 (HistArena no-alias invariant asserted before the in-place subtract), T-16-04-03 (build-synced-before-subtract enforced structurally + the Wave-0 ordering guard). No new network/auth/untrusted-input surface; zero new dependencies (T-16-04-SC).

## Next Phase Readiness

- Phase 17 (best-split finder) consumes `ConstructedLeafHists.{smaller, larger}` `hist_in_leaf`.
- Phase 18 (data partition / tree mutation) consumes the HistArena rotation contract for the whole-tree pool SWAP (`SplitTreeStructureKernel`), still correctly deferred.
- The `cuda_on_device_enabled()` seam is the call site the Phase-18/21 growth driver checks; `on_device_growth_supported()` stays frozen false until then.

## Self-Check: PASSED

- Files: FOUND `histogram.rs` (fix + entry), FOUND `subtract.rs` (subtract wiring), FOUND `lib.rs` (gate + growth), FOUND `rocm_cuda_mirror.rs` (tests)
- Artifacts: FOUND `fix_histogram_mfb_on`, `subtract_histogram_on_device`, `construct_histogram_for_leaf`, `cuda_on_device_enabled`, `on_device_growth_supported`
- Commits: FOUND `96a7456`, `7deed73`, `a02e239`
- `if off > 0` (compact) confirmed only in `fix_compact_kernel`/the fused scan kernel — ABSENT from `fix_histogram_mfb`
- Gates: rocm `--test rocm_cuda_mirror` → 25 passed; default `cargo test -p lgbm-compute` → all green (D-07)

---
*Phase: 16-on-device-histogram-constructor*
*Completed: 2026-06-30*
