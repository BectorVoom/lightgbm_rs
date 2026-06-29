---
phase: 14-foundation-shared-device-primitives-device-structs-rng
reviewed: 2026-06-29T12:03:14Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/primitives.rs
  - crates/lgbm-compute/src/kernels/random.rs
  - crates/lgbm-compute/src/kernels/split_info.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/tests/cuda_random_parity.rs
  - crates/lgbm-compute/tests/plane_intrinsic_smoke.rs
  - crates/lgbm-compute/tests/primitives_self.rs
  - crates/lgbm-compute/tests/split_info.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md
  - crates/oracle-harness/fixtures/primitives/argsort.txt
  - crates/oracle-harness/fixtures/primitives/percentile.txt
  - crates/oracle-harness/fixtures/primitives/prefix_sum.txt
  - crates/oracle-harness/fixtures/primitives/reduce.txt
  - crates/oracle-harness/tests/learner_parity.rs
  - crates/oracle-harness/tests/primitive_parity.rs
  - xtask/cpp/CMakeLists.txt
  - xtask/cpp/primitive_capture.cu
  - xtask/src/main.rs
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-06-29T12:03:14Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Phase 14 establishes the shared device primitives (prefix-sum, shuffle reductions,
bitonic argsort, percentile skeletons), the `CUDARandom` device LCG, and the SoA
pre-allocated `DeviceSplitInfo` split-record, plus the C++/HIP golden capture
harness and the `primitive_parity` cross-check.

The **implementation logic is sound and well-anchored**. I traced the prefix-sum
3-launch global structure, the reduction folds, the bitonic argsort comparator,
the RNG recurrence, and the SoA copy/alloc paths against their serial references
and the C++ transcription. Concrete correctness checks that hold:

- The device `CUDARandom` stream is bit-identical to the host `Random` (the
  internal-advance-then-loop-advance pattern in `random.rs` keeps state in sync;
  proven exact by `cuda_random_parity.rs`).
- The prefix-sum cross-block add-back math is correct for inclusive and exclusive,
  and the cpu-anchor single-owner folds are bit-exact vs serial references.
- The bitonic argsort `od`/`u32` loop has no underflow; the even-half-segment guard
  keeps `tid + half < aligned`, so `launch_unchecked` is in-bounds.
- `DeviceSplitInfo` allocates exactly `NUM_FIELD_BUFFERS` once and `copy_slot`
  performs zero allocation (`copy_within` for the slabs), matching D-08.
- The `cuda_on_device_env` / `on_device_eligible` seam (learner.rs) is genuinely
  OFF-by-default and the decide-once fork is dead with the env unset, so the host
  merge gate is byte-unchanged. No injection surface in any env/path read.

**No BLOCKER-class defect was found** — there is no demonstrably-wrong result on
the f64 cpu merge gate. The findings below are real weaknesses in the **C++
cross-validation coverage** of three primitives. Because "numerical fidelity vs the
C++ reference is the non-negotiable contract," gaps where a primitive's reference
parity is asserted only weakly (or not at all) are material and are filed as
WARNINGs.

## Warnings

### WR-01: max/min reduction goldens encode a sub-1024-thread 0-identity artifact; parity is masked, not validated

**File:** `crates/oracle-harness/fixtures/primitives/reduce.txt:9,13,17,21,25` (all `op=max` f64 → `out=0`); `crates/oracle-harness/tests/primitive_parity.rs:288-307`; `crates/lgbm-compute/src/kernels/primitives.rs:427-460`

**Issue:** Every committed `op=max` golden is exactly `0.0` (confirmed: all f64
max goldens = `0`, all min goldens nonzero). This is because the harness launches
`ShuffleReduceMax` as `<<<1, n>>>` with `n ∈ {32,96,256}` (`primitive_capture.cu:1172-1174`),
where `num_warp < warpSize`, so the verbatim reference folds a `0` for the
out-of-range warp lanes (`shared_mem_buffer[warpLane] : 0`, `primitive_capture.cu:246`)
and returns `max(true_max, 0)` = `0` for the all-negative `SpreadF64` inputs. The
Rust `reduce_max_f64_on` returns the **true max** (a negative). The cross-check
hides the divergence by applying `got.max(0.0)` before asserting
(`primitive_parity.rs:290-298`). Real LightGBM launches these reductions with full
1024-thread blocks (`num_warp == 32`), where the 0-floor does NOT occur — so the
committed goldens validate a non-representative artifact rather than production
reduction semantics, and the `op=max` reference parity is effectively untested. The
symmetric `.min(0.0)` bridge (`primitive_parity.rs:299-307`) is a no-op on this
all-negative data and conflates two different reference behaviors (max uses a `0`
identity, min uses `shared_mem_buffer[0]`), making the test's "0-identity bridge"
comment misleading.

**Fix:** Recapture the reduction goldens at the production block size
(`<<<1, 1024>>>` padding the input, or a representative mix that includes a
1024-thread case) so `num_warp == warpSize` and the golden carries the genuine
reference value; then assert `reduce_max_f64_on` bit-exact WITHOUT the `.max(0.0)`
bridge. Alternatively, add at least one max/min case with a positive-containing
input at full block width so the cross-check exercises a non-`0` max golden. At
minimum, document in the test that the committed max goldens are a sub-1024-thread
artifact and the `.max(0.0)` bridge is not a reference semantic.

### WR-02: weighted percentile has no C++ oracle — validated only against a paraphrase of itself

**File:** `crates/oracle-harness/fixtures/primitives/percentile.txt` (all 9 `weighted=1` records are `status=deferred_kaggle_nvcc` markers, none carry an `out=`); `crates/oracle-harness/tests/primitive_parity.rs:399-404` (skips weighted); `crates/lgbm-compute/tests/primitives_self.rs:334-361` (self-reference); `crates/lgbm-compute/src/kernels/primitives.rs:1059-1073`

**Issue:** The weighted `PercentileDevice` branch is non-idempotent on the spoofed
APU and was deferred to a Kaggle/nvcc capture that has not happened, so `percentile.txt`
contains zero real weighted outputs and `primitive_parity` skips every weighted
case. The only validation of `percentile_weighted_f32_on` is `primitives_self.rs`'s
`serial_percentile_weighted`, which re-implements the identical threshold-scan and
interpolation logic — it tests the code against a paraphrase of itself, not against
the C++ reference. The edge branch returns the raw, un-permuted `values[pos]`
(`primitives.rs:1066-1068`), which the author's own comment flags as "a reference
quirk the Phase-19 consumer revisits" — i.e. its correctness vs C++ is unconfirmed.
A transcription error shared between the primitive and its serial mirror would pass
green.

**Fix:** Capture the real weighted-percentile goldens on a CUDA box
(`LGBM_PRIMITIVE_WEIGHTED_PERCENTILE=1`, the harness path already exists) and
cross-check `percentile_weighted_f32_on` against them before any Phase-19 consumer
relies on it. Until then, explicitly mark `percentile_weighted_f32_on` in its
docstring as "no independent C++ oracle yet — self-validated only," so a future
consumer does not assume reference parity.

### WR-03: exclusive prefix-sum cross-warp lanes are never cross-validated against C++

**File:** `crates/oracle-harness/tests/primitive_parity.rs:58-70,187-191,207-211`; `crates/lgbm-compute/src/kernels/primitives.rs:82-112`

**Issue:** The verbatim `ShufflePrefixSumExclusive` reference records `0` at every
warp-start lane (`idx % 32 == 0`, `idx > 0`) — it is a within-warp building block,
not a true global exclusive scan. `primitive_parity` therefore SKIPS the assert at
exactly those lanes (`is_excl_warp_boundary`), asserting only that the golden there
is `0`. Consequently, for any `n > 32` the cross-warp combination of the Rust
exclusive scan is never compared against the C++ reference; only the within-warp
values are. The Rust true exclusive scan is validated solely by its own
`primitives_self.rs` serial test. This is a documented but real C++-cross-validation
gap for the exclusive primitive at block boundaries.

**Fix:** Either capture an exclusive-scan golden from the multi-kernel
`ShufflePrefixSumGlobal` exclusive path (which IS a true global scan) so the
cross-warp combination has a reference, or document that the exclusive scan's
cross-warp correctness rests entirely on the serial self-test, not the C++ oracle.

## Info

### IN-01: `DeviceSplitInfo` read accessors panic while write paths return `Result`

**File:** `crates/lgbm-compute/src/kernels/split_info.rs:378-379,479-480,492-493`

**Issue:** `scalars()`, `cat_threshold()`, and `cat_threshold_real()` `assert!(slot <
num_leaf_slots)` (panic) on an out-of-range slot, whereas `set_scalars`,
`set_cat_thresholds`, and `copy_slot` return `ComputeError::Runtime` via
`check_slot`. The asymmetry is documented (`# Panics`) and follows Rust indexing
convention, but it is inconsistent with the V5 "typed error, never a panic" boundary
the rest of the module observes (threat T-14-04-01).

**Fix:** Consider `try_scalars`/`try_cat_threshold` `Result`-returning variants for
the device-facing boundary, or document explicitly that read accessors are
infallible-by-contract host helpers exempt from V5.

### IN-02: `primitive_parity` hardcodes `WARP = 32`, silently coupling the test to warp-32 capture

**File:** `crates/oracle-harness/tests/primitive_parity.rs:58-70`

**Issue:** The exclusive warp-boundary skip uses `const WARP: usize = 32`. If the
prefix-sum goldens are ever recaptured on a warp-64 (GFX9) device, the boundary
lanes move to multiples of 64 and the test would wrongly assert `== 0` at idx 32
(now a real running-sum value), failing or — worse — mis-passing. The coupling is
commented but not enforced (no assertion that the fixture's capture warp width
matches `WARP`).

**Fix:** Record the capture warp width in the fixture header (or a metadata record)
and assert it equals `WARP` at parse time, so a warp-64 recapture fails loudly
instead of silently mis-validating.

### IN-03: `bitonic_argsort_global_on` cap (1<<20) permits pathologically slow single-owner cpu-anchor sorts

**File:** `crates/lgbm-compute/src/kernels/primitives.rs:850,874-892`

**Issue:** `MAX_GLOBAL_ARGSORT_ELEMENTS = 1 << 20`, but the cpu-anchor skeleton
runs the entire bitonic network on a single owner (`CubeDim::new_1d(1)`), i.e.
O(n·log²n) ≈ 2.2e8 serial compare-swaps at the cap. No test exercises anywhere near
this (max is 1500), so it is not a correctness issue, but a consumer that calls near
the cap before the Phase-19/22 multi-cube hardening lands would hang for a long time
with no guard. (Perf is out of v1 review scope; flagged only because the cap is a
silent foot-gun, not an algorithmic perf tuning concern.)

**Fix:** Lower the Phase-14 skeleton cap to the regime actually exercised/intended
this phase (e.g. a few × `BITONIC_SORT_NUM_ELEMENTS`) and raise it when the genuine
multi-cube decomposition lands, so the single-owner serial path can never be invoked
on a multi-megabyte input.

---

_Reviewed: 2026-06-29T12:03:14Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
