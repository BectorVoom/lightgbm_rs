---
phase: 11-gpu-fixedpoint-int-atomics
reviewed: 2026-06-22T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs
  - crates/lgbm-compute/tests/rocm_row_partition.rs
  - crates/oracle-harness/tests/kernel_parity.rs
findings:
  critical: 0
  warning: 5
  info: 4
  total: 9
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-06-22
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 11 replaces the ROCm resident histogram BUILD's f32-atomic accumulation with
u64 two's-complement fixed-point (S=2^30) integer LDS atomics, dequantizing
`(bits as i64)/2^30 → f64` inside `fix_compact_kernel`'s widen pass.

The core design is sound. I verified the four areas the phase context flagged:

1. **Quantize/dequantize seam** — `u64::cast_from(i64::cast_from(f32::round(v*S)))`
   on build and `f64::cast_from(i64::cast_from(bits))/S` on dequant correctly
   reinterpret u64 bits as a signed i64 (a bit-reinterpret cast in cubecl, not a
   saturating numeric cast), so two's-complement negatives round-trip. The
   wrapping `Atomic<u64>::fetch_add` is a correct i64 two's-complement add.
2. **Overflow guard** — `rows * max|v| * 2^30 >= i64::MAX` returns a typed error
   (build_fix_compact_resident_f64_on:2271-2295). `max|v|` is a correct per-bin
   upper bound (a bin sums ≤ rows values each ≤ max|v|). Mathematically sound.
3. **Two's-complement negative handling** — correct via the u64 wrapping atomic.
4. **Parity-gate integrity** — `kernel_parity_resident_build_fix_compact_equals_host_on_hip`
   (kernel_parity.rs:1748) now compares the LIVE u64 GPU chain to a freshly
   constructed **CPU f64 anchor** (construct_histograms_cpu + host fix + host
   compact), NOT GPU-vs-GPU. This correctly resolves the def-f8u-01 hazard and
   adds a determinism sub-assert. Gate integrity is good.

No BLOCKER-level defects found. The findings below are robustness/error-contract
and documentation-accuracy issues, plus minor harness defects.

## Warnings

### WR-01: Naive-fallback path uses `panic!` (assert!) where the contract is a typed error

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:1877-1883`
**Issue:** When `fixed_point == true` and any feature exceeds 256 bins (so
`max_w > HIST_LDS_MAX` routes to the naive fallback arm), `resident_raw_build_into`
executes `assert!(!fixed_point, ...)`, which PANICS. This is reached from
`build_fix_compact_resident_f64_on`, which returns `Result<_, ComputeError>` and
whose sibling guards (overflow at 2288, `num_bin == 0` at 2345, region overflow at
2352) all return typed `ComputeError`. The crate's stated V5 boundary contract
(threat T-04-01, repeated throughout lib.rs) is "never panic / UB on caller input."
A panic across the FFI/library boundary on a feature with `num_bin > 256` is an
inconsistent and unrecoverable failure mode versus the rest of the path. The
in-tree comment argues this "can't happen for max_bin ≤ 255", but the function is
`pub` and the precondition is caller-supplied, not type-enforced.
**Fix:** Return a typed error instead of asserting, so the boundary contract holds
uniformly. Thread the check up into `build_fix_compact_resident_f64_on` (which can
inspect `slot_off`/`slot_len` via `slot_off_sentinel`'s `max_w`) before dispatch:
```rust
if fixed_point && max_w > HIST_LDS_MAX as u32 {
    return Err(ComputeError::Runtime {
        detail: "fixed_point u64 build requires every feature <= 256 bins \
                 (max_w <= HIST_LDS_MAX); naive >256-bin fallback is f32-only".into(),
    });
}
```

### WR-02: `slot_off_sentinel` subtracts adjacent offsets unchecked — underflow on non-monotonic input

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:1731`
**Issue:** `s.windows(2).map(|w| w[1] - w[0])` computes each feature's `feat_len`
by subtracting adjacent `u32` slot offsets. If `slot_off` is not strictly
non-decreasing, or if `slot_len < slot_off.last()`, the subtraction underflows:
debug builds panic, release builds wrap to a huge `u32` that then drives
`max_w` and the LDS/naive branch selection AND, via the sentinel, the in-kernel
`feat_len = slot_off[f+1] - slot_off[f]` used as a device loop bound. `slot_off`
is caller-supplied to the `pub` `build_fix_compact_resident_f64_on`; nothing in
this function validates monotonicity before the subtraction.
**Fix:** Validate monotonicity / sentinel sanity at the boundary and return a typed
error rather than underflow-wrap:
```rust
for w in s.windows(2) {
    if w[1] < w[0] {
        return Err(ComputeError::Runtime {
            detail: "slot_off must be non-decreasing with slot_len sentinel".into(),
        });
    }
}
```
(Requires making `slot_off_sentinel` fallible or validating in the caller before the call.)

### WR-03: Overflow guard and quantize-precision docs overstate the effective fractional precision

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:622-630, 1231-1242`
**Issue:** The build kernel quantizes with `f32::round(ord_g[k] * SCALE_F32)` — an
**f32** multiply (line 1280-1281). For values of magnitude ~1, `v * 2^30 ≈ 2^30`,
which exceeds f32's 24-bit mantissa, so the low ~6 bits of the fixed-point integer
are lost to f32 rounding BEFORE the i64 cast. The doc comment claims "S = 2^30
keeps ≥ ~9 fractional bits"; in practice, for grad/hess near 1.0 the realized
precision is bounded by the f32 product (~24 significant bits, i.e. ~6 effective
fractional bits, not 9). The measured ~5.9e-9 residual and the 1e-7 gate still
hold, so this is not a correctness defect — but the precision claim in the
load-bearing constant's doc is inaccurate and could mislead a future tightening
of the gate.
**Fix:** Either compute the quantize in f64 (`(f64::from(ord_g[k]) * SCALE_F64).round()`
— if cubecl-hip can lower the f64 round here, matching the f64 dequant) to actually
realize the documented fractional precision, OR correct the doc to state the
effective precision is bounded by the f32 grad/hess product (~24 significant bits),
not a flat "9 fractional bits", and note the residual is f32-product-rounding plus
quantize-rounding.

### WR-04: Overflow guard indexes `gradients`/`hessians` by `leaf_rows` without length validation

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:2274-2283`
**Issue:** The overflow-guard scan does `gradients[i]` / `hessians[i]` for
`i = r as usize` over every `r` in `leaf_rows`, with no prior check that
`leaf_rows` values are `< gradients.len()` / `< hessians.len()`. The same
unchecked indexing recurs at `resident_raw_build_into:1774-1775`
(`gradients[r as usize]`). If a caller passes a `leaf_rows` entry `>= num_data`
(or `gradients` shorter than the resident column count), this panics with an
index-out-of-bounds rather than a typed `ComputeError`, again violating the
"never panic on caller input" V5 contract that the rest of the function honors.
The doc comments assert this is a "caller resident contract," but it is not
validated at this `pub` boundary.
**Fix:** Add a boundary check before the scan/gather:
```rust
let n = gradients.len();
if hessians.len() != n {
    return Err(ComputeError::LengthMismatch { expected: n, actual: hessians.len() });
}
if let Some(&bad) = leaf_rows.iter().find(|&&r| (r as usize) >= n) {
    return Err(ComputeError::Runtime {
        detail: format!("leaf_row {bad} out of range for {n} gradient rows"),
    });
}
```

### WR-05: Fixed-point parity gate uses a single tiny 10-row leaf — never exercises the row-partition (P>1) or large-leaf regime it was built for

**File:** `crates/oracle-harness/tests/kernel_parity.rs:1748-1877`
**Issue:** The phase's purpose is heavy-contention large leaves (the A/B example
targets 1M rows, P up to 16). The only correctness gate pinning the live u64 path
to the CPU f64 anchor uses `num_data = 10`, `leaf_rows` of length 6, 3 features.
At this size `row_partition_count` returns P=1 (well below `ROWPART_MIN_LEAF =
256_000`), so the multi-cube LDS→global merge accumulation order the phase
introduces is NEVER validated against the anchor. The accumulation is
order-independent for integers (the test's own determinism rationale), so the
result should still be exact — but that claim is exactly what a P>1 anchor
comparison would PROVE, and it is currently untested. A merge-indexing or
sentinel bug that only manifests with multiple row-partitions per feature would
slip this gate. The `rocm_row_partition.rs` test exercises P>1 but only on the
NON-RESIDENT f32 batched launcher (explicitly noted at its header lines 17-20),
not the u64 resident path.
**Fix:** Add a second case (or parametrize) with `leaf_rows` large enough and
`LGBM_ROWPART_MIN=0` to force P>1 through the u64 resident chain, asserting the
read-back still matches the CPU f64 anchor within `FIXEDPOINT_REL_GATE`. This
closes the gap between "the example benches the heavy/P-swept regime" and "a
correctness gate covers it."

## Info

### IN-01: Stale doc on `build_fix_compact_resident_readback_f64_on`

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:2409-2415`
**Issue:** The doc still describes this as proving the resident chain equals the
host build "within the ~1e-6 f32-atomic RAW-build tolerance" and says "Not on the
live path — the live wiring is deferred." After Plan 11-01 the underlying build is
u64 fixed-point (deterministic, ~5.9e-9), and the phase context states the u64
path IS the live production build. The tolerance and "deferred" framing are stale.
**Fix:** Update the doc to reflect the u64 fixed-point chain and its tightened
deterministic gate.

### IN-02: `fix_compact_kernel` test launcher comment references f32 RAW input that no longer exists

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:2370-2371`
**Issue:** The SAFETY comment in `build_fix_compact_resident_f64_on` reads
"`h_raw` (f32 IN) and `h_f64` (f64 OUT)" but `h_raw` is now a u64 fixed-point
buffer (allocated as `u64::as_bytes` at 2306-2307 and typed `&Array<u64>` in the
kernel). Comment drift; no functional impact.
**Fix:** Correct "f32 IN" to "u64 fixed-point IN" in the SAFETY comment.

### IN-03: `mrs_u` throughput metric computed even on `overlap` and only for u64 arm

**File:** `crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs:254`
**Issue:** `mrs_u` (Mr/s) is computed from the u64 median only; the f32 arm has no
symmetric throughput print. For an A/B harness whose verdict is the relative
ratio, a one-sided throughput number is slightly asymmetric reporting. Operator
harness only — not load-bearing.
**Fix:** Either print both arms' Mr/s or drop the one-sided metric; optional.

### IN-04: Percentile helper panics on NaN samples

**File:** `crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs:140-142`
**Issue:** `pct` does `v.partial_cmp(...).unwrap()`, which panics if any timing
sample is NaN. Wall-clock `Instant::elapsed` cannot produce NaN, so this is inert
in practice, but it is a latent panic in an operator tool.
**Fix:** Use `total_cmp` (`a.total_cmp(b)`) for a total order without `unwrap`.

---

_Reviewed: 2026-06-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
