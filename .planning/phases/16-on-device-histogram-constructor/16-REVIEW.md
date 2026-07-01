---
phase: 16-on-device-histogram-constructor
reviewed: 2026-07-01T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/kernels/histogram_arena.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/subtract.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/tests/rocm_cuda_mirror.rs
  - crates/oracle-harness/tests/kernel_parity.rs
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 16: Code Review Report

**Reviewed:** 2026-07-01
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 16 adds the on-device histogram-constructor path: the §13 two-tier u64
fixed-point BUILD kernels (`construct_leaf_hist_partition_u64` + its
`_GlobalMemory` spill twin), the `dequant_leaf_hist` widen, the FixHistogram
most-freq-bin repair (`fix_histogram_mfb`), the `HistArena` handle-rotation pool,
`subtract_histogram_on_device`, and the `construct_histogram_for_leaf` orchestrator,
all behind the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam.

The numerical-parity discipline is sound: the BUILD accumulates in **u64
fixed-point atomics** (integer adds are order-independent, so the two-tier
cross-block atomic merge is bit-exact regardless of atomic ordering — the
nondeterminism the module comments worry about is confined to the f32→2^30
quantization residual, which the ABS 5e-6 envelope absorbs). No f64 leaks into the
GPU per-row scatter (D-08 holds). The `HistArena` slab sizing is `checked_mul`
overflow-guarded, the no-alias rotation invariant is asserted, and the cpu f64
anchor paths (arena rotate + `subtract_histogram_on_device` on `cpu_client`) are
bit-exact by construction.

The defects found are concentrated in the **sparse / multi-partition build path**,
which no committed test exercises (all fixtures use single-partition layouts, or
multi-partition **dense** for the spill test). The strongest finding is a V5
boundary gap: the sparse launcher never validates the CSR `data` length against
`row_ptr` contents, and its `SAFETY` comment over-claims that it does, while
launching `launch_unchecked`. Because the whole path is gated OFF and unreachable
in production this phase, these are classified WARNING rather than BLOCKER, but
they must be closed before the Phase-18/21 growth driver wires the seam on.

## Warnings

### WR-01: Sparse build path never validates CSR `data` length — `launch_unchecked` OOB risk, and the `SAFETY` comment over-claims it does

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:360-381` (V5 ladder) and `:113-116` (sparse kernel read)
**Issue:** In `construct_leaf_hist_on_device`, the dense branch validates
`data.len() >= num_columns * num_data`, but the **sparse branch validates only
`row_ptr.len() == expect_rp`** — it never bounds-checks the CSR `data` (values)
slice at all. The kernel then does an unchecked read
`data[row_start + tx]` where `row_start = row_ptr[rp_base + idx]` is a
caller-supplied CSR pointer, under `construct_leaf_hist_partition_u64::launch_unchecked`.
A malformed/oversized `row_ptr` produces an out-of-bounds **device** read. The
`SAFETY` comment on the launch (`histogram.rs:439-441`) asserts "the V5 checks
above prove ... `data`/`row_ptr` cover every partition region" — but for the sparse
path that proof does not exist in the code, so the comment is inaccurate. This
directly contradicts the T-16-03-01 promise that V5 rejects bad args *before* any
`launch_unchecked`.
**Fix:** Validate the CSR values slice in the sparse branch before upload, e.g.
require `data.len() >= *row_ptr.iter().max().unwrap_or(&0) as usize` (or the
per-partition `row_ptr[rp_base + num_data] + max_cols_in_partition` upper bound),
returning `ComputeError::LengthMismatch` on shortfall; then correct the `SAFETY`
comment to state the actual invariant that was checked.

### WR-02: Multi-partition sparse indexing is inconsistent with the dense path and is exercised by no test

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:111-119` (shared) / `:212-220` (global) sparse arm
**Issue:** The dense arm indexes the values store with the partition base included
(`data[dense_part_base + idx * ncol_p + tx]`, `dense_part_base = lo * num_data`),
but the sparse arm indexes `data[row_start + tx]` using `row_start` taken verbatim
from `row_ptr` with **no partition base added**. The only sparse fixtures
(`sparse_partition_store` + the `small_columns` corpus) yield a **single**
partition (`lo == 0`, `dense_part_base == 0`), so the discrepancy is invisible. For
`num_feature_partitions > 1`, partition `p > 0`'s rows would be read from
partition 0's region unless `row_ptr` is globally based — but the test's relay
builds `row_ptr[rp_base + r] = r * ncol_p` (partition-local), so the intended
contract is ambiguous and unverified. No committed test drives a multi-partition
**sparse** build (the spill test that has >1 partition is dense).
**Fix:** Add a multi-partition sparse anchor test (2+ partitions, `row_ptr`
populated for each) and pin it to `cpu_anchor_columns`; make the `row_ptr`
base-offset contract explicit in the kernel doc (global vs partition-local), and
if partition-local, add `+ dense/csr part_base` in the sparse read to match the
dense arm.

### WR-03: `by = (256 / bx).max(1)` silently degenerates for wide partitions, producing an over-large or mis-shaped block

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:384-385`
**Issue:** `bx = layout.max_num_column_per_partition.max(1)` becomes the block's
x-dimension and `by = (256u32 / bx).max(1)`. If a partition has more than 256
columns, integer division gives `by = 1` and the block is `bx × 1` threads — which
can exceed the device max-threads-per-block (1024) for `bx > 1024`, and more subtly
means the row-worker count collapses to 1 with no diagnostic. There is no upper
clamp on `bx` and no validation that `bx * by` is within device limits before
`CubeDim::new_2d(bx, by)`.
**Fix:** Clamp `bx` to a supported max (e.g. `min(max_num_column_per_partition,
256)` with a tail loop over columns, or split wide partitions), and/or return a
typed `ComputeError` when `bx` exceeds the device's max block dimension rather than
launching a mis-shaped grid.

## Info

### IN-01: `cuda_on_device_enabled()` seam gate is defined and unit-tested but wired to no production call site

**File:** `crates/lgbm-compute/src/lib.rs:1298-1313`
**Issue:** The `LGBM_CUDA_ON_DEVICE` gate is read/cached and asserted OFF by a test,
but nothing in the production path calls it — `construct_histogram_for_leaf` is
invoked only by tests, and `on_device_growth_supported()` independently stays
`false`. This is intentional (the doc states the growth driver that consumes it is
Phase 18/21), so it is a documented dormant seam rather than accidental dead code —
flagged only so the wiring is not forgotten when Phase 18/21 lands.
**Fix:** No action this phase; ensure the Phase-18/21 growth loop actually ANDs in
`cuda_on_device_enabled()` at the call site, else the seam never takes effect.

### IN-02: `cuda_on_device_seam_off_by_default` asserts on the real process env via a process-global cache

**File:** `crates/lgbm-compute/src/lib.rs:3512-3515`
**Issue:** The test asserts `!cuda_on_device_enabled()`, whose value is derived from
the actual `LGBM_CUDA_ON_DEVICE` env var and frozen in a `OnceLock` on first call.
A CI/dev environment that exports `LGBM_CUDA_ON_DEVICE=1`, or a future test that
initializes the cache after setting the var, would make this assertion observe a
value it cannot control. Low risk today (no test sets it), but the coupling to
ambient env + a process-global cache is fragile.
**Fix:** Acceptable as-is; if hardening is desired, document the env precondition or
gate the assertion on `std::env::var(...).is_err()`.

### IN-03: `construct_histogram_for_leaf` builds the smaller leaf even when only the larger child passes the gate

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:784-826`
**Issue:** After the "both children fail → `Ok(None)`" early-return, the smaller
leaf is unconditionally built from data regardless of `smaller_ok`. This matches
the C++ subtraction-trick requirement (the smaller histogram is needed to derive
the larger even when the smaller leaf itself will not split), so it is correct — but
the local `smaller_ok` boolean is then never used again, which reads as a latent
"forgot to guard" and could confuse a maintainer.
**Fix:** No behavior change needed; add a one-line comment that `smaller_ok` is
deliberately not a build guard (only the both-fail case short-circuits), or bind it
to `_` to signal intent.

---

_Reviewed: 2026-07-01_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
