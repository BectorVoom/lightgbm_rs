---
phase: quick-260619-ol8
plan: 01
subsystem: lgbm-compute (rocm histogram kernels — measurement only)
tags: [benchmark, rocm, gfx1100, launch_unchecked, histogram, measurement-only]
requires: [crates/lgbm-compute/src/kernels/histogram.rs, crates/lgbm-compute/src/kernels/split.rs]
provides: [crates/lgbm-compute/examples/launch_unchecked_ab.rs, 260619-ol8-FINDINGS.md]
affects: []
tech-stack:
  added: []
  patterns: [dual-kernel-single-binary-interleaved-A/B, checked-vs-unchecked-twin, warm-median-spread, same-input-drift-guard]
key-files:
  created:
    - crates/lgbm-compute/examples/launch_unchecked_ab.rs
    - .planning/quick/260619-ol8-benchmark-the-launch-unchecked-swept-his/260619-ol8-FINDINGS.md
  modified: []
decisions:
  - "launch_unchecked is a robust ~1.8x compute-bound win ONLY for the f64 deterministic fused kernel; NULL/sub-noise for the f32-atomic and resident-LDS kernels (their bottleneck is atomics/latency/barriers, not the bounds-check codegen)."
  - "The nrw launch_unchecked sweep stays as-is — justified for fused, harmless (neutral) for atomic/LDS."
metrics:
  duration: ~25m
  completed: 2026-06-19
---

# Phase quick-260619-ol8 Plan 01: Benchmark the launch_unchecked-swept histogram kernels Summary

Quantified, on the real gfx1100, the per-launch overhead of `#[cube(launch)]` (bounds-check codegen) vs `#[cube(launch_unchecked)]` for the 3 hot-loop production histogram kernels via a dual-kernel single-binary interleaved A/B: the f64 deterministic FUSED kernel gets a robust, sign-stable ~9–16% (launch-bound) / ~40–46%≈1.8× (compute-bound) win, while the f32-atomic and resident-LDS kernels are NULL / sub-noise.

## What Was Built

- **`crates/lgbm-compute/examples/launch_unchecked_ab.rs`** (755 lines) — rocm-gated A/B micro-bench:
  - 3 bench-only `_checked` twin kernels (`construct_hist_kernel_atomic_f32_checked`, `construct_leaf_hist_resident_lds_kernel_checked`, `build_fix_scan_fused_kernel_checked`) with bodies copied VERBATIM from the shipped kernels in `histogram.rs` — only `#[cube(launch)]` vs the shipped `#[cube(launch_unchecked)]` differs, isolating exactly the bounds-check codegen.
  - Interleaved checked/unchecked launches per timed iteration, WARMUP discard, median + p25/p75 spread, device-sync (read-back) forced inside every timed call.
  - 3 kernels × {launch-bound, compute-bound} × {16, 64, 256} bins.
  - Same-input sanity asserts: f32-atomic envelope (ABS 5e-6 / REL 1e-5) for atomic+LDS, BIT-EQUAL for the fused f64 kernel — also the runtime guard that catches twin-vs-shipped drift (the SYNC WARNING).
  - CPU-only stub `main()` ("requires --features rocm") + the rocm `main()` body.
- **`260619-ol8-FINDINGS.md`** — the real gfx1100 numbers from 2 process runs, per kernel / per regime, with the honest MEASURABLE-vs-SUB-NOISE verdict.

## Key Results (gfx1100, 2 process runs)

| Kernel | launch-bound | compute-bound | verdict |
|--------|:-------------|:--------------|:--------|
| f32-atomic | deltas within spread, signs flip | small, signs flip | **NULL** |
| resident-LDS (P=1) | marginal +4–5% at 256 bins, within spread | flips / within spread | **effectively NULL** |
| fused build+fix+scan (f64) | +9–16%, sign-stable | +40–46% (≈1.8×), NON-overlapping spread | **MEASURABLE, large, robust** |

The fused-kernel win is pure bounds-check codegen (the f64 deterministic path makes both arms bit-identical), surfacing because that kernel runs long single-unit sequential loops (zero/build over bin×row, fix, compact, scan) where a per-access bounds branch compounds. The atomic/LDS kernels are atomic-contention / barrier / launch-latency bound, where the codegen is negligible — confirming mwr's transfer-/latency-bound-masks-launch-overhead expectation for those two.

## Verification

- `cargo build -p lgbm-compute` (CPU-only) — compiles (rocm twins cfg'd out, stub main).
- `cargo build -p lgbm-compute --features rocm --example launch_unchecked_ab` — compiles.
- `cargo run --release -p lgbm-compute --features rocm --example launch_unchecked_ab` — ran to completion on gfx1100 across 2 separate processes; all same-input sanity asserts passed (no panic) in both.
- `git diff --stat` for the task commits shows ONLY `crates/lgbm-compute/examples/launch_unchecked_ab.rs` — NO change to `histogram.rs`, `split.rs`, `learner.rs`, `lib.rs`, or any CPU anchor (T-ol8-03 satisfied).

## Deviations from Plan

**1. [Rule 3 - Blocking] cubecl launch API form**
- **Found during:** Task 2 (first rocm build).
- **Issue:** The plan suggested `_checked::launch(...)` as a SAFE call and `ScalarArg::new(...)` / `ArrayArg::from_raw_parts::<T>(&h, len, vec)` arg forms. In cubecl 0.10.0 (this workspace) the generated `::launch` is `unsafe` (same as `::launch_unchecked`), `ArrayArg::from_raw_parts(handle.clone(), len)` takes the handle by value with 2 args (no vectorization arg, no turbofish), and scalars are passed as plain values — matching the existing `gpu_row_partition.rs` / `gpu_multifeature.rs` examples and the shipped launchers in `histogram.rs`.
- **Fix:** Used the 2-arg `from_raw_parts(handle.clone(), len)` + plain scalars, and wrapped the `::launch` checked-arm calls in `unsafe { }` (the bench's `unsafe` is confined to these launch sites, with a SAFETY comment per call stating the host-proven in-range obligations). The A/B still isolates exactly the launch attribute (`launch` vs `launch_unchecked`); uploads/cube-count/readback are identical between arms.
- **Files modified:** `crates/lgbm-compute/examples/launch_unchecked_ab.rs` only.
- **Commit:** 8899b18.

**2. [Rule 3 - Blocking] `#[cube]` macro needs prelude traits at item scope**
- **Found during:** Task 1 (first rocm build — `dynamic_cast` / `CompilationArg` not in scope).
- **Issue:** Placing `use cubecl::prelude::*;` INSIDE each twin fn body (as initially drafted) does not bring the traits into the scope the `#[cube]` macro expansion needs.
- **Fix:** Added a single rocm-gated module-level `use cubecl::prelude::*;` (mirroring `histogram.rs`'s module-level import) and removed the per-fn inner `use`s. The CPU-only build keeps the import cfg'd out.
- **Files modified:** `crates/lgbm-compute/examples/launch_unchecked_ab.rs` only.
- **Commit:** 18524f3.

## Known Stubs

None. The fused-arm leaf scalars are synthetic but plausible (computed from the gathered grad/hess); `sum_hessian_bumped == sum_hessian_raw` (no 2*kEps bump) and `most_freq_bin=0`/`offset=0` deliberately skip FIX/COMPACT — this is a TIMING-only run, not a parity run, and is documented in the bench + FINDINGS.

## Self-Check: PASSED
- FOUND: crates/lgbm-compute/examples/launch_unchecked_ab.rs
- FOUND: .planning/quick/260619-ol8-benchmark-the-launch-unchecked-swept-his/260619-ol8-FINDINGS.md
- FOUND commit: 18524f3 (Task 1)
- FOUND commit: 8899b18 (Task 2)
