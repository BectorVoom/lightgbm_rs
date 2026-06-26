---
phase: quick-260627-agx
plan: 01
subsystem: lgbm-compute / kernels
status: complete
tags: [performance, gpu, rocm, cpu, vectorization, vector, subtract, bit-exact, cubecl]
requires:
  - spike-041 (Vector<P,N> launch recipe, VALIDATED)
provides:
  - subtract_hist_kernel_vec<F,N> (SIMD-vectorized element-wise subtract)
  - width-gated dispatch in all 3 subtract launchers
affects:
  - portable cuda/wgpu + generic-f64 subtract path (subtract_histograms_f64_on / _f32_on)
  - rocm resident subtract hot path (subtract_histograms_f64_from_handles_on)
tech-stack:
  added: []
  patterns:
    - "Vector<P: Scalar, N: Size> SIMD over Array<Vector<F,N>> (cubecl 0.10)"
    - "divisibility-gated dispatch (pick widest io_optimized width dividing n, else scalar)"
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/subtract.rs
decisions:
  - "Width gate uses the MAX io_optimized width via .next(); use only if it divides n exactly and >1, else scalar — no tail logic (lowest-risk, matches plan)"
  - "CPU f64 anchor subtract_histograms_cpu_native and the two scalar kernels left byte-untouched"
metrics:
  duration: ~25m
  completed: 2026-06-27
  tasks: 2
  files: 1
  commits: 1
---

# Phase quick-260627-agx Plan 01: Vectorize CubeCL Subtract Kernel with Vector<F,N> Summary

Wired the spike-041 `Vector<P,N>` SIMD vectorization into the production element-wise
histogram-subtract kernel, width-gated so the divisible production shape (256 bins → `2*num_bin`
divisible by the backend's max width) runs vectorized loads/stores while mixed-cardinality /
odd lengths fall back to the proven scalar kernels — byte-identical on every cell, CPU f64
anchor untouched, full merge gate (CPU goldens + rocm parity + rocm release build) green.

## What Was Built

**Task 1 — vectorized kernel + width-gated launchers** (commit `368a4e8`):
- New `#[cube(launch)] subtract_hist_kernel_vec<F: Float, N: Size>` over
  `&Array<Vector<F, N>>` — the same 1D grid-stride loop as the scalar kernels but each
  lane subtracts a whole `Vector<F, N>` (`out[i] = parent[i] - child[i]`, `while i < n_vec`).
  Bit-exact-by-construction: `Vector::sub` is element-wise, no float reorder, no
  atomics/reduction (spike-041 + CONVENTIONS 313–351).
- New private `pick_vec_width<R>(client, elem_size, n)` helper: returns the widest
  `client.io_optimized_vector_sizes(elem_size)` width (`.next()`, widest-first) when it is
  `> 1` AND `n % width == 0`, else `1` (sentinel = use scalar). No tail logic.
- Width-gated dispatch added to all three launchers, each preserving public signature,
  empty/zero guards, length-mismatch errors, read-back, and SAFETY contract:
  - `subtract_histograms_f64_on<R>` — portable/generic f64 path
  - `subtract_histograms_f32_on<R>` — no-f64 hip / f32 path
  - `subtract_histograms_f64_from_handles_on<R>` (cfg rocm) — the resident hot path
    (Handle-in/Handle-out, NO read-back)
  When `vs > 1`: launch `subtract_hist_kernel_vec::launch::<F, R>(client, count, dim, vs,
  …from_raw_parts(h, n/vs), n/vs)` (length in vector units over the same byte buffer; `vs`
  the runtime `N` arg right after `CubeDim`). Else: the existing scalar kernel verbatim.
- Two new unit tests (`subtract_vec_equals_serial_f64` / `_f32`) at the width-DIVISIBLE
  length `256000` (500 feat × 256 bin × 2), asserting `to_bits()` equality vs serial
  `p - c` on every cell — proving the vectorized branch itself is bit-exact on the cpu
  client (the existing 12345-length cases cover the non-divisible scalar fallback).

**Task 2 — full merge gate** (no code change; verification only).

## Merge Gate Results (all green)

| Gate | Result |
|------|--------|
| `cargo test -p lgbm-compute --lib subtract` | **10 passed**, 0 failed (incl. new `subtract_vec_equals_serial_f64/_f32`) |
| `cargo test -p lgbm-treelearner --lib` | **77 passed**, 0 failed, 2 ignored |
| `cargo test -p lgbm` | **41 passed** + doc-tests 0, 0 failed |
| `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` | **1 passed** (C++ golden parity) |
| `cargo test -p oracle-harness --features rocm` | all suites green — `kernel_parity_subtract_bit_exact_on_cpu`, `hip::kernel_parity_subtract_within_tol_on_hip`, `learner_parity_subtract`, `learner_parity_growth_path_subtract`, resident `hip::kernel_parity_resident_build_fix_compact_*`, kernel_parity.rs **21 passed** |
| `cargo build --release --features rocm` | **Finished** (3m08s) — resident path compiles vectorized |

No golden/fixture file modified anywhere in the diff (`git diff --stat ba5c365 HEAD` = only
`crates/lgbm-compute/src/kernels/subtract.rs`). ROCm IS available on this host (spoofed
gfx1100/gfx1152 APU); the `--features rocm` parity gate was actually run, not skipped.

## Deviations from Plan

None — plan executed as written. One in-scope interpretation choice, consistent with the
plan text: `pick_vec_width` selects the MAX `io_optimized` width via `.next()` and uses it
only if it divides `n` exactly (else scalar), rather than searching for the widest *divisor*.
This is the lowest-risk gate the plan specified ("`.next()` filtered so the result is `> 1`
AND `n % width == 0`, else `1`") and matches the spike-041 "on hip sweep to the MAX width"
finding; the production all-256-bin shape divides the max width on every backend.

## Bit-Exact / Parity Notes (the non-negotiable gate)

- The vectorized path is bit-exact **by construction** (`Vector::sub` is element-wise; no
  float op reordered; no atomics/reduction) — confirmed on the cpu client by the two new
  `to_bits()` tests at the divisible production length, and on hip by the existing
  `hip::kernel_parity_subtract_within_tol_on_hip` + learner_parity subtract suites.
- `subtract_histograms_cpu_native` (the f64 CPU anchor / hard merge gate) and the scalar
  `subtract_hist_kernel` / `subtract_hist_kernel_f32` kernels are byte-untouched — they
  remain the fallback for non-divisible lengths.
- No parity regression occurred, so no launcher needed reverting to scalar-only.

## Known Stubs

None. No tail handling for non-divisible lengths is implemented by design — those shapes
take the proven scalar kernel. "Mixed-cardinality tail vectorization" is documented in a
doc comment as a possible follow-on (ROI is ROCm-parity-track and bounded; subtract is a
non-dominant phase on an APU that loses to the 16-core CPU).

## Self-Check: PASSED

- FOUND: crates/lgbm-compute/src/kernels/subtract.rs (modified; contains `subtract_hist_kernel_vec` and `Vector<`)
- FOUND: commit 368a4e8 (feat — vectorized kernel + width-gated launchers)
- All merge-gate suites green; no golden/fixture file changed.
