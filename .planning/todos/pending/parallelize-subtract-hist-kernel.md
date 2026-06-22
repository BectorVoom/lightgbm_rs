---
title: Parallelize subtract_hist_kernel_f32 (GPU) — single-thread → workgroup, bit-exact
date: 2026-06-22
priority: medium
type: lever
run_with: /gsd-quick
---

# Parallelize the GPU histogram-subtract kernel

## Why

CubeCL profiling ([[cubecl-profiling-gpu-kernel-decomposition]]) found `subtract_hist_kernel_f32`
is **~32% of GPU device-kernel time** (1M×50, steady-state) and runs **single-threaded**:

```rust
// crates/lgbm-compute/src/kernels/subtract.rs
#[cube(launch)] pub fn subtract_hist_kernel_f32(parent, child, out: &mut Array<f32>) {
    if UNIT_POS == 0 { for i in 0..parent.len() { out[i] = parent[i] - child[i]; } }
}
// launched CubeCount::Static(1,1,1), CubeDim::new_1d(1)
```

Only thread 0 does the whole elementwise `parent − child` over ~25k elements on 1 of ~512 lanes.

## The fix (trivial + BIT-EXACT)

Launch a real workgroup (e.g. `CubeDim::new_1d(256)` × enough cubes to cover `parent.len()`)
and have each thread own disjoint elements via `ABSOLUTE_POS` (grid-stride loop):
```rust
let stride = CUBE_COUNT * CUBE_DIM;  // or appropriate
let mut i = ABSOLUTE_POS;
while i < parent.len() { out[i] = parent[i] - child[i]; i += stride; }
```
Each `out[i]` is independent → **byte-identical** to the current single-thread result (same
f32 subtract, no atomics, no reduction, no ordering). No ~1e-6 relaxation needed — this is
bit-exact, unlike the atomic BUILD.

## Scope

- Edit `subtract_hist_kernel_f32` body + its launcher dims (the f32/hip path only; the f64
  CPU-anchor `subtract_hist_kernel` is byte-unchanged — CPU merge gate untouched).
- Also consider the same single-thread pattern in `fix_compact` / scan if cheap (lower ROI:
  fix 1.9%, scan 11% but scan is a prefix-reduction, not embarrassingly parallel).
- Gate: bit-exact `cargo test -p lgbm-treelearner --lib` + `oracle-harness`; add a
  GPU subtract parallel==serial parity check if a rocm test harness exists.
- Measure with CubeCL profiling (recipe in the note) before/after: subtract device-time
  share should drop from ~32% toward ~0.

## Honest framing

On THIS 8-CU APU the GPU loses to the CPU overall ([[wide-tall-two-backend-root-cause]]),
so this won't flip routing here — its value is (a) correctness (no kernel should run 1-lane),
(b) bit-exact + zero risk, (c) real payoff on discrete gfx110x where the GPU path matters.
Reproduce the baseline: enable cubecl.toml profiling, run
`LGBM_BENCH_SWEEP=wide LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=50 LGBM_BENCH_ITERS=1 cargo run --release --features rocm --example bench_gpu_vs_cpu`.
