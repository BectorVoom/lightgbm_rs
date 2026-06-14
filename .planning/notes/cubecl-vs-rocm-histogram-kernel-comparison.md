---
title: CubeCL vs LightGBM ROCm histogram kernel — structural comparison
date: 2026-06-15
context: /gsd-explore "Compare learning kernel in C++ hip and cubecl, then optimise cubecl kernel"
type: note
---

# CubeCL vs LightGBM ROCm histogram kernel

Baseline = AMD's official ROCm fork `LightGBM-release-4.6.0.99` (the `.cu` differs from
mainline `LightGBM/`: AMD-patched, copyright AMD 2023, `WARPSIZE`=64 on GFX9). It runs
LightGBM's **CUDA kernels on ROCm via a hipify shim** (`include/LightGBM/cuda/cuda_rocm_interop.h`,
`#define atomicAdd_block atomicAdd`) — there is no separate hand-written HIP kernel.

Reference files:
- LightGBM: `src/treelearner/cuda/cuda_histogram_constructor.cu` (kernels),
  `cuda_histogram_constructor.cpp:147` (`CalcConstructHistogramKernelDim`), `.hpp:21-27` (constants).
- Ours: `crates/lgbm-compute/src/kernels/histogram.rs` — `construct_leaf_hist_resident_lds_kernel`
  (the wired hot path) + `construct_hist_kernel_lds_f32` (single-feature LDS).

Both implement the same algorithm: privatized sub-histogram → atomic-merge to global. The
difference is **work decomposition**, and that is where the perf gap lives.

## Launch geometry — the decisive difference

**LightGBM ROCm** (`CalcConstructHistogramKernelDim`):
```
block_dim_x = max_num_column_per_partition       // FEATURES on x
block_dim_y = 504 / max_num_column_per_partition  // ROWS on y (NUM_THREADS_PER_BLOCK=504)
grid_dim_x  = num_feature_partitions              // feature groups across blocks
grid_dim_y  = max(min_grid_dim_y, ceil(ceil(N/400)/block_dim_y))  // ROWS SPLIT ACROSS BLOCKS
```
- A block holds a 2D tile (feature-columns × rows), one `__shared__` sub-hist spanning ALL its
  columns' bins. `grid_dim_y` blocks each fold a row-slice of the same features and
  `atomicAdd_system`-merge partials into global. `NUM_DATA_PER_THREAD=400` rows folded in
  registers before any shared-mem touch. A big leaf saturates the whole GPU.

**CubeCL (ours):** `CubeCount = num_features`, `CubeDim = 256`. One feature = one workgroup =
one CU; 256 lanes stride ALL the leaf's rows. No row-partitioning — a 1M-row leaf is built by
256 lanes on a single CU while the other ~95 gfx1100 CUs sit idle for that feature.

## Side-by-side

| Dimension | LightGBM ROCm (AMD) | CubeCL (ours) | Gap |
|---|---|---|---|
| Feature → work | columns packed per block (`block_dim_x`) | one cube per feature | ours wastes a CU on small-bin features |
| Rows → work | split across `grid_dim_y` blocks + 400/thread in registers | one cube strides all rows | ★ ours can't saturate the GPU on big leaves |
| Privatization | `__shared__` sub-hist, `atomicAdd_block` | LDS `SharedMemory`, `fetch_add` | same idea ✓ |
| Global merge | `atomicAdd_system` partial sums | one `fetch_add`/cell/cube | same idea ✓ |
| Precision | `gpu_use_dp` → f32 OR f64 hist | f32 (→f64 widen) | parallel designs ✓ |
| 16-bit discretized | `int16_t shared_hist` + gradient discretizer | none | ★ halves atomic width + LDS |
| Register row-batching | `NUM_DATA_PER_THREAD=400` | none (1 row/iter) | ours hits LDS atomics far more often |
| Global-mem fallback | yes (sub-hist too big) | 256-bin cap → naive path | partial |
| Subtraction / FixHist | GPU kernels | `subtract.rs` + on-GPU fix ✓ | at parity ✓ |

## Optimization levers (priority order)

1. **★ Row-partitioning (the `grid_dim_y` analog).** Biggest win. Split a feature's rows across
   multiple cubes, each a partial LDS sub-hist + atomic-merge to global. Fixes the one-CU-per-feature
   ceiling. Re-frames spike-006's "atomic/latency-bound" finding: that latency is partly **starved
   occupancy**, not just atomic cost. f32-atomic-safe, stays in the ~1e-6 gate. → spike below.
2. **Register row-batching (`NUM_DATA_PER_THREAD`).** Each lane folds K rows into a register before
   touching LDS — cuts LDS-atomic frequency ~K×. Cheap, compounds with #1. → todo.
3. **16-bit discretized histogram.** Largest structural change; halves atomic width + LDS. Interacts
   with the f32/f64 parity contract — needs its own parity analysis. → seed.

## Key constants (LightGBM `.hpp:21-27`)
`NUM_DATA_PER_THREAD=400`, `NUM_THREADS_PER_BLOCK=504`, `NUM_FEATURE_PER_THREAD_GROUP=28`,
`SUBTRACT_BLOCK_SIZE=1024`, `FIX_HISTOGRAM_BLOCK_SIZE=512`, `USED_HISTOGRAM_BUFFER_NUM=8`.

Related: [[perf-gap-vs-cpp-40-80x]], [[l3-on-gpu-fixhistogram-deferred]],
[[cubecl-cpu-runs-parallel-kernels]] (spike-006 GPU u8 invalidation context).
