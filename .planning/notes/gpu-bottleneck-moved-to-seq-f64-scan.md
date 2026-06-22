---
title: GPU train bottleneck moved to on-device sequential-f64 SCAN (post-p9v)
date: 2026-06-22
context: gsd-explore — "investigate gpu kernel bottleneck on large data"
---

# GPU train bottleneck moved to the on-device sequential-f64 SCAN

## TL;DR

After `quick-260621-p9v` (resident-bin upload hoist), the GPU training bottleneck on
wide shapes is **no longer the histogram kernel or the upload** — it is the
**split-finding SCAN**, which is **~96% of GPU train wall** and **9.6× slower than the
identical scan on the CPU anchor**. Root cause: the GPU resident scan does a
**sequential-f64** build+scan on-device (`build_fix_scan_resident`) because **gfx1100
has no fast f64** — so it can't run the parallel-f64-atomic path the upstream CUDA
kernel assumes. The campaign's mental model (kernel/upload are the cost) is now stale.

## Measurement (LGBM_PHASE_PROF=1, LGBM_BENCH_SWEEP=wide, gfx1100, iters=8, leaves=31, 3 reps)

| shape      | GPU train | scan % of phases | hist+split % | partition % | binning (1-time) | resident upload |
|------------|-----------|------------------|--------------|-------------|------------------|-----------------|
| 250k×500   | 9.70s     | **97.0%**        | 98.5%        | 1.5%        | ~1.5s            | 0.30s (~2.7%)   |
| 500k×500   | 12.76s    | **96.5%**        | 98.1%        | 1.9%        | ~3.0s            | 0.63s           |
| 1M×500     | 19.70s    | **95.9%**        | 97.6%        | 2.4%        | ~6.1s            | 1.29s           |

- On-device histogram **build** ≈ 133ms/train (~1.5%) — thoroughly optimized, effectively solved.
- grad/hess, score-update, metric, in_iter_other: all <2.5% combined.

## CPU-vs-GPU comparison (same shape, same instrumentation, 250k×500)

| path             | train  | scan / train | vs CPU |
|------------------|--------|--------------|--------|
| CPU f64 anchor   | 1.80s  | 871ms        | 1×     |
| GPU (gfx1100)    | 9.70s  | **8,397ms**  | **9.6×** |

The scan is GPU-path-**specific**: same host-visible scan logic, but under the `rocm`
backend the leaf scan routes through `backend.scan_resident_leaf` /
`build_fix_scan_resident` (learner.rs:1967–1984), whose comment states it
*"BUILDS (sequential f64)"* on-device.

## Precision nuance (decides the fix)

Upstream LightGBM ships **two** GPU kernels at two precisions:

| upstream path | accumulator | source |
|---|---|---|
| OpenCL `device=gpu` (`gpu_tree_learner`, `histogram*.cl`) | **`gpu_hist_t = float`** (f32) | legacy/approximate |
| CUDA `device=cuda` (`cuda_histogram_constructor.cu`, `cuda_best_split_finder.cu`) | **`hist_t = double`** (f64; `ShuffleReduceSum<double>`) | modern; the AMD 4.6 HIP fork |

Our `cubecl-hip` mirror followed the **CUDA/HIP (f64)** path. Upstream's CUDA does
*parallel f64 atomics* (assumes datacenter NVIDIA f64). gfx1100 = "Plane YES / f64 NO /
atomic YES" → no fast f64 → sequential-f64 fallback → the 9.6×.

## The unlock (user-confirmed constraint 2026-06-22)

**f32 on the GPU is acceptable to the user.** This is sanctioned because:
- The project contract already holds the **ROCm path to ~1e-6, not bit-exact** (CPU
  f64-fold is the hard gate). Dropping GPU scan f64→f32 costs nothing contractual.
- Upstream's OpenCL kernel (`gpu_hist_t = float`) is precedent that f32 GPU
  histogram/split is a real, blessed LightGBM mode.
- gfx1100 is f32 + atomic capable → an f32 build+scan can be **parallel atomic** (fast).

**Fix direction:** replace the on-device sequential-f64 resident build+scan with a
**parallel-f32-atomic** on-device build+scan on the ROCm backend, held to ~1e-6 vs the
CPU anchor. See spike [[spike-f32-parallel-atomic-onfevice-scan]] and seed
[[f32-rocm-parallel-scan-path]].

## How to reproduce

```
LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide cargo run --release --features rocm --example bench_gpu_vs_cpu
# CPU comparison at one wide point:
LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide LGBM_BENCH_ROWS=250000 LGBM_BENCH_FEAT=500 cargo run --release --example bench_gpu_vs_cpu
```
