---
title: 16-bit discretized histogram path (gradient discretizer)
trigger_condition: Row-partitioning saturates the GPU and atomic WIDTH (not occupancy) becomes the next histogram-build bottleneck
planted_date: 2026-06-15
type: seed
context: /gsd-explore "Compare learning kernel in C++ hip and cubecl, then optimise cubecl kernel"
---

# Seed: 16-bit discretized histogram path

LightGBM's ROCm kernel has a discretized variant (`CUDAConstructDiscretizedHistogram*Kernel`,
`__shared__ int16_t shared_hist[...]`, driven by `cuda_gradient_discretizer.cu` and the
`use_quantized_grad_` / `num_bits_in_histogram_bins <= 16` switch). It quantizes gradients/hessians
to int16 and accumulates a 16-bit shared histogram — **halving both atomic width and LDS
footprint** vs our full-f32 cells. There is also a `USE_16BIT_HIST` global-memory fallback.

We have **no discretized path** — we accumulate f32 grad/hess into f32 LDS cells.

## Why a seed, not a todo

- Largest structural change of the three levers — needs a gradient-discretizer stage + a parity
  story for the int16 quantization, which **interacts directly with the f32/f64 ~1e-6 parity
  contract** ([[def-07-02-histogram-compaction-root-cause]] is the kind of subtlety to expect).
- Only pays off once **occupancy** is already solved — otherwise we're optimizing atomic width on
  an under-utilized GPU. Sequence it after [[row-partitioned-histogram-build]] and
  [[register-row-batching-histogram]].

## Trigger

Pull this forward when profiling (post row-partitioning) shows atomic **bandwidth/width** — not
occupancy or launch overhead — is the dominant histogram-build cost at the target scale.

## First steps when triggered

- Read `cuda_gradient_discretizer.cu` + the `CUDAConstructDiscretizedHistogramDenseKernel`
  (`cuda_histogram_constructor.cu:251`) and the `num_bits_in_histogram_bins` selection.
- Design the int16 quantization + its own parity gate vs the cpu f64 anchor BEFORE writing kernels.

Reference: [[cubecl-vs-rocm-histogram-kernel-comparison]].
