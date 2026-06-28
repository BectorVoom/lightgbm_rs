# Phase 14 — Pre-On-Device CUDA Benchmark Baseline

**Date:** 2026-06-29

On the current HEAD code (== `origin/master`; the 16 local commits ahead are docs-only with zero code diff), official Microsoft LightGBM's CUDA path is **~4.46× faster** than LightGBM-rs on a real NVIDIA Tesla T4 at 50 features, cold.

## Run configuration

- Hardware: Kaggle GPU — real NVIDIA Tesla T4.
- Workload: sklearn `make_classification`, 500,000 samples × 50 features, seed 42.
- Objective: binary objective + `binary_logloss`.
- `num_leaves` = 31; `learning_rate` = 0.1; `n_estimators` = 100; `device_type` = cuda.
- COLD / no warmup (`continue_benchmark.py` does not warm up).
- Built from `origin/master` (== current HEAD; the 16 local commits ahead are docs-only, zero code diff).
- Source: Kaggle kernel `boomvector/lgb-rs-cuda-bench` v9, run 2026-06-29.

## Results

| Implementation   | CUDA train time (s) |
| ---------------- | ------------------- |
| Official LightGBM | 3.36                |
| LightGBM-rs       | 14.98               |

→ official LightGBM is ~4.46× faster (cold, @50 features).

## Cold-vs-warm caveat

Cold-start overstates the gap because CubeCL kernels JIT-compile on first use. The prior WARM spike-051..054 baseline was 3.9×@50f / 1.9×@500f, so this cold 4.46×@50f is CONSISTENT with that baseline (not a regression), per the spike-findings "cold-ceiling-overstates-warm" rule.

## Phase 14 is perf-neutral

Phase 14 (Slice 0) is a no-kernel scaffold — `LGBM_CUDA_ON_DEVICE` is off by default, the `grow_tree_on_device` seam returns `Ok(None)`, and the `GpuBackend` on-device discriminator stays false — so it changes no compute. This baseline is therefore the PRE-on-device measurement.

## Baseline contract

The on-device CUDA tree learner (Phase 15 Slice 1 onward) must materially beat this 14.98 s / ~4.46×-slower baseline; this is the number Slice 1+ improvement is measured against.
