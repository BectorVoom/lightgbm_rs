# Spike Manifest

## Idea

Optimise lightgbm_rs **learning (train) speed**, GPU track: the ROCm (`cubecl-hip`,
f64) path is ~100× slower than the CPU f64 anchor on small/medium data because it is
launch-bound. Its reason to exist is winning at *large* data. Prove (or disprove) that
a dataset-size crossover exists where the GPU path beats the CPU anchor in train
wall-clock, and locate it. See `.planning/notes/gpu-track-goal-crossover-at-scale.md`.

## Requirements

- Backend stays **compile-time switched** (`--features rocm`); the CPU f64 anchor
  remains the default and the bit-exact merge gate. Speed work must not touch parity.
- "GPU is faster" is only claimed in the regime the data supports — never route
  small/medium datasets to the GPU.
- Crossover claims are wall-clock facts; the ~1e-6 GPU-vs-CPU parity contract at large
  shapes is a separate, still-open gate.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | gpu-cpu-crossover | standard | GPU train wall-clock crosses below the CPU anchor as rows scale up | ✅ VALIDATED — crossover ≈ 700k rows (feat=50/bins=255/31 leaves); GPU wins ≳1M, widening to 1.45× at 2M | gpu, rocm, performance, crossover, benchmark |
