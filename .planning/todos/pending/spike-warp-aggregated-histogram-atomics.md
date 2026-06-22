---
title: SPIKE — warp-aggregated atomics + bank-conflict layout for the GPU BUILD kernel
date: 2026-06-22
priority: medium
spike_ready: true
context: .planning/notes/gpu-build-atomic-contention-research.md
---

# SPIKE: warp-aggregated histogram atomics (device-time proxy)

**Launch with `/gsd-spike` on this file.** This is feasibility-gated, not a plan.

## Hypothesis

The f32-atomic histogram BUILD kernel (53% of GPU device-time at 1M×50) is
serialization-bound on intra-warp bin collisions. **Warp-aggregating** the atomics —
reduce all lanes hitting the same bin within a warp, then issue ONE atomic per
distinct bin — should cut atomic traffic up to 32× and shrink BUILD's CubeCL
profiler share. Layer a **bank-conflict-free LDS layout** (AMD 32-bank XOR-swizzle /
interleaved sum_g,sum_h) underneath as a cheap always-on win.

## The feasibility gate (answer FIRST, cheaply)

**Does CubeCL's `Plane` API expose `match_any`-style intra-warp bin matching**
(or ballot + shuffle/readlane primitives sufficient to build it)? The project already
uses the `Plane` API for warp-level ops (per CLAUDE.md). If the needed primitive is
absent or lowers poorly on cubecl-hip → the standout lever is blocked; fall back to
replication (weak at 255 bins) or the fixed-point-int path (research Q2).

## Measurement (the proxy — validatable on THIS box)

- Tool: CubeCL built-in profiler — repo-root `cubecl.toml` `[profiling] logger=...`
  (see `cubecl-profiling-gpu-kernel-decomposition.md`). The ONLY working GPU profiler
  on the gfx1152 APU.
- Metric: **BUILD's device-time share** before/after at 1M×50 steady-state
  (baseline 53%). Wall-clock will NOT move on 8 CUs — do not chase it (memory
  `gpu-is-spoofed-8cu-apu-not-gfx1100`).
- Repro harness: `LGBM_SCAN_PROF=1 LGBM_BENCH_SWEEP=wide cargo run --release
  --features rocm --example bench_gpu_vs_cpu`.

## Parity guardrails

- ROCm path owes ~1e-6 vs the CPU f64 anchor, NOT bit-exact. Fix warp-reduction
  order to keep it deterministic.
- Pin GPU trees to the CPU anchor for the parity gate, NEVER compare two
  nondeterministic GPU f32 paths to each other (memory `def-f8u-01-flaky-resident-hip-test`).

## Out of scope / known-weak

- R-way replication at 255 bins (LDS capacity-limited — finding #1). Only revisit
  for ≤64-bin features.
- Beating the CPU. Crossover is erased; this is future-silicon / parity-track work.
