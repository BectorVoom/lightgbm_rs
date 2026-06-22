---
title: Spike — f32 parallel-atomic on-device build+scan vs the seq-f64 9.6× gap
date: 2026-06-22
priority: high
type: spike
run_with: /gsd-spike
---

# Spike: f32 parallel-atomic on-device scan (ROCm)

## Hypothesis

The GPU SCAN being 9.6× the CPU scan ([[gpu-bottleneck-moved-to-seq-f64-scan]]) is
caused by the **sequential-f64** resident build+scan fallback on gfx1100 (no fast f64),
not by f64 arithmetic being intrinsically required. A **parallel-f32-atomic** on-device
build+scan (gfx1100 = atomic-capable) should recover most of the gap while staying
within the ~1e-6 ROCm parity contract vs the CPU f64 anchor.

## Questions to answer

1. **Decompose the 8.4s/train scan** (250k×500) into on-device sequential-f64 *compute*
   vs per-leaf *launch/read latency*. (If it's mostly launch latency, the fix is
   batching reads, not changing precision — different lever. Profiler shows compute is
   the likely bulk, but confirm.)
2. **Prototype** a parallel-f32-atomic build+scan kernel for one leaf; measure
   speedup vs `build_fix_scan_resident`.
3. **Parity**: f32 GPU scan vs CPU f64 anchor — does it hold ≤ ~1e-6 on 250k/500k/1M×500?

## Done-when

A measured speedup number for the f32 parallel-atomic path on 250k×500 (and ideally
1M×500), plus a parity verdict vs the CPU anchor. Result decides whether
[[f32-rocm-parallel-scan-path]] gets promoted to a phase.

## Guardrails (from prior campaign learnings)

- WARM-vs-COLD: cold ceiling overstates warm win 3–7×; use the bench warmup/median harness.
- Pin GPU trees to the CPU f64 anchor for parity; never compare two nondeterministic
  GPU f32 paths to each other at 1e-6 (def-f8u-01 lesson).
- Keep the CPU f64-fold path byte-identical — it's the hard merge gate.

## Reproduce the baseline gap

```
LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide cargo run --release --features rocm --example bench_gpu_vs_cpu
```
