---
spike: 053
name: cuda-build-launchconfig-autotune
type: standard
validates: "Given the APU-tuned row_partition_count under-partitions to P=1 (spike-040 latent mis-tune), when the build launch-config / BUILD_PSET ceiling is re-swept/autotuned against real NVIDIA SM count, then the histogram build phase speeds up bit-exact"
verdict: REFUTED
related: [051, 040, 037]
tags: [gpu, cuda, kaggle, autotune, occupancy, row-partition, build, refuted]
---

# Spike 053: CUDA build launch-config autotune — PREMISE REFUTED by 051

## What This Was Going To Validate

The hypothesis (from the spoofed-APU spike-040): the build under-partitions to P=1 at the
50-feature width, and autotune's `BUILD_PSET = [1,4,8,16,32]` ceiling (32) is too low for a
big NVIDIA GPU — so lifting the row-partition `P` / PSET ceiling should speed the build.

## Why It Was Not Built — Refuted by Spike-051

Spike-051 ran the decisive probe **inside its scope** (a zero-code `LGBM_AUTOTUNE_FORCE_P`
occupancy sweep on real CUDA), which **refuted this spike's entire premise** before any code:

- The build `hist+split` device-time is **flat-to-slightly-WORSE** as P rises
  (P=1: 4897ms → P=16: 5481ms → P=128: 5261ms). **P=1 is optimal.** There is no occupancy
  headroom to reclaim at the narrow 50-feature shape on a real NVIDIA GPU.
- Lifting the `BUILD_PSET` ceiling (>32) — this spike's deliverable — would do **nothing**
  (P=64/128 don't beat P=1).
- Worse, **autotune slightly UNDERperforms** the plain P=1 heuristic on cuda (`LGBM_AUTOTUNE=0`
  was the fastest arm, ~4% better than default-autotune) — so the build is already
  over-tuned, not under-tuned, on real hardware.

The APU's P-sensitivity (spike-040, ~10%) is an **artifact of the spoofed 8-CU APU** and does
NOT transfer to discrete CUDA. The build is launch-bound, not occupancy/throughput-bound
(see 051 + 052).

## Salvage

The one real cheap win in this neighborhood: **`LGBM_AUTOTUNE=0`** (force the P=1 heuristic)
is ~4% faster than default-autotune on the narrow CUDA shape. Consider it the cuda
narrow-shape default, or skip the cold-tune when BUILD_PSET's optimum is P=1.

## Verdict

**REFUTED (premise) — slot repurposed.** See `051-real-cuda-hist-reattribution/README.md`
(Finding 1) for the evidence. No code written.
