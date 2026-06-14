---
title: GPU track goal — prove a CPU-vs-GPU crossover at scale
date: 2026-06-14
context: /gsd-explore "investigate bottleneck and optimise in learning speed"
type: note
---

# GPU track goal: crossover-at-scale

## Decision

The ROCm (`cubecl-hip`, f32) path's reason to exist is **winning at large data**.
Its success criterion for the learning-speed effort is **a proven crossover dataset
size** — a shape where the GPU path beats the deterministic f64 CPU native anchor in
wall-clock. The deliverable is that crossover point itself (or proof that more launch
elimination is required before one exists).

Chosen during exploration over two alternatives:
- "kill launch overhead regardless of crossover" — rejected as a finish line; it's a
  *means*, not the goal.
- "just profile first" — folded into the crossover spike (the sweep *is* the profile).

Scale target: **whatever makes GPU win** — sweep sizes upward (50k → 200k → 1M rows ×
wider features) until GPU wins, rather than pinning a fixed realistic shape.

## Why this framing

- The CPU native path (R2, native f64 backend) is already the fast path and the
  bit-exact merge gate: ~2–4× vs C++ LightGBM 4.6, gap **growing with row count**.
- The GPU path is currently **~100× slower than CPU** on small/medium (small: GPU
  4.61s vs CPU 38.7ms) because it is **launch-bound**: ~50µs/dispatch ≫ a ≤256-bin
  scan (~1µs). The lever is **batching (fewer launches)**, NOT per-feature parallelism
  (per-feature parallelism ≈ 0 gain while launch-bound).
- The current bench harness tops out at **"large" = 20k×50**, which is *tiny* for a
  GPU. There may not be enough compute per launch at that shape for parallelism to
  amortize dispatch — so crossover (if reachable with today's kernels) likely lives at
  hundreds of thousands to millions of rows. "At scale" must be an actual measured
  number, not 20k×50.

## What already exists on the GPU track

- Batched histogram construction (lad-p3): GPU small 8.28→4.61s, medium 40.8→16.6s.
- LDS-privatized sub-histogram kernel + resident/batched build hot path wired live,
  3.5–9× (quick-260609-fw1). Host-device round-trip was measured NOT to be the
  bottleneck — see [[l3-on-gpu-fixhistogram-deferred]].
- Stated next launch-elimination steps: batched find_best_split, then subtract /
  partition batching → see seed [[batch-find-best-split-subtract-partition]].

## Next action

Spike: extend `crates/lgbm/examples/bench_train.rs` to sweep upward and find the
crossover (or prove launches still dominate at scale). See
`.planning/todos/pending/spike-gpu-cpu-crossover.md`.

Related: [[perf-gap-vs-cpp-40-80x]], [[rocm-gfx1100-available]],
[[cubecl-cpu-runs-parallel-kernels]].
