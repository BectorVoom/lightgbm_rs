---
title: Batch find_best_split / subtract / partition (GPU launch elimination)
trigger_condition: The crossover spike shows GPU launches still dominate wall-clock at the target scale
planted_date: 2026-06-14
type: seed
context: /gsd-explore "investigate bottleneck and optimise in learning speed"
---

# Seed: batch find_best_split / subtract / partition on GPU

## Idea

The GPU path is launch-bound (~50µs/dispatch ≫ ≤256-bin scan ~1µs). Histogram
construction is already batched (one launch/leaf, lad-p3) + LDS resident pool (fw1).
The remaining per-leaf/per-feature launches are in **find_best_split**, **histogram
subtract**, and **data partition**. Batching each to ~one launch per leaf (or per
tree level) cuts the launch count that scales with features → the dominant cost while
launch-bound.

## Trigger — REFRAMED 2026-06-14 by spike 001

Spike 001 (`001-gpu-cpu-crossover`) proved GPU already crosses CPU at ~700k rows with
today's kernels, so this is **no longer a prerequisite** for "GPU wins at scale" — it's
now an **optimisation** to *lower the crossover*. The GPU has a large launch-bound
floor (~5–6s for 30 iters, flat for rows ≤100k); batching find_best_split / subtract /
partition shaves that floor → moves the crossover **left** so GPU wins at smaller
datasets (below 700k). Quantified payoff: every second cut from the floor pulls the
crossover down by ~47k-rows-worth of CPU work.

Activate when: we want GPU to beat CPU below ~700k rows, OR profiling at the crossover
region shows dispatch count still dominating. Do NOT do it speculatively.

## Notes from prior art

- Roadmap order: batched **find_best_split** first (one launch/leaf), then subtract,
  then partition. See [[perf-gap-vs-cpp-40-80x]].
- find_best_split batching needs a careful `scan_leaf_histogram` refactor that
  preserves: GOSS parent-splittability, the split gates, and feature scan order — all
  required for the ~1e-6 contract. Default-loop trait method stays CPU bit-exact.
- Per-feature parallelism gives ≈ 0 gain while launch-bound — the win is fewer
  launches, not more threads.
- Each step is phase-sized and needs its own plan + the ~1e-6 ROCm parity gate.

Related: [[gpu-track-goal-crossover-at-scale]] (the note),
[[rocm-gfx1100-available]], [[cubecl-cpu-runs-parallel-kernels]].
