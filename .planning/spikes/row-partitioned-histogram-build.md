---
title: Spike — row-partitioned histogram build (the grid_dim_y analog)
date: 2026-06-15
priority: high
type: spike
context: /gsd-explore "Compare learning kernel in C++ hip and cubecl, then optimise cubecl kernel"
---

# Spike: row-partitioned histogram build

## Hypothesis

The wired `construct_leaf_hist_resident_lds_kernel` builds each feature's histogram with
**one cube per feature (one CU)**, 256 lanes striding all the leaf's rows. LightGBM's ROCm
kernel instead splits a feature's rows across `grid_dim_y` blocks (each a partial sub-hist +
`atomicAdd_system` merge). On a large leaf, ours leaves ~95/96 gfx1100 CUs idle per feature.

**Claim to test:** adding a row-partition dimension (multiple cubes per feature, each folding a
row-slice into its own LDS sub-hist, then atomic-merging to the global slot) gives a large
speedup on big leaves — and shows that spike-006's "atomic/latency-bound" was partly **starved
occupancy**, not pure atomic cost.

## Minimal experiment

- Take a single large leaf (e.g. 256k–1M rows, the regime where the GPU should win — see
  [[gpu-track-goal-crossover-at-scale]]).
- Variant A (baseline): current one-cube-per-feature LDS kernel.
- Variant B: `CubeCount = (num_features, row_partitions)`, each `(f, p)` cube folds rows
  `[p*stride, (p+1)*stride)` into a private LDS sub-hist, then merges its slot to global.
  Sweep `row_partitions ∈ {2,4,8,16, ~CU_count/num_features}`.
- Measure: wall-clock per build, vs CPU f64 anchor for the ~1e-6 gate (f32 atomics, already
  non-deterministic order — reuse the existing parity harness).

## Success criteria

- B beats A by a clear margin on the large-leaf case (target: meaningfully closes the
  [[perf-gap-vs-cpp-40-80x]] on the build phase), bit-parity gate still green (≤1e-6 vs anchor).
- A measured occupancy / atomic-stall delta that confirms (or refutes) the starved-occupancy
  reading of spike-006.

## Watch-outs

- Global-merge atomic traffic grows from `2*num_bin` to `row_partitions * 2*num_bin` per feature
  — find the point where merge cost cancels the parallelism win (LightGBM caps via
  `min_grid_dim_y` + `NUM_DATA_PER_THREAD=400`).
- Tail leaves are small; gate the row-split on leaf size so small leaves keep the 1-cube path.
- Keep the CPU f64 anchor path untouched (bit-exact merge gate).

Reference: see [[cubecl-vs-rocm-histogram-kernel-comparison]] for the full launch-geometry diff.
