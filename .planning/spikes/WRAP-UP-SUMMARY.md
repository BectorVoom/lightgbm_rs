# Spike Wrap-Up Summary

**Date:** 2026-06-17
**Spikes processed:** 4 (010–013)
**Feature areas:** Histogram & learning-path memory layout
**Skill output:** `./.claude/skills/spike-findings-lightgbm_rs/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 010 | histogram-pool-arena | standard | ✅ VALIDATED + SHIPPED | Histogram memory layout |
| 011 | parallel-build-scatter | standard | ❌ INVALIDATED (load-bearing) | Histogram memory layout |
| 012 | reuse-pool-across-trees | standard | ✅ VALIDATED + SHIPPED | Histogram memory layout |
| 013 | feature-splittable-arena | standard | ❌ INVALIDATED (sub-noise) | Learning-path allocation |

## Key Findings

- **Shipped (~7% large, bit-exact):** flatten the `HistogramPool` buffers `Vec<Vec<f64>>` →
  one flat arena (010, ~4%), then reuse the pool across trees instead of per-tree alloc
  (012, ~3% more). Both internal/storage-only ⇒ bit-exact vs the C++ golden.
- **Rejected with evidence:** the parallel build's per-feature `Vec<Vec<f64>>` accumulators
  are load-bearing — scatter regressed 13–21% via false sharing (011); the per-tree
  `feature_splittable` bool matrix is 0.005–0.25%/tree, not worth a refactor (013).
- **Method rule:** the cold isolated microbench overstates the warm end-to-end win 3–7×
  (allocator amortizes fixed-size per-iteration reallocs) — always confirm with
  `bench_train`. Captured in `.planning/spikes/CONVENTIONS.md`.
- **Decision rule:** flatten a per-iteration `vec![template; n]` only when MB-scale and
  not a per-thread private accumulator; per-leaf row lists are already flat (DataPartition).
- **Sweep status:** the learning-leaf `Vec<Vec<T>>` surface is exhausted. 001–009 remain
  for a future wrap-up.

## Shipped commits
- `d9cbae4` — spike 010 (flat arena)
- `5c8fa43` — spikes 012 (pool reuse) + 013 (feature_splittable not-worth-it)
- `c490905` — spike 011 (revert + load-bearing NOTE)
