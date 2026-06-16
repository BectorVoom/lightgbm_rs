---
name: spike-findings-lightgbm_rs
description: Implementation blueprint from spike experiments. Proven patterns + verified rules for optimising Vec<Vec<T>> / per-iteration allocation in the lightgbm_rs CPU training path (histogram pool, parallel build, per-leaf structures). Auto-loaded during training-path performance work.
---

<context>
## Project: lightgbm_rs

Pure-Rust LightGBM port (CubeCL, CPU + ROCm). The CPU f64 anchor must stay bit-exact to
C++ LightGBM 4.6. These findings come from a train-speed perf campaign — specifically the
"optimise `Vec<Vec<T>>` in learning" thread, which swept the training/leaf path's nested
and per-iteration allocations for cache/allocation wins that hold the bit-exact contract.

Spike sessions wrapped: 2026-06-17 (spikes 010–013). Earlier perf spikes 001–009 (GPU
crossover, columnar build, u8 bins, feature-parallel, row-partition) are NOT yet wrapped.
</context>

<requirements>
## Requirements

- The CPU f64 anchor stays **bit-exact to C++** — gate every change with
  `cargo test -p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`),
  `cargo test -p lgbm-treelearner --lib`, `cargo test -p lgbm`.
- `Vec<Vec<T>>` is **not categorically a pessimization** — decide per-instance by usage
  (the decision rule in the reference). Do not blanket-flatten.
- **Ship on the end-to-end `bench_train` number, not the isolated microbench** — the cold
  ceiling overstates the warm win 3–7×.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| Histogram & learning-path memory layout | references/histogram-learning-memory-layout.md | Flatten + reuse the histogram pool (~7% large, bit-exact, shipped); keep the parallel-build per-thread accumulators (load-bearing) and the KB-scale bool matrix (sub-noise); per-leaf rows are already flat |

## Source Files

Original spike READMEs + `#[ignore]`d microbenches are preserved in `sources/` (010–013).
</findings_index>

<metadata>
## Processed Spikes

- 010-histogram-pool-arena (VALIDATED + SHIPPED)
- 011-parallel-build-scatter (INVALIDATED — load-bearing)
- 012-reuse-pool-across-trees (VALIDATED + SHIPPED)
- 013-feature-splittable-arena (INVALIDATED — sub-noise)

Not yet processed: 001–009 (earlier GPU/columnar/parallel perf spikes).
</metadata>
