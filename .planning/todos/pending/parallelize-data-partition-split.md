---
title: Parallelize DataPartition::split (rayon over rows) — the tall-shape CPU lever
date: 2026-06-22
priority: medium
type: lever
run_with: /gsd-quick or /gsd-plan-phase
---

# Parallelize DataPartition::split

## Why

Root-caused in [[wide-tall-two-backend-root-cause]]: at tall-narrow shapes (1M×50) the
CPU is ~1.7× C++ per-iteration, and **29% of train is `DataPartition::split`**
(data_partition.rs:108–148), which is **single-threaded** — a plain `for` over the
leaf's rows. C++ parallelizes the same op:

```cpp
// LightGBM/src/treelearner/data_partition.hpp:55
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
```

With only 50 features the rayon-over-features histogram build can't hide the serial
partition (50 features ÷ 16 cores). At wide shapes partition is ~5% (dwarfed by the
500-feature build), so this lever is **tall-shape-specific** — biggest win at
many-rows-few-features.

## Scope

- Parallelize the row-reorder in `DataPartition::split` with rayon, mirroring C++'s
  `schedule(static, 512) if (num_data >= 1024)` — gate on a leaf-row threshold (like the
  histogram build's `LGBM_PAR_THRESHOLD=16384`) so small leaves stay serial (avoid the
  fork/join regression spike-005 saw on tiny leaves).
- **Bit-exact requirement:** the reordered local-position array must be IDENTICAL to the
  serial output (the partition order is load-bearing for the subtraction trick + row→leaf
  map). Use a deterministic static partition + prefix-sum (the C++ approach), NOT an
  order-nondeterministic scatter. Add a `split_parallel_equals_serial` test, like
  `build_histograms_parallel_equals_serial`.
- Gate: `cargo test -p lgbm-treelearner --lib` + `cargo test -p oracle-harness`
  (raw_bin_train_parity vs lib_lightgbm).

## Done-when

Measured train-wall improvement at tall-narrow (1M×50, where partition is 29%) with the
bit-exact gate green. Expected to close a chunk of the ~1.7× tall-narrow gap; ~0 at wide.

## Reproduce the baseline

```
LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=50 \
  cargo run --release --example bench_gpu_vs_cpu   # partition=29.2% of train
```
