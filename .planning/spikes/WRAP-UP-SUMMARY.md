# Spike Wrap-Up Summary

**Date:** 2026-06-17
**Spikes processed:** 13 (001–013, full campaign — 010–013 wrapped first, 001–009 appended)
**Feature areas:** CPU histogram build · GPU histogram kernel · GPU routing & quantization · Histogram/learning memory layout
**Skill output:** `./.claude/skills/spike-findings-lightgbm_rs/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | gpu-cpu-crossover | standard | ✅ VALIDATED | GPU routing & quantization |
| 002 | lowrow-phase-ab | standard | ✅ VALIDATED | CPU histogram build |
| 003 | columnar-hist-build | standard | ✅ VALIDATED + SHIPPED | CPU histogram build |
| 004 | columnar-u8-bins | standard | ✅ VALIDATED + SHIPPED | CPU histogram build |
| 005 | feature-parallel-build | standard | ✅ VALIDATED + SHIPPED | CPU histogram build |
| 006 | gpu-u8-bins | standard | ❌ INVALIDATED | GPU histogram kernel |
| 007 | row-partitioned-histogram-build | standard | ✅ VALIDATED | GPU histogram kernel |
| 008 | 16bit-discretized-hist | standard | ❌ INVALIDATED (exact) | GPU routing & quantization |
| 009 | multifeature-per-cube | standard | ❌ INVALIDATED | GPU histogram kernel |
| 010 | histogram-pool-arena | standard | ✅ VALIDATED + SHIPPED | Histogram memory layout |
| 011 | parallel-build-scatter | standard | ❌ INVALIDATED (load-bearing) | Histogram memory layout |
| 012 | reuse-pool-across-trees | standard | ✅ VALIDATED + SHIPPED | Histogram memory layout |
| 013 | feature-splittable-arena | standard | ❌ INVALIDATED (sub-noise) | Learning-path allocation |

## Key Findings (001–009, the perf campaign)

- **The histogram BUILD is the bottleneck** (002): 63–90% of CPU train, 5.2× slower than
  C++ at low rows; split-scan/partition are near parity. Localized via per-phase A/B vs
  `lib_lightgbm` 4.6 `-DUSE_TIMETAG`.
- **Four stacked bit-exact CPU wins shipped:** once-per-leaf gather (003, −33/−39% build),
  fused-branchless build (003b, needs validation relocated upstream), narrow u8/u16 bins
  (004, large train −49%), feature-parallel ≥16384-row leaves (005, large −26%). Cumulative
  large ≈ −67%.
- **GPU build is atomic/latency-bound, not bandwidth-bound** (006): the CPU u8 win does NOT
  transfer (~0%). The one GPU lever is row-partitioning to ~8 wkgrps/CU (007, ~1.35×);
  multi-feature packing is null at matched occupancy (009). With the CPU multi-threaded
  (005), GPU loses at every tested size → GPU work is ROCm-parity maintenance, not speed.
- **GPU wins on wall-clock ≳1M rows vs single-thread** (001, crossover ≈700k) — but moves to
  millions vs the multi-threaded anchor. **int16 quantized hist is irreducibly approximate**
  (008, ~3e-4 floor ≫ gate) → opt-in mode only, never the exact path.

## Key Findings (010–013, the Vec<Vec> thread)

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
