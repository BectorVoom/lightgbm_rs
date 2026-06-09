---
quick_id: 260609-f3g
type: benchmark
title: Benchmark the GPU partition change (260609-eu9) on large data
date: 2026-06-09
status: complete
hardware: gfx1100 (real ROCm), cubecl-hip, --release
verdict: 21–60× kernel speedup at 100k–5M rows; single-unit was a serial one-lane O(n) walk (~14 Mrows/s)
---

# Benchmark: GPU partition kernel — parallel (eu9) vs single-unit

## Method

Throwaway A/B microbenchmark (instrumentation reverted after; no code shipped).
Drove `data_partition_on` directly on `rocm_client()` (gfx1100) — the exact GPU path
`RocmBackend::data_partition` wraps — at n = 100k / 1M / 5M rows, num_bin=128,
deterministic bins spread across the range (both children populated). 3 warmup + 15
timed calls per size, `--release`. Temporary timing buckets inside `data_partition_on`
split each call into **upload** (host→dev) / **GPU** (launch+sync+readback) / **gather**
(host two-pass compaction). The GPU bucket isolates the kernel; upload+gather are
identical O(n) host work in both versions. A/B done by swapping only the kernel body +
launch config (parallel `ABSOLUTE_POS`, `ceil(n/256)×256`  ↔  single-unit `UNIT_POS==0`
loop, `CubeDim 1`) between runs.

## Results (steady-state medians)

| n (rows) | single-unit GPU bucket | parallel GPU bucket | **kernel speedup** | single-unit full call | parallel full call | full speedup |
|---|---|---|---|---|---|---|
| 100,000   | ~6.3 ms  | ~0.30 ms | **~21×** | 6.85 ms   | 0.60 ms  | 11.5× |
| 1,000,000 | ~68 ms   | ~1.6 ms  | **~42×** | 73.76 ms  | 4.48 ms  | 16.5× |
| 5,000,000 | ~333 ms  | ~5.5 ms  | **~60×** | 349.65 ms | 15.51 ms | 22.5× |

**Throughput (full `data_partition_on`):**
- single-unit: ~14.6 / 13.6 / 14.3 Mrows/s — **flat ~14 Mrows/s regardless of n**
- parallel:    167.6 / 223.3 / 322.3 Mrows/s — **rises with n** as fixed launch overhead amortizes

## Interpretation

- The single-unit kernel time scales **linearly with n at a fixed ~14 Mrows/s** — the
  textbook signature of a **serial one-lane walk** (`if UNIT_POS==0 { for i in 0..n }`).
  This is direct, quantified evidence for why the as-built GPU train was so slow
  ([[perf-gap-vs-cpp-40-80x]], kfu "~214× slower — single-unit kernels").
- The parallel kernel is **21–60× faster in isolation**, and the speedup **grows with
  data size** (more rows = more lanes utilized vs the fixed serial cost).
- The **full-call** speedup is smaller (11–22×) because the identical host upload +
  two-pass gather (O(n) CPU work, unchanged by this kernel change) dilute the kernel
  win. At 5M rows, upload≈5 ms + gather≈3.5 ms are now the floor of the parallel path
  (kernel ≈5.5 ms), so further partition speedup would need attacking the host
  gather/upload, not the kernel.

## Honest scope caveats

- This measures the partition kernel **in isolation at large n**. In real training,
  partition runs per split over **leaf subsets** (≤ num_data per level), and the
  production bench corpus is only **20k rows** where the GPU is **launch-bound**
  ([[l3-on-gpu-fixhistogram-deferred]]) — so end-to-end training impact at the current
  bench sizes is modest. The kernel win is real and large, and it **scales with data
  size**, so it matters for large-data GPU training specifically.
- CPU production path (`data_partition_cpu_native`) is unaffected — it never used the
  kernel.

## Reproduce

`LGBM_PART_BENCH=1 cargo test -p lgbm-compute --features rocm --release` with a small
rocm-gated test looping `data_partition_on` over large n + temporary timing buckets in
`data_partition_on` (both reverted in this task — re-add to re-measure). A/B by swapping
the kernel body/launch config to single-unit.
