---
title: Reconcile which histogram kernel production training vs bench_gpu_vs_cpu actually invokes
date: 2026-06-21
priority: high
type: todo
context: /gsd-explore "investigate gpu kernel has bottleneck(speed) of learning large data"
---

# Todo: reconcile prod vs bench GPU histogram kernel

The "GPU is ~5.4× slower than the CPU anchor on large data" finding may be measuring the
**wrong kernel**. Diff vs the AMD HIP fork showed the bench's live GPU column calls the **naive
global-atomic** kernel, not the LDS-privatized one:

- Live bench GPU path → `construct_histograms_parallel_f32_on`
  (`crates/lgbm-compute/src/lib.rs:1955`), which issues `2·n` **global** `Atomic<f32>::fetch_add`
  on hot bins (`crates/lgbm-compute/src/kernels/histogram.rs:401-402`).
- An LDS sub-histogram kernel mirroring the fork already exists: `construct_hist_kernel_lds_f32`
  / `construct_histograms_lds_f32_on` (`histogram.rs:760-804`) — **landed but apparently not
  wired into the bench column**.

**This contradicts memory [[cubecl-cpu-runs-parallel-kernels]] (fw1, b878eb5)**, which records the
LDS BUILD hot path as **wired live, 3.5–9×**. Both can be true: production training may already
use LDS while `bench_gpu_vs_cpu.rs` still calls the old global-atomic kernel.

**Task:** trace the real call graph for (a) production GPU training `Train`/`ConstructHistograms`
and (b) the `bench_gpu_vs_cpu.rs` GPU column. Confirm/deny: does production already use LDS? Does
the bench measure a stale kernel? Outcome decides whether there is a real gap to fix or just a
bench artifact to correct.

**Do this FIRST** — it gates [[verify-lds-atomic-lowering-gfx1100]] and any wire/A-B spike.

Reference: explorer diff of `LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_histogram_constructor.{cu,cpp}`
vs `crates/lgbm-compute/src/kernels/histogram.rs` + `lib.rs`.
