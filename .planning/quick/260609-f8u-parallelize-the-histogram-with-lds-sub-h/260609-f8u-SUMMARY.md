---
quick_id: 260609-f8u
title: Parallelize the histogram with LDS sub-histograms (eo5 Finding #2)
date: 2026-06-09
status: complete
type: implementation + benchmark
commit: 35c41f6
parity_class: neutral (production path unchanged; new kernel is f32 ~1e-6)
---

# Quick Task 260609-f8u — Summary

**LANDED (commit 35c41f6).** Implemented eo5 Finding #2: an LDS-privatized
sub-histogram GPU kernel. Correct + benchmarked + landed as an available primitive;
NOT wired into the production path (honest reasons below).

## Delivered

- `construct_hist_kernel_lds_f32` + `construct_histograms_lds_f32_on`
  (`crates/lgbm-compute/src/kernels/histogram.rs`) — per-cube LDS sub-histogram
  (shared-memory atomics) → single global merge. **Global atomic traffic `2*n` →
  `CUBE_COUNT*2*num_bin`.** First `SharedMemory`/`sync_cube` use in the codebase.
  Fixed 256-bin/2 KiB LDS cap (sidesteps cubecl comptime-size); >256 bins → naive.
- 4 new gfx1100 tests (correctness under contention, exact-vs-naive, <1e-5 vs f64
  anchor, A/B benchmark).

## Benchmark (gfx1100, LDS vs naive global-atomic)

| n | bins | speedup |
|---|---|---|
| 1M | 16 | **4.08×** |
| 1M | 256 | 1.16× |
| 5M | 16 | **4.63×** |
| 5M | 256 | 1.18× |

~4–4.6× under high contention (few bins), ~1.2× at 256 bins, **never slower**. Exactly
the LightGBM histogram-family rationale.

## Why NOT wired (honest)

`RocmBackend::construct_histograms` stays on the naive path. The gating test
`learner_parity_resident_equals_host_tree_on_hip` is **PRE-EXISTING FLAKY** — fails
~4/6 runs on the unchanged naive code (verified by reverting to master-equivalent):
the naive atomic's nondeterministic f32 order puts leaf 11 on the 1e-6 knife-edge vs
the resident chain (DEF-f8u-01). Non-regression can't be cleanly verified against a
~50%-flaky baseline, and wiring needs the resident/batched BUILD path LDS-ified too
(one accumulation order) — the larger follow-up. Ships as a primitive (t3t precedent).
**Production behavior unchanged.**

## Gate (GREEN)

Default merge gate 0-failed (lgbm 41 / python 55 / compute 18 / treelearner 65 /
boosting 75 / learner_parity 29 / kernel_parity 6); hip kernel_parity 15/15;
rocm_parallel_histogram 7/7; clippy clean.

## Follow-ups

- DEF-f8u-01: flaky `learner_parity_resident_equals_host_tree_on_hip` (deferred-items).
- LDS-ify the batched/resident BUILD hot path + unify accumulation order, then wire
  live (where the 4× reaches training).

## Files modified

- `crates/lgbm-compute/src/kernels/histogram.rs` (LDS kernel + launcher)
- `crates/lgbm-compute/src/lib.rs` (explanatory note; construct_histograms unchanged)
- `crates/lgbm-compute/tests/rocm_parallel_histogram.rs` (4 new tests)
