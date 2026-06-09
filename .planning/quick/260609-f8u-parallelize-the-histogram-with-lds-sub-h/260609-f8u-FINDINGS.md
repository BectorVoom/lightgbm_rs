---
quick_id: 260609-f8u
type: implementation + benchmark
title: LDS-privatized sub-histogram GPU kernel (eo5 Finding #2)
date: 2026-06-09
status: complete
hardware: gfx1100 (real ROCm), cubecl-hip 0.10.0
verdict: kernel DONE + proven + 4–4.6× faster under contention; landed as available primitive, NOT wired (flaky gating test + needs resident-path unification)
---

# LDS-Privatized Sub-Histogram Kernel (eo5 Finding #2)

## What was built (commit 35c41f6)

`construct_hist_kernel_lds_f32` + launcher `construct_histograms_lds_f32_on` in
`crates/lgbm-compute/src/kernels/histogram.rs` — the first `SharedMemory` / `sync_cube`
use in the codebase.

**Algorithm (mirrors LightGBM's OpenCL `histogram{16,64,256}.cl`):** each CUBE owns a
private sub-histogram in shared memory (LDS). All units in the cube atomic-add their
strided rows into the LDS copy (intra-workgroup contention only — far cheaper than
global), `sync_cube()`, then the cube merges its sub-histogram into the global output
with ONE global atomic per cell. **Global atomic traffic: `2*n` → `CUBE_COUNT*2*num_bin`.**

**cubecl constraint handled:** `SharedMemory::new` needs a COMPTIME size but num_bin is
runtime (32/64/128/256). Solution: allocate a fixed 256-bin (2 KiB) LDS max once
(≪ the gfx1100 64 KiB budget) and drive the active length with a runtime `lds_len`;
`num_bin > 256` falls back to the naive path. One kernel binary, no per-bin-count
specialization.

## Correctness — PROVEN on gfx1100 (4 new tests, all green)

- `lds_no_lost_updates_under_contention` — 50k rows → 4 bins, exact-integer grad/hess:
  result EXACTLY equals the known per-bin sums (no LDS or global atomic update lost).
- `lds_equals_naive_atomic_on_integer_data` — 30k rows, 128 bins: EXACTLY equals the
  naive global-atomic result.
- `lds_within_tolerance_of_cpu_f64_anchor` — 8k rows, 64 bins, real f32 data:
  `max_rel < 1e-5` vs the cpu f64 anchor (same gate the naive path is held to).
- f32 nondeterministic accumulation ⇒ same ~1e-6 ROCm contract; cpu f64 anchor untouched.

## Benchmark — LDS vs naive global-atomic (gfx1100, --release, 50 reps)

| n | bins | naive | LDS | **speedup** |
|---|---|---|---|---|
| 1,000,000 | 16 (high contention)  | 15193.8 µs | 3728.3 µs | **4.08×** |
| 1,000,000 | 256 (low contention)  |  4814.0 µs | 4162.9 µs | 1.16× |
| 5,000,000 | 16 | 74435.2 µs | 16071.8 µs | **4.63×** |
| 5,000,000 | 256 | 22206.0 µs | 18823.9 µs | 1.18× |

**Textbook result:** LDS wins big (~4–4.6×) when bins are few (high per-bin global
contention — exactly what LDS privatization relieves), and modestly (~1.2×) at 256
bins (already low contention; privatize+barrier+merge overhead nearly cancels). LDS is
**never slower** — a strict improvement. This is precisely why LightGBM ships a
bin-count kernel family.

## Why it is NOT wired into the production path (the honest call)

`RocmBackend::construct_histograms` still uses the naive `construct_histograms_parallel_f32_on`.
Two reasons:

1. **The would-be gating test is PRE-EXISTING FLAKY (DEF-f8u-01).**
   `learner_parity_resident_equals_host_tree_on_hip` fails **~4 of 6 runs on the
   unchanged naive code** (verified: reverted `construct_histograms` to naive,
   master-equivalent, and it still flaked 4/6). Cause: the naive atomic's
   nondeterministic f32 accumulation order puts **leaf 11's output on the 1e-6
   knife-edge** vs the resident chain (`abs_diff` hovers ~0.9e-6…1.1e-6 run-to-run).
   This is a real pre-existing defect, NOT introduced by this task.
2. **Non-regression can't be cleanly verified against a ~50%-flaky baseline,** and
   routing to LDS would give this path yet another f32 accumulation order. Wiring it
   live properly wants the resident/batched BUILD path (`build_fix_compact_resident` /
   `build_leaf_histograms_raw`) LDS-ified too, so both GPU paths share ONE accumulation
   order — the larger Finding #2 follow-up.

So the kernel ships as an **available, tested, benchmarked primitive** (the t3t
fused-kernel precedent: landed + proven + not wired). Production behavior is
**unchanged** (construct_histograms = naive, identical to master).

## Scope note

This LDS-ifies the SINGLE-FEATURE `construct_histograms` primitive. The training hot
path is the batched/resident kernels (`build_leaf_histograms_raw` → batched/resident
f32-atomic), which are NOT yet LDS-privatized — a bigger task (concatenated
multi-feature layout, per-feature LDS budgeting). That, plus unifying the resident
build's accumulation order, is the remaining Finding #2 work needed to make the LDS
win live in GPU training.

## Gate

- Default merge gate GREEN (0 failed): lgbm 41, python 55, compute 18, treelearner 65,
  boosting 75, learner_parity 29, kernel_parity 6.
- hip kernel_parity 15/15; rocm_parallel_histogram 7/7 (3 existing + 4 new LDS).
- clippy clean on the new code.
- Production `construct_histograms` path byte-unchanged (naive atomic).

## Follow-ups

- **DEF-f8u-01** (pre-existing): `learner_parity_resident_equals_host_tree_on_hip`
  flaky (~4/6) — the 1e-6 resident-vs-host tolerance is too tight for two
  nondeterministic f32-atomic paths at leaf 11's knife-edge. See deferred-items.
- **Finding #2 follow-up:** LDS-ify the batched/resident build path + unify the
  accumulation order, then wire LDS live (where the 4× actually reaches training).
