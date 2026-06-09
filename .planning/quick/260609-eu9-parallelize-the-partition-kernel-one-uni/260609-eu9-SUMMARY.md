---
quick_id: 260609-eu9
title: Parallelize the partition kernel (one-unit-per-row)
date: 2026-06-09
status: complete
type: implementation
parity_class: neutral
commit: b141a82
---

# Quick Task 260609-eu9 — Summary

**LANDED (commit b141a82).** Implemented Finding #1 from investigation 260609-eo5.

## What changed

`crates/lgbm-compute/src/kernels/partition.rs` — `data_partition_kernel` and its
launcher `data_partition_on`:

- **Kernel:** was `if UNIT_POS == 0 { for i in 0..bins.len() { route[i] = ... } }`
  (one lane walks every row). Now indexes by `ABSOLUTE_POS` with an `idx < bins.len()`
  bounds guard — **one unit per row**, each writing its own `route[idx]` (disjoint,
  no atomics).
- **Launcher:** was `CubeCount::Static(1,1,1) / CubeDim::new_1d(1)`. Now
  `CubeCount::Static(ceil(n/256),1,1) / CubeDim::new_1d(256)` (mirrors the parallel
  histogram launcher).
- Host stable two-pass gather: **unchanged**.
- Stale "single-unit" doc references updated.

## Why it is parity-neutral

`route[i] = SplitInner(bins[i])` is per-row independent (no cross-row carry) and
integer-only, so evaluation order is irrelevant — there is no float fold to keep
ordered (contrast the histogram f64 anchor). The order-dependent stable compaction
lives on the host and did not change.

## Design decision: unified kernel (not a rocm-gated fork)

Chose ONE parallel kernel for all runtimes rather than forking a rocm-gated parallel
variant (the pattern the atomic histogram uses). Rationale: partition needs **no
atomics** (disjoint writes), so the rocm atomic-codegen gating rationale does not
apply, and cubecl-cpu was empirically verified to execute the multi-cube
`ABSOLUTE_POS` form correctly. Matches the documented unified-CPU/GPU-kernel
preference. CpuBackend's production path (`data_partition_cpu_native`) does not use
the kernel and is untouched.

## Gate (GREEN)

- cubecl-cpu partition unit tests: **3/3** (incl. exact-reorder `partition_basic_threshold`)
- Default merge gate: lgbm-compute **18**, lgbm-treelearner **65**, learner_parity
  **29**, boosting_parity **75**, kernel_parity **6** — **0 failed**
- **hip kernel_parity: 15/15** incl. `kernel_parity_partition_exact_on_hip` —
  **BIT-EXACT** routing on the real gfx1100 (assert_eq, no tolerance)
- clippy clean on partition.rs

## Honest perf note

No wall-clock benchmark claim. This removes a fully-serial single-lane GPU traversal
of every row per split; the win is on the GPU (`RocmBackend`) path and materializes
on large/compute-bound data (the GPU has repeatedly profiled launch-bound on the
small synthetic benches — see 260609-eo5 / [[l3-on-gpu-fixhistogram-deferred]]). The
CPU native fast path is unaffected. Deliverable is the correctness-preserving
parallelization, not a measured speedup.

## Pre-existing issue found (OUT OF SCOPE, not a regression)

`crates/lgbm-compute/tests/rocm_backend_parity.rs` fails to compile under
`--features rocm` (4× `error[E0423]: expected value, found struct RocmBackend` at
lines 29/50/100/115 — `let gpu = RocmBackend;`). `RocmBackend` gained `RefCell`
device-state fields (resident_bins/resident_pool) in tasks nn7/p90, so it is no
longer a unit struct and this old test bitrotted. **Verified pre-existing:** stashing
this task's partition.rs edit reproduces the identical 4 errors on clean HEAD. The
partition coverage it intended (`rocm_backend_data_partition_matches`) is already
provided bit-exact by `kernel_parity_partition_exact_on_hip`. Logged to
`260609-eu9-deferred-items.md`.

## Files modified

- `crates/lgbm-compute/src/kernels/partition.rs` (commit b141a82)
