---
phase: quick-260625-hw2
plan: 01
subsystem: lgbm-treelearner / lgbm-compute
status: complete
tags: [partition, spike-027, fused-gather, cpu-path, bit-exact, perf]
requires:
  - "spike-027 (fused-gather-partition) VALIDATED + bit-exact"
provides:
  - "Backend::prefers_host_partition() discriminator (default false; CpuBackend true)"
  - "DataPartition::split fused u8-route host path on CpuBackend (byte-identical [left|right])"
affects:
  - "DataPartition::split numeric branch (CPU routing only; device path verbatim)"
tech-stack:
  added: []
  patterns:
    - "trait-method discriminator (default false, backend overrides true) — same pattern as wants_resident_bins / resident_pool_supported"
    - "fused gather→route(u8 scratch)→scatter, in place on indices slice"
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/data_partition.rs
decisions:
  - "Gate the fused path on a backend discriminator (prefers_host_partition), NOT a global env/flag — keeps device routing intact and avoids conflating with resident_pool_supported"
  - "Keep the V1 u8 route scratch (one gather); did NOT use the V2 2-gather variant which regresses at U32/4M"
  - "Per-row V5 bounds validation runs in ascending leaf-position order off the narrow BinColumn (faithful per-leaf analog of data_partition_cpu_native's lowest-global-index check)"
metrics:
  duration: "~25 min"
  completed: "2026-06-25"
  tasks: 3
  files: 2
---

# Phase quick-260625-hw2 Plan 01: Wire spike-027 V1 fused-gather partition Summary

Wired spike-027 V1 (fused-gather, u8 route scratch) into the production
`DataPartition::split` numeric branch on the CPU path only — one gather + a
¼-width u8 route scratch + one u32 scatter, replacing the materialize-then-op
path (leaf_rows clone + u32-widened leaf_feature_bins + backend.data_partition +
local→row remap). Byte-identical `[left|right]` order; the RocmBackend numeric
branch keeps routing on-device verbatim.

## What was built

- **Task 1** (`af95c96`): Added `Backend::prefers_host_partition(&self) -> bool`
  trait method, default `false`, placed alongside `wants_resident_bins`. CpuBackend
  overrides to `true`; RocmBackend inherits the default `false` (no override) so its
  on-device `data_partition` stays the routing path.
- **Task 2** (`5177abd`): Gated `DataPartition::split`'s numeric branch on
  `backend.prefers_host_partition()`. When true, a new private `split_fused_host`
  runs the spike-027 V1 fused path IN PLACE on `self.indices[begin..begin+count]`:
  V5 bounds validation first (num_bin>0, threshold<num_bin, every leaf-row bin
  <num_bin in ascending leaf position, matching the error variants/fields the
  device path returns), then the `make_router` decision (th = threshold+min_bin,
  th-=1 when most_freq_bin==0, default_to_right = most_freq_bin>threshold,
  out-of-[min,max] → default), then pass-1 gather+route+count into a u8 scratch and
  pass-2 direct row-id scatter. When false (RocmBackend / device), the original
  materialize-then-op path is kept verbatim. Added `split_fused_equals_serial`
  byte-identity test covering the most_freq_bin==0 branch and a U8 column over a
  scattered (randomly-gathered) leaf, asserting byte-identical [left|right] slice
  and matching (left_count,right_count) vs a `data_partition_cpu_native` reference.
- **Task 3**: Ran the bit-exact merge gate. No code changes needed.

## Bit-exact merge gate results

| Command | Result |
|---------|--------|
| `cargo test -p lgbm-treelearner --lib` | **77 passed, 0 failed**, 2 ignored (incl. new `split_fused_equals_serial` + 2 pre-existing split_* with fused path active) |
| `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` | **`raw_bin_train_matches_cpp_golden ... ok`** (golden parity holds — partition order feeds the histogram-subtraction trick) |
| `cargo test -p lgbm` | **41 passed, 0 failed** |
| `cargo check -p lgbm-compute --features rocm` | **Finished cleanly** — cfg(feature=rocm) path compile-valid (RocmBackend inherits default false) |

## Deviations from Plan

None — plan executed exactly as written. No code changes were required in Task 3
(all gates passed against the Task 1/2 implementation).

## Pre-existing (out-of-scope) test failures

`cargo test -p oracle-harness` (full crate, beyond the gate's target) has 2
**environment-only** failures unrelated to this change:
`config_drift::rust_alias_table_matches_cpp_alias_table_verbatim` and
`config_drift::rust_tables_cover_in_scope_cpp_params_and_aliases`. Both fail
reading `LightGBM/src/io/config_auto.cpp` — the untracked C++ reference tree that
is absent in git worktrees by design (project memory: "never git-add LightGBM/;
worktrees break for phases needing it"). These are not a parity regression: they
read a missing reference file, do not touch partition/training, and would pass in
the main checkout where `LightGBM/` is present. The parity-critical gate
(`raw_bin_train_matches_cpp_golden`) passes.

## Self-Check: PASSED

---

## Orchestrator integration note (post-execution)

The executor ran in a worktree branched from a **stale base** (`fc812ec`, predating the
phase-12 `scan_resident_siblings` lib.rs addition + the rayon-revert NOTE in
data_partition.rs). The two code commits were **cherry-picked onto master** (`f413e1d`,
`8eb6c9e`) — both auto-merged clean, and integration coherence was verified
(`scan_resident_siblings` + the rayon NOTE both preserved alongside the new
`prefers_host_partition` + fused path).

The bit-exact merge gate was **re-run on the real master tree** (not the stale worktree):
- `cargo test -p lgbm-treelearner --lib` — **77 passed, 0 failed**
- `cargo test -p oracle-harness` (FULL suite) — **all green**: raw_bin_train golden 2,
  boosting_parity 75, learner_parity 29, kernel_parity 7, config_drift 3, +others, 0 failed
- `cargo test -p lgbm` — **41 passed, 0 failed**

The earlier env-only config_drift failures the executor saw were a worktree artifact (the
untracked `LightGBM/` C++ ref tree is absent in worktrees); on the main checkout they pass.
