---
phase: 260625-j1l
plan: 01
subsystem: gpu-partition
status: complete
tags: [gpu, rocm, partition, narrow-upload, spike-029, bit-exact, additive]
requires:
  - spike-029 (gpu-narrow-upload-fuse, VALIDATED)
  - BinColumn (u8/u16/u32 narrow storage)
provides:
  - Backend::data_partition_native (additive, widening default + RocmBackend override)
  - data_partition_kernel<B: Int> (native-width route kernel)
  - data_partition_native_on (native-width device entry)
affects:
  - DataPartition::split device branch (re-gathers narrow BinColumn)
tech-stack:
  added: []
  patterns:
    - "generic-over-<B: Int> kernel + per-width launch monomorph (qix precedent)"
    - "additive Backend trait method with widening default body (spike-027 precedent)"
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/partition.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/data_partition.rs
    - crates/oracle-harness/tests/kernel_parity.rs
decisions:
  - "ADDITIVE: data_partition(&[u32]) signature + all callers byte-unchanged; new sibling data_partition_native(&BinColumn) only"
  - "Bit-EXACT by construction: u8/u16/u32 kernels read the same bin via u32::cast_from -> value-identical route -> byte-identical (reordered, split_point)"
  - "CpuBackend gets the widening default with ZERO edits (byte-unchanged)"
metrics:
  duration_min: 7.4
  completed: 2026-06-25
  tasks: 3
  files: 4
---

# Phase 260625-j1l Plan 01: Wire spike-029 GPU narrow-upload partition Summary

Wired spike-029 (VALIDATED ~1.2-1.7×, bit-exact) into the ROCm partition path: the
per-split device op now uploads a leaf's bins at NATIVE width (u8/u16/u32) instead of
u32-widening, cutting host→device upload + kernel-read ~4× on the common all-u8
(`max_bin≤255`) case. The change is ADDITIVE and BIT-EXACT — verified on the actual GPU.

## What was built

- **Task 1 (`7e68bd2`)** — Made `data_partition_kernel` generic `<B: Int>` reading the bin
  via `u32::cast_from` (qix histogram precedent). `data_partition_on` now launches the
  explicit `::<u32, R>` monomorph (byte-for-byte the prior kernel). Added
  `data_partition_native_on(&BinColumn, …)` that uploads `count × native-width` bytes and
  dispatches `::<u8|u16|u32, R>`, with the V5 boundary checks reading each bin via
  `BinColumn::bin` before the unsafe launch and a shared `gather_route` tail.
- **Task 2 (`507310d`)** — Added the ADDITIVE `Backend::data_partition_native(&BinColumn)`
  with a widening DEFAULT body (widen + delegate to `data_partition`), so CpuBackend and
  every non-overriding backend are byte-unchanged with zero edits. `RocmBackend` overrides
  it to call `data_partition_native_on` (native-width upload). `DataPartition::split`'s
  device branch now re-gathers a narrow `BinColumn` (`feature_bins.gather`) and calls
  `data_partition_native`; the local→global writeback / leaf bookkeeping / counts are
  unchanged.
- **Task 3 (`e65f389`)** — Extended both partition parity gates to route the SAME golden
  `partition.txt` cases through `data_partition_native` over a narrow `BinColumn` (golden
  num_bins are 8/10/16 ⇒ U8 arm), plus an explicit U16-band self-check (byte-identical to
  the u32 reference). CPU asserts vs the golden via `compare_exact_u32`; HIP asserts
  BIT-EXACT (no tolerance, f64-free).

## Bit-exact merge gate results

CPU:
- `cargo test -p lgbm-treelearner --lib` — **76 passed, 0 failed** (2 ignored).
- `cargo test -p oracle-harness` named gates:
  - `kernel_parity_partition_exact_on_cpu` — **1 passed** (U8 golden + U16 self-check).
  - `raw_bin_train_matches_cpp_golden` — **2 passed** (raw_bin_train_parity target).
  - full oracle-harness suite otherwise green (75 + 5 + 5 + 3 passed) EXCEPT two
    pre-existing `config_drift` failures (see Deferred Issues).
- `cargo test -p lgbm` — **41 passed, 0 failed**.
- `cargo test -p lgbm-compute --lib kernels::partition` — **8 passed** (incl. 5 new native
  tests).

ROCm ON HARDWARE (GPU + HIP present this session):
- `cargo build --features rocm` — **compiles** (workspace).
- `cargo test -p oracle-harness --features rocm --test kernel_parity kernel_parity_partition_exact_on_hip`
  — **1 passed, 0 failed** on the GPU. The new U8 (golden cases) AND U16 (self-check)
  narrow cells RAN on the GPU and were BIT-EXACT (no tolerance) to the golden /
  u32-reference. Output:
  `running 1 test / test hip::kernel_parity_partition_exact_on_hip ... ok / test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 14 filtered out`.

## Deviations from Plan

None — plan executed as written. Line numbers in the plan referenced a fuller HEAD; this
worktree is based on an older commit (the RocmBackend `build_leaf_histograms_raw` lacks the
later `width` param and there is no `split_fused_host`/`prefers_host_partition` yet), but the
ADDITIVE design applied cleanly: `data_partition_on`, the trait, and the device `split` branch
all matched the plan's anchors. The device `split` IS the only partition path in this branch,
and it remains behavior-unchanged apart from the native-width re-gather.

## Deferred Issues (out of scope — NOT a regression)

- `oracle-harness` `config_drift.rs` tests `rust_tables_cover_in_scope_cpp_params_and_aliases`
  and `rust_alias_table_matches_cpp_alias_table_verbatim` fail with
  `reading .../LightGBM/src/io/config_auto.cpp: No such file or directory`. These parse the
  read-only C++ reference tree `LightGBM/`, deliberately never git-added and so absent in this
  worktree (MEMORY: lightgbm-ref-tree-untracked). My commits did not touch `config_drift.rs`.
  Logged in `deferred-items.md`.

## Self-Check: PASSED

- Files exist: partition.rs, lib.rs, data_partition.rs, kernel_parity.rs (all modified, in
  commits 7e68bd2 / 507310d / e65f389).
- Commits exist: `7e68bd2`, `507310d`, `e65f389` (verified in git log).

---

## Orchestrator integration note (post-execution)

Executor ran in a worktree off a stale base (`fc812ec`, predating BOTH the qix `width` param
and the spike-027 `prefers_host_partition` wiring). The 3 commits were cherry-picked onto master:
`partition.rs` (generic kernel) + `kernel_parity.rs` auto-merged clean; `data_partition.rs`
conflicted and was resolved BY HAND — the narrow `feature_bins.gather()` + `data_partition_native`
landed in master's **device branch** (`else` of `prefers_host_partition`), leaving the
CpuBackend `split_fused_host` (spike-027) path untouched. Master commits: `4fe9025`, `3b79e69`,
`9ab8cb6`.

Bit-exact gate RE-RUN on the integrated master tree:
- CPU: lgbm-treelearner --lib **77/0**, lgbm-compute partition **8/0**, oracle golden
  (raw_bin_train_matches_cpp_golden) **1/0**, kernel_parity_partition_exact_on_cpu **ok** (new
  U8/U16 cell), lgbm **41/0**.
- **ROCm ON GPU: `hip::kernel_parity_partition_exact_on_hip ... ok` (1/0)** — U8/U16 narrow path
  through `data_partition_native` bit-exact on hardware.

config_drift failures the executor noted are the env-only worktree artifact (untracked
`LightGBM/` ref tree); they pass on the main checkout.
