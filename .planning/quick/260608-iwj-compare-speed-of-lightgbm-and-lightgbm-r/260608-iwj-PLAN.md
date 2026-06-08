---
quick_id: 260608-iwj
slug: compare-speed-of-lightgbm-and-lightgbm-r
date: 2026-06-08
mode: quick (discuss-equivalent; parity-gated)
---

# Quick Task 260608-iwj — Compare speed of lightgbm vs lightgbm_rs and optimise

Measure `lightgbm_rs` train/predict speed, then apply parity-safe optimizations
from the optimisor manual. The oracle-harness parity suite is the merge gate after
every code change — any bit-exact regression reverts that change.

## Tasks (atomic commits)

- **T1 — Benchmark harness + M0 baseline.** Add `crates/lgbm/examples/bench_train.rs`:
  deterministic synthetic identity-binned `DenseCorpus` at 3 sizes, median-of-N
  timing of `train` + `predict`, mimalloc gated behind an `lgbm` feature. Add the
  `mimalloc` optional dep + feature. Run M0 (system alloc, default release profile),
  record into REPORT.md. Commit.

- **T2 — Release profile (lever 1).** Add `[profile.release]` to workspace
  Cargo.toml: `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`. Run M1, record
  delta. Commit.

- **T3 — Global allocator (lever 2).** Wire `#[global_allocator]` = mimalloc into
  the bench (feature-gated) and the `lgbm-python` cdylib. Run M2
  (`--features mimalloc`), record delta. Commit.

- **T4 — smallvec + buffer-reuse (levers 3 & 4).** SmallVec for bounded small
  per-leaf/per-split buffers in `lgbm-treelearner`; reuse scratch/zeroed buffers in
  the gather + compute kernels to cut redundant allocation. Run M3, record delta.
  **Gate:** `cargo test -p oracle-harness` + `cargo test --workspace` stay green
  (except pre-existing `goss_parity_matrix` DEF-08-OOS-01). Revert any change that
  breaks a bit-exact test. Commit.

- **T5 — Report + C++ reference point.** Finish REPORT.md: per-lever before/after
  table, total speedup, one `lightgbm==4.6` pip-wheel timing point on matching
  synthetic data. SUMMARY.md + STATE.md "Quick Tasks Completed". Commit.

## Parity gate (non-negotiable)

`cargo test -p oracle-harness` must remain green (bit-exact CPU anchor) after T4.
Pre-existing known-failure `goss_parity_matrix` (DEF-08-OOS-01) is excluded.

## Out of scope

Half-precision f16/bf16 GPU kernels (breaks parity). C++ source build (using the
pinned pip wheel as the reference point instead).
