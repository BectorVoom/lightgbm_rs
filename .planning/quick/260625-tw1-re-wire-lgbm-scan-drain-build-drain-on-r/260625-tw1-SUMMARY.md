---
quick_id: 260625-tw1
slug: re-wire-lgbm-scan-drain-build-drain-on-r
description: Re-wire LGBM_SCAN_DRAIN build-drain onto the resident sibling-co-pack scan path
date: 2026-06-25
status: complete
tasks_completed: 2
tasks_total: 2
key-files:
  modified:
    - crates/lgbm-compute/src/kernels/split.rs
commits:
  - 128a4c2
---

# Quick Task 260625-tw1: Re-wire LGBM_SCAN_DRAIN on the co-pack scan path — Summary

Restored the `LGBM_SCAN_DRAIN` build-drain on the Phase-12 production sibling co-pack scan path (`find_best_splits_fused_siblings_from_handles_on`), which previously had no drain block — only the single-leaf `find_best_splits_fused_inner` did. The drain is double-env-gated and byte-neutral when off; the bit-exact gate is green.

## What changed (file:line)

`crates/lgbm-compute/src/kernels/split.rs`, function `find_best_splits_fused_siblings_from_handles_on`:

1. **`_scan_prof` binding** — added `let _scan_prof = crate::fusion_prof::scan_enabled();` immediately after the empty-batch early return (~split.rs:1600), mirroring the single-leaf binding at :1342. This was absent in the siblings fn.

2. **Drain block** — added a `LGBM_SCAN_DRAIN` co-pack analog block immediately BEFORE the scan kernel launch (before the `let scan_w = scan_cube_dim();` / `CubeCount` setup, ~split.rs:1725). It drains BOTH resident sibling histogram handles:
   ```rust
   if _scan_prof && crate::fusion_prof::scan_drain_enabled() {
       let t_drain = std::time::Instant::now();
       let _ = client.read_one_unchecked(hist_a_handle.clone());
       let _ = client.read_one_unchecked(hist_b_handle.clone());
       crate::fusion_prof::SCAN_DRAIN_NS.fetch_add(
           t_drain.elapsed().as_nanos() as u64,
           std::sync::atomic::Ordering::Relaxed,
       );
   }
   ```
   The two per-sibling resident histogram device Handles are `hist_a_handle` (A = smaller) and `hist_b_handle` (B = larger), the function's two leading parameters. They are cloned for the drain because the originals are moved into `ArrayArg::from_raw_parts(hist_a_handle, ...)` / `(hist_b_handle, ...)` at the launch. The drain is placed before the launch (and before any launch-side setup) so the scan launch+readback timing remains pure scan, matching the single-leaf ordering.

Net: 1 file changed, 20 insertions, 0 deletions.

## Gating / behavior-neutrality

The block is double-gated `_scan_prof && crate::fusion_prof::scan_drain_enabled()` = `LGBM_SCAN_PROF=1` AND `LGBM_SCAN_DRAIN=1`, identical to the single-leaf path. With either env off it is dead code (no device reads, no value/order changes) → default behavior is byte-unchanged.

## Verification

### Task 1 — build (`--features rocm`)
- `cargo build --features rocm -p lgbm-compute` → Finished, clean.
- `cargo build --features rocm` (full workspace) → Finished in 35.43s, clean. Compiles.

### Task 2 — bit-exact gate (default features, no rocm)
- `cargo test -p lgbm-treelearner --lib` → **77 passed; 0 failed; 2 ignored**.
- `cargo test -p oracle-harness` → all binaries green, **0 failed** across every test binary (incl. `raw_bin_train_matches_cpp_golden`, `rng_parity_replays_every_committed_case`, the 75-test main suite, etc.).

Bit-exact gate green; change is parity-neutral as designed.

## Deviations from Plan

None — plan executed exactly as written. The two sibling handle bindings (`hist_a_handle` / `hist_b_handle`) and the `_scan_prof` binding were resolved by reading the fn body as instructed.

## Out of scope (per plan)
- The pre-existingly-broken rocm oracle tests `hip::learner_parity_{resident,fused}_equals_host_tree_on_hip` (`subtract_resident: smaller slot is empty`) were NOT run for the gate and NOT touched.
- GPU drain-effectiveness verification on the real APU (`--features rocm` + GPU, reading `SCAN_DRAIN_NS` vs scan launch+readback) is the orchestrator's post-commit step.

## Self-Check: PASSED
- crates/lgbm-compute/src/kernels/split.rs — FOUND (modified, committed)
- commit 128a4c2 — FOUND in git log
