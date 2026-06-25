---
quick_id: 260626-a6t
slug: wire-spike-035-rocm-host-partition-defau
description: Wire spike-035 — make the ROCm backend route partition on the host by default
date: 2026-06-26
status: complete
---

# Quick Task 260626-a6t: Wire spike-035 (rocm host partition default-ON) — Summary

One-liner: Flipped `RocmBackend::prefers_host_partition()` from the spike-035 default-OFF
A/B gate to default-ON-with-OFF-switch, routing the ROCm partition through the shipped
spike-027 host fused path by default; full CPU bit-exact merge gate + ROCm anchor-pinned
re-pin gate stay green on the real GPU.

## Change

- **File:line:** `crates/lgbm-compute/src/lib.rs:2067-2068` (inside the `#[cfg(feature = "rocm")] impl Backend for RocmBackend` block).
- **Before:** `matches!(std::env::var("LGBM_ROCM_HOST_PARTITION").as_deref(), Ok("1"))` (default OFF; `=1` opt-in for the A/B).
- **After:** `!matches!(std::env::var("LGBM_ROCM_HOST_PARTITION").as_deref(), Ok("0"))` (default ON; `=0` forces the old device round-trip for benching/rollback), mirroring the `LGBM_SIBLING_COPACK` off-switch idiom.
- Comment updated to mark spike-035 SHIPPED (quick-260626-a6t) with the ~1.18-1.23x launch-bound / wash-at-wide / ~1e-6-parity (def-f8u-01, not a bit-exact swap) rationale.
- Net diff: 6 insertions, 9 deletions (one function, comment + body).
- **Commit:** `da3032f` — `feat(quick-260626-a6t): default rocm partition to the host fused path (wire spike-035)`.

## Build result (Task 1)

- `cargo build --features rocm -p lgbm-compute` → **compiles clean** (Finished dev profile, no warnings on the edited file). This machine has ROCm.

## Gate results (Task 2) — all three green

1. **`cargo test -p lgbm-treelearner --lib` (default)** — **77 passed / 0 failed** (2 ignored). Expected 77/0. ✅
2. **`cargo test -p oracle-harness` (default features, CPU bit-exact merge gate)** — **all suites green, 0 failed** across every test binary (kernel_parity_cpu 18→ N/A here; full set incl. boosting/learner/rank/raw_bin/rng parity). CPU f64 anchor path byte-untouched (change is inside `cfg(feature="rocm")`). ✅
3. **`cargo test -p oracle-harness --features rocm` (full hip + kernel parity ON THE GPU)** — **0 failed**, 15 `test result: ok` binaries. Key anchor-pinned re-pin tests:
   - `hip::learner_parity_resident_equals_host_tree_on_hip` ... **ok** ✅
   - `hip::learner_parity_fused_equals_host_tree_on_hip` ... **ok** ✅
   - `hip::kernel_parity_partition_exact_on_hip` ... **ok** ✅
   - kernel_parity suite: **18 passed / 0 failed**; learner_parity suite: **31 passed / 0 failed** (29 CPU + 2 hip anchor-pinned).

## Parity discipline (def-f8u-01)

This was NOT a bit-exact swap. No host-rocm == device-rocm assertion was added or expected;
the valid gate is the anchor-pinned hip tests (GPU f32 tree pinned to the cpu f64 anchor within
the ~1e-6 ROCm contract), all of which passed unchanged. No tolerance was loosened; the CPU f64
anchor was not touched.

## Deviations from Plan

None — plan executed exactly as written.

## Out of scope (per plan; orchestrator handles)

- Did NOT commit docs (PLAN/SUMMARY/STATE), the spike-035 README, or the MANIFEST row.
- Did NOT update ROADMAP.md.
- Did NOT git-add the untracked reference trees (`LightGBM*/`, `cuml-main/`, `.serena/`).
- Ran on master, no worktree isolation.

## Self-Check: PASSED

- Code change present at `crates/lgbm-compute/src/lib.rs:2067` (verified via grep).
- Commit `da3032f` exists in `git log`.
- All three gate suites green on this machine's real ROCm GPU.
