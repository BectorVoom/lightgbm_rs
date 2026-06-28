---
phase: 14-scaffold-oracle-slice-0
plan: 02
subsystem: treelearner
tags: [on-device-tree, env-gate, routing-fork, partition-payload, decide-once, cuda]

# Dependency graph
requires:
  - phase: 14 (plan 01)
    provides: "lgbm_dataset::LeafPartitionLayout payload P + Backend::on_device_growth_supported() (default false) + Backend::grow_tree_on_device() (default Ok(None))"
provides:
  - "lgbm-treelearner cuda_on_device_env() — inverse-default LGBM_CUDA_ON_DEVICE read (OFF unless `=1`)"
  - "SerialTreeLearner::on_device_eligible field, AND-gate-initialized ONCE at new (D-05)"
  - "DataPartition::from_payload(LeafPartitionLayout) reconstruction"
  - "train_inner decide-once on-device routing fork (Ok(None) ⇒ fall through, byte-unchanged)"
affects: [14-03 (oracle assert_on_device_tree_matches_cpu_anchor exercises the seam end-to-end via host fallback), slice-1 (real on-device kernel flips the discriminator and lights the fork live)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Read-at-new env gate (D-05): cache eligibility ONCE at construction, NOT per-train, when there is no per-train size-dependent input (intentional divergence from resident_eligible)"
    - "AND-gate eligibility: env toggle ANDs the backend discriminator so a false discriminator (CpuBackend, GpuBackend<R> in Slice 0) is unconditionally ineligible regardless of the env"
    - "Decide-once early-return fork at the TOP of train_inner; Some synthesizes the return tuple, Ok(None) falls through byte-unchanged (no host-fallback in production — D-02)"

key-files:
  created: []
  modified:
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-treelearner/src/data_partition.rs

key-decisions:
  - "D-05 honored: LGBM_CUDA_ON_DEVICE read ONCE at SerialTreeLearner::new via cuda_on_device_env(); on_device_eligible cached as backend.on_device_growth_supported() && cuda_on_device_env(), never recomputed in train_inner."
  - "D-02 honored: the production fork uses Ok(None) ⇒ fall through ONLY. NO unwrap_or_else(host_grow) host-fallback in the learner — that stand-in lives in Plan 03's oracle test."
  - "cuda_on_device_env() is the INVERSE default of autotune_enabled: OFF unless exactly `\"1\"` (unset/empty/`\"0\"`/malformed all yield false), guaranteeing SC#1 byte-unchanged with the env unset."

patterns-established:
  - "Inverse-default env gate via matches!(env::var(..).as_deref(), Ok(\"1\")) for opt-IN toggles (mirror-inverse of the autotune_enabled opt-OUT idiom)."
  - "POD payload → wrapper reconstruction (DataPartition::from_payload): thin field-move from a lower-crate mirror struct, keeping the trait seam acyclic."

requirements-completed: [ODL-01]

# Metrics
duration: 3min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 02: scaffold-oracle-slice-0 Summary

**ODL-01 learner wiring — the `cuda_on_device_env()` inverse-default gate, the `on_device_eligible` AND-gate field cached once at `new` (D-05), `DataPartition::from_payload`, and the decide-once `train_inner` routing fork that consumes the Plan 01 seam; with `LGBM_CUDA_ON_DEVICE` unset the fork is dead and the production path is byte-identical to master (SC#1).**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-29
- **Completed:** 2026-06-29
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `cuda_on_device_env() -> bool` in `lgbm-treelearner` — `matches!(std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(), Ok("1"))`, the INVERSE default of `autotune_enabled` (OFF unless exactly `"1"`; unset/empty/`"0"`/malformed all yield false, guaranteeing SC#1).
- `SerialTreeLearner::on_device_eligible: bool` field, initialized INLINE in the `new()` `Self { .. }` literal as `backend.on_device_growth_supported() && cuda_on_device_env()` — cached ONCE, never recomputed in `train_inner` (D-05 intentional divergence from `resident_eligible`). The AND-gate makes CpuBackend (discriminator false) and GpuBackend<R> (false in Slice 0) ineligible regardless of the env.
- `DataPartition::from_payload(lgbm_dataset::LeafPartitionLayout) -> Self` — a thin four-field move from the lower-crate POD payload, acyclic (treelearner already names both `DataPartition` and `lgbm_dataset`).
- The decide-once `train_inner` routing fork at the TOP of the function (line 704), ahead of the `resident_eligible` recompute (line 731): `if self.on_device_eligible { if let Some((tree, payload)) = self.backend.grow_tree_on_device(gradients, hessians, self.num_leaves, self.max_depth)? { return Ok((tree, Vec::new(), ColSamplerTrace::default(), DataPartition::from_payload(payload))); } }`. On `Ok(None)` execution falls through UNCHANGED; production has NO host-fallback (D-02).
- Full merge gate green AND byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset (775 passed / 3 ignored, 0 failed across the workspace).

## Task Commits

Each task was committed atomically:

1. **Task 1: Add cuda_on_device_env helper, on_device_eligible field, new()-time AND-gate init** - `eaad883` (feat)
2. **Task 2: Add DataPartition::from_payload and the decide-once train_inner routing fork** - `72e34bb` (feat)

## Files Created/Modified
- `crates/lgbm-treelearner/src/learner.rs` - Added the free `cuda_on_device_env()` helper, the `on_device_eligible` struct field (doc'd as D-05 read-once-at-new), its inline AND-gate `new()` init, and the decide-once on-device fork at the top of `train_inner`.
- `crates/lgbm-treelearner/src/data_partition.rs` - Added `DataPartition::from_payload(LeafPartitionLayout)` reconstruction.

## Decisions Made
- **D-05 read-at-new (NOT per-train).** `LGBM_CUDA_ON_DEVICE` is read exactly once at `SerialTreeLearner::new` and cached in `on_device_eligible`. On-device eligibility has no per-train, size-dependent input the way resident does (which size-gates on `num_data`), so a per-train env re-read buys nothing and adds a syscall per tree. The field is never reassigned inside `train_inner` (the divergence is deliberate, not an oversight).
- **D-02 no host-fallback in production.** The learner's fork uses `Ok(None) ⇒ fall through` ONLY. The `unwrap_or_else(host_grow)` stand-in for the not-yet-existent on-device tree belongs to Plan 03's oracle test, never the learner — routing production through a live fallback was explicitly avoided.
- **Inverse-default env parse.** `cuda_on_device_env` mirrors the shape of `autotune_enabled` but inverts the default (opt-IN, not opt-OUT), so the only value that enables the fork is exactly `"1"`.

## Deviations from Plan

None - plan executed exactly as written. Both tasks landed with the exact signatures, fork placement, and AND-gate semantics specified; no auto-fixes (Rules 1-3) or architectural changes (Rule 4) were needed.

## Issues Encountered
- An expected interim `dead_code` warning (`field on_device_eligible is never read`) appeared after Task 1, since the field is only read by the Task 2 fork. It resolved automatically once Task 2 wired the fork — no action required.

## Verification Evidence
- `cargo build -p lgbm-treelearner` exits 0 (no warnings after Task 2).
- `cargo test -p lgbm-treelearner` — 77 passed / 2 ignored; `cargo test -p oracle-harness --test learner_parity` — 29/29 (spine unregressed).
- Merge gate: `--test raw_bin_train_parity` 2/2; `--test kernel_parity` 7/7.
- `cargo test --workspace` — **775 passed, 3 ignored, 0 failed** (byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset, SC#1).
- Acceptance greps: `fn cuda_on_device_env` present with `Ok("1")` body; `on_device_eligible` has exactly two sites (field decl + single inline `new()` init, no `train_inner` reassignment); `from_payload` moves all four fields; the `if self.on_device_eligible` guard (line 704) precedes the `resident_eligible` recompute (line 731); no host-fallback `unwrap_or_else` in the learner (the only match is the D-02 explanatory comment).

## Threat Surface
- T-14-02 (Tampering, `cuda_on_device_env` env parse) — MITIGATED: strict `matches!(env::var(..).as_deref(), Ok("1"))`; any other value yields false. No path/format/injection surface.
- T-14-03 (Tampering/Repudiation, routing fork) — MITIGATED: eligibility ANDs the backend discriminator (false on all Slice-0 backends), so the fork is unreachable in production; `Ok(None)` falls through byte-unchanged. Enforced by the byte-unchanged merge gate (SC#1).
- No NEW security-relevant surface beyond the planned threat register — no new endpoints, auth paths, file access, or schema changes.

## User Setup Required
None - no external service configuration required. `LGBM_CUDA_ON_DEVICE` stays unset in production (the default-off toggle); Slice 1 will flip the backend discriminator to light the fork.

## Next Phase Readiness
- Plan 03 (oracle) can now exercise the seam end-to-end: `backend.grow_tree_on_device(..)?.unwrap_or_else(|| host_grow(..))` in the oracle TEST feeds `assert_on_device_tree_matches_cpu_anchor`, with `DataPartition::from_payload` available for any partition reconstruction.
- Slice 1 will wire a real on-device kernel and flip `on_device_growth_supported()` to true; the `on_device_eligible` AND-gate + `train_inner` fork then go live with no further wiring change — only the discriminator and the env toggle gate it.

## Self-Check: PASSED

Both modified files present; both task commits (`eaad883`, `72e34bb`) exist in git history.

---
*Phase: 14-scaffold-oracle-slice-0*
*Completed: 2026-06-29*
