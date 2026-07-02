---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
plan: 01
subsystem: lgbm-compute (on-device grow driver + histogram arena)
tags: [on-device, driver, wr-01, hardening, extract-param, parity-gate]
dependency_graph:
  requires:
    - "HistArena::swap free-slot fix (c9a7fd1, Phase 18)"
    - "grow_tree_on_device_driver + proving_slice_config (Phase 18)"
  provides:
    - "grow_tree_on_device_driver_with_cfg (explicit GainConfig extract-param variant)"
    - "WR-01 confirmed-closed audit trail on HistArena::swap"
  affects:
    - "plan 21-02 (min-data parity case C can call the driver directly with a constrained GainConfig)"
tech_stack:
  added: []
  patterns:
    - "extract-parameter refactor (thin delegator + _with_cfg variant)"
    - "additive change behind an untouched trait seam (byte-unchanged merge gate)"
key_files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram_arena.rs
    - crates/lgbm-compute/src/kernels/grow_driver.rs
decisions:
  - "WR-01 is confirmed (not rebuilt): the live driver carries per-leaf Vec<f64> histograms and never consumes HistArena, so the swap free-slot scan stays unit-test-locked."
  - "GainConfig threaded via an additive _with_cfg variant, NOT through the Backend::grow_tree_on_device trait seam, to keep the default merge gate byte-unchanged."
metrics:
  duration: ~15m
  completed: 2026-07-02
  tasks: 2
  files_modified: 2
status: complete
requirements: [ODL-18H]
---

# Phase 21 Plan 01: On-Device Driver Hardening (WR-01 confirm + `_with_cfg` extract) Summary

Hardened the lgbm-compute half of the on-device driver: confirmed-and-documented the
already-landed WR-01 `HistArena::swap` slot-aliasing fix, and extracted an additive
`grow_tree_on_device_driver_with_cfg` variant so a `GainConfig` (e.g. binding
`min_data_in_leaf`) can be threaded into the driver directly by a test — with the
`Backend::grow_tree_on_device` trait seam left byte-unchanged.

## What Was Built

### Task 1 — WR-01 `HistArena::swap` free-slot fix: confirmed + documented
- Confirmed by inspection that `HistArena::swap` selects the smaller child's `fresh`
  slot via a free-slot scan against the `leaf_to_slot` occupancy set
  (`let occupied: HashSet<_> = self.leaf_to_slot.values()...` + `let fresh = (0..num_slots).find(|s| *s != parent_slot && !occupied.contains(s))`), rather than
  the pre-fix modulo-successor heuristic — and drops the now-internal parent leaf's
  stale slot key.
- The modulo-successor picker at `:257` belongs to the separate `rotate` method and
  was left untouched.
- Added a 6-line doc block at the top of `HistArena::swap` recording: WR-01 closed in
  `c9a7fd1` (Phase 18); the free-slot scan prevents aliasing a live sibling's slot;
  and that the live grow driver (`grow_tree_on_device_driver`) does NOT consume this
  arena (it carries per-leaf `Vec<f64>` histograms), so the arena is unit-test-locked,
  not driver-wired.
- Repro tests green: `swap_multileaf_never_aliases_live_sibling_slot`,
  `swap_errors_when_pool_exhausted`, `swap_rejects_single_slot_pool` (15/15 in the
  `histogram_arena` filter). No fix re-applied, no assert weakened.

### Task 2 — `grow_tree_on_device_driver_with_cfg` extract-parameter refactor
- Renamed the driver body into a new public
  `grow_tree_on_device_driver_with_cfg<R: cubecl::Runtime>(...)` that takes the
  existing args plus a trailing `cfg: GainConfig`, replacing the internal
  `let cfg = proving_slice_config();` with the passed `cfg`.
- `grow_tree_on_device_driver<R>(...)` is now a thin delegator that calls the
  `_with_cfg` variant with `proving_slice_config()`.
- Body otherwise byte-identical: same length/num_leaves guards, ordered-f64 root fold,
  build/subtract/scan/partition sequence, shared break path, and typed
  `ComputeError::Runtime` / `LengthMismatch` boundaries.
- The `Backend::grow_tree_on_device` trait seam and all its callers
  (`CpuBackend`/`GpuBackend<R>`/`learner.rs`) are untouched; `crates/lgbm-compute/src/lib.rs` diff is empty.

## Verification

- `cargo test -p lgbm-compute --lib histogram_arena` — 15 passed, 0 failed; the three
  named WR-01 repro tests appear and pass.
- `grep "let occupied"` / `grep "let fresh = (0.."` — both hit inside `swap`.
- `cargo build -p lgbm-compute` — Finished (clean).
- `grep -c "fn grow_tree_on_device_driver_with_cfg"` == 1; delegation call present at
  `:432` passing `proving_slice_config()`.
- `git diff crates/lgbm-compute/src/lib.rs` — empty (trait seam unchanged).
- `cargo test -p oracle-harness --test learner_parity` — 32 passed, 0 failed
  (env-unset default merge gate byte-unchanged).

## Deviations from Plan

None — plan executed exactly as written.

## Notes for Downstream (plan 21-02)

- Plan 21-02 case C (min-data parity) can now call `grow_tree_on_device_driver_with_cfg`
  directly with a constrained `GainConfig` (e.g. `min_data_in_leaf > 1`) to make the
  constraint observably bind through the driver, without widening the trait seam.
- Per 21-RESEARCH Pitfall 1: the parity corpus in 21-02 does NOT exercise the
  `HistArena::swap` path (the live driver uses per-leaf `Vec<f64>`), so WR-01 remains
  covered only by the arena unit tests — this is intentional and documented, not a gap.

## Self-Check: PASSED
- FOUND: crates/lgbm-compute/src/kernels/histogram_arena.rs (WR-01 doc block + free-slot scan)
- FOUND: crates/lgbm-compute/src/kernels/grow_driver.rs (grow_tree_on_device_driver_with_cfg + delegator)
- FOUND commit 51a4bec (Task 1)
- FOUND commit 907bb5c (Task 2)
