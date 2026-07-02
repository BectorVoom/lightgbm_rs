---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
plan: 03
subsystem: planning-docs
tags: [bookkeeping, requirements, roadmap, d-05, odl-18h]
requires: [".planning/phases/20-on-device-score-updater-metrics/20-VERIFICATION.md"]
provides:
  - "ODL-18H (new Phase-21 hardening requirement ID)"
  - "ODL-18/ODL-19 re-attributed to Phase 20 / Complete"
  - "Re-cut ROADMAP Phase 21 body (hardening scope)"
affects: [".planning/REQUIREMENTS.md", ".planning/ROADMAP.md"]
tech-stack:
  added: []
  patterns: ["D-05 bookkeeping reconciliation (scoped markdown edits, grep-verified)"]
key-files:
  created: []
  modified:
    - ".planning/REQUIREMENTS.md"
    - ".planning/ROADMAP.md"
decisions:
  - "ODL-18/ODL-19 marked Complete under Phase 20 (delivered there per D-01, 20-VERIFICATION 6/6), not Phase 21"
  - "New ODL-18H hardening requirement created for the actual Phase-21 work (parity corpus + WR-01 confirmation + _with_cfg seam)"
  - "ROADMAP data->leaf aliasing / batched-read open question recorded as NOT affecting the Phase-21 parity gate (moot for the live driver / Phase-23 perf)"
metrics:
  duration_min: 2
  completed: "2026-07-02"
  tasks: 2
  files_changed: 2
status: complete
---

# Phase 21 Plan 03: D-05 Requirement/ROADMAP Bookkeeping Reconciliation Summary

Reconciled the planning docs with Phase-20 reality: marked ODL-18/ODL-19 Complete under Phase 20 (where the on-device driver was delivered per D-01), added the new ODL-18H hardening requirement mapped to Phase 21, and re-cut the stale ROADMAP Phase 21 body to the parity-hardening scope. Pure docs edit — no code, no behavior change.

## What Was Built

### Task 1 — REQUIREMENTS.md (commit `7b8d857`)
- ODL-18 and ODL-19 checklist boxes flipped to `- [x]`, each with a parenthetical citing Phase-20 delivery + 20-VERIFICATION.md 6/6 (crit 5 STRUCTURE gate / crit 6 no-f64-per-row).
- New `**ODL-18H**` checklist item (unchecked, checked at phase completion) describing the hardening scope: targeted STRUCTURE parity corpus (deep >2-live-leaf, no-split break, min-data/min-hessian-constrained), the WR-01 `HistArena::swap` free-slot fix confirmation + repro, and the additive `grow_tree_on_device_driver_with_cfg` seam — cpu-f64-anchored on the cubecl-cpu default lane, `LGBM_CUDA_ON_DEVICE`-gated.
- Traceability table: `ODL-18 | Phase 20 | Complete`, `ODL-19 | Phase 20 | Complete`, new `ODL-18H | Phase 21 | Pending` row in ODL order.
- Per-phase rollup: Phase 20 row now `ODL-16, ODL-17, ODL-18, ODL-19 | 4`; Phase 21 row now `On-device driver hardening + parity corpus | ODL-18H | 1`.
- Coverage summary: 23 total (was 22, +ODL-18H), 23 mapped (100%), no orphans/duplicates; each requirement mapped to exactly one phase (ODL-18/19 → Phase 20, ODL-18H → Phase 21). Rollup counts sum to 23.

### Task 2 — ROADMAP.md (commit `42a6864`)
- Phase 21 `####` body re-cut from the stale "end-to-end driver orchestrates the full per-leaf grow loop" text to the hardening scope: new heading, Goal (confirm WR-01 + broaden parity corpus + reconcile bookkeeping, no re-implementation), Requirements (ODL-18H; note ODL-18/19 delivered Phase 20 per D-01), 5 Success Criteria (targeted STRUCTURE gates bit-exact, WR-01 confirmed closed `c9a7fd1`, additive `_with_cfg` seam, env-unset byte-unchanged, bookkeeping reconciled).
- Notes: defer categorical → Phase 22 (ODL-22) and perf/default-on → Phase 23 (ODL-20/21); record the resolved open question — data→leaf `Handle` aliasing is MOOT for the live driver (host per-leaf rows, rebuilds `LeafPartitionLayout`), batched `client.read(vec![h])` is a Phase-23 perf concern — neither affects the Phase-21 parity gate.
- Tightened the top-level Phase 21 checklist summary line (:86), dropping the now-done "Re-cut via `/gsd-phase` before planning".

## Deviations from Plan

None — plan executed exactly as written. All scoped edits landed; no unrelated requirements or phase bodies touched.

## Verification

- Task 1 automated: `grep -q "ODL-18H"` AND `grep -q "ODL-18 | Phase 20 | Complete"` AND `grep -q "ODL-19 | Phase 20 | Complete"` → `BOOKKEEPING_OK`. ODL-18H occurrences = 5 (≥3 required). ODL-18/19 checklist lines confirmed `- [x]`.
- Task 2 automated: `grep -qi "Harden"` AND `grep -q "ODL-18H"` AND `grep -qi "Phase 23"` → `ROADMAP_RECUT_OK`. Stale "end-to-end on device and reconstitutes" body text absent from Phase 21 section; Phase 22/20 bodies intact.
- REQUIREMENTS coverage totals 23, 100% mapped, no orphans/duplicates.

## Notes for Reviewers

Sequential executor on the main working tree; normal commits (hooks on). A prior wave plan (21-01) had already touched ROADMAP.md — current on-disk content was read before editing (Phase 21 checklist line was already the D-01 "Hardening/Slack" text, now further tightened). No code paths exercised — this is a bookkeeping-only reconciliation closing the audit trail honestly (T-21-07 mitigated: ODL-18/19 attributed to Phase 20 with the VERIFICATION 6/6 citation, tables grep-consistent).

## Self-Check: PASSED
- FOUND: .planning/REQUIREMENTS.md (ODL-18H present, 5 occurrences)
- FOUND: .planning/ROADMAP.md (re-cut Phase 21 body, ODL-18H present)
- FOUND commit 7b8d857 (Task 1)
- FOUND commit 42a6864 (Task 2)
