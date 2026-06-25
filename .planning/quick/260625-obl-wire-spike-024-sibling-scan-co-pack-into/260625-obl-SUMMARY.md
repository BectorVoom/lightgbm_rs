---
phase: quick-260625-obl
plan: 01
subsystem: gpu-treelearner
status: complete
tags: [spike-024, sibling-scan-copack, rocm, bit-exact, verify-reconcile]
requires:
  - spike-024 sibling-scan co-pack (shipped phase 12)
provides:
  - verified ground truth that co-pack is live + default-on + bit-exact
  - reconciled records (skill reference + STATE.md)
affects:
  - .claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md
  - .planning/STATE.md
tech-stack:
  added: []
  patterns:
    - "three-way env override (None=default-on / 0=off / 1=force-on) mirroring LGBM_RESIDENT_FORCE"
    - "GPU bit-exact gates pinned to the cubecl-cpu native anchor, never GPU-vs-GPU (def-f8u-01)"
key-files:
  created: []
  modified:
    - .claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md
    - .planning/STATE.md
decisions:
  - "EXPECTED branch confirmed: co-pack is fully wired + default-on; NO kernel/learner edit (verify-only)"
  - "The task brief's 'kernel/launcher exist but NOT called from learner.rs' claim is REFUTED by grep"
metrics:
  duration: ~3m (CPU gates fast; ROCm gates warm — kernels already cached this session)
  completed: 2026-06-25
---

# Quick Task 260625-obl: Verify + reconcile spike-024 sibling-scan co-pack wiring Summary

Resolved the "is spike-024 wired?" records discrepancy by grep ground truth: the sibling-scan
co-pack is LIVE on master and DEFAULT-ON, bit-exact in both `LGBM_SIBLING_COPACK` modes on CPU
and ROCm, with no committed golden changed — and reconciled the skill reference + STATE.md to
that verified truth.

## What This Task Was

A VERIFY-AND-RECONCILE quick task, NOT a wiring task. A task-brief "fresh source read" claimed the
2-slot co-pack kernel/launcher exist in `split.rs` but are NOT called from `learner.rs`, while the
project SKILL reference said "WIRED phase 12". Step 0 (Task 1) was to resolve this discrepancy by
grep before touching any code, then run the mandatory bit-exact gates (Task 2) and reconcile the
records (Task 3).

## Task 1 — Ground truth: EXPECTED branch (already wired + default-on)

The grep + call-site reads RESOLVED the discrepancy in favor of the SKILL reference. The co-pack is
fully wired and default-on:

- **Call site:** `crates/lgbm-treelearner/src/learner.rs:1839` calls
  `self.backend.scan_resident_siblings(...)` inside the `copack_feats`-gated block.
- **Eligibility gate** (`learner.rs:1788`): ANDs
  `self.resident_eligible && copack_override != Some(false) && smaller_resident_only &&
  smaller_scannable && larger_leaf >= 0 && larger_is_resident_subtract &&
  larger_resident_slot == larger_slot_id && larger_slot_id.is_some() && !larger_unified &&
  larger_splits.sum_hessians > 0.0 && larger_splits.num_data_in_leaf > 0`, plus IDENTICAL spine
  membership (`smaller_feats == larger_feats && !smaller_feats.is_empty()`). Any false case falls
  back to the BYTE-UNCHANGED two-separate-scans path.
- **Override semantics** (`crates/lgbm-treelearner/src/resident_pool.rs:282`): `sibling_copack_override()`
  parses `LGBM_SIBLING_COPACK` three-way — `None` (unset) ⇒ **co-pack engages** (default heuristic);
  `Some("0")` ⇒ FORCE-OFF (byte-identical two-scan path); `Some("1")` ⇒ FORCE-ON (identical to unset
  today — no separate co-pack size threshold yet). So it is **default-ON; `=0` is the off switch.**
- **Backend dispatch:** `RocmBackend::scan_resident_siblings` (`crates/lgbm-compute/src/lib.rs:2507`)
  calls `kernels::split::find_best_splits_fused_siblings_from_handles_on` (`split.rs:1584`) which
  launches `find_best_splits_fused_siblings_kernel` (`split.rs:1079`). The default `Backend`
  implementation (`lib.rs:1029`) errors ("device-resident pool not supported") — CpuBackend inherits
  it and never reaches the co-pack path because the gate ANDs in `resident_eligible`.

**Verdict:** EXPECTED. The brief's "exists but NOT called from learner.rs" claim is **REFUTED**.
The SKILL reference's "WIRED phase 12" is **CORRECT**. No code was edited.

## Task 2 — Mandatory bit-exact gates (both co-pack modes)

Every gate was run TWICE: `LGBM_SIBLING_COPACK` UNSET (default path) and `=1` (force co-pack).

### CPU-anchor + facade + oracle gates (no GPU feature)

| Suite | UNSET (default) | LGBM_SIBLING_COPACK=1 |
|-------|-----------------|------------------------|
| `cargo test -p lgbm-treelearner --lib` | 77 passed / 0 failed (2 ignored) | 77 / 0 |
| `cargo test -p lgbm` | 41 / 0 | 41 / 0 |
| `cargo test -p oracle-harness` — full suite | all green | all green |
| ↳ `raw_bin_train_matches_cpp_golden` (C++ golden RED-FLAG sentinel) | ok | ok (byte-idempotent) |
| ↳ `learner_parity_*` (29 tests) | 29 / 0 | 29 / 0 |
| ↳ `kernel_parity_sibling_copack_equals_two_scans_on_cpu` (cubecl-cpu W=1) | ok | ok |

### ROCm split-parity gates (real GPU, `--features rocm`)

| Suite | UNSET (default) | LGBM_SIBLING_COPACK=1 |
|-------|-----------------|------------------------|
| `cargo test -p oracle-harness --features rocm kernel_parity` | 17 passed / 0 failed (1 filtered) | 17 / 0 |
| ↳ `hip::kernel_parity_split_within_tol_on_hip` (~1e-6 hip split gate) | ok | ok |
| ↳ `hip::kernel_parity_sibling_copack_equals_two_scans_on_hip` (2-slot==two-scans, pinned to the cubecl-cpu native anchor per def-f8u-01, NOT GPU-vs-GPU) | ok | ok |

All other hip kernel_parity gates (`build_fix_scan`, `resident_build_fix_compact` incl. P>1,
`partition_exact`, `histogram_within_tol`, `fused_equals_per_feature_and_native`) also green in both
modes.

### RED-FLAG sentinel: clean

`git status` after Task 2 showed **zero tracked-file modifications** — no committed oracle golden
changed (`raw_bin_train_matches_cpp_golden` green and byte-idempotent across both modes). Co-pack is
bit-exact BY CONSTRUCTION (each feature's sequential scan is unchanged; spike-024 proved B's two
halves byte-identical to A's two separate scans, every cell), so this is the expected outcome and no
re-pin was performed. The CPU f64 anchor stayed byte-untouched; the `LightGBM*/` / `cuml-main/` /
`.serena/` reference trees were never git-added.

## Task 3 — Records reconciled to verified ground truth

- **`.claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md`:** expanded
  the "Wiring (done, phase 12)" bullet to record the resolved call site
  (`learner.rs:1839` → `lib.rs:2507` → `split.rs:1584`→`:1079`), made the **default-ON, `=0`-is-off**
  semantics explicit (so a future reader cannot repeat the brief's "exists but unwired / needs the
  flag set" misread), listed the full gate predicate, and added the CPU+HIP gate names as the
  verification anchors plus the no-auto-re-pin golden RED-FLAG rule.
- **`.planning/STATE.md`:** updated `last_activity_desc`, the Current Position "Last activity" line,
  and added a `260625-obl` row to the Quick Tasks Completed table recording the verified verdict and
  the full gate evidence. Phase/plan progress counters were left unchanged (this is a verification
  quick task, not a phase plan).

## Code Commits

**None.** This was the EXPECTED verify-only branch: Tasks 1 and 2 produced no code change, and Task 3
produced only doc reconciliation (the skill reference + STATE.md, which the orchestrator commits as
the docs commit). There was no kernel/learner/source edit to commit atomically — that is the expected
and correct outcome for this task (the plan explicitly forbids editing the hot GPU growth loop on the
EXPECTED branch).

## Deviations from Plan

None — plan executed exactly as written. The EXPECTED branch held (co-pack already wired + default-on);
no escalation to a wiring follow-up was needed.

## Honest Payoff (no e2e-win claim on this hardware)

Co-pack halves the per-leaf scan-readback SYNC count (≈59→≈30 syncs/tree), reclaiming ~½ the genuine
scan-sync ≈ **~10–15% small/medium, ~1.5% wide** — and ONLY on a real discrete gfx110x. This box is
the spoofed 8-CU APU where the CPU anchor crushes the GPU at every size (spike-001: GPU 0.06–0.36× of
CPU at 20k–100k). On THIS hardware co-pack is ROCm-parity-track maintenance, like 021/022.
**No end-to-end speedup is claimed on this hardware.**

## Self-Check: PASSED

- SUMMARY file written: `.planning/quick/260625-obl-wire-spike-024-sibling-scan-co-pack-into/260625-obl-SUMMARY.md`
- Modified doc files present and on disk: skill reference + STATE.md (verified via Task 3 grep).
- No code commits to verify (none made — verify-only branch, expected).
- No committed golden changed; reference trees not git-added.
