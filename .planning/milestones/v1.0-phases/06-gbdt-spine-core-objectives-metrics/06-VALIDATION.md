---
phase: 6
slug: gbdt-spine-core-objectives-metrics
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-07
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `06-RESEARCH.md` § Validation Architecture (layered L1–L5 golden battery, D-10..D-13 + end-to-end D-06/D-07).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cargo test` (workspace convention, all prior phases) |
| **Config file** | none — `cargo test --workspace` |
| **Quick run command** | `cargo test -p <crate-under-edit>` (touched crate, < 30s) |
| **Full suite command** | `cargo test --workspace` |
| **Oracle capture (human-gated, NOT in routine test)** | `cargo run -p xtask -- boosting-oracle-capture` (extends `learner-oracle-capture`) |
| **Estimated runtime** | ~60 seconds (workspace) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-under-edit>` (the touched crate's tests)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full workspace green + all layered goldens (L1–L5) replay bit-exact on the cubecl-cpu anchor; capture idempotent (`git diff` empty after regen)
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

> Layered golden battery (capture source per layer in 06-RESEARCH.md): L1 grad/hess (D-10), L2 per-iter score (D-11), L3 per-round metric (D-12), L4 bagged indices via RNG-replay (D-13, Option A), L5 end-to-end model+predict (D-06/D-07, ~40 cells).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {N}-XX-XX | XX | W | OBJ-01/03 | — | per-row g/h bit-exact f32 (L1) | unit | `cargo test -p lgbm-objective gradients` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | OBJ-02 | — | custom closure == Python fobj reference | integration | `cargo test -p oracle-harness --test boosting_parity custom_objective` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | MET-01 | — | 7 metrics match per round (L3) | integration | `cargo test -p lgbm-metric eval` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | MET-02 | — | metric_freq / training-metric cadence | unit | `cargo test -p lgbm-metric metric_infra` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | BST-01 | — | GBDT loop grows same trees + iter count (L5) | integration | `cargo test -p oracle-harness --test boosting_parity` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | BST-02 | — | per-iter score accumulation bit-exact (L2) | integration | `cargo test -p oracle-harness --test boosting_parity score_accumulation` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | BST-03 | — | bagged rows match RNG sequence (L4 RNG-replay) | unit | `cargo test -p lgbm-boosting bagging_rng` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | BST-07 | — | early stopping fires at same iter (L3 + best_iteration) | integration | `cargo test -p oracle-harness --test boosting_parity early_stopping` | ❌ W0 | ⬜ pending |
| {N}-XX-XX | XX | W | API-01 | — | builder→Config→train→predict end-to-end (L5) | integration | `cargo test -p lgbm public_api` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · Task IDs assigned by the planner.*

---

## Wave 0 Requirements

- [ ] `crates/lgbm-objective/` — crate scaffold + `ObjectiveError` (thiserror) + the 5 objectives + custom closure type
- [ ] `crates/lgbm-metric/` — crate scaffold + `MetricError` + the 7 metrics + `factor_to_bigger_better`
- [ ] `crates/lgbm-boosting/` — `GBDT`, `ScoreUpdater`, `BaggingSampleStrategy`, early-stop
- [ ] `crates/lgbm/` — builder + `Booster` (D-05 fields) + `train`/`predict`
- [ ] `lgbm-treelearner`: expose `add_prediction_to_score(&tree, &mut [f64])` (data-partition scatter) + `renew_tree_output` hook (BST-02, regression_l1)
- [ ] `lgbm-model::Tree`: confirm/add `shrinkage(rate)`, `add_bias(val)` with `MaybeRoundToZero` (audit — some present from P5)
- [ ] `xtask boosting-oracle-capture` + extended `xtask/py/` capture (L1–L5) + `Random.NextFloat` sequence dumper for L4
- [ ] `oracle-harness/tests/fixtures/boosting/` golden corpus + `REFERENCE_MANIFEST.md` entries
- [ ] `tests/boosting_parity.rs` (or `-p oracle-harness --test boosting_parity`) layered replay harness

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real `lib_lightgbm` 4.6 oracle capture (L1–L5 goldens, ~40 cells) | D-06/D-07, all phase reqs | Requires the C++ toolchain + real binary (not present at routine test time); produces committed fixtures | Run `cargo run -p xtask -- boosting-oracle-capture`; verify `git diff` is empty after regen (idempotent) |
| D-13 direct bag dump (fallback only) | BST-03 | If RNG-replay (Option A) is deemed insufficient, the only faithful route is a debug build of `lib_lightgbm` with a bag dump | checkpoint:human-verify — flag in plan if Option A insufficient |

*Primary D-13 path (RNG-replay, Option A) is fully automated; the direct bag dump is fallback-only.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
