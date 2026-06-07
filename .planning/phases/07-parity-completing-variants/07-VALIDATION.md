---
phase: 7
slug: parity-completing-variants
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-07
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `07-RESEARCH.md` § Validation Architecture. Phase 7 follows the carried Phase-5/6 real-binary layered-parity discipline: CPU `cubecl-cpu` f64-fold is the bit-exact hard merge gate; every golden is capture-gated against real `lib_lightgbm` 4.6.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cargo test --workspace` (oracle-harness integration tests) |
| **Config file** | per-crate `Cargo.toml`; fixtures in `crates/oracle-harness/tests/fixtures/` |
| **Quick run command** | `cargo test -p oracle-harness --test <subsystem>_parity <case>` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | full workspace suite ~minutes (fixtures committed; no C++ toolchain at test time) |

---

## Sampling Rate

- **After every task commit:** Run the slice's own `cargo test -p oracle-harness --test <subsystem>_parity <case>`
- **After every plan wave:** Run `cargo test --workspace` (catches numeric-spine regression — **critical for W7 categorical re-open**)
- **Before `/gsd-verify-work`:** Full suite green **and** the D-06 no-regression numeric-spine goldens (`spine_real.txt`, `mfb_pos_real.txt`, growth-path/subtract gates) bit-exact
- **Max feedback latency:** seconds (per-subsystem quick run); minutes (full workspace)

---

## Per-Task Verification Map

> Per-task IDs are assigned by the planner. This is the requirement→test wave map the planner must honor; capture-gated goldens are `checkpoint:human-verify` and replay-skip-pass until `lightgbm==4.6.0` is installed and the fixture is captured.

| Requirement | Wave | Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------|----------|-----------|-------------------|-------------|--------|
| DEF-06-01 / D-05 | W0 | bagged-subset split-gain determinism (FP-trace diagnostic) | real-binary FP-trace | new xtask trace subcommand + diagnostic test | ❌ W0 | ⬜ pending |
| OBJ-04 | W1 | huber/fair/poisson/quantile/mape/gamma/tweedie grad/hess+model | layered parity | `cargo test -p oracle-harness --test boosting_parity <obj>` | ❌ W1 | ⬜ pending |
| OBJ-05 | W2 | cross_entropy / cross_entropy_lambda | layered parity | `... boosting_parity <obj>` | ❌ W2 | ⬜ pending |
| MET-03 | W3 | extended regression/xentropy/multiclass metrics | parity | `... metric_parity` (new) | ❌ W3 | ⬜ pending |
| BST-04 | W4 | GOSS sample + amplify | parity + RNG-replay | `... boosting_parity goss` | ❌ W0/W4 | ⬜ pending |
| BST-05 | W5 | DART drop + normalize | parity + RNG-replay | `... boosting_parity dart` | ❌ W5 | ⬜ pending |
| BST-06 | W6 | RF averaged trees, mandatory bagging | parity | `... boosting_parity rf` | ❌ W6 | ⬜ pending |
| TRL-06 | W7 | categorical split (bitset/gain/model-text) | parity + layered + **no-regression** | `... learner_parity categorical` | ❌ W7 | ⬜ pending |
| OBJ-06 / MET-04 | W8 | lambdarank/rank_xendcg + ndcg/map + bagging_by_query | per-query parity + RNG-replay | `... rank_parity` (new) | ❌ W8 | ⬜ pending |
| PRD-04 | W9 | SHAP/predict_contrib (sum == raw margin) | parity | `... predict_parity contrib` | ❌ W9 | ⬜ pending |
| PRD-05 | W9 | prediction early stop | parity | `... predict_parity early_stop` | ❌ W9 | ⬜ pending |
| ADV-01..07 | W10/W11 | monotone/interaction/forced-split/extra-trees/CEGB/refit/importance | parity per axis | new parity files | ❌ W10/W11 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] **W0 diagnostic harness** — real-binary FP-trace capture for `binary_bag1_es0_bfa1` tree-0 subset histogram + per-split gain (new `xtask` trace subcommand), to root-cause DEF-06-01 / D-05 before any bagging-dependent wave (GOSS/RF/L1+bagging).
- [ ] **New parity test files** — `metric_parity.rs`, `rank_parity.rs` (or extend existing); categorical cases in `learner_parity.rs`; goss/dart/rf cases in `boosting_parity.rs`; contrib/early-stop in a predict parity file.
- [ ] **RNG-replay goldens** — dedicated goss / dart / bagging_by_query (and extra-trees) draw+call-order fixtures (Phase-6 D-13 pattern).
- [ ] **Capture pipeline** — extend `REFERENCE_MANIFEST.md` + real-`lib_lightgbm`-4.6 capture subcommands per subsystem; `lightgbm==4.6.0` wheel install is human-gated before any golden enforces.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real `lib_lightgbm` 4.6 golden capture | all capture-gated goldens | C++ wheel install + capture is a human-gated `checkpoint:human-verify` step; replay tests skip-pass until fixtures exist | Install `lightgbm==4.6.0`, run the per-subsystem `xtask` capture subcommand, commit fixtures, confirm replay tests flip from skip to green |
| ROCm cross-check (optional) | per carried Phase-6 deferral | CPU bit-exact is the hard gate; ROCm re-check is a research/planning call | If elected, run the subsystem parity on `cubecl-hip` (gfx1100) within ~1e-6 of the f64-fold anchor |

---

## Validation Sign-Off

- [ ] All tasks have an `<automated>` verify or a Wave 0 dependency
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (diagnostic harness, new parity files, RNG-replay goldens, capture pipeline)
- [ ] No watch-mode flags
- [ ] Numeric-spine no-regression goldens enforced at the W7 wave merge
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
