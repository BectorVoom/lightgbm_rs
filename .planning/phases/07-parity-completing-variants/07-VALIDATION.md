---
phase: 7
slug: parity-completing-variants
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-07
validated: 2026-06-07
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
| DEF-06-01 / D-05 | W0 | bagged-subset split-gain determinism (FP-trace diagnostic) | real-binary FP-trace | `... boosting_parity subset_determinism_diagnostic` (+ `fixtures/determinism/*_subset_trace.txt`) | ✅ boosting_parity.rs | ✅ green |
| OBJ-04 | W1 | huber/fair/poisson/quantile/mape/gamma/tweedie grad/hess+model | layered parity | `cargo test -p oracle-harness --test boosting_parity <obj>` | ✅ boosting_parity.rs | ✅ green¹ |
| OBJ-05 | W2 | cross_entropy / cross_entropy_lambda | layered parity | `... boosting_parity cross_entropy` | ✅ boosting_parity.rs | ✅ green |
| MET-03 | W3 | extended regression/xentropy/multiclass metrics | parity | `... metric_parity` + `lgbm-metric` unit (auc/ap/auc_mu) | ✅ metric_parity.rs | ✅ green |
| BST-04 | W4 | GOSS sample + amplify | parity + RNG-replay | `... boosting_parity goss` | ✅ boosting_parity.rs | ✅ green |
| BST-05 | W5 | DART drop + normalize | parity + RNG-replay | `... boosting_parity dart` | ✅ boosting_parity.rs | ✅ green |
| BST-06 | W6 | RF averaged trees, mandatory bagging | parity | `... boosting_parity rf` | ✅ boosting_parity.rs | ✅ green |
| TRL-06 | W7 | categorical split (bitset/gain/model-text) | parity + layered + **no-regression** | `... learner_parity categorical` | ✅ learner_parity.rs | ✅ green |
| OBJ-06 / MET-04 | W8 | lambdarank/rank_xendcg + ndcg/map + bagging_by_query | per-query parity + RNG-replay | `... rank_parity` | ✅ rank_parity.rs | ✅ green |
| PRD-04 | W9 | SHAP/predict_contrib (sum == raw margin) | parity | `... predict_parity contrib` | ✅ predict_parity.rs | ✅ green |
| PRD-05 | W9 | prediction early stop | parity | `... predict_parity early_stop` | ✅ predict_parity.rs | ✅ green |
| ADV-01..05 | W10 | monotone/interaction/forced-split/extra-trees/CEGB | parity per axis + unit | `advanced_parity` + `lgbm-treelearner` learner unit (monotone/interaction/forced/extra_trees/cegb) | ✅ advanced_parity.rs + treelearner unit | ✅ green² |
| ADV-06/07 | W11 | refit/continue-training + feature importance | real-binary parity | `... advanced_parity` (refit/importance/continue) | ✅ advanced_parity.rs | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

¹ OBJ-04: supported cells green; 13 learner-level cells (fair, gamma, quantile bagged+iterated, tweedie bfa-off) `#[ignore]`'d under DEF-07-02 (g/h bit-exact; documented f64 split-gain knife-edge awaiting a source-built `lib_lightgbm` 4.6 FP trace). One monotone mixed-vector ULP cell `#[ignore]`'d under DEF-07-11. 17 ignores total — matches the deferred-items contract.
² ADV constraint-length validation (T-07-11-02) added in security gate: `lgbm::booster::tests::wrong_length_constraint_vectors_are_typed_errors`.

---

## Wave 0 Requirements

- [x] **W0 diagnostic harness** — real-binary FP-trace capture for the bagged-subset tree-0 histogram + per-split gain landed (`fixtures/determinism/*_subset_trace.txt`, `subset_determinism_diagnostic` test); root-caused DEF-06-01 / D-05 → the `min_gain_shift` RAW-vs-BUMPED `sum_hessian` operand fix (07-D05-DECISION.md), un-deferring regression_l1 + bagging.
- [x] **New parity test files** — `metric_parity.rs`, `rank_parity.rs`, `predict_parity.rs`, `advanced_parity.rs` created; categorical cases in `learner_parity.rs`; goss/dart/rf + objective cases in `boosting_parity.rs`.
- [x] **RNG-replay goldens** — goss (`goss_rng_replay`), dart (`dart_drop_rng_replay`), bagging-by-query (`bagging_by_query_seed3`), and rank_xendcg objective_seed draw+call-order fixtures committed.
- [x] **Capture pipeline** — per-subsystem real-`lib_lightgbm`-4.6 `xtask` capture subcommands + `REFERENCE_MANIFEST.md` extended; `lightgbm==4.6.0` wheel install human-gated; all replay tests flipped from skip to green.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real `lib_lightgbm` 4.6 golden capture | all capture-gated goldens | C++ wheel install + capture is a human-gated `checkpoint:human-verify` step; replay tests skip-pass until fixtures exist | Install `lightgbm==4.6.0`, run the per-subsystem `xtask` capture subcommand, commit fixtures, confirm replay tests flip from skip to green |
| ROCm cross-check (optional) | per carried Phase-6 deferral | CPU bit-exact is the hard gate; ROCm re-check is a research/planning call | If elected, run the subsystem parity on `cubecl-hip` (gfx1100) within ~1e-6 of the f64-fold anchor |

---

## Validation Audit 2026-06-07

State A re-audit after full phase execution. Every plan-time MISSING reference is now an existing, green test; no auditor pass needed (zero gaps).

| Metric | Count |
|--------|-------|
| Requirements audited | 13 (18 req IDs incl. ADV-01..07) |
| COVERED (green) | 13 |
| PARTIAL | 0 |
| MISSING | 0 |
| Gaps filled this run | 0 (all already covered) |

Evidence: `cargo test --workspace` → **683 passed / 0 failed / 17 ignored** (13 DEF-07-02 + 4 DEF-07-11 deferrals — contracted, not gaps). Per-subsystem parity files (`boosting_parity`, `metric_parity`, `rank_parity`, `learner_parity`, `predict_parity`, `advanced_parity`) all green; Wave-0 determinism diagnostic + RNG-replay goldens present. Cross-checked with `07-VERIFICATION.md` (18/18 passed) and `07-UAT.md` (10/10).

---

## Validation Sign-Off

- [x] All tasks have an `<automated>` verify or a Wave 0 dependency
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (diagnostic harness, new parity files, RNG-replay goldens, capture pipeline)
- [x] No watch-mode flags
- [x] Numeric-spine no-regression goldens enforced at the W7 wave merge
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-06-07 — post-execution audit confirms all 13 requirements COVERED with green automated parity tests (683/0/17); Wave 0 complete; 0 gaps. Originally approved 2026-06-07 (plan-checker VERIFICATION PASSED; Dimension-8 contract).
