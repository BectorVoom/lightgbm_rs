# Phase 21: Harden the On-Device Driver - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-02
**Phase:** 21-harden-on-device-driver (re-cut)
**Areas discussed:** Phase re-cut, Parity corpus breadth, WR-01 fix + repro, ROCm hardware pass, Requirement bookkeeping

---

## Phase re-cut (surfaced conflict)

Phase 21's ROADMAP detail body still described the end-to-end driver
(ODL-18/19), but Phase 20 (D-01 pull-forward) already delivered it — Phase 20
VERIFICATION.md `passed` 6/6 with `requirements_verified: [ODL-16,17,18,19]`. The
ROADMAP checklist itself said to re-cut Phase 21 to hardening or fold into 22/23.

| Option | Description | Selected |
|--------|-------------|----------|
| Harden the driver | Re-cut as hardening/parity-slack: fix WR-01, broaden STRUCTURE corpus, reconcile bookkeeping | ✓ |
| Fold into 22/23 | Bookkeeping only; go straight to Phase 22 categorical | |
| Wire device metric eval | Re-cut around routing EvalKernel into GBDT | |

**User's choice:** Harden the driver.
**Notes:** Parity is the project's hard contract and is currently proven on only
one 4-leaf case plus a known latent bug (WR-01).

---

## Parity corpus breadth

| Option | Description | Selected |
|--------|-------------|----------|
| Targeted risk cases | Deep >2-live-leaf, no-split/single-leaf, min_data/min_sum_hessian constrained | ✓ |
| Broad shape sweep | Full rows×features×num_leaves×constraints matrix | |
| Extend existing corpus | Route existing SerialTreeLearner parity configs through the on-device gate | |

**User's choice:** Targeted risk cases.

---

## WR-01 slot-aliasing fix + validation

| Option | Description | Selected |
|--------|-------------|----------|
| Fix + dedicated repro test | Free-slot-scan fix AND a >2-live-leaf regression test proven to alias under the old heuristic | ✓ |
| Fix + rely on corpus | Fix; trust the broadened corpus to catch regressions | |
| Fix only | Free-slot-scan fix, no new coverage | |

**User's choice:** Fix + dedicated repro test.

---

## ROCm hardware pass

| Option | Description | Selected |
|--------|-------------|----------|
| cubecl-cpu gate + ROCm smoke | Deterministic cubecl-cpu is the gate; ROCm run is best-effort, pinned to cpu anchor, non-blocking | ✓ |
| ROCm parity in-phase | Real-ROCm parity pass required within Phase 21 | |
| Defer ROCm to Phase 23 | No ROCm work in 21 at all | |

**User's choice:** cubecl-cpu gate + ROCm smoke.
**Notes:** Local GPU is a spoofed 8-CU APU; f32 non-determinism (def-f8u-01) —
full hardware validation is Phase 23's Kaggle DoD.

---

## Requirement bookkeeping + ROADMAP re-cut

| Option | Description | Selected |
|--------|-------------|----------|
| Mark done + new hardening ID | Mark ODL-18/19 Complete (Phase 20); add ODL-18H for corpus+WR-01; re-cut ROADMAP body via /gsd-phase | ✓ |
| Mark done + fold, no new ID | Mark ODL-18/19 done; hardening as slack under those IDs | |
| Keep 18/19 on Phase 21 | Leave ODL-18/19 on Phase 21; hardening completes them | |

**User's choice:** Mark done + new hardening ID.

---

## Claude's Discretion

- Exact fixture parameters for the targeted corpus (smallest configs that
  provably yield >2 live leaves and trigger each constraint/edge branch).
- Whether the WR-01 repro is a lgbm-compute unit test or an oracle-harness case.

## Deferred Ideas

- Wire on-device EvalKernel metric eval into GBDT (Phase-20 follow-up).
- Full parity sweep — rejected under D-02 in favor of targeted cases.
- Five GPU perf-campaign todos routed to Phase 23 (perf DoD).
