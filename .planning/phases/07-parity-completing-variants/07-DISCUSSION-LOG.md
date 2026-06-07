# Phase 7: Parity-Completing Variants - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-07
**Phase:** 7-parity-completing-variants
**Areas discussed:** Decomposition / sequencing, Oracle matrix scale, Deferred knife-edges, Categorical learner re-open

---

## Gray-area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Decomposition / sequencing | How to slice 18 reqs into waves; whether to stay one phase or split | ✓ |
| Oracle matrix scale | Per-feature focused goldens vs Phase-6-style exhaustive crossing | ✓ |
| Deferred knife-edges | Fix bagged-subset split-gain knife-edge (DEF-06-01 + L1+bagging) vs carry forward | ✓ |
| Categorical learner re-open | Bound the TRL-06 re-open of the bit-exact Phase-5 learner | ✓ |

**User's choice:** All four areas.

---

## Decomposition / sequencing

### Phase shape

| Option | Description | Selected |
|--------|-------------|----------|
| Keep one phase, many waves | Stay Phase 7, long sequence of small dependency-ordered waves, one end gate | ✓ |
| Split into sub-phases now | /gsd-phase split into 7.1/7.2/…, each independently discussed/planned/verified | |
| One phase, grouped wave-bands | Stay Phase 7 with named wave bands + checkpoint/partial-verify per band | |

**User's choice:** Keep one phase, many waves.

### Wave order

| Option | Description | Selected |
|--------|-------------|----------|
| Dependency-forced, low-risk first | Spine-first ethos: objectives/metrics → variants → categorical → ranking → predict → advanced | ✓ |
| Highest-value / riskiest first | Front-load categorical/ranking/SHAP so divergence surfaces early | |
| Let researcher decide the DAG | Hand the 6 groups + dependencies to RESEARCH.md / planner | |

**User's choice:** Dependency-forced, low-risk first.
**Notes:** → D-01/D-02/D-03. The bagged-subset determinism wave (D-05) was folded in as the gating wave 1.

---

## Oracle matrix scale

### Matrix philosophy

| Option | Description | Selected |
|--------|-------------|----------|
| Per-feature focused goldens | One golden per feature exercising its own path; no crossing | |
| Targeted crossing on risk axes | Focused by default, cross-product only known knife-edge axes | |
| Full cross-product (Phase-6 ethos) | Cross every feature against bagging/ES/bfa; maximal coverage | ✓ |

**User's choice:** Full cross-product (Phase-6 ethos).

### Cross-product scope (where axes aren't all meaningful)

| Option | Description | Selected |
|--------|-------------|----------|
| Per-subsystem relevant axes | Full cross-product over the axes that actually affect each subsystem's output; no meaningless cells | ✓ |
| Literal full crossing everywhere | Cross every feature against {bagging×ES×bfa} uniformly even where redundant | |
| Let researcher define axes per group | RESEARCH.md enumerates the meaningful axis set per subsystem | |

**User's choice:** Per-subsystem relevant axes.
**Notes:** → D-04. Exhaustive Phase-6 ethos retained, refined to exclude provably-meaningless cells (e.g. SHAP/importance don't cross against bfa).

---

## Deferred knife-edges

| Option | Description | Selected |
|--------|-------------|----------|
| Investigate as an early wave | Dedicated early diagnostic wave; gates GOSS/RF + L1+bagging; fix-or-document-with-cap | ✓ |
| Keep rejected, build variants around it | Leave L1+bagging typed-rejected + DEF-06-01 documented; same tolerance posture for GOSS/RF | |
| Hard requirement: must be bit-exact | Treat any bagged-subset structural divergence as a blocker, no tolerance | |

**User's choice:** Investigate as an early wave.
**Notes:** → D-05. Outcome branches: faithful fix → un-defer regression_l1+bagging + clear DEF-06-01; genuine f32 artifact → bounded documented divergence with hard `struct_divergent <= 1` cap. Decided before bagging-dependent variants build on it.

---

## Categorical learner re-open

| Option | Description | Selected |
|--------|-------------|----------|
| Additive branch + spine re-validate | New categorical branch alongside numeric path; numeric spine stays byte-untouched + bit-exact; own categorical corpus | ✓ |
| Treat as its own keystone sub-effort | Scope categorical as large as Phase 5; dedicated multi-wave battery | |
| Let researcher scope the re-open | Hand TRL-06 + the bit-exact invariant to RESEARCH.md | |

**User's choice:** Additive branch + spine re-validate.
**Notes:** → D-06/D-07. Mirrors 05-01's additive boundary re-open. HARD INVARIANT: existing numeric-spine real-binary goldens must still pass bit-exact. Categorical gets its own real lib_lightgbm 4.6 corpus (reusing Phase-2 bit-exact categorical binning) + per-split bitset/gain diagnostics + model-text round-trip.

---

## Wrap-up

| Option | Description | Selected |
|--------|-------------|----------|
| Write CONTEXT.md now | Big levers decided; remaining items covered by carried discipline | ✓ |
| Explore stochastic-variant RNG parity | GOSS/DART dedicated RNG-replay goldens à la D-13 | |
| Explore ranking + refit scope | Ranking-stack grouping + refit/continue-training boundary | |

**User's choice:** Write CONTEXT.md now.

## Claude's Discretion

- Exact wave DAG / plan boundaries within the 6 groups + early determinism wave (researcher proposes, plan-checker verifies).
- Precise per-subsystem axis enumeration for the full cross-product.
- Crate placement for new variants/objectives/metrics (extend existing crates vs new modules), bounded by the factory seams.
- Whether GOSS sampling + DART drop-selection each get a dedicated RNG-replay golden (strongly recommended by carried D-13 discipline; exact shape is researcher's call).
- Ranking-stack internal grouping (lambdarank/rank_xendcg/DCGCalculator/ndcg/map/bagging_by_query), bounded by "shared query infra lands together."
- Refit/continue-training (ADV-06) boundary and which Phase-3 model-I/O hooks it reuses.
- Whether any Phase-7 subsystem warrants a ROCm cross-check vs CPU-bit-exact-only.

## Deferred Ideas

- Python/PyO3 bindings — Phase 8.
- Parallel (rayon) CPU / multi-GPU boosting path — post-MVP optimization.
- ROCm cross-check of Phase-7 subsystems — research/planning call; CPU bit-exact is the hard gate.
- Out-of-milestone subsystems (distributed/network learners, linear-tree leaves, GPU tree learner) — not in v1.0.
