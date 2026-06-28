# Phase 14: Scaffold + Oracle (Slice 0) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-28
**Phase:** 14-scaffold-oracle-slice-0
**Areas discussed:** Oracle behavior, Seam contract, Tie-aware comparator timing, Env gate placement

---

## Oracle behavior (no kernel yet)

| Option | Description | Selected |
|--------|-------------|----------|
| Live via host fallback | Route the no-op seam to fall back to the host learner so the oracle runs GREEN on a real tree now — exercises comparator + plumbing before any kernel. | ✓ |
| Dormant scaffold | Comparator compiles, test `#[ignore]`/env-gated, no live assertion until Slice 1. | |
| Assert fallback contract | Oracle asserts the routing invariant only (env off ⇒ on-device never taken ⇒ byte-identical). | |

**User's choice:** Live via host fallback
**Notes:** Reconciled with the seam contract — the host-fallback `unwrap_or_else` lives in the ORACLE TEST as a stand-in for the not-yet-existent on-device tree; the production fork stays `Ok(None) ⇒ host` and byte-unchanged.

---

## Seam contract (`grow_tree_on_device` return type)

| Option | Description | Selected |
|--------|-------------|----------|
| `Result<Option<...>>` | `Ok(None)` when not handled ⇒ train_inner falls through; no error noise on default path. | ✓ |
| Typed `Err(NotSupported)` | Matches roadmap's literal "typed error/no-op" wording; fork matches on the error. | |
| Gate before call | Discriminator decides before calling; method only called when it will grow. | |

**User's choice:** `Result<Option<(Tree, DataPartition)>>`, default `Ok(None)`
**Notes:** Cleanest composition with the decide-once fork; `None` = "I didn't grow it".

---

## Tie-aware `default_left` comparator timing

| Option | Description | Selected |
|--------|-------------|----------|
| Ship comparator now (dormant) | Build the tie-aware comparator in Slice 0 (reuse kernel_parity.rs:1597 near-tie logic), dormant until a kernel exists; Phase 16 just activates it. | ✓ |
| Structure-only now, tie-aware in 16 | Slice 0 reuses existing structure-bit-exact assert; Phase 16 adds the tie-aware branch with the selection kernel. | |

**User's choice:** Ship comparator now (dormant)
**Notes:** Satisfies Phase 14 SC#3 ("tie-aware scaffold") and Phase 16's "do NOT defer the tie-aware assert" at once — the assert exists from Slice 0, goes live in Slice 2.

---

## Env gate read placement (`LGBM_CUDA_ON_DEVICE`)

| Option | Description | Selected |
|--------|-------------|----------|
| Decide-once in train_inner | Mirror `resident_eligible` exactly: compute `supported() && env` at the top of train_inner each train. | |
| Once at construction | Read env once in `SerialTreeLearner::new` and cache; avoids per-train syscalls; diverges from resident idiom. | ✓ |

**User's choice:** Once at construction
**Notes:** Intentional divergence from `resident_eligible` (which size-gates per-train on `num_data`); on-device eligibility has no per-train input, so a per-tree env re-read buys nothing. Still ANDs the backend discriminator. Flagged so the planner doesn't normalize it back into train_inner.

## Claude's Discretion

- Cached field name (`on_device_eligible` suggested), env-parse helper, comparator file location, and whether the comparator is a new fn or a tie-aware extension of `assert_gpu_tree_matches_cpu_anchor`.

## Deferred Ideas

- None raised outside phase scope — all deferrals are the planned downstream slices (Phases 15–19), already tracked in ROADMAP/REQUIREMENTS.
