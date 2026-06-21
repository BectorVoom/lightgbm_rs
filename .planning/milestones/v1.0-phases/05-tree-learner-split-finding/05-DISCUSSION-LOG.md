# Phase 5: Tree Learner + Split Finding - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-06
**Phase:** 5-tree-learner-split-finding
**Areas discussed:** Reference-tree capture strategy, Gradient/hessian fixture source, Phase slice/sequencing, Histogram-pool fidelity, Transcription scope, Per-split snapshot granularity, Tree-match unit

---

## Reference-tree capture strategy (Golden src)

| Option | Description | Selected |
|--------|-------------|----------|
| Extend header-only transcription | Verbatim-transcribe serial_tree_learner orchestration into xtask, emit per-split + per-tree goldens, commit. P1–P4 precedent; no C++ toolchain at test time. | ✓ |
| Vendor external_libs + build real lib_lightgbm | Build the genuine C++ learner, capture from the real binary. Highest fidelity but breaks untracked-LightGBM/worktree posture. | |
| Hybrid: transcribe scan, validate vs real build | Transcribe for committed goldens, one-time real-build cross-check, then rely on committed goldens. | |

**User's choice:** Extend header-only transcription
**Notes:** Continues the established no-toolchain-at-test-time discipline; `LightGBM/` stays untracked.

---

## Gradient/hessian fixture source (G/H input)

| Option | Description | Selected |
|--------|-------------|----------|
| Both: synthetic + captured first-iter g/h | Layer hand-crafted edge-case g/h AND g/h captured from a real C++ objective's iteration-1 on a real dataset. Maximally diagnostic. | ✓ |
| Synthetic deterministic g/h only | Hand-crafted vectors exercising every split path; self-contained but less representative. | |
| Captured real first-iteration g/h only | Most realistic but couples validation to objective capture; may miss synthetic edge cases. | |

**User's choice:** Both, layered
**Notes:** Captured-g/h objective/dataset/`boost_from_average` config left to researcher's discretion.

---

## Phase slice / sequencing (Slice)

| Option | Description | Selected |
|--------|-------------|----------|
| Spine first, then parity additions | Lock minimal faithful tree (row_wise, feature_fraction=1.0, numeric, leaf-wise, subtraction, partition) first; add col_wise (TRL-09) + feature-subsampling (TRL-08) on top. | ✓ |
| All TRL-01..09 together | Build full learner incl. col_wise + subsampling in one pass before validating. | |

**User's choice:** Spine first, then parity additions

---

## Histogram-pool fidelity (Hist pool)

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror FP-load-bearing parts only | Reproduce smaller-child selection / construct-vs-subtract / parent retention / subtract math, but simpler single-threaded allocation instead of the C++ pool. | |
| Mirror the full histogram pool + eviction | Port HistogramPool sizing/eviction/reuse faithfully too. Higher fidelity, more code, likely no parity difference. | ✓ |
| You decide | Defer the bar to research/planning. | |

**User's choice:** Mirror the full histogram pool + eviction
**Notes:** Strongest-fidelity bet — removes any risk a pool-ordering effect is observable in the FP path.

---

## Transcription scope (Transcr scope)

| Option | Description | Selected |
|--------|-------------|----------|
| Orchestration-only, reuse P4 kernel transcriptions | Transcribe only the learner orchestration; reuse Phase-4 kernel transcriptions for per-feature math. Smallest faithful surface. | |
| Full re-transcription of serial_tree_learner end-to-end | Re-transcribe everything incl. per-feature histogram + gain scan, independent of P4. Redundant cross-check, more code/drift risk. | ✓ |

**User's choice:** Full re-transcription end-to-end
**Notes:** The overlap with the Phase-4 kernel transcription becomes an intentional cross-check (CONTEXT D-02a: a guard must surface any drift between the two transcriptions).

---

## Per-split snapshot granularity (Snapshot)

| Option | Description | Selected |
|--------|-------------|----------|
| Per-feature best gain at each split decision | Snapshot each candidate feature's best gain/threshold/direction; verifies global argmax + tie-break. | |
| Full per-bin gain array for every candidate feature | Snapshot the entire bin-by-bin gain scan for every feature at every split. Maximally diagnostic, large goldens. | ✓ |
| Both, layered | Per-feature-best for every split + full per-bin for representative splits. | |

**User's choice:** Full per-bin gain array for every candidate feature

---

## Tree-match unit (Tree match)

| Option | Description | Selected |
|--------|-------------|----------|
| Full tree structure, bit-faithful | Assert every node's feature/threshold/direction AND every leaf's output value via Phase-3 %.17g machinery. | ✓ |
| Split decisions only | Assert split structure only; defer leaf-output parity to Phase 6. | |

**User's choice:** Full tree structure, bit-faithful
**Notes:** Leaf-output parity validated here because those values feed Phase-6 scores.

---

## Claude's Discretion

- Tree-learner crate placement/structure and learner↔`Backend` wiring.
- Leaf-wise priority-queue + leaf-split bookkeeping shape.
- Captured-g/h objective(s)/dataset/`boost_from_average` config (D-03).
- `force_col_wise=true`/`force_row_wise=true` capture configs (D-04).
- Leaf-wise queue tie-break determinism mechanism (bounded by "must match C++ selection order").
- Golden serialization/layering format (bounded by oracle-harness comparator + Phase-3 `%.17g`).

## Deferred Ideas

- TRL-06 categorical splits → Phase 7.
- GBDT spine / objectives / metrics / bagging / early stopping → Phase 6.
- DART / RF / GOSS → Phase 7.
- Monotone/interaction constraints, forced splits, extra-trees, CEGB, refit → Phase 7.
- Parallel (rayon/multi-GPU) histogram-build path → post-MVP optimization.
- ROCm cross-check of the orchestrated learner → research/planning call (CPU bit-exact is the hard gate).
