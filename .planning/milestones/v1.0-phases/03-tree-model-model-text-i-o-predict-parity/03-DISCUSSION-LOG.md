# Phase 3: Tree Model + Model Text I/O + Predict Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 3-tree-model-model-text-i-o-predict-parity
**Areas discussed:** Writer fidelity contract, Prediction input surface, Model/predict crate boundary, Predict-parity fixture corpus

---

## Writer Fidelity Contract (DAT-09 / SC#3)

| Option | Description | Selected |
|--------|-------------|----------|
| Byte-identical to C++ | Golden = a C++ SaveModelToString .txt; Rust writer reproduces it byte-for-byte (key order, whitespace, tree_sizes, %.17g, feature_infos). Strongest contract; round-trip trivially byte-stable. | ✓ |
| Self-consistent round-trip only | Rust write→read stable + schema-valid + %.17g + predicts identically, but bytes may differ from C++. | |

**User's choice:** Byte-identical to C++
**Notes:** Aligns with SC#3 "exact text schema" and the faithful-mirror discipline. Subsumes the SC#3 "bin mappers / feature metadata" wording — the writer emits whatever C++ emits (feature_infos min/max, not full BinMapper arrays); the byte-identical golden is the arbiter.

---

## Prediction Input Surface (PRD-01/02/03)

| Option | Description | Selected |
|--------|-------------|----------|
| Dense + CSR/CSC | Reuse Phase-2 ingest forms as predict inputs; max parity coverage, low marginal cost; no single-row plumbing. | ✓ |
| Dense + CSR/CSC + single-row | Also mirror the C++ Predictor single-row path. Fuller mirror, but single-row is mainly a Python convenience (Phase 8). | |
| Dense only | Dense matrices only; defer sparse predict. Leanest, leaves a parity gap. | |

**User's choice:** Dense + CSR/CSC
**Notes:** Single-row deferred to Phase 8. Prediction runs on raw feature values via stored model thresholds, single-threaded deterministic.

---

## Model/Predict Crate Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| One lgbm-model crate | Tree + GBDT-model repr + text load/save + predictor together. Echoes Phase 2 D-04 cohesion + "crate per subsystem" rule. | ✓ |
| Split repr/serde vs predict | lgbm-model (repr + text I/O) separate from a prediction crate. More crates, premature looseness. | |
| Extend lgbm-dataset | Put model + predict in the existing dataset crate. Rejected by precedent. | |

**User's choice:** One lgbm-model crate
**Notes:** Prediction and model-repr are tightly coupled; separate training/boosting crate arrives in Phase 6. lgbm-model depends on lgbm-dataset + lgbm-core.

---

## Predict-Parity Fixture Corpus

| Option | Description | Selected |
|--------|-------------|----------|
| regression + binary + multiclass + categorical + sub-range | Core coverage exercising every predict path: raw score, sigmoid + softmax/per-class indexing, pred_leaf, categorical-split decision, start/num_iteration slices. Layered goldens (raw / transformed / leaf-index / round-trip bytes). | ✓ |
| regression + binary only | Minimal; skips multiclass + categorical-split prediction. | |
| Add all Phase-7 objectives too | Train huber/poisson/ranking/etc. fixtures now. Scope creep — belongs in Phase 7. | |

**User's choice:** regression + binary + multiclass + categorical + sub-range
**Notes:** Core objectives only; non-core objective ConvertOutput parity stays in Phase 7. Fixtures C++-generated then committed; header-only/verbatim-transcription capture fallback if external_libs remain unvendored.

---

## Claude's Discretion

- Tree representation defaulted to the faithful C++ `tree.h` array-mirror (not an idiomatic node enum), per Phase-2 precedent — confirmed without a separate question.
- lgbm-model internal module layout, the Tree/ensemble field set, the model-text parser tokenizer strategy, golden file formats, decision_type bit-flag decoding — left to research/planning, bounded by "faithful C++ mirror, ~1e-6 f32 scores, byte-exact model text."

## Deferred Ideas

- Single-row prediction → Phase 8.
- SHAP / predict_contrib (PRD-04) + prediction early stopping (PRD-05) → Phase 7.
- Non-core objective ConvertOutput (huber/poisson/ranking/etc.) → Phase 7.
- DumpModel JSON + ModelToIfElse C++ codegen → out of v1 scope.
- Feature-importance reporting (ADV-07) → Phase 7.
- Parallel (rayon) prediction → later, separately-validated optimization.
