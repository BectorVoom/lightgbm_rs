# Phase 3: Tree Model + Model Text I/O + Predict Parity - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 delivers the ability to **load a C++-trained LightGBM `.txt` model into a faithful in-memory Tree/GBDT model, predict from it identically, and write the model back out in the exact text schema** — prediction parity proven independently of (and *before*) any training code exists. The numerical contract is the project default: f32 end-to-end, ~1e-6 absolute on the deterministic CPU path; the model text itself is compared **bit/byte-exact**.

In scope:
- **DAT-08** — model text **read**: parse a C++-trained `.txt` model (`GBDT::LoadModelFromString` family) into the in-memory model.
- **DAT-09** — model text **write**: emit the exact LightGBM text schema (trees, leaf values, `feature_infos`, `tree_sizes`, etc.) including `%.17g` float formatting, **byte-identical to C++ `SaveModelToString`**.
- **PRD-01** — raw-score prediction (sum of tree outputs) within ~1e-6 (f32) of C++ on the deterministic CPU path.
- **PRD-02** — transformed prediction (`ConvertOutput` sigmoid for binary / softmax for multiclass).
- **PRD-03** — leaf-index prediction (`pred_leaf`).
- **PRD-06** — sub-range prediction (`start_iteration` / `num_iteration` slice of the ensemble).
- Prediction on **dense + CSR/CSC** input matrices (reusing the Phase-2 ingest forms).

Out of scope: any **training / tree-learning** code (Phases 5–6); the histogram **compute backend** / CubeCL kernels (Phase 4); SHAP / `predict_contrib` and prediction early-stop (PRD-04/PRD-05, Phase 7); **single-row** prediction plumbing and the Python predict surface (Phase 8); non-core objectives' `ConvertOutput` (huber/poisson/ranking/etc., Phase 7); `DumpModel` JSON dump and `ModelToIfElse` C++ codegen (not a v1 requirement); feature-importance reporting (ADV-07, Phase 7).

</domain>

<decisions>
## Implementation Decisions

### Writer Fidelity Contract (DAT-09 / SC#3)
- **D-01:** The Rust model-text writer must be **byte-identical to C++ `GBDT::SaveModelToString`** for the same model. The golden is a C++-written `.txt`; the Rust writer must reproduce it byte-for-byte — key ordering, whitespace/newlines, `tree_sizes=` block, `feature_infos=`, and `%.17g` float formatting all match exactly. This is the strongest contract (SC#3 "exact text schema") and makes the load→predict→write→reload round-trip **trivially byte-stable**, consistent with the faithful-mirror discipline carried from Phase 2. (Chosen over a weaker "self-consistent round-trip only" contract, which would let a serialization divergence — key order/whitespace — hide behind "it still loads.")
- **D-01a:** Because the write contract is byte-exact, the SC#3 "bin mappers / feature metadata" wording is subsumed: the writer emits **whatever C++ emits** (notably `feature_infos=` per-feature min/max-range metadata — NOT full `BinMapper` bin-boundary arrays, which are not part of the predict model text). Research must confirm the exact set of emitted sections against `gbdt_model_text.cpp`; the byte-identical golden is the arbiter.

### Prediction Input Surface (PRD-01/02/03)
- **D-02:** Phase 3 predicts on **dense + CSR/CSC** input matrices, reusing the Phase-2 ingest forms. This maximizes parity coverage at low marginal cost (those input fixtures already exist and bin bit-identically). **Single-row** prediction (the C++ `Predictor` per-row `PredictFunction` path) is **deferred to Phase 8** (Python-binding convenience); SHAP/early-stop prediction modes are Phase 7.
- **D-02a:** Prediction runs on **raw feature values** using the model's stored real thresholds / categorical bitsets (the loaded-model path), NOT by re-binning through a `BinMapper`. The deterministic CPU path is **single-threaded**, matching the pinned `num_threads=1` reference.

### Crate Boundary
- **D-03:** **One new `lgbm-model` crate** holds the entire subsystem: the `Tree` + GBDT-ensemble model representation, model-text **load/save**, and the **predictor**. Echoes Phase 2 D-04 single-crate cohesion and the Phase 1 "crate per subsystem that introduces it" rule (`lgbm-*` naming). Prediction and model representation are tightly coupled (predict reads the tree arrays intimately), so splitting repr/serde from predict now is premature; the separate training/boosting crate arrives in Phase 6. (Chosen over a repr/serde-vs-predict split or extending `lgbm-dataset`.)

### Tree Model Representation (defaulted — faithful-mirror precedent)
- **D-04:** The in-memory `Tree` is a **faithful 1:1 mirror of C++ `tree.h`**: parallel arrays (`split_feature_`, `threshold_`, `decision_type_`, `left_child_`, `right_child_`, `leaf_value_`, `leaf_count_`, `internal_value_`, `cat_boundaries_`/`cat_threshold_`, …) rather than an idiomatic Rust node enum. Same rationale as Phase 2 D-01/D-02: the byte layout and traversal order are part of the parity surface, and the model text maps directly onto these arrays. The GBDT ensemble mirrors `models_` (a flat tree list with per-iteration/per-class indexing).

### Predict-Parity Fixture Corpus (DAT-08/09, PRD-01/02/03/06, ORA-03)
- **D-05:** Committed golden corpus covers **regression + binary + multiclass + categorical-split + sub-range** C++-trained models — the minimal set that exercises every Phase-3 prediction path: raw-score accumulation (PRD-01), sigmoid (binary) and softmax + **per-class / per-iteration tree indexing** (multiclass) `ConvertOutput` (PRD-02), `pred_leaf` (PRD-03), the **categorical-split Tree decision** path, and `start_iteration`/`num_iteration` slicing (PRD-06). Scope-bounded to **core** objectives only — Phase-7 objectives (huber/poisson/ranking/etc.) and their `ConvertOutput` parity belong in Phase 7.
- **D-06:** **Layered golden granularity per fixture** (mirrors Phase 2 D-07's maximally-diagnostic discipline): (1) **model-text round-trip bytes** — C++-written `.txt` vs Rust-written `.txt`, byte-exact (DAT-09); (2) **raw score** vector, f32 ~1e-6 (PRD-01); (3) **transformed** output, f32 ~1e-6 (PRD-02); (4) **leaf-index** vector, exact integer (PRD-03); (5) **sub-range** raw scores for representative `start_iteration`/`num_iteration` (PRD-06). A mismatch localizes the divergence to a specific stage (parse vs predict-math vs convert vs serialize).
- **D-07:** Fixtures are produced by **C++ training a model and emitting its `.txt`** via the golden-capture xtask pipeline (Phase 1/2 pattern): generate once from the in-repo C++ build, **commit**, replay with no C++ toolchain at normal test time. If `bin.cpp`/full-lib linkage remains infeasible (`external_libs` unvendored — see Phase 2 precedent), fall back to **header-only / verbatim transcription** capture, human-approved, numerically identical to `lib_lightgbm`. Reuse the Phase-2 fixture inputs (dense/CSR/CSC matrices, example datasets) as the prediction inputs.

### Claude's Discretion
- Exact `lgbm-model` internal module layout; the precise `Tree`/ensemble field set and accessor shape (bounded by "faithful `tree.h` mirror"); the model-text parser's tokenizer strategy (bounded by "byte-identical writer + exact-parse round-trip"); golden file formats/serialization; the predictor's dense-vs-sparse iteration structure; how `decision_type_` bit flags (default-left, categorical, missing-type) are decoded — all left to research/planning, bounded by "faithful C++ mirror, ~1e-6 f32 scores, byte-exact model text." When C++ behavior is the spec, the C++ source (below) is authoritative over any inferred default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ reference source (read-only port target — authoritative for all Phase-3 behavior)
- `LightGBM/include/LightGBM/tree.h` — `Tree` class: node arrays (`split_feature_`, `threshold_`, `decision_type_`, `left_child_`/`right_child_`, `leaf_value_`, `internal_value_`, `cat_boundaries_`/`cat_threshold_`), `Tree::Predict`/`PredictByMap`, `GetLeaf`, `DecisionType` bit-flag helpers, `Tree(const char* str, size_t* used_len)` construct-from-string — the D-04 mirror target.
- `LightGBM/src/io/tree.cpp` — `Tree::ToString` (per-tree text emit), `Tree` string-parse constructor, numerical/categorical decision logic, `%.17g` leaf/threshold formatting — authoritative for both predict-math and per-tree serialization.
- `LightGBM/src/boosting/gbdt_model_text.cpp` — `GBDT::SaveModelToString` / `LoadModelFromString` (the **byte-exact** D-01 target): `feature_infos=`, `tree_sizes=` boundaries, header key ordering, `pandas_categorical`, per-tree blocks; `GBDT::FeatureImportance` (not in Phase-3 scope but co-located).
- `LightGBM/src/boosting/gbdt_prediction.cpp` — `GBDT::PredictRaw` / `Predict` / `PredictLeafIndex` and the `start_iteration`/`num_iteration` sub-range slicing (PRD-01/02/03/06); `num_tree_per_iteration_` per-class indexing for multiclass.
- `LightGBM/include/LightGBM/boosting.h` — `PredictRaw`, `Predict`, `PredictLeafIndex`, `NumPredictOneRow`, `SaveModelToString`, `SaveModelToFile` signatures (`start_iteration`/`num_iteration` semantics; `-1 == all`).
- `LightGBM/src/application/predictor.hpp` — `Predictor` dense/sparse predict driver and `PredictFunction` shape (D-02 batch-matrix path; single-row path is the deferred portion).
- `LightGBM/src/objective/*.hpp` — `ConvertOutput` for the **core** objectives only (regression identity, binary sigmoid, multiclass softmax, multiclassova) — the PRD-02 transform (grad/hess are NOT in scope; only the output transform).

### Foundations to build on (Phase 1 + Phase 2 deliverables)
- `crates/lgbm-dataset/` — the binned columnar store, `BinMapper`, `Metadata`, and `from_mat`/`from_csr`/`from_csc` ingestion; Phase-3 prediction inputs reuse these forms. `lgbm-model` depends on `lgbm-dataset` (and transitively `lgbm-core`).
- `crates/lgbm-core/src/config/` — `Config` (objective, `num_class`, `start_iteration`, prediction-relevant params already modeled); `src/types.rs` (f32 types), `src/error.rs` (`thiserror` boundary-error idiom to extend into `lgbm-model`).
- `crates/oracle-harness/` — the comparator + committed-golden + idempotent-regen seam; `compare_exact_bytes` (model-text byte parity), `compare_exact_u32` (leaf-index), and the f32 ~1e-6 comparator (scores). Extend `REFERENCE_MANIFEST.md` for the model/predict fixtures.
- `xtask` `bin-capture` pattern + `xtask/cpp/` — the C++ golden-capture harness to extend with a model/predict capture subcommand (D-07).

### Project-level contract
- `.planning/PROJECT.md` — Core Value, Constraints, Key Decisions (f32/~1e-6; standard f32 accumulations).
- `.planning/REQUIREMENTS.md` — DAT-08, DAT-09, PRD-01, PRD-02, PRD-03, PRD-06 (Phase 3 requirements).
- `.planning/ROADMAP.md` §"Phase 3" — goal + 4 success criteria.
- `.planning/phases/02-dataset-binning-determinism-root/02-CONTEXT.md` — Phase 2 decisions carried forward (D-01 faithful-mirror, D-03 single-threaded determinism, D-04 single-crate cohesion, D-06/D-07 golden discipline).
- `.planning/phases/01-oracle-contract-foundations/01-CONTEXT.md` — Phase 1 crate-naming (`lgbm-*`) + golden-capture / idempotent-regen patterns.

### Codebase maps (reference C++ architecture)
- `.planning/codebase/STRUCTURE.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/ARCHITECTURE.md` — C++ layout, model-text conventions, prediction data flow.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `lgbm-dataset` `from_mat`/`from_csr`/`from_csc` + `Metadata` — the dense/CSR/CSC prediction inputs for Phase 3 reuse the Phase-2 ingest forms (already bit-identical to C++).
- `lgbm-core::Config` — `objective`, `num_class`, `start_iteration`, and prediction params already modeled and validated; `lgbm-model` reads these, doesn't redefine them.
- `oracle-harness` `compare_exact_bytes` / `compare_exact_u32` / f32-~1e-6 comparator — the model-text byte parity, leaf-index integer parity, and score ~1e-6 parity all plug into the existing comparator seam.
- `xtask` C++ golden-capture pipeline (`bin-capture` subcommand + `xtask/cpp/`) — extend with a model/predict capture subcommand; committed-golden + idempotent-regen discipline carries over verbatim.

### Established Patterns
- Faithful 1:1 C++ hand-port, flat/array structs, guarded by a parity test (Phase 1 D-11/D-12, Phase 2 D-01) — applies to `Tree`, the GBDT ensemble, and the model-text parser/writer.
- Committed fixtures + idempotent C++-regen; no C++ toolchain at normal test time; **header-only / verbatim transcription** capture fallback when `external_libs` are unvendored (Phase 1/2 precedent — likely needed again here).
- Bit/byte-exact comparison for discrete artifacts (model text, leaf indices) vs the ~1e-6 f32 oracle tolerance for continuous scores — same split Phase 2 used (bytes vs scores).
- Single-threaded deterministic core matching the pinned `deterministic=true force_row_wise=true num_threads=1` reference; per-row independence is the parallel-ready seam (Phase 2 D-03).
- C++ constants in play: `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`; `%.17g` float formatting is the serialization-parity linchpin.

### Integration Points
- New `crates/lgbm-model` depends on `lgbm-dataset` (predict inputs + metadata) and `lgbm-core` (Config/types/errors/RNG). Add it to the workspace `members` in the root `Cargo.toml` (currently lgbm-core, lgbm-compute, lgbm-dataset, oracle-harness, xtask).
- `lgbm-model` becomes the dependency root for Phase 6 (GBDT training emits these same `Tree`/ensemble structures and reuses the predictor for scoring) and Phase 8 (Python predict surface wraps this crate).
- The `LightGBM/` reference tree is **untracked** (never `git add` it); model fixtures must be C++-generated then **copied/committed** into `tests/fixtures/`, never referenced from the untracked tree at test time (see memory: lightgbm-ref-tree-untracked).

</code_context>

<specifics>
## Specific Ideas

- **Faithfulness over idiom, again:** every Phase-3 gray area resolved toward the closest C++ mirror — byte-identical model text (not merely round-trip-stable), faithful `tree.h` array layout (not an idiomatic node enum), single `lgbm-model` crate mirroring C++'s io/boosting/application cohesion. When in doubt, reproduce C++ behavior.
- **Byte-identical is the arbiter for serialization ambiguity:** rather than reverse-engineering exactly which metadata sections "count," the writer must reproduce the C++ `.txt` byte-for-byte — `%.17g`, key order, `tree_sizes`, `feature_infos` all fall out of that single contract.
- **Prediction before training is deliberate:** proving load→predict parity against a C++-trained model isolates the prediction-math + serialization surface from the (later, higher-risk) tree-learning surface, so any Phase-5/6 divergence can be ruled out of prediction.
- **Layered goldens stay maximally diagnostic:** separate model-text-bytes / raw-score / transformed / leaf-index / sub-range layers so a failure points at parse vs predict vs convert vs serialize, not just "prediction is off."

</specifics>

<deferred>
## Deferred Ideas

- **Single-row prediction** — the C++ `Predictor` per-row `PredictFunction` path; deferred to Phase 8 (Python-binding convenience). Phase 3 ships dense + CSR/CSC batch prediction only.
- **SHAP / `predict_contrib` (PRD-04)** and **prediction early stopping (PRD-05)** — Phase 7.
- **Non-core objective `ConvertOutput`** (huber/fair/poisson/quantile/mape/gamma/tweedie, cross-entropy, ranking) — Phase 7; Phase 3 covers only the core regression/binary/multiclass/multiclassova transforms.
- **`DumpModel` JSON dump + `ModelToIfElse` C++ codegen** — not v1 requirements; out of scope unless a later requirement reintroduces them.
- **Feature-importance reporting (ADV-07)** — co-located in `gbdt_model_text.cpp` but Phase 7.
- **Parallel (rayon) prediction** — Phase 3 ships single-threaded deterministic; per-row independence leaves the seam for a later, separately-validated parallel pass.

None other — discussion stayed within Phase 3 scope.

</deferred>

---

*Phase: 3-tree-model-model-text-i-o-predict-parity*
*Context gathered: 2026-06-05*
