---
phase: 03-tree-model-model-text-i-o-predict-parity
plan: 02
subsystem: model
tags: [lgbm-model, tree, ensemble, model-text, predict, DAT-08, DAT-09, PRD-01]
requires:
  - lgbm-model::format (format_g17/format_g6 — 03-01)
  - lgbm-model::error (ModelError — 03-01)
  - lgbm-core::types (K_ZERO_THRESHOLD)
  - oracle-harness (compare_within / compare_exact_bytes / ORACLE_TOL)
  - committed regression golden corpus (03-01)
provides:
  - "lgbm-model::tree::Tree — faithful parallel-array tree + predict/get_leaf + ToString + parse"
  - "lgbm-model::ensemble::GbdtModel — flat tree list + ntpi stride + predict_raw + init_predict + feature_importance_split_count"
  - "lgbm-model::model_text::{load, save} — byte-exact LoadModelFromString/SaveModelToString envelope"
  - "lgbm-model::predict::{predict_raw_mat, predict_raw_csr, predict_raw_csc} — raw-value driver (D-02a)"
  - "lgbm-model tests/golden/mod.rs shared golden loaders"
affects:
  - "03-03 (binary/multiclass/categorical-split predict + transform layers reuse Tree/GbdtModel/predict)"
  - "03-04 (sub-range parity reuses init_predict/predict_raw start/num args)"
tech-stack:
  added: []
  patterns:
    - "faithful 1:1 C++ parallel-array Tree (D-04), trailing-C++-name fields, #[inline] one-to-one decode helpers"
    - "keyed order-independent model-text parse + verbatim metadata/trailer preservation on round-trip"
    - "tree_sizes byte-boundary slicing with checked usize arithmetic"
    - "raw-value predict (no re-binning), f64 accumulate, f32 cast only at output boundary"
key-files:
  created:
    - crates/lgbm-model/src/tree.rs
    - crates/lgbm-model/src/ensemble.rs
    - crates/lgbm-model/src/model_text.rs
    - crates/lgbm-model/src/predict.rs
    - crates/lgbm-model/tests/golden/mod.rs
    - crates/lgbm-model/tests/model_text_roundtrip.rs
    - crates/lgbm-model/tests/predict_raw_parity.rs
  modified:
    - crates/lgbm-model/src/lib.rs
decisions:
  - "feature_importances recomputed via split-count over split_feature (matches committed golden: 25->50, 27->38, 5->31)"
  - "parameters:..EOF tail (incl. Python pandas_categorical:null) captured + re-emitted VERBATIM, not reconstructed"
  - "Tree parser STRICTER than C++: validates every array length + child/leaf index range before indexing (T-03-03), never panics"
metrics:
  duration: ~7 min
  completed: 2026-06-05
  tasks: 3
  files: 8
---

# Phase 3 Plan 02: Regression load → predict → write → reload vertical slice Summary

First end-to-end Phase-3 slice: load a C++-trained **regression** `model.txt` into a
faithful array-based `GbdtModel`, predict raw scores identically on dense + CSR + CSC inputs
(~1e-6, f64-accumulated), and write the model back **byte-identically** to the committed C++
`.txt`. Delivers DAT-08 (read), DAT-09 (byte-exact write), PRD-01 (raw score) — D-06 layers
1 + 2 — proving the whole parse → Tree arrays → ensemble raw-predict → byte-exact write spine
on the simplest objective before binary/multiclass/categorical/sub-range layer on.

## What Was Built

- **`tree.rs` — faithful parallel-array `Tree`** (Task 1, TDD). 1:1 mirror of `tree.h`/
  `tree.cpp` (D-04): the full Pattern-1 field set (`split_feature`/`threshold`/
  `decision_type`/`left_child`/`right_child`/`leaf_value`/`leaf_weight`/`leaf_count`/
  `internal_*`/`cat_boundaries`/`cat_threshold`/`shrinkage`/`is_linear`). `#[inline]`
  one-to-one decode helpers (`get_decision_type`/`get_missing_type`/`is_zero` reusing
  `K_ZERO_THRESHOLD`/`find_in_bitset`). `numerical_decision` (NaN→0 coercion unless NaN-type,
  Zero/NaN default-left routing, `<=` threshold), `categorical_decision` (built; parity
  asserted in 03-03), `get_leaf` (`~node`), `predict`/`predict_leaf_index`. `to_string()`
  reproduces the exact `Tree::ToString` section order + per-field formatter mode
  (`format_g17` for threshold/leaf_value/leaf_weight; `format_g6` for split_gain/internal_*/
  shrinkage; `decision_type` as int8-cast-to-int). Keyed order-independent `parse()` with the
  single-leaf early return, per-field fallbacks, and **strict** array-length + node/leaf-index
  validation (T-03-03) — linear-tree models rejected. 13 inline tests incl. a real regression
  `Tree=0` block byte-exact round-trip.
- **`ensemble.rs` — `GbdtModel`** (Task 2). Flat `Vec<Tree>` + `num_class`/
  `num_tree_per_iteration` (the `i*ntpi+k` stride, documented) + `max_feature_idx`/
  `label_index`/`average_output` + verbatim metadata strings + the `parameters:`..EOF
  `trailer`. `init_predict` sub-range clamp (`gbdt.h:426-435`), `predict_raw` f64 accumulation
  (`gbdt_prediction.cpp:13-32`), `feature_importance_split_count` (`gbdt_model_text.cpp:627`).
- **`model_text.rs` — `load`/`save`** (Task 2). `load` ports `LoadModelFromString`: header
  key parse, `tree_sizes=` byte-boundary tree slicing (checked `usize`, T-03-04),
  `feature_names`/`feature_infos` count checks vs `max_feature_idx+1` (T-03-05), and the
  `parameters:`..EOF tail captured **verbatim** (incl. the Python `pandas_categorical:null`
  line that pip-`lightgbm` appends after the C++ writer output). `save` ports
  `SaveModelToString`: exact envelope order, per-tree `Tree::to_string`, `tree_sizes=`
  computed from each serialized tree's byte length (incl. trailing `\n`), **recomputed**
  `feature_importances:` (descending stable-sort, >0 only), verbatim trailer.
- **`predict.rs` — dense/CSR/CSC raw driver** (Task 3, D-02a). Mirrors C++ `Predictor`:
  materializes a dense `f64` row buffer of width `max_feature_idx+1` from raw caller input
  (sparse/absent == 0.0, NO re-binning through `BinMapper`), feeds `predict_raw`, casts to
  `f32` only at the output boundary. Validated entries (`num_cols >= max_feature_idx+1`,
  shape/index range) → `ModelError::ShapeMismatch`, never panic (T-03-07).
- **Tests** (Task 3). `tests/golden/mod.rs` shared loaders (f64-bit rows, svm-like input,
  dense→CSR/CSC). `model_text_roundtrip.rs` (layer 1, `compare_exact_bytes`) and
  `predict_raw_parity.rs` (layer 2, `compare_within(ORACLE_TOL)` on dense+CSR+CSC) — both
  resolve fixtures via `CARGO_MANIFEST_DIR` with graceful SKIP pre-capture.

## Key Decisions

- **`feature_importances` recomputed (split-count), not preserved.** C++ recomputes on every
  `SaveModelToString`; the loader stores nothing. Spot-verified the committed regression
  golden's counts equal split counts over `split_feature` (Column_25=50, Column_27=38,
  Column_5=31). Descending stable-sort, `>0` only — reproduces the golden byte-exactly.
- **Verbatim `parameters:`..EOF trailer (incl. `pandas_categorical:null`).** The committed
  golden is the pip-`lightgbm`-written file, which appends `\npandas_categorical:null\n`
  AFTER the C++ writer output. Capturing the `parameters:`-onward bytes verbatim (rather than
  reconstructing C++'s stripped `loaded_parameter_`) makes the round-trip byte-stable without
  porting `Config::ToString`. Aligns with RESEARCH Don't-Hand-Roll line 277.
- **Tree parser is stricter than C++ (Security V5 / T-03-03).** Every parsed array length is
  validated against `num_leaves`/`num_cat`, and every child/leaf index is range-checked,
  BEFORE any indexing — returning `ModelError::MalformedModel` instead of the C++ raw-`[]`
  UB. Observably identical on valid models.

## Deviations from Plan

None — plan executed as written. The plan named a 5-file test layout (`predict_transform`,
`predict_leaf_parity`, `predict_subrange`) but the **03-02 tasks** only require the two
layer-1/layer-2 tests (`model_text_roundtrip`, `predict_raw_parity`); the other three layers
are 03-03/03-04 scope (the RESEARCH layout was the whole-phase plan). No deviation — the two
tests this plan's `<tasks>` specify are present and green.

## Verification

- `cargo test -p lgbm-model --lib` — 36 unit tests pass (tree 13, ensemble 4, model_text 5,
  predict 5, format 6, error 3), incl. real regression `Tree=0` block byte-exact round-trip
  and an early full-`model.txt` round-trip check (done inline during dev, then removed).
- `cargo test -p lgbm-model --test model_text_roundtrip` — PASSES: Rust-written regression
  `.txt` is **byte-identical** to the committed C++ `.txt` (`compare_exact_bytes` Ok) —
  DAT-08 + DAT-09 round-trip.
- `cargo test -p lgbm-model --test predict_raw_parity` — PASSES: dense AND CSR AND CSC raw
  scores within `ORACLE_TOL` (~1e-6) of the committed `raw.txt` golden (`compare_within` Ok)
  over all 7000 rows — PRD-01.
- `cargo test --workspace` — green; lgbm-core/dataset/oracle unaffected (no regression).
- Predict path confirmed raw-value (no `construct`/`from_mat`/`BinMapper` call); f64 row
  buffer + f64 accumulator, f32 cast only at output. No `_inner_` in the predict path.
- Staging discipline: every commit staged explicitly by path; `LightGBM/`, `.serena/`,
  `AGENTS.md`, `.planning/config.json` never staged.

## Notes for Later Plans

- 03-03 reuses `Tree` (categorical decode already built — assert categorical-split parity),
  `GbdtModel`, and the `predict` driver; adds the `ConvertOutput` shim (sigmoid/softmax/ova)
  + transformed/leaf-index layers (D-06 layers 3-4) over binary/multiclass/categorical
  corpora. `predict_leaf_index` already exists on `Tree`.
- 03-04 reuses `init_predict`/`predict_raw(start_iteration, num_iteration)` — the sub-range
  args already land here; only the PRD-06 parity test + golden replay remain.
- The byte-exact writer covers `num_cat>0` (`cat_boundaries`/`cat_threshold` lines) and the
  `monotone_constraints`/`average_output` envelope lines — exercised once those corpora load
  in 03-03 (categorical) / future RF work.

## Self-Check: PASSED

- Created files verified present (tree/ensemble/model_text/predict.rs, golden/mod.rs, two
  integration tests) — see git diff HEAD~3..HEAD.
- Commits verified: 83ca79e (Task 1), ffb483e (Task 2), 3e00138 (Task 3).
