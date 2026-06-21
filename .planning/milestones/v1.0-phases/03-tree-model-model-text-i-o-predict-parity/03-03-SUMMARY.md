---
phase: 03-tree-model-model-text-i-o-predict-parity
plan: 03
subsystem: model
tags: [lgbm-model, objective, convert-output, predict, leaf-index, softmax, PRD-02, PRD-03, DAT-08, DAT-09]
requires:
  - lgbm-model::tree::Tree (categorical_decision / find_in_bitset / get_leaf — 03-02)
  - lgbm-model::ensemble::GbdtModel (predict_raw + num_tree_per_iteration stride + objective_string — 03-02)
  - lgbm-model::predict (raw dense/CSR/CSC driver — 03-02)
  - lgbm-model::error (ModelError — 03-01)
  - oracle-harness (compare_within / compare_exact_u32 / ORACLE_TOL)
  - committed binary/multiclass/categorical golden corpora (03-01)
provides:
  - "lgbm-model::objective::ObjectiveKind — objective= line parser + the four core ConvertOutput transforms"
  - "lgbm-model::objective::{convert_regression, convert_binary, softmax, convert_multiclassova} — free fns"
  - "lgbm-model::predict::{predict_mat, predict_csr, predict_csc} — transformed driver (D-06 layer 3, PRD-02)"
  - "lgbm-model::predict::{predict_leaf_index_mat, predict_leaf_index_csr, predict_leaf_index_csc} — leaf-index driver (D-06 layer 4, PRD-03)"
affects:
  - "03-04 (sub-range parity reuses the transformed/raw drivers + init_predict)"
tech-stack:
  added: []
  patterns:
    - "objective ConvertOutput parsed from the model objective= line, NOT a training Config"
    - "softmax with max-subtraction (common.h:587) — naive exp/sum forbidden"
    - "per-(row x iter x class) leaf-index stride [iter0_class0, iter0_class1, ...] (Pitfall 8)"
    - "shared shape validators + dense/CSR/CSC materializers across raw/transformed/leaf entry points"
key-files:
  created:
    - crates/lgbm-model/src/objective.rs
    - crates/lgbm-model/tests/predict_transform.rs
    - crates/lgbm-model/tests/predict_leaf_parity.rs
  modified:
    - crates/lgbm-model/src/lib.rs
    - crates/lgbm-model/src/predict.rs
    - crates/lgbm-model/tests/model_text_roundtrip.rs
decisions:
  - "ObjectiveKind enum (Regression{sqrt}/Binary{sigmoid}/Multiclass{num_class}/MulticlassOva{num_class,sigmoid}) parsed from objective= line; non-core objectives -> ModelError (T-03-09)"
  - "softmax seeded wmax=input[0] then folds [1..] — exact C++ Common::Softmax order; verified stable on [1000,1001,1002] (T-03-08)"
  - "transformed-predict width re-validated == num_tree_per_iteration before convert (malformed model guard)"
  - "leaf-index output width = ntpi * num_iteration, ordered per-(iter x class) (Pitfall 8); leaf ids cast i32->u32 (non-negative after !node)"
metrics:
  duration: ~12 min
  completed: 2026-06-05
  tasks: 2
  files: 6
---

# Phase 3 Plan 03: Transformed + leaf-index predict (binary / multiclass / categorical) Summary

Second Phase-3 vertical slice: the four core `ConvertOutput` transforms (PRD-02) +
leaf-index prediction (PRD-03), proving transformed-score parity (binary sigmoid,
multiclass softmax with max-subtraction, regression identity) and exact leaf-index
parity — including the multiclass `num_tree_per_iteration==num_class` per-(row×iter×class)
stride and the categorical `cat_threshold` bitset decision path — on top of the regression
raw-score spine from 03-02. Each new corpus's byte-exact round-trip re-confirms DAT-08/DAT-09.

## What Was Built

- **`objective.rs` — the `ConvertOutput` shim** (Task 1, TDD). `ObjectiveKind` parsed
  straight off the model's `objective=` line (`binary sigmoid:1`, `multiclass num_class:3`,
  `regression`, `regression sqrt`) — `sigmoid:`/`num_class:`/`sqrt` tokens split exactly as
  C++ `Common::Split(str, ':')`, NOT read from a training `Config` (RESEARCH line 370). The
  four transforms are 1:1 ports: `convert_regression` (identity or `Sign(x)*x*x`, where
  `Sign = (x>0)-(x<0)`), `convert_binary` (`1/(1+exp(-sigmoid*input))`), `softmax` reproducing
  `Common::Softmax` **with the `wmax` max-subtraction** (`common.h:587`), and
  `convert_multiclassova` (per-class sigmoid). Any non-core objective (huber/poisson/lambdarank/…)
  returns `ModelError::MalformedModel` — never a silent identity default (T-03-09). 12 inline
  tests incl. softmax numerical-stability on `[1000,1001,1002]` (T-03-08) and Test 7 asserting
  the categorical `Tree` decode (`find_in_bitset`) routes categories {1,3}→left, {0,2}/neg/NaN→right.
- **`predict.rs` extended — transformed + leaf-index drivers** (Task 2).
  `predict_mat`/`predict_csr`/`predict_csc` (PRD-02): materialize the raw f64 row, call
  `GbdtModel::predict_raw` (full range), then `ObjectiveKind::convert` (softmax in-place for
  multiclass over the `num_class` outputs), f32 cast only at the boundary. Output width =
  `num_tree_per_iteration`. `predict_leaf_index_mat`/`_csr`/`_csc` (PRD-03): walk
  `trees[i*ntpi+k].predict_leaf_index(row)` into the per-(iter×class) stride, output length
  `ntpi * num_iteration`, ordered `[iter0_class0, iter0_class1, ..., iter1_class0, ...]`
  (Pitfall 8), as `u32`. Extracted shared shape validators (`validate_dense/csr/csc_shape`) and
  materializers (`scatter_csr_row`, `scatter_csc_dense`) so raw/transformed/leaf all apply the
  same boundary checks (T-03-11). 6 new inline tests (regression identity == raw, multiclass
  softmax sums to 1, leaf layout for ntpi=1 and ntpi=2, CSR/CSC == dense, out-of-scope objective
  → err, too-few-cols → ShapeMismatch).
- **Integration tests** (Task 2). `predict_transform.rs` (D-06 layer 3): regression/binary/
  multiclass transformed output within `ORACLE_TOL` of the committed `transformed.txt`, with a
  row/class divergence localizer. `predict_leaf_parity.rs` (D-06 layer 4): regression (10
  cols = num_iter) + multiclass (30 cols = 10 iter × 3 class) leaf ids exactly equal to
  `leaf.txt` (`compare_exact_u32`), the multiclass case proving the stride. Extended
  `model_text_roundtrip.rs` to byte-exact round-trip binary/multiclass/categorical too.

## Key Decisions

- **`ObjectiveKind` is the parsed objective state, derived from the model text.** The enum
  carries exactly the params each C++ `ObjectiveFunction` subclass parses from its string-ctor
  (`sigmoid_`/`num_class_`/`sqrt_`). The transformed driver resolves it once per call from
  `model.objective_string` and re-validates `num_output() == num_tree_per_iteration` — a
  malformed model (objective width ≠ tree stride) is rejected, never mis-transformed.
- **Softmax order matches C++ exactly.** `wmax` is seeded with `input[0]` then folds `input[1..]`
  (the C++ loop starts at `i=1`); the subtraction-then-exp-then-normalize order reproduces
  `Common::Softmax` to the bit on the committed golden. Verified stable (finite, sums to 1) on a
  large-magnitude input that would overflow a naive `exp/sum` (T-03-08).
- **Leaf ids are `u32` after the non-negative `!node`.** C++ `PredictLeafIndex` yields the
  `~node` leaf id (always ≥ 0); the golden stores `u32`. The driver casts `i32→u32` at push.

## Deviations from Plan

None — plan executed as written. (The plan left the leaf-index ensemble method placement to
discretion; it lives in `predict.rs` as `predict_row_leaf` over the `GbdtModel` stride, reusing
the existing `Tree::predict_leaf_index` from 03-02 — no new ensemble method was needed.)

## Verification

- `cargo test -p lgbm-model objective::` — 12 tests pass: the four core transforms, the
  `objective=` line parser (core kinds + out-of-scope rejection + bad-param rejection without
  panic), softmax max-subtraction stability, and the categorical decode (Test 7).
- `cargo test -p lgbm-model --test predict_transform` — PASSES: regression (identity), binary
  (sigmoid), multiclass (softmax, 3-class stride) transformed output within `ORACLE_TOL` of the
  committed `transformed.txt` over all rows — PRD-02.
- `cargo test -p lgbm-model --test predict_leaf_parity` — PASSES: regression (10 leaf cols) AND
  multiclass (30 leaf cols, per-(iter×class) stride) leaf ids exactly equal the committed
  `leaf.txt` (`compare_exact_u32`) — PRD-03.
- `cargo test -p lgbm-model --test model_text_roundtrip` — PASSES: regression/binary/multiclass/
  categorical models round-trip BYTE-IDENTICAL (the categorical fixture exercises the
  `cat_boundaries`/`cat_threshold` ToString lines) — DAT-08/DAT-09 re-confirmed.
- `cargo test --workspace` — green; 0 failures across all crates (lgbm-core/dataset/oracle/model).
- `cargo clippy -p lgbm-model` — no warnings in the new `objective.rs`/`predict.rs` code. The one
  pre-existing `ensemble.rs:96` `needless_range_loop` warning is the faithful 1:1 C++ stride loop
  from 03-02 (out of scope for this plan; not introduced here).
- Staging discipline: every commit staged explicitly by path; `LightGBM/`, `.serena/`,
  `AGENTS.md`, `.planning/config.json` never staged.

## Notes for Later Plans

- 03-04 (sub-range parity) reuses the transformed/raw drivers + `init_predict(start, num)`; the
  `predict_raw(start_iteration, num_iteration)` sub-range args already exist (03-02). The
  transformed driver currently calls `predict_raw(.., 0, -1)` (full range) — 03-04 will thread
  the `(start_iteration, num_iteration)` arguments through `predict_mat`/leaf for PRD-06, OR add
  sub-range variants; the objective `convert` and leaf stride are unchanged.
- `multiclassova` is implemented + unit-tested but has no committed corpus (the captured
  multiclass fixture is plain `multiclass`/softmax); a future ova corpus would exercise its
  golden path.
- The categorical predict path is parity-asserted at the unit level (Test 7) and via the
  categorical byte-exact round-trip; a categorical *transformed/leaf* golden replay would
  additionally cover the full categorical predict path end-to-end (the categorical corpus uses
  `objective=binary`, so its transformed/leaf goldens would slot directly into the existing
  layer-3/4 tests if added to the corpus list).

## Self-Check: PASSED

- Created files verified present: `objective.rs`, `predict_transform.rs`, `predict_leaf_parity.rs`.
- Commits verified: 48da5ab (Task 1), 33d7a58 (Task 2).
