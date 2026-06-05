---
phase: 03-tree-model-model-text-i-o-predict-parity
plan: 04
subsystem: model
tags: [lgbm-model, ensemble, predict, sub-range, init-predict, PRD-06, PRD-01]
requires:
  - lgbm-model::ensemble::GbdtModel (init_predict + predict_raw start/num args — 03-02)
  - lgbm-model::predict (raw dense/CSR/CSC driver — 03-02)
  - oracle-harness (compare_within / ORACLE_TOL)
  - committed subrange golden corpus (subrange.txt — 03-01)
provides:
  - "lgbm-model::predict::{predict_raw_mat_range, predict_raw_csr_range, predict_raw_csc_range} — sub-range raw driver (D-06 layer 5, PRD-06)"
  - "lgbm-model::ensemble::GbdtModel::init_predict — parity-asserted InitPredict clamp/slice (gbdt.h:426-435)"
affects:
  - "Phase 3 complete — full D-06 layered parity battery (layers 1-5) green"
tech-stack:
  added: []
  patterns:
    - "InitPredict sub-range clamp: start=clamp(start_iteration,0,total); num = if num_iteration>0 {min(num_iteration,total-start)} else {total-start} — -1 and 0 both mean all-from-start"
    - "full-range public entries delegate to *_range with (0,-1) — keeps 03-02/03-03 callers signature-stable"
    - "SLICE-header golden replay: parse `SLICE start=<s> num=<n>` + per-slice f64-bits line, compare_within per pair"
key-files:
  created:
    - crates/lgbm-model/tests/predict_subrange.rs
  modified:
    - crates/lgbm-model/src/ensemble.rs
    - crates/lgbm-model/src/predict.rs
decisions:
  - "init_predict already faithfully ports gbdt.h:426-435 from 03-02; Task 1 finalized it with the full <behavior> clamp/slice battery (Tests 1-5) rather than rewriting it"
  - "added predict_raw_{mat,csr,csc}_range threading start/num; the existing full-range predict_raw_{mat,csr,csc} now delegate with (0,-1) — no caller churn, no signature break for 03-02/03-03"
  - "sub-range test exercises DENSE only — input form is orthogonal to slice math (CSR/CSC raw parity covered in 03-02); the 4 committed slices (0,10)/(0,5)/(5,-1)/(1,1) already cover -1==all, bounded count, non-zero start"
  - "capture NOT extended — the 03-01 subrange.txt already records the representative pairs this test needs (byte-idempotency preserved by not touching capture)"
metrics:
  duration: ~10 min
  completed: 2026-06-05
  tasks: 2
  files: 3
---

# Phase 3 Plan 04: Sub-range prediction parity (start_iteration / num_iteration) Summary

Final Phase-3 vertical slice: sub-range raw prediction returning the C++-matching
slice of the ensemble for representative `(start_iteration, num_iteration)` pairs —
including `num_iteration == -1` (== all), a bounded count, and a non-zero start.
Closes PRD-06 and completes the D-06 layered parity battery (layers 1-5). The
accumulation loop and the `init_predict` clamp helper already existed (03-02); this
slice finalizes the clamp/slice math with the full `<behavior>` battery, threads the
two iteration-range params through the batch predict driver, and adds the layer-5
parity test that replays the committed C++ sub-range golden.

## What Was Built

- **`ensemble.rs` — parity-asserted `init_predict` slice math** (Task 1, TDD).
  `init_predict(start_iteration, num_iteration) -> (i32, i32)` is the existing 1:1
  port of C++ `GBDT::InitPredict` (`gbdt.h:426-435`): `total = models_.size()/ntpi`;
  `start = clamp(start_iteration, 0, total)`; `num = if num_iteration > 0 { min(num_iteration, total-start) } else { total-start }` (so `-1` AND `0` both mean
  "all from start", matching the C++ `num_iteration > 0` test). Task 1 added the full
  `<behavior>` battery on a 5-iteration model: Test 1 (`-1==all` → `(0,total)`, and
  `0` → `(0,total)`), Test 2 (bounded count → `(0, min(k,total))`), Test 3 (non-zero
  start → `(2, total-2)` / `(2, min(3,total-2))` / `(4,1)`), Test 4 (over-range clamp
  → `(total, 0)`, plus `i32::MAX`/`i32::MIN`/negative-start clamp and an empty-slice
  `predict_raw` returning the zero accumulator — T-03-12, never panics/OOB), and Test 5
  (slice accumulation: `predict_raw(row, 1, 2)` sums ONLY iterations 1 and 2 = 6.0,
  equals the manual sum of exactly those trees, and differs from the full-range 31.0).
- **`predict.rs` — sub-range threaded through the batch driver** (Task 1). Added
  `predict_raw_mat_range` / `predict_raw_csr_range` / `predict_raw_csc_range` accepting
  `start_iteration: i32` + `num_iteration: i32` and threading them into
  `GbdtModel::predict_raw` for every row (the per-row `predict_row` helper now carries
  the two params). The existing full-range `predict_raw_mat` / `_csr` / `_csc` are now
  thin wrappers that delegate with `(0, -1)` — so the 03-02 raw-parity and 03-03
  transformed/leaf callers keep their signatures and stay green. The clamp lives in
  `init_predict`, so extreme `start_iteration`/`num_iteration` never panic or index OOB.
- **`predict_subrange.rs` — D-06 layer-5 sub-range parity test** (Task 2). Parses the
  committed `subrange.txt` golden (a flat sequence of `SLICE start=<s> num=<n>` headers
  each followed by one `;`-separated f64-bits data line — the format emitted by
  `xtask model-capture`/`dump_subrange` in 03-01), loads the `subrange` `model.txt`,
  and for each recorded `(start, num)` pair calls `predict_raw_mat_range` on the
  `regression.train` matrix and asserts `compare_within(rust, golden, ORACLE_TOL)` is
  Ok, with a localizing message naming the `(start, num)` pair. Asserts the golden
  records ≥ 3 distinct slices covering `-1==all` (`num <= 0`), a bounded count
  (`num > 0`), and a non-zero start (`start > 0`). Graceful SKIP + regen eprintln when
  the fixture is absent. Dense input form only (CSR/CSC raw parity is covered in 03-02;
  sub-range is orthogonal to input form).

## Key Decisions

- **`init_predict` was finalized, not rewritten.** 03-02 already landed the faithful
  `gbdt.h:426-435` port; this plan's Task 1 added the complete `<behavior>` clamp/slice
  battery (incl. the over-range / extreme-value / empty-slice T-03-12 cases) and proved
  the slice differs from the full range — no behavior change to the existing port.
- **Full-range entries delegate to `*_range` with `(0, -1)`.** Rather than break the
  03-02/03-03 public signatures, the existing `predict_raw_mat`/`_csr`/`_csc` now call
  the new `_range` variants with the full-range defaults. No caller churn; the full
  Phase-3 suite stays green.
- **Capture was NOT extended.** The 03-01 `subrange.txt` already records the four
  representative slices `(0,10)`, `(0,5)`, `(5,-1)`, `(1,1)` — covering `-1==all`,
  bounded count, and non-zero start. Reusing them keeps `model-capture` byte-idempotent
  (no fixture re-emission, empty git diff).

## Deviations from Plan

None — plan executed as written. (The plan's `<artifacts>` lists `ensemble.rs` as the
home of the slice math; the per-row threading helper and the `_range` batch entries live
in `predict.rs` exactly as the plan's `<action>` directs. `init_predict` already existed
from 03-02 — Task 1 finalized + parity-asserted it rather than introducing it.)

## Verification

- `cargo test -p lgbm-model ensemble::` — 9 tests pass, incl. the full `<behavior>`
  battery: `init_predict_minus_one_is_all`, `init_predict_bounded_count`,
  `init_predict_non_zero_start`, `init_predict_over_range_clamps_to_empty` (T-03-12,
  extreme/negative/`i32::MAX`/`i32::MIN` all clamp, empty-slice predict returns zero —
  no panic), and `predict_raw_slice_accumulates_only_selected_iterations` (slice == manual
  sum of exactly the selected trees, differs from full range).
- `cargo test -p lgbm-model --test predict_subrange` — PASSES: dense sub-range raw scores
  for all four committed slices `(0,10)`/`(0,5)`/`(5,-1)`/`(1,1)` within `ORACLE_TOL`
  (~1e-6) of the C++ `subrange.txt` golden over all 7000 rows — PRD-06. The test asserts
  ≥ 3 distinct slices covering `-1==all`, bounded count, and non-zero start.
- `cargo test -p lgbm-model --test predict_raw_parity --test predict_transform --test predict_leaf_parity` —
  PASS: no regression from the threaded signature (full-range default unchanged).
- `cargo test --workspace` — GREEN; 0 failures across all crates (lgbm-core/dataset/oracle/
  model/xtask). Full D-06 layered battery (layers 1-5) passes. lgbm-model: 60 lib tests +
  4 roundtrip + 2 leaf + 1 raw + 1 subrange + 3 transform.
- `cargo clippy -p lgbm-model` — no new warnings in the added code. Pre-existing warnings
  (format.rs doc-list-item from 03-01; the `predict_raw` `k`-index stride loop from 03-02;
  lgbm-dataset warnings) are out of scope and not introduced here.
- Capture idempotency: `model-capture` was NOT touched, so the committed fixtures are
  unchanged (byte-idempotent by construction — empty git diff on the fixtures dir).
- Staging discipline: every commit staged explicitly by path; `LightGBM/`, `.serena/`,
  `AGENTS.md`, `.planning/config.json` never staged.

## Notes for Later Plans

- Sub-range is threaded only through the RAW batch driver (`predict_raw_*_range`). The
  transformed (`predict_mat`/`_csr`/`_csc`) and leaf-index drivers still predict the full
  range (`predict_raw(.., 0, -1)` / all iterations). A future plan wanting transformed or
  leaf-index sub-range would thread the same two params through those entries (the
  objective `convert` and leaf stride are unchanged; only the inner `predict_raw` call /
  the leaf `num_iter` loop bound would take the clamped `(start, num)`).
- The `multiclassova` sub-range path is supported by the slice math (ntpi-agnostic) but
  has no committed corpus; a future ova sub-range golden would slot into this test.

## Self-Check: PASSED

- Created file verified present: `crates/lgbm-model/tests/predict_subrange.rs`.
- Modified files verified present: `crates/lgbm-model/src/ensemble.rs`, `crates/lgbm-model/src/predict.rs`.
- Commits verified: bb29ec5 (Task 1), 642f5a5 (Task 2).
