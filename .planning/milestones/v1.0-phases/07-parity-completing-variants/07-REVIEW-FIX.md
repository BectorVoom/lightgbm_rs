---
phase: 07-parity-completing-variants
fixed_at: 2026-06-07T11:15:00Z
review_path: .planning/phases/07-parity-completing-variants/07-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 07: Code Review Fix Report

**Fixed at:** 2026-06-07T11:15:00Z
**Source review:** .planning/phases/07-parity-completing-variants/07-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (1 critical + 4 warning; INFO findings IN-01/02/03 out of `critical_warning` scope)
- Fixed: 5
- Skipped: 0

All fixes were verified against the in-tree C++ reference under `LightGBM/`
before application, ran clean through `cargo build --workspace`, and were
checked for parity-golden regressions (rank/metric parity suites GREEN). In two
cases (WR-01, WR-02) the C++ reference revealed that the REVIEW.md's literal fix
suggestion would itself diverge from C++; the more faithful bit-exact
translation was applied instead and is documented below.

## Fixed Issues

### CR-01: RF `average_output` division missing from transformed batch-predict API

**Files modified:** `crates/lgbm-model/src/predict.rs`
**Commit:** 60eb5f5
**Applied fix:** Added the Random Forest `average_output` division
(`raw_buf[k] /= num_iteration`) inside `predict_row_transformed`, BEFORE
`kind.convert`, mirroring C++ `GBDT::Predict` (`gbdt_prediction.cpp:57-61`) and
the in-crate `Booster::predict_row`. The raw path (`predict_raw`) is unchanged
(C++ `PredictRaw` also does not divide). This fixes `predict_mat` / `predict_csr`
/ `predict_csc` which all route through `predict_row_transformed`. Verified the
contrib path needs NO change: C++ `GBDT::PredictContrib` (`gbdt.cpp:640-651`)
does NOT apply `average_output_` — it sums per-tree contributions — and the Rust
`predict_row_contrib` already matches that. Added a regression test
(`transformed_rf_average_output_divides_by_num_iteration`) asserting the
transformed dense/CSR/CSC outputs equal the per-tree average for an
`average_output = true`, 2-tree model.

### WR-01: DART `max_drop <= 0` early-break diverges from C++ unbounded cap

**Files modified:** `crates/lgbm-boosting/src/gbdt.rs`
**Commit:** 4dab063
**Applied fix:** Replaced `cfg.max_drop.max(0) as usize` with `cfg.max_drop as
usize` in both the non-uniform and uniform drop loops and in the `reference_drop`
test mirror. Verified that Rust `i32 as usize` sign-extends to 64-bit
(`-1 -> 18446744073709551615`, `0 -> 0`), reproducing C++
`static_cast<size_t>(config_->max_drop)` (`dart.hpp:111,123`) EXACTLY for all
three cases: negative `max_drop` => huge bound => unbounded drops; `max_drop ==
0` => break after first push; `max_drop > 0` => break at the cap. NOTE: the
REVIEW.md's suggested fix (`cfg.max_drop > 0 && ...`) was NOT used because it
would NOT break for `max_drop == 0`, diverging from C++ (which breaks after the
first push for 0). The `i32 as usize` cast is the bit-exact translation.

### WR-02: rank labels truncated as gain-table index without an integer-ness check

**Files modified:** `crates/lgbm-metric/src/error.rs`,
`crates/lgbm-metric/src/dcg_calculator.rs`,
`crates/lgbm-objective/src/rank.rs`
**Commit:** e8cfecb
**Applied fix:** Added a `MetricError::NonIntegerLabel` variant and an
integer-ness check to `DcgCalculator::check_labels`, mirroring the FIRST check in
C++ `DCGCalculator::CheckLabel` (`dcg_calculator.cpp:148-153`):
`fabs(label - static_cast<int>(label)) > kEpsilon`. Used `K_EPSILON` (1e-15f) and
`as i32 as f32` truncation toward zero (NOT `floor`) to match C++ exactly, and
preserved the C++ check ORDER (integer-ness, then non-negative, then range).
NOTE: the REVIEW.md's `l.fract() != 0.0` suggestion was made faithful by using
the `kEpsilon` tolerance the C++ uses rather than an exact-zero compare. The
lambdarank ctor's `dcg.check_labels` call (`rank.rs:154`) inherits the new check;
its error mapping was extended to surface `NonIntegerLabel` with the actual label
value. Added a `check_labels_rejects_non_integer` test.

### WR-03: `multi_logloss` vs `multi_error` handle out-of-range labels inconsistently

**Files modified:** `crates/lgbm-metric/src/error.rs`,
`crates/lgbm-metric/src/multiclass.rs`
**Commit:** 9658bf1
**Applied fix:** Added a shared `check_multiclass_labels` helper (validates each
label is a non-negative integer `< num_class`, using the same `K_EPSILON`
integer-ness check as the ranking path) and a `MetricError::MulticlassLabelOutOfRange`
variant. Both `MultiLogloss::eval` and `MultiError::eval` now call it once up
front, so a bad label is rejected as a typed error CONSISTENTLY (previously
`multi_error` clamped with `.min(k_classes-1)` while `multi_logloss` floored via
`.get().unwrap_or(0.0)`). After validation, both index `rec[label]` directly,
matching C++ `LossOnPoint`'s unchecked `ref_score[static_cast<size_t>(label)]`
(`multiclass_metric.hpp:142-174`). Added a
`multiclass_metrics_reject_bad_labels_consistently` test covering out-of-range
and fractional labels through both metrics.

### WR-04: DART `inv_average_weight` 0/0 NaN is faithful-but-fragile

**Files modified:** `crates/lgbm-boosting/src/gbdt.rs`
**Commit:** 680fa78
**Applied fix:** Documentation-only, as the REVIEW.md fix specified ("No change
required for current parity"). Added a comment at the `inv_average_weight`
computation documenting the DELIBERATE `0/0 = NaN` mirroring C++ `dart.hpp:104`,
why it is safe under fresh training (the iter-0 drop loop is empty so the NaN is
never compared), and a WARNING to guard `sum_weight == 0` if DART
continue-training (`with_loaded_model` + `num_init_iteration_ > 0`) is ever
wired, where a non-empty drop loop could reach the NaN. No behavioral change — the
0/0 must remain bit-exact to C++.

## Skipped Issues

None.

The three INFO findings (IN-01 `safe_log` magic-number, IN-02 model `version=`
header check, IN-03 lazy-CEGB unchecked `usize` product) are outside the
`critical_warning` fix scope and were not attempted.

---

_Fixed: 2026-06-07T11:15:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
