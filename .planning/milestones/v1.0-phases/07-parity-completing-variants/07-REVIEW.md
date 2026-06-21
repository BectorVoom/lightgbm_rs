---
phase: 07-parity-completing-variants
reviewed: 2026-06-07T10:40:20Z
depth: standard
files_reviewed: 56
files_reviewed_list:
  - crates/lgbm-boosting/src/error.rs
  - crates/lgbm-boosting/src/gbdt.rs
  - crates/lgbm-boosting/src/lib.rs
  - crates/lgbm-boosting/src/objective.rs
  - crates/lgbm-boosting/src/sample_strategy.rs
  - crates/lgbm-boosting/src/score_updater.rs
  - crates/lgbm-compute/src/gain.rs
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-core/src/config/mod.rs
  - crates/lgbm-core/src/config/set.rs
  - crates/lgbm-core/tests/config_defaults.rs
  - crates/lgbm-core/tests/config_validation.rs
  - crates/lgbm-metric/src/binary.rs
  - crates/lgbm-metric/src/dcg_calculator.rs
  - crates/lgbm-metric/src/error.rs
  - crates/lgbm-metric/src/lib.rs
  - crates/lgbm-metric/src/multiclass.rs
  - crates/lgbm-metric/src/rank.rs
  - crates/lgbm-metric/src/regression.rs
  - crates/lgbm-metric/src/xentropy.rs
  - crates/lgbm-model/src/ensemble.rs
  - crates/lgbm-model/src/model_text.rs
  - crates/lgbm-model/src/objective.rs
  - crates/lgbm-model/src/predict.rs
  - crates/lgbm-model/src/tree.rs
  - crates/lgbm-objective/Cargo.toml
  - crates/lgbm-objective/src/error.rs
  - crates/lgbm-objective/src/lib.rs
  - crates/lgbm-objective/src/rank.rs
  - crates/lgbm-objective/src/regression.rs
  - crates/lgbm-objective/src/xentropy.rs
  - crates/lgbm-treelearner/src/cost_effective_gradient_boosting.rs
  - crates/lgbm-treelearner/src/data_partition.rs
  - crates/lgbm-treelearner/src/feature_histogram_categorical.rs
  - crates/lgbm-treelearner/src/forced_splits.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/src/lib.rs
  - crates/lgbm-treelearner/src/monotone_constraints.rs
  - crates/lgbm/src/booster.rs
  - crates/lgbm/src/builder.rs
  - crates/oracle-harness/tests/advanced_parity.rs
  - crates/oracle-harness/tests/boosting_parity.rs
  - crates/oracle-harness/tests/learner_parity.rs
  - crates/oracle-harness/tests/metric_parity.rs
  - crates/oracle-harness/tests/predict_parity.rs
  - crates/oracle-harness/tests/rank_parity.rs
  - xtask/cpp/kernel_capture.cpp
  - xtask/py/advanced_oracle_capture.py
  - xtask/py/boosting_oracle_capture.py
  - xtask/py/categorical_oracle_capture.py
  - xtask/py/constraints_oracle_capture.py
  - xtask/py/dart_oracle_capture.py
  - xtask/py/goss_oracle_capture.py
  - xtask/py/metric_oracle_capture.py
  - xtask/py/predict_mode_oracle_capture.py
  - xtask/py/rank_oracle_capture.py
  - xtask/py/rf_oracle_capture.py
  - xtask/py/subset_determinism_capture.py
  - xtask/src/main.rs
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 07: Code Review Report

**Reviewed:** 2026-06-07T10:40:20Z
**Depth:** standard
**Files Reviewed:** 56
**Status:** issues_found

## Summary

Phase 07 completes the boosting variants (DART, GOSS, Random Forest), the
objective family (huber/fair/quantile/mape/poisson/gamma/tweedie/xentropy/
lambdarank/rank_xendcg), the metric family (binary/multiclass/rank/extended
regression), the categorical/monotone/forced-split/CEGB tree-learner additives,
and the model-ops (refit, importance, predict modes). The numerical transcription
is generally very faithful — gradient/hessian math, the gain primitives, the
GOSS/bagging/DART RNG draw orders, the percentile/median renewal paths, and the
f32-vs-f64 cast boundaries were cross-checked against the in-tree C++ reference
and match line-for-line in nearly every case. RNG-replay goldens freeze the
load-bearing draw orders.

One genuine **correctness divergence from the C++ reference** was found: the
standalone transformed batch-predict API in `lgbm-model` does not apply the
Random Forest `average_output` division by `num_iteration`, so a model loaded and
predicted through `predict_mat`/`predict_csr`/`predict_csc` returns the SUM of
trees instead of the AVERAGE for RF models — diverging from C++ `GBDT::Predict`
and from the in-crate `Booster::predict_row` (which DOES divide). Several
lower-severity divergences and quality issues are also recorded.

## Critical Issues

### CR-01: RF `average_output` division missing from transformed batch-predict API

**File:** `crates/lgbm-model/src/predict.rs:328-342` (callers `predict_mat`
`:365`, `predict_csr` `:392`, `predict_csc` `:420`; contrib helper `:621-636`).

**Issue:** C++ `GBDT::Predict` (`gbdt_prediction.cpp:55-66`) applies
`output[k] /= num_iteration_for_pred_` when `average_output_ == true` (the Random
Forest flag) BEFORE `ConvertOutput`. The Rust `predict_row_transformed` calls
`model.predict_raw(...)` then `kind.convert(...)` with NO averaging step.
`GbdtModel::predict_raw` (`ensemble.rs:90-102`) is the raw SUM and never divides
(correct — C++ `PredictRaw` also doesn't). For a model with `average_output =
true` (set by `Gbdt::into_model` when `variant == Rf`, `gbdt.rs:1573`), the
transformed predict therefore returns `num_iteration ×` the correct
probability/score. The in-crate `Booster::predict_row` (`booster.rs:227-240`) DOES
apply the division, so the bug is specifically in the standalone `lgbm-model`
predict API that the C-API / FFI / model-file consumers use — a real
numerical-fidelity break for any RF model run through that surface, and an
internal inconsistency with the `Booster` path.

**Fix:** Apply the average division in the transformed/contrib paths (NOT the raw
path) when `model.average_output`, mirroring C++ order (divide before convert):
```rust
fn predict_row_transformed(
    model: &GbdtModel, kind: &ObjectiveKind, row: &[f64],
    raw_buf: &mut Vec<f64>, conv_buf: &mut [f64], out: &mut Vec<f32>,
) {
    raw_buf.clear();
    raw_buf.extend(model.predict_raw(row, 0, -1));
    if model.average_output {
        let (_s, num) = model.init_predict(0, -1);
        if num > 0 {
            for v in raw_buf.iter_mut() { *v /= num as f64; }
        }
    }
    kind.convert(raw_buf, conv_buf);
    for &v in conv_buf.iter() { out.push(v as f32); }
}
```
Add an RF round-trip test in `predict.rs` (`average_output = true`, 2+ trees)
asserting the transformed output equals `Booster::predict_row` and is the per-tree
average, not the sum. Confirm whether `predict_contrib` for RF also needs the
`/ num_iteration` scaling (C++ scales the SHAP base/contribs for averaged output).

## Warnings

### WR-01: DART `max_drop <= 0` early-break diverges from C++ unbounded cap

**File:** `crates/lgbm-boosting/src/gbdt.rs:1280`, `:1293` (test mirror
`reference_drop` at `:2105`, `:2117`).

**Issue:** The drop loop breaks when
`dart.drop_index.len() >= cfg.max_drop.max(0) as usize`. C++ (`dart.hpp:111,123`)
breaks on `drop_index_.size() >= static_cast<size_t>(config_->max_drop)`. For
`max_drop < 0`, `static_cast<size_t>(-1)` is a huge value so C++ never breaks
(unbounded drops), whereas the Rust `(-1).max(0) = 0` makes the break fire after
the FIRST drop. For `max_drop == 0` both break after the first push (consistent).
The default `max_drop = 50`, and there is no `CHECK` on `max_drop` in `config.h`,
so a user-set negative `max_drop` silently caps drops at 1 in Rust vs unbounded in
C++.

**Fix:** Only apply the cap when `max_drop > 0`, matching the C++ unsigned cast:
```rust
if cfg.max_drop > 0 && dart.drop_index.len() >= cfg.max_drop as usize {
    break;
}
```
Apply in both the non-uniform and uniform branches and in `reference_drop`.

### WR-02: rank labels truncated as gain-table index without an integer-ness check

**File:** `crates/lgbm-metric/src/dcg_calculator.rs:118-124`, `:137`, `:161`;
`crates/lgbm-objective/src/rank.rs:319`, `:324`.

**Issue:** Rank labels index the gain table via `l as usize` (a C-style truncation
toward zero). The C++ `DCGCalculator::CheckLabel` (`dcg_calculator.cpp:147-162`)
additionally fatals when the label is NOT an integer (`label !=
static_cast<int>(label)`). The Rust `check_labels` validates range (`>= 0`,
`< label_gain.len()`) but NOT integer-ness, so a fractional label like `2.7`
silently truncates to gain index 2 here whereas C++ aborts. The range check
prevents OOB (no panic), so this is a parity/robustness gap rather than a crash,
but a non-integer rank label yields a silently different DCG vs C++.

**Fix:** Add the integer-ness check to `check_labels`:
```rust
if l.fract() != 0.0 {
    return Err(MetricError::LabelOutOfRange { label: l as i64, num_gains: n_gains });
}
```
(or a dedicated `NonIntegerLabel` variant). Mirror in the lambdarank ctor's
`dcg.check_labels` call.

### WR-03: `multi_logloss` vs `multi_error` handle out-of-range labels inconsistently

**File:** `crates/lgbm-metric/src/multiclass.rs:337-341` (logloss) vs `:94`
(multi_error).

**Issue:** `MultiLogloss::eval` does `let kk = labels[i] as usize; rec.get(kk)...`
— a negative label (`-1.0f32 as usize` saturates) or a wrap does not panic (the
`.get` returns `None → 0.0 → floor`), but a label that truncates into
`[0, num_class)` silently scores the wrong class. `MultiError::eval` instead clamps
with `.min(k_classes - 1)`. Neither validates that the label is a non-negative
integer `< num_class`, which the C++ `MulticlassMetric` relies on the objective
`Init` having enforced — and the two metrics handle the bad-label case
differently.

**Fix:** Validate `labels[i]` is a non-negative integer `< num_class` once (return
`MetricError::LabelOutOfRange`) in both `MultiLogloss::eval` and
`MultiError::eval`, or document the precondition and keep the defensive handling
consistent between them.

### WR-04: DART `inv_average_weight` 0/0 NaN is faithful-but-fragile

**File:** `crates/lgbm-boosting/src/gbdt.rs:1271,1274`.

**Issue:** `inv_average_weight = tree_weight.len() as f64 / sum_weight`. On the
first drop-eligible iteration `sum_weight == 0`, so this is `0.0/0.0 = NaN`, and
the `max_drop` cap term at `:1274` is also NaN. This matches C++ (same `0/0`), and
on iter 0 the per-tree drop loop `for i in 0..iter` is empty so the NaN is never
compared — byte-equal to C++. The risk is a future refactor or a continue-training
path (`num_init_iteration_ > 0`, `with_loaded_model` + DART) reaching the NaN
comparison with a non-empty drop loop, where `NaN < x` is always false (no trees
dropped) — a silent divergence waiting to surface.

**Fix:** No change required for current parity. Add a comment documenting the
deliberate `0/0` (mirroring C++) and guard the NaN if continue-training with DART
is ever wired so it cannot leak into a non-empty drop loop.

## Info

### IN-01: `safe_log` magic-number `1e-35` duplicated instead of reusing the constant

**File:** `crates/lgbm-metric/src/regression.rs:54`.

**Issue:** `const K_ZERO_THRESHOLD: f64 = 1e-35;` is hard-coded locally, while the
project convention (stated in `dcg_calculator.rs` / `gain.rs` / `tree.rs`) is to
reuse `lgbm_core::types::K_ZERO_THRESHOLD` and "never redefine 1e-35 / 1e-15
locally." The value matches C++ `kZeroThreshold`, so no numeric defect, but the
duplicated literal can drift from the canonical constant.

**Fix:** Replace with `f64::from(lgbm_core::types::K_ZERO_THRESHOLD)` (and drop
the local const), matching the binary/multiclass/objective crates' pattern.

### IN-02: `model_text::load` silently ignores `version=` and unknown header keys

**File:** `crates/lgbm-model/src/model_text.rs:96`.

**Issue:** The header parse `ignore`s any unrecognized key
(`_ => { /* tree / version=v4 / other header keys: ignored */ }`), including
`version`. A non-`v4` or corrupted-header model loads without warning. C++
`LoadModelFromString` checks the version. Low risk (committed fixtures are v4;
feature counts are still validated against `max_feature_idx`), but a
forward-incompatible model is accepted silently.

**Fix:** Capture `version` and reject (or warn on) a version other than
`MODEL_VERSION` ("v4") as a `ModelError::MalformedModel`.

### IN-03: lazy-CEGB `feature_used_in_data` allocation uses an unchecked usize product

**File:** `crates/lgbm-treelearner/src/cost_effective_gradient_boosting.rs:73`.

**Issue:** `vec![false; (num_features as usize) * (num_data as usize)]` allocates
the full (feature × row) seen-set when any `cegb_penalty_feature_lazy` is set. The
multiply is an unchecked `as usize` product; on a 32-bit target or a pathological
`num_features * num_data` it could over-allocate or wrap. Inputs come from
config/dataset (not adversarial network input) and in-scope corpora are small, so
this is a robustness note (performance is out of v1 scope).

**Fix:** Use `checked_mul` on the `usize` product and surface a typed error if it
overflows, or gate the allocation behind a sanity bound.

---

_Reviewed: 2026-06-07T10:40:20Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
