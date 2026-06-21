---
phase: 08-python-bindings
reviewed: 2026-06-08T00:00:00Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - crates/lgbm-python/src/lib.rs
  - crates/lgbm-python/src/error.rs
  - crates/lgbm-python/src/marshal.rs
  - crates/lgbm-python/src/dataset.rs
  - crates/lgbm-python/src/booster.rs
  - crates/lgbm-python/src/params.rs
  - crates/lgbm-python/src/callbacks.rs
  - crates/lgbm-python/python/lightgbm_rs/__init__.py
  - crates/lgbm-python/python/lightgbm_rs/callback.py
  - crates/lgbm-python/python/lightgbm_rs/engine.py
  - crates/lgbm-python/python/lightgbm_rs/sklearn.py
  - crates/lgbm-python/python/lightgbm_rs/plotting.py
  - crates/lgbm-model/src/ensemble.rs
  - crates/lgbm/src/booster.rs
findings:
  critical: 2
  warning: 5
  info: 5
  total: 12
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-06-08
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Phase 8 wires a PyO3 FFI seam over the validated `lgbm` facade plus a pure-Python
package (engine / sklearn / callbacks / plotting). The marshalling layer is
generally careful: sparse `indptr`/`indices` validation runs before any indexing,
the single-widen-site contract is honoured, the custom objective/metric error
channel correctly avoids panics via a captured-`PyErr` slot with a length-mismatch
sentinel, and the GIL detach/attach nesting in the callback path is the correct
pattern.

However, the review surfaced two BLOCKER-class violations of the stated
boundary invariants, both centred on a **missing prediction-input width check**:
`Booster.predict` and `Booster.refit` marshal a numpy matrix into rows and call
the facade without ever verifying the column count matches the trained model's
feature width. A too-narrow matrix indexes `feature_values[split_feature]`
out of bounds inside `Tree::predict`/`get_leaf`, which panics. That panic occurs
inside `py.detach`, so it surfaces as a `PanicException` rather than the
contract-required `ValueError` — directly contradicting "numpy input must be
validated at the boundary (shape → ValueError) before indexing" and "no panic
crosses the FFI boundary."

The remaining findings are parity divergences (split-importance guard, dropped
DataFrame feature names, flat multiclass `preds`) and quality gaps.

## Critical Issues

### CR-01: `Booster.predict` does not validate prediction-matrix column count → OOB panic crosses FFI

**File:** `crates/lgbm-python/src/booster.rs:48-73` (with `crates/lgbm-model/src/tree.rs:223-248`)
**Issue:**
`predict` calls `numpy_dense_to_rows(&data)` (which only checks non-empty) then
`py.detach(|| self.inner.predict(&rows))`. The facade path is
`predict` → `predict_row` → `GbdtModel::predict_raw` → `Tree::predict` →
`get_leaf`, which indexes `feature_values[self.split_feature[node] as usize]`
**without bounds checking** (`tree.rs:227,232`). If the caller passes a matrix
whose column count is `<= max(split_feature)` (e.g. fewer columns than the model
was trained on — a very common user error, or any matrix narrower than
`max_feature_idx + 1`), this is an out-of-bounds slice index and panics.

The panic happens inside `py.detach`; PyO3 converts it to a `pyo3_runtime.PanicException`
on re-acquire rather than a clean `ValueError`. This violates two CLAUDE.md
boundary invariants: (1) "numpy input must be validated at the boundary
(shape → ValueError) before indexing" and (2) "no panic crosses the FFI boundary."
There is no upstream width guard — the Python wrappers (`engine._TrainedBooster.predict`,
`sklearn.predict`) only `ascontiguousarray(..., float64)`, they do not check width
either, and the raw `_core.Booster.predict` is part of the public surface.

**Fix:** Validate the column count against the model width at the FFI boundary
before detaching. Expose the model width and check it:
```rust
fn predict<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<'py, f64>)
    -> PyResult<Bound<'py, PyArray2<f32>>>
{
    let view = data.as_array();
    let ncols = view.shape()[1];
    let need = (self.inner.model().max_feature_idx + 1).max(0) as usize;
    if ncols < need {
        return Err(PyValueError::new_err(format!(
            "prediction input has {ncols} features but the model expects at least {need}"
        )));
    }
    let rows = numpy_dense_to_rows(&data)?;
    let preds: Vec<Vec<f32>> = py.detach(|| self.inner.predict(&rows));
    // ...
}
```
(Alternatively, harden `Tree::get_leaf`/`predict` to bounds-check and return a
typed error, but a boundary check is the cheaper, contract-aligned fix.)

### CR-02: `Booster.refit` does not validate data-matrix width → OOB panic crosses FFI

**File:** `crates/lgbm-python/src/booster.rs:97-122` (with `crates/lgbm-model/src/ensemble.rs:245-253` → `tree.rs:223-237`)
**Issue:**
`refit` marshals `data` via `numpy_dense_to_rows` (non-empty check only), validates
only `labels.len() == rows.len()`, then `py.detach(|| self.inner.refit_data(&rows, ...))`.
`refit_data` → `GbdtModel::refit_ensemble_l2` → `refit_one_tree` calls
`self.trees[tree_index].get_leaf(row)` (`ensemble.rs:247`), which again indexes
`feature_values[split_feature]` with no bounds check. A `data` matrix narrower
than `max_feature_idx + 1` panics out of bounds inside the GIL-released region,
surfacing as `PanicException` instead of a `ValueError`. Same invariant violation
as CR-01, on the refit path.

**Fix:** Add the same width guard before `py.detach`:
```rust
let view = data.as_array();
let need = (self.inner.model().max_feature_idx + 1).max(0) as usize;
if view.shape()[1] < need {
    return Err(PyValueError::new_err(format!(
        "refit input has {} features but the model expects at least {need}",
        view.shape()[1]
    )));
}
```

## Warnings

### WR-01: `feature_importance('split')` uses the UNGUARDED count → diverges from C++ LightGBM

**File:** `crates/lgbm/src/booster.rs:512-514` (consumed by `crates/lgbm-python/src/booster.rs:138-143`)
**Issue:**
`Booster::feature_importance_split()` delegates to
`GbdtModel::feature_importance_split_count()`, which counts **every** stored split
(`ensemble.rs:157-168`). But the project's own documentation states that C++
`FeatureImportance(importance_type=0)` counts a split toward its feature ONLY when
`split_gain > 0` — the *guarded* variant `feature_importance_split_count_guarded`
(`ensemble.rs:193-212`). The Python `feature_importance(importance_type='split')`
therefore over-counts for any tree containing a zero-/negative-gain split,
diverging from the official package and the parity path that uses the guarded
count. This is a silent numerical-parity defect on a public API.

**Fix:** Point the facade split importance at the guarded variant:
```rust
pub fn feature_importance_split(&self) -> Vec<u64> {
    self.model.feature_importance_split_count_guarded()
}
```
(Keep the unguarded method for callers that explicitly want the raw structural count.)

### WR-02: polars DataFrame feature names are discarded

**File:** `crates/lgbm-python/src/dataset.rs:123-131` and `crates/lgbm-python/src/marshal.rs:328-396`
**Issue:**
`polars_df_to_corpus` returns `(rows, names, cat_indices)`, but `from_polars`
binds `let (rows, _names, cat_indices) = ...` and drops `_names`. The resulting
`RawCorpus`/model carries only the default generated `Column_i` names, so
`feature_importance`, model text, and `plot_importance` never reflect the user's
DataFrame column names. This breaks the official-package expectation that
DataFrame columns name the features, and `plot_importance` already hard-codes
`Column_i` (plotting.py:46), compounding the loss.

**Fix:** Thread the names into the corpus (add a `feature_names: Option<Vec<String>>`
to `RawCorpus`/the model-naming path) and use them in `from_rows_with_categorical`,
or at minimum surface them so `feature_importance`/plotting can label features.

### WR-03: `coerce_grad_hess` does not reject non-finite grad/hess from a custom objective

**File:** `crates/lgbm-python/src/callbacks.rs:68-101`
**Issue:**
`coerce_grad_hess` validates dtype, ravel order, and length, but a user `fobj`
returning `NaN`/`Inf` gradients or hessians passes through unchecked into the
boosting loop. The custom-metric path *does* guard non-finite values
(`ensemble.rs:78`), but the objective path does not, so a buggy custom objective
silently corrupts the model (or produces NaN leaves) instead of raising a clear
`ValueError`. Given the explicit "validate input at the boundary" posture, the
objective path should be symmetric.

**Fix:** After the length check, reject non-finite entries:
```rust
if slice.iter().any(|v| !v.is_finite()) {
    return Err(PyValueError::new_err(format!(
        "custom objective {which} contains a non-finite value (NaN/Inf)"
    )));
}
```

### WR-04: `_EarlyStoppingCallback._is_train_set` never matches this engine's dataset names

**File:** `crates/lgbm-python/python/lightgbm_rs/callback.py:203-205, 218, 282-283` (with `engine.py:291-294`)
**Issue:**
`_is_train_set` returns `dataset_name == "train"`, but `engine.train` names
validation sets `valid_0`, `valid_1`, ... (engine.py:291) and never emits a
dataset named `"train"`. So the "skip the training set for early stopping" and
"only_train_set disables ES" logic (callback.py:218, 282) is dead — early stopping
will happily stop on what the user intended as a train-set eval, diverging from
the official package when a user passes the train set into `valid_sets`.

**Fix:** Either name the train set consistently when it is added to eval sets, or
detect it by identity/position rather than the literal name `"train"`. At minimum,
document that the train-set-skip is inert in this engine so callers don't rely on it.

### WR-05: `train` rejects `feval` without `fobj`, silently dropping a valid official use case

**File:** `crates/lgbm-python/src/booster.rs:298-308`
**Issue:**
On the built-in-objective path, supplying `feval` without `fobj` raises
`ValueError` ("feval requires fobj"). In the official package `feval` is fully
usable with a built-in objective. The Python `engine.train`/`sklearn.fit` route
custom metrics through `_core.train`'s `feval`, so a user who supplies only a
custom metric (no custom objective) gets a hard error instead of metric
evaluation. This is a functional gap masquerading as validation. The custom-metric
hook is only wired on the custom-objective facade path
(`train_custom_raw_with_metric`), so the limitation is real, but it should be
surfaced as a documented NotImplemented/feature gap rather than implying the user
mis-called the API — and the higher-level `engine`/`sklearn` feval path is
effectively non-functional with built-in objectives.

**Fix:** Either wire `feval` into the built-in `train_raw` path (preferred), or
make the error message explicit that custom metrics with built-in objectives are
not yet supported, and ensure `engine.train`/`sklearn.fit` do not advertise
`eval_metric=callable` as working with built-in objectives.

## Info

### IN-01: multiclass custom-objective `preds` passed flat, not 2-D (column-major)

**File:** `crates/lgbm-python/src/callbacks.rs:126-135`
**Issue:** The official `fobj(preds, dataset)` receives `preds` reshaped to
`(num_data, num_class)` for multiclass; here it is passed as a flat 1-D f64 array.
A user multiclass custom objective written against the official shape will mis-index.
**Fix:** For `num_class > 1`, reshape the owned numpy array to `(num_data, num_class)`
(order matching the facade's class-major buffer) before calling.

### IN-02: `coerce_value` float formatting of non-finite params produces unparseable strings

**File:** `crates/lgbm-python/src/params.rs:72-76`
**Issue:** `format!("{f}")` renders `NaN`/`inf`/`-inf` as `"NaN"`/`"inf"`, which
C++ `Atof` will not parse to the intended value. A param like
`learning_rate=float('inf')` would coerce to `"inf"` and silently mis-configure.
**Fix:** Reject non-finite float param values with a `ValueError`, or document the
constraint.

### IN-03: `_make_n_folds` can emit empty validation folds for small/skewed classes

**File:** `crates/lgbm-python/python/lightgbm_rs/engine.py:379-393`
**Issue:** Stratified splitting assigns `np.arange(len(cls_idx)) % nfold`; a class
with fewer than `nfold` members leaves some folds with zero test rows of that
class, and `_eval_metric('auc', ...)` then returns NaN for that fold, contaminating
the mean. Not a crash, but a silent metric corruption.
**Fix:** Warn (or fall back to non-stratified) when any class has fewer than `nfold`
members, mirroring sklearn/official behaviour.

### IN-04: `LGBMClassifier.fit` silently overrides an explicit `objective`

**File:** `crates/lgbm-python/python/lightgbm_rs/sklearn.py:345-350`
**Issue:** `if self._objective is None or self._objective in (None, "binary", "multiclass")`
overrides an explicitly-set `objective="binary"`/`"multiclass"` based on the
observed class count, so a user who deliberately set one is silently switched.
**Fix:** Only auto-select when `self._objective is None`; otherwise honour the
explicit objective (and error if it is incompatible with the class count).

### IN-05: `plot_importance` text x-offset is a hard-coded magic constant

**File:** `crates/lgbm-python/python/lightgbm_rs/plotting.py:62`
**Issue:** `ax.text(x + 1, y, label, ...)` hard-codes a `+1` data-space offset for
the value label, which misplaces labels for importances whose scale is far from
unit (e.g. gain importances in the thousands or fractions). Cosmetic only.
**Fix:** Offset by a fraction of the axis range or use `annotate` with a points-based
offset.

---

_Reviewed: 2026-06-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
