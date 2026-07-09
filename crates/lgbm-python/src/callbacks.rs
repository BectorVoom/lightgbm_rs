//! Custom objective / metric callback marshalling (PYB-04, SC#4 — 08-RESEARCH
//! Pattern 5). Each boost iteration the facade hands the current raw scores to a
//! Python callable and gets `(grad, hess)` back; the custom metric (`feval`) is
//! handed `(scores, labels)` and returns `(name, value, is_higher_better)`. Both
//! callbacks run DURING the GIL-released `train_custom_with_metric` (the outer
//! call is wrapped in `Python::detach`), so each invocation RE-ACQUIRES the GIL
//! via `Python::attach` (the correct nested attach/detach pattern, D-13).
//!
//! # The error channel (no panic across FFI — CLAUDE.md, T-08-06-02)
//!
//! The facade closures have the fixed shapes `Fn(&[f64]) -> (Vec<f32>, Vec<f32>)`
//! and `Fn(&[f64], &[f32]) -> (String, f64, bool)` — there is no `Result` channel
//! out of them. A Python exception raised inside a callback (or a length / dtype
//! validation failure) is therefore captured into a shared [`ErrSlot`] and the
//! callback returns a sentinel (an empty/short vector for the objective, `NaN`
//! for the metric). The facade detects the sentinel as a typed error
//! (`LengthMismatch` / `CustomMetric`); the binding then PREFERS the captured
//! `PyErr` so the ORIGINAL Python traceback propagates intact rather than the
//! generic facade error (T-08-06-01/02). No `unwrap`/`panic!` ever runs.
//!
//! # Multiclass layout (Pitfall 5 / Security V5)
//!
//! grad/hess length is validated `== num_data * num_class` BEFORE the facade ever
//! indexes the returned buffers (the wrong-length over-read guard). For multiclass
//! the official `__boost` ravels `order="F"` (class-major); the binding requests
//! that layout from numpy so the flat buffer matches the facade's class-major
//! score layout.

use std::sync::{Arc, Mutex};

use lgbm::CustomMetricClosure;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A shared, thread-safe slot holding the FIRST `PyErr` raised inside a callback.
/// `PyErr` is `Send`, so this can be captured by the `Send` (`Ungil`) closure the
/// outer `Python::detach` runs. The first error wins (later iterations are no-ops
/// once set), preserving the original traceback.
pub type ErrSlot = Arc<Mutex<Option<PyErr>>>;

/// Record `err` into `slot` if no earlier error was captured (first-error-wins).
fn record_err(slot: &ErrSlot, err: PyErr) {
    if let Ok(mut guard) = slot.lock()
        && guard.is_none()
    {
        *guard = Some(err);
    }
}

/// Take the captured `PyErr` (if any) out of `slot`. Used after the facade call
/// returns so the binding can prefer the original Python traceback over the
/// generic facade error.
pub fn take_err(slot: &ErrSlot) -> Option<PyErr> {
    slot.lock().ok().and_then(|mut g| g.take())
}

/// Coerce a Python object returned by the objective callback into an owned
/// `Vec<f32>` of EXACTLY `expected` elements, ravelled class-major (`order="F"`)
/// for multiclass — mirroring the official `__boost` (`np.asarray(...,
/// dtype=np.float32)` + `.ravel(order="F")`). Validates the length at the FFI
/// boundary (Security V5, T-08-06-01) so the facade never over-reads.
///
/// # Errors
/// `PyValueError` when the object is not a 1-/2-D float array OR its length is not
/// `expected` (never panics, never over-reads).
fn coerce_grad_hess(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    expected: usize,
    which: &str,
) -> PyResult<Vec<f32>> {
    // np.asarray(obj, dtype=np.float32).ravel(order="F") via the numpy module so a
    // python list / f64 array / non-contiguous view is all handled exactly as the
    // official wrapper does (T-08-06-03: f32 + contiguous after ravel).
    let np = py.import("numpy")?;
    let asarray = np.getattr("asarray")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("float32")?)?;
    let arr = asarray.call((obj,), Some(&kwargs))?;
    // ravel(order="F") => class-major flat buffer (multiclass length num_data*num_class).
    let ravel_kwargs = pyo3::types::PyDict::new(py);
    ravel_kwargs.set_item("order", "F")?;
    let flat = arr.getattr("ravel")?.call((), Some(&ravel_kwargs))?;

    let pyarr = flat.cast::<numpy::PyArray1<f32>>().map_err(|_| {
        PyValueError::new_err(format!(
            "custom objective {which} could not be coerced to a 1-D float32 array"
        ))
    })?;
    let ro: PyReadonlyArray1<'_, f32> = pyarr.readonly();
    let slice = ro.as_slice()?;
    if slice.len() != expected {
        return Err(PyValueError::new_err(format!(
            "custom objective {which} has length {} but must equal num_data*num_class = {expected}",
            slice.len()
        )));
    }
    Ok(slice.to_vec())
}

/// Marshal a Python custom-objective callable into the facade objective closure
/// `Fn(&[f64]) -> (Vec<f32>, Vec<f32>)` (PYB-04, SC#4).
///
/// The callable mirrors the official `fobj(preds, dataset) -> (grad, hess)`: it is
/// invoked with the current raw scores (the f64 accumulated margin, length
/// `num_data * num_class`) wrapped as an OWNED numpy f64 array (never a slice lent
/// to Python — SC#1). The returned `(grad, hess)` are coerced to f32, ravelled
/// `order="F"`, and length-validated against `num_data * num_class` (Pitfall 5 /
/// Security V5). A Python exception or wrong-length return is captured into `slot`
/// and the closure returns empty vectors, which the facade surfaces as a typed
/// `LengthMismatch`; the binding then prefers the captured `PyErr`.
///
/// The closure RE-ACQUIRES the GIL via `Python::attach` per iteration (the facade
/// runs it DURING the GIL-released train). Note the per-iter GIL round-trip cost.
pub fn make_obj_closure(
    py: Python<'_>,
    callable: Py<PyAny>,
    dataset: Py<PyAny>,
    expected_len: usize,
    slot: ErrSlot,
) -> impl Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + Send {
    let callable = callable.clone_ref(py);
    let dataset = dataset.clone_ref(py);
    move |scores: &[f64]| -> (Vec<f32>, Vec<f32>) {
        // Re-attach the GIL inside the boost loop (nested attach under the outer
        // detach — the correct pattern, no double-acquire; T-08-06-02).
        Python::attach(|py| {
            let scores_np = scores.to_vec().into_pyarray(py);
            let args = (dataset.bind(py), scores_np);
            let result: PyResult<(Vec<f32>, Vec<f32>)> = (|| {
                // Official signature is fobj(preds, dataset); pass preds first.
                let preds = args.1;
                let ret = callable.call1(py, (preds, args.0))?;
                let ret = ret.bind(py);
                let tuple = ret.cast::<PyTuple>().map_err(|_| {
                    PyValueError::new_err(
                        "custom objective must return a (grad, hess) tuple of two arrays",
                    )
                })?;
                if tuple.len() != 2 {
                    return Err(PyValueError::new_err(format!(
                        "custom objective must return exactly 2 arrays (grad, hess), got {}",
                        tuple.len()
                    )));
                }
                let grad = coerce_grad_hess(py, &tuple.get_item(0)?, expected_len, "grad")?;
                let hess = coerce_grad_hess(py, &tuple.get_item(1)?, expected_len, "hess")?;
                Ok((grad, hess))
            })();
            match result {
                Ok(gh) => gh,
                Err(e) => {
                    record_err(&slot, e);
                    // Sentinel: empty vectors trip the facade LengthMismatch guard
                    // (never an over-read); the binding prefers the captured PyErr.
                    (Vec::new(), Vec::new())
                }
            }
        })
    }
}

/// Marshal a Python custom-metric (`feval`) callable into the 08-01 facade
/// custom-metric closure `Fn(&[f64], &[f32]) -> (String, f64, bool)` (PYB-04,
/// custom-metric half). This is the EXACT shape `lgbm::CustomMetricClosure`
/// expects; it is passed as the `Option<feval_closure>` argument to the upstream
/// `lgbm::train_custom_with_metric` and feeds the SAME eval-history loop (no new
/// Rust facade hook — the hook landed in 08-01).
///
/// The callable mirrors the official `_EvalFunctionWrapper`:
/// `feval(preds, dataset) -> (eval_name, eval_result, is_higher_better)`. It is
/// invoked with the current raw scores (owned numpy f64) and re-acquires the GIL
/// via `Python::attach` per eval. A Python exception is captured into `slot` and
/// the closure returns `(name, NaN, false)`, which the facade surfaces as a typed
/// `CustomMetric` error; the binding then prefers the captured `PyErr`.
pub fn make_metric_closure(
    py: Python<'_>,
    callable: Py<PyAny>,
    dataset: Py<PyAny>,
    slot: ErrSlot,
) -> CustomMetricClosure {
    let callable = callable.clone_ref(py);
    let dataset = dataset.clone_ref(py);
    Box::new(move |scores: &[f64], _labels: &[f32]| -> (String, f64, bool) {
        Python::attach(|py| {
            let scores_np = scores.to_vec().into_pyarray(py);
            let result: PyResult<(String, f64, bool)> = (|| {
                let ret = callable.call1(py, (scores_np, dataset.bind(py)))?;
                let ret = ret.bind(py);
                let tuple = ret.cast::<PyTuple>().map_err(|_| {
                    PyValueError::new_err(
                        "custom metric must return (eval_name, eval_result, is_higher_better)",
                    )
                })?;
                if tuple.len() != 3 {
                    return Err(PyValueError::new_err(format!(
                        "custom metric must return a 3-tuple (name, value, higher_better), got {}",
                        tuple.len()
                    )));
                }
                let name: String = tuple.get_item(0)?.extract()?;
                let value: f64 = tuple.get_item(1)?.extract()?;
                let higher: bool = tuple.get_item(2)?.extract()?;
                Ok((name, value, higher))
            })();
            match result {
                Ok(v) => v,
                Err(e) => {
                    record_err(&slot, e);
                    // Sentinel: NaN trips the facade CustomMetric non-finite guard;
                    // the binding prefers the captured PyErr.
                    ("custom".to_string(), f64::NAN, false)
                }
            }
        })
    })
}

/// Validate a Python refit `label`/`data` 1-D array's dtype is float at the FFI
/// boundary (helper kept here so booster.rs stays focused on delegation). Returns
/// the owned f32 values. `PyValueError` on an empty array (Security V5).
pub fn refit_label_to_f32(arr: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f32>> {
    let view = arr.as_array();
    if view.is_empty() {
        return Err(PyValueError::new_err("refit label array must be non-empty"));
    }
    Ok(view.iter().map(|&v| v as f32).collect())
}
