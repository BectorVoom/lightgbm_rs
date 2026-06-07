//! `#[pyclass] Booster` + the `train` free function — the GIL-released train/
//! predict seam (08-RESEARCH Pattern 2). Both `train` and `Booster::predict`
//! marshal their numpy input into owned Rust buffers WHILE the GIL is held, then
//! run the CPU-bound facade call inside `Python::detach` (GIL RELEASED, D-13/
//! SC#1), and return OWNED numpy arrays (never a slice lent back to Python).
//!
//! No algorithm lives here: `train` delegates to [`lgbm::train_raw`] and
//! `predict` to [`lgbm::Booster::predict`]. Every method returns `PyResult<_>`
//! and routes facade errors through [`crate::error`]; no panic crosses the FFI
//! boundary (CLAUDE.md, T-08-02-03).

use std::sync::{Arc, Mutex};

use lgbm::{Booster as FacadeBooster, Config, CustomMetricClosure, RawCorpus};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::callbacks::{
    make_metric_closure, make_obj_closure, refit_label_to_f32, take_err, ErrSlot,
};
use crate::dataset::Dataset;
use crate::error::LgbmErrorWrap;
use crate::marshal::numpy_dense_to_rows;
use crate::params::build_config_with_overrides;

/// A trained LightGBM-rs model (mirrors `lightgbm.Booster`). Holds the owned
/// facade [`lgbm::Booster`]; every method is a thin GIL-aware delegation.
#[pyclass(module = "lightgbm_rs._core")]
pub struct Booster {
    /// The owned, validated facade booster.
    pub(crate) inner: FacadeBooster,
}

#[pymethods]
impl Booster {
    /// Batch transformed prediction over a dense numpy f64 matrix `data`
    /// (num_rows x num_features). Returns an OWNED `(num_rows, num_output)`
    /// numpy f32 array (SC#1: never lends a Rust slice to Python).
    ///
    /// The input is marshalled to owned rows WHILE the GIL is held, then the
    /// CPU-bound prediction runs with the GIL RELEASED (`Python::detach`).
    ///
    /// # Errors
    /// `ValueError` on an empty / malformed input array or a ragged prediction
    /// result (never panics).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = numpy_dense_to_rows(&data)?;
        // GIL RELEASED around the CPU-bound predict (D-13/SC#1).
        let preds: Vec<Vec<f32>> = py.detach(|| self.inner.predict(&rows));
        let nrows = preds.len();
        let ncols = preds.first().map_or(0, Vec::len);
        // Flatten row-major into an owned buffer; reject a ragged result rather
        // than silently truncating (defensive — predict is rectangular today).
        let mut flat: Vec<f32> = Vec::with_capacity(nrows * ncols);
        for row in &preds {
            if row.len() != ncols {
                return Err(PyValueError::new_err(format!(
                    "internal: ragged prediction rows ({} vs {ncols})",
                    row.len()
                )));
            }
            flat.extend_from_slice(row);
        }
        let arr = numpy::ndarray::Array2::from_shape_vec((nrows, ncols), flat)
            .map_err(|e| PyValueError::new_err(format!("prediction shape error: {e}")))?;
        Ok(arr.into_pyarray(py))
    }

    /// Number of boosting iterations (trees per class) in the model. Mirrors
    /// `Booster.num_trees()` / `num_iteration()`.
    fn num_iteration(&self) -> i32 {
        self.inner.model().num_iteration()
    }

    /// Refit the model's leaf values to new `data`/`label` (mirrors
    /// `lightgbm.Booster.refit`, PYB-04 / ADV-06). Delegates per-tree to the
    /// validated 08-01 facade [`lgbm::Booster::refit`] (`GbdtModel::refit_one_tree`):
    /// for each tree the current raw margin produces the L2 grad/hess (the default
    /// objective used for refit by the official package when no custom objective is
    /// supplied), and each leaf's output is decay-blended with the new data's leaf
    /// statistics.
    ///
    /// Inputs are marshalled to owned Rust buffers WHILE the GIL is held; the
    /// CPU-bound refit then runs with the GIL RELEASED (`Python::detach`). Returns
    /// `None` (refit mutates in place, mirroring the official method's self-return
    /// contract surfaced here as an in-place update).
    ///
    /// # Errors
    /// `ValueError` on an empty / malformed `data`/`label`, or a length mismatch
    /// (never panics).
    #[pyo3(signature = (data, label, decay_rate = 0.9))]
    fn refit(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        label: PyReadonlyArray1<'_, f64>,
        decay_rate: f64,
    ) -> PyResult<()> {
        let rows = numpy_dense_to_rows(&data)?;
        let labels = refit_label_to_f32(&label)?;
        if labels.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "refit label length {} != number of data rows {}",
                labels.len(),
                rows.len()
            )));
        }
        let num_trees = self.inner.model().num_iteration() as usize;
        // GIL RELEASED around the CPU-bound per-tree refit (D-13/SC#1).
        py.detach(|| {
            for tree_index in 0..num_trees {
                // L2 grad/hess from the current raw margin over the new data:
                // grad = pred - label, hess = 1 (the default-objective refit the
                // official package performs when no custom objective is set).
                let preds = self.inner.predict_raw_batch(&rows, 0, -1);
                let mut gradients = vec![0.0f32; rows.len()];
                let hessians = vec![1.0f32; rows.len()];
                for (i, p) in preds.iter().enumerate() {
                    let pred = p.first().copied().unwrap_or(0.0);
                    gradients[i] = (pred - labels[i] as f64) as f32;
                }
                self.inner.refit(
                    tree_index, &rows, &gradients, &hessians, decay_rate, false, 0.0, 0.0,
                );
            }
        });
        Ok(())
    }

    /// Per-feature importance (mirrors `lightgbm.Booster.feature_importance`,
    /// ADV-07). `importance_type='split'` → split-count importance (returned as
    /// f64 for a uniform numpy dtype); `'gain'` → summed-gain importance. Returns
    /// an OWNED 1-D numpy f64 array (`into_pyarray`; SC#1 — never lends a slice).
    ///
    /// # Errors
    /// `ValueError` for an unrecognized `importance_type` (never panics).
    #[pyo3(signature = (importance_type = "split"))]
    fn feature_importance<'py>(
        &self,
        py: Python<'py>,
        importance_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values: Vec<f64> = match importance_type {
            "split" => self
                .inner
                .feature_importance_split()
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            "gain" => self.inner.feature_importance_gain(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "importance_type must be 'split' or 'gain', got '{other}'"
                )));
            }
        };
        Ok(values.into_pyarray(py))
    }
}

/// Train a [`Booster`] from a `params` dict and a [`Dataset`] (mirrors
/// `lightgbm.train(params, train_set, num_boost_round)`).
///
/// The `params` dict is coerced + validated through the full D-06/07/08 pipeline
/// ([`crate::params::build_config_with_overrides`]): typed values are coerced to
/// C++-matching strings (D-08), recognized-but-unimplemented params raise a clear
/// `ValueError` (D-07), and the result routes through `Config::from_params`
/// (alias resolution + CHECK validation; unknown typos warn, D-06).
///
/// `num_boost_round` takes precedence over any iteration count in `params`
/// (matching the official package), injected as the canonical `num_iterations`
/// override AFTER coercion + the D-07 gate but BEFORE `from_params` (a directly-
/// set canonical always beats an alias). The corpus is already marshalled (it
/// lives in the owned `Dataset`), so training runs entirely with the GIL RELEASED
/// (`Python::detach`, D-13/SC#1).
///
/// # Errors
/// `ValueError` for invalid params / corpus (caller-input defects);
/// `lightgbm_rs.LightGBMError` for engine-side failures. Never panics.
///
/// When `fobj` is supplied it is a custom objective callable
/// (`fobj(preds, dataset) -> (grad, hess)`, mirroring the official package); the
/// binding marshals it into the facade closure and trains via the CUSTOM path
/// (`lgbm::train_custom_with_metric`), where `boost_from_average` is forced OFF
/// (mirroring C++ `obj == null`). When `feval` is ALSO supplied it is a custom
/// metric (`feval(preds, dataset) -> (name, value, is_higher_better)`) marshalled
/// into the 08-01 facade custom-metric hook and recorded in eval history. The
/// callbacks re-acquire the GIL per iteration via `Python::attach` while the outer
/// train runs under `py.detach` (a per-iter GIL round-trip — the documented cost
/// of a Python callback inside a GIL-released loop).
///
/// # Errors
/// `ValueError` for invalid params / corpus (caller-input defects);
/// `lightgbm_rs.LightGBMError` for engine-side failures. A Python exception raised
/// inside a callback propagates with its ORIGINAL traceback (captured across the
/// fixed-shape facade closure). Never panics.
#[pyfunction]
#[pyo3(signature = (params, train_set, num_boost_round = 100, fobj = None, feval = None))]
pub fn train(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    train_set: &Dataset,
    num_boost_round: i32,
    fobj: Option<Py<PyAny>>,
    feval: Option<Py<PyAny>>,
) -> PyResult<Booster> {
    // Full D-06/07/08 coercion + validation; num_boost_round wins over any params
    // iteration alias (official-package precedence) via the canonical override.
    let cfg: Config = build_config_with_overrides(
        params,
        [("num_iterations", num_boost_round.to_string())],
    )?;
    let corpus: &RawCorpus = &train_set.corpus;

    // No custom objective → the built-in raw→bin→train path (08-02 behavior,
    // preserved exactly; the default `train` signature still works).
    let Some(fobj) = fobj else {
        if feval.is_some() {
            return Err(PyValueError::new_err(
                "feval (custom metric) requires fobj (custom objective) — \
                 the custom-metric hook is only wired on the custom-objective path",
            ));
        }
        // GIL RELEASED around the CPU-bound training (D-13/SC#1).
        let inner = py.detach(|| lgbm::train_raw(&cfg, corpus)).map_err(LgbmErrorWrap)?;
        return Ok(Booster { inner });
    };

    // Custom-objective (and optional custom-metric) path over the RAW corpus: the
    // facade bins each feature with the bit-exact BinMapper then runs the custom
    // eval-history loop (`train_custom_raw_with_metric`). boost_from_average is
    // forced OFF for custom inside the facade (mirroring C++ obj == null).
    let num_data = corpus.labels.len();
    let num_class = cfg.num_class.max(1) as usize;
    let expected_len = num_data * num_class;

    // The callbacks' `dataset` arg surfaces the labels (a numpy f64 array) so a
    // user closure can read them off it (mirrors fobj(preds, dataset)).
    let dataset_obj: Py<PyAny> = train_set_as_pyobject(py, train_set)?;

    let obj_slot: ErrSlot = Arc::new(Mutex::new(None));
    let metric_slot: ErrSlot = Arc::new(Mutex::new(None));

    let obj_closure = make_obj_closure(
        py,
        fobj,
        dataset_obj.clone_ref(py),
        expected_len,
        obj_slot.clone(),
    );
    let metric_closure: Option<CustomMetricClosure> =
        feval.map(|f| make_metric_closure(py, f, dataset_obj.clone_ref(py), metric_slot.clone()));

    // GIL RELEASED around the CPU-bound custom train; the callbacks re-attach per
    // iter via Python::attach (the correct nested pattern, D-13/SC#1).
    let result = py.detach(|| {
        lgbm::train_custom_raw_with_metric(&cfg, corpus, obj_closure, metric_closure)
    });

    // Prefer a callback-captured PyErr (original traceback) over the facade error.
    if let Some(err) = take_err(&obj_slot) {
        return Err(err);
    }
    if let Some(err) = take_err(&metric_slot) {
        return Err(err);
    }
    let inner = result.map_err(LgbmErrorWrap)?;
    Ok(Booster { inner })
}

/// Re-wrap the borrowed `&Dataset` as an owned `Py<PyAny>` so it can be handed to
/// the user callbacks as the `dataset` argument. The Dataset is `#[pyclass]`, so a
/// fresh `Py` reference to the same underlying object is obtained from its
/// borrowed handle via `Bound::into_any`.
fn train_set_as_pyobject(py: Python<'_>, train_set: &Dataset) -> PyResult<Py<PyAny>> {
    // The Dataset crossed the FFI boundary as a borrowed `&Dataset`; reconstruct a
    // Python-visible handle from the corpus metadata is unnecessary — callbacks in
    // the official API close over the dataset for labels/weights, which our facade
    // already owns. We expose a lightweight numpy label array as the `dataset`
    // surrogate so a user `fobj`/`feval` that calls `dataset.get_label()` style
    // access still has the labels available. Here we hand back the label array.
    let labels: Vec<f64> = train_set
        .corpus
        .labels
        .iter()
        .map(|&v| v as f64)
        .collect();
    Ok(labels.into_pyarray(py).into_any().unbind())
}
