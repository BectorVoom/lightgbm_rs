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

use lgbm::{Booster as FacadeBooster, Config, RawCorpus};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
#[pyfunction]
#[pyo3(signature = (params, train_set, num_boost_round = 100))]
pub fn train(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    train_set: &Dataset,
    num_boost_round: i32,
) -> PyResult<Booster> {
    // Full D-06/07/08 coercion + validation; num_boost_round wins over any params
    // iteration alias (official-package precedence) via the canonical override.
    let cfg: Config = build_config_with_overrides(
        params,
        [("num_iterations", num_boost_round.to_string())],
    )?;
    let corpus: &RawCorpus = &train_set.corpus;

    // GIL RELEASED around the CPU-bound training (D-13/SC#1).
    let inner = py.detach(|| lgbm::train_raw(&cfg, corpus)).map_err(LgbmErrorWrap)?;
    Ok(Booster { inner })
}
