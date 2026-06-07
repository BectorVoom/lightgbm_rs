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

use std::collections::HashMap;

use lgbm::{Booster as FacadeBooster, Config, RawCorpus};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dataset::Dataset;
use crate::error::LgbmErrorWrap;
use crate::marshal::numpy_dense_to_rows;

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

/// Coerce a single Python parameter value into the canonical string form the
/// facade `Config::from_params` consumes. BASIC coercion only (08-02 thinnest
/// slice — the full typed-value layer is D-08 in 08-05):
/// - `bool` -> `"true"` / `"false"` (lowercase, as the C++ config parser expects)
/// - `str`  -> verbatim
/// - everything else (int / float / ...) -> its Python `str()`
fn python_value_to_string(v: &Bound<'_, PyAny>) -> PyResult<String> {
    // `bool` MUST be tried before `int` — a Python bool is an `int` subclass, but
    // `extract::<bool>()` only matches genuine bool objects, so ordering is safe.
    if let Ok(b) = v.extract::<bool>() {
        return Ok(if b { "true".to_string() } else { "false".to_string() });
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(s);
    }
    v.str()?.extract::<String>()
}

/// Coerce a Python `params` dict into the `HashMap<String, String>` the facade
/// alias/extraction pipeline (`Config::from_params`) consumes.
fn params_dict_to_map(params: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::with_capacity(params.len());
    for (k, v) in params.iter() {
        let key: String = k.extract()?;
        map.insert(key, python_value_to_string(&v)?);
    }
    Ok(map)
}

/// Train a [`Booster`] from a `params` dict and a [`Dataset`] (mirrors
/// `lightgbm.train(params, train_set, num_boost_round)`).
///
/// `num_boost_round` takes precedence over any iteration count in `params`
/// (matching the official package), and is injected as the canonical
/// `num_iterations` before alias resolution. The corpus is already marshalled
/// (it lives in the owned `Dataset`), so training runs entirely with the GIL
/// RELEASED (`Python::detach`, D-13/SC#1).
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
    let mut map = params_dict_to_map(params)?;
    // num_boost_round wins over any params iteration alias (official-package
    // precedence). Setting the CANONICAL key makes `from_params`' alias pass keep
    // it (a directly-set canonical always beats an alias).
    map.insert("num_iterations".to_string(), num_boost_round.to_string());

    let cfg: Config = Config::from_params(&map).map_err(|e| LgbmErrorWrap(e.into()))?;
    let corpus: &RawCorpus = &train_set.corpus;

    // GIL RELEASED around the CPU-bound training (D-13/SC#1).
    let inner = py.detach(|| lgbm::train_raw(&cfg, corpus)).map_err(LgbmErrorWrap)?;
    Ok(Booster { inner })
}
