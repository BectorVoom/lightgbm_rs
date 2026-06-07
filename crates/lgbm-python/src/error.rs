//! Error mapping at the PyO3 boundary: `lgbm::LgbmError` -> a Python exception
//! (08-RESEARCH Pattern 6 error taxonomy). The official-package-mirroring
//! `lightgbm_rs.LightGBMError` is the engine-error type; user-input defects map
//! to the standard Python `ValueError`.
//!
//! Hard rule (CLAUDE.md, T-08-02-03): no panic ever crosses the FFI boundary —
//! every fallible binding returns `PyResult<_>` and routes its `LgbmError`
//! through [`from_lgbm_error`].

use lgbm::LgbmError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pyo3::create_exception!(
    lightgbm_rs,
    LightGBMError,
    pyo3::exceptions::PyException,
    "Engine-side LightGBM-rs error (objective / metric / model / boosting / I/O / custom-metric)."
);

/// Map a facade [`LgbmError`] to a Python exception per the Pattern-6 taxonomy:
///
/// - `Config` / `InvalidCorpus` / `InvalidConstraintLength` -> `ValueError`
///   (these are caller-input defects — bad params or malformed data).
/// - `Objective` / `Metric` / `Model` / `Boosting` / `Io` / `CustomMetric`
///   -> `lightgbm_rs.LightGBMError` (engine-side failures, mirroring the
///   official package's `LightGBMError`).
pub fn from_lgbm_error(err: LgbmError) -> PyErr {
    let msg = err.to_string();
    match err {
        LgbmError::Config(_)
        | LgbmError::InvalidCorpus { .. }
        | LgbmError::InvalidConstraintLength { .. } => PyValueError::new_err(msg),
        LgbmError::Objective(_)
        | LgbmError::Metric(_)
        | LgbmError::Model(_)
        | LgbmError::Boosting(_)
        | LgbmError::Io { .. }
        | LgbmError::CustomMetric { .. } => LightGBMError::new_err(msg),
    }
}

/// `From<LgbmError> for PyErr` so binding code can use `?` on any facade call.
impl From<crate::error::LgbmErrorWrap> for PyErr {
    fn from(w: crate::error::LgbmErrorWrap) -> PyErr {
        from_lgbm_error(w.0)
    }
}

/// Newtype wrapper enabling the orphan-rule-safe `From<..> for PyErr` impl above
/// (both `LgbmError` and `PyErr` are foreign types). Binding code converts a
/// facade `Result` via `.map_err(LgbmErrorWrap)?`.
pub struct LgbmErrorWrap(pub LgbmError);

impl From<LgbmError> for LgbmErrorWrap {
    fn from(e: LgbmError) -> Self {
        LgbmErrorWrap(e)
    }
}
