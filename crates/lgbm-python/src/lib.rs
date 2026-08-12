//! `lightgbm_rs._core` — the compiled PyO3 extension (D-11/D-12).
//!
//! This crate is the thin FFI seam over the validated `lgbm` facade: it marshals
//! numpy input into owned Rust buffers, releases the GIL (`Python::detach`,
//! D-13/SC#1) around the CPU-bound train/predict, and returns owned numpy arrays.
//! It adds NO algorithm — every method delegates one layer down to `lgbm`.
//!
//! Cross-cutting invariants (CLAUDE.md): every `#[pymethods]`/`#[pyfn]` returns
//! `PyResult<_>`; `LgbmError` maps to a Python exception via
//! [`error::from_lgbm_error`]; no `unwrap`/`expect`/`panic!` crosses the FFI
//! boundary.

use pyo3::prelude::*;

/// Route Rust-side allocations through mimalloc inside the extension module. This
/// is parity-neutral (allocator choice does not affect floating-point results) and
/// only governs this cdylib's Rust allocations, never CPython's object allocator.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod booster;
pub mod callbacks;
pub mod dataset;
pub mod device;
pub mod error;
pub mod marshal;
pub mod params;

use booster::{train, Booster};
use dataset::Dataset;
use device::get_device_capabilities;
use error::LightGBMError;

/// The compiled extension module. Registers the `Dataset`/`Booster` classes, the
/// `train` free function, the `get_device_capabilities` introspection helper, and
/// the `LightGBMError` exception so the thin Python package can re-export them as
/// `lightgbm_rs.{Dataset,Booster,train,get_device_capabilities}`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<Booster>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_capabilities, m)?)?;
    m.add("LightGBMError", m.py().get_type::<LightGBMError>())?;
    Ok(())
}
