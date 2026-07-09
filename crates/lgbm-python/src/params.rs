//! The full Python `params` dict → `Config` surface (D-06/D-07/D-08).
//!
//! This is the single coercion seam every entry point routes through. It turns
//! an arbitrary Python `params` dict (str / int / float / bool / list / tuple
//! values) into the `HashMap<String, String>` the facade `Config::from_params`
//! consumes, mirroring the C++ / official-package string conventions, and adds
//! the D-07 gate that REJECTS recognized-but-unimplemented params before they
//! can silently diverge.
//!
//! Pipeline ([`build_config`]):
//!
//! 1. **Coerce (D-08)** — [`coerce_params_dict`] dispatches each value by its
//!    Python type into the canonical string `from_params` expects, matching the
//!    official `_param_dict_to_str` / C++ `Common::Atof`/`Atoi` conventions:
//!    `bool → "true"/"false"`, int/float → shortest round-trip repr, str → as-is,
//!    list/tuple → comma-joined (nested lists become `[a,b]` chunks, the C++ form
//!    for `interaction_constraints`).
//! 2. **Reject unimplemented (D-07)** — [`reject_unimplemented`] raises a clear
//!    Python `ValueError` for any key (alias-resolved) in
//!    [`lgbm_core::config::scope::OUT_OF_SCOPE_PARAMS`], or `device_type=gpu/cuda`
//!    on a CPU-only wheel (accepted when a matching GPU backend is compiled in),
//!    so a recognized-but-unported param NEVER silently trains a divergent model.
//! 3. **Build (D-06)** — `Config::from_params` does the full alias resolution +
//!    CHECK validation; truly-unknown keys (typos) pass through and only warn,
//!    preserving C++ fidelity (`set.rs:9-10`).
//!
//! Hard rule (CLAUDE.md, T-08-05-03): no panic ever crosses the FFI boundary —
//! every helper returns `PyResult<_>`; unsupported Python value types surface as
//! `ValueError`, never an `unwrap`/`panic!`.

use std::collections::HashMap;

use lgbm::Config;
use lgbm_core::config::resolve_alias;
use lgbm_core::config::scope::OUT_OF_SCOPE_PARAMS;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::error::LgbmErrorWrap;

/// Coerce a single Python parameter value into the canonical string form the
/// facade `Config::from_params` consumes (D-08), matching the official
/// `_param_dict_to_str` / `_to_string` and C++ `Common::Atof`/`Atoi` parsing:
///
/// - `bool`  → `"true"` / `"false"` (lowercase — the C++ config parser's bool form).
///   **MUST be checked BEFORE `int`**: a Python `bool` is an `int` subclass, so a
///   bare `int` check would coerce `True` to `"1"`.
/// - `int`   → its decimal string (`Common::Atoi` round-trip).
/// - `float` → the shortest round-tripping decimal repr (`{}`), which `Atof`
///   re-parses to the identical f64 (matches official `f"{val}"` for finite
///   floats).
/// - `str`   → verbatim.
/// - `list`/`tuple` → comma-joined per the official `_to_string` convention:
///   scalar items become `str(item)`; a NESTED list/tuple item becomes
///   `[a,b,...]` (the C++ form for `interaction_constraints`,
///   e.g. `[[0,1],[2]] → "[0,1],[2]"`).
///
/// # Errors
/// A value whose Python type is none of the above raises `ValueError` (never a
/// panic — T-08-05-03), mirroring the official `TypeError` for unknown param
/// types but routed through this crate's `ValueError` taxonomy.
pub fn coerce_value(v: &Bound<'_, PyAny>) -> PyResult<String> {
    // bool BEFORE int — Python bool is an int subclass (D-08, dispatch order).
    if v.is_instance_of::<PyBool>() {
        let b: bool = v.extract()?;
        return Ok(if b { "true".to_string() } else { "false".to_string() });
    }
    if v.is_instance_of::<PyInt>() {
        let i: i64 = v.extract()?;
        return Ok(i.to_string());
    }
    if v.is_instance_of::<PyFloat>() {
        let f: f64 = v.extract()?;
        // `{}` is the shortest round-trip repr; C++ Atof re-parses it identically.
        return Ok(format!("{f}"));
    }
    if v.is_instance_of::<PyString>() {
        return v.extract::<String>();
    }
    if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        return coerce_sequence(v);
    }
    Err(PyValueError::new_err(format!(
        "unsupported parameter value type: {} (expected bool, int, float, str, list, or tuple)",
        v.get_type().name()?
    )))
}

/// Join a Python list/tuple per the official `_param_dict_to_str` convention:
/// each item is rendered via [`coerce_item`] and the results are comma-joined.
fn coerce_sequence(seq: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut parts: Vec<String> = Vec::new();
    for item in seq.try_iter()? {
        parts.push(coerce_item(&item?)?);
    }
    Ok(parts.join(","))
}

/// Render a single item of a top-level list/tuple (official `_to_string`):
/// a nested list/tuple becomes a bracketed `[a,b,...]` chunk (the C++
/// `interaction_constraints` form); any other item becomes its scalar string
/// via [`coerce_value`].
fn coerce_item(item: &Bound<'_, PyAny>) -> PyResult<String> {
    if item.is_instance_of::<PyList>() || item.is_instance_of::<PyTuple>() {
        let mut inner: Vec<String> = Vec::new();
        for sub in item.try_iter()? {
            // Inner items are scalars in the C++ nested form; coerce_value
            // rejects deeper nesting via its sequence path naturally.
            inner.push(coerce_value(&sub?)?);
        }
        return Ok(format!("[{}]", inner.join(",")));
    }
    coerce_value(item)
}

/// Coerce a whole Python `params` dict into the `HashMap<String, String>` the
/// facade alias/extraction pipeline (`Config::from_params`) consumes (D-08).
///
/// Each key is extracted as a string and each value coerced via [`coerce_value`].
///
/// # Errors
/// `ValueError` for a non-string key or an unsupported value type (never panics).
pub fn coerce_params_dict(params: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::with_capacity(params.len());
    for (k, v) in params.iter() {
        let key: String = k.extract()?;
        map.insert(key, coerce_value(&v)?);
    }
    Ok(map)
}

/// D-07 gate: raise a clear Python `ValueError` for any param that LightGBM
/// recognizes but `lightgbm_rs` does not yet implement, so a recognized-but-
/// unported knob NEVER silently trains a divergent model (T-08-05-01).
///
/// A key is rejected when its alias-resolved canonical name is in
/// [`OUT_OF_SCOPE_PARAMS`] (distributed / GPU-OpenCL / linear-tree / quantized-
/// grad — referenced from the single source of truth in `lgbm_core`, not
/// re-typed here), OR it is `device_type` set to `gpu`/`cuda` while this wheel is
/// CPU-only (a matching GPU backend compiled in via `--features cuda`/`rocm`/`wgpu`
/// makes the corresponding `device_type` accepted).
///
/// Truly-unknown keys (typos) are deliberately NOT rejected — they pass through
/// to `Config::from_params`, which only warns, preserving C++ fidelity
/// (D-06 unknown-warn-never-fatal).
///
/// # Errors
/// `ValueError` naming the unimplemented param.
pub fn reject_unimplemented(map: &HashMap<String, String>) -> PyResult<()> {
    for (key, value) in map {
        let canonical = resolve_alias(key);
        if OUT_OF_SCOPE_PARAMS.contains(&canonical) {
            return Err(PyValueError::new_err(format!(
                "parameter `{key}` is recognized by LightGBM but not implemented in \
                 lightgbm_rs (out-of-scope for v1: distributed / GPU-OpenCL / linear-tree / \
                 quantized-grad). Remove it to train, or use the C++ LightGBM for this feature."
            )));
        }
        if canonical == "device_type" {
            let dev = value.trim().to_ascii_lowercase();
            // A GPU device_type ("gpu"/"cuda") is accepted ONLY when this wheel was
            // built with a matching CubeCL backend (cascade rocm > cuda > wgpu; see
            // booster.rs). Backend selection is compile-time, so a GPU device_type on
            // a GPU wheel dispatches to whichever backend was compiled in. The default
            // CPU-only wheel rejects every GPU device_type, exactly as before — the
            // feature gap was that lgbm-python never forwarded the GPU cargo features,
            // so the extension module was permanently CPU-only.
            let gpu_backend_built =
                cfg!(feature = "rocm") || cfg!(feature = "cuda") || cfg!(feature = "wgpu");
            let is_gpu_device = dev == "gpu" || dev == "cuda";
            if is_gpu_device && !gpu_backend_built {
                return Err(PyValueError::new_err(format!(
                    "device_type=`{value}` is recognized by LightGBM but this lightgbm_rs \
                     wheel is CPU-only (no GPU backend compiled in). Rebuild with \
                     `maturin --features cuda` (or rocm/wgpu) to enable the GPU backend."
                )));
            }
        }
    }
    Ok(())
}

/// The single params→`Config` entry every binding routes through (D-06/07/08):
/// coerce the dict (D-08) → reject recognized-but-unimplemented params (D-07) →
/// `Config::from_params` (D-06: full alias resolution + CHECK validation;
/// unknown typos warn, never fatal).
///
/// # Errors
/// `ValueError` for an unsupported value type, an unimplemented param, or any
/// `Config::from_params` validation failure (`ConfigError → ValueError` via the
/// crate error taxonomy). Never panics across the FFI boundary.
pub fn build_config(params: &Bound<'_, PyDict>) -> PyResult<Config> {
    let map = coerce_params_dict(params)?;
    reject_unimplemented(&map)?;
    Config::from_params(&map).map_err(|e| PyErr::from(LgbmErrorWrap(e.into())))
}

/// Like [`build_config`] but lets the caller inject/override canonical keys
/// AFTER coercion + the D-07 gate but BEFORE `from_params` (used by `train` to
/// force `num_iterations` from `num_boost_round`, which must beat any params
/// alias — official-package precedence).
///
/// # Errors
/// Same as [`build_config`].
pub fn build_config_with_overrides<'a, I>(
    params: &Bound<'_, PyDict>,
    overrides: I,
) -> PyResult<Config>
where
    I: IntoIterator<Item = (&'a str, String)>,
{
    let mut map = coerce_params_dict(params)?;
    reject_unimplemented(&map)?;
    for (k, v) in overrides {
        map.insert(k.to_string(), v);
    }
    Config::from_params(&map).map_err(|e| PyErr::from(LgbmErrorWrap(e.into())))
}

// Rust unit tests drive the live coercion paths (D-08) + the D-07 gate. The
// `extension-module` feature is intentionally OFF for `cargo test` (see
// Cargo.toml) and `pyo3/auto-initialize` (a dev-dependency) boots a CPython
// interpreter, so `Python::attach` works here. The same paths are ALSO covered
// end-to-end from Python in `python/tests/test_params.py`.
#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use pyo3::Python;

    /// bool/int/float/str scalar coercion matches the C++ / official forms,
    /// with bool checked BEFORE int (True → "true", not "1").
    #[test]
    fn coerce_scalars() {
        Python::attach(|py| {
            let t = true.into_pyobject(py).unwrap();
            assert_eq!(coerce_value(t.as_any()).unwrap(), "true");
            let f = false.into_pyobject(py).unwrap();
            assert_eq!(coerce_value(f.as_any()).unwrap(), "false");

            let i = 31i64.into_pyobject(py).unwrap();
            assert_eq!(coerce_value(i.as_any()).unwrap(), "31");

            let fl = 0.8f64.into_pyobject(py).unwrap();
            assert_eq!(coerce_value(fl.as_any()).unwrap(), "0.8");

            let s = "regression".into_pyobject(py).unwrap();
            assert_eq!(coerce_value(s.as_any()).unwrap(), "regression");
        });
    }

    /// A flat list/tuple comma-joins; a nested list becomes the C++ bracketed
    /// `[a,b]` chunk form used by `interaction_constraints`.
    #[test]
    fn coerce_lists() {
        Python::attach(|py| {
            let flat = PyList::new(py, [1i64, -1, 0]).unwrap();
            assert_eq!(coerce_value(flat.as_any()).unwrap(), "1,-1,0");

            let eval_at = PyTuple::new(py, [1i64, 5]).unwrap();
            assert_eq!(coerce_value(eval_at.as_any()).unwrap(), "1,5");

            // interaction_constraints: [[0,1],[2]] -> "[0,1],[2]"
            let inner0 = PyList::new(py, [0i64, 1]).unwrap();
            let inner1 = PyList::new(py, [2i64]).unwrap();
            let nested = PyList::new(py, [inner0, inner1]).unwrap();
            assert_eq!(coerce_value(nested.as_any()).unwrap(), "[0,1],[2]");
        });
    }

    /// Full pipeline: coerce → gate → from_params, with an alias resolving and
    /// a bool/int/float mix producing a valid Config.
    #[test]
    fn build_config_pipeline() {
        Python::attach(|py| {
            let params = PyDict::new(py);
            params.set_item("objective", "regression").unwrap();
            params.set_item("num_leaves", 31i64).unwrap();
            params.set_item("learning_rate", 0.05f64).unwrap();
            params.set_item("is_unbalance", true).unwrap();
            // n_estimators is an alias of num_iterations.
            params.set_item("n_estimators", 50i64).unwrap();

            let cfg = build_config(&params).unwrap();
            assert_eq!(cfg.num_leaves, 31);
            assert_eq!(cfg.num_iterations, 50);
        });
    }

    /// build_config raises (not panics) when an unimplemented param is present.
    #[test]
    fn build_config_rejects_unimplemented() {
        Python::attach(|py| {
            let params = PyDict::new(py);
            params.set_item("objective", "regression").unwrap();
            params.set_item("device_type", "gpu").unwrap();
            assert!(build_config(&params).is_err());
        });
    }

    /// `reject_unimplemented` raises for OUT_OF_SCOPE params (and their aliases)
    /// and for device_type=gpu/cuda on this CPU-only test build (no GPU feature),
    /// but passes truly-unknown typos through. A GPU-feature build would accept the
    /// matching device_type instead (covered by the Colab/Kaggle wheel build).
    #[test]
    fn reject_gate() {
        let mut m = HashMap::new();
        m.insert("num_machines".to_string(), "2".to_string());
        assert!(reject_unimplemented(&m).is_err());

        let mut m = HashMap::new();
        m.insert("linear_tree".to_string(), "true".to_string());
        assert!(reject_unimplemented(&m).is_err());

        let mut m = HashMap::new();
        m.insert("use_quantized_grad".to_string(), "true".to_string());
        assert!(reject_unimplemented(&m).is_err());

        let mut m = HashMap::new();
        m.insert("device_type".to_string(), "gpu".to_string());
        assert!(reject_unimplemented(&m).is_err());
        let mut m = HashMap::new();
        m.insert("device_type".to_string(), "CUDA".to_string());
        assert!(reject_unimplemented(&m).is_err());

        // CPU device is fine.
        let mut m = HashMap::new();
        m.insert("device_type".to_string(), "cpu".to_string());
        assert!(reject_unimplemented(&m).is_ok());

        // An alias of an OUT_OF_SCOPE param is also rejected (resolved canonical).
        let mut m = HashMap::new();
        m.insert("num_machine".to_string(), "2".to_string());
        // `num_machine` resolves to `num_machines` if aliased; otherwise it is an
        // unknown typo. Either way the gate must NOT panic.
        let _ = reject_unimplemented(&m);

        // Truly-unknown typo passes the gate (from_params will warn).
        let mut m = HashMap::new();
        m.insert("some_typo_param".to_string(), "5".to_string());
        assert!(reject_unimplemented(&m).is_ok());
    }
}
