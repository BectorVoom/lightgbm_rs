//! The device-capability seam: which `device_type` values THIS wheel can
//! actually train on, and the introspection helper that reports it to Python.
//!
//! Backend selection is a RUNTIME choice (`lgbm`'s `dispatch_device_backend`
//! matches on `Config::device_type`), but each GPU backend is still compiled in
//! behind a cargo feature so the default wheel needs no GPU toolchain. Those two
//! facts together mean a wheel has a *capability set* — the devices it was built
//! to reach — and the D-07 gate is a membership test against that set rather
//! than the flat "reject every GPU" rule it used to be.
//!
//! Keeping the set in ONE place (rather than re-deriving `cfg!` flags at each
//! call site) is what lets [`build_capabilities`] report exactly the same answer
//! the gate enforces — an introspection helper that could disagree with the gate
//! would be worse than none.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The `device_type` values this wheel can train on, in the order they are
/// reported to Python.
///
/// `cpu` is unconditional: `lgbm-compute` keeps its `cpu` feature on in every
/// build, so the f64 deterministic anchor is always reachable. Each GPU device is
/// present only when its cargo feature compiled the matching CubeCL backend in.
/// `gpu` is served by ROCm first and WGPU otherwise, mirroring the facade's
/// dispatch precedence.
#[must_use]
pub fn supported_devices() -> Vec<&'static str> {
    let mut out = vec!["cpu"];
    if cfg!(any(feature = "rocm", feature = "wgpu")) {
        out.push("gpu");
    }
    if cfg!(feature = "cuda") {
        out.push("cuda");
    }
    out
}

/// Whether `device` (a canonical, lowercased `device_type`) is trainable here.
#[must_use]
pub fn device_supported(device: &str) -> bool {
    supported_devices().contains(&device)
}

/// Whether `device` is a `device_type` lightgbm_rs knows at all — i.e. one of the
/// C++ closed enum's values — regardless of whether THIS wheel can reach it.
///
/// Separating "unknown spelling" from "known but not compiled in" keeps the two
/// error messages honest: an unknown value is `Config::from_params`' typed
/// `UnknownValue` (identical to the pure-Rust path), while a known-but-absent
/// device gets the actionable rebuild instruction.
#[must_use]
pub fn device_is_known(device: &str) -> bool {
    matches!(device, "cpu" | "gpu" | "cuda")
}

/// The CubeCL backend name backing each supported GPU device, for reporting.
///
/// `None` on a CPU-only wheel. The precedence (`rocm` before `wgpu` for the `gpu`
/// slot) matches the facade's dispatch arms.
#[must_use]
pub fn gpu_backend_name() -> Option<&'static str> {
    if cfg!(feature = "rocm") {
        Some("rocm")
    } else if cfg!(feature = "cuda") {
        Some("cuda")
    } else if cfg!(feature = "wgpu") {
        Some("wgpu")
    } else {
        None
    }
}

/// The cargo feature that would add `device` to this wheel's capability set —
/// used to make the D-07 rejection message actionable.
#[must_use]
pub fn feature_for_device(device: &str) -> &'static str {
    match device {
        "cuda" => "cuda",
        // The generic `gpu` slot is served by ROCm first, WGPU otherwise.
        "gpu" => "rocm` (or `wgpu",
        _ => "cpu",
    }
}

/// `lightgbm_rs.get_device_capabilities()` — report what this wheel can train on.
///
/// Returns a dict with:
/// - `devices`: the accepted `device_type` values (always includes `"cpu"`),
/// - `gpu_backend`: the CubeCL backend name compiled in, or `None`,
/// - `default_device`: the `device_type` used when params do not set one,
/// - `gpu_device_id`: whether `gpu_device_id` selects a device on this wheel,
/// - `inapplicable_params`: recognized device params accepted for compatibility
///   that have no effect on the CubeCL backends.
///
/// The `devices` list is the SAME set the params gate enforces, so callers can
/// branch on it without guessing (`"cuda" in caps["devices"]`).
///
/// # Errors
/// Only propagates dict-insertion failures; never panics across the FFI boundary.
#[pyfunction]
pub fn get_device_capabilities(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("devices", supported_devices())?;
    d.set_item("gpu_backend", gpu_backend_name())?;
    d.set_item("default_device", "cpu")?;
    // `gpu_device_id` maps onto the CubeCL device index for HIP/CUDA. WGPU picks
    // its adapter through `WgpuDevice` rather than a flat index, so it cannot honor
    // the knob and must not claim to.
    d.set_item(
        "gpu_device_id",
        cfg!(any(feature = "rocm", feature = "cuda")),
    )?;
    // Accepted so official LightGBM param dicts port over unchanged, but inert:
    // the CubeCL backends have no OpenCL platform dimension, and kernel precision
    // is fixed by the ~1e-6 parity contract rather than switchable.
    d.set_item("inapplicable_params", vec!["gpu_platform_id", "gpu_use_dp"])?;
    Ok(d.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_supported() {
        assert!(device_supported("cpu"));
        assert!(supported_devices().contains(&"cpu"));
    }

    /// The default test build compiles no GPU backend, so the capability set must
    /// be CPU-only and `gpu_backend` absent. (A `--features cuda` build flips both;
    /// the assertions are cfg-guarded so they pin the build they describe.)
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]
    #[test]
    fn cpu_only_wheel_reports_no_gpu() {
        assert_eq!(supported_devices(), vec!["cpu"]);
        assert_eq!(gpu_backend_name(), None);
        assert!(!device_supported("cuda"));
        assert!(!device_supported("gpu"));
    }

    #[test]
    fn every_supported_device_names_a_rebuild_feature() {
        for d in ["cpu", "gpu", "cuda"] {
            assert!(!feature_for_device(d).is_empty());
        }
    }
}
