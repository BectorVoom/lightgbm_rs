//! The CPU/GPU device hyperparameter seam — the typed reading of
//! `device_type` + `num_gpu` / `gpu_platform_id` / `gpu_device_id` /
//! `gpu_use_dp` that the `lgbm` facade dispatches on and the Python layer
//! reports.
//!
//! [`Config`] keeps the C++ field shapes verbatim (`device_type: String`, the
//! GPU knobs as `int`/`bool`) for parity with `config.h`. Everything that needs
//! to ACT on those fields — backend dispatch, device-index resolution, the
//! "this knob has no CubeCL analog" reporting — goes through the helpers here,
//! so the string literals `"cpu"` / `"gpu"` / `"cuda"` are matched in exactly
//! one place instead of being re-spelled at every call site.
//!
//! # Why a typed [`DeviceKind`] and not a bare string
//!
//! Backend dispatch is a closed 3-way choice, and getting it wrong means
//! silently training on the wrong device — the failure mode the parity contract
//! exists to prevent. An enum makes the facade's `match` exhaustive: adding a
//! device to [`crate::config::set`]'s `parse_device_type` fails compilation at
//! the dispatch site instead of falling into a `_ =>` fallback.

use super::Config;

/// The device a model trains on — the typed reading of `Config::device_type`.
///
/// The variants mirror the C++ closed enum (`config.cpp::GetDeviceType`:
/// `cpu` / `gpu` / `cuda`) one-for-one. `gpu` is the OpenCL device in C++ and
/// the ROCm/HIP CubeCL backend here; `cuda` is CUDA in both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// `device_type=cpu` — the native / cubecl-cpu f64 deterministic anchor and
    /// the project's hard merge gate.
    Cpu,
    /// `device_type=gpu` — the ROCm/HIP CubeCL backend (the C++ OpenCL slot).
    Gpu,
    /// `device_type=cuda` — the CUDA CubeCL backend.
    Cuda,
}

impl DeviceKind {
    /// The canonical `device_type` string for this device — the exact spelling
    /// `Config::device_type` holds and `parse_device_type` emits.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
            DeviceKind::Cuda => "cuda",
        }
    }

    /// Parse a CANONICAL `device_type` string (already lowercased + validated by
    /// `parse_device_type`). Returns `None` for anything else.
    ///
    /// This deliberately does NOT re-do the case folding or the alias handling —
    /// `Config::from_params` is the single validation gate, and a `Config` that
    /// exists always holds a canonical value.
    #[must_use]
    pub fn from_canonical(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(DeviceKind::Cpu),
            "gpu" => Some(DeviceKind::Gpu),
            "cuda" => Some(DeviceKind::Cuda),
            _ => None,
        }
    }

    /// Whether this device is a GPU (i.e. needs a compiled GPU backend).
    #[must_use]
    pub fn is_gpu(self) -> bool {
        matches!(self, DeviceKind::Gpu | DeviceKind::Cuda)
    }
}

impl Config {
    /// The typed device this config trains on.
    ///
    /// Falls back to [`DeviceKind::Cpu`] for an unrecognized string, which a
    /// `Config` built through `from_params` cannot hold (`parse_device_type`
    /// rejects unknown values with `ConfigError::UnknownValue`) — the fallback
    /// only covers a hand-constructed `Config` with a hand-set field, where
    /// defaulting to the deterministic CPU anchor is the safe reading.
    #[must_use]
    pub fn device_kind(&self) -> DeviceKind {
        DeviceKind::from_canonical(&self.device_type).unwrap_or(DeviceKind::Cpu)
    }

    /// The CubeCL device INDEX this config selects, resolved from
    /// `gpu_device_id`.
    ///
    /// `gpu_device_id < 0` (the `-1` default) resolves to device `0`, matching
    /// the C++ default resolution; any non-negative value is used verbatim as
    /// the index handed to `AmdDevice::new` / `CudaDevice::new`.
    #[must_use]
    pub fn gpu_device_index(&self) -> u32 {
        if self.gpu_device_id < 0 {
            0
        } else {
            // Non-negative i32 always fits u32.
            self.gpu_device_id as u32
        }
    }

    /// Human-readable warnings for device knobs that are ACCEPTED for
    /// official-package compatibility but have no effect on the CubeCL backends.
    ///
    /// Returned rather than logged so every caller decides how to surface them
    /// (the facade logs, the Python layer raises them into a `warnings.warn`).
    /// An empty vector means every device knob in this config is fully honored.
    ///
    /// Deliberately NOT warned about: `gpu_device_id` (honored as the CubeCL
    /// device index) and `num_gpu == 1` (the supported value; `> 1` is a hard
    /// validation error in `from_params`, not a warning).
    #[must_use]
    pub fn device_warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.gpu_platform_id >= 0 {
            out.push(format!(
                "gpu_platform_id={} has no effect: the CubeCL backends (HIP/CUDA) address \
                 devices by index within one runtime and have no OpenCL platform dimension. \
                 Select the device with `gpu_device_id` instead.",
                self.gpu_platform_id
            ));
        }
        if self.gpu_use_dp {
            out.push(
                "gpu_use_dp=true has no effect: the Rust GPU kernels use a FIXED precision \
                 split (f32 histogram cells, f64 gain/split-scan/subtract) that the ~1e-6 \
                 parity contract was validated against, so precision is not switchable. \
                 See docs/04-ROCM-GAPS.md."
                    .to_string(),
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_kind_round_trips_every_canonical() {
        for kind in [DeviceKind::Cpu, DeviceKind::Gpu, DeviceKind::Cuda] {
            assert_eq!(DeviceKind::from_canonical(kind.as_str()), Some(kind));
        }
        assert_eq!(DeviceKind::from_canonical("opencl"), None);
    }

    #[test]
    fn is_gpu_covers_both_gpu_devices() {
        assert!(!DeviceKind::Cpu.is_gpu());
        assert!(DeviceKind::Gpu.is_gpu());
        assert!(DeviceKind::Cuda.is_gpu());
    }

    #[test]
    fn default_config_is_cpu_device_zero_with_no_warnings() {
        let c = Config::default();
        assert_eq!(c.device_kind(), DeviceKind::Cpu);
        // -1 (the config.h default) resolves to device 0, as in C++.
        assert_eq!(c.gpu_device_index(), 0);
        assert!(c.device_warnings().is_empty());
    }

    // NOTE: this name deliberately says "device index", not the compute-runtime's
    // name — `lgbm-core` must not contain that token in a non-comment line
    // (the CMP-01 containment guard in `lgbm-compute/tests/cmp01_containment.rs`).
    #[test]
    fn gpu_device_id_maps_to_the_device_index() {
        let c = Config {
            gpu_device_id: 3,
            ..Default::default()
        };
        assert_eq!(c.gpu_device_index(), 3);
    }

    #[test]
    fn inapplicable_knobs_warn_but_do_not_fail() {
        let c = Config {
            gpu_platform_id: 0,
            gpu_use_dp: true,
            ..Default::default()
        };
        let w = c.device_warnings();
        assert_eq!(w.len(), 2, "both inapplicable knobs must be reported: {w:?}");
        assert!(w[0].contains("gpu_platform_id"));
        assert!(w[1].contains("gpu_use_dp"));
    }
}
