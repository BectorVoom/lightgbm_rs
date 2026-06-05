//! Split-gain math — a VERBATIM transcription of the C++ LightGBM gain formula
//! from `LightGBM/src/treelearner/feature_histogram.hpp:711-845` (commit
//! 195c26fc, VERSION 4.6.0.99).
//!
//! Per decision D-01a (04-CONTEXT.md), the gain math is implemented EARLY here
//! (Phase-5 TRL-04 consumes it, does not re-derive it) and lives INSIDE the
//! `find_best_split` kernel (D-01 whole-kernel op). This module provides:
//!
//! - The four scalar gain primitives ([`threshold_l1`], [`get_leaf_gain`],
//!   [`get_split_gains`], [`calculate_splitted_leaf_output`]) as `#[cube]`
//!   functions so they can be called from inside the
//!   [`crate::kernels::split`] scan kernel AND as plain host functions (a
//!   `#[cube]` function is also a regular Rust `fn`).
//! - [`GainConfig`] — the minimal comptime-friendly surface extracting the seven
//!   gain-relevant fields from `lgbm_core::Config` (we do NOT pass `&Config`
//!   into the kernel).
//! - [`SplitInfo`] — the result struct mirroring the C++ `SplitInfo`
//!   (`split_info.hpp`) fields the scan writes.
//!
//! ## Numerical contract
//! All gain math is computed in `f64` (C++ `double`), matching the reference.
//! The `kEpsilon = 1e-15f` algorithm constant is reused from
//! `lgbm_core::types::K_EPSILON` (an `f32` literal); the scan widens it to `f64`
//! exactly as C++ does (`double sum_right_hessian = kEpsilon;` promotes the
//! `float` constant to `double`). We never redefine `1e-15` / `1e-35` locally.
//!
//! ## Scope
//! Only the `USE_L1=false/true`, `USE_MAX_OUTPUT=false`, `USE_SMOOTHING=false`,
//! `USE_MC=false` (no monotone constraints) template instantiation is
//! transcribed — that is the default CPU path. `max_delta_step` (output
//! clamping), `path_smooth` (smoothing), and monotone constraints are Phase-7+
//! scope and are validated to be at their no-op defaults by the launcher.

use cubecl::prelude::*;

/// `static double ThresholdL1(double s, double l1)` (feature_histogram.hpp:711):
///
/// ```cpp
/// const double reg_s = std::max(0.0, std::fabs(s) - l1);
/// return Common::Sign(s) * reg_s;
/// ```
///
/// `Common::Sign(x) = (x > 0) - (x < 0)` (common.h:872) — i.e. `+1`, `-1`, or
/// `0`. We reproduce it with branch-free integer-flavored arithmetic in `f64`.
#[cube]
pub fn threshold_l1(s: f64, l1: f64) -> f64 {
    let reg_s = f64::max(0.0, f64::abs(s) - l1);
    // Sign(s) = (s > 0) - (s < 0), as f64. Expressed with branchless `select`
    // because the `if cond { 1.0 } else { 0.0 }` form mis-lowers to a constant on
    // cubecl-cpu (0.10.0) — it returned 0 for s<0, zeroing every L1 gain.
    let pos = select(s > 0.0, 1.0, 0.0);
    let neg = select(s < 0.0, 1.0, 0.0);
    (pos - neg) * reg_s
}

/// `GetLeafGain<USE_L1, false, false>` (feature_histogram.hpp:799-815, the
/// `!USE_MAX_OUTPUT && !USE_SMOOTHING` fast path):
///
/// ```cpp
/// if (USE_L1) {
///   const double sg_l1 = ThresholdL1(sum_gradients, l1);
///   return (sg_l1 * sg_l1) / (sum_hessians + l2);
/// } else {
///   return (sum_gradients * sum_gradients) / (sum_hessians + l2);
/// }
/// ```
///
/// `use_l1` is the runtime flag (`l1 != 0`); it selects the L1 branch exactly
/// as the C++ `USE_L1` template bool does.
#[cube]
pub fn get_leaf_gain(use_l1: bool, sum_gradients: f64, sum_hessians: f64, l1: f64, l2: f64) -> f64 {
    if use_l1 {
        let sg_l1 = threshold_l1(sum_gradients, l1);
        (sg_l1 * sg_l1) / (sum_hessians + l2)
    } else {
        (sum_gradients * sum_gradients) / (sum_hessians + l2)
    }
}

/// `GetSplitGains<false, USE_L1, false, false>` (feature_histogram.hpp:757-797,
/// the `!USE_MC` branch):
///
/// ```cpp
/// return GetLeafGain<...>(sum_left_gradients, sum_left_hessians, ...) +
///        GetLeafGain<...>(sum_right_gradients, sum_right_hessians, ...);
/// ```
#[cube]
pub fn get_split_gains(
    use_l1: bool,
    sum_left_gradients: f64,
    sum_left_hessians: f64,
    sum_right_gradients: f64,
    sum_right_hessians: f64,
    l1: f64,
    l2: f64,
) -> f64 {
    get_leaf_gain(use_l1, sum_left_gradients, sum_left_hessians, l1, l2)
        + get_leaf_gain(use_l1, sum_right_gradients, sum_right_hessians, l1, l2)
}

/// `CalculateSplittedLeafOutput<USE_L1, false, false>`
/// (feature_histogram.hpp:716-738, the `!USE_MAX_OUTPUT && !USE_SMOOTHING`
/// path):
///
/// ```cpp
/// if (USE_L1) {
///   ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
/// } else {
///   ret = -sum_gradients / (sum_hessians + l2);
/// }
/// ```
#[cube]
pub fn calculate_splitted_leaf_output(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
) -> f64 {
    if use_l1 {
        -threshold_l1(sum_gradients, l1) / (sum_hessians + l2)
    } else {
        -sum_gradients / (sum_hessians + l2)
    }
}

/// The minimal gain-config surface passed into [`crate::Backend::find_best_split`]
/// (extracted from `lgbm_core::Config`; we do NOT pass `&Config` into the
/// kernel — keep it small and `Copy`).
///
/// Fields map 1:1 to the `meta_->config->*` accesses in the scan. Only the
/// default-path subset is honored at the kernel layer; `max_delta_step` /
/// `path_smooth` are carried for completeness + validated no-op by the launcher
/// (Phase-7+ scope).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainConfig {
    /// `int min_data_in_leaf` (config default 20).
    pub min_data_in_leaf: i32,
    /// `double min_sum_hessian_in_leaf` (config default 1e-3).
    pub min_sum_hessian_in_leaf: f64,
    /// `double max_delta_step` (config default 0.0 — output clamping, Phase-7+).
    pub max_delta_step: f64,
    /// `double lambda_l1` (config default 0.0).
    pub lambda_l1: f64,
    /// `double lambda_l2` (config default 0.0).
    pub lambda_l2: f64,
    /// `double min_gain_to_split` (config default 0.0).
    pub min_gain_to_split: f64,
    /// `double path_smooth` (config default 0.0 — smoothing, Phase-7+).
    pub path_smooth: f64,
}

impl GainConfig {
    /// Extract the seven gain-relevant fields from a `lgbm_core::Config`.
    pub fn from_config(c: &lgbm_core::Config) -> Self {
        Self {
            min_data_in_leaf: c.min_data_in_leaf,
            min_sum_hessian_in_leaf: c.min_sum_hessian_in_leaf,
            max_delta_step: c.max_delta_step,
            lambda_l1: c.lambda_l1,
            lambda_l2: c.lambda_l2,
            min_gain_to_split: c.min_gain_to_split,
            path_smooth: c.path_smooth,
        }
    }

    /// True if the L1 branch (`USE_L1`) is active, i.e. `lambda_l1 != 0`.
    /// LightGBM selects `USE_L1` at `config->lambda_l1 > 0`
    /// (`feature_histogram.cpp` template dispatch); we mirror `> 0`.
    pub fn use_l1(&self) -> bool {
        self.lambda_l1 > 0.0
    }
}

/// The best-split result the scan produces, mirroring the C++ `SplitInfo`
/// (`split_info.hpp:22-54`) fields written by `FindBestThresholdSequentially`
/// + the `FindBestThreshold` finalization (`feature_histogram.hpp:1031-1056`).
///
/// `gain == kMinScore` (`-inf`) signals "no valid split found" (the C++
/// `output->gain` initial value, untouched when no candidate clears the gates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitInfo {
    /// `uint32_t threshold` — the winning bin threshold (`t-1+offset` for the
    /// reverse branch, `t+offset` for the forward branch).
    pub threshold: u32,
    /// `double gain` — the split gain net of `min_gain_shift`, times penalty.
    /// `f64::NEG_INFINITY` (== C++ `kMinScore`) when no split was found.
    pub gain: f64,
    /// `data_size_t left_count` — rows routed left.
    pub left_count: i32,
    /// `data_size_t right_count` — rows routed right.
    pub right_count: i32,
    /// `double left_sum_gradient`.
    pub left_sum_gradient: f64,
    /// `double left_sum_hessian` (the C++ value already has `kEpsilon`
    /// subtracted back off — see `feature_histogram.hpp:1042`).
    pub left_sum_hessian: f64,
    /// `double right_sum_gradient`.
    pub right_sum_gradient: f64,
    /// `double right_sum_hessian` (with `kEpsilon` subtracted back off).
    pub right_sum_hessian: f64,
    /// `double left_output`.
    pub left_output: f64,
    /// `double right_output`.
    pub right_output: f64,
    /// `bool default_left` — true iff the winner came from the REVERSE branch
    /// (`output->default_left = REVERSE;`, feature_histogram.hpp:1055).
    pub default_left: bool,
}

impl SplitInfo {
    /// The "no split found" sentinel: `gain == kMinScore` (`-inf`), every other
    /// field at its C++ default. Mirrors the freshly-constructed C++ `SplitInfo`
    /// after `FindBestThreshold` sets `output->gain = kMinScore` and finds no
    /// improving candidate.
    pub fn none() -> Self {
        Self {
            threshold: 0,
            gain: f64::NEG_INFINITY,
            left_count: 0,
            right_count: 0,
            left_sum_gradient: 0.0,
            left_sum_hessian: 0.0,
            right_sum_gradient: 0.0,
            right_sum_hessian: 0.0,
            left_output: 0.0,
            right_output: 0.0,
            default_left: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_l1_matches_sign_times_relu() {
        // Sign(s) * max(0, |s| - l1).
        assert_eq!(threshold_l1(5.0, 2.0), 3.0);
        assert_eq!(threshold_l1(-5.0, 2.0), -3.0);
        assert_eq!(threshold_l1(1.0, 2.0), 0.0); // |s| - l1 < 0 -> 0
        assert_eq!(threshold_l1(0.0, 2.0), 0.0); // Sign(0) == 0
    }

    #[test]
    fn leaf_gain_l1_vs_no_l1() {
        // No-L1: g^2 / (h + l2).
        assert_eq!(get_leaf_gain(false, 4.0, 2.0, 0.0, 0.0), 8.0);
        // L1: ThresholdL1(g,l1)^2 / (h + l2). g=4, l1=1 -> 3; 9/2 = 4.5.
        assert_eq!(get_leaf_gain(true, 4.0, 2.0, 1.0, 0.0), 4.5);
    }

    #[test]
    fn split_gains_sum_left_right() {
        let g = get_split_gains(false, 4.0, 2.0, 6.0, 3.0, 0.0, 0.0);
        // 16/2 + 36/3 = 8 + 12 = 20.
        assert_eq!(g, 20.0);
    }

    #[test]
    fn leaf_output_sign() {
        assert_eq!(calculate_splitted_leaf_output(false, 4.0, 2.0, 0.0, 0.0), -2.0);
        // L1: -ThresholdL1(4,1)/(2+0) = -3/2.
        assert_eq!(calculate_splitted_leaf_output(true, 4.0, 2.0, 1.0, 0.0), -1.5);
    }

    #[test]
    fn gain_config_from_default_config_is_noop_l1() {
        let cfg = lgbm_core::Config::default();
        let gc = GainConfig::from_config(&cfg);
        assert_eq!(gc.min_data_in_leaf, 20);
        assert!(!gc.use_l1(), "default lambda_l1 == 0 -> no L1");
    }
}
