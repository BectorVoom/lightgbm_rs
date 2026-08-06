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
//! The `USE_L1`, `USE_MAX_OUTPUT` (`max_delta_step`) and `USE_SMOOTHING`
//! (`path_smooth`) template axes are all transcribed. The two-axis `*_full`
//! primitives ([`calculate_splitted_leaf_output_full`], [`get_leaf_gain_full`],
//! [`get_split_gains_full`]) are the general form; the older
//! `USE_MAX_OUTPUT=false, USE_SMOOTHING=false` fast-path fns are retained and are
//! what `*_full` DELEGATES to on that axis, so the default path is byte-unchanged.
//! `USE_MC` (monotone constraints) is layered on top in
//! `lgbm_treelearner::monotone_constraints`.

use cubecl::prelude::*;

/// Single-rounding `a * b + c` — a genuine FUSED multiply-add, usable from BOTH a
/// `#[cube]` kernel and plain host code.
///
/// # Why this exists
///
/// C++ LightGBM is built by clang/gcc with `-ffp-contract=on` (the C++ default), so
/// `a*b + c` in the source may be emitted as a single `fma` instruction — one rounding
/// instead of two. Rust NEVER auto-contracts, so a verbatim transcription of an
/// expression the C++ compiler contracted is off by up to one ULP.
///
/// That is invisible almost everywhere in this port because the hot gain formula,
/// `GetLeafGain`'s closed form `sg²/(h+λ)`, is a multiply and a divide with no add to
/// contract into. It becomes visible the moment `max_delta_step` / `path_smooth`
/// switch the gain to [`get_leaf_gain_given_output`], whose body
/// `-(2·g·o + (h+λ)·o²)` is exactly a multiply-then-add. See that function for the
/// measurement that pins WHICH multiply the reference fuses.
///
/// Host: [`f64::mul_add`], which Rust guarantees rounds once. Device: cubecl's
/// [`fma`] IR instruction, via the `expand` twin below — the same
/// plain-fn + `mod ::expand` pairing cubecl uses for its own intrinsics, so the
/// `#[cube]` macro rewrites a call here into the device op while ordinary Rust callers
/// get the host body (cubecl's own `fma` would `unexpanded!()`-panic on the host).
pub fn fused_mul_add(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[doc(hidden)]
pub mod fused_mul_add {
    use super::*;

    pub fn expand(
        scope: &mut Scope,
        a: NativeExpand<f64>,
        b: NativeExpand<f64>,
        c: NativeExpand<f64>,
    ) -> NativeExpand<f64> {
        fma::expand(scope, a, b, c)
    }
}

/// f32 mirror of [`fused_mul_add`] (the no-f64 hip path).
pub fn fused_mul_add_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[doc(hidden)]
pub mod fused_mul_add_f32 {
    use super::*;

    pub fn expand(
        scope: &mut Scope,
        a: NativeExpand<f32>,
        b: NativeExpand<f32>,
        c: NativeExpand<f32>,
    ) -> NativeExpand<f32> {
        fma::expand(scope, a, b, c)
    }
}

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

/// `GetLeafGainGivenOutput<USE_L1>` (feature_histogram.hpp:817-829 /
/// cuda_leaf_splits.hpp:92-103): the leaf gain for a leaf whose output is FIXED
/// (already monotone-clamped, OR the smoothing-blended output) rather than the
/// unconstrained `-g/(h+l2)`. Used by the monotone (`USE_MC`) split-gain path AND
/// by the net-new `USE_SMOOTHING` form-(D) gain path.
///
/// Promoted to `#[cube]` (17-02) so the smoothing gain path runs on device; a
/// `#[cube]` fn is also a plain Rust `fn`, so the existing monotone host caller
/// (`monotone_constraints.rs`) is byte-unchanged.
///
/// # The fused multiply-add is load-bearing
///
/// The body is written `-fma(2·g, o, (h+λ)·o²)`, NOT the literal
/// `-(2·g·o + (h+λ)·o²)` of the C++ source, because the reference BUILD contracts
/// that expression: with `-ffp-contract=on` (the C++ default) clang fuses the add
/// with the FIRST multiply, rounding once instead of twice. Rust never
/// auto-contracts, so the literal transcription is off by up to one ULP — see
/// [`fused_mul_add`] for why this is the only place in the port where it shows.
///
/// WHICH multiply is fused was measured, not guessed. At a leaf where
/// `max_delta_step` clamps BOTH children to the same output, the split gain equals
/// the no-split gain exactly in real arithmetic, so the three candidate
/// formulations are distinguishable by a single bit. Replaying the reference's own
/// operands (`G=c056c1479e000000`, `H=40591d6d5a200000`,
/// `gl=c05651479f000000`, `hl=40578d9650100000`, `gr=bffbffffc0000000`,
/// `hr=4018fd70a1000001`, `max_delta_step=0.05`) through each:
///
/// | formulation | candidate − shift |
/// |---|---|
/// | `-(2·g·o + (h+λ)·o²)` — no contraction | +1 ULP |
/// | `-fma((h+λ)·o, o, 2·g·o)` — fuse the SECOND multiply | +1 ULP |
/// | `-fma(2·g, o, (h+λ)·o²)` — fuse the FIRST multiply | **0, matching the reference** |
///
/// Only the third reproduces the reference bit-for-bit, and adopting it turned the
/// `path_smooth`/`max_delta_step` oracle sweep from 12/13 to 13/13 cells with no
/// exceptions while leaving every pre-existing golden unchanged.
///
/// This does tie the port to a reference built with contraction ENABLED. That is the
/// same reference every fixture in `oracle-harness` is captured from
/// (`lightgbm==4.6.0`, PyPI), and `path_smooth_parity` fails loudly if it ever stops
/// matching.
#[cube]
pub fn get_leaf_gain_given_output(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    output: f64,
) -> f64 {
    if use_l1 {
        let sg_l1 = threshold_l1(sum_gradients, l1);
        -fused_mul_add(2.0 * sg_l1, output, (sum_hessians + l2) * output * output)
    } else {
        -fused_mul_add(2.0 * sum_gradients, output, (sum_hessians + l2) * output * output)
    }
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

// ===========================================================================
// Net-new USE_SMOOTHING (path_smooth) gain path — form (B) output-blend + form
// (D) given-output gain. VERBATIM transcription of
// `LightGBM/src/treelearner/cuda/cuda_leaf_splits.hpp:74-122` (the
// `USE_SMOOTHING=true` template branch). ADDITIVE ONLY: the three non-smoothing
// gain fns above (`threshold_l1`, `get_leaf_gain`, `calculate_splitted_leaf_output`,
// `get_split_gains`) are byte-unchanged (D-09) — the Wave-2 stage-1 body dispatches
// to these NEW fns when `use_smoothing` is set and the EXISTING fns otherwise.
//
// `parent_output` is the leaf's parent output (from `CUDALeafSplitsStruct`),
// threaded as a scalar. `num_data` is `data_size_t` (i32) exactly as the CUDA
// signature; the `num_data / path_smooth` division promotes it to double
// (`num_data as f64`) — matching the C++ int→double promotion. The `as` cast is
// used rather than `f64::cast_from` because these fns are ALSO called on the host
// (unit tests + the CPU anchor), where `cast_from`'s host stub panics; `as` has a
// real host impl and the `#[cube]` macro lowers it to a device cast.
// ===========================================================================

/// `CalculateSplittedLeafOutput<USE_L1, true>` — the `USE_SMOOTHING=true` form (B)
/// (cuda_leaf_splits.hpp:74-90): compute the base output exactly as
/// [`calculate_splitted_leaf_output`] does, then blend toward `parent_output`:
///
/// ```cpp
/// ret = ret * (num_data / path_smooth) / (num_data / path_smooth + 1)
///     + parent_output / (num_data / path_smooth + 1);
/// ```
///
/// Precedence is verbatim: `ret * nps / (nps + 1)` == `(ret * nps) / (nps + 1)`,
/// NOT `ret * (nps / (nps + 1))` (the two differ in the last bit).
#[cube]
pub fn calculate_splitted_leaf_output_smoothed(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    path_smooth: f64,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    // Reuse the non-smoothing base output (bit-identical to inlining the branch).
    let ret = calculate_splitted_leaf_output(use_l1, sum_gradients, sum_hessians, l1, l2);
    let n_over_ps = num_data as f64 / path_smooth;
    ret * n_over_ps / (n_over_ps + 1.0) + parent_output / (n_over_ps + 1.0)
}

/// `GetLeafGain<USE_L1, true>` — the `USE_SMOOTHING=true` form (D)
/// (cuda_leaf_splits.hpp:117-121): the gain is NOT the closed form `sg²/(h+l2)`;
/// it is [`get_leaf_gain_given_output`] evaluated at the smoothing-blended output:
///
/// ```cpp
/// const double output = CalculateSplittedLeafOutput<USE_L1, true>(...);
/// return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
/// ```
#[cube]
pub fn get_leaf_gain_smoothed(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    path_smooth: f64,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    let output = calculate_splitted_leaf_output_smoothed(
        use_l1,
        sum_gradients,
        sum_hessians,
        l1,
        l2,
        path_smooth,
        num_data,
        parent_output,
    );
    get_leaf_gain_given_output(use_l1, sum_gradients, sum_hessians, l1, l2, output)
}

/// `GetSplitGains<false, USE_L1, false, true>` (feature_histogram.hpp:757-797,
/// the `!USE_MC` branch at `USE_SMOOTHING=true`) — the per-candidate SCAN gain
/// comparison used by [`crate::kernels::split::split_scan_body`] /
/// `find_best_split_cpu_native` when `path_smooth != 0.0`. NOTE the per-side
/// `num_data` argument (`left_count`/`right_count`) is that SIDE's own row
/// count, NOT the parent leaf's `num_data` (`feature_histogram.hpp:772-778`
/// threads `left_count`/`right_count` into each `GetLeafGain` call).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn get_split_gains_smoothed(
    use_l1: bool,
    sum_left_gradients: f64,
    sum_left_hessians: f64,
    sum_right_gradients: f64,
    sum_right_hessians: f64,
    l1: f64,
    l2: f64,
    path_smooth: f64,
    left_count: i32,
    right_count: i32,
    parent_output: f64,
) -> f64 {
    get_leaf_gain_smoothed(
        use_l1,
        sum_left_gradients,
        sum_left_hessians,
        l1,
        l2,
        path_smooth,
        left_count,
        parent_output,
    ) + get_leaf_gain_smoothed(
        use_l1,
        sum_right_gradients,
        sum_right_hessians,
        l1,
        l2,
        path_smooth,
        right_count,
        parent_output,
    )
}

// ===========================================================================
// Net-new USE_MAX_OUTPUT (max_delta_step) gain path — VERBATIM transcription of
// `LightGBM/src/treelearner/feature_histogram.hpp:716-738` (`CalculateSplittedLeafOutput`
// `USE_MAX_OUTPUT=true` branch) + `:799-815` (`GetLeafGain`'s given-output else
// branch, selected whenever `USE_MAX_OUTPUT` OR `USE_SMOOTHING` is compile-time
// true). ADDITIVE ONLY (D-09): the non-clamped fns above are byte-unchanged. G5-2.
//
// C++ selects `USE_MAX_OUTPUT=true` at a COMPILE-TIME template dispatch keyed on
// `config->max_delta_step > 0` (`FuncForNumricalL1`, feature_histogram.hpp:246-261)
// — i.e. whenever these `_clamped` fns are the live path, `max_delta_step` is a
// per-CONFIG (not per-call) constant `> 0` for the whole scan. The caller
// (`split.rs`) is responsible for the runtime `max_delta_step != 0.0` dispatch
// gate between these fns and the non-clamped ones above — mirroring the C++
// template selection at the call site rather than inside these fns.
// ===========================================================================

/// `CalculateSplittedLeafOutput<USE_L1, true, false>` (feature_histogram.hpp:716-738,
/// the `USE_MAX_OUTPUT=true` branch, `USE_SMOOTHING=false`):
///
/// ```cpp
/// if (max_delta_step > 0 && std::fabs(ret) > max_delta_step) {
///   ret = Common::Sign(ret) * max_delta_step;
/// }
/// ```
#[cube]
pub fn calculate_splitted_leaf_output_clamped(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
) -> f64 {
    let ret = calculate_splitted_leaf_output(use_l1, sum_gradients, sum_hessians, l1, l2);
    let over = max_delta_step > 0.0 && f64::abs(ret) > max_delta_step;
    // Common::Sign(ret) * max_delta_step, branch-free (mirrors threshold_l1's
    // select-based Sign encoding — `if cond {1.0} else {0.0}` mis-lowers on
    // cubecl-cpu, WR-05-adjacent).
    let pos = select(ret > 0.0, 1.0, 0.0);
    let neg = select(ret < 0.0, 1.0, 0.0);
    let clamped = (pos - neg) * max_delta_step;
    select(over, clamped, ret)
}

/// `GetLeafGain<USE_L1, true, false>` (feature_histogram.hpp:799-815, the
/// given-output else-branch at `USE_MAX_OUTPUT=true`):
///
/// ```cpp
/// double output = CalculateSplittedLeafOutput<USE_L1, true, false>(...);
/// return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
/// ```
#[cube]
pub fn get_leaf_gain_clamped(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
) -> f64 {
    let output =
        calculate_splitted_leaf_output_clamped(use_l1, sum_gradients, sum_hessians, l1, l2, max_delta_step);
    get_leaf_gain_given_output(use_l1, sum_gradients, sum_hessians, l1, l2, output)
}

/// `GetSplitGains<false, USE_L1, true, false>` (feature_histogram.hpp:757-797,
/// the `!USE_MC` branch at `USE_MAX_OUTPUT=true`) — the per-candidate SCAN gain
/// comparison used by [`crate::kernels::split::split_scan_body`] /
/// `find_best_split_cpu_native` when `max_delta_step != 0.0` (a candidate's gain
/// is the SUM of each side's clamped-output gain, NOT the closed-form
/// `sg²/(h+l2)` fast path, once `USE_MAX_OUTPUT` is compile-time true in C++).
#[cube]
pub fn get_split_gains_clamped(
    use_l1: bool,
    sum_left_gradients: f64,
    sum_left_hessians: f64,
    sum_right_gradients: f64,
    sum_right_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
) -> f64 {
    get_leaf_gain_clamped(use_l1, sum_left_gradients, sum_left_hessians, l1, l2, max_delta_step)
        + get_leaf_gain_clamped(use_l1, sum_right_gradients, sum_right_hessians, l1, l2, max_delta_step)
}

// ===========================================================================
// The FULL two-axis (USE_MAX_OUTPUT x USE_SMOOTHING) gain path.
//
// VERBATIM transcription of `feature_histogram.hpp:715-829` — the general
// template bodies the three fast-path fns above are the
// `USE_MAX_OUTPUT=false, USE_SMOOTHING=false` corner of. `USE_MAX_OUTPUT` and
// `USE_SMOOTHING` are C++ TEMPLATE bools resolved once per feature histogram
// (`FuncForNumricalL1`/`L2`: `max_delta_step > 0` and `path_smooth > kEpsilon`),
// so they arrive here as runtime flags derived from the SAME config predicates —
// see [`GainConfig::use_max_output`] / [`GainConfig::use_smoothing`].
//
// The default (both off) path delegates to the fast-path fns and is therefore
// BIT-UNCHANGED, which is what keeps the existing goldens valid.
// ===========================================================================

/// `CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`
/// (feature_histogram.hpp:715-737) — the general form, in the C++ statement order:
///
/// ```cpp
/// ret = USE_L1 ? -ThresholdL1(g, l1) / (h + l2) : -g / (h + l2);
/// if (USE_MAX_OUTPUT) {
///   if (max_delta_step > 0 && std::fabs(ret) > max_delta_step) {
///     ret = Common::Sign(ret) * max_delta_step;
///   }
/// }
/// if (USE_SMOOTHING) {
///   ret = ret * (num_data / smoothing) / (num_data / smoothing + 1)
///       + parent_output / (num_data / smoothing + 1);
/// }
/// ```
///
/// The clamp order is load-bearing: C++ clamps the RAW output and THEN blends, so
/// the blend pulls a CLAMPED value toward the parent, never the other way round.
///
/// `use_smoothing` gates the blend as a branchless `select`; when it is false the
/// divisor is forced to `1.0` so `num_data / smoothing` cannot become `inf` (and
/// the blend `NaN`) for the `path_smooth == 0` default. The discarded value never
/// reaches the result either way — this only keeps intermediates finite.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn calculate_splitted_leaf_output_full(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
    use_smoothing: bool,
    path_smooth: f64,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    // USE_L1 base — the existing fast-path fn, so the default corner is bit-identical.
    let base = calculate_splitted_leaf_output(use_l1, sum_gradients, sum_hessians, l1, l2);

    // USE_MAX_OUTPUT: `ret = Sign(ret) * max_delta_step` when the guard fires.
    // `Common::Sign(x) = (x > 0) - (x < 0)` — the same branchless encoding
    // [`threshold_l1`] uses (the `if cond { 1.0 } else { 0.0 }` form mis-lowers on
    // cubecl-cpu). The guard can only fire for `|base| > max_delta_step > 0`, so
    // `base` is never 0 there and the sign is never the degenerate 0.
    let clamp = max_delta_step > 0.0 && f64::abs(base) > max_delta_step;
    let pos = select(base > 0.0, 1.0, 0.0);
    let neg = select(base < 0.0, 1.0, 0.0);
    let clamped = select(clamp, (pos - neg) * max_delta_step, base);

    // USE_SMOOTHING: verbatim precedence — `ret * nps / (nps + 1)`, NOT
    // `ret * (nps / (nps + 1))` (the two differ in the last bit).
    let ps = select(use_smoothing, path_smooth, 1.0);
    let n_over_ps = num_data as f64 / ps;
    let blended = clamped * n_over_ps / (n_over_ps + 1.0) + parent_output / (n_over_ps + 1.0);
    select(use_smoothing, blended, clamped)
}

/// `GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`
/// (feature_histogram.hpp:798-810) — the general form:
///
/// ```cpp
/// if (!USE_MAX_OUTPUT && !USE_SMOOTHING) {
///   return USE_L1 ? sg_l1*sg_l1 / (h + l2) : g*g / (h + l2);   // closed form
/// } else {
///   double output = CalculateSplittedLeafOutput<...>(...);
///   return GetLeafGainGivenOutput<USE_L1>(g, h, l1, l2, output);
/// }
/// ```
///
/// The branch is NOT a redundant optimisation: at the unconstrained output the two
/// forms are mathematically equal but differ by ULPs in floating point, so which
/// one runs is observable. C++ picks it from the TEMPLATE bools — i.e. purely from
/// the config, never per candidate — and so does this `select`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn get_leaf_gain_full(
    use_l1: bool,
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
    use_smoothing: bool,
    path_smooth: f64,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    let closed = get_leaf_gain(use_l1, sum_gradients, sum_hessians, l1, l2);
    let output = calculate_splitted_leaf_output_full(
        use_l1,
        sum_gradients,
        sum_hessians,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        num_data,
        parent_output,
    );
    let given = get_leaf_gain_given_output(use_l1, sum_gradients, sum_hessians, l1, l2, output);
    // `USE_MAX_OUTPUT || USE_SMOOTHING` — both are config-level predicates.
    let use_given = max_delta_step > 0.0 || use_smoothing;
    select(use_given, given, closed)
}

/// `GetSplitGains<USE_MC=false, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`
/// (feature_histogram.hpp:757-772, the `!USE_MC` branch) — the general form.
///
/// Note the per-side `left_count` / `right_count`: smoothing weights each child's
/// output by ITS OWN row count, so the two sides are NOT symmetric in `num_data`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn get_split_gains_full(
    use_l1: bool,
    sum_left_gradients: f64,
    sum_left_hessians: f64,
    sum_right_gradients: f64,
    sum_right_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
    use_smoothing: bool,
    path_smooth: f64,
    left_count: i32,
    right_count: i32,
    parent_output: f64,
) -> f64 {
    get_leaf_gain_full(
        use_l1,
        sum_left_gradients,
        sum_left_hessians,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        left_count,
        parent_output,
    ) + get_leaf_gain_full(
        use_l1,
        sum_right_gradients,
        sum_right_hessians,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        right_count,
        parent_output,
    )
}

// ===========================================================================
// f32 mirrors of the gain primitives for the no-f64 hip device (CMP-04).
//
// IDENTICAL formula structure and gate ORDER as the f64 anchors above — the ONLY
// difference is the scalar type (`f32` vs `f64`), since hip (gfx1100) cannot
// allocate f64 (RESEARCH Pitfall 2/3). The hip parity gate compares the f32 hip
// result to the f64 cpu anchor (collected to f32) within `ORACLE_TOL = 1e-6`,
// absorbing the f32-vs-f64 accumulation divergence (the divergence the contract
// was designed for; D-03a). These are gated on `Capabilities.has_f64 == false`.
// ===========================================================================

/// f32 mirror of [`threshold_l1`] (the no-f64 hip path).
#[cube]
pub fn threshold_l1_f32(s: f32, l1: f32) -> f32 {
    // WR-05: pin EVERY literal to f32 so cubecl `#[cube]` literal inference
    // cannot resolve a bare `1.0`/`0.0` to f64 (which `select`'s value type does
    // not pin from `s`) and silently widen / mix precision on the hip path. The
    // f32 path exists precisely to avoid f64 on gfx1100; an inferred f64 here
    // would defeat that.
    let reg_s = f32::max(0.0f32, f32::abs(s) - l1);
    let pos = select(s > 0.0f32, 1.0f32, 0.0f32);
    let neg = select(s < 0.0f32, 1.0f32, 0.0f32);
    (pos - neg) * reg_s
}

/// f32 mirror of [`get_leaf_gain`] (the no-f64 hip path).
#[cube]
pub fn get_leaf_gain_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
) -> f32 {
    if use_l1 {
        let sg_l1 = threshold_l1_f32(sum_gradients, l1);
        (sg_l1 * sg_l1) / (sum_hessians + l2)
    } else {
        (sum_gradients * sum_gradients) / (sum_hessians + l2)
    }
}

/// f32 mirror of [`get_split_gains`] (the no-f64 hip path).
#[cube]
pub fn get_split_gains_f32(
    use_l1: bool,
    sum_left_gradients: f32,
    sum_left_hessians: f32,
    sum_right_gradients: f32,
    sum_right_hessians: f32,
    l1: f32,
    l2: f32,
) -> f32 {
    get_leaf_gain_f32(use_l1, sum_left_gradients, sum_left_hessians, l1, l2)
        + get_leaf_gain_f32(use_l1, sum_right_gradients, sum_right_hessians, l1, l2)
}

/// f32 mirror of [`calculate_splitted_leaf_output`] (the no-f64 hip path).
#[cube]
pub fn calculate_splitted_leaf_output_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
) -> f32 {
    if use_l1 {
        -threshold_l1_f32(sum_gradients, l1) / (sum_hessians + l2)
    } else {
        -sum_gradients / (sum_hessians + l2)
    }
}

/// f32 mirror of [`calculate_splitted_leaf_output_smoothed`] (the no-f64 hip path).
#[cube]
pub fn calculate_splitted_leaf_output_smoothed_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
    path_smooth: f32,
    num_data: i32,
    parent_output: f32,
) -> f32 {
    // WR-05: pin every literal f32 (the `+ 1` denominators) so cubecl cannot widen
    // the blend to f64 on the hip path.
    let ret = calculate_splitted_leaf_output_f32(use_l1, sum_gradients, sum_hessians, l1, l2);
    let n_over_ps = num_data as f32 / path_smooth;
    ret * n_over_ps / (n_over_ps + 1.0f32) + parent_output / (n_over_ps + 1.0f32)
}

/// f32 mirror of [`get_leaf_gain_smoothed`] (the no-f64 hip path).
#[cube]
pub fn get_leaf_gain_smoothed_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
    path_smooth: f32,
    num_data: i32,
    parent_output: f32,
) -> f32 {
    let output = calculate_splitted_leaf_output_smoothed_f32(
        use_l1,
        sum_gradients,
        sum_hessians,
        l1,
        l2,
        path_smooth,
        num_data,
        parent_output,
    );
    get_leaf_gain_given_output_f32(use_l1, sum_gradients, sum_hessians, l1, l2, output)
}

/// f32 mirror of [`get_leaf_gain_given_output`] (the no-f64 hip path).
#[cube]
pub fn get_leaf_gain_given_output_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
    output: f32,
) -> f32 {
    // WR-05: every literal pinned f32 so cubecl cannot widen the `2.0` factor to
    // f64 on the hip path (the f32 path exists precisely to avoid f64 on gfx1100).
    if use_l1 {
        let sg_l1 = threshold_l1_f32(sum_gradients, l1);
        -fused_mul_add_f32(2.0f32 * sg_l1, output, (sum_hessians + l2) * output * output)
    } else {
        -fused_mul_add_f32(2.0f32 * sum_gradients, output, (sum_hessians + l2) * output * output)
    }
}

/// f32 mirror of [`calculate_splitted_leaf_output_full`] (the no-f64 hip path).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn calculate_splitted_leaf_output_full_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
    max_delta_step: f32,
    use_smoothing: bool,
    path_smooth: f32,
    num_data: i32,
    parent_output: f32,
) -> f32 {
    // WR-05: every literal pinned f32 so cubecl cannot widen the clamp/blend to f64.
    let base = calculate_splitted_leaf_output_f32(use_l1, sum_gradients, sum_hessians, l1, l2);
    let clamp = max_delta_step > 0.0f32 && f32::abs(base) > max_delta_step;
    let pos = select(base > 0.0f32, 1.0f32, 0.0f32);
    let neg = select(base < 0.0f32, 1.0f32, 0.0f32);
    let clamped = select(clamp, (pos - neg) * max_delta_step, base);
    let ps = select(use_smoothing, path_smooth, 1.0f32);
    let n_over_ps = num_data as f32 / ps;
    let blended = clamped * n_over_ps / (n_over_ps + 1.0f32) + parent_output / (n_over_ps + 1.0f32);
    select(use_smoothing, blended, clamped)
}

/// f32 mirror of [`get_leaf_gain_full`] (the no-f64 hip path).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn get_leaf_gain_full_f32(
    use_l1: bool,
    sum_gradients: f32,
    sum_hessians: f32,
    l1: f32,
    l2: f32,
    max_delta_step: f32,
    use_smoothing: bool,
    path_smooth: f32,
    num_data: i32,
    parent_output: f32,
) -> f32 {
    let closed = get_leaf_gain_f32(use_l1, sum_gradients, sum_hessians, l1, l2);
    let output = calculate_splitted_leaf_output_full_f32(
        use_l1,
        sum_gradients,
        sum_hessians,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        num_data,
        parent_output,
    );
    let given = get_leaf_gain_given_output_f32(use_l1, sum_gradients, sum_hessians, l1, l2, output);
    let use_given = max_delta_step > 0.0f32 || use_smoothing;
    select(use_given, given, closed)
}

/// f32 mirror of [`get_split_gains_full`] (the no-f64 hip path).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn get_split_gains_full_f32(
    use_l1: bool,
    sum_left_gradients: f32,
    sum_left_hessians: f32,
    sum_right_gradients: f32,
    sum_right_hessians: f32,
    l1: f32,
    l2: f32,
    max_delta_step: f32,
    use_smoothing: bool,
    path_smooth: f32,
    left_count: i32,
    right_count: i32,
    parent_output: f32,
) -> f32 {
    get_leaf_gain_full_f32(
        use_l1,
        sum_left_gradients,
        sum_left_hessians,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        left_count,
        parent_output,
    ) + get_leaf_gain_full_f32(
        use_l1,
        sum_right_gradients,
        sum_right_hessians,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        right_count,
        parent_output,
    )
}

/// The minimal gain-config surface passed into [`crate::Backend::find_best_split`]
/// (extracted from `lgbm_core::Config`; we do NOT pass `&Config` into the
/// kernel — keep it small and `Copy`).
///
/// Fields map 1:1 to the `meta_->config->*` accesses in the scan, with ONE
/// exception: [`parent_output`](Self::parent_output) is per-LEAF, not per-config.
/// C++ threads it as a separate argument alongside `meta_->config` into every gain
/// call (`FindBestThresholdSequentially(..., double parent_output)`); carrying it
/// in the struct that is already threaded to every one of those call sites is the
/// same information without a signature change at ~90 sites. See
/// [`with_parent_output`](Self::with_parent_output).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainConfig {
    /// `int min_data_in_leaf` (config default 20).
    pub min_data_in_leaf: i32,
    /// `double min_sum_hessian_in_leaf` (config default 1e-3).
    pub min_sum_hessian_in_leaf: f64,
    /// `double max_delta_step` (config default 0.0). `> 0` selects the C++
    /// `USE_MAX_OUTPUT` template branch — see [`Self::use_max_output`].
    pub max_delta_step: f64,
    /// `double lambda_l1` (config default 0.0).
    pub lambda_l1: f64,
    /// `double lambda_l2` (config default 0.0).
    pub lambda_l2: f64,
    /// `double min_gain_to_split` (config default 0.0).
    pub min_gain_to_split: f64,
    /// `double path_smooth` (config default 0.0). `> kEpsilon` selects the C++
    /// `USE_SMOOTHING` template branch — see [`Self::use_smoothing`].
    pub path_smooth: f64,
    /// The leaf's `parent_output` — C++ `SerialTreeLearner::GetParentOutput`
    /// (serial_tree_learner.cpp:1005-1017): the ROOT's own (clamped, UN-smoothed)
    /// output while `tree->num_leaves() == 1`, and `leaf_splits->weight()` (the
    /// output the parent split assigned this leaf) thereafter.
    ///
    /// NOT a `Config` field: it is per-leaf state, seeded to `0.0` by
    /// [`from_config`](Self::from_config) — which is exactly the literal `0` C++
    /// passes when computing the root's own output — and set per leaf by the tree
    /// learner via [`with_parent_output`](Self::with_parent_output). Read ONLY when
    /// [`use_smoothing`](Self::use_smoothing) is true.
    pub parent_output: f64,
    /// `int min_data_per_group` (config default 100) — categorical many-vs-many
    /// minimum rows per accumulated category group.
    pub min_data_per_group: i32,
    /// `int max_cat_threshold` (config default 32) — categorical many-vs-many cap
    /// on the number of categories on one side.
    pub max_cat_threshold: i32,
    /// `double cat_l2` (config default 10.0) — extra l2 ADDED to lambda_l2 ONLY in
    /// the per-category gain (NOT the `gain_shift` baseline — the deliberate
    /// asymmetry, feature_histogram.cpp:163-168,248).
    pub cat_l2: f64,
    /// `double cat_smooth` (config default 10.0) — categorical CTR smoothing +
    /// the `RoundInt(hess*cnt_factor) >= cat_smooth` many-vs-many filter.
    pub cat_smooth: f64,
    /// `int max_cat_to_onehot` (config default 4) — categorical features with
    /// `num_bin <= max_cat_to_onehot` use the one-hot (one-vs-rest) path.
    pub max_cat_to_onehot: i32,
}

impl Default for GainConfig {
    /// The config.h defaults (so test literals can fill the categorical fields via
    /// `..Default::default()` without restating them). Mirrors
    /// `lgbm_core::Config::default()` for every field.
    fn default() -> Self {
        Self::from_config(&lgbm_core::Config::default())
    }
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
            // Per-leaf, not config-derived: the tree learner overwrites this per leaf.
            parent_output: 0.0,
            min_data_per_group: c.min_data_per_group,
            max_cat_threshold: c.max_cat_threshold,
            cat_l2: c.cat_l2,
            cat_smooth: c.cat_smooth,
            max_cat_to_onehot: c.max_cat_to_onehot,
        }
    }

    /// True if the L1 branch (`USE_L1`) is active, i.e. `lambda_l1 != 0`.
    /// LightGBM selects `USE_L1` at `config->lambda_l1 > 0`
    /// (`feature_histogram.cpp` template dispatch); we mirror `> 0`.
    pub fn use_l1(&self) -> bool {
        self.lambda_l1 > 0.0
    }

    /// The C++ `USE_MAX_OUTPUT` template bool — `FuncForNumricalL1`
    /// (feature_histogram.hpp:248-259) branches on `config->max_delta_step > 0`.
    pub fn use_max_output(&self) -> bool {
        self.max_delta_step > 0.0
    }

    /// The C++ `USE_SMOOTHING` template bool — `FuncForNumricalL2`
    /// (feature_histogram.hpp:264-270) branches on
    /// `config->path_smooth > kEpsilon`, NOT on `> 0`. `kEpsilon` is the `1e-15f`
    /// FLOAT constant widened to double, so the threshold is reproduced exactly.
    pub fn use_smoothing(&self) -> bool {
        self.path_smooth > f64::from(lgbm_core::types::K_EPSILON)
    }

    /// This config with the per-leaf `parent_output` replaced — the C++
    /// `GetParentOutput(tree, leaf_splits)` result threaded into the leaf's scan.
    #[must_use]
    pub fn with_parent_output(&self, parent_output: f64) -> Self {
        Self {
            parent_output,
            ..*self
        }
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
    ///
    /// INVARIANT (WR-03): on a no-split result `default_left` is a **don't-care
    /// sentinel**, not a meaningful routing decision. We hard-code `true` to
    /// match the C++ default-constructed `SplitInfo::default_left = true`
    /// (`split_info.hpp`), but consumers MUST gate on `gain != kMinScore`
    /// (equivalently `is_splittable`) before reading `default_left`; it carries
    /// no information when no split was found. The bit-exact parity gate only
    /// asserts `default_left` on splittable winners for this reason.
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
    fn given_output_matches_closed_form() {
        // At the unconstrained output `o = -g/(h+l2)` the given-output gain form
        // (form D) equals the closed-form `get_leaf_gain` (sg²/(h+l2)). This proves
        // the #[cube]-promoted `get_leaf_gain_given_output` is consistent with the
        // reused non-smoothing path (USE_SMOOTHING=false). Values chosen so the
        // float arithmetic is bit-exact.
        // no-L1: o = -4/2 = -2 -> -(2·4·-2 + 2·4) = 8 == 16/2.
        let (g, h, l2) = (4.0_f64, 2.0_f64, 0.0_f64);
        let o = calculate_splitted_leaf_output(false, g, h, 0.0, l2);
        assert_eq!(
            get_leaf_gain_given_output(false, g, h, 0.0, l2, o),
            get_leaf_gain(false, g, h, 0.0, l2)
        );
        // L1: ThresholdL1(4,1)=3, o=-1.5 -> -(2·3·-1.5 + 2·2.25) = 4.5 == 9/2.
        let (g, h, l1, l2) = (4.0_f64, 2.0_f64, 1.0_f64, 0.0_f64);
        let o = calculate_splitted_leaf_output(true, g, h, l1, l2);
        assert_eq!(
            get_leaf_gain_given_output(true, g, h, l1, l2, o),
            get_leaf_gain(true, g, h, l1, l2)
        );
        // f32 mirror consistency at the same closed-form output.
        let (gf, hf, l2f) = (4.0_f32, 2.0_f32, 0.0_f32);
        let of = calculate_splitted_leaf_output_f32(false, gf, hf, 0.0f32, l2f);
        assert_eq!(
            get_leaf_gain_given_output_f32(false, gf, hf, 0.0f32, l2f, of),
            get_leaf_gain_f32(false, gf, hf, 0.0f32, l2f)
        );
    }

    #[test]
    fn smoothing_blend_matches_reference() {
        // Fixed inputs: no-L1, l2=0, path_smooth=2, num_data=10, parent_output=0.5.
        let (use_l1, g, h, l1, l2) = (false, 4.0_f64, 2.0_f64, 0.0_f64, 0.0_f64);
        let (ps, n, parent) = (2.0_f64, 10_i32, 0.5_f64);

        // Hand-computed form-(B) blend, transcribed verbatim (verbatim precedence:
        // `ret * nps / (nps+1)`, NOT `ret * (nps/(nps+1))`).
        let base = calculate_splitted_leaf_output(use_l1, g, h, l1, l2); // -2.0
        let nps = f64::from(n) / ps; // 5.0
        let expected_out = base * nps / (nps + 1.0) + parent / (nps + 1.0);
        assert_eq!(
            calculate_splitted_leaf_output_smoothed(use_l1, g, h, l1, l2, ps, n, parent),
            expected_out
        );

        // Form (D): gain = GetLeafGainGivenOutput at the blended output.
        let expected_gain = get_leaf_gain_given_output(use_l1, g, h, l1, l2, expected_out);
        assert_eq!(
            get_leaf_gain_smoothed(use_l1, g, h, l1, l2, ps, n, parent),
            expected_gain
        );

        // Directional sanity: small path_smooth (n/ps -> inf, weight -> 1) is
        // base-dominated; large path_smooth (n/ps -> 0, weight -> 0) is parent-dominated.
        let near_base =
            calculate_splitted_leaf_output_smoothed(use_l1, g, h, l1, l2, 1e-3, n, parent);
        assert!(
            (near_base - base).abs() < 1e-2,
            "small path_smooth should approach base output"
        );
        let near_parent =
            calculate_splitted_leaf_output_smoothed(use_l1, g, h, l1, l2, 1e6, n, parent);
        assert!(
            (near_parent - parent).abs() < 1e-2,
            "large path_smooth should approach parent output"
        );

        // f32 mirror consistency: form-(D) gain equals given-output at the f32 blend.
        let (gf, hf, l2f, psf, parentf) = (4.0_f32, 2.0_f32, 0.0_f32, 2.0_f32, 0.5_f32);
        let out_f32 =
            calculate_splitted_leaf_output_smoothed_f32(use_l1, gf, hf, 0.0f32, l2f, psf, n, parentf);
        assert_eq!(
            get_leaf_gain_smoothed_f32(use_l1, gf, hf, 0.0f32, l2f, psf, n, parentf),
            get_leaf_gain_given_output_f32(use_l1, gf, hf, 0.0f32, l2f, out_f32)
        );
    }

    #[test]
    fn leaf_output_clamped_at_max_delta_step() {
        // G5-2 (SPEC-G5-2): CalculateSplittedLeafOutput<USE_L1,true,false>
        // (feature_histogram.hpp:716-738, the USE_MAX_OUTPUT branch):
        //   if (max_delta_step > 0 && fabs(ret) > max_delta_step) ret = sign(ret)*max_delta_step;
        // unclamped ret = -g/(h+l2) = -8/2 = -4.0; |ret|=4.0 > max_delta_step=0.7 -> clamp to -0.7.
        let (use_l1, g, h, l1, l2, mds) = (false, 8.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 0.7_f64);
        let base = calculate_splitted_leaf_output(use_l1, g, h, l1, l2);
        assert_eq!(base, -4.0);
        assert_eq!(calculate_splitted_leaf_output_clamped(use_l1, g, h, l1, l2, mds), -0.7);

        // Positive-sign mirror: ret=+4.0 clamps to +0.7.
        let (g2,) = (-8.0_f64,);
        assert_eq!(
            calculate_splitted_leaf_output_clamped(use_l1, g2, h, l1, l2, mds),
            0.7
        );

        // max_delta_step=0.0 -> no-op, bit-exact to the unclamped base (the C++
        // `max_delta_step > 0` gate is false at 0.0, so the branch never fires).
        assert_eq!(
            calculate_splitted_leaf_output_clamped(use_l1, g, h, l1, l2, 0.0),
            base
        );

        // |ret| <= max_delta_step -> no-op (inside the envelope, not clamped).
        let (g3, mds3) = (1.0_f64, 0.7_f64); // ret = -0.5, |ret| < 0.7
        let base3 = calculate_splitted_leaf_output(use_l1, g3, h, l1, l2);
        assert_eq!(
            calculate_splitted_leaf_output_clamped(use_l1, g3, h, l1, l2, mds3),
            base3
        );

        // GetLeafGain<USE_L1,true,false>: GetLeafGainGivenOutput at the clamped output.
        let expected_gain = get_leaf_gain_given_output(use_l1, g, h, l1, l2, -0.7);
        assert_eq!(get_leaf_gain_clamped(use_l1, g, h, l1, l2, mds), expected_gain);

        // max_delta_step=0.0 gain path is bit-exact to the closed-form fast path
        // (matching the given_output_matches_closed_form invariant above).
        assert_eq!(
            get_leaf_gain_clamped(use_l1, g, h, l1, l2, 0.0),
            get_leaf_gain(use_l1, g, h, l1, l2)
        );
    }

    #[test]
    fn split_gains_smoothed_sums_per_side_leaf_gains() {
        // G5-3 (SPEC-G5-3): GetSplitGains<false, USE_L1, false, true>
        // (feature_histogram.hpp:757-797, USE_SMOOTHING=true) — the per-candidate
        // SCAN gain used once path_smooth != 0.0. NOTE: the per-side `num_data`
        // argument is that side's OWN row count (left_count/right_count), NOT the
        // parent leaf's num_data (GetSplitGains threads `left_count`/`right_count`
        // into each GetLeafGain call — feature_histogram.hpp:772-778).
        let (use_l1, l1, l2, ps, parent) = (false, 0.0_f64, 0.0_f64, 2.0_f64, 0.5_f64);
        let (sum_left_g, sum_left_h, left_count) = (4.0_f64, 2.0_f64, 10_i32);
        let (sum_right_g, sum_right_h, right_count) = (6.0_f64, 3.0_f64, 5_i32);
        let expected = get_leaf_gain_smoothed(use_l1, sum_left_g, sum_left_h, l1, l2, ps, left_count, parent)
            + get_leaf_gain_smoothed(use_l1, sum_right_g, sum_right_h, l1, l2, ps, right_count, parent);
        assert_eq!(
            get_split_gains_smoothed(
                use_l1, sum_left_g, sum_left_h, sum_right_g, sum_right_h, l1, l2, ps, left_count,
                right_count, parent,
            ),
            expected
        );
    }

    #[test]
    fn gain_config_from_default_config_is_noop_l1() {
        let cfg = lgbm_core::Config::default();
        let gc = GainConfig::from_config(&cfg);
        assert_eq!(gc.min_data_in_leaf, 20);
        assert!(!gc.use_l1(), "default lambda_l1 == 0 -> no L1");
    }
}
