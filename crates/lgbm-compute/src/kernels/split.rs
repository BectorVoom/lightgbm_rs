//! `find_best_split` cube kernel — the gain math lives INSIDE the kernel.
//!
//! VERBATIM transcription of `FindBestThresholdSequentially`
//! (`LightGBM/src/treelearner/feature_histogram.hpp:830-1057`, commit 195c26fc,
//! VERSION 4.6.0.99) for the default CPU template instantiation
//! `<USE_RAND=false, USE_MC=false, USE_L1=?, USE_MAX_OUTPUT=false,
//! USE_SMOOTHING=false, SKIP_DEFAULT_BIN=?, NA_AS_MISSING=false>`. Both the
//! REVERSE branch (`:854-936`, records `t-1+offset`) and the FORWARD branch
//! (`:937-1029`, records `t+offset`) are transcribed 1:1 — NO loop
//! restructuring, NO gate reordering.
//!
//! The gain primitives ([`get_split_gains`] / [`get_leaf_gain`] /
//! [`calculate_splitted_leaf_output`] / [`threshold_l1`]) are `#[cube]`
//! functions in [`crate::gain`] called from inside this scan — the gain formula
//! is computed in the kernel, not pre-supplied.
//!
//! ## Epsilon placements (load-bearing, verbatim)
//! - `FindBestThreshold` adds `2 * kEpsilon` to `sum_hessian` at scan entry
//!   (`feature_histogram.hpp:172`). The launcher passes `sum_hessian` already
//!   bumped, mirroring `find_best_threshold_fun_(sum_gradient, sum_hessian + 2 *
//!   kEpsilon, ...)`.
//! - REVERSE seeds `double sum_right_hessian = kEpsilon;` (`:856`).
//! - FORWARD seeds `double sum_left_hessian = kEpsilon;` (`:939`).
//! - At finalization the `kEpsilon` is subtracted back off the reported left/right
//!   sum_hessian (`:1042` / `:1053`).
//!
//! `kEpsilon = 1e-15f` is reused from `lgbm_core::types::K_EPSILON` (an f32
//! literal); promoting it to f64 reproduces the C++ `float`->`double` widening.
//!
//! ## Determinism
//! The scan is inherently sequential (a left-to-right / right-to-left running
//! sum), so it is launched single-owner (`CubeDim::new_1d(1)`) on the cubecl-cpu
//! anchor — the CMP-04 capability gate selects `ReducePath::Sequential` on cpu
//! (no Plane dependency).
//!
//! ## cubecl-cpu lowering constraints (load-bearing)
//! The cubecl-cpu (0.10.0) MLIR pass rejects two patterns this scan would
//! naturally use, so they are worked around WITHOUT changing the numerics:
//! 1. A loop-carried mutable reassigned inside the loop must be initialized from
//!    a LITERAL, never directly from a scalar kernel argument (arg-init + reassign
//!    => "operation with block successors must terminate its parent block"). Hence
//!    `best_gain`/`best_threshold`/etc. init from literals; the gain sentinel is
//!    `0.0` (valid gains are strictly positive, so this is observably identical to
//!    C++ `kMinScore`).
//! 2. Conditional in-loop stores via nested `if` mutation chains fail the same
//!    pass, so every conditional store uses branchless `select(cond, new, old)`.
//!
//! The C++ gate ORDER and arithmetic are preserved 1:1; only the control-flow
//! ENCODING differs.
//!
//! ## Output protocol
//! The kernel writes a fixed-length f64 `out` array the launcher decodes into a
//! [`SplitInfo`]:
//! `[is_splittable, threshold, gain, left_count, right_count,
//!   left_sum_gradient, left_sum_hessian, right_sum_gradient, right_sum_hessian,
//!   default_left]`. Counts/threshold/flags are carried as f64 and rounded back
//! by the launcher (all are small exact integers / 0|1 flags). The gain cell is
//! the RAW `best_gain` (the launcher applies the `- min_gain_shift`, `* penalty`
//! finalization and the `best_gain > output->gain + min_gain_shift` accept gate,
//! matching `feature_histogram.hpp:1031-1056`).

use cubecl::prelude::*;

use lgbm_core::types::K_EPSILON;

/// Plain per-feature parameter record for the batched per-leaf split scan
/// ([`Backend::find_best_splits_batched`](crate::Backend::find_best_splits_batched)).
/// Carries EXACTLY the per-feature args the single-feature
/// [`Backend::find_best_split`](crate::Backend::find_best_split) takes today
/// (`lib.rs`), plus the feature's `slot_off` into the concatenated leaf histogram
/// buffer (so each feature reads only `[slot_off, slot_off + 2*num_bin)` —
/// reusing the validated slot layout `build_leaf_histograms_raw` produces).
///
/// The leaf totals (`sum_gradient` / `sum_hessian` / `num_data`) and the
/// [`GainConfig`] are shared across the whole batch and passed separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedSplitFeature {
    /// This feature's start offset into the concatenated stride-2 f64 histogram
    /// buffer; its region is `[slot_off, slot_off + 2*num_bin)`.
    pub slot_off: usize,
    /// The feature's bin count (`f.num_bin`). The histogram region is `2*num_bin`.
    pub num_bin: u32,
    /// The feature's bin-layout offset (`f.offset`, 1 iff `most_freq_bin == 0`).
    pub offset: i32,
    /// The feature's default bin (`f.default_bin`).
    pub default_bin: u32,
    /// The feature's most-frequent bin (`f.most_freq_bin`).
    pub most_freq_bin: u32,
    /// Authoritative `SKIP_DEFAULT_BIN` dispatch flag (`f.skip_default_bin()`).
    pub skip_default_bin: bool,
    /// Authoritative `NA_AS_MISSING` dispatch flag (`f.na_as_missing()`).
    pub na_as_missing: bool,
    /// Authoritative FORWARD-branch dispatch flag (`f.run_forward()`).
    pub run_forward: bool,
}

use crate::error::ComputeError;
use crate::gain::{calculate_splitted_leaf_output, get_split_gains, GainConfig, SplitInfo};
use crate::runtime::ActiveRuntime;

/// `Common::RoundInt(double x) = static_cast<int>(x + 0.5f)` (common.h:904).
///
/// NOTE the `0.5f` is a **float** literal added to a `double` then truncated
/// toward zero — we reproduce the f32 widening of `0.5` exactly.
#[cube]
fn round_int(x: f64) -> i32 {
    i32::cast_from(x + f64::cast_from(0.5f32))
}

/// f32 mirror of [`round_int`] for the no-f64 hip path: `(int)(x + 0.5f)`.
#[cube]
fn round_int_f32(x: f32) -> i32 {
    i32::cast_from(x + 0.5f32)
}

/// The single shared `#[cube]` REVERSE+FORWARD best-split scan body — the SINGLE
/// SOURCE OF TRUTH for the f64 split math.
///
/// Both the single-feature launch kernel ([`find_best_split_kernel`], bases 0,0)
/// AND the fused per-leaf batched kernel ([`find_best_splits_fused_kernel`], cube
/// `f` passing `hist_base = slot_off[f]`, `out_base = f*12`) call this helper, so
/// the REVERSE+FORWARD scan, the gate ORDER, the eps placements, the
/// `t-1+offset` / `t+offset` threshold arithmetic, the branchless-`select`
/// encoding, the monotone `done` flag, and the finalization (subtract eps off the
/// reported hessians) exist exactly ONCE. NO f64 op is reordered relative to the
/// pre-extraction `find_best_split_kernel` body (CLAUDE.md non-negotiable #1).
///
/// `hist_base` is the feature's start cell in the (possibly concatenated) stride-2
/// `[g0,h0,g1,h1,...]` f64 histogram buffer; bin index `bi` reads
/// `hist[hist_base + bi]`. `out_base` is the feature's 12-cell output window;
/// finalization writes `out[out_base + 0..12]`. For the single-feature kernel both
/// bases are 0, so the extraction is observably identical to the prior body.
///
/// `sum_hessian` is the leaf total ALREADY bumped by `2*kEpsilon` (the host bumps
/// it before launch). The loop-carried mutables MUST init from LITERALS (cubecl-cpu
/// MLIR lowering constraint #1) and every conditional store MUST be branchless
/// `select` (constraint #2) — both encodings are kept verbatim here.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn split_scan_body(
    hist: &Array<f64>,
    hist_base: u32,
    out: &mut Array<f64>,
    out_base: u32,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32, // 0|1
    use_l1: u32,           // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    rev_count: i32, // host-computed REVERSE iteration count = max(0, num_bin-1)
    fwd_count: i32, // host-computed FORWARD iteration count = max(0, num_bin-1-offset)
) {
    // `hb`/`ob` are the feature's base offsets into the (concatenated) histogram and
    // the 12-cell `out` window. They are 0,0 for the single-feature kernel.
    let hb = hist_base as usize;
    let ob = out_base as usize;
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let use_l1_b = use_l1 != 0;
    let skip_def = skip_default_bin != 0;

    // GET_GRAD(hist, i) = hist[i<<1]; GET_HESS(hist, i) = hist[(i<<1)+1]
    // (bin.h:45). `cnt_factor = num_data / sum_hessian` (feature_histogram.hpp:843).
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;

    // best-split running state (feature_histogram.hpp:838-842).
    //
    // cubecl-cpu lowering constraint (load-bearing): a loop-carried mutable that
    // is reassigned inside the scan MUST be initialized from a LITERAL, never
    // directly from a scalar kernel argument (initializing from an arg + later
    // reassignment produces invalid MLIR — "operation with block successors must
    // terminate its parent block"). So all of these init from literals.
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    // best_gain sentinel = 0.0 (a LITERAL). C++ uses `kMinScore = -inf`, but every
    // VALID candidate gain here is strictly > `min_gain_shift >= 0` (gains are
    // `g²/(h+λ)` sums, non-negative), so 0.0 is an equally-valid sentinel: no
    // valid candidate is ever rejected by it, and "no split found" is signaled by
    // `is_splittable == 0` (NOT by best_gain), exactly as the host decodes below.
    let mut best_gain = 0.0f64;
    let mut best_left_count = 0i32;
    // threshold sentinel = 0 (LITERAL). The C++ "no split" threshold is `num_bin`,
    // but when `is_splittable == 0` the host returns `SplitInfo::none()`
    // (threshold 0) and never reads this, so the literal sentinel is observably
    // identical.
    let mut best_threshold = 0i32;
    // Carried as f64 flags (0.0 / 1.0); the launcher reads them back as `!= 0.0`.
    let mut is_splittable = 0.0f64;
    // default_left of the WINNING branch (REVERSE => true=1.0, FORWARD => false=0.0).
    let mut best_default_left = 1.0f64;

    // ====================== REVERSE branch (:854-936) ======================
    //
    // C++ iterates `for (int t = num_bin-1-offset; t >= 1-offset; --t)`. cubecl-cpu
    // reliably lowers bounded RANGE `for` loops (not decrementing `while` loops
    // with index mutation), so we iterate a forward counter `k in 0..count` and
    // recover `t = t_start - k`. C++ `break` is equivalent to a monotone
    // "stop accumulating new candidates" here: in REVERSE, as t decreases the
    // right side grows and the left shrinks monotonically, so once
    // `left_count < min_data_in_leaf` or `sum_left_hessian <
    // min_sum_hessian_in_leaf` holds it holds for every smaller t too — gating
    // those-and-all-later iterations off (`done`) yields the IDENTICAL winner the
    // C++ `break` produces. (Gate ORDER preserved exactly.)
    {
        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = f64::cast_from(K_EPSILON); // kEpsilon (:856)
        let mut right_count = 0i32;

        let t_start = num_bin - 1 - offset; // NA_AS_MISSING=0; t_end = 1 - offset
        let count = rev_count; // host-computed = max(0, t_start - (1-offset) + 1) = num_bin-1
        let mut done = false; // sticky flag emulating C++ `break` (monotone)

        // BRANCHLESS form: cubecl-cpu's MLIR lowering rejected the nested-`if`
        // mutation chains, so every conditional store is expressed via `select`
        // (the gate ORDER and the exact arithmetic are still 1:1 with C++).
        for k in 0..count {
            let t = t_start - k;
            // C++ REVERSE loop bound is `t >= t_end` with `t_end = 1 - offset`
            // (:860,863). The `0..count` counter form drops that lower bound, so
            // for `offset >= 2` (out of the C++ `offset ∈ {0,1}` contract) `t`
            // would go negative and `(t as usize)` would wrap to a huge index —
            // an OOB read (WR-02). Restore the bound: `in_range` reproduces the
            // C++ loop condition, and we clamp the read index to a valid cell so
            // a negative `t` reads bin 0 inertly (it is forced inactive anyway,
            // so it never contributes). For the valid `offset ∈ {0,1}` cases `t`
            // never drops below `t_end ∈ {0,1}`, so this is a strict no-op there.
            let in_range = t >= (1 - offset);
            // skip default bin (:864-868) — a `continue`, does NOT stop the scan.
            let skip = skip_def && (t + offset) == default_bin as i32;
            // Accumulate only when in range, not skipped and not already stopped.
            let active = in_range && !skip && !done;
            // Branchless clamp (cubecl-cpu mis-lowers nested-`if`): a negative `t`
            // reads bin 0; it is inactive so it never contributes to any sum.
            let t_safe = select(t < 0, 0i32, t);
            let bi = hb + (t_safe as usize) * 2; // GET_GRAD index = t<<1 (+ feature base)
            let g = hist[bi];
            let h = hist[bi + 1];
            sum_right_gradient += select(active, g, 0.0);
            sum_right_hessian += select(active, h, 0.0);
            right_count += select(active, round_int(h * cnt_factor), 0i32);

            // gate order (:877-892) — verbatim, expressed as flat booleans.
            let left_count = num_data - right_count;
            let sum_left_hessian = sum_hessian - sum_right_hessian;
            let sum_left_gradient = sum_gradient - sum_right_gradient;
            // `continue` gate (:878): too few right rows / too little right hess.
            let cont = right_count < min_data_in_leaf
                || sum_right_hessian < min_sum_hessian_in_leaf;
            // `break` gates (:884 / :890): monotone — once true, stay true.
            let brk = left_count < min_data_in_leaf
                || sum_left_hessian < min_sum_hessian_in_leaf;
            // sticky `done` (C++ `break`): set when an active, non-continue
            // iteration hits a break gate; never cleared.
            done = done || (active && !cont && brk);
            // A candidate is considered iff active, not a `continue`, not stopped.
            let consider = active && !cont && !done;

            // Always COMPUTE the gain (no early-out in branchless form), then gate
            // it to the 0.0 sentinel unless this is a valid candidate beating
            // min_gain_shift (:913). The winner update is a chain of `select`s.
            let current_gain = get_split_gains(
                use_l1_b,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
            );
            let valid = consider && current_gain > min_gain_shift;
            is_splittable = select(valid, 1.0, is_splittable);
            let cand_gain = select(valid, current_gain, 0.0); // 0.0 == sentinel (never beats best on invalid)
            let take = cand_gain > best_gain; // strict `>` (:920) — keep first on tie
            best_left_count = select(take, left_count, best_left_count);
            best_sum_left_gradient = select(take, sum_left_gradient, best_sum_left_gradient);
            best_sum_left_hessian = select(take, sum_left_hessian, best_sum_left_hessian);
            // left <= threshold, right > threshold => t-1 (:933)
            best_threshold = select(take, t - 1 + offset, best_threshold);
            best_gain = select(take, cand_gain, best_gain);
            best_default_left = select(take, 1.0, best_default_left); // REVERSE
        }
    }

    // ====================== FORWARD branch (:937-1029) =====================
    // C++ `for (int t = 0; t <= num_bin-2-offset; ++t)`. Same range-loop +
    // monotone-`done` treatment (FORWARD grows the left side, so once the right
    // side is too small / right hessian too small it stays so — `break`==`done`).
    {
        let mut sum_left_gradient = 0.0f64;
        let mut sum_left_hessian = f64::cast_from(K_EPSILON); // kEpsilon (:939)
        let mut left_count = 0i32;

        let count = fwd_count; // host-computed = max(0, num_bin - 1 - offset)
        let mut done = false;

        for t in 0..count {
            let skip = skip_def && (t + offset) == default_bin as i32;
            let active = !skip && !done;
            // t >= 0 always here (range starts at 0; NA_AS_MISSING=0 path).
            let bi = hb + (t as usize) * 2;
            let g = hist[bi];
            let h = hist[bi + 1];
            sum_left_gradient += select(active, g, 0.0);
            sum_left_hessian += select(active, h, 0.0);
            left_count += select(active, round_int(h * cnt_factor), 0i32);

            let right_count = num_data - left_count;
            let sum_right_hessian = sum_hessian - sum_left_hessian;
            let sum_right_gradient = sum_gradient - sum_left_gradient;
            // `continue` gate (:976) then `break` gates (:982 / :988).
            let cont = left_count < min_data_in_leaf
                || sum_left_hessian < min_sum_hessian_in_leaf;
            let brk = right_count < min_data_in_leaf
                || sum_right_hessian < min_sum_hessian_in_leaf;
            done = done || (active && !cont && brk);
            let consider = active && !cont && !done;

            let current_gain = get_split_gains(
                use_l1_b,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
            );
            let valid = consider && current_gain > min_gain_shift;
            is_splittable = select(valid, 1.0, is_splittable);
            let cand_gain = select(valid, current_gain, 0.0); // 0.0 == sentinel (never beats best on invalid)
            let take = cand_gain > best_gain;
            best_left_count = select(take, left_count, best_left_count);
            best_sum_left_gradient = select(take, sum_left_gradient, best_sum_left_gradient);
            best_sum_left_hessian = select(take, sum_left_hessian, best_sum_left_hessian);
            // forward records t+offset (NOT t-1+offset) (:1025)
            best_threshold = select(take, t + offset, best_threshold);
            best_gain = select(take, cand_gain, best_gain);
            best_default_left = select(take, 0.0, best_default_left); // FORWARD
        }
    }

    // ---- finalization (feature_histogram.hpp:1031-1056) -------------------
    // The launcher applies the `is_splittable && best_gain > output->gain +
    // min_gain_shift` accept gate + the `- min_gain_shift` / `* penalty`
    // adjustments; here we emit the RAW winner state. We DO compute the f64
    // outputs + subtract kEpsilon back off the reported hessians (:1042/:1053)
    // so the launcher has the exact SplitInfo cells.
    let eps = f64::cast_from(K_EPSILON);
    let left_output = calculate_splitted_leaf_output(
        use_l1_b,
        best_sum_left_gradient,
        best_sum_left_hessian,
        l1,
        l2,
    );
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian - best_sum_left_hessian;
    let right_output =
        calculate_splitted_leaf_output(use_l1_b, right_sum_gradient, right_sum_hessian, l1, l2);

    out[ob] = is_splittable; // already an f64 flag (0.0 / 1.0)
    out[ob + 1] = f64::cast_from(best_threshold);
    out[ob + 2] = best_gain; // RAW best_gain (kMinScore if none)
    out[ob + 3] = f64::cast_from(best_left_count);
    out[ob + 4] = f64::cast_from(num_data - best_left_count); // right_count
    out[ob + 5] = best_sum_left_gradient;
    out[ob + 6] = best_sum_left_hessian - eps; // reported left_sum_hessian (:1042)
    out[ob + 7] = right_sum_gradient;
    out[ob + 8] = right_sum_hessian - eps; // reported right_sum_hessian (:1053)
    out[ob + 9] = best_default_left; // already an f64 flag (0.0 / 1.0)
    out[ob + 10] = left_output;
    out[ob + 11] = right_output;
}

/// The single-feature `find_best_split` launch kernel — a THIN `#[cube(launch)]`
/// wrapper that delegates to the shared [`split_scan_body`] with bases `0, 0` (the
/// whole histogram is one feature; the 12-cell `out` window starts at 0). This
/// kernel holds NO scan logic of its own — the math lives once in
/// `split_scan_body`, shared with the fused per-leaf batched kernel.
///
/// Launched single-owner (`CubeDim::new_1d(1)`): the scan is inherently sequential.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32, // 0|1 (comptime-flavored runtime flag)
    use_l1: u32,           // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    rev_count: i32, // host-computed REVERSE iteration count = max(0, num_bin-1)
    fwd_count: i32, // host-computed FORWARD iteration count = max(0, num_bin-1-offset)
) {
    split_scan_body(
        hist,
        0u32,
        out,
        0u32,
        num_bin,
        offset,
        default_bin,
        skip_default_bin,
        use_l1,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        lambda_l1,
        lambda_l2,
        min_gain_shift,
        sum_gradient,
        sum_hessian,
        num_data,
        rev_count,
        fwd_count,
    );
}


/// Host-side `find_best_split` on the cpu reference runtime.
///
/// Validates inputs (V5, threat T-04-01) BEFORE the unsafe launch, computes
/// `gain_shift` + `min_gain_shift` (the `BeforeNumerical` pre-step,
/// feature_histogram.hpp:198-207) and the `sum_hessian + 2*kEpsilon` entry bump
/// (`:172`) on the host, runs the single-owner scan, then applies the
/// `FindBestThreshold` finalization accept-gate + `- min_gain_shift` net-gain.
///
/// `penalty` (feature splitting penalty, `meta_->penalty`, `output->gain *=
/// penalty` at `:174`) defaults to `1.0` here (feature-penalty support is not yet
/// implemented).
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `hist.len() != 2 * num_bin`.
/// - [`ComputeError::Runtime`] if `num_bin == 0` or `sum_hessian` is non-positive
///   (the C++ `cnt_factor = num_data / sum_hessian` would divide by ~0), or if
///   unsupported non-default gain params are supplied (max_delta_step /
///   path_smooth are not yet implemented).
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: u32,
    offset: i32,
    default_bin: u32,
    most_freq_bin: u32,
    skip_default_bin: bool,
    na_as_missing: bool,
    run_forward: bool,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<SplitInfo, ComputeError> {
    find_best_split_f64_on(
        client,
        hist,
        cfg,
        num_bin,
        offset,
        default_bin,
        most_freq_bin,
        skip_default_bin,
        na_as_missing,
        run_forward,
        sum_gradient,
        sum_hessian,
        num_data,
    )
}

/// The f64 `find_best_split` cube path, **generic over the runtime** `R` so it runs
/// on the cubecl-cpu anchor (via [`find_best_split_cpu`]) AND on cubecl-hip (the GPU
/// `RocmBackend`) — the same f64 kernel, bit-exact across both. Identical host
/// pre-step / scan / decode as before; only the runtime is generic.
///
/// # Errors
/// Same as [`find_best_split_cpu`].
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: u32,
    offset: i32,
    default_bin: u32,
    _most_freq_bin: u32,
    skip_default_bin: bool,
    na_as_missing: bool,
    run_forward: bool,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<SplitInfo, ComputeError> {
    // --- V5 boundary validation (T-04-01) ---
    // NA_AS_MISSING forward-branch (feature_histogram.hpp:945-961) is deferred —
    // this layer threads the flag and validates it false on every captured case,
    // but the NaN-missing forward preamble kernel body is NOT yet transcribed.
    // Surface a TYPED error rather than silently mis-computing (T-05-01-01): the
    // unimplemented branch can never produce a wrong SplitInfo.
    if na_as_missing {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                     implemented"
                .to_string(),
        });
    }
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: num_bin must be > 0".to_string(),
        });
    }
    let expected = 2usize
        .checked_mul(num_bin as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("num_bin {num_bin} overflows the histogram length"),
        })?;
    if hist.len() != expected {
        return Err(ComputeError::LengthMismatch {
            expected,
            actual: hist.len(),
        });
    }
    // Reject non-positive OR NaN sum_hessian (cnt_factor = num_data/sum_hessian).
    // `!(x > 0.0)` is deliberately NaN-catching here.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }
    // Only the default no-op output-clamp / smoothing path is transcribed. Reject
    // non-default values rather than silently mis-computing.
    //
    // This rejection is load-bearing: the on-device seam binds the learner's REAL
    // `GainConfig` (not a permissive proving slice), so a learner cfg carrying a
    // non-zero max_delta_step / path_smooth must keep ERRORING here rather than
    // silently enabling unimplemented semantics on device.
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }

    // FindBestThreshold entry bump: sum_hessian + 2 * kEpsilon (:172). The 2 and
    // kEpsilon widen to f64 exactly as C++ does. C++ applies this bump at the
    // `FindBestThreshold` call site — `find_best_threshold_fun_(sum_gradient,
    // sum_hessian + 2 * kEpsilon, ...)` (feature_histogram.hpp:174) — so EVERY
    // downstream consumer (both `BeforeNumerical`/`min_gain_shift` AND `cnt_factor`
    // and the per-bin scan) sees the BUMPED `sum_hessian`. Compute the bump FIRST
    // and thread the bumped value into `min_gain_shift` below.
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;

    // BeforeNumerical (feature_histogram.hpp:198-207): the whole-leaf gain, then
    // min_gain_shift = gain_shift + min_gain_to_split. Computed on the host with
    // the SAME f64 gain primitive the kernel uses (so it is bit-identical).
    //
    // `gain_shift` MUST use the `2*kEpsilon`-BUMPED `sum_hessian`, not the raw
    // value. In C++ the lambda receives the already-bumped `sum_hessian` and
    // passes it straight into `BeforeNumerical(... sum_hessian ...)`
    // (feature_histogram.hpp:400-401, 411-413, 424-425), so `GetLeafGain`'s
    // denominator is `sum_hessian + 2*kEpsilon + lambda_l2`. Using the raw
    // `sum_hessian` here makes `min_gain_shift` a few ULPs higher than C++, which
    // can incorrectly reject splits whose `current_gain` exceeds the true
    // `min_gain_shift` by only a single f64 ULP. `min_gain_shift` computed from the
    // bumped sum_hessian is bit-exact to the reference implementation.
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    // Host-computed iteration counts (the loops are bounded RANGE loops on the
    // device; computing the bound on the host keeps the kernel control flow
    // simple for the cubecl-cpu MLIR lowering).
    //   REVERSE: t = num_bin-1-offset .. 1-offset  ->  count = num_bin-1
    //   FORWARD: t = 0 .. num_bin-2-offset          ->  count = num_bin-1-offset
    let num_bin_i = num_bin as i32;
    let rev_count = (num_bin_i - 1).max(0);
    // FORWARD branch dispatch (feature_histogram.hpp:420-429): LightGBM runs the
    // FORWARD scan ONLY for `num_bin > 2 && missing_type != None` (Zero or NaN);
    // for `missing_type == None` (and num_bin <= 2) it dispatches the REVERSE
    // branch ONLY, so `FindBestThreshold:170`'s pre-set `default_left = true`
    // survives (decision_type == 2). The caller passes `run_forward` as a verbatim
    // transcription of that truth table; when it is false we drive `fwd_count = 0`
    // so the FORWARD loop iterates zero times and `best_default_left` keeps its
    // REVERSE/initial 1.0 — exactly mirroring C++ never invoking the FORWARD
    // FindBestThresholdSequentially for this missing_type.
    let fwd_count = if run_forward {
        (num_bin_i - 1 - offset).max(0)
    } else {
        0
    };

    let out_len = 12usize;
    let h_hist = client.create_from_slice(f64::as_bytes(hist));
    // The kernel WRITES (never `+=`) all 12 `out` cells unconditionally (single
    // unit, no early return), so `out` needs no zero-init.
    // `empty()` skips the host zero-alloc + upload. Contrast the accumulate/atomic
    // buffers in histogram.rs:161 which MUST stay zeroed.
    let h_out = client.empty(out_len * core::mem::size_of::<f64>());

    // SAFETY: `h_hist` was allocated for exactly `hist.len() == 2*num_bin`
    // elements and `h_out` for `out_len` cells; both outlive the launch. The
    // REVERSE scan iterates a forward counter but restores the C++ `t >= 1-offset`
    // bound (`in_range`) and clamps the read index to bin 0 for any negative `t`
    // (`t_safe = select(t<0,0,t)`), so even an out-of-contract `offset >= 2`
    // cannot wrap `(t as usize)` into an OOB read (WR-02). The FORWARD scan starts
    // at `t = 0`. Thus every `hist[(t<<1)+1]` index stays in `[0, 2*num_bin)`,
    // within the allocation. All cubecl unsafe is confined here (CMP-01).
    unsafe {
        find_best_split_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_hist, hist.len()),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            num_bin as i32,
            offset,
            default_bin as i32,
            if skip_default_bin { 1u32 } else { 0u32 },
            if use_l1 { 1u32 } else { 0u32 },
            cfg.min_data_in_leaf,
            cfg.min_sum_hessian_in_leaf,
            cfg.lambda_l1,
            cfg.lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian_bumped,
            num_data,
            rev_count,
            fwd_count,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);

    let is_splittable = cells[0] != 0.0;
    let raw_threshold = cells[1] as u32;
    let raw_gain = cells[2];
    let left_count = cells[3] as i32;
    let right_count = cells[4] as i32;
    let left_sum_gradient = cells[5];
    let left_sum_hessian = cells[6];
    let right_sum_gradient = cells[7];
    let right_sum_hessian = cells[8];
    let default_left = cells[9] != 0.0;
    let left_output = cells[10];
    let right_output = cells[11];

    // FindBestThreshold finalization accept gate (feature_histogram.hpp:1031):
    //   if (is_splittable && best_gain > output->gain + min_gain_shift)
    // output->gain starts at kMinScore (-inf), so the gate reduces to
    // is_splittable && raw_gain > -inf == is_splittable (for a finite gain).
    // The reported gain is best_gain - min_gain_shift, then * penalty (penalty=1).
    let penalty = 1.0f64;
    if is_splittable && raw_gain > f64::NEG_INFINITY {
        Ok(SplitInfo {
            threshold: raw_threshold,
            gain: (raw_gain - min_gain_shift) * penalty,
            left_count,
            right_count,
            left_sum_gradient,
            left_sum_hessian,
            right_sum_gradient,
            right_sum_hessian,
            left_output,
            right_output,
            default_left,
        })
    } else {
        Ok(SplitInfo::none())
    }
}


/// Default scan cube width on rocm. The fused split-scan was originally launched
/// `CubeCount=(num_features,1,1) × CubeDim(1)` — one SINGLE-THREADED cube per feature,
/// using 1 lane of each wave32 (~1/32 ALU utilization). Packing one feature per LANE
/// (`CubeDim(W)`, `CubeCount=ceil(num_features/W)`) keeps each feature's scan sequential
/// (bit-exact, no reorder) but fills the wave. Measurement showed W=64 as a robust
/// occupancy knee across GPU shapes without over-fitting to any one device's
/// latency-hiding peak. Override with `LGBM_SCAN_CUBEDIM`.
#[cfg(feature = "gpu")]
const SCAN_CUBE_DIM_DEFAULT: u32 = 64;

/// Scan cube width W (env `LGBM_SCAN_CUBEDIM`, default [`SCAN_CUBE_DIM_DEFAULT`]).
/// W is the lanes-per-cube (`CubeDim::new_1d(W)`); `CubeCount = ceil(num_features / W)`.
/// W=1 reproduces the original one-cube-per-feature launch byte-for-byte. Clamped to
/// `[1, 256]` (a wavefront is 32/64 lanes; >256 just wastes a too-large cube). Parse
/// failures or 0 fall back to the default — never a no-launch.
#[cfg(feature = "gpu")]
fn scan_cube_dim() -> u32 {
    std::env::var("LGBM_SCAN_CUBEDIM")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&w| w > 0)
        .map(|w| w.clamp(1, 256))
        .unwrap_or(SCAN_CUBE_DIM_DEFAULT)
}

/// Non-rocm builds (cubecl-cpu oracle parity path) keep W=1 unconditionally — the
/// occupancy lever is a GPU-only concern and the bit-exact gate must not depend on
/// an env var.
#[cfg(not(feature = "gpu"))]
fn scan_cube_dim() -> u32 {
    1
}

// ===================== SCAN-W autotune =====================
//
// CubeCL autotune for the split-SCAN `CubeDim` width `W`. The fused feature-per-lane
// scan is BIT-EXACT across every `W` — each feature's scan stays sequential (lane
// `f = ABSOLUTE_POS`, guarded `< n_feats`); `W` only changes which lane runs each
// still-sequential per-feature scan, NOT the result. So the tuner is free to pick the
// measured-fastest `W` per occupancy regime. Default-ON on rocm; `LGBM_AUTOTUNE=0` and
// an explicit `LGBM_SCAN_CUBEDIM` both fall back to `scan_cube_dim()` (the documented
// bound + the all-W parity seam). Mirrors the build-`P` machinery in `histogram.rs`.
#[cfg(feature = "gpu")]
use crate::kernels::autotune::{self, LaunchKey};
#[cfg(feature = "gpu")]
use cubecl::tune::{local_tuner, CloneInputGenerator, LocalTuner, Tunable, TunableSet};

/// The scan-`W` candidate set the SCAN tuner sweeps (`CubeDim::new_1d(W)`,
/// `CubeCount = ceil(n_slots / W)`). Each entry is clamped `[1, 256]` (a wavefront is
/// 32/64 lanes; `>256` just wastes a too-large cube). The SET only needs to SPAN the
/// occupancy regimes (`{32,64,128,256}`) so the tuner re-derives the per-GPU winner on
/// any future GPU (measure-don't-model). `W=1` is intentionally NOT in the set — it is
/// the one-cube-per-feature degenerate the lever exists to AVOID; it stays reachable as
/// the `LGBM_SCAN_CUBEDIM=1` / non-rocm bit-exact oracle.
///
/// `pub` so the parity gate (`oracle-harness/tests/kernel_parity.rs`) imports the SAME
/// source of truth it sweeps (WR-02): a hand-copied mirror would silently stop covering
/// a newly-added `W`. Stays `#[cfg(feature = "rocm")]` so the default build is
/// byte-unchanged.
#[cfg(feature = "gpu")]
pub const SCAN_WSET: &[u32] = &[32, 64, 128, 256];

/// The SINGLE-LEAF SCAN cache namespace — `local_tuner!("scan")` ⇒
/// `LocalTuner<LaunchKey, String>`, distinct from the build tuner's `"build"` namespace.
/// Holds the persistent key→fastest_index map (mirrored to disk via `std_io`).
#[cfg(feature = "gpu")]
static SCAN_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("scan");

/// The CO-PACK 2-slot sibling-scan cache namespace (the sibling-scan launcher).
/// A SEPARATE tuner from [`SCAN_TUNER`] so the two kernel families never share a cache
/// entry — their [`LaunchKey`] would otherwise collide on `(0, feats, bins)` yet
/// benchmark different kernels. Each tuner caches its own winner over the SHARED
/// [`SCAN_WSET`] (same `W` ordering ⇒ same `fastest_index`→`W` mapping).
#[cfg(feature = "gpu")]
static SCAN_SIBLINGS_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("scan_siblings");

/// Launch the SINGLE-LEAF fused split-scan ([`find_best_splits_fused_kernel`]) once at a
/// fixed `W`, reading the ordered handle slice
/// `[hist, out, slot, numbin, offset, defbin, skip, rev, fwd]`. Mirrors the production
/// launch site EXACTLY (same kernel, `CubeCount=ceil(n/W)`, `CubeDim(W)`, same arg
/// order/sizes) — only the handles arrive via a slice instead of named locals. This is
/// the single launcher the SCAN tuner's WSET variants call (one per `W`); keep it in
/// sync with the in-place fallback launch in [`find_best_splits_fused_inner`].
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn launch_scan_at<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    n: usize,
    buf_len: usize,
    out_len: usize,
    inputs: &[cubecl::server::Handle],
    use_l1: bool,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) {
    let cube_count = (n as u32).div_ceil(w);
    // SAFETY: identical to the production in-place launch — every per-feature region is
    // host-validated `<= buf_len`, the `out` window is within `out_len`, and all index
    // arrays are sized `n`. All cubecl unsafe is confined here (CMP-01).
    unsafe {
        find_best_splits_fused_kernel::launch(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(w),
            ArrayArg::from_raw_parts(inputs[0].clone(), buf_len),
            ArrayArg::from_raw_parts(inputs[1].clone(), out_len),
            ArrayArg::from_raw_parts(inputs[2].clone(), n),
            ArrayArg::from_raw_parts(inputs[3].clone(), n),
            ArrayArg::from_raw_parts(inputs[4].clone(), n),
            ArrayArg::from_raw_parts(inputs[5].clone(), n),
            ArrayArg::from_raw_parts(inputs[6].clone(), n),
            ArrayArg::from_raw_parts(inputs[7].clone(), n),
            ArrayArg::from_raw_parts(inputs[8].clone(), n),
            if use_l1 { 1u32 } else { 0u32 },
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            n as u32,
        );
    }
}

/// Build the SCAN-tuner [`TunableSet`] for ONE single-leaf split-scan call: one
/// [`Tunable`] per `W` in [`SCAN_WSET`] (each launching [`find_best_splits_fused_kernel`]
/// at that `W` via [`launch_scan_at`]), keyed
/// `LaunchKey { bucket: 0, feats: n, bins: num_bins }`.
///
/// `bucket: 0` (NOT `size_band(rows)`): the scan width depends on the feature/bin SHAPE
/// (how many sequential per-feature scans pack into a wave), NOT the per-leaf row count —
/// so the key is STABLE across a train and the cache amortizes without a per-leaf tuning
/// storm. `num_bins` is the widest feature's bin count (the per-feature slot-width
/// driver, matching the build tuner's `bins`).
///
/// `CloneInputGenerator` is CORRECT here (an OVERWRITE-class kernel): the scan kernel
/// WRITES each feature's 12-cell `out` window FRESH every run (a `store`, NOT the
/// accumulating BUILD kernel's `fetch_add`), so re-running a benchmark rep on the shared
/// `out` handle recomputes the IDENTICAL window. Do NOT "fix" this to a fresh-output
/// generator — that is only needed for the accumulating build (`FreshOutGenerator`,
/// histogram.rs). The set is rebuilt fresh per call (the dims bake into the closures);
/// the persistent winner lives in [`SCAN_TUNER`]'s key state, not here.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn scan_wset_tunable_set<R: cubecl::Runtime>(
    client: cubecl::prelude::ComputeClient<R>,
    n: usize,
    buf_len: usize,
    out_len: usize,
    num_bins: u32,
    use_l1: bool,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> TunableSet<LaunchKey, Vec<cubecl::server::Handle>, ()> {
    let kg = move |_: &Vec<cubecl::server::Handle>| LaunchKey {
        bucket: 0,
        feats: n as u32,
        bins: num_bins,
    };
    let mut set = TunableSet::new(kg, CloneInputGenerator);
    for &w in SCAN_WSET {
        let w = w.clamp(1, 256);
        let c = client.clone();
        set = set.with(Tunable::new(
            &format!("scan_W{w}"),
            move |inputs: Vec<cubecl::server::Handle>| {
                launch_scan_at(
                    &c,
                    w,
                    n,
                    buf_len,
                    out_len,
                    &inputs,
                    use_l1,
                    min_data_in_leaf,
                    min_sum_hessian_in_leaf,
                    lambda_l1,
                    lambda_l2,
                    min_gain_shift,
                    sum_gradient,
                    sum_hessian,
                    num_data,
                );
                Ok::<(), String>(())
            },
        ));
    }
    set
}

/// Launch the CO-PACK 2-slot sibling split-scan
/// ([`find_best_splits_fused_siblings_kernel`]) once at a fixed `W`, reading the ordered
/// handle slice `[hist_a, hist_b, out, slot, numbin, offset, defbin, skip, rev, fwd]`.
/// Mirrors the production co-pack launch site EXACTLY (`CubeCount=ceil(2n/W)`,
/// `CubeDim(W)`). The two leaf-scalar SETS (A = smaller, B = larger sibling) are passed
/// explicitly. Same OVERWRITE class as the single-leaf scan.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn launch_scan_siblings_at<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    n: usize,
    buf_len: usize,
    out_len: usize,
    inputs: &[cubecl::server::Handle],
    use_l1: bool,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    num_data_a: i32,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    num_data_b: i32,
) {
    // CubeCount over 2*n feature-slots (the lane mapping packs A then B).
    let cube_count = (2 * n as u32).div_ceil(w);
    // SAFETY: identical to the production co-pack launch — both histogram handles
    // describe `buf_len` f64 cells, every per-feature region is host-validated, the `out`
    // window is within `out_len = 2*n*12`, and all index arrays are sized `n`. cubecl
    // unsafe confined here (CMP-01).
    unsafe {
        find_best_splits_fused_siblings_kernel::launch(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(w),
            ArrayArg::from_raw_parts(inputs[0].clone(), buf_len),
            ArrayArg::from_raw_parts(inputs[1].clone(), buf_len),
            ArrayArg::from_raw_parts(inputs[2].clone(), out_len),
            ArrayArg::from_raw_parts(inputs[3].clone(), n),
            ArrayArg::from_raw_parts(inputs[4].clone(), n),
            ArrayArg::from_raw_parts(inputs[5].clone(), n),
            ArrayArg::from_raw_parts(inputs[6].clone(), n),
            ArrayArg::from_raw_parts(inputs[7].clone(), n),
            ArrayArg::from_raw_parts(inputs[8].clone(), n),
            ArrayArg::from_raw_parts(inputs[9].clone(), n),
            if use_l1 { 1u32 } else { 0u32 },
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift_a,
            sum_gradient_a,
            sum_hessian_a,
            num_data_a,
            min_gain_shift_b,
            sum_gradient_b,
            sum_hessian_b,
            num_data_b,
            n as u32,
        );
    }
}

/// Build the SCAN-tuner [`TunableSet`] for ONE co-pack 2-slot sibling-scan call: one
/// [`Tunable`] per `W` in [`SCAN_WSET`] (each launching
/// [`find_best_splits_fused_siblings_kernel`] at that `W` via [`launch_scan_siblings_at`]),
/// keyed `LaunchKey { bucket: 0, feats: n, bins: num_bins }`. Same OVERWRITE class /
/// `CloneInputGenerator` rationale as [`scan_wset_tunable_set`]; executed under the
/// SEPARATE [`SCAN_SIBLINGS_TUNER`] namespace.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn scan_wset_siblings_tunable_set<R: cubecl::Runtime>(
    client: cubecl::prelude::ComputeClient<R>,
    n: usize,
    buf_len: usize,
    out_len: usize,
    num_bins: u32,
    use_l1: bool,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    num_data_a: i32,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    num_data_b: i32,
) -> TunableSet<LaunchKey, Vec<cubecl::server::Handle>, ()> {
    let kg = move |_: &Vec<cubecl::server::Handle>| LaunchKey {
        bucket: 0,
        feats: n as u32,
        bins: num_bins,
    };
    let mut set = TunableSet::new(kg, CloneInputGenerator);
    for &w in SCAN_WSET {
        let w = w.clamp(1, 256);
        let c = client.clone();
        set = set.with(Tunable::new(
            &format!("scan_sib_W{w}"),
            move |inputs: Vec<cubecl::server::Handle>| {
                launch_scan_siblings_at(
                    &c,
                    w,
                    n,
                    buf_len,
                    out_len,
                    &inputs,
                    use_l1,
                    min_data_in_leaf,
                    min_sum_hessian_in_leaf,
                    lambda_l1,
                    lambda_l2,
                    min_gain_shift_a,
                    sum_gradient_a,
                    sum_hessian_a,
                    num_data_a,
                    min_gain_shift_b,
                    sum_gradient_b,
                    sum_hessian_b,
                    num_data_b,
                );
                Ok::<(), String>(())
            },
        ));
    }
    set
}
// ================== end SCAN-W autotune ====================

/// FUSED per-leaf batched best-split kernel. ONE launch finds EVERY feature's best
/// split for a leaf: lane `f` (`ABSOLUTE_POS`, guarded `< n_feats`) scans only its
/// `[slot_off[f], slot_off[f] + 2*num_bin[f])` region of the concatenated f64
/// histogram `hist` and writes only its 12-cell window `out[f*12 .. f*12+12]`
/// (threat T-mc5-02). The per-feature scan is sequential; the launch packs one
/// feature per LANE (`CubeCount=ceil(num_feats/W), CubeDim(W)`, the scan-occupancy
/// lever). `W=1` reproduces the original one-cube-per-feature shape byte-for-byte;
/// `W>1` only changes which thread runs each (still-sequential, bit-identical)
/// per-feature scan.
///
/// The leaf-level scalars (`use_l1` .. `num_data`) are identical across features
/// (the leaf totals + cfg + the host-computed `min_gain_shift`), so they are passed
/// ONCE; only the per-feature params are device Arrays indexed by `f`. The body is
/// the SHARED [`split_scan_body`] — the single source of the split math.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // LEAF-LEVEL scalars (shared across the batch).
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    // Number of features in the batch. The launch may round CubeCount up so the
    // tail cube has lanes with `ABSOLUTE_POS >= n_feats`; those lanes must no-op.
    n_feats: u32,
) {
    // Scan-occupancy lever: the feature index is the GLOBAL lane index
    // `ABSOLUTE_POS = CUBE_POS_X * CUBE_DIM + UNIT_POS`. With `CubeDim::new_1d(1)`
    // this is byte-identical to the original one-cube-per-feature launch
    // (`ABSOLUTE_POS == CUBE_POS_X`); with `CubeDim::new_1d(W)` it packs W features
    // per cube, ONE per lane. Each lane runs the SAME sequential `split_scan_body`
    // for its own feature over a DISJOINT histogram region — no shared state, no
    // reorder of the per-feature scan itself — so the per-feature f64 result is
    // bit-identical regardless of W; only wave ALU utilization changes.
    // `ABSOLUTE_POS` is `usize` in cubecl; cast to u32 so the `f * 12u32` window
    // offset and the `f < n_feats` guard keep the original kernel's exact types.
    let f = ABSOLUTE_POS as u32;
    if f < n_feats {
        let fi = f as usize;
        split_scan_body(
            hist,
            slot_off[fi],
            out,
            f * 12u32,
            num_bin[fi],
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            rev_count[fi],
            fwd_count[fi],
        );
    }
}

/// CO-PACKED 2-slot per-leaf best-split kernel. ONE launch
/// scans BOTH siblings of a split: the smaller child's histogram `hist_a` and the
/// larger child's `hist_b`, over `2*n_feats` feature-slots. Global lane
/// `g = ABSOLUTE_POS` (guarded `< 2*n_feats`): `g < n_feats` ⇒ sibling-A (smaller)
/// feature `g` into `out[g*12..]`; `n_feats <= g < 2*n_feats` ⇒ sibling-B (larger)
/// feature `g − n_feats` into `out[g*12..]`. So `out[0..n*12]` = sibling A and
/// `out[n*12..2n*12]` = sibling B (the 12-cell-per-feature window, per sibling).
///
/// The per-feature param Arrays (`slot_off` .. `fwd_count`, length `n`) are SHARED
/// between siblings — both children have the SAME dataset feature layout (same bins,
/// same slot offsets, same iteration counts), indexed by the local feature index
/// `fi`. The LEAF-LEVEL scalars are PER-SIBLING (`sum_gradient`/`sum_hessian`/
/// `num_data`/`min_gain_shift` differ — the smaller child is built, the larger is
/// subtract-derived). Each lane runs the SHARED [`split_scan_body`] over its
/// sibling's DISJOINT histogram region — the SAME sequential reverse+forward scan as
/// the single-slot kernel, no reorder — so each feature's f64 result
/// is BIT-IDENTICAL to two separate single-slot scans; co-packing only changes WHICH
/// launch a feature's scan runs in, not its math.
///
/// `split_scan_body` takes ONE `hist: &Array<f64>` and cannot select between two
/// Array refs into a binding, so the body is called in EACH arm with that sibling's
/// histogram + that sibling's leaf scalars.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_kernel(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    // SHARED per-feature params (length n; both siblings share the dataset layout).
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // SHARED cfg scalars.
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    // PER-SIBLING leaf scalars (A = smaller, B = larger).
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    num_data_a: i32,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    num_data_b: i32,
    // Per-sibling feature count (n). The launch rounds CubeCount up so the tail cube
    // has lanes with `g >= 2*n_feats`; those lanes must no-op.
    n_feats: u32,
) {
    let g = ABSOLUTE_POS as u32;
    let total = 2u32 * n_feats;
    if g < total {
        // Local feature index within the sibling; indexes the SHARED per-feature
        // Arrays (length n) for both siblings.
        let fi = if g < n_feats { g } else { g - n_feats } as usize;
        // Branch on the sibling: call the SHARED split_scan_body in each arm with
        // that sibling's histogram + leaf scalars, writing `out[g*12..]` (so A's n
        // features land at out[0..n*12], B's at out[n*12..2n*12]).
        if g < n_feats {
            split_scan_body(
                hist_a,
                slot_off[fi],
                out,
                g * 12u32,
                num_bin[fi],
                offset[fi],
                default_bin[fi],
                skip_default_bin[fi],
                use_l1,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l1,
                lambda_l2,
                min_gain_shift_a,
                sum_gradient_a,
                sum_hessian_a,
                num_data_a,
                rev_count[fi],
                fwd_count[fi],
            );
        } else {
            split_scan_body(
                hist_b,
                slot_off[fi],
                out,
                g * 12u32,
                num_bin[fi],
                offset[fi],
                default_bin[fi],
                skip_default_bin[fi],
                use_l1,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l1,
                lambda_l2,
                min_gain_shift_b,
                sum_gradient_b,
                sum_hessian_b,
                num_data_b,
                rev_count[fi],
                fwd_count[fi],
            );
        }
    }
}

// ============================================================================
// LDS-STAGED per-feature split scan — the scan-occupancy + memory-latency fix.
//
// The lane-per-feature fused kernels above launch `ceil(n/W)` cubes for `n`
// features — at the production shape (n≈50, W=64) that is ONE cube, i.e. one
// SM busy on the whole GPU, and every lane's sequential scan walks ~2·num_bin
// dependent f64 loads straight from GLOBAL memory (~500+ cycles each, un-hidden
// — the drain-mode scan phase measured ~1ms/launch on P100, spike084). These
// STAGED twins change ONLY the execution geometry, never the math:
//   * ONE CUBE PER FEATURE (`CubeCount = (n[, 2 siblings])`), so all features'
//     scans run on different SMs concurrently;
//   * the cube first COOPERATIVELY loads its feature's `2*num_bin` histogram
//     cells into shared memory (`SCAN_STAGE_MAX_CELLS` ≤ 256 bins — the same
//     cap as the LDS build), so the sequential scan's dependent loads hit LDS
//     (~1 cycle) instead of global memory;
//   * the REVERSE and FORWARD branches run in TWO LANES concurrently (lane 0 /
//     lane 1), each a VERBATIM transcription of the corresponding
//     `split_scan_body` branch, then lane 0 merges the two branch winners.
//
// BIT-EXACT by construction: each branch's f64 fold order is unchanged (same
// loop, same gates, same `select` encoding — only the buffer the loads hit
// differs, and staging is a pure copy). The two-branch merge reproduces the
// serial shared-best-state semantics exactly (the proven
// `find_best_split_cpu_native_2lane` combine): serial FORWARD `take` requires
// STRICT `cand_gain > best_gain` against a running best that already includes
// the REVERSE winner, which is equivalent to "FORWARD's within-branch first-max
// beats REVERSE's winner strictly" — so `select(fwd_gain > rev_gain, fwd, rev)`
// with REVERSE winning exact ties is the identical winner, threshold, counts,
// sums, and `default_left`. `is_splittable` is the OR of the branch flags
// (serial sets it on ANY valid candidate in either branch).
//
// GPU-only (`#[cfg(feature = "gpu")]`), and the launcher additionally gates on
// the RUNTIME being a real device (`R::name(client) != "cpu"`): the cubecl-cpu
// MLIR anchor keeps the byte-unchanged serial kernels (the bit-exact merge gate
// must not depend on SharedMemory/sync_cube lowering there). Escape hatch:
// `LGBM_SCAN_STAGED=0` restores the legacy lane-per-feature launch.
// ============================================================================

/// LDS staging capacity in f64 cells: `2 * 256` (one feature ≤ 256 bins, the
/// same per-feature cap as the LDS build's `HIST_LDS_MAX`). 4 KiB of shared
/// memory per cube; features wider than this fall back to the legacy kernel
/// (whole-launch fallback in the launcher, never a per-feature mix).
#[cfg(feature = "gpu")]
const SCAN_STAGE_MAX_CELLS: usize = 512;

/// Staged-scan cube width: enough lanes to make the cooperative LDS load fast
/// (512 cells / 64 lanes = 8 strided iterations) while wasting little on the
/// 2-active-lane scan phase. One wavefront on NVIDIA (2×32); half on AMD (64).
#[cfg(feature = "gpu")]
const SCAN_STAGED_CUBE_DIM: u32 = 64;

/// Staged-scan gate (env `LGBM_SCAN_STAGED`, default ON; `"0"` restores the
/// legacy lane-per-feature launch — the A/B escape hatch). Read fresh per call
/// (mirrors `scan_cube_dim`) so a test can flip it without restart.
///
/// VERDICT (spike092b, P100, 500k×50×100 trees, order-ALTERNATED warm-median
/// of 3, same-session A/B, predictions BIT-IDENTICAL max_abs = 0.0): staged
/// 9.87s vs legacy 10.39s (1.05×); drain-mode de-aliased phase times: scan
/// 2.73s → 2.04s, pick 0.62s → 0.30s, grow wall 8.95s → 7.70s. (An earlier
/// spike091 read "net-negative", but its staged branch was wired only into the
/// host-readback launchers — the live grow-driver scans via the no-readback
/// raw-handle helpers — and its FORWARD lane shared the REVERSE lane's warp,
/// so SIMT divergence serialized the branches; both fixed in the live-route
/// wiring commit: FORWARD runs in lane 32, a separate warp.) Remaining scan
/// headroom: ~2.0s drained is still launch/serial-loop dominated — the
/// per-candidate parallel-gain + parallel-argmax redesign is the next scan
/// lever, behind build (2.2s) and partition (1.65s).
#[cfg(feature = "gpu")]
fn scan_staged_enabled() -> bool {
    !matches!(std::env::var("LGBM_SCAN_STAGED").as_deref(), Ok("0"))
}

/// REVERSE-branch scan reading a STAGED (LDS) histogram — a VERBATIM
/// transcription of [`split_scan_body`]'s REVERSE block (`feature_histogram.hpp
/// :854-936`): same literal-init state, same gate order, same monotone `done`,
/// same branchless `select` encoding, same f64 op order. Differences are purely
/// mechanical: the histogram reads hit `sm` (the feature's region staged to
/// shared memory, base 0) instead of the global buffer, and the branch's final
/// best-state is written to `state[0..6]` (`[is_splittable, best_gain,
/// threshold, left_count, sum_left_gradient, sum_left_hessian]` — threshold and
/// count carried as exact small-integer f64s) instead of continuing into the
/// FORWARD branch. Any drift from `split_scan_body`'s REVERSE block is a
/// correctness bug (the Kaggle A/B gate pins new-vs-old predictions
/// bit-identical).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn scan_rev_branch_staged(
    sm: &Slice<f64>,
    state: &mut SliceMut<f64>,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32, // 0|1
    use_l1: u32,           // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
    num_data: i32,
    rev_count: i32,
) {
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let use_l1_b = use_l1 != 0;
    let skip_def = skip_default_bin != 0;
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;

    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut best_gain = 0.0f64;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = 0.0f64;

    let mut sum_right_gradient = 0.0f64;
    let mut sum_right_hessian = f64::cast_from(K_EPSILON); // kEpsilon (:856)
    let mut right_count = 0i32;

    let t_start = num_bin - 1 - offset;
    let count = rev_count;
    let mut done = false;

    for k in 0..count {
        let t = t_start - k;
        let in_range = t >= (1 - offset);
        let skip = skip_def && (t + offset) == default_bin;
        let active = in_range && !skip && !done;
        let t_safe = select(t < 0, 0i32, t);
        let bi = (t_safe as usize) * 2;
        let g = sm[bi];
        let h = sm[bi + 1];
        sum_right_gradient += select(active, g, 0.0);
        sum_right_hessian += select(active, h, 0.0);
        right_count += select(active, round_int(h * cnt_factor), 0i32);

        let left_count = num_data - right_count;
        let sum_left_hessian = sum_hessian - sum_right_hessian;
        let sum_left_gradient = sum_gradient - sum_right_gradient;
        let cont =
            right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        let brk = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;

        let current_gain = get_split_gains(
            use_l1_b,
            sum_left_gradient,
            sum_left_hessian,
            sum_right_gradient,
            sum_right_hessian,
            l1,
            l2,
        );
        let valid = consider && current_gain > min_gain_shift;
        is_splittable = select(valid, 1.0, is_splittable);
        let cand_gain = select(valid, current_gain, 0.0);
        let take = cand_gain > best_gain;
        best_left_count = select(take, left_count, best_left_count);
        best_sum_left_gradient = select(take, sum_left_gradient, best_sum_left_gradient);
        best_sum_left_hessian = select(take, sum_left_hessian, best_sum_left_hessian);
        best_threshold = select(take, t - 1 + offset, best_threshold);
        best_gain = select(take, cand_gain, best_gain);
    }

    state[0] = is_splittable;
    state[1] = best_gain;
    state[2] = f64::cast_from(best_threshold);
    state[3] = f64::cast_from(best_left_count);
    state[4] = best_sum_left_gradient;
    state[5] = best_sum_left_hessian;
}

/// FORWARD-branch scan reading a STAGED (LDS) histogram — a VERBATIM
/// transcription of [`split_scan_body`]'s FORWARD block (`feature_histogram.hpp
/// :937-1029`) with its OWN literal-init best state (the serial body continues
/// on the REVERSE state; the standalone branch instead reports its within-branch
/// first-max, and the lane-0 merge reproduces the serial shared-state winner —
/// see the module note above / `find_best_split_cpu_native_2lane`). Same
/// mechanical differences as [`scan_rev_branch_staged`]: LDS reads, 6-cell state
/// output.
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn scan_fwd_branch_staged(
    sm: &Slice<f64>,
    state: &mut SliceMut<f64>,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32, // 0|1
    use_l1: u32,           // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
    num_data: i32,
    fwd_count: i32,
) {
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let use_l1_b = use_l1 != 0;
    let skip_def = skip_default_bin != 0;
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;

    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut best_gain = 0.0f64;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = 0.0f64;

    let mut sum_left_gradient = 0.0f64;
    let mut sum_left_hessian = f64::cast_from(K_EPSILON); // kEpsilon (:939)
    let mut left_count = 0i32;

    let count = fwd_count;
    let mut done = false;

    for t in 0..count {
        let skip = skip_def && (t + offset) == default_bin;
        let active = !skip && !done;
        let bi = (t as usize) * 2;
        let g = sm[bi];
        let h = sm[bi + 1];
        sum_left_gradient += select(active, g, 0.0);
        sum_left_hessian += select(active, h, 0.0);
        left_count += select(active, round_int(h * cnt_factor), 0i32);

        let right_count = num_data - left_count;
        let sum_right_hessian = sum_hessian - sum_left_hessian;
        let sum_right_gradient = sum_gradient - sum_left_gradient;
        let cont = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        let brk =
            right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;

        let current_gain = get_split_gains(
            use_l1_b,
            sum_left_gradient,
            sum_left_hessian,
            sum_right_gradient,
            sum_right_hessian,
            l1,
            l2,
        );
        let valid = consider && current_gain > min_gain_shift;
        is_splittable = select(valid, 1.0, is_splittable);
        let cand_gain = select(valid, current_gain, 0.0);
        let take = cand_gain > best_gain;
        best_left_count = select(take, left_count, best_left_count);
        best_sum_left_gradient = select(take, sum_left_gradient, best_sum_left_gradient);
        best_sum_left_hessian = select(take, sum_left_hessian, best_sum_left_hessian);
        best_threshold = select(take, t + offset, best_threshold);
        best_gain = select(take, cand_gain, best_gain);
    }

    state[0] = is_splittable;
    state[1] = best_gain;
    state[2] = f64::cast_from(best_threshold);
    state[3] = f64::cast_from(best_left_count);
    state[4] = best_sum_left_gradient;
    state[5] = best_sum_left_hessian;
}

/// Merge the two staged branch states + finalize into the feature's 12-cell
/// `out` window — the serial shared-best-state semantics (REVERSE wins exact
/// ties via strict `>`; `default_left` = 1.0 iff the REVERSE branch holds the
/// winner, including the no-winner init state) followed by a VERBATIM
/// [`split_scan_body`] finalization (`feature_histogram.hpp:1031-1056` —
/// left/right outputs, the kEpsilon subtracted back off the reported hessians).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn merge_finalize_staged(
    state_rev: &Slice<f64>,
    state_fwd: &Slice<f64>,
    out: &mut Array<f64>,
    out_base: u32,
    use_l1: u32,
    lambda_l1: f64,
    lambda_l2: f64,
    sum_gradient: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
    num_data: i32,
) {
    let ob = out_base as usize;
    let use_l1_b = use_l1 != 0;
    let rev_gain = state_rev[1];
    let fwd_gain = state_fwd[1];
    // Serial FORWARD `take` is STRICT (`cand_gain > best_gain`) against a running
    // best that includes the REVERSE winner ⇒ FORWARD wins only strictly.
    let take_fwd = fwd_gain > rev_gain;
    let any_split = state_rev[0] != 0.0 || state_fwd[0] != 0.0;
    let is_splittable = select(any_split, 1.0, 0.0);
    let best_gain = select(take_fwd, fwd_gain, rev_gain);
    let best_threshold_f = select(take_fwd, state_fwd[2], state_rev[2]);
    let best_left_count_f = select(take_fwd, state_fwd[3], state_rev[3]);
    let best_sum_left_gradient = select(take_fwd, state_fwd[4], state_rev[4]);
    let best_sum_left_hessian = select(take_fwd, state_fwd[5], state_rev[5]);
    // REVERSE => true=1.0 (also the no-winner init), FORWARD => false=0.0.
    let best_default_left = select(take_fwd, 0.0, 1.0);
    // Exact small-integer f64 → i32 round-trip (the state cells carry i32 values).
    let best_left_count = i32::cast_from(best_left_count_f);

    let eps = f64::cast_from(K_EPSILON);
    let left_output = calculate_splitted_leaf_output(
        use_l1_b,
        best_sum_left_gradient,
        best_sum_left_hessian,
        lambda_l1,
        lambda_l2,
    );
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian - best_sum_left_hessian;
    let right_output = calculate_splitted_leaf_output(
        use_l1_b,
        right_sum_gradient,
        right_sum_hessian,
        lambda_l1,
        lambda_l2,
    );

    out[ob] = is_splittable;
    out[ob + 1] = best_threshold_f;
    out[ob + 2] = best_gain;
    out[ob + 3] = best_left_count_f;
    out[ob + 4] = f64::cast_from(num_data - best_left_count);
    out[ob + 5] = best_sum_left_gradient;
    out[ob + 6] = best_sum_left_hessian - eps;
    out[ob + 7] = right_sum_gradient;
    out[ob + 8] = right_sum_hessian - eps;
    out[ob + 9] = best_default_left;
    out[ob + 10] = left_output;
    out[ob + 11] = right_output;
}

/// LDS-STAGED fused per-leaf best-split kernel: ONE CUBE PER FEATURE
/// (`CubeCount::Static(n, 1, 1)`, `CubeDim::new_1d(SCAN_STAGED_CUBE_DIM)` — the
/// launcher launches EXACTLY `n` cubes, so no tail guard is needed and the
/// `sync_cube()`s are unconditionally uniform). All lanes cooperatively stage
/// the feature's `2*num_bin` histogram cells into LDS; lane 0 then runs the
/// VERBATIM REVERSE branch and lane 1 the VERBATIM FORWARD branch concurrently;
/// lane 0 merges + finalizes into `out[f*12..f*12+12]`. Bit-identical output to
/// [`find_best_splits_fused_kernel`] (see the module note above).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_staged_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // LEAF-LEVEL scalars (shared across the batch).
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    // Cooperative stage: all lanes copy this feature's histogram region into LDS
    // (a pure copy — the scan below reads the identical f64 bits). `num_bin` is
    // validated positive before launch, so the i32→u32 widen is exact.
    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    let mut c = UNIT_POS as usize;
    while c < cells {
        sm[c] = hist[base + c];
        c += cd;
    }
    sync_cube();

    if UNIT_POS == 0 {
        scan_rev_branch_staged(
            &sm.to_slice(),
            &mut state_rev.to_slice_mut(),
            num_bin[fi],
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            rev_count[fi],
        );
    }
    // FORWARD in lane 32 — a DIFFERENT WARP from the REVERSE lane on NVIDIA
    // (warp = 32 lanes), so the two serial branch scans genuinely overlap.
    // spike091 ran FORWARD in lane 1: lanes 0 and 1 share one warp, and SIMT
    // divergence SERIALIZES divergent branches within a warp — the "concurrent"
    // branches ran back-to-back. (On AMD wave64 both lanes share one wavefront
    // either way; this pick is the NVIDIA-optimal one and AMD-neutral.)
    if UNIT_POS == 32 {
        scan_fwd_branch_staged(
            &sm.to_slice(),
            &mut state_fwd.to_slice_mut(),
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            fwd_count[fi],
        );
    }
    sync_cube();

    if UNIT_POS == 0 {
        merge_finalize_staged(
            &state_rev.to_slice(),
            &state_fwd.to_slice(),
            out,
            f * 12u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            sum_gradient,
            sum_hessian,
            num_data,
        );
    }
}

/// LDS-STAGED co-packed 2-slot sibling best-split kernel: ONE CUBE PER
/// (FEATURE, SIBLING) — `CubeCount::Static(n, 2, 1)`; `CUBE_POS_Y` selects the
/// sibling (0 = A/smaller reads `hist_a`, 1 = B/larger reads `hist_b`) and its
/// leaf scalars. Output layout identical to
/// [`find_best_splits_fused_siblings_kernel`]: sibling A's feature `f` at
/// `out[f*12..]`, sibling B's at `out[(n+f)*12..]`. Bit-identical results (the
/// staged geometry note above; per-feature params are SHARED between siblings).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_staged_kernel(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    // SHARED per-feature params (length n; both siblings share the dataset layout).
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // SHARED cfg scalars.
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    // PER-SIBLING leaf scalars (A = smaller, B = larger).
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    num_data_a: i32,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    num_data_b: i32,
    // Per-sibling feature count (n) — the B sibling's out window offset.
    n_feats: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let is_b = CUBE_POS_Y != 0;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    // Select this cube's sibling scalars (an Array ref cannot be `select`ed, so
    // the stage loop below branches on the sibling instead).
    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

    // Cooperative stage from THIS sibling's histogram (pure copy). `num_bin` is
    // validated positive before launch, so the i32→u32 widen is exact.
    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    if is_b {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_b[base + c];
            c += cd;
        }
    } else {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_a[base + c];
            c += cd;
        }
    }
    sync_cube();

    if UNIT_POS == 0 {
        scan_rev_branch_staged(
            &sm.to_slice(),
            &mut state_rev.to_slice_mut(),
            num_bin[fi],
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            rev_count[fi],
        );
    }
    // FORWARD in lane 32 — a DIFFERENT WARP from the REVERSE lane on NVIDIA
    // (warp = 32 lanes), so the two serial branch scans genuinely overlap.
    // spike091 ran FORWARD in lane 1: lanes 0 and 1 share one warp, and SIMT
    // divergence SERIALIZES divergent branches within a warp — the "concurrent"
    // branches ran back-to-back. (On AMD wave64 both lanes share one wavefront
    // either way; this pick is the NVIDIA-optimal one and AMD-neutral.)
    if UNIT_POS == 32 {
        scan_fwd_branch_staged(
            &sm.to_slice(),
            &mut state_fwd.to_slice_mut(),
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            fwd_count[fi],
        );
    }
    sync_cube();

    if UNIT_POS == 0 {
        // A's feature f → window f; B's → window n_feats + f (the legacy lane
        // mapping `g = sib*n + f`).
        let g = select(is_b, n_feats + f, f);
        merge_finalize_staged(
            &state_rev.to_slice(),
            &state_fwd.to_slice(),
            out,
            g * 12u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            sum_gradient,
            sum_hessian,
            num_data,
        );
    }
}

// ============================================================================
// PARALLEL-CANDIDATE staged scan ("pargain", `LGBM_SCAN_PARGAIN`) — the scan
// redesign the spike092b root-cause map called for. The staged kernels above
// still run each branch's WHOLE candidate walk serially on one lane, with the
// heavy gain math (two f64 divisions per candidate, `get_split_gains`) inside
// the serial loop while 62 lanes idle. This variant splits each branch:
//
//   PHASE 1 (serial, 1 lane/branch — the part that MUST stay serial): the
//     accumulation walk, byte-for-byte the staged branch's adds in the same
//     order (`+= select(active, …)`, the `done` freeze, `round_int` counts),
//     but instead of computing gains it STORES each candidate's ACCUMULATED
//     pair + count + `consider` flag to LDS. ~15 cheap ops/candidate instead
//     of ~60 incl. two divides.
//   PHASE 2 (parallel, 32 lanes/branch — one warp each, no intra-warp
//     divergence between branches): each lane strides the stored candidates,
//     derives the complementary side EXACTLY as the serial body does (one
//     `total − accumulated` subtraction), computes the gain with the SAME
//     `get_split_gains` cube fn, and folds a per-lane lexicographic first-max.
//   PHASE 3 (1 lane/branch): reduce the 32 lane partials with the SAME
//     lexicographic order and assemble the branch's 6-cell state; the existing
//     [`merge_finalize_staged`] finishes identically.
//
// BIT-EXACTNESS: phase 1's f64 accumulation order is unchanged (same adds,
// same order — storing to LDS and reloading is exact); phase 2 computes each
// candidate's gain from bit-identical inputs with the bit-identical op
// sequence (`total − accumulated` is literally the serial body's derivation;
// the accumulated side is passed through untouched); and the serial `take =
// cand_gain > best_gain` over ascending visit order selects the max-gain,
// EARLIEST-visited candidate — exactly the lexicographic (gain desc, visit
// order asc) maximum, which is associative + commutative, so the per-lane
// partial + tree-shape-free serial reduction reproduces it bit-for-bit. The
// all-candidates-invalid / zero-gain edge keeps the serial init state via the
// `won = best_gain > 0.0` guard (serial `take` is strict against the 0.0
// init). `is_splittable` is the OR of `valid` over every candidate, exactly
// the serial monotone flag.
// ============================================================================

/// Per-branch candidate-state capacity (max candidates = 256-bin cap − 1,
/// padded to 256 cells).
#[cfg(feature = "gpu")]
const PARGAIN_MAX_CAND: usize = 256;

// Sentinel `best_k` for "no winner yet" is the in-kernel literal
// `i32::new(2_147_483_647)` (= i32::MAX): larger than any real candidate
// index, so the lexicographic (gain desc, k asc) fold never prefers it (the
// cube macro requires a literal here, not a host const).

/// Pargain gate (env `LGBM_SCAN_PARGAIN`, opt-in `"1"`; default OFF until the
/// real-CUDA A/B validates it). Only meaningful where the staged gates already
/// hold (real device, ≤256-bin features) — the launch helpers consult it INSIDE
/// the staged arm, so legacy/staged behavior is byte-unchanged when unset.
#[cfg(feature = "gpu")]
fn scan_pargain_enabled() -> bool {
    matches!(std::env::var("LGBM_SCAN_PARGAIN").as_deref(), Ok("1"))
}

/// PHASE 1, REVERSE branch: the accumulation walk of [`scan_rev_branch_staged`]
/// with the gain/best logic REMOVED and each candidate's state STORED —
/// `cand_ag`/`cand_ah` hold the branch-ACCUMULATED right-side pair (the serial
/// body derives the left side from the totals each iteration; phase 2 repeats
/// that derivation verbatim), `cand_lc` the derived left count, `cand_ok` the
/// serial `consider` flag. Accumulation ops and order are byte-identical to the
/// staged branch fn.
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn pargain_store_rev(
    sm: &Slice<f64>,
    cand_ag: &mut SliceMut<f64>,
    cand_ah: &mut SliceMut<f64>,
    cand_lc: &mut SliceMut<f64>,
    cand_ok: &mut SliceMut<f64>,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32, // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
    num_data: i32,
    rev_count: i32,
) {
    let skip_def = skip_default_bin != 0;
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;

    let mut sum_right_gradient = 0.0f64;
    let mut sum_right_hessian = f64::cast_from(K_EPSILON); // kEpsilon (:856)
    let mut right_count = 0i32;

    let t_start = num_bin - 1 - offset;
    let mut done = false;

    for k in 0..rev_count {
        let t = t_start - k;
        let in_range = t >= (1 - offset);
        let skip = skip_def && (t + offset) == default_bin;
        let active = in_range && !skip && !done;
        let t_safe = select(t < 0, 0i32, t);
        let bi = (t_safe as usize) * 2;
        let g = sm[bi];
        let h = sm[bi + 1];
        sum_right_gradient += select(active, g, 0.0);
        sum_right_hessian += select(active, h, 0.0);
        right_count += select(active, round_int(h * cnt_factor), 0i32);

        let left_count = num_data - right_count;
        let sum_left_hessian = sum_hessian - sum_right_hessian;
        let cont =
            right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        let brk = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;

        let ku = k as usize;
        cand_ag[ku] = sum_right_gradient;
        cand_ah[ku] = sum_right_hessian;
        cand_lc[ku] = f64::cast_from(left_count);
        cand_ok[ku] = select(consider, 1.0, 0.0);
    }
}

/// PHASE 1, FORWARD branch: the accumulation walk of [`scan_fwd_branch_staged`],
/// storing the branch-ACCUMULATED LEFT-side pair (the serial body derives the
/// right side from the totals; phase 2 repeats it verbatim). See
/// [`pargain_store_rev`].
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn pargain_store_fwd(
    sm: &Slice<f64>,
    cand_ag: &mut SliceMut<f64>,
    cand_ah: &mut SliceMut<f64>,
    cand_lc: &mut SliceMut<f64>,
    cand_ok: &mut SliceMut<f64>,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32, // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
    num_data: i32,
    fwd_count: i32,
) {
    let skip_def = skip_default_bin != 0;
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;

    let mut sum_left_gradient = 0.0f64;
    let mut sum_left_hessian = f64::cast_from(K_EPSILON); // kEpsilon (:939)
    let mut left_count = 0i32;

    let mut done = false;

    for t in 0..fwd_count {
        let skip = skip_def && (t + offset) == default_bin;
        let active = !skip && !done;
        let bi = (t as usize) * 2;
        let g = sm[bi];
        let h = sm[bi + 1];
        sum_left_gradient += select(active, g, 0.0);
        sum_left_hessian += select(active, h, 0.0);
        left_count += select(active, round_int(h * cnt_factor), 0i32);

        let right_count = num_data - left_count;
        let sum_right_hessian = sum_hessian - sum_left_hessian;
        let cont = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        let brk =
            right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;

        let ku = t as usize;
        cand_ag[ku] = sum_left_gradient;
        cand_ah[ku] = sum_left_hessian;
        cand_lc[ku] = f64::cast_from(left_count);
        cand_ok[ku] = select(consider, 1.0, 0.0);
    }
}

/// PHASE 2: one lane's strided walk over a branch's stored candidates. Derives
/// the complementary side EXACTLY as the serial body does (`total −
/// accumulated`, one subtraction each), computes the gain with the SAME
/// [`get_split_gains`] cube fn on bit-identical inputs, and folds the per-lane
/// lexicographic first-max (gain desc, visit order asc; the `cand_gain > 0.0`
/// guard keeps the equality clause from ever preferring a zero-gain candidate
/// over the serial 0.0-init state). Writes the lane's partial best + the
/// `valid`-OR flag.
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn pargain_lane_scan(
    cand_ag: &Slice<f64>,
    cand_ah: &Slice<f64>,
    cand_ok: &Slice<f64>,
    part_gain: &mut SliceMut<f64>,
    part_k: &mut SliceMut<f64>,
    part_any: &mut SliceMut<f64>,
    lane: u32, // 0..32 within the branch
    count: i32,
    acc_is_left: u32, // 1 ⇒ stored pair is the LEFT side (FWD); 0 ⇒ RIGHT (REV)
    use_l1: u32,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
) {
    let use_l1_b = use_l1 != 0;
    let acc_left = acc_is_left != 0;
    let mut best_gain = 0.0f64;
    // "no winner" sentinel: above any real candidate index (exact in f64).
    let mut best_k = 2147483647.0f64;
    let mut any_valid = 0.0f64;

    let mut k = lane as i32;
    while k < count {
        let ku = k as usize;
        if cand_ok[ku] != 0.0 {
            let acc_g = cand_ag[ku];
            let acc_h = cand_ah[ku];
            // The serial body's complement derivation, verbatim (one subtraction).
            let oth_g = sum_gradient - acc_g;
            let oth_h = sum_hessian - acc_h;
            let left_g = select(acc_left, acc_g, oth_g);
            let left_h = select(acc_left, acc_h, oth_h);
            let right_g = select(acc_left, oth_g, acc_g);
            let right_h = select(acc_left, oth_h, acc_h);
            let current_gain =
                get_split_gains(use_l1_b, left_g, left_h, right_g, right_h, lambda_l1, lambda_l2);
            let valid = current_gain > min_gain_shift;
            any_valid = select(valid, 1.0, any_valid);
            let cand_gain = select(valid, current_gain, 0.0);
            let kf = f64::cast_from(k);
            // Lexicographic first-max: strictly greater gain, or equal POSITIVE
            // gain at an earlier visit order.
            let take = cand_gain > best_gain
                || (cand_gain == best_gain && cand_gain > 0.0 && kf < best_k);
            best_gain = select(take, cand_gain, best_gain);
            best_k = select(take, kf, best_k);
        }
        k += 32;
    }
    part_gain[lane as usize] = best_gain;
    part_k[lane as usize] = best_k;
    part_any[lane as usize] = any_valid;
}

/// PHASE 3: reduce a branch's 32 lane partials with the SAME lexicographic
/// order and assemble the branch's 6-cell state (`[is_splittable, best_gain,
/// threshold, left_count, sum_left_gradient, sum_left_hessian]` — the exact
/// layout [`merge_finalize_staged`] consumes). `thr = thr_base + thr_step * k`
/// encodes both branches' threshold formulas (REV: `num_bin − 2 − k`; FWD:
/// `k + offset`). The `won` guard reproduces the serial 0.0-init state when no
/// candidate strictly beat it.
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn pargain_assemble_state(
    part_gain: &Slice<f64>,
    part_k: &Slice<f64>,
    part_any: &Slice<f64>,
    cand_ag: &Slice<f64>,
    cand_ah: &Slice<f64>,
    cand_lc: &Slice<f64>,
    state: &mut SliceMut<f64>,
    acc_is_left: u32,
    thr_base: i32,
    thr_step: i32,
    sum_gradient: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon (host)
) {
    let acc_left = acc_is_left != 0;
    let mut best_gain = 0.0f64;
    // "no winner" sentinel: above any real candidate index (exact in f64).
    let mut best_k = 2147483647.0f64;
    let mut any = 0.0f64;
    for l in 0..32usize {
        let g = part_gain[l];
        let k = part_k[l];
        any = select(part_any[l] != 0.0, 1.0, any);
        let take = g > best_gain || (g == best_gain && g > 0.0 && k < best_k);
        best_gain = select(take, g, best_gain);
        best_k = select(take, k, best_k);
    }
    let won = best_gain > 0.0;
    // Clamp the sentinel so the (selected-away) loads/arithmetic stay in range.
    let k_safe = select(won, best_k, 0.0);
    let ku = u32::cast_from(k_safe) as usize;
    let acc_g = cand_ag[ku];
    let acc_h = cand_ah[ku];
    let oth_g = sum_gradient - acc_g;
    let oth_h = sum_hessian - acc_h;
    let slg = select(acc_left, acc_g, oth_g);
    let slh = select(acc_left, acc_h, oth_h);
    let lc_f = cand_lc[ku];
    let thr = thr_base + thr_step * i32::cast_from(k_safe);

    state[0] = select(any != 0.0, 1.0, 0.0);
    state[1] = select(won, best_gain, 0.0);
    state[2] = select(won, f64::cast_from(thr), 0.0);
    state[3] = select(won, lc_f, 0.0);
    state[4] = select(won, slg, 0.0);
    state[5] = select(won, slh, 0.0);
}

/// PARGAIN single-leaf kernel: [`find_best_splits_fused_staged_kernel`]'s twin
/// (identical signature + output) with the phase-1/2/3 split above. Lanes 0-31
/// are the REVERSE warp, 32-63 the FORWARD warp; phase 1 runs on lanes 0/32,
/// phase 2 on the full warps, phase 3 on lanes 0/32, and lane 0 merges via the
/// SAME [`merge_finalize_staged`].
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_staged_par_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // LEAF-LEVEL scalars (shared across the batch).
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);
    // Per-branch candidate state (phase 1 → phase 2/3).
    let mut rev_ag = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_ah = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_lc = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_ok = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_ag = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_ah = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_lc = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_ok = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    // Per-lane partials (32 per branch).
    let mut rev_pg = SharedMemory::<f64>::new(32usize);
    let mut rev_pk = SharedMemory::<f64>::new(32usize);
    let mut rev_pa = SharedMemory::<f64>::new(32usize);
    let mut fwd_pg = SharedMemory::<f64>::new(32usize);
    let mut fwd_pk = SharedMemory::<f64>::new(32usize);
    let mut fwd_pa = SharedMemory::<f64>::new(32usize);

    // Cooperative stage (identical to the staged kernel).
    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    let mut c = UNIT_POS as usize;
    while c < cells {
        sm[c] = hist[base + c];
        c += cd;
    }
    sync_cube();

    // PHASE 1: the two serial accumulation walkers, separate warps.
    if UNIT_POS == 0 {
        pargain_store_rev(
            &sm.to_slice(),
            &mut rev_ag.to_slice_mut(),
            &mut rev_ah.to_slice_mut(),
            &mut rev_lc.to_slice_mut(),
            &mut rev_ok.to_slice_mut(),
            num_bin[fi],
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            sum_hessian,
            num_data,
            rev_count[fi],
        );
    }
    if UNIT_POS == 32 {
        pargain_store_fwd(
            &sm.to_slice(),
            &mut fwd_ag.to_slice_mut(),
            &mut fwd_ah.to_slice_mut(),
            &mut fwd_lc.to_slice_mut(),
            &mut fwd_ok.to_slice_mut(),
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            sum_hessian,
            num_data,
            fwd_count[fi],
        );
    }
    sync_cube();

    // PHASE 2: warp-split parallel gain scans.
    if UNIT_POS < 32 {
        pargain_lane_scan(
            &rev_ag.to_slice(),
            &rev_ah.to_slice(),
            &rev_ok.to_slice(),
            &mut rev_pg.to_slice_mut(),
            &mut rev_pk.to_slice_mut(),
            &mut rev_pa.to_slice_mut(),
            UNIT_POS,
            rev_count[fi],
            0u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
        );
    } else {
        pargain_lane_scan(
            &fwd_ag.to_slice(),
            &fwd_ah.to_slice(),
            &fwd_ok.to_slice(),
            &mut fwd_pg.to_slice_mut(),
            &mut fwd_pk.to_slice_mut(),
            &mut fwd_pa.to_slice_mut(),
            UNIT_POS - 32,
            fwd_count[fi],
            1u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
        );
    }
    sync_cube();

    // PHASE 3: per-branch partial reduction + state assembly, separate warps.
    if UNIT_POS == 0 {
        pargain_assemble_state(
            &rev_pg.to_slice(),
            &rev_pk.to_slice(),
            &rev_pa.to_slice(),
            &rev_ag.to_slice(),
            &rev_ah.to_slice(),
            &rev_lc.to_slice(),
            &mut state_rev.to_slice_mut(),
            0u32,
            num_bin[fi] - 2,
            -1i32,
            sum_gradient,
            sum_hessian,
        );
    }
    if UNIT_POS == 32 {
        pargain_assemble_state(
            &fwd_pg.to_slice(),
            &fwd_pk.to_slice(),
            &fwd_pa.to_slice(),
            &fwd_ag.to_slice(),
            &fwd_ah.to_slice(),
            &fwd_lc.to_slice(),
            &mut state_fwd.to_slice_mut(),
            1u32,
            offset[fi],
            1i32,
            sum_gradient,
            sum_hessian,
        );
    }
    sync_cube();

    if UNIT_POS == 0 {
        merge_finalize_staged(
            &state_rev.to_slice(),
            &state_fwd.to_slice(),
            out,
            f * 12u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            sum_gradient,
            sum_hessian,
            num_data,
        );
    }
}

/// PARGAIN co-packed 2-slot sibling kernel:
/// [`find_best_splits_fused_siblings_staged_kernel`]'s twin (identical
/// signature, geometry `(n, 2, 1)`, and output layout) with the phase-1/2/3
/// split. See [`find_best_splits_fused_staged_par_kernel`].
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_staged_par_kernel(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    // SHARED per-feature params (length n; both siblings share the dataset layout).
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // SHARED cfg scalars.
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    // PER-SIBLING leaf scalars (A = smaller, B = larger).
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    num_data_a: i32,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    num_data_b: i32,
    // Per-sibling feature count (n) — the B sibling's out window offset.
    n_feats: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let is_b = CUBE_POS_Y != 0;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);
    let mut rev_ag = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_ah = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_lc = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_ok = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_ag = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_ah = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_lc = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut fwd_ok = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut rev_pg = SharedMemory::<f64>::new(32usize);
    let mut rev_pk = SharedMemory::<f64>::new(32usize);
    let mut rev_pa = SharedMemory::<f64>::new(32usize);
    let mut fwd_pg = SharedMemory::<f64>::new(32usize);
    let mut fwd_pk = SharedMemory::<f64>::new(32usize);
    let mut fwd_pa = SharedMemory::<f64>::new(32usize);

    // Select this cube's sibling scalars (an Array ref cannot be `select`ed, so
    // the stage loop below branches on the sibling instead).
    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

    // Cooperative stage from THIS sibling's histogram (pure copy).
    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    if is_b {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_b[base + c];
            c += cd;
        }
    } else {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_a[base + c];
            c += cd;
        }
    }
    sync_cube();

    if UNIT_POS == 0 {
        pargain_store_rev(
            &sm.to_slice(),
            &mut rev_ag.to_slice_mut(),
            &mut rev_ah.to_slice_mut(),
            &mut rev_lc.to_slice_mut(),
            &mut rev_ok.to_slice_mut(),
            num_bin[fi],
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            sum_hessian,
            num_data,
            rev_count[fi],
        );
    }
    if UNIT_POS == 32 {
        pargain_store_fwd(
            &sm.to_slice(),
            &mut fwd_ag.to_slice_mut(),
            &mut fwd_ah.to_slice_mut(),
            &mut fwd_lc.to_slice_mut(),
            &mut fwd_ok.to_slice_mut(),
            offset[fi],
            default_bin[fi],
            skip_default_bin[fi],
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            sum_hessian,
            num_data,
            fwd_count[fi],
        );
    }
    sync_cube();

    if UNIT_POS < 32 {
        pargain_lane_scan(
            &rev_ag.to_slice(),
            &rev_ah.to_slice(),
            &rev_ok.to_slice(),
            &mut rev_pg.to_slice_mut(),
            &mut rev_pk.to_slice_mut(),
            &mut rev_pa.to_slice_mut(),
            UNIT_POS,
            rev_count[fi],
            0u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
        );
    } else {
        pargain_lane_scan(
            &fwd_ag.to_slice(),
            &fwd_ah.to_slice(),
            &fwd_ok.to_slice(),
            &mut fwd_pg.to_slice_mut(),
            &mut fwd_pk.to_slice_mut(),
            &mut fwd_pa.to_slice_mut(),
            UNIT_POS - 32,
            fwd_count[fi],
            1u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
        );
    }
    sync_cube();

    if UNIT_POS == 0 {
        pargain_assemble_state(
            &rev_pg.to_slice(),
            &rev_pk.to_slice(),
            &rev_pa.to_slice(),
            &rev_ag.to_slice(),
            &rev_ah.to_slice(),
            &rev_lc.to_slice(),
            &mut state_rev.to_slice_mut(),
            0u32,
            num_bin[fi] - 2,
            -1i32,
            sum_gradient,
            sum_hessian,
        );
    }
    if UNIT_POS == 32 {
        pargain_assemble_state(
            &fwd_pg.to_slice(),
            &fwd_pk.to_slice(),
            &fwd_pa.to_slice(),
            &fwd_ag.to_slice(),
            &fwd_ah.to_slice(),
            &fwd_lc.to_slice(),
            &mut state_fwd.to_slice_mut(),
            1u32,
            offset[fi],
            1i32,
            sum_gradient,
            sum_hessian,
        );
    }
    sync_cube();

    if UNIT_POS == 0 {
        let g = select(is_b, n_feats + f, f);
        merge_finalize_staged(
            &state_rev.to_slice(),
            &state_fwd.to_slice(),
            out,
            g * 12u32,
            use_l1,
            lambda_l1,
            lambda_l2,
            sum_gradient,
            sum_hessian,
            num_data,
        );
    }
}

/// Launch the staged single-leaf scan — the SERIAL-branch staged kernel by
/// default, or its PARGAIN twin when `LGBM_SCAN_PARGAIN=1` (identical
/// signature, geometry, and — by the pargain module note — bit-identical
/// output). Single source for the four staged launch sites so the kernel
/// choice can never diverge between them.
///
/// # Safety
/// Same obligations as the direct staged launch: per-feature regions validated
/// `<= buf_len`, out sized `n*12`, per-feature arrays sized exactly `n`, and
/// `CubeCount::Static(n)` matching the kernels' no-guard `CUBE_POS_X < n`
/// contract.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_staged_single_scan<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    n: usize,
    hist: cubecl::server::Handle,
    buf_len: usize,
    h_out: cubecl::server::Handle,
    out_len: usize,
    h_slot: cubecl::server::Handle,
    h_numbin: cubecl::server::Handle,
    h_offset: cubecl::server::Handle,
    h_defbin: cubecl::server::Handle,
    h_skip: cubecl::server::Handle,
    h_rev: cubecl::server::Handle,
    h_fwd: cubecl::server::Handle,
    use_l1: bool,
    cfg: &GainConfig,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian_bumped: f64,
    num_data: i32,
) {
    macro_rules! launch_with {
        ($kernel:ident) => {
            unsafe {
                $kernel::launch(
                    client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(SCAN_STAGED_CUBE_DIM),
                    ArrayArg::from_raw_parts(hist.clone(), buf_len),
                    ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot.clone(), n),
                    ArrayArg::from_raw_parts(h_numbin.clone(), n),
                    ArrayArg::from_raw_parts(h_offset.clone(), n),
                    ArrayArg::from_raw_parts(h_defbin.clone(), n),
                    ArrayArg::from_raw_parts(h_skip.clone(), n),
                    ArrayArg::from_raw_parts(h_rev.clone(), n),
                    ArrayArg::from_raw_parts(h_fwd.clone(), n),
                    if use_l1 { 1u32 } else { 0u32 },
                    cfg.min_data_in_leaf,
                    cfg.min_sum_hessian_in_leaf,
                    cfg.lambda_l1,
                    cfg.lambda_l2,
                    min_gain_shift,
                    sum_gradient,
                    sum_hessian_bumped,
                    num_data,
                );
            }
        };
    }
    if scan_pargain_enabled() {
        launch_with!(find_best_splits_fused_staged_par_kernel);
    } else {
        launch_with!(find_best_splits_fused_staged_kernel);
    }
}

/// Sibling twin of [`launch_staged_single_scan`] — the co-packed 2-slot staged
/// launch, choosing the serial-branch or PARGAIN kernel by the same gate.
///
/// # Safety
/// Same obligations as the direct staged sibling launch; geometry
/// `CubeCount::Static(n, 2, 1)` matches the kernels' `CUBE_POS_X < n`,
/// `CUBE_POS_Y < 2` no-guard contract.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_staged_siblings_scan<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    n: usize,
    hist_a: cubecl::server::Handle,
    hist_b: cubecl::server::Handle,
    buf_len: usize,
    h_out: cubecl::server::Handle,
    out_len: usize,
    h_slot: cubecl::server::Handle,
    h_numbin: cubecl::server::Handle,
    h_offset: cubecl::server::Handle,
    h_defbin: cubecl::server::Handle,
    h_skip: cubecl::server::Handle,
    h_rev: cubecl::server::Handle,
    h_fwd: cubecl::server::Handle,
    use_l1: bool,
    cfg: &GainConfig,
    a_scalars: (f64, f64, f64, i32), // (min_gain_shift, sum_gradient, sum_hessian_bumped, num_data)
    b_scalars: (f64, f64, f64, i32),
) {
    macro_rules! launch_with {
        ($kernel:ident) => {
            unsafe {
                $kernel::launch(
                    client,
                    CubeCount::Static(n as u32, 2, 1),
                    CubeDim::new_1d(SCAN_STAGED_CUBE_DIM),
                    ArrayArg::from_raw_parts(hist_a.clone(), buf_len),
                    ArrayArg::from_raw_parts(hist_b.clone(), buf_len),
                    ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot.clone(), n),
                    ArrayArg::from_raw_parts(h_numbin.clone(), n),
                    ArrayArg::from_raw_parts(h_offset.clone(), n),
                    ArrayArg::from_raw_parts(h_defbin.clone(), n),
                    ArrayArg::from_raw_parts(h_skip.clone(), n),
                    ArrayArg::from_raw_parts(h_rev.clone(), n),
                    ArrayArg::from_raw_parts(h_fwd.clone(), n),
                    if use_l1 { 1u32 } else { 0u32 },
                    cfg.min_data_in_leaf,
                    cfg.min_sum_hessian_in_leaf,
                    cfg.lambda_l1,
                    cfg.lambda_l2,
                    a_scalars.0,
                    a_scalars.1,
                    a_scalars.2,
                    a_scalars.3,
                    b_scalars.0,
                    b_scalars.1,
                    b_scalars.2,
                    b_scalars.3,
                    n as u32,
                );
            }
        };
    }
    if scan_pargain_enabled() {
        launch_with!(find_best_splits_fused_siblings_staged_par_kernel);
    } else {
        launch_with!(find_best_splits_fused_siblings_staged_kernel);
    }
}

/// FUSED batched per-leaf best-split launcher, **generic over the runtime** `R`.
/// Finds the best split for EVERY feature in
/// `feats` in ONE launch of [`find_best_splits_fused_kernel`], returning one
/// [`SplitInfo`] per input feature **in input order** (T-lsx-01).
///
/// Both backends route through this single launcher: `CpuBackend` over
/// [`ActiveRuntime`](crate::runtime::ActiveRuntime) (cubecl-cpu) and `RocmBackend`
/// over the cubecl-hip runtime — the literal MERGE of the split-finding path. The
/// f64 result is bit-identical to the per-feature `find_best_splits_batched_f64_on`
/// loop (the shared `split_scan_body` is the same math, scanned over the same region
/// per feature; the only difference is launch count).
///
/// Host-side BEFORE the single launch (V5, threat T-mc5-01, CLAUDE.md non-neg #4),
/// for EACH feature: `num_bin == 0` / `2*num_bin` overflow / `na_as_missing` /
/// non-default `max_delta_step`/`path_smooth` → typed error;
/// `slot_off + 2*num_bin > buf.len()` → [`ComputeError::LengthMismatch`]. The
/// leaf-level `!(sum_hessian > 0.0)` is rejected ONCE. Empty `feats` → `Ok(vec![])`
/// with NO launch (T-mc5-03). All cubecl `unsafe` is confined here (CMP-01).
///
/// # Errors
/// As above; propagates length / scope / deferred-branch typed errors.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_batched_fused_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    buf: &[f64],
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<Vec<SplitInfo>, ComputeError> {
    // Empty batch: no launch (T-mc5-03). (Mirror the Handle variant's early return
    // so the buf-based path never allocates a device handle for nothing.)
    if feats.is_empty() {
        return Ok(Vec::new());
    }
    // Upload the host histogram buffer ONCE and delegate to the Handle-consuming
    // body — the SINGLE SOURCE of the fused-scan validation/launch/decode (the
    // resident scan reuses the SAME body with a device-resident Handle instead of
    // this fresh upload, guaranteeing byte-identical numerics between the host-buf and
    // resident scans). `as_bytes` is the same `CubeElement` call the kernel uses.
    let h_hist = client.create_from_slice(f64::as_bytes(buf));
    find_best_splits_batched_fused_f64_from_handle_on(
        client,
        h_hist,
        buf.len(),
        feats,
        cfg,
        sum_gradient,
        sum_hessian,
        num_data,
    )
}

/// Upload a host f64 slice to a device `Handle` and return it — a thin
/// CMP-01-respecting helper so cubecl-free callers (the oracle-harness resident-scan
/// parity test) can feed a raw Handle to
/// [`find_best_splits_batched_fused_f64_from_handle_on`] without naming `cubecl`
/// types. `as_bytes` is the same `CubeElement` call the kernel launchers use.
pub fn upload_f64_buffer<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    buf: &[f64],
) -> cubecl::server::Handle {
    client.create_from_slice(f64::as_bytes(buf))
}

/// Read an f64 device `Handle` back to a `Vec<f64>` — a thin
/// CMP-01-respecting helper so cubecl-free callers (the fused==host oracle) can read
/// a resident histogram Handle without naming `cubecl` types. `len` is the cell
/// count the Handle describes. `read_one_unchecked` is the same readback the kernel
/// launchers use.
pub fn read_f64_handle<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handle: cubecl::server::Handle,
    len: usize,
) -> Vec<f64> {
    let bytes = client.read_one_unchecked(handle);
    let v = f64::from_bytes(&bytes).to_vec();
    debug_assert_eq!(v.len(), len, "read_f64_handle length");
    v
}

/// Handle-consuming sibling of [`find_best_splits_batched_fused_f64_on`] —
/// the device-resident fused per-leaf split scan. IDENTICAL to the
/// host-buf launcher EXCEPT it CONSUMES a device `Handle` for the concatenated
/// stride-2 f64 histogram buffer (NO `client.create_from_slice` upload), so the
/// resident chain's fixed+compacted histogram never leaves the device. The SAME
/// per-feature V5 validation (against `buf_len`), the SAME leaf scalars
/// (`min_gain_shift` / `sum_hessian_bumped`), the SAME
/// [`find_best_splits_fused_kernel`], and the SAME 12-cell decode + accept-gate are
/// used — only the `n*12` `SplitInfo` cells are read back; the histogram Handle is
/// never read back. The host-buf launcher delegates here after a one-time upload, so
/// the two paths are byte-identical by construction (the resident==host invariant,
/// non-negotiable #2).
///
/// `hist_handle` must describe exactly `buf_len` f64 cells (the caller's slot_len);
/// the per-feature `[slot_off, slot_off + 2*num_bin)` regions are validated against
/// `buf_len` before launch.
///
/// # Errors
/// As [`find_best_splits_batched_fused_f64_on`] (length / scope / deferred-branch
/// typed errors).
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_batched_fused_f64_from_handle_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<Vec<SplitInfo>, ComputeError> {
    // Delegate to the shared inner body — the single source of the fused-scan
    // validation/launch/decode. The histogram lives in `hist_handle` (describing
    // `buf_len` f64 cells); only the per-feature region BOUNDS are validated.
    find_best_splits_fused_inner(
        client,
        hist_handle,
        buf_len,
        feats,
        cfg,
        sum_gradient,
        sum_hessian,
        num_data,
    )
}

/// Shared inner body for the fused per-leaf split scan — the
/// single source of the validation / scalar pre-step / launch / decode. Both the
/// host-buf launcher (after a one-time upload) and the resident
/// Handle-consuming launcher call this with an already-allocated `hist_handle`
/// describing `buf_len` f64 cells, so the host-buf and resident scans are
/// byte-identical (non-negotiable #2). All cubecl `unsafe` is confined here (CMP-01).
///
/// # Errors
/// As [`find_best_splits_batched_fused_f64_on`].
#[allow(clippy::too_many_arguments)]
fn find_best_splits_fused_inner<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<Vec<SplitInfo>, ComputeError> {
    // Empty batch: no launch (T-mc5-03).
    if feats.is_empty() {
        return Ok(Vec::new());
    }

    // Leaf-level, checked once: only the default smoothing/clamp path is
    // transcribed. Reject non-default values rather than mis-compute.
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_batched: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    // Reject non-positive OR NaN sum_hessian once (leaf-level; cnt_factor divides
    // by the bumped sum_hessian). `!(x > 0.0)` is deliberately NaN-catching.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_batched: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }

    // Env-gated (`LGBM_SCAN_PROF=1`) per-leaf scan round-trip timers. Inert
    // when off (the `Instant`s are still taken but never read — negligible, and parity
    // is untouched since no value/order changes). Marshal starts here.
    let _scan_prof = crate::fusion_prof::scan_enabled();
    let _t_marshal = std::time::Instant::now();

    // Per-feature V5 validation + per-feature device-array assembly (BEFORE launch).
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    for f in feats {
        if f.na_as_missing {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                         implemented"
                    .to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "find_best_splits_batched: slot_off + region overflows".to_string(),
            })?;
        if end > buf_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: buf_len,
            });
        }
        // Per-feature iteration counts — IDENTICAL to find_best_split_f64_on.
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
    }

    // LEAF-LEVEL scalars computed ONCE (identical across features) — the 2*kEpsilon
    // entry bump + min_gain_shift, exactly as find_best_split_f64_on does per call.
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    // Marshal done; upload begins.
    if _scan_prof {
        crate::fusion_prof::SCAN_MARSHAL_NS.fetch_add(
            _t_marshal.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let _t_upload = std::time::Instant::now();

    let out_len = n * 12;
    // The histogram buffer is the caller-supplied device Handle (resident path) or
    // the host-buf launcher's one-time upload — either way it describes `buf_len`
    // f64 cells and is never read back (only the SplitInfo cells are).
    let h_hist = hist_handle;
    // `out` needs NO zero-fill upload: every kernel variant (legacy lane-per-
    // feature, staged, and the autotune probes — all OVERWRITE-class) writes all
    // 12 cells of every feature window unconditionally at finalization, so an
    // uninitialized device allocation is observably identical to the old
    // uploaded-zeros buffer (one fewer H2D per scan).
    let h_out = client.empty(out_len * std::mem::size_of::<f64>());
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));

    // Upload done.
    if _scan_prof {
        crate::fusion_prof::SCAN_UPLOAD_NS.fetch_add(
            _t_upload.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    // DIAGNOSTIC (`LGBM_SCAN_DRAIN=1`): force-drain the async-queued f32-atomic
    // build by reading the resident histogram handle BEFORE the scan launch, so build
    // device-compute is attributed to `build_drain` instead of materializing inside the
    // scan's readback sync. Removes build/scan overlap → diagnostic only, off by default.
    if _scan_prof && crate::fusion_prof::scan_drain_enabled() {
        let t_drain = std::time::Instant::now();
        let _ = client.read_one_unchecked(h_hist.clone());
        crate::fusion_prof::SCAN_DRAIN_NS.fetch_add(
            t_drain.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let _t_launch = std::time::Instant::now();

    // STAGED cube-per-feature scan (the scan-occupancy + LDS-latency fix) —
    // takes precedence over BOTH the autotune W-set and the legacy W launch when
    //   * `LGBM_SCAN_STAGED != 0` (default ON — escape hatch),
    //   * the runtime is a real device (`R::name != "cpu"`; the cubecl-cpu
    //     anchor keeps the byte-unchanged serial kernel — the bit-exact merge
    //     gate must not depend on LDS/sync lowering there), and
    //   * every feature's region fits the LDS stage (≤ 256 bins — whole-launch
    //     fallback, never a per-feature mix).
    // Bit-identical results (see the staged-kernel module note), so no parity
    // seam is needed; the env is purely a perf escape hatch.
    #[cfg(feature = "gpu")]
    let staged = scan_staged_enabled()
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && num_bin_a.iter().all(|&nb| (nb as usize) * 2 <= SCAN_STAGE_MAX_CELLS);
    #[cfg(not(feature = "gpu"))]
    let staged = false;

    #[cfg(feature = "gpu")]
    if staged {
        // SAFETY: identical obligations to the legacy launch below (validated
        // per-feature regions ≤ buf_len; out window `f*12..f*12+12` within the
        // `n*12` allocation; per-feature arrays sized exactly `n`). The launch
        // geometry guarantees CUBE_POS_X < n (CubeCount::Static(n)), matching
        // the kernel's no-guard contract. The helper picks the serial-branch
        // staged kernel or its bit-identical PARGAIN twin (`LGBM_SCAN_PARGAIN`).
        unsafe {
            launch_staged_single_scan(
                client,
                n,
                h_hist.clone(),
                buf_len,
                h_out.clone(),
                out_len,
                h_slot.clone(),
                h_numbin.clone(),
                h_offset.clone(),
                h_defbin.clone(),
                h_skip.clone(),
                h_rev.clone(),
                h_fwd.clone(),
                use_l1,
                cfg,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                num_data,
            );
        }
    }

    // Autotune-or-fallback selection of the scan width W.
    //   (a) autotune default-ON UNLESS `LGBM_AUTOTUNE=0` OR an explicit
    //       `LGBM_SCAN_CUBEDIM` override (the override always wins — it is the documented
    //       escape hatch + the all-W parity seam). The tuner drives the launch over
    //       `SCAN_WSET`; its winner writes the real `h_out` (CloneInputGenerator → the
    //       final winning run uses the ORIGINAL handles, an OVERWRITE-class kernel).
    //   (b) else → the EXISTING `scan_cube_dim()` direct launch, byte-for-byte unchanged
    //       (covers `LGBM_AUTOTUNE=0`, an explicit `LGBM_SCAN_CUBEDIM`, and the non-rocm
    //       `scan_cube_dim()==1` bit-exact oracle path).
    #[cfg(feature = "gpu")]
    let autotuned = !staged
        && autotune::autotune_enabled()
        && std::env::var_os("LGBM_SCAN_CUBEDIM").is_none();
    #[cfg(not(feature = "gpu"))]
    let autotuned = false;

    #[cfg(feature = "gpu")]
    if autotuned {
        // The widest feature's bin count drives the per-feature slot width (the scan key
        // shape; bucket=0 since W tracks the feature/bin shape, not the row count).
        let num_bins = num_bin_a.iter().copied().max().unwrap_or(0).max(0) as u32;
        let handles: Vec<cubecl::server::Handle> = vec![
            h_hist.clone(),
            h_out.clone(),
            h_slot.clone(),
            h_numbin.clone(),
            h_offset.clone(),
            h_defbin.clone(),
            h_skip.clone(),
            h_rev.clone(),
            h_fwd.clone(),
        ];
        let set = std::sync::Arc::new(scan_wset_tunable_set(
            client.clone(),
            n,
            buf_len,
            out_len,
            num_bins,
            use_l1,
            cfg.min_data_in_leaf,
            cfg.min_sum_hessian_in_leaf,
            cfg.lambda_l1,
            cfg.lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian_bumped,
            num_data,
        ));
        SCAN_TUNER.execute(&autotune::cache_namespace_id(), client, set, handles);
    }

    if !autotuned && !staged {
        // Scan-occupancy lever: pack one feature per LANE. `scan_cube_dim()`
        // (env `LGBM_SCAN_CUBEDIM`; rocm default W=64, W=1 = byte-identical to the
        // original) is the cube width W; `CubeCount = ceil(n / W)`. The kernel indexes
        // features by the global lane `ABSOLUTE_POS` and guards `f < n_feats`, so the
        // tail cube's spare lanes no-op and the result is bit-identical to W=1 for
        // every W.
        let scan_w = scan_cube_dim();
        let cube_count = (n as u32).div_ceil(scan_w);

        // SAFETY: every handle is sized to its slice and outlives the launch. Lane `f`
        // (`ABSOLUTE_POS`, guarded `< n_feats` in the kernel) reads only
        // `[slot_off[f], slot_off[f]+2*num_bin[f])` — each validated `<= buf.len()` above
        // — and writes only `out[f*12 .. f*12+12]` within the `n*12` allocation; the
        // shared `split_scan_body` carries the same in-range / negative-`t` clamp guards
        // as the single-feature kernel. All per-feature index arrays have exactly `n`
        // elements. All cubecl unsafe is confined here (CMP-01).
        unsafe {
            find_best_splits_fused_kernel::launch(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(scan_w),
                ArrayArg::from_raw_parts(h_hist, buf_len),
                ArrayArg::from_raw_parts(h_out.clone(), out_len),
                ArrayArg::from_raw_parts(h_slot, n),
                ArrayArg::from_raw_parts(h_numbin, n),
                ArrayArg::from_raw_parts(h_offset, n),
                ArrayArg::from_raw_parts(h_defbin, n),
                ArrayArg::from_raw_parts(h_skip, n),
                ArrayArg::from_raw_parts(h_rev, n),
                ArrayArg::from_raw_parts(h_fwd, n),
                if use_l1 { 1u32 } else { 0u32 },
                cfg.min_data_in_leaf,
                cfg.min_sum_hessian_in_leaf,
                cfg.lambda_l1,
                cfg.lambda_l2,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                num_data,
                n as u32,
            );
        }
    }

    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);

    // Decode each feature's 12-cell window with the SAME accept-gate as
    // find_best_split_f64_on (feature_histogram.hpp:1031-1056). Push in input order.
    let penalty = 1.0f64;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let base = f * 12;
        let is_splittable = cells[base] != 0.0;
        let raw_threshold = cells[base + 1] as u32;
        let raw_gain = cells[base + 2];
        let left_count = cells[base + 3] as i32;
        let right_count = cells[base + 4] as i32;
        let left_sum_gradient = cells[base + 5];
        let left_sum_hessian = cells[base + 6];
        let right_sum_gradient = cells[base + 7];
        let right_sum_hessian = cells[base + 8];
        let default_left = cells[base + 9] != 0.0;
        let left_output = cells[base + 10];
        let right_output = cells[base + 11];

        if is_splittable && raw_gain > f64::NEG_INFINITY {
            out.push(SplitInfo {
                threshold: raw_threshold,
                gain: (raw_gain - min_gain_shift) * penalty,
                left_count,
                right_count,
                left_sum_gradient,
                left_sum_hessian,
                right_sum_gradient,
                right_sum_hessian,
                left_output,
                right_output,
                default_left,
            });
        } else {
            out.push(SplitInfo::none());
        }
    }
    // Launch+readback+decode done.
    if _scan_prof {
        crate::fusion_prof::SCAN_LAUNCH_NS.fetch_add(
            _t_launch.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    Ok(out)
}

/// CO-PACKED 2-slot resident best-split launcher — scans BOTH
/// siblings of a split (the smaller child `hist_a_handle` and the larger child
/// `hist_b_handle`) in ONE launch of [`find_best_splits_fused_siblings_kernel`] with
/// ONE `read_one_unchecked` readback, returning `(vec_a, vec_b)` — one
/// [`SplitInfo`] per input feature, in input order, per sibling.
///
/// This is the device-launch-structural win of co-packing: it replaces the TWO
/// separate `find_best_splits_fused_inner` calls (two launches, two blocking
/// readbacks / syncs) the fall-back uses with ONE launch + ONE sync. The per-feature
/// V5 validation + device-array assembly is done ONCE (the feature/region layout is
/// SHARED between siblings — both children have the same dataset feature layout). The
/// `2*kEpsilon` bump + `min_gain_shift` are computed ONCE PER SIBLING with that
/// sibling's RAW totals, exactly as the single-slot path computes them once. The
/// 12-cell decode + accept-gate is the SAME as [`find_best_splits_fused_inner`],
/// applied to BOTH halves (features `0..n` against `min_gain_shift_a`, features
/// `n..2n` against `min_gain_shift_b`).
///
/// Bit-exact by construction: each feature's sequential scan is the SAME shared
/// `split_scan_body` over the SAME disjoint region with that sibling's leaf scalars
/// (no reorder); only WHICH launch it runs in changes vs two single-slot scans.
///
/// `hist_a_handle` / `hist_b_handle` each describe exactly `buf_len` f64 cells (the
/// caller's shared `slot_len`). Empty `feats` → `Ok((vec![], vec![]))` with NO
/// launch. Both sibling `sum_hessian`s are rejected once each (cnt_factor divides by
/// the bumped sum_hessian). All cubecl `unsafe` is confined here (CMP-01).
///
/// # Errors
/// As [`find_best_splits_batched_fused_f64_on`] (length / scope / deferred-branch
/// typed errors), checked once on the SHARED `feats`; plus `!(sum_hessian > 0.0)`
/// per sibling.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_from_handles_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_a_handle: cubecl::server::Handle,
    hist_b_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    // (sum_gradient, sum_hessian, num_data) per sibling (A = smaller, B = larger).
    a_totals: (f64, f64, i32),
    b_totals: (f64, f64, i32),
) -> Result<(Vec<SplitInfo>, Vec<SplitInfo>), ComputeError> {
    // Empty batch: no launch (mirror the single-slot T-mc5-03 early return).
    if feats.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Env-gated (`LGBM_SCAN_PROF=1`) scan round-trip profiling. Bound here so the
    // LGBM_SCAN_DRAIN co-pack analog below can gate identically to the single-leaf path
    // (find_best_splits_fused_inner). Inert when off (parity untouched).
    let _scan_prof = crate::fusion_prof::scan_enabled();

    // Leaf-level, checked once on the shared cfg.
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    let (sum_gradient_a, sum_hessian_a, num_data_a) = a_totals;
    let (sum_gradient_b, sum_hessian_b, num_data_b) = b_totals;
    // Reject non-positive OR NaN sum_hessian once PER SIBLING (cnt_factor divides by
    // the bumped sum_hessian). `!(x > 0.0)` is deliberately NaN-catching.
    //
    // WR-02: this is a DEFENSIVE / unreachable boundary check, NOT the gate. The
    // co-pack eligibility gate in the learner (`find_best_splits`) is the SINGLE
    // SOURCE OF TRUTH for scannability: co-pack only fires when BOTH siblings already
    // satisfy `sum_hessians > 0.0 && num_data_in_leaf > 0`, mirroring the per-leaf
    // `scan_leaf_histogram` early-out to `none()`. So in the production co-pack path
    // these rejects can never trip. They remain as a hard error only for the rare
    // standalone/test caller of `find_best_splits_siblings`: a non-positive-hessian
    // sibling there is a contract violation, not a "this leaf can't split" — the
    // learner has already degraded such a leaf before ever reaching co-pack.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_a > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings: smaller-sibling sum_hessian must be > 0".to_string(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_b > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings: larger-sibling sum_hessian must be > 0".to_string(),
        });
    }

    // Per-feature V5 validation + per-feature device-array assembly ONCE (the
    // feature/region layout is SHARED between siblings).
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    for f in feats {
        if f.na_as_missing {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                         implemented"
                    .to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "find_best_splits_siblings: slot_off + region overflows".to_string(),
            })?;
        if end > buf_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: buf_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
    }

    // LEAF-LEVEL scalars computed ONCE PER SIBLING — the 2*kEpsilon entry bump +
    // min_gain_shift with each sibling's RAW totals, exactly as the single-slot path
    // computes them once.
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let use_l1 = cfg.use_l1();
    let sum_hessian_a_bumped = sum_hessian_a + two_eps;
    let min_gain_shift_a = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_a,
        sum_hessian_a_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    ) + cfg.min_gain_to_split;
    let sum_hessian_b_bumped = sum_hessian_b + two_eps;
    let min_gain_shift_b = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_b,
        sum_hessian_b_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    ) + cfg.min_gain_to_split;

    // `out` packs A then B contiguously: 2*n features × 12 cells. NO zero-fill
    // upload: every kernel variant writes all 12 cells of every window
    // unconditionally (see the single-slot launcher note).
    let out_len = 2 * n * 12;
    let h_out = client.empty(out_len * std::mem::size_of::<f64>());
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));

    // DIAGNOSTIC (`LGBM_SCAN_DRAIN=1`) — co-pack analog: drain
    // BOTH sibling resident histogram handles before the scan launch so each child's async
    // f32-atomic build is attributed to SCAN_DRAIN_NS, not the scan readback, mirroring
    // the single-leaf drain on the production co-pack path.
    // Gated identically (`_scan_prof && scan_drain_enabled()`) → off by default, parity-neutral.
    if _scan_prof && crate::fusion_prof::scan_drain_enabled() {
        let t_drain = std::time::Instant::now();
        let _ = client.read_one_unchecked(hist_a_handle.clone());
        let _ = client.read_one_unchecked(hist_b_handle.clone());
        crate::fusion_prof::SCAN_DRAIN_NS.fetch_add(
            t_drain.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    // STAGED cube-per-(feature, sibling) scan — the SAME precedence/gates as the
    // single-slot launcher (`LGBM_SCAN_STAGED != 0`, real device runtime, every
    // feature ≤ 256 bins). ONE launch of `CubeCount::Static(n, 2, 1)` covers
    // both siblings; bit-identical output layout + values (staged-kernel note).
    #[cfg(feature = "gpu")]
    let staged = scan_staged_enabled()
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && num_bin_a.iter().all(|&nb| (nb as usize) * 2 <= SCAN_STAGE_MAX_CELLS);
    #[cfg(not(feature = "gpu"))]
    let staged = false;

    #[cfg(feature = "gpu")]
    if staged {
        // SAFETY: identical obligations to the legacy sibling launch below; the
        // geometry guarantees CUBE_POS_X < n and CUBE_POS_Y < 2, matching the
        // kernel's no-guard contract. The helper picks the serial-branch staged
        // kernel or its bit-identical PARGAIN twin (`LGBM_SCAN_PARGAIN`).
        unsafe {
            launch_staged_siblings_scan(
                client,
                n,
                hist_a_handle.clone(),
                hist_b_handle.clone(),
                buf_len,
                h_out.clone(),
                out_len,
                h_slot.clone(),
                h_numbin.clone(),
                h_offset.clone(),
                h_defbin.clone(),
                h_skip.clone(),
                h_rev.clone(),
                h_fwd.clone(),
                use_l1,
                cfg,
                (min_gain_shift_a, sum_gradient_a, sum_hessian_a_bumped, num_data_a),
                (min_gain_shift_b, sum_gradient_b, sum_hessian_b_bumped, num_data_b),
            );
        }
    }

    // Autotune-or-fallback selection of the scan width W — the SAME
    // guard as the single-leaf `find_best_splits_fused_inner`, here for the co-pack
    // 2-slot sibling scan (the production hot path). The sibling kernel is the
    // SAME OVERWRITE class (each lane writes a fresh 12-cell window), so its tunable set
    // also uses CloneInputGenerator; it runs under the SEPARATE `SCAN_SIBLINGS_TUNER`
    // namespace so its cache never collides with the single-leaf scan's.
    #[cfg(feature = "gpu")]
    let autotuned = !staged
        && autotune::autotune_enabled()
        && std::env::var_os("LGBM_SCAN_CUBEDIM").is_none();
    #[cfg(not(feature = "gpu"))]
    let autotuned = false;

    #[cfg(feature = "gpu")]
    if autotuned {
        let num_bins = num_bin_a.iter().copied().max().unwrap_or(0).max(0) as u32;
        let handles: Vec<cubecl::server::Handle> = vec![
            hist_a_handle.clone(),
            hist_b_handle.clone(),
            h_out.clone(),
            h_slot.clone(),
            h_numbin.clone(),
            h_offset.clone(),
            h_defbin.clone(),
            h_skip.clone(),
            h_rev.clone(),
            h_fwd.clone(),
        ];
        let set = std::sync::Arc::new(scan_wset_siblings_tunable_set(
            client.clone(),
            n,
            buf_len,
            out_len,
            num_bins,
            use_l1,
            cfg.min_data_in_leaf,
            cfg.min_sum_hessian_in_leaf,
            cfg.lambda_l1,
            cfg.lambda_l2,
            min_gain_shift_a,
            sum_gradient_a,
            sum_hessian_a_bumped,
            num_data_a,
            min_gain_shift_b,
            sum_gradient_b,
            sum_hessian_b_bumped,
            num_data_b,
        ));
        SCAN_SIBLINGS_TUNER.execute(&autotune::cache_namespace_id(), client, set, handles);
    }

    if !autotuned && !staged {
        // CubeCount over 2*n feature-slots (the lane mapping packs A then B).
        let scan_w = scan_cube_dim();
        let cube_count = (2 * n as u32).div_ceil(scan_w);

        // SAFETY: both histogram handles describe `buf_len` f64 cells; every per-feature
        // region `[slot_off, slot_off+2*num_bin)` is validated `<= buf_len` above; all
        // per-feature index arrays have exactly `n` elements; lane `g` (guarded
        // `< 2*n_feats`) reads only its sibling's validated region and writes only
        // `out[g*12 .. g*12+12]` within the `2*n*12` allocation. All cubecl unsafe is
        // confined here (CMP-01).
        unsafe {
            find_best_splits_fused_siblings_kernel::launch(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(scan_w),
                ArrayArg::from_raw_parts(hist_a_handle, buf_len),
                ArrayArg::from_raw_parts(hist_b_handle, buf_len),
                ArrayArg::from_raw_parts(h_out.clone(), out_len),
                ArrayArg::from_raw_parts(h_slot, n),
                ArrayArg::from_raw_parts(h_numbin, n),
                ArrayArg::from_raw_parts(h_offset, n),
                ArrayArg::from_raw_parts(h_defbin, n),
                ArrayArg::from_raw_parts(h_skip, n),
                ArrayArg::from_raw_parts(h_rev, n),
                ArrayArg::from_raw_parts(h_fwd, n),
                if use_l1 { 1u32 } else { 0u32 },
                cfg.min_data_in_leaf,
                cfg.min_sum_hessian_in_leaf,
                cfg.lambda_l1,
                cfg.lambda_l2,
                min_gain_shift_a,
                sum_gradient_a,
                sum_hessian_a_bumped,
                num_data_a,
                min_gain_shift_b,
                sum_gradient_b,
                sum_hessian_b_bumped,
                num_data_b,
                n as u32,
            );
        }
    }

    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);

    // Decode BOTH halves with the SAME 12-cell accept-gate as
    // find_best_splits_fused_inner. Features `0..n` (sibling A, offset 0) against
    // `min_gain_shift_a`; features `n..2n` (sibling B, offset n) against
    // `min_gain_shift_b`. Each pushed in input order.
    let decode_half = |feat_offset: usize, min_gain_shift: f64| -> Vec<SplitInfo> {
        let penalty = 1.0f64;
        let mut out = Vec::with_capacity(n);
        for f in 0..n {
            let base = (feat_offset + f) * 12;
            let is_splittable = cells[base] != 0.0;
            let raw_threshold = cells[base + 1] as u32;
            let raw_gain = cells[base + 2];
            let left_count = cells[base + 3] as i32;
            let right_count = cells[base + 4] as i32;
            let left_sum_gradient = cells[base + 5];
            let left_sum_hessian = cells[base + 6];
            let right_sum_gradient = cells[base + 7];
            let right_sum_hessian = cells[base + 8];
            let default_left = cells[base + 9] != 0.0;
            let left_output = cells[base + 10];
            let right_output = cells[base + 11];

            if is_splittable && raw_gain > f64::NEG_INFINITY {
                out.push(SplitInfo {
                    threshold: raw_threshold,
                    gain: (raw_gain - min_gain_shift) * penalty,
                    left_count,
                    right_count,
                    left_sum_gradient,
                    left_sum_hessian,
                    right_sum_gradient,
                    right_sum_hessian,
                    left_output,
                    right_output,
                    default_left,
                });
            } else {
                out.push(SplitInfo::none());
            }
        }
        out
    };

    let vec_a = decode_half(0, min_gain_shift_a);
    let vec_b = decode_half(n, min_gain_shift_b);
    Ok((vec_a, vec_b))
}

// ============================================================================
// DEVICE-SIDE cross-feature reduce of a leaf's raw per-feature scan output
// DIRECTLY into a target `SplitSoa` leaf slot — a zero-per-feature-array-readback
// path wired into the resident grow loop to retire host-argmax readbacks.
//
// The reduce kernel `sync_best_split_leaf_kernel` (best_split.rs) is NOT reused
// here. Its `take` condition is strict-gain-only (first-task-index-wins-on-a-tie)
// — correct for its OWN 2-record winner+pad frontier-fold, but WRONG for a genuine
// N-feature cross-feature reduce where an EXACT gain tie must be broken by the
// LOWER real feature index (the cpu-f64 `SerialTreeLearner::split_gt` anchor
// `argmax_over_resident_splits` implements). So this builds ONE NEW kernel that
// reuses BOTH proven idioms verbatim: the array-only output-slot-as-accumulator
// SHAPE (`sync_best_split_leaf_kernel`) AND the two-key (gain, real-feature-index)
// tie-break FORMULA (`find_best_leaf_kernel`, the SAME formula proven bit-exact
// for the cross-LEAF pick).
// ============================================================================

/// Cross-FEATURE reduce BODY. Decodes one leaf's raw `n*12`-cell scan
/// window (`find_best_splits_fused_kernel` / `find_best_splits_fused_siblings_kernel`
/// / `build_fix_scan_fused_kernel` all emit the SAME layout) with the SAME
/// accept-gate + net-gain + `split_gt` tie-break as the host `argmax_over_resident_splits`,
/// writing the winner DIRECTLY into a target `SplitSoa` slot (`out_slot`). ZERO
/// device→host transfer.
///
/// The 12-cell decode order is IDENTICAL to `find_best_splits_fused_inner`'s host
/// decode: `[0]`=is_splittable flag (0.0/1.0), `[1]`=raw_threshold, `[2]`=RAW gain
/// (net of `min_gain_shift` only at export), `[5..9]`=left/right grad/hess sums,
/// `[9]`=default_left flag (0.0/1.0). Cells `[3]`/`[4]` (counts), `[10]`/`[11]`
/// (outputs) are NOT carried by `SplitSoa` (it stores the 4 sums, not counts/outputs).
///
/// Array-only-`select` discipline (matching `sync_best_split_leaf_kernel`): the running
/// winner lives in `out_*[slot]` (indexed by the constant `slot`, always dominates);
/// every `select` operand is an array load or a literal — the only unification cubecl
/// 0.10 accepts. Selection compares the RAW gain (`raw[dbase+2]`) directly (monotone in
/// the exported net gain for a SINGLE leaf, since `net = (raw - min_gain_shift) * penalty`
/// with `penalty = 1` and one shift per leaf ⇒ raw-argmax == net-argmax AND raw-ties ==
/// net-ties), so the winning gain is copied verbatim during the loop and converted to the
/// reported net gain ONCE after the loop (bit-exact to the host `(raw_gain - min_gain_shift)
/// * penalty`; a no-split slot's `neg_inf` sentinel maps `neg_inf → neg_inf`). The
/// two-key tie-break reuses `find_best_leaf_kernel`'s formula: strictly-greater gain OR
/// (exact gain tie AND strictly-lower real feature key), with the running feature key in
/// `out_feat[slot]` (seeded `-1.0`; the first valid always takes via the gain path since
/// every finite gain beats the `neg_inf` seed, so the `-1.0` key never decides a pick —
/// exactly as the host seeds `best_real = -1`). No readback.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn reduce_scan_output_into_leaf_kernel(
    raw: &Array<f64>,
    real_feats: &Array<f64>,
    out_valid: &mut Array<f64>,
    out_gain: &mut Array<f64>,
    out_feat: &mut Array<f64>,
    out_thr: &mut Array<f64>,
    out_dleft: &mut Array<f64>,
    out_ncat: &mut Array<f64>,
    out_lsum_g: &mut Array<f64>,
    out_lsum_h: &mut Array<f64>,
    out_rsum_g: &mut Array<f64>,
    out_rsum_h: &mut Array<f64>,
    out_lval: &mut Array<f64>,
    out_rval: &mut Array<f64>,
    raw_base: u32,
    n_feats: u32,
    out_slot: u32,
    min_gain_shift: f64,
    penalty: f64,
    neg_inf: f64,
) {
    let rb = raw_base as usize;
    let n = n_feats as usize;
    let slot = out_slot as usize;
    // Seed the slot to the no-valid-split sentinel (mirrors `sync_best_split_leaf_kernel`:
    // valid=0, gain=neg_inf, feat=-1, thr/dleft/ncat/4-sums = 0). Array writes from the
    // `neg_inf` arg are fine (no loop-carried local to init from a scalar arg).
    out_valid[slot] = 0.0;
    out_gain[slot] = neg_inf;
    out_feat[slot] = -1.0;
    out_thr[slot] = 0.0;
    out_dleft[slot] = 0.0;
    out_ncat[slot] = 0.0;
    out_lsum_g[slot] = 0.0;
    out_lsum_h[slot] = 0.0;
    out_rsum_g[slot] = 0.0;
    out_rsum_h[slot] = 0.0;
    out_lval[slot] = 0.0;
    out_rval[slot] = 0.0;
    for t in 0..n {
        let dbase = rb + t * 12;
        // Accept-gate: is_splittable (0.0/1.0 flag) && raw_gain > neg_inf — IDENTICAL to
        // the host decode (`cells[base] != 0.0 && raw_gain > f64::NEG_INFINITY`).
        let v = (raw[dbase] != 0.0) && (raw[dbase + 2] > neg_inf);
        // Two-key split_gt compare on the RAW gain (monotone in the exported net gain for a
        // single leaf) + the real feature key. Array-only operands (running best in
        // `out_gain[slot]` / `out_feat[slot]`).
        let strictly_gain = raw[dbase + 2] > out_gain[slot];
        let tie_gain = raw[dbase + 2] == out_gain[slot];
        let feat_lower = real_feats[t] < out_feat[slot];
        let better = strictly_gain || (tie_gain && feat_lower);
        let take = v && better;
        out_valid[slot] = select(take, 1.0, out_valid[slot]);
        out_gain[slot] = select(take, raw[dbase + 2], out_gain[slot]);
        out_feat[slot] = select(take, real_feats[t], out_feat[slot]);
        out_thr[slot] = select(take, raw[dbase + 1], out_thr[slot]);
        out_dleft[slot] = select(take, raw[dbase + 9], out_dleft[slot]);
        out_ncat[slot] = select(take, 0.0, out_ncat[slot]);
        out_lsum_g[slot] = select(take, raw[dbase + 5], out_lsum_g[slot]);
        out_lsum_h[slot] = select(take, raw[dbase + 6], out_lsum_h[slot]);
        out_rsum_g[slot] = select(take, raw[dbase + 7], out_rsum_g[slot]);
        out_rsum_h[slot] = select(take, raw[dbase + 8], out_rsum_h[slot]);
        // Carry the winning feature's child leaf OUTPUTS device→device
        // (raw cells [10]/[11], the SAME `left_output`/`right_output` the host decode reads —
        // NOT recomputed from the eps-adjusted sums). Same array-only-`select` discipline.
        out_lval[slot] = select(take, raw[dbase + 10], out_lval[slot]);
        out_rval[slot] = select(take, raw[dbase + 11], out_rval[slot]);
    }
    // Convert the running RAW winner gain to the reported NET gain, ONCE — bit-exact to the
    // host decode `(raw_gain - min_gain_shift) * penalty`. A no-split slot's `neg_inf`
    // sentinel maps `(neg_inf - min_gain_shift) * penalty == neg_inf`, matching
    // `SplitInfo::none()`'s `gain == kMinScore`.
    out_gain[slot] = (out_gain[slot] - min_gain_shift) * penalty;
}

/// Launch [`reduce_scan_output_into_leaf_kernel`] over `n_feats` raw cells starting at
/// `raw_base`, folding the winner into `out`'s `out_slot`. Confines the cubecl `unsafe`
/// (CMP-01). Shared by the single-leaf, co-pack (twice), and f64-fused-escape-hatch
/// launchers. `real_feats` must have `>= n_feats` elements; `h_raw` must describe
/// `>= raw_base + n_feats*12` cells; `out_slot < out.len`. All bounds are host-proven by
/// the callers' V5 validation before launch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_reduce_into_leaf<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    h_raw: cubecl::server::Handle,
    raw_len: usize,
    real_feats: &[i32],
    n_feats: usize,
    out: &crate::kernels::best_split::SplitSoa,
    out_slot: usize,
    raw_base: usize,
    min_gain_shift: f64,
) {
    let rf: Vec<f64> = real_feats.iter().take(n_feats).map(|&r| f64::from(r)).collect();
    let h_rf = client.create_from_slice(f64::as_bytes(&rf));
    // SAFETY: single-owner static geometry (Route C). The kernel seeds + folds `out_*[slot]`
    // (`out_slot < out.len`, caller-validated) and reads only `raw[raw_base .. raw_base +
    // n_feats*12)` (caller-validated `<= raw_len`) and `real_feats[0 .. n_feats)`
    // (`h_rf` sized `n_feats`). Every handle outlives the launch. All cubecl unsafe confined
    // here (CMP-01).
    unsafe {
        reduce_scan_output_into_leaf_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_raw, raw_len),
            ArrayArg::from_raw_parts(h_rf, n_feats),
            ArrayArg::from_raw_parts(out.valid.clone(), out.len),
            ArrayArg::from_raw_parts(out.gain.clone(), out.len),
            ArrayArg::from_raw_parts(out.feat.clone(), out.len),
            ArrayArg::from_raw_parts(out.thr.clone(), out.len),
            ArrayArg::from_raw_parts(out.dleft.clone(), out.len),
            ArrayArg::from_raw_parts(out.ncat.clone(), out.len),
            ArrayArg::from_raw_parts(out.left_sum_gradients.clone(), out.len),
            ArrayArg::from_raw_parts(out.left_sum_hessians.clone(), out.len),
            ArrayArg::from_raw_parts(out.right_sum_gradients.clone(), out.len),
            ArrayArg::from_raw_parts(out.right_sum_hessians.clone(), out.len),
            ArrayArg::from_raw_parts(out.left_output.clone(), out.len),
            ArrayArg::from_raw_parts(out.right_output.clone(), out.len),
            raw_base as u32,
            n_feats as u32,
            out_slot as u32,
            min_gain_shift,
            1.0f64,
            f64::NEG_INFINITY,
        );
    }
}

/// Shared V5 validation + `min_gain_shift` pre-step + fused-scan launch for the
/// no-readback reduce launchers. Byte-for-byte the SAME per-feature V5
/// validation + the SAME `2*kEpsilon` bump + `min_gain_shift` + the SAME
/// [`find_best_splits_fused_kernel`] launch as [`find_best_splits_fused_inner`], but it
/// RETURNS the raw `h_out` Handle (n*12 cells) + `min_gain_shift` INSTEAD of reading it
/// back — the reduce kernel folds it on device. The direct `scan_cube_dim()` launch is
/// used (NOT the autotune path): `W` is bit-neutral (every `W` byte-identical),
/// so skipping the tuner here keeps the reduce launchers self-contained and leaves the
/// shipped host-readback `find_best_splits_fused_inner` byte-unchanged (additive
/// only). Empty `feats` → `Ok(None)` (no launch). Mirrors `find_best_splits_fused_inner`'s
/// error contract exactly.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn fused_scan_to_raw_handle<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<Option<(cubecl::server::Handle, usize, f64)>, ComputeError> {
    if feats.is_empty() {
        return Ok(None);
    }
    // Leaf-level scope checks (identical to `find_best_splits_fused_inner`).
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_reduce: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_reduce: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }

    // Per-feature V5 validation + device-array assembly (BEFORE launch) — IDENTICAL to
    // `find_best_splits_fused_inner`, including the whole-batch `na_as_missing` reject.
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    for f in feats {
        if f.na_as_missing {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                         implemented"
                    .to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "find_best_splits_reduce: slot_off + region overflows".to_string(),
            })?;
        if end > buf_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: buf_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
    }

    // LEAF-LEVEL scalars ONCE — the 2*kEpsilon bump + min_gain_shift (identical math).
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let out_len = n * 12;
    // NO zero-fill upload: the scan kernel (legacy or staged) writes all 12 cells
    // of every feature window unconditionally, and the reduce kernel reads only
    // those cells — an uninitialized allocation is observably identical.
    let h_out = client.empty(out_len * std::mem::size_of::<f64>());
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));

    // STAGED cube-per-feature branch (opt-in `LGBM_SCAN_STAGED=1`) on the LIVE
    // no-readback route — same gates as `find_best_splits_fused_inner` (real
    // device runtime, every feature ≤ 256 bins). Bit-identical output cells, so
    // the reduce launcher downstream is untouched.
    #[cfg(feature = "gpu")]
    if scan_staged_enabled()
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && num_bin_a.iter().all(|&nb| (nb as usize) * 2 <= SCAN_STAGE_MAX_CELLS)
    {
        // SAFETY: identical obligations to the legacy launch below; the geometry
        // guarantees CUBE_POS_X < n, matching the kernel's no-guard contract.
        // The helper picks the serial-branch staged kernel or its bit-identical
        // PARGAIN twin (`LGBM_SCAN_PARGAIN`).
        unsafe {
            launch_staged_single_scan(
                client,
                n,
                hist_handle,
                buf_len,
                h_out.clone(),
                out_len,
                h_slot,
                h_numbin,
                h_offset,
                h_defbin,
                h_skip,
                h_rev,
                h_fwd,
                use_l1,
                cfg,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                num_data,
            );
        }
        return Ok(Some((h_out, out_len, min_gain_shift)));
    }

    // Direct `scan_cube_dim()` launch (bit-neutral W; no autotune — see fn doc). This is the
    // SAME `find_best_splits_fused_kernel` the host-readback path launches; only the readback
    // is dropped (the reduce kernel consumes `h_out` on device).
    let scan_w = scan_cube_dim();
    let cube_count = (n as u32).div_ceil(scan_w);
    // SAFETY: every per-feature region `[slot_off[f], slot_off[f]+2*num_bin[f])` is validated
    // `<= buf_len` above; lane `f` (guarded `< n_feats`) reads only its region and writes
    // `out[f*12 .. f*12+12]` within the `n*12` allocation; all per-feature index arrays have
    // exactly `n` elements. All cubecl unsafe confined here (CMP-01).
    unsafe {
        find_best_splits_fused_kernel::launch(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(scan_w),
            ArrayArg::from_raw_parts(hist_handle, buf_len),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            ArrayArg::from_raw_parts(h_slot, n),
            ArrayArg::from_raw_parts(h_numbin, n),
            ArrayArg::from_raw_parts(h_offset, n),
            ArrayArg::from_raw_parts(h_defbin, n),
            ArrayArg::from_raw_parts(h_skip, n),
            ArrayArg::from_raw_parts(h_rev, n),
            ArrayArg::from_raw_parts(h_fwd, n),
            if use_l1 { 1u32 } else { 0u32 },
            cfg.min_data_in_leaf,
            cfg.min_sum_hessian_in_leaf,
            cfg.lambda_l1,
            cfg.lambda_l2,
            min_gain_shift,
            sum_gradient,
            sum_hessian_bumped,
            num_data,
            n as u32,
        );
    }
    Ok(Some((h_out, out_len, min_gain_shift)))
}

/// SINGLE-LEAF no-readback reduce-into-leaf launcher. Mirrors
/// [`find_best_splits_batched_fused_f64_from_handle_on`]'s signature + validation but
/// takes `real_feats: &[i32]` (feature-position → real feature index, the tie-break key)
/// + a target [`SplitSoa`] + `out_leaf` INSTEAD of returning `Vec<SplitInfo>`. Runs the
/// SAME per-feature V5 validation + the SAME [`find_best_splits_fused_kernel`] scan, then
/// folds the winner into `out`'s `out_leaf` slot via [`reduce_scan_output_into_leaf_kernel`]
/// — issuing ZERO device→host transfer of the per-feature array (the histogram Handle and
/// the raw scan output both stay resident). Bit-exact to
/// `find_best_splits_batched_fused_f64_from_handle_on` + `argmax_over_resident_splits`.
///
/// `real_feats` must have exactly `feats.len()` elements (one real feature index per
/// feature position). Empty `feats` → `Ok(())` with NO launch (the caller's pre-zeroed
/// frontier slot already reads as no-valid-split).
///
/// # Errors
/// As [`find_best_splits_batched_fused_f64_from_handle_on`] (length / scope / deferred-branch
/// typed errors); plus [`ComputeError::LengthMismatch`] if `real_feats.len() != feats.len()`
/// or `out_leaf >= out.len`.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_reduce_into_leaf_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf: usize,
) -> Result<(), ComputeError> {
    if real_feats.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: real_feats.len(),
        });
    }
    if !feats.is_empty() && out_leaf >= out.len {
        return Err(ComputeError::Runtime {
            detail: format!(
                "find_best_splits_reduce: out_leaf {out_leaf} out of range [0, {})",
                out.len
            ),
        });
    }
    let scanned = fused_scan_to_raw_handle(
        client,
        hist_handle,
        buf_len,
        feats,
        cfg,
        sum_gradient,
        sum_hessian,
        num_data,
    )?;
    if let Some((h_out, out_len, min_gain_shift)) = scanned {
        launch_reduce_into_leaf(
            client,
            h_out,
            out_len,
            real_feats,
            feats.len(),
            out,
            out_leaf,
            0,
            min_gain_shift,
        );
    }
    Ok(())
}

/// Shared V5 validation + per-sibling `min_gain_shift` pre-step + co-packed sibling scan
/// launch for the no-readback co-pack reduce launcher. Byte-for-byte the SAME
/// validation + the SAME per-sibling `2*kEpsilon` bump + the SAME
/// [`find_best_splits_fused_siblings_kernel`] launch as
/// [`find_best_splits_fused_siblings_from_handles_on`], but RETURNS the raw `2*n*12`
/// `h_out` Handle + `(min_gain_shift_a, min_gain_shift_b)` INSTEAD of reading it back. Uses
/// the direct `scan_cube_dim()` launch (bit-neutral `W`; no autotune) so the reduce launcher
/// is self-contained and the shipped `find_best_splits_fused_siblings_from_handles_on` stays
/// byte-unchanged (additive only). Empty `feats` → `Ok(None)`.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn fused_scan_siblings_to_raw_handle<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_a_handle: cubecl::server::Handle,
    hist_b_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    a_totals: (f64, f64, i32),
    b_totals: (f64, f64, i32),
) -> Result<Option<(cubecl::server::Handle, usize, usize, f64, f64)>, ComputeError> {
    if feats.is_empty() {
        return Ok(None);
    }
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings_reduce: max_delta_step / path_smooth are Phase-7+ \
                     scope (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    let (sum_gradient_a, sum_hessian_a, num_data_a) = a_totals;
    let (sum_gradient_b, sum_hessian_b, num_data_b) = b_totals;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_a > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings_reduce: smaller-sibling sum_hessian must be > 0"
                .to_string(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_b > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings_reduce: larger-sibling sum_hessian must be > 0"
                .to_string(),
        });
    }

    // Per-feature V5 validation + device-array assembly ONCE (SHARED between siblings) —
    // IDENTICAL to `find_best_splits_fused_siblings_from_handles_on`, including the
    // whole-batch `na_as_missing` reject.
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    for f in feats {
        if f.na_as_missing {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                         implemented"
                    .to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "find_best_split: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "find_best_splits_siblings_reduce: slot_off + region overflows".to_string(),
            })?;
        if end > buf_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: buf_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
    }

    // Per-sibling leaf scalars (the 2*kEpsilon bump + min_gain_shift) — IDENTICAL math.
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let use_l1 = cfg.use_l1();
    let sum_hessian_a_bumped = sum_hessian_a + two_eps;
    let min_gain_shift_a = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_a,
        sum_hessian_a_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    ) + cfg.min_gain_to_split;
    let sum_hessian_b_bumped = sum_hessian_b + two_eps;
    let min_gain_shift_b = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_b,
        sum_hessian_b_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    ) + cfg.min_gain_to_split;

    let out_len = 2 * n * 12;
    // NO zero-fill upload: the scan kernel (legacy or staged) writes all 12 cells
    // of every window unconditionally (single-slot helper note).
    let h_out = client.empty(out_len * std::mem::size_of::<f64>());
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));

    // STAGED cube-per-(feature, sibling) branch (opt-in `LGBM_SCAN_STAGED=1`) on
    // the LIVE per-split co-pack route — the hot path (one call per split). Same
    // gates as the single-slot helper; bit-identical output layout + cells.
    #[cfg(feature = "gpu")]
    if scan_staged_enabled()
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && num_bin_a.iter().all(|&nb| (nb as usize) * 2 <= SCAN_STAGE_MAX_CELLS)
    {
        // SAFETY: identical obligations to the legacy launch below; the geometry
        // guarantees CUBE_POS_X < n and CUBE_POS_Y < 2 (kernel no-guard contract).
        // The helper picks the serial-branch staged kernel or its bit-identical
        // PARGAIN twin (`LGBM_SCAN_PARGAIN`).
        unsafe {
            launch_staged_siblings_scan(
                client,
                n,
                hist_a_handle,
                hist_b_handle,
                buf_len,
                h_out.clone(),
                out_len,
                h_slot,
                h_numbin,
                h_offset,
                h_defbin,
                h_skip,
                h_rev,
                h_fwd,
                use_l1,
                cfg,
                (min_gain_shift_a, sum_gradient_a, sum_hessian_a_bumped, num_data_a),
                (min_gain_shift_b, sum_gradient_b, sum_hessian_b_bumped, num_data_b),
            );
        }
        return Ok(Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)));
    }

    // Direct `scan_cube_dim()` launch over 2*n feature-slots (A then B) — the SAME
    // `find_best_splits_fused_siblings_kernel` the host-readback path launches; only the
    // readback is dropped.
    let scan_w = scan_cube_dim();
    let cube_count = (2 * n as u32).div_ceil(scan_w);
    // SAFETY: both histogram handles describe `buf_len` f64 cells; every per-feature region
    // `[slot_off, slot_off+2*num_bin)` is validated `<= buf_len` above; all per-feature index
    // arrays have exactly `n` elements; lane `g` (guarded `< 2*n_feats`) reads only its
    // sibling's validated region and writes only `out[g*12 .. g*12+12]` within the `2*n*12`
    // allocation. All cubecl unsafe confined here (CMP-01).
    unsafe {
        find_best_splits_fused_siblings_kernel::launch(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(scan_w),
            ArrayArg::from_raw_parts(hist_a_handle, buf_len),
            ArrayArg::from_raw_parts(hist_b_handle, buf_len),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            ArrayArg::from_raw_parts(h_slot, n),
            ArrayArg::from_raw_parts(h_numbin, n),
            ArrayArg::from_raw_parts(h_offset, n),
            ArrayArg::from_raw_parts(h_defbin, n),
            ArrayArg::from_raw_parts(h_skip, n),
            ArrayArg::from_raw_parts(h_rev, n),
            ArrayArg::from_raw_parts(h_fwd, n),
            if use_l1 { 1u32 } else { 0u32 },
            cfg.min_data_in_leaf,
            cfg.min_sum_hessian_in_leaf,
            cfg.lambda_l1,
            cfg.lambda_l2,
            min_gain_shift_a,
            sum_gradient_a,
            sum_hessian_a_bumped,
            num_data_a,
            min_gain_shift_b,
            sum_gradient_b,
            sum_hessian_b_bumped,
            num_data_b,
            n as u32,
        );
    }
    Ok(Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)))
}

/// CO-PACK (2-sibling) no-readback reduce-into-leaves launcher. Mirrors
/// [`find_best_splits_fused_siblings_from_handles_on`]'s signature but writes BOTH siblings'
/// winners DIRECTLY into two target [`SplitSoa`] slots (`out_leaf_a` / `out_leaf_b`) via TWO
/// invocations of the shared [`reduce_scan_output_into_leaf_kernel`] on the SAME
/// `2*n*12` `h_out` (raw_base `0` for sibling A, `n*12` for sibling B), issuing ZERO
/// device→host transfer of the per-feature array. Each half is bit-exact to
/// `argmax_over_resident_splits` over that sibling's `Vec<SplitInfo>`.
///
/// `real_feats` (the feature-position → real feature index tie-break key, SHARED by both
/// siblings since they have the same feature layout) must have exactly `feats.len()` elements.
/// Empty `feats` → `Ok(())` with NO launch.
///
/// # Errors
/// As [`find_best_splits_fused_siblings_from_handles_on`] (length / scope / deferred-branch
/// typed errors, `!(sum_hessian > 0.0)` per sibling); plus [`ComputeError::LengthMismatch`] if
/// `real_feats.len() != feats.len()`, or [`ComputeError::Runtime`] if either `out_leaf` is out
/// of range.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_reduce_into_leaves_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_a_handle: cubecl::server::Handle,
    hist_b_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    a_totals: (f64, f64, i32),
    b_totals: (f64, f64, i32),
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf_a: usize,
    out_leaf_b: usize,
) -> Result<(), ComputeError> {
    if real_feats.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: real_feats.len(),
        });
    }
    if !feats.is_empty() && (out_leaf_a >= out.len || out_leaf_b >= out.len) {
        return Err(ComputeError::Runtime {
            detail: format!(
                "find_best_splits_siblings_reduce: out_leaf_a {out_leaf_a} / out_leaf_b \
                 {out_leaf_b} out of range [0, {})",
                out.len
            ),
        });
    }
    let scanned = fused_scan_siblings_to_raw_handle(
        client,
        hist_a_handle,
        hist_b_handle,
        buf_len,
        feats,
        cfg,
        a_totals,
        b_totals,
    )?;
    if let Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)) = scanned {
        // Sibling A: features [0, n), raw_base 0.
        launch_reduce_into_leaf(
            client,
            h_out.clone(),
            out_len,
            real_feats,
            n,
            out,
            out_leaf_a,
            0,
            min_gain_shift_a,
        );
        // Sibling B: features [n, 2n), raw_base n*12.
        launch_reduce_into_leaf(
            client,
            h_out,
            out_len,
            real_feats,
            n,
            out,
            out_leaf_b,
            n * 12,
            min_gain_shift_b,
        );
    }
    Ok(())
}

/// **Native** host f64 best-split scan — the production cpu-anchor path (R2).
///
/// Bit-IDENTICAL to [`find_best_split_cpu`] (the single-unit
/// `find_best_split_kernel`): the SAME host pre-step (V5 validation, the
/// `2*kEpsilon` entry bump, `min_gain_shift`), the SAME REVERSE+FORWARD scan with
/// the SAME gate ORDER / eps placements / operand orders, and the SAME decode +
/// accept-gate — but run as plain Rust instead of a `CubeDim::new_1d(1)` cubecl
/// launch, dropping the fixed ~20–50µs/call dispatch cost (this op runs once per
/// feature per leaf, the other half of the per-(feature,leaf) launch overhead
/// after `construct_histograms`).
///
/// The `select(c, a, b)` branchless encoding the kernel needs for cubecl-cpu's
/// MLIR lowering becomes plain `if` here; the arithmetic is unchanged
/// (`sum += select(active, x, 0.0)` ≡ `if active { sum += x }` since `+ 0.0` is a
/// no-op, and the gain primitives are pure). `find_best_split_cpu` is retained for
/// the kernel-parity / ROCm-mirror tests; the f32 hip path is untouched.
///
/// # Errors
/// Same as [`find_best_split_cpu`] (V5 validation, deferred `na_as_missing`,
/// non-default smoothing/clamp).
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_cpu_native(
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: u32,
    offset: i32,
    default_bin: u32,
    _most_freq_bin: u32,
    skip_default_bin: bool,
    na_as_missing: bool,
    run_forward: bool,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<SplitInfo, ComputeError> {
    use crate::gain::{calculate_splitted_leaf_output as calc_out, get_leaf_gain, get_split_gains};

    // ---- V5 boundary validation (identical to find_best_split_cpu) ----
    if na_as_missing {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                     implemented"
                .to_string(),
        });
    }
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: num_bin must be > 0".to_string(),
        });
    }
    let expected = 2usize
        .checked_mul(num_bin as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("num_bin {num_bin} overflows the histogram length"),
        })?;
    if hist.len() != expected {
        return Err(ComputeError::LengthMismatch {
            expected,
            actual: hist.len(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }

    // ---- host pre-step: 2*kEpsilon bump + min_gain_shift (verbatim) ----
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = get_leaf_gain(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let num_bin_i = num_bin as i32;
    let rev_count = (num_bin_i - 1).max(0);
    let fwd_count = if run_forward {
        (num_bin_i - 1 - offset).max(0)
    } else {
        0
    };

    // `Common::RoundInt(x) = (int)(x + 0.5f)` — the 0.5f widens to f64 exactly, the
    // cast truncates toward zero (matches `i32::cast_from` in the kernel).
    let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };

    let l1 = cfg.lambda_l1;
    let l2 = cfg.lambda_l2;
    let cnt_factor = f64::from(num_data) / sum_hessian_bumped;
    let min_sum_hessian_in_leaf = cfg.min_sum_hessian_in_leaf;
    let min_data_in_leaf = cfg.min_data_in_leaf;

    // Best-split running state — same literal inits as the kernel.
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut best_gain = 0.0f64;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = false;
    let mut best_default_left = true; // REVERSE default

    // ===================== REVERSE branch (:854-936) =====================
    {
        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = f64::from(K_EPSILON);
        let mut right_count = 0i32;
        let t_start = num_bin_i - 1 - offset;
        let mut done = false;
        for k in 0..rev_count {
            let t = t_start - k;
            let in_range = t >= (1 - offset);
            let skip = skip_default_bin && (t + offset) == default_bin as i32;
            let active = in_range && !skip && !done;
            if active {
                // `active` implies `in_range` (t >= 1-offset >= 0 for offset∈{0,1}).
                let bi = (t as usize) * 2;
                sum_right_gradient += hist[bi];
                sum_right_hessian += hist[bi + 1];
                right_count += round_int(hist[bi + 1] * cnt_factor);
            }
            // Gates computed EVERY iteration (the running sums are unchanged when
            // inactive); `done` is sticky and gates later iterations.
            let left_count = num_data - right_count;
            let sum_left_hessian = sum_hessian_bumped - sum_right_hessian;
            let sum_left_gradient = sum_gradient - sum_right_gradient;
            let cont =
                right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
            let brk = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
            done = done || (active && !cont && brk);
            let consider = active && !cont && !done;
            if consider {
                let current_gain = get_split_gains(
                    use_l1,
                    sum_left_gradient,
                    sum_left_hessian,
                    sum_right_gradient,
                    sum_right_hessian,
                    l1,
                    l2,
                );
                if current_gain > min_gain_shift {
                    is_splittable = true;
                    if current_gain > best_gain {
                        best_left_count = left_count;
                        best_sum_left_gradient = sum_left_gradient;
                        best_sum_left_hessian = sum_left_hessian;
                        best_threshold = t - 1 + offset; // left<=thr, right>thr (:933)
                        best_gain = current_gain;
                        best_default_left = true;
                    }
                }
            }
        }
    }

    // ===================== FORWARD branch (:937-1029) ====================
    {
        let mut sum_left_gradient = 0.0f64;
        let mut sum_left_hessian = f64::from(K_EPSILON);
        let mut left_count = 0i32;
        let mut done = false;
        for t in 0..fwd_count {
            let skip = skip_default_bin && (t + offset) == default_bin as i32;
            let active = !skip && !done;
            if active {
                let bi = (t as usize) * 2;
                sum_left_gradient += hist[bi];
                sum_left_hessian += hist[bi + 1];
                left_count += round_int(hist[bi + 1] * cnt_factor);
            }
            let right_count = num_data - left_count;
            let sum_right_hessian = sum_hessian_bumped - sum_left_hessian;
            let sum_right_gradient = sum_gradient - sum_left_gradient;
            let cont =
                left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
            let brk = right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
            done = done || (active && !cont && brk);
            let consider = active && !cont && !done;
            if consider {
                let current_gain = get_split_gains(
                    use_l1,
                    sum_left_gradient,
                    sum_left_hessian,
                    sum_right_gradient,
                    sum_right_hessian,
                    l1,
                    l2,
                );
                if current_gain > min_gain_shift {
                    is_splittable = true;
                    if current_gain > best_gain {
                        best_left_count = left_count;
                        best_sum_left_gradient = sum_left_gradient;
                        best_sum_left_hessian = sum_left_hessian;
                        best_threshold = t + offset; // forward records t+offset (:1025)
                        best_gain = current_gain;
                        best_default_left = false;
                    }
                }
            }
        }
    }

    // ---- finalization (feature_histogram.hpp:1031-1056), same operand orders ----
    let eps = f64::from(K_EPSILON);
    let left_output = calc_out(use_l1, best_sum_left_gradient, best_sum_left_hessian, l1, l2);
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian_bumped - best_sum_left_hessian;
    let right_output = calc_out(use_l1, right_sum_gradient, right_sum_hessian, l1, l2);

    // Accept gate (:1031): is_splittable && best_gain > -inf (finite). Reported
    // gain is best_gain - min_gain_shift (penalty == 1).
    if is_splittable && best_gain > f64::NEG_INFINITY {
        Ok(SplitInfo {
            threshold: best_threshold as u32,
            gain: best_gain - min_gain_shift,
            left_count: best_left_count,
            right_count: num_data - best_left_count,
            left_sum_gradient: best_sum_left_gradient,
            left_sum_hessian: best_sum_left_hessian - eps, // (:1042)
            right_sum_gradient,
            right_sum_hessian: right_sum_hessian - eps, // (:1053)
            left_output,
            right_output,
            default_left: best_default_left,
        })
    } else {
        Ok(SplitInfo::none())
    }
}

/// One pass's winning candidate — the cross-lane combine unit for the additive
/// 2-lane native scan ([`find_best_split_cpu_native_2lane`]). Carries EXACTLY the
/// loop-carried `best_*` state the serial [`find_best_split_cpu_native`] threads
/// through its REVERSE-then-FORWARD scan, so a pass run in isolation produces the
/// same winner it would as one half of the serial scan.
#[derive(Debug, Clone, Copy)]
struct PassWinner {
    is_splittable: bool,
    best_gain: f64,
    best_left_count: i32,
    best_sum_left_gradient: f64,
    best_sum_left_hessian: f64,
    best_threshold: i32,
    best_default_left: bool,
}

impl PassWinner {
    /// The "no candidate" seed — IDENTICAL to the serial scan's literal inits
    /// (`best_gain = 0.0`, `is_splittable = false`). `best_default_left` defaults to
    /// the pass's own branch convention (REVERSE => true, FORWARD => false) and is
    /// only consulted when this pass actually wins.
    #[inline]
    fn seed(default_left: bool) -> Self {
        PassWinner {
            is_splittable: false,
            best_gain: 0.0,
            best_left_count: 0,
            best_sum_left_gradient: 0.0,
            best_sum_left_hessian: 0.0,
            best_threshold: 0,
            best_default_left: default_left,
        }
    }
}

/// REVERSE pass run in ISOLATION — byte-for-byte the REVERSE branch of
/// [`find_best_split_cpu_native`] (same loop bound, same gate ORDER, same operand
/// orders, same `t-1+offset` threshold, same `best_gain = 0.0` literal seed, same
/// strict-`>` keep-first-on-tie winner update). Returns its standalone winner.
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan_reverse_pass(
    hist: &[f64],
    use_l1: bool,
    l1: f64,
    l2: f64,
    num_bin_i: i32,
    offset: i32,
    default_bin: u32,
    skip_default_bin: bool,
    rev_count: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian_bumped: f64,
    cnt_factor: f64,
    num_data: i32,
    round_int: impl Fn(f64) -> i32,
) -> PassWinner {
    use crate::gain::get_split_gains;
    let mut w = PassWinner::seed(true); // REVERSE default_left = true
    let mut sum_right_gradient = 0.0f64;
    let mut sum_right_hessian = f64::from(K_EPSILON);
    let mut right_count = 0i32;
    let t_start = num_bin_i - 1 - offset;
    let mut done = false;
    for k in 0..rev_count {
        let t = t_start - k;
        let in_range = t >= (1 - offset);
        let skip = skip_default_bin && (t + offset) == default_bin as i32;
        let active = in_range && !skip && !done;
        if active {
            let bi = (t as usize) * 2;
            sum_right_gradient += hist[bi];
            sum_right_hessian += hist[bi + 1];
            right_count += round_int(hist[bi + 1] * cnt_factor);
        }
        let left_count = num_data - right_count;
        let sum_left_hessian = sum_hessian_bumped - sum_right_hessian;
        let sum_left_gradient = sum_gradient - sum_right_gradient;
        let cont = right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        let brk = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;
        if consider {
            let current_gain = get_split_gains(
                use_l1,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
            );
            if current_gain > min_gain_shift {
                w.is_splittable = true;
                if current_gain > w.best_gain {
                    w.best_left_count = left_count;
                    w.best_sum_left_gradient = sum_left_gradient;
                    w.best_sum_left_hessian = sum_left_hessian;
                    w.best_threshold = t - 1 + offset;
                    w.best_gain = current_gain;
                    w.best_default_left = true;
                }
            }
        }
    }
    w
}

/// FORWARD pass run in ISOLATION — byte-for-byte the FORWARD branch of
/// [`find_best_split_cpu_native`], EXCEPT its winner is seeded at `best_gain = 0.0`
/// (the serial scan instead enters FORWARD carrying the REVERSE winner's gain). The
/// cross-lane combine in [`find_best_split_cpu_native_2lane`] restores the serial
/// semantics exactly: the serial FORWARD only displaces the REVERSE winner when a
/// forward candidate's gain STRICTLY exceeds it, and the first forward candidate to
/// attain the forward-pass maximum is the same winner whether the running best was
/// seeded at `0.0` or at `rev_gain < fwd_max` — so `(fwd.gain > rev.gain) ? fwd :
/// rev` reproduces the serial winner (and reverse-wins-ties) bit-for-bit.
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan_forward_pass(
    hist: &[f64],
    use_l1: bool,
    l1: f64,
    l2: f64,
    offset: i32,
    default_bin: u32,
    skip_default_bin: bool,
    fwd_count: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian_bumped: f64,
    cnt_factor: f64,
    num_data: i32,
    round_int: impl Fn(f64) -> i32,
) -> PassWinner {
    use crate::gain::get_split_gains;
    let mut w = PassWinner::seed(false); // FORWARD default_left = false
    let mut sum_left_gradient = 0.0f64;
    let mut sum_left_hessian = f64::from(K_EPSILON);
    let mut left_count = 0i32;
    let mut done = false;
    for t in 0..fwd_count {
        let skip = skip_default_bin && (t + offset) == default_bin as i32;
        let active = !skip && !done;
        if active {
            let bi = (t as usize) * 2;
            sum_left_gradient += hist[bi];
            sum_left_hessian += hist[bi + 1];
            left_count += round_int(hist[bi + 1] * cnt_factor);
        }
        let right_count = num_data - left_count;
        let sum_right_hessian = sum_hessian_bumped - sum_left_hessian;
        let sum_right_gradient = sum_gradient - sum_left_gradient;
        let cont = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        let brk = right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;
        if consider {
            let current_gain = get_split_gains(
                use_l1,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
            );
            if current_gain > min_gain_shift {
                w.is_splittable = true;
                if current_gain > w.best_gain {
                    w.best_left_count = left_count;
                    w.best_sum_left_gradient = sum_left_gradient;
                    w.best_sum_left_hessian = sum_left_hessian;
                    w.best_threshold = t + offset;
                    w.best_gain = current_gain;
                    w.best_default_left = false;
                }
            }
        }
    }
    w
}

/// **ADDITIVE** 2-lane native f64 best-split scan — a
/// bit-identical alternative to [`find_best_split_cpu_native`] that runs the
/// REVERSE and FORWARD passes on two `rayon::join` lanes and combines their
/// standalone winners with a final argmax. The serial
/// [`find_best_split_cpu_native`] remains the SINGLE bit-exact source of truth; this
/// is gated behind a measured-win flag (see the `find_best_splits_batched` trait
/// default's `LGBM_SPLIT_2LANE` dispatch) and never replaces it.
///
/// ## Bit-exactness contract
/// Each lane's pass is byte-for-byte its serial counterpart (no intra-pass
/// reordering — the f64 loop-carried accumulation order inside each pass is
/// untouched). The ONLY change is that the two passes no longer share a running
/// `best_gain`; the cross-lane combine `(fwd.gain > rev.gain) ? fwd : rev` restores
/// the serial cross-pass semantics exactly:
/// - REVERSE-wins-ties: the serial FORWARD uses strict `>` against the carried
///   REVERSE gain, so an equal-gain forward candidate never displaces REVERSE. The
///   combine's strict `>` reproduces this.
/// - Winner identity: when `fwd_max > rev_gain` the serial FORWARD records the FIRST
///   forward candidate attaining `fwd_max` (each `>` update keeps the first to reach
///   a new max); the isolated FORWARD (seeded at `0.0`) records the same first
///   attainer. When `fwd_max <= rev_gain` the serial keeps REVERSE; the combine does
///   too.
/// - `is_splittable` is the OR of the two passes (the serial flag is set in either).
///
/// All other host steps (V5 validation, `2*kEpsilon` bump, `min_gain_shift`,
/// `cnt_factor`, finalization eps subtraction, accept gate) are IDENTICAL to the
/// serial function — only the scan body is split across two lanes.
///
/// # Errors
/// Same as [`find_best_split_cpu_native`].
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_cpu_native_2lane(
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: u32,
    offset: i32,
    default_bin: u32,
    _most_freq_bin: u32,
    skip_default_bin: bool,
    na_as_missing: bool,
    run_forward: bool,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<SplitInfo, ComputeError> {
    use crate::gain::{calculate_splitted_leaf_output as calc_out, get_leaf_gain};

    // ---- V5 boundary validation (identical to find_best_split_cpu_native) ----
    if na_as_missing {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: na_as_missing (NA_AS_MISSING forward branch) not yet \
                     implemented"
                .to_string(),
        });
    }
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: num_bin must be > 0".to_string(),
        });
    }
    let expected = 2usize
        .checked_mul(num_bin as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("num_bin {num_bin} overflows the histogram length"),
        })?;
    if hist.len() != expected {
        return Err(ComputeError::LengthMismatch {
            expected,
            actual: hist.len(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }

    // ---- host pre-step: 2*kEpsilon bump + min_gain_shift (verbatim) ----
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = get_leaf_gain(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let num_bin_i = num_bin as i32;
    let rev_count = (num_bin_i - 1).max(0);
    let fwd_count = if run_forward {
        (num_bin_i - 1 - offset).max(0)
    } else {
        0
    };

    let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };
    let l1 = cfg.lambda_l1;
    let l2 = cfg.lambda_l2;
    let cnt_factor = f64::from(num_data) / sum_hessian_bumped;
    let min_sum_hessian_in_leaf = cfg.min_sum_hessian_in_leaf;
    let min_data_in_leaf = cfg.min_data_in_leaf;

    // ---- the two passes on two lanes (rayon::join). Each pass is independent
    // (its own running sums); the closures borrow `hist` + the leaf scalars
    // immutably, so there is no shared mutable state and the parallelism cannot
    // perturb either pass's f64 accumulation order. ----
    let (rev, fwd) = rayon::join(
        || {
            scan_reverse_pass(
                hist,
                use_l1,
                l1,
                l2,
                num_bin_i,
                offset,
                default_bin,
                skip_default_bin,
                rev_count,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                cnt_factor,
                num_data,
                &round_int,
            )
        },
        || {
            scan_forward_pass(
                hist,
                use_l1,
                l1,
                l2,
                offset,
                default_bin,
                skip_default_bin,
                fwd_count,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                cnt_factor,
                num_data,
                &round_int,
            )
        },
    );

    // ---- cross-lane argmax (reverse-wins-ties: strict `>` favours the pass that
    // ran FIRST in the serial scan, i.e. REVERSE). Bit-identical to the serial
    // winner per the contract above. ----
    let win = if fwd.best_gain > rev.best_gain { fwd } else { rev };
    let is_splittable = rev.is_splittable || fwd.is_splittable;
    let best_sum_left_gradient = win.best_sum_left_gradient;
    let best_sum_left_hessian = win.best_sum_left_hessian;
    let best_gain = win.best_gain;
    let best_left_count = win.best_left_count;
    let best_threshold = win.best_threshold;
    let best_default_left = win.best_default_left;

    // ---- finalization (IDENTICAL operand orders to find_best_split_cpu_native) ----
    let eps = f64::from(K_EPSILON);
    let left_output = calc_out(use_l1, best_sum_left_gradient, best_sum_left_hessian, l1, l2);
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian_bumped - best_sum_left_hessian;
    let right_output = calc_out(use_l1, right_sum_gradient, right_sum_hessian, l1, l2);

    if is_splittable && best_gain > f64::NEG_INFINITY {
        Ok(SplitInfo {
            threshold: best_threshold as u32,
            gain: best_gain - min_gain_shift,
            left_count: best_left_count,
            right_count: num_data - best_left_count,
            left_sum_gradient: best_sum_left_gradient,
            left_sum_hessian: best_sum_left_hessian - eps,
            right_sum_gradient,
            right_sum_hessian: right_sum_hessian - eps,
            left_output,
            right_output,
            default_left: best_default_left,
        })
    } else {
        Ok(SplitInfo::none())
    }
}


// A `cfg_skip_default_bin(default_bin, num_bin)` heuristic (`default_bin < num_bin`)
// is deliberately NOT used here. The authoritative `SKIP_DEFAULT_BIN`/`NA_AS_MISSING`
// flags (`feature_histogram.hpp:284-285`: `num_bin > 2 && missing_type == Zero` for
// skip, `num_bin > 2 && missing_type == NaN` for na_as_missing, both false for
// `missing_type == None`) are derived in the learner from `bin_mapper.missing_type()`
// and threaded through `Backend::find_best_split` / `find_best_split_cpu` /
// `find_best_split_raw_f32_on` as explicit params, so the kernel dispatch matches
// C++ exactly instead of approximating it from the bin layout.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    /// Minimal smoke test: a clean separable two-region histogram with relaxed
    /// gates yields a finite-gain split. Verifies the kernel launches, the scan
    /// runs, and the launcher decodes a `SplitInfo`. (Bit-exact C++ parity is the
    /// job of `oracle-harness/tests/kernel_parity.rs` over the committed golden.)
    #[test]
    fn find_best_split_smoke_finds_a_split() {
        let client = cpu_client();
        // num_bin = 4; stride-2 [g,h] per bin. Put gradient mass that clearly
        // separates low bins (negative grad) from high bins (positive grad).
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![
            -10.0, 5.0, // bin 0
            -8.0, 5.0, // bin 1
            9.0, 5.0, // bin 2
            8.0, 5.0, // bin 3
        ];
        let sum_gradient: f64 = -10.0 - 8.0 + 9.0 + 8.0;
        let sum_hessian: f64 = 20.0;
        let num_data = 20i32; // 5 per bin (hess*cnt_factor = 5 * 20/20 = 5)

        // Relaxed gates so a split is admissible.
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        // offset=0, default_bin out of range so SKIP_DEFAULT_BIN never fires.
        let si = find_best_split_cpu(
            &client,
            &hist,
            &cfg,
            num_bin,
            0,
            num_bin, // default_bin == num_bin -> not skipped
            0,
            false, // skip_default_bin
            false, // na_as_missing
            true,  // run_forward (exercise both scan branches in the smoke launch)
            sum_gradient,
            sum_hessian,
            num_data,
        )
        .expect("split should succeed");

        assert!(si.gain.is_finite(), "expected a finite-gain split, got {si:?}");
        assert!(si.left_count > 0 && si.right_count > 0);
        assert_eq!(si.left_count + si.right_count, num_data);
    }

    /// Tight gates make no split admissible -> `gain == -inf` (kMinScore).
    #[test]
    fn find_best_split_no_admissible_split_returns_none() {
        let client = cpu_client();
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = GainConfig {
            min_data_in_leaf: 1_000_000, // impossible to satisfy
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let si = find_best_split_cpu(
            &client, &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, 4.0, 4.0, 8,
        )
        .expect("call ok");
        assert_eq!(si.gain, f64::NEG_INFINITY, "no split -> kMinScore");
    }

    #[test]
    fn find_best_split_rejects_bad_length() {
        let client = cpu_client();
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        // hist len 6 != 2*num_bin(4)=8
        let err = find_best_split_cpu(
            &client,
            &[0.0; 6],
            &cfg,
            4,
            0,
            4,
            0,
            false,
            false,
            true,
            1.0,
            1.0,
            4,
        )
        .unwrap_err();
        assert!(matches!(err, ComputeError::LengthMismatch { .. }));
    }

    /// `na_as_missing == true` is a TYPED error (the NA_AS_MISSING forward branch
    /// is deferred) — never a panic, never a wrong SplitInfo (T-05-01-01). This is
    /// asserted BEFORE any length/num_bin validation so the deferral is
    /// unambiguous even on otherwise-valid input.
    #[test]
    fn find_best_split_na_as_missing_is_typed_error() {
        let client = cpu_client();
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0];
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let err = find_best_split_cpu(
            &client,
            &hist,
            &cfg,
            num_bin,
            0,
            num_bin,
            0,
            false, // skip_default_bin
            true,  // na_as_missing -> deferred typed error
            false, // run_forward (unreached: na_as_missing errors first)
            -1.0,
            20.0,
            20,
        )
        .expect_err("na_as_missing must be a typed error, not a wrong split");
        match err {
            ComputeError::Runtime { detail } => {
                assert!(
                    detail.contains("na_as_missing"),
                    "error must name na_as_missing, got: {detail}"
                );
            }
            other => panic!("expected ComputeError::Runtime, got {other:?}"),
        }
    }

    // ===================================================================
    // The additive 2-lane native scan (`find_best_split_cpu_native_2lane`) MUST be
    // byte-for-byte identical to the serial source of truth
    // (`find_best_split_cpu_native`) across the full parameter matrix. Parity is
    // the non-negotiable gate; the 2-lane variant is only a latency lever, never a
    // numeric change.
    // ===================================================================

    /// Bit-compare two `SplitInfo`s: every f64 field by its raw bit pattern (so a
    /// signed-zero or any ULP difference is caught), every integer/flag by `==`.
    fn split_info_bit_eq(a: &SplitInfo, b: &SplitInfo) -> bool {
        a.threshold == b.threshold
            && a.left_count == b.left_count
            && a.right_count == b.right_count
            && a.default_left == b.default_left
            && a.gain.to_bits() == b.gain.to_bits()
            && a.left_sum_gradient.to_bits() == b.left_sum_gradient.to_bits()
            && a.left_sum_hessian.to_bits() == b.left_sum_hessian.to_bits()
            && a.right_sum_gradient.to_bits() == b.right_sum_gradient.to_bits()
            && a.right_sum_hessian.to_bits() == b.right_sum_hessian.to_bits()
            && a.left_output.to_bits() == b.left_output.to_bits()
            && a.right_output.to_bits() == b.right_output.to_bits()
    }

    /// Assert serial == 2-lane for ONE parameter set, returning both for context.
    #[allow(clippy::too_many_arguments)]
    fn assert_2lane_eq(
        hist: &[f64],
        cfg: &GainConfig,
        num_bin: u32,
        offset: i32,
        default_bin: u32,
        most_freq_bin: u32,
        skip_default_bin: bool,
        run_forward: bool,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) {
        let serial = find_best_split_cpu_native(
            hist, cfg, num_bin, offset, default_bin, most_freq_bin, skip_default_bin, false,
            run_forward, sum_gradient, sum_hessian, num_data,
        )
        .expect("serial ok");
        let twolane = find_best_split_cpu_native_2lane(
            hist, cfg, num_bin, offset, default_bin, most_freq_bin, skip_default_bin, false,
            run_forward, sum_gradient, sum_hessian, num_data,
        )
        .expect("2lane ok");
        assert!(
            split_info_bit_eq(&serial, &twolane),
            "2-lane diverged from serial.\n serial = {serial:?}\n 2lane  = {twolane:?}\n \
             (num_bin={num_bin} offset={offset} skip_default_bin={skip_default_bin} \
             run_forward={run_forward})"
        );
    }

    /// Sweep a representative parameter matrix: separable + flat + noisy
    /// histograms × offset ∈ {0,1} × skip_default_bin × run_forward × L1 on/off ×
    /// a range of min_data_in_leaf gates (some admit a split, some don't). Every
    /// cell must be bit-identical.
    #[test]
    fn split_2lane_equals_serial_matrix() {
        // A few hand-built histograms of varying widths (stride-2 [g,h] per bin).
        let make = |bins: &[(f64, f64)]| -> Vec<f64> {
            let mut v = Vec::with_capacity(bins.len() * 2);
            for &(g, h) in bins {
                v.push(g);
                v.push(h);
            }
            v
        };
        // separable: low bins negative grad, high bins positive grad.
        let separable = make(&[
            (-10.0, 5.0),
            (-8.0, 5.0),
            (-3.0, 5.0),
            (4.0, 5.0),
            (9.0, 5.0),
            (8.0, 5.0),
        ]);
        // flat: no informative split.
        let flat = make(&[(1.0, 2.0); 6]);
        // noisy: irregular masses (exercises a non-trivial argmax / tie landscape).
        let noisy = make(&[
            (-2.0, 3.0),
            (5.0, 1.0),
            (-7.0, 4.0),
            (3.0, 2.0),
            (-1.0, 6.0),
            (6.0, 2.0),
        ]);
        let histos: [&[f64]; 3] = [&separable, &flat, &noisy];

        for hist in histos {
            let num_bin = (hist.len() / 2) as u32;
            // sums = column totals (the leaf totals the scan divides through).
            let mut sg = 0.0f64;
            let mut sh = 0.0f64;
            for b in 0..num_bin as usize {
                sg += hist[b * 2];
                sh += hist[b * 2 + 1];
            }
            // Keep sum_hessian strictly positive (cnt_factor divides by it).
            if !(sh > 0.0) {
                continue;
            }
            let num_data = sh.round() as i32; // ~1 row per hess unit
            for &offset in &[0i32, 1] {
                for &skip in &[false, true] {
                    for &run_forward in &[false, true] {
                        for &l1 in &[0.0f64, 0.5] {
                            for &min_data in &[1i32, 2, 5, 100] {
                                let cfg = GainConfig {
                                    min_data_in_leaf: min_data,
                                    min_sum_hessian_in_leaf: 0.0,
                                    max_delta_step: 0.0,
                                    lambda_l1: l1,
                                    lambda_l2: 0.0,
                                    min_gain_to_split: 0.0,
                                    path_smooth: 0.0,
                                    ..Default::default()
                                };
                                // default_bin = 1 so skip_default_bin actually fires
                                // for an in-range bin when enabled.
                                assert_2lane_eq(
                                    hist, &cfg, num_bin, offset, 1, 0, skip, run_forward, sg, sh,
                                    num_data,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Targeted tie-break case: a histogram engineered so the best REVERSE
    /// candidate and the best FORWARD candidate have EQUAL gain. The serial scan
    /// keeps REVERSE (FORWARD's `>` is strict against the carried reverse gain);
    /// the 2-lane combine's strict `>` must reproduce that exactly (REVERSE wins,
    /// default_left == true).
    #[test]
    fn split_2lane_reverse_wins_exact_tie() {
        // Symmetric histogram: gradient mass is mirror-antisymmetric about the
        // centre so the single interior split (left|right at the midpoint) is the
        // unique best, and BOTH the reverse threshold (t-1+offset) and the forward
        // threshold (t+offset) that the two passes record land on the SAME midpoint
        // with the SAME gain — a genuine cross-pass tie.
        let hist: Vec<f64> = vec![
            -6.0, 4.0, // bin 0
            -6.0, 4.0, // bin 1
            6.0, 4.0, // bin 2
            6.0, 4.0, // bin 3
        ];
        let num_bin = 4u32;
        let sum_gradient = -6.0 - 6.0 + 6.0 + 6.0; // 0.0
        let sum_hessian = 16.0;
        let num_data = 16i32;
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let serial = find_best_split_cpu_native(
            &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient, sum_hessian,
            num_data,
        )
        .expect("serial ok");
        let twolane = find_best_split_cpu_native_2lane(
            &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient, sum_hessian,
            num_data,
        )
        .expect("2lane ok");
        assert!(
            split_info_bit_eq(&serial, &twolane),
            "tie-break diverged:\n serial={serial:?}\n 2lane={twolane:?}"
        );
        // The serial winner on this symmetric tie is the REVERSE candidate
        // (default_left == true); the 2-lane combine must agree.
        assert!(serial.gain.is_finite());
        assert!(
            serial.default_left,
            "reverse should win the exact tie (default_left == true), got {serial:?}"
        );
    }

    // ===================== split_reduce_into_leaf =====================

    /// A `BatchedSplitFeature` with only `na_as_missing` varying (the rest are inert for
    /// the reduce, which reads the raw scan cells not the feature layout).
    fn rfeat(na_as_missing: bool) -> BatchedSplitFeature {
        BatchedSplitFeature {
            slot_off: 0,
            num_bin: 4,
            offset: 0,
            default_bin: 4,
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing,
            run_forward: true,
        }
    }

    /// Hand-populate one feature's 12-cell raw window (matching the
    /// `find_best_splits_fused_kernel` decode layout) into `cells` at feature index `f`.
    #[allow(clippy::too_many_arguments)]
    fn put_raw(
        cells: &mut [f64],
        f: usize,
        is_splittable: bool,
        threshold: u32,
        raw_gain: f64,
        default_left: bool,
        lsg: f64,
        lsh: f64,
        rsg: f64,
        rsh: f64,
    ) {
        let b = f * 12;
        cells[b] = if is_splittable { 1.0 } else { 0.0 };
        cells[b + 1] = f64::from(threshold);
        cells[b + 2] = raw_gain;
        cells[b + 3] = 0.0; // left_count (not carried by SplitSoa)
        cells[b + 4] = 0.0; // right_count
        cells[b + 5] = lsg;
        cells[b + 6] = lsh;
        cells[b + 7] = rsg;
        cells[b + 8] = rsh;
        cells[b + 9] = if default_left { 1.0 } else { 0.0 };
        cells[b + 10] = 0.0; // left_output (not carried)
        cells[b + 11] = 0.0; // right_output
    }

    /// A `SplitInfo` with the fields the reduce carries (gain/threshold/default_left/4 sums).
    #[allow(clippy::too_many_arguments)]
    fn si(
        threshold: u32,
        gain: f64,
        default_left: bool,
        lsg: f64,
        lsh: f64,
        rsg: f64,
        rsh: f64,
    ) -> SplitInfo {
        SplitInfo {
            threshold,
            gain,
            left_count: 0,
            right_count: 0,
            left_sum_gradient: lsg,
            left_sum_hessian: lsh,
            right_sum_gradient: rsg,
            right_sum_hessian: rsh,
            left_output: 0.0,
            right_output: 0.0,
            default_left,
        }
    }

    /// Assert the device SoA slot's carried fields (gain/feat/thr/dleft/4 sums) equal the
    /// host argmax winner + expected real feature index, BIT-EXACT.
    fn assert_slot_eq(
        got: &crate::kernels::split_info::SplitScalars,
        want: &SplitInfo,
        want_feat: i32,
    ) {
        assert!(got.is_valid, "device slot must be a valid split");
        assert_eq!(got.gain.to_bits(), want.gain.to_bits(), "gain bit-exact");
        assert_eq!(got.inner_feature_index, want_feat, "winning real feature index");
        assert_eq!(got.threshold, want.threshold, "threshold");
        assert_eq!(got.default_left, want.default_left, "default_left");
        assert_eq!(
            got.left_sum_gradients.to_bits(),
            want.left_sum_gradient.to_bits(),
            "left_sum_gradients bit-exact"
        );
        assert_eq!(
            got.left_sum_hessians.to_bits(),
            want.left_sum_hessian.to_bits(),
            "left_sum_hessians bit-exact"
        );
        assert_eq!(
            got.right_sum_gradients.to_bits(),
            want.right_sum_gradient.to_bits(),
            "right_sum_gradients bit-exact"
        );
        assert_eq!(
            got.right_sum_hessians.to_bits(),
            want.right_sum_hessian.to_bits(),
            "right_sum_hessians bit-exact"
        );
        // The child leaf OUTPUTS carried device→device (raw cells [10]/[11]).
        // On the host-readback parity tests `want` is a REAL scan winner, so this proves the
        // reduce carries the SAME `left_output`/`right_output` the host decode reads.
        assert_eq!(
            got.left_value.to_bits(),
            want.left_output.to_bits(),
            "left_output bit-exact"
        );
        assert_eq!(
            got.right_value.to_bits(),
            want.right_output.to_bits(),
            "right_output bit-exact"
        );
    }

    /// The raw-cell-level engineered TIE fixture — mirrors
    /// `on_device_argmax_reduce_matches_host_argmax_bit_for_bit`'s Case B at the kernel-decode
    /// level: real feats `[7, 2, 5, 2]`, net gains `[8, 8, 4, 8]` (min_gain_shift = 0). The new
    /// reduce kernel must resolve the 3-way max-gain tie to fpos 1 (real index 2 — the LOWEST
    /// among the tied {7,2,2}, first fpos among equal reals), bit-exact on gain/threshold/
    /// default_left/all 4 sums, exactly as the host `argmax_over_resident_splits`.
    #[test]
    fn split_reduce_into_leaf_tie_fixture() {
        use crate::kernels::best_split::SplitSoa;
        use crate::kernels::grow_driver::argmax_over_resident_splits;

        let client = cpu_client();

        // Distinct threshold/sums per feature so the winner is unambiguous.
        let splits = vec![
            si(100, 8.0, true, 1.0, 2.0, 3.0, 4.0),   // fpos 0, real 7
            si(111, 8.0, false, 11.0, 12.0, 13.0, 14.0), // fpos 1, real 2 <- expected winner
            si(122, 4.0, true, 21.0, 22.0, 23.0, 24.0), // fpos 2, real 5
            si(133, 8.0, false, 31.0, 32.0, 33.0, 34.0), // fpos 3, real 2
        ];
        let real_feats = vec![7i32, 2, 5, 2];
        let feats = vec![rfeat(false), rfeat(false), rfeat(false), rfeat(false)];

        // Host ground truth.
        let (host_best, host_fpos) = argmax_over_resident_splits(&splits, &feats, &real_feats);
        assert_eq!(host_fpos, 1, "host tie-break must pick fpos 1");

        // Hand-build the raw n*12 cells (raw_gain == net gain since min_gain_shift = 0).
        let mut cells = vec![0.0f64; 4 * 12];
        for (f, s) in splits.iter().enumerate() {
            put_raw(
                &mut cells,
                f,
                true,
                s.threshold,
                s.gain, // raw_gain (min_gain_shift = 0 ⇒ raw == net)
                s.default_left,
                s.left_sum_gradient,
                s.left_sum_hessian,
                s.right_sum_gradient,
                s.right_sum_hessian,
            );
        }
        let h_raw = client.create_from_slice(f64::as_bytes(&cells));
        let out = SplitSoa::zeroed(&client, 1);
        launch_reduce_into_leaf(&client, h_raw, cells.len(), &real_feats, 4, &out, 0, 0, 0.0);

        let got = out.read_record(&client, 0);
        assert_slot_eq(&got, &host_best, real_feats[host_fpos as usize]);
        assert_eq!(got.inner_feature_index, 2, "tie must resolve to real feature index 2");
    }

    /// Build a concatenated stride-2 f64 histogram for a batch of features that all sum to the
    /// SAME leaf totals (`sum_gradient`, `sum_hessian`) — a valid single-leaf batch.
    #[allow(clippy::type_complexity)]
    fn batch_hist() -> (Vec<f64>, Vec<BatchedSplitFeature>, Vec<i32>, f64, f64, i32) {
        // Each feature's bins sum to (sum_g = -1, sum_h = 20).
        let fa: [(f64, f64); 4] = [(-10.0, 5.0), (-8.0, 5.0), (9.0, 5.0), (8.0, 5.0)];
        let fb: [(f64, f64); 4] = [(-2.0, 4.0), (5.0, 6.0), (-7.0, 5.0), (3.0, 5.0)];
        let fc: [(f64, f64); 3] = [(-6.0, 8.0), (2.0, 6.0), (3.0, 6.0)];
        let mut buf = Vec::new();
        let push = |buf: &mut Vec<f64>, bins: &[(f64, f64)]| {
            for &(g, h) in bins {
                buf.push(g);
                buf.push(h);
            }
        };
        let so_a = 0usize;
        push(&mut buf, &fa);
        let so_b = buf.len();
        push(&mut buf, &fb);
        let so_c = buf.len();
        push(&mut buf, &fc);
        let mk = |slot_off: usize, num_bin: u32| BatchedSplitFeature {
            slot_off,
            num_bin,
            offset: 0,
            default_bin: num_bin, // out of range ⇒ never skipped
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: true,
        };
        let feats = vec![mk(so_a, 4), mk(so_b, 4), mk(so_c, 3)];
        let real_feats = vec![5i32, 2, 9];
        (buf, feats, real_feats, -1.0, 20.0, 20)
    }

    fn relaxed_cfg() -> GainConfig {
        GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        }
    }

    /// The single-leaf no-readback launcher folds a winner into the target `SplitSoa`
    /// slot BIT-EXACT to `find_best_splits_batched_fused_f64_from_handle_on` (host readback) +
    /// `argmax_over_resident_splits`, on a real (non-tied) multi-feature histogram batch.
    #[test]
    fn split_reduce_into_leaf_matches_host_readback() {
        use crate::kernels::best_split::SplitSoa;
        use crate::kernels::grow_driver::argmax_over_resident_splits;

        let client = cpu_client();
        let (buf, feats, real_feats, sg, sh, nd) = batch_hist();
        let cfg = relaxed_cfg();

        // Host: readback launcher → Vec<SplitInfo> → host argmax.
        let h_hist = client.create_from_slice(f64::as_bytes(&buf));
        let host_splits = find_best_splits_batched_fused_f64_from_handle_on(
            &client, h_hist, buf.len(), &feats, &cfg, sg, sh, nd,
        )
        .expect("host readback launcher");
        let (host_best, host_fpos) = argmax_over_resident_splits(&host_splits, &feats, &real_feats);
        assert!(host_fpos >= 0, "there must be a valid winner");
        assert!(host_best.gain.is_finite(), "winner gain must be finite");

        // Device: no-readback reduce into slot 0.
        let h_hist2 = client.create_from_slice(f64::as_bytes(&buf));
        let out = SplitSoa::zeroed(&client, 1);
        find_best_splits_fused_reduce_into_leaf_on(
            &client, h_hist2, buf.len(), &feats, &real_feats, &cfg, sg, sh, nd, &out, 0,
        )
        .expect("device reduce launcher");
        let got = out.read_record(&client, 0);
        assert_slot_eq(&got, &host_best, real_feats[host_fpos as usize]);
    }

    /// The na_as_missing whole-batch reject is PRESERVED — the reduce launcher returns
    /// the SAME typed `ComputeError::Runtime` the host-readback launcher returns for the
    /// identical input (not silently bypassed).
    #[test]
    fn split_reduce_into_leaf_na_as_missing_reject() {
        use crate::kernels::best_split::SplitSoa;

        let client = cpu_client();
        let (buf, mut feats, real_feats, sg, sh, nd) = batch_hist();
        feats[1].na_as_missing = true; // the middle feature carries NA_AS_MISSING
        let cfg = relaxed_cfg();

        // Host-readback launcher rejects the whole batch.
        let h_hist = client.create_from_slice(f64::as_bytes(&buf));
        let host_err = find_best_splits_batched_fused_f64_from_handle_on(
            &client, h_hist, buf.len(), &feats, &cfg, sg, sh, nd,
        )
        .unwrap_err();
        assert!(matches!(host_err, ComputeError::Runtime { .. }));

        // Device reduce launcher rejects identically.
        let h_hist2 = client.create_from_slice(f64::as_bytes(&buf));
        let out = SplitSoa::zeroed(&client, 1);
        let dev_err = find_best_splits_fused_reduce_into_leaf_on(
            &client, h_hist2, buf.len(), &feats, &real_feats, &cfg, sg, sh, nd, &out, 0,
        )
        .unwrap_err();
        assert!(
            matches!(dev_err, ComputeError::Runtime { .. }),
            "na_as_missing whole-batch reject must be preserved, got {dev_err:?}"
        );
    }

    // ================= co-pack siblings_reduce_into_leaves =================

    /// The raw-cell-level fixture (co-pack) — sibling A's features TIE at the
    /// max gain across out-of-fpos-order real indices (fpos 1 wins, real 2), sibling B has a
    /// single clear winner (fpos 2, real 5). Two invocations of the shared reduce kernel
    /// (raw_base 0 for A, n*12 for B) fold each winner into its own `SplitSoa` slot, bit-exact
    /// to `argmax_over_resident_splits` on that sibling's `Vec<SplitInfo>`.
    #[test]
    fn siblings_reduce_into_leaves_tie_fixture() {
        use crate::kernels::best_split::SplitSoa;
        use crate::kernels::grow_driver::argmax_over_resident_splits;

        let client = cpu_client();
        let real_feats = vec![7i32, 2, 5, 2];
        let feats = vec![rfeat(false), rfeat(false), rfeat(false), rfeat(false)];

        // Sibling A: 3-way tie at gain 8 (fpos 0/1/3), fpos 1 (real 2) wins.
        let splits_a = vec![
            si(100, 8.0, true, 1.0, 2.0, 3.0, 4.0),
            si(111, 8.0, false, 11.0, 12.0, 13.0, 14.0), // winner
            si(122, 4.0, true, 21.0, 22.0, 23.0, 24.0),
            si(133, 8.0, false, 31.0, 32.0, 33.0, 34.0),
        ];
        // Sibling B: clear winner at fpos 2 (real 5, gain 10).
        let splits_b = vec![
            si(200, 3.0, true, 41.0, 42.0, 43.0, 44.0),
            si(211, 5.0, false, 51.0, 52.0, 53.0, 54.0),
            si(222, 10.0, true, 61.0, 62.0, 63.0, 64.0), // winner
            si(233, 1.0, false, 71.0, 72.0, 73.0, 74.0),
        ];

        let (host_a, fpos_a) = argmax_over_resident_splits(&splits_a, &feats, &real_feats);
        let (host_b, fpos_b) = argmax_over_resident_splits(&splits_b, &feats, &real_feats);
        assert_eq!(fpos_a, 1, "sibling A tie → fpos 1");
        assert_eq!(fpos_b, 2, "sibling B clear winner → fpos 2");

        // Hand-build the co-packed 2*n*12 raw array (A at [0,n), B at [n,2n)); shift = 0.
        let mut cells = vec![0.0f64; 2 * 4 * 12];
        for (f, s) in splits_a.iter().enumerate() {
            put_raw(&mut cells, f, true, s.threshold, s.gain, s.default_left,
                s.left_sum_gradient, s.left_sum_hessian, s.right_sum_gradient, s.right_sum_hessian);
        }
        for (f, s) in splits_b.iter().enumerate() {
            put_raw(&mut cells, 4 + f, true, s.threshold, s.gain, s.default_left,
                s.left_sum_gradient, s.left_sum_hessian, s.right_sum_gradient, s.right_sum_hessian);
        }
        let h_raw = client.create_from_slice(f64::as_bytes(&cells));
        let out = SplitSoa::zeroed(&client, 2);
        // Sibling A → slot 0 (raw_base 0); sibling B → slot 1 (raw_base n*12).
        launch_reduce_into_leaf(&client, h_raw.clone(), cells.len(), &real_feats, 4, &out, 0, 0, 0.0);
        launch_reduce_into_leaf(&client, h_raw, cells.len(), &real_feats, 4, &out, 1, 4 * 12, 0.0);

        let got_a = out.read_record(&client, 0);
        let got_b = out.read_record(&client, 1);
        assert_slot_eq(&got_a, &host_a, real_feats[fpos_a as usize]);
        assert_slot_eq(&got_b, &host_b, real_feats[fpos_b as usize]);
        assert_eq!(got_a.inner_feature_index, 2, "sibling A tie → real feature 2");
        assert_eq!(got_b.inner_feature_index, 5, "sibling B → real feature 5");
    }

    /// A second stride-2 histogram batch (same 4/4/3-bin layout as `batch_hist`) whose features
    /// all sum to (sum_g = 0, sum_h = 12) — the "larger sibling" of the co-pack parity test.
    fn batch_hist_b() -> (Vec<f64>, f64, f64, i32) {
        let ga: [(f64, f64); 4] = [(-5.0, 3.0), (-4.0, 3.0), (6.0, 3.0), (3.0, 3.0)];
        let gb: [(f64, f64); 4] = [(1.0, 3.0), (-2.0, 3.0), (4.0, 3.0), (-3.0, 3.0)];
        let gc: [(f64, f64); 3] = [(-4.0, 4.0), (1.0, 4.0), (3.0, 4.0)];
        let mut buf = Vec::new();
        for bins in [&ga[..], &gb[..], &gc[..]] {
            for &(g, h) in bins {
                buf.push(g);
                buf.push(h);
            }
        }
        (buf, 0.0, 12.0, 12)
    }

    /// The co-pack no-readback launcher matches
    /// `find_best_splits_fused_siblings_from_handles_on` + `argmax_over_resident_splits` on BOTH
    /// sibling halves, bit-exact, on a real (non-tied) synthetic batch.
    #[test]
    fn siblings_reduce_into_leaves_matches_host_readback() {
        use crate::kernels::best_split::SplitSoa;
        use crate::kernels::grow_driver::argmax_over_resident_splits;

        let client = cpu_client();
        let (buf_a, feats, real_feats, sg_a, sh_a, nd_a) = batch_hist();
        let (buf_b, sg_b, sh_b, nd_b) = batch_hist_b();
        assert_eq!(buf_a.len(), buf_b.len(), "co-pack siblings share the feature layout");
        let cfg = relaxed_cfg();
        let a_totals = (sg_a, sh_a, nd_a);
        let b_totals = (sg_b, sh_b, nd_b);

        // Host: co-pack readback launcher → (vec_a, vec_b) → per-sibling host argmax.
        let h_a = client.create_from_slice(f64::as_bytes(&buf_a));
        let h_b = client.create_from_slice(f64::as_bytes(&buf_b));
        let (vec_a, vec_b) = find_best_splits_fused_siblings_from_handles_on(
            &client, h_a, h_b, buf_a.len(), &feats, &cfg, a_totals, b_totals,
        )
        .expect("host co-pack readback launcher");
        let (host_a, fpos_a) = argmax_over_resident_splits(&vec_a, &feats, &real_feats);
        let (host_b, fpos_b) = argmax_over_resident_splits(&vec_b, &feats, &real_feats);
        assert!(fpos_a >= 0 && fpos_b >= 0, "both siblings must have a valid winner");

        // Device: co-pack no-readback reduce into slots 0 (A) and 1 (B).
        let h_a2 = client.create_from_slice(f64::as_bytes(&buf_a));
        let h_b2 = client.create_from_slice(f64::as_bytes(&buf_b));
        let out = SplitSoa::zeroed(&client, 2);
        find_best_splits_fused_siblings_reduce_into_leaves_on(
            &client, h_a2, h_b2, buf_a.len(), &feats, &real_feats, &cfg, a_totals, b_totals, &out, 0, 1,
        )
        .expect("device co-pack reduce launcher");
        let got_a = out.read_record(&client, 0);
        let got_b = out.read_record(&client, 1);
        assert_slot_eq(&got_a, &host_a, real_feats[fpos_a as usize]);
        assert_slot_eq(&got_b, &host_b, real_feats[fpos_b as usize]);
    }
}
