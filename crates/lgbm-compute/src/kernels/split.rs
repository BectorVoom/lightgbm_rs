//! `find_best_split` cube kernel — the gain math lives INSIDE the kernel (D-01a).
//!
//! VERBATIM transcription of `FindBestThresholdSequentially`
//! (`LightGBM/src/treelearner/feature_histogram.hpp:830-1057`, commit 195c26fc,
//! VERSION 4.6.0.99) for the default CPU template instantiation
//! `<USE_RAND=false, USE_MC=false, USE_L1=?, USE_MAX_OUTPUT=false,
//! USE_SMOOTHING=false, SKIP_DEFAULT_BIN=?, NA_AS_MISSING=false>`. Both the
//! REVERSE branch (`:854-936`, records `t-1+offset`) and the FORWARD branch
//! (`:937-1029`, records `t+offset`) are transcribed 1:1 — NO loop
//! restructuring, NO gate reordering (RESEARCH Pitfall 5).
//!
//! The gain primitives ([`get_split_gains`] / [`get_leaf_gain`] /
//! [`calculate_splitted_leaf_output`] / [`threshold_l1`]) are `#[cube]`
//! functions in [`crate::gain`] called from inside this scan — the gain formula
//! is computed in the kernel, not pre-supplied (D-01a).
//!
//! ## Epsilon placements (RESEARCH Pitfall 4 — load-bearing, verbatim)
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

use crate::error::ComputeError;
use crate::gain::{
    calculate_splitted_leaf_output, calculate_splitted_leaf_output_f32, get_split_gains,
    get_split_gains_f32, GainConfig, SplitInfo,
};
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

/// The single-owner ordered best-split scan (REVERSE + FORWARD branches).
///
/// `cfg`-style scalars (`min_data_in_leaf`, hessian/gain gates, lambdas,
/// `min_gain_shift`, `use_l1`, `skip_default_bin`, `offset`, `default_bin`) are
/// passed as comptime/scalar args. `sum_gradient`/`sum_hessian`/`num_data` are
/// the leaf totals (`sum_hessian` already bumped by `2*kEpsilon`). `hist` is the
/// stride-2 `[g0,h0,g1,h1,...]` f64 histogram.
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
    // Single-owner ordered scan: the launch uses `CubeDim::new_1d(1)` so exactly
    // ONE unit exists (UNIT_POS == 0). The scan is inherently sequential; one
    // owner is the cpu deterministic anchor. (No UNIT_POS guard is needed at
    // CubeDim 1, and wrapping the whole complex body in an `if` tripped the
    // cubecl-cpu MLIR lowering.)
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
    // C++ `break` produces. (RESEARCH Pitfall 5: gate ORDER preserved exactly.)
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
            let skip = skip_def && (t + offset) == default_bin;
            // Accumulate only when in range, not skipped and not already stopped.
            let active = in_range && !skip && !done;
            // Branchless clamp (cubecl-cpu mis-lowers nested-`if`): a negative `t`
            // reads bin 0; it is inactive so it never contributes to any sum.
            let t_safe = select(t < 0, 0i32, t);
            let bi = (t_safe as usize) * 2; // GET_GRAD index = t<<1
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
            let skip = skip_def && (t + offset) == default_bin;
            let active = !skip && !done;
            // t >= 0 always here (range starts at 0; NA_AS_MISSING=0 path).
            let bi = (t as usize) * 2;
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

    out[0] = is_splittable; // already an f64 flag (0.0 / 1.0)
    out[1] = f64::cast_from(best_threshold);
    out[2] = best_gain; // RAW best_gain (kMinScore if none)
    out[3] = f64::cast_from(best_left_count);
    out[4] = f64::cast_from(num_data - best_left_count); // right_count
    out[5] = best_sum_left_gradient;
    out[6] = best_sum_left_hessian - eps; // reported left_sum_hessian (:1042)
    out[7] = right_sum_gradient;
    out[8] = right_sum_hessian - eps; // reported right_sum_hessian (:1053)
    out[9] = best_default_left; // already an f64 flag (0.0 / 1.0)
    out[10] = left_output;
    out[11] = right_output;
}

/// f32-cell mirror of [`find_best_split_kernel`] for the no-f64 hip device
/// (CMP-04). The scan structure, gate ORDER, branchless-`select` encoding, and
/// monotone-`done` flag are IDENTICAL to the f64 kernel — the ONLY difference is
/// the accumulation/gain cell type (`f32` instead of `f64`), since hip (gfx1100)
/// cannot allocate f64 (RESEARCH Pitfall 2/3). The `out` protocol is the same 12
/// cells, here in f32. The hip parity gate compares these to the f64 cpu anchor
/// (collected to f32) within `ORACLE_TOL = 1e-6` (D-03a).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_kernel_f32(
    hist: &Array<f32>,
    out: &mut Array<f32>,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f32,
    lambda_l1: f32,
    lambda_l2: f32,
    min_gain_shift: f32,
    sum_gradient: f32,
    sum_hessian: f32,
    num_data: i32,
    rev_count: i32,
    fwd_count: i32,
) {
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let use_l1_b = use_l1 != 0;
    let skip_def = skip_default_bin != 0;

    let cnt_factor = f32::cast_from(num_data) / sum_hessian;

    let mut best_sum_left_gradient = 0.0f32;
    let mut best_sum_left_hessian = 0.0f32;
    let mut best_gain = 0.0f32;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = 0.0f32;
    let mut best_default_left = 1.0f32;

    // ====================== REVERSE branch (:854-936) ======================
    {
        let mut sum_right_gradient = 0.0f32;
        let mut sum_right_hessian = f32::cast_from(K_EPSILON); // kEpsilon (:856)
        let mut right_count = 0i32;

        let t_start = num_bin - 1 - offset;
        let count = rev_count;
        let mut done = false;

        for k in 0..count {
            let t = t_start - k;
            // Restore the C++ `t >= t_end` (t_end = 1 - offset) lower bound that
            // the `0..count` counter form drops, and clamp the read index so a
            // negative `t` (offset >= 2, out of the C++ offset∈{0,1} contract)
            // cannot wrap `(t as usize)` into an OOB read (WR-02). No-op for the
            // valid offset∈{0,1} cases.
            let in_range = t >= (1 - offset);
            let skip = skip_def && (t + offset) == default_bin;
            let active = in_range && !skip && !done;
            let t_safe = select(t < 0, 0i32, t);
            let bi = (t_safe as usize) * 2;
            let g = hist[bi];
            let h = hist[bi + 1];
            sum_right_gradient += select(active, g, 0.0);
            sum_right_hessian += select(active, h, 0.0);
            right_count += select(active, round_int_f32(h * cnt_factor), 0i32);

            let left_count = num_data - right_count;
            let sum_left_hessian = sum_hessian - sum_right_hessian;
            let sum_left_gradient = sum_gradient - sum_right_gradient;
            let cont = right_count < min_data_in_leaf
                || sum_right_hessian < min_sum_hessian_in_leaf;
            let brk = left_count < min_data_in_leaf
                || sum_left_hessian < min_sum_hessian_in_leaf;
            done = done || (active && !cont && brk);
            let consider = active && !cont && !done;

            let current_gain = get_split_gains_f32(
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
            best_default_left = select(take, 1.0, best_default_left);
        }
    }

    // ====================== FORWARD branch (:937-1029) =====================
    {
        let mut sum_left_gradient = 0.0f32;
        let mut sum_left_hessian = f32::cast_from(K_EPSILON); // kEpsilon (:939)
        let mut left_count = 0i32;

        let count = fwd_count;
        let mut done = false;

        for t in 0..count {
            let skip = skip_def && (t + offset) == default_bin;
            let active = !skip && !done;
            let bi = (t as usize) * 2;
            let g = hist[bi];
            let h = hist[bi + 1];
            sum_left_gradient += select(active, g, 0.0);
            sum_left_hessian += select(active, h, 0.0);
            left_count += select(active, round_int_f32(h * cnt_factor), 0i32);

            let right_count = num_data - left_count;
            let sum_right_hessian = sum_hessian - sum_left_hessian;
            let sum_right_gradient = sum_gradient - sum_left_gradient;
            let cont = left_count < min_data_in_leaf
                || sum_left_hessian < min_sum_hessian_in_leaf;
            let brk = right_count < min_data_in_leaf
                || sum_right_hessian < min_sum_hessian_in_leaf;
            done = done || (active && !cont && brk);
            let consider = active && !cont && !done;

            let current_gain = get_split_gains_f32(
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
            best_default_left = select(take, 0.0, best_default_left);
        }
    }

    // ---- finalization (feature_histogram.hpp:1031-1056) -------------------
    let eps = f32::cast_from(K_EPSILON);
    let left_output = calculate_splitted_leaf_output_f32(
        use_l1_b,
        best_sum_left_gradient,
        best_sum_left_hessian,
        l1,
        l2,
    );
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian - best_sum_left_hessian;
    let right_output =
        calculate_splitted_leaf_output_f32(use_l1_b, right_sum_gradient, right_sum_hessian, l1, l2);

    out[0] = is_splittable;
    out[1] = f32::cast_from(best_threshold);
    out[2] = best_gain;
    out[3] = f32::cast_from(best_left_count);
    out[4] = f32::cast_from(num_data - best_left_count);
    out[5] = best_sum_left_gradient;
    out[6] = best_sum_left_hessian - eps;
    out[7] = right_sum_gradient;
    out[8] = right_sum_hessian - eps;
    out[9] = best_default_left;
    out[10] = left_output;
    out[11] = right_output;
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
/// penalty` at `:174`) defaults to `1.0` here (Phase-7+ feature-penalty scope).
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `hist.len() != 2 * num_bin`.
/// - [`ComputeError::Runtime`] if `num_bin == 0` or `sum_hessian` is non-positive
///   (the C++ `cnt_factor = num_data / sum_hessian` would divide by ~0), or if
///   unsupported non-default gain params are supplied (max_delta_step /
///   path_smooth — Phase-7+).
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
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
    // NA_AS_MISSING forward-branch (feature_histogram.hpp:945-961) is deferred
    // (RESEARCH A5) — this layer threads the flag and validates it false on every
    // captured case, but the NaN-missing forward preamble kernel body is NOT yet
    // transcribed. Surface a TYPED error rather than silently mis-computing
    // (T-05-01-01): the unimplemented branch can never produce a wrong SplitInfo.
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
    // Phase-4 scope: only the default no-op output-clamp / smoothing path is
    // transcribed. Reject non-default values rather than silently mis-computing.
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }

    // BeforeNumerical (feature_histogram.hpp:198-207): the whole-leaf gain, then
    // min_gain_shift = gain_shift + min_gain_to_split. Computed on the host with
    // the SAME f64 gain primitive the kernel uses (so it is bit-identical).
    let use_l1 = cfg.use_l1();
    // gain_shift is GetLeafGain over the (un-bumped) leaf totals.
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient,
        sum_hessian,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    // FindBestThreshold entry bump: sum_hessian + 2 * kEpsilon (:172). The 2 and
    // kEpsilon widen to f64 exactly as C++ does.
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;

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
    let zeros = vec![0.0f64; out_len];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

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

/// Host-side `find_best_split` in **f32 cells** on ANY runtime (the no-f64 hip
/// path; CMP-03/CMP-04). Mirrors [`find_best_split_cpu`]'s host pre-step
/// (`BeforeNumerical` `min_gain_shift`, the `2*kEpsilon` entry bump) and V5
/// validation, but in f32, and returns the RAW 12 `out` cells the f32 scan wrote
/// `[is_splittable, threshold, gain, left_count, right_count, left_sum_gradient,
///  left_sum_hessian, right_sum_gradient, right_sum_hessian, default_left,
///  left_output, right_output]`. The hip parity gate compares these to the cpu
/// f64 anchor's raw cells (collected to f32) within `ORACLE_TOL = 1e-6`.
///
/// Generic over `R: Runtime` so it runs on the cubecl-cpu client (f32 reference)
/// AND the cubecl-hip client (the real GPU).
///
/// # Errors
/// Same as [`find_best_split_cpu`] (length / num_bin / sum_hessian / scope
/// validation, V5).
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_raw_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist: &[f32],
    cfg: &GainConfig,
    num_bin: u32,
    offset: i32,
    default_bin: u32,
    skip_default_bin: bool,
    na_as_missing: bool,
    run_forward: bool,
    sum_gradient: f32,
    sum_hessian: f32,
    num_data: i32,
) -> Result<Vec<f32>, ComputeError> {
    // --- V5 boundary validation (T-04-01) ---
    // NA_AS_MISSING forward branch deferred (RESEARCH A5) — typed error, never a
    // silent wrong answer (T-05-01-01). Signature-consistent with the f64 path.
    if na_as_missing {
        return Err(ComputeError::Runtime {
            detail: "find_best_split_f32: na_as_missing (NA_AS_MISSING forward branch) not yet \
                     implemented"
                .to_string(),
        });
    }
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split_f32: num_bin must be > 0".to_string(),
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
            detail: "find_best_split_f32: sum_hessian must be > 0".to_string(),
        });
    }
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "find_best_split_f32: max_delta_step / path_smooth are Phase-7+ scope"
                .to_string(),
        });
    }

    // BeforeNumerical (host, f32): gain_shift over the un-bumped leaf totals.
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain_f32(
        use_l1,
        sum_gradient,
        sum_hessian,
        cfg.lambda_l1 as f32,
        cfg.lambda_l2 as f32,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split as f32;

    let two_eps = 2.0f32 * f32::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;

    let num_bin_i = num_bin as i32;
    let rev_count = (num_bin_i - 1).max(0);
    // FORWARD branch dispatch (feature_histogram.hpp:420-429) — see the f64
    // `find_best_split_cpu` for the full transcription. `run_forward` is the
    // verbatim `num_bin > 2 && missing_type == Zero` truth value; when false we
    // drive `fwd_count = 0` so only the REVERSE branch contributes.
    let fwd_count = if run_forward {
        (num_bin_i - 1 - offset).max(0)
    } else {
        0
    };

    let out_len = 12usize;
    let h_hist = client.create_from_slice(f32::as_bytes(hist));
    let zeros = vec![0.0f32; out_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    // SAFETY: identical handle/length correspondence to `find_best_split_cpu` —
    // `h_hist` sized `2*num_bin`, `h_out` sized 12 f32 cells, both outliving the
    // launch; the scan reads `hist[(t<<1)+1]` only for in-range `t`. cubecl
    // `unsafe` confined here (CMP-01).
    unsafe {
        find_best_split_kernel_f32::launch(
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
            cfg.min_sum_hessian_in_leaf as f32,
            cfg.lambda_l1 as f32,
            cfg.lambda_l2 as f32,
            min_gain_shift,
            sum_gradient,
            sum_hessian_bumped,
            num_data,
            rev_count,
            fwd_count,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).to_vec())
}

// The Phase-4 `cfg_skip_default_bin(default_bin, num_bin)` heuristic
// (`default_bin < num_bin`) was REMOVED in Phase-5 (Plan 05-01, RESEARCH
// Pitfall 1). The authoritative `SKIP_DEFAULT_BIN`/`NA_AS_MISSING` flags
// (`feature_histogram.hpp:284-285`: `num_bin > 2 && missing_type == Zero` for
// skip, `num_bin > 2 && missing_type == NaN` for na_as_missing, both false for
// `missing_type == None`) are now derived in the learner from
// `bin_mapper.missing_type()` and threaded through `Backend::find_best_split` /
// `find_best_split_cpu` / `find_best_split_raw_f32_on` as explicit params, so the
// kernel dispatch matches C++ exactly instead of approximating it from the bin
// layout.

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
    /// is deferred, RESEARCH A5) — never a panic, never a wrong SplitInfo
    /// (T-05-01-01). This is asserted BEFORE any length/num_bin validation so the
    /// deferral is unambiguous even on otherwise-valid input.
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
}
