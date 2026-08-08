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
// The fast-path (`USE_MAX_OUTPUT=false, USE_SMOOTHING=false`) primitives are still
// used by the `gpu`-only staged/official scan bodies below; the `*_full` pair is
// what the shared `split_scan_body` and the native host scan call.
#[cfg(feature = "gpu")]
use crate::gain::{calculate_splitted_leaf_output, get_split_gains};
use crate::gain::{calculate_splitted_leaf_output_full, get_split_gains_full, GainConfig, SplitInfo};
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
///
/// `na_as_missing` (0|1) admits the C++ `NA_AS_MISSING=true` template arm
/// (`feature_histogram.hpp:830-1057`, SPEC-G4-1/T-G4-1): REVERSE excludes the
/// TOP bin (`num_bin-1`, the NaN sentinel bin `na_as_missing()` routes NaN rows
/// into) from its sweep (`t_start -= na_as_missing`, `:859`) so that bin is
/// implicitly folded into "left" for every REVERSE candidate; FORWARD, when
/// `offset == 1` (the implicit most-frequent-bin optimization), pre-seeds its
/// accumulators with the reconstructed bin-0 value (`:945-961`) via SEQUENTIAL
/// subtraction of every explicit bin from the leaf totals (bit-exact operation
/// order — NOT sum-then-subtract-once) and evaluates one extra "virtual `t=-1`"
/// candidate (`left = {bin0}`, `threshold = offset-1`) before the normal `t=0..`
/// sweep. Every caller OTHER than [`find_best_split_kernel`] passes `0u32`
/// (`na_as_missing` is rejected upstream on those paths, P-4) — `0u32` is a
/// true no-op here (every na_as_missing-gated term selects its "false" branch),
/// so this extension does not perturb their existing behavior.
///
/// `max_delta_step` (G5-2, T-G5-2): the C++ `config->max_delta_step` leaf-output
/// clamp (`CalculateSplittedLeafOutput`'s `USE_MAX_OUTPUT` branch,
/// feature_histogram.hpp:716-738). When non-zero, BOTH the per-candidate scan
/// gain (`GetSplitGains<..,USE_MAX_OUTPUT,..>`) AND the finalization leaf
/// outputs dispatch to the clamped form ([`crate::gain::get_split_gains_clamped`]
/// / [`crate::gain::calculate_splitted_leaf_output_clamped`]) instead of the
/// closed-form fast path. Every caller OTHER than [`find_best_split_kernel`]
/// passes `0.0f64` (`max_delta_step` is rejected upstream on those paths, P-4) —
/// a true no-op (the `!= 0.0` dispatch below always selects the unchanged
/// closed-form path), so this extension does not perturb their existing behavior.
///
/// `path_smooth` / `parent_output` (G5-3, T-G5-3): the C++ `config->path_smooth`
/// leaf-output blend-toward-parent (`CalculateSplittedLeafOutput`'s
/// `USE_SMOOTHING` branch, feature_histogram.hpp:733-736). When `path_smooth !=
/// 0.0`, BOTH the per-candidate scan gain AND the finalization leaf outputs
/// dispatch to the smoothed form ([`crate::gain::get_split_gains_smoothed`] /
/// [`crate::gain::calculate_splitted_leaf_output_smoothed`]), using that SIDE's
/// own row count (`left_count`/`right_count`) as the smoothing `num_data`
/// (`feature_histogram.hpp:772-778`). `max_delta_step` and `path_smooth`
/// non-default SIMULTANEOUSLY is rejected upstream (the composed clamp+smooth
/// form is not transcribed). Every caller OTHER than [`find_best_split_kernel`]
/// passes `0.0f64`/`0.0f64` — a true no-op, so this extension does not perturb
/// their existing behavior.
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
    na_as_missing: u32,    // 0|1 (T-G4-1)
    use_l1: u32,           // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    // The C++ `USE_MAX_OUTPUT` axis: `max_delta_step` clamps every computed leaf
    // output, and `max_delta_step > 0` also selects the given-output gain FORM.
    max_delta_step: f64,
    // The C++ `USE_SMOOTHING` template bool as a 0|1 runtime flag — the host
    // resolves `path_smooth > kEpsilon` once, mirroring `FuncForNumricalL2`.
    use_smoothing: u32, // 0|1
    path_smooth: f64,
    // The leaf's `GetParentOutput`; read only when `use_smoothing != 0`.
    parent_output: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    rev_count: i32, // host-computed REVERSE iteration count = max(0, num_bin-1-na_as_missing)
    fwd_count: i32, // host-computed FORWARD iteration count = max(0, num_bin-1-offset [+1 iff na_as_missing&&offset==1])
) {
    // `hb`/`ob` are the feature's base offsets into the (concatenated) histogram and
    // the 12-cell `out` window. They are 0,0 for the single-feature kernel.
    let hb = hist_base as usize;
    let ob = out_base as usize;
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let use_l1_b = use_l1 != 0;
    let use_smoothing_b = use_smoothing != 0;
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
    // C++ initializes `best_gain = kMinScore` (-inf). A `0.0` sentinel USED to stand
    // in for that, justified by "every valid gain is non-negative because gains are
    // `g²/(h+λ)` sums". That justification DIES under `max_delta_step` / `path_smooth`:
    // both switch `GetLeafGain` to the given-output form `-(2·g·o + (h+λ)·o²)`, which
    // is freely NEGATIVE once `o` is clamped or blended away from the unconstrained
    // optimum. With a `0.0` sentinel a leaf whose every candidate had negative gain
    // kept `best_gain = 0.0` and `best_threshold = 0` while `is_splittable` still went
    // true, so the launcher reported a BOGUS split at bin 0 with net gain
    // `0 - min_gain_shift` — a large POSITIVE number whenever `min_gain_shift` is
    // negative, which then won the best-first race.
    //
    // cubecl-cpu's MLIR lowering requires loop-carried mutables to init from LITERALS,
    // so -inf cannot be the initializer here. `has_best` (a 0.0/1.0 literal flag)
    // reproduces it exactly instead: the FIRST valid candidate is always taken (C++
    // `x > -inf`), and every later one needs a strict `>` (C++ `x > best_gain`).
    let mut best_gain = 0.0f64;
    let mut has_best = 0.0f64;
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

        // t_start excludes the TOP bin (na_as_missing sentinel, `-1`) so it is
        // never explicitly swept in REVERSE (feature_histogram.hpp:859);
        // implicitly folded into "left" via `sum_left = sum_gradient -
        // sum_right`. `na_as_missing == 0` reproduces `num_bin-1-offset` verbatim.
        let t_start = num_bin - 1 - offset - i32::cast_from(na_as_missing); // t_end = 1 - offset
        let count = rev_count; // host-computed = max(0, num_bin-1-na_as_missing)
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
            let current_gain = get_split_gains_full(
                use_l1_b,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
                max_delta_step,
                use_smoothing_b,
                path_smooth,
                left_count,
                right_count,
                parent_output,
            );
            let valid = consider && current_gain > min_gain_shift;
            is_splittable = select(valid, 1.0, is_splittable);
            // C++ `if (current_gain > best_gain)` against a -inf init: the first
            // valid candidate always wins, later ones need a strict `>` (keep
            // first on tie).
            let take = valid && (has_best == 0.0 || current_gain > best_gain);
            has_best = select(take, 1.0, has_best);
            best_left_count = select(take, left_count, best_left_count);
            best_sum_left_gradient = select(take, sum_left_gradient, best_sum_left_gradient);
            best_sum_left_hessian = select(take, sum_left_hessian, best_sum_left_hessian);
            // left <= threshold, right > threshold => t-1 (:933)
            best_threshold = select(take, t - 1 + offset, best_threshold);
            best_gain = select(take, current_gain, best_gain);
            best_default_left = select(take, 1.0, best_default_left); // REVERSE
        }
    }

    // ====================== FORWARD branch (:937-1029) =====================
    // C++ `for (int t = 0; t <= num_bin-2-offset; ++t)`. Same range-loop +
    // monotone-`done` treatment (FORWARD grows the left side, so once the right
    // side is too small / right hessian too small it stays so — `break`==`done`).
    {
        let mut sum_left_gradient = 0.0f64;
        // Literal-init the accumulator at 0.0 (cubecl-cpu MLIR lowering
        // constraint #1) and fold in the WHOLE desired initial value via ONE
        // `select` + `+=`, rather than starting from the `kEpsilon` literal and
        // separately adding an na_preamble delta: `0.0 + X == X` exactly for any
        // finite `X` (no intermediate rounding), so this is bit-exact to a
        // direct assignment in EITHER branch — matching
        // [`find_best_split_cpu_native`]'s `sum_left_hessian = kEpsilon;` /
        // `sum_left_hessian = sum_hessian - kEpsilon;` exactly (verified by
        // `find_best_split_na_as_missing_native_matches_kernel`).
        let mut sum_left_hessian = 0.0f64;
        let mut left_count = 0i32;

        // NA_AS_MISSING FORWARD preamble (feature_histogram.hpp:945-961): ONLY
        // when `offset == 1` (the implicit most-frequent-bin, bin 0, is NOT
        // stored explicitly in `hist`). Reconstruct bin 0 by pre-seeding the
        // accumulators with the FULL leaf totals, then subtracting every
        // EXPLICIT bin in ascending order — bit-exact SEQUENTIAL subtraction
        // (`sum_gradient - g0 - g1 - ... `), NOT a sum-then-subtract-once
        // shortcut, matching C++'s loop-accumulated operation order exactly.
        // `all_bins` is `select`-zeroed (a true 0-iteration loop) when the
        // preamble is inactive, so this costs nothing on the (far more common)
        // non-`na_as_missing` / `offset==0` paths.
        let na_preamble = (na_as_missing != 0u32) && (offset == 1);
        sum_left_gradient += select(na_preamble, sum_gradient, 0.0);
        sum_left_hessian += select(
            na_preamble,
            sum_hessian - f64::cast_from(K_EPSILON),
            f64::cast_from(K_EPSILON), // kEpsilon (:939), the non-na_preamble init
        );
        left_count += select(na_preamble, num_data, 0i32);
        let all_bins = select(na_preamble, num_bin - offset, 0i32);
        for i in 0..all_bins {
            let bi2 = hb + (i as usize) * 2;
            sum_left_gradient -= hist[bi2];
            sum_left_hessian -= hist[bi2 + 1];
            left_count -= round_int(hist[bi2 + 1] * cnt_factor);
        }
        // `fwd_start = -1` iff the preamble is active — the FIRST FORWARD
        // candidate is the "virtual `t=-1`" one (`left = {bin0}` only, no bin
        // add this iteration; `threshold = offset - 1`). Otherwise unchanged
        // (`fwd_start = 0`, byte-identical to the pre-T-G4-1 body).
        let fwd_start = select(na_preamble, -1i32, 0i32);

        let count = fwd_count; // host-computed; +1 over the base count iff na_preamble
        let mut done = false;

        for k in 0..count {
            let t = fwd_start + k;
            let skip = skip_def && (t + offset) == default_bin as i32;
            let active = !skip && !done;
            // C++ `if (t >= 0) { sum_left_gradient += ...; }` (:969) — the
            // virtual `t=-1` preamble candidate adds nothing here (its
            // contribution was already folded in above).
            let do_add = active && (t >= 0);
            let t_safe = select(t < 0, 0i32, t); // clamp so `t=-1` still reads a valid (unused) cell
            let bi = hb + (t_safe as usize) * 2;
            let g = hist[bi];
            let h = hist[bi + 1];
            sum_left_gradient += select(do_add, g, 0.0);
            sum_left_hessian += select(do_add, h, 0.0);
            left_count += select(do_add, round_int(h * cnt_factor), 0i32);

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

            let current_gain = get_split_gains_full(
                use_l1_b,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
                max_delta_step,
                use_smoothing_b,
                path_smooth,
                left_count,
                right_count,
                parent_output,
            );
            let valid = consider && current_gain > min_gain_shift;
            is_splittable = select(valid, 1.0, is_splittable);
            // C++ `if (current_gain > best_gain)` against a -inf init: the first
            // valid candidate always wins, later ones need a strict `>` (keep
            // first on tie).
            let take = valid && (has_best == 0.0 || current_gain > best_gain);
            has_best = select(take, 1.0, has_best);
            best_left_count = select(take, left_count, best_left_count);
            best_sum_left_gradient = select(take, sum_left_gradient, best_sum_left_gradient);
            best_sum_left_hessian = select(take, sum_left_hessian, best_sum_left_hessian);
            // forward records t+offset (NOT t-1+offset) (:1025)
            best_threshold = select(take, t + offset, best_threshold);
            best_gain = select(take, current_gain, best_gain);
            best_default_left = select(take, 0.0, best_default_left); // FORWARD
        }
    }

    // ---- finalization (feature_histogram.hpp:1031-1056) -------------------
    // The launcher applies the `is_splittable && best_gain > output->gain +
    // min_gain_shift` accept gate + the `- min_gain_shift` / `* penalty`
    // adjustments; here we emit the RAW winner state. We DO compute the f64
    // outputs + subtract kEpsilon back off the reported hessians (:1042/:1053)
    // so the launcher has the exact SplitInfo cells. G5-2/G5-3: clamped/smoothed
    // dispatch, same as the per-candidate scan gain above.
    let eps = f64::cast_from(K_EPSILON);
    // Each side's smoothing weight uses that side's OWN row count (:1034-1049).
    let best_right_count = num_data - best_left_count;
    let left_output = calculate_splitted_leaf_output_full(
        use_l1_b,
        best_sum_left_gradient,
        best_sum_left_hessian,
        l1,
        l2,
        max_delta_step,
        use_smoothing_b,
        path_smooth,
        best_left_count,
        parent_output,
    );
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian - best_sum_left_hessian;
    let right_output = calculate_splitted_leaf_output_full(
        use_l1_b,
        right_sum_gradient,
        right_sum_hessian,
        l1,
        l2,
        max_delta_step,
        use_smoothing_b,
        path_smooth,
        best_right_count,
        parent_output,
    );

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
    na_as_missing: u32,    // 0|1 (T-G4-1)
    use_l1: u32,           // 0|1
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    rev_count: i32, // host-computed REVERSE iteration count = max(0, num_bin-1-na_as_missing)
    fwd_count: i32, // host-computed FORWARD iteration count
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
        na_as_missing,
        use_l1,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        lambda_l1,
        lambda_l2,
        max_delta_step,
        use_smoothing,
        path_smooth,
        parent_output,
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
    // NA_AS_MISSING (feature_histogram.hpp:945-961, T-G4-1) is now transcribed in
    // `split_scan_body`'s shared REVERSE/FORWARD scan — see the `na_as_missing`
    // doc comment there for the exact fold. The typed-error gate that used to
    // live here has been removed for THIS host target only (P-4); the
    // fused/batched/staged/resident-reduce kernel families keep rejecting
    // upstream of the batch reaching `split_scan_body` (their own `if
    // f.na_as_missing { return Err(...) }` checks).
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
    // max_delta_step and path_smooth (individually or combined) are fully
    // transcribed below via the two-axis gain primitives — no rejection needed.

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
    // The C++ `USE_SMOOTHING` template bool (`path_smooth > kEpsilon`),
    // resolved once per launch exactly like `USE_L1` above.
    let use_smoothing = cfg.use_smoothing();
    let gain_shift = crate::gain::get_leaf_gain_full(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
        cfg.max_delta_step,
        use_smoothing,
        cfg.path_smooth,
        num_data,
        cfg.parent_output,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    // Host-computed iteration counts (the loops are bounded RANGE loops on the
    // device; computing the bound on the host keeps the kernel control flow
    // simple for the cubecl-cpu MLIR lowering).
    //   REVERSE: t = num_bin-1-offset-na_as_missing .. 1-offset
    //            -> count = num_bin-1-na_as_missing (feature_histogram.hpp:859)
    //   FORWARD: t = 0 .. num_bin-2-offset          -> count = num_bin-1-offset
    //            (+1, starting at t=-1, iff na_as_missing && offset==1 — the
    //            NA_AS_MISSING preamble's virtual first candidate, :945-961)
    let num_bin_i = num_bin as i32;
    let rev_count = (num_bin_i - 1 - i32::from(na_as_missing)).max(0);
    // NA_AS_MISSING FORWARD preamble applies ONLY when offset==1 (the implicit
    // most-frequent-bin optimization) — see `split_scan_body`'s doc comment.
    let na_preamble = na_as_missing && offset == 1;
    // FORWARD branch dispatch (feature_histogram.hpp:396-441): LightGBM runs the
    // FORWARD scan for `num_bin > 2 && missing_type != None` (Zero OR NaN); for
    // `missing_type == None` (and num_bin <= 2) it dispatches the REVERSE branch
    // ONLY, so `FindBestThreshold:170`'s pre-set `default_left = true` survives
    // (decision_type == 2). The caller passes `run_forward` as a verbatim
    // transcription of that truth table; when it is false we drive `fwd_count = 0`
    // so the FORWARD loop iterates zero times and `best_default_left` keeps its
    // REVERSE/initial 1.0 — exactly mirroring C++ never invoking the FORWARD
    // FindBestThresholdSequentially for this missing_type.
    let fwd_count = if run_forward {
        let base = (num_bin_i - 1 - offset).max(0);
        if na_preamble { base + 1 } else { base }
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
            if na_as_missing { 1u32 } else { 0u32 },
            if use_l1 { 1u32 } else { 0u32 },
            cfg.min_data_in_leaf,
            cfg.min_sum_hessian_in_leaf,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.max_delta_step,
            if use_smoothing { 1u32 } else { 0u32 },
            cfg.path_smooth,
            cfg.parent_output,
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
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output: f64,
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
            max_delta_step,
            use_smoothing,
            path_smooth,
            parent_output,
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
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output: f64,
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
                    max_delta_step,
                    use_smoothing,
                    path_smooth,
                    parent_output,
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
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output_a: f64,
    parent_output_b: f64,
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
            max_delta_step,
            use_smoothing,
            path_smooth,
            parent_output_a,
            parent_output_b,
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
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output_a: f64,
    parent_output_b: f64,
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
                    max_delta_step,
                    use_smoothing,
                    path_smooth,
                    parent_output_a,
                    parent_output_b,
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
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output: f64,
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
            0u32, // na_as_missing: rejected upstream on this fused/batched path (P-4) — true no-op
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            max_delta_step,
            use_smoothing,
            path_smooth,
            parent_output,
            min_gain_shift,
            sum_gradient,
            sum_hessian,
            num_data,
            rev_count[fi],
            fwd_count[fi],
        );
    }
}

/// SPEC-DRGL-12: device-`num_data` twin of [`find_best_splits_fused_kernel`]. Identical
/// per-feature scan (the SHARED [`split_scan_body`]) — the ONLY difference is that this
/// child's `num_data` (its row count) is resolved ON DEVICE from the resident split/role
/// record instead of being handed in as a host scalar. That is exactly the capability
/// SPEC-DRGL-05's `read_split` deferral needs: once the split point is deferred, the host
/// no longer knows the child count at scan-launch time.
///
/// Count resolution mirrors T-04's fixed-grid BUILD
/// (`construct_leaf_hist_resident_lds_kernel_u64_fixed_grid`): `ranges[6*split_slot+2]`
/// = `left_count` = `split_point`; `roles[3*split_slot]` = `smaller_is_left` (0/1); the
/// larger count is `parent_count - split_point` (`parent_count` is host-known — the
/// PARENT's row count, set a prior iteration, NOT the deferred value — so no new sync).
/// `is_smaller` (0/1) selects which child this scan is for. Real-device only.
/// SPEC-DRGL-12: resolve a child leaf's `num_data` (row count) ON DEVICE from the
/// resident split/role record, so the scan can run before the host reads `split_point`
/// back. Same field layout as the T-04 fixed-grid BUILD: `ranges[6*split_slot+2]` =
/// `left_count` = `split_point`; `roles[3*split_slot]` = `smaller_is_left` (0/1); the
/// larger count is `parent_count - split_point` (`parent_count` host-known, no sync).
/// `which` selects the child (SPEC-DRGL-05 generalization): `0=Left, 1=Right, 2=Smaller,
/// 3=Larger`. LEFT/RIGHT (used by the deferred loop) read only `ranges` — NO roles — so the
/// sums stay host-known (the pick export carries left/right sums directly); Smaller/Larger
/// (used by the isolation tests / T-12/T-13) consult `roles[3*split_slot]`. Shared by the
/// legacy and parprefix devcount twins so both compute the identical count.
#[cfg(feature = "gpu")]
#[cube]
fn resolve_child_num_data(
    ranges: &Array<i32>,
    roles: &Array<i32>,
    split_slot: u32,
    which: u32,
    parent_count: i32,
) -> i32 {
    let split_point = ranges[(split_slot * 6 + 2) as usize];
    let smaller_is_left = roles[(split_slot * 3) as usize] != 0;
    let left_count = split_point;
    let right_count = parent_count - split_point;
    let smaller_count = select(smaller_is_left, left_count, right_count);
    let larger_count = select(smaller_is_left, right_count, left_count);
    // which<2 ⇒ LEFT(0)/RIGHT(1) (no roles); else SMALLER(2)/LARGER(3).
    let by_lr = select(which == 0u32, left_count, right_count);
    let by_sl = select(which == 2u32, smaller_count, larger_count);
    select(which < 2u32, by_lr, by_sl)
}

#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_kernel_devcount(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // LEAF-LEVEL scalars (shared across the batch) — as find_best_splits_fused_kernel.
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    // DEVICE `num_data` source (SPEC-DRGL-12) — REPLACES the host `num_data: i32` scalar.
    ranges: &Array<i32>,
    roles: &Array<i32>,
    split_slot: u32,
    which: u32,        // 0=Left 1=Right 2=Smaller 3=Larger (SPEC-DRGL-05 generalization)
    parent_count: i32, // host-known parent row count (upper bound; no new sync)
    n_feats: u32,
) {
    // Resolve this child's num_data ON DEVICE (same field layout as the T-04 build).
    let num_data = resolve_child_num_data(ranges, roles, split_slot, which, parent_count);

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
            0u32, // na_as_missing: rejected upstream on this fused/batched path (P-4) — true no-op
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            max_delta_step,
            use_smoothing,
            path_smooth,
            parent_output,
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
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output_a: f64,
    parent_output_b: f64,
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
                0u32, // na_as_missing: rejected upstream on this fused/batched path (P-4) — true no-op
                use_l1,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l1,
                lambda_l2,
                max_delta_step,
                use_smoothing,
                path_smooth,
                parent_output_a,
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
                0u32, // na_as_missing: rejected upstream on this fused/batched path (P-4) — true no-op
                use_l1,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l1,
                lambda_l2,
                max_delta_step,
                use_smoothing,
                path_smooth,
                parent_output_b,
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

/// SPEC-DRGL-13: device-`num_data` twin of [`find_best_splits_fused_siblings_kernel`].
/// Identical co-pack scan, but BOTH siblings' `num_data` are resolved ON DEVICE from the
/// resident split/role record ([`resolve_child_num_data`]) — the smaller child's count
/// (`is_smaller=1`) for sibling A, the larger's (`is_smaller=0`) for sibling B — instead of
/// host `num_data_a`/`num_data_b` scalars. This lets the co-pack scan (the default live hip
/// arm when both children are scannable) run before the host reads `split_point` back (the
/// SPEC-DRGL-05 deferral). Real-device only. The larger count is `parent_count -
/// split_point` (parent count host-known; no new sync).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_kernel_devcount(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    max_delta_step: f64,
    use_smoothing: u32,
    path_smooth: f64,
    parent_output_a: f64,
    parent_output_b: f64,
    // PER-SIBLING leaf scalars (A = smaller, B = larger) — num_data resolved on device below.
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    // DEVICE `num_data` source (SPEC-DRGL-13) — REPLACES the host num_data_a/num_data_b.
    ranges: &Array<i32>,
    roles: &Array<i32>,
    split_slot: u32,
    which_a: u32, // sibling A's child selector (SPEC-DRGL-05: 0=Left/2=Smaller)
    which_b: u32, // sibling B's child selector (1=Right/3=Larger)
    parent_count: i32,
    n_feats: u32,
) {
    // Resolve BOTH children's num_data ON DEVICE per the caller's which_a/which_b.
    let num_data_a = resolve_child_num_data(ranges, roles, split_slot, which_a, parent_count);
    let num_data_b = resolve_child_num_data(ranges, roles, split_slot, which_b, parent_count);

    let g = ABSOLUTE_POS as u32;
    let total = 2u32 * n_feats;
    if g < total {
        let fi = if g < n_feats { g } else { g - n_feats } as usize;
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
                0u32, // na_as_missing: rejected upstream on this fused/batched path (P-4) — true no-op
                use_l1,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l1,
                lambda_l2,
                max_delta_step,
                use_smoothing,
                path_smooth,
                parent_output_a,
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
                0u32, // na_as_missing: rejected upstream on this fused/batched path (P-4) — true no-op
                use_l1,
                min_data_in_leaf,
                min_sum_hessian_in_leaf,
                lambda_l1,
                lambda_l2,
                max_delta_step,
                use_smoothing,
                path_smooth,
                parent_output_b,
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
/// Whether the LDS-staged / official / pargain / parprefix scan variants may run for
/// this config.
///
/// Those variants are PERFORMANCE forks of [`split_scan_body`] — each re-transcribes
/// the REVERSE/FORWARD scan against a shared-memory or plane-parallel layout, and each
/// implements only the default `USE_MAX_OUTPUT=false, USE_SMOOTHING=false` gain. With
/// `max_delta_step` or `path_smooth` active the gain switches to the given-output form
/// (and needs the per-side row counts + the leaf's `parent_output`), which those bodies
/// do not carry.
///
/// Rather than fork the semantics into five more scan bodies that no CI on a
/// GPU-less machine can execute, the variants simply DECLINE for such a config and the
/// launcher falls through to the shared `split_scan_body` path — which is the
/// reference every variant is required to reproduce bit-for-bit anyway. The parameters
/// therefore work on every backend; only the optimization opts out.
fn scan_variants_applicable(cfg: &GainConfig) -> bool {
    !cfg.use_max_output() && !cfg.use_smoothing()
}

fn scan_staged_enabled() -> bool {
    !matches!(std::env::var("LGBM_SCAN_STAGED").as_deref(), Ok("0"))
}

/// SPEC-DRGL-05: whether the DEFERRED grow loop's required scan configuration holds on
/// this client — the fused-subtract OFFICIAL staged co-scan (no pargain) + the par
/// reduce on a real device. The deferred driver arm consults this so a default-ON
/// deferral silently falls back to the byte-identical eager loop on any other config
/// (instead of the Backend seam's typed error).
#[cfg(feature = "gpu")]
pub fn deferred_scan_config_applies<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    cfg: &GainConfig,
) -> bool {
    <R as cubecl::Runtime>::name(client) != "cpu"
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && !scan_pargain_enabled(<R as cubecl::Runtime>::name(client))
        && scan_official_enabled(client)
        && reduce_par_enabled(client)
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

/// PARALLEL-PREFIX single-leaf kernel: the pargain twin with phase 1 replaced by
/// the all-lanes [`parprefix_store_rev`]/[`parprefix_store_fwd`] chunked block-scan
/// (rev then fwd, reusing the `ct_*`/`lmin` scratch). Phases 2/3 + merge are byte-
/// identical to the pargain kernel. ROCm-only (gated by `scan_parprefix_enabled`).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_staged_parprefix_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
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
    // Parallel-prefix scratch (chunk totals/bases + break minima), reused rev→fwd.
    let mut ct_g = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_h = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_c = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut lmin = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);

    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    let mut c = UNIT_POS as usize;
    while c < cells {
        sm[c] = hist[base + c];
        c += cd;
    }
    sync_cube();

    // PHASE 1: all-lanes parallel prefix, rev then fwd (reuse ct/lmin scratch).
    parprefix_store_rev(
        &sm.to_slice(),
        &mut rev_ag.to_slice_mut(),
        &mut rev_ah.to_slice_mut(),
        &mut rev_lc.to_slice_mut(),
        &mut rev_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
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
    sync_cube();
    parprefix_store_fwd(
        &sm.to_slice(),
        &mut fwd_ag.to_slice_mut(),
        &mut fwd_ah.to_slice_mut(),
        &mut fwd_lc.to_slice_mut(),
        &mut fwd_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
        offset[fi],
        default_bin[fi],
        skip_default_bin[fi],
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        sum_hessian,
        num_data,
        fwd_count[fi],
    );
    sync_cube();

    // PHASE 2: warp-split parallel gain scans (unchanged from pargain).
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

    // PHASE 3 + merge (unchanged from pargain).
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

/// SPEC-DRGL-12: device-`num_data` twin of [`find_best_splits_fused_staged_parprefix_kernel`]
/// (the LIVE default hip single-child scan). BYTE-FOR-BYTE the same parallel-prefix scan —
/// the ONLY change is that this child's `num_data` is resolved ON DEVICE
/// ([`resolve_child_num_data`]) from the resident split/role record instead of a host
/// scalar. This is the variant the SPEC-DRGL-05 deferral must use on hip so the deferred
/// tree stays byte-identical to the (parprefix) flag-OFF tree — a legacy twin would differ
/// by ~1 ULP (parprefix reorders the f64 reduction). Real-device only.
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_staged_parprefix_kernel_devcount(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    // DEVICE `num_data` source (SPEC-DRGL-12) — REPLACES the host `num_data: i32` scalar.
    ranges: &Array<i32>,
    roles: &Array<i32>,
    split_slot: u32,
    which: u32,
    parent_count: i32,
) {
    // Resolve this child's num_data ON DEVICE — the ONLY difference vs the host twin.
    let num_data = resolve_child_num_data(ranges, roles, split_slot, which, parent_count);

    let f = CUBE_POS_X;
    let fi = f as usize;
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
    let mut ct_g = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_h = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_c = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut lmin = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);

    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    let mut c = UNIT_POS as usize;
    while c < cells {
        sm[c] = hist[base + c];
        c += cd;
    }
    sync_cube();

    // PHASE 1: all-lanes parallel prefix, rev then fwd (reuse ct/lmin scratch).
    parprefix_store_rev(
        &sm.to_slice(),
        &mut rev_ag.to_slice_mut(),
        &mut rev_ah.to_slice_mut(),
        &mut rev_lc.to_slice_mut(),
        &mut rev_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
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
    sync_cube();
    parprefix_store_fwd(
        &sm.to_slice(),
        &mut fwd_ag.to_slice_mut(),
        &mut fwd_ah.to_slice_mut(),
        &mut fwd_lc.to_slice_mut(),
        &mut fwd_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
        offset[fi],
        default_bin[fi],
        skip_default_bin[fi],
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        sum_hessian,
        num_data,
        fwd_count[fi],
    );
    sync_cube();

    // PHASE 2: warp-split parallel gain scans (unchanged from pargain).
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

    // PHASE 3 + merge (unchanged from pargain).
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

/// FUSED-SUBTRACT twin of [`find_best_splits_fused_siblings_staged_kernel`] — folds
/// the subtraction trick INTO the co-pack sibling scan, removing the separate
/// `subtract_resident` launch (the spike095/096 host-enqueue lever). Sibling A
/// (smaller) reads `hist_smaller` and scans, IDENTICALLY to the base kernel.
/// Sibling B (larger) does NOT read a pre-subtracted buffer — it computes
/// `d = hist_parent[bin] − hist_smaller[bin]` in its cooperative staging, WRITES `d`
/// to `larger_out` (a FRESH buffer distinct from `hist_parent` — no read/write
/// aliasing) so the derived larger histogram is materialized for the next level's
/// subtraction, and stages `d` into LDS to scan. Every other stage (the REV/FWD
/// branch scans, the merge/finalize, the output layout) is BYTE-FOR-BYTE the base
/// kernel.
///
/// BIT-EXACTNESS: the subtraction is the SAME elementwise f64 `parent[i] −
/// smaller[i]` [`subtract_hist_kernel`] computes (non-negotiable #3: the derived
/// larger is NOT re-FixHistogram'd), so `larger_out` is bit-identical to
/// `subtract_resident`'s output AND the LDS values sibling B scans are bit-identical
/// to scanning that separately-subtracted buffer ⇒ the 12-cell scan output is
/// unchanged. STAGED (real-device) only, like its base kernel.
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_subtract_staged_kernel(
    hist_smaller: &Array<f64>,
    hist_parent: &Array<f64>,
    // The FRESH larger-child histogram (`parent − smaller`), written by the sibling-B
    // cubes during staging — the materialized derived larger for future subtraction.
    larger_out: &mut Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
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
    n_feats: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let is_b = CUBE_POS_Y != 0;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    if is_b {
        // FUSED SUBTRACT: stage `parent − smaller`, materialize it into `larger_out`
        // (fresh buffer, no aliasing with `hist_parent`), and stage into LDS. One
        // lane per `c` ⇒ each `larger_out[base+c]` written exactly once.
        let mut c = UNIT_POS as usize;
        while c < cells {
            let d = hist_parent[base + c] - hist_smaller[base + c];
            larger_out[base + c] = d;
            sm[c] = d;
            c += cd;
        }
    } else {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_smaller[base + c];
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
// OFFICIAL-SHAPE parallel scan ("official", `LGBM_SCAN_OFFICIAL`) — P1b. The
// staged/pargain/parprefix kernels all keep `SCAN_STAGED_CUBE_DIM=64` and run the
// per-candidate walk on 1-2 lanes (staged) or a 32-lane warp (pargain). Official
// LightGBM's `FindBestSplitsForLeafKernel<<<num_tasks, 256>>>` uses ONE THREAD PER
// BIN, a block prefix-sum across all 256 threads, then a block argmax — 4× the
// per-feature parallelism at num_bin=255 and ~25.6k active threads (100 blocks ×
// 256) vs the staged path's ~200 active lanes. This is the "clean full-shape
// rewrite" the P1b design (`docs/p1b-official-scan-kernel-design.md`) calls for; the
// hypothesis is that the 256-wide geometry wins where pargain/parprefix lost (both
// kept CUBE_DIM=64). Default OFF both backends until the Kaggle P100 A/B settles it.
//
// ALGORITHM (proven bit-exact on the cpu lane, `scan_pargain_parity.rs::official_
// branch` + `assert_official_parity`, committed `b30b014`): the serial `done`
// early-break recurrence is REMOVED and replaced with the STATELESS per-candidate
// guard `active && !cont && !brk` — equal to the serial considered-set because
// reverse `right_count` is non-decreasing ⇒ `cont` monotone true→false and `brk`
// monotone false→true, so `{consider} == {active && !cont && !brk}` with no
// recurrence (forward symmetric). NEAR side = the directly-accumulated prefix, FAR
// side = complement `total − acc`. Counts/threshold stay integer-exact under any add
// order; only g/h gains reorder (block prefix) → the documented ~1e-6 GPU envelope
// (same contract as pargain/parprefix; the cpu ANCHOR never runs this kernel).
// ============================================================================

/// Cross-plane carry width for the official block collectives — sized to the max
/// plane width (wave64) so plane-0's exclusive-scan writes (`stage[lane]`, lane <
/// plane_dim) never index out of bounds; the number of planes read back is only
/// `ceil(256/plane_dim) ≤ 8`. Mirrors `best_split::STAGE1_N_PLANES_MAX` (=32, valid
/// on the wave32/warp32 validated GPUs) but widened to 64 for robustness.
#[cfg(feature = "gpu")]
const OFFICIAL_SCAN_STAGE: usize = 64;

/// Official-shape block width: 256 threads, one lane per bin (num_bin ≤ 255 ⇒ no
/// striding of the scan itself; the histogram stage still strides `2*num_bin ≤ 512`
/// cells). Matches official LightGBM's `NUM_THREADS_PER_BLOCK` for the split finder.
#[cfg(feature = "gpu")]
const SCAN_OFFICIAL_CUBE_DIM: u32 = 256;

/// Two-level within-block f64 INCLUSIVE prefix-sum — the f64 twin of
/// `best_split::stage1_block_scan`. `plane_inclusive_sum` intra-plane, then plane 0
/// exclusive-scans the per-plane totals under a `SharedMemory` carry, each unit adds
/// its plane base back. Returns `UNIT_POS`'s block-wide inclusive prefix of `v` in
/// lane order. The plane reorder is why the gain is ~1e-6 (not bit-exact) — integer
/// counts summed through this stay exact (< 2^53).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::manual_div_ceil)] // matches best_split::stage1_block_scan (cubecl lowering)
fn block_inclusive_scan_f64(v: f64, plane_dim: u32) -> f64 {
    let mut stage = SharedMemory::<f64>::new(OFFICIAL_SCAN_STAGE);
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let local = plane_inclusive_sum(v);
    if lane == pd - 1 {
        stage[plane_id] = local;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { f64::new(0.0) };
        stage[lane] = plane_exclusive_sum(t);
    }
    sync_cube();
    let base = stage[plane_id];
    base + local
}

/// Block-wide f64 MAX, broadcast to EVERY lane (two-level `plane_max` + LDS carry +
/// a 1-cell result broadcast). `ident` pads the cross-plane lanes beyond `n_planes`;
/// callers pass a value ≤ every real input (0.0 is safe here — every input is a
/// non-negative gain or a 0/1 flag).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::manual_div_ceil)]
fn block_max_f64(v: f64, plane_dim: u32, ident: f64) -> f64 {
    let mut stage = SharedMemory::<f64>::new(OFFICIAL_SCAN_STAGE);
    let mut result = SharedMemory::<f64>::new(1usize);
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let pmax = plane_max(v);
    if lane == 0 {
        stage[plane_id] = pmax;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { ident };
        let total = plane_max(t);
        if lane == 0 {
            result[0] = total;
        }
    }
    sync_cube();
    result[0]
}

/// Block-wide u32 MIN, broadcast to EVERY lane (two-level `plane_min` + LDS carry).
/// `ident` (= the "no candidate" sentinel `u32::MAX`) pads the cross-plane lanes so
/// they never win. Used for the argmax tie-break: the lowest `UNIT_POS` among the
/// gain-tied winners (≡ the serial strict-`>` first-max, lowest k).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::manual_div_ceil)]
fn block_min_u32(v: u32, plane_dim: u32, ident: u32) -> u32 {
    let mut stage = SharedMemory::<u32>::new(OFFICIAL_SCAN_STAGE);
    let mut result = SharedMemory::<u32>::new(1usize);
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let pmin = plane_min(v);
    if lane == 0 {
        stage[plane_id] = pmin;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { ident };
        let total = plane_min(t);
        if lane == 0 {
            result[0] = total;
        }
    }
    sync_cube();
    result[0]
}

/// One official-shape branch (reverse `forward=0` or forward `forward=1`) over a
/// STAGED (LDS) histogram, ALL `CUBE_DIM` lanes cooperating. Lane `k = UNIT_POS`
/// owns candidate k (`k < count`; higher lanes inert). Writes the branch's 6-cell
/// state (`[is_splittable, best_gain, threshold, left_count, sum_left_gradient,
/// sum_left_hessian]`) that [`merge_finalize_staged`] consumes — the SAME state
/// layout the serial [`scan_rev_branch_staged`]/[`scan_fwd_branch_staged`] produce.
///
/// MUST be called by every lane uniformly (the block collectives sync internally);
/// do NOT wrap in a divergent `if UNIT_POS == …`. The NEAR side is always the
/// directly-accumulated prefix (`acc_*`), the FAR side its complement — for reverse
/// NEAR = right, for forward NEAR = left (the committed `official_branch`'s exact
/// arithmetic). Real-GPU only (block collectives + f64 LDS).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn official_branch_block(
    sm: &Slice<f64>,
    state: &mut SliceMut<f64>,
    forward: u32, // 0 = reverse, 1 = forward
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
    count: i32,
    plane_dim: u32,
) {
    let fwd = forward != 0;
    let use_l1_b = use_l1 != 0;
    let skip_def = skip_default_bin != 0;
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;
    let k = UNIT_POS as i32;
    let t_start = num_bin - 1 - offset;

    // ---- per-bin contribution (masked); higher lanes (k >= count) inert ----
    // reverse: t = t_start − k with an explicit in-range gate; forward: t = k, no
    // gate (fold via `|| fwd`). Verbatim from `official_branch`.
    let in_count = k < count;
    let t = select(fwd, k, t_start - k);
    let in_range = t >= (1 - offset);
    let in_range_eff = in_range || fwd;
    let skip = skip_def && (t + offset) == default_bin;
    let active = in_count && in_range_eff && !skip;
    let t_safe = select(t < 0, 0i32, t);
    let bi = (t_safe as usize) * 2;
    let g_raw = sm[bi];
    let h_raw = sm[bi + 1];
    let g = select(active, g_raw, 0.0);
    let h = select(active, h_raw, 0.0);
    let qc = select(active, round_int(h_raw * cnt_factor), 0i32);
    let qc_f = f64::cast_from(qc);

    // ---- block inclusive prefix-sums in lane order (near-side accumulation) ----
    let acc_g = block_inclusive_scan_f64(g, plane_dim);
    // kEpsilon seed added AFTER the scan (constant offset — matches the serial
    // `sum_*_hessian = kEpsilon` init; the plane reorder makes this ~1e-6 anyway).
    let acc_h = block_inclusive_scan_f64(h, plane_dim) + f64::cast_from(K_EPSILON);
    let acc_cnt_f = block_inclusive_scan_f64(qc_f, plane_dim);
    let acc_cnt = i32::cast_from(acc_cnt_f);

    // ---- near = acc, far = complement; left/right by branch direction ----
    let sum_left_gradient = select(fwd, acc_g, sum_gradient - acc_g);
    let sum_left_hessian = select(fwd, acc_h, sum_hessian - acc_h);
    let left_count = select(fwd, acc_cnt, num_data - acc_cnt);
    let sum_right_gradient = select(fwd, sum_gradient - acc_g, acc_g);
    let sum_right_hessian = select(fwd, sum_hessian - acc_h, acc_h);
    // NEAR (=acc) too small ⇒ `cont`; FAR (=complement) too small ⇒ `brk`. NEAR is
    // acc regardless of direction, so no branch needed here.
    let far_cnt = num_data - acc_cnt;
    let far_h = sum_hessian - acc_h;
    let cont = acc_cnt < min_data_in_leaf || acc_h < min_sum_hessian_in_leaf;
    let brk = far_cnt < min_data_in_leaf || far_h < min_sum_hessian_in_leaf;
    let consider = active && !cont && !brk;

    let current_gain = get_split_gains(
        use_l1_b,
        sum_left_gradient,
        sum_left_hessian,
        sum_right_gradient,
        sum_right_hessian,
        lambda_l1,
        lambda_l2,
    );
    let valid = consider && current_gain > min_gain_shift;
    let cand_gain = select(valid, current_gain, 0.0);
    let threshold = select(fwd, t + offset, t - 1 + offset);

    // ---- block argmax (gain desc, k asc) + is_splittable OR ----
    let flag = block_max_f64(select(valid, 1.0, 0.0), plane_dim, 0.0);
    let gmax = block_max_f64(cand_gain, plane_dim, 0.0);
    // `gmax` is a selected element value (max, not a reduction), so the winning
    // lane's `cand_gain == gmax` holds bit-for-bit; `gmax > 0` excludes the all-
    // invalid case (every valid gain > min_gain_shift ≥ 0 ⇒ cand_gain > 0).
    let eligible = valid && cand_gain == gmax && gmax > 0.0;
    let key = select(eligible, UNIT_POS, 4294967295u32);
    let winner_k = block_min_u32(key, plane_dim, 4294967295u32);

    // ---- lane 0 seeds the no-split state, then the winner writes its payload ----
    if UNIT_POS == 0 {
        state[0] = flag;
        state[1] = 0.0;
        state[2] = 0.0;
        state[3] = 0.0;
        state[4] = 0.0;
        state[5] = 0.0;
    }
    sync_cube();
    if UNIT_POS == winner_k {
        state[1] = cand_gain;
        state[2] = f64::cast_from(threshold);
        state[3] = f64::cast_from(left_count);
        state[4] = sum_left_gradient;
        state[5] = sum_left_hessian;
    }
    sync_cube();
}

/// OFFICIAL-SHAPE single-leaf kernel: ONE CUBE PER FEATURE, `SCAN_OFFICIAL_CUBE_DIM`
/// (256) lanes. Cooperatively stages the feature's histogram into LDS, then runs the
/// all-lanes [`official_branch_block`] REVERSE then FORWARD (reusing the LDS stage),
/// and lane 0 merges + finalizes into `out[f*12..]`. ~1e-6 vs
/// [`find_best_splits_fused_staged_kernel`] (block prefix reorders the f64 g/h sums;
/// counts/threshold/is_splittable are integer-exact). Real-GPU only.
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_staged_official_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    plane_dim: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    let mut c = UNIT_POS as usize;
    while c < cells {
        sm[c] = hist[base + c];
        c += cd;
    }
    sync_cube();

    official_branch_block(
        &sm.to_slice(),
        &mut state_rev.to_slice_mut(),
        0u32,
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
        plane_dim,
    );
    sync_cube();
    official_branch_block(
        &sm.to_slice(),
        &mut state_fwd.to_slice_mut(),
        1u32,
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
        fwd_count[fi],
        plane_dim,
    );
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

/// OFFICIAL-SHAPE co-pack sibling twin of [`find_best_splits_fused_staged_official_kernel`]
/// (the LIVE default arm the grow driver co-packs). `CUBE_POS_Y` selects the sibling
/// (A = smaller, B = larger); each writes its 12-cell window (A → `f`, B → `n_feats + f`).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_staged_official_kernel(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
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
    n_feats: u32,
    plane_dim: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let is_b = CUBE_POS_Y != 0;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

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

    official_branch_block(
        &sm.to_slice(),
        &mut state_rev.to_slice_mut(),
        0u32,
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
        plane_dim,
    );
    sync_cube();
    official_branch_block(
        &sm.to_slice(),
        &mut state_fwd.to_slice_mut(),
        1u32,
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
        fwd_count[fi],
        plane_dim,
    );
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

/// FUSED-SUBTRACT official twin of [`find_best_splits_fused_siblings_staged_official_kernel`]
/// — sibling B stages `d = parent − smaller` into `larger_out` (fresh buffer) AND
/// into LDS, folding the subtraction trick into the co-pack scan (the subtract-fuse
/// lever). Byte-identical `larger_out` to the staged subtract twin; the branch scans
/// are the official-shape block scans. Real-GPU only.
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_subtract_staged_official_kernel(
    hist_smaller: &Array<f64>,
    hist_parent: &Array<f64>,
    larger_out: &mut Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
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
    n_feats: u32,
    plane_dim: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let is_b = CUBE_POS_Y != 0;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    if is_b {
        let mut c = UNIT_POS as usize;
        while c < cells {
            let d = hist_parent[base + c] - hist_smaller[base + c];
            larger_out[base + c] = d;
            sm[c] = d;
            c += cd;
        }
    } else {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_smaller[base + c];
            c += cd;
        }
    }
    sync_cube();

    official_branch_block(
        &sm.to_slice(),
        &mut state_rev.to_slice_mut(),
        0u32,
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
        plane_dim,
    );
    sync_cube();
    official_branch_block(
        &sm.to_slice(),
        &mut state_fwd.to_slice_mut(),
        1u32,
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
        fwd_count[fi],
        plane_dim,
    );
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

/// DEVICE-`num_data` twin of
/// [`find_best_splits_fused_siblings_subtract_staged_official_kernel`] (§12
/// final-mile lever 1 — the single-sync deferral needs the LIVE cuda scan
/// variant, which is the OFFICIAL shape, to resolve both children's counts on
/// device; the legacy/parprefix devcount twins would scan with a DIFFERENT
/// summation order and could flip 1-ULP winners vs the eager arm). Body is
/// verbatim; only the two host `num_data_{a,b}` scalars are replaced by the
/// shared [`resolve_child_num_data`] read of the resident `ranges`/`roles`
/// record (same field layout as the T-04 fixed-grid build).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_subtract_staged_official_kernel_devcount(
    hist_smaller: &Array<f64>,
    hist_parent: &Array<f64>,
    larger_out: &mut Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    // LEFT/RIGHT per-branch scalars (host-known from the pick export even under the
    // read_split deferral); the kernel selects the SMALLER child's set for branch A and
    // the LARGER's for branch B via the resident role record — the exact values the
    // eager arm passes as `_a`/`_b`, so the scan is bit-identical.
    min_gain_shift_left: f64,
    sum_gradient_left: f64,
    sum_hessian_left: f64,
    min_gain_shift_right: f64,
    sum_gradient_right: f64,
    sum_hessian_right: f64,
    ranges: &Array<i32>,
    roles: &Array<i32>,
    split_slot: u32,
    parent_count: i32,
    n_feats: u32,
    plane_dim: u32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let is_b = CUBE_POS_Y != 0;
    let mut sm = SharedMemory::<f64>::new(SCAN_STAGE_MAX_CELLS);
    let mut state_rev = SharedMemory::<f64>::new(8usize);
    let mut state_fwd = SharedMemory::<f64>::new(8usize);

    // A = smaller child, B = larger child (the hist buffers are physically
    // smaller/derived-larger); counts via the shared resolver (which=2/3), scalars
    // via the same role bit.
    let num_data_a = resolve_child_num_data(ranges, roles, split_slot, 2u32, parent_count);
    let num_data_b = resolve_child_num_data(ranges, roles, split_slot, 3u32, parent_count);
    let smaller_is_left = roles[(split_slot * 3) as usize] != 0;

    let mgs_a = select(smaller_is_left, min_gain_shift_left, min_gain_shift_right);
    let sg_a = select(smaller_is_left, sum_gradient_left, sum_gradient_right);
    let sh_a = select(smaller_is_left, sum_hessian_left, sum_hessian_right);
    let mgs_b = select(smaller_is_left, min_gain_shift_right, min_gain_shift_left);
    let sg_b = select(smaller_is_left, sum_gradient_right, sum_gradient_left);
    let sh_b = select(smaller_is_left, sum_hessian_right, sum_hessian_left);

    let min_gain_shift = select(is_b, mgs_b, mgs_a);
    let sum_gradient = select(is_b, sg_b, sg_a);
    let sum_hessian = select(is_b, sh_b, sh_a);
    let num_data = select(is_b, num_data_b, num_data_a);

    let base = slot_off[fi] as usize;
    let cells = (u32::cast_from(num_bin[fi]) as usize) * 2;
    let cd = CUBE_DIM as usize;
    if is_b {
        let mut c = UNIT_POS as usize;
        while c < cells {
            let d = hist_parent[base + c] - hist_smaller[base + c];
            larger_out[base + c] = d;
            sm[c] = d;
            c += cd;
        }
    } else {
        let mut c = UNIT_POS as usize;
        while c < cells {
            sm[c] = hist_smaller[base + c];
            c += cd;
        }
    }
    sync_cube();

    official_branch_block(
        &sm.to_slice(),
        &mut state_rev.to_slice_mut(),
        0u32,
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
        plane_dim,
    );
    sync_cube();
    official_branch_block(
        &sm.to_slice(),
        &mut state_fwd.to_slice_mut(),
        1u32,
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
        fwd_count[fi],
        plane_dim,
    );
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

/// Pargain gate — BACKEND-AWARE default (env `LGBM_SCAN_PARGAIN=1/0` overrides
/// either way). DEFAULT ON for the ROCm/AMD runtime (`"hip"`), OFF for CUDA/NVIDIA
/// (`"cuda"`) and everything else. Only meaningful where the staged gates already
/// hold (real device, ≤256-bin features) — the launch helpers consult it INSIDE the
/// staged arm, so legacy/staged behavior is byte-unchanged on the cpu anchor.
///
/// The two backends respond with OPPOSITE SIGN to pargain (parallel per-candidate
/// gain + parallel argmax), so the default is per-backend:
/// - **ROCm/AMD (default ON):** measured 1.51× scan win on gfx1152 (2026-07-13,
///   100k×50 real-GPU drain: default staged scan 2545 ms → pargain 1269 ms; wall
///   6136 → 4065 ms), tree BIT-EXACT to the u64 integer path + within the 500k f64
///   envelope on real ROCm. AMD's weaker f64 makes the serial per-candidate divides
///   expensive, so parallelizing them across 32 lanes/branch dominates the barrier
///   cost. ROCm is the project's primary validated GPU target (CLAUDE.md).
/// - **CUDA/NVIDIA (default OFF):** NET-NEGATIVE on P100 (spike094, order-alternated
///   warm-median of 3, preds BIT-IDENTICAL max_abs 0.0, counts scan_pargain=2980):
///   pargain 8.89s vs staged 8.73s (0.98×), drained scan 2.15→2.37s. P100's strong
///   1:2 f64 makes the serial divides cheap, so the phase-split's extra LDS traffic +
///   barriers cost more than the parallel gains save.
///
/// `LGBM_SCAN_PARGAIN=1` forces ON (e.g. a consumer 1:32-f64 CUDA card where AMD's
/// calculus applies); `=0` forces OFF (e.g. same-session A/B on ROCm). Bit-exact
/// either way — the hatch prices the wall delta, it does not gate correctness.
#[cfg(feature = "gpu")]
fn scan_pargain_enabled(runtime_name: &str) -> bool {
    match std::env::var("LGBM_SCAN_PARGAIN").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        // Default: ON for ROCm/AMD (the 1.51× win), OFF for CUDA/NVIDIA + cpu.
        _ => runtime_name == "hip",
    }
}

/// POSITIVE tripwire — bumped once per staged-scan launch that dispatched the
/// PARGAIN kernel (the bench-protocol counts proof: never trust a wall delta
/// without ledger confirmation the code-under-test ran). Folded into the
/// `phase_prof` COUNTS line as `scan_pargain=`. Unconditional (a Relaxed
/// increment per LAUNCH, not per candidate — timing-neutral at ~1/split).
#[cfg(feature = "gpu")]
pub static SCAN_PARGAIN_CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Swap the pargain-launch tripwire to zero and return the prior value
/// (consumed by `phase_prof::dump`). Present on every build (0 without `gpu`)
/// so the dump site needs no cfg.
pub fn scan_pargain_count_take() -> u64 {
    #[cfg(feature = "gpu")]
    {
        SCAN_PARGAIN_CNT.swap(0, std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}

/// PARALLEL-PREFIX scan gate (env `LGBM_SCAN_PARPREFIX=1`, ROCm-only, opt-in). It
/// REPLACES pargain's serial single-lane phase-1 accumulate with an all-lanes
/// chunked block-scan (the scan is 84% O(num_bin)-bound on gfx1152 — the serial
/// accumulate is the residual after pargain parallelized the gain eval). Reorders
/// the f64 prefix adds ⇒ ~1e-6 vs the bit-exact serial (counts/break are integer-
/// exact); allowed on the GPU ~1e-6 gate. Default OFF until the measured drain
/// scan-bucket win flips it (like pargain's rollout). CUDA/cpu unaffected.
#[cfg(feature = "gpu")]
fn scan_parprefix_enabled(runtime_name: &str) -> bool {
    match std::env::var("LGBM_SCAN_PARPREFIX").as_deref() {
        // `=1` forces ON on either GPU backend (A/B — also valid on a consumer 1:32-f64
        // CUDA card where AMD's calculus applies); `=0` forces OFF (same-session A/B).
        Ok("1") => runtime_name == "hip" || runtime_name == "cuda",
        Ok("0") => false,
        // Default: ON for ROCm/AMD — the biggest device-compute lever on the primary
        // validated GPU (gfx1152: scan 23.6→17.8 ms/tree = 1.34×, wall 64.0→57.0 = 1.11×;
        // splits bit-equal to the legacy kernel, gain/tree within the ~1e-6 GPU envelope).
        // parprefix PRECEDES pargain in the launcher, so it supersedes round-11's
        // pargain-on-hip default. OFF for CUDA/NVIDIA (spike104: NET-NEGATIVE on P100 —
        // cheap 1:2 f64 + occupancy starvation make the parallel-scan barriers cost more
        // than the serial accumulate they remove) + cpu (the bit-exact anchor never runs it).
        _ => runtime_name == "hip",
    }
}

/// POSITIVE tripwire — bumped once per staged-scan launch that dispatched the
/// PARPREFIX kernel (bench-protocol counts proof). Folded into `phase_prof` as
/// `scan_parprefix=`.
#[cfg(feature = "gpu")]
pub static SCAN_PARPREFIX_CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Swap the parprefix-launch tripwire to zero and return the prior value.
pub fn scan_parprefix_count_take() -> u64 {
    #[cfg(feature = "gpu")]
    {
        SCAN_PARPREFIX_CNT.swap(0, std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}

/// OFFICIAL-SHAPE scan gate (`LGBM_SCAN_OFFICIAL`, P1b) — the 256-wide,
/// one-lane-per-bin block-prefix-sum + block-argmax scan
/// ([`find_best_splits_fused_staged_official_kernel`] et al.). Backend-aware default
/// (like pargain/parprefix), set by the measured Kaggle P100 A/B
/// (`lgb-rs-p1b-official-scan`, 2026-07-15):
/// - **CUDA/NVIDIA (default ON):** 256-wide wins where pargain/parprefix (CUBE_DIM=64)
///   lost — base 4.604s → official 4.233s = **1.0876× (−371ms)**, preds BIT-IDENTICAL
///   (max_abs 0.0, no split flipped on the bench corpus despite the ~1e-6 internal
///   gain reorder), counts scan_official=2980, CUDA parity gate green; nsys shows the
///   scan kernel drop **131 µs → 16–20 µs/launch** (the subtract-fuse twin 20.1 µs
///   ×2385, single-scan 16.1 µs), approaching official LightGBM's ~7.5 µs class.
/// - **ROCm/AMD (default OFF):** parprefix is already the proven hip scan winner
///   ([[cudagraph-campaign]] 1.33× scan); a gfx1151 drain (2026-07-15) put official at
///   714 ms/scan vs parprefix 691 ms (~3% slower, within box noise) — no win, so hip
///   keeps parprefix. Official stays available on hip via `LGBM_SCAN_OFFICIAL=1`.
///
/// `=1` forces ON on either GPU backend; `=0` forces OFF (reverts the CUDA default —
/// e.g. to restore the bit-exact-by-construction staged scan). The cpu anchor +
/// planeless devices never run it (so the f64 merge gate + bit-exact anchor tests are
/// untouched). When enabled it PRECEDES parprefix/pargain in the staged launchers +
/// the subtract-fuse launcher. Takes the client (not just the runtime name) so the
/// guard can require plane support and the launch can source `plane_dim` from
/// `hardware.plane_size_max` — the same block-collective plumbing P1a's
/// [`reduce_par_enabled`] uses.
#[cfg(feature = "gpu")]
pub fn scan_official_enabled<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> bool {
    let name = <R as cubecl::Runtime>::name(client);
    if name == "cpu" {
        return false;
    }
    if client.properties().hardware.plane_size_max < 2 {
        return false;
    }
    match SCAN_OFFICIAL_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    match std::env::var("LGBM_SCAN_OFFICIAL").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        // Default: ON for CUDA (the measured 1.0876× P100 win), OFF for hip
        // (parprefix wins there) + everything else.
        _ => name == "cuda",
    }
}

/// Same-session A/B override for [`scan_official_enabled`]. 0 = unset (env decides),
/// 1 = force ON, 2 = force OFF — mirrors [`REDUCE_PAR_OVERRIDE`] so a Kaggle spike can
/// alternate arms in one process without re-launching.
#[cfg(feature = "gpu")]
static SCAN_OFFICIAL_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Set/clear the in-process [`scan_official_enabled`] override (`Some(true)` = ON,
/// `Some(false)` = OFF, `None` = defer to env).
#[cfg(feature = "gpu")]
pub fn set_scan_official_override(v: Option<bool>) {
    let code = match v {
        None => 0u8,
        Some(true) => 1,
        Some(false) => 2,
    };
    SCAN_OFFICIAL_OVERRIDE.store(code, std::sync::atomic::Ordering::Relaxed);
}

/// POSITIVE tripwire — bumped once per staged-scan launch that dispatched the
/// OFFICIAL kernel (bench-protocol counts proof). Folded into `phase_prof` as
/// `scan_official=`.
#[cfg(feature = "gpu")]
pub static SCAN_OFFICIAL_CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Swap the official-launch tripwire to zero and return the prior value.
pub fn scan_official_count_take() -> u64 {
    #[cfg(feature = "gpu")]
    {
        SCAN_OFFICIAL_CNT.swap(0, std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}

/// POSITIVE tripwire (SPEC-DRGL-12) — bumped once per fused-scan launch that sourced
/// `num_data` ON DEVICE (the `Device` [`NumDataSrc`] path: parprefix twin or legacy
/// twin). Positive proof the device-`num_data` scan actually ran, for the bench-protocol
/// counts ledger; folded into `phase_prof` as `scan_numdata_dev=`.
#[cfg(feature = "gpu")]
pub static SCAN_NUMDATA_DEV_CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Swap the device-`num_data` scan tripwire to zero and return the prior value.
pub fn scan_numdata_dev_count_take() -> u64 {
    #[cfg(feature = "gpu")]
    {
        SCAN_NUMDATA_DEV_CNT.swap(0, std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}

// ===================== per-grow scan-descriptor hoist =====================
//
// Every fused-scan launch used to re-upload SEVEN per-feature descriptor arrays
// (slot_off / num_bin / offset / default_bin / skip_default_bin / rev_count /
// fwd_count) + the reduce's `rf` real-feature tie-break array — ALL derived from
// the per-grow-CONSTANT `feats` / `real_feats` — via per-launch
// `create_from_slice` calls (~8 small H2D uploads × ~3600 scans/train). The hoist
// uploads them ONCE per grow ([`ScanDescHandles`], cached by the GpuBackend and
// invalidated in `reset_resident_pool`) and passes the cached handles to the SAME
// kernels: identical bytes, identical launch geometry ⇒ bit-exact by construction.
// The per-launch path stays byte-unchanged for callers with no cache (tests, a
// geometry mismatch, hatch OFF).

/// Test/in-process override for the desc-hoist gate: `0` unset (env decides),
/// `1` forced ON, `-1` forced OFF. Mirrors `set_partition_fuse_bc_smem_override`.
static DESC_HOIST_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(0);

/// Force the desc-hoist gate for in-process A/B (`Some(true/false)`) or restore
/// env control (`None`). Test-only seam; production reads the env.
pub fn set_desc_hoist_override(v: Option<bool>) {
    let code = match v {
        None => 0,
        Some(true) => 1,
        Some(false) => -1,
    };
    DESC_HOIST_OVERRIDE.store(code, std::sync::atomic::Ordering::Relaxed);
}

/// Desc-hoist gate (env `LGBM_DESC_HOIST`, DEFAULT ON — validated 1.055× on real
/// CUDA, spike101: hoist 7.549s vs base 7.961s warm-median, preds BIT-IDENTICAL
/// max_abs 0.0, counts proof desc_hoist=5980 vs 0, drained build 1358→1050ms /
/// scan 1992→1899ms; `"0"` restores the per-launch uploads for A/B/rollback).
/// Read FRESH per call (mirrors `scan_staged_enabled`) so one process can A/B it;
/// the override wins over the env. Consumed by the `gpu`-gated GpuBackend caches
/// (the cpu build has no per-grow descriptor cache).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn desc_hoist_enabled() -> bool {
    match DESC_HOIST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        -1 => false,
        _ => !matches!(std::env::var("LGBM_DESC_HOIST").as_deref(), Ok("0")),
    }
}

/// POSITIVE tripwire — bumped once per scan launch that consumed the CACHED
/// descriptor set (the bench-protocol counts proof). Folded into the `phase_prof`
/// COUNTS line as `desc_hoist=`.
pub static SCAN_DESC_CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Swap the desc-hoist tripwire to zero and return the prior value (consumed by
/// `phase_prof::dump`).
pub fn scan_desc_count_take() -> u64 {
    SCAN_DESC_CNT.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// The per-grow device-resident scan descriptor set: the 7 per-feature arrays every
/// fused-scan kernel consumes + (when built with `real_feats`) the reduce kernels'
/// `rf` tie-break array. `Handle`s are ref-counted — `Clone` is cheap. Geometry
/// (`n`, `buf_len`) is carried so a consumer can verify the cache matches its call
/// before trusting it (mismatch ⇒ per-launch fallback, byte-unchanged).
#[derive(Clone)]
pub struct ScanDescHandles {
    n: usize,
    buf_len: usize,
    /// Every feature fits the staged LDS cap (`2*num_bin <= SCAN_STAGE_MAX_CELLS`) —
    /// precomputed so the staged-branch gate needs no per-launch re-scan of `feats`.
    /// Read only by the `gpu`-gated staged branches (the cpu build has no staged path).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    staged_capable: bool,
    h_slot: cubecl::server::Handle,
    h_numbin: cubecl::server::Handle,
    h_offset: cubecl::server::Handle,
    h_defbin: cubecl::server::Handle,
    h_skip: cubecl::server::Handle,
    h_rev: cubecl::server::Handle,
    h_fwd: cubecl::server::Handle,
    h_rf: Option<cubecl::server::Handle>,
}

impl ScanDescHandles {
    /// Whether this cache was built for exactly this call's geometry.
    #[must_use]
    pub fn matches(&self, n: usize, buf_len: usize) -> bool {
        self.n == n && self.buf_len == buf_len
    }
}

/// Build + upload the per-grow scan descriptor set: the SAME per-feature V5
/// validation loop + the SAME array assembly the per-launch path runs (byte-identical
/// arrays), uploaded once. `ctx` prefixes the two context-dependent error messages so
/// a cache-time validation failure reads identically to the per-launch failure it
/// replaces. `real_feats` (when given) must be the fpos-ordered real-feature-index
/// vector with exactly `feats.len()` entries — it becomes the reduce kernels' `rf`.
///
/// # Errors
/// The SAME per-feature V5 errors as the per-launch assembly (`na_as_missing`,
/// `num_bin == 0`, region overflow, region beyond `buf_len`).
pub fn upload_scan_desc<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    feats: &[BatchedSplitFeature],
    real_feats: Option<&[i32]>,
    buf_len: usize,
    ctx: &str,
) -> Result<ScanDescHandles, ComputeError> {
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
                detail: format!("{ctx}: slot_off + region overflows"),
            })?;
        if end > buf_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: buf_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push((num_bin_i - 1).max(0));
        fwd_count_a.push(if f.run_forward { (num_bin_i - 1 - f.offset).max(0) } else { 0 });
    }
    #[cfg(feature = "gpu")]
    let staged_capable =
        feats.iter().all(|f| (f.num_bin as usize) * 2 <= SCAN_STAGE_MAX_CELLS);
    #[cfg(not(feature = "gpu"))]
    let staged_capable = false;
    let h_rf = real_feats.map(|rfs| {
        let rf: Vec<f64> = rfs.iter().take(n).map(|&r| f64::from(r)).collect();
        client.create_from_slice(f64::as_bytes(&rf))
    });
    Ok(ScanDescHandles {
        n,
        buf_len,
        staged_capable,
        h_slot: client.create_from_slice(u32::as_bytes(&slot_off_a)),
        h_numbin: client.create_from_slice(i32::as_bytes(&num_bin_a)),
        h_offset: client.create_from_slice(i32::as_bytes(&offset_a)),
        h_defbin: client.create_from_slice(i32::as_bytes(&default_bin_a)),
        h_skip: client.create_from_slice(u32::as_bytes(&skip_default_bin_a)),
        h_rev: client.create_from_slice(i32::as_bytes(&rev_count_a)),
        h_fwd: client.create_from_slice(i32::as_bytes(&fwd_count_a)),
        h_rf,
    })
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

// ============================================================================
// PARALLEL-PREFIX phase 1 (LGBM_SCAN_PARPREFIX) — the all-lanes replacement for
// pargain's single-lane serial accumulate. Each of CUBE_DIM lanes owns a
// contiguous CHUNK of the branch's candidates, serially prefixes its chunk, then
// lane 0 exclusive-scans the 64 chunk totals; every lane adds its base to get the
// GLOBAL prefix (this reorders the f64 adds at chunk boundaries — the accepted
// ~1e-6 residue; counts are integer-exact). The `done`/break recurrence is
// reformulated as a break-POINT `tstar` = the first candidate meeting the break
// gate (min-reduced across lanes); `consider = k<tstar && mask && !cont`. The
// emitted (ag, ah, lc, ok) drive the UNCHANGED phase 2/3. Validated logically by
// `parprefix_*` in tests/scan_pargain_parity.rs (bit-equal with serial prefix).
// ============================================================================

/// PARALLEL-PREFIX phase 1, FORWARD branch. All CUBE_DIM lanes cooperate. Scratch:
/// `ct_g/ct_h/ct_c` (>= CUBE_DIM cells) hold per-lane chunk totals then exclusive
/// bases; `lmin` (>= CUBE_DIM+1) holds per-lane break minima then `tstar` at
/// `[CUBE_DIM]`. Emits LEFT-accumulated (ag, ah) + left_count (lc) + ok.
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn parprefix_store_fwd(
    sm: &Slice<f64>,
    cand_ag: &mut SliceMut<f64>,
    cand_ah: &mut SliceMut<f64>,
    cand_lc: &mut SliceMut<f64>,
    cand_ok: &mut SliceMut<f64>,
    ct_g: &mut SliceMut<f64>,
    ct_h: &mut SliceMut<f64>,
    ct_c: &mut SliceMut<f64>,
    lmin: &mut SliceMut<f64>,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    sum_hessian: f64,
    num_data: i32,
    fwd_count: i32,
) {
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;
    let skip_def = skip_default_bin != 0;
    let cd = CUBE_DIM as i32;
    let lane = UNIT_POS;
    let chunk = (fwd_count + cd - 1) / cd;
    let lo = lane as i32 * chunk;
    let hi_raw = lo + chunk;
    let hi = select(hi_raw < fwd_count, hi_raw, fwd_count);

    // Step A: serial masked prefix over this lane's chunk; store LOCAL prefix.
    let mut cg = f64::new(0.0);
    let mut ch = f64::new(0.0);
    let mut cc = 0i32;
    let mut k = lo;
    while k < hi {
        let skip = skip_def && (k + offset) == default_bin;
        let bi = (k as usize) * 2;
        cg += select(skip, f64::new(0.0), sm[bi]);
        ch += select(skip, f64::new(0.0), sm[bi + 1]);
        cc += select(skip, 0i32, round_int(sm[bi + 1] * cnt_factor));
        let ku = k as usize;
        cand_ag[ku] = cg;
        cand_ah[ku] = ch;
        cand_lc[ku] = f64::cast_from(cc);
        k += 1;
    }
    ct_g[lane as usize] = cg;
    ct_h[lane as usize] = ch;
    ct_c[lane as usize] = f64::cast_from(cc);
    sync_cube();

    // Step B: lane 0 exclusive-scans the chunk totals into ct_* (base per lane).
    if lane == 0 {
        let mut ag = f64::new(0.0);
        let mut ah = f64::new(0.0);
        let mut ac = f64::new(0.0);
        let mut l = 0i32;
        while l < cd {
            let lu = l as usize;
            let tg = ct_g[lu];
            let th = ct_h[lu];
            let tc = ct_c[lu];
            ct_g[lu] = ag;
            ct_h[lu] = ah;
            ct_c[lu] = ac;
            ag += tg;
            ah += th;
            ac += tc;
            l += 1;
        }
    }
    sync_cube();

    // Step C: add base (+kEpsilon on h) ⇒ GLOBAL prefix; compute per-lane break min.
    let base_g = ct_g[lane as usize];
    let base_h = ct_h[lane as usize];
    let base_c = ct_c[lane as usize];
    let eps = f64::cast_from(K_EPSILON);
    let mut my_break = f64::cast_from(fwd_count);
    let mut k2 = lo;
    while k2 < hi {
        let ku = k2 as usize;
        let gg = cand_ag[ku] + base_g;
        let hh = cand_ah[ku] + base_h + eps;
        let lcnt = i32::cast_from(cand_lc[ku] + base_c);
        cand_ag[ku] = gg;
        cand_ah[ku] = hh;
        cand_lc[ku] = f64::cast_from(lcnt);
        let skip = skip_def && (k2 + offset) == default_bin;
        let right_count = num_data - lcnt;
        let sum_right_hessian = sum_hessian - hh;
        let cont = lcnt < min_data_in_leaf || hh < min_sum_hessian_in_leaf;
        let brk = right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf;
        let is_break = !skip && !cont && brk;
        let kf = f64::cast_from(k2);
        my_break = select(is_break && kf < my_break, kf, my_break);
        k2 += 1;
    }
    lmin[lane as usize] = my_break;
    sync_cube();

    // Step D: lane 0 min-reduces the per-lane breaks ⇒ tstar at lmin[cd].
    if lane == 0 {
        let mut ts = f64::cast_from(fwd_count);
        let mut l = 0i32;
        while l < cd {
            let v = lmin[l as usize];
            ts = select(v < ts, v, ts);
            l += 1;
        }
        lmin[cd as usize] = ts;
    }
    sync_cube();
    let tstar = lmin[cd as usize];

    // Step E: emit consider flags.
    let mut k3 = lo;
    while k3 < hi {
        let ku = k3 as usize;
        let skip = skip_def && (k3 + offset) == default_bin;
        let hh = cand_ah[ku];
        let lcnt = i32::cast_from(cand_lc[ku]);
        let cont = lcnt < min_data_in_leaf || hh < min_sum_hessian_in_leaf;
        let consider = f64::cast_from(k3) < tstar && !skip && !cont;
        cand_ok[ku] = select(consider, f64::new(1.0), f64::new(0.0));
        k3 += 1;
    }
}

/// PARALLEL-PREFIX phase 1, REVERSE branch. Candidate k ↦ bin t = num_bin−1−offset−k;
/// accumulates the RIGHT side; emits (ag=sum_right_g, ah=sum_right_h, lc=left_count).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn parprefix_store_rev(
    sm: &Slice<f64>,
    cand_ag: &mut SliceMut<f64>,
    cand_ah: &mut SliceMut<f64>,
    cand_lc: &mut SliceMut<f64>,
    cand_ok: &mut SliceMut<f64>,
    ct_g: &mut SliceMut<f64>,
    ct_h: &mut SliceMut<f64>,
    ct_c: &mut SliceMut<f64>,
    lmin: &mut SliceMut<f64>,
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    sum_hessian: f64,
    num_data: i32,
    rev_count: i32,
) {
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;
    let skip_def = skip_default_bin != 0;
    let cd = CUBE_DIM as i32;
    let lane = UNIT_POS;
    let t_start = num_bin - 1 - offset;
    let chunk = (rev_count + cd - 1) / cd;
    let lo = lane as i32 * chunk;
    let hi_raw = lo + chunk;
    let hi = select(hi_raw < rev_count, hi_raw, rev_count);

    // Step A: serial masked RIGHT-side prefix over this lane's chunk.
    let mut cg = f64::new(0.0);
    let mut ch = f64::new(0.0);
    let mut cc = 0i32;
    let mut k = lo;
    while k < hi {
        let t = t_start - k;
        let in_range = t >= (1 - offset);
        let skip = skip_def && (t + offset) == default_bin;
        let active = in_range && !skip;
        let t_safe = select(t < 0, 0i32, t);
        let bi = (t_safe as usize) * 2;
        cg += select(active, sm[bi], f64::new(0.0));
        ch += select(active, sm[bi + 1], f64::new(0.0));
        cc += select(active, round_int(sm[bi + 1] * cnt_factor), 0i32);
        let ku = k as usize;
        cand_ag[ku] = cg;
        cand_ah[ku] = ch;
        cand_lc[ku] = f64::cast_from(cc);
        k += 1;
    }
    ct_g[lane as usize] = cg;
    ct_h[lane as usize] = ch;
    ct_c[lane as usize] = f64::cast_from(cc);
    sync_cube();

    if lane == 0 {
        let mut ag = f64::new(0.0);
        let mut ah = f64::new(0.0);
        let mut ac = f64::new(0.0);
        let mut l = 0i32;
        while l < cd {
            let lu = l as usize;
            let tg = ct_g[lu];
            let th = ct_h[lu];
            let tc = ct_c[lu];
            ct_g[lu] = ag;
            ct_h[lu] = ah;
            ct_c[lu] = ac;
            ag += tg;
            ah += th;
            ac += tc;
            l += 1;
        }
    }
    sync_cube();

    // Step C: GLOBAL right-side prefix; left_count = num_data − right_count.
    let base_g = ct_g[lane as usize];
    let base_h = ct_h[lane as usize];
    let base_c = ct_c[lane as usize];
    let eps = f64::cast_from(K_EPSILON);
    let mut my_break = f64::cast_from(rev_count);
    let mut k2 = lo;
    while k2 < hi {
        let ku = k2 as usize;
        let gg = cand_ag[ku] + base_g;
        let hh = cand_ah[ku] + base_h + eps;
        let right_count = i32::cast_from(cand_lc[ku] + base_c);
        cand_ag[ku] = gg;
        cand_ah[ku] = hh;
        cand_lc[ku] = f64::cast_from(num_data - right_count);
        let t = t_start - k2;
        let in_range = t >= (1 - offset);
        let skip = skip_def && (t + offset) == default_bin;
        let left_count = num_data - right_count;
        let sum_left_hessian = sum_hessian - hh;
        let cont = right_count < min_data_in_leaf || hh < min_sum_hessian_in_leaf;
        let brk = left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf;
        let is_break = in_range && !skip && !cont && brk;
        let kf = f64::cast_from(k2);
        my_break = select(is_break && kf < my_break, kf, my_break);
        k2 += 1;
    }
    lmin[lane as usize] = my_break;
    sync_cube();

    if lane == 0 {
        let mut ts = f64::cast_from(rev_count);
        let mut l = 0i32;
        while l < cd {
            let v = lmin[l as usize];
            ts = select(v < ts, v, ts);
            l += 1;
        }
        lmin[cd as usize] = ts;
    }
    sync_cube();
    let tstar = lmin[cd as usize];

    let mut k3 = lo;
    while k3 < hi {
        let ku = k3 as usize;
        let t = t_start - k3;
        let in_range = t >= (1 - offset);
        let skip = skip_def && (t + offset) == default_bin;
        // right_count = num_data − left_count (lc currently holds left_count).
        let right_count = num_data - i32::cast_from(cand_lc[ku]);
        let hh = cand_ah[ku];
        let cont = right_count < min_data_in_leaf || hh < min_sum_hessian_in_leaf;
        let consider = f64::cast_from(k3) < tstar && in_range && !skip && !cont;
        cand_ok[ku] = select(consider, f64::new(1.0), f64::new(0.0));
        k3 += 1;
    }
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

/// PARALLEL-PREFIX co-packed sibling kernel: the pargain siblings twin with phase 1
/// replaced by the all-lanes [`parprefix_store_rev`]/[`parprefix_store_fwd`]. This is
/// the CO-PACK path the live grow driver uses, so it carries the measurable win.
/// ROCm-only (gated by `scan_parprefix_enabled`).
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_staged_parprefix_kernel(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
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
    let mut ct_g = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_h = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_c = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut lmin = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);

    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

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

    // PHASE 1: all-lanes parallel prefix, rev then fwd.
    parprefix_store_rev(
        &sm.to_slice(),
        &mut rev_ag.to_slice_mut(),
        &mut rev_ah.to_slice_mut(),
        &mut rev_lc.to_slice_mut(),
        &mut rev_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
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
    sync_cube();
    parprefix_store_fwd(
        &sm.to_slice(),
        &mut fwd_ag.to_slice_mut(),
        &mut fwd_ah.to_slice_mut(),
        &mut fwd_lc.to_slice_mut(),
        &mut fwd_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
        offset[fi],
        default_bin[fi],
        skip_default_bin[fi],
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        sum_hessian,
        num_data,
        fwd_count[fi],
    );
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

/// SPEC-DRGL-13: device-`num_data` twin of
/// [`find_best_splits_fused_siblings_staged_parprefix_kernel`] (the LIVE default hip co-pack
/// scan). BYTE-FOR-BYTE the same parallel-prefix co-pack scan — the ONLY change is that BOTH
/// siblings' `num_data` are resolved ON DEVICE ([`resolve_child_num_data`]: A = smaller, B =
/// larger) from the resident split/role record instead of host scalars. This is the variant
/// SPEC-DRGL-05's deferral must use on hip so the deferred co-pack fold stays byte-identical
/// to the (parprefix) flag-OFF fold. Real-device only.
#[cfg(feature = "gpu")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_staged_parprefix_kernel_devcount(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift_a: f64,
    sum_gradient_a: f64,
    sum_hessian_a: f64,
    min_gain_shift_b: f64,
    sum_gradient_b: f64,
    sum_hessian_b: f64,
    // DEVICE `num_data` source (SPEC-DRGL-13) — REPLACES host num_data_a/num_data_b.
    ranges: &Array<i32>,
    roles: &Array<i32>,
    split_slot: u32,
    which_a: u32,
    which_b: u32,
    parent_count: i32,
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
    let mut ct_g = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_h = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut ct_c = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);
    let mut lmin = SharedMemory::<f64>::new(PARGAIN_MAX_CAND);

    // Resolve BOTH children's num_data ON DEVICE per the caller's which_a/which_b.
    let num_data_a = resolve_child_num_data(ranges, roles, split_slot, which_a, parent_count);
    let num_data_b = resolve_child_num_data(ranges, roles, split_slot, which_b, parent_count);

    let min_gain_shift = select(is_b, min_gain_shift_b, min_gain_shift_a);
    let sum_gradient = select(is_b, sum_gradient_b, sum_gradient_a);
    let sum_hessian = select(is_b, sum_hessian_b, sum_hessian_a);
    let num_data = select(is_b, num_data_b, num_data_a);

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

    // PHASE 1: all-lanes parallel prefix, rev then fwd.
    parprefix_store_rev(
        &sm.to_slice(),
        &mut rev_ag.to_slice_mut(),
        &mut rev_ah.to_slice_mut(),
        &mut rev_lc.to_slice_mut(),
        &mut rev_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
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
    sync_cube();
    parprefix_store_fwd(
        &sm.to_slice(),
        &mut fwd_ag.to_slice_mut(),
        &mut fwd_ah.to_slice_mut(),
        &mut fwd_lc.to_slice_mut(),
        &mut fwd_ok.to_slice_mut(),
        &mut ct_g.to_slice_mut(),
        &mut ct_h.to_slice_mut(),
        &mut ct_c.to_slice_mut(),
        &mut lmin.to_slice_mut(),
        offset[fi],
        default_bin[fi],
        skip_default_bin[fi],
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        sum_hessian,
        num_data,
        fwd_count[fi],
    );
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
    if scan_official_enabled(client) && scan_variants_applicable(cfg) {
        SCAN_OFFICIAL_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let plane_dim = client.properties().hardware.plane_size_max;
        // Official runs 256-wide (one lane per bin) with its own plane_dim arg — a
        // dedicated launch, not the SCAN_STAGED_CUBE_DIM `launch_with!` macro.
        unsafe {
            find_best_splits_fused_staged_official_kernel::launch(
                client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(SCAN_OFFICIAL_CUBE_DIM),
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
                plane_dim,
            );
        }
    } else if scan_parprefix_enabled(<R as cubecl::Runtime>::name(client))
        && scan_variants_applicable(cfg)
    {
        SCAN_PARPREFIX_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        launch_with!(find_best_splits_fused_staged_parprefix_kernel);
    } else if scan_pargain_enabled(<R as cubecl::Runtime>::name(client))
        && scan_variants_applicable(cfg)
    {
        SCAN_PARGAIN_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    if scan_official_enabled(client) && scan_variants_applicable(cfg) {
        SCAN_OFFICIAL_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let plane_dim = client.properties().hardware.plane_size_max;
        unsafe {
            find_best_splits_fused_siblings_staged_official_kernel::launch(
                client,
                CubeCount::Static(n as u32, 2, 1),
                CubeDim::new_1d(SCAN_OFFICIAL_CUBE_DIM),
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
                plane_dim,
            );
        }
    } else if scan_parprefix_enabled(<R as cubecl::Runtime>::name(client))
        && scan_variants_applicable(cfg)
    {
        SCAN_PARPREFIX_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        launch_with!(find_best_splits_fused_siblings_staged_parprefix_kernel);
    } else if scan_pargain_enabled(<R as cubecl::Runtime>::name(client))
        && scan_variants_applicable(cfg)
    {
        SCAN_PARGAIN_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        NumDataSrc::Host(num_data),
    )
}

/// The source of a fused scan's per-child `num_data` (row count). `Host` is today's
/// default (a host scalar). `Device` (SPEC-DRGL-12, real-device only) resolves the
/// count ON DEVICE from the resident split/role record so the scan can run BEFORE the
/// host reads `split_point` back (the SPEC-DRGL-05 `read_split` deferral). The `Device`
/// variant forces the legacy lane-per-feature scan kernel's device-`num_data` twin
/// ([`find_best_splits_fused_kernel_devcount`]), bypassing the staged/autotune variants —
/// still bit-identical, since every scan variant shares [`split_scan_body`] and the
/// resolved count equals the host scalar it replaces.
pub(crate) enum NumDataSrc {
    /// Host-known child count (the shipped default path — byte-unchanged).
    Host(i32),
    /// Device-resident child count (real-device only). `ranges`/`roles` are the
    /// [`DeviceLeafSplits`](crate::kernels::partition::DeviceLeafSplits) buffers;
    /// `parent_count` is the host-known parent row count (upper bound, no new sync).
    /// The fields are consumed only by the `#[cfg(feature = "gpu")]` launch arm, so a
    /// cpu build reads none of them (it takes the typed-error arm instead).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    Device {
        ranges: cubecl::server::Handle,
        ranges_len: usize,
        roles: cubecl::server::Handle,
        roles_len: usize,
        split_slot: u32,
        /// Child selector: 0=Left 1=Right 2=Smaller 3=Larger (SPEC-DRGL-05).
        which: u32,
        parent_count: i32,
    },
}

/// SPEC-DRGL-13: the co-pack (two-sibling) analog of [`NumDataSrc`]. `Host` ⇒ both siblings'
/// counts come from the `a_totals`/`b_totals` scalars (byte-unchanged default). `Device` ⇒
/// resolve BOTH on device from ONE resident split/role record (A = smaller via `is_smaller=1`,
/// B = larger via `is_smaller=0`; larger = `parent_count - split_point`), so the co-pack scan
/// can run before the host reads `split_point` back (the SPEC-DRGL-05 deferral).
pub(crate) enum SiblingNumDataSrc {
    /// Counts come from the `a_totals`/`b_totals` tuples (the shipped default path).
    Host,
    /// Device-resident counts for both siblings (real-device only), from ONE split slot.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    Device {
        ranges: cubecl::server::Handle,
        ranges_len: usize,
        roles: cubecl::server::Handle,
        roles_len: usize,
        split_slot: u32,
        /// Sibling A/B child selectors: 0=Left 1=Right 2=Smaller 3=Larger (SPEC-DRGL-05).
        which_a: u32,
        which_b: u32,
        parent_count: i32,
    },
}

/// SPEC-DRGL-12: device-`num_data` twin of
/// [`find_best_splits_batched_fused_f64_from_handle_on`] — identical fused scan, but the
/// child's `num_data` is resolved ON DEVICE from the resident split/role record
/// (`ranges`/`roles` of a [`DeviceLeafSplits`](crate::kernels::partition::DeviceLeafSplits))
/// plus the host-known `parent_count`, instead of a host `num_data` scalar. Returns the
/// SAME `Vec<SplitInfo>` (one per feature, input order) as the host-`num_data` scan, and
/// is byte-identical to it for the same device state — the property SPEC-DRGL-05's
/// deferral relies on. Real-device only (the `Device` path errors on a cpu backend).
///
/// # Errors
/// As [`find_best_splits_batched_fused_f64_from_handle_on`]; plus a typed error if a
/// non-GPU build reaches the device path.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_batched_fused_f64_devcount_from_handle_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    ranges: cubecl::server::Handle,
    ranges_len: usize,
    roles: cubecl::server::Handle,
    roles_len: usize,
    split_slot: u32,
    which: u32,
    parent_count: i32,
) -> Result<Vec<SplitInfo>, ComputeError> {
    find_best_splits_fused_inner(
        client,
        hist_handle,
        buf_len,
        feats,
        cfg,
        sum_gradient,
        sum_hessian,
        NumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which,
            parent_count,
        },
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
    num_data_src: NumDataSrc,
) -> Result<Vec<SplitInfo>, ComputeError> {
    // Empty batch: no launch (T-mc5-03).
    if feats.is_empty() {
        return Ok(Vec::new());
    }

    // `num_data` reaches the kernel only at the launch (validation / min_gain_shift /
    // rev-fwd counts / decode are all num_data-free). `Host` supplies it as a scalar
    // (byte-unchanged default path); `Device` (SPEC-DRGL-12) forces the legacy kernel's
    // device-`num_data` twin, which reads the count from the resident split/role record.
    // `is_device` gates the staged/autotune width variants off (Device forces the legacy
    // twin); those gates are `#[cfg(feature = "gpu")]`, so on a cpu build it is unused.
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
    let is_device = matches!(num_data_src, NumDataSrc::Device { .. });
    let num_data_host: i32 = match &num_data_src {
        NumDataSrc::Host(n) => *n,
        NumDataSrc::Device { .. } => 0, // unused on the Device path (legacy twin reads it on device)
    };

    // Leaf-level, checked once: only the default smoothing/clamp path is
    // transcribed. Reject non-default values rather than mis-compute.
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
    // The C++ `USE_SMOOTHING` template bool (`path_smooth > kEpsilon`),
    // resolved once per launch exactly like `USE_L1` above.
    let use_smoothing = cfg.use_smoothing();
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
    // SPEC-DRGL-12: the Device num_data source takes its OWN device-`num_data` launch
    // path (`device_parprefix` below on the default hip parprefix arm, else the legacy
    // twin in the fallback) — the HOST staged/autotune width variants stay on the
    // host-scalar path only. `is_device` therefore forces `staged`/`autotuned` off.
    #[cfg(feature = "gpu")]
    let staged = !is_device
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
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
                num_data_host,
            );
        }
    }

    // SPEC-DRGL-12: the Device num_data source on a real device with the parprefix scan
    // enabled (the flag-OFF default on hip) uses the parprefix kernel's device-`num_data`
    // TWIN — BYTE-IDENTICAL to the host parprefix scan, so a deferred (device-count) child
    // scan reproduces the non-deferred (host-count) parprefix winner exactly. Falls back to
    // the legacy devcount twin (below) only when parprefix is off (a non-default config).
    #[cfg(feature = "gpu")]
    let device_parprefix = is_device
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && scan_parprefix_enabled(<R as cubecl::Runtime>::name(client))
        && num_bin_a.iter().all(|&nb| (nb as usize) * 2 <= SCAN_STAGE_MAX_CELLS);
    #[cfg(not(feature = "gpu"))]
    let device_parprefix = false;

    #[cfg(feature = "gpu")]
    if device_parprefix {
        SCAN_PARPREFIX_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let NumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which,
            parent_count,
        } = &num_data_src
        {
            // SAFETY: identical obligations to `launch_staged_single_scan`; geometry
            // CubeCount::Static(n) matches the kernel's CUBE_POS_X < n no-guard contract.
            unsafe {
                find_best_splits_fused_staged_parprefix_kernel_devcount::launch(
                    client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(SCAN_STAGED_CUBE_DIM),
                    ArrayArg::from_raw_parts(h_hist.clone(), buf_len),
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
                    ArrayArg::from_raw_parts(ranges.clone(), *ranges_len),
                    ArrayArg::from_raw_parts(roles.clone(), *roles_len),
                    *split_slot,
                    *which,
                    *parent_count,
                );
            }
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
    let autotuned = !is_device
        && !staged
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
            cfg.max_delta_step,
            if use_smoothing { 1u32 } else { 0u32 },
            cfg.path_smooth,
            cfg.parent_output,
            min_gain_shift,
            sum_gradient,
            sum_hessian_bumped,
            num_data_host,
        ));
        SCAN_TUNER.execute(&autotune::cache_namespace_id(), client, set, handles);
    }

    if !autotuned && !staged && !device_parprefix {
        // Scan-occupancy lever: pack one feature per LANE. `scan_cube_dim()`
        // (env `LGBM_SCAN_CUBEDIM`; rocm default W=64, W=1 = byte-identical to the
        // original) is the cube width W; `CubeCount = ceil(n / W)`. The kernel indexes
        // features by the global lane `ABSOLUTE_POS` and guards `f < n_feats`, so the
        // tail cube's spare lanes no-op and the result is bit-identical to W=1 for
        // every W. SPEC-DRGL-12 fallback: a `Device` num_data source that did NOT take
        // the parprefix twin above (parprefix off) lands here on the legacy devcount twin.
        let scan_w = scan_cube_dim();
        let cube_count = (n as u32).div_ceil(scan_w);

        // SAFETY: every handle is sized to its slice and outlives the launch. Lane `f`
        // (`ABSOLUTE_POS`, guarded `< n_feats` in the kernel) reads only
        // `[slot_off[f], slot_off[f]+2*num_bin[f])` — each validated `<= buf.len()` above
        // — and writes only `out[f*12 .. f*12+12]` within the `n*12` allocation; the
        // shared `split_scan_body` carries the same in-range / negative-`t` clamp guards
        // as the single-feature kernel. All per-feature index arrays have exactly `n`
        // elements. All cubecl unsafe is confined here (CMP-01). SPEC-DRGL-12: the
        // `Device` num_data source launches the device-`num_data` twin
        // ([`find_best_splits_fused_kernel_devcount`]) — SAME per-feature scan, count read
        // from the resident split/role record instead of the host `num_data` scalar.
        match num_data_src {
            NumDataSrc::Host(_) => unsafe {
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
                    cfg.max_delta_step,
                    if use_smoothing { 1u32 } else { 0u32 },
                    cfg.path_smooth,
                    cfg.parent_output,
                    min_gain_shift,
                    sum_gradient,
                    sum_hessian_bumped,
                    num_data_host,
                    n as u32,
                );
            },
            #[cfg(feature = "gpu")]
            NumDataSrc::Device {
                ranges,
                ranges_len,
                roles,
                roles_len,
                split_slot,
                which,
                parent_count,
            } => unsafe {
                SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                find_best_splits_fused_kernel_devcount::launch(
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
                    cfg.max_delta_step,
                    if use_smoothing { 1u32 } else { 0u32 },
                    cfg.path_smooth,
                    cfg.parent_output,
                    min_gain_shift,
                    sum_gradient,
                    sum_hessian_bumped,
                    ArrayArg::from_raw_parts(ranges, ranges_len),
                    ArrayArg::from_raw_parts(roles, roles_len),
                    split_slot,
                    which,
                    parent_count,
                    n as u32,
                );
            },
            #[cfg(not(feature = "gpu"))]
            NumDataSrc::Device { .. } => {
                return Err(ComputeError::Runtime {
                    detail: "find_best_splits (device num_data): the resident device-num_data \
                             scan requires a GPU backend"
                        .to_string(),
                });
            }
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
    // (sum_gradient, sum_hessian, num_data, parent_output) per sibling
    // (A = smaller, B = larger). `parent_output` is per-LEAF — the two siblings get
    // DIFFERENT values (their parent split's left_output / right_output), so it
    // cannot ride along in the shared `cfg`.
    a_totals: (f64, f64, i32, f64),
    b_totals: (f64, f64, i32, f64),
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
    let (sum_gradient_a, sum_hessian_a, num_data_a, parent_output_a) = a_totals;
    let (sum_gradient_b, sum_hessian_b, num_data_b, parent_output_b) = b_totals;
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
    // The C++ `USE_SMOOTHING` template bool (`path_smooth > kEpsilon`),
    // resolved once per launch exactly like `USE_L1` above.
    let use_smoothing = cfg.use_smoothing();
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
        && scan_variants_applicable(cfg)
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
            cfg.max_delta_step,
            if use_smoothing { 1u32 } else { 0u32 },
            cfg.path_smooth,
            parent_output_a,
            parent_output_b,
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
                cfg.max_delta_step,
                if use_smoothing { 1u32 } else { 0u32 },
                cfg.path_smooth,
                parent_output_a,
                parent_output_b,
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

/// PLANE-PARALLEL reduce gate (`LGBM_REDUCE_PAR`, P1a of the nsys round —
/// `docs/ondevice-cuda-perf-plan.md` §5). The serial reduce kernels run ONE thread
/// walking `n_feats * 12` f64 cells from global memory (~119 µs/launch on P100 vs
/// official's 3.7 µs analog — 283 ms/train on the co-pack reduce alone); the plane
/// twins fold lane-strided subsets and pick the winner with two plane collectives.
/// BIT-EXACT by construction (see `reduce_window_par_body`), so the hatch is
/// benchmark-only. REAL-DEVICE ONLY: cubecl-cpu has `plane_size == 1` / no plane
/// ops — the cpu anchor keeps the serial kernel byte-unchanged. DEFAULT ON
/// (`LGBM_REDUCE_PAR=0` reverts): P100 A/B `lgb-rs-p1a-reduce-par` (2026-07-15,
/// order-alternated warm-median-of-3, 500k×50×100) measured 5.177→4.880 s =
/// 1.061× with preds bit-identical (max_abs 0.0) and the kernel at 3.5 µs vs
/// 119 µs serial (nsys); counts proof reduce_par=2880 vs 0. hip byte-identity
/// gates green on gfx1151 (`reduce_par_parity`, `cuda_on_device` 7/7 forced-ON).
#[cfg(feature = "gpu")]
pub fn reduce_par_enabled<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
) -> bool {
    if <R as cubecl::Runtime>::name(client) == "cpu" {
        return false;
    }
    if client.properties().hardware.plane_size_max < 2 {
        return false;
    }
    match REDUCE_PAR_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| !matches!(std::env::var("LGBM_REDUCE_PAR").as_deref(), Ok("0")))
}

/// Same-session A/B override for [`reduce_par_enabled`]. 0 = unset (env decides),
/// 1 = force ON, 2 = force OFF. The cpu/plane hard gates still apply.
#[cfg(feature = "gpu")]
static REDUCE_PAR_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Test/harness hook: force the plane-parallel reduce ON/OFF or defer to the env.
#[cfg(feature = "gpu")]
pub fn set_reduce_par_override(v: Option<bool>) {
    let code = match v {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    REDUCE_PAR_OVERRIDE.store(code, std::sync::atomic::Ordering::Relaxed);
}

/// POSITIVE tripwire — bumped once per reduce launch that dispatched a PLANE-PARALLEL
/// twin (bench-protocol counts proof). Folded into the `phase_prof` COUNTS line as
/// `reduce_par=`.
#[cfg(feature = "gpu")]
pub static REDUCE_PAR_CNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Swap the plane-parallel-reduce tripwire to zero and return the prior value
/// (consumed by `phase_prof::dump`). Present on every build (0 without `gpu`).
pub fn reduce_par_count_take() -> u64 {
    #[cfg(feature = "gpu")]
    {
        REDUCE_PAR_CNT.swap(0, std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}

/// PLANE-PARALLEL cross-feature reduce BODY — the all-lanes twin of the serial
/// fold in [`reduce_scan_output_into_leaf_kernel`]. Every unit of ONE plane
/// strides the `n_feats` raw records with the IDENTICAL accept-gate + two-key
/// `split_gt` compare (strictly-greater raw gain, exact-tie → lower real feature
/// key), then two plane collectives (`plane_max` on the lane-best gain,
/// `plane_min` on the feature key among gain-tied lanes) select the winner lane,
/// which re-reads its record's cells and writes the slot.
///
/// ## Why this is BIT-EXACT (not ~1e-6)
/// Within one window the (raw_gain desc, real_feat asc) key set is STRICTLY
/// totally ordered: feature keys are distinct per record, and NaN gains are
/// excluded by the accept-gate (`raw_gain > neg_inf` is false for NaN). A strict
/// total order has a UNIQUE maximum, so ANY reduction order — lane-strided local
/// folds + plane collectives here, ascending single-thread walk in the serial
/// kernel — selects the SAME record. The winning fields are copied verbatim from
/// the same raw cells, and the net-gain transform
/// `(raw_gain - min_gain_shift) * penalty` is the identical single f64 op.
/// The all-invalid window writes the identical seed sentinel (valid=0,
/// gain=`(neg_inf - min_gain_shift) * penalty`, feat=-1, rest 0).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::too_many_arguments)]
fn reduce_window_par_body(
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
    rb_u: u32,
    n_feats: u32,
    slot_u: u32,
    min_gain_shift: f64,
    penalty: f64,
    neg_inf: f64,
) {
    let rb = rb_u as usize;
    let n = n_feats as usize;
    let slot = slot_u as usize;
    // Lane-local fold over the strided subset — the serial compare verbatim,
    // carried in f64 locals (real-device only; no cpu-MLIR local-carry limits).
    let mut lv = f64::new(0.0);
    let mut lg = neg_inf;
    let mut lf = f64::new(-1.0);
    let mut lt = f64::new(0.0);
    let mut t = UNIT_POS as usize;
    while t < n {
        let dbase = rb + t * 12;
        let v = (raw[dbase] != 0.0) && (raw[dbase + 2] > neg_inf);
        let strictly_gain = raw[dbase + 2] > lg;
        let tie_gain = raw[dbase + 2] == lg;
        let feat_lower = real_feats[t] < lf;
        let take = v && (strictly_gain || (tie_gain && feat_lower));
        lv = select(take, 1.0, lv);
        lg = select(take, raw[dbase + 2], lg);
        lf = select(take, real_feats[t], lf);
        lt = select(take, f64::cast_from(t as u32), lt);
        t += CUBE_DIM as usize;
    }
    // Plane argmax: max gain over the plane, then min feature key among the
    // gain-tied lanes. Invalid lanes hold (neg_inf, -1) and get a +inf key, so
    // they never tie a valid lane (a valid lane's gain is > neg_inf by the gate).
    let any = plane_max(lv);
    let m = plane_max(lg);
    let pos_inf = f64::new(0.0) - neg_inf;
    let fk = select(lg == m, lf, pos_inf);
    let fmin = plane_min(fk);
    let win = (lv != 0.0) && (lg == m) && (lf == fmin);
    if win {
        // EXACTLY ONE lane: distinct feature keys ⇒ (m, fmin) names one record.
        let dbase = rb + (u32::cast_from(lt) as usize) * 12;
        out_valid[slot] = 1.0;
        out_feat[slot] = lf;
        out_thr[slot] = raw[dbase + 1];
        out_dleft[slot] = raw[dbase + 9];
        out_ncat[slot] = 0.0;
        out_lsum_g[slot] = raw[dbase + 5];
        out_lsum_h[slot] = raw[dbase + 6];
        out_rsum_g[slot] = raw[dbase + 7];
        out_rsum_h[slot] = raw[dbase + 8];
        out_lval[slot] = raw[dbase + 10];
        out_rval[slot] = raw[dbase + 11];
        out_gain[slot] = (lg - min_gain_shift) * penalty;
    }
    if (any == 0.0) && (UNIT_POS == 0u32) {
        // No valid record: the serial kernel's final state is its seed with the
        // net-gain transform applied — reproduce it verbatim.
        out_valid[slot] = 0.0;
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
        out_gain[slot] = (neg_inf - min_gain_shift) * penalty;
    }
}

/// PLANE-PARALLEL twin of [`reduce_scan_output_into_leaf_kernel`] — one plane
/// (`CubeDim = plane_size`), same args, same slot writes. Dispatched by
/// [`launch_reduce_into_leaf`] under [`reduce_par_enabled`].
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn reduce_scan_output_into_leaf_par_kernel(
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
    reduce_window_par_body(
        raw, real_feats, out_valid, out_gain, out_feat, out_thr, out_dleft, out_ncat,
        out_lsum_g, out_lsum_h, out_rsum_g, out_rsum_h, out_lval, out_rval, raw_base, n_feats,
        out_slot, min_gain_shift, penalty, neg_inf,
    );
}

/// PLANE-PARALLEL twin of [`reduce_scan_output_into_two_leaves_kernel`] — TWO
/// cubes of one plane each (`CubeCount::Static(2,1,1)`), cube `task` folds its
/// sibling window into its slot exactly as the serial twin. Dispatched by
/// [`launch_reduce_into_two_leaves`] under [`reduce_par_enabled`].
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn reduce_scan_output_into_two_leaves_par_kernel(
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
    n_feats: u32,
    out_slot_a: u32,
    out_slot_b: u32,
    min_gain_shift_a: f64,
    min_gain_shift_b: f64,
    penalty: f64,
    neg_inf: f64,
) {
    let task = CUBE_POS_X;
    let rb = select(task == 0, 0u32, n_feats * 12u32);
    let slot = select(task == 0, out_slot_a, out_slot_b);
    let min_gain_shift = select(task == 0, min_gain_shift_a, min_gain_shift_b);
    reduce_window_par_body(
        raw, real_feats, out_valid, out_gain, out_feat, out_thr, out_dleft, out_ncat,
        out_lsum_g, out_lsum_h, out_rsum_g, out_rsum_h, out_lval, out_rval, rb, n_feats, slot,
        min_gain_shift, penalty, neg_inf,
    );
}

/// SPEC-DRGL-05 deferral: DEVICE-TARGET twin of
/// [`reduce_scan_output_into_two_leaves_par_kernel`]. Under the read_split deferral the
/// host does not know which child leaf is the SMALLER sibling (window A) — this twin
/// resolves the fold TARGETS and per-branch `min_gain_shift` from the resident role
/// record: A → `select(smaller_is_left, left, right)` of the two host-known child leaf
/// slots/scalars, B → the other. Same fold body — bit-identical winners.
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn reduce_scan_output_into_two_leaves_par_device_target_kernel(
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
    roles: &Array<i32>,
    split_slot: u32,
    n_feats: u32,
    out_slot_left: u32,
    out_slot_right: u32,
    min_gain_shift_left: f64,
    min_gain_shift_right: f64,
    penalty: f64,
    neg_inf: f64,
) {
    let task = CUBE_POS_X;
    let smaller_is_left = roles[(split_slot * 3) as usize] != 0;
    let slot_a = select(smaller_is_left, out_slot_left, out_slot_right);
    let slot_b = select(smaller_is_left, out_slot_right, out_slot_left);
    let mgs_a = select(smaller_is_left, min_gain_shift_left, min_gain_shift_right);
    let mgs_b = select(smaller_is_left, min_gain_shift_right, min_gain_shift_left);
    let rb = select(task == 0, 0u32, n_feats * 12u32);
    let slot = select(task == 0, slot_a, slot_b);
    let min_gain_shift = select(task == 0, mgs_a, mgs_b);
    reduce_window_par_body(
        raw, real_feats, out_valid, out_gain, out_feat, out_thr, out_dleft, out_ncat,
        out_lsum_g, out_lsum_h, out_rsum_g, out_rsum_h, out_lval, out_rval, rb, n_feats, slot,
        min_gain_shift, penalty, neg_inf,
    );
}

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
pub fn launch_reduce_into_leaf<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    h_raw: cubecl::server::Handle,
    raw_len: usize,
    real_feats: &[i32],
    n_feats: usize,
    out: &crate::kernels::best_split::SplitSoa,
    out_slot: usize,
    raw_base: usize,
    min_gain_shift: f64,
    // Per-grow cached `rf` handle (LGBM_DESC_HOIST): the SAME f64 image of
    // `real_feats[..n_feats]` already device-resident. The caller guarantees the
    // cache was built from THIS `real_feats` with length >= `n_feats` (the entry
    // fns validate `real_feats.len() == feats.len() == n_feats` before passing it).
    h_rf_cached: Option<cubecl::server::Handle>,
) {
    let h_rf = h_rf_cached.unwrap_or_else(|| {
        let rf: Vec<f64> = real_feats.iter().take(n_feats).map(|&r| f64::from(r)).collect();
        client.create_from_slice(f64::as_bytes(&rf))
    });
    // PLANE-PARALLEL twin (LGBM_REDUCE_PAR, real-device only) — bit-exact winner,
    // one plane instead of one thread. Same args, same slot writes.
    #[cfg(feature = "gpu")]
    if reduce_par_enabled(client) {
        let pd = client.properties().hardware.plane_size_max;
        REDUCE_PAR_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: identical bounds contract to the serial launch below (caller-proven);
        // the kernel folds only `out_*[out_slot]` and reads only the caller-validated
        // `raw` window + `real_feats[0..n)`. Every handle outlives the launch.
        unsafe {
            reduce_scan_output_into_leaf_par_kernel::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(pd),
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
        return;
    }
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

/// The TWO-TASK batched twin of [`reduce_scan_output_into_leaf_kernel`]: ONE
/// launch (`CubeCount::Static(2, 1, 1)`) folds BOTH co-pack siblings' winners in a
/// single dispatch — `CUBE_POS_X` selects the sibling (0 = A/smaller reads raw
/// window base 0 → slot `out_slot_a`; 1 = B/larger reads base `n*12` → slot
/// `out_slot_b`). Each cube is single-owner (`UNIT_POS == 0`) and writes ONLY its
/// own frontier slot (`out_slot_a != out_slot_b`, disjoint), so the two tasks never
/// race — the result is BIT-IDENTICAL to the two separate
/// [`reduce_scan_output_into_leaf_kernel`] launches it replaces (same per-slot seed,
/// same per-feature `split_gt` fold, same net-gain conversion). Halves the co-pack
/// reduce launch count (2 → 1) — a pure host-enqueue win (spike095: ~100µs/launch).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn reduce_scan_output_into_two_leaves_kernel(
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
    n_feats: u32,
    out_slot_a: u32,
    out_slot_b: u32,
    min_gain_shift_a: f64,
    min_gain_shift_b: f64,
    penalty: f64,
    neg_inf: f64,
) {
    let task = CUBE_POS_X;
    if UNIT_POS == 0 {
        // Per-sibling params (A = raw window base 0 → slot A; B = base n*12 → slot B).
        let n = n_feats as usize;
        let rb = select(task == 0, 0u32, n_feats * 12u32) as usize;
        let slot = select(task == 0, out_slot_a, out_slot_b) as usize;
        let min_gain_shift = select(task == 0, min_gain_shift_a, min_gain_shift_b);
        // The seed + fold is IDENTICAL to the single-leaf kernel (verbatim).
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
            let v = (raw[dbase] != 0.0) && (raw[dbase + 2] > neg_inf);
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
            out_lval[slot] = select(take, raw[dbase + 10], out_lval[slot]);
            out_rval[slot] = select(take, raw[dbase + 11], out_rval[slot]);
        }
        out_gain[slot] = (out_gain[slot] - min_gain_shift) * penalty;
    }
}

/// Launch [`reduce_scan_output_into_two_leaves_kernel`] — ONE dispatch folding
/// BOTH co-pack siblings' winners (sibling A raw window `[0, n*12)` → `out_slot_a`;
/// sibling B `[n*12, 2n*12)` → `out_slot_b`). Replaces two
/// [`launch_reduce_into_leaf`] calls. `real_feats` (shared feature layout) must have
/// `>= n_feats` elements; `h_raw` must describe `>= 2*n_feats*12` cells; both
/// `out_slot`s `< out.len` and DISTINCT. Bounds host-proven by the caller.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_reduce_into_two_leaves<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    h_raw: cubecl::server::Handle,
    raw_len: usize,
    real_feats: &[i32],
    n_feats: usize,
    out: &crate::kernels::best_split::SplitSoa,
    out_slot_a: usize,
    out_slot_b: usize,
    min_gain_shift_a: f64,
    min_gain_shift_b: f64,
    // Per-grow cached `rf` handle (LGBM_DESC_HOIST) — see `launch_reduce_into_leaf`.
    h_rf_cached: Option<cubecl::server::Handle>,
) {
    let h_rf = h_rf_cached.unwrap_or_else(|| {
        let rf: Vec<f64> = real_feats.iter().take(n_feats).map(|&r| f64::from(r)).collect();
        client.create_from_slice(f64::as_bytes(&rf))
    });
    // PLANE-PARALLEL twin (LGBM_REDUCE_PAR, real-device only) — bit-exact winners,
    // two one-plane cubes instead of two single threads.
    if reduce_par_enabled(client) {
        let pd = client.properties().hardware.plane_size_max;
        REDUCE_PAR_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: identical bounds contract to the serial launch below (caller-proven):
        // cube `task` folds ONLY `out_*[out_slot_{a|b}]` (distinct) and reads its
        // `raw` window + `real_feats[0..n)`. Every handle outlives the launch.
        unsafe {
            reduce_scan_output_into_two_leaves_par_kernel::launch_unchecked(
                client,
                CubeCount::Static(2, 1, 1),
                CubeDim::new_1d(pd),
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
                n_feats as u32,
                out_slot_a as u32,
                out_slot_b as u32,
                min_gain_shift_a,
                min_gain_shift_b,
                1.0f64,
                f64::NEG_INFINITY,
            );
        }
        return;
    }
    // SAFETY: two single-owner cubes (Route C, geometry `Static(2,1,1)`). Cube `task`
    // seeds + folds ONLY `out_*[out_slot_{a|b}]` (both `< out.len`, DISTINCT, caller-
    // validated ⇒ no cross-cube write race) and reads only `raw[rb .. rb + n*12)` with
    // `rb ∈ {0, n*12}` (⇒ `raw[0 .. 2n*12) <= raw_len`, caller-validated) and
    // `real_feats[0 .. n)` (`h_rf` sized `n_feats`). Every handle outlives the launch.
    // All cubecl unsafe confined here (CMP-01).
    unsafe {
        reduce_scan_output_into_two_leaves_kernel::launch_unchecked(
            client,
            CubeCount::Static(2, 1, 1),
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
            n_feats as u32,
            out_slot_a as u32,
            out_slot_b as u32,
            min_gain_shift_a,
            min_gain_shift_b,
            1.0f64,
            f64::NEG_INFINITY,
        );
    }
}

/// SPEC-DRGL-05 deferral: DEVICE-TARGET two-leaf reduce launcher — fold targets and
/// per-branch `min_gain_shift` resolved on device from the resident role record
/// ([`reduce_scan_output_into_two_leaves_par_device_target_kernel`]). PAR-ONLY, real
/// device only (the deferred arm requires the cuda-default `reduce_par` config); a
/// non-par/cpu client is a typed error so the deferral gate can surface it once.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn launch_reduce_into_two_leaves_device_target<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    h_raw: cubecl::server::Handle,
    raw_len: usize,
    real_feats: &[i32],
    n_feats: usize,
    out: &crate::kernels::best_split::SplitSoa,
    roles: cubecl::server::Handle,
    roles_len: usize,
    split_slot: u32,
    out_slot_left: usize,
    out_slot_right: usize,
    min_gain_shift_left: f64,
    min_gain_shift_right: f64,
    h_rf_cached: Option<cubecl::server::Handle>,
) -> Result<(), ComputeError> {
    if !reduce_par_enabled(client) || <R as cubecl::Runtime>::name(client) == "cpu" {
        return Err(ComputeError::Runtime {
            detail: "launch_reduce_into_two_leaves_device_target: the deferred arm requires \
                     the par reduce on a real device (LGBM_REDUCE_PAR)"
                .to_string(),
        });
    }
    let h_rf = h_rf_cached.unwrap_or_else(|| {
        let rf: Vec<f64> = real_feats.iter().take(n_feats).map(|&r| f64::from(r)).collect();
        client.create_from_slice(f64::as_bytes(&rf))
    });
    let pd = client.properties().hardware.plane_size_max;
    REDUCE_PAR_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: identical bounds contract to `launch_reduce_into_two_leaves`'s par arm;
    // additionally reads `roles[3*split_slot]` (caller-validated within `roles_len`).
    unsafe {
        reduce_scan_output_into_two_leaves_par_device_target_kernel::launch_unchecked(
            client,
            CubeCount::Static(2, 1, 1),
            CubeDim::new_1d(pd),
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
            ArrayArg::from_raw_parts(roles, roles_len),
            split_slot,
            n_feats as u32,
            out_slot_left as u32,
            out_slot_right as u32,
            min_gain_shift_left,
            min_gain_shift_right,
            1.0f64,
            f64::NEG_INFINITY,
        );
    }
    Ok(())
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
    num_data_src: NumDataSrc,
    // Per-grow cached descriptor set (LGBM_DESC_HOIST). `Some` with matching
    // geometry skips the per-launch array assembly + 7 uploads (same bytes, same
    // kernels — bit-exact); anything else takes the byte-unchanged per-launch path.
    desc: Option<&ScanDescHandles>,
) -> Result<Option<(cubecl::server::Handle, usize, f64)>, ComputeError> {
    if feats.is_empty() {
        return Ok(None);
    }
    // SPEC-DRGL-12: `Host` supplies num_data as a scalar (byte-unchanged default path);
    // `Device` resolves it on device in the scan kernel (parprefix twin on hip, else the
    // legacy twin). `num_data` reaches the kernel only at the launch.
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
    let is_device = matches!(num_data_src, NumDataSrc::Device { .. });
    let num_data_host: i32 = match &num_data_src {
        NumDataSrc::Host(n) => *n,
        NumDataSrc::Device { .. } => 0,
    };
    // Leaf-level scope checks (identical to `find_best_splits_fused_inner`).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_reduce: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }

    // Per-feature V5 validation + device-array assembly (BEFORE launch) — IDENTICAL to
    // `find_best_splits_fused_inner`, including the whole-batch `na_as_missing` reject.
    // A MATCHING per-grow cached set (`desc`) ran this same validation + uploaded these
    // same bytes once at cache time, so it skips both (bit-exact by construction).
    let n = feats.len();
    let owned_desc: ScanDescHandles;
    let d: &ScanDescHandles = match desc {
        Some(d) if d.matches(n, buf_len) => {
            SCAN_DESC_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            d
        }
        _ => {
            owned_desc =
                upload_scan_desc(client, feats, None, buf_len, "find_best_splits_reduce")?;
            &owned_desc
        }
    };

    // LEAF-LEVEL scalars ONCE — the 2*kEpsilon bump + min_gain_shift (identical math).
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;
    let use_l1 = cfg.use_l1();
    // The C++ `USE_SMOOTHING` template bool (`path_smooth > kEpsilon`),
    // resolved once per launch exactly like `USE_L1` above.
    let use_smoothing = cfg.use_smoothing();
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
    let h_slot = d.h_slot.clone();
    let h_numbin = d.h_numbin.clone();
    let h_offset = d.h_offset.clone();
    let h_defbin = d.h_defbin.clone();
    let h_skip = d.h_skip.clone();
    let h_rev = d.h_rev.clone();
    let h_fwd = d.h_fwd.clone();

    // STAGED cube-per-feature branch (opt-in `LGBM_SCAN_STAGED=1`) on the LIVE
    // no-readback route — same gates as `find_best_splits_fused_inner` (real
    // device runtime, every feature ≤ 256 bins). Bit-identical output cells, so
    // the reduce launcher downstream is untouched.
    #[cfg(feature = "gpu")]
    if !is_device
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && d.staged_capable
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
                num_data_host,
            );
        }
        return Ok(Some((h_out, out_len, min_gain_shift)));
    }

    // SPEC-DRGL-12: the Device num_data source on a real device with parprefix enabled
    // (the flag-OFF default on hip) uses the parprefix kernel's device-`num_data` TWIN —
    // BYTE-IDENTICAL to the host parprefix scan, so a deferred child scan folds the SAME
    // winner the non-deferred scan would. Falls back to the legacy devcount twin below
    // when parprefix is off (non-default config).
    #[cfg(feature = "gpu")]
    if is_device
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && scan_parprefix_enabled(<R as cubecl::Runtime>::name(client))
        && d.staged_capable
    {
        SCAN_PARPREFIX_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let NumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which,
            parent_count,
        } = &num_data_src
        {
            // SAFETY: identical obligations to `launch_staged_single_scan`; geometry
            // CubeCount::Static(n) matches the kernel's CUBE_POS_X < n no-guard contract.
            unsafe {
                find_best_splits_fused_staged_parprefix_kernel_devcount::launch(
                    client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(SCAN_STAGED_CUBE_DIM),
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
                    ArrayArg::from_raw_parts(ranges.clone(), *ranges_len),
                    ArrayArg::from_raw_parts(roles.clone(), *roles_len),
                    *split_slot,
                    *which,
                    *parent_count,
                );
            }
        }
        return Ok(Some((h_out, out_len, min_gain_shift)));
    }

    // Direct `scan_cube_dim()` launch (bit-neutral W; no autotune — see fn doc). This is the
    // SAME `find_best_splits_fused_kernel` the host-readback path launches; only the readback
    // is dropped (the reduce kernel consumes `h_out` on device). SPEC-DRGL-12: a `Device`
    // num_data source that did NOT take the parprefix twin (parprefix off) lands on the
    // legacy devcount twin here.
    let scan_w = scan_cube_dim();
    let cube_count = (n as u32).div_ceil(scan_w);
    // SAFETY: every per-feature region `[slot_off[f], slot_off[f]+2*num_bin[f])` is validated
    // `<= buf_len` above; lane `f` (guarded `< n_feats`) reads only its region and writes
    // `out[f*12 .. f*12+12]` within the `n*12` allocation; all per-feature index arrays have
    // exactly `n` elements. All cubecl unsafe confined here (CMP-01).
    match num_data_src {
        NumDataSrc::Host(_) => unsafe {
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
                cfg.max_delta_step,
                if use_smoothing { 1u32 } else { 0u32 },
                cfg.path_smooth,
                cfg.parent_output,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                num_data_host,
                n as u32,
            );
        },
        #[cfg(feature = "gpu")]
        NumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which,
            parent_count,
        } => unsafe {
            SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            find_best_splits_fused_kernel_devcount::launch(
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
                cfg.max_delta_step,
                if use_smoothing { 1u32 } else { 0u32 },
                cfg.path_smooth,
                cfg.parent_output,
                min_gain_shift,
                sum_gradient,
                sum_hessian_bumped,
                ArrayArg::from_raw_parts(ranges, ranges_len),
                ArrayArg::from_raw_parts(roles, roles_len),
                split_slot,
                which,
                parent_count,
                n as u32,
            );
        },
        #[cfg(not(feature = "gpu"))]
        NumDataSrc::Device { .. } => {
            return Err(ComputeError::Runtime {
                detail: "find_best_splits_reduce (device num_data): the resident device-num_data \
                         scan requires a GPU backend"
                    .to_string(),
            });
        }
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
    // Per-grow cached descriptor set (LGBM_DESC_HOIST); `None` ⇒ byte-unchanged
    // per-launch assembly. A geometry mismatch is ignored (per-launch fallback).
    desc: Option<&ScanDescHandles>,
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
    // The cached `rf` is only trusted on an exact geometry match (same guard the
    // scan helper applies); the entry validated `real_feats.len() == feats.len()`.
    let desc_ok = desc.filter(|d| d.matches(feats.len(), buf_len));
    let scanned = fused_scan_to_raw_handle(
        client,
        hist_handle,
        buf_len,
        feats,
        cfg,
        sum_gradient,
        sum_hessian,
        NumDataSrc::Host(num_data),
        desc_ok,
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
            desc_ok.and_then(|d| d.h_rf.clone()),
        );
    }
    Ok(())
}

/// SPEC-DRGL-12: device-`num_data` twin of [`find_best_splits_fused_reduce_into_leaf_on`]
/// — the no-readback single-leaf scan+fold that resolves the child's `num_data` ON DEVICE
/// (from a [`DeviceLeafSplits`](crate::kernels::partition::DeviceLeafSplits)'s `ranges`/
/// `roles` + the host-known `parent_count`) instead of a host scalar. This is the launcher
/// the driver's deferred (SPEC-DRGL-05) child scan calls: the scan runs before the host
/// reads `split_point` back, folding the winner into frontier slot `out_leaf`. On hip it
/// dispatches the parprefix devcount twin, so the folded winner is byte-identical to the
/// non-deferred (host-count) parprefix fold. Real-device only.
///
/// # Errors
/// As [`find_best_splits_fused_reduce_into_leaf_on`]; plus a typed error if a non-GPU build
/// reaches the device path.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_reduce_into_leaf_devcount_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    sum_gradient: f64,
    sum_hessian: f64,
    ranges: cubecl::server::Handle,
    ranges_len: usize,
    roles: cubecl::server::Handle,
    roles_len: usize,
    split_slot: u32,
    which: u32,
    parent_count: i32,
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf: usize,
    desc: Option<&ScanDescHandles>,
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
                "find_best_splits_reduce_devcount: out_leaf {out_leaf} out of range [0, {})",
                out.len
            ),
        });
    }
    let desc_ok = desc.filter(|d| d.matches(feats.len(), buf_len));
    let scanned = fused_scan_to_raw_handle(
        client,
        hist_handle,
        buf_len,
        feats,
        cfg,
        sum_gradient,
        sum_hessian,
        NumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which,
            parent_count,
        },
        desc_ok,
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
            desc_ok.and_then(|d| d.h_rf.clone()),
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
    a_totals: (f64, f64, i32, f64),
    b_totals: (f64, f64, i32, f64),
    // SPEC-DRGL-13: `Host` ⇒ counts from a_totals/b_totals (byte-unchanged); `Device` ⇒ both
    // resolved on device in the scan kernel (parprefix siblings twin on hip, else legacy twin).
    num_data_src: SiblingNumDataSrc,
    // Per-grow cached descriptor set (LGBM_DESC_HOIST) — see `fused_scan_to_raw_handle`.
    desc: Option<&ScanDescHandles>,
) -> Result<Option<(cubecl::server::Handle, usize, usize, f64, f64)>, ComputeError> {
    if feats.is_empty() {
        return Ok(None);
    }
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
    let is_device = matches!(num_data_src, SiblingNumDataSrc::Device { .. });
    let (sum_gradient_a, sum_hessian_a, num_data_a, parent_output_a) = a_totals;
    let (sum_gradient_b, sum_hessian_b, num_data_b, parent_output_b) = b_totals;
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
    // whole-batch `na_as_missing` reject. A MATCHING per-grow cached set (`desc`) ran
    // this same validation + uploaded these same bytes once at cache time (bit-exact).
    let n = feats.len();
    let owned_desc: ScanDescHandles;
    let d: &ScanDescHandles = match desc {
        Some(d) if d.matches(n, buf_len) => {
            SCAN_DESC_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            d
        }
        _ => {
            owned_desc = upload_scan_desc(
                client,
                feats,
                None,
                buf_len,
                "find_best_splits_siblings_reduce",
            )?;
            &owned_desc
        }
    };

    // Per-sibling leaf scalars (the 2*kEpsilon bump + min_gain_shift) — IDENTICAL math.
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let use_l1 = cfg.use_l1();
    // The C++ `USE_SMOOTHING` template bool (`path_smooth > kEpsilon`),
    // resolved once per launch exactly like `USE_L1` above.
    let use_smoothing = cfg.use_smoothing();
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
    let h_slot = d.h_slot.clone();
    let h_numbin = d.h_numbin.clone();
    let h_offset = d.h_offset.clone();
    let h_defbin = d.h_defbin.clone();
    let h_skip = d.h_skip.clone();
    let h_rev = d.h_rev.clone();
    let h_fwd = d.h_fwd.clone();

    // STAGED cube-per-(feature, sibling) branch (opt-in `LGBM_SCAN_STAGED=1`) on
    // the LIVE per-split co-pack route — the hot path (one call per split). Same
    // gates as the single-slot helper; bit-identical output layout + cells.
    #[cfg(feature = "gpu")]
    if !is_device
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && d.staged_capable
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

    // SPEC-DRGL-13: the Device num_data source on a real device with parprefix enabled (the
    // flag-OFF default on hip) uses the co-pack parprefix kernel's device-`num_data` TWIN —
    // BYTE-IDENTICAL to the host parprefix co-pack scan, so a deferred co-pack fold reproduces
    // the non-deferred fold exactly. Falls back to the legacy siblings devcount twin below when
    // parprefix is off (non-default config).
    #[cfg(feature = "gpu")]
    if is_device
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && scan_parprefix_enabled(<R as cubecl::Runtime>::name(client))
        && d.staged_capable
    {
        SCAN_PARPREFIX_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let SiblingNumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which_a,
            which_b,
            parent_count,
        } = &num_data_src
        {
            // SAFETY: identical obligations to `launch_staged_siblings_scan`; geometry
            // CubeCount::Static(n, 2) matches the kernel's CUBE_POS_X<n, CUBE_POS_Y<2 contract.
            unsafe {
                find_best_splits_fused_siblings_staged_parprefix_kernel_devcount::launch(
                    client,
                    CubeCount::Static(n as u32, 2, 1),
                    CubeDim::new_1d(SCAN_STAGED_CUBE_DIM),
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
                    min_gain_shift_b,
                    sum_gradient_b,
                    sum_hessian_b_bumped,
                    ArrayArg::from_raw_parts(ranges.clone(), *ranges_len),
                    ArrayArg::from_raw_parts(roles.clone(), *roles_len),
                    *split_slot,
                    *which_a,
                    *which_b,
                    *parent_count,
                    n as u32,
                );
            }
        }
        return Ok(Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)));
    }

    // Direct `scan_cube_dim()` launch over 2*n feature-slots (A then B) — the SAME
    // `find_best_splits_fused_siblings_kernel` the host-readback path launches; only the
    // readback is dropped. SPEC-DRGL-13: a `Device` source that did NOT take the parprefix
    // twin (parprefix off) lands on the legacy siblings devcount twin here.
    let scan_w = scan_cube_dim();
    let cube_count = (2 * n as u32).div_ceil(scan_w);
    // SAFETY: both histogram handles describe `buf_len` f64 cells; every per-feature region
    // `[slot_off, slot_off+2*num_bin)` is validated `<= buf_len` above; all per-feature index
    // arrays have exactly `n` elements; lane `g` (guarded `< 2*n_feats`) reads only its
    // sibling's validated region and writes only `out[g*12 .. g*12+12]` within the `2*n*12`
    // allocation. All cubecl unsafe confined here (CMP-01).
    match num_data_src {
        SiblingNumDataSrc::Host => unsafe {
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
                cfg.max_delta_step,
                if use_smoothing { 1u32 } else { 0u32 },
                cfg.path_smooth,
                parent_output_a,
                parent_output_b,
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
        },
        #[cfg(feature = "gpu")]
        SiblingNumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which_a,
            which_b,
            parent_count,
        } => unsafe {
            SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            find_best_splits_fused_siblings_kernel_devcount::launch(
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
                cfg.max_delta_step,
                if use_smoothing { 1u32 } else { 0u32 },
                cfg.path_smooth,
                parent_output_a,
                parent_output_b,
                min_gain_shift_a,
                sum_gradient_a,
                sum_hessian_a_bumped,
                min_gain_shift_b,
                sum_gradient_b,
                sum_hessian_b_bumped,
                ArrayArg::from_raw_parts(ranges, ranges_len),
                ArrayArg::from_raw_parts(roles, roles_len),
                split_slot,
                which_a,
                which_b,
                parent_count,
                n as u32,
            );
        },
        #[cfg(not(feature = "gpu"))]
        SiblingNumDataSrc::Device { .. } => {
            return Err(ComputeError::Runtime {
                detail: "find_best_splits_siblings_reduce (device num_data): the resident \
                         device-num_data co-pack scan requires a GPU backend"
                    .to_string(),
            });
        }
    }
    Ok(Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)))
}

/// FUSED-SUBTRACT co-pack scan launcher: runs
/// [`find_best_splits_fused_siblings_subtract_staged_kernel`] — the subtraction trick
/// FOLDED into the co-pack sibling scan, so the caller can DROP the separate
/// `subtract_resident` launch. Sibling A scans `hist_smaller`; sibling B scans
/// `hist_parent − hist_smaller` and MATERIALIZES that derived larger histogram into a
/// FRESH `larger_out` handle (returned so the caller assigns it to the larger slot).
///
/// STAGED-ONLY: returns `Ok(None)` when the staged path is not taken (env off, non-real
/// device such as the cubecl-cpu anchor, PARGAIN opt-in, or a feature exceeding the LDS
/// stage cap), so the caller falls back to the separate `subtract_resident` +
/// [`fused_scan_siblings_to_raw_handle`] chain — byte-unchanged. On the taken path
/// returns `(h_out, out_len, n, min_gain_shift_a, min_gain_shift_b, larger_out)`.
///
/// # Errors
/// The SAME per-feature V5 validation / scope / `!(sum_hessian > 0.0)` errors as
/// [`fused_scan_siblings_to_raw_handle`].
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn fused_subtract_scan_siblings_to_raw_handle<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_smaller_handle: cubecl::server::Handle,
    hist_parent_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    cfg: &GainConfig,
    a_totals: (f64, f64, i32, f64),
    b_totals: (f64, f64, i32, f64),
    // SPEC-DRGL-05 deferral: `Host` ⇒ counts from a_totals/b_totals (byte-unchanged);
    // `Device` ⇒ both resolved on device — OFFICIAL-shape twin only (the live cuda
    // variant); a Device source outside the official arm is a typed error.
    num_data_src: SiblingNumDataSrc,
    // Per-grow cached descriptor set (LGBM_DESC_HOIST) — see `fused_scan_to_raw_handle`.
    desc: Option<&ScanDescHandles>,
) -> Result<
    Option<(cubecl::server::Handle, usize, usize, f64, f64, cubecl::server::Handle)>,
    ComputeError,
> {
    if feats.is_empty() {
        return Ok(None);
    }
    let is_device = matches!(num_data_src, SiblingNumDataSrc::Device { .. });
    // STAGED-capability gate (mirrors `fused_scan_siblings_to_raw_handle`'s staged
    // branch): real device, staged ON, NOT pargain (the fused kernel is a twin of the
    // SERIAL-branch staged kernel only), every feature ≤ the LDS stage cap. When any
    // fails, signal fallback to the separate subtract + scan.
    let staged_capable = scan_staged_enabled()
        && scan_variants_applicable(cfg)
        && !scan_pargain_enabled(<R as cubecl::Runtime>::name(client))
        && <R as cubecl::Runtime>::name(client) != "cpu"
        && feats.iter().all(|f| (f.num_bin as usize) * 2 <= SCAN_STAGE_MAX_CELLS);
    if !staged_capable {
        return Ok(None);
    }
    let (sum_gradient_a, sum_hessian_a, num_data_a, parent_output_a) = a_totals;
    let (sum_gradient_b, sum_hessian_b, num_data_b, parent_output_b) = b_totals;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_a > 0.0) || !(sum_hessian_b > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "find_best_splits_siblings_subtract: both siblings' sum_hessian must be > 0"
                .to_string(),
        });
    }

    // Per-feature V5 validation + device-array assembly (SHARED between siblings) —
    // IDENTICAL to `fused_scan_siblings_to_raw_handle`. A MATCHING per-grow cached set
    // (`desc`) ran this same validation + uploaded these same bytes once (bit-exact).
    let n = feats.len();
    let owned_desc: ScanDescHandles;
    let d: &ScanDescHandles = match desc {
        Some(d) if d.matches(n, buf_len) => {
            SCAN_DESC_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            d
        }
        _ => {
            owned_desc = upload_scan_desc(
                client,
                feats,
                None,
                buf_len,
                "find_best_splits_siblings_subtract",
            )?;
            &owned_desc
        }
    };

    let two_eps = 2.0 * f64::from(K_EPSILON);
    let use_l1 = cfg.use_l1();
    let sum_hessian_a_bumped = sum_hessian_a + two_eps;
    let min_gain_shift_a = crate::gain::get_leaf_gain(
        use_l1, sum_gradient_a, sum_hessian_a_bumped, cfg.lambda_l1, cfg.lambda_l2,
    ) + cfg.min_gain_to_split;
    let sum_hessian_b_bumped = sum_hessian_b + two_eps;
    let min_gain_shift_b = crate::gain::get_leaf_gain(
        use_l1, sum_gradient_b, sum_hessian_b_bumped, cfg.lambda_l1, cfg.lambda_l2,
    ) + cfg.min_gain_to_split;

    let out_len = 2 * n * 12;
    let h_out = client.empty(out_len * std::mem::size_of::<f64>());
    // The FRESH derived-larger buffer (`parent − smaller`), `buf_len` f64 cells — the
    // sibling-B cubes write every feature region (they tile `[0, buf_len)`).
    let larger_out = client.empty(buf_len * std::mem::size_of::<f64>());
    let h_slot = d.h_slot.clone();
    let h_numbin = d.h_numbin.clone();
    let h_offset = d.h_offset.clone();
    let h_defbin = d.h_defbin.clone();
    let h_skip = d.h_skip.clone();
    let h_rev = d.h_rev.clone();
    let h_fwd = d.h_fwd.clone();

    // SAFETY: geometry `Static(n, 2, 1)` guarantees CUBE_POS_X < n, CUBE_POS_Y < 2
    // (the kernel's no-guard contract). Both hist handles + `larger_out` describe
    // `buf_len` f64 cells; every per-feature region `[slot_off, slot_off+2*num_bin)` is
    // validated `<= buf_len`; sibling-B cubes write ONLY their feature region of
    // `larger_out` (disjoint across features, one lane per cell) and read `hist_parent` /
    // `hist_smaller` at the same region; `h_out` is `2*n*12`; per-feature arrays sized `n`.
    // `larger_out` is a FRESH handle (never aliases `hist_parent`). All cubecl unsafe
    // confined here (CMP-01).
    if scan_official_enabled(client) && scan_variants_applicable(cfg) {
        // OFFICIAL-SHAPE subtract twin (P1b) — same subtraction trick (sibling B
        // materializes `parent − smaller` into `larger_out`), but the branch scans
        // are the 256-wide block scans. 256 lanes + its own `plane_dim` arg.
        SCAN_OFFICIAL_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let plane_dim = client.properties().hardware.plane_size_max;
        if let SiblingNumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which_a,
            which_b,
            parent_count,
        } = &num_data_src
        {
            SCAN_NUMDATA_DEV_CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: identical obligations to the Host launch below; the devcount twin
            // additionally reads `ranges[6*split_slot+2]` / `roles[3*split_slot]`, both
            // validated within their handles by the caller (slot < capacity).
            unsafe {
                find_best_splits_fused_siblings_subtract_staged_official_kernel_devcount::launch(
                    client,
                    CubeCount::Static(n as u32, 2, 1),
                    CubeDim::new_1d(SCAN_OFFICIAL_CUBE_DIM),
                    ArrayArg::from_raw_parts(hist_smaller_handle, buf_len),
                    ArrayArg::from_raw_parts(hist_parent_handle, buf_len),
                    ArrayArg::from_raw_parts(larger_out.clone(), buf_len),
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
                    // Under the deferral, `a_totals`/`b_totals` are the LEFT/RIGHT
                    // children's totals (host-known from the pick export); the kernel
                    // role-selects the smaller/larger branch scalars on device.
                    min_gain_shift_a,
                    sum_gradient_a,
                    sum_hessian_a_bumped,
                    min_gain_shift_b,
                    sum_gradient_b,
                    sum_hessian_b_bumped,
                    ArrayArg::from_raw_parts(ranges.clone(), *ranges_len),
                    ArrayArg::from_raw_parts(roles.clone(), *roles_len),
                    *split_slot,
                    *parent_count,
                    n as u32,
                    plane_dim,
                );
            }
            return Ok(Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b, larger_out)));
        }
        unsafe {
            find_best_splits_fused_siblings_subtract_staged_official_kernel::launch(
                client,
                CubeCount::Static(n as u32, 2, 1),
                CubeDim::new_1d(SCAN_OFFICIAL_CUBE_DIM),
                ArrayArg::from_raw_parts(hist_smaller_handle, buf_len),
                ArrayArg::from_raw_parts(hist_parent_handle, buf_len),
                ArrayArg::from_raw_parts(larger_out.clone(), buf_len),
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
                plane_dim,
            );
        }
    } else if is_device {
        return Err(ComputeError::Runtime {
            detail: "fused_subtract_scan_siblings_to_raw_handle: Device num_data source \
                     requires the OFFICIAL-shape scan (scan_official_enabled) — the \
                     deferred loop must not mix scan variants"
                .to_string(),
        });
    } else {
        unsafe {
            find_best_splits_fused_siblings_subtract_staged_kernel::launch(
                client,
                CubeCount::Static(n as u32, 2, 1),
                CubeDim::new_1d(SCAN_STAGED_CUBE_DIM),
                ArrayArg::from_raw_parts(hist_smaller_handle, buf_len),
                ArrayArg::from_raw_parts(hist_parent_handle, buf_len),
                ArrayArg::from_raw_parts(larger_out.clone(), buf_len),
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
    Ok(Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b, larger_out)))
}

/// FUSED-SUBTRACT co-pack reduce-into-leaves: the subtraction trick folded into the
/// scan ([`fused_subtract_scan_siblings_to_raw_handle`]) followed by the batched
/// two-leaf reduce, folding BOTH siblings' winners into `out`. Returns
/// `Ok(Some(larger_out))` (the materialized derived-larger histogram Handle the caller
/// assigns to the larger slot) on the fused (staged) path, or `Ok(None)` when the
/// staged path is not taken — the caller then runs the separate `subtract_resident` +
/// [`find_best_splits_fused_siblings_reduce_into_leaves_on`] chain.
///
/// # Errors
/// As [`fused_subtract_scan_siblings_to_raw_handle`]; plus
/// [`ComputeError::LengthMismatch`] if `real_feats.len() != feats.len()`, or
/// [`ComputeError::Runtime`] if either `out_leaf` is out of range.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_subtract_reduce_into_leaves_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_smaller_handle: cubecl::server::Handle,
    hist_parent_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    a_totals: (f64, f64, i32, f64),
    b_totals: (f64, f64, i32, f64),
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf_a: usize,
    out_leaf_b: usize,
    // Per-grow cached descriptor set (LGBM_DESC_HOIST); `None` ⇒ byte-unchanged
    // per-launch assembly. A geometry mismatch is ignored (per-launch fallback).
    desc: Option<&ScanDescHandles>,
) -> Result<Option<cubecl::server::Handle>, ComputeError> {
    if real_feats.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: real_feats.len(),
        });
    }
    if !feats.is_empty() && (out_leaf_a >= out.len || out_leaf_b >= out.len) {
        return Err(ComputeError::Runtime {
            detail: format!(
                "find_best_splits_siblings_subtract_reduce: out_leaf_a {out_leaf_a} / \
                 out_leaf_b {out_leaf_b} out of range [0, {})",
                out.len
            ),
        });
    }
    let desc_ok = desc.filter(|d| d.matches(feats.len(), buf_len));
    let scanned = fused_subtract_scan_siblings_to_raw_handle(
        client,
        hist_smaller_handle,
        hist_parent_handle,
        buf_len,
        feats,
        cfg,
        a_totals,
        b_totals,
        SiblingNumDataSrc::Host,
        desc_ok,
    )?;
    let Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b, larger_out)) = scanned else {
        return Ok(None); // fallback signal (empty feats OR staged path not taken)
    };
    // ONE batched dispatch folds both siblings' winners (the same batched reduce the
    // non-fused co-pack path uses).
    let h_rf_cached = desc_ok.and_then(|d| d.h_rf.clone());
    // ONE batched dispatch folds both siblings in one CubeCount::Static(2,1,1) launch.
    launch_reduce_into_two_leaves(
        client, h_out, out_len, real_feats, n, out, out_leaf_a, out_leaf_b,
        min_gain_shift_a, min_gain_shift_b, h_rf_cached,
    );
    Ok(Some(larger_out))
}

/// SPEC-DRGL-05 deferral: device-`num_data` twin of
/// [`find_best_splits_fused_siblings_subtract_reduce_into_leaves_on`] — the
/// FUSED-SUBTRACT co-pack scan with BOTH children's counts resolved on device
/// from the resident `ranges`/`roles` record (OFFICIAL-shape kernel twin, the
/// live cuda variant — so a deferred fold uses the SAME scan variant as the
/// eager arm, bit-for-bit). Returns `Ok(Some(larger_out))` on the fused path or
/// `Ok(None)` when the staged/official gates don't hold (caller falls back to
/// the separate subtract + devcount co-scan, which is variant-consistent with
/// what the eager arm would use under those same gates).
///
/// # Errors
/// As [`find_best_splits_fused_siblings_subtract_reduce_into_leaves_on`]; plus
/// the typed Device-variant-mismatch error when official is enabled for the
/// scan but not applicable to `cfg`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_subtract_reduce_into_leaves_devcount_on<
    R: cubecl::Runtime,
>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_smaller_handle: cubecl::server::Handle,
    hist_parent_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    // LEFT/RIGHT children's totals (host-known from the pick export); the kernel
    // role-selects the smaller/larger branch on device.
    left_totals: (f64, f64, i32, f64),
    right_totals: (f64, f64, i32, f64),
    ranges: cubecl::server::Handle,
    ranges_len: usize,
    roles: cubecl::server::Handle,
    roles_len: usize,
    split_slot: u32,
    parent_count: i32,
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf_left: usize,
    out_leaf_right: usize,
    desc: Option<&ScanDescHandles>,
) -> Result<Option<cubecl::server::Handle>, ComputeError> {
    if real_feats.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: real_feats.len(),
        });
    }
    if !feats.is_empty() && (out_leaf_left >= out.len || out_leaf_right >= out.len) {
        return Err(ComputeError::Runtime {
            detail: format!(
                "subtract_reduce_devcount: out_leaf_left {out_leaf_left} / out_leaf_right \
                 {out_leaf_right} out of range [0, {})",
                out.len
            ),
        });
    }
    let desc_ok = desc.filter(|d| d.matches(feats.len(), buf_len));
    let scanned = fused_subtract_scan_siblings_to_raw_handle(
        client,
        hist_smaller_handle,
        hist_parent_handle,
        buf_len,
        feats,
        cfg,
        left_totals,
        right_totals,
        SiblingNumDataSrc::Device {
            ranges,
            ranges_len,
            roles: roles.clone(),
            roles_len,
            split_slot,
            // which fields are unused by the official which-aware twin (it
            // role-selects internally) but keep the Smaller/Larger convention.
            which_a: 2,
            which_b: 3,
            parent_count,
        },
        desc_ok,
    )?;
    let Some((h_out, out_len, n, min_gain_shift_left, min_gain_shift_right, larger_out)) = scanned
    else {
        return Ok(None);
    };
    let h_rf_cached = desc_ok.and_then(|d| d.h_rf.clone());
    launch_reduce_into_two_leaves_device_target(
        client, h_out, out_len, real_feats, n, out, roles, roles_len, split_slot,
        out_leaf_left, out_leaf_right, min_gain_shift_left, min_gain_shift_right, h_rf_cached,
    )?;
    Ok(Some(larger_out))
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
    a_totals: (f64, f64, i32, f64),
    b_totals: (f64, f64, i32, f64),
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf_a: usize,
    out_leaf_b: usize,
    // Per-grow cached descriptor set (LGBM_DESC_HOIST); `None` ⇒ byte-unchanged
    // per-launch assembly. A geometry mismatch is ignored (per-launch fallback).
    desc: Option<&ScanDescHandles>,
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
    let desc_ok = desc.filter(|d| d.matches(feats.len(), buf_len));
    let scanned = fused_scan_siblings_to_raw_handle(
        client,
        hist_a_handle,
        hist_b_handle,
        buf_len,
        feats,
        cfg,
        a_totals,
        b_totals,
        SiblingNumDataSrc::Host,
        desc_ok,
    )?;
    if let Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)) = scanned {
        let h_rf_cached = desc_ok.and_then(|d| d.h_rf.clone());
        // ONE batched dispatch folds BOTH siblings' winners (A: features [0, n) →
        // out_leaf_a; B: features [n, 2n) → out_leaf_b). Bit-identical to the two
        // separate `launch_reduce_into_leaf` calls it replaces (disjoint slots, same
        // per-slot fold), at half the reduce launch count — the spike095 host-enqueue
        // win. GPU-only: the batched kernel is `#[cfg(feature = "gpu")]`; the cpu
        // anchor never reaches this co-pack resident path (`resident_pool_supported()
        // == false`), so the two-call fallback is kept for the non-gpu build.
        #[cfg(feature = "gpu")]
        launch_reduce_into_two_leaves(
            client, h_out, out_len, real_feats, n, out, out_leaf_a, out_leaf_b,
            min_gain_shift_a, min_gain_shift_b, h_rf_cached,
        );
        #[cfg(not(feature = "gpu"))]
        {
            launch_reduce_into_leaf(
                client, h_out.clone(), out_len, real_feats, n, out, out_leaf_a, 0,
                min_gain_shift_a, h_rf_cached.clone(),
            );
            launch_reduce_into_leaf(
                client, h_out, out_len, real_feats, n, out, out_leaf_b, n * 12,
                min_gain_shift_b, h_rf_cached,
            );
        }
    }
    Ok(())
}

/// SPEC-DRGL-13: device-`num_data` twin of
/// [`find_best_splits_fused_siblings_reduce_into_leaves_on`] — the no-readback co-pack
/// scan+fold that resolves BOTH siblings' `num_data` ON DEVICE from ONE resident split/role
/// record (`split_slot`, `parent_count`) instead of the host `a_totals.2`/`b_totals.2`
/// scalars (which are IGNORED here — the sums are still used). This is the launcher the
/// driver's deferred (SPEC-DRGL-05) co-pack scan calls. On hip it dispatches the parprefix
/// siblings devcount twin, so both folded winners are byte-identical to the non-deferred
/// (host-count) parprefix folds. Real-device only.
///
/// # Errors
/// As [`find_best_splits_fused_siblings_reduce_into_leaves_on`]; plus a typed error if a
/// non-GPU build reaches the device path.
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_fused_siblings_reduce_into_leaves_devcount_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist_a_handle: cubecl::server::Handle,
    hist_b_handle: cubecl::server::Handle,
    buf_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    // (sum_gradient, sum_hessian, _num_data) per sibling — the num_data component is IGNORED
    // (resolved on device); the sums drive min_gain_shift + the child-output finalization.
    a_totals: (f64, f64, i32, f64),
    b_totals: (f64, f64, i32, f64),
    ranges: cubecl::server::Handle,
    ranges_len: usize,
    roles: cubecl::server::Handle,
    roles_len: usize,
    split_slot: u32,
    which_a: u32,
    which_b: u32,
    parent_count: i32,
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf_a: usize,
    out_leaf_b: usize,
    desc: Option<&ScanDescHandles>,
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
                "find_best_splits_siblings_reduce_devcount: out_leaf_a {out_leaf_a} / out_leaf_b \
                 {out_leaf_b} out of range [0, {})",
                out.len
            ),
        });
    }
    let desc_ok = desc.filter(|d| d.matches(feats.len(), buf_len));
    let scanned = fused_scan_siblings_to_raw_handle(
        client,
        hist_a_handle,
        hist_b_handle,
        buf_len,
        feats,
        cfg,
        a_totals,
        b_totals,
        SiblingNumDataSrc::Device {
            ranges,
            ranges_len,
            roles,
            roles_len,
            split_slot,
            which_a,
            which_b,
            parent_count,
        },
        desc_ok,
    )?;
    if let Some((h_out, out_len, n, min_gain_shift_a, min_gain_shift_b)) = scanned {
        let h_rf_cached = desc_ok.and_then(|d| d.h_rf.clone());
        // Identical fold to the host launcher (num_data-independent).
        #[cfg(feature = "gpu")]
        launch_reduce_into_two_leaves(
            client, h_out, out_len, real_feats, n, out, out_leaf_a, out_leaf_b,
            min_gain_shift_a, min_gain_shift_b, h_rf_cached,
        );
        #[cfg(not(feature = "gpu"))]
        {
            launch_reduce_into_leaf(
                client, h_out.clone(), out_len, real_feats, n, out, out_leaf_a, 0,
                min_gain_shift_a, h_rf_cached.clone(),
            );
            launch_reduce_into_leaf(
                client, h_out, out_len, real_feats, n, out, out_leaf_b, n * 12,
                min_gain_shift_b, h_rf_cached,
            );
        }
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
/// Same as [`find_best_split_cpu`] (V5 validation; `na_as_missing` is admitted,
/// see the REVERSE/FORWARD sections below; `max_delta_step`/`path_smooth`,
/// individually or combined, are fully transcribed via the two-axis gain
/// primitives).
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
    use crate::gain::{
        calculate_splitted_leaf_output_full as calc_out, get_leaf_gain_full, get_split_gains,
        get_split_gains_full,
    };

    // ---- V5 boundary validation (identical to find_best_split_cpu) ----
    // NA_AS_MISSING (feature_histogram.hpp:945-961, T-G4-1) is now transcribed
    // below (REVERSE excludes the top/NaN-sentinel bin; FORWARD folds bin 0 in
    // via the na_preamble when offset==1). The gate that used to reject it here
    // has been removed for this host target only (P-4).
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
    // max_delta_step and path_smooth (individually or combined) are fully
    // transcribed below via the two-axis gain primitives — no rejection needed.
    // ---- host pre-step: 2*kEpsilon bump + min_gain_shift (verbatim) ----
    let two_eps = 2.0 * f64::from(K_EPSILON);
    let sum_hessian_bumped = sum_hessian + two_eps;
    let use_l1 = cfg.use_l1();
    // The C++ `USE_MAX_OUTPUT` / `USE_SMOOTHING` template bools, resolved once per
    // feature histogram from the config (feature_histogram.hpp:248-270).
    let use_smoothing = cfg.use_smoothing();
    let max_delta_step = cfg.max_delta_step;
    let parent_output = cfg.parent_output;
    // C++ resolves USE_MAX_OUTPUT/USE_SMOOTHING as TEMPLATE bools — the default
    // instantiation never touches the clamp/smooth chain. `get_split_gains_full`'s
    // branchless select-form (required for the `#[cube]` device twin) computes that
    // whole chain (~5 extra f64 divides per side) and discards it on the default
    // path, which regressed the host scan ~7x. Dispatch ONCE per scan instead:
    // `get_leaf_gain_full` returns exactly `get_leaf_gain`'s closed form when both
    // axes are off, so the fast-path call is bit-identical.
    let use_full = use_smoothing || max_delta_step > 0.0;
    // `BeforeNumerical` (feature_histogram.hpp:192-207) uses the SAME
    // `GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>` instantiation as the
    // per-candidate scan, over the WHOLE-leaf sums and the leaf's own row count —
    // so under smoothing the baseline is the smoothed whole-leaf gain, not the
    // closed form.
    let gain_shift = get_leaf_gain_full(
        use_l1,
        sum_gradient,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
        max_delta_step,
        use_smoothing,
        cfg.path_smooth,
        num_data,
        parent_output,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let num_bin_i = num_bin as i32;
    // REVERSE excludes the top/NaN-sentinel bin when na_as_missing (:859).
    let rev_count = (num_bin_i - 1 - i32::from(na_as_missing)).max(0);
    // FORWARD preamble (:945-961) applies ONLY when offset==1.
    let na_preamble = na_as_missing && offset == 1;
    let fwd_count = if run_forward {
        let base = (num_bin_i - 1 - offset).max(0);
        if na_preamble { base + 1 } else { base }
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

    // Best-split running state. `best_gain` is C++ `kMinScore` (-inf), NOT 0.0: under
    // `max_delta_step` / `path_smooth` the given-output gain form is freely negative,
    // so a 0.0 sentinel silently discards every negative-gain candidate (see the
    // matching note in `split_scan_body`). Plain Rust has no literal-init restriction,
    // so -inf is used directly here.
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = false;
    let mut best_default_left = true; // REVERSE default

    // ===================== REVERSE branch (:854-936) =====================
    {
        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = f64::from(K_EPSILON);
        let mut right_count = 0i32;
        // Excludes the top bin (the NaN sentinel `na_as_missing()` routes NaN
        // rows to) from the REVERSE sweep when na_as_missing — it is implicitly
        // folded into "left" via `sum_left = sum_gradient - sum_right` for every
        // REVERSE candidate (feature_histogram.hpp:859).
        let t_start = num_bin_i - 1 - offset - i32::from(na_as_missing);
        let mut done = false;
        for k in 0..rev_count {
            // `done` is sticky and every later iteration is fully inert (sums
            // untouched, no candidate considered) — breaking out is behaviorally
            // identical and matches the C++ loop's `break` (:c++ feature_histogram).
            if done {
                break;
            }
            let t = t_start - k;
            let in_range = t >= (1 - offset);
            let skip = skip_default_bin && (t + offset) == default_bin as i32;
            let active = in_range && !skip && !done;
            if active {
                // `active` implies `in_range` (t >= 1-offset >= 0 for offset∈{0,1}).
                let bi = (t as usize) * 2;
                // SAFETY: `t <= num_bin - 1 - offset` by `t_start` construction, so
                // `bi + 1 <= 2*num_bin - 1 < hist.len()` (validated above).
                debug_assert!(bi + 1 < hist.len());
                let (hg, hh) = unsafe { (*hist.get_unchecked(bi), *hist.get_unchecked(bi + 1)) };
                sum_right_gradient += hg;
                sum_right_hessian += hh;
                right_count += round_int(hh * cnt_factor);
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
                let current_gain = if use_full {
                    get_split_gains_full(
                        use_l1,
                        sum_left_gradient,
                        sum_left_hessian,
                        sum_right_gradient,
                        sum_right_hessian,
                        l1,
                        l2,
                        max_delta_step,
                        use_smoothing,
                        cfg.path_smooth,
                        left_count,
                        right_count,
                        parent_output,
                    )
                } else {
                    // Default axes off — bit-identical closed form (see `use_full`).
                    get_split_gains(
                        use_l1,
                        sum_left_gradient,
                        sum_left_hessian,
                        sum_right_gradient,
                        sum_right_hessian,
                        l1,
                        l2,
                    )
                };
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

        // NA_AS_MISSING FORWARD preamble (:945-961): ONLY when offset==1 (bin 0,
        // the implicit most-frequent bin, is NOT stored explicitly in `hist`).
        // Reconstruct bin 0 by pre-seeding the accumulators with the FULL leaf
        // totals, then subtracting every EXPLICIT bin in ascending order — bit-
        // exact SEQUENTIAL subtraction (`sum_gradient - g0 - g1 - ...`), NOT a
        // sum-then-subtract-once shortcut, matching C++'s loop-accumulated
        // operation order exactly.
        let fwd_start = if na_preamble {
            sum_left_gradient = sum_gradient;
            sum_left_hessian = sum_hessian_bumped - f64::from(K_EPSILON);
            left_count = num_data;
            for i in 0..(num_bin_i - offset) {
                let bi = (i as usize) * 2;
                sum_left_gradient -= hist[bi];
                sum_left_hessian -= hist[bi + 1];
                left_count -= round_int(hist[bi + 1] * cnt_factor);
            }
            -1i32
        } else {
            0i32
        };

        let mut done = false;
        for k in 0..fwd_count {
            // Sticky-`done` early exit — see the REVERSE branch note.
            if done {
                break;
            }
            let t = fwd_start + k;
            let skip = skip_default_bin && (t + offset) == default_bin as i32;
            let active = !skip && !done;
            // C++ `if (t >= 0) { sum_left_gradient += ...; }` (:969) — the virtual
            // `t=-1` preamble candidate adds nothing here (already folded above).
            if active && t >= 0 {
                let bi = (t as usize) * 2;
                // SAFETY: `t <= fwd_start + fwd_count - 1 <= num_bin - 1 - offset`,
                // so `bi + 1 < 2*num_bin == hist.len()` (validated above).
                debug_assert!(bi + 1 < hist.len());
                let (hg, hh) = unsafe { (*hist.get_unchecked(bi), *hist.get_unchecked(bi + 1)) };
                sum_left_gradient += hg;
                sum_left_hessian += hh;
                left_count += round_int(hh * cnt_factor);
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
                let current_gain = if use_full {
                    get_split_gains_full(
                        use_l1,
                        sum_left_gradient,
                        sum_left_hessian,
                        sum_right_gradient,
                        sum_right_hessian,
                        l1,
                        l2,
                        max_delta_step,
                        use_smoothing,
                        cfg.path_smooth,
                        left_count,
                        right_count,
                        parent_output,
                    )
                } else {
                    // Default axes off — bit-identical closed form (see `use_full`).
                    get_split_gains(
                        use_l1,
                        sum_left_gradient,
                        sum_left_hessian,
                        sum_right_gradient,
                        sum_right_hessian,
                        l1,
                        l2,
                    )
                };
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
    // The winner's per-side outputs use that side's OWN row count for the
    // smoothing weight (feature_histogram.hpp:1034-1049).
    let best_right_count = num_data - best_left_count;
    let left_output = calc_out(
        use_l1,
        best_sum_left_gradient,
        best_sum_left_hessian,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        cfg.path_smooth,
        best_left_count,
        parent_output,
    );
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian_bumped - best_sum_left_hessian;
    let right_output = calc_out(
        use_l1,
        right_sum_gradient,
        right_sum_hessian,
        l1,
        l2,
        max_delta_step,
        use_smoothing,
        cfg.path_smooth,
        best_right_count,
        parent_output,
    );

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

    /// Bit-compare two `SplitInfo`s: every f64 field by its raw bit pattern (so a
    /// signed-zero or any ULP difference is caught), every integer/flag by `==`.
    fn split_info_bit_eq(a: &SplitInfo, b: &SplitInfo) -> bool {
        a.threshold == b.threshold
            && a.gain.to_bits() == b.gain.to_bits()
            && a.left_count == b.left_count
            && a.right_count == b.right_count
            && a.left_sum_gradient.to_bits() == b.left_sum_gradient.to_bits()
            && a.left_sum_hessian.to_bits() == b.left_sum_hessian.to_bits()
            && a.right_sum_gradient.to_bits() == b.right_sum_gradient.to_bits()
            && a.right_sum_hessian.to_bits() == b.right_sum_hessian.to_bits()
            && a.left_output.to_bits() == b.left_output.to_bits()
            && a.right_output.to_bits() == b.right_output.to_bits()
            && a.default_left == b.default_left
    }

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

    /// G5-2 (SPEC-G5-2): non-default `max_delta_step` must no longer be rejected
    /// by [`find_best_split_cpu_native`], and the returned leaf outputs must
    /// cross-check against [`crate::gain::calculate_splitted_leaf_output_clamped`]
    /// computed independently from the returned per-side sums (the internal RAW
    /// sums the scan used, i.e. the reported `left_sum_hessian`/`right_sum_hessian`
    /// with `kEpsilon` added back, per the `:1042/:1053` doc note on
    /// [`SplitInfo`]).
    #[test]
    fn find_best_split_cpu_native_admits_max_delta_step() {
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0];
        let sum_gradient: f64 = -10.0 - 8.0 + 9.0 + 8.0;
        let sum_hessian: f64 = 20.0;
        let num_data = 20i32;
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.7,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let si = find_best_split_cpu_native(
            &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient, sum_hessian,
            num_data,
        )
        .expect("max_delta_step must no longer be rejected");
        assert!(si.gain.is_finite(), "expected a finite-gain split, got {si:?}");
        let use_l1 = cfg.use_l1();
        let eps = f64::from(K_EPSILON);
        let expected_left = crate::gain::calculate_splitted_leaf_output_clamped(
            use_l1,
            si.left_sum_gradient,
            si.left_sum_hessian + eps,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.max_delta_step,
        );
        let expected_right = crate::gain::calculate_splitted_leaf_output_clamped(
            use_l1,
            si.right_sum_gradient,
            si.right_sum_hessian + eps,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.max_delta_step,
        );
        assert_eq!(si.left_output, expected_left);
        assert_eq!(si.right_output, expected_right);
        // Sanity: the clamp is actually exercised (both leaf outputs are within
        // the [-0.7, 0.7] envelope) for this fixture, not silently falling
        // through to the unclamped base path.
        assert!(si.left_output.abs() <= 0.7 + 1e-12, "{}", si.left_output);
        assert!(si.right_output.abs() <= 0.7 + 1e-12, "{}", si.right_output);
    }

    /// G5-2 (SPEC-G5-2): the KERNEL host target ([`find_best_split_cpu`], which
    /// dispatches to [`find_best_split_f64_on`]) must ALSO admit non-default
    /// `max_delta_step`, and — being the SAME shared [`split_scan_body`] the
    /// native fn's independent transcription is checked against — must produce
    /// the IDENTICAL `SplitInfo` as [`find_best_split_cpu_native`] on the same
    /// inputs (the existing kernel-vs-native parity idiom this file already uses
    /// for na_as_missing).
    #[test]
    fn find_best_split_cpu_admits_max_delta_step_matches_native() {
        let client = cpu_client();
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0];
        let sum_gradient: f64 = -10.0 - 8.0 + 9.0 + 8.0;
        let sum_hessian: f64 = 20.0;
        let num_data = 20i32;
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.7,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let si_kernel = find_best_split_cpu(
            &client, &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient,
            sum_hessian, num_data,
        )
        .expect("max_delta_step must no longer be rejected by the kernel host target");
        let si_native = find_best_split_cpu_native(
            &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient, sum_hessian,
            num_data,
        )
        .expect("native");
        assert_eq!(si_kernel, si_native, "kernel vs native must be bit-identical");
        assert!(si_kernel.left_output.abs() <= 0.7 + 1e-12);
        assert!(si_kernel.right_output.abs() <= 0.7 + 1e-12);
    }

    /// G5-3 (SPEC-G5-3): non-default `path_smooth` (+ a fixed `parent_output`)
    /// must no longer be rejected by [`find_best_split_cpu_native`], and the
    /// returned leaf outputs must cross-check against
    /// [`crate::gain::calculate_splitted_leaf_output_smoothed`] computed
    /// independently from the returned per-side sums + counts (the internal RAW
    /// sums the scan used, i.e. the reported `left_sum_hessian`/`right_sum_hessian`
    /// with `kEpsilon` added back) — the idiom the plan's Red step describes: "a
    /// cross-check against the already-tested gain.rs function, not a fresh
    /// hand-derivation".
    #[test]
    fn find_best_split_cpu_native_admits_path_smooth() {
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0];
        let sum_gradient: f64 = -10.0 - 8.0 + 9.0 + 8.0;
        let sum_hessian: f64 = 20.0;
        let num_data = 20i32;
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 2.0,
            parent_output: 0.25,
            ..Default::default()
        };
        let si = find_best_split_cpu_native(
            &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient, sum_hessian,
            num_data,
        )
        .expect("path_smooth must no longer be rejected");
        assert!(si.gain.is_finite(), "expected a finite-gain split, got {si:?}");
        let use_l1 = cfg.use_l1();
        let eps = f64::from(K_EPSILON);
        let expected_left = crate::gain::calculate_splitted_leaf_output_smoothed(
            use_l1,
            si.left_sum_gradient,
            si.left_sum_hessian + eps,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.path_smooth,
            si.left_count,
            cfg.parent_output,
        );
        let expected_right = crate::gain::calculate_splitted_leaf_output_smoothed(
            use_l1,
            si.right_sum_gradient,
            si.right_sum_hessian + eps,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.path_smooth,
            si.right_count,
            cfg.parent_output,
        );
        assert_eq!(si.left_output, expected_left);
        assert_eq!(si.right_output, expected_right);
    }

    /// G5-3: the KERNEL host target ([`find_best_split_cpu`]) must ALSO admit
    /// non-default `path_smooth`, and must produce the IDENTICAL `SplitInfo` as
    /// [`find_best_split_cpu_native`] on the same inputs.
    #[test]
    fn find_best_split_cpu_admits_path_smooth_matches_native() {
        let client = cpu_client();
        let num_bin = 4u32;
        let hist: Vec<f64> = vec![-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0];
        let sum_gradient: f64 = -10.0 - 8.0 + 9.0 + 8.0;
        let sum_hessian: f64 = 20.0;
        let num_data = 20i32;
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 2.0,
            parent_output: 0.25,
            ..Default::default()
        };
        let si_kernel = find_best_split_cpu(
            &client, &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient,
            sum_hessian, num_data,
        )
        .expect("path_smooth must no longer be rejected by the kernel host target");
        let si_native = find_best_split_cpu_native(
            &hist, &cfg, num_bin, 0, num_bin, 0, false, false, true, sum_gradient, sum_hessian,
            num_data,
        )
        .expect("native");
        assert_eq!(si_kernel, si_native, "kernel vs native must be bit-identical");
    }

    /// `max_delta_step` and `path_smooth` non-default SIMULTANEOUSLY are BOTH
    /// admitted and composed correctly (clamp first, THEN the smoothing blend —
    /// `feature_histogram.hpp:715-737`), via the two-axis `*_full` gain
    /// primitives — no rejection. Cross-checks the winner's leaf outputs against
    /// `calculate_splitted_leaf_output_full` computed independently from the
    /// returned per-side sums + counts, the SAME idiom the single-axis tests
    /// above use.
    #[test]
    fn find_best_split_cpu_native_admits_combined_clamp_and_smooth() {
        let hist: Vec<f64> = vec![-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0];
        let sum_gradient: f64 = -10.0 - 8.0 + 9.0 + 8.0;
        let sum_hessian: f64 = 20.0;
        let num_data = 20i32;
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.5,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 2.0,
            parent_output: 0.25,
            ..Default::default()
        };
        let si = find_best_split_cpu_native(
            &hist, &cfg, 4, 0, 4, 0, false, false, true, sum_gradient, sum_hessian, num_data,
        )
        .expect("max_delta_step and path_smooth combined must both be admitted");
        assert!(si.gain.is_finite(), "expected a finite-gain split, got {si:?}");

        let use_l1 = cfg.use_l1();
        let use_smoothing = cfg.use_smoothing();
        let expected_left = crate::gain::calculate_splitted_leaf_output_full(
            use_l1,
            si.left_sum_gradient,
            si.left_sum_hessian,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.max_delta_step,
            use_smoothing,
            cfg.path_smooth,
            si.left_count,
            cfg.parent_output,
        );
        let expected_right = crate::gain::calculate_splitted_leaf_output_full(
            use_l1,
            si.right_sum_gradient,
            si.right_sum_hessian,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.max_delta_step,
            use_smoothing,
            cfg.path_smooth,
            si.right_count,
            cfg.parent_output,
        );
        assert_eq!(si.left_output, expected_left);
        assert_eq!(si.right_output, expected_right);
        // The clamp must have ACTUALLY bound (`max_delta_step=0.5` on outputs
        // that would otherwise exceed it) for this to be a real exercise of the
        // composed path, not a no-op clamp with only smoothing doing anything.
        assert!(
            si.left_output.abs() <= 0.5 + 1e-9 && si.right_output.abs() <= 0.5 + 1e-9,
            "expected both outputs clamped to ±0.5, got {si:?}"
        );
    }

    /// UPDATED (T-G4-1, was `find_best_split_na_as_missing_is_typed_error`):
    /// `na_as_missing == true` is now ADMITTED and COMPUTED by
    /// [`find_best_split_cpu`] (the `find_best_split_f64_on`/kernel-parity host
    /// target), not rejected — SPEC-G4-1. `offset == 0` here, so the FORWARD
    /// preamble (`na_preamble`, T-G4-1) is inactive; only the REVERSE
    /// top-bin-exclusion applies. Hand-computed from `feature_histogram.hpp:
    /// 830-1057` against this histogram (bins 0..3, `offset=0`, bin 3 is the
    /// `na_as_missing` sentinel bin):
    ///   bin0 g=-10 h=5, bin1 g=-8 h=5, bin2 g=9 h=5, bin3(NaN) g=8 h=5.
    /// REVERSE (excludes bin3 from its sweep, `t_start = num_bin-1-offset-1 =
    /// 2`) best candidate is `t=1` (threshold=1, gain≈22.87); FORWARD (offset=0
    /// ⇒ no preamble, unmodified `t=0..2` sweep) best candidate is `t=1`
    /// (`left={bin0,bin1}`, `right={bin2,bin3(NaN)}`, gain≈61.3) — FORWARD wins,
    /// so the winning split's `default_left == false` (NaN routes to the RIGHT
    /// child, i.e. with `bin2`/`bin3`).
    #[test]
    fn find_best_split_na_as_missing_offset0_admits_and_computes() {
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
        let split = find_best_split_cpu(
            &client,
            &hist,
            &cfg,
            num_bin,
            0,
            num_bin,
            0,
            false, // skip_default_bin
            true,  // na_as_missing -> now admitted (T-G4-1)
            true,  // run_forward -> true for missing_type != None (checker Issue 2 fix)
            -1.0,
            20.0,
            20,
        )
        .expect("na_as_missing must now be admitted and computed, not a typed error");
        assert_eq!(split.threshold, 1, "FORWARD t=1 wins over REVERSE t=1");
        assert!(
            !split.default_left,
            "FORWARD branch wins => NaN routes RIGHT (default_left=false)"
        );
        assert_eq!(split.left_count, 10);
        assert_eq!(split.right_count, 10);
        assert_eq!(split.left_sum_gradient, -18.0);
        assert_eq!(split.right_sum_gradient, 17.0);
        assert!(
            (split.left_sum_hessian - 10.0).abs() < 1e-6,
            "left_sum_hessian: {}",
            split.left_sum_hessian
        );
        assert!(
            (split.right_sum_hessian - 10.0).abs() < 1e-6,
            "right_sum_hessian: {}",
            split.right_sum_hessian
        );
        assert!(
            (split.gain - 61.25).abs() < 1e-2,
            "gain: {} (expected ~61.25 = 61.3 - min_gain_shift(~0.05))",
            split.gain
        );
    }

    /// T-G4-1 Red: the NA_AS_MISSING FORWARD preamble (`feature_histogram.hpp:
    /// 945-961`), exercised at `offset == 1` where bin 0 (the implicit
    /// most-frequent bin) must be reconstructed via subtraction and folded into
    /// the FIRST forward candidate. Checker Issue 2: this scenario is
    /// constructed so the CORRECT winning split comes from the FORWARD branch
    /// (and specifically from the preamble-seeded `t=0` candidate, which needs
    /// bin 0 correctly reconstructed) — asserting merely `Ok(..)` would NOT
    /// catch a reverse-only-scan regression, so this test also runs the SAME
    /// histogram with `run_forward=false` (the pre-fix `run_forward()` truth
    /// table for a NaN feature) and asserts it yields a DIFFERENT, WRONG
    /// `default_left` — proving the forward branch is load-bearing for
    /// correctness, not merely for coverage.
    ///
    /// Histogram (`num_bin=4, offset=1`; bin0 implicit/most-frequent, bin3 is
    /// the `na_as_missing` sentinel): bin0 g=-1 h=1 (reconstructed via
    /// subtraction), bin1(data_[0]) g=-1 h=1, bin2(data_[1]) g=1 h=1,
    /// bin3/NaN(data_[2]) g=1 h=1. Hand-computed from `feature_histogram.hpp:
    /// 830-1057`:
    ///   - REVERSE (excludes bin3, `t_start=1`): best is `t=1`
    ///     (threshold=1, left={bin0,bin1,bin3(NaN, implicit)}, gain≈1.333).
    ///   - FORWARD (`na_preamble` seeds bin0 via subtraction, virtual `t=-1`
    ///     then `t=0,1`): best is `t=0` (threshold=1, left={bin0,bin1},
    ///     right={bin2,bin3(NaN)}, gain≈4.0) — FORWARD wins.
    /// So the TRUE winner is `threshold=1, default_left=false, gain≈4.0`; a
    /// buggy reverse-only scan (`run_forward=false`) would instead report
    /// `threshold=1, default_left=true, gain≈1.333` — same nominal threshold,
    /// OPPOSITE (wrong) NaN routing and a much lower gain.
    #[test]
    fn find_best_split_na_as_missing_offset1_forward_preamble_wins() {
        let client = cpu_client();
        let num_bin = 4u32;
        // COMPACTED layout (`learner::compact_histogram`): cell `c` holds REAL
        // bin `c+offset`'s data; the buffer keeps its full `2*num_bin` length
        // with the dropped bin-0 slot ZEROED at the tail (never read directly —
        // reconstructed via subtraction in the `na_preamble`).
        // cell0=real bin1, cell1=real bin2, cell2=real bin3(NaN), cell3=padding.
        let hist: Vec<f64> = vec![-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let cfg = GainConfig {
            min_data_in_leaf: 0,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let correct = find_best_split_cpu(
            &client,
            &hist,
            &cfg,
            num_bin,
            1, // offset=1 -> exercises the FORWARD na_preamble
            0,
            0,
            false, // skip_default_bin
            true,  // na_as_missing
            true,  // run_forward=true -- the checker Issue 2 fix in action
            0.0,
            4.0,
            4,
        )
        .expect("na_as_missing must be admitted and computed (T-G4-1)");
        assert_eq!(correct.threshold, 1);
        assert!(
            !correct.default_left,
            "FORWARD's preamble-seeded t=0 candidate must win: NaN routes RIGHT"
        );
        assert_eq!(correct.left_count, 2);
        assert_eq!(correct.right_count, 2);
        assert_eq!(correct.left_sum_gradient, -2.0);
        assert_eq!(correct.right_sum_gradient, 2.0);
        assert!(
            (correct.gain - 4.0).abs() < 1e-6,
            "gain: {} (expected ~4.0)",
            correct.gain
        );

        // Companion: the PRE-FIX `run_forward()` truth table (Zero-only) would
        // have passed `run_forward=false` for this NaN feature, running REVERSE
        // ONLY. Prove that gives a DIFFERENT (wrong) answer — the same nominal
        // threshold but the OPPOSITE `default_left` and a much lower gain —
        // demonstrating why checker Issue 2's `run_forward()` fix is
        // load-bearing, not cosmetic.
        let reverse_only = find_best_split_cpu(
            &client, &hist, &cfg, num_bin, 1, 0, 0, false, true,
            false, // run_forward=false: the PRE-FIX (buggy) dispatch for NaN
            0.0, 4.0, 4,
        )
        .expect("REVERSE-only must still succeed (it's a valid, just WRONG, split)");
        assert_eq!(
            reverse_only.threshold, 1,
            "same nominal threshold as the correct answer"
        );
        assert!(
            reverse_only.default_left,
            "REVERSE-only wrongly folds NaN into LEFT (opposite of the true answer)"
        );
        assert!(
            (reverse_only.gain - 1.333_333).abs() < 1e-3,
            "gain: {} (expected ~1.333, well below the true 4.0)",
            reverse_only.gain
        );
        assert_ne!(
            correct.default_left, reverse_only.default_left,
            "forward-branch inclusion changes the winning split's NaN routing"
        );
    }

    /// T-G4-1 (P-4 dual-target requirement): [`find_best_split_cpu_native`] (the
    /// `CpuBackend` production path) and [`find_best_split_cpu`] (the
    /// `find_best_split_f64_on` cubecl-cpu kernel-parity/ROCm-mirror path) MUST
    /// stay bit-identical for `na_as_missing == true` too, exactly as they
    /// already are for every other parameter combination
    /// (`split_2lane_equals_serial_matrix` sweeps the non-`na_as_missing` space).
    /// Sweeps both `offset ∈ {0,1}` (only `offset==1` exercises the FORWARD
    /// `na_preamble`) over the two hand-built histograms above.
    #[test]
    fn find_best_split_na_as_missing_native_matches_kernel() {
        let client = cpu_client();
        let cfg = GainConfig {
            min_data_in_leaf: 0,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let cases: [(u32, i32, &[f64], f64, f64, i32); 2] = [
            (4, 0, &[-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0], -1.0, 20.0, 20),
            (4, 1, &[-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], 0.0, 4.0, 4),
        ];
        for (num_bin, offset, hist, sum_gradient, sum_hessian, num_data) in cases {
            let native = find_best_split_cpu_native(
                hist, &cfg, num_bin, offset, 0, 0, false, true, true, sum_gradient, sum_hessian,
                num_data,
            )
            .expect("native: na_as_missing must be admitted");
            let kernel = find_best_split_cpu(
                &client, hist, &cfg, num_bin, offset, 0, 0, false, true, true, sum_gradient,
                sum_hessian, num_data,
            )
            .expect("kernel: na_as_missing must be admitted");
            assert!(
                split_info_bit_eq(&native, &kernel),
                "native/kernel diverged at offset={offset}.\n native={native:?}\n kernel={kernel:?}"
            );
        }
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
        launch_reduce_into_leaf(&client, h_raw, cells.len(), &real_feats, 4, &out, 0, 0, 0.0, None);

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
            &client, h_hist2, buf.len(), &feats, &real_feats, &cfg, sg, sh, nd, &out, 0, None,
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
            &client, h_hist2, buf.len(), &feats, &real_feats, &cfg, sg, sh, nd, &out, 0, None,
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
        launch_reduce_into_leaf(&client, h_raw.clone(), cells.len(), &real_feats, 4, &out, 0, 0, 0.0, None);
        launch_reduce_into_leaf(&client, h_raw, cells.len(), &real_feats, 4, &out, 1, 4 * 12, 0.0, None);

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
        // The 4th tuple slot is the sibling's `parent_output`; 0.0 is inert here
        // because `relaxed_cfg()` leaves `path_smooth` at its no-smoothing default.
        let a_totals = (sg_a, sh_a, nd_a, 0.0);
        let b_totals = (sg_b, sh_b, nd_b, 0.0);

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
            &client, h_a2, h_b2, buf_a.len(), &feats, &real_feats, &cfg, a_totals, b_totals,
            &out, 0, 1, None,
        )
        .expect("device co-pack reduce launcher");
        let got_a = out.read_record(&client, 0);
        let got_b = out.read_record(&client, 1);
        assert_slot_eq(&got_a, &host_a, real_feats[fpos_a as usize]);
        assert_slot_eq(&got_b, &host_b, real_feats[fpos_b as usize]);
    }

    /// The BATCHED two-leaf reduce ([`launch_reduce_into_two_leaves`], ONE dispatch)
    /// writes BIT-IDENTICAL frontier slots to TWO separate [`launch_reduce_into_leaf`]
    /// calls — pinning the "batching changes nothing" invariant directly (guards
    /// against the two kernels drifting apart in future edits). Distinct per-sibling
    /// winners + a distinct `min_gain_shift` per sibling exercise both tasks' param
    /// selection. GPU-gated (the batched kernel is `#[cfg(feature = "gpu")]`).
    #[cfg(feature = "gpu")]
    #[test]
    fn batched_two_leaf_reduce_equals_two_separate_launches() {
        use crate::kernels::best_split::SplitSoa;

        let client = cpu_client();
        // 2n*12 raw cells: sibling A features [0, n) with a clear winner at fpos 2;
        // sibling B features [n, 2n) with a clear winner at fpos 0 — different fpos,
        // so a task/base mix-up would show. min_gain_shift differs per sibling.
        let n = 4usize;
        let mut cells = vec![0.0f64; 2 * n * 12];
        // Sibling A (base 0): gains 3,5,9,4 → winner fpos 2.
        put_raw(&mut cells, 0, true, 10, 3.0, false, 1.0, 2.0, 3.0, 4.0);
        put_raw(&mut cells, 1, true, 11, 5.0, true, 5.0, 6.0, 7.0, 8.0);
        put_raw(&mut cells, 2, true, 12, 9.0, false, 9.0, 10.0, 11.0, 12.0);
        put_raw(&mut cells, 3, true, 13, 4.0, true, 13.0, 14.0, 15.0, 16.0);
        // Sibling B (base n*12): gains 8,2,1,6 → winner fpos 0.
        put_raw(&mut cells, n, true, 20, 8.0, true, 21.0, 22.0, 23.0, 24.0);
        put_raw(&mut cells, n + 1, true, 21, 2.0, false, 25.0, 26.0, 27.0, 28.0);
        put_raw(&mut cells, n + 2, true, 22, 1.0, true, 29.0, 30.0, 31.0, 32.0);
        put_raw(&mut cells, n + 3, true, 23, 6.0, false, 33.0, 34.0, 35.0, 36.0);
        let real_feats = vec![7i32, 2, 5, 3];
        let (mgs_a, mgs_b) = (0.5, 1.25); // distinct per-sibling gain shifts

        // BASELINE: two separate single-leaf launches into slots 0 and 1.
        let base = SplitSoa::zeroed(&client, 2);
        let h1 = client.create_from_slice(f64::as_bytes(&cells));
        launch_reduce_into_leaf(&client, h1, cells.len(), &real_feats, n, &base, 0, 0, mgs_a, None);
        let h2 = client.create_from_slice(f64::as_bytes(&cells));
        launch_reduce_into_leaf(&client, h2, cells.len(), &real_feats, n, &base, 1, n * 12, mgs_b, None);

        // CANDIDATE: one batched launch folding both.
        let batched = SplitSoa::zeroed(&client, 2);
        let hb = client.create_from_slice(f64::as_bytes(&cells));
        launch_reduce_into_two_leaves(
            &client, hb, cells.len(), &real_feats, n, &batched, 0, 1, mgs_a, mgs_b, None,
        );

        for slot in 0..2 {
            let want = base.read_record(&client, slot);
            let got = batched.read_record(&client, slot);
            assert_eq!(want.is_valid, got.is_valid, "slot {slot} valid");
            assert_eq!(want.gain.to_bits(), got.gain.to_bits(), "slot {slot} gain");
            assert_eq!(
                want.inner_feature_index, got.inner_feature_index,
                "slot {slot} feat"
            );
            assert_eq!(want.threshold, got.threshold, "slot {slot} threshold");
            assert_eq!(want.default_left, got.default_left, "slot {slot} default_left");
            assert_eq!(
                want.left_sum_gradients.to_bits(),
                got.left_sum_gradients.to_bits(),
                "slot {slot} lsg"
            );
            assert_eq!(
                want.left_sum_hessians.to_bits(),
                got.left_sum_hessians.to_bits(),
                "slot {slot} lsh"
            );
            assert_eq!(
                want.right_sum_gradients.to_bits(),
                got.right_sum_gradients.to_bits(),
                "slot {slot} rsg"
            );
            assert_eq!(
                want.right_sum_hessians.to_bits(),
                got.right_sum_hessians.to_bits(),
                "slot {slot} rsh"
            );
        }
        // Sanity: the two siblings resolved to their distinct expected winners.
        assert_eq!(batched.read_record(&client, 0).threshold, 12, "sibling A winner");
        assert_eq!(batched.read_record(&client, 1).threshold, 20, "sibling B winner");
    }

    /// The FUSED-SUBTRACT co-pack launcher
    /// ([`find_best_splits_fused_siblings_subtract_reduce_into_leaves_on`]) SIGNALS FALLBACK
    /// (`Ok(None)`, writing NOTHING into `out`) on the cubecl-cpu runtime — the staged kernel
    /// family (SharedMemory + sync_cube) does NOT lower there, so the fused path is
    /// real-device-only and the backend must run the byte-unchanged separate
    /// `subtract_resident` + co-scan instead. Pins that contract (the backend's fallback
    /// branch depends on it). The REAL kernel-vs-separate bit-exactness is the CUDA spike's
    /// job (like the pargain / resident-perm kernel layers). GPU-gated (the launcher is
    /// `#[cfg(feature = "gpu")]`).
    #[cfg(feature = "gpu")]
    #[test]
    fn subtract_fuse_signals_fallback_on_cpu_runtime() {
        use crate::kernels::best_split::SplitSoa;

        let client = cpu_client();
        // Two valid sibling leaf totals over the shared feature layout.
        let (buf_a, feats, real_feats, sg_a, sh_a, nd_a) = batch_hist();
        let (buf_b, sg_b, sh_b, nd_b) = batch_hist_b();
        let cfg = relaxed_cfg();
        // `hist_smaller` = A's buffer, `hist_parent` = B's buffer (stand-ins — the launcher
        // gates on the RUNTIME before touching them).
        let h_smaller = client.create_from_slice(f64::as_bytes(&buf_a));
        let h_parent = client.create_from_slice(f64::as_bytes(&buf_b));
        let out = SplitSoa::zeroed(&client, 2);

        let got = find_best_splits_fused_siblings_subtract_reduce_into_leaves_on(
            &client, h_smaller, h_parent, buf_a.len(), &feats, &real_feats, &cfg,
            // 4th slot = the sibling's `parent_output`; inert at `relaxed_cfg()`'s
            // no-smoothing default.
            (sg_a, sh_a, nd_a, 0.0), (sg_b, sh_b, nd_b, 0.0), &out, 0, 1, None,
        )
        .expect("launcher must not error on the fallback path");
        assert!(
            got.is_none(),
            "on the cubecl-cpu runtime the staged fused-subtract path must NOT be taken \
             (Ok(None) fallback signal), got Some(larger_out)"
        );
        // And it must have written NOTHING into `out` (the caller runs the fallback that does).
        for slot in 0..2 {
            let rec = out.read_record(&client, slot);
            assert!(!rec.is_valid, "slot {slot} must stay the zeroed sentinel (no fold on None)");
        }
    }

    // ================= per-grow scan-descriptor hoist (LGBM_DESC_HOIST) =================

    /// The HOISTED descriptor handles produce BYTE-IDENTICAL frontier folds to the
    /// per-launch uploads — pinning "cached handles ≡ fresh uploads" directly on the
    /// runnable cubecl-cpu lane (single-leaf entry, legacy scan + reduce). The cached
    /// arm passes `upload_scan_desc`'s handles (incl. the cached `rf`); the baseline
    /// passes `None`. Same histogram, same cfg ⇒ every SplitSoa field must match
    /// bit-for-bit.
    #[test]
    fn desc_hoist_reduce_into_leaf_byte_identical() {
        use crate::kernels::best_split::SplitSoa;

        let client = cpu_client();
        let (buf, feats, real_feats, sg, sh, nd) = batch_hist();
        let cfg = relaxed_cfg();

        let h1 = client.create_from_slice(f64::as_bytes(&buf));
        let base = SplitSoa::zeroed(&client, 1);
        find_best_splits_fused_reduce_into_leaf_on(
            &client, h1, buf.len(), &feats, &real_feats, &cfg, sg, sh, nd, &base, 0, None,
        )
        .expect("baseline reduce launcher");

        let desc = upload_scan_desc(&client, &feats, Some(&real_feats), buf.len(), "test")
            .expect("desc upload");
        assert!(desc.matches(feats.len(), buf.len()), "cache geometry must match");
        let h2 = client.create_from_slice(f64::as_bytes(&buf));
        let hoisted = SplitSoa::zeroed(&client, 1);
        find_best_splits_fused_reduce_into_leaf_on(
            &client, h2, buf.len(), &feats, &real_feats, &cfg, sg, sh, nd, &hoisted, 0,
            Some(&desc),
        )
        .expect("hoisted reduce launcher");
        // Counts tripwire: the hoisted arm bumped the shared counter at least once.
        // (LOWER bound, not exact — the counter is process-global and the other
        // desc-hoist tests bump it concurrently under the parallel test runner.)
        assert!(
            scan_desc_count_take() >= 1,
            "the hoisted arm must consume the cache (counts tripwire)"
        );

        let want = base.read_record(&client, 0);
        let got = hoisted.read_record(&client, 0);
        assert!(want.is_valid, "fixture must produce a real split (non-vacuous)");
        assert_eq!(want.is_valid, got.is_valid, "valid");
        assert_eq!(want.gain.to_bits(), got.gain.to_bits(), "gain");
        assert_eq!(want.inner_feature_index, got.inner_feature_index, "feat");
        assert_eq!(want.threshold, got.threshold, "threshold");
        assert_eq!(want.default_left, got.default_left, "default_left");
        assert_eq!(want.left_sum_gradients.to_bits(), got.left_sum_gradients.to_bits(), "lsg");
        assert_eq!(want.left_sum_hessians.to_bits(), got.left_sum_hessians.to_bits(), "lsh");
        assert_eq!(want.right_sum_gradients.to_bits(), got.right_sum_gradients.to_bits(), "rsg");
        assert_eq!(want.right_sum_hessians.to_bits(), got.right_sum_hessians.to_bits(), "rsh");
    }

    /// Same invariant on the CO-PACK siblings entry: hoisted descriptors ≡ per-launch
    /// uploads, both frontier slots bit-identical (exercises the batched two-leaf
    /// reduce's cached `rf` on the gpu build, the two-call fallback otherwise).
    #[test]
    fn desc_hoist_siblings_reduce_byte_identical() {
        use crate::kernels::best_split::SplitSoa;

        let client = cpu_client();
        let (buf_a, feats, real_feats, sg_a, sh_a, nd_a) = batch_hist();
        let (buf_b, sg_b, sh_b, nd_b) = batch_hist_b();
        let cfg = relaxed_cfg();

        let run = |desc: Option<&ScanDescHandles>| {
            let h_a = client.create_from_slice(f64::as_bytes(&buf_a));
            let h_b = client.create_from_slice(f64::as_bytes(&buf_b));
            let out = SplitSoa::zeroed(&client, 2);
            find_best_splits_fused_siblings_reduce_into_leaves_on(
                &client, h_a, h_b, buf_a.len(), &feats, &real_feats, &cfg,
                (sg_a, sh_a, nd_a, 0.0), (sg_b, sh_b, nd_b, 0.0), &out, 0, 1, desc,
            )
            .expect("siblings reduce launcher");
            out
        };
        let base = run(None);
        let desc = upload_scan_desc(&client, &feats, Some(&real_feats), buf_a.len(), "test")
            .expect("desc upload");
        let hoisted = run(Some(&desc));

        for slot in 0..2 {
            let want = base.read_record(&client, slot);
            let got = hoisted.read_record(&client, slot);
            assert_eq!(want.is_valid, got.is_valid, "slot {slot} valid");
            assert_eq!(want.gain.to_bits(), got.gain.to_bits(), "slot {slot} gain");
            assert_eq!(want.inner_feature_index, got.inner_feature_index, "slot {slot} feat");
            assert_eq!(want.threshold, got.threshold, "slot {slot} threshold");
            assert_eq!(want.default_left, got.default_left, "slot {slot} default_left");
        }
        assert!(
            base.read_record(&client, 0).is_valid && base.read_record(&client, 1).is_valid,
            "fixture must produce real splits on BOTH siblings (non-vacuous)"
        );
    }
}
