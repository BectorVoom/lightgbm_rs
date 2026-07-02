//! On-device categorical split finding (§6.3 bitset construction + §8.1
//! categorical split evaluator) — **22-03** (ODL-22).
//!
//! ## Why this file exists (crate cycle)
//! The authoritative categorical logic lives in
//! `crates/lgbm-treelearner/src/feature_histogram_categorical.rs`
//! (`find_best_threshold_categorical` + `construct_bitset`), already bit-exact to
//! real `lib_lightgbm` 4.6 on both committed goldens. The on-device grow driver
//! (`grow_tree_on_device_driver`) lives in `lgbm-compute`, **below**
//! `lgbm-treelearner`, so that host code CANNOT be imported (crate cycle, memory
//! `on-device-driver-crate-cycle-constraint`). It is therefore **transcribed
//! byte-for-byte** into this crate as additive native bookkeeping. The dominant
//! risk is transcription FIDELITY (the `kEpsilon` bump site, the `l2` asymmetry,
//! the ctr sort tie order), not design.
//!
//! ## Single-owner f64 anchor discipline (def-f8u-01)
//! All math is `f64` (C++ `double`); the many-vs-many ctr sort reuses
//! [`super::primitives::bitonic_argsort_on`] on `CubeDim::new_1d(1)` (cubecl-cpu
//! has no planes; primitives.rs:51). Never a second GPU f32 categorical path —
//! anchor to the f64 fold.
//!
//! ## Scope
//! This module produces the split decision + the winner bitsets. It does NOT wire
//! into the driver (22-04) and does NOT write `DeviceSplitInfo` slabs.
//!
//! ## The deliberate `l2` asymmetry (feature_histogram_categorical.rs:126-130,223)
//! One-hot uses the ORIGINAL `lambda_l2`; many-vs-many uses `lambda_l2 + cat_l2`.
//! `gain_shift` (the `min_gain_to_split` baseline) always uses the ORIGINAL
//! `lambda_l2`. Reproduced exactly below (Pitfall 3).
//!
//! ## `sum_hessian` pre-bump caller contract (Pitfall 2)
//! [`find_best_threshold_categorical`] expects `sum_hessian` ALREADY bumped by
//! `+2*kEpsilon` (mirroring the host learner call site, learner.rs:2813-2814). The
//! evaluator does NOT bump internally; 22-04's driver branch MUST bump before the
//! call.

use crate::error::ComputeError;
use crate::gain::{
    calculate_splitted_leaf_output, get_leaf_gain, get_split_gains, GainConfig, SplitInfo,
};
use cubecl::prelude::*;
use lgbm_core::types::K_EPSILON;

/// The categorical split result: the standard [`SplitInfo`] (gain/outputs/sums/
/// counts/`default_left`) PLUS the chosen category set as a list of REAL BINS
/// (`output->cat_threshold`, each already `+ offset`). 22-04 maps these into the
/// numeric `SplitScalars` shape and feeds `cat_threshold` (via
/// [`set_real_threshold`]) to the existing `split_categorical_on_device` /
/// `partition_categorical_on_device` entrypoints.
///
/// `split.gain == f64::NEG_INFINITY` (C++ `kMinScore`) signals "no categorical
/// split found" (`is_splittable == false`).
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalSplit {
    /// The standard split-info fields (gain net of `min_gain_shift`, outputs,
    /// sums, counts, `default_left == false`). `cat_threshold` below carries the
    /// category set instead of a numeric `threshold` bin.
    pub split: SplitInfo,
    /// `output->cat_threshold` — the winning category bins (one-hot: a single bin;
    /// many-vs-many: `num_cat_threshold` bins from the sorted scan), each already
    /// `+ offset`.
    pub cat_threshold: Vec<u32>,
}

impl CategoricalSplit {
    /// The "no categorical split found" sentinel (`gain == kMinScore`, empty
    /// bitset). `default_left = false` mirrors the C++ `output->default_left =
    /// false` set at the TOP of `FindBestThresholdCategoricalInner`.
    #[must_use]
    pub fn none() -> Self {
        let mut split = SplitInfo::none();
        split.default_left = false;
        Self {
            split,
            cat_threshold: Vec::new(),
        }
    }

    /// True iff a splittable categorical winner was found.
    #[must_use]
    pub fn is_splittable(&self) -> bool {
        self.split.gain != f64::NEG_INFINITY
    }
}

/// `Common::ConstructBitset(vals, n)` (`common.h`): build a 32-bit-block bitset
/// where bit `vals[i]` is set. The block count is `max(vals)/32 + 1`.
///
/// Byte-for-byte transcription of the host `construct_bitset`
/// (feature_histogram_categorical.rs:424-435). A single-owner sequential OR into a
/// pre-zeroed `Vec<u32>` — NO `Atomic<i64>` (broken on this cubecl) and no
/// per-split device alloc; for the fixture-scale category sets this is
/// bit-identical to the host (RESEARCH Anti-Patterns).
#[must_use]
pub fn construct_bitset(vals: &[u32]) -> Vec<u32> {
    if vals.is_empty() {
        return Vec::new();
    }
    let max_val = *vals.iter().max().unwrap();
    let n_blocks = (max_val / 32 + 1) as usize;
    let mut bits = vec![0u32; n_blocks];
    for &v in vals {
        bits[(v / 32) as usize] |= 1u32 << (v % 32);
    }
    bits
}

/// The two-bitset producer (`SetRealThreshold`, host learner.rs:3697-3721,
/// Pattern 3). Maps the winning inner bins → real category values via
/// `bin_to_category`, and produces BOTH:
/// - the REAL category-value bitset (`construct_bitset` over the mapped values —
///   this is the serialized model `cat_threshold_`), and
/// - the INNER-bin bitset — the routing-key form consumed by
///   `route_to_left_categorical`, whose per-row lookup is
///   `FindInBitset(bitset, bin - min_bin + offset)` (Pitfall 4). The bit for each
///   winning bin is therefore placed at the SAME transformed key the router looks
///   up: `winning_bin - min_bin + offset`.
///
/// `cat_threshold_bins` are the finder's `output->cat_threshold`. Each value is a
/// RAW winning bin (the finder emits `t + offset`, and the compacted histogram
/// slot `t` corresponds to raw bin `t + offset`, so the emitted value equals the
/// member row's raw bin). Because the router keys a member row's raw bin `bin` as
/// `bin - min_bin + offset`, the inner bitset MUST set its bits at the matching
/// transformed key `winning_bin - min_bin + offset` — NOT at the raw winning bin.
/// (CR-01: the two coincide only when `min_bin == offset`, which holds for the
/// two committed `most_freq_bin == 1 ⇒ offset 0, min_bin 0` fixtures but NOT for a
/// `most_freq_bin == 0 ⇒ offset 1` feature, where the router would otherwise route
/// members to the wrong child.)
///
/// Bounds handling for the REAL mapping mirrors the host exactly
/// (`.get(bin).copied().unwrap_or(bin)`): an out-of-range bin does NOT panic and
/// does NOT index out of bounds (threat T-22-06). Winning bins are never negative
/// (the finder scans `bin_start = 1 - offset >= 0`, never the NaN dummy bin 0), and
/// `winning_bin >= min_bin` for any in-range member, so the transformed inner key
/// is non-negative.
#[must_use]
pub fn set_real_threshold(
    cat_threshold_bins: &[i32],
    bin_to_category: &[i32],
    min_bin: i32,
    offset: i32,
) -> (Vec<u32>, Vec<u32>) {
    // Real category-value bitset: map each winning bin → RealThreshold
    // (`bin_2_categorical_[bin]`, C++ BinToValue) with host-faithful bounds.
    let cat_values: Vec<u32> = cat_threshold_bins
        .iter()
        .map(|&bin| {
            let v = bin_to_category.get(bin as usize).copied().unwrap_or(bin);
            v as u32
        })
        .collect();
    let real_bitset = construct_bitset(&cat_values);
    let inner_bitset = construct_inner_bitset(cat_threshold_bins, min_bin, offset);
    (real_bitset, inner_bitset)
}

/// Build ONLY the inner-bin routing bitset (the second element of
/// [`set_real_threshold`]'s pair). Places each winning bin's bit at the EXACT
/// transformed key `route_to_left_categorical` looks up for a member row —
/// `bin - min_bin + offset` — so device routing is key-consistent with the bitset
/// for every offset, not only the `min_bin == offset` case (CR-01). Extracted so
/// the driver can build the inner bitset without recomputing the REAL category
/// mapping (which it already has staged in the `cat_threshold_real` slab, IN-01)
/// while keeping the CR-01 inner-key transform in ONE place.
#[must_use]
pub fn construct_inner_bitset(cat_threshold_bins: &[i32], min_bin: i32, offset: i32) -> Vec<u32> {
    let inner_keys: Vec<u32> = cat_threshold_bins
        .iter()
        .map(|&b| (b - min_bin + offset) as u32)
        .collect();
    construct_bitset(&inner_keys)
}

/// `static_cast<int>(Common::RoundInt(x))` == `static_cast<int>(x + 0.5f)`
/// (`common.h`), reproduced with the f32-rounding-then-truncate form the numeric
/// spine uses (`learner.rs:1098`) so the count reconstruction is bit-identical.
#[inline]
fn round_int(x: f64) -> i32 {
    (x + f64::from(0.5f32)) as i32
}

/// `FeatureHistogram::FindBestThresholdCategoricalInner<false,false,USE_L1,false,false>`
/// (`feature_histogram.cpp:143-382`) — the byte-for-byte transcription of the host
/// `find_best_threshold_categorical` (feature_histogram_categorical.rs:93-326).
/// The many-vs-many ctr ordering is the host's f64 `std::stable_sort` transcribed
/// verbatim (the def-f8u-01 single-owner f64 anchor must match the golden bit-exact,
/// including NaN ctr on degenerate child leaves — an f32 bitonic argsort diverged on
/// ties/NaN and leaked padding indices out of bounds; see the sort site).
///
/// `hist` is the compacted+fixed per-feature histogram (`2*num_bin` cells,
/// `[grad0,hess0,grad1,hess1,...]`), exactly as the numeric scan receives it.
/// `offset` is `meta_->offset` (the categorical-feature offset; `most_freq_bin ==
/// 0 -> 1`, else `0`). `sum_gradient` / `sum_hessian` are the leaf totals;
/// **`sum_hessian` is ALREADY bumped by `+2*kEpsilon` by the caller** (Pitfall 2 —
/// the evaluator does NOT bump internally) so `gain_shift`, `cnt_factor`, and the
/// per-category gains see the same bumped value as the numeric path.
///
/// Single-owner f64 anchor (def-f8u-01): all arithmetic is host-serial f64.
///
/// # Errors
/// Returns [`ComputeError`] only for a malformed histogram slice; the evaluator
/// itself is infallible. `client` is retained for API symmetry with the numeric
/// finders (and future on-device dispatch) — the f64 anchor sort is host-serial.
#[allow(clippy::too_many_arguments)]
pub fn find_best_threshold_categorical<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: i32,
    offset: i32,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> Result<CategoricalSplit, ComputeError> {
    // Retained for API symmetry with the numeric finders; the f64 anchor sort is
    // host-serial (no device dispatch), so `client` is currently unused in the body.
    let _ = client;
    let use_l1 = cfg.use_l1();
    let l1 = cfg.lambda_l1;
    let eps = f64::from(K_EPSILON);

    let get_grad = |t: i32| hist[(t as usize) << 1];
    let get_hess = |t: i32| hist[((t as usize) << 1) + 1];

    let mut is_splittable = false;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_left_count: i32 = 0;
    let mut best_sum_left_gradient: f64 = 0.0;
    let mut best_sum_left_hessian: f64 = 0.0;

    // gain_shift uses the ORIGINAL l2 (the deliberate asymmetry,
    // feature_histogram.cpp:164-168).
    let gain_shift = get_leaf_gain(use_l1, sum_gradient, sum_hessian, l1, cfg.lambda_l2);
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let bin_start = 1 - offset;
    let bin_end = num_bin - offset;
    let cnt_factor = f64::from(num_data) / sum_hessian;

    let use_onehot = num_bin <= cfg.max_cat_to_onehot;

    // Carried out of the many-vs-many branch for the winner-bitset construction.
    let mut sorted_idx: Vec<i32> = Vec::new();
    let mut used_bin: i32 = -1;
    let mut best_threshold: i32 = -1;
    let mut best_dir: i32 = 1;

    if use_onehot {
        // one-hot: uses the ORIGINAL lambda_l2 (NOT + cat_l2).
        let l2 = cfg.lambda_l2;
        let mut t = bin_start;
        while t < bin_end {
            let grad = get_grad(t);
            let hess = get_hess(t);
            let cnt = round_int(hess * cnt_factor);
            // if data not enough, or sum hessian too small
            if cnt < cfg.min_data_in_leaf || hess < cfg.min_sum_hessian_in_leaf {
                t += 1;
                continue;
            }
            let other_count = num_data - cnt;
            if other_count < cfg.min_data_in_leaf {
                t += 1;
                continue;
            }
            let sum_other_hessian = sum_hessian - hess - eps;
            if sum_other_hessian < cfg.min_sum_hessian_in_leaf {
                t += 1;
                continue;
            }
            let sum_other_gradient = sum_gradient - grad;
            // current split gain (other | this), this-side hess bumped by +kEpsilon.
            let current_gain = get_split_gains(
                use_l1,
                sum_other_gradient,
                sum_other_hessian,
                grad,
                hess + eps,
                l1,
                l2,
            );
            if current_gain <= min_gain_shift {
                t += 1;
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                best_threshold = t;
                best_sum_left_gradient = grad;
                best_sum_left_hessian = hess + eps;
                best_left_count = cnt;
                best_gain = current_gain;
            }
            t += 1;
        }
        return Ok(finalize(
            is_splittable,
            best_gain,
            min_gain_shift,
            best_sum_left_gradient,
            best_sum_left_hessian,
            best_left_count,
            sum_gradient,
            sum_hessian,
            num_data,
            use_l1,
            l1,
            l2,
            eps,
            true,
            offset,
            best_threshold,
            best_dir,
            &sorted_idx,
            used_bin,
        ));
    }

    // ---- many-vs-many ----
    let mut l2 = cfg.lambda_l2;
    for i in bin_start..bin_end {
        // C++ `Common::RoundInt(...) >= meta_->config->cat_smooth`: the `int` is
        // promoted to `double` for the comparison (cat_smooth is a double).
        if f64::from(round_int(get_hess(i) * cnt_factor)) >= cfg.cat_smooth {
            sorted_idx.push(i);
        }
    }
    used_bin = sorted_idx.len() as i32;

    // many-vs-many uses lambda_l2 + cat_l2 (feature_histogram.cpp:248) — the
    // increment happens ONLY here, AFTER the one-hot branch returns (Pitfall 3).
    l2 += cfg.cat_l2;

    let cat_smooth = cfg.cat_smooth;
    // ctr = grad / (hess + cat_smooth), ASCENDING. Transcribed VERBATIM from the host
    // anchor (feature_histogram_categorical.rs:226-232): an f64 `std::stable_sort`.
    // `categorical_split.rs` IS the single-owner f64 anchor (def-f8u-01), so its ctr
    // order MUST match the host (== the real lib_lightgbm 4.6 golden) bit-exact — for
    // BOTH the tie-free top-level split AND the degenerate deeper child leaves the
    // full grow loop reaches, where a zero-hessian bin yields ctr = 0/0 = NaN
    // (`partial_cmp -> Equal` keeps NaN stable in place, exactly as the host).
    //
    // An earlier f32 `bitonic_argsort_on` substitution here was UNSAFE on the real
    // grow loop: it (a) diverged from the f64-stable order on ties/NaN, and (b) on
    // NaN keys leaked its power-of-two padding indices into the truncated permutation,
    // indexing `sorted_idx` out of bounds (panic). 22-05's D-01 #1 many-vs-many gate
    // (the linchpin) drives the full device grow loop and surfaced both. The f64
    // stable sort is the faithful transcription and is crash-free by construction.
    let ctr = |t: i32| -> f64 { get_grad(t) / (get_hess(t) + cat_smooth) };
    sorted_idx.sort_by(|&a, &b| {
        ctr(a)
            .partial_cmp(&ctr(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let find_direction = [1i32, -1i32];
    let start_position = [0i32, used_bin - 1];
    let max_num_cat = cfg.max_cat_threshold.min((used_bin + 1) / 2);

    is_splittable = false;
    for out_i in 0..find_direction.len() {
        let dir = find_direction[out_i];
        let mut start_pos = start_position[out_i];
        let min_data_per_group = cfg.min_data_per_group;
        let mut cnt_cur_group: i32 = 0;
        let mut sum_left_gradient: f64 = 0.0;
        let mut sum_left_hessian: f64 = eps;
        let mut left_count: i32 = 0;
        let mut i = 0i32;
        while i < used_bin && i < max_num_cat {
            let t = sorted_idx[start_pos as usize];
            start_pos += dir;
            let grad = get_grad(t);
            let hess = get_hess(t);
            let cnt = round_int(hess * cnt_factor);

            sum_left_gradient += grad;
            sum_left_hessian += hess;
            left_count += cnt;
            cnt_cur_group += cnt;

            if left_count < cfg.min_data_in_leaf || sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                i += 1;
                continue;
            }
            let right_count = num_data - left_count;
            if right_count < cfg.min_data_in_leaf || right_count < min_data_per_group {
                break;
            }
            let sum_right_hessian = sum_hessian - sum_left_hessian;
            if sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                break;
            }
            if cnt_cur_group < min_data_per_group {
                i += 1;
                continue;
            }
            cnt_cur_group = 0;

            let sum_right_gradient = sum_gradient - sum_left_gradient;
            let current_gain = get_split_gains(
                use_l1,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
            );
            if current_gain <= min_gain_shift {
                i += 1;
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                best_left_count = left_count;
                best_sum_left_gradient = sum_left_gradient;
                best_sum_left_hessian = sum_left_hessian;
                best_threshold = i;
                best_gain = current_gain;
                best_dir = dir;
            }
            i += 1;
        }
    }

    Ok(finalize(
        is_splittable,
        best_gain,
        min_gain_shift,
        best_sum_left_gradient,
        best_sum_left_hessian,
        best_left_count,
        sum_gradient,
        sum_hessian,
        num_data,
        use_l1,
        l1,
        l2,
        eps,
        false,
        offset,
        best_threshold,
        best_dir,
        &sorted_idx,
        used_bin,
    ))
}

/// The `if (is_splittable_) { ... }` finalization block
/// (`feature_histogram.cpp:342-381`): compute left/right outputs (with the
/// per-category l2), counts, sums, the net gain, and build `output->cat_threshold`
/// (one-hot: `[best_threshold + offset]`; many-vs-many: the first `best_threshold
/// + 1` of `sorted_idx` in the winning direction, each `+ offset`).
#[allow(clippy::too_many_arguments)]
fn finalize(
    is_splittable: bool,
    best_gain: f64,
    min_gain_shift: f64,
    best_sum_left_gradient: f64,
    best_sum_left_hessian: f64,
    best_left_count: i32,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    use_l1: bool,
    l1: f64,
    l2: f64,
    eps: f64,
    use_onehot: bool,
    offset: i32,
    best_threshold: i32,
    best_dir: i32,
    sorted_idx: &[i32],
    used_bin: i32,
) -> CategoricalSplit {
    if !is_splittable {
        return CategoricalSplit::none();
    }

    let left_output = calculate_splitted_leaf_output(
        use_l1,
        best_sum_left_gradient,
        best_sum_left_hessian,
        l1,
        l2,
    );
    let left_count = best_left_count;
    let left_sum_gradient = best_sum_left_gradient;
    let left_sum_hessian = best_sum_left_hessian - eps;

    let right_output = calculate_splitted_leaf_output(
        use_l1,
        sum_gradient - best_sum_left_gradient,
        sum_hessian - best_sum_left_hessian,
        l1,
        l2,
    );
    let right_count = num_data - best_left_count;
    let right_sum_gradient = sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = sum_hessian - best_sum_left_hessian - eps;

    let gain = best_gain - min_gain_shift;

    let cat_threshold: Vec<u32> = if use_onehot {
        vec![(best_threshold + offset) as u32]
    } else {
        let num_cat_threshold = best_threshold + 1;
        let mut v = Vec::with_capacity(num_cat_threshold as usize);
        if best_dir == 1 {
            for i in 0..num_cat_threshold {
                v.push((sorted_idx[i as usize] + offset) as u32);
            }
        } else {
            for i in 0..num_cat_threshold {
                v.push((sorted_idx[(used_bin - 1 - i) as usize] + offset) as u32);
            }
        }
        v
    };

    let mut split = SplitInfo::none();
    split.gain = gain;
    split.left_count = left_count;
    split.right_count = right_count;
    split.left_sum_gradient = left_sum_gradient;
    split.left_sum_hessian = left_sum_hessian;
    split.right_sum_gradient = right_sum_gradient;
    split.right_sum_hessian = right_sum_hessian;
    split.left_output = left_output;
    split.right_output = right_output;
    // The categorical path always sets default_left = false.
    split.default_left = false;
    // `threshold` is unused for categorical splits (the bitset replaces it).
    split.threshold = 0;

    CategoricalSplit {
        split,
        cat_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gain::{get_leaf_gain, get_split_gains};
    use crate::runtime::cpu_client;

    /// Build a `2*num_bin` compacted histogram from per-bin (grad, hess) pairs
    /// (`[grad0,hess0,grad1,hess1,...]`), exactly the shape the numeric scan uses.
    fn hist_from(pairs: &[(f64, f64)]) -> Vec<f64> {
        let mut h = Vec::with_capacity(pairs.len() * 2);
        for &(g, he) in pairs {
            h.push(g);
            h.push(he);
        }
        h
    }

    /// The `cat_corpus` config convention (learner_parity.rs:1472-1481).
    fn cat_cfg(cat_l2: f64, cat_smooth: f64, max_cat_to_onehot: i32) -> GainConfig {
        let mut cfg = GainConfig::default();
        cfg.min_data_in_leaf = 1;
        cfg.min_sum_hessian_in_leaf = 1e-3;
        cfg.lambda_l2 = 0.0;
        cfg.cat_l2 = cat_l2;
        cfg.cat_smooth = cat_smooth;
        cfg.min_data_per_group = 1;
        cfg.max_cat_threshold = 32;
        cfg.max_cat_to_onehot = max_cat_to_onehot;
        cfg
    }

    const EPS: f64 = 1e-15; // K_EPSILON widened; bump applied at the CALL site.

    // -------------------------------------------------------------------------
    // §8.1 evaluator — one-hot branch (num_bin <= max_cat_to_onehot).
    // -------------------------------------------------------------------------

    #[test]
    fn onehot_branch_picks_single_category() {
        // num_bin=4 (<= max_cat_to_onehot=4) -> one-hot. offset=1.
        // Mirrors the host `onehot_picks_a_single_category` fixture.
        let client = cpu_client();
        let mut cfg = cat_cfg(10.0, 10.0, 4);
        cfg.min_sum_hessian_in_leaf = 0.0;
        let hist = hist_from(&[(0.0, 0.0), (10.0, 5.0), (-10.0, 5.0), (1.0, 5.0)]);
        // sum_hessian pre-bumped by +2*kEpsilon (caller contract, Pitfall 2).
        let r =
            find_best_threshold_categorical(&client, &hist, &cfg, 4, 1, 1.0, 15.0 + 2.0 * EPS, 30)
                .unwrap();
        assert!(r.is_splittable(), "expected a one-hot split");
        assert_eq!(r.cat_threshold.len(), 1, "one-hot yields a single category");
        assert!(!r.split.default_left, "categorical split never defaults left");
    }

    #[test]
    fn onehot_branch_uses_original_l2_not_cat_l2() {
        // One-hot uses lambda_l2 (0), NEVER + cat_l2. Proof: a huge cat_l2 must not
        // change the one-hot gain (Pitfall 3).
        let client = cpu_client();
        let hist = hist_from(&[(0.0, 0.0), (10.0, 5.0), (-10.0, 5.0), (1.0, 5.0)]);
        let mut cfg0 = cat_cfg(0.0, 10.0, 4);
        cfg0.min_sum_hessian_in_leaf = 0.0;
        let mut cfg_big = cat_cfg(1000.0, 10.0, 4);
        cfg_big.min_sum_hessian_in_leaf = 0.0;
        let g0 =
            find_best_threshold_categorical(&client, &hist, &cfg0, 4, 1, 1.0, 15.0 + 2.0 * EPS, 30)
                .unwrap();
        let gb = find_best_threshold_categorical(
            &client,
            &hist,
            &cfg_big,
            4,
            1,
            1.0,
            15.0 + 2.0 * EPS,
            30,
        )
        .unwrap();
        assert_eq!(
            g0.split.gain, gb.split.gain,
            "one-hot gain must be independent of cat_l2 (uses lambda_l2 only)"
        );
    }

    // -------------------------------------------------------------------------
    // §8.1 evaluator — many-vs-many branch, pinned to the committed goldens.
    // -------------------------------------------------------------------------

    /// cat_onehot fixture (num_bin=5 > max_cat_to_onehot=4 => many-vs-many path,
    /// but the clear single-category winner reproduces the golden `cat_threshold=8`).
    fn cat_onehot_hist() -> Vec<f64> {
        hist_from(&[(0.0, 0.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-100.0, 10.0)])
    }

    /// cat_manyvsmany fixture: per-bin grad sums 0,-10,-20,-80,-90,-100, uniform
    /// hess=10, cat_smooth=0 => ctr = 0,-1,-2,-8,-9,-10 (strictly distinct, A1).
    fn cat_manyvsmany_hist() -> Vec<f64> {
        hist_from(&[
            (0.0, 0.0),
            (0.0, 10.0),
            (-10.0, 10.0),
            (-20.0, 10.0),
            (-80.0, 10.0),
            (-90.0, 10.0),
            (-100.0, 10.0),
        ])
    }

    #[test]
    fn manyvsmany_ctr_values_are_tie_free() {
        // A1 / Pitfall 1: the f32-bitonic order equals the f64-stable order ONLY if
        // ctr is tie-free. Assert strict distinctness before locking.
        let hist = cat_manyvsmany_hist();
        let cat_smooth = 0.0;
        let ctr: Vec<f64> = (1..7)
            .map(|t: i32| hist[(t as usize) << 1] / (hist[((t as usize) << 1) + 1] + cat_smooth))
            .collect();
        assert_eq!(ctr.len(), 6);
        // Strict pairwise distinctness (no sort needed) — A1 / Pitfall 1.
        for a in 0..ctr.len() {
            for b in (a + 1)..ctr.len() {
                assert!(ctr[a] != ctr[b], "cat_manyvsmany ctr must be tie-free");
            }
        }
    }

    #[test]
    fn onehot_fixture_matches_golden() {
        // Winner = bin 4 alone -> real bitset [8]; net gain 250.0 (hand-verified).
        let client = cpu_client();
        let cfg = cat_cfg(10.0, 10.0, 4);
        let hist = cat_onehot_hist();
        let r =
            find_best_threshold_categorical(&client, &hist, &cfg, 5, 0, -100.0, 40.0 + 2.0 * EPS, 40)
                .unwrap();
        assert!(r.is_splittable());
        let bins: Vec<i32> = r.cat_threshold.iter().map(|&b| b as i32).collect();
        let (real, _inner) = set_real_threshold(&bins, &[-1, 0, 1, 2, 3], 0, 0);
        assert_eq!(real, vec![8u32], "cat_onehot real bitset != golden (8)");
        assert!(
            (r.split.gain - 250.0).abs() < 1e-9,
            "cat_onehot net gain {} != 250.0",
            r.split.gain
        );
    }

    #[test]
    fn manyvsmany_fixture_root_matches_golden() {
        // Root winner = bins {6,5,4} -> real bitset [56]; net gain 345.0.
        let client = cpu_client();
        let cfg = cat_cfg(10.0, 0.0, 4);
        let hist = cat_manyvsmany_hist();
        let r =
            find_best_threshold_categorical(&client, &hist, &cfg, 7, 0, -300.0, 60.0 + 2.0 * EPS, 60)
                .unwrap();
        assert!(r.is_splittable());
        let bins: Vec<i32> = r.cat_threshold.iter().map(|&b| b as i32).collect();
        let (real, _inner) = set_real_threshold(&bins, &[-1, 0, 1, 2, 3, 4, 5], 0, 0);
        assert_eq!(real, vec![56u32], "cat_manyvsmany root real bitset != golden (56)");
        assert!(
            (r.split.gain - 345.0).abs() < 1e-9,
            "cat_manyvsmany root net gain {} != 345.0",
            r.split.gain
        );
    }

    #[test]
    fn manyvsmany_adds_cat_l2() {
        // Many-vs-many uses lambda_l2 + cat_l2: a different cat_l2 must change the
        // gain (Pitfall 3), unlike the one-hot path.
        let client = cpu_client();
        let hist = cat_manyvsmany_hist();
        let r0 = find_best_threshold_categorical(
            &client,
            &hist,
            &cat_cfg(0.0, 0.0, 4),
            7,
            0,
            -300.0,
            60.0 + 2.0 * EPS,
            60,
        )
        .unwrap();
        let r10 = find_best_threshold_categorical(
            &client,
            &hist,
            &cat_cfg(10.0, 0.0, 4),
            7,
            0,
            -300.0,
            60.0 + 2.0 * EPS,
            60,
        )
        .unwrap();
        assert!(r0.is_splittable() && r10.is_splittable());
        assert!(
            r0.split.gain != r10.split.gain,
            "many-vs-many gain must depend on cat_l2"
        );
    }

    #[test]
    fn manyvsmany_gain_uses_cat_l2_bit_exact() {
        // Pin the winning-split gain to an independent computation with l2 =
        // lambda_l2 + cat_l2 (bit-exact, same f64 ops as the evaluator).
        let client = cpu_client();
        let cfg = cat_cfg(10.0, 0.0, 4);
        let hist = cat_manyvsmany_hist();
        let sum_h = 60.0 + 2.0 * EPS;
        let r =
            find_best_threshold_categorical(&client, &hist, &cfg, 7, 0, -300.0, sum_h, 60).unwrap();
        // Winner left = {6,5,4}: sum_left_grad = -270, sum_left_hess = 30 + eps.
        let l2 = cfg.lambda_l2 + cfg.cat_l2;
        let best_gain = get_split_gains(false, -270.0, 30.0 + EPS, -30.0, sum_h - (30.0 + EPS), 0.0, l2);
        let gain_shift = get_leaf_gain(false, -300.0, sum_h, 0.0, cfg.lambda_l2);
        let expected_net = best_gain - gain_shift;
        assert_eq!(r.split.gain, expected_net, "gain not bit-exact vs +cat_l2 anchor");
    }

    // -------------------------------------------------------------------------
    // §6.3 construct_bitset — bit-identical to the host packer.
    // -------------------------------------------------------------------------

    #[test]
    fn construct_bitset_empty_is_empty() {
        assert_eq!(construct_bitset(&[]), Vec::<u32>::new());
    }

    #[test]
    fn construct_bitset_single_bit() {
        // [0] -> one block, bit 0.
        assert_eq!(construct_bitset(&[0]), vec![0b1]);
        // [5] -> one block, bit 5.
        assert_eq!(construct_bitset(&[5]), vec![1u32 << 5]);
    }

    #[test]
    fn construct_bitset_spanning_two_blocks() {
        // [0,31,32] -> max 32 -> n_blocks = 32/32+1 = 2. bits 0,31 in block 0; bit 0
        // (32 % 32) in block 1.
        let b = construct_bitset(&[0, 31, 32]);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], (1u32 << 0) | (1u32 << 31));
        assert_eq!(b[1], 1u32 << 0);
    }

    #[test]
    fn construct_bitset_matches_host_doc_example() {
        // Mirrors the host test `construct_bitset_sets_expected_bits`.
        let b = construct_bitset(&[0, 1, 33]);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], 0b11);
        assert_eq!(b[1], 1 << 1);
    }

    // -------------------------------------------------------------------------
    // set_real_threshold — REAL bitset pinned to the committed real 4.6 goldens.
    // -------------------------------------------------------------------------

    // bin_2_categorical for both committed fixtures maps bin -> category value:
    //   cat_onehot:      [-1, 0, 1, 2, 3]            (bins 0..4)
    //   cat_manyvsmany:  [-1, 0, 1, 2, 3, 4, 5]      (bins 0..6)
    // Both use most_freq_bin=1 => offset = 0.

    #[test]
    fn set_real_threshold_onehot_matches_golden_bitset() {
        // cat_onehot golden `cat_threshold=8` (bit 3). The finder selects bin 4
        // (the only bin with a non-zero gradient); bin_2_categorical[4] = 3, so the
        // real bitset is `1<<3 = 8`.
        let bin_2_categorical = [-1, 0, 1, 2, 3];
        // min_bin == 0, offset == 0 ⇒ inner key == raw winning bin (4).
        let (real, inner) = set_real_threshold(&[4], &bin_2_categorical, 0, 0);
        assert_eq!(real, vec![8u32], "cat_onehot real bitset != golden (8)");
        // inner bitset over the winning bin 4 itself: 1<<4 = 16.
        assert_eq!(inner, vec![16u32]);
    }

    #[test]
    fn set_real_threshold_manyvsmany_root_matches_golden_bitset() {
        // cat_manyvsmany golden node-0 `cat_threshold=56` (bits 3,4,5). The finder
        // selects bins {6,5,4} (the three lowest-ctr categories);
        // bin_2_categorical maps them to {5,4,3}, so the real bitset is
        // (1<<5)|(1<<4)|(1<<3) = 56.
        let bin_2_categorical = [-1, 0, 1, 2, 3, 4, 5];
        let (real, _inner) = set_real_threshold(&[6, 5, 4], &bin_2_categorical, 0, 0);
        assert_eq!(real, vec![56u32], "cat_manyvsmany real bitset != golden (56)");
    }

    #[test]
    fn set_real_threshold_inner_key_offset1_matches_router_key() {
        // CR-01 regression: for a `most_freq_bin == 0 ⇒ offset 1` feature, the
        // router (`route_to_left_categorical`) keys a member row's raw bin `bin` as
        // `bin - min_bin + offset`. The inner bitset MUST set its bit at that same
        // transformed key, NOT at the raw winning bin. With min_bin=0, offset=1 and
        // a winning raw bin of 4, the router looks up 4 - 0 + 1 = 5, so the inner
        // bit must be at 5 (`1<<5 = 32`), not at 4 (`1<<4 = 16`).
        let bin_2_categorical = [-1, 0, 1, 2, 3];
        let (_real, inner) = set_real_threshold(&[4], &bin_2_categorical, 0, 1);
        assert_eq!(
            inner,
            vec![1u32 << 5],
            "offset==1 inner bit must sit at the router's transformed key (bin+1)"
        );
        // The REAL (model) bitset is unaffected by the routing offset: bin 4 → cat 3.
        assert_eq!(_real, vec![1u32 << 3]);
    }

    #[test]
    fn set_real_threshold_inner_key_nonzero_min_bin() {
        // General key-consistency: with min_bin=2, offset=0 and winning raw bin 5,
        // the router looks up 5 - 2 + 0 = 3, so the inner bit must sit at 3.
        let bin_2_categorical = [-1, 0, 1, 2, 3, 4, 5];
        let (_real, inner) = set_real_threshold(&[5], &bin_2_categorical, 2, 0);
        assert_eq!(inner, vec![1u32 << 3]);
    }

    #[test]
    fn set_real_threshold_out_of_range_bin_does_not_panic() {
        // Host bounds handling: `.get(bin).unwrap_or(bin)` — an out-of-range bin
        // falls back to the bin index itself, no panic, no OOB (threat T-22-06).
        let bin_2_categorical = [-1, 0, 1];
        let (real, _inner) = set_real_threshold(&[9], &bin_2_categorical, 0, 0);
        assert_eq!(real, construct_bitset(&[9]));
    }
}
