//! Categorical split finding — a VERBATIM transcription of the C++ LightGBM
//! `FeatureHistogram::FindBestThresholdCategoricalInner`
//! (`LightGBM/src/treelearner/feature_histogram.cpp:143-382`, the
//! `USE_RAND=false, USE_MC=false, USE_MAX_OUTPUT=false, USE_SMOOTHING=false`
//! default CPU instantiation — the only one in scope, mirroring the numeric
//! anchor in `gain.rs`).
//!
//! ## D-06 HARD INVARIANT (PURELY ADDITIVE)
//! This module is reached ONLY for `bin_type == Categorical` features, dispatched
//! at the TOP of the learner's per-feature loop. The numeric (continuous) split
//! spine in `learner.rs` is NEVER routed through here and stays byte-untouched and
//! bit-exact vs real `lib_lightgbm` 4.6.
//!
//! ## The deliberate l2 asymmetry (feature_histogram.cpp:163-168,248)
//! The `gain_shift` baseline (used for `min_gain_to_split` / `is_splittable`) uses
//! the ORIGINAL `lambda_l2`; the per-category split gain uses `lambda_l2 + cat_l2`.
//! This asymmetry is intentional in C++ ("If no smoothing, the parent output is
//! calculated with the larger categorical l2, whereas min_split_gain uses the
//! original l2") and is reproduced exactly below.
//!
//! ## Numerical contract
//! All gain math is `f64` (C++ `double`); `kEpsilon = 1e-15f` is reused from
//! `lgbm_core::types::K_EPSILON` (widened to f64 exactly as the C++ promotes the
//! `float` constant). `RoundInt(x) = static_cast<int>(x + 0.5f)` is reproduced
//! with the `(x + 0.5f32 as f64) as i32` form used by the numeric spine
//! (`learner.rs:1098`) so the count reconstruction is bit-identical.

use lgbm_compute::gain::{
    calculate_splitted_leaf_output_full, get_leaf_gain_full, get_leaf_gain_given_output,
    get_split_gains_full, GainConfig,
};
use lgbm_compute::gain::SplitInfo;
use lgbm_core::types::K_EPSILON;

/// The categorical split result: the standard [`SplitInfo`] (gain/outputs/sums/
/// counts/default_left) PLUS the chosen category bitset as a list of REAL bins
/// (`output->cat_threshold`, the `num_cat_threshold`-long category list). The
/// learner converts these bins to inner + real bitsets when it grows the node.
///
/// `split.gain == kMinScore` (`-inf`) signals "no categorical split found"
/// (`is_splittable_ == false`).
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalSplit {
    /// The standard split-info fields (gain net of `min_gain_shift`, outputs,
    /// sums, counts, `default_left`). `cat_threshold` below carries the category
    /// set instead of a numeric `threshold` bin.
    pub split: SplitInfo,
    /// `output->cat_threshold` — the winning category bins (one-hot: a single
    /// bin; many-vs-many: `num_cat_threshold` bins from the sorted scan, already
    /// `+ offset`).
    pub cat_threshold: Vec<u32>,
}

impl CategoricalSplit {
    /// The "no categorical split found" sentinel (`gain == kMinScore`, empty
    /// bitset). `default_left = false` mirrors the C++ `output->default_left =
    /// false` set at the TOP of `FindBestThresholdCategoricalInner` (the
    /// categorical path never sets `default_left = true`).
    pub fn none() -> Self {
        let mut split = SplitInfo::none();
        split.default_left = false;
        Self {
            split,
            cat_threshold: Vec::new(),
        }
    }

    /// True iff a splittable categorical winner was found.
    pub fn is_splittable(&self) -> bool {
        self.split.gain != f64::NEG_INFINITY
    }
}

/// `static_cast<int>(Common::RoundInt(x))` == `static_cast<int>(x + 0.5f)`
/// (`common.h`), reproduced with the f32-rounding-then-truncate form the numeric
/// spine uses (`learner.rs:1098`) so the count reconstruction is bit-identical.
#[inline]
fn round_int(x: f64) -> i32 {
    (x + f64::from(0.5f32)) as i32
}

/// `FeatureHistogram::FindBestThresholdCategoricalInner<false,false,USE_L1,false,false>`
/// (`feature_histogram.cpp:143-382`).
///
/// `hist` is the compacted+fixed per-feature histogram (`2*num_bin` cells,
/// `[grad0,hess0,grad1,hess1,...]`), exactly as the numeric scan receives it.
/// `offset` is `meta_->offset` (the categorical-feature offset; `most_freq_bin ==
/// 0 -> 1`, else `0` — the same rule the numeric spine uses). `sum_gradient` /
/// `sum_hessian` are the leaf totals; `sum_hessian` is ALREADY bumped by
/// `+2*kEpsilon` by the caller (mirroring `FindBestThreshold`'s
/// `sum_hessian + 2*kEpsilon` at feature_histogram.hpp:172) so `gain_shift` and
/// the per-category gains see the same bumped value as the numeric path.
///
/// Returns a [`CategoricalSplit`]; `is_splittable() == false` when no candidate
/// clears the gates.
pub fn find_best_threshold_categorical(
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: i32,
    offset: i32,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
) -> CategoricalSplit {
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

    // The C++ USE_MAX_OUTPUT / USE_SMOOTHING template axes.
    let mds = cfg.max_delta_step;
    let smoothing = cfg.use_smoothing();
    let ps = cfg.path_smooth;
    let parent = cfg.parent_output;

    // `gain_shift` uses the ORIGINAL l2 (the deliberate asymmetry,
    // feature_histogram.cpp:157-169), and the categorical path takes a DIFFERENT
    // shape under smoothing than the numeric one does:
    //
    // ```cpp
    // if (USE_SMOOTHING) {
    //   gain_shift = GetLeafGainGivenOutput<USE_L1>(g, h, l1, l2, parent_output);
    // } else {
    //   // "special case for no smoothing to preserve existing behaviour": the
    //   // parent output is computed with the LARGER categorical l2 while
    //   // min_split_gain uses the original l2.
    //   gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, false>(
    //       g, h, l1, l2, max_delta_step, 0, num_data, 0);
    // }
    // ```
    //
    // Note the smoothing branch evaluates the gain AT `parent_output` itself, not
    // at a re-derived leaf output — transcribed verbatim.
    let gain_shift = if smoothing {
        get_leaf_gain_given_output(use_l1, sum_gradient, sum_hessian, l1, cfg.lambda_l2, parent)
    } else {
        get_leaf_gain_full(
            use_l1,
            sum_gradient,
            sum_hessian,
            l1,
            cfg.lambda_l2,
            mds,
            false,
            0.0,
            num_data,
            0.0,
        )
    };
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let bin_start = 1 - offset;
    let bin_end = num_bin - offset;
    let cnt_factor = f64::from(num_data) / sum_hessian;

    // Per-category gain uses l2 += cat_l2 (feature_histogram.cpp:248). For the
    // one-hot path C++ uses the ORIGINAL l2 (`l2` is incremented only in the
    // many-vs-many `else`, AFTER the one-hot branch's GetSplitGains call). We
    // mirror that: one-hot uses `lambda_l2`, many-vs-many uses `lambda_l2 + cat_l2`.
    let use_onehot = num_bin <= cfg.max_cat_to_onehot;

    // Carried out of the many-vs-many branch for the winner-bitset construction.
    let mut sorted_idx: Vec<i32> = Vec::new();
    let mut used_bin: i32 = -1;
    let mut best_threshold: i32 = -1;
    let mut best_dir: i32 = 1;

    if use_onehot {
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
            let current_gain = get_split_gains_full(
                use_l1,
                sum_other_gradient,
                sum_other_hessian,
                grad,
                hess + eps,
                l1,
                l2,
                mds,
                smoothing,
                ps,
                other_count,
                cnt,
                parent,
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
        // many-vs-many state stays at defaults; the winner bitset uses
        // best_threshold + offset (a single bin).
        let l2_for_output = l2;
        return finalize(
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
            l2_for_output,
            mds,
            smoothing,
            ps,
            parent,
            eps,
            true,
            offset,
            best_threshold,
            best_dir,
            &sorted_idx,
            used_bin,
        );
    }

    // ---- many-vs-many ----
    let mut l2 = cfg.lambda_l2;
    for i in bin_start..bin_end {
        // C++ `Common::RoundInt(...) >= meta_->config->cat_smooth`: the `int` is
        // promoted to `double` for the comparison (cat_smooth is a double). We
        // compare in f64 to match fractional cat_smooth exactly.
        if f64::from(round_int(get_hess(i) * cnt_factor)) >= cfg.cat_smooth {
            sorted_idx.push(i);
        }
    }
    used_bin = sorted_idx.len() as i32;

    l2 += cfg.cat_l2;

    let cat_smooth = cfg.cat_smooth;
    let ctr = |t: i32| -> f64 { get_grad(t) / (get_hess(t) + cat_smooth) };
    // std::stable_sort ascending by ctr. Rust's sort_by is stable.
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
            let current_gain = get_split_gains_full(
                use_l1,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                l1,
                l2,
                mds,
                smoothing,
                ps,
                left_count,
                right_count,
                parent,
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

    finalize(
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
        mds,
        smoothing,
        ps,
        parent,
        eps,
        false,
        offset,
        best_threshold,
        best_dir,
        &sorted_idx,
        used_bin,
    )
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
    // The C++ USE_MAX_OUTPUT / USE_SMOOTHING axes + the leaf's parent output.
    mds: f64,
    smoothing: bool,
    ps: f64,
    parent: f64,
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

    let right_count_out = num_data - best_left_count;
    // Each side's smoothing weight uses that side's OWN row count
    // (feature_histogram.cpp:343-355).
    let left_output = calculate_splitted_leaf_output_full(
        use_l1,
        best_sum_left_gradient,
        best_sum_left_hessian,
        l1,
        l2,
        mds,
        smoothing,
        ps,
        best_left_count,
        parent,
    );
    let left_count = best_left_count;
    let left_sum_gradient = best_sum_left_gradient;
    let left_sum_hessian = best_sum_left_hessian - eps;

    let right_output = calculate_splitted_leaf_output_full(
        use_l1,
        sum_gradient - best_sum_left_gradient,
        sum_hessian - best_sum_left_hessian,
        l1,
        l2,
        mds,
        smoothing,
        ps,
        right_count_out,
        parent,
    );
    let right_count = right_count_out;
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
    // The categorical path always sets default_left = false (the C++
    // FindBestThresholdCategoricalInner sets it at the top and never flips it).
    split.default_left = false;
    // `threshold` is unused for categorical splits (the bitset replaces it).
    split.threshold = 0;

    CategoricalSplit {
        split,
        cat_threshold,
    }
}

/// `Common::ConstructBitset(vals, n)` (`common.h`): build a 32-bit-block bitset
/// where bit `vals[i]` is set. The block count is `max(vals)/32 + 1`.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg() -> GainConfig {
        let c = lgbm_core::Config::default();
        GainConfig::from_config(&c)
    }

    /// Build a `2*num_bin` compacted histogram from per-bin (grad, hess) pairs.
    fn hist_from(pairs: &[(f64, f64)]) -> Vec<f64> {
        let mut h = Vec::with_capacity(pairs.len() * 2);
        for &(g, he) in pairs {
            h.push(g);
            h.push(he);
        }
        h
    }

    #[test]
    fn construct_bitset_sets_expected_bits() {
        let b = construct_bitset(&[0, 1, 33]);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], 0b11);
        assert_eq!(b[1], 1 << 1);
    }

    #[test]
    fn no_split_when_all_gated() {
        // num_bin small -> one-hot path; tiny hessians fail min_data_in_leaf.
        let hist = hist_from(&[(0.0, 0.0), (1.0, 0.001), (1.0, 0.001)]);
        let cfg = base_cfg();
        let r = find_best_threshold_categorical(&hist, &cfg, 3, 1, 2.0, 0.002, 4);
        assert!(!r.is_splittable());
    }

    #[test]
    fn onehot_picks_a_single_category() {
        // num_bin=4 (<= max_cat_to_onehot=4) -> one-hot. offset=1.
        // bins 1..4 each large hessian; one bin has a very different gradient.
        // Relax leaf gates so a split is admissible.
        let mut cfg = base_cfg();
        cfg.min_data_in_leaf = 1;
        cfg.min_sum_hessian_in_leaf = 0.0;
        let hist = hist_from(&[
            (0.0, 0.0), // bin 0 (offset slot, unused for offset=1)
            (10.0, 5.0),
            (-10.0, 5.0),
            (1.0, 5.0),
        ]);
        let r = find_best_threshold_categorical(&hist, &cfg, 4, 1, 1.0, 15.0, 30);
        assert!(r.is_splittable(), "expected a one-hot split");
        assert_eq!(r.cat_threshold.len(), 1, "one-hot yields a single category");
    }
}
