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
//! ## 22-03 Task 1 (this slice): §6.3 bitset construction
//! [`construct_bitset`] + the two-bitset [`set_real_threshold`] producer land here.
//! The §8.1 [`find_best_threshold_categorical`] evaluator is a compiling stub
//! (returns the "no split" sentinel) until Task 2 fills it under TDD.

use crate::error::ComputeError;
use crate::gain::{GainConfig, SplitInfo};
use cubecl::prelude::*;

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
///   this is the serialized model `cat_threshold_` AND, on the single-group
///   spine, the partition routing key), and
/// - the INNER-bin bitset (`construct_bitset` over the winning bins as-is — the
///   routing-key form consistent with `route_to_left_categorical`,
///   `FindInBitset(bitset, bin - min_bin + offset)`, Pitfall 4).
///
/// `cat_threshold_bins` are the finder's `output->cat_threshold` (each already
/// `+ offset`; on the single-group categorical spine `min_bin == 0`, so the inner
/// bin IS the routing key). Bounds handling mirrors the host exactly
/// (`.get(bin).copied().unwrap_or(bin)`): an out-of-range bin does NOT panic and
/// does NOT index out of bounds (threat T-22-06). Winning bins are never negative
/// (the finder scans `bin_start = 1 - offset >= 0`, never the NaN dummy bin 0).
#[must_use]
pub fn set_real_threshold(
    cat_threshold_bins: &[i32],
    bin_to_category: &[i32],
    _offset: i32,
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

    // Inner-bin bitset: the winning bins already carry `+ offset` (min_bin == 0 on
    // the single-group spine), so they ARE the routing keys consumed by
    // `route_to_left_categorical` (`FindInBitset(bitset, bin - 0 + 0)`).
    let inner_keys: Vec<u32> = cat_threshold_bins.iter().map(|&b| b as u32).collect();
    let inner_bitset = construct_bitset(&inner_keys);

    (real_bitset, inner_bitset)
}

/// §8.1 categorical split evaluator — **Task 2 fills this under TDD**. Currently a
/// compiling stub returning the "no split" sentinel so the Task-1 slice builds.
///
/// See the Task-2 doc for the full contract (`sum_hessian` pre-bump, the l2
/// asymmetry, and the [`super::primitives::bitonic_argsort_on`] ctr sort).
///
/// # Errors
/// [`ComputeError`] (Task 2) propagated from the ctr sort.
#[allow(clippy::too_many_arguments)]
pub fn find_best_threshold_categorical<R: cubecl::Runtime>(
    _client: &ComputeClient<R>,
    _hist: &[f64],
    _cfg: &GainConfig,
    _num_bin: i32,
    _offset: i32,
    _sum_gradient: f64,
    _sum_hessian: f64,
    _num_data: i32,
) -> Result<CategoricalSplit, ComputeError> {
    Ok(CategoricalSplit::none())
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
        let mut ctr: Vec<f64> = (1..7)
            .map(|t: i32| hist[(t as usize) << 1] / (hist[((t as usize) << 1) + 1] + cat_smooth))
            .collect();
        let n = ctr.len();
        ctr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in ctr.windows(2) {
            assert!(w[0] != w[1], "cat_manyvsmany ctr must be tie-free");
        }
        assert_eq!(n, 6);
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
        let (real, _inner) = set_real_threshold(&bins, &[-1, 0, 1, 2, 3], 0);
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
        let (real, _inner) = set_real_threshold(&bins, &[-1, 0, 1, 2, 3, 4, 5], 0);
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
        let (real, inner) = set_real_threshold(&[4], &bin_2_categorical, 0);
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
        let (real, _inner) = set_real_threshold(&[6, 5, 4], &bin_2_categorical, 0);
        assert_eq!(real, vec![56u32], "cat_manyvsmany real bitset != golden (56)");
    }

    #[test]
    fn set_real_threshold_out_of_range_bin_does_not_panic() {
        // Host bounds handling: `.get(bin).unwrap_or(bin)` — an out-of-range bin
        // falls back to the bin index itself, no panic, no OOB (threat T-22-06).
        let bin_2_categorical = [-1, 0, 1];
        let (real, _inner) = set_real_threshold(&[9], &bin_2_categorical, 0);
        assert_eq!(real, construct_bitset(&[9]));
    }
}
