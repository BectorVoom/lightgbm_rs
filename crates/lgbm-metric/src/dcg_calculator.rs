//! `DcgCalculator` — the shared DCG gain/discount tables for ranking, ported 1:1
//! from the C++ reference. Built ONCE at construction and consumed by BOTH the
//! ranking objectives (lambdarank/rank_xendcg, `lgbm-objective`) and the ranking
//! metrics (ndcg/map, this crate).
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/metric/dcg_calculator.cpp`:
//!   - `DefaultLabelGain` (:33-41): when `label_gain` is empty, fill `2^i - 1`
//!     for `i in 0..31` (entry 0 is `0.0`). `max_label = 31` to avoid overflow.
//!   - `Init` (:43-52): copy `label_gain` into `label_gain_`; build
//!     `discount_[i] = 1 / log2(2 + i)` for `i in 0..kMaxPosition` (10000). The
//!     tables are STATIC members in C++ — built ONCE; this struct mirrors that by
//!     building them once at [`DcgCalculator::new`] and never recomputing per query
//!     (RESEARCH Anti-Pattern: do NOT recompute the discount table per query).
//!   - `GetDiscount(k)` = `discount_[k]`.
//!   - `CalMaxDCGAtK` (:54-76): the ideal DCG at `k` over a query's labels (used by
//!     lambdarank's `inverse_max_dcgs_`).
//!   - `CalMaxDCG` (:78-107): the ideal DCG at a set of `ks` in one pass (used by
//!     the ndcg metric's per-query `inverse_max_dcgs_`).
//!   - `CalDCG` (:109-132): the realized DCG at a set of `ks` — stable-sort the
//!     query's rows by DESCENDING score, then accumulate `label_gain * discount`.
//!   - `CheckLabel` (:147-162): rank labels must be non-negative ints
//!     `< label_gain_.size()` — surfaced here as [`MetricError::LabelOutOfRange`].
//!
//! `deterministic = true` strips the C++ OpenMP; all reductions here are ordered
//! sequential f64 folds (the bit-exact deterministic anchor).

use crate::error::MetricError;

/// C++ `DCGCalculator::kMaxPosition` (`dcg_calculator.cpp:17`) — the discount
/// table size (max rows per query).
pub const K_MAX_POSITION: usize = 10000;

/// The shared DCG gain + discount tables, built once and consumed by the ranking
/// objective and metric.
#[derive(Debug, Clone, PartialEq)]
pub struct DcgCalculator {
    /// `label_gain_` — the per-label gain table (`2^i - 1` by default).
    label_gain: Vec<f64>,
    /// `discount_[i] = 1 / log2(2 + i)` for `i in 0..kMaxPosition` (built ONCE).
    discount: Vec<f64>,
}

impl DcgCalculator {
    /// C++ `DCGCalculator::DefaultLabelGain` (`dcg_calculator.cpp:33-41`): if
    /// `input` is empty, return `[0, 1, 3, 7, …, 2^30 - 1]` (31 entries). Otherwise
    /// return the input verbatim.
    pub fn default_label_gain(input: &[f64]) -> Vec<f64> {
        if !input.is_empty() {
            return input.to_vec();
        }
        // label_gain = 2^i - 1; max_label = 31 to avoid overflow.
        let max_label = 31;
        let mut g = Vec::with_capacity(max_label);
        g.push(0.0f64);
        for i in 1..max_label {
            // C++ static_cast<double>((1 << i) - 1).
            g.push(((1i64 << i) - 1) as f64);
        }
        g
    }

    /// C++ `DCGCalculator::DefaultEvalAt` (`dcg_calculator.cpp:20-31`): when
    /// `eval_at` is empty, default to `[1, 2, 3, 4, 5]`. Otherwise return the input
    /// verbatim (the `CHECK_GT(eval_at[i], 0)` is enforced at config parse time).
    pub fn default_eval_at(eval_at: &[i32]) -> Vec<i32> {
        if eval_at.is_empty() {
            (1..=5).collect()
        } else {
            eval_at.to_vec()
        }
    }

    /// C++ `DCGCalculator::Init` (`dcg_calculator.cpp:43-52`): copy the (possibly
    /// defaulted) `label_gain` and build the discount table ONCE. Call
    /// [`DcgCalculator::default_label_gain`] first to apply the fallback.
    pub fn new(label_gain: Vec<f64>) -> Self {
        let mut discount = Vec::with_capacity(K_MAX_POSITION);
        for i in 0..K_MAX_POSITION {
            // discount_[i] = 1.0 / log2(2.0 + i).
            discount.push(1.0 / (2.0 + i as f64).log2());
        }
        Self {
            label_gain,
            discount,
        }
    }

    /// The `label_gain_` table.
    pub fn label_gain(&self) -> &[f64] {
        &self.label_gain
    }

    /// C++ `DCGCalculator::GetDiscount(k)` = `discount_[k]`.
    #[inline]
    pub fn discount(&self, k: usize) -> f64 {
        self.discount[k]
    }

    /// C++ `DCGCalculator::CheckLabel` (`dcg_calculator.cpp:147-162`): every rank
    /// label must be a non-negative int `< label_gain_.size()`. Returns the first
    /// offending label as a typed error.
    ///
    /// # Errors
    /// [`MetricError::NonIntegerLabel`] when a label is not an integer, then
    /// [`MetricError::LabelOutOfRange`] when a label is negative or
    /// `>= label_gain.len()` (T-07-09-02, WR-02). Mirrors the C++ check ORDER
    /// (`dcg_calculator.cpp:148-161`): integer-ness first (`fabs(label -
    /// static_cast<int>(label)) > kEpsilon`), then non-negative, then range. The
    /// gain-table index is `static_cast<int>(label)`, which truncates toward zero,
    /// so a fractional label must be rejected before it silently picks a wrong gain.
    pub fn check_labels(&self, labels: &[f32]) -> Result<(), MetricError> {
        let n_gains = self.label_gain.len();
        // C++ `kEpsilon = 1e-15f` (score_t); the comparison is done in f32 with
        // `static_cast<int>` truncation toward zero (NOT `floor`).
        let eps = lgbm_core::types::K_EPSILON;
        for &l in labels {
            let delta = (l - (l as i32 as f32)).abs();
            if delta > eps {
                return Err(MetricError::NonIntegerLabel { label: l as f64 });
            }
            if l < 0.0 {
                return Err(MetricError::LabelOutOfRange {
                    label: l as i64,
                    num_gains: n_gains,
                });
            }
            let li = l as usize;
            if li >= n_gains {
                return Err(MetricError::LabelOutOfRange {
                    label: l as i64,
                    num_gains: n_gains,
                });
            }
        }
        Ok(())
    }

    /// C++ `DCGCalculator::CalMaxDCGAtK` (`dcg_calculator.cpp:54-76`): the ideal
    /// (maximum) DCG at position `k` over a single query's `labels`. Greedily places
    /// the highest-gain labels in the top positions.
    pub fn cal_max_dcg_at_k(&self, k: usize, labels: &[f32]) -> f64 {
        let num_data = labels.len();
        let mut ret = 0.0f64;
        let mut label_cnt = vec![0i64; self.label_gain.len()];
        for &l in labels {
            label_cnt[l as usize] += 1;
        }
        let mut top_label = self.label_gain.len() as i64 - 1;
        let k = k.min(num_data);
        for j in 0..k {
            while top_label > 0 && label_cnt[top_label as usize] <= 0 {
                top_label -= 1;
            }
            if top_label < 0 {
                break;
            }
            ret += self.discount[j] * self.label_gain[top_label as usize];
            label_cnt[top_label as usize] -= 1;
        }
        ret
    }

    /// C++ `DCGCalculator::CalMaxDCG` (`dcg_calculator.cpp:78-107`): the ideal DCG
    /// at each `k` in `ks` (assumed ascending) over a single query's `labels`, in
    /// one pass. Writes into `out` (length `ks.len()`).
    pub fn cal_max_dcg(&self, ks: &[i32], labels: &[f32], out: &mut [f64]) {
        let num_data = labels.len() as i32;
        let mut label_cnt = vec![0i64; self.label_gain.len()];
        for &l in labels {
            label_cnt[l as usize] += 1;
        }
        let mut cur_result = 0.0f64;
        let mut cur_left = 0i32;
        let mut top_label = self.label_gain.len() as i64 - 1;
        for (i, &k_raw) in ks.iter().enumerate() {
            let cur_k = k_raw.min(num_data);
            let mut j = cur_left;
            while j < cur_k {
                while top_label > 0 && label_cnt[top_label as usize] <= 0 {
                    top_label -= 1;
                }
                if top_label < 0 {
                    break;
                }
                cur_result += self.discount[j as usize] * self.label_gain[top_label as usize];
                label_cnt[top_label as usize] -= 1;
                j += 1;
            }
            out[i] = cur_result;
            cur_left = cur_k;
        }
    }

    /// C++ `DCGCalculator::CalDCG` (`dcg_calculator.cpp:109-132`): the realized DCG
    /// at each `k` in `ks` (assumed ascending), sorting the query's rows by
    /// DESCENDING `score` (stable). Writes into `out` (length `ks.len()`).
    pub fn cal_dcg(&self, ks: &[i32], labels: &[f32], score: &[f64], out: &mut [f64]) {
        let num_data = labels.len() as i32;
        // Stable sort by descending score (C++ std::stable_sort, score[a] > score[b]).
        let mut sorted_idx: Vec<usize> = (0..labels.len()).collect();
        // A stable sort with a "greater-than" key: use sort_by with reverse partial_cmp,
        // breaking ties by original index to match std::stable_sort's preservation.
        sorted_idx.sort_by(|&a, &b| {
            // descending: b vs a; stable_sort keeps original order on equal scores.
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut cur_result = 0.0f64;
        let mut cur_left = 0i32;
        for (i, &k_raw) in ks.iter().enumerate() {
            let cur_k = k_raw.min(num_data);
            let mut j = cur_left;
            while j < cur_k {
                let idx = sorted_idx[j as usize];
                cur_result += self.label_gain[labels[idx] as usize] * self.discount[j as usize];
                j += 1;
            }
            out[i] = cur_result;
            cur_left = cur_k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_label_gain_is_powers_of_two_minus_one() {
        let g = DcgCalculator::default_label_gain(&[]);
        assert_eq!(g.len(), 31);
        assert_eq!(g[0], 0.0);
        assert_eq!(g[1], 1.0); // 2^1 - 1
        assert_eq!(g[2], 3.0); // 2^2 - 1
        assert_eq!(g[3], 7.0); // 2^3 - 1
        assert_eq!(g[4], 15.0);
        assert_eq!(g[30], ((1i64 << 30) - 1) as f64);
        // A supplied label_gain is returned verbatim.
        let custom = vec![0.0, 2.0, 5.0];
        assert_eq!(DcgCalculator::default_label_gain(&custom), custom);
    }

    #[test]
    fn default_eval_at_is_one_through_five() {
        assert_eq!(DcgCalculator::default_eval_at(&[]), vec![1, 2, 3, 4, 5]);
        assert_eq!(DcgCalculator::default_eval_at(&[1, 3]), vec![1, 3]);
    }

    #[test]
    fn discount_table_built_once_matches_formula() {
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(&[]));
        // discount_[i] = 1 / log2(2 + i).
        for i in [0usize, 1, 2, 9, 100] {
            let expect = 1.0 / (2.0 + i as f64).log2();
            assert_eq!(dcg.discount(i).to_bits(), expect.to_bits(), "discount[{i}]");
        }
        // discount[0] = 1/log2(2) = 1.0.
        assert_eq!(dcg.discount(0), 1.0);
    }

    #[test]
    fn cal_max_dcg_at_k_places_top_labels_first() {
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(&[]));
        // labels [3, 1, 2, 0]: ideal order is gain(3),gain(2),gain(1),gain(0).
        let labels = [3.0f32, 1.0, 2.0, 0.0];
        let got = dcg.cal_max_dcg_at_k(4, &labels);
        let g = DcgCalculator::default_label_gain(&[]);
        let expect = g[3] * dcg.discount(0)
            + g[2] * dcg.discount(1)
            + g[1] * dcg.discount(2)
            + g[0] * dcg.discount(3);
        assert_eq!(got.to_bits(), expect.to_bits());
    }

    #[test]
    fn cal_dcg_sorts_by_descending_score() {
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(&[]));
        // rows: (label, score). Best score first.
        let labels = [1.0f32, 3.0, 0.0, 2.0];
        let score = [0.1f64, 0.9, 0.2, 0.5]; // desc order: idx1(0.9),idx3(0.5),idx2(0.2),idx0(0.1)
        let ks = [4i32];
        let mut out = [0.0f64];
        dcg.cal_dcg(&ks, &labels, &score, &mut out);
        let g = DcgCalculator::default_label_gain(&[]);
        let expect = g[3] * dcg.discount(0)  // idx1 label 3
            + g[2] * dcg.discount(1)         // idx3 label 2
            + g[0] * dcg.discount(2)         // idx2 label 0
            + g[1] * dcg.discount(3); // idx0 label 1
        assert_eq!(out[0].to_bits(), expect.to_bits());
    }

    #[test]
    fn cal_max_dcg_multi_k_one_pass() {
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(&[]));
        let labels = [3.0f32, 1.0, 2.0, 0.0];
        let ks = [1i32, 2, 4];
        let mut out = [0.0f64; 3];
        dcg.cal_max_dcg(&ks, &labels, &mut out);
        // out[0] = top-1, out[1] = top-2, out[2] = top-4 (cumulative).
        assert_eq!(out[0], dcg.cal_max_dcg_at_k(1, &labels));
        assert_eq!(out[1], dcg.cal_max_dcg_at_k(2, &labels));
        assert_eq!(out[2], dcg.cal_max_dcg_at_k(4, &labels));
    }

    #[test]
    fn check_labels_rejects_negative_and_oob() {
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(&[]));
        assert!(dcg.check_labels(&[0.0, 1.0, 2.0]).is_ok());
        let err = dcg.check_labels(&[0.0, -1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LabelOutOfRange { .. }));
        // label 31 with a 31-entry gain table -> out of [0, 31).
        let err = dcg.check_labels(&[31.0]).unwrap_err();
        assert!(matches!(err, MetricError::LabelOutOfRange { label: 31, .. }));
    }

    #[test]
    fn check_labels_rejects_non_integer() {
        // WR-02: a fractional rank label (2.7) would silently truncate to gain
        // index 2; C++ `DCGCalculator::CheckLabel` fatals on this. The integer
        // check fires BEFORE the range check (C++ order).
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(&[]));
        let err = dcg.check_labels(&[0.0, 2.7]).unwrap_err();
        assert!(matches!(err, MetricError::NonIntegerLabel { .. }));
        // Integer-valued floats pass.
        assert!(dcg.check_labels(&[0.0, 1.0, 3.0]).is_ok());
        // The integer check precedes the range check: a fractional out-of-range
        // value still reports NonIntegerLabel first (matches C++ order).
        let err = dcg.check_labels(&[31.5]).unwrap_err();
        assert!(matches!(err, MetricError::NonIntegerLabel { .. }));
    }
}
