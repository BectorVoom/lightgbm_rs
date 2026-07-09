//! Ranking metrics — `ndcg` and `map`, per-query evaluation over query
//! boundaries, ported 1:1 from the C++ reference. Both share the init-once
//! [`DcgCalculator`] gain/discount tables (ndcg directly; map uses only the
//! descending-score sort discipline).
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/metric/rank_metric.hpp` — `NDCGMetric`:
//!   - ctor (:21-29): `eval_at = config.eval_at` (DefaultEvalAt fallback);
//!     `DCGCalculator::Init(DefaultLabelGain(config.label_gain))`.
//!   - `Init` (:33-76): per-query `inverse_max_dcgs_[i][j]` = `1 / CalMaxDCG(...)`,
//!     or `-1` when the ideal DCG is `0` (all-negative query → ndcg defined as 1).
//!   - `factor_to_bigger_better` (:82-84) = `+1`.
//!   - `Eval` (:86-144): per query, if `inverse_max_dcgs_[i][0] <= 0` add `1` for
//!     every `k` (all-negative query); else add `CalDCG(...) * inverse_max_dcgs`.
//!     Average over `sum_query_weights_` (= `num_queries` unweighted).
//! - `LightGBM/src/metric/map_metric.hpp` — `MapMetric`:
//!   - ctor (:22-26): `eval_at` (DefaultEvalAt); no DCG table needed.
//!   - `Init` (:31-64): `npos_per_query_[i]` = count of `label > 0.5` per query.
//!   - `CalMapAtK` (:74-104): sort the query's rows by DESCENDING score, walk in
//!     rank order; on a positive (`label > 0.5`) increment `num_hit` and add
//!     `num_hit / (j + 1)`; AP@k = `sum_ap / min(npos, k)` (or `1` when `npos == 0`).
//!   - `factor_to_bigger_better` (:70-72) = `+1`.
//!
//! `deterministic = true` strips the C++ OpenMP; all reductions here are ordered
//! sequential f64 folds over the query list (the bit-exact deterministic anchor).

use crate::dcg_calculator::DcgCalculator;
use crate::error::MetricError;

/// Validate a `query_boundaries` prefix-sum array against `num_data` (threat
/// T-07-09-01): must start at `0`, be monotone non-decreasing, and end at
/// `num_data`. Returns the number of queries (`len - 1`) on success.
///
/// # Errors
/// [`MetricError::InvalidQueryBoundaries`] when any invariant is violated, so a
/// hostile boundary array can never feed a negative-length query to the per-query
/// loop or index out of bounds.
pub fn validate_query_boundaries(
    query_boundaries: &[i32],
    num_data: usize,
) -> Result<usize, MetricError> {
    if query_boundaries.is_empty() {
        return Err(MetricError::InvalidQueryBoundaries {
            what: "ranking tasks require non-empty query_boundaries".to_string(),
        });
    }
    if query_boundaries[0] != 0 {
        return Err(MetricError::InvalidQueryBoundaries {
            what: format!("query_boundaries[0] must be 0, got {}", query_boundaries[0]),
        });
    }
    for w in query_boundaries.windows(2) {
        if w[1] < w[0] {
            return Err(MetricError::InvalidQueryBoundaries {
                what: format!("query_boundaries must be non-decreasing ({} > {})", w[0], w[1]),
            });
        }
    }
    let last = *query_boundaries.last().unwrap();
    if last as usize != num_data {
        return Err(MetricError::InvalidQueryBoundaries {
            what: format!("query_boundaries must end at num_data ({last} != {num_data})"),
        });
    }
    Ok(query_boundaries.len() - 1)
}

/// The ranking-metric factory enum, mirroring the C++ string-keyed
/// `Metric::CreateMetric` ndcg/map arms. Both are per-query, multi-`@k`,
/// `factor_to_bigger_better == +1`.
#[derive(Debug, Clone, PartialEq)]
pub enum RankMetric {
    /// `ndcg` — normalized discounted cumulative gain, averaged over queries.
    Ndcg {
        /// The `@k` cutoffs (`config.eval_at`, DefaultEvalAt → `[1..=5]`).
        eval_at: Vec<i32>,
        /// The shared init-once DCG gain/discount tables.
        dcg: DcgCalculator,
    },
    /// `map` — mean average precision, averaged over queries.
    Map {
        /// The `@k` cutoffs (`config.eval_at`, DefaultEvalAt → `[1..=5]`).
        eval_at: Vec<i32>,
    },
}

impl RankMetric {
    /// Construct the `ndcg` metric: apply the DefaultEvalAt / DefaultLabelGain
    /// fallbacks and build the init-once [`DcgCalculator`].
    pub fn ndcg(eval_at: &[i32], label_gain: &[f64]) -> Self {
        let eval_at = DcgCalculator::default_eval_at(eval_at);
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(label_gain));
        RankMetric::Ndcg { eval_at, dcg }
    }

    /// Construct the `map` metric: apply the DefaultEvalAt fallback (no DCG table).
    pub fn map(eval_at: &[i32]) -> Self {
        RankMetric::Map {
            eval_at: DcgCalculator::default_eval_at(eval_at),
        }
    }

    /// Parse a ranking metric name into a [`RankMetric`]. Recognizes the C++
    /// `ndcg`/`map` aliases. Returns `Ok(None)` for non-ranking metric names so the
    /// caller can fall through to the regression/binary/etc. factories.
    pub fn parse(name: &str, eval_at: &[i32], label_gain: &[f64]) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "ndcg" | "lambdarank" => Some(Self::ndcg(eval_at, label_gain)),
            "map" | "mean_average_precision" => Some(Self::map(eval_at)),
            _ => None,
        }
    }

    /// The per-`@k` metric names (`ndcg@1`, `map@3`, …) in `eval_at` order.
    pub fn names(&self) -> Vec<String> {
        match self {
            RankMetric::Ndcg { eval_at, .. } => {
                eval_at.iter().map(|k| format!("ndcg@{k}")).collect()
            }
            RankMetric::Map { eval_at } => eval_at.iter().map(|k| format!("map@{k}")).collect(),
        }
    }

    /// C++ `factor_to_bigger_better` (`rank_metric.hpp:82`, `map_metric.hpp:70`) =
    /// `+1` for both (ndcg/map are utilities, bigger is better).
    pub fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    /// The `@k` cutoffs this metric evaluates.
    pub fn eval_at(&self) -> &[i32] {
        match self {
            RankMetric::Ndcg { eval_at, .. } => eval_at,
            RankMetric::Map { eval_at } => eval_at,
        }
    }

    /// Evaluate the metric over `score`/`labels` grouped by `query_boundaries`,
    /// returning one value per `@k` (averaged over queries).
    ///
    /// # Errors
    /// - [`MetricError::LengthMismatch`] when `score`/`labels` disagree.
    /// - [`MetricError::InvalidQueryBoundaries`] (T-07-09-01) on a malformed
    ///   boundary array.
    /// - [`MetricError::LabelOutOfRange`] (ndcg only, T-07-09-02) on a label outside
    ///   the gain table.
    pub fn eval(
        &self,
        score: &[f64],
        labels: &[f32],
        query_boundaries: &[i32],
    ) -> Result<Vec<f64>, MetricError> {
        let n = score.len();
        if labels.len() != n {
            return Err(MetricError::LengthMismatch { expected: n, actual: labels.len() });
        }
        let num_queries = validate_query_boundaries(query_boundaries, n)?;
        let sum_query_weights = num_queries as f64;
        match self {
            RankMetric::Ndcg { eval_at, dcg } => {
                dcg.check_labels(labels)?;
                let nk = eval_at.len();
                let mut result = vec![0.0f64; nk];
                let mut tmp_dcg = vec![0.0f64; nk];
                let mut inv_max = vec![0.0f64; nk];
                for q in 0..num_queries {
                    let start = query_boundaries[q] as usize;
                    let end = query_boundaries[q + 1] as usize;
                    let q_labels = &labels[start..end];
                    let q_score = &score[start..end];
                    // inverse_max_dcgs for this query (Init :60-75).
                    dcg.cal_max_dcg(eval_at, q_labels, &mut inv_max);
                    for v in inv_max.iter_mut() {
                        if *v > 0.0 {
                            *v = 1.0 / *v;
                        } else {
                            // all-negative query marker.
                            *v = -1.0;
                        }
                    }
                    if inv_max[0] <= 0.0 {
                        // all-negative query: ndcg = 1 for every k.
                        for r in result.iter_mut() {
                            *r += 1.0;
                        }
                    } else {
                        dcg.cal_dcg(eval_at, q_labels, q_score, &mut tmp_dcg);
                        for j in 0..nk {
                            result[j] += tmp_dcg[j] * inv_max[j];
                        }
                    }
                }
                for r in result.iter_mut() {
                    *r /= sum_query_weights;
                }
                Ok(result)
            }
            RankMetric::Map { eval_at } => {
                let nk = eval_at.len();
                let mut result = vec![0.0f64; nk];
                let mut tmp_map = vec![0.0f64; nk];
                for q in 0..num_queries {
                    let start = query_boundaries[q] as usize;
                    let end = query_boundaries[q + 1] as usize;
                    let q_labels = &labels[start..end];
                    let q_score = &score[start..end];
                    let npos = q_labels.iter().filter(|&&l| l > 0.5).count() as i32;
                    cal_map_at_k(eval_at, npos, q_labels, q_score, &mut tmp_map);
                    for j in 0..nk {
                        result[j] += tmp_map[j];
                    }
                }
                for r in result.iter_mut() {
                    *r /= sum_query_weights;
                }
                Ok(result)
            }
        }
    }
}

/// C++ `MapMetric::CalMapAtK` (`map_metric.hpp:74-104`): average precision at each
/// `k` for one query. `npos` is the number of positive (`label > 0.5`) rows.
fn cal_map_at_k(ks: &[i32], npos: i32, label: &[f32], score: &[f64], out: &mut [f64]) {
    let num_data = label.len() as i32;
    // Stable sort by descending score (C++ std::stable_sort, score[a] > score[b]).
    let mut sorted_idx: Vec<usize> = (0..label.len()).collect();
    sorted_idx.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut num_hit = 0i32;
    let mut sum_ap = 0.0f64;
    let mut cur_left = 0i32;
    for (i, &k_raw) in ks.iter().enumerate() {
        let cur_k = k_raw.min(num_data);
        let mut j = cur_left;
        while j < cur_k {
            let idx = sorted_idx[j as usize];
            if label[idx] > 0.5 {
                num_hit += 1;
                // sum_ap += num_hit / (j + 1.0f) — C++ uses an f32 denominator (j+1.0f).
                sum_ap += num_hit as f64 / (j as f32 + 1.0f32) as f64;
            }
            j += 1;
        }
        if npos > 0 {
            out[i] = sum_ap / (npos.min(cur_k)) as f64;
        } else {
            out[i] = 1.0;
        }
        cur_left = cur_k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_is_plus_one_for_both() {
        let ndcg = RankMetric::ndcg(&[1, 3], &[]);
        let map = RankMetric::map(&[1, 3]);
        assert_eq!(ndcg.factor_to_bigger_better(), 1.0);
        assert_eq!(map.factor_to_bigger_better(), 1.0);
    }

    #[test]
    fn parse_aliases() {
        assert!(matches!(
            RankMetric::parse("ndcg", &[], &[]),
            Some(RankMetric::Ndcg { .. })
        ));
        assert!(matches!(
            RankMetric::parse("map", &[], &[]),
            Some(RankMetric::Map { .. })
        ));
        assert!(RankMetric::parse("l2", &[], &[]).is_none());
    }

    #[test]
    fn default_eval_at_drives_names() {
        let ndcg = RankMetric::ndcg(&[], &[]);
        assert_eq!(
            ndcg.names(),
            vec!["ndcg@1", "ndcg@2", "ndcg@3", "ndcg@4", "ndcg@5"]
        );
        let map = RankMetric::map(&[2]);
        assert_eq!(map.names(), vec!["map@2"]);
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // One query, perfectly ranked (labels descending with score descending).
        let ndcg = RankMetric::ndcg(&[3], &[]);
        let labels = [3.0f32, 2.0, 1.0, 0.0];
        let score = [0.9f64, 0.8, 0.7, 0.6];
        let qb = [0i32, 4];
        let got = ndcg.eval(&score, &labels, &qb).unwrap();
        // Perfect ranking => ndcg@3 == 1.
        assert!((got[0] - 1.0).abs() < 1e-12, "ndcg = {}", got[0]);
    }

    #[test]
    fn ndcg_all_negative_query_is_one() {
        // All-zero labels: inverse_max_dcg <= 0 => ndcg defined as 1.
        let ndcg = RankMetric::ndcg(&[2], &[]);
        let labels = [0.0f32, 0.0, 0.0];
        let score = [0.1f64, 0.5, 0.2];
        let qb = [0i32, 3];
        let got = ndcg.eval(&score, &labels, &qb).unwrap();
        assert_eq!(got[0], 1.0);
    }

    #[test]
    fn ndcg_imperfect_ranking_less_than_one() {
        // Worst row scored highest => ndcg < 1.
        let ndcg = RankMetric::ndcg(&[3], &[]);
        let labels = [3.0f32, 2.0, 1.0, 0.0];
        let score = [0.1f64, 0.2, 0.3, 0.9]; // label-0 row scored highest.
        let qb = [0i32, 4];
        let got = ndcg.eval(&score, &labels, &qb).unwrap();
        assert!(got[0] < 1.0 && got[0] > 0.0, "ndcg = {}", got[0]);
    }

    #[test]
    fn map_average_precision_hand_computed() {
        // One query, binary labels [1,0,1,0]; scores rank them as idx0,idx2,idx1,idx3.
        // map@4: hits at ranks 1 and 2 => AP = (1/1 + 2/2)/2 = 1.0.
        let map = RankMetric::map(&[4]);
        let labels = [1.0f32, 0.0, 1.0, 0.0];
        let score = [0.9f64, 0.3, 0.8, 0.1];
        let qb = [0i32, 4];
        let got = map.eval(&score, &labels, &qb).unwrap();
        assert!((got[0] - 1.0).abs() < 1e-12, "map = {}", got[0]);
    }

    #[test]
    fn map_npos_zero_is_one() {
        let map = RankMetric::map(&[2]);
        let labels = [0.0f32, 0.0];
        let score = [0.5f64, 0.1];
        let qb = [0i32, 2];
        let got = map.eval(&score, &labels, &qb).unwrap();
        assert_eq!(got[0], 1.0);
    }

    #[test]
    fn multi_query_averages() {
        // Two queries, each perfectly ranked => average ndcg = 1.
        let ndcg = RankMetric::ndcg(&[2], &[]);
        let labels = [2.0f32, 1.0, 3.0, 0.0];
        let score = [0.9f64, 0.1, 0.9, 0.1];
        let qb = [0i32, 2, 4];
        let got = ndcg.eval(&score, &labels, &qb).unwrap();
        assert!((got[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_query_boundaries_rejected() {
        let ndcg = RankMetric::ndcg(&[1], &[]);
        let labels = [1.0f32, 0.0];
        let score = [0.5f64, 0.1];
        // does not end at num_data.
        assert!(matches!(
            ndcg.eval(&score, &labels, &[0, 1]).unwrap_err(),
            MetricError::InvalidQueryBoundaries { .. }
        ));
        // does not start at 0.
        assert!(matches!(
            ndcg.eval(&score, &labels, &[1, 2]).unwrap_err(),
            MetricError::InvalidQueryBoundaries { .. }
        ));
        // non-monotonic.
        assert!(matches!(
            ndcg.eval(&[0.0; 3], &[0.0f32; 3], &[0, 3, 2]).unwrap_err(),
            MetricError::InvalidQueryBoundaries { .. }
        ));
    }

    #[test]
    fn ndcg_negative_label_rejected() {
        let ndcg = RankMetric::ndcg(&[1], &[]);
        let labels = [-1.0f32, 0.0];
        let score = [0.5f64, 0.1];
        let qb = [0i32, 2];
        assert!(matches!(
            ndcg.eval(&score, &labels, &qb).unwrap_err(),
            MetricError::LabelOutOfRange { .. }
        ));
    }
}
