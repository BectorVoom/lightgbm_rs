//! Ranking objectives — `lambdarank` (LambdaRank with NDCG) and `rank_xendcg`
//! (cross-entropy NDCG), ported 1:1 from the C++ reference. Both are per-query
//! pairwise/list objectives over `query_boundaries` (captured at construction,
//! mirroring [`crate::multiclass::MulticlassSoftmax`]'s construction-captures-
//! metadata shape), and lambdarank shares the init-once [`DcgCalculator`] gain/
//! discount tables with the ndcg metric (D-02/D-03).
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/objective/rank_objective.hpp`:
//!   - `RankingObjective::GetGradients` (:59-91): iterate queries; per query call
//!     `GetGradientsForOneQuery` over the `[start, start+cnt)` row slice.
//!   - `LambdarankNDCG` ctor (:140-154): `sigmoid`, `lambdarank_norm`,
//!     `lambdarank_truncation_level`; `DCGCalculator::Init(DefaultLabelGain(...))`;
//!     `sigmoid > 0` fatal → typed error.
//!   - `LambdarankNDCG::Init` (:161-178): per-query `inverse_max_dcgs_[i]` =
//!     `1 / CalMaxDCGAtK(truncation_level, ...)`; `ConstructSigmoidTable`.
//!   - `GetGradientsForOneQuery` (:180-266): stable-sort the query's rows by
//!     DESCENDING score; for pairs with at least one doc above the truncation level
//!     and DIFFERENT labels, accumulate the pairwise lambda/hessian using the
//!     sigmoid lookup table + the `delta_pair_NDCG` (dcg_gap · paired_discount ·
//!     inverse_max_dcg). Optional `norm_` rescale by `log2(1 + sum_lambdas) /
//!     sum_lambdas`.
//!   - `ConstructSigmoidTable` (:281-294): `_sigmoid_bins = 1024 * 1024` entries,
//!     `1 / (1 + exp(score · sigmoid))` over `[min_sigmoid_input/sigmoid/2, -...]`.
//!   - `RankXENDCG::Init` (:387-392): one `Random(seed + i)` per query.
//!   - `RankXENDCG::GetGradientsForOneQuery` (:394-446): softmax the scores; per
//!     row draw `Phi(label, NextFloat())` to form the ground-truth distribution,
//!     then accumulate first/second/third-order gradient terms. The RNG draw order
//!     (one `NextFloat()` per row, query rng `seed + query_id`) is load-bearing —
//!     the rank_xendcg objective_seed RNG-replay golden freezes it.
//!   - `Phi` (:448-450): `Common::Pow(2, (int)l) - g`.
//!
//! `deterministic = true` strips the C++ OpenMP (`schedule(guided)` over queries);
//! the per-query loop here is an ordered sequential f64 fold — the bit-exact
//! deterministic anchor. Position-bias (`num_position_ids_ > 0`) and per-row
//! `weights_` are recorded in the formula but out of scope for the W8 spine
//! (unweighted, no positions); they are a documented later surface.

use lgbm_core::random::Random;
use lgbm_core::types::K_EPSILON;
use lgbm_metric::dcg_calculator::DcgCalculator;
use lgbm_model::objective::softmax;

use crate::error::ObjectiveError;

/// C++ `kMinScore = -inf` sentinel used by lambdarank to skip dropped docs.
const K_MIN_SCORE: f64 = f64::NEG_INFINITY;

/// C++ `LambdarankNDCG::_sigmoid_bins = 1024 * 1024` (rank_objective.hpp:365).
const SIGMOID_BINS: usize = 1024 * 1024;

/// Validate a `query_boundaries` prefix-sum array against `num_data` (threat
/// T-07-09-01). Mirrors [`lgbm_metric::rank::validate_query_boundaries`] but
/// returns an [`ObjectiveError`].
fn validate_query_boundaries(qb: &[i32], num_data: usize) -> Result<usize, ObjectiveError> {
    if qb.is_empty() {
        return Err(ObjectiveError::Unsupported {
            name: "ranking objectives require non-empty query_boundaries".to_string(),
        });
    }
    if qb[0] != 0 {
        return Err(ObjectiveError::Unsupported {
            name: format!("query_boundaries[0] must be 0, got {}", qb[0]),
        });
    }
    for w in qb.windows(2) {
        if w[1] < w[0] {
            return Err(ObjectiveError::Unsupported {
                name: format!("query_boundaries must be non-decreasing ({} > {})", w[0], w[1]),
            });
        }
    }
    let last = *qb.last().unwrap() as usize;
    if last != num_data {
        return Err(ObjectiveError::Unsupported {
            name: format!("query_boundaries must end at num_data ({last} != {num_data})"),
        });
    }
    Ok(qb.len() - 1)
}

/// Guard rank labels `>= 0` (threat T-07-09-02) — the rank label range guard.
/// (The full `< label_gain.size()` check is owned by [`DcgCalculator::check_labels`]
/// inside lambdarank; this is the cheap non-negativity gate that both objectives
/// share.)
fn check_rank_labels(labels: &[f32], objective: &str) -> Result<(), ObjectiveError> {
    for &l in labels {
        if l < 0.0 {
            return Err(ObjectiveError::LabelRange {
                label: l as f64,
                objective: objective.to_string(),
                reason: "ranking labels must be non-negative".to_string(),
            });
        }
    }
    Ok(())
}

/// The `lambdarank` (LambdaRank with NDCG) objective, training side.
#[derive(Debug, Clone, PartialEq)]
pub struct Lambdarank {
    /// `sigmoid_` (`config.sigmoid`, `> 0`).
    sigmoid: f64,
    /// `norm_` (`config.lambdarank_norm`).
    norm: bool,
    /// `truncation_level_` (`config.lambdarank_truncation_level`).
    truncation_level: i32,
    /// `query_boundaries_` captured at construction (prefix-sum, len num_queries+1).
    query_boundaries: Vec<i32>,
    /// `label_` — the per-row labels captured at construction (the C++ `label_`
    /// pointer; ranking labels are static for the whole training run).
    query_labels: Vec<f32>,
    /// The shared init-once DCG gain/discount tables.
    dcg: DcgCalculator,
    /// `inverse_max_dcgs_[q]` — `1 / CalMaxDCGAtK(truncation_level, query q)`.
    inverse_max_dcgs: Vec<f64>,
    /// `sigmoid_table_` — `1 / (1 + exp(score · sigmoid))` over the input range.
    sigmoid_table: Vec<f64>,
    /// `min_sigmoid_input_` (after the ctor `/ sigmoid / 2` adjustment).
    min_sigmoid_input: f64,
    /// `max_sigmoid_input_` (= `-min_sigmoid_input_`).
    max_sigmoid_input: f64,
    /// `sigmoid_table_idx_factor_`.
    sigmoid_table_idx_factor: f64,
}

impl Lambdarank {
    /// Construct + run the C++ `Init`: validate `sigmoid > 0` and labels `>= 0`,
    /// apply the DefaultLabelGain fallback, build the init-once [`DcgCalculator`],
    /// compute per-query `inverse_max_dcgs_`, and build the sigmoid lookup table.
    ///
    /// # Errors
    /// - [`ObjectiveError::Unsupported`] when `sigmoid <= 0` or `query_boundaries`
    ///   are malformed (T-07-09-01).
    /// - [`ObjectiveError::LabelRange`] / [`lgbm_metric::error::MetricError`]-derived
    ///   when a rank label is out of range (T-07-09-02).
    pub fn new(
        sigmoid: f64,
        norm: bool,
        truncation_level: i32,
        label_gain: &[f64],
        labels: &[f32],
        query_boundaries: &[i32],
    ) -> Result<Self, ObjectiveError> {
        if sigmoid <= 0.0 {
            return Err(ObjectiveError::Unsupported {
                name: format!("lambdarank sigmoid {sigmoid} must be > 0"),
            });
        }
        let num_queries = validate_query_boundaries(query_boundaries, labels.len())?;
        check_rank_labels(labels, "lambdarank")?;
        let dcg = DcgCalculator::new(DcgCalculator::default_label_gain(label_gain));
        // DCGCalculator::CheckLabel — also bounds-check against the gain table.
        dcg.check_labels(labels).map_err(|e| {
            let (label, reason) = match e {
                lgbm_metric::error::MetricError::LabelOutOfRange { label, .. } => {
                    (label as f64, "ranking label exceeds the label_gain table size")
                }
                // WR-02: a fractional rank label would silently truncate to a wrong
                // gain index (C++ `DCGCalculator::CheckLabel` fatals first on this).
                lgbm_metric::error::MetricError::NonIntegerLabel { label } => {
                    (label, "ranking label must be an integer")
                }
                _ => (f64::NAN, "ranking label failed DCG validation"),
            };
            ObjectiveError::LabelRange {
                label,
                objective: "lambdarank".to_string(),
                reason: reason.to_string(),
            }
        })?;

        // Per-query inverse max DCG at the truncation level (Init :165-175).
        let mut inverse_max_dcgs = Vec::with_capacity(num_queries);
        for q in 0..num_queries {
            let start = query_boundaries[q] as usize;
            let end = query_boundaries[q + 1] as usize;
            let mut m = dcg.cal_max_dcg_at_k(truncation_level as usize, &labels[start..end]);
            if m > 0.0 {
                m = 1.0 / m;
            }
            inverse_max_dcgs.push(m);
        }

        // ConstructSigmoidTable (rank_objective.hpp:281-294). Note the in-place
        // mutation of min_sigmoid_input_ in the C++ ctor: `min = min / sigmoid / 2`.
        let mut min_sigmoid_input = -50.0f64; // C++ default min_sigmoid_input_ = -50.
        min_sigmoid_input = min_sigmoid_input / sigmoid / 2.0;
        let max_sigmoid_input = -min_sigmoid_input;
        let sigmoid_table_idx_factor =
            SIGMOID_BINS as f64 / (max_sigmoid_input - min_sigmoid_input);
        let mut sigmoid_table = Vec::with_capacity(SIGMOID_BINS);
        for i in 0..SIGMOID_BINS {
            let score = i as f64 / sigmoid_table_idx_factor + min_sigmoid_input;
            sigmoid_table.push(1.0 / (1.0 + (score * sigmoid).exp()));
        }

        Ok(Self {
            sigmoid,
            norm,
            truncation_level,
            query_boundaries: query_boundaries.to_vec(),
            query_labels: labels.to_vec(),
            dcg,
            inverse_max_dcgs,
            sigmoid_table,
            min_sigmoid_input,
            max_sigmoid_input,
            sigmoid_table_idx_factor,
        })
    }

    /// C++ `LambdarankNDCG::GetSigmoid` (rank_objective.hpp:268-279) — the lookup.
    #[inline]
    fn get_sigmoid(&self, score: f64) -> f64 {
        if score <= self.min_sigmoid_input {
            self.sigmoid_table[0]
        } else if score >= self.max_sigmoid_input {
            self.sigmoid_table[SIGMOID_BINS - 1]
        } else {
            let idx = ((score - self.min_sigmoid_input) * self.sigmoid_table_idx_factor) as usize;
            self.sigmoid_table[idx]
        }
    }

    /// C++ `ObjectiveFunction::NumModelPerIteration` = 1 (ranking grows one tree per
    /// iteration over the whole corpus).
    pub fn num_model_per_iteration(&self) -> i32 {
        1
    }

    /// C++ `RankingObjective::GetGradients` (:59-91) → per-query
    /// `GetGradientsForOneQuery` (:180-266). `score` is the f64 raw score over all
    /// rows; `gradients`/`hessians` (f32) receive the per-row lambdas/hessians.
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] (V5) when the slices disagree with the
    /// corpus length.
    pub fn get_gradients(
        &self,
        score: &[f64],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let n = score.len();
        if gradients.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: gradients.len() });
        }
        if hessians.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: hessians.len() });
        }
        let label_gain = self.dcg.label_gain();
        let num_queries = self.query_boundaries.len() - 1;
        for q in 0..num_queries {
            let start = self.query_boundaries[q] as usize;
            let cnt = self.query_boundaries[q + 1] as usize - start;
            self.gradients_for_one_query(
                q,
                cnt,
                &score[start..start + cnt],
                &mut gradients[start..start + cnt],
                &mut hessians[start..start + cnt],
                label_gain,
            );
        }
        Ok(())
    }

    /// C++ `LambdarankNDCG::GetGradientsForOneQuery` (rank_objective.hpp:180-266),
    /// ported 1:1. `label` is read from the captured construction labels via the
    /// caller-supplied `labels` slice — but the per-query label is captured here:
    /// the labels live in `self`-adjacent storage, so this takes them as a slice.
    #[allow(clippy::too_many_arguments)]
    fn gradients_for_one_query(
        &self,
        query_id: usize,
        cnt: usize,
        score: &[f64],
        lambdas: &mut [f32],
        hessians: &mut [f32],
        label_gain: &[f64],
    ) {
        // The labels for this query are needed; fetch them from the stored copy.
        let q_labels = &self.query_labels[self.query_boundaries[query_id] as usize
            ..self.query_boundaries[query_id] as usize + cnt];
        let inverse_max_dcg = self.inverse_max_dcgs[query_id];
        for i in 0..cnt {
            lambdas[i] = 0.0;
            hessians[i] = 0.0;
        }
        // sorted_idx by DESCENDING score (stable).
        let mut sorted_idx: Vec<usize> = (0..cnt).collect();
        sorted_idx.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        // best/worst score (skipping a kMinScore worst).
        let best_score = score[sorted_idx[0]];
        let mut worst_idx = cnt - 1;
        if worst_idx > 0 && score[sorted_idx[worst_idx]] == K_MIN_SCORE {
            worst_idx -= 1;
        }
        let worst_score = score[sorted_idx[worst_idx]];
        let mut sum_lambdas = 0.0f64;
        let trunc = self.truncation_level as usize;
        let mut i = 0usize;
        while i + 1 < cnt && i < trunc {
            if score[sorted_idx[i]] == K_MIN_SCORE {
                i += 1;
                continue;
            }
            for j in (i + 1)..cnt {
                if score[sorted_idx[j]] == K_MIN_SCORE {
                    continue;
                }
                // skip pairs with the same label.
                if q_labels[sorted_idx[i]] == q_labels[sorted_idx[j]] {
                    continue;
                }
                let (high_rank, low_rank) =
                    if q_labels[sorted_idx[i]] > q_labels[sorted_idx[j]] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                let high = sorted_idx[high_rank];
                let high_label = q_labels[high] as usize;
                let high_score = score[high];
                let high_label_gain = label_gain[high_label];
                let high_discount = self.dcg.discount(high_rank);
                let low = sorted_idx[low_rank];
                let low_label = q_labels[low] as usize;
                let low_score = score[low];
                let low_label_gain = label_gain[low_label];
                let low_discount = self.dcg.discount(low_rank);

                let delta_score = high_score - low_score;
                let dcg_gap = high_label_gain - low_label_gain;
                let paired_discount = (high_discount - low_discount).abs();
                let mut delta_pair_ndcg = dcg_gap * paired_discount * inverse_max_dcg;
                if self.norm && best_score != worst_score {
                    // C++ uses 0.01f + fabs(delta_score) (f32 literal).
                    delta_pair_ndcg /= (0.01f32 as f64) + delta_score.abs();
                }
                let mut p_lambda = self.get_sigmoid(delta_score);
                let mut p_hessian = p_lambda * (1.0 - p_lambda);
                p_lambda *= -self.sigmoid * delta_pair_ndcg;
                p_hessian *= self.sigmoid * self.sigmoid * delta_pair_ndcg;
                lambdas[low] -= p_lambda as f32;
                hessians[low] += p_hessian as f32;
                lambdas[high] += p_lambda as f32;
                hessians[high] += p_hessian as f32;
                sum_lambdas -= 2.0 * p_lambda;
            }
            i += 1;
        }
        if self.norm && sum_lambdas > 0.0 {
            let norm_factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
            for i in 0..cnt {
                lambdas[i] = (lambdas[i] as f64 * norm_factor) as f32;
                hessians[i] = (hessians[i] as f64 * norm_factor) as f32;
            }
        }
    }
}

/// C++ `Common::Pow(2, power)` for an integer power (`common.h:248`) — repeated
/// multiplication (NOT `powf`), matching the C++ bit-for-bit for small int powers.
/// Used by rank_xendcg's `Phi`.
fn pow2_int(power: i32) -> f64 {
    let mut ret = 1.0f64;
    let mut base = 2.0f64;
    let mut p = power;
    // C++ Common::Pow: while (power > 0) { if (power & 1) ret *= base; power >>= 1; base *= base; }
    while p > 0 {
        if p & 1 == 1 {
            ret *= base;
        }
        p >>= 1;
        base *= base;
    }
    ret
}

/// The `rank_xendcg` (cross-entropy NDCG) objective, training side.
///
/// Per-query list objective: softmax the scores into `rho`, draw a per-row gamma
/// `Phi(label, NextFloat())` to form the ground-truth distribution, then
/// accumulate first/second/third-order gradient terms. The per-query RNG
/// (`Random(objective_seed + query_id)`, one `NextFloat()` per row in row order) is
/// the load-bearing draw the objective_seed RNG-replay golden freezes.
#[derive(Debug, Clone, PartialEq)]
pub struct RankXendcg {
    /// `seed_` (`config.objective_seed`).
    seed: i32,
    /// `query_boundaries_` captured at construction.
    query_boundaries: Vec<i32>,
}

impl RankXendcg {
    /// Construct + validate: query boundaries well-formed (T-07-09-01) and labels
    /// `>= 0` (T-07-09-02). The per-query `Random(seed + q)` instances are
    /// constructed fresh on each `get_gradients` call (the C++ `rands_` are built in
    /// `Init` and the objective is invoked once per iteration; re-seeding from
    /// `seed + q` each iteration is faithful because the C++ also re-draws from the
    /// SAME `seed + q` stream — note: the C++ `rands_` ADVANCE across iterations, so
    /// see the note in [`Self::get_gradients`]).
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] (malformed boundaries) /
    /// [`ObjectiveError::LabelRange`] (negative label).
    pub fn new(
        seed: i32,
        labels: &[f32],
        query_boundaries: &[i32],
    ) -> Result<Self, ObjectiveError> {
        validate_query_boundaries(query_boundaries, labels.len())?;
        check_rank_labels(labels, "rank_xendcg")?;
        Ok(Self {
            seed,
            query_boundaries: query_boundaries.to_vec(),
        })
    }

    /// C++ `ObjectiveFunction::NumModelPerIteration` = 1.
    pub fn num_model_per_iteration(&self) -> i32 {
        1
    }

    /// C++ `RankXENDCG::GetGradients` over the per-query RNG. `rands` is supplied by
    /// the caller so the RNG state ADVANCES across iterations exactly as the C++
    /// `mutable std::vector<Random> rands_` does (built once in `Init`, advanced on
    /// every `GetGradients`). Use [`Self::make_rands`] to build the initial vector.
    ///
    /// `labels` is the per-row label slice (the C++ `label_` pointer).
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] (V5) on slice disagreement.
    pub fn get_gradients(
        &self,
        score: &[f64],
        labels: &[f32],
        rands: &mut [Random],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let n = score.len();
        if labels.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: labels.len() });
        }
        if gradients.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: gradients.len() });
        }
        if hessians.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: hessians.len() });
        }
        let num_queries = self.query_boundaries.len() - 1;
        // `q` indexes query_boundaries[q], query_boundaries[q+1] AND rands[q]
        // simultaneously — a faithful per-query dispatch (C++ RankingObjective::
        // GetGradients), not expressible as a single-collection iterator.
        #[allow(clippy::needless_range_loop)]
        for q in 0..num_queries {
            let start = self.query_boundaries[q] as usize;
            let cnt = self.query_boundaries[q + 1] as usize - start;
            Self::gradients_for_one_query(
                cnt,
                &labels[start..start + cnt],
                &score[start..start + cnt],
                &mut rands[q],
                &mut gradients[start..start + cnt],
                &mut hessians[start..start + cnt],
            );
        }
        Ok(())
    }

    /// Build the per-query `Random(seed + q)` vector (the C++ `rands_` built in
    /// `Init`, rank_objective.hpp:389-391). Construct ONCE per training run and pass
    /// to every [`Self::get_gradients`] call so the stream advances across iters.
    pub fn make_rands(&self) -> Vec<Random> {
        let num_queries = self.query_boundaries.len() - 1;
        (0..num_queries as i32)
            .map(|q| Random::new(self.seed + q))
            .collect()
    }

    /// C++ `RankXENDCG::GetGradientsForOneQuery` (rank_objective.hpp:394-446).
    fn gradients_for_one_query(
        cnt: usize,
        label: &[f32],
        score: &[f64],
        rng: &mut Random,
        lambdas: &mut [f32],
        hessians: &mut [f32],
    ) {
        // Skip groups with too few items.
        if cnt <= 1 {
            for i in 0..cnt {
                lambdas[i] = 0.0;
                hessians[i] = 0.0;
            }
            return;
        }
        // rho = Softmax(score).
        let mut rho = vec![0.0f64; cnt];
        softmax(score, &mut rho);

        // params[i] = Phi(label[i], NextFloat()); inv_denominator = 1/max(eps, Σ params).
        let mut params = vec![0.0f64; cnt];
        let mut inv_denominator = 0.0f64;
        for i in 0..cnt {
            params[i] = phi(label[i], rng.next_float() as f64);
            inv_denominator += params[i];
        }
        inv_denominator = 1.0 / (K_EPSILON as f64).max(inv_denominator);

        // First order terms.
        let mut sum_l1 = 0.0f64;
        for i in 0..cnt {
            let term = -params[i] * inv_denominator + rho[i];
            lambdas[i] = term as f32;
            params[i] = term / (1.0 - rho[i]);
            sum_l1 += params[i];
        }
        // Second order terms.
        let mut sum_l2 = 0.0f64;
        for i in 0..cnt {
            let term = rho[i] * (sum_l1 - params[i]);
            lambdas[i] += term as f32;
            params[i] = term / (1.0 - rho[i]);
            sum_l2 += params[i];
        }
        // Third order terms + hessian.
        for i in 0..cnt {
            lambdas[i] += (rho[i] * (sum_l2 - params[i])) as f32;
            hessians[i] = (rho[i] * (1.0 - rho[i])) as f32;
        }
    }
}

/// C++ `RankXENDCG::Phi(l, g) = Common::Pow(2, (int)l) - g` (rank_objective.hpp:448).
fn phi(l: f32, g: f64) -> f64 {
    pow2_int(l as i32) - g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_query() -> (Vec<f32>, Vec<i32>) {
        // One query, 4 docs, graded labels.
        (vec![3.0f32, 2.0, 1.0, 0.0], vec![0i32, 4])
    }

    #[test]
    fn lambdarank_rejects_nonpositive_sigmoid() {
        let (labels, qb) = perfect_query();
        assert!(Lambdarank::new(0.0, true, 30, &[], &labels, &qb).is_err());
        assert!(Lambdarank::new(-1.0, true, 30, &[], &labels, &qb).is_err());
    }

    #[test]
    fn lambdarank_rejects_negative_label() {
        let labels = vec![1.0f32, -1.0];
        let qb = vec![0i32, 2];
        assert!(matches!(
            Lambdarank::new(1.0, true, 30, &[], &labels, &qb).unwrap_err(),
            ObjectiveError::LabelRange { .. }
        ));
    }

    #[test]
    fn lambdarank_perfect_ranking_zero_lambdas_on_concordant_pairs() {
        // When score is already perfectly concordant with labels, lambdas push the
        // already-correct order — the sum of lambdas across a doc is bounded; the key
        // invariant is grad/hess are finite and zero-sum within a query.
        let (labels, qb) = perfect_query();
        let obj = Lambdarank::new(1.0, false, 30, &[], &labels, &qb).unwrap();
        // perfectly concordant scores.
        let score = vec![3.0f64, 2.0, 1.0, 0.0];
        let mut grad = vec![0.0f32; 4];
        let mut hess = vec![0.0f32; 4];
        obj.get_gradients(&score, &mut grad, &mut hess).unwrap();
        for (g, h) in grad.iter().zip(&hess) {
            assert!(g.is_finite() && h.is_finite());
            assert!(*h >= 0.0, "hessian must be non-negative");
        }
        // Within a query, sum of lambdas is ~0 (each pair contributes +p to high, -p
        // to low) modulo norm.
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-4, "within-query lambdas sum ~0, got {sum}");
    }

    #[test]
    fn lambdarank_misranked_pushes_gradients() {
        // Worst doc scored highest => non-zero lambdas pushing it down.
        let (labels, qb) = perfect_query();
        let obj = Lambdarank::new(1.0, false, 30, &[], &labels, &qb).unwrap();
        let score = vec![0.0f64, 1.0, 2.0, 3.0]; // reversed
        let mut grad = vec![0.0f32; 4];
        let mut hess = vec![0.0f32; 4];
        obj.get_gradients(&score, &mut grad, &mut hess).unwrap();
        let any_nonzero = grad.iter().any(|&g| g.abs() > 1e-7);
        assert!(any_nonzero, "misranked query must produce non-zero lambdas");
    }

    #[test]
    fn pow2_int_matches_powers_of_two() {
        assert_eq!(pow2_int(0), 1.0);
        assert_eq!(pow2_int(1), 2.0);
        assert_eq!(pow2_int(2), 4.0);
        assert_eq!(pow2_int(3), 8.0);
        assert_eq!(pow2_int(10), 1024.0);
    }

    #[test]
    fn rank_xendcg_rejects_negative_label() {
        let labels = vec![1.0f32, -2.0];
        let qb = vec![0i32, 2];
        assert!(matches!(
            RankXendcg::new(5, &labels, &qb).unwrap_err(),
            ObjectiveError::LabelRange { .. }
        ));
    }

    /// Verbatim re-implementation of the C++ `RankXENDCG::GetGradientsForOneQuery`
    /// over the proven `lgbm_core::Random` — the objective_seed RNG-replay reference.
    fn reference_xendcg(
        seed: i32,
        labels: &[f32],
        score: &[f64],
        qb: &[i32],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = score.len();
        let mut grad = vec![0.0f32; n];
        let mut hess = vec![0.0f32; n];
        let num_q = qb.len() - 1;
        let mut rands: Vec<Random> = (0..num_q as i32).map(|q| Random::new(seed + q)).collect();
        for q in 0..num_q {
            let start = qb[q] as usize;
            let cnt = qb[q + 1] as usize - start;
            let s = &score[start..start + cnt];
            let l = &labels[start..start + cnt];
            if cnt <= 1 {
                continue;
            }
            let mut rho = vec![0.0f64; cnt];
            softmax(s, &mut rho);
            let mut params = vec![0.0f64; cnt];
            let mut inv_den = 0.0f64;
            for i in 0..cnt {
                params[i] = pow2_int(l[i] as i32) - rands[q].next_float() as f64;
                inv_den += params[i];
            }
            inv_den = 1.0 / (K_EPSILON as f64).max(inv_den);
            let mut sum_l1 = 0.0;
            for i in 0..cnt {
                let term = -params[i] * inv_den + rho[i];
                grad[start + i] = term as f32;
                params[i] = term / (1.0 - rho[i]);
                sum_l1 += params[i];
            }
            let mut sum_l2 = 0.0;
            for i in 0..cnt {
                let term = rho[i] * (sum_l1 - params[i]);
                grad[start + i] += term as f32;
                params[i] = term / (1.0 - rho[i]);
                sum_l2 += params[i];
            }
            for i in 0..cnt {
                grad[start + i] += (rho[i] * (sum_l2 - params[i])) as f32;
                hess[start + i] = (rho[i] * (1.0 - rho[i])) as f32;
            }
        }
        (grad, hess)
    }

    #[test]
    fn rank_xendcg_matches_rng_replay_reference_bit_exact() {
        // Two queries; the per-query Random(seed + q) draw order must be bit-exact
        // vs the verbatim reference (the objective_seed RNG-replay).
        let labels = vec![3.0f32, 1.0, 0.0, 2.0, 1.0, 0.0];
        let score = vec![0.5f64, -0.2, 0.1, 0.9, 0.3, -0.4];
        let qb = vec![0i32, 3, 6];
        for seed in [5i32, 7] {
            let obj = RankXendcg::new(seed, &labels, &qb).unwrap();
            let mut rands = obj.make_rands();
            let mut grad = vec![0.0f32; 6];
            let mut hess = vec![0.0f32; 6];
            obj.get_gradients(&score, &labels, &mut rands, &mut grad, &mut hess)
                .unwrap();
            let (eg, eh) = reference_xendcg(seed, &labels, &score, &qb);
            for i in 0..6 {
                assert_eq!(grad[i].to_bits(), eg[i].to_bits(), "seed={seed} grad[{i}]");
                assert_eq!(hess[i].to_bits(), eh[i].to_bits(), "seed={seed} hess[{i}]");
            }
        }
    }

    #[test]
    fn rank_xendcg_skips_singleton_query() {
        let labels = vec![2.0f32];
        let qb = vec![0i32, 1];
        let obj = RankXendcg::new(5, &labels, &qb).unwrap();
        let mut rands = obj.make_rands();
        let mut grad = vec![9.0f32];
        let mut hess = vec![9.0f32];
        obj.get_gradients(&[0.5f64], &labels, &mut rands, &mut grad, &mut hess)
            .unwrap();
        assert_eq!(grad[0], 0.0);
        assert_eq!(hess[0], 0.0);
    }
}
