//! Binary classification metrics — `binary_logloss`, `binary_error`, `auc`,
//! ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/metric/binary_metric.hpp:60-97` — `BinaryMetric::Eval`: the
//!   prob-space metrics call `objective->ConvertOutput(&score[i], &prob)` FIRST
//!   (`:80-83`), then `sum_loss += LossOnPoint(label, prob)`, averaged by
//!   `sum_weights_` (`= num_data` unweighted).
//! - `binary_metric.hpp:119-130` — `BinaryLoglossMetric::LossOnPoint`:
//!   `label <= 0`: `-log(1 - prob)` if `1 - prob > kEpsilon`; `label > 0`:
//!   `-log(prob)` if `prob > kEpsilon`; else `-log(kEpsilon)`.
//! - `binary_metric.hpp:143-149` — `BinaryErrorMetric::LossOnPoint`:
//!   `prob <= 0.5`: `(label > 0)`; else `(label <= 0)` (the 0/1 error indicator).
//! - `binary_metric.hpp:194-251` — `AUCMetric::Eval`: descending-by-score sort
//!   (`Common::ParallelSort` → `std::sort`, UNSTABLE for `num_threads <= 1`) + the
//!   grouped accumulation loop. AUC is tie-group-invariant, so an unstable sort is
//!   bit-safe (Pitfall 1). `factor_to_bigger_better = +1` (vs -1 for the losses).
//!
//! The prob transform (`ConvertOutput`) is REUSED from
//! [`lgbm_model::objective::convert_binary`] — this crate does NOT re-port it
//! (Open-Q1). The logloss floor uses [`lgbm_core::types::K_EPSILON`] (1e-15f),
//! never a fresh literal.

use lgbm_core::types::K_EPSILON;
use lgbm_model::objective::convert_binary;

use crate::error::MetricError;

/// Binary-classification metric factory, mirroring the C++ string-keyed
/// `CreateMetric`. Each carries the `sigmoid` it needs to map the raw score to a
/// probability (binary_logloss/binary_error) — `auc` is sort-based and needs no
/// transform (it operates on the raw score order directly).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryMetric {
    /// `binary_logloss` — `-log(p)` / `-log(1-p)` with the `kEpsilon` floor.
    BinaryLogloss {
        /// `sigmoid_` for the `ConvertOutput` transform.
        sigmoid: f64,
    },
    /// `binary_error` — the 0/1 error indicator at the 0.5 probability threshold.
    BinaryError {
        /// `sigmoid_` for the `ConvertOutput` transform.
        sigmoid: f64,
    },
    /// `auc` — area under the ROC curve (sort-based; no prob transform needed).
    Auc,
}

impl BinaryMetric {
    /// C++ `Metric::Name`.
    pub fn name(&self) -> &'static str {
        match self {
            BinaryMetric::BinaryLogloss { .. } => "binary_logloss",
            BinaryMetric::BinaryError { .. } => "binary_error",
            BinaryMetric::Auc => "auc",
        }
    }

    /// C++ `factor_to_bigger_better`: `-1` for the losses, `+1` for AUC.
    pub fn factor_to_bigger_better(&self) -> f64 {
        match self {
            BinaryMetric::BinaryLogloss { .. } | BinaryMetric::BinaryError { .. } => -1.0,
            BinaryMetric::Auc => 1.0,
        }
    }

    /// Evaluate the metric over the f64 accumulated raw `scores` + f32 `labels`
    /// (unweighted path, `objective != nullptr`).
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] (V5 boundary) if `scores.len() !=
    /// labels.len()`.
    pub fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, MetricError> {
        let n = scores.len();
        if labels.len() != n {
            return Err(MetricError::LengthMismatch { expected: n, actual: labels.len() });
        }
        match self {
            BinaryMetric::BinaryLogloss { sigmoid } => {
                if n == 0 {
                    return Ok(0.0);
                }
                let eps = K_EPSILON as f64;
                let mut sum_loss = 0.0f64;
                for i in 0..n {
                    // ConvertOutput first (prob-space metric).
                    let prob = convert_binary(scores[i], *sigmoid);
                    sum_loss += binary_logloss_on_point(labels[i], prob, eps);
                }
                Ok(sum_loss / n as f64)
            }
            BinaryMetric::BinaryError { sigmoid } => {
                if n == 0 {
                    return Ok(0.0);
                }
                let mut sum_loss = 0.0f64;
                for i in 0..n {
                    let prob = convert_binary(scores[i], *sigmoid);
                    sum_loss += binary_error_on_point(labels[i], prob);
                }
                Ok(sum_loss / n as f64)
            }
            BinaryMetric::Auc => Ok(auc(scores, labels)),
        }
    }
}

/// C++ `BinaryLoglossMetric::LossOnPoint` (binary_metric.hpp:119-130).
#[inline]
fn binary_logloss_on_point(label: f32, prob: f64, eps: f64) -> f64 {
    if label <= 0.0 {
        if 1.0 - prob > eps {
            return -(1.0 - prob).ln();
        }
    } else if prob > eps {
        return -prob.ln();
    }
    -eps.ln()
}

/// C++ `BinaryErrorMetric::LossOnPoint` (binary_metric.hpp:143-149).
#[inline]
fn binary_error_on_point(label: f32, prob: f64) -> f64 {
    if prob <= 0.5 {
        // error if the true class is positive.
        if label > 0.0 { 1.0 } else { 0.0 }
    } else {
        // error if the true class is negative.
        if label <= 0.0 { 1.0 } else { 0.0 }
    }
}

/// C++ `AUCMetric::Eval` (binary_metric.hpp:194-251), unweighted.
///
/// Descending-by-score sort + grouped accumulation. The sort is UNSTABLE
/// (`std::sort` at `num_threads <= 1`) but the grouped accumulation makes AUC
/// tie-order-invariant (Pitfall 1), so the result is bit-identical regardless of
/// how ties are ordered.
fn auc(scores: &[f64], labels: &[f32]) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 1.0;
    }
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    // Descending by score. unstable sort is tie-safe (Pitfall 1).
    sorted_idx.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    let mut cur_pos = 0.0f64;
    let mut sum_pos = 0.0f64;
    let mut accum = 0.0f64;
    let mut cur_neg = 0.0f64;
    let mut threshold = scores[sorted_idx[0]];
    let sum_weights = n as f64;
    for &idx in &sorted_idx {
        let cur_label = labels[idx];
        let cur_score = scores[idx];
        if cur_score != threshold {
            threshold = cur_score;
            accum += cur_neg * (cur_pos * 0.5 + sum_pos);
            sum_pos += cur_pos;
            cur_neg = 0.0;
            cur_pos = 0.0;
        }
        if cur_label <= 0.0 {
            cur_neg += 1.0;
        } else {
            cur_pos += 1.0;
        }
    }
    accum += cur_neg * (cur_pos * 0.5 + sum_pos);
    sum_pos += cur_pos;
    if sum_pos > 0.0 && sum_pos != sum_weights {
        accum / (sum_pos * (sum_weights - sum_pos))
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    const TOL: f64 = ORACLE_TOL as f64;

    #[test]
    fn binary_logloss_matches_hand_computed() {
        // sigmoid=1: prob = 1/(1+exp(-score)). score=0 -> p=0.5.
        // labels [1,0], scores [0,0] -> -log(0.5) + -log(1-0.5) = 2*log(2); /2 =
        // log(2).
        let m = BinaryMetric::BinaryLogloss { sigmoid: 1.0 };
        let got = m.eval(&[0.0f64, 0.0], &[1.0f32, 0.0]).unwrap();
        assert!((got - 2.0f64.ln()).abs() < TOL, "logloss = {got}");
    }

    #[test]
    fn binary_error_matches_hand_computed() {
        // sigmoid=1. score>0 -> p>0.5 -> predict positive. score<0 -> predict neg.
        // row0: label=1, score=2 (p>0.5) -> correct (0). row1: label=0, score=2
        // (p>0.5) -> error (1). row2: label=0, score=-3 (p<0.5) -> correct (0).
        // sum=1 /3.
        let m = BinaryMetric::BinaryError { sigmoid: 1.0 };
        let got = m.eval(&[2.0f64, 2.0, -3.0], &[1.0f32, 0.0, 0.0]).unwrap();
        assert!((got - 1.0 / 3.0).abs() < TOL, "error = {got}");
    }

    #[test]
    fn auc_perfect_separation_is_one() {
        // positives all score higher than negatives -> AUC = 1.
        let scores = [3.0f64, 2.5, 1.0, 0.5];
        let labels = [1.0f32, 1.0, 0.0, 0.0];
        assert!((BinaryMetric::Auc.eval(&scores, &labels).unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn auc_is_tie_order_invariant() {
        // A corpus with TIED scores: AUC must be identical regardless of how the
        // (unstable) sort orders the ties (Pitfall 1). We assert against the same
        // corpus presented in two different input orders (the sort is the only
        // thing that can reorder ties).
        let scores_a = [1.0f64, 1.0, 1.0, 0.0, 0.0];
        let labels_a = [1.0f32, 0.0, 1.0, 0.0, 1.0];
        // Reorder rows (a permutation) — same multiset, different presentation.
        let scores_b = [0.0f64, 1.0, 1.0, 0.0, 1.0];
        let labels_b = [1.0f32, 1.0, 0.0, 0.0, 1.0];
        let a = BinaryMetric::Auc.eval(&scores_a, &labels_a).unwrap();
        let b = BinaryMetric::Auc.eval(&scores_b, &labels_b).unwrap();
        assert_eq!(a.to_bits(), b.to_bits(), "AUC must be bit-identical across tie orders: {a} vs {b}");
    }

    #[test]
    fn auc_factor_is_plus_one_losses_minus_one() {
        assert_eq!(BinaryMetric::Auc.factor_to_bigger_better(), 1.0);
        assert_eq!(BinaryMetric::BinaryLogloss { sigmoid: 1.0 }.factor_to_bigger_better(), -1.0);
        assert_eq!(BinaryMetric::BinaryError { sigmoid: 1.0 }.factor_to_bigger_better(), -1.0);
    }

    #[test]
    fn binary_metric_length_mismatch_is_typed_error() {
        let err = BinaryMetric::Auc.eval(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LengthMismatch { .. }));
    }
}
