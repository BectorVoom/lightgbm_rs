//! Regression metrics — `l2`, `rmse`, `l1` reductions over (scores, labels),
//! ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/metric/regression_metric.hpp`:
//!   - `RegressionMetric::Eval` (the `sum_loss += LossOnPoint(label, score)`
//!     reduction with `objective == nullptr` — for the regression spine the
//!     metric is fed the raw score directly, identity ConvertOutput).
//!   - `RegressionMetric::AverageLoss` (`sum_loss / sum_weights`; `sum_weights =
//!     num_data` unweighted).
//!   - `L2Metric::LossOnPoint` (`(score - label)^2`).
//!   - `RMSEMetric::LossOnPoint` (`(score - label)^2`) + `RMSEMetric::AverageLoss`
//!     (`sqrt(sum_loss / sum_weights)`).
//!   - `L1Metric::LossOnPoint` (`|score - label|`).
//!   - `RegressionMetric::factor_to_bigger_better` (`-1.0`).
//!
//! The `l1` metric is included now even though the spine metric list is
//! `l2`/`rmse`: its formula is trivial (`|score - label|`) and it pairs with the
//! regression_l1 OBJECTIVE in 06-03. Adding the metric here is cheaper than
//! re-touching the file in 06-03 and keeps the regression-metric family together.

use crate::error::MetricError;

/// The evaluation-metric factory enum, mirroring the C++ string-keyed
/// `Metric::CreateMetric`. 06-02 ships the three regression metrics; binary /
/// multiclass / AUC land in 06-04+.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `l2` — mean squared error: `Σ(score - label)^2 / num_data`.
    L2,
    /// `rmse` — root mean squared error: `sqrt(Σ(score - label)^2 / num_data)`.
    Rmse,
    /// `l1` — mean absolute error: `Σ|score - label| / num_data`.
    L1,
}

impl Metric {
    /// Parse a metric name into a [`Metric`]. Recognizes the C++ aliases for the
    /// three regression metrics.
    ///
    /// # Errors
    /// [`MetricError::Unsupported`] for an out-of-scope / unrecognized metric name.
    pub fn parse(name: &str) -> Result<Metric, MetricError> {
        match name {
            "l2" | "regression_l2" | "mean_squared_error" | "mse" | "regression" => Ok(Metric::L2),
            "rmse" | "root_mean_squared_error" | "l2_root" => Ok(Metric::Rmse),
            "l1" | "mean_absolute_error" | "mae" | "regression_l1" => Ok(Metric::L1),
            other => Err(MetricError::Unsupported {
                name: other.to_string(),
            }),
        }
    }

    /// C++ `Metric::Name` — the canonical metric name string (the name C++ uses in
    /// `record_evaluation` / model text).
    pub fn name(&self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Rmse => "rmse",
            Metric::L1 => "l1",
        }
    }

    /// C++ `factor_to_bigger_better` — `-1` for these losses (lower is better).
    pub fn factor_to_bigger_better(&self) -> f64 {
        match self {
            Metric::L2 | Metric::Rmse | Metric::L1 => -1.0,
        }
    }

    /// C++ `RegressionMetric::Eval` (unweighted, `objective == nullptr` path): the
    /// ordered f64 reduction of `LossOnPoint(label, score)` then `AverageLoss`.
    ///
    /// `scores` is the f64 accumulated raw score; `labels` is the f32 label. For
    /// the regression spine the metric reads the raw score directly (identity
    /// ConvertOutput), exactly as C++ does when the objective is regression.
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] (V5 boundary) if `scores.len() !=
    /// labels.len()` — validated before any reduction.
    pub fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, MetricError> {
        let n = scores.len();
        if labels.len() != n {
            return Err(MetricError::LengthMismatch {
                expected: n,
                actual: labels.len(),
            });
        }
        if n == 0 {
            return Ok(0.0);
        }
        // Ordered sequential f64 fold (deterministic anchor): sum_loss += LossOnPoint.
        let mut sum_loss = 0.0f64;
        for i in 0..n {
            let diff = scores[i] - labels[i] as f64;
            sum_loss += match self {
                // L2 / RMSE LossOnPoint: (score - label)^2.
                Metric::L2 | Metric::Rmse => diff * diff,
                // L1 LossOnPoint: |score - label|.
                Metric::L1 => diff.abs(),
            };
        }
        let sum_weights = n as f64;
        let loss = match self {
            // L2 / L1 AverageLoss: sum_loss / sum_weights.
            Metric::L2 | Metric::L1 => sum_loss / sum_weights,
            // RMSE AverageLoss: sqrt(sum_loss / sum_weights).
            Metric::Rmse => (sum_loss / sum_weights).sqrt(),
        };
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    const TOL: f64 = ORACLE_TOL as f64;

    #[test]
    fn parse_recognizes_regression_metrics() {
        assert_eq!(Metric::parse("l2").unwrap(), Metric::L2);
        assert_eq!(Metric::parse("rmse").unwrap(), Metric::Rmse);
        assert_eq!(Metric::parse("l1").unwrap(), Metric::L1);
        assert_eq!(Metric::parse("mae").unwrap(), Metric::L1);
        assert!(Metric::parse("auc").is_err());
    }

    #[test]
    fn eval_l2_matches_hand_computed() {
        // scores - labels = [1, -2, 0.5]; squares = [1, 4, 0.25]; sum = 5.25;
        // /3 = 1.75.
        let scores = [2.0f64, 1.0, 3.5];
        let labels = [1.0f32, 3.0, 3.0];
        let l2 = Metric::L2.eval(&scores, &labels).unwrap();
        assert!((l2 - 1.75).abs() < TOL, "l2 = {l2}");
    }

    #[test]
    fn eval_rmse_is_sqrt_of_l2() {
        let scores = [2.0f64, 1.0, 3.5];
        let labels = [1.0f32, 3.0, 3.0];
        let l2 = Metric::L2.eval(&scores, &labels).unwrap();
        let rmse = Metric::Rmse.eval(&scores, &labels).unwrap();
        assert!((rmse - l2.sqrt()).abs() < TOL, "rmse {rmse} vs sqrt(l2) {}", l2.sqrt());
        assert!((rmse - 1.75f64.sqrt()).abs() < TOL);
    }

    #[test]
    fn eval_l1_matches_hand_computed() {
        // |scores - labels| = [1, 2, 0.5]; sum = 3.5; /3 = 1.1666...
        let scores = [2.0f64, 1.0, 3.5];
        let labels = [1.0f32, 3.0, 3.0];
        let l1 = Metric::L1.eval(&scores, &labels).unwrap();
        assert!((l1 - (3.5 / 3.0)).abs() < TOL, "l1 = {l1}");
    }

    #[test]
    fn eval_length_mismatch_is_typed_error() {
        let err = Metric::L2.eval(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LengthMismatch { .. }));
    }

    #[test]
    fn factor_to_bigger_better_is_minus_one() {
        for m in [Metric::L2, Metric::Rmse, Metric::L1] {
            assert_eq!(m.factor_to_bigger_better(), -1.0);
        }
    }
}
