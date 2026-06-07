//! Multiclass metric — `multi_logloss`, ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/metric/multiclass_metric.hpp:60-120` — `MulticlassMetric::Eval`:
//!   for each row, gather the class-major raw scores `raw_score[k] =
//!   score[num_data * k + i]` (`:68-72`), call `objective->ConvertOutput(raw_score,
//!   rec)` (`:74` — softmax for `multiclass`, per-class sigmoid for
//!   `multiclassova`), then `sum_loss += LossOnPoint(label, rec)`, averaged by
//!   `sum_weights_` (`= num_data` unweighted).
//! - `multiclass_metric.hpp:167-174` — `MultiSoftmaxLoglossMetric::LossOnPoint`:
//!   `k = (size_t)label`; `rec[k] > kEpsilon ? -log(rec[k]) : -log(kEpsilon)`.
//!
//! The prob transform (`ConvertOutput`) is REUSED from
//! [`lgbm_model::ObjectiveKind::convert`] — this crate does NOT re-port the softmax
//! / per-class sigmoid (Open-Q1). The logloss floor uses
//! [`lgbm_core::types::K_EPSILON`] (1e-15f), never a fresh literal.
//!
//! Note: `multi_logloss` (the softmax metric) and the `multiclassova` logloss share
//! `MulticlassMetric` in C++ — the only difference is which `ConvertOutput` the
//! objective supplies. We carry the parsed [`lgbm_model::ObjectiveKind`] so the SAME
//! metric handles both: `multiclass` → softmax probs, `multiclassova` → per-class
//! sigmoid probs (LightGBM names the metric `multi_logloss` in both cases).

use lgbm_core::types::K_EPSILON;
use lgbm_model::ObjectiveKind;

use crate::error::MetricError;

/// The `multi_logloss` metric over the class-major raw score buffer.
///
/// Carries the predict-side [`ObjectiveKind`] (softmax for `multiclass`, per-class
/// sigmoid for `multiclassova`) used to map the raw per-class scores to a
/// probability vector before `LossOnPoint`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiLogloss {
    /// The objective transform (`multiclass` softmax / `multiclassova` sigmoid).
    objective: ObjectiveKind,
    /// `num_class_` (the number of class-major score blocks / output width).
    num_class: i32,
}

impl MultiLogloss {
    /// Construct from the parsed [`ObjectiveKind`] (must be a multiclass kind) +
    /// `num_class`. Mirrors the C++ `MulticlassMetric` which reads `num_class` from
    /// config and uses `objective->ConvertOutput`.
    pub fn new(objective: ObjectiveKind, num_class: i32) -> Self {
        Self {
            objective,
            num_class,
        }
    }

    /// C++ `Metric::Name` — always `multi_logloss` (the C++
    /// `MultiSoftmaxLoglossMetric::Name`; the ova case reuses the same metric).
    pub fn name(&self) -> &'static str {
        "multi_logloss"
    }

    /// C++ `factor_to_bigger_better`: `-1` (a loss).
    pub fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    /// Evaluate `multi_logloss` over the class-major f64 `scores` (length
    /// `num_data * num_class`) + the f32 integer-class `labels` (length `num_data`).
    ///
    /// Per row `i`: gather `raw[k] = scores[num_data * k + i]`, `ConvertOutput` →
    /// `rec`, then `LossOnPoint = rec[label] > kEpsilon ? -log(rec[label]) :
    /// -log(kEpsilon)`. Averaged by `num_data`.
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] (V5 boundary) when `scores.len() !=
    /// num_data * num_class` (derived from `labels.len()`).
    pub fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, MetricError> {
        let num_data = labels.len();
        let k_classes = self.num_class.max(0) as usize;
        let total = num_data * k_classes;
        if scores.len() != total {
            return Err(MetricError::LengthMismatch { expected: total, actual: scores.len() });
        }
        if num_data == 0 || k_classes == 0 {
            return Ok(0.0);
        }
        let eps = K_EPSILON as f64;
        let mut raw = vec![0.0f64; k_classes];
        let mut rec = vec![0.0f64; k_classes];
        let mut sum_loss = 0.0f64;
        for i in 0..num_data {
            // Gather the class-major raw scores for this row.
            for k in 0..k_classes {
                raw[k] = scores[num_data * k + i];
            }
            // ConvertOutput first (softmax / per-class sigmoid).
            self.objective.convert(&raw, &mut rec);
            // LossOnPoint: -log(rec[label]) with the kEpsilon floor.
            let kk = labels[i] as usize;
            // Defensive: a label out of range would be a caller error; clamp to the
            // floor rather than index OOB (Security V5). The softmax objective's
            // Init already range-checks training labels.
            let p = rec.get(kk).copied().unwrap_or(0.0);
            sum_loss += if p > eps { -p.ln() } else { -eps.ln() };
        }
        Ok(sum_loss / num_data as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    const TOL: f64 = ORACLE_TOL as f64;

    #[test]
    fn multi_logloss_matches_hand_computed_softmax() {
        // 2 classes, 2 rows, class-major scores. row0 label=0, row1 label=1.
        // scores: class0 = [2.0, 0.0], class1 = [0.0, 2.0].
        let scores = [2.0f64, 0.0, 0.0, 2.0];
        let labels = [0.0f32, 1.0];
        let m = MultiLogloss::new(ObjectiveKind::Multiclass { num_class: 2 }, 2);
        let got = m.eval(&scores, &labels).unwrap();
        // Hand: row0 raw=[2,0] -> softmax p0 = e^0/(e^0+e^-2) (after max-sub) =
        // 1/(1+e^-2); loss = -log(p0). row1 symmetric. average.
        let p_row0 = {
            // softmax of [2,0]: exp(2-2)=1, exp(0-2)=e^-2; p0 = 1/(1+e^-2).
            1.0 / (1.0 + (-2.0f64).exp())
        };
        let loss0 = -p_row0.ln();
        let loss1 = loss0; // symmetric
        let expect = (loss0 + loss1) / 2.0;
        assert!((got - expect).abs() < TOL, "multi_logloss {got} != {expect}");
    }

    #[test]
    fn multi_logloss_ova_uses_per_class_sigmoid() {
        // multiclassova: ConvertOutput is per-class sigmoid (NOT softmax). 2 classes.
        let scores = [1.0f64, -1.0, -1.0, 1.0];
        let labels = [0.0f32, 1.0];
        let m = MultiLogloss::new(
            ObjectiveKind::MulticlassOva { num_class: 2, sigmoid: 1.0 },
            2,
        );
        let got = m.eval(&scores, &labels).unwrap();
        // Hand: per-class sigmoid p = 1/(1+e^-x). row0 label=0: rec[0]=sig(1.0).
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        let loss0 = -(sig(1.0)).ln();
        let loss1 = -(sig(1.0)).ln(); // row1 label=1 -> rec[1]=sig(1.0)
        let expect = (loss0 + loss1) / 2.0;
        assert!((got - expect).abs() < TOL, "ova multi_logloss {got} != {expect}");
    }

    #[test]
    fn multi_logloss_floors_tiny_prob() {
        // A wildly wrong prediction (true class score very negative) floors at
        // -log(kEpsilon) rather than producing -inf.
        let scores = [-1000.0f64, 1000.0, 1000.0, -1000.0];
        let labels = [0.0f32, 1.0];
        let m = MultiLogloss::new(ObjectiveKind::Multiclass { num_class: 2 }, 2);
        let got = m.eval(&scores, &labels).unwrap();
        assert!(got.is_finite(), "must floor, not -inf: {got}");
        assert!((got - (-(K_EPSILON as f64).ln())).abs() < 1e-6);
    }

    #[test]
    fn multi_logloss_length_mismatch_is_typed_error() {
        let m = MultiLogloss::new(ObjectiveKind::Multiclass { num_class: 3 }, 3);
        // 2 labels => expect 6 scores; supply 4.
        let err = m.eval(&[1.0, 2.0, 3.0, 4.0], &[0.0f32, 1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LengthMismatch { .. }));
    }

    #[test]
    fn multi_logloss_name_and_factor() {
        let m = MultiLogloss::new(ObjectiveKind::Multiclass { num_class: 3 }, 3);
        assert_eq!(m.name(), "multi_logloss");
        assert_eq!(m.factor_to_bigger_better(), -1.0);
    }
}
