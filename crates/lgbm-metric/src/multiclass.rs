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

/// `multi_error` (top-k error) metric — `MultiErrorMetric` (multiclass_metric.hpp:138-160).
///
/// `LossOnPoint`: count classes with `score[k] >= score[label]`; if that count
/// exceeds `multi_error_top_k`, the row is an error (1), else 0. Averaged by
/// num_data. `factor_to_bigger_better = -1` (a loss). Operates on the
/// post-`ConvertOutput` `rec` vector, but the comparison is order-preserving under
/// the softmax / per-class-sigmoid monotone transform, so it is computed over the
/// transformed probs exactly as C++ does (`MulticlassMetric::Eval` :74).
#[derive(Debug, Clone, PartialEq)]
pub struct MultiError {
    objective: ObjectiveKind,
    num_class: i32,
    /// `config.multi_error_top_k` (C++ default 1 = the usual multi-error).
    top_k: i32,
}

impl MultiError {
    /// Construct from the parsed [`ObjectiveKind`] + `num_class` + `multi_error_top_k`.
    pub fn new(objective: ObjectiveKind, num_class: i32, top_k: i32) -> Self {
        Self {
            objective,
            num_class,
            top_k,
        }
    }

    /// C++ `MultiErrorMetric::Name` (multiclass_metric.hpp:153-158): `multi_error`
    /// when `multi_error_top_k == 1`, else `multi_error@<k>`.
    pub fn name(&self) -> String {
        if self.top_k == 1 {
            "multi_error".to_string()
        } else {
            format!("multi_error@{}", self.top_k)
        }
    }

    /// C++ `factor_to_bigger_better`: `-1` (a loss).
    pub fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    /// Evaluate `multi_error` over the class-major f64 `scores`
    /// (`num_data * num_class`) + f32 integer-class `labels` (`num_data`).
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] when `scores.len() != num_data * num_class`.
    pub fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, MetricError> {
        let num_data = labels.len();
        let k_classes = self.num_class.max(0) as usize;
        let total = num_data * k_classes;
        if scores.len() != total {
            return Err(MetricError::LengthMismatch {
                expected: total,
                actual: scores.len(),
            });
        }
        if num_data == 0 || k_classes == 0 {
            return Ok(0.0);
        }
        let mut raw = vec![0.0f64; k_classes];
        let mut rec = vec![0.0f64; k_classes];
        let mut sum_loss = 0.0f64;
        for i in 0..num_data {
            for k in 0..k_classes {
                raw[k] = scores[num_data * k + i];
            }
            self.objective.convert(&raw, &mut rec);
            // MultiErrorMetric::LossOnPoint (multiclass_metric.hpp:142-151).
            let kk = (labels[i] as usize).min(k_classes - 1);
            let ref_k = rec[kk];
            let mut num_larger = 0i32;
            let mut is_error = 0.0f64;
            for v in &rec {
                if *v >= ref_k {
                    num_larger += 1;
                }
                if num_larger > self.top_k {
                    is_error = 1.0;
                    break;
                }
            }
            sum_loss += is_error;
        }
        Ok(sum_loss / num_data as f64)
    }
}

/// `auc_mu` multiclass AUC metric — `AucMuMetric` (multiclass_metric.hpp:183-365).
///
/// A faithful 1:1 port of the unweighted `AucMuMetric::Eval`, using the default
/// equal-weight class matrix (all-ones off-diagonal, zero diagonal —
/// `Config::GetAucMuWeights`, config.cpp:220-225) since `auc_mu_weights` is not yet
/// a parsed param. `factor_to_bigger_better = +1` (bigger is better). Operates on
/// the RAW class-major scores directly (no ConvertOutput — `Eval` ignores the
/// objective).
#[derive(Debug, Clone, PartialEq)]
pub struct AucMu {
    num_class: usize,
    /// `auc_mu_weights_matrix` — `num_class x num_class`, ones off-diagonal / 0 diagonal.
    class_weights: Vec<Vec<f64>>,
}

impl AucMu {
    /// Construct with `num_class` and the C++ default equal-weight matrix.
    pub fn new(num_class: i32) -> Self {
        let nc = num_class.max(0) as usize;
        let mut class_weights = vec![vec![1.0f64; nc]; nc];
        for (i, row) in class_weights.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        Self {
            num_class: nc,
            class_weights,
        }
    }

    /// Construct from the DERIVED `num_class x num_class` weight matrix
    /// (`Config::auc_mu_weights_matrix`, produced by `Config::GetAucMuWeights`),
    /// mirroring `AucMuMetric::Init` (`multiclass_metric.hpp:187`).
    ///
    /// A matrix whose shape does not match `num_class` falls back to the
    /// equal-weight default — the same effective behavior as a C++ `Config` that
    /// never ran `GetAucMuWeights` (an empty matrix), and never a panic.
    pub fn with_weights(num_class: i32, class_weights: &[Vec<f64>]) -> Self {
        let nc = num_class.max(0) as usize;
        if class_weights.len() != nc || class_weights.iter().any(|r| r.len() != nc) {
            return Self::new(num_class);
        }
        Self {
            num_class: nc,
            class_weights: class_weights.to_vec(),
        }
    }

    /// C++ `Metric::Name`.
    pub fn name(&self) -> &'static str {
        "auc_mu"
    }

    /// C++ `factor_to_bigger_better`: `+1` (bigger is better).
    pub fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    /// Evaluate `auc_mu` over class-major f64 `scores` (`num_data * num_class`) +
    /// f32 integer-class `labels` (`num_data`). Unweighted.
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] when `scores.len() != num_data * num_class`.
    pub fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, MetricError> {
        let num_data = labels.len();
        let nc = self.num_class;
        let total = num_data * nc;
        if scores.len() != total {
            return Err(MetricError::LengthMismatch {
                expected: total,
                actual: scores.len(),
            });
        }
        if num_data == 0 || nc < 2 {
            return Ok(0.0);
        }
        let eps = K_EPSILON as f64;

        // sort the data indices by true class (stable to mirror ParallelSort's
        // by-key order; auc_mu reads class blocks, so ties within a class are
        // order-irrelevant for the block boundaries).
        let mut sorted_data_idx: Vec<usize> = (0..num_data).collect();
        sorted_data_idx.sort_by(|&a, &b| labels[a].total_cmp(&labels[b]));

        // class sizes.
        let mut class_sizes = vec![0i64; nc];
        for &l in labels {
            let c = l as usize;
            if c < nc {
                class_sizes[c] += 1;
            }
        }

        // S[i][j] accumulation.
        let mut s = vec![vec![0.0f64; nc]; nc];
        let mut i_start = 0i64;
        for i in 0..nc {
            let j_start_base = i_start + class_sizes[i];
            let mut j_start = j_start_base;
            for j in (i + 1)..nc {
                // curr_v = class_weights[i] - class_weights[j].
                let curr_v: Vec<f64> = (0..nc)
                    .map(|k| self.class_weights[i][k] - self.class_weights[j][k])
                    .collect();
                let t1 = curr_v[i] - curr_v[j];

                // data indices belonging to class i or j.
                let mut class_i_j: Vec<usize> = Vec::new();
                for k in 0..class_sizes[i] {
                    class_i_j.push(sorted_data_idx[(i_start + k) as usize]);
                }
                for k in 0..class_sizes[j] {
                    class_i_j.push(sorted_data_idx[(j_start + k) as usize]);
                }

                // distance from separating hyperplane: t1 * (curr_v . score[a]).
                let mut dist: Vec<(usize, f64)> = class_i_j
                    .iter()
                    .map(|&a| {
                        let mut v_a = 0.0f64;
                        for (m, &cv) in curr_v.iter().enumerate() {
                            v_a += cv * scores[num_data * m + a];
                        }
                        (a, t1 * v_a)
                    })
                    .collect();

                // sort by distance; on ~tie put j-class first (label larger first).
                dist.sort_by(|a, b| {
                    if (a.1 - b.1).abs() < eps {
                        // label[a] > label[b] first  =>  descending label.
                        labels[b.0].total_cmp(&labels[a.0])
                    } else {
                        a.1.total_cmp(&b.1)
                    }
                });

                // accumulate S[i][j] (unweighted).
                let mut num_j = 0.0f64;
                let mut last_j_dist = 0.0f64;
                let mut num_current_j = 0.0f64;
                for &(a, curr_dist) in &dist {
                    if labels[a] as usize == i {
                        if (curr_dist - last_j_dist).abs() < eps {
                            s[i][j] += num_j - 0.5 * num_current_j;
                        } else {
                            s[i][j] += num_j;
                        }
                    } else {
                        num_j += 1.0;
                        if (curr_dist - last_j_dist).abs() < eps {
                            num_current_j += 1.0;
                        } else {
                            last_j_dist = curr_dist;
                            num_current_j = 1.0;
                        }
                    }
                }
                j_start += class_sizes[j];
            }
            let _ = j_start;
            i_start += class_sizes[i];
        }

        // ans.
        let mut ans = 0.0f64;
        for i in 0..nc {
            for j in (i + 1)..nc {
                ans += (s[i][j] / class_sizes[i] as f64) / class_sizes[j] as f64;
            }
        }
        ans = (2.0 * ans / nc as f64) / (nc as f64 - 1.0);
        Ok(ans)
    }
}

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

    // --- MET-03 multi_error / auc_mu (07-04) ---

    #[test]
    fn multi_error_counts_argmax_mismatch() {
        // 2 classes, 2 rows. top_k=1. row0 label=0 with class0 winning -> correct.
        // row1 label=1 with class0 winning -> error. error rate = 0.5.
        let scores = [2.0f64, 2.0, 0.0, 0.0]; // class0=[2,2], class1=[0,0]
        let labels = [0.0f32, 1.0];
        let m = MultiError::new(ObjectiveKind::Multiclass { num_class: 2 }, 2, 1);
        let got = m.eval(&scores, &labels).unwrap();
        assert!((got - 0.5).abs() < TOL, "multi_error {got} != 0.5");
        assert_eq!(m.name(), "multi_error");
        assert_eq!(m.factor_to_bigger_better(), -1.0);
    }

    #[test]
    fn multi_error_all_correct_is_zero() {
        // class0 wins for label0, class1 wins for label1.
        let scores = [2.0f64, 0.0, 0.0, 2.0];
        let labels = [0.0f32, 1.0];
        let m = MultiError::new(ObjectiveKind::Multiclass { num_class: 2 }, 2, 1);
        assert!(m.eval(&scores, &labels).unwrap() < TOL);
    }

    #[test]
    fn multi_error_length_mismatch_is_typed_error() {
        let m = MultiError::new(ObjectiveKind::Multiclass { num_class: 3 }, 3, 1);
        let err = m.eval(&[1.0, 2.0, 3.0, 4.0], &[0.0f32, 1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LengthMismatch { .. }));
    }

    #[test]
    fn auc_mu_perfect_separation_is_one() {
        // 2 classes, perfectly separable: class0 rows score high on class0,
        // class1 rows score high on class1. auc_mu = 1.
        // 4 rows: 0,0,1,1. class-major scores: class0 = [3,2,0,0], class1=[0,0,2,3].
        let scores = [3.0f64, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0];
        let labels = [0.0f32, 0.0, 1.0, 1.0];
        let m = AucMu::new(2);
        let got = m.eval(&scores, &labels).unwrap();
        assert!((got - 1.0).abs() < 1e-6, "auc_mu perfect should be 1, got {got}");
        assert_eq!(m.name(), "auc_mu");
        assert_eq!(m.factor_to_bigger_better(), 1.0);
    }

    #[test]
    fn auc_mu_default_matrix_is_ones_off_diagonal() {
        let m = AucMu::new(3);
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 0.0 } else { 1.0 };
                assert_eq!(m.class_weights[i][j], expect);
            }
        }
    }

    #[test]
    fn auc_mu_length_mismatch_is_typed_error() {
        let m = AucMu::new(3);
        let err = m.eval(&[1.0, 2.0, 3.0, 4.0], &[0.0f32, 1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LengthMismatch { .. }));
    }
}
