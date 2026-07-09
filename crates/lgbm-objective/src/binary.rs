//! `binary` objective — the sigmoid logloss training side (grad/hess +
//! `BoostFromScore` + class-need-train), ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/objective/binary_objective.hpp:105-136` — `GetGradients`
//!   (no-weights): `is_pos = label > 0`; `label_val = {-1,+1}[is_pos]`;
//!   `label_weight = label_weights_[is_pos]` (1.0 each on the unbalanced default);
//!   `response = -label_val * sigmoid_ / (1 + exp(label_val * sigmoid_ *
//!   score[i]))` (f64); `grad = (score_t)(response * label_weight)`;
//!   `hess = (score_t)(|response| * (sigmoid_ - |response|) * label_weight)`.
//! - `binary_objective.hpp:139-165` — `BoostFromScore`: `pavg = (Σ is_pos) /
//!   num_data` clamped to `[kEpsilon, 1 - kEpsilon]`; return `log(pavg / (1 -
//!   pavg)) / sigmoid_`.
//! - `binary_objective.hpp:167-169` — `ClassNeedTrain` returns `need_train_`,
//!   which is false when the corpus contains only one class (`Init`:80-84).
//! - `binary_objective.hpp:175-177` — `ConvertOutput`: `1/(1+exp(-sigmoid_*x))`
//!   (reused from [`lgbm_model::ObjectiveKind::convert`], not re-ported here).
//!
//! The spine corpora are unweighted and use the balanced default
//! (`label_weights_ = [1.0, 1.0]`, `scale_pos_weight = 1.0`, `is_unbalance =
//! false`), so `label_weight == 1.0` for both classes. `is_unbalance` /
//! `scale_pos_weight` weighting is recorded in the formula but not yet exercised.
//! Labels outside `{0,1}` are treated as `label > 0`
//! (verbatim C++ — no array is indexed by the label, so no OOB risk).

use lgbm_core::types::K_EPSILON;

use crate::apply_grad_hess;
use crate::error::ObjectiveError;

/// The `binary` (sigmoid logloss) objective, training side.
#[derive(Debug, Clone, PartialEq)]
pub struct Binary {
    /// `sigmoid_` (C++ `config.sigmoid`, default 1.0; must be `> 0`).
    pub sigmoid: f64,
}

impl Binary {
    /// Construct with a validated `sigmoid` (`> 0`, mirroring the C++ `Log::Fatal`
    /// gate `binary_objective.hpp:27-29` → typed error here, never a panic).
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] when `sigmoid <= 0` (the C++ fatal-check
    /// surfaced as a typed error).
    pub fn new(sigmoid: f64) -> Result<Self, ObjectiveError> {
        if sigmoid <= 0.0 {
            return Err(ObjectiveError::Unsupported {
                name: format!("binary sigmoid {sigmoid} must be > 0"),
            });
        }
        Ok(Self { sigmoid })
    }

    /// C++ `BinaryLogloss::GetGradients` (no-weights, balanced default).
    ///
    /// `score` is the f64 accumulated raw score; `label` is f32; `grad`/`hess` are
    /// written as `score_t = f32`. Per row (verbatim op order):
    /// - `is_pos = label[i] > 0`; `label_val = if is_pos { 1 } else { -1 }` (f64).
    /// - `response = -label_val * sigmoid_ / (1 + exp(label_val * sigmoid_ *
    ///   score[i]))` (f64).
    /// - `grad[i] = (f32)(response)`; `hess[i] = (f32)(|response| * (sigmoid_ -
    ///   |response|))` (label_weight = 1.0 on the balanced default).
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] (V5 boundary) if the four slices disagree
    /// in length — validated before any per-row write.
    pub fn get_gradients(
        &self,
        score: &[f64],
        label: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let n = score.len();
        if label.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: label.len() });
        }
        if gradients.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: gradients.len() });
        }
        if hessians.len() != n {
            return Err(ObjectiveError::LengthMismatch { expected: n, actual: hessians.len() });
        }
        let sigmoid = self.sigmoid;
        // Balanced default: label_weight = 1.0 for both classes.
        let label_weight = 1.0f64;
        // The per-row body is a pure function of (score[i], label[i]) written
        // to disjoint gradients[i]/hessians[i] — no reduction, no cross-row order — so
        // partitioning the row range is bit-exact. `apply_grad_hess` runs rayon above
        // the grain floor (measured ~5.1× win: 100M single-core `exp()` → all cores)
        // and the serial loop below it. Per-row math is verbatim from the C++ mirror.
        apply_grad_hess(score, label, gradients, hessians, |s, l| {
            let is_pos = l > 0.0;
            let label_val: f64 = if is_pos { 1.0 } else { -1.0 };
            let response = -label_val * sigmoid / (1.0 + (label_val * sigmoid * s).exp());
            let abs_response = response.abs();
            (
                (response * label_weight) as f32,
                (abs_response * (sigmoid - abs_response) * label_weight) as f32,
            )
        });
        Ok(())
    }

    /// C++ `BinaryLogloss::BoostFromScore` (unweighted): `pavg = (Σ is_pos) /
    /// num_data` clamped to `[kEpsilon, 1 - kEpsilon]`; return `log(pavg / (1 -
    /// pavg)) / sigmoid_`. With `deterministic = true` the C++ OpenMP reduction is
    /// stripped, so this is a single ordered sequential count.
    pub fn boost_from_score(&self, label: &[f32]) -> f64 {
        if label.is_empty() {
            return 0.0;
        }
        let sumw = label.len() as f64;
        let mut suml = 0.0f64;
        for &l in label {
            // `is_pos_(label) = label > 0` -> 1 / 0.
            suml += if l > 0.0 { 1.0 } else { 0.0 };
        }
        let mut pavg = suml / sumw;
        let eps = K_EPSILON as f64;
        pavg = pavg.min(1.0 - eps);
        pavg = pavg.max(eps);
        (pavg / (1.0 - pavg)).ln() / self.sigmoid
    }

    /// C++ `BinaryLogloss::ClassNeedTrain` (`need_train_`): false when the corpus
    /// contains only one class (all positive or all negative) — then a constant
    /// (1-leaf) tree is pushed instead of training. Full degenerate
    /// handling is exercised elsewhere; the spine binary corpora carry
    /// both classes.
    pub fn class_need_train(&self, label: &[f32]) -> bool {
        let mut has_pos = false;
        let mut has_neg = false;
        for &l in label {
            if l > 0.0 {
                has_pos = true;
            } else {
                has_neg = true;
            }
            if has_pos && has_neg {
                return true;
            }
        }
        false
    }

    /// Whether `|init|` exceeds the `kEpsilon` gate (mirrors
    /// [`crate::Objective::init_score_is_significant`]).
    pub fn init_score_is_significant(init: f64) -> bool {
        init.abs() > K_EPSILON as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    /// Reproduce the C++ response formula in f64, then f32-cast — the bit-exact
    /// expected value.
    fn expect_gh(label: f32, score: f64, sigmoid: f64) -> (f32, f32) {
        let label_val: f64 = if label > 0.0 { 1.0 } else { -1.0 };
        let response = -label_val * sigmoid / (1.0 + (label_val * sigmoid * score).exp());
        let abs_response = response.abs();
        (response as f32, (abs_response * (sigmoid - abs_response)) as f32)
    }

    #[test]
    fn binary_gradients_bit_exact_f32() {
        let obj = Binary::new(1.0).unwrap();
        let label = [1.0f32, 0.0, 1.0, 0.0];
        let score = [0.5f64, 0.5, -2.0, 3.0];
        let mut grad = [0.0f32; 4];
        let mut hess = [0.0f32; 4];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        for i in 0..4 {
            let (eg, eh) = expect_gh(label[i], score[i], 1.0);
            assert_eq!(grad[i], eg, "grad[{i}] bit-exact");
            assert_eq!(hess[i], eh, "hess[{i}] bit-exact");
        }
        // At score=0, pos row: response = -1/(1+1) = -0.5; hess = 0.5*(1-0.5)=0.25.
        let (g0, h0) = expect_gh(1.0, 0.0, 1.0);
        assert_eq!(g0, -0.5f32);
        assert_eq!(h0, 0.25f32);
    }

    #[test]
    fn binary_gradients_sigmoid_2() {
        let obj = Binary::new(2.0).unwrap();
        let label = [1.0f32, 0.0];
        let score = [0.3f64, -0.7];
        let mut grad = [0.0f32; 2];
        let mut hess = [0.0f32; 2];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        for i in 0..2 {
            let (eg, eh) = expect_gh(label[i], score[i], 2.0);
            assert_eq!(grad[i], eg);
            assert_eq!(hess[i], eh);
        }
    }

    #[test]
    fn binary_boost_from_score_is_logit_of_base_rate() {
        let obj = Binary::new(1.0).unwrap();
        // 3 positive of 5 -> pavg = 0.6 -> log(0.6/0.4)/1 = log(1.5).
        let label = [1.0f32, 1.0, 1.0, 0.0, 0.0];
        let init = obj.boost_from_score(&label);
        assert!((init - (0.6f64 / 0.4).ln()).abs() < ORACLE_TOL as f64, "init = {init}");
    }

    #[test]
    fn binary_boost_from_score_clamps_all_one_class() {
        let obj = Binary::new(1.0).unwrap();
        // all positive -> pavg = 1.0 clamped to 1 - kEps -> finite, large positive.
        let label = [1.0f32; 4];
        let init = obj.boost_from_score(&label);
        assert!(init.is_finite() && init > 0.0, "clamped init must be finite: {init}");
    }

    #[test]
    fn binary_class_need_train() {
        let obj = Binary::new(1.0).unwrap();
        assert!(obj.class_need_train(&[1.0, 0.0, 1.0]));
        assert!(!obj.class_need_train(&[1.0, 1.0, 1.0]));
        assert!(!obj.class_need_train(&[0.0, 0.0]));
    }

    #[test]
    fn binary_length_mismatch_is_typed_error() {
        let obj = Binary::new(1.0).unwrap();
        let err = obj
            .get_gradients(&[1.0f64, 2.0], &[1.0f32], &mut [0.0f32; 2], &mut [0.0f32; 2])
            .unwrap_err();
        assert!(matches!(err, ObjectiveError::LengthMismatch { .. }));
    }

    #[test]
    fn binary_rejects_nonpositive_sigmoid() {
        assert!(Binary::new(0.0).is_err());
        assert!(Binary::new(-1.0).is_err());
    }
}
