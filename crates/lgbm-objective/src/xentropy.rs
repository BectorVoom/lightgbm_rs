//! `cross_entropy` / `cross_entropy_lambda` objectives (OBJ-05) — the
//! cross-entropy point-loss training side (grad/hess + `BoostFromScore`), ported
//! 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/objective/xentropy_objective.hpp` `CrossEntropy`:
//!   - `Init` (`:53-72`): `CheckElementsIntervalClosed<label_t>(label_, 0, 1)` —
//!     every label in `[0, 1]` (surfaced here as a typed [`ObjectiveError::LabelRange`]).
//!   - `GetGradients` (`:74-126`, no-weights branch): the numerically-stable
//!     `log1pexp`-cutoff form. For `score > -37`: `exp_tmp = exp(-score)`,
//!     `grad = ((1-label) - label*exp_tmp) / (1+exp_tmp)`,
//!     `hess = exp_tmp / (1+exp_tmp)²`. Else: `exp_tmp = exp(score)`,
//!     `grad = exp_tmp - label`, `hess = exp_tmp`.
//!   - `BoostFromScore` (`:142-166`): `pavg = (Σ label)/num_data` clamped to
//!     `[kEpsilon, 1-kEpsilon]`; return `log(pavg / (1-pavg))`.
//!   - `ConvertOutput` (`:131`): `1/(1+exp(-input))` (reused from
//!     [`lgbm_model::ObjectiveKind`], NOT re-ported here — Open-Q1).
//! - `xentropy_objective.hpp` `CrossEntropyLambda`:
//!   - `Init` (`:185-208`): same `[0, 1]` label check.
//!   - `GetGradients` (`:210-237`, no-weights branch): `z = 1/(1+exp(-score))`,
//!     `grad = z - label`, `hess = z*(1-z)` (exactly equivalent to CrossEntropy
//!     with unit weights — see the C++ comment `:211`).
//!   - `BoostFromScore` (`:261-285`): `havg = (Σ label)/num_data`; return
//!     `log(expm1(havg))`.
//!   - `ConvertOutput` (`:251`): `log1p(exp(input))` (the lambda, NOT a probability).
//!
//! The spine corpora are unweighted, so only the no-weights branch is ported here
//! (the weighted branch is recorded in the citation but exercised in a later wave).

use lgbm_core::types::K_EPSILON;

use crate::apply_grad_hess;
use crate::error::ObjectiveError;

/// The cross-entropy objective variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XentropyKind {
    /// `cross_entropy` (`xentropy`): probability parameterization, sigmoid
    /// `ConvertOutput`, `BoostFromScore = log(pavg/(1-pavg))`.
    CrossEntropy,
    /// `cross_entropy_lambda` (`xentlambda`): the alternative parameterization,
    /// `log1p(exp)` `ConvertOutput`, `BoostFromScore = log(expm1(havg))`.
    CrossEntropyLambda,
}

/// The `cross_entropy` / `cross_entropy_lambda` (OBJ-05) objective, training side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xentropy {
    /// Which cross-entropy parameterization (grad/hess + BoostFromScore shape).
    kind: XentropyKind,
}

impl Xentropy {
    /// Construct the cross-entropy objective for the given parameterization.
    pub fn new(kind: XentropyKind) -> Self {
        Self { kind }
    }

    /// Parse the objective name into a cross-entropy variant.
    ///
    /// C++ aliases (`objective_function.cpp` factory): `cross_entropy` / `xentropy`
    /// → [`XentropyKind::CrossEntropy`]; `cross_entropy_lambda` / `xentlambda` →
    /// [`XentropyKind::CrossEntropyLambda`].
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] for any non-xentropy name.
    pub fn parse(spec: &str) -> Result<Xentropy, ObjectiveError> {
        let name = spec.split_whitespace().next().unwrap_or("");
        match name {
            "cross_entropy" | "xentropy" => Ok(Xentropy::new(XentropyKind::CrossEntropy)),
            "cross_entropy_lambda" | "xentlambda" => {
                Ok(Xentropy::new(XentropyKind::CrossEntropyLambda))
            }
            other => Err(ObjectiveError::Unsupported {
                name: other.to_string(),
            }),
        }
    }

    /// The cross-entropy parameterization.
    pub fn kind(&self) -> XentropyKind {
        self.kind
    }

    /// C++ `CrossEntropy::Init` / `CrossEntropyLambda::Init` label guard
    /// (`xentropy_objective.hpp:57` / `:194`): every label in the closed interval
    /// `[0, 1]` (`Common::CheckElementsIntervalClosed<label_t>`). Surfaced as a
    /// typed error (Security V5 / T-07-03-01) — never a panic.
    ///
    /// # Errors
    /// [`ObjectiveError::LabelRange`] for the first label outside `[0, 1]`.
    pub fn check_labels(&self, labels: &[f32]) -> Result<(), ObjectiveError> {
        let name = self.name();
        for &l in labels {
            if !(0.0..=1.0).contains(&l) {
                return Err(ObjectiveError::LabelRange {
                    label: l as f64,
                    objective: name.to_string(),
                    reason: "label must be in the closed interval [0, 1]".to_string(),
                });
            }
        }
        Ok(())
    }

    /// The canonical C++ objective name (`GetName()`).
    pub fn name(&self) -> &'static str {
        match self.kind {
            XentropyKind::CrossEntropy => "cross_entropy",
            XentropyKind::CrossEntropyLambda => "cross_entropy_lambda",
        }
    }

    /// C++ `GetGradients` (no-weights branch) for the active parameterization.
    ///
    /// `score` is the f64 accumulated raw score; `label` is f32; `grad`/`hess` are
    /// written as `score_t = f32`.
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
        // spike-068: both arms' per-row bodies are unchanged; the serial `for i in
        // 0..n` loop is replaced by `apply_grad_hess` (serial below the grain floor,
        // rayon at/above it). Each output element is a pure function of its own row's
        // (score, label) written to a disjoint slot, so this is bit-exact.
        match self.kind {
            XentropyKind::CrossEntropy => {
                // CrossEntropy::GetGradients (no-weights, xentropy_objective.hpp:98-110):
                // the log1pexp-cutoff stable form. The C++ literals `1.0f`/`1` are f32
                // inside the f64 expression — reproduce the exact `(1.0f - label_[i])`
                // op order (label promoted to f64, `1.0f` is exactly 1.0 so the f32
                // literal is immaterial; preserved as `1.0` for clarity).
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let lab = l as f64;
                    if s > -37.0 {
                        let exp_tmp = (-s).exp();
                        (
                            (((1.0 - lab) - lab * exp_tmp) / (1.0 + exp_tmp)) as f32,
                            (exp_tmp / ((1.0 + exp_tmp) * (1.0 + exp_tmp))) as f32,
                        )
                    } else {
                        let exp_tmp = s.exp();
                        ((exp_tmp - lab) as f32, exp_tmp as f32)
                    }
                });
            }
            XentropyKind::CrossEntropyLambda => {
                // CrossEntropyLambda::GetGradients (no-weights,
                // xentropy_objective.hpp:215-219):
                //   z = 1 / (1 + exp(-score[i]))        (f64)
                //   grad = (score_t)(z - label_[i])
                //   hess = (score_t)(z * (1 - z))
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let lab = l as f64;
                    let z = 1.0 / (1.0 + (-s).exp());
                    ((z - lab) as f32, (z * (1.0 - z)) as f32)
                });
            }
        }
        Ok(())
    }

    /// C++ `BoostFromScore` (no-weights) for the active parameterization.
    ///
    /// - cross_entropy (`:142-166`): `pavg = (Σ label)/num_data` clamped to
    ///   `[kEpsilon, 1-kEpsilon]`; return `log(pavg / (1-pavg))`.
    /// - cross_entropy_lambda (`:261-285`): `havg = (Σ label)/num_data`; return
    ///   `log(expm1(havg))`.
    ///
    /// With `deterministic = true` the C++ OpenMP reduction is stripped, so this is
    /// a single ordered sequential f64 fold over the labels in row order.
    pub fn boost_from_score(&self, label: &[f32]) -> f64 {
        if label.is_empty() {
            return 0.0;
        }
        let sumw = label.len() as f64;
        let mut suml = 0.0f64;
        for &l in label {
            suml += l as f64;
        }
        let avg = suml / sumw;
        match self.kind {
            XentropyKind::CrossEntropy => {
                let eps = K_EPSILON as f64;
                let mut pavg = avg;
                pavg = pavg.min(1.0 - eps);
                pavg = pavg.max(eps);
                (pavg / (1.0 - pavg)).ln()
            }
            // log(expm1(havg)) — std::expm1 then std::log. expm1(0) == 0 → log(0) ==
            // -inf, matching the C++ (an all-zero label corpus is a degenerate case
            // the caller's check_labels does not forbid for lambda).
            XentropyKind::CrossEntropyLambda => avg.exp_m1().ln(),
        }
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

    #[test]
    fn parse_xentropy_aliases() {
        assert_eq!(
            Xentropy::parse("cross_entropy").unwrap().kind(),
            XentropyKind::CrossEntropy
        );
        assert_eq!(
            Xentropy::parse("xentropy").unwrap().kind(),
            XentropyKind::CrossEntropy
        );
        assert_eq!(
            Xentropy::parse("cross_entropy_lambda").unwrap().kind(),
            XentropyKind::CrossEntropyLambda
        );
        assert_eq!(
            Xentropy::parse("xentlambda").unwrap().kind(),
            XentropyKind::CrossEntropyLambda
        );
        assert!(Xentropy::parse("regression").is_err());
    }

    #[test]
    fn check_labels_rejects_out_of_unit_interval() {
        let xe = Xentropy::new(XentropyKind::CrossEntropy);
        assert!(xe.check_labels(&[0.0, 0.5, 1.0]).is_ok());
        let err = xe.check_labels(&[0.0, 1.5]).unwrap_err();
        assert!(matches!(err, ObjectiveError::LabelRange { .. }));
        let err2 = xe.check_labels(&[-0.1, 0.5]).unwrap_err();
        assert!(matches!(err2, ObjectiveError::LabelRange { .. }));
        // The lambda variant guards the same interval.
        let xl = Xentropy::new(XentropyKind::CrossEntropyLambda);
        assert!(xl.check_labels(&[2.0]).is_err());
    }

    /// Reproduce the C++ cross_entropy stable-form grad/hess in f64, then f32-cast.
    fn expect_xe_gh(label: f32, score: f64) -> (f32, f32) {
        let lab = label as f64;
        if score > -37.0 {
            let exp_tmp = (-score).exp();
            (
                (((1.0 - lab) - lab * exp_tmp) / (1.0 + exp_tmp)) as f32,
                (exp_tmp / ((1.0 + exp_tmp) * (1.0 + exp_tmp))) as f32,
            )
        } else {
            let exp_tmp = score.exp();
            ((exp_tmp - lab) as f32, exp_tmp as f32)
        }
    }

    #[test]
    fn cross_entropy_gradients_bit_exact_f32() {
        let xe = Xentropy::new(XentropyKind::CrossEntropy);
        let label = [1.0f32, 0.0, 0.5, 1.0];
        let score = [0.5f64, 0.5, -2.0, -40.0]; // last hits the score <= -37 branch
        let mut grad = [0.0f32; 4];
        let mut hess = [0.0f32; 4];
        xe.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        for i in 0..4 {
            let (eg, eh) = expect_xe_gh(label[i], score[i]);
            assert_eq!(grad[i], eg, "grad[{i}] bit-exact");
            assert_eq!(hess[i], eh, "hess[{i}] bit-exact");
        }
        // At score=0, label=1: exp_tmp=1; grad=((0)-1)/2=-0.5; hess=1/4=0.25.
        let (g0, h0) = expect_xe_gh(1.0, 0.0);
        assert_eq!(g0, -0.5f32);
        assert_eq!(h0, 0.25f32);
    }

    #[test]
    fn cross_entropy_lambda_gradients_match_sigmoid_form() {
        let xl = Xentropy::new(XentropyKind::CrossEntropyLambda);
        let label = [1.0f32, 0.0, 0.5];
        let score = [0.5f64, -1.0, 2.0];
        let mut grad = [0.0f32; 3];
        let mut hess = [0.0f32; 3];
        xl.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        for i in 0..3 {
            let z = 1.0 / (1.0 + (-score[i]).exp());
            assert_eq!(grad[i], (z - label[i] as f64) as f32);
            assert_eq!(hess[i], (z * (1.0 - z)) as f32);
        }
    }

    #[test]
    fn cross_entropy_boost_from_score_is_logit_of_label_mean() {
        let xe = Xentropy::new(XentropyKind::CrossEntropy);
        // labels in [0,1]; mean = 0.6 -> log(0.6/0.4).
        let label = [1.0f32, 1.0, 1.0, 0.0, 0.0];
        let init = xe.boost_from_score(&label);
        assert!((init - (0.6f64 / 0.4).ln()).abs() < ORACLE_TOL as f64, "init = {init}");
    }

    #[test]
    fn cross_entropy_lambda_boost_from_score_is_log_expm1() {
        let xl = Xentropy::new(XentropyKind::CrossEntropyLambda);
        let label = [1.0f32, 0.0, 0.5, 0.5]; // mean = 0.5
        let init = xl.boost_from_score(&label);
        assert!((init - 0.5f64.exp_m1().ln()).abs() < ORACLE_TOL as f64, "init = {init}");
    }

    #[test]
    fn xentropy_length_mismatch_is_typed_error() {
        let xe = Xentropy::new(XentropyKind::CrossEntropy);
        let err = xe
            .get_gradients(&[1.0f64, 2.0], &[0.5f32], &mut [0.0f32; 2], &mut [0.0f32; 2])
            .unwrap_err();
        assert!(matches!(err, ObjectiveError::LengthMismatch { .. }));
    }
}
