//! Regression objectives — the training-side gradient/hessian + `BoostFromScore`
//! math, ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source, not
//! training memory):
//! - `LightGBM/src/objective/regression_objective.hpp` `RegressionL2loss`:
//!   - `GetGradients` (the no-weights `grad = (score_t)(score - label)`,
//!     `hess = 1.0f`; the weighted inner-f32-cast
//!     `grad = (score_t)((score_t)(score - label) * w)`, `hess = (score_t)w`).
//!   - `BoostFromScore` (`suml = Σ label`, `sumw = num_data`, return
//!     `suml / sumw` — the label mean; with `deterministic = true` the
//!     `if(!deterministic_)` strips the OpenMP reduction so the sum is a single
//!     ordered sequential fold).
//!   - `Init`/`ConvertOutput` `sqrt` inversion (`label = Sign(label) *
//!     sqrt(|label|)`, `ConvertOutput = Sign(x) * x * x`).
//!   - `IsConstantHessian` = true (no weights).
//!
//! The **predict-side** transform (`ConvertOutput`) is NOT re-ported here: it
//! already lives in [`lgbm_model::ObjectiveKind`] (Open-Q1 recommendation — keep
//! ConvertOutput in lgbm-model; this crate owns the training side only). The
//! `sqrt` variant's training-side label transform IS owned here (it mutates the
//! training labels, which is a training concern).

use lgbm_core::types::K_EPSILON;

use crate::error::ObjectiveError;

/// C++ `Common::Sign` (`common.h:873`): `(x > 0) - (x < 0)` → `-1 | 0 | 1`.
#[inline]
fn sign(x: f64) -> f64 {
    ((x > 0.0) as i32 - (x < 0.0) as i32) as f64
}

/// The training-side objective factory enum, mirroring the C++ string-keyed
/// `ObjectiveFunction::CreateObjectiveFunction` (one variant per supported
/// objective). 06-02 ships only the regression L2/sqrt spine variant; the
/// remaining objectives (regression_l1 / binary / multiclass / custom) land in
/// 06-03+.
#[derive(Debug, Clone, PartialEq)]
pub enum Objective {
    /// `regression` (L2). `sqrt = true` is the `regression sqrt` variant whose
    /// training labels are pre-transformed `Sign(label) * sqrt(|label|)` and whose
    /// `ConvertOutput` inverts it (`Sign(x) * x * x`).
    Regression {
        /// The `sqrt` token was present on the objective spec.
        sqrt: bool,
    },
}

impl Objective {
    /// Parse the objective name (mirroring the C++ string-ctor first-token +
    /// `sqrt` flag). 06-02 recognizes only the regression L2 family
    /// (`regression`, `regression_l2`, `mse`, `l2`, `mean_squared_error`) plus the
    /// `regression sqrt` variant. Any other name is rejected (Phase-7 / later-wave
    /// scope) — never a silent default.
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] for an out-of-scope / unrecognized
    /// objective name.
    pub fn parse(spec: &str) -> Result<Objective, ObjectiveError> {
        let mut tokens = spec.split_whitespace();
        let name = tokens.next().unwrap_or("");
        let mut sqrt = false;
        for tok in tokens {
            if tok == "sqrt" {
                sqrt = true;
            }
        }
        match name {
            "regression" | "regression_l2" | "mean_squared_error" | "mse" | "l2" => {
                Ok(Objective::Regression { sqrt })
            }
            other => Err(ObjectiveError::Unsupported {
                name: other.to_string(),
            }),
        }
    }

    /// Build the objective from a resolved [`lgbm_core::Config`] (D-02 — the
    /// builder never forks the objective name; it routes the config's
    /// `objective` string through [`Self::parse`]).
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] when `config.objective` is out of 06-02
    /// scope.
    pub fn from_config(config: &lgbm_core::Config) -> Result<Objective, ObjectiveError> {
        Objective::parse(&config.objective)
    }

    /// C++ `IsConstantHessian` (no weights → true). The spine path is unweighted,
    /// so the hessian is the constant `1.0`.
    pub fn is_constant_hessian(&self) -> bool {
        match self {
            Objective::Regression { .. } => true,
        }
    }

    /// C++ `IsRenewTreeOutput` — false for L2 (the learner's Newton output is the
    /// final leaf value; no median-residual renewal). `regression_l1` overrides
    /// this in 06-03.
    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            Objective::Regression { .. } => false,
        }
    }

    /// The training-side label transform applied once at `Init`
    /// (`regression_objective.hpp` `RegressionL2loss::Init`): for the `sqrt`
    /// variant each label becomes `Sign(label) * sqrt(|label|)`. Returns the
    /// (possibly transformed) labels the objective should train against; for the
    /// non-sqrt path the labels pass through unchanged.
    pub fn transform_labels(&self, labels: &[f32]) -> Vec<f32> {
        match self {
            Objective::Regression { sqrt } => {
                if *sqrt {
                    labels
                        .iter()
                        .map(|&l| {
                            (sign(l as f64) * (l as f64).abs().sqrt()) as f32
                        })
                        .collect()
                } else {
                    labels.to_vec()
                }
            }
        }
    }

    /// C++ `RegressionL2loss::GetGradients` (no-weights path).
    ///
    /// `score` is the f64 accumulated raw score; `label` is the (already
    /// `Init`-transformed) f32 label; `gradients`/`hessians` are written as
    /// `score_t = f32`. Per row:
    /// `grad[i] = (f32)(score[i] - label[i] as f64)`, `hess[i] = 1.0f32`.
    ///
    /// The subtraction is done in f64 (`score[i] - label[i] as f64`) and cast to
    /// f32 exactly once — mirroring the C++ `static_cast<score_t>(score[i] -
    /// label_[i])` (the C++ `label_` is f32, promoted to the f64 `score` before
    /// the subtract, then narrowed to `score_t`).
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] (V5 boundary) if the four slices do not
    /// all have the same length — validated before any per-row write.
    pub fn get_gradients(
        &self,
        score: &[f64],
        label: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let n = score.len();
        if label.len() != n {
            return Err(ObjectiveError::LengthMismatch {
                expected: n,
                actual: label.len(),
            });
        }
        if gradients.len() != n {
            return Err(ObjectiveError::LengthMismatch {
                expected: n,
                actual: gradients.len(),
            });
        }
        if hessians.len() != n {
            return Err(ObjectiveError::LengthMismatch {
                expected: n,
                actual: hessians.len(),
            });
        }
        match self {
            Objective::Regression { .. } => {
                for i in 0..n {
                    gradients[i] = (score[i] - label[i] as f64) as f32;
                    hessians[i] = 1.0f32;
                }
            }
        }
        Ok(())
    }

    /// C++ `RegressionL2loss::BoostFromScore` (unweighted): the label mean
    /// `suml / sumw` where `suml = Σ label` (f64) and `sumw = num_data`.
    ///
    /// With `deterministic = true` (the oracle config) C++ strips the OpenMP
    /// reduction (`if(!deterministic_)`), so this is a single ordered sequential
    /// f64 fold over the labels in row order — the bit-exact deterministic anchor.
    pub fn boost_from_score(&self, label: &[f32]) -> f64 {
        match self {
            Objective::Regression { .. } => {
                if label.is_empty() {
                    return 0.0;
                }
                let mut suml = 0.0f64;
                for &l in label {
                    suml += l as f64;
                }
                let sumw = label.len() as f64;
                suml / sumw
            }
        }
    }

    /// Whether `|init|` exceeds the `kEpsilon` gate the GBDT loop uses to decide
    /// whether `BoostFromAverage` adds the init score / `AddBias` folds it
    /// (`gbdt.cpp:327`). Exposed so the boosting layer never re-derives the
    /// constant.
    pub fn init_score_is_significant(init: f64) -> bool {
        init.abs() > K_EPSILON as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    #[test]
    fn parse_regression_family() {
        assert_eq!(
            Objective::parse("regression").unwrap(),
            Objective::Regression { sqrt: false }
        );
        assert_eq!(
            Objective::parse("l2").unwrap(),
            Objective::Regression { sqrt: false }
        );
        assert_eq!(
            Objective::parse("regression sqrt").unwrap(),
            Objective::Regression { sqrt: true }
        );
    }

    #[test]
    fn parse_rejects_out_of_scope() {
        assert!(Objective::parse("binary").is_err());
        assert!(Objective::parse("multiclass num_class:3").is_err());
        assert!(Objective::parse("lambdarank").is_err());
    }

    #[test]
    fn gradients_l2_bit_exact_f32() {
        // grad[i] = (f32)(score - label); hess = 1.0. Hand-computed.
        let obj = Objective::Regression { sqrt: false };
        let score = [3.0f64, -2.5, 0.0, 100.25];
        let label = [1.0f32, 0.5, -4.0, 100.0];
        let mut grad = [0.0f32; 4];
        let mut hess = [0.0f32; 4];
        obj.get_gradients(&score, &label, &mut grad, &mut hess)
            .unwrap();
        // Compute the expected with the SAME f64-subtract-then-f32-cast op order.
        for i in 0..4 {
            let expect = (score[i] - label[i] as f64) as f32;
            assert_eq!(grad[i], expect, "grad[{i}] must be bit-exact f32");
            assert_eq!(hess[i], 1.0f32);
        }
        // Spot-check a known value: 3.0 - 1.0 = 2.0; 100.25 - 100.0 = 0.25.
        assert_eq!(grad[0], 2.0f32);
        assert_eq!(grad[3], 0.25f32);
    }

    #[test]
    fn gradients_length_mismatch_is_typed_error() {
        let obj = Objective::Regression { sqrt: false };
        let score = [1.0f64, 2.0];
        let label = [1.0f32]; // wrong length
        let mut grad = [0.0f32; 2];
        let mut hess = [0.0f32; 2];
        let err = obj
            .get_gradients(&score, &label, &mut grad, &mut hess)
            .unwrap_err();
        assert!(matches!(err, ObjectiveError::LengthMismatch { .. }));
    }

    #[test]
    fn boost_from_score_is_label_mean_ordered() {
        let obj = Objective::Regression { sqrt: false };
        let label = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        // Ordered f64 mean = 15 / 5 = 3.0.
        let init = obj.boost_from_score(&label);
        assert!((init - 3.0).abs() < ORACLE_TOL as f64);
        // Determinism: the ordered fold is reproducible.
        assert_eq!(init, obj.boost_from_score(&label));
    }

    #[test]
    fn sqrt_transform_labels() {
        let obj = Objective::Regression { sqrt: true };
        let label = [4.0f32, -9.0, 0.0];
        let t = obj.transform_labels(&label);
        // Sign(4)*sqrt(4) = 2; Sign(-9)*sqrt(9) = -3; 0 -> 0.
        assert_eq!(t[0], 2.0f32);
        assert_eq!(t[1], -3.0f32);
        assert_eq!(t[2], 0.0f32);
        // Non-sqrt passes through.
        let obj2 = Objective::Regression { sqrt: false };
        assert_eq!(obj2.transform_labels(&label), label.to_vec());
    }

    #[test]
    fn flags_match_cpp() {
        let obj = Objective::Regression { sqrt: false };
        assert!(obj.is_constant_hessian());
        assert!(!obj.is_renew_tree_output());
        assert!(Objective::init_score_is_significant(1.0));
        assert!(!Objective::init_score_is_significant(0.0));
    }
}
