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
use crate::percentile::{percentile_fun, weighted_percentile_fun};

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
    /// `regression_l1` (`RegressionL1loss`, `regression_objective.hpp:207-288`):
    /// `grad = (score_t)Sign(score - label)`, `hess = 1.0f`; `BoostFromScore` is
    /// the label MEDIAN (`PercentileFun` at alpha = 0.5, NOT the mean); and
    /// `IsRenewTreeOutput() == true` — after each tree, every leaf's output is
    /// overwritten with the median RESIDUAL of its rows (Pitfall 2/3).
    RegressionL1,
    /// `huber` (`RegressionHuberLoss`, `regression_objective.hpp:293`): a clipped
    /// L2/L1 hybrid. `grad = clamp(score-label, -alpha, +alpha)` in f64 then cast
    /// f32; `hess = 1.0` (constant). `alpha` is the Huber δ (config `alpha`, default
    /// 0.9). Inherits L2's `BoostFromScore` (label mean); `IsRenewTreeOutput()`
    /// false; `sqrt` is force-disabled in C++ (it warns and clears the flag).
    Huber {
        /// The Huber δ threshold (config `alpha`).
        alpha: f64,
    },
    /// `fair` (`RegressionFairLoss`, `regression_objective.hpp:351`): a smooth
    /// robust loss. `grad = c*x/(|x|+c)`, `hess = c²/(|x|+c)²` with `x = score-label`
    /// (config `fair_c`, default 1.0). `IsConstantHessian()` false; inherits L2's
    /// `BoostFromScore`; `IsRenewTreeOutput()` false.
    Fair {
        /// The Fair `c` parameter (config `fair_c`).
        c: f64,
    },
    /// `quantile` (`RegressionQuantileloss`, `regression_objective.hpp:481`):
    /// pinball loss. `grad = (1-alpha)` if `score-label >= 0` else `-alpha`;
    /// `hess = 1.0`. `BoostFromScore` is the label percentile at `alpha`;
    /// `IsRenewTreeOutput()` true (renew with the residual percentile at `alpha`).
    /// C++ `CHECK(alpha > 0 && alpha < 1)`.
    Quantile {
        /// The quantile level (config `alpha`, default 0.9).
        alpha: f64,
    },
    /// `mape` (`RegressionMAPELOSS`, `regression_objective.hpp:579`): subclasses
    /// `RegressionL1loss` with a per-row `label_weight = 1/max(1, |label|)` (f32,
    /// computed at Init). `grad = Sign(score-label) * label_weight`; `hess = 1.0`
    /// (constant, unweighted). `BoostFromScore` is the weighted label median (alpha
    /// 0.5) with `label_weight`; `IsRenewTreeOutput()` true (weighted residual
    /// median at 0.5 with `label_weight`).
    Mape,
}

/// C++ MAPE `label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i]))`
/// (`regression_objective.hpp:601`), computed in f32 at Init. Exposed as a free
/// helper so both `get_gradients` and the renewal path share the SAME f32 op order.
#[inline]
fn mape_label_weight(label: f32) -> f32 {
    1.0f32 / label.abs().max(1.0f32)
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
            // C++ aliases for RegressionL1loss (objective_function.cpp / config
            // alias table): `regression_l1`, `l1`, `mean_absolute_error`, `mae`.
            "regression_l1" | "l1" | "mean_absolute_error" | "mae" => Ok(Objective::RegressionL1),
            // huber/fair/quantile carry params filled by `from_config`; parse seeds
            // the config.h defaults (alpha 0.9, fair_c 1.0) so the bare-name parse is
            // a valid variant. quantile's `< 1` CHECK is enforced in `from_config`
            // (it sees the resolved Config value).
            "huber" => Ok(Objective::Huber { alpha: 0.9 }),
            "fair" => Ok(Objective::Fair { c: 1.0 }),
            "quantile" => Ok(Objective::Quantile { alpha: 0.9 }),
            // `mape` config aliases (config_auto.cpp valid-objective list):
            // `mape`, `mean_absolute_percentage_error`.
            "mape" | "mean_absolute_percentage_error" => Ok(Objective::Mape),
            other => Err(ObjectiveError::Unsupported {
                name: other.to_string(),
            }),
        }
    }

    /// Build the objective from a resolved [`lgbm_core::Config`] (D-02 — the
    /// builder never forks the objective name; it routes the config's
    /// `objective` string through [`Self::parse`]).
    ///
    /// reg_sqrt (GAP E, 06-06): in C++ `reg_sqrt` is a CONFIG flag the regression
    /// objective reads (`RegressionL2loss::sqrt_`), independent of the objective
    /// string. The 06-02 port keyed `sqrt` only off the `"regression sqrt"` string
    /// token, so `config.reg_sqrt = true` (e.g. via `TrainingBuilder.reg_sqrt(true)`)
    /// had no effect — `reg_sqrt=1` was not drivable end-to-end. Here we OR the config
    /// flag into the parsed `sqrt` so EITHER route (the `"... sqrt"` token OR
    /// `config.reg_sqrt`) activates the sqrt transform, matching C++.
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] when `config.objective` is out of 06-02
    /// scope.
    pub fn from_config(config: &lgbm_core::Config) -> Result<Objective, ObjectiveError> {
        match Objective::parse(&config.objective)? {
            Objective::Regression { sqrt } => Ok(Objective::Regression {
                sqrt: sqrt || config.reg_sqrt,
            }),
            // huber δ is config `alpha` (RegressionHuberLoss ctor:
            // `alpha_ = config.alpha`). sqrt is force-disabled in C++ but that is a
            // label-transform concern handled by `transform_labels` (huber returns
            // labels unchanged regardless).
            Objective::Huber { .. } => Ok(Objective::Huber { alpha: config.alpha }),
            // fair c is config `fair_c` (RegressionFairLoss ctor: `c_ = config.fair_c`).
            Objective::Fair { .. } => Ok(Objective::Fair { c: config.fair_c }),
            // quantile alpha is config `alpha`; C++ `CHECK(alpha > 0 && alpha < 1)`
            // (regression_objective.hpp:483). The generic Config check covers `> 0`;
            // enforce the `< 1` half here as a typed reject (never a panic).
            Objective::Quantile { .. } => {
                if !(config.alpha > 0.0 && config.alpha < 1.0) {
                    return Err(ObjectiveError::InvalidParam {
                        param: "alpha".to_string(),
                        objective: "quantile".to_string(),
                        value: config.alpha,
                        reason: "must satisfy 0 < alpha < 1".to_string(),
                    });
                }
                Ok(Objective::Quantile { alpha: config.alpha })
            }
            // reg_sqrt only applies to the L2 regression family in C++; the rest
            // (L1 / mape) ignore it.
            other => Ok(other),
        }
    }

    /// C++ `IsConstantHessian` (no weights → true). The spine path is unweighted,
    /// so the hessian is the constant `1.0`.
    pub fn is_constant_hessian(&self) -> bool {
        match self {
            // L1 hess is the constant 1.0 too (unweighted) — RegressionL1loss
            // inherits IsConstantHessian from RegressionL2loss.
            // Huber hess = 1.0 (constant, inherits L2). Mape hess = 1.0 and overrides
            // IsConstantHessian()==true (regression_objective.hpp:665).
            Objective::Regression { .. }
            | Objective::RegressionL1
            | Objective::Huber { .. }
            | Objective::Mape => true,
            // Quantile hess = 1.0 (constant); RegressionQuantileloss does NOT
            // override IsConstantHessian so it inherits L2's `true` — correct, the
            // hessian is the constant 1.0.
            Objective::Quantile { .. } => true,
            // Fair hess = c²/(|x|+c)² is per-row; RegressionFairLoss overrides
            // IsConstantHessian()==false (regression_objective.hpp:386).
            Objective::Fair { .. } => false,
        }
    }

    /// C++ `IsRenewTreeOutput` — false for L2 (the learner's Newton output is the
    /// final leaf value; no median-residual renewal). `regression_l1` overrides
    /// this in 06-03.
    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            // L2/huber/fair: the Newton output is the final leaf value (no renewal).
            Objective::Regression { .. } | Objective::Huber { .. } | Objective::Fair { .. } => {
                false
            }
            // RegressionL1loss / RegressionQuantileloss / RegressionMAPELOSS all
            // override IsRenewTreeOutput() == true.
            Objective::RegressionL1 | Objective::Quantile { .. } | Objective::Mape => true,
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
            // L1 has no Init label transform (no sqrt option). Huber/Fair/Quantile
            // inherit RegressionL2loss but carry no `sqrt` flag (huber force-disables
            // it in C++; fair/quantile never set it via the typed builder), so the
            // labels pass through. Mape subclasses L1 — no transform.
            Objective::RegressionL1
            | Objective::Huber { .. }
            | Objective::Fair { .. }
            | Objective::Quantile { .. }
            | Objective::Mape => labels.to_vec(),
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
            Objective::RegressionL1 => {
                // RegressionL1loss::GetGradients (no-weights,
                // regression_objective.hpp:217-225):
                //   diff = score[i] - label_[i]  (f64; label promoted)
                //   grad[i] = (score_t)Common::Sign(diff)
                //   hess[i] = 1.0f
                for i in 0..n {
                    let diff = score[i] - label[i] as f64;
                    gradients[i] = sign(diff) as f32;
                    hessians[i] = 1.0f32;
                }
            }
            Objective::Huber { alpha } => {
                // RegressionHuberLoss::GetGradients (no-weights,
                // regression_objective.hpp:312-325):
                //   diff = score[i] - label_[i]          (f64; label promoted)
                //   if |diff| <= alpha: grad = (score_t)diff
                //   else:               grad = (score_t)(Sign(diff) * alpha)
                //   hess = 1.0f
                for i in 0..n {
                    let diff = score[i] - label[i] as f64;
                    gradients[i] = if diff.abs() <= *alpha {
                        diff as f32
                    } else {
                        (sign(diff) * *alpha) as f32
                    };
                    hessians[i] = 1.0f32;
                }
            }
            Objective::Fair { c } => {
                // RegressionFairLoss::GetGradients (no-weights,
                // regression_objective.hpp:364-369):
                //   x = score[i] - label_[i]                          (f64)
                //   grad = (score_t)(c * x / (|x| + c))
                //   hess = (score_t)(c * c / ((|x| + c) * (|x| + c)))
                let c = *c;
                for i in 0..n {
                    let x = score[i] - label[i] as f64;
                    let denom = x.abs() + c;
                    gradients[i] = (c * x / denom) as f32;
                    hessians[i] = (c * c / (denom * denom)) as f32;
                }
            }
            Objective::Quantile { alpha } => {
                // RegressionQuantileloss::GetGradients (no-weights,
                // regression_objective.hpp:494-503): `alpha_` is a `score_t` (f32) in
                // C++ and `delta` is the f32-cast residual; the branch + grad are all
                // f32.
                //   delta = (score_t)(score[i] - label_[i])
                //   grad  = (delta >= 0) ? (1.0f - alpha) : -alpha
                //   hess  = 1.0f
                let alpha_f32 = *alpha as f32;
                for i in 0..n {
                    let delta = (score[i] - label[i] as f64) as f32;
                    gradients[i] = if delta >= 0.0 {
                        1.0f32 - alpha_f32
                    } else {
                        -alpha_f32
                    };
                    hessians[i] = 1.0f32;
                }
            }
            Objective::Mape => {
                // RegressionMAPELOSS::GetGradients (no-weights,
                // regression_objective.hpp:619-625):
                //   diff = score[i] - label_[i]                       (f64)
                //   grad = (score_t)(Sign(diff) * label_weight_[i])
                //   hess = 1.0f
                // where label_weight_[i] = 1/max(1,|label|) computed in f32 at Init.
                for i in 0..n {
                    let diff = score[i] - label[i] as f64;
                    let lw = mape_label_weight(label[i]);
                    gradients[i] = (sign(diff) * lw as f64) as f32;
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
            // L2 label mean. Huber/Fair inherit RegressionL2loss::BoostFromScore
            // unchanged (no override), so they use the SAME ordered f64 fold.
            Objective::Regression { .. } | Objective::Huber { .. } | Objective::Fair { .. } => {
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
            // RegressionQuantileloss::BoostFromScore (regression_objective.hpp:531):
            // the unweighted label percentile at `alpha` (PercentileFun over the
            // labels, NOT the mean). label_t (f32) promoted to f64.
            Objective::Quantile { alpha } => {
                let data: Vec<f64> = label.iter().map(|&l| l as f64).collect();
                percentile_fun(&data, *alpha)
            }
            // RegressionMAPELOSS::BoostFromScore (regression_objective.hpp:635): the
            // WEIGHTED label median (alpha = 0.5) with per-row label_weight =
            // 1/max(1,|label|).
            Objective::Mape => {
                let data: Vec<f64> = label.iter().map(|&l| l as f64).collect();
                let weights: Vec<f64> =
                    label.iter().map(|&l| mape_label_weight(l) as f64).collect();
                weighted_percentile_fun(&data, &weights, 0.5)
            }
            // RegressionL1loss::BoostFromScore (regression_objective.hpp:236-249):
            // the unweighted percentile at alpha = 0.5 — the label MEDIAN via
            // PercentileFun (NOT the mean). The macro reads `label_[i]` as
            // `label_t` (f32); we promote each to f64 (the percentile arithmetic is
            // the same; PercentileFun's casts are width-preserving for the median).
            Objective::RegressionL1 => {
                let data: Vec<f64> = label.iter().map(|&l| l as f64).collect();
                percentile_fun(&data, 0.5)
            }
        }
    }

    /// `RenewTreeOutput` per-leaf body — the (weighted) percentile of a leaf's
    /// RESIDUALS. `residuals` are the per-row `label[row] - score[row]` values for
    /// the rows in one leaf (the C++ `residual_getter`), in the data partition's row
    /// order; `labels` are the SAME rows' (untransformed) labels in the SAME order
    /// (needed only for MAPE's `label_weight`). Returns the new leaf output. The
    /// boosting layer gathers these for each leaf and calls this; for
    /// `IsRenewTreeOutput() == false` objectives the loop never invokes it.
    ///
    /// - `regression_l1` (`regression_objective.hpp:253-283`): unweighted residual
    ///   median (`PercentileFun`, alpha = 0.5).
    /// - `quantile` (`regression_objective.hpp:548-577`): unweighted residual
    ///   percentile at the objective's `alpha`.
    /// - `mape` (`regression_objective.hpp:642-660`): WEIGHTED residual median
    ///   (alpha = 0.5) with per-row `label_weight = 1/max(1,|label|)`.
    ///
    /// Only the unweighted L1/quantile path and the MAPE weighted path on the
    /// in-scope corpora are exercised; the weighted L1/quantile variants live in
    /// [`crate::percentile::weighted_percentile_fun`].
    pub fn renew_leaf_output(&self, residuals: &[f64], labels: &[f32]) -> f64 {
        match self {
            Objective::RegressionL1 => percentile_fun(residuals, 0.5),
            Objective::Quantile { alpha } => percentile_fun(residuals, *alpha),
            Objective::Mape => {
                let weights: Vec<f64> =
                    labels.iter().map(|&l| mape_label_weight(l) as f64).collect();
                weighted_percentile_fun(residuals, &weights, 0.5)
            }
            // Non-renew objectives never reach here (guarded by is_renew_tree_output);
            // fall back to the unweighted median rather than panic.
            Objective::Regression { .. } | Objective::Huber { .. } | Objective::Fair { .. } => {
                percentile_fun(residuals, 0.5)
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

    #[test]
    fn parse_regression_l1_family() {
        assert_eq!(Objective::parse("regression_l1").unwrap(), Objective::RegressionL1);
        assert_eq!(Objective::parse("l1").unwrap(), Objective::RegressionL1);
        assert_eq!(Objective::parse("mae").unwrap(), Objective::RegressionL1);
        assert_eq!(
            Objective::parse("mean_absolute_error").unwrap(),
            Objective::RegressionL1
        );
    }

    #[test]
    fn gradients_l1_sign_and_unit_hessian() {
        // grad[i] = Sign(score - label); hess = 1.0.
        let obj = Objective::RegressionL1;
        let score = [3.0f64, -2.5, 0.0, 100.25, 7.0];
        let label = [1.0f32, 0.5, 0.0, 100.0, 7.0];
        let mut grad = [0.0f32; 5];
        let mut hess = [0.0f32; 5];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        // 3-1>0 -> +1; -2.5-0.5<0 -> -1; 0-0==0 -> 0; 100.25-100>0 -> +1; 7-7==0 -> 0.
        assert_eq!(grad, [1.0f32, -1.0, 0.0, 1.0, 0.0]);
        assert_eq!(hess, [1.0f32; 5]);
    }

    #[test]
    fn gradients_l1_flags() {
        let obj = Objective::RegressionL1;
        assert!(obj.is_constant_hessian());
        assert!(obj.is_renew_tree_output());
    }

    #[test]
    fn boost_from_score_l1_is_median_not_mean() {
        let obj = Objective::RegressionL1;
        // Odd count median: [1,2,3,4,100] -> mean = 22, median (PercentileFun) = 3.
        let label = [1.0f32, 2.0, 3.0, 4.0, 100.0];
        let init = obj.boost_from_score(&label);
        assert_eq!(init, 3.0, "L1 BoostFromScore must be the median, not the mean");
        // Sanity: the L2 mean of the same labels is 22.0 — proving the divergence.
        let l2 = Objective::Regression { sqrt: false }.boost_from_score(&label);
        assert!((l2 - 22.0).abs() < 1e-9);
    }

    #[test]
    fn renew_leaf_output_is_median_residual() {
        let obj = Objective::RegressionL1;
        // residuals (label - score) for a leaf's rows. L1 ignores the labels arg.
        let residuals = [10.0f64, 8.0, 2.0, 0.0]; // even count, median(PercentileFun) = 5.
        assert_eq!(obj.renew_leaf_output(&residuals, &[0.0f32; 4]), 5.0);
        let odd = [1.0f64, 2.0, 3.0];
        assert_eq!(obj.renew_leaf_output(&odd, &[0.0f32; 3]), 2.0);
    }

    // ---- huber/fair/quantile/mape (Plan 07-02, OBJ-04 family A) ----

    fn cfg_with(objective: &str, alpha: f64, fair_c: f64) -> lgbm_core::Config {
        let mut c = lgbm_core::Config::default();
        c.objective = objective.to_string();
        c.alpha = alpha;
        c.fair_c = fair_c;
        c
    }

    #[test]
    fn parse_family_a_names() {
        assert_eq!(Objective::parse("huber").unwrap(), Objective::Huber { alpha: 0.9 });
        assert_eq!(Objective::parse("fair").unwrap(), Objective::Fair { c: 1.0 });
        assert_eq!(
            Objective::parse("quantile").unwrap(),
            Objective::Quantile { alpha: 0.9 }
        );
        assert_eq!(Objective::parse("mape").unwrap(), Objective::Mape);
        assert_eq!(
            Objective::parse("mean_absolute_percentage_error").unwrap(),
            Objective::Mape
        );
    }

    #[test]
    fn from_config_fills_params() {
        let h = Objective::from_config(&cfg_with("huber", 0.5, 1.0)).unwrap();
        assert_eq!(h, Objective::Huber { alpha: 0.5 });
        let f = Objective::from_config(&cfg_with("fair", 0.9, 2.0)).unwrap();
        assert_eq!(f, Objective::Fair { c: 2.0 });
        let q = Objective::from_config(&cfg_with("quantile", 0.1, 1.0)).unwrap();
        assert_eq!(q, Objective::Quantile { alpha: 0.1 });
    }

    #[test]
    fn quantile_alpha_check_rejects_out_of_range() {
        // alpha must satisfy 0 < alpha < 1 (C++ CHECK). alpha=1.0 is rejected.
        let err = Objective::from_config(&cfg_with("quantile", 1.0, 1.0)).unwrap_err();
        assert!(matches!(err, ObjectiveError::InvalidParam { .. }));
        // alpha just under 1 is accepted.
        assert!(Objective::from_config(&cfg_with("quantile", 0.999, 1.0)).is_ok());
    }

    #[test]
    fn family_a_renew_and_hessian_flags() {
        assert!(!Objective::Huber { alpha: 0.9 }.is_renew_tree_output());
        assert!(!Objective::Fair { c: 1.0 }.is_renew_tree_output());
        assert!(Objective::Quantile { alpha: 0.9 }.is_renew_tree_output());
        assert!(Objective::Mape.is_renew_tree_output());
        // Constant-hessian: huber/quantile/mape true; fair false.
        assert!(Objective::Huber { alpha: 0.9 }.is_constant_hessian());
        assert!(Objective::Quantile { alpha: 0.9 }.is_constant_hessian());
        assert!(Objective::Mape.is_constant_hessian());
        assert!(!Objective::Fair { c: 1.0 }.is_constant_hessian());
    }

    #[test]
    fn huber_gradient_is_clamped_to_alpha() {
        // residual far beyond alpha yields exactly ±alpha as f32; within yields diff.
        let obj = Objective::Huber { alpha: 0.9 };
        let score = [100.0f64, -100.0, 0.5, -0.5];
        let label = [0.0f32, 0.0, 0.0, 0.0];
        let mut grad = [0.0f32; 4];
        let mut hess = [0.0f32; 4];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        assert_eq!(grad[0], 0.9f32); // +100 clamped to +alpha
        assert_eq!(grad[1], -0.9f32); // -100 clamped to -alpha
        assert_eq!(grad[2], 0.5f32); // within alpha -> diff
        assert_eq!(grad[3], -0.5f32);
        assert_eq!(hess, [1.0f32; 4]);
    }

    #[test]
    fn fair_gradient_and_hessian() {
        // grad = c*x/(|x|+c); hess = c²/(|x|+c)². Hand-computed for c=1, x=3.
        let obj = Objective::Fair { c: 1.0 };
        let score = [3.0f64];
        let label = [0.0f32];
        let mut grad = [0.0f32; 1];
        let mut hess = [0.0f32; 1];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        // x=3: grad = 1*3/(3+1) = 0.75; hess = 1/(4*4) = 0.0625.
        assert_eq!(grad[0], (1.0f64 * 3.0 / 4.0) as f32);
        assert_eq!(hess[0], (1.0f64 / 16.0) as f32);
    }

    #[test]
    fn quantile_gradient_sign_style() {
        // grad = (1-alpha) if delta>=0 else -alpha; hess=1.
        let obj = Objective::Quantile { alpha: 0.9 };
        let score = [5.0f64, -5.0, 0.0];
        let label = [0.0f32, 0.0, 0.0];
        let mut grad = [0.0f32; 3];
        let mut hess = [0.0f32; 3];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        assert_eq!(grad[0], 1.0f32 - 0.9f32); // delta>=0
        assert_eq!(grad[1], -0.9f32); // delta<0
        assert_eq!(grad[2], 1.0f32 - 0.9f32); // delta==0 -> >= branch
        assert_eq!(hess, [1.0f32; 3]);
    }

    #[test]
    fn quantile_boost_from_score_is_label_percentile() {
        let obj = Objective::Quantile { alpha: 0.9 };
        // PercentileFun([1..10], 0.9): float_pos=(10-1)*(1-0.9)=0.9, pos=1,
        //   bias=0.9, desc=[10,9,...,1], v1=desc[0]=10, v2=desc[1]=9 ->
        //   10 - (10-9)*0.9 = 9.1.
        let label: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        assert!((obj.boost_from_score(&label) - 9.1).abs() < 1e-9);
    }

    #[test]
    fn quantile_renew_uses_alpha() {
        let obj = Objective::Quantile { alpha: 0.9 };
        let residuals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // Same as the percentile above -> 9.1. labels ignored for quantile.
        let got = obj.renew_leaf_output(&residuals, &[0.0f32; 10]);
        assert!((got - 9.1).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn mape_gradient_weights_by_inverse_label_magnitude() {
        // grad = Sign(score-label) * 1/max(1,|label|); hess=1.
        let obj = Objective::Mape;
        let score = [10.0f64, 0.0, 2.0];
        let label = [4.0f32, 5.0, 0.5];
        let mut grad = [0.0f32; 3];
        let mut hess = [0.0f32; 3];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        // row0: diff=6>0 -> +1 * 1/max(1,4)=1/4=0.25.
        assert_eq!(grad[0], (1.0f64 * (1.0f32 / 4.0) as f64) as f32);
        // row1: diff=-5<0 -> -1 * 1/max(1,5)=1/5=0.2.
        assert_eq!(grad[1], (-1.0f64 * (1.0f32 / 5.0) as f64) as f32);
        // row2: diff=1.5>0 -> +1 * 1/max(1,0.5)=1/1=1.0.
        assert_eq!(grad[2], 1.0f32);
        assert_eq!(hess, [1.0f32; 3]);
    }

    #[test]
    fn huber_fair_boost_from_score_is_label_mean() {
        let label = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!((Objective::Huber { alpha: 0.9 }.boost_from_score(&label) - 3.0).abs()
            < ORACLE_TOL as f64);
        assert!((Objective::Fair { c: 1.0 }.boost_from_score(&label) - 3.0).abs()
            < ORACLE_TOL as f64);
    }
}
