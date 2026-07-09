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

use crate::apply_grad_hess;
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
    /// `poisson` (`RegressionPoissonLoss`, `regression_objective.hpp:397`): the
    /// score `f` is the LOG mean. `grad = exp(f) - label` (f64 exp, cast f32);
    /// `hess = exp(f) * exp(max_delta_step)` where `max_delta_step = config
    /// poisson_max_delta_step` (default 0.7). `BoostFromScore = SafeLog(L2 mean)`
    /// (`:469`); `ConvertOutput = exp(f)`. `IsConstantHessian()` false (hessian
    /// depends on `exp(f)`). Init guard: every label `>= 0` and `Σ label != 0`.
    Poisson {
        /// `max_delta_step_` (config `poisson_max_delta_step`); `exp(max_delta_step)`
        /// scales the hessian (`regression_objective.hpp:441`).
        max_delta_step: f64,
    },
    /// `gamma` (`RegressionGammaLoss`, `regression_objective.hpp:680`): subclasses
    /// Poisson but overrides `GetGradients`: `exp_score = exp(-f)`,
    /// `grad = 1 - label*exp_score`, `hess = label*exp_score` (f64, cast f32).
    /// Inherits Poisson's `BoostFromScore = SafeLog(L2 mean)`, `ConvertOutput =
    /// exp(f)`, label `>= 0` guard, and `IsConstantHessian() == false`.
    Gamma,
    /// `tweedie` (`RegressionTweedieLoss`, `regression_objective.hpp:707`):
    /// subclasses Poisson. With `rho = config tweedie_variance_power` in `[1, 2)`:
    /// `exp_1 = exp((1-rho)*f)`, `exp_2 = exp((2-rho)*f)`;
    /// `grad = -label*exp_1 + exp_2`,
    /// `hess = -label*(1-rho)*exp_1 + (2-rho)*exp_2` (f64, cast f32). Inherits
    /// Poisson's `BoostFromScore = SafeLog(L2 mean)`, `ConvertOutput = exp(f)`,
    /// label `>= 0` guard, and `IsConstantHessian() == false`.
    Tweedie {
        /// `rho_` (config `tweedie_variance_power`, range `[1, 2)`).
        rho: f64,
    },
}

/// C++ `Common::SafeLog` (`common.h:877`): `log(x)` for `x > 0`, else `-inf`.
/// Used by the poisson/gamma/tweedie `BoostFromScore` to map the L2 label mean
/// into the log-mean score space (`regression_objective.hpp:469`).
#[inline]
fn safe_log(x: f64) -> f64 {
    if x > 0.0 { x.ln() } else { f64::NEG_INFINITY }
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
            // poisson/gamma/tweedie (OBJ-04 exp/log family). poisson/tweedie carry
            // params filled by `from_config` (config.h defaults: poisson_max_delta_step
            // 0.7, tweedie_variance_power 1.5); the bare-name parse seeds them so it is
            // a valid variant. gamma has no objective param.
            "poisson" => Ok(Objective::Poisson { max_delta_step: 0.7 }),
            "gamma" => Ok(Objective::Gamma),
            "tweedie" => Ok(Objective::Tweedie { rho: 1.5 }),
            other => Err(ObjectiveError::Unsupported {
                name: other.to_string(),
            }),
        }
    }

    /// The canonical objective name (the first-token string this variant parses
    /// from), used by the on-device routing seam (Phase-31 31-04) to classify the
    /// objective through `lgbm_compute::device_objective_supported`. This mirrors the
    /// C++ `ObjectiveFunction::GetName()` roster; the `sqrt` L2 sub-variant still
    /// reports `"regression"` (its on-device grad/hess kernel is the L2 kernel — the
    /// sqrt inversion is a ConvertOutput-time link, not a grad/hess fork).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Objective::Regression { .. } => "regression",
            Objective::RegressionL1 => "regression_l1",
            Objective::Huber { .. } => "huber",
            Objective::Fair { .. } => "fair",
            Objective::Quantile { .. } => "quantile",
            Objective::Mape => "mape",
            Objective::Poisson { .. } => "poisson",
            Objective::Gamma => "gamma",
            Objective::Tweedie { .. } => "tweedie",
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
            // poisson max_delta_step is config `poisson_max_delta_step`
            // (RegressionPoissonLoss ctor: `max_delta_step_ =
            // config.poisson_max_delta_step`). The generic Config check covers `> 0`.
            Objective::Poisson { .. } => Ok(Objective::Poisson {
                max_delta_step: config.poisson_max_delta_step,
            }),
            // tweedie rho is config `tweedie_variance_power` (RegressionTweedieLoss
            // ctor: `rho_ = config.tweedie_variance_power`). The generic Config check
            // covers the `[1, 2)` range.
            Objective::Tweedie { .. } => Ok(Objective::Tweedie {
                rho: config.tweedie_variance_power,
            }),
            // reg_sqrt only applies to the L2 regression family in C++; the rest
            // (L1 / mape / gamma) ignore it.
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
            // poisson/gamma/tweedie hessians all depend on exp(score) (per-row);
            // RegressionPoissonLoss overrides IsConstantHessian()==false
            // (regression_objective.hpp:474) and gamma/tweedie inherit it.
            Objective::Poisson { .. } | Objective::Gamma | Objective::Tweedie { .. } => false,
        }
    }

    /// C++ `IsRenewTreeOutput` — false for L2 (the learner's Newton output is the
    /// final leaf value; no median-residual renewal). `regression_l1` overrides
    /// this in 06-03.
    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            // L2/huber/fair: the Newton output is the final leaf value (no renewal).
            // poisson/gamma/tweedie inherit RegressionL2loss::IsRenewTreeOutput() ==
            // false (the Newton step over exp-derived g/h is the final leaf value).
            Objective::Regression { .. }
            | Objective::Huber { .. }
            | Objective::Fair { .. }
            | Objective::Poisson { .. }
            | Objective::Gamma
            | Objective::Tweedie { .. } => false,
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
            // poisson/gamma/tweedie force-disable sqrt in C++ (the Poisson ctor warns
            // and clears the flag) and carry no label transform — labels pass through.
            Objective::RegressionL1
            | Objective::Huber { .. }
            | Objective::Fair { .. }
            | Objective::Quantile { .. }
            | Objective::Mape
            | Objective::Poisson { .. }
            | Objective::Gamma
            | Objective::Tweedie { .. } => labels.to_vec(),
        }
    }

    /// The C++ objective `Init` label-domain guard surfaced as a typed `Result`
    /// (Security V5 / T-07-03-01), called once at train time before any gradient
    /// work. poisson/gamma/tweedie require every label `>= 0` AND `Σ label != 0`
    /// (`regression_objective.hpp:417-424`); the L2/L1/huber/fair/quantile/mape
    /// objectives have no label-domain guard (they accept any finite label).
    ///
    /// # Errors
    /// [`ObjectiveError::LabelRange`] for the first negative label (or a zero
    /// label-sum) fed to poisson/gamma/tweedie — never a panic.
    pub fn check_labels(&self, labels: &[f32]) -> Result<(), ObjectiveError> {
        let name = match self {
            Objective::Poisson { .. } => "poisson",
            Objective::Gamma => "gamma",
            Objective::Tweedie { .. } => "tweedie",
            // No label-domain guard for the other regression objectives.
            _ => return Ok(()),
        };
        // C++ Common::ObtainMinMaxSum → miny < 0 fatal, sumy == 0 fatal. The
        // deterministic ordered fold over f32 labels (promoted to f64 for the sum, as
        // the C++ `double sumy`).
        let mut sumy = 0.0f64;
        for &l in labels {
            if l < 0.0 {
                return Err(ObjectiveError::LabelRange {
                    label: l as f64,
                    objective: name.to_string(),
                    reason: "at least one target label is negative (must be >= 0)".to_string(),
                });
            }
            sumy += l as f64;
        }
        if sumy == 0.0 {
            return Err(ObjectiveError::LabelRange {
                label: 0.0,
                objective: name.to_string(),
                reason: "sum of labels is zero".to_string(),
            });
        }
        Ok(())
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
        // spike-068: each arm's per-row body is unchanged; only the serial `for i in
        // 0..n` loop is replaced by `apply_grad_hess` (serial below the grain floor,
        // rayon at/above it). Every output element is a pure function of its own row's
        // (score, label) written to a disjoint slot, so this is bit-exact.
        match self {
            Objective::Regression { .. } => {
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    ((s - l as f64) as f32, 1.0f32)
                });
            }
            Objective::RegressionL1 => {
                // RegressionL1loss::GetGradients (no-weights,
                // regression_objective.hpp:217-225):
                //   diff = score[i] - label_[i]  (f64; label promoted)
                //   grad[i] = (score_t)Common::Sign(diff)
                //   hess[i] = 1.0f
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let diff = s - l as f64;
                    (sign(diff) as f32, 1.0f32)
                });
            }
            Objective::Huber { alpha } => {
                // RegressionHuberLoss::GetGradients (no-weights,
                // regression_objective.hpp:312-325):
                //   diff = score[i] - label_[i]          (f64; label promoted)
                //   if |diff| <= alpha: grad = (score_t)diff
                //   else:               grad = (score_t)(Sign(diff) * alpha)
                //   hess = 1.0f
                let alpha = *alpha;
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let diff = s - l as f64;
                    let g = if diff.abs() <= alpha {
                        diff as f32
                    } else {
                        (sign(diff) * alpha) as f32
                    };
                    (g, 1.0f32)
                });
            }
            Objective::Fair { c } => {
                // RegressionFairLoss::GetGradients (no-weights,
                // regression_objective.hpp:364-369):
                //   x = score[i] - label_[i]                          (f64)
                //   grad = (score_t)(c * x / (|x| + c))
                //   hess = (score_t)(c * c / ((|x| + c) * (|x| + c)))
                let c = *c;
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let x = s - l as f64;
                    let denom = x.abs() + c;
                    ((c * x / denom) as f32, (c * c / (denom * denom)) as f32)
                });
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
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let delta = (s - l as f64) as f32;
                    let g = if delta >= 0.0 {
                        1.0f32 - alpha_f32
                    } else {
                        -alpha_f32
                    };
                    (g, 1.0f32)
                });
            }
            Objective::Mape => {
                // RegressionMAPELOSS::GetGradients (no-weights,
                // regression_objective.hpp:619-625):
                //   diff = score[i] - label_[i]                       (f64)
                //   grad = (score_t)(Sign(diff) * label_weight_[i])
                //   hess = 1.0f
                // where label_weight_[i] = 1/max(1,|label|) computed in f32 at Init.
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let diff = s - l as f64;
                    let lw = mape_label_weight(l);
                    ((sign(diff) * lw as f64) as f32, 1.0f32)
                });
            }
            Objective::Poisson { max_delta_step } => {
                // RegressionPoissonLoss::GetGradients (no-weights,
                // regression_objective.hpp:439-445):
                //   exp_max_delta_step = exp(max_delta_step_)   (hoisted, f64)
                //   exp_score = exp(score[i])                   (f64)
                //   grad = (score_t)(exp_score - label_[i])
                //   hess = (score_t)(exp_score * exp_max_delta_step)
                let exp_max_delta_step = max_delta_step.exp();
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let exp_score = s.exp();
                    ((exp_score - l as f64) as f32, (exp_score * exp_max_delta_step) as f32)
                });
            }
            Objective::Gamma => {
                // RegressionGammaLoss::GetGradients (no-weights,
                // regression_objective.hpp:694-700):
                //   exp_score = exp(-score[i])                  (f64)
                //   grad = (score_t)(1.0 - label_[i] * exp_score)
                //   hess = (score_t)(label_[i] * exp_score)
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let exp_score = (-s).exp();
                    let lab = l as f64;
                    ((1.0 - lab * exp_score) as f32, (lab * exp_score) as f32)
                });
            }
            Objective::Tweedie { rho } => {
                // RegressionTweedieLoss::GetGradients (no-weights,
                // regression_objective.hpp:729-737):
                //   exp_1 = exp((1 - rho_) * score[i])          (f64)
                //   exp_2 = exp((2 - rho_) * score[i])          (f64)
                //   grad = (score_t)(-label_[i] * exp_1 + exp_2)
                //   hess = (score_t)(-label_[i] * (1 - rho_) * exp_1 + (2 - rho_) * exp_2)
                let rho = *rho;
                let one_minus_rho = 1.0 - rho;
                let two_minus_rho = 2.0 - rho;
                apply_grad_hess(score, label, gradients, hessians, |s, l| {
                    let exp_1 = (one_minus_rho * s).exp();
                    let exp_2 = (two_minus_rho * s).exp();
                    let lab = l as f64;
                    (
                        (-lab * exp_1 + exp_2) as f32,
                        (-lab * one_minus_rho * exp_1 + two_minus_rho * exp_2) as f32,
                    )
                });
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
            // RegressionQuantileloss::BoostFromScore (regression_objective.hpp:524):
            // the unweighted label percentile via `PercentileFun(label_t, ...)`. Two
            // f32-fidelity details vs the f64 renew path: (1) `alpha_` is `score_t`
            // (f32), promoted to double inside the macro — so round alpha through f32;
            // (2) the macro instantiates T = `label_t` (f32): `ref_data` is f32 and the
            // result is `static_cast<label_t>(v1 - (v1 - v2) * bias)`, so the final
            // value is f32-narrowed. The labels are already f32; cast the percentile
            // result to f32 (then back to f64 for the score) to match C++.
            Objective::Quantile { alpha } => {
                let data: Vec<f64> = label.iter().map(|&l| l as f64).collect();
                let a = (*alpha as f32) as f64;
                percentile_fun(&data, a) as f32 as f64
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
            // RegressionPoissonLoss::BoostFromScore (regression_objective.hpp:469):
            // `SafeLog(RegressionL2loss::BoostFromScore(0))` — the LOG of the L2 label
            // mean (the score space is the log-mean). gamma/tweedie subclass Poisson
            // and inherit this BoostFromScore unchanged. The L2 mean is the SAME
            // ordered sequential f64 fold as the L2/huber/fair arm above.
            Objective::Poisson { .. } | Objective::Gamma | Objective::Tweedie { .. } => {
                if label.is_empty() {
                    return safe_log(0.0);
                }
                let mut suml = 0.0f64;
                for &l in label {
                    suml += l as f64;
                }
                let sumw = label.len() as f64;
                safe_log(suml / sumw)
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
            // C++ `RegressionQuantileloss::RenewTreeOutput` calls
            // `PercentileFun(double, ..., alpha_)` where `alpha_` is a `score_t`
            // (f32). Inside the macro `float_pos = (cnt-1) * (1.0 - alpha)` promotes
            // that f32 to double, so the effective alpha is the f32-rounded value
            // (e.g. `(double)0.9f = 0.8999999761581421`, NOT the exact f64 0.9).
            // Round through f32 to reproduce the C++ `pos`/`bias` selection bit-for-bit.
            Objective::Quantile { alpha } => percentile_fun(residuals, (*alpha as f32) as f64),
            Objective::Mape => {
                let weights: Vec<f64> =
                    labels.iter().map(|&l| mape_label_weight(l) as f64).collect();
                weighted_percentile_fun(residuals, &weights, 0.5)
            }
            // Non-renew objectives never reach here (guarded by is_renew_tree_output);
            // fall back to the unweighted median rather than panic.
            Objective::Regression { .. }
            | Objective::Huber { .. }
            | Objective::Fair { .. }
            | Objective::Poisson { .. }
            | Objective::Gamma
            | Objective::Tweedie { .. } => percentile_fun(residuals, 0.5),
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
        let mut c = lgbm_core::Config {
            objective: objective.to_string(),
            ..Default::default()
        };
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
        // PercentileFun([1..10], alpha): float_pos=(10-1)*(1-alpha), pos=1,
        //   bias=float_pos, desc=[10,9,...,1], v1=desc[0]=10, v2=desc[1]=9 ->
        //   10 - (10-9)*bias. C++ `alpha_` is `score_t` (f32) and BoostFromScore
        //   instantiates `PercentileFun(label_t)` (f32 result), so the EXACT value is
        //   the f32-rounded `10 - bias32` where bias32 derives from `(double)0.9f`,
        //   NOT the exact-f64 9.1. Assert against that faithful value.
        let label: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let a = (0.9f32) as f64;
        let bias = 9.0 * (1.0 - a); // pos-1 == 0
        let expected = (10.0 - (10.0 - 9.0) * bias) as f32 as f64;
        let got = obj.boost_from_score(&label);
        assert!((got - expected).abs() < 1e-12, "got {got} expected {expected}");
        // It is still ~9.1 (the f32-alpha shift is ~2.4e-7), guarding against a gross error.
        assert!((got - 9.1).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn quantile_renew_uses_alpha() {
        let obj = Objective::Quantile { alpha: 0.9 };
        let residuals: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // RenewTreeOutput instantiates `PercentileFun(double)` but `alpha_` is still
        // f32, so the effective alpha is `(double)0.9f`. The result is f64 (no final
        // f32 cast). labels ignored for quantile.
        let a = (0.9f32) as f64;
        let bias = 9.0 * (1.0 - a);
        let expected = 10.0 - (10.0 - 9.0) * bias;
        let got = obj.renew_leaf_output(&residuals, &[0.0f32; 10]);
        assert!((got - expected).abs() < 1e-12, "got {got} expected {expected}");
        assert!((got - 9.1).abs() < 1e-6, "got {got}");
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

    // ---- poisson/gamma/tweedie (Plan 07-03, OBJ-04 exp/log family) ----

    #[test]
    fn parse_exp_log_family_names() {
        assert_eq!(
            Objective::parse("poisson").unwrap(),
            Objective::Poisson { max_delta_step: 0.7 }
        );
        assert_eq!(Objective::parse("gamma").unwrap(), Objective::Gamma);
        assert_eq!(
            Objective::parse("tweedie").unwrap(),
            Objective::Tweedie { rho: 1.5 }
        );
    }

    #[test]
    fn from_config_fills_exp_log_params() {
        // poisson_max_delta_step default 0.7; the alt 0.1 axis resolves.
        let mut c = lgbm_core::Config {
            objective: "poisson".to_string(),
            ..Default::default()
        };
        assert_eq!(
            Objective::from_config(&c).unwrap(),
            Objective::Poisson { max_delta_step: 0.7 }
        );
        c.poisson_max_delta_step = 0.1;
        assert_eq!(
            Objective::from_config(&c).unwrap(),
            Objective::Poisson { max_delta_step: 0.1 }
        );
        // tweedie_variance_power default 1.5; the alt 1.1/1.9 axes resolve.
        let mut t = lgbm_core::Config {
            objective: "tweedie".to_string(),
            ..Default::default()
        };
        assert_eq!(
            Objective::from_config(&t).unwrap(),
            Objective::Tweedie { rho: 1.5 }
        );
        t.tweedie_variance_power = 1.9;
        assert_eq!(
            Objective::from_config(&t).unwrap(),
            Objective::Tweedie { rho: 1.9 }
        );
    }

    #[test]
    fn exp_log_family_flags() {
        // Non-constant hessian (exp-dependent), no renewal.
        for obj in [
            Objective::Poisson { max_delta_step: 0.7 },
            Objective::Gamma,
            Objective::Tweedie { rho: 1.5 },
        ] {
            assert!(!obj.is_constant_hessian(), "{obj:?} hessian is per-row");
            assert!(!obj.is_renew_tree_output(), "{obj:?} has no renewal");
        }
    }

    #[test]
    fn poisson_gradient_and_hessian() {
        // grad = exp(score) - label; hess = exp(score) * exp(max_delta_step).
        let obj = Objective::Poisson { max_delta_step: 0.7 };
        let score = [0.0f64, 1.0];
        let label = [1.0f32, 2.0];
        let mut grad = [0.0f32; 2];
        let mut hess = [0.0f32; 2];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        let emds = 0.7f64.exp();
        assert_eq!(grad[0], (0.0f64.exp() - 1.0) as f32); // exp(0)=1; 1-1=0
        assert_eq!(grad[1], (1.0f64.exp() - 2.0) as f32);
        assert_eq!(hess[0], (0.0f64.exp() * emds) as f32);
        assert_eq!(hess[1], (1.0f64.exp() * emds) as f32);
    }

    #[test]
    fn gamma_gradient_and_hessian() {
        // exp_score = exp(-score); grad = 1 - label*exp_score; hess = label*exp_score.
        let obj = Objective::Gamma;
        let score = [0.0f64, 2.0];
        let label = [3.0f32, 1.0];
        let mut grad = [0.0f32; 2];
        let mut hess = [0.0f32; 2];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        for i in 0..2 {
            let es = (-score[i]).exp();
            let lab = label[i] as f64;
            assert_eq!(grad[i], (1.0 - lab * es) as f32);
            assert_eq!(hess[i], (lab * es) as f32);
        }
    }

    #[test]
    fn tweedie_gradient_and_hessian() {
        let rho = 1.5f64;
        let obj = Objective::Tweedie { rho };
        let score = [0.5f64, -1.0];
        let label = [2.0f32, 4.0];
        let mut grad = [0.0f32; 2];
        let mut hess = [0.0f32; 2];
        obj.get_gradients(&score, &label, &mut grad, &mut hess).unwrap();
        for i in 0..2 {
            let e1 = ((1.0 - rho) * score[i]).exp();
            let e2 = ((2.0 - rho) * score[i]).exp();
            let lab = label[i] as f64;
            assert_eq!(grad[i], (-lab * e1 + e2) as f32);
            assert_eq!(hess[i], (-lab * (1.0 - rho) * e1 + (2.0 - rho) * e2) as f32);
        }
    }

    #[test]
    fn exp_log_boost_from_score_is_safe_log_of_label_mean() {
        // BoostFromScore = SafeLog(L2 mean). mean([1,2,3,4,5]) = 3 -> ln(3).
        let label = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let expect = 3.0f64.ln();
        for obj in [
            Objective::Poisson { max_delta_step: 0.7 },
            Objective::Gamma,
            Objective::Tweedie { rho: 1.5 },
        ] {
            assert!(
                (obj.boost_from_score(&label) - expect).abs() < ORACLE_TOL as f64,
                "{obj:?} boost_from_score != ln(mean)"
            );
        }
    }

    #[test]
    fn exp_log_check_labels_rejects_negative_and_zero_sum() {
        // A negative label is rejected for each exp/log objective.
        for obj in [
            Objective::Poisson { max_delta_step: 0.7 },
            Objective::Gamma,
            Objective::Tweedie { rho: 1.5 },
        ] {
            let err = obj.check_labels(&[1.0f32, -0.5, 2.0]).unwrap_err();
            assert!(
                matches!(err, ObjectiveError::LabelRange { .. }),
                "{obj:?} must reject a negative label"
            );
            // A zero label-sum is rejected.
            let err0 = obj.check_labels(&[0.0f32, 0.0]).unwrap_err();
            assert!(matches!(err0, ObjectiveError::LabelRange { .. }));
            // A valid non-negative non-zero-sum corpus passes.
            assert!(obj.check_labels(&[0.0f32, 1.0, 2.0]).is_ok());
        }
        // The non-exp/log regression objectives have no label-domain guard.
        assert!(Objective::Regression { sqrt: false }.check_labels(&[-1.0f32]).is_ok());
        assert!(Objective::Huber { alpha: 0.9 }.check_labels(&[-100.0f32]).is_ok());
    }
}
