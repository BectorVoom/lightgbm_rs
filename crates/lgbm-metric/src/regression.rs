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
//!
//! # MET-03 extended regression metrics (07-04)
//! The extended family — `quantile`/`huber`/`fair`/`poisson`/`mape`/`gamma`/
//! `gamma_deviance`/`tweedie` — is added here as additional `LossOnPoint` arms in
//! the SAME ordered f64 fold, ported 1:1 from `regression_metric.hpp`:
//!   - `QuantileMetric::LossOnPoint` (:157-164), `config.alpha`.
//!   - `HuberLossMetric::LossOnPoint` (:191-198), `config.alpha`.
//!   - `FairLossMetric::LossOnPoint` (:212-216), `config.fair_c`, with `log1p`.
//!   - `PoissonMetric::LossOnPoint` (:229-235), `eps = 1e-10f` score floor.
//!   - `MAPEMetric::LossOnPoint` (:248-250), `max(1, |label|)` denominator.
//!   - `GammaMetric::LossOnPoint` (:261-268), `SafeLog` deviance form.
//!   - `GammaDevianceMetric::LossOnPoint` (:284-288) + `AverageLoss = sum*2`
//!     (:292-294), `epsilon = 1e-9`.
//!   - `TweedieMetric::LossOnPoint` (:305-314), `config.tweedie_variance_power`.
//!
//! ## ConvertOutput routing (the prob-space transform)
//! `RegressionMetric::Eval` applies `objective->ConvertOutput` per row when an
//! objective is present (`regression_metric.hpp:74-92`). For the in-scope cells the
//! objective is the matching family member: `poisson`/`gamma`/`tweedie` objectives
//! have `ConvertOutput = exp` (`ObjectiveKind::Poisson`,
//! `regression_objective.hpp:460`), so those three metrics apply `exp` to the raw
//! score before `LossOnPoint`. `quantile`/`huber`/`fair`/`mape` map to the
//! identity-`ConvertOutput` regression objective, so they score the raw score
//! directly. This mirrors the real-binary capture (objective == metric family).

use lgbm_model::objective::convert_poisson;

use crate::error::MetricError;

/// C++ `Common::SafeLog` (`common.h`): `x > kZeroThreshold ? log(x) : log(kZeroThreshold)`
/// with `kZeroThreshold = 1e-35f`. Used by the gamma metric's deviance form.
#[inline]
fn safe_log(x: f64) -> f64 {
    const K_ZERO_THRESHOLD: f64 = 1e-35;
    if x > K_ZERO_THRESHOLD {
        x.ln()
    } else {
        K_ZERO_THRESHOLD.ln()
    }
}

/// Extra config parameters the extended regression metrics read (mirroring the
/// C++ `Config` fields `RegressionMetric` consults via `config_`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionMetricParams {
    /// `config.alpha` — quantile/huber. C++ default 0.9 (quantile) / used by huber.
    pub alpha: f64,
    /// `config.fair_c` — fair loss `c`. C++ default 1.0.
    pub fair_c: f64,
    /// `config.tweedie_variance_power` — tweedie `rho` in `[1, 2)`. C++ default 1.5.
    pub tweedie_variance_power: f64,
}

impl Default for RegressionMetricParams {
    fn default() -> Self {
        // The C++ Config defaults for the metric-relevant fields.
        Self {
            alpha: 0.9,
            fair_c: 1.0,
            tweedie_variance_power: 1.5,
        }
    }
}

/// The evaluation-metric factory enum, mirroring the C++ string-keyed
/// `Metric::CreateMetric`. 06-02 ships the three regression metrics; binary /
/// multiclass / AUC land in 06-04+.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// `l2` — mean squared error: `Σ(score - label)^2 / num_data`.
    L2,
    /// `rmse` — root mean squared error: `sqrt(Σ(score - label)^2 / num_data)`.
    Rmse,
    /// `l1` — mean absolute error: `Σ|score - label| / num_data`.
    L1,
    /// `quantile` — pinball loss at `config.alpha` (`QuantileMetric`).
    Quantile {
        /// `config.alpha`.
        alpha: f64,
    },
    /// `huber` — Huber loss with knee at `config.alpha` (`HuberLossMetric`).
    Huber {
        /// `config.alpha`.
        alpha: f64,
    },
    /// `fair` — Fair loss with scale `config.fair_c` (`FairLossMetric`).
    Fair {
        /// `config.fair_c`.
        fair_c: f64,
    },
    /// `poisson` — Poisson deviance; `ConvertOutput = exp` then `score - label*ln(score)`.
    Poisson,
    /// `mape` — mean absolute percentage error (`MAPEMetric`).
    Mape,
    /// `gamma` — Gamma negative log-likelihood deviance; `ConvertOutput = exp`.
    Gamma,
    /// `gamma_deviance` — Gamma deviance with the `sum_loss * 2` average; `exp`.
    GammaDeviance,
    /// `tweedie` — Tweedie deviance at `config.tweedie_variance_power`; `exp`.
    Tweedie {
        /// `config.tweedie_variance_power` (`rho`).
        tweedie_variance_power: f64,
    },
}

impl Metric {
    /// Parse a metric name into a [`Metric`]. Recognizes the C++ aliases for the
    /// three regression metrics.
    ///
    /// # Errors
    /// [`MetricError::Unsupported`] for an out-of-scope / unrecognized metric name.
    pub fn parse(name: &str) -> Result<Metric, MetricError> {
        Metric::parse_with_params(name, &RegressionMetricParams::default())
    }

    /// Parse a metric name into a [`Metric`], binding the C++ `config_` params the
    /// parametrized arms (`quantile`/`huber`/`fair`/`tweedie`) read. The non-param
    /// arms ignore `params`.
    ///
    /// # Errors
    /// [`MetricError::Unsupported`] for an out-of-scope / unrecognized metric name.
    pub fn parse_with_params(
        name: &str,
        params: &RegressionMetricParams,
    ) -> Result<Metric, MetricError> {
        match name {
            "l2" | "regression_l2" | "mean_squared_error" | "mse" | "regression" => Ok(Metric::L2),
            "rmse" | "root_mean_squared_error" | "l2_root" => Ok(Metric::Rmse),
            "l1" | "mean_absolute_error" | "mae" | "regression_l1" => Ok(Metric::L1),
            "quantile" => Ok(Metric::Quantile { alpha: params.alpha }),
            "huber" => Ok(Metric::Huber { alpha: params.alpha }),
            "fair" => Ok(Metric::Fair {
                fair_c: params.fair_c,
            }),
            "poisson" => Ok(Metric::Poisson),
            "mape" => Ok(Metric::Mape),
            "gamma" => Ok(Metric::Gamma),
            "gamma_deviance" => Ok(Metric::GammaDeviance),
            "tweedie" => Ok(Metric::Tweedie {
                tweedie_variance_power: params.tweedie_variance_power,
            }),
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
            Metric::Quantile { .. } => "quantile",
            Metric::Huber { .. } => "huber",
            Metric::Fair { .. } => "fair",
            Metric::Poisson => "poisson",
            Metric::Mape => "mape",
            Metric::Gamma => "gamma",
            Metric::GammaDeviance => "gamma_deviance",
            Metric::Tweedie { .. } => "tweedie",
        }
    }

    /// C++ `factor_to_bigger_better` — `-1` for every regression loss (lower is
    /// better). All `RegressionMetric` subclasses return `-1` (`regression_metric.hpp:34`).
    pub fn factor_to_bigger_better(&self) -> f64 {
        -1.0
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
        // For the exp-ConvertOutput metrics (poisson/gamma/tweedie) the raw score is
        // transformed first (regression_metric.hpp:74-92, ConvertOutput = exp).
        let mut sum_loss = 0.0f64;
        for i in 0..n {
            let label = labels[i] as f64;
            let score = match self {
                Metric::Poisson | Metric::Gamma | Metric::GammaDeviance | Metric::Tweedie { .. } => {
                    convert_poisson(scores[i])
                }
                _ => scores[i],
            };
            sum_loss += self.loss_on_point(label, score);
        }
        let sum_weights = n as f64;
        let loss = match self {
            // RMSE AverageLoss: sqrt(sum_loss / sum_weights).
            Metric::Rmse => (sum_loss / sum_weights).sqrt(),
            // GammaDeviance AverageLoss: sum_loss * 2 (regression_metric.hpp:292-294).
            Metric::GammaDeviance => sum_loss * 2.0,
            // Default AverageLoss: sum_loss / sum_weights.
            _ => sum_loss / sum_weights,
        };
        Ok(loss)
    }

    /// C++ `<Metric>::LossOnPoint(label, score, config)` for each regression arm.
    /// `score` is post-`ConvertOutput` (already exp'd for poisson/gamma/tweedie).
    #[inline]
    fn loss_on_point(&self, label: f64, score: f64) -> f64 {
        match self {
            // L2 / RMSE: (score - label)^2.
            Metric::L2 | Metric::Rmse => {
                let diff = score - label;
                diff * diff
            }
            // L1: |score - label|.
            Metric::L1 => (score - label).abs(),
            // Quantile (regression_metric.hpp:157-164): pinball loss.
            Metric::Quantile { alpha } => {
                let delta = label - score;
                if delta < 0.0 {
                    (alpha - 1.0) * delta
                } else {
                    alpha * delta
                }
            }
            // Huber (regression_metric.hpp:191-198).
            Metric::Huber { alpha } => {
                let diff = score - label;
                if diff.abs() <= *alpha {
                    0.5 * diff * diff
                } else {
                    alpha * (diff.abs() - 0.5 * alpha)
                }
            }
            // Fair (regression_metric.hpp:212-216): c*x - c^2*log1p(x/c).
            Metric::Fair { fair_c } => {
                let x = (score - label).abs();
                let c = *fair_c;
                c * x - c * c * (x / c).ln_1p()
            }
            // Poisson (regression_metric.hpp:229-235): eps floor 1e-10f, score - label*ln(score).
            Metric::Poisson => {
                let eps = 1e-10_f32 as f64;
                let s = if score < eps { eps } else { score };
                s - label * s.ln()
            }
            // MAPE (regression_metric.hpp:248-250): |label-score| / max(1, |label|).
            Metric::Mape => (label - score).abs() / (1.0_f32 as f64).max(label.abs()),
            // Gamma (regression_metric.hpp:261-268): psi=1 deviance form via SafeLog.
            Metric::Gamma => {
                let psi = 1.0;
                let theta = -1.0 / score;
                let a = psi;
                let b = -safe_log(-theta);
                let c = 1.0 / psi * safe_log(label / psi) - safe_log(label) - 0.0;
                -((label * theta - b) / a + c)
            }
            // GammaDeviance (regression_metric.hpp:284-288): epsilon 1e-9.
            Metric::GammaDeviance => {
                let epsilon = 1.0e-9;
                let tmp = label / (score + epsilon);
                tmp - safe_log(tmp) - 1.0
            }
            // Tweedie (regression_metric.hpp:305-314): eps floor 1e-10f.
            Metric::Tweedie {
                tweedie_variance_power,
            } => {
                let rho = *tweedie_variance_power;
                let eps = 1e-10_f32 as f64;
                let s = if score < eps { eps } else { score };
                let a = label * ((1.0 - rho) * s.ln()).exp() / (1.0 - rho);
                let b = ((2.0 - rho) * s.ln()).exp() / (2.0 - rho);
                -a + b
            }
        }
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
        for m in [
            Metric::L2,
            Metric::Rmse,
            Metric::L1,
            Metric::Quantile { alpha: 0.9 },
            Metric::Huber { alpha: 1.0 },
            Metric::Fair { fair_c: 1.0 },
            Metric::Poisson,
            Metric::Mape,
            Metric::Gamma,
            Metric::GammaDeviance,
            Metric::Tweedie {
                tweedie_variance_power: 1.5,
            },
        ] {
            assert_eq!(m.factor_to_bigger_better(), -1.0, "{}", m.name());
        }
    }

    // --- MET-03 extended regression metric arms (07-04) ---

    #[test]
    fn parse_recognizes_extended_metric_names() {
        let p = RegressionMetricParams {
            alpha: 0.7,
            fair_c: 2.0,
            tweedie_variance_power: 1.3,
        };
        assert_eq!(
            Metric::parse_with_params("quantile", &p).unwrap(),
            Metric::Quantile { alpha: 0.7 }
        );
        assert_eq!(
            Metric::parse_with_params("huber", &p).unwrap(),
            Metric::Huber { alpha: 0.7 }
        );
        assert_eq!(
            Metric::parse_with_params("fair", &p).unwrap(),
            Metric::Fair { fair_c: 2.0 }
        );
        assert_eq!(Metric::parse("poisson").unwrap(), Metric::Poisson);
        assert_eq!(Metric::parse("mape").unwrap(), Metric::Mape);
        assert_eq!(Metric::parse("gamma").unwrap(), Metric::Gamma);
        assert_eq!(Metric::parse("gamma_deviance").unwrap(), Metric::GammaDeviance);
        assert_eq!(
            Metric::parse_with_params("tweedie", &p).unwrap(),
            Metric::Tweedie {
                tweedie_variance_power: 1.3
            }
        );
        // names round-trip.
        for (n, m) in [
            ("quantile", Metric::Quantile { alpha: 0.9 }),
            ("huber", Metric::Huber { alpha: 0.9 }),
            ("fair", Metric::Fair { fair_c: 1.0 }),
            ("poisson", Metric::Poisson),
            ("mape", Metric::Mape),
            ("gamma", Metric::Gamma),
            ("gamma_deviance", Metric::GammaDeviance),
            (
                "tweedie",
                Metric::Tweedie {
                    tweedie_variance_power: 1.5,
                },
            ),
        ] {
            assert_eq!(m.name(), n);
        }
    }

    #[test]
    fn eval_quantile_matches_hand_computed() {
        // alpha=0.9. delta = label - score. delta<0 -> (alpha-1)*delta; else alpha*delta.
        // row0: score 2, label 1 -> delta=-1 -> (0.9-1)*-1 = 0.1.
        // row1: score 1, label 3 -> delta=2 -> 0.9*2 = 1.8.
        let scores = [2.0f64, 1.0];
        let labels = [1.0f32, 3.0];
        let got = Metric::Quantile { alpha: 0.9 }.eval(&scores, &labels).unwrap();
        let expect = (0.1 + 1.8) / 2.0;
        assert!((got - expect).abs() < TOL, "quantile {got} != {expect}");
    }

    #[test]
    fn eval_huber_matches_hand_computed() {
        // alpha=1.0. row0 diff=1 (|d|<=a) -> 0.5*1 = 0.5.
        // row1 diff=-2 (|d|>a) -> a*(|d|-0.5a) = 1*(2-0.5)=1.5.
        let scores = [2.0f64, 1.0];
        let labels = [1.0f32, 3.0];
        let got = Metric::Huber { alpha: 1.0 }.eval(&scores, &labels).unwrap();
        let expect = (0.5 + 1.5) / 2.0;
        assert!((got - expect).abs() < TOL, "huber {got} != {expect}");
    }

    #[test]
    fn eval_fair_matches_hand_computed() {
        // c=1.0. x=|score-label|. loss = c*x - c^2*log1p(x/c).
        let scores = [3.0f64, 0.0];
        let labels = [1.0f32, 1.0];
        let c = 1.0f64;
        let l0 = c * 2.0 - c * c * (2.0f64 / c).ln_1p();
        let l1 = c * 1.0 - c * c * (1.0f64 / c).ln_1p();
        let got = Metric::Fair { fair_c: 1.0 }.eval(&scores, &labels).unwrap();
        assert!((got - (l0 + l1) / 2.0).abs() < TOL, "fair {got}");
    }

    #[test]
    fn eval_poisson_applies_exp_convert_output() {
        // Poisson metric ConvertOutput = exp. raw=ln(2) -> score=2. label=1.
        // LossOnPoint = score - label*ln(score) = 2 - 1*ln(2).
        let raw = [2.0f64.ln(), 3.0f64.ln()];
        let labels = [1.0f32, 2.0];
        let l0 = 2.0 - 1.0 * 2.0f64.ln();
        let l1 = 3.0 - 2.0 * 3.0f64.ln();
        let got = Metric::Poisson.eval(&raw, &labels).unwrap();
        assert!((got - (l0 + l1) / 2.0).abs() < TOL, "poisson {got}");
    }

    #[test]
    fn eval_mape_matches_hand_computed() {
        // |label-score| / max(1, |label|).
        // row0: |1-2|/max(1,1)=1. row1: |3-1|/max(1,3)=2/3.
        let scores = [2.0f64, 1.0];
        let labels = [1.0f32, 3.0];
        let got = Metric::Mape.eval(&scores, &labels).unwrap();
        let expect = (1.0 + 2.0 / 3.0) / 2.0;
        assert!((got - expect).abs() < TOL, "mape {got} != {expect}");
    }

    #[test]
    fn eval_gamma_deviance_doubles_and_applies_exp() {
        // GammaDeviance: ConvertOutput=exp, AverageLoss = sum*2.
        // raw=ln(2)->score=2. tmp = label/(score+1e-9). loss=tmp-log(tmp)-1.
        let raw = [2.0f64.ln()];
        let labels = [1.0f32];
        let score = 2.0f64;
        let tmp = 1.0 / (score + 1.0e-9);
        let expect = (tmp - tmp.ln() - 1.0) * 2.0;
        let got = Metric::GammaDeviance.eval(&raw, &labels).unwrap();
        assert!((got - expect).abs() < TOL, "gamma_deviance {got} != {expect}");
    }

    #[test]
    fn eval_gamma_is_finite_and_applies_exp() {
        let raw = [1.0f64, 0.5];
        let labels = [2.0f32, 3.0];
        let got = Metric::Gamma.eval(&raw, &labels).unwrap();
        assert!(got.is_finite(), "gamma must be finite: {got}");
    }

    #[test]
    fn eval_tweedie_matches_hand_computed() {
        // rho=1.5, ConvertOutput=exp. raw=ln(2)->s=2. label=1.
        let raw = [2.0f64.ln()];
        let labels = [1.0f32];
        let rho = 1.5f64;
        let s = 2.0f64;
        let a = 1.0 * ((1.0 - rho) * s.ln()).exp() / (1.0 - rho);
        let b = ((2.0 - rho) * s.ln()).exp() / (2.0 - rho);
        let expect = -a + b;
        let got = Metric::Tweedie {
            tweedie_variance_power: 1.5,
        }
        .eval(&raw, &labels)
        .unwrap();
        assert!((got - expect).abs() < TOL, "tweedie {got} != {expect}");
    }
}
