//! Cross-entropy family metrics — `cross_entropy`, `cross_entropy_lambda`,
//! `kullback_leibler` — ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source,
//! `LightGBM/src/metric/xentropy_metric.hpp`):
//! - `XentLoss(label, prob)` (`:35-50`): the clipped binary cross-entropy
//!   `-(a + b)` with `a = label*log(prob)`, `b = (1-label)*log(1-prob)`, each clipped
//!   at `log_arg_epsilon = 1.0e-12`. NOTE the literal `1e-12` floor here is
//!   DISTINCT from `kEpsilon` (1e-15) — it is the cross-entropy-specific clip.
//! - `XentLambdaLoss(label, weight, hhat)` (`:53-55`): `XentLoss(label,
//!   1 - exp(-weight*hhat))`.
//! - `YentLoss(p)` (`:60-66`): the (negative) label entropy, the KL-divergence
//!   offset term presummed over the labels.
//! - `CrossEntropyMetric::Eval` (`:106-139`): prob-space — `objective->ConvertOutput`
//!   (sigmoid) FIRST, then `XentLoss`, averaged by `sum_weights_` (= num_data
//!   unweighted). `factor_to_bigger_better = -1`.
//! - `CrossEntropyLambdaMetric::Eval` (`:191-225`): `ConvertOutput = log1p(exp)`
//!   (the lambda, `ObjectiveKind::CrossEntropyLambda`), then `XentLambdaLoss`,
//!   averaged by `num_data` (NOT sum_weights). `factor = -1`.
//! - `KullbackLeiblerDivergence::Eval` (`:298-331`): identical to cross-entropy
//!   (sigmoid ConvertOutput + XentLoss / sum_weights) PLUS the precomputed
//!   `presum_label_entropy_` offset term (`:282-292`). `factor = -1`.
//!
//! The prob transform (`ConvertOutput`) is REUSED from the model objective shim —
//! this crate does NOT re-port it (Open-Q1): `cross_entropy` / `kullback_leibler`
//! use `convert_binary(score, 1.0)` (`ObjectiveKind::CrossEntropy`),
//! `cross_entropy_lambda` uses `convert_cross_entropy_lambda` (log1p(exp)).

use lgbm_model::objective::{convert_binary, convert_cross_entropy_lambda};

use crate::error::MetricError;

/// The cross-entropy clip floor (`xentropy_metric.hpp:36`): `1.0e-12`. This is the
/// cross-entropy-specific `log_arg_epsilon`, NOT `kEpsilon` (1e-15).
const LOG_ARG_EPSILON: f64 = 1.0e-12;

/// The cross-entropy-family metric factory, mirroring the C++ string-keyed
/// `CreateMetric` arms for `cross_entropy` / `cross_entropy_lambda` /
/// `kullback_leibler`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XentropyMetric {
    /// `cross_entropy` — sigmoid ConvertOutput, `XentLoss` / sum_weights.
    CrossEntropy,
    /// `cross_entropy_lambda` — `log1p(exp)` ConvertOutput, `XentLambdaLoss`
    /// (with weight 1) / num_data.
    CrossEntropyLambda,
    /// `kullback_leibler` — like cross-entropy plus the label-entropy offset.
    KullbackLeibler,
}

impl XentropyMetric {
    /// Parse a metric name into an [`XentropyMetric`]. Recognizes the C++ aliases
    /// (`xentropy`/`xentlambda`/`kldiv`).
    ///
    /// # Errors
    /// [`MetricError::Unsupported`] for an unrecognized / out-of-scope name.
    pub fn parse(name: &str) -> Result<XentropyMetric, MetricError> {
        match name {
            "cross_entropy" | "xentropy" => Ok(XentropyMetric::CrossEntropy),
            "cross_entropy_lambda" | "xentlambda" => Ok(XentropyMetric::CrossEntropyLambda),
            "kullback_leibler" | "kldiv" => Ok(XentropyMetric::KullbackLeibler),
            other => Err(MetricError::Unsupported {
                name: other.to_string(),
            }),
        }
    }

    /// C++ `Metric::Name`.
    pub fn name(&self) -> &'static str {
        match self {
            XentropyMetric::CrossEntropy => "cross_entropy",
            XentropyMetric::CrossEntropyLambda => "cross_entropy_lambda",
            XentropyMetric::KullbackLeibler => "kullback_leibler",
        }
    }

    /// C++ `factor_to_bigger_better`: `-1` for all three (losses).
    pub fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    /// Evaluate the metric over the f64 accumulated raw `scores` + f32 `labels`
    /// (unweighted path). Each prob-space metric applies `ConvertOutput` first.
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] (V5 boundary) if `scores.len() != labels.len()`.
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
        let sum_weights = n as f64;
        match self {
            XentropyMetric::CrossEntropy => {
                // ConvertOutput = sigmoid (sigmoid_ == 1.0; xentropy has no sigmoid param).
                let mut sum_loss = 0.0f64;
                for i in 0..n {
                    let p = convert_binary(scores[i], 1.0);
                    sum_loss += xent_loss(labels[i], p);
                }
                Ok(sum_loss / sum_weights)
            }
            XentropyMetric::CrossEntropyLambda => {
                // ConvertOutput = log1p(exp) (the lambda hhat). Weight is 1.0.
                // Averaged by num_data (NOT sum_weights) per the C++ Eval.
                let mut sum_loss = 0.0f64;
                for i in 0..n {
                    let hhat = convert_cross_entropy_lambda(scores[i]);
                    sum_loss += xent_lambda_loss(labels[i], 1.0, hhat);
                }
                Ok(sum_loss / n as f64)
            }
            XentropyMetric::KullbackLeibler => {
                // Offset term: presummed (negative) label entropy / sum_weights.
                let mut presum_label_entropy = 0.0f64;
                for &label in labels {
                    presum_label_entropy += yent_loss(label as f64);
                }
                presum_label_entropy /= sum_weights;
                // ConvertOutput = sigmoid, then XentLoss / sum_weights, + offset.
                let mut sum_loss = 0.0f64;
                for i in 0..n {
                    let p = convert_binary(scores[i], 1.0);
                    sum_loss += xent_loss(labels[i], p);
                }
                Ok(presum_label_entropy + sum_loss / sum_weights)
            }
        }
    }
}

/// C++ `XentLoss(label, prob)` (xentropy_metric.hpp:35-50). prob clipped at 1e-12.
#[inline]
fn xent_loss(label: f32, prob: f64) -> f64 {
    let label = label as f64;
    let mut a = label;
    if prob > LOG_ARG_EPSILON {
        a *= prob.ln();
    } else {
        a *= LOG_ARG_EPSILON.ln();
    }
    let mut b = 1.0 - label;
    if 1.0 - prob > LOG_ARG_EPSILON {
        b *= (1.0 - prob).ln();
    } else {
        b *= LOG_ARG_EPSILON.ln();
    }
    -(a + b)
}

/// C++ `XentLambdaLoss(label, weight, hhat)` (xentropy_metric.hpp:53-55).
#[inline]
fn xent_lambda_loss(label: f32, weight: f64, hhat: f64) -> f64 {
    xent_loss(label, 1.0 - (-weight * hhat).exp())
}

/// C++ `YentLoss(p)` (xentropy_metric.hpp:60-66): the (negative) label entropy.
/// `x*log(x) = 0` at x=0,1, so only the in-(0,1) terms are added.
#[inline]
fn yent_loss(p: f64) -> f64 {
    let mut hp = 0.0;
    if p > 0.0 {
        hp += p * p.ln();
    }
    let q = 1.0 - p;
    if q > 0.0 {
        hp += q * q.ln();
    }
    hp
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    const TOL: f64 = ORACLE_TOL as f64;

    #[test]
    fn parse_recognizes_xentropy_metrics() {
        assert_eq!(
            XentropyMetric::parse("cross_entropy").unwrap(),
            XentropyMetric::CrossEntropy
        );
        assert_eq!(
            XentropyMetric::parse("xentropy").unwrap(),
            XentropyMetric::CrossEntropy
        );
        assert_eq!(
            XentropyMetric::parse("cross_entropy_lambda").unwrap(),
            XentropyMetric::CrossEntropyLambda
        );
        assert_eq!(
            XentropyMetric::parse("kullback_leibler").unwrap(),
            XentropyMetric::KullbackLeibler
        );
        assert!(XentropyMetric::parse("nope").is_err());
    }

    #[test]
    fn names_and_factors() {
        for m in [
            XentropyMetric::CrossEntropy,
            XentropyMetric::CrossEntropyLambda,
            XentropyMetric::KullbackLeibler,
        ] {
            assert_eq!(m.factor_to_bigger_better(), -1.0, "{}", m.name());
        }
        assert_eq!(XentropyMetric::CrossEntropy.name(), "cross_entropy");
        assert_eq!(
            XentropyMetric::CrossEntropyLambda.name(),
            "cross_entropy_lambda"
        );
        assert_eq!(XentropyMetric::KullbackLeibler.name(), "kullback_leibler");
    }

    #[test]
    fn cross_entropy_applies_sigmoid_convert_output() {
        // ConvertOutput = sigmoid. score=0 -> p=0.5.
        // label=1: XentLoss = -(1*log(0.5) + 0) = log(2).
        // label=0: XentLoss = -(0 + 1*log(0.5)) = log(2). avg = log(2).
        let scores = [0.0f64, 0.0];
        let labels = [1.0f32, 0.0];
        let got = XentropyMetric::CrossEntropy.eval(&scores, &labels).unwrap();
        assert!((got - 2.0f64.ln()).abs() < TOL, "cross_entropy {got}");
    }

    #[test]
    fn cross_entropy_lambda_matches_hand_computed() {
        // ConvertOutput = log1p(exp(score)). score=0 -> hhat=log(2).
        // XentLambdaLoss(label, 1, hhat) = XentLoss(label, 1-exp(-hhat)).
        let scores = [0.0f64];
        let labels = [1.0f32];
        let hhat = 2.0f64.ln();
        let p = 1.0 - (-hhat).exp();
        let expect = xent_loss(1.0, p);
        let got = XentropyMetric::CrossEntropyLambda
            .eval(&scores, &labels)
            .unwrap();
        assert!((got - expect).abs() < TOL, "xentlambda {got} != {expect}");
    }

    #[test]
    fn kullback_leibler_adds_label_entropy_offset() {
        // KL = cross_entropy + presum_label_entropy. For label in {0,1} YentLoss=0
        // (x*log x = 0 at the endpoints), so KL == cross_entropy for binary labels.
        let scores = [0.0f64, 0.0];
        let labels = [1.0f32, 0.0];
        let ce = XentropyMetric::CrossEntropy.eval(&scores, &labels).unwrap();
        let kl = XentropyMetric::KullbackLeibler.eval(&scores, &labels).unwrap();
        assert!((kl - ce).abs() < TOL, "KL {kl} should equal CE {ce} for 0/1 labels");

        // For a continuous label 0.5, YentLoss(0.5) = log(0.5) != 0, so KL differs.
        let labels2 = [0.5f32];
        let scores2 = [0.0f64];
        let ce2 = XentropyMetric::CrossEntropy.eval(&scores2, &labels2).unwrap();
        let kl2 = XentropyMetric::KullbackLeibler.eval(&scores2, &labels2).unwrap();
        let offset = yent_loss(0.5);
        assert!((kl2 - (ce2 + offset)).abs() < TOL, "KL offset wrong");
    }

    #[test]
    fn length_mismatch_is_typed_error() {
        let err = XentropyMetric::CrossEntropy.eval(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, MetricError::LengthMismatch { .. }));
    }
}
