//! The `metric=` parameter → eval-metric factory, mirroring the C++ string-keyed
//! `Metric::CreateMetric` (`LightGBM/src/metric/metric.cpp:19-130`).
//!
//! # Why this exists
//!
//! `Config::metric` is resolved by `lgbm_core::config::set` (alias-canonicalized
//! and deduplicated by `ParseMetrics`, falling back to the objective name when the
//! user supplies none). This module turns that `Vec<String>` into the concrete
//! metric objects the training loop evaluates — the step the port previously
//! skipped, so `metric=auc` silently did nothing.
//!
//! # C++ behaviors reproduced verbatim
//!
//! - **An unknown metric name is SILENTLY DROPPED.** `CreateMetric` returns
//!   `nullptr` for a name it does not recognize and `GBDT::Init` skips a null
//!   metric, so `metric=not_a_metric` trains with NO evaluation and no error.
//!   (Verified against `lightgbm==4.6.0`: `evals_result` comes back empty.)
//! - **`custom` produces no metric.** `ParseMetricAlias` folds
//!   `none`/`null`/`na`/`custom` to `custom`, which has no factory arm — the same
//!   silent drop, which is how `metric=None` disables evaluation.
//! - **One metric object can emit SEVERAL values**, each with its own name:
//!   `ndcg`/`map` emit one per `eval_at` cutoff (`ndcg@1`, `ndcg@3`, …), matching
//!   `Metric::GetName()` returning a `vector<string>`.

use lgbm_core::Config;
use lgbm_metric::{AucMu, BinaryMetric, Metric, MultiError, MultiLogloss, RankMetric,
                  RegressionMetricParams, XentropyMetric};
use lgbm_model::ObjectiveKind;

use crate::error::LgbmError;

/// A user-supplied custom-metric (feval) closure: `(raw_scores, labels) ->
/// (name, value, is_higher_better)`, mirroring the C++ `_EvalFunctionWrapper`
/// contract the Python `feval` marshals into. The closure is called once per eval
/// cadence with the SAME `(scores, labels)` the built-in metrics see, so it feeds
/// the SAME eval-history loop.
pub type CustomMetricClosure = Box<dyn Fn(&[f64], &[f32]) -> (String, f64, bool) + Send>;

/// One configured eval metric. Each arm wraps the corresponding `lgbm-metric`
/// implementation; the enum exists so the training loop can hold a heterogeneous
/// metric list without dynamic dispatch (the C++ `std::vector<Metric*>` analog).
pub enum EvalMetric {
    /// The pointwise regression family (`l1`/`l2`/`rmse`/`quantile`/`huber`/
    /// `fair`/`poisson`/`mape`/`gamma`/`gamma_deviance`/`tweedie`) over raw scores.
    Reg(Metric),
    /// The binary family (`binary_logloss`/`binary_error`/`auc`/`average_precision`).
    Bin(BinaryMetric),
    /// `multi_logloss` over the class-major score buffer.
    Multi(MultiLogloss),
    /// `multi_error` (named `multi_error@k` when `multi_error_top_k > 1`).
    MultiErr(MultiError),
    /// `auc_mu` over the class-major score buffer, weighted by
    /// `Config::auc_mu_weights_matrix`.
    AucMu(AucMu),
    /// The cross-entropy family (`cross_entropy`/`cross_entropy_lambda`/
    /// `kullback_leibler`).
    Xent(XentropyMetric),
    /// `ndcg` / `map` — MULTI-VALUED (one output per `eval_at` cutoff) and
    /// query-grouped, so evaluation additionally needs query boundaries.
    Rank(RankMetric),
    /// A user custom-metric (feval) closure. Its name and value come from the
    /// closure; it feeds the SAME eval-history loop as the built-ins.
    Custom(CustomMetricClosure),
}

impl EvalMetric {
    /// The per-output metric names, in output order (C++ `Metric::GetName()`).
    /// Every arm but `Rank` yields exactly one name.
    pub fn names(&self) -> Vec<String> {
        match self {
            EvalMetric::Reg(m) => vec![m.name().to_string()],
            EvalMetric::Bin(m) => vec![m.name().to_string()],
            EvalMetric::Multi(m) => vec![m.name().to_string()],
            EvalMetric::MultiErr(m) => vec![m.name()],
            EvalMetric::AucMu(m) => vec![m.name().to_string()],
            EvalMetric::Xent(m) => vec![m.name().to_string()],
            EvalMetric::Rank(m) => m.names(),
            // The custom-metric name is only knowable by CALLING the closure, so
            // the history setup uses this placeholder and the first eval replaces
            // it with the closure-supplied name (see `resolved_names`).
            EvalMetric::Custom(_) => vec!["custom".to_string()],
        }
    }

    /// Number of values this metric contributes to one eval row.
    pub fn num_outputs(&self) -> usize {
        match self {
            EvalMetric::Rank(m) => m.eval_at().len(),
            _ => 1,
        }
    }

    /// C++ `factor_to_bigger_better` — `+1` when bigger is better (AUC family,
    /// ndcg/map), `-1` for losses. Drives the early-stopping comparison direction.
    /// Repeated once per output value.
    pub fn factors(&self) -> Vec<f64> {
        let f = match self {
            EvalMetric::Reg(m) => m.factor_to_bigger_better(),
            EvalMetric::Bin(m) => m.factor_to_bigger_better(),
            EvalMetric::Multi(_) => -1.0,
            EvalMetric::MultiErr(m) => m.factor_to_bigger_better(),
            EvalMetric::AucMu(m) => m.factor_to_bigger_better(),
            EvalMetric::Xent(m) => m.factor_to_bigger_better(),
            EvalMetric::Rank(m) => m.factor_to_bigger_better(),
            // The custom metric's direction is dynamic; default to "lower is
            // better" for the ES factor (the value-recording path is unaffected).
            EvalMetric::Custom(_) => -1.0,
        };
        vec![f; self.num_outputs()]
    }

    /// Evaluate, returning one value per output name.
    ///
    /// `query_boundaries` is required by (and only read by) the `Rank` arm; the
    /// pointwise arms ignore it.
    ///
    /// # Errors
    /// [`LgbmError::Metric`] on a length mismatch or an out-of-range label;
    /// [`LgbmError::Unsupported`] when a ranking metric is evaluated without query
    /// boundaries; [`LgbmError::CustomMetric`] on a non-finite feval value.
    pub fn eval(
        &self,
        scores: &[f64],
        labels: &[f32],
        query_boundaries: Option<&[i32]>,
    ) -> Result<Vec<f64>, LgbmError> {
        match self {
            EvalMetric::Reg(m) => one(m.eval(scores, labels)),
            EvalMetric::Bin(m) => one(m.eval(scores, labels)),
            EvalMetric::Multi(m) => one(m.eval(scores, labels)),
            EvalMetric::MultiErr(m) => one(m.eval(scores, labels)),
            EvalMetric::AucMu(m) => one(m.eval(scores, labels)),
            EvalMetric::Xent(m) => one(m.eval(scores, labels)),
            EvalMetric::Rank(m) => {
                let qb = query_boundaries.ok_or_else(|| LgbmError::Unsupported {
                    name: format!(
                        "the `{}` metric is query-grouped and needs query/group boundaries, \
                         which this corpus does not carry",
                        m.names().first().map_or("ndcg", |s| s.as_str())
                    ),
                })?;
                m.eval(scores, labels, qb).map_err(LgbmError::Metric)
            }
            EvalMetric::Custom(f) => {
                let (_name, value, _higher) = f(scores, labels);
                // A NaN / non-finite custom value is surfaced as a typed error,
                // never silently recorded.
                if !value.is_finite() {
                    return Err(LgbmError::CustomMetric {
                        detail: "custom metric (feval) returned a non-finite value".into(),
                    });
                }
                Ok(vec![value])
            }
        }
    }

    /// The closure-supplied names (for `Custom`) or the static names. For `Custom`
    /// this calls the closure with the actual eval inputs so the recorded history
    /// key matches the user's metric name.
    pub fn resolved_names(&self, scores: &[f64], labels: &[f32]) -> Vec<String> {
        match self {
            EvalMetric::Custom(f) => vec![f(scores, labels).0],
            other => other.names(),
        }
    }

    /// True for the feval arm (the training loop resolves its history key lazily).
    pub fn is_custom(&self) -> bool {
        matches!(self, EvalMetric::Custom(_))
    }
}

fn one(r: Result<f64, lgbm_metric::MetricError>) -> Result<Vec<f64>, LgbmError> {
    r.map(|v| vec![v]).map_err(LgbmError::Metric)
}

/// Build the configured metric list from `config.metric` — the Rust
/// `Metric::CreateMetric` loop (`gbdt.cpp` calls it once per name in
/// `config.metric`).
///
/// Unrecognized names — including the `custom` sentinel that `none`/`null`/`na`
/// canonicalize to — are SILENTLY SKIPPED, exactly as the C++ factory's
/// `nullptr` return is. An empty result means "train with no evaluation", which
/// is the documented effect of `metric=None`.
pub fn create_metrics(config: &Config) -> Vec<EvalMetric> {
    let reg_params = RegressionMetricParams {
        alpha: config.alpha,
        fair_c: config.fair_c,
        tweedie_variance_power: config.tweedie_variance_power,
    };
    let sigmoid = config.sigmoid;
    let num_class = config.num_class;
    // The ConvertOutput transform the multiclass metrics apply. `multiclassova`
    // uses per-class sigmoids; everything else uses softmax (mirroring how the
    // C++ multiclass metrics read `config.objective`).
    let multi_kind = if is_ova(&config.objective) {
        ObjectiveKind::MulticlassOva { num_class, sigmoid }
    } else {
        ObjectiveKind::Multiclass { num_class }
    };

    let mut out = Vec::with_capacity(config.metric.len());
    for name in &config.metric {
        // The regression family first — it owns the largest arm set and its
        // `parse` already rejects (rather than mis-maps) foreign names.
        if let Ok(m) = Metric::parse_with_params(name, &reg_params) {
            out.push(EvalMetric::Reg(m));
            continue;
        }
        let built = match name.as_str() {
            "binary_logloss" => Some(EvalMetric::Bin(BinaryMetric::BinaryLogloss { sigmoid })),
            "binary_error" => Some(EvalMetric::Bin(BinaryMetric::BinaryError { sigmoid })),
            "auc" => Some(EvalMetric::Bin(BinaryMetric::Auc)),
            "average_precision" => Some(EvalMetric::Bin(BinaryMetric::AveragePrecision)),
            "multi_logloss" => Some(EvalMetric::Multi(MultiLogloss::new(
                multi_kind.clone(),
                num_class,
            ))),
            "multi_error" => Some(EvalMetric::MultiErr(MultiError::new(
                multi_kind.clone(),
                num_class,
                config.multi_error_top_k,
            ))),
            "auc_mu" => Some(EvalMetric::AucMu(AucMu::with_weights(
                num_class,
                &config.auc_mu_weights_matrix,
            ))),
            "ndcg" | "map" => RankMetric::parse(name, &config.eval_at, &config.label_gain)
                .map(EvalMetric::Rank),
            other => XentropyMetric::parse(other).ok().map(EvalMetric::Xent),
        };
        // `None` here is the C++ `CreateMetric` nullptr: skip, do not fail.
        if let Some(m) = built {
            out.push(m);
        }
    }
    out
}

/// `true` for the `multiclassova` objective family (per-class sigmoid outputs).
fn is_ova(objective: &str) -> bool {
    matches!(
        objective.split_whitespace().next().unwrap_or(""),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(pairs: &[(&str, &str)]) -> Config {
        let params: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Config::from_params(&params).expect("config builds")
    }

    fn names(pairs: &[(&str, &str)]) -> Vec<String> {
        create_metrics(&cfg(pairs))
            .iter()
            .flat_map(EvalMetric::names)
            .collect()
    }

    #[test]
    fn objective_fallback_produces_the_objective_metric() {
        assert_eq!(names(&[("objective", "regression")]), ["l2"]);
        assert_eq!(names(&[("objective", "regression_l1")]), ["l1"]);
        assert_eq!(names(&[("objective", "binary")]), ["binary_logloss"]);
        assert_eq!(names(&[("objective", "huber")]), ["huber"]);
    }

    #[test]
    fn explicit_metric_list_selects_those_metrics() {
        assert_eq!(names(&[("metric", "l2,rmse")]), ["l2", "rmse"]);
        assert_eq!(names(&[("objective", "binary"), ("metric", "auc")]), ["auc"]);
        assert_eq!(
            names(&[("objective", "binary"), ("metric", "binary_error,average_precision")]),
            ["binary_error", "average_precision"]
        );
        assert_eq!(names(&[("metric", "mape,gamma_deviance,tweedie")]), ["mape", "gamma_deviance", "tweedie"]);
    }

    #[test]
    fn none_and_unknown_metrics_produce_an_empty_metric_list() {
        // Matches lightgbm==4.6.0: `evals_result` is empty in both cases, no error.
        for v in ["none", "null", "na", "custom", "not_a_metric"] {
            assert!(names(&[("metric", v)]).is_empty(), "metric={v}");
        }
    }

    #[test]
    fn xentropy_metrics_are_built() {
        assert_eq!(
            names(&[("objective", "cross_entropy"), ("metric", "xentropy,xentlambda,kldiv")]),
            ["cross_entropy", "cross_entropy_lambda", "kullback_leibler"]
        );
    }

    #[test]
    fn multiclass_metrics_are_built_with_the_top_k_suffix() {
        assert_eq!(
            names(&[
                ("objective", "multiclass"),
                ("num_class", "3"),
                ("metric", "multi_logloss,multi_error,auc_mu"),
            ]),
            ["multi_logloss", "multi_error", "auc_mu"]
        );
        assert_eq!(
            names(&[
                ("objective", "multiclass"),
                ("num_class", "4"),
                ("metric", "multi_error"),
                ("multi_error_top_k", "2"),
            ]),
            ["multi_error@2"]
        );
    }

    #[test]
    fn rank_metrics_expand_one_output_per_sorted_eval_at() {
        assert_eq!(
            names(&[("objective", "lambdarank"), ("metric", "ndcg"), ("eval_at", "5,1,3")]),
            ["ndcg@1", "ndcg@3", "ndcg@5"]
        );
        assert_eq!(
            names(&[("objective", "lambdarank"), ("metric", "map"), ("eval_at", "2,1")]),
            ["map@1", "map@2"]
        );
        // Default eval_at is [1..=5] (DCGCalculator::DefaultEvalAt).
        assert_eq!(names(&[("objective", "lambdarank")]).len(), 5);
    }

    #[test]
    fn parametrized_regression_metrics_read_their_config_fields() {
        let m = create_metrics(&cfg(&[("metric", "quantile"), ("alpha", "0.25")]));
        assert!(matches!(m[0], EvalMetric::Reg(Metric::Quantile { alpha }) if alpha == 0.25));
        let m = create_metrics(&cfg(&[("metric", "fair"), ("fair_c", "3.5")]));
        assert!(matches!(m[0], EvalMetric::Reg(Metric::Fair { fair_c }) if fair_c == 3.5));
        let m = create_metrics(&cfg(&[("metric", "tweedie"), ("tweedie_variance_power", "1.8")]));
        assert!(matches!(
            m[0],
            EvalMetric::Reg(Metric::Tweedie { tweedie_variance_power }) if tweedie_variance_power == 1.8
        ));
    }

    #[test]
    fn auc_mu_picks_up_the_configured_weight_matrix() {
        let c = cfg(&[
            ("objective", "multiclass"),
            ("num_class", "2"),
            ("metric", "auc_mu"),
            ("auc_mu_weights", "0,2,3,0"),
        ]);
        let metrics = create_metrics(&c);
        let expected = AucMu::with_weights(2, &c.auc_mu_weights_matrix);
        assert!(matches!(&metrics[0], EvalMetric::AucMu(a) if *a == expected));
    }

    #[test]
    fn a_ranking_metric_without_query_boundaries_is_a_typed_error_not_a_panic() {
        let metrics = create_metrics(&cfg(&[("objective", "lambdarank"), ("metric", "ndcg")]));
        let err = metrics[0]
            .eval(&[0.1, 0.2], &[1.0, 0.0], None)
            .expect_err("ranking metric needs boundaries");
        assert!(err.to_string().contains("query/group boundaries"), "{err}");
    }
}
