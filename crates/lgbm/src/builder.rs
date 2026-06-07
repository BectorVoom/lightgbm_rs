//! The public training builder (D-01) — an idiomatic owned Rust builder whose
//! setters resolve into [`lgbm_core::Config`] (D-02: Config is the single source
//! of truth; the builder NEVER forks defaults/aliases/validation).
//!
//! Idiomatic ergonomics live on the OUTSIDE (this builder); the faithful 1:1 C++
//! mirror lives BELOW the API boundary (the loop / objective / metric crates).
//! Every per-parameter setter records a `(key, value)` into a param map that is
//! resolved through `lgbm_core::Config::from_params` — the verbatim alias table +
//! CHECK validation — at [`build`](TrainingBuilder::build). Invalid combinations
//! return a typed [`crate::error::LgbmError`] (never a panic — Security V5,
//! threats T-06-02-01/03).

use std::collections::HashMap;

use lgbm_core::Config;

use crate::error::LgbmError;

/// An idiomatic owned training-config builder. Each setter returns `Self` so calls
/// chain; [`build`](Self::build) resolves the accumulated params into a validated
/// [`Config`].
///
/// D-03 full surface: a setter per in-scope spine parameter PLUS a
/// [`from_config`](Self::from_config) escape hatch so the oracle (and advanced
/// callers) can drive any parity-relevant `Config` field directly.
#[derive(Debug, Clone, Default)]
pub struct TrainingBuilder {
    /// Accumulated raw `(key, value)` params, resolved via `Config::from_params`.
    params: HashMap<String, String>,
    /// An optional pre-built `Config` (the `from_config` escape hatch). When set,
    /// `build` returns it directly (params still override individual fields is NOT
    /// supported — `from_config` is all-or-nothing, matching "drive any field").
    preset: Option<Config>,
}

impl TrainingBuilder {
    /// Start a fresh builder (all C++ defaults until overridden).
    pub fn new() -> Self {
        Self::default()
    }

    /// The `from_config(Config)` escape hatch (D-03): drive the loop from a
    /// fully-specified `Config`, bypassing the per-setter param map. Useful for the
    /// oracle to set any parity-relevant field. When set, [`build`](Self::build)
    /// returns this config verbatim.
    pub fn from_config(mut self, config: Config) -> Self {
        self.preset = Some(config);
        self
    }

    /// `objective` (e.g. `"regression"`). 06-02 supports the regression L2 family.
    pub fn objective(mut self, objective: &str) -> Self {
        self.params.insert("objective".into(), objective.into());
        self
    }

    /// `num_iterations` (boosting rounds).
    pub fn num_iterations(mut self, n: i32) -> Self {
        self.params.insert("num_iterations".into(), n.to_string());
        self
    }

    /// `learning_rate` (the per-tree shrinkage).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.params.insert("learning_rate".into(), lr.to_string());
        self
    }

    /// `num_leaves` (the leaf-wise growth cap).
    pub fn num_leaves(mut self, n: i32) -> Self {
        self.params.insert("num_leaves".into(), n.to_string());
        self
    }

    /// `max_depth` (`<= 0` = no cap).
    pub fn max_depth(mut self, d: i32) -> Self {
        self.params.insert("max_depth".into(), d.to_string());
        self
    }

    /// `min_data_in_leaf`.
    pub fn min_data_in_leaf(mut self, n: i32) -> Self {
        self.params.insert("min_data_in_leaf".into(), n.to_string());
        self
    }

    /// `metric` (e.g. `"l2,rmse"`).
    pub fn metric(mut self, metric: &str) -> Self {
        self.params.insert("metric".into(), metric.into());
        self
    }

    /// `num_class` (the multiclass class count; default 1). Required `>= 1` for the
    /// `multiclass`/`multiclassova` objectives — the GBDT loop grows `num_class`
    /// trees per iteration over the class-major layout.
    pub fn num_class(mut self, n: i32) -> Self {
        self.params.insert("num_class".into(), n.to_string());
        self
    }

    /// `sigmoid` (the logistic slope for `binary`/`multiclassova`; default 1.0,
    /// must be `> 0`).
    pub fn sigmoid(mut self, s: f64) -> Self {
        self.params.insert("sigmoid".into(), s.to_string());
        self
    }

    /// `boost_from_average` (the C++ regression default; D-15).
    pub fn boost_from_average(mut self, on: bool) -> Self {
        self.params
            .insert("boost_from_average".into(), on.to_string());
        self
    }

    /// `reg_sqrt` (GAP E / OBJ-03): when `true`, the regression objective fits L2 on
    /// the `Sign(label)*sqrt(|label|)` pre-transformed target and inverts via
    /// `ConvertOutput = Sign(x)*x*x`. Inserts the `reg_sqrt` raw param so it routes
    /// into `lgbm-core::Config.reg_sqrt` via `Config::from_params` (set.rs:314 /
    /// scope.rs:153) — mirrors the `boost_from_average` bool setter. Without this
    /// setter `reg_sqrt=1` was not drivable end-to-end through the builder.
    pub fn reg_sqrt(mut self, on: bool) -> Self {
        self.params.insert("reg_sqrt".into(), on.to_string());
        self
    }

    /// `alpha` (OBJ-04): the shared Huber δ AND quantile percentile level (config.h
    /// default 0.9, CHECK `> 0`; quantile additionally requires `< 1`). Routes into
    /// `lgbm-core::Config.alpha` via `Config::from_params` (set.rs:316 /
    /// scope.rs:154).
    pub fn alpha(mut self, v: f64) -> Self {
        self.params.insert("alpha".into(), v.to_string());
        self
    }

    /// `fair_c` (OBJ-04): the Fair-loss `c` parameter (config.h default 1.0, CHECK
    /// `> 0`). Routes into `lgbm-core::Config.fair_c` via `Config::from_params`
    /// (set.rs:319 / scope.rs:155).
    pub fn fair_c(mut self, v: f64) -> Self {
        self.params.insert("fair_c".into(), v.to_string());
        self
    }

    /// `poisson_max_delta_step` (OBJ-04): the Poisson hessian safeguard
    /// `exp(max_delta_step)` scale (config.h default 0.7, CHECK `> 0`). Routes into
    /// `lgbm-core::Config.poisson_max_delta_step` via `Config::from_params`
    /// (set.rs:322 / scope.rs:156).
    pub fn poisson_max_delta_step(mut self, v: f64) -> Self {
        self.params
            .insert("poisson_max_delta_step".into(), v.to_string());
        self
    }

    /// `tweedie_variance_power` (OBJ-04): the Tweedie `rho` parameter (config.h
    /// default 1.5, CHECK in `[1, 2)`). Routes into
    /// `lgbm-core::Config.tweedie_variance_power` via `Config::from_params`
    /// (set.rs:325 / scope.rs:157).
    pub fn tweedie_variance_power(mut self, v: f64) -> Self {
        self.params
            .insert("tweedie_variance_power".into(), v.to_string());
        self
    }

    /// `seed` (the master seed; derives the sub-seeds via `Config::from_params`).
    pub fn seed(mut self, seed: i32) -> Self {
        self.params.insert("seed".into(), seed.to_string());
        self
    }

    /// `deterministic` (the bit-exact ordered-reduction flag).
    pub fn deterministic(mut self, on: bool) -> Self {
        self.params.insert("deterministic".into(), on.to_string());
        self
    }

    /// `lambda_l1` / `lambda_l2` regularization.
    pub fn lambda_l1(mut self, v: f64) -> Self {
        self.params.insert("lambda_l1".into(), v.to_string());
        self
    }

    /// `lambda_l2` regularization.
    pub fn lambda_l2(mut self, v: f64) -> Self {
        self.params.insert("lambda_l2".into(), v.to_string());
        self
    }

    /// `bagging_fraction` (row subsampling rate, `(0, 1]`; BST-03). Paired with
    /// [`bagging_freq`](Self::bagging_freq) and [`bagging_seed`](Self::bagging_seed).
    pub fn bagging_fraction(mut self, v: f64) -> Self {
        self.params.insert("bagging_fraction".into(), v.to_string());
        self
    }

    /// `bagging_freq` (re-bag every k iters; `0` disables bagging).
    pub fn bagging_freq(mut self, k: i32) -> Self {
        self.params.insert("bagging_freq".into(), k.to_string());
        self
    }

    /// `bagging_seed` (the per-block RNG seed base; C++ default 3).
    pub fn bagging_seed(mut self, seed: i32) -> Self {
        self.params.insert("bagging_seed".into(), seed.to_string());
        self
    }

    /// `boosting` (the boosting type / sample-strategy alias, e.g. `"gbdt"` or
    /// `"goss"`; BST-04). `boosting=goss` is the C++ alias-expansion that resolves to
    /// `boosting=gbdt` + `data_sample_strategy=goss` (config.cpp; `set.rs:472-476`).
    pub fn boosting(mut self, kind: &str) -> Self {
        self.params.insert("boosting".into(), kind.into());
        self
    }

    /// `data_sample_strategy` (`"bagging"` (default) or `"goss"`; BST-04). Routes into
    /// `lgbm-core::Config.data_sample_strategy` via `Config::from_params`. Prefer
    /// [`goss`](Self::goss) for the GOSS convenience.
    pub fn data_sample_strategy(mut self, strategy: &str) -> Self {
        self.params
            .insert("data_sample_strategy".into(), strategy.into());
        self
    }

    /// `top_rate` (GOSS — the retained-largest-gradient fraction; config.h default
    /// 0.2, CHECK `[0,1]` with `top_rate + other_rate <= 1`; BST-04). Routes into
    /// `lgbm-core::Config.top_rate` via `Config::from_params` (set.rs:201-203).
    pub fn top_rate(mut self, v: f64) -> Self {
        self.params.insert("top_rate".into(), v.to_string());
        self
    }

    /// `other_rate` (GOSS — the randomly-sampled fraction of the rest; config.h
    /// default 0.1, CHECK `[0,1]` with `top_rate + other_rate <= 1`; BST-04). Routes
    /// into `lgbm-core::Config.other_rate` via `Config::from_params` (set.rs:205-207).
    pub fn other_rate(mut self, v: f64) -> Self {
        self.params.insert("other_rate".into(), v.to_string());
        self
    }

    /// GOSS convenience (BST-04): select gradient-based one-side sampling with the
    /// given `top_rate` / `other_rate`. Equivalent to
    /// `.boosting("goss").top_rate(top).other_rate(other)` — the `boosting=goss`
    /// alias-expansion sets `data_sample_strategy=goss` + `boosting=gbdt`.
    pub fn goss(self, top_rate: f64, other_rate: f64) -> Self {
        self.boosting("goss").top_rate(top_rate).other_rate(other_rate)
    }

    /// `early_stopping_round` (stop after this many non-improving rounds; `0`
    /// disables early stopping; BST-07).
    pub fn early_stopping_round(mut self, n: i32) -> Self {
        self.params.insert("early_stopping_round".into(), n.to_string());
        self
    }

    /// `early_stopping_min_delta` (an improvement must exceed this to count).
    pub fn early_stopping_min_delta(mut self, v: f64) -> Self {
        self.params
            .insert("early_stopping_min_delta".into(), v.to_string());
        self
    }

    /// `first_metric_only` (use only the first metric for the stop decision).
    pub fn first_metric_only(mut self, on: bool) -> Self {
        self.params.insert("first_metric_only".into(), on.to_string());
        self
    }

    /// `metric_freq` (evaluate metrics every k iters; MET-02).
    pub fn metric_freq(mut self, k: i32) -> Self {
        self.params.insert("metric_freq".into(), k.to_string());
        self
    }

    /// `is_provide_training_metric` (add the training set to the eval list; MET-02).
    pub fn is_provide_training_metric(mut self, on: bool) -> Self {
        self.params
            .insert("is_provide_training_metric".into(), on.to_string());
        self
    }

    /// Resolve the accumulated params (or the `from_config` preset) into a
    /// validated [`Config`] (D-02 — routed through `Config::from_params`'s alias
    /// table + CHECK validation).
    ///
    /// # Errors
    /// [`LgbmError::Config`] when a param value / combination fails the C++ CHECK
    /// validation (e.g. `learning_rate <= 0`, `num_iterations < 0`) — never a
    /// panic (T-06-02-01/03).
    pub fn build(self) -> Result<Config, LgbmError> {
        if let Some(cfg) = self.preset {
            return Ok(cfg);
        }
        Config::from_params(&self.params).map_err(LgbmError::Config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_routes_into_config() {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .boost_from_average(true)
            .seed(7)
            .deterministic(true)
            .build()
            .expect("valid config");
        assert_eq!(cfg.objective, "regression");
        assert_eq!(cfg.num_iterations, 10);
        assert!((cfg.learning_rate - 0.1).abs() < 1e-12);
        assert_eq!(cfg.num_leaves, 4);
        assert!(cfg.boost_from_average);
        assert_eq!(cfg.seed, 7);
        assert!(cfg.deterministic);
    }

    #[test]
    fn goss_setters_route_into_config() {
        // BST-04: top_rate/other_rate/data_sample_strategy must round-trip into Config.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .data_sample_strategy("goss")
            .top_rate(0.2)
            .other_rate(0.1)
            .build()
            .unwrap();
        assert_eq!(cfg.data_sample_strategy, "goss");
        assert!((cfg.top_rate - 0.2).abs() < 1e-12);
        assert!((cfg.other_rate - 0.1).abs() < 1e-12);
    }

    #[test]
    fn boosting_goss_alias_expands_to_gbdt_plus_data_sample_strategy() {
        // The C++ alias-expansion: boosting=goss => boosting=gbdt + data_sample_strategy=goss
        // (set.rs:472-476). Both the `.boosting("goss")` setter and the `.goss(..)`
        // convenience must trigger it.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .boosting("goss")
            .build()
            .unwrap();
        assert_eq!(cfg.boosting, "gbdt", "goss alias must expand boosting to gbdt");
        assert_eq!(cfg.data_sample_strategy, "goss");

        let cfg2 = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .goss(0.1, 0.05)
            .build()
            .unwrap();
        assert_eq!(cfg2.boosting, "gbdt");
        assert_eq!(cfg2.data_sample_strategy, "goss");
        assert!((cfg2.top_rate - 0.1).abs() < 1e-12);
        assert!((cfg2.other_rate - 0.05).abs() < 1e-12);
    }

    #[test]
    fn reg_sqrt_setter_round_trips_into_config() {
        // GAP E / OBJ-03: reg_sqrt=1 must be drivable end-to-end through the builder
        // (the D-03 full-param-surface contract). The setter inserts the `reg_sqrt`
        // raw param which routes into Config.reg_sqrt via Config::from_params.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .reg_sqrt(true)
            .build()
            .unwrap();
        assert!(cfg.reg_sqrt, "reg_sqrt(true) must round-trip into Config.reg_sqrt");
        // default stays false.
        let cfg0 = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .build()
            .unwrap();
        assert!(!cfg0.reg_sqrt, "reg_sqrt defaults to false");
    }

    #[test]
    fn exp_log_param_setters_round_trip_into_config() {
        // OBJ-04 (07-03): poisson_max_delta_step / tweedie_variance_power must be
        // drivable end-to-end through the builder (the D-03 full-param-surface
        // contract). The setters insert the raw params which route into Config via
        // Config::from_params.
        let cfg = TrainingBuilder::new()
            .objective("poisson")
            .num_iterations(5)
            .num_leaves(4)
            .poisson_max_delta_step(0.1)
            .build()
            .unwrap();
        assert!((cfg.poisson_max_delta_step - 0.1).abs() < 1e-12);

        let cfg_t = TrainingBuilder::new()
            .objective("tweedie")
            .num_iterations(5)
            .num_leaves(4)
            .tweedie_variance_power(1.9)
            .build()
            .unwrap();
        assert!((cfg_t.tweedie_variance_power - 1.9).abs() < 1e-12);

        // defaults stay at the config.h values.
        let cfg0 = TrainingBuilder::new()
            .objective("poisson")
            .num_iterations(5)
            .num_leaves(4)
            .build()
            .unwrap();
        assert!((cfg0.poisson_max_delta_step - 0.7).abs() < 1e-12);
        assert!((cfg0.tweedie_variance_power - 1.5).abs() < 1e-12);
    }

    #[test]
    fn builder_rejects_invalid_without_panic() {
        // learning_rate <= 0 fails the C++ CHECK_GT — typed error, not a panic.
        let err = TrainingBuilder::new()
            .objective("regression")
            .learning_rate(0.0)
            .build();
        assert!(err.is_err(), "learning_rate=0 must be rejected");
    }

    #[test]
    fn from_config_escape_hatch() {
        let mut base = Config::default();
        base.num_leaves = 9;
        base.objective = "regression".into();
        let cfg = TrainingBuilder::new().from_config(base.clone()).build().unwrap();
        assert_eq!(cfg, base);
    }
}
