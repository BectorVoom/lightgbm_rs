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

    /// `predict_contrib` (PRD-04): when `true`, prediction emits TreeSHAP per-feature
    /// contributions (+ the expected-value base) instead of plain scores. Routes into
    /// `lgbm-core::Config.predict_contrib` via `Config::from_params` (set.rs:290 /
    /// scope.rs:137).
    pub fn predict_contrib(mut self, on: bool) -> Self {
        self.params.insert("predict_contrib".into(), on.to_string());
        self
    }

    /// `pred_early_stop` (PRD-05): enable the per-N-iterations prediction-early-stop
    /// margin hook. Routes into `lgbm-core::Config.pred_early_stop` (set.rs:292 /
    /// scope.rs:139). config.h default false.
    pub fn pred_early_stop(mut self, on: bool) -> Self {
        self.params.insert("pred_early_stop".into(), on.to_string());
        self
    }

    /// `pred_early_stop_freq` (PRD-05): how many accumulated iterations between margin
    /// checks (config.h default 10). Routes into `Config.pred_early_stop_freq`
    /// (set.rs:293 / scope.rs:140).
    pub fn pred_early_stop_freq(mut self, v: i32) -> Self {
        self.params
            .insert("pred_early_stop_freq".into(), v.to_string());
        self
    }

    /// `pred_early_stop_margin` (PRD-05): the decisive-margin threshold (config.h
    /// default 10.0). Routes into `Config.pred_early_stop_margin` (set.rs:294 /
    /// scope.rs:141).
    pub fn pred_early_stop_margin(mut self, v: f64) -> Self {
        self.params
            .insert("pred_early_stop_margin".into(), v.to_string());
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

    /// `cat_l2` (TRL-06) — extra L2 added to `lambda_l2` ONLY in the per-category
    /// split gain (config.h default 10.0). The deliberate categorical asymmetry.
    pub fn cat_l2(mut self, v: f64) -> Self {
        self.params.insert("cat_l2".into(), v.to_string());
        self
    }

    /// `cat_smooth` (TRL-06) — categorical CTR smoothing + the
    /// `RoundInt(hess*cnt_factor) >= cat_smooth` many-vs-many filter (default 10.0).
    pub fn cat_smooth(mut self, v: f64) -> Self {
        self.params.insert("cat_smooth".into(), v.to_string());
        self
    }

    /// `min_data_per_group` (TRL-06) — categorical many-vs-many minimum rows per
    /// accumulated category group (config.h default 100, `> 0`).
    pub fn min_data_per_group(mut self, v: i32) -> Self {
        self.params.insert("min_data_per_group".into(), v.to_string());
        self
    }

    /// `max_cat_threshold` (TRL-06) — cap on categories on one side of a
    /// many-vs-many split (config.h default 32, `> 0`).
    pub fn max_cat_threshold(mut self, v: i32) -> Self {
        self.params.insert("max_cat_threshold".into(), v.to_string());
        self
    }

    /// `max_cat_to_onehot` (TRL-06) — categorical features with
    /// `num_bin <= max_cat_to_onehot` use the one-hot path (config.h default 4, `> 0`).
    pub fn max_cat_to_onehot(mut self, v: i32) -> Self {
        self.params.insert("max_cat_to_onehot".into(), v.to_string());
        self
    }

    /// `monotone_constraints` (ADV-01) — per-feature `+1`/`-1`/`0` constraint
    /// vector (config.h `[]`; comma-separated, each in `{-1,0,1}`).
    pub fn monotone_constraints(mut self, constraints: &[i32]) -> Self {
        let csv = constraints
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.params.insert("monotone_constraints".into(), csv);
        self
    }

    /// `monotone_constraints_method` (ADV-01) — `basic`/`intermediate`/`advanced`
    /// (config.h default `basic`).
    pub fn monotone_constraints_method(mut self, method: &str) -> Self {
        self.params
            .insert("monotone_constraints_method".into(), method.to_string());
        self
    }

    /// `monotone_penalty` (ADV-01) — the monotone-split gain penalty (config.h
    /// default 0.0, `>= 0`).
    pub fn monotone_penalty(mut self, v: f64) -> Self {
        self.params.insert("monotone_penalty".into(), v.to_string());
        self
    }

    /// `interaction_constraints` (ADV-02) — the allowed co-occurring-feature
    /// groups spec (config.h default `""`, e.g. `"[0,1],[2]"`).
    pub fn interaction_constraints(mut self, spec: &str) -> Self {
        self.params
            .insert("interaction_constraints".into(), spec.to_string());
        self
    }

    /// `forced_splits_filename` (ADV-03) — the forced-splits JSON file
    /// (config.h default `""`; aliases `fs`/`forced_splits`).
    pub fn forced_splits_filename(mut self, path: &str) -> Self {
        self.params
            .insert("forcedsplits_filename".into(), path.to_string());
        self
    }

    /// `extra_trees` (ADV-04) — randomized-threshold split selection
    /// (config.h default false).
    pub fn extra_trees(mut self, on: bool) -> Self {
        self.params.insert("extra_trees".into(), on.to_string());
        self
    }

    /// `extra_seed` (ADV-04) — the extra-trees per-feature RNG seed
    /// (config.h default 6).
    pub fn extra_seed(mut self, seed: i32) -> Self {
        self.params.insert("extra_seed".into(), seed.to_string());
        self
    }

    /// `cegb_tradeoff` (ADV-05) — the CEGB cost/gain tradeoff (config.h default
    /// 1.0, `>= 0`).
    pub fn cegb_tradeoff(mut self, v: f64) -> Self {
        self.params.insert("cegb_tradeoff".into(), v.to_string());
        self
    }

    /// `cegb_penalty_split` (ADV-05) — the per-split cost (config.h default 0.0,
    /// `>= 0`).
    pub fn cegb_penalty_split(mut self, v: f64) -> Self {
        self.params
            .insert("cegb_penalty_split".into(), v.to_string());
        self
    }

    /// `cegb_penalty_feature_lazy` (ADV-05) — per-feature on-demand cost vector
    /// (config.h default `[]`).
    pub fn cegb_penalty_feature_lazy(mut self, v: &[f64]) -> Self {
        let csv = v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        self.params.insert("cegb_penalty_feature_lazy".into(), csv);
        self
    }

    /// `cegb_penalty_feature_coupled` (ADV-05) — per-feature coupled cost vector
    /// (config.h default `[]`).
    pub fn cegb_penalty_feature_coupled(mut self, v: &[f64]) -> Self {
        let csv = v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        self.params
            .insert("cegb_penalty_feature_coupled".into(), csv);
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

    /// `feature_fraction` (per-tree feature sub-sampling rate, `(0, 1]`; the
    /// `colsample_bytree` alias). For Random Forest (`boosting=rf`) this is the
    /// alternative mandatory-randomization source to bagging (`rf.hpp:37`): RF
    /// requires either active bagging OR `0 < feature_fraction < 1`.
    pub fn feature_fraction(mut self, v: f64) -> Self {
        self.params.insert("feature_fraction".into(), v.to_string());
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

    /// `drop_rate` (DART — the per-tree drop probability scale; config.h default 0.1,
    /// CHECK `[0,1]`; BST-05). Routes into `lgbm-core::Config.drop_rate` via
    /// `Config::from_params` (set.rs:187-189).
    pub fn drop_rate(mut self, v: f64) -> Self {
        self.params.insert("drop_rate".into(), v.to_string());
        self
    }

    /// `max_drop` (DART — the cap on dropped trees per iteration; config.h default 50;
    /// `<= 0` disables the cap; BST-05). Routes into `lgbm-core::Config.max_drop`
    /// (set.rs:191).
    pub fn max_drop(mut self, n: i32) -> Self {
        self.params.insert("max_drop".into(), n.to_string());
        self
    }

    /// `skip_drop` (DART — the probability of skipping all drops this iteration;
    /// config.h default 0.5, CHECK `[0,1]`; BST-05). Routes into
    /// `lgbm-core::Config.skip_drop` (set.rs:193-195).
    pub fn skip_drop(mut self, v: f64) -> Self {
        self.params.insert("skip_drop".into(), v.to_string());
        self
    }

    /// `uniform_drop` (DART — drop each tree with a uniform `drop_rate` instead of a
    /// weight-scaled probability, and disable `tree_weight_` bookkeeping; config.h
    /// default false; BST-05). Routes into `lgbm-core::Config.uniform_drop`
    /// (set.rs:198).
    pub fn uniform_drop(mut self, on: bool) -> Self {
        self.params.insert("uniform_drop".into(), on.to_string());
        self
    }

    /// `xgboost_dart_mode` (DART — select the xgboost normalize branch; config.h
    /// default false; BST-05). Routes into `lgbm-core::Config.xgboost_dart_mode`
    /// (set.rs:197).
    pub fn xgboost_dart_mode(mut self, on: bool) -> Self {
        self.params.insert("xgboost_dart_mode".into(), on.to_string());
        self
    }

    /// `drop_seed` (DART — the seed for the single advancing drop RNG; config.h
    /// default 4; BST-05). Routes into `lgbm-core::Config.drop_seed` (set.rs:199).
    pub fn drop_seed(mut self, seed: i32) -> Self {
        self.params.insert("drop_seed".into(), seed.to_string());
        self
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

    /// `bagging_by_query` (BST-03 / 07-09) — when `true`, the bagging draw's minimal
    /// unit is a whole QUERY (un-deferred in 07-09 alongside the ranking objectives).
    /// Note the C++ `CheckParamConflict` forces this OFF unless
    /// `data_sample_strategy == "bagging"` (set.rs:490-492), so pair it with default
    /// bagging + a `bagging_freq`/`bagging_fraction`.
    pub fn bagging_by_query(mut self, on: bool) -> Self {
        self.params.insert("bagging_by_query".into(), on.to_string());
        self
    }

    /// `objective_seed` (OBJ-06 / rank_xendcg) — the seed base for the per-query
    /// gamma RNG (`Random(objective_seed + q)`); config.h default 5. Routes into
    /// `lgbm-core::Config.objective_seed` via `Config::from_params` (set.rs:300).
    pub fn objective_seed(mut self, seed: i32) -> Self {
        self.params.insert("objective_seed".into(), seed.to_string());
        self
    }

    /// `eval_at` (MET-04 / ndcg/map) — the `@k` cutoffs as a comma-separated list,
    /// e.g. `"1,3,5"` (aliases ndcg_eval_at/ndcg_at/map_eval_at/map_at; config.h
    /// default empty → DefaultEvalAt `[1..=5]`). Routes into
    /// `lgbm-core::Config.eval_at`.
    pub fn eval_at(mut self, ks: &[i32]) -> Self {
        let csv = ks.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");
        self.params.insert("eval_at".into(), csv);
        self
    }

    /// `label_gain` (OBJ-06 / MET-04) — the per-label gain table as a comma-separated
    /// list, e.g. `"0,1,3,7"` (config.h default empty → DefaultLabelGain `2^i - 1`).
    /// Routes into `lgbm-core::Config.label_gain`.
    pub fn label_gain(mut self, gains: &[f64]) -> Self {
        let csv = gains
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.params.insert("label_gain".into(), csv);
        self
    }

    /// `lambdarank_truncation_level` (OBJ-06 / lambdarank) — the max pair rank for
    /// the lambda accumulation + the ideal-DCG truncation (config.h default 30,
    /// `> 0`). Routes into `lgbm-core::Config.lambdarank_truncation_level`.
    pub fn lambdarank_truncation_level(mut self, n: i32) -> Self {
        self.params
            .insert("lambdarank_truncation_level".into(), n.to_string());
        self
    }

    /// `lambdarank_norm` (OBJ-06 / lambdarank) — normalize the lambdas by the score
    /// distance + the `log2(1 + sum_lambdas)` factor (config.h default true). Routes
    /// into `lgbm-core::Config.lambdarank_norm`.
    pub fn lambdarank_norm(mut self, on: bool) -> Self {
        self.params.insert("lambdarank_norm".into(), on.to_string());
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
    fn categorical_setters_route_into_config() {
        // TRL-06: cat_l2/cat_smooth/min_data_per_group/max_cat_threshold/
        // max_cat_to_onehot must round-trip into Config.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .cat_l2(7.5)
            .cat_smooth(3.0)
            .min_data_per_group(50)
            .max_cat_threshold(16)
            .max_cat_to_onehot(2)
            .build()
            .unwrap();
        assert!((cfg.cat_l2 - 7.5).abs() < 1e-12);
        assert!((cfg.cat_smooth - 3.0).abs() < 1e-12);
        assert_eq!(cfg.min_data_per_group, 50);
        assert_eq!(cfg.max_cat_threshold, 16);
        assert_eq!(cfg.max_cat_to_onehot, 2);
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
    fn dart_setters_route_into_config() {
        // BST-05: drop_rate/max_drop/skip_drop/uniform_drop/xgboost_dart_mode/drop_seed
        // + boosting=dart must round-trip into Config.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .boosting("dart")
            .drop_rate(0.3)
            .max_drop(2)
            .skip_drop(0.25)
            .uniform_drop(true)
            .xgboost_dart_mode(true)
            .drop_seed(17)
            .build()
            .unwrap();
        assert_eq!(cfg.boosting, "dart", "boosting=dart must round-trip");
        assert!((cfg.drop_rate - 0.3).abs() < 1e-12);
        assert_eq!(cfg.max_drop, 2);
        assert!((cfg.skip_drop - 0.25).abs() < 1e-12);
        assert!(cfg.uniform_drop);
        assert!(cfg.xgboost_dart_mode);
        assert_eq!(cfg.drop_seed, 17);
    }

    #[test]
    fn rf_setters_route_into_config() {
        // BST-06: boosting=rf + the mandatory-randomization setters must round-trip.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .boosting("rf")
            .bagging_fraction(0.7)
            .bagging_freq(1)
            .feature_fraction(0.8)
            .build()
            .unwrap();
        assert_eq!(cfg.boosting, "rf", "boosting=rf must round-trip");
        assert!((cfg.bagging_fraction - 0.7).abs() < 1e-12);
        assert_eq!(cfg.bagging_freq, 1);
        assert!((cfg.feature_fraction - 0.8).abs() < 1e-12);
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
    fn predict_mode_setters_route_into_config() {
        // PRD-04 / PRD-05: predict_contrib + pred_early_stop/_freq/_margin must be
        // drivable end-to-end through the builder.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .predict_contrib(true)
            .pred_early_stop(true)
            .pred_early_stop_freq(3)
            .pred_early_stop_margin(2.5)
            .build()
            .unwrap();
        assert!(cfg.predict_contrib, "predict_contrib(true) must round-trip");
        assert!(cfg.pred_early_stop, "pred_early_stop(true) must round-trip");
        assert_eq!(cfg.pred_early_stop_freq, 3);
        assert!((cfg.pred_early_stop_margin - 2.5).abs() < 1e-12);

        // defaults stay config.h values.
        let cfg0 = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .build()
            .unwrap();
        assert!(!cfg0.predict_contrib);
        assert!(!cfg0.pred_early_stop);
        assert_eq!(cfg0.pred_early_stop_freq, 10);
        assert!((cfg0.pred_early_stop_margin - 10.0).abs() < 1e-12);
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
    fn ranking_setters_route_into_config() {
        // OBJ-06 / MET-04 (07-09): objective_seed/eval_at/label_gain/
        // lambdarank_truncation_level/lambdarank_norm/sigmoid + objective=lambdarank
        // must round-trip into Config.
        let cfg = TrainingBuilder::new()
            .objective("lambdarank")
            .num_iterations(5)
            .num_leaves(4)
            .objective_seed(7)
            .eval_at(&[1, 3, 5])
            .label_gain(&[0.0, 1.0, 3.0, 7.0])
            .lambdarank_truncation_level(20)
            .lambdarank_norm(false)
            .sigmoid(2.0)
            .build()
            .unwrap();
        assert_eq!(cfg.objective, "lambdarank");
        assert_eq!(cfg.objective_seed, 7);
        assert_eq!(cfg.eval_at, vec![1, 3, 5]);
        assert_eq!(cfg.label_gain, vec![0.0, 1.0, 3.0, 7.0]);
        assert_eq!(cfg.lambdarank_truncation_level, 20);
        assert!(!cfg.lambdarank_norm);
        assert!((cfg.sigmoid - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rank_xendcg_and_metrics_route_into_config() {
        let cfg = TrainingBuilder::new()
            .objective("rank_xendcg")
            .metric("ndcg,map")
            .num_iterations(5)
            .num_leaves(4)
            .objective_seed(5)
            .eval_at(&[1, 2])
            .build()
            .unwrap();
        assert_eq!(cfg.objective, "rank_xendcg");
        assert_eq!(cfg.eval_at, vec![1, 2]);
    }

    #[test]
    fn bagging_by_query_setter_routes_into_config() {
        // 07-09: bagging_by_query=true round-trips when data_sample_strategy=bagging
        // (the default); the C++ CheckParamConflict keeps it on in that case.
        let cfg = TrainingBuilder::new()
            .objective("lambdarank")
            .num_iterations(5)
            .num_leaves(4)
            .bagging_fraction(0.7)
            .bagging_freq(1)
            .bagging_by_query(true)
            .build()
            .unwrap();
        assert!(cfg.bagging_by_query, "bagging_by_query=true must round-trip");
    }

    #[test]
    fn adv_constraint_setters_route_into_config() {
        // W10 ADV-01..05 builder setters all round-trip into Config.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(5)
            .num_leaves(4)
            .monotone_constraints(&[1, -1, 0])
            .monotone_constraints_method("intermediate")
            .monotone_penalty(5.0)
            .interaction_constraints("[0,1],[2]")
            .forced_splits_filename("forced.json")
            .extra_trees(true)
            .extra_seed(9)
            .cegb_tradeoff(0.5)
            .cegb_penalty_split(0.1)
            .cegb_penalty_feature_coupled(&[1.0, 2.0, 3.0])
            .cegb_penalty_feature_lazy(&[0.5, 0.5, 0.5])
            .build()
            .unwrap();
        assert_eq!(cfg.monotone_constraints, vec![1, -1, 0]);
        assert_eq!(cfg.monotone_constraints_method, "intermediate");
        assert_eq!(cfg.monotone_penalty, 5.0);
        assert_eq!(cfg.interaction_constraints, "[0,1],[2]");
        assert_eq!(cfg.forcedsplits_filename, "forced.json");
        assert!(cfg.extra_trees);
        assert_eq!(cfg.extra_seed, 9);
        assert_eq!(cfg.cegb_tradeoff, 0.5);
        assert_eq!(cfg.cegb_penalty_split, 0.1);
        assert_eq!(cfg.cegb_penalty_feature_coupled, vec![1.0, 2.0, 3.0]);
        assert_eq!(cfg.cegb_penalty_feature_lazy, vec![0.5, 0.5, 0.5]);
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
