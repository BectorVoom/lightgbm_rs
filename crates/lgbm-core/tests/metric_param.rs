//! The `metric` parameter pipeline: `GetMetricType` → `ParseMetrics` →
//! `ParseMetricAlias` (config.cpp:130-162, config.h:1290-1318), plus the
//! `auc_mu_weights` matrix derivation (`Config::GetAucMuWeights`,
//! config.cpp:218-247), the `eval_at` sort (config.cpp:287) and the
//! objective-vs-metric multiclass conflict check (config.cpp:328-339).
//!
//! Every expectation below is grounded in the read-only C++ reference; a drift
//! here means the Rust port evaluates a different metric set than upstream for
//! the same params, which no downstream parity test would catch on its own.

use std::collections::HashMap;

use lgbm_core::Config;

fn cfg(pairs: &[(&str, &str)]) -> Config {
    let params: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Config::from_params(&params).expect("config builds")
}

fn cfg_err(pairs: &[(&str, &str)]) -> String {
    let params: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Config::from_params(&params)
        .expect_err("config must be rejected")
        .to_string()
}

// --- GetMetricType: objective fallback --------------------------------------

#[test]
fn absent_metric_falls_back_to_the_objective_name() {
    // config.cpp:158-161 — "add names of objective function if not providing metric".
    assert_eq!(cfg(&[("objective", "regression")]).metric, vec!["l2"]);
    assert_eq!(cfg(&[("objective", "regression_l1")]).metric, vec!["l1"]);
    assert_eq!(cfg(&[("objective", "binary")]).metric, vec!["binary_logloss"]);
    assert_eq!(cfg(&[("objective", "lambdarank")]).metric, vec!["ndcg"]);
    assert_eq!(cfg(&[("objective", "huber")]).metric, vec!["huber"]);
    assert_eq!(cfg(&[("objective", "quantile")]).metric, vec!["quantile"]);
    assert_eq!(
        cfg(&[("objective", "multiclass"), ("num_class", "3")]).metric,
        vec!["multi_logloss"]
    );
    // The default objective (`regression`) also yields `l2` with no params at all.
    assert_eq!(Config::from_params(&HashMap::new()).unwrap().metric, vec!["l2"]);
}

#[test]
fn empty_metric_value_is_treated_as_absent() {
    // C++ `GetString` requires `count > 0 && !empty()`, so `metric=` leaves the
    // list empty AND `value.size() == 0`, which triggers the objective fallback.
    assert_eq!(cfg(&[("objective", "binary"), ("metric", "")]).metric, vec!["binary_logloss"]);
}

// --- ParseMetrics: split, alias, dedup --------------------------------------

#[test]
fn explicit_metric_list_is_split_aliased_and_deduped_in_order() {
    assert_eq!(cfg(&[("metric", "l2,rmse")]).metric, vec!["l2", "rmse"]);
    // Order-preserving dedup on the CANONICAL name: `mse` and `l2` collapse.
    assert_eq!(cfg(&[("metric", "rmse,mse,l2,l2_root")]).metric, vec!["rmse", "l2"]);
    // `Common::Split` drops empty segments (common.h:102-122).
    assert_eq!(cfg(&[("metric", "l2,,rmse")]).metric, vec!["l2", "rmse"]);
    // Uppercase input is lowercased before ParseMetrics (config.cpp:155).
    assert_eq!(cfg(&[("metric", "L2,RMSE")]).metric, vec!["l2", "rmse"]);
}

#[test]
fn metric_of_only_separators_stays_empty_without_objective_fallback() {
    // `Common::Split(",", ',')` returns an EMPTY vector, but `value.size() != 0`,
    // so the `metric->empty() && value.size() == 0` fallback does NOT fire.
    assert!(cfg(&[("objective", "binary"), ("metric", ",")]).metric.is_empty());
}

#[test]
fn metric_none_aliases_canonicalize_to_custom() {
    // config.h:1314-1315 — none/null/custom/na all mean "no built-in metric".
    for v in ["none", "None", "null", "na", "custom", "NA"] {
        assert_eq!(cfg(&[("metric", v)]).metric, vec!["custom"], "metric={v}");
    }
}

#[test]
fn metric_aliases_cover_every_parse_metric_alias_arm() {
    // config.h:1290-1318, one case per returned canonical name.
    let cases: &[(&str, &str)] = &[
        ("regression", "l2"),
        ("regression_l2", "l2"),
        ("mean_squared_error", "l2"),
        ("mse", "l2"),
        ("l2_root", "rmse"),
        ("root_mean_squared_error", "rmse"),
        ("regression_l1", "l1"),
        ("mean_absolute_error", "l1"),
        ("mae", "l1"),
        ("binary", "binary_logloss"),
        ("lambdarank", "ndcg"),
        ("rank_xendcg", "ndcg"),
        ("xendcg", "ndcg"),
        ("xe_ndcg", "ndcg"),
        ("xe_ndcg_mart", "ndcg"),
        ("xendcg_mart", "ndcg"),
        ("mean_average_precision", "map"),
        ("xentropy", "cross_entropy"),
        ("xentlambda", "cross_entropy_lambda"),
        ("kldiv", "kullback_leibler"),
        ("mean_absolute_percentage_error", "mape"),
    ];
    for (input, canonical) in cases {
        assert_eq!(cfg(&[("metric", input)]).metric, vec![*canonical], "metric={input}");
    }
    // The multiclass alias arm needs a matching objective to pass the conflict check.
    for input in ["multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"] {
        let c = cfg(&[("objective", "multiclass"), ("num_class", "3"), ("metric", input)]);
        assert_eq!(c.metric, vec!["multi_logloss"], "metric={input}");
    }
}

#[test]
fn unrecognized_metric_names_pass_through_unchanged() {
    // ParseMetricAlias returns `type` for anything it does not know; the metric
    // FACTORY is where an unknown name becomes an error, not the config layer.
    assert_eq!(cfg(&[("metric", "auc")]).metric, vec!["auc"]);
    assert_eq!(cfg(&[("metric", "average_precision")]).metric, vec!["average_precision"]);
    assert_eq!(cfg(&[("metric", "gamma_deviance")]).metric, vec!["gamma_deviance"]);
    assert_eq!(cfg(&[("metric", "not_a_metric")]).metric, vec!["not_a_metric"]);
}

#[test]
fn metrics_and_metric_types_aliases_route_to_metric() {
    assert_eq!(cfg(&[("metrics", "rmse")]).metric, vec!["rmse"]);
    assert_eq!(cfg(&[("metric_types", "rmse")]).metric, vec!["rmse"]);
}

// --- CheckParamConflict: objective vs metric multiclass ---------------------

#[test]
fn multiclass_objective_with_binary_metric_is_rejected() {
    let err = cfg_err(&[("objective", "multiclass"), ("num_class", "3"), ("metric", "l2")]);
    assert!(err.contains("don't match"), "unexpected error: {err}");
}

#[test]
fn non_multiclass_objective_with_multiclass_metric_is_rejected() {
    for m in ["multi_logloss", "multi_error", "auc_mu"] {
        let err = cfg_err(&[("objective", "binary"), ("metric", m)]);
        assert!(err.contains("don't match"), "metric={m}: unexpected error: {err}");
    }
}

#[test]
fn multiclass_objective_with_multiclass_metrics_is_accepted() {
    let c = cfg(&[
        ("objective", "multiclass"),
        ("num_class", "4"),
        ("metric", "multi_logloss,multi_error,auc_mu"),
    ]);
    assert_eq!(c.metric, vec!["multi_logloss", "multi_error", "auc_mu"]);
}

// --- GetAucMuWeights --------------------------------------------------------

#[test]
fn default_auc_mu_weights_matrix_is_ones_off_diagonal() {
    let c = cfg(&[("objective", "multiclass"), ("num_class", "3"), ("metric", "auc_mu")]);
    assert!(c.auc_mu_weights.is_empty());
    assert_eq!(
        c.auc_mu_weights_matrix,
        vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]
    );
}

#[test]
fn explicit_auc_mu_weights_populate_the_matrix_with_a_forced_zero_diagonal() {
    // C++ overwrites a non-zero diagonal with 0 (and only logs about it).
    let c = cfg(&[
        ("objective", "multiclass"),
        ("num_class", "2"),
        ("metric", "auc_mu"),
        ("auc_mu_weights", "9,2,3,7"),
    ]);
    assert_eq!(c.auc_mu_weights, vec![9.0, 2.0, 3.0, 7.0]);
    assert_eq!(c.auc_mu_weights_matrix, vec![vec![0.0, 2.0], vec![3.0, 0.0]]);
}

#[test]
fn auc_mu_weights_with_wrong_length_is_rejected() {
    let err = cfg_err(&[
        ("objective", "multiclass"),
        ("num_class", "3"),
        ("metric", "auc_mu"),
        ("auc_mu_weights", "1,1,1"),
    ]);
    assert!(err.contains("auc_mu_weights"), "unexpected error: {err}");
}

#[test]
fn auc_mu_weights_with_a_zero_off_diagonal_is_rejected() {
    let err = cfg_err(&[
        ("objective", "multiclass"),
        ("num_class", "2"),
        ("metric", "auc_mu"),
        ("auc_mu_weights", "0,0,1,0"),
    ]);
    assert!(err.contains("auc_mu_weights"), "unexpected error: {err}");
}

// --- eval_at sort -----------------------------------------------------------

#[test]
fn eval_at_is_sorted_ascending() {
    // config.cpp:287 `std::sort(eval_at.begin(), eval_at.end())`.
    let c = cfg(&[("objective", "lambdarank"), ("eval_at", "5,1,3")]);
    assert_eq!(c.eval_at, vec![1, 3, 5]);
}

// --- feature_contri / max_bin_by_feature ------------------------------------

#[test]
fn feature_contri_parses_and_accepts_its_aliases() {
    assert_eq!(cfg(&[("feature_contri", "1.0,0.5,2")]).feature_contri, vec![1.0, 0.5, 2.0]);
    for alias in ["feature_contrib", "fc", "fp", "feature_penalty"] {
        assert_eq!(cfg(&[(alias, "0.25,4")]).feature_contri, vec![0.25, 4.0], "alias={alias}");
    }
    assert!(Config::default().feature_contri.is_empty());
}

#[test]
fn max_bin_by_feature_parses_and_rejects_entries_below_two() {
    assert_eq!(cfg(&[("max_bin_by_feature", "16,32,255")]).max_bin_by_feature, vec![16, 32, 255]);
    // dataset_loader.cpp:616 `CHECK_GT(min_element, 1)`.
    let err = cfg_err(&[("max_bin_by_feature", "16,1")]);
    assert!(err.contains("max_bin_by_feature"), "unexpected error: {err}");
    assert!(Config::default().max_bin_by_feature.is_empty());
}
