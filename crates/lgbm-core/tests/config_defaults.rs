//! CFG-01: `Config::default()` must reproduce the C++ `config.h` default values
//! 1:1 for the in-scope hyperparameters.
//!
//! Every assertion below is grounded in a `config.h` member initializer (or
//! `config_auto.cpp` member default). A drift in any default silently breaks
//! parity with the C++ reference, so this test pins the representative set
//! called out in the plan plus a broader spread.

use lgbm_core::Config;

#[test]
fn representative_defaults_match_cpp() {
    let c = Config::default();
    // The plan's named representative set.
    assert_eq!(c.num_iterations, 100);
    assert_eq!(c.learning_rate, 0.1);
    assert_eq!(c.num_leaves, 31);
    assert_eq!(c.max_depth, -1);
    assert_eq!(c.min_data_in_leaf, 20);
    assert_eq!(c.min_sum_hessian_in_leaf, 1e-3);
    assert_eq!(c.bagging_fraction, 1.0);
    assert_eq!(c.feature_fraction, 1.0);
    assert_eq!(c.lambda_l1, 0.0);
    assert_eq!(c.lambda_l2, 0.0);
    assert_eq!(c.max_bin, 255);
    assert_eq!(c.seed, 0);
}

#[test]
fn feature_contri_default_is_empty() {
    // G5-1: `feature_contri` (config.h:539) default `[]` — an empty vector means
    // every feature's penalty is implicitly 1.0 (no-op) at the consumer.
    let c = Config::default();
    assert_eq!(c.feature_contri, Vec::<f64>::new());
}

#[test]
fn string_defaults_match_cpp() {
    let c = Config::default();
    assert_eq!(c.objective, "regression");
    assert_eq!(c.boosting, "gbdt");
    assert_eq!(c.data_sample_strategy, "bagging");
    assert_eq!(c.tree_learner, "serial");
    assert_eq!(c.device_type, "cpu");
    assert_eq!(c.task, "train");
    assert_eq!(c.monotone_constraints_method, "basic");
    assert_eq!(c.output_model, "LightGBM_model.txt");
    assert_eq!(c.convert_model, "gbdt_prediction.cpp");
    assert_eq!(c.output_result, "LightGBM_predict_result.txt");
    assert_eq!(c.data, "");
}

#[test]
fn bool_defaults_match_cpp() {
    let c = Config::default();
    assert!(!c.deterministic);
    assert!(!c.force_col_wise);
    assert!(!c.force_row_wise);
    assert!(c.is_enable_sparse);
    assert!(c.enable_bundle);
    assert!(c.use_missing);
    assert!(!c.zero_as_missing);
    assert!(c.feature_pre_filter);
    assert!(c.boost_from_average);
    assert!(c.lambdarank_norm);
}

#[test]
fn numeric_spread_defaults_match_cpp() {
    let c = Config::default();
    assert_eq!(c.num_threads, 0);
    assert_eq!(c.histogram_pool_size, -1.0);
    assert_eq!(c.bagging_freq, 0);
    assert_eq!(c.pos_bagging_fraction, 1.0);
    assert_eq!(c.neg_bagging_fraction, 1.0);
    assert_eq!(c.feature_fraction_bynode, 1.0);
    assert_eq!(c.min_gain_to_split, 0.0);
    assert_eq!(c.drop_rate, 0.1);
    assert_eq!(c.max_drop, 50);
    assert_eq!(c.skip_drop, 0.5);
    assert_eq!(c.top_rate, 0.2);
    assert_eq!(c.other_rate, 0.1);
    assert_eq!(c.min_data_per_group, 100);
    assert_eq!(c.max_cat_threshold, 32);
    assert_eq!(c.cat_l2, 10.0);
    assert_eq!(c.cat_smooth, 10.0);
    assert_eq!(c.max_cat_to_onehot, 4);
    assert_eq!(c.top_k, 20);
    assert_eq!(c.refit_decay_rate, 0.9);
    assert_eq!(c.cegb_tradeoff, 1.0);
    assert_eq!(c.path_smooth, 0.0);
    assert_eq!(c.verbosity, 1);
    assert_eq!(c.min_data_in_bin, 3);
    assert_eq!(c.bin_construct_sample_cnt, 200000);
    assert_eq!(c.num_class, 1);
    assert_eq!(c.scale_pos_weight, 1.0);
    assert_eq!(c.sigmoid, 1.0);
    assert_eq!(c.alpha, 0.9);
    assert_eq!(c.fair_c, 1.0);
    assert_eq!(c.poisson_max_delta_step, 0.7);
    assert_eq!(c.tweedie_variance_power, 1.5);
    assert_eq!(c.lambdarank_truncation_level, 30);
    assert_eq!(c.metric_freq, 1);
    assert_eq!(c.multi_error_top_k, 1);
    assert_eq!(c.snapshot_freq, -1);
    assert_eq!(c.num_iteration_predict, -1);
    assert_eq!(c.pred_early_stop_freq, 10);
    assert_eq!(c.pred_early_stop_margin, 10.0);
}

#[test]
fn pred_early_stop_defaults_and_resolution() {
    // config.h defaults: pred_early_stop=false, _freq=10, _margin=10.0.
    let c = Config::default();
    assert!(!c.pred_early_stop);
    assert_eq!(c.pred_early_stop_freq, 10);
    assert_eq!(c.pred_early_stop_margin, 10.0);

    // And they resolve from raw params (PRD-05 builder/CLI drivability).
    let mut params = std::collections::HashMap::new();
    params.insert("pred_early_stop".to_string(), "true".to_string());
    params.insert("pred_early_stop_freq".to_string(), "1".to_string());
    params.insert("pred_early_stop_margin".to_string(), "2.0".to_string());
    let c = Config::from_params(&params).expect("resolve pred_early_stop params");
    assert!(c.pred_early_stop);
    assert_eq!(c.pred_early_stop_freq, 1);
    assert_eq!(c.pred_early_stop_margin, 2.0);
}

#[test]
fn derived_seed_config_h_defaults() {
    // config_auto.cpp / config.h member defaults BEFORE seed derivation.
    let c = Config::default();
    assert_eq!(c.data_random_seed, 1);
    assert_eq!(c.bagging_seed, 3);
    assert_eq!(c.drop_seed, 4);
    assert_eq!(c.feature_fraction_seed, 2);
    assert_eq!(c.objective_seed, 5);
    assert_eq!(c.extra_seed, 6);
}
