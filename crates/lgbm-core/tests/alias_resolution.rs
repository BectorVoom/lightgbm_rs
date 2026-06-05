//! CFG-02: parameter aliases resolve to canonical names, matching the C++
//! `config_auto.cpp::alias_table()` verbatim.

use lgbm_core::config::resolve_alias;

#[test]
fn plan_named_aliases_resolve() {
    assert_eq!(resolve_alias("n_estimators"), "num_iterations");
    assert_eq!(resolve_alias("eta"), "learning_rate");
    assert_eq!(resolve_alias("num_leaf"), "num_leaves");
    assert_eq!(resolve_alias("num_thread"), "num_threads");
}

#[test]
fn canonical_resolves_to_itself() {
    assert_eq!(resolve_alias("num_iterations"), "num_iterations");
    assert_eq!(resolve_alias("learning_rate"), "learning_rate");
    assert_eq!(resolve_alias("num_leaves"), "num_leaves");
    assert_eq!(resolve_alias("max_bin"), "max_bin");
}

#[test]
fn unknown_param_passes_through() {
    // C++ behavior: unknown params warn, never error; resolve to themselves.
    assert_eq!(resolve_alias("totally_made_up_param"), "totally_made_up_param");
    assert_eq!(resolve_alias(""), "");
}

#[test]
fn broad_alias_spread_resolves() {
    // A spread across many canonical targets, mirroring alias_table().
    assert_eq!(resolve_alias("num_iteration"), "num_iterations");
    assert_eq!(resolve_alias("num_boost_round"), "num_iterations");
    assert_eq!(resolve_alias("nrounds"), "num_iterations");
    assert_eq!(resolve_alias("shrinkage_rate"), "learning_rate");
    assert_eq!(resolve_alias("max_leaves"), "num_leaves");
    assert_eq!(resolve_alias("max_leaf"), "num_leaves");
    assert_eq!(resolve_alias("n_jobs"), "num_threads");
    assert_eq!(resolve_alias("device"), "device_type");
    assert_eq!(resolve_alias("random_seed"), "seed");
    assert_eq!(resolve_alias("random_state"), "seed");
    assert_eq!(resolve_alias("min_data"), "min_data_in_leaf");
    assert_eq!(resolve_alias("min_child_samples"), "min_data_in_leaf");
    assert_eq!(resolve_alias("min_child_weight"), "min_sum_hessian_in_leaf");
    assert_eq!(resolve_alias("subsample"), "bagging_fraction");
    assert_eq!(resolve_alias("colsample_bytree"), "feature_fraction");
    assert_eq!(resolve_alias("reg_alpha"), "lambda_l1");
    assert_eq!(resolve_alias("reg_lambda"), "lambda_l2");
    assert_eq!(resolve_alias("lambda"), "lambda_l2");
    assert_eq!(resolve_alias("min_split_gain"), "min_gain_to_split");
    assert_eq!(resolve_alias("max_bins"), "max_bin");
    assert_eq!(resolve_alias("verbose"), "verbosity");
    assert_eq!(resolve_alias("num_classes"), "num_class");
    assert_eq!(resolve_alias("rate_drop"), "drop_rate");
    assert_eq!(resolve_alias("topk"), "top_k");
}
