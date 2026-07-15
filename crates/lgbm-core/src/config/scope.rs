//! The explicit in-scope single-machine parameter set — the resolution of
//! RESEARCH Open Question 1 (the in-scope ~110 vs the 131 auto-extracted
//! params).
//!
//! [`IN_SCOPE_PARAMS`] enumerates the canonical parameter names that the v1
//! Rust port exposes and validates. It is the full `config_auto.cpp`
//! `parameter_set()` / `GetMembersFromString` canonical member list MINUS the
//! out-of-v1-scope groups documented below. The drift-checker
//! (`oracle-harness/tests/config_drift.rs`) asserts that this list is a
//! superset of the in-scope subset of the C++ source, catching upstream drift.
//!
//! # Out-of-v1-scope exclusions
//!
//! - **Distributed learning** (deferred): `num_machines`, `local_listen_port`,
//!   `time_out`, `machine_list_filename`, `machines`. The v1 port targets
//!   single-machine training only (CLAUDE.md: CPU/ROCm single-node parity);
//!   the C++ `Network` layer / `is_parallel` paths are out of scope.
//! - **GPU/OpenCL device tuning** (deferred): `num_gpu`, `gpu_platform_id`,
//!   `gpu_device_id`, `gpu_use_dp`. The Rust compute backend uses CubeCL (ROCm),
//!   not the C++ OpenCL `gpu` device knobs; these have no Rust analog in v1.
//!
//! (Linear-tree params `linear_tree` / `linear_lambda` are now IN scope — the
//! per-leaf linear leaf fitting is implemented and C++-oracle-verified, see
//! `lgbm_treelearner::linear::fit_linear_leaves`.)

/// Canonical in-scope single-machine parameter names (Open Question 1).
///
/// Every name here corresponds to a canonical member in
/// `config_auto.cpp::GetMembersFromString`. The list deliberately EXCLUDES the
/// distributed / GPU-OpenCL / linear-tree groups (see the
/// module docs). The drift-checker asserts this set covers every in-scope
/// canonical the C++ source declares.
pub const IN_SCOPE_PARAMS: &[&str] = &[
    // --- core / objective ---
    "config",
    "task",
    "objective",
    "boosting",
    "data_sample_strategy",
    "data",
    "valid",
    // --- core training ---
    "num_iterations",
    "learning_rate",
    "num_leaves",
    "tree_learner",
    "num_threads",
    "device_type",
    "seed",
    "deterministic",
    // --- learning control ---
    "force_col_wise",
    "force_row_wise",
    "histogram_pool_size",
    "max_depth",
    "min_data_in_leaf",
    "min_sum_hessian_in_leaf",
    "bagging_fraction",
    "pos_bagging_fraction",
    "neg_bagging_fraction",
    "bagging_freq",
    "bagging_seed",
    "bagging_by_query",
    "feature_fraction",
    "feature_fraction_bynode",
    "feature_fraction_seed",
    "extra_trees",
    "extra_seed",
    "early_stopping_round",
    "early_stopping_min_delta",
    "first_metric_only",
    "max_delta_step",
    "lambda_l1",
    "lambda_l2",
    "min_gain_to_split",
    "drop_rate",
    "max_drop",
    "skip_drop",
    "xgboost_dart_mode",
    "uniform_drop",
    "drop_seed",
    "top_rate",
    "other_rate",
    "min_data_per_group",
    "max_cat_threshold",
    "cat_l2",
    "cat_smooth",
    "max_cat_to_onehot",
    "top_k",
    "monotone_constraints",
    "monotone_constraints_method",
    "monotone_penalty",
    "linear_tree",
    "linear_lambda",
    "use_quantized_grad",
    "num_grad_quant_bins",
    "quant_train_renew_leaf",
    "stochastic_rounding",
    "feature_contri",
    "forcedsplits_filename",
    "refit_decay_rate",
    "cegb_tradeoff",
    "cegb_penalty_split",
    "cegb_penalty_feature_lazy",
    "cegb_penalty_feature_coupled",
    "path_smooth",
    "interaction_constraints",
    "verbosity",
    // --- IO config ---
    "input_model",
    "output_model",
    "saved_feature_importance_type",
    "snapshot_freq",
    "max_bin",
    "max_bin_by_feature",
    "min_data_in_bin",
    "bin_construct_sample_cnt",
    "data_random_seed",
    "is_enable_sparse",
    "enable_bundle",
    "use_missing",
    "zero_as_missing",
    "feature_pre_filter",
    "pre_partition",
    "two_round",
    "header",
    "label_column",
    "weight_column",
    "group_column",
    "ignore_column",
    "categorical_feature",
    "forcedbins_filename",
    "save_binary",
    "precise_float_parser",
    "parser_config_file",
    // --- predict config ---
    "start_iteration_predict",
    "num_iteration_predict",
    "predict_raw_score",
    "predict_leaf_index",
    "predict_contrib",
    "predict_disable_shape_check",
    "pred_early_stop",
    "pred_early_stop_freq",
    "pred_early_stop_margin",
    "output_result",
    // --- convert config ---
    "convert_model_language",
    "convert_model",
    // --- objective config ---
    "objective_seed",
    "num_class",
    "is_unbalance",
    "scale_pos_weight",
    "sigmoid",
    "boost_from_average",
    "reg_sqrt",
    "alpha",
    "fair_c",
    "poisson_max_delta_step",
    "tweedie_variance_power",
    "lambdarank_truncation_level",
    "lambdarank_norm",
    "label_gain",
    "lambdarank_position_bias_regularization",
    // --- metric config ---
    "metric",
    "metric_freq",
    "is_provide_training_metric",
    "eval_at",
    "multi_error_top_k",
    "auc_mu_weights",
];

/// Canonical parameter names explicitly OUT of v1 scope (documented for the
/// drift-checker so it skips — rather than fails on — these).
///
/// Grouped exactly as in the module docs: distributed, GPU-OpenCL.
pub const OUT_OF_SCOPE_PARAMS: &[&str] = &[
    // distributed
    "num_machines",
    "local_listen_port",
    "time_out",
    "machine_list_filename",
    "machines",
    // GPU / OpenCL
    "num_gpu",
    "gpu_platform_id",
    "gpu_device_id",
    "gpu_use_dp",
];
