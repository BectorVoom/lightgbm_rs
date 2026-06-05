//! The `from_params` pipeline — the Rust port of the C++ `Config::Set`
//! (`config.cpp` lines 257-308) + `GetMembersFromString` CHECK constraints
//! (`config_auto.cpp`) + `CheckParamConflict` mutations (`config.cpp`
//! lines 314-474).
//!
//! Four stages, in the exact C++ order:
//!
//! 1. **Alias resolution** — incoming keys are mapped to canonical names via
//!    [`resolve_alias`]; unknown keys pass through with no error (matches the
//!    C++ "Unknown parameter" warning / no-op).
//! 2. **Seed derivation** — if `seed` is provided, the six sub-seeds derive from
//!    a fresh [`Random`] in the EXACT C++ order (Pitfall 4 — a wrong order
//!    silently corrupts every later sampling phase).
//! 3. **Member extraction + CHECK validation** — each in-scope param is parsed
//!    to its typed value; parse failures, unknown enum strings, and `CHECK_*`
//!    range violations become typed [`ConfigError`]s. NO panic on hostile input
//!    (Security V5 / T-1-01).
//! 4. **Conflict mutations** — the in-scope `CheckParamConflict` side-effects
//!    are applied in place (`max_depth`→`num_leaves`, `path_smooth`→
//!    `min_data_in_leaf`, etc.); multiclass mismatches become `Err`.
//!
//! All numeric/enum parsing mirrors the C++ coercion: bool accepts
//! `true/+/false/-` (case-insensitive); enum strings are lowercased before
//! matching, exactly as `GetBoostingType`/`GetObjectiveType`/etc. do.

use std::collections::HashMap;

use crate::config::alias::resolve_alias;
use crate::config::Config;
use crate::error::ConfigError;
use crate::random::Random;
use crate::types::K_EPSILON;

impl Config {
    /// Build a [`Config`] from a parameter map, reproducing the C++
    /// `Config::Set` pipeline (alias resolve → seed derive → member extract +
    /// CHECK validation → conflict mutations).
    ///
    /// Returns a typed [`ConfigError`] on any invalid input; never panics.
    pub fn from_params(params: &HashMap<String, String>) -> Result<Config, ConfigError> {
        from_params(params)
    }
}

/// Free-function form of [`Config::from_params`].
pub fn from_params(params: &HashMap<String, String>) -> Result<Config, ConfigError> {
    // --- Stage 1: alias resolution -------------------------------------------
    // Map every incoming key to its canonical name. Unknown keys pass through
    // (no error). If two aliases collide onto the same canonical, last-writer
    // wins here — the C++ KeyAliasTransform emits a warning and keeps one; for
    // parity of the *resolved value* we keep a single entry per canonical.
    let mut resolved: HashMap<String, String> = HashMap::with_capacity(params.len());
    for (key, value) in params {
        let canonical = resolve_alias(key).to_string();
        resolved.insert(canonical, value.clone());
    }

    let mut cfg = Config::default();

    // --- Stage 2: seed derivation (EXACT C++ order) --------------------------
    // config.cpp lines 259-268. A fresh Random(seed); int_max = i16::MAX.
    if let Some(seed_str) = resolved.get("seed") {
        let seed = parse_int("seed", seed_str)?;
        cfg.seed = seed;
        let mut rand = Random::new(seed);
        let int_max = i16::MAX as i32; // 32767
        cfg.data_random_seed = rand.next_short(0, int_max);
        cfg.bagging_seed = rand.next_short(0, int_max);
        cfg.drop_seed = rand.next_short(0, int_max);
        cfg.feature_fraction_seed = rand.next_short(0, int_max);
        cfg.objective_seed = rand.next_short(0, int_max);
        cfg.extra_seed = rand.next_short(0, int_max);
    }

    // --- Enum-typed parses (config.cpp GetTaskType/GetBoostingType/... ) ------
    // These run BEFORE GetMembersFromString in C++ (config.cpp lines 270-279).
    if let Some(v) = resolved.get("task") {
        cfg.task = parse_task(v)?;
    }
    if let Some(v) = resolved.get("boosting") {
        cfg.boosting = parse_boosting(v)?;
    }
    if let Some(v) = resolved.get("data_sample_strategy") {
        cfg.data_sample_strategy = parse_data_sample_strategy(v)?;
    }
    if let Some(v) = resolved.get("objective") {
        // No closed enum in C++ (ParseObjectiveAlias is a passthrough for
        // unknowns), but the validation tests expect "nonsense" → UnknownValue.
        cfg.objective = parse_objective(v)?;
    }
    if let Some(v) = resolved.get("device_type") {
        cfg.device_type = parse_device_type(v)?;
    }
    if let Some(v) = resolved.get("tree_learner") {
        cfg.tree_learner = parse_tree_learner(v)?;
    }

    // --- Stage 3: member extraction + CHECK validation -----------------------
    // Mirrors config_auto.cpp::GetMembersFromString order + CHECK_* constraints.
    // Only the in-scope (single-machine) members are extracted/validated.

    get_string(&resolved, "data", &mut cfg.data);

    get_int(&resolved, "num_iterations", &mut cfg.num_iterations)?;
    check_ge("num_iterations", cfg.num_iterations, 0)?;

    get_double(&resolved, "learning_rate", &mut cfg.learning_rate)?;
    check_gt_f("learning_rate", cfg.learning_rate, 0.0)?;

    let num_leaves_explicit = get_int(&resolved, "num_leaves", &mut cfg.num_leaves)?;
    check_gt("num_leaves", cfg.num_leaves, 1)?;
    check_le("num_leaves", cfg.num_leaves, 131072)?;

    get_int(&resolved, "num_threads", &mut cfg.num_threads)?;

    get_bool(&resolved, "deterministic", &mut cfg.deterministic)?;
    get_bool(&resolved, "force_col_wise", &mut cfg.force_col_wise)?;
    get_bool(&resolved, "force_row_wise", &mut cfg.force_row_wise)?;

    get_double(&resolved, "histogram_pool_size", &mut cfg.histogram_pool_size)?;

    get_int(&resolved, "max_depth", &mut cfg.max_depth)?;

    get_int(&resolved, "min_data_in_leaf", &mut cfg.min_data_in_leaf)?;
    check_ge("min_data_in_leaf", cfg.min_data_in_leaf, 0)?;

    get_double(&resolved, "min_sum_hessian_in_leaf", &mut cfg.min_sum_hessian_in_leaf)?;
    check_ge_f("min_sum_hessian_in_leaf", cfg.min_sum_hessian_in_leaf, 0.0)?;

    get_double(&resolved, "bagging_fraction", &mut cfg.bagging_fraction)?;
    check_gt_f("bagging_fraction", cfg.bagging_fraction, 0.0)?;
    check_le_f("bagging_fraction", cfg.bagging_fraction, 1.0)?;

    get_double(&resolved, "pos_bagging_fraction", &mut cfg.pos_bagging_fraction)?;
    check_gt_f("pos_bagging_fraction", cfg.pos_bagging_fraction, 0.0)?;
    check_le_f("pos_bagging_fraction", cfg.pos_bagging_fraction, 1.0)?;

    get_double(&resolved, "neg_bagging_fraction", &mut cfg.neg_bagging_fraction)?;
    check_gt_f("neg_bagging_fraction", cfg.neg_bagging_fraction, 0.0)?;
    check_le_f("neg_bagging_fraction", cfg.neg_bagging_fraction, 1.0)?;

    get_int(&resolved, "bagging_freq", &mut cfg.bagging_freq)?;
    get_int(&resolved, "bagging_seed", &mut cfg.bagging_seed)?;
    get_bool(&resolved, "bagging_by_query", &mut cfg.bagging_by_query)?;

    get_double(&resolved, "feature_fraction", &mut cfg.feature_fraction)?;
    check_gt_f("feature_fraction", cfg.feature_fraction, 0.0)?;
    check_le_f("feature_fraction", cfg.feature_fraction, 1.0)?;

    get_double(&resolved, "feature_fraction_bynode", &mut cfg.feature_fraction_bynode)?;
    check_gt_f("feature_fraction_bynode", cfg.feature_fraction_bynode, 0.0)?;
    check_le_f("feature_fraction_bynode", cfg.feature_fraction_bynode, 1.0)?;

    get_int(&resolved, "feature_fraction_seed", &mut cfg.feature_fraction_seed)?;
    get_bool(&resolved, "extra_trees", &mut cfg.extra_trees)?;
    get_int(&resolved, "extra_seed", &mut cfg.extra_seed)?;
    get_int(&resolved, "early_stopping_round", &mut cfg.early_stopping_round)?;

    get_double(&resolved, "early_stopping_min_delta", &mut cfg.early_stopping_min_delta)?;
    check_ge_f("early_stopping_min_delta", cfg.early_stopping_min_delta, 0.0)?;

    get_bool(&resolved, "first_metric_only", &mut cfg.first_metric_only)?;
    get_double(&resolved, "max_delta_step", &mut cfg.max_delta_step)?;

    get_double(&resolved, "lambda_l1", &mut cfg.lambda_l1)?;
    check_ge_f("lambda_l1", cfg.lambda_l1, 0.0)?;

    get_double(&resolved, "lambda_l2", &mut cfg.lambda_l2)?;
    check_ge_f("lambda_l2", cfg.lambda_l2, 0.0)?;

    get_double(&resolved, "min_gain_to_split", &mut cfg.min_gain_to_split)?;
    check_ge_f("min_gain_to_split", cfg.min_gain_to_split, 0.0)?;

    get_double(&resolved, "drop_rate", &mut cfg.drop_rate)?;
    check_ge_f("drop_rate", cfg.drop_rate, 0.0)?;
    check_le_f("drop_rate", cfg.drop_rate, 1.0)?;

    get_int(&resolved, "max_drop", &mut cfg.max_drop)?;

    get_double(&resolved, "skip_drop", &mut cfg.skip_drop)?;
    check_ge_f("skip_drop", cfg.skip_drop, 0.0)?;
    check_le_f("skip_drop", cfg.skip_drop, 1.0)?;

    get_bool(&resolved, "xgboost_dart_mode", &mut cfg.xgboost_dart_mode)?;
    get_bool(&resolved, "uniform_drop", &mut cfg.uniform_drop)?;
    get_int(&resolved, "drop_seed", &mut cfg.drop_seed)?;

    get_double(&resolved, "top_rate", &mut cfg.top_rate)?;
    check_ge_f("top_rate", cfg.top_rate, 0.0)?;
    check_le_f("top_rate", cfg.top_rate, 1.0)?;

    get_double(&resolved, "other_rate", &mut cfg.other_rate)?;
    check_ge_f("other_rate", cfg.other_rate, 0.0)?;
    check_le_f("other_rate", cfg.other_rate, 1.0)?;

    get_int(&resolved, "min_data_per_group", &mut cfg.min_data_per_group)?;
    check_gt("min_data_per_group", cfg.min_data_per_group, 0)?;

    get_int(&resolved, "max_cat_threshold", &mut cfg.max_cat_threshold)?;
    check_gt("max_cat_threshold", cfg.max_cat_threshold, 0)?;

    get_double(&resolved, "cat_l2", &mut cfg.cat_l2)?;
    check_ge_f("cat_l2", cfg.cat_l2, 0.0)?;

    get_double(&resolved, "cat_smooth", &mut cfg.cat_smooth)?;
    check_ge_f("cat_smooth", cfg.cat_smooth, 0.0)?;

    get_int(&resolved, "max_cat_to_onehot", &mut cfg.max_cat_to_onehot)?;
    check_gt("max_cat_to_onehot", cfg.max_cat_to_onehot, 0)?;

    get_int(&resolved, "top_k", &mut cfg.top_k)?;
    check_gt("top_k", cfg.top_k, 0)?;

    get_string(&resolved, "monotone_constraints_method", &mut cfg.monotone_constraints_method);

    get_double(&resolved, "monotone_penalty", &mut cfg.monotone_penalty)?;
    check_ge_f("monotone_penalty", cfg.monotone_penalty, 0.0)?;

    get_string(&resolved, "forcedsplits_filename", &mut cfg.forcedsplits_filename);

    get_double(&resolved, "refit_decay_rate", &mut cfg.refit_decay_rate)?;
    check_ge_f("refit_decay_rate", cfg.refit_decay_rate, 0.0)?;
    check_le_f("refit_decay_rate", cfg.refit_decay_rate, 1.0)?;

    get_double(&resolved, "cegb_tradeoff", &mut cfg.cegb_tradeoff)?;
    check_ge_f("cegb_tradeoff", cfg.cegb_tradeoff, 0.0)?;

    get_double(&resolved, "cegb_penalty_split", &mut cfg.cegb_penalty_split)?;
    check_ge_f("cegb_penalty_split", cfg.cegb_penalty_split, 0.0)?;

    get_double(&resolved, "path_smooth", &mut cfg.path_smooth)?;
    check_ge_f("path_smooth", cfg.path_smooth, 0.0)?;

    get_string(&resolved, "interaction_constraints", &mut cfg.interaction_constraints);
    get_int(&resolved, "verbosity", &mut cfg.verbosity)?;

    get_string(&resolved, "input_model", &mut cfg.input_model);
    get_string(&resolved, "output_model", &mut cfg.output_model);
    get_int(&resolved, "saved_feature_importance_type", &mut cfg.saved_feature_importance_type)?;
    get_int(&resolved, "snapshot_freq", &mut cfg.snapshot_freq)?;

    get_int(&resolved, "max_bin", &mut cfg.max_bin)?;
    check_gt("max_bin", cfg.max_bin, 1)?;

    get_int(&resolved, "min_data_in_bin", &mut cfg.min_data_in_bin)?;
    check_gt("min_data_in_bin", cfg.min_data_in_bin, 0)?;

    get_int(&resolved, "bin_construct_sample_cnt", &mut cfg.bin_construct_sample_cnt)?;
    check_gt("bin_construct_sample_cnt", cfg.bin_construct_sample_cnt, 0)?;

    get_int(&resolved, "data_random_seed", &mut cfg.data_random_seed)?;

    get_bool(&resolved, "is_enable_sparse", &mut cfg.is_enable_sparse)?;
    get_bool(&resolved, "enable_bundle", &mut cfg.enable_bundle)?;
    get_bool(&resolved, "use_missing", &mut cfg.use_missing)?;
    get_bool(&resolved, "zero_as_missing", &mut cfg.zero_as_missing)?;
    get_bool(&resolved, "feature_pre_filter", &mut cfg.feature_pre_filter)?;
    get_bool(&resolved, "pre_partition", &mut cfg.pre_partition)?;
    get_bool(&resolved, "two_round", &mut cfg.two_round)?;
    get_bool(&resolved, "header", &mut cfg.header)?;

    get_string(&resolved, "label_column", &mut cfg.label_column);
    get_string(&resolved, "weight_column", &mut cfg.weight_column);
    get_string(&resolved, "group_column", &mut cfg.group_column);
    get_string(&resolved, "ignore_column", &mut cfg.ignore_column);
    get_string(&resolved, "categorical_feature", &mut cfg.categorical_feature);
    get_string(&resolved, "forcedbins_filename", &mut cfg.forcedbins_filename);

    get_bool(&resolved, "save_binary", &mut cfg.save_binary)?;
    get_bool(&resolved, "precise_float_parser", &mut cfg.precise_float_parser)?;
    get_string(&resolved, "parser_config_file", &mut cfg.parser_config_file);

    get_int(&resolved, "start_iteration_predict", &mut cfg.start_iteration_predict)?;
    get_int(&resolved, "num_iteration_predict", &mut cfg.num_iteration_predict)?;
    get_bool(&resolved, "predict_raw_score", &mut cfg.predict_raw_score)?;
    get_bool(&resolved, "predict_leaf_index", &mut cfg.predict_leaf_index)?;
    get_bool(&resolved, "predict_contrib", &mut cfg.predict_contrib)?;
    get_bool(&resolved, "predict_disable_shape_check", &mut cfg.predict_disable_shape_check)?;
    get_bool(&resolved, "pred_early_stop", &mut cfg.pred_early_stop)?;
    get_int(&resolved, "pred_early_stop_freq", &mut cfg.pred_early_stop_freq)?;
    get_double(&resolved, "pred_early_stop_margin", &mut cfg.pred_early_stop_margin)?;
    get_string(&resolved, "output_result", &mut cfg.output_result);

    get_string(&resolved, "convert_model_language", &mut cfg.convert_model_language);
    get_string(&resolved, "convert_model", &mut cfg.convert_model);

    get_int(&resolved, "objective_seed", &mut cfg.objective_seed)?;

    get_int(&resolved, "num_class", &mut cfg.num_class)?;
    check_gt("num_class", cfg.num_class, 0)?;

    get_bool(&resolved, "is_unbalance", &mut cfg.is_unbalance)?;

    get_double(&resolved, "scale_pos_weight", &mut cfg.scale_pos_weight)?;
    check_gt_f("scale_pos_weight", cfg.scale_pos_weight, 0.0)?;

    get_double(&resolved, "sigmoid", &mut cfg.sigmoid)?;
    check_gt_f("sigmoid", cfg.sigmoid, 0.0)?;

    get_bool(&resolved, "boost_from_average", &mut cfg.boost_from_average)?;
    get_bool(&resolved, "reg_sqrt", &mut cfg.reg_sqrt)?;

    get_double(&resolved, "alpha", &mut cfg.alpha)?;
    check_gt_f("alpha", cfg.alpha, 0.0)?;

    get_double(&resolved, "fair_c", &mut cfg.fair_c)?;
    check_gt_f("fair_c", cfg.fair_c, 0.0)?;

    get_double(&resolved, "poisson_max_delta_step", &mut cfg.poisson_max_delta_step)?;
    check_gt_f("poisson_max_delta_step", cfg.poisson_max_delta_step, 0.0)?;

    get_double(&resolved, "tweedie_variance_power", &mut cfg.tweedie_variance_power)?;
    check_ge_f("tweedie_variance_power", cfg.tweedie_variance_power, 1.0)?;
    check_lt_f("tweedie_variance_power", cfg.tweedie_variance_power, 2.0)?;

    get_int(&resolved, "lambdarank_truncation_level", &mut cfg.lambdarank_truncation_level)?;
    check_gt("lambdarank_truncation_level", cfg.lambdarank_truncation_level, 0)?;

    get_bool(&resolved, "lambdarank_norm", &mut cfg.lambdarank_norm)?;

    get_double(
        &resolved,
        "lambdarank_position_bias_regularization",
        &mut cfg.lambdarank_position_bias_regularization,
    )?;
    check_ge_f(
        "lambdarank_position_bias_regularization",
        cfg.lambdarank_position_bias_regularization,
        0.0,
    )?;

    get_int(&resolved, "metric_freq", &mut cfg.metric_freq)?;
    check_gt("metric_freq", cfg.metric_freq, 0)?;

    get_bool(&resolved, "is_provide_training_metric", &mut cfg.is_provide_training_metric)?;

    get_int(&resolved, "multi_error_top_k", &mut cfg.multi_error_top_k)?;
    check_gt("multi_error_top_k", cfg.multi_error_top_k, 0)?;

    // --- Stage 4: CheckParamConflict (in-scope mutations + fatals) -----------
    check_param_conflict(&mut cfg, num_leaves_explicit)?;

    Ok(cfg)
}

/// Port of the in-scope subset of `CheckParamConflict` (config.cpp 314-474).
fn check_param_conflict(cfg: &mut Config, num_leaves_explicit: bool) -> Result<(), ConfigError> {
    // multiclass objective vs num_class + objective/metric multiclass mismatch.
    let objective_multiclass =
        is_multiclass_objective(&cfg.objective) || (cfg.objective == "custom" && cfg.num_class > 1);
    if objective_multiclass {
        if cfg.num_class <= 1 {
            return Err(ConfigError::Conflict {
                detail: "number of classes should be specified and greater than 1 for multiclass training".to_string(),
            });
        }
    } else if cfg.task == "train" && cfg.num_class != 1 {
        return Err(ConfigError::Conflict {
            detail: "number of classes must be 1 for non-multiclass training".to_string(),
        });
    }

    // max_depth > 0 && num_leaves NOT explicitly set → num_leaves = 2^max_depth
    // when that is more restrictive (config.cpp lines 384-398).
    if cfg.max_depth > 0 && !num_leaves_explicit {
        // std::pow(2, max_depth) in double. max_depth is capped so this stays
        // finite for realistic values; guard the shift against overflow.
        let full_num_leaves = 2f64.powi(cfg.max_depth);
        if full_num_leaves < cfg.num_leaves as f64 {
            cfg.num_leaves = full_num_leaves as i32;
        }
    }

    // device_type cuda forces row-wise; gpu forces col-wise (lines 399-417).
    if cfg.device_type == "gpu" {
        cfg.force_col_wise = true;
        cfg.force_row_wise = false;
    } else if cfg.device_type == "cuda" {
        cfg.force_col_wise = false;
        cfg.force_row_wise = true;
    }

    // path_smooth > kEpsilon && min_data_in_leaf < 2 → min_data_in_leaf = 2
    // (config.cpp lines 439-442). kEpsilon is f32 1e-15; compare in f64.
    if cfg.path_smooth > K_EPSILON as f64 && cfg.min_data_in_leaf < 2 {
        cfg.min_data_in_leaf = 2;
    }

    // min_data_in_leaf <= 0 && min_sum_hessian_in_leaf <= kEpsilon
    //   → min_data_in_leaf = 1 (config.cpp lines 457-462).
    if cfg.min_data_in_leaf <= 0 && cfg.min_sum_hessian_in_leaf <= K_EPSILON as f64 {
        cfg.min_data_in_leaf = 1;
    }

    // boosting == goss → boosting = gbdt, data_sample_strategy = goss (463-468).
    if cfg.boosting == "goss" {
        cfg.boosting = "gbdt".to_string();
        cfg.data_sample_strategy = "goss".to_string();
    }

    // bagging_by_query && data_sample_strategy != bagging → false (470-473).
    if cfg.bagging_by_query && cfg.data_sample_strategy != "bagging" {
        cfg.bagging_by_query = false;
    }

    Ok(())
}

/// C++ `CheckMultiClassObjective` (config.cpp lines 310-312).
fn is_multiclass_objective(objective: &str) -> bool {
    objective == "multiclass" || objective == "multiclassova"
}

// --- Typed getters: parse only if present & non-empty (C++ GetInt/Double/...) ---

/// Returns `true` if the key was present (and thus parsed/assigned).
fn get_int(
    params: &HashMap<String, String>,
    name: &str,
    out: &mut i32,
) -> Result<bool, ConfigError> {
    if let Some(v) = present(params, name) {
        *out = parse_int(name, v)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn get_double(
    params: &HashMap<String, String>,
    name: &str,
    out: &mut f64,
) -> Result<bool, ConfigError> {
    if let Some(v) = present(params, name) {
        *out = parse_double(name, v)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn get_bool(
    params: &HashMap<String, String>,
    name: &str,
    out: &mut bool,
) -> Result<bool, ConfigError> {
    if let Some(v) = present(params, name) {
        *out = parse_bool(name, v)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn get_string(params: &HashMap<String, String>, name: &str, out: &mut String) -> bool {
    if let Some(v) = present(params, name) {
        *out = v.clone();
        true
    } else {
        false
    }
}

/// C++ getters treat an absent OR empty value as "not provided".
fn present<'a>(params: &'a HashMap<String, String>, name: &str) -> Option<&'a String> {
    params.get(name).filter(|v| !v.is_empty())
}

fn parse_int(name: &str, value: &str) -> Result<i32, ConfigError> {
    value
        .trim()
        .parse::<i32>()
        .map_err(|_| ConfigError::InvalidType {
            param: name.to_string(),
            value: value.to_string(),
        })
}

fn parse_double(name: &str, value: &str) -> Result<f64, ConfigError> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| ConfigError::InvalidType {
            param: name.to_string(),
            value: value.to_string(),
        })
}

/// C++ `GetBool`: accepts `true/+` and `false/-`, case-insensitive.
fn parse_bool(name: &str, value: &str) -> Result<bool, ConfigError> {
    match value.to_lowercase().as_str() {
        "true" | "+" => Ok(true),
        "false" | "-" => Ok(false),
        _ => Err(ConfigError::InvalidType {
            param: name.to_string(),
            value: value.to_string(),
        }),
    }
}

// --- Enum-typed parses (config.cpp GetBoostingType/etc., lowercased) ---------

fn parse_boosting(value: &str) -> Result<String, ConfigError> {
    match value.to_lowercase().as_str() {
        "gbdt" | "gbrt" => Ok("gbdt".to_string()),
        "dart" => Ok("dart".to_string()),
        "goss" => Ok("goss".to_string()),
        "rf" | "random_forest" => Ok("rf".to_string()),
        _ => Err(ConfigError::UnknownValue {
            param: "boosting".to_string(),
            value: value.to_string(),
        }),
    }
}

fn parse_data_sample_strategy(value: &str) -> Result<String, ConfigError> {
    match value.to_lowercase().as_str() {
        "goss" => Ok("goss".to_string()),
        "bagging" => Ok("bagging".to_string()),
        _ => Err(ConfigError::UnknownValue {
            param: "data_sample_strategy".to_string(),
            value: value.to_string(),
        }),
    }
}

fn parse_task(value: &str) -> Result<String, ConfigError> {
    match value.to_lowercase().as_str() {
        "train" | "training" => Ok("train".to_string()),
        "predict" | "prediction" | "test" => Ok("predict".to_string()),
        "convert_model" => Ok("convert_model".to_string()),
        "refit" | "refit_tree" => Ok("refit".to_string()),
        "save_binary" => Ok("save_binary".to_string()),
        _ => Err(ConfigError::UnknownValue {
            param: "task".to_string(),
            value: value.to_string(),
        }),
    }
}

fn parse_device_type(value: &str) -> Result<String, ConfigError> {
    match value.to_lowercase().as_str() {
        "cpu" => Ok("cpu".to_string()),
        "gpu" => Ok("gpu".to_string()),
        "cuda" => Ok("cuda".to_string()),
        _ => Err(ConfigError::UnknownValue {
            param: "device_type".to_string(),
            value: value.to_string(),
        }),
    }
}

fn parse_tree_learner(value: &str) -> Result<String, ConfigError> {
    match value.to_lowercase().as_str() {
        "serial" => Ok("serial".to_string()),
        "feature" | "feature_parallel" => Ok("feature".to_string()),
        "data" | "data_parallel" => Ok("data".to_string()),
        "voting" | "voting_parallel" => Ok("voting".to_string()),
        _ => Err(ConfigError::UnknownValue {
            param: "tree_learner".to_string(),
            value: value.to_string(),
        }),
    }
}

/// The recognized objective set (config.cpp `ParseObjectiveAlias` + the
/// objective factory). C++ passes unknowns through to a later `Log::Fatal` in
/// the objective factory; for the config boundary we treat a clearly-unknown
/// objective as `UnknownValue` (validation tests require "nonsense" → Err).
fn parse_objective(value: &str) -> Result<String, ConfigError> {
    let v = value.to_lowercase();
    // Canonical objective names + the aliases ParseObjectiveAlias maps.
    const KNOWN: &[&str] = &[
        "regression",
        "regression_l2",
        "l2",
        "mean_squared_error",
        "mse",
        "l2_root",
        "root_mean_squared_error",
        "rmse",
        "regression_l1",
        "l1",
        "mean_absolute_error",
        "mae",
        "huber",
        "fair",
        "poisson",
        "quantile",
        "mape",
        "mean_absolute_percentage_error",
        "gamma",
        "tweedie",
        "binary",
        "lambdarank",
        "rank_xendcg",
        "xendcg",
        "xe_ndcg",
        "xe_ndcg_mart",
        "xendcg_mart",
        "multiclass",
        "softmax",
        "multiclassova",
        "multiclass_ova",
        "ova",
        "ovr",
        "cross_entropy",
        "xentropy",
        "cross_entropy_lambda",
        "xentlambda",
        "custom",
    ];
    if KNOWN.contains(&v.as_str()) {
        Ok(map_objective_alias(&v))
    } else {
        Err(ConfigError::UnknownValue {
            param: "objective".to_string(),
            value: value.to_string(),
        })
    }
}

/// Subset of `ParseObjectiveAlias` (config.cpp) — maps recognized aliases to
/// the canonical objective name.
fn map_objective_alias(v: &str) -> String {
    match v {
        "regression_l2" | "l2" | "mean_squared_error" | "mse" | "l2_root"
        | "root_mean_squared_error" | "rmse" => "regression".to_string(),
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => "regression_l1".to_string(),
        "mape" | "mean_absolute_percentage_error" => "mape".to_string(),
        "multiclass" | "softmax" => "multiclass".to_string(),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => "multiclassova".to_string(),
        "cross_entropy" | "xentropy" => "cross_entropy".to_string(),
        "cross_entropy_lambda" | "xentlambda" => "cross_entropy_lambda".to_string(),
        "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => {
            "rank_xendcg".to_string()
        }
        other => other.to_string(),
    }
}

// --- CHECK_* helpers → typed OutOfRange errors -------------------------------

fn check_ge(name: &str, value: i32, bound: i32) -> Result<(), ConfigError> {
    if value >= bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!(">= {bound}")))
    }
}

fn check_gt(name: &str, value: i32, bound: i32) -> Result<(), ConfigError> {
    if value > bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!("> {bound}")))
    }
}

fn check_le(name: &str, value: i32, bound: i32) -> Result<(), ConfigError> {
    if value <= bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!("<= {bound}")))
    }
}

fn check_ge_f(name: &str, value: f64, bound: f64) -> Result<(), ConfigError> {
    if value >= bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!(">= {bound}")))
    }
}

fn check_gt_f(name: &str, value: f64, bound: f64) -> Result<(), ConfigError> {
    if value > bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!("> {bound}")))
    }
}

fn check_le_f(name: &str, value: f64, bound: f64) -> Result<(), ConfigError> {
    if value <= bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!("<= {bound}")))
    }
}

fn check_lt_f(name: &str, value: f64, bound: f64) -> Result<(), ConfigError> {
    if value < bound {
        Ok(())
    } else {
        Err(out_of_range(name, value, format!("< {bound}")))
    }
}

fn out_of_range<T: std::fmt::Display>(name: &str, value: T, bound: String) -> ConfigError {
    ConfigError::OutOfRange {
        param: name.to_string(),
        value: value.to_string(),
        bound,
    }
}
