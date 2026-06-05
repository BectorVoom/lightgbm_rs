//! The hand-ported Rust `Config` — a single flat struct mirroring the C++
//! `LightGBM::Config` 1:1 (same public field names, same default values).
//!
//! Analog: `LightGBM/include/LightGBM/config.h` (the documented member
//! declarations with their `= default` initializers) plus
//! `LightGBM/src/io/config_auto.cpp::GetMembersFromString` (the canonical
//! member set / order). This is the single configuration bag every later crate
//! imports, mirroring the C++ "single `Config` everywhere" pattern (D-12).
//!
//! # Design (D-12)
//!
//! ONE flat `pub struct Config` with public fields named identically to the
//! C++ `Config`. No nested sub-structs. `impl Default for Config` sets every
//! field to the exact `config.h` default. Numeric widths mirror the C++ widths
//! (`int` → `i32`, `double` → `f64`, `bool` → `bool`) for parity — this is the
//! config bag, NOT the f32 score/gradient contract (which applies to
//! gradients/scores, not configuration scalars).

pub mod alias;
pub mod scope;
pub mod set;

pub use alias::resolve_alias;
pub use scope::IN_SCOPE_PARAMS;

/// Flat configuration struct mirroring C++ `LightGBM::Config` 1:1.
///
/// Field names and default values are ported verbatim from `config.h`. The
/// in-scope single-machine parameter set is enumerated in [`scope`]; the full
/// alias table is in [`alias`]. Construction from a parameter map (with alias
/// resolution, seed derivation, CHECK validation, and conflict mutations) lives
/// in [`set::from_params`].
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    // --- Core / objective ---
    /// `std::string task` (canonical "train"). Held as a lowercased string.
    pub task: String,
    /// `std::string objective` — loss objective. config.h default: "regression".
    pub objective: String,
    /// `std::string boosting` — boosting type. config.h default: "gbdt".
    pub boosting: String,
    /// `std::string data_sample_strategy`. config.h default: "bagging".
    pub data_sample_strategy: String,
    /// `std::string data` — training data path. config.h default: "".
    pub data: String,

    // --- Core training control ---
    /// `int num_iterations`. config.h default: 100.
    pub num_iterations: i32,
    /// `double learning_rate`. config.h default: 0.1.
    pub learning_rate: f64,
    /// `int num_leaves`. config.h default: 31 (`kDefaultNumLeaves`).
    pub num_leaves: i32,
    /// `std::string tree_learner`. config.h default: "serial".
    pub tree_learner: String,
    /// `int num_threads`. config.h default: 0.
    pub num_threads: i32,
    /// `std::string device_type`. config.h default: "cpu".
    pub device_type: String,
    /// `int seed`. config.h default: 0.
    pub seed: i32,
    /// `bool deterministic`. config.h default: false.
    pub deterministic: bool,

    // --- Learning control ---
    /// `bool force_col_wise`. config.h default: false.
    pub force_col_wise: bool,
    /// `bool force_row_wise`. config.h default: false.
    pub force_row_wise: bool,
    /// `double histogram_pool_size`. config.h default: -1.0.
    pub histogram_pool_size: f64,
    /// `int max_depth`. config.h default: -1.
    pub max_depth: i32,
    /// `int min_data_in_leaf`. config.h default: 20.
    pub min_data_in_leaf: i32,
    /// `double min_sum_hessian_in_leaf`. config.h default: 1e-3.
    pub min_sum_hessian_in_leaf: f64,
    /// `double bagging_fraction`. config.h default: 1.0.
    pub bagging_fraction: f64,
    /// `double pos_bagging_fraction`. config.h default: 1.0.
    pub pos_bagging_fraction: f64,
    /// `double neg_bagging_fraction`. config.h default: 1.0.
    pub neg_bagging_fraction: f64,
    /// `int bagging_freq`. config.h default: 0.
    pub bagging_freq: i32,
    /// `bool bagging_by_query`. config.h default: false.
    pub bagging_by_query: bool,
    /// `double feature_fraction`. config.h default: 1.0.
    pub feature_fraction: f64,
    /// `double feature_fraction_bynode`. config.h default: 1.0.
    pub feature_fraction_bynode: f64,
    /// `bool extra_trees`. config.h default: false.
    pub extra_trees: bool,
    /// `int early_stopping_round`. config.h default: 0.
    pub early_stopping_round: i32,
    /// `double early_stopping_min_delta`. config.h default: 0.0.
    pub early_stopping_min_delta: f64,
    /// `bool first_metric_only`. config.h default: false.
    pub first_metric_only: bool,
    /// `double max_delta_step`. config.h default: 0.0.
    pub max_delta_step: f64,
    /// `double lambda_l1`. config.h default: 0.0.
    pub lambda_l1: f64,
    /// `double lambda_l2`. config.h default: 0.0.
    pub lambda_l2: f64,
    /// `double min_gain_to_split`. config.h default: 0.0.
    pub min_gain_to_split: f64,
    /// `double drop_rate`. config.h default: 0.1.
    pub drop_rate: f64,
    /// `int max_drop`. config.h default: 50.
    pub max_drop: i32,
    /// `double skip_drop`. config.h default: 0.5.
    pub skip_drop: f64,
    /// `bool xgboost_dart_mode`. config.h default: false.
    pub xgboost_dart_mode: bool,
    /// `bool uniform_drop`. config.h default: false.
    pub uniform_drop: bool,
    /// `double top_rate`. config.h default: 0.2.
    pub top_rate: f64,
    /// `double other_rate`. config.h default: 0.1.
    pub other_rate: f64,
    /// `int min_data_per_group`. config.h default: 100.
    pub min_data_per_group: i32,
    /// `int max_cat_threshold`. config.h default: 32.
    pub max_cat_threshold: i32,
    /// `double cat_l2`. config.h default: 10.0.
    pub cat_l2: f64,
    /// `double cat_smooth`. config.h default: 10.0.
    pub cat_smooth: f64,
    /// `int max_cat_to_onehot`. config.h default: 4.
    pub max_cat_to_onehot: i32,
    /// `int top_k`. config.h default: 20.
    pub top_k: i32,
    /// `std::string monotone_constraints_method`. config.h default: "basic".
    pub monotone_constraints_method: String,
    /// `double monotone_penalty`. config.h default: 0.0.
    pub monotone_penalty: f64,
    /// `std::string forcedsplits_filename`. config.h default: "".
    pub forcedsplits_filename: String,
    /// `double refit_decay_rate`. config.h default: 0.9.
    pub refit_decay_rate: f64,
    /// `double cegb_tradeoff`. config.h default: 1.0.
    pub cegb_tradeoff: f64,
    /// `double cegb_penalty_split`. config.h default: 0.0.
    pub cegb_penalty_split: f64,
    /// `double path_smooth`. config.h default: 0.
    pub path_smooth: f64,
    /// `std::string interaction_constraints`. config.h default: "".
    pub interaction_constraints: String,
    /// `int verbosity`. config.h default: 1.
    pub verbosity: i32,

    // --- IO config ---
    /// `std::string input_model`. config.h default: "".
    pub input_model: String,
    /// `std::string output_model`. config.h default: "LightGBM_model.txt".
    pub output_model: String,
    /// `int saved_feature_importance_type`. config.h default: 0.
    pub saved_feature_importance_type: i32,
    /// `int snapshot_freq`. config.h default: -1.
    pub snapshot_freq: i32,
    /// `int max_bin`. config.h default: 255.
    pub max_bin: i32,
    /// `int min_data_in_bin`. config.h default: 3.
    pub min_data_in_bin: i32,
    /// `int bin_construct_sample_cnt`. config.h default: 200000.
    pub bin_construct_sample_cnt: i32,
    /// `bool is_enable_sparse`. config.h default: true.
    pub is_enable_sparse: bool,
    /// `bool enable_bundle`. config.h default: true.
    pub enable_bundle: bool,
    /// `bool use_missing`. config.h default: true.
    pub use_missing: bool,
    /// `bool zero_as_missing`. config.h default: false.
    pub zero_as_missing: bool,
    /// `bool feature_pre_filter`. config.h default: true.
    pub feature_pre_filter: bool,
    /// `bool pre_partition`. config.h default: false.
    pub pre_partition: bool,
    /// `bool two_round`. config.h default: false.
    pub two_round: bool,
    /// `bool header`. config.h default: false.
    pub header: bool,
    /// `std::string label_column`. config.h default: "".
    pub label_column: String,
    /// `std::string weight_column`. config.h default: "".
    pub weight_column: String,
    /// `std::string group_column`. config.h default: "".
    pub group_column: String,
    /// `std::string ignore_column`. config.h default: "".
    pub ignore_column: String,
    /// `std::string categorical_feature`. config.h default: "".
    pub categorical_feature: String,
    /// `std::string forcedbins_filename`. config.h default: "".
    pub forcedbins_filename: String,
    /// `bool save_binary`. config.h default: false.
    pub save_binary: bool,
    /// `bool precise_float_parser`. config.h default: false.
    pub precise_float_parser: bool,
    /// `std::string parser_config_file`. config.h default: "".
    pub parser_config_file: String,

    // --- Predict config ---
    /// `int start_iteration_predict`. config.h default: 0.
    pub start_iteration_predict: i32,
    /// `int num_iteration_predict`. config.h default: -1.
    pub num_iteration_predict: i32,
    /// `bool predict_raw_score`. config.h default: false.
    pub predict_raw_score: bool,
    /// `bool predict_leaf_index`. config.h default: false.
    pub predict_leaf_index: bool,
    /// `bool predict_contrib`. config.h default: false.
    pub predict_contrib: bool,
    /// `bool predict_disable_shape_check`. config.h default: false.
    pub predict_disable_shape_check: bool,
    /// `bool pred_early_stop`. config.h default: false.
    pub pred_early_stop: bool,
    /// `int pred_early_stop_freq`. config.h default: 10.
    pub pred_early_stop_freq: i32,
    /// `double pred_early_stop_margin`. config.h default: 10.0.
    pub pred_early_stop_margin: f64,
    /// `std::string output_result`. config.h default: "LightGBM_predict_result.txt".
    pub output_result: String,

    // --- Convert config ---
    /// `std::string convert_model_language`. config.h default: "".
    pub convert_model_language: String,
    /// `std::string convert_model`. config.h default: "gbdt_prediction.cpp".
    pub convert_model: String,

    // --- Objective config ---
    /// `int num_class`. config.h default: 1.
    pub num_class: i32,
    /// `bool is_unbalance`. config.h default: false.
    pub is_unbalance: bool,
    /// `double scale_pos_weight`. config.h default: 1.0.
    pub scale_pos_weight: f64,
    /// `double sigmoid`. config.h default: 1.0.
    pub sigmoid: f64,
    /// `bool boost_from_average`. config.h default: true.
    pub boost_from_average: bool,
    /// `bool reg_sqrt`. config.h default: false.
    pub reg_sqrt: bool,
    /// `double alpha`. config.h default: 0.9.
    pub alpha: f64,
    /// `double fair_c`. config.h default: 1.0.
    pub fair_c: f64,
    /// `double poisson_max_delta_step`. config.h default: 0.7.
    pub poisson_max_delta_step: f64,
    /// `double tweedie_variance_power`. config.h default: 1.5.
    pub tweedie_variance_power: f64,
    /// `int lambdarank_truncation_level`. config.h default: 30.
    pub lambdarank_truncation_level: i32,
    /// `bool lambdarank_norm`. config.h default: true.
    pub lambdarank_norm: bool,
    /// `double lambdarank_position_bias_regularization`. config.h default: 0.0.
    pub lambdarank_position_bias_regularization: f64,

    // --- Metric config ---
    /// `int metric_freq`. config.h default: 1.
    pub metric_freq: i32,
    /// `bool is_provide_training_metric`. config.h default: false.
    pub is_provide_training_metric: bool,
    /// `int multi_error_top_k`. config.h default: 1.
    pub multi_error_top_k: i32,

    // --- Derived sub-seeds (config_auto.cpp members; config.h defaults) ---
    /// `int data_random_seed`. config.h default: 1 (overwritten by seed derivation).
    pub data_random_seed: i32,
    /// `int bagging_seed`. config.h default: 3 (overwritten by seed derivation).
    pub bagging_seed: i32,
    /// `int drop_seed`. config.h default: 4 (overwritten by seed derivation).
    pub drop_seed: i32,
    /// `int feature_fraction_seed`. config.h default: 2 (overwritten by seed derivation).
    pub feature_fraction_seed: i32,
    /// `int objective_seed`. config.h default: 5 (overwritten by seed derivation).
    pub objective_seed: i32,
    /// `int extra_seed`. config.h default: 6 (overwritten by seed derivation).
    pub extra_seed: i32,
}

impl Default for Config {
    /// Mirrors the C++ `Config` member initializers in `config.h` 1:1.
    fn default() -> Self {
        Config {
            task: "train".to_string(),
            objective: "regression".to_string(),
            boosting: "gbdt".to_string(),
            data_sample_strategy: "bagging".to_string(),
            data: String::new(),

            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            tree_learner: "serial".to_string(),
            num_threads: 0,
            device_type: "cpu".to_string(),
            seed: 0,
            deterministic: false,

            force_col_wise: false,
            force_row_wise: false,
            histogram_pool_size: -1.0,
            max_depth: -1,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            bagging_fraction: 1.0,
            pos_bagging_fraction: 1.0,
            neg_bagging_fraction: 1.0,
            bagging_freq: 0,
            bagging_by_query: false,
            feature_fraction: 1.0,
            feature_fraction_bynode: 1.0,
            extra_trees: false,
            early_stopping_round: 0,
            early_stopping_min_delta: 0.0,
            first_metric_only: false,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            drop_rate: 0.1,
            max_drop: 50,
            skip_drop: 0.5,
            xgboost_dart_mode: false,
            uniform_drop: false,
            top_rate: 0.2,
            other_rate: 0.1,
            min_data_per_group: 100,
            max_cat_threshold: 32,
            cat_l2: 10.0,
            cat_smooth: 10.0,
            max_cat_to_onehot: 4,
            top_k: 20,
            monotone_constraints_method: "basic".to_string(),
            monotone_penalty: 0.0,
            forcedsplits_filename: String::new(),
            refit_decay_rate: 0.9,
            cegb_tradeoff: 1.0,
            cegb_penalty_split: 0.0,
            path_smooth: 0.0,
            interaction_constraints: String::new(),
            verbosity: 1,

            input_model: String::new(),
            output_model: "LightGBM_model.txt".to_string(),
            saved_feature_importance_type: 0,
            snapshot_freq: -1,
            max_bin: 255,
            min_data_in_bin: 3,
            bin_construct_sample_cnt: 200000,
            is_enable_sparse: true,
            enable_bundle: true,
            use_missing: true,
            zero_as_missing: false,
            feature_pre_filter: true,
            pre_partition: false,
            two_round: false,
            header: false,
            label_column: String::new(),
            weight_column: String::new(),
            group_column: String::new(),
            ignore_column: String::new(),
            categorical_feature: String::new(),
            forcedbins_filename: String::new(),
            save_binary: false,
            precise_float_parser: false,
            parser_config_file: String::new(),

            start_iteration_predict: 0,
            num_iteration_predict: -1,
            predict_raw_score: false,
            predict_leaf_index: false,
            predict_contrib: false,
            predict_disable_shape_check: false,
            pred_early_stop: false,
            pred_early_stop_freq: 10,
            pred_early_stop_margin: 10.0,
            output_result: "LightGBM_predict_result.txt".to_string(),

            convert_model_language: String::new(),
            convert_model: "gbdt_prediction.cpp".to_string(),

            num_class: 1,
            is_unbalance: false,
            scale_pos_weight: 1.0,
            sigmoid: 1.0,
            boost_from_average: true,
            reg_sqrt: false,
            alpha: 0.9,
            fair_c: 1.0,
            poisson_max_delta_step: 0.7,
            tweedie_variance_power: 1.5,
            lambdarank_truncation_level: 30,
            lambdarank_norm: true,
            lambdarank_position_bias_regularization: 0.0,

            metric_freq: 1,
            is_provide_training_metric: false,
            multi_error_top_k: 1,

            // config.h defaults (overwritten by from_params seed derivation when `seed` is given).
            data_random_seed: 1,
            bagging_seed: 3,
            drop_seed: 4,
            feature_fraction_seed: 2,
            objective_seed: 5,
            extra_seed: 6,
        }
    }
}
