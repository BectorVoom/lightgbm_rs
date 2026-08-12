//! CFG-03 / D-14 / T-1-01: `Config::from_params` validation fidelity, exercised
//! across a RANDOMIZED set of in-scope / boundary / invalid inputs.
//!
//! The randomization is itself DETERMINISTIC: it is driven by a fixed in-test
//! seed via the ported `Random` LCG (no wall-clock / OS entropy), so re-runs are
//! reproducible. For valid/in-bound cases we assert `Ok` and that the parsed
//! field round-trips the input (`(params -> outcome)` parity). For
//! invalid/out-of-bound cases (C++ hard CHECK failures) we assert the correct
//! typed `Err`. NO generated input may ever panic.

use std::collections::HashMap;

use lgbm_core::config::Config;
use lgbm_core::error::ConfigError;
use lgbm_core::random::Random;

fn params(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

// ---------------------------------------------------------------------------
// Fixed named regression cases (kept alongside randomized coverage).
// ---------------------------------------------------------------------------

#[test]
fn named_invalid_cases_return_typed_errors() {
    // num_leaves=1 → CHECK_GT(num_leaves, 1)
    assert!(matches!(
        Config::from_params(&params(&[("num_leaves", "1")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // num_leaves=-1 → CHECK_GT(num_leaves, 1)
    assert!(matches!(
        Config::from_params(&params(&[("num_leaves", "-1")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // learning_rate=0 → CHECK_GT(learning_rate, 0.0)
    assert!(matches!(
        Config::from_params(&params(&[("learning_rate", "0")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // bagging_fraction=1.5 → CHECK_LE(bagging_fraction, 1.0)
    assert!(matches!(
        Config::from_params(&params(&[("bagging_fraction", "1.5")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // num_iterations=-1 → CHECK_GE(num_iterations, 0)
    assert!(matches!(
        Config::from_params(&params(&[("num_iterations", "-1")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // num_iterations=abc → parse failure → InvalidType
    assert!(matches!(
        Config::from_params(&params(&[("num_iterations", "abc")])),
        Err(ConfigError::InvalidType { .. })
    ));
    // objective=nonsense → UnknownValue
    assert!(matches!(
        Config::from_params(&params(&[("objective", "nonsense")])),
        Err(ConfigError::UnknownValue { .. })
    ));
    // boosting=nonsense → UnknownValue
    assert!(matches!(
        Config::from_params(&params(&[("boosting", "nonsense")])),
        Err(ConfigError::UnknownValue { .. })
    ));
    // ADV-01: monotone_constraints entry outside {-1,0,1} → OutOfRange.
    assert!(matches!(
        Config::from_params(&params(&[("monotone_constraints", "1,2,0")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // ADV-01: monotone_constraints_method nonsense → UnknownValue.
    assert!(matches!(
        Config::from_params(&params(&[("monotone_constraints_method", "nonsense")])),
        Err(ConfigError::UnknownValue { .. })
    ));
    // ADV-05: cegb_tradeoff negative → OutOfRange.
    assert!(matches!(
        Config::from_params(&params(&[("cegb_tradeoff", "-1")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    // ADV-06: refit_decay_rate outside [0,1] → OutOfRange (config.h CHECK [0,1]).
    assert!(matches!(
        Config::from_params(&params(&[("refit_decay_rate", "1.5")])),
        Err(ConfigError::OutOfRange { .. })
    ));
    assert!(matches!(
        Config::from_params(&params(&[("refit_decay_rate", "-0.1")])),
        Err(ConfigError::OutOfRange { .. })
    ));
}

#[test]
fn adv_refit_importance_params_resolve() {
    // ADV-06: refit_decay_rate default 0.9; valid in-range resolves.
    let c = Config::from_params(&params(&[])).unwrap();
    assert_eq!(c.refit_decay_rate, 0.9, "default refit_decay_rate");
    assert_eq!(c.saved_feature_importance_type, 0, "default split selector");
    let c = Config::from_params(&params(&[("refit_decay_rate", "0.0")])).unwrap();
    assert_eq!(c.refit_decay_rate, 0.0);
    let c = Config::from_params(&params(&[("refit_decay_rate", "1.0")])).unwrap();
    assert_eq!(c.refit_decay_rate, 1.0);
    // ADV-06: input_model string passes through (+ aliases model_input/model_in).
    let c = Config::from_params(&params(&[("model_input", "base.txt")])).unwrap();
    assert_eq!(c.input_model, "base.txt");
    // ADV-07: saved_feature_importance_type selects split(0)/gain(1).
    let c = Config::from_params(&params(&[("saved_feature_importance_type", "1")])).unwrap();
    assert_eq!(c.saved_feature_importance_type, 1);
}

#[test]
fn adv_constraint_params_parse_with_aliases() {
    // ADV-01 monotone_constraints vector + alias `mc`.
    let c = Config::from_params(&params(&[("mc", "1,-1,0")])).unwrap();
    assert_eq!(c.monotone_constraints, vec![1, -1, 0]);
    // method enum + penalty.
    let c = Config::from_params(&params(&[
        ("monotone_constraints_method", "intermediate"),
        ("monotone_penalty", "5.0"),
    ]))
    .unwrap();
    assert_eq!(c.monotone_constraints_method, "intermediate");
    assert_eq!(c.monotone_penalty, 5.0);
    // ADV-02 interaction_constraints (stored as the raw string spec).
    let c = Config::from_params(&params(&[("interaction_constraints", "[0,1],[2]")])).unwrap();
    assert_eq!(c.interaction_constraints, "[0,1],[2]");
    // ADV-03 forced_splits_filename alias `fs` resolves to forcedsplits_filename.
    let c = Config::from_params(&params(&[("forced_splits", "fs.json")])).unwrap();
    assert_eq!(c.forcedsplits_filename, "fs.json");
    // ADV-04 extra_trees + extra_seed.
    let c = Config::from_params(&params(&[("extra_trees", "true"), ("extra_seed", "9")])).unwrap();
    assert!(c.extra_trees);
    assert_eq!(c.extra_seed, 9);
    // ADV-05 cegb params incl. feature-penalty vectors.
    let c = Config::from_params(&params(&[
        ("cegb_tradeoff", "0.5"),
        ("cegb_penalty_split", "0.1"),
        ("cegb_penalty_feature_coupled", "1.0,2.0"),
        ("cegb_penalty_feature_lazy", "0.5,0.5"),
    ]))
    .unwrap();
    assert_eq!(c.cegb_tradeoff, 0.5);
    assert_eq!(c.cegb_penalty_split, 0.1);
    assert_eq!(c.cegb_penalty_feature_coupled, vec![1.0, 2.0]);
    assert_eq!(c.cegb_penalty_feature_lazy, vec![0.5, 0.5]);
    // G5-1 feature_contri (config.h:535-539, aliases feature_contrib/fc/fp/
    // feature_penalty; config_auto.cpp:470 `Common::StringToArray<double>(tmp_str, ',')`
    // — comma-separated doubles, same parse as the cegb vectors above).
    let c = Config::from_params(&params(&[("feature_contri", "0.5,1.0")])).unwrap();
    assert_eq!(c.feature_contri, vec![0.5, 1.0]);
    // alias `fc`.
    let c = Config::from_params(&params(&[("fc", "0.0,2.0,1.0")])).unwrap();
    assert_eq!(c.feature_contri, vec![0.0, 2.0, 1.0]);
}

#[test]
fn valid_boundary_cases_return_ok_and_roundtrip() {
    // num_leaves=2 is the smallest valid value (CHECK_GT(.,1)).
    let c = Config::from_params(&params(&[("num_leaves", "2")])).unwrap();
    assert_eq!(c.num_leaves, 2);
    // num_leaves=131072 is the largest valid (CHECK_LE(.,131072)).
    let c = Config::from_params(&params(&[("num_leaves", "131072")])).unwrap();
    assert_eq!(c.num_leaves, 131072);
    // learning_rate just above 0.
    let c = Config::from_params(&params(&[("learning_rate", "0.0001")])).unwrap();
    assert_eq!(c.learning_rate, 0.0001);
    // bagging_fraction == 1.0 (upper inclusive bound).
    let c = Config::from_params(&params(&[("bagging_fraction", "1.0")])).unwrap();
    assert_eq!(c.bagging_fraction, 1.0);
    // feature_fraction just above 0.
    let c = Config::from_params(&params(&[("feature_fraction", "0.000001")])).unwrap();
    assert_eq!(c.feature_fraction, 0.000001);
    // num_iterations == 0 (CHECK_GE(.,0) inclusive).
    let c = Config::from_params(&params(&[("num_iterations", "0")])).unwrap();
    assert_eq!(c.num_iterations, 0);
}

#[test]
fn alias_resolves_then_validates() {
    // n_estimators → num_iterations; eta → learning_rate.
    let c = Config::from_params(&params(&[("n_estimators", "250"), ("eta", "0.05")])).unwrap();
    assert_eq!(c.num_iterations, 250);
    assert_eq!(c.learning_rate, 0.05);
    // Invalid via alias still surfaces the canonical-named error.
    assert!(matches!(
        Config::from_params(&params(&[("eta", "0")])),
        Err(ConfigError::OutOfRange { .. })
    ));
}

#[test]
fn bool_coercion_matches_cpp() {
    assert!(Config::from_params(&params(&[("deterministic", "true")])).unwrap().deterministic);
    assert!(Config::from_params(&params(&[("deterministic", "+")])).unwrap().deterministic);
    assert!(!Config::from_params(&params(&[("deterministic", "false")])).unwrap().deterministic);
    assert!(!Config::from_params(&params(&[("deterministic", "-")])).unwrap().deterministic);
    // Bad bool → InvalidType, never panic.
    assert!(matches!(
        Config::from_params(&params(&[("deterministic", "maybe")])),
        Err(ConfigError::InvalidType { .. })
    ));
}

// ---------------------------------------------------------------------------
// QGP-01/03/04: quantized-gradient boolean params (`use_quantized_grad`,
// `quant_train_renew_leaf`, `stochastic_rounding`) parse and default correctly.
// ---------------------------------------------------------------------------

#[test]
fn quantized_grad_bool_params_parse_and_default() {
    // use_quantized_grad: default false, roundtrip true/false, invalid -> InvalidType.
    let c = Config::from_params(&params(&[])).unwrap();
    assert!(!c.use_quantized_grad, "default use_quantized_grad must be false");

    let c = Config::from_params(&params(&[("use_quantized_grad", "true")])).unwrap();
    assert!(c.use_quantized_grad);

    let c = Config::from_params(&params(&[("use_quantized_grad", "false")])).unwrap();
    assert!(!c.use_quantized_grad);

    assert!(matches!(
        Config::from_params(&params(&[("use_quantized_grad", "maybe")])),
        Err(ConfigError::InvalidType { param, .. }) if param == "use_quantized_grad"
    ));

    // quant_train_renew_leaf: default false, roundtrip true/false, invalid -> InvalidType.
    let c = Config::from_params(&params(&[])).unwrap();
    assert!(!c.quant_train_renew_leaf, "default quant_train_renew_leaf must be false");

    let c = Config::from_params(&params(&[("quant_train_renew_leaf", "true")])).unwrap();
    assert!(c.quant_train_renew_leaf);

    let c = Config::from_params(&params(&[("quant_train_renew_leaf", "false")])).unwrap();
    assert!(!c.quant_train_renew_leaf);

    assert!(matches!(
        Config::from_params(&params(&[("quant_train_renew_leaf", "maybe")])),
        Err(ConfigError::InvalidType { param, .. }) if param == "quant_train_renew_leaf"
    ));

    // stochastic_rounding: default TRUE (C++ config.h default), roundtrip, invalid -> InvalidType.
    let c = Config::from_params(&params(&[])).unwrap();
    assert!(c.stochastic_rounding, "default stochastic_rounding must be true (C++ config.h default)");

    let c = Config::from_params(&params(&[("stochastic_rounding", "false")])).unwrap();
    assert!(!c.stochastic_rounding);

    let c = Config::from_params(&params(&[("stochastic_rounding", "true")])).unwrap();
    assert!(c.stochastic_rounding);

    assert!(matches!(
        Config::from_params(&params(&[("stochastic_rounding", "maybe")])),
        Err(ConfigError::InvalidType { param, .. }) if param == "stochastic_rounding"
    ));
}

#[test]
fn num_grad_quant_bins_parses_and_validates_range() {
    let c = Config::from_params(&params(&[])).unwrap();
    assert_eq!(c.num_grad_quant_bins, 4, "default num_grad_quant_bins");

    let c = Config::from_params(&params(&[("num_grad_quant_bins", "128")])).unwrap();
    assert_eq!(c.num_grad_quant_bins, 128);

    // boundaries: 1 and 254 are valid (GradientDiscretizer::new's own 1..=254 contract).
    let c = Config::from_params(&params(&[("num_grad_quant_bins", "1")])).unwrap();
    assert_eq!(c.num_grad_quant_bins, 1);
    let c = Config::from_params(&params(&[("num_grad_quant_bins", "254")])).unwrap();
    assert_eq!(c.num_grad_quant_bins, 254);

    assert!(matches!(
        Config::from_params(&params(&[("num_grad_quant_bins", "0")])),
        Err(ConfigError::OutOfRange { param, .. }) if param == "num_grad_quant_bins"
    ));
    assert!(matches!(
        Config::from_params(&params(&[("num_grad_quant_bins", "255")])),
        Err(ConfigError::OutOfRange { param, .. }) if param == "num_grad_quant_bins"
    ));
    assert!(matches!(
        Config::from_params(&params(&[("num_grad_quant_bins", "-4")])),
        Err(ConfigError::OutOfRange { param, .. }) if param == "num_grad_quant_bins"
    ));
}

// ---------------------------------------------------------------------------
// CR-01: empty-string value == ABSENT (no-op), matching C++ Get* helpers which
// guard on `params.count(name) > 0 && !params.at(name).empty()` (config.h
// 1165-1218). Applies to the `seed` lookup and the six enum reads (task,
// boosting, data_sample_strategy, objective, device_type, tree_learner).
// ---------------------------------------------------------------------------

#[test]
fn empty_seed_is_noop() {
    let def = Config::default();
    let c = Config::from_params(&params(&[("seed", "")])).unwrap();
    // seed keeps its default; the six derived sub-seeds keep theirs (the
    // derivation block is skipped because seed is treated as absent).
    assert_eq!(c.seed, def.seed, "empty seed must keep default seed");
    assert_eq!(c.data_random_seed, def.data_random_seed);
    assert_eq!(c.bagging_seed, def.bagging_seed);
    assert_eq!(c.drop_seed, def.drop_seed);
    assert_eq!(c.feature_fraction_seed, def.feature_fraction_seed);
    assert_eq!(c.objective_seed, def.objective_seed);
    assert_eq!(c.extra_seed, def.extra_seed);
}

#[test]
fn empty_enum_values_are_noop() {
    let def = Config::default();

    let c = Config::from_params(&params(&[("task", "")])).unwrap();
    assert_eq!(c.task, def.task, "empty task must keep default");

    let c = Config::from_params(&params(&[("boosting", "")])).unwrap();
    assert_eq!(c.boosting, def.boosting, "empty boosting must keep default");

    let c = Config::from_params(&params(&[("data_sample_strategy", "")])).unwrap();
    assert_eq!(
        c.data_sample_strategy, def.data_sample_strategy,
        "empty data_sample_strategy must keep default"
    );

    let c = Config::from_params(&params(&[("objective", "")])).unwrap();
    assert_eq!(c.objective, def.objective, "empty objective must keep default");

    let c = Config::from_params(&params(&[("device_type", "")])).unwrap();
    assert_eq!(c.device_type, def.device_type, "empty device_type must keep default");

    let c = Config::from_params(&params(&[("tree_learner", "")])).unwrap();
    assert_eq!(c.tree_learner, def.tree_learner, "empty tree_learner must keep default");
}

#[test]
fn non_empty_invalid_enum_still_errors() {
    // Control: present() filters ONLY the empty string, not all validation. A
    // non-empty invalid value must still return the typed UnknownValue error.
    assert!(matches!(
        Config::from_params(&params(&[("objective", "nonsense")])),
        Err(ConfigError::UnknownValue { .. })
    ));
    assert!(matches!(
        Config::from_params(&params(&[("boosting", "nonsense")])),
        Err(ConfigError::UnknownValue { .. })
    ));
    assert!(matches!(
        Config::from_params(&params(&[("device_type", "nonsense")])),
        Err(ConfigError::UnknownValue { .. })
    ));
}

// ---------------------------------------------------------------------------
// CheckParamConflict mutations.
// ---------------------------------------------------------------------------

#[test]
fn max_depth_reduces_unset_num_leaves() {
    // max_depth=3, num_leaves unset → num_leaves = 2^3 = 8 (more restrictive).
    let c = Config::from_params(&params(&[("max_depth", "3")])).unwrap();
    assert_eq!(c.num_leaves, 8);
}

#[test]
fn explicit_num_leaves_not_overridden_by_max_depth() {
    // num_leaves explicitly set → max_depth must NOT override it.
    let c = Config::from_params(&params(&[("max_depth", "3"), ("num_leaves", "31")])).unwrap();
    assert_eq!(c.num_leaves, 31);
}

#[test]
fn path_smooth_raises_min_data_in_leaf() {
    // path_smooth>kEpsilon && min_data_in_leaf<2 → min_data_in_leaf=2.
    let c = Config::from_params(&params(&[("path_smooth", "0.1"), ("min_data_in_leaf", "1")]))
        .unwrap();
    assert_eq!(c.min_data_in_leaf, 2);
}

#[test]
fn min_data_zero_and_zero_hessian_sets_one() {
    let c = Config::from_params(&params(&[
        ("min_data_in_leaf", "0"),
        ("min_sum_hessian_in_leaf", "0"),
    ]))
    .unwrap();
    assert_eq!(c.min_data_in_leaf, 1);
}

// ---------------------------------------------------------------------------
// CPU / GPU device knobs (now in-scope: parsed, validated, and consumed by the
// facade's runtime backend dispatch).
// ---------------------------------------------------------------------------

#[test]
fn gpu_device_knobs_parse_and_round_trip() {
    let c = Config::from_params(&params(&[
        ("device_type", "cuda"),
        ("num_gpu", "1"),
        ("gpu_platform_id", "2"),
        ("gpu_device_id", "3"),
        ("gpu_use_dp", "true"),
    ]))
    .unwrap();
    assert_eq!(c.device_type, "cuda");
    assert_eq!(c.num_gpu, 1);
    assert_eq!(c.gpu_platform_id, 2);
    assert_eq!(c.gpu_device_id, 3);
    assert!(c.gpu_use_dp);
}

#[test]
fn num_gpu_below_one_is_rejected() {
    // C++ CHECK_GT(num_gpu, 0).
    assert!(matches!(
        Config::from_params(&params(&[("num_gpu", "0")])),
        Err(ConfigError::OutOfRange { .. })
    ));
}

#[test]
fn multi_gpu_is_rejected_not_silently_single_device() {
    // The Rust port trains on ONE device; asking for several must be a typed
    // error rather than a silent single-device train.
    assert!(matches!(
        Config::from_params(&params(&[("num_gpu", "2")])),
        Err(ConfigError::OutOfRange { .. })
    ));
}

#[test]
fn device_kind_and_index_derive_from_the_parsed_knobs() {
    use lgbm_core::config::DeviceKind;

    let c = Config::from_params(&params(&[("device_type", "gpu"), ("gpu_device_id", "2")])).unwrap();
    assert_eq!(c.device_kind(), DeviceKind::Gpu);
    assert_eq!(c.gpu_device_index(), 2);

    // The `device` alias reaches device_type, and the -1 default maps to index 0.
    let c = Config::from_params(&params(&[("device", "cuda")])).unwrap();
    assert_eq!(c.device_kind(), DeviceKind::Cuda);
    assert_eq!(c.gpu_device_index(), 0);

    assert_eq!(Config::default().device_kind(), DeviceKind::Cpu);
}

#[test]
fn inapplicable_gpu_knobs_warn_instead_of_failing() {
    // gpu_platform_id / gpu_use_dp have no CubeCL analog: accepted (so official
    // param dicts port unchanged) but reported, never silently honored.
    let c = Config::from_params(&params(&[
        ("device_type", "cuda"),
        ("gpu_platform_id", "0"),
        ("gpu_use_dp", "true"),
    ]))
    .unwrap();
    let warnings = c.device_warnings();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(warnings.iter().any(|w| w.contains("gpu_platform_id")));
    assert!(warnings.iter().any(|w| w.contains("gpu_use_dp")));

    // The defaults must stay silent.
    assert!(Config::default().device_warnings().is_empty());
}

#[test]
fn cuda_forces_row_wise() {
    let c = Config::from_params(&params(&[("device_type", "cuda")])).unwrap();
    assert!(c.force_row_wise);
    assert!(!c.force_col_wise);
}

#[test]
fn goss_boosting_rewrites_to_gbdt_plus_strategy() {
    let c = Config::from_params(&params(&[("boosting", "goss")])).unwrap();
    assert_eq!(c.boosting, "gbdt");
    assert_eq!(c.data_sample_strategy, "goss");
}

#[test]
fn multiclass_objective_requires_num_class() {
    // multiclass with num_class=1 (default) → Conflict.
    assert!(matches!(
        Config::from_params(&params(&[("objective", "multiclass")])),
        Err(ConfigError::Conflict { .. })
    ));
    // multiclass with num_class>1 → Ok.
    let c = Config::from_params(&params(&[("objective", "multiclass"), ("num_class", "3")]))
        .unwrap();
    assert_eq!(c.objective, "multiclass");
    assert_eq!(c.num_class, 3);
}

// ---------------------------------------------------------------------------
// D-14 randomized in-scope / boundary / invalid coverage (deterministic).
// ---------------------------------------------------------------------------

/// A field under randomized test, with its CHECK bounds expressed as f64.
struct FieldSpec {
    name: &'static str,
    /// Inclusive lower bound, or None if unbounded below.
    lo: Option<(f64, bool)>, // (bound, inclusive)
    /// Inclusive upper bound, or None if unbounded above.
    hi: Option<(f64, bool)>, // (bound, inclusive)
    is_int: bool,
}

fn int_field(name: &'static str, lo: Option<(f64, bool)>, hi: Option<(f64, bool)>) -> FieldSpec {
    FieldSpec { name, lo, hi, is_int: true }
}
fn dbl_field(name: &'static str, lo: Option<(f64, bool)>, hi: Option<(f64, bool)>) -> FieldSpec {
    FieldSpec { name, lo, hi, is_int: false }
}

/// True if `value` satisfies the field's bounds.
fn in_bounds(spec: &FieldSpec, value: f64) -> bool {
    if let Some((b, incl)) = spec.lo {
        if incl {
            if value < b {
                return false;
            }
        } else if value <= b {
            return false;
        }
    }
    if let Some((b, incl)) = spec.hi {
        if incl {
            if value > b {
                return false;
            }
        } else if value >= b {
            return false;
        }
    }
    true
}

#[test]
fn randomized_in_scope_boundary_invalid_coverage_is_deterministic_and_panic_free() {
    // Fields with simple numeric CHECK bounds (subset spanning GE/GT/LE/LT).
    let specs = vec![
        int_field("num_iterations", Some((0.0, true)), None),
        int_field("num_leaves", Some((1.0, false)), Some((131072.0, true))),
        int_field("min_data_in_leaf", Some((0.0, true)), None),
        int_field("max_cat_threshold", Some((0.0, false)), None),
        int_field("top_k", Some((0.0, false)), None),
        int_field("max_bin", Some((1.0, false)), None),
        int_field("min_data_in_bin", Some((0.0, false)), None),
        // NOTE: num_class is intentionally excluded here — its validity is
        // cross-parameter (objective vs num_class in CheckParamConflict), not a
        // simple self-bound. It is covered by `multiclass_objective_requires_num_class`.
        int_field("metric_freq", Some((0.0, false)), None),
        dbl_field("learning_rate", Some((0.0, false)), None),
        dbl_field("bagging_fraction", Some((0.0, false)), Some((1.0, true))),
        dbl_field("feature_fraction", Some((0.0, false)), Some((1.0, true))),
        dbl_field("lambda_l1", Some((0.0, true)), None),
        dbl_field("lambda_l2", Some((0.0, true)), None),
        dbl_field("min_gain_to_split", Some((0.0, true)), None),
        dbl_field("drop_rate", Some((0.0, true)), Some((1.0, true))),
        dbl_field("skip_drop", Some((0.0, true)), Some((1.0, true))),
        dbl_field("sigmoid", Some((0.0, false)), None),
        dbl_field("poisson_max_delta_step", Some((0.0, false)), None),
        dbl_field("tweedie_variance_power", Some((1.0, true)), Some((2.0, false))),
    ];

    // Candidate values to draw from: well inside, at boundary, just inside,
    // just outside, far outside, plus malformed strings.
    let numeric_candidates: &[f64] = &[
        -1000.0, -1.5, -1.0, -0.5, 0.0, 0.0001, 0.5, 1.0, 1.0001, 1.5, 2.0, 31.0,
        131072.0, 131073.0, 200000.0, 1e6,
    ];
    let malformed: &[&str] = &["abc", "", "1.2.3", "nan_but_text", "+-"];

    // Deterministic driver: fixed in-test seed, ported Random LCG.
    let mut rng = Random::new(0x5EED_1234u32 as i32);
    let mut total = 0;

    for _ in 0..2000 {
        let spec = &specs[(rng.next_int(0, specs.len() as i32)) as usize];

        // ~15% of the time inject a malformed string (must → InvalidType, no panic).
        if rng.next_int(0, 100) < 15 {
            let raw = malformed[(rng.next_int(0, malformed.len() as i32)) as usize];
            let kv = params(&[(spec.name, raw)]);
            // Empty string == "not provided" in C++ getters → should be Ok.
            let res = Config::from_params(&kv);
            if raw.is_empty() {
                assert!(res.is_ok(), "empty value for {} should be no-op Ok", spec.name);
            } else {
                assert!(
                    matches!(res, Err(ConfigError::InvalidType { .. })),
                    "malformed `{raw}` for {} must be InvalidType, got {res:?}",
                    spec.name
                );
            }
            total += 1;
            continue;
        }

        // Numeric candidate (round for int fields).
        let mut val = numeric_candidates[(rng.next_int(0, numeric_candidates.len() as i32)) as usize];
        if spec.is_int {
            val = val.round();
        }
        let val_str = if spec.is_int {
            format!("{}", val as i64)
        } else {
            format!("{val}")
        };

        let kv = params(&[(spec.name, &val_str)]);
        let res = Config::from_params(&kv);

        // Parse the actual stored value back for the in-bounds round-trip check.
        let parsed = val_str.parse::<f64>().ok();
        let valid = parsed.map(|p| in_bounds(spec, p)).unwrap_or(false);

        if valid {
            let cfg = res.expect(&format!(
                "in-bounds {}={val_str} must be Ok",
                spec.name
            ));
            // (params -> outcome) parity: parsed field round-trips the input.
            assert_field_roundtrip(&cfg, spec.name, parsed.unwrap(), spec.is_int);
        } else {
            assert!(
                matches!(res, Err(ConfigError::OutOfRange { .. }) | Err(ConfigError::InvalidType { .. })),
                "out-of-bounds {}={val_str} must be a typed Err, got {res:?}",
                spec.name
            );
        }
        total += 1;
    }

    assert!(total >= 2000, "expected full randomized coverage run");
}

fn assert_field_roundtrip(cfg: &Config, name: &str, expected: f64, is_int: bool) {
    macro_rules! eq_int {
        ($field:expr) => {
            assert_eq!($field as f64, expected, "round-trip mismatch for {name}")
        };
    }
    macro_rules! eq_dbl {
        ($field:expr) => {
            assert!(($field - expected).abs() < 1e-12, "round-trip mismatch for {name}")
        };
    }
    if is_int {
        match name {
            "num_iterations" => eq_int!(cfg.num_iterations),
            "num_leaves" => eq_int!(cfg.num_leaves),
            "min_data_in_leaf" => {
                // min_data_in_leaf may be mutated by conflict rules when 0; only
                // assert round-trip for values >= 2 (no mutation path applies).
                if expected >= 2.0 {
                    eq_int!(cfg.min_data_in_leaf);
                }
            }
            "max_cat_threshold" => eq_int!(cfg.max_cat_threshold),
            "top_k" => eq_int!(cfg.top_k),
            "max_bin" => eq_int!(cfg.max_bin),
            "min_data_in_bin" => eq_int!(cfg.min_data_in_bin),
            "num_class" => eq_int!(cfg.num_class),
            "metric_freq" => eq_int!(cfg.metric_freq),
            _ => {}
        }
    } else {
        match name {
            "learning_rate" => eq_dbl!(cfg.learning_rate),
            "bagging_fraction" => eq_dbl!(cfg.bagging_fraction),
            "feature_fraction" => eq_dbl!(cfg.feature_fraction),
            "lambda_l1" => eq_dbl!(cfg.lambda_l1),
            "lambda_l2" => eq_dbl!(cfg.lambda_l2),
            "min_gain_to_split" => eq_dbl!(cfg.min_gain_to_split),
            "drop_rate" => eq_dbl!(cfg.drop_rate),
            "skip_drop" => eq_dbl!(cfg.skip_drop),
            "sigmoid" => eq_dbl!(cfg.sigmoid),
            "poisson_max_delta_step" => eq_dbl!(cfg.poisson_max_delta_step),
            "tweedie_variance_power" => eq_dbl!(cfg.tweedie_variance_power),
            _ => {}
        }
    }
}

#[test]
fn fuzz_hostile_strings_never_panic() {
    // A deterministic spray of hostile/random key+value strings: the contract
    // is simply that from_params returns (Ok or Err) without ever panicking.
    let mut rng = Random::new(0x1357_9BDFu32 as i32);
    let keys = [
        "num_leaves", "learning_rate", "objective", "boosting", "device_type",
        "deterministic", "max_depth", "made_up_key", "seed", "num_class",
    ];
    let charset: &[u8] = b"0123456789.-+eE abcXYZ_/";
    for _ in 0..3000 {
        let key = keys[(rng.next_int(0, keys.len() as i32)) as usize];
        let len = rng.next_int(0, 12) as usize;
        let mut val = String::with_capacity(len);
        for _ in 0..len {
            let c = charset[(rng.next_int(0, charset.len() as i32)) as usize];
            val.push(c as char);
        }
        let kv = params(&[(key, val.as_str())]);
        // Must not panic; result may be Ok or any ConfigError variant.
        let _ = Config::from_params(&kv);
    }
}
