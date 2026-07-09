//! CFG-02: parameter aliases resolve to canonical names, matching the C++
//! `config_auto.cpp::alias_table()` verbatim.

use std::collections::HashMap;

use lgbm_core::config::resolve_alias;
use lgbm_core::config::Config;

fn params(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

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

// ---------------------------------------------------------------------------
// CR-02: deterministic alias-collision resolution in `from_params`, mirroring
// C++ `ParameterAlias::KeyAliasTransform` + `Config::SortAlias` (config.h
// lines 1220-1261). A directly-set CANONICAL key beats any alias; alias-vs-alias
// collisions onto the same canonical break ties by SortAlias `(key.len(), key)`
// (shorter first, then lexicographic) — NEVER HashMap iteration order.
// ---------------------------------------------------------------------------

#[test]
fn canonical_beats_alias_on_collision() {
    // num_iterations (canonical, directly set) vs n_estimators (alias).
    // The C++ second KeyAliasTransform loop only writes the alias value into
    // params[canonical] when the canonical key is ABSENT; here it is present,
    // so the directly-set canonical (10) wins and the alias (20) is dropped.
    let c = Config::from_params(&params(&[("num_iterations", "10"), ("n_estimators", "20")]))
        .unwrap();
    assert_eq!(c.num_iterations, 10, "directly-set canonical must beat alias");
}

#[test]
fn alias_vs_alias_sortalias_winner() {
    // Two aliases of num_iterations, no direct canonical. SortAlias picks the
    // shorter key first: num_round (len 9) < n_estimators (len 12), so num_round
    // (30) wins deterministically regardless of map insertion/iteration order.
    let c = Config::from_params(&params(&[("n_estimators", "20"), ("num_round", "30")]))
        .unwrap();
    assert_eq!(
        c.num_iterations, 30,
        "SortAlias winner (num_round, shorter) must beat n_estimators"
    );

    // Insertion order reversed: same deterministic outcome.
    let c = Config::from_params(&params(&[("num_round", "30"), ("n_estimators", "20")]))
        .unwrap();
    assert_eq!(c.num_iterations, 30, "SortAlias outcome must be order-independent");
}

#[test]
fn alias_vs_alias_equal_length_breaks_lexicographically() {
    // n_iter (len 6) and nrounds (len 7) are different lengths; pick a true
    // equal-length pair to exercise the lexicographic tie-break: num_tree and
    // num_iteration differ in length, so instead use two equal-length aliases.
    // "num_tree" (len 8) vs "num_iter"? num_iter is not an alias. Use the
    // equal-length pair "num_trees" (len 9) vs "num_round" (len 9): lexicographic
    // "num_round" < "num_trees" (… 'r' < 't'), so num_round wins.
    let c = Config::from_params(&params(&[("num_trees", "40"), ("num_round", "30")]))
        .unwrap();
    assert_eq!(
        c.num_iterations, 30,
        "equal-length tie must break lexicographically (num_round < num_trees)"
    );
}

#[test]
fn colliding_alias_resolution_is_deterministic() {
    // Run the colliding-alias input N>=50 times in one process and assert a
    // single distinct outcome — the determinism test the suite was missing.
    use std::collections::HashSet;
    let mut outcomes: HashSet<i32> = HashSet::new();
    for _ in 0..200 {
        let c = Config::from_params(&params(&[("num_iterations", "10"), ("n_estimators", "20")]))
            .unwrap();
        outcomes.insert(c.num_iterations);
    }
    assert_eq!(
        outcomes.len(),
        1,
        "colliding-alias resolution must be deterministic across runs, got {outcomes:?}"
    );
    assert!(outcomes.contains(&10), "deterministic outcome must be the canonical value 10");

    // Also assert determinism for the alias-vs-alias case.
    let mut alias_outcomes: HashSet<i32> = HashSet::new();
    for _ in 0..200 {
        let c = Config::from_params(&params(&[("n_estimators", "20"), ("num_round", "30")]))
            .unwrap();
        alias_outcomes.insert(c.num_iterations);
    }
    assert_eq!(
        alias_outcomes.len(),
        1,
        "alias-vs-alias resolution must be deterministic, got {alias_outcomes:?}"
    );
    assert!(alias_outcomes.contains(&30), "SortAlias winner must be num_round (30)");
}
