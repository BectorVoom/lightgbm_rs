//! FND-01: when `seed` is provided, the six sub-seeds derive from a fresh
//! `Random(seed)` via `next_short(0, 32767)` called six times in the EXACT C++
//! order (config.cpp lines 259-268):
//!   data_random_seed, bagging_seed, drop_seed, feature_fraction_seed,
//!   objective_seed, extra_seed.

use std::collections::HashMap;

use lgbm_core::config::Config;
use lgbm_core::random::Random;

fn params(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Independently recompute the six successive draws and compare order + values.
fn expected_subseeds(seed: i32) -> [i32; 6] {
    let mut rand = Random::new(seed);
    let int_max = i16::MAX as i32; // 32767
    [
        rand.next_short(0, int_max),
        rand.next_short(0, int_max),
        rand.next_short(0, int_max),
        rand.next_short(0, int_max),
        rand.next_short(0, int_max),
        rand.next_short(0, int_max),
    ]
}

#[test]
fn seed_42_derives_six_subseeds_in_exact_order() {
    let cfg = Config::from_params(&params(&[("seed", "42")])).unwrap();
    let exp = expected_subseeds(42);
    assert_eq!(cfg.data_random_seed, exp[0]);
    assert_eq!(cfg.bagging_seed, exp[1]);
    assert_eq!(cfg.drop_seed, exp[2]);
    assert_eq!(cfg.feature_fraction_seed, exp[3]);
    assert_eq!(cfg.objective_seed, exp[4]);
    assert_eq!(cfg.extra_seed, exp[5]);
    assert_eq!(cfg.seed, 42);
}

#[test]
fn multiple_seeds_match_random_lcg() {
    for &seed in &[0_i32, 1, 7, 42, 100, 12400, 1_592_594_996_i32.wrapping_abs()] {
        let cfg = Config::from_params(&params(&[("seed", &seed.to_string())])).unwrap();
        let exp = expected_subseeds(seed);
        assert_eq!(
            [
                cfg.data_random_seed,
                cfg.bagging_seed,
                cfg.drop_seed,
                cfg.feature_fraction_seed,
                cfg.objective_seed,
                cfg.extra_seed,
            ],
            exp,
            "sub-seed derivation mismatch for seed {seed}"
        );
    }
}

#[test]
fn random_seed_alias_drives_derivation() {
    // `random_seed`/`random_state` are aliases for `seed`.
    let via_alias = Config::from_params(&params(&[("random_state", "42")])).unwrap();
    let exp = expected_subseeds(42);
    assert_eq!(via_alias.data_random_seed, exp[0]);
    assert_eq!(via_alias.extra_seed, exp[5]);
}

#[test]
fn no_seed_keeps_config_h_defaults() {
    // Without an explicit seed, the config.h member defaults stand.
    let cfg = Config::from_params(&HashMap::new()).unwrap();
    assert_eq!(cfg.data_random_seed, 1);
    assert_eq!(cfg.bagging_seed, 3);
    assert_eq!(cfg.drop_seed, 4);
    assert_eq!(cfg.feature_fraction_seed, 2);
    assert_eq!(cfg.objective_seed, 5);
    assert_eq!(cfg.extra_seed, 6);
}
