//! QGP-05: quantized-gradient param names must be reclassified from
//! `OUT_OF_SCOPE_PARAMS` to `IN_SCOPE_PARAMS` now that `Config::from_params`
//! (T-01..T-04) parses them.

use lgbm_core::config::scope::{IN_SCOPE_PARAMS, OUT_OF_SCOPE_PARAMS};

#[test]
fn quantized_grad_params_are_in_scope() {
    for key in [
        "use_quantized_grad",
        "num_grad_quant_bins",
        "quant_train_renew_leaf",
        "stochastic_rounding",
    ] {
        assert!(IN_SCOPE_PARAMS.contains(&key), "{key} must be listed in IN_SCOPE_PARAMS");
        assert!(
            !OUT_OF_SCOPE_PARAMS.contains(&key),
            "{key} must NOT remain in OUT_OF_SCOPE_PARAMS"
        );
    }
}
