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

/// The CPU/GPU device knobs are reclassified in the same way: `Config::from_params`
/// now parses and validates all four, `device_type` drives the facade's RUNTIME
/// backend dispatch, and `gpu_device_id` selects the CubeCL device index — so the
/// Python D-07 gate must stop rejecting them.
#[test]
fn gpu_device_params_are_in_scope() {
    for key in ["num_gpu", "gpu_platform_id", "gpu_device_id", "gpu_use_dp"] {
        assert!(IN_SCOPE_PARAMS.contains(&key), "{key} must be listed in IN_SCOPE_PARAMS");
        assert!(
            !OUT_OF_SCOPE_PARAMS.contains(&key),
            "{key} must NOT remain in OUT_OF_SCOPE_PARAMS"
        );
    }
}

/// Distributed learning stays out of scope — the reclassification above must not
/// have emptied the gate that keeps unported params from silently training.
#[test]
fn distributed_params_stay_out_of_scope() {
    for key in [
        "num_machines",
        "local_listen_port",
        "time_out",
        "machine_list_filename",
        "machines",
    ] {
        assert!(
            OUT_OF_SCOPE_PARAMS.contains(&key),
            "{key} must remain in OUT_OF_SCOPE_PARAMS"
        );
        assert!(!IN_SCOPE_PARAMS.contains(&key), "{key} must NOT be in IN_SCOPE_PARAMS");
    }
}
