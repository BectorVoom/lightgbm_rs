//! Tri-state `LGBM_CUDA_ON_DEVICE` resolver + `on_device_default` unit tests (23-01).
//!
//! Two independent things are locked here:
//!
//! 1. The tri-state MAPPING (D-01, V5 exact-match closed enum) is tested through the
//!    pure [`lgbm_compute::cuda_on_device_override_from`] helper — NOT through the
//!    public OnceLock-cached resolver. The resolver reads the env once per process
//!    (Pitfall P-1), so it cannot be exercised across multiple values in one test
//!    binary; the pure helper is the OnceLock initializer's own mapping, so asserting
//!    it asserts the real contract without fighting the cache. There is deliberately
//!    NO test that flips the env between two reads in one process (P-1 warning sign).
//!
//! 2. The cpu-build DEVICE DEFAULT is OFF (SC-4 anchor): under a `cpu` build with the
//!    env unset the resolved default is `false`, so every backend is byte-unchanged.

use lgbm_compute::cuda_on_device_override_from;

/// D-01: exact-string closed enum. `"1"` forces on, `"0"` forces off, and EVERYTHING
/// else (unset/empty/malformed) maps to `None` = follow the device default.
#[test]
fn tri_state_mapping_is_exact_closed_enum() {
    // Force-on / force-off: the two first-class operator overrides.
    assert_eq!(cuda_on_device_override_from(Some("1")), Some(true), "\"1\" => force on");
    assert_eq!(cuda_on_device_override_from(Some("0")), Some(false), "\"0\" => force off");

    // Unset and every malformed/empty value follow the device default (None).
    assert_eq!(cuda_on_device_override_from(None), None, "unset => follow default");
    assert_eq!(cuda_on_device_override_from(Some("")), None, "empty => follow default");
    assert_eq!(cuda_on_device_override_from(Some("2")), None, "other digit => follow default");
    assert_eq!(cuda_on_device_override_from(Some("true")), None, "\"true\" => follow default");
}

/// The `"0"` off-switch and `"1"` on-switch are distinguished at the mapping level
/// (D-01) — `"0"` is a first-class force-off, not the same as unset.
#[test]
fn zero_is_force_off_one_is_force_on() {
    assert_eq!(
        cuda_on_device_override_from(Some("0")),
        Some(false),
        "\"0\" must be an explicit force-off, distinct from unset (None)"
    );
    assert_eq!(
        cuda_on_device_override_from(Some("1")),
        Some(true),
        "\"1\" must be an explicit force-on"
    );
    assert_ne!(
        cuda_on_device_override_from(Some("0")),
        cuda_on_device_override_from(None),
        "force-off (Some(false)) and follow-default (None) must not collapse"
    );
}

/// SC-4 anchor: on a `cpu` build (no `-F cuda`), with `LGBM_CUDA_ON_DEVICE` unset in a
/// fresh process, the resolver returns `false` — the pre-verdict safe state (D-09), so
/// the cpu/rocm merge gate is byte-unchanged. Guarded `#[cfg(not(feature = "cuda"))]`
/// because a `cuda` build's device default is a separate contract (flipped in 23-04).
#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_build_default_is_off_when_env_unset() {
    // This test binary is invoked without LGBM_CUDA_ON_DEVICE set by the merge gate.
    // The override is None (unset) and on_device_default() is false on the cpu build,
    // so the resolved value must be false.
    assert!(
        !lgbm_compute::cuda_on_device_enabled(),
        "cpu-build default with env unset must be OFF (SC-4 byte-unchanged)"
    );
}
