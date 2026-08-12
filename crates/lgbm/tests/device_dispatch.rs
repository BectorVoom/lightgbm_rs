//! Facade-level proof that the CPU/GPU device hyperparameters reach the backend
//! dispatch through the PUBLIC training entry points.
//!
//! # Why this file exists
//!
//! `device_type` parsed and validated but selected nothing: the backend was fixed
//! at COMPILE time by the `rocm`/`cuda`/`wgpu` cargo features, so a wheel could
//! never switch devices and `device_type` was inert. `gpu_device_id` / `num_gpu` /
//! `gpu_platform_id` / `gpu_use_dp` were not parsed at all.
//!
//! Every test below is a BEHAVIORAL assertion about what training does with those
//! params — that the CPU anchor is reachable on any build, that an unavailable
//! device fails loudly instead of silently training somewhere else, and that the
//! knobs which do not affect the CPU path leave results bit-identical — so the
//! file stays honest if the dispatch implementation moves.
//!
//! The GPU arms are cfg-guarded: on a build WITHOUT a GPU backend the assertion is
//! that the request is refused; a `--features cuda`/`rocm` build asserts the
//! opposite. Neither is skipped.

use std::collections::HashMap;

use lgbm::{train_raw, Config, DeviceKind, LgbmError, RawCorpus};

/// A small deterministic regression corpus — enough trees to expose any
/// divergence in the dispatched path, small enough to stay fast.
fn corpus(config: &Config) -> RawCorpus {
    let n = 200usize;
    let f0: Vec<f64> = (0..n).map(|i| (i % 20) as f64).collect();
    let f1: Vec<f64> = (0..n).map(|i| ((i / 20) % 10) as f64).collect();
    let labels: Vec<f32> = (0..n).map(|i| (2.0 * f0[i] - 1.5 * f1[i]) as f32).collect();
    let mut raw = RawCorpus::from_columns(vec![f0, f1], labels);
    raw.config = config.clone();
    raw
}

fn cfg(pairs: &[(&str, &str)]) -> Config {
    let mut params: HashMap<String, String> = [
        ("objective", "regression"),
        ("num_iterations", "8"),
        ("num_leaves", "8"),
        ("learning_rate", "0.3"),
        ("min_data_in_leaf", "5"),
        ("seed", "1"),
        ("deterministic", "true"),
        // Keep the device warnings off stderr during the test run.
        ("verbosity", "-1"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    for (k, v) in pairs {
        params.insert(k.to_string(), v.to_string());
    }
    Config::from_params(&params).expect("config must parse")
}

/// The per-row predictions -- the finest-grained observable of the grown model.
fn preds(config: &Config) -> Vec<Vec<f32>> {
    let c = corpus(config);
    let booster = train_raw(config, &c).expect("train must succeed");
    let rows: Vec<Vec<f64>> = (0..200)
        .map(|i| vec![(i % 20) as f64, ((i / 20) % 10) as f64])
        .collect();
    booster.predict(&rows)
}

// ---------------------------------------------------------------------------
// device_type routing
// ---------------------------------------------------------------------------

/// `device_type=cpu` is reachable on EVERY build (the `cpu` compute feature is
/// always on) and is what an unset `device_type` resolves to — so the two must
/// train bit-identical models.
#[test]
fn explicit_cpu_matches_the_default_device() {
    let default = preds(&cfg(&[]));
    let explicit = preds(&cfg(&[("device_type", "cpu")]));
    assert_eq!(
        default, explicit,
        "device_type=cpu must be the default path, bit-for-bit"
    );
}

/// The `device` alias reaches `device_type` through the same dispatch.
#[test]
fn device_alias_routes_to_the_same_backend() {
    assert_eq!(preds(&cfg(&[("device", "cpu")])), preds(&cfg(&[])));
}

/// A device with no compiled backend must be a typed `UnsupportedDevice` naming
/// the rebuild feature — NEVER a silent fallback to the CPU, which would train a
/// model on hardware the caller did not choose and report success.
#[test]
fn unavailable_device_is_refused_not_silently_downgraded() {
    for (device, available) in [
        ("gpu", cfg!(any(feature = "rocm", feature = "wgpu"))),
        ("cuda", cfg!(feature = "cuda")),
    ] {
        let config = cfg(&[("device_type", device)]);
        assert_eq!(config.device_kind().as_str(), device);
        let result = train_raw(&config, &corpus(&config));
        if available {
            assert!(
                result.is_ok(),
                "device_type={device} must train on a build whose backend is compiled in"
            );
            // "It did not error" is weak. The dispatched GPU model must also AGREE
            // with the CPU f64 anchor to the project's ~1e-6 output contract
            // (CLAUDE.md) — otherwise runtime dispatch could be routing to a
            // backend that trains something else entirely and still report success.
            let cpu = preds(&cfg(&[("device_type", "cpu")]));
            let gpu = preds(&config);
            assert_eq!(gpu.len(), cpu.len());
            for (row, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
                for (k, (gv, cv)) in g.iter().zip(c).enumerate() {
                    assert!(
                        (gv - cv).abs() <= 1e-6 * cv.abs().max(1.0),
                        "device_type={device} row {row} output {k}: {gv} vs cpu anchor {cv}"
                    );
                }
            }
        } else {
            match result {
                Err(LgbmError::UnsupportedDevice { device: d, detail }) => {
                    assert_eq!(d, device);
                    assert!(
                        detail.contains("--features"),
                        "the error must name the cargo feature to rebuild with: {detail}"
                    );
                }
                Err(other) => panic!("expected UnsupportedDevice for {device}, got {other:?}"),
                Ok(_) => panic!(
                    "device_type={device} trained despite having no compiled backend — \
                     a silent fallback to another device"
                ),
            }
        }
    }
}

/// The typed device reading is what dispatch keys on, so it must agree with the
/// canonical string for every value of the C++ closed enum.
#[test]
fn device_kind_covers_the_closed_enum() {
    assert_eq!(cfg(&[]).device_kind(), DeviceKind::Cpu);
    assert_eq!(cfg(&[("device_type", "gpu")]).device_kind(), DeviceKind::Gpu);
    assert_eq!(
        cfg(&[("device_type", "cuda")]).device_kind(),
        DeviceKind::Cuda
    );
}

// ---------------------------------------------------------------------------
// GPU tuning knobs
// ---------------------------------------------------------------------------

/// The GPU knobs are parsed and validated on every build, and none of them
/// perturbs a CPU train — `gpu_device_id` selects a CubeCL device (irrelevant on
/// the CPU anchor) while `gpu_platform_id`/`gpu_use_dp` have no CubeCL analog at
/// all. Bit-identical predictions are the assertion that they are inert here
/// rather than accidentally wired into the CPU path.
#[test]
fn gpu_knobs_are_accepted_and_inert_on_cpu() {
    let base = preds(&cfg(&[]));
    for pairs in [
        vec![("num_gpu", "1")],
        vec![("gpu_device_id", "3")],
        vec![("gpu_platform_id", "0")],
        vec![("gpu_use_dp", "true")],
    ] {
        assert_eq!(
            preds(&cfg(&pairs)),
            base,
            "{pairs:?} must not change the CPU-trained model"
        );
    }
}

/// `gpu_device_id` resolves to the CubeCL device INDEX the GPU arms bind, with
/// the C++ `-1` default meaning device 0.
#[test]
fn gpu_device_id_resolves_to_a_device_index() {
    assert_eq!(cfg(&[]).gpu_device_index(), 0);
    assert_eq!(cfg(&[("gpu_device_id", "-1")]).gpu_device_index(), 0);
    assert_eq!(cfg(&[("gpu_device_id", "2")]).gpu_device_index(), 2);
}

/// The knobs with no CubeCL analog are reported, not silently honored — a user
/// who sets `gpu_use_dp` expecting f64 GPU math must be told it does nothing.
#[test]
fn inapplicable_knobs_are_reported() {
    assert!(cfg(&[]).device_warnings().is_empty());
    let warnings = cfg(&[("gpu_platform_id", "0"), ("gpu_use_dp", "true")]).device_warnings();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
}

// ---------------------------------------------------------------------------
// num_threads
// ---------------------------------------------------------------------------

/// `num_threads` controls parallelism only: the host folds are order-stable, so
/// pinning the pool to one worker must reproduce the default-threaded model
/// EXACTLY. A divergence here would mean a reduction depends on the worker count,
/// breaking the determinism contract.
///
/// This also exercises `configure_threads`, which sizes the process-global pool
/// (the C++ `omp_set_num_threads` mapping) — the assertion holds whichever train
/// in this binary happens to run first.
#[test]
fn num_threads_does_not_change_results() {
    let base = preds(&cfg(&[]));
    assert_eq!(preds(&cfg(&[("num_threads", "1")])), base);
    // `n_jobs` / `nthread` are aliases of the same canonical param.
    assert_eq!(preds(&cfg(&[("n_jobs", "1")])), base);
    assert_eq!(preds(&cfg(&[("nthread", "1")])), base);
}
