//! MET-03 extended-metric parity replay (Phase 7 W3, plan 07-04).
//!
//! A new VALIDATION Wave-0 file: it replays the Rust `Metric::eval` (and the
//! xentropy / multiclass / binary metric variants) over the REAL lib_lightgbm 4.6
//! captured (scores, labels) and asserts the Rust value matches the captured
//! metric value within ORACLE_TOL (bit-exact where the algorithm permits).
//!
//! KEY INSIGHT: metrics are pure evaluation functions (score + label -> number);
//! they do NOT train, so they are INDEPENDENT of the DEF-07-02 learner-level
//! split-gain knife-edge. The capture supplies the real-binary scores; the metric
//! just evaluates them, so every in-scope metric (incl. fair/gamma/tweedie) reaches
//! faithful parity.
//!
//! Capture-gated (mirrors `boosting_parity.rs`): each cell SKIP-passes (returns
//! early) when its golden triplet is absent, so a fresh checkout without the
//! `cargo run -p xtask -- metric-oracle-capture` run still builds and the suite is
//! green. Once the goldens are committed the cells assert real parity.
//!
//! Goldens live under the TRACKED oracle-harness crate dir
//! (`tests/fixtures/metric/<name>_{scores,labels,value}.txt`), NEVER the untracked
//! `LightGBM/` reference tree.

use std::path::PathBuf;

use lgbm_compute::device_metric::DeviceMetricKind;
use lgbm_compute::kernels::metric_pointwise::{eval_metric_on, MetricEvalParams};
use lgbm_compute::runtime::cpu_client;
use lgbm_metric::{
    AucMu, BinaryMetric, Metric, MultiError, RegressionMetricParams, XentropyMetric,
};
use lgbm_model::ObjectiveKind;
use oracle_harness::comparator::ORACLE_TOL;

/// The committed metric golden directory — TRACKED under the oracle-harness crate,
/// NEVER the untracked C++/LightGBM reference tree. Populated by
/// `cargo run -p xtask -- metric-oracle-capture`.
fn metric_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metric")
}

/// SKIP gracefully (returning `None`) when a golden file is absent — matching the
/// boosting_parity idiom (a fresh checkout without the capture run still builds).
fn read_golden(name: &str) -> Option<String> {
    let path = metric_dir().join(name);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "metric_parity: SKIP — golden {} not found. Run \
                 `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- metric-oracle-capture`.",
                path.display()
            );
            None
        }
    }
}

/// First non-comment, non-empty data line of a golden file.
fn data_line(text: &str) -> &str {
    text.lines()
        .find(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .expect("golden has a data line")
}

fn parse_f64_bits(text: &str) -> Vec<f64> {
    data_line(text)
        .split_whitespace()
        .map(|t| f64::from_bits(t.parse::<u64>().expect("u64 bits")))
        .collect()
}

fn parse_f32_bits(text: &str) -> Vec<f32> {
    data_line(text)
        .split_whitespace()
        .map(|t| f32::from_bits(t.parse::<u32>().expect("u32 bits")))
        .collect()
}

/// Load the (scores, labels, captured-value) triplet for a metric `name`. Returns
/// `None` (SKIP) when any of the three goldens is absent.
fn load_triplet(name: &str) -> Option<(Vec<f64>, Vec<f32>, f64)> {
    let scores = read_golden(&format!("{name}_scores.txt"))?;
    let labels = read_golden(&format!("{name}_labels.txt"))?;
    let value = read_golden(&format!("{name}_value.txt"))?;
    let scores = parse_f64_bits(&scores);
    let labels = parse_f32_bits(&labels);
    let value = parse_f64_bits(&value)[0];
    Some((scores, labels, value))
}

/// Assert a Rust-computed metric value matches the captured real-binary value
/// within ORACLE_TOL.
fn assert_metric(name: &str, rust: f64, captured: f64) {
    let diff = (rust - captured).abs();
    assert!(
        diff <= ORACLE_TOL as f64,
        "{name}: Rust {rust} vs real-binary {captured} (abs_diff {diff} > {})",
        ORACLE_TOL
    );
}

// --- regression-family cells (identity ConvertOutput) ---

#[test]
fn quantile_parity() {
    let Some((scores, labels, captured)) = load_triplet("quantile") else {
        return;
    };
    // The capture trains with default alpha=0.9.
    let m = Metric::Quantile { alpha: 0.9 };
    assert_metric("quantile", m.eval(&scores, &labels).unwrap(), captured);
}

#[test]
fn huber_parity() {
    let Some((scores, labels, captured)) = load_triplet("huber") else {
        return;
    };
    let m = Metric::Huber { alpha: 0.9 };
    assert_metric("huber", m.eval(&scores, &labels).unwrap(), captured);
}

#[test]
fn fair_parity() {
    let Some((scores, labels, captured)) = load_triplet("fair") else {
        return;
    };
    let m = Metric::Fair {
        fair_c: RegressionMetricParams::default().fair_c,
    };
    assert_metric("fair", m.eval(&scores, &labels).unwrap(), captured);
}

#[test]
fn mape_parity() {
    let Some((scores, labels, captured)) = load_triplet("mape") else {
        return;
    };
    assert_metric("mape", Metric::Mape.eval(&scores, &labels).unwrap(), captured);
}

// --- exp-ConvertOutput regression cells (objective == metric family) ---

#[test]
fn poisson_parity() {
    let Some((scores, labels, captured)) = load_triplet("poisson") else {
        return;
    };
    assert_metric(
        "poisson",
        Metric::Poisson.eval(&scores, &labels).unwrap(),
        captured,
    );
}

#[test]
fn gamma_parity() {
    let Some((scores, labels, captured)) = load_triplet("gamma") else {
        return;
    };
    assert_metric("gamma", Metric::Gamma.eval(&scores, &labels).unwrap(), captured);
}

#[test]
fn gamma_deviance_parity() {
    let Some((scores, labels, captured)) = load_triplet("gamma_deviance") else {
        return;
    };
    assert_metric(
        "gamma_deviance",
        Metric::GammaDeviance.eval(&scores, &labels).unwrap(),
        captured,
    );
}

#[test]
fn tweedie_parity() {
    let Some((scores, labels, captured)) = load_triplet("tweedie") else {
        return;
    };
    let m = Metric::Tweedie {
        tweedie_variance_power: RegressionMetricParams::default().tweedie_variance_power,
    };
    assert_metric("tweedie", m.eval(&scores, &labels).unwrap(), captured);
}

// --- xentropy-family cells ([0,1] labels) ---

#[test]
fn cross_entropy_parity() {
    let Some((scores, labels, captured)) = load_triplet("cross_entropy") else {
        return;
    };
    assert_metric(
        "cross_entropy",
        XentropyMetric::CrossEntropy.eval(&scores, &labels).unwrap(),
        captured,
    );
}

#[test]
fn cross_entropy_lambda_parity() {
    let Some((scores, labels, captured)) = load_triplet("cross_entropy_lambda") else {
        return;
    };
    assert_metric(
        "cross_entropy_lambda",
        XentropyMetric::CrossEntropyLambda
            .eval(&scores, &labels)
            .unwrap(),
        captured,
    );
}

#[test]
fn kullback_leibler_parity() {
    let Some((scores, labels, captured)) = load_triplet("kullback_leibler") else {
        return;
    };
    assert_metric(
        "kullback_leibler",
        XentropyMetric::KullbackLeibler
            .eval(&scores, &labels)
            .unwrap(),
        captured,
    );
}

// --- binary: average_precision ---

#[test]
fn average_precision_parity() {
    let Some((scores, labels, captured)) = load_triplet("average_precision") else {
        return;
    };
    assert_metric(
        "average_precision",
        BinaryMetric::AveragePrecision.eval(&scores, &labels).unwrap(),
        captured,
    );
}

// --- multiclass: multi_error, auc_mu (class-major scores, num_class=3) ---

#[test]
fn multi_error_parity() {
    let Some((scores, labels, captured)) = load_triplet("multi_error") else {
        return;
    };
    let m = MultiError::new(ObjectiveKind::Multiclass { num_class: 3 }, 3, 1);
    assert_metric("multi_error", m.eval(&scores, &labels).unwrap(), captured);
}

#[test]
fn auc_mu_parity() {
    let Some((scores, labels, captured)) = load_triplet("auc_mu") else {
        return;
    };
    let m = AucMu::new(3);
    assert_metric("auc_mu", m.eval(&scores, &labels).unwrap(), captured);
}

// --- builder accepts the new metric names into the multi-metric list (MET-02) ---

#[test]
fn builder_accepts_extended_metric_names() {
    use lgbm::TrainingBuilder;
    // The builder routes a comma-separated metric list through params (MET-02). It
    // must accept every new metric name without rejecting the config at build time.
    let list = "quantile,huber,fair,poisson,mape,gamma,gamma_deviance,tweedie,\
                cross_entropy,cross_entropy_lambda,kullback_leibler,\
                average_precision,multi_error,auc_mu";
    let _cfg = TrainingBuilder::new()
        .objective("regression")
        .metric(list)
        .num_iterations(1)
        .num_leaves(2)
        .min_data_in_leaf(1)
        .build()
        .expect("builder must accept the extended metric list");

    // Each new metric name parses through its relevant metric-layer factory
    // (the names the multi-metric list routes to). This is the routing contract:
    // no in-scope metric name is rejected as Unsupported.
    let params = RegressionMetricParams::default();
    for n in [
        "quantile",
        "huber",
        "fair",
        "poisson",
        "mape",
        "gamma",
        "gamma_deviance",
        "tweedie",
    ] {
        assert!(
            Metric::parse_with_params(n, &params).is_ok(),
            "regression metric {n} must parse"
        );
    }
    for n in ["cross_entropy", "cross_entropy_lambda", "kullback_leibler"] {
        assert!(XentropyMetric::parse(n).is_ok(), "xentropy metric {n} must parse");
    }
}

// =========================================================================
// ON-DEVICE metric parity cells (Plan 20-02, ODL-17) — the §12 EvalKernel
// value composed with ConvertOutput (D-04), anchored to the SAME real
// lib_lightgbm 4.6 goldens on the cpu f64 anchor (D-03 / D-05). Only the
// TWELVE device-supported pointwise losses get a device cell; the ranking /
// AUC / multiclass / cross-entropy families stay host-fallback replay (above),
// per the Plan-00 `metric_supported` discriminator.
// =========================================================================

/// Load a golden triplet, run the composed on-device Eval on the cpu f64 anchor,
/// and assert it matches the captured `lib_lightgbm` value within `ORACLE_TOL`
/// (SKIP if the golden is absent). Never GPU-vs-GPU — the cpu anchor is the gate.
fn assert_device_metric(name: &str, kind: DeviceMetricKind, params: MetricEvalParams) {
    let Some((scores, labels, captured)) = load_triplet(name) else {
        return;
    };
    let client = cpu_client();
    let device = eval_metric_on(&client, kind, &scores, &labels, None, params)
        .expect("on-device EvalKernel");
    assert_metric(&format!("device/{name}"), device, captured);
}

// --- pass-through cells (no ConvertOutput): l2 / rmse / l1 / quantile / huber /
//     fair / mape. rmse/l2/l1 ASSERT (Plan 00 captured their goldens). ---

#[test]
fn device_l2_parity() {
    assert_device_metric("l2", DeviceMetricKind::L2, MetricEvalParams::default());
}

#[test]
fn device_rmse_parity() {
    assert_device_metric("rmse", DeviceMetricKind::Rmse, MetricEvalParams::default());
}

#[test]
fn device_l1_parity() {
    assert_device_metric("l1", DeviceMetricKind::L1, MetricEvalParams::default());
}

#[test]
fn device_quantile_parity() {
    // The capture trains with default alpha = 0.9.
    assert_device_metric("quantile", DeviceMetricKind::Quantile, MetricEvalParams::default());
}

#[test]
fn device_huber_parity() {
    assert_device_metric("huber", DeviceMetricKind::Huber, MetricEvalParams::default());
}

#[test]
fn device_fair_parity() {
    assert_device_metric("fair", DeviceMetricKind::Fair, MetricEvalParams::default());
}

#[test]
fn device_mape_parity() {
    assert_device_metric("mape", DeviceMetricKind::Mape, MetricEvalParams::default());
}

// --- exp-ConvertOutput cells (CONVERT_EXP composed BEFORE EvalKernel):
//     poisson / gamma / gamma_deviance / tweedie. ---

#[test]
fn device_poisson_parity() {
    assert_device_metric("poisson", DeviceMetricKind::Poisson, MetricEvalParams::default());
}

#[test]
fn device_gamma_parity() {
    assert_device_metric("gamma", DeviceMetricKind::Gamma, MetricEvalParams::default());
}

#[test]
fn device_gamma_deviance_parity() {
    assert_device_metric(
        "gamma_deviance",
        DeviceMetricKind::GammaDeviance,
        MetricEvalParams::default(),
    );
}

#[test]
fn device_tweedie_parity() {
    assert_device_metric("tweedie", DeviceMetricKind::Tweedie, MetricEvalParams::default());
}

// --- sigmoid-ConvertOutput cell (the binary sigmoid composed BEFORE EvalKernel).
//     binary_logloss ASSERTS (Plan 00 captured its golden with objective binary,
//     sigmoid = 1.0). ---

#[test]
fn device_binary_logloss_parity() {
    assert_device_metric(
        "binary_logloss",
        DeviceMetricKind::BinaryLogloss,
        MetricEvalParams::default(),
    );
}
