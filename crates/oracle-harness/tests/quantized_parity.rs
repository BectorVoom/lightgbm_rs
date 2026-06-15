//! Phase-10 Wave 4 — C++ `use_quantized_grad` parity oracle (harness, dependency-free).
//!
//! Goldens (LightGBM 4.6, `use_quantized_grad=True, stochastic_rounding=False,
//! num_grad_quant_bins=128`, deterministic) under `tests/fixtures/quantized/`:
//!   - `quant_binary.pred`    — one prediction per row (plain text)
//!   - `quant_binary.xy.csv`  — features + label (Wave 3b training input)
//!   - `quant_binary.json`    — full golden (model text, params) for Python/reference
//! Regenerate: `.venv/bin/python crates/oracle-harness/tests/fixtures/quantized/gen_golden.py`
//!
//! `golden_predictions_well_formed` is ACTIVE (proves the oracle is wired + non-degenerate —
//! guards against the int8-overflow stump `num_grad_quant_bins=256` produced). The Rust-vs-C++
//! comparison activates with Wave 3b (the production quantized training path).

use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quantized").join(name)
}

fn read_preds() -> Vec<f64> {
    std::fs::read_to_string(fixture("quant_binary.pred"))
        .expect("quantized golden predictions present")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().expect("pred parses as f64"))
        .collect()
}

/// The oracle exists and carries a NON-DEGENERATE learned model (prediction variance) — the
/// guard that would have caught the `num_grad_quant_bins=256` int8-overflow stump (all preds
/// equal, std 0). Binary-objective probabilities must stay in (0,1).
#[test]
fn golden_predictions_well_formed() {
    let preds = read_preds();
    assert_eq!(preds.len(), 512, "expected 512 golden predictions");
    assert!(preds.iter().all(|&p| (0.0..=1.0).contains(&p)), "binary preds must be in [0,1]");
    let mean = preds.iter().sum::<f64>() / preds.len() as f64;
    let var = preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / preds.len() as f64;
    assert!(var > 1e-3, "golden model is degenerate (pred var {var:.2e}) — regenerate (int8 overflow?)");

    // The xy.csv training input exists with the matching row count (Wave 3b reads it).
    let rows = std::fs::read_to_string(fixture("quant_binary.xy.csv"))
        .expect("xy.csv present")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .count();
    assert_eq!(rows, 512, "xy.csv row count must match predictions");
}

/// Read `quant_binary.xy.csv` → (rows of f64 features, f32 labels).
fn read_xy() -> (Vec<Vec<f64>>, Vec<f32>) {
    let text = std::fs::read_to_string(fixture("quant_binary.xy.csv")).expect("xy.csv present");
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let vals: Vec<f64> = line.split(',').map(|s| s.trim().parse().unwrap()).collect();
        let (feat, lab) = vals.split_at(vals.len() - 1);
        rows.push(feat.to_vec());
        labels.push(lab[0] as f32);
    }
    (rows, labels)
}

/// Wave 3b: train the Rust quantized Booster on the golden's corpus with its params and
/// compare per-row predictions to the C++ `use_quantized_grad` golden. Asserts BOTH absolute
/// parity (exact path f32-bit-faithful; quantized within the approximate residual) AND the
/// quantization-DELTA match. (The earlier ~0.28 absolute gap was a RawCorpus binning-config
/// footgun — `train_raw`'s config now drives binning — NOT a quantization issue.)
#[test]
fn rust_quantized_train_matches_cpp() {
    use lgbm::{train_raw, Config, RawCorpus, TrainingBuilder};

    let (rows, labels) = read_xy();
    let golden = read_preds();
    assert_eq!(rows.len(), golden.len());

    // Match gen_golden.py params exactly.
    let mut cfg: Config = TrainingBuilder::new()
        .objective("binary")
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(7)
        .min_data_in_leaf(5)
        .seed(1)
        .deterministic(true)
        .build()
        .expect("config builds");
    cfg.max_bin = 63;
    cfg.num_threads = 1;
    cfg.force_row_wise = true;
    cfg.feature_pre_filter = false;
    cfg.use_quantized_grad = true;
    cfg.num_grad_quant_bins = 128;
    cfg.stochastic_rounding = false;

    let corpus = RawCorpus::new(rows.clone(), labels.clone());
    let booster = train_raw(&cfg, &corpus).expect("rust quantized train ok");
    let pred: Vec<f32> = booster.predict(&rows).iter().map(|r| r[0]).collect();

    // Diagnostic: Rust EXACT on the same data, vs C++ exact — isolates exact-path parity
    // from quantization-path parity.
    let mut cfg_exact = cfg.clone();
    cfg_exact.use_quantized_grad = false;
    let exact_booster = train_raw(&cfg_exact, &RawCorpus::new(rows.clone(), labels))
        .expect("rust exact train ok");
    let pred_exact: Vec<f32> = exact_booster.predict(&rows).iter().map(|r| r[0]).collect();
    let cpp_exact = read_named_preds("quant_binary.pred_exact");

    let delta = |a: &[f32], b: &[f64]| {
        let (mut mx, mut sm) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (f64::from(*x) - y).abs();
            mx = mx.max(d);
            sm += d;
        }
        (mx, sm / a.len() as f64)
    };
    let (qe_max, qe_mean) = delta(&pred, &golden); // Rust-quant vs C++-quant (absolute)
    let (xe_max, xe_mean) = delta(&pred_exact, &cpp_exact); // Rust-exact vs C++-exact (absolute)
    eprintln!("Rust-exact vs C++-exact (continuous RawCorpus): max={xe_max:.3e} mean={xe_mean:.3e}");
    eprintln!("Rust-quant vs C++-quant (absolute):             max={qe_max:.3e} mean={qe_mean:.3e}");

    // ABSOLUTE parity gates — now meaningful since the RawCorpus binning-config footgun is fixed
    // (train_raw's config drives binning). The exact path is f32-bit-faithful on continuous data;
    // the quantized model matches C++ to the (approximate) quantization residual.
    assert!(xe_max < 1e-5, "exact RawCorpus path diverged from C++: max={xe_max:.3e} (f32 gate)");
    assert!(qe_max < 1e-2, "quantized model diverged from C++ beyond the approximate residual: max={qe_max:.3e}");

    // The QUANTIZATION DELTA is the faithful gate: quantization must perturb the Rust model
    // the same way it perturbs C++ (both relative to their own exact model). This isolates the
    // quantization implementation from the separate, pre-existing exact-path RawCorpus gap.
    let rdelta = pred.iter().zip(pred_exact.iter())
        .map(|(q, e)| f64::from((*q - *e).abs()))
        .fold((0.0f64, 0.0f64), |(mx, sm), d| (mx.max(d), sm + d));
    let (rq_max, rq_mean) = (rdelta.0, rdelta.1 / pred.len() as f64);
    let cdelta = golden.iter().zip(cpp_exact.iter())
        .map(|(q, e)| (q - e).abs())
        .fold((0.0f64, 0.0f64), |(mx, sm), d| (mx.max(d), sm + d));
    let (cq_max, cq_mean) = (cdelta.0, cdelta.1 / golden.len() as f64);
    eprintln!("QUANT EFFECT  Rust |quant-exact|: max={rq_max:.3e} mean={rq_mean:.3e}");
    eprintln!("QUANT EFFECT  C++  |quant-exact|: max={cq_max:.3e} mean={cq_mean:.3e}");

    // Gate: the Rust quantization perturbation matches C++'s in magnitude (same approximate
    // regime). The pre-existing exact-path RawCorpus gap (xe_*) is OUT OF SCOPE for phase-10
    // and is asserted only as a recorded baseline, not a pass/fail.
    assert!(
        rq_mean < 2.0 * cq_mean.max(1e-4) && rq_max < 3.0 * cq_max.max(1e-3),
        "Rust quantization effect (max={rq_max:.3e} mean={rq_mean:.3e}) is not in the same \
         regime as C++'s (max={cq_max:.3e} mean={cq_mean:.3e}) — the quantizer is off"
    );
}

/// W6: leaf renewal (`quant_train_renew_leaf`) recomputes leaf outputs from the original
/// gradients. Verify the Rust renew EFFECT (renew vs no-renew) matches C++'s in magnitude —
/// the same delta-gate logic as the quantization effect (isolates the feature from the
/// pre-existing RawCorpus exact-path gap).
#[test]
fn rust_quant_renew_leaf_matches_cpp_effect() {
    use lgbm::{train_raw, Config, RawCorpus, TrainingBuilder};

    let (rows, labels) = read_xy();
    let mut cfg: Config = TrainingBuilder::new()
        .objective("binary").num_iterations(10).learning_rate(0.1)
        .num_leaves(7).min_data_in_leaf(5).seed(1).deterministic(true)
        .build().unwrap();
    cfg.max_bin = 63;
    cfg.num_threads = 1;
    cfg.force_row_wise = true;
    cfg.feature_pre_filter = false;
    cfg.use_quantized_grad = true;
    cfg.num_grad_quant_bins = 128;
    cfg.stochastic_rounding = false;

    let pred_q: Vec<f32> = train_raw(&cfg, &RawCorpus::new(rows.clone(), labels.clone()))
        .unwrap().predict(&rows).iter().map(|r| r[0]).collect();
    let mut cfg_r = cfg.clone();
    cfg_r.quant_train_renew_leaf = true;
    let pred_qr: Vec<f32> = train_raw(&cfg_r, &RawCorpus::new(rows.clone(), labels))
        .unwrap().predict(&rows).iter().map(|r| r[0]).collect();

    let rust_eff = pred_qr.iter().zip(pred_q.iter())
        .map(|(a, b)| f64::from((*a - *b).abs()))
        .fold((0.0f64, 0.0f64), |(mx, sm), d| (mx.max(d), sm + d));
    let (re_max, re_mean) = (rust_eff.0, rust_eff.1 / pred_q.len() as f64);

    let cpp_q = read_preds();
    let cpp_r = read_named_preds("quant_binary.pred_renew");
    let cpp_eff = cpp_r.iter().zip(cpp_q.iter())
        .map(|(a, b)| (a - b).abs())
        .fold((0.0f64, 0.0f64), |(mx, sm), d| (mx.max(d), sm + d));
    let (ce_max, ce_mean) = (cpp_eff.0, cpp_eff.1 / cpp_q.len() as f64);
    eprintln!("RENEW EFFECT  Rust |renew-quant|: max={re_max:.3e} mean={re_mean:.3e}");
    eprintln!("RENEW EFFECT  C++  |renew-quant|: max={ce_max:.3e} mean={ce_mean:.3e}");

    // Renew must (a) actually change predictions and (b) in C++'s regime (small refinement).
    assert!(re_max > 1e-6, "renew had no effect — not wired");
    assert!(
        re_mean < 5.0 * ce_mean.max(1e-4) && re_max < 5.0 * ce_max.max(1e-3),
        "Rust renew effect (max={re_max:.3e}) not in C++'s regime (max={ce_max:.3e})"
    );
}

fn read_named_preds(name: &str) -> Vec<f64> {
    std::fs::read_to_string(fixture(name))
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect()
}
