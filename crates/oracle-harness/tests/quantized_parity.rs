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

/// ACTIVATE in Wave 3b: train the Rust quantized Booster from `quant_binary.xy.csv` with the
/// golden's params and assert per-row predictions match `quant_binary.pred` within the
/// quantized contract (NOT the exact ~1e-6 anchor — spike-008).
#[test]
#[ignore = "Wave 3b: needs the production Rust quantized training path"]
fn rust_quantized_train_matches_cpp() {
    let _golden = read_preds();
    unimplemented!("activated by Wave 3b: train Rust quantized, compare to _golden");
}
