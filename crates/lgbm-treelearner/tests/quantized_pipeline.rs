//! Phase-10 Wave 3a: end-to-end check that the quantized build pipeline
//! (discretize → integer histogram → de-quant) feeds the REAL split-gain machinery and
//! recovers the exact path's split decision within the quantization envelope (spike-008).
//!
//! This is the numeric pipeline proof; production GBDT-loop/backend wiring is Wave 3b, and
//! parity to C++ `use_quantized_grad` goldens is Wave 4.

use lgbm_compute::gain::get_split_gains;
use lgbm_treelearner::gradient_discretizer::{construct_int_histogram, GradientDiscretizer};

/// Scan a stride-2 `[g0,h0,…]` histogram for the best left/right split (no L1/L2),
/// returning (threshold_bin, gain). Plain prefix-sum scan — the same shape the real
/// split-finder uses, exercised here on the de-quantized histogram.
fn best_split(hist: &[f64], num_bin: usize, total_g: f64, total_h: f64) -> (usize, f64) {
    let mut best = (0usize, f64::NEG_INFINITY);
    let (mut lg, mut lh) = (0.0f64, 0.0f64);
    for t in 0..num_bin - 1 {
        lg += hist[t * 2];
        lh += hist[t * 2 + 1];
        let (rg, rh) = (total_g - lg, total_h - lh);
        let gain = get_split_gains(false, lg, lh, rg, rh, 0.0, 0.0);
        if gain > best.1 {
            best = (t, gain);
        }
    }
    best
}

#[test]
fn quantized_pipeline_recovers_exact_split() {
    let num_bin = 8usize;
    let n = 800usize;
    // 1 feature, gradient trends with the bin → an unambiguous split at bin 3 (low|high).
    let binned: Vec<u32> = (0..n).map(|i| (i % num_bin) as u32).collect();
    let grad: Vec<f32> = binned.iter().map(|&b| (b as f32 - 3.5) * 0.5).collect();
    let hess = vec![1.0f32; n]; // constant hessian

    // Exact f64 histogram + its best split.
    let mut exact = vec![0.0f64; 2 * num_bin];
    for i in 0..n {
        let c = binned[i] as usize * 2;
        exact[c] += f64::from(grad[i]);
        exact[c + 1] += f64::from(hess[i]);
    }
    let tg: f64 = grad.iter().map(|&x| f64::from(x)).sum();
    let th = n as f64;
    let (exact_bin, exact_gain) = best_split(&exact, num_bin, tg, th);

    // Quantized pipeline at 254 levels (max for i8 discretized storage; ~1e-2 rel envelope).
    let mut d = GradientDiscretizer::new(254, true);
    let q = d.discretize(&grad, &hess);
    let rows: Vec<u32> = (0..n as u32).collect();
    let int_h = construct_int_histogram(&binned, &q, &rows, num_bin as u32);
    let deq = d.dequantize_histogram(&int_h);
    let dtg: f64 = (0..num_bin).map(|b| deq[b * 2]).sum();
    let dth: f64 = (0..num_bin).map(|b| deq[b * 2 + 1]).sum();
    let (q_bin, q_gain) = best_split(&deq, num_bin, dtg, dth);

    // Same split decision as exact…
    assert_eq!(q_bin, exact_bin, "quantized split bin {q_bin} != exact {exact_bin}");
    assert_eq!(exact_bin, 3, "sanity: the unambiguous split is at bin 3");
    // …and the gain within the quantization envelope (well under 5%).
    let rel = (q_gain - exact_gain).abs() / exact_gain.abs();
    assert!(rel < 0.05, "quantized gain {q_gain} vs exact {exact_gain} (rel {rel:.3e})");
    // Constant-hessian de-quant is exact (each bin's hess sum = its row count).
    assert!((dth - th).abs() < 1e-9, "constant-hessian totals must match exactly");
}
