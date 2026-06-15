//! Spike 008 — CPU parity probe for the 16-bit DISCRETIZED histogram (Lever 3).
//!
//! LightGBM's discretized build (`CUDAConstructDiscretizedHistogramDenseKernel` +
//! `cuda_gradient_discretizer.cu`) is an OPT-IN approximate mode (`use_quantized_grad`,
//! default FALSE): each row's (grad, hess) is quantized to int16 via
//! `q = round(value * scale)`, `scale = (bins/2) / abs_max`, packed into one int32, and
//! accumulated with a SINGLE integer atomic (the speed win: one int atomic vs two f32).
//!
//! The speed is real — but the gating question for lightgbm_rs (whose core contract is
//! ~1e-6 parity to C++) is the PARITY ENVELOPE of the quantization itself. This is a CPU
//! probe (no GPU needed): quantize, build the int histogram, de-quantize, and compare to
//! the exact f64 histogram. If the drift is ≫ the 1e-5 f32 gate, the discretized path can
//! only ever be a SEPARATE approximate mode, never a drop-in for the exact build.
//!
//! Run: cargo run --release --example quant_parity_probe   (no --features needed)

fn main() {
    // Deterministic LCG (Date/rand are unavailable in some sandboxes; keep it self-contained).
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let n = 200_000usize;
    let num_bin = 64usize;

    // Realistic BINARY-logloss gradients: g = pred - label ∈ (-1, 1), h = pred(1-pred) ∈ (0, 0.25].
    // (Bounded + the common case; L2 regression grads are unbounded and quantize even worse.)
    let mut grad = vec![0.0f32; n];
    let mut hess = vec![0.0f32; n];
    let mut binned = vec![0u32; n];
    for i in 0..n {
        let u = (next() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        let label = if (next() & 1) == 0 { 0.0f32 } else { 1.0f32 };
        let pred = u; // pretend current sigmoid output
        grad[i] = pred - label; // ∈ (-1, 1)
        hess[i] = (pred * (1.0 - pred)).max(1e-6); // ∈ (0, 0.25]
        binned[i] = (next() % num_bin as u64) as u32;
    }

    // Exact f64 histogram (the anchor).
    let mut exact = vec![0.0f64; 2 * num_bin];
    for i in 0..n {
        let c = binned[i] as usize * 2;
        exact[c] += f64::from(grad[i]);
        exact[c + 1] += f64::from(hess[i]);
    }

    let grad_abs_max = grad.iter().fold(0.0f32, |m, &g| m.max(g.abs()));
    let hess_abs_max = hess.iter().fold(0.0f32, |m, &h| m.max(h.abs()));

    // Max relative error over the cells whose exact magnitude is meaningful.
    let max_rel = |q: &[f64]| -> f64 {
        exact
            .iter()
            .zip(q.iter())
            .map(|(e, x)| (e - x).abs() / e.abs().max(1.0))
            .fold(0.0, f64::max)
    };

    println!("# quant parity probe  n={n} num_bin={num_bin}  binary-logloss grads");
    println!("# grad_abs_max={grad_abs_max:.4} hess_abs_max={hess_abs_max:.4}");
    println!("# LightGBM default num_grad_quant_bins=4; use_quantized_grad=FALSE (opt-in approx mode).");
    println!("# project exact gate: ~1e-6 (CPU anchor) / <1e-5 (f32 GPU). Compare:\n");
    println!("{:>12} | {:>14} | {:>14} | {:>10}", "quant_bins", "rel(determ.)", "rel(stochastic)", "vs 1e-5");

    for &bins in &[4usize, 16, 256, 4096, 65536] {
        let gscale = (bins as f32 / 2.0) / grad_abs_max;
        let hscale = (bins as f32) / hess_abs_max;

        // Deterministic round-to-nearest.
        let mut det = vec![0.0f64; 2 * num_bin];
        // Stochastic rounding (unbiased): round up with prob = frac.
        let mut sto = vec![0.0f64; 2 * num_bin];
        for i in 0..n {
            let c = binned[i] as usize * 2;
            let gq = (grad[i] * gscale).round() as i64;
            let hq = (hess[i] * hscale).round() as i64;
            det[c] += gq as f64 / f64::from(gscale);
            det[c + 1] += hq as f64 / f64::from(hscale);

            let gf = grad[i] * gscale;
            let hf = hess[i] * hscale;
            let r1 = (next() >> 40) as f32 / (1u64 << 24) as f32;
            let r2 = (next() >> 40) as f32 / (1u64 << 24) as f32;
            let gqs = gf.floor() as i64 + i64::from((gf - gf.floor()) > r1);
            let hqs = hf.floor() as i64 + i64::from((hf - hf.floor()) > r2);
            sto[c] += gqs as f64 / f64::from(gscale);
            sto[c + 1] += hqs as f64 / f64::from(hscale);
        }

        let rd = max_rel(&det);
        let rs = max_rel(&sto);
        let verdict = if rd.min(rs) < 1e-5 { "PASS" } else { "FAIL" };
        println!("{bins:>12} | {rd:>14.3e} | {rs:>14.3e} | {verdict:>10}");
    }
    println!("\n# A row sums into a bin of ~{} rows; quantization step = abs_max/(bins/2).", n / num_bin);
    println!("# 'FAIL' = the quantization drift exceeds the project's exact-parity gate at that bin count.");
}
