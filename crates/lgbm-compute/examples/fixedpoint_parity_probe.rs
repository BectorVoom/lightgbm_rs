//! Spike 018a — research Q2 parity GATE (cheap CPU host probe, NO GPU; spike-008 method).
//!
//! Finding #3 proposed fixed-point INTEGER atomic accumulation of the histogram bins.
//! Its SPEED claim is already falsified on this hardware (phase-10 W5: packed-int / f32 =
//! 1.00–1.01×, the LDS build is contention/latency-bound, not atomic-count/-type bound).
//! The ONLY live rationale left is DETERMINISM: integer add is order-independent, so a
//! fixed-point GPU histogram would be reproducible run-to-run (the f32-atomic path is NOT
//! — the root of def-f8u-01, which had to pin GPU trees to the CPU anchor).
//!
//! This probe answers the GATE question, purely on CPU, before any GPU plumbing:
//!   Q: Can fixed-point i64 accumulation of the EXACT f32 grads (scale S, dequantize
//!      sum/S) stay within the ~1e-6 ROCm gate of the f64 anchor — INCLUDING the
//!      cancelling-gradient regime — AND beat the f32-atomic envelope, while being
//!      bit-identical regardless of summation order (determinism)?
//!
//! Unlike spike-008 (int16 PACKED quantization → 3.2e-4, invalidated), this is WIDE
//! fixed-point: each g_i → round(g_i * S) as an i64; the accumulation itself is EXACT
//! integer arithmetic, so the only error is the per-element quantization (≤0.5/S), which
//! a large S drives below the gate. i32 is shown to OVERFLOW (why i64 is required).
//!
//! Run: cargo run --release -p lgbm-compute --example fixedpoint_parity_probe

fn main() {
    // Representative single-bin accumulations (the histogram cell is a sum over all rows
    // that fall in one bin). N up to 1M = a hot bin on a large leaf.
    const N: usize = 1_000_000;

    // --- gradient distributions (f32, matching score_t) ---
    // (a) "residual" — signed, O(0.5) magnitude, partial cancellation (the test regime).
    let g_resid: Vec<f32> = (0..N).map(|i| ((i as f32 * 0.013).sin() * 0.5)).collect();
    // (b) "cancelling" — near-zero net sum (the catastrophic-f32 regime spike-008/p93 hit).
    let g_cancel: Vec<f32> = (0..N)
        .map(|i| (if i % 2 == 0 { 0.5f32 } else { -0.5f32 }) + (i as f32 * 0.0007).sin() * 1e-3)
        .collect();
    // (c) "hessian-like" — all positive ~1.0 (large net sum, magnitude stress).
    let g_hess: Vec<f32> = (0..N).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();

    let cases: [(&str, &Vec<f32>); 3] =
        [("residual ±0.5", &g_resid), ("cancelling ~0", &g_cancel), ("hessian +1", &g_hess)];

    // f64 ground-truth anchor (ascending, exactly what the cpu f64 fold computes).
    let anchor = |g: &[f32]| -> f64 { g.iter().map(|&x| x as f64).sum() };

    // f32 sequential sum (naive ascending) — and a SHUFFLED-order f32 sum to emulate the
    // nondeterministic atomic accumulation order on the GPU (the f32-atomic envelope).
    let f32_seq = |g: &[f32]| -> f32 { g.iter().copied().fold(0.0f32, |a, b| a + b) };
    let f32_shuf = |g: &[f32], seed: u64| -> f32 {
        // deterministic LCG permutation-ish: sum in a strided/scrambled order.
        let mut acc = 0.0f32;
        let mut idx = (seed % g.len() as u64) as usize;
        let step = 2_654_435_761u64 as usize | 1;
        for _ in 0..g.len() {
            acc += g[idx];
            idx = (idx + step) % g.len();
        }
        acc
    };

    // Fixed-point accumulation at scale S: q_i = round(g_i * S) as i64 (or i32), sum exact,
    // dequantize = sum as f64 / S. Order-INDEPENDENT (integer add is associative).
    let fixed_i64 = |g: &[f32], s: f64| -> (f64, bool) {
        let mut sum: i64 = 0;
        let mut overflow = false;
        for &x in g {
            let q = (x as f64 * s).round() as i64;
            match sum.checked_add(q) {
                Some(v) => sum = v,
                None => {
                    overflow = true;
                    break;
                }
            }
        }
        (sum as f64 / s, overflow)
    };
    let fixed_i32 = |g: &[f32], s: f64| -> (f64, bool) {
        let mut sum: i32 = 0;
        let mut overflow = false;
        for &x in g {
            let q = (x as f64 * s).round() as i32;
            match sum.checked_add(q) {
                Some(v) => sum = v,
                None => {
                    overflow = true;
                    break;
                }
            }
        }
        (sum as f64 / s, overflow)
    };

    let rel = |v: f64, a: f64| -> f64 { (v - a).abs() / a.abs().max(1.0) };

    // i64 scale sweep: 2^16 .. 2^34 (S=2^30 ≈ 1.07e9; sum ≤ 1e6 * |g|·S ≤ ~1e15 ≪ i64 max).
    let scales: [u32; 5] = [16, 20, 24, 30, 34];

    println!("# Spike 018a — fixed-point parity probe (CPU host, N={N})");
    println!("# gate = ~1e-6 ROCm contract vs the f64 anchor. f32-atomic envelope = the GPU's CURRENT error.");
    println!("# det? = fixed-point is order-independent (integer add) ⇒ bit-identical under shuffle.\n");

    for (name, g) in cases {
        let a = anchor(g);
        let seq = f32_seq(g) as f64;
        let sh1 = f32_shuf(g, 1) as f64;
        let sh2 = f32_shuf(g, 7) as f64;
        let env = rel(sh1, a).max(rel(sh2, a)).max(rel(seq, a)); // f32-atomic envelope
        println!("## {name:<14}  anchor={a:.6e}");
        println!(
            "  f32 seq        rel={:.2e}   |  f32-atomic envelope (shuffled orders) rel≈{:.2e}  [GPU's current error, NONDET]",
            rel(seq, a),
            env
        );
        // i32 at a mid scale to demonstrate overflow on a hot bin.
        let (v32, ov32) = fixed_i32(g, (1u64 << 20) as f64);
        println!(
            "  i32  S=2^20    {}",
            if ov32 { "OVERFLOW (why i64 is required for large leaves)".to_string() } else { format!("rel={:.2e}", rel(v32, a)) }
        );
        for &p in &scales {
            let s = (1u64 << p) as f64;
            let (v, ov) = fixed_i64(g, s);
            // determinism check: a shuffled-order fixed-point sum must be BIT-IDENTICAL.
            let mut sum_shuf: i64 = 0;
            let mut idx = 3usize % g.len();
            let step = 2_654_435_761u64 as usize | 1;
            for _ in 0..g.len() {
                sum_shuf += (g[idx] as f64 * s).round() as i64;
                idx = (idx + step) % g.len();
            }
            let det = (sum_shuf as f64 / s) == v;
            let r = rel(v, a);
            let gate = if r <= 1e-6 { "PASS" } else { "FAIL" };
            println!(
                "  i64  S=2^{p:<2}    rel={r:.2e}  gate({gate})  det={}  {}",
                if det { "yes" } else { "NO" },
                if ov { "OVERFLOW" } else { "" }
            );
        }
        println!();
    }
    println!("# Verdict logic: if some feasible i64 scale PASSES the ~1e-6 gate on ALL cases");
    println!("# (esp. 'cancelling ~0') AND beats the f32-atomic envelope AND det=yes, then");
    println!("# fixed-point is a DETERMINISTIC, gate-clearing accumulation — the parity gate opens.");
    println!("# (Speed is separately NULL per W5; determinism is the only live rationale.)");
}
