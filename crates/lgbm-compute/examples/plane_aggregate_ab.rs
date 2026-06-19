//! A/B micro-bench: warp-aggregated (`use_plane=true`) vs baseline non-plane
//! (`use_plane=false`) f32-atomic histogram scatter on the local gfx1100 — quick
//! task 260619-p93 (QUICK-260619-P93).
//!
//! WHAT THIS QUANTIFIES: the one genuinely un-benched GPU histogram CONTENTION
//! lever — warp-aggregating the global f32 `fetch_add`s so that within-plane
//! same-bin adds collapse into ONE global atomic per distinct bin per plane
//! (`construct_hist_kernel_atomic_f32_plane`, histogram.rs). The two arms are the
//! SAME launcher (`construct_histograms_parallel_f32_plane_on`) toggling ONLY the
//! `use_plane` comptime flag, so the uploads / cube count / readback are IDENTICAL
//! and the delta isolates EXACTLY the warp-aggregation codegen. (No `_checked` twin
//! file is needed here — the two comptime specializations ARE the twins; the same-
//! input assert below is the runtime drift guard. Any future edit to the plane
//! kernel body changes both arms together by construction.)
//!
//! EXPECTED RESULT — likely NULL (RESEARCH; 3 corroborating in-repo findings):
//! the global-atomic kernel is contention / scattered-read-latency bound
//! (gpu-hist-levers-closed, spike-006, 260619-ol8), and at 256 bins a 32-lane wave
//! hits ~30 DISTINCT bins, so there is almost nothing to amortize — the ballot/
//! shuffle aggregation overhead is then pure cost. Any win, if it exists, is
//! confined to the LOW-bin (16/64) regime where lanes collide. **The bin sweep is
//! therefore the PRIMARY independent variable** (the collision-rate axis is the
//! whole point of this lever). A credible benched NULL kept as a rocm-gated
//! PRIMITIVE (NOT wired) is a fully valid outcome — same disposition as ngo/ol8/j9t.
//! Do NOT manufacture a win.
//!
//! MEASUREMENT-ONLY: the plane kernel + launcher are a rocm-gated primitive; the
//! production histogram path, the wired LDS/resident kernels, and the CPU f64
//! anchor are UNTOUCHED. The only artifacts of this task are this example, the
//! re-pin test (`tests/rocm_plane_aggregate.rs`), the kernel/launcher (Task 1), and
//! a FINDINGS.md.
//!
//! WARM-VS-COLD (load-bearing, spike-findings SKILL — cold-ceiling-overstates-warm):
//! WARMUP discarded launches per arm, MEDIAN of N timed launches, a device sync
//! (result read-back) FORCED INSIDE every timed call, baseline/plane arms
//! INTERLEAVED so thermal/clock drift hits both equally, and the bench is meant to
//! be RE-RUN across >=2 process restarts to confirm the delta SIGN is stable vs
//! noise. A delta within the p25/p75 spread, or whose sign flips across runs, is
//! SUB-NOISE / NULL (the ol8 disposition rule).
//!
//! This is a MANUAL rocm bench — NOT a test gate.
//!
//! Run: cargo run --release -p lgbm-compute --features rocm --example plane_aggregate_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100). Re-run with it.");
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::kernels::histogram::construct_histograms_parallel_f32_plane_on;
    use lgbm_compute::runtime::{probe_capabilities, rocm_client};
    use std::time::Instant;

    let client = rocm_client();

    // The plane arm REQUIRES plane collectives — gate on the existing probe (never
    // re-rolled). On gfx1100 has_plane is true; bail loudly otherwise so a NULL is
    // never silently reported on a device that can't run the lever.
    let caps = probe_capabilities(&client);
    assert!(
        caps.has_plane,
        "plane_aggregate_ab requires has_plane (gfx1100 wave32); probed {caps:?}"
    );
    println!("# probed capabilities: {caps:?}");

    // Median helper (sort, take middle) — same as the existing harnesses.
    let median = |xs: &mut [f64]| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };
    // p25 / p75 spread of an arm (the reader judges significance against this).
    let spread = |xs: &[f64]| -> (f64, f64) {
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = s[s.len() / 4];
        let hi = s[(s.len() * 3) / 4];
        (lo, hi)
    };

    // Same-input drift guard — this is a GPU-f32-vs-GPU-f32 sanity/drift check, NOT
    // the parity gate (the parity gate pins the plane variant to the CPU f64 ANCHOR
    // in tests/rocm_plane_aggregate.rs; per DEF-f8u-01 two nondeterministic GPU f32
    // paths must never be held to a 1e-6 gate against EACH OTHER, only to the f64
    // anchor). At a SMALL leaf (≤ a few thousand rows) the baseline (per-row atomic
    // chain) and plane (same-bin tree reduction) arms agree tightly (ABS 5e-6 / REL
    // 1e-5). At a LARGE leaf (200k rows into few bins) BOTH arms' f32 cells sum
    // thousands of near-cancelling residual gradients, so the two DIFFERENT f32
    // accumulation orders legitimately diverge ~few-e-5 rel from each other (the
    // documented large-leaf f32-cancellation effect — RESEARCH §parity-assessment;
    // measured on gfx1100: baseline & plane BOTH drift ~3e-4 from the f64 anchor at
    // n=50k/16 bins). A divergence ABOVE this leaf-aware bound would indicate the
    // plane aggregation is genuinely WRONG (e.g. a naive plane_sum mis-routing adds,
    // RESEARCH Pitfall 1). `rel_tol` is widened for the compute-bound regime ONLY as
    // a drift guard; it is NOT a parity gate and does NOT relax any test tolerance.
    let assert_same_input_f32 = |a: &[f64], b: &[f64], rel_tol: f64, cell: &str| {
        const ABS: f64 = 5e-6;
        assert_eq!(a.len(), b.len(), "{cell}: histogram length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let diff = (x - y).abs();
            let tol = ABS + rel_tol * x.abs();
            assert!(
                diff <= tol,
                "{cell}: cell {i} diverged beyond the leaf-aware f32 drift bound \
                 (rel_tol={rel_tol:e}) — baseline {x} vs plane {y} (|diff| {diff} > tol {tol}); \
                 the plane same-bin aggregation may be incorrect (RESEARCH Pitfall 1)"
            );
        }
    };

    // Deterministic generators (no RNG; same hash family as the other harnesses).
    let gen_grad = |n: usize| -> Vec<f32> { (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect() };
    let gen_hess = |n: usize| -> Vec<f32> { (0..n).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect() };

    // The bin sweep is the PRIMARY independent variable (collision-rate axis).
    const BIN_SWEEP: [u32; 3] = [16, 64, 256];

    println!("# plane (warp-aggregated atomics) vs baseline (per-row global atomics) A/B  —  gfx1100, 260619-p93");
    println!("# baseline arm = construct_histograms_parallel_f32_plane_on(.., use_plane=false)  (per-row fetch_add)");
    println!("# plane    arm = construct_histograms_parallel_f32_plane_on(.., use_plane=true)   (same-bin warp aggregate)");
    println!("# SAME launcher both arms (only the comptime use_plane flag differs) ⇒ uploads/cubes/readback identical.");
    println!("# INTERLEAVED baseline/plane per timed iter; WARMUP discarded; median + p25/p75 spread.");
    println!("# device sync FORCED inside each timed call (read_one_unchecked read-back).");
    println!("# delta% = (baseline - plane) / baseline * 100   (>0 => plane faster)");
    println!("# RE-RUN across >=2 process restarts to confirm the delta SIGN is stable vs noise.");
    println!("# EXPECTED: NULL — at 256 bins a 32-lane wave hits ~30 distinct bins (nothing to amortize);");
    println!("#   the kernel is contention/scattered-read-latency bound (gpu-hist-levers-closed, spike-006, ol8).");
    println!();

    println!("## f32-atomic histogram: warp-aggregate vs per-row");
    println!(
        "{:>13} | {:>5} | {:>11} | {:>11} | {:>8} | {:>17} | {:>17}",
        "regime", "bins", "base_ms", "plane_ms", "delta%", "base p25/p75", "plane p25/p75"
    );
    println!("{:-<14}+{:-<7}+{:-<13}+{:-<13}+{:-<10}+{:-<19}+{:-<19}", "", "", "", "", "", "", "");

    // (regime label, n rows, warmup, timed). Launch-bound: small n, many cheap
    // launches. Compute-bound: large n so the 2*n scatter dominates.
    let cells: [(&str, usize, usize, usize); 2] = [
        ("launch-bound", 2_048, 15, 21),
        ("compute-bound", 200_000, 5, 7),
    ];
    for (regime, n, warmup, timed) in cells {
        for &num_bin in &BIN_SWEEP {
            // binned[i] in [0, num_bin); deterministic hash (same as launch_unchecked_ab.rs).
            let binned: Vec<u32> = (0..n)
                .map(|i| {
                    let h = (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(0x9E37_79B9);
                    (h % num_bin as u64) as u32
                })
                .collect();
            let grad = gen_grad(n);
            let hess = gen_hess(n);

            let baseline_call =
                || construct_histograms_parallel_f32_plane_on(&client, &binned, &grad, &hess, num_bin, false).unwrap();
            let plane_call =
                || construct_histograms_parallel_f32_plane_on(&client, &binned, &grad, &hess, num_bin, true).unwrap();

            // Interleaved warmup.
            for _ in 0..warmup {
                let _ = baseline_call();
                let _ = plane_call();
            }
            let mut b_ms = Vec::with_capacity(timed);
            let mut p_ms = Vec::with_capacity(timed);
            let mut b_hist = Vec::new();
            let mut p_hist = Vec::new();
            for _ in 0..timed {
                let t = Instant::now();
                let bh = baseline_call();
                b_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let ph = plane_call();
                p_ms.push(t.elapsed().as_secs_f64() * 1e3);
                b_hist = bh;
                p_hist = ph;
            }
            let cell = format!("plane {regime} bins={num_bin}");
            // Leaf-aware drift bound (NOT the parity gate — see assert def): the
            // small launch-bound leaf agrees tightly (1e-5); the 200k-row compute-
            // bound leaf legitimately diverges more between the two f32 orders due to
            // f32 cancellation (both arms ~3e-4 from the f64 anchor), so the f32-vs-
            // f32 drift guard there is loosened to 1e-3 — still tight enough to catch
            // a genuinely mis-routed aggregation (which would be O(1) wrong).
            let rel_tol = if n >= 100_000 { 1e-3 } else { 1e-5 };
            assert_same_input_f32(&b_hist, &p_hist, rel_tol, &cell);
            let b_med = median(&mut b_ms);
            let p_med = median(&mut p_ms);
            let (bl, bh_) = spread(&b_ms);
            let (pl, ph_) = spread(&p_ms);
            let delta = (b_med - p_med) / b_med * 100.0;
            println!(
                "{:>13} | {:>5} | {:>11.4} | {:>11.4} | {:>7.2} | {:>7.4}/{:>7.4} | {:>7.4}/{:>7.4}",
                regime, num_bin, b_med, p_med, delta, bl, bh_, pl, ph_
            );
        }
    }

    println!();
    println!("# LEGEND: delta% > 0 => plane (warp-aggregated) faster; < 0 => baseline (per-row) faster.");
    println!("# The same-input asserts above passed (f32 envelope) ⇒ the plane aggregation is CORRECT.");
    println!("# A delta within the p25/p75 spread, or whose SIGN flips across process restarts, is");
    println!("#   SUB-NOISE / NULL — re-run >=2 times and compare. Per RESEARCH the win is expected NULL");
    println!("#   (contention-bound; ~30 distinct bins per 32-lane wave at 256 bins). MEASUREMENT ONLY.");
}
