//! Spike 024 — batch-sibling-scans: ISOLATED launch+readback A/B.
//!
//! Spike-023 (the re-profile kill-check) found the post-021 GPU per-tree cost is
//! ~59 scan-readback SYNCS/tree = ONE per leaf-node, i.e. the TWO siblings of every
//! split are scanned in TWO separate launches + TWO separate readbacks. At small/medium
//! the genuine scan-readback SYNC is ~48%/35% of the scan round-trip (~44 µs/sync ≈ pure
//! fixed launch+sync latency at small). This spike isolates the structural question
//! BEFORE plumbing the growth loop (CONVENTIONS: "probe in an isolated A/B before
//! plumbing a multi-kernel change" — it killed 006/008/009/011 cheaply):
//!
//!   BASELINE (A, = current production): scan sibling-A then sibling-B as TWO separate
//!     `scan_one` launches, each with its own `read_one_unchecked` ⇒ 2 launches, 2 syncs.
//!   CANDIDATE (B, = co-packed): ONE `scan_two` launch over BOTH siblings' histograms
//!     (2×n_feats feature-slots, global lane g<n ⇒ sibling-A feat g, g≥n ⇒ sibling-B
//!     feat g−n), ONE `read_one_unchecked` ⇒ 1 launch, 1 sync.
//!
//! Both arms run the SAME per-feature sequential reverse+forward best-split scan over a
//! DISJOINT histogram region (no reorder — the spike-021 feature-per-lane invariant), so
//! A and B compute BIT-IDENTICAL (best_gain, best_bin) per feature: the ratio isolates
//! the launch/readback STRUCTURE (1 sync vs 2), not the split math. CORRECTNESS GATE:
//! B[0..n] == A_a and B[n..2n] == A_b, byte-for-byte.
//!
//! The two siblings carry DIFFERENT leaf totals (smaller built, larger subtract-derived)
//! ⇒ hist_a and hist_b are generated from different seeds. Feature-per-lane W=64 (the
//! SHIPPED spike-021 packing). Sweep covers the 023 regimes: n∈{12,30,128,512} (small→
//! wide), num_bin∈{32,128} (launch-bound vs compute-bound).
//!
//! Spoofed 8-CU gfx1152 APU ⇒ judge the SIGN of the ratio (co-pack saves one fixed sync
//! per pair; the win should be largest where per-scan compute is smallest = small/few-bin,
//! and shrink toward 1.0 as each scan's own compute dominates the shared launch). Run:
//!   cargo run --release -p lgbm-compute --features rocm --example spike024_sibling_scan_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("requires --features rocm (gfx1100/APU).");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

// Representative per-feature best-split scan, inlined directly in each kernel (the 022b
// precedent — the cube macro supports neither a #[cube] helper here nor a macro_rules!
// expansion inside the body). Forward sweep: gain = left_g²/(left_h+λ)+right_g²/(right_h+λ);
// writes (best_gain, best_bin). Proportional to num_bin ⇒ faithful per-scan compute.

/// 1-slot scan (the BASELINE arm, run once per sibling). Lane g = ABSOLUTE_POS scans
/// feature g of `hist`. Feature-per-lane: CubeDim=W, CubeCount=ceil(n_feats/W).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn scan_one(hist: &Array<f64>, out: &mut Array<f64>, num_bin: u32, n_feats: u32) {
    let g = ABSOLUTE_POS as u32;
    if g < n_feats {
        let base = (g * num_bin * 2u32) as usize;
        let nb = num_bin as usize;
        let mut tot_g = 0.0f64;
        let mut tot_h = 0.0f64;
        for b in 0..nb {
            tot_g += hist[base + b * 2];
            tot_h += hist[base + b * 2 + 1];
        }
        let mut acc_g = 0.0f64;
        let mut acc_h = 0.0f64;
        let mut best_gain = 0.0f64; // all gains >=0 => 0.0 sentinel (022b)
        let mut best_bin = 0u32;
        for b in 0..nb {
            acc_g += hist[base + b * 2];
            acc_h += hist[base + b * 2 + 1];
            let right_g = tot_g - acc_g;
            let right_h = tot_h - acc_h;
            let gain = acc_g * acc_g / (acc_h + 1.0) + right_g * right_g / (right_h + 1.0);
            if gain > best_gain {
                best_gain = gain;
                best_bin = b as u32;
            }
        }
        out[(g * 2u32) as usize] = best_gain;
        out[(g * 2u32 + 1) as usize] = f64::cast_from(best_bin);
    }
}

/// 2-slot scan (the CANDIDATE arm, ONE launch for both siblings). Global lane g; g<n ⇒
/// sibling-A feature g; n≤g<2n ⇒ sibling-B feature (g−n). Both write out[g*2], so
/// out[0..n*2] = sibling A, out[n*2..2n*2] = sibling B. CubeDim=W, CubeCount=ceil(2n/W).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scan_two(
    hist_a: &Array<f64>,
    hist_b: &Array<f64>,
    out: &mut Array<f64>,
    num_bin: u32,
    n_feats: u32,
) {
    let g = ABSOLUTE_POS as u32;
    let total = 2u32 * n_feats;
    if g < total {
        // Local feature index within the sibling; pick which histogram below.
        let fi = if g < n_feats { g } else { g - n_feats };
        let base = (fi * num_bin * 2u32) as usize;
        let nb = num_bin as usize;
        let mut tot_g = 0.0f64;
        let mut tot_h = 0.0f64;
        for b in 0..nb {
            // branch the READ by sibling (Array refs can't be selected into a binding).
            if g < n_feats {
                tot_g += hist_a[base + b * 2];
                tot_h += hist_a[base + b * 2 + 1];
            } else {
                tot_g += hist_b[base + b * 2];
                tot_h += hist_b[base + b * 2 + 1];
            }
        }
        let mut acc_g = 0.0f64;
        let mut acc_h = 0.0f64;
        let mut best_gain = 0.0f64; // all gains >=0 => 0.0 sentinel (022b)
        let mut best_bin = 0u32;
        for b in 0..nb {
            if g < n_feats {
                acc_g += hist_a[base + b * 2];
                acc_h += hist_a[base + b * 2 + 1];
            } else {
                acc_g += hist_b[base + b * 2];
                acc_h += hist_b[base + b * 2 + 1];
            }
            let right_g = tot_g - acc_g;
            let right_h = tot_h - acc_h;
            let gain = acc_g * acc_g / (acc_h + 1.0) + right_g * right_g / (right_h + 1.0);
            if gain > best_gain {
                best_gain = gain;
                best_bin = b as u32;
            }
        }
        out[(g * 2u32) as usize] = best_gain;
        out[(g * 2u32 + 1) as usize] = f64::cast_from(best_bin);
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const W: u32 = 64; // SHIPPED spike-021 feature-per-lane packing.
    const LAUNCHES: usize = 30;
    const REPS: usize = 9;
    let client = rocm_client();

    // (n_feats, num_bin): the 023 regimes. small/medium = launch-bound (fixed sync floor
    // dominates), wide = compute-bound (each scan's own work dominates the shared launch).
    let sweep: [(u32, u32, &str); 6] = [
        (12, 32, "small  n=12  nb=32 "),
        (30, 64, "medium n=30  nb=64 "),
        (40, 128, "large  n=40  nb=128"),
        (128, 128, "n=128 nb=128       "),
        (256, 128, "n=256 nb=128       "),
        (512, 128, "wide  n=512 nb=128 "),
    ];

    println!("# spike024 — batch-sibling-scans: 2×1-slot (baseline=production) vs 1×2-slot (co-packed)");
    println!("# W={W} feature-per-lane (021), {LAUNCHES} launches/timing, {REPS} interleaved reps");
    println!("# ratio = baseline(2 launch+2 read) / candidate(1 launch+1 read); >1 ⇒ co-pack FASTER. Judge SIGN.\n");
    println!("{:<20} {:>11} {:>11} {:>8}  {:>9}", "regime", "A_2x1 µs", "B_1x2 µs", "ratio", "parity");

    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    for &(nf, nb, label) in &sweep {
        // Two siblings, DIFFERENT leaf totals ⇒ different seeds.
        let mk = |seed: u64| -> Vec<f64> {
            let n = (nf * nb) as usize;
            let mut hist = vec![0.0f64; n * 2];
            for i in 0..n {
                let z = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503).wrapping_add(1));
                let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
                let h = 0.5 + u * 4.0;
                hist[i * 2] = (u - 0.5) * 2.0 * h;
                hist[i * 2 + 1] = h;
            }
            hist
        };
        let hist_a = mk(1);
        let hist_b = mk(7);
        let d_a = client.create_from_slice(f64::as_bytes(&hist_a));
        let d_b = client.create_from_slice(f64::as_bytes(&hist_b));
        let out1_len = (nf * 2) as usize;
        let out2_len = (2 * nf * 2) as usize;

        // ---- BASELINE A: two separate launch+readbacks (sibling A then B). ----
        let bench_a = || -> (Vec<f64>, Vec<f64>) {
            let oa = client.create_from_slice(f64::as_bytes(&vec![0.0f64; out1_len]));
            let ob = client.create_from_slice(f64::as_bytes(&vec![0.0f64; out1_len]));
            let cubes = nf.div_ceil(W);
            unsafe {
                scan_one::launch_unchecked(
                    &client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(W),
                    ArrayArg::from_raw_parts(d_a.clone(), (nf * nb * 2) as usize),
                    ArrayArg::from_raw_parts(oa.clone(), out1_len),
                    nb,
                    nf,
                );
            }
            let ra = f64::from_bytes(&client.read_one_unchecked(oa)).to_vec();
            unsafe {
                scan_one::launch_unchecked(
                    &client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(W),
                    ArrayArg::from_raw_parts(d_b.clone(), (nf * nb * 2) as usize),
                    ArrayArg::from_raw_parts(ob.clone(), out1_len),
                    nb,
                    nf,
                );
            }
            let rb = f64::from_bytes(&client.read_one_unchecked(ob)).to_vec();
            (ra, rb)
        };

        // ---- CANDIDATE B: one launch+readback for both siblings. ----
        let bench_b = || -> Vec<f64> {
            let o = client.create_from_slice(f64::as_bytes(&vec![0.0f64; out2_len]));
            let cubes = (2 * nf).div_ceil(W);
            unsafe {
                scan_two::launch_unchecked(
                    &client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(W),
                    ArrayArg::from_raw_parts(d_a.clone(), (nf * nb * 2) as usize),
                    ArrayArg::from_raw_parts(d_b.clone(), (nf * nb * 2) as usize),
                    ArrayArg::from_raw_parts(o.clone(), out2_len),
                    nb,
                    nf,
                );
            }
            f64::from_bytes(&client.read_one_unchecked(o)).to_vec()
        };

        // CORRECTNESS GATE: B's two halves == A's two results, byte-for-byte.
        let (ra, rb) = bench_a();
        let b = bench_b();
        let parity = b[0..out1_len] == ra[..] && b[out1_len..] == rb[..];

        // Timed, interleaved A/B reps (kill warmup drift); median + [p25,p75].
        let mut ta = Vec::with_capacity(REPS);
        let mut tb = Vec::with_capacity(REPS);
        for _ in 0..2 {
            // warmup
            let _ = bench_a();
            let _ = bench_b();
        }
        for _ in 0..REPS {
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                let _ = bench_a();
            }
            ta.push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                let _ = bench_b();
            }
            tb.push(t.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
        }
        let a_med = pct(&mut ta.clone(), 0.5);
        let b_med = pct(&mut tb.clone(), 0.5);
        let a_lo = pct(&mut ta.clone(), 0.25);
        let a_hi = pct(&mut ta.clone(), 0.75);
        let b_lo = pct(&mut tb.clone(), 0.25);
        let b_hi = pct(&mut tb.clone(), 0.75);
        println!(
            "{label:<20} {a_med:>7.1}[{a_lo:.0},{a_hi:.0}] {b_med:>7.1}[{b_lo:.0},{b_hi:.0}] {:>8.3}  {:>9}",
            a_med / b_med,
            if parity { "OK" } else { "MISMATCH!" },
        );
    }
}
