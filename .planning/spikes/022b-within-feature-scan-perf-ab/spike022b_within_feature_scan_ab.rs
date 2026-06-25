//! Spike 022b — within-feature parallel scan PERF A/B (the deferred 022 perf question).
//!
//! spike-022 retired the PARITY risk (within-feature reordered scan is parity-safe
//! within ~1e-6). The remaining question is ROI: does a cooperative within-feature
//! scan (K lanes per feature) beat spike-021's feature-per-LANE scan (1 lane per
//! feature), and IN WHICH REGIME? Hypothesis (from 021's occupancy reasoning):
//!   - WIDE (many features): 021 already saturates the device with feature-level
//!     parallelism ⇒ cooperation adds LDS/sync overhead with no idle HW to fill ⇒
//!     REGRESSION.
//!   - NARROW (few features): 021 leaves most lanes idle ⇒ cooperation fills them
//!     and shortens per-feature latency ⇒ WIN.
//!
//! Design (CONVENTIONS "in-kernel A/B via a comptime factor", p93/017): ONE kernel
//! `scan_coop` with `#[comptime] coop: u32` = K lanes per feature.
//!   - K=1  ⇒ feature-per-lane, ONE lane does the whole sequential scan = the
//!     spike-021 baseline (byte-identical path: no LDS, no sync).
//!   - K>1  ⇒ the K lanes segment-scan the bins, combine prefixes + argmax via LDS.
//! Both arms compute the SAME per-bin gain (a representative `g²/(h+λ)` split gain)
//! and the SAME argmax, so the ratio isolates the parallelism structure. Correctness
//! gate: K>1 must reproduce K=1's (best_gain, best_bin) per feature.
//!
//! Spoofed 8-CU gfx1152 APU ⇒ judge the SIGN of the device-time ratio across feature
//! counts, not the magnitude. Run:
//!   cargo run --release -p lgbm-compute --features rocm --example spike022b_within_feature_scan_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("requires --features rocm (gfx1100/APU).");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const CUBE_DIM: u32 = 256;
/// LDS scratch: 4 arrays × CUBE_DIM f64 (seg_g, seg_h, best_gain, best_bin) = 8 KiB.
#[cfg(feature = "rocm")]
const LDS: usize = CUBE_DIM as usize;

/// Within-feature cooperative best-split scan, parameterized by `coop` = K lanes per
/// feature. `hist` is F features × num_bin bins, interleaved (grad, hess) f64. Output
/// is 2 f64 per feature: (best_gain, best_bin).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scan_coop(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    num_bin: u32,
    n_feats: u32,
    cube_dim: u32, // ACTUAL launched CubeDim (≤ 256); controls occupancy fairly
    #[comptime] coop: u32,
) {
    let k = coop;
    let fpc = cube_dim / k; // features per cube
    let local_feat = UNIT_POS / k;
    let lane = UNIT_POS % k;
    let gfeat = CUBE_POS_X * fpc + local_feat;

    if gfeat < n_feats {
        let hb = (gfeat * num_bin * 2u32) as usize;
        let nb = num_bin as usize;

        if coop == 1u32 {
            // ---- K=1: feature-per-lane sequential scan (the spike-021 baseline) ----
            // total
            let mut tot_g = 0.0f64;
            let mut tot_h = 0.0f64;
            for b in 0..nb {
                tot_g += hist[hb + b * 2];
                tot_h += hist[hb + b * 2 + 1];
            }
            let mut pg = 0.0f64;
            let mut ph = 0.0f64;
            let mut bg = 0.0f64; // all gains >=0 ⇒ 0.0 sentinel (like split_scan_body)
            let mut bb = 0u32;
            for b in 0..nb {
                pg += hist[hb + b * 2];
                ph += hist[hb + b * 2 + 1];
                // valid split point: left=[0..=b], right=[b+1..]; need b < nb-1
                if b + 1 < nb {
                    let rg = tot_g - pg;
                    let rh = tot_h - ph;
                    let gain = pg * pg / (ph + 1.0) + rg * rg / (rh + 1.0);
                    if gain > bg {
                        bg = gain;
                        bb = b as u32;
                    }
                }
            }
            out[(gfeat * 2) as usize] = bg;
            out[(gfeat * 2 + 1) as usize] = f64::cast_from(bb);
        } else {
            // ---- K>1: cooperative segmented scan + LDS combine + LDS argmax ----
            let mut seg_g = SharedMemory::<f64>::new(LDS);
            let mut seg_h = SharedMemory::<f64>::new(LDS);
            let mut best_gain = SharedMemory::<f64>::new(LDS);
            let mut best_bin = SharedMemory::<f64>::new(LDS);
            let seg = num_bin / k; // bins per lane (num_bin divisible by k in this bench)
            let slot = (local_feat * k + lane) as usize;

            // phase 1 — this lane's segment total
            let mut sg = 0.0f64;
            let mut sh = 0.0f64;
            for j in 0..seg {
                let b = (lane * seg + j) as usize;
                sg += hist[hb + b * 2];
                sh += hist[hb + b * 2 + 1];
            }
            seg_g[slot] = sg;
            seg_h[slot] = sh;
            sync_cube();

            // phase 2 — feature total + this lane's exclusive prefix offset
            let mut tot_g = 0.0f64;
            let mut tot_h = 0.0f64;
            let mut off_g = 0.0f64;
            let mut off_h = 0.0f64;
            for l in 0..k {
                let s = (local_feat * k + l) as usize;
                let g = seg_g[s];
                let h = seg_h[s];
                tot_g += g;
                tot_h += h;
                if l < lane {
                    off_g += g;
                    off_h += h;
                }
            }

            // phase 3 — re-walk this lane's segment, gains with the global prefix
            let mut pg = off_g;
            let mut ph = off_h;
            let mut bg = 0.0f64; // all gains >=0 ⇒ 0.0 sentinel (like split_scan_body)
            let mut bb = 0u32;
            for j in 0..seg {
                let b = lane * seg + j;
                pg += hist[hb + (b as usize) * 2];
                ph += hist[hb + (b as usize) * 2 + 1];
                if b + 1 < num_bin {
                    let rg = tot_g - pg;
                    let rh = tot_h - ph;
                    let gain = pg * pg / (ph + 1.0) + rg * rg / (rh + 1.0);
                    if gain > bg {
                        bg = gain;
                        bb = b;
                    }
                }
            }
            best_gain[slot] = bg;
            best_bin[slot] = f64::cast_from(bb);
            sync_cube();

            // phase 4 — lane 0 reduces the K partial winners
            if lane == 0u32 {
                let base = (local_feat * k) as usize;
                let mut wg = best_gain[base];
                let mut wb = best_bin[base];
                for l in 1..k {
                    let s = base + l as usize;
                    if best_gain[s] > wg {
                        wg = best_gain[s];
                        wb = best_bin[s];
                    }
                }
                out[(gfeat * 2) as usize] = wg;
                out[(gfeat * 2 + 1) as usize] = wb;
            }
        }
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const NUM_BIN: u32 = 256;
    const LAUNCHES: usize = 30;
    const REPS: usize = 9;
    let client = rocm_client();

    // Candidate configs = (CubeDim, K). The BASELINE is the SHIPPED spike-021:
    // feature-per-lane at W=64 (K=1, cd=64). Every candidate is reported vs THAT, so
    // the comparison isolates within-feature cooperation from raw occupancy (the
    // first-run confound: K=1 @ cd=256 is only F/256 cubes ⇒ under-occupied, an
    // unfairly weak baseline). cd=256/K=1 is kept in the sweep to SHOW that confound.
    let configs: [(u32, u32, &str); 8] = [
        (64, 1, "cd64 K1  = SHIPPED 021 (baseline)"),
        (256, 1, "cd256 K1 (under-occupied baseline)"),
        (64, 2, "cd64  K2 coop"),
        (64, 4, "cd64  K4 coop"),
        (64, 8, "cd64  K8 coop"),
        (256, 8, "cd256 K8 coop"),
        (256, 16, "cd256 K16 coop"),
        (256, 32, "cd256 K32 coop"),
    ];
    let feat_sweep: [u32; 5] = [8, 32, 128, 256, 512];

    println!("# spike022b — within-feature cooperative scan vs the SHIPPED spike-021 (feature-per-lane W=64)");
    println!("# num_bin={NUM_BIN}, {LAUNCHES} launches/timing, {REPS} interleaved reps; ratio = baseline021 / candidate");
    println!("# (>1 ⇒ candidate FASTER than shipped 021). Judge the SIGN (spoofed 8-CU APU).\n");

    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    for &f in &feat_sweep {
        // deterministic histogram: F features × NUM_BIN bins, interleaved (grad,hess)
        let n = (f * NUM_BIN) as usize;
        let mut hist = vec![0.0f64; n * 2];
        for i in 0..n {
            let z = (i as u64).wrapping_mul(2_654_435_761).wrapping_add(1);
            let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
            let h = 0.5 + u * 4.0;
            hist[i * 2] = (u - 0.5) * 2.0 * h;
            hist[i * 2 + 1] = h;
        }
        let d_hist = client.create_from_slice(f64::as_bytes(&hist));
        let out_len = (f * 2) as usize;

        // bench one (cd, comptime-K) config
        macro_rules! bench {
            ($cd:expr, $k:expr) => {{
                let out = client.create_from_slice(f64::as_bytes(&vec![0.0f64; out_len]));
                let cubes = f.div_ceil($cd / $k);
                let t = Instant::now();
                for _ in 0..LAUNCHES {
                    unsafe {
                        scan_coop::launch_unchecked(
                            &client,
                            CubeCount::Static(cubes, 1, 1),
                            CubeDim::new_1d($cd),
                            ArrayArg::from_raw_parts(d_hist.clone(), n * 2),
                            ArrayArg::from_raw_parts(out.clone(), out_len),
                            NUM_BIN,
                            f,
                            $cd,
                            $k,
                        );
                    }
                }
                let bytes = client.read_one_unchecked(out);
                (t.elapsed().as_secs_f64() * 1e3, f64::from_bytes(&bytes).to_vec())
            }};
        }
        // dispatch a (cd,K) pair from runtime values to the comptime-K launch
        macro_rules! run {
            ($cd:expr, $k:expr) => {
                match $k {
                    1 => bench!($cd, 1u32),
                    2 => bench!($cd, 2u32),
                    4 => bench!($cd, 4u32),
                    8 => bench!($cd, 8u32),
                    16 => bench!($cd, 16u32),
                    _ => bench!($cd, 32u32),
                }
            };
        }

        let (_, ref_out) = run!(64u32, 1u32); // correctness reference + warm
        for &(cd, k, _) in &configs {
            let _ = run!(cd, k);
        }

        println!("## F={f} features");
        for &(cd, k, label) in &configs {
            // interleaved: baseline021 (cd64 K1) sampled alongside each candidate
            let (mut sc, mut sb): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
            let mut mism = 0u32;
            let mut max_rel = 0.0f64;
            for _ in 0..REPS {
                sb.push(run!(64u32, 1u32).0);
                let (tc, oc) = run!(cd, k);
                sc.push(tc);
                for fi in 0..f as usize {
                    if (oc[fi * 2 + 1] - ref_out[fi * 2 + 1]).abs() > 0.5 {
                        mism += 1;
                    }
                    let rel = (oc[fi * 2] - ref_out[fi * 2]).abs() / ref_out[fi * 2].abs().max(1e-300);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
            }
            let (c25, c50, c75) = (pct(&mut sc, 0.25), pct(&mut sc, 0.5), pct(&mut sc, 0.75));
            let (b25, b50, b75) = (pct(&mut sb, 0.25), pct(&mut sb, 0.5), pct(&mut sb, 0.75));
            let ratio = b50 / c50;
            let sep = if c75 < b25 {
                "SEP-WIN"
            } else if c25 > b75 {
                "SEP-LOSS"
            } else {
                "≈tie"
            };
            println!(
                "  {label:<34} {c50:>6.1}ms[{c25:.0}..{c75:.0}]  vs021={ratio:>4.2}x {sep:>8}  [mism={mism} rel={max_rel:.0e}]"
            );
            let _ = b75;
        }
        println!();
    }
    println!("# Fair question: does ANY cooperative config beat the SHIPPED 021 (cd64 K1)?");
    println!("# If cd256-K1 is much slower than cd64-K1, the run-1 'coop wins everywhere' was an");
    println!("# occupancy confound; the honest test is each coop config's vs021 ratio.");
}
