//! Spike 030 — RE-ATTRIBUTE the post-u64 wide histogram BUILD. The last build attribution
//! (spike-015) declared it "atomic-contention-bound ~820 Mr/s" — but that PREDATES the u64
//! fixed-point ship (spike-018/019), which made the per-row atomic a native single-instruction
//! `ds_add_u64`. The manifest's iron rule is "re-profile after every build change — the
//! bottleneck has moved three times (014→015→023)." It was never re-profiled after u64.
//!
//! Hypothesis: post-u64 the wide build (one-cube-per-feature, P=1 at 500 feats) is now
//! MEMORY-bound, not atomic-bound. The dominant traffic is the CROSS-FEATURE redundant
//! grad/hess read: every feature's cube re-reads the SAME leaf's `ord_g`/`ord_h` from global
//! memory, so grad+hess traffic = feats × rows × 8 B ≈ 8× the (u8) bin traffic.
//!
//! METHOD — "remove the suspect" A/B on the LIVE u64 resident kernel. Four variants, each
//! deletes ONE suspected cost; whichever deletion moves the clock IS the bottleneck:
//!   FULL      the live build (baseline)
//!   NOATOMIC  reads + quantize kept, per-row LDS fetch_add removed (one final write keeps
//!             the loads live vs DCE)            → isolates the ATOMIC cost
//!   CONST_GH  atomic + bin gather kept, grad/hess reads replaced by a constant
//!                                               → isolates the grad/hess READ bandwidth
//!   SEQ_BIN   grad/hess + atomic kept, bin index = k%num_bin (no resident_bins / leaf_rows
//!             gather)                           → isolates the RANDOM-GATHER latency
//! Plus a NUM_BIN sweep {16,256} on FULL: if atomic-CONTENTION-bound, 16 bins (more
//! collisions/bin) is SLOWER; if memory/latency-bound, ~flat in bins.
//!
//! Interpretation:
//!   FULL ≈ NOATOMIC                       ⇒ NOT atomic-bound (015's verdict is stale post-u64)
//!   FULL ≫ CONST_GH                       ⇒ grad/hess READ bandwidth dominates ⇒ 031 GREEN
//!                                            (cut the cross-feature grad/hess redundancy)
//!   FULL ≫ SEQ_BIN  (but ≈ CONST_GH·…)    ⇒ random bin-gather LATENCY dominates instead
//!   achieved GB/s near the APU DDR5 peak   ⇒ bandwidth-bound (corroborates CONST_GH)
//!
//! Spoofed 8-CU APU ⇒ judge the SIGN/mechanism, not the magnitude. Run:
//!   cargo run --release --features rocm --example spike030_build_roofline_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike-030 requires --features rocm (gfx1100/gfx1152 APU).");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_U64: usize = 512; // 2 u64/bin, NUM_BIN ≤ 256
#[cfg(feature = "rocm")]
const SCALE: f32 = 1_073_741_824.0; // 2^30 (== production SCALE_F32)

// ─── FULL: byte-faithful copy of construct_leaf_hist_resident_lds_kernel_u64 ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_full(
    resident_bins: &Array<u8>, // native u8 width (prod resident, spike-029/qix)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ─── NOATOMIC: same reads+quantize, per-row LDS atomic REMOVED. `acc` keeps the loads live
//     (cubecl would DCE them otherwise); one final atomic per unit = negligible. ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_noatomic(
    resident_bins: &Array<u8>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    let mut acc = 0u64;
    while k < r {
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as u64;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
        // keep every load + the quantize observable, but NO per-row atomic / LDS traffic.
        acc = acc + qg + qh + bin;
        k += stride;
    }
    if (UNIT_POS as usize) < fl {
        out[base + UNIT_POS as usize].fetch_add(acc);
    }
}

// ─── CONST_GH: atomic + bin gather kept; grad/hess GLOBAL reads replaced by a constant
//     (removes 8 B/row of sequential grad/hess traffic). ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_const_gh(
    resident_bins: &Array<u8>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>, // present for identical signature/occupancy, but NOT read in the loop
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let qconst = u64::cast_from(i64::cast_from(f32::round(0.5f32 * SCALE)));
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(qconst); // same atomic, NO grad/hess global read
        sub[ti + 1].fetch_add(qconst);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ─── COAL_BIN: byte-for-byte FULL, but the bin read is COALESCED (`col + k`, sequential)
//     instead of the random gather (`col + leaf_rows[k]`). Same 500 MB array, same byte
//     count — ONLY the access pattern changes. Separates "uncoalesced PATTERN" (→ COAL_BIN
//     fast ⇒ reorder-to-coalesced is the lever) from "bin-array bandwidth" (→ COAL_BIN slow). ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_coalbin(
    resident_bins: &Array<u8>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        // COALESCED bin read: consecutive units read consecutive addresses.
        let bin = u32::cast_from(resident_bins[col + k]) as usize;
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ─── SEQ_BIN: grad/hess + atomic kept; bin = k%num_bin (no resident_bins / leaf_rows
//     gather). Removes the RANDOM-gather latency + the indirection reads. ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_seqbin(
    resident_bins: &Array<u8>, // unread (signature parity)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_bin: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let nb = num_bin;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let bin = k % nb; // sequential, no global gather
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const CUBE_DIM: u32 = 256;
    const LAUNCHES: usize = 20;
    const REPS: usize = 9;
    const FEATS: usize = 500; // wide shape ⇒ P=1 (target_cubes/500 = 1)
    let client = rocm_client();

    // wide configs from the manifest (250k = CPU beats GPU ~4×; 1M = the crossover regime).
    let configs: [(usize, usize); 2] = [(250_000, 256), (1_000_000, 256)];
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    for (num_data, num_bin) in configs {
        let feat_len = (2 * num_bin) as u32;
        let slot_len = FEATS * 2 * num_bin;
        let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();

        // u8 bins (prod native width). Random within a feature column ⇒ cache-hostile gather.
        let mut bins: Vec<u8> = Vec::with_capacity(FEATS * num_data);
        for f in 0..FEATS {
            for row in 0..num_data {
                let h = (row as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                bins.push((h % num_bin as u64) as u8);
            }
        }
        let leaf_rows: Vec<u32> = (0..num_data as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % num_data as u32)
            .collect();
        // REAL_ORDER: the actual stable-partition leaf order is a MONOTONE-INCREASING subset
        // (not a random permutation). Model a 50%-selectivity leaf (one level below root):
        // every other row, sorted ⇒ stride-2 gather, partially coalesced. Bounds how much of
        // the random-permutation penalty real training actually pays.
        let leaf_rows_real: Vec<u32> = (0..num_data as u32).step_by(2).collect();
        let ord_g: Vec<f32> = (0..num_data).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let ord_h: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
        let r = leaf_rows.len();

        let r_real = leaf_rows_real.len();
        let d_bins = client.create_from_slice(u8::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_rows_real = client.create_from_slice(u32::as_bytes(&leaf_rows_real));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        // bytes read/launch (FULL): feats × rows × (1 bin + 4 leaf_rows + 4 g + 4 h).
        let bytes_full = (FEATS as f64) * (r as f64) * 13.0;
        let reads = (FEATS * r) as f64;

        macro_rules! bench {
            ($k:ident, $usebin:expr) => {{
                let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
                let t = Instant::now();
                for _ in 0..LAUNCHES {
                    unsafe {
                        $k::launch_unchecked(
                            &client,
                            CubeCount::Static(FEATS as u32, 1, 1),
                            CubeDim::new_1d(CUBE_DIM),
                            ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                            ArrayArg::from_raw_parts(d_rows.clone(), r),
                            ArrayArg::from_raw_parts(d_g.clone(), r),
                            ArrayArg::from_raw_parts(d_h.clone(), r),
                            ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                            $usebin,
                            feat_len,
                            ArrayArg::from_raw_parts(out.clone(), slot_len),
                        );
                    }
                }
                let _ = client.read_one_unchecked(out);
                t.elapsed().as_secs_f64() * 1e3
            }};
        }

        // REAL_ORDER: build_full over the monotone 50%-subset (lengths = r_real). Reported as
        // Mr/s so it compares fairly to FULL/COAL despite the halved row count.
        let bench_real = || -> f64 {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_full::launch_unchecked(
                        &client,
                        CubeCount::Static(FEATS as u32, 1, 1),
                        CubeDim::new_1d(CUBE_DIM),
                        ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                        ArrayArg::from_raw_parts(d_rows_real.clone(), r_real),
                        ArrayArg::from_raw_parts(d_g.clone(), r_real),
                        ArrayArg::from_raw_parts(d_h.clone(), r_real),
                        ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                        num_data,
                        feat_len,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed().as_secs_f64() * 1e3
        };

        // warm
        for _ in 0..2 {
            let _ = bench_real();
            let _ = bench!(build_full, num_data);
            let _ = bench!(build_noatomic, num_data);
            let _ = bench!(build_const_gh, num_data);
            let _ = bench!(build_coalbin, num_data);
            let _ = bench!(build_seqbin, num_bin as usize);
        }

        let (mut sf, mut sn, mut sc, mut sb, mut ss, mut sr): (
            Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>,
        ) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..REPS {
            sf.push(bench!(build_full, num_data));
            sn.push(bench!(build_noatomic, num_data));
            sc.push(bench!(build_const_gh, num_data));
            sb.push(bench!(build_coalbin, num_data));
            ss.push(bench!(build_seqbin, num_bin as usize));
            sr.push(bench_real());
        }
        let f50 = pct(&mut sf, 0.5);
        let (n50, c50, b50, s50, r50) = (
            pct(&mut sn, 0.5),
            pct(&mut sc, 0.5),
            pct(&mut sb, 0.5),
            pct(&mut ss, 0.5),
            pct(&mut sr, 0.5),
        );
        let gbps = bytes_full * LAUNCHES as f64 / (f50 / 1e3) / 1e9;
        let mrs = reads * LAUNCHES as f64 / (f50 / 1e3) / 1e6;

        println!("\n## {FEATS} feat × {num_data} rows × {num_bin} bins  (P=1, wide regime)");
        println!("#   FULL      {f50:7.1} ms   ({mrs:.0} Mr/s, {gbps:.1} GB/s effective)");
        println!(
            "#   NOATOMIC  {n50:7.1} ms   ({:.0}% of FULL)  ← reads+quantize only",
            n50 / f50 * 100.0
        );
        println!(
            "#   CONST_GH  {c50:7.1} ms   ({:.0}% of FULL)  ← no grad/hess read",
            c50 / f50 * 100.0
        );
        println!(
            "#   COAL_BIN  {b50:7.1} ms   ({:.0}% of FULL)  ← same bin array, COALESCED read",
            b50 / f50 * 100.0
        );
        println!(
            "#   SEQ_BIN   {s50:7.1} ms   ({:.0}% of FULL)  ← no bin array read at all",
            s50 / f50 * 100.0
        );
        // Mr/s normalizes the halved row count so REAL/FULL/COAL compare fairly.
        let mrs_real = (FEATS * r_real) as f64 * LAUNCHES as f64 / (r50 / 1e3) / 1e6;
        let mrs_coal = reads * LAUNCHES as f64 / (b50 / 1e3) / 1e6;
        println!(
            "#   REAL_ORDER (monotone 50% subset) {r50:7.1} ms  ({mrs_real:.0} Mr/s)  vs FULL {mrs:.0} / COAL {mrs_coal:.0} Mr/s",
        );
        println!(
            "#   ⇒ atomic ≈ {:.0}%   grad/hess-read ≈ {:.0}%   UNCOALESCED-pattern ≈ {:.0}%   bin-bandwidth ≈ {:.0}%",
            (f50 - n50).max(0.0) / f50 * 100.0,
            (f50 - c50).max(0.0) / f50 * 100.0,
            (f50 - b50).max(0.0) / f50 * 100.0,
            (b50 - s50).max(0.0) / f50 * 100.0,
        );
    }

    println!("\n# VERDICT KEY:");
    println!("#  NOATOMIC ≈ FULL (atomic share small)        ⇒ NOT atomic-bound (015 stale post-u64)");
    println!("#  CONST_GH ≪ FULL (grad/hess-read share big)  ⇒ MEMORY-bound on grad/hess ⇒ 031 GREEN");
    println!("#  SEQ_BIN  ≪ FULL but CONST_GH ≈ FULL         ⇒ random-gather LATENCY-bound instead");
    println!("#  effective GB/s near APU DDR5 peak (~60-100) ⇒ bandwidth-bound corroborated");
}
