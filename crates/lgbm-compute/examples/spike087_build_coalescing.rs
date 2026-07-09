//! Spike 087 — build+scan COMPUTE deep-dive: the Kaggle-ready real-CUDA build-coalescing
//! probe. This is spike-030's `build_roofline_ab` made RUNTIME-GENERIC and CUDA-buildable,
//! so the same kernels run on the local spoofed-APU (`--features rocm`) AND on real discrete
//! NVIDIA via Kaggle (`--features cuda`). Spike-030 established on the APU that the post-u64
//! wide build is UNCOALESCED-BIN-GATHER-bound (86–93%), and that the production stable-
//! partition MONOTONE row order already banks ~70% of the coalesced ceiling (REAL_ORDER
//! ~1.4× off COAL) — so on the APU the build is "effectively tuned" and 030 explicitly
//! deferred the real verdict: *"reopens only on discrete gfx110x — re-run the probe there."*
//!
//! THE OPEN QUESTION this probe answers on real CUDA: does discrete NVIDIA (128-B coalesced
//! transactions + a real L1/L2 hierarchy, unlike the shared-DDR5 APU) leave MORE coalescing
//! headroom than the APU's ~1.4× (⇒ a reorder-to-coalesced lever with real headroom), or
//! does its cache absorb the uncoalesced gather (⇒ no lever, and the ~2× architectural floor
//! from spike-054 dominates)? The headline metric is REAL_ORDER vs COAL throughput.
//!
//! METHOD — "remove the suspect" (verbatim spike-030): each kernel deletes ONE suspected
//! cost; whichever deletion moves the clock IS the bottleneck.
//!   FULL      the production build replica (random gather, atomic, grad/hess read)
//!   NOATOMIC  reads+quantize kept, per-row LDS fetch_add removed        → the ATOMIC cost
//!   CONST_GH  atomic+gather kept, grad/hess reads → constant            → grad/hess READ bw
//!   COAL_BIN  FULL but the bin read is COALESCED (`col+k`, sequential)  → uncoalesced PATTERN
//!   SEQ_BIN   grad/hess+atomic kept, bin = k%num_bin (no gather at all) → random-gather latency
//!   REAL_ORDER FULL kernel over the MONOTONE production row subset      → the headroom actually paid
//!
//! Spoofed 8-CU APU ⇒ judge SIGN/mechanism not magnitude; on real CUDA the magnitude is
//! the deliverable. Run:
//!   local APU:  cargo run --release -p lgbm-compute --features rocm --example spike087_build_coalescing
//!   Kaggle CUDA: cargo run --release -p lgbm-compute --features cuda --example spike087_build_coalescing

#[cfg(not(any(feature = "rocm", feature = "cuda")))]
fn main() {
    eprintln!(
        "spike-087 requires --features rocm (local APU) or --features cuda (Kaggle real NVIDIA)."
    );
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
use cubecl::prelude::*;

#[cfg(any(feature = "rocm", feature = "cuda"))]
const HIST_LDS_U64: usize = 512; // 2 u64/bin, NUM_BIN ≤ 256
#[cfg(any(feature = "rocm", feature = "cuda"))]
const SCALE: f32 = 1_073_741_824.0; // 2^30 (== production SCALE_F32)

// ─── FULL: byte-faithful copy of construct_leaf_hist_resident_lds_kernel_u64 ───
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_full(
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

// ─── COAL_BIN: byte-for-byte FULL, but the bin read is COALESCED (`col + k`) instead of the
//     random gather (`col + leaf_rows[k]`). Same array, same byte count — ONLY the access
//     pattern changes. FULL ≫ COAL_BIN ⇒ uncoalesced PATTERN dominates ⇒ reorder is the lever. ───
#[cfg(any(feature = "rocm", feature = "cuda"))]
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
        let bin = u32::cast_from(resident_bins[col + k]) as usize; // COALESCED
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

// (The NOATOMIC / CONST_GH / SEQ_BIN "remove-the-suspect" kernels that complete the
// mechanism breakdown are in the spike-030 source, sources/030-wide-build-roofline-
// reattribution/. This Kaggle probe keeps the headline coalescing pair FULL + COAL_BIN plus
// the REAL_ORDER production-order run — the quantity whose real-CUDA value 030 deferred.)

/// The whole probe, generic over the CubeCL runtime — one code path serves the hip (APU) and
/// cuda (real NVIDIA) backends; only the client the caller passes differs. Prints, per wide
/// config, FULL / COAL_BIN / SEQ_BIN / REAL_ORDER device-time medians + Mr/s so the headline
/// REAL_ORDER-vs-COAL coalescing headroom is directly readable.
#[cfg(any(feature = "rocm", feature = "cuda"))]
fn run<R: cubecl::Runtime>(client: cubecl::prelude::ComputeClient<R>) {
    use std::time::Instant;

    const LAUNCHES: usize = 20;
    const REPS: usize = 9;
    const FEATS: usize = 500; // wide shape ⇒ P=1

    let configs: [(usize, usize); 2] = [(250_000, 256), (1_000_000, 256)];
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    println!("# spike087 — build coalescing probe (FEATS={FEATS}, wide P=1). REAL_ORDER vs COAL = the headroom.");
    for (num_data, num_bin) in configs {
        let feat_len = (2 * num_bin) as u32;
        let slot_len = FEATS * 2 * num_bin;
        let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();

        let mut bins: Vec<u8> = Vec::with_capacity(FEATS * num_data);
        for f in 0..FEATS {
            for row in 0..num_data {
                let h = (row as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                bins.push((h % num_bin as u64) as u8);
            }
        }
        // Random permutation (worst gather) and the MONOTONE production subset (50%-selectivity
        // leaf one level below root: stride-2, sorted ⇒ partially coalesced — what training pays).
        let leaf_rows: Vec<u32> = (0..num_data as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % num_data as u32)
            .collect();
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

        // Time a build-kernel launch batch; return the median device-ms/launch over REPS.
        // Args mirror the kernel signature (bins, rows, ord_g, ord_h, slot_off, scalar,
        // feat_len, out). Launch form is verbatim spike-030 (cubecl 0.10):
        // `ArrayArg::from_raw_parts(handle.clone(), len)`, scalars passed directly, `::<R>` for
        // the generic-fn context (matches histogram.rs:1167).
        macro_rules! bench_build {
            ($k:ident, $bins:expr, $rows:expr, $nrows:expr) => {{
                let mut samples = Vec::with_capacity(REPS);
                for _ in 0..REPS {
                    let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
                    let t = Instant::now();
                    for _ in 0..LAUNCHES {
                        unsafe {
                            $k::launch_unchecked::<R>(
                                &client,
                                CubeCount::Static(FEATS as u32, 1, 1),
                                CubeDim::new_1d(256),
                                ArrayArg::from_raw_parts($bins.clone(), FEATS * num_data),
                                ArrayArg::from_raw_parts($rows.clone(), $nrows),
                                ArrayArg::from_raw_parts(d_g.clone(), $nrows),
                                ArrayArg::from_raw_parts(d_h.clone(), $nrows),
                                ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                                num_data,
                                feat_len,
                                ArrayArg::from_raw_parts(out.clone(), slot_len),
                            );
                        }
                    }
                    let _ = client.read_one_unchecked(out);
                    samples.push(t.elapsed().as_secs_f64() * 1000.0 / LAUNCHES as f64);
                }
                pct(&mut samples, 0.5)
            }};
        }
        let full = bench_build!(build_full, d_bins, d_rows, r);
        let coal = bench_build!(build_coalbin, d_bins, d_rows, r);
        let real = bench_build!(build_full, d_bins, d_rows_real, r_real);
        let mrs = |ms: f64, rows: usize| (FEATS as f64 * rows as f64) / (ms * 1e3);

        println!(
            "\n## {FEATS} feat × {num_data} rows × {num_bin} bins  (P=1, wide)"
        );
        println!("#   FULL       {full:8.1} ms   ({:.0} Mr/s)  random gather (worst)", mrs(full, r));
        println!(
            "#   COAL_BIN   {coal:8.1} ms   ({:.0} Mr/s)  {:.0}% of FULL  <- coalesced ceiling",
            mrs(coal, r),
            100.0 * coal / full
        );
        println!(
            "#   REAL_ORDER {real:8.1} ms   ({:.0} Mr/s)  <- production monotone order (the headroom actually paid)",
            mrs(real, r_real)
        );
        println!(
            "#   >>> HEADROOM: REAL_ORDER is {:.2}× the coalesced ceiling ({:.0} vs {:.0} Mr/s).",
            mrs(coal, r) / mrs(real, r_real),
            mrs(real, r_real),
            mrs(coal, r)
        );
    }
    println!("\n# READING (real CUDA is the deliverable — APU already showed ~1.4×):");
    println!("#  REAL_ORDER Mr/s NEAR COAL Mr/s (headroom → 1.0×) ⇒ NO reorder lever (cache absorbs it).");
    println!("#  REAL_ORDER Mr/s ≪ COAL Mr/s (headroom ≫ 1.4×)   ⇒ a reorder-to-coalesced lever HAS headroom on real CUDA.");
    println!("#  Either way the spike-054 architectural ~2× floor bounds the end-to-end win; SEQ_BIN/NOATOMIC/");
    println!("#  CONST_GH breakdown for the mechanism is in the spike-030 source (sources/030-...).");
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
fn main() {
    run(lgbm_compute::runtime::rocm_client());
}

#[cfg(feature = "cuda")]
fn main() {
    run(lgbm_compute::runtime::cuda_client());
}
