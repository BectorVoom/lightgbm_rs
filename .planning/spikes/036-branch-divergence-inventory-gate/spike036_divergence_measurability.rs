//! Spike 036 — CAN the spoofed 8-CU APU even MEASURE warp/branch divergence?
//!
//! "Optimize conditional branching in GPU kernels" presumes (a) a divergent branch on a
//! hot path and (b) that a divergence delta is observable on this hardware. The campaign's
//! iron rules say neither is free: rocprof is unsupported on gfx1152, and the GPU is a
//! shared-DDR5 APU where every prior GPU number is sign-only. Before A/B-ing any real
//! kernel (037 trip-count divergence, 038 break-vs-select) we must first answer the GATE:
//! does a DELIBERATELY-INJECTED, KNOWN-magnitude divergence separate above the APU's own
//! timing noise? If even a 32× injected wave imbalance collapses to ~1× wall-clock, then no
//! divergence lever on the real (already-mostly-branchless) kernels can be measured here.
//!
//! METHOD — a controlled-divergence LADDER. Every arm does the SAME TOTAL WORK
//! (sum of per-lane loop trip counts = CUBE_DIM * K, identical across arms); only the
//! DISTRIBUTION across wavefront lanes changes. The loop body is constant ALU/iteration
//! (an fma into a register sink written once → no memory traffic, no DCE), and each lane's
//! trip count comes from a DEVICE ARRAY so the compiler cannot specialize or hoist it.
//!
//!   UNIFORM   counts[lane] = K            every lane → no intra-wave imbalance
//!   DIV2      lane%2==0 ? 2K : 0          half the lanes idle    (same total)
//!   DIV4      lane%4==0 ? 4K : 0          ¾ idle                 (same total)
//!   DIV32     lane%32==0 ? 32K : 0        1 active lane / 32-wave (same total)
//!
//! If the wavefront executes in lockstep masked to its SLOWEST lane (textbook divergence
//! cost), wall-clock scales 1 : 2 : 4 : 32 even though useful work is constant — the idle
//! masked lanes are pure waste. Robust to wave32 vs wave64 (the imbalance is INTERLEAVED
//! within any contiguous 32- or 64-lane group).
//!
//! READ:
//!   ratios ≈ 1:2:4:32  ⇒ divergence IS resolvable here ⇒ 037/038 measurability premise HOLDS
//!   ratios collapse →1  ⇒ APU cannot resolve injected divergence ⇒ 037/038 DEAD on arrival
//!                          (no divergence lever is sign-measurable on this hardware)
//!
//! Spoofed 8-CU APU ⇒ judge the SIGN/ladder SHAPE, not absolute ms. Run:
//!   cargo run --release --features rocm --example spike036_divergence_measurability

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike-036 requires --features rocm (gfx1100/gfx1152 APU).");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

// One cube of CUBE_DIM lanes; the divergence pattern repeats per cube. The per-lane trip
// count is read from `counts[UNIT_POS]` (length CUBE_DIM), so EVERY cube sees the same
// intra-wave imbalance. `acc` is a register sink written once at the end (defeats DCE);
// `w` is a runtime scalar so `acc = acc*c + w` never folds to a closed form.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn diverge(counts: &Array<u32>, out: &mut Array<f32>, w: f32) {
    let lane = UNIT_POS as usize; // within-cube lane (0..CUBE_DIM)
    let gpos = ABSOLUTE_POS; // unique global lane id (usize)
    let n = counts[lane]; // data-dependent trip count (drives divergence)
    let mut acc = 0.0f32; // loop-carried mutable MUST init from a literal
    let mut i = 0u32;
    while i < n {
        // constant ALU per iteration; no memory, no branch inside the body — the ONLY
        // cross-lane difference is HOW MANY times this runs (the trip-count divergence).
        acc = acc * 1.0000001f32 + w;
        i += 1u32;
    }
    out[gpos] = acc;
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const CUBE_DIM: u32 = 64; // lanes/cube (covers wave32 ×2 and wave64 ×1)
    const CUBE_COUNT: u32 = 2048; // fill the device (8 CU APU); pattern repeats per cube
    const K: u32 = 8192; // UNIFORM per-lane iterations
    const LAUNCHES: usize = 20; // accumulate launches into one reused buffer
    const REPS: usize = 11;

    let client = rocm_client();
    let cd = CUBE_DIM as usize;

    // Per-lane trip-count patterns. Each sums to CUBE_DIM*K ⇒ IDENTICAL total work.
    let uniform: Vec<u32> = (0..cd).map(|_| K).collect();
    let div2: Vec<u32> = (0..cd).map(|l| if l % 2 == 0 { 2 * K } else { 0 }).collect();
    let div4: Vec<u32> = (0..cd).map(|l| if l % 4 == 0 { 4 * K } else { 0 }).collect();
    let div32: Vec<u32> = (0..cd).map(|l| if l % 32 == 0 { 32 * K } else { 0 }).collect();
    // sanity: all four distributions carry the same total work
    let total = |v: &[u32]| v.iter().map(|&x| x as u64).sum::<u64>();
    assert_eq!(total(&uniform), total(&div2));
    assert_eq!(total(&uniform), total(&div4));
    assert_eq!(total(&uniform), total(&div32));

    let d_uniform = client.create_from_slice(u32::as_bytes(&uniform));
    let d_div2 = client.create_from_slice(u32::as_bytes(&div2));
    let d_div4 = client.create_from_slice(u32::as_bytes(&div4));
    let d_div32 = client.create_from_slice(u32::as_bytes(&div32));
    let n_lanes = (CUBE_COUNT * CUBE_DIM) as usize;
    let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n_lanes]));
    let w = std::hint::black_box(1.0e-3f32);

    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    macro_rules! bench {
        ($counts:expr) => {{
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    diverge::launch_unchecked(
                        &client,
                        CubeCount::Static(CUBE_COUNT, 1, 1),
                        CubeDim::new_1d(CUBE_DIM),
                        ArrayArg::from_raw_parts($counts.clone(), cd),
                        ArrayArg::from_raw_parts(out.clone(), n_lanes),
                        w,
                    );
                }
            }
            let _ = client.read_one_unchecked(out.clone());
            t.elapsed().as_secs_f64() * 1e3
        }};
    }

    // warm
    for _ in 0..3 {
        let _ = bench!(d_uniform);
        let _ = bench!(d_div2);
        let _ = bench!(d_div4);
        let _ = bench!(d_div32);
    }

    let (mut su, mut s2, mut s4, mut s32): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) =
        (vec![], vec![], vec![], vec![]);
    for _ in 0..REPS {
        su.push(bench!(d_uniform));
        s2.push(bench!(d_div2));
        s4.push(bench!(d_div4));
        s32.push(bench!(d_div32));
    }

    let (u50, u25, u75) = (pct(&mut su, 0.5), pct(&mut su, 0.25), pct(&mut su, 0.75));
    let (d2_50, d2_25, d2_75) = (pct(&mut s2, 0.5), pct(&mut s2, 0.25), pct(&mut s2, 0.75));
    let (d4_50, d4_25, d4_75) = (pct(&mut s4, 0.5), pct(&mut s4, 0.25), pct(&mut s4, 0.75));
    let (d32_50, d32_25, d32_75) =
        (pct(&mut s32, 0.5), pct(&mut s32, 0.25), pct(&mut s32, 0.75));

    // useful work / sec (constant numerator across arms) — collapses as divergence wastes slots.
    let work = (n_lanes as f64) * (K as f64) * (LAUNCHES as f64); // total fma's per bench
    let giters = |ms: f64| work / (ms / 1e3) / 1e9;

    println!("\n# spike-036 divergence measurability ladder");
    println!(
        "# CUBE_DIM={CUBE_DIM} CUBE_COUNT={CUBE_COUNT} K={K} LAUNCHES={LAUNCHES} REPS={REPS}"
    );
    println!("# every arm = SAME total work; only the intra-wave distribution differs.\n");
    println!("# arm       median   [p25 .. p75]    ratio/UNIFORM   Giter/s");
    println!(
        "#  UNIFORM  {u50:7.1} [{u25:6.1}..{u75:6.1}]   1.00× (ref)     {:5.1}",
        giters(u50)
    );
    println!(
        "#  DIV2      {d2_50:7.1} [{d2_25:6.1}..{d2_75:6.1}]   {:.2}× (ideal 2)  {:5.1}",
        d2_50 / u50,
        giters(d2_50)
    );
    println!(
        "#  DIV4      {d4_50:7.1} [{d4_25:6.1}..{d4_75:6.1}]   {:.2}× (ideal 4)  {:5.1}",
        d4_50 / u50,
        giters(d4_50)
    );
    println!(
        "#  DIV32     {d32_50:7.1} [{d32_25:6.1}..{d32_75:6.1}]   {:.2}× (ideal 32) {:5.1}",
        d32_50 / u50,
        giters(d32_50)
    );

    // SEP test: is each rung's p25 above UNIFORM's p75 (a robust, non-overlapping separation)?
    let sep2 = d2_25 > u75;
    let sep32 = d32_25 > u75;
    println!("\n# VERDICT:");
    println!(
        "#  DIV2 p25 ({d2_25:.1}) > UNIFORM p75 ({u75:.1})?  {}  (divergence separable at 2×?)",
        if sep2 { "YES" } else { "NO" }
    );
    println!(
        "#  DIV32 p25 ({d32_25:.1}) > UNIFORM p75 ({u75:.1})? {}  (separable at 32×?)",
        if sep32 { "YES" } else { "NO" }
    );
    println!(
        "#  ⇒ {}",
        if sep32 && d32_50 / u50 > 4.0 {
            "RESOLVABLE — injected divergence shows up in wall-clock ⇒ 037/038 measurability HOLDS (still sign-only)"
        } else if sep2 {
            "WEAK — small divergence separates but the ladder is far below ideal ⇒ only LARGE divergence levers are worth A/B-ing"
        } else {
            "NOT RESOLVABLE — even 32× injected imbalance collapses to noise ⇒ no divergence lever is measurable here; 037/038 DEAD on arrival"
        }
    );
}
