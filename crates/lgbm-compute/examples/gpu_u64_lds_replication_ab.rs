//! Spike 020 GPU micro-benchmark: does PER-WARP (per-wave) LDS sub-histogram REPLICATION
//! ON TOP OF the live u64 fixed-point resident BUILD kernel give a sign-stable device-time
//! win OVER the non-replicated u64 baseline — or has the u64 fixed-point switch (spike-018/
//! 019) ALREADY captured the contention win, making replication redundant (NULL)?
//!
//! BACKGROUND
//! ----------
//! * The LIVE production resident build is `construct_leaf_hist_resident_lds_kernel_u64`
//!   (crates/lgbm-compute/src/kernels/histogram.rs:1246): one cube per feature, ONE u64
//!   fixed-point (S=2^30) sub-histogram in LDS shared by all 256 threads (8 wave32 warps),
//!   then an LDS→global merge. All 256 threads `Atomic<u64>::fetch_add` into the SAME
//!   sub-hist → 256-way intra-cube contention on the LDS atomics.
//! * Spike-017 replicated the *f32* sub-hist per-warp (R replicas; warp w → replica w%R;
//!   merge ascending). Spike-018/019 then switched the live kernel to u64 fixed-point.
//! * OPEN QUESTION: integer (u64) LDS atomics may already be cheap enough that the 256-way
//!   contention is no longer the bottleneck (it may now be scattered-read latency bound).
//!   If so, replicating the u64 sub-hist is redundant → NULL. This spike settles it on the
//!   device-time proxy by A/B-ing the SAME u64 kernel at R8 (candidate) vs R1 (baseline).
//!
//! KERNEL UNDER TEST (`build_u64_repl`)
//! ------------------------------------
//! The live u64 kernel's body verbatim — IDENTICAL resident double-indirection gather
//! (`resident_bins[f*num_data + leaf_rows[k]]`), sentinel `slot_off` of length
//! `num_features + 1`, row-partition over `CUBE_POS_Y`/`CUBE_COUNT_Y`, u64 quantize
//! `round(v * 2^30)` → i64-bits, wrapping `Atomic<u64>::fetch_add`, LDS→global merge —
//! WITH the spike-017 per-warp replication grafted on: `#[comptime] replicas` sub-hists in
//! LDS, warp `w` folds into replica `(UNIT_POS/PLANE_DIM) % R`, then the cube sums all R
//! replicas per cell in ASCENDING (deterministic) order and issues ONE global atomic.
//!
//! **R=1 reduces EXACTLY to the live non-replicated u64 kernel** (single replica, rbase=0,
//! merge loop sums one term) — so sweeping R cleanly isolates the replication effect, and
//! the R8==R1 parity assert is a real correctness gate (integer adds are order-independent,
//! so the ascending replica-merge MUST be bit-exact — any mismatch is a FINDING).
//!
//! MEASUREMENT DISCIPLINE (spike-018b CONVENTIONS — copied verbatim)
//! ----------------------------------------------------------------
//!   * Compute-throughput, NOT launch-bound: accumulate `LAUNCHES` launches into ONE reused
//!     buffer per timed sample, then read back ONCE (the single readback forces the sync).
//!   * Interleaved warm-up reps, then `REPS=9` interleaved reps; report `median[p25..p75]`.
//!   * SEP-WIN iff `cand_p75 < base_p25` (R8's box below R1's box ⇒ decisive); else overlap.
//!     `ratio = base_median / cand_median` (>1 ⇒ R8 faster).
//!   * RUN THE PROCESS ≥ 2× and check sign-stability — a single APU run is noise-confounded.
//!
//! REGIMES: HEAVY wide 16feat×1M (primary; spike-019's high-contention regime) + LIGHT
//! 16feat×200k (control, expect null). Sweep R∈{1,2,4,8} (confirm spike-017's
//! all-or-nothing-at-the-warp-boundary under u64) and P∈{1,16} (does any win compose with
//! row-partition, as fixed-point's did?).
//!
//! NOTE — the GPU here is an HSA-spoofed 8-CU APU (Radeon 860M), NOT a discrete gfx1100.
//! Absolute throughput is APU-confounded; trust only the RELATIVE A/B ratio + SEP sign.
//!
//! Run: cargo run --release --features rocm --example gpu_u64_lds_replication_ab
//!      (run it ≥ 2× and confirm the per-(regime,R,P) SEP sign is stable.)

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

/// LDS sub-hist cap PER REPLICA: 512 u64 cells = 256 bins (grad+hess interleaved) = 4 KiB.
/// Matches the live kernel's `HIST_LDS_MAX` (histogram.rs:620). u64 = 8 bytes/cell.
#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// Fixed-point quantize scale S = 2^30 — IDENTICAL to the live kernel's `SCALE_F32`
/// (histogram.rs:630). The build accumulates `round(value * S)` as a two's-complement i64
/// stored BITS-as-u64 via integer LDS atomics; dequant `(bits as i64)/S` happens later.
#[cfg(feature = "rocm")]
const SCALE_F32: f32 = 1_073_741_824.0; // 2^30

/// Per-warp-REPLICATED u64 fixed-point resident LDS build. The live
/// `construct_leaf_hist_resident_lds_kernel_u64` body with spike-017 per-warp replication.
/// `R` (comptime) u64 sub-histograms live in LDS; warp `w` accumulates into replica
/// `(UNIT_POS/PLANE_DIM) % R`; the cube sums all R replicas per cell (ascending = the same
/// deterministic order at every R) and issues ONE wrapping u64 global atomic per cell.
/// **R=1 reduces byte-for-byte to the live non-replicated u64 kernel.**
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_u64_repl<B: Int>(
    resident_bins: &Array<B>, // native bin width (here u32), widened to a u32 index
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,       // resident column stride
    out: &mut Array<Atomic<u64>>,
    #[comptime] replicas: usize,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature (× P row-partitions over CUBE_POS_Y)
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;
    let nrep = replicas;

    // R replicas, each `feat_len` active cells, packed densely (replica rr at
    // [rr*feat_len, rr*feat_len+feat_len)). Comptime size = R * HIST_LDS_MAX ≥ R*feat_len.
    let sub = SharedMemory::<Atomic<u64>>::new(replicas * HIST_LDS_MAX);

    // 1. zero the R replicas' active cells (total = nrep*feat_len), strided across the cube.
    let total = nrep * feat_len;
    let mut c = UNIT_POS as usize;
    while c < total {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();

    // 2. each warp folds into its OWN replica: replica = (warp_id) % R; base = replica*feat_len.
    //    Resident gather + row-partition IDENTICAL to the live kernel (P=CUBE_COUNT_Y).
    let replica = (UNIT_POS as usize / PLANE_DIM as usize) % nrep;
    let rbase = replica * feat_len;
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = rbase + bin * 2;
        // quantize `round(v * 2^30)` → i64 → store its bits as u64 (live build_u64 idiom).
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE_F32)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE_F32)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();

    // 3. merge: for each active cell m, sum across the R replicas (ascending = deterministic;
    //    integer adds are order-independent so the result is exact at every R) and issue ONE
    //    wrapping u64 global atomic (== two's-complement i64 add).
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        let mut acc = 0u64;
        let mut rr = 0usize;
        while rr < nrep {
            acc += sub[rr * feat_len + m].load();
            rr += 1;
        }
        out[base + m].fetch_add(acc);
        m += cd;
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100/APU). Re-run with it.");
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const NUM_BIN: usize = 256; // LightGBM default max_bin; keeps the LDS branch (feat_len ≤ 512)
    const LAUNCHES: usize = 20; // launches accumulated per timed sample (compute-throughput)
    const CUBE_DIM: u32 = 256; // matches the production launcher; 256/PLANE_DIM(32) = 8 waves
    const REPS: usize = 9; // interleaved reps for median + p25/p75

    let client = rocm_client();

    // Decisive spike-019 regimes. HEAVY wide is the high-contention regime where the f32
    // replication win (spike-017) and the u64 switch (spike-019) both landed; LIGHT is the
    // null/overlap control. Each swept over R∈{1,2,4,8} and P∈{1,16}.
    let configs: [(usize, usize, &str); 2] = [
        (16, 1_000_000, "HEAVY wide  16feat×1M  (big-root large-leaf: primary; win if u64 still contention-bound)"),
        (16, 200_000, "LIGHT       16feat×200k (low rows/cube: null/overlap acceptable)"),
    ];
    let r_sweep: [usize; 4] = [1, 2, 4, 8]; // R1=baseline; R2/R4/R8 confirm warp-boundary behavior
    let p_sweep: [u32; 2] = [1, 16]; // does any win compose with row-partition?

    let pct = |v: &mut Vec<f64>, q: f64| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    println!("# Spike 020 — per-warp LDS REPLICATION on the LIVE u64 fixed-point resident BUILD, device-time A/B");
    println!("# kernel = build_u64_repl (live construct_leaf_hist_resident_lds_kernel_u64 body + spike-017 per-warp replication).");
    println!("# baseline = R1 (== live non-replicated u64 kernel, byte-identical behavior); candidate = R8 (1 replica per wave32).");
    println!("# compute-throughput: {LAUNCHES} launches accumulated → ONE readback; median[p25..p75] over {REPS} interleaved reps.");
    println!("# SEP-WIN iff R8_p75 < R1_p25; ratio = R1_median / R8_median (>1 ⇒ R8 faster). RUN ≥2× — verdict is the stable SEP sign.");
    println!("# LDS/cube at R8 = R*HIST_LDS_MAX*8 = {} B (must fit 64 KiB). APU=8 CU; absolute Mr/s APU-confounded.", 8 * HIST_LDS_MAX * 8);

    for (feats, num_data, label) in configs {
        // Synthetic resident inputs (feature-major resident bins f*num_data + row; double
        // indirection through leaf_rows). Same generator as the spike-018/019 harnesses.
        let mut bins: Vec<u32> = Vec::with_capacity(feats * num_data);
        for f in 0..feats {
            for row in 0..num_data {
                let h = (row as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                bins.push((h % NUM_BIN as u64) as u32);
            }
        }
        let leaf_rows: Vec<u32> = (0..num_data as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761)) % num_data as u32)
            .collect();
        let ord_g: Vec<f32> = (0..num_data).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let ord_h: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();

        let feat_len = (2 * NUM_BIN) as u32;
        let slot_off: Vec<u32> = (0..=feats as u32).map(|f| f * feat_len).collect();
        let slot_len = feats * 2 * NUM_BIN;
        let r = leaf_rows.len();
        let reads_per_launch = (feats * r) as f64;

        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        macro_rules! launch {
            ($p:expr, $out:expr, $rep:expr) => {
                build_u64_repl::launch_unchecked::<u32, _>(
                    &client,
                    CubeCount::Static(feats as u32, $p, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), feats * num_data),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), feats + 1),
                    num_data,
                    ArrayArg::from_raw_parts($out.clone(), slot_len),
                    $rep,
                )
            };
        }

        // One clean launch into a freshly-zeroed out; returns the raw u64 histogram cells.
        let run_once = |p: u32, rep: usize| -> Vec<u64> {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            unsafe {
                launch!(p, out, rep);
            }
            u64::from_bytes(&client.read_one_unchecked(out)).to_vec()
        };

        // accumulate-LAUNCHES-then-ONE-read (compute-throughput, NOT launch-bound).
        let bench = |p: u32, rep: usize| -> f64 {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    launch!(p, out, rep);
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed().as_secs_f64() * 1e3
        };

        println!("\n## REGIME: {label}");

        // ---- PARITY: R8 must be BIT-EXACT to R1 (raw u64 cells), at P=1 and P=16. Also a
        // CPU integer reference for feature 0 (sum of quantized grad/hess per bin) proving
        // CORRECTNESS, not merely R1==R8.
        let cpu_ref_feat0 = |bins_f0: &[u32]| -> Vec<u64> {
            let mut h = vec![0i64; 2 * NUM_BIN];
            for k in 0..r {
                let b = bins_f0[leaf_rows[k] as usize] as usize;
                let qg = (ord_g[k] * SCALE_F32).round() as i64;
                let qh = (ord_h[k] * SCALE_F32).round() as i64;
                h[b * 2] = h[b * 2].wrapping_add(qg);
                h[b * 2 + 1] = h[b * 2 + 1].wrapping_add(qh);
            }
            h.iter().map(|&x| x as u64).collect()
        };
        let cpu_f0 = cpu_ref_feat0(&bins[0..num_data]);
        for &p in &p_sweep {
            let r1 = run_once(p, 1);
            let r8 = run_once(p, 8);
            let bit_exact = r1 == r8;
            // CPU-ref check on feature 0 (cells [0, 2*NUM_BIN)). The host f32 quantize-
            // multiply (`v * 2^30`) can differ from the device's by a few quantize ULP per
            // row (different f32 rounding / FMA contraction), so EXACT host==device is not
            // expected; the real correctness gate is `bit-exact R8==R1` (exact integer
            // replica-merge). We report exact equality AND the max per-cell quantize-unit
            // gap (|Δ|/S in grad units) so a near-zero gap proves the histogram is the
            // right sum, not a structural bug.
            let cpu_exact = r8[0..2 * NUM_BIN] == cpu_f0[..];
            let mut max_unit_gap = 0.0f64;
            for i in 0..2 * NUM_BIN {
                let d = (r8[i] as i64).wrapping_sub(cpu_f0[i] as i64);
                max_unit_gap = max_unit_gap.max((d as f64).abs() / SCALE_F32 as f64);
            }
            let cpu_ok = cpu_exact || max_unit_gap < 1e-2;
            // Find first mismatching cell for diagnosis if any.
            let mut first_mismatch = None;
            for i in 0..slot_len {
                if r1[i] != r8[i] {
                    first_mismatch = Some((i, r1[i], r8[i]));
                    break;
                }
            }
            println!(
                "  PARITY P={p:>2}: R8==R1 bit-exact={bit_exact}  cpu_ref(feat0): exact={cpu_exact} max_unit_gap={max_unit_gap:.2e} ok={cpu_ok}{}",
                match first_mismatch {
                    Some((i, a, b)) => format!("  !!! FIRST MISMATCH cell {i}: R1={a} R8={b}"),
                    None => String::new(),
                }
            );
            if !bit_exact {
                eprintln!("  *** FINDING: R8 != R1 bit-exact at P={p} — integer replica-merge SHOULD be exact!");
            }
            if !cpu_ok {
                eprintln!("  *** FINDING: R8 feature-0 histogram diverges from CPU integer reference at P={p} beyond quantize-ULP tolerance (max_unit_gap={max_unit_gap:.2e})!");
            }
        }

        // ---- A/B device-time. warm-up the full (R,P) grid, then interleaved REPS.
        for &p in &p_sweep {
            for &rep in &r_sweep {
                let _ = bench(p, rep);
            }
        }

        for &p in &p_sweep {
            // collect samples for every R at this P, interleaved within each rep.
            let mut samples: Vec<Vec<f64>> = vec![vec![]; r_sweep.len()];
            for _ in 0..REPS {
                for (i, &rep) in r_sweep.iter().enumerate() {
                    samples[i].push(bench(p, rep));
                }
            }
            let mut med = vec![0.0f64; r_sweep.len()];
            let mut p25 = vec![0.0f64; r_sweep.len()];
            let mut p75 = vec![0.0f64; r_sweep.len()];
            for i in 0..r_sweep.len() {
                p25[i] = pct(&mut samples[i], 0.25);
                p75[i] = pct(&mut samples[i], 0.75);
                med[i] = pct(&mut samples[i], 0.5);
            }
            let base_med = med[0]; // R1
            let base_p25 = p25[0];
            println!("  -- P={p:>2}  (R1 = baseline) --");
            for (i, &rep) in r_sweep.iter().enumerate() {
                let mrs = reads_per_launch * LAUNCHES as f64 / (med[i] / 1e3) / 1e6;
                let ratio = base_med / med[i];
                let sep = if rep != 1 && p75[i] < base_p25 { "SEP-WIN" } else { "overlap" };
                println!(
                    "    R{rep}: {:.0}ms[{:.0}..{:.0}]  {mrs:.0}Mr/s  ratio R1/R{rep}={ratio:.2}x  {sep}",
                    med[i], p25[i], p75[i]
                );
            }
        }
    }

    println!("\n# Diagnosis:");
    println!("#  R8 SEP-WIN over R1 (sign-stable across ≥2 runs) ⇒ u64 LDS atomics STILL contention-bound; replication wins.");
    println!("#  R8 overlap/regress vs R1 ⇒ NULL — the u64 fixed-point switch ALREADY captured the contention win (replication redundant).");
    println!("#  R2/R4 should mirror spike-017's all-or-nothing-at-the-warp-boundary; P=16 tests whether any win composes with row-partition.");
    println!("#  (8-CU APU: absolute Mr/s is APU-confounded; only the RELATIVE R1/R ratio + SEP sign are load-bearing. Run the PROCESS ≥2×.)");
}
