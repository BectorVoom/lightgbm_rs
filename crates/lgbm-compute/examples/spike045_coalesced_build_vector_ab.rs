//! Spike 045 — the COALESCED-BUILD + `Vector<P,N>` rewrite (the user's "coalesced reads,
//! LDS-buffered scattered writes" architecture). The ONE un-probed cell after 030/043.
//!
//! ## What 030 and 043 already settled (don't re-measure)
//! - 043: vectorizing grad/hess on the PERMUTED gather is null→regression — because the
//!   dominant cost is the un-vectorizable permuted bin gather (`bins[col+leaf_rows[k]]`),
//!   and `Vector` only loads CONTIGUOUS addresses.
//! - 030: a COALESCED scalar build (`bins[col+k]`) is ~1.4× over the REAL monotone leaf
//!   order (not the 5–10× the random probe overstated), and the reorder PASS "can't
//!   amortize — read-once". But 030 NEVER timed the reorder pass, and never vectorized.
//!
//! ## The genuinely new question (this spike)
//! The user's architecture removes 043's blocker by REORDERING each leaf's rows contiguous
//! FIRST, then reading grad/hess/bin coalesced as `Vector<P,N>` and scattering into LDS.
//! Two things nobody measured:
//!   (A) NET: is `t(reorder gather-copy) + t(coalesced build)` < `t(permuted build)`?
//!       (030 assumed reorder free; here we time it. A write-coalesced streaming gather copy
//!        may be cheaper per byte than the same gather embedded in the atomic build's stall.)
//!   (B) Does `Vector<P,N>` finally pay on the COALESCED layout (COAL_V vs COAL_S) — the cell
//!       043 was structurally blocked from testing? If yes, it transfers to discrete gfx110x
//!       where the permuted penalty (030) is harsher than on this shared-DDR5 APU.
//!
//! Arms (all build the byte-identical u64 histogram, P=1 wide, the HONEST monotone order):
//!   FULL    permuted scalar build over the monotone 50%-subset      (030's real baseline)
//!   REORDER gather-copy bins/g/h into leaf order (write-coalesced)  (the price of admission)
//!   COAL_S  coalesced SCALAR build over the reordered arrays        (030's COAL ceiling)
//!   COAL_V  coalesced VECTOR build: Vector<N> loads + N LDS scatters (THE new lever)
//!
//! Bit-exact by construction: identical `round(v·2^30)`→i64-bits→u64 atomic add; integer
//! adds are order-independent ⇒ all arms produce the byte-identical histogram. (Reorder is a
//! pure permutation of the SAME rows ⇒ same multiset of adds.)
//!
//! HIP-ONLY: the u64-atomic build needs `Atomic<u64>`, which cubecl-cpu does not implement
//! (panics) — exactly why the CPU anchor uses the f64 non-atomic fold (018b). Run twice:
//!   cargo run --release --features rocm --example spike045_coalesced_build_vector_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("spike-045 is hip-only (u64 atomics); rebuild with --features rocm.");
}

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_U64: usize = 512; // 2 u64/bin, NUM_BIN ≤ 256
#[cfg(feature = "rocm")]
const SCALE: f32 = 1_073_741_824.0; // 2^30 (== production SCALE_F32)

// ─── FULL: permuted scalar build (byte-faithful to spike-043 build_scalar / the prod kernel). ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_full(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    num_data: usize,
    feat_len: usize,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let base = f * feat_len;
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let col = f * num_data;
    let mut k = UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ─── REORDER (bins): per feature, write-coalesced gather `bins_c[col+k] = bins[col+lr[k]]`.
//     This MOVES the permuted gather out of the atomic build into a pure streaming copy. ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn reorder_bins(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    num_data: usize,
    r: usize,
    out: &mut Array<u32>,
) {
    let f = CUBE_POS_X as usize;
    let col_src = f * num_data; // source: bins stride = num_data
    let col_dst = f * r; // dest: bins_c stride = r (leaf row count), NOT num_data
    let cd = CUBE_DIM as usize;
    let mut k = UNIT_POS as usize;
    while k < r {
        out[col_dst + k] = resident_bins[col_src + leaf_rows[k] as usize];
        k += cd;
    }
}

// ─── COAL_S: coalesced SCALAR build over the reordered arrays (`col + k`, sequential). ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_coal_scalar(
    bins_c: &Array<u32>,
    g_c: &Array<f32>,
    h_c: &Array<f32>,
    num_data: usize,
    feat_len: usize,
    r: usize,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let base = f * feat_len;
    let cd = CUBE_DIM as usize;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let col = f * num_data;
    let mut k = UNIT_POS as usize;
    while k < r {
        let bin = bins_c[col + k] as usize; // COALESCED
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(g_c[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(h_c[k] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ─── COAL_V: coalesced VECTOR build. Read N consecutive rows' bin/g/h as one `Vector<_,N>`
//     load each (contiguous ⇒ vectorizable, unlike 043's permuted gather), then unroll N
//     scalar LDS scatters — the user's "coalesced vector read → scattered LDS accumulate".
//     `#[unroll] for j in 0..N::value()` is the cubecl idiom (runtime_tests/vector.rs:80) —
//     `N::value()` stays comptime so `vb[j]` is a comptime lane index. (An intermediate
//     `let nlanes = N::value() as usize` breaks comptime ⇒ runtime Vector index ⇒ segfault.) ───
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn build_coal_vec<N: Size>(
    bins_c: &Array<Vector<u32, N>>,
    g_c: &Array<Vector<f32, N>>,
    h_c: &Array<Vector<f32, N>>,
    rows_v: usize, // r / N  (rows per feature, in vector units)
    feat_len: usize,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let base = f * feat_len;
    let cd = CUBE_DIM as usize;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let col_v = f * rows_v;
    let mut kv = UNIT_POS as usize;
    while kv < rows_v {
        let vb = bins_c[col_v + kv]; // coalesced vector load (N bins)
        let vg = g_c[kv];
        let vh = h_c[kv];
        #[unroll]
        for j in 0..N::value() {
            let ti = vb[j] as usize * 2;
            sub[ti].fetch_add(u64::cast_from(i64::cast_from(f32::round(vg[j] * SCALE))));
            sub[ti + 1].fetch_add(u64::cast_from(i64::cast_from(f32::round(vh[j] * SCALE))));
        }
        kv += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const CUBE_DIM: u32 = 256;
    const LAUNCHES: usize = 10;
    const REPS: usize = 5;
    const FEATS: usize = 500; // wide ⇒ P=1
    let client = rocm_client();
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    // 250k = CPU beats GPU ~4×; 1M = the crossover regime (matches spike-030).
    let only_250k = std::env::var("S045_250K").is_ok();
    let configs: &[(usize, usize)] = if only_250k {
        &[(250_000, 256)]
    } else {
        &[(250_000, 256), (1_000_000, 256)]
    };

    for &(num_data, num_bin) in configs {
        let feat_len = 2 * num_bin;
        let slot_len = FEATS * feat_len;
        // bins: random within a feature column ⇒ cache-hostile gather (== 030/043 data gen).
        let bins: Vec<u32> = (0..FEATS * num_data)
            .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % num_bin as u64) as u32)
            .collect();
        // HONEST order: the stable-partition leaf order is a MONOTONE subset (030's decisive
        // caveat) — a 50%-selectivity leaf = every other row, sorted. NOT a random permutation.
        let leaf_rows: Vec<u32> = (0..num_data as u32).step_by(2).collect();
        let r = leaf_rows.len(); // rows in this leaf
        let ord_g: Vec<f32> = (0..num_data).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let ord_h: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();

        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));

        // reorder destinations (contiguous per feature in leaf order).
        let bins_c_len = FEATS * r;
        let d_bins_c = client.create_from_slice(u32::as_bytes(&vec![0u32; bins_c_len]));

        let count = CubeCount::Static(FEATS as u32, 1, 1);
        let dim = CubeDim::new_1d(CUBE_DIM);

        // ── timed closures ──
        let bench_full = || -> f64 {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_full::launch_unchecked(
                        &client, count.clone(), dim.clone(),
                        ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                        ArrayArg::from_raw_parts(d_rows.clone(), r),
                        ArrayArg::from_raw_parts(d_g.clone(), r),
                        ArrayArg::from_raw_parts(d_h.clone(), r),
                        num_data, feat_len,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed().as_secs_f64() * 1e3
        };

        // REORDER pass = the BINS gather only (write-coalesced). grad/hess are ALREADY
        // leaf-ordered in production (`ordered_gradients_`, read at sequential k by build_full),
        // so they need NO reorder — only the global bin matrix does. This is the price of
        // admission for the coalesced build.
        let bench_reorder = || -> f64 {
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    reorder_bins::launch_unchecked(
                        &client, count.clone(), dim.clone(),
                        ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                        ArrayArg::from_raw_parts(d_rows.clone(), r),
                        num_data, r,
                        ArrayArg::from_raw_parts(d_bins_c.clone(), bins_c_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(d_bins_c.clone());
            t.elapsed().as_secs_f64() * 1e3
        };

        // materialize the reordered arrays once for the coalesced builds.
        unsafe {
            reorder_bins::launch_unchecked(
                &client, count.clone(), dim.clone(),
                ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                ArrayArg::from_raw_parts(d_rows.clone(), r),
                num_data, r,
                ArrayArg::from_raw_parts(d_bins_c.clone(), bins_c_len),
            );
        }
        let _ = client.read_one_unchecked(d_bins_c.clone());

        let bench_coal_s = || -> f64 {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_coal_scalar::launch_unchecked(
                        &client, count.clone(), dim.clone(),
                        ArrayArg::from_raw_parts(d_bins_c.clone(), bins_c_len),
                        ArrayArg::from_raw_parts(d_g.clone(), r),
                        ArrayArg::from_raw_parts(d_h.clone(), r),
                        r, feat_len, r,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed().as_secs_f64() * 1e3
        };

        // COAL_V: needs r % N == 0. r = num_data/2 (125000 / 500000) → div by 4 and 2 OK.
        // Fixed-N kernels: no runtime N positional arg; array lengths in vector units.
        let bench_coal_v = |nwidth: usize| -> (f64, Vec<u8>) {
            assert!(r % nwidth == 0, "r={r} not divisible by N={nwidth}");
            let rows_v = r / nwidth;
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_coal_vec::launch_unchecked(
                        &client, count.clone(), dim.clone(),
                        nwidth, // N runtime value, right after CubeDim
                        ArrayArg::from_raw_parts(d_bins_c.clone(), bins_c_len / nwidth),
                        ArrayArg::from_raw_parts(d_g.clone(), r / nwidth),
                        ArrayArg::from_raw_parts(d_h.clone(), r / nwidth),
                        rows_v, feat_len,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                    );
                }
            }
            let got = client.read_one_unchecked(out).to_vec();
            (t.elapsed().as_secs_f64() * 1e3, got)
        };

        // ── bit-exact gate: FULL vs COAL_S vs COAL_V(4) must be byte-identical ──
        let bytes_of = |run: &dyn Fn() -> Vec<u8>| run();
        let full_bytes = {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            unsafe {
                build_full::launch_unchecked(
                    &client, count.clone(), dim.clone(),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    num_data, feat_len,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
            client.read_one_unchecked(out).to_vec()
        };
        let coal_s_bytes = {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            unsafe {
                build_coal_scalar::launch_unchecked(
                    &client, count.clone(), dim.clone(),
                    ArrayArg::from_raw_parts(d_bins_c.clone(), bins_c_len),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    r, feat_len, r,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
            client.read_one_unchecked(out).to_vec()
        };
        let coal_v4_bytes = {
            // SINGLE launch (accumulating kernel — N launches would give N× the histogram,
            // the 037/038 lesson). Matches the single-launch full_bytes/coal_s_bytes.
            let nwidth = 4usize;
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            unsafe {
                build_coal_vec::launch_unchecked(
                    &client, count.clone(), dim.clone(),
                    nwidth,
                    ArrayArg::from_raw_parts(d_bins_c.clone(), bins_c_len / nwidth),
                    ArrayArg::from_raw_parts(d_g.clone(), r / nwidth),
                    ArrayArg::from_raw_parts(d_h.clone(), r / nwidth),
                    r / nwidth, feat_len,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
            client.read_one_unchecked(out).to_vec()
        };
        let _ = bytes_of;
        let exact_s = full_bytes == coal_s_bytes;
        let exact_v = full_bytes == coal_v4_bytes;

        // warm
        for _ in 0..2 {
            let _ = bench_full();
            let _ = bench_reorder();
            let _ = bench_coal_s();
            let _ = bench_coal_v(4);
        }

        let (mut sf, mut sr, mut ss, mut sv2, mut sv4) =
            (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..REPS {
            sf.push(bench_full());
            sr.push(bench_reorder());
            ss.push(bench_coal_s());
            sv2.push(bench_coal_v(2).0);
            sv4.push(bench_coal_v(4).0);
        }
        let f50 = pct(&mut sf, 0.5);
        let r50 = pct(&mut sr, 0.5);
        let s50 = pct(&mut ss, 0.5);
        let v2 = pct(&mut sv2, 0.5);
        let v4 = pct(&mut sv4, 0.5);

        println!("\n## {FEATS} feat × {num_data} rows × {num_bin} bins  (P=1, monotone 50% leaf, r={r})");
        println!("#   FULL    (permuted scalar)   {f50:8.1} ms   [baseline]");
        println!("#   REORDER (gather-copy pass)  {r50:8.1} ms   ({:.0}% of FULL)", r50 / f50 * 100.0);
        println!("#   COAL_S  (coalesced scalar)  {s50:8.1} ms   ({:.2}× vs FULL)   bit_exact={exact_s}", f50 / s50);
        println!("#   COAL_V2 (coalesced vec2)    {v2:8.1} ms   ({:.2}× vs FULL, {:.2}× vs COAL_S)", f50 / v2, s50 / v2);
        println!("#   COAL_V4 (coalesced vec4)    {v4:8.1} ms   ({:.2}× vs FULL, {:.2}× vs COAL_S)   bit_exact={exact_v}", f50 / v4, s50 / v4);
        println!("#   ── NET (production-honest): REORDER + best-COAL vs FULL ──");
        let best_coal = s50.min(v2).min(v4);
        let net = r50 + best_coal;
        println!("#   NET = REORDER + best_COAL = {r50:.1} + {best_coal:.1} = {net:.1} ms  ⇒ {:.2}× vs FULL  ({})",
            f50 / net, if net < f50 { "WIN" } else { "LOSS — reorder eats the coalescing win" });
    }

    println!("\n# KEY: COAL_V/COAL_S answers 043's reopened cell (does Vector pay once contiguous?).");
    println!("#      NET answers whether the whole architecture beats the permuted build at all.");
}
