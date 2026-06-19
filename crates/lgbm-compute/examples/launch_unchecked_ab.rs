//! A/B micro-bench: `#[cube(launch)]` (in-kernel bounds-check codegen) vs
//! `#[cube(launch_unchecked)]` (codegen dropped) for the 3 HOT-LOOP production
//! histogram kernels on the local gfx1100 — quick task 260619-ol8 (OL8-01).
//!
//! WHAT THIS QUANTIFIES: quick-260619-nrw swept all 8 production rocm-gated
//! histogram kernels from `#[cube(launch)]` to `#[cube(launch_unchecked)]`
//! (drops the per-access bounds-branch codegen in the hot scatter loops) but
//! could only report the win QUALITATIVELY. quick-260619-mwr proved WHY: the
//! attribute is COMPTIME — it cannot be toggled at runtime in a single binary —
//! AND that the realistic regime is TRANSFER-bound, which masks launch overhead.
//! This bench closes that gap with the only A/B that IS possible in one process:
//! a `_checked` TWIN (identical body, `#[cube(launch)]`) defined beside each
//! shipped `_unchecked` kernel, launched interleaved, so the delta isolates
//! EXACTLY the bounds-check codegen and nothing else.
//!
//! WHY TWINS EXIST: `launch` vs `launch_unchecked` is a comptime kernel attribute;
//! you cannot select it at runtime. So the only way to A/B in one binary is to
//! compile BOTH a checked and an unchecked variant of the same body. The shipped
//! kernels are `launch_unchecked`; the bench-only `_checked` twins below are
//! BYTE-IDENTICAL bodies annotated `#[cube(launch)]`. Only the attribute differs.
//!
//! MEASUREMENT-ONLY: these twins are NOT wired into lib.rs / the learner / the
//! wired path; production kernels, launchers, and the CPU f64 anchor are
//! UNTOUCHED. The only new artifacts of this task are this example + a FINDINGS.md.
//!
//! *** SYNC WARNING (load-bearing maintenance contract) ***
//! The `_checked` twins below DUPLICATE the shipped production kernel bodies for
//! timing only. Any future edit to a shipped kernel (`construct_hist_kernel_atomic_f32`,
//! `construct_leaf_hist_resident_lds_kernel`, `build_fix_scan_fused_kernel` in
//! `crates/lgbm-compute/src/kernels/histogram.rs`) MUST be mirrored here, or the A/B
//! silently compares two DIFFERENT computations and the result is meaningless. The
//! same-input sanity assert in `main()` (f32-atomic envelope for the atomic/LDS
//! kernels, bit-equal for the deterministic fused kernel) catches divergence at
//! runtime — if a production edit diverges the bodies, the assert fails loudly.
//!
//! WARM-VS-COLD (load-bearing, spike-findings SKILL — cold-ceiling-overstates-warm):
//! WARMUP discarded launches per arm, MEDIAN of N timed launches, a device sync
//! (result read-back) FORCED INSIDE every timed call, checked/unchecked arms
//! INTERLEAVED so thermal/clock drift hits both equally, and the bench is meant to
//! be re-run across >=2 process restarts to confirm the delta SIGN is stable vs noise.
//!
//! This is a MANUAL rocm bench — NOT a test gate.
//!
//! Run: cargo run --release -p lgbm-compute --features rocm --example launch_unchecked_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100). Re-run with it.");
}

// ===========================================================================
// BENCH-ONLY `_checked` TWIN KERNELS.
//
// Each body is COPIED VERBATIM from the shipped kernel in
// `crates/lgbm-compute/src/kernels/histogram.rs`; the ONLY difference is the
// launch attribute — `#[cube(launch)]` here vs the shipped `#[cube(launch_unchecked)]`
// — so the A/B isolates exactly the in-kernel bounds-check codegen. See the
// SYNC WARNING in the module doc above.
// ===========================================================================

// The `#[cube]` macro expansion needs the cubecl prelude traits in scope at the
// item level (same as histogram.rs's module-level `use cubecl::prelude::*;`).
#[cfg(feature = "rocm")]
use cubecl::prelude::*;

/// LDS sub-histogram cap, re-declared locally because the production
/// `HIST_LDS_MAX` in histogram.rs is private. MUST equal the production const
/// (512 = 2 * 256, one feature <= 256 bins).
#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// CHECKED twin of `construct_hist_kernel_atomic_f32` (histogram.rs ~388).
/// Body byte-identical; attribute `#[cube(launch)]` (shipped is launch_unchecked).
#[cfg(feature = "rocm")]
#[cube(launch)]
pub fn construct_hist_kernel_atomic_f32_checked(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    // Bounds check: the launch rounds the unit count up to a multiple of the cube
    // dim, so the tail units (idx >= len) must stay idle (manual §4 Safe Indexing).
    if idx < binned.len() {
        let ti = binned[idx] as usize * 2; // grad cell at bin<<1, hess at +1
        out[ti].fetch_add(grad[idx]);
        out[ti + 1].fetch_add(hess[idx]);
    }
}

/// CHECKED twin of `construct_leaf_hist_resident_lds_kernel` (histogram.rs ~931).
/// Body byte-identical; attribute `#[cube(launch)]` (shipped is launch_unchecked).
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_resident_lds_kernel_checked(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    // 1. zero this feature's active LDS cells.
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    // 2. scatter THIS partition's strided rows into LDS (resident on-device gather).
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    // 3. merge LDS → this feature's global slot.
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// CHECKED twin of `build_fix_scan_fused_kernel` (histogram.rs ~2041).
/// Body byte-identical; attribute `#[cube(launch)]` (shipped is launch_unchecked).
/// Calls the SAME shared `split_scan_body` via the public crate path (the bench is
/// an external example crate, so `lgbm_compute::kernels::split::split_scan_body`).
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_scan_fused_kernel_checked(
    // Device-resident binned columns (feature-major, `f*num_data + row`) — INPUT.
    resident_bins: &Array<u32>,
    // The leaf's row indices (subset of 0..num_data) — INPUT.
    leaf_rows: &Array<u32>,
    // The leaf's grad/hess gathered host-side in leaf_rows order — INPUT (f32).
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    // f64 fixed+compacted histogram — OUTPUT (caller zeroes it before launch).
    hist: &mut Array<f64>,
    // RAW 12-cell-per-feature SplitInfo — OUTPUT.
    out: &mut Array<f64>,
    // Per-feature params (length == num_features).
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    most_freq_bin: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    scan_active: &Array<u32>,
    // Stride of a resident column = full train row count.
    num_data_stride: usize,
    // LEAF-LEVEL scalars (shared across the batch).
    sum_gradient_raw: f64,
    sum_hessian_raw: f64,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_hessian_bumped: f64,
    num_data: i32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    let mfb = most_freq_bin[fi];
    let off = offset[fi];

    // ---- Stage 1: SEQUENTIAL f64 BUILD (ascending leaf-row order = cpu anchor) ----
    for w in 0..nb {
        let wbi = base + (w as usize) * 2;
        hist[wbi] = 0.0;
        hist[wbi + 1] = 0.0;
    }
    let rows = ord_g.len();
    for k in 0..rows {
        let row = leaf_rows[k] as usize;
        let bin = resident_bins[fi * num_data_stride + row];
        let cell = base + bin as usize * 2;
        hist[cell] += f64::cast_from(ord_g[k]);
        hist[cell + 1] += f64::cast_from(ord_h[k]);
    }

    // ---- Stage 2: FIX (fix_compact_kernel:674-703, VERBATIM) ----
    let do_fix = mfb > 0 && mfb < nb;
    if do_fix {
        let mfbu = mfb as usize;
        let mut g = 0.0f64;
        let mut h = 0.0f64;
        g += sum_gradient_raw;
        h += sum_hessian_raw;
        let count = nb;
        for i in 0..count {
            let bi = base + (i as usize) * 2;
            let gi = hist[bi];
            let hi = hist[bi + 1];
            let take = i != mfb;
            g -= select(take, gi, 0.0);
            h -= select(take, hi, 0.0);
        }
        let mi = base + mfbu * 2;
        hist[mi] = g;
        hist[mi + 1] = h;
    }

    // ---- Stage 3: COMPACT (fix_compact_kernel:705-732, VERBATIM) ----
    if off > 0 {
        if off >= nb {
            for c in 0..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        } else {
            let keep = nb - off;
            for c in 0..keep {
                let dst = base + (c as usize) * 2;
                let src = base + ((c + off) as usize) * 2;
                hist[dst] = hist[src];
                hist[dst + 1] = hist[src + 1];
            }
            for c in keep..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        }
    }

    // ---- Stage 4: SCAN (shared split_scan_body, split.rs:144) ----
    if scan_active[fi] != 0 {
        lgbm_compute::kernels::split::split_scan_body(
            hist,
            slot_off[fi],
            out,
            f * 12u32,
            nb,
            off,
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient_raw,
            sum_hessian_bumped,
            num_data,
            rev_count[fi],
            fwd_count[fi],
        );
    }
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::kernels::histogram::construct_histograms_parallel_f32_on;
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    let client = rocm_client();

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

    // Same-input correctness check (f32-atomic envelope, NOT the parity gate). The
    // real GPU-vs-CPU-f64-anchor parity gate stays in rocm_cuda_mirror.rs / kernel_parity.
    // This ALSO catches twin-vs-shipped DRIFT (the SYNC WARNING): if a future production
    // edit diverges the bodies, the checked and unchecked arms disagree and this fails.
    let assert_same_input_f32 = |a: &[f64], b: &[f64], cell: &str| {
        const ABS: f64 = 5e-6;
        const REL: f64 = 1e-5;
        assert_eq!(a.len(), b.len(), "{cell}: histogram length mismatch (twin drift?)");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let diff = (x - y).abs();
            let tol = ABS + REL * x.abs();
            assert!(
                diff <= tol,
                "{cell}: cell {i} diverged beyond the f32-atomic envelope — \
                 checked {x} vs unchecked {y} (|diff| {diff} > tol {tol}); \
                 twin body may have drifted from the shipped kernel (SYNC WARNING)"
            );
        }
    };
    // Fused kernel is f64/deterministic ⇒ checked and unchecked MUST be BIT-IDENTICAL.
    let assert_same_input_bits = |a: &[f64], b: &[f64], cell: &str| {
        assert_eq!(a.len(), b.len(), "{cell}: fused output length mismatch (twin drift?)");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.to_bits() == y.to_bits(),
                "{cell}: fused cell {i} NOT bit-equal — checked {x} vs unchecked {y}; \
                 the f64 deterministic fused twin diverged from the shipped kernel (SYNC WARNING)"
            );
        }
    };

    // Deterministic feature-major bin generator (no RNG; same hash as the other harnesses).
    let gen_resident = |feats: usize, num_data: usize, num_bin: u32| -> Vec<u32> {
        let mut resident = Vec::with_capacity(feats * num_data);
        for f in 0..feats {
            for row in 0..num_data {
                let h = (row as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((f as u64).wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                resident.push((h % num_bin as u64) as u32);
            }
        }
        resident
    };
    let gen_grad = |n: usize| -> Vec<f32> { (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect() };
    let gen_hess = |n: usize| -> Vec<f32> { (0..n).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect() };

    const BIN_SWEEP: [u32; 3] = [16, 64, 256];

    println!("# launch_unchecked vs launch (bounds-check codegen) A/B  —  gfx1100, 260619-ol8");
    println!("# checked   arm = #[cube(launch)]          twin (per-access bounds-branch codegen ON)");
    println!("# unchecked arm = #[cube(launch_unchecked)] shipped (bounds-branch codegen DROPPED)");
    println!("# INTERLEAVED checked/unchecked per timed iter; WARMUP discarded; median + p25/p75 spread.");
    println!("# device sync FORCED inside each timed call (out[0] / read_one_unchecked read-back).");
    println!("# delta% = (checked - unchecked) / checked * 100   (>0 => unchecked faster)");
    println!("# RE-RUN across >=2 process restarts to confirm the delta SIGN is stable vs noise.");
    println!();

    // ===================================================================
    // KERNEL 1 — f32-atomic (construct_hist_kernel_atomic_f32).
    // Launch-bound cells: small leaf, minimal transfer, many cheap launches (high TIMED).
    // Compute-bound contrast: large leaf so the 2*n scatter dominates.
    // ===================================================================
    println!("## kernel 1: f32-atomic histogram (construct_hist_kernel_atomic_f32)");
    println!(
        "{:>12} | {:>5} | {:>11} | {:>11} | {:>8} | {:>17} | {:>17}",
        "regime", "bins", "checked_ms", "unchk_ms", "delta%", "checked p25/p75", "unchk p25/p75"
    );
    println!("{:-<13}+{:-<7}+{:-<13}+{:-<13}+{:-<10}+{:-<19}+{:-<19}", "", "", "", "", "", "", "");
    // (regime label, n rows, feats unused for atomic, warmup, timed)
    let atomic_cells: [(&str, usize, usize, usize); 2] = [
        ("launch-bound", 2_048, 15, 21),
        ("compute-bound", 200_000, 5, 7),
    ];
    for (regime, n, warmup, timed) in atomic_cells {
        for &num_bin in &BIN_SWEEP {
            // binned[i] in [0, num_bin); deterministic hash.
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
            let out_len = 2 * num_bin as usize;

            // UNCHECKED arm: the shipped launcher (internally ::launch_unchecked).
            let unchecked_call = || construct_histograms_parallel_f32_on(&client, &binned, &grad, &hess, num_bin).unwrap();

            // CHECKED arm: replicate the shipped launcher body inline but call the
            // _checked twin's SAFE ::launch — identical uploads / cube count / readback,
            // ONLY the launch attribute differs.
            let checked_call = || {
                let h_bin = client.create_from_slice(u32::as_bytes(&binned));
                let h_grad = client.create_from_slice(f32::as_bytes(&grad));
                let h_hess = client.create_from_slice(f32::as_bytes(&hess));
                let zeros = vec![0.0f32; out_len];
                let h_out = client.create_from_slice(f32::as_bytes(&zeros));
                let cube_dim = 256u32;
                let cube_count = (n as u32).div_ceil(cube_dim);
                // SAFETY: same host-proven in-range obligations as the shipped
                // construct_histograms_parallel_f32_on launcher — binned/grad/hess sized n,
                // out sized out_len = 2*num_bin, the kernel guards idx < n, and every
                // binned[i] < num_bin keeps out[bin*2 + 1] in range. ::launch keeps the
                // bounds-check codegen; the unchecked arm drops it — the only difference.
                unsafe {
                    construct_hist_kernel_atomic_f32_checked::launch(
                        &client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new_1d(cube_dim),
                        ArrayArg::from_raw_parts(h_bin.clone(), n),
                        ArrayArg::from_raw_parts(h_grad.clone(), n),
                        ArrayArg::from_raw_parts(h_hess.clone(), n),
                        ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    );
                }
                let bytes = client.read_one_unchecked(h_out);
                f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect::<Vec<f64>>()
            };

            // Interleaved warmup + timing.
            for _ in 0..warmup {
                let _ = checked_call();
                let _ = unchecked_call();
            }
            let mut c_ms = Vec::with_capacity(timed);
            let mut u_ms = Vec::with_capacity(timed);
            let mut c_hist = Vec::new();
            let mut u_hist = Vec::new();
            for _ in 0..timed {
                let t = Instant::now();
                let ch = checked_call();
                c_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let uh = unchecked_call();
                u_ms.push(t.elapsed().as_secs_f64() * 1e3);
                c_hist = ch;
                u_hist = uh;
            }
            let cell = format!("atomic {regime} bins={num_bin}");
            assert_same_input_f32(&c_hist, &u_hist, &cell);
            let c_med = median(&mut c_ms);
            let u_med = median(&mut u_ms);
            let (cl, ch_) = spread(&c_ms);
            let (ul, uh_) = spread(&u_ms);
            let delta = (c_med - u_med) / c_med * 100.0;
            println!(
                "{:>12} | {:>5} | {:>11.4} | {:>11.4} | {:>7.2} | {:>7.4}/{:>7.4} | {:>7.4}/{:>7.4}",
                regime, num_bin, c_med, u_med, delta, cl, ch_, ul, uh_
            );
        }
    }
    println!();

    // ===================================================================
    // KERNEL 2 — resident-LDS (construct_leaf_hist_resident_lds_kernel), P=1.
    // Replicate the LDS-branch host setup of resident_raw_build_into inline; upload the
    // resident bins ONCE per cell (outside the timed loop, per mirror_vs_lds.rs).
    // ===================================================================
    println!("## kernel 2: resident-LDS leaf histogram (construct_leaf_hist_resident_lds_kernel, P=1)");
    println!(
        "{:>12} | {:>5} | {:>11} | {:>11} | {:>8} | {:>17} | {:>17}",
        "regime", "bins", "checked_ms", "unchk_ms", "delta%", "checked p25/p75", "unchk p25/p75"
    );
    println!("{:-<13}+{:-<7}+{:-<13}+{:-<13}+{:-<10}+{:-<19}+{:-<19}", "", "", "", "", "", "", "");
    // (regime, num_data, feats, leaf rows, warmup, timed)
    let lds_cells: [(&str, usize, usize, usize, usize, usize); 2] = [
        ("launch-bound", 4_096, 4, 1_024, 15, 21),
        ("compute-bound", 200_000, 50, 200_000, 3, 7),
    ];
    for (regime, num_data, feats, leaf_n, warmup, timed) in lds_cells {
        for &num_bin in &BIN_SWEEP {
            let resident = gen_resident(feats, num_data, num_bin);
            let grad = gen_grad(num_data);
            let hess = gen_hess(num_data);
            let leaf_rows: Vec<u32> = (0..leaf_n as u32).collect();
            // Sentinel slot_off (feats + 1 entries, final = slot_len), same layout as production.
            let slot_len = feats * 2 * num_bin as usize;
            let mut slot_s: Vec<u32> = (0..feats).map(|f| (f * 2 * num_bin as usize) as u32).collect();
            slot_s.push(slot_len as u32);
            // Host-gather ord_g/ord_h to leaf length (the launcher does this internally).
            let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| grad[r as usize]).collect();
            let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hess[r as usize]).collect();

            // Upload the big resident bin buffer ONCE per cell (outside both timed arms).
            let resident_handle = client.create_from_slice(u32::as_bytes(&resident));

            // Shared inline launch driver; `unchecked` selects the launch attribute.
            let drive = |unchecked: bool| -> Vec<f64> {
                let h_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
                let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
                let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
                let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));
                let zeros = vec![0.0f32; slot_len];
                let h_out = client.create_from_slice(f32::as_bytes(&zeros));
                let rows = leaf_rows.len();
                // P=1 (the small/medium gate): CubeCount::Static(feats, 1, 1).
                // SAFETY: same host-proven in-range obligations as the shipped launcher
                // (resident_bins sized feats*num_data, leaf_rows ⊂ [0,num_data), slot
                // sentinel sized feats+1, out sized slot_len, every num_bin <= 256). The
                // ONLY difference between the arms is launch_unchecked vs launch.
                if unchecked {
                    use lgbm_compute::kernels::histogram::construct_leaf_hist_resident_lds_kernel;
                    unsafe {
                        construct_leaf_hist_resident_lds_kernel::launch_unchecked(
                            &client,
                            CubeCount::Static(feats as u32, 1, 1),
                            CubeDim::new_1d(256),
                            ArrayArg::from_raw_parts(resident_handle.clone(), feats * num_data),
                            ArrayArg::from_raw_parts(h_rows.clone(), rows),
                            ArrayArg::from_raw_parts(h_g.clone(), rows),
                            ArrayArg::from_raw_parts(h_h.clone(), rows),
                            ArrayArg::from_raw_parts(h_slot.clone(), feats + 1),
                            num_data,
                            ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                        );
                    }
                } else {
                    unsafe {
                        construct_leaf_hist_resident_lds_kernel_checked::launch(
                            &client,
                            CubeCount::Static(feats as u32, 1, 1),
                            CubeDim::new_1d(256),
                            ArrayArg::from_raw_parts(resident_handle.clone(), feats * num_data),
                            ArrayArg::from_raw_parts(h_rows.clone(), rows),
                            ArrayArg::from_raw_parts(h_g.clone(), rows),
                            ArrayArg::from_raw_parts(h_h.clone(), rows),
                            ArrayArg::from_raw_parts(h_slot.clone(), feats + 1),
                            num_data,
                            ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                        );
                    }
                }
                let bytes = client.read_one_unchecked(h_out);
                f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect()
            };

            for _ in 0..warmup {
                let _ = drive(false);
                let _ = drive(true);
            }
            let mut c_ms = Vec::with_capacity(timed);
            let mut u_ms = Vec::with_capacity(timed);
            let mut c_hist = Vec::new();
            let mut u_hist = Vec::new();
            for _ in 0..timed {
                let t = Instant::now();
                let ch = drive(false);
                c_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let uh = drive(true);
                u_ms.push(t.elapsed().as_secs_f64() * 1e3);
                c_hist = ch;
                u_hist = uh;
            }
            let cell = format!("lds {regime} bins={num_bin}");
            assert_same_input_f32(&c_hist, &u_hist, &cell);
            let c_med = median(&mut c_ms);
            let u_med = median(&mut u_ms);
            let (cl, ch_) = spread(&c_ms);
            let (ul, uh_) = spread(&u_ms);
            let delta = (c_med - u_med) / c_med * 100.0;
            println!(
                "{:>12} | {:>5} | {:>11.4} | {:>11.4} | {:>7.2} | {:>7.4}/{:>7.4} | {:>7.4}/{:>7.4}",
                regime, num_bin, c_med, u_med, delta, cl, ch_, ul, uh_
            );
        }
    }
    println!();

    // ===================================================================
    // KERNEL 3 — fused build+fix+scan (build_fix_scan_fused_kernel), f64/deterministic.
    // Minimal direct device-arg setup (NOT the full build_fix_scan_resident_f64_on
    // marshalling). One cube per feature, CubeDim::new_1d(1). The two arms MUST be
    // bit-identical (f64 sequential fold, no atomics).
    // ===================================================================
    println!("## kernel 3: fused build+fix+scan (build_fix_scan_fused_kernel, f64 deterministic)");
    println!(
        "{:>12} | {:>5} | {:>11} | {:>11} | {:>8} | {:>17} | {:>17}",
        "regime", "bins", "checked_ms", "unchk_ms", "delta%", "checked p25/p75", "unchk p25/p75"
    );
    println!("{:-<13}+{:-<7}+{:-<13}+{:-<13}+{:-<10}+{:-<19}+{:-<19}", "", "", "", "", "", "", "");
    // (regime, num_data, feats, leaf rows, warmup, timed)
    let fused_cells: [(&str, usize, usize, usize, usize, usize); 2] = [
        ("launch-bound", 4_096, 4, 1_024, 15, 21),
        ("compute-bound", 200_000, 50, 200_000, 3, 7),
    ];
    for (regime, num_data, feats, leaf_n, warmup, timed) in fused_cells {
        for &num_bin in &BIN_SWEEP {
            let resident = gen_resident(feats, num_data, num_bin);
            let grad = gen_grad(num_data);
            let hess = gen_hess(num_data);
            let leaf_rows: Vec<u32> = (0..leaf_n as u32).collect();
            let rows = leaf_rows.len();
            let nb_i = num_bin as i32;
            let slot_len = feats * 2 * num_bin as usize;
            // Per-feature arrays (length == feats). slot_off NON-sentinel (the fused kernel
            // indexes slot_off[fi] directly, no f+1 lookup). most_freq_bin=0 ⇒ FIX skipped;
            // offset=0 ⇒ COMPACT skipped; scan_active=1 ⇒ scanned.
            let slot_off_a: Vec<u32> = (0..feats).map(|f| (f * 2 * num_bin as usize) as u32).collect();
            let num_bin_a: Vec<i32> = vec![nb_i; feats];
            let offset_a: Vec<i32> = vec![0; feats];
            let mfb_a: Vec<i32> = vec![0; feats];
            let default_bin_a: Vec<i32> = vec![0; feats];
            let skip_a: Vec<u32> = vec![0; feats];
            let rev_a: Vec<i32> = vec![(nb_i - 1).max(0); feats];
            let fwd_a: Vec<i32> = vec![(nb_i - 1).max(0); feats];
            let scan_a: Vec<u32> = vec![1; feats];
            // Leaf scalars from the gathered grad/hess (plausible; the scan divides by sum_hessian).
            let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| grad[r as usize]).collect();
            let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hess[r as usize]).collect();
            let sum_gradient_raw: f64 = ord_g.iter().map(|&x| f64::from(x)).sum();
            let sum_hessian_raw: f64 = ord_h.iter().map(|&x| f64::from(x)).sum();
            let sum_hessian_bumped = sum_hessian_raw; // no 2*kEps bump needed for a timing-only run

            let resident_handle = client.create_from_slice(u32::as_bytes(&resident));

            let drive = |unchecked: bool| -> Vec<f64> {
                let h_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
                let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
                let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
                let zeros64 = vec![0.0f64; slot_len];
                let h_hist = client.create_from_slice(f64::as_bytes(&zeros64));
                let out_len = feats * 12;
                let out_zeros = vec![0.0f64; out_len];
                let h_out = client.create_from_slice(f64::as_bytes(&out_zeros));
                let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
                let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
                let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
                let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));
                let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
                let h_skip = client.create_from_slice(u32::as_bytes(&skip_a));
                let h_rev = client.create_from_slice(i32::as_bytes(&rev_a));
                let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_a));
                let h_scan = client.create_from_slice(u32::as_bytes(&scan_a));
                // SAFETY: cube f < feats reads its resident column + its validated slot
                // region only; every per-feature array sized feats; leaf_rows ⊂ [0,num_data).
                // The ONLY difference between the arms is launch_unchecked vs launch.
                if unchecked {
                    use lgbm_compute::kernels::histogram::build_fix_scan_fused_kernel;
                    unsafe {
                        build_fix_scan_fused_kernel::launch_unchecked(
                            &client,
                            CubeCount::Static(feats as u32, 1, 1),
                            CubeDim::new_1d(1),
                            ArrayArg::from_raw_parts(resident_handle.clone(), feats * num_data),
                            ArrayArg::from_raw_parts(h_rows.clone(), rows),
                            ArrayArg::from_raw_parts(h_g.clone(), rows),
                            ArrayArg::from_raw_parts(h_h.clone(), rows),
                            ArrayArg::from_raw_parts(h_hist.clone(), slot_len),
                            ArrayArg::from_raw_parts(h_out.clone(), out_len),
                            ArrayArg::from_raw_parts(h_slot.clone(), feats),
                            ArrayArg::from_raw_parts(h_numbin.clone(), feats),
                            ArrayArg::from_raw_parts(h_offset.clone(), feats),
                            ArrayArg::from_raw_parts(h_mfb.clone(), feats),
                            ArrayArg::from_raw_parts(h_defbin.clone(), feats),
                            ArrayArg::from_raw_parts(h_skip.clone(), feats),
                            ArrayArg::from_raw_parts(h_rev.clone(), feats),
                            ArrayArg::from_raw_parts(h_fwd.clone(), feats),
                            ArrayArg::from_raw_parts(h_scan.clone(), feats),
                            num_data,
                            sum_gradient_raw,
                            sum_hessian_raw,
                            0u32,
                            0i32,
                            0.0f64,
                            0.0f64,
                            0.0f64,
                            0.0f64,
                            sum_hessian_bumped,
                            num_data as i32,
                        );
                    }
                } else {
                    unsafe {
                        build_fix_scan_fused_kernel_checked::launch(
                            &client,
                            CubeCount::Static(feats as u32, 1, 1),
                            CubeDim::new_1d(1),
                            ArrayArg::from_raw_parts(resident_handle.clone(), feats * num_data),
                            ArrayArg::from_raw_parts(h_rows.clone(), rows),
                            ArrayArg::from_raw_parts(h_g.clone(), rows),
                            ArrayArg::from_raw_parts(h_h.clone(), rows),
                            ArrayArg::from_raw_parts(h_hist.clone(), slot_len),
                            ArrayArg::from_raw_parts(h_out.clone(), out_len),
                            ArrayArg::from_raw_parts(h_slot.clone(), feats),
                            ArrayArg::from_raw_parts(h_numbin.clone(), feats),
                            ArrayArg::from_raw_parts(h_offset.clone(), feats),
                            ArrayArg::from_raw_parts(h_mfb.clone(), feats),
                            ArrayArg::from_raw_parts(h_defbin.clone(), feats),
                            ArrayArg::from_raw_parts(h_skip.clone(), feats),
                            ArrayArg::from_raw_parts(h_rev.clone(), feats),
                            ArrayArg::from_raw_parts(h_fwd.clone(), feats),
                            ArrayArg::from_raw_parts(h_scan.clone(), feats),
                            num_data,
                            sum_gradient_raw,
                            sum_hessian_raw,
                            0u32,
                            0i32,
                            0.0f64,
                            0.0f64,
                            0.0f64,
                            0.0f64,
                            sum_hessian_bumped,
                            num_data as i32,
                        );
                    }
                }
                // Read back the SplitInfo out window (forces device sync).
                let bytes = client.read_one_unchecked(h_out);
                f64::from_bytes(&bytes).to_vec()
            };

            for _ in 0..warmup {
                let _ = drive(false);
                let _ = drive(true);
            }
            let mut c_ms = Vec::with_capacity(timed);
            let mut u_ms = Vec::with_capacity(timed);
            let mut c_out = Vec::new();
            let mut u_out = Vec::new();
            for _ in 0..timed {
                let t = Instant::now();
                let co = drive(false);
                c_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let uo = drive(true);
                u_ms.push(t.elapsed().as_secs_f64() * 1e3);
                c_out = co;
                u_out = uo;
            }
            let cell = format!("fused {regime} bins={num_bin}");
            assert_same_input_bits(&c_out, &u_out, &cell);
            let c_med = median(&mut c_ms);
            let u_med = median(&mut u_ms);
            let (cl, ch_) = spread(&c_ms);
            let (ul, uh_) = spread(&u_ms);
            let delta = (c_med - u_med) / c_med * 100.0;
            println!(
                "{:>12} | {:>5} | {:>11.4} | {:>11.4} | {:>7.2} | {:>7.4}/{:>7.4} | {:>7.4}/{:>7.4}",
                regime, num_bin, c_med, u_med, delta, cl, ch_, ul, uh_
            );
        }
    }

    println!();
    println!("# LEGEND: delta% > 0 => unchecked (launch_unchecked) is faster; < 0 => checked faster.");
    println!("# launch_unchecked ONLY removes in-kernel bounds-check codegen — numerics are UNCHANGED");
    println!("#   (the same-input asserts above passed: f32-envelope for atomic/LDS, bit-equal for fused).");
    println!("# A delta within the p25/p75 spread, or whose SIGN flips across process restarts, is");
    println!("#   SUB-NOISE / NULL — re-run >=2 times and compare. The launch-bound cells are the only");
    println!("#   place a per-launch codegen delta could plausibly surface (mwr: realistic regime is");
    println!("#   transfer-bound, which masks it). MEASUREMENT ONLY — production kernels untouched.");
}
