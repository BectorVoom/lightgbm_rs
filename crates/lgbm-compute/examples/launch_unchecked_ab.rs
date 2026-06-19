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
    // Filled in Task 2.
    eprintln!("launch_unchecked_ab: main body wired in Task 2.");
}
