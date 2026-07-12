//! Bit-exactness gates for the PARALLEL-CANDIDATE staged scan
//! (`LGBM_SCAN_PARGAIN`, `find_best_splits_fused_staged_par_kernel` and its
//! sibling twin).
//!
//! Two layers:
//! 1. ALGORITHM parity (default cpu lane, always runs): a plain-Rust
//!    transcription of the pargain phases (serial accumulate+store → 32-lane
//!    strided gain scan → lexicographic partial reduce → state assembly) must
//!    reproduce a VERBATIM plain-Rust transcription of the serial staged
//!    branch scans (`scan_rev_branch_staged`/`scan_fwd_branch_staged` +
//!    `merge_finalize_staged` — themselves pinned to the legacy kernel by the
//!    existing staged gates) BITWISE on all 12 output cells, across a fan-out
//!    of bin counts, offsets, skip/L1/min_data variants, `run_forward` off,
//!    early-`done` corpora, and palindromic EXACT-gain-tie corpora.
//! 2. KERNEL parity (REAL GPU only — `cuda`/`rocm`): the pargain kernel vs the
//!    legacy serial kernel, 12-cell bitwise, on device. The staged kernel
//!    family (SharedMemory + sync_cube) does NOT lower reliably on cubecl-cpu
//!    — the SAME reason the production launchers gate staged scans to real
//!    devices — so the kernel-level comparison runs in the CUDA spike session,
//!    not locally.
//!
//! Layer 1 proves the redesigned DECISION PROCEDURE (the load-bearing novelty:
//! stored-state gains + lexicographic first-max tie-break) is exactly the
//! serial one; layer 2 + the spike's driver-level bit-identical-predictions
//! A/B prove the cube transcription of that procedure.

#![cfg(feature = "gpu")]

use lgbm_compute::gain::{calculate_splitted_leaf_output, get_leaf_gain, get_split_gains};

const K_EPSILON: f64 = 1e-15;

/// Deterministic LCG (no rand dep).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32() % 2000) / 100.0 - 10.0
    }
}

/// Host mirror of the kernel `round_int` (`(int)(x + 0.5f)` — the f32 literal
/// widened to f64, then truncation toward zero; both Rust `as` and the device
/// cast truncate toward zero).
fn round_int(x: f64) -> i32 {
    (x + f64::from(0.5f32)) as i32
}

/// One feature's scan-input bundle (the launcher-computed per-feature values).
#[derive(Clone, Debug)]
struct Feat {
    hist: Vec<f64>, // 2*num_bin cells (g,h interleaved)
    num_bin: i32,
    offset: i32,
    default_bin: i32,
    skip_default_bin: bool,
    run_forward: bool,
}

impl Feat {
    fn rev_count(&self) -> i32 {
        (self.num_bin - 1).max(0)
    }
    fn fwd_count(&self) -> i32 {
        if self.run_forward {
            (self.num_bin - 1 - self.offset).max(0)
        } else {
            0
        }
    }
}

/// Leaf-level scan scalars shared by both transcriptions.
#[derive(Clone, Copy, Debug)]
struct Scalars {
    use_l1: bool,
    lambda_l1: f64,
    lambda_l2: f64,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64, // ALREADY bumped by 2*kEpsilon
    num_data: i32,
}

/// A branch's 6-cell state: [is_splittable, best_gain, threshold, left_count,
/// sum_left_gradient, sum_left_hessian].
type BranchState = [f64; 6];

// =========================================================================
// SERIAL reference — VERBATIM plain-Rust transcription of
// `scan_rev_branch_staged` / `scan_fwd_branch_staged` (`select` → `if`, the
// arithmetic and gate order unchanged).
// =========================================================================

fn serial_rev(f: &Feat, s: &Scalars) -> BranchState {
    let cnt_factor = f64::from(s.num_data) / s.sum_hessian;
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut best_gain = 0.0f64;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = 0.0f64;
    let mut sum_right_gradient = 0.0f64;
    let mut sum_right_hessian = K_EPSILON;
    let mut right_count = 0i32;
    let t_start = f.num_bin - 1 - f.offset;
    let mut done = false;
    for k in 0..f.rev_count() {
        let t = t_start - k;
        let in_range = t >= (1 - f.offset);
        let skip = f.skip_default_bin && (t + f.offset) == f.default_bin;
        let active = in_range && !skip && !done;
        let t_safe = if t < 0 { 0 } else { t };
        let bi = (t_safe as usize) * 2;
        if active {
            sum_right_gradient += f.hist[bi];
            sum_right_hessian += f.hist[bi + 1];
            right_count += round_int(f.hist[bi + 1] * cnt_factor);
        }
        let left_count = s.num_data - right_count;
        let sum_left_hessian = s.sum_hessian - sum_right_hessian;
        let sum_left_gradient = s.sum_gradient - sum_right_gradient;
        let cont = right_count < s.min_data_in_leaf
            || sum_right_hessian < s.min_sum_hessian_in_leaf;
        let brk =
            left_count < s.min_data_in_leaf || sum_left_hessian < s.min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;
        let current_gain = get_split_gains(
            s.use_l1,
            sum_left_gradient,
            sum_left_hessian,
            sum_right_gradient,
            sum_right_hessian,
            s.lambda_l1,
            s.lambda_l2,
        );
        let valid = consider && current_gain > s.min_gain_shift;
        if valid {
            is_splittable = 1.0;
        }
        let cand_gain = if valid { current_gain } else { 0.0 };
        if cand_gain > best_gain {
            best_left_count = left_count;
            best_sum_left_gradient = sum_left_gradient;
            best_sum_left_hessian = sum_left_hessian;
            best_threshold = t - 1 + f.offset;
            best_gain = cand_gain;
        }
    }
    [
        is_splittable,
        best_gain,
        f64::from(best_threshold),
        f64::from(best_left_count),
        best_sum_left_gradient,
        best_sum_left_hessian,
    ]
}

fn serial_fwd(f: &Feat, s: &Scalars) -> BranchState {
    let cnt_factor = f64::from(s.num_data) / s.sum_hessian;
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut best_gain = 0.0f64;
    let mut best_left_count = 0i32;
    let mut best_threshold = 0i32;
    let mut is_splittable = 0.0f64;
    let mut sum_left_gradient = 0.0f64;
    let mut sum_left_hessian = K_EPSILON;
    let mut left_count = 0i32;
    let mut done = false;
    for t in 0..f.fwd_count() {
        let skip = f.skip_default_bin && (t + f.offset) == f.default_bin;
        let active = !skip && !done;
        let bi = (t as usize) * 2;
        if active {
            sum_left_gradient += f.hist[bi];
            sum_left_hessian += f.hist[bi + 1];
            left_count += round_int(f.hist[bi + 1] * cnt_factor);
        }
        let right_count = s.num_data - left_count;
        let sum_right_hessian = s.sum_hessian - sum_left_hessian;
        let sum_right_gradient = s.sum_gradient - sum_left_gradient;
        let cont =
            left_count < s.min_data_in_leaf || sum_left_hessian < s.min_sum_hessian_in_leaf;
        let brk = right_count < s.min_data_in_leaf
            || sum_right_hessian < s.min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;
        let current_gain = get_split_gains(
            s.use_l1,
            sum_left_gradient,
            sum_left_hessian,
            sum_right_gradient,
            sum_right_hessian,
            s.lambda_l1,
            s.lambda_l2,
        );
        let valid = consider && current_gain > s.min_gain_shift;
        if valid {
            is_splittable = 1.0;
        }
        let cand_gain = if valid { current_gain } else { 0.0 };
        if cand_gain > best_gain {
            best_left_count = left_count;
            best_sum_left_gradient = sum_left_gradient;
            best_sum_left_hessian = sum_left_hessian;
            best_threshold = t + f.offset;
            best_gain = cand_gain;
        }
    }
    [
        is_splittable,
        best_gain,
        f64::from(best_threshold),
        f64::from(best_left_count),
        best_sum_left_gradient,
        best_sum_left_hessian,
    ]
}

// =========================================================================
// PARGAIN transcription — the kernel's phase 1/2/3, faithfully including the
// 32-lane strided partition and the per-lane → cross-lane reduction order.
// =========================================================================

struct Stored {
    ag: Vec<f64>, // branch-accumulated pair
    ah: Vec<f64>,
    lc: Vec<f64>, // left_count (exact small-int f64, as the kernel stores it)
    ok: Vec<f64>, // consider flag (1.0/0.0)
}

fn pargain_store_rev(f: &Feat, s: &Scalars) -> Stored {
    let count = f.rev_count().max(0) as usize;
    let mut st = Stored {
        ag: vec![0.0; count],
        ah: vec![0.0; count],
        lc: vec![0.0; count],
        ok: vec![0.0; count],
    };
    let cnt_factor = f64::from(s.num_data) / s.sum_hessian;
    let mut sum_right_gradient = 0.0f64;
    let mut sum_right_hessian = K_EPSILON;
    let mut right_count = 0i32;
    let t_start = f.num_bin - 1 - f.offset;
    let mut done = false;
    for k in 0..f.rev_count() {
        let t = t_start - k;
        let in_range = t >= (1 - f.offset);
        let skip = f.skip_default_bin && (t + f.offset) == f.default_bin;
        let active = in_range && !skip && !done;
        let t_safe = if t < 0 { 0 } else { t };
        let bi = (t_safe as usize) * 2;
        if active {
            sum_right_gradient += f.hist[bi];
            sum_right_hessian += f.hist[bi + 1];
            right_count += round_int(f.hist[bi + 1] * cnt_factor);
        }
        let left_count = s.num_data - right_count;
        let sum_left_hessian = s.sum_hessian - sum_right_hessian;
        let cont = right_count < s.min_data_in_leaf
            || sum_right_hessian < s.min_sum_hessian_in_leaf;
        let brk =
            left_count < s.min_data_in_leaf || sum_left_hessian < s.min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;
        let ku = k as usize;
        st.ag[ku] = sum_right_gradient;
        st.ah[ku] = sum_right_hessian;
        st.lc[ku] = f64::from(left_count);
        st.ok[ku] = if consider { 1.0 } else { 0.0 };
    }
    st
}

fn pargain_store_fwd(f: &Feat, s: &Scalars) -> Stored {
    let count = f.fwd_count().max(0) as usize;
    let mut st = Stored {
        ag: vec![0.0; count],
        ah: vec![0.0; count],
        lc: vec![0.0; count],
        ok: vec![0.0; count],
    };
    let cnt_factor = f64::from(s.num_data) / s.sum_hessian;
    let mut sum_left_gradient = 0.0f64;
    let mut sum_left_hessian = K_EPSILON;
    let mut left_count = 0i32;
    let mut done = false;
    for t in 0..f.fwd_count() {
        let skip = f.skip_default_bin && (t + f.offset) == f.default_bin;
        let active = !skip && !done;
        let bi = (t as usize) * 2;
        if active {
            sum_left_gradient += f.hist[bi];
            sum_left_hessian += f.hist[bi + 1];
            left_count += round_int(f.hist[bi + 1] * cnt_factor);
        }
        let right_count = s.num_data - left_count;
        let sum_right_hessian = s.sum_hessian - sum_left_hessian;
        let cont =
            left_count < s.min_data_in_leaf || sum_left_hessian < s.min_sum_hessian_in_leaf;
        let brk = right_count < s.min_data_in_leaf
            || sum_right_hessian < s.min_sum_hessian_in_leaf;
        done = done || (active && !cont && brk);
        let consider = active && !cont && !done;
        let tu = t as usize;
        st.ag[tu] = sum_left_gradient;
        st.ah[tu] = sum_left_hessian;
        st.lc[tu] = f64::from(left_count);
        st.ok[tu] = if consider { 1.0 } else { 0.0 };
    }
    st
}

/// Phases 2+3: the 32-lane strided gain scan + lexicographic reduction +
/// state assembly, exactly as the kernel partitions and orders them.
fn pargain_scan_assemble(
    st: &Stored,
    count: i32,
    acc_is_left: bool,
    thr_base: i32,
    thr_step: i32,
    s: &Scalars,
) -> BranchState {
    const SENTINEL_K: f64 = 2147483647.0;
    // Phase 2: per-lane partials (lane l walks k = l, l+32, l+64, …).
    let mut part_gain = [0.0f64; 32];
    let mut part_k = [SENTINEL_K; 32];
    let mut part_any = [0.0f64; 32];
    for lane in 0..32usize {
        let mut best_gain = 0.0f64;
        let mut best_k = SENTINEL_K;
        let mut any_valid = 0.0f64;
        let mut k = lane as i32;
        while k < count {
            let ku = k as usize;
            if st.ok[ku] != 0.0 {
                let acc_g = st.ag[ku];
                let acc_h = st.ah[ku];
                let oth_g = s.sum_gradient - acc_g;
                let oth_h = s.sum_hessian - acc_h;
                let (left_g, left_h, right_g, right_h) = if acc_is_left {
                    (acc_g, acc_h, oth_g, oth_h)
                } else {
                    (oth_g, oth_h, acc_g, acc_h)
                };
                let current_gain = get_split_gains(
                    s.use_l1, left_g, left_h, right_g, right_h, s.lambda_l1, s.lambda_l2,
                );
                let valid = current_gain > s.min_gain_shift;
                if valid {
                    any_valid = 1.0;
                }
                let cand_gain = if valid { current_gain } else { 0.0 };
                let kf = f64::from(k);
                let take = cand_gain > best_gain
                    || (cand_gain == best_gain && cand_gain > 0.0 && kf < best_k);
                if take {
                    best_gain = cand_gain;
                    best_k = kf;
                }
            }
            k += 32;
        }
        part_gain[lane] = best_gain;
        part_k[lane] = best_k;
        part_any[lane] = any_valid;
    }
    // Phase 3: serial reduce of the 32 partials + assembly.
    let mut best_gain = 0.0f64;
    let mut best_k = SENTINEL_K;
    let mut any = 0.0f64;
    for l in 0..32usize {
        let g = part_gain[l];
        let k = part_k[l];
        if part_any[l] != 0.0 {
            any = 1.0;
        }
        let take = g > best_gain || (g == best_gain && g > 0.0 && k < best_k);
        if take {
            best_gain = g;
            best_k = k;
        }
    }
    let won = best_gain > 0.0;
    let k_safe = if won { best_k } else { 0.0 };
    let ku = k_safe as u32 as usize;
    // count == 0 ⇒ no stored candidates at all ⇒ the (selected-away) loads read
    // nothing real; mirror the kernel's in-bounds clamp with a zero default.
    let (acc_g, acc_h, lc_f) = if st.ag.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        (st.ag[ku], st.ah[ku], st.lc[ku])
    };
    let oth_g = s.sum_gradient - acc_g;
    let oth_h = s.sum_hessian - acc_h;
    let (slg, slh) = if acc_is_left { (acc_g, acc_h) } else { (oth_g, oth_h) };
    let thr = thr_base + thr_step * (k_safe as u32 as i32);
    [
        if any != 0.0 { 1.0 } else { 0.0 },
        if won { best_gain } else { 0.0 },
        if won { f64::from(thr) } else { 0.0 },
        if won { lc_f } else { 0.0 },
        if won { slg } else { 0.0 },
        if won { slh } else { 0.0 },
    ]
}

/// VERBATIM plain-Rust `merge_finalize_staged` (12 cells).
fn merge_finalize(rev: &BranchState, fwd: &BranchState, s: &Scalars) -> [f64; 12] {
    let take_fwd = fwd[1] > rev[1];
    let any_split = rev[0] != 0.0 || fwd[0] != 0.0;
    let pick = |i: usize| if take_fwd { fwd[i] } else { rev[i] };
    let best_gain = pick(1);
    let best_threshold_f = pick(2);
    let best_left_count_f = pick(3);
    let best_sum_left_gradient = pick(4);
    let best_sum_left_hessian = pick(5);
    let best_default_left = if take_fwd { 0.0 } else { 1.0 };
    let best_left_count = best_left_count_f as i32;
    let left_output = calculate_splitted_leaf_output(
        s.use_l1,
        best_sum_left_gradient,
        best_sum_left_hessian,
        s.lambda_l1,
        s.lambda_l2,
    );
    let right_sum_gradient = s.sum_gradient - best_sum_left_gradient;
    let right_sum_hessian = s.sum_hessian - best_sum_left_hessian;
    let right_output = calculate_splitted_leaf_output(
        s.use_l1,
        right_sum_gradient,
        right_sum_hessian,
        s.lambda_l1,
        s.lambda_l2,
    );
    [
        if any_split { 1.0 } else { 0.0 },
        best_threshold_f,
        best_gain,
        best_left_count_f,
        f64::from(s.num_data - best_left_count),
        best_sum_left_gradient,
        best_sum_left_hessian - K_EPSILON,
        right_sum_gradient,
        right_sum_hessian - K_EPSILON,
        best_default_left,
        left_output,
        right_output,
    ]
}

fn assert_algo_parity(label: &str, f: &Feat, s: &Scalars) {
    let serial = merge_finalize(&serial_rev(f, s), &serial_fwd(f, s), s);
    let rev = pargain_scan_assemble(
        &pargain_store_rev(f, s),
        f.rev_count(),
        false,
        f.num_bin - 2,
        -1,
        s,
    );
    let fwd =
        pargain_scan_assemble(&pargain_store_fwd(f, s), f.fwd_count(), true, f.offset, 1, s);
    let par = merge_finalize(&rev, &fwd, s);
    for i in 0..12 {
        assert_eq!(
            serial[i].to_bits(),
            par[i].to_bits(),
            "[{label}] cell {i}: serial {} != pargain {}",
            serial[i],
            par[i]
        );
    }
}

fn random_feat(lcg: &mut Lcg, num_bin: i32, offset: i32, skip_def: bool, run_forward: bool) -> Feat {
    let mut hist = Vec::with_capacity((num_bin as usize) * 2);
    for _ in 0..num_bin {
        hist.push(lcg.next_f64()); // g
        hist.push(f64::from(lcg.next_u32() % 50) / 10.0 + 0.1); // h > 0
    }
    Feat {
        hist,
        num_bin,
        offset,
        default_bin: num_bin.min(3),
        skip_default_bin: skip_def,
        run_forward,
    }
}

fn scalars_for(f: &Feat, use_l1: bool, min_data: i32, min_gain_to_split: f64) -> Scalars {
    let mut g = 0.0;
    let mut h = 0.0;
    for b in 0..f.num_bin as usize {
        g += f.hist[2 * b];
        h += f.hist[2 * b + 1];
    }
    let sum_hessian = h + 2.0 * K_EPSILON;
    let (l1, l2) = if use_l1 { (0.5, 1.0) } else { (0.0, 1.0) };
    let gain_shift = get_leaf_gain(use_l1, g, sum_hessian, l1, l2);
    Scalars {
        use_l1,
        lambda_l1: l1,
        lambda_l2: l2,
        min_data_in_leaf: min_data,
        min_sum_hessian_in_leaf: 1e-3,
        min_gain_shift: gain_shift + min_gain_to_split,
        sum_gradient: g,
        sum_hessian,
        num_data: (h.round() as i32).max(2) * 100,
    }
}

#[test]
fn pargain_algorithm_matches_serial_fan_out() {
    let mut lcg = Lcg(0x5ca1ab1e);
    let cases: Vec<(i32, i32, bool, bool, bool, i32, f64)> = vec![
        (2, 0, false, false, false, 1, 0.0),
        (2, 1, false, true, false, 1, 0.0),
        (8, 0, false, true, false, 1, 0.0),
        (8, 1, true, true, false, 1, 0.0),
        (64, 0, false, true, false, 20, 0.0),
        (64, 1, true, true, true, 20, 0.0),
        (255, 0, false, true, false, 1, 0.0),
        (255, 1, true, true, false, 50, 0.0),
        (255, 1, false, true, true, 1, 1.5),
        (16, 1, false, true, false, 1, -100.0),
    ];
    for (ci, &(nb, off, skip, fwd_on, l1, min_data, mgs)) in cases.iter().enumerate() {
        // 8 corpora per shape for coverage of done/cont paths + gain landscapes.
        for rep in 0..8 {
            let f = random_feat(&mut lcg, nb, off, skip, fwd_on);
            let s = scalars_for(&f, l1, min_data, mgs);
            assert_algo_parity(
                &format!("case {ci}.{rep}: nb={nb} off={off} skip={skip} fwd={fwd_on} l1={l1}"),
                &f,
                &s,
            );
        }
    }
}

#[test]
fn pargain_algorithm_matches_serial_early_done() {
    // min_data_in_leaf near half the rows: the brk/done freeze fires mid-scan
    // in both branches; every post-freeze candidate must stay dead.
    let mut lcg = Lcg(0xdead);
    for rep in 0..16 {
        let f = random_feat(&mut lcg, 32, 1, false, true);
        let mut s = scalars_for(&f, false, 1, 0.0);
        s.min_data_in_leaf = (f64::from(s.num_data) * 0.42) as i32;
        assert_algo_parity(&format!("early-done rep {rep}"), &f, &s);
    }
}

#[test]
fn pargain_algorithm_matches_serial_on_exact_gain_ties() {
    // EMPTY-BIN plateau — the production-realistic EXACT tie: a zero bin adds
    // 0.0 to every accumulator (bit-neutral), so consecutive candidates
    // straddling it carry BIT-IDENTICAL state and hence bitwise-equal gains.
    // The serial scan takes the EARLIEST (strict `>` never replaces); the
    // pargain lexicographic first-max must pick the same candidate. Layout
    // (num_bin=8, offset=1): b1={-4,1} … zeros … b4={4,1} — the optimal split
    // sits on a 3-candidate plateau in EACH branch. Verified non-vacuous.
    let num_bin = 8usize;
    let mut hist = vec![0.0f64; num_bin * 2];
    hist[2] = -4.0; // b1.g
    hist[3] = 1.0; // b1.h
    hist[8] = 4.0; // b4.g
    hist[9] = 1.0; // b4.h
    let f = Feat {
        hist,
        num_bin: num_bin as i32,
        offset: 1,
        default_bin: 0,
        skip_default_bin: false,
        run_forward: true,
    };
    let s = scalars_for(&f, false, 1, 0.0);
    // Non-vacuity: the FWD branch's winning gain is shared by >= 2 candidates.
    {
        let st = pargain_store_fwd(&f, &s);
        let mut gains: Vec<f64> = Vec::new();
        for ku in 0..st.ag.len() {
            if st.ok[ku] != 0.0 {
                let oth_g = s.sum_gradient - st.ag[ku];
                let oth_h = s.sum_hessian - st.ah[ku];
                gains.push(get_split_gains(
                    false, st.ag[ku], st.ah[ku], oth_g, oth_h, 0.0, 1.0,
                ));
            }
        }
        let mx = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ties = gains.iter().filter(|&&g| g == mx).count();
        assert!(ties >= 2, "tie corpus must actually tie (got {ties} maxima of {mx})");
    }
    assert_algo_parity("empty-bin plateau tie", &f, &s);

    // A denser variant: several plateaus + a skip_default_bin hole.
    let mut lcg = Lcg(0x71e);
    for rep in 0..8 {
        let nb = 32usize;
        let mut hist = vec![0.0f64; nb * 2];
        // Populate every 4th bin only — every candidate trio in between ties.
        for b in (1..nb).step_by(4) {
            hist[2 * b] = f64::from(lcg.next_u32() % 9) - 4.0;
            hist[2 * b + 1] = 1.0;
        }
        let f = Feat {
            hist,
            num_bin: nb as i32,
            offset: 1,
            default_bin: 5,
            skip_default_bin: rep % 2 == 0,
            run_forward: true,
        };
        let s = scalars_for(&f, false, 1, 0.0);
        assert_algo_parity(&format!("sparse plateau rep {rep}"), &f, &s);
    }
}

// =========================================================================
// Layer 2 — kernel-vs-kernel on a REAL GPU (cuda/rocm): the pargain kernel
// against the legacy serial kernel, 12-cell bitwise. Runs in the CUDA spike.
// =========================================================================
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod real_gpu_gated {
    use super::{random_feat, Feat, Lcg, K_EPSILON};
    use cubecl::prelude::*;
    use lgbm_compute::gain::get_leaf_gain;
    use lgbm_compute::kernels::split::{
        find_best_splits_fused_kernel, find_best_splits_fused_staged_par_kernel,
    };

    #[cfg(feature = "cuda")]
    type GpuRt = lgbm_compute::runtime::CudaRuntime;
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    type GpuRt = lgbm_compute::runtime::RocmRuntime;

    #[cfg(feature = "cuda")]
    fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRt> {
        lgbm_compute::runtime::cuda_client()
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRt> {
        lgbm_compute::runtime::rocm_client()
    }

    #[test]
    fn pargain_kernel_matches_legacy_kernel_on_device() {
        let client = gpu_client();
        let mut lcg = Lcg(0x0dd);
        let cases: Vec<(i32, i32, bool, bool, bool, i32, f64)> = vec![
            (2, 1, false, true, false, 1, 0.0),
            (8, 1, true, true, false, 1, 0.0),
            (64, 0, false, true, false, 20, 0.0),
            (255, 1, true, true, true, 50, 0.0),
            (255, 0, false, true, false, 1, 0.0),
            (32, 1, false, true, false, 400, 0.0),
            (100, 1, true, false, false, 5, 0.0),
        ];
        for (ci, &(nb, off, skip, fwd_on, l1, min_data, mgs)) in cases.iter().enumerate() {
            let feats: Vec<Feat> =
                (0..5).map(|_| random_feat(&mut lcg, nb, off, skip, fwd_on)).collect();
            let n = feats.len();
            let (l1v, l2v) = if l1 { (0.5, 1.0) } else { (0.0, 1.0) };
            let (mut sum_g, mut sum_h) = (0.0f64, 0.0f64);
            for b in 0..feats[0].num_bin as usize {
                sum_g += feats[0].hist[2 * b];
                sum_h += feats[0].hist[2 * b + 1];
            }
            let sum_h_b = sum_h + 2.0 * K_EPSILON;
            let min_gain_shift = get_leaf_gain(l1, sum_g, sum_h_b, l1v, l2v) + mgs;
            let num_data = (sum_h.round() as i32).max(2) * 100;

            let mut hist: Vec<f64> = Vec::new();
            let (mut slot, mut nbn, mut offs, mut dbn, mut skp, mut rv, mut fw): (
                Vec<u32>,
                Vec<i32>,
                Vec<i32>,
                Vec<i32>,
                Vec<u32>,
                Vec<i32>,
                Vec<i32>,
            ) = Default::default();
            for f in &feats {
                slot.push(hist.len() as u32);
                nbn.push(f.num_bin);
                offs.push(f.offset);
                dbn.push(f.default_bin);
                skp.push(u32::from(f.skip_default_bin));
                rv.push(f.rev_count());
                fw.push(f.fwd_count());
                hist.extend_from_slice(&f.hist);
            }
            let buf_len = hist.len();
            let out_len = n * 12;
            let h_hist = client.create_from_slice(f64::as_bytes(&hist));
            let h_slot = client.create_from_slice(u32::as_bytes(&slot));
            let h_nbn = client.create_from_slice(i32::as_bytes(&nbn));
            let h_off = client.create_from_slice(i32::as_bytes(&offs));
            let h_dbn = client.create_from_slice(i32::as_bytes(&dbn));
            let h_skp = client.create_from_slice(u32::as_bytes(&skp));
            let h_rv = client.create_from_slice(i32::as_bytes(&rv));
            let h_fw = client.create_from_slice(i32::as_bytes(&fw));
            let zeros = vec![0.0f64; out_len];
            let out_a = client.create_from_slice(f64::as_bytes(&zeros));
            let out_b = client.create_from_slice(f64::as_bytes(&zeros));

            // SAFETY: contiguous region tiling; out sized n*12; arrays sized n;
            // legacy guards lane < n_feats; pargain geometry = exactly n cubes.
            unsafe {
                find_best_splits_fused_kernel::launch(
                    &client,
                    CubeCount::Static((n as u32).div_ceil(32), 1, 1),
                    CubeDim::new_1d(32),
                    ArrayArg::from_raw_parts(h_hist.clone(), buf_len),
                    ArrayArg::from_raw_parts(out_a.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot.clone(), n),
                    ArrayArg::from_raw_parts(h_nbn.clone(), n),
                    ArrayArg::from_raw_parts(h_off.clone(), n),
                    ArrayArg::from_raw_parts(h_dbn.clone(), n),
                    ArrayArg::from_raw_parts(h_skp.clone(), n),
                    ArrayArg::from_raw_parts(h_rv.clone(), n),
                    ArrayArg::from_raw_parts(h_fw.clone(), n),
                    u32::from(l1),
                    min_data,
                    1e-3f64,
                    l1v,
                    l2v,
                    min_gain_shift,
                    sum_g,
                    sum_h_b,
                    num_data,
                    n as u32,
                );
                find_best_splits_fused_staged_par_kernel::launch(
                    &client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(h_hist.clone(), buf_len),
                    ArrayArg::from_raw_parts(out_b.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot.clone(), n),
                    ArrayArg::from_raw_parts(h_nbn.clone(), n),
                    ArrayArg::from_raw_parts(h_off.clone(), n),
                    ArrayArg::from_raw_parts(h_dbn.clone(), n),
                    ArrayArg::from_raw_parts(h_skp.clone(), n),
                    ArrayArg::from_raw_parts(h_rv.clone(), n),
                    ArrayArg::from_raw_parts(h_fw.clone(), n),
                    u32::from(l1),
                    min_data,
                    1e-3f64,
                    l1v,
                    l2v,
                    min_gain_shift,
                    sum_g,
                    sum_h_b,
                    num_data,
                );
            }
            let a = f64::from_bytes(&client.read_one_unchecked(out_a)).to_vec();
            let b = f64::from_bytes(&client.read_one_unchecked(out_b)).to_vec();
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "case {ci} out[{i}]: legacy {x} != pargain {y}"
                );
            }
        }
    }
}
