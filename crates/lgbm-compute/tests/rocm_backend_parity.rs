//! GPU (RocmBackend) vs CPU (CpuBackend) bit-exact parity for all four hot-path
//! ops, on the real gfx1100. Feature-gated on `rocm` — runs only under
//! `cargo test -p lgbm-compute --features rocm`, never in a CPU-only build (SC#1).
//!
//! Proves the switchable GPU dispatch (260608-kfu) is numerically faithful: the f64
//! kernels run on cubecl-hip bit-exactly to the native-f64 CPU anchor.
#![cfg(feature = "rocm")]

use cubecl::prelude::*;

use lgbm_compute::gain::GainConfig;
use lgbm_compute::kernels::best_split::{
    find_best_from_all_splits_on, find_best_splits_stage1_f32_on, find_best_splits_stage1_on,
    sync_best_split_for_leaf_on, SplitFindTask, Stage1Scalars,
};
use lgbm_compute::kernels::split_info::SplitScalars;
use lgbm_compute::runtime::{cpu_client, rocm_client};
use lgbm_compute::{Backend, CpuBackend, RocmBackend};

/// The ~1e-6 f32 gain tolerance for the tie-aware `default_left` comparator (SC#3,
/// 17-RESEARCH §"Tie-Aware default_left"; the ORACLE_TOL of the histogram parity harness).
const ORACLE_TOL: f64 = 1e-6;

fn assert_bit_exact(cpu: &[f64], gpu: &[f64], what: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{what}: length mismatch");
    for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: cell {i} diverged — cpu {a} ({:#x}) vs gpu {b} ({:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn rocm_backend_construct_histograms_bit_exact() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();

    let binned = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3];
    let grad: Vec<f32> = binned.iter().map(|&b| (b as f32) * 1.5 - 2.0).collect();
    let hess = vec![1.0f32; binned.len()];
    let num_bin = 4u32;

    let h_cpu = cpu
        .construct_histograms(&cc, &binned, &grad, &hess, num_bin)
        .unwrap();
    let h_gpu = gpu
        .construct_histograms(&gc, &binned, &grad, &hess, num_bin)
        .unwrap();
    assert_bit_exact(&h_cpu, &h_gpu, "construct_histograms");
}

#[test]
fn rocm_backend_find_best_split_bit_exact() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();
    let cfg = GainConfig::default();

    // Build a histogram on the CPU and find the best split on both backends.
    let binned = vec![0u32, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3];
    let grad: Vec<f32> = vec![-3.0, -2.5, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, -2.0, 0.0, 1.5, 2.5];
    let hess = vec![1.0f32; binned.len()];
    let num_bin = 4u32;
    let hist = cpu
        .construct_histograms(&cc, &binned, &grad, &hess, num_bin)
        .unwrap();
    let sum_g: f64 = grad.iter().map(|&g| g as f64).sum();
    let sum_h: f64 = hess.iter().map(|&h| h as f64).sum();
    let num_data = binned.len() as i32;

    let s_cpu = cpu
        .find_best_split(
            &cc, &hist, &cfg, num_bin, 0, 0, 0, false, false, false, sum_g, sum_h, num_data,
        )
        .unwrap();
    let s_gpu = gpu
        .find_best_split(
            &gc, &hist, &cfg, num_bin, 0, 0, 0, false, false, false, sum_g, sum_h, num_data,
        )
        .unwrap();

    // Compare the SplitInfo field-by-field as bits.
    let to_cells = |s: &lgbm_compute::SplitInfo| -> Vec<f64> {
        vec![
            s.threshold as f64,
            s.gain,
            s.left_count as f64,
            s.right_count as f64,
            s.left_sum_gradient,
            s.left_sum_hessian,
            s.right_sum_gradient,
            s.right_sum_hessian,
            s.left_output,
            s.right_output,
            if s.default_left { 1.0 } else { 0.0 },
        ]
    };
    assert_bit_exact(&to_cells(&s_cpu), &to_cells(&s_gpu), "find_best_split");
}

#[test]
fn rocm_backend_subtract_histograms_bit_exact() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();

    let parent: Vec<f64> = (0..16).map(|i| (i as f64) * 0.25 - 1.0).collect();
    let child: Vec<f64> = (0..16).map(|i| (i as f64) * 0.1).collect();

    let d_cpu = cpu.subtract_histograms(&cc, &parent, &child).unwrap();
    let d_gpu = gpu.subtract_histograms(&gc, &parent, &child).unwrap();
    assert_bit_exact(&d_cpu, &d_gpu, "subtract_histograms");
}

// ===========================================================================
// SC#3 — tie-aware default_left parity on hip (17-05 Task 3).
//
// Drives the FULL stage-1 → stage-2 → stage-3 best-split pipeline on `CpuBackend`
// (the f64 fold, the anchor) and `RocmBackend` (the f32 mirror), NEVER GPU-vs-GPU
// (def-f8u-01): every hip f32 result is compared to the cpu f64 fold. The comparator
// accepts a `default_left` flip ONLY on a verified f32 tie (same threshold + same
// left_count + f32-equal gains within ~1e-6); a flip on ANY non-tie split HARD-FAILS;
// empty / no-valid-split fixtures pass with `is_valid=false` and no spurious flip.
//
// Manual-only per 17-VALIDATION (the ROCm host): the cpu-only gate is the compile
// (`cargo build -p lgbm-compute --tests --features rocm`); the ~1e-6 assertion runs
// under `cargo test -p lgbm-compute --features rocm default_left_tie` on the APU.
// ===========================================================================

/// A default continuous [`SplitFindTask`] (no categorical / na / skip); `reverse` +
/// `assume_out_default_left` drive the two-task fwd/rev pair whose stage-2 winner may
/// flip `default_left` on an f32 tie.
fn tie_task(feat: i32, num_bin: u32, reverse: bool, assume_left: bool) -> SplitFindTask {
    SplitFindTask {
        inner_feature_index: feat,
        reverse,
        skip_default_bin: false,
        na_as_missing: false,
        assume_out_default_left: assume_left,
        is_categorical: false,
        is_one_hot: false,
        hist_offset: 0,
        mfb_offset: 0,
        num_bin,
        default_bin: num_bin, // out of range → no default-bin skip
        rand_threshold: -1,
    }
}

/// Default-template [`Stage1Scalars`] (no L1 / smoothing / rand) with the given leaf
/// totals and `min_data_in_leaf` guard.
fn tie_scalars(sum_g: f64, sum_h: f64, num_data: i32, min_data: i32) -> Stage1Scalars {
    Stage1Scalars {
        use_l1: false,
        use_smoothing: false,
        use_rand: false,
        is_larger: false,
        min_data_in_leaf: min_data,
        min_sum_hessian_in_leaf: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        parent_output: 0.0,
        sum_gradient: sum_g,
        sum_hessian: sum_h,
        num_data,
        parent_gain: 0.0,
        rng_seed: 0,
    }
}

/// Run the stage-1→2 pipeline for ONE leaf on the f64 fold (the anchor): every task's
/// stage-1 record, then the stage-2 cross-feature reduce (smaller path). The larger
/// half of the slab mirrors the smaller (single-leaf test).
fn leaf_winner_f64<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    tasks: &[SplitFindTask],
    hist: &[f64],
    sc: &Stage1Scalars,
) -> SplitScalars {
    let per_task: Vec<SplitScalars> = tasks
        .iter()
        .map(|t| find_best_splits_stage1_on(client, hist, t, sc).expect("stage1 f64"))
        .collect();
    let n = per_task.len();
    let mut slab = per_task.clone();
    slab.extend(per_task);
    sync_best_split_for_leaf_on(client, &slab, n, true).expect("stage2 f64")
}

/// Run the stage-1→2 pipeline for ONE leaf on the f32 mirror (anchored to the f64 fold,
/// never GPU-vs-GPU): the single-owner f32 stage-1 fold + the stage-2 reduce.
fn leaf_winner_f32<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    tasks: &[SplitFindTask],
    hist: &[f32],
    sc: &Stage1Scalars,
) -> SplitScalars {
    let per_task: Vec<SplitScalars> = tasks
        .iter()
        .map(|t| find_best_splits_stage1_f32_on(client, hist, t, sc).expect("stage1 f32"))
        .collect();
    let n = per_task.len();
    let mut slab = per_task.clone();
    slab.extend(per_task);
    sync_best_split_for_leaf_on(client, &slab, n, true).expect("stage2 f32")
}

/// The tie-aware comparator (SC#3): `default_left` may flip ONLY on a verified f32 tie
/// (same threshold + same left_count + gains within `ORACLE_TOL`); on a non-flip the
/// ordinary parity holds (threshold + left_count exact, gains within tol); a flip on any
/// non-tie split HARD-FAILS. `is_valid=false` on both → pass (empty / sparse).
fn assert_split_tie_aware(cpu: &SplitScalars, gpu: &SplitScalars, what: &str) {
    assert_eq!(cpu.is_valid, gpu.is_valid, "{what}: is_valid mismatch (cpu vs gpu)");
    if !cpu.is_valid {
        return; // empty / no-valid-split: both invalid, no spurious flip.
    }
    let same_threshold = cpu.threshold == gpu.threshold;
    let same_left_count = cpu.left_count == gpu.left_count;
    let gains_close = (cpu.gain - gpu.gain).abs() <= ORACLE_TOL;
    if cpu.default_left == gpu.default_left {
        // No flip → ordinary parity: threshold + left_count exact, gains within ~1e-6.
        assert!(
            same_threshold,
            "{what}: threshold {} vs {} (no default_left flip)",
            cpu.threshold, gpu.threshold
        );
        assert!(
            same_left_count,
            "{what}: left_count {} vs {} (no default_left flip)",
            cpu.left_count, gpu.left_count
        );
        assert!(
            gains_close,
            "{what}: gain {} vs {} exceeds {ORACLE_TOL}",
            cpu.gain, gpu.gain
        );
    } else {
        // A flip is accepted ONLY on a verified f32 tie; otherwise HARD-FAIL.
        let verified_tie = same_threshold && same_left_count && gains_close;
        assert!(
            verified_tie,
            "{what}: default_left flipped on a NON-tie split (HARD-FAIL) — cpu(dl={}, thr={}, \
             lc={}, gain={}) vs gpu(dl={}, thr={}, lc={}, gain={})",
            cpu.default_left, cpu.threshold, cpu.left_count, cpu.gain,
            gpu.default_left, gpu.threshold, gpu.left_count, gpu.gain
        );
    }
}

#[test]
fn rocm_backend_default_left_tie() {
    let cc = cpu_client();
    let gc = rocm_client();
    // Anchor (cpu f64) vs mirror (rocm f32) — the pipeline is generic over the runtime; the
    // BackEnds tag which client is the anchor and which is the mirror (never GPU-vs-GPU).
    let _anchor = CpuBackend;
    let _mirror = RocmBackend::default();

    // ---- Near-tie leaf: a symmetric feature with a fwd(assume=false)+rev(assume=true)
    // task pair whose gains are f32-equal, so the stage-2 winner's default_left may flip
    // on the f32 mirror — the ACCEPTED-flip branch. ----
    let hist_tie: Vec<f64> = vec![-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let sc_tie = tie_scalars(0.0, 4.0, 4, 1);
    let tasks_tie = [tie_task(0, 4, false, false), tie_task(0, 4, true, true)];
    let cpu_tie = leaf_winner_f64(&cc, &tasks_tie, &hist_tie, &sc_tie);
    let hist_tie_f32: Vec<f32> = hist_tie.iter().map(|&x| x as f32).collect();
    let gpu_tie = leaf_winner_f32(&gc, &tasks_tie, &hist_tie_f32, &sc_tie);
    assert!(cpu_tie.is_valid, "near-tie: the cpu fold must find a valid split");
    assert_split_tie_aware(&cpu_tie, &gpu_tie, "near-tie leaf");

    // Drive stage-3 on both to exercise the single-readback 8-int export (SC#2). The
    // export's default_left ([2]) is tie-gated against the threshold ([1]).
    let mut cpu_leaves = vec![cpu_tie, SplitScalars::default()];
    let mut gpu_leaves = vec![gpu_tie, SplitScalars::default()];
    let cpu_export = find_best_from_all_splits_on(&cc, &mut cpu_leaves, 0, -1, 1).unwrap().cells;
    let gpu_export = find_best_from_all_splits_on(&gc, &mut gpu_leaves, 0, -1, 1).unwrap().cells;
    assert_eq!(cpu_export[6], gpu_export[6], "export best_leaf_index parity");
    if cpu_export[2] != gpu_export[2] {
        assert_eq!(
            cpu_export[1], gpu_export[1],
            "export default_left flip only on a verified tie (same threshold)"
        );
    }

    // ---- Clean-margin leaf: a SINGLE forward task over an asymmetric feature. A fwd/rev
    // PAIR is an inherent gain tie (both label the same physical split, whose gain is
    // scan-direction-symmetric), so `default_left` only flips there on a verified tie;
    // a single task writes `default_left = assume_out_default_left` VERBATIM (Pitfall 3),
    // so it is IDENTICAL on both backends — the no-flip branch, and the guard that a
    // non-tie flip would HARD-FAIL. ----
    let hist_clean: Vec<f64> = vec![-5.0, 1.0, -4.0, 1.0, 0.5, 1.0, 0.5, 1.0];
    let sc_clean = tie_scalars(-8.0, 4.0, 4, 1);
    let tasks_clean = [tie_task(1, 4, false, false)];
    let cpu_clean = leaf_winner_f64(&cc, &tasks_clean, &hist_clean, &sc_clean);
    let hist_clean_f32: Vec<f32> = hist_clean.iter().map(|&x| x as f32).collect();
    let gpu_clean = leaf_winner_f32(&gc, &tasks_clean, &hist_clean_f32, &sc_clean);
    assert!(cpu_clean.is_valid, "clean-margin: the cpu fold must find a valid split");
    assert_split_tie_aware(&cpu_clean, &gpu_clean, "clean-margin leaf");
    assert_eq!(
        cpu_clean.default_left, gpu_clean.default_left,
        "clean-margin single task: default_left is written verbatim → never flips"
    );

    // ---- Empty leaf: an impossible min_data_in_leaf gates every bin out → both backends
    // yield is_valid=false, no spurious default_left flip (SC#3 empty fixture). ----
    let sc_empty = tie_scalars(0.0, 4.0, 4, 1000);
    let tasks_empty = [tie_task(2, 4, false, false)];
    let cpu_empty = leaf_winner_f64(&cc, &tasks_empty, &hist_tie, &sc_empty);
    let gpu_empty = leaf_winner_f32(&gc, &tasks_empty, &hist_tie_f32, &sc_empty);
    assert!(!cpu_empty.is_valid && !gpu_empty.is_valid, "empty leaf: both is_valid=false");
    assert_split_tie_aware(&cpu_empty, &gpu_empty, "empty leaf");
}

#[test]
fn rocm_backend_data_partition_matches() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();

    let bins = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 2, 1, 3, 0];
    let (r_cpu, sp_cpu) = cpu.data_partition(&cc, &bins, 4, 0, 3, 1, 0).unwrap();
    let (r_gpu, sp_gpu) = gpu.data_partition(&gc, &bins, 4, 0, 3, 1, 0).unwrap();
    assert_eq!(sp_cpu, sp_gpu, "data_partition split_point mismatch");
    assert_eq!(r_cpu, r_gpu, "data_partition reordered indices mismatch");
}
