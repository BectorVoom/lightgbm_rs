//! Phase-20 on-device score-updater parity (20-01 / ODL-16, §11).
//!
//! Cells:
//! - `constant_ops_kernel_matches_host_score_updater` (Task 1) — the two §11
//!   whole-array scalar kernels (`add_score_constant_on` / `multiply_score_constant_on`)
//!   are bit-exact to the host `ScoreUpdater::{add_constant, multiply_score}` on the
//!   cpu f64 anchor, at a NON-root `tree_id` (proving the `offset = num_data * tree_id`
//!   class slice is correct).
//! - `resident_toggle_mirrors_host_accumulation` (Task 2) — the `boosting_on_cuda_`
//!   resident path (`add_constant_on` / `multiply_score_on`) mirrored back into the
//!   host `score_` is bit-exact to the pure-host accumulation over the same trees.
//! - `resident_train_path_delegates_to_phase18_kernel` (Task 2 / D-02) — the per-leaf
//!   `add_tree_train_path_on_device` delegates to the Phase-18
//!   `add_prediction_to_score_on_device` tree-walk kernel (NO new tree-walk kernel)
//!   and mirrors the per-row raw-margin delta into the correct class slice.
//!
//! Anchor discipline (D-05 / D-07): the host f64 `ScoreUpdater` is the oracle; on the
//! cpu f64 anchor every op is BIT-EXACT (`compare_exact_f64_bits`). NEVER GPU-vs-GPU
//! (def-f8u-01). With `LGBM_CUDA_ON_DEVICE` unset the boosting `ScoreUpdater` default
//! is the byte-unchanged host path; these cells force the resident toggle explicitly
//! via `set_boosting_on_cuda(true)` (the Plan-20-04 driver seam).

use lgbm_boosting::score_updater::ScoreUpdater;
use lgbm_compute::kernels::predict::PredictTree;
use lgbm_compute::kernels::score_updater::{add_score_constant_on, multiply_score_constant_on};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::CpuBackend;
use oracle_harness::comparator::compare_exact_f64_bits;

/// A small 2-class × 3-row f64 score buffer (class 0 = [1,2,3], class 1 = [4,5,6]).
fn seed_scores() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
}

/// Task 1 — the §11 constant-op kernels are bit-exact to the host
/// `ScoreUpdater::{add_constant, multiply_score}` on the cpu f64 anchor, applied to a
/// NON-root class slice (`tree_id = 1`, `offset = num_data * 1`).
#[test]
fn constant_ops_kernel_matches_host_score_updater() {
    let client = cpu_client();
    let num_data = 3i32;

    // --- AddScoreConstant on class 1 ---
    let base = seed_scores();
    // Host reference: ScoreUpdater::add_constant(0.5, tree_id = 1).
    let mut host = ScoreUpdater::new(num_data, 2, Some(&base));
    host.add_constant(0.5, 1);
    // Device kernel over the whole resident buffer.
    let dev = add_score_constant_on(&client, &base, num_data, 1, 0.5).expect("add kernel");
    compare_exact_f64_bits(&dev, host.scores())
        .unwrap_or_else(|m| panic!("add_score_constant_on != host add_constant: {m:?}"));

    // --- MultiplyScoreConstant on class 1 (non-root offset proof) ---
    let base = seed_scores();
    let mut host = ScoreUpdater::new(num_data, 2, Some(&base));
    host.multiply_score(2.0, 1);
    let dev = multiply_score_constant_on(&client, &base, num_data, 1, 2.0).expect("mul kernel");
    compare_exact_f64_bits(&dev, host.scores())
        .unwrap_or_else(|m| panic!("multiply_score_constant_on != host multiply_score: {m:?}"));

    // Root tree_id (offset 0) also matches, and leaves class 1 untouched.
    let base = seed_scores();
    let mut host = ScoreUpdater::new(num_data, 2, Some(&base));
    host.add_constant(-1.0, 0);
    let dev = add_score_constant_on(&client, &base, num_data, 0, -1.0).expect("add root");
    compare_exact_f64_bits(&dev, host.scores())
        .unwrap_or_else(|m| panic!("add_score_constant_on(root) != host: {m:?}"));
}

/// Task 2 — the resident `boosting_on_cuda_` constant path, mirrored back into the
/// host `score_`, is bit-exact to a pure-host `ScoreUpdater` accumulation over the
/// same sequence of trees (init add + shrinkage multiply, per class).
#[test]
fn resident_toggle_mirrors_host_accumulation() {
    let client = cpu_client();
    let num_data = 4i32;
    let num_class = 2i32;

    // Pure-host reference (toggle OFF — byte-unchanged path).
    let mut host = ScoreUpdater::new(num_data, num_class, None);
    // Resident (toggle FORCED ON — the Plan-20-04 driver seam).
    let mut resident = ScoreUpdater::new(num_data, num_class, None);
    resident.set_boosting_on_cuda(true);
    assert!(resident.boosting_on_cuda(), "toggle must be forced on for this cell");

    // Two "iterations" over two classes: init-score add then a shrinkage multiply.
    for tree_id in 0..num_class {
        let init = 1.5 * f64::from(tree_id + 1);
        host.add_constant(init, tree_id);
        resident
            .add_constant_on::<CpuBackend>(&client, init, tree_id)
            .expect("resident add_constant_on");

        let shrink = 0.1;
        host.multiply_score(shrink, tree_id);
        resident
            .multiply_score_on::<CpuBackend>(&client, shrink, tree_id)
            .expect("resident multiply_score_on");
    }

    compare_exact_f64_bits(resident.scores(), host.scores())
        .unwrap_or_else(|m| panic!("resident mirror != pure-host accumulation: {m:?}"));

    // A default-constructed `ScoreUpdater` must adopt the LIBRARY-resolved on-device
    // default — `cuda_on_device_enabled()`, i.e. the `LGBM_CUDA_ON_DEVICE` override
    // falling back to the build's `on_device_default()`. Asserting a hard-coded ON
    // here encoded a default that has since changed, so this cell failed on any
    // env-unset CPU-only build even though `ScoreUpdater` was behaving correctly.
    let default_su = ScoreUpdater::new(num_data, num_class, None);
    assert_eq!(
        default_su.boosting_on_cuda(),
        lgbm_compute::cuda_on_device_enabled(),
        "a default ScoreUpdater must adopt the resolved on-device default"
    );
}

/// The Phase-18 numeric parity tree (predict.rs golden shape): 2 features, 3 leaves,
/// raw margins `[-0.5, 1.75, -0.5, 1.75, -0.5]` for the fixed bin rows below.
#[allow(clippy::type_complexity)]
fn numeric_tree() -> (
    Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>,
    Vec<i32>, Vec<u32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>,
) {
    let feat_min = vec![1, 1];
    let feat_max = vec![9, 9];
    let feat_default = vec![0, 0];
    let feat_mfb = vec![0, 0];
    let feat_offset = vec![0, 0];
    let feat_column = vec![0, 1];
    let sfi = vec![0, 1];
    let thr = vec![3u32, 2];
    let dt = vec![6, 0];
    let lc = vec![1, -1];
    let rc = vec![-3, -2];
    let leaf = vec![-0.5f64, 0.25, 1.75];
    (feat_min, feat_max, feat_default, feat_mfb, feat_offset, feat_column, sfi, thr, dt, lc, rc, leaf)
}

/// Task 2 / D-02 — the resident training-path `add_tree_train_path_on_device`
/// delegates to the Phase-18 `add_prediction_to_score_on_device` tree-walk kernel
/// (no new tree-walk kernel) and adds its per-row raw-margin delta into the correct
/// class slice, mirrored to host.
#[test]
fn resident_train_path_delegates_to_phase18_kernel() {
    let client = cpu_client();
    let num_data = 5i32;
    let num_class = 2i32;

    let (fmin, fmax, fdef, fmfb, foff, fcol, sfi, thr, dt, lc, rc, leaf) = numeric_tree();
    let tree = PredictTree {
        feat_min: &fmin,
        feat_max: &fmax,
        feat_default: &fdef,
        feat_most_freq_bin: &fmfb,
        feat_offset: &foff,
        feat_column: &fcol,
        split_feature_inner: &sfi,
        threshold_in_bin: &thr,
        decision_type: &dt,
        left_child: &lc,
        right_child: &rc,
        leaf_value: &leaf,
        bitset_inner: &[],
        cat_boundaries_inner: &[],
    };
    // rows row-major [5 × 2]; the golden raw margins for these bins.
    let rows: Vec<u32> = vec![2, 1, 5, 4, 1, 3, 9, 2, 3, 1];
    let margins = [-0.5f64, 1.75, -0.5, 1.75, -0.5];

    // Base score: class 1 pre-loaded so the offset math is exercised at a non-root id.
    let mut base = vec![0.0f64; (num_data * num_class) as usize];
    let off = num_data as usize; // tree_id = 1 offset.
    for (i, v) in [10.0, 20.0, 30.0, 40.0, 50.0].iter().enumerate() {
        base[off + i] = *v;
    }

    let mut resident = ScoreUpdater::new(num_data, num_class, Some(&base));
    resident.set_boosting_on_cuda(true);
    resident
        .add_tree_train_path_on_device::<CpuBackend>(&client, &tree, &rows, 2, 8, 1)
        .expect("resident per-leaf device delegate");

    // Expected: class 1 = base + margins; class 0 untouched.
    let expected_class1: Vec<f64> =
        [10.0, 20.0, 30.0, 40.0, 50.0].iter().zip(margins.iter()).map(|(b, m)| b + m).collect();
    compare_exact_f64_bits(resident.class_scores(1), &expected_class1)
        .unwrap_or_else(|m| panic!("resident train-path delta (class 1) mismatch: {m:?}"));
    compare_exact_f64_bits(resident.class_scores(0), &vec![0.0f64; num_data as usize])
        .unwrap_or_else(|m| panic!("resident train-path leaked into class 0: {m:?}"));
}
