//! 23-02 Task 3 (L-1 / SC-2) — INSTRUMENTATION test: an on-device grow bumps the
//! compute-owned launch counter to a NON-ZERO value, folded through
//! [`on_device_launch_count_take`]. This proves the counter is wired at the real
//! build/subtract/scan dispatch sites so an on-device train can report a visible
//! `device_launches=` figure (previously suppressed at 0, P-2) for the Kaggle A/B
//! harness (ODL-20).
//!
//! Scope: this asserts INSTRUMENTATION, not parity — the on-device grow's numerical
//! faithfulness is covered by the existing `learner_parity` STRUCTURE gate (pinned to
//! the cpu f64 anchor). Here we only exercise the per-leaf build/subtract/scan loop on
//! the cubecl-cpu runtime over a tiny synthetic continuous-feature corpus and read the
//! counter back.

use lgbm_compute::kernels::grow_driver::{
    grow_tree_on_device_driver, on_device_launch_count_take,
};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};

/// The host-path launch denominator SC-2 targets: 8,570 launches / 100 trees.
const HOST_BASELINE_LAUNCHES: u64 = 8570;
/// The per-TREE host baseline (WR-02): 8,570 / 100 trees ≈ 86 leaf-level launches per
/// tree. The old test compared a SINGLE tiny tree against the whole 100-tree denominator,
/// so the bound was ~100× loose and could not fail for the very launch-non-collapse
/// regression SC-2 exists to detect. Comparing like-for-like on a per-tree basis makes
/// the bound meaningful.
const HOST_BASELINE_PER_TREE: u64 = HOST_BASELINE_LAUNCHES / 100;

/// `offset_for_most_freq_bin(0)` == 1 (drop bin 0 under the compacted convention). We
/// hardcode it here because the helper lives in `lgbm-treelearner`, ABOVE this crate.
const OFFSET_MFB_0: i32 = 1;

/// Build a tiny synthetic continuous-feature corpus: `num_features` 8-bin numeric
/// columns over `num_data` rows with strongly-separated MONOTONE gradients so every
/// split has a distinct positive gain and the driver actually runs the per-leaf
/// build/subtract/scan loop.
fn tiny_corpus(num_features: usize, num_data: usize) -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
    const NUM_BIN: u32 = 8;
    let features: Vec<GrowFeature> = (0..num_features)
        .map(|fi| {
            // Feature 0 is monotone in the row index (dominates splits); the rest are
            // phase-shifted so they carry real but weaker structure.
            let bins: Vec<u32> = (0..num_data)
                .map(|r| (((r + fi * 37) * NUM_BIN as usize / num_data) as u32) % NUM_BIN)
                .collect();
            GrowFeature {
                bins: BinColumn::new(bins, NUM_BIN),
                num_bin: NUM_BIN,
                offset: OFFSET_MFB_0,
                min_bin: 0,
                max_bin: NUM_BIN - 1,
                default_bin: NUM_BIN,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                real_feature_index: fi as i32,
                bin_type: BinType::Numerical,
                // Numeric feature: categorical metadata is inert.
                bin_to_category: Vec::new(),
                cat_smooth: 10.0,
                cat_l2: 10.0,
                max_cat_threshold: 32,
                max_cat_to_onehot: 4,
                min_data_per_group: 100,
            }
        })
        .collect();

    // Monotone gradients centered at zero ⇒ clean, near-tie-free positive-gain splits.
    let half = num_data as f32 / 2.0;
    let gradients: Vec<f32> = (0..num_data).map(|r| (r as f32 - half) * 0.1).collect();
    let hessians: Vec<f32> = vec![1.0f32; num_data];
    (features, gradients, hessians)
}

/// L-1 / SC-2: an on-device grow bumps the compute-owned launch counter to a non-zero
/// value that (a) sits at or below the PER-TREE host baseline and (b) matches the exact
/// PER-LEAF collapse bound — a per-feature regression (the WR-01 failure mode) would
/// scale with `num_features` and blow past this bound, so the test can actually fail for
/// launch non-collapse.
#[test]
fn on_device_grow_bumps_nonzero_launch_count() {
    // P-1: enable the read-once phase-prof gate BEFORE the FIRST counter read (the
    // `bump_launch`/`take` OnceLock reads env exactly once per process). This is the
    // first thing the single test process does.
    // SAFETY: set at the very top of the test process before any threads read the
    // OnceLock-cached gate; no concurrent env access.
    unsafe {
        std::env::set_var("LGBM_PHASE_PROF", "1");
    }

    // Clear any residual count so the assertion measures THIS grow only.
    let _ = on_device_launch_count_take();

    let client = cpu_client();
    let (features, gradients, hessians) = tiny_corpus(3, 256);
    let num_leaves = 8;
    let max_depth = -1; // no depth cap

    let (tree, _layout) = grow_tree_on_device_driver(
        &client,
        &gradients,
        &hessians,
        &features,
        num_leaves,
        max_depth,
    )
    .expect("on-device driver must grow the tiny corpus");

    // Sanity: the corpus actually grew a real multi-leaf tree (so the per-leaf loop ran).
    assert!(
        tree.num_leaves > 1,
        "corpus must split at least once so build/subtract/scan dispatch (got num_leaves={})",
        tree.num_leaves
    );

    let launches = on_device_launch_count_take();

    // The counter is wired: an on-device grow reports a NON-ZERO launch count (was
    // suppressed at 0 before, P-2).
    assert!(
        launches > 0,
        "on-device grow must bump a non-zero device launch count, got {launches}"
    );

    // WR-02: the EXACT per-leaf collapse bound. The driver bumps ONCE PER LEAF-LEVEL
    // build/subtract/scan (WR-01), independent of `num_features`. For a grow of
    // `num_leaves` leaves (`num_leaves - 1` splits) the maximum leaf-level launches are:
    //   builds   = 1 (root) + (num_leaves - 1) smaller-child builds
    //   subtracts = (num_leaves - 1)   (one larger-child subtract per split)
    //   scans    = 1 (root) + 2*(num_leaves - 1)  (both children per split, at most)
    // i.e. `2 + 4*(num_leaves - 1)`. A per-FEATURE regression (the WR-01 failure mode)
    // would multiply the build+scan terms by `num_features` and exceed this bound, so
    // this assertion actually fails for launch non-collapse — unlike the old ~100×-loose
    // `< 8570` check.
    let per_leaf_collapse_bound = 2 + 4 * (num_leaves as u64 - 1);
    assert!(
        launches <= per_leaf_collapse_bound,
        "on-device launch count {launches} must be <= the per-leaf collapse bound \
         {per_leaf_collapse_bound} (= 2 + 4*(num_leaves-1), num_leaves={num_leaves}); \
         exceeding it means the counter scaled with num_features (per-feature, not \
         per-leaf) — the WR-01 launch non-collapse regression"
    );

    // Like-for-like per-TREE sanity vs the host baseline (WR-02): a single collapsed
    // tree stays at or below the per-tree host denominator (8570/100 ≈ 86), not merely
    // below the whole 100-tree figure.
    assert!(
        launches <= HOST_BASELINE_PER_TREE,
        "on-device single-tree launch count {launches} must be <= the per-tree host \
         baseline {HOST_BASELINE_PER_TREE} (= {HOST_BASELINE_LAUNCHES}/100 trees)"
    );
}
