//! The on-device launch counter reports an HONEST REAL-DISPATCH
//! total, and the test can DETECT launch non-collapse (a per-FEATURE
//! regression). Folded through [`on_device_launch_count_take`], the counter is bumped
//! ONCE per real device dispatch on the resident arm (upload / build_fix_scan /
//! build_resident / subtract / scan / data_partition_native; a co-packed sibling scan is
//! ONE dispatch) and once per per-leaf phase on the anchor arm — so the count is
//! INDEPENDENT of `num_features`. A per-feature layout would scale the build+scan terms
//! with `num_features`, so the num_features-independence assertion below FAILS on the
//! non-collapse regression the old ~100×-loose `< 8570` bound could never catch.
//!
//! ## Two lanes
//! - **Default build (cpu lane):** `CpuBackend::resident_pool_supported() == false`, so the
//!   driver takes the ANCHOR arm. This lane is the reachable merge-gate check: it asserts
//!   the launch count is EQUAL at `num_features = 3` and `= 12` (the definitive
//!   launch-collapse signal) and sits at/under the per-leaf collapse bound.
//! - **`--features rocm` (resident lane):** `RocmBackend::with_resident(true)` drives the
//!   FAST resident arm. This lane asserts the EXACT analytic real-dispatch closed form for
//!   the fully-grown tiny corpus under the default HOST partition route, AND re-checks
//!   num_features-independence on the real resident launchers.
//!
//! Scope: INSTRUMENTATION, not parity — the on-device grow's numerical faithfulness is
//! covered by the `learner_parity` STRUCTURE gate (pinned to the cpu f64 anchor).
//!
//! ## Resident row permutation note (launch closed form UNCHANGED)
//! The row permutation is resident (a device index RANGE into a single `perm` buffer)
//! and partitions it IN PLACE per split. On the DEFAULT tested lane the partition still routes on
//! the HOST (`prefers_host_partition()`), which issues NO device dispatch, and the build / subtract
//! / scan dispatch layout is untouched — so the real-dispatch closed form
//! `4 + 3*(num_leaves-1)` is UNCHANGED and still num_features-independent. (On the non-default
//! on-device-partition arm the resident scatter is still ONE dispatch, so that arm's form is also
//! unchanged.) No readback/dispatch is removed on the tested lane, so no closed form is revised.

use lgbm_compute::kernels::grow_driver::{
    grow_tree_on_device_driver, on_device_f64_fused_count_take, on_device_launch_count_take,
    on_device_rootbuild_u64_count_take,
};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, CpuBackend, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};

/// `offset_for_most_freq_bin(0)` == 1 (drop bin 0 under the compacted convention). We
/// hardcode it here because the helper lives in `lgbm-treelearner`, ABOVE this crate.
const OFFSET_MFB_0: i32 = 1;

/// Tiny corpus size + shape shared by every lane: `NUM_LEAVES` leaves over `NUM_DATA`
/// rows. `NUM_DATA` is large enough that every one of the `NUM_LEAVES - 1` splits yields
/// two scannable (`>= 2`-row) children, so the tree grows to the full `NUM_LEAVES` and the
/// resident lane hits the exact analytic real-dispatch count.
const NUM_LEAVES: i32 = 8;
const NUM_DATA: usize = 256;

/// Build a tiny synthetic continuous-feature corpus: `num_features` 8-bin numeric columns
/// over `num_data` rows. Feature 0 is monotone in the row index and thus DOMINATES every
/// split (lowest real index also wins gain ties), so the tree STRUCTURE — and therefore the
/// per-leaf partition / scannability / dispatch count — is IDENTICAL regardless of how many
/// extra (weaker) features are present. That invariance is exactly what the
/// num_features-independence assertion exploits to detect a per-feature launch regression.
fn tiny_corpus(num_features: usize, num_data: usize) -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
    const NUM_BIN: u32 = 8;
    let features: Vec<GrowFeature> = (0..num_features)
        .map(|fi| {
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

/// Grow the tiny corpus on the ANCHOR arm (`CpuBackend`) with `num_features` columns and
/// return the real-dispatch launch count for THIS grow only. Clears the counter first so
/// the reading is isolated. Also sanity-checks the corpus grew a real multi-leaf tree.
fn anchor_launch_count(num_features: usize) -> u64 {
    let _ = on_device_launch_count_take(); // isolate this grow's count
    let client = cpu_client();
    let (features, gradients, hessians) = tiny_corpus(num_features, NUM_DATA);
    let (tree, _layout) = grow_tree_on_device_driver(
        &CpuBackend,
        &client,
        &gradients,
        &hessians,
        &features,
        NUM_LEAVES,
        -1, // no depth cap
    )
    .expect("on-device anchor driver must grow the tiny corpus");
    assert!(
        tree.num_leaves > 1,
        "corpus must split at least once so build/subtract/scan dispatch (num_features={num_features}, got num_leaves={})",
        tree.num_leaves
    );
    on_device_launch_count_take()
}

/// cpu lane: the launch count is NON-ZERO and INDEPENDENT of `num_features`.
///
/// Growing the SAME dominant-feature corpus with 3 vs 12 feature columns must report the
/// EXACT same launch count — the per-leaf/real-dispatch counter never scales with the
/// number of features. A per-FEATURE regression (the WR-01 failure mode) would
/// multiply the build+scan launches by `num_features`, breaking this equality.
#[test]
fn on_device_launch_count_is_num_features_independent() {
    // Enable the read-once phase-prof gate BEFORE the FIRST counter read (the
    // `bump_launch`/`take` OnceLock reads env exactly once per process). This single test
    // owns the process, so setting it first is race-free.
    // SAFETY: set at the very top before any grow reads the OnceLock-cached gate; no
    // concurrent env access in this single-threaded test.
    unsafe {
        std::env::set_var("LGBM_PHASE_PROF", "1");
    }

    let launches_3 = anchor_launch_count(3);
    let launches_12 = anchor_launch_count(12);

    // The counter is wired: an on-device grow reports a NON-ZERO launch count.
    assert!(
        launches_3 > 0,
        "on-device grow must bump a non-zero device launch count, got {launches_3}"
    );

    // The definitive launch-COLLAPSE check: identical count at 3 and 12 features. A
    // per-feature regression scales build+scan with num_features and breaks this equality.
    assert_eq!(
        launches_3, launches_12,
        "on-device launch count must be INDEPENDENT of num_features (got {launches_3} at 3 \
         features vs {launches_12} at 12) — inequality means the counter scaled per-feature \
         (the spike-055 / WR-01 launch non-collapse regression)"
    );

    // Per-leaf collapse upper bound (like-for-like, single tree): builds = 1 root +
    // (num_leaves-1) smaller-child; subtracts = (num_leaves-1); scans <= 1 root +
    // 2*(num_leaves-1) ⇒ `2 + 4*(num_leaves-1)`. A per-feature layout would exceed it.
    let per_leaf_collapse_bound = 2 + 4 * (NUM_LEAVES as u64 - 1);
    assert!(
        launches_3 <= per_leaf_collapse_bound,
        "anchor-arm launch count {launches_3} must be <= the per-leaf collapse bound \
         {per_leaf_collapse_bound} (= 2 + 4*(num_leaves-1), num_leaves={NUM_LEAVES})"
    );

    // NEGATIVE guard, CI-reachable on the cpu anchor lane: the f64 single-owner fused build
    // (`build_fix_scan_resident_f64_on`, a much slower kernel) is NEVER dispatched on the
    // default on-device path. The cpu anchor arm does not enter the resident driver at all,
    // so this is trivially 0 here; the rocm resident lane below exercises the real SWAPPED
    // path where the guard is load-bearing (it asserts == 0 there too, on the arm that USED
    // to dispatch the f64 fused build). Drain the positive counter as well so no
    // process-global state leaks into the resident lane's per-grow reads.
    let _ = on_device_rootbuild_u64_count_take();
    let f64_fused = on_device_f64_fused_count_take();
    assert_eq!(
        f64_fused, 0,
        "on-device f64 single-owner fused build must NEVER dispatch on the default path \
         (LGBM_ONDEVICE_F64_FUSED unset); got {f64_fused} — the spike-052 slow kernel silently \
         returned to the on-device build"
    );

    // On a rocm host, additionally assert the EXACT analytic real-dispatch closed form on
    // the resident fast arm (and re-check num_features-independence there), PLUS the
    // positive/negative swap proof on the arm that actually runs the u64 build.
    #[cfg(feature = "rocm")]
    resident_analytic_lane();
}

/// `--features rocm` lane: drive the FAST resident arm (`RocmBackend::with_resident(true)`)
/// and assert the EXACT analytic real-dispatch count for the fully-grown tiny corpus under
/// the DEFAULT host partition route, then re-check num_features-independence on the real
/// resident launchers. The env gate is already set by the caller.
///
/// Analytic real-dispatch closed form (host partition route, DEFAULT co-pack ON, every split
/// has two scannable children so the tree reaches `num_leaves` leaves). The on-device root +
/// directly-built build runs the PARALLEL u64 fixed-point kernel via `build_resident_leaf`
/// (BUILD only) + a SEPARATE resident scan (not a fused f64 single-owner build+scan):
///   upload (1) + root [ build (1) + scan (1) = 2 ]
///     + per split [ build_resident (1) + subtract (1) + `scan_resident_siblings` (1) = 3 ]
///   = `3 + 3*(num_leaves - 1)`.
/// The host partition route issues NO device dispatch (0). This closed form assumes DEFAULT
/// co-pack (ON) — the test removes `LGBM_SIBLING_COPACK` below to pin it. The per-split count
/// is co-pack-DEPENDENT (co-pack-OFF is build_resident + subtract + smaller scan + larger
/// scan = 4), so the closed form is stated for the default co-pack-ON path the test exercises.
#[cfg(feature = "rocm")]
fn resident_analytic_lane() {
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::{Backend, RocmBackend};

    // Ensure the default HOST partition route (the analytic form assumes it: on-device
    // partition would add one dispatch per split) AND the default co-pack-ON path (the
    // closed form is co-pack-dependent post-swap). Removed after so no state leaks.
    // SAFETY: single-threaded test; no concurrent env access.
    unsafe {
        std::env::remove_var("LGBM_ROCM_HOST_PARTITION");
        std::env::remove_var("LGBM_SIBLING_COPACK");
        // Ensure the DEFAULT (swapped) u64 build path — never the f64-fused escape hatch.
        std::env::remove_var("LGBM_ONDEVICE_F64_FUSED");
    }

    let backend = RocmBackend::with_resident(true);
    assert!(
        backend.resident_pool_supported(),
        "with_resident(true) must drive the resident fast arm"
    );
    let client = rocm_client();

    // Returns (launches, rootbuild_u64_count, f64_fused_count, num_leaves, leaf_count) for
    // THIS grow only — all four process-global counters are drained before and read after so
    // each grow's figures are isolated.
    let grow_resident = |num_features: usize| -> (u64, u64, u64, i32, Vec<i32>) {
        let _ = on_device_launch_count_take();
        let _ = on_device_rootbuild_u64_count_take();
        let _ = on_device_f64_fused_count_take();
        let (features, g, h) = tiny_corpus(num_features, NUM_DATA);
        let (tree, layout) =
            grow_tree_on_device_driver(&backend, &client, &g, &h, &features, NUM_LEAVES, -1)
                .expect("resident driver must grow the tiny corpus");
        (
            on_device_launch_count_take(),
            on_device_rootbuild_u64_count_take(),
            on_device_f64_fused_count_take(),
            tree.num_leaves,
            layout.leaf_count,
        )
    };

    let (launches_3, rootbuild_3, f64_fused_3, leaves_3, leaf_count_3) = grow_resident(3);
    let (launches_12, _rootbuild_12, f64_fused_12, leaves_12, _) = grow_resident(12);

    // POSITIVE proof: the root + directly-built build sites actually ran the PARALLEL u64
    // fixed-point kernel — a nonzero root-build sub-counter (>= 1 for the root; the co-pack
    // splits build via the already-u64 subtract path and do NOT bump this scoped counter).
    // Proves the kernel choice AUTOMATICALLY, not by code-tracing.
    assert!(
        rootbuild_3 > 0,
        "on-device root/directly-built build must dispatch the parallel u64 kernel (rootbuild \
         sub-counter > 0), got {rootbuild_3} — the converted sites did not run u64"
    );

    // NEGATIVE guard: the f64 single-owner fused build must NEVER be dispatched on the
    // default on-device path — this is the arm that could otherwise silently run
    // `build_fix_scan_resident_f64_on` (a much slower kernel). `== 0` at BOTH feature
    // counts so the slow kernel can never silently return on-device.
    assert_eq!(
        f64_fused_3, 0,
        "resident lane: f64 single-owner fused build must NEVER dispatch on the default swapped \
         path, got {f64_fused_3}"
    );
    assert_eq!(
        f64_fused_12, 0,
        "resident lane: f64 single-owner fused build must NEVER dispatch on the default swapped \
         path (12 features), got {f64_fused_12}"
    );

    // The corpus must reach the full leaf count with every leaf scannable (>= 2 rows) so
    // the exact closed form applies.
    assert_eq!(
        leaves_3, NUM_LEAVES,
        "resident lane: corpus must grow the full {NUM_LEAVES} leaves for the exact bound (got {leaves_3})"
    );
    assert!(
        leaf_count_3.iter().all(|&c| c >= 2),
        "resident lane: every leaf must have >= 2 rows so both children of every split were \
         scannable (3-dispatch splits); leaf_count={leaf_count_3:?}"
    );

    // num_features-independence on the REAL resident launchers.
    assert_eq!(
        leaves_3, leaves_12,
        "resident lane: tree must be identical across num_features (feature 0 dominates)"
    );
    assert_eq!(
        launches_3, launches_12,
        "resident lane: real-dispatch launch count must be INDEPENDENT of num_features \
         (got {launches_3} at 3 vs {launches_12} at 12) — a per-feature scan/build layout \
         would break this"
    );

    // EXACT analytic real-dispatch closed form (host partition route, DEFAULT co-pack ON):
    // bin upload (1) + grad/hess upload (1) + root [build (1) + scan (1)] + per split
    // [build_resident (1) + subtract (1) + siblings_scan (1)] = 4 + 3*(num_leaves-1). The
    // on-device grad/hess gather hoists the per-build host grad/hess upload to ONE
    // once-per-grow device dispatch (num_features- AND num_leaves-independent), counted
    // like the bin upload.
    let analytic = 4 + 3 * (NUM_LEAVES as u64 - 1);
    assert_eq!(
        launches_3, analytic,
        "resident lane: real-dispatch launch count {launches_3} must equal the analytic \
         closed form {analytic} (= 4 + 3*(num_leaves-1): bin_upload + grad_hess_upload + \
         root[build+scan] + build_resident/subtract/siblings_scan per split, host partition \
         route, co-pack ON; num_leaves={NUM_LEAVES})"
    );
}
