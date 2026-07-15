//! The on-device BLOCKING-READBACK sync counter reports an HONEST,
//! NON-ZERO, num_features-INDEPENDENT total, and is DISTINCT from the launch counter.
//!
//! Where `on_device_launch_count` proves the DISPATCH total collapses per-leaf (not
//! per-feature), this proves the companion SYNC total — the real device→host blocking
//! readbacks (scan / on-device partition / tree-split `right_leaf_index`) — is wired and
//! counts REAL syncs, not one-per-feature and not one-per-child (the co-packed sibling
//! scan is ONE sync).
//!
//! ## Two lanes, two closed forms
//! - **cpu anchor lane (the RUNNABLE reference):** the blocking-readback count is the
//!   documented pre-collapse baseline `ANCHOR_SYNC_BASELINE = 1 + 3*(num_leaves-1)` (root
//!   scan + per split [`split_on_device` + 2 SEPARATE child scans]). This lane does NOT
//!   collapse — it IS the reference the resident lane's drop is measured against
//!   (`CpuBackend` is the anchor arm; it never co-packs).
//! - **rocm resident lane (cfg-gated, real-hardware):** the resident control plane's count
//!   is the EXACT closed form `1 + (num_leaves-1)` = `num_leaves` (root scan + the
//!   per-iteration §8.3 pick export — the ONE by-design host crossing). Every per-split
//!   scan readback is retired: each scanned child's winner (`gain`/`left_output`/
//!   `right_output`) is folded device→device into the resident frontier instead of being
//!   read back to the host, on all 3 scan arms (default separate-scan, default-ON co-pack,
//!   and the f64-fused escape hatch). This is STRICTLY BELOW the cpu anchor's pre-collapse
//!   baseline for every `num_leaves > 1`, and the exact drop from the anchor is
//!   `2*(num_leaves-1)` (both per-split child-scan readbacks folded device→device; the
//!   `split_on_device` readback is replaced count-neutrally by the per-iteration pick
//!   export).
//!
//! Genuinely removing MORE readbacks (a reduce-before-copy resident device reduce, or a
//! resident device scatter for the row permutation) would need new kernels this test does
//! not exercise and is not verifiable on the local spoofed 8-CU APU; those are wall-clock
//! refinements confirmed separately on real hardware, not a change to this closed form.
//!
//! ## Two lanes (mirrors `on_device_launch_count`)
//! - **Default build (cpu anchor lane):** `CpuBackend::resident_pool_supported() == false`,
//!   so the driver takes the ANCHOR arm. Every scanned leaf issues ONE blocking scan
//!   readback and every split issues ONE `split_on_device` readback, so the sync count is
//!   `scanned_leaves + splits` — NON-ZERO and INDEPENDENT of `num_features` (feature 0
//!   dominates every split, so the tree STRUCTURE, and therefore every scan/split, is
//!   identical at 3 vs 12 features). A per-feature readback layout would scale the scan
//!   term with `num_features` and break the equality.
//! - **`--features rocm` (resident lane):** drives the resident fast arm and asserts the
//!   EXACT analytic sync closed form `1 + (num_leaves-1)` = `num_leaves`: root
//!   scan + the per-iteration §8.3 pick export, with the per-split scan readbacks folded
//!   device→device (retired) and NO partition readback on the host route — plus
//!   num_features-independence on the real resident launchers.
//!
//! Scope: INSTRUMENTATION, not parity — the on-device grow's faithfulness is covered by the
//! `learner_parity` STRUCTURE gate (pinned to the cpu f64 anchor).

use lgbm_compute::kernels::grow_driver::{
    grow_tree_on_device_driver, on_device_sync_count_take,
};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, CpuBackend, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};

/// `offset_for_most_freq_bin(0)` == 1 (drop bin 0 under the compacted convention). The
/// helper lives in `lgbm-treelearner`, ABOVE this crate, so hardcode it (as the launch test).
const OFFSET_MFB_0: i32 = 1;

/// Tiny corpus shape shared by every lane (identical to `on_device_launch_count`): 8 leaves
/// over 256 rows, every split scannable so the tree reaches the full leaf count.
const NUM_LEAVES: i32 = 8;
const NUM_DATA: usize = 256;

/// DOCUMENTED PRE-COLLAPSE BASELINE: the cpu
/// anchor lane's blocking-readback count for the fully-grown tiny corpus. Root scan (1) +
/// per split [split_on_device (1) + 2 child scans (2)] = `1 + 3*(num_leaves-1)`. This is the
/// baseline the resident lane's on-device argmax + row permutation drive down (see the
/// module doc for the resident lane's closed form).
const ANCHOR_SYNC_BASELINE: u64 = 1 + 3 * (NUM_LEAVES as u64 - 1);

/// Build the SAME tiny synthetic dominant-feature corpus the launch-count test uses:
/// `num_features` 8-bin numeric columns over `num_data` rows; feature 0 is monotone in the
/// row index so it DOMINATES every split ⇒ the tree structure (and thus every scan/split
/// readback) is identical regardless of how many extra weaker features exist. That invariance
/// is what the num_features-independence assertion exploits.
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

    let half = num_data as f32 / 2.0;
    let gradients: Vec<f32> = (0..num_data).map(|r| (r as f32 - half) * 0.1).collect();
    let hessians: Vec<f32> = vec![1.0f32; num_data];
    (features, gradients, hessians)
}

/// Grow the tiny corpus on the ANCHOR arm (`CpuBackend`) with `num_features` columns and
/// return the blocking-readback SYNC count for THIS grow only (drained first for isolation).
fn anchor_sync_count(num_features: usize) -> u64 {
    let _ = on_device_sync_count_take(); // isolate this grow's syncs
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
        "corpus must split at least once so scan/split readbacks fire (num_features={num_features}, got num_leaves={})",
        tree.num_leaves
    );
    on_device_sync_count_take()
}

/// cpu anchor lane: the blocking-readback sync count is NON-ZERO, INDEPENDENT of
/// `num_features`, and equals the documented pre-collapse baseline.
#[test]
fn on_device_sync_count_is_num_features_independent() {
    // P-1: enable the read-once phase-prof gate BEFORE the FIRST counter read (the
    // `bump_sync`/`take` OnceLock reads env exactly once per process). This single test owns
    // the process, so setting it first is race-free.
    // SAFETY: set at the very top before any grow reads the OnceLock-cached gate; no
    // concurrent env access in this single-threaded test.
    unsafe {
        std::env::set_var("LGBM_PHASE_PROF", "1");
    }

    let syncs_3 = anchor_sync_count(3);
    let syncs_12 = anchor_sync_count(12);

    // The counter is wired: an on-device grow reports a NON-ZERO blocking-readback count.
    assert!(
        syncs_3 > 0,
        "on-device grow must bump a non-zero blocking-readback sync count, got {syncs_3}"
    );

    // The counter-trap guard: identical count at 3 and 12 features. A per-feature readback
    // regression (or one-sync-per-child on co-packed scans) would break this equality.
    assert_eq!(
        syncs_3, syncs_12,
        "blocking-readback sync count must be INDEPENDENT of num_features (got {syncs_3} at 3 \
         vs {syncs_12} at 12) — inequality means a readback scaled per-feature (the counter trap)"
    );

    // The documented pre-collapse baseline: root scan +
    // per-split [split_on_device + 2 child scans] = 1 + 3*(num_leaves-1).
    assert_eq!(
        syncs_3, ANCHOR_SYNC_BASELINE,
        "cpu anchor lane blocking-readback baseline must equal {ANCHOR_SYNC_BASELINE} \
         (= 1 + 3*(num_leaves-1): root scan + per split [split_on_device + 2 child scans], \
         num_leaves={NUM_LEAVES}); got {syncs_3}"
    );

    // On a rocm host, additionally assert the EXACT analytic sync closed form on the resident
    // fast arm and re-check num_features-independence there.
    #[cfg(feature = "rocm")]
    resident_sync_lane();
    // SPEC-DRGL-06: the DEFERRED-sync arm (LGBM_GROW_DEFER_SYNC=1) has its own, LOWER closed
    // form — the per-split read_split is fused into the pick read.
    #[cfg(feature = "rocm")]
    resident_defer_sync_lane();
}

/// `--features rocm` lane: drive the resident fast arm (`RocmBackend::with_resident(true)`)
/// and assert the EXACT analytic blocking-readback sync count under the DEFAULT host
/// partition route + co-pack ON, then re-check num_features-independence on the real
/// resident launchers. The env gate is already set by the caller.
///
/// Analytic sync closed form: `1 + (num_leaves-1)` =
/// `num_leaves`. Re-derived here from a fresh `bump_sync()` grep of `grow_driver.rs`
/// (`grow_tree_on_device_resident`):
///   - root scan (1): `scan_resident_and_argmax` bumps ONE readback (grow_driver.rs ~:1870);
///     the f64-fused root escape hatch bumps its fused build+scan readback instead (~:2277) —
///     still exactly 1.
///   - per-iteration §8.3 pick export (num_leaves-1): `frontier_pick_best_leaf_device` reads
///     back the ~10-cell winner ONCE per grow-loop iteration (grow_driver.rs ~:2443).
///   - per-split scan readbacks: ZERO. Every scanned child's winner is folded device→device
///     into the resident frontier
///     (`scan_resident_leaf_into_frontier` / `scan_resident_siblings_into_frontier` /
///     `build_fix_scan_resident_into_frontier`) — NO `bump_sync`, on ALL 3 arms (co-pack
///     default-ON, `LGBM_SIBLING_COPACK=0`, `LGBM_ONDEVICE_F64_FUSED=1`).
///   - partition readbacks: ZERO on the DEFAULT host partition route (`prefers_host_partition`);
///     the on-device partition arm's `bump_sync` (~:1993) is not on this lane.
/// Builds / subtracts / uploads / the scheduled tree-split (R3, no-readback) never bump. The
/// count is num_features-independent AND identical across all 3 arms.
///
/// This closed form supersedes an earlier `1 + 2*(num_leaves-1)` form that counted a
/// per-split co-packed siblings scan readback: the zero-readback reduce-into-frontier fold
/// retired that per-split scan readback, dropping the resident lane from `1 + 2*(n-1)` to
/// `1 + (n-1)`.
#[cfg(feature = "rocm")]
fn resident_sync_lane() {
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::{Backend, RocmBackend};

    // Ensure the DEFAULT host partition route (on-device partition would add one readback per
    // split) AND default co-pack ON (the closed form is co-pack-dependent), and the default
    // (u64-swapped) build (never the f64-fused escape hatch, which adds a fused-scan readback).
    // SAFETY: single-threaded test; no concurrent env access.
    unsafe {
        std::env::remove_var("LGBM_ROCM_HOST_PARTITION");
        std::env::remove_var("LGBM_SIBLING_COPACK");
        std::env::remove_var("LGBM_ONDEVICE_F64_FUSED");
    }
    // The RESIDENT-PERM partition arm is now DEFAULT-ON (spike093) and has its own
    // sync closed form (asserted separately below) — pin it OFF for the legacy
    // host-partition closed form this lane was derived for (read-once env gate ⇒
    // in-process override). Restored to `None` at the end of the lane.
    lgbm_compute::kernels::grow_driver::set_partition_resident_override(Some(false));

    let backend = RocmBackend::with_resident(true);
    assert!(
        backend.resident_pool_supported(),
        "with_resident(true) must drive the resident fast arm"
    );
    let client = rocm_client();

    let grow_resident = |num_features: usize| -> (u64, i32) {
        let _ = on_device_sync_count_take();
        let (features, g, h) = tiny_corpus(num_features, NUM_DATA);
        let (tree, _layout) =
            grow_tree_on_device_driver(&backend, &client, &g, &h, &features, NUM_LEAVES, -1)
                .expect("resident driver must grow the tiny corpus");
        (on_device_sync_count_take(), tree.num_leaves)
    };

    let (syncs_3, leaves_3) = grow_resident(3);
    let (syncs_12, leaves_12) = grow_resident(12);

    assert_eq!(
        leaves_3, NUM_LEAVES,
        "resident lane: corpus must grow the full {NUM_LEAVES} leaves for the exact bound (got {leaves_3})"
    );
    assert_eq!(
        leaves_3, leaves_12,
        "resident lane: tree must be identical across num_features (feature 0 dominates)"
    );
    assert_eq!(
        syncs_3, syncs_12,
        "resident lane: blocking-readback sync count must be INDEPENDENT of num_features \
         (got {syncs_3} at 3 vs {syncs_12} at 12)"
    );

    // EXACT analytic sync closed form (host partition route, co-pack ON):
    // root scan (1) + per-iteration §8.3 pick export (num_leaves-1) = 1 + (num_leaves-1) =
    // num_leaves. The per-split scan readbacks are RETIRED (each scanned child's
    // winner is folded device→device into the resident frontier — no `bump_sync`).
    let analytic = 1 + (NUM_LEAVES as u64 - 1);
    assert_eq!(
        syncs_3, analytic,
        "resident lane: blocking-readback sync count {syncs_3} must equal the analytic closed \
         form {analytic} (= 1 + (num_leaves-1): root scan + per-iteration §8.3 pick export; \
         per-split scans fold device→device post-Plan-08, host partition route, co-pack ON; \
         num_leaves={NUM_LEAVES})"
    );

    // The RESIDENT control plane's per-leaf blocking-readback count is STRICTLY BELOW the
    // cpu anchor's pre-collapse baseline `1 + 3*(num_leaves-1)`. This is the machine-checked
    // drop — exhibited on THIS lane (the resident arm genuinely co-packs), NOT faked on the
    // cpu anchor lane (which IS the baseline reference). It counts REAL dispatches: the
    // co-packed sibling scan is ONE readback (not two), the host partition route is ZERO,
    // the resident permutation adds none.
    assert!(
        syncs_3 < ANCHOR_SYNC_BASELINE,
        "ODP3-07: resident blocking-readback count {syncs_3} must be STRICTLY BELOW the Plan-01 \
         pre-collapse baseline {ANCHOR_SYNC_BASELINE} (= 1 + 3*(num_leaves-1)); the resident \
         control plane failed to drop the sync count"
    );

    // The drop from the anchor baseline is EXACTLY `2*(num_leaves-1)`: relative
    // to the anchor's per-split [split_on_device + 2 child scans], the resident lane (a) folds
    // BOTH per-split child-scan readbacks device→device into the frontier (removes
    // 2 per split) and (b) replaces the per-split `split_on_device` readback with the
    // per-iteration §8.3 pick export (count-neutral: one per iteration either way). Net removal =
    // 2 per split = `2*(num_leaves-1)`. Pinning the EXACT delta (not just `<`) proves the drop is
    // the real device→device fold, not a coincidental per-leaf proxy.
    let expected_drop = 2 * (NUM_LEAVES as u64 - 1);
    assert_eq!(
        ANCHOR_SYNC_BASELINE - syncs_3,
        expected_drop,
        "ODP3-07/ODS-02: the resident count must be EXACTLY {expected_drop} syncs below the \
         anchor baseline (2 per split: both child-scan readbacks folded device→device into the \
         resident frontier post-Plan-08; the split_on_device→pick-export swap is count-neutral); \
         got a drop of {}",
        ANCHOR_SYNC_BASELINE - syncs_3
    );

    // The RESIDENT-PERM partition arm (DEFAULT since spike093): its own EXACT sync
    // closed form adds one small child-range readback per split (`read_leaf`, the
    // split-point scalar the host row bookkeeping needs — the same crossing the
    // reference CUDADataPartition::SplitInner performs) and one per-grow tail perm
    // readback (the layout rebuild):
    //   [1 root scan + (L-1) picks] + (L-1) read_leaf + 1 tail = 2 + 2*(L-1) = 2*L.
    lgbm_compute::kernels::grow_driver::set_partition_resident_override(Some(true));
    let (syncs_rp, leaves_rp) = grow_resident(3);
    lgbm_compute::kernels::grow_driver::set_partition_resident_override(None);
    assert_eq!(
        leaves_rp, NUM_LEAVES,
        "resident-perm lane: corpus must grow the full {NUM_LEAVES} leaves (got {leaves_rp})"
    );
    let analytic_rp = 2 * NUM_LEAVES as u64;
    assert_eq!(
        syncs_rp, analytic_rp,
        "resident-perm lane: blocking-readback sync count {syncs_rp} must equal the analytic \
         closed form {analytic_rp} (= 2*num_leaves: root scan + per-iteration pick + per-split \
         child-range readback + per-grow tail perm readback; num_leaves={NUM_LEAVES})"
    );
}

/// SPEC-DRGL-06 (`--features rocm`): the DEFERRED-sync arm (`LGBM_GROW_DEFER_SYNC=1`) on the
/// resident-perm partition arm. The per-split `read_split` is FUSED into the pick's readback
/// (SPEC-DRGL-05), so the eager arm's `2*num_leaves` collapses to a LOWER closed form,
/// re-derived here from a fresh `bump_sync()` grep of the deferred loop
/// (`grow_tree_on_device_resident`'s `grow_defer_sync_enabled()` block):
///   - root scan (1): the shared setup's `scan_resident_and_argmax`, unchanged.
///   - per grow-loop iteration (num_leaves-1): ONE `bump_sync` — the batched
///     `client.read([ranges, pick_export])` that fuses the previous split's deferred
///     `read_split` with this split's pick. (The eager arm bumps TWICE here: pick + read_split.)
///   - grow tail (2): the LAST split's deferred `read_split` (1) + the per-grow perm readback (1).
///     (The eager arm's tail is just the perm readback = 1; the deferral moves the last
///     read_split here.)
///   => 1 + (num_leaves-1) + 2 = num_leaves + 2.
/// EXACT-equality asserted (never `<=`). The default (flag-OFF) `2*num_leaves` form is left
/// intact above — this is an ADDITIVE lane for the opt-in arm.
#[cfg(feature = "rocm")]
fn resident_defer_sync_lane() {
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::RocmBackend;
    unsafe {
        std::env::remove_var("LGBM_ROCM_HOST_PARTITION");
        std::env::remove_var("LGBM_SIBLING_COPACK");
        std::env::remove_var("LGBM_ONDEVICE_F64_FUSED");
    }
    let backend = RocmBackend::with_resident(true);
    let client = rocm_client();
    let grow_defer = |num_features: usize| -> (u64, i32) {
        let _ = on_device_sync_count_take();
        let (features, g, h) = tiny_corpus(num_features, NUM_DATA);
        // The deferral requires the resident-perm arm; force it ON alongside the flag.
        lgbm_compute::kernels::grow_driver::set_partition_resident_override(Some(true));
        lgbm_compute::kernels::grow_driver::set_grow_defer_sync_override(Some(true));
        let (tree, _layout) =
            grow_tree_on_device_driver(&backend, &client, &g, &h, &features, NUM_LEAVES, -1)
                .expect("deferred resident driver must grow the tiny corpus");
        lgbm_compute::kernels::grow_driver::set_grow_defer_sync_override(None);
        lgbm_compute::kernels::grow_driver::set_partition_resident_override(None);
        (on_device_sync_count_take(), tree.num_leaves)
    };
    let (syncs_3, leaves_3) = grow_defer(3);
    let (syncs_12, leaves_12) = grow_defer(12);
    assert_eq!(
        leaves_3, NUM_LEAVES,
        "defer lane: corpus must grow the full {NUM_LEAVES} leaves (got {leaves_3})"
    );
    assert_eq!(
        syncs_3, syncs_12,
        "defer lane: sync count must be num_features-independent (got {syncs_3} vs {syncs_12})"
    );
    let analytic_defer = NUM_LEAVES as u64 + 2;
    assert_eq!(
        syncs_3, analytic_defer,
        "defer lane: blocking-readback sync count {syncs_3} must equal the deferred closed form \
         {analytic_defer} (= num_leaves + 2: root scan + per-iteration fused pick/read_split + \
         tail [last read_split + perm]; num_leaves={NUM_LEAVES}); the eager arm is 2*num_leaves"
    );
    // The deferral must be STRICTLY below the eager resident-perm form (the whole point).
    assert!(
        syncs_3 < 2 * NUM_LEAVES as u64,
        "defer lane: {syncs_3} must be strictly below the eager 2*num_leaves = {}",
        2 * NUM_LEAVES as u64
    );
}
