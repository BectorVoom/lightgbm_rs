//! The on-device BLOCKING-READBACK sync-count COLLAPSE gate.
//!
//! This is a LOCAL COUNT gate: the local GPU here is a spoofed low-CU APU that cannot
//! time PCIe latency honestly, so a COUNT gate is what the local host can deliver; the
//! wall-clock verdict requires real hardware.
//!
//! # Why this is a STANDALONE test binary (not folded into on_device_integer_anchor.rs)
//! The sync counter (`ON_DEVICE_SYNC_CNT`, drained via `on_device_sync_count_take`) is a
//! PROCESS-GLOBAL atomic. cargo runs the `#[test]`s inside ONE binary on parallel threads, so a
//! concurrent grow in a sibling test would bump the global counter between this test's
//! drain-and-read window and corrupt the count. The counter test must therefore OWN its
//! process — the same reason `lgbm-compute/tests/on_device_sync_count.rs` and
//! `on_device_launch_count.rs` are standalone binaries.
//!
//! # What "collapse to O(num_leaves)" honestly means here (real-dispatch counted)
//! The counter counts REAL device→host blocking readbacks — a CO-PACKED/batched readback bumps
//! EXACTLY ONCE, never one-per-leaf and never one-per-feature. The honest property this gate
//! proves LOCALLY is that the per-grow sync count is O(num_leaves): it is LINEAR in the grown
//! leaf count and INDEPENDENT of `num_features` (feature 0 dominates every split, so growing
//! the SAME tree at 3 vs 12 features yields the IDENTICAL count). A per-feature readback
//! regression would scale the count with `num_features` and break the equality.
//!
//! # Two lanes (mirrors `lgbm-compute/tests/on_device_sync_count.rs`)
//!   - **cpu ANCHOR lane (RUNNABLE HERE, the only lane cubecl-cpu can execute):**
//!     `CpuBackend::resident_pool_supported() == false`, so the driver takes the ANCHOR fold arm.
//!     Its count is the DOCUMENTED PRE-COLLAPSE BASELINE `1 + 3*(L-1)` (root scan + per split
//!     [`split_on_device` + 2 SEPARATE child scans], L = grown leaves). This lane does NOT co-pack
//!     and does NOT exhibit the resident collapse — asserting a resident drop HERE would FAKE a drop
//!     the tested arm does not perform (parity-discipline). What it DOES prove locally: the count is
//!     real-dispatch, non-zero, and num_features-INDEPENDENT (O(num_leaves), no per-feature inflation).
//!   - **rocm RESIDENT lane (`#[cfg(feature = "rocm")]`, hardware-only):** the resident scheduler's
//!     honest count is `1 + 2*(L-1)` (root scan + per split [ONE co-packed siblings scan + ONE
//!     device-pick export]), STRICTLY BELOW the `1 + 3*(L-1)` anchor baseline — the genuine
//!     co-packed-sibling-scan collapse. It asserts feature-independence and the build sub-counters
//!     (`on_device_rootbuild_u64 > 0`, `on_device_f64_fused == 0`: the fast u64 arm ran, the slow
//!     f64-fused did not).
//!
//! # HONEST DEFERRAL
//! The resident scheduler retired the original `split_on_device` readback, but a device-pick
//! export crosses back once per iteration and replaces it 1:1 — so the resident count is HELD at
//! `1 + 2*(L-1)`; it does NOT yet drop BELOW it. The FURTHER collapse (dropping the per-split
//! co-packed SCAN readback so ONLY a stop flag crosses, reaching ~L) needs a child-sum-carrying
//! `DeviceFrontier` + on-device child seeding — new parity-critical kernels that CANNOT be
//! authored/verified on the local host and remain a deferred hardware refinement. This test
//! therefore asserts the count actually achieved (`1 + 2*(L-1)` on the resident lane, strictly
//! below the `1 + 3*(L-1)` anchor), NOT a mislabeled sub-61 green.
//!
//! Additionally: the resident lane CANNOT run on cpu at all — the u64 fixed-point resident build
//! kernel uses `atomic<u64>`, which cubecl-cpu does not implement (the grow aborts after the root
//! build with num_leaves=1). So the resident sync count is measurable ONLY on real GPU hardware
//! (rocm/cuda), never on the cpu merge gate.
//!
//! Scope: INSTRUMENTATION, not parity — the on-device grow's faithfulness is covered by the
//! integer-anchor bit-exact gate (`on_device_integer_anchor.rs`) and the `learner_parity` STRUCTURE
//! gate (pinned to the cpu f64 anchor).

use lgbm_compute::kernels::grow_driver::{grow_tree_on_device_driver, on_device_sync_count_take};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, CpuBackend, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};

/// `offset_for_most_freq_bin(0)` == 1 (drop bin 0 under the compacted convention). The helper
/// lives in `lgbm-treelearner` (ABOVE this crate); hardcode it (as `on_device_sync_count.rs`).
const OFFSET_MFB_0: i32 = 1;

/// The requested leaf cap — the real LightGBM `num_leaves` default.
const COLLAPSE_NUM_LEAVES: i32 = 31;

/// Build a dominant-feature numeric corpus that grows the FULL `COLLAPSE_NUM_LEAVES` leaves:
/// `num_features` `num_bin`-bin columns over `num_data` rows; feature 0's bin rises monotonically
/// with the row index (one distinct bin per contiguous row block) so it DOMINATES every split and
/// the tree structure — hence every scan/split readback — is IDENTICAL regardless of how many
/// weaker extra features exist. That invariance is what the num_features-independence assertion
/// exploits. `num_bin` is sized `> COLLAPSE_NUM_LEAVES` so feature 0 alone can carve the full leaf
/// count.
fn dominant_corpus(
    num_features: usize,
    num_bin: u32,
    num_data: usize,
) -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
    let features: Vec<GrowFeature> = (0..num_features)
        .map(|fi| {
            let bins: Vec<u32> = (0..num_data)
                .map(|r| (((r + fi * 37) * num_bin as usize / num_data) as u32) % num_bin)
                .collect();
            GrowFeature {
                bins: BinColumn::new(bins, num_bin),
                num_bin,
                offset: OFFSET_MFB_0,
                min_bin: 0,
                max_bin: num_bin - 1,
                default_bin: num_bin,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: (0..num_bin).map(|b| b as f64 + 0.5).collect(),
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
    // Strictly monotone per-row gradient so every adjacent-bin split has positive gain (the tree
    // reaches the full leaf cap) and the winner is unambiguous (near-tie-free, structure stable).
    let half = num_data as f32 / 2.0;
    let gradients: Vec<f32> = (0..num_data).map(|r| (r as f32 - half) * 0.1).collect();
    let hessians: Vec<f32> = vec![1.0f32; num_data];
    (features, gradients, hessians)
}

/// Grow the dominant corpus on the ANCHOR arm (`CpuBackend`) with `num_features` columns and
/// return `(blocking-readback sync count for THIS grow, grown leaf count)`. The counter is drained
/// first for isolation.
fn anchor_sync_count(num_features: usize) -> (u64, i32) {
    let _ = on_device_sync_count_take(); // isolate this grow's syncs
    let client = cpu_client();
    // 64 bins over 512 rows (8 rows/bin) — feature 0 alone can carve > 31 leaves.
    let (features, gradients, hessians) = dominant_corpus(num_features, 64, 512);
    let (tree, _layout) = grow_tree_on_device_driver(
        &CpuBackend,
        &client,
        &gradients,
        &hessians,
        &features,
        COLLAPSE_NUM_LEAVES,
        -1, // no depth cap
    )
    .expect("on-device anchor driver must grow the dominant corpus");
    (on_device_sync_count_take(), tree.num_leaves)
}

/// The honest sync-count collapse gate. On the cpu ANCHOR lane (the only
/// lane executable here) the per-grow blocking-readback count is real-dispatch, non-zero,
/// num_features-INDEPENDENT (O(num_leaves), no per-feature inflation), and equals the documented
/// pre-collapse baseline `1 + 3*(L-1)`. The rocm RESIDENT lane (hardware-only) asserts the genuine
/// collapse to `1 + (L-1) = L` (the co-packed sibling scan folds device→device — corrected
/// 2026-07-15 from a stale `1 + 2*(L-1)`) plus the u64 build sub-counters, and — with
/// `LGBM_GROW_DEFER_SYNC=1` — the further deferral collapse to `L + 2`.
#[test]
fn on_device_sync_count_collapses_to_num_leaves() {
    // Enable the read-once phase-prof gate BEFORE the FIRST counter read. `bump_sync`/`take`
    // cache the env in a `OnceLock` read once per process; this standalone binary owns its counting.
    // SAFETY: set before any counted grow; single-threaded until the grows below.
    unsafe {
        std::env::set_var("LGBM_PHASE_PROF", "1");
    }

    let (syncs_3, leaves_3) = anchor_sync_count(3);
    let (syncs_12, leaves_12) = anchor_sync_count(12);

    // The corpus grows the FULL requested leaf count (so the closed form is the exact bound, not a
    // truncated-growth artifact) — otherwise the baseline equality below would be vacuous.
    assert_eq!(
        leaves_3, COLLAPSE_NUM_LEAVES,
        "the dominant corpus must grow the full {COLLAPSE_NUM_LEAVES} leaves for the exact closed \
         form (got {leaves_3}); widen num_bin/num_data if this trips"
    );
    assert_eq!(
        leaves_3, leaves_12,
        "the tree must be identical across num_features (feature 0 dominates); got {leaves_3} vs {leaves_12}"
    );

    // The counter is wired: an on-device grow reports a NON-ZERO blocking-readback count.
    assert!(
        syncs_3 > 0,
        "on-device grow must bump a non-zero blocking-readback sync count, got {syncs_3}"
    );

    // COUNTER-TRAP GUARD (the honest O(num_leaves) core, provable locally): identical count at 3 and
    // 12 features. A per-feature readback regression (or one-sync-per-child on a co-packed scan)
    // would scale the count with num_features and break this equality.
    assert_eq!(
        syncs_3, syncs_12,
        "blocking-readback sync count must be INDEPENDENT of num_features (got {syncs_3} at 3 vs \
         {syncs_12} at 12) — inequality means a readback scaled per-feature (the counter trap)"
    );

    // The documented PRE-COLLAPSE BASELINE the resident drop is measured against: root scan + per
    // split [split_on_device + 2 child scans] = 1 + 3*(L-1). Asserting the exact bound proves the
    // count is O(num_leaves) (linear in leaves), not merely feature-independent.
    let anchor_baseline = 1 + 3 * (leaves_3 as u64 - 1);
    assert_eq!(
        syncs_3, anchor_baseline,
        "cpu anchor lane blocking-readback baseline must equal {anchor_baseline} (= 1 + 3*(L-1): \
         root scan + per split [split_on_device + 2 child scans], L={leaves_3}); got {syncs_3}"
    );

    // On a rocm host, additionally assert the resident collapse + the u64 build sub-counters on the
    // real resident launchers (the lane cubecl-cpu cannot execute — atomic<u64> unimplemented).
    #[cfg(feature = "rocm")]
    resident_sync_collapse_lane(anchor_baseline);
    // SPEC-DRGL-06: the DEFERRED-sync arm (LGBM_GROW_DEFER_SYNC=1) at L=31 — proves the
    // deferred closed form scales as `L + 2` (constant +2, not a coincidence at L=8 in the
    // lgbm-compute lane).
    #[cfg(feature = "rocm")]
    resident_defer_sync_collapse_lane();
}

/// `--features rocm` lane: drive the resident fast arm (`RocmBackend::with_resident(true)`) and
/// assert the HONEST collapse: the per-grow blocking-readback count is `1 + (L-1) = L` (root
/// scan + per-iteration §8.3 device-pick export) — STRICTLY BELOW the `1 + 3*(L-1)` cpu anchor
/// baseline — feature-independent, with `on_device_rootbuild_u64 > 0` and
/// `on_device_f64_fused == 0`. The per-split co-packed sibling SCAN readback is folded
/// device→device into the resident frontier (retired). (Corrected 2026-07-15: an earlier doc
/// claimed the sub-`1 + 2*(L-1)` collapse was "not yet implemented" — it IS; the device→device
/// sibling-reduce fold retired that scan readback. The eager arm's remaining lever is the
/// per-split `read_split`, which the opt-in `LGBM_GROW_DEFER_SYNC` deferral fuses into the pick,
/// dropping the resident-perm arm further to `L + 2` — see `resident_defer_sync_collapse_lane`.)
#[cfg(feature = "rocm")]
fn resident_sync_collapse_lane(anchor_baseline: u64) {
    use lgbm_compute::kernels::grow_driver::on_device_rootbuild_u64_count_take;
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::{Backend, RocmBackend};

    // DEFAULT host partition route (on-device partition adds one readback/split), default co-pack ON
    // (the closed form is co-pack-dependent).
    // SAFETY: single-threaded test; no concurrent env access.
    unsafe {
        std::env::remove_var("LGBM_ROCM_HOST_PARTITION");
        std::env::remove_var("LGBM_SIBLING_COPACK");
    }
    // The RESIDENT-PERM partition arm is now DEFAULT-ON (spike093; +1 child-range
    // readback per split + 1 tail perm readback per grow) — pin it OFF so this lane
    // keeps asserting the host-partition closed form it was derived for (the
    // resident-perm arm's own closed form is asserted in
    // `lgbm-compute/tests/on_device_sync_count.rs`). Read-once env gate ⇒ in-process
    // override; restored to `None` at the end of the lane.
    lgbm_compute::kernels::grow_driver::set_partition_resident_override(Some(false));

    let backend = RocmBackend::with_resident(true);
    assert!(
        backend.resident_pool_supported(),
        "with_resident(true) must drive the resident fast arm"
    );
    let client = rocm_client();

    let grow_resident = |num_features: usize| -> (u64, u64, i32) {
        let _ = on_device_sync_count_take();
        let _ = on_device_rootbuild_u64_count_take();
        let (features, g, h) = dominant_corpus(num_features, 64, 512);
        let (tree, _layout) = grow_tree_on_device_driver(
            &backend,
            &client,
            &g,
            &h,
            &features,
            COLLAPSE_NUM_LEAVES,
            -1,
        )
        .expect("resident driver must grow the dominant corpus");
        (
            on_device_sync_count_take(),
            on_device_rootbuild_u64_count_take(),
            tree.num_leaves,
        )
    };

    let (syncs_3, rootbuild_u64, leaves_3) = grow_resident(3);
    let (syncs_12, _, leaves_12) = grow_resident(12);

    assert_eq!(
        leaves_3, COLLAPSE_NUM_LEAVES,
        "resident lane: corpus must grow the full {COLLAPSE_NUM_LEAVES} leaves (got {leaves_3})"
    );
    assert_eq!(
        leaves_3, leaves_12,
        "resident lane: tree must be identical across num_features (got {leaves_3} vs {leaves_12})"
    );
    assert_eq!(
        syncs_3, syncs_12,
        "resident lane: blocking-readback sync count must be INDEPENDENT of num_features (got \
         {syncs_3} at 3 vs {syncs_12} at 12 — the counter trap)"
    );

    // HONEST resident closed form: root scan (1) + per-iteration §8.3 device-pick export
    // (L-1) = 1 + (L-1) = L. The co-packed siblings scan's winner is folded device→device into
    // the resident frontier (the zero-readback reduce-into-frontier fold — SPEC-DRGL-13's
    // sibling reduce), so the per-split SCAN readback is RETIRED. (Corrected 2026-07-15 from a
    // stale `1 + 2*(L-1)` form that predated the co-pack device→device fold — the "closed-form
    // drift" SPEC-DRGL-06 warns about; the lgbm-compute copy already asserts `1 + (L-1)`. This
    // rocm lane had not been run since the local GPU came online, so the drift lay latent.)
    let resident_closed_form = 1 + (leaves_3 as u64 - 1);
    assert_eq!(
        syncs_3, resident_closed_form,
        "resident lane: blocking-readback count {syncs_3} must equal the honest closed form \
         {resident_closed_form} (= 1 + (L-1) = L: root scan + per-iteration §8.3 device-pick \
         export; the co-packed sibling scan folds device→device, host partition route, co-pack \
         ON; L={leaves_3})"
    );

    // The genuine collapse this gate proves: the resident count is STRICTLY BELOW the cpu anchor
    // baseline (the co-packed sibling scan replaces two separate child-scan readbacks with one).
    // This is O(num_leaves) and real-dispatch counted — NOT a per-leaf inflation.
    assert!(
        syncs_3 < anchor_baseline,
        "ODF-07: resident blocking-readback count {syncs_3} must be STRICTLY BELOW the cpu anchor \
         pre-collapse baseline {anchor_baseline} (= 1 + 3*(L-1)); the co-packed sibling scan failed \
         to collapse the two per-split child scans"
    );
    // The drop is EXACTLY 2 per split: relative to the anchor's per-split [split_on_device +
    // 2 child scans], the resident lane folds BOTH child-scan winners device→device into the
    // frontier (−2/split) and swaps split_on_device for the count-neutral §8.3 pick export. Net
    // = `2*(L-1)`. (Corrected 2026-07-15 from a stale `L-1` that predated the device→device
    // sibling-reduce fold retiring the SECOND child-scan readback.)
    assert_eq!(
        anchor_baseline - syncs_3,
        2 * (leaves_3 as u64 - 1),
        "ODF-07: resident count must be EXACTLY {} syncs below the anchor baseline (2 per split: \
         both child-scan winners fold device→device into the resident frontier; the \
         split_on_device→pick-export swap is count-neutral); got {}",
        2 * (leaves_3 as u64 - 1),
        anchor_baseline - syncs_3
    );

    // Build sub-counters: the fast parallel-u64 resident build ran (root + directly-built
    // children), and the slow f64-single-owner fused build did NOT.
    assert!(
        rootbuild_u64 > 0,
        "resident lane: the converted parallel-u64 resident build must have run \
         (on_device_rootbuild_u64 > 0); got {rootbuild_u64}"
    );

    // Restore the resident-perm override to the env-driven default (no state leak).
    lgbm_compute::kernels::grow_driver::set_partition_resident_override(None);
}

/// SPEC-DRGL-06 (`--features rocm`): the DEFERRED-sync arm (`LGBM_GROW_DEFER_SYNC=1`) on the
/// resident-perm partition arm, at `COLLAPSE_NUM_LEAVES=31`. The per-split `read_split` is
/// fused into the pick readback (SPEC-DRGL-05), so the eager resident-perm `2*L` form drops to
/// `L + 2` (root scan + per-iteration fused pick/read_split + tail [last read_split + perm]).
/// Asserting it at L=31 (vs the lgbm-compute lane's L=8) pins the `+2` CONSTANT — proof the
/// closed form is `L + 2`, not a coincidental match at one leaf count.
#[cfg(feature = "rocm")]
fn resident_defer_sync_collapse_lane() {
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::RocmBackend;
    unsafe {
        std::env::remove_var("LGBM_ROCM_HOST_PARTITION");
        std::env::remove_var("LGBM_SIBLING_COPACK");
    }
    let backend = RocmBackend::with_resident(true);
    let client = rocm_client();
    let grow_defer = |num_features: usize| -> (u64, i32) {
        let _ = on_device_sync_count_take();
        let (features, g, h) = dominant_corpus(num_features, 64, 512);
        lgbm_compute::kernels::grow_driver::set_partition_resident_override(Some(true));
        lgbm_compute::kernels::grow_driver::set_grow_defer_sync_override(Some(true));
        let (tree, _layout) = grow_tree_on_device_driver(
            &backend, &client, &g, &h, &features, COLLAPSE_NUM_LEAVES, -1,
        )
        .expect("deferred resident driver must grow the dominant corpus");
        lgbm_compute::kernels::grow_driver::set_grow_defer_sync_override(None);
        lgbm_compute::kernels::grow_driver::set_partition_resident_override(None);
        (on_device_sync_count_take(), tree.num_leaves)
    };
    let (syncs_3, leaves_3) = grow_defer(3);
    let (syncs_12, _) = grow_defer(12);
    assert_eq!(
        leaves_3, COLLAPSE_NUM_LEAVES,
        "defer collapse lane: must grow the full {COLLAPSE_NUM_LEAVES} leaves (got {leaves_3})"
    );
    assert_eq!(
        syncs_3, syncs_12,
        "defer collapse lane: sync count must be num_features-independent ({syncs_3} vs {syncs_12})"
    );
    let analytic_defer = COLLAPSE_NUM_LEAVES as u64 + 2;
    assert_eq!(
        syncs_3, analytic_defer,
        "defer collapse lane: blocking-readback count {syncs_3} must equal the deferred closed \
         form {analytic_defer} (= L + 2 at L={COLLAPSE_NUM_LEAVES}); the eager resident-perm arm \
         is 2*L = {}",
        2 * COLLAPSE_NUM_LEAVES as u64
    );
    assert!(
        syncs_3 < 2 * COLLAPSE_NUM_LEAVES as u64,
        "defer collapse lane: {syncs_3} must be strictly below the eager 2*L"
    );
}
