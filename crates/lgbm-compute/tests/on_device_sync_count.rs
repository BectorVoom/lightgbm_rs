//! 26-01 (ODP3-07) — the on-device BLOCKING-READBACK sync counter reports an HONEST,
//! NON-ZERO, num_features-INDEPENDENT total, and is DISTINCT from the launch counter.
//!
//! Where `on_device_launch_count` proves the DISPATCH total collapses per-leaf (not
//! per-feature), this proves the companion SYNC total — the real device→host blocking
//! readbacks (scan / on-device partition / tree-split `right_leaf_index`) — is wired and
//! counts REAL syncs, not one-per-feature and not one-per-child (the co-packed sibling
//! scan is ONE sync). This is the counter Plans 02/03 drive DOWN by moving the argmax +
//! row permutation on device; Plan 05 asserts the aggregate drop against the baseline this
//! test documents.
//!
//! ## 26-02 status (M3/M7 seam wired; readback COUNT drop deferred to the resident reduce)
//! Plan 02 moved the per-feature best-split argmax, the cross-leaf best-leaf pick, and the
//! root grad/hess sum behind the `Backend` seam (`scan_resident_leaf_argmax` /
//! `scan_resident_siblings_argmax` / `best_leaf_reduce` / `root_grad_hess_sum`), so on the
//! GpuBackend arm the host argmax/best-leaf/root-fold are RETIRED and the driver keeps only
//! the winning ~8-int split (the host-side payload collapses to the winner). The per-leaf
//! blocking-readback COUNT is deliberately UNCHANGED this plan: the resident scan launcher
//! (`find_best_splits_batched_fused_f64_from_handle_on`) still reads the per-feature Vec back
//! before the reduce, so the scan is still exactly ONE sync. Genuinely eliminating that
//! readback (reduce-before-copy, the §8.2 resident device reduce that returns the single 8-int
//! export) is the real-hardware refinement Plan 05 confirms on real CUDA — it is not
//! runtime-verifiable on the local spoofed 8-CU APU. The analytic below therefore still holds.
//!
//! ## 26-03 status (M1/M6 resident row permutation wired; readback COUNT unchanged on the tested lanes)
//! Plan 03 made the row permutation RESIDENT for the whole grow (`ResidentDriverLeaf.rows` is now
//! a device index RANGE into a single `perm` buffer, the `cuda_data_indices_` mirror), partitioned
//! IN PLACE per split (`partition_resident_range`), retiring the per-leaf `rows.clone()`, the
//! cloned `parent_rows`, and the local→global `parent_rows` map-back Vec on the GpuBackend arm.
//! The per-leaf blocking-readback COUNT on the two lanes this test exercises is UNCHANGED:
//!   - The DEFAULT resident lane routes the partition on the HOST (`prefers_host_partition()` ⇒
//!     `partition_leaf_stable`), which issues NO device readback — it never bumped a partition
//!     sync, so making its rows resident removes nothing from the closed form (`1 + 2*(n-1)`).
//!   - On the on-device-partition arm (`LGBM_ROCM_HOST_PARTITION=0`, not the tested default) the
//!     resident permutation now stays on device (scattered in place) so the host no longer
//!     materializes the per-leaf row Vec — but the split_point still crosses back once, and a
//!     readback is a readback, so the COUNT is still one sync per split.
//! Genuinely dropping the count (the resident device scatter that keeps the permutation on device
//! AND fuses the single split_point export, no intermediate route readback) is the §9 resident
//! scatter kernel — a real-hardware refinement (Plan 05 / Kaggle), not authorable/verifiable on
//! the spoofed 8-CU APU, exactly as 26-02 deferred the reduce-before-copy readback. The analytic
//! below is therefore unchanged and still holds for the wiring as it stands (Plan01→02→03).
//!
//! ## 26-04 status (M4/M5, ODP3-05 — async copy / stream overlap: count is AT the deferred-sync floor)
//! Plan 04 resolved M4/M5 by establishing, against the installed cubecl-runtime-0.10.0 (NOT
//! from memory — see `docs/on-device-streams-M4-M5.md`), that:
//!   - async BATCHED device→host copy IS expressible (`ComputeClient::read/read_async(Vec<Handle>)`,
//!     client.rs:109/131; `read_one_unchecked(h) ≡ read(vec![h])`, client.rs:145) and is ALREADY
//!     the production idiom — the co-packed sibling scan drains BOTH children in ONE readback
//!     (the deferred-sync batching of the two independent sibling reads, spike-024), and every
//!     other per-split readback consolidates one handle.
//!   - TRUE per-operation multi-stream overlap (§8 streams 0/1, §9 streams 1–3) is NOT cleanly
//!     expressible in cubecl 0.10 (`set_stream` pub unsafe/CubeCL-internal, StreamId thread-derived,
//!     runtime auto-merges) AND is architecturally inapplicable (the subtraction trick makes larger
//!     = parent − smaller a strict dependency, not two independent builds).
//! CONSEQUENCE FOR THIS TEST: the blocking-readback COUNT is UNCHANGED — it is already at the
//! single-drain deferred-sync FLOOR. Deferred-sync batching cannot drop it further (the two sibling
//! reads are already ONE co-packed drain; the remaining per-split reads — `split_on_device`'s
//! `right_leaf_index` and the partition split_point — are strictly data-dependent and stay blocking).
//! True async streams would record overlap WITHOUT changing the count; since they are not expressible,
//! the count is simply held. Plan 04 encodes the boundary in code (`Backend::supports_async_device_copy`
//! / `supports_multi_stream_overlap` / `read_batched`) and cpu-pins the single-drain seam
//! (`grow_driver::tests::read_batched_single_drain_matches_per_handle_reads`); no `bump_sync` site
//! changed, so both closed forms below still hold exactly.
//!
//! ## 26-05 status (ODP3-07 FINALIZED — the honest sync-count regression + the drop chain)
//! This plan finalizes the ODP3-07 regression now that the resident control plane is complete.
//! The honest, machine-checked claim is:
//!   - **cpu anchor lane (the RUNNABLE reference):** the blocking-readback count is the
//!     DOCUMENTED PRE-COLLAPSE BASELINE `ANCHOR_SYNC_BASELINE = 1 + 3*(num_leaves-1)` (root
//!     scan + per split [`split_on_device` + 2 SEPARATE child scans]). This lane does NOT
//!     drop — it IS the reference the drop is measured against (`CpuBackend` is the anchor
//!     arm; it never co-packs). Asserting a drop HERE would fake a drop the tested lane does
//!     not exhibit — deliberately NOT done (parity-discipline).
//!   - **rocm resident lane (cfg-gated, real-hardware):** the resident control plane's count
//!     is the EXACT closed form `1 + 2*(num_leaves-1)` (root scan + per split [ONE co-packed
//!     siblings scan + `split_on_device`]), which is STRICTLY BELOW the pre-collapse baseline
//!     `1 + 3*(num_leaves-1)` for every `num_leaves > 1`. The strict-below assertion lives on
//!     THIS lane (`resident_sync_lane`) — where the drop is genuinely exhibited.
//!
//! ### The drop chain (Plan01 baseline → M3/M7 → M1/M6 → M4/M5), honestly attributed
//! The single machine-checkable count drop from `1 + 3*(n-1)` to `1 + 2*(n-1)` is the
//! CO-PACKED SIBLING SCAN collapsing the two separate per-split child-scan readbacks into ONE
//! drain (spike-024, default-ON). The M2 (grad/hess-once), M3/M7 (on-device argmax / best-leaf
//! / root-sum), M1/M6 (resident row permutation) and M4/M5 (async-copy / stream boundary) work
//! of Plans 01-04 collapsed the host-side PAYLOAD and the host CONTROL LOOP and hit the
//! deferred-sync single-drain FLOOR — each HELD the count (they removed no additional
//! device→host readback on the tested lanes: the resident scan launcher still drains once
//! before the reduce, the DEFAULT lane routes the partition on the host = ZERO partition
//! syncs, and the remaining per-split reads are strict data-dependencies). Genuinely removing
//! MORE readbacks (reduce-before-copy, a resident device scatter) needs NEW kernels the phase
//! prohibits and cannot author/verify on the spoofed 8-CU APU; those are wall-clock
//! refinements confirmed on real CUDA (the Plan-06 Kaggle A/B), NOT a change to this count.
//! So ODP3-07 is discharged as: the resident count `1 + 2*(n-1)` is strictly below the
//! pre-collapse baseline `1 + 3*(n-1)`, num_features-independent, real-dispatch counted — and
//! the counter is NOT weakened, falsified, or faked on the lane that does not exhibit a drop.
//!
//! ## 31-08 status (ODS-02, D-081-1 — the per-split scan readback is RETIRED; resident lane `1 + (n-1)`)
//! Plan 08 (the checkpoint-approved Option A full fix for spike-081 D-081-1, the surviving
//! per-split sync floor the 4 failed on-device A/Bs kept hitting) folds every scanned child's
//! WINNER device→device into the resident frontier — carrying `gain`/`left_output`/`right_output`
//! through the frontier SoA + §8.3 export — so all 3 per-split scan `bump_sync` sites
//! (`scan_resident_*_into_frontier` on the default separate-scan, default-ON co-pack, AND
//! `LGBM_ONDEVICE_F64_FUSED=1` arms) are GENUINELY retired (0 remaining per-split scan crossings).
//! The resident lane's closed form therefore DROPS from `1 + 2*(num_leaves-1)` to
//! `1 + (num_leaves-1)` = `num_leaves` (root scan + the per-iteration §8.3 pick export — the ONE
//! by-design host crossing, unchanged). This is INDEPENDENTLY RE-DERIVED here from a fresh
//! `bump_sync()` grep of `grow_tree_on_device_resident` (see `resident_sync_lane`'s doc), and it
//! agrees with Plan 08's own real-HIP trace (8 for the tiny 8-leaf corpus, on all 3 arms, at 3
//! and 12 features). This SUPERSEDES the 26-02/26-03/26-04 notes below that held the count at
//! `1 + 2*(n-1)` "until a real-hardware reduce-before-copy refinement" — Plan 08 IS that
//! refinement, now landed and machine-checked. The cpu anchor lane (`ANCHOR_SYNC_BASELINE =
//! 1 + 3*(n-1)`) is a DIFFERENT function (`grow_tree_on_device_driver_with_cfg`) Plan 08 did NOT
//! touch, so it is unchanged and still the reference the drop is measured against.
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
//!   EXACT analytic sync closed form — POST-Plan-08 `1 + (num_leaves-1)` = `num_leaves`: root
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

/// DOCUMENTED PRE-COLLAPSE BASELINE (26-01, for Plan 05's regression assertion): the cpu
/// anchor lane's blocking-readback count for the fully-grown tiny corpus. Root scan (1) +
/// per split [split_on_device (1) + 2 child scans (2)] = `1 + 3*(num_leaves-1)`. This is the
/// pre-M1/M2/M3-collapse figure Plans 02/03 will drive down (their on-device argmax + row
/// permutation remove host scan/partition syncs); Plan 05 asserts the drop against it.
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

/// ODP3-07 (cpu anchor lane): the blocking-readback sync count is NON-ZERO, INDEPENDENT of
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

    // The documented pre-collapse baseline (Plan 05 regression anchor): root scan +
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
}

/// `--features rocm` lane: drive the resident fast arm (`RocmBackend::with_resident(true)`)
/// and assert the EXACT analytic blocking-readback sync count under the DEFAULT host
/// partition route + co-pack ON, then re-check num_features-independence on the real
/// resident launchers. The env gate is already set by the caller.
///
/// Analytic sync closed form — POST-Plan-31-08 (ODS-02, D-081-1): `1 + (num_leaves-1)` =
/// `num_leaves`. Re-derived here from a fresh `bump_sync()` grep of `grow_driver.rs`
/// (`grow_tree_on_device_resident`), independently cross-checking Plan 08's own Task-2 trace
/// (`31-08-SUMMARY.md`) rather than restating it:
///   - root scan (1): `scan_resident_and_argmax` bumps ONE readback (grow_driver.rs ~:1870);
///     the f64-fused root escape hatch bumps its fused build+scan readback instead (~:2277) —
///     still exactly 1.
///   - per-iteration §8.3 pick export (num_leaves-1): `frontier_pick_best_leaf_device` reads
///     back the ~10-cell winner ONCE per grow-loop iteration (grow_driver.rs ~:2443).
///   - per-split scan readbacks: ZERO. Plan 08 (checkpoint-approved Option A) folds every
///     scanned child's winner device→device into the resident frontier
///     (`scan_resident_leaf_into_frontier` / `scan_resident_siblings_into_frontier` /
///     `build_fix_scan_resident_into_frontier`) — NO `bump_sync`, on ALL 3 arms (co-pack
///     default-ON, `LGBM_SIBLING_COPACK=0`, `LGBM_ONDEVICE_F64_FUSED=1`).
///   - partition readbacks: ZERO on the DEFAULT host partition route (`prefers_host_partition`);
///     the on-device partition arm's `bump_sync` (~:1993) is not on this lane.
/// Builds / subtracts / uploads / the scheduled tree-split (R3, no-readback) never bump. The
/// count is num_features-independent AND identical across all 3 arms (Plan 08 traced 8 for
/// each on real HIP; this re-derivation agrees).
///
/// SUPERSEDES the pre-Plan-08 note (the old `1 + 2*(num_leaves-1)` form, which counted a
/// per-split co-packed siblings scan readback): Plan 03's frontier re-sourcing (D-31-A) was a
/// correctness/architecture fix that left the sync count untouched by design; Plan 08's
/// zero-readback reduce-into-frontier fold is what actually retired the per-split scan
/// readback, dropping the resident lane from `1 + 2*(n-1)` to `1 + (n-1)`.
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

    // EXACT analytic sync closed form — POST-Plan-31-08 (host partition route, co-pack ON):
    // root scan (1) + per-iteration §8.3 pick export (num_leaves-1) = 1 + (num_leaves-1) =
    // num_leaves. The per-split scan readbacks are RETIRED (Plan 08 folds each scanned child's
    // winner device→device into the resident frontier — no `bump_sync`).
    let analytic = 1 + (NUM_LEAVES as u64 - 1);
    assert_eq!(
        syncs_3, analytic,
        "resident lane: blocking-readback sync count {syncs_3} must equal the analytic closed \
         form {analytic} (= 1 + (num_leaves-1): root scan + per-iteration §8.3 pick export; \
         per-split scans fold device→device post-Plan-08, host partition route, co-pack ON; \
         num_leaves={NUM_LEAVES})"
    );

    // ODP3-07 REGRESSION (the honest win, real-dispatch counted): the RESIDENT control
    // plane's per-leaf blocking-readback count is STRICTLY BELOW the Plan-01 pre-collapse
    // baseline `1 + 3*(num_leaves-1)`. This is the machine-checked drop — exhibited on THIS
    // lane (the resident arm genuinely co-packs), NOT faked on the cpu anchor lane (which IS
    // the baseline reference). It counts REAL dispatches: the co-packed sibling scan is ONE
    // readback (not two), the host partition route is ZERO, the resident permutation adds none.
    assert!(
        syncs_3 < ANCHOR_SYNC_BASELINE,
        "ODP3-07: resident blocking-readback count {syncs_3} must be STRICTLY BELOW the Plan-01 \
         pre-collapse baseline {ANCHOR_SYNC_BASELINE} (= 1 + 3*(num_leaves-1)); the resident \
         control plane failed to drop the sync count"
    );

    // POST-Plan-31-08 the drop from the anchor baseline is EXACTLY `2*(num_leaves-1)`: relative
    // to the anchor's per-split [split_on_device + 2 child scans], the resident lane (a) folds
    // BOTH per-split child-scan readbacks device→device into the frontier (Plan 07/08 — removes
    // 2 per split) and (b) replaces the per-split `split_on_device` readback with the
    // per-iteration §8.3 pick export (count-neutral: one per iteration either way). Net removal =
    // 2 per split = `2*(num_leaves-1)`. Pinning the EXACT delta (not just `<`) proves the drop is
    // the real device→device fold, not a per-leaf proxy (the Phase-24 trap). (Pre-Plan-08 this
    // delta was `num_leaves-1`, attributed solely to the co-pack collapse; Plan 08's zero-readback
    // per-split fold doubled it.)
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
}
