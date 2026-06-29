//! Structural self-tests for the SoA pre-allocated device split-record
//! (Phase 14 Plan 04, ODL-02 / D-05 / D-06 / D-08).
//!
//! These tests pin the Phase-14 contract: the record allocates one device buffer
//! per field exactly once (no per-split / per-copy device alloc), the host-staged
//! slot-copy deep-copies every field exactly and leaves other slots untouched, and
//! the reserved categorical slabs are sized off `MAX_CAT_PER_SPLIT` (Open Q3).
//! No GPU is required — the cpu anchor client (`cpu_client`) drives allocation.

use lgbm_compute::kernels::split_info::{
    DeviceSplitInfo, SplitScalars, MAX_CAT_PER_SPLIT, NUM_FIELD_BUFFERS,
};
use lgbm_compute::runtime::{cpu_client, ActiveRuntime};

/// A distinct, fully-populated scalar record keyed off `seed` so each slot is
/// unambiguously identifiable when asserting copy/untouched semantics.
fn distinct_scalars(seed: i64) -> SplitScalars {
    SplitScalars {
        is_valid: seed % 2 == 0,
        leaf_index: seed as i32,
        gain: seed as f64 * 1.5,
        inner_feature_index: (seed as i32) + 7,
        threshold: (seed as u32).wrapping_mul(13),
        default_left: seed % 3 == 0,
        left_sum_gradients: seed as f64 + 0.25,
        left_sum_hessians: seed as f64 + 0.5,
        left_sum_gh_quant: seed * 1000,
        left_count: (seed as i32) * 2,
        left_gain: seed as f64 - 1.0,
        left_value: seed as f64 / 3.0,
        right_sum_gradients: seed as f64 + 100.0,
        right_sum_hessians: seed as f64 + 200.0,
        right_sum_gh_quant: seed * -777,
        right_count: (seed as i32) * 3,
        right_gain: seed as f64 + 9.0,
        right_value: seed as f64 / 7.0,
        num_cat_threshold: 0,
    }
}

#[test]
fn new_allocates_one_buffer_per_field_exactly_once() {
    let client = cpu_client();
    let split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 8).unwrap();
    // 18 numeric + num_cat_threshold + 2 reserved categorical slabs = 21 buffers,
    // allocated entirely in `new` (no per-slot / per-record alloc anywhere else).
    assert_eq!(split.device_allocations(), NUM_FIELD_BUFFERS);
    assert_eq!(split.num_leaf_slots(), 8);
}

#[test]
fn allocation_count_unchanged_by_copy_and_writes() {
    // The "allocated exactly once" invariant (D-08): exercising the copy/index/set
    // paths must NOT allocate any new device buffer.
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 4).unwrap();
    split.set_scalars(0, &distinct_scalars(11)).unwrap();
    split.set_cat_thresholds(0, &[1, 2, 3], &[10, 20, 30]).unwrap();
    for _ in 0..50 {
        split.copy_slot(0, 1).unwrap();
        split.copy_slot(1, 2).unwrap();
        let _ = split.scalars(2);
        let _ = split.cat_threshold(2);
    }
    assert_eq!(
        split.device_allocations(),
        NUM_FIELD_BUFFERS,
        "no allocation may happen on the copy/index/set paths"
    );
}

#[test]
fn set_and_read_scalars_roundtrip() {
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 3).unwrap();
    let s = distinct_scalars(42);
    split.set_scalars(1, &s).unwrap();
    assert_eq!(split.scalars(1), s);
}

#[test]
fn copy_slot_deep_copies_every_field_and_leaves_others_untouched() {
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 4).unwrap();

    // Write distinct values into every slot so an accidental cross-slot write is
    // detectable.
    for slot in 0..4 {
        let mut s = distinct_scalars(slot as i64 + 1);
        s.num_cat_threshold = 0;
        split.set_scalars(slot, &s).unwrap();
    }
    // Give the source slot (0) a categorical payload; the dest slot (2) starts empty.
    split.set_cat_thresholds(0, &[5, 6], &[50, 60]).unwrap();

    let src_scalars = split.scalars(0);
    let untouched_1 = split.scalars(1);
    let untouched_3 = split.scalars(3);

    // `src_scalars` was read AFTER set_cat_thresholds, so its num_cat_threshold == 2.
    assert_eq!(src_scalars.num_cat_threshold, 2);

    // Deep-copy slot 0 -> slot 2.
    split.copy_slot(0, 2).unwrap();

    // Every field at dst (2) equals src (0) — including num_cat_threshold.
    assert_eq!(split.scalars(2), src_scalars, "dst scalars must equal src");
    // The categorical slab window copied exactly.
    assert_eq!(split.cat_threshold(2), &[5, 6]);
    assert_eq!(split.cat_threshold_real(2), &[50, 60]);
    // Source itself is unchanged.
    assert_eq!(split.scalars(0), src_scalars);
    assert_eq!(split.cat_threshold(0), &[5, 6]);
    // Other slots are completely untouched.
    assert_eq!(split.scalars(1), untouched_1);
    assert_eq!(split.scalars(3), untouched_3);
}

#[test]
fn cat_thresholds_reserved_to_max_cat_per_split() {
    // The reserved slab accepts exactly MAX_CAT_PER_SPLIT thresholds and rejects one
    // more (Open Q3 — the cap is the only thing that changes the slab length).
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 2).unwrap();

    let full: Vec<u32> = (0..MAX_CAT_PER_SPLIT as u32).collect();
    let full_real: Vec<i32> = (0..MAX_CAT_PER_SPLIT as i32).collect();
    split.set_cat_thresholds(0, &full, &full_real).unwrap();
    assert_eq!(split.cat_threshold(0).len(), MAX_CAT_PER_SPLIT);
    assert_eq!(split.cat_threshold(0), full.as_slice());

    let over: Vec<u32> = (0..MAX_CAT_PER_SPLIT as u32 + 1).collect();
    let over_real: Vec<i32> = (0..MAX_CAT_PER_SPLIT as i32 + 1).collect();
    assert!(
        split.set_cat_thresholds(0, &over, &over_real).is_err(),
        "exceeding MAX_CAT_PER_SPLIT must be rejected (slab is reserved-width)"
    );
}

#[test]
fn out_of_range_slot_is_rejected() {
    // V5 boundary (threat T-14-04-01): out-of-bounds slot indices error, never UB.
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 2).unwrap();
    assert!(split.set_scalars(2, &distinct_scalars(1)).is_err());
    assert!(split.copy_slot(0, 2).is_err());
    assert!(split.copy_slot(2, 0).is_err());
    assert!(split.set_cat_thresholds(5, &[1], &[1]).is_err());
}

#[test]
fn cat_threshold_length_mismatch_is_rejected() {
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 2).unwrap();
    assert!(split.set_cat_thresholds(0, &[1, 2], &[1]).is_err());
}

#[test]
fn zero_slots_is_rejected() {
    let client = cpu_client();
    assert!(DeviceSplitInfo::<ActiveRuntime>::new(&client, 0).is_err());
}

#[test]
fn copy_slot_to_self_is_noop() {
    let client = cpu_client();
    let mut split = DeviceSplitInfo::<ActiveRuntime>::new(&client, 2).unwrap();
    let s = distinct_scalars(9);
    split.set_scalars(0, &s).unwrap();
    split.copy_slot(0, 0).unwrap();
    assert_eq!(split.scalars(0), s);
}
