//! Phase-28 28-07 (ODF-06, WR-01) — the resident device cross-leaf best-leaf pick
//! (§8.3, `find_best_from_all_splits_device` → `find_best_leaf_kernel`) breaks an EXACT
//! gain tie by the SAME key as the cpu-f64 anchor `SerialTreeLearner::split_gt`: gain
//! descending, then the LOWER real feature index (`-1` ⇒ `i32::MAX`), then the lower leaf
//! index.
//!
//! This is the regression that catches WR-01: before the fix the device pick compared
//! ONLY gain and kept the LOWEST LEAF INDEX on a tie (ignoring the winning feature index),
//! so on any exact gain tie the resident (GPU) driver grew a DIFFERENT tree than the
//! non-resident cpu-f64 merge-gate anchor. The fixture below deliberately induces an exact
//! gain tie between two valid leaves whose real feature indices ORDER OPPOSITE to their
//! leaf indices, so the leaf-index-only pick and the split_gt pick DISAGREE — the test
//! FAILS against the pre-fix kernel and passes after.
//!
//! Runs on the cubecl-cpu f64 anchor (the merge gate) — `find_best_leaf_kernel` is an
//! ungated `#[cube]` kernel, so this parity gate needs NO GPU hardware (28-REVIEW
//! verification caveat: the resident sequencing is otherwise only compile-covered).

use lgbm_compute::kernels::best_split::{
    find_best_from_all_splits_device, find_best_from_all_splits_on, SplitSoa,
};
use lgbm_compute::kernels::split_info::SplitScalars;
use lgbm_compute::runtime::cpu_client;

/// A minimal per-leaf winning-split record: `is_valid`, the winner's REAL feature index
/// (the §8.3 tie-break key the resident frontier stores in `feat`), and `gain`.
fn rec(gain: f64, real_feat: i32, valid: bool) -> SplitScalars {
    SplitScalars {
        is_valid: valid,
        inner_feature_index: real_feat,
        gain,
        threshold: 3,
        default_left: false,
        ..SplitScalars::default()
    }
}

/// The cpu-f64 anchor cross-leaf argmax, byte-for-byte the tie rule of
/// `grow_driver::split_gt` (gain desc, real-feature asc with `-1 ⇒ i32::MAX`, leaf asc):
/// iterate leaves ascending, replace the running winner ONLY on a strictly-better split.
fn split_gt_argmax(recs: &[SplitScalars]) -> i32 {
    let mut best_leaf = -1i32;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feat = i32::MAX;
    for (leaf, r) in recs.iter().enumerate() {
        if !r.is_valid {
            continue;
        }
        let rf = if r.inner_feature_index == -1 {
            i32::MAX
        } else {
            r.inner_feature_index
        };
        let better = r.gain > best_gain || (r.gain == best_gain && rf < best_feat);
        if best_leaf == -1 || better {
            best_leaf = leaf as i32;
            best_gain = r.gain;
            best_feat = rf;
        }
    }
    best_leaf
}

/// WR-01: on an EXACT gain tie the device §8.3 pick selects the leaf whose winning split
/// has the LOWER real feature index — identical to the cpu-f64 anchor `split_gt` fold —
/// NOT merely the lowest leaf index. FAILS against the pre-fix leaf-index-only kernel.
#[test]
fn device_pick_matches_split_gt_on_exact_gain_tie() {
    let client = cpu_client();

    // Frontier (4 slots): leaves 0 and 1 tie at gain 10.0 but their real feature indices
    // (5 vs 2) order OPPOSITE to their leaf indices — the leaf-index-only pick keeps leaf
    // 0, split_gt keeps leaf 1 (real feat 2 < 5). Leaf 2 is a lower-gain distractor; slot
    // 3 is the reserved freshly-created slot (invalid).
    let recs = vec![
        rec(10.0, 5, true),  // leaf 0 — same gain, HIGHER real feature index
        rec(10.0, 2, true),  // leaf 1 — same gain, LOWER real feature index ⇒ split_gt winner
        rec(8.0, 1, true),   // leaf 2 — strictly lower gain
        rec(0.0, -1, false), // leaf 3 — reserved freshly-created slot (invalid)
    ];
    let cur_num_leaves = 3usize; // leaves 0,1,2 are in the argmax window
    let smaller_leaf_index = 0i32;
    let larger_leaf_index = 1i32;

    // The cpu-f64 anchor: split_gt picks leaf 1 (the lower real feature index on the tie).
    let expected = split_gt_argmax(&recs[..cur_num_leaves]);
    assert_eq!(expected, 1, "sanity: the split_gt anchor picks the lower-feature leaf");

    // The device §8.3 pick over the resident frontier.
    let frontier = SplitSoa::from_records(&client, &recs);
    let best_leaf_slot = client.empty(core::mem::size_of::<f64>());
    let dev_export = find_best_from_all_splits_device(
        &client,
        &frontier,
        &best_leaf_slot,
        smaller_leaf_index,
        larger_leaf_index,
        cur_num_leaves,
    )
    .expect("device §8.3 pick over the tie-inducing frontier");
    let device_best_leaf = dev_export.cells[6] as i32;

    // The load-bearing WR-01 assertion: the device pick uses split_gt's tie-break, NOT the
    // lowest-leaf-index (which would pick leaf 0).
    assert_eq!(
        device_best_leaf, expected,
        "device §8.3 pick must break the exact gain tie by the LOWER real feature index \
         (split_gt), picking leaf {expected} — the pre-fix leaf-index-only kernel picks leaf 0"
    );

    // The host mirror fold agrees with both (device == split_gt fold == host fold).
    let mut per_leaf = recs.clone();
    let host_export = find_best_from_all_splits_on(
        &client,
        &mut per_leaf,
        smaller_leaf_index,
        larger_leaf_index,
        cur_num_leaves,
    )
    .expect("host §8.3 fold over the tie-inducing frontier");
    assert_eq!(
        host_export.cells[6] as i32,
        expected,
        "the host find_best_from_all_splits_on fold must also break the tie by split_gt"
    );
    assert_eq!(
        device_best_leaf, host_export.cells[6] as i32,
        "the device §8.3 pick and the host fold must agree on the tie"
    );
}
