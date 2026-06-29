//! Serial-f64-reference-anchored self-tests for the full-depth grow-loop
//! primitives (Phase 14 Plan 03, D-01).
//!
//! ## Anchor discipline (D-10, def-f8u-01)
//! Every device result is compared against a plain **serial Rust fold** executed
//! on the cubecl-cpu f64 anchor (`cpu_client`) — NEVER GPU-vs-GPU. Inputs are
//! exact integers (representable in f64/f32) so the cpu anchor asserts
//! **bit-exact**; the index-only argsort is always bit-exact (it is an integer
//! permutation). The ROCm/CUDA f32 leg (~1e-6) and the C++-fixture cross-check
//! land in 14-06 — this file pins the cpu-anchor behaviour.
//!
//! ## Why the cpu anchor is a serial fold, not a plane kernel
//! 14-01 proved cubecl-cpu has NO plane support (`has_plane == false`,
//! `plane_size == 1`); the plane intrinsics abort at launch. The cpu-anchor
//! kernels are therefore single-owner (`CubeDim::new_1d(1)`) serial folds — the
//! same determinism mandate the shipped `construct_hist_kernel` uses. The
//! plane-intrinsic GPU variants (built on `plane_inclusive_sum` etc.) are
//! `rocm`-gated and cross-validated in 14-06.

use lgbm_compute::kernels::primitives::{
    bitonic_argsort_on, dot_product_f64_on, prefix_sum_exclusive_f64_on,
    prefix_sum_inclusive_f64_on, reduce_max_f64_on, reduce_min_f64_on, reduce_sum_f64_on,
};
use lgbm_compute::runtime::cpu_client;

// --- serial references (the anchor, D-10) ---

fn serial_inclusive(data: &[f64]) -> Vec<f64> {
    let mut acc = 0.0f64;
    let mut out = Vec::with_capacity(data.len());
    for &x in data {
        acc += x;
        out.push(acc);
    }
    out
}

fn serial_exclusive(data: &[f64]) -> Vec<f64> {
    let mut acc = 0.0f64;
    let mut out = Vec::with_capacity(data.len());
    for &x in data {
        out.push(acc);
        acc += x;
    }
    out
}

#[test]
fn prefix_sum_inclusive_within_block() {
    // Behaviour: inclusive block scan of [1,2,3,4] -> [1,3,6,10].
    let client = cpu_client();
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let got = prefix_sum_inclusive_f64_on(&client, &data, 256).unwrap();
    assert_eq!(got, vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn prefix_sum_exclusive_within_block() {
    // Behaviour: exclusive block scan of [1,2,3,4] -> [0,1,3,6].
    let client = cpu_client();
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    let got = prefix_sum_exclusive_f64_on(&client, &data, 256).unwrap();
    assert_eq!(got, vec![0.0, 1.0, 3.0, 6.0]);
}

#[test]
fn prefix_sum_multi_block_matches_serial() {
    // Behaviour: a global scan over an array spanning MANY blocks equals the
    // serial running sum across the whole array. block_size = 64 over 1000
    // elements forces ~16 blocks, exercising the 3-launch global structure
    // (block scan -> block-totals scan -> add-back) and the reused scratch.
    // All partial sums of 1..=1000 (max 500500) are exactly representable in
    // f64, so grouping is irrelevant -> bit-exact.
    let client = cpu_client();
    let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();

    let got_incl = prefix_sum_inclusive_f64_on(&client, &data, 64).unwrap();
    assert_eq!(got_incl, serial_inclusive(&data));

    let got_excl = prefix_sum_exclusive_f64_on(&client, &data, 64).unwrap();
    assert_eq!(got_excl, serial_exclusive(&data));
}

#[test]
fn prefix_sum_block_boundary_exact() {
    // A small, hand-checkable multi-block case (block_size = 2 over 7 elements
    // -> 4 blocks, the last partial) to lock the cross-block add-back math.
    let client = cpu_client();
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let got_incl = prefix_sum_inclusive_f64_on(&client, &data, 2).unwrap();
    assert_eq!(got_incl, vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let got_excl = prefix_sum_exclusive_f64_on(&client, &data, 2).unwrap();
    assert_eq!(got_excl, vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0]);
}

// =========================================================================
// Task 2: shuffle reductions (sum / max / min, dot-product)
// =========================================================================
//
// Open Q2 (f64 reduction-order policy) RESOLVED per reduction:
// - sum  : BIT-EXACT vs a serial Rust f64 fold in ASCENDING index order — the
//          cpu anchor's single-owner fold IS that order (matched-order policy).
// - max/min: BIT-EXACT, order-INDEPENDENT (no rounding; selection only).
// - dotprod: BIT-EXACT vs `acc += a[i]*b[i]` in ascending order (matched order).
// The f32 hip warp-tree reductions are held to ~1e-6 only (14-06, never asserted
// GPU-vs-GPU here).

fn serial_sum(data: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for &x in data {
        acc += x;
    }
    acc
}

fn serial_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[test]
fn reduce_sum_max_min_within_plane() {
    let client = cpu_client();
    let data = vec![1.0f64, 2.0, 3.0, 4.0];
    assert_eq!(reduce_sum_f64_on(&client, &data).unwrap(), 10.0);
    assert_eq!(reduce_max_f64_on(&client, &data).unwrap(), 4.0);
    assert_eq!(reduce_min_f64_on(&client, &data).unwrap(), 1.0);
}

#[test]
fn reduce_dot_product() {
    let client = cpu_client();
    let a = vec![1.0f64, 2.0, 3.0];
    let b = vec![4.0f64, 5.0, 6.0];
    // 1*4 + 2*5 + 3*6 = 32.
    assert_eq!(dot_product_f64_on(&client, &a, &b).unwrap(), 32.0);
}

#[test]
fn reduce_cross_plane_matches_serial() {
    // Length > any plane width (1000) folds correctly. f64 sum is bit-exact vs
    // the serial ascending fold (matched-order policy); max/min order-independent.
    let client = cpu_client();
    let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    assert_eq!(reduce_sum_f64_on(&client, &data).unwrap(), serial_sum(&data));
    assert_eq!(reduce_max_f64_on(&client, &data).unwrap(), 1000.0);
    assert_eq!(reduce_min_f64_on(&client, &data).unwrap(), 1.0);

    let b: Vec<f64> = (1..=1000).map(|i| (1001 - i) as f64).collect();
    assert_eq!(
        dot_product_f64_on(&client, &data, &b).unwrap(),
        serial_dot(&data, &b)
    );
}

#[test]
fn reduce_dot_length_mismatch_errors() {
    let client = cpu_client();
    assert!(dot_product_f64_on(&client, &[1.0, 2.0], &[1.0]).is_err());
}

#[test]
fn prefix_sum_empty_and_single() {
    // Empty / single-element inputs handled without panic.
    let client = cpu_client();
    assert_eq!(
        prefix_sum_inclusive_f64_on(&client, &[], 256).unwrap(),
        Vec::<f64>::new()
    );
    assert_eq!(
        prefix_sum_exclusive_f64_on(&client, &[], 256).unwrap(),
        Vec::<f64>::new()
    );
    assert_eq!(
        prefix_sum_inclusive_f64_on(&client, &[42.0], 256).unwrap(),
        vec![42.0]
    );
    assert_eq!(
        prefix_sum_exclusive_f64_on(&client, &[42.0], 256).unwrap(),
        vec![0.0]
    );
}

// =========================================================================
// Task 3: single-block index-only bitonic argsort
// =========================================================================
//
// Serial reference mirroring the AMD-fork `cuda_algorithms.hpp`
// `BitonicArgSort_1024` comparator EXACTLY (strict `>`, `outer_segment_index`
// ascending parity, sentinel-padded to the next power of two). The kernel is
// asserted permutation-bit-exact against THIS reference (an integer permutation
// has no float tolerance), including a tie-rich input that locks the convention.

fn ref_depth_aligned(num_items: usize) -> (u32, u32) {
    let mut depth = 1u32;
    let mut aligned = 1u32;
    let mut r = (num_items as u32).saturating_sub(1);
    while r > 0 {
        r >>= 1;
        aligned <<= 1;
        depth += 1;
    }
    (depth, aligned)
}

fn serial_bitonic_argsort(keys: &[f32], ascending: bool) -> Vec<i32> {
    let num_items = keys.len();
    if num_items == 0 {
        return Vec::new();
    }
    let (depth, aligned) = ref_depth_aligned(num_items);
    let sentinel = if ascending {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut padded = keys.to_vec();
    padded.resize(aligned as usize, sentinel);
    let mut indices: Vec<i32> = (0..aligned as i32).collect();
    for od in (1..depth).rev() {
        let outer_segment_length = 1u32 << (depth - od);
        for inner in od..depth {
            let segment_length = 1u32 << (depth - inner);
            let half = segment_length >> 1;
            for tid in 0..aligned {
                let osi = tid / outer_segment_length;
                let asc = if ascending {
                    osi & 1 == 0
                } else {
                    osi & 1 == 1
                };
                let hsi = tid / half;
                if hsi & 1 == 0 {
                    let cmp = tid + half;
                    let ka = padded[indices[tid as usize] as usize];
                    let kb = padded[indices[cmp as usize] as usize];
                    if (ka > kb) == asc {
                        indices.swap(tid as usize, cmp as usize);
                    }
                }
            }
        }
    }
    indices[..num_items].to_vec()
}

#[test]
fn argsort_distinct_keys_ascending() {
    // Behaviour: argsort of [3,1,2] (identity start) -> permutation [1,2,0].
    let client = cpu_client();
    let keys = vec![3.0f32, 1.0, 2.0];
    let (perm, keys_after) = bitonic_argsort_on(&client, &keys, true).unwrap();
    assert_eq!(perm, vec![1, 2, 0]);
    // The value/key array is never mutated — only the index array is reordered.
    assert_eq!(keys_after, keys, "keys must be unmutated (index-only argsort)");
}

#[test]
fn argsort_matches_serial_distinct() {
    let client = cpu_client();
    let keys = vec![5.0f32, 2.0, 9.0, 1.0, 7.0, 3.0, 8.0];
    let (perm, _) = bitonic_argsort_on(&client, &keys, true).unwrap();
    assert_eq!(perm, serial_bitonic_argsort(&keys, true));
    let gathered: Vec<f32> = perm.iter().map(|&i| keys[i as usize]).collect();
    assert!(gathered.windows(2).all(|w| w[0] <= w[1]), "ascending");
}

#[test]
fn argsort_tie_rich_matches_serial() {
    // Tie-rich input ([2,1,2,1]) yields the SAME permutation the C++-mirrored
    // comparator produces (locks the tie convention before 14-06's fixture check).
    let client = cpu_client();
    let keys = vec![2.0f32, 1.0, 2.0, 1.0];
    let (perm, keys_after) = bitonic_argsort_on(&client, &keys, true).unwrap();
    assert_eq!(
        perm,
        serial_bitonic_argsort(&keys, true),
        "tie order must match the C++ comparator/tie convention"
    );
    assert_eq!(keys_after, keys);
    // Valid permutation of 0..4 and ascending by key.
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
    let gathered: Vec<f32> = perm.iter().map(|&i| keys[i as usize]).collect();
    assert!(gathered.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn argsort_single_and_empty() {
    let client = cpu_client();
    let (perm, ka) = bitonic_argsort_on(&client, &[42.0f32], true).unwrap();
    assert_eq!(perm, vec![0]);
    assert_eq!(ka, vec![42.0]);
    let (ep, ek) = bitonic_argsort_on(&client, &[], true).unwrap();
    assert_eq!(ep, Vec::<i32>::new());
    assert_eq!(ek, Vec::<f32>::new());
}
