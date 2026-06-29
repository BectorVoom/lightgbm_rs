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
    bitonic_argsort_global_on, bitonic_argsort_items_on, bitonic_argsort_on, dot_product_f64_on,
    percentile_unweighted_f32_on, percentile_weighted_f32_on, prefix_sum_exclusive_f64_on,
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

// =========================================================================
// 14-05 Task 1: weighted + unweighted percentile SKELETON
// =========================================================================
//
// Serial f64 references mirroring the C++ `PercentileDevice` (unweighted +
// weighted branches) EXACTLY, both composing the same `serial_bitonic_argsort`
// ascending sort the device skeleton composes. The device skeleton is asserted
// BIT-EXACT vs (a) hand-computed concrete values and (b) the serial reference on
// non-uniform inputs — never GPU-vs-GPU (D-10).

fn serial_percentile_unweighted(values: &[f32], alpha: f64) -> f32 {
    let len = values.len();
    if len == 1 {
        return values[0];
    }
    let perm = serial_bitonic_argsort(values, true);
    let float_pos = (1.0 - alpha) * len as f64;
    let pos = float_pos as usize;
    if pos < 1 {
        return values[perm[0] as usize];
    }
    if pos >= len {
        return values[perm[len - 1] as usize];
    }
    let bias = float_pos - pos as f64;
    let v1 = values[perm[pos - 1] as usize] as f64;
    let v2 = values[perm[pos] as usize] as f64;
    (v1 - (v1 - v2) * bias) as f32
}

fn serial_percentile_weighted(values: &[f32], weights: &[f64], alpha: f64) -> f32 {
    let len = values.len();
    if len == 1 {
        return values[0];
    }
    let perm = serial_bitonic_argsort(values, true);
    let mut wps = vec![0.0f64; len];
    let mut acc = 0.0f64;
    for i in 0..len {
        acc += weights[perm[i] as usize];
        wps[i] = acc;
    }
    let threshold = wps[len - 1] * (1.0 - alpha);
    let mut pos = len;
    for index in 0..len {
        if wps[index] > threshold && (index == 0 || wps[index - 1] <= threshold) {
            pos = index;
        }
    }
    let pos = pos.min(len - 1);
    if pos == 0 || pos == len - 1 {
        return values[pos];
    }
    let v1 = values[perm[pos - 1] as usize] as f64;
    let v2 = values[perm[pos] as usize] as f64;
    let frac = (threshold - wps[pos - 1]) / (wps[pos] - wps[pos - 1]);
    (v1 - (v1 - v2) * frac) as f32
}

#[test]
fn percentile_unweighted_median_concrete() {
    // Behaviour: alpha=0.5 over [1,2,3,4,5] -> float_pos=2.5, pos=2, bias=0.5;
    // interpolate sorted[1]=2 and sorted[2]=3 -> 2 - (2-3)*0.5 = 2.5. Order
    // independent (unsorted input gives the same sorted order).
    let client = cpu_client();
    let sorted = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(percentile_unweighted_f32_on(&client, &sorted, 0.5).unwrap(), 2.5);
    let unsorted = vec![3.0f32, 1.0, 4.0, 2.0, 5.0];
    assert_eq!(
        percentile_unweighted_f32_on(&client, &unsorted, 0.5).unwrap(),
        2.5
    );
}

#[test]
fn percentile_unweighted_matches_serial() {
    let client = cpu_client();
    let values = vec![5.0f32, 2.0, 9.0, 1.0, 7.0, 3.0, 8.0, 4.0];
    for &alpha in &[0.0f64, 0.1, 0.5, 0.9, 1.0] {
        assert_eq!(
            percentile_unweighted_f32_on(&client, &values, alpha).unwrap(),
            serial_percentile_unweighted(&values, alpha),
            "alpha = {alpha}"
        );
    }
    // Empty -> error (V5 boundary); single -> the element.
    assert!(percentile_unweighted_f32_on(&client, &[], 0.5).is_err());
    assert_eq!(percentile_unweighted_f32_on(&client, &[7.0], 0.5).unwrap(), 7.0);
}

#[test]
fn percentile_weighted_concrete_and_serial() {
    let client = cpu_client();
    // Concrete edge branch: weights [1,1,1,5], alpha=0.5 over [1,2,3,4].
    // wps=[1,2,3,8], threshold=4.0, crossing at pos=3 == len-1 -> values[3]=4.0.
    let values = vec![1.0f32, 2.0, 3.0, 4.0];
    let weights = vec![1.0f64, 1.0, 1.0, 5.0];
    assert_eq!(
        percentile_weighted_f32_on(&client, &values, &weights, 0.5).unwrap(),
        4.0
    );
    // Concrete interior: uniform weights, alpha=0.5 -> threshold 2.0, pos=2,
    // frac=(2-2)/(3-2)=0 -> values[perm[1]]=2.0.
    let uniform = vec![1.0f64, 1.0, 1.0, 1.0];
    assert_eq!(
        percentile_weighted_f32_on(&client, &values, &uniform, 0.5).unwrap(),
        2.0
    );
    // Non-uniform vs serial reference across alphas (unsorted values, tie-free).
    let v2 = vec![5.0f32, 2.0, 9.0, 1.0, 7.0];
    let w2 = vec![2.0f64, 1.0, 4.0, 3.0, 1.0];
    for &alpha in &[0.0f64, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(
            percentile_weighted_f32_on(&client, &v2, &w2, alpha).unwrap(),
            serial_percentile_weighted(&v2, &w2, alpha),
            "alpha = {alpha}"
        );
    }
    // Mismatched weights length -> error.
    assert!(percentile_weighted_f32_on(&client, &values, &[1.0], 0.5).is_err());
}

// =========================================================================
// 14-05 Task 2: multi-block / global argsort SKELETON
// =========================================================================
//
// The skeleton extends the single-block network to inputs spanning more than one
// single-block tile (> 1024). Permutation asserted BIT-EXACT vs the same
// `serial_bitonic_argsort` reference (an integer permutation, no float tol),
// including a tie-rich case.

#[test]
fn argsort_global_multi_block_matches_serial() {
    // 1500 elements > BITONIC_SORT_NUM_ELEMENTS (1024) forces the input to span
    // more than one single-block tile. Distinct descending keys -> the ascending
    // argsort permutation reverses them; asserted bit-exact vs the serial network.
    let client = cpu_client();
    let keys: Vec<f32> = (0..1500).map(|i| (1500 - i) as f32).collect();
    let (perm, keys_after) = bitonic_argsort_global_on(&client, &keys, true).unwrap();
    assert_eq!(perm, serial_bitonic_argsort(&keys, true));
    assert_eq!(keys_after, keys, "keys must be unmutated (index-only)");
    let gathered: Vec<f32> = perm.iter().map(|&i| keys[i as usize]).collect();
    assert!(gathered.windows(2).all(|w| w[0] <= w[1]), "ascending");
}

#[test]
fn argsort_global_tie_rich_matches_serial() {
    // Tie-rich multi-block input: 1100 elements (> 1024) with only 3 distinct keys
    // so ties are dense across tile boundaries. Permutation must match the serial
    // C++-mirrored comparator exactly.
    let client = cpu_client();
    let keys: Vec<f32> = (0..1100).map(|i| (i % 3) as f32).collect();
    let (perm, keys_after) = bitonic_argsort_global_on(&client, &keys, true).unwrap();
    assert_eq!(perm, serial_bitonic_argsort(&keys, true));
    assert_eq!(keys_after, keys);
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    let expected: Vec<i32> = (0..1100).collect();
    assert_eq!(sorted, expected, "valid permutation of 0..1100");
    let gathered: Vec<f32> = perm.iter().map(|&i| keys[i as usize]).collect();
    assert!(gathered.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn argsort_global_empty() {
    let client = cpu_client();
    let (ep, ek) = bitonic_argsort_global_on(&client, &[], true).unwrap();
    assert_eq!(ep, Vec::<i32>::new());
    assert_eq!(ek, Vec::<f32>::new());
}

// =========================================================================
// 14-05 Task 3: per-segment ranking items-sort SKELETON
// =========================================================================
//
// Sorts indices WITHIN each segment by key (index-only), each segment a LOCAL
// 0-based permutation. Asserted bit-exact vs a serial per-segment reference that
// reuses the same `serial_bitonic_argsort` convention, including a tie-rich
// segment.

fn serial_items_sort(keys: &[f32], boundaries: &[i32], ascending: bool) -> Vec<i32> {
    let mut out = Vec::with_capacity(keys.len());
    for q in 0..boundaries.len() - 1 {
        let start = boundaries[q] as usize;
        let end = boundaries[q + 1] as usize;
        out.extend(serial_bitonic_argsort(&keys[start..end], ascending));
    }
    out
}

#[test]
fn items_sort_per_segment_matches_serial() {
    // Three segments of differing lengths: [0,3), [3,5), [5,9). Each segment's
    // slice of the output is that segment's local (0-based) permutation.
    let client = cpu_client();
    let keys = vec![
        3.0f32, 1.0, 2.0, // seg 0: -> local perm [1,2,0]
        9.0, 4.0, // seg 1: -> [1,0]
        5.0, 8.0, 1.0, 7.0, // seg 2
    ];
    let boundaries = vec![0i32, 3, 5, 9];
    let got = bitonic_argsort_items_on(&client, &keys, &boundaries, true).unwrap();
    assert_eq!(got, serial_items_sort(&keys, &boundaries, true));
    // Hand-check the first two segments' local permutations.
    assert_eq!(&got[0..3], &[1, 2, 0]);
    assert_eq!(&got[3..5], &[1, 0]);
    // Each segment slice is a valid local permutation and ascending by key.
    for q in 0..boundaries.len() - 1 {
        let s = boundaries[q] as usize;
        let e = boundaries[q + 1] as usize;
        let seg_perm = &got[s..e];
        let mut sorted = seg_perm.to_vec();
        sorted.sort_unstable();
        let expected: Vec<i32> = (0..(e - s) as i32).collect();
        assert_eq!(sorted, expected, "segment {q} local permutation");
        let gathered: Vec<f32> = seg_perm.iter().map(|&i| keys[s + i as usize]).collect();
        assert!(gathered.windows(2).all(|w| w[0] <= w[1]), "segment {q} ascending");
    }
}

#[test]
fn items_sort_tie_rich_segment_matches_serial() {
    // A tie-rich segment ([2,1,2,1]) plus a distinct segment; the tie order must
    // match the C++-mirrored comparator segment-by-segment.
    let client = cpu_client();
    let keys = vec![2.0f32, 1.0, 2.0, 1.0, 5.0, 3.0, 4.0];
    let boundaries = vec![0i32, 4, 7];
    let got = bitonic_argsort_items_on(&client, &keys, &boundaries, true).unwrap();
    assert_eq!(got, serial_items_sort(&keys, &boundaries, true));
    assert_eq!(
        &got[0..4],
        &serial_bitonic_argsort(&keys[0..4], true)[..],
        "tie-rich segment matches the locked convention"
    );
}

#[test]
fn items_sort_rejects_malformed_boundaries() {
    let client = cpu_client();
    let keys = vec![1.0f32, 2.0, 3.0];
    // Does not start at 0.
    assert!(bitonic_argsort_items_on(&client, &keys, &[1, 3], true).is_err());
    // Does not end at keys.len().
    assert!(bitonic_argsort_items_on(&client, &keys, &[0, 2], true).is_err());
    // Not non-decreasing.
    assert!(bitonic_argsort_items_on(&client, &keys, &[0, 2, 1, 3], true).is_err());
    // Too few entries.
    assert!(bitonic_argsort_items_on(&client, &keys, &[0], true).is_err());
}
