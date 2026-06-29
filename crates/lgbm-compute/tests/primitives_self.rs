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
    dot_product_f64_on, prefix_sum_exclusive_f64_on, prefix_sum_inclusive_f64_on,
    reduce_max_f64_on, reduce_min_f64_on, reduce_sum_f64_on,
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
