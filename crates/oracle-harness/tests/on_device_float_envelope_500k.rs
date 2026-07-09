//! Phase-29 29-04 (OCX-04, Task 3) — the ≥500k-row NON-UNIFORM-hessian float-reduction
//! ENVELOPE gate over the FULL resident chain.
//!
//! # The blind spot this gate closes (spike-072 Results 5)
//!
//! Every float-reduction gate before Phase 29 ran SMALL corpora with UNIFORM hessians — the
//! exact regime where the spike-072 f32 root-fold bias was INVISIBLE. Iter-0 binary-logloss
//! hessians are uniformly ≈0.25, which is exactly 32 ulps at accumulator magnitude ~124k, so
//! every add is EXACT and the serial f32 fold shows no bias (spike-072 item 13, "Tree 0
//! escapes"). The bias only appears once (a) the corpus is large enough for the accumulator
//! to reach a magnitude where ~0.25 addends round, AND (b) the hessians are NON-UNIFORM
//! (iter-1+ binary logloss). This gate runs at ≥500k rows with NON-UNIFORM (binary-logloss-
//! like) hessians so the f32-accumulation class is OBSERVABLE — it must never be
//! "optimized away" back to a small/uniform corpus.
//!
//! # Two lanes (mirrors `on_device_sync_count.rs` / `on_device_integer_anchor.rs`)
//!   - **rocm RESIDENT lane (`#[cfg(feature = "rocm")]`, hardware-only):** ONE resident grow
//!     (`RocmBackend::default()`) on the hip client — the only lane where an f32-accumulation
//!     defect is visible (the cubecl-cpu anchor has `has_f64=true`, so it cannot see this
//!     class). Budget: ONE dataset allocation + ONE grow (num_leaves ≤ 15), well under 3 min.
//!   - **cpu ANCHOR companion (always runnable):** the SAME assertions on the cubecl-cpu f64
//!     anchor arm — proves the harness itself and gives CI a lane for future refactors.
//!
//! # Assertions (device-vs-HOST-f64 ONLY — never GPU-f32-vs-GPU-f32, def-f8u-01)
//!   (a) the seeded root grad/hess sums are BIT-EXACT to an independent host f64 ascending
//!       fold (the 29-01 root-fold contract at the driver level);
//!   (b) for a SAMPLE of grown leaves, the emitted leaf value is within a DOCUMENTED envelope
//!       of the host f64 truth recomputed over that leaf's ACTUAL rows (from the layout);
//!   (c) every emitted leaf value is FINITE and |v| is sane (the spike-072 blow-up signature
//!       is 1.27e3 → ±inf; healthy binary-logloss leaves are O(1)).

use lgbm_compute::gain::calculate_splitted_leaf_output;
use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, CpuBackend, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};
use lgbm_dataset::LeafPartitionLayout;
use lgbm_model::Tree;

const OFFSET_MFB_0: i32 = 1;
const NUM_DATA: usize = 500_000;
const NUM_FEATURES: usize = 8;
const NUM_BIN: u32 = 255;
const NUM_LEAVES: i32 = 15;

/// Deterministic splitmix64 — a seeded PRNG with NO external dependency (the plan's
/// "seeded StdRng" role; no new crate is added this phase per the threat register).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn next_bin(&mut self, num_bin: u32) -> u32 {
        (self.next_u64() % u64::from(num_bin)) as u32
    }
}

/// Generate the ≥500k NON-UNIFORM corpus: `NUM_FEATURES` random bin columns, plus
/// binary-logloss-like gradients/hessians. For each row a probability `p ∈ (0.05, 0.95)` and
/// a Bernoulli(p) label `y` give `g = p − y` (non-uniform gradient) and `h = p(1−p)`
/// (non-uniform hessian) — the second-iteration profile the f32 fold bias needs to surface.
fn envelope_corpus() -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
    let mut rng = SplitMix64(0x5EED_0072_2904_0001);

    let mut columns: Vec<Vec<u32>> = vec![Vec::with_capacity(NUM_DATA); NUM_FEATURES];
    let mut gradients: Vec<f32> = Vec::with_capacity(NUM_DATA);
    let mut hessians: Vec<f32> = Vec::with_capacity(NUM_DATA);

    for _ in 0..NUM_DATA {
        for col in columns.iter_mut() {
            col.push(rng.next_bin(NUM_BIN));
        }
        let p = 0.05 + 0.9 * rng.next_f32(); // p ∈ (0.05, 0.95)
        let y: f32 = if rng.next_f32() < p { 1.0 } else { 0.0 };
        gradients.push(p - y); // binary-logloss gradient
        hessians.push(p * (1.0 - p)); // binary-logloss hessian (non-uniform)
    }

    let features: Vec<GrowFeature> = columns
        .into_iter()
        .enumerate()
        .map(|(fi, bins)| GrowFeature {
            bins: BinColumn::new(bins, NUM_BIN),
            num_bin: NUM_BIN,
            offset: OFFSET_MFB_0,
            min_bin: 0,
            max_bin: NUM_BIN - 1,
            default_bin: NUM_BIN,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: (0..NUM_BIN).map(|b| b as f64 + 0.5).collect(),
            real_feature_index: fi as i32,
            bin_type: BinType::Numerical,
            bin_to_category: Vec::new(),
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            max_cat_to_onehot: 4,
            min_data_per_group: 100,
        })
        .collect();

    (features, gradients, hessians)
}

/// Independent host f64 ASCENDING fold (the 29-01 reference; NOT `root_grad_hess_fold`, so
/// the assertion is against a fold computed in the test, not the code under test).
fn host_f64_fold(rows: impl Iterator<Item = u32>, grad: &[f32], hess: &[f32]) -> (f64, f64) {
    let mut sg = 0.0f64;
    let mut sh = 0.0f64;
    for r in rows {
        sg += f64::from(grad[r as usize]);
        sh += f64::from(hess[r as usize]);
    }
    (sg, sh)
}

/// The shared assertion body: given a grown `(tree, layout)` and the device root seed, assert
/// (a) root seed bit-exact to the host fold, (b) sampled leaf values within `leaf_atol +
/// leaf_rtol·|truth|` of the host f64 truth over each leaf's actual rows, (c) all leaves
/// finite + sane. `leaf_atol`/`leaf_rtol` carry an in-code derivation at each call site.
fn assert_envelope(
    tree: &Tree,
    layout: &LeafPartitionLayout,
    device_root: (f64, f64),
    grad: &[f32],
    hess: &[f32],
    leaf_atol: f64,
    leaf_rtol: f64,
) {
    // (a) ROOT SEED bit-exact to the independent host f64 ascending fold (29-01 contract).
    let host_root = host_f64_fold(0..NUM_DATA as u32, grad, hess);
    assert_eq!(
        device_root.0.to_bits(),
        host_root.0.to_bits(),
        "root sum_gradient not bit-exact to host f64 fold: device {} vs host {} (29-01)",
        device_root.0,
        host_root.0
    );
    assert_eq!(
        device_root.1.to_bits(),
        host_root.1.to_bits(),
        "root sum_hessian not bit-exact to host f64 fold: device {} vs host {} (29-01)",
        device_root.1,
        host_root.1
    );

    // (c) every leaf value finite + sane (the spike-072 1.27e3 → ±inf blow-up signature).
    for (leaf, &v) in tree.leaf_value.iter().enumerate() {
        assert!(v.is_finite(), "leaf {leaf} value is non-finite ({v}) — spike-072 blow-up");
        assert!(v.abs() < 1e3, "leaf {leaf} value {v} implausibly large (blow-up signature)");
    }

    // (b) sampled leaf values within the documented envelope of the host f64 truth fold over
    // each leaf's ACTUAL rows. use_l1=false, l1=l2=0 (proving config), so the expected leaf
    // output is -sum_g/sum_h over the leaf's rows.
    let final_leaves = tree.num_leaves as usize;
    assert_eq!(
        layout.leaf_begin.len(),
        final_leaves,
        "layout leaf count {} != tree leaf count {final_leaves}",
        layout.leaf_begin.len()
    );
    // Sample a spread of leaves (first, middle, last, and a few between).
    let sample: Vec<usize> = (0..final_leaves).step_by((final_leaves / 6).max(1)).collect();
    for &leaf in &sample {
        let begin = layout.leaf_begin[leaf] as usize;
        let count = layout.leaf_count[leaf] as usize;
        let rows = layout.indices[begin..begin + count].iter().copied();
        let (sg, sh) = host_f64_fold(rows, grad, hess);
        let expected = calculate_splitted_leaf_output(false, sg, sh, 0.0, 0.0);
        let got = tree.leaf_value[leaf];
        let envelope = leaf_atol + leaf_rtol * expected.abs();
        assert!(
            (got - expected).abs() <= envelope,
            "leaf {leaf} value {got} diverged from host f64 truth {expected} by {} > {envelope} \
             (sum_g={sg}, sum_h={sh}, n={count})",
            (got - expected).abs()
        );
    }
}

/// Task 3 (cpu ANCHOR companion) — the same properties at 500k on the cubecl-cpu f64 anchor
/// arm. Proves the harness; leaf values are within a TIGHT f64-reorder envelope (the emitted
/// leaf sums are bin-grouped folds of the histogram while the host truth is a row-order fold —
/// f64 non-associativity ≲ n·u·Σ|h| ≈ 7e-6 abs on the sum at n=5e5, so a leaf-value envelope
/// of atol=1e-7 / rtol=1e-9 is generous). Root seed is BIT-EXACT (both are ascending folds).
#[test]
fn float_envelope_500k_cpu_anchor() {
    use lgbm_compute::Backend;
    let client = cpu_client();
    let (features, gradients, hessians) = envelope_corpus();

    // The anchor arm's seed path (default `root_grad_hess_sum` = the ascending f64 fold);
    // assert_envelope compares it bit-exact to an INDEPENDENT host fold computed in-test.
    let device_root = CpuBackend.root_grad_hess_sum(&client, &gradients, &hessians);

    let (tree, layout) =
        grow_tree_on_device_driver(&CpuBackend, &client, &gradients, &hessians, &features, NUM_LEAVES, -1)
            .expect("cpu-anchor 500k grow");

    assert_envelope(&tree, &layout, device_root, &gradients, &hessians, 1e-7, 1e-9);
}

/// Task 3 (rocm RESIDENT lane) — ONE ≥500k non-uniform-hessian resident grow on real hip
/// hardware. Root seed BIT-EXACT to the host f64 fold (29-01 on-device fold); sampled leaf
/// values within the u64-dequant envelope: the resident build quantizes grad/hess at
/// S = 2^30, so each cell carries ≤ 2^-31 abs error and a leaf sum over n rows ≤ n·2^-31 ≈
/// 2.3e-4 at n=5e5; propagated through -sg/sh (sh = O(n·0.1) ≈ 6e4) that is a leaf-value
/// atol ≈ 1e-3 / rtol ≈ 1e-3 — loose but DERIVED, and (c) finiteness is the real spike-072
/// guard. Compares device-vs-host-f64 ONLY (def-f8u-01: never GPU-f32-vs-GPU-f32).
#[cfg(feature = "rocm")]
#[test]
fn float_envelope_500k_rocm_resident() {
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::{Backend, RocmBackend};

    let backend = RocmBackend::default();
    assert!(backend.resident_pool_supported(), "default RocmBackend must drive the resident chain");
    let hip = rocm_client();
    let (features, gradients, hessians) = envelope_corpus();

    // The 29-01 on-device f64 root fold (the resident driver's own seed path).
    let device_root = backend.root_grad_hess_sum(&hip, &gradients, &hessians);

    let (tree, layout) =
        grow_tree_on_device_driver(&backend, &hip, &gradients, &hessians, &features, NUM_LEAVES, -1)
            .expect("rocm resident 500k grow");

    // u64-dequant leaf envelope (derived above): atol 1e-3, rtol 1e-3.
    assert_envelope(&tree, &layout, device_root, &gradients, &hessians, 1e-3, 1e-3);
}
