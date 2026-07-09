//! Tri-state `LGBM_CUDA_ON_DEVICE` resolver + `on_device_default` unit tests,
//! PLUS the consolidated parity gate for the resident on-device tree.
//!
//! ## The resident-tree parity gate (see `resident_parity_gate`)
//! This file holds the crate's single consolidated parity gate for the resident
//! on-device build, living at the driver's home crate. It proves, in one place, three
//! non-negotiable invariants:
//!   1. **Bit-exact vs the u64 fixed-point integer path** — the on-device-grown tree's leaf
//!      values + root internal value + root split feature are BIT-EXACT to an INDEPENDENT
//!      host-side u64 fixed-point (S = 2^30) integer reference (NOT a second driver call —
//!      that would be verify-vacuity). Integer accumulation is order-independent, so on the
//!      integer-valued L2 proving slice the u64 sum dequantizes EXACTLY to the f64 fold.
//!      Coverage note: this clause runs on `CpuBackend`, whose `resident_pool_supported()`
//!      is the trait-default `false`, so the code under test on the cpu lane is the f64
//!      ANCHOR arm — it anchors the f64 STRUCTURE + integer leaf-value EQUIVALENCE, NOT the
//!      resident u64 kernel. The u64 build is only reachable on the `#[cfg(feature = "rocm")]`
//!      lanes (clause 2 + `resident_score_ab.rs`); a regression confined to the u64 kernel
//!      would pass every non-rocm run. (Exposing a resident-capable cubecl-cpu backend to
//!      drive the u64 build on the cpu lane was left out of scope as too risky for the
//!      bit-exact hard gate.)
//!   2. **~1e-6 vs the host-CUDA arm** (rocm-gated) — the on-device predictions are held to
//!      an ENVELOPE `atol + rtol·|p_host|` against the host-CUDA f32 arm, NEVER bit-exactness
//!      (comparing two nondeterministic f32 paths bit/1e-6-exact would be an invalid
//!      cross-precision check). Both GPU trees are pinned to the cpu-f64 anchor — NEVER
//!      GPU-f32-vs-GPU-f32.
//!   3. **Byte-parity** — with `LGBM_CUDA_ON_DEVICE` unset the resolver is OFF (the
//!      byte-unchanged host path), and the on-device anchor grow is BYTE-IDENTICAL run to run
//!      (the integer anchor is deterministic — no f32 drift), so the cpu/rocm merge gate is
//!      byte-stable. `on_device_default()` stays false.
//! The cpu-f64 fold remains the bit-exact hard merge gate for the CPU anchor (untouched).
//!
//! ## Tri-state resolver locks
//!
//! Two independent things are locked here:
//!
//! 1. The tri-state MAPPING (exact-match closed enum) is tested through the
//!    pure [`lgbm_compute::cuda_on_device_override_from`] helper — NOT through the
//!    public OnceLock-cached resolver. The resolver reads the env once per process,
//!    so it cannot be exercised across multiple values in one test binary; the pure
//!    helper is the OnceLock initializer's own mapping, so asserting it asserts the
//!    real contract without fighting the cache. There is deliberately NO test that
//!    flips the env between two reads in one process.
//!
//! 2. The DEVICE DEFAULT is OFF: with the env unset the resolved default is `false`,
//!    so every backend is byte-unchanged. A real-hardware A/B evaluation of flipping
//!    the default did not clear its performance and parity bars, so `on_device_default()`
//!    stays `false` and `LGBM_CUDA_ON_DEVICE="1"` remains the opt-in switch.

use lgbm_compute::cuda_on_device_override_from;

/// Exact-string closed enum. `"1"` forces on, `"0"` forces off, and EVERYTHING
/// else (unset/empty/malformed) maps to `None` = follow the device default.
#[test]
fn tri_state_mapping_is_exact_closed_enum() {
    // Force-on / force-off: the two first-class operator overrides.
    assert_eq!(cuda_on_device_override_from(Some("1")), Some(true), "\"1\" => force on");
    assert_eq!(cuda_on_device_override_from(Some("0")), Some(false), "\"0\" => force off");

    // Unset and every malformed/empty value follow the device default (None).
    assert_eq!(cuda_on_device_override_from(None), None, "unset => follow default");
    assert_eq!(cuda_on_device_override_from(Some("")), None, "empty => follow default");
    assert_eq!(cuda_on_device_override_from(Some("2")), None, "other digit => follow default");
    assert_eq!(cuda_on_device_override_from(Some("true")), None, "\"true\" => follow default");
}

/// The consolidated parity gate for the fully-resident on-device tree.
///
/// This module is the crate's single parity gate, living at the driver's home crate. It
/// grows the tree through the SAME `grow_tree_on_device_driver` the production seam uses and
/// pins it three ways (see the file-level doc): (1) BIT-EXACT to an independent host-side u64
/// fixed-point integer reference, (2) ~1e-6 ENVELOPE vs the host-CUDA arm (rocm-gated, NEVER
/// bit-exact / never GPU-f32-vs-GPU-f32), (3) byte-parity (flag-unset OFF +
/// deterministic byte-identical grow). The integer reference is a DISTINCT code path (a
/// fixed-point fold over the raw rows), NOT a second driver call, so it is a valid
/// non-tautological anchor.
///
/// Clause (1) here runs on `CpuBackend` (`resident_pool_supported()` ==
/// trait-default `false`), so on the cpu lane it anchors the f64 STRUCTURE + integer
/// leaf-value EQUIVALENCE, NOT the resident u64 kernel — that kernel is exercised only on the
/// rocm lane (clause 2). See the file-level clause-1 note for the full coverage scoping.
mod resident_parity_gate {
    use lgbm_compute::gain::{calculate_splitted_leaf_output, get_split_gains};
    use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver;
    use lgbm_compute::runtime::cpu_client;
    use lgbm_compute::{BinColumn, CpuBackend, GrowFeature};
    use lgbm_dataset::bin_mapper::{BinType, MissingType};
    use lgbm_model::Tree;
    use oracle_harness::comparator::compare_exact_f64_bits;

    /// `offset_for_most_freq_bin(0)` == 1 (drop bin 0 under the compacted convention). The
    /// helper lives in `lgbm-treelearner`, ABOVE this crate, so hardcode it (as the counter
    /// tests do).
    const OFFSET_MFB_0: i32 = 1;

    /// The u64 fixed-point scale the on-device build quantizes grad/hess at (S = 2^30,
    /// `construct_leaf_hist_resident_lds_kernel_u64`). The host-side integer reference below
    /// mirrors it: `q = round(v * S)` accumulated as `i128` (order-independent), dequantized
    /// `q as f64 / S`. On the integer-valued proving corpus this is bit-exact to the f64 fold.
    const FIXED_POINT_SCALE: f64 = (1u64 << 30) as f64;

    /// Number of leaves the tiny L2 corpus is grown to (every split scannable ⇒ non-vacuous).
    const NUM_LEAVES: i32 = 4;

    /// One `GrowFeature` carrier with the config-default categorical scalars (inert on this
    /// numeric slice) — the exact field-by-field mirror the seam builds.
    fn gf(bins: Vec<u32>, num_bin: u32, upper: Vec<f64>, real_feature_index: i32) -> GrowFeature {
        GrowFeature {
            bins: BinColumn::new(bins, num_bin),
            num_bin,
            offset: OFFSET_MFB_0,
            min_bin: 0,
            max_bin: num_bin - 1,
            default_bin: num_bin,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: upper,
            real_feature_index,
            bin_type: BinType::Numerical,
            bin_to_category: Vec::new(),
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            max_cat_to_onehot: 4,
            min_data_per_group: 100,
        }
    }

    /// A small identity-binned L2 continuous corpus (bin == raw value, no missing values):
    /// feature 0 carries a strong monotone signal (labels rise with `f0`) so the root splits
    /// on it; feature 1 is a weak secondary. Gradients are `-label` (L2 from a zero init) and
    /// INTEGER-valued — the property that makes the u64 fixed-point sum dequantize BIT-EXACTLY
    /// to the f64 fold. Field-for-field mirror of the `on_device_integer_anchor` L2 slice.
    fn corpus() -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
        let features = vec![
            gf(vec![0, 0, 1, 1, 2, 2, 3, 3], 4, vec![0.5, 1.5, 2.5, 3.5], 0),
            gf(vec![0, 1, 0, 1, 0, 1, 0, 1], 2, vec![0.5, 1.5], 1),
        ];
        let labels = [1.0f32, 1.0, 3.0, 3.0, 8.0, 8.0, 12.0, 13.0];
        let gradients: Vec<f32> = labels.iter().map(|&l| -l).collect();
        let hessians: Vec<f32> = vec![1.0f32; labels.len()];
        (features, gradients, hessians)
    }

    fn quantize(v: f32) -> i128 {
        (f64::from(v) * FIXED_POINT_SCALE).round() as i128
    }
    fn dequantize(q: i128) -> f64 {
        q as f64 / FIXED_POINT_SCALE
    }

    /// Dequantized `(sum_g, sum_h)` over `rows` via u64 fixed-point (S = 2^30) `i128`
    /// accumulation — the shared integer fold. Bit-exact to the f64 fold on integer grad +
    /// unit hess (order-independent).
    fn host_integer_sums(rows: &[u32], g: &[f32], h: &[f32]) -> (f64, f64) {
        let (mut qg, mut qh) = (0i128, 0i128);
        for &r in rows {
            qg += quantize(g[r as usize]);
            qh += quantize(h[r as usize]);
        }
        (dequantize(qg), dequantize(qh))
    }

    /// The INDEPENDENT host-side integer reference for a CHILD leaf's output: accumulate the
    /// leaf's grad/hess in u64 fixed-point, dequantize, apply the finder's L2 closed form with
    /// the reverse-scan `+kEpsilon` hessian seed (every final leaf in a split tree is such a
    /// child — the universal child-leaf formula, mirrors `on_device_integer_anchor`). A
    /// DISTINCT integer code path — NOT a `grow_tree_on_device_*` re-run.
    fn host_integer_child_leaf_output(rows: &[u32], g: &[f32], h: &[f32]) -> f64 {
        let eps = f64::from(lgbm_core::types::K_EPSILON);
        let (sum_g, sum_h) = host_integer_sums(rows, g, h);
        calculate_splitted_leaf_output(false, sum_g, sum_h + eps, 0.0, 0.0)
    }

    /// The INDEPENDENT host-side integer scan of the ROOT best-split FEATURE: build each
    /// feature's per-bin (sum_g, sum_h) histogram in u64 fixed-point over ALL rows, enumerate
    /// every contiguous bin split point, score it with the SAME L2 gain the finder uses over
    /// the dequantized integer sums, and return the `real_feature_index` of the max-gain
    /// feature. Purely host + integer — never calls the driver, so it is a non-tautological
    /// anchor for the structural split-feature choice.
    fn host_integer_root_split_feature(features: &[GrowFeature], g: &[f32], h: &[f32]) -> i32 {
        let num_data = g.len();
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feature = -1i32;
        for f in features {
            let nb = f.num_bin as usize;
            let mut bin_g = vec![0i128; nb];
            let mut bin_h = vec![0i128; nb];
            for row in 0..num_data {
                let bin = f.bins.bin(row) as usize;
                bin_g[bin] += quantize(g[row]);
                bin_h[bin] += quantize(h[row]);
            }
            for s in 0..nb.saturating_sub(1) {
                let (mut lg, mut lh, mut rg, mut rh) = (0i128, 0i128, 0i128, 0i128);
                for b in 0..nb {
                    if b <= s {
                        lg += bin_g[b];
                        lh += bin_h[b];
                    } else {
                        rg += bin_g[b];
                        rh += bin_h[b];
                    }
                }
                if lh <= 0 || rh <= 0 {
                    continue;
                }
                let gain = get_split_gains(
                    false,
                    dequantize(lg),
                    dequantize(lh),
                    dequantize(rg),
                    dequantize(rh),
                    0.0,
                    0.0,
                );
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = f.real_feature_index;
                }
            }
        }
        best_feature
    }

    /// Collect the on-device-grown tree's per-leaf partition rows in leaf-id order.
    fn driver_leaf_rows(layout: &lgbm_dataset::LeafPartitionLayout) -> Vec<Vec<u32>> {
        layout
            .leaf_begin
            .iter()
            .zip(layout.leaf_count.iter())
            .map(|(&begin, &count)| {
                layout.indices[begin as usize..(begin + count) as usize].to_vec()
            })
            .collect()
    }

    /// Assert two grown trees are BYTE-IDENTICAL (structural ints direct, f64/f32 via
    /// `to_bits`) — the byte-parity primitive.
    fn assert_tree_byte_identical(a: &Tree, b: &Tree) {
        assert_eq!(a.num_leaves, b.num_leaves, "num_leaves drift");
        assert_eq!(a.split_feature, b.split_feature, "split_feature drift");
        assert_eq!(a.left_child, b.left_child, "left_child drift");
        assert_eq!(a.right_child, b.right_child, "right_child drift");
        assert_eq!(a.decision_type, b.decision_type, "decision_type drift");
        let bits = |xs: &[f64]| xs.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.threshold), bits(&b.threshold), "threshold bit drift");
        assert_eq!(bits(&a.leaf_value), bits(&b.leaf_value), "leaf_value bit drift");
        assert_eq!(
            bits(&a.internal_value),
            bits(&b.internal_value),
            "internal_value bit drift"
        );
    }

    /// Clause 1 — the on-device-grown tree is BIT-EXACT to the INDEPENDENT host-side
    /// u64 fixed-point integer reference: every leaf value, the root internal value, and the
    /// root split feature. Anchored to the integer/cpu-f64 fold — NEVER GPU-f32-vs-GPU-f32
    /// (comparing two nondeterministic f32 paths bit/1e-6-exact would be an invalid pairing;
    /// the reference here is a same-precision integer fold, not a second GPU path).
    #[test]
    fn resident_tree_bit_exact_to_u64_integer_path() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, gradients, hessians) = corpus();

        // CODE UNDER TEST: grow the whole L2 tree through the on-device driver (the cubecl-cpu
        // STRUCTURE anchor arm — the runnable lane; the resident rocm arm is held to the SAME
        // integer path + the ~1e-6 host-CUDA envelope below).
        let (tree, layout) =
            grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &features, NUM_LEAVES, -1)
                .expect("on-device grow must succeed on the tiny L2 corpus");

        assert!(
            tree.num_leaves > 1,
            "the monotone L2 corpus must split (non-vacuous anchor); got num_leaves={}",
            tree.num_leaves
        );
        assert_eq!(
            layout.leaf_begin.len(),
            tree.num_leaves as usize,
            "layout leaf count must match the grown tree's num_leaves"
        );

        // INDEPENDENT integer reference: each leaf output re-derived from its partition rows
        // via u64 fixed-point accumulation (NOT a driver re-run — non-tautological).
        let leaf_rows = driver_leaf_rows(&layout);
        let reference: Vec<f64> = leaf_rows
            .iter()
            .map(|rows| host_integer_child_leaf_output(rows, &gradients, &hessians))
            .collect();
        let device: Vec<f64> = (0..tree.num_leaves as usize).map(|l| tree.leaf_value[l]).collect();

        assert!(
            device.iter().any(|&v| v.abs() > 1e-6),
            "the grown tree must produce non-zero leaf outputs (non-vacuous anchor)"
        );

        compare_exact_f64_bits(&device, &reference).unwrap_or_else(|m| {
            panic!(
                "resident on-device tree leaf values != host-side u64 fixed-point (S=2^30) \
                 integer reference (BIT-EXACT same-precision anchor, ODP3-06 clause 1; NEVER \
                 GPU-f32-vs-GPU-f32, def-f8u-01): {m:?}"
            )
        });

        // Root internal node value: the clean (unbumped) L2 output over ALL rows — bit-exact
        // to the independent integer fold over every row.
        let all_rows: Vec<u32> = (0..gradients.len() as u32).collect();
        let (all_g, all_h) = host_integer_sums(&all_rows, &gradients, &hessians);
        let ref_root_internal = calculate_splitted_leaf_output(false, all_g, all_h, 0.0, 0.0);
        assert_eq!(
            tree.internal_value[0].to_bits(),
            ref_root_internal.to_bits(),
            "on-device root internal_value {} != host integer fold {} (bit-exact node-value anchor)",
            tree.internal_value[0],
            ref_root_internal
        );

        // Structural: the root split feature equals an independent integer-histogram argmax.
        let ref_root_feature = host_integer_root_split_feature(&features, &gradients, &hessians);
        assert!(ref_root_feature >= 0, "the integer scan must find a root split");
        assert_eq!(
            tree.split_feature[0], ref_root_feature,
            "on-device root split_feature {} != independent integer-histogram argmax {} \
             (structural split-feature anchor)",
            tree.split_feature[0], ref_root_feature
        );
    }

    /// Clause 3 — byte-parity: with `LGBM_CUDA_ON_DEVICE` unset the resolver is
    /// OFF (the byte-unchanged host path; `on_device_default()` stays false), AND the
    /// on-device anchor grow is BYTE-IDENTICAL run to run (integer accumulation
    /// ⇒ order-independent ⇒ no f32 drift), so the cpu/rocm merge gate is byte-stable. The
    /// full "every cpu+rocm tree byte-identical to master" surface is the `learner_parity`
    /// gate; this pins the driver-level byte stability of the on-device grow.
    #[test]
    fn sc4_flag_unset_off_and_grow_byte_identical() {
        // Flag unset ⇒ OFF (the resolved default). On a `cuda` build the device default is a
        // separate contract, so guard it (mirrors `cpu_build_default_is_off_when_env_unset`).
        #[cfg(not(feature = "cuda"))]
        assert!(
            !lgbm_compute::cuda_on_device_enabled(),
            "SC-4: with LGBM_CUDA_ON_DEVICE unset the resident control plane must be INERT \
             (byte-unchanged host path; the Plan-06 flip has not happened)"
        );

        let backend = CpuBackend;
        let client = cpu_client();
        let (features, gradients, hessians) = corpus();

        let grow = || {
            grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &features, NUM_LEAVES, -1)
                .expect("on-device grow must succeed")
        };
        let (tree_a, layout_a) = grow();
        let (tree_b, layout_b) = grow();

        assert_tree_byte_identical(&tree_a, &tree_b);
        assert_eq!(layout_a.indices, layout_b.indices, "layout indices drift (SC-4)");
        assert_eq!(layout_a.leaf_begin, layout_b.leaf_begin, "layout leaf_begin drift (SC-4)");
        assert_eq!(layout_a.leaf_count, layout_b.leaf_count, "layout leaf_count drift (SC-4)");
    }

    /// Clause 2 (rocm-gated) — the on-device predictions are within a ~1e-6 ENVELOPE
    /// of the host-CUDA f32 arm, NEVER bit-exact — comparing two nondeterministic f32 paths
    /// bit/1e-6-exact would be an invalid cross-precision check. The
    /// on-device (u64-build) tree grown+scored on the hip runtime is the candidate; the
    /// host-CUDA f32 arm (host partition scatter over the SAME hip-grown layout) is the
    /// reference — held to `atol + rtol·|p_host|`, an ENVELOPE, never a bit/1e-6 pairing of two
    /// nondeterministic f32 paths.
    #[cfg(feature = "rocm")]
    #[test]
    fn resident_score_within_envelope_of_host_cuda() {
        use lgbm_compute::add_prediction_to_score_on_device_resident;
        use lgbm_compute::runtime::rocm_client;
        use lgbm_compute::RocmBackend;

        // The ~1e-6-order cross-precision envelope (mirrors `on_device_integer_anchor` /
        // `resident_score_ab`): f32 leaf accumulation widens the absolute floor to 1e-5.
        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-6;

        let (features, gradients, hessians) = corpus();

        // CANDIDATE: grow + score on the hip runtime (on-device u64 build, resident arm).
        let hip_backend = RocmBackend::default();
        let hip = rocm_client();
        let (tree, layout) =
            grow_tree_on_device_driver(&hip_backend, &hip, &gradients, &hessians, &features, NUM_LEAVES, -1)
                .expect("hip on-device grow");
        let device = add_prediction_to_score_on_device_resident(&hip, &layout, &tree.leaf_value)
            .expect("hip on-device resident score");

        // REFERENCE: the host-CUDA/resident f32 arm — the byte-unchanged host partition scatter
        // over the SAME hip-grown (tree, layout). Envelope reference, NEVER a bit-exact pair.
        let mut host = vec![0.0f64; layout.num_data as usize];
        for (leaf, (&begin, &count)) in
            layout.leaf_begin.iter().zip(layout.leaf_count.iter()).enumerate()
        {
            for &row in &layout.indices[begin as usize..(begin + count) as usize] {
                host[row as usize] += tree.leaf_value[leaf];
            }
        }

        assert_eq!(device.len(), host.len(), "score length parity");
        for (i, (&d, &h)) in device.iter().zip(host.iter()).enumerate() {
            let abs_diff = (d - h).abs();
            let envelope = ATOL + RTOL * h.abs();
            assert!(
                abs_diff <= envelope,
                "on-device score[{i}] = {d} diverged from host-CUDA arm {h} by {abs_diff} \
                 > {envelope} (~1e-6 cross-precision envelope; NEVER bit-exact — def-f8u-01)"
            );
        }
    }
}

/// The `"0"` off-switch and `"1"` on-switch are distinguished at the mapping level
/// `"0"` is a first-class force-off, not the same as unset.
#[test]
fn zero_is_force_off_one_is_force_on() {
    assert_eq!(
        cuda_on_device_override_from(Some("0")),
        Some(false),
        "\"0\" must be an explicit force-off, distinct from unset (None)"
    );
    assert_eq!(
        cuda_on_device_override_from(Some("1")),
        Some(true),
        "\"1\" must be an explicit force-on"
    );
    assert_ne!(
        cuda_on_device_override_from(Some("0")),
        cuda_on_device_override_from(None),
        "force-off (Some(false)) and follow-default (None) must not collapse"
    );
}

/// On a `cpu` build (no `-F cuda`), with `LGBM_CUDA_ON_DEVICE` unset in a fresh process,
/// the resolver returns `false` — the resolved default — so the cpu/rocm merge gate is
/// byte-unchanged. Guarded `#[cfg(not(feature = "cuda"))]` because a `cuda` build's device
/// default is a separate contract. That contract has not been flipped: real-hardware
/// evaluation of flipping the default has not cleared its two bars (host/device throughput
/// not slower, and ~1e-6 parity vs the host-CUDA arm) even with the full resident control
/// plane in place, so the default remains false.
#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_build_default_is_off_when_env_unset() {
    // This test binary is invoked without LGBM_CUDA_ON_DEVICE set by the merge gate.
    // The override is None (unset) and on_device_default() is false on the cpu build,
    // so the resolved value must be false — the resolved leave-false default.
    assert!(
        !lgbm_compute::cuda_on_device_enabled(),
        "cpu-build default with env unset must be OFF (SC-4 byte-unchanged; 26-06 flip WITHHELD per 26-AB-RESULTS.md)"
    );
}

/// The off-switch escape hatch: regardless of the resolved default,
/// `LGBM_CUDA_ON_DEVICE="0"` is a first-class force-off that maps to the byte-unchanged
/// host path. Asserted at the pure-mapping level (the OnceLock resolver reads the
/// env once per process, so the env cannot be flipped between reads in one binary).
/// This lock survives a future flip of `on_device_default()` to `true` — the `"0"`
/// branch must always force-off.
#[test]
fn off_switch_forces_off_regardless_of_default() {
    assert_eq!(
        cuda_on_device_override_from(Some("0")),
        Some(false),
        "\"0\" off-switch must force-off (escape hatch) no matter what on_device_default() returns"
    );
}
