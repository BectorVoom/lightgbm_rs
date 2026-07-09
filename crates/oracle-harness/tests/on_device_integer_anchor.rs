//! Re-anchor the on-device parity gate to a valid, SAME-PRECISION integer reference.
//!
//! # Why this file exists
//!
//! Comparing the on-device **f64-deterministic** predictions against host-CUDA
//! **f32-nondeterministic** predictions directly is an INVALID cross-precision pairing:
//! holding two DIFFERENT-precision GPU paths to bit- (or 1e-6) equality is a false-fail
//! mode — a split near-tie flips under f32 rounding and cascades to a large divergence in
//! raw-logit space.
//!
//! The on-device histogram BUILD is a parallel **u64 fixed-point** kernel (S = 2^30).
//! Integer accumulation is ORDER-INDEPENDENT ⇒ EXACTLY reproducible, which makes a genuine
//! same-precision bit-exact anchor possible. This file installs a two-tier gate:
//!
//!   Tier 1 (HERE, cpu lane): the on-device-grown tree is BIT-EXACT to an
//!     INDEPENDENT **host-side integer** reference — a u64/i128 fixed-point (S = 2^30)
//!     recomputation of the leaf outputs over the grow's OWN partition, plus an
//!     independent integer-histogram argmax of the root split feature. The reference is
//!     NOT a second `grow_tree_on_device_*` call (that would be the verify-vacuity
//!     landmine — comparing the driver to itself proves nothing).
//!
//!     Coverage scope: on the cpu lane `CpuBackend`'s
//!     `resident_pool_supported()` is the trait-default `false`, so the CODE UNDER TEST
//!     here is the f64 ANCHOR arm (`build_leaf_hist` → `construct_histograms_f64_on`), NOT
//!     the resident **u64** kernel (`build_resident_leaf`). This tier therefore anchors the
//!     f64 STRUCTURE + integer leaf-value EQUIVALENCE — it does NOT exercise the u64 build.
//!     The u64 build itself is reachable only on the `#[cfg(feature = "rocm")]` lanes
//!     (Tier 2 below + `resident_score_ab.rs` + `cuda_on_device.rs`); a regression confined
//!     to the u64 kernel would pass every non-rocm run. (A resident-capable cubecl-cpu
//!     backend variant that could drive `build_resident_leaf` on the cpu lane was left out
//!     of scope as too risky for the bit-exact hard gate.)
//!
//!   Tier 2 (below + `resident_score_ab.rs`, rocm lane): the on-device predictions
//!     are held to a ~1e-6 ENVELOPE vs the host-CUDA arm — a cross-precision check, NEVER
//!     bit-exactness — so the invalid-pairing false-fail is structurally impossible.
//!
//! The cpu-f64 fold stays the bit-exact hard merge gate for the CPU anchor (untouched).
//!
//! # Scope
//!
//! The L2 continuous-feature proving slice (identity-binned, numeric splits, no missing
//! values, no RenewTreeOutput) — the regime where the on-device driver structure is the
//! cubecl-cpu f64 STRUCTURE anchor and where, because grad/hess are INTEGER-valued, the
//! u64 fixed-point sum dequantizes BIT-EXACTLY to the f64 fold (|grad| ≤ 13, hess = 1, ≤ 8
//! rows ⇒ every partial sum is an exactly-representable f64 integer, so addition is
//! order-independent — the same property that makes the u64 build bit-exact).

use lgbm_compute::gain::{calculate_splitted_leaf_output, get_split_gains};
use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_model::Tree;
use oracle_harness::comparator::compare_exact_f64_bits;

/// The u64 fixed-point scale the on-device build quantizes grad/hess at (S = 2^30,
/// `construct_leaf_hist_resident_lds_kernel_u64`). The host-side integer reference below
/// mirrors it: `q = round(v * S)` accumulated as `i128` (order-independent), dequantized
/// `q as f64 / S`. On the integer-valued proving corpus this is bit-exact to the f64 fold.
const FIXED_POINT_SCALE: f64 = (1u64 << 30) as f64;

/// A small identity-binned L2 continuous corpus (bin == raw value, no missing values) —
/// a field-for-field mirror of the `resident_score_ab` L2 slice. Feature 0 carries a
/// strong monotone signal (labels rise with `f0`) so the root splits on it; feature 1 is
/// a weak secondary. Returns `(features, labels)`. Gradients are `-label` (L2 from a zero
/// init) and are INTEGER-valued, which is what makes the u64 fixed-point sum dequantize
/// bit-exactly to the f64 fold.
fn corpus() -> (Vec<lgbm_treelearner::FeatureColumn>, Vec<f32>) {
    use lgbm_treelearner::FeatureColumn;
    let f0_bins = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
    let f1_bins = vec![0u32, 1, 0, 1, 0, 1, 0, 1];
    let labels = vec![1.0f32, 1.0, 3.0, 3.0, 8.0, 8.0, 12.0, 13.0];

    let f0 = FeatureColumn {
        bins: BinColumn::new(f0_bins, 4),
        num_bin: 4,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 3,
        default_bin: 4,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
        real_feature_index: 0,
        ..Default::default()
    };
    let f1 = FeatureColumn {
        bins: BinColumn::new(f1_bins, 2),
        num_bin: 2,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 1,
        default_bin: 2,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5],
        real_feature_index: 1,
        ..Default::default()
    };
    (vec![f0, f1], labels)
}

/// Build the on-device grow driver's `GrowFeature` carriers from [`corpus`] — the exact
/// field-by-field mirror `SerialTreeLearner`'s on_device_eligible block builds at the
/// seam (config-default categorical scalars, inert on this numeric slice).
fn grow_features() -> Vec<lgbm_compute::GrowFeature> {
    let (features, _labels) = corpus();
    features
        .iter()
        .map(|f| lgbm_compute::GrowFeature {
            bins: f.bins.clone(),
            num_bin: f.num_bin,
            offset: f.offset,
            min_bin: f.min_bin,
            max_bin: f.max_bin,
            default_bin: f.default_bin,
            most_freq_bin: f.most_freq_bin,
            missing_type: f.missing_type,
            bin_upper_bound: f.bin_upper_bound.clone(),
            real_feature_index: f.real_feature_index,
            bin_type: f.bin_type,
            bin_to_category: f.bin_to_category.clone(),
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            max_cat_to_onehot: 4,
            min_data_per_group: 100,
        })
        .collect()
}

/// L2 gradients (`-label`) and unit hessians for the corpus.
fn grad_hess(labels: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let gradients: Vec<f32> = labels.iter().map(|&l| -l).collect();
    let hessians: Vec<f32> = vec![1.0f32; labels.len()];
    (gradients, hessians)
}

/// Quantize an f32 value to the u64 fixed-point domain (S = 2^30), returned as `i128`
/// so a leaf's accumulation is exact and order-independent (mirrors the on-device u64
/// kernel's per-lane `round(v * S)` + integer atomic add).
fn quantize(v: f32) -> i128 {
    (f64::from(v) * FIXED_POINT_SCALE).round() as i128
}

/// Dequantize a fixed-point `i128` accumulation back to f64 (`q / S`).
fn dequantize(q: i128) -> f64 {
    q as f64 / FIXED_POINT_SCALE
}

/// The INDEPENDENT host-side integer reference for a CHILD leaf's output: accumulate the
/// leaf's grad/hess in u64 fixed-point (S = 2^30) over `rows`, dequantize, and apply the
/// L2 closed-form the finder uses — `-sum_g / (sum_h + kEpsilon)`. The `+kEpsilon` is the
/// reverse-scan hessian seed (`feature_histogram.hpp:856`): on the numeric proving slice
/// (single REVERSE scan, no forward preamble, no default-bin skip) BOTH the left and the
/// right child of every split carry exactly ONE `kEpsilon` in the hessian used for the
/// output (`split.rs:359-370` — the eps is subtracted back off only for the REPORTED
/// `left/right_sum_hessian` at cells 6/8, NOT for the output at cells 10/11). Every final
/// leaf in a split tree is such a child, so this is the universal child-leaf formula. This
/// is a DISTINCT integer-histogram code path (NOT a `grow_tree_on_device_*` re-run): it
/// re-derives the leaf value from the raw rows via fixed-point integers.
fn host_integer_child_leaf_output(rows: &[u32], gradients: &[f32], hessians: &[f32]) -> f64 {
    let eps = f64::from(lgbm_core::types::K_EPSILON);
    let (sum_g, sum_h) = host_integer_sums(rows, gradients, hessians);
    calculate_splitted_leaf_output(false, sum_g, sum_h + eps, 0.0, 0.0)
}

/// Dequantized (sum_g, sum_h) over `rows` via u64 fixed-point (S = 2^30) accumulation —
/// the shared integer fold. On integer-valued grad + unit hess this is bit-exact to the
/// f64 fold (order-independent).
fn host_integer_sums(rows: &[u32], gradients: &[f32], hessians: &[f32]) -> (f64, f64) {
    let mut q_g: i128 = 0;
    let mut q_h: i128 = 0;
    for &r in rows {
        q_g += quantize(gradients[r as usize]);
        q_h += quantize(hessians[r as usize]);
    }
    (dequantize(q_g), dequantize(q_h))
}

/// The INDEPENDENT host-side integer scan of the ROOT best-split FEATURE: build each
/// feature's per-bin (sum_g, sum_h) histogram in u64 fixed-point over ALL rows, enumerate
/// every contiguous bin split point, score it with the SAME L2 gain the finder uses
/// (`get_split_gains`, `l1 = l2 = 0`) over the dequantized integer sums, and return the
/// `real_feature_index` of the max-gain feature. Purely host + integer — it never calls
/// the driver, so it is a valid non-tautological anchor for "which feature the root split
/// on". (The exact gain VALUE / threshold convention is intentionally NOT reproduced here
/// — the leaf-output bit-exact gate above already pins the numeric leaf values; this pins
/// the STRUCTURAL split-feature choice against an integer histogram.)
fn host_integer_root_split_feature(
    features: &[lgbm_compute::GrowFeature],
    gradients: &[f32],
    hessians: &[f32],
) -> i32 {
    let num_data = gradients.len();
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_feature = -1i32;
    for f in features {
        let nb = f.num_bin as usize;
        let mut bin_g = vec![0i128; nb];
        let mut bin_h = vec![0i128; nb];
        for row in 0..num_data {
            let bin = f.bins.bin(row) as usize;
            bin_g[bin] += quantize(gradients[row]);
            bin_h[bin] += quantize(hessians[row]);
        }
        // Enumerate every contiguous split point s: left = bins [0..=s], right = (s..].
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
            // Both sides must be non-empty (hess = count > 0) for a real split.
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

/// Collect the on-device-grown tree's per-leaf outputs in leaf-id order (`tree.leaf_value`
/// is indexed by leaf id; the layout's `leaf_begin/leaf_count` are in the same leaf-id
/// order the driver appends them — see `grow_tree_on_device_driver`'s layout assembly).
fn driver_leaf_rows(layout: &lgbm_dataset::LeafPartitionLayout) -> Vec<Vec<u32>> {
    layout
        .leaf_begin
        .iter()
        .zip(layout.leaf_count.iter())
        .map(|(&begin, &count)| layout.indices[begin as usize..(begin + count) as usize].to_vec())
        .collect()
}

/// The on-device-grown tree's leaf values are BIT-EXACT to the
/// INDEPENDENT host-side u64 fixed-point (S = 2^30) integer reference, and the root split
/// feature matches an independent integer-histogram argmax.
///
/// Coverage scope: this test runs on `CpuBackend`, whose
/// `resident_pool_supported()` is the trait-default `false`, so the driver takes the ANCHOR
/// arm (`build_leaf_hist` → `construct_histograms_f64_on`, f64) — NOT the resident u64 build
/// (`build_resident_leaf` → `..._u64`). What it anchors on the cpu lane is therefore the f64
/// STRUCTURE + the integer leaf-value EQUIVALENCE, not the u64 kernel itself. Bit-exactness
/// holds because, on this integer-valued corpus, BOTH the f64 fold and the i128/u64
/// fixed-point fold equal the exact integer sums (order-independent). The u64 resident kernel
/// is exercised only on the `#[cfg(feature = "rocm")]` lanes (see `resident_score_ab.rs` /
/// `cuda_on_device.rs`).
///
/// This is still a valid, non-tautological, SAME-precision anchor the integer build makes
/// possible: the reference re-derives the leaf outputs and the root split feature from the
/// raw rows via fixed-point INTEGER accumulation — NOT by a second `grow_tree_on_device_*`
/// call, and NOT against the f32-nondeterministic host-CUDA path (that envelope lives in
/// the section below).
#[test]
fn on_device_tree_leaf_values_bit_exact_to_host_integer_reference() {
    let backend = CpuBackend;
    let client = cpu_client();
    let (_features, labels) = corpus();
    let (gradients, hessians) = grad_hess(&labels);
    let gf = grow_features();

    // CODE UNDER TEST: grow the whole L2 tree on device (cubecl-cpu STRUCTURE anchor).
    let (tree, layout) =
        grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &gf, 4, -1)
            .expect("on-device grow");

    // The monotone corpus must actually split (a never-split root would make the anchor
    // vacuous — the leaf-value comparison would be a single trivial cell).
    assert!(
        tree.num_leaves > 1,
        "the monotone L2 corpus must split (non-trivial integer anchor); got num_leaves={}",
        tree.num_leaves
    );
    assert_eq!(
        layout.leaf_begin.len(),
        tree.num_leaves as usize,
        "layout leaf count must match the grown tree's num_leaves"
    );

    // INDEPENDENT integer reference: each leaf's output re-derived from its partition rows
    // via u64 fixed-point accumulation (NOT a driver re-run).
    let leaf_rows = driver_leaf_rows(&layout);
    let reference: Vec<f64> = leaf_rows
        .iter()
        .map(|rows| host_integer_child_leaf_output(rows, &gradients, &hessians))
        .collect();
    let device: Vec<f64> = (0..tree.num_leaves as usize)
        .map(|leaf| tree.leaf_value[leaf])
        .collect();

    // Sanity: the tree moved (non-trivial leaf spread) — otherwise the anchor is vacuous.
    assert!(
        device.iter().any(|&v| v.abs() > 1e-6),
        "the grown tree must produce non-zero leaf outputs (non-vacuous anchor)"
    );

    compare_exact_f64_bits(&device, &reference).unwrap_or_else(|m| {
        panic!(
            "on-device tree leaf values != host-side integer (u64 fixed-point S=2^30) \
             reference (BIT-EXACT same-precision anchor, def-f8u-01 tier 1); the cpu-lane \
             f64 build (CpuBackend anchor arm) diverged from the independent integer fold on \
             the L2 slice — see WR-02: the u64 kernel itself is covered on the rocm lane: {m:?}"
        )
    });

    // NODE VALUE: the root internal node's value is the clean (unbumped) L2 output over
    // ALL rows — `internal_value[0] = calculate_splitted_leaf_output(root_sum_g,
    // root_sum_h)` (grow_driver root seed, NO kEpsilon) — bit-exact to the independent
    // integer fold over every row.
    let (all_g, all_h) = host_integer_sums(
        &(0..gradients.len() as u32).collect::<Vec<_>>(),
        &gradients,
        &hessians,
    );
    let ref_root_internal = calculate_splitted_leaf_output(false, all_g, all_h, 0.0, 0.0);
    assert_eq!(
        tree.internal_value[0].to_bits(),
        ref_root_internal.to_bits(),
        "on-device root internal_value {} != host integer fold {} (bit-exact node-value \
         anchor); device_bits={:#018x} ref_bits={:#018x}",
        tree.internal_value[0],
        ref_root_internal,
        tree.internal_value[0].to_bits(),
        ref_root_internal.to_bits()
    );

    // STRUCTURAL: the root split feature equals an independent integer-histogram argmax.
    let ref_root_feature = host_integer_root_split_feature(&gf, &gradients, &hessians);
    assert!(
        ref_root_feature >= 0,
        "the integer scan must find a root split"
    );
    assert_eq!(
        tree.split_feature[0], ref_root_feature,
        "on-device root split_feature {} != independent integer-histogram argmax {} \
         (structural split-feature anchor)",
        tree.split_feature[0], ref_root_feature
    );
}

/// Determinism: growing the SAME corpus twice on device yields a
/// BYTE-IDENTICAL tree (integer accumulation ⇒ order-independent ⇒ exactly reproducible),
/// proving the integer anchor is stable (no run-to-run drift the ~1e-6 f32 path would
/// have). Compares every numeric field bit-for-bit (`to_bits`) plus the structural fields
/// and the row→leaf layout.
#[test]
fn on_device_grow_is_deterministic() {
    let backend = CpuBackend;
    let client = cpu_client();
    let (_features, labels) = corpus();
    let (gradients, hessians) = grad_hess(&labels);
    let gf = grow_features();

    let (tree_a, layout_a) =
        grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &gf, 4, -1)
            .expect("on-device grow (run A)");
    let (tree_b, layout_b) =
        grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &gf, 4, -1)
            .expect("on-device grow (run B)");

    assert_tree_byte_identical(&tree_a, &tree_b);

    assert_eq!(
        layout_a.num_data, layout_b.num_data,
        "layout num_data drift"
    );
    assert_eq!(layout_a.indices, layout_b.indices, "layout indices drift");
    assert_eq!(
        layout_a.leaf_begin, layout_b.leaf_begin,
        "layout leaf_begin drift"
    );
    assert_eq!(
        layout_a.leaf_count, layout_b.leaf_count,
        "layout leaf_count drift"
    );
}

/// Assert two grown trees are BYTE-IDENTICAL: structural integer fields compared directly,
/// f64/f32 fields compared via `to_bits` (bit-exact, not tolerance).
fn assert_tree_byte_identical(a: &Tree, b: &Tree) {
    assert_eq!(a.num_leaves, b.num_leaves, "num_leaves drift");
    assert_eq!(a.split_feature, b.split_feature, "split_feature drift");
    assert_eq!(a.left_child, b.left_child, "left_child drift");
    assert_eq!(a.right_child, b.right_child, "right_child drift");
    assert_eq!(a.decision_type, b.decision_type, "decision_type drift");

    let bits = |xs: &[f64]| xs.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&a.threshold),
        bits(&b.threshold),
        "threshold bit drift"
    );
    assert_eq!(
        bits(&a.leaf_value),
        bits(&b.leaf_value),
        "leaf_value bit drift"
    );
    assert_eq!(
        bits(&a.internal_value),
        bits(&b.internal_value),
        "internal_value bit drift"
    );
    assert_eq!(
        a.split_gain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.split_gain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "split_gain bit drift"
    );
}

/// Collect the LEAF ids under a tree node (walking the `~leaf` child encoding — see
/// `lgbm-model/src/tree.rs`: `left_child`/`right_child` hold an internal node index when `>= 0`
/// and a leaf id `~child` when `< 0`). Called with an internal node index (`>= 0`) at the top.
fn collect_leaves_under(tree: &Tree, node_or_leaf: i32, out: &mut Vec<usize>) {
    if node_or_leaf < 0 {
        out.push((!node_or_leaf) as usize); // leaf id = ~child
    } else {
        let n = node_or_leaf as usize;
        collect_leaves_under(tree, tree.left_child[n], out);
        collect_leaves_under(tree, tree.right_child[n], out);
    }
}

/// The rewritten resident-scheduler tree is BIT-EXACT to the u64
/// fixed-point INTEGER anchor path across the WHOLE tree (every internal node value + every leaf
/// value + the root split feature), NOT just the root+children the `_host_integer_reference` gate
/// pins. This consolidates the anchor-parity gate onto the same-precision integer reference
/// the u64 build makes possible.
///
/// Coverage scope (identical to `_host_integer_reference`): on the cpu lane
/// `CpuBackend::resident_pool_supported() == false`, so the CODE UNDER TEST is the f64 ANCHOR arm
/// (the RESIDENT u64 kernel that the resident scheduler drives runs ONLY on a resident backend, and
/// cannot execute on cubecl-cpu — `atomic<u64>` is unimplemented there). This lane therefore
/// anchors the f64 STRUCTURE + the integer leaf/node-value EQUIVALENCE; the ACTUAL resident
/// scheduler is held to the ~1e-6 envelope vs this same anchor on the `#[cfg(feature = "rocm")]`
/// lane (below + `resident_score_ab.rs` hip module) — NEVER GPU-f32-vs-GPU-f32.
///
/// Bit-exactness holds because, on the integer-valued L2 proving corpus, BOTH the f64 fold and the
/// i128/u64 fixed-point fold equal the exact integer sums (order-independent). The reference
/// re-derives every node/leaf output from the raw rows via fixed-point INTEGER accumulation — NOT
/// a second `grow_tree_on_device_*` call (the verify-vacuity landmine). The order-preserving
/// cross-feature tie-break is anchor-defined: the winner selection the driver records is what the
/// integer reference re-scores, and the near-tie-free monotone corpus makes the choice unambiguous.
#[test]
fn resident_tree_bit_exact_to_u64_integer_path() {
    let backend = CpuBackend;
    let client = cpu_client();
    let (_features, labels) = corpus();
    let (gradients, hessians) = grad_hess(&labels);
    let gf = grow_features();

    let (tree, layout) =
        grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &gf, 4, -1)
            .expect("on-device grow");

    // A multi-leaf tree (root + ≥2 internal nodes) so the full-tree diff is non-vacuous.
    assert!(
        tree.num_leaves > 2,
        "the monotone L2 corpus must grow a multi-node tree (root + internal nodes) for a \
         non-vacuous full-tree integer anchor; got num_leaves={}",
        tree.num_leaves
    );

    // (1) EVERY LEAF VALUE bit-exact to the u64 integer child-leaf fold over its partition rows.
    let leaf_rows = driver_leaf_rows(&layout);
    let leaf_reference: Vec<f64> = leaf_rows
        .iter()
        .map(|rows| host_integer_child_leaf_output(rows, &gradients, &hessians))
        .collect();
    let leaf_device: Vec<f64> = (0..tree.num_leaves as usize)
        .map(|l| tree.leaf_value[l])
        .collect();
    compare_exact_f64_bits(&leaf_device, &leaf_reference).unwrap_or_else(|m| {
        panic!(
            "resident-path tree leaf values != host u64 integer (S=2^30) fold (bit-exact \
             same-precision anchor, def-f8u-01 tier 1): {m:?}"
        )
    });

    // (2) EVERY INTERNAL NODE VALUE bit-exact. `internal_value[node] = leaf_value[leaf]` at the
    // moment that leaf split (tree.rs:231/343): the ROOT node (index 0) carries the CLEAN L2 output
    // over all its rows (NO kEpsilon — the grow-driver root seed), and every NON-ROOT internal node
    // carries the kEpsilon-bearing CHILD output (every non-root node was a child leaf first). So the
    // reference per node is the integer fold over that node's rows, eps-less at the root and
    // eps-bearing elsewhere — re-derived independently from the raw rows.
    let eps = f64::from(lgbm_core::types::K_EPSILON);
    for node in 0..(tree.num_leaves as usize - 1) {
        let mut leaves_here = Vec::new();
        collect_leaves_under(&tree, node as i32, &mut leaves_here);
        let mut rows = Vec::new();
        for &leaf in &leaves_here {
            rows.extend_from_slice(&leaf_rows[leaf]);
        }
        let (sum_g, sum_h) = host_integer_sums(&rows, &gradients, &hessians);
        let node_reference = if node == 0 {
            calculate_splitted_leaf_output(false, sum_g, sum_h, 0.0, 0.0)
        } else {
            calculate_splitted_leaf_output(false, sum_g, sum_h + eps, 0.0, 0.0)
        };
        assert_eq!(
            tree.internal_value[node].to_bits(),
            node_reference.to_bits(),
            "internal node {node} value {} != host integer fold {} (bit-exact node-value anchor; \
             node0=no-eps root seed, node>0=kEpsilon child output); device_bits={:#018x} \
             ref_bits={:#018x}",
            tree.internal_value[node],
            node_reference,
            tree.internal_value[node].to_bits(),
            node_reference.to_bits()
        );
    }

    // (3) STRUCTURAL: the root split feature equals an independent integer-histogram argmax
    // (anchor-defined winner over the integer sums).
    let ref_root_feature = host_integer_root_split_feature(&gf, &gradients, &hessians);
    assert!(
        ref_root_feature >= 0,
        "the integer scan must find a root split"
    );
    assert_eq!(
        tree.split_feature[0], ref_root_feature,
        "root split_feature {} != independent integer-histogram argmax {}",
        tree.split_feature[0], ref_root_feature
    );

    // On a rocm host, additionally hold the ACTUAL resident-scheduler tree
    // (`RocmBackend::with_resident(true)`) to the ~1e-6 envelope vs THIS cpu-f64/integer anchor
    // (the anchor is the cpu fold, NEVER a second GPU f32 run).
    #[cfg(feature = "rocm")]
    resident_tree_within_envelope_of_integer_anchor(&gradients, &hessians, &gf, &leaf_reference);
}

/// `--features rocm` lane: the ACTUAL resident scheduler (`RocmBackend::with_resident(true)`)
/// grows the tree on the hip runtime; its per-leaf values are held to the ~1e-6 f32 envelope vs the
/// cpu-f64/u64-integer anchor `leaf_reference` (the anchor is the cpu fold, NEVER a
/// second GPU f32 run). This is the tree-structure companion to `resident_score_ab.rs`'s hip score
/// within-envelope variants.
#[cfg(feature = "rocm")]
fn resident_tree_within_envelope_of_integer_anchor(
    gradients: &[f32],
    hessians: &[f32],
    gf: &[lgbm_compute::GrowFeature],
    leaf_reference: &[f64],
) {
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::{Backend, RocmBackend};

    const ATOL: f64 = 1e-5; // f32 leaf-accumulation envelope (mirrors ROCM_LEAF_VALUE_TOL)
    const RTOL: f64 = 1e-6;

    let backend = RocmBackend::with_resident(true);
    assert!(
        backend.resident_pool_supported(),
        "with_resident(true) must drive the ACTUAL 28-03 resident scheduler"
    );
    let hip = rocm_client();
    let (tree, layout) = grow_tree_on_device_driver(&backend, &hip, gradients, hessians, gf, 4, -1)
        .expect("hip resident on-device grow");

    // Sort both leaf-value vectors so the envelope compare is order-independent (the resident
    // §8.3 leaf-order tie-break may index leaves differently than the anchor arm on an exact tie;
    // the corpus is near-tie-free so the MULTISET of leaf values matches within the f32 envelope).
    let mut device: Vec<f64> = (0..tree.num_leaves as usize)
        .map(|l| tree.leaf_value[l])
        .collect();
    let mut anchor: Vec<f64> = leaf_reference.to_vec();
    assert_eq!(
        device.len(),
        anchor.len(),
        "resident hip tree leaf count {} != anchor {} (structure diverged beyond the envelope)",
        device.len(),
        anchor.len()
    );
    device.sort_by(|a, b| a.partial_cmp(b).unwrap());
    anchor.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let _ = &layout;
    for (i, (&d, &a)) in device.iter().zip(anchor.iter()).enumerate() {
        let abs_diff = (d - a).abs();
        let envelope = ATOL + RTOL * a.abs();
        assert!(
            abs_diff <= envelope,
            "resident hip tree leaf[{i}] = {d} diverged from cpu-f64/integer anchor {a} by \
             {abs_diff} > {envelope} (~1e-6 cross-precision envelope; NEVER bit-exact — def-f8u-01)"
        );
    }
}

// ===========================================================================
// The ~1e-6 CROSS-PRECISION envelope vs the host-CUDA arm (rocm-gated).
//
// This is the second tier of the two-tier gate: the on-device predictions are held to an
// ENVELOPE `atol + rtol·|p_host|` against the host-CUDA/resident f32 arm — NEVER
// bit-exactness, so a near-tie false-fail is structurally impossible. Compiled out on the
// cpu-only lane; exercised on a rocm host. The bit-exact SAME-precision anchor lives in
// Tier 1 above; the combined grow+score ~1e-6 hip cell also lives in `resident_score_ab.rs`.
// ===========================================================================
#[cfg(feature = "rocm")]
mod rocm_envelope {
    use super::*;
    use lgbm_compute::add_prediction_to_score_on_device_resident;
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::RocmBackend;

    /// The ~1e-6-order cross-precision envelope. The CLAUDE.md ROCm bar is ~1e-6; f32 leaf
    /// accumulation widens the absolute floor to 1e-5 (mirrors `resident_score_ab`'s
    /// `ROCM_SCORE_TOL` and `learner_parity::ROCM_LEAF_VALUE_TOL`). Applied as
    /// `atol + rtol·|p_host|` — an ENVELOPE, never bit-exactness.
    const ATOL: f64 = 1e-5;
    const RTOL: f64 = 1e-6;

    /// The on-device predictions vs the host-CUDA/resident f32 arm within the
    /// ~1e-6 envelope. On-device (u64-build) tree grown + scored on the hip runtime is the
    /// candidate; the host-CUDA f32 arm (cpu-anchor host partition scatter over the
    /// hip-grown layout) is the reference. This holds the cross-precision pair to an
    /// ENVELOPE, never bit-exactness.
    /// NEVER compares the on-device tree bit-exact to a host-CUDA f32 path.
    #[test]
    fn on_device_score_within_envelope_of_host_cuda() {
        let (_features, labels) = corpus();
        let (gradients, hessians) = grad_hess(&labels);
        let gf = grow_features();

        // CANDIDATE: grow + score on the hip runtime (on-device u64 build).
        let hip_backend = RocmBackend::default();
        let hip = rocm_client();
        let (tree, layout) =
            grow_tree_on_device_driver(&hip_backend, &hip, &gradients, &hessians, &gf, 4, -1)
                .expect("hip on-device grow");
        let device = add_prediction_to_score_on_device_resident(&hip, &layout, &tree.leaf_value)
            .expect("hip on-device resident score");

        // REFERENCE: the host-CUDA/resident f32 arm — the byte-unchanged host partition
        // scatter over the SAME hip-grown (tree, layout). Never compared bit-exactly to the
        // candidate; here it is the ENVELOPE reference.
        let mut host = vec![0.0f64; layout.num_data as usize];
        for (leaf, (&begin, &count)) in layout
            .leaf_begin
            .iter()
            .zip(layout.leaf_count.iter())
            .enumerate()
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
