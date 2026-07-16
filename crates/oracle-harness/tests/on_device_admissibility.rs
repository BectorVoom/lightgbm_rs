//! The on-device scan admissibility gate.
//!
//! # What this proves
//!
//! C++'s `min_data_in_leaf` / `min_sum_hessian_in_leaf` admissibility gate
//! (`feature_histogram.hpp` FindBestThresholdSequentially) rejects near-empty splits
//! before they can win. If the on-device seam is pinned to a PERMISSIVE config
//! (`min_data_in_leaf = 1`, `min_sum_hessian_in_leaf = 0.0`), a ~1-row `cnt_factor`
//! estimate can pass `>= 1` and a kEpsilon-magnitude prefix can pass `>= 0.0`, minting a
//! near-zero-hessian denominator and an `inf`-gain winner. The production on-device seam
//! (`SerialTreeLearner` → `Backend::grow_tree_on_device` → `grow_tree_on_device_driver`)
//! must thread the learner's REAL `GainConfig` across the seam so LightGBM's default
//! admissibility (20 / 1e-3) binds on every on-device grow.
//!
//! # Lanes
//!
//! - **Lane A** (seam-level, cpu anchor — runs everywhere): grow a whole tree through the
//!   production on-device driver on a corpus engineered so the max-gain split isolates a
//!   `< min_data_in_leaf`-row micro-cluster. Admissibility is recomputed FROM THE DATA (the
//!   grow's own row→leaf partition + the raw hessians), NEVER from `SplitInfo.left_count`
//!   (a `cnt_factor` row-count estimate). Under the real config every leaf must be
//!   admissible.
//! - **Lane B** (unit, cpu anchor): drive the shared scan (`Backend::find_best_split`,
//!   `split_scan_body`) on a crafted histogram carrying a true-near-zero low-bin prefix
//!   plus real mass in the high bins. Under a permissive config the near-empty prefix can
//!   WIN with a sub-`min_sum_hessian` left child; under the real config the admissibility
//!   gate rejects it. Every returned `SplitInfo` is asserted to have a FINITE gain (`inf`
//!   impossible by construction) and admissible children (gated on the scanned HESSIAN
//!   SUMS, not the count estimate).

use lgbm_compute::gain::GainConfig;
use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver_with_cfg;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{Backend, BinColumn, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;

/// LightGBM's real defaults for the two admissibility knobs this gate is about
/// (`config.h`: `min_data_in_leaf = 20`, `min_sum_hessian_in_leaf = 1e-3`), L2 continuous.
/// This is the config the host learner binds and the config the FIXED on-device seam must
/// bind — NOT the permissive `proving_slice_config()` the pre-fix seam pinned.
fn realistic_cfg() -> GainConfig {
    GainConfig {
        min_data_in_leaf: 20,
        min_sum_hessian_in_leaf: 1e-3,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    }
}

/// A PERMISSIVE proving-slice config (`min_data_in_leaf = 1`, `min_sum_hessian_in_leaf =
/// 0.0`) — admissibility effectively OFF. Lane B feeds this to the scan to reproduce a
/// near-empty prefix passing the gate.
fn permissive_cfg() -> GainConfig {
    GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    }
}

// ===========================================================================
// Lane A — seam-level: grow through the production on-device driver, recompute
// admissibility from the DATA.
// ===========================================================================

/// A small deterministic L2 corpus with an extreme-gain micro-cluster of 10 rows.
///
/// Feature 0 has `num_bin = 10`. Bins 0..=8 hold 20 rows each (180 rows) with a monotone
/// label = bin index (so admissible interior splits exist). Bin 9 holds a 10-row
/// micro-cluster with an EXTREME label (1000) → the max-gain split isolates those 10 rows.
/// With `min_data_in_leaf = 20` the host/C++ learner REFUSES that split (child has 10 < 20
/// rows); the on-device tree must too. Gradients are `-label` (L2 from a zero init), hessians
/// are unit (so per-leaf hessian sum == row count, and `cnt_factor == 1` ⇒ the count estimate
/// equals the actual count — a clean min_data demonstration).
fn micro_cluster_corpus() -> (Vec<lgbm_treelearner::FeatureColumn>, Vec<f32>, Vec<f32>) {
    use lgbm_treelearner::{offset_for_most_freq_bin, FeatureColumn};

    let mut bins: Vec<u32> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    // Bulk: bins 0..=8, 20 rows each, monotone labels.
    for b in 0u32..9 {
        for _ in 0..20 {
            bins.push(b);
            labels.push(b as f32);
        }
    }
    // Micro-cluster: bin 9, 10 rows, extreme label.
    for _ in 0..10 {
        bins.push(9);
        labels.push(1000.0);
    }

    let f0 = FeatureColumn {
        bins: BinColumn::new(bins, 10),
        num_bin: 10,
        offset: offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 9,
        default_bin: 10,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        real_feature_index: 0,
        ..Default::default()
    };

    let gradients: Vec<f32> = labels.iter().map(|&l| -l).collect();
    let hessians: Vec<f32> = vec![1.0f32; labels.len()];
    (vec![f0], gradients, hessians)
}

/// Field-by-field mirror of the `on_device_eligible` block's `GrowFeature` build.
fn grow_features(features: &[lgbm_treelearner::FeatureColumn]) -> Vec<lgbm_compute::GrowFeature> {
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

/// Per-leaf actual row lists from the grow's own partition layout (leaf-id order).
fn driver_leaf_rows(layout: &lgbm_dataset::LeafPartitionLayout) -> Vec<Vec<u32>> {
    layout
        .leaf_begin
        .iter()
        .zip(layout.leaf_count.iter())
        .map(|(&begin, &count)| layout.indices[begin as usize..(begin + count) as usize].to_vec())
        .collect()
}

/// Grow the tree through the CONFIG-BOUND on-device driver entry — the exact fn the FIXED
/// production seam ([`Backend::grow_tree_on_device_with_cfg`]) delegates to, threading the
/// caller's REAL `GainConfig`.
///
/// Calls [`grow_tree_on_device_driver_with_cfg`] with the realistic config, so
/// `min_data_in_leaf` / `min_sum_hessian_in_leaf` bind on every scan. The parameterless
/// [`grow_tree_on_device_driver`] pins the permissive proving config instead — retained for
/// the anchor gates but NEVER used on a production path.
fn grow_via_production_seam(
    backend: &CpuBackend,
    gradients: &[f32],
    hessians: &[f32],
    gf: &[lgbm_compute::GrowFeature],
    num_leaves: i32,
    max_depth: i32,
    cfg: GainConfig,
) -> (lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout) {
    let client = cpu_client();
    grow_tree_on_device_driver_with_cfg(
        backend, &client, gradients, hessians, gf, num_leaves, max_depth, cfg,
    )
    .expect("on-device grow (config-bound production seam)")
}

/// Lane A — the on-device-grown tree honors `min_data_in_leaf` / `min_sum_hessian_in_leaf`
/// recomputed FROM THE DATA. Fails with a pinned permissive config (which would isolate the
/// 10-row micro-cluster into its own leaf); passes once the real config binds through the seam.
#[test]
fn on_device_seam_honors_min_data_in_leaf() {
    let backend = CpuBackend;
    let (features, gradients, hessians) = micro_cluster_corpus();
    let gf = grow_features(&features);
    let cfg = realistic_cfg();

    let (tree, layout) =
        grow_via_production_seam(&backend, &gradients, &hessians, &gf, 4, -1, cfg);

    // The corpus must split (else the admissibility check is vacuous).
    assert!(
        tree.num_leaves > 1,
        "corpus must split for a non-vacuous admissibility gate; got num_leaves={}",
        tree.num_leaves
    );

    // Recompute EVERY leaf's actual row count + hessian sum from the partition + raw data.
    // Every leaf is a terminal child of some split, so per-leaf admissibility ⇒ every split's
    // children are admissible (a child is a union of >= 1 leaf).
    let leaf_rows = driver_leaf_rows(&layout);
    for (leaf, rows) in leaf_rows.iter().enumerate() {
        let actual_n = rows.len() as i32;
        let actual_h: f64 = rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();
        assert!(
            actual_n >= cfg.min_data_in_leaf,
            "leaf {leaf} has {actual_n} actual rows < min_data_in_leaf {} — the on-device seam \
             admitted an inadmissible split (spike-072 item-15 fingerprint: the pinned permissive \
             proving config disabled C++ admissibility on the production on-device arm)",
            cfg.min_data_in_leaf
        );
        assert!(
            actual_h >= cfg.min_sum_hessian_in_leaf,
            "leaf {leaf} has actual hessian sum {actual_h} < min_sum_hessian_in_leaf {} — \
             inadmissible split admitted on the on-device seam",
            cfg.min_sum_hessian_in_leaf
        );
    }
}

// ===========================================================================
// Lane B — unit: the shared scan rejects a near-empty prefix, and never mints
// a non-finite gain.
// ===========================================================================

/// A crafted single-feature histogram (concatenated `[g,h]` cells) reproducing a
/// near-empty low-bin residue: bin 0 carries a tiny (`~0.36`-scale gradient, `1e-4` hessian)
/// near-empty prefix; bins 1..=3 carry real mass. `sum_hessian` / `num_data` are chosen so
/// `cnt_factor = num_data / sum_hessian ≈ 60000` ⇒ bin 0's row ESTIMATE `round(h0·cnt_factor)`
/// is 6 (passes `min_data_in_leaf = 1`, fails `= 20`) while its hessian `1e-4` fails
/// `min_sum_hessian_in_leaf = 1e-3` and passes `= 0.0`. Under the permissive config the
/// near-empty thr=0 split therefore WINS (its left gain `0.36²/1e-4 ≈ 1296` dominates the
/// real interior splits `~0.48`); under the real config it is rejected.
fn crafted_near_empty_histogram() -> (Vec<f64>, f64, f64, i32) {
    let hist = vec![
        0.36, 1e-4, // bin 0 — the near-empty phantom prefix
        1.0, 8.0, // bin 1
        1.0, 8.0, // bin 2
        1.0, 8.0, // bin 3
    ];
    let sum_gradient = 0.36 + 3.0;
    let sum_hessian = 1e-4 + 24.0;
    let num_data = 1_440_006; // ⇒ cnt_factor ≈ 60000 ⇒ bin-0 estimate = round(6.0) = 6
    (hist, sum_gradient, sum_hessian, num_data)
}

/// Run the shared scan (`Backend::find_best_split` → `split_scan_body`) on a histogram.
fn scan(cfg: &GainConfig, hist: &[f64], sum_g: f64, sum_h: f64, num_data: i32, num_bin: u32) -> lgbm_compute::gain::SplitInfo {
    let backend = CpuBackend;
    let client = cpu_client();
    backend
        .find_best_split(
            &client, hist, cfg, num_bin, /*offset*/ 0, /*default_bin*/ num_bin,
            /*most_freq_bin*/ 0, /*skip_default_bin*/ false, /*na_as_missing*/ false,
            /*run_forward*/ true, sum_g, sum_h, num_data,
            /*parent_output*/ 0.0,
        )
        .expect("find_best_split")
}

/// Assert admissibility on an emitted `SplitInfo` under `cfg` — gated on the scanned HESSIAN
/// SUMS (ground truth), never on the cnt_factor count estimate.
fn assert_split_admissible(si: &lgbm_compute::gain::SplitInfo, cfg: &GainConfig, ctx: &str) {
    // Property (impossible by construction): finite gain or the no-split sentinel — never inf/NaN.
    assert!(
        si.gain.is_finite() || si.gain == f64::NEG_INFINITY,
        "{ctx}: scan emitted a non-finite gain {} — gain=inf must be impossible by construction \
         (spike-072 item-15)",
        si.gain
    );
    if si.gain != f64::NEG_INFINITY {
        assert!(
            si.left_sum_hessian >= cfg.min_sum_hessian_in_leaf,
            "{ctx}: winning split's LEFT child hessian {} < min_sum_hessian_in_leaf {} — the scan \
             admitted a near-empty prefix (spike-072 fingerprint: scan_h≈0-kEpsilon, actual_n=0)",
            si.left_sum_hessian,
            cfg.min_sum_hessian_in_leaf
        );
        assert!(
            si.right_sum_hessian >= cfg.min_sum_hessian_in_leaf,
            "{ctx}: winning split's RIGHT child hessian {} < min_sum_hessian_in_leaf {}",
            si.right_sum_hessian,
            cfg.min_sum_hessian_in_leaf
        );
    }
}

/// Lane B — under the REAL config the shared scan rejects the near-empty prefix and returns
/// only an admissible, finite-gain winner. Under a permissive config the thr=0 near-empty
/// prefix would win with a sub-`min_sum_hessian` left child; the real config rejects it.
#[test]
fn resident_scan_rejects_near_empty_prefix() {
    let cfg = realistic_cfg();
    let (hist, sum_g, sum_h, num_data) = crafted_near_empty_histogram();
    let si = scan(&cfg, &hist, sum_g, sum_h, num_data, 4);
    assert_split_admissible(&si, &cfg, "crafted near-empty histogram under real config");

    // The near-empty thr=0 prefix must NOT be the winner under the real config: if a split was
    // found it is one of the admissible interior thresholds (left child >= min_sum_hessian).
    if si.gain != f64::NEG_INFINITY {
        assert!(
            si.left_sum_hessian >= 1.0,
            "under the real config the winner should be an admissible interior split (left \
             hessian ~8), not the near-empty bin-0 prefix; got left_sum_hessian={}",
            si.left_sum_hessian
        );
    }
}

/// Lane B property sweep: across several crafted histograms × BOTH the
/// permissive and the real config, the scan NEVER returns a non-finite gain (gain=inf is
/// impossible by construction — the kEpsilon-seeded denominators keep every emitted gain
/// finite), and under the REAL config every emitted winner is admissible. The permissive arm
/// documents that admissibility is what the pre-fix production seam threw away, while the
/// finite-gain guarantee holds under ANY config.
#[test]
fn resident_scan_never_emits_non_finite_gain() {
    // A family of crafted histograms: (hist, sum_g, sum_h, num_data, num_bin).
    let cases: Vec<(Vec<f64>, f64, f64, i32, u32)> = vec![
        // The near-empty low-bin prefix residue.
        {
            let (h, g, s, n) = crafted_near_empty_histogram();
            (h, g, s, n, 4)
        },
        // A NEGATIVE tiny residue in bin 0 (the subtract-derived phantom can be < 0), plus real
        // mass — the reverse-scan kEpsilon seed must still keep the gain finite.
        (
            vec![0.36, -1e-6, 1.0, 8.0, 1.0, 8.0, 1.0, 8.0],
            0.36 + 3.0,
            -1e-6 + 24.0,
            1_440_000,
            4,
        ),
        // A genuinely near-zero prefix on both g and h.
        (
            vec![1e-9, 1e-9, 2.0, 10.0, 2.0, 10.0, 2.0, 10.0, 2.0, 10.0],
            1e-9 + 8.0,
            1e-9 + 40.0,
            200_000,
            5,
        ),
    ];

    for (i, (hist, sum_g, sum_h, num_data, num_bin)) in cases.iter().enumerate() {
        for (name, cfg) in [("permissive", permissive_cfg()), ("real", realistic_cfg())] {
            let si = scan(&cfg, hist, *sum_g, *sum_h, *num_data, *num_bin);
            assert!(
                si.gain.is_finite() || si.gain == f64::NEG_INFINITY,
                "case {i} ({name} cfg): scan emitted a non-finite gain {} — gain=inf must be \
                 impossible by construction (spike-072 item-15)",
                si.gain
            );
            // Under the REAL config, any emitted winner must be admissible.
            if name == "real" {
                assert_split_admissible(&si, &cfg, &format!("sweep case {i} (real cfg)"));
            }
        }
    }
}

// ===========================================================================
// The `--features rocm` HARDWARE lane: prove admissibility binds on the real hip
// resident scheduler, plus a non-uniform-hessian SMOKE (a small-corpus/uniform-hessian
// blind spot that a real-data corpus can hide).
//
// Runtime budget: both lanes use SMALL/moderate corpora and grow a single tree — well
// under the hip training budget. A full end-to-end proof over a much larger corpus is
// out of scope here.
// ===========================================================================
#[cfg(feature = "rocm")]
mod rocm {
    use super::*;
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::{Backend, RocmBackend};

    /// Grow the given corpus on the REAL hip resident scheduler under the realistic config,
    /// then assert every leaf is admissible (recomputed from the partition + raw hessians) and
    /// every leaf value is finite.
    fn grow_and_assert_admissible(
        gf: &[lgbm_compute::GrowFeature],
        gradients: &[f32],
        hessians: &[f32],
        num_leaves: i32,
        ctx: &str,
    ) {
        let backend = RocmBackend::with_resident(true);
        assert!(
            backend.resident_pool_supported(),
            "{ctx}: with_resident(true) must drive the real hip resident scheduler"
        );
        let client = rocm_client();
        let cfg = realistic_cfg();
        let (tree, layout) = grow_tree_on_device_driver_with_cfg(
            &backend, &client, gradients, hessians, gf, num_leaves, -1, cfg,
        )
        .unwrap_or_else(|e| panic!("{ctx}: hip resident grow failed: {e:?}"));

        // Every leaf value must be finite (an inadmissible split can drive leaves to ±inf/NaN).
        for (leaf, &v) in tree.leaf_value.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{ctx}: leaf {leaf} value {v} is not finite (spike-072 corruption fingerprint)"
            );
        }

        // Admissibility recomputed from the DATA (never SplitInfo.left_count).
        for (leaf, rows) in driver_leaf_rows(&layout).iter().enumerate() {
            let actual_n = rows.len() as i32;
            let actual_h: f64 = rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();
            assert!(
                actual_n >= cfg.min_data_in_leaf,
                "{ctx}: leaf {leaf} has {actual_n} actual rows < min_data_in_leaf {} on the hip \
                 resident arm",
                cfg.min_data_in_leaf
            );
            assert!(
                actual_h >= cfg.min_sum_hessian_in_leaf,
                "{ctx}: leaf {leaf} actual hessian sum {actual_h} < min_sum_hessian_in_leaf {} — a \
                 near-empty-child split was admitted on the hip resident arm (spike-072 item-15)",
                cfg.min_sum_hessian_in_leaf
            );
        }
    }

    /// Lane A mirror on hip hardware — the micro-cluster corpus grows an admissible tree on the
    /// real resident scheduler under the real config.
    #[test]
    fn on_device_seam_honors_min_data_on_hip() {
        let (features, gradients, hessians) = micro_cluster_corpus();
        let gf = grow_features(&features);
        grow_and_assert_admissible(&gf, &gradients, &hessians, 4, "hip lane-A mirror");
    }

    /// A moderate (~50k-row) deterministic corpus with NON-UNIFORM binary-logloss-like hessians
    /// (`h = p·(1-p)`, `p = sigmoid(varied score)`, in `(0, 0.25]`) across 8 features — a class
    /// of corpus that small, uniform-hessian test corpora do not exercise.
    fn non_uniform_corpus() -> (Vec<lgbm_compute::GrowFeature>, Vec<f32>, Vec<f32>) {
        use lgbm_dataset::BinType;
        use lgbm_treelearner::offset_for_most_freq_bin;
        const N: usize = 50_000;
        const F: usize = 8;
        const NUM_BIN: u32 = 256;

        // splitmix64 for deterministic, dependency-free pseudo-randomness.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        let mut columns: Vec<Vec<u32>> = vec![Vec::with_capacity(N); F];
        let mut gradients: Vec<f32> = Vec::with_capacity(N);
        let mut hessians: Vec<f32> = Vec::with_capacity(N);

        for _ in 0..N {
            let mut signal = 0.0f64;
            for col in columns.iter_mut() {
                let bin = (next() % u64::from(NUM_BIN)) as u32;
                col.push(bin);
                signal += (f64::from(bin) - 127.5) / 127.5;
            }
            // A varied score → non-uniform p → non-uniform hessian (never the uniform 0.25).
            let score = 0.35 * signal / F as f64;
            let p = 1.0 / (1.0 + (-score).exp());
            // Deterministic label from a second draw thresholded by p.
            let label = if (next() as f64 / u64::MAX as f64) < p { 1.0 } else { 0.0 };
            gradients.push((p - label) as f32);
            hessians.push((p * (1.0 - p)) as f32);
        }

        let upper: Vec<f64> = (0..NUM_BIN).map(|b| f64::from(b) + 0.5).collect();
        let gf = columns
            .into_iter()
            .enumerate()
            .map(|(i, bins)| lgbm_compute::GrowFeature {
                bins: BinColumn::new(bins, NUM_BIN),
                num_bin: NUM_BIN,
                offset: offset_for_most_freq_bin(0),
                min_bin: 0,
                max_bin: NUM_BIN - 1,
                default_bin: NUM_BIN,
                most_freq_bin: 0,
                missing_type: MissingType::None,
                bin_upper_bound: upper.clone(),
                real_feature_index: i as i32,
                bin_type: BinType::Numerical,
                bin_to_category: Vec::new(),
                cat_smooth: 10.0,
                cat_l2: 10.0,
                max_cat_threshold: 32,
                max_cat_to_onehot: 4,
                min_data_per_group: 100,
            })
            .collect::<Vec<_>>();
        (gf, gradients, hessians)
    }

    /// Non-uniform-hessian SMOKE on hip hardware: grow a full tree; assert every leaf value is
    /// finite and no leaf carries a near-zero actual child hessian (the fingerprint of an
    /// admitted near-empty split). Hardware-real, non-uniform-hessian coverage — kept to a
    /// single moderate grow (well under the hip runtime budget).
    #[test]
    fn on_device_non_uniform_hessian_smoke_on_hip() {
        let (gf, gradients, hessians) = non_uniform_corpus();
        grow_and_assert_admissible(&gf, &gradients, &hessians, 31, "hip non-uniform-hessian smoke");
    }
}
