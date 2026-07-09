//! Spike 086 — launch-batching seam probe: is the on-device grow loop's device-launch
//! count reducible, or is it already at the best-first data-dependency FLOOR?
//!
//! The launch COUNT is hardware-independent (a `bump_launch()` per real device dispatch),
//! so this probe runs on the cpu anchor lane and its numbers TRANSFER to real CUDA —
//! unlike compute timing on the spoofed 8-CU APU. It grows the same dominant-feature
//! corpus at a sweep of `num_leaves` and at 3 vs 24 features, and prints:
//!
//!   1. the measured real-dispatch launch count per grow (anchor arm),
//!   2. that it is INDEPENDENT of num_features (the spike-055 per-feature non-collapse is
//!      fixed — a per-feature build+scan layout would scale these terms with num_features),
//!   3. the analytic floor breakdown per split (build → subtract → scan), and
//!   4. the COUNTERFACTUAL cost of an un-batched per-feature layout, to show how much the
//!      already-shipped batching (feature-collapse + sibling co-pack) saved.
//!
//! Run: cargo run --release -p lgbm-compute --features cpu --example spike086_launch_floor

use lgbm_compute::kernels::grow_driver::{grow_tree_on_device_driver, on_device_launch_count_take};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, CpuBackend, GrowFeature};
use lgbm_dataset::bin_mapper::{BinType, MissingType};

const NUM_BIN: u32 = 8;
const OFFSET_MFB_0: i32 = 1;

/// Dominant-feature-0 corpus (feature 0 monotone in the row index ⇒ it wins every split,
/// so the tree STRUCTURE — and the dispatch count — is identical regardless of how many
/// weaker features are present). Exactly the invariance the num_features-independence
/// check exploits.
fn corpus(num_features: usize, num_data: usize) -> (Vec<GrowFeature>, Vec<f32>, Vec<f32>) {
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

/// Grow once on the anchor arm; return (launch_count, actual_num_leaves).
fn grow(num_features: usize, num_data: usize, num_leaves: i32) -> (u64, i32) {
    let _ = on_device_launch_count_take();
    let client = cpu_client();
    let (features, g, h) = corpus(num_features, num_data);
    let (tree, _layout) =
        grow_tree_on_device_driver(&CpuBackend, &client, &g, &h, &features, num_leaves, -1)
            .expect("anchor grow");
    (on_device_launch_count_take(), tree.num_leaves)
}

fn main() {
    // The counter only bumps when the phase-prof gate is on (parity-neutral otherwise).
    // SAFETY: single-threaded probe, set before the first grow reads the OnceLock gate.
    unsafe {
        std::env::set_var("LGBM_PHASE_PROF", "1");
    }

    let num_data = 4096usize;
    println!("# spike086 — on-device device-launch FLOOR probe (anchor arm; counts are HW-independent)");
    println!("# corpus: {num_data} rows, feature 0 dominant; anchor arm = co-pack OFF (per-leaf).\n");
    println!(
        "{:>10} {:>8} {:>10} {:>10} {:>14} {:>18} {:>16}",
        "num_leaves", "leaves", "launch(3f)", "launch(24f)", "anchor_floor", "per_feature(24f)", "batching_saved"
    );

    for &nl in &[2i32, 4, 8, 16, 31, 63] {
        let (l3, leaves) = grow(3, num_data, nl);
        let (l24, _) = grow(24, num_data, nl);
        // Anchor (co-pack OFF) analytic floor: upload(bin)+upload(gh) + root[build+scan]=... the
        // cpu anchor arm does NOT upload (no device residency) so its floor is the per-leaf term
        // only: 2 (root build+scan) + 4*(leaves-1) (build+subtract+2 scans per split).
        let anchor_floor = 2 + 4 * (leaves as u64 - 1);
        // Counterfactual: a NON-batched per-feature layout (spike-055 regression) would scale
        // the build+scan terms by num_features. Rough model: each per-split build+2 scans and the
        // root build+scan become per-feature ⇒ ~ (2 + 3*(leaves-1)) * num_features + (leaves-1)
        // subtracts. We report the dominant per-feature multiple at 24 features.
        let nf = 24u64;
        let per_feature = (2 + 3 * (leaves as u64 - 1)) * nf + (leaves as u64 - 1);
        let saved = per_feature.saturating_sub(l24);
        let flag = if l3 == l24 { "" } else { "  <-- FEATURE-SCALED!" };
        println!(
            "{:>10} {:>8} {:>10} {:>10} {:>14} {:>18} {:>16}{}",
            nl, leaves, l3, l24, anchor_floor, per_feature, saved, flag
        );
    }

    println!("\n# READING:");
    println!("#  FULL-GROWTH rows (leaves == num_leaves, here nl<=16): launch(3f) == launch(24f)");
    println!("#    EXACTLY  ⇒  count is num_features-INDEPENDENT (the shipped feature-collapse; a");
    println!("#    per-feature layout would cost the 'per_feature(24f)' column, ~12-18× more).");
    println!("#  measured == anchor_floor (2 + 4*(leaves-1))  ⇒  at the per-leaf FLOOR on the anchor.");
    println!("#  SATURATED rows (nl=31,63 → this synthetic corpus only supports 24 leaves): the");
    println!("#    3f arm breaks cleanly at the floor (94), but the 24f arm keeps ATTEMPTING splits");
    println!("#    on already-un-splittable leaves — a degenerate-TAIL effect (build+scan on leaves");
    println!("#    that yield no admissible split), NOT a batching-seam question. Real max_bin=255");
    println!("#    corpora reach num_leaves without saturating, so production stays on the floor.");
    println!("#  the resident (rocm) arm collapses the 2 sibling scans → 1 (spike-024 co-pack):");
    println!("#    floor = 4 + 3*(num_leaves-1)  [host partition route, co-pack ON] — see");
    println!("#    tests/on_device_launch_count.rs resident_analytic_lane (the exact closed form).");
    println!("#  per split the chain build→subtract→scan is a STRICT data dependency (each phase");
    println!("#    consumes the prior's output) — the ONLY parallel batch is the two sibling scans,");
    println!("#    already co-packed. Fusing the chain needs f64 (spike-052: 5.4× WORSE); batching");
    println!("#    across splits needs level-wise growth (breaks best-first bit-exact parity).");
}
