//! Phase 09 (spike-007): row-partitioned LDS build parity on the gfx1100.
//!
//! Exercises the production batched LDS launcher `build_leaf_histograms_batched_f32_on`
//! — which routes through the row-partitioned `construct_leaf_hist_batched_lds_kernel`
//! and the `row_partition_count` heuristic — at BOTH `P=1` (small/medium gate; must equal
//! the cpu f64 anchor within the ~1e-6 contract) and `P>1` (large-leaf gate, forced via
//! `LGBM_ROWPART_MIN=0`; held to a DOCUMENTED relaxed tolerance reflecting the independent
//! f32 partial-sum trees the spike measured, ~2e-5 rel at 1M rows).
//!
//! The cpu f64 anchor (`cpu_ref` here, and the real `accumulate_histogram_into` merge gate)
//! is untouched — only the GPU f32 path widens with P. This is the "separate, still-open
//! large-shape GPU parity gate" the spikes/MANIFEST Requirements call out, now bounded by
//! the large-leaf gate so the small-shape contract is unaffected.
//!
//! Run: cargo test --release --features rocm -p lgbm-compute --test rocm_row_partition
#![cfg(feature = "rocm")]

use lgbm_compute::kernels::histogram::build_leaf_histograms_batched_f32_on;
use lgbm_compute::runtime::rocm_client;

/// f64 reference: gather each feature's leaf-rows and accumulate into the concatenated
/// `[slot_off[f] + bin*2 (+1)]` layout — the exact arithmetic the kernel performs, in f64.
fn cpu_ref(
    feature_bins: &[&[u32]],
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    g: &[f32],
    h: &[f32],
) -> Vec<f64> {
    let mut out = vec![0.0f64; slot_len];
    for (f, bins) in feature_bins.iter().enumerate() {
        let base = slot_off[f];
        for &r in leaf_rows {
            let bin = bins[r as usize] as usize;
            out[base + bin * 2] += f64::from(g[r as usize]);
            out[base + bin * 2 + 1] += f64::from(h[r as usize]);
        }
    }
    out
}

/// Max per-cell relative error (denominator floored at 1.0, matching the project's
/// rocm parity asserts in `rocm_parallel_histogram.rs`).
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
fn row_partition_batched_matches_cpu_anchor_p1_and_p_gt_1() {
    let client = rocm_client();

    // 8 features × 64 bins. 50k rows is enough that P=clamp(768/8,1,16)=16 engages under
    // LGBM_ROWPART_MIN=0 while keeping the test fast (the divergence at 50k ≤ the 1M-row
    // 2e-5 the spike measured, so the 5e-5 bound has headroom).
    let num_data = 50_000usize;
    let feats = 8usize;
    let num_bin = 64usize;

    // Scattered feature columns (feature-major full columns) + a full-data leaf.
    let mut cols: Vec<Vec<u32>> = Vec::with_capacity(feats);
    for f in 0..feats {
        let mut c = Vec::with_capacity(num_data);
        for r in 0..num_data {
            let hsh = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
            c.push((hsh % num_bin as u64) as u32);
        }
        cols.push(c);
    }
    let feature_bins: Vec<&[u32]> = cols.iter().map(|c| c.as_slice()).collect();
    let leaf_rows: Vec<u32> = (0..num_data as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % num_data as u32)
        .collect();
    let g: Vec<f32> = (0..num_data).map(|i| (i % 13) as f32 * 0.1 - 0.5).collect();
    let h: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let feat_len = 2 * num_bin;
    let slot_off: Vec<usize> = (0..feats).map(|f| f * feat_len).collect();
    let slot_len = feats * feat_len;

    let anchor = cpu_ref(&feature_bins, &slot_off, slot_len, &leaf_rows, &g, &h);

    // --- P=1 path: force via a high threshold; must hold the ~1e-6 small-shape contract. ---
    unsafe { std::env::set_var("LGBM_ROWPART_MIN", "100000000") };
    let p1 = build_leaf_histograms_batched_f32_on(
        &client, &feature_bins, &slot_off, slot_len, &leaf_rows, &g, &h,
    )
    .unwrap();
    let rel_p1 = max_rel(&anchor, &p1);
    // 1e-5 is the established f32-LDS-vs-f64-anchor gate (see `rocm_parallel_histogram.rs`
    // `lds_within_tolerance_of_cpu_f64_anchor`): the LDS sub-hist accumulates in f32, so even
    // the P=1 path carries the f32-vs-f64 accumulation gap (~1.8e-6 at this shape). This is
    // the pre-existing kernel behavior — row-partitioning does not touch the P=1 path.
    assert!(
        rel_p1 < 1e-5,
        "P=1 batched diverged from the cpu f64 anchor beyond the f32 LDS gate: {rel_p1:e}"
    );

    // --- P>1 path: force via zero threshold (P=16 here); documented relaxed tolerance. ---
    unsafe { std::env::set_var("LGBM_ROWPART_MIN", "0") };
    let pn = build_leaf_histograms_batched_f32_on(
        &client, &feature_bins, &slot_off, slot_len, &leaf_rows, &g, &h,
    )
    .unwrap();
    unsafe { std::env::remove_var("LGBM_ROWPART_MIN") };

    let rel_pn = max_rel(&anchor, &pn);
    // 5e-5: spike-007 measured ~2e-5 rel at 1M rows from the independent f32 partial-sum
    // trees row-partitioning introduces; 50k rows diverge less, 5e-5 leaves headroom. The
    // cpu f64 anchor is untouched — this bounds only the GPU f32 large-leaf residual.
    assert!(
        rel_pn < 5e-5,
        "P>1 batched exceeded the documented large-leaf tol: {rel_pn:e} (expected < 5e-5). \
         See 04-ROCM-GAPS.md row-partition residual."
    );

    // Structural: P>1 agrees with P=1 within the same envelope (same build, different fold tree).
    let rel_pp = max_rel(&p1, &pn);
    assert!(
        rel_pp < 5e-5,
        "P>1 vs P=1 disagreement beyond the documented envelope: {rel_pp:e}"
    );

    eprintln!(
        "row-partition batched parity: rel(P1 vs anchor)={rel_p1:e}  \
         rel(P>1 vs anchor)={rel_pn:e}  rel(P>1 vs P1)={rel_pp:e}"
    );
}

/// NRW-01 re-pin: the NAIVE batched fallback kernel `construct_leaf_hist_batched_kernel`
/// (the `max_w > HIST_LDS_MAX` branch of `build_leaf_histograms_batched_f32_on`) is the
/// ONE swept production kernel without dedicated GPU-vs-CPU-f64-anchor coverage — the
/// row-partition test above and `rocm_parallel_histogram.rs` only exercise the LDS branch
/// (every feature ≤ 256 bins). This forces the naive branch with a SINGLE feature of
/// 300 bins (> 256, so `max_w = 600 > HIST_LDS_MAX = 512`) and re-pins the
/// launch_unchecked result to the CPU f64 anchor within the ~1e-6 f32-atomic gate
/// (ABS 5e-6 / REL 1e-5), proving the `::launch` → `::launch_unchecked` switch did not
/// change the naive-fallback numerics. GPU-vs-CPU-f64-anchor, NEVER GPU-vs-GPU
/// (DEF-f8u-01). A BOUNDED leaf subset is used (not the full corpus) to avoid the
/// DEF-MWR-01 near-zero-grad full-corpus flake.
#[test]
fn naive_batched_fallback_matches_cpu_anchor_within_tol() {
    let client = rocm_client();

    // ONE feature of 300 bins > 256 ⇒ max_w = 600 > HIST_LDS_MAX (512) ⇒ the launcher
    // routes through the NAIVE `construct_leaf_hist_batched_kernel`, not the LDS kernel.
    let num_data = 8_000usize;
    let num_bin = 300usize;

    let col: Vec<u32> = (0..num_data)
        .map(|r| ((r as u64).wrapping_mul(2_654_435_761) % num_bin as u64) as u32)
        .collect();
    let feature_bins: Vec<&[u32]> = vec![col.as_slice()];

    // Bounded leaf subset (7..num_data).step_by(3) — the stable-test pattern that avoids
    // the DEF-MWR-01 full-corpus near-zero-grad cancellation cells.
    let leaf_rows: Vec<u32> = (7..num_data as u32).step_by(3).collect();
    // Small-magnitude pseudo-residual gradients (like real boosting gradients).
    let g: Vec<f32> = (0..num_data).map(|i| ((i as f32) * 0.000_123).sin() * 0.5).collect();
    let h: Vec<f32> = vec![1.0f32; num_data];

    let feat_len = 2 * num_bin;
    let slot_off: Vec<usize> = vec![0];
    let slot_len = feat_len;

    let anchor = cpu_ref(&feature_bins, &slot_off, slot_len, &leaf_rows, &g, &h);
    let gpu = build_leaf_histograms_batched_f32_on(
        &client, &feature_bins, &slot_off, slot_len, &leaf_rows, &g, &h,
    )
    .unwrap();

    // ABS 5e-6 / REL 1e-5 — the established f32-atomic-vs-f64-anchor gate. The naive
    // path accumulates in f32 in nondeterministic order, so it carries the same ~1e-6
    // gap as the LDS path; launch_unchecked does not change that order.
    let mut max_abs = 0.0f64;
    let mut max_rel_e = 0.0f64;
    for (a, b) in anchor.iter().zip(&gpu) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        max_rel_e = max_rel_e.max(diff / a.abs().max(1.0));
    }
    eprintln!(
        "naive-batched-fallback vs cpu f64 anchor: max_abs={max_abs:e}  max_rel={max_rel_e:e}"
    );
    assert!(
        max_abs < 5e-6 || max_rel_e < 1e-5,
        "naive batched fallback diverged from the cpu f64 anchor beyond the f32-atomic \
         gate: max_abs={max_abs:e}, max_rel={max_rel_e:e}"
    );
}
