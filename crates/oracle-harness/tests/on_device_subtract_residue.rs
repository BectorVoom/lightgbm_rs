//! Phase-29 29-03 (OCX-03) — the subtract-residue exact-zero gate.
//!
//! # Why this file exists (spike-072 item 15b)
//!
//! On the opt-in on-device path, a subtract-derived leaf's low bins whose TRUE mass is
//! zero carried a ~0.36 phantom hessian, feeding the (29-02-fixed) admissibility hole a
//! fake near-empty prefix that won with `gain=inf`. 29-03's forensics
//! (`29-SUBTRACT-RESIDUE.md`) named the mechanism to the line: the on-device dense
//! resident build accumulates EVERY bin (mfb included), but the transcribed C++
//! `Dataset::FixHistogram` step then OVERWRITES the forced-empty most-frequent-bin cell
//! with `external_seed − Σ_bin-grouped hist`. Because the external leaf seed is a
//! ROW-ORDER fold (or a scan prefix) while the histogram is a BIN-GROUPED fold, f64
//! non-associativity makes them disagree, and the difference is swept into the
//! globally-empty mfb cell — the phantom.
//!
//! The fix (disposition (a), 29-03): preserve the dense-BUILT mfb cell instead of
//! reconstructing it from the external seed. For a globally-empty bin the built value is
//! exactly 0 at ALL scales (n·u·Σ|h| can exceed the 1e-3 admissibility floor at n≳1e7,
//! so a bound+gate-defusal disposition (b) was ruled INVALID per the 29-CONTEXT locked
//! rule). This is a provable no-op in the anchor regime (integer grad/hess ⇒ row-order ==
//! bin-grouped ⇒ external_seed == Σ hist), so every anchor gate stays green.
//!
//! # Scope / lanes
//!
//! The u64 fixed-point resident build + `fix_compact_kernel` FixHistogram are the GPU
//! path (`#[cfg(feature="gpu")]`); on the cpu lane `CpuBackend` uses the f64 host fold, so
//! this kernel is CI-reachable only on the `#[cfg(feature="rocm")]` hardware lane (same
//! scope note as `on_device_integer_anchor` WR-02). The complementary cpu-lane proof —
//! that the subtract op itself never manufactures residue (`x − x = +0.0`) — lives in
//! `lgbm_compute::kernels::subtract` unit tests
//! (`subtract_identical_cells_are_exact_positive_zero`).

#![cfg(feature = "rocm")]

use lgbm_compute::kernels::histogram::fix_compact_f64_on;
use lgbm_compute::runtime::rocm_client;

/// The FixHistogram reconstruction must leave a globally-empty most-frequent-bin cell
/// EXACTLY `+0.0`, even when the leaf's external seed disagrees with the histogram's own
/// bin-grouped total by a residue far larger than the ~0.36 spike-072 phantom.
///
/// Construction: one feature, `num_bin = 8`, `offset = 0` (no compaction shift),
/// `most_freq_bin = 3` seeded as the forced-empty bin (built mass exactly 0). All other
/// bins carry integer mass (so the u64 `round(v·2^30)` quantization is lossless and the
/// built total is exact). The leaf seed passed to the fix is DELIBERATELY offset from that
/// built total by `δ = 0.37` on BOTH grad and hess — the residue the old
/// `seed − Σothers` reconstruction would sweep into the empty mfb cell.
///
/// Pre-fix: `hist[mfb] = seed − Σ_{i≠mfb} built = 0 + δ = δ` (RED — the phantom).
/// Post-fix: `hist[mfb] = built_mfb = 0` (GREEN — exact zero at all scales).
#[test]
fn forced_empty_mfb_reconstructs_exact_zero_on_hip() {
    let client = rocm_client();

    let num_bin: u32 = 8;
    let mfb: u32 = 3; // forced-empty most-frequent bin (0 rows globally)
    let offset: i32 = 0; // no compaction — the mfb cell keeps raw index `mfb`

    // f32 RAW histogram, stride-2 `[g0,h0,g1,h1,…]`. INTEGER mass everywhere (lossless
    // 2^30 quantization) EXCEPT the forced-empty mfb bin (bin 3), which is exactly 0.
    let raw: Vec<f32> = {
        let mut v = vec![0.0f32; 2 * num_bin as usize];
        // bins 0,1,2,4,5,6,7 carry integer grad/hess; bin 3 (mfb) stays (0,0).
        let masses: [(usize, f32, f32); 7] = [
            (0, -5.0, 7.0),
            (1, 3.0, 4.0),
            (2, -2.0, 9.0),
            (4, 6.0, 5.0),
            (5, -1.0, 8.0),
            (6, 4.0, 3.0),
            (7, -7.0, 6.0),
        ];
        for (b, g, h) in masses {
            v[2 * b] = g;
            v[2 * b + 1] = h;
        }
        v
    };

    // Exact built totals (integers ⇒ f64-exact). The mfb bin contributes 0.
    let built_total_g: f64 = raw.iter().step_by(2).map(|&x| f64::from(x)).sum();
    let built_total_h: f64 = raw.iter().skip(1).step_by(2).map(|&x| f64::from(x)).sum();

    // Leaf seed DISAGREES with the histogram's own total by δ = 0.37 (the residue the old
    // reconstruction would inject into the empty mfb cell).
    let delta = 0.37f64;
    let seed_g = built_total_g + delta;
    let seed_h = built_total_h + delta;

    // (slot_off, num_bin, offset, most_freq_bin)
    let feats = [(0usize, num_bin, offset, mfb)];
    let fixed = fix_compact_f64_on(&client, &raw, &feats, seed_g, seed_h).unwrap();

    let mi = 2 * mfb as usize;
    assert_eq!(
        fixed[mi].to_bits(),
        0.0f64.to_bits(),
        "OCX-03: forced-empty mfb GRADIENT cell must reconstruct to +0.0 (built value), \
         got {} — the external seed's δ={delta} residue leaked into the empty cell \
         (spike-072 item 15b fingerprint)",
        fixed[mi]
    );
    assert_eq!(
        fixed[mi + 1].to_bits(),
        0.0f64.to_bits(),
        "OCX-03: forced-empty mfb HESSIAN cell must reconstruct to +0.0 (built value), \
         got {} — the external seed's δ={delta} residue leaked into the empty cell",
        fixed[mi + 1]
    );

    // Sanity: the real (non-mfb) bins are untouched by the fix — their built mass is
    // preserved exactly (the fix only ever touched the mfb cell).
    for b in 0..num_bin as usize {
        if b as u32 == mfb {
            continue;
        }
        assert_eq!(
            fixed[2 * b].to_bits(),
            f64::from(raw[2 * b]).to_bits(),
            "bin {b} grad must be the untouched built value"
        );
        assert_eq!(
            fixed[2 * b + 1].to_bits(),
            f64::from(raw[2 * b + 1]).to_bits(),
            "bin {b} hess must be the untouched built value"
        );
    }
}
