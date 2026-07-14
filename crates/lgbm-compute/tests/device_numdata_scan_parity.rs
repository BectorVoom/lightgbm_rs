//! SPEC-DRGL-12 (REAL GPU only): the device-`num_data` single-child scan sources
//! the child's row count ON DEVICE from the resident split/role record
//! (`DeviceLeafSplits` `ranges`/`roles`, SPEC-DRGL-01/02) instead of a host
//! `num_data` scalar, and is f64-BYTE-IDENTICAL to the host-`num_data` scan fed the
//! same count. This is the scan-side twin of T-04's fixed-grid BUILD
//! (`fixed_grid_build_byte_identical_to_exact_grid`) and the capability
//! SPEC-DRGL-05's read_split-deferral needs (the scan can no longer be handed
//! `num_data = split_point` host-side once that read is deferred).
//!
//! Runs REAL-DEVICE only (`rocm`/`cuda`): the resident split/role buffers +
//! the `<R>::name(client) != "cpu"` device gate are the same real-device precedent
//! as every other resident-chain gate. On cubecl-cpu there is nothing to test (the
//! host-scalar path is unchanged and IS the anchor).
#![cfg(any(feature = "rocm", feature = "cuda"))]

use lgbm_compute::gain::{GainConfig, SplitInfo};
use lgbm_compute::kernels::partition::{
    assign_smaller_larger_roles_device, DeviceLeafSplits, LEAF_SPLIT_STRIDE, ROLE_STRIDE,
};
use lgbm_compute::kernels::split::{
    find_best_splits_batched_fused_f64_devcount_from_handle_on,
    find_best_splits_batched_fused_f64_from_handle_on, upload_f64_buffer,
};
use lgbm_compute::BatchedSplitFeature;

#[cfg(feature = "cuda")]
type GpuRt = lgbm_compute::runtime::CudaRuntime;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
type GpuRt = lgbm_compute::runtime::RocmRuntime;

#[cfg(feature = "cuda")]
fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRt> {
    lgbm_compute::runtime::cuda_client()
}
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
fn gpu_client() -> cubecl::prelude::ComputeClient<GpuRt> {
    lgbm_compute::runtime::rocm_client()
}

/// Pin BOTH sides of the comparison onto the SAME (legacy lane-per-feature) scan kernel
/// so the ONLY variable under test is the `num_data` SOURCE (host scalar vs on-device),
/// not the scan variant. This matters: on hip the host path defaults to the staged
/// PARPREFIX kernel, whose parallel f64 reduction reorders additions and so differs from
/// the sequential legacy scan by ~1 ULP — a real, pre-existing cross-variant gap (hip is
/// ~1e-6, not bit-exact) that is NOT what SPEC-DRGL-12 is about. The device-`num_data`
/// twin (T-12) is the legacy kernel's twin, so we force the host onto legacy too
/// (`LGBM_SCAN_STAGED=0`) for a byte-exact, correctly-attributed comparison.
///
/// (Extending the device-`num_data` path to the staged/parprefix kernels — so the
/// deferred scan keeps the parprefix perf win on hip — is the documented follow-on
/// within SPEC-DRGL-12/13; the legacy twin proves the num_data-sourcing MECHANISM first.)
///
/// `LGBM_AUTOTUNE=0` additionally pins the autotuner off (its bench server cannot run
/// headless). Both are read-once gates set before the first scan launch; this test binary
/// hosts a single test, so the process-global env writes race with nothing.
fn pin_legacy_scan_variant() {
    // SAFETY: test-only single values, set before any reader caches the flags; the only
    // reader in this process is this one test.
    unsafe {
        std::env::set_var("LGBM_AUTOTUNE", "0");
        std::env::set_var("LGBM_SCAN_STAGED", "0");
    }
}

/// Deterministic LCG (no rand dep) — the spike corpus generator pattern.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Build a synthetic concatenated stride-2 f64 histogram for two features
/// (`num_bins[0]` + `num_bins[1]` bins): cell `[2*bin]` = grad, `[2*bin+1]` = hess.
/// Hess is a small positive integer per bin so `round_int(h*cnt_factor)` yields
/// meaningful per-bin counts and the `min_data_in_leaf` gate is exercised. Returns
/// `(hist, slot_off, sum_gradient, sum_hessian)` with the leaf totals summed off the
/// histogram so a real split is found.
fn synth_hist(seed: u64, num_bins: [usize; 2]) -> (Vec<f64>, [usize; 2], f64, f64) {
    let mut lcg = Lcg(seed);
    let total_bins = num_bins[0] + num_bins[1];
    let mut hist = vec![0.0f64; 2 * total_bins];
    let slot_off = [0usize, 2 * num_bins[0]];
    let mut sum_g = 0.0f64;
    let mut sum_h = 0.0f64;
    for (fi, &nb) in num_bins.iter().enumerate() {
        for bin in 0..nb {
            let g = (lcg.next_u32() % 17) as f64 - 8.0; // grad in [-8, 8]
            let h = (lcg.next_u32() % 4) as f64 + 1.0; // hess in [1, 4]
            hist[slot_off[fi] + 2 * bin] = g;
            hist[slot_off[fi] + 2 * bin + 1] = h;
            sum_g += g;
            sum_h += h;
        }
    }
    (hist, slot_off, sum_g, sum_h)
}

fn feats_of(slot_off: [usize; 2], num_bins: [usize; 2]) -> Vec<BatchedSplitFeature> {
    (0..2)
        .map(|f| BatchedSplitFeature {
            slot_off: slot_off[f],
            num_bin: num_bins[f] as u32,
            offset: 1,
            default_bin: 0,
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing: false,
            run_forward: true,
        })
        .collect()
}

/// Strict f64-byte + i32/bool equality of two `SplitInfo`s (the "byte-identical"
/// contract — bit compare the f64 fields so `-0.0`/NaN can never sneak a false pass).
fn assert_split_bit_identical(a: &SplitInfo, b: &SplitInfo, ctx: &str) {
    assert_eq!(a.threshold, b.threshold, "{ctx}: threshold");
    assert_eq!(a.left_count, b.left_count, "{ctx}: left_count");
    assert_eq!(a.right_count, b.right_count, "{ctx}: right_count");
    assert_eq!(a.default_left, b.default_left, "{ctx}: default_left");
    for (name, x, y) in [
        ("gain", a.gain, b.gain),
        ("left_sum_gradient", a.left_sum_gradient, b.left_sum_gradient),
        ("left_sum_hessian", a.left_sum_hessian, b.left_sum_hessian),
        ("right_sum_gradient", a.right_sum_gradient, b.right_sum_gradient),
        ("right_sum_hessian", a.right_sum_hessian, b.right_sum_hessian),
        ("left_output", a.left_output, b.left_output),
        ("right_output", a.right_output, b.right_output),
    ] {
        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: {name} ({x} vs {y})");
    }
}

/// SPEC-DRGL-12: for a left-smaller AND a right-smaller split, and for BOTH the
/// smaller and the larger child, the device-`num_data` scan (count resolved on device
/// from `ranges`/`roles` + the host parent count) is byte-identical to the
/// host-`num_data` scan fed the matching child count.
#[test]
fn device_numdata_scan_byte_identical_to_host_numdata_scan() {
    pin_legacy_scan_variant();
    let client = gpu_client();
    let num_bins = [16usize, 8];
    let (hist, slot_off, sum_g, sum_h) = synth_hist(0x5ca_0d12, num_bins);
    let feats = feats_of(slot_off, num_bins);
    let cfg = GainConfig::default();

    // Parent span [p_begin, p_begin+p_count). split_point chosen so the two cases give
    // a left-smaller and a right-smaller child; num_data straddles min_data_in_leaf so a
    // wrong device count would shift the winner.
    let (p_begin, p_count) = (0i32, 251i32);
    for (split_point, expect_left_smaller) in [(100i32, true), (200i32, false)] {
        let left_count = split_point;
        let right_count = p_count - split_point;
        let smaller_is_left = left_count < right_count;
        assert_eq!(smaller_is_left, expect_left_smaller, "role sanity for split_point {split_point}");
        let smaller_count = if smaller_is_left { left_count } else { right_count };
        let larger_count = if smaller_is_left { right_count } else { left_count };

        // Resident split state: record the split point, resolve the role ON DEVICE.
        let mut ls = DeviceLeafSplits::new(&client, 1).expect("alloc");
        ls.record_split(&client, split_point, p_begin, p_count);
        assign_smaller_larger_roles_device(&client, &ls, 0, 99, 98).expect("roles");

        for is_smaller in [true, false] {
            let host_num_data = if is_smaller { smaller_count } else { larger_count };
            let ctx = format!(
                "split_point={split_point} smaller_is_left={smaller_is_left} is_smaller={is_smaller} \
                 host_num_data={host_num_data}"
            );

            // HOST-num_data scan (the anchor): the child count as a host scalar.
            let h_host = upload_f64_buffer(&client, &hist);
            let host = find_best_splits_batched_fused_f64_from_handle_on(
                &client, h_host, hist.len(), &feats, &cfg, sum_g, sum_h, host_num_data,
            )
            .expect("host-num_data scan");

            // DEVICE-num_data scan: count resolved on device from ranges/roles + p_count.
            let h_dev = upload_f64_buffer(&client, &hist);
            let dev = find_best_splits_batched_fused_f64_devcount_from_handle_on(
                &client,
                h_dev,
                hist.len(),
                &feats,
                &cfg,
                sum_g,
                sum_h,
                ls.ranges_handle().clone(),
                LEAF_SPLIT_STRIDE * ls.capacity(),
                ls.roles_handle().clone(),
                ROLE_STRIDE * ls.capacity(),
                0, // split_slot
                is_smaller,
                p_count, // host-known parent row count (upper bound)
            )
            .expect("device-num_data scan");

            assert_eq!(host.len(), dev.len(), "{ctx}: feature count");
            for (f, (h, d)) in host.iter().zip(dev.iter()).enumerate() {
                assert_split_bit_identical(h, d, &format!("{ctx} feat={f}"));
            }
            // The split must actually be found for at least one feature, else the test is
            // vacuous (a no-split-vs-no-split match proves nothing about num_data).
            assert!(
                host.iter().any(|s| s.gain > f64::NEG_INFINITY),
                "{ctx}: host scan found no split — pick a more discriminating corpus"
            );
        }
    }
}
