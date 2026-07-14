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
use lgbm_compute::kernels::best_split::SplitSoa;
use lgbm_compute::kernels::split::{
    find_best_splits_batched_fused_f64_devcount_from_handle_on,
    find_best_splits_batched_fused_f64_from_handle_on,
    find_best_splits_fused_reduce_into_leaf_devcount_on, find_best_splits_fused_reduce_into_leaf_on,
    upload_f64_buffer,
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

/// Pin the CubeCL autotuner off (its bench server cannot run headless). Read-once gate;
/// set before the first scan launch. This test binary hosts a single test, so the
/// process-global write races with nothing.
fn pin_autotune_off() {
    // SAFETY: test-only single value, set before any reader caches the flag.
    unsafe { std::env::set_var("LGBM_AUTOTUNE", "0") };
}

/// Force BOTH sides of the comparison onto the SAME scan variant, so the ONLY variable
/// under test is the `num_data` SOURCE (host scalar vs on-device), not the scan kernel.
/// This is load-bearing: the hip scan variants are NOT bit-identical to each other — the
/// staged PARPREFIX kernel's parallel f64 reduction reorders adds and so differs from the
/// sequential LEGACY scan by ~1 ULP (hip is ~1e-6, not bit-exact). The device-`num_data`
/// path mirrors the host dispatch (parprefix twin when parprefix is on, else the legacy
/// twin), so pinning the variant on both sides gives a byte-exact comparison of each.
/// The flags are read per-launch (not cached), so switching them mid-test is safe.
///
/// - `("0","0")` ⇒ legacy lane-per-feature kernel on both sides (the fallback twin).
/// - `("1","1")` ⇒ the staged PARPREFIX kernel on both sides (the LIVE hip default —
///   the variant SPEC-DRGL-05's deferral must reproduce byte-for-byte).
fn force_scan_variant(staged: &str, parprefix: &str) {
    // SAFETY: test-only, set between sequential (never concurrent) scan launches.
    unsafe {
        std::env::set_var("LGBM_SCAN_STAGED", staged);
        std::env::set_var("LGBM_SCAN_PARPREFIX", parprefix);
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

/// Run the full left-smaller/right-smaller × smaller/larger byte-identity sweep once, on
/// whatever scan variant is currently forced (`variant` labels it in the assertion
/// context). The ONLY difference between the two scans is the `num_data` source.
fn run_parity_sweep(client: &cubecl::prelude::ComputeClient<GpuRt>, variant: &str) {
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
        let mut ls = DeviceLeafSplits::new(client, 1).expect("alloc");
        ls.record_split(client, split_point, p_begin, p_count);
        assign_smaller_larger_roles_device(client, &ls, 0, 99, 98).expect("roles");

        for is_smaller in [true, false] {
            let host_num_data = if is_smaller { smaller_count } else { larger_count };
            let ctx = format!(
                "variant={variant} split_point={split_point} smaller_is_left={smaller_is_left} \
                 is_smaller={is_smaller} host_num_data={host_num_data}"
            );

            // HOST-num_data scan (the anchor): the child count as a host scalar.
            let h_host = upload_f64_buffer(client, &hist);
            let host = find_best_splits_batched_fused_f64_from_handle_on(
                client, h_host, hist.len(), &feats, &cfg, sum_g, sum_h, host_num_data,
            )
            .expect("host-num_data scan");

            // DEVICE-num_data scan: count resolved on device from ranges/roles + p_count.
            let h_dev = upload_f64_buffer(client, &hist);
            let dev = find_best_splits_batched_fused_f64_devcount_from_handle_on(
                client,
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

/// SPEC-DRGL-12: for a left-smaller AND a right-smaller split, and for BOTH the smaller
/// and the larger child, the device-`num_data` scan (count resolved on device from
/// `ranges`/`roles` + the host parent count) is byte-identical to the host-`num_data`
/// scan fed the matching child count — on BOTH the LIVE hip default (staged parprefix)
/// AND the legacy fallback kernel. Each variant pins host+device onto the SAME kernel so
/// the only variable is the num_data source; the parprefix case is the one SPEC-DRGL-05's
/// deferral must reproduce byte-for-byte to keep the deferred tree == the flag-OFF tree.
#[test]
fn device_numdata_scan_byte_identical_to_host_numdata_scan() {
    pin_autotune_off();
    let client = gpu_client();
    // LIVE hip default: staged + parprefix on both sides.
    force_scan_variant("1", "1");
    run_parity_sweep(&client, "parprefix");
    // Legacy fallback: lane-per-feature kernel on both sides (parprefix off).
    force_scan_variant("0", "0");
    run_parity_sweep(&client, "legacy");
}

/// Run the reduce-INTO-FRONTIER sweep once on the currently-forced variant: fold the
/// host-`num_data` scan into SoA slot 0 and the device-`num_data` scan into slot 1 through
/// the NO-READBACK path (`find_best_splits_fused_reduce_into_leaf[_devcount]_on`, the one the
/// `Backend::scan_resident_leaf_into_frontier_devcount` method + the SPEC-DRGL-05 driver
/// call), then assert the two folded records are byte-identical.
fn run_reduce_sweep(client: &cubecl::prelude::ComputeClient<GpuRt>, variant: &str) {
    let num_bins = [16usize, 8];
    let (hist, slot_off, sum_g, sum_h) = synth_hist(0x5ca_0d12, num_bins);
    let feats = feats_of(slot_off, num_bins);
    let real_feats = [0i32, 1i32];
    let cfg = GainConfig::default();
    let (p_begin, p_count) = (0i32, 251i32);
    for (split_point, _) in [(100i32, true), (200i32, false)] {
        let left_count = split_point;
        let right_count = p_count - split_point;
        let smaller_is_left = left_count < right_count;
        let smaller_count = if smaller_is_left { left_count } else { right_count };
        let larger_count = if smaller_is_left { right_count } else { left_count };
        let mut ls = DeviceLeafSplits::new(client, 1).expect("alloc");
        ls.record_split(client, split_point, p_begin, p_count);
        assign_smaller_larger_roles_device(client, &ls, 0, 99, 98).expect("roles");

        for is_smaller in [true, false] {
            let host_num_data = if is_smaller { smaller_count } else { larger_count };
            let ctx = format!("variant={variant} split_point={split_point} is_smaller={is_smaller}");
            let soa = SplitSoa::zeroed(client, 2);

            // HOST-num_data fold into slot 0 (the anchor).
            let h_host = upload_f64_buffer(client, &hist);
            find_best_splits_fused_reduce_into_leaf_on(
                client, h_host, hist.len(), &feats, &real_feats, &cfg, sum_g, sum_h,
                host_num_data, &soa, 0, None,
            )
            .expect("host reduce");

            // DEVICE-num_data fold into slot 1 (count resolved on device).
            let h_dev = upload_f64_buffer(client, &hist);
            find_best_splits_fused_reduce_into_leaf_devcount_on(
                client, h_dev, hist.len(), &feats, &real_feats, &cfg, sum_g, sum_h,
                ls.ranges_handle().clone(), LEAF_SPLIT_STRIDE * ls.capacity(),
                ls.roles_handle().clone(), ROLE_STRIDE * ls.capacity(),
                0, is_smaller, p_count, &soa, 1, None,
            )
            .expect("device reduce");

            let a = soa.read_record(client, 0);
            let b = soa.read_record(client, 1);
            assert_eq!(a.is_valid, b.is_valid, "{ctx}: is_valid");
            assert_eq!(a.inner_feature_index, b.inner_feature_index, "{ctx}: feat");
            assert_eq!(a.threshold, b.threshold, "{ctx}: threshold");
            assert_eq!(a.default_left, b.default_left, "{ctx}: default_left");
            for (name, x, y) in [
                ("gain", a.gain, b.gain),
                ("l_sum_g", a.left_sum_gradients, b.left_sum_gradients),
                ("l_sum_h", a.left_sum_hessians, b.left_sum_hessians),
                ("r_sum_g", a.right_sum_gradients, b.right_sum_gradients),
                ("r_sum_h", a.right_sum_hessians, b.right_sum_hessians),
                ("l_val", a.left_value, b.left_value),
                ("r_val", a.right_value, b.right_value),
            ] {
                assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: {name} ({x} vs {y})");
            }
            assert!(a.is_valid, "{ctx}: host fold found no valid split — corpus not discriminating");
        }
    }
}

/// SPEC-DRGL-12: the NO-READBACK device-`num_data` fold (`fused_scan_to_raw_handle`'s Device
/// path — the launcher `Backend::scan_resident_leaf_into_frontier_devcount` and the
/// SPEC-DRGL-05 driver call) produces a frontier record byte-identical to the host-`num_data`
/// fold, on BOTH the live parprefix variant and the legacy fallback.
#[test]
fn device_numdata_reduce_into_leaf_byte_identical_to_host() {
    pin_autotune_off();
    let client = gpu_client();
    force_scan_variant("1", "1");
    run_reduce_sweep(&client, "parprefix");
    force_scan_variant("0", "0");
    run_reduce_sweep(&client, "legacy");
}
