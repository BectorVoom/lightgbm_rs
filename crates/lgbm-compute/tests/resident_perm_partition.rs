//! Bit-exactness gates for the DEVICE-RESIDENT permutation partition
//! (`LGBM_PARTITION_RESIDENT`, the `cuda_data_indices_` analog) on the runnable
//! cubecl-cpu lane — the same anchor discipline as `partition_parity`: the device
//! fold is pinned byte-equal to the host stable-partition anchor, never
//! GPU-vs-GPU.
//!
//! Three layers:
//! 1. KERNEL parity (default features): `ResidentPermPartition::partition_leaf`
//!    (fused mark+block-scan → totals+ranges → stable scatter, in place on the
//!    device perm) vs `partition_leaf_stable_fused` (the shipped host anchor) over
//!    a MULTI-SPLIT sequence spanning the missing-type × default-direction fan-out
//!    and >1 scan blocks — after every split the whole device perm and the
//!    resident child ranges must match the host mirror exactly.
//! 2. BUILD parity (REAL GPU only — `rocm`/`cuda`): the rows-HANDLE resident
//!    build (`build_fix_compact_resident_rows_handle_f64_on`, rows = an OFFSET
//!    VIEW of a device buffer) vs the host-rows twin fed the same row ids —
//!    f64-bit identical output (also proves cubecl offset-Handle views on the
//!    runtime under test).
//! 3. DRIVER parity (REAL GPU only): `GpuBackend<R>` grows the same corpus with
//!    the resident-perm arm forced OFF then ON
//!    (`set_partition_resident_override`) — trees and layouts must be
//!    byte-identical.
//!
//! Layers 2/3 CANNOT run on cubecl-cpu: the u64 fixed-point resident build is
//! atomic-based and cubecl-cpu 0.10 has no atomics
//! (`Operation::Atomic → todo!()` in its MLIR visitor) — the same reason every
//! existing resident-chain gate lives behind `#[cfg(feature = "rocm")]`. They
//! run in the real-CUDA spike session (`--features cuda`).

use lgbm_compute::kernels::data_partition::partition_leaf_stable_fused;
use lgbm_compute::kernels::partition::{DeviceLeafSplits, ResidentPermPartition};
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::{BinColumn, ResidentBinWidth};

/// Deterministic LCG (no rand dep) — the spike corpus generator pattern.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// One synthetic feature column: `num_data` bins in `[0, num_bin)`.
fn column(seed: u64, num_data: usize, num_bin: u32) -> Vec<u32> {
    let mut lcg = Lcg(seed);
    (0..num_data).map(|_| lcg.next_u32() % num_bin).collect()
}

/// One split step of the parity walk: partition `perm[begin..begin+count)` on
/// `feature` with the given route params, writing child ranges into `leaf_id`.
#[allow(clippy::too_many_arguments)]
struct Step {
    feature: usize,
    begin: usize,
    count: usize,
    leaf_id: usize,
    min_bin: u32,
    max_bin: u32,
    default_bin: u32,
    most_freq_bin: u32,
    missing_type: u8,
    default_left: bool,
    threshold: u32,
}

/// The adversarial multi-split step list (shared by the anchor-parity walk and the
/// BC-fusion A/B): MissingType::{None, Zero, NaN} × default_left × mfb-varied, plus a
/// block-boundary-straddling grandchild and a small single-block tail.
fn walk_steps() -> Vec<Step> {
    vec![
        Step { feature: 0, begin: 0, count: 700, leaf_id: 0, min_bin: 1, max_bin: 15, default_bin: 16, most_freq_bin: 0, missing_type: 0, default_left: false, threshold: 7 },
        Step { feature: 1, begin: 0, count: 350, leaf_id: 1, min_bin: 1, max_bin: 7, default_bin: 2, most_freq_bin: 0, missing_type: 1, default_left: true, threshold: 3 },
        Step { feature: 2, begin: 350, count: 350, leaf_id: 2, min_bin: 0, max_bin: 10, default_bin: 0, most_freq_bin: 3, missing_type: 2, default_left: false, threshold: 5 },
        Step { feature: 0, begin: 100, count: 400, leaf_id: 3, min_bin: 1, max_bin: 15, default_bin: 4, most_freq_bin: 4, missing_type: 1, default_left: false, threshold: 11 },
        Step { feature: 1, begin: 620, count: 80, leaf_id: 4, min_bin: 1, max_bin: 7, default_bin: 0, most_freq_bin: 2, missing_type: 2, default_left: true, threshold: 1 },
    ]
}

/// The BC-FUSION (`LGBM_PARTITION_FUSE_BC`) produces BYTE-IDENTICAL results to the
/// default 3-launch partition — pinning the "folding stage B into the scatter changes
/// nothing" invariant DIRECTLY on the runnable cubecl-cpu lane (the partition kernels
/// lower there, unlike the staged scan family). Runs the SAME adversarial multi-split
/// walk with the fusion forced OFF then ON and asserts the final perm + every split's
/// child ranges match bit-for-bit. Uses `set_partition_fuse_bc_override` (the env gate
/// is read-once) and restores it to `None` at the end.
#[test]
fn partition_bc_fusion_byte_identical_to_three_launch() {
    use cubecl::prelude::CubeElement;
    use lgbm_compute::kernels::partition::set_partition_fuse_bc_override;

    let client = cpu_client();
    let num_data = 700usize;
    let num_bins = [16u32, 8, 11];
    let cols: Vec<Vec<u32>> =
        (0..3).map(|f| column(0x5eed + f as u64, num_data, num_bins[f])).collect();
    let mut concat_u8: Vec<u8> = Vec::with_capacity(3 * num_data);
    for col in &cols {
        concat_u8.extend(col.iter().map(|&b| b as u8));
    }
    let bins_handle = client.create_from_slice(u8::as_bytes(&concat_u8));

    // Run the whole walk under one arm; return (final perm, per-step child-range sextuples).
    let run_arm = |fuse_bc: bool| -> (Vec<u32>, Vec<[i32; 6]>) {
        set_partition_fuse_bc_override(Some(fuse_bc));
        let rp = ResidentPermPartition::new(&client, num_data).expect("state alloc");
        let leaf_splits = DeviceLeafSplits::new(&client, 8).expect("ranges alloc");
        let mut ranges = Vec::new();
        for s in walk_steps() {
            rp.partition_leaf(
                &client, &bins_handle, ResidentBinWidth::U8, s.feature * num_data, 3 * num_data,
                num_bins[s.feature], s.min_bin, s.max_bin, s.default_bin, s.most_freq_bin,
                s.missing_type, s.default_left, s.threshold, &leaf_splits, s.leaf_id,
                s.begin as i32, s.count as i32,
            )
            .expect("partition_leaf");
            let cr = leaf_splits.read_split(&client, s.leaf_id);
            ranges.push([
                cr.left_start, cr.left_end, cr.left_count, cr.right_start, cr.right_end,
                cr.right_count,
            ]);
        }
        (rp.read_perm(&client), ranges)
    };

    let (perm_off, ranges_off) = run_arm(false);
    let (perm_on, ranges_on) = run_arm(true);
    set_partition_fuse_bc_override(None);

    assert_eq!(perm_off, perm_on, "BC-fused perm diverged from the 3-launch perm");
    assert_eq!(ranges_off, ranges_on, "BC-fused child ranges diverged from the 3-launch ranges");

    // SharedMemory BC-fusion arm: on the cubecl-cpu runtime `partition_bc_fused` is
    // FALSE (the SMEM kernel can't lower there), so forcing the SMEM gate ON must
    // FALL BACK to the 3-launch path and stay byte-identical — pinning that the cpu
    // anchor is never routed into the real-device-only kernel. (The SMEM kernel's own
    // bit-exactness is validated on real CUDA via the driver bit-identical-preds A/B.)
    use lgbm_compute::kernels::partition::{partition_bc_fused, set_partition_fuse_bc_smem_override};
    assert!(
        !partition_bc_fused(&client),
        "on the cubecl-cpu runtime the SMEM BC-fusion must be gated OFF (no cross-unit \
         SharedMemory), so partition_bc_fused must be false with no override set"
    );
    set_partition_fuse_bc_smem_override(Some(true));
    assert!(
        !partition_bc_fused(&client),
        "even with LGBM_PARTITION_FUSE_BC_SMEM forced ON, the cpu runtime must stay on \
         the 3-launch path (real-device gate)"
    );
    let (perm_smem, ranges_smem) = run_arm(false);
    set_partition_fuse_bc_smem_override(None);
    assert_eq!(perm_off, perm_smem, "SMEM-gate-on cpu run diverged (fallback broken)");
    assert_eq!(ranges_off, ranges_smem, "SMEM-gate-on cpu ranges diverged (fallback broken)");

    // Non-vacuity: the root split actually routed rows both ways.
    assert!(
        ranges_off[0][2] > 0 && ranges_off[0][5] > 0,
        "root split must be non-trivial (both children non-empty), got {:?}",
        ranges_off[0]
    );
}

/// Layer 1 — the multi-split kernel-parity walk. 700 rows forces ≥3 scan blocks
/// (`scan_block_size` floors at 256), and the step list walks numeric routing
/// through MissingType::{None, Zero, NaN} × default_left × a min_is_max-free and
/// mfb-varied param set — the same axes the `partition_parity` fan-out pins.
#[test]
fn resident_perm_partition_matches_host_anchor_multi_split() {
    let client = cpu_client();
    let num_data = 700usize;
    let num_bins = [16u32, 8, 11];
    let cols: Vec<Vec<u32>> =
        (0..3).map(|f| column(0x5eed + f as u64, num_data, num_bins[f])).collect();

    // The RESIDENT concat buffer at uniform u8 width (all num_bin ≤ 256), the
    // exact feature-major layout `upload_resident_bins` produces.
    let mut concat_u8: Vec<u8> = Vec::with_capacity(3 * num_data);
    for col in &cols {
        concat_u8.extend(col.iter().map(|&b| b as u8));
    }
    use cubecl::prelude::CubeElement;
    let bins_handle = client.create_from_slice(u8::as_bytes(&concat_u8));

    let rp = ResidentPermPartition::new(&client, num_data).expect("state alloc + iota");
    let leaf_splits = DeviceLeafSplits::new(&client, 8).expect("ranges alloc");

    // Host mirror: the authoritative anchor perm, updated with the shipped host
    // stable partition after every step.
    let mut host_perm: Vec<u32> = (0..num_data as u32).collect();

    // Iota must match before any split.
    assert_eq!(rp.read_perm(&client), host_perm, "identity seed drift");

    let steps = [
        // Root split: full range, MissingType::None, threshold mid.
        Step { feature: 0, begin: 0, count: 700, leaf_id: 0, min_bin: 1, max_bin: 15, default_bin: 16, most_freq_bin: 0, missing_type: 0, default_left: false, threshold: 7 },
        // Left child on feature 1, MissingType::Zero (mz sentinel), default_left.
        Step { feature: 1, begin: 0, count: 350, leaf_id: 1, min_bin: 1, max_bin: 7, default_bin: 2, most_freq_bin: 0, missing_type: 1, default_left: true, threshold: 3 },
        // Right child on feature 2, MissingType::NaN (mna sentinel), !default_left.
        Step { feature: 2, begin: 350, count: 350, leaf_id: 2, min_bin: 0, max_bin: 10, default_bin: 0, most_freq_bin: 3, missing_type: 2, default_left: false, threshold: 5 },
        // A grandchild sub-range straddling a 256-block boundary, mfb == default_bin
        // (mfb_is_zero coincidence under MissingType::Zero).
        Step { feature: 0, begin: 100, count: 400, leaf_id: 3, min_bin: 1, max_bin: 15, default_bin: 4, most_freq_bin: 4, missing_type: 1, default_left: false, threshold: 11 },
        // A small tail range (single block, < block_size), NaN + default_left.
        Step { feature: 1, begin: 620, count: 80, leaf_id: 4, min_bin: 1, max_bin: 7, default_bin: 0, most_freq_bin: 2, missing_type: 2, default_left: true, threshold: 1 },
    ];

    for (si, s) in steps.iter().enumerate() {
        // DEVICE: in-place resident partition (3 launches, no readback).
        rp.partition_leaf(
            &client,
            &bins_handle,
            ResidentBinWidth::U8,
            s.feature * num_data,
            3 * num_data,
            num_bins[s.feature],
            s.min_bin,
            s.max_bin,
            s.default_bin,
            s.most_freq_bin,
            s.missing_type,
            s.default_left,
            s.threshold,
            &leaf_splits,
            s.leaf_id,
            s.begin as i32,
            s.count as i32,
        )
        .unwrap_or_else(|e| panic!("step {si}: resident partition_leaf failed: {e:?}"));

        // HOST anchor: the shipped fused stable partition over the same sub-range.
        let full_col = BinColumn::new(cols[s.feature].clone(), num_bins[s.feature]);
        let sub: Vec<u32> = host_perm[s.begin..s.begin + s.count].to_vec();
        let (reordered, split_point) = partition_leaf_stable_fused(
            &full_col,
            &sub,
            num_bins[s.feature],
            s.min_bin,
            s.max_bin,
            s.default_bin,
            s.most_freq_bin,
            s.missing_type,
            s.default_left,
            s.threshold,
        )
        .unwrap_or_else(|e| panic!("step {si}: host anchor failed: {e:?}"));
        host_perm[s.begin..s.begin + s.count].copy_from_slice(&reordered);

        // The WHOLE device perm must equal the host mirror byte-for-byte.
        assert_eq!(
            rp.read_perm(&client),
            host_perm,
            "step {si}: device perm diverged from the host stable-partition anchor"
        );

        // The resident child ranges must carry the host split point.
        let cr = leaf_splits.read_split(&client, s.leaf_id);
        assert_eq!(cr.left_start, s.begin as i32, "step {si}: left_start");
        assert_eq!(cr.left_count, split_point as i32, "step {si}: left_count (split point)");
        assert_eq!(cr.left_end, (s.begin + split_point) as i32, "step {si}: left_end");
        assert_eq!(cr.right_start, (s.begin + split_point) as i32, "step {si}: right_start");
        assert_eq!(cr.right_end, (s.begin + s.count) as i32, "step {si}: right_end");
        assert_eq!(cr.right_count, (s.count - split_point) as i32, "step {si}: right_count");

        // Non-vacuous: at least one step must route rows BOTH ways.
        if si == 0 {
            assert!(
                split_point > 0 && split_point < s.count,
                "step 0 must split non-trivially (corpus/threshold sanity)"
            );
        }
    }
}

/// Layers 2 + 3 — REAL-GPU-only (the u64 atomic build cannot run on cubecl-cpu;
/// see the module doc). `cuda` wins the runtime pick when both features are on.
#[cfg(any(feature = "rocm", feature = "cuda"))]
mod real_gpu_gated {
    use super::*;
    use cubecl::prelude::CubeElement;
    use lgbm_compute::kernels::histogram::{
        build_fix_compact_resident_readback_f64_on, build_fix_compact_resident_rows_handle_f64_on,
    };

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

    /// The host-rows twin routes its build through the CubeCL autotuner by default,
    /// whose benchmark server cannot run on this headless cubecl-cpu test lane — pin
    /// `LGBM_AUTOTUNE=0` (the read-once gate, so this must run before the first
    /// autotune-path launch in the process; both gpu tests call it, same value, so
    /// ordering between them is immaterial). The bench protocol pins the same value.
    fn pin_autotune_off() {
        // SAFETY: test-only, single value, set before (or idempotently after) any
        // reader thread caches the OnceLock — the only readers are these two tests.
        unsafe { std::env::set_var("LGBM_AUTOTUNE", "0") };
    }

    /// Layer 2 — the rows-HANDLE build (rows = an offset view into a device
    /// buffer) is f64-BIT-IDENTICAL to the host-rows twin fed the same row ids.
    #[test]
    fn rows_handle_build_bit_identical_to_host_rows_build() {
        pin_autotune_off();
        let client = gpu_client();
        let num_data = 500usize;
        let num_bins = [16u32, 8];
        let cols: Vec<Vec<u32>> =
            (0..2).map(|f| column(0xb1d + f as u64, num_data, num_bins[f])).collect();
        let mut concat_u8: Vec<u8> = Vec::with_capacity(2 * num_data);
        for col in &cols {
            concat_u8.extend(col.iter().map(|&b| b as u8));
        }
        let bins_handle = client.create_from_slice(u8::as_bytes(&concat_u8));

        // Integer-ish grad/hess (exact in f32) + a shuffled row buffer whose middle
        // sub-range is the "leaf" — the offset view under test.
        let mut lcg = Lcg(0xfeed);
        let gradients: Vec<f32> = (0..num_data).map(|_| (lcg.next_u32() % 17) as f32 - 8.0).collect();
        let hessians: Vec<f32> = vec![1.0f32; num_data];
        let mut perm: Vec<u32> = (0..num_data as u32).collect();
        for i in (1..num_data).rev() {
            perm.swap(i, (lcg.next_u32() as usize) % (i + 1));
        }
        let (begin, count) = (137usize, 251usize);
        let leaf_rows: Vec<u32> = perm[begin..begin + count].to_vec();

        let slot_off: Vec<usize> = vec![0, 2 * num_bins[0] as usize];
        let slot_len = 2 * (num_bins[0] + num_bins[1]) as usize;
        // (slot_off, num_bin, offset, most_freq_bin) — mfb 0 ⇒ compaction offset 1.
        let fix_feats: Vec<(usize, u32, i32, u32)> =
            vec![(slot_off[0], num_bins[0], 1, 0), (slot_off[1], num_bins[1], 1, 0)];
        let sum_g: f64 = leaf_rows.iter().map(|&r| f64::from(gradients[r as usize])).sum();
        let sum_h: f64 = leaf_rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();

        // HOST-ROWS twin (the shipped chain, host gather arm).
        let host = build_fix_compact_resident_readback_f64_on(
            &client,
            bins_handle.clone(),
            ResidentBinWidth::U8,
            2,
            num_data,
            &slot_off,
            slot_len,
            &leaf_rows,
            &gradients,
            &hessians,
            &fix_feats,
            sum_g,
            sum_h,
        )
        .expect("host-rows build");

        // ROWS-HANDLE twin: the same rows via an OFFSET VIEW of the full perm buffer
        // + the resident grad/hess on-device gather.
        let perm_handle = client.create_from_slice(u32::as_bytes(&perm));
        let rows_view = perm_handle
            .clone()
            .offset_start((begin * core::mem::size_of::<u32>()) as u64);
        let grad_h = client.create_from_slice(f32::as_bytes(&gradients));
        let hess_h = client.create_from_slice(f32::as_bytes(&hessians));
        let max_abs = gradients
            .iter()
            .chain(hessians.iter())
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        let (handle, len) = build_fix_compact_resident_rows_handle_f64_on(
            &client,
            bins_handle,
            ResidentBinWidth::U8,
            2,
            num_data,
            &slot_off,
            slot_len,
            rows_view,
            count,
            &fix_feats,
            sum_g,
            sum_h,
            f64::from(max_abs),
            (grad_h, hess_h, num_data),
            None,
        )
        .expect("rows-handle build");
        assert_eq!(len, slot_len);
        let bytes = client.read_one_unchecked(handle);
        let device = f64::from_bytes(&bytes);

        assert_eq!(host.len(), device.len(), "histogram length drift");
        for (i, (a, b)) in host.iter().zip(device.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "cell {i}: rows-handle build diverged from host-rows twin ({a} vs {b})"
            );
        }
        assert!(
            host.iter().any(|&v| v != 0.0),
            "non-vacuous: the leaf histogram must be non-zero"
        );
    }

    /// Layer 3 — full-driver A/B on `GpuBackend<CpuRuntime>`: the resident-perm
    /// arm OFF vs ON must grow BYTE-IDENTICAL trees and layouts. Runs both arms in
    /// ONE test (the override is process-global; no other test in this binary
    /// touches the driver).
    #[test]
    fn driver_resident_perm_arm_grows_byte_identical_tree() {
        use lgbm_compute::kernels::grow_driver::{
            grow_tree_on_device_driver, set_partition_resident_override,
        };
        use lgbm_compute::{GpuBackend, GrowFeature};
        use lgbm_dataset::bin_mapper::{BinType, MissingType};

        pin_autotune_off();
        let num_data = 600usize;
        let num_bins = [12u32, 7, 9];
        // NaN is deliberately absent: the on-device SCAN path does not implement the
        // NA_AS_MISSING forward branch yet (a pre-existing driver scope limit, not a
        // partition concern — the partition NaN fan-out is pinned at layer 1).
        let missing = [MissingType::None, MissingType::Zero, MissingType::None];
        let cols: Vec<Vec<u32>> =
            (0..3).map(|f| column(0xd21f + f as u64, num_data, num_bins[f])).collect();

        let features: Vec<GrowFeature> = (0..3)
            .map(|f| GrowFeature {
                bins: BinColumn::new(cols[f].clone(), num_bins[f]),
                num_bin: num_bins[f],
                offset: 1,
                min_bin: 0,
                max_bin: num_bins[f] - 1,
                default_bin: num_bins[f],
                most_freq_bin: 0,
                missing_type: missing[f],
                bin_upper_bound: (0..num_bins[f]).map(|b| b as f64 + 0.5).collect(),
                real_feature_index: f as i32,
                bin_type: BinType::Numerical,
                bin_to_category: Vec::new(),
                cat_smooth: 10.0,
                cat_l2: 10.0,
                max_cat_threshold: 32,
                max_cat_to_onehot: 4,
                min_data_per_group: 100,
            })
            .collect();
        let mut lcg = Lcg(0x9add);
        let gradients: Vec<f32> =
            (0..num_data).map(|_| (lcg.next_u32() % 21) as f32 - 10.0).collect();
        let hessians: Vec<f32> = vec![1.0f32; num_data];

        let backend = GpuBackend::<GpuRt>::default();
        let client = gpu_client();
        let grow = || {
            grow_tree_on_device_driver(
                &backend, &client, &gradients, &hessians, &features, 8, -1,
            )
            .expect("resident driver grow")
        };

        set_partition_resident_override(Some(false));
        let (tree_off, layout_off) = grow();
        set_partition_resident_override(Some(true));
        let (tree_on, layout_on) = grow();
        set_partition_resident_override(None);

        assert!(tree_off.num_leaves > 2, "corpus must split multiple times (non-vacuous)");

        // Byte-identical trees.
        assert_eq!(tree_off.num_leaves, tree_on.num_leaves, "num_leaves drift");
        assert_eq!(tree_off.split_feature, tree_on.split_feature, "split_feature drift");
        assert_eq!(tree_off.left_child, tree_on.left_child, "left_child drift");
        assert_eq!(tree_off.right_child, tree_on.right_child, "right_child drift");
        let bits = |xs: &[f64]| xs.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&tree_off.threshold), bits(&tree_on.threshold), "threshold drift");
        assert_eq!(bits(&tree_off.leaf_value), bits(&tree_on.leaf_value), "leaf_value drift");
        assert_eq!(
            bits(&tree_off.internal_value),
            bits(&tree_on.internal_value),
            "internal_value drift"
        );

        // Byte-identical layouts (the perm readback path).
        assert_eq!(layout_off.indices, layout_on.indices, "layout indices drift");
        assert_eq!(layout_off.leaf_begin, layout_on.leaf_begin, "leaf_begin drift");
        assert_eq!(layout_off.leaf_count, layout_on.leaf_count, "leaf_count drift");
    }
}
