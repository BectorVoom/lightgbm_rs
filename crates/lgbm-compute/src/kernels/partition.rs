//! `data_partition` cube kernel — the stable row->{left,right} routing.
//!
//! Mirrors `DataPartition::Split` (`data_partition.hpp:101-120`), whose per-row
//! left/right decision is `DenseBin::SplitInner`
//! (`LightGBM/src/io/dense_bin.hpp:314-394`, commit 195c26fc). This transcribes
//! the `MissingType::None` instantiation
//! `SplitInner<MISS_IS_ZERO=false, MISS_IS_NA=false, MFB_IS_ZERO=false,
//! MFB_IS_NA=false, USE_MIN_BIN=true>` — the default numeric routing (no missing
//! handling). Missing/NA routing is not yet implemented.
//!
//! The op returns a STABLE reordered index array — left rows in original
//! relative order followed by right rows in original relative order — plus a
//! `split_point` (= the left-row count; left indices occupy `[0, split_point)`,
//! right `[split_point, len)`). The tree learner owns `leaf_begin_`/
//! `leaf_count_` bookkeeping, so this op returns only the partition, not the
//! leaf-tree state.
//!
//! ## Design
//! The kernel computes a per-row routing flag (`route[i] == 1` ⇒ right/`gt`,
//! `0` ⇒ left/`lte`) faithfully from the C++ `SplitInner` body; the host then
//! does the trivial STABLE two-pass gather (all left rows first in original
//! order, then all right rows). Splitting the work this way keeps the cube kernel
//! a flat per-row map (cubecl-cpu-friendly) while the load-bearing routing
//! decision still lives in the kernel.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::error::ComputeError;
use crate::kernels::data_partition::{
    partition_on_device, route_to_left, scan_block_size, RouteFlags,
};
use crate::BinColumn;

/// Per-row routing map (the `SplitInner` decision, `MissingType::None` path).
///
/// For each row, writes `route[i] = 1` if the row goes RIGHT (`gt_indices`),
/// `0` if LEFT (`lte_indices`), exactly mirroring `dense_bin.hpp:346-365`:
///
/// ```cpp
/// auto th = threshold + min_bin;  (--th if most_freq_bin == 0)
/// // default direction: default_indices = (most_freq_bin <= threshold) ? lte : gt
/// if (bin < minb || bin > maxb) -> default      // USE_MIN_BIN, no-missing
/// else if (bin > th)            -> gt
/// else                          -> lte
/// ```
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn data_partition_kernel<B: Int>(
    // The bin column is NATIVE-WIDTH (u8/u16/u32), read via `u32::cast_from` to a
    // u32 INDEX — value-identical to the prior `u32` monomorph (`u32::cast_from(x:
    // u32)` is the identity cast), so the `<u32>` launch is byte-for-byte the
    // previous kernel. The narrow widths upload 4× fewer bytes and read 4× less
    // device memory (same pattern as the histogram `<B: Int>` path,
    // histogram.rs:1069).
    bins: &Array<B>,
    route: &mut Array<u32>,
    min_bin: i32,
    max_bin: i32,
    threshold: i32,
    most_freq_bin: i32,
) {
    // ONE unit PER ROW (`ABSOLUTE_POS`). The `SplitInner` decision is per-row
    // INDEPENDENT (`route[i] = f(bins[i])`, no cross-row carry) and integer-only,
    // so there is no order to preserve — unlike the histogram f64 fold, which must
    // stay single-owner sequential for bit-exactness. Each unit writes its OWN
    // `route[i]` (disjoint, no atomics). Previously this scanned all rows on a
    // single lane (`UNIT_POS == 0`); the parallel form is bit-identical (the host
    // gather below is unchanged) and lets the GPU use all its lanes.
    let i = ABSOLUTE_POS;
    // Tail units (i >= len) stay idle: the launch rounds the unit count up to a
    // multiple of the cube dim (manual §4 Safe Indexing).
    if i < bins.len() {
        // th = threshold + min_bin; if most_freq_bin == 0 then --th
        // (dense_bin.hpp:322-327). default_to_right = !(most_freq_bin <= threshold)
        // i.e. the default (out-of-[min,max]) rows go gt unless most_freq_bin <=
        // threshold (then they go lte) (:336-339).
        let mut th = threshold + min_bin;
        if most_freq_bin == 0 {
            th -= 1;
        }
        // default (out-of-[min,max]) rows go gt unless `most_freq_bin <=
        // threshold` (then lte) — dense_bin.hpp:336-339. Equivalent to
        // `most_freq_bin > threshold`.
        let default_to_right = most_freq_bin > threshold; // 1=gt, 0=lte
        // Widen the native-width bin to a u32 INDEX, then to i32 for the signed
        // compares — value-identical to the prior `bins[i] as i32` on the `<u32>`
        // monomorph (`u32::cast_from(x: u32)` is the identity).
        let bin = u32::cast_from(bins[i]) as i32;
        // USE_MIN_BIN, no-missing: out-of-[minb,maxb] -> default direction.
        let is_default = bin < min_bin || bin > max_bin;
        let gt = bin > th; // in-range: bin > th -> gt, else lte
        // route = default ? default_to_right : (bin > th)
        let go_right = select(is_default, default_to_right, gt);
        route[i] = select(go_right, 1u32, 0u32);
    }
}


/// **Native** host `data_partition` — the production cpu-anchor path.
///
/// Bit-IDENTICAL to the `data_partition_kernel` cubecl path (one-unit-per-row
/// routing + host gather): the SAME integer `SplitInner` routing decision and the
/// SAME stable two-pass gather (left rows in original order, then right), without
/// the cubecl launch. The op is u32-only so there is no float order to preserve.
/// The cubecl `data_partition_kernel` is retained for the on-device / native-width
/// route ([`data_partition_native_on`]).
///
/// # Errors
/// [`ComputeError::Runtime`] if `num_bin == 0` or `threshold >= num_bin`;
/// [`ComputeError::BinIndexOutOfRange`] if any `bins[i] >= num_bin`.
#[allow(clippy::too_many_arguments)]
pub fn data_partition_cpu_native(
    bins: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> Result<(Vec<u32>, usize), ComputeError> {
    // --- Boundary validation (identical to data_partition_on) ---
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "data_partition: num_bin must be > 0".to_string(),
        });
    }
    if threshold >= num_bin {
        return Err(ComputeError::Runtime {
            detail: format!("data_partition: threshold {threshold} >= num_bin {num_bin}"),
        });
    }
    for (row, &b) in bins.iter().enumerate() {
        if b >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange {
                row,
                bin: b,
                num_bin,
            });
        }
    }

    let n = bins.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }

    // Routing decision (dense_bin.hpp:322-365), integer-only — identical to the
    // kernel: th = threshold + min_bin (−1 if most_freq_bin == 0); out-of-[min,max]
    // rows take the default direction (`most_freq_bin > threshold` ⇒ gt).
    let min_b = min_bin as i32;
    let max_b = max_bin as i32;
    let thr = threshold as i32;
    let mut th = thr + min_b;
    if most_freq_bin == 0 {
        th -= 1;
    }
    let default_to_right = most_freq_bin as i32 > thr;
    let go_right = |b: u32| -> bool {
        let bin = b as i32;
        if bin < min_b || bin > max_b {
            default_to_right
        } else {
            bin > th
        }
    };

    // Stable two-pass gather: left rows (route==0) then right rows (route==1), each
    // in original order. split_point = left-row count.
    let mut reordered: Vec<u32> = Vec::with_capacity(n);
    for (i, &b) in bins.iter().enumerate() {
        if !go_right(b) {
            reordered.push(i as u32);
        }
    }
    let split_point = reordered.len();
    for (i, &b) in bins.iter().enumerate() {
        if go_right(b) {
            reordered.push(i as u32);
        }
    }
    Ok((reordered, split_point))
}


/// Stable two-pass gather of a per-row `route[]` into a `(reordered, split_point)`
/// partition: left rows (`route==0`) in original order, then right rows
/// (`route==1`) in original order; `split_point` = the left-row count. Shared by
/// `data_partition_on` and [`data_partition_native_on`] so the gather tail is
/// byte-identical across the widened and native-width upload paths.
fn gather_route(route: &[u32], n: usize) -> (Vec<u32>, usize) {
    let mut reordered: Vec<u32> = Vec::with_capacity(n);
    for (i, &r) in route.iter().enumerate().take(n) {
        if r == 0 {
            reordered.push(i as u32);
        }
    }
    let split_point = reordered.len();
    for (i, &r) in route.iter().enumerate().take(n) {
        if r != 0 {
            reordered.push(i as u32);
        }
    }
    (reordered, split_point)
}

/// **Native-width** host `data_partition` on ANY runtime — a narrow upload path.
/// Identical routing + stable gather to `data_partition_on`, but uploads
/// the leaf's bins at their NATIVE [`BinColumn`] width (u8/u16/u32) instead of a
/// u32-widened buffer: a U8 column uploads `count × 1` bytes (4× fewer) and launches
/// the `::<u8>` kernel monomorph; U16 → `::<u16>`; U32 → `::<u32>`.
///
/// Returns a `(reordered, split_point)` BYTE-IDENTICAL to `data_partition_on` fed the
/// same column widened to u32 — the u8/u16/u32 kernels read the same bin value via
/// `u32::cast_from`, so the routing (and thus the gather) is value-identical. Bit-EXACT
/// by construction (partition is f64-free).
///
/// # Errors
/// Same as `data_partition_on`: [`ComputeError::Runtime`] if `num_bin == 0` or
/// `threshold >= num_bin`; [`ComputeError::BinIndexOutOfRange`] for any
/// `bins.bin(i) >= num_bin`.
#[allow(clippy::too_many_arguments)]
pub fn data_partition_native_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bins: &BinColumn,
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    threshold: u32,
    most_freq_bin: u32,
) -> Result<(Vec<u32>, usize), ComputeError> {
    use cubecl::prelude::CubeElement;

    // --- Boundary validation, reading each bin via the `BinColumn` widening
    // accessor BEFORE the unsafe create_from_slice + launch. ---
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "data_partition: num_bin must be > 0".to_string(),
        });
    }
    if threshold >= num_bin {
        return Err(ComputeError::Runtime {
            detail: format!("data_partition: threshold {threshold} >= num_bin {num_bin}"),
        });
    }
    let n = bins.len();
    for i in 0..n {
        let bin = bins.bin(i);
        if bin >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange {
                row: i,
                bin,
                num_bin,
            });
        }
    }

    if n == 0 {
        return Ok((Vec::new(), 0));
    }

    let zeros = vec![0u32; n];
    let h_route = client.create_from_slice(u32::as_bytes(&zeros));

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_bins`/`h_route` each allocated for exactly `n` elements and outlive
    // the launch; the kernel bounds-checks `i < n` and writes only indices `0..n`. The
    // narrow upload is `n` elements of the native width (value-faithful — the bin is an
    // index, byte-identical across widths). All cubecl unsafe is confined here.
    macro_rules! launch_native {
        ($w:ty, $slice:expr) => {{
            let h_bins = client.create_from_slice(<$w>::as_bytes($slice));
            unsafe {
                data_partition_kernel::launch::<$w, R>(
                    client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(h_bins, n),
                    ArrayArg::from_raw_parts(h_route.clone(), n),
                    min_bin as i32,
                    max_bin as i32,
                    threshold as i32,
                    most_freq_bin as i32,
                );
            }
        }};
    }
    match bins {
        BinColumn::U8(v) => launch_native!(u8, v),
        BinColumn::U16(v) => launch_native!(u16, v),
        BinColumn::U32(v) => launch_native!(u32, v),
    }

    let bytes = client.read_one_unchecked(h_route);
    let route = u32::from_bytes(&bytes);

    Ok(gather_route(&route, n))
}

// =========================================================================
// Device-resident child ranges — the §9 AggregateBlockOffset "write
// cuda_leaf_data_start/_end/_num_data on device" stage, without the
// split-point readback (grow_driver.rs:1582).
// =========================================================================

/// A leaf's two child row-ranges, as they live (device-resident) in a
/// [`DeviceLeafSplits`] slot. Start/end are OFFSETS into the single resident
/// permutation buffer (`cuda_data_indices_`); the left child occupies
/// `[left_start, left_end)` and the right child `[right_start, right_end)`
/// (`right_start == left_end`, adjacent sub-ranges of the just-partitioned parent
/// span). `left_count == left_end - left_start` is the split point; `right_count`
/// the remainder. Mirrors the reference `cuda_leaf_data_start_` /
/// `cuda_leaf_data_end_` / `cuda_leaf_num_data_` triple (§9), read back ONLY by the
/// test golden — the driver's grow loop consumes them on device by handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildRanges {
    /// Left child's first row offset (== the parent's `p_begin`).
    pub left_start: i32,
    /// Left child's end offset (== `left_start + left_count`, the split point).
    pub left_end: i32,
    /// Left child's row count (the split point = # rows routed LEFT).
    pub left_count: i32,
    /// Right child's first row offset (== `left_end`, adjacent).
    pub right_start: i32,
    /// Right child's end offset (== the parent's `p_begin + p_count`).
    pub right_end: i32,
    /// Right child's row count (`p_count - split_point`).
    pub right_count: i32,
}

/// The number of `i32` fields a [`DeviceLeafSplits`] stores per leaf id
/// (`left_start, left_end, left_count, right_start, right_end, right_count`).
pub const LEAF_SPLIT_STRIDE: usize = 6;

/// A device-resident per-leaf child-range struct (§9 `cuda_leaf_data_start_` /
/// `_end_` / `_num_data_` analog), allocated once and indexed by leaf id. A single
/// `i32` device buffer of `LEAF_SPLIT_STRIDE * num_leaves` cells;
/// [`partition_child_ranges_device`] writes leaf `L`'s six fields into
/// `[6L, 6L+6)` ON DEVICE so the split point never crosses back to the host.
/// The grow loop reads the ranges by handle; only the test golden reads them
/// back ([`Self::read_leaf`]).
pub struct DeviceLeafSplits<R: cubecl::Runtime> {
    /// `LEAF_SPLIT_STRIDE * num_leaves` i32 cells; leaf `L` owns `[6L, 6L+6)`.
    ranges: Handle,
    /// The width (leaf-id capacity) the buffer is sized for.
    num_leaves: usize,
    /// Ties the struct to its CubeCL runtime `R` without storing one.
    _runtime: PhantomData<fn() -> R>,
}

impl<R: cubecl::Runtime> DeviceLeafSplits<R> {
    /// Allocate a zeroed child-range buffer for `num_leaves` leaf ids on the device.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `num_leaves == 0`.
    pub fn new(client: &ComputeClient<R>, num_leaves: usize) -> Result<Self, ComputeError> {
        if num_leaves == 0 {
            return Err(ComputeError::Runtime {
                detail: "DeviceLeafSplits::new: num_leaves must be >= 1".to_string(),
            });
        }
        use cubecl::prelude::CubeElement;
        let zeros = vec![0i32; LEAF_SPLIT_STRIDE * num_leaves];
        let ranges = client.create_from_slice(i32::as_bytes(&zeros));
        Ok(Self {
            ranges,
            num_leaves,
            _runtime: PhantomData,
        })
    }

    /// The leaf-id capacity.
    #[must_use]
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// A borrow of the device child-range i32 buffer handle (read on device by the grow loop).
    #[must_use]
    pub fn ranges_handle(&self) -> &Handle {
        &self.ranges
    }

    /// Read leaf `leaf_id`'s child ranges back to the host. Production use: the
    /// RESIDENT-PERM partition arm's once-per-split split-point readback (the one
    /// small host crossing its row bookkeeping needs — the same readback the
    /// reference `CUDADataPartition::SplitInner` performs); also the test goldens.
    /// Panics if `leaf_id >= num_leaves`.
    #[must_use]
    pub fn read_leaf(&self, client: &ComputeClient<R>, leaf_id: usize) -> ChildRanges {
        use cubecl::prelude::CubeElement;
        assert!(leaf_id < self.num_leaves, "DeviceLeafSplits::read_leaf: leaf_id out of range");
        let bytes = client.read_one_unchecked(self.ranges.clone());
        let all = i32::from_bytes(&bytes);
        let b = LEAF_SPLIT_STRIDE * leaf_id;
        ChildRanges {
            left_start: all[b],
            left_end: all[b + 1],
            left_count: all[b + 2],
            right_start: all[b + 3],
            right_end: all[b + 4],
            right_count: all[b + 5],
        }
    }
}

/// Write leaf `leaf_id`'s six child-range fields into the resident `ranges` buffer
/// ON DEVICE from the device-resident split point + the parent's `[p_begin,
/// p_begin+p_count)` span. Single-owner (`ABSOLUTE_POS == 0`), integer-only — the §9
/// AggregateBlockOffset "write cuda_leaf_data_start/_end/_num_data" stage. No value
/// crosses back to the host: the split point stays resident.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn write_child_ranges_kernel(
    ranges: &mut Array<i32>,
    leaf_id: u32,
    split_point: i32,
    p_begin: i32,
    p_count: i32,
) {
    if ABSOLUTE_POS == 0 {
        let base = (leaf_id * 6) as usize;
        ranges[base] = p_begin; // left_start
        ranges[base + 1] = p_begin + split_point; // left_end
        ranges[base + 2] = split_point; // left_count
        ranges[base + 3] = p_begin + split_point; // right_start
        ranges[base + 4] = p_begin + p_count; // right_end
        ranges[base + 5] = p_count - split_point; // right_count
    }
}

/// Partition a leaf's rows via the §9 `mark → prefix-sum → scatter` and write the
/// child left/right start/end/count into the resident [`DeviceLeafSplits`] slot
/// `leaf_id` ON DEVICE — the split point never crosses back to the host as a scalar
/// the driver consumes.
///
/// The routing + stable scatter reuse the proven §9 device fold
/// ([`partition_on_device`], bit-exact across the full missing-type × default-direction
/// fan-out — the `partition_parity` gates), so the returned `reordered` permutation is
/// byte-identical to [`partition_on_device`] / the cpu f64 anchor
/// ([`partition_on_device`]'s `partition_leaf_stable` reference). The child ranges are
/// then written into the device struct by [`write_child_ranges_kernel`] (static
/// single-owner geometry — never a dynamic cube count).
///
/// `p_begin` / `p_count` are the parent leaf's resident-span offset + length; the two
/// children become the adjacent sub-ranges `[p_begin, p_begin+split_point)` and
/// `[p_begin+split_point, p_begin+p_count)`. Returns the stable-ordered GLOBAL row
/// permutation (`reordered`) the caller scatters into the resident buffer; the split
/// point is NOT returned (it lives, resident, in `leaf_splits[leaf_id]`).
///
/// # Scope
/// The routing scatter still materializes `reordered` on the host inside
/// [`partition_on_device`] — a fully device-side to-left-total + resident scatter
/// (so even `reordered` never crosses back) would require real hardware to author and
/// verify and is not implemented here. What ships here is the resident child-range
/// seam: the child ranges are device-resident and the split point is no longer a host
/// scalar the grow loop reads.
///
/// # Errors
/// [`ComputeError`] from [`partition_on_device`] (bad `num_bin`/threshold, an
/// out-of-range bin, a length mismatch, or a device launch/readback failure); or
/// [`ComputeError::Runtime`] if `leaf_id >= leaf_splits.num_leaves()`.
#[allow(clippy::too_many_arguments)]
pub fn partition_child_ranges_device<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    bins: &BinColumn,
    data_indices: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    default_bin: u32,
    most_freq_bin: u32,
    missing_type: u8,
    default_left: bool,
    threshold: u32,
    leaf_splits: &DeviceLeafSplits<R>,
    leaf_id: usize,
    p_begin: i32,
    p_count: i32,
) -> Result<Vec<u32>, ComputeError> {
    if leaf_id >= leaf_splits.num_leaves {
        return Err(ComputeError::Runtime {
            detail: format!(
                "partition_child_ranges_device: leaf_id {leaf_id} >= num_leaves {}",
                leaf_splits.num_leaves
            ),
        });
    }

    // §9 mark → prefix-sum → scatter (full fan-out, bit-exact to the cpu anchor).
    let (reordered, split_point) = partition_on_device(
        client,
        bins,
        data_indices,
        num_bin,
        min_bin,
        max_bin,
        default_bin,
        most_freq_bin,
        missing_type,
        default_left,
        threshold,
    )?;

    // §9 AggregateBlockOffset: write the child ranges into the resident device slot —
    // the split point stays on device. Static single-owner geometry (never a dynamic
    // cube count).
    // SAFETY: `ranges` is sized `LEAF_SPLIT_STRIDE * num_leaves`; `leaf_id <
    // num_leaves` (checked above), so the six writes `[6*leaf_id, 6*leaf_id+6)` stay
    // in-bounds. Single owner (`ABSOLUTE_POS == 0`). cubecl unsafe confined here.
    let ranges_len = LEAF_SPLIT_STRIDE * leaf_splits.num_leaves;
    unsafe {
        write_child_ranges_kernel::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(leaf_splits.ranges.clone(), ranges_len),
            leaf_id as u32,
            split_point as i32,
            p_begin,
            p_count,
        );
    }

    Ok(reordered)
}

// =========================================================================
// DEVICE-RESIDENT permutation partition (`cuda_data_indices_` residency) —
// the CUDADataPartition-style in-place split of the resident row permutation.
//
// The prior "device" arm (`partition_child_ranges_device` → `partition_on_device`)
// still crossed the host FOUR times per split: the bins_sub host gather + upload,
// the mark readback, the host-driven prefix sum, and the reordered-rows readback.
// This section keeps the WHOLE permutation resident for an entire tree grow:
//   A. `resident_mark_block_scan_kernel` — reads the split feature's column from
//      the RESIDENT concatenated bin buffer through the resident perm sub-range,
//      snapshots the parent's row ids, marks left/right via the SHARED
//      [`route_to_left`] decision, and block-scans the marks (one cube per block,
//      parallel mark + single-owner per-block exclusive scan).
//   B. `resident_scan_totals_write_ranges_kernel` — single-owner exclusive scan of
//      the ≤1024 block totals (in place), stashes the left total (= split point)
//      into the sentinel cell, and writes the six child-range fields into the
//      resident [`DeviceLeafSplits`] slot (the fused `write_child_ranges_kernel`).
//   C. `resident_scatter_kernel` — stable scatter of the snapshot back into the
//      resident perm sub-range (left rows first in original order, then right),
//      exactly [`split_inner_scatter_kernel`]'s dest math.
// 3 launches, ZERO host crossings; the host reads back only the 6-int child
// ranges afterwards (the split-point scalar its bookkeeping needs).
//
// BIT-EXACTNESS: the mark reuses [`route_to_left`] verbatim (the single-source
// decision the `partition_parity` gates pin byte-equal to the cpu f64 anchor);
// the scatter destination math is integer-only and identical to
// `split_inner_scatter_kernel` + `gather_route` (left dest = global exclusive
// left rank; right dest = left total + rights-before), so the resulting
// permutation is byte-equal to `partition_leaf_stable` on the same inputs.
// =========================================================================

/// Identity-fill (`out[i] = i`) — seeds the resident permutation at the root
/// (`0..num_data` in order, the same ascending order the host `perm` iota had).
#[cube(launch)]
fn iota_u32_kernel(out: &mut Array<u32>, n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        out[i] = u32::cast_from(i);
    }
}

/// Stage A of the resident partition: per-block fused mark + snapshot + exclusive
/// block scan over the parent's resident sub-range `perm[p_begin .. p_begin+n)`.
///
/// One cube per block (`CUBE_POS_X`), `block_size` elements each. The parallel
/// phase (all units, stride `CUBE_DIM`) reads `di = perm[p_begin+i]`, snapshots it
/// into `snap[i]`, reads the split feature's bin from the RESIDENT concatenated
/// buffer (`bins[col_off + di]`, native width via the `<B: Int>` monomorph) and
/// writes `to_left[i]` via the SHARED [`route_to_left`] decision (identical
/// comptime flag fan-out to [`gen_data_to_left_kernel`]). After `sync_cube`,
/// unit 0 serially exclusive-scans its block's marks into `local_excl` and books
/// the block total — the same single-owner geometry as the primitives'
/// `block_scan_body`, fused here to save a launch.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn resident_mark_block_scan_kernel<B: Int>(
    bins: &Array<B>,
    perm: &Array<u32>,
    snap: &mut Array<u32>,
    to_left: &mut Array<u32>,
    local_excl: &mut Array<u32>,
    block_totals: &mut Array<u32>,
    col_off: u32,
    p_begin: u32,
    n: u32,
    block_size: u32,
    min_bin: i32,
    max_bin: i32,
    default_bin: i32,
    most_freq_bin: i32,
    threshold: i32,
    #[comptime] miss_is_zero: bool,
    #[comptime] miss_is_na: bool,
    #[comptime] mfb_is_zero: bool,
    #[comptime] mfb_is_na: bool,
    #[comptime] min_is_max: bool,
    #[comptime] default_left: bool,
) {
    let b = CUBE_POS_X as usize;
    let bs = block_size as usize;
    let nn = n as usize;
    let start = b * bs;
    let end = start + bs;
    let lim = if end < nn { end } else { nn };

    // Parallel mark + snapshot (each unit strides the block).
    let mut i = start + UNIT_POS as usize;
    while i < lim {
        let di = perm[p_begin as usize + i];
        snap[i] = di;
        let bin = u32::cast_from(bins[col_off as usize + di as usize]) as i32;
        to_left[i] = route_to_left(
            bin,
            min_bin,
            max_bin,
            default_bin,
            most_freq_bin,
            threshold,
            miss_is_zero,
            miss_is_na,
            mfb_is_zero,
            mfb_is_na,
            min_is_max,
            default_left,
        );
        i += CUBE_DIM as usize;
    }
    sync_cube();

    // Single-owner per-block exclusive scan (visibility of the parallel phase's
    // global writes within the cube is guaranteed by the barrier above).
    if UNIT_POS == 0 {
        let mut acc = 0u32;
        let mut j = start;
        while j < lim {
            local_excl[j] = acc;
            acc += to_left[j];
            j += 1;
        }
        block_totals[b] = acc;
    }
}

/// Stage B: single-owner exclusive scan of the `num_blocks` block totals IN PLACE
/// (after this, `block_totals[b]` = left rows strictly before block `b`), the left
/// TOTAL (= split point) stashed into the sentinel cell `block_totals[num_blocks]`,
/// and the six child-range fields written into the resident [`DeviceLeafSplits`]
/// slot — the fused [`write_child_ranges_kernel`], with the split point sourced
/// ON DEVICE from the scan instead of a host scalar.
#[cube(launch)]
fn resident_scan_totals_write_ranges_kernel(
    block_totals: &mut Array<u32>,
    ranges: &mut Array<i32>,
    num_blocks: u32,
    leaf_id: u32,
    p_begin: i32,
    p_count: i32,
) {
    if ABSOLUTE_POS == 0 {
        let nb = num_blocks as usize;
        let mut acc = 0u32;
        let mut b = 0usize;
        while b < nb {
            let t = block_totals[b];
            block_totals[b] = acc;
            acc += t;
            b += 1;
        }
        block_totals[nb] = acc;
        let sp = i32::cast_from(acc);
        let base = (leaf_id * 6) as usize;
        ranges[base] = p_begin; // left_start
        ranges[base + 1] = p_begin + sp; // left_end
        ranges[base + 2] = sp; // left_count (the split point)
        ranges[base + 3] = p_begin + sp; // right_start
        ranges[base + 4] = p_begin + p_count; // right_end
        ranges[base + 5] = p_count - sp; // right_count
    }
}

/// Stage C: the stable scatter back into the resident perm sub-range. For element
/// `i` of the parent's span, the global exclusive left rank is
/// `block_totals[b] + local_excl[i]`; left rows land at that rank, right rows at
/// `left_total + (i − rank)` — byte-for-byte [`split_inner_scatter_kernel`]'s dest
/// math, reading the snapshot (`snap`, taken in stage A) so the in-place write to
/// `perm` can never race its own read.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn resident_scatter_kernel(
    snap: &Array<u32>,
    to_left: &Array<u32>,
    local_excl: &Array<u32>,
    block_totals: &Array<u32>,
    perm: &mut Array<u32>,
    p_begin: u32,
    n: u32,
    block_size: u32,
    num_blocks: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let b = i / (block_size as usize);
        let excl = block_totals[b] + local_excl[i];
        let total = block_totals[num_blocks as usize];
        let go_left = to_left[i] == 1u32;
        let iu = u32::cast_from(i);
        let right_dest = total + (iu - excl);
        let dest = select(go_left, excl, right_dest);
        perm[p_begin as usize + dest as usize] = snap[i];
    }
}

/// FUSED stage B+C: the stable scatter that ALSO computes its block's exclusive
/// base from the RAW per-block totals — folding [`resident_scan_totals_write_ranges_kernel`]
/// (stage B) INTO the scatter so stage B is NOT a separate launch (the spike095/096
/// host-enqueue lever: stage B is a 1-cube trivial kernel whose whole cost is the
/// ~91µs launch). ONE CUBE PER BLOCK (`CubeCount::Static(num_blocks, 1, 1)`): cube `b`
/// owns block `b`'s elements `[b*block_size, min((b+1)*block_size, n))`. Unit 0 serially
/// sums the raw block totals — `base_b = Σ block_totals[0..b]` (the exclusive base, the
/// value stage B's in-place scan wrote to `block_totals[b]`) and `total = Σ
/// block_totals[0..num_blocks]` (the split point) — into shared memory; after the
/// barrier every unit scatters its strided share with the IDENTICAL dest math as
/// [`resident_scatter_kernel`]. Cube 0 additionally writes the six child-range fields
/// (what stage B wrote). BIT-EXACT: the per-block base and total are the SAME integer
/// sums stage B's exclusive scan produces, so `base_b + local_excl[i]` (and the total)
/// are identical ⇒ the permutation + ranges are byte-equal to the separate-B path.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn resident_scatter_fused_bc_kernel(
    snap: &Array<u32>,
    to_left: &Array<u32>,
    local_excl: &Array<u32>,
    // RAW per-block left counts (stage A output; NOT exclusive-scanned — stage B skipped).
    block_totals: &Array<u32>,
    perm: &mut Array<u32>,
    ranges: &mut Array<i32>,
    p_begin: u32,
    n: u32,
    block_size: u32,
    num_blocks: u32,
    leaf_id: u32,
    p_count: i32,
) {
    let b = CUBE_POS_X;
    // Compute this block's exclusive base (`Σ block_totals[0..b]`) + the left TOTAL
    // (`Σ block_totals[0..num_blocks]`) DIRECTLY from the raw per-block counts. Each unit
    // recomputes independently (a ≤1024-add serial sum) rather than sharing via
    // SharedMemory — cubecl-cpu does NOT share SharedMemory across units (the same
    // primitive the staged scan family can't lower there), and a per-unit recompute keeps
    // the kernel cubecl-cpu-runnable so the fusion is validated LOCALLY + bit-exact. The
    // sums are integer-deterministic, so every unit gets the IDENTICAL base_b / total that
    // stage B's exclusive scan produced.
    let mut acc = 0u32;
    let mut base_b = 0u32;
    let mut k = 0u32;
    while k < num_blocks {
        if k == b {
            base_b = acc; // exclusive base for block b = running sum BEFORE block b
        }
        acc += block_totals[k as usize];
        k += 1;
    }
    let total = acc; // total left = split point
    // Cube 0's unit 0 writes the six child-range fields (what stage B wrote), from `total`.
    if b == 0 && UNIT_POS == 0 {
        let sp = i32::cast_from(total);
        let pb = i32::cast_from(p_begin);
        let base_idx = (leaf_id * 6) as usize;
        ranges[base_idx] = pb; // left_start
        ranges[base_idx + 1] = pb + sp; // left_end
        ranges[base_idx + 2] = sp; // left_count (split point)
        ranges[base_idx + 3] = pb + sp; // right_start
        ranges[base_idx + 4] = pb + p_count; // right_end
        ranges[base_idx + 5] = p_count - sp; // right_count
    }
    // Scatter this block's elements (span index i ∈ [b*block_size, min(., n))),
    // strided by CUBE_DIM. Dest math IDENTICAL to `resident_scatter_kernel`.
    let start = (b * block_size) as usize;
    let nn = n as usize;
    let end_raw = start + block_size as usize;
    let end = if end_raw < nn { end_raw } else { nn };
    let mut i = start + UNIT_POS as usize;
    while i < end {
        let excl = base_b + local_excl[i];
        let go_left = to_left[i] == 1u32;
        let iu = u32::cast_from(i);
        let right_dest = total + (iu - excl);
        let dest = select(go_left, excl, right_dest);
        perm[p_begin as usize + dest as usize] = snap[i];
        i += CUBE_DIM as usize;
    }
}

/// The device-resident row permutation for one tree grow (the `cuda_data_indices_`
/// analog) plus its partition scratch. Allocated ONCE per grow; every split
/// repartitions a sub-range IN PLACE on device (3 launches, no host crossing);
/// the histogram build reads each leaf's rows straight out of [`Self::rows_view`]
/// (an offset Handle view — no per-build `create_from_slice` row upload); the
/// end-of-grow layout rebuild reads the whole buffer back ONCE
/// ([`Self::read_perm`]).
pub struct ResidentPermPartition<R: cubecl::Runtime> {
    /// The resident permutation, `num_data` u32 cells. Identity after `new`.
    perm: Handle,
    /// Stage-A snapshot of the parent span's row ids (scatter input).
    snap: Handle,
    /// Stage-A left/right marks.
    to_left: Handle,
    /// Stage-A per-element exclusive left rank WITHIN its block.
    local_excl: Handle,
    /// `MAX_SCAN_BLOCKS + 1` u32 cells: per-block left totals, exclusive-scanned in
    /// place by stage B; the sentinel cell holds the left TOTAL (split point).
    block_totals: Handle,
    /// Row count every span must stay within.
    num_data: usize,
    _runtime: PhantomData<fn() -> R>,
}

/// The block-count cap the ≤1024-iteration single-owner stage-B scan relies on
/// (mirrors the primitives' `MAX_GLOBAL_SCAN_BLOCKS`; `scan_block_size` guarantees
/// it for any `n`).
const MAX_SCAN_BLOCKS: usize = 1024;

impl<R: cubecl::Runtime> ResidentPermPartition<R> {
    /// Allocate the perm + scratch buffers and seed the permutation with the
    /// identity (`0..num_data`) via one tiny device launch.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `num_data == 0`.
    pub fn new(client: &ComputeClient<R>, num_data: usize) -> Result<Self, ComputeError> {
        if num_data == 0 {
            return Err(ComputeError::Runtime {
                detail: "ResidentPermPartition: num_data must be > 0".to_string(),
            });
        }
        let bytes = num_data * core::mem::size_of::<u32>();
        let perm = client.empty(bytes);
        let snap = client.empty(bytes);
        let to_left = client.empty(bytes);
        let local_excl = client.empty(bytes);
        let block_totals =
            client.empty((MAX_SCAN_BLOCKS + 1) * core::mem::size_of::<u32>());
        let cube_dim = 256u32;
        let cube_count = (num_data as u32).div_ceil(cube_dim);
        // SAFETY: `perm` is sized exactly `num_data` u32 cells and outlives the
        // launch; the kernel bounds-guards `i < n`. cubecl unsafe confined here.
        unsafe {
            iota_u32_kernel::launch::<R>(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(perm.clone(), num_data),
                num_data as u32,
            );
        }
        Ok(Self {
            perm,
            snap,
            to_left,
            local_excl,
            block_totals,
            num_data,
            _runtime: PhantomData,
        })
    }

    /// An offset Handle VIEW of the resident permutation starting at row-slot
    /// `begin` — the leaf's rows `perm[begin..begin+count)` for a histogram build
    /// (`ArrayArg::from_raw_parts(view, count)` at the launch site). No copy, no
    /// upload; `Handle` is ref-counted so the clone is cheap.
    #[must_use]
    pub fn rows_view(&self, begin: usize) -> Handle {
        self.perm
            .clone()
            .offset_start((begin * core::mem::size_of::<u32>()) as u64)
    }

    /// Read the whole resident permutation back to the host — the ONCE-per-grow
    /// tail crossing that feeds the host `LeafPartitionLayout` rebuild.
    #[must_use]
    pub fn read_perm(&self, client: &ComputeClient<R>) -> Vec<u32> {
        use cubecl::prelude::CubeElement;
        let bytes = client.read_one_unchecked(self.perm.clone());
        u32::from_bytes(&bytes).to_vec()
    }

    /// Partition the parent leaf's resident span `perm[p_begin..p_begin+p_count)`
    /// IN PLACE on device (stages A/B/C — 3 launches, no readback) and write the
    /// child ranges into the resident [`DeviceLeafSplits`] slot `leaf_id`. The
    /// split feature's bins are read from the RESIDENT concatenated buffer
    /// (`resident_bins`, feature column at element offset `col_off`, uniform
    /// native `width`).
    ///
    /// The caller reads the split point back afterwards via
    /// [`DeviceLeafSplits::read_leaf`] (the one small per-split host crossing its
    /// row bookkeeping needs).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on a zero `num_bin`, an out-of-range `threshold`,
    /// an out-of-bounds span, or a `leaf_id` outside the ranges struct.
    #[allow(clippy::too_many_arguments)]
    pub fn partition_leaf(
        &self,
        client: &ComputeClient<R>,
        resident_bins: &Handle,
        width: crate::ResidentBinWidth,
        col_off: usize,
        bins_len: usize,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: u8,
        default_left: bool,
        threshold: u32,
        leaf_splits: &DeviceLeafSplits<R>,
        leaf_id: usize,
        p_begin: i32,
        p_count: i32,
    ) -> Result<(), ComputeError> {
        if num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "resident partition: num_bin must be > 0".to_string(),
            });
        }
        if threshold >= num_bin {
            return Err(ComputeError::Runtime {
                detail: format!("resident partition: threshold {threshold} >= num_bin {num_bin}"),
            });
        }
        if leaf_id >= leaf_splits.num_leaves {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "resident partition: leaf_id {leaf_id} >= num_leaves {}",
                    leaf_splits.num_leaves
                ),
            });
        }
        if p_begin < 0
            || p_count <= 0
            || (p_begin as usize).saturating_add(p_count as usize) > self.num_data
        {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "resident partition: span [{p_begin}, {p_begin}+{p_count}) out of range for \
                     num_data {}",
                    self.num_data
                ),
            });
        }

        let n = p_count as usize;
        let block_size = scan_block_size(n);
        let num_blocks = n.div_ceil(block_size as usize);
        debug_assert!(num_blocks <= MAX_SCAN_BLOCKS, "scan_block_size cap violated");

        // The SAME flag fan-out the host anchor + `gen_data_to_left_kernel` derive —
        // single source, so the mark decision is byte-identical.
        let f = RouteFlags::derive(min_bin, max_bin, default_bin, most_freq_bin, missing_type, default_left);

        // ---- Stage A: fused mark + snapshot + per-block exclusive scan. ----
        // SAFETY: `bins` is bound at `bins_len` elements and every read is
        // `col_off + di` with `di < num_data` (upload-time contract) and
        // `col_off + num_data <= bins_len` (checked below); `perm`/`snap`/`to_left`/
        // `local_excl` all hold `num_data` u32 cells and the kernel touches only
        // `[p_begin, p_begin+n)` / `[0, n)` (span checked above); `block_totals`
        // holds `MAX_SCAN_BLOCKS+1` cells and `b < num_blocks <= MAX_SCAN_BLOCKS`.
        // cubecl unsafe confined here.
        if col_off.saturating_add(self.num_data) > bins_len {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "resident partition: feature column [{col_off}, {col_off}+{}) out of range \
                     for resident bins len {bins_len}",
                    self.num_data
                ),
            });
        }
        macro_rules! launch_mark {
            ($w:ty) => {
                unsafe {
                    resident_mark_block_scan_kernel::launch::<$w, R>(
                        client,
                        CubeCount::Static(num_blocks as u32, 1, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(resident_bins.clone(), bins_len),
                        ArrayArg::from_raw_parts(self.perm.clone(), self.num_data),
                        ArrayArg::from_raw_parts(self.snap.clone(), self.num_data),
                        ArrayArg::from_raw_parts(self.to_left.clone(), self.num_data),
                        ArrayArg::from_raw_parts(self.local_excl.clone(), self.num_data),
                        ArrayArg::from_raw_parts(self.block_totals.clone(), MAX_SCAN_BLOCKS + 1),
                        col_off as u32,
                        p_begin as u32,
                        n as u32,
                        block_size,
                        min_bin as i32,
                        max_bin as i32,
                        default_bin as i32,
                        most_freq_bin as i32,
                        threshold as i32,
                        f.miss_is_zero,
                        f.miss_is_na,
                        f.mfb_is_zero,
                        f.mfb_is_na,
                        f.min_is_max,
                        f.default_left,
                    );
                }
            };
        }
        match width {
            crate::ResidentBinWidth::U8 => launch_mark!(u8),
            crate::ResidentBinWidth::U16 => launch_mark!(u16),
            crate::ResidentBinWidth::U32 => launch_mark!(u32),
        }

        let ranges_len = LEAF_SPLIT_STRIDE * leaf_splits.num_leaves;
        if partition_fuse_bc_enabled() {
            // ---- FUSED stage B+C (one launch): the cube-per-block scatter computes its
            // own exclusive block base from the RAW `block_totals` (folding stage B in) and
            // cube 0 writes the child ranges. Bit-exact to the separate B + C path (same
            // integer sums, same dest math), one fewer launch/split — the host-enqueue win.
            // SAFETY: `block_totals` holds the `num_blocks` raw counts (`nb <= MAX_SCAN_BLOCKS`);
            // cube `b < num_blocks` sums `block_totals[0..num_blocks]` (in range) into 2 LDS
            // cells; every scatter dest is a permutation index `< n` so the perm write stays in
            // `[p_begin, p_begin+n) ⊂ [0, num_data)`; `ranges` holds `6*num_leaves` cells and
            // `leaf_id < num_leaves`. cubecl unsafe confined here. ----
            unsafe {
                resident_scatter_fused_bc_kernel::launch::<R>(
                    client,
                    CubeCount::Static(num_blocks as u32, 1, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(self.snap.clone(), self.num_data),
                    ArrayArg::from_raw_parts(self.to_left.clone(), self.num_data),
                    ArrayArg::from_raw_parts(self.local_excl.clone(), self.num_data),
                    ArrayArg::from_raw_parts(self.block_totals.clone(), MAX_SCAN_BLOCKS + 1),
                    ArrayArg::from_raw_parts(self.perm.clone(), self.num_data),
                    ArrayArg::from_raw_parts(leaf_splits.ranges.clone(), ranges_len),
                    p_begin as u32,
                    n as u32,
                    block_size,
                    num_blocks as u32,
                    leaf_id as u32,
                    p_count,
                );
            }
            return Ok(());
        }

        // ---- Stage B: block-totals scan + child-range write (single owner). ----
        // SAFETY: `block_totals` holds `MAX_SCAN_BLOCKS+1` cells (`nb <= MAX_SCAN_BLOCKS`
        // so the sentinel write is in range); `ranges` holds `6*num_leaves` cells and
        // `leaf_id < num_leaves` (checked above). Single owner. cubecl unsafe confined.
        unsafe {
            resident_scan_totals_write_ranges_kernel::launch::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(self.block_totals.clone(), MAX_SCAN_BLOCKS + 1),
                ArrayArg::from_raw_parts(leaf_splits.ranges.clone(), ranges_len),
                num_blocks as u32,
                leaf_id as u32,
                p_begin,
                p_count,
            );
        }

        // ---- Stage C: stable scatter back into the resident span. ----
        // SAFETY: every dest is a permutation index `< n` (left ranks `< total <= n`,
        // right dests `total + rights_before < n`), so the write stays within
        // `[p_begin, p_begin+n) ⊂ [0, num_data)`. Reads only stage-A/B outputs. cubecl
        // unsafe confined here.
        let cube_dim = 256u32;
        let cube_count = (n as u32).div_ceil(cube_dim);
        unsafe {
            resident_scatter_kernel::launch::<R>(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(self.snap.clone(), self.num_data),
                ArrayArg::from_raw_parts(self.to_left.clone(), self.num_data),
                ArrayArg::from_raw_parts(self.local_excl.clone(), self.num_data),
                ArrayArg::from_raw_parts(self.block_totals.clone(), MAX_SCAN_BLOCKS + 1),
                ArrayArg::from_raw_parts(self.perm.clone(), self.num_data),
                p_begin as u32,
                n as u32,
                block_size,
                num_blocks as u32,
            );
        }
        Ok(())
    }
}

/// Read-once `LGBM_PARTITION_FUSE_BC=="1"` — OPT-IN, default OFF (MEASURED
/// NET-NEGATIVE on P100, see below): fold the resident partition's stage B
/// (block-totals scan + child-range write, a 1-cube kernel whose whole cost is the
/// ~91µs launch) INTO the stage-C scatter via [`resident_scatter_fused_bc_kernel`] —
/// one fewer launch per split. Bit-exact by construction (the cube-per-block scatter
/// recomputes the SAME integer exclusive base + total the stage-B scan produced), and
/// validated LOCALLY (the partition kernels lower on cubecl-cpu, unlike the staged
/// scan family) by `partition_bc_fusion_byte_identical_to_three_launch`.
///
/// VERDICT (spike098, P100, 500k×50×100 trees, order-ALTERNATED warm-median of 3,
/// preds BIT-IDENTICAL max_abs=0.0): fusebc 8.15s vs base 7.97s (0.978×, ~180ms
/// SLOWER); drained partition bucket 596→720ms. Root cause: to stay
/// cubecl-cpu-runnable the fused scatter has EACH unit recompute its block base by a
/// serial ≤1024-add sum of the raw totals (cubecl-cpu does not share SharedMemory
/// across units, so a 1-thread-computes-into-SM design corrupts there). That
/// per-unit redundant sum (256 units × up-to-1024 adds × up-to-1024 cubes) costs
/// MORE device time than the ~91µs launch it removes. A SharedMemory version
/// (1 thread computes) would fix the device cost but is real-device-only (loses the
/// local validation) — not worth the two-path complexity for a lever this size. The
/// hatch is KEPT (bit-exact) for hardware where launch throughput is the harder
/// limit (weaker per-launch dispatch / fewer SMs) and the redundant sum is cheap.
#[must_use]
pub fn partition_fuse_bc_enabled() -> bool {
    // Same-session A/B override (mirrors the grow-driver hatches): the env gate is
    // read-once, so an in-process A/B harness / test flips this atomic instead.
    match PARTITION_FUSE_BC_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("LGBM_PARTITION_FUSE_BC").map(|v| v == "1").unwrap_or(false)
    })
}

/// Same-session A/B override for [`partition_fuse_bc_enabled`].
/// 0 = unset (defer to `LGBM_PARTITION_FUSE_BC`), 1 = force ON, 2 = force OFF.
static PARTITION_FUSE_BC_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// Test/harness hook: force the partition BC-fusion ON (`Some(true)`), OFF
/// (`Some(false)`), or defer to the env gate (`None`). Exists so a same-session A/B
/// (the local parity test + the spike) can flip the arm in ONE process — the env
/// gate is read-once. Timing-neutral (`Relaxed` atomic).
pub fn set_partition_fuse_bc_override(v: Option<bool>) {
    let code = match v {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    PARTITION_FUSE_BC_OVERRIDE.store(code, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    #[test]
    fn partition_basic_threshold() {
        // num_bin = 8, min_bin=0, max_bin=7, threshold=3, most_freq_bin=8 (>thr,
        // so out-of-range default would go gt — but all bins are in range here).
        // bin <= 3 -> left; bin > 3 -> right. Stable order preserved per side.
        let bins = vec![1u32, 5, 3, 7, 0, 4, 2, 6];
        let (reordered, split_point) =
            data_partition_cpu_native(&bins, 8, 0, 7, 3, 8).unwrap();
        // left rows (bin<=3) in original order: idx 0(b1),2(b3),4(b0),6(b2)
        // right rows (bin>3): idx 1(b5),3(b7),5(b4),7(b6)
        assert_eq!(split_point, 4);
        assert_eq!(reordered, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    #[test]
    fn partition_rejects_threshold_out_of_range() {
        let err = data_partition_cpu_native(&[0, 1, 2], 3, 0, 2, 3, 3).unwrap_err();
        assert!(matches!(err, ComputeError::Runtime { .. }));
    }

    #[test]
    fn partition_rejects_bad_bin() {
        let err = data_partition_cpu_native(&[0, 9, 1], 3, 0, 2, 1, 3).unwrap_err();
        assert!(matches!(err, ComputeError::BinIndexOutOfRange { .. }));
    }

    // The native-width path must route BYTE-IDENTICALLY to the u32-widened host
    // reference `data_partition_cpu_native` — value-identical routing across the
    // u8/u16/u32 monomorphs (`u32::cast_from`).

    #[test]
    fn partition_native_u8_matches_widened() {
        let client = cpu_client();
        let bins = vec![1u32, 5, 3, 7, 0, 4, 2, 6]; // num_bin=8 -> BinColumn::U8
        let expected = data_partition_cpu_native(&bins, 8, 0, 7, 3, 8).unwrap();
        let col = BinColumn::new(bins.clone(), 8);
        assert!(matches!(col, BinColumn::U8(_)));
        let got = data_partition_native_on(&client, &col, 8, 0, 7, 3, 8).unwrap();
        assert_eq!(got, expected);
        assert_eq!(got.1, 4);
        assert_eq!(got.0, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    #[test]
    fn partition_native_u16_matches_widened() {
        let client = cpu_client();
        // num_bin=512 -> BinColumn::U16. Representative split params.
        let bins = vec![0u32, 300, 64, 511, 7];
        let expected = data_partition_cpu_native(&bins, 512, 0, 511, 63, 0).unwrap();
        let col = BinColumn::new(bins.clone(), 512);
        assert!(matches!(col, BinColumn::U16(_)));
        let got = data_partition_native_on(&client, &col, 512, 0, 511, 63, 0).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn partition_native_u32_matches_widened() {
        let client = cpu_client();
        // num_bin > 65536 -> BinColumn::U32.
        let bins = vec![0u32, 70_000, 100, 65_540, 9];
        let expected = data_partition_cpu_native(&bins, 70_001, 0, 70_000, 50, 0).unwrap();
        let col = BinColumn::new(bins.clone(), 70_001);
        assert!(matches!(col, BinColumn::U32(_)));
        let got = data_partition_native_on(&client, &col, 70_001, 0, 70_000, 50, 0).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn partition_native_rejects_threshold_out_of_range() {
        let client = cpu_client();
        let col = BinColumn::new(vec![0, 1, 2], 3);
        let err = data_partition_native_on(&client, &col, 3, 0, 2, 3, 3).unwrap_err();
        assert!(matches!(err, ComputeError::Runtime { .. }));
    }

    #[test]
    fn partition_native_rejects_bad_bin() {
        let client = cpu_client();
        // bin 9 >= num_bin 3; build a U32 column so the out-of-range value survives.
        let col = BinColumn::U32(vec![0, 9, 1]);
        let err = data_partition_native_on(&client, &col, 3, 0, 2, 1, 3).unwrap_err();
        assert!(matches!(err, ComputeError::BinIndexOutOfRange { .. }));
    }
}
