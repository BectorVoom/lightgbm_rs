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
use crate::kernels::data_partition::partition_on_device;
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

    /// Read leaf `leaf_id`'s child ranges back to the host (TEST/DEBUG ONLY —
    /// production code keeps them resident). Panics if `leaf_id >= num_leaves`.
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
