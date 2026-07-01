//! On-device data partition — `mark → prefix-sum → scatter` (§9, ODL-13).
//!
//! The NEW §9-faithful device data-partition path, PARALLEL to the shipped
//! host-gather [`crate::kernels::partition`] (D-01, never rebuilt/extended). It
//! reproduces the reference `GenDataToLeftBitVector → PrepareOffset →
//! AggregateBlockOffset → SplitInner → CopyDataIndices` pipeline as a
//! `mark → prefix-sum → scatter` row permutation — NEVER sorting (§17). Row order
//! fixes per-leaf f32 accumulation order, so the post-scatter permutation must
//! match the reference bit-for-bit (D-04 CONFIRMED: the reference block-tiled
//! scatter is order-equivalent to a plain single-owner stable partition).
//!
//! ## What lives here (18-02, ODL-13)
//! - [`route_to_left`] — the shared `pub(crate) #[cube]` numeric route decision
//!   (the full comptime flag fan-out, D-02), using branchless `select` stores
//!   only (SP-2, cubecl-cpu MLIR constraint). SINGLE SOURCE both the mark kernel
//!   here and the 18-04 predict tree-walk call (Pitfall 4).
//! - [`find_in_bitset`] + [`route_to_left_categorical`] — the shared categorical
//!   membership route (D-03), also `pub(crate)` for 18-04 reuse.
//! - [`gen_data_to_left_kernel`] / [`gen_data_to_left_categorical_kernel`] — the
//!   per-row native-width (u8/u16/u32) mark kernels.
//! - [`split_inner_scatter_kernel`] — the `SplitInner`/`CopyDataIndices` scatter,
//!   deriving the exclusive left rank from the inclusive scan `[tid-1]`
//!   (Pitfall 2), preserving the `global_thread_index < num_data_in_leaf` guard
//!   (T-18-01).
//! - [`update_data_index_to_leaf_kernel`] — `UpdateDataIndexToLeafIndex`
//!   (row→leaf map, consuming `right_leaf_index`, the §1/§10 ordering invariant).
//! - [`partition_leaf_stable`] / [`partition_categorical_stable`] — the cpu f64
//!   stable-partition ANCHOR (D-04 CONFIRMED, never GPU-vs-GPU, D-12).
//! - [`partition_on_device`] / [`partition_categorical_on_device`] — the full
//!   device `mark → prefix-sum → scatter` fold on any runtime.
//! - [`SplitPacket`] / [`split_tree_structure_packet`] — the 16-int
//!   `SplitTreeStructure` child-stats packet (D-08).
//!
//! Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-13); anchored to
//! the cubecl-cpu f64 fold (D-12), never GPU-vs-GPU.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::kernels::primitives::{prefix_sum_exclusive_u32_on, prefix_sum_inclusive_u16_on};
use crate::kernels::split_info::SplitScalars;
use crate::BinColumn;

// =========================================================================
// Shared route decision (D-02 numeric / D-03 categorical) — the SINGLE SOURCE
// both the partition mark kernel and the 18-04 predict tree-walk call.
// =========================================================================

/// The shared numeric route decision — returns `1` if the row routes LEFT
/// (`lte`), `0` if RIGHT (`gt`). Transcribes the VERBATIM `SplitInner` full flag
/// fan-out (`dense_bin.hpp:314-394` + the `Split()` dispatcher :405-421, mirrored
/// by `xtask/cpp/kernel_capture.cpp::SplitRouteFanout`) with the seven comptime
/// flags (D-02). Uses **branchless `select` stores only** (SP-2): the comptime
/// bools are folded to `i32` consts (`mz`/`mna`/`ftm`/`mdt`) at the top, and every
/// per-row branch is a `select`, so there is no nested-if mutation chain
/// (cubecl-cpu MLIR constraint).
///
/// `missing_type` maps to `(miss_is_zero, miss_is_na)`; `mfb_is_zero`/`mfb_is_na`
/// are the most-freq-bin coincidence flags; `min_is_max` selects the degenerate
/// single-non-default-bin branch; `default_left` steers the missing/NA rows.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_to_left(
    bin: i32,
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
) -> u32 {
    // th = threshold + min_bin (−1 if most_freq_bin == 0); t_zero = min_bin +
    // default_bin (−1 if most_freq_bin == 0). Branchless via select.
    let mfb0 = select(most_freq_bin == 0, 1i32, 0i32);
    let th = threshold + min_bin - mfb0;
    let t_zero = min_bin + default_bin - mfb0;

    // default_indices target (dense_bin.hpp:332-339): gt unless most_freq_bin <=
    // threshold; 1 = gt/right, 0 = lte/left.
    let default_target = select(most_freq_bin <= threshold, 0i32, 1i32);

    // Comptime-folded consts (single source of the flag algebra). `mz`/`mna` gate
    // the missing-sentinel rows; `mdt` = missing_default target; `ftm` = fold the
    // out-of-range rows to the missing route.
    let mz = if miss_is_zero && !mfb_is_zero { 1i32 } else { 0i32 };
    let mna = if miss_is_na && !mfb_is_na { 1i32 } else { 0i32 };
    let mdt = if (miss_is_zero || miss_is_na) && default_left {
        0i32
    } else {
        1i32
    };
    let ftm = if (miss_is_na && mfb_is_na) || (miss_is_zero && mfb_is_zero) {
        1i32
    } else {
        0i32
    };

    let cond_missing_zero = (mz == 1) && (bin == t_zero);
    let cond_missing_na = (mna == 1) && (bin == max_bin);

    let route = if min_is_max {
        // Degenerate min_bin == max_bin branch (dense_bin.hpp:366-391).
        let max_bin_target = select(max_bin <= th, 0i32, 1i32);
        let is_max = bin == max_bin;
        // bin == maxb: (miss_na && !mfb_na) ? mdt : max_bin_target.
        let max_route = select(mna == 1, mdt, max_bin_target);
        // bin != maxb: ftm ? mdt : default_target.
        let nonmax_route = select(ftm == 1, mdt, default_target);
        let base = select(is_max, max_route, nonmax_route);
        // The missing-zero sentinel (bin == t_zero) takes precedence.
        select(cond_missing_zero, mdt, base)
    } else {
        // Main min_bin < max_bin branch (dense_bin.hpp:314-365).
        let in_range = (bin >= min_bin) && (bin <= max_bin);
        let inrange_route = select(bin > th, 1i32, 0i32);
        let oor_route = select(ftm == 1, mdt, default_target);
        let base = select(in_range, inrange_route, oor_route);
        // Missing/NA sentinel rows (bin == t_zero or bin == maxb) override.
        select(cond_missing_zero || cond_missing_na, mdt, base)
    };

    // to_left = 1 when the route is lte/left (route == 0).
    select(route == 0, 1u32, 0u32)
}

/// Shared bitset membership test (`Common::FindInBitset`, `common.h:836-843`;
/// mirrored by `kernel_capture.cpp::FindInBitsetHost`). Returns `1` if bit `pos`
/// is set, `0` otherwise. Preserves the `pos/32 >= n → 0` bound check (T-18-03,
/// bitset OOB); `pub(crate)` for the 18-04 predict cat branch. Branchless: the
/// out-of-range word index is clamped to `0` (the bitset is non-empty, `n >= 1`)
/// and the result forced to `0` via `select` — no divergent control flow.
#[cube]
pub(crate) fn find_in_bitset(bits: &Array<u32>, n: u32, pos: u32) -> u32 {
    let word = pos / 32;
    let safe_word = select(word >= n, 0u32, word);
    let raw = (bits[safe_word as usize] >> (pos % 32)) & 1u32;
    select(word >= n, 0u32, raw)
}

/// The shared categorical route decision — returns `1` if the row routes LEFT
/// (member), `0` if RIGHT (non-member). Transcribes the VERBATIM
/// `SplitCategoricalInner<USE_MIN_BIN=true>` (`dense_bin.hpp:450-483`, mirrored by
/// `kernel_capture.cpp::SplitCategoricalRoute`, D-03): membership via
/// `FindInBitset(bitset, bin − min_bin + offset)` with `offset = (mfb == 0) ? 1 :
/// 0`, and the out-of-[min,max] rows folding to the default direction
/// (`most_freq_bin > 0 && member(most_freq_bin)` ⇒ lte). Branchless `select`.
#[cube]
pub(crate) fn route_to_left_categorical(
    bin: i32,
    min_bin: i32,
    max_bin: i32,
    most_freq_bin: i32,
    bitset: &Array<u32>,
    bitset_len: u32,
) -> u32 {
    let offset = select(most_freq_bin == 0, 1i32, 0i32);
    // default_target: most_freq_bin > 0 && member(most_freq_bin) ⇒ lte(0), else gt(1).
    let mfb_pos = select(most_freq_bin > 0, most_freq_bin, 0i32) as u32;
    let mfb_member = find_in_bitset(bitset, bitset_len, mfb_pos);
    let default_target = select((most_freq_bin > 0) && (mfb_member == 1), 0i32, 1i32);

    let in_range = (bin >= min_bin) && (bin <= max_bin);
    // pos only meaningful in range; clamp to 0 otherwise (guarded by select below).
    let raw_pos = bin - min_bin + offset;
    let safe_pos = select(in_range, raw_pos, 0i32) as u32;
    let member = find_in_bitset(bitset, bitset_len, safe_pos);
    let inrange_route = select(member == 1, 0i32, 1i32);
    let route = select(in_range, inrange_route, default_target);

    select(route == 0, 1u32, 0u32)
}

// =========================================================================
// Mark kernels (GenDataToLeftBitVector, numeric + categorical) — native width.
// =========================================================================

/// `GenDataToLeftBitVectorKernel` (`cuda_data_partition.cu:290`) — per-row numeric
/// mark. One unit PER ROW (`ABSOLUTE_POS`); reads the native-width bin
/// (u8/u16/u32 via the `<B: Int>` monomorph, `u32::cast_from`) and writes
/// `to_left[i] ∈ {0,1}` via the shared [`route_to_left`] decision. Bounds-guarded
/// (`i < bins.len()`, T-18-01).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_data_to_left_kernel<B: Int>(
    bins: &Array<B>,
    to_left: &mut Array<u32>,
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
    let i = ABSOLUTE_POS;
    if i < bins.len() {
        let bin = u32::cast_from(bins[i]) as i32;
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
    }
}

/// `GenDataToLeftBitVectorKernel_Categorical` (`cuda_data_partition.cu:582`) — the
/// membership mark, sharing the PrepareOffset/AggregateBlockOffset/SplitInner
/// machinery; only the per-row decision changes ([`route_to_left_categorical`]).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_data_to_left_categorical_kernel<B: Int>(
    bins: &Array<B>,
    to_left: &mut Array<u32>,
    bitset: &Array<u32>,
    min_bin: i32,
    max_bin: i32,
    most_freq_bin: i32,
    bitset_len: u32,
) {
    let i = ABSOLUTE_POS;
    if i < bins.len() {
        let bin = u32::cast_from(bins[i]) as i32;
        to_left[i] =
            route_to_left_categorical(bin, min_bin, max_bin, most_freq_bin, bitset, bitset_len);
    }
}

// =========================================================================
// Scatter (SplitInner + CopyDataIndices) + UpdateDataIndexToLeafIndex.
// =========================================================================

/// `SplitInnerKernel` + `CopyDataIndicesKernel` (`cuda_data_partition.cu:909-937`)
/// — the stable scatter. `excl_left_rank[i]` is the EXCLUSIVE count of left rows
/// strictly before `i` (the inclusive-scan `[tid-1]` derivation, Pitfall 2). Left
/// rows land at `out[rank]`; right rows at `out[to_left_total + (i − rank)]`
/// (`i − rank` = the exclusive count of right rows before `i`). The
/// `global_thread_index < n` guard is preserved (T-18-01); each unit writes ONE
/// disjoint destination (no atomics). The result is a plain stable partition
/// (left rows in original order, then right rows in original order), byte-equal to
/// the cpu f64 anchor (D-04 CONFIRMED).
#[cube(launch)]
fn split_inner_scatter_kernel(
    data_indices: &Array<u32>,
    to_left: &Array<u32>,
    excl_left_rank: &Array<u32>,
    out: &mut Array<u32>,
    to_left_total: u32,
    n: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let di = data_indices[i];
        let rank = excl_left_rank[i] as usize;
        let go_left = to_left[i] == 1u32;
        let rights_before = i - rank;
        let right_dest = to_left_total as usize + rights_before;
        let dest = select(go_left, rank, right_dest);
        out[dest] = di;
    }
}

/// `UpdateDataIndexToLeafIndexKernel` (`cuda_data_partition.cu:129`) — writes the
/// destination leaf id into the row→leaf map. Left rows keep `left_leaf_index`,
/// right rows take `right_leaf_index` (the §1/§10 Split-before-partition ordering
/// invariant: `right_leaf_index` is the id `CUDATree.Split` returned, Pitfall 3).
/// Independent, scatter-free per-row write; `di` indexes the FULL num_data map.
#[cube(launch)]
fn update_data_index_to_leaf_kernel(
    data_indices: &Array<u32>,
    to_left: &Array<u32>,
    leaf_map: &mut Array<i32>,
    left_leaf_index: i32,
    right_leaf_index: i32,
    n: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let di = data_indices[i] as usize;
        let leaf = select(to_left[i] == 1u32, left_leaf_index, right_leaf_index);
        leaf_map[di] = leaf;
    }
}

// =========================================================================
// cpu f64 stable-partition ANCHOR (D-04 CONFIRMED — plain stable partition).
// =========================================================================

/// The comptime flag fan-out derived from the runtime split params, EXACTLY as
/// `kernel_capture.cpp::SplitRouteFanout` derives them (`:516-520`) and the
/// `Split()` dispatcher (`dense_bin.hpp:405-421`). Returned as plain bools so the
/// host anchor and the device mark kernel launch use the identical flags.
#[derive(Clone, Copy, Debug)]
struct RouteFlags {
    miss_is_zero: bool,
    miss_is_na: bool,
    mfb_is_zero: bool,
    mfb_is_na: bool,
    min_is_max: bool,
    default_left: bool,
}

impl RouteFlags {
    fn derive(
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: u8,
        default_left: bool,
    ) -> Self {
        let miss_is_zero = missing_type == 1;
        let miss_is_na = missing_type == 2;
        let mfb_is_zero = miss_is_zero && (default_bin == most_freq_bin);
        // kernel_capture.cpp:519-520 — max_bin == most_freq_bin + min_bin && mfb > 0.
        let mfb_is_na = miss_is_na && (max_bin == most_freq_bin + min_bin && most_freq_bin > 0);
        RouteFlags {
            miss_is_zero,
            miss_is_na,
            mfb_is_zero,
            mfb_is_na,
            min_is_max: min_bin == max_bin,
            default_left,
        }
    }
}

/// Plain-Rust mirror of [`route_to_left`] — the host anchor's per-row decision.
/// Bit-identical integer routing (returns `true` for LEFT/lte). Kept as a separate
/// transcription so the cpu f64 anchor never launches a kernel (D-12).
#[allow(clippy::too_many_arguments)]
fn route_left_host(
    bin: i32,
    min_bin: i32,
    max_bin: i32,
    default_bin: i32,
    most_freq_bin: i32,
    threshold: i32,
    f: RouteFlags,
) -> bool {
    let mfb0 = i32::from(most_freq_bin == 0);
    let th = threshold + min_bin - mfb0;
    let t_zero = min_bin + default_bin - mfb0;
    let default_target = if most_freq_bin <= threshold { 0 } else { 1 };
    let mz = f.miss_is_zero && !f.mfb_is_zero;
    let mna = f.miss_is_na && !f.mfb_is_na;
    let mdt = if (f.miss_is_zero || f.miss_is_na) && f.default_left {
        0
    } else {
        1
    };
    let ftm = (f.miss_is_na && f.mfb_is_na) || (f.miss_is_zero && f.mfb_is_zero);
    let cond_missing_zero = mz && (bin == t_zero);
    let cond_missing_na = mna && (bin == max_bin);

    let route = if f.min_is_max {
        let max_bin_target = if max_bin <= th { 0 } else { 1 };
        if cond_missing_zero {
            mdt
        } else if bin != max_bin {
            if ftm {
                mdt
            } else {
                default_target
            }
        } else if mna {
            mdt
        } else {
            max_bin_target
        }
    } else if cond_missing_zero || cond_missing_na {
        mdt
    } else if bin < min_bin || bin > max_bin {
        if ftm {
            mdt
        } else {
            default_target
        }
    } else if bin > th {
        1
    } else {
        0
    };
    route == 0
}

/// Plain-Rust mirror of [`route_to_left_categorical`] (returns `true` for
/// LEFT/member). See [`route_left_host`].
fn route_left_categorical_host(
    bin: i32,
    min_bin: i32,
    max_bin: i32,
    most_freq_bin: i32,
    bitset: &[u32],
) -> bool {
    let find = |pos: i32| -> bool {
        if pos < 0 {
            return false;
        }
        let w = (pos as usize) / 32;
        if w >= bitset.len() {
            return false;
        }
        ((bitset[w] >> ((pos as u32) % 32)) & 1) != 0
    };
    let offset = if most_freq_bin == 0 { 1 } else { 0 };
    let default_target = if most_freq_bin > 0 && find(most_freq_bin) {
        0
    } else {
        1
    };
    let route = if bin < min_bin || bin > max_bin {
        default_target
    } else if find(bin - min_bin + offset) {
        0
    } else {
        1
    };
    route == 0
}

/// Validate the split params at the host boundary (SP-4 / T-18-02) before any
/// `launch`. Rejects `num_bin == 0`, `threshold >= num_bin`, any `bin >= num_bin`,
/// and a `data_indices`/`bins` length mismatch.
fn validate_partition(
    bins: &BinColumn,
    num_bin: u32,
    threshold: u32,
    data_indices: &[u32],
) -> Result<(), ComputeError> {
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
    if data_indices.len() != bins.len() {
        return Err(ComputeError::LengthMismatch {
            expected: bins.len(),
            actual: data_indices.len(),
        });
    }
    for i in 0..bins.len() {
        let b = bins.bin(i);
        if b >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange { row: i, bin: b, num_bin });
        }
    }
    Ok(())
}

/// The cpu f64 stable-partition ANCHOR (D-04 CONFIRMED) — numeric route. Over the
/// leaf's `data_indices` slice, left-keepers appear first in original relative
/// order, then right-keepers in original relative order; `split_point` =
/// `count(route_left)`. `bins[i]` is the bin of the row `data_indices[i]`.
///
/// # Errors
/// [`ComputeError::Runtime`]/[`ComputeError::BinIndexOutOfRange`]/
/// [`ComputeError::LengthMismatch`] per [`validate_partition`].
#[allow(clippy::too_many_arguments)]
pub fn partition_leaf_stable(
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
) -> Result<(Vec<u32>, usize), ComputeError> {
    validate_partition(bins, num_bin, threshold, data_indices)?;
    let f = RouteFlags::derive(min_bin, max_bin, default_bin, most_freq_bin, missing_type, default_left);
    let mut left = Vec::with_capacity(bins.len());
    let mut right = Vec::new();
    for (i, &di) in data_indices.iter().enumerate() {
        let bin = bins.bin(i) as i32;
        if route_left_host(
            bin,
            min_bin as i32,
            max_bin as i32,
            default_bin as i32,
            most_freq_bin as i32,
            threshold as i32,
            f,
        ) {
            left.push(di);
        } else {
            right.push(di);
        }
    }
    let split_point = left.len();
    left.extend_from_slice(&right);
    Ok((left, split_point))
}

/// The cpu f64 stable-partition ANCHOR — categorical membership route (D-03).
///
/// # Errors
/// [`ComputeError::Runtime`] if `num_bin == 0`; [`ComputeError::LengthMismatch`]
/// on a `data_indices`/`bins` mismatch; [`ComputeError::BinIndexOutOfRange`] for
/// any `bin >= num_bin`.
pub fn partition_categorical_stable(
    bins: &BinColumn,
    data_indices: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    most_freq_bin: u32,
    bitset: &[u32],
) -> Result<(Vec<u32>, usize), ComputeError> {
    validate_categorical(bins, num_bin, data_indices)?;
    let mut left = Vec::with_capacity(bins.len());
    let mut right = Vec::new();
    for (i, &di) in data_indices.iter().enumerate() {
        let bin = bins.bin(i) as i32;
        if route_left_categorical_host(bin, min_bin as i32, max_bin as i32, most_freq_bin as i32, bitset) {
            left.push(di);
        } else {
            right.push(di);
        }
    }
    let split_point = left.len();
    left.extend_from_slice(&right);
    Ok((left, split_point))
}

/// Categorical host-boundary validation (no scalar threshold to check).
fn validate_categorical(bins: &BinColumn, num_bin: u32, data_indices: &[u32]) -> Result<(), ComputeError> {
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "data_partition(cat): num_bin must be > 0".to_string(),
        });
    }
    if data_indices.len() != bins.len() {
        return Err(ComputeError::LengthMismatch {
            expected: bins.len(),
            actual: data_indices.len(),
        });
    }
    for i in 0..bins.len() {
        let b = bins.bin(i);
        if b >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange { row: i, bin: b, num_bin });
        }
    }
    Ok(())
}

// =========================================================================
// Device mark → prefix-sum → scatter drivers (any runtime).
// =========================================================================

/// Choose a `block_size` keeping `num_blocks <= 1024` (the primitives' single-tile
/// cap) — `block_size >= 256`, growing for large leaves.
fn scan_block_size(n: usize) -> u32 {
    (n.div_ceil(1024) as u32).max(256)
}

/// Run the `mark → prefix-sum → scatter` device permutation over a marked
/// `to_left[]`, reusing the 18-01 u32 exclusive (`AggregateBlockOffset`) + u16
/// inclusive (`PrepareOffset`) scans and the [`split_inner_scatter_kernel`]. The
/// caller supplies `to_left` (from a numeric/categorical mark). Returns
/// `(reordered, split_point)` — the stable permutation of `data_indices`.
fn scatter_marked<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    data_indices: &[u32],
    to_left: &[u32],
) -> Result<(Vec<u32>, usize), ComputeError> {
    let n = data_indices.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }
    let block_size = scan_block_size(n);

    // AggregateBlockOffset (u32 EXCLUSIVE): excl[i] = # left rows strictly before i.
    let excl = prefix_sum_exclusive_u32_on(client, to_left, block_size)?;
    let to_left_total = excl[n - 1] + to_left[n - 1];

    // PrepareOffset (u16 INCLUSIVE): incl[i] = # left rows in [0, i]. Bit-exact
    // cross-check of the [tid-1] inclusive↔exclusive relation (Pitfall 2). Guarded
    // to n <= 65535 for the u16 cell width. IN-02: this scan feeds ONLY the
    // debug_assert below, so the whole block is compiled out of release builds — it
    // must never launch three extra kernels per partition in a release run.
    #[cfg(debug_assertions)]
    if n <= u16::MAX as usize {
        let to_left_u16: Vec<u16> = to_left.iter().map(|&x| x as u16).collect();
        let incl = prefix_sum_inclusive_u16_on(client, &to_left_u16, block_size)?;
        debug_assert_eq!(
            u32::from(incl[n - 1]),
            to_left_total,
            "PrepareOffset inclusive tail must equal AggregateBlockOffset exclusive tail + to_left[n-1]"
        );
    }

    let h_di = client.create_from_slice(u32::as_bytes(data_indices));
    let h_tl = client.create_from_slice(u32::as_bytes(to_left));
    let h_excl = client.create_from_slice(u32::as_bytes(&excl));
    let h_out = client.empty(core::mem::size_of_val(data_indices));

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: every input handle is sized exactly `n` u32 cells and outlives the
    // launch; the kernel bounds-guards `i < n` and each unit writes ONE disjoint
    // `out[dest]` with `dest ∈ [0, n)` (a permutation index). cubecl unsafe
    // confined here (CMP-01 / T-18-01).
    unsafe {
        split_inner_scatter_kernel::launch::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_di, n),
            ArrayArg::from_raw_parts(h_tl, n),
            ArrayArg::from_raw_parts(h_excl, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            to_left_total,
            n as u32,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    let reordered = u32::from_bytes(&bytes).to_vec();
    Ok((reordered, to_left_total as usize))
}

/// The full device numeric `mark → prefix-sum → scatter` fold (any runtime). Runs
/// [`gen_data_to_left_kernel`] (native-width dispatch) then [`scatter_marked`].
/// Returns a `(reordered, split_point)` BYTE-EQUAL to [`partition_leaf_stable`]
/// (D-04). Anchored to the cpu f64 fold, never GPU-vs-GPU (D-12).
///
/// # Errors
/// As [`partition_leaf_stable`].
#[allow(clippy::too_many_arguments)]
pub fn partition_on_device<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
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
) -> Result<(Vec<u32>, usize), ComputeError> {
    validate_partition(bins, num_bin, threshold, data_indices)?;
    let n = data_indices.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }
    let f = RouteFlags::derive(min_bin, max_bin, default_bin, most_freq_bin, missing_type, default_left);
    let to_left = mark_numeric(client, bins, min_bin, max_bin, default_bin, most_freq_bin, threshold, f)?;
    scatter_marked(client, data_indices, &to_left)
}

/// The full device categorical `mark → prefix-sum → scatter` fold (any runtime).
///
/// # Errors
/// As [`partition_categorical_stable`]; also [`ComputeError::Runtime`] if the
/// bitset is empty.
#[allow(clippy::too_many_arguments)]
pub fn partition_categorical_on_device<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bins: &BinColumn,
    data_indices: &[u32],
    num_bin: u32,
    min_bin: u32,
    max_bin: u32,
    most_freq_bin: u32,
    bitset: &[u32],
) -> Result<(Vec<u32>, usize), ComputeError> {
    validate_categorical(bins, num_bin, data_indices)?;
    let n = data_indices.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }
    if bitset.is_empty() {
        return Err(ComputeError::Runtime {
            detail: "data_partition(cat): bitset must be non-empty".to_string(),
        });
    }
    let to_left = mark_categorical(client, bins, min_bin, max_bin, most_freq_bin, bitset)?;
    scatter_marked(client, data_indices, &to_left)
}

/// Launch the numeric mark kernel (native-width dispatch) and read back `to_left`.
#[allow(clippy::too_many_arguments)]
fn mark_numeric<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bins: &BinColumn,
    min_bin: u32,
    max_bin: u32,
    default_bin: u32,
    most_freq_bin: u32,
    threshold: u32,
    f: RouteFlags,
) -> Result<Vec<u32>, ComputeError> {
    use cubecl::prelude::CubeElement;
    let n = bins.len();
    let h_to_left = client.empty(n * core::mem::size_of::<u32>());
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    macro_rules! launch_mark {
        ($w:ty, $slice:expr) => {{
            let h_bins = client.create_from_slice(<$w>::as_bytes($slice));
            // SAFETY: `h_bins` sized `n` native-width cells, `h_to_left` `n` u32
            // cells; both outlive the launch; the kernel bounds-guards `i < n`.
            unsafe {
                gen_data_to_left_kernel::launch::<$w, R>(
                    client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(h_bins, n),
                    ArrayArg::from_raw_parts(h_to_left.clone(), n),
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
        }};
    }
    match bins {
        BinColumn::U8(v) => launch_mark!(u8, v),
        BinColumn::U16(v) => launch_mark!(u16, v),
        BinColumn::U32(v) => launch_mark!(u32, v),
    }

    let bytes = client.read_one_unchecked(h_to_left);
    Ok(u32::from_bytes(&bytes).to_vec())
}

/// Launch the categorical mark kernel (native-width dispatch) and read `to_left`.
fn mark_categorical<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bins: &BinColumn,
    min_bin: u32,
    max_bin: u32,
    most_freq_bin: u32,
    bitset: &[u32],
) -> Result<Vec<u32>, ComputeError> {
    use cubecl::prelude::CubeElement;
    let n = bins.len();
    let h_to_left = client.empty(n * core::mem::size_of::<u32>());
    let h_bitset = client.create_from_slice(u32::as_bytes(bitset));
    let bitset_len = bitset.len() as u32;
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    macro_rules! launch_cat {
        ($w:ty, $slice:expr) => {{
            let h_bins = client.create_from_slice(<$w>::as_bytes($slice));
            // SAFETY: `h_bins` sized `n` native cells, `h_to_left` `n` u32, `h_bitset`
            // `bitset.len()`; all outlive the launch; kernel guards `i < n` and
            // `find_in_bitset` guards the word index (T-18-03).
            unsafe {
                gen_data_to_left_categorical_kernel::launch::<$w, R>(
                    client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(h_bins, n),
                    ArrayArg::from_raw_parts(h_to_left.clone(), n),
                    ArrayArg::from_raw_parts(h_bitset.clone(), bitset.len()),
                    min_bin as i32,
                    max_bin as i32,
                    most_freq_bin as i32,
                    bitset_len,
                );
            }
        }};
    }
    match bins {
        BinColumn::U8(v) => launch_cat!(u8, v),
        BinColumn::U16(v) => launch_cat!(u16, v),
        BinColumn::U32(v) => launch_cat!(u32, v),
    }

    let bytes = client.read_one_unchecked(h_to_left);
    Ok(u32::from_bytes(&bytes).to_vec())
}

/// Host driver for [`update_data_index_to_leaf_kernel`] — writes the destination
/// leaf id into a `num_data`-sized row→leaf map for the marked leaf rows. Consumes
/// `right_leaf_index` (§1/§10 ordering invariant, Pitfall 3). Rows not in the leaf
/// keep the initial `-1`.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `data_indices.len() != to_left.len()`, or
/// [`ComputeError::Runtime`] if any `data_index >= num_data`.
pub fn update_data_index_to_leaf_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    data_indices: &[u32],
    to_left: &[u32],
    num_data: usize,
    left_leaf_index: i32,
    right_leaf_index: i32,
) -> Result<Vec<i32>, ComputeError> {
    if data_indices.len() != to_left.len() {
        return Err(ComputeError::LengthMismatch {
            expected: data_indices.len(),
            actual: to_left.len(),
        });
    }
    for &di in data_indices {
        if di as usize >= num_data {
            return Err(ComputeError::Runtime {
                detail: format!("update_data_index_to_leaf: data_index {di} >= num_data {num_data}"),
            });
        }
    }
    let n = data_indices.len();
    let init = vec![-1i32; num_data];
    let h_map = client.create_from_slice(i32::as_bytes(&init));
    if n == 0 {
        let bytes = client.read_one_unchecked(h_map);
        return Ok(i32::from_bytes(&bytes).to_vec());
    }
    let h_di = client.create_from_slice(u32::as_bytes(data_indices));
    let h_tl = client.create_from_slice(u32::as_bytes(to_left));
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_di`/`h_tl` sized `n`, `h_map` sized `num_data`; all outlive the
    // launch; kernel guards `i < n`; every `data_indices[i] < num_data` (checked).
    unsafe {
        update_data_index_to_leaf_kernel::launch::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_di, n),
            ArrayArg::from_raw_parts(h_tl, n),
            ArrayArg::from_raw_parts(h_map.clone(), num_data),
            left_leaf_index,
            right_leaf_index,
            n as u32,
        );
    }
    let bytes = client.read_one_unchecked(h_map);
    Ok(i32::from_bytes(&bytes).to_vec())
}

// =========================================================================
// 16-int SplitTreeStructure child-stats packet (D-08).
// =========================================================================

/// The 16-int `cuda_split_info_buffer` (`SplitTreeStructureKernel:799-825`) — 8
/// ints + 4 f64 (packed into the upper 8 ints on-device). `smaller`/`larger` are
/// assigned by `leaf_num_data[left] < leaf_num_data[right]` (`:823`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitPacket {
    /// `[left_leaf, left_num_data, left_data_start, right_leaf, right_num_data,
    /// right_data_start, smaller_child_leaf, larger_child_leaf]`.
    pub ints: [i32; 8],
    /// `[left_sum_hessians, right_sum_hessians, left_sum_gradients,
    /// right_sum_gradients]`.
    pub sums: [f64; 4],
}

/// Pack the 16-int `SplitTreeStructure` child-stats packet (D-08). The per-side
/// sums are read from [`SplitScalars`] (the `CUDASplitInfo` record); `smaller`/
/// `larger` follow the `left_num < right_num` branch (`cuda_data_partition.cu:823`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn split_tree_structure_packet(
    left_leaf: i32,
    left_num_data: i32,
    left_data_start: i32,
    right_leaf: i32,
    right_num_data: i32,
    right_data_start: i32,
    scalars: &SplitScalars,
) -> SplitPacket {
    let (smaller, larger) = if left_num_data < right_num_data {
        (left_leaf, right_leaf)
    } else {
        (right_leaf, left_leaf)
    };
    SplitPacket {
        ints: [
            left_leaf,
            left_num_data,
            left_data_start,
            right_leaf,
            right_num_data,
            right_data_start,
            smaller,
            larger,
        ],
        sums: [
            scalars.left_sum_hessians,
            scalars.right_sum_hessians,
            scalars.left_sum_gradients,
            scalars.right_sum_gradients,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    // Golden `basic` PCASE: num_bin=8 min=0 max=7 thr=3 mfb=8, PORDER 0;2;4;6;1;3;5;7.
    #[test]
    fn anchor_basic_matches_golden() {
        let bins = BinColumn::new(vec![1, 5, 3, 7, 0, 4, 2, 6], 8);
        let di: Vec<u32> = (0..8).collect();
        let (order, split) = partition_leaf_stable(&bins, &di, 8, 0, 7, 0, 8, 0, false, 3).unwrap();
        assert_eq!(split, 4);
        assert_eq!(order, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    }

    // The device fold must be BYTE-EQUAL to the anchor (D-04), never GPU-vs-GPU.
    #[test]
    fn device_matches_anchor_basic() {
        let client = cpu_client();
        let bins = BinColumn::new(vec![1, 5, 3, 7, 0, 4, 2, 6], 8);
        let di: Vec<u32> = (0..8).collect();
        let anchor = partition_leaf_stable(&bins, &di, 8, 0, 7, 0, 8, 0, false, 3).unwrap();
        let device = partition_on_device(&client, &bins, &di, 8, 0, 7, 0, 8, 0, false, 3).unwrap();
        assert_eq!(device, anchor);
    }

    // missing_zero_dl0 fan-out case: PORDER 0;2;6;1;3;4;5;7 split 3.
    #[test]
    fn anchor_missing_zero_fanout() {
        let bins = BinColumn::new(vec![1, 3, 2, 5, 3, 6, 4, 3], 8);
        let di: Vec<u32> = (0..8).collect();
        let (order, split) = partition_leaf_stable(&bins, &di, 8, 1, 6, 2, 5, 1, false, 3).unwrap();
        assert_eq!(split, 3);
        assert_eq!(order, vec![0, 2, 6, 1, 3, 4, 5, 7]);
    }

    #[test]
    fn categorical_anchor_and_device_agree() {
        let client = cpu_client();
        // cat_onehot golden: num_bin=6 min=1 max=5 mfb=0 bitset=[2].
        let bins = BinColumn::new(vec![1, 2, 3, 1, 4, 1, 5, 2], 6);
        let di: Vec<u32> = (0..8).collect();
        let anchor = partition_categorical_stable(&bins, &di, 6, 1, 5, 0, &[2]).unwrap();
        assert_eq!(anchor.1, 3);
        assert_eq!(anchor.0, vec![0, 3, 5, 1, 2, 4, 6, 7]);
        let device = partition_categorical_on_device(&client, &bins, &di, 6, 1, 5, 0, &[2]).unwrap();
        assert_eq!(device, anchor);
    }

    #[test]
    fn packet_smaller_larger_branch() {
        let mut s = SplitScalars::default();
        s.left_sum_hessians = 3.0;
        s.right_sum_hessians = 5.0;
        s.left_sum_gradients = -1.0;
        s.right_sum_gradients = 2.0;
        let pk = split_tree_structure_packet(0, 7, 0, 1, 13, 7, &s);
        assert_eq!(pk.ints, [0, 7, 0, 1, 13, 7, 0, 1]); // left smaller
        assert_eq!(pk.sums, [3.0, 5.0, -1.0, 2.0]);
        let pk2 = split_tree_structure_packet(2, 20, 5, 3, 8, 25, &s);
        assert_eq!(pk2.ints[6], 3); // right smaller
        assert_eq!(pk2.ints[7], 2);
    }

    #[test]
    fn rejects_bad_bin_and_threshold() {
        let bins = BinColumn::new(vec![0, 1, 2], 3);
        let di: Vec<u32> = (0..3).collect();
        assert!(matches!(
            partition_leaf_stable(&bins, &di, 3, 0, 2, 0, 3, 0, false, 3),
            Err(ComputeError::Runtime { .. })
        ));
        let bad = BinColumn::U32(vec![0, 9, 1]);
        assert!(matches!(
            partition_leaf_stable(&bad, &di, 3, 0, 2, 0, 1, 0, false, 1),
            Err(ComputeError::BinIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn update_leaf_map_writes_right_leaf() {
        let client = cpu_client();
        let di = vec![0u32, 1, 2, 3];
        let to_left = vec![1u32, 0, 1, 0];
        let map = update_data_index_to_leaf_on(&client, &di, &to_left, 4, 5, 9).unwrap();
        assert_eq!(map, vec![5, 9, 5, 9]);
    }
}
