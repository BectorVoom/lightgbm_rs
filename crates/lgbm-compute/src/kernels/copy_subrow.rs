//! `CopySubrow` row-subset gather + on-device bagging draw — **15-04**.
//!
//! Owning plan: **15-04**. Scope locked by **ODL-04**, **D-05** (block structure),
//! **D-06** (arbitrary/GOSS indices), **D-07** (the gather kernel), **D-08** (cpu
//! f64 anchor), **V5 / T-15-IDX** (out-of-range index rejection BEFORE launch).
//!
//! ## What lives here (design doc §3 `CopySubrowKernel_ColumnData`)
//! - [`copy_subrow_kernel`] — the D-07 gather kernel: one unit per SELECTED row
//!   copies `in[used_indices[local]] → out[local]`, producing the compacted binned
//!   subset for a bagging/subset selection. The kernel body is the real skeleton
//!   (it is trivial and width-generic over `B: Int`).
//! - [`copy_subrow_on`] — the host launcher that validates indices (V5) and drives
//!   the kernel per [`BinColumn`] width.
//! - [`bagging_draw_on`] — the on-device bagging-draw wrapper that reproduces the
//!   host `[in-bag asc] ++ [OOB desc]` draw (D-05).
//!
//! ## Layout-difference warning (Pitfall 3)
//! `CopySubrow` gathers the COLUMN-major store ([`super::column_data`]); it is NOT
//! the row-major partition-local `data[idx * ncol + tx]` buffer of
//! [`super::row_data`] (§13).
//!
//! ## RNG / security (V6, carried from [`super::random`])
//! The bagging draw reuses the deterministic, NON-cryptographic `Random` LCG; it
//! exists ONLY to reproduce the C++ reference bit-for-bit and MUST NEVER be a
//! source of secure randomness.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::BinColumn;

/// `COPY_SUBROW_BLOCK_SIZE` (§3) — one block of selected rows per launch tile.
pub const COPY_SUBROW_BLOCK_SIZE: u32 = 1024;

/// `bagging_rand_block_` (sample_strategy.h:75) — one `Random` per 1024-row block,
/// seeded `bagging_seed + block_index`. Declared LOCALLY (NOT imported from
/// `lgbm-boosting`, which depends on `lgbm-compute` — importing it back is a crate
/// cycle; see the module prohibition).
pub const BAGGING_RAND_BLOCK: i32 = 1024;

/// The D-07 row-subset gather kernel (§3 `CopySubrowKernel_ColumnData`): one unit
/// per SELECTED row `local` copies `out_col[local] = in_col[used_indices[local]]`.
/// Width-generic over `B: Int` (u8/u16/u32) — the bin is an index, byte-faithful
/// across widths (the spike-029 native-width precedent, see [`super::partition`]).
///
/// Tail units (`local >= num_used`) stay idle — the launch rounds the unit count up
/// to a multiple of the cube dim (manual §4 Safe Indexing).
#[cube(launch)]
pub fn copy_subrow_kernel<B: Int>(
    in_col: &Array<B>,
    out_col: &mut Array<B>,
    used_indices: &Array<i32>,
    num_used: u32,
) {
    // `ABSOLUTE_POS` is a `usize` lane index; the scalar `num_used` is `u32`, so
    // widen it for the guard. Arrays index by `usize` in cube (the histogram
    // `resident_bins[... as usize]` idiom, histogram.rs:1106-1108).
    let local = ABSOLUTE_POS;
    if local < num_used as usize {
        // `used_indices[local]` is a non-negative row id (the host validates
        // `0 <= idx < num_data` BEFORE launch, V5/T-15-IDX), so the `as usize`
        // index cast is value-faithful.
        let src = used_indices[local] as usize;
        out_col[local] = in_col[src];
    }
}

/// Host launcher for [`copy_subrow_kernel`]: validate `used_indices` (V5 —
/// `0 <= idx < num_data` for every entry, returning [`ComputeError`] BEFORE any
/// launch, T-15-IDX) then gather `column` into the compacted subset at its native
/// [`BinColumn`] width.
///
/// # Errors
/// [`ComputeError`] if any `used_indices[i] < 0` or `>= num_data`.
pub fn copy_subrow_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    column: &BinColumn,
    used_indices: &[i32],
    num_data: usize,
) -> Result<BinColumn, ComputeError> {
    let _ = (client, column, used_indices, num_data);
    todo!("15-04: CopySubrow host launcher — validate indices (V5) then launch the native-width gather")
}

/// On-device bagging draw: reproduce the host `[in-bag asc] ++ [OOB desc]` layout
/// (D-05) — per-block `Random::new(bagging_seed + block)`, `next_float() as f64 <
/// bagging_fraction` routes in-bag (left, ascending) else OOB (right), then the OOB
/// tail is reversed. Returns `(bag_data_indices, bag_data_cnt)`.
///
/// # Errors
/// [`ComputeError`] on a runtime/launch failure.
pub fn bagging_draw_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    num_data: i32,
    bagging_seed: i32,
    bagging_fraction: f64,
) -> Result<Vec<i32>, ComputeError> {
    let _ = (client, num_data, bagging_seed, bagging_fraction);
    todo!("15-04: on-device bagging draw — per-block RNG route + OOB-tail reverse (D-05)")
}
