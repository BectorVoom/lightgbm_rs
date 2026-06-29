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
use crate::kernels::random::draw_next_float_on;
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
    use cubecl::prelude::CubeElement;

    // --- CR-02: enforce the SAFETY invariant the kernel launch relies on. The kernel
    // reads `in_col[used_indices[local]]` from a device buffer uploaded from `column`
    // (length `column.len()`), but the per-element index check below bounds indices by
    // `num_data` — an INDEPENDENT public parameter. If `num_data > column.len()`, an
    // index in `[column.len(), num_data)` would pass validation yet produce an
    // out-of-bounds device read. Validate the source-buffer relationship at the boundary,
    // BEFORE any launch, so the `SAFETY` comment's `num_data == n_in` is enforced rather
    // than assumed. ---
    if column.len() != num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: column.len(),
        });
    }

    // --- V5 / T-15-IDX boundary validation: every used index must be in
    // `[0, num_data)` BEFORE any unsafe launch (the `partition.rs:233-241`
    // per-index precedent). The C++ raw-pointer subset table has no bounds check;
    // the Rust port adds one so a malformed index can never reach an OOB device
    // read. ---
    for (pos, &idx) in used_indices.iter().enumerate() {
        if idx < 0 {
            return Err(ComputeError::Runtime {
                detail: format!("copy_subrow: used_indices[{pos}] = {idx} is negative"),
            });
        }
        if idx as usize >= num_data {
            return Err(ComputeError::BinIndexOutOfRange {
                row: pos,
                bin: idx as u32,
                num_bin: num_data as u32,
            });
        }
    }

    let num_used = used_indices.len();
    // Same-width empty subset (no launch needed) — keeps the narrow storage.
    if num_used == 0 {
        return Ok(match column {
            BinColumn::U8(_) => BinColumn::U8(Vec::new()),
            BinColumn::U16(_) => BinColumn::U16(Vec::new()),
            BinColumn::U32(_) => BinColumn::U32(Vec::new()),
        });
    }

    let h_used = client.create_from_slice(i32::as_bytes(used_indices));
    let cube_dim = COPY_SUBROW_BLOCK_SIZE;
    let cube_count = (num_used as u32).div_ceil(cube_dim);

    // Dispatch the kernel monomorph on the column's NATIVE width (u8/u16/u32) — the
    // `partition.rs` `launch_native!` idiom — so the subset uploads/reads at the same
    // narrow width and the output `BinColumn` keeps that width (D-07, no widen).
    macro_rules! gather_native {
        ($w:ty, $variant:path, $slice:expr) => {{
            // T-15-PART: guard the output byte sizing against `usize` overflow.
            let elem = core::mem::size_of::<$w>();
            let out_bytes = num_used.checked_mul(elem).ok_or_else(|| ComputeError::Runtime {
                detail: format!(
                    "copy_subrow: output sizing {num_used} * {elem} overflows usize"
                ),
            })?;
            let n_in = $slice.len();
            let h_in = client.create_from_slice(<$w>::as_bytes($slice));
            let h_out = client.empty(out_bytes);
            // SAFETY: `h_in` holds the full source column (`n_in` elems), `h_out` is
            // sized for exactly `num_used` elems, and `h_used` holds `num_used` i32s;
            // all three outlive the launch. The kernel bounds-checks `local <
            // num_used` and each `used_indices[local]` was validated in `[0, num_data)`
            // above (and `num_data == n_in`), so every `in_col[used_indices[local]]`
            // read is in range. All cubecl unsafe is confined here (CMP-01).
            unsafe {
                copy_subrow_kernel::launch::<$w, R>(
                    client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(h_in, n_in),
                    ArrayArg::from_raw_parts(h_out.clone(), num_used),
                    ArrayArg::from_raw_parts(h_used.clone(), num_used),
                    num_used as u32,
                );
            }
            let bytes = client.read_one_unchecked(h_out);
            $variant(<$w>::from_bytes(&bytes).to_vec())
        }};
    }

    let result = match column {
        BinColumn::U8(v) => gather_native!(u8, BinColumn::U8, v),
        BinColumn::U16(v) => gather_native!(u16, BinColumn::U16, v),
        BinColumn::U32(v) => gather_native!(u32, BinColumn::U32, v),
    };
    Ok(result)
}

/// On-device bagging draw: reproduce the host `[in-bag asc] ++ [OOB desc]` index
/// layout (D-05) bit-for-bit. One device RNG task per `BAGGING_RAND_BLOCK`-row block
/// (seeded `bagging_seed + block`) draws the per-row `NextFloat` stream via the
/// Phase-14 [`draw_next_float_on`] launcher; the route is then computed host-side —
/// `(draw as f64) < bagging_fraction` sends a row in-bag (left, ascending) else OOB
/// (right) — and the OOB tail is reversed so the layout reads `[in-bag asc] ++
/// [OOB desc]`. Returns the `bag_data_indices` layout.
///
/// ## Seed-supply seam (Pitfall 5 — iteration-0 anchor only)
/// `seeds[b] = bagging_seed + b` reproduces the host draw at iteration 0 (each block
/// `Random` constructed once from `bagging_seed + block`). Multi-iteration training
/// advances per-block RNG state across iterations; carrying that continuity needs the
/// host to SUPPLY the per-block state — an explicit Phase-21 seam, NOT this phase.
///
/// ## Anchor discipline (D-05 / D-08)
/// The draw stream is anchored to the inline HOST `bag_data_indices` reference, never
/// GPU-vs-GPU (def-f8u-01). The f32 `NextFloat` is promoted to f64 BEFORE the `<
/// bagging_fraction` compare (Pitfall 6) — matching the C++ `NextFloat() < fraction`.
///
/// # Errors
/// [`ComputeError`] on a runtime/launch failure (e.g. draw-output sizing overflow,
/// guarded inside [`draw_next_float_on`]).
pub fn bagging_draw_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    num_data: i32,
    bagging_seed: i32,
    bagging_fraction: f64,
) -> Result<Vec<i32>, ComputeError> {
    let nd = num_data.max(0);
    if nd == 0 {
        return Ok(Vec::new());
    }

    // One block (and one device RNG task) per BAGGING_RAND_BLOCK rows; seed each
    // block `bagging_seed + block` (iteration-0 convention, see the seed-supply
    // seam). `BAGGING_RAND_BLOCK` is a LOCAL const — never imported from
    // `lgbm-boosting` (crate cycle).
    // Explicit ceil-div (NOT `i32::div_ceil`, which collides with the cubecl prelude
    // `DivCeil` trait under `unstable_name_collisions`); mirrors the host reference's
    // `(nd + BLOCK - 1) / BLOCK`.
    let n_blocks = (nd + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK;
    let seeds: Vec<u32> = (0..n_blocks).map(|b| (bagging_seed + b) as u32).collect();

    // Each task draws BAGGING_RAND_BLOCK consecutive NextFloats, row-major: row `i`'s
    // draw is `draws[block*1024 + i%1024] == draws[i]` (block == i/1024), so the flat
    // stream is directly row-indexed. f32-exact via the shared `Random` recurrence.
    let draws = draw_next_float_on(client, &seeds, BAGGING_RAND_BLOCK as u32)?;

    // Host-side route (Open Question 2 — simplest, fully anchored): promote the f32
    // draw to f64 (Pitfall 6) and compare `< bagging_fraction`, filling in-bag left
    // (ascending) / OOB right, then reverse the OOB tail → `[in-bag asc] ++ [OOB desc]`
    // (mirrors `sample_strategy` + the `threading.h:152-155` one-buffer reverse).
    let nd_usize = nd as usize;
    let mut buf = vec![0i32; nd_usize];
    let mut left = 0usize;
    let mut right = nd_usize;
    for (i, &draw_f32) in draws.iter().enumerate().take(nd_usize) {
        let draw = f64::from(draw_f32);
        if draw < bagging_fraction {
            buf[left] = i as i32;
            left += 1;
        } else {
            right -= 1;
            buf[right] = i as i32;
        }
    }
    buf[left..nd_usize].reverse();
    Ok(buf)
}
