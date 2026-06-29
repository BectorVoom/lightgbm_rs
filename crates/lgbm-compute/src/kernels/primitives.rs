//! Shared device primitives (CubeCL `#[cube]` kernels) — full-depth grow-loop subset.
//!
//! Owning plans: filled by **14-03** (full-depth grow-loop subset) and **14-05**
//! (anchor-pinned skeletons). Scope locked by **ODL-01**, **D-01**, **D-02**.
//!
//! ## What lives here
//! - **Prefix-sum** (inclusive/exclusive): block + multi-kernel global scan.
//!   - cpu anchor: single-owner serial fold per block (`CubeDim::new_1d(1)`),
//!     the 3-launch global structure (block scan -> block-totals scan -> add-back)
//!     with ONE reused `client.empty` scratch (RESEARCH Pattern 3, D-05 pre-alloc).
//!   - GPU (`rocm`): the within-plane level uses the cubecl-0.10
//!     `plane_inclusive_sum` / `plane_exclusive_sum` intrinsics, block-wide via a
//!     segmented `SharedMemory` LDS scan + `sync_cube()` (RESEARCH Pattern 2).
//! - **Shuffle reductions** sum/max/min + dot-product.
//!   - cpu anchor: single-owner serial fold (ascending order -> the matched
//!     reduction order that makes the f64 result bit-exact vs a serial Rust fold,
//!     RESEARCH Open Q2 RESOLVED — see [`reduce_sum_f64_on`]).
//!   - GPU (`rocm`): `plane_sum` / `plane_max` / `plane_min` (dot = mul then sum).
//! - **Single-block bitonic argsort** (index-only): the comparator reads
//!   `keys[indices[i]]` and swaps ONLY `indices`, mirroring the AMD-fork
//!   `cuda_algorithms.hpp` `BitonicArgSort_1024` comparator/tie order EXACTLY
//!   (strict `>`, `outer_segment_index` ascending parity). Single-owner serial on
//!   the cpu anchor (the bitonic network is deterministic given the comparator, so
//!   one unit walking each stage in `tid` order is bit-identical to the parallel
//!   form — disjoint compare-swap pairs within a stage). [full depth — D-01]
//!
//! ## Deferred (NOT in this plan)
//! - **Percentile** (weighted/unweighted), **multi-block / `…Global` argsort**, and
//!   **`BitonicArgSortItems`** as anchor-pinned **skeletons** — 14-05 (D-02).
//! - The **recursive `num_blocks > 1024` global scan** (arrays > ~1M elements where
//!   the block-totals no longer fit a single tile and launch 2 must recurse) is
//!   **OUT OF SCOPE this phase and OWNED by Phase 15** (on-device dataset, its first
//!   real consumer). The Phase-14 kernel GUARDS `num_blocks <= 1024`
//!   ([`MAX_GLOBAL_SCAN_BLOCKS`]) with a typed `ComputeError` rather than silently
//!   truncating.
//!
//! ## Analog file
//! `crates/lgbm-compute/src/kernels/histogram.rs` — the in-repo prior art for the
//! `#[cube]` generic-body + thin per-cell-type launch-wrapper convention, the
//! `SharedMemory`/`sync_cube()` LDS idiom, and the `launch_unchecked` SAFETY
//! wrapper. The CPU anchor stays a **plain serial f64 fold** (D-10).
//!
//! ## 14-01 de-risk input consumed (RESEARCH Open Q1 / Pitfall 1)
//! cubecl-hip lowers all four plane scan/reduction intrinsics (~1e-6, no
//! `plane_shuffle_up` fallback); cubecl-cpu has NO plane support, so the cpu
//! anchor is the serial fold and the plane kernels are `rocm`-gated.

use cubecl::prelude::*;

use crate::error::ComputeError;

/// The single-block cap for the bitonic argsort (AMD-fork
/// `BITONIC_SORT_NUM_ELEMENTS`). Inputs longer than this need the multi-block
/// `…Global` extension — a 14-05 skeleton (D-02), out of scope here.
pub const BITONIC_SORT_NUM_ELEMENTS: usize = 1024;

/// The Phase-14 single-tile cap for the global prefix-sum's block-totals scan.
/// `num_blocks <= 1024` keeps launch 2 a single non-recursive block scan; larger
/// inputs (the recursive case) are a Phase-15 concern (see module docs).
pub const MAX_GLOBAL_SCAN_BLOCKS: usize = 1024;

// =========================================================================
// Task 1: block + multi-kernel global prefix-sum (inclusive/exclusive)
// =========================================================================

/// The shared single-owner ordered block-scan body — the SINGLE SOURCE OF TRUTH
/// for the prefix-sum math (mirrors the `hist_fold_body` idiom). Generic over the
/// cell type `N`; the f64 cpu-anchor and f32 hip-mirror launch wrappers both
/// delegate here so the ascending order, the `UNIT_POS == 0` ownership, and the
/// inclusive/exclusive branch exist exactly once.
///
/// One CUBE owns one block (`CUBE_POS_X`); a single unit folds the block's slice
/// `[b*block_size, min((b+1)*block_size, n))` in ascending order and writes the
/// block's TOTAL sum into `block_totals[b]` (identical for inclusive/exclusive —
/// the running accumulator equals the block sum at the end of either branch).
#[cube]
fn block_scan_body<N: Numeric>(
    data: &Array<N>,
    out: &mut Array<N>,
    block_totals: &mut Array<N>,
    block_size: u32,
    n: u32,
    inclusive: u32,
) {
    if UNIT_POS == 0 {
        let bs = block_size as usize;
        let nn = n as usize;
        let b = CUBE_POS_X as usize;
        let start = b * bs;
        let end = start + bs;
        let lim = if end < nn { end } else { nn };
        let mut acc = N::from_int(0);
        let mut i = start;
        while i < lim {
            if inclusive == 1 {
                acc += data[i];
                out[i] = acc;
            } else {
                out[i] = acc;
                acc += data[i];
            }
            i += 1;
        }
        block_totals[b] = acc;
    }
}

/// Exclusive-scan the per-block totals in place (single owner). After this launch
/// `block_totals[b]` holds the sum of ALL blocks strictly before `b` — the base
/// added back to every element of block `b` in [`add_base_body`].
#[cube]
fn scan_block_totals_body<N: Numeric>(block_totals: &mut Array<N>, num_blocks: u32) {
    if UNIT_POS == 0 {
        let nb = num_blocks as usize;
        let mut acc = N::from_int(0);
        let mut b = 0usize;
        while b < nb {
            let t = block_totals[b];
            block_totals[b] = acc;
            acc += t;
            b += 1;
        }
    }
}

/// Add each block's exclusive base (`block_bases[b]`) back to every element of
/// block `b`. One CUBE per block, single owner.
#[cube]
fn add_base_body<N: Numeric>(
    out: &mut Array<N>,
    block_bases: &Array<N>,
    block_size: u32,
    n: u32,
) {
    if UNIT_POS == 0 {
        let bs = block_size as usize;
        let nn = n as usize;
        let b = CUBE_POS_X as usize;
        let start = b * bs;
        let end = start + bs;
        let lim = if end < nn { end } else { nn };
        let base = block_bases[b];
        let mut i = start;
        while i < lim {
            out[i] += base;
            i += 1;
        }
    }
}

// --- thin per-cell-type launch wrappers (f64 cpu anchor / f32 hip mirror) ---

#[cube(launch_unchecked)]
fn block_scan_kernel_f64(
    data: &Array<f64>,
    out: &mut Array<f64>,
    block_totals: &mut Array<f64>,
    block_size: u32,
    n: u32,
    inclusive: u32,
) {
    block_scan_body::<f64>(data, out, block_totals, block_size, n, inclusive);
}

#[cube(launch_unchecked)]
fn block_scan_kernel_f32(
    data: &Array<f32>,
    out: &mut Array<f32>,
    block_totals: &mut Array<f32>,
    block_size: u32,
    n: u32,
    inclusive: u32,
) {
    block_scan_body::<f32>(data, out, block_totals, block_size, n, inclusive);
}

#[cube(launch_unchecked)]
fn scan_block_totals_kernel_f64(block_totals: &mut Array<f64>, num_blocks: u32) {
    scan_block_totals_body::<f64>(block_totals, num_blocks);
}

#[cube(launch_unchecked)]
fn scan_block_totals_kernel_f32(block_totals: &mut Array<f32>, num_blocks: u32) {
    scan_block_totals_body::<f32>(block_totals, num_blocks);
}

#[cube(launch_unchecked)]
fn add_base_kernel_f64(
    out: &mut Array<f64>,
    block_bases: &Array<f64>,
    block_size: u32,
    n: u32,
) {
    add_base_body::<f64>(out, block_bases, block_size, n);
}

#[cube(launch_unchecked)]
fn add_base_kernel_f32(
    out: &mut Array<f32>,
    block_bases: &Array<f32>,
    block_size: u32,
    n: u32,
) {
    add_base_body::<f32>(out, block_bases, block_size, n);
}

/// Validate the prefix-sum host inputs at the V5 boundary (T-14-03-01/02) and
/// return `num_blocks`. Rejects `block_size == 0` and `num_blocks > 1024` (the
/// Phase-15 recursion guard) BEFORE any `launch_unchecked` / device alloc.
fn validate_scan_inputs(n: usize, block_size: u32) -> Result<usize, ComputeError> {
    if block_size == 0 {
        return Err(ComputeError::Runtime {
            detail: "prefix_sum: block_size must be >= 1".to_string(),
        });
    }
    // num_blocks computed in usize (T-14-03-02: no u32 overflow in sizing).
    let num_blocks = n.div_ceil(block_size as usize);
    if num_blocks > MAX_GLOBAL_SCAN_BLOCKS {
        return Err(ComputeError::Runtime {
            detail: format!(
                "prefix_sum: num_blocks {num_blocks} > {MAX_GLOBAL_SCAN_BLOCKS} — the recursive \
                 >1024-block global scan is a Phase-15 concern (owned by the on-device dataset); \
                 use a larger block_size or split the input"
            ),
        });
    }
    Ok(num_blocks)
}

/// Inclusive block+global prefix-sum on the f64 cpu anchor.
///
/// `out[i] = data[0] + data[1] + ... + data[i]`. Routes through the 3-launch
/// global structure (RESEARCH Pattern 3) with ONE reused `client.empty` scratch
/// for the block totals. Bit-exact vs a serial Rust inclusive scan for
/// exactly-representable inputs (D-10 anchor).
///
/// # Errors
/// [`ComputeError::Runtime`] if `block_size == 0` or `num_blocks > 1024`.
pub fn prefix_sum_inclusive_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
    block_size: u32,
) -> Result<Vec<f64>, ComputeError> {
    prefix_sum_f64_on(client, data, block_size, true)
}

/// Exclusive block+global prefix-sum on the f64 cpu anchor.
///
/// `out[i] = data[0] + ... + data[i-1]` (`out[0] == 0`). Same 3-launch structure
/// and reused scratch as [`prefix_sum_inclusive_f64_on`].
///
/// # Errors
/// [`ComputeError::Runtime`] if `block_size == 0` or `num_blocks > 1024`.
pub fn prefix_sum_exclusive_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
    block_size: u32,
) -> Result<Vec<f64>, ComputeError> {
    prefix_sum_f64_on(client, data, block_size, false)
}

fn prefix_sum_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
    block_size: u32,
    inclusive: bool,
) -> Result<Vec<f64>, ComputeError> {
    let n = data.len();
    let num_blocks = validate_scan_inputs(n, block_size)?;
    if n == 0 {
        return Ok(Vec::new());
    }

    let h_in = client.create_from_slice(f64::as_bytes(data));
    // `out` is fully written by launch 1 (every i in [0, n)), so empty is safe.
    let h_out = client.empty(n * core::mem::size_of::<f64>());
    // ONE reused scratch (D-05 pre-alloc): written by launch 1, scanned in place
    // by launch 2, read as the add-back base by launch 3.
    let h_totals = client.empty(num_blocks * core::mem::size_of::<f64>());
    let incl = u32::from(inclusive);

    // SAFETY: `h_in`/`h_out` are sized exactly `n` f64 cells and `h_totals`
    // `num_blocks` cells, each outliving every launch. Each cube owns block
    // `CUBE_POS_X < num_blocks` and folds only indices in `[b*block_size,
    // min((b+1)*block_size, n)) ⊆ [0, n)` (the `lim` clamp), so every `out[i]` /
    // `data[i]` access is host-proven < n and every `block_totals[b]` < num_blocks.
    // cubecl unsafe is confined here (CMP-01, T-14-03-01).
    unsafe {
        block_scan_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_in, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            ArrayArg::from_raw_parts(h_totals.clone(), num_blocks),
            block_size,
            n as u32,
            incl,
        );
        scan_block_totals_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_totals.clone(), num_blocks),
            num_blocks as u32,
        );
        add_base_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            ArrayArg::from_raw_parts(h_totals, num_blocks),
            block_size,
            n as u32,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// Inclusive block+global prefix-sum on f32 cells (the hip-mirror cell type;
/// runs on any runtime). Held to ~1e-6 vs the f64 anchor on the ROCm leg (14-06).
///
/// # Errors
/// [`ComputeError::Runtime`] if `block_size == 0` or `num_blocks > 1024`.
pub fn prefix_sum_inclusive_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    block_size: u32,
) -> Result<Vec<f32>, ComputeError> {
    prefix_sum_f32_on(client, data, block_size, true)
}

/// Exclusive block+global prefix-sum on f32 cells. See
/// [`prefix_sum_inclusive_f32_on`].
///
/// # Errors
/// [`ComputeError::Runtime`] if `block_size == 0` or `num_blocks > 1024`.
pub fn prefix_sum_exclusive_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    block_size: u32,
) -> Result<Vec<f32>, ComputeError> {
    prefix_sum_f32_on(client, data, block_size, false)
}

fn prefix_sum_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    block_size: u32,
    inclusive: bool,
) -> Result<Vec<f32>, ComputeError> {
    let n = data.len();
    let num_blocks = validate_scan_inputs(n, block_size)?;
    if n == 0 {
        return Ok(Vec::new());
    }

    let h_in = client.create_from_slice(f32::as_bytes(data));
    let h_out = client.empty(n * core::mem::size_of::<f32>());
    let h_totals = client.empty(num_blocks * core::mem::size_of::<f32>());
    let incl = u32::from(inclusive);

    // SAFETY: identical bounds contract to `prefix_sum_f64_on`, f32 cells.
    unsafe {
        block_scan_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_in, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            ArrayArg::from_raw_parts(h_totals.clone(), num_blocks),
            block_size,
            n as u32,
            incl,
        );
        scan_block_totals_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_totals.clone(), num_blocks),
            num_blocks as u32,
        );
        add_base_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            ArrayArg::from_raw_parts(h_totals, num_blocks),
            block_size,
            n as u32,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).to_vec())
}
