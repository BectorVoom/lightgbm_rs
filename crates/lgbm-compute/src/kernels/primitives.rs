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

// =========================================================================
// Task 2: shuffle reductions (sum / max / min, dot-product)
// =========================================================================

/// Single-owner ordered SUM fold — ascending index order (the matched reduction
/// order, Open Q2). Writes the scalar result to `out[0]`.
#[cube]
fn reduce_sum_body<N: Numeric>(data: &Array<N>, out: &mut Array<N>, n: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let mut acc = N::from_int(0);
        let mut i = 0usize;
        while i < nn {
            acc += data[i];
            i += 1;
        }
        out[0] = acc;
    }
}

/// Single-owner MAX reduction (order-independent; `n >= 1` host-guaranteed).
#[cube]
fn reduce_max_body<N: Numeric>(data: &Array<N>, out: &mut Array<N>, n: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let mut acc = data[0];
        let mut i = 1usize;
        while i < nn {
            let v = data[i];
            if v > acc {
                acc = v;
            }
            i += 1;
        }
        out[0] = acc;
    }
}

/// Single-owner MIN reduction (order-independent; `n >= 1` host-guaranteed).
#[cube]
fn reduce_min_body<N: Numeric>(data: &Array<N>, out: &mut Array<N>, n: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let mut acc = data[0];
        let mut i = 1usize;
        while i < nn {
            let v = data[i];
            if v < acc {
                acc = v;
            }
            i += 1;
        }
        out[0] = acc;
    }
}

/// Single-owner DOT-PRODUCT fold — `acc += a[i]*b[i]` in ascending order (matched
/// order, Open Q2). The C++ `ShuffleReduceDotProd` is elementwise-mul then sum.
#[cube]
fn dot_body<N: Numeric>(a: &Array<N>, b: &Array<N>, out: &mut Array<N>, n: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let mut acc = N::from_int(0);
        let mut i = 0usize;
        while i < nn {
            acc += a[i] * b[i];
            i += 1;
        }
        out[0] = acc;
    }
}

// --- thin per-cell-type launch wrappers (f64 cpu anchor / f32 hip mirror) ---

#[cube(launch_unchecked)]
fn reduce_sum_kernel_f64(data: &Array<f64>, out: &mut Array<f64>, n: u32) {
    reduce_sum_body::<f64>(data, out, n);
}
#[cube(launch_unchecked)]
fn reduce_sum_kernel_f32(data: &Array<f32>, out: &mut Array<f32>, n: u32) {
    reduce_sum_body::<f32>(data, out, n);
}
#[cube(launch_unchecked)]
fn reduce_max_kernel_f64(data: &Array<f64>, out: &mut Array<f64>, n: u32) {
    reduce_max_body::<f64>(data, out, n);
}
#[cube(launch_unchecked)]
fn reduce_max_kernel_f32(data: &Array<f32>, out: &mut Array<f32>, n: u32) {
    reduce_max_body::<f32>(data, out, n);
}
#[cube(launch_unchecked)]
fn reduce_min_kernel_f64(data: &Array<f64>, out: &mut Array<f64>, n: u32) {
    reduce_min_body::<f64>(data, out, n);
}
#[cube(launch_unchecked)]
fn reduce_min_kernel_f32(data: &Array<f32>, out: &mut Array<f32>, n: u32) {
    reduce_min_body::<f32>(data, out, n);
}
#[cube(launch_unchecked)]
fn dot_kernel_f64(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>, n: u32) {
    dot_body::<f64>(a, b, out, n);
}
#[cube(launch_unchecked)]
fn dot_kernel_f32(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>, n: u32) {
    dot_body::<f32>(a, b, out, n);
}

/// Which single-input reduction to launch.
enum ReduceOp {
    Sum,
    Max,
    Min,
}

/// f64 single-input reduction on the cpu anchor.
///
/// SUM is bit-exact vs a serial ascending Rust fold (matched-order policy,
/// Open Q2); MAX/MIN are order-independent and bit-exact. Empty input: SUM -> 0.0;
/// MAX/MIN -> [`ComputeError::LengthMismatch`] (no identity element).
fn reduce_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
    op: ReduceOp,
) -> Result<f64, ComputeError> {
    let n = data.len();
    if n == 0 {
        return match op {
            ReduceOp::Sum => Ok(0.0),
            ReduceOp::Max | ReduceOp::Min => Err(ComputeError::LengthMismatch {
                expected: 1,
                actual: 0,
            }),
        };
    }
    let h_in = client.create_from_slice(f64::as_bytes(data));
    // `out[0]` is unconditionally written by the single owner -> empty is safe.
    let h_out = client.empty(core::mem::size_of::<f64>());

    // SAFETY: `h_in` sized exactly `n` f64 cells, `h_out` one cell; both outlive
    // the launch. The single owner reads only `data[0..n)` and writes `out[0]`,
    // both host-proven in range. cubecl unsafe confined here (CMP-01).
    unsafe {
        match op {
            ReduceOp::Sum => reduce_sum_kernel_f64::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(h_in, n),
                ArrayArg::from_raw_parts(h_out.clone(), 1),
                n as u32,
            ),
            ReduceOp::Max => reduce_max_kernel_f64::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(h_in, n),
                ArrayArg::from_raw_parts(h_out.clone(), 1),
                n as u32,
            ),
            ReduceOp::Min => reduce_min_kernel_f64::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(h_in, n),
                ArrayArg::from_raw_parts(h_out.clone(), 1),
                n as u32,
            ),
        }
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes)[0])
}

/// Sum reduction on the f64 cpu anchor (bit-exact, ascending matched order).
///
/// # Errors
/// Never (empty -> 0.0). Returns `Result` for API symmetry with max/min.
pub fn reduce_sum_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
) -> Result<f64, ComputeError> {
    reduce_f64_on(client, data, ReduceOp::Sum)
}

/// Max reduction on the f64 cpu anchor (order-independent, bit-exact).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] on empty input (no identity element).
pub fn reduce_max_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
) -> Result<f64, ComputeError> {
    reduce_f64_on(client, data, ReduceOp::Max)
}

/// Min reduction on the f64 cpu anchor (order-independent, bit-exact).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] on empty input (no identity element).
pub fn reduce_min_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f64],
) -> Result<f64, ComputeError> {
    reduce_f64_on(client, data, ReduceOp::Min)
}

/// Dot-product reduction on the f64 cpu anchor — `sum_i a[i]*b[i]` in ascending
/// order (matched-order policy, Open Q2). Empty -> 0.0.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `a.len() != b.len()`.
pub fn dot_product_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    a: &[f64],
    b: &[f64],
) -> Result<f64, ComputeError> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }
    let h_a = client.create_from_slice(f64::as_bytes(a));
    let h_b = client.create_from_slice(f64::as_bytes(b));
    let h_out = client.empty(core::mem::size_of::<f64>());

    // SAFETY: `h_a`/`h_b` sized exactly `n` f64 cells, `h_out` one cell; all
    // outlive the launch. The single owner reads `a[0..n)`/`b[0..n)` and writes
    // `out[0]`, host-proven in range. cubecl unsafe confined here (CMP-01).
    unsafe {
        dot_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_a, n),
            ArrayArg::from_raw_parts(h_b, n),
            ArrayArg::from_raw_parts(h_out.clone(), 1),
            n as u32,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes)[0])
}

// =========================================================================
// Task 3: single-block index-only bitonic argsort
// =========================================================================

/// Compute the bitonic `(depth, aligned)` for `num_items`, replicating the
/// AMD-fork `BitonicArgSort_1024` alignment loop (`aligned` = next power of two
/// >= `num_items`, `depth` = log2(aligned) + 1).
fn bitonic_depth_aligned(num_items: usize) -> (u32, u32) {
    let mut depth = 1u32;
    let mut aligned = 1u32;
    let mut r = (num_items as u32).saturating_sub(1);
    while r > 0 {
        r >>= 1;
        aligned <<= 1;
        depth += 1;
    }
    (depth, aligned)
}

/// Single-owner index-only bitonic argsort over `indices` (identity start, length
/// `aligned`), keyed by the READ-ONLY `keys` array. Mirrors the AMD-fork
/// `cuda_algorithms.hpp` `BitonicArgSort_1024` comparator EXACTLY:
/// - the comparator reads `keys[indices[tid]]` (indirection) and swaps ONLY
///   `indices` — `keys` is never written (index-only, D-01);
/// - strict `>` comparison; ascending parity `outer_segment_index & 1 == 0`
///   (`ASCENDING`) so ties resolve identically to the C++ reference (Pitfall 5);
/// - the network walks `outer_depth = depth-1 .. 1`, `inner_depth = outer_depth
///   .. depth-1`, comparing `tid` with `tid + half_segment_length` when the
///   half-segment index is even.
///
/// Single owner (`UNIT_POS == 0`) walks every `tid` in stage order. Within one
/// `inner_depth` stage the compare-swap pairs are DISJOINT (each lower element
/// pairs with exactly one higher partner), so the serial `tid` walk is bit-
/// identical to the parallel form with a `sync_cube()` between stages — the cpu
/// anchor needs no barrier (single owner) and no plane op (14-01: cpu has none).
#[cube(launch_unchecked)]
fn bitonic_argsort_kernel(
    keys: &Array<f32>,
    indices: &mut Array<i32>,
    aligned: u32,
    depth: u32,
    ascending: u32,
) {
    if UNIT_POS == 0 {
        let mut od = depth - 1;
        // od runs depth-1 down to 1; when depth == 1 (single element) od == 0 and
        // the guard skips the network entirely. od never decrements below 0.
        while od >= 1 {
            let outer_segment_length = 1u32 << (depth - od);
            let mut inner = od;
            while inner < depth {
                let segment_length = 1u32 << (depth - inner);
                let half = segment_length >> 1;
                let mut tid = 0u32;
                while tid < aligned {
                    let osi = tid / outer_segment_length;
                    let asc = if ascending == 1 {
                        (osi & 1u32) == 0u32
                    } else {
                        (osi & 1u32) == 1u32
                    };
                    let hsi = tid / half;
                    if (hsi & 1u32) == 0u32 {
                        let cmp = tid + half;
                        let ia = indices[tid as usize] as usize;
                        let ib = indices[cmp as usize] as usize;
                        let gt = keys[ia] > keys[ib];
                        if gt == asc {
                            let tmp = indices[tid as usize];
                            indices[tid as usize] = indices[cmp as usize];
                            indices[cmp as usize] = tmp;
                        }
                    }
                    tid += 1;
                }
                inner += 1;
            }
            od -= 1;
        }
    }
}

/// Single-block index-only bitonic argsort on the cpu anchor.
///
/// Returns `(permutation, keys_after)` where `permutation[r]` is the index of the
/// `r`-th smallest (ascending) / largest (descending) key, and `keys_after` is the
/// keys array read back AFTER the sort (always equal to the input — proves the
/// index-only / values-unmoved contract). Internally the keys are sentinel-padded
/// to the next power of two (`+inf` ascending / `-inf` descending) so the padding
/// sorts to the tail; the returned slices are truncated to `keys.len()`.
///
/// `num_items <= 1024` (single block); larger inputs need the multi-block
/// `…Global` argsort — a 14-05 skeleton (D-02).
///
/// # Errors
/// [`ComputeError::Runtime`] if `keys.len() > 1024`.
pub fn bitonic_argsort_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    keys: &[f32],
    ascending: bool,
) -> Result<(Vec<i32>, Vec<f32>), ComputeError> {
    let num_items = keys.len();
    if num_items == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if num_items > BITONIC_SORT_NUM_ELEMENTS {
        return Err(ComputeError::Runtime {
            detail: format!(
                "bitonic_argsort: num_items {num_items} > {BITONIC_SORT_NUM_ELEMENTS} — the \
                 multi-block …Global argsort is a 14-05 skeleton (D-02)"
            ),
        });
    }

    let (depth, aligned) = bitonic_depth_aligned(num_items);
    let al = aligned as usize;
    let sentinel = if ascending {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    let mut padded = keys.to_vec();
    padded.resize(al, sentinel);
    let indices: Vec<i32> = (0..aligned as i32).collect();

    let h_keys = client.create_from_slice(f32::as_bytes(&padded));
    let h_idx = client.create_from_slice(i32::as_bytes(&indices));

    // SAFETY: `h_keys` and `h_idx` are each sized exactly `aligned` (>= num_items)
    // and outlive the launch. Every device access is host-proven in range: `tid`
    // and `cmp = tid + half` both fall in `[0, aligned)` (the even-half-segment
    // guard keeps `tid` in the lower half so `cmp < aligned`), and every
    // `indices[..]` value is in `[0, aligned)` (identity start, only swapped). The
    // keys array is READ-ONLY in the kernel (`&Array<f32>`). cubecl unsafe is
    // confined here (CMP-01 / T-14-03-01).
    unsafe {
        bitonic_argsort_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_keys.clone(), al),
            ArrayArg::from_raw_parts(h_idx.clone(), al),
            aligned,
            depth,
            u32::from(ascending),
        );
    }

    let idx_bytes = client.read_one_unchecked(h_idx);
    let perm_full = i32::from_bytes(&idx_bytes);
    let keys_bytes = client.read_one_unchecked(h_keys);
    let keys_after = f32::from_bytes(&keys_bytes);

    // Truncate the sentinel padding (it sorted to the tail for the real elements).
    Ok((
        perm_full[..num_items].to_vec(),
        keys_after[..num_items].to_vec(),
    ))
}

// =========================================================================
// GPU plane-intrinsic variants (rocm-gated) — built on the cubecl-0.10 plane
// intrinsics (RESEARCH Pattern 1/2), the GPU leg of D-01. These are the
// `has_plane` device kernels the f32 hip mirror dispatches; the cpu anchor above
// is the serial fold (14-01: cubecl-cpu has NO plane support). They are
// CROSS-VALIDATED against the C++/HIP fixtures in 14-06 (this plan ships them;
// the fixture parity gate lands next), held to ~1e-6 vs the f64 anchor — NEVER
// asserted GPU-vs-GPU (def-f8u-01, D-10).
//
// The within-plane level maps directly onto `plane_inclusive_sum` /
// `plane_exclusive_sum` / `plane_sum` / `plane_max` / `plane_min`; the block-wide
// (cross-plane) level stages per-plane partials in a <=32-slot `SharedMemory`
// buffer combined under `sync_cube()` (intra-cube barrier ONLY — no cross-cube
// barrier exists, RESEARCH Pattern 2/3). Single-block (`n <= CubeDim`, <= 1024);
// the multi-block global composition is wired by the first on-device consumer
// (Phase 15) reusing the serial 3-launch structure above.

/// Max planes per cube (1024 / 32 wave; AMD wave64 uses 16, 32 is the safe cap).
/// `usize` to match `SharedMemory::new` (the comptime LDS size type, mirroring
/// `histogram::HIST_LDS_MAX`).
#[cfg(feature = "rocm")]
const N_PLANES_MAX: usize = 32;

/// Block-wide inclusive/exclusive prefix-sum via plane scan + LDS staging
/// (RESEARCH Pattern 2). One cube; `exclusive = inclusive - own_value`.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn plane_block_scan_kernel_f32(
    data: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    plane_dim: u32,
    inclusive: u32,
) {
    let mut stage = SharedMemory::<f32>::new(N_PLANES_MAX);
    let nn = n as usize;
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let v = if i < nn { data[i] } else { f32::new(0.0) };
    // 1. inclusive scan WITHIN each plane.
    let local = plane_inclusive_sum(v);
    // 2. last lane of each plane stages its plane total.
    if lane == pd - 1 {
        stage[plane_id] = local;
    }
    sync_cube();
    // 3. plane 0 exclusive-scans the per-plane totals (n_planes <= plane width).
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { f32::new(0.0) };
        stage[lane] = plane_exclusive_sum(t);
    }
    sync_cube();
    // 4. add the plane base back; derive exclusive from inclusive - v.
    if i < nn {
        let base = stage[plane_id];
        let result = if inclusive == 1 {
            base + local
        } else {
            base + local - v
        };
        out[i] = result;
    }
}

/// Block-wide SUM via `plane_sum` + LDS combine (RESEARCH Pattern 1).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn plane_reduce_sum_kernel_f32(data: &Array<f32>, out: &mut Array<f32>, n: u32, plane_dim: u32) {
    let mut stage = SharedMemory::<f32>::new(N_PLANES_MAX);
    let nn = n as usize;
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let v = if i < nn { data[i] } else { f32::new(0.0) };
    let psum = plane_sum(v);
    if lane == 0 {
        stage[plane_id] = psum;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { f32::new(0.0) };
        let total = plane_sum(t);
        if lane == 0 {
            out[0] = total;
        }
    }
}

/// Block-wide MAX via `plane_max` + LDS combine; `ident` = the padding identity
/// (`-inf`) supplied by the host so out-of-range lanes never win.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn plane_reduce_max_kernel_f32(
    data: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    plane_dim: u32,
    ident: f32,
) {
    let mut stage = SharedMemory::<f32>::new(N_PLANES_MAX);
    let nn = n as usize;
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let v = if i < nn { data[i] } else { ident };
    let pmax = plane_max(v);
    if lane == 0 {
        stage[plane_id] = pmax;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { ident };
        let total = plane_max(t);
        if lane == 0 {
            out[0] = total;
        }
    }
}

/// Block-wide MIN via `plane_min` + LDS combine; `ident` = `+inf`.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn plane_reduce_min_kernel_f32(
    data: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    plane_dim: u32,
    ident: f32,
) {
    let mut stage = SharedMemory::<f32>::new(N_PLANES_MAX);
    let nn = n as usize;
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let v = if i < nn { data[i] } else { ident };
    let pmin = plane_min(v);
    if lane == 0 {
        stage[plane_id] = pmin;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { ident };
        let total = plane_min(t);
        if lane == 0 {
            out[0] = total;
        }
    }
}

/// Block-wide DOT-PRODUCT: elementwise multiply then `plane_sum` + LDS combine
/// (RESEARCH Pattern 1, C++ `ShuffleReduceDotProd`).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn plane_dot_kernel_f32(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    plane_dim: u32,
) {
    let mut stage = SharedMemory::<f32>::new(N_PLANES_MAX);
    let nn = n as usize;
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    let v = if i < nn { a[i] * b[i] } else { f32::new(0.0) };
    let psum = plane_sum(v);
    if lane == 0 {
        stage[plane_id] = psum;
    }
    sync_cube();
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { f32::new(0.0) };
        let total = plane_sum(t);
        if lane == 0 {
            out[0] = total;
        }
    }
}

/// Round a block size up to a multiple of `plane_dim`, capped at 1024 (the
/// single-block plane-scan cap; multi-block global is Phase-15).
#[cfg(feature = "rocm")]
fn plane_block_dim(n: usize, plane_dim: u32) -> Result<u32, ComputeError> {
    if n > 1024 {
        return Err(ComputeError::Runtime {
            detail: format!(
                "plane primitive: n {n} > 1024 single-block cap — multi-block GPU global \
                 composition is wired by the Phase-15 consumer"
            ),
        });
    }
    let pd = plane_dim.max(1);
    Ok((n as u32).div_ceil(pd) * pd)
}

/// GPU (`has_plane`) block-wide inclusive prefix-sum on f32 cells, built on the
/// plane intrinsics. Cross-validated vs the C++ fixture in 14-06 (~1e-6).
///
/// # Errors
/// [`ComputeError::Runtime`] if `n > 1024` (single-block cap).
#[cfg(feature = "rocm")]
pub fn plane_prefix_sum_inclusive_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    plane_dim: u32,
) -> Result<Vec<f32>, ComputeError> {
    plane_prefix_sum_f32_on(client, data, plane_dim, true)
}

/// GPU block-wide exclusive prefix-sum. See [`plane_prefix_sum_inclusive_f32_on`].
///
/// # Errors
/// [`ComputeError::Runtime`] if `n > 1024`.
#[cfg(feature = "rocm")]
pub fn plane_prefix_sum_exclusive_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    plane_dim: u32,
) -> Result<Vec<f32>, ComputeError> {
    plane_prefix_sum_f32_on(client, data, plane_dim, false)
}

#[cfg(feature = "rocm")]
fn plane_prefix_sum_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    plane_dim: u32,
    inclusive: bool,
) -> Result<Vec<f32>, ComputeError> {
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let block_dim = plane_block_dim(n, plane_dim)?;
    let h_in = client.create_from_slice(f32::as_bytes(data));
    let h_out = client.empty(n * core::mem::size_of::<f32>());
    // SAFETY: `h_in`/`h_out` sized exactly `n`; the kernel guards every access by
    // `i < n` and stages only `<= N_PLANES_MAX` per-plane partials. cubecl unsafe
    // confined here (CMP-01).
    unsafe {
        plane_block_scan_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(block_dim),
            ArrayArg::from_raw_parts(h_in, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            n as u32,
            plane_dim.max(1),
            u32::from(inclusive),
        );
    }
    Ok(f32::from_bytes(&client.read_one_unchecked(h_out)).to_vec())
}

/// GPU (`has_plane`) block-wide f32 reductions, built on `plane_sum`/`plane_max`/
/// `plane_min`. Cross-validated vs the C++ fixture in 14-06 (~1e-6).
///
/// # Errors
/// [`ComputeError::Runtime`] if `n > 1024`; [`ComputeError::LengthMismatch`] on
/// empty max/min or mismatched dot lengths.
#[cfg(feature = "rocm")]
pub fn plane_reduce_sum_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    plane_dim: u32,
) -> Result<f32, ComputeError> {
    if data.is_empty() {
        return Ok(0.0);
    }
    let block_dim = plane_block_dim(data.len(), plane_dim)?;
    let h_in = client.create_from_slice(f32::as_bytes(data));
    let h_out = client.empty(core::mem::size_of::<f32>());
    // SAFETY: see `plane_prefix_sum_f32_on`; single scalar out.
    unsafe {
        plane_reduce_sum_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(block_dim),
            ArrayArg::from_raw_parts(h_in, data.len()),
            ArrayArg::from_raw_parts(h_out.clone(), 1),
            data.len() as u32,
            plane_dim.max(1),
        );
    }
    Ok(f32::from_bytes(&client.read_one_unchecked(h_out))[0])
}

/// GPU block-wide MAX. See [`plane_reduce_sum_f32_on`].
///
/// # Errors
/// As [`plane_reduce_sum_f32_on`]; empty -> [`ComputeError::LengthMismatch`].
#[cfg(feature = "rocm")]
pub fn plane_reduce_max_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    plane_dim: u32,
) -> Result<f32, ComputeError> {
    if data.is_empty() {
        return Err(ComputeError::LengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let block_dim = plane_block_dim(data.len(), plane_dim)?;
    let h_in = client.create_from_slice(f32::as_bytes(data));
    let h_out = client.empty(core::mem::size_of::<f32>());
    // SAFETY: see `plane_prefix_sum_f32_on`; single scalar out.
    unsafe {
        plane_reduce_max_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(block_dim),
            ArrayArg::from_raw_parts(h_in, data.len()),
            ArrayArg::from_raw_parts(h_out.clone(), 1),
            data.len() as u32,
            plane_dim.max(1),
            f32::NEG_INFINITY,
        );
    }
    Ok(f32::from_bytes(&client.read_one_unchecked(h_out))[0])
}

/// GPU block-wide MIN. See [`plane_reduce_sum_f32_on`].
///
/// # Errors
/// As [`plane_reduce_sum_f32_on`]; empty -> [`ComputeError::LengthMismatch`].
#[cfg(feature = "rocm")]
pub fn plane_reduce_min_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    data: &[f32],
    plane_dim: u32,
) -> Result<f32, ComputeError> {
    if data.is_empty() {
        return Err(ComputeError::LengthMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let block_dim = plane_block_dim(data.len(), plane_dim)?;
    let h_in = client.create_from_slice(f32::as_bytes(data));
    let h_out = client.empty(core::mem::size_of::<f32>());
    // SAFETY: see `plane_prefix_sum_f32_on`; single scalar out.
    unsafe {
        plane_reduce_min_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(block_dim),
            ArrayArg::from_raw_parts(h_in, data.len()),
            ArrayArg::from_raw_parts(h_out.clone(), 1),
            data.len() as u32,
            plane_dim.max(1),
            f32::INFINITY,
        );
    }
    Ok(f32::from_bytes(&client.read_one_unchecked(h_out))[0])
}

/// GPU block-wide DOT-PRODUCT. See [`plane_reduce_sum_f32_on`].
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `a.len() != b.len()`; `Runtime` if n>1024.
#[cfg(feature = "rocm")]
pub fn plane_dot_product_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    a: &[f32],
    b: &[f32],
    plane_dim: u32,
) -> Result<f32, ComputeError> {
    if a.len() != b.len() {
        return Err(ComputeError::LengthMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    let block_dim = plane_block_dim(a.len(), plane_dim)?;
    let h_a = client.create_from_slice(f32::as_bytes(a));
    let h_b = client.create_from_slice(f32::as_bytes(b));
    let h_out = client.empty(core::mem::size_of::<f32>());
    // SAFETY: `h_a`/`h_b` sized `a.len()`; single scalar out. cubecl unsafe
    // confined here (CMP-01).
    unsafe {
        plane_dot_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(block_dim),
            ArrayArg::from_raw_parts(h_a, a.len()),
            ArrayArg::from_raw_parts(h_b, a.len()),
            ArrayArg::from_raw_parts(h_out.clone(), 1),
            a.len() as u32,
            plane_dim.max(1),
        );
    }
    Ok(f32::from_bytes(&client.read_one_unchecked(h_out))[0])
}
