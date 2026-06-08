//! Minimal `construct_histograms` cube kernel — the D-04a determinism anchor.
//!
//! Transcribes the C++ accumulation body verbatim from
//! `LightGBM/src/io/dense_bin.hpp:99-141` (`ConstructHistogramInner`,
//! `USE_HESSIAN` path):
//!
//! ```cpp
//! hist_t* grad = out;        // hist_t = double
//! hist_t* hess = out + 1;
//! const auto ti = static_cast<uint32_t>(data(idx)) << 1;   // bin<<1
//! grad[ti] += ordered_gradients[i];   // f32 read, f64 accumulate
//! hess[ti] += ordered_hessians[i];
//! ```
//!
//! i.e. the histogram is laid out stride-2 interleaved `[g0,h0,g1,h1,…]` with
//! the grad cell at `bin*2` and the hess cell at `bin*2 + 1`. Gradients and
//! hessians are read as f32 (`score_t = float`) but summed into f64 cells
//! (`hist_t = double`, RESEARCH Pitfall 3).
//!
//! **Determinism mandate (RESEARCH Pitfall 1, D-04/D-04a):** cubecl-cpu spawns
//! one OS worker thread per cube unit — it is NOT a single-threaded sequential
//! executor. To make the f64 fold bit-stable (matching the C++ `num_threads=1`
//! ordered fold), the kernel is launched with `CubeDim::new_1d(1)` so exactly
//! ONE unit owns the entire fold, in row order. Any multi-unit accumulation
//! into shared cells (atomics) would be order-nondeterministic — and atomics
//! aren't supported on cubecl-cpu anyway.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// The single-owner ordered f64 fold (RESEARCH Pattern 1 + `dense_bin.hpp`).
///
/// Only `UNIT_POS == 0` executes the fold, in ascending row order, so the f64
/// summation order is fixed and matches the C++ `num_threads=1` reference.
///
/// This is the **cpu anchor** path: gradients/hessians are read as f32 but
/// summed into f64 cells (`hist_t = double`) — bit-exact vs C++ (Pitfall 3).
#[cube(launch)]
pub fn construct_hist_kernel(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f64>,
) {
    // Single-owner ordered fold — the deterministic anchor (Pitfall 1).
    if UNIT_POS == 0 {
        for i in 0..binned.len() {
            // ti = bin<<1; grad cell at ti, hess cell at ti+1 (dense_bin.hpp:120).
            // `binned[i]` is u32; widen to usize for indexing the f64 `out` array.
            let ti = binned[i] as usize * 2;
            out[ti] += f64::cast_from(grad[i]); // f32 read, f64 accumulate
            out[ti + 1] += f64::cast_from(hess[i]);
        }
    }
}

/// The f32-cell mirror of [`construct_hist_kernel`] for the no-f64 hip device
/// (RESEARCH Pitfall 2/3, CMP-04). IDENTICAL fold structure and row order — the
/// ONLY difference is the accumulation cell type (`f32` instead of `f64`). hip
/// (gfx1100) cannot allocate f64, so the histogram accumulates in f32, accepting
/// the ~1e-6-tolerated divergence from the cpu f64 anchor (the divergence the
/// oracle contract was designed to absorb, NOT a bug). The capability gate
/// (`has_f64 == false`) routes the hip launch here; cpu keeps the f64 kernel.
#[cube(launch)]
pub fn construct_hist_kernel_f32(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f32>,
) {
    // SAME single-owner ordered fold, SAME row order — f32 cells only (Pitfall 3).
    if UNIT_POS == 0 {
        for i in 0..binned.len() {
            let ti = binned[i] as usize * 2;
            out[ti] += grad[i]; // f32 read, f32 accumulate (no f64 on hip)
            out[ti + 1] += hess[i];
        }
    }
}

/// Host-side `construct_histograms` on the cpu reference runtime.
///
/// Validates every kernel input at the `Backend` boundary (Security V5, threat
/// T-04-01) BEFORE the `unsafe` launch, then runs the single-owner ordered fold
/// and returns the f64 histogram `[g0,h0,g1,h1,…]` of length `2 * num_bin`.
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `grad`/`hess`/`binned` lengths differ.
/// - [`ComputeError::BinIndexOutOfRange`] if any `binned[i] >= num_bin`.
pub fn construct_histograms_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    construct_histograms_f64_on(client, binned, grad, hess, num_bin)
}

/// The f64 `construct_histograms` cube path, **generic over the runtime** `R` so it
/// runs on the cubecl-cpu anchor (via [`construct_histograms_cpu`]) AND on
/// cubecl-hip (the GPU `RocmBackend`). The gfx1100 executes this f64 kernel
/// bit-exactly to the CPU anchor (verified: `max_abs_diff=0`), even though
/// `probe_capabilities().has_f64` is reported `false` — the flag is conservative,
/// the f64 op is real. Same single-owner ordered fold (`CubeDim::new_1d(1)`) and V5
/// validation as before.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
pub fn construct_histograms_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    // --- V5 boundary validation (T-04-01): never panic / UB on caller input ---
    // Shared with the f32 hip path (`construct_histograms_f32_on`); `out` is sized
    // 2 * num_bin cells, the `bin<<1` index math is overflow-guarded, and every
    // bin is range-checked.
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;

    let n = binned.len();
    let h_bin = client.create_from_slice(u32::as_bytes(binned));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    // The kernel ACCUMULATES into `out` (`out[ti] += ...`), so `out` must start
    // zeroed. `client.empty` returns UNINITIALIZED device memory from the pool —
    // it may recycle a prior launch's buffer, so a fresh launch would fold on top
    // of stale values. Allocate from an explicit zero slice to match the C++
    // histogram being zeroed before accumulation.
    let zeros = vec![0.0f64; out_len];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

    // SAFETY: `ArrayArg::from_raw_parts(handle, len)` requires that each handle
    // was allocated for exactly `len` elements of the declared element type and
    // outlives the launch. We just allocated `h_bin`/`h_grad`/`h_hess` from
    // slices of length `n` and `h_out` for `out_len` f64 cells, and the input
    // validation above guarantees every `binned[i] < num_bin` so the kernel's
    // `out[bin*2 + 1]` write stays within the `out_len` allocation (T-04-01/02).
    // All cubecl `unsafe` is confined to this crate (CMP-01).
    unsafe {
        construct_hist_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1), // single unit owns the entire ordered fold (Pitfall 1)
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// **Native** host f64 fold — the production cpu-anchor path (R2).
///
/// Bit-IDENTICAL to [`construct_histograms_cpu`] (the single-unit
/// `construct_hist_kernel`): the exact same ascending-row-order accumulation of
/// `f32`-read gradients/hessians into `f64` cells, with the same `bin << 1` index
/// math and the same V5 boundary validation. The cubecl-cpu kernel launches that
/// loop as a `CubeDim::new_1d(1)` single owner — a fixed ~20–50µs dispatch cost
/// per call wrapping a trivial sequential loop. This native version drops that
/// overhead (5–210× faster per call; `probe_hist` measured bit_exact=true at
/// R=300/2000/20000) while producing byte-identical output, because the
/// arithmetic and order are the same.
///
/// `construct_histograms_cpu` is retained for the kernel-parity / ROCm-mirror
/// tests; the f32 hip path ([`construct_histograms_f32_on`]) is untouched.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
pub fn construct_histograms_cpu_native(
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;
    let mut out = vec![0.0f64; out_len];
    // Ascending row order, f32 read → f64 accumulate, grad at bin<<1 / hess at +1 —
    // the verbatim `construct_hist_kernel` body (dense_bin.hpp:99-141). The
    // validation above guarantees every `binned[i] < num_bin`, so `ti + 1` stays in
    // bounds; the loop uses checked indexing regardless (no `unsafe`).
    for (i, &bin) in binned.iter().enumerate() {
        let ti = bin as usize * 2;
        out[ti] += f64::from(grad[i]);
        out[ti + 1] += f64::from(hess[i]);
    }
    Ok(out)
}

/// Validate the `construct_histograms` inputs (shared by the f64 cpu path and
/// the f32 hip path). Returns the histogram length `2 * num_bin` on success.
///
/// # Errors
/// - [`ComputeError::LengthMismatch`] if `grad`/`hess`/`binned` lengths differ.
/// - [`ComputeError::BinIndexOutOfRange`] if any `binned[i] >= num_bin`.
fn validate_histogram_inputs(
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<usize, ComputeError> {
    if grad.len() != binned.len() {
        return Err(ComputeError::LengthMismatch {
            expected: binned.len(),
            actual: grad.len(),
        });
    }
    if hess.len() != binned.len() {
        return Err(ComputeError::LengthMismatch {
            expected: binned.len(),
            actual: hess.len(),
        });
    }
    let out_len = 2usize
        .checked_mul(num_bin as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("num_bin {num_bin} overflows the histogram allocation size"),
        })?;
    for (row, &bin) in binned.iter().enumerate() {
        if bin >= num_bin {
            return Err(ComputeError::BinIndexOutOfRange { row, bin, num_bin });
        }
    }
    Ok(out_len)
}

/// Host-side `construct_histograms` in **f32 cells** on ANY runtime (the no-f64
/// hip path; CMP-03/CMP-04). Same V5 boundary validation and same single-owner
/// ordered fold as [`construct_histograms_cpu`], but accumulates into f32 cells
/// (hip cannot allocate f64). Returns the `2 * num_bin` f32 histogram. The hip
/// parity gate compares this against the cpu f64 anchor collected to `Vec<f32>`
/// within `ORACLE_TOL = 1e-6` (RESEARCH Pitfall 3, D-03a).
///
/// Generic over `R: Runtime` so it runs on the cubecl-cpu client (to produce the
/// f32 cpu reference in tests) AND on the cubecl-hip client (the real GPU path).
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation, V5).
pub fn construct_histograms_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f32>, ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;

    let n = binned.len();
    let h_bin = client.create_from_slice(u32::as_bytes(binned));
    let h_grad = client.create_from_slice(f32::as_bytes(grad));
    let h_hess = client.create_from_slice(f32::as_bytes(hess));
    let zeros = vec![0.0f32; out_len];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    // SAFETY: identical handle/length correspondence to `construct_histograms_cpu`
    // — `h_bin`/`h_grad`/`h_hess` sized `n`, `h_out` sized `out_len` f32 cells,
    // and the validation above keeps every `out[bin*2 + 1]` write in range. All
    // cubecl `unsafe` is confined to this crate (CMP-01).
    unsafe {
        construct_hist_kernel_f32::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).to_vec())
}
