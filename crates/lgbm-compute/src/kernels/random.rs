//! `CUDARandom` device LCG (`#[cube]`) + host parity launchers — **14-04**.
//!
//! Owning plan: **14-04**. Scope locked by **ODL-02**, **D-04**.
//!
//! ## What lives here
//! A `#[cube]` LCG whose draw stream is **bit-identical** to the host
//! `lgbm_core::random::Random` (the AMD-fork `include/LightGBM/cuda/cuda_random.hpp`
//! `CUDARandom` device-side analog), plus host launchers
//! ([`draw_rand_int16_on`] / [`draw_rand_int32_on`] / [`draw_next_float_on`]) that
//! seed N device tasks, draw K each, and read the streams back for the D-04 parity
//! assertion (the host `Random` IS the oracle — NOT a new C++ capture, and NEVER
//! GPU-vs-GPU, D-10).
//!
//! ## Recurrence (`crates/lgbm-core/src/random.rs:44-77`, mirrored on device)
//! `x = 214013 * x + 2531011` advanced with **plain `u32` `*`/`+`** — the native
//! hardware two's-complement wrap. NOT `wrapping_add`/`wrapping_mul` (those are host
//! methods, not `#[cube]` intrinsics — RESEARCH Pitfall 2 / Assumption A2) and NOT
//! any `Atomic<i64>` (broken on cubecl-hip 0.10). The `cuda_random_parity` test IS
//! the end-to-end proof that the plain-`u32` device arithmetic wraps identically to
//! the host's `wrapping_*` recurrence (A2).
//!
//! - `RandInt16` = `(advance >> 16) & 0x7FFF` as `i32`
//! - `RandInt32` = `advance & 0x7FFFFFFF` as `i32`
//! - `NextFloat` = `RandInt16 as f32 / 32768.0f32` (divisor EXACTLY `32768.0`, f32
//!   end-to-end — NOT `65536`, NOT computed in f64)
//!
//! ## Determinism (RESEARCH Pitfall 1, mirroring `histogram.rs`)
//! The draw kernels are launched single-owner (`CubeDim::new_1d(1)`): one unit walks
//! every task in ascending order, each task's output window is disjoint, so there is
//! no cross-unit race and the stream is bit-stable on both the cpu anchor and hip.
//! Per-task state is independent (no shared accumulation), so the result is
//! identical to a fully task-parallel launch — single-owner is the safe, portable
//! form here.
//!
//! ## Security V6 negative control (carried forward from `random.rs:8-14`)
//! `CUDARandom`/`Random` are **deterministic, NON-cryptographic** reproducibility
//! PRNGs (the classic MSVC `rand` recurrence). They exist ONLY to reproduce the C++
//! reference bit-for-bit and **MUST NEVER be used as a source of secure
//! randomness** (Security V6 / threat T-14-04-02). The device port inherits this
//! constraint verbatim.
//!
//! ## Analog files
//! `crates/lgbm-core/src/random.rs` (the host `Random` LCG — the parity oracle) +
//! `crates/lgbm-compute/src/kernels/histogram.rs` (the `create_from_slice` ->
//! `launch_unchecked` -> `read_one_unchecked` launcher shape + the single-owner
//! determinism mandate).

use cubecl::prelude::*;

use crate::error::ComputeError;

/// Advance the LCG state: `x = 214013 * x + 2531011` with plain `u32` arithmetic
/// (native two's-complement wrap — NOT `wrapping_*`, RESEARCH Pitfall 2/A2).
/// Returns the advanced state.
#[cube]
fn cuda_rand_advance(state: u32) -> u32 {
    state * 214013u32 + 2531011u32
}

/// `RandInt16`: advance and return `(x >> 16) & 0x7FFF` as `i32` (15-bit
/// non-negative). Mirrors host `Random::rand_int16`.
#[cube]
fn cuda_rand_int16(state: u32) -> i32 {
    ((cuda_rand_advance(state) >> 16) & 0x7FFFu32) as i32
}

/// `RandInt32`: advance and return `x & 0x7FFFFFFF` as `i32` (31-bit
/// non-negative). Mirrors host `Random::rand_int32`.
#[cube]
fn cuda_rand_int32(state: u32) -> i32 {
    (cuda_rand_advance(state) & 0x7FFF_FFFFu32) as i32
}

/// `NextFloat`: `RandInt16(state) as f32 / 32768.0f32` (f32 end-to-end, divisor
/// exactly `32768.0`). Mirrors host `Random::next_float`.
#[cube]
fn cuda_next_float(state: u32) -> f32 {
    (cuda_rand_int16(state) as f32) / 32768.0f32
}

// --- per-draw-type single-owner draw kernels ---------------------------------
//
// Each kernel threads `state = cuda_rand_advance(state)` once per draw (so the
// state evolves exactly as the host's sequential `x = ...` recurrence) and writes
// the extracted draw into the disjoint per-task output window `[t*k, t*k + k)`.

#[cube(launch_unchecked)]
fn draw_int16_kernel(seeds: &Array<u32>, out: &mut Array<i32>, n: u32, k: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let kk = k as usize;
        let mut t = 0usize;
        while t < nn {
            let mut state = seeds[t];
            let mut j = 0usize;
            while j < kk {
                out[t * kk + j] = cuda_rand_int16(state);
                state = cuda_rand_advance(state);
                j += 1;
            }
            t += 1;
        }
    }
}

#[cube(launch_unchecked)]
fn draw_int32_kernel(seeds: &Array<u32>, out: &mut Array<i32>, n: u32, k: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let kk = k as usize;
        let mut t = 0usize;
        while t < nn {
            let mut state = seeds[t];
            let mut j = 0usize;
            while j < kk {
                out[t * kk + j] = cuda_rand_int32(state);
                state = cuda_rand_advance(state);
                j += 1;
            }
            t += 1;
        }
    }
}

#[cube(launch_unchecked)]
fn draw_float_kernel(seeds: &Array<u32>, out: &mut Array<f32>, n: u32, k: u32) {
    if UNIT_POS == 0 {
        let nn = n as usize;
        let kk = k as usize;
        let mut t = 0usize;
        while t < nn {
            let mut state = seeds[t];
            let mut j = 0usize;
            while j < kk {
                out[t * kk + j] = cuda_next_float(state);
                state = cuda_rand_advance(state);
                j += 1;
            }
            t += 1;
        }
    }
}

/// Validate the draw launcher inputs at the V5 boundary (threat T-14-04-03: no
/// overflow in `n * k` output sizing) and return the output length.
fn validate_draw_inputs(num_seeds: usize, k: u32) -> Result<usize, ComputeError> {
    num_seeds
        .checked_mul(k as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!(
                "CUDARandom draw: output length {num_seeds} * {k} overflows usize"
            ),
        })
}

/// Draw `k` consecutive `RandInt16` values from each seeded device task, returned
/// row-major as `[task0_draw0 .. task0_draw{k-1}, task1_draw0 .. ]`
/// (length `seeds.len() * k`). `seeds[i]` is the initial LCG state (the host
/// `Random::new(seed)` sets `x = seed as u32`, so pass `seed as u32`).
///
/// Bit-identical to the host `Random` `RandInt16` stream (D-04).
///
/// # Errors
/// [`ComputeError::Runtime`] if `seeds.len() * k` overflows `usize`.
pub fn draw_rand_int16_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    seeds: &[u32],
    k: u32,
) -> Result<Vec<i32>, ComputeError> {
    let out_len = validate_draw_inputs(seeds.len(), k)?;
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let n = seeds.len();
    let h_seeds = client.create_from_slice(u32::as_bytes(seeds));
    // The kernel WRITES every out cell (disjoint per-task windows cover [0, out_len)),
    // so `empty` is safe (no zero-init needed) — mirrors the `split.rs` idiom.
    let h_out = client.empty(out_len * core::mem::size_of::<i32>());

    // SAFETY: `h_seeds` is sized exactly `n` u32 cells and `h_out` `out_len` i32
    // cells, each outliving the launch. The single owner walks `t in [0, n)` and
    // `j in [0, k)`, writing only `out[t*k + j] < out_len` (host-proven by
    // `validate_draw_inputs`). cubecl unsafe confined here (CMP-01, T-14-04-01).
    unsafe {
        draw_int16_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_seeds, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            n as u32,
            k,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(i32::from_bytes(&bytes).to_vec())
}

/// Draw `k` consecutive `RandInt32` values from each seeded device task (same
/// layout as [`draw_rand_int16_on`]). Bit-identical to the host `Random`
/// `RandInt32` stream (D-04).
///
/// # Errors
/// [`ComputeError::Runtime`] if `seeds.len() * k` overflows `usize`.
pub fn draw_rand_int32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    seeds: &[u32],
    k: u32,
) -> Result<Vec<i32>, ComputeError> {
    let out_len = validate_draw_inputs(seeds.len(), k)?;
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let n = seeds.len();
    let h_seeds = client.create_from_slice(u32::as_bytes(seeds));
    let h_out = client.empty(out_len * core::mem::size_of::<i32>());

    // SAFETY: identical sizing/index argument as `draw_rand_int16_on` (T-14-04-01).
    unsafe {
        draw_int32_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_seeds, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            n as u32,
            k,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(i32::from_bytes(&bytes).to_vec())
}

/// Draw `k` consecutive `NextFloat` values from each seeded device task (same
/// layout as [`draw_rand_int16_on`]). Bit-identical to the host `Random`
/// `NextFloat` stream — `f32::to_bits()`-exact (D-04).
///
/// # Errors
/// [`ComputeError::Runtime`] if `seeds.len() * k` overflows `usize`.
pub fn draw_next_float_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    seeds: &[u32],
    k: u32,
) -> Result<Vec<f32>, ComputeError> {
    let out_len = validate_draw_inputs(seeds.len(), k)?;
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let n = seeds.len();
    let h_seeds = client.create_from_slice(u32::as_bytes(seeds));
    let h_out = client.empty(out_len * core::mem::size_of::<f32>());

    // SAFETY: identical sizing/index argument as `draw_rand_int16_on`, `h_out`
    // sized `out_len` f32 cells (T-14-04-01).
    unsafe {
        draw_float_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_seeds, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
            n as u32,
            k,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).to_vec())
}
