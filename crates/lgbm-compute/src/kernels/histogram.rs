//! Minimal `construct_histograms` cube kernel — the determinism anchor.
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
//! (`hist_t = double`).
//!
//! **Determinism mandate:** cubecl-cpu spawns
//! one OS worker thread per cube unit — it is NOT a single-threaded sequential
//! executor. To make the f64 fold bit-stable (matching the C++ `num_threads=1`
//! ordered fold), the kernel is launched with `CubeDim::new_1d(1)` so exactly
//! ONE unit owns the entire fold, in row order. Any multi-unit accumulation
//! into shared cells (atomics) would be order-nondeterministic — and atomics
//! aren't supported on cubecl-cpu anyway.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// The single shared `#[cube]` single-owner ordered histogram fold — the SINGLE
/// SOURCE OF TRUTH for the deterministic fold math; the structural analog of
/// [`crate::kernels::split::split_scan_body`].
///
/// Generic over the accumulation cell type `N: Numeric`: both the f64 cpu-anchor
/// launch kernel ([`construct_hist_kernel`], `N = f64`) AND the f32 hip-mirror
/// launch kernel ([`construct_hist_kernel_f32`], `N = f32`) call this helper, so
/// the single-owner ordered fold, the `UNIT_POS == 0` ownership, the ascending
/// row order, and the `bin<<1` stride-2 cell layout exist exactly ONCE. The ONLY
/// difference between the two launch entry points is the `N` it is instantiated
/// with — eliminating a hand-duplicated fold loop (a drift hazard).
///
/// Only `UNIT_POS == 0` executes the fold, in ascending row order, so the
/// summation order is fixed and matches the C++ `num_threads=1` reference.
///
/// Gradients/hessians are always READ as f32 (`score_t = float`); the cast
/// `N::cast_from(grad[i])` is the accumulation widening:
/// - For `N = f64` it is the f32→f64 widening — byte-identical to
///   `f64::cast_from(grad[i])` (the bit-exact cpu anchor).
/// - For `N = f32` it is the identity cast — observably identical to
///   `out[ti] += grad[i]` (the ~1e-6-tolerated hip mirror).
#[cube]
fn hist_fold_body<N: Numeric>(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<N>,
) {
    // Single-owner ordered fold — the deterministic anchor.
    if UNIT_POS == 0 {
        for i in 0..binned.len() {
            // ti = bin<<1; grad cell at ti, hess cell at ti+1 (dense_bin.hpp:120).
            // `binned[i]` is u32; widen to usize for indexing the `out` array.
            let ti = binned[i] as usize * 2;
            out[ti] += N::cast_from(grad[i]); // f32 read, N-cell accumulate
            out[ti + 1] += N::cast_from(hess[i]);
        }
    }
}

/// The single-owner ordered f64 fold (mirrors `dense_bin.hpp`) — a
/// THIN `#[cube(launch)]` wrapper that delegates to the shared generic
/// [`hist_fold_body`] with `N = f64`. This kernel holds NO fold logic of its
/// own; the math lives once in `hist_fold_body`, shared with the f32 hip
/// mirror.
///
/// This is the **cpu anchor** path: gradients/hessians are read as f32 but
/// summed into f64 cells (`hist_t = double`) — bit-exact vs C++.
/// `f64::cast_from(grad[i])` is exactly what `N::cast_from` lowers to for
/// `N = f64`.
#[cube(launch)]
pub fn construct_hist_kernel(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f64>,
) {
    hist_fold_body::<f64>(binned, grad, hess, out);
}

/// The f32-cell mirror of [`construct_hist_kernel`] for the no-f64 hip device —
/// a THIN `#[cube(launch)]` wrapper that delegates to the shared generic
/// [`hist_fold_body`] with `N = f32`. IDENTICAL fold structure and row order to
/// the f64 kernel (they share the helper) — the ONLY difference is the
/// accumulation cell type (`f32` instead of `f64`). hip (gfx1100) cannot
/// allocate f64, so the histogram accumulates in f32, accepting the
/// ~1e-6-tolerated divergence from the cpu f64 anchor (the divergence the
/// oracle contract was designed to absorb, NOT a bug). The capability gate
/// (`has_f64 == false`) routes the hip launch here; cpu keeps the f64 kernel.
/// For `N = f32`, `N::cast_from(grad[i])` is the identity cast — observably
/// identical to `out[ti] += grad[i]`.
#[cube(launch)]
pub fn construct_hist_kernel_f32(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<f32>,
) {
    hist_fold_body::<f32>(binned, grad, hess, out);
}

/// Host-side `construct_histograms` on the cpu reference runtime.
///
/// Validates every kernel input at the `Backend` boundary BEFORE the `unsafe`
/// launch, then runs the single-owner ordered fold
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
/// the f64 op is real. Same single-owner ordered fold (`CubeDim::new_1d(1)`) and
/// input validation as before.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation).
pub fn construct_histograms_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
) -> Result<Vec<f64>, ComputeError> {
    // --- Boundary validation: never panic / UB on caller input ---
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
    // `out[bin*2 + 1]` write stays within the `out_len` allocation.
    // All cubecl `unsafe` is confined to this crate.
    unsafe {
        construct_hist_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1), // single unit owns the entire ordered fold
            ArrayArg::from_raw_parts(h_bin, n),
            ArrayArg::from_raw_parts(h_grad, n),
            ArrayArg::from_raw_parts(h_hess, n),
            ArrayArg::from_raw_parts(h_out.clone(), out_len),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// **Native** host f64 fold — the production cpu-anchor path.
///
/// Bit-IDENTICAL to [`construct_histograms_cpu`] (the single-unit
/// `construct_hist_kernel`): the exact same ascending-row-order accumulation of
/// `f32`-read gradients/hessians into `f64` cells, with the same `bin << 1` index
/// math and the same boundary validation. The cubecl-cpu kernel launches that
/// loop as a `CubeDim::new_1d(1)` single owner — a fixed ~20–50µs dispatch cost
/// per call wrapping a trivial sequential loop. This native version drops that
/// overhead (5–210× faster per call) while producing byte-identical output,
/// because the arithmetic and order are the same.
///
/// `construct_histograms_cpu` is retained for the kernel-parity / ROCm-mirror
/// tests; the f32 hip path (`construct_histograms_f32_on`) is untouched.
///
/// # Errors
/// Same as [`construct_histograms_cpu`] (length / bin-range validation).
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
    //
    // This loop is intentionally INLINE here (not delegated to
    // `accumulate_histogram_into`): the two bodies are kept byte-identical by the
    // `accumulate_into_is_bit_identical_to_native` unit test (f64::to_bits cell-by-
    // cell on multiple shapes) — a STRONGER drift guard than textual sharing, and one
    // that does not perturb this hot allocate-then-fold path's large-row codegen
    // (measured: routing through a shared `&mut [f64]` helper / a second validation
    // pass regressed the 200k-row build ~5%).
    for (i, &bin) in binned.iter().enumerate() {
        let ti = bin as usize * 2;
        out[ti] += f64::from(grad[i]);
        out[ti + 1] += f64::from(hess[i]);
    }
    Ok(out)
}

/// **Fold-in-place** native f64 accumulator — the per-feature build hot path.
///
/// Folds the SAME ascending rows as [`construct_histograms_cpu_native`] directly
/// into a caller-owned, **caller-pre-zeroed** `out` sub-slice — NO intermediate
/// per-feature `Vec`, NO copy. This eliminates the per-feature alloc + memset +
/// memcpy that measurably dominated low-row train time (build = 232µs/iter vs
/// C++ 44.5µs/iter).
///
/// The arithmetic is byte-for-byte identical to `construct_histograms_cpu_native`:
/// the same rows in the same ascending order, the same `bin << 1` cell layout, the
/// same `f32`-read → `f64`-accumulate. The only difference is that this writes into
/// a pre-zeroed sub-slice instead of allocating + zeroing its own buffer. The two
/// loop bodies are kept byte-identical — and the fold ORDER frozen (the project's
/// bit-exact CPU f64 merge gate depends on it) — by the
/// `accumulate_into_is_bit_identical_to_native` unit test (f64::to_bits cell-by-cell
/// on multiple shapes), NOT by textual sharing (a shared helper measurably regressed
/// the large-row native path; see `construct_histograms_cpu_native`).
///
/// `out` MUST be pre-zeroed by the caller over `out[0..2*num_bin]` (the caller owns
/// zeroing so it can zero a larger multi-feature buffer once). This function does
/// NOT zero `out`.
///
/// # Errors
/// - Same length / bin-range validation as [`construct_histograms_cpu_native`]
///   (runs BEFORE any write).
/// - [`ComputeError::LengthMismatch`] if `out.len() < 2 * num_bin` (no panic).
pub fn accumulate_histogram_into(
    binned: &[u32],
    grad: &[f32],
    hess: &[f32],
    num_bin: u32,
    out: &mut [f64],
) -> Result<(), ComputeError> {
    let out_len = validate_histogram_inputs(binned, grad, hess, num_bin)?;
    // The caller's sub-slice must hold the full 2*num_bin histogram. Surface a
    // typed error (no panic) BEFORE writing any cell.
    if out.len() < out_len {
        return Err(ComputeError::LengthMismatch {
            expected: out_len,
            actual: out.len(),
        });
    }
    // Ascending row order, f32 read → f64 accumulate, grad at bin<<1 / hess at +1 —
    // the verbatim `construct_hist_kernel` body (dense_bin.hpp:99-141). The
    // validation above guarantees every `binned[i] < num_bin`, so `ti + 1` stays in
    // bounds; the loop uses checked indexing regardless (no `unsafe`). Folding into a
    // pre-zeroed sub-slice yields bytes identical to the allocating path.
    for (i, &bin) in binned.iter().enumerate() {
        let ti = bin as usize * 2;
        out[ti] += f64::from(grad[i]);
        out[ti + 1] += f64::from(hess[i]);
    }
    Ok(())
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




/// LDS sub-histogram cap: `2 * 256` f32 cells = 2 KiB of shared memory per cube
/// (grad+hess interleaved for up to 256 bins). cubecl `SharedMemory::new` needs a
/// COMPTIME size, but num_bin varies at runtime (32/64/128/256); rather than
/// specialize one kernel binary per bin count (LightGBM's `histogram{16,64,256}.cl`
/// family approach), we allocate the fixed 256-bin max once (2 KiB ≪ the gfx1100
/// 64 KiB LDS budget) and drive the active length with the runtime `lds_len`.
///
/// Gated on the `gpu` umbrella feature rather than `rocm` specifically: the two LDS
/// construct items (`construct_hist_kernel_lds_f32` + `construct_histograms_lds_f32_on`)
/// that cuda/wgpu reuse reference this cap, so it must compile under any GPU backend.
/// It is a plain `usize` const — no `cubecl_hip_sys`/`rocm_client` involvement.
#[cfg(feature = "gpu")]
const HIST_LDS_MAX: usize = 512;

/// Fixed-point quantize scale S = 2^30. The resident LDS BUILD
/// accumulates `round(value * S)` as a two's-complement i64 stored BITS-as-u64 via
/// integer LDS atomics (`Atomic<u64>::fetch_add`), then `fix_compact_kernel` dequantizes
/// `(bits as i64) / S` back to f64 in its widen pass. S = 2^30 keeps ≥ ~9 fractional
/// bits while leaving the i64 magnitude safe to ~1e9 rows × |g| ≤ 8. The
/// build-side constant is f32 (the quantize multiplies the f32 `ord_g`/`ord_h`); the
/// dequant-side `SCALE_F64` (in `fix_compact_kernel`) is the same value in f64.
#[cfg(feature = "gpu")]
const SCALE_F32: f32 = 1_073_741_824.0; // 2^30

/// Row-partition (`grid_dim_y` analog) tuning.
/// The LDS build launches one cube per feature; on a large leaf that is only
/// `num_features` workgroups, starving the GPU (gfx1100 = 96 CUs). Splitting a feature's
/// rows across `P` cubes raises occupancy: `P=16` (~8 workgroups/CU) gives a stable
/// **1.3–1.4×** speedup, while **`P=32` over-partitions and regresses**, so `P` is a tuned
/// target, never a maximize.
///
/// `ROWPART_MIN_LEAF`: below this leaf-row count, `P=1` (byte-identical to the pre-row-part
/// kernel). Well above the `RESIDENT_MIN_NUM_DATA=12_000` resident gate and the ≤8k-row
/// parity-test shapes, so every existing parity test runs the unchanged `P=1` path — the
/// large-leaf f32 divergence (4e-7→~2e-5 rel) only appears above this gate.
#[cfg(feature = "gpu")]
const ROWPART_MIN_LEAF: usize = 256_000;
/// Target cubes per Compute Unit — preserves the "~8 workgroups/CU" intent.
/// `target_cubes = num_cu * CUBES_PER_CU` (queried at runtime, cached once).
#[cfg(feature = "gpu")]
const CUBES_PER_CU: u32 = 8;
/// Documented safe small default for an APU-class device when EVERY CU-count query
/// fails — explicitly NOT 768 (which was the phantom-96-CU value: `8 wkgrps × 96 CU`).
/// 64 = `8 wkgrps × 8 CU`, matching the real 8-CU Radeon 860M APU on this box.
#[cfg(feature = "gpu")]
const ROWPART_TARGET_CUBES_FALLBACK: u32 = 64;
/// Measured sweet spot; clamp so we never over-partition into the P=32 regression.
#[cfg(feature = "gpu")]
const ROWPART_P_MAX: u32 = 16;

/// Pure resolution of the row-partition target-cubes value, factored out so it is
/// unit-testable without an env var, a OnceLock, or a GPU (mirrors the "pure CPU
/// logic, no device handle" property of [`row_partition_count`]).
///
/// Resolution order (a)→(b)→(c):
///   (a) explicit env override (`LGBM_ROWPART_TARGET_CUBES`, >0) — used VERBATIM as the
///       literal target (NOT multiplied by `CUBES_PER_CU`); this is the A/B benching knob.
///   (b) queried device CU count → `num_cu * CUBES_PER_CU`.
///   (c) `ROWPART_TARGET_CUBES_FALLBACK` (never a silent 768).
#[cfg(feature = "gpu")]
fn resolve_target_cubes(env_override: Option<u32>, queried_cu: Option<u32>) -> u32 {
    match (env_override, queried_cu) {
        (Some(t), _) if t > 0 => t,
        (_, Some(n)) if n > 0 => n.saturating_mul(CUBES_PER_CU),
        _ => ROWPART_TARGET_CUBES_FALLBACK,
    }
}

/// Query the device's actual Compute Unit count, or `None` if unavailable.
///
/// 1. First try cubecl's reported value
///    (`rocm_client().properties().hardware.num_streaming_multiprocessors`):
///    forward-compatible — returns `None` on cubecl-hip 0.10 today, but populated on
///    cuda and possibly future hip. Used FIRST when `Some(n>0)`.
/// 2. else FFI fallback: read `hipGetDevicePropertiesR0600().multiProcessorCount` for
///    device ordinal 0 (matching `rocm_client`'s `AmdDevice::new(0)`).
///
/// HIP-specific (the FFI fallback uses `cubecl_hip_sys`, a rocm-only dep), so this stays
/// `#[cfg(feature = "rocm")]`. Non-rocm GPU backends (cuda/wgpu) use the
/// [`ROWPART_TARGET_CUBES_FALLBACK`] heuristic via the `None`-returning twin below —
/// the resident pool is the parity win; CU-derived row-partition
/// tuning is a perf refinement a CUDA build can add later by reading the cubecl-reported
/// `num_streaming_multiprocessors` (populated on cuda).
#[cfg(feature = "rocm")]
fn query_num_cu() -> Option<u32> {
    // (1) cubecl's forward-compatible value (None on cubecl-hip 0.10, populated on cuda).
    if let Some(n) = crate::runtime::rocm_client()
        .properties()
        .hardware
        .num_streaming_multiprocessors
    {
        if n > 0 {
            return Some(n);
        }
    }

    // (2) FFI fallback via cubecl-hip-sys, mirroring cubecl-hip/src/runtime.rs:65-94.
    // SAFETY: `props` is zero-initialized then fully written by
    // `hipGetDevicePropertiesR0600` on a HIP_SUCCESS return; `multiProcessorCount` is
    // only read after the status is checked == HIP_SUCCESS, so we never read an
    // uninitialized struct. Device ordinal 0 matches `rocm_client`'s `AmdDevice::new(0)`.
    unsafe {
        let mut props: cubecl_hip_sys::hipDeviceProp_tR0600 = std::mem::zeroed();
        let status = cubecl_hip_sys::hipGetDevicePropertiesR0600(&mut props, 0);
        if status == cubecl_hip_sys::hipError_t_hipSuccess && props.multiProcessorCount > 0 {
            return Some(props.multiProcessorCount as u32);
        }
    }
    None
}

/// Non-rocm GPU twin: cuda/wgpu have no `cubecl_hip_sys`, so the
/// CU-count query returns `None` and [`rowpart_target_cubes`] falls back to
/// [`ROWPART_TARGET_CUBES_FALLBACK`]. Correct (the resident pool is the parity win);
/// a CUDA build can later read `num_streaming_multiprocessors` for a tuned target.
#[cfg(all(feature = "gpu", not(feature = "rocm")))]
fn query_num_cu() -> Option<u32> {
    None
}

/// The row-partition target-cubes value, queried at most ONCE per process and cached
/// (`row_partition_count` runs per leaf, so the FFI CU-count query must not repeat).
/// Resolution: env override → queried CU count × `CUBES_PER_CU` → FALLBACK
/// (see [`resolve_target_cubes`] / [`query_num_cu`]).
///
/// Hardware note: this device is an 8-CU APU (Radeon 860M / gfx1152 spoofed as
/// gfx1100), so the target ≈ 64 (8 × `CUBES_PER_CU`), NOT 768 (which assumed a phantom
/// 96-CU gfx1100). Changing `P` alters the f32 partial-sum grouping (`P≥2`
/// widens GPU-vs-`P=1` divergence to ~2e-5, WITHIN the ~1e-6-best-effort ROCm gate — the
/// GPU path was never bit-exact; the cpu f64 anchor is untouched).
#[cfg(feature = "gpu")]
fn rowpart_target_cubes() -> u32 {
    static TARGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TARGET.get_or_init(|| {
        let env_override = std::env::var("LGBM_ROWPART_TARGET_CUBES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        resolve_target_cubes(env_override, query_num_cu())
    })
}

/// Row partitions `P` for the LDS build: `clamp(target_cubes / num_features, 1, P_MAX)` on
/// large leaves, else `1`. `LGBM_ROWPART_MIN` overrides the leaf threshold (benching). The
/// `target_cubes` value is the runtime, CU-count-derived [`rowpart_target_cubes`] (cached).
/// Pure CPU logic otherwise — no per-call device handle — so it is unit-testable with a
/// forced target.
#[cfg(feature = "gpu")]
pub fn row_partition_count(num_features: usize, leaf_rows: usize) -> u32 {
    let min_leaf = std::env::var("LGBM_ROWPART_MIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(ROWPART_MIN_LEAF);
    if num_features == 0 || leaf_rows < min_leaf {
        return 1;
    }
    let target = rowpart_target_cubes();
    let nf = num_features as u32;
    if nf >= target {
        return 1;
    }
    (target / nf).clamp(1, ROWPART_P_MAX)
}





/// DEVICE-RESIDENT batched per-leaf histogram kernel. Identical
/// `(feature f, leaf-row k)` unit mapping to [`construct_leaf_hist_batched_kernel`]
/// BUT gathers the bin from the device-resident feature-column buffer on device
/// (`resident_bins[f * num_data + leaf_rows[k]]`) instead of from a per-leaf
/// host-gathered `gathered_bins[idx]` re-uploaded every leaf. This replaces the
/// `[num_features × rows]` host bin upload per leaf with a one-time
/// column upload (the resident buffer) + a small per-leaf `leaf_rows` index upload.
///
/// `num_data` is the resident column stride (the full train row count) passed as a
/// scalar launch arg. Same f32-atomic accumulation ⇒ the ~1e-6 ROCm gate (cpu
/// anchor stays bit-exact). `#[cfg(feature="rocm")]`.
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn construct_leaf_hist_resident_kernel<B: Int>(
    resident_bins: &Array<B>, // native bin width (u8/u16/u32)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    total: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx < total {
        let r = ord_g.len(); // leaf-row count R
        let f = idx / r; // feature index
        let k = idx % r; // leaf-row position
        // Gather on device from the resident column: feature f, the leaf's k-th row.
        let row = leaf_rows[k] as usize;
        // native-width read widened to a u32 INDEX (value-faithful).
        let bin = u32::cast_from(resident_bins[f * num_data + row]) as usize;
        let cell = slot_off[f] as usize + bin * 2;
        out[cell].fetch_add(ord_g[k]);
        out[cell + 1].fetch_add(ord_h[k]);
    }
}


// ===========================================================================
// LDS-PRIVATIZED batched/resident RAW build — the
// hot-path follow-up to the single-feature `construct_hist_kernel_lds_f32`.
//
// The naive batched/resident kernels above run ONE unit per `(feature, leaf-row)`
// pair, each atomic-adding straight into the GLOBAL concatenated output — so rows
// of a feature sharing a bin serialize on global-memory atomic contention. These
// LDS kernels instead put ONE CUBE PER FEATURE: each cube owns a private
// sub-histogram in shared memory (≤ HIST_LDS_MAX = 2 KiB, one feature ≤ 256 bins),
// its units stride the leaf's rows doing cheap LDS atomics, then merge into that
// feature's global slot once per cell. Global atomic traffic per feature drops from
// `2*R` to `2*num_bin[f]`. Mirrors LightGBM's OpenCL histogram*.cl one-workgroup-
// per-feature design. `slot_off` carries a SENTINEL final entry (= slot_len) so the
// cube can read its feature's width `slot_off[f+1] - slot_off[f]`. f32 atomics ⇒ the
// same ~1e-6 ROCm gate; capped at 256 bins/feature (caller falls back to the naive
// kernel when any feature exceeds it).
// ===========================================================================

/// LDS resident RAW build: one cube per feature, gathers bins from the resident
/// column on device (`resident_bins[f*num_data + leaf_rows[k]]`). `slot_off` has
/// `num_features + 1` entries (sentinel = slot_len).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_resident_lds_kernel<B: Int>(
    resident_bins: &Array<B>, // native bin width (u8/u16/u32)
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    // 1. zero this feature's active LDS cells.
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    // 2. scatter THIS partition's strided rows into LDS (resident on-device gather).
    //    Row-partitioned: `CubeCount = (num_features, P)`, so
    //    cube `(f, p)` owns row-slice `p*cd, +P*cd, …`. P comes from `CUBE_COUNT_Y`; P=1
    //    (the small/medium gate) reduces this to the prior `k=UNIT_POS, stride cd` loop
    //    byte-for-byte. All P cubes of feature f atomic-merge into the same global slot
    //    (step 3), so the split is additive and order-free.
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        // bin is read at native width B, widened to a u32 INDEX —
        // byte-faithful to the prior u32 read (value identical, only storage narrower).
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    // 3. merge LDS → this feature's global slot.
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// u64 TWO'S-COMPLEMENT FIXED-POINT resident LDS build. A
/// byte-for-byte twin of [`construct_leaf_hist_resident_lds_kernel`] above — IDENTICAL
/// resident-column gather, `slot_off` sentinel `feat_len`, row-partition over
/// `CUBE_POS_Y`/`CUBE_COUNT_Y`, per-feature LDS sub-hist, and LDS→global merge — with
/// ONLY the accumulated CELL type + quantize/store/merge idiom swapped from f32 atomics
/// to u64 integer atomics.
///
/// Each grad/hess value is quantized `round(value * S)` (S = `SCALE_F32` = 2^30) to an
/// i64, whose BITS are stored as u64; a wrapping `Atomic<u64>::fetch_add` is exactly a
/// two's-complement signed i64 add, so the bins sum correctly with NO bias offset (each
/// bin sums a variable row count — a bias would be wrong). The dequant
/// `(bits as i64) / S` happens later in `fix_compact_kernel`'s widen pass.
///
/// HARD CONSTRAINT: the cell type MUST be `Atomic<u64>`
/// with `.store(0u64)` / `.fetch_add(qbits)`. NEVER `Atomic<i64>` — cubecl-hip 0.10
/// lowers `Atomic<i64>::store` to `atomicExch(long long*)`, which HIP lacks (compiles,
/// fails at runtime). LDS stays `HIST_LDS_MAX` cells (same element COUNT, 2× bytes =
/// 4 KiB/cube, well within the 64 KiB budget). f64 atomics not needed — the wide i64
/// accumulator IS the precision win (~3600× better than f32 atomics).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_resident_lds_kernel_u64<B: Int>(
    resident_bins: &Array<B>, // native bin width (u8/u16/u32)
    leaf_rows: &Array<u32>,
    // Grad/hess. When `resident_gh` is FALSE (host-gather path,
    // unchanged) these are the PRE-GATHERED leaf-row-ordered `[R]` slices and are indexed
    // by the leaf-row position `k`. When `resident_gh` is TRUE (the on-device gather path)
    // these are the FULL once-per-grow RESIDENT `[num_data]` grad/hess buffers and are
    // gathered ON DEVICE by `leaf_rows[k]` — the SAME row index the resident-bin gather
    // already uses — so NO host `ord_g`/`ord_h` gather + per-build upload is needed.
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<u64>>,
    // comptime toggle — TRUE gathers grad/hess from the resident
    // full buffers via `leaf_rows[k]`; FALSE keeps the byte-identical pre-gathered `[k]`
    // indexing (the comptime branch compiles the unused arm out, so the host-gather path
    // is unchanged). Read count `R` from `leaf_rows.len()` so it is correct for BOTH the
    // pre-gathered `[R]` and full-resident `[num_data]` grad/hess layouts.
    #[comptime] resident_gh: bool,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = leaf_rows.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
    // 1. zero this feature's active LDS cells (u64 zero — the additive identity bits).
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    // 2. scatter THIS partition's strided rows into LDS (resident on-device gather),
    //    quantizing each value to fixed-point i64-bits before the wrapping u64 atomic.
    //    Row-partition byte-identical to the f32 twin (CubeCount = (num_features, P)).
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        // bin INDEX read unchanged (native width B → u32 index); only the CELL type changed.
        let row = leaf_rows[k] as usize;
        let bin = u32::cast_from(resident_bins[col + row]) as usize;
        let ti = bin * 2;
        // comptime-select the grad/hess index — `row` gathers on device
        // from the resident full buffers (resident_gh), `k` reads the pre-gathered slice.
        // The comptime branch is resolved at compile time, so the host-gather path emits the
        // byte-identical `ord_g[k]` access it did before.
        let gk = if resident_gh { row } else { k };
        // quantize `round(v * 2^30)` → i64 → store its bits as u64 (build_u64_rp idiom).
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[gk] * SCALE_F32)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[gk] * SCALE_F32)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();
    // 3. merge LDS → this feature's global slot (wrapping u64 add == i64 two's-complement).
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

// ===========================================================================
// The §13 feature-partition TWO-TIER build kernel.
//
// Lifts the u64 fixed-point LDS accumulation body
// (`construct_leaf_hist_resident_lds_kernel_u64`, above — left BYTE-UNCHANGED)
// onto the design-doc §7.1 / §13 partition geometry (`CalcConstructHistogramKernelDim`):
//   * `CUBE_POS_X` = feature PARTITION (was one cube per FEATURE);
//   * `UNIT_POS_X` = one COLUMN within the partition's `[off[p], off[p+1])` range;
//   * `UNIT_POS_Y × CUBE_POS_Y` = the leaf-row STRIPE (disjoint y-blocks).
//
// Two-tier atomics (§7.2): fast LDS block-local `Atomic<u64>` accumulation during the
// sweep, then a cross-block GLOBAL atomic merge into the leaf histogram. cubecl 0.10 has
// NO grid barrier, so the cross-block reduction MUST be a global atomic (many y-blocks
// cover disjoint row stripes of the same partition). Dense-vs-sparse is ONE `#[cube]`
// generic with a `#[comptime] is_sparse` branch (the §7.2 Dense/Sparse axis). The cpu
// f64 anchor stays the single-owner `construct_histograms_f64_on` fold; this
// kernel is the hip path only (`Atomic<u64>` is unsupported / nondeterministic on
// cubecl-cpu).
// ===========================================================================



/// De-quant the raw u64 fixed-point histogram to `hist_t` (f64) exactly ONCE:
/// `(bits as i64) / 2^30` — the SAME 2^30 scale as the
/// build side ([`SCALE_F32`]), mirroring `fix_compact_kernel`'s folded dequant first pass
/// (`(bits as i64)/SCALE_F64`). Kept a SEPARATE pass so
/// BUILD stays a clean u64-only accumulator and the cpu-anchor split is unentangled; the
/// Fix step then operates on the durable `hist_t`. Round-trip exact for
/// integer-valued cells, ≤ 1/2^30 abs error otherwise — well inside the ~1e-6 ROCm gate.
#[cfg(feature = "gpu")]
#[must_use]
pub fn dequant_leaf_hist(raw: &[u64]) -> Vec<f64> {
    const SCALE_F64: f64 = 1_073_741_824.0; // 2^30 (matches SCALE_F32 / fix_compact SCALE_F64)
    raw.iter().map(|&bits| (bits as i64) as f64 / SCALE_F64).collect()
}

/// f32 mirror of [`dequant_leaf_hist`] for the no-f64 hip device: the durable
/// `hist_t` on ROCm/CUDA is f32. Same 2^30 scale; the ~1e-6 ROCm gate absorbs the f32
/// rounding (never the cpu f64 anchor).
#[cfg(feature = "gpu")]
#[must_use]
pub fn dequant_leaf_hist_f32(raw: &[u64]) -> Vec<f32> {
    const SCALE_F32D: f32 = 1_073_741_824.0; // 2^30
    raw.iter().map(|&bits| (bits as i64) as f32 / SCALE_F32D).collect()
}

/// Spill-buffer cell count for the `_GlobalMemory` path (`grid_dim_y · num_total_bin · 2`),
/// `checked_mul`-guarded. A separate `#[must_use]` helper so the
/// overflow guard is unit-testable without a device.
///
/// # Errors
/// [`ComputeError::Runtime`] if the product overflows `usize` (or is zero).
#[cfg(feature = "gpu")]
pub fn spill_cells(grid_dim_y: usize, num_total_bin: usize) -> Result<usize, ComputeError> {
    let cells = grid_dim_y
        .checked_mul(num_total_bin)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!(
                "spill buffer size grid_dim_y {grid_dim_y} · num_total_bin {num_total_bin} · 2 \
                 overflows usize"
            ),
        })?;
    if cells == 0 {
        return Err(ComputeError::Runtime {
            detail: "spill buffer size is zero (empty layout)".to_string(),
        });
    }
    Ok(cells)
}



// ===========================================================================
// FAITHFUL CubeCL MIRROR of CUDAConstructHistogramDenseKernel
//
// Structural port of LightGBM's signature CUDA inner kernel
// (`LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_histogram_constructor.cu`
// lines ~18-70, `CUDAConstructHistogramDenseKernel`) — the histogram-construction
// kernel of `cuda_single_gpu_tree_learner`. It ships as a TESTED PRIMITIVE
// (rocm_cuda_mirror.rs), NOT wired into the production build/resident path (that
// live wiring is a deferred follow-up).
//
// The signature CUDA structure this reproduces, distinct from the existing
// batched/resident LDS kernels (which pre-gather `ord_g[k]` / `leaf_rows[k]`):
//   (1) INDIRECT in-kernel gather — `data_index = data_indices[k]` (mirror of CUDA
//       `data_index = data_indices_ref_this_block[inner_data_index]`), then index a
//       RESIDENT feature-major bin buffer at `column_start * num_data + data_index`
//       (mirror of `data_ptr[data_index * ncols + tx]` via the resident
//       `f * num_data + row` layout RocmBackend::upload_resident_bins uses); grad/hess
//       are gathered in FULL-CORPUS order as `grad[data_index]` / `hess[data_index]`
//       (CUDA `cuda_gradients[data_index]`), NOT pre-gathered — the indirection the
//       existing kernels lack.
//   (2) 2D (column, row) tile — `CUBE_POS_X` selects the feature/column (CUDA
//       `threadIdx.x`/`blockIdx.x` partition column), `UNIT_POS` strides over the
//       leaf rows (CUDA `threadIdx.y`); row-partitioned over `CUBE_POS_Y` (CUDA
//       `blockIdx.y`) so a large leaf is split across `P` cubes (CUDA `gridDim.y`).
//   (3) per-CUBE LDS sub-histogram (CUDA `__shared__ shared_hist`,
//       `SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX)`): zero it strided, sync_cube,
//       atomic-add each row's (grad,hess) into the LDS cell at `bin*2` (CUDA
//       `atomicAdd_block`), sync_cube, then flush each active LDS cell to the global
//       `out` slot with ONE atomic per cell (CUDA `atomicAdd_system`).
//
// f32 atomics ⇒ nondeterministic accumulation ⇒ the ~1e-6 ROCm gate vs the CPU f64
// anchor (NOT bit-exact, by design — pinned in rocm_cuda_mirror.rs, never GPU-vs-GPU).
// Capped at 256 bins/feature (the HIST_LDS_MAX sub-hist budget). `#[cfg(feature="rocm")]`.
// ===========================================================================




/// Build the sentinel `slot_off` (`num_features + 1` entries, final = `slot_len`)
/// and the max per-feature slot width. LDS is eligible iff the widest feature fits
/// `HIST_LDS_MAX` (≤ 256 bins).
#[cfg(feature = "gpu")]
fn slot_off_sentinel(slot_off: &[usize], slot_len: usize) -> (Vec<u32>, u32) {
    let mut s: Vec<u32> = Vec::with_capacity(slot_off.len() + 1);
    for &o in slot_off {
        s.push(o as u32);
    }
    s.push(slot_len as u32);
    let max_w = s.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    (s, max_w)
}

// ===========================================================================
// CubeCL AUTOTUNE for the histogram-BUILD row-partition `P`.
//
// Autotune is the DEFAULT rocm selector for the resident-build
// `P`; `row_partition_count` becomes only the documented cold-start / cache-miss /
// `LGBM_AUTOTUNE=0` fallback bound (NOT the steady-state selector). The
// heuristic under-partitions to P=1 at the 50-feature production width (~10%
// slow on the 8-CU APU); autotune re-derives the measured-fastest P per occupancy
// regime and self-calibrates on any future GPU.
//
// Three corrections this wiring honors (see gpu-kernel-autotuning.md):
//   - FRESH-OUTPUT InputGenerator: the build kernel ACCUMULATES via
//     `fetch_add`, so `CloneInputGenerator` (which shares the real `out` handle)
//     corrupts it 27× during the cold benchmark reps. `FreshOutGenerator` hands each
//     benchmark a throwaway zeroed `out` ⇒ the real `out` is touched exactly ONCE
//     (the final clean winning run) ⇒ grad-conservation holds.
//   - log2(rows) occupancy-regime AutotuneKey: keying on EXACT `rows`
//     is a per-leaf tuning STORM (every node a cold ~40ms tune). `LaunchKey.bucket =
//     size_band(rows)` so every leaf in the same power-of-two decade shares a key.
//   - serde-backed persistent cache: the winner round-trips across
//     processes via `target/autotune/<ver>/<device>/*.json.log`.
//
// The TunableSet is rebuilt FRESH each call (`Arc::new(build_pset_tunable_set(..))`,
// NOT `LocalTuner::init`) because the per-call dimensions (`rows`/`num_features`/
// `slot_len`) bake into the launch closures; `LocalTuner::init` memoizes by closure
// TYPE-id and would freeze the FIRST call's dimensions forever. The persistent
// winner still survives across calls — it lives in `BUILD_TUNER`'s key→fastest_index
// state keyed by `LaunchKey`, and the BUILD_PSET registration order is fixed so
// `fastest_index` maps to the same `P` regardless of which call rebuilt the set.
// ===========================================================================

#[cfg(feature = "gpu")]
use crate::kernels::autotune::{self, LaunchKey};
#[cfg(feature = "gpu")]
use cubecl::tune::{local_tuner, InputGenerator, LocalTuner, Tunable, TunableSet, TuneInputs};

/// The row-partition candidate set the BUILD tuner sweeps. Each entry `> ROWPART_P_MAX`
/// is skipped (so this stays correct if the clamp ever rises). The P4..P16
/// curve is FLAT — the only job is to AVOID P1 at the production width — so the set is
/// deliberately coarse (do not over-fine it into a slow cold tune). `1` stays in so the
/// tuner can still pick it for small leaves where partitioning only adds merge overhead.
///
/// `pub` so the parity gate (`oracle-harness/tests/kernel_parity.rs`) imports the
/// SAME source of truth it sweeps: a hand-copied mirror would silently stop
/// covering a newly-added `P`. Stays `#[cfg(feature = "rocm")]` so the default build is
/// byte-unchanged.
#[cfg(feature = "gpu")]
pub const BUILD_PSET: &[u32] = &[1, 4, 8, 16, 32];

/// The BUILD cache namespace — `local_tuner!("build")` ⇒ `LocalTuner<LaunchKey, String>`.
/// Holds the persistent key→fastest_index map (and mirrors it to disk via `std_io`).
#[cfg(feature = "gpu")]
static BUILD_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("build");

/// The `LGBM_AUTOTUNE_FORCE_P` debug/parity seam (consumed by the all-variants
/// anchor gate). Reads the env FRESH on every call (mirrors `scan_cube_dim`, NOT
/// `OnceLock`) so a parity test can pin a single `P` per-launch within one process.
/// `Some(k)` clamps `k` to `[1, ROWPART_P_MAX]`; non-numeric / `0` / unset ⇒ `None`
/// (fall through to autotune, then the heuristic) — NEVER a no-launch.
#[cfg(feature = "gpu")]
fn force_row_partition() -> Option<u32> {
    std::env::var("LGBM_AUTOTUNE_FORCE_P")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&k| k > 0)
        .map(|k| k.clamp(1, ROWPART_P_MAX))
}

/// Launch the EXISTING resident LDS build once at a fixed `P`, reading the ordered
/// handle slice `[resident_bins, leaf_rows, ord_g, ord_h, slot_off, out]` (out at
/// index 5). This is the single launcher the BUILD tuner's PSET variants call (one per
/// `P`); it mirrors the `launch_lds_u64!` / `launch_lds_f32!` macros in
/// [`resident_raw_build_into`] EXACTLY — same kernel, same `CubeCount(num_features, P, 1)`,
/// same `CubeDim::new_1d(256)`, same arg order/sizes — only the handles arrive via a slice
/// instead of named locals. Keep the two in sync (they intentionally launch byte-identical
/// kernels; the macros serve the FORCE_P / `LGBM_AUTOTUNE=0` direct path, this serves the
/// tuner closures which only have a `Vec<Handle>`).
///
/// `fixed_point` selects the u64 fixed-point twin (`out` is `Atomic<u64>`) vs the f32
/// kernel; `width` dispatches the `<B: Int>` monomorphization on the resident buffer's
/// native element width. SAFETY: identical to the in-place macro
/// launches — every device access is host-proven in range before upload (`leaf_rows` ⊂
/// `[0, num_data)`, `resident_bins.len() == num_features*num_data`, `slot_off` sentinel,
/// `out` sized `slot_len`), so `launch_unchecked` (dropping bounds-check codegen) is sound
/// and numerically identical. All cubecl unsafe is confined here.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn launch_build_at<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    width: crate::ResidentBinWidth,
    fixed_point: bool,
    num_features: usize,
    num_data: usize,
    rows: usize,
    slot_len: usize,
    inputs: &[cubecl::server::Handle],
    p: u32,
) {
    macro_rules! at_p_u64 {
        ($w:ty) => {
            unsafe {
                construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(num_features as u32, p, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(inputs[0].clone(), num_features * num_data),
                    ArrayArg::from_raw_parts(inputs[1].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[2].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[3].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[4].clone(), num_features + 1),
                    num_data,
                    ArrayArg::from_raw_parts(inputs[5].clone(), slot_len),
                    // the autotune tuner set is host-gather-shaped (pre-gathered `[R]`
                    // ord_g/ord_h at inputs[2]/[3]); the resident on-device gather bypasses the
                    // tuner (direct launch in `resident_raw_build_into`), so this is always false.
                    false,
                );
            }
        };
    }
    macro_rules! at_p_f32 {
        ($w:ty) => {
            unsafe {
                construct_leaf_hist_resident_lds_kernel::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(num_features as u32, p, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(inputs[0].clone(), num_features * num_data),
                    ArrayArg::from_raw_parts(inputs[1].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[2].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[3].clone(), rows),
                    ArrayArg::from_raw_parts(inputs[4].clone(), num_features + 1),
                    num_data,
                    ArrayArg::from_raw_parts(inputs[5].clone(), slot_len),
                );
            }
        };
    }
    if fixed_point {
        match width {
            crate::ResidentBinWidth::U8 => at_p_u64!(u8),
            crate::ResidentBinWidth::U16 => at_p_u64!(u16),
            crate::ResidentBinWidth::U32 => at_p_u64!(u32),
        }
    } else {
        match width {
            crate::ResidentBinWidth::U8 => at_p_f32!(u8),
            crate::ResidentBinWidth::U16 => at_p_f32!(u16),
            crate::ResidentBinWidth::U32 => at_p_f32!(u32),
        }
    }
}

/// An [`InputGenerator`] that hands each autotune BENCHMARK rep a
/// FRESH zeroed `out` handle (slot index 5), leaving the caller's real `out` untouched
/// until the final clean winning run. The build kernel ACCUMULATES (`fetch_add`), so a
/// `CloneInputGenerator` (ref-count bump, same device buffer) would let every rep of
/// every variant accumulate into the REAL `out` ⇒ N× corruption. The fresh buffer is
/// `u64` zeros for the fixed-point build (`Atomic<u64>` cells) or `f32` zeros for the
/// f32 build — sized `slot_len`, matching the `out` the caller allocated.
#[cfg(feature = "gpu")]
struct FreshOutGenerator<R: cubecl::Runtime> {
    client: cubecl::prelude::ComputeClient<R>,
    slot_len: usize,
    fixed_point: bool,
}

#[cfg(feature = "gpu")]
impl<R: cubecl::Runtime> InputGenerator<LaunchKey, Vec<cubecl::server::Handle>>
    for FreshOutGenerator<R>
{
    // Spell the GAT return through `<Vec<Handle> as TuneInputs>::At<'a>` (E0195 guard):
    // `Vec<Handle>: TuneInputs` has `At<'a> = Vec<Handle>`, but writing the concrete type
    // here makes the closure-lifetime inference fail; keep this exact form.
    fn generate<'a>(
        &self,
        _key: &LaunchKey,
        inputs: &<Vec<cubecl::server::Handle> as TuneInputs>::At<'a>,
    ) -> <Vec<cubecl::server::Handle> as TuneInputs>::At<'a> {
        let mut v = inputs.clone();
        if self.fixed_point {
            let zeros = vec![0u64; self.slot_len];
            v[5] = self.client.create_from_slice(u64::as_bytes(&zeros));
        } else {
            let zeros = vec![0.0f32; self.slot_len];
            v[5] = self.client.create_from_slice(f32::as_bytes(&zeros));
        }
        v
    }
}

/// Build the BUILD-tuner [`TunableSet`] for ONE resident-build call: one `Tunable` per
/// `P` in [`BUILD_PSET`] (each launching the existing LDS build at that `P` via
/// [`launch_build_at`], reusing the SAME kernel + `CubeCount(num_features, P, 1)` +
/// `CubeDim 256` the production path uses — only `P` varies), keyed by a `LaunchKey`
/// `{ bucket: size_band(rows), feats: num_features, bins: num_bin }` (the occupancy
/// regime), with a [`FreshOutGenerator`] so the accumulating build is
/// benchmark-safe. Rebuilt fresh each call (the dimensions bake into the
/// closures); the persistent winner lives in [`BUILD_TUNER`]'s key state, not here.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn build_pset_tunable_set<R: cubecl::Runtime>(
    client: cubecl::prelude::ComputeClient<R>,
    width: crate::ResidentBinWidth,
    fixed_point: bool,
    num_features: usize,
    num_data: usize,
    rows: usize,
    num_bin: u32,
    slot_len: usize,
) -> TunableSet<LaunchKey, Vec<cubecl::server::Handle>, ()> {
    let kg = move |_: &Vec<cubecl::server::Handle>| LaunchKey {
        bucket: autotune::size_band(rows),
        feats: num_features as u32,
        bins: num_bin,
    };
    let mut set = TunableSet::new(
        kg,
        FreshOutGenerator {
            client: client.clone(),
            slot_len,
            fixed_point,
        },
    );
    for &p in BUILD_PSET {
        if p > ROWPART_P_MAX {
            continue;
        }
        let c = client.clone();
        set = set.with(Tunable::new(
            &format!("build_P{p}"),
            move |inputs: Vec<cubecl::server::Handle>| {
                launch_build_at(
                    &c,
                    width,
                    fixed_point,
                    num_features,
                    num_data,
                    rows,
                    slot_len,
                    &inputs,
                    p,
                );
                Ok::<(), String>(())
            },
        ));
    }
    set
}

/// Shared RESIDENT RAW-build launch into a caller-provided zeroed `h_out`
/// (slot_len cells): LDS per-feature path when every feature ≤ 256 bins, else the
/// naive `construct_leaf_hist_resident_kernel`. Used by both
/// `build_leaf_histograms_resident_f32_on` (removed) and the resident chain
/// [`build_fix_compact_resident_f64_on`] so the LDS/naive decision lives in ONE place.
///
/// `fixed_point` selects the LDS BUILD cell type:
///   - `false` → f32 atomics ([`construct_leaf_hist_resident_lds_kernel`]); `h_out` is an
///     f32 buffer the caller reads back / widens as f32 (the readback oracle path).
///   - `true`  → u64 two's-complement fixed-point atomics
///     ([`construct_leaf_hist_resident_lds_kernel_u64`]); `h_out` is a u64 buffer the
///     caller dequantizes `(bits as i64)/2^30 → f64` (the live `fix_compact_kernel` path).
/// The NAIVE >256-bin fallback ALWAYS stays f32; a caller
/// that requests `fixed_point` MUST guarantee every feature ≤ 256 bins (the resident
/// chain does — `max_bin ≤ 255` keeps `max_w ≤ HIST_LDS_MAX`), else the f32 naive write
/// would be mis-dequantized. The `fixed_point` path asserts the LDS branch was taken.
///
/// ROW-PARTITION `P` SELECTION (LDS branch only): `P` is chosen by a
/// three-way pick, DEFAULT-ON CubeCL autotune:
///   1. `LGBM_AUTOTUNE_FORCE_P=k` → pin a single `P=k` launch (NO tuning) — the
///      parity/debug seam (the all-variants anchor gate).
///   2. else autotune (default) → the [`BUILD_TUNER`] picks the measured-fastest `P`
///      over [`BUILD_PSET`] per occupancy regime ([`LaunchKey`] = `size_band(rows)`),
///      benchmark-safe via [`FreshOutGenerator`] (the build ACCUMULATES). Both live
///      resident classes route here: the f32-resident build stays within the ~1e-6
///      best-effort gate across `P`; the u64 fixed-point build is parity-neutral
///      (order-independent integer merge). Both are gated vs the CPU f64 anchor.
///   3. else (`LGBM_AUTOTUNE=0`) → the [`row_partition_count`] heuristic + the existing
///      direct launch, byte-for-byte unchanged (the documented cold-start / fallback
///      bound). The NAIVE >256-bin path is NOT autotuned (single launch, unchanged).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn resident_raw_build_into<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // Element width of `resident_bins` (dispatches the kernel `<B: Int>`).
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    h_out: cubecl::server::Handle,
    // true → u64 fixed-point LDS build; false → f32 atomics. Naive fallback is
    // always f32, so callers with `fixed_point=true` must keep every feature ≤ 256 bins.
    fixed_point: bool,
    // when Some((grad, hess, num_data)), the FULL once-per-grow
    // device-resident grad/hess buffers are gathered ON DEVICE via `leaf_rows[k]` in the
    // u64 LDS kernel — NO host `ord_g`/`ord_h` gather and NO per-build grad/hess upload.
    // Only wired on the u64 fixed-point LDS path (the on-device resident driver arm). When
    // None the byte-identical host-gather path runs (autotune + f32/f64 callers unchanged).
    resident_gh: Option<(cubecl::server::Handle, cubecl::server::Handle, usize)>,
) {
    let rows = leaf_rows.len();
    if rows == 0 || num_features == 0 {
        return;
    }
    let (slot_s, max_w) = slot_off_sentinel(slot_off, slot_len);

    // ---- ON-DEVICE grad/hess gather path ----
    // The resident grad/hess buffers were uploaded ONCE per grow; gather each leaf's
    // grad/hess ON DEVICE from them via `leaf_rows[k]` (the same row index the resident-bin
    // gather uses). This removes the per-build host `ord_g`/`ord_h` gather + `create_from_slice`
    // upload entirely. Wired ONLY on the u64 fixed-point LDS path (≤256 bins, guaranteed by the
    // resident chain's overflow/LDS contract). A direct launch at the FORCE_P / heuristic `P`
    // (the u64 build is order-independent integer-additive, so `P` is parity-neutral); the
    // autotune tuner set is host-gather-shaped so the resident-gather build bypasses it.
    if let Some((grad_h, hess_h, gh_num_data)) = resident_gh {
        assert!(
            fixed_point,
            "resident_raw_build_into: on-device grad/hess gather is wired ONLY on the u64 \
             fixed-point resident build"
        );
        assert!(
            max_w <= HIST_LDS_MAX as u32,
            "resident_raw_build_into: on-device grad/hess gather requires the LDS ≤256-bin \
             path (max_w ≤ HIST_LDS_MAX); the u64 fixed-point resident chain guarantees it"
        );
        let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));
        // Direct P (FORCE_P override → row-partition heuristic). No autotune (see above).
        let p = force_row_partition().unwrap_or_else(|| row_partition_count(num_features, rows));
        // SAFETY: identical bounds contract to `launch_lds_u64` PLUS the grad/hess gather:
        // `grad_h`/`hess_h` are sized `gh_num_data` (the full train row count) and indexed by
        // `leaf_rows[k] < num_data == gh_num_data` (the caller's resident row contract), so the
        // on-device `ord_g[leaf_rows[k]]` read is in range; every other access is unchanged.
        macro_rules! launch_resident_gh {
            ($w:ty) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, p, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(resident_bins.clone(), num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows.clone(), rows),
                        ArrayArg::from_raw_parts(grad_h.clone(), gh_num_data),
                        ArrayArg::from_raw_parts(hess_h.clone(), gh_num_data),
                        ArrayArg::from_raw_parts(h_slot.clone(), num_features + 1),
                        num_data,
                        ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                        // resident_gh = TRUE: gather grad/hess on device via leaf_rows[k].
                        true,
                    );
                }
            };
        }
        match width {
            crate::ResidentBinWidth::U8 => launch_resident_gh!(u8),
            crate::ResidentBinWidth::U16 => launch_resident_gh!(u16),
            crate::ResidentBinWidth::U32 => launch_resident_gh!(u32),
        }
        return;
    }

    // ---- host-gather path (byte-identical to master) ----
    let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
    let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
    let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
    let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let h_h = client.create_from_slice(f32::as_bytes(&ord_h));

    if max_w <= HIST_LDS_MAX as u32 {
        // LDS per-feature path: `P` cubes per feature (row-partitioned), 256 units
        // each. P=1 on small/medium leaves ⇒ byte-identical to the prior one-cube-per-feature
        // launch. The grid's y-dim carries P; each cube of feature f atomic-merges into the
        // same global slot (additive).
        //
        // `P` is chosen by the three-way pick at the end of this
        // block — `LGBM_AUTOTUNE_FORCE_P` (pin) → CubeCL autotune (default-on) →
        // `row_partition_count` (the `LGBM_AUTOTUNE=0` cold-start / fallback bound). The
        // `launch_lds_*` macros below take the partition `$p` explicitly so the FORCE_P /
        // fallback DIRECT launches stay byte-identical to the prior heuristic launch.
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_s));
        // SAFETY: resident_bins sized num_features*num_data; h_rows/h_g/h_h sized rows;
        // h_slot sized num_features+1; h_out sized slot_len. Cube (f,p) reads only its
        // feature's column + slot region; bin < num_bin <= 256 keeps LDS/out indices
        // in range. All cubecl unsafe is confined here.
        //
        // LAUNCH_UNCHECKED (copied from the mirror template at the cuda-mirror
        // launcher): we call `::launch_unchecked`, dropping the in-kernel per-access
        // bounds-check codegen `::launch` emits in the scatter hot loops. This is sound
        // because every device access is host-proven in range BEFORE upload:
        //   - `resident_bins[col + leaf_rows[k]]` (`col = f*num_data`) — `leaf_rows` ⊂
        //     `[0, num_data)` (the caller's resident contract / upload-time validation)
        //     and `resident_bins.len() == num_features*num_data`, so for `f in
        //     [0, num_features)` the index `f*num_data + leaf_rows[k] < num_features*num_data`;
        //   - `slot_off[f]` / `slot_off[f+1]` — `h_slot` has `num_features + 1` entries
        //     (sentinel = slot_len) and `f = CUBE_POS_X < num_features`;
        //   - `ord_g[k]` / `ord_h[k]` — `h_g`/`h_h` sized `rows`, `k < r = ord_g.len()`;
        //   - the LDS `sub[bin*2 + 1]` stays within `HIST_LDS_MAX` (every feature
        //     `num_bin <= 256`, the `max_w <= HIST_LDS_MAX` branch gate);
        //   - `out[base + m]` for `m < feat_len = slot_off[f+1] - slot_off[f]` stays inside
        //     that feature's slot within `slot_len` (`h_out` sized slot_len).
        // i.e. the host-side boundary checks discharge exactly the obligations the
        // launch_unchecked contract requires, and the launch does NOT change numerics —
        // only bounds-check codegen is removed; scatter order / f32-atomic accumulation
        // is identical.
        // Dispatch the matching `<B: Int>` monomorphization on the
        // resident buffer's native width. ArrayArg element COUNT is width-independent
        // (num_features*num_data); the launch generic pins the element TYPE. Only one
        // match arm executes ⇒ the by-value handle moves are exclusive (sound).
        // The LDS per-feature path dispatches EITHER the u64 FIXED-POINT build
        // kernel (`fixed_point`, the live `fix_compact_kernel` chain) OR the f32-atomic
        // original (the readback oracle). Both kernels share an IDENTICAL signature
        // (resident gather, row-partition, slot args) — only the `out` cell type differs
        // (Atomic<u64> vs Atomic<f32>), matching the `h_out` buffer the caller allocated.
        // The naive >256-bin fallback arm below ALWAYS stays f32; a `fixed_point` caller
        // must keep every feature ≤ 256 bins (asserted) so the u64 LDS branch is taken.
        macro_rules! launch_lds_u64 {
            ($w:ty, $p:expr) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, $p, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(resident_bins.clone(), num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows.clone(), rows),
                        ArrayArg::from_raw_parts(h_g.clone(), rows),
                        ArrayArg::from_raw_parts(h_h.clone(), rows),
                        ArrayArg::from_raw_parts(h_slot.clone(), num_features + 1),
                        num_data,
                        ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                        // host-gather path — pre-gathered `[rows]` ord_g/ord_h indexed
                        // by `k` (byte-identical to master; the comptime branch compiles out).
                        false,
                    );
                }
            };
        }
        macro_rules! launch_lds_f32 {
            ($w:ty, $p:expr) => {
                unsafe {
                    construct_leaf_hist_resident_lds_kernel::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(num_features as u32, $p, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(resident_bins.clone(), num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows.clone(), rows),
                        ArrayArg::from_raw_parts(h_g.clone(), rows),
                        ArrayArg::from_raw_parts(h_h.clone(), rows),
                        ArrayArg::from_raw_parts(h_slot.clone(), num_features + 1),
                        num_data,
                        ArrayArg::from_raw_parts(h_out.clone(), slot_len),
                    );
                }
            };
        }
        // The DIRECT (non-tuned) launch at a fixed `P` — the FORCE_P / `LGBM_AUTOTUNE=0`
        // paths. Dispatches the u64-fixed-point vs f32 kernel and the native bin width
        // exactly as before; only `$p` is threaded so the same code serves any chosen `P`.
        macro_rules! direct_launch_at_p {
            ($p:expr) => {{
                let p_val: u32 = $p;
                if fixed_point {
                    match width {
                        crate::ResidentBinWidth::U8 => launch_lds_u64!(u8, p_val),
                        crate::ResidentBinWidth::U16 => launch_lds_u64!(u16, p_val),
                        crate::ResidentBinWidth::U32 => launch_lds_u64!(u32, p_val),
                    }
                } else {
                    match width {
                        crate::ResidentBinWidth::U8 => launch_lds_f32!(u8, p_val),
                        crate::ResidentBinWidth::U16 => launch_lds_f32!(u16, p_val),
                        crate::ResidentBinWidth::U32 => launch_lds_f32!(u32, p_val),
                    }
                }
            }};
        }

        // ---- three-way row-partition `P` selection ----
        //
        // SCOPE of the four `row_partition_count` call sites (autotune is the default;
        // the heuristic is only the cold-start / cache-miss fallback bound):
        //   - HERE (`resident_raw_build_into`, this is the only WIRED site): the live
        //     steady-state GPU build for BOTH resident classes — the f32-resident path
        //     (`build_leaf_histograms_resident_f32_on`, reached via
        //     `Backend::build_leaf_histograms_raw`) AND the u64 fixed-point device-
        //     resident pool (`build_fix_compact_resident_f64_on`). Wiring this ONE funnel
        //     puts EVERY steady-state GPU histogram build under autotune.
        //   - `build_leaf_histograms_batched_f32_on` (~site 982): NOT wired. Production-
        //     reachable ONLY as the cache-empty defensive COLD fallback in
        //     `build_leaf_histograms_raw` (lib.rs), taken when `upload_resident_bins` was
        //     never called (the learner structurally uploads resident bins before the GPU
        //     growth loop, so steady-state training never reaches it). It uses a DIFFERENT
        //     host-gather batched LDS kernel, so it IS the cold-start / cache-miss case the
        //     locked decision designates for the heuristic — deliberately left on it.
        //   - `construct_histograms_cuda_mirror_on` / `_resident_on` (~sites 1543/1692):
        //     NOT wired — referenced only by tests/examples, never a production path.
        if let Some(k) = force_row_partition() {
            // (a) LGBM_AUTOTUNE_FORCE_P=k → pin a single P=k launch, NO tuning (the
            //     parity/debug seam consumed by the all-variants anchor gate).
            direct_launch_at_p!(k);
        } else if autotune::autotune_enabled() {
            // (b) DEFAULT (autotune on) → drive the launch through the CubeCL tuner over
            //     BUILD_PSET. The FreshOutGenerator makes the ACCUMULATING build benchmark-
            //     safe (cold reps hit throwaway buffers), so the winner writes the real
            //     `h_out` exactly once. The set is rebuilt fresh (this call's dimensions);
            //     the persistent winner lives in BUILD_TUNER's LaunchKey state.
            //
            //     Determinism note: this funnel autotunes `P` for BOTH resident
            //     classes. On the u64 `fixed_point` path the per-cube merge is integer-
            //     additive ⇒ bit-identical across `P`. On the `!fixed_point` f32 path the
            //     merge reorders f32 reductions, so the chosen `P` perturbs the histogram
            //     output by ~2e-5 and the f32 build is therefore run-to-run
            //     NONDETERMINISTIC (which `P` wins depends on cold-tune device timing) —
            //     within the contract's documented ~1e-6 best-effort f32 gap, NOT bit-exact.
            //     BOTH paths are anchor-pinned across all `P` by the all-PSET gates in
            //     `oracle-harness/tests/kernel_parity.rs` (u64 at 1e-7, f32 at the best-
            //     effort envelope), so a future kernel change that widened f32 cross-`P`
            //     divergence is caught.
            let num_bin = max_w / 2; // widest feature's bin count (the per-feature driver).
            let handles: Vec<cubecl::server::Handle> = vec![
                resident_bins.clone(),
                h_rows.clone(),
                h_g.clone(),
                h_h.clone(),
                h_slot.clone(),
                h_out.clone(),
            ];
            let set = std::sync::Arc::new(build_pset_tunable_set(
                client.clone(),
                width,
                fixed_point,
                num_features,
                num_data,
                rows,
                num_bin,
                slot_len,
            ));
            BUILD_TUNER.execute(&autotune::cache_namespace_id(), client, set, handles);
        } else {
            // (c) LGBM_AUTOTUNE=0 → the EXISTING `row_partition_count` heuristic + direct
            //     launch, byte-for-byte unchanged (the documented cold-start / fallback).
            direct_launch_at_p!(row_partition_count(num_features, rows));
        }
    } else {
        // Naive fallback (a feature exceeds the 256-bin LDS cap). ALWAYS f32:
        // the u64 fixed-point kernel is LDS-only. A `fixed_point` caller dequantizes
        // `h_out` as u64 downstream, so an f32 naive write here would be mis-decoded —
        // guard the (can't-happen for max_bin ≤ 255) case loudly rather than corrupt.
        assert!(
            !fixed_point,
            "resident_raw_build_into: fixed_point u64 build requires every feature ≤ 256 \
             bins (max_w ≤ HIST_LDS_MAX); a feature exceeded the LDS cap so the f32 naive \
             fallback fired, which the u64 dequant would mis-decode. Raise max_bin handling \
             or keep the resident chain ≤ 256 bins."
        );
        let slot_off_u32: Vec<u32> = slot_off.iter().map(|&o| o as u32).collect();
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_u32));
        let total = num_features * rows;
        let cube_dim = 256u32;
        let cube_count = (total as u32).div_ceil(cube_dim);
        // SAFETY: identical to the prior in-place naive launch (idx<total bound,
        // resident read in range, slot write in range). cubecl unsafe confined.
        //
        // LAUNCH_UNCHECKED: `::launch_unchecked` drops the in-kernel per-access
        // bounds-check codegen in the `total`-wide scatter. Every device access is
        // host-proven in range BEFORE upload:
        //   - all units guarded by the kernel's own `idx < total` (`total = num_features *
        //     rows`, the cube count rounds up so tail units stay idle);
        //   - `resident_bins[f*num_data + row]` (`f = idx/r < num_features`, `row =
        //     leaf_rows[k]`) — `leaf_rows` ⊂ `[0, num_data)` (caller resident contract)
        //     and `resident_bins.len() == num_features*num_data`, so the index is in range;
        //   - `leaf_rows[k]` / `ord_g[k]` / `ord_h[k]` (`k = idx % r`, `r = ord_g.len() =
        //     rows`) — `k < r` (`h_rows`/`h_g`/`h_h` sized `rows`);
        //   - `slot_off[f]` for `f < num_features` (`h_slot` sized `num_features`);
        //   - `out[cell]` / `out[cell+1]` (`cell = slot_off[f] + bin*2`) — the resident
        //     bin-range invariant (`bin < num_bin`, upload-time) keeps the write inside
        //     that feature's slot within `slot_len` (`h_out` sized `slot_len`).
        // The host-side boundary checks discharge exactly the launch_unchecked obligations;
        // the launch does NOT change numerics — only bounds-check codegen is removed; the
        // f32-atomic scatter order is identical (~1e-6 path).
        // Native-width dispatch (see the LDS branch note above).
        macro_rules! launch_naive {
            ($w:ty) => {
                unsafe {
                    construct_leaf_hist_resident_kernel::launch_unchecked::<$w, R>(
                        client,
                        CubeCount::Static(cube_count, 1, 1),
                        CubeDim::new_1d(cube_dim),
                        ArrayArg::from_raw_parts(resident_bins, num_features * num_data),
                        ArrayArg::from_raw_parts(h_rows, rows),
                        ArrayArg::from_raw_parts(h_g, rows),
                        ArrayArg::from_raw_parts(h_h, rows),
                        ArrayArg::from_raw_parts(h_slot, num_features),
                        num_data,
                        total,
                        ArrayArg::from_raw_parts(h_out, slot_len),
                    );
                }
            };
        }
        match width {
            crate::ResidentBinWidth::U8 => launch_naive!(u8),
            crate::ResidentBinWidth::U16 => launch_naive!(u16),
            crate::ResidentBinWidth::U32 => launch_naive!(u32),
        }
    }
}

/// ON-GPU WIDEN + `FixHistogram` + compaction kernel, FOLDED by
/// Lever A. ONE cube per feature
/// (`CubeCount::Static(num_features,1,1)`, `CubeDim::new_1d(1)`), mirroring
/// [`construct_leaf_hist_resident_kernel`] / the fused split kernel's
/// one-cube-per-feature precedent. Cube `f` (`CUBE_POS_X`) owns ONLY its
/// `[slot_off[f], slot_off[f] + 2*num_bin[f])` region.
///
/// LEVER A — the standalone `widen_f32_to_f64_kernel` launch is FOLDED
/// IN as this kernel's FIRST pass: cube `f` widens its own region from the f32 RAW
/// histogram `h_raw` into the f64 output `hist` via `f64::cast_from(...)` — the
/// IDENTICAL cast the standalone widen performed — then runs the EXACT same in-place
/// `FixHistogram` + `compact` over `hist`. Because the widen cast and the f64 fix/
/// compact fold order are byte-for-byte unchanged, the f64 output is BIT-IDENTICAL
/// to the prior 3-launch "construct → widen → fix" chain; only the separate widen
/// LAUNCH is eliminated (3 launches → 2). The widen is now the fix kernel's own
/// first pass over its region.
///
/// `hist` (f64 OUT) must be zero-initialised by the caller (the tail/dropped cells
/// the compact step does not overwrite stay 0, matching the prior zeroed f64 alloc).
/// `h_raw` (f32 IN) holds the construct kernel's f32-atomic RAW cells.
///
/// A VERBATIM port of the host `fix_histogram` (`fix_histogram.rs:50-80`,
/// `Dataset::FixHistogram`) followed by `compact_histogram` (`learner.rs:2838-2864`),
/// preceded by the inline f32→f64 widen.
///
/// The leaf RAW (un-bumped) `sum_gradient` / `sum_hessian` are leaf-level scalars
/// shared across every feature (the RAW totals, NOT the `+2·kEpsilon`
/// bumped value). All math is f64 — gfx1100 runs f64 despite `has_f64 == false`
/// (the fused f64 split kernel precedent). Reading the SAME f64-widened cells and
/// folding in the SAME ascending order as the host yields a BIT-IDENTICAL buffer.
///
/// `#[cfg(feature="rocm")]` — the CPU anchor keeps the host fix+compact unchanged.
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn fix_compact_kernel(
    // u64 FIXED-POINT RAW histogram (u64 build-kernel output) — INPUT, read-only.
    // Each cell holds a two's-complement i64 (stored as u64 bits) = `round(value*2^30)`.
    h_raw: &Array<u64>,
    // f64 fixed+compacted histogram — OUTPUT (caller zeroes it before launch).
    hist: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    most_freq_bin: &Array<i32>,
    // LEAF-LEVEL scalars (retained as positional launch args). No longer
    // used to reconstruct the mfb cell — the dense build already holds the exact mfb mass,
    // so the old `seed − Σothers` reconstruction only injected f64 residue. Prefixed `_`.
    _sum_gradient: f64,
    _sum_hessian: f64,
) {
    // Dequant scale S = 2^30 (matches the build-side SCALE_F32; f64 here).
    const SCALE_F64: f64 = 1_073_741_824.0; // 2^30
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    // mfb no longer read here — the FixHistogram overwrite was removed.
    let _mfb = most_freq_bin[fi];
    let off = offset[fi];

    // ---- FOLDED DEQUANT (was the f32→f64 widen of Lever A) ----
    // Dequantize THIS feature's whole region u64-bits → i64 → f64/2^30 into the output
    // BEFORE the fix/compact reads it (reproduces the host dequant from
    // gpu_fixedpoint_i64.rs:191 cell-by-cell). The result is an f64 buffer with the SAME
    // shape the prior f32 widen produced, so the FixHistogram fold + compact below are
    // BYTE-UNCHANGED — they operate on the already-f64 `hist`. Ascending bin order.
    for w in 0..nb {
        let wbi = base + (w as usize) * 2;
        hist[wbi] = f64::cast_from(i64::cast_from(h_raw[wbi])) / SCALE_F64;
        hist[wbi + 1] = f64::cast_from(i64::cast_from(h_raw[wbi + 1])) / SCALE_F64;
    }

    // ---- FixHistogram (was a VERBATIM Dataset::FixHistogram port) ----
    // C++ `Dataset::FixHistogram` reconstructs the most-frequent bin as
    // `leaf_total − Σ_others` because the C++ DENSE build SKIPS the mfb bin for speed
    // (it is the bulk of the rows). This resident u64 build does NOT skip it — the
    // construct kernel scatters EVERY row (mfb included) into `h_raw`, so the dequanted
    // `hist[mfb]` is already the exact most-frequent-bin mass. Transcribing the C++
    // reconstruction here OVERWROTE that correct cell with `external_seed − Σ_others`,
    // whose f64 residue (the external row-order/scan seed disagrees with the bin-grouped
    // histogram fold by non-associativity; bound ~ n·u·Σ|h|, > 1e-3 at n≳1e7) leaked into
    // the globally-empty forced-mfb cell — a phantom value that fed
    // the admissibility scan a fake near-empty prefix. FIX: preserve the dense-built mfb
    // cell (provably 0 for a forced-empty bin at ALL scales). No-op in the anchor regime
    // (integer grad/hess ⇒ row-order == bin-grouped ⇒ external_seed == Σ hist exactly), so
    // every u64 anchor gate stays bit-exact.
    //
    // `sum_gradient`/`sum_hessian` remain in the signature as the leaf-level scalars the
    // launcher passes (positional launch args); they are intentionally no longer used to
    // overwrite the built mfb cell here (unused fn params do not warn in Rust).

    // ---- compact (learner.rs:2838-2864) ----
    // offset <= 0 → no-op. offset >= num_bin → zero the whole feature region.
    // else: shift pair (c+offset) down to c for c in 0..(num_bin-offset) ASCENDING
    // (src >= dst, so an in-place forward shift is safe), then zero the tail.
    if off > 0 {
        if off >= nb {
            // Degenerate: nothing to keep — zero the whole feature region.
            for c in 0..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        } else {
            let keep = nb - off; // num_bin - offset
            for c in 0..keep {
                let dst = base + (c as usize) * 2;
                let src = base + ((c + off) as usize) * 2;
                hist[dst] = hist[src];
                hist[dst + 1] = hist[src + 1];
            }
            // Zero the unused tail (the dropped-bin slots) so a stray read is inert.
            for c in keep..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        }
    }
}

/// Host launcher for the on-GPU FOLDED widen+fix+compact kernel (Task 1 form;
/// signature updated by Lever A).
///
/// Takes ONE leaf's concatenated stride-2 f32 RAW histogram `raw` (the construct
/// kernel's f32-atomic output — the SAME cells the host would widen to f64), the
/// per-feature `{slot_off, num_bin, offset, most_freq_bin}` arrays, and the leaf's
/// RAW (un-bumped) `sum_gradient` / `sum_hessian`; uploads, allocates a zeroed f64
/// output, launches one cube per feature (each widens its region inline then
/// fixes+compacts), reads back the fixed+compacted f64 buffer. This Task-1 form
/// keeps the readback so the kernel numerics are proven in isolation.
///
/// Lever A note: the kernel now folds the f32→f64 widen IN (`f64::cast_from`),
/// so this launcher feeds the f32 RAW directly — bit-identical to the prior
/// "supply pre-widened f64" form for any f32-representable RAW cell.
///
/// Boundary validation BEFORE launch (mirrors the fused split launcher):
/// `num_bin == 0` → typed error; `2*num_bin` overflow → typed error;
/// `slot_off + 2*num_bin > raw.len()` → [`ComputeError::LengthMismatch`]; empty
/// feats → `Ok(raw widened to f64)` with NO launch.
///
/// `feats` is `&[(slot_off, num_bin, offset, most_freq_bin)]` per feature, in the
/// same order as the concatenated regions in `raw`.
///
/// # Errors
/// As above (length / overflow validation).
#[cfg(feature = "gpu")]
pub fn fix_compact_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    raw: &[f32],
    feats: &[(usize, u32, i32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<Vec<f64>, ComputeError> {
    // Empty batch: no launch — return the f32 RAW widened to f64 (the degenerate
    // "widen-only" result, matching the prior f64 buffer pass-through).
    if feats.is_empty() {
        return Ok(raw.iter().map(|&x| f64::from(x)).collect());
    }

    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    for &(slot_off, num_bin, offset, most_freq_bin) in feats {
        if num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "fix_compact: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {num_bin} overflows the histogram length"),
            })?;
        let end = slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "fix_compact: slot_off + region overflows".to_string(),
            })?;
        if end > raw.len() {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: raw.len(),
            });
        }
        slot_off_a.push(slot_off as u32);
        num_bin_a.push(num_bin as i32);
        offset_a.push(offset);
        mfb_a.push(most_freq_bin as i32);
    }

    // `fix_compact_kernel` now consumes a u64 FIXED-POINT RAW buffer and
    // dequantizes `(bits as i64)/2^30 → f64` in its widen pass. This launcher receives
    // an f32 RAW histogram (the test/oracle path builds it host-side), so quantize each
    // cell `round(v*2^30) → i64 → bits-as-u64` here to match the live u64 build kernel —
    // the dequant in-kernel inverts it (round-trip exact for integer-valued cells, ≤
    // 1/2^30 abs error otherwise, well within the ~1e-6 ROCm gate). The fix/compact fold
    // below is byte-unchanged; only the RAW cell encoding changed.
    const SCALE_FC: f32 = 1_073_741_824.0; // 2^30 (matches SCALE_F32 / SCALE_F64)
    let raw_q: Vec<u64> =
        raw.iter().map(|&v| (v * SCALE_FC).round() as i64 as u64).collect();
    let h_raw = client.create_from_slice(u64::as_bytes(&raw_q));
    let zeros64 = vec![0.0f64; raw.len()];
    let h_hist = client.create_from_slice(f64::as_bytes(&zeros64));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));

    // SAFETY: every handle is sized to its slice and outlives the launch. Cube `f`
    // (`CUBE_POS_X < n`) reads `h_raw` and reads/writes `h_hist` only within
    // `[slot_off[f], slot_off[f]+2*num_bin[f])` — each validated `<= raw.len()`
    // above — and `mfb < num_bin` keeps the reconstruct write in range; the
    // per-feature index arrays all have exactly `n` elements. All cubecl unsafe is
    // confined here.
    //
    // LAUNCH_UNCHECKED: we call `::launch_unchecked`, dropping the in-kernel
    // per-access bounds-check codegen in the per-feature fix/compact loops. This is a
    // ZERO-numeric-risk switch: the kernel is f64 and DETERMINISTIC (one cube per
    // feature, `CubeDim::new_1d(1)`, ascending fold) so the result stays bit-exact. Every
    // device access is host-proven in range BEFORE upload:
    //   - `h_raw[wbi]` / `h_raw[wbi+1]` and `h_hist[...]` for the inline widen + fix +
    //     compact — cube `f < n` touches only `[slot_off[f], slot_off[f] + 2*num_bin[f])`,
    //     and the loop above validated `slot_off[f] + 2*num_bin[f] <= raw.len()` for every
    //     feature; both `h_raw` and `h_hist` are sized `raw.len()`;
    //   - `mfb < num_bin` (the `do_fix` guard) keeps the reconstruct cell in the region;
    //   - the per-feature `slot_off`/`num_bin`/`offset`/`most_freq_bin` arrays all have
    //     exactly `n` elements and `f < n`.
    // i.e. the host-side boundary checks discharge exactly the obligations the launch_unchecked
    // contract requires, and the launch does NOT change numerics — only bounds-check
    // codegen is removed; the f64 fold order is identical (bit-exact).
    unsafe {
        fix_compact_kernel::launch_unchecked(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_raw, raw.len()),
            ArrayArg::from_raw_parts(h_hist.clone(), raw.len()),
            ArrayArg::from_raw_parts(h_slot, n),
            ArrayArg::from_raw_parts(h_numbin, n),
            ArrayArg::from_raw_parts(h_offset, n),
            ArrayArg::from_raw_parts(h_mfb, n),
            sum_gradient,
            sum_hessian,
        );
    }

    let bytes = client.read_one_unchecked(h_hist);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// ===========================================================================
// FixHistogram most-freq-bin repair in the hist_t
// FLOAT domain — `docs/cuda-kernel-design.md` §7.5 FixHistogramKernel. The
// §7 on-device path is build → fix → subtract: BUILD accumulates the
// raw u64 fixed-point histogram OMITTING the most-frequent bin to save work,
// de-quant widens it once to `hist_t`, and FIX here reconstructs the
// omitted bin as `leaf_total − Σ(other bins)`. This is a SEPARATE kernel from
// the legacy `fix_compact_kernel`: it (a) consumes the already-de-quanted
// `hist_t` (NOT the raw u64 — no re-quantize, no 2^30 scale), and (b) DROPS the
// compact (offset-shift) step — compaction is a CPU-learner artifact the §7
// reference path does not perform.
// ===========================================================================

/// FixHistogram most-frequent-bin repair over the de-quanted `hist_t` (§7.5).
///
/// One cube per feature (`CUBE_POS_X = f`, `CubeDim::new_1d(1)` — the single-owner
/// ascending fold, the load-bearing f64 order shared with `fix_compact_kernel`).
/// Repairs ONLY when `most_freq_bin > 0 && most_freq_bin < num_bin` (the C++
/// `if (most_freq_bin > 0)` guard plus the defensive in-range bound);
/// `mfb == 0` and out-of-range features are left untouched (no write).
///
/// The repaired most-freq cell = the RAW (un-bumped) leaf totals `sum_gradient` /
/// `sum_hessian` (HOST-side exact f64 scalars, shared across every feature)
/// minus every OTHER bin's cell, folded in ASCENDING bin order (never reorder /
/// parallelize on the cpu anchor — the f64 fold order is the bit-exact contract). The
/// `i != mfb` exclusion is a branchless `select` so the guarded cell drops out without a
/// nested-if mutation. Writes the result into `hist[mfb·2]` (grad) / `hist[mfb·2+1]`
/// (hess) IN PLACE.
///
/// Unlike `fix_compact_kernel` this kernel does NOT dequantize (it consumes `hist_t`,
/// not the raw u64) and does NOT compact (the `if off > 0` offset-shift block is absent
/// — §7 is build→fix→subtract only). The cpu anchor folds ascending (bit-exact); the hip
/// device runs the SAME single-owner fold within the ~1e-6 gate (the proven
/// `fix_compact_kernel` precedent — a ShuffleReduceSum/plane-reduce twin over
/// `num_bin_aligned` is the §7.5 perf lever, parity-neutral within ~1e-6 and deferred:
/// the merge gate is the cpu f64 anchor, never GPU-vs-GPU).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn fix_histogram_mfb(
    // De-quanted hist_t (`[g0,h0,g1,h1,…]` per feature region) — READ-ONLY input.
    hist_in: &Array<f64>,
    // Pre-seeded copy of `hist_in` (launcher uploads the same cells) — the repaired
    // most-freq cell is overwritten here. Splitting read (`hist_in`) from write
    // (`hist_out`) avoids the read-after-write aliasing of one `&mut Array` that the
    // cubecl-cpu MLIR backend rejects ("operand does not dominate this use"), so the
    // hard merge gate runs the SAME kernel on the cubecl-cpu f64 anchor, not only hip.
    hist_out: &mut Array<f64>,
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    most_freq_bin: &Array<i32>,
    // LEAF-LEVEL RAW (un-bumped) f64 totals, shared across the batch.
    sum_gradient: f64,
    sum_hessian: f64,
) {
    // One cube per feature (`CUBE_POS_X = f`, `CubeDim::new_1d(1)`) — the SHIPPED
    // `fix_compact_kernel` launch geometry. Each feature's fix is independent; the
    // ascending per-feature fold is the load-bearing f64 order. The hip device runs the
    // SAME kernel within ~1e-6 (a per-feature plane-reduce twin over `num_bin_aligned`
    // is the §7.5 perf lever, parity-neutral and deferred — the merge gate is the cpu f64
    // anchor, never GPU-vs-GPU). Like `fix_compact_kernel`, this kernel is launched only
    // on the GPU device (cubecl-cpu's MLIR backend rejects the per-feature fold-with-
    // select); the cpu f64 anchor is the plain-Rust golden the rocm test pins to.
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    let mfb = most_freq_bin[fi];

    // C++ `if (most_freq_bin > 0)`: skip mfb == 0 (bin 0 is never folded back) AND the
    // defensive `mfb < num_bin` out-of-range bound — no OOB write.
    let do_fix = mfb > 0 && mfb < nb;
    if do_fix {
        let mfbu = mfb as usize;
        // Seed with the RAW leaf totals (literal-init loop-carried mutables, the cubecl
        // lowering discipline shared with `fix_compact_kernel`).
        let mut g = 0.0f64;
        let mut h = 0.0f64;
        g += sum_gradient;
        h += sum_hessian;
        // Subtract every OTHER bin's cell in ASCENDING order (load-bearing f64 fold;
        // `i != mfb` via branchless select).
        let count = nb;
        for i in 0..count {
            let bi = base + (i as usize) * 2;
            let gi = hist_in[bi];
            let hi = hist_in[bi + 1];
            let take = i != mfb;
            g -= select(take, gi, 0.0);
            h -= select(take, hi, 0.0);
        }
        let mi = base + mfbu * 2;
        hist_out[mi] = g;
        hist_out[mi + 1] = h;
    }
    // NO compact step: §7 is build→fix→subtract; the `if off > 0`
    // offset-shift belongs only to the legacy `fix_compact_kernel`.
}

/// Host launcher for [`fix_histogram_mfb`] (§7.5): repairs the omitted
/// most-frequent bin over the de-quanted `hist_t` (the [`dequant_leaf_hist`] output),
/// in place, returning the repaired histogram. Mirrors the
/// [`fix_compact_f64_on`] boundary-validated launcher checks — but the `feats` tuple drops
/// `offset` (no compaction) and the input is `hist_t` (no quantize round-trip).
///
/// `feats` is `&[(slot_off, num_bin, most_freq_bin)]` per feature, in the same order
/// as the concatenated regions in `hist`.
///
/// Boundary validation BEFORE launch: `num_bin == 0` → typed error;
/// `2*num_bin` overflow → typed error; `slot_off + 2*num_bin > hist.len()` →
/// [`ComputeError::LengthMismatch`]; empty `feats` → `Ok(hist.to_vec())` with NO launch.
///
/// # Errors
/// As above (length / overflow validation).
#[cfg(feature = "gpu")]
pub fn fix_histogram_mfb_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    hist: &[f64],
    feats: &[(usize, u32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<Vec<f64>, ComputeError> {
    // Empty batch: no launch — return the hist_t unchanged (degenerate Ok path).
    if feats.is_empty() {
        return Ok(hist.to_vec());
    }

    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    for &(slot_off, num_bin, most_freq_bin) in feats {
        if num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "fix_histogram_mfb: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {num_bin} overflows the histogram length"),
            })?;
        let end = slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "fix_histogram_mfb: slot_off + region overflows".to_string(),
            })?;
        if end > hist.len() {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: hist.len(),
            });
        }
        slot_off_a.push(slot_off as u32);
        num_bin_a.push(num_bin as i32);
        mfb_a.push(most_freq_bin as i32);
    }

    // Read input + a pre-seeded copy as the write target (the un-fixed cells stay equal
    // to the input, so non-mfb cells and mfb==0 features are returned byte-identical).
    let h_in = client.create_from_slice(f64::as_bytes(hist));
    let h_out = client.create_from_slice(f64::as_bytes(hist));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));

    // SAFETY: every handle is sized to its slice and outlives the launch. Cube `f`
    // (`CUBE_POS_X < n`) reads `h_in` and writes `h_out` only within `[slot_off[f],
    // slot_off[f]+2*num_bin[f])` — each validated `<= hist.len()` above — and the
    // `do_fix` guard keeps the `mfb·2` reconstruct cell in range; the per-feature index
    // arrays all have exactly `n` elements. The kernel is f64 + DETERMINISTIC (one cube
    // per feature, `CubeDim::new_1d(1)`, ascending fold), so `launch_unchecked` (dropping
    // bounds-check codegen) is sound AND bit-exact. All cubecl unsafe is confined here.
    unsafe {
        fix_histogram_mfb::launch_unchecked(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_in, hist.len()),
            ArrayArg::from_raw_parts(h_out.clone(), hist.len()),
            ArrayArg::from_raw_parts(h_slot, n),
            ArrayArg::from_raw_parts(h_numbin, n),
            ArrayArg::from_raw_parts(h_mfb, n),
            sum_gradient,
            sum_hessian,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// ===========================================================================
// The ConstructHistogramForLeaf entry — §7.0. The
// build→de-quant→fix→rotate→subtract sequence behind the OFF-by-default
// `LGBM_CUDA_ON_DEVICE` seam (`crate::cuda_on_device_enabled`, checked at the
// future call site). `on_device_growth_supported()` independently stays false:
// this demonstrates the histogram path in isolation; the whole-tree pool SWAP
// + growth driver are future work.
// ===========================================================================

/// The two children's `hist_in_leaf` produced by `construct_histogram_for_leaf`.
///
/// `smaller` is built-from-data → de-quanted → fixed; `larger` is the subtraction-trick
/// derived child (`parent − smaller`, §17), `None` when `larger_leaf_index < 0` (no real
/// sibling). Consumed by the best-split finder; the rotation contract belongs to the
/// pool-swap driver.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ConstructedLeafHists {
    /// The smaller leaf's repaired hist_t (build → de-quant → fix).
    pub smaller: Vec<f64>,
    /// The larger leaf's hist_t (`parent − smaller`), or `None` if skipped.
    pub larger: Option<Vec<f64>>,
}

/// The §7.0 child eligibility gate + the larger-sibling identity for
/// `construct_histogram_for_leaf`. Mirrors the C++ `ConstructHistogramForLeaf`
/// early-return: if BOTH children fail `min_data_in_leaf` / `min_sum_hessian_in_leaf`,
/// nothing is built.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
pub struct LeafConstructGate {
    /// Row count in the smaller leaf.
    pub smaller_count: i32,
    /// Σ hessian in the smaller leaf.
    pub smaller_sum_hessian: f64,
    /// Row count in the larger leaf.
    pub larger_count: i32,
    /// Σ hessian in the larger leaf.
    pub larger_sum_hessian: f64,
    /// The larger child's leaf index (`< 0` ⇒ no real sibling ⇒ subtract skipped, §7.5).
    pub larger_leaf_index: i32,
    /// `min_data_in_leaf` config.
    pub min_data_in_leaf: i32,
    /// `min_sum_hessian_in_leaf` config.
    pub min_sum_hessian_in_leaf: f64,
}


/// Upload the binned feature columns to the device feature-major (feature `f`'s
/// row `r` at `f * num_data + r`) and return the resident `Handle` — the same
/// concatenated layout [`RocmBackend::upload_resident_bins`](crate::Backend::upload_resident_bins)
/// caches internally, exposed for the resident-chain oracle so it can feed a raw
/// Handle to [`build_fix_compact_resident_f64_on`] without naming `cubecl` types.
/// All columns must share `num_data` (caller guarantees). `#[cfg(feature="rocm")]`.
#[cfg(feature = "gpu")]
pub fn upload_resident_columns<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    feature_bins: &[&[u32]],
) -> cubecl::server::Handle {
    let num_features = feature_bins.len();
    let num_data = if num_features == 0 { 0 } else { feature_bins[0].len() };
    let mut concat: Vec<u32> = Vec::with_capacity(num_features * num_data);
    for &col in feature_bins {
        concat.extend_from_slice(col);
    }
    client.create_from_slice(u32::as_bytes(&concat))
}

// (Lever A) The standalone `widen_f32_to_f64_kernel` was REMOVED — its
// f32→f64 cast is now folded into `fix_compact_kernel`'s first pass (the inline
// `f64::cast_from` widen), eliminating a per-leaf GPU launch. See the FOLDED WIDEN
// block in `fix_compact_kernel`.

/// DEVICE-RESIDENT build→fix→compact chain (Task 2 step 1; FOLDED to
/// 2 launches by Lever A).
///
/// Runs the resident build kernel ([`construct_leaf_hist_resident_kernel`]) into an
/// f32-atomic device buffer, then launches the on-GPU FOLDED widen+fix+compact
/// ([`fix_compact_kernel`]) which widens that f32 buffer to f64 INLINE (its first
/// pass, `f64::cast_from` — matching the host readback widening) and fixes+compacts
/// in one launch, and RETURNS the fixed+compacted f64 device `Handle` (NOT a Vec)
/// plus the buffer length. NO readback — the histogram VALUES never leave the device.
/// The standalone widen launch is GONE (3 launches → 2: construct + folded fix).
///
/// This is the resident analog of `build_leaf_histograms_resident_f32_on` +
/// host fix+compact: the whole per-leaf build→fix→compact chain runs on device. The
/// `fix_feats` carry the per-feature `(slot_off, num_bin, offset, most_freq_bin)`;
/// `sum_gradient`/`sum_hessian` are the leaf RAW (un-bumped) totals.
///
/// # Errors
/// [`ComputeError::Runtime`] on a degenerate layout; propagates the same
/// validation as [`fix_compact_f64_on`] / the resident build launcher.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_compact_resident_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    fix_feats: &[(usize, u32, i32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
    // when Some((grad, hess, num_data)), the on-device resident build
    // gathers grad/hess ON DEVICE from these once-per-grow full buffers via `leaf_rows` —
    // no host `ord_g`/`ord_h` gather + per-build upload. `gradients`/`hessians` are STILL
    // read here for the host-side overflow guard (a cheap O(rows) scan, NOT an upload).
    // None keeps the byte-identical host-gather build.
    resident_gh: Option<(cubecl::server::Handle, cubecl::server::Handle, usize)>,
) -> Result<(cubecl::server::Handle, usize), ComputeError> {
    // ---- 0. OVERFLOW GUARD ----
    // The u64 fixed-point build sums `round(v * 2^30)` (i64) across a bin's rows. The
    // worst-case single-bin magnitude is `rows * max|v| * 2^30`; it MUST fit in i64 or
    // the two's-complement add wraps to a WRONG value. Bound: i64@2^30 is safe to
    // ~1e9 rows × |g| ≤ 8. We bound `max|v|` by the actual leaf grad/hess
    // (a one-pass scan of the rows we are about to accumulate) — NOT a clamp; on
    // violation we return a typed error rather than silently overflow. The grad/hess are
    // small in practice (regression residuals / Newton steps), so this never trips on
    // sane data; it documents + enforces the contract for pathological extreme leaves.
    {
        let rows = leaf_rows.len() as f64;
        let mut max_abs = 0.0f64;
        for &r in leaf_rows {
            let i = r as usize;
            let g = gradients[i].abs() as f64;
            let h = hessians[i].abs() as f64;
            if g > max_abs {
                max_abs = g;
            }
            if h > max_abs {
                max_abs = h;
            }
        }
        // 2^30 scale; compare against i64::MAX in f64 (exact enough — both sides are
        // upper bounds and the margin to i64::MAX ≈ 9.2e18 is enormous for sane leaves).
        let worst = rows * max_abs * 1_073_741_824.0_f64; // rows * max|v| * 2^30
        if worst >= i64::MAX as f64 {
            return Err(ComputeError::Runtime {
                detail: "fixed-point histogram accumulation may overflow i64 at S=2^30 \
                         (rows x |value| x 2^30 exceeds i64::MAX)"
                    .to_string(),
            });
        }
    }

    // ---- 1. RESIDENT RAW build into a u64 fixed-point device buffer ----
    // LDS-privatized per-feature build when every feature ≤ 256 bins (naive fallback
    // otherwise) — the SAME `resident_raw_build_into` the readback launcher uses, so
    // the resident-pool chain and the host path share one accumulation structure.
    // u64 fixed-point RAW merge target (was f32). The u64 LDS build accumulates
    // `round(v*2^30)` as two's-complement i64-bits; `fix_compact_kernel` dequantizes
    // `(bits as i64)/2^30 → f64` in its widen pass. The grad/hess INPUTS (`h_g`/`h_h` in
    // `resident_raw_build_into`) STAY f32 — the kernel quantizes them in-kernel; ONLY this
    // merge target widened to u64. Same `slot_len` element COUNT, 2× bytes.
    let zeros_u64 = vec![0u64; slot_len];
    let h_raw = client.create_from_slice(u64::as_bytes(&zeros_u64));
    resident_raw_build_into(
        client,
        resident_bins,
        width,
        num_features,
        num_data,
        slot_off,
        slot_len,
        leaf_rows,
        gradients,
        hessians,
        h_raw.clone(),
        true, // u64 fixed-point build → dequantized in fix_compact_kernel
        resident_gh, // Some ⇒ on-device grad/hess gather, no host upload
    );

    // ---- 2. (Lever A) Allocate the zeroed f64 OUTPUT. The standalone
    //         widen launch is GONE — the folded fix kernel below widens each feature
    //         region from `h_raw` (f32) into `h_f64` (f64) inline as its first pass,
    //         then fixes+compacts. `fix_feats` covers EVERY feature region contiguously
    //         (the learner enumerates all features; regions tile [0, slot_len)), so the
    //         per-feature inline widen covers the whole buffer exactly as the prior
    //         full-buffer widen did. The per-leaf spine launch count drops 3 → 2
    //         (construct + folded fix). When `fix_feats` is empty (no features / no
    //         rows) there is nothing to widen and `h_f64` stays zeroed — matching the
    //         prior degenerate path (construct skipped ⇒ widen of zeros ⇒ zeros). ----
    let zeros64 = vec![0.0f64; slot_len];
    let h_f64 = client.create_from_slice(f64::as_bytes(&zeros64));

    // ---- 3. ON-GPU FOLDED widen+fix+compact over the f64 buffer (Lever A kernel) ----
    if !fix_feats.is_empty() {
        let n = fix_feats.len();
        let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
        let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
        let mut offset_a: Vec<i32> = Vec::with_capacity(n);
        let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
        for &(so, nb, off, mfb) in fix_feats {
            if nb == 0 {
                return Err(ComputeError::Runtime {
                    detail: "build_fix_compact_resident: num_bin must be > 0".to_string(),
                });
            }
            let cells = 2usize.checked_mul(nb as usize).ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {nb} overflows the histogram length"),
            })?;
            let end = so.checked_add(cells).ok_or_else(|| ComputeError::Runtime {
                detail: "build_fix_compact_resident: slot_off + region overflows".to_string(),
            })?;
            if end > slot_len {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: slot_len,
                });
            }
            slot_off_a.push(so as u32);
            num_bin_a.push(nb as i32);
            offset_a.push(off);
            mfb_a.push(mfb as i32);
        }
        let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
        let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
        let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
        let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));
        // SAFETY: `h_raw` (f32 IN) and `h_f64` (f64 OUT) are both sized `slot_len`;
        // cube `f < n` reads `h_raw` and reads/writes `h_f64` only within its validated
        // `[slot_off[f], slot_off[f]+2*num_bin[f]) <= slot_len` region (inline widen +
        // fix + compact) and `mfb < num_bin` keeps the reconstruct in range. cubecl
        // unsafe confined here.
        //
        // LAUNCH_UNCHECKED: `::launch_unchecked` drops the in-kernel per-access
        // bounds-check codegen in the fix/compact loops. ZERO numeric risk — same f64
        // deterministic kernel as `fix_compact_f64_on` (one cube per feature, ascending
        // fold, bit-exact). Host-proven accesses BEFORE upload:
        //   - `h_raw[...]` / `h_f64[...]` — cube `f < n` touches only
        //     `[slot_off[f], slot_off[f] + 2*num_bin[f])`, validated `<= slot_len` in the
        //     loop above; both buffers sized `slot_len`;
        //   - `mfb < num_bin` (the `do_fix` guard) keeps the reconstruct cell in range;
        //   - the per-feature `slot_off`/`num_bin`/`offset`/`most_freq_bin` arrays all have
        //     exactly `n` elements and `f < n`.
        // The host-side boundary checks discharge exactly the launch_unchecked obligations;
        // the launch does NOT change numerics — only bounds-check codegen is removed; the f64
        // fold order is identical (bit-exact).
        unsafe {
            fix_compact_kernel::launch_unchecked(
                client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(h_raw, slot_len),
                ArrayArg::from_raw_parts(h_f64.clone(), slot_len),
                ArrayArg::from_raw_parts(h_slot, n),
                ArrayArg::from_raw_parts(h_numbin, n),
                ArrayArg::from_raw_parts(h_offset, n),
                ArrayArg::from_raw_parts(h_mfb, n),
                sum_gradient,
                sum_hessian,
            );
        }
    }

    Ok((h_f64, slot_len))
}

/// Readback variant of [`build_fix_compact_resident_f64_on`] (Task 2
/// step 1 validation): runs the SAME device-resident build→fix→compact chain but
/// reads the f64 buffer back to a `Vec<f64>`. Used by the oracle to prove the
/// resident chain equals the host build (`build_leaf_histograms_resident_f32_on`)
/// + host `fix_histogram` + host `compact_histogram` (within the ~1e-6 f32-atomic
/// RAW-build tolerance; the fix+compact step itself is bit-exact, Task 1). Not on
/// the live path — the live wiring is deferred.
///
/// # Errors
/// Same as [`build_fix_compact_resident_f64_on`].
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_compact_resident_readback_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    fix_feats: &[(usize, u32, i32, u32)],
    sum_gradient: f64,
    sum_hessian: f64,
) -> Result<Vec<f64>, ComputeError> {
    let (handle, len) = build_fix_compact_resident_f64_on(
        client,
        resident_bins,
        width,
        num_features,
        num_data,
        slot_off,
        slot_len,
        leaf_rows,
        gradients,
        hessians,
        fix_feats,
        sum_gradient,
        sum_hessian,
        None, // readback oracle keeps the host grad/hess gather (unchanged)
    )?;
    debug_assert_eq!(len, slot_len);
    let bytes = client.read_one_unchecked(handle);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// ===========================================================================
// FUSED per-feature build + fix + compact + best-split scan kernel.
//
// ONE cube per feature (`CubeCount::Static(num_features,1,1)`, `CubeDim::new_1d(1)`)
// — single-owner ⇒ BIT-EXACT (the cpu-anchor f64 fold order), NO atomics, NO
// cross-cube barrier. Cube `f` (`CUBE_POS_X`) owns ONLY its region
// `[slot_off[f], slot_off[f] + 2*num_bin[f])`. The kernel collapses today's
// directly-built-leaf chain — construct_leaf_hist_resident_kernel(1) +
// fix_compact_kernel(1) + find_best_splits_fused_kernel(1) = 3 launches — into ONE.
//
// Stage 1 BUILD: SEQUENTIAL f64 gather→fold in ASCENDING leaf-row order (the CPU
//   anchor order ⇒ bit-exact, NOT the ~1e-6 f32-atomic path). Mirrors
//   `construct_leaf_hist_resident_kernel`'s bin layout / resident indexing
//   (histogram.rs:511-533) but sequential into THIS cube's f64 region.
// Stage 2 FIX: inlines `fix_compact_kernel`'s fix logic VERBATIM
//   (histogram.rs:674-703) — RAW (un-bumped) sum_gradient/sum_hessian seed,
//   ascending subtract via branchless `select`.
// Stage 3 COMPACT: inlines `fix_compact_kernel`'s compact logic VERBATIM
//   (histogram.rs:705-732) — offset shift + tail zero.
// Stage 4 SCAN: calls the SHARED `split_scan_body` (split.rs:144) over the
//   fixed+compacted region with the 2*kEpsilon-BUMPED sum_hessian + host
//   min_gain_shift (the SAME scan operands `find_best_splits_fused_kernel` uses),
//   writing the RAW 12-cell SplitInfo to `out[f*12..]`.
//
// Output BOTH the resident fixed+compacted f64 histogram (for the subtraction
// trick) AND the per-feature SplitInfos, in ONE launch.
//
// `#[cfg(feature="rocm")]` — the CPU anchor keeps the host build/fix/compact/scan
// unchanged (the fused gate is OFF on cpu).
// ===========================================================================

/// Fused per-feature BUILD + FIX + COMPACT + SCAN kernel. See the
/// module-level block above. Cube `f` owns its region of `hist` (f64 OUT, the
/// resident fixed+compacted histogram, caller-zeroed) and writes its 12-cell
/// window `out[f*12 .. f*12+12]` (the RAW SplitInfo cells, host-decoded).
///
/// The LEAF-LEVEL scalars are shared across every feature: the RAW (un-bumped)
/// `sum_gradient_raw` / `sum_hessian_raw` feed the FIX; the
/// 2*kEpsilon-BUMPED `sum_hessian_bumped` + the host `min_gain_shift` feed the SCAN
/// (the distinct operands, matching `find_best_splits_fused_kernel`).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_scan_fused_kernel<B: Int>(
    // Device-resident binned columns (feature-major, `f*num_data + row`) — INPUT.
    // Native bin width (u8/u16/u32).
    resident_bins: &Array<B>,
    // The leaf's row indices (subset of 0..num_data) — INPUT.
    leaf_rows: &Array<u32>,
    // The leaf's grad/hess gathered host-side in leaf_rows order — INPUT (f32).
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    // f64 fixed+compacted histogram — OUTPUT (caller zeroes it before launch).
    hist: &mut Array<f64>,
    // RAW 12-cell-per-feature SplitInfo — OUTPUT.
    out: &mut Array<f64>,
    // Per-feature params (length == num_features).
    slot_off: &Array<u32>,
    num_bin: &Array<i32>,
    offset: &Array<i32>,
    most_freq_bin: &Array<i32>,
    default_bin: &Array<i32>,
    skip_default_bin: &Array<u32>,
    rev_count: &Array<i32>,
    fwd_count: &Array<i32>,
    // 0|1 per feature — whether THIS feature is scanned (a gated-out feature still has
    // its histogram BUILT+fixed+compacted, for the subtraction trick, but is NOT
    // scanned: its 12-cell out window is left zeroed ⇒ host decodes is_splittable == 0).
    scan_active: &Array<u32>,
    // Stride of a resident column = full train row count.
    num_data_stride: usize,
    // LEAF-LEVEL scalars (shared across the batch). `sum_gradient_raw` feeds the Stage-4
    // scan; `sum_hessian_raw` is retained as a positional launch arg but is no
    // longer used to reconstruct the mfb cell (prefixed `_`).
    sum_gradient_raw: f64,
    _sum_hessian_raw: f64,
    use_l1: u32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    min_gain_shift: f64,
    sum_hessian_bumped: f64,
    num_data: i32,
) {
    let f = CUBE_POS_X;
    let fi = f as usize;
    let base = slot_off[fi] as usize;
    let nb = num_bin[fi];
    // mfb no longer read here — the Stage-2 FixHistogram overwrite was removed.
    let _mfb = most_freq_bin[fi];
    let off = offset[fi];

    // ---- Stage 1: SEQUENTIAL f64 BUILD (ascending leaf-row order = cpu anchor) ----
    // Zero this cube's region first (2 cells per bin), then gather each leaf row's
    // bin from the resident column and ASCENDING-fold f32 grad/hess into the f64
    // cells. `f64::cast_from(score_t f32)` reproduces the C++ float->double widen;
    // the ascending fold order matches the host sequential build EXACTLY
    // (the bit-exact contract is non-negotiable). NO atomics (single-owner cube).
    for w in 0..nb {
        let wbi = base + (w as usize) * 2;
        hist[wbi] = 0.0;
        hist[wbi + 1] = 0.0;
    }
    let rows = ord_g.len();
    for k in 0..rows {
        let row = leaf_rows[k] as usize;
        // native-width read widened to a u32 INDEX (value-faithful).
        let bin = u32::cast_from(resident_bins[fi * num_data_stride + row]) as usize;
        let cell = base + bin * 2;
        hist[cell] += f64::cast_from(ord_g[k]);
        hist[cell + 1] += f64::cast_from(ord_h[k]);
    }

    // ---- Stage 2: FIX (was fix_compact_kernel's Dataset::FixHistogram) ----
    // Stage 1 above scattered EVERY leaf row (mfb bin included) into `hist`, so the built
    // mfb cell is already the exact most-frequent-bin mass. The C++ `leaf_total − Σothers`
    // reconstruction exists because the C++ dense build SKIPS the mfb bin; transcribing it
    // here overwrote the correct cell with `sum_*_raw − Σ_bin-grouped hist`, sweeping the
    // row-order-vs-bin-grouped f64 fold residue into the globally-empty forced-mfb cell
    // (a phantom value). FIX: preserve the dense-built mfb cell —
    // provably 0 for a forced-empty bin at all scales; a no-op in the anchor regime
    // (integer grad/hess ⇒ sum_*_raw == Σ hist exactly).
    //
    // `sum_gradient_raw` is still consumed by the Stage-4 scan; `sum_hessian_raw` remains a
    // launch arg but is intentionally no longer used to overwrite the built mfb cell.

    // ---- Stage 3: COMPACT (fix_compact_kernel:705-732, VERBATIM) ----
    if off > 0 {
        if off >= nb {
            for c in 0..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        } else {
            let keep = nb - off;
            for c in 0..keep {
                let dst = base + (c as usize) * 2;
                let src = base + ((c + off) as usize) * 2;
                hist[dst] = hist[src];
                hist[dst + 1] = hist[src + 1];
            }
            for c in keep..nb {
                let dst = base + (c as usize) * 2;
                hist[dst] = 0.0;
                hist[dst + 1] = 0.0;
            }
        }
    }

    // ---- Stage 4: SCAN (shared split_scan_body, split.rs:144) ----
    // Build/fix/compact above ran for EVERY feature (the complete histogram the
    // subtraction trick needs). The SCAN runs only for SCAN-ACTIVE features (the spine
    // subset that passed the learner's col-sampler / parent-splittability / interaction
    // gates); a gated-out feature leaves its 12-cell out window ZEROED ⇒ the host
    // decodes is_splittable == 0 (a no-split sentinel) and never selects it. Over the
    // fixed+compacted region; hist_base = slot_off[f], out_base = f*12. The SAME leaf
    // scalars as find_best_splits_fused_kernel (split.rs:976-996): the 2*kEpsilon-BUMPED
    // sum_hessian + host min_gain_shift for the scan entry.
    if scan_active[fi] != 0 {
        crate::kernels::split::split_scan_body(
            hist,
            slot_off[fi],
            out,
            f * 12u32,
            nb,
            off,
            default_bin[fi],
            skip_default_bin[fi],
            use_l1,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            lambda_l1,
            lambda_l2,
            min_gain_shift,
            sum_gradient_raw,
            sum_hessian_bumped,
            num_data,
            rev_count[fi],
            fwd_count[fi],
        );
    }
}

/// Host launcher for the FUSED build+fix+compact+scan kernel.
///
/// Drives [`build_fix_scan_fused_kernel`] in ONE launch. `feats` is the FULL
/// per-feature list (every feature in fpos order) — build + fix + compact run for
/// EVERY feature so the returned resident histogram is COMPLETE (the subtraction
/// trick derives the larger child from it). `scan_active[fpos]` selects which
/// features are SCANNED (the spine subset that passed the learner's
/// col-sampler / parent-splittability / interaction gates); gated-out features are
/// BUILT but NOT scanned.
///
/// Returns BOTH the resident fixed+compacted f64 histogram `Handle` (kept on device)
/// AND one [`SplitInfo`] per SCAN-ACTIVE feature, in scan-active order (matching the
/// learner's `batched_feats`). Mirrors the validation + marshalling of
/// [`build_fix_compact_resident_f64_on`] AND the host pre-step + decode/accept-gate
/// of the fused split scan (split.rs:1212-1311).
///
/// The leaf RAW (un-bumped) `sum_gradient_raw` / `sum_hessian_raw` feed the FIX;
/// the launcher computes the 2*kEpsilon-BUMPED sum_hessian +
/// min_gain_shift for the scan exactly as `find_best_splits_fused_inner` does.
///
/// # Errors
/// [`ComputeError::Runtime`] / [`ComputeError::LengthMismatch`] on degenerate
/// layout (mirrors the fused split launcher's per-feature boundary checks + the leaf-level
/// `sum_hessian > 0` / `max_delta_step`/`path_smooth` default-path checks).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_scan_resident_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    // native element width of `resident_bins`.
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data_stride: usize,
    // Per-feature slot offsets ride on `feats` (`f.slot_off`); this slice is accepted
    // for signature symmetry with `build_fix_compact_resident_f64_on` and asserted.
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    // FULL per-feature list (fpos order) — build+fix+compact covers ALL of them.
    feats: &[crate::kernels::split::BatchedSplitFeature],
    // `scan_active[fpos]` — whether feature fpos is SCANNED (length == feats.len()).
    scan_active: &[bool],
    cfg: &crate::gain::GainConfig,
    sum_gradient_raw: f64,
    sum_hessian_raw: f64,
    num_data: i32,
) -> Result<(cubecl::server::Handle, usize, Vec<crate::gain::SplitInfo>), ComputeError> {
    use crate::gain::SplitInfo;

    // Empty batch / empty leaf: no launch — a zeroed resident hist + empty splits.
    let rows = leaf_rows.len();
    if feats.is_empty() || rows == 0 || num_features == 0 {
        let zeros64 = vec![0.0f64; slot_len];
        let h_f64 = client.create_from_slice(f64::as_bytes(&zeros64));
        return Ok((h_f64, slot_len, Vec::new()));
    }
    if scan_active.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: scan_active.len(),
        });
    }

    // Leaf-level default-path + sum_hessian checks (identical to the fused split
    // launcher; the scan divides cnt_factor by the bumped sum_hessian).
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_raw > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }

    // `feats` is the FULL per-feature list (fpos order), so it agrees positionally
    // with the pool's `slot_off` layout (consistency guard).
    debug_assert!(
        slot_off.len() >= feats.len()
            && feats.iter().enumerate().all(|(i, f)| slot_off[i] == f.slot_off),
        "feats[*].slot_off must agree positionally with the pool slot_off layout"
    );

    // Per-feature validation + device-array assembly (BEFORE launch).
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut scan_active_a: Vec<u32> = Vec::with_capacity(n);
    for (fpos, f) in feats.iter().enumerate() {
        if f.na_as_missing && scan_active[fpos] {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: na_as_missing not yet implemented".to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "build_fix_scan_resident: slot_off + region overflows".to_string(),
            })?;
        if end > slot_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: slot_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        mfb_a.push(f.most_freq_bin as i32);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
        scan_active_a.push(if scan_active[fpos] { 1u32 } else { 0u32 });
    }

    // LEAF-LEVEL scalars computed ONCE (the 2*kEpsilon entry bump + min_gain_shift),
    // exactly as find_best_splits_fused_inner (split.rs:1213-1223).
    let two_eps = 2.0 * f64::from(lgbm_core::types::K_EPSILON);
    let sum_hessian_bumped = sum_hessian_raw + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_raw,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    // Per-leaf uploads (leaf_rows + the leaf's gathered grad/hess) + the zeroed
    // outputs. The big resident bins matrix is ALREADY on device.
    let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
    let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
    let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
    let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let zeros64 = vec![0.0f64; slot_len];
    let h_hist = client.create_from_slice(f64::as_bytes(&zeros64));
    let out_len = n * 12;
    let out_zeros = vec![0.0f64; out_len];
    let h_out = client.create_from_slice(f64::as_bytes(&out_zeros));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));
    let h_scan = client.create_from_slice(u32::as_bytes(&scan_active_a));

    // SAFETY: cube `f < n` reads the resident column at `f*num_data_stride +
    // leaf_rows[k]` (leaf_rows ⊂ 0..num_data_stride keeps it in range; the resident
    // buffer is sized num_features*num_data_stride), and reads/writes `h_hist` only
    // within its validated `[slot_off[f], slot_off[f]+2*num_bin[f]) <= slot_len`
    // region, writing `h_out[f*12 .. f*12+12]` within the n*12 allocation. `bin <
    // num_bin` (resident invariant) keeps the build cell in range; `mfb < num_bin`
    // keeps the reconstruct in range. Every per-feature index array has exactly `n`
    // elements; every handle outlives the launch. All cubecl unsafe confined here.
    //
    // LAUNCH_UNCHECKED: `::launch_unchecked` drops the in-kernel per-access
    // bounds-check codegen in the build + fix + scan loops. ZERO numeric risk — the kernel
    // is f64 and DETERMINISTIC (one cube per feature, `CubeDim::new_1d(1)`, SEQUENTIAL
    // ascending leaf-row fold, NO atomics) so it stays the bit-exact cpu-anchor order.
    // Every device access is host-proven in range BEFORE upload:
    //   - `resident_bins[f*num_data_stride + leaf_rows[k]]` — `leaf_rows` ⊂
    //     `[0, num_data_stride)` (caller resident contract) and the resident buffer is
    //     sized `num_features*num_data_stride`, so `f < n <= num_features` keeps it in range;
    //   - `leaf_rows[k]` for `k < rows` (`h_rows` sized `rows`); `ord_g[k]`/`ord_h[k]`
    //     (`h_g`/`h_h` sized `rows`);
    //   - `hist[...]` (the f64 build/fix/compact) — cube `f < n` touches only
    //     `[slot_off[f], slot_off[f] + 2*num_bin[f])`, validated `<= slot_len` in the
    //     per-feature loop above (`h_hist` sized `slot_len`); `bin < num_bin` keeps the
    //     build cell and `mfb < num_bin` the reconstruct cell in that region;
    //   - `out[f*12 .. f*12+12]` within the `n*12` allocation (`h_out` sized `n*12`);
    //   - every per-feature param array (`slot_off`/`num_bin`/`offset`/`most_freq_bin`/
    //     `default_bin`/`skip_default_bin`/`rev_count`/`fwd_count`/`scan_active`) has
    //     exactly `n` elements and `f < n`.
    // The host-side boundary checks discharge exactly the launch_unchecked obligations;
    // the launch does NOT change numerics — only bounds-check codegen is removed; the f64
    // sequential fold / scan order is identical (bit-exact).
    //
    // PERF/BENEFIT (measured — dual-kernel single-binary interleaved A/B
    // on gfx1100, `examples/launch_unchecked_ab.rs`): this is the ONE swept kernel where
    // launch_unchecked pays off measurably — ~9–16% faster launch-bound, ~40–46% (≈1.8×)
    // faster compute-bound, sign-stable with non-overlapping p25/p75 spread. Because the
    // checked/unchecked arms are bit-identical f64, the delta is PURE bounds-check codegen.
    // It surfaces HERE (and not in the f32-atomic / resident-LDS kernels, where measurement
    // showed NULL) precisely because this kernel runs long SINGLE-UNIT SEQUENTIAL loops
    // (`CubeDim::new_1d(1)`, one cube per feature) — a per-access bounds branch compounds
    // over every leaf-row × bin iteration with nothing to hide behind, whereas the atomic
    // kernels are atomic-contention / memory-latency bound and mask it. So launch_unchecked
    // is strongly justified for this kernel on perf grounds, not merely safe.
    // Dispatch the fused kernel's `<B: Int>` monomorphization on the
    // resident buffer's native width (only the `resident_bins` ArrayArg type changes;
    // every other arg is width-independent). Exactly one match arm runs ⇒ the by-value
    // handle moves are exclusive.
    macro_rules! launch_fused {
        ($w:ty) => {
            unsafe {
                build_fix_scan_fused_kernel::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(resident_bins, num_features * num_data_stride),
                    ArrayArg::from_raw_parts(h_rows, rows),
                    ArrayArg::from_raw_parts(h_g, rows),
                    ArrayArg::from_raw_parts(h_h, rows),
                    ArrayArg::from_raw_parts(h_hist.clone(), slot_len),
                    ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot, n),
                    ArrayArg::from_raw_parts(h_numbin, n),
                    ArrayArg::from_raw_parts(h_offset, n),
                    ArrayArg::from_raw_parts(h_mfb, n),
                    ArrayArg::from_raw_parts(h_defbin, n),
                    ArrayArg::from_raw_parts(h_skip, n),
                    ArrayArg::from_raw_parts(h_rev, n),
                    ArrayArg::from_raw_parts(h_fwd, n),
                    ArrayArg::from_raw_parts(h_scan, n),
                    num_data_stride,
                    sum_gradient_raw,
                    sum_hessian_raw,
                    if use_l1 { 1u32 } else { 0u32 },
                    cfg.min_data_in_leaf,
                    cfg.min_sum_hessian_in_leaf,
                    cfg.lambda_l1,
                    cfg.lambda_l2,
                    min_gain_shift,
                    sum_hessian_bumped,
                    num_data,
                );
            }
        };
    }
    match width {
        crate::ResidentBinWidth::U8 => launch_fused!(u8),
        crate::ResidentBinWidth::U16 => launch_fused!(u16),
        crate::ResidentBinWidth::U32 => launch_fused!(u32),
    }

    // Read back ONLY the SplitInfo cells; the histogram Handle stays resident.
    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);

    // Decode with the SAME accept-gate as find_best_splits_fused_inner
    // (split.rs:1277-1311). Only SCAN-ACTIVE features have a meaningful out window
    // (gated-out features were built but not scanned ⇒ their window is zeroed ⇒
    // is_splittable == 0). Push the active features' SplitInfos in scan-active order
    // (matching the learner's `batched_feats`).
    let penalty = 1.0f64;
    let active_count = scan_active.iter().filter(|&&a| a).count();
    let mut splits = Vec::with_capacity(active_count);
    for f in 0..n {
        if !scan_active[f] {
            continue;
        }
        let dbase = f * 12;
        let is_splittable = cells[dbase] != 0.0;
        let raw_threshold = cells[dbase + 1] as u32;
        let raw_gain = cells[dbase + 2];
        let left_count = cells[dbase + 3] as i32;
        let right_count = cells[dbase + 4] as i32;
        let left_sum_gradient = cells[dbase + 5];
        let left_sum_hessian = cells[dbase + 6];
        let right_sum_gradient = cells[dbase + 7];
        let right_sum_hessian = cells[dbase + 8];
        let default_left = cells[dbase + 9] != 0.0;
        let left_output = cells[dbase + 10];
        let right_output = cells[dbase + 11];

        if is_splittable && raw_gain > f64::NEG_INFINITY {
            splits.push(SplitInfo {
                threshold: raw_threshold,
                gain: (raw_gain - min_gain_shift) * penalty,
                left_count,
                right_count,
                left_sum_gradient,
                left_sum_hessian,
                right_sum_gradient,
                right_sum_hessian,
                left_output,
                right_output,
                default_left,
            });
        } else {
            splits.push(SplitInfo::none());
        }
    }

    Ok((h_hist, slot_len, splits))
}

/// The build+fix+compact+scan portion of
/// [`build_fix_scan_resident_f64_on`] WITHOUT the per-feature-array readback+decode —
/// returns the resident histogram Handle + the raw `n*12` scan-output Handle +
/// `min_gain_shift` so the device reduce can fold the winner on device. Byte-for-byte the
/// SAME per-feature validation + marshalling + [`build_fix_scan_fused_kernel`] launch as
/// `build_fix_scan_resident_f64_on`; the ONLY difference is it stops BEFORE the
/// `read_one_unchecked(h_out)` and hands the caller `h_out` instead. Duplicated (not
/// refactored out of `build_fix_scan_resident_f64_on`) to keep that shipped function
/// byte-unchanged (additive-only); the bit-exact parity test guards against drift.
/// Assumes a NON-empty leaf (the caller handles the empty early return).
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn build_fix_scan_fused_to_raw_handle<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data_stride: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    feats: &[crate::kernels::split::BatchedSplitFeature],
    scan_active: &[bool],
    cfg: &crate::gain::GainConfig,
    sum_gradient_raw: f64,
    sum_hessian_raw: f64,
    num_data: i32,
) -> Result<
    (cubecl::server::Handle, usize, cubecl::server::Handle, usize, f64),
    ComputeError,
> {
    let rows = leaf_rows.len();
    if scan_active.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: scan_active.len(),
        });
    }
    // Leaf-level default-path + sum_hessian checks (identical to the host-readback launcher).
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: max_delta_step / path_smooth are Phase-7+ scope \
                     (only the default 0.0 path is transcribed)"
                .to_string(),
        });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_hessian_raw > 0.0) {
        return Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: sum_hessian must be > 0 (cnt_factor divides by it)"
                .to_string(),
        });
    }
    debug_assert!(
        slot_off.len() >= feats.len()
            && feats.iter().enumerate().all(|(i, f)| slot_off[i] == f.slot_off),
        "feats[*].slot_off must agree positionally with the pool slot_off layout"
    );

    // Per-feature validation + device-array assembly (BEFORE launch) — IDENTICAL.
    let n = feats.len();
    let mut slot_off_a: Vec<u32> = Vec::with_capacity(n);
    let mut num_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut offset_a: Vec<i32> = Vec::with_capacity(n);
    let mut mfb_a: Vec<i32> = Vec::with_capacity(n);
    let mut default_bin_a: Vec<i32> = Vec::with_capacity(n);
    let mut skip_default_bin_a: Vec<u32> = Vec::with_capacity(n);
    let mut rev_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut fwd_count_a: Vec<i32> = Vec::with_capacity(n);
    let mut scan_active_a: Vec<u32> = Vec::with_capacity(n);
    for (fpos, f) in feats.iter().enumerate() {
        if f.na_as_missing && scan_active[fpos] {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: na_as_missing not yet implemented".to_string(),
            });
        }
        if f.num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: num_bin must be > 0".to_string(),
            });
        }
        let cells = 2usize
            .checked_mul(f.num_bin as usize)
            .ok_or_else(|| ComputeError::Runtime {
                detail: format!("num_bin {} overflows the histogram length", f.num_bin),
            })?;
        let end = f
            .slot_off
            .checked_add(cells)
            .ok_or_else(|| ComputeError::Runtime {
                detail: "build_fix_scan_resident: slot_off + region overflows".to_string(),
            })?;
        if end > slot_len {
            return Err(ComputeError::LengthMismatch {
                expected: end,
                actual: slot_len,
            });
        }
        let num_bin_i = f.num_bin as i32;
        let rev_count = (num_bin_i - 1).max(0);
        let fwd_count = if f.run_forward {
            (num_bin_i - 1 - f.offset).max(0)
        } else {
            0
        };
        slot_off_a.push(f.slot_off as u32);
        num_bin_a.push(num_bin_i);
        offset_a.push(f.offset);
        mfb_a.push(f.most_freq_bin as i32);
        default_bin_a.push(f.default_bin as i32);
        skip_default_bin_a.push(if f.skip_default_bin { 1u32 } else { 0u32 });
        rev_count_a.push(rev_count);
        fwd_count_a.push(fwd_count);
        scan_active_a.push(if scan_active[fpos] { 1u32 } else { 0u32 });
    }

    // LEAF-LEVEL scalars ONCE (the 2*kEpsilon entry bump + min_gain_shift) — IDENTICAL.
    let two_eps = 2.0 * f64::from(lgbm_core::types::K_EPSILON);
    let sum_hessian_bumped = sum_hessian_raw + two_eps;
    let use_l1 = cfg.use_l1();
    let gain_shift = crate::gain::get_leaf_gain(
        use_l1,
        sum_gradient_raw,
        sum_hessian_bumped,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;

    let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| gradients[r as usize]).collect();
    let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hessians[r as usize]).collect();
    let h_rows = client.create_from_slice(u32::as_bytes(leaf_rows));
    let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let h_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let zeros64 = vec![0.0f64; slot_len];
    let h_hist = client.create_from_slice(f64::as_bytes(&zeros64));
    let out_len = n * 12;
    let out_zeros = vec![0.0f64; out_len];
    let h_out = client.create_from_slice(f64::as_bytes(&out_zeros));
    let h_slot = client.create_from_slice(u32::as_bytes(&slot_off_a));
    let h_numbin = client.create_from_slice(i32::as_bytes(&num_bin_a));
    let h_offset = client.create_from_slice(i32::as_bytes(&offset_a));
    let h_mfb = client.create_from_slice(i32::as_bytes(&mfb_a));
    let h_defbin = client.create_from_slice(i32::as_bytes(&default_bin_a));
    let h_skip = client.create_from_slice(u32::as_bytes(&skip_default_bin_a));
    let h_rev = client.create_from_slice(i32::as_bytes(&rev_count_a));
    let h_fwd = client.create_from_slice(i32::as_bytes(&fwd_count_a));
    let h_scan = client.create_from_slice(u32::as_bytes(&scan_active_a));

    // SAFETY: identical bounds discipline to `build_fix_scan_resident_f64_on` (every device
    // access host-proven in range by the per-feature validation loop above; `leaf_rows` ⊂
    // `[0, num_data_stride)`). All cubecl unsafe confined here.
    macro_rules! launch_fused {
        ($w:ty) => {
            unsafe {
                build_fix_scan_fused_kernel::launch_unchecked::<$w, R>(
                    client,
                    CubeCount::Static(n as u32, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(resident_bins, num_features * num_data_stride),
                    ArrayArg::from_raw_parts(h_rows, rows),
                    ArrayArg::from_raw_parts(h_g, rows),
                    ArrayArg::from_raw_parts(h_h, rows),
                    ArrayArg::from_raw_parts(h_hist.clone(), slot_len),
                    ArrayArg::from_raw_parts(h_out.clone(), out_len),
                    ArrayArg::from_raw_parts(h_slot, n),
                    ArrayArg::from_raw_parts(h_numbin, n),
                    ArrayArg::from_raw_parts(h_offset, n),
                    ArrayArg::from_raw_parts(h_mfb, n),
                    ArrayArg::from_raw_parts(h_defbin, n),
                    ArrayArg::from_raw_parts(h_skip, n),
                    ArrayArg::from_raw_parts(h_rev, n),
                    ArrayArg::from_raw_parts(h_fwd, n),
                    ArrayArg::from_raw_parts(h_scan, n),
                    num_data_stride,
                    sum_gradient_raw,
                    sum_hessian_raw,
                    if use_l1 { 1u32 } else { 0u32 },
                    cfg.min_data_in_leaf,
                    cfg.min_sum_hessian_in_leaf,
                    cfg.lambda_l1,
                    cfg.lambda_l2,
                    min_gain_shift,
                    sum_hessian_bumped,
                    num_data,
                );
            }
        };
    }
    match width {
        crate::ResidentBinWidth::U8 => launch_fused!(u8),
        crate::ResidentBinWidth::U16 => launch_fused!(u16),
        crate::ResidentBinWidth::U32 => launch_fused!(u32),
    }

    Ok((h_hist, slot_len, h_out, out_len, min_gain_shift))
}

/// f64-fused escape-hatch arm: NO-READBACK sibling of
/// [`build_fix_scan_resident_f64_on`]. Launches the SAME
/// [`build_fix_scan_fused_kernel`] (build + fix + compact + scan, all features), then —
/// INSTEAD of reading back the `n*12` per-feature array and decoding a `Vec<SplitInfo>` —
/// folds the winner DIRECTLY into `out`'s `out_leaf` slot via split.rs's
/// [`crate::kernels::split::launch_reduce_into_leaf`] (the SAME device reduce kernel the
/// single-leaf / co-pack launchers use), issuing ZERO device→host transfer of the
/// per-feature array. Returns only `(resident histogram Handle, slot_len)`.
///
/// The reduce runs over ALL `feats.len()` per-feature windows (`raw_base = 0`): a
/// scan-INACTIVE feature's 12-cell window is left zero-filled by the fused kernel
/// (`histogram.rs:~2356`, the `if scan_active[fi] != 0` guard), so it decodes as
/// `is_splittable = 0` and the reduce's accept-gate skips it — NO separate scan_active
/// mask is needed device-side. `real_feats` is the FULL fpos-order real feature index
/// list (length `feats.len()`); inactive features are skipped by the gate, and the active
/// features are consulted in ascending-fpos order (identical to
/// `build_fix_scan_resident_f64_on`'s scan-active decode order), so the `split_gt`
/// tie-break winner is bit-identical to `argmax_over_splits`/`argmax_over_resident_splits`
/// over the active features.
///
/// The existing [`build_fix_scan_resident_f64_on`] is BYTE-UNCHANGED — this is a purely
/// additive sibling reusing the SAME `build_fix_scan_fused_kernel` (never modified).
///
/// # Errors
/// As [`build_fix_scan_resident_f64_on`] (per-feature validation checks + the leaf-level
/// `sum_hessian > 0` / `max_delta_step`/`path_smooth` default-path checks); plus
/// [`ComputeError::LengthMismatch`] if `real_feats.len() != feats.len()` or
/// `out_leaf >= out.len` on a non-empty leaf.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn build_fix_scan_resident_reduce_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    resident_bins: cubecl::server::Handle,
    width: crate::ResidentBinWidth,
    num_features: usize,
    num_data_stride: usize,
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    gradients: &[f32],
    hessians: &[f32],
    feats: &[crate::kernels::split::BatchedSplitFeature],
    scan_active: &[bool],
    real_feats: &[i32],
    cfg: &crate::gain::GainConfig,
    sum_gradient_raw: f64,
    sum_hessian_raw: f64,
    num_data: i32,
    out: &crate::kernels::best_split::SplitSoa,
    out_leaf: usize,
) -> Result<(cubecl::server::Handle, usize), ComputeError> {
    if real_feats.len() != feats.len() {
        return Err(ComputeError::LengthMismatch {
            expected: feats.len(),
            actual: real_feats.len(),
        });
    }

    // Empty batch / empty leaf: no launch — a zeroed resident hist. The caller's pre-zeroed
    // frontier slot already reads as no-valid-split, so `out` is left untouched (matching
    // `build_fix_scan_resident_f64_on`'s empty-batch early return, which returns an empty
    // splits vec ⇒ the host argmax picks nothing).
    let rows = leaf_rows.len();
    if feats.is_empty() || rows == 0 || num_features == 0 {
        let zeros64 = vec![0.0f64; slot_len];
        let h_f64 = client.create_from_slice(f64::as_bytes(&zeros64));
        return Ok((h_f64, slot_len));
    }
    if !feats.is_empty() && out_leaf >= out.len {
        return Err(ComputeError::Runtime {
            detail: format!(
                "build_fix_scan_resident_reduce: out_leaf {out_leaf} out of range [0, {})",
                out.len
            ),
        });
    }

    // Reuse the EXISTING host-readback launcher's build+fix+compact+scan + the SAME
    // min_gain_shift, but we need the raw `h_out` handle (NOT the decoded splits). To keep
    // `build_fix_scan_resident_f64_on` byte-unchanged (additive-only), this sibling
    // reproduces the SAME validation + launch inline via the fused-scan helper below rather
    // than editing that function. We obtain the raw scan handle + min_gain_shift, then reduce.
    let (h_hist, len, h_out, out_len, min_gain_shift) = build_fix_scan_fused_to_raw_handle(
        client,
        resident_bins,
        width,
        num_features,
        num_data_stride,
        slot_off,
        slot_len,
        leaf_rows,
        gradients,
        hessians,
        feats,
        scan_active,
        cfg,
        sum_gradient_raw,
        sum_hessian_raw,
        num_data,
    )?;

    // Fold the winner into `out[out_leaf]` on device — ZERO per-feature-array readback. The
    // reduce runs over all `feats.len()` windows; inactive features decode `is_splittable=0`
    // and are skipped by the accept-gate.
    crate::kernels::split::launch_reduce_into_leaf(
        client,
        h_out,
        out_len,
        real_feats,
        feats.len(),
        out,
        out_leaf,
        0,
        min_gain_shift,
    );

    Ok((h_hist, len))
}

#[cfg(test)]
mod tests {
    use super::{accumulate_histogram_into, construct_histograms_cpu_native};
    use crate::error::ComputeError;

    /// The f64-fused escape-hatch arm: the NO-READBACK
    /// [`super::build_fix_scan_resident_reduce_f64_on`] folds a winner into the target
    /// `SplitSoa` slot BIT-EXACT to [`super::build_fix_scan_resident_f64_on`] (host readback)
    /// + `argmax_over_resident_splits` over ONLY the scan-active features — including a case
    /// where a scan-INACTIVE feature carries the strongest split (its zero-filled window must
    /// decode `is_splittable = 0` and NEVER win). Runs on the cpu-f64 anchor under `--features
    /// gpu` (the resident kernels compile+run on cubecl-cpu).
    #[cfg(feature = "gpu")]
    #[test]
    fn build_fix_scan_resident_reduce_matches_host_readback() {
        use super::{
            build_fix_scan_resident_f64_on, build_fix_scan_resident_reduce_f64_on,
            upload_resident_columns,
        };
        use crate::kernels::best_split::SplitSoa;
        use crate::kernels::grow_driver::argmax_over_resident_splits;
        use crate::kernels::split::BatchedSplitFeature;
        use crate::runtime::cpu_client;

        let client = cpu_client();

        // 3 features over 8 rows. f1 is the MOST separable (would win if scanned) — the
        // scan-inactive "trap". Bins are exactly f32-representable via the grad/hess below.
        let num_data = 8usize;
        let f0: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let f1: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1]; // clean 2-way separation
        let f2: Vec<u32> = vec![0, 1, 2, 0, 1, 2, 0, 1];
        let feature_bins: Vec<&[u32]> = vec![&f0, &f1, &f2];
        let num_bins = [4u32, 2, 3];
        let mut slot_off = Vec::new();
        let mut acc = 0usize;
        for &nb in &num_bins {
            slot_off.push(acc);
            acc += 2 * nb as usize;
        }
        let slot_len = acc;

        let leaf_rows: Vec<u32> = (0..num_data as u32).collect();
        // Mixed-sign grads so f1's clean split is genuinely the strongest candidate.
        let gradients: Vec<f32> = vec![-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
        let hessians: Vec<f32> = vec![1.0f32; num_data];
        let sum_g: f64 = leaf_rows.iter().map(|&r| f64::from(gradients[r as usize])).sum();
        let sum_h: f64 = leaf_rows.iter().map(|&r| f64::from(hessians[r as usize])).sum();
        let num_data_in_leaf = leaf_rows.len() as i32;

        let cfg = crate::gain::GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        };
        let feats: Vec<BatchedSplitFeature> = (0..3)
            .map(|f| BatchedSplitFeature {
                slot_off: slot_off[f],
                num_bin: num_bins[f],
                offset: 0,
                default_bin: num_bins[f],
                most_freq_bin: 0,
                skip_default_bin: false,
                na_as_missing: false,
                run_forward: true,
            })
            .collect();
        let real_feats = vec![5i32, 1, 9]; // f1's real index is 1

        // Two scan_active configs: all-active (active order == fpos order), and f1 INACTIVE
        // (the strongest split must be excluded).
        for scan_active in [vec![true, true, true], vec![true, false, true]] {
            // Host: build_fix_scan_resident_f64_on → Vec<SplitInfo> (scan-active order).
            let (_h_host, len_host, host_splits) = build_fix_scan_resident_f64_on(
                &client,
                upload_resident_columns(&client, &feature_bins),
                crate::ResidentBinWidth::U32,
                3,
                num_data,
                &slot_off,
                slot_len,
                &leaf_rows,
                &gradients,
                &hessians,
                &feats,
                &scan_active,
                &cfg,
                sum_g,
                sum_h,
                num_data_in_leaf,
            )
            .expect("host build_fix_scan_resident");

            // Restrict the argmax inputs to the scan-active features, in active order.
            let active_feats: Vec<BatchedSplitFeature> = feats
                .iter()
                .zip(scan_active.iter())
                .filter(|&(_, &a)| a)
                .map(|(f, _)| f.clone())
                .collect();
            let active_reals: Vec<i32> = real_feats
                .iter()
                .zip(scan_active.iter())
                .filter(|&(_, &a)| a)
                .map(|(&r, _)| r)
                .collect();
            assert_eq!(host_splits.len(), active_feats.len(), "active split count");
            let (host_best, host_fpos_active) =
                argmax_over_resident_splits(&host_splits, &active_feats, &active_reals);
            assert!(host_best.gain.is_finite(), "a valid winner must exist");
            let want_feat = active_reals[host_fpos_active as usize];

            // Device: build_fix_scan_resident_reduce_f64_on → SoA slot (zero readback).
            let out = SplitSoa::zeroed(&client, 1);
            let (_h_dev, len_dev) = build_fix_scan_resident_reduce_f64_on(
                &client,
                upload_resident_columns(&client, &feature_bins),
                crate::ResidentBinWidth::U32,
                3,
                num_data,
                &slot_off,
                slot_len,
                &leaf_rows,
                &gradients,
                &hessians,
                &feats,
                &scan_active,
                &real_feats,
                &cfg,
                sum_g,
                sum_h,
                num_data_in_leaf,
                &out,
                0,
            )
            .expect("device build_fix_scan_resident_reduce");
            assert_eq!(len_dev, len_host, "resident histogram length must match");

            let got = out.read_record(&client, 0);
            assert!(got.is_valid, "device slot must be a valid split");
            assert_eq!(got.gain.to_bits(), host_best.gain.to_bits(), "gain bit-exact");
            assert_eq!(got.inner_feature_index, want_feat, "winning real feature index");
            assert_eq!(got.threshold, host_best.threshold, "threshold");
            assert_eq!(got.default_left, host_best.default_left, "default_left");
            assert_eq!(
                got.left_sum_gradients.to_bits(),
                host_best.left_sum_gradient.to_bits(),
                "left_sum_gradients bit-exact"
            );
            assert_eq!(
                got.left_sum_hessians.to_bits(),
                host_best.left_sum_hessian.to_bits(),
                "left_sum_hessians bit-exact"
            );
            assert_eq!(
                got.right_sum_gradients.to_bits(),
                host_best.right_sum_gradient.to_bits(),
                "right_sum_gradients bit-exact"
            );
            assert_eq!(
                got.right_sum_hessians.to_bits(),
                host_best.right_sum_hessian.to_bits(),
                "right_sum_hessians bit-exact"
            );
            // When f1 is inactive its real index (1) must NOT be the winner.
            if !scan_active[1] {
                assert_ne!(
                    got.inner_feature_index, 1,
                    "a scan-inactive feature must never win the device reduce"
                );
            }
        }
    }

    /// Assert two f64 histograms are bit-identical, cell-by-cell (`to_bits()`).
    fn assert_bits_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "histogram lengths differ");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "cell {i} differs: {x} ({:#018x}) vs {y} ({:#018x})",
                x.to_bits(),
                y.to_bits(),
            );
        }
    }

    /// The fold-in-place accumulator produces bytes identical to the allocating
    /// path on multiple shapes — the bit-exact merge-gate invariant in unit form.
    #[test]
    fn accumulate_into_is_bit_identical_to_native() {
        // Shape A: small, a few bins, repeated bins, mixed-sign gradients.
        let binned_a: Vec<u32> = vec![0, 2, 1, 2, 0, 3, 1, 2, 3, 0];
        let grad_a: Vec<f32> = vec![0.5, -1.25, 3.0, 0.125, -2.5, 7.75, -0.0625, 1.5, -4.0, 0.25];
        let hess_a: Vec<f32> = vec![1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.125, 2.5, 1.0];
        let num_bin_a = 4u32;

        let want_a =
            construct_histograms_cpu_native(&binned_a, &grad_a, &hess_a, num_bin_a).unwrap();
        let mut got_a = vec![0.0f64; 2 * num_bin_a as usize];
        accumulate_histogram_into(&binned_a, &grad_a, &hess_a, num_bin_a, &mut got_a).unwrap();
        assert_bits_eq(&want_a, &got_a);

        // Shape B: larger, more bins, values chosen so fold ORDER matters for f64
        // accumulation (catastrophic-cancellation-sensitive magnitudes).
        let n = 257usize;
        let num_bin_b = 16u32;
        let mut binned_b = Vec::with_capacity(n);
        let mut grad_b = Vec::with_capacity(n);
        let mut hess_b = Vec::with_capacity(n);
        for i in 0..n {
            binned_b.push((i as u32) % num_bin_b);
            // Alternating large/small magnitudes to expose any reordering.
            let g = if i % 2 == 0 {
                1e7_f32 + i as f32
            } else {
                -(1e-3_f32) * i as f32
            };
            grad_b.push(g);
            hess_b.push((i as f32).mul_add(0.001, 0.5));
        }

        let want_b =
            construct_histograms_cpu_native(&binned_b, &grad_b, &hess_b, num_bin_b).unwrap();
        let mut got_b = vec![0.0f64; 2 * num_bin_b as usize];
        accumulate_histogram_into(&binned_b, &grad_b, &hess_b, num_bin_b, &mut got_b).unwrap();
        assert_bits_eq(&want_b, &got_b);
    }

    /// Folding into a pre-zeroed sub-slice of a LARGER buffer writes only its own
    /// region and matches the standalone native build bit-for-bit (the multi-feature
    /// `out` layout `build_leaf_histograms_raw` uses).
    #[test]
    fn accumulate_into_subslice_matches_native() {
        let binned: Vec<u32> = vec![0, 1, 1, 0, 2, 2, 1];
        let grad: Vec<f32> = vec![1.0, -2.0, 0.5, 4.0, -1.0, 0.25, 3.0];
        let hess: Vec<f32> = vec![0.5, 1.0, 2.0, 0.25, 1.5, 0.75, 0.125];
        let num_bin = 3u32;
        let cells = 2 * num_bin as usize;

        let want = construct_histograms_cpu_native(&binned, &grad, &hess, num_bin).unwrap();

        // A larger pre-zeroed buffer with the histogram placed at an offset.
        let off = cells; // place feature in the second slot
        let mut buf = vec![0.0f64; 3 * cells];
        accumulate_histogram_into(&binned, &grad, &hess, num_bin, &mut buf[off..off + cells])
            .unwrap();

        assert_bits_eq(&want, &buf[off..off + cells]);
        // Untouched regions stay exactly zero.
        assert!(buf[..off].iter().all(|&v| v.to_bits() == 0.0f64.to_bits()));
        assert!(buf[off + cells..].iter().all(|&v| v.to_bits() == 0.0f64.to_bits()));
    }

    /// An undersized `out` sub-slice is a typed `LengthMismatch`, NOT a panic.
    #[test]
    fn accumulate_into_rejects_short_out() {
        let binned: Vec<u32> = vec![0, 1, 2];
        let grad: Vec<f32> = vec![1.0, 2.0, 3.0];
        let hess: Vec<f32> = vec![1.0, 1.0, 1.0];
        let num_bin = 4u32; // needs 8 cells
        let mut out = vec![0.0f64; 4]; // too short

        let err =
            accumulate_histogram_into(&binned, &grad, &hess, num_bin, &mut out).unwrap_err();
        assert!(
            matches!(err, ComputeError::LengthMismatch { expected, actual }
                if expected == 8 && actual == 4),
            "expected LengthMismatch{{8,4}}, got {err:?}"
        );
    }

    /// `resolve_target_cubes` pure resolution order (a)→(b)→(c): env override used
    /// VERBATIM (>0) → queried CU × `CUBES_PER_CU` → `FALLBACK`. No env/OnceLock/GPU —
    /// exercises the resolution logic directly with explicit args (PREFERRED route (i)).
    #[cfg(feature = "rocm")]
    #[test]
    fn resolve_target_cubes_order() {
        use super::{resolve_target_cubes, CUBES_PER_CU, ROWPART_TARGET_CUBES_FALLBACK};
        // (a) env override (>0) wins verbatim, NOT multiplied by CUBES_PER_CU.
        assert_eq!(resolve_target_cubes(Some(768), Some(8)), 768);
        assert_eq!(resolve_target_cubes(Some(64), None), 64);
        // env override of 0 is ignored → falls through.
        assert_eq!(resolve_target_cubes(Some(0), Some(8)), 8 * CUBES_PER_CU);
        // (b) queried CU count → num_cu * CUBES_PER_CU (8 CUs here → 64).
        assert_eq!(resolve_target_cubes(None, Some(8)), 8 * CUBES_PER_CU);
        assert_eq!(resolve_target_cubes(None, Some(8)), 64);
        assert_eq!(resolve_target_cubes(None, Some(96)), 768); // a real gfx1100 would still get 768
        // (c) neither → documented FALLBACK (never a silent 768).
        assert_eq!(resolve_target_cubes(None, None), ROWPART_TARGET_CUBES_FALLBACK);
        assert_eq!(resolve_target_cubes(None, Some(0)), ROWPART_TARGET_CUBES_FALLBACK);
        assert_eq!(ROWPART_TARGET_CUBES_FALLBACK, 64);
    }

    /// The row-partition heuristic: 1 on small/few-feature shapes (so the
    /// build stays byte-identical to the pre-row-part kernel), a tuned P in [2, P_MAX]
    /// on large-leaf × few-feature shapes, never exceeding P_MAX (the P=32 regression
    /// guard). Pure CPU logic — no GPU. Expressed against the runtime/cached `target`
    /// (`rowpart_target_cubes()`) so it is robust regardless of host hardware. Assumes
    /// `LGBM_ROWPART_MIN` is unset (default).
    #[cfg(feature = "rocm")]
    #[test]
    fn row_partition_count_heuristic() {
        use super::{row_partition_count, rowpart_target_cubes, ROWPART_MIN_LEAF, ROWPART_P_MAX};
        // Bind the runtime/cached target once; all asserts are expressed in terms of it
        // so the test is deterministic regardless of the queried CU count.
        let target = rowpart_target_cubes();
        assert!(target >= 2, "target_cubes={target} too small to exercise the tuned path");
        // Small leaf → P=1 (covers the ≤8k-row parity-test shapes + the 12k resident gate).
        assert_eq!(row_partition_count(50, 8_000), 1);
        assert_eq!(row_partition_count(50, ROWPART_MIN_LEAF - 1), 1);
        // Degenerate / already-saturated → 1.
        assert_eq!(row_partition_count(0, 10_000_000), 1);
        assert_eq!(row_partition_count(target as usize, ROWPART_MIN_LEAF), 1);
        assert_eq!(row_partition_count(target as usize + 50, ROWPART_MIN_LEAF), 1);
        // Large leaf + few features → the clamped tuned value matches the formula.
        let nf = 50u32;
        let expected = if nf >= target { 1 } else { (target / nf).clamp(1, ROWPART_P_MAX) };
        assert_eq!(row_partition_count(nf as usize, ROWPART_MIN_LEAF), expected);
        // Very few features → clamp to P_MAX (target/1, target/2 both ≥ P_MAX once
        // target ≥ 2*P_MAX, which holds for both the 8-CU APU (64) and a gfx1100 (768)).
        if target >= 2 * ROWPART_P_MAX {
            assert_eq!(row_partition_count(1, ROWPART_MIN_LEAF), ROWPART_P_MAX);
            assert_eq!(row_partition_count(2, ROWPART_MIN_LEAF), ROWPART_P_MAX);
        }
    }

    /// Confirm the queried Compute Unit count on THIS device. The box is a Radeon 860M
    /// APU (8 CUs, gfx1152 spoofed as gfx1100), so `query_num_cu()` should report 8 and
    /// `rowpart_target_cubes()` ≈ 64 (8 × CUBES_PER_CU), NOT the phantom-96-CU 768.
    /// Soft (eprintln + >0 check) so it never blocks the gate if the FFI query is
    /// environment-flaky.
    #[cfg(feature = "rocm")]
    #[test]
    fn queried_cu_count_is_8() {
        use super::{query_num_cu, rowpart_target_cubes};
        let cu = query_num_cu();
        let target = rowpart_target_cubes();
        eprintln!("query_num_cu() = {cu:?}; rowpart_target_cubes() = {target}");
        match cu {
            Some(n) => {
                assert!(n > 0, "queried CU count must be positive, got {n}");
                if n != 8 {
                    eprintln!("NOTE: expected 8 CUs on this Radeon 860M APU, queried {n}");
                }
            }
            None => eprintln!("NOTE: query_num_cu() returned None (FFI unavailable); using fallback"),
        }
        // With no LGBM_ROWPART_TARGET_CUBES override, target must NOT be the old 768
        // phantom-96-CU value unless the device genuinely has 96 CUs.
        assert!(target > 0, "target_cubes must be positive");
        if std::env::var("LGBM_ROWPART_TARGET_CUBES").is_err() {
            assert_ne!(
                target, 768,
                "target_cubes is still 768 with no override — phantom-96-CU value not removed"
            );
        }
    }

    /// Grad-conservation — the BUILD-tuner load-bearing-generator proof.
    ///
    /// Driving the build over `BUILD_PSET` with the production [`super::FreshOutGenerator`]
    /// (via [`super::build_pset_tunable_set`]) leaves the real `out` holding EXACTLY ONE
    /// histogram: `Σ(grad cells) == feats · Σ(ord_g)` to `rel_err 0`. A `CloneInputGenerator`
    /// control arm — same PSET, same kernels — instead lets every cold-benchmark rep
    /// `fetch_add` into the REAL `out`, inflating the sum ≫1× (the accumulating-kernel
    /// hazard the manual warns about). Both arms use a UNIQUE cache namespace so each
    /// always cold-tunes (exercises the benchmark reps that expose the difference).
    #[cfg(feature = "rocm")]
    #[test]
    fn build_tuner_grad_conservation_fresh_vs_clone() {
        use super::{build_pset_tunable_set, launch_build_at, LaunchKey, BUILD_PSET, ROWPART_P_MAX};
        use crate::kernels::autotune;
        use crate::runtime::rocm_client;
        use cubecl::prelude::*;
        use cubecl::server::Handle;
        use cubecl::tune::{local_tuner, CloneInputGenerator, LocalTuner, Tunable, TunableSet};
        use std::sync::Arc;

        let rows: usize = 50_000;
        let feats: usize = 8;
        let num_data = rows;
        let num_bin: u32 = 256;
        let slot_len = feats * num_bin as usize * 2;

        // Deterministic LCG bins + mixed-sign grads (a wrong fold is visible in the sum).
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % num_bin).collect();
        let leaf_rows: Vec<u32> = (0..rows as u32).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32) - 3.0).collect();
        let ord_h: Vec<f32> = vec![1.0f32; rows];
        let slot_off: Vec<u32> = (0..=feats as u32).map(|f| f * (num_bin * 2)).collect();

        let sum_g: f64 = ord_g.iter().map(|&x| f64::from(x)).sum();
        let expected = feats as f64 * sum_g;

        let client = rocm_client();
        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        let total_grad = |h: &Handle| -> f64 {
            let bytes = rocm_client().read_one_unchecked(h.clone());
            f32::from_bytes(&bytes).iter().step_by(2).map(|&x| f64::from(x)).sum()
        };
        let fresh_out = || client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        // ---- Arm A: CloneInputGenerator control → CORRUPTED (accumulated across reps) ----
        static CLONE_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("build_test_clone");
        let out_a = fresh_out();
        {
            let handles: Vec<Handle> = vec![
                d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out_a.clone(),
            ];
            let kg = move |_: &Vec<Handle>| LaunchKey {
                bucket: autotune::size_band(rows),
                feats: feats as u32,
                bins: num_bin,
            };
            let mut set = TunableSet::<LaunchKey, Vec<Handle>, ()>::new(kg, CloneInputGenerator);
            for &p in BUILD_PSET {
                if p > ROWPART_P_MAX {
                    continue;
                }
                let c = client.clone();
                set = set.with(Tunable::new(&format!("build_P{p}"), move |inp: Vec<Handle>| {
                    launch_build_at(
                        &c, crate::ResidentBinWidth::U32, false, feats, num_data, rows, slot_len, &inp, p,
                    );
                    Ok::<(), String>(())
                }));
            }
            CLONE_TUNER.execute(&format!("test_clone_{uniq}"), &client, Arc::new(set), handles);
        }
        let tg_a = total_grad(&out_a);

        // ---- Arm B: FreshOutGenerator (the production builder) → CORRECT (touched once) ----
        static FRESH_TUNER: LocalTuner<LaunchKey, String> = local_tuner!("build_test_fresh");
        let out_b = fresh_out();
        {
            let handles: Vec<Handle> = vec![
                d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out_b.clone(),
            ];
            let set = build_pset_tunable_set(
                client.clone(), crate::ResidentBinWidth::U32, false, feats, num_data, rows, num_bin, slot_len,
            );
            FRESH_TUNER.execute(&format!("test_fresh_{uniq}"), &client, Arc::new(set), handles);
        }
        let tg_b = total_grad(&out_b);
        let rel_err_b = (tg_b - expected).abs() / expected.abs().max(1.0);
        let ratio_a = tg_a / expected;
        eprintln!(
            "build_tuner grad-conservation: expected={expected:.1} cloneΣ={tg_a:.1} ({ratio_a:.2}×) \
             freshΣ={tg_b:.1} (rel_err {rel_err_b:.2e})"
        );
        assert!(
            rel_err_b < 1e-4,
            "FreshOutGenerator arm must conserve grad (real `out` touched once): rel_err {rel_err_b:.2e}"
        );
        assert!(
            ratio_a > 1.5,
            "CloneInputGenerator control must inflate ≫1× (got {ratio_a:.2}×) — proves the \
             fresh-output generator choice is load-bearing"
        );
    }

    /// The u64 fixed-point build is an order-independent integer additive merge, so every
    /// `P` in `BUILD_PSET` yields a BIT-IDENTICAL `out` — `P` is parity-neutral on the live
    /// fixed-point resident path (anchored to the CPU f64 reference). The f32 path
    /// is NOT bit-identical across P (~2e-5, inside the ~1e-6 best-effort gate),
    /// so only the u64 path is asserted bit-equal here.
    #[cfg(feature = "rocm")]
    #[test]
    fn build_tuner_u64_bit_identical_across_p() {
        use super::launch_build_at;
        use crate::runtime::rocm_client;
        use cubecl::prelude::*;
        use cubecl::server::Handle;

        let rows = 40_000usize;
        let feats = 8usize;
        let num_data = rows;
        let num_bin = 256u32;
        let slot_len = feats * num_bin as usize * 2;
        let mut s: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let bins: Vec<u32> = (0..feats * num_data).map(|_| next() % num_bin).collect();
        let leaf_rows: Vec<u32> = (0..rows as u32).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| ((i % 7) as f32) - 3.0).collect();
        let ord_h: Vec<f32> = vec![1.0f32; rows];
        let slot_off: Vec<u32> = (0..=feats as u32).map(|f| f * (num_bin * 2)).collect();
        let client = rocm_client();
        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));
        let run_at = |p: u32| -> Vec<u8> {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let inputs: Vec<Handle> = vec![
                d_bins.clone(), d_rows.clone(), d_g.clone(), d_h.clone(), d_slot.clone(), out.clone(),
            ];
            launch_build_at(
                &client, crate::ResidentBinWidth::U32, true, feats, num_data, rows, slot_len, &inputs, p,
            );
            rocm_client().read_one_unchecked(out).to_vec()
        };
        let p1 = run_at(1);
        for &p in &[4u32, 8, 16] {
            let pp = run_at(p);
            assert_eq!(
                p1, pp,
                "u64 fixed-point build differs between P=1 and P={p} — must be parity-neutral"
            );
        }
    }
}
