//! `subtract_histograms` cube kernel — the histogram-subtraction MATH (A3).
//!
//! VERBATIM transcription of `FeatureHistogram::Subtract`
//! (`LightGBM/src/treelearner/feature_histogram.hpp:99-145`, commit 195c26fc,
//! VERSION 4.6.0.99), the default (`USE_DIST_GRAD = false`) f64 path:
//!
//! ```cpp
//! for (int i = 0; i < (meta_->num_bin - meta_->offset) * 2; ++i) {
//!   data_[i] -= other.data_[i];
//! }
//! ```
//!
//! i.e. element-wise `derived[i] = parent[i] - child[i]` over the stride-2
//! `[g0,h0,g1,h1,…]` f64 cells (both the gradient AND hessian cells). This is the
//! histogram-subtraction trick's MATH; WHICH child is subtracted from the parent
//! (the smaller sibling) is Phase-5 learner orchestration — RESEARCH open
//! question A3 resolves the subtract OP itself as in-scope at the kernel layer
//! (it is histogram-layer math).
//!
//! ## Determinism / launch shape
//! The op is element-wise (each output cell is independent), so it is launched
//! across a real workgroup as a 1D grid-stride loop (`CubeDim::new_1d(256)` ×
//! `CubeCount::Static(64,1,1)`): each thread owns disjoint indices
//! `{ABSOLUTE_POS, ABSOLUTE_POS+stride, …}`, `stride = CUBE_COUNT_X * CUBE_DIM_X`.
//! This is BIT-EXACT to the prior single-thread serial loop — no atomics, no
//! reduction, no ordering, no contention — so the byte-identical result holds on
//! cubecl-cpu AND cubecl-hip (CMP-04). There is no reduction non-determinism here.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// The element-wise `parent - child` fold (FeatureHistogram::Subtract, the
/// default f64 path — the cpu anchor). 1D grid-stride loop: each thread owns the
/// disjoint indices `{ABSOLUTE_POS, ABSOLUTE_POS+stride, …}` with
/// `stride = CUBE_COUNT_X * CUBE_DIM_X`. BIT-EXACT-by-construction vs the serial
/// loop — every `out[i]` is an independent f64 subtract with no atomics, no
/// reduction, and no cross-thread/cross-element ordering, so disjoint-thread
/// execution yields the byte-identical result on any backend (CMP-04 cpu/hip
/// parity holds). The `while i < parent.len()` bound guards every write, so the
/// grid may over-cover any `len` (the launch over-provisions lanes).
#[cube(launch)]
pub fn subtract_hist_kernel(parent: &Array<f64>, child: &Array<f64>, out: &mut Array<f64>) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    let n = parent.len() as usize;
    while i < n {
        out[i] = parent[i] - child[i];
        i += stride;
    }
}

/// The f32-cell mirror of [`subtract_hist_kernel`] for the no-f64 hip device
/// (CMP-04). IDENTICAL 1D grid-stride structure — the ONLY difference is the cell
/// type (`f32` vs `f64`), since hip cannot allocate f64 (RESEARCH Pitfall 2/3).
/// Same bit-exact-by-construction property: independent f32 subtract per cell, no
/// atomics/reduction/ordering, `while i < parent.len()` bounds every write. The
/// capability gate (`has_f64 == false`) routes the hip launch here; cpu keeps the
/// f64 kernel.
#[cube(launch)]
pub fn subtract_hist_kernel_f32(parent: &Array<f32>, child: &Array<f32>, out: &mut Array<f32>) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    let n = parent.len() as usize;
    while i < n {
        out[i] = parent[i] - child[i];
        i += stride;
    }
}

/// Host-side `subtract_histograms` on the cpu reference runtime.
///
/// Computes `derived[i] = parent[i] - child[i]` over the `2 * num_bin` f64 cells
/// (FeatureHistogram::Subtract). Validates equal lengths (V5, threat T-04-01)
/// before the unsafe launch.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_cpu(
    client: &cubecl::prelude::ComputeClient<ActiveRuntime>,
    parent: &[f64],
    child: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    subtract_histograms_f64_on(client, parent, child)
}

/// The f64 `subtract_histograms` cube path, **generic over the runtime** `R` so it
/// runs on the cubecl-cpu anchor (via [`subtract_histograms_cpu`]) AND on
/// cubecl-hip (the GPU `RocmBackend`) — the same f64 element-wise kernel, bit-exact
/// across both.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_f64_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    parent: &[f64],
    child: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    // --- V5 boundary validation (T-04-01): equal `2*num_bin` lengths ---
    if parent.len() != child.len() {
        return Err(ComputeError::LengthMismatch {
            expected: parent.len(),
            actual: child.len(),
        });
    }

    let n = parent.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let h_parent = client.create_from_slice(f64::as_bytes(parent));
    let h_child = client.create_from_slice(f64::as_bytes(child));
    let zeros = vec![0.0f64; n];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

    // SAFETY: `h_parent`/`h_child`/`h_out` were each allocated for exactly `n`
    // f64 elements (validated equal above) and outlive the launch; the kernel
    // reads/writes only indices `0..n` (the grid over-covers but the
    // `while i < parent.len()` bound guards every write). All cubecl unsafe is
    // confined here (CMP-01).
    unsafe {
        subtract_hist_kernel::launch(
            client,
            CubeCount::Static(64, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(h_parent, n),
            ArrayArg::from_raw_parts(h_child, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// **Native** host f64 element-wise `parent - child` — the production cpu-anchor
/// path (R2). Bit-IDENTICAL to [`subtract_histograms_cpu`] (the single-unit
/// `subtract_hist_kernel`): the same `derived[i] = parent[i] - child[i]` over the
/// `2*num_bin` cells and the same V5 length check, without the cubecl launch.
/// The cubecl path is retained for the kernel-parity / ROCm-mirror tests.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_cpu_native(
    parent: &[f64],
    child: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    if parent.len() != child.len() {
        return Err(ComputeError::LengthMismatch {
            expected: parent.len(),
            actual: child.len(),
        });
    }
    Ok(parent.iter().zip(child).map(|(p, c)| p - c).collect())
}

/// Handle-in/Handle-out resident subtract (260608-p90 Task 2A) — the device-resident
/// sibling of [`subtract_histograms_f64_on`]. Derives `out[i] = parent[i] - child[i]`
/// over `len` stride-2 f64 cells entirely on device: it CONSUMES the `parent` and
/// `child` device Handles, allocates a fresh `out` Handle of `len` f64 cells, launches
/// the VERBATIM element-wise [`subtract_hist_kernel`] (the same EXACT-math kernel the
/// host path uses — no new math), and RETURNS the `out` Handle. NO read-back — the
/// derived larger child's histogram never leaves the device. The caller guarantees
/// both input Handles describe `len` f64 cells (the pool's `slot_len`).
///
/// # Errors
/// [`ComputeError::Runtime`] if `len == 0` (degenerate — no cells to subtract).
#[cfg(feature = "rocm")]
pub fn subtract_histograms_f64_from_handles_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    parent: cubecl::server::Handle,
    child: cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, ComputeError> {
    if len == 0 {
        return Err(ComputeError::Runtime {
            detail: "subtract_histograms_from_handles: len must be > 0".to_string(),
        });
    }
    let zeros = vec![0.0f64; len];
    let h_out = client.create_from_slice(f64::as_bytes(&zeros));

    // SAFETY: `parent`/`child`/`h_out` each describe exactly `len` f64 cells (the
    // caller guarantees the inputs; `h_out` is allocated for `len` here) and outlive
    // the launch; the kernel reads/writes only indices `0..len` (the grid over-covers
    // but the `while i < parent.len()` bound guards every write). All cubecl unsafe is
    // confined here (CMP-01).
    unsafe {
        subtract_hist_kernel::launch(
            client,
            CubeCount::Static(64, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(parent, len),
            ArrayArg::from_raw_parts(child, len),
            ArrayArg::from_raw_parts(h_out.clone(), len),
        );
    }

    Ok(h_out)
}

/// Host-side `subtract_histograms` in **f32 cells** on ANY runtime (the no-f64
/// hip path; CMP-03/CMP-04). Same `derived[i] = parent[i] - child[i]` math and
/// V5 validation as [`subtract_histograms_cpu`], but in f32 cells. Generic over
/// `R: Runtime` so it runs on the cubecl-cpu client (f32 reference) AND the
/// cubecl-hip client (the real GPU).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()`.
pub fn subtract_histograms_f32_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    parent: &[f32],
    child: &[f32],
) -> Result<Vec<f32>, ComputeError> {
    if parent.len() != child.len() {
        return Err(ComputeError::LengthMismatch {
            expected: parent.len(),
            actual: child.len(),
        });
    }
    let n = parent.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let h_parent = client.create_from_slice(f32::as_bytes(parent));
    let h_child = client.create_from_slice(f32::as_bytes(child));
    let zeros = vec![0.0f32; n];
    let h_out = client.create_from_slice(f32::as_bytes(&zeros));

    // SAFETY: identical handle/length correspondence to `subtract_histograms_cpu`
    // — three handles each sized `n` f32 cells (validated equal), outliving the
    // launch; the kernel touches only `0..n` (the grid over-covers but the
    // `while i < parent.len()` bound guards every write). cubecl `unsafe` confined
    // here (CMP-01).
    unsafe {
        subtract_hist_kernel_f32::launch(
            client,
            CubeCount::Static(64, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(h_parent, n),
            ArrayArg::from_raw_parts(h_child, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    Ok(f32::from_bytes(&bytes).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    #[test]
    fn subtract_elementwise() {
        let client = cpu_client();
        let parent = vec![10.0f64, 5.0, 8.0, 4.0];
        let child = vec![3.0f64, 2.0, 1.0, 1.0];
        let got = subtract_histograms_cpu(&client, &parent, &child).unwrap();
        assert_eq!(got, vec![7.0, 3.0, 7.0, 3.0]);
    }

    #[test]
    fn subtract_length_mismatch() {
        let client = cpu_client();
        let err = subtract_histograms_cpu(&client, &[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, ComputeError::LengthMismatch { .. }));
    }

    #[test]
    fn subtract_empty_ok() {
        let client = cpu_client();
        let got = subtract_histograms_cpu(&client, &[], &[]).unwrap();
        assert!(got.is_empty());
    }

    /// The grid-stride parallel f64 kernel must be `to_bits()`-identical, cell by
    /// cell, to a plain serial Rust `parent[i] - child[i]` reference — on a
    /// representative length (25600 = 50 feat × 256 bins × 2) AND an odd length
    /// (12345) that exercises the stride remainder (not a multiple of the grid).
    #[test]
    fn subtract_parallel_equals_serial_f64() {
        let client = cpu_client();
        for &len in &[25600usize, 12345usize] {
            // Deterministic, varied (incl. negatives + fractions) inputs.
            let parent: Vec<f64> = (0..len)
                .map(|i| (i as f64) * 0.5 - 1234.5 + (i as f64).sin())
                .collect();
            let child: Vec<f64> = (0..len)
                .map(|i| (i as f64) * 0.25 + 7.0 - (i as f64).cos())
                .collect();

            let serial: Vec<f64> = parent.iter().zip(&child).map(|(p, c)| p - c).collect();
            let parallel = subtract_histograms_f64_on(&client, &parent, &child).unwrap();

            assert_eq!(parallel.len(), len, "len {len}: output length");
            for i in 0..len {
                assert_eq!(
                    parallel[i].to_bits(),
                    serial[i].to_bits(),
                    "len {len}, cell {i}: parallel {} vs serial {} not byte-identical",
                    parallel[i],
                    serial[i]
                );
            }
        }
    }

    /// f32 mirror of [`subtract_parallel_equals_serial_f64`]: the grid-stride f32
    /// kernel is `to_bits()`-identical to a plain serial f32 `parent - child` on
    /// the representative (25600) and stride-remainder (12345) lengths.
    #[test]
    fn subtract_parallel_equals_serial_f32() {
        let client = cpu_client();
        for &len in &[25600usize, 12345usize] {
            let parent: Vec<f32> = (0..len)
                .map(|i| (i as f32) * 0.5 - 1234.5 + (i as f32).sin())
                .collect();
            let child: Vec<f32> = (0..len)
                .map(|i| (i as f32) * 0.25 + 7.0 - (i as f32).cos())
                .collect();

            let serial: Vec<f32> = parent.iter().zip(&child).map(|(p, c)| p - c).collect();
            let parallel = subtract_histograms_f32_on(&client, &parent, &child).unwrap();

            assert_eq!(parallel.len(), len, "len {len}: output length");
            for i in 0..len {
                assert_eq!(
                    parallel[i].to_bits(),
                    serial[i].to_bits(),
                    "len {len}, cell {i}: parallel {} vs serial {} not byte-identical",
                    parallel[i],
                    serial[i]
                );
            }
        }
    }
}
