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
//! single-owner (`CubeDim::new_1d(1)`) to keep the cpu launch shape consistent
//! with the other kernels (the CMP-04 gate selects `ReducePath::Sequential` on
//! cpu). There is no reduction non-determinism here.

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;

/// The element-wise `parent - child` fold (FeatureHistogram::Subtract, the
/// default f64 path — the cpu anchor). Only unit 0 runs the fold (CubeDim 1).
#[cube(launch)]
pub fn subtract_hist_kernel(parent: &Array<f64>, child: &Array<f64>, out: &mut Array<f64>) {
    if UNIT_POS == 0 {
        for i in 0..parent.len() {
            out[i] = parent[i] - child[i];
        }
    }
}

/// The f32-cell mirror of [`subtract_hist_kernel`] for the no-f64 hip device
/// (CMP-04). IDENTICAL element-wise `parent - child` structure — the ONLY
/// difference is the cell type (`f32` vs `f64`), since hip cannot allocate f64
/// (RESEARCH Pitfall 2/3). The capability gate (`has_f64 == false`) routes the
/// hip launch here; cpu keeps the f64 kernel.
#[cube(launch)]
pub fn subtract_hist_kernel_f32(parent: &Array<f32>, child: &Array<f32>, out: &mut Array<f32>) {
    if UNIT_POS == 0 {
        for i in 0..parent.len() {
            out[i] = parent[i] - child[i];
        }
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
    // reads/writes only indices `0..n`. All cubecl unsafe is confined here (CMP-01).
    unsafe {
        subtract_hist_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
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
    // launch; the kernel touches only `0..n`. cubecl `unsafe` confined here (CMP-01).
    unsafe {
        subtract_hist_kernel_f32::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
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
}
