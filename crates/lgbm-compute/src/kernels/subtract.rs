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
/// default f64 path). Only unit 0 runs the fold (CubeDim 1).
#[cube(launch)]
pub fn subtract_hist_kernel(parent: &Array<f64>, child: &Array<f64>, out: &mut Array<f64>) {
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
