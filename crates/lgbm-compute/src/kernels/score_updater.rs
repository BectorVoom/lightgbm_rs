//! On-device score updater — the resident `double` score-buffer constant ops.
//!
//! ## What lives here
//! The Rust `#[cube]` port of the `AddScoreConstantKernel` /
//! `MultiplyScoreConstantKernel` — two whole-array scalar ops over the resident
//! `double* cuda_score_` accumulator, applied to one tree's class-major slice
//! (`offset = num_data * tree_id`):
//! - `AddScoreConstant`:      `score[offset + i] += val`  (init-score / no-split
//!   single-leaf tree — the host `ScoreUpdater::add_constant`, score_updater.rs:82).
//! - `MultiplyScoreConstant`: `score[offset + i] *= val`  (shrinkage / DART rescale
//!   / RF running-average — the host `ScoreUpdater::multiply_score`,
//!   score_updater.rs:98).
//!
//! The per-leaf training-path `AddScore` does NOT get a fresh tree-walk kernel here:
//! it delegates to
//! [`crate::kernels::predict::add_prediction_to_score_on_device`].
//!
//! ## Kernel shape
//! Each op is a trivial elementwise `#[cube]` body over an `Array<f64>`, bounds-
//! guarded `i < num_data`, wrapped in a `#[cube(launch_unchecked)]` kernel and a
//! host launcher following the `objective_regression.rs::convert_output_on`
//! skeleton (exact-size `create_from_slice` → bounds-guarded `launch_unchecked` →
//! `read_one_unchecked`), with `unsafe` confined to the launch site. The `f64`
//! score buffer is **reference-blessed** (the host `ScoreUpdater::score_` is
//! `std::vector<double>`, score_updater.rs:31) — this is a whole-array scalar op,
//! not a per-row grow/build hot loop.
//!
//! ## Anchor discipline
//! The host `lgbm_boosting::score_updater::ScoreUpdater::{add_constant,
//! multiply_score}` f64 accumulation is the parity oracle. On the default cubecl-cpu
//! f64 anchor the device op is **bit-exact** to it; the ROCm f32-storage mirror is
//! held to the ~1e-6 envelope — never compared GPU-vs-GPU. Additive and OFF by
//! default behind the `LGBM_CUDA_ON_DEVICE` seam (the boosting-layer toggle keys off
//! [`crate::cuda_on_device_enabled`]).

use cubecl::prelude::*;

use crate::error::ComputeError;

// =========================================================================
// §11 constant-op kernels — `score[offset + i] {+=,*=} val` over a resident
// `Array<f64>`, bounds-guarded `i < num_data`.
// =========================================================================

/// `AddScoreConstantKernel` body: `score[offset + i] += val` for `i < num_data`.
/// Mirrors the host `ScoreUpdater::add_constant` (score_updater.rs:82-88).
#[cube]
fn add_constant_body(score: &mut Array<f64>, val: f64, offset: u32, num_data: u32) {
    let i = ABSOLUTE_POS;
    if i < num_data as usize {
        score[offset as usize + i] += val;
    }
}

/// `MultiplyScoreConstantKernel` body: `score[offset + i] *= val` for `i < num_data`.
/// Mirrors the host `ScoreUpdater::multiply_score` (score_updater.rs:98-104).
#[cube]
fn multiply_constant_body(score: &mut Array<f64>, val: f64, offset: u32, num_data: u32) {
    let i = ABSOLUTE_POS;
    if i < num_data as usize {
        score[offset as usize + i] *= val;
    }
}

/// f64 cpu-anchor `AddScoreConstant` wrapper (the deterministic f64 reference).
/// The ROCm f32-storage mirror would instantiate the same body over the
/// resident buffer.
#[cube(launch_unchecked)]
fn add_score_constant_kernel_f64(score: &mut Array<f64>, val: f64, offset: u32, num_data: u32) {
    add_constant_body(score, val, offset, num_data);
}

/// f64 cpu-anchor `MultiplyScoreConstant` wrapper.
#[cube(launch_unchecked)]
fn multiply_score_constant_kernel_f64(
    score: &mut Array<f64>,
    val: f64,
    offset: u32,
    num_data: u32,
) {
    multiply_constant_body(score, val, offset, num_data);
}

/// Which resident constant op to launch (add vs multiply).
#[derive(Clone, Copy)]
enum ConstantOp {
    Add,
    Multiply,
}

/// The class-major slice offset for `tree_id` (host `ScoreUpdater::offset`,
/// score_updater.rs:64): `offset = num_data * tree_id`, computed overflow-safe in
/// `usize`. Validates `num_data >= 0`, `tree_id >= 0`, and that the whole slice
/// `[offset, offset + num_data)` fits inside the supplied `score` buffer BEFORE any
/// device alloc / launch.
fn checked_slice(score_len: usize, num_data: i32, tree_id: i32) -> Result<usize, ComputeError> {
    if num_data < 0 {
        return Err(ComputeError::Runtime {
            detail: format!("num_data must be >= 0, got {num_data}"),
        });
    }
    if tree_id < 0 {
        return Err(ComputeError::Runtime {
            detail: format!("tree_id must be >= 0, got {tree_id}"),
        });
    }
    let nd = num_data as usize;
    let offset = nd.checked_mul(tree_id as usize).ok_or_else(|| ComputeError::Runtime {
        detail: format!("offset overflow: num_data({num_data}) * tree_id({tree_id})"),
    })?;
    let end = offset.checked_add(nd).ok_or_else(|| ComputeError::Runtime {
        detail: format!("offset+num_data overflow: {offset} + {nd}"),
    })?;
    if end > score_len {
        return Err(ComputeError::LengthMismatch { expected: end, actual: score_len });
    }
    Ok(offset)
}

/// Shared launcher for the two §11 constant ops. Uploads the WHOLE resident score
/// buffer, applies the scalar op to the class-major slice `[offset, offset +
/// num_data)` on device, and reads the whole buffer back (the host-mirror shape —
/// the boosting toggle keeps this resident across iterations, mirroring back only
/// when a non-resident consumer reads).
fn apply_constant_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    op: ConstantOp,
    score: &[f64],
    num_data: i32,
    tree_id: i32,
    val: f64,
) -> Result<Vec<f64>, ComputeError> {
    let offset = checked_slice(score.len(), num_data, tree_id)?;
    // Empty class slice (num_data == 0) or empty buffer: nothing to launch.
    if num_data == 0 || score.is_empty() {
        return Ok(score.to_vec());
    }

    let n = score.len();
    let h_score = client.create_from_slice(f64::as_bytes(score));

    let cube_dim = 256u32;
    let cube_count = (num_data as u32).div_ceil(cube_dim);
    let offset_u32 = offset as u32;
    let nd_u32 = num_data as u32;

    // SAFETY: `h_score` is sized exactly `n = score.len()` f64 cells and outlives the
    // launch. `checked_slice` has proven `offset + num_data <= n`, and the kernel
    // bounds-guards `i < num_data`, so every `score[offset + i]` access lies in
    // `[offset, offset + num_data) ⊆ [0, n)`. cubecl unsafe is confined here.
    unsafe {
        match op {
            ConstantOp::Add => add_score_constant_kernel_f64::launch_unchecked(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(h_score.clone(), n),
                val,
                offset_u32,
                nd_u32,
            ),
            ConstantOp::Multiply => multiply_score_constant_kernel_f64::launch_unchecked(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(h_score.clone(), n),
                val,
                offset_u32,
                nd_u32,
            ),
        }
    }

    let bytes = client.read_one_unchecked(h_score);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// `AddScoreConstant` (§11): add the scalar `val` to every row of class `tree_id`
/// in the resident `double` score buffer (`score[offset + i] += val`,
/// `offset = num_data * tree_id`). Returns the whole updated buffer.
///
/// The device analog of the host `ScoreUpdater::add_constant` (score_updater.rs:82)
/// — bit-exact to it on the cpu f64 anchor; ~1e-6 on ROCm (never GPU-vs-GPU).
///
/// # Errors
/// [`ComputeError::Runtime`] if `num_data < 0` / `tree_id < 0` / the `offset`
/// arithmetic overflows; [`ComputeError::LengthMismatch`] if `[offset, offset +
/// num_data)` does not fit inside `score`.
pub fn add_score_constant_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    score: &[f64],
    num_data: i32,
    tree_id: i32,
    val: f64,
) -> Result<Vec<f64>, ComputeError> {
    apply_constant_on(client, ConstantOp::Add, score, num_data, tree_id, val)
}

/// `MultiplyScoreConstant` (§11): multiply every row of class `tree_id` in the
/// resident `double` score buffer by the scalar `val` (`score[offset + i] *= val`,
/// `offset = num_data * tree_id`). Returns the whole updated buffer.
///
/// The device analog of the host `ScoreUpdater::multiply_score` (score_updater.rs:98)
/// — bit-exact to it on the cpu f64 anchor; ~1e-6 on ROCm (never GPU-vs-GPU).
///
/// # Errors
/// [`ComputeError::Runtime`] if `num_data < 0` / `tree_id < 0` / the `offset`
/// arithmetic overflows; [`ComputeError::LengthMismatch`] if `[offset, offset +
/// num_data)` does not fit inside `score`.
pub fn multiply_score_constant_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    score: &[f64],
    num_data: i32,
    tree_id: i32,
    val: f64,
) -> Result<Vec<f64>, ComputeError> {
    apply_constant_on(client, ConstantOp::Multiply, score, num_data, tree_id, val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    /// Add-constant on a non-root tree_id updates ONLY that class slice, bit-exact
    /// to the host reference `score[offset + i] += val`.
    #[test]
    fn add_constant_matches_host_at_nonroot_tree() {
        let client = cpu_client();
        // 2 classes, 3 rows each: class 0 = [1,2,3], class 1 = [4,5,6].
        let score = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let got = add_score_constant_on(&client, &score, 3, 1, 0.5).expect("add");
        // Only class 1 (offset 3) changes.
        let want = vec![1.0, 2.0, 3.0, 4.5, 5.5, 6.5];
        assert_eq!(got, want);
    }

    /// Multiply-constant on a non-root tree_id updates ONLY that class slice.
    #[test]
    fn multiply_constant_matches_host_at_nonroot_tree() {
        let client = cpu_client();
        let score = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let got = multiply_score_constant_on(&client, &score, 3, 1, 2.0).expect("mul");
        let want = vec![1.0, 2.0, 3.0, 8.0, 10.0, 12.0];
        assert_eq!(got, want);
    }

    /// num_data == 0 is an inert no-op (empty class slice); the buffer round-trips.
    #[test]
    fn zero_num_data_is_noop() {
        let client = cpu_client();
        let score = vec![1.0, 2.0];
        let got = add_score_constant_on(&client, &score, 0, 0, 9.0).expect("noop");
        assert_eq!(got, score);
    }

    /// A negative num_data / tree_id is rejected before any launch.
    #[test]
    fn negative_inputs_rejected() {
        let client = cpu_client();
        let score = vec![1.0, 2.0, 3.0];
        assert!(add_score_constant_on(&client, &score, -1, 0, 1.0).is_err());
        assert!(add_score_constant_on(&client, &score, 3, -1, 1.0).is_err());
    }

    /// A slice `[offset, offset + num_data)` that overruns the buffer is rejected.
    #[test]
    fn out_of_range_slice_rejected() {
        let client = cpu_client();
        let score = vec![1.0, 2.0, 3.0];
        // tree_id 2 → offset 4, past the 3-element buffer.
        let e = multiply_score_constant_on(&client, &score, 2, 2, 1.0);
        assert!(matches!(e, Err(ComputeError::LengthMismatch { .. })));
    }
}
