//! On-device multiclass objective — softmax grad/hess + per-row softmax (§5.3, ODL-07).
//!
//! Owning phase: **19** (ODL-07). Filled by **19-03**.
//!
//! ## What lives here
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_multiclass_objective.cu`
//! (§5.3 of `docs/cuda-kernel-design.md`) — the class-major
//! `GetGradientsKernel_MulticlassSoftmax<USE_WEIGHT>` grad/hess and the
//! `ConvertOutputCUDAKernel_MulticlassSoftmax` per-row softmax-probability
//! inverse-link. Multiclass-OVA reuses the binary per-class response math (§5.2).
//!
//! ## Kernel shape (Pattern 3 / D-07)
//! The softmax grad/hess body is ONE generic `#[cube]` fn over `F: Float`, ONE
//! thread per ROW `i` (`ABSOLUTE_POS < num_data`), looping the `num_class` classes
//! with the LOAD-BEARING class-major stride `[num_data*k + i]` (Pitfall 3): gather
//! `rec[k] = score[num_data*k+i]`, softmax (max-subtraction, D-09 pre-allocated
//! per-row scratch buffer), then write `grad[num_data*k+i]`/`hess[num_data*k+i]`.
//! The cpu **f64** anchor wrapper is the deterministic reference (D-10); the per-row
//! math runs in `F` and the GPU/hip mirror instantiates the SAME body with `F = f32`
//! (D-07). The launcher casts the f64 grad/hess read-back to `f32` (`score_t`),
//! reproducing the golden's `f64-compute → f32-cast` order bit-for-bit — the f64
//! `exp` ~1-ULP residual is absorbed by the f32 cast (D-01), so the result is
//! BIT-EXACT vs both the host anchor and the class-major real `multiclass_gh` golden.
//!
//! ## Softmax buffer (D-09)
//! The per-row length-`K` softmax scratch (`double* cuda_softmax_buffer`) is a single
//! `client.empty` device handle of `num_data*num_class` f64 cells, pre-allocated ONCE
//! before the launch (in the Phase-21 growth loop it would be hoisted and reused
//! across iterations). The kernel writes `exp(rec[k]-wmax)` into it then divides by
//! `wsum` in place — the SAME max-subtraction algorithm the host `softmax` uses
//! (`lgbm_model::objective::softmax`), NOT a numerically-different re-implementation.
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::multiclass` grad/hess (the cpu f64 fold) is the parity
//! oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default cubecl-cpu f64 anchor
//! exercises this module (D-08); it is additive and OFF by default behind
//! `LGBM_CUDA_ON_DEVICE` (D-06). Numerically faithful to
//! `crates/lgbm-objective/src/multiclass.rs`.
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;

// =========================================================================
// Task 1: class-major softmax grad/hess #[cube] kernel + softmax ConvertOutput.
// =========================================================================

/// The generic softmax grad/hess body — one `#[cube]` fn over the cell type `F`, ONE
/// thread per ROW (Pattern 3). Transcribed VERBATIM from the host anchor
/// `multiclass.rs:156-178` with the class-major `[num_data*k + i]` stride (Pitfall 3):
/// - gather `rec[k] = scores[num_data*k + i]` for `k in 0..num_class`;
/// - `softmax` (max-subtraction): `wmax = max_k rec[k]`; `buf[idx] = exp(rec[k]-wmax)`;
///   `wsum = Σ_k buf`; `p_k = buf[idx] / wsum` (SAME algorithm as
///   `lgbm_model::objective::softmax`, D-09 — buffer as `output`, divide in place);
/// - `grad[num_data*k+i] = (label[i]==k) ? p_k - 1 : p_k`;
/// - `hess[num_data*k+i] = factor * p_k * (1 - p_k)`, `factor = num_class/(num_class-1)`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn softmax_grad_hess_body<F: Float>(
    scores: &Array<F>,
    labels: &Array<F>,
    softmax_buf: &mut Array<F>,
    grad: &mut Array<F>,
    hess: &mut Array<F>,
    num_data: usize,
    num_class: usize,
    factor: F,
) {
    // `ABSOLUTE_POS` is a `usize` lane index; the stride scalars are `usize` too
    // (the histogram `resident_bins[f*num_data + row]` idiom, histogram.rs:1096-1108).
    let i = ABSOLUTE_POS;
    if i < num_data {
        // wmax = max_k rec[k] (seed with class 0 at [num_data*0 + i] = i, C++ order).
        let mut wmax = scores[i];
        for k in 1..num_class {
            let v = scores[num_data * k + i];
            if v > wmax {
                wmax = v;
            }
        }
        // exp(rec[k]-wmax) into the per-row scratch; accumulate wsum (host softmax).
        let mut wsum = F::new(0.0);
        for k in 0..num_class {
            let idx = num_data * k + i;
            let e = (scores[idx] - wmax).exp();
            softmax_buf[idx] = e;
            wsum += e;
        }
        // grad/hess with the class-major stride; label==k selects the p-1 branch.
        let label_i = labels[i];
        for k in 0..num_class {
            let idx = num_data * k + i;
            let p = softmax_buf[idx] / wsum;
            let is_label = label_i == F::cast_from(k);
            grad[idx] = select(is_label, p - F::new(1.0), p);
            hess[idx] = factor * p * (F::new(1.0) - p);
        }
    }
}

/// f64 cpu-anchor wrapper (the deterministic f64-fold reference, D-10). The GPU/hip
/// f32 mirror would instantiate `softmax_grad_hess_body::<f32>` identically (D-07).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn softmax_grad_hess_kernel_f64(
    scores: &Array<f64>,
    labels: &Array<f64>,
    softmax_buf: &mut Array<f64>,
    grad: &mut Array<f64>,
    hess: &mut Array<f64>,
    num_data: usize,
    num_class: usize,
    factor: f64,
) {
    softmax_grad_hess_body::<f64>(
        scores, labels, softmax_buf, grad, hess, num_data, num_class, factor,
    );
}

/// Validate the softmax grad/hess host inputs at the V5 boundary (T-19-03-01) BEFORE
/// any device alloc / launch: `scores` must be exactly `num_data * num_class` and
/// `labels` exactly `num_data`.
fn validate_softmax(
    scores: usize,
    labels: usize,
    num_class: usize,
) -> Result<usize, ComputeError> {
    if num_class < 1 {
        return Err(ComputeError::LengthMismatch { expected: 1, actual: num_class });
    }
    if scores % num_class != 0 {
        return Err(ComputeError::LengthMismatch { expected: scores, actual: num_class });
    }
    let num_data = scores / num_class;
    if labels != num_data {
        return Err(ComputeError::LengthMismatch { expected: num_data, actual: labels });
    }
    Ok(num_data)
}

/// Compute multiclass-softmax grad/hess on the f64 cpu anchor.
///
/// `scores` is the WHOLE class-major f64 score buffer (length `num_data*num_class`,
/// class `k`'s rows at `[k*num_data, (k+1)*num_data)`); `labels` are the f32 integer
/// class labels (length `num_data`); `num_class` is `config.num_class`. Returns
/// `(grad, hess)` as class-major `f32` (`score_t`), the f64 body result cast once —
/// reproducing the C++ `f64-compute → f32-cast` order bit-for-bit.
///
/// The softmax `exp`'s ~1-ULP f64 residual is absorbed by the `score_t` f32 cast
/// (D-01), so the result is BIT-EXACT vs BOTH the host `MulticlassSoftmax::
/// get_gradients` f64 anchor and the class-major real `multiclass_gh` golden.
///
/// The D-09 per-row softmax scratch is a single `client.empty` handle sized
/// `num_data*num_class`, allocated ONCE before the launch.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `scores` is not `num_data*num_class` or
/// `labels` is not `num_data` (V5 boundary).
pub fn softmax_get_gradients_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    num_class: usize,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    let total = scores.len();
    let num_data = validate_softmax(total, labels.len(), num_class)?;
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    // factor_ = num_class / (num_class - 1.0f) (multiclass.rs:84).
    let factor = num_class as f64 / (num_class as f64 - 1.0f32 as f64);

    let h_scores = client.create_from_slice(f64::as_bytes(scores));
    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    // D-09: the per-row length-K softmax scratch — ONE reused client.empty handle,
    // pre-allocated ONCE outside the (single) launch (hoisted across iters in Phase 21).
    let h_softmax = client.empty(total * core::mem::size_of::<f64>());
    let h_grad = client.empty(total * core::mem::size_of::<f64>());
    let h_hess = client.empty(total * core::mem::size_of::<f64>());

    // ONE thread per ROW (num_data), each looping the num_class classes.
    let cube_dim = 256u32;
    let cube_count = (num_data as u32).div_ceil(cube_dim);

    // SAFETY: `h_scores`/`h_softmax`/`h_grad`/`h_hess` are each sized exactly `total`
    // f64 cells and `h_labels` `num_data`; all outlive the launch. The kernel
    // bounds-guards `i < num_data` and every class-major access `num_data*k + i` with
    // `k < num_class` lands in `[0, total)`; `labels[i]` in `[0, num_data)`. cubecl
    // unsafe confined here (CMP-01 / T-19-03-01).
    unsafe {
        softmax_grad_hess_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_scores, total),
            ArrayArg::from_raw_parts(h_labels, num_data),
            ArrayArg::from_raw_parts(h_softmax, total),
            ArrayArg::from_raw_parts(h_grad.clone(), total),
            ArrayArg::from_raw_parts(h_hess.clone(), total),
            num_data,
            num_class,
            factor,
        );
    }

    let grad_bytes = client.read_one_unchecked(h_grad);
    let hess_bytes = client.read_one_unchecked(h_hess);
    let grad: Vec<f32> = f64::from_bytes(&grad_bytes).iter().map(|&g| g as f32).collect();
    let hess: Vec<f32> = f64::from_bytes(&hess_bytes).iter().map(|&h| h as f32).collect();
    Ok((grad, hess))
}

// --- softmax ConvertOutput inverse-link (per-row softmax probability) ---

/// The per-row softmax ConvertOutput body (`ConvertOutputCUDAKernel_
/// MulticlassSoftmax`): one thread per ROW, `out[num_data*k+i] = softmax(rec)_k`.
/// Uses `out` itself as the exp scratch then divides in place — the SAME
/// max-subtraction algorithm as `lgbm_model::objective::softmax`.
#[cube]
fn softmax_convert_body<F: Float>(
    input: &Array<F>,
    out: &mut Array<F>,
    num_data: usize,
    num_class: usize,
) {
    let i = ABSOLUTE_POS;
    if i < num_data {
        let mut wmax = input[i];
        for k in 1..num_class {
            let v = input[num_data * k + i];
            if v > wmax {
                wmax = v;
            }
        }
        let mut wsum = F::new(0.0);
        for k in 0..num_class {
            let idx = num_data * k + i;
            let e = (input[idx] - wmax).exp();
            out[idx] = e;
            wsum += e;
        }
        for k in 0..num_class {
            let idx = num_data * k + i;
            out[idx] = out[idx] / wsum;
        }
    }
}

#[cube(launch_unchecked)]
fn softmax_convert_kernel_f64(
    input: &Array<f64>,
    out: &mut Array<f64>,
    num_data: usize,
    num_class: usize,
) {
    softmax_convert_body::<f64>(input, out, num_data, num_class);
}

/// Apply the per-row softmax ConvertOutput inverse-link on the f64 anchor over a
/// class-major raw-score buffer: `out[num_data*k+i] = softmax(rec_i)_k`. Returns the
/// class-major f64 probabilities, bit-exact vs the host `softmax` (same f64 libm
/// `exp`, same max-subtraction op order).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `input` is not `num_data*num_class`.
pub fn softmax_convert_output_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    input: &[f64],
    num_class: usize,
) -> Result<Vec<f64>, ComputeError> {
    let total = input.len();
    if num_class < 1 || (total != 0 && total % num_class != 0) {
        return Err(ComputeError::LengthMismatch { expected: total, actual: num_class });
    }
    if total == 0 {
        return Ok(Vec::new());
    }
    let num_data = total / num_class;
    let h_in = client.create_from_slice(f64::as_bytes(input));
    let h_out = client.empty(total * core::mem::size_of::<f64>());
    let cube_dim = 256u32;
    let cube_count = (num_data as u32).div_ceil(cube_dim);

    // SAFETY: `h_in`/`h_out` are each sized exactly `total` f64 cells and outlive the
    // launch; the kernel bounds-guards `i < num_data` and every `num_data*k+i` with
    // `k < num_class` lands in `[0, total)`. cubecl unsafe confined here (CMP-01).
    unsafe {
        softmax_convert_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_in, total),
            ArrayArg::from_raw_parts(h_out.clone(), total),
            num_data,
            num_class,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// =========================================================================
// Task 2: MulticlassOVA (one-vs-all) — reuse the binary-logloss response math
// per class at offset = num_data*i (§5.2 CUDAMulticlassOVA = binary-per-class).
// =========================================================================

/// The MulticlassOVA grad/hess body — ONE thread per ROW, looping the `num_class`
/// one-vs-all classes at `offset = num_data * i`. Transcribes the binary-logloss
/// response math locally (mirroring `binary.rs:87-94`) so this plan is self-contained
/// (no cross-file dependency on 19-02): per class `i`, `is_pos = (label == i)`,
/// `label_val = is_pos ? 1 : -1`, `response = -label_val*sigmoid/(1+exp(label_val*
/// sigmoid*score))`, `grad = response`, `hess = |response|*(sigmoid-|response|)`.
#[cube]
fn ova_grad_hess_body<F: Float>(
    scores: &Array<F>,
    labels: &Array<F>,
    grad: &mut Array<F>,
    hess: &mut Array<F>,
    num_data: usize,
    num_class: usize,
    sigmoid: F,
) {
    let row = ABSOLUTE_POS;
    if row < num_data {
        let label_i = labels[row];
        for i in 0..num_class {
            // class-major offset for one-vs-all class i.
            let offset = num_data * i;
            let idx = offset + row;
            let is_pos = label_i == F::cast_from(i);
            let label_val = select(is_pos, F::new(1.0), F::new(-1.0));
            let response =
                -label_val * sigmoid / (F::new(1.0) + (label_val * sigmoid * scores[idx]).exp());
            let abs_response = response.abs();
            grad[idx] = response;
            hess[idx] = abs_response * (sigmoid - abs_response);
        }
    }
}

#[cube(launch_unchecked)]
fn ova_grad_hess_kernel_f64(
    scores: &Array<f64>,
    labels: &Array<f64>,
    grad: &mut Array<f64>,
    hess: &mut Array<f64>,
    num_data: usize,
    num_class: usize,
    sigmoid: f64,
) {
    ova_grad_hess_body::<f64>(scores, labels, grad, hess, num_data, num_class, sigmoid);
}

/// Compute multiclass-OVA (one-vs-all) grad/hess on the f64 cpu anchor.
///
/// `scores` is the class-major f64 score buffer (length `num_data*num_class`);
/// `labels` are the f32 integer class labels (length `num_data`); `sigmoid` is
/// `config.sigmoid` (`> 0`). Per class `i`, the binary-logloss response is applied at
/// `offset = num_data*i` with `is_pos = (label == i)` — parity-neutral vs the host
/// `MulticlassOva` (K independent `Binary`, Discretion). Returns `(grad, hess)` as
/// class-major `f32`, bit-exact vs the host anchor and the real `multiclassova_gh`
/// golden (per-row math, no accumulation → the f64 `exp` residual is f32-cast-absorbed).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `scores` is not `num_data*num_class` or
/// `labels` is not `num_data` (V5 boundary).
pub fn multiclassova_get_gradients_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    num_class: usize,
    sigmoid: f64,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    let total = scores.len();
    let num_data = validate_softmax(total, labels.len(), num_class)?;
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    let h_scores = client.create_from_slice(f64::as_bytes(scores));
    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_grad = client.empty(total * core::mem::size_of::<f64>());
    let h_hess = client.empty(total * core::mem::size_of::<f64>());

    let cube_dim = 256u32;
    let cube_count = (num_data as u32).div_ceil(cube_dim);

    // SAFETY: `h_scores`/`h_grad`/`h_hess` are each sized exactly `total` f64 cells and
    // `h_labels` `num_data`; all outlive the launch. The kernel bounds-guards
    // `row < num_data` and every `num_data*i + row` with `i < num_class` lands in
    // `[0, total)`. cubecl unsafe confined here (CMP-01 / T-19-03-01).
    unsafe {
        ova_grad_hess_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_scores, total),
            ArrayArg::from_raw_parts(h_labels, num_data),
            ArrayArg::from_raw_parts(h_grad.clone(), total),
            ArrayArg::from_raw_parts(h_hess.clone(), total),
            num_data,
            num_class,
            sigmoid,
        );
    }

    let grad_bytes = client.read_one_unchecked(h_grad);
    let hess_bytes = client.read_one_unchecked(h_hess);
    let grad: Vec<f32> = f64::from_bytes(&grad_bytes).iter().map(|&g| g as f32).collect();
    let hess: Vec<f32> = f64::from_bytes(&hess_bytes).iter().map(|&h| h as f32).collect();
    Ok((grad, hess))
}
