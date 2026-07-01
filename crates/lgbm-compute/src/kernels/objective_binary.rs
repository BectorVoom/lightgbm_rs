//! On-device binary objective — sigmoid grad/hess + boost-from-score (§5.2, ODL-06).
//!
//! Owning phase: **19** (ODL-06). Filled by **19-02**.
//!
//! ## What lives here
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_binary_objective.cu`
//! (§5.2 of `docs/cuda-kernel-design.md`) — the per-row
//! `GetGradientsKernel_BinaryLogloss<USE_LABEL_WEIGHT,USE_WEIGHT>` sigmoid grad/hess,
//! the `ConvertOutputCUDAKernel_BinaryLogloss` sigmoid-probability inverse-link, the
//! two-stage `BoostFromScoreKernel_1/2_BinaryLogloss<USE_WEIGHT>` label-prior logit
//! init (Task 2 — device reduce COMPOSED with a host scalar finalize), and the
//! `ResetOVACUDALabelKernel` one-vs-all label rewrite (Task 2, shared with multiclass
//! OVA).
//!
//! ## Kernel shape (Pattern 1 / D-07)
//! ONE generic `#[cube]` grad/hess body over `F: Float` with TWO `#[comptime] bool`
//! params — `use_label_weight` (the `<USE_LABEL_WEIGHT>` template flag: the
//! per-class `label_weights_[is_pos]` reweighting for `is_unbalance` /
//! `scale_pos_weight`) and `use_weight` (the `<USE_WEIGHT>` per-row sample weight) —
//! both of which fold their branch out at expansion (no runtime weight branch). The
//! sigmoid `response` math is factored into a single `#[cube] fn` helper so it exists
//! ONCE (mirroring `random.rs`'s small-helper decomposition). The cpu **f64** anchor
//! wrapper is the deterministic reference (D-10); the per-row math runs in `F` and the
//! GPU/hip mirror would instantiate the SAME body with `F = f32` (D-07). The launcher
//! casts the f64 grad/hess read-back to `f32` (`score_t`), reproducing the golden's
//! `f64-compute → f32-cast` order bit-for-bit.
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::binary::Binary` grad/hess (the cpu f64 fold) is the
//! parity oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default cubecl-cpu f64
//! anchor exercises this module (D-08); it is additive and OFF by default behind
//! `LGBM_CUDA_ON_DEVICE` (D-06). Numerically faithful to
//! `crates/lgbm-objective/src/binary.rs`.
//!
//! ## BoostFromScore atomic residual (D-05 / Pitfall 5)
//! The per-row grad/hess carries NO accumulation, so it is BIT-EXACT vs the golden.
//! The two-stage `boost_from_score` init scalar reduces `Σ is_pos` on device (the
//! CUDA `atomicAdd` order) — the documented f32-vs-f64 accumulation residual (D-05) —
//! so it is anchored to the host f64 `boost_from_score` within `ORACLE_TOL`, NOT
//! bit-exact.
#![allow(unused_imports)]

use cubecl::prelude::*;
use lgbm_core::types::K_EPSILON;

use crate::error::ComputeError;
use crate::kernels::primitives::reduce_sum_f64_on;

// =========================================================================
// Task 1: binary grad/hess #[cube] kernel + sigmoid ConvertOutput.
// =========================================================================

/// The SINGLE-SOURCE-OF-TRUTH sigmoid `response` helper (`binary.rs:90`, verbatim):
/// `response = -label_val * sigmoid / (1 + exp(label_val * sigmoid * score))`. This
/// is the one place the sigmoid math is written; the grad/hess body and (by identity)
/// the whole objective are built on it.
#[cube]
fn binary_response<F: Float>(score: F, label_val: F, sigmoid: F) -> F {
    -label_val * sigmoid / (F::new(1.0) + (label_val * sigmoid * score).exp())
}

/// The generic grad/hess body — one `#[cube]` fn over the cell type `F`, with the
/// `<USE_LABEL_WEIGHT, USE_WEIGHT>` template mapped to two `#[comptime] bool` params
/// (Pattern 1). Transcribed VERBATIM from the host anchor `binary.rs:87-94`:
/// - `is_pos = label > 0`; `label_val = is_pos ? 1 : -1`.
/// - `response = binary_response(score, label_val, sigmoid)`.
/// - `grad = response`; `hess = |response| * (sigmoid - |response|)`.
/// - when `use_label_weight`: both `*= label_weights_[is_pos]` (the balanced default
///   is `1.0` for both classes → bit-identical to the off branch).
/// - when `use_weight`: both `*= weight[i]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn grad_hess_body<F: Float>(
    scores: &Array<F>,
    labels: &Array<F>,
    weights: &Array<F>,
    grad: &mut Array<F>,
    hess: &mut Array<F>,
    sigmoid: F,
    label_weight_neg: F,
    label_weight_pos: F,
    #[comptime] use_label_weight: bool,
    #[comptime] use_weight: bool,
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let is_pos = labels[i] > F::new(0.0);
        let label_val = select(is_pos, F::new(1.0), F::new(-1.0));
        let response = binary_response::<F>(scores[i], label_val, sigmoid);
        let abs_response = response.abs();

        let mut g = response;
        let mut h = abs_response * (sigmoid - abs_response);

        if use_label_weight {
            // C++ `label_weight = label_weights_[is_pos]` — the per-class reweight.
            let lw = select(is_pos, label_weight_pos, label_weight_neg);
            g = g * lw;
            h = h * lw;
        }
        if use_weight {
            // C++ weighted forms multiply BOTH grad and hess by the per-row weight.
            g = g * weights[i];
            h = h * weights[i];
        }
        grad[i] = g;
        hess[i] = h;
    }
}

/// f64 cpu-anchor wrapper (the deterministic f64-fold reference, D-10). The GPU/hip
/// f32 mirror would instantiate `grad_hess_body::<f32>` identically (D-07).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn grad_hess_kernel_f64(
    scores: &Array<f64>,
    labels: &Array<f64>,
    weights: &Array<f64>,
    grad: &mut Array<f64>,
    hess: &mut Array<f64>,
    sigmoid: f64,
    label_weight_neg: f64,
    label_weight_pos: f64,
    #[comptime] use_label_weight: bool,
    #[comptime] use_weight: bool,
) {
    grad_hess_body::<f64>(
        scores,
        labels,
        weights,
        grad,
        hess,
        sigmoid,
        label_weight_neg,
        label_weight_pos,
        use_label_weight,
        use_weight,
    );
}

/// Validate the grad/hess host inputs at the V5 boundary (T-19-02-01) BEFORE any
/// device alloc / launch: `labels`/`weights` (when present) must match `scores`.
fn validate_gh(scores: usize, labels: usize, weights: Option<usize>) -> Result<(), ComputeError> {
    if labels != scores {
        return Err(ComputeError::LengthMismatch { expected: scores, actual: labels });
    }
    if let Some(w) = weights {
        if w != scores {
            return Err(ComputeError::LengthMismatch { expected: scores, actual: w });
        }
    }
    Ok(())
}

/// Compute binary-logloss grad/hess on the f64 cpu anchor.
///
/// `scores` are the f64 accumulated raw scores (`score_t` promoted); `labels` are the
/// f32 labels (`is_pos = label > 0`); `weights` is `Some(&[f32])` for the per-row
/// weighted path (or `None`); `label_weights` is `Some((neg, pos))` for the per-class
/// `is_unbalance`/`scale_pos_weight` reweight (or `None` → the balanced `1.0/1.0`
/// default). `sigmoid` is `config.sigmoid` (`> 0`). Returns `(grad, hess)` as `f32`
/// (`score_t`), the f64 body result cast once — reproducing the C++ `f64-compute →
/// f32-cast` order bit-for-bit.
///
/// The per-row math carries NO accumulation, so the result is BIT-EXACT vs both the
/// host `Binary::get_gradients` f64 anchor and the real `binary_gh` golden (D-01).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if the slices disagree in length.
#[allow(clippy::too_many_arguments)]
pub fn get_gradients_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    sigmoid: f64,
    label_weights: Option<(f64, f64)>,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    let n = scores.len();
    validate_gh(n, labels.len(), weights.map(<[f32]>::len))?;
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    let use_weight = weights.is_some();
    let use_label_weight = label_weights.is_some();
    // The kernel signature needs a weights handle even when use_weight is false (the
    // comptime branch compiles the read out, so a 1-cell dummy is never accessed).
    let weights_f64: Vec<f64> = weights
        .map(|w| w.iter().map(|&x| f64::from(x)).collect())
        .unwrap_or_else(|| vec![0.0f64]);
    let (lw_neg, lw_pos) = label_weights.unwrap_or((1.0, 1.0));

    let h_scores = client.create_from_slice(f64::as_bytes(scores));
    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_weights = client.create_from_slice(f64::as_bytes(&weights_f64));
    let h_grad = client.empty(n * core::mem::size_of::<f64>());
    let h_hess = client.empty(n * core::mem::size_of::<f64>());

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);
    let w_len = weights_f64.len();

    // SAFETY: `h_scores`/`h_labels`/`h_grad`/`h_hess` are each sized exactly `n` f64
    // cells and `h_weights` `w_len`; all outlive the launch. The kernel bounds-guards
    // `i < scores.len()` so every `scores/labels/grad/hess[i]` access is in `[0, n)`;
    // `weights[i]` is only read on the (comptime) use_weight path, where `w_len == n`.
    // cubecl unsafe is confined here (CMP-01 / T-19-02-01).
    unsafe {
        grad_hess_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_scores, n),
            ArrayArg::from_raw_parts(h_labels, n),
            ArrayArg::from_raw_parts(h_weights, w_len),
            ArrayArg::from_raw_parts(h_grad.clone(), n),
            ArrayArg::from_raw_parts(h_hess.clone(), n),
            sigmoid,
            lw_neg,
            lw_pos,
            use_label_weight,
            use_weight,
        );
    }

    let grad_bytes = client.read_one_unchecked(h_grad);
    let hess_bytes = client.read_one_unchecked(h_hess);
    let grad: Vec<f32> = f64::from_bytes(&grad_bytes).iter().map(|&g| g as f32).collect();
    let hess: Vec<f32> = f64::from_bytes(&hess_bytes).iter().map(|&h| h as f32).collect();
    Ok((grad, hess))
}

// --- sigmoid ConvertOutput inverse-link (elementwise) ---

/// Elementwise sigmoid ConvertOutput body: `1 / (1 + exp(-sigmoid * x))`. Anchored to
/// `lgbm_model::objective::convert_binary` (verbatim op order).
#[cube]
fn convert_body<F: Float>(input: &Array<F>, out: &mut Array<F>, sigmoid: F) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        let x = input[i];
        out[i] = F::new(1.0) / (F::new(1.0) + (-sigmoid * x).exp());
    }
}

#[cube(launch_unchecked)]
fn convert_kernel_f64(input: &Array<f64>, out: &mut Array<f64>, sigmoid: f64) {
    convert_body::<f64>(input, out, sigmoid);
}

/// Apply the sigmoid ConvertOutput inverse-link elementwise on the f64 anchor:
/// `prob = 1 / (1 + exp(-sigmoid * raw))`. Returns the transformed f64 probabilities
/// (the model's ConvertOutput is f64), bit-exact vs `convert_binary` (same f64 libm
/// `exp`, same op order).
///
/// # Errors
/// Never (empty → empty); returns `Result` for launcher symmetry.
pub fn sigmoid_convert_output_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    input: &[f64],
    sigmoid: f64,
) -> Result<Vec<f64>, ComputeError> {
    let n = input.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let h_in = client.create_from_slice(f64::as_bytes(input));
    let h_out = client.empty(n * core::mem::size_of::<f64>());
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_in`/`h_out` are each sized exactly `n` f64 cells and outlive the
    // launch; the kernel bounds-guards `i < input.len()`. cubecl unsafe confined
    // here (CMP-01).
    unsafe {
        convert_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_in, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            sigmoid,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}
