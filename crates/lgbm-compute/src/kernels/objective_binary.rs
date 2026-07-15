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
use cubecl::server::Handle;
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

/// Resident-Handle grad/hess kernel (ODS-02x): the f32-OUTPUT variant of the §5.2
/// binary-logloss math. The per-row body is transcribed VERBATIM from
/// [`grad_hess_body`] at `F = f64` (the deterministic f64 fold, D-10) — only the
/// STORE differs: each output is written as `f32::cast_from(...)` directly (the
/// device-side `f64-compute → f32-cast` order, matching `get_gradients_on`'s
/// host-side `as f32` bit-for-bit, D-07). The score arrives as an already-resident
/// `Array<f64>` Handle (no host upload) and the grad/hess stay on device as f32
/// output Handles (no readback in the launcher) — the drop-in shape
/// [`crate::ResidentGradHess`] (f32 grad/hess) expects. `f64::new(x)` is the same
/// constructor `grad_hess_body::<f64>` expands to; `binary_response::<f64>` reuses the
/// single-source-of-truth sigmoid helper.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn grad_hess_resident_kernel_f32(
    scores: &Array<f64>,
    labels: &Array<f64>,
    weights: &Array<f64>,
    grad: &mut Array<f32>,
    hess: &mut Array<f32>,
    sigmoid: f64,
    label_weight_neg: f64,
    label_weight_pos: f64,
    #[comptime] use_label_weight: bool,
    #[comptime] use_weight: bool,
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let is_pos = labels[i] > f64::new(0.0);
        let label_val = select(is_pos, f64::new(1.0), f64::new(-1.0));
        let response = binary_response::<f64>(scores[i], label_val, sigmoid);
        let abs_response = response.abs();

        let mut g = response;
        let mut h = abs_response * (sigmoid - abs_response);

        if use_label_weight {
            let lw = select(is_pos, label_weight_pos, label_weight_neg);
            g = g * lw;
            h = h * lw;
        }
        if use_weight {
            g = g * weights[i];
            h = h * weights[i];
        }
        grad[i] = f32::cast_from(g);
        hess[i] = f32::cast_from(h);
    }
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

/// Resident-Handle binary-logloss grad/hess launcher (ODS-02x) — the device-in/
/// device-out sibling of [`get_gradients_on`]. Takes an already-resident `Array<f64>`
/// score `Handle` (NEVER a host slice — the driver keeps the score device-resident,
/// so there is no per-call `create_from_slice` of the whole score) and writes f32
/// grad/hess directly into two freshly allocated device Handles which it returns
/// WITHOUT reading back (the caller — Plan 04's `gbdt.rs` grad/hess branch — decides
/// when/if to read them, or feeds them straight into the resident build). Only the
/// small per-train-constant `labels`/`weights` are uploaded here. `sigmoid` /
/// `label_weights` carry the same meaning as [`get_gradients_on`]; `num_data` is
/// explicit because the Handle carries no length.
///
/// Bit-exact vs [`get_gradients_on`] for the same inputs across both `label_weights`
/// states (proven by `get_gradients_resident_on_matches_host_slice`): the kernel runs
/// the same f64 per-row math and the same `f64-compute → f32-cast` order, only moving
/// the cast onto the device (`f32::cast_from`) instead of the host (`as f32`).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `labels`/`weights` disagree with `num_data`.
#[allow(clippy::too_many_arguments)]
pub fn get_gradients_resident_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores_handle: &Handle,
    num_data: usize,
    labels: &[f32],
    weights: Option<&[f32]>,
    sigmoid: f64,
    label_weights: Option<(f64, f64)>,
) -> Result<(Handle, Handle), ComputeError> {
    validate_gh(num_data, labels.len(), weights.map(<[f32]>::len))?;
    if num_data == 0 {
        return Ok((client.empty(0), client.empty(0)));
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

    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_weights = client.create_from_slice(f64::as_bytes(&weights_f64));
    let h_grad = client.empty(num_data * core::mem::size_of::<f32>());
    let h_hess = client.empty(num_data * core::mem::size_of::<f32>());

    let cube_dim = 256u32;
    let cube_count = (num_data as u32).div_ceil(cube_dim);
    let w_len = weights_f64.len();

    // SAFETY: `scores_handle` is the caller's resident `[num_data]` f64 score buffer;
    // `h_labels`/`h_grad`/`h_hess` are each sized exactly `num_data` cells (f64 in,
    // f32 out) and `h_weights` `w_len`; all outlive the launch. The kernel
    // bounds-guards `i < scores.len()` so every `scores/labels/grad/hess[i]` access is
    // in `[0, num_data)`; `weights[i]` is only read on the (comptime) use_weight path,
    // where `w_len == num_data`. cubecl unsafe is confined here (CMP-01 / T-31-02).
    unsafe {
        grad_hess_resident_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(scores_handle.clone(), num_data),
            ArrayArg::from_raw_parts(h_labels, num_data),
            ArrayArg::from_raw_parts(h_weights, w_len),
            ArrayArg::from_raw_parts(h_grad.clone(), num_data),
            ArrayArg::from_raw_parts(h_hess.clone(), num_data),
            sigmoid,
            lw_neg,
            lw_pos,
            use_label_weight,
            use_weight,
        );
    }

    Ok((h_grad, h_hess))
}

/// [`get_gradients_resident_on`] over a per-train [`GradResidency`]
/// (`crate::kernels::grow_driver::GradResidency`): the f64 label buffer was uploaded
/// ONCE by [`crate::ResidentScore::grad_residency`] and the f32 grad/hess outputs are
/// REUSED across iterations (each launch fully overwrites every `i < num_data` cell,
/// so reuse is value-identical to the prior per-iter `client.empty` pair). This
/// removes the per-iteration host f32→f64 label convert + `4·num_data`-byte
/// host→device label upload + two device output allocations the non-cached launcher
/// pays every call. SAME kernel, launch geometry, and scalar args as
/// [`get_gradients_resident_on`] — bit-exact by construction; only the buffer
/// provenance differs. The resident envelope is unweighted (`use_weight == false`
/// compiles the weights read out; the 1-cell dummy is cached in the residency).
///
/// # Errors
/// Currently infallible for `num_data > 0` (the `Result` mirrors the sibling
/// launcher so call sites propagate uniformly).
pub fn get_gradients_resident_cached_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores_handle: &Handle,
    num_data: usize,
    residency: &crate::kernels::grow_driver::GradResidency,
    sigmoid: f64,
    label_weights: Option<(f64, f64)>,
) -> Result<(Handle, Handle), ComputeError> {
    if num_data == 0 {
        return Ok((client.empty(0), client.empty(0)));
    }
    let use_label_weight = label_weights.is_some();
    let (lw_neg, lw_pos) = label_weights.unwrap_or((1.0, 1.0));
    // Unweighted envelope: 1-cell dummy, never read (`use_weight == false`).
    let h_weights = client.create_from_slice(f64::as_bytes(&[0.0f64]));

    let cube_dim = 256u32;
    let cube_count = (num_data as u32).div_ceil(cube_dim);

    // SAFETY: `scores_handle`/`residency.labels_f64` are `[num_data]` f64 buffers and
    // `residency.grad`/`residency.hess` `[num_data]` f32 buffers (all sized by
    // `ResidentScore::grad_residency` against the SAME `num_data`); all outlive the
    // launch. The kernel bounds-guards `i < scores.len()`; the weights dummy is never
    // read (`use_weight == false` compiles the read out). cubecl unsafe is confined
    // here (CMP-01).
    unsafe {
        grad_hess_resident_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(scores_handle.clone(), num_data),
            ArrayArg::from_raw_parts(residency.labels_f64.clone(), num_data),
            ArrayArg::from_raw_parts(h_weights, 1),
            ArrayArg::from_raw_parts(residency.grad.clone(), num_data),
            ArrayArg::from_raw_parts(residency.hess.clone(), num_data),
            sigmoid,
            lw_neg,
            lw_pos,
            use_label_weight,
            false,
        );
    }

    Ok((residency.grad.clone(), residency.hess.clone()))
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

// =========================================================================
// Task 2: two-stage BoostFromScore (logit init) — device reduce COMPOSED with
// a host scalar finalize (D-08 — compose the primitive, never hand-roll a
// reduce) — plus the OVA (one-vs-all) label-reset kernel.
// =========================================================================

/// The two-stage binary `BoostFromScore` (the C++
/// `BoostFromScoreKernel_1/2_BinaryLogloss<USE_WEIGHT>` analog, mirroring
/// `binary.rs:102-117` verbatim):
/// - **stage 1** (device, the `atomicAdd` reduce): `Σ is_pos` via
///   [`reduce_sum_f64_on`] where `is_pos = (label > 0) ? 1 : 0`; on the weighted path
///   `Σ is_pos·w` and `sumw = Σ w` are reduced too (D-08 — COMPOSE the primitive).
/// - **stage 2** (`<<<1,1>>>` analog, an f64 host scalar — D-07 allows f64 in the
///   scalar BoostFromScore): `pavg = clamp(Σ / N, ε, 1-ε); init = ln(pavg/(1-pavg)) /
///   sigmoid`. `ε = K_EPSILON as f64` (the `1e-15f32 as f64` narrow — matched to the
///   host anchor's `K_EPSILON as f64` bit-for-bit).
///
/// `weights` is `Some(&[f32])` for the per-row weighted prior (or `None` → unweighted,
/// `sumw = N` exactly as `binary.rs:106`). Empty labels → `0.0`.
///
/// The device `Σ is_pos` reduce is the documented f32-vs-f64 atomicAdd-order residual
/// (D-05 / Pitfall 5) — the caller asserts this init within `ORACLE_TOL`, NOT
/// bit-exact.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `weights` is present and disagrees with
/// `labels`; propagates the [`reduce_sum_f64_on`] error.
pub fn boost_from_score_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    labels: &[f32],
    weights: Option<&[f32]>,
    sigmoid: f64,
) -> Result<f64, ComputeError> {
    if labels.is_empty() {
        return Ok(0.0);
    }
    if let Some(w) = weights {
        if w.len() != labels.len() {
            return Err(ComputeError::LengthMismatch { expected: labels.len(), actual: w.len() });
        }
    }

    // Stage 1: device reduce of Σ is_pos [and, on the weighted path, Σ is_pos·w + Σ w].
    let (suml, sumw) = if let Some(w) = weights {
        let is_pos_w: Vec<f64> = labels
            .iter()
            .zip(w)
            .map(|(&l, &wi)| if l > 0.0 { f64::from(wi) } else { 0.0 })
            .collect();
        let w_f64: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
        (reduce_sum_f64_on(client, &is_pos_w)?, reduce_sum_f64_on(client, &w_f64)?)
    } else {
        let is_pos: Vec<f64> = labels.iter().map(|&l| if l > 0.0 { 1.0 } else { 0.0 }).collect();
        // Unweighted: sumw = num_data exactly (binary.rs:106).
        (reduce_sum_f64_on(client, &is_pos)?, labels.len() as f64)
    };

    // Stage 2: host f64 scalar finalize (mirrors binary.rs:112-116 verbatim).
    let mut pavg = suml / sumw;
    let eps = f64::from(K_EPSILON);
    pavg = pavg.min(1.0 - eps);
    pavg = pavg.max(eps);
    Ok((pavg / (1.0 - pavg)).ln() / sigmoid)
}

// --- OVA (one-vs-all) label reset (elementwise) ---

/// The `ResetOVACUDALabelKernel` body: per-row `out = (label == class) ? +1 : -1` —
/// the one-vs-all label rewrite shared with the multiclass OVA path. `class_id` is
/// integer-valued (class indices stored as `F`), so the `==` is exact.
#[cube]
fn reset_ova_body<F: Float>(labels: &Array<F>, out: &mut Array<F>, class_id: F) {
    let i = ABSOLUTE_POS;
    if i < labels.len() {
        out[i] = select(labels[i] == class_id, F::new(1.0), F::new(-1.0));
    }
}

#[cube(launch_unchecked)]
fn reset_ova_kernel_f64(labels: &Array<f64>, out: &mut Array<f64>, class_id: f64) {
    reset_ova_body::<f64>(labels, out, class_id);
}

/// Rewrite `labels` into the one-vs-all `±1` target for class `class_id` on the f64
/// anchor: `out[i] = (label[i] == class_id) ? +1 : -1`. Returns the `±1` labels as
/// `f32`. Bit-exact (elementwise compare/select, no accumulation).
///
/// # Errors
/// Never (empty → empty); returns `Result` for launcher symmetry.
pub fn reset_ova_label_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    labels: &[f32],
    class_id: i32,
) -> Result<Vec<f32>, ComputeError> {
    let n = labels.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    let h_in = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_out = client.empty(n * core::mem::size_of::<f64>());
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_in`/`h_out` are each sized exactly `n` f64 cells and outlive the
    // launch; the kernel bounds-guards `i < labels.len()`. cubecl unsafe confined
    // here (CMP-01 / T-19-02-01).
    unsafe {
        reset_ova_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_in, n),
            ArrayArg::from_raw_parts(h_out.clone(), n),
            f64::from(class_id),
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).iter().map(|&x| x as f32).collect())
}

#[cfg(test)]
mod resident_parity {
    use super::{get_gradients_on, get_gradients_resident_on};
    use crate::runtime::cpu_client;
    use cubecl::prelude::*;

    /// ODS-03: the resident-Handle binary launcher is BIT-EXACT (`to_bits()`
    /// element-wise) vs the existing host-slice [`get_gradients_on`] across BOTH
    /// `label_weights` states (`None` balanced default and `Some((neg, pos))`) and
    /// both the unweighted and weighted paths — proving the Handle-in/Handle-out
    /// plumbing carries no numeric difference (the only change is moving the
    /// `f64 → f32` cast from host `as f32` onto the device `f32::cast_from`).
    #[test]
    fn get_gradients_resident_on_matches_host_slice() {
        let client = cpu_client();
        // Mixed +/- scores and both label classes exercise the is_pos select, the
        // sigmoid response sign, and the |response| hessian form.
        let scores: Vec<f64> = vec![0.5, -0.25, 1.0, -2.0, 0.0, 1.5, -3.0, 0.75];
        let labels: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let weights: Vec<f32> = vec![1.0, 2.0, 0.5, 1.25, 3.0, 0.75, 1.0, 2.5];
        let n = scores.len();
        let sigmoid = 1.0f64;

        for label_weights in [None, Some((0.8f64, 1.3f64))] {
            for weights_opt in [None, Some(weights.as_slice())] {
                let (host_g, host_h) = get_gradients_on(
                    &client,
                    &scores,
                    &labels,
                    weights_opt,
                    sigmoid,
                    label_weights,
                )
                .expect("host-slice binary launcher");

                // Upload the score ONCE to a Handle (the driver's resident buffer
                // analog), then call the resident launcher with the Handle.
                let scores_handle = client.create_from_slice(f64::as_bytes(&scores));
                let (h_grad, h_hess) = get_gradients_resident_on(
                    &client,
                    &scores_handle,
                    n,
                    &labels,
                    weights_opt,
                    sigmoid,
                    label_weights,
                )
                .expect("resident binary launcher");

                // Read back ONLY for the assertion (never inside the launcher).
                let res_g = f32::from_bytes(&client.read_one_unchecked(h_grad)).to_vec();
                let res_h = f32::from_bytes(&client.read_one_unchecked(h_hess)).to_vec();

                assert_eq!(res_g.len(), n);
                assert_eq!(res_h.len(), n);
                let lw = label_weights.is_some();
                let w = weights_opt.is_some();
                for i in 0..n {
                    assert_eq!(
                        res_g[i].to_bits(),
                        host_g[i].to_bits(),
                        "grad bit mismatch label_weights={lw} weighted={w} i={i} \
                         (resident={} host={})",
                        res_g[i],
                        host_g[i]
                    );
                    assert_eq!(
                        res_h[i].to_bits(),
                        host_h[i].to_bits(),
                        "hess bit mismatch label_weights={lw} weighted={w} i={i} \
                         (resident={} host={})",
                        res_h[i],
                        host_h[i]
                    );
                }
            }
        }
    }

    /// The CACHED resident launcher (per-train [`GradResidency`]) is BIT-EXACT vs
    /// the per-call resident launcher across both `label_weights` states AND across
    /// repeated calls over the SAME residency (the reuse contract: every launch
    /// fully overwrites the reused grad/hess outputs, so iteration k's result is
    /// independent of iteration k-1's residue).
    #[test]
    fn get_gradients_resident_cached_on_matches_uncached() {
        use super::get_gradients_resident_cached_on;
        use crate::ResidentScore;

        let client = cpu_client();
        let labels: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let n = labels.len();
        let sigmoid = 1.0f64;

        for scores in [
            vec![0.5, -0.25, 1.0, -2.0, 0.0, 1.5, -3.0, 0.75],
            vec![-1.5, 0.75, 0.0, 2.0, -0.5, -1.0, 3.0, -0.25],
        ] {
            let rs = ResidentScore::from_host_scores(&client, &scores);
            // Two calls over ONE residency with different label_weights — the
            // second call's (different) result must be its own launch's output,
            // proving each launch fully overwrites the reused grad/hess buffers.
            for label_weights in [Some((0.8f64, 1.3f64)), None] {
                let residency =
                    rs.grad_residency(&client, &labels).expect("residency builds");
                let (cg, ch) = get_gradients_resident_cached_on(
                    &client,
                    rs.score_handle(),
                    n,
                    residency,
                    sigmoid,
                    label_weights,
                )
                .expect("cached resident binary launcher");

                let scores_handle = client.create_from_slice(f64::as_bytes(&scores));
                let (ug, uh) = get_gradients_resident_on(
                    &client,
                    &scores_handle,
                    n,
                    &labels,
                    None,
                    sigmoid,
                    label_weights,
                )
                .expect("uncached resident binary launcher");

                let cg = f32::from_bytes(&client.read_one_unchecked(cg)).to_vec();
                let ch = f32::from_bytes(&client.read_one_unchecked(ch)).to_vec();
                let ug = f32::from_bytes(&client.read_one_unchecked(ug)).to_vec();
                let uh = f32::from_bytes(&client.read_one_unchecked(uh)).to_vec();
                for i in 0..n {
                    assert_eq!(cg[i].to_bits(), ug[i].to_bits(), "grad mismatch i={i}");
                    assert_eq!(ch[i].to_bits(), uh[i].to_bits(), "hess mismatch i={i}");
                }
            }
        }
    }
}
