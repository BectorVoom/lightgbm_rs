//! On-device regression objectives — grad/hess + inverse-link (§5.1, ODL-05).
//!
//! Owning phase: **19** (ODL-05). Filled by **19-01**.
//!
//! ## What lives here
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_regression_objective.cu`
//! (§5.1 of `docs/cuda-kernel-design.md`) — the per-row
//! `GetGradientsKernel_Regression{L2,L1,Huber,Fair,Poisson,Quantile}<USE_WEIGHT>`
//! grad/hess kernels, the `ConvertOutputCUDAKernel_Regression{,_Poisson}`
//! inverse-link (sqrt/exp), the `BoostFromScore` init (mean via reduce / median via
//! percentile / Poisson label-check + safe_log — Task 2), and the
//! `RenewTreeOutputCUDAKernel_Regression{L1,Quantile}` per-leaf median refit
//! (Task 3, HOST-orchestrated — see the discrepancy note below).
//!
//! ## Kernel shape (Pattern 1 / D-135 / D-07)
//! ONE generic `#[cube]` grad/hess body over `F: Float` with a `#[comptime]
//! objective_tag: u32` (the six §5.1 objectives fold to straight-line code at
//! expansion — no runtime objective branch) plus a `#[comptime] use_weight: bool`
//! (the `<USE_WEIGHT>` template flag). The cpu **f64** anchor wrapper is the
//! deterministic reference (D-10); the per-row math therefore runs in `F` and the
//! GPU/hip mirror instantiates the SAME body with `F = f32` (D-07: no hardcoded f64
//! per-row hot loop — the f64 here is the anchor's f64-fold, not a production f32
//! path). The launcher casts the f64 grad/hess read-back to `f32` (`score_t`),
//! reproducing the golden's `f64-compute → f32-cast` order bit-for-bit.
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::regression::Objective` grad/hess (the cpu f64 fold) is
//! the parity oracle — NEVER GPU-vs-GPU (D-05 / def-f8u-01). The default cubecl-cpu
//! f64 anchor exercises this module (D-08); it is additive and OFF by default behind
//! `LGBM_CUDA_ON_DEVICE` (D-06). Numerically faithful to
//! `crates/lgbm-objective/src/regression.rs`.
//!
//! ## Discrepancy 1 — no `percentile_device` per-leaf kernel
//! `RenewTreeOutput` (L1/Quantile per-leaf median) is a HOST-ORCHESTRATED loop over
//! leaves calling [`crate::kernels::primitives::percentile_unweighted_f32_on`] per
//! leaf. There is NO `percentile_device` per-segment kernel (it does not exist in
//! the primitive set); the host loop is parity-identical to the CUDA
//! one-block-per-leaf result (same sort, same percentile index math).
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::kernels::primitives::{bitonic_argsort_on, reduce_min_f64_on, reduce_sum_f64_on};

// =========================================================================
// Objective tags (comptime dispatch selector) + convert modes.
// =========================================================================

/// L2 (`regression`): `grad = diff`, `hess = 1` (`regression_objective.hpp` L2).
pub const TAG_L2: u32 = 0;
/// L1 (`regression_l1`): `grad = Sign(diff)`, `hess = 1`.
pub const TAG_L1: u32 = 1;
/// Quantile (`quantile`): `grad = diff>=0 ? 1-alpha : -alpha`, `hess = 1`
/// (`param` = the f32-rounded quantile alpha).
pub const TAG_QUANTILE: u32 = 2;
/// Huber (`huber`): `grad = |diff|<=alpha ? diff : Sign(diff)*alpha`, `hess = 1`
/// (`param` = the Huber delta alpha).
pub const TAG_HUBER: u32 = 3;
/// Fair (`fair`): `grad = c*diff/(|diff|+c)`, `hess = c²/(|diff|+c)²`
/// (`param` = the Fair `c`).
pub const TAG_FAIR: u32 = 4;
/// Poisson (`poisson`): `grad = exp(score)-label`, `hess = exp(score)*exp(mds)`
/// (`param` = `exp(max_delta_step)`, hoisted on the host like the C++ reference).
pub const TAG_POISSON: u32 = 5;

/// ConvertOutput identity (non-sqrt regression / L1 / huber / fair / quantile).
pub const CONVERT_PASSTHROUGH: u32 = 0;
/// ConvertOutput `Sign(x)*x*x` (the `regression sqrt` variant).
pub const CONVERT_SQRT_SQUARE: u32 = 1;
/// ConvertOutput `exp(x)` (Poisson log-mean inverse-link).
pub const CONVERT_EXP: u32 = 2;

// =========================================================================
// Task 1: six regression grad/hess #[cube] kernels + ConvertOutput.
// =========================================================================

/// `Common::Sign` as a branchless `#[cube]` helper: `(x>0) - (x<0)` → `-1|0|1`
/// (mirrors `regression.rs::sign`). Branchless `select` stores only (the
/// cubecl-cpu MLIR idiom, `data_partition.rs`).
#[cube]
fn sign_f<F: Float>(x: F) -> F {
    let pos = select(x > F::new(0.0), F::new(1.0), F::new(0.0));
    select(x < F::new(0.0), F::new(-1.0), pos)
}

/// The SINGLE-SOURCE-OF-TRUTH grad/hess body — one generic `#[cube]` fn over the
/// cell type `F`, dispatching the six §5.1 objectives via the `#[comptime]
/// objective_tag` (folds to straight-line code at expansion) and the `<USE_WEIGHT>`
/// template via `#[comptime] use_weight`. `diff = score - label` is the shared
/// per-row form; each objective's link is transcribed VERBATIM from the host anchor
/// `regression.rs` match arms (L2 :407, L1 :413, Huber :425, Fair :442, Quantile
/// :456, Poisson :489).
#[cube]
#[allow(unused_assignments, clippy::too_many_arguments)]
fn grad_hess_body<F: Float>(
    scores: &Array<F>,
    labels: &Array<F>,
    weights: &Array<F>,
    grad: &mut Array<F>,
    hess: &mut Array<F>,
    param: F,
    #[comptime] objective_tag: u32,
    #[comptime] use_weight: bool,
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let s = scores[i];
        let l = labels[i];
        let diff = s - l;

        // Seed values are overwritten by the exhaustive comptime tag chain below
        // (the `#[cube]` macro restructures the branches so definite-init of a bare
        // binding does not hold — the seed keeps it a plain init; unused_assignments
        // is allowed on the fn).
        let mut g = F::new(0.0);
        let mut h = F::new(1.0);

        if objective_tag == TAG_L2 {
            // grad = diff; hess = 1.
            g = diff;
            h = F::new(1.0);
        } else if objective_tag == TAG_L1 {
            // grad = Sign(diff); hess = 1.
            g = sign_f::<F>(diff);
            h = F::new(1.0);
        } else if objective_tag == TAG_QUANTILE {
            // grad = (diff >= 0) ? (1 - alpha) : -alpha; hess = 1. `param` is the
            // f32-rounded alpha (score_t in C++). `1 - param` / `-param` are single
            // ops whose f32-narrowed output matches the C++ f32 subtract bit-for-bit.
            g = select(diff >= F::new(0.0), F::new(1.0) - param, F::new(0.0) - param);
            h = F::new(1.0);
        } else if objective_tag == TAG_HUBER {
            // grad = |diff|<=alpha ? diff : Sign(diff)*alpha; hess = 1.
            let ad = diff.abs();
            g = select(ad <= param, diff, sign_f::<F>(diff) * param);
            h = F::new(1.0);
        } else if objective_tag == TAG_FAIR {
            // grad = c*x/(|x|+c); hess = c²/(|x|+c)². `param` = c.
            let denom = diff.abs() + param;
            g = param * diff / denom;
            h = (param * param) / (denom * denom);
        } else {
            // TAG_POISSON: grad = exp(score) - label; hess = exp(score)*exp(mds).
            // `param` = exp(max_delta_step), hoisted on the host (regression.rs:496).
            let es = s.exp();
            g = es - l;
            h = es * param;
        }

        if use_weight {
            // C++ weighted forms multiply BOTH grad and hess by the per-row weight
            // (regression_objective.hpp) — grad *= w, hess *= w. For all-1.0 weights
            // this is bit-identical to the unweighted branch (weight-equivalence).
            grad[i] = g * weights[i];
            hess[i] = h * weights[i];
        } else {
            grad[i] = g;
            hess[i] = h;
        }
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
    param: f64,
    #[comptime] objective_tag: u32,
    #[comptime] use_weight: bool,
) {
    grad_hess_body::<f64>(scores, labels, weights, grad, hess, param, objective_tag, use_weight);
}

/// Validate the grad/hess host inputs at the V5 boundary (T-19-01-01) BEFORE any
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

/// Compute grad/hess for one regression objective on the f64 cpu anchor.
///
/// `scores` are the f64 accumulated raw scores (`score_t` promoted); `labels` are
/// the (already Init-transformed) f32 labels; `weights` is `Some(&[f32])` for the
/// weighted path (or `None`). `param` carries the objective scalar (quantile: the
/// f32-rounded alpha; huber: delta; fair: c; poisson: `exp(max_delta_step)`; L2/L1:
/// ignored). Returns `(grad, hess)` as `f32` (`score_t`), the f64 body result cast
/// once — reproducing the C++ `f64-compute → f32-cast` order bit-for-bit.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if the slices disagree in length.
pub fn get_gradients_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    objective_tag: u32,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    param: f64,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    let n = scores.len();
    validate_gh(n, labels.len(), weights.map(<[f32]>::len))?;
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    let use_weight = weights.is_some();
    // The kernel signature needs a weights handle even when use_weight is false (the
    // comptime branch compiles the read out, so a 1-cell dummy is never accessed).
    let weights_f64: Vec<f64> = weights
        .map(|w| w.iter().map(|&x| f64::from(x)).collect())
        .unwrap_or_else(|| vec![0.0f64]);

    let h_scores = client.create_from_slice(f64::as_bytes(scores));
    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_weights = client.create_from_slice(f64::as_bytes(&weights_f64));
    let h_grad = client.empty(n * core::mem::size_of::<f64>());
    let h_hess = client.empty(n * core::mem::size_of::<f64>());

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);
    let w_len = weights_f64.len();

    // SAFETY: `h_scores`/`h_labels`/`h_grad`/`h_hess` are each sized exactly `n`
    // f64 cells and `h_weights` `w_len`; all outlive the launch. The kernel
    // bounds-guards `i < scores.len()` so every `scores/labels/grad/hess[i]` access
    // is in `[0, n)`; `weights[i]` is only read on the (comptime) use_weight path,
    // where `w_len == n`. cubecl unsafe is confined here (CMP-01 / T-19-01-01).
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
            param,
            objective_tag,
            use_weight,
        );
    }

    let grad_bytes = client.read_one_unchecked(h_grad);
    let hess_bytes = client.read_one_unchecked(h_hess);
    let grad: Vec<f32> = f64::from_bytes(&grad_bytes).iter().map(|&g| g as f32).collect();
    let hess: Vec<f32> = f64::from_bytes(&hess_bytes).iter().map(|&h| h as f32).collect();
    Ok((grad, hess))
}

// --- per-objective convenience launchers (thin `get_gradients_on` wrappers) ---

/// L2 grad/hess (`regression`). See [`get_gradients_on`].
///
/// # Errors
/// Propagates [`get_gradients_on`].
pub fn get_gradients_l2_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    get_gradients_on(client, TAG_L2, scores, labels, weights, 0.0)
}

/// L1 grad/hess (`regression_l1`). See [`get_gradients_on`].
///
/// # Errors
/// Propagates [`get_gradients_on`].
pub fn get_gradients_l1_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    get_gradients_on(client, TAG_L1, scores, labels, weights, 0.0)
}

/// Quantile grad/hess (`quantile`); `alpha` is rounded through f32 (C++ `alpha_` is
/// `score_t`) before the launch. See [`get_gradients_on`].
///
/// # Errors
/// Propagates [`get_gradients_on`].
pub fn get_gradients_quantile_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    alpha: f64,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    get_gradients_on(client, TAG_QUANTILE, scores, labels, weights, f64::from(alpha as f32))
}

/// Huber grad/hess (`huber`); `alpha` is the Huber delta. See [`get_gradients_on`].
///
/// # Errors
/// Propagates [`get_gradients_on`].
pub fn get_gradients_huber_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    alpha: f64,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    get_gradients_on(client, TAG_HUBER, scores, labels, weights, alpha)
}

/// Fair grad/hess (`fair`); `c` is the Fair parameter. See [`get_gradients_on`].
///
/// # Errors
/// Propagates [`get_gradients_on`].
pub fn get_gradients_fair_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    c: f64,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    get_gradients_on(client, TAG_FAIR, scores, labels, weights, c)
}

/// Poisson grad/hess (`poisson`); `max_delta_step` is `exp`-hoisted on the host
/// (matching `regression.rs:496`) before the launch. See [`get_gradients_on`].
///
/// # Errors
/// Propagates [`get_gradients_on`].
pub fn get_gradients_poisson_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    max_delta_step: f64,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    get_gradients_on(client, TAG_POISSON, scores, labels, weights, max_delta_step.exp())
}

// --- ConvertOutput inverse-link (elementwise) ---

/// Elementwise ConvertOutput body: passthrough / `Sign(x)*x*x` / `exp(x)` selected
/// by the `#[comptime] mode`. Anchored to `lgbm_model::objective::convert_regression`
/// / `convert_poisson`.
#[cube]
#[allow(unused_assignments)]
fn convert_body<F: Float>(input: &Array<F>, out: &mut Array<F>, #[comptime] mode: u32) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        let x = input[i];
        let mut y = x;
        if mode == CONVERT_PASSTHROUGH {
            y = x;
        } else if mode == CONVERT_SQRT_SQUARE {
            y = sign_f::<F>(x) * x * x;
        } else {
            y = x.exp();
        }
        out[i] = y;
    }
}

#[cube(launch_unchecked)]
fn convert_kernel_f64(input: &Array<f64>, out: &mut Array<f64>, #[comptime] mode: u32) {
    convert_body::<f64>(input, out, mode);
}

/// Apply the ConvertOutput inverse-link elementwise on the f64 anchor.
///
/// `mode` is [`CONVERT_PASSTHROUGH`] / [`CONVERT_SQRT_SQUARE`] / [`CONVERT_EXP`].
/// Returns the transformed f64 outputs (the model's ConvertOutput is f64).
///
/// # Errors
/// Never (empty → empty); returns `Result` for launcher symmetry.
pub fn convert_output_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    input: &[f64],
    mode: u32,
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
            mode,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// =========================================================================
// Task 2: BoostFromScore init — device reduce/percentile COMPOSED with a host
// scalar finalize (D-08 — compose the primitives, never hand-roll a reduce).
// =========================================================================

/// `Common::SafeLog` (`common.h:877`): `ln(x)` for `x > 0`, else `-inf`. Mirrors
/// `regression.rs::safe_log`; the scalar finalize is legitimately f64 (D-07 allows
/// f64 in the scalar BoostFromScore).
fn safe_log(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Device-composed `PercentileFun` (the CPU BoostFromScore/RenewTreeOutput
/// reference, `lgbm_objective::percentile::percentile_fun`): sort DESCENDING via the
/// shared [`bitonic_argsort_on`] device primitive (index-only), then apply the exact
/// `PercentileFun` index math as an f64 scalar finalize (D-07 allows f64 in the
/// scalar path). Returns the raw f64 percentile (the caller applies the f32 narrow
/// where the C++ reference does — quantile BoostFromScore only).
///
/// **Deviation (Rule 1 — hardening the percentile skeleton at its Phase-19
/// consumer):** the [`crate::kernels::primitives::percentile_unweighted_f32_on`]
/// skeleton uses a `(1-alpha)*len` index convention that DIVERGES from the CPU
/// `PercentileFun` `(len-1)*(1-alpha)` convention the real-lib_lightgbm
/// BoostFromScore / RenewTreeOutput goldens use (e.g. the spine-label median is 10
/// under the skeleton vs the reference 11). Rather than mutate that phase-14
/// primitive (its ROCm percentile fixture would silently drift), this Phase-19
/// consumer — the designated percentile hardening owner — composes the argsort
/// primitive with the `PercentileFun` finalize here so the init/renew scalars are
/// bit-exact vs the host f64 anchor.
fn percentile_fun_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    values: &[f32],
    alpha: f64,
) -> Result<f64, ComputeError> {
    let cnt = values.len();
    if cnt <= 1 {
        return Ok(if cnt == 0 { 0.0 } else { f64::from(values[0]) });
    }
    // Descending sort via the device bitonic argsort (index-only): desc[k] =
    // values[perm[k]]. For distinct keys this is uniquely descending (== the host
    // `sorted_descending`); ties inherit the argsort's stable parity.
    let (perm, _after) = bitonic_argsort_on(client, values, false)?;
    let desc: Vec<f64> = perm.iter().map(|&i| f64::from(values[i as usize])).collect();

    let float_pos = (cnt as f64 - 1.0) * (1.0 - alpha);
    let pos = float_pos as i64 + 1;
    if pos < 1 {
        return Ok(desc[0]);
    }
    if pos >= cnt as i64 {
        return Ok(desc[cnt - 1]);
    }
    let pos = pos as usize;
    let bias = float_pos - (pos - 1) as f64;
    let v1 = desc[pos - 1];
    let v2 = desc[pos];
    Ok(v1 - (v1 - v2) * bias)
}

/// The unweighted label mean via the device `reduce_sum_f64_on` primitive divided by
/// `num_data` — the SAME ordered f64 fold as `regression.rs:548-558` (bit-exact
/// deterministic anchor). Empty → `0.0`.
fn label_mean_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    labels_f64: &[f64],
) -> Result<f64, ComputeError> {
    if labels_f64.is_empty() {
        return Ok(0.0);
    }
    let suml = reduce_sum_f64_on(client, labels_f64)?;
    Ok(suml / labels_f64.len() as f64)
}

/// Compute the regression BoostFromScore init scalar for one objective, COMPOSING
/// the existing device primitives (D-08):
/// - L2/Huber/Fair: the label mean via [`reduce_sum_f64_on`] (`regression.rs:548`).
/// - L1: the label median via [`percentile_unweighted_f32_on`] at α = 0.5
///   (`regression.rs:586`).
/// - Quantile: the label percentile at the **f32-rounded** α
///   (`regression.rs:567` — `alpha_` is `score_t`; the result is f32-narrowed).
/// - Poisson: the label non-negativity/finiteness check via [`reduce_min_f64_on`] +
///   [`reduce_sum_f64_on`] (Pitfall 6 — the other objectives skip this) THEN
///   `safe_log(mean)` (`regression.rs:595-605`).
///
/// `alpha` is used only by Quantile (ignored otherwise).
///
/// # Errors
/// [`ComputeError::Runtime`] if a Poisson label is negative or the label-sum is zero
/// (mirroring `regression.rs::check_labels`); propagates the primitive errors.
pub fn boost_from_score_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    objective_tag: u32,
    labels: &[f32],
    alpha: f64,
) -> Result<f64, ComputeError> {
    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();

    if objective_tag == TAG_L1 {
        // The label MEDIAN (PercentileFun at α = 0.5, NOT the mean; regression.rs:586).
        return percentile_fun_on(client, labels, 0.5);
    }
    if objective_tag == TAG_QUANTILE {
        // The label percentile at the f32-rounded α (C++ `alpha_` is score_t and the
        // PercentileFun result is f32-narrowed; regression.rs:567-570).
        let v = percentile_fun_on(client, labels, f64::from(alpha as f32))?;
        return Ok(f64::from(v as f32));
    }
    if objective_tag == TAG_POISSON {
        // Label-domain guard (Pitfall 6): every label >= 0 AND Σ label != 0.
        if !labels.is_empty() {
            let miny = reduce_min_f64_on(client, &labels_f64)?;
            if miny < 0.0 {
                return Err(ComputeError::Runtime {
                    detail: format!(
                        "poisson BoostFromScore: label {miny} is negative (must be >= 0)"
                    ),
                });
            }
            let sumy = reduce_sum_f64_on(client, &labels_f64)?;
            if sumy == 0.0 {
                return Err(ComputeError::Runtime {
                    detail: "poisson BoostFromScore: sum of labels is zero".to_string(),
                });
            }
        }
        return Ok(safe_log(label_mean_on(client, &labels_f64)?));
    }
    // TAG_L2 / TAG_HUBER / TAG_FAIR: the label mean.
    label_mean_on(client, &labels_f64)
}

// =========================================================================
// Task 3: RenewTreeOutput (L1/Quantile per-leaf median) — HOST-ORCHESTRATED
// loop over leaves (Discrepancy 1 — there is NO percentile_device per-leaf
// kernel; the host loop is parity-identical to the CUDA one-block-per-leaf
// result: same sort, same percentile index math).
// =========================================================================

/// RenewTreeOutput for L1/Quantile: recompute each leaf's output as the (unweighted)
/// percentile of that leaf's RESIDUALS. `leaf_residuals[k]` are the per-row
/// `label - score` residuals for the rows in leaf `k` (the C++ `residual_getter`),
/// in the data partition's row order; returns one new leaf value per leaf.
///
/// This is a HOST loop over leaves calling the device-composed [`percentile_fun_on`]
/// (Discrepancy 1 — NO `percentile_device` per-leaf kernel exists; the host loop is
/// parity-identical to the CUDA one-block-per-leaf refit). Mirrors
/// `regression.rs:627-651`:
/// - L1: unweighted residual median (PercentileFun at α = 0.5).
/// - Quantile: unweighted residual percentile at the f32-rounded α (`alpha_` is
///   score_t; the renew result is NOT f32-narrowed — unlike BoostFromScore).
///
/// `alpha` is used only by Quantile (ignored for L1).
///
/// # Errors
/// [`ComputeError::Runtime`] for a non-renew `objective_tag` (only L1/Quantile renew,
/// mirroring `is_renew_tree_output`); propagates the argsort primitive error.
pub fn renew_tree_output_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    objective_tag: u32,
    alpha: f64,
    leaf_residuals: &[Vec<f32>],
) -> Result<Vec<f64>, ComputeError> {
    let percentile_alpha = if objective_tag == TAG_L1 {
        0.5
    } else if objective_tag == TAG_QUANTILE {
        // C++ `alpha_` is score_t (f32); round α through f32 before the index math.
        f64::from(alpha as f32)
    } else {
        return Err(ComputeError::Runtime {
            detail: format!(
                "renew_tree_output: objective_tag {objective_tag} does not renew leaves \
                 (only L1/Quantile have IsRenewTreeOutput() == true)"
            ),
        });
    };

    let mut out = Vec::with_capacity(leaf_residuals.len());
    for residuals in leaf_residuals {
        out.push(percentile_fun_on(client, residuals, percentile_alpha)?);
    }
    Ok(out)
}
