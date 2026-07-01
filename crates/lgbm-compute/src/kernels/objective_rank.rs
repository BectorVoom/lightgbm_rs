//! On-device ranking objectives — lambdarank / rank_xendcg grad/hess (§5.4, ODL-08).
//!
//! Owning phase: **19** (ODL-08). Filled by **19-04**.
//!
//! ## What lives here
//! The Rust `#[cube]` port of `src/objective/cuda/cuda_rank_objective.cu`
//! (§5.4 of `docs/cuda-kernel-design.md`) — the per-query
//! `GetGradientsKernel_LambdarankNDCG<…>{,_Sorted}` pairwise-lambda grad/hess and the
//! `GetGradientsKernel_RankXENDCG_{SharedMemory,GlobalMemory}` cross-entropy-NDCG
//! grad/hess (with the per-query `CUDARandom(objective_seed + q)` gamma draw).
//!
//! ## Layout & determinism (block-per-query, D-05 / RESEARCH Pitfall 1)
//! The CUDA reference runs block-per-query-group (`NUM_QUERY_PER_BLOCK = 10`) and
//! accumulates pairwise λ via `atomicAdd_block` — a NON-deterministic add order that
//! is the documented residual (D-05). The cubecl-**cpu f64 fold** is the deterministic
//! anchor (D-08/D-10): both ranking kernels are launched **single-owner**
//! (`CubeDim::new_1d(1)`, `if UNIT_POS == 0`), one unit walking every query in
//! ascending order, each query's rows a disjoint output window — so there is NO
//! cross-query race and the stream is bit-stable (the proven `random.rs` mandate).
//! The block-per-query LAYOUT is faithful; the atomic order is realized as an ordered
//! per-query fold. The GPU f32 mirror (score_t = f32) would use `atomicAdd_block`
//! with the documented order-residual, held to ~1e-6 — NEVER GPU-vs-GPU (def-f8u-01).
//!
//! ## Anchor discipline (D-05)
//! The host `lgbm_objective::rank::{Lambdarank,RankXendcg}` grad/hess (the cpu f64
//! fold; grad/hess ACCUMULATE in f32 exactly as the host `&mut [f32]` buffers do,
//! intermediate λ math in f64) is the parity oracle — NEVER GPU-vs-GPU (D-05 /
//! def-f8u-01). Anchored against the real `lib_lightgbm` 4.6 `lambdarank_gh_iter{1,N}`
//! grad/hess goldens (D-01) and the `rank_xendcg_objseed5` RNG-replay golden.
//!
//! ## Composition (D-08 — do NOT rebuild)
//! - Per-query DESCENDING-score item ranking composes
//!   [`crate::kernels::primitives::bitonic_argsort_items_on`] (segment = query).
//! - Per-item RankXENDCG gamma draws compose
//!   [`crate::kernels::random::draw_next_float_on`] (the bit-identical LCG); the draw
//!   ORDER is load-bearing — per query `q`, `Random(seed + q)` yields one `NextFloat`
//!   per row row-major (rank.rs:482-487).
//!
//! Numerically faithful to `crates/lgbm-objective/src/rank.rs`.
#![allow(unused_imports)]

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::kernels::primitives::bitonic_argsort_items_on;

/// C++ `LambdarankNDCG::_sigmoid_bins = 1024 * 1024` (rank_objective.hpp:365) — the
/// sigmoid lookup-table resolution, replicated so the device grad/hess is bit-exact
/// vs the host `Lambdarank::get_sigmoid` table lookup (NOT a direct-`exp` re-derive:
/// the golden was produced by the rank.rs port, which uses this quantized table).
const SIGMOID_BINS: usize = 1024 * 1024;

/// The CUDA `NUM_QUERY_PER_BLOCK` grid-packing constant (§5.4) — recorded for
/// fidelity; the deterministic f64 anchor realizes it as a single-owner per-query
/// fold (see the module doc). Autotune of the grid is deferred (Phase 21+).
pub const NUM_QUERY_PER_BLOCK: u32 = 10;


/// Build the C++ `ConstructSigmoidTable` (rank_objective.hpp:281-294) host-side so the
/// device kernel does the identical `GetSigmoid` lookup: `min = -50 / sigmoid / 2`,
/// `max = -min`, `factor = BINS / (max - min)`, `table[i] = 1 / (1 + exp(score ·
/// sigmoid))`. Returns `(table, min, max, factor)`.
fn build_sigmoid_table(sigmoid: f64) -> (Vec<f64>, f64, f64, f64) {
    let min_sigmoid_input = -50.0f64 / sigmoid / 2.0;
    let max_sigmoid_input = -min_sigmoid_input;
    let factor = SIGMOID_BINS as f64 / (max_sigmoid_input - min_sigmoid_input);
    let mut table = Vec::with_capacity(SIGMOID_BINS);
    for i in 0..SIGMOID_BINS {
        let score = i as f64 / factor + min_sigmoid_input;
        table.push(1.0 / (1.0 + (score * sigmoid).exp()));
    }
    (table, min_sigmoid_input, max_sigmoid_input, factor)
}

/// Validate a `query_boundaries` prefix-sum against `num_data` at the V5 boundary
/// (T-19-04-02): non-empty, `[0] == 0`, non-decreasing, `last == num_data`. Returns
/// the query count.
fn validate_query_boundaries(qb: &[i32], num_data: usize) -> Result<usize, ComputeError> {
    if qb.len() < 2 || qb[0] != 0 {
        return Err(ComputeError::Runtime {
            detail: "ranking: query_boundaries must be non-empty and start at 0".to_string(),
        });
    }
    for w in qb.windows(2) {
        if w[1] < w[0] {
            return Err(ComputeError::Runtime {
                detail: format!("ranking: query_boundaries not non-decreasing ({} > {})", w[0], w[1]),
            });
        }
    }
    let last = *qb.last().unwrap();
    if last < 0 || last as usize != num_data {
        return Err(ComputeError::LengthMismatch { expected: num_data, actual: last.max(0) as usize });
    }
    Ok(qb.len() - 1)
}

// =========================================================================
// Task 1: LambdaRank-NDCG grad/hess (shared + >2048 `_Sorted`).
// =========================================================================

/// C++ `LambdarankNDCG::GetSigmoid` (rank_objective.hpp:268-279) — the table lookup,
/// device-side. `bins_m1 = SIGMOID_BINS - 1`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn get_sigmoid_dev(
    table: &Array<f64>,
    score: f64,
    min_in: f64,
    max_in: f64,
    factor: f64,
    bins_m1: u32,
) -> f64 {
    // Default covers `score <= min_in` → table[0].
    let mut out = table[0];
    if score >= max_in {
        out = table[bins_m1 as usize];
    } else if score > min_in {
        // idx = floor((score - min) * factor); score in (min,max) → idx in [0, bins).
        let idx = u32::cast_from((score - min_in) * factor);
        out = table[idx as usize];
    }
    out
}

/// The LambdaRank-NDCG per-query pairwise λ/hess fold (rank.rs:274-366), single-owner.
/// `grad`/`hess` are `f32` (the host `&mut [f32]` accumulation buffers); the pairwise
/// λ math runs in `f64`, cast to `f32` on every accumulation — bit-identical to the
/// host anchor's f32 fold. `sorted_idx` holds per-query LOCAL 0-based DESCENDING-score
/// permutations (from `bitonic_argsort_items_on`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn lambdarank_body(
    scores: &Array<f64>,
    labels: &Array<f64>,
    sorted_idx: &Array<i32>,
    query_boundaries: &Array<i32>,
    inverse_max_dcgs: &Array<f64>,
    discounts: &Array<f64>,
    label_gain: &Array<f64>,
    sigmoid_table: &Array<f64>,
    grad: &mut Array<f32>,
    hess: &mut Array<f32>,
    num_queries: u32,
    sigmoid: f64,
    trunc: u32,
    norm: u32,
    norm_eps: f64,
    sig_min: f64,
    sig_max: f64,
    sig_factor: f64,
    sig_bins_m1: u32,
) {
    if UNIT_POS == 0 {
        let mut q = 0u32;
        while q < num_queries {
            let qi = q as usize;
            let start = query_boundaries[qi] as usize;
            let end = query_boundaries[qi + 1] as usize;
            let cnt = end - start;
            // Zero the query's rows (also the cnt<=1 no-pair result).
            let mut z = 0usize;
            while z < cnt {
                grad[start + z] = 0.0f32;
                hess[start + z] = 0.0f32;
                z += 1;
            }
            if cnt >= 2 {
                let inv_max = inverse_max_dcgs[qi];
                // best/worst score for the norm gate (no kMinScore in the spine path).
                let best_local = sorted_idx[start] as usize;
                let best_score = scores[start + best_local];
                let worst_local = sorted_idx[start + cnt - 1] as usize;
                let worst_score = scores[start + worst_local];
                let best_ne_worst = best_score != worst_score;

                let mut sum_lambdas = 0.0f64;
                let cnt_m1 = (cnt as u32) - 1u32;
                let ilim = if trunc < cnt_m1 { trunc } else { cnt_m1 };
                let mut i = 0u32;
                while i < ilim {
                    let mut j = i + 1u32;
                    while j < cnt as u32 {
                        let hi_i_local = sorted_idx[start + i as usize] as usize;
                        let hi_j_local = sorted_idx[start + j as usize] as usize;
                        let lab_i = labels[start + hi_i_local];
                        let lab_j = labels[start + hi_j_local];
                        if lab_i != lab_j {
                            let i_is_high = lab_i > lab_j;
                            let high_rank = select(i_is_high, i, j);
                            let low_rank = select(i_is_high, j, i);
                            let high_local = sorted_idx[start + high_rank as usize] as usize;
                            let low_local = sorted_idx[start + low_rank as usize] as usize;
                            let high_row = start + high_local;
                            let low_row = start + low_local;
                            let high_gain = label_gain[u32::cast_from(labels[high_row]) as usize];
                            let low_gain = label_gain[u32::cast_from(labels[low_row]) as usize];
                            let high_disc = discounts[high_rank as usize];
                            let low_disc = discounts[low_rank as usize];
                            let delta_score = scores[high_row] - scores[low_row];
                            let dcg_gap = high_gain - low_gain;
                            let paired_discount = (high_disc - low_disc).abs();
                            let mut delta_pair_ndcg = dcg_gap * paired_discount * inv_max;
                            if norm == 1u32 && best_ne_worst {
                                delta_pair_ndcg /= norm_eps + delta_score.abs();
                            }
                            let mut p_lambda = get_sigmoid_dev(
                                sigmoid_table, delta_score, sig_min, sig_max, sig_factor, sig_bins_m1,
                            );
                            let mut p_hessian = p_lambda * (1.0f64 - p_lambda);
                            p_lambda *= -sigmoid * delta_pair_ndcg;
                            p_hessian *= sigmoid * sigmoid * delta_pair_ndcg;
                            // f32 accumulation (host `lambdas[.] -= p_lambda as f32`).
                            grad[low_row] -= f32::cast_from(p_lambda);
                            hess[low_row] += f32::cast_from(p_hessian);
                            grad[high_row] += f32::cast_from(p_lambda);
                            hess[high_row] += f32::cast_from(p_hessian);
                            sum_lambdas -= 2.0f64 * p_lambda;
                        }
                        j += 1u32;
                    }
                    i += 1u32;
                }
                // norm rescale (rank.rs:359-365): (v as f64 * factor) as f32.
                if norm == 1u32 && sum_lambdas > 0.0f64 {
                    // log2(x) = ln(x) / ln(2) — cubecl exposes `ln` but not `log2`
                    // (frontend/operation/unary.rs). The <=1-ULP vs libm `log2` scales
                    // the whole query's grad/hess by (1 ± ~1e-16) — negligible vs
                    // ORACLE_TOL (the norm path stays anchor-faithful, D-05).
                    let log2_num = (1.0f64 + sum_lambdas).ln() / 0.693_147_180_559_945_3f64;
                    let norm_factor = log2_num / sum_lambdas;
                    let mut a = 0usize;
                    while a < cnt {
                        grad[start + a] = f32::cast_from(f64::cast_from(grad[start + a]) * norm_factor);
                        hess[start + a] = f32::cast_from(f64::cast_from(hess[start + a]) * norm_factor);
                        a += 1;
                    }
                }
            }
            q += 1u32;
        }
    }
}

/// Shared-memory LambdaRank-NDCG variant (`GetGradientsKernel_LambdarankNDCG`,
/// per-query items fit shared capacity).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lambdarank_kernel_shared_f64(
    scores: &Array<f64>,
    labels: &Array<f64>,
    sorted_idx: &Array<i32>,
    query_boundaries: &Array<i32>,
    inverse_max_dcgs: &Array<f64>,
    discounts: &Array<f64>,
    label_gain: &Array<f64>,
    sigmoid_table: &Array<f64>,
    grad: &mut Array<f32>,
    hess: &mut Array<f32>,
    num_queries: u32,
    sigmoid: f64,
    trunc: u32,
    norm: u32,
    norm_eps: f64,
    sig_min: f64,
    sig_max: f64,
    sig_factor: f64,
    sig_bins_m1: u32,
) {
    lambdarank_body(
        scores, labels, sorted_idx, query_boundaries, inverse_max_dcgs, discounts,
        label_gain, sigmoid_table, grad, hess, num_queries, sigmoid, trunc, norm,
        norm_eps, sig_min, sig_max, sig_factor, sig_bins_m1,
    );
}

/// The `>2048` `_Sorted` (`<MAX_ITEM_GT_1024>`) LambdaRank-NDCG variant
/// (`GetGradientsKernel_LambdarankNDCG_Sorted`): the CUDA path pre-sorts the query
/// globally (shared capacity exceeded). On the host-orchestrated f64 anchor the sort
/// is staged identically by `bitonic_argsort_items_on`, so the accumulation body is
/// shared — the two kernels are asserted to produce identical grad/hess.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lambdarank_kernel_sorted_f64(
    scores: &Array<f64>,
    labels: &Array<f64>,
    sorted_idx: &Array<i32>,
    query_boundaries: &Array<i32>,
    inverse_max_dcgs: &Array<f64>,
    discounts: &Array<f64>,
    label_gain: &Array<f64>,
    sigmoid_table: &Array<f64>,
    grad: &mut Array<f32>,
    hess: &mut Array<f32>,
    num_queries: u32,
    sigmoid: f64,
    trunc: u32,
    norm: u32,
    norm_eps: f64,
    sig_min: f64,
    sig_max: f64,
    sig_factor: f64,
    sig_bins_m1: u32,
) {
    lambdarank_body(
        scores, labels, sorted_idx, query_boundaries, inverse_max_dcgs, discounts,
        label_gain, sigmoid_table, grad, hess, num_queries, sigmoid, trunc, norm,
        norm_eps, sig_min, sig_max, sig_factor, sig_bins_m1,
    );
}

/// Compute LambdaRank-NDCG grad/hess on the f64 cpu anchor.
///
/// `scores` are the f64 raw scores (all rows); `labels` the f32 rank labels;
/// `query_boundaries` the prefix-sum (len `num_queries + 1`). `inverse_max_dcgs`
/// (per query), `discounts` (`discount[rank]`, len >= max query length), and
/// `label_gain` (`gain[label]`) are the init-once DCG tables (owned by
/// `lgbm_metric::DcgCalculator`; supplied by the caller so `lgbm-compute` stays free
/// of the metric-crate edge). Returns `(grad, hess)` as `f32` (`score_t`).
///
/// The per-query DESCENDING-score ranking composes `bitonic_argsort_items_on`
/// (segment = query); the sigmoid uses the replicated 1M-bin lookup table so the
/// result is bit-exact vs the host `Lambdarank::get_gradients` f64 anchor and within
/// `ORACLE_TOL` of the real `lambdarank_gh` golden (the atomic-order residual, D-05).
///
/// # Errors
/// [`ComputeError`] if `labels`/`query_boundaries` are malformed at the V5 boundary.
#[allow(clippy::too_many_arguments)]
pub fn lambdarank_get_gradients_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    query_boundaries: &[i32],
    inverse_max_dcgs: &[f64],
    discounts: &[f64],
    label_gain: &[f64],
    sigmoid: f64,
    norm: bool,
    truncation_level: i32,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    lambdarank_impl(
        client, scores, labels, query_boundaries, inverse_max_dcgs, discounts,
        label_gain, sigmoid, norm, truncation_level, false,
    )
}

/// The `>2048` `_Sorted` (`<MAX_ITEM_GT_1024>`) LambdaRank-NDCG launcher (D-03). Same
/// contract as [`lambdarank_get_gradients_on`]; routes through the `_Sorted` kernel.
/// (True per-query lengths `> BITONIC_SORT_NUM_ELEMENTS` remain the deferred
/// multi-block-sort hardening in `primitives.rs`; the variant is built and validated
/// on the spine corpus to prove the `_Sorted` code path.)
#[allow(clippy::too_many_arguments)]
pub fn lambdarank_get_gradients_sorted_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    query_boundaries: &[i32],
    inverse_max_dcgs: &[f64],
    discounts: &[f64],
    label_gain: &[f64],
    sigmoid: f64,
    norm: bool,
    truncation_level: i32,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    lambdarank_impl(
        client, scores, labels, query_boundaries, inverse_max_dcgs, discounts,
        label_gain, sigmoid, norm, truncation_level, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn lambdarank_impl<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    scores: &[f64],
    labels: &[f32],
    query_boundaries: &[i32],
    inverse_max_dcgs: &[f64],
    discounts: &[f64],
    label_gain: &[f64],
    sigmoid: f64,
    norm: bool,
    truncation_level: i32,
    max_item_gt_1024: bool,
) -> Result<(Vec<f32>, Vec<f32>), ComputeError> {
    let n = scores.len();
    if labels.len() != n {
        return Err(ComputeError::LengthMismatch { expected: n, actual: labels.len() });
    }
    let num_queries = validate_query_boundaries(query_boundaries, n)?;
    if inverse_max_dcgs.len() != num_queries {
        return Err(ComputeError::LengthMismatch { expected: num_queries, actual: inverse_max_dcgs.len() });
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Per-query DESCENDING-score ranking (compose the primitive; segment = query).
    let scores_f32: Vec<f32> = scores.iter().map(|&s| s as f32).collect();
    let mut sorted_idx = bitonic_argsort_items_on(client, &scores_f32, query_boundaries, false)?;
    // The host anchor's std sort is STABLE with an ascending-index tie-break
    // (rank.rs:293-298); `bitonic_argsort_items_on`'s comparator has NO index
    // tie-break, so equal keys resolve network-dependently (e.g. a size-3 tie →
    // reversed). Real ranking scores tie constantly (rows sharing a leaf value), so
    // we CANONICALIZE ties to the reference order: within each query, bitonic has
    // already placed equal scores CONTIGUOUSLY; we reorder each equal-score run to
    // ascending original index (matching the std-stable convention the golden was
    // derived with). Distinct scores → runs of length 1 → no-op, bitonic's order is
    // used verbatim. The kernel reads the ORIGINAL f64 `scores`.
    for q in 0..num_queries {
        let start = query_boundaries[q] as usize;
        let end = query_boundaries[q + 1] as usize;
        let seg = &mut sorted_idx[start..end];
        let mut i = 0usize;
        while i < seg.len() {
            let mut j = i + 1;
            while j < seg.len()
                && scores[start + seg[j] as usize] == scores[start + seg[i] as usize]
            {
                j += 1;
            }
            if j - i > 1 {
                seg[i..j].sort_unstable();
            }
            i = j;
        }
    }

    let (table, sig_min, sig_max, sig_factor) = build_sigmoid_table(sigmoid);
    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();

    let h_scores = client.create_from_slice(f64::as_bytes(scores));
    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_sorted = client.create_from_slice(i32::as_bytes(&sorted_idx));
    let h_qb = client.create_from_slice(i32::as_bytes(query_boundaries));
    let h_inv = client.create_from_slice(f64::as_bytes(inverse_max_dcgs));
    let h_disc = client.create_from_slice(f64::as_bytes(discounts));
    let h_gain = client.create_from_slice(f64::as_bytes(label_gain));
    let h_table = client.create_from_slice(f64::as_bytes(&table));
    let h_grad = client.empty(n * core::mem::size_of::<f32>());
    let h_hess = client.empty(n * core::mem::size_of::<f32>());

    let norm_flag = u32::from(norm);
    let norm_eps = f64::from(0.01f32); // C++ `0.01f + fabs(delta_score)` (f32 literal).
    let sig_bins_m1 = (SIGMOID_BINS - 1) as u32;

    // SAFETY: every input handle is sized exactly to its slice and outlives the launch;
    // the single owner (`UNIT_POS == 0`) walks disjoint per-query row windows in
    // `[0, n)`, indexing `sorted_idx`/`discounts` by within-query rank (`< cnt`),
    // `label_gain` by the (non-negative integer) label, and the sigmoid table by a
    // clamped `[0, BINS)` index. cubecl unsafe confined here (CMP-01 / T-19-04-02).
    unsafe {
        let cc = CubeCount::Static(1, 1, 1);
        let cd = CubeDim::new_1d(1);
        if max_item_gt_1024 {
            lambdarank_kernel_sorted_f64::launch_unchecked(
                client, cc, cd,
                ArrayArg::from_raw_parts(h_scores, n),
                ArrayArg::from_raw_parts(h_labels, n),
                ArrayArg::from_raw_parts(h_sorted, n),
                ArrayArg::from_raw_parts(h_qb, num_queries + 1),
                ArrayArg::from_raw_parts(h_inv, num_queries),
                ArrayArg::from_raw_parts(h_disc, discounts.len()),
                ArrayArg::from_raw_parts(h_gain, label_gain.len()),
                ArrayArg::from_raw_parts(h_table, SIGMOID_BINS),
                ArrayArg::from_raw_parts(h_grad.clone(), n),
                ArrayArg::from_raw_parts(h_hess.clone(), n),
                num_queries as u32,
                sigmoid, truncation_level as u32, norm_flag, norm_eps,
                sig_min, sig_max, sig_factor, sig_bins_m1,
            );
        } else {
            lambdarank_kernel_shared_f64::launch_unchecked(
                client, cc, cd,
                ArrayArg::from_raw_parts(h_scores, n),
                ArrayArg::from_raw_parts(h_labels, n),
                ArrayArg::from_raw_parts(h_sorted, n),
                ArrayArg::from_raw_parts(h_qb, num_queries + 1),
                ArrayArg::from_raw_parts(h_inv, num_queries),
                ArrayArg::from_raw_parts(h_disc, discounts.len()),
                ArrayArg::from_raw_parts(h_gain, label_gain.len()),
                ArrayArg::from_raw_parts(h_table, SIGMOID_BINS),
                ArrayArg::from_raw_parts(h_grad.clone(), n),
                ArrayArg::from_raw_parts(h_hess.clone(), n),
                num_queries as u32,
                sigmoid, truncation_level as u32, norm_flag, norm_eps,
                sig_min, sig_max, sig_factor, sig_bins_m1,
            );
        }
    }

    let grad = f32::from_bytes(&client.read_one_unchecked(h_grad)).to_vec();
    let hess = f32::from_bytes(&client.read_one_unchecked(h_hess)).to_vec();
    Ok((grad, hess))
}

