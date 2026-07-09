//! `lgbm-objective` — the training-side objective layer (gradients/hessians,
//! `BoostFromScore`, `RenewTreeOutput`).
//!
//! Faithful 1:1 port target: `LightGBM/src/objective/` (the header-only
//! `*_objective.hpp` implementations + `objective_function.cpp` factory). This
//! crate owns the **training** side only:
//! - `GetGradients(score, gradients, hessians)` — per-row grad/hess (the
//!   GPU-relevant hot path, RESEARCH §"Objective Formulas").
//! - `BoostFromScore` — the initial-score baseline (mean / median / etc.).
//! - `RenewTreeOutput` — the leaf-output renewal hook (regression_l1 median, lands
//!   in 06-03).
//!
//! The **predict-side** transform (`ConvertOutput`, e.g. `softmax`) already lives
//! in [`lgbm_model::ObjectiveKind::convert_output`] and is REUSED here (Open-Q1):
//! this crate does NOT re-port it.
//!
//! Wave-0 scaffold (06-01): this plan creates the compiling skeleton + the
//! [`error`] boundary only. The enum-dispatch objective factory (mirroring the
//! C++ string-keyed `CreateObjectiveFunction`) and the per-objective math land in
//! 06-02 (L2 spine) and 06-03+ (regression_l1 / binary / multiclass / custom).

/// spike-068 grain guard: the minimum row count at which `get_gradients` switches
/// from the serial per-row loop to the rayon-parallel one. Below this floor the
/// rayon fork/join overhead outweighs the per-row `exp()` work (tiny corpora keep
/// the serial loop); at/above it the embarrassingly-parallel independent-per-row
/// grad/hess computation wins (measured 5.1× on 1M×30 binary). Bit-exactness is
/// invariant across the branch: every output element is a pure function of its own
/// row's inputs written to a disjoint slot, so partitioning the row range changes
/// no float (threat T-sk7-02).
pub(crate) const GRAD_HESS_PAR_MIN_ROWS: usize = 50_000;

/// spike-068 serial/parallel dispatch shared by the independent-per-row
/// `get_gradients` implementations (binary / regression / xentropy). `f(score[i],
/// label[i]) -> (grad[i], hess[i])` is a pure per-row function written to disjoint
/// slots, so the rayon partition is bit-exact vs the serial loop (threat T-sk7-02):
/// no reduction, no cross-row order, so partitioning the row range changes no float.
/// Rayon at/above [`GRAD_HESS_PAR_MIN_ROWS`], serial below it. Callers pre-validate
/// the four slice lengths, so both paths are in-bounds.
#[inline]
pub(crate) fn apply_grad_hess<F>(
    score: &[f64],
    label: &[f32],
    gradients: &mut [f32],
    hessians: &mut [f32],
    f: F,
) where
    F: Fn(f64, f32) -> (f32, f32) + Sync + Send,
{
    use rayon::prelude::*;
    let n = score.len();
    if n >= GRAD_HESS_PAR_MIN_ROWS {
        gradients
            .par_iter_mut()
            .zip(hessians.par_iter_mut())
            .zip(score.par_iter().zip(label.par_iter()))
            .for_each(|((g, h), (&s, &l))| {
                let (gv, hv) = f(s, l);
                *g = gv;
                *h = hv;
            });
    } else {
        for i in 0..n {
            let (gv, hv) = f(score[i], label[i]);
            gradients[i] = gv;
            hessians[i] = hv;
        }
    }
}

pub mod binary;
pub mod custom;
pub mod error;
pub mod multiclass;
pub mod percentile;
pub mod rank;
pub mod regression;
pub mod xentropy;

pub use binary::Binary;
pub use custom::CustomObjective;
pub use error::ObjectiveError;
pub use multiclass::{MulticlassOva, MulticlassSoftmax};
pub use rank::{Lambdarank, RankXendcg};
pub use regression::Objective;
pub use xentropy::{Xentropy, XentropyKind};

// Re-export the predict-side transform so downstream callers (metric / boosting /
// facade) have a single objective import surface. The canonical owner remains
// `lgbm-model` (Open-Q1); this is a re-export, NOT a re-port.
pub use lgbm_model::ObjectiveKind;
