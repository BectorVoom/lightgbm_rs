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

pub mod error;
pub mod regression;

pub use error::ObjectiveError;
pub use regression::Objective;

// Re-export the predict-side transform so downstream callers (metric / boosting /
// facade) have a single objective import surface. The canonical owner remains
// `lgbm-model` (Open-Q1); this is a re-export, NOT a re-port.
pub use lgbm_model::ObjectiveKind;
