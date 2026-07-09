//! `custom` objective (OBJ-02 / D-04) — the user-supplied grad/hess closure that
//! mirrors the Python `fobj` contract.
//!
//! Faithful-mirror citations:
//! - `LightGBM/src/objective/objective_function.cpp` — `CreateObjectiveFunction`
//!   returns `nullptr` for `objective = "custom"` (there is no built-in objective).
//! - `LightGBM/src/boosting/gbdt.cpp:355-372` — when `objective_function_ ==
//!   nullptr`, `TrainOneIter(gradients, hessians)` is called with the
//!   USER-SUPPLIED grad/hess and BoostFromAverage is skipped (`obj == null`).
//! - Python `fobj(preds, train_data) -> (grad, hess)` — `preds` is the current
//!   RAW accumulated margin (f64 `score_`), shape `num_data * num_class`.
//!
//! ## D-04 Rust mapping
//! The closure is invoked by the GBDT loop at step (2) IN PLACE of
//! `Objective::get_gradients`, passing the current `score_` as `&[f64]` (the raw
//! accumulated margin). Its returned `(grad, hess)` replace the built-in formula's
//! output. `boost_from_average` is forced OFF for custom (mirroring the C++ `obj ==
//! null` BoostFromAverage skip).
//!
//! **`preds` precision (DEVIATION from RESEARCH D-04):** RESEARCH said to pass the
//! score "cast to f32 (mirror Python's f32 preds)". But LightGBM 4.6's Python
//! wrapper actually passes the raw margin as **f64** to a custom objective
//! (verified empirically — `preds.dtype == float64`). Casting to f32 first breaks
//! bit-exact parity with the native cross-anchor (a custom L2-equivalent closure
//! must reproduce the native regression(L2) g/h, which subtracts in f64). So this
//! port passes `&[f64]` preds — the real-binary behavior is authoritative over the
//! inferred default (06-03 Task 3, Rule 1).
//!
//! The D-04 contract names `Fn(preds: &[f32], dataset: &Dataset) -> (Vec<f32>,
//! Vec<f32>)`. Here the closure closes over whatever dataset metadata it needs
//! (labels/weights) exactly as Python's `fobj` closes over `train_data` — so the
//! stored closure takes only `preds` and returns `(grad, hess)`. This keeps the
//! boosting loop decoupled from `&Dataset` (which it does not hold) while staying
//! faithful to the `fobj` contract: the Phase-8 PyO3 binding wraps the
//! Python callable, captures the `Dataset`, and exposes exactly this `preds ->
//! (grad, hess)` shape to the loop.
//!
//! ## Security V5 (T-06-03-01)
//! The closure is UNTRUSTED — its returned `(grad, hess)` lengths are validated to
//! equal `num_data * num_class` at the loop boundary. A wrong-length return is a
//! typed [`ObjectiveError::LengthMismatch`], NEVER an out-of-bounds index or a
//! panic.

use crate::error::ObjectiveError;

/// A user-supplied custom objective: a closure mapping the current raw scores
/// (`preds`, f64) to per-row `(gradients, hessians)`.
///
/// Construct with [`CustomObjective::new`]; invoke via
/// [`CustomObjective::get_gradients`], which validates the returned lengths.
pub struct CustomObjective<'a> {
    closure: Box<dyn Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + 'a>,
}

impl<'a> CustomObjective<'a> {
    /// Wrap a `preds -> (grad, hess)` closure (the D-04 `fobj` mapping). `preds` is
    /// the raw accumulated margin (`&[f64]`, matching LightGBM 4.6's Python
    /// wrapper). The closure closes over any dataset metadata (labels/weights) it
    /// needs, exactly like Python's `fobj` closes over `train_data`.
    pub fn new<F>(closure: F) -> Self
    where
        F: Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + 'a,
    {
        Self {
            closure: Box::new(closure),
        }
    }

    /// Invoke the closure on the current raw scores and write the validated
    /// `(grad, hess)` into the caller's buffers (mirroring
    /// `Objective::get_gradients`'s in-place contract).
    ///
    /// `score` is the f64 accumulated raw score, passed to the closure verbatim as
    /// `&[f64]` (LightGBM 4.6 hands a custom objective f64 preds). The closure's
    /// returned `(grad, hess)` MUST each have length `score.len()` (== `num_data *
    /// num_class`).
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] (V5, T-06-03-01) if the closure returns a
    /// grad or hess vector whose length differs from `num_data` — surfaced as a
    /// typed error, never an OOB index or panic.
    pub fn get_gradients(
        &self,
        score: &[f64],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let n = score.len();
        let (grad, hess) = (self.closure)(score);
        if grad.len() != n {
            return Err(ObjectiveError::LengthMismatch {
                expected: n,
                actual: grad.len(),
            });
        }
        if hess.len() != n {
            return Err(ObjectiveError::LengthMismatch {
                expected: n,
                actual: hess.len(),
            });
        }
        if gradients.len() != n || hessians.len() != n {
            return Err(ObjectiveError::LengthMismatch {
                expected: n,
                actual: gradients.len().min(hessians.len()),
            });
        }
        gradients.copy_from_slice(&grad);
        hessians.copy_from_slice(&hess);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_closure_is_invoked_in_place_of_get_gradients() {
        // A custom L2-equivalent fobj: grad = score - label, hess = 1 (the
        // DELIBERATE cross-anchor choice — same g/h as native regression(L2)).
        let labels = [1.0f32, 2.0, 3.0];
        let obj = CustomObjective::new(|preds: &[f64]| {
            // f64-subtract then f32-cast (mirroring the native L2 g/h op order).
            let grad: Vec<f32> = preds
                .iter()
                .zip(labels.iter())
                .map(|(&p, &l)| (p - l as f64) as f32)
                .collect();
            let hess = vec![1.0f32; preds.len()];
            (grad, hess)
        });
        let score = [4.0f64, 4.0, 4.0];
        let mut grad = [0.0f32; 3];
        let mut hess = [0.0f32; 3];
        obj.get_gradients(&score, &mut grad, &mut hess).unwrap();
        assert_eq!(grad, [3.0f32, 2.0, 1.0]); // 4-1, 4-2, 4-3
        assert_eq!(hess, [1.0f32; 3]);
    }

    #[test]
    fn custom_wrong_length_grad_is_typed_error_not_panic() {
        // Returns a too-short grad vector — must be a typed error, never an OOB
        // index / panic (T-06-03-01, Security V5).
        let obj = CustomObjective::new(|_preds: &[f64]| (vec![0.0f32; 1], vec![1.0f32; 3]));
        let score = [1.0f64, 2.0, 3.0];
        let mut grad = [0.0f32; 3];
        let mut hess = [0.0f32; 3];
        let err = obj.get_gradients(&score, &mut grad, &mut hess).unwrap_err();
        assert!(matches!(err, ObjectiveError::LengthMismatch { .. }));
    }

    #[test]
    fn custom_wrong_length_hess_is_typed_error() {
        let obj = CustomObjective::new(|_preds: &[f64]| (vec![0.0f32; 3], vec![1.0f32; 2]));
        let score = [1.0f64, 2.0, 3.0];
        let mut grad = [0.0f32; 3];
        let mut hess = [0.0f32; 3];
        let err = obj.get_gradients(&score, &mut grad, &mut hess).unwrap_err();
        assert!(matches!(err, ObjectiveError::LengthMismatch { .. }));
    }
}
