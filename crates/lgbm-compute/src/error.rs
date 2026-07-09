//! Structured domain error types at the `lgbm-compute` boundary (FND-04, CMP-01).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`, and never pull the app-layer error crate into this
//! library crate (that crate is for app/high-level layers only). These variants surface
//! the `Backend`-boundary input-validation contract (Security V5, threat
//! T-04-01): an out-of-range bin index, an inconsistent array length, an
//! unavailable device capability, or a cubecl launch failure become typed
//! `Result` errors instead of panics / undefined behaviour.

use thiserror::Error;

/// Errors raised at the `lgbm-compute` `Backend` boundary.
///
/// Mirrors the Phase-2 `DatasetError` discipline: every kernel input is
/// validated to a typed error before any `unsafe` launch, so a malformed
/// caller request can never reach undefined behaviour in a kernel.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ComputeError {
    /// A per-row bin index fell outside the valid `[0, num_bin)` range.
    ///
    /// Security V5 / threat T-04-01: an out-of-bounds bin would index past the
    /// `out` histogram allocation (`2 * num_bin` f64 cells), so it is rejected
    /// before launch rather than producing UB.
    #[error("bin index `{bin}` at row `{row}` is out of range (num_bin = {num_bin})")]
    BinIndexOutOfRange {
        /// The offending row index.
        row: usize,
        /// The out-of-range bin value.
        bin: u32,
        /// The exclusive upper bound (number of bins).
        num_bin: u32,
    },

    /// Two kernel input/output arrays had inconsistent lengths.
    ///
    /// Threat T-04-01: e.g. `grad.len() != hess.len()`, `binned.len()` mismatch,
    /// or `out.len() != 2 * num_bin`. Validated before launch so the
    /// handle/length correspondence invariant of `ArrayArg::from_raw_parts`
    /// holds.
    #[error("array length mismatch: expected `{expected}`, got `{actual}`")]
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },

    /// A device capability required by the selected reduce path was absent.
    ///
    /// Raised by the startup capability gate (CMP-04) when a divergent feature
    /// (`Plane::Ops`, f64 storage, or f32 atomic-add) is required but the active
    /// runtime does not advertise it.
    #[error("required device capability unavailable: `{feature}`")]
    CapabilityUnavailable {
        /// A human-readable name of the missing capability.
        feature: String,
    },

    /// A cubecl runtime / kernel-launch failure, wrapped opaquely.
    ///
    /// Keeps cubecl error types out of the public boundary (CMP-01) while still
    /// surfacing a diagnostic detail string.
    #[error("compute runtime error: {detail}")]
    Runtime {
        /// A human-readable description of the underlying runtime failure.
        detail: String,
    },

    /// A non-finite (NaN or ±inf) value was produced on the opt-in on-device grow
    /// path — a seeded root grad/hess sum or an emitted leaf value.
    ///
    /// OCX-04 (spike-072 method takeaway (d)): GBDT's C++-faithful no-split
    /// pop-loop silently turns ONE NaN boosting iteration into a truncated model
    /// (the observed 13-tree collapse). This tripwire makes the on-device arm FAIL
    /// LOUDLY at first onset instead — the guard lives on the compute/grow return
    /// path, NEVER in the boosting pop-loop (whose bit-faithful semantics stay
    /// unchanged). The `detail` names the offending quantity (leaf index / value or
    /// the root sums) so the failure is diagnosable at the seam.
    #[error("non-finite value on the on-device grow path: {detail}")]
    NonFinite {
        /// A human-readable description naming the non-finite quantity.
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_index_out_of_range_displays_values() {
        let e = ComputeError::BinIndexOutOfRange {
            row: 7,
            bin: 300,
            num_bin: 256,
        };
        let msg = e.to_string();
        assert!(msg.contains("300"), "must name the offending bin");
        assert!(msg.contains("7"), "must name the offending row");
        assert!(msg.contains("256"), "must name num_bin");
    }

    #[test]
    fn length_mismatch_displays_values() {
        let e = ComputeError::LengthMismatch {
            expected: 10,
            actual: 9,
        };
        let msg = e.to_string();
        assert!(msg.contains("10") && msg.contains('9'));
    }

    #[test]
    fn capability_unavailable_names_feature() {
        let e = ComputeError::CapabilityUnavailable {
            feature: "Plane::Ops".to_string(),
        };
        assert!(e.to_string().contains("Plane::Ops"));
    }

    #[test]
    fn non_finite_names_quantity() {
        // OCX-04: the on-device tripwire error must name the offending quantity so a
        // NaN/±inf failure is diagnosable at the learner seam (spike-072 takeaway (d)).
        let e = ComputeError::NonFinite {
            detail: "leaf 7 value is inf".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("non-finite"), "must self-identify as a non-finite failure");
        assert!(msg.contains("leaf 7") && msg.contains("inf"), "must name the offending quantity");
    }
}
