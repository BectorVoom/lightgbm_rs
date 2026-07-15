//! Structured domain error types at the `lgbm-model` boundary (Security V5).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`. These mirror the C++ `Log::Fatal`/`CHECK_*` sites
//! in the model-text load + prediction paths, but surfaced as typed `Result`
//! errors instead of fatal logs / panics: the boundary contract is
//! **"never panic on caller input"** (Security V5, RESEARCH §Security Domain).
//!
//! One variant per V5 input-validation class (RESEARCH §Security, the C++
//! `Log::Fatal` sites a model-text reader must defend against):
//! - missing required key / parsed-array-length inconsistency / out-of-bounds
//!   node index while parsing a model `.txt` (`MalformedModel`) — mirrors
//!   `gbdt_model_text.cpp:494,514` and the `tree.cpp` `Tree(const char*, ...)`
//!   `Log::Fatal` sites;
//! - predict-input feature count != `max_feature_idx + 1` (`ShapeMismatch`).
//!
//! The model-text parser + predictor are built in later plans (03-02+); this
//! plan defines the boundary contract up-front so no later code path can panic
//! on caller input.

use thiserror::Error;

/// Errors raised while loading / validating a model `.txt` or running prediction.
///
/// Maps the C++ model-text + predict fatals/checks to typed errors:
/// - missing key / inconsistent array length / OOB node index in the model-text
///   parser → [`ModelError::MalformedModel`] (`gbdt_model_text.cpp:494,514`,
///   `tree.cpp` `Tree(const char*, size_t*)` `Log::Fatal` sites)
/// - predict-input feature count vs `max_feature_idx + 1` mismatch →
///   [`ModelError::ShapeMismatch`]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// The model text is malformed: a required `key=value` line is missing, a
    /// parsed array length is inconsistent with `num_leaves`/`num_cat`, or a
    /// child/node index is out of bounds. Mirrors the C++ `Log::Fatal` sites in
    /// `gbdt_model_text.cpp:494,514` and the `tree.cpp`
    /// `Tree(const char*, size_t*)` parser, surfaced as a typed error so a
    /// hostile model file can never panic the parser.
    #[error("malformed model: {detail}")]
    MalformedModel {
        /// Human-readable description of the structural defect.
        detail: String,
    },

    /// A prediction input's feature count does not match the model's
    /// `max_feature_idx + 1`. Validated before any indexing into the input row
    /// so an out-of-range feature vector returns an error instead of an OOB
    /// read.
    #[error("shape mismatch: {detail}")]
    ShapeMismatch {
        /// Human-readable description of the offending shapes.
        detail: String,
    },

    /// The requested operation is not supported for this model. Mirrors a C++
    /// `Log::Fatal` site that rejects a valid-but-unsupported combination of
    /// model + request (not malformed input) — e.g. `PredictContrib` on a
    /// `linear_tree=true` model is refused outright in C++
    /// (`predictor.hpp:89-90`: contrib is never wired up for linear trees), so
    /// the Rust port refuses it too rather than inventing a SHAP variant with
    /// no C++ counterpart to stay in parity with.
    #[error("unsupported: {detail}")]
    Unsupported {
        /// Human-readable description of the unsupported combination.
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_model_constructs_and_displays() {
        let e = ModelError::MalformedModel {
            detail: "missing required key `num_leaves`".to_string(),
        };
        let msg = e.to_string();
        assert!(!msg.is_empty(), "Display string must be non-empty");
        assert!(msg.contains("malformed model"));
        assert!(msg.contains("num_leaves"));
    }

    #[test]
    fn shape_mismatch_displays_detail() {
        let e = ModelError::ShapeMismatch {
            detail: "input has 27 features, model expects 28".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("shape mismatch"));
        assert!(msg.contains("expects 28"));
    }

    #[test]
    fn errors_are_clone_and_eq() {
        let a = ModelError::MalformedModel {
            detail: "x".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);

        let c = ModelError::ShapeMismatch {
            detail: "y".to_string(),
        };
        assert_ne!(a, c);
    }
}
