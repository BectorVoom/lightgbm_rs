//! Structured domain error types at the `lgbm-dataset` boundary (FND-04, Security V5).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`. These mirror the C++ `Log::Fatal`/`CHECK_*` sites in
//! the ingestion path (`c_api.cpp`, `dataset.cpp`, `dataset_loader.cpp`), but
//! surfaced as typed `Result` errors instead of fatal logs / panics: the boundary
//! contract is **"never panic on caller input"** (Security V5, RESEARCH §Security).
//!
//! One variant per V5 input-validation class (RESEARCH §Security V5 row 584):
//! - matrix/vector dims & label/weight length (`ShapeMismatch`),
//! - CSR/CSC `indptr` monotonicity & `indices` bounds (`MalformedSparse`),
//! - query boundaries sum == num_rows (`QueryBoundary`),
//! - non-negative `max_bin` / `min_data_in_bin` (`InvalidConfig`).
//!
//! Most variants are exercised by later ingestion plans (`from_mat`/`from_csr`/
//! `from_csc`), but the full boundary contract exists now so no later code path
//! can panic on caller input.

use thiserror::Error;

/// Errors raised while ingesting / validating caller-provided dataset input.
///
/// Maps the C++ ingestion fatals/checks to typed errors:
/// - matrix dims / label-length `CHECK`s → [`DatasetError::ShapeMismatch`]
/// - CSR/CSC `indptr`/`indices` validation → [`DatasetError::MalformedSparse`]
/// - query-boundary sum `CHECK` → [`DatasetError::QueryBoundary`]
/// - negative `max_bin`/`min_data_in_bin` `CHECK_GT` → [`DatasetError::InvalidConfig`]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// Matrix/vector shape mismatch (e.g. `labels.len() != num_rows`, weight
    /// length mismatch, or matrix `nrow * ncol` inconsistent with the data
    /// buffer length). Validated before any indexing into caller buffers.
    #[error("shape mismatch: {detail}")]
    ShapeMismatch {
        /// Human-readable description of the offending shapes.
        detail: String,
    },

    /// Malformed CSR/CSC sparse matrix: non-monotone `indptr`, an `indptr`
    /// entry out of range, or an `indices` value outside `[0, ncols)`
    /// (CSR) / `[0, nrows)` (CSC). Prevents an out-of-bounds read on
    /// attacker-controlled sparse structure.
    #[error("malformed sparse matrix: {detail}")]
    MalformedSparse {
        /// Human-readable description of the structural defect.
        detail: String,
    },

    /// Query/group boundaries do not partition the rows: the sum of group
    /// sizes (or the final boundary) does not equal `num_rows`, or the
    /// boundaries are not monotone non-decreasing.
    #[error("invalid query boundary: {detail}")]
    QueryBoundary {
        /// Human-readable description of the boundary defect.
        detail: String,
    },

    /// An out-of-range or otherwise invalid binning configuration value, e.g.
    /// a negative or zero `max_bin`, or a negative `min_data_in_bin`. C++ uses
    /// `CHECK_GT(max_bin, 0)`; we return this rather than aborting.
    #[error("invalid config: {detail}")]
    InvalidConfig {
        /// Human-readable description of the invalid configuration.
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_constructs_and_displays() {
        let e = DatasetError::ShapeMismatch {
            detail: "labels.len()=3 != num_rows=4".to_string(),
        };
        let msg = e.to_string();
        assert!(!msg.is_empty(), "Display string must be non-empty");
        assert!(msg.contains("shape mismatch"));
        assert!(msg.contains("num_rows=4"));
    }

    #[test]
    fn malformed_sparse_displays_detail() {
        let e = DatasetError::MalformedSparse {
            detail: "indptr not monotone at k=2".to_string(),
        };
        assert!(e.to_string().contains("indptr not monotone"));
    }

    #[test]
    fn query_boundary_displays_detail() {
        let e = DatasetError::QueryBoundary {
            detail: "group sizes sum to 9, expected 10".to_string(),
        };
        assert!(e.to_string().contains("query boundary"));
    }

    #[test]
    fn invalid_config_displays_detail() {
        let e = DatasetError::InvalidConfig {
            detail: "max_bin must be > 0, got 0".to_string(),
        };
        assert!(e.to_string().contains("invalid config"));
        assert!(e.to_string().contains("max_bin"));
    }

    #[test]
    fn errors_are_clone_and_eq() {
        let a = DatasetError::InvalidConfig {
            detail: "x".to_string(),
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
