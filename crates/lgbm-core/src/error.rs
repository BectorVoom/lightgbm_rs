//! Structured domain error types at the `lgbm-core` boundary (FND-04).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`. These mirror the C++ `Log::Fatal` sites in
//! `config.cpp`/`config_auto.cpp`, but surfaced as typed `Result` errors
//! instead of fatal logs / panics (Security V5: never panic on user input).
//!
//! The variants below are the minimum set the config plan (Plan 02) extends.

use thiserror::Error;

/// Errors raised while parsing or validating a [`crate`] configuration.
///
/// Maps the C++ config fatals to typed errors:
/// - `Log::Fatal("Parameter %s should be of type int...")` → [`ConfigError::InvalidType`]
/// - `Log::Fatal("Unknown boosting type %s")` (and analogous enum fatals) → [`ConfigError::UnknownValue`]
/// - `CHECK_GT/GE/LE/LT(field, bound)` → [`ConfigError::OutOfRange`]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// A parameter value could not be parsed into the expected primitive type.
    #[error("parameter `{param}` has invalid value `{value}` for its expected type")]
    InvalidType {
        /// The offending parameter name.
        param: String,
        /// The raw value that failed to parse.
        value: String,
    },

    /// An enum-valued parameter received a string outside its known set
    /// (e.g. unknown `boosting`/`objective`/`device_type`/`tree_learner`/`task`).
    #[error("parameter `{param}` has unknown value `{value}`")]
    UnknownValue {
        /// The offending parameter name.
        param: String,
        /// The unknown value.
        value: String,
    },

    /// A parameter value violated a `CHECK_*` range constraint.
    #[error("parameter `{param}` value `{value}` is out of range ({bound})")]
    OutOfRange {
        /// The offending parameter name.
        param: String,
        /// The value that violated the bound.
        value: String,
        /// A human-readable description of the violated bound (e.g. `> 0`, `(0, 1]`).
        bound: String,
    },
}

/// Top-level error type for the `lgbm-core` crate boundary.
///
/// Wraps the more specific domain errors so that callers can match a single
/// error type. Extended by later plans/phases as new subsystems land.
#[derive(Debug, Error)]
pub enum CoreError {
    /// A configuration parsing/validation error.
    #[error(transparent)]
    Config(#[from] ConfigError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_error_constructs_and_displays() {
        let e = ConfigError::InvalidType {
            param: "num_leaves".to_string(),
            value: "abc".to_string(),
        };
        let msg = e.to_string();
        assert!(!msg.is_empty(), "Display string must be non-empty");
        assert!(msg.contains("num_leaves"));
    }

    #[test]
    fn out_of_range_displays_bound() {
        let e = ConfigError::OutOfRange {
            param: "learning_rate".to_string(),
            value: "0".to_string(),
            bound: "> 0".to_string(),
        };
        assert!(e.to_string().contains("> 0"));
    }

    #[test]
    fn core_error_wraps_config_error() {
        let ce = ConfigError::UnknownValue {
            param: "boosting".to_string(),
            value: "nope".to_string(),
        };
        let core: CoreError = ce.into();
        assert!(!core.to_string().is_empty());
    }
}
