//! The ORA-01 abs-diff oracle comparator at the locked `~1e-6` tolerance (D-02).
//!
//! # Tolerance scope
//!
//! This `~1e-6` **oracle** tolerance is distinct from `lgbm_core::types::K_EPSILON`
//! (`1e-15`), which is an *algorithm* constant, not a comparison tolerance.
//!
//! The tolerance applies **only to float comparisons**. Integer draws and
//! exact-bit `f32` RNG draws are compared for *exact* equality (see the RNG
//! parity test), not within a tolerance.

use std::fmt;

/// The locked oracle comparison tolerance (D-02): `~1e-6` absolute, f32.
pub const ORACLE_TOL: f32 = 1e-6;

/// Reports the first index at which two slices diverged beyond tolerance, or a
/// length mismatch. Downstream parity suites use the index to localize a
/// divergence within the randomized golden set (D-14).
#[derive(Debug, Clone, PartialEq)]
pub enum Mismatch {
    /// The two slices had different lengths.
    LengthMismatch {
        /// Length of the Rust-produced slice.
        rust_len: usize,
        /// Length of the C++ golden slice.
        cpp_len: usize,
    },
    /// A value at `index` exceeded the tolerance.
    ValueMismatch {
        /// First offending index.
        index: usize,
        /// Rust value at that index.
        rust: f32,
        /// C++ golden value at that index.
        cpp: f32,
        /// Absolute difference observed.
        abs_diff: f32,
        /// Tolerance that was exceeded.
        tol: f32,
    },
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mismatch::LengthMismatch { rust_len, cpp_len } => write!(
                f,
                "length mismatch: rust={rust_len}, cpp={cpp_len}"
            ),
            Mismatch::ValueMismatch {
                index,
                rust,
                cpp,
                abs_diff,
                tol,
            } => write!(
                f,
                "value mismatch at index {index}: rust={rust}, cpp={cpp}, abs_diff={abs_diff} > tol={tol}"
            ),
        }
    }
}

impl std::error::Error for Mismatch {}

/// Returns `true` iff `|a - b| <= tol`.
pub fn abs_diff_within(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Compares two float slices element-wise within `tol`, returning the first
/// offending index (or a length mismatch) on failure.
pub fn compare_within(rust: &[f32], cpp: &[f32], tol: f32) -> Result<(), Mismatch> {
    if rust.len() != cpp.len() {
        return Err(Mismatch::LengthMismatch {
            rust_len: rust.len(),
            cpp_len: cpp.len(),
        });
    }
    for (index, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        let abs_diff = (r - c).abs();
        if abs_diff > tol {
            return Err(Mismatch::ValueMismatch {
                index,
                rust: r,
                cpp: c,
                abs_diff,
                tol,
            });
        }
    }
    Ok(())
}
