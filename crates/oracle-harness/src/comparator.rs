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
    /// An exact-equality comparison diverged at `index`. Used by the binning
    /// goldens (bin-index vectors, f64-bit boundary arrays, storage bytes) which
    /// are compared bit-exact, NOT within the `~1e-6` oracle tolerance.
    ///
    /// The diverging values are rendered as strings so the same variant serves
    /// `u32`, raw-`f64`-bits, and `u8` comparisons with one first-divergence shape.
    ExactMismatch {
        /// First offending index.
        index: usize,
        /// Rust value at that index (rendered).
        rust: String,
        /// C++ golden value at that index (rendered).
        cpp: String,
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
            Mismatch::ExactMismatch { index, rust, cpp } => write!(
                f,
                "exact mismatch at index {index}: rust={rust}, cpp={cpp}"
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

// ---------------------------------------------------------------------------
// Exact-equality comparators for the binning goldens (Phase 2).
//
// Binning is compared bit-exact — bin-index vectors, the f64 `bin_upper_bound_`
// array (via `.to_bits()`), and storage-layout bytes — NOT within the `~1e-6`
// oracle tolerance. Each reports the first divergence as
// [`Mismatch::ExactMismatch`] so a parity failure localizes to one index.
// ---------------------------------------------------------------------------

/// Compares two `u32` slices (e.g. per-row bin-index vectors) for exact
/// equality, returning the first divergence (or a length mismatch).
pub fn compare_exact_u32(rust: &[u32], cpp: &[u32]) -> Result<(), Mismatch> {
    if rust.len() != cpp.len() {
        return Err(Mismatch::LengthMismatch {
            rust_len: rust.len(),
            cpp_len: cpp.len(),
        });
    }
    for (index, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        if r != c {
            return Err(Mismatch::ExactMismatch {
                index,
                rust: r.to_string(),
                cpp: c.to_string(),
            });
        }
    }
    Ok(())
}

/// Compares two `f64` slices (e.g. `bin_upper_bound_`) **bit-exact** by raw IEEE
/// bit pattern (`f64::to_bits`), NOT within a tolerance. This is the correct
/// comparison for deterministically-computed bin boundaries: any 1-ULP drift is
/// a real divergence. `+inf` and `NaN` compare by their canonical bit patterns;
/// a `NaN` bin sentinel matches another `NaN` only if the bits are identical
/// (both sides emit the same `f64::NAN` / C++ `NaN`).
pub fn compare_exact_f64_bits(rust: &[f64], cpp: &[f64]) -> Result<(), Mismatch> {
    if rust.len() != cpp.len() {
        return Err(Mismatch::LengthMismatch {
            rust_len: rust.len(),
            cpp_len: cpp.len(),
        });
    }
    for (index, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        if r.to_bits() != c.to_bits() {
            return Err(Mismatch::ExactMismatch {
                index,
                rust: format!("{r} (bits=0x{:016X})", r.to_bits()),
                cpp: format!("{c} (bits=0x{:016X})", c.to_bits()),
            });
        }
    }
    Ok(())
}

/// Compares two byte slices (e.g. `DenseBin`/`SparseBin` storage layout) for
/// exact equality, returning the first divergence (or a length mismatch). Used
/// by later storage-layout goldens.
pub fn compare_exact_bytes(rust: &[u8], cpp: &[u8]) -> Result<(), Mismatch> {
    if rust.len() != cpp.len() {
        return Err(Mismatch::LengthMismatch {
            rust_len: rust.len(),
            cpp_len: cpp.len(),
        });
    }
    for (index, (&r, &c)) in rust.iter().zip(cpp.iter()).enumerate() {
        if r != c {
            return Err(Mismatch::ExactMismatch {
                index,
                rust: format!("0x{r:02X}"),
                cpp: format!("0x{c:02X}"),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod exact_tests {
    use super::*;

    #[test]
    fn exact_u32_matches_and_reports_divergence() {
        assert!(compare_exact_u32(&[0, 1, 2], &[0, 1, 2]).is_ok());
        let err = compare_exact_u32(&[0, 9, 2], &[0, 1, 2]).unwrap_err();
        match err {
            Mismatch::ExactMismatch { index, .. } => assert_eq!(index, 1),
            other => panic!("expected ExactMismatch, got {other:?}"),
        }
        assert!(matches!(
            compare_exact_u32(&[0, 1], &[0, 1, 2]),
            Err(Mismatch::LengthMismatch { .. })
        ));
    }

    #[test]
    fn exact_f64_bits_distinguishes_one_ulp() {
        let a = 1.0_f64;
        let b = a.next_up(); // 1 ULP above
        assert!(compare_exact_f64_bits(&[a], &[a]).is_ok());
        let err = compare_exact_f64_bits(&[a], &[b]).unwrap_err();
        assert!(matches!(err, Mismatch::ExactMismatch { index: 0, .. }));
        // +inf compares equal by bits.
        assert!(compare_exact_f64_bits(&[f64::INFINITY], &[f64::INFINITY]).is_ok());
    }

    #[test]
    fn exact_bytes_matches_and_reports_divergence() {
        assert!(compare_exact_bytes(&[0xAB, 0xCD], &[0xAB, 0xCD]).is_ok());
        let err = compare_exact_bytes(&[0xAB, 0x00], &[0xAB, 0xCD]).unwrap_err();
        assert!(matches!(err, Mismatch::ExactMismatch { index: 1, .. }));
    }
}
