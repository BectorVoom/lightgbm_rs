//! Numerical type aliases and constants ported 1:1 from the C++ reference
//! `LightGBM/include/LightGBM/meta.h`.
//!
//! This is the **f32 single-precision numerical contract** (D-01/D-02/D-03):
//! scores, gradients, hessians, and labels are `f32`, matching the C++ defaults
//! (`score_t`/`label_t` = `float`; `SCORE_T_USE_DOUBLE`/`LABEL_T_USE_DOUBLE` are
//! NOT defined). Do NOT introduce f64 score/label aliases — out-precisioning the
//! f32 reference breaks parity rather than improving it.

/// `typedef int32_t data_size_t;` — row/sample index type (signed by design).
pub type DataSizeT = i32;

/// `typedef float score_t;` — scores and gradients (f32 contract, D-01).
pub type ScoreT = f32;

/// `typedef float label_t;` — labels and weights (f32 contract, D-01).
pub type LabelT = f32;

/// `typedef int32_t comm_size_t;` — network/communication sizes.
pub type CommSizeT = i32;

/// `const score_t kMinScore = -std::numeric_limits<score_t>::infinity();`
pub const K_MIN_SCORE: f32 = f32::NEG_INFINITY;

/// `const score_t kMaxScore = std::numeric_limits<score_t>::infinity();`
pub const K_MAX_SCORE: f32 = f32::INFINITY;

/// `const score_t kEpsilon = 1e-15f;` — an algorithm constant. NOTE: this is
/// distinct from the oracle comparison tolerance (`~1e-6`), which lives in the
/// oracle-harness crate.
pub const K_EPSILON: f32 = 1e-15;

/// `const double kZeroThreshold = 1e-35f;` — the literal is written `1e-35f`
/// but stored as a `double` in C++, so the Rust analog is `f64`.
pub const K_ZERO_THRESHOLD: f64 = 1e-35;

/// `#define NO_SPECIFIC (-1)`
pub const NO_SPECIFIC: i32 = -1;

/// `const int kAlignedSize = 32;`
pub const K_ALIGNED_SIZE: i32 = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_and_label_are_f32() {
        // ScoreT/LabelT must be f32 (size + behavior); DataSizeT/CommSizeT i32.
        assert_eq!(core::mem::size_of::<ScoreT>(), 4);
        assert_eq!(core::mem::size_of::<LabelT>(), 4);
        assert_eq!(core::mem::size_of::<DataSizeT>(), 4);
        assert_eq!(core::mem::size_of::<CommSizeT>(), 4);
        // Type-level proof that the aliases resolve to f32/i32.
        let _s: ScoreT = 1.0f32;
        let _l: LabelT = 1.0f32;
        let _d: DataSizeT = 1i32;
        let _c: CommSizeT = 1i32;
    }

    #[test]
    fn meta_constants_match_cpp() {
        assert_eq!(K_EPSILON, 1e-15_f32);
        assert_eq!(K_ZERO_THRESHOLD, 1e-35_f64);
        assert_eq!(K_MIN_SCORE, f32::NEG_INFINITY);
        assert_eq!(K_MAX_SCORE, f32::INFINITY);
        assert_eq!(NO_SPECIFIC, -1_i32);
        assert_eq!(K_ALIGNED_SIZE, 32_i32);
    }
}
