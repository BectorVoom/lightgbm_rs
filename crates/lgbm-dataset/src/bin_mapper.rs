//! Placeholder — replaced in Task 2 with the bit-exact numeric `BinMapper`
//! port. Declared here so the crate scaffold (Task 1) compiles as a workspace
//! member.

/// C++ `enum BinType` (`bin.h`). Numeric vs categorical binning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinType {
    /// Numerical (continuous) feature.
    Numerical,
    /// Categorical feature.
    Categorical,
}

/// C++ `enum MissingType` (`bin.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingType {
    /// No missing values.
    None,
    /// Zero treated as missing.
    Zero,
    /// NaN treated as missing.
    NaN,
}

/// Placeholder — full numeric port lands in Task 2.
#[derive(Debug, Clone)]
pub struct BinMapper {
    /// Upper bound (inclusive) for each numeric bin (f64).
    pub bin_upper_bound_: Vec<f64>,
}
