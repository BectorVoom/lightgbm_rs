//! Bit-for-bit port of the `Metadata` owned vectors + query-weight derivation
//! from `LightGBM/src/io/metadata.cpp` (`CalculateQueryWeights` 742-756,
//! `SetQueriesFromIterator` 522-549, `FinishLoad` 779-781).
//!
//! Follows the `lgbm-core/src/config/mod.rs` flat-struct discipline: ONE flat
//! `pub struct` whose fields are named after the C++ `Metadata` members, each
//! doc-commented with its C++ type and the file/line it ports.
//!
//! # The f32 numerical contract (do NOT widen)
//!
//! C++ stores labels/weights as `label_t` (= `float`, `types.rs` `LabelT`) and
//! `init_score` as `double`. Query weights are computed in `label_t` arithmetic:
//! the per-query sum is accumulated in `float` and divided by the (int) group
//! size, which converts to `float`. Widening labels/weights to f64 would
//! **break** parity (over-precision), so this struct uses `LabelT` (f32) for
//! `label`/`weights`/`query_weights` and `f64` only for `init_score`, exactly as
//! C++ does (`types.rs` contract; RESEARCH §Security V5 / DAT-06).
//!
//! # Validation at the boundary (Security V5)
//!
//! All caller input is validated at construction — never a panic on caller input:
//! - `labels.len() == num_rows` (else [`DatasetError::ShapeMismatch`]);
//! - `weights.len() in {0, num_rows}` (else [`DatasetError::ShapeMismatch`]);
//! - `init_score.len() in {0, num_rows}` (else [`DatasetError::ShapeMismatch`]);
//! - `query_boundaries` monotone non-decreasing, starting at `0`, ending at
//!   `num_rows` (else [`DatasetError::QueryBoundary`]).
//!
//! The C++ `SetQueriesFromIterator` takes per-group *counts* and builds the
//! prefix-sum `query_boundaries_`; it `Log::Fatal`s when the count sum differs
//! from `#data`. We accept the already-prefix-summed `query_boundaries` directly
//! (matching the in-memory C++ `query_boundaries_` representation) and surface the
//! equivalent check (`last == num_rows`) as a typed error.

use lgbm_core::types::{DataSizeT, LabelT};

use crate::error::DatasetError;

/// Owned per-row dataset metadata + derived per-query weights.
///
/// Mirrors the C++ `Metadata` data members (`include/LightGBM/dataset.h`,
/// `src/io/metadata.cpp`). Built + validated via [`Metadata::new`]; the query
/// weights are derived in [`Metadata::finish_load`] (C++
/// `Metadata::CalculateQueryWeights`).
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    /// C++ `std::vector<label_t> label_` — per-row label (f32, `LabelT`).
    pub label: Vec<LabelT>,
    /// C++ `std::vector<label_t> weights_` — per-row weight (f32, `LabelT`).
    /// Empty when the caller supplied no weights.
    pub weights: Vec<LabelT>,
    /// C++ `std::vector<double> init_score_` — per-row (or per-row*class) initial
    /// score (`double`). Empty when the caller supplied no init scores.
    pub init_score: Vec<f64>,
    /// C++ `std::vector<data_size_t> query_boundaries_` (`i32`, `DataSizeT`) —
    /// the prefix-sum group boundaries: `query_boundaries[0] == 0`,
    /// `query_boundaries[num_queries] == num_rows`. Empty when there are no
    /// groups (the non-ranking case).
    pub query_boundaries: Vec<DataSizeT>,
    /// C++ `std::vector<label_t> query_weights_` (f32, `LabelT`) — DERIVED in
    /// [`Metadata::finish_load`] from `weights` + `query_boundaries` (mean weight
    /// per group). Empty until `finish_load` runs (and stays empty when either
    /// `weights` or `query_boundaries` is empty, per C++
    /// `CalculateQueryWeights`).
    pub query_weights: Vec<LabelT>,
    /// C++ `data_size_t num_data_` — number of rows.
    pub num_data: DataSizeT,
}

impl Metadata {
    /// Validate caller-provided metadata vectors and build a [`Metadata`] in the
    /// pre-`finish_load` state (`query_weights` empty). Mirrors the C++
    /// `Metadata::Init` / `SetLabel` / `SetWeights` / `SetQueries` length and
    /// query-sum checks, surfaced as typed [`DatasetError`] instead of
    /// `Log::Fatal` (Security V5 — never panic on caller input).
    ///
    /// `query_boundaries` is the already-prefix-summed boundary vector (the
    /// in-memory C++ `query_boundaries_` shape): it must be empty OR start at `0`,
    /// be monotone non-decreasing, and end at `num_rows`.
    pub fn new(
        label: Vec<LabelT>,
        weights: Vec<LabelT>,
        init_score: Vec<f64>,
        query_boundaries: Vec<DataSizeT>,
    ) -> Result<Self, DatasetError> {
        let num_data = label.len() as DataSizeT;

        // weights: empty or exactly num_rows (C++ SetWeights length CHECK).
        if !weights.is_empty() && weights.len() != label.len() {
            return Err(DatasetError::ShapeMismatch {
                detail: format!(
                    "weights.len()={} must be 0 or num_rows={}",
                    weights.len(),
                    label.len()
                ),
            });
        }

        // init_score: empty or a multiple of num_rows (C++ stores num_class*num_rows;
        // for the single-class MVP this is exactly num_rows). Require a multiple of
        // num_rows so multiclass init scores are also accepted.
        if !init_score.is_empty() {
            if label.is_empty() {
                return Err(DatasetError::ShapeMismatch {
                    detail: format!(
                        "init_score.len()={} but num_rows=0",
                        init_score.len()
                    ),
                });
            }
            if init_score.len() % label.len() != 0 {
                return Err(DatasetError::ShapeMismatch {
                    detail: format!(
                        "init_score.len()={} must be a multiple of num_rows={}",
                        init_score.len(),
                        label.len()
                    ),
                });
            }
        }

        // query_boundaries: empty, or [0, .., num_rows] monotone non-decreasing
        // (C++ SetQueriesFromIterator query-sum CHECK, surfaced as typed error).
        if !query_boundaries.is_empty() {
            if query_boundaries[0] != 0 {
                return Err(DatasetError::QueryBoundary {
                    detail: format!(
                        "query_boundaries[0]={} must be 0",
                        query_boundaries[0]
                    ),
                });
            }
            for w in query_boundaries.windows(2) {
                if w[1] < w[0] {
                    return Err(DatasetError::QueryBoundary {
                        detail: format!(
                            "query_boundaries not monotone non-decreasing: {} then {}",
                            w[0], w[1]
                        ),
                    });
                }
            }
            let last = *query_boundaries.last().unwrap();
            if last != num_data {
                return Err(DatasetError::QueryBoundary {
                    detail: format!(
                        "query_boundaries last={last} must equal num_rows={num_data}"
                    ),
                });
            }
        }

        Ok(Metadata {
            label,
            weights,
            init_score,
            query_boundaries,
            query_weights: Vec::new(),
            num_data,
        })
    }

    /// C++ `Metadata::FinishLoad` → `CalculateQueryBoundaries` →
    /// `CalculateQueryWeights` (`metadata.cpp:742-756`, `779-781`).
    ///
    /// Derives `query_weights` (mean weight per group) when BOTH `weights` and
    /// `query_boundaries` are present; otherwise leaves `query_weights` empty
    /// (the exact C++ early-return). The per-query mean is computed in `LabelT`
    /// (f32) arithmetic — the inner `+=` accumulates in f32 and the divide by the
    /// (int) group size converts the divisor to f32 — bit-identical to C++.
    pub fn finish_load(&mut self) {
        // C++ CalculateQueryWeights early-return (metadata.cpp:743-745).
        if self.weights.is_empty() || self.query_boundaries.is_empty() {
            return;
        }
        let num_queries = (self.query_boundaries.len() - 1) as DataSizeT;
        let mut query_weights = vec![0.0f32; num_queries as usize];
        for i in 0..num_queries as usize {
            let start = self.query_boundaries[i];
            let end = self.query_boundaries[i + 1];
            // accumulate in f32 (label_t), matching C++ `query_weights_[i] += weights_[j]`.
            let mut acc: LabelT = 0.0f32;
            for j in start..end {
                acc += self.weights[j as usize];
            }
            // divide by the int group size, which converts to f32 (C++ `/= (int)`).
            acc /= (end - start) as LabelT;
            query_weights[i] = acc;
        }
        self.query_weights = query_weights;
    }

    /// Number of rows.
    pub fn num_data(&self) -> DataSizeT {
        self.num_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_weight_length() {
        let err = Metadata::new(
            vec![1.0, 2.0, 3.0],
            vec![1.0, 1.0], // wrong length (2 != 3)
            Vec::new(),
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, DatasetError::ShapeMismatch { .. }));
    }

    #[test]
    fn new_accepts_empty_weights() {
        let m = Metadata::new(vec![1.0, 2.0, 3.0], Vec::new(), Vec::new(), Vec::new()).unwrap();
        assert_eq!(m.num_data(), 3);
        assert!(m.weights.is_empty());
    }

    #[test]
    fn new_rejects_non_monotone_query_boundaries() {
        let err = Metadata::new(
            vec![1.0, 2.0, 3.0, 4.0],
            Vec::new(),
            Vec::new(),
            vec![0, 3, 2, 4], // 3 then 2 — not monotone
        )
        .unwrap_err();
        assert!(matches!(err, DatasetError::QueryBoundary { .. }));
    }

    #[test]
    fn new_rejects_query_boundaries_not_ending_at_num_rows() {
        let err = Metadata::new(
            vec![1.0, 2.0, 3.0, 4.0],
            Vec::new(),
            Vec::new(),
            vec![0, 2, 3], // last=3 != num_rows=4
        )
        .unwrap_err();
        assert!(matches!(err, DatasetError::QueryBoundary { .. }));
    }

    #[test]
    fn new_rejects_query_boundaries_not_starting_at_zero() {
        let err = Metadata::new(
            vec![1.0, 2.0, 3.0],
            Vec::new(),
            Vec::new(),
            vec![1, 2, 3],
        )
        .unwrap_err();
        assert!(matches!(err, DatasetError::QueryBoundary { .. }));
    }

    #[test]
    fn finish_load_derives_mean_query_weight() {
        // 2 groups: [0,2)=rows 0,1; [2,5)=rows 2,3,4.
        // weights = [1,3, 2,4,6]; means = (1+3)/2=2, (2+4+6)/3=4.
        let mut m = Metadata::new(
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 3.0, 2.0, 4.0, 6.0],
            Vec::new(),
            vec![0, 2, 5],
        )
        .unwrap();
        m.finish_load();
        assert_eq!(m.query_weights, vec![2.0f32, 4.0f32]);
    }

    #[test]
    fn finish_load_no_query_weights_without_groups_or_weights() {
        // No groups -> empty query_weights.
        let mut m1 = Metadata::new(
            vec![0.0, 0.0],
            vec![1.0, 2.0],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        m1.finish_load();
        assert!(m1.query_weights.is_empty());

        // No weights -> empty query_weights.
        let mut m2 = Metadata::new(
            vec![0.0, 0.0],
            Vec::new(),
            Vec::new(),
            vec![0, 1, 2],
        )
        .unwrap();
        m2.finish_load();
        assert!(m2.query_weights.is_empty());
    }

    #[test]
    fn label_and_weights_are_f32() {
        // Type-level proof: the fields are Vec<f32>, not Vec<f64>.
        let m = Metadata::new(vec![1.5f32], vec![2.5f32], vec![3.5f64], Vec::new()).unwrap();
        let _l: &Vec<f32> = &m.label;
        let _w: &Vec<f32> = &m.weights;
        let _i: &Vec<f64> = &m.init_score;
    }
}
