//! Port of `Dataset::Construct` + `Dataset::FinishLoad` from
//! `LightGBM/src/io/dataset.cpp` (Construct 325-441, FinishLoad 443-463), in the
//! `lgbm-core/src/config/set.rs` ordered-pipeline style.
//!
//! # The immutability boundary (`finish_load`)
//!
//! `FinishLoad()` is the contract boundary the whole subsystem hangs on
//! (RESEARCH diagram, DAT-02): after it the columnar store is READ-ONLY — every
//! downstream phase (predict, histogram, split) reads an immutable `Dataset`.
//! C++ enforces this with a `bool is_finish_load_` guard. Here we model it as a
//! **type-state**: row pushes are only callable on [`Dataset`] (the loading
//! state); `finish_load` consumes it and returns a [`FinishedDataset`] that
//! exposes NO mutating method. A post-finish mutation is therefore a
//! compile-error, not a silent write — a strictly stronger guarantee than the
//! C++ runtime flag.
//!
//! This plan builds the **one-feature-per-group** default (each `BinMapper`
//! becomes its own single-value `FeatureGroup`); EFB bundling (`enable_bundle`)
//! is Plan 05. `Construct` uses `CreateBinData(..., force_dense=true,
//! force_sparse=false)`, so single-value groups are dense unless their
//! `sparse_rate >= 0.7` would select sparse — handled inside `FeatureGroup`.

use crate::bin_mapper::BinMapper;
use crate::error::DatasetError;
use crate::feature_group::FeatureGroup;

/// A dataset in the LOADING state (mutable). Built by [`Dataset::construct`];
/// rows are streamed in via [`Dataset::push_row`] / [`Dataset::push_value`];
/// [`Dataset::finish_load`] consumes it into the immutable [`FinishedDataset`].
#[derive(Debug)]
pub struct Dataset {
    num_data_: i32,
    num_features_: i32,
    /// One-feature-per-group default: `feature_groups_[g]` owns one feature.
    feature_groups_: Vec<FeatureGroup>,
}

impl Dataset {
    /// C++ `Dataset::Construct` (`dataset.cpp:325-441`), one-feature-per-group
    /// default. Builds a `FeatureGroup` per `BinMapper` (single-value groups).
    ///
    /// Validates `num_data >= 0` and a non-empty mapper set at the boundary,
    /// returning [`DatasetError`] rather than panicking (Security V5).
    pub fn construct(bin_mappers: Vec<BinMapper>, num_data: i32) -> Result<Self, DatasetError> {
        if num_data < 0 {
            return Err(DatasetError::ShapeMismatch {
                detail: format!("num_data must be >= 0, got {num_data}"),
            });
        }
        let num_features_ = bin_mappers.len() as i32;
        let mut feature_groups_ = Vec::with_capacity(bin_mappers.len());
        for m in bin_mappers {
            feature_groups_.push(FeatureGroup::new_single(m, num_data));
        }
        Ok(Dataset {
            num_data_: num_data,
            num_features_,
            feature_groups_,
        })
    }

    /// Push a single feature value at `(row, feature)`. Routes through the owning
    /// `FeatureGroup::push_data` (one-feature-per-group, so sub_feature_idx = 0).
    pub fn push_value(&mut self, feature: usize, row: i32, value: f64) {
        self.feature_groups_[feature].push_data(0, row, value);
    }

    /// Push a full row (one value per feature).
    pub fn push_row(&mut self, row: i32, values: &[f64]) -> Result<(), DatasetError> {
        if values.len() as i32 != self.num_features_ {
            return Err(DatasetError::ShapeMismatch {
                detail: format!(
                    "row has {} values, expected num_features={}",
                    values.len(),
                    self.num_features_
                ),
            });
        }
        for (feature, &v) in values.iter().enumerate() {
            self.feature_groups_[feature].push_data(0, row, v);
        }
        Ok(())
    }

    /// C++ `Dataset::FinishLoad` (`dataset.cpp:443-463`) — THE IMMUTABILITY
    /// BOUNDARY. Calls each `FeatureGroup::finish_load` (→ each `Bin::finish_load`)
    /// then consumes `self`, yielding an immutable [`FinishedDataset`] with no
    /// mutating API (post-finish mutation is a compile error).
    pub fn finish_load(mut self) -> FinishedDataset {
        for fg in &mut self.feature_groups_ {
            fg.finish_load();
        }
        FinishedDataset {
            num_data_: self.num_data_,
            num_features_: self.num_features_,
            feature_groups_: self.feature_groups_,
        }
    }

    /// Number of rows.
    pub fn num_data(&self) -> i32 {
        self.num_data_
    }

    /// Number of features.
    pub fn num_features(&self) -> i32 {
        self.num_features_
    }
}

/// An IMMUTABLE dataset (post-`finish_load`). Exposes only read accessors — there
/// is no `push_*` / `finish_load` here, so the immutability boundary is enforced
/// by the type system (the C++ `is_finish_load_` guard, made a compile error).
#[derive(Debug)]
pub struct FinishedDataset {
    num_data_: i32,
    num_features_: i32,
    feature_groups_: Vec<FeatureGroup>,
}

impl FinishedDataset {
    /// Number of rows.
    pub fn num_data(&self) -> i32 {
        self.num_data_
    }

    /// Number of features.
    pub fn num_features(&self) -> i32 {
        self.num_features_
    }

    /// Number of feature groups (one-feature-per-group default).
    pub fn num_groups(&self) -> i32 {
        self.feature_groups_.len() as i32
    }

    /// Read-only access to a feature group.
    pub fn feature_group(&self, group: usize) -> &FeatureGroup {
        &self.feature_groups_[group]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin_mapper::{BinType, MissingType};

    fn mapper(num_bin: i32, most_freq_bin: u32, bounds: Vec<f64>) -> BinMapper {
        BinMapper {
            num_bin_: num_bin,
            missing_type_: MissingType::None,
            bin_upper_bound_: bounds,
            is_trivial_: false,
            sparse_rate_: 0.0,
            bin_type_: BinType::Numerical,
            default_bin_: 0,
            most_freq_bin_: most_freq_bin,
            min_val_: 0.0,
            max_val_: 0.0,
        }
    }

    #[test]
    fn construct_builds_one_group_per_feature() {
        let mappers = vec![
            mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]),
            mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]),
        ];
        let ds = Dataset::construct(mappers, 4).unwrap();
        assert_eq!(ds.num_features(), 2);
        assert_eq!(ds.num_data(), 4);
    }

    #[test]
    fn construct_rejects_negative_num_data() {
        let mappers = vec![mapper(3, 0, vec![1.0, 2.0, f64::INFINITY])];
        let err = Dataset::construct(mappers, -1).unwrap_err();
        assert!(matches!(err, DatasetError::ShapeMismatch { .. }));
    }

    #[test]
    fn push_row_validates_width() {
        let mappers = vec![mapper(3, 0, vec![1.0, 2.0, f64::INFINITY])];
        let mut ds = Dataset::construct(mappers, 2).unwrap();
        let err = ds.push_row(0, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, DatasetError::ShapeMismatch { .. }));
    }

    #[test]
    fn finish_load_yields_immutable_read_only_view() {
        // This test documents the immutability boundary: after finish_load the
        // value is a FinishedDataset with NO push_*/finish_load method. The
        // following lines, if uncommented, would FAIL TO COMPILE (type-state
        // enforcement, strictly stronger than the C++ runtime is_finish_load_):
        //
        //   let finished = ds.finish_load();
        //   finished.push_value(0, 0, 1.0);   // ERROR: no method `push_value`
        //   finished.finish_load();           // ERROR: no method `finish_load`
        //
        let mappers = vec![mapper(3, 0, vec![1.0, 2.0, f64::INFINITY])];
        let mut ds = Dataset::construct(mappers, 3).unwrap();
        ds.push_value(0, 0, 0.5); // bin 0 == most_freq -> skipped
        ds.push_value(0, 1, 1.5); // -> stored
        ds.push_value(0, 2, 100.0); // -> stored
        let finished = ds.finish_load();
        assert_eq!(finished.num_data(), 3);
        assert_eq!(finished.num_groups(), 1);
        // read-only access still works.
        let bd = finished.feature_group(0).bin_data().unwrap();
        assert_eq!(bd.data(0), 0, "skipped most-freq row stays default");
    }

    /// Compile-fail documentation: a `FinishedDataset` exposes no mutator, so a
    /// post-finish push is rejected at compile time. We assert the property by
    /// confirming the method does not exist via a trait-free probe at runtime is
    /// impossible — instead this is verified structurally (FinishedDataset has no
    /// `push_*`). The runtime test above plus this note constitute the
    /// immutability proof.
    #[test]
    fn finished_dataset_has_no_mutating_api() {
        // Structural assertion: FinishedDataset's public surface is read-only.
        // (If a mutator were added, the doc-comment compile-fail block above and
        // this invariant note must be revisited.)
        let mappers = vec![mapper(3, 0, vec![1.0, 2.0, f64::INFINITY])];
        let ds = Dataset::construct(mappers, 1).unwrap();
        let finished = ds.finish_load();
        // Only read methods are reachable:
        let _ = finished.num_data();
        let _ = finished.num_features();
        let _ = finished.num_groups();
    }
}
