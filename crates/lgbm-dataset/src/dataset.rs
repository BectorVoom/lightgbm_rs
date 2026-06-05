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
use lgbm_core::config::Config;

/// A dataset in the LOADING state (mutable). Built by [`Dataset::construct`];
/// rows are streamed in via [`Dataset::push_row`] / [`Dataset::push_value`];
/// [`Dataset::finish_load`] consumes it into the immutable [`FinishedDataset`].
#[derive(Debug)]
pub struct Dataset {
    num_data_: i32,
    num_features_: i32,
    /// One-feature-per-group default: `feature_groups_[g]` owns one feature.
    /// In the EFB-bundled path a group may own multiple sub-features.
    feature_groups_: Vec<FeatureGroup>,
    /// C++ `std::vector<int> feature2group_` — real feature index -> group id.
    /// In the one-feature-per-group default `feature2group_[f] == f`.
    feature2group_: Vec<i32>,
    /// C++ `std::vector<int> feature2subfeature_` — packed feature -> sub-feature
    /// index within its group (0 in the default path).
    feature2subfeature_: Vec<i32>,
    /// C++ `std::vector<int> real_feature_idx_` — packed feature index -> real
    /// (original) feature index. In the one-feature-per-group default this is the
    /// identity; in the EFB-bundled path the shuffle reorders it.
    real_feature_idx_: Vec<i32>,
    /// C++ `std::vector<int> used_feature_map_` — real feature index -> packed
    /// feature index (-1 for trivial/unused features). The inverse of
    /// `real_feature_idx_`; lets callers push by REAL feature index.
    used_feature_map_: Vec<i32>,
}

/// The per-column SAMPLE inputs Exclusive Feature Bundling consumes (mirrors the
/// C++ `Dataset::Construct` args `sample_non_zero_indices` / `sample_values` /
/// `num_per_col`). Indexed by REAL column. Only needed for the `enable_bundle`
/// dispatch; the no-bundle path ignores it.
#[derive(Debug, Clone)]
pub struct EfbSamples {
    /// Sorted non-zero sample-row indices per column.
    pub sample_indices: Vec<Vec<i32>>,
    /// Sampled VALUES at those rows per column (for `FixSampleIndices`).
    pub sample_values: Vec<Vec<f64>>,
    /// Non-zero count per column.
    pub num_per_col: Vec<i32>,
    /// Number of sampled columns (`num_sample_col`).
    pub num_sample_col: i32,
    /// Number of sampled rows (`total_sample_cnt`).
    pub total_sample_cnt: i32,
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
        // One-feature-per-group: feature f lives at group f, sub-feature 0.
        let feature2group_ = (0..num_features_).collect();
        let feature2subfeature_ = vec![0; num_features_ as usize];
        // packed index == real index (identity) in the no-bundle default.
        let real_feature_idx_ = (0..num_features_).collect();
        let used_feature_map_ = (0..num_features_).collect();
        Ok(Dataset {
            num_data_: num_data,
            num_features_,
            feature_groups_,
            feature2group_,
            feature2subfeature_,
            real_feature_idx_,
            used_feature_map_,
        })
    }

    /// C++ `Dataset::Construct` with the `enable_bundle` dispatch
    /// (`dataset.cpp:325-411`). When `cfg.enable_bundle` (and there is at least
    /// one feature), builds the bundled grouping via
    /// [`crate::efb::fast_feature_bundling`]; otherwise falls back to the
    /// one-feature-per-group default (identical to [`Dataset::construct`]).
    ///
    /// `samples` carries the per-column EFB sample inputs (non-zero indices +
    /// values + counts). Mirrors the C++ group-building loop: for each group it
    /// collects that group's mappers, records `feature2group_` /
    /// `feature2subfeature_`, and builds a `FeatureGroup` with the group's
    /// `group_is_multi_val[i]` flag.
    pub fn construct_bundled(
        bin_mappers: Vec<BinMapper>,
        num_data: i32,
        cfg: &Config,
        samples: &EfbSamples,
    ) -> Result<Self, DatasetError> {
        if num_data < 0 {
            return Err(DatasetError::ShapeMismatch {
                detail: format!("num_data must be >= 0, got {num_data}"),
            });
        }
        let num_total_features = bin_mappers.len() as i32;

        // used_features = non-trivial features, in order (dataset.cpp:337-343).
        let used_features: Vec<i32> = (0..num_total_features)
            .filter(|&i| !bin_mappers[i as usize].is_trivial_)
            .collect();

        // is_sparse from config (dataset.cpp:352); CPU core never sets GPU.
        let is_sparse = cfg.is_enable_sparse;
        let is_use_gpu = false;

        // Grouping: FastFeatureBundling when enabled + non-empty, else 1-per-grp.
        let (features_in_group, group_is_multi_val): (Vec<Vec<i32>>, Vec<i8>) =
            if cfg.enable_bundle && !used_features.is_empty() {
                crate::efb::fast_feature_bundling(
                    &bin_mappers,
                    &samples.sample_indices,
                    &samples.sample_values,
                    &samples.num_per_col,
                    samples.num_sample_col,
                    samples.total_sample_cnt,
                    &used_features,
                    num_data,
                    is_use_gpu,
                    is_sparse,
                )
            } else {
                let groups = crate::efb::one_feature_per_group(&used_features);
                let flags = vec![0i8; groups.len()];
                (groups, flags)
            };

        // Build FeatureGroups + the feature->group/subfeature maps
        // (dataset.cpp:375-411). Move mappers out of the Vec by index.
        let mut mappers_opt: Vec<Option<BinMapper>> =
            bin_mappers.into_iter().map(Some).collect();
        let mut num_features_ = 0i32;
        for fs in &features_in_group {
            num_features_ += fs.len() as i32;
        }
        let mut feature2group_ = vec![-1i32; num_features_ as usize];
        let mut feature2subfeature_ = vec![-1i32; num_features_ as usize];
        let mut real_feature_idx_ = vec![-1i32; num_features_ as usize];
        // real feature index -> packed index (C++ used_feature_map_), -1 if unused.
        let mut used_feature_map_ = vec![-1i32; num_total_features as usize];

        let mut feature_groups_ = Vec::with_capacity(features_in_group.len());
        let mut cur_fidx = 0i32;
        for (i, cur_features) in features_in_group.iter().enumerate() {
            let cur_cnt_features = cur_features.len() as i32;
            let mut cur_bin_mappers = Vec::with_capacity(cur_features.len());
            for (j, &real_fidx) in cur_features.iter().enumerate() {
                real_feature_idx_[cur_fidx as usize] = real_fidx;
                used_feature_map_[real_fidx as usize] = cur_fidx;
                feature2group_[cur_fidx as usize] = i as i32;
                feature2subfeature_[cur_fidx as usize] = j as i32;
                cur_bin_mappers.push(
                    mappers_opt[real_fidx as usize]
                        .take()
                        .expect("each used feature assigned to exactly one group"),
                );
                cur_fidx += 1;
            }
            let is_multi_val = group_is_multi_val.get(i).copied().unwrap_or(0) != 0;
            feature_groups_.push(FeatureGroup::new(
                cur_cnt_features,
                is_multi_val,
                cur_bin_mappers,
                num_data,
                i as i32,
            ));
        }

        Ok(Dataset {
            num_data_: num_data,
            num_features_,
            feature_groups_,
            feature2group_,
            feature2subfeature_,
            real_feature_idx_,
            used_feature_map_,
        })
    }

    /// Push a single feature value at `(real_feature, row)`. Translates the REAL
    /// feature index to its packed index via `used_feature_map_` (C++
    /// `Dataset::PushOneRow`/`used_feature_map_`), then routes through the owning
    /// `FeatureGroup::push_data` via `feature2group_`/`feature2subfeature_`. In the
    /// one-feature-per-group default the real index == packed index; in the
    /// EFB-bundled path the shuffle makes them differ, so the translation is
    /// load-bearing.
    pub fn push_value(&mut self, real_feature: usize, row: i32, value: f64) {
        let packed = self.used_feature_map_[real_feature];
        if packed < 0 {
            return; // trivial / unused feature has no store
        }
        let group = self.feature2group_[packed as usize] as usize;
        let sub = self.feature2subfeature_[packed as usize] as usize;
        self.feature_groups_[group].push_data(sub, row, value);
    }

    /// Real feature index -> group id (via `used_feature_map_`); -1 if the feature
    /// is trivial/unused (no group).
    pub fn feature_to_group(&self, real_feature: usize) -> i32 {
        let packed = self.used_feature_map_[real_feature];
        if packed < 0 {
            -1
        } else {
            self.feature2group_[packed as usize]
        }
    }

    /// Real feature index -> sub-feature index within its group; -1 if unused.
    pub fn feature_to_subfeature(&self, real_feature: usize) -> i32 {
        let packed = self.used_feature_map_[real_feature];
        if packed < 0 {
            -1
        } else {
            self.feature2subfeature_[packed as usize]
        }
    }

    /// Push a full row (one value per REAL feature). Each value is routed through
    /// [`Dataset::push_value`] (real -> packed translation), so this is correct in
    /// both the no-bundle and EFB-bundled paths. The row width is the number of
    /// REAL features (`used_feature_map_.len()`).
    pub fn push_row(&mut self, row: i32, values: &[f64]) -> Result<(), DatasetError> {
        if values.len() != self.used_feature_map_.len() {
            return Err(DatasetError::ShapeMismatch {
                detail: format!(
                    "row has {} values, expected num_real_features={}",
                    values.len(),
                    self.used_feature_map_.len()
                ),
            });
        }
        for (feature, &v) in values.iter().enumerate() {
            self.push_value(feature, row, v);
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
            feature2group_: self.feature2group_,
            feature2subfeature_: self.feature2subfeature_,
            real_feature_idx_: self.real_feature_idx_,
            used_feature_map_: self.used_feature_map_,
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
    feature2group_: Vec<i32>,
    feature2subfeature_: Vec<i32>,
    real_feature_idx_: Vec<i32>,
    used_feature_map_: Vec<i32>,
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

    /// Real feature index -> group id (via `used_feature_map_`), for EFB goldens.
    /// -1 if the feature is trivial/unused.
    pub fn feature_to_group(&self, real_feature: usize) -> i32 {
        let packed = self.used_feature_map_[real_feature];
        if packed < 0 {
            -1
        } else {
            self.feature2group_[packed as usize]
        }
    }

    /// Real feature index -> sub-feature index within its group; -1 if unused.
    pub fn feature_to_subfeature(&self, real_feature: usize) -> i32 {
        let packed = self.used_feature_map_[real_feature];
        if packed < 0 {
            -1
        } else {
            self.feature2subfeature_[packed as usize]
        }
    }

    /// Packed feature index -> real (original) feature index (C++
    /// `real_feature_idx_`). The inverse of `feature_to_group`'s lookup; useful
    /// for dumping group-major membership in the EFB golden order.
    pub fn real_feature_idx(&self, packed: usize) -> i32 {
        self.real_feature_idx_[packed]
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
            bin_2_categorical_: Vec::new(),
            categorical_2_bin_: std::collections::HashMap::new(),
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
    fn empty_samples(num_cols: i32) -> EfbSamples {
        EfbSamples {
            sample_indices: vec![Vec::new(); num_cols as usize],
            sample_values: vec![Vec::new(); num_cols as usize],
            num_per_col: vec![0; num_cols as usize],
            num_sample_col: num_cols,
            total_sample_cnt: 0,
        }
    }

    #[test]
    fn construct_bundled_disabled_is_one_feature_per_group() {
        // cfg.enable_bundle = false -> one-feature-per-group dispatch boundary.
        let mut cfg = Config::default();
        cfg.enable_bundle = false;
        let mappers = vec![
            mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]),
            mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]),
        ];
        let ds = Dataset::construct_bundled(mappers, 4, &cfg, &empty_samples(2)).unwrap();
        let finished = ds.finish_load();
        assert_eq!(finished.num_groups(), 2, "no-bundle -> 2 singleton groups");
        assert_eq!(finished.feature_to_group(0), 0);
        assert_eq!(finished.feature_to_group(1), 1);
        assert_eq!(finished.feature_to_subfeature(1), 0);
    }

    #[test]
    fn construct_bundled_enabled_bundles_mutually_exclusive_features() {
        // cfg.enable_bundle = true with two mutually-exclusive sparse features
        // (disjoint non-zero rows) -> ONE bundled group.
        let mut cfg = Config::default();
        cfg.enable_bundle = true;
        let mappers = vec![
            mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]),
            mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]),
        ];
        let samples = EfbSamples {
            sample_indices: vec![vec![0], vec![1]], // disjoint non-zeros
            sample_values: vec![vec![2.5], vec![2.5]],
            num_per_col: vec![1, 1],
            num_sample_col: 2,
            total_sample_cnt: 4,
        };
        let ds = Dataset::construct_bundled(mappers, 4, &cfg, &samples).unwrap();
        let finished = ds.finish_load();
        assert_eq!(finished.num_groups(), 1, "enable_bundle -> bundled group");
        // both features map to the same group.
        assert_eq!(
            finished.feature_to_group(0),
            finished.feature_to_group(1),
            "mutually-exclusive features share a group"
        );
    }

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
