//! Bit-for-bit port of `FeatureGroup` bin-offset packing + `PushData` from
//! `LightGBM/include/LightGBM/feature_group.h` (primary ctor 39-76, `AllocateBins`
//! 212-229, `PushData` 253-267, `CreateBinData` 586-612).
//!
//! Follows the `lgbm-core/src/config/set.rs` ordered-pipeline doc style: the
//! construction stages are listed in the EXACT C++ order with line cites.
//!
//! # Construction stages (primary `FeatureGroup(num_feature, is_multi_val, ...)`)
//!
//! 1. accumulate `sum_sparse_rate` over sub-feature mappers, divide by count
//!    (`feature_group.h:45-50`);
//! 2. `offset = 1`; if `sum_sparse_rate < multi_val_bin_sparse_threshold (0.25)`
//!    AND `is_multi_val` -> `offset = 0; is_dense_multi_val = true` (51-57);
//! 3. `num_total_bin_ = offset` (accumulated in `u64`, mirroring C++ `uint64_t`
//!    dataset.cpp:383 — overflow guard, RESEARCH §Security T-02-04) (59);
//! 4. the `group_id == 0 && is_dense_multi_val && most_freq_bin(0) > 0`
//!    force-one-bin special case -> `num_total_bin_ = 1` (62-65);
//! 5. `bin_offsets_ = [num_total_bin_]` (66);
//! 6. per feature: `num_bin = mapper.num_bin(); if most_freq_bin == 0 { num_bin
//!    -= offset } num_total_bin_ += num_bin; bin_offsets_.push(num_total_bin_)`
//!    (67-74);
//! 7. `CreateBinData(num_data, is_multi_val, force_dense=true, force_sparse=false)`
//!    (75) — so single-value groups are DENSE in the common MVP path
//!    (RESEARCH §8).
//!
//! Thresholds: `kSparseThreshold = 0.7` (bin.h:43),
//! `multi_val_bin_sparse_threshold = 0.25` (bin.h:599).

use crate::bin::{Bin, create_dense_bin, create_sparse_bin};
use crate::bin_mapper::BinMapper;

/// C++ `kSparseThreshold` (`bin.h:43`).
const K_SPARSE_THRESHOLD: f64 = 0.7;
/// C++ `MultiValBin::multi_val_bin_sparse_threshold` (`bin.h:599`).
const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25;

/// C++ `class FeatureGroup` (`feature_group.h`). One group of one-or-more
/// sub-features sharing a single columnar bin store (single-value group = one
/// feature, the one-feature-per-group default this plan builds; EFB multi-val
/// bundling is Plan 05).
#[derive(Debug)]
pub struct FeatureGroup {
    /// C++ `int num_feature_`.
    num_feature_: i32,
    /// C++ `bool is_multi_val_`.
    is_multi_val_: bool,
    /// C++ `bool is_dense_multi_val_`. Set during offset packing; consumed by the
    /// EFB multi-val store layout in Plan 05 (kept now to mirror C++ state and
    /// keep the offset-packing branch faithful).
    #[allow(dead_code)]
    is_dense_multi_val_: bool,
    /// C++ `bool is_sparse_`.
    is_sparse_: bool,
    /// C++ `std::vector<std::unique_ptr<BinMapper>> bin_mappers_`.
    bin_mappers_: Vec<BinMapper>,
    /// C++ `std::vector<uint32_t> bin_offsets_`.
    pub bin_offsets_: Vec<u32>,
    /// C++ `uint64_t num_total_bin_` — accumulated in `u64` (overflow guard).
    pub num_total_bin_: u64,
    /// C++ `std::unique_ptr<Bin> bin_data_` (single-value group store).
    bin_data_: Option<Box<dyn Bin>>,
    /// C++ `std::vector<std::unique_ptr<Bin>> multi_bin_data_` (multi-val store).
    multi_bin_data_: Vec<Box<dyn Bin>>,
}

impl FeatureGroup {
    /// C++ primary `FeatureGroup(num_feature, is_multi_val, bin_mappers,
    /// num_data, group_id)` (`feature_group.h:39-76`).
    pub fn new(
        num_feature: i32,
        is_multi_val: bool,
        bin_mappers: Vec<BinMapper>,
        num_data: i32,
        group_id: i32,
    ) -> Self {
        assert_eq!(bin_mappers.len() as i32, num_feature);

        // Stage 1: mean sparse rate.
        let mut sum_sparse_rate = 0.0f64;
        for m in &bin_mappers {
            sum_sparse_rate += m.sparse_rate_;
        }
        sum_sparse_rate /= num_feature as f64;

        // Stage 2: offset + dense-multi-val decision.
        let mut offset: i32 = 1;
        let mut is_dense_multi_val_ = false;
        if sum_sparse_rate < MULTI_VAL_BIN_SPARSE_THRESHOLD && is_multi_val {
            offset = 0;
            is_dense_multi_val_ = true;
        }

        // Stage 3: num_total_bin_ accumulator (u64 — overflow guard, T-02-04).
        let mut num_total_bin_: u64 = offset as u64;

        // Stage 4: force-one-bin special case.
        if group_id == 0
            && num_feature > 0
            && is_dense_multi_val_
            && bin_mappers[0].most_freq_bin_ > 0
        {
            num_total_bin_ = 1;
        }

        // Stage 5: bin_offsets_ seeded.
        let mut bin_offsets_: Vec<u32> = vec![num_total_bin_ as u32];

        // Stage 6: per-feature offset packing.
        for m in &bin_mappers {
            let mut num_bin = m.num_bin_;
            if m.most_freq_bin_ == 0 {
                num_bin -= offset;
            }
            // widen to u64 for the accumulation (C++ uint64_t += int).
            num_total_bin_ += num_bin as u64;
            bin_offsets_.push(num_total_bin_ as u32);
        }

        let mut fg = FeatureGroup {
            num_feature_: num_feature,
            is_multi_val_: is_multi_val,
            is_dense_multi_val_,
            is_sparse_: false,
            bin_mappers_: bin_mappers,
            bin_offsets_,
            num_total_bin_,
            bin_data_: None,
            multi_bin_data_: Vec::new(),
        };

        // Stage 7: CreateBinData(num_data, is_multi_val, force_dense=true, force_sparse=false).
        fg.create_bin_data(num_data, is_multi_val, true, false);
        fg
    }

    /// C++ single-feature convenience ctor `FeatureGroup(bin_mappers, num_data)`
    /// (`feature_group.h:93-111`): always single-value, `num_total_bin_` seeded
    /// at 1 (bin 0 stores the default), `CreateBinData(num_data, false, false,
    /// false)` so it picks sparse iff `sparse_rate >= 0.7`.
    pub fn new_single(bin_mapper: BinMapper, num_data: i32) -> Self {
        let mut num_total_bin_: u64 = 1;
        let mut bin_offsets_: Vec<u32> = vec![num_total_bin_ as u32];
        let mut num_bin = bin_mapper.num_bin_;
        if bin_mapper.most_freq_bin_ == 0 {
            num_bin -= 1;
        }
        num_total_bin_ += num_bin as u64;
        bin_offsets_.push(num_total_bin_ as u32);

        let mut fg = FeatureGroup {
            num_feature_: 1,
            is_multi_val_: false,
            is_dense_multi_val_: false,
            is_sparse_: false,
            bin_mappers_: vec![bin_mapper],
            bin_offsets_,
            num_total_bin_,
            bin_data_: None,
            multi_bin_data_: Vec::new(),
        };
        fg.create_bin_data(num_data, false, false, false);
        fg
    }

    /// C++ `void CreateBinData(num_data, is_multi_val, force_dense, force_sparse)`
    /// (`feature_group.h:586-612`).
    fn create_bin_data(
        &mut self,
        num_data: i32,
        is_multi_val: bool,
        force_dense: bool,
        force_sparse: bool,
    ) {
        if is_multi_val {
            self.multi_bin_data_.clear();
            for m in &self.bin_mappers_ {
                let addi = if m.most_freq_bin_ == 0 { 0 } else { 1 };
                if m.sparse_rate_ >= K_SPARSE_THRESHOLD {
                    self.multi_bin_data_
                        .push(create_sparse_bin(num_data, m.num_bin_ + addi));
                } else {
                    self.multi_bin_data_
                        .push(create_dense_bin(num_data, m.num_bin_ + addi));
                }
            }
            self.is_multi_val_ = true;
        } else {
            if force_sparse
                || (!force_dense
                    && self.num_feature_ == 1
                    && self.bin_mappers_[0].sparse_rate_ >= K_SPARSE_THRESHOLD)
            {
                self.is_sparse_ = true;
                self.bin_data_ = Some(create_sparse_bin(num_data, self.num_total_bin_ as i32));
            } else {
                self.is_sparse_ = false;
                self.bin_data_ = Some(create_dense_bin(num_data, self.num_total_bin_ as i32));
            }
            self.is_multi_val_ = false;
        }
    }

    /// C++ `inline void PushData(int tid, int sub_feature_idx, data_size_t
    /// line_idx, double value)` (`feature_group.h:253-267`), transcribed verbatim
    /// (RESEARCH §10, Pitfall 3 — do NOT skip the most-freq skip, the `-1`, or the
    /// `+bin_offsets_`):
    ///
    /// ```text
    /// bin = mapper.value_to_bin(value);
    /// if bin == mapper.most_freq_bin(): return;        // skip most-freq (implicit)
    /// if mapper.most_freq_bin() == 0: bin -= 1;
    /// if is_multi_val: multi_bin_data[sub].push(row, bin + 1);
    /// else: bin += bin_offsets[sub]; bin_data.push(row, bin);
    /// ```
    pub fn push_data(&mut self, sub_feature_idx: usize, line_idx: i32, value: f64) {
        let mapper = &self.bin_mappers_[sub_feature_idx];
        let mut bin = mapper.value_to_bin(value);
        let most_freq = mapper.most_freq_bin_;
        if bin == most_freq {
            return;
        }
        if most_freq == 0 {
            bin -= 1;
        }
        if self.is_multi_val_ {
            self.multi_bin_data_[sub_feature_idx].push(line_idx, bin + 1);
        } else {
            bin += self.bin_offsets_[sub_feature_idx];
            self.bin_data_
                .as_mut()
                .expect("bin_data_ allocated by create_bin_data")
                .push(line_idx, bin);
        }
    }

    /// C++ `FeatureGroup::FinishLoad` — propagate the immutability boundary to
    /// each owned `Bin`.
    pub fn finish_load(&mut self) {
        if self.is_multi_val_ {
            for b in &mut self.multi_bin_data_ {
                b.finish_load();
            }
        } else if let Some(b) = self.bin_data_.as_mut() {
            b.finish_load();
        }
    }

    /// Number of sub-features in the group.
    pub fn num_feature(&self) -> i32 {
        self.num_feature_
    }

    /// Whether this group uses a sparse single-value store.
    pub fn is_sparse(&self) -> bool {
        self.is_sparse_
    }

    /// Whether this group is a multi-val (bundled) group.
    pub fn is_multi_val(&self) -> bool {
        self.is_multi_val_
    }

    /// Read access to the single-value bin store (for golden byte-layout tests).
    pub fn bin_data(&self) -> Option<&dyn Bin> {
        self.bin_data_.as_deref()
    }

    /// Read access to a multi-val sub-feature store (for golden tests).
    pub fn multi_bin_data(&self, sub: usize) -> Option<&dyn Bin> {
        self.multi_bin_data_.get(sub).map(|b| b.as_ref())
    }

    /// Sub-feature bin mapper accessor.
    pub fn bin_mapper(&self, sub: usize) -> &BinMapper {
        &self.bin_mappers_[sub]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin_mapper::{BinType, MissingType};

    /// Hand-build a numeric mapper with explicit bins for offset/push tests.
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
    fn offset_packing_two_features_most_freq_zero_on_first() {
        // Single-value group is not the path here; use the primary ctor with two
        // features in ONE group but is_multi_val=false is invalid (multi feature
        // => multi_val). Use is_multi_val=true so offset logic engages, but force
        // dense-multi-val via low sparse rate? Instead test the offset formula
        // directly through the primary ctor with is_multi_val=false single-val
        // path is for 1 feature only. So model a 2-feature dense-multi-val group.
        //
        // feature 0: num_bin=5, most_freq_bin=0  -> contributes 5 - offset
        // feature 1: num_bin=4, most_freq_bin=2  -> contributes 4
        // sum_sparse_rate small (0.0) and is_multi_val=true => offset=0,
        // is_dense_multi_val=true. group_id=1 (not 0) so no force-one-bin.
        //   num_total_bin_ = offset(0)
        //   bin_offsets_ = [0]
        //   f0: 5 - 0 = 5 -> num_total_bin_=5, push 5
        //   f1: 4        -> num_total_bin_=9, push 9
        let f0 = mapper(5, 0, vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY]);
        let f1 = mapper(4, 2, vec![1.0, 2.0, 3.0, f64::INFINITY]);
        let fg = FeatureGroup::new(2, true, vec![f0, f1], 8, 1);
        assert_eq!(fg.num_total_bin_, 9, "u64 num_total_bin_ formula");
        assert_eq!(fg.bin_offsets_, vec![0, 5, 9]);
    }

    #[test]
    fn offset_packing_single_value_dense_group() {
        // Single-value group via new_single: num_total_bin_ seeded at 1.
        // feature: num_bin=5, most_freq_bin=0 -> 5 - 1 = 4 -> num_total_bin_=5.
        let f0 = mapper(5, 0, vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY]);
        let fg = FeatureGroup::new_single(f0, 8);
        assert_eq!(fg.num_total_bin_, 5);
        assert_eq!(fg.bin_offsets_, vec![1, 5]);
    }

    #[test]
    fn force_one_bin_special_case_group0_dense_multi_val() {
        // group_id=0, dense-multi-val, first feature most_freq_bin>0 => force 1.
        let f0 = mapper(5, 2, vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY]);
        let f1 = mapper(4, 1, vec![1.0, 2.0, 3.0, f64::INFINITY]);
        let fg = FeatureGroup::new(2, true, vec![f0, f1], 8, 0);
        // num_total_bin_ forced to 1; offset=0 (low sparse rate, multi_val).
        //   bin_offsets_=[1]; f0: 5 -> 6; f1: 4 -> 10.
        assert_eq!(fg.bin_offsets_[0], 1);
        assert_eq!(fg.num_total_bin_, 10);
    }

    #[test]
    fn push_data_skips_most_freq_and_subtracts_one_when_zero() {
        // Single dense group, most_freq_bin=0. bin_offsets_=[1, ...].
        // bounds [1,2,inf]: value 0.5 -> bin 0 == most_freq => SKIP (no push).
        //                   value 1.5 -> bin 1; most_freq==0 => bin=0;
        //                                bin += bin_offsets_[0]=1 => push(row,1).
        let f0 = mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]);
        let mut fg = FeatureGroup::new_single(f0, 4);
        // most-freq value (bin 0) is skipped: nothing pushed at row 0.
        fg.push_data(0, 0, 0.5);
        // a non-most-freq value pushes the offset-adjusted bin.
        fg.push_data(0, 1, 1.5);
        fg.finish_load();
        let bd = fg.bin_data().unwrap();
        assert_eq!(bd.data(0), 0, "most-freq row was skipped -> stays default 0");
        assert_eq!(bd.data(1), 1, "bin 1 -1 +offset(1) = 1");
    }

    #[test]
    fn push_data_no_subtract_when_most_freq_nonzero() {
        // most_freq_bin=2 (nonzero): no -1 adjustment. bin_offsets_=[1, ...].
        // value mapping to bin 2 is skipped; bin 1 -> +offset(1) = 2.
        let f0 = mapper(3, 2, vec![1.0, 2.0, f64::INFINITY]);
        let mut fg = FeatureGroup::new_single(f0, 4);
        fg.push_data(0, 0, 100.0); // bin 2 == most_freq => skip
        fg.push_data(0, 1, 1.5); // bin 1 -> +1 = 2
        fg.finish_load();
        let bd = fg.bin_data().unwrap();
        assert_eq!(bd.data(0), 0, "most-freq skipped");
        assert_eq!(bd.data(1), 2, "bin 1 + offset(1), no -1");
    }

    #[test]
    fn multi_val_push_emits_bin_plus_one_reserving_bin_zero() {
        // EFB multi-val branch: a non-most-freq value pushes `bin + 1` into the
        // per-sub-feature store so bin 0 stays reserved for the implicit
        // most-freq (feature_group.h:253-267, RESEARCH §10). Build a 2-feature
        // dense-multi-val group (low sparse rate -> offset=0, is_multi_val=true).
        // feature 0: bounds [1,2,inf], most_freq_bin=0.
        //   value 1.5 -> value_to_bin = bin 1; most_freq(0) != bin => not skipped;
        //   most_freq==0 => bin -= 1 => 0; multi-val => push(row, bin + 1) = 1.
        let f0 = mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]);
        let f1 = mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]);
        let mut fg = FeatureGroup::new(2, true, vec![f0, f1], 2, 1);
        assert!(fg.is_multi_val(), "2-feature group with is_multi_val=true");
        fg.push_data(0, 0, 1.5); // feature 0, row 0 -> bin 1 -> -1 -> 0 -> +1 = 1
        fg.push_data(0, 1, 100.0); // feature 0, row 1 -> bin 2 -> -1 -> 1 -> +1 = 2
        fg.finish_load();
        let store = fg.multi_bin_data(0).unwrap();
        assert_eq!(store.data(0), 1, "bin 1 -> -1 -> 0 -> +1 reserves bin 0");
        assert_eq!(store.data(1), 2, "bin 2 -> -1 -> 1 -> +1");
    }

    #[test]
    fn multi_val_push_skips_most_freq_leaving_bin_zero() {
        // The most-freq value is skipped entirely in the multi-val branch, so its
        // cell stays the default 0 (the implicit most-freq slot reserved by +1).
        let f0 = mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]);
        let f1 = mapper(3, 0, vec![1.0, 2.0, f64::INFINITY]);
        let mut fg = FeatureGroup::new(2, true, vec![f0, f1], 2, 1);
        fg.push_data(0, 0, 0.5); // bin 0 == most_freq => SKIP (no push)
        fg.finish_load();
        let store = fg.multi_bin_data(0).unwrap();
        assert_eq!(store.data(0), 0, "most-freq skipped -> bin 0 reserved");
    }

    #[test]
    fn offset_zero_dense_multi_val_path_when_sum_sparse_rate_below_quarter() {
        // The `offset=0` dense-multi-val path triggers when
        // `sum_sparse_rate < multi_val_bin_sparse_threshold (0.25)` AND
        // is_multi_val (feature_group.h:51-57, RESEARCH §9). With sparse_rate 0.0
        // (< 0.25) and is_multi_val=true, offset becomes 0, so a most_freq_bin==0
        // feature contributes `num_bin - 0 = num_bin` (no -1) and bin_offsets_
        // start at 0.
        let f0 = mapper(5, 0, vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY]); // most_freq 0
        let f1 = mapper(4, 1, vec![1.0, 2.0, 3.0, f64::INFINITY]); // most_freq 1
        // group_id=1 (not 0) avoids the force-one-bin special case.
        let fg = FeatureGroup::new(2, true, vec![f0, f1], 8, 1);
        // offset=0 => f0 contributes 5 - 0 = 5; f1 contributes 4.
        //   num_total_bin_ = 0; bin_offsets_ = [0]; -> [0, 5, 9].
        assert_eq!(fg.bin_offsets_[0], 0, "offset=0 dense-multi-val seeds at 0");
        assert_eq!(fg.bin_offsets_, vec![0, 5, 9]);
        assert_eq!(fg.num_total_bin_, 9);
    }

    #[test]
    fn offset_one_when_sum_sparse_rate_at_or_above_quarter() {
        // Conversely, when sum_sparse_rate >= 0.25 the dense-multi-val path is NOT
        // taken: offset stays 1, so a most_freq_bin==0 feature contributes
        // `num_bin - 1` and bin_offsets_ start at 1.
        let mut f0 = mapper(5, 0, vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY]);
        let mut f1 = mapper(4, 1, vec![1.0, 2.0, 3.0, f64::INFINITY]);
        f0.sparse_rate_ = 0.5; // mean 0.375 >= 0.25
        f1.sparse_rate_ = 0.25;
        let fg = FeatureGroup::new(2, true, vec![f0, f1], 8, 1);
        // offset=1 => f0 (most_freq 0): 5 - 1 = 4; f1: 4.
        //   num_total_bin_ = 1; bin_offsets_ = [1] -> [1, 5, 9].
        assert_eq!(fg.bin_offsets_[0], 1, "offset=1 (not dense-multi-val)");
        assert_eq!(fg.bin_offsets_, vec![1, 5, 9]);
        assert_eq!(fg.num_total_bin_, 9);
    }
}
