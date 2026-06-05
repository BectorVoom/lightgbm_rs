//! Bit-for-bit port of Exclusive Feature Bundling (EFB) from
//! `LightGBM/src/io/dataset.cpp`: `GetConflictCount` / `MarkUsed` /
//! `FixSampleIndices` (60-105), `FindGroups` (107-244), and
//! `FastFeatureBundling` (246-323).
//!
//! Follows the `lgbm-core/src/config/set.rs` ordered-pipeline doc style AND the
//! `lgbm-core/src/random.rs` RNG-consumption discipline: EVERY random draw is
//! routed through `lgbm_core::random::Random`, seeded EXACTLY as C++
//! (`Random(num_data)` for both the per-feature group search inside
//! `FindGroups` and the final group shuffle in `FastFeatureBundling`).
//!
//! # Why EFB is the determinism endgame (RESEARCH Pitfall 5)
//!
//! EFB runs `FindGroups` TWICE — once in the natural `used_features` order, once
//! in feature-order-by-non-zero-count (a STABLE sort, count DESC) — keeps the
//! result with FEWER groups, then shuffles the groups with `Random(num_data)`.
//! Any deviation in the `Sample` / `NextShort` sequence, the stable sort, or the
//! conflict-count comparisons changes the grouping. So:
//!
//! - all randomness goes through Phase-1 `Random` (`new(num_data)` + `.sample` +
//!   `.next_short(i + 1, num_group)`), NEVER `rand::`;
//! - sorts are STABLE (`sort_by`, NEVER `sort_unstable`);
//! - the shuffle swaps BOTH parallel vectors (`features_in_group` and
//!   `group_is_multi_val`) element-wise in the SAME loop iteration — the C++
//!   note "`std::swap` for `vector<bool>` will cause the wrong result" is why the
//!   two swaps live together (RESEARCH Anti-pattern, dataset.cpp:317-319);
//! - widths are kept exact (`int`→`i32`, `data_size_t`→`i32`).
//!
//! # Inputs (mirror the C++ call args)
//!
//! Per used feature `fidx`:
//! - `sample_indices[fidx]`: the SORTED non-zero sample-row indices for that
//!   column (C++ `int** sample_indices`);
//! - `num_per_col[fidx]`: the count of those non-zeros (C++ `const int*
//!   num_per_col`);
//! - `sample_values[fidx]`: the sampled VALUES at those rows (used only by
//!   `FixSampleIndices`).
//!
//! `total_sample_cnt` is the number of sampled ROWS (C++ `data_size_t
//! total_sample_cnt`), `num_data` the full row count (the RNG seed).

use crate::bin_mapper::BinMapper;
use lgbm_core::random::Random;

/// C++ `OneFeaturePerGroup(used_features)` (`dataset.cpp:50-58`): the default
/// (no-bundle) grouping — one single-feature group per used feature, in order.
pub fn one_feature_per_group(used_features: &[i32]) -> Vec<Vec<i32>> {
    used_features.iter().map(|&f| vec![f]).collect()
}

/// C++ `int GetConflictCount(const std::vector<bool>& mark, const int* indices,
/// int num_indices, data_size_t max_cnt)` (`dataset.cpp:60-72`).
///
/// Count how many of `indices` are already marked; EARLY-RETURN `-1` the moment
/// the count exceeds `max_cnt` (the conflict-budget short-circuit).
fn get_conflict_count(mark: &[bool], indices: &[i32], num_indices: i32, max_cnt: i32) -> i32 {
    let mut ret = 0i32;
    for i in 0..num_indices as usize {
        if mark[indices[i] as usize] {
            ret += 1;
        }
        if ret > max_cnt {
            return -1;
        }
    }
    ret
}

/// C++ `void MarkUsed(std::vector<bool>* mark, const int* indices, data_size_t
/// num_indices)` (`dataset.cpp:74-80`).
fn mark_used(mark: &mut [bool], indices: &[i32], num_indices: i32) {
    for i in 0..num_indices as usize {
        mark[indices[i] as usize] = true;
    }
}

/// C++ `std::vector<int> FixSampleIndices(const BinMapper* bin_mapper, int
/// num_total_samples, int num_indices, const int* sample_indices, const double*
/// sample_values)` (`dataset.cpp:82-105`).
///
/// Returns the (dense) list of sample rows whose bin != most_freq_bin, but ONLY
/// when `default_bin != most_freq_bin` (otherwise empty — the sparse
/// representation is already correct). This changes the conflict counts EFB
/// sees, so it is transcribed verbatim (RESEARCH §11). `sample_indices` are the
/// sorted non-zero rows; `sample_values` the values at those rows.
fn fix_sample_indices(
    bin_mapper: &BinMapper,
    num_total_samples: i32,
    num_indices: i32,
    sample_indices: &[i32],
    sample_values: &[f64],
) -> Vec<i32> {
    let mut ret: Vec<i32> = Vec::new();
    if bin_mapper.default_bin_ == bin_mapper.most_freq_bin_ {
        return ret;
    }
    let mut i = 0i32;
    let mut j = 0i32;
    while i < num_total_samples {
        if j < num_indices && sample_indices[j as usize] < i {
            j += 1;
        } else if j < num_indices && sample_indices[j as usize] == i {
            if bin_mapper.value_to_bin(sample_values[j as usize]) != bin_mapper.most_freq_bin_ {
                ret.push(i);
            }
            i += 1;
        } else {
            ret.push(i);
            i += 1;
        }
    }
    ret
}

/// C++ `std::vector<std::vector<int>> FindGroups(...)` (`dataset.cpp:107-244`),
/// transcribed verbatim. Greedy single-val grouping with an RNG-sampled group
/// search, then (when `is_sparse`) a dense/sparse second-pass split.
///
/// Returns the groups (each a list of real feature indices); `multi_val_group`
/// is filled with the per-group multi-val flag.
///
/// `sample_indices[col]` / `num_per_col[col]` index by REAL column (`fidx`); a
/// `fidx >= num_sample_col` is a "filtered feature" (no samples).
#[allow(clippy::too_many_arguments)]
fn find_groups(
    bin_mappers: &[BinMapper],
    find_order: &[i32],
    sample_indices: &[Vec<i32>],
    num_per_col: &[i32],
    num_sample_col: i32,
    total_sample_cnt: i32,
    num_data: i32,
    is_use_gpu: bool,
    is_sparse: bool,
    multi_val_group: &mut Vec<i8>,
) -> Vec<Vec<i32>> {
    let max_search_group: i32 = 100;
    let max_bin_per_group: i32 = 256;
    let single_val_max_conflict_cnt: i32 = total_sample_cnt / 10000;
    multi_val_group.clear();

    let mut rand = Random::new(num_data);
    let mut features_in_group: Vec<Vec<i32>> = Vec::new();
    let mut conflict_marks: Vec<Vec<bool>> = Vec::new();
    let mut group_used_row_cnt: Vec<i32> = Vec::new();
    let mut group_total_data_cnt: Vec<i32> = Vec::new();
    let mut group_num_bin: Vec<i32> = Vec::new();

    // first round: fill the single val group
    for &fidx in find_order {
        let fidx_u = fidx as usize;
        let is_filtered_feature = fidx >= num_sample_col;
        let cur_non_zero_cnt: i32 = if is_filtered_feature {
            0
        } else {
            num_per_col[fidx_u]
        };
        let mut available_groups: Vec<i32> = Vec::new();
        for gid in 0..features_in_group.len() as i32 {
            let g = gid as usize;
            let cur_num_bin = group_num_bin[g]
                + bin_mappers[fidx_u].num_bin_
                + if bin_mappers[fidx_u].most_freq_bin_ == 0 {
                    -1
                } else {
                    0
                };
            if group_total_data_cnt[g] + cur_non_zero_cnt
                <= total_sample_cnt + single_val_max_conflict_cnt
            {
                if !is_use_gpu || cur_num_bin <= max_bin_per_group {
                    available_groups.push(gid);
                }
            }
        }
        let mut search_groups: Vec<i32> = Vec::new();
        if !available_groups.is_empty() {
            let last = available_groups.len() as i32 - 1;
            let indices = rand.sample(last, std::cmp::min(last, max_search_group - 1));
            // always push the last group
            search_groups.push(*available_groups.last().unwrap());
            for idx in indices {
                search_groups.push(available_groups[idx as usize]);
            }
        }
        let mut best_gid: i32 = -1;
        let mut best_conflict_cnt: i32 = -1;
        for gid in &search_groups {
            let g = *gid as usize;
            let rest_max_cnt =
                single_val_max_conflict_cnt - group_total_data_cnt[g] + group_used_row_cnt[g];
            let cnt = if is_filtered_feature {
                0
            } else {
                get_conflict_count(
                    &conflict_marks[g],
                    &sample_indices[fidx_u],
                    num_per_col[fidx_u],
                    rest_max_cnt,
                )
            };
            if cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2 {
                best_gid = *gid;
                best_conflict_cnt = cnt;
                break;
            }
        }
        if best_gid >= 0 {
            let g = best_gid as usize;
            features_in_group[g].push(fidx);
            group_total_data_cnt[g] += cur_non_zero_cnt;
            group_used_row_cnt[g] += cur_non_zero_cnt - best_conflict_cnt;
            if !is_filtered_feature {
                mark_used(
                    &mut conflict_marks[g],
                    &sample_indices[fidx_u],
                    num_per_col[fidx_u],
                );
            }
            group_num_bin[g] += bin_mappers[fidx_u].num_bin_
                + if bin_mappers[fidx_u].default_bin_ == 0 {
                    -1
                } else {
                    0
                };
        } else {
            features_in_group.push(vec![fidx]);
            let mut new_mark = vec![false; total_sample_cnt as usize];
            if !is_filtered_feature {
                mark_used(&mut new_mark, &sample_indices[fidx_u], num_per_col[fidx_u]);
            }
            conflict_marks.push(new_mark);
            group_total_data_cnt.push(cur_non_zero_cnt);
            group_used_row_cnt.push(cur_non_zero_cnt);
            group_num_bin.push(
                1 + bin_mappers[fidx_u].num_bin_
                    + if bin_mappers[fidx_u].most_freq_bin_ == 0 {
                        -1
                    } else {
                        0
                    },
            );
        }
    }
    if !is_sparse {
        multi_val_group.resize(features_in_group.len(), 0);
        return features_in_group;
    }

    let mut second_round_features: Vec<i32> = Vec::new();
    let mut features_in_group2: Vec<Vec<i32>> = Vec::new();
    let mut conflict_marks2: Vec<Vec<bool>> = Vec::new();

    let dense_threshold = 0.4f64;
    for gid in 0..features_in_group.len() {
        let dense_rate = group_used_row_cnt[gid] as f64 / total_sample_cnt as f64;
        if dense_rate >= dense_threshold {
            features_in_group2.push(std::mem::take(&mut features_in_group[gid]));
            conflict_marks2.push(std::mem::take(&mut conflict_marks[gid]));
        } else {
            for &fidx in &features_in_group[gid] {
                second_round_features.push(fidx);
            }
        }
    }

    features_in_group = features_in_group2;
    conflict_marks = conflict_marks2;
    multi_val_group.resize(features_in_group.len(), 0);
    if !second_round_features.is_empty() {
        features_in_group.push(Vec::new());
        conflict_marks.push(vec![false; total_sample_cnt as usize]);
        let mut is_multi_val: bool = is_use_gpu;
        let mut conflict_cnt = 0i32;
        for &fidx in &second_round_features {
            let fidx_u = fidx as usize;
            features_in_group.last_mut().unwrap().push(fidx);
            if !is_multi_val {
                let rest_max_cnt = single_val_max_conflict_cnt - conflict_cnt;
                let last_idx = conflict_marks.len() - 1;
                let cnt = get_conflict_count(
                    &conflict_marks[last_idx],
                    &sample_indices[fidx_u],
                    num_per_col[fidx_u],
                    rest_max_cnt,
                );
                conflict_cnt += cnt;
                if cnt < 0 || conflict_cnt > single_val_max_conflict_cnt {
                    is_multi_val = true;
                    continue;
                }
                mark_used(
                    &mut conflict_marks[last_idx],
                    &sample_indices[fidx_u],
                    num_per_col[fidx_u],
                );
            }
        }
        multi_val_group.push(is_multi_val as i8);
    }
    features_in_group
}

/// C++ `std::vector<std::vector<int>> FastFeatureBundling(...)`
/// (`dataset.cpp:246-323`), transcribed verbatim.
///
/// Pipeline (in EXACT C++ order):
/// 1. build `feature_non_zero_cnt` over `used_features` (filtered features -> 0);
/// 2. STABLE-sort an index permutation by non-zero count DESC -> the
///    `feature_order_by_cnt` ordering;
/// 3. `FixSampleIndices` each sampled feature (only changes rows when
///    `default_bin != most_freq_bin`), updating the per-col counts/indices used
///    for grouping;
/// 4. run `FindGroups` TWICE (natural `used_features` order, and
///    `feature_order_by_cnt`); KEEP the result with FEWER groups;
/// 5. shuffle the chosen groups with `Random(num_data)`: for `i` in
///    `0..num_group-1`, `j = rand.next_short(i + 1, num_group)`, then swap BOTH
///    `features_in_group[i]<->[j]` AND `group_is_multi_val[i]<->[j]` in the SAME
///    iteration.
///
/// Returns `(features_in_group, group_is_multi_val)`.
#[allow(clippy::too_many_arguments)]
pub fn fast_feature_bundling(
    bin_mappers: &[BinMapper],
    sample_indices: &[Vec<i32>],
    sample_values: &[Vec<f64>],
    num_per_col: &[i32],
    num_sample_col: i32,
    total_sample_cnt: i32,
    used_features: &[i32],
    num_data: i32,
    is_use_gpu: bool,
    is_sparse: bool,
) -> (Vec<Vec<i32>>, Vec<i8>) {
    // 1. per-used-feature non-zero counts (dense features sort first).
    let mut feature_non_zero_cnt: Vec<usize> = Vec::with_capacity(used_features.len());
    for &fidx in used_features {
        if fidx < num_sample_col {
            feature_non_zero_cnt.push(num_per_col[fidx as usize] as usize);
        } else {
            feature_non_zero_cnt.push(0);
        }
    }

    // 2. STABLE sort an index permutation by non-zero count, bigger first.
    let mut sorted_idx: Vec<usize> = (0..used_features.len()).collect();
    sorted_idx.sort_by(|&a, &b| feature_non_zero_cnt[b].cmp(&feature_non_zero_cnt[a]));

    let mut feature_order_by_cnt: Vec<i32> = Vec::with_capacity(sorted_idx.len());
    for &sidx in &sorted_idx {
        feature_order_by_cnt.push(used_features[sidx]);
    }

    // 3. FixSampleIndices: rebuild the per-col sample indices/counts EFB uses.
    //    We work on owned per-column copies so the two FindGroups passes see the
    //    same fixed inputs (mirrors C++ overwriting sample_indices[fidx] +
    //    tmp_num_per_col[fidx]).
    let mut fixed_indices: Vec<Vec<i32>> = sample_indices.to_vec();
    let mut tmp_num_per_col: Vec<i32> = vec![0; num_sample_col as usize];
    for &fidx in used_features {
        if fidx >= num_sample_col {
            continue;
        }
        let fidx_u = fidx as usize;
        let ret = fix_sample_indices(
            &bin_mappers[fidx_u],
            total_sample_cnt,
            num_per_col[fidx_u],
            &sample_indices[fidx_u],
            &sample_values[fidx_u],
        );
        if !ret.is_empty() {
            tmp_num_per_col[fidx_u] = ret.len() as i32;
            fixed_indices[fidx_u] = ret;
        } else {
            tmp_num_per_col[fidx_u] = num_per_col[fidx_u];
        }
    }

    // 4. FindGroups twice; keep the fewer-group result.
    let mut group_is_multi_val: Vec<i8> = Vec::new();
    let mut features_in_group = find_groups(
        bin_mappers,
        used_features,
        &fixed_indices,
        &tmp_num_per_col,
        num_sample_col,
        total_sample_cnt,
        num_data,
        is_use_gpu,
        is_sparse,
        &mut group_is_multi_val,
    );
    let mut group_is_multi_val2: Vec<i8> = Vec::new();
    let group2 = find_groups(
        bin_mappers,
        &feature_order_by_cnt,
        &fixed_indices,
        &tmp_num_per_col,
        num_sample_col,
        total_sample_cnt,
        num_data,
        is_use_gpu,
        is_sparse,
        &mut group_is_multi_val2,
    );

    if features_in_group.len() > group2.len() {
        features_in_group = group2;
        group_is_multi_val = group_is_multi_val2;
    }

    // 5. shuffle groups. Random(num_data); swap BOTH parallel vectors per iter.
    let num_group = features_in_group.len() as i32;
    let mut tmp_rand = Random::new(num_data);
    for i in 0..(num_group - 1).max(0) {
        let j = tmp_rand.next_short(i + 1, num_group);
        features_in_group.swap(i as usize, j as usize);
        // `std::swap` on vector<bool> is wrong in C++ — element-wise swap of the
        // parallel multi-val flags in the SAME iteration (RESEARCH Anti-pattern).
        group_is_multi_val.swap(i as usize, j as usize);
    }
    (features_in_group, group_is_multi_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin_mapper::{BinType, MissingType};

    fn mapper(num_bin: i32, most_freq_bin: u32, default_bin: u32) -> BinMapper {
        BinMapper {
            num_bin_: num_bin,
            missing_type_: MissingType::None,
            bin_upper_bound_: (1..num_bin).map(|i| i as f64).chain([f64::INFINITY]).collect(),
            is_trivial_: false,
            sparse_rate_: 0.0,
            bin_type_: BinType::Numerical,
            default_bin_: default_bin,
            most_freq_bin_: most_freq_bin,
            min_val_: 0.0,
            max_val_: 0.0,
            bin_2_categorical_: Vec::new(),
            categorical_2_bin_: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn one_feature_per_group_is_identity_singletons() {
        assert_eq!(one_feature_per_group(&[0, 1, 2]), vec![vec![0], vec![1], vec![2]]);
        assert!(one_feature_per_group(&[]).is_empty());
    }

    #[test]
    fn get_conflict_count_early_returns_minus_one_past_budget() {
        let mark = vec![true, true, true, false];
        // indices [0,1,2] all marked, max_cnt 1 -> exceeds after 2nd -> -1.
        assert_eq!(get_conflict_count(&mark, &[0, 1, 2], 3, 1), -1);
        // within budget: 1 marked, max 3 -> 1.
        assert_eq!(get_conflict_count(&mark, &[0, 3], 2, 3), 1);
    }

    #[test]
    fn fix_sample_indices_empty_when_default_equals_most_freq() {
        // default_bin == most_freq_bin -> early empty (sparse repr is correct).
        let m = mapper(3, 0, 0);
        let out = fix_sample_indices(&m, 4, 1, &[2], &[1.5]);
        assert!(out.is_empty());
    }

    #[test]
    fn fix_sample_indices_densifies_when_default_differs_from_most_freq() {
        // default_bin (0) != most_freq_bin (1): walk all rows, keep rows whose
        // bin != most_freq. Sampled row 1 has value 0.5 -> bin 0 (!= 1) -> kept;
        // unsampled rows 0,2,3 are treated as value 0 -> default bin 0 (!= 1)
        // -> kept. So all four rows are returned.
        let m = mapper(3, 1, 0); // bounds [1,2,inf]
        let out = fix_sample_indices(&m, 4, 1, &[1], &[0.5]);
        assert_eq!(out, vec![0, 1, 2, 3]);
    }

    #[test]
    fn two_mutually_exclusive_sparse_features_bundle_into_one_group() {
        // Classic EFB win: two features whose non-zero rows never overlap bundle
        // into a single group (zero conflicts). 4 sampled rows; feature 0 nonzero
        // at row 0, feature 1 nonzero at row 1 — disjoint.
        let bin_mappers = vec![mapper(3, 0, 0), mapper(3, 0, 0)];
        let sample_indices = vec![vec![0], vec![1]];
        let sample_values = vec![vec![2.5], vec![2.5]];
        let num_per_col = vec![1, 1];
        let (groups, multi) = fast_feature_bundling(
            &bin_mappers,
            &sample_indices,
            &sample_values,
            &num_per_col,
            2,    // num_sample_col
            4,    // total_sample_cnt
            &[0, 1],
            4,     // num_data (RNG seed)
            false, // is_use_gpu
            false, // is_sparse (single-val pass only)
        );
        // Disjoint non-zeros -> one bundled group containing both features.
        assert_eq!(groups.len(), 1, "mutually-exclusive features bundle");
        assert_eq!(groups[0].len(), 2);
        let mut sorted = groups[0].clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
        assert_eq!(multi.len(), 1);
    }

    #[test]
    fn conflicting_features_stay_in_separate_groups() {
        // Both features non-zero at the SAME row -> conflict -> cannot bundle
        // (single_val_max_conflict_cnt = 4/10000 = 0, so any conflict splits).
        let bin_mappers = vec![mapper(3, 0, 0), mapper(3, 0, 0)];
        let sample_indices = vec![vec![0], vec![0]]; // both nonzero at row 0
        let sample_values = vec![vec![2.5], vec![2.5]];
        let num_per_col = vec![1, 1];
        let (groups, _multi) = fast_feature_bundling(
            &bin_mappers,
            &sample_indices,
            &sample_values,
            &num_per_col,
            2,
            4,
            &[0, 1],
            4,
            false,
            false,
        );
        assert_eq!(groups.len(), 2, "conflicting features do not bundle");
    }
}
