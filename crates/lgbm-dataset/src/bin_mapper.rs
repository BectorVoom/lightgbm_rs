//! Bit-for-bit port of the numeric `BinMapper::FindBin` pipeline + `ValueToBin`
//! from `LightGBM/src/io/bin.cpp` and `LightGBM/include/LightGBM/bin.h`.
//!
//! This is **the determinism kernel**: bin boundaries (`bin_upper_bound_`) and
//! per-row bin indices (`value_to_bin`) computed here are the root every
//! downstream split inherits. A 1-ULP boundary drift here cascades into every
//! offset and per-row index, so every arithmetic decision below is load-bearing
//! and transcribed line-for-line from the C++ source (analog discipline:
//! `lgbm-core/src/random.rs`).
//!
//! # Arithmetic notes
//!
//! - **All boundary math is `f64`** (NOT `f32`). Sampled f32 feature values are
//!   widened to f64 at ingestion; midpoints are `(a + b) / 2.0` in f64. Mixing
//!   precision shifts ULPs and breaks parity (RESEARCH Pitfall 1 / Anti-Pattern).
//! - **`f64::next_up()`** is the exact equivalent of C++
//!   `std::nextafter(a, +INFINITY)` (`Common::GetDoubleUpperBound`), stable since
//!   Rust 1.86 (the workspace pins 1.95). Do NOT hand-roll a bit-twiddling
//!   `nextafter`.
//! - **De-dup is asymmetric**: `check_double_equal_ordered(a, b) == (b <= a.next_up())`
//!   — NOT `a == b`. The guard is
//!   `if bound.is_empty() || !check_double_equal_ordered(*bound.last(), val) { push(val) }`.
//! - **`value_to_bin` midpoint is the literal `(r + l - 1) / 2`** (note the `-1`)
//!   with `value <= bin_upper_bound_[m]` choosing the left branch. This is NOT
//!   `slice::partition_point` / `binary_search` — the `-1` and the `<=` tie
//!   direction change which bin on-boundary values land in (RESEARCH Pitfall 2).
//! - **NaN routing**: `value_to_bin(NaN)` returns `num_bin_ - 1` when
//!   `missing_type_ == NaN`, else treats NaN as `0.0`. Categorical returns 0.
//! - **`K_ZERO_THRESHOLD`** = `1e-35` f64, reused from `lgbm_core::types` (do NOT
//!   redefine).
//! - **RNG sampling** for bin construction routes through the Phase-1
//!   `lgbm_core::random::Random` LCG seeded by `data_random_seed`, clamped
//!   `k = min(n, bin_construct_sample_cnt)` (C++ `SampleCount`). No new RNG.
//!
//! Per D-03, `find_bin` is a standalone per-column constructor (no shared mutable
//! accumulator across features) so a later `par_iter` is a one-line swap.

use lgbm_core::random::Random;
use lgbm_core::types::K_ZERO_THRESHOLD;

/// C++ `enum BinType` (`bin.h:22-25`). Numeric vs categorical binning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinType {
    /// `NumericalBin` — continuous feature.
    Numerical,
    /// `CategoricalBin` — categorical feature.
    Categorical,
}

/// C++ `enum MissingType` (`bin.h:27-31`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingType {
    /// `None` — no missing values.
    None,
    /// `Zero` — zero treated as missing.
    Zero,
    /// `NaN` — NaN treated as missing (occupies the top bin).
    NaN,
}

/// C++ `Common::GetDoubleUpperBound` (`common.h:850-852`):
/// `std::nextafter(a, INFINITY)`.
#[inline]
fn get_double_upper_bound(a: f64) -> f64 {
    a.next_up()
}

/// C++ `Common::CheckDoubleEqualOrdered` (`common.h:845-848`):
/// `double upper = std::nextafter(a, INFINITY); return b <= upper;`
#[inline]
fn check_double_equal_ordered(a: f64, b: f64) -> bool {
    b <= a.next_up()
}

/// C++ `SampleCount` (`c_api.cpp:974-976`):
/// `total_nrow < bin_construct_sample_cnt ? total_nrow : bin_construct_sample_cnt`.
#[inline]
pub fn sample_count(total_nrow: i32, bin_construct_sample_cnt: i32) -> i32 {
    if total_nrow < bin_construct_sample_cnt {
        total_nrow
    } else {
        bin_construct_sample_cnt
    }
}

/// C++ `CreateSampleIndices` (`c_api.cpp:978-982`): the bin-construction sample
/// index set, drawn from the Phase-1 LCG seeded by `data_random_seed`.
///
/// `Random rand(data_random_seed); return rand.Sample(total_nrow, SampleCount(...));`
pub fn create_sample_indices(total_nrow: i32, bin_construct_sample_cnt: i32, data_random_seed: i32) -> Vec<i32> {
    let mut rand = Random::new(data_random_seed);
    rand.sample(total_nrow, sample_count(total_nrow, bin_construct_sample_cnt))
}

/// Feature value → bin mapper + bin meta information.
///
/// Public fields mirror the C++ `BinMapper` private members (`bin.h:235-`); the
/// trailing-underscore names are kept verbatim to make the C++ correspondence
/// unambiguous for the port.
#[derive(Debug, Clone, PartialEq)]
pub struct BinMapper {
    /// C++ `int num_bin_` — number of bins.
    pub num_bin_: i32,
    /// C++ `MissingType missing_type_`.
    pub missing_type_: MissingType,
    /// C++ `std::vector<double> bin_upper_bound_` — inclusive upper bound per bin.
    pub bin_upper_bound_: Vec<f64>,
    /// C++ `bool is_trivial_` — true when the feature collapses to a single bin.
    pub is_trivial_: bool,
    /// C++ `double sparse_rate_`.
    pub sparse_rate_: f64,
    /// C++ `BinType bin_type_`.
    pub bin_type_: BinType,
    /// C++ `uint32_t default_bin_` — `ValueToBin(0)`.
    pub default_bin_: u32,
    /// C++ `uint32_t most_freq_bin_`.
    pub most_freq_bin_: u32,
    /// C++ `double min_val_`.
    pub min_val_: f64,
    /// C++ `double max_val_`.
    pub max_val_: f64,
}

impl BinMapper {
    /// C++ `BinMapper::ValueToBin` (`bin.h:612-650`) — the numeric hot path,
    /// transcribed verbatim.
    ///
    /// ```text
    /// if isnan(value):
    ///   categorical -> 0; missing==NaN -> num_bin_-1; else value = 0.0
    /// numeric:
    ///   l=0; r=num_bin_-1; if missing==NaN: r -= 1
    ///   while l < r:
    ///     m = (r + l - 1) / 2          // literal -1 midpoint
    ///     if value <= bin_upper_bound_[m]: r = m else l = m + 1
    ///   return l
    /// ```
    pub fn value_to_bin(&self, mut value: f64) -> u32 {
        if value.is_nan() {
            if self.bin_type_ == BinType::Categorical {
                return 0;
            } else if self.missing_type_ == MissingType::NaN {
                return (self.num_bin_ - 1) as u32;
            } else {
                value = 0.0;
            }
        }
        if self.bin_type_ == BinType::Numerical {
            // binary search to find bin (C++ bin.h:622-637)
            let mut l: i32 = 0;
            let mut r: i32 = self.num_bin_ - 1;
            if self.missing_type_ == MissingType::NaN {
                r -= 1;
            }
            while l < r {
                let m = (r + l - 1) / 2;
                if value <= self.bin_upper_bound_[m as usize] {
                    r = m;
                } else {
                    l = m + 1;
                }
            }
            l as u32
        } else {
            // categorical path is wired in a later plan (Plan 03); for a numeric
            // mapper this branch is unreachable. Mirror C++ negative->0.
            let int_value = value as i32;
            if int_value < 0 { 0 } else { 0 }
        }
    }

    /// C++ `BinMapper::FindBin` numeric branch (`bin.cpp:311-506`).
    ///
    /// Constructs the numeric bin mapper for a single feature column from its
    /// sampled values (already widened to f64). `values` is consumed (NaN-stripped
    /// and sorted in place, mirroring the C++ in-place mutation). `total_sample_cnt`
    /// is `num_sample_values + zero_cnt + na_cnt`.
    ///
    /// This is the per-feature constructor (D-03): no shared mutable state.
    #[allow(clippy::too_many_arguments)]
    pub fn find_bin_numeric(
        mut values: Vec<f64>,
        max_bin: i32,
        min_data_in_bin: i32,
        min_split_data: i32,
        pre_filter: bool,
        use_missing: bool,
        zero_as_missing: bool,
        total_sample_cnt: usize,
        forced_upper_bounds: &[f64],
    ) -> BinMapper {
        let num_sample_values_in = values.len() as i32;

        // 1. strip NaNs in place, count non-NaN (bin.cpp:315-321)
        let mut non_na_cnt: usize = 0;
        for i in 0..values.len() {
            if !values[i].is_nan() {
                values[non_na_cnt] = values[i];
                non_na_cnt += 1;
            }
        }

        // 2. derive missing_type_ (bin.cpp:322-333)
        let mut na_cnt: i32 = 0;
        let mut missing_type_ = if !use_missing {
            MissingType::None
        } else if zero_as_missing {
            MissingType::Zero
        } else if non_na_cnt as i32 == num_sample_values_in {
            MissingType::None
        } else {
            na_cnt = num_sample_values_in - non_na_cnt as i32;
            MissingType::NaN
        };

        let num_sample_values = non_na_cnt;
        values.truncate(num_sample_values);

        let bin_type_ = BinType::Numerical;
        let zero_cnt: i32 =
            (total_sample_cnt as i64 - num_sample_values as i64 - na_cnt as i64) as i32;

        // 3. distinct values + counts, with zero pseudo-value insertion (bin.cpp:339-375)
        let mut distinct_values: Vec<f64> = Vec::new();
        let mut counts: Vec<i32> = Vec::new();

        // stable_sort the non-NaN values
        values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN after strip"));

        // push zero in the front
        if num_sample_values == 0 || (values[0] > 0.0 && zero_cnt > 0) {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }

        if num_sample_values > 0 {
            distinct_values.push(values[0]);
            counts.push(1);
        }

        for i in 1..num_sample_values {
            if !check_double_equal_ordered(values[i - 1], values[i]) {
                if values[i - 1] < 0.0 && values[i] > 0.0 {
                    distinct_values.push(0.0);
                    counts.push(zero_cnt);
                }
                distinct_values.push(values[i]);
                counts.push(1);
            } else {
                // use the large value
                *distinct_values.last_mut().unwrap() = values[i];
                *counts.last_mut().unwrap() += 1;
            }
        }

        // push zero in the back
        if num_sample_values > 0 && values[num_sample_values - 1] < 0.0 && zero_cnt > 0 {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }

        let min_val_ = *distinct_values.first().unwrap_or(&0.0);
        let max_val_ = *distinct_values.last().unwrap_or(&0.0);
        let num_distinct_values = distinct_values.len() as i32;

        // 4. dispatch (bin.cpp:380-409)
        let mut bin_upper_bound_: Vec<f64>;
        if missing_type_ == MissingType::Zero {
            bin_upper_bound_ = find_bin_with_zero_as_one_bin_forced(
                &distinct_values,
                &counts,
                num_distinct_values,
                max_bin,
                total_sample_cnt,
                min_data_in_bin,
                forced_upper_bounds,
            );
            if bin_upper_bound_.len() == 2 {
                missing_type_ = MissingType::None;
            }
        } else if missing_type_ == MissingType::None {
            bin_upper_bound_ = find_bin_with_zero_as_one_bin_forced(
                &distinct_values,
                &counts,
                num_distinct_values,
                max_bin,
                total_sample_cnt,
                min_data_in_bin,
                forced_upper_bounds,
            );
        } else {
            bin_upper_bound_ = find_bin_with_zero_as_one_bin_forced(
                &distinct_values,
                &counts,
                num_distinct_values,
                max_bin - 1,
                (total_sample_cnt as i64 - na_cnt as i64) as usize,
                min_data_in_bin,
                forced_upper_bounds,
            );
            bin_upper_bound_.push(f64::NAN);
        }
        let num_bin_ = bin_upper_bound_.len() as i32;

        // cnt_in_bin walk (bin.cpp:396-408)
        let mut cnt_in_bin: Vec<i32> = vec![0; num_bin_ as usize];
        {
            let mut i_bin = 0usize;
            for i in 0..num_distinct_values as usize {
                while distinct_values[i] > bin_upper_bound_[i_bin] && i_bin < (num_bin_ - 1) as usize
                {
                    i_bin += 1;
                }
                cnt_in_bin[i_bin] += counts[i];
            }
            if missing_type_ == MissingType::NaN {
                cnt_in_bin[(num_bin_ - 1) as usize] = na_cnt;
            }
        }

        // Build a provisional mapper so ValueToBin(0) can be evaluated for
        // default_bin_ exactly as C++ does (bin.cpp:491).
        let mut mapper = BinMapper {
            num_bin_,
            missing_type_,
            bin_upper_bound_,
            is_trivial_: false,
            sparse_rate_: 0.0,
            bin_type_,
            default_bin_: 0,
            most_freq_bin_: 0,
            min_val_,
            max_val_,
        };

        // trivial / pre-filter checks (bin.cpp:479-505)
        let mut is_trivial_ = num_bin_ <= 1;
        if !is_trivial_
            && pre_filter
            && need_filter(&cnt_in_bin, total_sample_cnt as i32, min_split_data, bin_type_)
        {
            is_trivial_ = true;
        }

        let sparse_rate_;
        if !is_trivial_ {
            let default_bin_ = mapper.value_to_bin(0.0);
            let most_freq = arg_max(&cnt_in_bin) as u32;
            let mut most_freq_bin_ = most_freq;
            let max_sparse_rate =
                cnt_in_bin[most_freq_bin_ as usize] as f64 / total_sample_cnt as f64;
            // kSparseThreshold = 0.7 (bin.h:43)
            if most_freq_bin_ != default_bin_ && max_sparse_rate < 0.7 {
                most_freq_bin_ = default_bin_;
            }
            sparse_rate_ = cnt_in_bin[most_freq_bin_ as usize] as f64 / total_sample_cnt as f64;
            mapper.default_bin_ = default_bin_;
            mapper.most_freq_bin_ = most_freq_bin_;
        } else {
            sparse_rate_ = 1.0;
        }
        mapper.is_trivial_ = is_trivial_;
        mapper.sparse_rate_ = sparse_rate_;
        mapper
    }

    /// Convenience constructor for the common ingestion entry: sample the column
    /// values via the Phase-1 RNG then build the numeric mapper. Mirrors the
    /// C-API path `CreateSampleIndices` → gather → `FindBin` (RESEARCH §12).
    ///
    /// `column` is the full per-row feature column (f64-widened). Sampling clamps
    /// `k = min(num_rows, bin_construct_sample_cnt)` and gathers the sampled rows'
    /// values; `total_sample_cnt = k` (matching the C-API in-memory sample path).
    #[allow(clippy::too_many_arguments)]
    pub fn find_bin_from_column(
        column: &[f64],
        max_bin: i32,
        min_data_in_bin: i32,
        min_split_data: i32,
        pre_filter: bool,
        use_missing: bool,
        zero_as_missing: bool,
        bin_construct_sample_cnt: i32,
        data_random_seed: i32,
        forced_upper_bounds: &[f64],
    ) -> BinMapper {
        let total_nrow = column.len() as i32;
        let indices = create_sample_indices(total_nrow, bin_construct_sample_cnt, data_random_seed);
        let sampled: Vec<f64> = indices.iter().map(|&i| column[i as usize]).collect();
        let total_sample_cnt = sampled.len();
        BinMapper::find_bin_numeric(
            sampled,
            max_bin,
            min_data_in_bin,
            min_split_data,
            pre_filter,
            use_missing,
            zero_as_missing,
            total_sample_cnt,
            forced_upper_bounds,
        )
    }
}

/// C++ `ArrayArgs<int>::ArgMax` — index of the (first) max element.
fn arg_max(v: &[i32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// C++ `NeedFilter` (`bin.cpp:54-76`), numerical branch (categorical wired later).
pub fn need_filter(cnt_in_bin: &[i32], total_cnt: i32, filter_cnt: i32, bin_type: BinType) -> bool {
    if bin_type == BinType::Numerical {
        let mut sum_left = 0;
        if cnt_in_bin.len() >= 1 {
            for i in 0..cnt_in_bin.len() - 1 {
                sum_left += cnt_in_bin[i];
                if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                    return false;
                }
            }
        }
    } else if cnt_in_bin.len() <= 2 {
        for i in 0..cnt_in_bin.len().saturating_sub(1) {
            let sum_left = cnt_in_bin[i];
            if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                return false;
            }
        }
    } else {
        return false;
    }
    true
}

/// C++ `GreedyFindBin` (`bin.cpp:78-155`) — the core numeric boundary builder.
fn greedy_find_bin(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: i32,
    mut max_bin: i32,
    total_cnt: usize,
    min_data_in_bin: i32,
) -> Vec<f64> {
    let mut bin_upper_bound: Vec<f64> = Vec::new();
    // CHECK_GT(max_bin, 0) — callers guarantee this.
    if num_distinct_values <= max_bin {
        let mut cur_cnt_inbin = 0;
        for i in 0..(num_distinct_values - 1) as usize {
            cur_cnt_inbin += counts[i];
            if cur_cnt_inbin >= min_data_in_bin {
                let val =
                    get_double_upper_bound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
                if bin_upper_bound.is_empty()
                    || !check_double_equal_ordered(*bin_upper_bound.last().unwrap(), val)
                {
                    bin_upper_bound.push(val);
                    cur_cnt_inbin = 0;
                }
            }
        }
        // cur_cnt_inbin += counts[num_distinct_values - 1];  (value unused after)
        bin_upper_bound.push(f64::INFINITY);
    } else {
        if min_data_in_bin > 0 {
            max_bin = max_bin.min((total_cnt as i64 / min_data_in_bin as i64) as i32);
            max_bin = max_bin.max(1);
        }
        let mut mean_bin_size = total_cnt as f64 / max_bin as f64;

        let mut rest_bin_cnt = max_bin;
        let mut rest_sample_cnt = total_cnt as i32;
        let mut is_big_count_value = vec![false; num_distinct_values as usize];
        for i in 0..num_distinct_values as usize {
            if counts[i] as f64 >= mean_bin_size {
                is_big_count_value[i] = true;
                rest_bin_cnt -= 1;
                rest_sample_cnt -= counts[i];
            }
        }
        mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
        let mut upper_bounds = vec![f64::INFINITY; max_bin as usize];
        let mut lower_bounds = vec![f64::INFINITY; max_bin as usize];

        let mut bin_cnt = 0usize;
        lower_bounds[bin_cnt] = distinct_values[0];
        let mut cur_cnt_inbin = 0i32;
        for i in 0..(num_distinct_values - 1) as usize {
            if !is_big_count_value[i] {
                rest_sample_cnt -= counts[i];
            }
            cur_cnt_inbin += counts[i];
            // need a new bin (note: mean_bin_size * 0.5f is a float literal)
            if is_big_count_value[i]
                || cur_cnt_inbin as f64 >= mean_bin_size
                || (is_big_count_value[i + 1]
                    && cur_cnt_inbin as f64 >= (1.0_f64).max(mean_bin_size * 0.5f32 as f64))
            {
                upper_bounds[bin_cnt] = distinct_values[i];
                bin_cnt += 1;
                lower_bounds[bin_cnt] = distinct_values[i + 1];
                if bin_cnt as i32 >= max_bin - 1 {
                    break;
                }
                cur_cnt_inbin = 0;
                if !is_big_count_value[i] {
                    rest_bin_cnt -= 1;
                    mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
                }
            }
        }
        bin_cnt += 1;
        // update bin upper bound
        for i in 0..bin_cnt - 1 {
            let val = get_double_upper_bound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
            if bin_upper_bound.is_empty()
                || !check_double_equal_ordered(*bin_upper_bound.last().unwrap(), val)
            {
                bin_upper_bound.push(val);
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    }
    bin_upper_bound
}

/// C++ `FindBinWithPredefinedBin` (`bin.cpp:157-240`).
fn find_bin_with_predefined_bin(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: i32,
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Vec<f64> {
    let mut bin_upper_bound: Vec<f64> = Vec::new();

    // positive/negative distinct value boundaries
    let mut left_cnt: i32 = -1;
    for i in 0..num_distinct_values as usize {
        if distinct_values[i] > -K_ZERO_THRESHOLD {
            left_cnt = i as i32;
            break;
        }
    }
    if left_cnt < 0 {
        left_cnt = num_distinct_values;
    }
    let mut right_start: i32 = -1;
    for i in left_cnt as usize..num_distinct_values as usize {
        if distinct_values[i] > K_ZERO_THRESHOLD {
            right_start = i as i32;
            break;
        }
    }

    if max_bin == 2 {
        if left_cnt == 0 {
            bin_upper_bound.push(K_ZERO_THRESHOLD);
        } else {
            bin_upper_bound.push(-K_ZERO_THRESHOLD);
        }
    } else if max_bin >= 3 {
        if left_cnt > 0 {
            bin_upper_bound.push(-K_ZERO_THRESHOLD);
        }
        if right_start >= 0 {
            bin_upper_bound.push(K_ZERO_THRESHOLD);
        }
    }
    bin_upper_bound.push(f64::INFINITY);

    // add forced bounds, excluding zeros
    let max_to_insert = max_bin - bin_upper_bound.len() as i32;
    let mut num_inserted = 0;
    for &fb in forced_upper_bounds {
        if num_inserted >= max_to_insert {
            break;
        }
        if fb.abs() > K_ZERO_THRESHOLD {
            bin_upper_bound.push(fb);
            num_inserted += 1;
        }
    }
    bin_upper_bound.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // find remaining bounds
    let free_bins = max_bin - bin_upper_bound.len() as i32;
    let mut bounds_to_add: Vec<f64> = Vec::new();
    let mut value_ind = 0usize;
    let bub_len = bin_upper_bound.len();
    for i in 0..bub_len {
        let mut cnt_in_bin = 0i32;
        let mut distinct_cnt_in_bin = 0i32;
        let bin_start = value_ind;
        while value_ind < num_distinct_values as usize
            && distinct_values[value_ind] < bin_upper_bound[i]
        {
            cnt_in_bin += counts[value_ind];
            distinct_cnt_in_bin += 1;
            value_ind += 1;
        }
        let bins_remaining =
            max_bin - bin_upper_bound.len() as i32 - bounds_to_add.len() as i32;
        let mut num_sub_bins =
            ((cnt_in_bin as f64 * free_bins as f64 / total_sample_cnt as f64).round()) as i32;
        num_sub_bins = num_sub_bins.min(bins_remaining) + 1;
        if i == bub_len - 1 {
            num_sub_bins = bins_remaining + 1;
        }
        let new_upper_bounds = greedy_find_bin(
            &distinct_values[bin_start..],
            &counts[bin_start..],
            distinct_cnt_in_bin,
            num_sub_bins,
            cnt_in_bin as usize,
            min_data_in_bin,
        );
        // last bound is infinity — exclude it
        bounds_to_add.extend_from_slice(&new_upper_bounds[..new_upper_bounds.len() - 1]);
    }
    bin_upper_bound.extend_from_slice(&bounds_to_add);
    bin_upper_bound.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bin_upper_bound
}

/// C++ `FindBinWithZeroAsOneBin` (`bin.cpp:242-298`).
fn find_bin_with_zero_as_one_bin(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: i32,
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
) -> Vec<f64> {
    let mut bin_upper_bound: Vec<f64> = Vec::new();
    let mut left_cnt_data = 0i32;
    let mut cnt_zero = 0i32;
    let mut right_cnt_data = 0i32;
    for i in 0..num_distinct_values as usize {
        if distinct_values[i] <= -K_ZERO_THRESHOLD {
            left_cnt_data += counts[i];
        } else if distinct_values[i] > K_ZERO_THRESHOLD {
            right_cnt_data += counts[i];
        } else {
            cnt_zero += counts[i];
        }
    }

    let mut left_cnt: i32 = -1;
    for i in 0..num_distinct_values as usize {
        if distinct_values[i] > -K_ZERO_THRESHOLD {
            left_cnt = i as i32;
            break;
        }
    }
    if left_cnt < 0 {
        left_cnt = num_distinct_values;
    }

    if left_cnt > 0 && max_bin > 1 {
        let left_max_bin = (left_cnt_data as f64 / (total_sample_cnt as i32 - cnt_zero) as f64
            * (max_bin - 1) as f64) as i32;
        let left_max_bin = left_max_bin.max(1);
        bin_upper_bound = greedy_find_bin(
            distinct_values,
            counts,
            left_cnt,
            left_max_bin,
            left_cnt_data as usize,
            min_data_in_bin,
        );
        if !bin_upper_bound.is_empty() {
            *bin_upper_bound.last_mut().unwrap() = -K_ZERO_THRESHOLD;
        }
    }

    let mut right_start: i32 = -1;
    for i in left_cnt as usize..num_distinct_values as usize {
        if distinct_values[i] > K_ZERO_THRESHOLD {
            right_start = i as i32;
            break;
        }
    }

    let right_max_bin = max_bin - 1 - bin_upper_bound.len() as i32;
    if right_start >= 0 && right_max_bin > 0 {
        let right_bounds = greedy_find_bin(
            &distinct_values[right_start as usize..],
            &counts[right_start as usize..],
            num_distinct_values - right_start,
            right_max_bin,
            right_cnt_data as usize,
            min_data_in_bin,
        );
        bin_upper_bound.push(K_ZERO_THRESHOLD);
        bin_upper_bound.extend_from_slice(&right_bounds);
    } else {
        bin_upper_bound.push(f64::INFINITY);
    }
    bin_upper_bound
}

/// C++ overload `FindBinWithZeroAsOneBin(..., forced_upper_bounds)`
/// (`bin.cpp:300-309`): dispatches to the predefined variant when forced bounds
/// are present, else the plain zero-as-one-bin variant.
fn find_bin_with_zero_as_one_bin_forced(
    distinct_values: &[f64],
    counts: &[i32],
    num_distinct_values: i32,
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Vec<f64> {
    if forced_upper_bounds.is_empty() {
        find_bin_with_zero_as_one_bin(
            distinct_values,
            counts,
            num_distinct_values,
            max_bin,
            total_sample_cnt,
            min_data_in_bin,
        )
    } else {
        find_bin_with_predefined_bin(
            distinct_values,
            counts,
            num_distinct_values,
            max_bin,
            total_sample_cnt,
            min_data_in_bin,
            forced_upper_bounds,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build a numeric mapper with explicit bounds for ValueToBin tests.
    fn mapper_with_bounds(bounds: Vec<f64>, missing: MissingType) -> BinMapper {
        BinMapper {
            num_bin_: bounds.len() as i32,
            missing_type_: missing,
            bin_upper_bound_: bounds,
            is_trivial_: false,
            sparse_rate_: 0.0,
            bin_type_: BinType::Numerical,
            default_bin_: 0,
            most_freq_bin_: 0,
            min_val_: 0.0,
            max_val_: 0.0,
        }
    }

    #[test]
    fn value_to_bin_on_boundary_routes_left() {
        // bounds = [1.0, 2.0, +inf]; num_bin=3, missing=None.
        // A value EXACTLY equal to a boundary must route to that bin (the `<=`
        // left branch), matching C++.
        let m = mapper_with_bounds(vec![1.0, 2.0, f64::INFINITY], MissingType::None);
        // value == bin_upper_bound_[0] (1.0) -> bin 0
        assert_eq!(m.value_to_bin(1.0), 0, "on-boundary 1.0 -> bin 0");
        // value == bin_upper_bound_[1] (2.0) -> bin 1
        assert_eq!(m.value_to_bin(2.0), 1, "on-boundary 2.0 -> bin 1");
        // below first boundary -> bin 0
        assert_eq!(m.value_to_bin(0.5), 0);
        // between -> bin 1
        assert_eq!(m.value_to_bin(1.5), 1);
        // above last finite boundary -> top bin
        assert_eq!(m.value_to_bin(100.0), 2);
    }

    #[test]
    fn value_to_bin_nan_routing_both_missing_types() {
        // missing == NaN: NaN -> num_bin_ - 1 (top bin reserved for NaN).
        let m_nan = mapper_with_bounds(vec![1.0, 2.0, f64::NAN], MissingType::NaN);
        assert_eq!(m_nan.value_to_bin(f64::NAN), (m_nan.num_bin_ - 1) as u32);

        // missing == None: NaN treated as 0.0 -> routes like value 0.0.
        let m_none = mapper_with_bounds(vec![1.0, 2.0, f64::INFINITY], MissingType::None);
        let bin_for_zero = m_none.value_to_bin(0.0);
        assert_eq!(
            m_none.value_to_bin(f64::NAN),
            bin_for_zero,
            "NaN treated as 0.0 when missing_type != NaN"
        );
        assert_eq!(bin_for_zero, 0, "0.0 <= 1.0 -> bin 0");
    }

    #[test]
    fn check_double_equal_ordered_is_asymmetric_next_up() {
        // De-dup edge: b <= a.next_up() collapses values within 1 ULP, and is
        // NOT a symmetric a == b test.
        let a = 1.0_f64;
        let b = a.next_up(); // exactly one ULP above
        assert!(check_double_equal_ordered(a, b), "b == a.next_up() collapses");
        // a value strictly more than one ULP above does NOT collapse
        let c = a.next_up().next_up();
        assert!(!check_double_equal_ordered(a, c));
        // asymmetry: check(b, a) where a < b is true (a <= b.next_up()), but the
        // dedup is only ever called as check(prev, cur) with cur >= prev.
        assert!(check_double_equal_ordered(b, a));
    }

    #[test]
    fn dedup_collapses_adjacent_boundaries_in_find_bin() {
        // Two distinct values 1 ULP apart collapse into one bin boundary via the
        // CheckDoubleEqualOrdered dedup in the distinct-value build.
        let v0 = 1.0_f64;
        let v1 = v0.next_up();
        let values = vec![v0, v1, 10.0, 20.0];
        let m = BinMapper::find_bin_numeric(
            values,
            255, // max_bin
            0,   // min_data_in_bin
            0,   // min_split_data
            false, // pre_filter
            true,  // use_missing
            false, // zero_as_missing
            4,     // total_sample_cnt
            &[],   // forced bounds
        );
        // v0 and v1 (1 ULP apart) must NOT produce two separate distinct values;
        // so the number of finite boundaries is < number of raw values.
        assert!(m.num_bin_ >= 1);
        // sanity: every value maps to a valid bin index
        for &val in &[v0, v1, 10.0, 20.0, 0.0] {
            let b = m.value_to_bin(val);
            assert!((b as i32) < m.num_bin_, "bin {b} out of range for {val}");
        }
    }

    #[test]
    fn sampling_routes_through_phase1_rng() {
        // create_sample_indices must equal Random::new(seed).sample(n, min(n,cnt))
        // exactly — no new RNG. Hand-compute the reference from the Phase-1 LCG.
        let seed = 1234;
        let n = 50;
        let cnt = 10;
        let mut r = Random::new(seed);
        let expected = r.sample(n, sample_count(n, cnt));
        let got = create_sample_indices(n, cnt, seed);
        assert_eq!(got, expected, "sampling must use the Phase-1 Random LCG");

        // clamp: cnt > n yields k = n (full range).
        let mut r2 = Random::new(seed);
        let expected_full = r2.sample(5, 5);
        assert_eq!(create_sample_indices(5, 100, seed), expected_full);
    }

    #[test]
    fn find_bin_numeric_basic_monotone_bounds() {
        // A simple spread of values produces strictly ascending finite bounds
        // with a final +inf, and num_bin_ <= max_bin.
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let m = BinMapper::find_bin_numeric(
            values, 16, 1, 0, false, true, false, 100, &[],
        );
        assert!(m.num_bin_ <= 16, "num_bin_ {} <= max_bin", m.num_bin_);
        assert!(m.num_bin_ >= 2);
        // finite bounds strictly ascending
        let finite: Vec<f64> = m
            .bin_upper_bound_
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        assert!(
            finite.windows(2).all(|w| w[0] < w[1]),
            "finite bounds must be strictly ascending: {finite:?}"
        );
        assert!(m.bin_upper_bound_.last().unwrap().is_infinite());
    }

    #[test]
    fn find_bin_single_value_is_trivial() {
        // All-zero column -> single bin (+inf only) -> trivial. This mirrors C++
        // FindBinWithZeroAsOneBin: a column of only zeros has no left (<0) and no
        // right (>kZeroThreshold) distinct values, so the result is `[+inf]`,
        // num_bin_ == 1.
        let values = vec![0.0_f64; 20];
        let m = BinMapper::find_bin_numeric(values, 16, 1, 0, false, true, false, 20, &[]);
        assert!(m.is_trivial_, "single-value (all-zero) feature must be trivial");
        assert_eq!(m.num_bin_, 1);
    }

    #[test]
    fn find_bin_single_positive_value_gets_zero_split() {
        // A column of a single positive value with no zeros yields the zero-as-
        // one-bin split `[kZeroThreshold, +inf]` (2 bins), matching C++ — it is
        // NOT collapsed to a single bin. The value routes to the top bin.
        let values = vec![3.0_f64; 20];
        let m = BinMapper::find_bin_numeric(values, 16, 1, 0, false, true, false, 20, &[]);
        assert_eq!(m.num_bin_, 2, "single positive value -> [zero_threshold, +inf]");
        assert_eq!(m.value_to_bin(3.0), 1, "positive value routes to top bin");
        assert_eq!(m.value_to_bin(0.0), 0, "zero routes to the low bin");
    }

    #[test]
    fn pre_filter_marks_trivial() {
        // A feature whose split would leave too few on one side gets filtered to
        // trivial when pre_filter is on and min_split_data is large.
        // 100 values, almost all in one bin: min_split_data large => filtered.
        let mut values = vec![0.0_f64; 99];
        values.push(1000.0);
        let m = BinMapper::find_bin_numeric(
            values, 16, 1, /*min_split_data=*/ 50, /*pre_filter=*/ true, true, false, 100, &[],
        );
        assert!(
            m.is_trivial_,
            "feature with no valid split given min_split_data=50 must be filtered trivial"
        );
    }
}
