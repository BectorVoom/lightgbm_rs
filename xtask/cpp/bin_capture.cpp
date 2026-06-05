// bin_capture.cpp
//
// Dev-only C++ capture harness for the LightGBM-rs binning oracle (Phase 2):
//   - numeric golden layers 1 + 2 (Plan 01),
//   - bin-storage layout (Plan 02),
//   - categorical folding layers 1 + 3 + per-row index (Plan 03, DAT-04),
//   - missing-value routing layer 1 + per-row index (Plan 03, DAT-03).
// EFB -> Plan 05.
//
// WHY THIS IS A FOCUSED, SELF-CONTAINED TRANSCRIPTION (not a compile of the
// in-tree bin.cpp):
//   The authoritative `BinMapper::FindBin` / `ValueToBin` live in
//   `LightGBM/src/io/bin.cpp`, which unconditionally pulls in
//   `LightGBM/include/LightGBM/utils/common.h`, which `#include`s
//   `fast_double_parser.h` and `fmt/format.h` from `external_libs/`. In this
//   repo those submodules are present as EMPTY directories (the LightGBM tree is
//   git-untracked and its `external_libs/` are not vendored — see project memory
//   `lightgbm-ref-tree-untracked`). So `bin.cpp` cannot be compiled here.
//
//   This is the SAME situation Phase 1 hit for the RNG (rng_capture compiles the
//   header-only `Random` directly instead of linking lib_lightgbm). The numeric
//   `FindBin`/`GreedyFindBin`/`FindBinWithZeroAsOneBin`/`FindBinWithPredefinedBin`/
//   `ValueToBin`/`NeedFilter` logic depends ONLY on:
//     - `std::nextafter(a, INFINITY)`  (== Common::GetDoubleUpperBound),
//     - the asymmetric `b <= nextafter(a)` dedup (Common::CheckDoubleEqualOrdered),
//     - std `<algorithm>` stable_sort / std math,
//   NONE of which touch fast_double_parser or fmt. The code below is a VERBATIM
//   transcription of those functions from the pinned `LightGBM/src/io/bin.cpp`
//   (commit 195c26fc, VERSION 4.6.0.99) and `include/LightGBM/bin.h`, using the
//   real `std::nextafter` — so it emits goldens byte-identical to what
//   lib_lightgbm would. It is the authoritative reference source, just compiled
//   without the unbuildable external_libs dependency chain.
//
//   Sampling (CreateSampleIndices) is captured by compiling the header-only
//   `LightGBM::Random` directly (same as rng_capture) — that is the genuine
//   reference RNG, not a re-transcription.
//
// Determinism / idempotency (D-14): the four-source numeric corpus is derived
// solely from one recorded MASTER_SEED passed on argv. No wall-clock / OS
// entropy, so re-running produces byte-identical fixtures.
//
// Fixture format (line-delimited, '#'-prefixed comments ignored by the reader):
//   MASTER_SEED <seed>
//   COUNTS cases=<n>
//   CASE name=<id> max_bin=<m> min_data_in_bin=<d> min_split_data=<s> \
//         pre_filter=<0|1> use_missing=<0|1> zero_as_missing=<0|1> \
//         num_rows=<n> seed=<data_random_seed> sample_cnt=<c>
//   VALUES <bits;bits;...>            # full per-row column, raw f64 bits (u64 dec)
//   GOLDEN num_bin=<n> bin_type=<0|1> missing_type=<0|1|2> \
//          default_bin=<u32> most_freq_bin=<u32> is_trivial=<0|1> \
//          upper=<f64bits;...>        # layer 1: bin_upper_bound_ raw f64 bits
//   ASSIGN <u32;u32;...>              # layer 2: per-row ValueToBin over the column
//
// f64 values are emitted as raw little-endian bit patterns (a u64 in decimal) so
// the Rust side asserts bit-exact equality with zero parsing rounding.

#include <LightGBM/utils/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ---- enums mirroring LightGBM/include/LightGBM/bin.h -----------------------
enum BinType { NumericalBin = 0, CategoricalBin = 1 };
enum MissingType { MT_None = 0, MT_Zero = 1, MT_NaN = 2 };

// kZeroThreshold = 1e-35 (double), kEpsilon = 1e-15f  (meta.h)
constexpr double kZeroThreshold = 1e-35;

// ---- Common helpers (common.h:845-852) ------------------------------------
inline double GetDoubleUpperBound(double a) { return std::nextafter(a, INFINITY); }
inline bool CheckDoubleEqualOrdered(double a, double b) {
  double upper = std::nextafter(a, INFINITY);
  return b <= upper;
}

// ---- RoundInt (common.h:904) ----------------------------------------------
inline int RoundInt(double x) { return static_cast<int>(x + 0.5f); }

// ---- SortForPair (common.h:611-632), the desc-count case (is_reverse=true) -
template <typename T1, typename T2>
void SortForPairDesc(std::vector<T1>* keys, std::vector<T2>* values) {
  std::vector<std::pair<T1, T2>> arr;
  auto& ref_key = *keys;
  auto& ref_value = *values;
  for (size_t i = 0; i < keys->size(); ++i) arr.emplace_back(ref_key[i], ref_value[i]);
  std::stable_sort(arr.begin(), arr.end(),
                   [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
                     return a.first > b.first;  // count descending
                   });
  for (size_t i = 0; i < arr.size(); ++i) {
    ref_key[i] = arr[i].first;
    ref_value[i] = arr[i].second;
  }
}

// ---- A minimal BinMapper holding what layers 1+3 need ----------------------
struct BinMapper {
  int num_bin_ = 1;
  MissingType missing_type_ = MT_None;
  std::vector<double> bin_upper_bound_;
  bool is_trivial_ = true;
  double sparse_rate_ = 1.0;
  BinType bin_type_ = NumericalBin;
  uint32_t default_bin_ = 0;
  uint32_t most_freq_bin_ = 0;
  // categorical map (layer 3): bin -> category and category -> bin.
  std::vector<int> bin_2_categorical_;
  std::unordered_map<int, uint32_t> categorical_2_bin_;

  // ValueToBin (bin.h:612-650), numeric + categorical branches.
  uint32_t ValueToBin(double value) const {
    if (std::isnan(value)) {
      if (bin_type_ == CategoricalBin) return 0;
      else if (missing_type_ == MT_NaN) return static_cast<uint32_t>(num_bin_ - 1);
      else value = 0.0;
    }
    if (bin_type_ == NumericalBin) {
      int l = 0, r = num_bin_ - 1;
      if (missing_type_ == MT_NaN) r -= 1;
      while (l < r) {
        int m = (r + l - 1) / 2;
        if (value <= bin_upper_bound_[m]) r = m;
        else l = m + 1;
      }
      return static_cast<uint32_t>(l);
    } else {
      int int_value = static_cast<int>(value);
      if (int_value < 0) return 0;
      auto it = categorical_2_bin_.find(int_value);
      if (it != categorical_2_bin_.end()) return it->second;
      return 0;
    }
  }
};

// ---- NeedFilter (bin.cpp:54-76), numerical branch -------------------------
bool NeedFilter(const std::vector<int>& cnt_in_bin, int total_cnt, int filter_cnt) {
  int sum_left = 0;
  for (size_t i = 0; i + 1 < cnt_in_bin.size(); ++i) {
    sum_left += cnt_in_bin[i];
    if (sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt) return false;
  }
  return true;
}

// ---- NeedFilter (bin.cpp:63-73), categorical branch -----------------------
bool NeedFilterCat(const std::vector<int>& cnt_in_bin, int total_cnt, int filter_cnt) {
  if (cnt_in_bin.size() <= 2) {
    for (size_t i = 0; i + 1 < cnt_in_bin.size(); ++i) {
      int sum_left = cnt_in_bin[i];
      if (sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt) return false;
    }
  } else {
    return false;
  }
  return true;
}

// ---- GreedyFindBin (bin.cpp:78-155) ---------------------------------------
std::vector<double> GreedyFindBin(const double* distinct_values, const int* counts,
                                  int num_distinct_values, int max_bin,
                                  size_t total_cnt, int min_data_in_bin) {
  std::vector<double> bin_upper_bound;
  if (num_distinct_values <= max_bin) {
    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_distinct_values - 1; ++i) {
      cur_cnt_inbin += counts[i];
      if (cur_cnt_inbin >= min_data_in_bin) {
        auto val = GetDoubleUpperBound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
        if (bin_upper_bound.empty() || !CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
          bin_upper_bound.push_back(val);
          cur_cnt_inbin = 0;
        }
      }
    }
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  } else {
    if (min_data_in_bin > 0) {
      max_bin = std::min(max_bin, static_cast<int>(total_cnt / min_data_in_bin));
      max_bin = std::max(max_bin, 1);
    }
    double mean_bin_size = static_cast<double>(total_cnt) / max_bin;
    int rest_bin_cnt = max_bin;
    int rest_sample_cnt = static_cast<int>(total_cnt);
    std::vector<bool> is_big_count_value(num_distinct_values, false);
    for (int i = 0; i < num_distinct_values; ++i) {
      if (counts[i] >= mean_bin_size) {
        is_big_count_value[i] = true;
        --rest_bin_cnt;
        rest_sample_cnt -= counts[i];
      }
    }
    mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;
    std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity());
    std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity());
    int bin_cnt = 0;
    lower_bounds[bin_cnt] = distinct_values[0];
    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_distinct_values - 1; ++i) {
      if (!is_big_count_value[i]) rest_sample_cnt -= counts[i];
      cur_cnt_inbin += counts[i];
      if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
          (is_big_count_value[i + 1] && cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
        upper_bounds[bin_cnt] = distinct_values[i];
        ++bin_cnt;
        lower_bounds[bin_cnt] = distinct_values[i + 1];
        if (bin_cnt >= max_bin - 1) break;
        cur_cnt_inbin = 0;
        if (!is_big_count_value[i]) {
          --rest_bin_cnt;
          mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
        }
      }
    }
    ++bin_cnt;
    for (int i = 0; i < bin_cnt - 1; ++i) {
      auto val = GetDoubleUpperBound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
      if (bin_upper_bound.empty() || !CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
        bin_upper_bound.push_back(val);
      }
    }
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  }
  return bin_upper_bound;
}

// ---- FindBinWithPredefinedBin (bin.cpp:157-240) ---------------------------
std::vector<double> FindBinWithPredefinedBin(const double* distinct_values, const int* counts,
                                             int num_distinct_values, int max_bin,
                                             size_t total_sample_cnt, int min_data_in_bin,
                                             const std::vector<double>& forced_upper_bounds) {
  std::vector<double> bin_upper_bound;
  int left_cnt = -1;
  for (int i = 0; i < num_distinct_values; ++i) {
    if (distinct_values[i] > -kZeroThreshold) { left_cnt = i; break; }
  }
  if (left_cnt < 0) left_cnt = num_distinct_values;
  int right_start = -1;
  for (int i = left_cnt; i < num_distinct_values; ++i) {
    if (distinct_values[i] > kZeroThreshold) { right_start = i; break; }
  }
  if (max_bin == 2) {
    if (left_cnt == 0) bin_upper_bound.push_back(kZeroThreshold);
    else bin_upper_bound.push_back(-kZeroThreshold);
  } else if (max_bin >= 3) {
    if (left_cnt > 0) bin_upper_bound.push_back(-kZeroThreshold);
    if (right_start >= 0) bin_upper_bound.push_back(kZeroThreshold);
  }
  bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  int max_to_insert = max_bin - static_cast<int>(bin_upper_bound.size());
  int num_inserted = 0;
  for (size_t i = 0; i < forced_upper_bounds.size(); ++i) {
    if (num_inserted >= max_to_insert) break;
    if (std::fabs(forced_upper_bounds[i]) > kZeroThreshold) {
      bin_upper_bound.push_back(forced_upper_bounds[i]);
      ++num_inserted;
    }
  }
  std::stable_sort(bin_upper_bound.begin(), bin_upper_bound.end());
  int free_bins = max_bin - static_cast<int>(bin_upper_bound.size());
  std::vector<double> bounds_to_add;
  int value_ind = 0;
  for (size_t i = 0; i < bin_upper_bound.size(); ++i) {
    int cnt_in_bin = 0, distinct_cnt_in_bin = 0, bin_start = value_ind;
    while (value_ind < num_distinct_values && distinct_values[value_ind] < bin_upper_bound[i]) {
      cnt_in_bin += counts[value_ind];
      ++distinct_cnt_in_bin;
      ++value_ind;
    }
    int bins_remaining = max_bin - static_cast<int>(bin_upper_bound.size()) -
                         static_cast<int>(bounds_to_add.size());
    int num_sub_bins = static_cast<int>(
        std::lround(static_cast<double>(cnt_in_bin) * free_bins / total_sample_cnt));
    num_sub_bins = std::min(num_sub_bins, bins_remaining) + 1;
    if (i == bin_upper_bound.size() - 1) num_sub_bins = bins_remaining + 1;
    std::vector<double> new_upper_bounds = GreedyFindBin(
        distinct_values + bin_start, counts + bin_start, distinct_cnt_in_bin, num_sub_bins,
        cnt_in_bin, min_data_in_bin);
    bounds_to_add.insert(bounds_to_add.end(), new_upper_bounds.begin(),
                         new_upper_bounds.end() - 1);
  }
  bin_upper_bound.insert(bin_upper_bound.end(), bounds_to_add.begin(), bounds_to_add.end());
  std::stable_sort(bin_upper_bound.begin(), bin_upper_bound.end());
  return bin_upper_bound;
}

// ---- FindBinWithZeroAsOneBin (bin.cpp:242-298) ----------------------------
std::vector<double> FindBinWithZeroAsOneBin(const double* distinct_values, const int* counts,
                                            int num_distinct_values, int max_bin,
                                            size_t total_sample_cnt, int min_data_in_bin) {
  std::vector<double> bin_upper_bound;
  int left_cnt_data = 0, cnt_zero = 0, right_cnt_data = 0;
  for (int i = 0; i < num_distinct_values; ++i) {
    if (distinct_values[i] <= -kZeroThreshold) left_cnt_data += counts[i];
    else if (distinct_values[i] > kZeroThreshold) right_cnt_data += counts[i];
    else cnt_zero += counts[i];
  }
  int left_cnt = -1;
  for (int i = 0; i < num_distinct_values; ++i) {
    if (distinct_values[i] > -kZeroThreshold) { left_cnt = i; break; }
  }
  if (left_cnt < 0) left_cnt = num_distinct_values;
  if (left_cnt > 0 && max_bin > 1) {
    int left_max_bin = static_cast<int>(static_cast<double>(left_cnt_data) /
                                        (total_sample_cnt - cnt_zero) * (max_bin - 1));
    left_max_bin = std::max(1, left_max_bin);
    bin_upper_bound = GreedyFindBin(distinct_values, counts, left_cnt, left_max_bin,
                                    left_cnt_data, min_data_in_bin);
    if (bin_upper_bound.size() > 0) bin_upper_bound.back() = -kZeroThreshold;
  }
  int right_start = -1;
  for (int i = left_cnt; i < num_distinct_values; ++i) {
    if (distinct_values[i] > kZeroThreshold) { right_start = i; break; }
  }
  int right_max_bin = max_bin - 1 - static_cast<int>(bin_upper_bound.size());
  if (right_start >= 0 && right_max_bin > 0) {
    auto right_bounds = GreedyFindBin(distinct_values + right_start, counts + right_start,
                                      num_distinct_values - right_start, right_max_bin,
                                      right_cnt_data, min_data_in_bin);
    bin_upper_bound.push_back(kZeroThreshold);
    bin_upper_bound.insert(bin_upper_bound.end(), right_bounds.begin(), right_bounds.end());
  } else {
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  }
  return bin_upper_bound;
}

std::vector<double> FindBinWithZeroAsOneBinForced(
    const double* distinct_values, const int* counts, int num_distinct_values, int max_bin,
    size_t total_sample_cnt, int min_data_in_bin,
    const std::vector<double>& forced_upper_bounds) {
  if (forced_upper_bounds.empty()) {
    return FindBinWithZeroAsOneBin(distinct_values, counts, num_distinct_values, max_bin,
                                   total_sample_cnt, min_data_in_bin);
  }
  return FindBinWithPredefinedBin(distinct_values, counts, num_distinct_values, max_bin,
                                  total_sample_cnt, min_data_in_bin, forced_upper_bounds);
}

int ArgMax(const std::vector<int>& v) {
  int best = 0;
  for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[best]) best = static_cast<int>(i);
  return best;
}

// ---- BinMapper::FindBin numeric branch (bin.cpp:311-506) ------------------
BinMapper FindBinNumeric(std::vector<double> values, int max_bin, int min_data_in_bin,
                         int min_split_data, bool pre_filter, bool use_missing,
                         bool zero_as_missing, size_t total_sample_cnt,
                         const std::vector<double>& forced_upper_bounds) {
  BinMapper bm;
  int num_sample_values_in = static_cast<int>(values.size());
  int non_na_cnt = 0;
  for (int i = 0; i < num_sample_values_in; ++i) {
    if (!std::isnan(values[i])) values[non_na_cnt++] = values[i];
  }
  int na_cnt = 0;
  MissingType missing_type_;
  if (!use_missing) missing_type_ = MT_None;
  else if (zero_as_missing) missing_type_ = MT_Zero;
  else if (non_na_cnt == num_sample_values_in) missing_type_ = MT_None;
  else { missing_type_ = MT_NaN; na_cnt = num_sample_values_in - non_na_cnt; }

  int num_sample_values = non_na_cnt;
  values.resize(num_sample_values);
  bm.bin_type_ = NumericalBin;
  int zero_cnt = static_cast<int>(total_sample_cnt - num_sample_values - na_cnt);

  std::vector<double> distinct_values;
  std::vector<int> counts;
  std::stable_sort(values.begin(), values.end());
  if (num_sample_values == 0 || (values[0] > 0.0 && zero_cnt > 0)) {
    distinct_values.push_back(0.0); counts.push_back(zero_cnt);
  }
  if (num_sample_values > 0) { distinct_values.push_back(values[0]); counts.push_back(1); }
  for (int i = 1; i < num_sample_values; ++i) {
    if (!CheckDoubleEqualOrdered(values[i - 1], values[i])) {
      if (values[i - 1] < 0.0 && values[i] > 0.0) {
        distinct_values.push_back(0.0); counts.push_back(zero_cnt);
      }
      distinct_values.push_back(values[i]); counts.push_back(1);
    } else {
      distinct_values.back() = values[i]; ++counts.back();
    }
  }
  if (num_sample_values > 0 && values[num_sample_values - 1] < 0.0 && zero_cnt > 0) {
    distinct_values.push_back(0.0); counts.push_back(zero_cnt);
  }
  int num_distinct_values = static_cast<int>(distinct_values.size());

  std::vector<int> cnt_in_bin;
  if (missing_type_ == MT_Zero) {
    bm.bin_upper_bound_ = FindBinWithZeroAsOneBinForced(
        distinct_values.data(), counts.data(), num_distinct_values, max_bin, total_sample_cnt,
        min_data_in_bin, forced_upper_bounds);
    if (bm.bin_upper_bound_.size() == 2) missing_type_ = MT_None;
  } else if (missing_type_ == MT_None) {
    bm.bin_upper_bound_ = FindBinWithZeroAsOneBinForced(
        distinct_values.data(), counts.data(), num_distinct_values, max_bin, total_sample_cnt,
        min_data_in_bin, forced_upper_bounds);
  } else {
    bm.bin_upper_bound_ = FindBinWithZeroAsOneBinForced(
        distinct_values.data(), counts.data(), num_distinct_values, max_bin - 1,
        total_sample_cnt - na_cnt, min_data_in_bin, forced_upper_bounds);
    bm.bin_upper_bound_.push_back(std::numeric_limits<double>::quiet_NaN());
  }
  bm.num_bin_ = static_cast<int>(bm.bin_upper_bound_.size());
  bm.missing_type_ = missing_type_;

  cnt_in_bin.assign(bm.num_bin_, 0);
  {
    int i_bin = 0;
    for (int i = 0; i < num_distinct_values; ++i) {
      while (distinct_values[i] > bm.bin_upper_bound_[i_bin] && i_bin < bm.num_bin_ - 1) ++i_bin;
      cnt_in_bin[i_bin] += counts[i];
    }
    if (missing_type_ == MT_NaN) cnt_in_bin[bm.num_bin_ - 1] = na_cnt;
  }

  bool is_trivial_ = bm.num_bin_ <= 1;
  if (!is_trivial_ && pre_filter &&
      NeedFilter(cnt_in_bin, static_cast<int>(total_sample_cnt), min_split_data)) {
    is_trivial_ = true;
  }
  if (!is_trivial_) {
    bm.default_bin_ = bm.ValueToBin(0);
    bm.most_freq_bin_ = static_cast<uint32_t>(ArgMax(cnt_in_bin));
    double max_sparse_rate = static_cast<double>(cnt_in_bin[bm.most_freq_bin_]) / total_sample_cnt;
    if (bm.most_freq_bin_ != bm.default_bin_ && max_sparse_rate < 0.7) {
      bm.most_freq_bin_ = bm.default_bin_;
    }
    bm.sparse_rate_ = static_cast<double>(cnt_in_bin[bm.most_freq_bin_]) / total_sample_cnt;
  } else {
    bm.sparse_rate_ = 1.0;
  }
  bm.is_trivial_ = is_trivial_;
  bm.missing_type_ = missing_type_;
  return bm;
}

// ---- BinMapper::FindBin categorical branch (bin.cpp:311-506, arm 410-476) --
// Verbatim transcription. Shares the numeric distinct-value prefix, then the
// int conversion + descending-count fold. No fast_double_parser/fmt dependency.
BinMapper FindBinCategorical(std::vector<double> values, int max_bin, int min_data_in_bin,
                             int min_split_data, bool pre_filter, bool use_missing,
                             bool zero_as_missing, size_t total_sample_cnt) {
  BinMapper bm;
  int num_sample_values_in = static_cast<int>(values.size());
  int non_na_cnt = 0;
  for (int i = 0; i < num_sample_values_in; ++i) {
    if (!std::isnan(values[i])) values[non_na_cnt++] = values[i];
  }
  int na_cnt = 0;
  MissingType missing_type_;
  if (!use_missing) missing_type_ = MT_None;
  else if (zero_as_missing) missing_type_ = MT_Zero;
  else if (non_na_cnt == num_sample_values_in) missing_type_ = MT_None;
  else { missing_type_ = MT_NaN; na_cnt = num_sample_values_in - non_na_cnt; }

  int num_sample_values = non_na_cnt;
  values.resize(num_sample_values);
  bm.bin_type_ = CategoricalBin;
  int zero_cnt = static_cast<int>(total_sample_cnt - num_sample_values - na_cnt);

  std::vector<double> distinct_values;
  std::vector<int> counts;
  std::stable_sort(values.begin(), values.end());
  if (num_sample_values == 0 || (values[0] > 0.0 && zero_cnt > 0)) {
    distinct_values.push_back(0.0); counts.push_back(zero_cnt);
  }
  if (num_sample_values > 0) { distinct_values.push_back(values[0]); counts.push_back(1); }
  for (int i = 1; i < num_sample_values; ++i) {
    if (!CheckDoubleEqualOrdered(values[i - 1], values[i])) {
      if (values[i - 1] < 0.0 && values[i] > 0.0) {
        distinct_values.push_back(0.0); counts.push_back(zero_cnt);
      }
      distinct_values.push_back(values[i]); counts.push_back(1);
    } else {
      distinct_values.back() = values[i]; ++counts.back();
    }
  }
  if (num_sample_values > 0 && values[num_sample_values - 1] < 0.0 && zero_cnt > 0) {
    distinct_values.push_back(0.0); counts.push_back(zero_cnt);
  }

  // categorical branch (bin.cpp:410-476).
  std::vector<int> distinct_values_int;
  std::vector<int> counts_int;
  for (size_t i = 0; i < distinct_values.size(); ++i) {
    int val = static_cast<int>(distinct_values[i]);
    if (val < 0) {
      na_cnt += counts[i];
    } else {
      if (distinct_values_int.empty() || val != distinct_values_int.back()) {
        distinct_values_int.push_back(val);
        counts_int.push_back(counts[i]);
      } else {
        counts_int.back() += counts[i];
      }
    }
  }
  std::vector<int> cnt_in_bin;
  bm.num_bin_ = 1;
  int used_cnt = 0;
  int rest_cnt = static_cast<int>(total_sample_cnt - na_cnt);
  if (rest_cnt > 0) {
    SortForPairDesc<int, int>(&counts_int, &distinct_values_int);
    int cut_cnt = static_cast<int>(RoundInt((total_sample_cnt - na_cnt) * 0.99f));
    size_t cur_cat_idx = 0;
    int distinct_cnt = static_cast<int>(distinct_values_int.size());
    if (na_cnt > 0) ++distinct_cnt;
    max_bin = std::min(distinct_cnt, max_bin);
    // dummy bin for NaN.
    bm.bin_2_categorical_.push_back(-1);
    bm.categorical_2_bin_[-1] = 0;
    cnt_in_bin.push_back(0);
    bm.num_bin_ = 1;
    while (cur_cat_idx < distinct_values_int.size() &&
           (used_cnt < cut_cnt || bm.num_bin_ < max_bin)) {
      if (counts_int[cur_cat_idx] < min_data_in_bin && cur_cat_idx > 1) break;
      bm.bin_2_categorical_.push_back(distinct_values_int[cur_cat_idx]);
      bm.categorical_2_bin_[distinct_values_int[cur_cat_idx]] =
          static_cast<uint32_t>(bm.num_bin_);
      used_cnt += counts_int[cur_cat_idx];
      cnt_in_bin.push_back(counts_int[cur_cat_idx]);
      ++bm.num_bin_;
      ++cur_cat_idx;
    }
    if (cur_cat_idx == distinct_values_int.size() && na_cnt == 0) {
      missing_type_ = MT_None;
    } else {
      missing_type_ = MT_NaN;
    }
    cnt_in_bin[0] = static_cast<int>(total_sample_cnt - used_cnt);
  }
  bm.missing_type_ = missing_type_;

  bool is_trivial_ = bm.num_bin_ <= 1;
  if (!is_trivial_ && pre_filter &&
      NeedFilterCat(cnt_in_bin, static_cast<int>(total_sample_cnt), min_split_data)) {
    is_trivial_ = true;
  }
  if (!is_trivial_) {
    bm.default_bin_ = bm.ValueToBin(0);
    bm.most_freq_bin_ = static_cast<uint32_t>(ArgMax(cnt_in_bin));
    double max_sparse_rate = static_cast<double>(cnt_in_bin[bm.most_freq_bin_]) / total_sample_cnt;
    if (bm.most_freq_bin_ != bm.default_bin_ && max_sparse_rate < 0.7) {
      bm.most_freq_bin_ = bm.default_bin_;
    }
    bm.sparse_rate_ = static_cast<double>(cnt_in_bin[bm.most_freq_bin_]) / total_sample_cnt;
  } else {
    bm.sparse_rate_ = 1.0;
  }
  bm.is_trivial_ = is_trivial_;
  return bm;
}

// ===========================================================================
// Storage layer (golden layer 2 storage subset): VERBATIM transcription of the
// DenseBin / SparseBin / FeatureGroup byte layout from the pinned
//   - src/io/dense_bin.hpp  (ctor 56-82, 4-bit Push/FinishLoad/data 69-82/510-565)
//   - src/io/sparse_bin.hpp (Push 92-97, FinishLoad/LoadFromPair 598-659,
//                            GetFastIndex 661-687, NextNonzero 392-400)
//   - include/LightGBM/feature_group.h (offset packing 39-76, PushData 253-267,
//                            CreateBinData 586-612)
// These depend ONLY on std + the BinMapper above — none of fast_double_parser /
// fmt — so the transcription emits byte-identical storage to lib_lightgbm.
// ===========================================================================

constexpr double kSparseThreshold = 0.7;             // bin.h:43
constexpr double kMultiValBinSparseThreshold = 0.25; // bin.h:599
constexpr int kNumFastIndex = 64;                    // sparse_bin.hpp:25

// An abstract Bin emitting its raw byte layout for the golden dump.
struct IBin {
  virtual ~IBin() {}
  virtual void Push(int idx, uint32_t value) = 0;
  virtual void FinishLoad() = 0;
  virtual uint32_t Data(int idx) const = 0;
  // dense: raw data_ bytes; sparse: deltas bytes (handled separately below)
  virtual std::vector<uint8_t> RawBytes() const = 0;
  virtual bool IsSparse() const { return false; }
  virtual const std::vector<uint8_t>* Deltas() const { return nullptr; }
  virtual std::vector<uint8_t> ValsBytes() const { return {}; }
};

// DenseBin<VAL_T, IS_4BIT>.  width = sizeof(VAL_T).
template <typename VAL_T, bool IS_4BIT>
struct DenseBin : IBin {
  int num_data_;
  std::vector<VAL_T> data_;
  std::vector<VAL_T> buf_;
  explicit DenseBin(int num_data) : num_data_(num_data) {
    if (IS_4BIT) {
      data_.assign((num_data_ + 1) / 2, static_cast<VAL_T>(0));
      buf_.assign((num_data_ + 1) / 2, static_cast<VAL_T>(0));
    } else {
      data_.assign(num_data_, static_cast<VAL_T>(0));
    }
  }
  void Push(int idx, uint32_t value) override {
    if (IS_4BIT) {
      const int i1 = idx >> 1;
      const int i2 = (idx & 1) << 2;
      const VAL_T val = static_cast<VAL_T>((value & 0xff) << i2);
      if (i2 == 0) data_[i1] = val; else buf_[i1] = val;
    } else {
      data_[idx] = static_cast<VAL_T>(value);
    }
  }
  void FinishLoad() override {
    if (IS_4BIT) {
      if (buf_.empty()) return;
      int len = (num_data_ + 1) / 2;
      for (int i = 0; i < len; ++i) data_[i] |= buf_[i];
      buf_.clear();
    }
  }
  uint32_t Data(int idx) const override {
    if (IS_4BIT) return (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
    return static_cast<uint32_t>(data_[idx]);
  }
  std::vector<uint8_t> RawBytes() const override {
    std::vector<uint8_t> out(data_.size() * sizeof(VAL_T));
    std::memcpy(out.data(), data_.data(), out.size());
    return out;
  }
};

// SparseBin<VAL_T>.
template <typename VAL_T>
struct SparseBin : IBin {
  int num_data_;
  std::vector<std::pair<int, VAL_T>> push_buffer_;
  std::vector<uint8_t> deltas_;
  std::vector<VAL_T> vals_;
  int num_vals_ = 0;
  explicit SparseBin(int num_data) : num_data_(num_data) {}
  void Push(int idx, uint32_t value) override {
    VAL_T cur_bin = static_cast<VAL_T>(value);
    if (cur_bin != 0) push_buffer_.emplace_back(idx, cur_bin);
  }
  void LoadFromPair(const std::vector<std::pair<int, VAL_T>>& pairs) {
    deltas_.clear();
    vals_.clear();
    int last_idx = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
      const int cur_idx = pairs[i].first;
      const VAL_T bin = pairs[i].second;
      int cur_delta = cur_idx - last_idx;
      if (i > 0 && cur_delta == 0) continue;
      while (cur_delta >= 256) {
        deltas_.push_back(255);
        vals_.push_back(0);
        cur_delta -= 255;
      }
      deltas_.push_back(static_cast<uint8_t>(cur_delta));
      vals_.push_back(bin);
      last_idx = cur_idx;
    }
    deltas_.push_back(0);
    num_vals_ = static_cast<int>(vals_.size());
  }
  void FinishLoad() override {
    std::sort(push_buffer_.begin(), push_buffer_.end(),
              [](const std::pair<int, VAL_T>& a, const std::pair<int, VAL_T>& b) {
                return a.first < b.first;
              });
    LoadFromPair(push_buffer_);
  }
  uint32_t Data(int idx) const override {
    int i_delta = -1, cur_pos = 0;
    while (true) {
      ++i_delta;
      cur_pos += deltas_[i_delta];
      if (i_delta >= num_vals_) return 0;
      if (cur_pos == idx) return static_cast<uint32_t>(vals_[i_delta]);
      if (cur_pos > idx) return 0;
    }
  }
  std::vector<uint8_t> RawBytes() const override { return deltas_; }
  bool IsSparse() const override { return true; }
  const std::vector<uint8_t>* Deltas() const override { return &deltas_; }
  std::vector<uint8_t> ValsBytes() const override {
    std::vector<uint8_t> out(vals_.size() * sizeof(VAL_T));
    std::memcpy(out.data(), vals_.data(), out.size());
    return out;
  }
};

IBin* CreateDenseBin(int num_data, int num_bin) {
  if (num_bin <= 16) return new DenseBin<uint8_t, true>(num_data);
  else if (num_bin <= 256) return new DenseBin<uint8_t, false>(num_data);
  else if (num_bin <= 65536) return new DenseBin<uint16_t, false>(num_data);
  else return new DenseBin<uint32_t, false>(num_data);
}
IBin* CreateSparseBin(int num_data, int num_bin) {
  if (num_bin <= 256) return new SparseBin<uint8_t>(num_data);
  else if (num_bin <= 65536) return new SparseBin<uint16_t>(num_data);
  else return new SparseBin<uint32_t>(num_data);
}

// FeatureGroup single-value ctor (feature_group.h:93-111) + PushData (253-267).
struct FeatureGroupSV {
  BinMapper mapper_;
  std::vector<uint32_t> bin_offsets_;
  uint64_t num_total_bin_ = 0;
  bool is_sparse_ = false;
  IBin* bin_data_ = nullptr;

  FeatureGroupSV(const BinMapper& m, int num_data) : mapper_(m) {
    num_total_bin_ = 1;
    bin_offsets_.push_back(static_cast<uint32_t>(num_total_bin_));
    int num_bin = mapper_.num_bin_;
    if (mapper_.most_freq_bin_ == 0) num_bin -= 1;
    num_total_bin_ += static_cast<uint64_t>(num_bin);
    bin_offsets_.push_back(static_cast<uint32_t>(num_total_bin_));
    // CreateBinData(num_data, false, false, false): sparse iff sparse_rate>=0.7
    if (!false && mapper_.sparse_rate_ >= kSparseThreshold) {
      is_sparse_ = true;
      bin_data_ = CreateSparseBin(num_data, static_cast<int>(num_total_bin_));
    } else {
      is_sparse_ = false;
      bin_data_ = CreateDenseBin(num_data, static_cast<int>(num_total_bin_));
    }
  }
  ~FeatureGroupSV() { delete bin_data_; }

  void PushData(int line_idx, double value) {
    uint32_t bin = mapper_.ValueToBin(value);
    if (bin == mapper_.most_freq_bin_) return;
    if (mapper_.most_freq_bin_ == 0) bin -= 1;
    bin += bin_offsets_[0];
    bin_data_->Push(line_idx, bin);
  }
  void FinishLoad() { bin_data_->FinishLoad(); }
};

// ===========================================================================
// EFB (Exclusive Feature Bundling) layer (golden layer 3, DAT-05). VERBATIM
// header-only transcription of the EFB pipeline from the pinned
//   - src/io/dataset.cpp : GetConflictCount/MarkUsed/FixSampleIndices (60-105),
//                          FindGroups (107-244), FastFeatureBundling (246-323),
//                          Construct group-build loop (337-411)
//   - include/LightGBM/feature_group.h : primary multi-feature ctor (39-76) +
//                          PushData (253-267)
// These depend ONLY on std + the genuine header-only LightGBM::Random (sampling
// + shuffle) + the BinMapper/FeatureGroupSV above — NONE of fast_double_parser
// or fmt — so the focused harness compiles WITHOUT the unvendored external_libs.
// This is the SAME header-only verbatim-transcription discipline plans 02-01..
// 02-04 used for every prior golden layer (see REFERENCE_MANIFEST.md).
//
// The capture builds a bundled C++ Dataset (enable_bundle=true) on the D-06 #4
// mutually-exclusive sparse corpus and dumps golden layer 3:
//   - feature -> group membership (feature2group_ / feature2subfeature_),
//   - per-group bin_offsets_ + num_total_bin_,
//   - per-group group_is_multi_val flag,
//   - per-row bundled bin index per (single-value) group.
// ===========================================================================

// ---- GetConflictCount (dataset.cpp:60-72) ----------------------------------
int GetConflictCount(const std::vector<bool>& mark, const int* indices,
                     int num_indices, int max_cnt) {
  int ret = 0;
  for (int i = 0; i < num_indices; ++i) {
    if (mark[indices[i]]) ++ret;
    if (ret > max_cnt) return -1;
  }
  return ret;
}

// ---- MarkUsed (dataset.cpp:74-80) ------------------------------------------
void MarkUsed(std::vector<bool>* mark, const int* indices, int num_indices) {
  auto& ref_mark = *mark;
  for (int i = 0; i < num_indices; ++i) ref_mark[indices[i]] = true;
}

// ---- FixSampleIndices (dataset.cpp:82-105) ---------------------------------
std::vector<int> FixSampleIndices(const BinMapper* bin_mapper,
                                  int num_total_samples, int num_indices,
                                  const int* sample_indices,
                                  const double* sample_values) {
  std::vector<int> ret;
  if (bin_mapper->default_bin_ == bin_mapper->most_freq_bin_) return ret;
  int i = 0, j = 0;
  while (i < num_total_samples) {
    if (j < num_indices && sample_indices[j] < i) {
      ++j;
    } else if (j < num_indices && sample_indices[j] == i) {
      if (bin_mapper->ValueToBin(sample_values[j]) != bin_mapper->most_freq_bin_) {
        ret.push_back(i);
      }
      ++i;
    } else {
      ret.push_back(i++);
    }
  }
  return ret;
}

// ---- FindGroups (dataset.cpp:107-244) --------------------------------------
std::vector<std::vector<int>> FindGroups(
    const std::vector<BinMapper>& bin_mappers, const std::vector<int>& find_order,
    int** sample_indices, const int* num_per_col, int num_sample_col,
    int total_sample_cnt, int num_data, bool is_use_gpu, bool is_sparse,
    std::vector<int8_t>* multi_val_group) {
  const int max_search_group = 100;
  const int max_bin_per_group = 256;
  const int single_val_max_conflict_cnt = total_sample_cnt / 10000;
  multi_val_group->clear();

  LightGBM::Random rand(num_data);
  std::vector<std::vector<int>> features_in_group;
  std::vector<std::vector<bool>> conflict_marks;
  std::vector<int> group_used_row_cnt;
  std::vector<int> group_total_data_cnt;
  std::vector<int> group_num_bin;

  for (auto fidx : find_order) {
    bool is_filtered_feature = fidx >= num_sample_col;
    const int cur_non_zero_cnt = is_filtered_feature ? 0 : num_per_col[fidx];
    std::vector<int> available_groups;
    for (int gid = 0; gid < static_cast<int>(features_in_group.size()); ++gid) {
      auto cur_num_bin = group_num_bin[gid] + bin_mappers[fidx].num_bin_ +
                         (bin_mappers[fidx].most_freq_bin_ == 0 ? -1 : 0);
      if (group_total_data_cnt[gid] + cur_non_zero_cnt <=
          total_sample_cnt + single_val_max_conflict_cnt) {
        if (!is_use_gpu || cur_num_bin <= max_bin_per_group) {
          available_groups.push_back(gid);
        }
      }
    }
    std::vector<int> search_groups;
    if (!available_groups.empty()) {
      int last = static_cast<int>(available_groups.size()) - 1;
      auto indices = rand.Sample(last, std::min(last, max_search_group - 1));
      search_groups.push_back(available_groups.back());
      for (auto idx : indices) search_groups.push_back(available_groups[idx]);
    }
    int best_gid = -1;
    int best_conflict_cnt = -1;
    for (auto gid : search_groups) {
      const int rest_max_cnt = single_val_max_conflict_cnt -
                               group_total_data_cnt[gid] + group_used_row_cnt[gid];
      const int cnt = is_filtered_feature
                          ? 0
                          : GetConflictCount(conflict_marks[gid], sample_indices[fidx],
                                             num_per_col[fidx], rest_max_cnt);
      if (cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2) {
        best_gid = gid;
        best_conflict_cnt = cnt;
        break;
      }
    }
    if (best_gid >= 0) {
      features_in_group[best_gid].push_back(fidx);
      group_total_data_cnt[best_gid] += cur_non_zero_cnt;
      group_used_row_cnt[best_gid] += cur_non_zero_cnt - best_conflict_cnt;
      if (!is_filtered_feature) {
        MarkUsed(&conflict_marks[best_gid], sample_indices[fidx], num_per_col[fidx]);
      }
      group_num_bin[best_gid] +=
          bin_mappers[fidx].num_bin_ + (bin_mappers[fidx].default_bin_ == 0 ? -1 : 0);
    } else {
      features_in_group.emplace_back();
      features_in_group.back().push_back(fidx);
      conflict_marks.emplace_back(total_sample_cnt, false);
      if (!is_filtered_feature) {
        MarkUsed(&(conflict_marks.back()), sample_indices[fidx], num_per_col[fidx]);
      }
      group_total_data_cnt.emplace_back(cur_non_zero_cnt);
      group_used_row_cnt.emplace_back(cur_non_zero_cnt);
      group_num_bin.push_back(1 + bin_mappers[fidx].num_bin_ +
                              (bin_mappers[fidx].most_freq_bin_ == 0 ? -1 : 0));
    }
  }
  if (!is_sparse) {
    multi_val_group->resize(features_in_group.size(), false);
    return features_in_group;
  }
  std::vector<int> second_round_features;
  std::vector<std::vector<int>> features_in_group2;
  std::vector<std::vector<bool>> conflict_marks2;
  const double dense_threshold = 0.4;
  for (int gid = 0; gid < static_cast<int>(features_in_group.size()); ++gid) {
    const double dense_rate =
        static_cast<double>(group_used_row_cnt[gid]) / total_sample_cnt;
    if (dense_rate >= dense_threshold) {
      features_in_group2.push_back(std::move(features_in_group[gid]));
      conflict_marks2.push_back(std::move(conflict_marks[gid]));
    } else {
      for (auto fidx : features_in_group[gid]) second_round_features.push_back(fidx);
    }
  }
  features_in_group = features_in_group2;
  conflict_marks = conflict_marks2;
  multi_val_group->resize(features_in_group.size(), false);
  if (!second_round_features.empty()) {
    features_in_group.emplace_back();
    conflict_marks.emplace_back(total_sample_cnt, false);
    bool is_multi_val = is_use_gpu ? true : false;
    int conflict_cnt = 0;
    for (auto fidx : second_round_features) {
      features_in_group.back().push_back(fidx);
      if (!is_multi_val) {
        const int rest_max_cnt = single_val_max_conflict_cnt - conflict_cnt;
        const auto cnt = GetConflictCount(conflict_marks.back(), sample_indices[fidx],
                                          num_per_col[fidx], rest_max_cnt);
        conflict_cnt += cnt;
        if (cnt < 0 || conflict_cnt > single_val_max_conflict_cnt) {
          is_multi_val = true;
          continue;
        }
        MarkUsed(&(conflict_marks.back()), sample_indices[fidx], num_per_col[fidx]);
      }
    }
    multi_val_group->push_back(is_multi_val);
  }
  return features_in_group;
}

// ---- FastFeatureBundling (dataset.cpp:246-323) -----------------------------
std::vector<std::vector<int>> FastFeatureBundling(
    const std::vector<BinMapper>& bin_mappers, int** sample_indices,
    double** sample_values, const int* num_per_col, int num_sample_col,
    int total_sample_cnt, const std::vector<int>& used_features, int num_data,
    bool is_use_gpu, bool is_sparse, std::vector<int8_t>* multi_val_group) {
  std::vector<size_t> feature_non_zero_cnt;
  feature_non_zero_cnt.reserve(used_features.size());
  for (auto fidx : used_features) {
    if (fidx < num_sample_col) feature_non_zero_cnt.emplace_back(num_per_col[fidx]);
    else feature_non_zero_cnt.emplace_back(0);
  }
  std::vector<int> sorted_idx;
  sorted_idx.reserve(used_features.size());
  for (int i = 0; i < static_cast<int>(used_features.size()); ++i) sorted_idx.emplace_back(i);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&feature_non_zero_cnt](int a, int b) {
                     return feature_non_zero_cnt[a] > feature_non_zero_cnt[b];
                   });
  std::vector<int> feature_order_by_cnt;
  feature_order_by_cnt.reserve(sorted_idx.size());
  for (auto sidx : sorted_idx) feature_order_by_cnt.push_back(used_features[sidx]);

  std::vector<std::vector<int>> tmp_indices;
  std::vector<int> tmp_num_per_col(num_sample_col, 0);
  for (auto fidx : used_features) {
    if (fidx >= num_sample_col) continue;
    auto ret = FixSampleIndices(&bin_mappers[fidx], total_sample_cnt,
                                num_per_col[fidx], sample_indices[fidx],
                                sample_values[fidx]);
    if (!ret.empty()) {
      tmp_indices.push_back(ret);
      tmp_num_per_col[fidx] = static_cast<int>(ret.size());
      sample_indices[fidx] = tmp_indices.back().data();
    } else {
      tmp_num_per_col[fidx] = num_per_col[fidx];
    }
  }
  std::vector<int8_t> group_is_multi_val, group_is_multi_val2;
  auto features_in_group =
      FindGroups(bin_mappers, used_features, sample_indices, tmp_num_per_col.data(),
                 num_sample_col, total_sample_cnt, num_data, is_use_gpu, is_sparse,
                 &group_is_multi_val);
  auto group2 =
      FindGroups(bin_mappers, feature_order_by_cnt, sample_indices, tmp_num_per_col.data(),
                 num_sample_col, total_sample_cnt, num_data, is_use_gpu, is_sparse,
                 &group_is_multi_val2);
  if (features_in_group.size() > group2.size()) {
    features_in_group = group2;
    group_is_multi_val = group_is_multi_val2;
  }
  int num_group = static_cast<int>(features_in_group.size());
  LightGBM::Random tmp_rand(num_data);
  for (int i = 0; i < num_group - 1; ++i) {
    int j = tmp_rand.NextShort(i + 1, num_group);
    std::swap(features_in_group[i], features_in_group[j]);
    // std::swap for vector<bool> would be wrong — swap the parallel flag too.
    std::swap(group_is_multi_val[i], group_is_multi_val[j]);
  }
  *multi_val_group = group_is_multi_val;
  return features_in_group;
}

// FeatureGroup multi-feature ctor (feature_group.h:39-76) + PushData (253-267),
// single-val store path (CreateBinData(..., force_dense=true)). Multi-val groups
// in the EFB-win corpus below collapse to single-value bundled groups, so we
// dump the single-value bin_data per-row index; the multi-val flag is recorded.
struct FeatureGroupMV {
  int num_feature_;
  bool is_multi_val_;
  bool is_dense_multi_val_ = false;
  bool is_sparse_ = false;
  std::vector<BinMapper> bin_mappers_;
  std::vector<uint32_t> bin_offsets_;
  uint64_t num_total_bin_ = 0;
  IBin* bin_data_ = nullptr;

  FeatureGroupMV(int num_feature, int8_t is_multi_val,
                 std::vector<BinMapper> bin_mappers, int num_data, int group_id)
      : num_feature_(num_feature), is_multi_val_(is_multi_val > 0),
        bin_mappers_(std::move(bin_mappers)) {
    double sum_sparse_rate = 0.0;
    for (auto& m : bin_mappers_) sum_sparse_rate += m.sparse_rate_;
    sum_sparse_rate /= num_feature_;
    int offset = 1;
    is_dense_multi_val_ = false;
    if (sum_sparse_rate < kMultiValBinSparseThreshold && is_multi_val_) {
      offset = 0;
      is_dense_multi_val_ = true;
    }
    num_total_bin_ = offset;
    if (group_id == 0 && num_feature_ > 0 && is_dense_multi_val_ &&
        bin_mappers_[0].most_freq_bin_ > 0) {
      num_total_bin_ = 1;
    }
    bin_offsets_.emplace_back(static_cast<uint32_t>(num_total_bin_));
    for (int i = 0; i < num_feature_; ++i) {
      auto num_bin = bin_mappers_[i].num_bin_;
      if (bin_mappers_[i].most_freq_bin_ == 0) num_bin -= offset;
      num_total_bin_ += static_cast<uint64_t>(num_bin);
      bin_offsets_.emplace_back(static_cast<uint32_t>(num_total_bin_));
    }
    // CreateBinData(num_data, is_multi_val_, force_dense=true, force_sparse=false).
    // The EFB-win corpus produces single-value bundled groups; build the dense
    // single-value store so we can dump per-row bundled indices byte-for-byte.
    if (!is_multi_val_) {
      bin_data_ = CreateDenseBin(num_data, static_cast<int>(num_total_bin_));
    }
  }
  ~FeatureGroupMV() { delete bin_data_; }

  void PushData(int sub_feature_idx, int line_idx, double value) {
    uint32_t bin = bin_mappers_[sub_feature_idx].ValueToBin(value);
    if (bin == bin_mappers_[sub_feature_idx].most_freq_bin_) return;
    if (bin_mappers_[sub_feature_idx].most_freq_bin_ == 0) bin -= 1;
    if (!is_multi_val_) {
      bin += bin_offsets_[sub_feature_idx];
      bin_data_->Push(line_idx, bin);
    }
  }
  void FinishLoad() {
    if (!is_multi_val_ && bin_data_) bin_data_->FinishLoad();
  }
};

// ---- serialization helpers ------------------------------------------------
uint64_t F64Bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(b));
  return b;
}

uint32_t F32Bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}

// A small deterministic case generator reusing the reference Random LCG so the
// corpus is reproducible from MASTER_SEED.
struct CaseGen {
  LightGBM::Random rng;
  explicit CaseGen(int master_seed) : rng(master_seed) {}
  int NextInt(int lo, int hi) { return rng.NextInt(lo, hi); }
  double NextUnit() { return static_cast<double>(rng.NextFloat()); }
};

struct CaseSpec {
  std::string name;
  std::vector<double> column;  // full per-row column (f64)
  int max_bin;
  int min_data_in_bin;
  int min_split_data;
  bool pre_filter;
  bool use_missing;
  bool zero_as_missing;
  int data_random_seed;
  int sample_cnt;
};

void EmitCase(std::ofstream& out, const CaseSpec& cs) {
  // sample indices via the REAL reference Random (CreateSampleIndices).
  const int total_nrow = static_cast<int>(cs.column.size());
  const int k = total_nrow < cs.sample_cnt ? total_nrow : cs.sample_cnt;
  LightGBM::Random rand(cs.data_random_seed);
  std::vector<int> idx = rand.Sample(total_nrow, k);
  std::vector<double> sampled;
  sampled.reserve(idx.size());
  for (int i : idx) sampled.push_back(cs.column[i]);
  const size_t total_sample_cnt = sampled.size();

  BinMapper bm = FindBinNumeric(sampled, cs.max_bin, cs.min_data_in_bin, cs.min_split_data,
                                cs.pre_filter, cs.use_missing, cs.zero_as_missing,
                                total_sample_cnt, /*forced=*/{});

  out << "CASE name=" << cs.name << " max_bin=" << cs.max_bin
      << " min_data_in_bin=" << cs.min_data_in_bin << " min_split_data=" << cs.min_split_data
      << " pre_filter=" << (cs.pre_filter ? 1 : 0)
      << " use_missing=" << (cs.use_missing ? 1 : 0)
      << " zero_as_missing=" << (cs.zero_as_missing ? 1 : 0)
      << " num_rows=" << total_nrow << " seed=" << cs.data_random_seed
      << " sample_cnt=" << cs.sample_cnt << "\n";

  out << "VALUES ";
  for (size_t i = 0; i < cs.column.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(cs.column[i]);
  }
  out << "\n";

  out << "GOLDEN num_bin=" << bm.num_bin_ << " bin_type=" << static_cast<int>(bm.bin_type_)
      << " missing_type=" << static_cast<int>(bm.missing_type_)
      << " default_bin=" << bm.default_bin_ << " most_freq_bin=" << bm.most_freq_bin_
      << " is_trivial=" << (bm.is_trivial_ ? 1 : 0) << " upper=";
  for (size_t i = 0; i < bm.bin_upper_bound_.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(bm.bin_upper_bound_[i]);
  }
  out << "\n";

  out << "ASSIGN ";
  for (size_t i = 0; i < cs.column.size(); ++i) {
    if (i) out << ";";
    out << bm.ValueToBin(cs.column[i]);
  }
  out << "\n";
}

// Build the four-source numeric corpus deterministically from the master seed.
std::vector<CaseSpec> BuildCorpus(int master_seed) {
  std::vector<CaseSpec> cases;
  CaseGen gen(master_seed);

  // (1) synthetic randomized distributions sweeping the config knobs.
  const int max_bins[] = {2, 16, 64, 255};
  const int min_data_in_bins[] = {1, 3, 20};
  const int sample_cnts[] = {64, 256, 100000};
  int synth_id = 0;
  for (int mb : max_bins) {
    for (int md : min_data_in_bins) {
      for (int sc : sample_cnts) {
        int n = gen.NextInt(50, 1200);
        std::vector<double> col;
        col.reserve(n);
        // mixture: uniform with some clustering and a sign spread.
        int mode = gen.NextInt(0, 3);
        for (int i = 0; i < n; ++i) {
          double u = gen.NextUnit();
          double v;
          if (mode == 0) v = (u - 0.5) * 200.0;          // signed spread
          else if (mode == 1) v = u * 1000.0;            // all positive
          else v = std::floor(u * 8.0);                  // few clusters
          col.push_back(v);
        }
        CaseSpec cs;
        cs.name = "synth" + std::to_string(synth_id++);
        cs.column = std::move(col);
        cs.max_bin = mb;
        cs.min_data_in_bin = md;
        cs.min_split_data = 1;
        cs.pre_filter = false;
        cs.use_missing = true;
        cs.zero_as_missing = false;
        cs.data_random_seed = gen.NextInt(1, 2000000000);
        cs.sample_cnt = sc;
        cases.push_back(std::move(cs));
      }
    }
  }

  const double qnan = std::numeric_limits<double>::quiet_NaN();
  auto edge = [&](const std::string& name, std::vector<double> col, int max_bin,
                  bool use_missing, bool zero_as_missing, bool pre_filter, int min_split) {
    CaseSpec cs;
    cs.name = name;
    cs.column = std::move(col);
    cs.max_bin = max_bin;
    cs.min_data_in_bin = 1;
    cs.min_split_data = min_split;
    cs.pre_filter = pre_filter;
    cs.use_missing = use_missing;
    cs.zero_as_missing = zero_as_missing;
    cs.data_random_seed = gen.NextInt(1, 2000000000);
    cs.sample_cnt = 100000;  // sample all rows
    cases.push_back(std::move(cs));
  };

  // (2) curated numeric edge-case battery.
  edge("nan_missing", {1.0, 2.0, qnan, 3.0, qnan, 4.0, 5.0, 2.0, 1.0, 3.0}, 16, true, false,
       false, 1);
  edge("zero_signed", {0.0, -0.0, 1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 3.0, -3.0}, 16, true, false,
       false, 1);
  edge("on_boundary", {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0}, 4, true, false,
       false, 1);
  edge("all_missing", {qnan, qnan, qnan, qnan, qnan}, 16, true, false, false, 1);
  edge("single_value", {7.0, 7.0, 7.0, 7.0, 7.0, 7.0}, 16, true, false, false, 1);
  edge("all_zero", {0.0, 0.0, 0.0, 0.0, 0.0}, 16, true, false, false, 1);
  edge("zero_as_missing", {0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0}, 16, true, true,
       false, 1);
  edge("pre_filter_trivial",
       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0}, 16, true, false, true, 5);
  edge("dense_many",
       [] {
         std::vector<double> v;
         for (int i = 0; i < 500; ++i) v.push_back(static_cast<double>(i));
         return v;
       }(),
       64, true, false, false, 1);

  return cases;
}

// ===========================================================================
// Storage-layout corpus (golden layer-2 STORAGE subset, DAT-02). For a set of
// columns spanning each width path (num_bin <=16 4-bit, <=256, <=65536, >65536)
// plus a sparse/zero-heavy column, build the C++ FindBin mapper, then the
// single-value FeatureGroup, push every row, FinishLoad, and dump the raw
// DenseBin data_ / SparseBin deltas_+vals_+fast_index / FeatureGroup
// bin_offsets_+num_total_bin_ — the exact memory Phase 4 reads.
// ===========================================================================

struct StorageCaseSpec {
  std::string name;
  std::vector<double> column;  // full per-row column (f64)
  int max_bin;                 // drives the width path
  int min_data_in_bin;
  bool use_missing;
  bool zero_as_missing;
};

void EmitStorageCase(std::ofstream& out, const StorageCaseSpec& cs) {
  const int num_rows = static_cast<int>(cs.column.size());
  // Build the mapper over the FULL column (sample_cnt = all rows) so the storage
  // layout is deterministic and self-contained for the replay test.
  BinMapper bm = FindBinNumeric(cs.column, cs.max_bin, cs.min_data_in_bin,
                                /*min_split_data=*/1, /*pre_filter=*/false,
                                cs.use_missing, cs.zero_as_missing,
                                static_cast<size_t>(num_rows), /*forced=*/{});

  FeatureGroupSV fg(bm, num_rows);
  for (int r = 0; r < num_rows; ++r) fg.PushData(r, cs.column[r]);
  fg.FinishLoad();

  out << "SCASE name=" << cs.name << " max_bin=" << cs.max_bin
      << " min_data_in_bin=" << cs.min_data_in_bin
      << " use_missing=" << (cs.use_missing ? 1 : 0)
      << " zero_as_missing=" << (cs.zero_as_missing ? 1 : 0)
      << " num_rows=" << num_rows << "\n";

  // full input column (raw f64 bits) so the Rust side rebuilds an identical mapper.
  out << "VALUES ";
  for (int i = 0; i < num_rows; ++i) {
    if (i) out << ";";
    out << F64Bits(cs.column[i]);
  }
  out << "\n";

  // mapper meta the Rust side cross-checks before comparing storage.
  out << "MAPPER num_bin=" << bm.num_bin_
      << " missing_type=" << static_cast<int>(bm.missing_type_)
      << " most_freq_bin=" << bm.most_freq_bin_
      << " is_trivial=" << (bm.is_trivial_ ? 1 : 0) << "\n";

  // FeatureGroup offset layout.
  out << "FG num_total_bin=" << fg.num_total_bin_ << " bin_offsets=";
  for (size_t i = 0; i < fg.bin_offsets_.size(); ++i) {
    if (i) out << ";";
    out << fg.bin_offsets_[i];
  }
  out << " is_sparse=" << (fg.is_sparse_ ? 1 : 0) << "\n";

  // Storage bytes. For a dense store: STORE_DENSE <byte;byte;...>.
  // For a sparse store: STORE_SPARSE deltas=<byte;...> vals=<byte;...>.
  const IBin* bd = fg.bin_data_;
  if (bd->IsSparse()) {
    out << "STORE_SPARSE deltas=";
    const auto* d = bd->Deltas();
    for (size_t i = 0; i < d->size(); ++i) {
      if (i) out << ";";
      out << static_cast<int>((*d)[i]);
    }
    out << " vals=";
    auto vb = bd->ValsBytes();
    for (size_t i = 0; i < vb.size(); ++i) {
      if (i) out << ";";
      out << static_cast<int>(vb[i]);
    }
    out << "\n";
  } else {
    out << "STORE_DENSE bytes=";
    auto rb = bd->RawBytes();
    for (size_t i = 0; i < rb.size(); ++i) {
      if (i) out << ";";
      out << static_cast<int>(rb[i]);
    }
    out << "\n";
  }
}

std::vector<StorageCaseSpec> BuildStorageCorpus(int master_seed) {
  std::vector<StorageCaseSpec> cases;
  CaseGen gen(master_seed);

  // Helper: build a randomized positive-spread column of n rows.
  auto rand_col = [&](int n, double scale) {
    std::vector<double> col;
    col.reserve(n);
    for (int i = 0; i < n; ++i) col.push_back(gen.NextUnit() * scale);
    return col;
  };

  // (1) 4-bit path: num_bin <= 16. Few distinct values, max_bin small.
  {
    StorageCaseSpec cs;
    cs.name = "dense_4bit";
    // ~8 distinct clusters -> num_bin well under 16 -> DenseBin<u8,true>.
    std::vector<double> col;
    for (int i = 0; i < 60; ++i) col.push_back(std::floor(gen.NextUnit() * 8.0) + 1.0);
    cs.column = std::move(col);
    cs.max_bin = 16; cs.min_data_in_bin = 1; cs.use_missing = true; cs.zero_as_missing = false;
    cases.push_back(std::move(cs));
  }
  // (2) u8 path: 16 < num_bin <= 256.
  {
    StorageCaseSpec cs;
    cs.name = "dense_u8";
    cs.column = rand_col(400, 500.0);
    cs.max_bin = 200; cs.min_data_in_bin = 1; cs.use_missing = true; cs.zero_as_missing = false;
    cases.push_back(std::move(cs));
  }
  // (3) u16 path: 256 < num_bin <= 65536. Needs > 256 distinct -> max_bin large.
  {
    StorageCaseSpec cs;
    cs.name = "dense_u16";
    std::vector<double> col;
    for (int i = 0; i < 1000; ++i) col.push_back(static_cast<double>(i) + gen.NextUnit());
    cs.column = std::move(col);
    cs.max_bin = 1000; cs.min_data_in_bin = 1; cs.use_missing = true; cs.zero_as_missing = false;
    cases.push_back(std::move(cs));
  }
  // (4) u32 path: num_bin > 65536. Requires a huge max_bin + many distinct.
  {
    StorageCaseSpec cs;
    cs.name = "dense_u32";
    std::vector<double> col;
    for (int i = 0; i < 70000; ++i) col.push_back(static_cast<double>(i));
    cs.column = std::move(col);
    cs.max_bin = 70000; cs.min_data_in_bin = 1; cs.use_missing = true; cs.zero_as_missing = false;
    cases.push_back(std::move(cs));
  }
  // (5) sparse / zero-heavy column: high zero rate -> sparse_rate >= 0.7 -> SparseBin.
  {
    StorageCaseSpec cs;
    cs.name = "sparse_zero_heavy";
    std::vector<double> col;
    for (int i = 0; i < 400; ++i) {
      // ~90% zeros, occasional nonzero -> most_freq_bin sparse, sparse store.
      col.push_back((gen.NextUnit() < 0.1) ? (gen.NextUnit() * 50.0 + 1.0) : 0.0);
    }
    cs.column = std::move(col);
    cs.max_bin = 64; cs.min_data_in_bin = 1; cs.use_missing = true; cs.zero_as_missing = false;
    cases.push_back(std::move(cs));
  }
  // (6) 4-bit with an odd row count (exercises the (n+1)/2 last-nibble byte).
  {
    StorageCaseSpec cs;
    cs.name = "dense_4bit_odd";
    std::vector<double> col;
    for (int i = 0; i < 41; ++i) col.push_back(std::floor(gen.NextUnit() * 6.0) + 1.0);
    cs.column = std::move(col);
    cs.max_bin = 16; cs.min_data_in_bin = 1; cs.use_missing = true; cs.zero_as_missing = false;
    cases.push_back(std::move(cs));
  }

  return cases;
}

// ===========================================================================
// Categorical corpus (golden layers 1 + 3 + per-row index, DAT-04). Builds the
// categorical BinMapper over the FULL column (sample_cnt = all rows so the map
// is self-contained for replay), then dumps layer 1 (num_bin/missing_type/
// default_bin/most_freq_bin/is_trivial), layer 3 (categorical_2_bin_ as sorted
// key:bin pairs + bin_2_categorical_), and the per-row ValueToBin vector.
// ===========================================================================

struct CatCaseSpec {
  std::string name;
  std::vector<double> column;  // integer-valued categories as f64
  int max_bin;
  int min_data_in_bin;
  bool use_missing;
  bool zero_as_missing;
  bool pre_filter;
  int min_split_data;
};

void EmitCatCase(std::ofstream& out, const CatCaseSpec& cs) {
  const int num_rows = static_cast<int>(cs.column.size());
  BinMapper bm = FindBinCategorical(cs.column, cs.max_bin, cs.min_data_in_bin,
                                    cs.min_split_data, cs.pre_filter, cs.use_missing,
                                    cs.zero_as_missing, static_cast<size_t>(num_rows));

  out << "CCASE name=" << cs.name << " max_bin=" << cs.max_bin
      << " min_data_in_bin=" << cs.min_data_in_bin
      << " use_missing=" << (cs.use_missing ? 1 : 0)
      << " zero_as_missing=" << (cs.zero_as_missing ? 1 : 0)
      << " pre_filter=" << (cs.pre_filter ? 1 : 0)
      << " min_split_data=" << cs.min_split_data << " num_rows=" << num_rows << "\n";

  out << "VALUES ";
  for (int i = 0; i < num_rows; ++i) {
    if (i) out << ";";
    out << F64Bits(cs.column[i]);
  }
  out << "\n";

  out << "MAPPER num_bin=" << bm.num_bin_ << " bin_type=" << static_cast<int>(bm.bin_type_)
      << " missing_type=" << static_cast<int>(bm.missing_type_)
      << " default_bin=" << bm.default_bin_ << " most_freq_bin=" << bm.most_freq_bin_
      << " is_trivial=" << (bm.is_trivial_ ? 1 : 0) << "\n";

  // layer 3: categorical_2_bin_ as sorted key:bin pairs (sorted by key for a
  // stable, diff-friendly, order-independent dump).
  std::vector<std::pair<int, uint32_t>> c2b(bm.categorical_2_bin_.begin(),
                                            bm.categorical_2_bin_.end());
  std::sort(c2b.begin(), c2b.end(),
            [](const std::pair<int, uint32_t>& a, const std::pair<int, uint32_t>& b) {
              return a.first < b.first;
            });
  out << "C2B ";
  for (size_t i = 0; i < c2b.size(); ++i) {
    if (i) out << ";";
    out << c2b[i].first << ":" << c2b[i].second;
  }
  out << "\n";

  out << "B2C ";
  for (size_t i = 0; i < bm.bin_2_categorical_.size(); ++i) {
    if (i) out << ";";
    out << bm.bin_2_categorical_[i];
  }
  out << "\n";

  out << "ASSIGN ";
  for (int i = 0; i < num_rows; ++i) {
    if (i) out << ";";
    out << bm.ValueToBin(cs.column[i]);
  }
  out << "\n";
}

std::vector<CatCaseSpec> BuildCatCorpus(int master_seed) {
  std::vector<CatCaseSpec> cases;
  CaseGen gen(master_seed);

  auto cat = [&](const std::string& name, std::vector<double> col, int max_bin,
                 int min_data_in_bin, bool use_missing, bool zero_as_missing,
                 bool pre_filter, int min_split) {
    CatCaseSpec cs;
    cs.name = name;
    cs.column = std::move(col);
    cs.max_bin = max_bin;
    cs.min_data_in_bin = min_data_in_bin;
    cs.use_missing = use_missing;
    cs.zero_as_missing = zero_as_missing;
    cs.pre_filter = pre_filter;
    cs.min_split_data = min_split;
    cases.push_back(std::move(cs));
  };

  // (1) basic multi-categorical with varied counts (desc-count fold + ties).
  {
    std::vector<double> col;
    for (int i = 0; i < 8; ++i) col.push_back(5.0);
    for (int i = 0; i < 5; ++i) col.push_back(2.0);
    for (int i = 0; i < 5; ++i) col.push_back(9.0);  // tie with 2 on count
    for (int i = 0; i < 1; ++i) col.push_back(0.0);
    cat("cat_basic", col, 256, 1, true, false, false, 1);
  }
  // (2) rare level triggers the min_data_in_bin fold-break.
  {
    std::vector<double> col;
    for (int i = 0; i < 10; ++i) col.push_back(10.0);
    for (int i = 0; i < 10; ++i) col.push_back(20.0);
    for (int i = 0; i < 10; ++i) col.push_back(30.0);
    col.push_back(40.0);  // rare -> fold-break with min_data_in_bin=5
    cat("cat_rare_foldbreak", col, 256, 5, true, false, false, 1);
  }
  // (3) negative / out-of-range categories -> NaN accumulation.
  {
    std::vector<double> col = {-1.0, -1.0, 3.0, 3.0, 3.0, 7.0, 7.0, -5.0, 0.0, 0.0};
    cat("cat_negative", col, 256, 1, true, false, false, 1);
  }
  // (4) high-cardinality: many distinct categories, max_bin caps the fold.
  {
    std::vector<double> col;
    for (int c = 0; c < 50; ++c) {
      int reps = gen.NextInt(1, 6);
      for (int r = 0; r < reps; ++r) col.push_back(static_cast<double>(c));
    }
    cat("cat_high_cardinality", col, 16, 1, true, false, false, 1);
  }
  // (5) the 0.99f cut binds: one dominant category + a long rare tail.
  {
    std::vector<double> col;
    for (int i = 0; i < 99; ++i) col.push_back(1.0);
    col.push_back(2.0);
    cat("cat_099_cut", col, 8, 1, true, false, false, 1);
  }
  // (6) all categories consumed, no NaN -> missing_type None.
  {
    std::vector<double> col = {0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 3.0};
    cat("cat_no_missing", col, 256, 1, true, false, false, 1);
  }

  return cases;
}

// ===========================================================================
// Missing-edge corpus (golden layer 1 + per-row index, DAT-03). Sweeps NaN per
// MissingType (None/Zero/NaN via use_missing/zero_as_missing), +0.0/-0.0 signed
// zeros, all-missing, single-value. Numeric mapper (missing routing lives on the
// numeric path). Builds over the FULL column (sample_cnt = all rows).
// ===========================================================================

struct MissCaseSpec {
  std::string name;
  std::vector<double> column;
  int max_bin;
  bool use_missing;
  bool zero_as_missing;
};

void EmitMissCase(std::ofstream& out, const MissCaseSpec& cs) {
  const int num_rows = static_cast<int>(cs.column.size());
  BinMapper bm = FindBinNumeric(cs.column, cs.max_bin, /*min_data_in_bin=*/1,
                                /*min_split_data=*/1, /*pre_filter=*/false,
                                cs.use_missing, cs.zero_as_missing,
                                static_cast<size_t>(num_rows), /*forced=*/{});

  out << "MCASE name=" << cs.name << " max_bin=" << cs.max_bin
      << " use_missing=" << (cs.use_missing ? 1 : 0)
      << " zero_as_missing=" << (cs.zero_as_missing ? 1 : 0)
      << " num_rows=" << num_rows << "\n";

  out << "VALUES ";
  for (int i = 0; i < num_rows; ++i) {
    if (i) out << ";";
    out << F64Bits(cs.column[i]);
  }
  out << "\n";

  out << "MAPPER num_bin=" << bm.num_bin_ << " bin_type=" << static_cast<int>(bm.bin_type_)
      << " missing_type=" << static_cast<int>(bm.missing_type_)
      << " default_bin=" << bm.default_bin_ << " most_freq_bin=" << bm.most_freq_bin_
      << " is_trivial=" << (bm.is_trivial_ ? 1 : 0) << "\n";

  out << "ASSIGN ";
  for (int i = 0; i < num_rows; ++i) {
    if (i) out << ";";
    out << bm.ValueToBin(cs.column[i]);
  }
  out << "\n";
}

std::vector<MissCaseSpec> BuildMissCorpus(int /*master_seed*/) {
  std::vector<MissCaseSpec> cases;
  const double qnan = std::numeric_limits<double>::quiet_NaN();
  auto miss = [&](const std::string& name, std::vector<double> col, int max_bin,
                  bool use_missing, bool zero_as_missing) {
    MissCaseSpec cs;
    cs.name = name;
    cs.column = std::move(col);
    cs.max_bin = max_bin;
    cs.use_missing = use_missing;
    cs.zero_as_missing = zero_as_missing;
    cases.push_back(std::move(cs));
  };

  // A base column with NaNs swept across the three MissingType resolutions.
  std::vector<double> with_nan = {1.0, 2.0, qnan, 3.0, qnan, 4.0, 5.0, 2.0, 1.0, 3.0};
  miss("nan_missing_none_nouse", with_nan, 16, /*use_missing=*/false, false);  // -> None
  miss("nan_missing_nan", with_nan, 16, /*use_missing=*/true, false);          // -> NaN top bin
  miss("nan_zero_as_missing", with_nan, 16, /*use_missing=*/true, true);       // -> Zero

  // signed zeros: +0.0 and -0.0 must route identically.
  std::vector<double> signed_zero = {0.0, -0.0, 1.0, -1.0, 0.0, 2.0, -2.0, -0.0, 3.0, -3.0};
  miss("signed_zero_use", signed_zero, 16, true, false);
  miss("signed_zero_zero_as_missing", signed_zero, 16, true, true);

  // all-missing column -> single-bin (NaN) mapper, never a panic.
  miss("all_missing", {qnan, qnan, qnan, qnan, qnan}, 16, true, false);

  // single distinct value (no missing).
  miss("single_value", {7.0, 7.0, 7.0, 7.0, 7.0}, 16, true, false);

  // zero_as_missing with explicit zeros in the data.
  miss("zeros_as_missing", {0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 6.0}, 16, true, true);

  return cases;
}

// ===========================================================================
// Metadata golden (DAT-06): a small MASTER_SEED-derived labels/weights/
// init_score/query-boundary set + the C++ Metadata::CalculateQueryWeights
// output (mean weight per group, computed in label_t==float arithmetic).
// Transcribed VERBATIM from src/io/metadata.cpp:742-756 — accumulate in float,
// divide by the (int) group size (converts to float). Bit-exact via F32Bits.
// ===========================================================================

struct MetaCaseSpec {
  std::string name;
  std::vector<float> label;
  std::vector<float> weights;            // empty -> no weights
  std::vector<double> init_score;        // empty -> no init scores
  std::vector<int32_t> query_boundaries; // empty -> no groups (prefix-sum form)
};

// C++ Metadata::CalculateQueryWeights (metadata.cpp:742-756), verbatim.
std::vector<float> CalculateQueryWeights(const std::vector<float>& weights,
                                         const std::vector<int32_t>& query_boundaries) {
  std::vector<float> query_weights;
  if (weights.size() == 0 || query_boundaries.size() == 0) {
    return query_weights;  // empty (C++ early-return)
  }
  const int32_t num_queries = static_cast<int32_t>(query_boundaries.size()) - 1;
  query_weights = std::vector<float>(num_queries);
  for (int32_t i = 0; i < num_queries; ++i) {
    query_weights[i] = 0.0f;
    for (int32_t j = query_boundaries[i]; j < query_boundaries[i + 1]; ++j) {
      query_weights[i] += weights[j];
    }
    query_weights[i] /= (query_boundaries[i + 1] - query_boundaries[i]);
  }
  return query_weights;
}

void EmitMetaCase(std::ofstream& out, const MetaCaseSpec& cs) {
  std::vector<float> qw = CalculateQueryWeights(cs.weights, cs.query_boundaries);

  out << "MCASE name=" << cs.name << " num_rows=" << cs.label.size() << "\n";

  out << "LABEL ";
  for (size_t i = 0; i < cs.label.size(); ++i) {
    if (i) out << ";";
    out << F32Bits(cs.label[i]);
  }
  out << "\n";

  out << "WEIGHTS ";
  for (size_t i = 0; i < cs.weights.size(); ++i) {
    if (i) out << ";";
    out << F32Bits(cs.weights[i]);
  }
  out << "\n";

  out << "INIT_SCORE ";
  for (size_t i = 0; i < cs.init_score.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(cs.init_score[i]);
  }
  out << "\n";

  out << "QUERY_BOUNDARIES ";
  for (size_t i = 0; i < cs.query_boundaries.size(); ++i) {
    if (i) out << ";";
    out << cs.query_boundaries[i];
  }
  out << "\n";

  out << "QUERY_WEIGHTS ";
  for (size_t i = 0; i < qw.size(); ++i) {
    if (i) out << ";";
    out << F32Bits(qw[i]);
  }
  out << "\n";
}

std::vector<MetaCaseSpec> BuildMetaCorpus(int master_seed) {
  std::vector<MetaCaseSpec> cases;
  CaseGen gen(master_seed);

  // (1) randomized ranking case: groups of varying size with random weights.
  {
    MetaCaseSpec cs;
    cs.name = "ranking_random";
    const int num_groups = 5;
    int32_t boundary = 0;
    cs.query_boundaries.push_back(0);
    for (int g = 0; g < num_groups; ++g) {
      const int group_size = gen.NextInt(1, 8);
      for (int i = 0; i < group_size; ++i) {
        cs.label.push_back(static_cast<float>(gen.NextInt(0, 4)));
        cs.weights.push_back(static_cast<float>(gen.NextUnit() * 4.0 + 0.1));
        cs.init_score.push_back(gen.NextUnit() * 2.0 - 1.0);
      }
      boundary += group_size;
      cs.query_boundaries.push_back(boundary);
    }
    cases.push_back(std::move(cs));
  }

  // (2) curated even/odd groups: integer weights so the mean is exact in f32.
  {
    MetaCaseSpec cs;
    cs.name = "groups_int_weights";
    cs.label = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    cs.weights = {1.0f, 3.0f, 2.0f, 4.0f, 6.0f};
    cs.init_score = {0.1, 0.2, 0.3, 0.4, 0.5};
    cs.query_boundaries = {0, 2, 5};  // means: (1+3)/2=2, (2+4+6)/3=4
    cases.push_back(std::move(cs));
  }

  // (3) labels only, no weights, no groups (the common non-ranking case ->
  //     empty query_weights).
  {
    MetaCaseSpec cs;
    cs.name = "labels_only";
    cs.label = {0.0f, 1.0f, 0.0f, 1.0f};
    cases.push_back(std::move(cs));
  }

  // (4) weights but no groups -> still empty query_weights.
  {
    MetaCaseSpec cs;
    cs.name = "weights_no_groups";
    cs.label = {1.0f, 2.0f, 3.0f};
    cs.weights = {0.5f, 1.5f, 2.5f};
    cases.push_back(std::move(cs));
  }

  // (5) fractional weights that exercise f32 rounding in the mean.
  {
    MetaCaseSpec cs;
    cs.name = "groups_frac_weights";
    cs.label = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    cs.weights = {0.1f, 0.2f, 0.3f, 0.7f, 0.9f, 1.1f};
    cs.query_boundaries = {0, 3, 6};  // means in f32: (0.1+0.2+0.3)/3, (0.7+0.9+1.1)/3
    cases.push_back(std::move(cs));
  }

  return cases;
}

// ===========================================================================
// Example-dataset parity (DAT-07): load a copied LightGBM example dataset (TSV,
// column 0 = label, columns 1.. = numeric features), bin each feature with the
// pinned deterministic config, and dump layer 1 (bin_upper_bound_ + internals
// per feature) + layer 2 (per-row ValueToBin per feature). End-to-end proof on
// real data. The harness reads the COPIED fixture path passed on argv — NEVER
// the untracked LightGBM/ tree.
// ===========================================================================

// Fixed config for example binning (matches the C-API defaults used in the Rust
// ingest test): max_bin=255, min_data_in_bin=3, min_data_in_leaf=20,
// no pre-filter, use_missing, sample all rows.
constexpr int kExampleMaxBin = 255;
constexpr int kExampleMinDataInBin = 3;
constexpr int kExampleMinDataInLeaf = 20;
// Cap rows for a manageable golden; the Rust test ingests the SAME prefix.
constexpr int kExampleMaxRows = 500;

// Parse a whitespace/tab-delimited numeric matrix (column 0 = label, dropped).
// Returns features as a column-major vector<vector<double>> plus num_rows.
struct ExampleData {
  int num_rows = 0;
  int num_features = 0;
  std::vector<std::vector<double>> columns;  // columns[f][row]
};

ExampleData LoadExample(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "error: cannot open example dataset: " << path << "\n";
    std::exit(1);
  }
  ExampleData data;
  std::string line;
  while (static_cast<int>(data.num_rows) < kExampleMaxRows && std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::vector<double> row_vals;
    double v;
    while (ss >> v) row_vals.push_back(v);
    if (row_vals.empty()) continue;
    // column 0 is the label -> features are row_vals[1..].
    const int nfeat = static_cast<int>(row_vals.size()) - 1;
    if (data.num_features == 0) {
      data.num_features = nfeat;
      data.columns.resize(nfeat);
    }
    for (int f = 0; f < nfeat; ++f) {
      data.columns[f].push_back(row_vals[f + 1]);
    }
    data.num_rows++;
  }
  return data;
}

void EmitExampleDataset(std::ofstream& out, const std::string& name, const std::string& path,
                        int data_random_seed) {
  ExampleData data = LoadExample(path);

  out << "DATASET name=" << name << " num_rows=" << data.num_rows
      << " num_features=" << data.num_features << " max_bin=" << kExampleMaxBin
      << " min_data_in_bin=" << kExampleMinDataInBin << " seed=" << data_random_seed << "\n";

  for (int f = 0; f < data.num_features; ++f) {
    const std::vector<double>& col = data.columns[f];
    // sample all rows (sample_cnt > num_rows -> k = num_rows), matching the Rust
    // ingest path with bin_construct_sample_cnt = 100000.
    const int total_nrow = static_cast<int>(col.size());
    const int sample_cnt = 100000;
    const int k = total_nrow < sample_cnt ? total_nrow : sample_cnt;
    LightGBM::Random rand(data_random_seed);
    std::vector<int> idx = rand.Sample(total_nrow, k);
    std::vector<double> sampled;
    sampled.reserve(idx.size());
    for (int i : idx) sampled.push_back(col[i]);

    BinMapper bm = FindBinNumeric(sampled, kExampleMaxBin, kExampleMinDataInBin,
                                  kExampleMinDataInLeaf, /*pre_filter=*/false,
                                  /*use_missing=*/true, /*zero_as_missing=*/false,
                                  sampled.size(), /*forced=*/{});

    out << "FEATURE f=" << f << " num_bin=" << bm.num_bin_
        << " missing_type=" << static_cast<int>(bm.missing_type_)
        << " default_bin=" << bm.default_bin_ << " most_freq_bin=" << bm.most_freq_bin_
        << " is_trivial=" << (bm.is_trivial_ ? 1 : 0) << " upper=";
    for (size_t i = 0; i < bm.bin_upper_bound_.size(); ++i) {
      if (i) out << ";";
      out << F64Bits(bm.bin_upper_bound_[i]);
    }
    out << "\n";

    out << "ASSIGN f=" << f << " ";
    for (int r = 0; r < total_nrow; ++r) {
      if (r) out << ";";
      out << bm.ValueToBin(col[r]);
    }
    out << "\n";
  }
}

// ===========================================================================
// EFB grouping corpus (golden layer 3, DAT-05; D-06 #4 mutually-exclusive
// sparse feature sets + a control where bundling does nothing). Each case is a
// full row-major matrix of integer-binned numeric features. The harness:
//   1. builds a BinMapper per feature over the FULL column (self-contained);
//   2. derives the per-column non-zero SAMPLE indices/values (sample all rows,
//      so total_sample_cnt == num_rows; deterministic, no extra RNG);
//   3. runs FastFeatureBundling(enable_bundle=true) -> feature->group + flags;
//   4. builds a FeatureGroupMV per group, pushes every row, FinishLoad;
//   5. dumps feature2group/feature2subfeature, per-group bin_offsets_ +
//      num_total_bin_ + multi_val flag, and the per-row bundled bin index per
//      single-value group.
// ===========================================================================

struct EfbCaseSpec {
  std::string name;
  int num_rows;
  int num_features;
  std::vector<std::vector<double>> columns;  // columns[f][row]
  int max_bin;
  bool is_enable_sparse;  // io_config.is_enable_sparse (drives the 2nd FindGroups pass)
};

void EmitEfbCase(std::ofstream& out, const EfbCaseSpec& cs) {
  const int num_rows = cs.num_rows;
  const int num_features = cs.num_features;

  // 1. mapper per feature over the full column.
  std::vector<BinMapper> mappers;
  mappers.reserve(num_features);
  for (int f = 0; f < num_features; ++f) {
    mappers.push_back(FindBinNumeric(cs.columns[f], cs.max_bin, /*min_data_in_bin=*/1,
                                     /*min_split_data=*/1, /*pre_filter=*/false,
                                     /*use_missing=*/true, /*zero_as_missing=*/false,
                                     static_cast<size_t>(num_rows), /*forced=*/{}));
  }

  // 2. per-column non-zero SAMPLE indices/values (sample == all rows).
  std::vector<std::vector<int>> sample_indices(num_features);
  std::vector<std::vector<double>> sample_values(num_features);
  std::vector<int> num_per_col(num_features, 0);
  for (int f = 0; f < num_features; ++f) {
    for (int r = 0; r < num_rows; ++r) {
      // a "non-zero" sample entry is any row whose value is not exactly 0.0
      // (mirrors the C-API sampler that records non-default entries).
      if (cs.columns[f][r] != 0.0) {
        sample_indices[f].push_back(r);
        sample_values[f].push_back(cs.columns[f][r]);
      }
    }
    num_per_col[f] = static_cast<int>(sample_indices[f].size());
  }

  // used_features = non-trivial features, in order (dataset.cpp:337-343).
  std::vector<int> used_features;
  for (int f = 0; f < num_features; ++f) {
    if (!mappers[f].is_trivial_) used_features.push_back(f);
  }

  // raw int** / double** views for the verbatim EFB signature.
  std::vector<int*> idx_ptrs(num_features, nullptr);
  std::vector<double*> val_ptrs(num_features, nullptr);
  for (int f = 0; f < num_features; ++f) {
    idx_ptrs[f] = sample_indices[f].empty() ? nullptr : sample_indices[f].data();
    val_ptrs[f] = sample_values[f].empty() ? nullptr : sample_values[f].data();
  }

  // 3. grouping (FastFeatureBundling when enabled + non-empty, else 1-per-grp).
  std::vector<int8_t> group_is_multi_val(used_features.size(), 0);
  std::vector<std::vector<int>> features_in_group;
  if (!used_features.empty()) {
    features_in_group = FastFeatureBundling(
        mappers, idx_ptrs.data(), val_ptrs.data(), num_per_col.data(), num_features,
        /*total_sample_cnt=*/num_rows, used_features, /*num_data=*/num_rows,
        /*is_use_gpu=*/false, cs.is_enable_sparse, &group_is_multi_val);
  }
  const int num_groups = static_cast<int>(features_in_group.size());

  // 4. build per-group FeatureGroupMV + feature->group maps (dataset.cpp:375-411).
  int total_feat = 0;
  for (auto& fs : features_in_group) total_feat += static_cast<int>(fs.size());
  std::vector<int> feature2group(total_feat, -1);
  std::vector<int> feature2subfeature(total_feat, -1);
  std::vector<int> real_feature_idx(total_feat, -1);

  std::vector<FeatureGroupMV*> groups;
  int cur_fidx = 0;
  for (int i = 0; i < num_groups; ++i) {
    const auto& cur_features = features_in_group[i];
    const int cur_cnt = static_cast<int>(cur_features.size());
    std::vector<BinMapper> cur_mappers;
    cur_mappers.reserve(cur_cnt);
    for (int j = 0; j < cur_cnt; ++j) {
      int real_fidx = cur_features[j];
      real_feature_idx[cur_fidx] = real_fidx;
      feature2group[cur_fidx] = i;
      feature2subfeature[cur_fidx] = j;
      cur_mappers.push_back(mappers[real_fidx]);
      ++cur_fidx;
    }
    groups.push_back(new FeatureGroupMV(cur_cnt, group_is_multi_val[i],
                                        std::move(cur_mappers), num_rows, i));
  }

  // push every row through the owning group (single-value groups store an index).
  // feature -> (group, sub) lookup by real feature index.
  std::vector<int> rf2group(num_features, -1), rf2sub(num_features, -1);
  for (int cf = 0; cf < total_feat; ++cf) {
    rf2group[real_feature_idx[cf]] = feature2group[cf];
    rf2sub[real_feature_idx[cf]] = feature2subfeature[cf];
  }
  for (int r = 0; r < num_rows; ++r) {
    for (int f = 0; f < num_features; ++f) {
      int g = rf2group[f];
      if (g < 0) continue;  // trivial / unused feature
      groups[g]->PushData(rf2sub[f], r, cs.columns[f][r]);
    }
  }
  for (auto* g : groups) g->FinishLoad();

  // 5. dump.
  out << "ECASE name=" << cs.name << " num_rows=" << num_rows
      << " num_features=" << num_features << " max_bin=" << cs.max_bin
      << " is_enable_sparse=" << (cs.is_enable_sparse ? 1 : 0)
      << " num_groups=" << num_groups << " num_used=" << used_features.size() << "\n";

  // full input matrix (raw f64 bits, row-major) so the Rust side rebuilds it.
  for (int f = 0; f < num_features; ++f) {
    out << "FCOL f=" << f << " ";
    for (int r = 0; r < num_rows; ++r) {
      if (r) out << ";";
      out << F64Bits(cs.columns[f][r]);
    }
    out << "\n";
  }

  // feature -> group membership (indexed by the packed cur_fidx order, which is
  // group-major; we also emit the REAL feature index per slot).
  out << "MEMBERSHIP ";
  for (int cf = 0; cf < total_feat; ++cf) {
    if (cf) out << ";";
    out << "real=" << real_feature_idx[cf] << ",group=" << feature2group[cf]
        << ",sub=" << feature2subfeature[cf];
  }
  out << "\n";

  // per-group layout: bin_offsets_, num_total_bin_, multi_val flag, sub-features.
  for (int i = 0; i < num_groups; ++i) {
    out << "GROUP g=" << i << " num_total_bin=" << groups[i]->num_total_bin_
        << " is_multi_val=" << (groups[i]->is_multi_val_ ? 1 : 0) << " bin_offsets=";
    for (size_t k = 0; k < groups[i]->bin_offsets_.size(); ++k) {
      if (k) out << ";";
      out << groups[i]->bin_offsets_[k];
    }
    out << " features=";
    for (size_t k = 0; k < features_in_group[i].size(); ++k) {
      if (k) out << ";";
      out << features_in_group[i][k];
    }
    out << "\n";

    // per-row bundled bin index for single-value groups (the bundled store).
    if (!groups[i]->is_multi_val_ && groups[i]->bin_data_) {
      out << "GROW g=" << i << " ";
      for (int r = 0; r < num_rows; ++r) {
        if (r) out << ";";
        out << groups[i]->bin_data_->Data(r);
      }
      out << "\n";
    }
  }

  for (auto* g : groups) delete g;
}

std::vector<EfbCaseSpec> BuildEfbCorpus(int master_seed) {
  std::vector<EfbCaseSpec> cases;
  CaseGen gen(master_seed);

  // (1) D-06 #4: mutually-exclusive sparse feature sets. Each feature is non-zero
  //     on a DISJOINT block of rows, so EFB bundles them all into ONE group.
  {
    EfbCaseSpec cs;
    cs.name = "mutually_exclusive_sparse";
    cs.num_rows = 40;
    cs.num_features = 4;
    cs.max_bin = 16;
    cs.is_enable_sparse = true;
    cs.columns.assign(cs.num_features, std::vector<double>(cs.num_rows, 0.0));
    // feature f is non-zero only on rows [f*10, f*10+10): disjoint blocks.
    for (int f = 0; f < cs.num_features; ++f) {
      for (int r = f * 10; r < f * 10 + 10; ++r) {
        cs.columns[f][r] = static_cast<double>((r % 5) + 1);  // bins 1..5
      }
    }
    cases.push_back(std::move(cs));
  }

  // (2) a second mutually-exclusive set with more features + uneven blocks.
  {
    EfbCaseSpec cs;
    cs.name = "mutually_exclusive_uneven";
    cs.num_rows = 30;
    cs.num_features = 3;
    cs.max_bin = 32;
    cs.is_enable_sparse = true;
    cs.columns.assign(cs.num_features, std::vector<double>(cs.num_rows, 0.0));
    // disjoint uneven blocks: f0 rows [0,5), f1 rows [5,20), f2 rows [20,30).
    int starts[] = {0, 5, 20};
    int ends[] = {5, 20, 30};
    for (int f = 0; f < cs.num_features; ++f) {
      for (int r = starts[f]; r < ends[f]; ++r) {
        cs.columns[f][r] = static_cast<double>((r % 7) + 1);
      }
    }
    cases.push_back(std::move(cs));
  }

  // (3) CONTROL: NO features are mutually exclusive (all non-zero on every row)
  //     -> bundling does nothing -> one single-feature group per feature.
  {
    EfbCaseSpec cs;
    cs.name = "control_no_bundle";
    cs.num_rows = 20;
    cs.num_features = 3;
    cs.max_bin = 16;
    cs.is_enable_sparse = true;
    cs.columns.assign(cs.num_features, std::vector<double>(cs.num_rows, 0.0));
    for (int f = 0; f < cs.num_features; ++f) {
      for (int r = 0; r < cs.num_rows; ++r) {
        // every feature non-zero on every row -> conflicts everywhere.
        cs.columns[f][r] = static_cast<double>((r % 4) + 1 + f);
      }
    }
    cases.push_back(std::move(cs));
  }

  return cases;
}

// ===========================================================================
// Default-config in-memory ingest golden (GAP-1/GAP-2 closure, SC#1/SC#5).
//
// This emitter exercises the DEFAULT scaled-filter_cnt path that every prior
// ingest golden avoided (they all forced pre_filter=false). With
// feature_pre_filter=true (default) and bin_construct_sample_cnt < num_rows
// (sampling active), C++ DatasetLoader::ConstructFromSampleData
// (dataset_loader.cpp:623-624) passes a SCALED pre-filter threshold
//   filter_cnt = static_cast<data_size_t>(
//       static_cast<double>(min_data_in_leaf * total_sample_size) / num_dist_data)
// to FindBin, NOT the raw min_data_in_leaf. For the dense in-memory path
// (c_api.cpp:1360-1374) total_sample_size = sample_cnt (k) and
// num_dist_data = total_nrow.
//
// Dimensions are chosen so the divergence is genuinely TRIGGERED:
//   num_rows = 200, sample_cnt = 50, min_data_in_leaf = 20
//   -> filter_cnt = (20 * 50) / 200 = 5 (integer truncation) != 20.
// At least one engineered feature has a best split whose minority side count
// lands in [5, 20): it PASSES need_filter at filter_cnt=5 (non-trivial) but
// FAILS at the raw threshold 20 (trivial), so is_trivial_ flips.
//
// The golden carries the RAW matrix (single source of truth for the data) so
// the Rust test reads byte-identical f64 cells and feeds them to from_mat —
// it never regenerates the Random-derived matrix. The ASSIGN line carries the
// STORED per-row bin (the value the dataset's FeatureGroup actually stores via
// the new_single + PushData path), so a flip in is_trivial_ shows up as a
// per-row store divergence too.
// ===========================================================================

constexpr int kDefaultIngestNumRows = 200;
constexpr int kDefaultIngestNumFeatures = 4;
constexpr int kDefaultIngestSampleCnt = 50;
constexpr int kDefaultIngestMinDataInLeaf = 20;
constexpr int kDefaultIngestMaxBin = 255;
constexpr int kDefaultIngestMinDataInBin = 3;

// Replicate FeatureGroup::new_single + PushData (feature_group.h:93-111,
// 253-267) to compute the STORED bin a single-value dense group would hold for
// `value`. Per C++ Dataset::Construct (dataset.cpp:337-343), a TRIVIAL feature is
// DROPPED — it gets no FeatureGroup and no store — so this is only ever called
// for NON-trivial features (the call site gates on !bm.is_trivial_). For a
// non-trivial single-value group this is exactly what the Rust FeatureGroup
// stores per row.
uint32_t StoredBinSingleGroup(const BinMapper& bm, double value) {
  // bin_offsets_[0] for new_single is 1 (num_total_bin_ seeded at 1).
  const uint32_t offset = 1;
  uint32_t bin = bm.ValueToBin(value);
  if (bin == bm.most_freq_bin_) {
    return 0;  // most-freq is skipped -> cell stays the default 0.
  }
  if (bm.most_freq_bin_ == 0) {
    bin -= 1;
  }
  bin += offset;
  return bin;
}

void EmitDefaultConfigIngest(std::ofstream& out, int master_seed) {
  const int num_rows = kDefaultIngestNumRows;
  const int num_features = kDefaultIngestNumFeatures;
  const int seed = master_seed + 4242;  // dedicated, stable seed for this golden

  // Build the synthetic matrix deterministically via the header-only Random
  // (no wall-clock entropy). columns[f][row].
  //
  // CRITICAL: the Rust ingest entry `from_mat` takes `&[f32]` and widens f32 ->
  // f64 at a single site. To stay byte-identical we generate every cell as an
  // f32-REPRESENTABLE value (cast through float, then store as double), so the
  // f64 binning arithmetic operates on exactly the same bits on both sides.
  std::vector<std::vector<double>> columns(num_features,
                                           std::vector<double>(num_rows, 0.0));
  // A single Random instance drives all feature value generation so the matrix
  // is a pure function of `seed`.
  LightGBM::Random gen(seed);
  for (int r = 0; r < num_rows; ++r) {
    // f0: continuous spread over [0, 100) -> non-trivial regardless of threshold.
    columns[0][r] = static_cast<double>(static_cast<float>(gen.NextInt(0, 1000)) / 10.0f);
    // f1: ENGINEERED flip feature. ~75% of rows take the common value 1.0, the
    // rest take a distinct high value 9.0. The best split separates the two
    // clusters; the minority cluster's sampled count lands in [5, 20) (here 10
    // of 50), so the feature is non-trivial at filter_cnt=5 (10 >= 5 on both
    // sides) but trivial at the raw threshold 20 (10 < 20). This is the flip.
    columns[1][r] = static_cast<double>((gen.NextInt(0, 100) < 25) ? 9.0f : 1.0f);
    // f2: near-constant (all 5.0 but a 2-row tail) -> trivial both ways.
    columns[2][r] = static_cast<double>((r < 2) ? 7.0f : 5.0f);
    // f3: balanced two-cluster (~50/50) -> non-trivial both ways.
    columns[3][r] = static_cast<double>((gen.NextInt(0, 100) < 50) ? 2.0f : 8.0f);
  }

  // DATASET header so the Rust test configures an IDENTICAL Config.
  out << "DATASET num_rows=" << num_rows << " num_features=" << num_features
      << " max_bin=" << kDefaultIngestMaxBin
      << " min_data_in_bin=" << kDefaultIngestMinDataInBin
      << " min_data_in_leaf=" << kDefaultIngestMinDataInLeaf
      << " bin_construct_sample_cnt=" << kDefaultIngestSampleCnt
      << " seed=" << seed << "\n";

  // RAW matrix (single source of truth). Row-major, f64 raw bits, ';'-joined.
  out << "MATRIX rows=" << num_rows << " cols=" << num_features << " ";
  bool first_cell = true;
  for (int r = 0; r < num_rows; ++r) {
    for (int f = 0; f < num_features; ++f) {
      if (!first_cell) out << ";";
      first_cell = false;
      out << F64Bits(columns[f][r]);
    }
  }
  out << "\n";

  // 1. Build a BinMapper per feature AND capture its sampled per-row values (the
  //    same single Random(seed).Sample draw used to bin) so the EFB grouping
  //    inputs reuse the ingest sampled set — no second RNG draw.
  std::vector<BinMapper> mappers;
  mappers.reserve(num_features);
  std::vector<std::vector<double>> sampled_columns(num_features);
  std::vector<int> filter_cnts(num_features, 0);
  for (int f = 0; f < num_features; ++f) {
    const std::vector<double>& col = columns[f];
    // Replicate the dense in-memory C-API sample path: Random(seed).Sample over
    // ALL rows then gather. total_sample_size = k (the sampled count).
    LightGBM::Random rand(seed);
    std::vector<int> idx = rand.Sample(num_rows, kDefaultIngestSampleCnt);
    std::vector<double> sampled;
    sampled.reserve(idx.size());
    for (int i : idx) sampled.push_back(col[i]);
    const int k = static_cast<int>(sampled.size());

    // SCALED filter_cnt exactly as dataset_loader.cpp:623-624 (divide in double
    // then truncate to int). total_sample_size = k, num_dist_data = num_rows.
    const int filter_cnt = static_cast<int>(
        static_cast<double>(kDefaultIngestMinDataInLeaf * k) / num_rows);
    filter_cnts[f] = filter_cnt;

    mappers.push_back(FindBinNumeric(sampled, kDefaultIngestMaxBin,
                                     kDefaultIngestMinDataInBin,
                                     /*min_split_data=*/filter_cnt,
                                     /*pre_filter=*/true, /*use_missing=*/true,
                                     /*zero_as_missing=*/false, sampled.size(),
                                     /*forced=*/{}));
    sampled_columns[f] = std::move(sampled);
  }
  const int sample_cnt = static_cast<int>(
      sampled_columns.empty() ? 0 : sampled_columns[0].size());

  // 2. Compute feature2group_/feature2subfeature_ via the IN-FILE verbatim
  //    FastFeatureBundling transcription over the !is_trivial_ used-feature set
  //    (dataset.cpp:337-369), mirroring the single C++ Dataset::Construct. There
  //    is NO LightGBM::Dataset object here (header-only harness; linking
  //    lib_lightgbm is forbidden) — the grouping comes from the in-file EFB
  //    transcription fed the BinMappers above.
  //
  //    SAMPLED-SET convention (matches Task 1's ingest EfbSamples, c_api.cpp:
  //    1352-1374): outer position i = 0..sample_cnt; keep value when
  //    |v| > kZeroThreshold || isnan(v); pushed index = i (sampled-set POSITION);
  //    total_sample_cnt = sample_cnt (NOT num_rows). On this dense fixture the
  //    grouping is convention-independent (single_val_max_conflict_cnt =
  //    sample_cnt/10000 = 0, every feature dense -> strict one-feature-per-group),
  //    so this is a robustness/documentation safeguard that also avoids the
  //    forbidden lib_lightgbm link — NOT a live correctness divergence today.
  constexpr double kZeroThreshold = 1e-35;
  std::vector<std::vector<int>> efb_idx(num_features);
  std::vector<std::vector<double>> efb_val(num_features);
  std::vector<int> efb_num_per_col(num_features, 0);
  for (int f = 0; f < num_features; ++f) {
    const std::vector<double>& sc = sampled_columns[f];
    for (int i = 0; i < static_cast<int>(sc.size()); ++i) {
      const double v = sc[i];
      if (std::fabs(v) > kZeroThreshold || std::isnan(v)) {
        efb_idx[f].push_back(i);  // sampled-set POSITION, not raw row
        efb_val[f].push_back(v);
      }
    }
    efb_num_per_col[f] = static_cast<int>(efb_idx[f].size());
  }

  // used_features = non-trivial features, in order (dataset.cpp:337-343).
  std::vector<int> used_features;
  for (int f = 0; f < num_features; ++f) {
    if (!mappers[f].is_trivial_) used_features.push_back(f);
  }

  std::vector<int*> idx_ptrs(num_features, nullptr);
  std::vector<double*> val_ptrs(num_features, nullptr);
  for (int f = 0; f < num_features; ++f) {
    idx_ptrs[f] = efb_idx[f].empty() ? nullptr : efb_idx[f].data();
    val_ptrs[f] = efb_val[f].empty() ? nullptr : efb_val[f].data();
  }

  std::vector<int8_t> group_is_multi_val(used_features.size(), 0);
  std::vector<std::vector<int>> features_in_group;
  if (!used_features.empty()) {
    // enable_bundle defaults true -> FastFeatureBundling (dataset.cpp:362-369).
    features_in_group = FastFeatureBundling(
        mappers, idx_ptrs.data(), val_ptrs.data(), efb_num_per_col.data(),
        num_features, /*total_sample_cnt=*/sample_cnt, used_features,
        /*num_data=*/num_rows, /*is_use_gpu=*/false, /*is_sparse=*/false,
        &group_is_multi_val);
  }

  // Build feature2group_/feature2subfeature_ over the used-feature grouping
  // (dataset.cpp:375-411). real feature -> (group, subfeature), -1 if trivial.
  std::vector<int> rf2group(num_features, -1), rf2sub(num_features, -1);
  for (int i = 0; i < static_cast<int>(features_in_group.size()); ++i) {
    const auto& fs = features_in_group[i];
    for (int j = 0; j < static_cast<int>(fs.size()); ++j) {
      rf2group[fs[j]] = i;
      rf2sub[fs[j]] = j;
    }
  }

  // 3. Emit. The FEATURE line carries is_trivial for ALL features (so the Rust
  //    test asserts the flip + the exclusion); a NON-trivial feature additionally
  //    carries group=/subfeature= and an ASSIGN line; a TRIVIAL feature gets NO
  //    group/subfeature and NO ASSIGN (it is dropped from the store —
  //    dataset.cpp:337-343).
  for (int f = 0; f < num_features; ++f) {
    const BinMapper& bm = mappers[f];
    out << "FEATURE f=" << f << " num_bin=" << bm.num_bin_
        << " missing_type=" << static_cast<int>(bm.missing_type_)
        << " default_bin=" << bm.default_bin_
        << " most_freq_bin=" << bm.most_freq_bin_
        << " is_trivial=" << (bm.is_trivial_ ? 1 : 0)
        << " filter_cnt=" << filter_cnts[f];
    if (!bm.is_trivial_) {
      out << " group=" << rf2group[f] << " subfeature=" << rf2sub[f];
    }
    out << " upper=";
    for (size_t i = 0; i < bm.bin_upper_bound_.size(); ++i) {
      if (i) out << ";";
      out << F64Bits(bm.bin_upper_bound_[i]);
    }
    out << "\n";

    // STORED per-row bin — ONLY for non-trivial features (trivial -> no store).
    if (!bm.is_trivial_) {
      const std::vector<double>& col = columns[f];
      out << "ASSIGN f=" << f << " ";
      for (int r = 0; r < num_rows; ++r) {
        if (r) out << ";";
        out << StoredBinSingleGroup(bm, col[r]);
      }
      out << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // argv:
  //   <numeric_out> <master_seed>
  //   <numeric_out> <master_seed> <storage_out>
  //   <numeric_out> <master_seed> <storage_out> <categorical_out> <missing_out>
  //   ... <missing_out> <metadata_out>
  //   ... <metadata_out> <efb_out>
  //   ... <efb_out> <default_config_out>
  //   ... <default_config_out> <example_out> <example_in1> [<example_in2> ...]
  if (argc != 3 && argc != 4 && argc != 6 && argc != 7 && argc != 8 && argc != 9 &&
      argc < 11) {
    std::cerr << "usage: bin_capture <numeric_out> <master_seed> "
                 "[<storage_out> [<categorical_out> <missing_out> "
                 "[<metadata_out> [<efb_out> [<default_config_out> "
                 "[<example_out> <example_in>...]]]]]]\n";
    return 2;
  }
  const std::string out_path = argv[1];
  const int master_seed = std::stoi(argv[2]);

  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "error: cannot open output file: " << out_path << "\n";
    return 1;
  }

  std::vector<CaseSpec> corpus = BuildCorpus(master_seed);

  out << "# LightGBM-rs numeric binning golden set (layers 1+2, randomized-at-capture, D-14)\n";
  out << "# Generated by xtask/cpp/bin_capture.cpp — a verbatim transcription of the pinned\n";
  out << "# LightGBM bin.cpp/bin.h numeric FindBin+ValueToBin (external_libs unbuildable here;\n";
  out << "# see file header + REFERENCE_MANIFEST.md). Sampling uses the real header-only Random.\n";
  out << "MASTER_SEED " << master_seed << "\n";
  out << "COUNTS cases=" << corpus.size() << "\n";

  for (const auto& cs : corpus) EmitCase(out, cs);

  out.flush();
  if (!out) {
    std::cerr << "error: failed while writing output\n";
    return 1;
  }
  std::cerr << "bin_capture: wrote " << corpus.size() << " numeric cases to " << out_path << "\n";

  if (argc >= 4) {
    const std::string store_path = argv[3];
    std::ofstream sout(store_path, std::ios::binary | std::ios::trunc);
    if (!sout) {
      std::cerr << "error: cannot open storage output file: " << store_path << "\n";
      return 1;
    }
    std::vector<StorageCaseSpec> scorpus = BuildStorageCorpus(master_seed);
    sout << "# LightGBM-rs bin-storage layout golden set (DAT-02, randomized-at-capture, D-14)\n";
    sout << "# Verbatim transcription of dense_bin.hpp/sparse_bin.hpp/feature_group.h storage\n";
    sout << "# layout (see bin_capture.cpp header + REFERENCE_MANIFEST.md). Byte-exact.\n";
    sout << "MASTER_SEED " << master_seed << "\n";
    sout << "COUNTS cases=" << scorpus.size() << "\n";
    for (const auto& cs : scorpus) EmitStorageCase(sout, cs);
    sout.flush();
    if (!sout) {
      std::cerr << "error: failed while writing storage output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote " << scorpus.size() << " storage cases to " << store_path << "\n";
  }

  if (argc >= 6) {
    // (a) categorical corpus (layers 1 + 3 + per-row index, DAT-04).
    const std::string cat_path = argv[4];
    std::ofstream cout(cat_path, std::ios::binary | std::ios::trunc);
    if (!cout) {
      std::cerr << "error: cannot open categorical output file: " << cat_path << "\n";
      return 1;
    }
    std::vector<CatCaseSpec> ccorpus = BuildCatCorpus(master_seed);
    cout << "# LightGBM-rs categorical folding golden set (layers 1+3 + per-row index, DAT-04)\n";
    cout << "# Verbatim transcription of bin.cpp/bin.h categorical FindBin+ValueToBin\n";
    cout << "# (external_libs unbuildable here; see file header + REFERENCE_MANIFEST.md).\n";
    cout << "MASTER_SEED " << master_seed << "\n";
    cout << "COUNTS cases=" << ccorpus.size() << "\n";
    for (const auto& cs : ccorpus) EmitCatCase(cout, cs);
    cout.flush();
    if (!cout) {
      std::cerr << "error: failed while writing categorical output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote " << ccorpus.size() << " categorical cases to " << cat_path
              << "\n";

    // (b) missing-edge corpus (layer 1 + per-row index, DAT-03).
    const std::string miss_path = argv[5];
    std::ofstream mout(miss_path, std::ios::binary | std::ios::trunc);
    if (!mout) {
      std::cerr << "error: cannot open missing output file: " << miss_path << "\n";
      return 1;
    }
    std::vector<MissCaseSpec> mcorpus = BuildMissCorpus(master_seed);
    mout << "# LightGBM-rs missing-value routing golden set (layer 1 + per-row index, DAT-03)\n";
    mout << "# Numeric FindBin/ValueToBin missing routing across use_missing/zero_as_missing.\n";
    mout << "MASTER_SEED " << master_seed << "\n";
    mout << "COUNTS cases=" << mcorpus.size() << "\n";
    for (const auto& cs : mcorpus) EmitMissCase(mout, cs);
    mout.flush();
    if (!mout) {
      std::cerr << "error: failed while writing missing output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote " << mcorpus.size() << " missing cases to " << miss_path
              << "\n";
  }

  if (argc >= 7) {
    // metadata corpus (labels/weights/init_score/query + CalculateQueryWeights).
    const std::string meta_path = argv[6];
    std::ofstream mout(meta_path, std::ios::binary | std::ios::trunc);
    if (!mout) {
      std::cerr << "error: cannot open metadata output file: " << meta_path << "\n";
      return 1;
    }
    std::vector<MetaCaseSpec> mcorpus = BuildMetaCorpus(master_seed);
    mout << "# LightGBM-rs metadata golden set (DAT-06: query-weight derivation)\n";
    mout << "# Verbatim transcription of metadata.cpp CalculateQueryWeights (label_t==float\n";
    mout << "# mean per group). f32 values are raw IEEE bits (u32 dec); init_score is f64 bits.\n";
    mout << "MASTER_SEED " << master_seed << "\n";
    mout << "COUNTS cases=" << mcorpus.size() << "\n";
    for (const auto& cs : mcorpus) EmitMetaCase(mout, cs);
    mout.flush();
    if (!mout) {
      std::cerr << "error: failed while writing metadata output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote " << mcorpus.size() << " metadata cases to " << meta_path
              << "\n";
  }

  if (argc >= 8) {
    // EFB grouping corpus (golden layer 3, DAT-05): argv[7] = output golden.
    const std::string efb_path = argv[7];
    std::ofstream fout(efb_path, std::ios::binary | std::ios::trunc);
    if (!fout) {
      std::cerr << "error: cannot open EFB output file: " << efb_path << "\n";
      return 1;
    }
    std::vector<EfbCaseSpec> fcorpus = BuildEfbCorpus(master_seed);
    fout << "# LightGBM-rs EFB grouping golden set (layer 3, DAT-05; D-06 #4)\n";
    fout << "# Verbatim header-only transcription of dataset.cpp FastFeatureBundling/\n";
    fout << "# FindGroups/GetConflictCount/FixSampleIndices + feature_group.h group\n";
    fout << "# layout (external_libs unvendored; see bin_capture.cpp header +\n";
    fout << "# REFERENCE_MANIFEST.md). RNG/shuffle use the real header-only Random.\n";
    fout << "MASTER_SEED " << master_seed << "\n";
    fout << "COUNTS cases=" << fcorpus.size() << "\n";
    for (const auto& cs : fcorpus) EmitEfbCase(fout, cs);
    fout.flush();
    if (!fout) {
      std::cerr << "error: failed while writing EFB output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote " << fcorpus.size() << " EFB cases to " << efb_path << "\n";
  }

  if (argc >= 9) {
    // default-config in-memory ingest golden (GAP-1/GAP-2): argv[8] = output
    // golden. FIXED positional slot, BEFORE the variadic example tail.
    const std::string default_cfg_out = argv[8];
    std::ofstream dout(default_cfg_out, std::ios::binary | std::ios::trunc);
    if (!dout) {
      std::cerr << "error: cannot open default-config output file: " << default_cfg_out << "\n";
      return 1;
    }
    dout << "# LightGBM-rs default-config in-memory ingest golden (GAP-1/GAP-2, SC#1/SC#5)\n";
    dout << "# Scaled-filter_cnt ConstructFromSampleData path (feature_pre_filter=true,\n";
    dout << "# bin_construct_sample_cnt < num_rows). filter_cnt = (min_data_in_leaf *\n";
    dout << "# sample_cnt) / num_rows (dataset_loader.cpp:623-624). Carries the raw matrix.\n";
    dout << "MASTER_SEED " << master_seed << "\n";
    EmitDefaultConfigIngest(dout, master_seed);
    dout.flush();
    if (!dout) {
      std::cerr << "error: failed while writing default-config output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote default-config ingest golden to " << default_cfg_out << "\n";
  }

  if (argc >= 11) {
    // example datasets: argv[9] = output golden, argv[10..] = copied input paths.
    const std::string example_out = argv[9];
    std::ofstream eout(example_out, std::ios::binary | std::ios::trunc);
    if (!eout) {
      std::cerr << "error: cannot open example output file: " << example_out << "\n";
      return 1;
    }
    eout << "# LightGBM-rs example-dataset binning golden set (DAT-07, layers 1+2)\n";
    eout << "# End-to-end FindBin+ValueToBin over copied LightGBM example datasets\n";
    eout << "# (read from the COMMITTED fixtures dir, never the untracked LightGBM/ tree).\n";
    eout << "MASTER_SEED " << master_seed << "\n";
    const int kFirstExampleArgv = 10;  // argv slot of the first example input
    const int num_examples = argc - kFirstExampleArgv;
    eout << "COUNTS datasets=" << num_examples << "\n";
    int written = 0;
    for (int a = kFirstExampleArgv; a < argc; ++a) {
      const std::string in_path = argv[a];
      // dataset name = the file stem (after the last '/').
      std::string name = in_path;
      const size_t slash = name.find_last_of('/');
      if (slash != std::string::npos) name = name.substr(slash + 1);
      // derive a per-dataset seed from the master seed + a STABLE positional base
      // (8 + index). The default-config arg was inserted at argv[8] and the
      // example block shifted from argv[8] to argv[9] (kFirstExampleArgv 9->10);
      // anchoring the seed to the original base (8) keeps the example goldens
      // byte-unchanged by this plan.
      const int seed = master_seed + 8 + (a - kFirstExampleArgv);
      EmitExampleDataset(eout, name, in_path, seed);
      written++;
    }
    eout.flush();
    if (!eout) {
      std::cerr << "error: failed while writing example output\n";
      return 1;
    }
    std::cerr << "bin_capture: wrote " << written << " example datasets to " << example_out << "\n";
  }
  return 0;
}
