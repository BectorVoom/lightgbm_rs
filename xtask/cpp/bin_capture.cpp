// bin_capture.cpp
//
// Dev-only C++ capture harness for the LightGBM-rs binning oracle (Phase 2,
// golden layers 1 + 2 — numeric features only; categorical -> Plan 03, EFB ->
// Plan 05).
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
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
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

// ---- A minimal BinMapper holding only what layers 1+2 need ----------------
struct BinMapper {
  int num_bin_ = 1;
  MissingType missing_type_ = MT_None;
  std::vector<double> bin_upper_bound_;
  bool is_trivial_ = true;
  double sparse_rate_ = 1.0;
  BinType bin_type_ = NumericalBin;
  uint32_t default_bin_ = 0;
  uint32_t most_freq_bin_ = 0;

  // ValueToBin (bin.h:612-650), numeric branch.
  uint32_t ValueToBin(double value) const {
    if (std::isnan(value)) {
      if (bin_type_ == CategoricalBin) return 0;
      else if (missing_type_ == MT_NaN) return static_cast<uint32_t>(num_bin_ - 1);
      else value = 0.0;
    }
    int l = 0, r = num_bin_ - 1;
    if (missing_type_ == MT_NaN) r -= 1;
    while (l < r) {
      int m = (r + l - 1) / 2;
      if (value <= bin_upper_bound_[m]) r = m;
      else l = m + 1;
    }
    return static_cast<uint32_t>(l);
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

// ---- serialization helpers ------------------------------------------------
uint64_t F64Bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(b));
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

}  // namespace

int main(int argc, char** argv) {
  // argv: <out_path> <master_seed> [<storage_out_path>]
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: bin_capture <out_path> <master_seed> [<storage_out_path>]\n";
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

  if (argc == 4) {
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
  return 0;
}
