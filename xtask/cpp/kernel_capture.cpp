// kernel_capture.cpp
//
// Dev-only C++ capture harness for the LightGBM-rs compute-kernel oracle
// (Phase 4, Plan 04-02): the HISTOGRAM golden layer (D-02 / D-02a). The
// split/partition layers are added in 04-03 by extending this same file.
//
// WHY THIS IS A FOCUSED, SELF-CONTAINED TRANSCRIPTION (not a compile of the
// in-tree feature_histogram.cpp / dense_bin.hpp):
//   The authoritative `ConstructHistogram` lives in `LightGBM/src/io/dense_bin.hpp`
//   and `LightGBM/src/io/sparse_bin.hpp`, both of which (via `<LightGBM/bin.h>`
//   -> `common.h`) pull in `fast_double_parser.h` + `fmt/format.h` from
//   `external_libs/`. In this repo those submodules are present only as EMPTY
//   directories (the LightGBM tree is git-untracked and its `external_libs/` are
//   not vendored — see project memory `lightgbm-ref-tree-untracked`). So neither
//   `dense_bin.hpp` nor `feature_histogram.cpp` can be compiled here.
//
//   This is the SAME situation Phase 1 hit for the RNG (rng_capture compiles the
//   header-only `Random` directly) and Phase 2 hit for binning (bin_capture
//   verbatim-transcribes FindBin/ValueToBin). The `ConstructHistogram`
//   accumulation body depends ONLY on:
//     - the bin index `data(idx)` from the binned store (DenseBin/SparseBin),
//     - `ti = static_cast<uint32_t>(data(idx)) << 1` (stride-2 [grad,hess]),
//     - `hist_t (= double)` accumulation of the f32 (`score_t = float`) grad/hess,
//   NONE of which touch fast_double_parser or fmt. The code below VERBATIM-
//   transcribes those bodies from the pinned `dense_bin.hpp:99-141` and
//   `sparse_bin.hpp:102-152` (commit 195c26fc, VERSION 4.6.0.99) — so it emits
//   goldens byte-identical to what lib_lightgbm would compute. It is the
//   authoritative reference source, just compiled without the unbuildable
//   external_libs dependency chain.
//
//   The DenseBin/SparseBin/BinMapper bin-storage forms are REUSED from
//   bin_capture.cpp's transcription (D-02a: reuse the Phase-2 binned-store
//   forms) so the histogram input is bit-faithful.
//
//   Synthetic inputs (sampling + grad/hess spread) are derived from the genuine
//   header-only `LightGBM::Random` (same as rng_capture) — not a re-transcription.
//
// Determinism / idempotency (D-14): the synthetic histogram corpus is derived
// solely from one recorded KERNEL_MASTER_SEED passed on argv. No wall-clock / OS
// entropy, so re-running produces byte-identical fixtures (empty `git diff`).
//
// Fixture format (line-delimited, '#'-prefixed comments ignored by the reader):
//   KERNEL_MASTER_SEED <seed>
//   COUNTS hist=<n>
//   HCASE name=<id> layout=<dense|sparse> num_bin=<n> num_rows=<n> \
//         skip_default_bin=<0|1> note=<text>
//   BINS <u32;u32;...>            # per-row bin index (== Bin::data(idx))
//   GRAD <f32bits;...>            # per-row ordered_gradients, raw f32 bits (u32 dec)
//   HESS <f32bits;...>            # per-row ordered_hessians,  raw f32 bits (u32 dec)
//   HIST <f64bits;...>            # the [g0,h0,g1,h1,...] f64 cells, raw f64 bits (u64 dec)
//
// f32 values are emitted as raw little-endian bit patterns (a u32 in decimal) and
// f64 cells as raw u64 bit patterns, so the Rust side asserts bit-exact equality
// with zero parsing rounding (compare_exact_f64_bits).

#include <LightGBM/utils/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Reference type aliases (LightGBM/include/LightGBM/meta.h): score_t = float
// (default; SCORE_T_USE_DOUBLE NOT defined, D-01), hist_t = double (Pitfall 3),
// data_size_t = int32_t.
// ---------------------------------------------------------------------------
typedef float score_t;
typedef double hist_t;
typedef int32_t data_size_t;

// ---------------------------------------------------------------------------
// Bin-storage forms — a focused REUSE of the DenseBin/SparseBin transcription in
// bin_capture.cpp (lines 588-714), restricted to what ConstructHistogram needs:
// Push(idx,value) / FinishLoad() / data(idx). VAL_T width selection mirrors
// Bin::CreateDenseBin / CreateSparseBin (bin.cpp:613-622). The histogram result
// is independent of VAL_T (data(idx) widens to u32), but covering u8/u16/u32
// widths exercises the same per-bit-width store paths the kernel reads against.
// ---------------------------------------------------------------------------
struct IBin {
  virtual ~IBin() {}
  virtual void Push(int idx, uint32_t value) = 0;
  virtual void FinishLoad() = 0;
  virtual uint32_t Data(int idx) const = 0;
  virtual bool IsSparse() const { return false; }
};

// DenseBin<VAL_T, IS_4BIT>.  width = sizeof(VAL_T). (bin_capture.cpp:600-641)
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
};

// SparseBin<VAL_T>.  (bin_capture.cpp:644-702)
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
      // WR-06: check the bound BEFORE dereferencing deltas_[i_delta]. The old
      // order read deltas_[i_delta] first and only then tested i_delta >=
      // num_vals_, so an idx past the last stored row (or a malformed delta
      // stream) was an out-of-bounds read in this golden-generator — UB that
      // would silently corrupt the authoritative fixture. Matches the upstream
      // sparse_bin.hpp discipline (bound-check immediately after ++i_delta).
      if (i_delta >= num_vals_) return 0;
      cur_pos += deltas_[i_delta];
      if (cur_pos == idx) return static_cast<uint32_t>(vals_[i_delta]);
      if (cur_pos > idx) return 0;
    }
  }
  bool IsSparse() const override { return true; }

  // VERBATIM transcription of sparse_bin.hpp:138-152 (ConstructHistogram, the
  // non-indices USE_HESSIAN f64 path). Iterates the stored non-zero values in
  // row (cur_pos) order: ti = vals_[i_delta] << 1; out[ti] += grad[cur_pos];
  // out[ti+1] += hess[cur_pos]. Rows whose stored bin is 0 are simply absent
  // from the sparse stream (Push drops cur_bin==0), so bin-0 cells stay 0 — the
  // sparse "default/most-freq-bin == 0 skip". The ACC_GH macro
  // (sparse_bin.hpp:102-105) is `ti = vals_[i_delta] << 1; hist[ti] += g;
  // hist[ti+1] += h;`.
  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* ordered_hessians,
                          hist_t* out) const {
    data_size_t i_delta = -1, cur_pos = 0;
    // InitIndex(start, &i_delta, &cur_pos) for start==0 reduces to i_delta=-1,
    // cur_pos=0 then the first ++i_delta below advances to deltas_[0]; we inline
    // the start==0 fast path (our captures always use start=0, end=num_rows).
    cur_pos += deltas_[++i_delta];
    while (cur_pos < start && i_delta < num_vals_) {
      cur_pos += deltas_[++i_delta];
    }
    while (cur_pos < end && i_delta < num_vals_) {
      const uint32_t ti = static_cast<uint32_t>(vals_[i_delta]) << 1;
      out[ti] += ordered_gradients[cur_pos];
      out[ti + 1] += ordered_hessians[cur_pos];
      cur_pos += deltas_[++i_delta];
    }
  }
};

IBin* CreateDenseBin(int num_data, int num_bin) {
  if (num_bin <= 16) return new DenseBin<uint8_t, true>(num_data);
  else if (num_bin <= 256) return new DenseBin<uint8_t, false>(num_data);
  else if (num_bin <= 65536) return new DenseBin<uint16_t, false>(num_data);
  else return new DenseBin<uint32_t, false>(num_data);
}

// ---------------------------------------------------------------------------
// VERBATIM transcription of dense_bin.hpp:130-141 — the non-prefetch,
// non-indices, USE_HESSIAN ConstructHistogramInner tail:
//
//   for (; i < end; ++i) {
//     const auto idx = i;                                  // USE_INDICES=false
//     const auto ti = static_cast<uint32_t>(data(idx)) << 1;
//     grad[ti] += ordered_gradients[i];                    // hist_t* grad = out
//     hess[ti] += ordered_hessians[i];                     // hist_t* hess = out+1
//   }
//
// f32 read (score_t), f64 accumulate (hist_t = double). Stride-2 [grad,hess]
// with grad at out[ti], hess at out[ti+1]. This is exactly the single-owner
// ordered fold the cubecl-cpu kernel reproduces (the bit-exact anchor, 04-01).
// ---------------------------------------------------------------------------
void DenseConstructHistogram(const IBin& bin, data_size_t start, data_size_t end,
                             const score_t* ordered_gradients,
                             const score_t* ordered_hessians, hist_t* out) {
  hist_t* grad = out;
  hist_t* hess = out + 1;
  for (data_size_t i = start; i < end; ++i) {
    const data_size_t idx = i;
    const uint32_t ti = static_cast<uint32_t>(bin.Data(idx)) << 1;
    grad[ti] += ordered_gradients[i];
    hess[ti] += ordered_hessians[i];
  }
}

// ---- serialization helpers (bin_capture.cpp:1061-1071) --------------------
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

// ===========================================================================
// 04-03: GAIN MATH + SCAN + PARTITION + SUBTRACT transcription.
//
// VERBATIM transcription of the default-CPU-path gain math + best-threshold scan
// from `LightGBM/src/treelearner/feature_histogram.hpp:711-1057` (commit
// 195c26fc, VERSION 4.6.0.99), the partition routing from
// `dense_bin.hpp:314-394` (`SplitInner`, MissingType::None), and the subtraction
// math from `feature_histogram.hpp:99-145` (default USE_DIST_GRAD=false path).
// These depend ONLY on the f64 histogram cells / bin indices and the scalar
// config — none of fast_double_parser / fmt — so they compile standalone here,
// exactly as the histogram transcription above (external_libs unvendored).
//
// kEpsilon = 1e-15f (meta.h:54); kMinScore = -inf (meta.h:50). RoundInt(x) =
// (int)(x + 0.5f) (common.h:904). Sign(x) = (x>0)-(x<0) (common.h:872). The
// `2*kEpsilon` entry bump (feature_histogram.hpp:172), the REVERSE
// `sum_right_hessian = kEpsilon` seed (:856), and the FORWARD `sum_left_hessian =
// kEpsilon` seed (:939) are reproduced verbatim, as are the gate ORDER and the
// `t-1+offset` (REVERSE) vs `t+offset` (FORWARD) threshold recording.
// ===========================================================================
static const float kEpsilonF = 1e-15f;       // meta.h: const score_t kEpsilon = 1e-15f;
static const double kMinScore = -std::numeric_limits<double>::infinity();

static int CppSign(double x) { return (x > 0.0) - (x < 0.0); }
static int CppRoundInt(double x) { return static_cast<int>(x + 0.5f); }

static double ThresholdL1(double s, double l1) {
  const double reg_s = std::max(0.0, std::fabs(s) - l1);
  return CppSign(s) * reg_s;
}
// GetLeafGain<USE_L1,false,false> (feature_histogram.hpp:799-815 fast path).
static double GetLeafGain(bool use_l1, double g, double h, double l1, double l2) {
  if (use_l1) {
    const double sg = ThresholdL1(g, l1);
    return (sg * sg) / (h + l2);
  }
  return (g * g) / (h + l2);
}
// GetSplitGains<false,USE_L1,false,false> (:757-797, !USE_MC branch).
static double GetSplitGains(bool use_l1, double lg, double lh, double rg, double rh,
                            double l1, double l2) {
  return GetLeafGain(use_l1, lg, lh, l1, l2) + GetLeafGain(use_l1, rg, rh, l1, l2);
}
// CalculateSplittedLeafOutput<USE_L1,false,false> (:716-738 path).
static double CalcLeafOutput(bool use_l1, double g, double h, double l1, double l2) {
  if (use_l1) return -ThresholdL1(g, l1) / (h + l2);
  return -g / (h + l2);
}

// The split config (the seven GainConfig fields + the bin-layout descriptors).
struct SplitCfg {
  int min_data_in_leaf;
  double min_sum_hessian_in_leaf;
  double lambda_l1;
  double lambda_l2;
  double min_gain_to_split;
  int num_bin;
  int offset;
  int default_bin;
  // AUTHORITATIVE C++ dispatch flags (feature_histogram.hpp:284-285), derived
  // from missing_type + num_bin>2 by the caller — NOT the Phase-4
  // `default_bin < num_bin` heuristic (removed Plan 05-01, RESEARCH Pitfall 1):
  //   skip_default_bin == (num_bin > 2 && missing_type == Zero)
  //   na_as_missing    == (num_bin > 2 && missing_type == NaN)
  // both false for missing_type == None.
  bool skip_default_bin;
  bool na_as_missing;
};

// The decoded winner (mirrors the Rust SplitInfo).
struct WinSplit {
  bool is_splittable = false;
  uint32_t threshold = 0;
  double gain = kMinScore;       // RAW best_gain (before -min_gain_shift)
  int left_count = 0;
  int right_count = 0;
  double left_sum_gradient = 0.0;
  double left_sum_hessian = 0.0; // reported (kEpsilon already subtracted)
  double right_sum_gradient = 0.0;
  double right_sum_hessian = 0.0;
  double left_output = 0.0;
  double right_output = 0.0;
  bool default_left = true;
};

// VERBATIM FindBestThresholdSequentially (feature_histogram.hpp:830-1057) for
// <USE_RAND=false,USE_MC=false,USE_L1=?,USE_MAX_OUTPUT=false,USE_SMOOTHING=false,
//  NA_AS_MISSING=false>. Records PER-CANDIDATE gains (a gain value when a bin is
// a considered candidate, NaN when it is gated out / skipped) for BOTH branches
// so the golden localizes a divergence to the scan, not just the winner.
//
// `sum_hessian` here is ALREADY bumped by 2*kEpsilon (the FindBestThreshold entry
// at :172); `min_gain_shift = GetLeafGain(un-bumped totals) + min_gain_to_split`.
static WinSplit FindBestThreshold(const std::vector<double>& hist, const SplitCfg& cfg,
                                  bool use_l1, double sum_gradient,
                                  double sum_hessian_bumped, int num_data,
                                  double min_gain_shift,
                                  std::vector<double>* cand_rev,
                                  std::vector<double>* cand_fwd) {
  const double l1 = cfg.lambda_l1, l2 = cfg.lambda_l2;
  const int offset = cfg.offset;
  const double cnt_factor = static_cast<double>(num_data) / sum_hessian_bumped;
  const double qnan = std::numeric_limits<double>::quiet_NaN();

  double best_sum_left_gradient = 0.0, best_sum_left_hessian = 0.0;
  double best_gain = kMinScore;
  int best_left_count = 0;
  uint32_t best_threshold = static_cast<uint32_t>(cfg.num_bin);
  bool is_splittable = false;
  bool best_default_left = true;

  auto GET_GRAD = [&](int t) { return hist[(static_cast<size_t>(t) << 1)]; };
  auto GET_HESS = [&](int t) { return hist[(static_cast<size_t>(t) << 1) + 1]; };

  // ---- REVERSE (:854-936): t high -> low, record t-1+offset ----
  cand_rev->clear();
  {
    double sum_right_gradient = 0.0;
    double sum_right_hessian = kEpsilonF;  // :856
    int right_count = 0;
    int t = cfg.num_bin - 1 - offset;      // NA_AS_MISSING=0
    const int t_end = 1 - offset;
    for (; t >= t_end; --t) {
      if (cfg.skip_default_bin && (t + offset) == cfg.default_bin) {
        cand_rev->push_back(qnan);
        continue;
      }
      sum_right_gradient += GET_GRAD(t);
      sum_right_hessian += GET_HESS(t);
      right_count += CppRoundInt(GET_HESS(t) * cnt_factor);
      if (right_count < cfg.min_data_in_leaf ||
          sum_right_hessian < cfg.min_sum_hessian_in_leaf) {
        cand_rev->push_back(qnan);
        continue;
      }
      int left_count = num_data - right_count;
      if (left_count < cfg.min_data_in_leaf) { cand_rev->push_back(qnan); break; }
      double sum_left_hessian = sum_hessian_bumped - sum_right_hessian;
      if (sum_left_hessian < cfg.min_sum_hessian_in_leaf) { cand_rev->push_back(qnan); break; }
      double sum_left_gradient = sum_gradient - sum_right_gradient;
      double current_gain = GetSplitGains(use_l1, sum_left_gradient, sum_left_hessian,
                                          sum_right_gradient, sum_right_hessian, l1, l2);
      if (current_gain <= min_gain_shift) { cand_rev->push_back(qnan); continue; }
      cand_rev->push_back(current_gain);
      is_splittable = true;
      if (current_gain > best_gain) {
        best_left_count = left_count;
        best_sum_left_gradient = sum_left_gradient;
        best_sum_left_hessian = sum_left_hessian;
        best_threshold = static_cast<uint32_t>(t - 1 + offset);  // :933
        best_gain = current_gain;
        best_default_left = true;  // REVERSE
      }
    }
  }

  // ---- FORWARD (:937-1029): t low -> high, record t+offset ----
  cand_fwd->clear();
  {
    double sum_left_gradient = 0.0;
    double sum_left_hessian = kEpsilonF;  // :939
    int left_count = 0;
    int t = 0;
    const int t_end = cfg.num_bin - 2 - offset;
    for (; t <= t_end; ++t) {
      if (cfg.skip_default_bin && (t + offset) == cfg.default_bin) {
        cand_fwd->push_back(qnan);
        continue;
      }
      // t >= 0 always (NA_AS_MISSING=0).
      sum_left_gradient += GET_GRAD(t);
      sum_left_hessian += GET_HESS(t);
      left_count += CppRoundInt(GET_HESS(t) * cnt_factor);
      if (left_count < cfg.min_data_in_leaf ||
          sum_left_hessian < cfg.min_sum_hessian_in_leaf) {
        cand_fwd->push_back(qnan);
        continue;
      }
      int right_count = num_data - left_count;
      if (right_count < cfg.min_data_in_leaf) { cand_fwd->push_back(qnan); break; }
      double sum_right_hessian = sum_hessian_bumped - sum_left_hessian;
      if (sum_right_hessian < cfg.min_sum_hessian_in_leaf) { cand_fwd->push_back(qnan); break; }
      double sum_right_gradient = sum_gradient - sum_left_gradient;
      double current_gain = GetSplitGains(use_l1, sum_left_gradient, sum_left_hessian,
                                          sum_right_gradient, sum_right_hessian, l1, l2);
      if (current_gain <= min_gain_shift) { cand_fwd->push_back(qnan); continue; }
      cand_fwd->push_back(current_gain);
      is_splittable = true;
      if (current_gain > best_gain) {
        best_left_count = left_count;
        best_sum_left_gradient = sum_left_gradient;
        best_sum_left_hessian = sum_left_hessian;
        best_threshold = static_cast<uint32_t>(t + offset);  // :1025
        best_gain = current_gain;
        best_default_left = false;  // FORWARD
      }
    }
  }

  // ---- finalization (:1031-1056) ----
  WinSplit w;
  if (is_splittable && best_gain > kMinScore) {  // output->gain starts at kMinScore
    w.is_splittable = true;
    w.threshold = best_threshold;
    w.left_output = CalcLeafOutput(use_l1, best_sum_left_gradient, best_sum_left_hessian, l1, l2);
    w.left_count = best_left_count;
    w.left_sum_gradient = best_sum_left_gradient;
    w.left_sum_hessian = best_sum_left_hessian - kEpsilonF;  // :1042
    double rsg = sum_gradient - best_sum_left_gradient;
    double rsh = sum_hessian_bumped - best_sum_left_hessian;
    w.right_output = CalcLeafOutput(use_l1, rsg, rsh, l1, l2);
    w.right_count = num_data - best_left_count;
    w.right_sum_gradient = rsg;
    w.right_sum_hessian = rsh - kEpsilonF;  // :1053
    w.gain = best_gain;  // RAW; the Rust host applies -min_gain_shift (penalty=1)
    w.default_left = best_default_left;
  }
  return w;
}

// VERBATIM SplitInner routing (dense_bin.hpp:314-394, MissingType::None path:
// MISS_IS_ZERO=false, MISS_IS_NA=false, MFB_IS_ZERO=false, MFB_IS_NA=false,
// USE_MIN_BIN=true). Emits the per-row route (0=lte/left, 1=gt/right).
static std::vector<int> SplitRoute(const std::vector<uint32_t>& bins, int min_bin,
                                   int max_bin, int threshold, int most_freq_bin) {
  int th = threshold + min_bin;
  if (most_freq_bin == 0) --th;
  const bool default_to_right = !(most_freq_bin <= threshold);
  std::vector<int> route(bins.size(), 0);
  for (size_t i = 0; i < bins.size(); ++i) {
    const int bin = static_cast<int>(bins[i]);
    const bool is_default = (bin < min_bin || bin > max_bin);
    const bool gt = bin > th;
    route[i] = (is_default ? default_to_right : gt) ? 1 : 0;
  }
  return route;
}

// A deterministic case generator reusing the reference Random LCG so the corpus
// is reproducible from KERNEL_MASTER_SEED (bin_capture.cpp:1073-1080).
struct CaseGen {
  LightGBM::Random rng;
  explicit CaseGen(int master_seed) : rng(master_seed) {}
  int NextInt(int lo, int hi) { return rng.NextInt(lo, hi); }
  float NextFloat() { return rng.NextFloat(); }
};

struct HCaseSpec {
  std::string name;
  bool sparse;              // dense vs sparse store layout
  int num_bin;
  std::vector<uint32_t> bins;   // per-row bin index (0..num_bin)
  std::vector<float> grad;      // per-row ordered_gradients
  std::vector<float> hess;      // per-row ordered_hessians
  bool skip_default_bin;        // whether this case stresses the bin-0 (default) skip
  std::string note;
};

void EmitHCase(std::ofstream& out, const HCaseSpec& cs) {
  const data_size_t n = static_cast<data_size_t>(cs.bins.size());

  // Build the layout-appropriate bin store, FinishLoad, then read back the
  // per-row bin index via data(idx) — this is EXACTLY the input the Rust
  // `construct_histograms` kernel receives (it folds over `Bin::data(idx)` for
  // every row, dense-style). For a DenseBin this round-trips the pushed bins
  // verbatim; for a SparseBin it round-trips through the delta-encoded stream
  // (rows whose stored bin is 0 read back as 0). The histogram is then the
  // single-owner ordered f64 fold over those read-back bins (dense_bin.hpp:130-141),
  // which the cubecl-cpu kernel reproduces bit-exact (the D-04 anchor).
  //
  // NOTE on the sparse `ConstructHistogram` (sparse_bin.hpp:138-152) transcribed
  // above: it is numerically identical to this dense fold for every NON-zero bin
  // (it iterates the same (row, bin, grad, hess) tuples in row order), differing
  // only in that it never touches bin-0 cells. The sparse corpus below therefore
  // uses bins in [1, num_bin) (no bin-0 rows) so the dense fold over the
  // round-tripped `data(idx)` == the sparse `ConstructHistogram` — exercising the
  // sparse store path while keeping the golden replayable by the dense kernel.
  std::vector<uint32_t> bins(n);
  std::vector<hist_t> hist(static_cast<size_t>(2 * cs.num_bin), 0.0);
  if (cs.sparse) {
    // Width selection mirrors Bin::CreateSparseBin (bin.cpp): u8 for num_bin<=256.
    IBin* sb = nullptr;
    if (cs.num_bin <= 256) sb = new SparseBin<uint8_t>(n);
    else if (cs.num_bin <= 65536) sb = new SparseBin<uint16_t>(n);
    else sb = new SparseBin<uint32_t>(n);
    for (data_size_t i = 0; i < n; ++i) sb->Push(i, cs.bins[i]);
    sb->FinishLoad();
    for (data_size_t i = 0; i < n; ++i) bins[i] = sb->Data(i);
    delete sb;
  } else {
    IBin* db = CreateDenseBin(n, cs.num_bin);
    for (data_size_t i = 0; i < n; ++i) db->Push(i, cs.bins[i]);
    db->FinishLoad();
    for (data_size_t i = 0; i < n; ++i) bins[i] = db->Data(i);
    delete db;
  }
  // Single-owner ordered f64 fold over the round-tripped per-row bins.
  {
    hist_t* grad = hist.data();
    hist_t* hess = hist.data() + 1;
    for (data_size_t i = 0; i < n; ++i) {
      const uint32_t ti = static_cast<uint32_t>(bins[i]) << 1;
      grad[ti] += cs.grad[i];
      hess[ti] += cs.hess[i];
    }
  }

  out << "HCASE name=" << cs.name << " layout=" << (cs.sparse ? "sparse" : "dense")
      << " num_bin=" << cs.num_bin << " num_rows=" << n
      << " skip_default_bin=" << (cs.skip_default_bin ? 1 : 0)
      << " note=" << cs.note << "\n";

  out << "BINS ";
  for (size_t i = 0; i < bins.size(); ++i) {
    if (i) out << ";";
    out << bins[i];
  }
  out << "\n";

  out << "GRAD ";
  for (size_t i = 0; i < cs.grad.size(); ++i) {
    if (i) out << ";";
    out << F32Bits(cs.grad[i]);
  }
  out << "\n";

  out << "HESS ";
  for (size_t i = 0; i < cs.hess.size(); ++i) {
    if (i) out << ";";
    out << F32Bits(cs.hess[i]);
  }
  out << "\n";

  out << "HIST ";
  for (size_t i = 0; i < hist.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(hist[i]);
  }
  out << "\n";
}

// Build the histogram corpus deterministically from the master seed (D-02a path
// coverage): dense + sparse layouts, most-frequent/default-bin (bin-0) skip,
// missing/zero routing, multiple bit widths (u8 4-bit / u8 / u16 / u32), and a
// grad/hess sign+magnitude spread that stresses the f64 reduction.
std::vector<HCaseSpec> BuildHistCorpus(int master_seed) {
  std::vector<HCaseSpec> cases;
  CaseGen gen(master_seed);

  // A grad/hess spread that mixes signs and spans ~1e-3 .. ~1e3 so the
  // non-associative f64 summation order is observable (matches the 04-01 spike
  // synthetic flavor).
  auto spread_grad = [&](int n, int salt) {
    std::vector<float> g(n);
    for (int i = 0; i < n; ++i) {
      // Deterministic magnitude ladder × random sign, all from the LCG.
      const float mag = (i % 4 == 0) ? 1e3f : (i % 4 == 1) ? 1e-3f
                       : (i % 4 == 2) ? 7.5f : 0.125f;
      const float sign = (gen.NextInt(0, 2) == 0) ? 1.0f : -1.0f;
      const float jitter = gen.NextFloat();        // [0,1)
      g[i] = sign * mag * (1.0f + jitter) + static_cast<float>(salt) * 0.0f;
    }
    return g;
  };
  auto spread_hess = [&](int n) {
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) {
      const float mag = (i % 3 == 0) ? 2.0e2f : (i % 3 == 1) ? 5e-2f : 0.75f;
      h[i] = mag * (1.0f + gen.NextFloat());       // hessians are non-negative
    }
    return h;
  };
  // Random per-row bins. `lo` is the lowest bin emitted: dense cases use lo=0
  // (bin-0 / default-bin rows ARE folded into out[0]/out[1], exercising the
  // dense default-bin accumulation), while sparse cases use lo=1 (no bin-0
  // rows) so the dense fold over the round-tripped data(idx) equals the sparse
  // ConstructHistogram, which never touches bin-0 cells. With `bias_low` the
  // case piles extra rows on `lo` (the default/most-freq bin) to stress that
  // routing path.
  auto rand_bins = [&](int n, int num_bin, int lo, bool bias_low) {
    std::vector<uint32_t> b(n);
    for (int i = 0; i < n; ++i) {
      if (bias_low && gen.NextInt(0, 2) == 0) {
        b[i] = static_cast<uint32_t>(lo);  // route to the default/most-freq bin
      } else {
        b[i] = static_cast<uint32_t>(gen.NextInt(lo, num_bin));
      }
    }
    return b;
  };

  // Bit-width sweep: num_bin chosen to land in each Create*Bin width bucket.
  //   8   -> DenseBin<u8,4bit>  / SparseBin<u8>
  //   200 -> DenseBin<u8>       / SparseBin<u8>
  //   600 -> DenseBin<u16>      / SparseBin<u16>
  //   70000-> DenseBin<u32>     / SparseBin<u32>
  struct WidthCfg { const char* tag; int num_bin; int num_rows; };
  const WidthCfg widths[] = {
      {"w4bit", 8, 40},
      {"w8", 200, 64},
      {"w16", 600, 48},
      {"w32", 70000, 32},
  };

  for (const auto& w : widths) {
    for (int sparse = 0; sparse <= 1; ++sparse) {
      // sparse cases use lo=1 (no bin-0 rows; see EmitHCase note); dense use lo=0.
      const int lo = sparse ? 1 : 0;
      // (a) general spread, bins uniform across [lo, num_bin).
      {
        HCaseSpec cs;
        cs.name = std::string(w.tag) + (sparse ? "_sparse" : "_dense") + "_spread";
        cs.sparse = (sparse != 0);
        cs.num_bin = w.num_bin;
        cs.bins = rand_bins(w.num_rows, w.num_bin, lo, /*bias_low=*/false);
        cs.grad = spread_grad(w.num_rows, 0);
        cs.hess = spread_hess(w.num_rows);
        cs.skip_default_bin = false;
        cs.note = "uniform-bins-gradhess-spread";
        cases.push_back(std::move(cs));
      }
      // (b) default-bin skip / low-bin routing: pile rows on the lowest bin
      //     (bin 0 for dense — the default-bin accumulation; bin 1 for sparse —
      //     the lowest non-zero bin, since sparse never folds bin 0).
      {
        HCaseSpec cs;
        cs.name = std::string(w.tag) + (sparse ? "_sparse" : "_dense") + "_defbin";
        cs.sparse = (sparse != 0);
        cs.num_bin = w.num_bin;
        cs.bins = rand_bins(w.num_rows, w.num_bin, lo, /*bias_low=*/true);
        cs.grad = spread_grad(w.num_rows, 1);
        cs.hess = spread_hess(w.num_rows);
        cs.skip_default_bin = true;
        cs.note = "default-low-bin-routing";
        cases.push_back(std::move(cs));
      }
    }
  }

  // (c) all-rows-on-one-nonzero-bin (degenerate accumulation, single cell).
  {
    HCaseSpec cs;
    cs.name = "single_bin_pileup";
    cs.sparse = false;
    cs.num_bin = 16;
    const int n = 24;
    cs.bins.assign(n, 5u);  // every row -> bin 5
    cs.grad = spread_grad(n, 2);
    cs.hess = spread_hess(n);
    cs.skip_default_bin = false;
    cs.note = "all-rows-one-bin";
    cases.push_back(std::move(cs));
  }
  // (d) all-rows-on-bin-0 through the SPARSE store: every Push has cur_bin==0 so
  //     the delta stream is EMPTY; data(idx) reads back 0 for every row, and the
  //     dense fold over those zeros piles all grad/hess into out[0]/out[1]. This
  //     exercises the empty-sparse-stream round-trip and the bin-0 default path.
  {
    HCaseSpec cs;
    cs.name = "all_bin0_sparse";
    cs.sparse = true;
    cs.num_bin = 16;
    const int n = 20;
    cs.bins.assign(n, 0u);
    cs.grad = spread_grad(n, 3);
    cs.hess = spread_hess(n);
    cs.skip_default_bin = true;
    cs.note = "all-bin0-empty-sparse-stream-roundtrip";
    cases.push_back(std::move(cs));
  }

  return cases;
}

// ===========================================================================
// 04-03 SPLIT golden. Each case fixes a histogram + GainConfig + offset/default
// + leaf totals (sum_gradient/sum_hessian/num_data) and emits per-candidate
// gains (REVERSE + FORWARD) AND the winning SplitInfo, so a failure localizes to
// the gain scan, not just the winner. The corpus covers BOTH a reverse-winner
// (default_left=1, threshold t-1+offset) and a forward-winner case, plus a
// default-bin-skip case (D-02a routing) and a no-admissible-split case.
// ===========================================================================
struct SCaseSpec {
  std::string name;
  SplitCfg cfg;
  std::vector<double> hist;  // 2*num_bin f64 cells
  double sum_gradient;       // leaf totals (UN-bumped sum_hessian)
  double sum_hessian;
  int num_data;
  std::string note;
};

void EmitSCase(std::ofstream& out, const SCaseSpec& cs) {
  const bool use_l1 = cs.cfg.lambda_l1 > 0.0;
  // BeforeNumerical (feature_histogram.hpp:198-207): gain_shift over UN-bumped
  // totals; min_gain_shift = gain_shift + min_gain_to_split.
  const double gain_shift =
      GetLeafGain(use_l1, cs.sum_gradient, cs.sum_hessian, cs.cfg.lambda_l1, cs.cfg.lambda_l2);
  const double min_gain_shift = gain_shift + cs.cfg.min_gain_to_split;
  // FindBestThreshold entry bump (:172): sum_hessian + 2*kEpsilon.
  const double sum_hessian_bumped = cs.sum_hessian + 2.0 * static_cast<double>(kEpsilonF);

  std::vector<double> cand_rev, cand_fwd;
  WinSplit w = FindBestThreshold(cs.hist, cs.cfg, use_l1, cs.sum_gradient,
                                 sum_hessian_bumped, cs.num_data, min_gain_shift,
                                 &cand_rev, &cand_fwd);

  out << "SCASE name=" << cs.name << " num_bin=" << cs.cfg.num_bin
      << " offset=" << cs.cfg.offset << " default_bin=" << cs.cfg.default_bin
      << " skip_default_bin=" << (cs.cfg.skip_default_bin ? 1 : 0)
      << " na_as_missing=" << (cs.cfg.na_as_missing ? 1 : 0)
      << " use_l1=" << (use_l1 ? 1 : 0)
      << " min_data_in_leaf=" << cs.cfg.min_data_in_leaf
      << " min_sum_hessian_in_leaf=" << F64Bits(cs.cfg.min_sum_hessian_in_leaf)
      << " lambda_l1=" << F64Bits(cs.cfg.lambda_l1)
      << " lambda_l2=" << F64Bits(cs.cfg.lambda_l2)
      << " min_gain_to_split=" << F64Bits(cs.cfg.min_gain_to_split)
      << " sum_gradient=" << F64Bits(cs.sum_gradient)
      << " sum_hessian=" << F64Bits(cs.sum_hessian) << " num_data=" << cs.num_data
      << " note=" << cs.note << "\n";

  out << "SHIST ";
  for (size_t i = 0; i < cs.hist.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(cs.hist[i]);
  }
  out << "\n";

  out << "SCAND_REV ";
  for (size_t i = 0; i < cand_rev.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(cand_rev[i]);
  }
  out << "\n";
  out << "SCAND_FWD ";
  for (size_t i = 0; i < cand_fwd.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(cand_fwd[i]);
  }
  out << "\n";

  // The Rust host reports gain = (raw_gain - min_gain_shift)*penalty (penalty=1);
  // we emit BOTH min_gain_shift and the raw gain so the parity can reconstruct.
  const double reported_gain =
      w.is_splittable ? (w.gain - min_gain_shift) : kMinScore;
  out << "SWIN is_splittable=" << (w.is_splittable ? 1 : 0)
      << " threshold=" << w.threshold
      << " gain=" << F64Bits(reported_gain)
      << " min_gain_shift=" << F64Bits(min_gain_shift)
      << " left_count=" << w.left_count << " right_count=" << w.right_count
      << " left_sum_gradient=" << F64Bits(w.left_sum_gradient)
      << " left_sum_hessian=" << F64Bits(w.left_sum_hessian)
      << " right_sum_gradient=" << F64Bits(w.right_sum_gradient)
      << " right_sum_hessian=" << F64Bits(w.right_sum_hessian)
      << " left_output=" << F64Bits(w.left_output)
      << " right_output=" << F64Bits(w.right_output)
      << " default_left=" << (w.default_left ? 1 : 0) << "\n";
}

std::vector<SCaseSpec> BuildSplitCorpus() {
  std::vector<SCaseSpec> cases;
  auto cfg = [](int nb, int mdl, double msh, double l1, double l2, double mgts,
                int off, int defb, bool skip, bool na = false) {
    SplitCfg c;
    c.num_bin = nb; c.min_data_in_leaf = mdl; c.min_sum_hessian_in_leaf = msh;
    c.lambda_l1 = l1; c.lambda_l2 = l2; c.min_gain_to_split = mgts;
    c.offset = off; c.default_bin = defb; c.skip_default_bin = skip;
    c.na_as_missing = na;
    return c;
  };

  // (a) FORWARD-winner: low bins clearly negative grad, high bins positive grad.
  //     With offset=0 and default_bin out of range (no skip), the forward branch
  //     (left grows from bin 0) finds the best split first → default_left=0,
  //     threshold = t+offset.
  {
    SCaseSpec s;
    s.name = "forward_winner";
    s.cfg = cfg(4, 1, 0.0, 0.0, 0.0, 0.0, 0, 4, false);
    s.hist = {-10.0, 5.0, -8.0, 5.0, 9.0, 5.0, 8.0, 5.0};
    s.sum_gradient = -10.0 - 8.0 + 9.0 + 8.0;
    s.sum_hessian = 20.0;
    s.num_data = 20;
    s.note = "low-neg-high-pos-forward-wins";
    cases.push_back(s);
  }
  // (b) REVERSE-winner: construct so the BEST gain is first attained by the
  //     reverse scan at a threshold recorded as t-1+offset (default_left=1). A
  //     symmetric-but-tie-broken layout: the reverse scan reaches the max gain at
  //     an earlier (higher-t) point. Use a layout whose unique best split is the
  //     boundary the reverse branch records first.
  {
    SCaseSpec s;
    s.name = "reverse_winner";
    s.cfg = cfg(4, 1, 0.0, 0.0, 0.0, 0.0, 0, 4, false);
    // Three low bins with mild gradient, one high bin with large positive grad,
    // so the best split isolates the top bin: reverse scan (right side growing
    // from the top) records it as t-1+offset before forward reaches it.
    s.hist = {1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 30.0, 5.0};
    s.sum_gradient = 1.0 + 1.0 + 1.0 + 30.0;
    s.sum_hessian = 20.0;
    s.num_data = 20;
    s.note = "top-bin-isolated-reverse-records-first";
    cases.push_back(s);
  }
  // (c) default-bin skip (D-02a): offset=1, default_bin=1 skipped during the scan.
  {
    SCaseSpec s;
    s.name = "default_bin_skip";
    s.cfg = cfg(5, 1, 0.0, 0.0, 0.0, 0.0, 1, 1, true);
    s.hist = {0.0, 0.0, -6.0, 4.0, -5.0, 4.0, 7.0, 4.0, 8.0, 4.0};
    s.sum_gradient = -6.0 - 5.0 + 7.0 + 8.0;
    s.sum_hessian = 16.0;
    s.num_data = 16;
    s.note = "offset1-defbin1-skipped";
    cases.push_back(s);
  }
  // (d) L1 regularization path (use_l1=true) so the L1 gain branch is exercised.
  {
    SCaseSpec s;
    s.name = "l1_forward";
    s.cfg = cfg(4, 1, 0.0, 0.5, 0.1, 0.0, 0, 4, false);
    s.hist = {-12.0, 6.0, -7.0, 6.0, 6.0, 6.0, 11.0, 6.0};
    s.sum_gradient = -12.0 - 7.0 + 6.0 + 11.0;
    s.sum_hessian = 24.0;
    s.num_data = 24;
    s.note = "l1-l2-regularized";
    cases.push_back(s);
  }
  // (e) no admissible split (min_data_in_leaf impossible) → is_splittable=0.
  {
    SCaseSpec s;
    s.name = "no_split";
    s.cfg = cfg(4, 1000000, 0.0, 0.0, 0.0, 0.0, 0, 4, false);
    s.hist = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    s.sum_gradient = 4.0;
    s.sum_hessian = 4.0;
    s.num_data = 8;
    s.note = "gates-reject-all";
    cases.push_back(s);
  }
  // (f) skip_default_bin==false DIVERGENCE case (Plan 05-01, RESEARCH Pitfall 1).
  //     A numeric feature with missing_type == None, num_bin > 2, and
  //     default_bin (2) < num_bin (4). The OLD Phase-4 heuristic
  //     `default_bin < num_bin` would set SKIP_DEFAULT_BIN = true and SKIP bin 2
  //     during the scan; the AUTHORITATIVE flag for missing_type == None is
  //     `skip_default_bin = false` (and na_as_missing = false), so bin 2 MUST
  //     participate in BOTH the forward and reverse accumulation.
  //
  //     g/h are hand-picked (brute-force verified) so the WINNING SplitInfo
  //     genuinely DIFFERS between the authoritative (skip=false) and the old
  //     heuristic (skip=true): the unique best split is the boundary
  //     {0,1,2} | {3} (threshold = 2). With skip_default_bin == FALSE the FORWARD
  //     branch reaches t = 2 (accumulating bin 2 into the left side) and records
  //     it FIRST among equal-gain candidates, so `default_left = 0`. With
  //     skip_default_bin == TRUE (the old heuristic) the forward branch SKIPS
  //     t == default_bin == 2 (a `continue`), so the SAME boundary is found only
  //     by the REVERSE branch (right = {3}, recorded as t-1+offset), yielding the
  //     opposite `default_left = 1`. Same threshold + gain, OPPOSITE default_left
  //     — a real, observable divergence the old heuristic would have gotten wrong.
  {
    SCaseSpec s;
    s.name = "skip_default_bin_false";
    // num_bin=4, offset=0, default_bin=2, skip=false, na=false (missing_type==None).
    s.cfg = cfg(4, 1, 0.0, 0.0, 0.0, 0.0, 0, 2, /*skip=*/false, /*na=*/false);
    // 5 rows per bin, hess=5 each (cnt_factor = 20/20 = 1). bins 0,1,2 negative
    // grad, bin 3 the least-negative — the best split isolates {0,1,2} from {3}
    // at threshold 2, which the forward branch reaches only by accumulating the
    // default bin (bin 2).
    s.hist = {-10.0, 5.0, -10.0, 5.0, -10.0, 5.0, 1.0, 5.0};
    s.sum_gradient = -10.0 - 10.0 - 10.0 + 1.0;  // = -29.0
    s.sum_hessian = 20.0;
    s.num_data = 20;
    s.note = "missing_type-None-defbin2-NOT-skipped-default_left-divergence";
    cases.push_back(s);
  }
  return cases;
}

// ===========================================================================
// 04-03 PARTITION golden. Each case fixes a binned column + min/max/threshold/
// most_freq_bin and emits the stable reordered index array + split_point (the
// SAME shape Backend::data_partition returns).
// ===========================================================================
struct PCaseSpec {
  std::string name;
  std::vector<uint32_t> bins;
  int num_bin;
  int min_bin;
  int max_bin;
  int threshold;
  int most_freq_bin;
  std::string note;
};

void EmitPCase(std::ofstream& out, const PCaseSpec& cs) {
  // WR-01: refuse to emit a golden the kernel can never reproduce. The Rust
  // data_partition_on returns ComputeError::Runtime when threshold >= num_bin
  // (partition.rs), but this generator routes regardless. Mirror the kernel
  // contract at capture time so an out-of-contract PCaseSpec fails loudly HERE
  // rather than as a panic!("data_partition failed") at replay.
  if (cs.threshold >= cs.num_bin) {
    std::cerr << "PCASE " << cs.name << ": threshold (" << cs.threshold
              << ") >= num_bin (" << cs.num_bin << ")\n";
    std::abort();
  }
  std::vector<int> route = SplitRoute(cs.bins, cs.min_bin, cs.max_bin, cs.threshold,
                                      cs.most_freq_bin);
  // Stable two-pass gather: left (route 0) then right (route 1), each in order.
  std::vector<uint32_t> reordered;
  reordered.reserve(cs.bins.size());
  for (size_t i = 0; i < route.size(); ++i)
    if (route[i] == 0) reordered.push_back(static_cast<uint32_t>(i));
  const int split_point = static_cast<int>(reordered.size());
  for (size_t i = 0; i < route.size(); ++i)
    if (route[i] != 0) reordered.push_back(static_cast<uint32_t>(i));

  out << "PCASE name=" << cs.name << " num_bin=" << cs.num_bin
      << " min_bin=" << cs.min_bin << " max_bin=" << cs.max_bin
      << " threshold=" << cs.threshold << " most_freq_bin=" << cs.most_freq_bin
      << " note=" << cs.note << "\n";
  out << "PBINS ";
  for (size_t i = 0; i < cs.bins.size(); ++i) {
    if (i) out << ";";
    out << cs.bins[i];
  }
  out << "\n";
  out << "PORDER ";
  for (size_t i = 0; i < reordered.size(); ++i) {
    if (i) out << ";";
    out << reordered[i];
  }
  out << "\n";
  out << "PSPLIT " << split_point << "\n";
}

std::vector<PCaseSpec> BuildPartitionCorpus(int master_seed) {
  std::vector<PCaseSpec> cases;
  CaseGen gen(master_seed ^ 0x5151);
  // (a) basic in-range threshold split.
  {
    PCaseSpec p;
    p.name = "basic"; p.bins = {1, 5, 3, 7, 0, 4, 2, 6};
    p.num_bin = 8; p.min_bin = 0; p.max_bin = 7; p.threshold = 3; p.most_freq_bin = 8;
    p.note = "in-range-thr3"; cases.push_back(p);
  }
  // (b) default routing: most_freq_bin <= threshold so out-of-range go left.
  {
    PCaseSpec p;
    p.name = "default_left"; p.bins = {0, 2, 9, 1, 5, 9, 3};
    p.num_bin = 10; p.min_bin = 1; p.max_bin = 6; p.threshold = 4; p.most_freq_bin = 2;
    p.note = "oor-go-left"; cases.push_back(p);
  }
  // (c) default routing right: most_freq_bin > threshold so out-of-range go right.
  {
    PCaseSpec p;
    p.name = "default_right"; p.bins = {0, 2, 9, 1, 5, 9, 3};
    p.num_bin = 10; p.min_bin = 1; p.max_bin = 6; p.threshold = 2; p.most_freq_bin = 7;
    p.note = "oor-go-right"; cases.push_back(p);
  }
  // (d) randomized larger case (stability stress).
  {
    PCaseSpec p;
    p.name = "random"; p.num_bin = 16; p.min_bin = 0; p.max_bin = 15;
    p.threshold = 7; p.most_freq_bin = 16;
    const int n = 40;
    p.bins.resize(n);
    for (int i = 0; i < n; ++i) p.bins[i] = static_cast<uint32_t>(gen.NextInt(0, 16));
    p.note = "random-40-rows"; cases.push_back(p);
  }
  return cases;
}

// ===========================================================================
// 04-03 SUBTRACT golden. parent - child over 2*num_bin f64 cells.
// ===========================================================================
struct SubCaseSpec {
  std::string name;
  std::vector<double> parent;
  std::vector<double> child;
  std::string note;
};

void EmitSubCase(std::ofstream& out, const SubCaseSpec& cs) {
  // FeatureHistogram::Subtract default path: derived[i] = parent[i] - child[i].
  std::vector<double> derived(cs.parent.size());
  for (size_t i = 0; i < cs.parent.size(); ++i) derived[i] = cs.parent[i] - cs.child[i];

  out << "SUBCASE name=" << cs.name << " len=" << cs.parent.size()
      << " note=" << cs.note << "\n";
  auto emit = [&](const char* tag, const std::vector<double>& v) {
    out << tag << " ";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) out << ";";
      out << F64Bits(v[i]);
    }
    out << "\n";
  };
  emit("SUBPARENT", cs.parent);
  emit("SUBCHILD", cs.child);
  emit("SUBDERIVED", derived);
}

std::vector<SubCaseSpec> BuildSubtractCorpus(int master_seed) {
  std::vector<SubCaseSpec> cases;
  CaseGen gen(master_seed ^ 0x2626);
  // (a) simple.
  {
    SubCaseSpec s;
    s.name = "simple"; s.parent = {10.0, 5.0, 8.0, 4.0}; s.child = {3.0, 2.0, 1.0, 1.0};
    s.note = "4-bin"; cases.push_back(s);
  }
  // (b) spread (mixed signs / magnitudes) over a larger histogram.
  {
    SubCaseSpec s;
    s.name = "spread"; const int nb = 12;
    s.parent.resize(2 * nb); s.child.resize(2 * nb);
    for (int i = 0; i < 2 * nb; ++i) {
      const double mag = (i % 4 == 0) ? 1e3 : (i % 4 == 1) ? 1e-3 : (i % 4 == 2) ? 7.5 : 0.125;
      const double sign = (gen.NextInt(0, 2) == 0) ? 1.0 : -1.0;
      s.parent[i] = sign * mag * (1.0 + gen.NextFloat());
      s.child[i] = sign * mag * 0.25 * (1.0 + gen.NextFloat());
    }
    s.note = "spread-12-bin"; cases.push_back(s);
  }
  return cases;
}

int main(int argc, char** argv) {
  // argv: <hist_out> <master_seed> <split_out> <partition_out> <subtract_out>
  if (argc != 6) {
    std::cerr << "usage: kernel_capture <hist_out> <master_seed> "
                 "<split_out> <partition_out> <subtract_out>\n";
    return 2;
  }
  const std::string out_path = argv[1];
  const int master_seed = std::stoi(argv[2]);
  const std::string split_path = argv[3];
  const std::string partition_path = argv[4];
  const std::string subtract_path = argv[5];

  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "error: cannot open output file: " << out_path << "\n";
    return 1;
  }

  std::vector<HCaseSpec> corpus = BuildHistCorpus(master_seed);

  out << "# LightGBM-rs compute-kernel histogram golden set (Phase 4, D-02/D-02a,\n";
  out << "# randomized-at-capture, D-14). Generated by xtask/cpp/kernel_capture.cpp —\n";
  out << "# a VERBATIM transcription of the pinned LightGBM dense_bin.hpp:130-141 /\n";
  out << "# sparse_bin.hpp:138-152 ConstructHistogram (external_libs unbuildable here;\n";
  out << "# see file header + REFERENCE_MANIFEST.md). Synthetic inputs use the real\n";
  out << "# header-only LightGBM::Random. hist_t=double accumulation of f32 grad/hess.\n";
  out << "KERNEL_MASTER_SEED " << master_seed << "\n";
  out << "COUNTS hist=" << corpus.size() << "\n";

  for (const auto& cs : corpus) EmitHCase(out, cs);

  out.flush();
  if (!out) {
    std::cerr << "error: failed while writing output\n";
    return 1;
  }
  std::cerr << "kernel_capture: wrote " << corpus.size() << " histogram cases to "
            << out_path << "\n";

  // ---- SPLIT golden ----
  std::ofstream sout(split_path, std::ios::binary | std::ios::trunc);
  if (!sout) {
    std::cerr << "error: cannot open output file: " << split_path << "\n";
    return 1;
  }
  std::vector<SCaseSpec> scorpus = BuildSplitCorpus();
  sout << "# LightGBM-rs find_best_split golden (Phase 4 04-03). VERBATIM\n";
  sout << "# FindBestThresholdSequentially + gain math from feature_histogram.hpp\n";
  sout << "# 711-1057 (commit 195c26fc). Per-candidate gains (REVERSE + FORWARD,\n";
  sout << "# NaN when gated) + the winning SplitInfo; covers reverse-winner,\n";
  sout << "# forward-winner, default-bin skip, L1, and no-split cases.\n";
  sout << "KERNEL_MASTER_SEED " << master_seed << "\n";
  sout << "COUNTS split=" << scorpus.size() << "\n";
  for (const auto& cs : scorpus) EmitSCase(sout, cs);
  sout.flush();
  if (!sout) { std::cerr << "error: failed writing split golden\n"; return 1; }
  std::cerr << "kernel_capture: wrote " << scorpus.size() << " split cases to "
            << split_path << "\n";

  // ---- PARTITION golden ----
  std::ofstream pout(partition_path, std::ios::binary | std::ios::trunc);
  if (!pout) {
    std::cerr << "error: cannot open output file: " << partition_path << "\n";
    return 1;
  }
  std::vector<PCaseSpec> pcorpus = BuildPartitionCorpus(master_seed);
  pout << "# LightGBM-rs data_partition golden (Phase 4 04-03). VERBATIM SplitInner\n";
  pout << "# routing (dense_bin.hpp:314-394, MissingType::None) + stable two-pass\n";
  pout << "# gather. Emits the reordered index array + split_point.\n";
  pout << "KERNEL_MASTER_SEED " << master_seed << "\n";
  pout << "COUNTS partition=" << pcorpus.size() << "\n";
  for (const auto& cs : pcorpus) EmitPCase(pout, cs);
  pout.flush();
  if (!pout) { std::cerr << "error: failed writing partition golden\n"; return 1; }
  std::cerr << "kernel_capture: wrote " << pcorpus.size() << " partition cases to "
            << partition_path << "\n";

  // ---- SUBTRACT golden ----
  std::ofstream subout(subtract_path, std::ios::binary | std::ios::trunc);
  if (!subout) {
    std::cerr << "error: cannot open output file: " << subtract_path << "\n";
    return 1;
  }
  std::vector<SubCaseSpec> subcorpus = BuildSubtractCorpus(master_seed);
  subout << "# LightGBM-rs subtract_histograms golden (Phase 4 04-03). VERBATIM\n";
  subout << "# FeatureHistogram::Subtract (feature_histogram.hpp:99-145, default\n";
  subout << "# USE_DIST_GRAD=false): derived[i] = parent[i] - child[i].\n";
  subout << "KERNEL_MASTER_SEED " << master_seed << "\n";
  subout << "COUNTS subtract=" << subcorpus.size() << "\n";
  for (const auto& cs : subcorpus) EmitSubCase(subout, cs);
  subout.flush();
  if (!subout) { std::cerr << "error: failed writing subtract golden\n"; return 1; }
  std::cerr << "kernel_capture: wrote " << subcorpus.size() << " subtract cases to "
            << subtract_path << "\n";

  return 0;
}
