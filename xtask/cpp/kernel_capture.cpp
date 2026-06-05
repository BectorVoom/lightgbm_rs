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
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
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
      cur_pos += deltas_[i_delta];
      if (i_delta >= num_vals_) return 0;
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

int main(int argc, char** argv) {
  // argv: <hist_out> <master_seed>
  if (argc != 3) {
    std::cerr << "usage: kernel_capture <hist_out> <master_seed>\n";
    return 2;
  }
  const std::string out_path = argv[1];
  const int master_seed = std::stoi(argv[2]);

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
  return 0;
}
