// learner_capture.cpp
//
// Dev-only C++ capture harness for the LightGBM-rs SERIAL TREE LEARNER oracle
// (Phase 5). This is the FULL transcription (Plan 05-03): it verbatim-transcribes
// the leaf-wise growth loop of `SerialTreeLearner::Train` over a fixed synthetic
// gradient/hessian corpus and emits the committed `spine.txt` golden carrying
//   - per-split (D-06) full per-bin gain arrays for EVERY candidate feature, and
//   - the per-tree (D-07) grown-tree field set (raw bits, so the Rust side can
//     reconstruct the reference Tree and compare via the shared %.17g formatter).
//
// WHY A SELF-CONTAINED TRANSCRIPTION (not a compile of serial_tree_learner.cpp):
//   The authoritative `SerialTreeLearner` lives in
//   `LightGBM/src/treelearner/serial_tree_learner.cpp`, which (via
//   `<LightGBM/dataset.h>`/`<LightGBM/bin.h>` -> `common.h`) transitively pulls in
//   `fast_double_parser.h` + `fmt/format.h` from `external_libs/`. In this repo
//   those submodules are present only as EMPTY directories (the LightGBM tree is
//   git-untracked and its `external_libs/` are not vendored — see project memory
//   `lightgbm-ref-tree-untracked`). So `serial_tree_learner.cpp` cannot be
//   compiled and no `lib_lightgbm` can be built or linked.
//
//   This is the SAME situation Phases 1/2/4 hit (rng/bin/kernel capture). The
//   per-feature histogram + gain scan is transcribed from the pinned
//   `feature_histogram.hpp` (commit 195c26fc, VERSION 4.6.0.99) — the SAME
//   structure `kernel_capture.cpp` uses (D-02a cross-check, RESEARCH Pitfall 6) —
//   and the leaf-wise loop, smaller-child selection, FixHistogram, data partition,
//   and Tree::Split from the pinned `serial_tree_learner.cpp` / `dataset.cpp` /
//   `tree.h`. Synthetic g/h are pinned to `missing_type == None` to defer the
//   NA_AS_MISSING forward branch (RESEARCH A5). Only the header-only
//   `LightGBM::Random` is included (to silence the toolchain-presence check); the
//   corpus is fixed/hand-crafted, NOT RNG-derived, so it is byte-idempotent.
//
// Fixture format (line-delimited, '#'-prefixed comments ignored) — TWO kinds:
//
//   LEARNER_MASTER_SEED <seed>
//   COUNTS splits=<n> trees=<n>
//
//   # per-split snapshot (D-06): the FULL per-bin gain arrays (REVERSE + FORWARD,
//   # NaN where gated) for ONE candidate feature at one split decision.
//   PSPLIT split=<i> leaf=<l> feature=<f> num_bin=<n> \
//     rev=<f64bits;...> fwd=<f64bits;...> winner=<f64bits>
//
//   # per-tree (D-07): the grown tree's field set as raw little-endian bits so the
//   # Rust side reconstructs the reference Tree and serializes via its shared
//   # %.17g formatter, terminated by ENDTREE.
//   PTREE name=<id> num_leaves=<n>
//   PT_SPLIT_FEATURE <i...>
//   PT_THRESHOLD_BITS <u64...>
//   PT_DECISION_TYPE <i...>
//   PT_SPLIT_GAIN_BITS <u32...>
//   PT_LEFT_CHILD <i...>
//   PT_RIGHT_CHILD <i...>
//   PT_LEAF_VALUE_BITS <u64...>
//   PT_LEAF_WEIGHT_BITS <u64...>
//   PT_LEAF_COUNT <i...>
//   PT_INTERNAL_VALUE_BITS <u64...>
//   PT_INTERNAL_COUNT <i...>
//   ENDTREE
//
// f64/f32 values are emitted as raw little-endian bit patterns (u64/u32 decimal)
// so the Rust side parses bit-exact (zero rounding).

#include <LightGBM/utils/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Raw-bit serialization helpers (copied from kernel_capture.cpp).
// ---------------------------------------------------------------------------
static uint64_t F64Bits(double d) {
  uint64_t u;
  std::memcpy(&u, &d, sizeof(u));
  return u;
}
static uint32_t F32Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
static void EmitF64BitsList(std::ostream& out, const std::vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(v[i]);
  }
}

// ===========================================================================
// Gain math — VERBATIM from feature_histogram.hpp:711-845, identical in structure
// to kernel_capture.cpp (the D-02a cross-check counterpart). meta.h: kEpsilon =
// 1e-15f. RoundInt(x) = (int)(x + 0.5f) (common.h:904). Sign(x)=(x>0)-(x<0).
// ===========================================================================
static const float kEpsilonF = 1e-15f;
static const double kMinScore = -std::numeric_limits<double>::infinity();

static int CppSign(double x) { return (x > 0.0) - (x < 0.0); }
static int CppRoundInt(double x) { return static_cast<int>(x + 0.5f); }
static double ThresholdL1(double s, double l1) {
  const double reg_s = std::max(0.0, std::fabs(s) - l1);
  return CppSign(s) * reg_s;
}
static double GetLeafGain(bool use_l1, double g, double h, double l1, double l2) {
  if (use_l1) {
    const double sg = ThresholdL1(g, l1);
    return (sg * sg) / (h + l2);
  }
  return (g * g) / (h + l2);
}
static double GetSplitGains(bool use_l1, double lg, double lh, double rg, double rh,
                            double l1, double l2) {
  return GetLeafGain(use_l1, lg, lh, l1, l2) + GetLeafGain(use_l1, rg, rh, l1, l2);
}
static double CalcLeafOutput(bool use_l1, double g, double h, double l1, double l2) {
  if (use_l1) return -ThresholdL1(g, l1) / (h + l2);
  return -g / (h + l2);
}

struct SplitCfg {
  int min_data_in_leaf;
  double min_sum_hessian_in_leaf;
  double lambda_l1;
  double lambda_l2;
  double min_gain_to_split;
  int num_bin;
  int offset;
  int default_bin;
  bool skip_default_bin;  // num_bin>2 && missing_type==Zero
  bool na_as_missing;     // num_bin>2 && missing_type==NaN (deferred, always false here)
};

struct WinSplit {
  bool is_splittable = false;
  uint32_t threshold = 0;
  double gain = kMinScore;  // RAW best_gain (before -min_gain_shift)
  int left_count = 0;
  int right_count = 0;
  double left_sum_gradient = 0.0;
  double left_sum_hessian = 0.0;  // reported (kEpsilon already subtracted)
  double right_sum_gradient = 0.0;
  double right_sum_hessian = 0.0;
  double left_output = 0.0;
  double right_output = 0.0;
  bool default_left = true;
};

// VERBATIM FindBestThresholdSequentially (feature_histogram.hpp:830-1057) for
// <USE_RAND=false,USE_MC=false,USE_L1=?,USE_MAX_OUTPUT=false,USE_SMOOTHING=false,
//  NA_AS_MISSING=false>. `sum_hessian` here is ALREADY bumped by 2*kEpsilon.
// Records PER-CANDIDATE gains (NaN where gated) for BOTH branches (D-06).
static WinSplit FindBestThreshold(const std::vector<double>& hist, const SplitCfg& cfg,
                                  bool use_l1, double sum_gradient,
                                  double sum_hessian_bumped, int num_data,
                                  double min_gain_shift, std::vector<double>* cand_rev,
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
    int t = cfg.num_bin - 1 - offset;
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
        best_threshold = static_cast<uint32_t>(t - 1 + offset);
        best_gain = current_gain;
        best_default_left = true;
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
        best_threshold = static_cast<uint32_t>(t + offset);
        best_gain = current_gain;
        best_default_left = false;
      }
    }
  }

  WinSplit w;
  if (is_splittable && best_gain > kMinScore) {
    w.is_splittable = true;
    w.threshold = best_threshold;
    w.left_output = CalcLeafOutput(use_l1, best_sum_left_gradient, best_sum_left_hessian, l1, l2);
    w.left_count = best_left_count;
    w.left_sum_gradient = best_sum_left_gradient;
    w.left_sum_hessian = best_sum_left_hessian - kEpsilonF;
    double rsg = sum_gradient - best_sum_left_gradient;
    double rsh = sum_hessian_bumped - best_sum_left_hessian;
    w.right_output = CalcLeafOutput(use_l1, rsg, rsh, l1, l2);
    w.right_count = num_data - best_left_count;
    w.right_sum_gradient = rsg;
    w.right_sum_hessian = rsh - kEpsilonF;
    w.gain = best_gain;
    w.default_left = best_default_left;
  }
  return w;
}

// ===========================================================================
// The spine learner state (a faithful transcription subset of
// serial_tree_learner.{h,cpp} + tree.h for the pinned config).
// ===========================================================================

// One feature column (the spine input).
struct Feature {
  std::vector<uint32_t> bins;  // per-row bin index, length num_data
  int num_bin;
  int offset;
  int min_bin;
  int max_bin;
  int default_bin;
  int most_freq_bin;
  int missing_type;  // 0=None,1=Zero,2=NaN (pinned None on the spine)
  std::vector<double> bin_upper_bound;
  int real_feature_index;
};

static bool SkipDefaultBin(const Feature& f) {
  return f.num_bin > 2 && f.missing_type == 1;
}

// Grown tree (parallel arrays, mirroring lgbm-model::Tree growth state).
struct GrownTree {
  int num_leaves = 1;
  std::vector<int> split_feature;
  std::vector<double> threshold;
  std::vector<int> decision_type;  // int8 packed (default_left bit + missing<<2)
  std::vector<float> split_gain;
  std::vector<int> left_child;
  std::vector<int> right_child;
  std::vector<double> leaf_value;
  std::vector<double> leaf_weight;
  std::vector<int> leaf_count;
  std::vector<double> internal_value;
  std::vector<int> internal_count;
  std::vector<int> leaf_depth;
  std::vector<int> leaf_parent;
};

// Tree::Split (tree.h:543-585 + tree.cpp:61-75), numerical path.
static void TreeSplit(GrownTree* t, int leaf, int real_feature, uint32_t threshold_bin,
                      double threshold_real, double left_value, double right_value,
                      int left_count, int right_count, double left_weight,
                      double right_weight, float gain, int missing_type,
                      bool default_left) {
  const int new_node_idx = t->num_leaves - 1;
  const int new_leaf = t->num_leaves;
  const int parent = t->leaf_parent[leaf];
  if (parent >= 0) {
    if (t->right_child[parent] == ~leaf) t->right_child[parent] = new_node_idx;
    else t->left_child[parent] = new_node_idx;
  }
  t->split_feature.push_back(real_feature);
  t->split_gain.push_back(gain);
  t->threshold.push_back(threshold_real);
  t->internal_value.push_back(t->leaf_value[leaf]);
  t->internal_count.push_back(left_count + right_count);
  t->left_child.push_back(~leaf);
  t->right_child.push_back(~new_leaf);
  int dt = 0;
  if (default_left) dt |= 2;
  dt |= (missing_type & 3) << 2;
  t->decision_type.push_back(dt);
  t->leaf_value[leaf] = left_value;
  t->leaf_weight[leaf] = left_weight;
  t->leaf_count[leaf] = left_count;
  t->leaf_value.push_back(right_value);
  t->leaf_weight.push_back(right_weight);
  t->leaf_count.push_back(right_count);
  t->leaf_depth.push_back(t->leaf_depth[leaf] + 1);
  t->leaf_depth[leaf] += 1;
  t->leaf_parent[leaf] = new_node_idx;
  t->leaf_parent.push_back(new_node_idx);
  (void)threshold_bin;
  t->num_leaves += 1;
}

// Ordered f64 fold over a row-index list (LeafSplits::Init deterministic branch).
static void LeafSums(const std::vector<float>& g, const std::vector<float>& h,
                     const std::vector<uint32_t>& rows, double* sum_g, double* sum_h) {
  double sg = 0.0, sh = 0.0;
  for (uint32_t r : rows) {
    sg += static_cast<double>(g[r]);
    sh += static_cast<double>(h[r]);
  }
  *sum_g = sg;
  *sum_h = sh;
}

// FixHistogram (dataset.cpp:1488-1506), RAW leaf sums.
static void FixHistogram(std::vector<double>* hist, int most_freq_bin, double sum_g,
                         double sum_h) {
  if (most_freq_bin <= 0) return;
  const int num_bin = static_cast<int>(hist->size() / 2);
  double g = sum_g, hh = sum_h;
  for (int i = 0; i < num_bin; ++i) {
    if (i != most_freq_bin) {
      g -= (*hist)[(static_cast<size_t>(i) << 1)];
      hh -= (*hist)[(static_cast<size_t>(i) << 1) + 1];
    }
  }
  (*hist)[(static_cast<size_t>(most_freq_bin) << 1)] = g;
  (*hist)[(static_cast<size_t>(most_freq_bin) << 1) + 1] = hh;
}

// Direct ordered f64 histogram fold over a leaf's rows (dense_bin.hpp:130-141).
static std::vector<double> ConstructHistogram(const Feature& f, const std::vector<float>& g,
                                              const std::vector<float>& h,
                                              const std::vector<uint32_t>& rows) {
  std::vector<double> hist(static_cast<size_t>(2 * f.num_bin), 0.0);
  for (uint32_t r : rows) {
    const int bin = static_cast<int>(f.bins[r]);
    const size_t ti = static_cast<size_t>(bin) << 1;
    hist[ti] += static_cast<double>(g[r]);
    hist[ti + 1] += static_cast<double>(h[r]);
  }
  return hist;
}

// SplitInner data-partition route (dense_bin.hpp:314-394, MissingType::None).
static void PartitionLeaf(const Feature& f, std::vector<uint32_t>* leaf_rows,
                          int threshold, std::vector<uint32_t>* left,
                          std::vector<uint32_t>* right) {
  int th = threshold + f.min_bin;
  if (f.most_freq_bin == 0) --th;
  const bool default_to_right = !(f.most_freq_bin <= threshold);
  left->clear();
  right->clear();
  for (uint32_t r : *leaf_rows) {
    const int bin = static_cast<int>(f.bins[r]);
    const bool is_default = (bin < f.min_bin || bin > f.max_bin);
    const bool gt = bin > th;
    const bool go_right = is_default ? default_to_right : gt;
    if (go_right) right->push_back(r);
    else left->push_back(r);
  }
}

// SplitInfo + operator> tie-break (split_info.hpp:138-165): gain, then smaller
// feature (-1 -> INT32_MAX).
static bool SplitGt(const WinSplit& a, int af, double a_reported, const WinSplit& b,
                    int bf, double b_reported) {
  if (a_reported != b_reported) return a_reported > b_reported;
  int aff = (af == -1) ? std::numeric_limits<int>::max() : af;
  int bff = (bf == -1) ? std::numeric_limits<int>::max() : bf;
  return aff < bff;
}

struct LeafState {
  std::vector<uint32_t> rows;
  double sum_g = 0.0;
  double sum_h = 0.0;
  int depth = 0;
  int num_data_in_leaf() const { return static_cast<int>(rows.size()); }
};

// The full leaf-wise growth over the corpus features. Emits PSPLIT records to
// `out`. Returns the grown tree.
struct Corpus {
  std::string name;
  std::vector<Feature> features;
  std::vector<float> grad;
  std::vector<float> hess;
  SplitCfg base_cfg;  // gain params (num_bin/offset/default_bin set per-feature)
  int num_leaves;
  int max_depth;
};

static GrownTree GrowTree(std::ostream& out, const Corpus& c) {
  const int num_data = static_cast<int>(c.grad.size());
  const bool use_l1 = c.base_cfg.lambda_l1 > 0.0;

  // Root leaf sums.
  std::vector<uint32_t> all_rows(num_data);
  for (int i = 0; i < num_data; ++i) all_rows[i] = static_cast<uint32_t>(i);
  double root_g, root_h;
  LeafSums(c.grad, c.hess, all_rows, &root_g, &root_h);
  const double root_output =
      CalcLeafOutput(use_l1, root_g, root_h, c.base_cfg.lambda_l1, c.base_cfg.lambda_l2);

  GrownTree tree;
  tree.num_leaves = 1;
  tree.leaf_value = {root_output};
  tree.leaf_weight = {0.0};
  tree.leaf_count = {num_data};
  tree.leaf_depth = {0};
  tree.leaf_parent = {-1};

  // Per-leaf state.
  std::vector<LeafState> leaves;
  {
    LeafState root_state;
    root_state.rows = all_rows;
    root_state.sum_g = root_g;
    root_state.sum_h = root_h;
    root_state.depth = 0;
    leaves.push_back(root_state);
  }

  // Per-leaf best split + feature (flat vector + ArgMax).
  std::vector<WinSplit> best(c.num_leaves);
  std::vector<int> best_feature(c.num_leaves, -1);
  std::vector<double> best_reported(c.num_leaves, kMinScore);

  int split_record = 0;

  // Per-leaf buffered per-feature snapshot (cand_rev/cand_fwd/winner), so emission
  // can be deferred to the iteration where the Rust `find_best_splits` emits it
  // (root at iter0; a split's children at the NEXT iteration's start — the last
  // split's children are never emitted, exactly like Rust).
  struct FeatSnap {
    int feature;
    int num_bin;
    std::vector<double> rev;
    std::vector<double> fwd;
    double winner;
  };
  std::vector<std::vector<FeatSnap>> leaf_snap(c.num_leaves);

  // Scan a leaf for its best per-feature split, BUFFERING the per-bin gain snapshot
  // (D-06) into leaf_snap[leaf]. Emission is deferred (see EmitLeafSnap) so the
  // emit ORDER mirrors the Rust `find_best_splits` structure exactly.
  auto find_best_for_leaf = [&](int leaf) {
    LeafState& ls = leaves[leaf];
    best[leaf] = WinSplit{};
    best_feature[leaf] = -1;
    best_reported[leaf] = kMinScore;
    leaf_snap[leaf].clear();
    if (!(ls.sum_h > 0.0) || ls.rows.empty()) return;

    for (const Feature& f : c.features) {
      std::vector<double> hist = ConstructHistogram(f, c.grad, c.hess, ls.rows);
      FixHistogram(&hist, f.most_freq_bin, ls.sum_g, ls.sum_h);

      SplitCfg cfg = c.base_cfg;
      cfg.num_bin = f.num_bin;
      cfg.offset = f.offset;
      cfg.default_bin = f.default_bin;
      cfg.skip_default_bin = SkipDefaultBin(f);
      cfg.na_as_missing = false;

      const double gain_shift =
          GetLeafGain(use_l1, ls.sum_g, ls.sum_h, cfg.lambda_l1, cfg.lambda_l2);
      const double min_gain_shift = gain_shift + cfg.min_gain_to_split;
      const double sum_hessian_bumped = ls.sum_h + 2.0 * static_cast<double>(kEpsilonF);

      std::vector<double> cand_rev, cand_fwd;
      WinSplit w = FindBestThreshold(hist, cfg, use_l1, ls.sum_g, sum_hessian_bumped,
                                     ls.num_data_in_leaf(), min_gain_shift, &cand_rev,
                                     &cand_fwd);
      const double reported = w.is_splittable ? (w.gain - min_gain_shift) : kMinScore;

      // Buffer the per-feature snapshot for deferred emission.
      FeatSnap fs;
      fs.feature = f.real_feature_index;
      fs.num_bin = f.num_bin;
      fs.rev = cand_rev;
      fs.fwd = cand_fwd;
      fs.winner = reported;
      leaf_snap[leaf].push_back(fs);

      if (w.is_splittable && reported > kMinScore &&
          SplitGt(w, f.real_feature_index, reported, best[leaf], best_feature[leaf],
                  best_reported[leaf])) {
        best[leaf] = w;
        best[leaf].gain = reported;  // store the REPORTED gain for selection
        best_feature[leaf] = f.real_feature_index;
        best_reported[leaf] = reported;
      }
    }
  };

  // Emit one leaf's buffered per-feature snapshot (the deferred D-06 emission).
  auto emit_leaf_snap = [&](int leaf) {
    for (const FeatSnap& fs : leaf_snap[leaf]) {
      out << "PSPLIT split=" << split_record << " leaf=" << leaf
          << " feature=" << fs.feature << " num_bin=" << fs.num_bin << " rev=";
      EmitF64BitsList(out, fs.rev);
      out << " fwd=";
      EmitF64BitsList(out, fs.fwd);
      out << " winner=" << F64Bits(fs.winner) << "\n";
      ++split_record;
    }
  };

  // Seed leaf 0's best — this IS the root decision in Rust's find_best_splits, so
  // it EMITS first (the smaller leaf == the only leaf, no larger).
  find_best_for_leaf(0);
  emit_leaf_snap(0);

  for (int split = 0; split < c.num_leaves - 1; ++split) {
    // ArgMax over best_reported.
    int best_leaf = 0;
    for (int i = 1; i < static_cast<int>(leaves.size()); ++i) {
      if (SplitGt(best[i], best_feature[i], best_reported[i], best[best_leaf],
                  best_feature[best_leaf], best_reported[best_leaf])) {
        best_leaf = i;
      }
    }
    if (best_reported[best_leaf] <= 0.0) break;

    // max_depth gate.
    if (c.max_depth > 0 && leaves[best_leaf].depth >= c.max_depth) {
      best_reported[best_leaf] = kMinScore;
      best[best_leaf].gain = kMinScore;
      // re-pick on the next iteration (no eligible leaf -> loop ends).
      bool any = false;
      for (size_t i = 0; i < leaves.size(); ++i)
        if (best_reported[i] > 0.0) any = true;
      if (!any) break;
      --split;
      continue;
    }

    const WinSplit w = best[best_leaf];
    const int feat = best_feature[best_leaf];
    const Feature& f = *std::find_if(c.features.begin(), c.features.end(),
                                     [&](const Feature& x) {
                                       return x.real_feature_index == feat;
                                     });

    // Partition the leaf.
    std::vector<uint32_t> left_rows, right_rows;
    PartitionLeaf(f, &leaves[best_leaf].rows, static_cast<int>(w.threshold), &left_rows,
                  &right_rows);

    const int new_left = best_leaf;
    const int new_right = tree.num_leaves;

    const double threshold_real =
        (static_cast<size_t>(w.threshold) < f.bin_upper_bound.size())
            ? f.bin_upper_bound[w.threshold]
            : static_cast<double>(w.threshold);
    const float split_gain_field = static_cast<float>(w.gain + c.base_cfg.min_gain_to_split);

    // ACTUAL partition counts (serial_tree_learner.cpp:788-791, update_cnt=true):
    // the tree's leaf_count records data_partition_->leaf_count, NOT the SplitInfo
    // reconstructed counts (which can disagree by ±1 for fractional hessians).
    const int actual_left_count = static_cast<int>(left_rows.size());
    const int actual_right_count = static_cast<int>(right_rows.size());
    TreeSplit(&tree, best_leaf, feat, w.threshold, threshold_real, w.left_output,
              w.right_output, actual_left_count, actual_right_count, w.left_sum_hessian,
              w.right_sum_hessian, split_gain_field, f.missing_type, w.default_left);

    // Update leaf states: new_left keeps best_leaf slot, append new_right.
    double lg, lh, rg, rh;
    LeafSums(c.grad, c.hess, left_rows, &lg, &lh);
    LeafSums(c.grad, c.hess, right_rows, &rg, &rh);
    leaves[new_left].rows = left_rows;
    leaves[new_left].sum_g = lg;
    leaves[new_left].sum_h = lh;
    leaves[new_left].depth = tree.leaf_depth[new_left];
    LeafState rstate;
    rstate.rows = right_rows;
    rstate.sum_g = rg;
    rstate.sum_h = rh;
    rstate.depth = tree.leaf_depth[new_right];
    leaves.push_back(rstate);

    // Recompute best for the two children (always, so the NEXT ArgMax sees them).
    find_best_for_leaf(new_left);
    find_best_for_leaf(new_right);

    // Emit the children's per-bin snapshots in smaller-then-larger order (by
    // partition row count), mirroring the Rust `find_best_splits` emit order — but
    // ONLY when another split iteration remains (Rust emits a split's children at
    // the START of the next iteration, so the LAST split's children are never
    // emitted). `split` is 0-based; the final iteration is `num_leaves - 2`.
    if (split < c.num_leaves - 2) {
      const int cnt_left = static_cast<int>(left_rows.size());
      const int cnt_right = static_cast<int>(right_rows.size());
      if (cnt_left < cnt_right) {
        emit_leaf_snap(new_left);   // smaller = left
        emit_leaf_snap(new_right);  // larger = right
      } else {
        emit_leaf_snap(new_right);  // smaller = right
        emit_leaf_snap(new_left);   // larger = left
      }
    }
  }

  return tree;
}

// ===========================================================================
// Corpus construction — fixed synthetic g/h exercising every spine path, pinned
// missing_type == None (A5). Hand-crafted (NOT RNG), byte-idempotent.
// ===========================================================================
static Corpus BuildCorpus() {
  Corpus c;
  c.name = "spine_a";
  // 12 rows; gradient sign/magnitude spread so multiple splits have positive gain.
  c.grad = {-6.0f, -6.0f, -5.0f, -5.0f, -1.0f, -1.0f,
            1.0f,  1.0f,  5.0f,  5.0f,  6.0f,  6.0f};
  c.hess = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  // Feature 0: 6 bins, 2 rows per bin (monotone gradient -> clean splits).
  Feature f0;
  f0.bins = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  f0.num_bin = 6;
  f0.offset = 0;
  f0.min_bin = 0;
  f0.max_bin = 5;
  f0.default_bin = 6;  // out of range
  f0.most_freq_bin = 0;
  f0.missing_type = 0;  // None
  f0.bin_upper_bound = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
  f0.real_feature_index = 0;
  c.features.push_back(f0);

  // Feature 1: 4 bins, a different partition of the same rows (a competing
  // feature so the cross-feature argmax + tie-break is exercised).
  Feature f1;
  f1.bins = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 2, 3};
  f1.num_bin = 4;
  f1.offset = 0;
  f1.min_bin = 0;
  f1.max_bin = 3;
  f1.default_bin = 4;
  f1.most_freq_bin = 0;
  f1.missing_type = 0;
  f1.bin_upper_bound = {0.5, 1.5, 2.5, 3.5};
  f1.real_feature_index = 1;
  c.features.push_back(f1);

  c.base_cfg.min_data_in_leaf = 1;
  c.base_cfg.min_sum_hessian_in_leaf = 0.0;
  c.base_cfg.lambda_l1 = 0.0;
  c.base_cfg.lambda_l2 = 0.0;
  c.base_cfg.min_gain_to_split = 0.0;
  c.base_cfg.num_bin = 6;
  c.base_cfg.offset = 0;
  c.base_cfg.default_bin = 6;
  c.base_cfg.skip_default_bin = false;
  c.base_cfg.na_as_missing = false;
  c.num_leaves = 4;
  c.max_depth = -1;
  return c;
}

template <typename T>
static void EmitIntList(std::ostream& out, const char* tag, const std::vector<T>& v) {
  out << tag;
  for (size_t i = 0; i < v.size(); ++i) out << " " << v[i];
  out << "\n";
}
static void EmitF64BitsLine(std::ostream& out, const char* tag,
                            const std::vector<double>& v) {
  out << tag;
  for (size_t i = 0; i < v.size(); ++i) out << " " << F64Bits(v[i]);
  out << "\n";
}
static void EmitF32BitsLine(std::ostream& out, const char* tag,
                            const std::vector<float>& v) {
  out << tag;
  for (size_t i = 0; i < v.size(); ++i) out << " " << F32Bits(v[i]);
  out << "\n";
}

static void EmitTree(std::ostream& out, const std::string& name, const GrownTree& t) {
  out << "PTREE name=" << name << " num_leaves=" << t.num_leaves << "\n";
  EmitIntList(out, "PT_SPLIT_FEATURE", t.split_feature);
  EmitF64BitsLine(out, "PT_THRESHOLD_BITS", t.threshold);
  EmitIntList(out, "PT_DECISION_TYPE", t.decision_type);
  EmitF32BitsLine(out, "PT_SPLIT_GAIN_BITS", t.split_gain);
  EmitIntList(out, "PT_LEFT_CHILD", t.left_child);
  EmitIntList(out, "PT_RIGHT_CHILD", t.right_child);
  EmitF64BitsLine(out, "PT_LEAF_VALUE_BITS", t.leaf_value);
  EmitF64BitsLine(out, "PT_LEAF_WEIGHT_BITS", t.leaf_weight);
  EmitIntList(out, "PT_LEAF_COUNT", t.leaf_count);
  EmitF64BitsLine(out, "PT_INTERNAL_VALUE_BITS", t.internal_value);
  EmitIntList(out, "PT_INTERNAL_COUNT", t.internal_count);
  out << "ENDTREE\n";
}

// ===========================================================================
// Plan 05-04 additions: force_col_wise (TRL-09), feature subsampling (TRL-08),
// and the captured real-g/h corpus (D-03).
// ===========================================================================

// --- ColSampler (col_sampler.hpp:22-179) VERBATIM transcription for the pinned
// no-interaction-constraint, single-group spine. `Random` is the header-only
// reference LCG. `valid_feature_indices_ = 0..num_features` (every feature valid);
// `InnerFeatureIndex(real) == real`. The PARITY CONTRACT is the Sample call order.
struct CppColSampler {
  double fraction_bytree;
  double fraction_bynode;
  bool need_reset_bytree;
  int used_cnt_bytree;
  std::vector<int> valid_feature_indices;
  std::vector<int> used_feature_indices;
  std::vector<int8_t> is_feature_used;
  LightGBM::Random random;

  static int GetCnt(size_t total_cnt, double fraction) {
    const int min = std::min(1, static_cast<int>(total_cnt));
    // Common::RoundInt(x) = (int)(x + 0.5f).
    const int used = static_cast<int>(
        static_cast<double>(static_cast<double>(total_cnt) * fraction) + 0.5f);
    return std::max(used, min);
  }

  CppColSampler(double ff, double ffn, int seed, std::vector<int> valid, int num_features)
      : fraction_bytree(ff),
        fraction_bynode(ffn),
        valid_feature_indices(std::move(valid)),
        is_feature_used(num_features, 1),
        random(seed) {
    if (fraction_bytree >= 1.0) {
      need_reset_bytree = false;
      used_cnt_bytree = static_cast<int>(valid_feature_indices.size());
    } else {
      need_reset_bytree = true;
      used_cnt_bytree = GetCnt(valid_feature_indices.size(), fraction_bytree);
    }
    ResetByTree();
  }

  void ResetByTree() {
    if (!need_reset_bytree) return;
    std::memset(is_feature_used.data(), 0, sizeof(int8_t) * is_feature_used.size());
    used_feature_indices = random.Sample(
        static_cast<int>(valid_feature_indices.size()), used_cnt_bytree);
    for (size_t i = 0; i < used_feature_indices.size(); ++i) {
      int used_feature = valid_feature_indices[used_feature_indices[i]];
      is_feature_used[used_feature] = 1;  // InnerFeatureIndex == identity
    }
  }

  std::vector<int8_t> GetByNode() {
    const int num_features = static_cast<int>(is_feature_used.size());
    if (fraction_bynode >= 1.0) {
      return std::vector<int8_t>(num_features, 1);
    }
    std::vector<int8_t> ret(num_features, 0);
    if (need_reset_bytree) {
      int used_feature_cnt = GetCnt(used_feature_indices.size(), fraction_bynode);
      auto sampled = random.Sample(static_cast<int>(used_feature_indices.size()),
                                   used_feature_cnt);
      for (size_t i = 0; i < sampled.size(); ++i) {
        int used_feature = valid_feature_indices[used_feature_indices[sampled[i]]];
        ret[used_feature] = 1;
      }
    } else {
      int used_feature_cnt = GetCnt(valid_feature_indices.size(), fraction_bynode);
      auto sampled = random.Sample(static_cast<int>(valid_feature_indices.size()),
                                   used_feature_cnt);
      for (size_t i = 0; i < sampled.size(); ++i) {
        int used_feature = valid_feature_indices[sampled[i]];
        ret[used_feature] = 1;
      }
    }
    return ret;
  }

  std::vector<int> SelectedBytree() const {
    std::vector<int> out;
    for (int f : valid_feature_indices) {
      if (is_feature_used[f]) out.push_back(f);
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  // col_sampler.hpp:181-183 — the per-tree used flag for one inner feature.
  bool is_feature_used_bytree(int inner_feature) const {
    return static_cast<size_t>(inner_feature) < is_feature_used.size() &&
           is_feature_used[inner_feature] != 0;
  }
};

static std::vector<int> MaskToIndices(const std::vector<int8_t>& mask,
                                      const std::vector<Feature>& features) {
  std::vector<int> out;
  for (const Feature& f : features) {
    if (static_cast<size_t>(f.real_feature_index) < mask.size() &&
        mask[f.real_feature_index]) {
      out.push_back(f.real_feature_index);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Emit a `;`-separated int list (empty -> empty).
static void EmitIntListSemi(std::ostream& out, const std::vector<int>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out << ";";
    out << v[i];
  }
}

// Combine the per-tree bytree flag with one per-node GetByNode mask, exactly as
// the Rust learner's `find_best_splits` does:
//   is_feature_used[f] = is_feature_used_bytree(f) && node_selected(f).
// Returns the EFFECTIVE per-(real-feature) used mask for the scan.
static std::vector<int8_t> CombineMask(const CppColSampler& cs,
                                       const std::vector<int8_t>& node,
                                       const std::vector<Feature>& features) {
  // The effective mask is indexed by real_feature_index (same id space as the
  // raw masks); size it to cover every feature column.
  size_t n = node.size();
  std::vector<int8_t> eff(n, 0);
  for (const Feature& f : features) {
    const int fi = f.real_feature_index;
    const bool used = cs.is_feature_used_bytree(fi) &&
                      (static_cast<size_t>(fi) < n && node[fi] != 0);
    if (static_cast<size_t>(fi) < n) eff[fi] = used ? 1 : 0;
  }
  return eff;
}

// Emit the TRL-08 feature-subsampling RNG-parity golden. This grows the tree with
// the SAME col-sampler gate the Rust learner applies (per-node GetByNode draws,
// smaller-leaf-then-larger, with the scan gated to the selected features), so the
// grown SHAPE — and therefore the draw COUNT and ORDER — is bit-identical to the
// Rust `train_with_col_sampler_trace` trace. Emits:
//   CS_CONFIG ...
//   CS_BYTREE <selected real feature indices, ResetByTree>
//   CS_NODE order=<i> <selected real feature indices for that GetByNode draw>
// The CS_NODE records appear in DRAW ORDER (root first, then smaller-then-larger
// per split decision), exactly the order Rust pushes onto `bynode_selected`.
static void EmitColSamplerGolden(std::ostream& out, const Corpus& c, double ff,
                                 double ffn, int seed) {
  const int num_features = static_cast<int>(c.features.size());
  std::vector<int> valid(num_features);
  for (int i = 0; i < num_features; ++i) valid[i] = c.features[i].real_feature_index;
  int max_real = -1;
  for (int v : valid) max_real = std::max(max_real, v);
  const int nf = std::max(max_real + 1, num_features);

  CppColSampler cs(ff, ffn, seed, valid, nf);

  out << "CS_CONFIG feature_fraction_bits=" << F64Bits(ff)
      << " feature_fraction_bynode_bits=" << F64Bits(ffn) << " seed=" << seed
      << " num_features=" << num_features << "\n";
  out << "CS_BYTREE ";
  EmitIntListSemi(out, cs.SelectedBytree());
  out << "\n";

  const int num_data = static_cast<int>(c.grad.size());
  std::vector<uint32_t> all_rows(num_data);
  for (int i = 0; i < num_data; ++i) all_rows[i] = static_cast<uint32_t>(i);

  const bool use_l1 = c.base_cfg.lambda_l1 > 0.0;
  double root_g, root_h;
  LeafSums(c.grad, c.hess, all_rows, &root_g, &root_h);

  GrownTree tree;
  tree.num_leaves = 1;
  tree.leaf_value = {CalcLeafOutput(use_l1, root_g, root_h, c.base_cfg.lambda_l1,
                                    c.base_cfg.lambda_l2)};
  tree.leaf_weight = {0.0};
  tree.leaf_count = {num_data};
  tree.leaf_depth = {0};
  tree.leaf_parent = {-1};

  std::vector<LeafState> leaves;
  {
    LeafState rs;
    rs.rows = all_rows;
    rs.sum_g = root_g;
    rs.sum_h = root_h;
    rs.depth = 0;
    leaves.push_back(rs);
  }
  std::vector<WinSplit> best(c.num_leaves);
  std::vector<int> best_feature(c.num_leaves, -1);
  std::vector<double> best_reported(c.num_leaves, kMinScore);

  // Find the best split for `leaf`, scanning ONLY the features the effective mask
  // selects (the col-sampler gate). `mask` is indexed by real_feature_index; an
  // empty mask means "all features" (the root before any draw never happens — the
  // root is always preceded by exactly one GetByNode draw, mirroring Rust).
  auto find_best_for_leaf = [&](int leaf, const std::vector<int8_t>& mask) {
    LeafState& ls = leaves[leaf];
    best[leaf] = WinSplit{};
    best_feature[leaf] = -1;
    best_reported[leaf] = kMinScore;
    if (!(ls.sum_h > 0.0) || ls.rows.empty()) return;
    for (const Feature& f : c.features) {
      const int fi = f.real_feature_index;
      if (!mask.empty()) {
        const bool used = (static_cast<size_t>(fi) < mask.size()) && mask[fi] != 0;
        if (!used) continue;  // gated out this node
      }
      std::vector<double> hist = ConstructHistogram(f, c.grad, c.hess, ls.rows);
      FixHistogram(&hist, f.most_freq_bin, ls.sum_g, ls.sum_h);
      SplitCfg cfg = c.base_cfg;
      cfg.num_bin = f.num_bin;
      cfg.offset = f.offset;
      cfg.default_bin = f.default_bin;
      cfg.skip_default_bin = SkipDefaultBin(f);
      cfg.na_as_missing = false;
      const double gain_shift =
          GetLeafGain(use_l1, ls.sum_g, ls.sum_h, cfg.lambda_l1, cfg.lambda_l2);
      const double min_gain_shift = gain_shift + cfg.min_gain_to_split;
      const double sum_hessian_bumped = ls.sum_h + 2.0 * static_cast<double>(kEpsilonF);
      std::vector<double> cr, cf;
      WinSplit w = FindBestThreshold(hist, cfg, use_l1, ls.sum_g, sum_hessian_bumped,
                                     ls.num_data_in_leaf(), min_gain_shift, &cr, &cf);
      const double reported = w.is_splittable ? (w.gain - min_gain_shift) : kMinScore;
      if (w.is_splittable && reported > kMinScore &&
          SplitGt(w, fi, reported, best[leaf], best_feature[leaf], best_reported[leaf])) {
        best[leaf] = w;
        best[leaf].gain = reported;
        best_feature[leaf] = fi;
        best_reported[leaf] = reported;
      }
    }
  };

  // ---- the leaf-wise loop, mirroring `train_inner` (serial_tree_learner.cpp) ----
  // left_leaf / right_leaf track the children produced by the last split; the
  // first iteration's left_leaf is the root (leaf 0), right_leaf = -1.
  int draw_order = 0;
  int left_leaf = 0;
  int right_leaf = -1;
  for (int split = 0; split < c.num_leaves - 1; ++split) {
    // BeforeFindBestSplit gates (both-children-too-small / max_depth). On this
    // corpus the gates never fire, but transcribe them so the structure matches.
    const int num_left = static_cast<int>(leaves[left_leaf].rows.size());
    bool did = true;
    if (c.max_depth > 0 && tree.leaf_depth[left_leaf] >= c.max_depth) {
      did = false;
    } else {
      const int min2 = c.base_cfg.min_data_in_leaf * 2;
      if (right_leaf >= 0) {
        const int num_right = static_cast<int>(leaves[right_leaf].rows.size());
        if (num_right < min2 && num_left < min2) did = false;
      } else if (num_left < min2) {
        did = false;
      }
    }

    if (did) {
      // Smaller-child selection (Pitfall 3): on the root smaller = left, no larger.
      int smaller_leaf, larger_leaf;
      bool has_larger;
      if (right_leaf < 0) {
        smaller_leaf = left_leaf;
        larger_leaf = -1;
        has_larger = false;
      } else {
        const int nl = static_cast<int>(leaves[left_leaf].rows.size());
        const int nr = static_cast<int>(leaves[right_leaf].rows.size());
        if (nl < nr) {
          smaller_leaf = left_leaf;
          larger_leaf = right_leaf;
        } else {
          smaller_leaf = right_leaf;
          larger_leaf = left_leaf;
        }
        has_larger = true;
      }

      // GetByNode draw ORDER: SMALLER first (always), LARGER second (only when a
      // larger child exists). Each call advances the shared PRNG once. Record the
      // RAW selection (MaskToIndices) — the Rust trace pushes the same.
      std::vector<int8_t> smaller_raw = cs.GetByNode();
      out << "CS_NODE order=" << draw_order++ << " ";
      EmitIntListSemi(out, MaskToIndices(smaller_raw, c.features));
      out << "\n";
      std::vector<int8_t> larger_raw;
      if (has_larger) {
        larger_raw = cs.GetByNode();
        out << "CS_NODE order=" << draw_order++ << " ";
        EmitIntListSemi(out, MaskToIndices(larger_raw, c.features));
        out << "\n";
      }

      // Scan each leaf gated to its effective mask (bytree && node-selected).
      const std::vector<int8_t> smaller_eff = CombineMask(cs, smaller_raw, c.features);
      find_best_for_leaf(smaller_leaf, smaller_eff);
      if (has_larger) {
        const std::vector<int8_t> larger_eff = CombineMask(cs, larger_raw, c.features);
        find_best_for_leaf(larger_leaf, larger_eff);
      }
    }

    // ArgMax over best_reported (first-max via SplitGt).
    int best_leaf = 0;
    for (int i = 1; i < static_cast<int>(leaves.size()); ++i) {
      if (SplitGt(best[i], best_feature[i], best_reported[i], best[best_leaf],
                  best_feature[best_leaf], best_reported[best_leaf])) {
        best_leaf = i;
      }
    }
    if (best_reported[best_leaf] <= 0.0) break;  // no positive-gain split

    const WinSplit w = best[best_leaf];
    const int feat = best_feature[best_leaf];
    const Feature& f = *std::find_if(
        c.features.begin(), c.features.end(),
        [&](const Feature& x) { return x.real_feature_index == feat; });
    std::vector<uint32_t> left_rows, right_rows;
    PartitionLeaf(f, &leaves[best_leaf].rows, static_cast<int>(w.threshold),
                  &left_rows, &right_rows);
    const int new_left = best_leaf;
    const int new_right = tree.num_leaves;
    const double threshold_real =
        (static_cast<size_t>(w.threshold) < f.bin_upper_bound.size())
            ? f.bin_upper_bound[w.threshold]
            : static_cast<double>(w.threshold);
    const float gain_field = static_cast<float>(w.gain + c.base_cfg.min_gain_to_split);
    TreeSplit(&tree, best_leaf, feat, w.threshold, threshold_real, w.left_output,
              w.right_output, static_cast<int>(left_rows.size()),
              static_cast<int>(right_rows.size()), w.left_sum_hessian,
              w.right_sum_hessian, gain_field, f.missing_type, w.default_left);

    double lg, lh, rg, rh;
    LeafSums(c.grad, c.hess, left_rows, &lg, &lh);
    LeafSums(c.grad, c.hess, right_rows, &rg, &rh);
    leaves[new_left].rows = left_rows;
    leaves[new_left].sum_g = lg;
    leaves[new_left].sum_h = lh;
    leaves[new_left].depth = tree.leaf_depth[new_left];
    LeafState rstate;
    rstate.rows = right_rows;
    rstate.sum_g = rg;
    rstate.sum_h = rh;
    rstate.depth = tree.leaf_depth[new_right];
    leaves.push_back(rstate);

    // Reset the just-split leaf + the two children to "no split" until the next
    // iteration's find_best recomputes them (mirrors Rust). The children are
    // rescanned at the START of the next iteration via the GetByNode draws above.
    best[best_leaf] = WinSplit{};
    best_feature[best_leaf] = -1;
    best_reported[best_leaf] = kMinScore;
    best[new_left] = WinSplit{};
    best_feature[new_left] = -1;
    best_reported[new_left] = kMinScore;
    if (new_right < static_cast<int>(best.size())) {
      best[new_right] = WinSplit{};
      best_feature[new_right] = -1;
      best_reported[new_right] = kMinScore;
    }

    left_leaf = new_left;
    right_leaf = new_right;
  }
}

// --- Iteration-1 g/h transcription (D-03). regression-l2: grad=(float)(0-label),
// hess=1; binary-logloss: response=-label*sigmoid/(1+exp(0)) with label in {-1,+1},
// sigmoid=1, boost_from_average=false (score=0). score_t = float.
static void GhRegressionL2(const std::vector<float>& labels, std::vector<float>* g,
                           std::vector<float>* h) {
  g->resize(labels.size());
  h->resize(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    const double score = 0.0;  // boost_from_average=false
    (*g)[i] = static_cast<float>(score - static_cast<double>(labels[i]));
    (*h)[i] = 1.0f;
  }
}
static void GhBinaryLogloss(const std::vector<float>& labels, std::vector<float>* g,
                            std::vector<float>* h) {
  const double sigmoid = 1.0;
  g->resize(labels.size());
  h->resize(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    const int is_pos = labels[i] > 0 ? 1 : 0;
    const int label = is_pos ? 1 : -1;            // label_val_ = {-1, +1}
    const double label_weight = 1.0;              // label_weights_ default 1
    const double score = 0.0;                     // boost_from_average=false
    const double response = -label * sigmoid / (1.0f + std::exp(label * sigmoid * score));
    const double abs_response = std::fabs(response);
    (*g)[i] = static_cast<float>(response * label_weight);
    (*h)[i] = static_cast<float>(abs_response * (sigmoid - abs_response) * label_weight);
  }
}

// A 4-feature corpus for the TRL-08 feature-subsampling golden. Four numeric
// features each cleanly splittable (so the per-node feature gate genuinely
// changes which feature wins), 16 rows, num_leaves=4. missing_type==None (A5).
// Hand-crafted (NOT RNG), byte-idempotent. The g/h have a sign/magnitude spread
// so several features carry positive gain and the gated argmax is exercised.
static Corpus BuildColSamplerCorpus() {
  Corpus c;
  c.name = "col_sampler_a";
  // 16 rows. Gradient increases with row index (monotone) so a clean separating
  // split exists; hessian = 1 everywhere.
  c.grad = {-8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
            1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f};
  c.hess = std::vector<float>(16, 1.0f);

  // Feature 0: 4 bins, monotone with the gradient (the strongest splitter).
  auto make_feat = [](std::vector<uint32_t> bins, int num_bin, int real_idx) {
    Feature f;
    f.bins = std::move(bins);
    f.num_bin = num_bin;
    f.offset = 0;
    f.min_bin = 0;
    f.max_bin = num_bin - 1;
    f.default_bin = num_bin;  // out of range
    f.most_freq_bin = 0;
    f.missing_type = 0;  // None
    f.bin_upper_bound.clear();
    for (int b = 0; b < num_bin; ++b) f.bin_upper_bound.push_back(b + 0.5);
    f.real_feature_index = real_idx;
    return f;
  };
  // f0: bin = row/4 (monotone, 4 bins) — aligned with the gradient.
  c.features.push_back(make_feat(
      {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, 4, 0));
  // f1: bin = row/8 mapped to 2 bins then refined — a coarser monotone split.
  c.features.push_back(make_feat(
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, 2, 1));
  // f2: an alternating-then-monotone partition (a weaker competitor).
  c.features.push_back(make_feat(
      {0, 1, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 3, 2, 3, 2}, 4, 2));
  // f3: a 3-bin monotone-ish split (another competitor).
  c.features.push_back(make_feat(
      {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}, 3, 3));

  c.base_cfg.min_data_in_leaf = 1;
  c.base_cfg.min_sum_hessian_in_leaf = 0.0;
  c.base_cfg.lambda_l1 = 0.0;
  c.base_cfg.lambda_l2 = 0.0;
  c.base_cfg.min_gain_to_split = 0.0;
  c.base_cfg.num_bin = 4;
  c.base_cfg.offset = 0;
  c.base_cfg.default_bin = 4;
  c.base_cfg.skip_default_bin = false;
  c.base_cfg.na_as_missing = false;
  c.num_leaves = 4;
  c.max_depth = -1;
  return c;
}

// Build the D-03 captured-real-g/h corpus: a fixed binned design matrix over a
// fixed set of REAL labels, with iteration-1 g/h computed by the genuine
// objective formula (`GhRegressionL2` / `GhBinaryLogloss`, boost_from_average=
// false). The realistic distribution comes from the objective math on the real
// labels; the binned features are deterministic (no NaN-missing — A5). `objective`
// is 0=regression-l2, 1=binary-logloss.
static Corpus BuildRealGhCorpus(int objective) {
  Corpus c;
  // 20 rows. The features are MONOTONE partitions of the row index (each bin is a
  // contiguous block of rows, like the spine's feature 0), so a threshold cleanly
  // bisects the rows and the gain-scan threshold and the data partition agree
  // EXACTLY (no degenerate / 0-row child). The realistic distribution comes from
  // the OBJECTIVE math on real labels (grad spread), not from a contrived binning.
  //
  // Labels: a realistic real-valued / {0,1} spread, BUT ordered so the iteration-1
  // gradient (grad = score - label = -label) separates the low-row block (negative
  // grad after negation of positive labels) from the high-row block — giving a
  // clean monotone split structure on feature 0, then feature 1 within a child.
  std::vector<float> reg_labels = {
      2.6f,  2.1f,  1.9f,  1.5f,  1.2f,  0.8f,  0.5f,  0.2f,  0.1f,  -0.1f,
      -0.3f, -0.6f, -0.9f, -1.2f, -1.4f, -1.7f, -2.0f, -2.3f, -2.7f, -3.1f};
  // Binary {0,1}: the per-bin positive/negative RATIO decreases monotonically
  // across feature 0's four 5-row bins (bin0: 5 pos, bin1: 4 pos/1 neg, bin2:
  // 1 pos/4 neg, bin3: 5 neg), so the iteration-1 gradient (label=1 -> grad=-0.5,
  // label=0 -> grad=+0.5) is monotone increasing per bin — feature 0 then splits
  // cleanly at TWO distinct thresholds into non-degenerate (>=5-row) children.
  std::vector<float> bin_labels = {
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // bin0: 5 positive
      1.0f, 1.0f, 1.0f, 1.0f, 0.0f,  // bin1: 4 pos / 1 neg
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // bin2: 1 pos / 4 neg
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // bin3: 5 negative

  if (objective == 0) {
    c.name = "real_gh_regression";
    GhRegressionL2(reg_labels, &c.grad, &c.hess);
  } else {
    c.name = "real_gh_binary";
    GhBinaryLogloss(bin_labels, &c.grad, &c.hess);
  }
  const int num_data = 20;

  // Feature 0: 4 bins, 5 rows per bin (monotone in the row index). The strongest
  // splitter — a threshold bisects the rows into contiguous blocks.
  Feature f0;
  f0.bins = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3};
  f0.num_bin = 4;
  f0.offset = 0;
  f0.min_bin = 0;
  f0.max_bin = 3;
  f0.default_bin = 4;
  f0.most_freq_bin = 0;
  f0.missing_type = 0;
  f0.bin_upper_bound = {0.5, 1.5, 2.5, 3.5};
  f0.real_feature_index = 0;
  c.features.push_back(f0);

  // Feature 1: a WEAK, near-constant competitor — its bins are spread so that
  // each bin carries a balanced mix of +/- gradient (low separating gain), so
  // feature 0 (the monotone splitter aligned with the gradient) wins BOTH splits
  // cleanly. This keeps a second feature in the corpus (exercising the cross-
  // feature argmax + tie-break) without introducing a degenerate split.
  Feature f1;
  f1.bins = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  f1.num_bin = 2;
  f1.offset = 0;
  f1.min_bin = 0;
  f1.max_bin = 1;
  f1.default_bin = 2;
  f1.most_freq_bin = 0;
  f1.missing_type = 0;
  f1.bin_upper_bound = {0.5, 1.5};
  f1.real_feature_index = 1;
  c.features.push_back(f1);

  (void)num_data;
  // min_data_in_leaf=3 (a realistic non-trivial floor) keeps every split's
  // ACTUAL children non-degenerate (>=3 rows each), so the actual partition counts
  // the faithful tree records (serial_tree_learner.cpp:790-791) never collapse a
  // child to 0 rows. min_sum_hessian_in_leaf relaxed so the binary objective's
  // 0.25 hessians (5.0 total over 20 rows) do not gate splits.
  c.base_cfg.min_data_in_leaf = 3;
  c.base_cfg.min_sum_hessian_in_leaf = 0.0;
  c.base_cfg.lambda_l1 = 0.0;
  c.base_cfg.lambda_l2 = 0.0;
  c.base_cfg.min_gain_to_split = 0.0;
  c.base_cfg.num_bin = 4;
  c.base_cfg.offset = 0;
  c.base_cfg.default_bin = 4;
  c.base_cfg.skip_default_bin = false;
  c.base_cfg.na_as_missing = false;
  // num_leaves: regression-l2 (hessian==1, reconstructed count == actual count)
  // grows a clean 3-leaf tree (two splits). binary-logloss has fractional 0.25
  // hessians, where the gain-scan's `round_int(hess*cnt_factor)` reconstructed
  // count can over-count vs the threshold partition on the RESIDUAL leaf of a
  // second split — isolating a degenerate (0-row) child. A single clean split
  // (num_leaves=2) on the strongest feature avoids that corner while still proving
  // D-07 full-tree parity on realistic binary-logloss g/h. (The regression case
  // carries the deeper multi-split tree.)
  c.num_leaves = (objective == 0) ? 3 : 2;
  c.max_depth = -1;
  return c;
}

// Count PSPLIT records (each begins a line with "PSPLIT ") in a body buffer.
static int CountPSplit(const std::string& body_str) {
  int split_count = 0;
  size_t pos = 0;
  const std::string needle = "PSPLIT ";
  while ((pos = body_str.find(needle, pos)) != std::string::npos) {
    if (pos == 0 || body_str[pos - 1] == '\n') ++split_count;
    pos += needle.size();
  }
  return split_count;
}

// Write one spine-style golden (LEARNER_MASTER_SEED + COUNTS + PSPLIT body +
// PTREE) for a corpus grown WITHOUT feature subsampling. Used for `spine.txt`,
// `col_wise.txt`, and the per-objective `real_gh` tree blocks.
static int WriteSpineStyleGolden(std::ostream& out, const Corpus& corpus,
                                 int master_seed, const char* header_note,
                                 bool emit_seed_header) {
  out << "# " << header_note << "\n";
  if (emit_seed_header) out << "LEARNER_MASTER_SEED " << master_seed << "\n";
  std::ostringstream bodybuf;
  GrownTree tree = GrowTree(bodybuf, corpus);
  const std::string body_str = bodybuf.str();
  out << "COUNTS splits=" << CountPSplit(body_str) << " trees=1\n";
  out << body_str;
  EmitTree(out, corpus.name, tree);
  return CountPSplit(body_str);
}

// Emit the captured g/h vector (raw f32 bits) for the real_gh golden.
static void EmitGhVector(std::ostream& out, const char* tag, const std::vector<float>& v) {
  out << tag;
  for (size_t i = 0; i < v.size(); ++i) out << " " << F32Bits(v[i]);
  out << "\n";
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: learner_capture <spine_out> <col_wise_out> "
                 "<col_sampler_out> <real_gh_out> <master_seed>\n";
    return 2;
  }
  const std::string spine_path = argv[1];
  const std::string col_wise_path = argv[2];
  const std::string col_sampler_path = argv[3];
  const std::string real_gh_path = argv[4];
  const int master_seed = std::stoi(argv[5]);
  // Touch the header-only reference Random so the include is exercised. (The
  // spine/col_wise/real_gh corpora are fixed/hand-crafted, NOT RNG-derived; the
  // col_sampler golden DOES drive the genuine reference Random::Sample — that is
  // its whole point — but from a recorded seed, so all four are byte-idempotent.)
  LightGBM::Random rng(master_seed);
  (void)rng.NextInt(0, 1);
  (void)F32Bits(0.0f);

  // ---- (1) spine.txt — the Plan-05-03 spine golden (unchanged corpus) ----
  {
    std::ofstream out(spine_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "error: cannot open output file: " << spine_path << "\n";
      return 1;
    }
    out << "# LightGBM-rs serial tree-learner golden set (Phase 5, Plan 05-03).\n";
    out << "# Generated by xtask/cpp/learner_capture.cpp — VERBATIM transcription of\n";
    out << "# the leaf-wise growth loop (serial_tree_learner.cpp) + FindBestThreshold\n";
    out << "# (feature_histogram.hpp, the D-02a kernel-capture counterpart) over a\n";
    out << "# FIXED synthetic g/h corpus pinned to missing_type==None (RESEARCH A5).\n";
    out << "# external_libs unbuildable here; see file header + REFERENCE_MANIFEST.md.\n";
    Corpus corpus = BuildCorpus();
    int split_count = WriteSpineStyleGolden(
        out, corpus, master_seed,
        "spine corpus (12 rows / 2 features, force_row_wise, feature_fraction=1.0)",
        true);
    out.flush();
    if (!out) {
      std::cerr << "error: failed while writing " << spine_path << "\n";
      return 1;
    }
    std::cerr << "learner_capture: wrote spine golden (" << split_count
              << " split records, 1 tree) to " << spine_path << "\n";
  }

  // ---- (2) col_wise.txt — the SAME spine corpus grown under force_col_wise ----
  // The transcription is strategy-agnostic (the per-bin gain math + leaf-wise loop
  // do not depend on histogram-build ORDER), so the grown tree is identical to
  // spine.txt's. The Rust side grows the corpus under BOTH RowWise and ColWise and
  // asserts they are equal to each other AND to this golden (TRL-09, Pitfall 5).
  {
    std::ofstream out(col_wise_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "error: cannot open output file: " << col_wise_path << "\n";
      return 1;
    }
    out << "# LightGBM-rs tree-learner force_col_wise golden (Phase 5, Plan 05-04,\n";
    out << "# TRL-09). The col-major histogram build differs from row-major ONLY in\n";
    out << "# accumulation ORDER, not result (Pitfall 5); on the single-thread anchor\n";
    out << "# the grown tree is bit-identical to spine.txt. The Rust replay asserts\n";
    out << "# force_row_wise == force_col_wise == this golden (String equality).\n";
    Corpus corpus = BuildCorpus();
    corpus.name = "col_wise_a";
    int split_count = WriteSpineStyleGolden(
        out, corpus, master_seed,
        "spine corpus grown under force_col_wise (== force_row_wise tree)", true);
    out.flush();
    if (!out) {
      std::cerr << "error: failed while writing " << col_wise_path << "\n";
      return 1;
    }
    std::cerr << "learner_capture: wrote col_wise golden (" << split_count
              << " split records, 1 tree) to " << col_wise_path << "\n";
  }

  // ---- (3) col_sampler.txt — the TRL-08 feature-subsampling RNG-parity golden ----
  // feature_fraction=1.0 (no per-tree reset draw), feature_fraction_bynode=0.5
  // (per-node GetByNode draws), feature_fraction_seed = master_seed. The genuine
  // reference Random::Sample drives the selection; the Rust ColSampler reproduces
  // the EXACT draw sequence (T-05-04-01). The grown shape is col-sampler-gated, so
  // the draw COUNT/ORDER matches the Rust learner's trace.
  {
    std::ofstream out(col_sampler_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "error: cannot open output file: " << col_sampler_path << "\n";
      return 1;
    }
    out << "# LightGBM-rs ColSampler RNG-parity golden (Phase 5, Plan 05-04, TRL-08).\n";
    out << "# Drives the GENUINE header-only reference Random::Sample (col_sampler.hpp\n";
    out << "# transcription): ResetByTree once (CS_BYTREE), GetByNode per node in DRAW\n";
    out << "# ORDER (CS_NODE, smaller-leaf then larger-leaf per split). The Rust\n";
    out << "# ColSampler reproduces the exact selected feature indices.\n";
    out << "LEARNER_MASTER_SEED " << master_seed << "\n";
    const double ff = 1.0;       // feature_fraction (no per-tree reset)
    const double ffn = 0.5;      // feature_fraction_bynode (per-node draws)
    const int ff_seed = master_seed;
    Corpus corpus = BuildColSamplerCorpus();
    EmitColSamplerGolden(out, corpus, ff, ffn, ff_seed);
    out.flush();
    if (!out) {
      std::cerr << "error: failed while writing " << col_sampler_path << "\n";
      return 1;
    }
    std::cerr << "learner_capture: wrote col_sampler golden to " << col_sampler_path
              << "\n";
  }

  // ---- (4) real_gh.txt — the D-03 captured iteration-1 g/h corpus ----
  // Two objectives (regression-l2, binary-logloss), boost_from_average=false, g/h
  // from the genuine objective math on fixed real labels (realistic distribution).
  // Each block emits the captured g/h (raw f32 bits) + the grown reference tree
  // (PSPLIT + PTREE), grown WITHOUT subsampling (feature_fraction=1.0).
  {
    std::ofstream out(real_gh_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      std::cerr << "error: cannot open output file: " << real_gh_path << "\n";
      return 1;
    }
    out << "# LightGBM-rs captured real iteration-1 g/h golden (Phase 5, Plan 05-04,\n";
    out << "# D-03). regression-l2 (grad=score-label, hess=1) + binary-logloss\n";
    out << "# (response=-label*sigmoid/(1+exp(label*sigmoid*score))) iteration-1 g/h\n";
    out << "# with boost_from_average=false (score=0), score_t=float, over fixed real\n";
    out << "# labels (realistic distribution). The learner grows the SAME tree as C++\n";
    out << "# on these g/h (D-07 full tree, missing_type==None — A5).\n";
    out << "LEARNER_MASTER_SEED " << master_seed << "\n";
    for (int objective = 0; objective <= 1; ++objective) {
      Corpus corpus = BuildRealGhCorpus(objective);
      out << "GH_CORPUS name=" << corpus.name << " objective="
          << (objective == 0 ? "regression_l2" : "binary_logloss")
          << " num_data=" << corpus.grad.size()
          << " num_features=" << corpus.features.size()
          << " num_leaves=" << corpus.num_leaves << "\n";
      EmitGhVector(out, "GH_GRAD", corpus.grad);
      EmitGhVector(out, "GH_HESS", corpus.hess);
      // Per-feature bin layout so the Rust replay reconstructs the identical
      // FeatureColumn inputs (no re-binning).
      for (const Feature& f : corpus.features) {
        out << "GH_FEATURE real=" << f.real_feature_index << " num_bin=" << f.num_bin
            << " most_freq_bin=" << f.most_freq_bin << " default_bin=" << f.default_bin
            << " min_bin=" << f.min_bin << " max_bin=" << f.max_bin << " bins=";
        for (size_t i = 0; i < f.bins.size(); ++i) {
          if (i) out << ";";
          out << f.bins[i];
        }
        out << " upper=";
        for (size_t i = 0; i < f.bin_upper_bound.size(); ++i) {
          if (i) out << ";";
          out << F64Bits(f.bin_upper_bound[i]);
        }
        out << "\n";
      }
      int split_count = WriteSpineStyleGolden(
          out, corpus, master_seed,
          (objective == 0 ? "real_gh regression-l2 iter-1 reference tree"
                          : "real_gh binary-logloss iter-1 reference tree"),
          false);
      std::cerr << "learner_capture: wrote real_gh " << corpus.name << " ("
                << split_count << " split records, 1 tree)\n";
    }
    out.flush();
    if (!out) {
      std::cerr << "error: failed while writing " << real_gh_path << "\n";
      return 1;
    }
    std::cerr << "learner_capture: wrote real_gh golden to " << real_gh_path << "\n";
  }

  return 0;
}
