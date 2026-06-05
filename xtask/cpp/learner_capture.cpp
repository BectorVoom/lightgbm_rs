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

    TreeSplit(&tree, best_leaf, feat, w.threshold, threshold_real, w.left_output,
              w.right_output, w.left_count, w.right_count, w.left_sum_hessian,
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

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: learner_capture <learner_out> <master_seed>\n";
    return 2;
  }
  const std::string out_path = argv[1];
  const int master_seed = std::stoi(argv[2]);
  // Touch the header-only reference Random so the include is exercised (the corpus
  // itself is fixed/hand-crafted, NOT RNG-derived — byte-idempotent).
  LightGBM::Random rng(master_seed);
  (void)rng.NextInt(0, 1);
  (void)F32Bits(0.0f);

  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "error: cannot open output file: " << out_path << "\n";
    return 1;
  }

  out << "# LightGBM-rs serial tree-learner golden set (Phase 5, Plan 05-03).\n";
  out << "# Generated by xtask/cpp/learner_capture.cpp — VERBATIM transcription of\n";
  out << "# the leaf-wise growth loop (serial_tree_learner.cpp) + FindBestThreshold\n";
  out << "# (feature_histogram.hpp, the D-02a kernel-capture counterpart) over a\n";
  out << "# FIXED synthetic g/h corpus pinned to missing_type==None (RESEARCH A5).\n";
  out << "# external_libs unbuildable here; see file header + REFERENCE_MANIFEST.md.\n";
  out << "LEARNER_MASTER_SEED " << master_seed << "\n";

  // Grow the tree into an in-memory body buffer (so the PSPLIT count is known
  // before the COUNTS header is written), then emit COUNTS + body + PTREE.
  Corpus corpus = BuildCorpus();
  std::ostringstream bodybuf;
  GrownTree tree = GrowTree(bodybuf, corpus);
  const std::string body_str = bodybuf.str();

  // Count PSPLIT records (each begins a line with "PSPLIT ").
  int split_count = 0;
  {
    size_t pos = 0;
    const std::string needle = "PSPLIT ";
    while ((pos = body_str.find(needle, pos)) != std::string::npos) {
      // Count only at a line start (pos == 0 or preceded by '\n').
      if (pos == 0 || body_str[pos - 1] == '\n') ++split_count;
      pos += needle.size();
    }
  }

  out << "COUNTS splits=" << split_count << " trees=1\n";
  out << body_str;
  EmitTree(out, corpus.name, tree);

  out.flush();
  if (!out) {
    std::cerr << "error: failed while writing output\n";
    return 1;
  }
  std::cerr << "learner_capture: wrote spine golden (" << split_count
            << " split records, 1 tree) to " << out_path << "\n";
  return 0;
}
