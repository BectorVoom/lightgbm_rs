// learner_capture.cpp
//
// Dev-only C++ capture harness for the LightGBM-rs SERIAL TREE LEARNER oracle
// (Phase 5). This is the Wave-1 (Plan 05-02) SCAFFOLD: it establishes the
// record format + capture pipeline and emits ONE trivial placeholder fixture so
// the `learner_parity.rs` replay harness has a committed target. The FULL learner
// transcription (leaf-wise loop, histogram subtraction trick, data partition,
// col_sampler) — emitting real per-split (D-06) and per-tree (D-07) goldens —
// lands in Plan 03/04 by extending this same file.
//
// WHY THIS IS A FOCUSED, SELF-CONTAINED TRANSCRIPTION (not a compile of the
// in-tree serial_tree_learner.cpp):
//   The authoritative `SerialTreeLearner` lives in
//   `LightGBM/src/treelearner/serial_tree_learner.cpp`, which (via
//   `<LightGBM/dataset.h>`/`<LightGBM/bin.h>` -> `common.h`) transitively pulls
//   in `fast_double_parser.h` + `fmt/format.h` from `external_libs/`. In this
//   repo those submodules are present only as EMPTY directories (the LightGBM
//   tree is git-untracked and its `external_libs/` are not vendored — see project
//   memory `lightgbm-ref-tree-untracked`). So `serial_tree_learner.cpp` cannot be
//   compiled here, and no `lib_lightgbm` can be built or linked.
//
//   This is the SAME situation Phases 1/2/4 hit for the RNG (rng_capture compiles
//   the header-only `Random` directly), binning (bin_capture verbatim-transcribes
//   FindBin/ValueToBin), and the compute kernels (kernel_capture verbatim-
//   transcribes ConstructHistogram + FindBestThresholdSequentially). Plan 03/04
//   will VERBATIM-transcribe the learner growth loop from the pinned
//   `serial_tree_learner.cpp` (commit 195c26fc, VERSION 4.6.0.99) — reusing the
//   already-transcribed gain/split math from `kernel_capture.cpp`'s
//   `FindBestThreshold` (D-02a cross-check, RESEARCH Pitfall 6) so the two
//   emitters stay bit-identical on shared inputs — and emit goldens byte-identical
//   to what lib_lightgbm would produce. Synthetic inputs use the genuine
//   header-only `LightGBM::Random`. No external_libs, no lib_lightgbm link, no
//   C++ toolchain at `cargo test` time (the golden is committed).
//
// Determinism / idempotency (D-14): the learner corpus is derived solely from one
// recorded LEARNER_MASTER_SEED passed on argv. No wall-clock / OS entropy, so
// re-running produces byte-identical fixtures (empty `git diff`).
//
// Fixture format (line-delimited, '#'-prefixed comments ignored by the reader) —
// TWO golden kinds:
//
//   LEARNER_MASTER_SEED <seed>
//   COUNTS splits=<n> trees=<n>
//
//   # per-split snapshot (D-06): the FULL per-bin gain array (one f64 per
//   # candidate bin, NaN where a candidate is gated) + the winning split's gain,
//   # so a divergence localizes to the gain scan, not just the winner.
//   PSPLIT split=<i> feature=<f> num_bin=<n> gains=<f64bits;...> winner=<f64bits>
//
//   # per-tree (D-07): a name line, then the grown tree's Tree::to_string() text
//   # block (the Phase-3 %.17g machinery is the arbiter), terminated by a line
//   # containing only ENDTREE.
//   PTREE name=<id>
//   <Tree::to_string() lines...>
//   ENDTREE
//
// f64 values are emitted as raw little-endian bit patterns (a u64 in decimal) so
// the Rust side asserts bit-exact equality with zero parsing rounding
// (compare_exact_f64_bits); the per-tree block is compared as a String.

#include <LightGBM/utils/random.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Raw-bit serialization helpers (copied from kernel_capture.cpp): emit f64 as a
// decimal u64 bit pattern so the Rust replay parses bit-exact (zero rounding).
// ---------------------------------------------------------------------------
static uint64_t F64Bits(double d) {
  uint64_t u;
  std::memcpy(&u, &d, sizeof(u));
  return u;
}

// (F32Bits is unused by the scaffold but kept for the Plan-03 per-split arrays
// that mirror kernel_capture's f32 ordered grad/hess inputs.)
static uint32_t F32Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

// A `;`-separated raw-bit f64 list writer (the PSPLIT `gains=` payload form).
static void EmitF64BitsList(std::ofstream& out, const std::vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out << ";";
    out << F64Bits(v[i]);
  }
}

int main(int argc, char** argv) {
  // argv: <learner_out> <master_seed>
  if (argc != 3) {
    std::cerr << "usage: learner_capture <learner_out> <master_seed>\n";
    return 2;
  }
  const std::string out_path = argv[1];
  const int master_seed = std::stoi(argv[2]);
  (void)master_seed;  // recorded; the scaffold corpus is fixed (Plan 03 uses it).

  // Touch the header-only reference Random so the link/include is exercised now
  // (Plan 03 drives the synthetic learner corpus off it). The draw is discarded.
  LightGBM::Random rng(master_seed);
  (void)rng.NextInt(0, 1);
  (void)F32Bits(0.0f);  // silence unused-fn warning until Plan 03 uses it.

  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    std::cerr << "error: cannot open output file: " << out_path << "\n";
    return 1;
  }

  out << "# LightGBM-rs serial tree-learner golden set (Phase 5) — SCAFFOLD.\n";
  out << "# Generated by xtask/cpp/learner_capture.cpp. The full per-split (D-06)\n";
  out << "# + per-tree (D-07) corpus (verbatim serial_tree_learner.cpp transcription,\n";
  out << "# external_libs unbuildable here — see file header + REFERENCE_MANIFEST.md)\n";
  out << "# lands in Plan 03/04. This placeholder exercises both record formats so\n";
  out << "# the learner_parity.rs harness has a committed target.\n";
  out << "LEARNER_MASTER_SEED " << master_seed << "\n";
  out << "COUNTS splits=1 trees=1\n";

  // --- one trivial PSPLIT placeholder (num_bin=2: a single 2-bin gain array) ---
  // Plan 03 replaces these with real FindBestThreshold per-bin gains; the scaffold
  // uses a finite winner so the harness's structural asserts pass.
  std::vector<double> gains = {0.0, 1.5};  // per-bin candidate gains
  out << "PSPLIT split=0 feature=0 num_bin=2 gains=";
  EmitF64BitsList(out, gains);
  out << " winner=" << F64Bits(1.5) << "\n";

  // --- one trivial PTREE placeholder: a 2-leaf Tree::to_string() block ---
  // This mirrors the byte-exact Phase-3 serialized form (the D-07 arbiter). Plan
  // 03/04 replaces it with the grown reference tree's text.
  out << "PTREE name=scaffold\n";
  out << "num_leaves=2\n";
  out << "num_cat=0\n";
  out << "split_feature=0\n";
  out << "split_gain=1.5\n";
  out << "threshold=0.5\n";
  out << "decision_type=2\n";
  out << "left_child=-1\n";
  out << "right_child=-2\n";
  out << "leaf_value=10 20\n";
  out << "leaf_weight=1 1\n";
  out << "leaf_count=1 1\n";
  out << "internal_value=0\n";
  out << "internal_weight=0\n";
  out << "internal_count=2\n";
  out << "is_linear=0\n";
  out << "shrinkage=1\n";
  out << "ENDTREE\n";

  out.flush();
  if (!out) {
    std::cerr << "error: failed while writing output\n";
    return 1;
  }
  std::cerr << "learner_capture: wrote scaffold golden (1 split, 1 tree) to "
            << out_path << "\n";
  return 0;
}
