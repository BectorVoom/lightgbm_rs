# Phase 5: Tree Learner + Split Finding - Research

**Researched:** 2026-06-06
**Domain:** Histogram-based serial decision-tree learner (leaf-wise growth, split finding, subtraction trick, data partition, feature subsampling) — pure-Rust port of C++ LightGBM `SerialTreeLearner`
**Confidence:** HIGH (the authoritative C++ reference is on disk and was read directly; all algorithmic claims are `[CITED]` from `LightGBM/src/treelearner/*` at the pinned VERSION 4.6.0.99 / commit 195c26fc)

## Summary

Phase 5 is the **orchestration** layer that turns the Phase-4 `Backend` kernels (`construct_histograms` → `find_best_split` → `subtract_histograms` → `data_partition`) into a leaf-wise (best-first) tree that is structurally and numerically bit-faithful to C++ `SerialTreeLearner::Train`. The per-feature histogram accumulation and per-bin gain scan **already exist** in `crates/lgbm-compute` (Phase 4, D-01a); this phase owns the loop above them: the leaf-wise best-split priority selection (`ArrayArgs::ArgMax` over `best_split_per_leaf_`), the `BeforeFindBestSplit` smaller/larger-child selection that drives the subtraction trick, the histogram pool bookkeeping, `DataPartition` row→leaf routing, `LeafSplits` sum seeding, `ColSampler` RNG-driven feature subsampling, and the `Tree` mutation (`Tree::Split`) that grows the model.

The whole algorithm is single-threaded-deterministic in the pinned reference config (`deterministic=true force_row_wise=true num_threads=1`), so every reduction is an ordered fold — there is no parallel non-determinism to reproduce, only **exact arithmetic order, exact gate order, and exact f32/f64 type boundaries**. The keystone risks are: (1) the smaller-child selection rule (`num_data_in_left_child < num_data_in_right_child`) that decides which child is constructed vs subtracted; (2) the `FixHistogram` most-freq-bin reconstruction that runs **before** every scan; (3) the `SKIP_DEFAULT_BIN`/`NA_AS_MISSING` dispatch (the Phase-4 `cfg_skip_default_bin` heuristic is a **known divergence** flagged for this phase); (4) the `SplitInfo::operator>` tie-break (gain, then smaller feature index); (5) the `kEpsilon`/`2·kEpsilon` placements; and (6) the `ColSampler` `Random::Sample` call sequence.

**Primary recommendation:** Build a new `lgbm-treelearner` crate that hand-ports `SerialTreeLearner` 1:1 on top of the existing `Backend` trait, sequencing **spine-first** (D-04): prove `force_row_wise` + `feature_fraction=1.0` numeric leaf-wise growth at per-split parity, then layer `force_col_wise` (TRL-09) and feature-subsampling RNG (TRL-08). Capture per-split golden snapshots (full per-bin gain arrays per candidate feature, D-06) and full-tree structural goldens (D-07) by **extending the `xtask` header-only transcription harness** with a `learner-capture` subcommand that re-transcribes the entire learner (D-01/D-02). **Resolve the Phase-4 `cfg_skip_default_bin` heuristic** by threading the authoritative `missing_type`-derived `skip_default_bin`/`na_as_missing` flags through the kernel call — this is in-scope work this phase inherits.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Leaf-wise best-first growth loop (`Train`) | `lgbm-treelearner` (new) | — | Orchestration; calls Backend ops, owns the leaf queue |
| Cross-feature argmax + tie-break (`FindBestSplitsFromHistograms`) | `lgbm-treelearner` | `lgbm-compute` (per-feature `find_best_split`) | Learner aggregates per-feature `SplitInfo`s; the per-bin gain math is Phase-4 kernel |
| Per-bin gain scan (per-feature `find_best_split`) | `lgbm-compute` (exists) | — | D-01a: gain math lives in the kernel; learner consumes it |
| Histogram construction | `lgbm-compute` (exists) | — | Phase-4 `construct_histograms` whole-kernel op |
| Histogram subtraction (math) | `lgbm-compute` (exists) | `lgbm-treelearner` (WHICH child) | Subtract MATH is the kernel; smaller-child SELECTION is learner orchestration |
| `FixHistogram` (most-freq-bin reconstruct) | `lgbm-treelearner` | `lgbm-dataset` (bin_mapper meta) | Runs in the learner before each scan; not currently a Backend op (see Open Q1) |
| Data partition (row→leaf bookkeeping) | `lgbm-treelearner` (`DataPartition`) | `lgbm-compute` (`data_partition` op) | Kernel returns reordered indices + split point; learner owns `leaf_begin_`/`leaf_count_`/`indices_` |
| Leaf-split sum seeding (`LeafSplits`) | `lgbm-treelearner` | — | Ordered fold over leaf rows; deterministic branch |
| Feature subsampling RNG (`ColSampler`) | `lgbm-treelearner` | `lgbm-core::Random` | Reproduces the C++ `Random::Sample` draw sequence + call order |
| Histogram pool sizing/eviction | `lgbm-treelearner` (`HistogramPool`) | — | D-05: mirror the full pool, not just FP-load-bearing parts |
| Tree mutation (`Tree::Split`) | `lgbm-model` (extend) | `lgbm-treelearner` | The grown `Tree` lives in `lgbm-model`; Phase 5 adds a `split` mutation method (Phase 3 was load/predict/write only) |
| Per-split + per-tree golden capture | `xtask` (extend) | `oracle-harness` | D-01/D-02 transcription harness + the comparators |

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — Reference-tree capture via the header-only C++ transcription harness.** Extend the P1–P4 `xtask` transcription harness to produce the reference tree + per-split snapshots. `external_libs` are unvendored and `LightGBM/` is untracked, so the full C++ `serial_tree_learner` cannot be linked at test time; the capture harness transcribes the learner verbatim, emits goldens over fixed g/h + binned inputs, and commits them. Replayable with no C++ toolchain at normal test time.
- **D-02 — Full end-to-end re-transcription of `serial_tree_learner`** in the golden-emitter (NOT orchestration-only). The emitter transcribes the *whole* learner including the per-feature histogram accumulation + gain scan, **independently** of the Phase-4 kernel transcriptions.
  - **D-02a (watch):** Two independent transcriptions of the per-feature histogram/gain math now exist (P4 kernel-capture + P5 learner-capture). They MUST agree bit-for-bit where they overlap. Plan a guard that surfaces drift rather than letting them diverge silently.
- **D-03 — Both g/h sources, layered:** (1) synthetic deterministic g/h hand-crafted to exercise every split path (sign/magnitude spread, ties, missing/zero routing, default-bin skip, subtraction-trick edge cases); (2) captured real first-iteration g/h from a real C++ objective's iteration-1 on a real dataset. The captured-g/h objective(s)/dataset, `boost_from_average` on/off, and exact capture config are **Claude's discretion**, bounded by the faithful-mirror contract.
- **D-04 — Spine first, then parity additions.** Lock and per-split-validate the minimal faithful tree first — `force_row_wise` + default `feature_fraction=1.0` (no subsampling) + numeric splits + leaf-wise growth + subtraction trick + data partition (TRL-01, 02, 03, 04, 05, 07). THEN add `force_col_wise` (TRL-09) and per-node feature-subsampling RNG parity (TRL-08) as validated additions on top of the proven spine.
- **D-05 — Mirror the full C++ histogram pool + eviction** faithfully (sizing/eviction/reuse machinery alongside smaller-child selection, which child is constructed vs subtracted, parent-histogram retention, subtract math/order). Not just the FP-load-bearing parts.
- **D-06 — Per-split snapshot = full per-bin gain array for every candidate feature** at each split decision (the entire bin-by-bin gain scan for every feature at every split — localizes divergence to a specific (feature, bin), not just "wrong winner").
- **D-07 — Tree-match unit = full tree structure, bit-faithful.** Equality of the entire grown tree: every internal node's split feature, threshold/bin, and missing/default direction, AND every leaf's output value (`CalculateSplittedLeafOutput` with `lambda_l1`/`lambda_l2`/`path_smooth`/`max_delta_step`), compared via the **Phase-3 model-text `%.17g`** machinery. Leaf-output parity is in-scope here.

### Carried Forward (locked by prior phases — not re-litigated)

- **Faithful C++ mirror** discipline (P1 D-11/D-12, P2 D-01, P3 D-04, P4 D-04/D-01): reproduce *which* child is constructed vs subtracted, the default-bin skip, `kEpsilon`/`2·kEpsilon` placement, and tie-break order — never an idiomatic redesign.
- **f32 end-to-end, ~1e-6 absolute, standard f32 accumulations** into f64 histogram cells (`hist_t = double`); integer-quantized histograms dropped (P1 D-02/D-03).
- **Single-threaded deterministic core** matching the pinned `deterministic=true force_row_wise=true num_threads=1` reference (P2 D-03); per-row/per-feature independence is the parallel/GPU seam, not exercised this phase.
- **The gain math lives in the Phase-4 kernel** (P4 D-01/D-01a): the runtime learner consumes `find_best_split`, it does NOT re-derive per-bin gains in the runtime path. (The golden-emitter re-transcribes per D-02.)
- **`lgbm-compute` is the single CubeCL seam** (P1 D-09, CMP-01): the Phase-5 learner depends on the `Backend` trait, never on a `cubecl` runtime type.
- **Committed goldens + idempotent C++-regen + header-only/verbatim transcription** capture when `external_libs` unvendored; `LightGBM/` is untracked — never `git add` it; goldens committed into `tests/fixtures/`.
- **CPU is the bit-exact hard gate; ROCm is a separate ~1e-6 gate** (P4 D-03/D-04): the cubecl-cpu path is the deterministic anchor.

### Claude's Discretion

- The tree-learner crate placement/structure (new `lgbm-treelearner` crate vs existing crate) and the learner↔`Backend` wiring; the leaf-wise priority-queue data structure and the leaf-split bookkeeping (`leaf_begin_`/`leaf_count_`) shape; the captured-g/h objective(s)/dataset/`boost_from_average` config (D-03); the `force_col_wise=true`/`force_row_wise=true` capture configs (D-04); the leaf-wise queue tie-break determinism mechanism (bounded by "must match C++ selection order"); the golden serialization/layering format for per-split + per-tree fixtures (bounded by the oracle-harness comparator + Phase-3 `%.17g`). When C++ behavior is the spec, the C++ source is authoritative over any inferred default.

### Deferred Ideas (OUT OF SCOPE)

- **TRL-06 categorical splits** (`SplitCategorical`/`FindBestThresholdCategorical`: `max_cat_threshold`, `cat_smooth`, `min_data_per_group`, `max_cat_to_onehot`, `cat_l2`) — Phase 7.
- **GBDT spine, objectives, metrics, bagging, early stopping** — Phase 6. This phase grows one tree from fixed g/h.
- **DART / RF / GOSS** variants — Phase 7.
- **Monotone / interaction constraints, forced splits/bins, extra-trees, CEGB, refit** — Phase 7.
- **Parallel (rayon) CPU or multi-GPU histogram-build path** — post-MVP optimization on the per-feature/per-row independence seam; must still match the deterministic anchor when added.
- **Captured-g/h objective/dataset specifics + `boost_from_average` config** — Claude's discretion under D-03.
- **ROCm cross-check of the full learner** — research/planning call (CPU bit-exact is the hard gate per P4 D-03).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRL-01 | Histogram-based serial learner (`ConstructHistograms` → `FindBestSplitsFromHistograms` → `Split`) | Algorithm flow documented from `serial_tree_learner.cpp:179-618` (Train loop, FindBestSplits, ConstructHistograms, FindBestSplitsFromHistograms); Backend ops already exist |
| TRL-02 | Histogram subtraction trick — byte-identical FP path | Smaller-child selection rule `serial_tree_learner.cpp:369-378`; subtract math `feature_histogram.hpp:140-144` (`data_[i] -= other.data_[i]` over `(num_bin - offset)*2` cells); `Backend::subtract_histograms` exists |
| TRL-03 | Leaf-wise (best-first) growth with `num_leaves`/`max_depth` caps | `Train` loop `serial_tree_learner.cpp:218-236` (`num_leaves-1` splits, `ArgMax`), `max_depth` gate `BeforeFindBestSplit:343-352` |
| TRL-04 | Split-gain scan + exact gain formula + tie-breaking | Gain primitives exist (`gain.rs`); tie-break is `SplitInfo::operator>` `split_info.hpp:138-165` (gain, then smaller feature index); `min_gain_to_split` added back at `serial_tree_learner.cpp:804` |
| TRL-05 | Numerical threshold splits + C++ missing/zero routing | `SplitInner:779-806`; `SKIP_DEFAULT_BIN`/`NA_AS_MISSING` dispatch `feature_histogram.hpp:284-285`; data-partition routing (`Backend::data_partition` exists) |
| TRL-07 | Data partition (row→leaf) feeding subtraction | `DataPartition` `data_partition.hpp` (`Split`, `leaf_begin_`/`leaf_count_`/`indices_`); `Backend::data_partition` returns reordered indices + split point |
| TRL-08 | Feature subsampling per-tree/per-node | `ColSampler` `col_sampler.hpp` (`ResetByTree`, `GetByNode`, `Random::Sample`); `lgbm-core::Random::sample` exists |
| TRL-09 | `force_row_wise`/`force_col_wise`, output-matching | Strategy selected in `GetShareStates` `serial_tree_learner.cpp:81-112`; both must produce identical trees (the difference is histogram-build ORDER, not result) |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| (workspace-internal) `lgbm-compute` | path dep | `Backend` trait: `construct_histograms`, `find_best_split`, `subtract_histograms`, `data_partition` | Phase-4 deliverable; CMP-01 CubeCL seam — the learner's ONLY compute dependency |
| (workspace-internal) `lgbm-dataset` | path dep | Immutable binned columnar store; `Bin` trait, `FeatureGroup` offsets, `bin_mapper` (`num_bin`, `default_bin`, `most_freq_bin`, `missing_type`, `bin_type`) | Phase-2 determinism root; the histogram/partition input — do NOT re-bin |
| (workspace-internal) `lgbm-model` | path dep | `Tree` struct + `%.17g` formatter — the grown-tree target + D-07 comparison machinery | Phase-3; add a `Tree::split` mutation method here |
| (workspace-internal) `lgbm-core` | path dep | `Config` (gain/split/subsampling params), f32 `types`, `Random` LCG, `thiserror` error idiom | Phase-1 foundation; `Random::sample` drives `ColSampler` parity |
| `thiserror` | `2.0.18` `[VERIFIED: workspace Cargo.toml]` | Structured `TreeLearnerError` at the crate boundary | Project FND-04 convention |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl` | `0.10.0` `[VERIFIED: workspace Cargo.toml]` | Compute client type — **confined to `lgbm-compute`** | NEVER named by `lgbm-treelearner` (CMP-01 guard). The learner takes a `&Backend` / its client through the Backend API only |
| `anyhow` | `1.0.102` `[VERIFIED: workspace Cargo.toml]` | Ergonomic error propagation in xtask/tests | Capture harness + parity tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `lgbm-treelearner` crate | Add modules to `lgbm-model` | A new crate keeps the loosely-coupled-by-responsibility workspace convention (FND-02) and a clean `thiserror` boundary; `lgbm-model` would conflate the static model with the growth engine. Recommend **new crate** |
| `BinaryHeap` priority queue for leaf-wise | `Vec<SplitInfo>` + linear `ArgMax` each iteration | C++ uses a flat `best_split_per_leaf_` vector scanned by `ArrayArgs::ArgMax` (`serial_tree_learner.cpp:225`). A `BinaryHeap` would change tie-break order. Recommend **flat Vec + ArgMax** to mirror C++ exactly |

**Installation:**
No external packages. All dependencies are workspace path crates already present. Add a new member crate to `Cargo.toml [workspace] members`.

**Version verification:** No registry packages are introduced by this phase — every dependency is an in-workspace path crate or an already-pinned workspace dependency (`thiserror 2.0.18`, `anyhow 1.0.102`, `cubecl 0.10.0`, all `[VERIFIED: workspace Cargo.toml]`). The package legitimacy audit below is therefore trivially clean.

## Package Legitimacy Audit

> This phase installs **no new external packages**. All code is workspace-internal (path dependencies) plus the three already-pinned workspace dependencies introduced and audited in prior phases.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `thiserror` | crates.io | 7+ yrs | very high | github.com/dtolnay/thiserror | N/A (pre-pinned P1) | Approved (no change) |
| `anyhow` | crates.io | 6+ yrs | very high | github.com/dtolnay/anyhow | N/A (pre-pinned P1) | Approved (no change) |
| `cubecl` | crates.io | alpha (0.10.0) | moderate | github.com/tracel-ai/cubecl | N/A (pre-pinned P4, confined to lgbm-compute) | Approved (no change) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*slopcheck was not run because no package is being added or changed; the disposition is "no new dependency surface." If the planner discovers a need for a new crate (e.g. a priority-queue helper), it must run the Package Legitimacy Gate then — but the recommendation here is to use no new crate.*

## Architecture Patterns

### System Architecture Diagram

```text
                          fixed gradients[] + hessians[]  (f32, from a golden fixture, NOT a runtime objective)
                                         │
                                         ▼
   ┌───────────────────────── SerialTreeLearner::Train ──────────────────────────┐
   │                                                                              │
   │  BeforeTrain ──► col_sampler.ResetByTree() ──► data_partition.Init()         │
   │     │              (Random::Sample, TRL-08)        (all rows → leaf 0)        │
   │     └──► smaller_leaf_splits.Init(grad,hess)  (root sum: ordered f64 fold)    │
   │     └──► tree.SetLeafOutput(0, CalculateSplittedLeafOutput(...))  (root leaf) │
   │                                                                              │
   │  for split in 0 .. num_leaves-1:                                             │
   │   ┌───────────────────────────────────────────────────────────────────────┐│
   │   │ BeforeFindBestSplit(left_leaf,right_leaf)                              ││
   │   │   • max_depth gate, min_data*2 gate                                    ││
   │   │   • SMALLER-CHILD SELECTION (TRL-02 keystone):                         ││
   │   │       num_data_in_left < num_data_in_right ?                           ││
   │   │         parent→larger; smaller=right(pool.Move); else parent→larger    ││
   │   │       use_subtract = (parent_hist != null)                             ││
   │   ├───────────────────────────────────────────────────────────────────────┤│
   │   │ FindBestSplits                                                          ││
   │   │   is_feature_used[f] = col_sampler.is_feature_used_bytree() & splittable││
   │   │   ConstructHistograms ───────────────► Backend::construct_histograms    ││  (Phase 4)
   │   │     (smaller leaf; larger via Backend::subtract_histograms if use_subtract)││
   │   │   FindBestSplitsFromHistograms:                                         ││
   │   │     for each used feature f:                                            ││
   │   │       FixHistogram(f, sum_g, sum_h)  ◄── most-freq-bin reconstruct      ││  (learner)
   │   │       ComputeBestSplitForFeature ───► Backend::find_best_split (gain)   ││  (Phase 4, D-01a)
   │   │       smaller_best = max(smaller_best, new_split)  (SplitInfo::op>)     ││
   │   │     best_split_per_leaf_[smaller_leaf] = ArgMax(smaller_best)           ││
   │   │     best_split_per_leaf_[larger_leaf]  = ArgMax(larger_best)            ││
   │   ├───────────────────────────────────────────────────────────────────────┤│
   │   │ best_leaf = ArgMax(best_split_per_leaf_)   (leaf-wise, TRL-03)          ││
   │   │ if best gain <= 0: break                                                ││
   │   │ Split / SplitInner(best_leaf):                                          ││
   │   │   data_partition.Split ──────────────► Backend::data_partition (TRL-07) ││  (Phase 4 + learner bookkeeping)
   │   │   tree.Split(...)  (grow node + 2 leaves, leaf_depth/parent) (D-07)     ││  (lgbm-model)
   │   │   smaller/larger_leaf_splits.Init(child sums)  (seed next iteration)    ││
   │   └───────────────────────────────────────────────────────────────────────┘│
   │                                                                              │
   └──────────────────────────────────► Tree  ───────────────────────────────────┘
                                         │
                                  (Phase-6 GBDT consumes this Tree)
```

### Recommended Project Structure

```text
crates/lgbm-treelearner/
├── Cargo.toml                    # path deps: lgbm-compute, lgbm-dataset, lgbm-model, lgbm-core; thiserror
└── src/
    ├── lib.rs                    # SerialTreeLearner facade + TreeLearnerError (thiserror boundary)
    ├── learner.rs               # Train loop, BeforeTrain, BeforeFindBestSplit, FindBestSplits, SplitInner
    ├── leaf_splits.rs           # LeafSplits: sum_gradient/sum_hessian/weight seeding (deterministic fold)
    ├── data_partition.rs        # DataPartition: leaf_begin_/leaf_count_/indices_ + Split (wraps Backend op)
    ├── histogram_pool.rs        # HistogramPool: sizing/eviction/Get/Move/ResetMap (D-05 full mirror)
    ├── col_sampler.rs           # ColSampler: feature_fraction(_bynode) Random::Sample parity (TRL-08)
    └── split_info.rs            # SplitInfo + operator> tie-break (gain, then smaller feature)
```

(Plus: `lgbm-model::Tree` gains a `split(...)` mutation method; `xtask` gains a `learner-capture` subcommand; `oracle-harness/tests/learner_parity.rs` is the per-split + per-tree replay.)

### Pattern 1: Smaller-child selection drives the subtraction trick (TRL-02 keystone)

**What:** After a split produces a left and right child, the NEXT iteration builds the histogram of the child with FEWER rows directly and DERIVES the larger child by subtracting (`larger = parent - smaller`). The selection rule is purely by row count.
**When to use:** Every non-root split.
**Example:**
```cpp
// Source: LightGBM/src/treelearner/serial_tree_learner.cpp:369-378 (BeforeFindBestSplit)
if (right_leaf < 0) {                                  // root: no subtraction
  histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  larger_leaf_histogram_array_ = nullptr;
} else if (num_data_in_left_child < num_data_in_right_child) {
  if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) {
    parent_leaf_histogram_array_ = larger_leaf_histogram_array_;   // parent kept as larger
  }
  histogram_pool_.Move(left_leaf, right_leaf);
  histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);  // smaller = left
} else {
  if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) {
    parent_leaf_histogram_array_ = larger_leaf_histogram_array_;
  }
  histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_); // smaller = right
}
// use_subtract = (parent_leaf_histogram_array_ != nullptr)   (line 398)
```
**Subtract math (Phase-4 kernel exists, `Backend::subtract_histograms`):**
```cpp
// Source: feature_histogram.hpp:140-144 (USE_DIST_GRAD=false path — the f32/f64 path)
for (int i = 0; i < (meta_->num_bin - meta_->offset) * 2; ++i) {
  data_[i] -= other.data_[i];   // larger -= smaller, stride-2 [g,h], from offset
}
```
Note `meta_->offset` (1 when `most_freq_bin == 0`, else 0) and the subtract runs over `num_bin - offset` bins — the learner must pass the correctly-offset slice.

### Pattern 2: FixHistogram runs before EVERY scan (most-freq-bin reconstruction)

**What:** The histogram for the `most_freq_bin` is not accumulated directly; it is reconstructed as `leaf_total - sum(all other bins)`. This happens for every feature, in both the directly-built and subtracted leaves, **immediately before** `find_best_split`.
**Why load-bearing:** The gain scan reads the most-freq-bin cell; if it isn't fixed first, the scan is wrong. This is a learner-side step NOT currently in the Backend.
**Example:**
```cpp
// Source: src/io/dataset.cpp:1488-1506
const int most_freq_bin = bin_mapper->GetMostFreqBin();
if (most_freq_bin > 0) {
  GET_GRAD(data, most_freq_bin) = sum_gradient;        // leaf total
  GET_HESS(data, most_freq_bin) = sum_hessian;
  for (int i = 0; i < num_bin; ++i) {
    if (i != most_freq_bin) {
      GET_GRAD(data, most_freq_bin) -= GET_GRAD(data, i);
      GET_HESS(data, most_freq_bin) -= GET_HESS(data, i);
    }
  }
}
```
The `sum_hessian` passed here is the leaf's `smaller_leaf_splits_->sum_hessians()` (NOT the `+2·kEpsilon` bumped one — that bump happens inside `FindBestThreshold`).

### Pattern 3: Tie-break is gain, then smallest feature index (TRL-04)

**What:** `SplitInfo::operator>` first compares gain; on a gain tie it prefers the SMALLER feature index (with `-1` mapped to `INT32_MAX`). `ArgMax` keeps the first-encountered on a full tie. The per-bin scan keeps the first threshold on a gain tie (strict `>` in `current_gain > best_gain`).
**Example:**
```cpp
// Source: split_info.hpp:138-165
inline bool operator > (const SplitInfo& si) const {
  // (NaN→kMinScore guards) ...
  if (local_gain != other_gain) return local_gain > other_gain;
  int local_feature = (this->feature == -1) ? INT32_MAX : this->feature;
  int other_feature = (si.feature == -1)   ? INT32_MAX : si.feature;
  return local_feature < other_feature;   // smaller feature wins ties
}
```
**The reported gain has `min_gain_to_split` added back** when stored on the tree node: `static_cast<float>(best_split_info.gain + config_->min_gain_to_split)` (`serial_tree_learner.cpp:804`). The Phase-4 `find_best_split` returns gain already net of `min_gain_shift` (`= gain_shift + min_gain_to_split`), so the learner adds `min_gain_to_split` back only for the tree-model `split_gain` field, NOT for selection.

### Pattern 4: Root leaf output and parent_output (path-smoothing seed)

**What:** The root leaf output is set by hand before the loop; `GetParentOutput` returns the root's own output for a single-leaf tree (no smoothing at root), else the leaf's stored `weight()`.
**Example:**
```cpp
// Source: serial_tree_learner.cpp:205-208 (root) and :1005-1017 (GetParentOutput)
tree->SetLeafOutput(0, CalculateSplittedLeafOutput<true,true,true,false>(
  sum_g, sum_h, lambda_l1, lambda_l2, max_delta_step, BasicConstraint(),
  path_smooth, num_data, 0));
// GetParentOutput: num_leaves==1 ? CalculateSplittedLeafOutput(...) : leaf_splits->weight()
```
For the spine (D-04: `path_smooth=0`), `parent_output` is computed but does not affect the gain (the default `USE_SMOOTHING=false` path ignores it). Carry it faithfully anyway.

### Anti-Patterns to Avoid

- **Re-deriving per-bin gains in the runtime learner.** The gain math is the Phase-4 `find_best_split` kernel (D-01a). The learner calls it; it only owns the cross-feature argmax. Re-deriving would duplicate (and risk diverging from) the kernel.
- **Using the Phase-4 `cfg_skip_default_bin(default_bin, num_bin)` heuristic unchanged.** It is a documented Phase-4 approximation (`default_bin < num_bin`) that is NOT a faithful transcription of the C++ dispatch (`num_bin > 2 && missing_type == Zero`). See Pitfall 1 — this phase must replace it.
- **`BinaryHeap`/`BTreeMap` leaf queue.** Changes tie-break vs the C++ flat-vector `ArgMax`. Use a flat `Vec<SplitInfo>` + first-max `ArgMax`.
- **Parallel/unordered sum folds in `LeafSplits`.** The deterministic reference disables the OpenMP reduction (`if (... && !deterministic_)`). Use a single ordered fold.
- **"Improving" the subtraction trick or pool eviction.** D-05: mirror the full pool faithfully — eviction order can be observable in the FP path if the parent histogram is reused.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-bin gain scan | A new threshold loop | `Backend::find_best_split` (`kernels/split.rs`) | Already a verbatim transcription, bit-exact vs C++ golden; D-01a |
| Histogram accumulation | A new fold | `Backend::construct_histograms` | Phase-4, bit-exact |
| Histogram subtraction math | A new subtract loop | `Backend::subtract_histograms` | Phase-4, bit-exact |
| Row→leaf reorder | A new partition | `Backend::data_partition` (returns reordered indices + split point) | Phase-4, bit-exact; learner only owns `leaf_begin_`/`leaf_count_` |
| `%.17g` / `{:g}` tree serialization for D-07 compare | A new float formatter | `lgbm-model::format::{format_g17, format_g6}` | Phase-3, bit-exact vs C printf `%g` |
| Feature-subsample RNG | A new PRNG | `lgbm-core::Random` (`sample`, `next_short`) | Phase-1, bit-exact vs C++ `Random::Sample` |
| Bin layout (num_bin/offset/default_bin/most_freq_bin/missing_type) | Re-deriving from data | `lgbm-dataset` `bin_mapper` accessors | Phase-2 determinism root; re-deriving risks divergence |
| Abs-diff / bit-exact comparison | A new comparator | `oracle-harness::{compare_within, compare_exact_*}` | Phase-1/2/3 seam |

**Key insight:** ~80% of the numeric heavy lifting already exists as bit-exact Phase-4 kernels and Phase-3 formatters. Phase 5 is **glue + bookkeeping + RNG sequencing**, not new numerics. The ONE genuinely-new numeric is `FixHistogram` (most-freq-bin reconstruct) and the `LeafSplits` ordered sum folds — both are short, deterministic, and easy to golden.

## Runtime State Inventory

> Phase 5 is a greenfield code-addition phase (new crate + new tests + extended capture harness). It alters NO stored data, live-service config, OS-registered state, or secrets. The only "state" considerations are committed fixtures and the capture harness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no database or datastore is involved. | none |
| Live service config | None — no external service. | none |
| OS-registered state | None — no scheduled task / daemon. | none |
| Secrets/env vars | `$LGBM_CAPTURE_PYTHON` is referenced by the Phase-3 `model-capture` harness; the D-03 captured-g/h path MAY reuse a pip-lightgbm capture (same env var). Code reads it only at capture time. No new secret. | none (reuse existing env var if a pip capture is chosen for D-03) |
| Build artifacts | New `tests/fixtures/` golden files (per-split + per-tree learner goldens) will be **committed** (D-01). The untracked `LightGBM/` tree must NEVER be `git add`-ed (memory: lightgbm-ref-tree-untracked). The new `lgbm-treelearner` crate adds a `target/` build artifact (gitignored, no action). | Commit goldens into `tests/fixtures/`; verify `.gitignore` still excludes `LightGBM/` and `target/` |

**Nothing found in categories Stored data / Live service config / OS-registered state:** verified — Phase 5 touches only the Rust workspace and committed test fixtures.

## Common Pitfalls

### Pitfall 1: The Phase-4 `cfg_skip_default_bin` heuristic is a known divergence this phase MUST resolve

**What goes wrong:** Phase-4's `kernels/split.rs::cfg_skip_default_bin` approximates the C++ `SKIP_DEFAULT_BIN` template flag as `default_bin < num_bin`. The authoritative predicate is `num_bin > 2 && missing_type == MissingType::Zero` for skip, and `num_bin > 2 && missing_type == MissingType::NaN` selects the **NA_AS_MISSING** forward-branch special case (which the Phase-4 kernel does NOT implement). With `missing_type == None`, both are false.
**Why it happens:** Phase-4 deferred threading `missing_type` into the kernel surface (recorded in `04-REVIEW-FIX.md`, and in the `cfg_skip_default_bin` doc comment) because it changes the `Backend::find_best_split` signature.
**How to avoid:** Thread the authoritative `skip_default_bin` and `na_as_missing` flags — derived in the learner from the feature's `bin_mapper.missing_type()` + `num_bin > 2` — through `ComputeBestSplitForFeature` → `Backend::find_best_split`. Add a golden case where `default_bin < num_bin` but `skip_default_bin == false` (e.g. `missing_type == None`) to exercise the divergence. **NA_AS_MISSING** (`feature_histogram.hpp:945-961`) is a distinct forward-branch preamble (offset==1: seed left from the full leaf, subtract every bin, start `t=-1`) — if any D-03/synthetic case uses `missing_type == NaN` with `num_bin > 2`, the kernel needs that branch added too. The spine (D-04) can pin synthetic cases to `missing_type == None` to defer NA_AS_MISSING, but the plan must DECIDE this explicitly and the captured-g/h dataset (D-03) may force the issue.
**Warning signs:** A feature whose `default_bin` differs from `num_bin` but whose missing-type is `None` selects a different threshold than C++; or any NaN-missing feature in the captured dataset diverges.
**Authority:** `feature_histogram.hpp:284-285` (the dispatch); `kernels/split.rs:806-838` (the Phase-4 heuristic + its own follow-up note).

### Pitfall 2: `FixHistogram` ordering and the un-bumped sum_hessian

**What goes wrong:** Calling `find_best_split` before reconstructing the `most_freq_bin` cell, or passing the `+2·kEpsilon`-bumped hessian into `FixHistogram`.
**Why it happens:** The `2·kEpsilon` bump is internal to `FindBestThreshold` (`feature_histogram.hpp:172`); `FixHistogram` uses the raw leaf sum (`serial_tree_learner.cpp:531-534`). Two different hessian values flow through within one feature's processing.
**How to avoid:** Reconstruct most-freq-bin with the raw `leaf_splits->sum_hessians()`, THEN call the Backend split (which the Phase-4 host code internally adds `2·kEpsilon` to). Order: `FixHistogram(raw sums)` → `find_best_split(raw sums; +2εk applied inside)`.
**Warning signs:** Most-freq-bin gain off by ~`cnt_factor * 2·kEpsilon`; or the wrong feature wins on a tie.

### Pitfall 3: Smaller-child selection uses GLOBAL data counts, not the bagged subset

**What goes wrong:** Selecting the smaller child by the wrong count (or by sum_hessian instead of count).
**Why it happens:** `BeforeFindBestSplit` uses `GetGlobalDataCountInLeaf` = `data_partition_->leaf_count(leaf)` (`serial_tree_learner.cpp:353-354,369`), i.e. the row COUNT in the partition. After `SplitInner`, the next-iteration leaf-splits seeding ALSO selects smaller/larger by `best_split_info.left_count < best_split_info.right_count` (`:851`). Both must be the partition row counts.
**How to avoid:** Drive both the subtraction-trick selection AND the `LeafSplits::Init` seeding off `DataPartition::leaf_count`. No bagging this phase, so global == local.
**Warning signs:** The wrong child is built directly; subtracted child has negative or mismatched cells.

### Pitfall 4: `kEpsilon`/`2·kEpsilon` and the `cnt = RoundInt(hess * cnt_factor)` count reconstruction

**What goes wrong:** Per-bin row counts in the scan are RECONSTRUCTED from `hess * cnt_factor` where `cnt_factor = num_data / sum_hessian`, NOT counted directly. `Common::RoundInt(x) = (int)(x + 0.5f)` uses a **float** `0.5f`. Getting the rounding or the f32 `0.5` wrong shifts the `min_data_in_leaf` gates.
**Why it happens:** LightGBM never stores per-bin counts; it infers them from the hessian (valid because each row contributes `hessian` to its bin and `1` to the count, scaled by `cnt_factor`). This is already correct in the Phase-4 kernel (`round_int`), but the learner-capture transcription (D-02) must reproduce it identically for the cross-check (D-02a).
**How to avoid:** Reuse the existing `round_int` (`kernels/split.rs:77-80`) semantics; the learner-emitter must transcribe `Common::RoundInt` with the f32 `0.5f` literal verbatim.
**Warning signs:** Off-by-one `left_count`/`right_count`; gates fire one bin early/late.

### Pitfall 5: `force_row_wise` vs `force_col_wise` must produce the SAME tree (TRL-09), not a different one

**What goes wrong:** Treating the two strategies as producing different (both-valid) trees. They differ ONLY in histogram-build traversal ORDER (row-major vs column-major accumulation); the resulting histograms — and therefore every split and the whole tree — must be **identical** for a deterministic single-thread run.
**Why it happens:** The strategy is a performance knob (`GetShareStates`, `serial_tree_learner.cpp:88-94`); for the deterministic anchor both reduce to the same ordered fold over the same data.
**How to avoid:** Capture goldens under BOTH `force_row_wise=true` and `force_col_wise=true` (D-04) and assert the produced trees are bit-identical to EACH OTHER and to C++. The Phase-4 `construct_histograms` is a single whole-kernel op; if both strategies route through it identically on the deterministic anchor, the learner difference may be a no-op at the Backend layer — VERIFY this and document whether `force_col_wise` needs a distinct accumulation path or is observationally identical (it likely is, on the single-thread anchor).
**Warning signs:** The two strategies produce different trees (a real bug, not acceptable parity).

### Pitfall 6: Two-transcription drift (D-02a)

**What goes wrong:** The Phase-4 kernel emitter (`kernel_capture.cpp`) and the new Phase-5 learner emitter both transcribe the per-feature histogram + gain scan. Over time they silently diverge (e.g. one gets a bug fix the other doesn't).
**Why it happens:** Deliberate redundancy (belt-and-suspenders for the keystone) without an explicit guard.
**How to avoid:** Add a consistency test that feeds the SAME synthetic per-feature inputs to both transcriptions and asserts bit-identical per-feature histograms + gains. Make it a committed parity test, not just a manual check.
**Warning signs:** The learner-capture per-split golden disagrees with the Phase-4 `split.txt` golden on the same inputs.

### Pitfall 7: `cubecl-cpu` spawns one OS thread per cube unit (carried from Phase 4)

**What goes wrong:** Assuming single-owner kernels are truly single-threaded.
**Why it happens:** Documented in P4 RESEARCH Pitfall 1 / settled by D-04a — `CubeDim::new_1d(1)` is bit-exact across launches despite the worker thread, because exactly one unit owns the fold.
**How to avoid:** Keep all learner-driven Backend calls on the single-owner path (already the case). No new exposure this phase.
**Warning signs:** None expected (settled), but re-assert if a new kernel shape is introduced.

## Code Examples

### The leaf-wise growth loop (TRL-01, TRL-03)
```cpp
// Source: LightGBM/src/treelearner/serial_tree_learner.cpp:218-236
for (int split = init_splits; split < config_->num_leaves - 1; ++split) {
  if (BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
    FindBestSplits(tree_ptr);                       // construct + find best per feature
  }
  int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
  const SplitInfo& best = best_split_per_leaf_[best_leaf];
  if (best.gain <= 0.0) break;                       // no positive-gain split → stop
  Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
  cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
}
```

### max_depth + min_data gates (TRL-03)
```cpp
// Source: serial_tree_learner.cpp:343-363 (BeforeFindBestSplit)
if (config_->max_depth > 0 && tree->leaf_depth(left_leaf) >= config_->max_depth) {
  best_split_per_leaf_[left_leaf].gain = kMinScore;     // = -inf, never chosen
  if (right_leaf >= 0) best_split_per_leaf_[right_leaf].gain = kMinScore;
  return false;
}
if (num_data_in_right_child < min_data_in_leaf*2 &&
    num_data_in_left_child  < min_data_in_leaf*2) {      // both children too small
  best_split_per_leaf_[left_leaf].gain = kMinScore; /* + right */ return false;
}
```

### Tree growth mutation — `lgbm-model::Tree` needs this method (D-07)
```cpp
// Source: LightGBM/include/LightGBM/tree.h:543-585 (inline Tree::Split structural part)
// new_node_idx = num_leaves_ - 1; rewire parent's child pointer to this node;
// split_feature_inner_/split_feature_/split_gain_ set;
// left_child_[node]=~leaf, right_child_[node]=~num_leaves_;
// internal_value_[node]=leaf_value_[leaf]; leaf_value_[leaf]=left_value; leaf_value_[num_leaves_]=right_value;
// leaf_depth_[num_leaves_]=leaf_depth_[leaf]+1; leaf_depth_[leaf]++; ++num_leaves_;
// Source: tree.cpp:61-75 (the public Split adds): decision_type_ (default_left, missing_type),
//   threshold_in_bin_, threshold_ (double).
```
The Phase-3 `lgbm-model::Tree` has all these fields (`left_child`, `right_child`, `split_feature`, `threshold`, `decision_type`, `split_gain`, `leaf_value`, `leaf_weight`, `leaf_count`, `internal_value`, `internal_weight`, `internal_count`) but no mutation method and no `leaf_depth`/`leaf_parent`/`split_feature_inner`/`threshold_in_bin`/`leaf_parent` tracking arrays (those are growth-time only). Phase 5 adds the growth-time arrays + a `split()` method; the serialized form already matches.

### ColSampler per-tree feature selection (TRL-08)
```cpp
// Source: col_sampler.hpp:74-89 (ResetByTree) — only when feature_fraction < 1.0
used_feature_indices_ = random_.Sample(valid_feature_indices_.size(), used_cnt_bytree_);
// GetCnt(total, fraction) = max( RoundInt(total*fraction), min(1,total) )  (:33-37)
// GetByNode (:91-179) draws AGAIN with random_.Sample for feature_fraction_bynode < 1.0
```
`lgbm-core::Random::sample(n, k)` already exists and is bit-exact (FND-01). The parity risk is the CALL SEQUENCE: `ResetByTree` draws once per tree; `GetByNode` draws once per node (for the smaller leaf, then the larger leaf, in `FindBestSplitsFromHistograms:479,487`). Reproduce the exact order.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-bin counts stored | Counts inferred via `RoundInt(hess * cnt_factor)` | long-standing LightGBM | Must transcribe the f32-`0.5f` rounding (Pitfall 4) |
| (this port) Phase-4 `cfg_skip_default_bin` heuristic | Authoritative `missing_type`-derived `skip_default_bin`/`na_as_missing` | **this phase** | Resolves the documented Phase-4 divergence (Pitfall 1) |

**Deprecated/outdated:**
- The integer-quantized histogram path (`use_quantized_grad`, `FindBestThresholdInt`, `GetHistBitsInLeaf`, `gradient_discretizer_`) is **dropped project-wide** (P1 D-02/D-03). Ignore every `if (config_->use_quantized_grad)` branch — the port only implements the `else` (f32/f64) branch.
- Monotone constraints / CEGB / linear-tree branches in `serial_tree_learner.cpp` are Phase-7+ — implement the no-constraint, no-cegb, non-linear default path only.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `force_col_wise` is observationally identical to `force_row_wise` on the single-thread deterministic anchor (the Phase-4 `construct_histograms` whole-kernel op routes both the same), so TRL-09 may not need a distinct accumulation path | Pitfall 5 | If wrong, `force_col_wise` needs its own column-major build path in `lgbm-compute` (a Phase-4 boundary re-open). MITIGATION: capture both goldens early (D-04) and compare; this is empirically checkable in the first wave |
| A2 | Recommending a NEW `lgbm-treelearner` crate (vs extending `lgbm-model`) | Standard Stack / Structure | Low risk — Claude's discretion per CONTEXT. If the planner prefers a module in `lgbm-model`, the algorithm content is unchanged |
| A3 | `FixHistogram` should live in the learner (not be added as a new `Backend` op) | Architectural Map / Open Q1 | Medium — see Open Q1. If it belongs in the Backend, it's a small Phase-4-style kernel addition; either placement is bit-exact-able |
| A4 | The D-03 captured-g/h fixture can reuse the Phase-3 `$LGBM_CAPTURE_PYTHON` pip-lightgbm capture path for iteration-1 g/h | Runtime State / D-03 | Low — Claude's discretion. If pip can't expose iteration-1 g/h directly, a header-only objective transcription (regression-l2 grad=score-label, hess=1) produces the same g/h deterministically |
| A5 | NA_AS_MISSING (NaN-missing forward branch) can be deferred from the D-04 spine by pinning synthetic cases to `missing_type == None`, with the captured-g/h dataset (D-03) chosen to avoid NaN-missing features OR explicitly adding the branch | Pitfall 1 | Medium — if the realistic D-03 dataset has NaN-missing numeric features, the kernel needs the NA_AS_MISSING branch (`feature_histogram.hpp:945-961`) added. The plan must decide dataset composition vs branch-implementation explicitly |

**If this table is empty:** it is not — five assumptions need confirmation during planning/discuss.

## Open Questions

1. **Where does `FixHistogram` (most-freq-bin reconstruct) belong — learner or Backend?**
   - What we know: C++ runs it in the learner (`Dataset::FixHistogram`) immediately before each per-feature scan, on the raw leaf sums. It is a short, deterministic f64 loop.
   - What's unclear: whether to keep it learner-side (simplest, matches C++ call site) or fold it into a Backend op (keeps all histogram-cell math behind the CubeCL seam).
   - Recommendation: keep it **learner-side** as a plain f64 function (it operates on the host-side histogram `Vec<f64>` returned by `construct_histograms`); golden it as part of the per-split snapshot. Revisit only if a future ROCm path needs it GPU-resident.

2. **Does `force_col_wise` require a distinct compute path (A1)?**
   - What we know: the two strategies differ only in build order; results must be identical on the deterministic anchor.
   - What's unclear: whether the existing Phase-4 `construct_histograms` already covers both, or `force_col_wise` needs a column-major accumulation that produces the same f64 cells.
   - Recommendation: in Wave 1, capture both goldens and assert tree-equality; if identical at the Backend layer, TRL-09 is satisfied by a config flag with no new kernel. Document the finding.

3. **D-03 captured-g/h: which objective/dataset/`boost_from_average`?** (Claude's discretion)
   - What we know: needs realistic iteration-1 g/h from a real C++ objective on a real dataset; the Phase-2/3 example datasets (regression 28-feature, binary 28-feature, 500 rows) are already committed.
   - Recommendation: use `regression` (l2) iteration-1 (`grad = score - label`, `hess = 1`) with `boost_from_average=false` for the cleanest deterministic g/h, on the committed regression example matrix; add a `binary` (logloss) iteration-1 case for sign/magnitude spread. Avoid NaN-missing features in the chosen columns (ties to A5). Capture via the existing `$LGBM_CAPTURE_PYTHON` pip path or a header-only objective transcription.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (edition 2024) | building the new crate + tests | ✓ | 1.95.0 (`rust-toolchain.toml`) | — |
| `cubecl-cpu` (via `lgbm-compute`) | running Backend ops on the deterministic anchor | ✓ | 0.10.0 | — |
| `LightGBM/` reference tree | reading the port target source (read-only) | ✓ | on disk (untracked), VERSION 4.6.0.99 | — |
| C++ toolchain (for capture) | one-time golden regen via `xtask learner-capture` | ✓ (header-only transcription, no `external_libs`/lib link) | system g++ | header-only transcription is the fallback-by-design (D-01); no `lib_lightgbm` build needed |
| `$LGBM_CAPTURE_PYTHON` (pip lightgbm 4.6.0) | OPTIONAL — D-03 captured-g/h fixture | ✓ (used in Phase 3) | 4.6.0 | header-only objective transcription produces the same iteration-1 g/h |
| ROCm gfx1100 GPU | OPTIONAL — if the learner is re-checked on ROCm (deferred decision) | ✓ | ROCm 7.1.1 / cubecl-hip 0.10.0 | CPU bit-exact is the hard gate (P4 D-03); ROCm learner check is deferrable |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none blocking — the C++ capture is header-only by design; pip-lightgbm and ROCm are optional.

## Validation Architecture

> `workflow.nyquist_validation` is `true` in config.json — this section is included.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + `oracle-harness` golden comparators |
| Config file | none (workspace `cargo test`); fixtures under `tests/fixtures/` |
| Quick run command | `cargo test -p lgbm-treelearner` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRL-01 | Construct→find→split flow grows a tree | integration (golden replay) | `cargo test -p oracle-harness --test learner_parity` | ❌ Wave 0 |
| TRL-02 | Subtraction trick + smaller-child selection bit-faithful | integration (per-split snapshot) | `cargo test -p oracle-harness --test learner_parity subtract` | ❌ Wave 0 |
| TRL-03 | Leaf-wise growth respects `num_leaves`/`max_depth` | unit | `cargo test -p lgbm-treelearner leaf_wise_caps` | ❌ Wave 0 |
| TRL-04 | Per-split full per-bin gain arrays match (D-06) + tie-break | integration (per-split snapshot) | `cargo test -p oracle-harness --test learner_parity per_bin_gains` | ❌ Wave 0 |
| TRL-05 | Missing/zero/default-bin routing | integration | `cargo test -p oracle-harness --test learner_parity missing_routing` | ❌ Wave 0 |
| TRL-07 | Data partition row→leaf feeds subtraction | unit + integration | `cargo test -p lgbm-treelearner data_partition` | ❌ Wave 0 |
| TRL-08 | Feature subsample RNG parity (call sequence) | integration | `cargo test -p oracle-harness --test learner_parity col_sampler_rng` | ❌ Wave 0 |
| TRL-09 | `force_row_wise`==`force_col_wise`==C++ tree | integration (tree equality) | `cargo test -p oracle-harness --test learner_parity row_vs_col` | ❌ Wave 0 |
| D-02a | Two transcriptions agree bit-for-bit | integration (cross-check) | `cargo test -p oracle-harness --test learner_parity transcription_crosscheck` | ❌ Wave 0 |
| D-07 | Full grown tree bit-faithful (incl. leaf outputs) via `%.17g` | integration (model-text compare) | `cargo test -p oracle-harness --test learner_parity full_tree` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p lgbm-treelearner` (unit-level: leaf queue, gates, partition, col_sampler, fix_histogram)
- **Per wave merge:** `cargo test --workspace` (full golden replay + cross-workspace regression)
- **Phase gate:** Full suite green + per-split AND per-tree goldens replay bit-exact before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `crates/lgbm-treelearner/` crate + `Cargo.toml` (new workspace member) — covers all TRL-*
- [ ] `xtask` `learner-capture` subcommand + `xtask/cpp/learner_capture.cpp` (header-only D-01/D-02 transcription) — emits per-split + per-tree goldens
- [ ] `tests/fixtures/learner/` committed goldens (per-split per-bin gain arrays + full-tree model-text) for synthetic + captured-g/h corpora
- [ ] `crates/oracle-harness/tests/learner_parity.rs` — the per-split + per-tree replay harness
- [ ] `REFERENCE_MANIFEST.md` extension — pin the learner fixture set + capture config (D-04 row/col, D-03 g/h source)
- [ ] `lgbm-model::Tree::split(...)` mutation method + growth-time arrays (`leaf_depth`, `leaf_parent`, `split_feature_inner`, `threshold_in_bin`) — covered by the D-07 full-tree golden

*(No existing test infrastructure covers the learner; all rows are Wave 0. The comparators (`oracle-harness`), the `%.17g` formatter (`lgbm-model`), the capture pattern (`xtask`), and the Backend ops (`lgbm-compute`) all exist and are reused.)*

## Security Domain

> `security_enforcement` is `true`, `security_asvs_level: 1` in config.json — this section is included. Phase 5 is an internal numeric/compute library with NO network, auth, session, or untrusted external input at runtime; the ASVS surface is minimal and dominated by input-validation at the trust boundary (matrix/g/h/config ingestion).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (library) |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access control surface |
| V5 Input Validation | yes | Validate ALL public-entry input (g/h slice lengths == num_data, bin indices `< num_bin`, `num_leaves >= 1`, `sum_hessian > 0` before `cnt_factor` division) → typed `TreeLearnerError`, NEVER a panic/UB. Mirror the Phase-4 V5 pattern (`ComputeError::BinIndexOutOfRange`/`LengthMismatch`) at the learner boundary. The Phase-4 kernels already validate at their boundary; the learner adds its own at the `Train(gradients, hessians)` entry |
| V6 Cryptography | no | No crypto (the `Random` LCG is a deterministic PRNG for parity, NOT security — never used for secrets) |

### Known Threat Patterns for the Rust tree-learner

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds bin index → OOB histogram/partition read | Tampering / DoS | Bounds-check `bin < num_bin` at the boundary; the Phase-4 ops already return `BinIndexOutOfRange`; the learner must not pre-index unchecked |
| Length mismatch (g/h vs num_data; hist vs 2·num_bin) | Tampering | `LengthMismatch` typed error before any `unsafe`/kernel launch (Phase-4 precedent) |
| Division by zero / NaN `cnt_factor` (`num_data / sum_hessian`) | DoS | Reject non-positive/NaN `sum_hessian` at the boundary (`!(sum_hessian > 0.0)` NaN-catching, as in `find_best_split_cpu`) |
| Integer overflow in `2 * num_bin` histogram sizing | Tampering | `checked_mul` (Phase-4 precedent in `find_best_split_cpu`) |
| `num_bin` arithmetic underflow in REVERSE-scan bound (`offset >= 2`) | DoS / OOB | Already mitigated in the Phase-4 kernel (`in_range` + `t_safe` clamp, WR-02); preserve when wiring |
| Panic-as-control-flow at the library boundary | DoS | All boundary failures are `Result<_, TreeLearnerError>`; `unsafe` (cubecl launches) stays confined to `lgbm-compute` (CMP-01) |

**Note:** The g/h and config are produced internally (Phase-6 objective) or from committed fixtures — not from an untrusted network — so the realistic threat is a programming bug surfacing as UB/panic, which the V5 typed-boundary discipline (already established in Phase 4) prevents. No new attack surface is introduced.

## Sources

### Primary (HIGH confidence)
- `LightGBM/src/treelearner/serial_tree_learner.cpp` (read: `:1-120` Init/GetShareStates, `:120-245` ResetConfig/Train, `:288-402` BeforeTrain/BeforeFindBestSplit/FindBestSplits, `:404-618` ConstructHistograms/FindBestSplitsFromHistograms, `:620-918` ForceSplits/SplitInner, `:960-1017` ComputeBestSplitForFeature/GetParentOutput) — the primary port target
- `LightGBM/src/treelearner/serial_tree_learner.h` — member state (`data_partition_`, `histogram_pool_`, `best_split_per_leaf_`, `smaller/larger_leaf_splits_`, `col_sampler_`)
- `LightGBM/src/treelearner/feature_histogram.hpp` (`:99-145` Subtract, `:165-208` FindBestThreshold/BeforeNumerical, `:272-285` SKIP_DEFAULT_BIN/NA_AS_MISSING dispatch, `:711-815` ThresholdL1/CalculateSplittedLeafOutput/GetSplitGains/GetLeafGain, `:830-1039` FindBestThresholdSequentially scan, `:1367-1486` HistogramPool)
- `LightGBM/src/treelearner/data_partition.hpp` — `DataPartition` (`Init`, `Split`, `leaf_begin_`/`leaf_count_`/`indices_`)
- `LightGBM/src/treelearner/leaf_splits.hpp` — `LeafSplits` Init variants + the `!deterministic_` ordered-fold gating
- `LightGBM/src/treelearner/split_info.hpp` — `SplitInfo` + `operator>` tie-break (gain, then smaller feature)
- `LightGBM/src/treelearner/col_sampler.hpp` — `ColSampler` (`ResetByTree`, `GetByNode`, `Random::Sample`)
- `LightGBM/src/io/dataset.cpp:1488-1545` — `FixHistogram` (most-freq-bin reconstruct)
- `LightGBM/include/LightGBM/tree.h:543-585` + `LightGBM/src/io/tree.cpp:61-75` — `Tree::Split` (structural growth)
- `crates/lgbm-compute/src/lib.rs`, `gain.rs`, `kernels/split.rs` — the existing Backend surface (`construct_histograms`/`find_best_split`/`subtract_histograms`/`data_partition`, gain primitives, `cfg_skip_default_bin` heuristic + its Phase-5 follow-up note)
- `crates/lgbm-model/src/tree.rs` — the `Tree` struct (D-07 target/comparison)
- `crates/lgbm-core/src/random.rs` — `Random::sample`/`next_short` (TRL-08 RNG)
- `.planning/phases/05-tree-learner-split-finding/05-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `.planning/ROADMAP.md`

### Secondary (MEDIUM confidence)
- `.planning/STATE.md` Plan 04-03 result (the `cfg_skip_default_bin` L1-gain codegen + skip-default-bin follow-up context)
- `04-REVIEW-FIX.md` reference (the `skip_default_bin` Phase-5 follow-up, cited via `kernels/split.rs:830-835` doc comment)

### Tertiary (LOW confidence)
- None — all claims are sourced from the on-disk C++ reference or the existing Rust workspace.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all deps are in-workspace and already pinned/verified; no registry lookups needed
- Architecture: HIGH — the entire algorithm was read directly from the authoritative on-disk C++ reference
- Pitfalls: HIGH — each pitfall is tied to a specific C++ source line or a documented Phase-4 follow-up (the `cfg_skip_default_bin` divergence is explicitly flagged in the Phase-4 code)
- `force_col_wise` equivalence (A1) and `FixHistogram` placement (A3): MEDIUM — empirically checkable in Wave 1, flagged as open questions

**Research date:** 2026-06-06
**Valid until:** 2026-07-06 (stable — the C++ reference is pinned/read-only; the Rust workspace is internal. 30 days.)
