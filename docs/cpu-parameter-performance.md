# CPU parameter performance vs LightGBM 4.6

Measured 2026-08-05. This document records (a) a **correctness defect** found while
profiling, (b) the CPU speed position against the C++ reference, and (c) the
per-parameter cost matrix, including three parameters whose Rust cost profile
diverges sharply from C++ and are therefore ranked optimization targets.

## Environment

| | |
|---|---|
| Machine | Apple M1, 4 performance + 4 efficiency cores, 16 GB |
| Build | `--release` (`opt-level=3`, `lto="fat"`, `codegen-units=1`) |
| Reference | `lightgbm==4.6.0` (PyPI wheel, OpenMP, no GPU backend compiled in) |
| Workload | 200 000 rows × 50 continuous features, `objective=binary`, 100 iterations |
| Baseline params | `num_leaves=31`, `learning_rate=0.1`, `max_bin=255`, `seed=1`, `deterministic=true` |
| Method | warm (first run discarded), median of ≥3 timed reps, binning INSIDE the timed region for both engines |

LightGBM is always reported at its **best** thread count (the minimum over
`num_threads ∈ {1, 4, 8}`); `lightgbm_rs` uses the full rayon pool. This is
deliberately the *unfavourable* comparison for the Rust port — LightGBM is tuned
per cell, `lightgbm_rs` is not.

## 1. Correctness defect found while profiling (FIXED)

The `LGBM_UNIFIED_*` "unified fusion" host path (fused build+fix+scan for the
smaller child, fused subtract+scan for the larger) produces a **structurally wrong
model**:

| | two-step path | fused path |
|---|---|---|
| trees grown | 100 | **6** |
| first-tree max split gain | 1.0 × 10⁴ | **4.9 × 10⁸** |
| max prediction magnitude | ~10⁰ | **3.4 × 10³⁸** (f32 overflow) |

The pre-fix default was `core_scaled_threshold(100, rayon_cores())`, which
evaluates to 32–49 on a machine with ≤2 rayon threads and 83–100 above that. The
gate therefore **fired only on low-core machines**, so the same source tree
trained a broken model on a small box and a correct model on a large one — the
model was thread-count dependent (verified: 6 trees at `RAYON_NUM_THREADS=1`,
100 trees at `8`, on unmodified `master`).

**Fix:** both gates now default to `usize::MAX` (never fires), pinned by
`unified_fusion_defaults_are_disabled` in `crates/lgbm-compute/src/lib.rs`. The
path stays reachable via the env vars for debugging. Re-enabling requires a parity
test comparing fused output against two-step output.

After the fix the grown model is byte-identical at 1, 2, 4 and 8 rayon threads.

## 2. Speed position

### Change landed

`par_build_threshold` (the leaf-row count at/above which a histogram build
parallelizes over features) was `16384`. With `num_leaves=31` over 200k rows the
average leaf holds ~6.5k rows, so nearly every non-root build fell to the serial
path and the 8-core run scaled only **1.64×** over 1-core.

| `LGBM_PAR_THRESHOLD` | 16384 | 4096 | 1024 | 256 | 64 | 0 |
|---|---|---|---|---|---|---|
| train wall (ms) | 2457 | 2236 | **2121** | 2086 | 2102 | 2031 |

New default: **1024** — most of the win while still keeping genuinely tiny leaves
serial, so a box with higher rayon dispatch cost than this one cannot regress.
Parity-neutral: the parallel and serial paths fold in the same order into disjoint
regions (`build_histograms_parallel_equals_serial`).

Net effect on the baseline workload: **2582 ms → 2021 ms (22% faster)**.

### Where that leaves us

| Workload | lightgbm_rs | LightGBM 4.6 (best nthread) | |
|---|---|---|---|
| 200k × 50 | 2021 ms | 2045 ms (4t) | **parity** |
| 7k × 28 | 204 ms | 172 ms (1t) | 1.19× slower |
| 200k × 50, single-threaded | 4313 ms | 4696 ms (1t) | **1.09× faster** |

**The Rust engine is faster than LightGBM per-core; it loses on parallel
scaling.** 1 → 8 threads gains 1.64× here versus LightGBM's 2.3×. Closing that is
the remaining work, and §3 says where.

## 3. Per-parameter cost matrix

`ratio` = LightGBM time ÷ lightgbm_rs time. **Below 1.00 means lightgbm_rs is
slower.** LightGBM numbers are its best over `num_threads ∈ {1,4,8}`, re-tuned per
row.

| Parameter setting | lightgbm_rs (ms) | LightGBM (ms) | ratio |
|---|---:|---:|---:|
| baseline (`num_leaves=31 max_bin=255`) | 2033 | 1644 | 0.81× |
| `num_leaves=15` | 1370 | 1321 | 0.96× |
| `num_leaves=63` | 3110 | 2549 | 0.82× |
| `num_leaves=255` | 7901 | 5409 | 0.68× |
| `max_bin=15` | 1615 | 1256 | 0.78× |
| `max_bin=63` | 1611 | 1337 | 0.83× |
| `max_bin=511` | 2967 | 2439 | 0.82× |
| `max_depth=6` | 1722 | 1593 | 0.93× |
| `min_data_in_leaf=100` | 2016 | 1813 | 0.90× |
| `feature_fraction=0.5` (before §4 fix) | 2028 | 1356 | **0.67×** |
| `feature_fraction=0.5` (after §4 fix) | 1487 | 1381 | 0.93× |
| `bagging_freq=1 bagging_fraction=0.5` | 4518 | 1675 | **0.37×** |
| `boosting=dart` | 11380 | 2740 | **0.24×** |
| `boosting=goss` | 3945 | 1653 | **0.42×** |
| `extra_trees=true` | 1822 | 1736 | 0.95× |
| `linear_tree=true` | 5397 | 2130 | **0.39×** |
| `use_quantized_grad=true` | 2119 | 1429 | 0.67× |
| `force_col_wise=true` | 2080 | 1785 | 0.86× |

### Reading the matrix

The port tracks LightGBM within 4–19% across the ordinary tuning knobs
(`num_leaves`, `max_bin`, `max_depth`, `min_data_in_leaf`, `extra_trees`). Five
cells are structurally worse and are the ranked optimization targets:

1. ~~**`feature_fraction` buys NOTHING (0.67×).**~~ **FIXED — see §4.**

2. **Bagging is 2.2× slower than no bagging (0.37×).** Halving the rows should
   roughly halve build work; instead it more than doubles total time. Suspect the
   per-iteration bagged-index derivation, not the histogram fold.

3. **DART is 5.6× the baseline (0.24×).** DART re-predicts the dropped subset each
   iteration; the shape suggests an O(trees²) re-prediction rather than an
   incremental score update.

4. **`linear_tree` is 2.7× the baseline (0.39×)** versus LightGBM's 1.3× — the
   per-leaf least-squares fit is the suspect.

5. **`num_leaves=255` degrades to 0.68×** from 0.96× at `num_leaves=15`: the gap
   widens with leaf count, consistent with per-leaf fixed overhead (fork/join,
   per-leaf allocation) rather than per-row work.

### Structural note on the histogram build

`build_histograms_into` allocates a fresh `Vec<f64>` per feature per leaf build
(50 features × 31 leaves × 100 iterations ≈ 155k allocations) and folds
column-wise: for each feature, a scattered gather over that feature's bin column.
LightGBM's dense path uses a **row-major** multi-value bin so one row's bins for
all features are contiguous.

A row-major mirror would remain **bit-exact** provided each thread owns a disjoint
*feature* block and walks rows in ascending `leaf_rows` order — the per-(feature,
bin) accumulation sequence is then unchanged. This is the natural follow-on to
optimization 1 and was not attempted here.

## Reproducing

```bash
# Rust, with optional parameter overrides
cargo build --release -p lgbm --example bench_real
LGBM_BENCH_PARAMS="num_leaves=63,max_bin=127" \
  ./target/release/examples/bench_real <data.tsv> tsv-label0 binary bench

# phase breakdown (build / scan / partition)
LGBM_PHASE_PROF=1 ./target/release/examples/bench_real <data.tsv> tsv-label0 binary bench

# determinism check — must be byte-identical across thread counts
for t in 1 2 4 8; do
  RAYON_NUM_THREADS=$t ./target/release/examples/bench_real <data.tsv> tsv-label0 binary dump-model
done
```

The dataset is 200 000 rows × 50 standard-normal features with a linear-separable
binary label (`numpy.random.default_rng(42)`), written label-first as TSV.

## Caveats

- Single machine, single micro-architecture. The M1's 4P+4E asymmetry penalizes
  fork/join barriers more than a homogeneous x86 box would, so the scaling gap
  measured here is an upper bound on some hardware and a lower bound on others.
- The `par_build_threshold` retune is justified by the sweep above on THIS box;
  the previous value was tuned on a 16-core machine. 1024 was chosen over the
  locally-optimal 0 specifically to bound that risk.
- No GPU numbers: this machine has neither ROCm nor CUDA, so the `cubecl-hip` /
  `cubecl-cuda` paths were not exercised.

---

# 4. `feature_fraction` — build mask + two correctness bugs (2026-08-06)

Chasing the "`feature_fraction` buys nothing" finding from §3 uncovered **three
independent defects**, two of them correctness, not performance.

## 4.1 The build mask (performance, result-neutral)

`build_leaf_histogram_into` passed every feature to `build_leaf_histograms_raw`;
the column-sampling mask gated only the **scan**. Since the build is ~67% of train
wall, the discarded histograms were pure waste.

`Backend::build_leaf_histograms_raw` and `build_histograms_into` now take a `used:
Option<&[bool]>` mask. A masked-out feature costs no gather, no fold and no rayon
task, and its output region stays zeroed.

The mask is C++'s `is_feature_used` (`serial_tree_learner.cpp:387-400`):

```text
is_feature_used[f] = is_feature_used_bytree(f) ∧ parent_histogram[f].is_splittable()
```

It deliberately EXCLUDES the per-node `feature_fraction_bynode` draw. C++ builds
and scans every bytree-selected feature and applies the bynode mask only to the
final split argmax (`ComputeBestSplitForFeature`'s `if (new_split > *best_split &&
is_feature_used)`). Folding bynode into the build would be wrong: the
subtract-derived sibling draws a DIFFERENT bynode mask, so a feature skipped for
one child can still be needed by the other. Skipping is safe precisely because
every consumer gates on a SUBSET of this mask.

**Verified result-neutral**: with the mask forced off, predictions are
BIT-IDENTICAL to the masked build. It removes work without moving a single cell.

## 4.2 `feature_fraction` never reached the tree learner (correctness)

`SerialTreeLearner::with_feature_fraction` was called only from
`learner_parity` — nothing in `lgbm::train*` ever called it. **Every model trained
through the public API or the Python binding ignored `feature_fraction` entirely
and used all columns.** Same class as the `LearnerConstraints` gap: parsed,
validated, then dropped. Now wired in `booster.rs`.

## 4.3 `ColSampler` re-seeded per tree, and off by one draw (correctness)

With sampling finally active, output still diverged from C++ by 0.53 absolute.
Two further defects:

- **Re-seeded per tree.** `ColSampler::new` ran once per tree, each time
  constructing `Random::new(feature_fraction_seed)` — so every tree drew the SAME
  column subset instead of advancing the stream. C++ holds one `ColSampler
  col_sampler_` member (`serial_tree_learner.h:234`) for the whole train. Fixed by
  caching it on the learner beside `hist_pool`.
- **Off by one tree.** C++ draws TWICE before the first tree: `SetTrainingData()`
  ends with `ResetByTree()` (`col_sampler.hpp:51`, called at Init
  `serial_tree_learner.cpp:59`), then `BeforeTrain()` calls `ResetByTree()` per
  tree (`:293`). So **C++ tree N uses draw N+2.** Consuming the Init draw for tree
  0 shifted every subset by one — visibly, Rust's tree N+1 used C++'s tree N
  subset. Fixed by resetting on every tree including the first.

## 4.4 Why this survived so long

The only committed golden, `learner/col_sampler.txt`, pins ONE cell —
`feature_fraction=1.0, feature_fraction_bynode=0.5` — over a SINGLE tree. At
`feature_fraction=1.0` the per-tree draw is a no-op, so the whole
`feature_fraction` path was unverified, and a single-tree fixture cannot see a
per-tree redraw bug at all.

New gate: `crates/oracle-harness/tests/feature_fraction_parity.rs` +
`fixtures/feature_fraction/` — 5 fractions × 8 trees, asserting raw predictions
within `ORACLE_TOL` **and the per-tree selected feature sets tree by tree**.
Confirmed to have teeth: re-introducing the off-by-one makes 2 of 3 tests fail.

## 4.5 Results

Parity vs `lightgbm==4.6.0` (200k × 50, binary, 20 iters), max absolute
prediction difference:

| `feature_fraction` | before | after |
|---|---|---|
| 1.0 (control) | 3.0e-08 | 3.0e-08 |
| 0.75 | — | **3.0e-08** |
| 0.5 | 2.5e-01 | **3.0e-08** |
| 0.25 | — | **3.0e-08** |

Speed (100 iters, all cores):

| `feature_fraction` | train ms | build ms | LightGBM (best) | ratio |
|---|---:|---:|---:|---:|
| 1.0 | 2026 | 1084 | 1805 | 0.89× |
| 0.75 | 1760 | 889 | 1593 | 0.91× |
| 0.5 | **1485** (was 2028) | **696** (was ~1084) | 1381 | **0.93×** (was 0.67×) |
| 0.25 | 1229 | 562 | 1126 | 0.87× |
| 0.1 | 1059 | 451 | — | — |

`feature_fraction` now scales the build roughly with the fraction, as it should.

## 4.6 `feature_fraction_bynode` (fixed 2026-08-06)

A FOURTH defect, in the same family. The port used the per-node mask to **skip the
scan**; C++ scans every bytree-selected feature — running `FixHistogram` +
`FindBestThreshold`, so `is_splittable_` is set from real data — and applies the
per-node mask only to the winner selection:

```cpp
// ComputeBestSplitForFeature, serial_tree_learner.cpp:1000-1002
if (new_split > *best_split && is_feature_used) { *best_split = new_split; }
```

The Rust skip left `this_leaf_splittable[f] = false` for every bynode-excluded
feature, and that false propagates to BOTH children through `parent_splittable`,
permanently removing features C++ would still consider deeper in the tree. It is a
compounding error: invisible at the root, growing with depth.

The same applies to **interaction constraints**, which C++ folds into the very same
`GetByNode` mask (`col_sampler.hpp:91-111`) — so they were mis-gated identically
and are fixed by the same change.

**Fix:** `scan_leaf_histogram` now takes the per-node mask as a separate
`node_used` argument used ONLY by an `argmax_admissible(fpos, real_fidx)`
predicate at the two cross-feature argmax sites. The scan gate is the per-tree
`is_feature_used_bytree` mask alone. `spine_batched_feats` (the batched-scan
membership, which must mirror the scan) was realigned to the same gate.

### Results

Max absolute prediction difference vs `lightgbm==4.6.0` (200k × 50, 20 iters):

| `feature_fraction` | `bynode` | before | after |
|---|---|---|---|
| 1.0 | 1.0 | 3.0e-08 | 3.0e-08 |
| 1.0 | 0.5 | **2.9e-01** | **3.0e-08** |
| 1.0 | 0.25 | — | **3.0e-08** |
| 0.5 | 0.5 | **2.8e-01** | **3.0e-08** |
| 0.75 | 0.25 | — | **3.0e-08** |

Cost, and the sanity check that matters:

| Setting | lgbm_rs | LightGBM | ratio |
|---|---:|---:|---:|
| baseline (no sampling) | 2097 | 1803 | 0.86× |
| `feature_fraction=0.5` | 1480 | 1355 | 0.92× |
| `feature_fraction_bynode=0.5` | 2099 | 1796 | 0.86× |
| both `=0.5` | 1480 | 1384 | 0.94× |

`bynode=0.5` costs the SAME as baseline now — in **both** engines (Rust 2099 vs
2097; LightGBM 1796 vs 1803). That is the correct outcome, not a regression:
`feature_fraction_bynode` is not a work-reduction knob in LightGBM, it only
restricts which split may win. The port previously appeared cheaper here purely
because it was skipping work C++ does.

The golden now spans 9 cells (5 bytree + 4 bynode) × 8 trees. Re-introducing the
scan-skip makes 2 of the 3 tests fail with `abs_diff: 0.256` — verified, not assumed.
