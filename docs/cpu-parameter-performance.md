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

---

# 5. Structural fixes: bagging / GOSS / DART (2026-08-06)

§3 ranked five cells as "structurally worse". Three of them — bagging, GOSS and
DART — turned out to share a **single** root cause, and one more (`feature_fraction`)
was already fixed in §4. This section reports the cause, the fix and the re-measured
matrix.

## 5.1 The cause: per-row `Vec` materialization on every predict-side re-score

Every path that scores rows through `Tree::predict` (rather than the training-path
partition scatter) went through a closure typed `Fn(i32) -> Vec<f64>`. Each call
allocated a fresh row vector:

| Path | Allocations per iteration |
|---|---|
| bagging / GOSS OOB + in-bag scoring | one `Vec<f64>` per scored row (= `num_data`) |
| DART `DroppingTrees` | `num_data` for the snapshot **plus** `num_data` per dropped tree (the `.clone()` in the closure) |
| DART `Normalize` | same again |
| RF `rf_update_score` | `num_data` |

On the 200k-row workload DART allocated on the order of 10⁷ `Vec<f64>` **per
iteration**. That is the whole of its 5.6× penalty — not, as §3 guessed, an
O(trees²) re-prediction.

**Fix:** `Gbdt` now holds one lazily-built ROW-MAJOR `(Vec<f64>, usize)` dense view
of the feature columns (`Gbdt::dense_rows`), built at most once per train because
`features` is immutable after construction. `ScoreUpdater` gained
`add_tree_predict_path_dense` / `add_tree_scaled_all_dense`, which borrow
`dense[row*width .. (row+1)*width]` instead of receiving an owned vector. Same
values, same order, same arithmetic — a representation change only.

## 5.2 A second cause, bagging-only: the subset gather

`SerialTreeLearner::train_on_subset{,_returning_partition}` rebuilds the C++
`tmp_subset_` each iteration. Two defects:

1. The `in_bag` → `u32` row-id widening sat **inside the per-feature closure**, so a
   `bag_size`-long `Vec<u32>` was allocated and refilled once **per feature** — 50×
   the necessary work on a 50-feature corpus.
2. The gather ran serially, though each feature's gather is independent.

Both are fixed in the new shared `gather_subset_features`, which hoists the widening
and parallelizes over features above a 4096-cell threshold. Result-neutral: each
column's gather is a pure elementwise permutation, and `map` is order-preserving.

## 5.3 Re-measured matrix

Same machine, corpus and method as §3. `before` / `after` are the same binary built
either side of §5.1-5.2; LightGBM is re-timed in the same session at its best
`num_threads ∈ {1,4,8}`.

| Parameter setting | before (ms) | after (ms) | LightGBM (ms) | ratio before | ratio after |
|---|---:|---:|---:|---:|---:|
| baseline (`num_leaves=31 max_bin=255`) | 2136 | 2038 | 1791 | 0.84× | 0.88× |
| `bagging_freq=1 bagging_fraction=0.5` | 4548 | **2605** | 1700 | 0.37× | **0.65×** |
| `boosting=dart` | 11305 | **3853** | 2795 | 0.25× | **0.73×** |
| `boosting=goss` | 3981 | **2452** | 1606 | 0.40× | **0.65×** |
| `num_leaves=15` | 1369 | 1463 | 1294 | — | 0.88× |
| `num_leaves=63` | 3105 | 3121 | 2549 | — | 0.82× |
| `num_leaves=255` | 7915 | 8110 | 5340 | — | 0.66× |
| `max_bin=15` | 1623 | 1623 | 1255 | — | 0.77× |
| `max_bin=63` | 1704 | 1695 | 1323 | — | 0.78× |
| `max_bin=511` | 3081 | 2959 | 2382 | — | 0.80× |
| `max_depth=6` | 1737 | 1759 | 1596 | — | 0.91× |
| `min_data_in_leaf=100` | 2102 | 2031 | 1789 | — | 0.88× |
| `feature_fraction=0.5` | 1482 | 1479 | 1344 | — | 0.91× |
| `feature_fraction_bynode=0.5` | 2038 | 2039 | 1793 | — | 0.88× |
| `extra_trees=true` | 1814 | 1826 | 1736 | — | 0.95× |
| `linear_tree=true` | 5499 | 5478 | 2140 | — | **0.39×** |
| `use_quantized_grad=true` | 2147 | 2145 | 1439 | — | 0.67× |
| `lambda_l1=0.1` | 2206 | 2225 | 1858 | — | 0.84× |
| `lambda_l2=1.0` | 2067 | 2083 | 1781 | — | 0.85× |
| `min_gain_to_split=0.1` | 2063 | 2086 | 1820 | — | 0.87× |
| `is_unbalance=true` | — | 2077 | — | — | — |
| `scale_pos_weight=3` | — | 2033 | — | — | — |

Speedups: **DART 2.9×, bagging 1.75×, GOSS 1.6×.** Untouched cells move by ±5%,
which is this box's run-to-run noise — read those columns as unchanged, not as
regressions. `is_unbalance` / `scale_pos_weight` (newly wired, §6) cost nothing
measurable: one extra multiply per row.

## 5.4 What is left

`linear_tree` (0.39×) is now the only cell that is structurally worse, and it does
NOT share the cause above — its cost is the per-leaf normal-equation accumulation in
`fit_linear_leaves` (O(rows × dim²) with `dim = path features + 1`). Undiagnosed.

For the ordinary tuning knobs the port sits at **0.77-0.95× of LightGBM**, i.e.
5-23% slower. `LGBM_PHASE_PROF=1` on the baseline attributes that to:

```
build=1087ms (59%)   scan=328ms (18%)   partition=379ms (21%)
```

The histogram BUILD is the whole gap and the §3 "structural note" still names the
fix: LightGBM's dense path uses a **row-major multi-value bin**, so one row's bins
for all features are contiguous; `build_histograms_into` instead does one scattered
gather per feature per leaf. On a leaf of ~6.5k rows scattered over 200k, the
column-wise form touches ~50 cache lines per row where a row-major form touches ~1.
This is memory-latency-bound, which is consistent with the measured ~1.7 ns per
histogram cell.

The row-major mirror stays **bit-exact** provided each thread owns a disjoint
*feature* block and walks rows in ascending `leaf_rows` order — the per-(feature,bin)
accumulation sequence is then unchanged. It was not attempted here: it is a
layout change to the dataset representation, not a local edit, and needs its own
parity gate.

Re-tuning `par_build_threshold` is exhausted as a lever — re-swept on the current
tree, `0 / 256 / 1024` give `2032 / 2057 / 2033` ms. The default stays 1024.

---

# 6. Parameter implementation status (2026-08-06)

An audit of all 130 `Config` fields for reads outside `lgbm-core` found four
parameters that parsed, validated, and then did nothing. All four are now wired and
gated by tests.

| Parameter | Was | Now |
|---|---|---|
| `is_unbalance` | `Binary` hard-coded `label_weight = 1.0` — the parameter trained the balanced model | `Binary::with_class_weights` derives C++ `label_weights_` (`binary_objective.hpp:85-100`); `MulticlassOva` builds one weighted `Binary` per class, as C++ does. Gate: `oracle-harness/tests/class_weight_parity.rs` (8 cells vs `lightgbm==4.6.0`) |
| `scale_pos_weight` | same | same; the C++ "cannot set is_unbalance and scale_pos_weight at the same time" fatal is a typed error |
| `categorical_feature` | only `RawCorpus::categorical_features` was honoured; the PARAMETER binned every column numeric | parsed per `dataset_loader.cpp:168-189`; the `name:` form is an explicit error (an in-memory corpus has no column names). Gate: `lgbm/tests/param_wiring.rs` |
| `bagging_by_query` | rejected outright — the facade dropped the corpus's query boundaries | forwards them and selects the query-grouped draw; still a typed error for an ungrouped corpus. Gate: `lgbm/tests/param_wiring.rs` |

Also wired: `objective=lambdarank` / `objective=rank_xendcg`, which the boosting
layer implemented but `lgbm::booster::resolve_objective` had no arm for. This
removes all 15 `UNSUPPORTED` entries from the string-parameter oracle sweep
(`string_param_parity.rs`), which now reports **0 documented-unsupported** across
129 cells.

## Still not implemented

`max_delta_step` and `path_smooth` return a typed `ComputeError` from the split
kernels ("only the default 0.0 path is transcribed") rather than a wrong model. The
gain math for both already exists in `lgbm_compute::gain`; what is missing is
threading the per-candidate `num_data` and `parent_output` through the five
`find_best_split*` CubeCL kernel signatures. Not attempted here.

The remaining out-of-scope parameters are unchanged and documented in
`lgbm_core::config::scope` (distributed learning, OpenCL device tuning) or are
CLI / dataset-loader-layer file and column specs, enumerated with their reason in
the string-param golden's `skipped` map.

---

# 7. `max_delta_step` and `path_smooth` implemented (2026-08-06)

§6 listed both as the only remaining unimplemented parameters — they returned a typed
`ComputeError` ("only the default 0.0 path is transcribed") from six rejection sites in
the split kernels. Both are now implemented.

## 7.1 What they are

They are the C++ `USE_MAX_OUTPUT` and `USE_SMOOTHING` template axes of
`FeatureHistogram`, selected once per feature histogram from the config
(`feature_histogram.hpp:248-270`): `max_delta_step > 0` and `path_smooth > kEpsilon`.

```text
ret = -ThresholdL1(g, l1) / (h + l2)                          // base
if USE_MAX_OUTPUT && max_delta_step > 0 && |ret| > max_delta_step:
    ret = Sign(ret) * max_delta_step                          // clamp
if USE_SMOOTHING:
    ret = ret*n/ps / (n/ps + 1) + parent_output / (n/ps + 1)  // blend toward parent
```

Either axis ALSO switches the gain FORM: `GetLeafGain` stops returning the closed form
`sg²/(h+λ)` and returns `GetLeafGainGivenOutput` evaluated at the computed output.
They are equal in exact arithmetic and differ in floating point, so the switch is
observable even when the clamp never binds.

## 7.2 What had to be threaded

`path_smooth` needs two things the split path did not carry: each candidate side's
OWN row count, and the leaf's `parent_output`
(C++ `SerialTreeLearner::GetParentOutput`). The counts were already computed in every
scan. `parent_output` now rides in `GainConfig` — C++ passes `(config, parent_output)`
as a pair into every gain call, and bundling them avoids a signature change at ~90
sites. The two co-packed-sibling paths are the exception: siblings have DIFFERENT
parent outputs, so theirs travel in the per-leaf totals tuple.

`LeafSplits::weight` supplies it and already encodes both branches of
`GetParentOutput` without a `tree->num_leaves() == 1` test: the root carries its own
clamped, un-smoothed output (`init`), a child carries the output its parent split
assigned it (`init_from_split`).

Converted: the shared `#[cube] split_scan_body` (and the five launch kernels that call
it), the native host scan `find_best_split_cpu_native`, the categorical finder, the
monotone finder, the extra-trees randomized finder, the forced-split threshold
gatherer, and the snapshot re-scan.

## 7.3 A latent bug the axes exposed

Both scan bodies initialized `best_gain = 0.0` instead of C++'s `kMinScore` (-inf),
justified by "every valid gain is non-negative because gains are `g²/(h+λ)` sums".
That justification dies under the given-output form, which is freely negative. A leaf
whose every candidate had negative gain kept `best_gain = 0.0` and `best_threshold = 0`
while `is_splittable` still went true, so the launcher reported a BOGUS split at bin 0
with net gain `0 - min_gain_shift` — large and POSITIVE whenever `min_gain_shift` is
negative, which then won the best-first race. The host scan now uses `-inf` directly;
the `#[cube]` body uses a `has_best` flag because cubecl-cpu requires loop-carried
mutables to init from literals.

## 7.4 Verification

New gate: `crates/oracle-harness/tests/path_smooth_parity.rs` +
`fixtures/path_smooth/` — 13 cells vs `lightgbm==4.6.0` spanning both axes, their
combination, and their interaction with `lambda_l1`, over 16-leaf trees (smoothing
CHAINS through depth, so a stump cannot see a `parent_output` error). It asserts raw
predictions AND the per-tree leaf values, plus that every active cell's C++ model
differs from the all-defaults control (so the sweep cannot pass against an
implementation that ignores the parameters).

**All 13 cells reach parity within `ORACLE_TOL`; the whole pre-existing suite is
unchanged and green.**

### 7.4.1 The one cell that needed `-ffp-contract` parity

One cell — `max_delta_step = 0.05`, no smoothing — initially diverged, and running it
down produced the most interesting finding of this work.

That setting is mathematically DEGENERATE on this corpus: it binds so tightly that at
many candidate thresholds BOTH children clamp to the same output, and then

```text
GetLeafGain(left) + GetLeafGain(right) == GetLeafGain(whole leaf) == gain_shift
```

holds EXACTLY in real arithmetic. The split's net gain is exactly zero, so
`is_splittable_` turns on a strict `>` between two differently-associated
floating-point sums. The port landed ONE ULP above (`1.78e-15` on a gain of `8.85`);
the reference landed exactly at zero. C++ then propagates non-splittability to BOTH
children (`serial_tree_learner.cpp:390-395`), so that single bit deleted a feature from
an entire subtree and moved predictions by 0.01.

**Cause: `-ffp-contract`.** The reference is built by clang/gcc with contraction ON
(the C++ default), so `2·g·o + (h+λ)·o²` is emitted as a single fused multiply-add —
one rounding, not two. Rust never auto-contracts. This is invisible everywhere else in
the port because the default gain formula, `GetLeafGain`'s closed form `sg²/(h+λ)`, is
a multiply and a divide with no add to contract into; it appears the moment
`max_delta_step` / `path_smooth` switch the gain to `GetLeafGainGivenOutput`.

**WHICH multiply is fused was measured, not guessed.** Because the degenerate cell
makes the three candidate formulations differ by exactly one bit, replaying the
reference's own operands through each is decisive:

| formulation | candidate − shift |
|---|---|
| `-(2·g·o + (h+λ)·o²)` — no contraction | +1 ULP |
| `-fma((h+λ)·o, o, 2·g·o)` — fuse the SECOND multiply | +1 ULP |
| `-fma(2·g, o, (h+λ)·o²)` — fuse the FIRST multiply | **0, matching the reference** |

Adopting the third made the whole tree match node-for-node and gain-for-gain, took the
sweep from 12/13 to 13/13, and left every pre-existing golden untouched — including the
monotone-constraint goldens, which call the same `GetLeafGainGivenOutput`.

**Mechanism.** `gain::fused_mul_add` is a genuine single-rounding multiply-add usable
from BOTH a `#[cube]` kernel and host code: `f64::mul_add` on the host, cubecl's `fma`
IR instruction on device, paired via the plain-fn + `mod ::expand` idiom cubecl uses
for its own intrinsics (cubecl's `fma` alone would `unexpanded!()`-panic on the host,
and `f64::mul_add` alone has no `__expand`).

**Caveat, recorded deliberately.** This ties the port to a reference built with
contraction ENABLED. A LightGBM built with `-ffp-contract=off`, or for a target without
FMA, would not contract and the port would then be the one that is one ULP off. That is
an accepted trade: contraction-on is the C++ default and `lightgbm==4.6.0` from PyPI is
the reference every fixture in `oracle-harness` is captured against.
`path_smooth_parity` fails loudly if that ever stops holding, and
`lgbm-compute/tests/gain_full_probe.rs::the_given_output_gain_is_fused_exactly_as_the_reference_is`
pins the fusion directly against the reference's own operand bits (verified to FAIL if
the fusion is reverted), so a regression names its own cause instead of surfacing as a
mismatched tree.

## 7.5 GPU scan variants

The LDS-staged / official / pargain / parprefix scan kernels (`--features gpu`) are
performance forks of `split_scan_body`, each re-transcribing the scan against a
different memory layout and each implementing only the default gain. Rather than fork
the new semantics into five more bodies that no GPU-less CI can execute, they DECLINE
via `scan_variants_applicable(cfg)` when either axis is active, and the launcher falls
through to the shared body — the reference every variant must reproduce anyway. So the
parameters work on every backend; only the optimization opts out. `--features gpu`
compiles clean (`--all-targets`).
