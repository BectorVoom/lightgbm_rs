# LightGBM CUDA Kernels — Design Document

**Scope.** This document describes the *complete* public CUDA backend of LightGBM
(`device_type=cuda`): every `__global__` kernel and the host orchestration around
it, across the whole training and prediction pipeline. It covers ~40 kernels in
15 `.cu` files plus the shared device-primitive and utility headers. It is written
as the porting reference for the pure-Rust/CubeCL rewrite, so each subsystem lists
exact host/device boundaries, input/output types, launch geometry, and algorithm.

The reference tree is read-only C++ under `LightGBM/`. File map:

| Subsystem | File(s) | Lines |
|-----------|---------|-------|
| Shared device primitives | `src/cuda/cuda_algorithms.cu`, `include/LightGBM/cuda/cuda_utils.hu` | 512 + 212 |
| Dataset on device | `src/io/cuda/cuda_column_data.cu` | 61 |
| Tree model I/O | `src/io/cuda/cuda_tree.cu` | 459 |
| Gradient discretizer | `src/treelearner/cuda/cuda_gradient_discretizer.cu` | 171 |
| Objective functions | `src/objective/cuda/cuda_{regression,binary,multiclass,rank}_objective.cu` | 481+209+108+661 |
| Metrics | `src/metric/cuda/cuda_pointwise_metric.cu` | 78 |
| Score updater | `src/boosting/cuda/cuda_score_updater.cu` | 45 |
| Leaf splits + tree-learner driver | `src/treelearner/cuda/cuda_leaf_splits.cu`, `cuda_single_gpu_tree_learner.cu` | 245 + 293 |
| Histogram constructor | `src/treelearner/cuda/cuda_histogram_constructor.cu` | 960 |
| Best split finder | `src/treelearner/cuda/cuda_best_split_finder.cu` | 2239 |
| Data partition | `src/treelearner/cuda/cuda_data_partition.cu` | 1121 |

---

## Complete CUDA File Inventory (all 58 files)

The CUDA backend is **58 files** across seven directories. The `.cu` files hold
`__global__` kernels (documented in §1–§12); the `.cpp`/`.hpp`/`.hu`/`.h` files are
host orchestration, device structs, and device-only helpers (documented in §13–§17
and §2.4). Every file is accounted for below; "§" is where it is covered.

**`include/LightGBM/cuda/` (11)** — public headers / device structs

| File | Coverage |
|------|----------|
| `cuda_algorithms.hpp` | §2.4 — `__device__` scans/reductions/argsort + a few `__global__` wrappers |
| `cuda_utils.hu` | §2.4 — memory helpers, `CUDAVector`, `atomic_add_long_t`, `SafeLog` |
| `cuda_row_data.hpp` | §13 — device binned matrix + partition layout |
| `cuda_column_data.hpp` | §14 — `CUDAColumnData` decl |
| `cuda_metadata.hpp` | §14 — `CUDAMetadata` decl |
| `cuda_metric.hpp` | §12 — `CUDAMetricInterface` base |
| `cuda_objective_function.hpp` | §14 — `CUDAObjectiveInterface` base |
| `cuda_tree.hpp` | §14 — `CUDATree` decl + `__device__` decision-type helpers |
| `cuda_split_info.hpp` | §17 — `CUDASplitInfo` struct |
| `cuda_random.hpp` | §17 — `CUDARandom` LCG |
| `vector_cudahost.h` | §14 — `CHAllocator` (pinned host mem), `LGBM_config_` |

**`src/cuda/` (2)** | `cuda_algorithms.cu` → §2.4 · `cuda_utils.cpp` → §14

**`src/io/cuda/` (6)** — device data & model I/O

| File | Coverage |
|------|----------|
| `cuda_column_data.cu` | §3 — `CopySubrowKernel_ColumnData` |
| `cuda_column_data.cpp` | §14 — host column store |
| `cuda_row_data.cpp` | §13 — partition layout build/upload |
| `cuda_metadata.cpp` | §14 — label/weight/query upload |
| `cuda_tree.cu` | §10 — Split/Shrinkage/prediction kernels |
| `cuda_tree.cpp` | §14 — host tree, `ToHost`/sync |

**`src/boosting/cuda/` (3)** | `cuda_score_updater.cu` → §11 · `.cpp` → §14 · `.hpp` → §11

**`src/metric/cuda/` (7)** | `cuda_pointwise_metric.cu` → §12 · `{pointwise,binary,regression}_metric.{hpp,cpp}` → §12 (device losses)

**`src/objective/cuda/` (12)** | `cuda_{regression,binary,multiclass,rank}_objective.cu` → §5 · matching `.hpp`/`.cpp` → §5 / §14

**`src/treelearner/cuda/` (17)** — the tree-learning core

| Triplet (`.cu` / `.cpp` / `.hpp`) | Coverage |
|-----------------------------------|----------|
| `cuda_single_gpu_tree_learner.*` | §6 — driver |
| `cuda_leaf_splits.*` | §6 — root init |
| `cuda_histogram_constructor.*` | §7 — histogram build |
| `cuda_best_split_finder.*` | §8 — split finding |
| `cuda_data_partition.*` | §9 — row partition |
| `cuda_gradient_discretizer.{cu,hpp}` (no `.cpp`) | §4 — quantization |

---

## 0. Per-Kernel Quick Index

Every `__global__` kernel, by subsystem. "Geometry" is the launch config; closely
related template variants are grouped on one row. `§` links to the detailed entry.

### Shared device primitives (§2.4) — `cuda_algorithms.cu`, `cuda_utils.hu`

| Primitive | Kind | Role |
|-----------|------|------|
| `ShufflePrefixSum<T>` / `…Exclusive<T>` | `__device__` | block-wide warp-shuffle scan |
| `GlobalMemoryPrefixSum<T>`, `ShufflePrefixSumGlobal<T>` | `__device__` / host | global-array scan |
| `GlobalInclusiveArgPrefixSum<…>` | host | scan over index-gathered values |
| `ShuffleReduceSum/Max/Min<T>`, `BlockReduceSum<T>` | `__device__`/`__global__` | block reductions |
| `ShuffleReduceSum/Min/DotProdGlobal<…>` | host | multi-block scalar reductions |
| `BitonicArgSort_1024/2048`, `…Device`, `…Global`, `…ItemsGlobal` | `__device__`/host | index argsort (values unmoved) |
| `PercentileDevice<…>` / `PercentileGlobal<…>` | `__device__` / host | (weighted) α-quantile |

### Dataset & discretizer (§3–§4)

| Kernel | Geometry | File | § | Role |
|--------|----------|------|---|------|
| `CopySubrowKernel_ColumnData` | `ceil(N/1024)×1024` | `cuda_column_data.cu` | §3 | gather binned subset rows |
| `ReduceMinMaxKernel` | `nblk×1024` | `cuda_gradient_discretizer.cu` | §4 | per-block grad/hess extrema |
| `ReduceBlockMinMaxKernel` | `1×1024` | `cuda_gradient_discretizer.cu` | §4 | fold extrema → quant scales |
| `DiscretizeGradientsKernel<STOCHASTIC>` | `nblk×1024` | `cuda_gradient_discretizer.cu` | §4 | quantize grad/hess → int16 |

### Objectives (§5)

| Kernel | Geometry | File | § | Role |
|--------|----------|------|---|------|
| `GetGradientsKernel_Regression{L2,L1,Huber,Fair,Poisson,Quantile}<USE_WEIGHT>` | `ceil(N/1024)×1024` | `cuda_regression_objective.cu` | §5.1 | per-row grad/hess |
| `ConvertOutputCUDAKernel_Regression{,_Poisson}` | `ceil(N/1024)×1024` | `cuda_regression_objective.cu` | §5.1 | inverse link (sqrt/exp) |
| `RenewTreeOutputCUDAKernel_Regression{L1,Quantile}<USE_WEIGHT>` | `num_leaves × ½–¼ block` | `cuda_regression_objective.cu` | §5.1 | per-leaf median refit |
| `GetGradientsKernel_BinaryLogloss<USE_LABEL_WEIGHT,USE_WEIGHT>` | `ceil(N/1024)×1024` | `cuda_binary_objective.cu` | §5.2 | sigmoid grad/hess |
| `BoostFromScoreKernel_1/2_BinaryLogloss<USE_WEIGHT>` | `nblk×1024` / `1×1` | `cuda_binary_objective.cu` | §5.2 | init score (label/weight sums) |
| `ConvertOutputCUDAKernel_BinaryLogloss` | `ceil(N/1024)×1024` | `cuda_binary_objective.cu` | §5.2 | sigmoid probability |
| `ResetOVACUDALabelKernel` | `ceil(N/1024)×1024` | `cuda_binary_objective.cu` | §5.2 | one-vs-all label rewrite |
| `GetGradientsKernel_MulticlassSoftmax<USE_WEIGHT>` | `ceil(N/1024)×1024` | `cuda_multiclass_objective.cu` | §5.3 | softmax grad/hess (class-major) |
| `ConvertOutputCUDAKernel_MulticlassSoftmax` | `ceil(N/1024)×1024` | `cuda_multiclass_objective.cu` | §5.3 | per-row softmax probs |
| `GetGradientsKernel_LambdarankNDCG<…>{,_Sorted}` | `ceil(Q/10) × ≤2048` | `cuda_rank_objective.cu` | §5.4 | pairwise lambda grad/hess |
| `GetGradientsKernel_RankXENDCG_{SharedMemory,GlobalMemory}` | `ceil(Q/10) × ≤2048` | `cuda_rank_objective.cu` | §5.4 | cross-entropy NDCG grad/hess |

### Leaf splits & tree-learner driver (§6)

| Kernel | Geometry | File | § | Role |
|--------|----------|------|---|------|
| `CUDAInitValuesKernel1/2<USE_INDICES>` | `nblk×1024` / `1×1024` | `cuda_leaf_splits.cu` | §6.1 | root sums → leaf struct |
| `CUDAInitValuesKernel3/4<USE_INDICES>` | `nblk×1024` / `1×1024` | `cuda_leaf_splits.cu` | §6.1 | root sums, quantized |
| `InitValuesEmptyKernel` | `1×1` | `cuda_leaf_splits.cu` | §6.1 | empty-leaf sentinel |
| `ReduceLeafStatKernel_{SharedMemory,GlobalMemory}` | `ceil(N/1024)×1024` | `cuda_single_gpu_tree_learner.cu` | §6.2 | per-leaf grad/hess sums |
| `CalcRefitLeafOutputKernel<USE_L1,USE_SMOOTHING>` | `ceil(L/1024)×1024` | `cuda_single_gpu_tree_learner.cu` | §6.2 | refit leaf values |
| `SetRealThresholdKernel` | `ceil(C/1024)×1024` | `cuda_single_gpu_tree_learner.cu` | §6.3 | inner-bin → real category |
| `CalcBitsetLenKernel<T,IS_INNER>` / `ReduceBlockMaxLen` | `nblk×1024` / `1×1024` | `cuda_single_gpu_tree_learner.cu` | §6.3 | categorical bitset length |
| `CUDAConstructBitsetKernel<T,IS_INNER>` | `ceil(C/1024)×1024` | `cuda_single_gpu_tree_learner.cu` | §6.3 | set category bits |

### Histogram constructor (§7) — `cuda_histogram_constructor.cu`

| Kernel | Geometry | § | Role |
|--------|----------|---|------|
| `CUDAConstructHistogram{Dense,Sparse}Kernel[_GlobalMemory]<…>` | `(parts, gridY)×(cols,504/cols)` | §7.1 | build smaller-leaf histogram |
| `CUDAConstructDiscretizedHistogram{Dense,Sparse}Kernel[_GlobalMemory]<…>` | same | §7.1 | build, quantized (int) |
| `FixHistogramKernel` / `FixHistogramDiscretizedKernel<16bit?>` | `n_fix_feat × 512` | §7.3 | reconstruct most-freq bin |
| `SubtractHistogramKernel` | `ceil(2·bins/1024)×1024` | §7.3 | larger = parent − smaller |
| `SubtractHistogramDiscretizedKernel<S,L,P 16bit?>` / `CopyChangedNumBitHistogram` | `ceil(bins/1024)×1024` | §7.3 | subtract, quantized |

### Best split finder (§8) — `cuda_best_split_finder.cu`

| Kernel | Geometry | § | Role |
|--------|----------|---|------|
| `FindBestSplitsForLeafKernel<USE_RAND,USE_L1,USE_SMOOTHING,IS_LARGER>` | `num_tasks × 256` | §8.1 | per-feature best threshold |
| `FindBestSplitsDiscretizedForLeafKernel<…>` | `num_tasks × 256` | §8.1 | per-feature, quantized |
| `FindBestSplitsForLeafKernel_GlobalMemory<…>` | `num_tasks × 256` | §8.1 | per-feature, low-VRAM |
| `SyncBestSplitForLeafKernel` / `…AllBlocks` | `nblk_per_leaf × 1024` / `1×1` | §8.2 | cross-feature argmax per leaf |
| `SetInvalidLeafSplitInfoKernel` | `1×1` | §8.2 | mark leaf split invalid |
| `FindBestFromAllSplitsKernel` | `1 × 256` | §8.3 | cross-leaf argmax |
| `PrepareLeafBestSplitInfo` | `6 × 1` | §8.3 | pack 8-int result buffer |
| `AllocateCatVectorsKernel` / `InitCUDARandomKernel` | `ceil(n/256)×256` | §8.3 | setup: cat slabs / RNG seed |

### Data partition (§9) — `cuda_data_partition.cu`

| Kernel | Geometry | § | Role |
|--------|----------|---|------|
| `FillDataIndicesBeforeTrainKernel` / `FillDataIndexToLeafIndexKernel` | `ceil(N/1024)×1024` | §9 | root index/leaf-map init |
| `GenDataToLeftBitVectorKernel<…,BIN_TYPE>{,_Categorical}` | `grid_dim_×block_dim_` | §9 | mark left/right + block scan |
| `UpdateDataIndexToLeafIndexKernel<…,BIN_TYPE>{,_Categorical}` | `grid_dim_×block_dim_` | §9 | write row→leaf map |
| `AggregateBlockOffsetKernel0/1` | `1×1024` | §9 | cross-block prefix → leaf bounds |
| `SplitInnerKernel` | `grid_dim_×block_dim_` | §9 | scatter rows to child ranges |
| `SplitTreeStructureKernel` | `4×5` | §9 | child roles + hist-pool swap |
| `CopyDataIndicesKernel` | `grid_dim_×block_dim_` | §9 | write reordered indices back |
| `AddPredictionToScoreKernel<USE_BAGGING>` | `ceil(N/1024)×1024` | §9 | score += leaf output |
| `RenewDiscretizedTreeLeavesKernel` | `num_leaves × 1024` | §9 | recompute leaf stats (quantized) |

### Tree model I/O & score updater & metrics (§10–§12)

| Kernel | Geometry | File | § | Role |
|--------|----------|------|---|------|
| `SplitKernel` / `SplitCategoricalKernel` | `3×5` / `3×6` | `cuda_tree.cu` | §10 | mutate tree node |
| `ShrinkageKernel` / `AddBiasKernel` | `ceil(L/1024)×1024` | `cuda_tree.cu` | §10 | leaf `×rate` / `+bias` |
| `AddPredictionToScoreKernel<USE_INDICES>` | `ceil(N/tpb)×tpb` | `cuda_tree.cu` | §10 | **prediction** (tree walk) |
| `AddScoreConstantKernel` / `MultiplyScoreConstantKernel` | `ceil(N/tpb)×tpb` | `cuda_score_updater.cu` | §11 | whole-array `+= / *=` |
| `EvalKernel<CUDA_METRIC,USE_WEIGHTS>` | `ceil(N/1024)×1024` | `cuda_pointwise_metric.cu` | §12 | per-row loss + reduce |

*(N = num_data, L = num_leaves, C = num_cat_threshold, Q = num_queries, tpb = block size.)*

---

## 1. Pipeline Overview

GPU training keeps **all** training state resident on the device — gradients,
histograms, split records, the row-index permutation, the tree model, and the
cumulative score array. Two host layers orchestrate, doing no per-row math: the
**boosting layer** (`GBDT`, `gbdt.cpp`) drives gradient computation, shrinkage,
score update, and metric eval around each tree; the **tree learner**
(`CUDASingleGPUTreeLearner::Train`) drives the per-leaf grow loop. Both only size
grids, resolve template specializations from config, and sequence kernel launches
on CUDA streams. Only a handful of scalars (best-split metadata, bitset lengths)
cross back to the host per iteration.

Sequence verified against `GBDT::TrainOneIter` (`gbdt.cpp:230,403,413,415`) and
`CUDASingleGPUTreeLearner::Train`:

```
 ┌─────────────────────────────── one boosting iteration ─────────────────────────────────────┐
 │  [boosting layer — GBDT]                                                                     │
 │  Objective.GetGradients(scores) ─► gradients/hessians  (f32, per row)                        │
 │        ▼                                                                                      │
 │  tree_learner.Train(grad, hess):   [CUDASingleGPUTreeLearner]                                 │
 │    BeforeTrain ; (optional) GradientDiscretizer.Discretize ─► int16 packed grad|hess          │
 │    LeafSplits.InitValues  ─► root CUDALeafSplitsStruct (sum grad/hess, gain, value)          │
 │    ┌──────────── per leaf, repeated up to num_leaves-1 times (break on best_leaf == -1) ──┐  │
 │    │  HistogramConstructor.ConstructHistogramForLeaf (smaller leaf)                        │  │
 │    │  HistogramConstructor.SubtractHistogramForLeaf  (larger = parent − smaller)           │  │
 │    │  BestSplitFinder.FindBestSplitsForLeaf  (per-feature, smaller & larger)               │  │
 │    │  BestSplitFinder.FindBestFromAllSplits  (cross-leaf argmax → 8-int copy-back)         │  │
 │    │  (categorical) ConstructBitsetForCategoricalSplit                                      │  │
 │    │  CUDATree.Split / SplitCategorical   (mutate the on-device tree structure) ───┐        │  │
 │    │  DataPartition.Split  (reorder row indices into two contiguous child ranges) ◄┘ FIRST  │  │
 │    └───────────────────────────────────────────────────────────────────────────────────────┘  │
 │    (optional, quantized) RenewDiscretizedTreeLeaves                                           │
 │        ▼                                                                                      │
 │  [boosting layer — GBDT]                                                                      │
 │  Tree.Shrinkage(rate) ; ScoreUpdater.UpdateScore (cumulative score += leaf outputs)          │
 │  (optional) Objective.RenewTreeOutput (L1/quantile leaf-value refit)                         │
 │  Metric.Eval (train/validation loss)                                                         │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Ordering note.** `CUDATree.Split` runs **before** `DataPartition.Split`
> (`Train` lines ~140 then ~148): the tree mutation returns `right_leaf_index`,
> which the partition step then uses when assigning rows to the two child slices.

---

## 2. Common Conventions

### 2.1 Scalar types (the numerical contract)

| Type | Definition | Role |
|------|------------|------|
| `score_t` | `float` | gradient / hessian element, per-row randoms |
| `label_t` | `float` | label / weight element |
| `hist_t` | `double` | **histogram accumulator** (Σgrad, Σhess per bin) |
| `int_hist_t` | `int32_t` | packed integer histogram element (quantized path) |
| `data_size_t` | `int32_t` | row index / count |
| `atomic_add_long_t` | `unsigned long long` | 64-bit `atomicAdd` operand (quantized 32-bit hist) |

**Scores are `double`** end-to-end on device (the f64 accumulator), even though
gradients/hessians and labels are `f32`. This is the source of the project's
"f64-fold CPU anchor is bit-exact, ROCm f32 is ~1e-6" parity split.

### 2.2 Host vs. device split (applies to every subsystem)

* **Host** (the `CUDA*` C++ classes): owns all device buffers (via `CUDAVector<T>`
  / raw `AllocateCUDAMemory`), computes grid/block dims, resolves compile-time
  template flags from runtime `Config`, launches kernels on named streams, and
  issues the few `CopyFromCUDADeviceToHost(Async)` transfers. It performs *no*
  per-row arithmetic.
* **Device** (`__global__` kernels): all per-row / per-bin / per-leaf math,
  reductions, scans, sorts, and scatters.

### 2.3 Kernel-launch idioms

* **One thread per row** for element-wise passes (objectives, metrics, score add),
  `num_blocks = ceil(num_data / 1024)`, block size 1024.
* **Two-level reduction** for scalars: a per-block partial via warp-shuffle
  (`ShuffleReduceSum/Max/Min`), then a single-block fold (`*Global`) or
  `atomicAdd_system` into one cell.
* **Template explosion** instead of runtime branching in hot loops: bin width
  (`uint8/16/32`), row-pointer width, weighted/unweighted, L1/smoothing,
  16-bit/32-bit histogram, smaller/larger leaf, global-vs-shared memory — all are
  compile-time `bool`/type parameters dispatched through nested `Inner0/1/2…`
  host helpers.

### 2.4 Shared device primitives (`cuda_algorithms.*`, `cuda_utils.hu`)

The building blocks every other subsystem reuses. Tunables:
`GLOBAL_PREFIX_SUM_BLOCK_SIZE=1024`, `BITONIC_SORT_NUM_ELEMENTS=1024`,
`BITONIC_SORT_DEPTH=11`.

* **Prefix sums.** `ShufflePrefixSum<T>` / `…Exclusive<T>` — block-wide warp-shuffle
  scan (per-warp totals staged in a 32-slot shared buffer). `GlobalMemoryPrefixSum<T>`
  scans a global array within one block. Host wrappers `ShufflePrefixSumGlobal<T>`
  (3-kernel full-array scan; instantiated `uint16/32/64`) and
  `GlobalInclusiveArgPrefixSum<label_t,double,data_size_t>` (scan over values
  gathered through a sorted-index array).
* **Reductions.** `ShuffleReduceSum/Max/Min<T>` (block), `BlockReduceSum<T>`
  (`__global__` fold). Host wrappers `ShuffleReduceSumGlobal<VAL,REDUCE>`,
  `ShuffleReduceMinGlobal`, `ShuffleReduceDotProdGlobal` (per-block reduce then
  single-block final).
* **Bitonic argsort.** `BitonicArgSort_1024/2048<VAL,IDX,ASC>` (single block,
  ≤1024/2048), `BitonicArgSortDevice<…,BLOCK_DIM,MAX_DEPTH>` (general, shared for
  low levels + global for high), multi-block `BitonicArgSortGlobal<…>` (specialized
  `<double,data_size_t,{true,false}>`, `<label_t,data_size_t,false>`), and
  `BitonicArgSortItemsGlobal` (per-query sort for ranking). Sorts an *index* array;
  never moves the values.
* **Percentile.** `PercentileDevice<…,USE_WEIGHT>` / host `PercentileGlobal<…>` —
  (optionally weighted) α-quantile via argsort + weight-prefix-sum crossing. Used
  by L1/quantile/Huber objectives.
* **Underlying `__global__` kernels (16).** Each multi-block host wrapper above is a
  multi-kernel pass (`cuda_algorithms.cu`, plus `PercentileGlobalKernel` in the
  `.hpp`):
  * `ShufflePrefixSumGlobal` → `ShufflePrefixSumGlobalKernel`,
    `ShufflePrefixSumGlobalReduceBlockKernel`, `ShufflePrefixSumGlobalAddBase`
  * `GlobalInclusiveArgPrefixSum` → `GlobalInclusiveArgPrefixSumKernel`,
    `GlobalInclusivePrefixSumReduceBlockKernel`, `GlobalInclusivePrefixSumAddBlockBaseKernel`
  * `ShuffleReduceSumGlobal` → `ShuffleReduceSumGlobalKernel` + `BlockReduceSum`
  * `ShuffleReduceMinGlobal` → `ShuffleReduceMinGlobalKernel` + `ShuffleBlockReduceMin`
  * `ShuffleReduceDotProdGlobal` → `ShuffleReduceDotProdGlobalKernel`
  * `BitonicArgSortGlobal` → `BitonicArgSortGlobalKernel`, `BitonicArgSortMergeKernel`,
    `BitonicArgCompareKernel`
  * `BitonicArgSortItemsGlobal` → `BitonicArgSortItemsGlobalKernel`
  * `PercentileGlobal` → `PercentileGlobalKernel`
* **Utilities (`cuda_utils.hu`).** `gpuAssert` + `CUDASUCCESS_OR_FATAL` macros;
  typed memory helpers (`AllocateCUDAMemory`, `InitCUDAMemoryFromHostMemory`,
  `CopyFromCUDADeviceToHost[Async]`, `SetCUDAMemory`, …); RAII `CUDAVector<T>`
  (`Resize`/`InitFromHostVector`/`RawData`); `SafeLog<T>` (log, −∞ for x≤0);
  `SynchronizeCUDADevice`.

---

## 3. Dataset on Device (`cuda_column_data.cu`)

`CUDAColumnData` holds the binned feature matrix column-wise on device. Each
column buffer is `uint8/16/32` depending on its bin count; a `void* const*`
table of column pointers plus a `uint8_t* column_bit_type` array let kernels
dispatch on width at runtime.

* **`CopySubrowKernel_ColumnData`** — launcher `LaunchCopySubrowKernel`,
  `<<<ceil(num_used_indices/1024), 1024>>>`, `COPY_SUBROW_BLOCK_SIZE=1024`.
  * IN: `void* const* in_cuda_data_by_column`, `const uint8_t* cuda_column_bit_type`,
    `const data_size_t* cuda_used_indices`, `data_size_t num_used_indices`,
    `int num_column`. OUT: `void** out_cuda_data_by_column`.
  * Logic: one thread per selected row gathers that row across all columns,
    dispatching per-column on bit width and copying
    `in[used_indices[local]] → out[local]`. Produces the compacted binned dataset
    for a bagging/subset selection.

---

## 4. Gradient Discretizer (`cuda_gradient_discretizer.cu`)

Optional quantized-training front-end: converts `f32` gradients/hessians to
`int16` packed into an `int8_t` buffer of length `2*num_data` (hessian at even
slot `2i`, gradient at odd slot `2i+1`). `CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE=1024`.
Three kernels run in sequence (each `SynchronizeCUDADevice`-fenced):

* **`ReduceMinMaxKernel`** — `<<<num_reduce_blocks, 1024>>>`, one thread/row.
  IN `num_data`, `const score_t* input_gradients`, `…hessians`. OUT four
  `score_t*` per-block extrema buffers. Per-block `ShuffleReduceMin/Max<score_t>`
  of grad and hess.
* **`ReduceBlockMinMaxKernel`** — `<<<1, 1024>>>`. Folds the per-block extrema,
  then thread 0 writes the **scales**: `grad_scale = grad_abs_max / (bins/2)`,
  `hess_scale = hess_abs_max / bins`, and their inverses (the multipliers applied
  during quantization). `bins = num_grad_quant_bins`.
* **`DiscretizeGradientsKernel<STOCHASTIC_ROUNDING>`** — `<<<num_reduce_blocks, 1024>>>`.
  IN inputs + `grad_scale_ptr`/`hess_scale_ptr` (inverse scales) + per-tree RNG
  (`random_values_use_start[iter]`, `gradient_random_values`, `hessian_random_values`).
  OUT `int8_t* output_gradients_and_hessians` (as `int16_t*`). Sign-aware rounding:
  stochastic adds a per-row U[0,1) value before truncation, deterministic adds 0.5;
  grad → slot `2i+1`, hess → slot `2i`.

The exposed `grad_scale_ptr()`/`hess_scale_ptr()` (the forward fp scales) feed
de-quantization in the discretized split finder and leaf-splits kernels.

---

## 5. Objective Functions (Gradient/Hessian)

All derive from `CUDAObjectiveInterface<HOST_OBJECTIVE>`, overriding
`GetGradients(const double* scores, score_t* gradients, score_t* hessians)` →
virtual `LaunchGetGradientsKernel`. Also: `ConvertOutputCUDA` (inverse link for
prediction), `LaunchCalcInitScoreKernel` (BoostFromScore), `RenewTreeOutputCUDA`
(leaf-value refit). `cuda_labels_`/`cuda_weights_` are `const label_t*`; a null
weight pointer selects the unweighted template branch. Every element-wise kernel
is one-thread-per-row, `<<<ceil(num_data/1024), 1024>>>`,
`GET_GRADIENTS_BLOCK_SIZE_*=1024`.

**CUDA-supported objectives (exactly 11, from `objective_function.cpp`):**
`CUDARegression{L2,L1,Quantile}loss`, `CUDARegression{Huber,Fair,Poisson}Loss`,
`CUDABinaryLogloss`, `CUDALambdarankNDCG`, `CUDARankXENDCG`, `CUDAMulticlassSoftmax`,
`CUDAMulticlassOVA`. **Not CUDA-supported** (CPU-only — even though their *metrics*
are, §12.1): MAPE, Gamma, Gamma-deviance, Tweedie, cross-entropy (xentropy/xentlambda),
MAP/rank-MAP.

### 5.1 Regression (`cuda_regression_objective.cu`)

Six gradient kernels (note the naming asymmetry — only L2/L1/Quantile carry the
`Regression` prefix). Common IN: `const double* cuda_scores`,
`const label_t* cuda_labels`, `const label_t* cuda_weights`, `num_data`
(+ objective scalar). OUT `score_t* gradients`, `score_t* hessians`. `diff = score − label`:

| Objective | Kernel `<bool USE_WEIGHT>` | grad | hess | extra IN |
|-----------|----------------------------|------|------|----------|
| L2 | `GetGradientsKernel_RegressionL2` | `diff` | `1` | — |
| L1 | `GetGradientsKernel_RegressionL1` | `sign(diff)` | `1` | — |
| Huber | `GetGradientsKernel_Huber` | `|diff|≤α ? diff : sign(diff)·α` | `1` | `alpha` |
| Fair | `GetGradientsKernel_Fair` | `c·diff/(|diff|+c)` | `c²/(|diff|+c)²` | `c` |
| Poisson | `GetGradientsKernel_Poisson` | `exp(score)−label` | `exp(score)·exp(maxδ)` | `max_delta_step` |
| Quantile | `GetGradientsKernel_RegressionQuantile` | `diff≥0 ? 1−α : −α` | `1` | `alpha` |

(All × weight when weighted.) **ConvertOutput:** `ConvertOutputCUDAKernel_Regression`
applies `sign(x)·x²` when `sqrt_` (no-op pass-through otherwise);
`ConvertOutputCUDAKernel_Regression_Poisson` applies `exp`. **RenewTreeOutput:**
`RenewTreeOutputCUDAKernel_RegressionL1<USE_WEIGHT>` and
`RenewTreeOutputCUDAKernel_RegressionQuantile<USE_WEIGHT>` run **one block per leaf**,
computing the (weighted) median/quantile leaf value via `PercentileDevice`. **Init
scores (BoostFromScore):** mean via `ShuffleReduceSumGlobal`/`…DotProdGlobal`
(L2/Huber/Fair) or median via `PercentileGlobal` (L1/Quantile). **Poisson** also runs
`LaunchCheckLabelKernel` (label non-negativity / finiteness via
`ShuffleReduceSumGlobal` + `ShuffleReduceMinGlobal`).

### 5.2 Binary (`cuda_binary_objective.cu`)

* **`GetGradientsKernel_BinaryLogloss<USE_LABEL_WEIGHT, USE_WEIGHT>`** — IN adds
  `const double* cuda_label_weights` (per-class), `const double sigmoid`. With
  `label=±1`: `response = −label·σ/(1+exp(label·σ·score))`; `grad=response`,
  `hess=|response|·(σ−|response|)`.
* **`BoostFromScoreKernel_1_BinaryLogloss<USE_WEIGHT>`** + **`BoostFromScoreKernel_2_BinaryLogloss<USE_WEIGHT>`**
  — kernel 1 is a two-stage warp reduction summing labels & weights
  (`atomicAdd_system`); kernel 2 (`<<<1,1>>>`) computes
  `init_score = log(pavg/(1−pavg))/σ`, `pavg` clamped to `[ε, 1−ε]`.
* **`ConvertOutputCUDAKernel_BinaryLogloss`** — `1/(1+exp(−σ·input))`.
* **`ResetOVACUDALabelKernel`** — rewrites labels to one-vs-all `{0,1}` for class.

### 5.3 Multiclass (`cuda_multiclass_objective.cu`)

* **`GetGradientsKernel_MulticlassSoftmax<USE_WEIGHT>`** — one thread per row,
  loops over classes. Scores/grads/hess are **class-major** (`[k·num_data+i]`).
  `SoftmaxCUDA` → p; class k: `grad = p−1 if label==k else p`,
  `hess = factor·p·(1−p)`, `factor=(K−1)/K`. Scratch `double* cuda_softmax_buffer`.
* **`ConvertOutputCUDAKernel_MulticlassSoftmax`** — per-row softmax to probabilities.
  (`CUDAMulticlassOVA` reuses the binary path per class.)

### 5.4 Rank / LambdaRank (`cuda_rank_objective.cu`)

`NUM_QUERY_PER_BLOCK=10`; **block per group of queries**, block size =
`max_items_in_query_aligned_`. `cuda_query_boundaries_` delimits each query.

* **`GetGradientsKernel_LambdarankNDCG<MAX_ITEM_GT_1024, NUM_RANK_LABEL>`** — loads
  scores to shared, `BitonicArgSort_1024/2048` ranks items, iterates differently-
  labeled pairs touching the top `truncation_level`: `ΔNDCG = (gain_hi−gain_lo)·
  |disc_hi−disc_lo|·inv_max_dcg`, `p = 1/(1+exp(σ·Δscore))`, `λ = −σ·ΔNDCG·p`,
  `hess = σ²·ΔNDCG·p·(1−p)`, accumulated per item via `atomicAdd_block`. Optional
  `norm` rescales by `log2(1+Σλ)/Σλ`. IN adds `cuda_inverse_max_dcgs`,
  `cuda_label_gain`, `sigmoid`, `truncation_level`, `norm`.
* **`GetGradientsKernel_LambdarankNDCG_Sorted<NUM_RANK_LABEL>`** — `max_items>2048`
  variant; pre-sorts via `BitonicArgSortItemsGlobal`, accumulates into global
  gradient/hessian buffers.
* **`GetGradientsKernel_RankXENDCG_SharedMemory<SHARED_MEMORY_SIZE>`** /
  **`GetGradientsKernel_RankXENDCG_GlobalMemory`** (shared 1024/2048 vs global, by
  `max_items_in_query_aligned_`) — softmax (`ShuffleReduceMax/Sum`) → ρ;
  `φ = 2^label − rand` via `__device__ CUDAPhi(label, g)` (per-item randoms
  `cuda_item_rands_`, generated by `GenerateItemRands` + host→device copy);
  cross-entropy-NDCG lambdas via two reduction passes (`sum_l1`, `sum_l2`);
  `grad=λ`, `hess=ρ·(1−ρ)`. The global variant stashes intermediates in the hessian
  output buffer + `cuda_params_buffer` when items exceed shared capacity.

`CUDALambdarankNDCG` and `CUDARankXENDCG` derive from
`CUDALambdaRankObjectiveInterface`. Neither rank objective provides a
`ConvertOutputCUDA` or `RenewTreeOutput` (base no-ops).

---

## 6. Leaf Splits & Tree-Learner Driver

`CUDASingleGPUTreeLearner` (host, subclass of `SerialTreeLearner`) owns six
`unique_ptr` subsystem objects and drives the per-iteration loop (§1): two
`CUDALeafSplits` (`cuda_smaller_leaf_splits_`, `cuda_larger_leaf_splits_`), a
`CUDAHistogramConstructor`, a `CUDABestSplitFinder`, a `CUDADataPartition`, and the
optional `CUDAGradientDiscretizer` (created only when `use_quantized_grad`).
`CUDALeafSplits` seeds the root and exposes the per-leaf `CUDALeafSplitsStruct`:

```cpp
struct CUDALeafSplitsStruct {
  int leaf_index;
  double sum_of_gradients, sum_of_hessians;   // IN to hist-fix & split finder
  int64_t sum_of_gradients_hessians;          // packed, quantized path
  data_size_t num_data_in_leaf;
  double gain, leaf_value;
  const data_size_t* data_indices_in_leaf;    // this leaf's rows
  hist_t* hist_in_leaf;                        // destination histogram
};
```

Tunables: `CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE=1024`,
`NUM_THREADS_PER_BLOCK_LEAF_SPLITS=1024`, `NUM_DATA_THREAD_ADD_LEAF_SPLITS=6`.

### 6.1 Root initialization (`cuda_leaf_splits.cu`)

* **`CUDAInitValuesKernel1<USE_INDICES>`** + **`CUDAInitValuesKernel2`** —
  non-quantized. K1 (`<<<num_blocks_init, 1024>>>`) block-reduces grad/hess via
  `ShuffleReduceSum<double>` into per-block partials (optionally bagging-indexed).
  K2 (`<<<1,1024>>>`) folds partials and thread 0 fills the root struct (leaf 0,
  sums, num_data, root gain & value, data-index/histogram pointers). IN
  `score_t* gradients/hessians`, `lambda_l1/l2`; OUT `CUDALeafSplitsStruct*`.
* **`CUDAInitValuesKernel3<USE_INDICES>`** + **`CUDAInitValuesKernel4`** — quantized.
  Reads packed `int16_t` grad/hess, `ShuffleReduceSum<int64_t>`, writes de-quantized
  double sums (× scales) **and** the packed `int64 sum_of_gradients_hessians`.
* **`InitValuesEmptyKernel`** — `<<<1,1>>>` sets the empty sentinel
  (`leaf_index=−1`, null pointers).

K1–K4 are dispatched by `CUDALeafSplits::LaunchInitValuesKernel` (non-quantized vs
quantized overloads, `USE_INDICES` chosen by whether `cuda_bagging_data_indices ==
nullptr`); the sentinel by `LaunchInitValuesEmptyKernel`.

### 6.2 Leaf-stat reduction & refit (`cuda_single_gpu_tree_learner.cu`)

* **`ReduceLeafStatKernel_SharedMemory`** / **`ReduceLeafStatKernel_GlobalMemory`** —
  launcher `LaunchReduceLeafStatKernel`, `<<<ceil(num_data/1024), 1024>>>`. The
  shared-mem variant (`num_leaves≤2048`) uses `2·num_leaves·sizeof(double)` dynamic
  shared; each thread maps its row→leaf and `atomicAdd_block`s grad/hess, then
  `atomicAdd_system` to global per-leaf buffers. The global-memory variant
  (`num_leaves>2048`) accumulates into per-block slices instead. IN
  `score_t* gradients/hessians`, `num_leaves`, `const int* data_index_to_leaf_index`.
  OUT `double* leaf_grad_stat_buffer`, `leaf_hess_stat_buffer`.
* **`CalcRefitLeafOutputKernel<USE_L1, USE_SMOOTHING>`** — launchers
  `LaunchReduceLeafStatKernel` and `LaunchCalcLeafValuesGivenGradStat`,
  `<<<ceil(num_leaves/1024), 1024>>>`, one thread/leaf (four template instantiations
  on `lambda_l1>0` × `path_smooth>0`). Computes new leaf output (optionally
  path-smoothed), scales by `shrinkage_rate`, blends
  `refit_decay·old + (1−refit_decay)·new`. IN leaf grad/hess buffers,
  `num_data_in_leaf`, parent/child link arrays, `lambda_l1/l2`, `path_smooth`,
  `shrinkage_rate`, `refit_decay_rate`. IN/OUT `double* leaf_value`. Used for
  `FitByExistingTree`/refit.

### 6.3 Categorical-split bitset construction

For categorical splits the driver (`LaunchConstructBitsetForCategoricalSplitKernel`,
via the templated free-function helpers `CUDABitsetLen<T,IS_INNER>` /
`CUDAConstructBitset<T,IS_INNER>`) materializes a bitset of selected categories:

* **`SetRealThresholdKernel`** — maps inner bins back to real feature values via
  `categorical_bin_to_value` + `categorical_bin_offsets`, writing
  `CUDASplitInfo::cat_threshold_real`.
* **`CalcBitsetLenKernel<T,IS_INNER>`** + **`ReduceBlockMaxLen`** — compute the max
  bitset length (`val/32+1`) via `ShuffleReduceMax`; `<<<…,1024>>>` then `<<<1,1024>>>`.
* **`CUDAConstructBitsetKernel<T,IS_INNER>`** — one thread per threshold value sets
  its bit with `atomicAdd_system(out + val/32, 1<<(val%32))` into a pre-zeroed
  `uint32_t* out`.

---

## 7. Histogram Constructor (`cuda_histogram_constructor.cu`, 960 lines)

The core compute kernel — builds, per leaf, a per-feature histogram of
`(Σgrad, Σhess)` per bin. This is the single most performance-critical kernel and
is the *only* CUDA histogram file (confirmed: no other `.cu` constructs
histograms; the OpenCL `gpu_tree_learner` + `ocl/*.cl` is a separate non-CUDA
backend). The file holds **13 `__global__` kernels**: 4 standard build + 4
discretized build + `FixHistogramKernel` + `SubtractHistogramKernel` +
`FixHistogramDiscretizedKernel` + `SubtractHistogramDiscretizedKernel` +
`CopyChangedNumBitHistogram`.

### 7.0 Role, invocation, and histogram storage

* **Entry** `ConstructHistogramForLeaf(smaller, larger, n_smaller, n_larger,
  sum_hess_smaller, sum_hess_larger, num_bits)` (`.cpp:117`). Early-returns if
  **both** children fail the `min_data_in_leaf` / `min_sum_hessian_in_leaf` guards;
  otherwise `LaunchConstructHistogramKernel(smaller, n_smaller, num_bits)` then
  `SynchronizeCUDADevice`. Only the **smaller** leaf is built from data.
* **Storage.** `cuda_hist_` is a flat device arena of
  `num_total_bin_ · 2 · num_leaves_` `hist_t`(=`double`), zeroed in `BeforeTrain`.
  Each leaf's `CUDALeafSplitsStruct::hist_in_leaf` points into it. Layout is
  **interleaved** per bin `b`: `hist_in_leaf[2b]=Σgrad`, `hist_in_leaf[2b+1]=Σhess`
  — hence the pervasive `<<1`. The smaller/larger leaves' histogram pointers are
  assigned (and the parent's reused for the larger child) by `SplitTreeStructureKernel`
  (§9), realizing the subtraction trick's pointer aliasing.
* **Tunables (`.hpp`):** `NUM_DATA_PER_THREAD=400`, `NUM_THREADS_PER_BLOCK=504`,
  `NUM_FEATURE_PER_THREAD_GROUP=28`, `SUBTRACT_BLOCK_SIZE=1024`,
  `FIX_HISTOGRAM_BLOCK_SIZE=512`, `FIX_HISTOGRAM_SHARED_MEM_SIZE=1024`,
  `USED_HISTOGRAM_BUFFER_NUM=8`, `min_grid_dim_y_=160`.
  `DP_SHARED_HIST_SIZE=6144` (5176 when `CUDART_VERSION==10000`),
  `SP_SHARED_HIST_SIZE=2×` (from `cuda_row_data.hpp`).

### 7.1 Thread/block mapping & geometry (`CalcConstructHistogramKernelDim`)

```
block_dim_x = max_num_column_per_partition            // one x-thread per feature column
block_dim_y = NUM_THREADS_PER_BLOCK / block_dim_x     // = 504 / cols  → row workers
grid_dim_x  = num_feature_partitions                  // one block-x per feature partition
grid_dim_y  = max(160, ceil(ceil(n_leaf / 400) / block_dim_y))
```

So **`blockIdx.x` selects a feature partition** (a group of columns whose histogram
fits in shared memory — see §13 `CUDARowData`), **`threadIdx.x` selects one column**
in that partition, and **`threadIdx.y × blockIdx.y` stripes the leaf's rows**. The
`grid_dim_y` floor of 160 deliberately over-decomposes small leaves across many
row-blocks for occupancy — at the cost of more `atomicAdd_system` merge traffic,
which is the dominant scaling bottleneck.

### 7.2 Standard build kernels (non-quantized)

Four variants over two axes — Dense/Sparse × Shared/`_GlobalMemory`:

| Kernel | Storage read | Histogram lives in |
|--------|--------------|--------------------|
| `CUDAConstructHistogramDenseKernel<BIN,HIST,SH>` | column-major `data[idx·ncol+tx]` | `__shared__ HIST_TYPE shared_hist[SH]` |
| `CUDAConstructHistogramSparseKernel<BIN,PTR,HIST,SH>` | CSR `data[row_start+tx]` | `__shared__` |
| `CUDAConstructHistogramDenseKernel_GlobalMemory<BIN,HIST>` | column-major | `cuda_hist_buffer_` slice |
| `CUDAConstructHistogramSparseKernel_GlobalMemory<BIN,HIST,PTR>` | CSR | `cuda_hist_buffer_` slice |

Type params: `BIN_TYPE ∈ {uint8,uint16,uint32}` (`bit_type()`),
`PTR_TYPE ∈ {uint16,uint32,uint64}` (`row_ptr_bit_type()`, sparse only),
`HIST_TYPE ∈ {float (SP), double (DP)}` (`gpu_use_dp_`), `SHARED_HIST_SIZE` matching.

Canonical signature (dense shared) — note `hist_in_leaf` inside the struct is the
sole OUT:

```cpp
template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,   // IN; ->hist_in_leaf is OUT, ->data_indices_in_leaf IN
  const score_t* cuda_gradients, const score_t* cuda_hessians,  // IN  [num_data]   (f32)
  const BIN_TYPE* data,                              // IN  partitioned binned matrix
  const uint32_t* column_hist_offsets,               // IN  per-column bin offset within partition
  const uint32_t* column_hist_offsets_full,          // IN  per-partition hist [start,end)
  const int* feature_partition_column_index_offsets, // IN  partition -> column range
  const data_size_t num_data);                       // IN
```

**Three-phase body (dense shared):**

```
Phase 0 — setup (per thread)
  partition = blockIdx.x;  cols [pcs,pce) = feature_partition_column_index_offsets[partition..+1]
  data_ptr  = data + pcs * num_data                          // base of this partition's columns
  [phs,phe) = column_hist_offsets_full[partition..+1]         // partition's hist span
  n_items   = (phe - phs) << 1                                // grad+hess entries to zero/flush
  n_per_thread = ceil(n_leaf / (gridDim.y*blockDim.y))

Phase 1 — zero shared (cooperative, stride num_threads_per_block); __syncthreads()

Phase 2 — scatter-accumulate (the hot loop)
  if threadIdx.x < num_columns_in_partition:                 // one column per x-thread
    shared_hist_ptr = shared_hist + (column_hist_offsets[pcs+threadIdx.x] << 1)
    block_start = blockIdx.y * blockDim.y * n_per_thread
    for inner = threadIdx.y; inner < block_num_data; inner += blockDim.y:
        idx  = data_indices_ref_this_block[inner]             // gather this leaf's row id
        grad = cuda_gradients[idx];  hess = cuda_hessians[idx]
        bin  = data_ptr[idx * num_columns_in_partition + threadIdx.x]   // column-major load
        atomicAdd_block(shared_hist_ptr + (bin<<1),     grad)
        atomicAdd_block(shared_hist_ptr + (bin<<1) + 1, hess)
  __syncthreads();

Phase 3 — merge block-partial into global leaf histogram
  feat_hist = smaller_leaf_splits->hist_in_leaf + (phs << 1)
  for i = thread_idx; i < n_items; i += num_threads_per_block:
      atomicAdd_system(feat_hist + i, shared_hist[i])         // cross-block (per blockIdx.y) merge
```

**Two-tier atomics** are the defining structure: fast `atomicAdd_block` into shared
memory during the sweep, then `atomicAdd_system` to merge each `blockIdx.y` block's
partial into the global histogram. Because many y-blocks cover disjoint row stripes
of the same partition, the global merge must be atomic.

**Sparse variants** differ only in the bin fetch: `block_row_ptr = row_ptr +
blockIdx.x·(num_data+1)`; per row `row_start/row_end = block_row_ptr[idx..+1]`,
`row_size = row_end−row_start`, and `if (threadIdx.x < row_size)` reads
`bin = data_ptr[row_start + threadIdx.x]` (`data_ptr = data + partition_ptr[blockIdx.x]`).
The iteration uses a precomputed `num_iteration_this` (handling the `blockDim.y`
remainder) instead of the dense `inner < block_num_data` bound.

**`_GlobalMemory` variants** replace `__shared__ shared_hist` with a slice of
`cuda_hist_buffer_` at `(blockIdx.y · num_total_bin + phs) · 2` — one partial
histogram per y-block in global memory. Used when `NumLargeBinPartition() > 0`
(a single column's histogram exceeds shared capacity). Buffer sized in `Init` as
`grid_dim_y · num_total_bin · {4 if DP, 2 if SP}` (`·1` for quantized).

### 7.3 Discretized build kernels (quantized training)

Selected when `use_quantized_grad_`. Four variants, each templated
`<BIN_TYPE, [PTR_TYPE,] SHARED_HIST_SIZE, bool USE_16BIT_HIST>`:

* `CUDAConstructDiscretizedHistogramDenseKernel`
* `CUDAConstructDiscretizedHistogramSparseKernel`
* `CUDAConstructDiscretizedHistogramDenseKernel_GlobalMemory`
* `CUDAConstructDiscretizedHistogramSparseKernel_GlobalMemory`

Gradients & hessians arrive **pre-packed by the discretizer
(§4)** as one `int32` per row (`int16` grad in the high half, `int16` hess in the
low half) — read from `cuda_gradients_` reinterpreted as `const int32_t*`.

```cpp
__shared__ int16_t shared_hist[SHARED_HIST_SIZE];
int32_t* shared_hist_packed = reinterpret_cast<int32_t*>(shared_hist);  // one int32 per bin
...
const int32_t grad_and_hess = cuda_gradients_and_hessians[idx];          // packed
const uint32_t bin = data_ptr[...];
atomicAdd_block(shared_hist_packed + bin, grad_and_hess);                 // ONE atomic, both stats
```

The shared histogram is therefore **half the entries** (one packed `int32` per bin,
not a grad/hess pair) and uses **one** `atomicAdd_block` per row. The global flush
depends on `USE_16BIT_HIST`:

* **`USE_16BIT_HIST=true`** (`num_bits ≤ 16`): write the packed `int32` directly via
  `atomicAdd_system` into an `int32_t*` view of `hist_in_leaf` — the global
  histogram stays 16+16-bit.
* **`USE_16BIT_HIST=false`** (`num_bits = 32`): **unpack** the `int32` to `int64`
  (`(int16)(p>>16) << 32) | (p & 0xffff)`) and `atomicAdd_system` into an
  `atomic_add_long_t*` (`unsigned long long`) view — the global histogram widens to
  32+32-bit to avoid overflow.

**Per-leaf bit-width selection** (`GradientDiscretizer::SetNumBitsInHistogramBin`,
shared with the CPU path; the CUDA learner calls it and passes `num_bits` into
`ConstructHistogramForLeaf`): for each leaf `max_stat_per_bin = num_data_in_leaf ·
num_grad_quant_bins`, then `< 256 → 8`, `< 65536 → 16`, else `32` bits. The kernel
collapses {8,16} → `USE_16BIT_HIST=true` and {32} → `false`. The 8-bit case has no
separate kernel — it shares the 16-bit packed path.

### 7.4 Host dispatch ladder

`LaunchConstructHistogramKernel` resolves the full instantiation through six nested
single-decision helpers (each fixes one type/flag, then recurses):

```
LaunchConstructHistogramKernel
  └─ HIST_TYPE/SHARED_HIST_SIZE  ← shared_hist_size() & gpu_use_dp_   (double/DP or float/SP)
     └─ Inner  : BIN_TYPE        ← bit_type()          ∈ {8,16,32}
        └─ Inner0: PTR_TYPE      ← row_ptr_bit_type()  ∈ {16,32,64}
           └─ Inner1: USE_GLOBAL_MEM_BUFFER ← NumLargeBinPartition() > 0
              └─ Inner2: CalcConstructHistogramKernelDim(), then branch on
                         use_quantized_grad_ × is_sparse() × (num_bits ≤ 16)
                         → final <<<grid, block, 0, cuda_stream_>>> launch
```

All types/flags become template constants, so the hot scatter loop has no runtime
branches.

### 7.5 Finalization kernels (`SubtractHistogramForLeaf`)

After the smaller leaf is built, the larger sibling is derived and omitted bins
repaired. `LaunchSubtractHistogramKernel` runs Fix then Subtract; the discretized
path additionally handles bit-width changes between parent/children.

**Non-quantized:**

* **`FixHistogramKernel`** — `<<<need_fix_histogram_features_.size(), 512>>>`,
  `__shared__ hist_t shared_mem_buffer[32]`. One block per feature whose
  `most_freq_bin ≠ 0` (those bins were skipped during the scatter — most-frequent-bin
  omission). Each thread loads one bin's grad/hess (0 for the most-frequent or
  out-of-range bin), `ShuffleReduceSum<hist_t>` over `num_bin_aligned`, then thread 0
  writes the omitted bin: `feat_hist[mfb·2] = leaf_sum_gradients − Σ`,
  `feat_hist[mfb·2+1] = leaf_sum_hessians − Σ`. `need_fix_histogram_features_` and
  the power-of-two `num_bin_aligned` are precomputed host-side in `InitFeatureMetaInfo`.
* **`SubtractHistogramKernel`** — `<<<ceil(2·num_total_bin/1024), 1024>>>`. One thread
  per element: guarded by `larger.leaf_index ≥ 0`,
  `larger_leaf_hist[i] −= smaller_leaf_hist[i]` (the subtraction trick).

**Quantized:**

* **`FixHistogramDiscretizedKernel<USE_16BIT_HIST>`** — same structure with the bin
  read/written as packed `int32` (16-bit) or `int64` (32-bit), and the leaf total
  taken from `sum_of_gradients_hessians` (packed `int64`); reduce via
  `ShuffleReduceSum<int32_t|int64_t>`.
* **`SubtractHistogramDiscretizedKernel<SMALLER_16BIT, LARGER_16BIT, PARENT_16BIT>`** —
  `<<<ceil(num_total_bin/1024), 1024>>>`. Four launch cases (chosen from the parent's
  and children's bit widths) because the smaller, larger, and parent histograms may
  each be 16- or 32-bit:
  * all 16-bit → straight `int32` subtract;
  * parent 32-bit, smaller 16-bit → load smaller as `int32`, sign-expand to `int64`,
    subtract from the `int64` parent in place;
  * larger needs widening → subtract into `num_bit_change_buffer` (an `int32` staging
    buffer), then **`CopyChangedNumBitHistogram`** (`<<<…,1024>>>`) copies it back into
    the larger leaf's histogram;
  * all 32-bit → straight `int64` subtract.
  `CHECK_LE` asserts enforce the monotonic bit-width relationship (parent ≥ child).

### 7.6 Memory & numerical notes

* The shared accumulation precision (`float` SP vs `double` DP) is `gpu_use_dp_`;
  the durable `cuda_hist_` is always `hist_t = double`. SP atomic ordering is
  non-deterministic → not bit-reproducible (the f32-vs-f64 residual the ROCm gate
  tolerates; see §17). The quantized path is integer and **exactly** reproducible.
* `cuda_hist_buffer_` (global spill) and `hist_buffer_for_num_bit_change_`
  (`num_total_bin·2`) are sized in `Init`; the latter backs the discretized
  bit-width-change subtract.

---

## 8. Best Split Finder (`cuda_best_split_finder.cu`)

Three-stage pipeline. A `(leaf, feature)` *task* (`SplitFindTask`: inner_feature,
reverse, skip_default_bin, na_as_missing, is_categorical, is_one_hot, hist_offset,
mfb_offset, num_bin, default_bin) drives stage 1; smaller-leaf task `t` writes
`CUDASplitInfo[t]`, larger-leaf task `t` writes `[t+num_tasks]`. Tunables:
`NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER=256`, `NUM_THREADS_FIND_BEST_LEAF=256`,
`NUM_TASKS_PER_SYNC_BLOCK=1024`. Smaller leaf on stream 0, larger on stream 1.

### 8.1 Stage 1 — per-feature split evaluation (one block per task)

* **`FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, IS_LARGER>`** —
  `<<<num_tasks, 256, 0, stream>>>`. The four bool flags fan out from
  `extra_trees_`/`lambda_l1_`/`use_smoothing_` via `Inner0/1/2`. IN:
  `is_feature_used_bytree`, `tasks`, `CUDARandom*`, the leaf `CUDALeafSplitsStruct`
  (parent gain, sum grad/hess, num_data, `hist_in_leaf`), config scalars
  (`min_data_in_leaf`, `min_sum_hessian_in_leaf`, `min_gain_to_split`,
  `lambda_l1/l2`, `path_smooth`, `cat_smooth`, `cat_l2`, `max_cat_threshold`,
  `min_data_per_group`). OUT: `CUDASplitInfo* cuda_best_split_info` (one/task).
  Reads the feature's `hist_t` histogram and dispatches by feature kind to one of
  two `__device__` cores:
  * **Numerical** — `__device__ FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1,
    USE_SMOOTHING, REVERSE>` (REVERSE = forward/reverse default-bin scan direction).
    Each thread loads one bin's `(grad,hess)`, block-wide `ShufflePrefixSum` →
    cumulative left (or right, if REVERSE) sums, derives the complement from parent
    totals, recovers counts via `cnt_factor = num_data/sum_hessians` +
    `__double2int_rn`. Applies min-data/min-hessian guards, computes gain via
    `CUDALeafSplits::GetSplitGains<USE_L1,USE_SMOOTHING>`, keeps if
    `> parent_gain + min_gain_to_split`. The winning thread (`ReduceBestGain` argmax)
    writes `CUDASplitInfo`: `is_valid`, `threshold`, `gain`, `default_left`,
    left/right sum grad/hess, count, value (`CalculateSplittedLeafOutput`), gain
    (`GetLeafGainGivenOutput`).
  * **Categorical** — `__device__ FindBestSplitsForLeafKernelCategoricalInner<USE_RAND,
    USE_L1, USE_SMOOTHING>`. One-hot tests each category as a singleton left set;
    many-category computes per-bin `grad/(hess+cat_smooth)`, sorts bins by it with
    `BitonicArgSort_1024`, then prefix-sweeps both directions up to
    `max_num_cat = min(max_cat_threshold, (used_bin+1)/2)`, writing the category list
    `cat_threshold[0..]` (adds `cat_l2` to `l2`).
* **`FindBestSplitsDiscretizedForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, IS_LARGER>`**
  — integer-histogram variant (quantized). Dispatches to `__device__
  FindBestSplitsDiscretizedForLeafKernelInner<…, BIN_HIST_TYPE, ACC_HIST_TYPE,
  USE_16BIT_BIN_HIST, USE_16BIT_ACC_HIST>` with `int32` bin storage and `int32`/`int64`
  accumulator by `num_bits ≤ 16`. Unpacks grad/hess, `ShufflePrefixSum` on the packed
  accumulator, rescales via `grad_scale`/`hess_scale`; also fills
  `left/right_sum_of_gradients_hessians` (`int64`). Extra IN:
  `smaller/larger_leaf_num_bits_in_histogram_bin`, `grad_scale`, `hess_scale`,
  `max_cat_to_onehot`. **Categorical unsupported** (`asm("trap;")`).
* **`FindBestSplitsForLeafKernel_GlobalMemory<…, IS_LARGER>`** — low-VRAM path
  (`use_global_memory_`); cores `__device__ FindBestSplitsForLeafKernelInner_GlobalMemory`
  / `FindBestSplitsForLeafKernelCategoricalInner_GlobalMemory` run the same gain math
  over strided global loops (`GlobalMemoryPrefixSum`, `BitonicArgSortDevice`) with the
  scratch buffers `feature_hist_{grad,hess,stat}_buffer` + `feature_hist_index_buffer`,
  for blocks with more bins than threads. (The discretized global-memory path is an
  unimplemented `TODO` — there is no `FindBestSplitsDiscretizedForLeafKernel_GlobalMemory`.)

**Block argmax device helpers (within a task):** `ReduceBestGainWarp` (warp
`__shfl_down_sync`) → `ReduceBestGainBlock` (cross-warp via shared) → `ReduceBestGain`
— argmax over `(gain, found, thread_index)` to pick the single best-threshold thread.
`ReduceBestSplit` is the analogous `(found, gain, shared_read_index)` block reduction.

### 8.2 Stage 2 — cross-feature reduction per leaf

* **`SyncBestSplitForLeafKernel`** — `<<<ceil(num_tasks/1024), 1024, 0, stream>>>`.
  Block-reduces per-task `(is_valid, gain)` with `ReduceBestGain`; thread 0 copies
  the winning `CUDASplitInfo` into `cuda_leaf_best_split_info[leaf + block·num_leaves]`,
  stamping `inner_feature_index`.
* **`SyncBestSplitForLeafKernelAllBlocks`** — `<<<1,1>>>`, folds per-block winners
  when `num_blocks_per_leaf>1`.
* **`SetInvalidLeafSplitInfoKernel`** — marks a leaf's split invalid when not valid.

### 8.3 Stage 3 — cross-leaf argmax & export

* **`FindBestFromAllSplitsKernel`** — `<<<1, 256>>>`. Cross-leaf argmax via the
  `__device__` family `ReduceBestGainForLeavesWarp` → `ReduceBestGainForLeavesBlock` →
  `ReduceBestGainForLeaves` (over `(gain, leaf_index)`); thread 0 writes
  `buffer[6]=best_leaf`, `buffer[7]=num_cat_threshold`, and invalidates the chosen
  leaf (and the freshly created leaf slot) so it isn't re-picked.
* **`PrepareLeafBestSplitInfo`** — `<<<6,1>>>`, copies the smaller/larger leaves'
  `inner_feature_index`/`threshold`/`default_left` into an 8-int buffer.

The only device→host transfer per iteration is this 8-int
`cuda_best_split_info_buffer_`. Setup kernels: `AllocateCatVectorsKernel` (point
each `CUDASplitInfo.cat_threshold` at a pre-allocated slab),
`InitCUDARandomKernel` (seed per-task RNG for extra-trees).

---

## 9. Data Partition (`cuda_data_partition.cu`)

Maintains a single permutation `data_size_t* cuda_data_indices_` where each leaf
occupies a contiguous `[leaf_data_start, leaf_data_end)`. A `Split` reorders the
parent slice in place so the two children become adjacent contiguous ranges — by
mark→prefix-sum→scatter, **never sorting**. Tunables: all block sizes 1024
(`FILL_INDICES_…`, `SPLIT_INDICES_…`, `AGGREGATE_…`). `CalcBlockDim`:
`min_num_blocks = num_data≤100 ? 1 : 80`, power-of-two `block_dim_≤1024`,
`grid_dim_ = ceil(num_data/block_dim_)`.

* **`FillDataIndicesBeforeTrainKernel`** — identity-init `data_indices[i]=i`, all
  rows → leaf 0 (before any split).
* **`FillDataIndexToLeafIndexKernel`** — reset the row→leaf map to root.
* **`GenDataToLeftBitVectorKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA,
  MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, USE_MIN_BIN, BIN_TYPE>`** —
  `<<<grid_dim_, block_dim_>>>` on stream 0. Each thread reads its row's bin from
  `column_data` and, via the templated missing/default/min-max logic, sets
  `to_left=1` when `bin≤th` (else 0). `PrepareOffset` runs a block
  `ShufflePrefixSum<uint16_t>`, writing each thread's exclusive left-rank into
  `uint16_t* block_to_left_offset` and the block's left/right totals into
  `data_size_t*` buffers. IN `BIN_TYPE* column_data`, `num_data_in_leaf`,
  `data_indices_in_leaf`, `th`, `t_zero_bin`, `max_bin`, `min_bin`, default-direction
  flags.
* **`UpdateDataIndexToLeafIndexKernel<…, BIN_TYPE>`** — stream 3 (concurrent); same
  decision but writes the destination leaf id directly into
  `int* cuda_data_index_to_leaf_index` (consumed later by score update).
* **`GenDataToLeftBitVectorKernel_Categorical<BIN_TYPE, USE_MIN_BIN>`** /
  **`UpdateDataIndexToLeafIndexKernel_Categorical<…>`** — membership via
  `CUDAFindInBitset(bitset, bitset_len, bin−min_bin+mfb_offset)`.
* **`AggregateBlockOffsetKernel0`** (when `grid_dim_ > 1024`, `<<<1,1024>>>`) /
  **`AggregateBlockOffsetKernel1`** (else, `<<<1, num_blocks_final_aligned>>>`) —
  on stream 0, `__shared__ uint32_t shared_mem_buffer[32]` + `to_left_total_count`.
  A single block scans all `num_blocks+1` per-block left/right counts into a global
  exclusive prefix sum — Kernel0 does a serial chunk-scan + `ShufflePrefixSum<uint32_t>`
  for large grids, Kernel1 a one-shot warp scan — shifting right offsets by
  `to_left_total` so left rows occupy `[0,to_left_total)` and right rows the rest.
  Writes children `cuda_leaf_data_start/_end/_num_data`.
* **`SplitInnerKernel`** — `<<<grid_dim_, block_dim_>>>` on stream 1 (the scatter):
  each thread recovers its in-block exclusive left rank, scatters left rows to
  `out[to_left_base + rank]`, right rows to `out[to_right_base + (tid − rank)]` in
  scratch `cuda_out_data_indices_in_leaf_`.
* **`SplitTreeStructureKernel`** — `<<<4,5>>>` on stream 0: scalar fan-out — writes
  child leaf outputs, packs the 16-int `cuda_split_info_buffer` for host copy-back,
  assigns smaller-vs-larger child roles and swaps the histogram-pool pointers
  (implementing the parent-minus-sibling reuse).
* **`CopyDataIndicesKernel`** — `<<<grid_dim_, block_dim_>>>` on stream 2: copies the
  reordered scratch back over the parent's slice, finalizing contiguity.
* **`AddPredictionToScoreKernel<USE_BAGGING>`** — `<<<ceil(num_data/1024),1024>>>`:
  per row, looks up its leaf and adds that leaf's output to `double* cuda_scores`.
* **`RenewDiscretizedTreeLeavesKernel`** — launcher `LaunchReduceLeafGradStat`,
  one block per leaf, `__shared__ double[32]`; `ShuffleReduceSum<double>` of grad/hess
  over the leaf's contiguous slice to recompute leaf stats (quantized path).

**Host orchestration.** `Split` (host) calls `CalcBlockDim` → `GenDataToLeftBitVector`
→ `SplitInner`. The bin-classification kernels fan out their compile-time flags
through nested launchers: numeric `LaunchGenDataToLeftBitVectorKernel → …Inner →
…Inner0..4` (and `…_Inner0..4` for `LaunchUpdateDataIndexToLeafIndexKernel`), with the
categorical path via `LaunchGenDataToLeftBitVectorCategoricalKernel`. A single
`LaunchSplitInnerKernel` then sequences `AggregateBlockOffsetKernel{0,1}` (stream 0),
`SplitInnerKernel` (stream 1), `CopyDataIndicesKernel` (stream 2), and
`SplitTreeStructureKernel` (stream 0) across `cuda_streams_[0..3]`. Per-split device→host
transfer is the 16-int `cuda_split_info_buffer_` (`CopyFromCUDADeviceToHostAsync`):
child num_data, start, and sum grad/hess refs. `UpdateDataIndexToLeafIndexKernel`
runs concurrently on stream 3 during the bit-vector pass.

---

## 10. Tree Model I/O (`cuda_tree.cu`)

`CUDATree` holds the model as flat device arrays (`cuda_leaf_value_`,
`cuda_left_child_`, `cuda_decision_type_`, …). Helper `__device__` functions
pack/unpack the `int8_t` decision-type bitfield (`Set/GetDecisionTypeCUDA`,
`…MissingType`) and do categorical lookup (`FindInBitsetCUDA<T>`).

* **`SplitKernel`** — `<<<3,5>>>` (15 threads, one per node field). Applies one
  numerical split: thread 0 rewires parent/child links; threads 1–14 each write one
  field (split feature, gain, internal/leaf weights & values from `CUDASplitInfo`
  (NaN→0), counts, depth, decision-type bits, bin/real thresholds).
* **`SplitCategoricalKernel`** — `<<<3,6>>>` (17 threads): same fan-out with
  `kCategoricalMask`, stores `num_cat`, and extends `cat_boundaries`/`…_inner` by
  the bitset lengths.
* **`ShrinkageKernel`** — `<<<ceil(num_leaves/1024),1024>>>`: `leaf_value *= rate`.
* **`AddBiasKernel`** — `leaf_value += val`.
* **`AddPredictionToScoreKernel<USE_INDICES>`** — one thread per row, walks the tree
  from node 0: reads the row's raw bin from `CUDAColumnData` (dispatching on 8/16/32
  column width), remaps into the feature's offset range (else `most_freq_bin`),
  branches via categorical bitset (`FindInBitsetCUDA`) or numeric threshold with
  missing/`default_left` handling, and at a leaf adds `leaf_value[~node]` to
  `double* score`. This is the **prediction kernel**.

---

## 11. Score Updater (`cuda_score_updater.cu`)

The cumulative per-row score is `double* cuda_score_`, laid out per tree at
`offset = num_data·tree_id`. The per-leaf addition (`AddScore(const Tree*, …)`) is
delegated to `tree→AddPredictionToScore` / `tree_learner→AddPredictionToScore`
(the §9/§10 kernels). The two kernels here are whole-array scalar ops:

* **`AddScoreConstantKernel`** — `score[i] += val` (init score / no-split single-leaf
  tree).
* **`MultiplyScoreConstantKernel`** — `score[i] *= val` (shrinkage / DART rescale).

Both `<<<ceil(num_data/num_threads_per_block_), num_threads_per_block_>>>`. When
`boosting_on_cuda_` is false, each is followed by a `CopyFromCUDADeviceToHost` to
mirror into the host `score_` vector.

---

## 12. Metrics (`cuda_pointwise_metric.cu`)

Per-row loss + two-stage reduction to a scalar. Each concrete metric supplies a
`__device__ static MetricOnPointCUDA` (L2: `(score−label)²`; binary logloss:
`−log(score)` / `−log(1−score)`; quantile: pinball).

* **`EvalKernel<CUDA_METRIC, USE_WEIGHTS>`** — `<<<ceil(num_data/1024), 1024>>>`,
  `NUM_DATA_PER_EVAL_THREAD=1024`, one thread/row, `__shared__ double[32]`. IN
  `num_data`, `const label_t* labels`, `const label_t* weights`,
  `const double* scores`, `const double param`. OUT
  `double* reduce_block_buffer` (per-block partial loss at `blockIdx.x`; weight at
  `blockIdx.x+gridDim.x`). Per-thread `point_metric = MetricOnPointCUDA(…)` (×weight),
  `ShuffleReduceSum<double>` to one value per block. The host then folds the blocks
  with `ShuffleReduceSumGlobal<double,double>` → scalar `*sum_loss` (and `*sum_weight`,
  else `num_data`).

### 12.1 Concrete loss device functions (the metric `.hpp` files)

`EvalKernel` is metric-agnostic: each concrete metric supplies a
`__device__ static double MetricOnPointCUDA(label_t label, double score, double param)`
(inlined via the `CUDA_METRIC` template parameter — **`__device__`-only, never
`__global__`**) plus a host ctor and optional `GetParamFromConfig()`. The `.cpp`
files only register/construct metrics and emit `Init` instantiations; the math lives
in the headers. Class chain:
`HOST_METRIC → CUDAMetricInterface → CUDAPointwiseMetricInterface →
CUDA{Regression,Binary}MetricInterface → concrete`.

`Eval` flow: optional `objective->ConvertOutputCUDA` (inverse link into
`score_convert_buffer_`) → `LaunchEvalKernel` → regression returns
`AverageLoss(sum_loss, sum_weight)` (where RMSE's `AverageLoss` applies the
`sqrt`), binary returns `sum_loss / sum_weight`. With `d = score − label`:

| Metric class | `param` | `MetricOnPointCUDA` | final |
|--------------|---------|---------------------|-------|
| `CUDARMSEMetric` | — | `d²` | mean → **sqrt** |
| `CUDAL2Metric` | — | `d²` | mean |
| `CUDAL1Metric` | — | `|d|` | mean |
| `CUDAQuantileMetric` | `alpha` | `δ=label−score; δ<0 ? (α−1)·δ : α·δ` | mean |
| `CUDAHuberLossMetric` | `alpha` | `|d|≤α ? 0.5·d² : α·(|d|−0.5·α)` | mean |
| `CUDAFairLossMetric` | `fair_c` | `x=|d|,c: c·x − c²·log1p(x/c)` | mean |
| `CUDAPoissonMetric` | — | `s=max(score,1e-10): s − label·log(s)` | mean |
| `CUDAMAPEMetric` | — | `|d| / max(1, |label|)` | mean |
| `CUDAGammaMetric` | — | `θ=−1/score: −((label·θ + SafeLog(−θ)) + SafeLog(label/1)−SafeLog(label))` | mean |
| `CUDAGammaDevianceMetric` | — | `t=label/(score+1e-9): t − SafeLog(t) − 1` | mean |
| `CUDATweedieMetric` | `tweedie_variance_power` | `s=max(score,1e-10),ρ: −label·e^{(1−ρ)ln s}/(1−ρ) + e^{(2−ρ)ln s}/(2−ρ)` | mean |
| `CUDABinaryLoglossMetric` | — | prob input: `label≤0 ? −log(1−score) : −log(score)` (clamped by `kEpsilon`) | `Σloss/Σweight` |

All twelve are instantiated at the bottom of `cuda_pointwise_metric.cpp`. Metrics
needing a scalar (`Quantile`/`Huber`→`alpha`, `Fair`→`fair_c`,
`Tweedie`→`tweedie_variance_power`) override `GetParamFromConfig()`; the rest ignore
`param`.

**CUDA-supported metrics (exactly these 12 pointwise losses).** All are regression-
or binary-pointwise. **Not CUDA-supported** (the `Metric::CreateMetric` `#ifdef USE_CUDA`
branch in `metric.cpp` returns the **CPU** class even on CUDA): `AUCMetric`,
`AucMuMetric`, `NDCGMetric`, `MapMetric`, and the multiclass metrics
(`MultiErrorMetric`, `MultiSoftmaxLoglossMetric`, `CrossEntropy*`, `KullbackLeibler`).
There is **no** CUDA rank/AUC/multiclass metric file — `metric/cuda/` contains only
the pointwise, binary, and regression headers. So with `device=cuda`, those metrics
still evaluate on the host (their inputs are copied back as needed).

---

## 13. Device Row Data (`CUDARowData`)

`CUDARowData` (`include/LightGBM/cuda/cuda_row_data.hpp`, `src/io/cuda/cuda_row_data.cpp`)
is the on-device, row-wise binned feature matrix **plus the feature-partition layout
the histogram kernel (§7) is built around**. Pure host-side infrastructure: it owns
device buffers and uploads via `InitCUDAMemoryFromHostMemory`, but defines no
`__global__` kernel of its own (besides the bagging `CopySubrow` path).

**Storage & width selectors.** Dense or sparse (`is_sparse()`). `bit_type()` ∈
{8,16,32} picks the live bin buffer (`cuda_data_uint{8,16,32}_t_`);
`row_ptr_bit_type()` ∈ {16,32,64} picks the sparse CSR row-pointer buffer. The 3×3
combinations are dispatched explicitly (`InitSparseData<BIN_TYPE, PTR_TYPE>`).

**Feature partitions.** Features are grouped so **one partition's histogram fits in
shared memory**: budget `max_num_bin_per_partition = shared_hist_size_ / 2`
(each entry is a grad+hess pair). `DivideCUDAFeatureGroups` walks feature groups;
a column whose bin count exceeds the budget becomes its **own large-bin partition**
(`NumLargeBinPartition()` > 0 → kernel uses the `_GlobalMemory` path), otherwise
columns pack until the budget is hit. Arrays consumed by the histogram launcher:

| Accessor | Type | Meaning |
|----------|------|---------|
| `cuda_feature_partition_column_index_offsets()` | `const int*` | partition *i* owns columns `[off[i], off[i+1])` |
| `cuda_column_hist_offsets()` | `const uint32_t*` | per-column bin offset **within its partition** |
| `cuda_partition_hist_offsets()` | `const uint32_t*` | global bin offset where each partition begins |
| `max_num_column_per_partition()` | `int` | sizes block_dim_x (dense) / per-row nnz (sparse) |
| `num_feature_partitions()` | `int` | = histogram grid_dim_x |
| `shared_hist_size()` | `int` | shared-mem capacity (`DP_/SP_SHARED_HIST_SIZE`) |
| `NumLargeBinPartition()` | `int` | # single-column partitions too big for SMEM |

**Accessors** `GetBin<BIN_TYPE>()`, `GetRowPtr<PTR_TYPE>()`,
`GetPartitionPtr<PTR_TYPE>()` (each dispatched on width, `Log::Fatal` otherwise)
hand the launcher the live buffers without branching. **Init** computes the layout,
fetches host row-wise data + widths from `GetRowWiseData(...)`, re-lays dense data
per-partition (`GetDenseDataPartitioned`) or builds per-partition CSR
(`GetSparseDataPartitioned`, subtracting `partition_hist_start` to make bins
partition-local), uploads, and `SynchronizeCUDADevice`. `shared_hist_size_` =
`gpu_use_dp_ ? DP_SHARED_HIST_SIZE : SP_SHARED_HIST_SIZE`.

---

## 14. Host-Side Infrastructure Classes

The host classes that own device memory and drive the kernels above:

* **`CUDAMetadata`** (`cuda_metadata.{hpp,cpp}`) — device mirror of `Metadata`. Owns &
  uploads `cuda_label_`, `cuda_weights_`, `cuda_query_boundaries_`,
  `cuda_query_weights_`, `cuda_init_score_` — the buffers objectives (§5) and metrics
  (§12) read. `Init(...)` bulk-uploads; `SetLabel`/`SetWeights`/`SetQuery`/`SetInitScore`
  patch incrementally. Launches no compute kernel.
* **`CUDAColumnData`** (`cuda_column_data.{hpp,cpp}`) — device columnar binned store.
  Owns per-column buffers (`cuda_data_by_column_`) + per-feature meta
  (`cuda_column_bit_type_`, `cuda_feature_{min_bin,max_bin,offset,most_freq_bin,
  default_bin}_`, missing/mfb flags, `cuda_feature_to_column_`). `Init` expands
  dense 4/8/16/32-bit and sparse columns to device; `CopySubrow` (drives
  `CopySubrowKernel`, §3) builds bagging subsets. Consumed by the prediction kernel (§10).
* **`CUDATree`** (`cuda_tree.{hpp,cpp}`) — `Tree` subclass holding topology as parallel
  device arrays (`cuda_{left,right}_child_`, `cuda_threshold{,_in_bin}_`,
  `cuda_decision_type_`, `cuda_leaf_value_`, …) + categorical `CUDAVector` bitsets +
  its own stream. `Split`/`SplitCategorical` drive §10 kernels; `AddPredictionToScore`
  drives the prediction kernel; `Shrinkage`/`AddBias` follow the host `Tree::` update;
  `ToHost()`/`SyncLeafOutput*` move arrays host↔device. Declares the `__device__`
  decision/missing-type helpers the kernels use.
* **`CUDAScoreUpdater`** (`cuda_score_updater.{cpp,hpp}`) — `ScoreUpdater` subclass
  owning `cuda_score_` (`num_data × num_tree_per_iteration`). Constant `AddScore`/
  `MultiplyScore` drive §11 kernels; tree `AddScore` delegates to
  `tree->AddPredictionToScore` / `tree_learner->AddPredictionToScore`.
  `boosting_on_cuda_=false` mirrors `cuda_score_` back to host `score_` each call;
  `true` keeps it resident.
* **`CUDAObjectiveInterface<HOST_OBJECTIVE>`** (`cuda_objective_function.hpp`) — CRTP
  base wrapping a host objective for CUDA. Mixes in `cuda_labels_`/`cuda_weights_`
  (from `CUDAMetadata`); routes `GetGradients→LaunchGetGradientsKernel`,
  `ConvertOutputCUDA`, `BoostFromScore→LaunchCalcInitScoreKernel`,
  `RenewTreeOutputCUDA`, each `SynchronizeCUDADevice`-fenced (§5).
* **`cuda_utils.cpp`** — out-of-line backers for `cuda_utils.hu`:
  `SynchronizeCUDADevice` (`cudaDeviceSynchronize` + `gpuAssert`),
  `PrintLastCUDAError`, `SetCUDADevice` (set only if different), `GetCUDADevice`.
* **`vector_cudahost.h`** — process-global `LGBM_config_` (`current_device`/
  `current_learner` selectors) + `CHAllocator<T>`, an STL allocator using
  `cudaHostAlloc(cudaHostAllocPortable)` / `cudaFreeHost` for **pinned** host memory
  when device is CUDA (fast async H2D), falling back to `_mm_malloc` otherwise.

---

## 15. Device Structs & RNG

* **`CUDASplitInfo`** (`cuda_split_info.hpp`) — the per-split record produced by §8
  and consumed by §9/§10/§6.3. Fields: `is_valid`, `leaf_index`, `gain`,
  `inner_feature_index`, `threshold` (`uint32_t`), `default_left`; per side
  `{left,right}_sum_gradients`/`_sum_hessians` (`double`),
  `_sum_of_gradients_hessians` (`int64_t`, quantized), `_count` (`data_size_t`),
  `_gain`, `_value` (`double`); and categorical `num_cat_threshold`,
  `uint32_t* cat_threshold`, `int* cat_threshold_real` (slabs from
  `AllocateCatVectorsKernel`). Defines `__device__` ctor/dtor and a deep-copy
  `operator=` (so the argmax reductions can copy whole records between leaf slots).
* **`CUDARandom`** (`cuda_random.hpp`) — `__device__` LCG matching the CPU `Random`:
  `x = 214013·x + 2531011`; `RandInt16 = (x>>16)&0x7FFF`, `RandInt32 = x&0x7FFFFFFF`,
  `NextFloat = RandInt16/32768`. Seeded per-task by `InitCUDARandomKernel` (§8.3) for
  extra-randomized-trees threshold selection. Bit-identical stream to the host RNG is
  a parity requirement.

---

## 16. End-to-End Sequencing

Per boosting iteration. Steps 1, 5–7 are driven by the **boosting layer**
(`GBDT::TrainOneIter`); steps 2–4 are inside `CUDASingleGPUTreeLearner::Train`.
All compute is on device, streamed:

1. **[GBDT]** `Objective::GetGradients(scores → grad/hess)` (§5).
2. `Train`: `BeforeTrain`; optional `GradientDiscretizer::DiscretizeGradients` (§4);
   `CUDALeafSplits::InitValues` → root struct (§6.1).
3. For each split, up to `num_leaves−1` (break when `FindBestFromAllSplits`
   returns `best_leaf == −1`):
   a. `HistogramConstructor::ConstructHistogramForLeaf` (smaller leaf) + `SubtractHistogramForLeaf` (larger = parent − smaller) (§7).
   b. `BestSplitFinder::FindBestSplitsForLeaf` (smaller & larger) → `FindBestFromAllSplits` → best `(leaf, feature, threshold)` (§8); 8-int copy-back.
   c. (categorical) `ConstructBitsetForCategoricalSplit` — `SetRealThreshold` + bitset kernels (§6.3).
   d. `CUDATree::Split` / `SplitCategorical` → mutate the model, returns `right_leaf_index` (§10). **← before partition.**
   e. `DataPartition::Split` → reorder row indices into the two child ranges, update leaf bounds (§9); 16-int copy-back.
4. `Train` (optional, quantized): `RenewDiscretizedTreeLeaves` (§9).
5. **[GBDT]** `Tree::Shrinkage(rate)` (§10); `ScoreUpdater::UpdateScore` (§11).
6. **[GBDT]** (optional) `Objective::RenewTreeOutput` (L1/quantile leaf refit, §5.1).
7. **[GBDT]** `Metric::Eval` on train/validation scores (§12).

**Prediction** uses only `CUDATree::AddPredictionToScoreKernel` (§10) over a
`CUDAColumnData` dataset, plus the objective's `ConvertOutputCUDA` inverse link.

---

## 17. Port Considerations (lightgbm_rs / CubeCL)

Connecting the reference to the project's parity contract (CPU f64-fold anchor =
bit-exact merge gate; ROCm f32 = ~1e-6 best-effort):

* **Atomic ordering is non-deterministic** in every accumulating kernel
  (`atomicAdd_block`/`atomicAdd_system` in the histogram builder, leaf-stat reducer,
  binary BoostFromScore, lambdarank). f32 accumulation is therefore not
  bit-reproducible — the documented source of the f32-vs-f64 residual the ROCm gate
  tolerates. The CPU anchor must fold in a **fixed reduction order** to stay
  bit-exact. The `double` score accumulator and `hist_t=double` are load-bearing.
* **The subtraction trick is a correctness requirement, not just a speed one**:
  building the larger child directly would take a different rounding path than
  `parent − smaller`. Reproduce build-smaller / subtract-larger, including the
  histogram-pool pointer swap in `SplitTreeStructureKernel`.
* **Most-frequent-bin omission + fix-up** (`FixHistogramKernel`) and the interleaved
  `[2b]/[2b+1]` layout are behavioral — bin indexing and the reconstructed default-bin
  value must match exactly.
* **Quantized (discretized) mode** is integer arithmetic end-to-end (discretizer →
  int histogram → int split finder → de-quant), hence exactly reproducible — the
  natural bit-exact GPU target. Match the `int16` packing (hess even, grad odd) and
  the scale formulas (`grad_abs_max/(bins/2)`, `hess_abs_max/bins`).
* **Mark→prefix-sum→scatter partitioning** (no sort) and the **block prefix-sum
  conventions** (`ShufflePrefixSum` exclusive/inclusive) must be matched so the row
  permutation — and thus per-leaf data order, which affects f32 accumulation order —
  is identical to the reference.
* **Template-flag explosion** (bin width, weighted, L1/smoothing, 16/32-bit hist,
  global-vs-shared) maps to Rust generics / CubeCL comptime; the shared-vs-global
  spill threshold is a capacity choice with no parity impact as long as the
  in-strategy reduction order is fixed.
* **Shared primitives** (`ShufflePrefixSum`, `ShuffleReduceSum/Max/Min`,
  `BitonicArgSort*`, `PercentileDevice`) are the reusable kernels to port first —
  every subsystem above builds on them.
```
