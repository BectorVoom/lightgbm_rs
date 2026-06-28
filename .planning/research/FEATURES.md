# Feature Research

**Domain:** On-device (whole-tree-on-GPU) histogram tree learner — porting official LightGBM's `CUDASingleGPUTreeLearner` to the lightgbm_rs CubeCL backend (milestone v1.1)
**Researched:** 2026-06-28
**Confidence:** HIGH (read directly from the reference source under `LightGBM/src/treelearner/cuda/` — file:line cited throughout; this is a faithful read of the algorithm to port, not a survey)

> Note: this file supersedes the v1.0 feature-landscape research (preserved in git history) with the v1.1 milestone's algorithm spec.

---

## Executive Summary (read this first)

Official LightGBM's CUDA learner grows the **entire leaf-wise tree on-device**. The host loop issues a fixed, small number of *large* kernels per split — build → subtract → find-best-per-leaf → find-best-across-leaves → partition — and the only host↔device traffic per split is **two tiny scalar buffers** (an 8-int "which leaf/feature/threshold won" packet and a 16-int "child counts/starts/sums" packet). Everything structural — the per-leaf row partition, the histogram pool, the smaller/larger child selection, the histogram-pointer rotation that powers the subtraction trick, and the seeding of the two child `LeafSplits` structs — happens **inside kernels with no host round-trip of bulk data**.

This is the exact inversion of our current host-driven port (`crates/lgbm-treelearner/src/learner.rs`), which runs the same compute bodies but orchestrates them from the host with ~86 launches/tree and a host-resident `HistogramPool` + `DataPartition`. The compute kernels we already have (build, subtract, find_best_split/scan, data_partition) are **reusable**; what is new is the **on-device orchestration state** (`CUDADataPartition`, `CUDALeafSplits` device structs, the `hist_t**` pool, the `CUDABestSplitFinder` cross-leaf reduce, the `SplitTreeStructureKernel`).

**The non-negotiable numerical caveat:** the reference stores histograms and does all gain/leaf-output math in **f64** (`hist_t = double`, `bin.h:33`), but the *hot per-row accumulation* defaults to **f32** (`gpu_use_dp=false`, `config.h:1131`). We must keep the CPU f64 fold bit-exact (anchor/merge gate) while keeping the new CUDA hot loops off f64 (consumer NVIDIA f64 = 1/32 f32 — spike-052 saw a 5.4× regression from one f64-fused kernel). The f64 used in the split-finder/leaf-splits is over **bins and scalars**, not over data rows — that is acceptable and is what the CPU anchor already matches.

---

## On-Device Data Structures (the state to port)

These live resident on the device for the whole tree; the host holds only mirrors/scalars.

| Structure | Owner / file:line | Shape | Role |
|-----------|-------------------|-------|------|
| `cuda_data_indices_` | `cuda_data_partition.hpp:354` | `data_size_t[num_data]` | Row indices permuted so each leaf's rows are contiguous. The partition itself. |
| `cuda_leaf_data_start_` / `_end_` / `cuda_leaf_num_data_` | hpp:355-360 | `data_size_t[num_leaves]` | Per-leaf slice bounds into `cuda_data_indices_`. |
| `cuda_data_index_to_leaf_index_` | hpp:369 | `int[num_data]` | Inverse map row→leaf; used only by `AddPredictionToScore` (and bagging). |
| `cuda_hist_pool_` | hpp:362 | `hist_t*[num_leaves]` | Per-leaf **pointer** into the one big histogram arena. The subtraction trick rotates these pointers (no data copy). |
| `cuda_hist_` (arena) | `cuda_histogram_constructor.hpp:166` | `hist_t[num_leaves * 2 * num_total_bin]` | One contiguous arena; grad+hess interleaved (`2*`). Smaller child gets slot `cuda_hist + 2*right_leaf_index*num_total_bin` (`cuda_data_partition.cu:829`). |
| `CUDALeafSplitsStruct` (×2: smaller/larger) | `cuda_leaf_splits.hpp:21-32` | single struct each | The "current frontier" descriptor: `leaf_index`, `sum_of_gradients/hessians` (f64), `num_data_in_leaf`, `gain`, `leaf_value`, `data_indices_in_leaf*`, `hist_in_leaf*`. Seeded **on-device** by the split kernel. |
| `cuda_leaf_best_split_info_` | `cuda_best_split_finder.hpp:213` | `CUDASplitInfo[num_leaves]` | One best split per existing leaf (persists across iters — only the two touched leaves are recomputed). |
| `cuda_split_info_buffer_` | data_partition.hpp:379 | `int[16]` | The tiny D→H packet from `Split` (child counts/starts/sums). |
| `cuda_best_split_info_buffer_` | best_split_finder.hpp:217 | `int[8]` | The tiny D→H packet from `FindBestFromAllSplits` (winning leaf/feature/threshold/default_left/num_cat). |
| split scratch: `cuda_block_to_left_offset_`, `cuda_block_data_to_left/right_offset_`, `cuda_out_data_indices_in_leaf_` | hpp:367-375 | `[num_data]` / `[max_blocks+1]` | Per-block prefix-sum scratch for the stable partition scatter. |

`CUDASplitInfo` (`cuda_split_info.hpp:16-101`) is the rich per-split record: `is_valid`, `gain`, `inner_feature_index`, `threshold`, `default_left`, and **both children's** `sum_gradients/hessians` (f64), `count`, `gain`, `value`, plus optional categorical `cat_threshold[]`.

---

## Step-Ordered Behavioral Spec — `Train()` (the core deliverable)

Reference: `cuda_single_gpu_tree_learner.cpp:158-345`. Leaf-wise (best-first) growth, `num_leaves - 1` splits max.

### Phase 0 — `BeforeTrain()` (once per tree, L97-152)
1. If boosting not on GPU: copy `gradients_`/`hessians_` **H→D once** (L99-104). This is the only bulk H→D per tree.
2. `cuda_data_partition_->BeforeTrain()` (L113): fills `cuda_data_indices_ = [0..num_data)` (or bagging subset), zeroes leaf arrays, sets leaf 0 to cover all rows (`cuda_data_partition.cpp:120-137`).
3. `cuda_histogram_constructor_->BeforeTrain(grad, hess)` (L132): binds gradient/hessian device pointers (`cuda_histogram_constructor.cpp:76`).
4. `cuda_smaller_leaf_splits_->InitValues(...)` (L133-143): **on-device** reduces root sum_gradients/sum_hessians, writes them back to host `leaf_sum_gradients_[0]`/`[0]` (the root sums are the one early D→H scalar). Sets root `hist_in_leaf` pointer.
5. Host bookkeeping: `leaf_num_data_[0]=root_num_data`, `larger_leaf_splits_->InitValues()` (empty), `col_sampler_.ResetByTree()`, `best_split_finder_->BeforeTrain(...)`, `leaf_data_start_[0]=0`, `smaller_leaf_index_=0`, `larger_leaf_index_=-1` (L145-151).
6. Back in `Train`: create `CUDATree`, set **root leaf output by hand** (not produced by a split) via `CalculateSplittedLeafOutput` and `SyncLeafOutputFromHostToCUDA` (L166-173).

### Phase 1 — per-split loop, `for i in 0..num_leaves-1` (L174-336)
Each iteration grows the tree by exactly one leaf (turns one leaf into two). The "smaller leaf" = the child with fewer rows from the *previous* split (root on iter 0); "larger leaf" = the sibling (−1 on iter 0).

**Step 1 — Construct histogram for the smaller leaf only** (L181-188 → `cuda_histogram_constructor.cpp:117`)
- Builds the histogram for `smaller_leaf_index_` **only**. Early-out if the leaf fails min_data/min_hessian gates (`.cpp:126-128`).
- Reuses our existing build kernel body. f32 accumulation by default (`gpu_use_dp`), promoted into the f64 `hist_t` arena slot `cuda_hist_pool_[smaller]`.

**Step 2 — Subtract for the larger leaf** (L211-217 → `cuda_histogram_constructor.cpp:133`)
- `larger.hist = parent.hist − smaller.hist`, in place over the larger child's arena slot. No build for the larger child. On iter 0 (`larger_leaf_index_ == -1`) this is a no-op for the larger leaf.
- The pointer rotation that makes "parent" available is done in Step 7's split kernel (below).

**Step 3 — Select features by node** (L219 → `.cpp:558`)
- Only active if interaction constraints or `feature_fraction_bynode < 1.0` (`select_features_by_node_`, L59). Copies per-node feature masks **H→D** (small). Otherwise a no-op.

**Step 4 — Find best split per leaf** (L224-242 → `cuda_best_split_finder.cpp:324`)
- Host computes `is_smaller_leaf_valid` / `is_larger_leaf_valid` from counts + min_data/min_hessian gates (`.cpp:337-340`).
- Launches a kernel over **(leaf × feature-scan-task)**: `split_find_tasks_` is a flattened task list (≥1 per feature: forward/reverse/one-hot variants — `SplitFindTask`, best_split_finder.hpp:28-41). Each task scans one feature's histogram with prefix sums, computing best threshold + split gain in **f64** (gain math in `cuda_leaf_splits.hpp:74-140`).
- `LaunchSyncBestSplitForLeafKernel` (`.cpp:350`) reduces per-task results into **one** `cuda_leaf_best_split_info_[leaf]` for each of the (up to two) touched leaves. Both leaves' best splits are computed in this single dispatch. Then one `SynchronizeCUDADevice`.

**Step 5 — Find best leaf across ALL leaves** (L247-273 → `cuda_best_split_finder.cpp:355`)
- `FindBestFromAllSplitsKernel<<<1,256>>>` reduces `cuda_leaf_best_split_info_[0..cur_num_leaves)` to the single global best leaf (`cuda_best_split_finder.cu:2161-2192`). Note: leaves untouched this iter keep their cached best split — only 2 were recomputed, but the reduce scans all.
- `PrepareLeafBestSplitInfo<<<6,1>>>` packs an **8-int** buffer; copied **D→H**: `best_leaf_index_`, smaller/larger best feature/threshold/default_left, `num_cat_threshold_`.
- Returns a **device pointer** into `cuda_leaf_best_split_info_` (`best_split_info`, `.cpp:380`) — the rich split record stays on device.

**Step 6 — Stop check + host tree-structure update** (L276-303)
- If `best_leaf_index_ == -1`: no positive-gain split → `break` (L276-279). (Matches our host learner's early-stop.)
- Categorical: `ConstructBitsetForCategoricalSplit` builds the device bitset (L282-284).
- Host updates the `CUDATree` node structure: `tree->Split(...)` / `SplitCategorical(...)` (L286-303). **This needs host-side metadata**: `RealFeatureIndex`, `RealThreshold`, `FeatureBinMapper`, `missing_type` — i.e. the host keeps the dataset bin-mapper metadata even though the data is on device. The `best_split_info` device pointer is passed through so the tree records child sums/values.

**Step 7 — On-device partition (no bulk round-trip)** (L305-324 → `cuda_data_partition.cpp:139` / `.cu:946`)
This is the heart of "whole tree on-device". Sub-steps:
1. `CalcBlockDim(num_data_in_leaf)` — grid/block sizing (`.cu:276`).
2. `GenDataToLeftBitVector` (`.cpp:194`): per-row, decide left/right from the column bin, threshold, `default_left`, and missing handling. Heavily templated over compile-time bools (`MIN_IS_MAX`, `MISSING_IS_ZERO`, `MISSING_IS_NA`, `MFB_IS_ZERO/NA`, `MAX_TO_LEFT`, `is_single_feature_in_column`) × `BIN_TYPE` (hpp:177-231). Numerical vs categorical (bitset) variants.
3. `AggregateBlockOffsetKernel` (`.cu:970-986`): single-block prefix sum of per-block left/right counts → stable scatter offsets.
4. `SplitInnerKernel` (`.cu:907`): scatter each row into the left or right region of `cuda_out_data_indices_in_leaf_` using the block prefix offsets (stable, order-preserving).
5. `SplitTreeStructureKernel<<<4,5>>>` (`.cu:785`): the on-device bookkeeping kernel. With ~18 single-thread branches it:
   - writes child `leaf_output` values (L802-804),
   - **decides smaller vs larger child on-device** by comparing `cuda_leaf_num_data[left]` vs `[right]` (L825),
   - **rotates the histogram pool pointers**: larger child inherits the parent's hist buffer (so next iter's subtract works), smaller child gets the fresh arena slot `cuda_hist + 2*right_leaf_index*num_total_bin` (L827-831, L895-898),
   - **seeds both child `CUDALeafSplitsStruct`s** in place — sums (f64), counts, gain, value, `data_indices_in_leaf` pointer offset, `hist_in_leaf` pointer, `leaf_index` (L832-904),
   - writes the **16-int** `cuda_split_info_buffer_` for the host (L806-822).
6. Copy the 16-int buffer **D→H** (`.cu:1013`); host reads child `num_data`, `data_start`, and child sum_grad/sum_hess (packed as doubles at offset 8) into `leaf_num_data_[...]`, `leaf_data_start_[...]`, `leaf_sum_*[...]` (L1016-1031).
7. `CopyDataIndicesKernel` (`.cu:1020`): copy the reordered slice back into `cuda_data_indices_` at the parent's `data_start` — the partition is now updated in place.

**Step 8 — Host frontier bookkeeping for next iter** (L328-334)
- Recompute `smaller_leaf_index_` / `larger_leaf_index_` from the host's `leaf_num_data_` mirror (L328-329). (This mirrors the device-side decision in Step 7.5 so the next iteration's launches target the right leaves.)

### Phase 2 — finalize (L337-344)
- `SynchronizeCUDADevice` (L337), optional `RenewDiscretizedTreeLeaves` (quantized only), `tree->ToHost()` pulls the finished tree structure D→H once (L343).

---

## Minimal Host↔Device Communication (the launch-bound budget)

| Moment | Direction | Payload | Notes |
|--------|-----------|---------|-------|
| Per tree (BeforeTrain) | H→D | `grad[]`, `hess[]` (once) | Only bulk H→D; skipped if boosting already on GPU. |
| Per tree (BeforeTrain) | D→H | root sum_grad/sum_hess (2 doubles) | From `InitValues`. |
| Per split — Step 5 | D→H | **8 ints** | best leaf/feature/threshold/default_left/num_cat. |
| Per split — Step 7 | D→H | **16 ints** (incl. 4 doubles packed) | child counts/starts/sums. |
| Per split — Step 3 | H→D | per-node feature mask | **only** under interaction-constraints / `feature_fraction_bynode<1`. |
| Per tree (finalize) | D→H | `CUDATree` structure | `tree->ToHost()`, once. |

**Key insight for the planner:** steady-state per-split traffic is **24 ints** (two tiny packets), not bulk data. But these two D→H copies each carry a `SynchronizeCUDADevice` and **serialize the best-first dependency chain** (build→subtract→find→partition must complete before the next leaf is known). spike-051..054 confirmed on real NVIDIA the wall is **launch latency of this serial chain**, not throughput or sync cost — so the win comes from *fewer, larger launches per split*, exactly what this architecture delivers vs our ~86/tree host loop. The host still serializes per split; the gain is collapsing many small per-feature/per-phase launches into a handful of whole-frontier kernels and removing the host round-trip of indices/histograms.

---

## Feature Landscape

### Table Stakes (must port for behavioral parity)

| Feature | Why Expected (reference) | Complexity | Notes |
|---------|--------------------------|------------|-------|
| On-device row partition (`cuda_data_indices_` + leaf slices) | The partition that lets all later kernels run resident | HIGH | Stable scatter via block prefix sums (`SplitInnerKernel`); reuses our `data_partition` fuse work. |
| On-device histogram arena + `hist_t**` pool with pointer rotation | Subtraction trick without a copy; resident frontier | HIGH | New: our `HistogramPool` is host-side today. Larger child inherits parent buffer (`.cu:827-831`). |
| Build-smaller / subtract-larger per split | The core histogram economy; only 1 build per split | MEDIUM | Compute bodies already exist; need on-device pool wiring. |
| Per-leaf best-split scan over `split_find_tasks_` | Find best threshold+gain per feature, f64 gain math | MEDIUM | Reuses our find_best_split/scan kernel; gain math in `cuda_leaf_splits.hpp`. |
| Cross-leaf best-leaf reduce (`FindBestFromAllSplits`) | Best-first growth picks the globally best leaf | MEDIUM | New on-device reduce + 8-int D→H packet. |
| On-device `SplitTreeStructureKernel` (child seeding + smaller/larger pick + pool rotation) | Eliminates the host round-trip of child state | HIGH | The structural heart; 16-int D→H mirror for host. |
| Host tree-structure mirror (`CUDATree` + bin-mapper metadata) | `tree->Split` needs RealFeature/RealThreshold/missing_type on host | LOW | Host keeps dataset metadata; only scalars cross. |
| min_data_in_leaf / min_sum_hessian gates + "no positive gain" stop | Parity with serial learner stopping rules | LOW | Already in our host learner (`learner.rs`); replicate the gate placement (host-side validity flags). |
| Coexistence / feature-gate with existing host-orchestrated path | PROJECT.md non-negotiable — ROCm + CPU routing untouched | MEDIUM | New learner is an alternate path, not a replacement. |

### Differentiators (advanced; port after the core slice proves out)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Quantized-gradient path (`use_quantized_grad`) | The int16/u64 fixed-point fast path; aligns with our "no f64 hot loops" mandate | HIGH | Reference `cuda_gradient_discretizer_`, per-leaf bit-width selection (`GetHistBitsInLeaf`). Maps to v2 QNT-01 + our shipped Phase-10 quantized mode. |
| Categorical splits on-device (bitset gen) | Full parity for categorical features | MEDIUM | `ConstructBitsetForCategoricalSplit`, `SplitCategorical`, `cat_threshold[]` in `CUDASplitInfo`. |
| Bagging subset (`SetBaggingData` / `use_bagging_`) | Resident subset indices | MEDIUM | `cuda_data_index_to_leaf_index_` inverse map; restore order in `UpdateTrainScore`. |
| `select_features_by_node_` (interaction constraints / feature_fraction_bynode) | Per-node feature sampling | MEDIUM | The only steady-state H→D in the loop; gate it off by default. |
| On-device score update (`AddPredictionToScoreKernel`) + leaf renew | Keep scores resident between trees (boosting on GPU) | MEDIUM | `UpdateTrainScore`, `RenewDiscretizedTreeLeaves`, `FitByExistingTree`/refit. |
| `gpu_use_dp` toggle (f64 hist accumulation) | Optional accuracy mode | LOW | Default false (f32). Our CPU anchor is the f64 reference already. |

### Anti-Features (do NOT port / avoid)

| Feature | Why It Looks Tempting | Why Problematic | Alternative |
|---------|----------------------|-----------------|-------------|
| f64 hot-loop kernels (per-row accumulation in double) | "Matches the f64 `hist_t` storage exactly" | Consumer NVIDIA f64 = 1/32 f32; spike-052 measured **5.4× regression** from one f64-fused kernel | Keep f32 accumulation (default `gpu_use_dp=false`) + the u64 fixed-point build path; pin parity to the CPU f64 anchor, not to device f64. |
| Per-`SplitInfo` device `cudaMalloc`/`new` for cat_threshold (`cuda_split_info.hpp:81-95`) | Faithful to reference | Per-split device heap alloc is slow + awkward in CubeCL | Pre-allocate cat-threshold vectors (`AllocateCatVectors`, `.cu:2196`) — the reference itself does this for the leaf array. |
| Forcing `build_fix_scan` fusion on CUDA | "Fewer launches must be faster" | spike-052: 5.4× worse on real NVIDIA (it was the f64 kernel) | Fuse *orchestration* (whole-frontier kernels), not the f64 math path. |
| Tuning build occupancy / row-partition `P` for CUDA | APU showed ~10% sensitivity | spike-053: refuted on real NVIDIA (P=1 optimal); APU signs don't transfer | Don't lift `BUILD_PSET` for CUDA; let autotune self-calibrate. |
| Replacing the host-orchestrated / ROCm / CPU path | "One unified learner" | Breaks the bit-exact CPU merge gate + ROCm parity routing | Add as a feature-gated alternate path (PROJECT.md non-negotiable). |
| Multi-stream micro-optimization first | Reference uses 4 streams | Sync-cost has no headroom (spike-052: ~0.14ms/sync, co-pack already banks it) | Get the single-stream on-device loop correct first; streams later if measured. |

---

## Feature Dependencies

```
On-device row partition (cuda_data_indices_ + leaf slices)
    └──requires──> Stable block-prefix-sum scatter (SplitInnerKernel + AggregateBlockOffset)

On-device histogram arena + hist_t** pool
    └──requires──> On-device row partition (leaf slices index the arena)
    └──enables───> Build-smaller / subtract-larger economy
                       └──requires──> SplitTreeStructureKernel pool-pointer rotation

Cross-leaf best-leaf reduce (FindBestFromAllSplits)
    └──requires──> Per-leaf best-split scan (cuda_leaf_best_split_info_[])

SplitTreeStructureKernel (child seeding + smaller/larger pick + pool rotation)
    └──requires──> On-device partition + histogram pool + CUDASplitInfo on device
    └──produces──> the 16-int host mirror that drives the next iteration's launches

Host CUDATree + bin-mapper metadata ──required-by──> tree->Split (RealThreshold/RealFeature)

Quantized path ──enhances──> Build/subtract/find (separate int16/u64 kernels)
Categorical splits ──requires──> bitset gen + cat_threshold vectors (pre-allocated)
Bagging ──requires──> cuda_data_index_to_leaf_index_ inverse map
```

### Dependency Notes
- **Histogram pool requires the partition:** leaf slices (`cuda_leaf_data_start/num_data`) index both the row array and the arena slot; build the partition first.
- **Subtraction trick requires the pool-pointer rotation:** the larger child must inherit the parent's buffer *before* Step 2's subtract — that rotation is done in the *previous* iteration's `SplitTreeStructureKernel` (`.cu:827-831`). Ordering is load-bearing.
- **Host tree update requires host metadata:** even fully on-device, `tree->Split` needs `RealThreshold`/`RealFeatureIndex`/`missing_type` from the host bin-mapper (L297-303) — keep dataset metadata host-resident.
- **Best-first stop:** `best_leaf_index_ == -1` is computed by the device reduce and surfaced via the 8-int packet — the host loop branches on it (L276).

---

## MVP Definition

### Launch With (v1.1 core slice)
- [ ] On-device row partition + leaf slices (`cuda_data_indices_`, start/num_data) — without this nothing stays resident.
- [ ] On-device histogram arena + `hist_t**` pool with the subtraction-trick pointer rotation — the resident frontier.
- [ ] Build-smaller / subtract-larger per split, reusing existing build/subtract kernels.
- [ ] Per-leaf best-split scan + cross-leaf best-leaf reduce (8-int D→H packet).
- [ ] `SplitTreeStructureKernel` equivalent: on-device child seeding + smaller/larger selection + 16-int host mirror.
- [ ] Numerical scope: **continuous features, no bagging, no quantization** — match the CPU f64 anchor (bit-exact merge gate) and hold CUDA to ~1e-6.
- [ ] Feature-gated coexistence with the host-orchestrated / ROCm / CPU paths.
- [ ] Real-CUDA (Kaggle) A/B vs official as the verification surface (target: materially below today's 3.9×@50f / 1.9×@500f).

### Add After Validation (v1.x)
- [ ] Categorical splits on-device (bitset + pre-allocated cat_threshold) — trigger: core slice is bit-exact on continuous.
- [ ] Bagging subset path (`SetBaggingData`, inverse leaf map) — trigger: GOSS/bagging configs needed on GPU.
- [ ] On-device score update / boosting-on-GPU resident scores — trigger: per-tree H↔D of scores shows up in the profile.
- [ ] `select_features_by_node_` (interaction constraints / feature_fraction_bynode).

### Future Consideration (v2+)
- [ ] Quantized-gradient on-device path (`use_quantized_grad`, int16/u64) — defer: large surface; maps to v2 QNT-01 and our existing CPU-only Phase-10 quantized mode.
- [ ] `gpu_use_dp` f64-accumulation accuracy mode — defer: anchor already covers f64 reference.
- [ ] Refit / `FitByExistingTree` / leaf renew on GPU — defer: not on the train-speed critical path.
- [ ] Multi-stream overlap of the 4 reference streams — defer: measure first (sync has no headroom per spike-052).

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| On-device partition + leaf slices | HIGH | HIGH | P1 |
| Histogram arena + pool pointer rotation | HIGH | HIGH | P1 |
| Build-smaller / subtract-larger (reuse kernels) | HIGH | MEDIUM | P1 |
| Per-leaf scan + cross-leaf reduce | HIGH | MEDIUM | P1 |
| SplitTreeStructureKernel (on-device child seed) | HIGH | HIGH | P1 |
| Feature-gated coexistence | HIGH | MEDIUM | P1 |
| Categorical on-device | MEDIUM | MEDIUM | P2 |
| Bagging subset | MEDIUM | MEDIUM | P2 |
| On-device score update | MEDIUM | MEDIUM | P2 |
| select_features_by_node | LOW | MEDIUM | P2 |
| Quantized on-device path | MEDIUM | HIGH | P3 |
| gpu_use_dp f64 mode | LOW | LOW | P3 |
| Refit / leaf renew on GPU | LOW | MEDIUM | P3 |

---

## Reuse Map (existing lightgbm_rs assets the planner can lean on)

| Reference component | Existing asset to reuse | Gap to close |
|---------------------|-------------------------|--------------|
| `CUDAHistogramConstructor` build kernel | Our build kernel (Phase 4 + u64 fixed-point, spike-018) + `resident_pool.rs` | Wire it to write into a per-leaf arena slot chosen by an on-device pool, not a host pool. |
| `SubtractHistogramForLeaf` | Our subtract kernel + `fix_histogram.rs` | Drive it off the on-device pool pointers (parent inherited by larger child). |
| `FindBestSplitsForLeaf` (per-feature scan) | Our `find_best_split`/scan kernel (feature-per-lane, spike-021) + `split_info.rs` | Add the per-leaf validity gating + write into a persistent `cuda_leaf_best_split_info_[]`. |
| `FindBestFromAllSplits` cross-leaf reduce | **New** | No host-side analog — the host `learner.rs` picks best leaf on host today. |
| `CUDADataPartition::Split` | Our `data_partition.rs` (fused-gather + narrow-upload wins, spikes 027/029) | Move leaf bookkeeping + smaller/larger pick + child seeding **on-device** (`SplitTreeStructureKernel`); today it's host-side `DataPartition::split`. |
| `CUDALeafSplitsStruct` | `leaf_splits.rs` (`LeafSplits`) | Promote to a device-resident struct seeded by the split kernel, not host-init each iter. |
| `cuda_hist_pool_` (`hist_t**`) | `histogram_pool.rs` / `resident_pool.rs` (host-side flat arena) | Make the pool + pointer rotation device-resident. |
| Host growth loop | `learner.rs::train_inner` (build→fix→compact→subtract→scan→partition per leaf) | Same algorithm; replace host orchestration with the few-large-kernel device loop. |

**Net new device-resident state to build:** the `hist_t**` pool rotation, the two persistent `CUDALeafSplitsStruct`s, the persistent `cuda_leaf_best_split_info_[num_leaves]`, the cross-leaf reduce, and the `SplitTreeStructureKernel` (child-seed + smaller/larger pick + 16-int mirror). Everything else is orchestration change around kernels we already have.

---

## f64 / f32 Usage in the Reference (parity flags)

| Location | Type | Hot loop? | Our stance |
|----------|------|-----------|------------|
| `hist_t` histogram storage (`bin.h:33`) | **f64** | storage, not a loop | Keep f64 *storage*; CPU anchor matches. |
| Per-row histogram accumulation (`gpu_use_dp`, `config.h:1131` default false) | **f32** | YES (per data row) | Keep f32 (+ u64 fixed-point build). **Never** f64 here (spike-052: 5.4× regression). |
| `CUDALeafSplitsStruct` sums, gain, value (`cuda_leaf_splits.hpp:21-32`) | **f64** | scalar | Acceptable — scalars per leaf, matches anchor. |
| Split-gain math (`GetSplitGains`/`CalculateSplittedLeafOutput`, `cuda_leaf_splits.hpp:74-140`) | **f64** | over **bins**, not rows | Acceptable — bin-count loop, matches CPU f64 anchor; this is *why* the CPU anchor stays the bit-exact gate. |
| `CUDASplitInfo` sums/gains (`cuda_split_info.hpp`) | **f64** | scalar | Acceptable. |
| Quantized path (`int16` packed grad+hess, `int_hist_t`) | **int / u64 fixed-point** | YES | The fast hot path; aligns with "no f64 hot loops" mandate. Defer to v2. |

**Bottom line for the planner:** the f64 in the reference is confined to histogram *storage* and *bin/scalar* gain math — exactly what our CPU f64 anchor already reproduces bit-exact. The only place f64 would land on a per-row hot loop is the accumulation, and the reference itself defaults that to f32. Port accordingly: f32 (+ u64 fixed-point) per-row accumulation, f64 bin/scalar gain math, CPU f64 fold remains the merge gate, CUDA held to ~1e-6.

---

## Sources

- `LightGBM/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp:97-345` (the growth loop) — HIGH (primary reference source, direct read)
- `LightGBM/src/treelearner/cuda/cuda_data_partition.{hpp,cpp,cu}` (on-device partition, `SplitTreeStructureKernel` `.cu:785-1032`) — HIGH
- `LightGBM/src/treelearner/cuda/cuda_best_split_finder.{hpp,cpp,cu}` (per-leaf scan + cross-leaf reduce `.cu:2161`) — HIGH
- `LightGBM/src/treelearner/cuda/cuda_histogram_constructor.{hpp,cpp}` (arena + build/subtract drivers) — HIGH
- `LightGBM/src/treelearner/cuda/cuda_leaf_splits.hpp` (`CUDALeafSplitsStruct` + f64 gain math) — HIGH
- `LightGBM/include/LightGBM/cuda/cuda_split_info.hpp` (`CUDASplitInfo`) — HIGH
- `LightGBM/include/LightGBM/bin.h:33` (`hist_t = double`), `config.h:1131` (`gpu_use_dp=false`) — HIGH
- `crates/lgbm-treelearner/src/learner.rs` (existing host growth loop being replaced) — HIGH
- `.claude/skills/spike-findings-lightgbm_rs/references/cuda-architectural-launch-bound.md` (spikes 051–054: launch-bound mechanism, f64/occupancy/fusion verdicts) — HIGH
- `.planning/PROJECT.md` (v1.1 milestone goal + non-negotiables) — HIGH

---
*Feature research for: on-device CUDA single-GPU tree learner port (milestone v1.1)*
*Researched: 2026-06-28*
