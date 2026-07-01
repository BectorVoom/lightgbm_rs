# Phase 18: On-Device Data Partition, Tree Mutation & Prediction - Research

**Researched:** 2026-07-01
**Domain:** CUDA→CubeCL kernel port — device data partition (`cuda_data_partition.cu` §9), tree mutation + prediction (`cuda_tree.cu` §10), anchored to the cubecl-cpu f64 fold
**Confidence:** HIGH (all findings verified against the in-tree AMD-fork reference source `LightGBM-release-4.6.0.99/` and the existing `lgbm-compute` crate)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Build a NEW §9-faithful on-device `mark → prefix-sum → scatter` partition — parallel to the shipped host-gather `data_partition_kernel`, NOT an extension of it. The existing `partition.rs` is a routing-decision *reference* only.
- **D-02:** Wire the FULL `GenDataToLeftBitVector` template flag fan-out now — `<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, USE_MIN_BIN, BIN_TYPE>` as CubeCL comptime.
- **D-03:** Build the categorical partition routing FULLY this phase (`GenDataToLeftBitVectorKernel_Categorical` / `UpdateDataIndexToLeafIndexKernel_Categorical` via `CUDAFindInBitset`) — membership only, anchored against a fixture model with categorical splits. Cat split-FINDING stays Phase 22.
- **D-04:** The cpu f64 anchor reproduces the reference order via a single-owner STABLE partition (left rows in original relative order, then right rows in original relative order) — order-equivalent to §9's block-tiled scatter. **Research MUST first verify order-equivalence; if the reference order is NOT a plain stable partition, escalate to the faithful block-tiled anchor.** → **RESOLVED CONFIRMED below (§ "D-04 Gate").**
- **D-05:** Build the §10 tree-walk `AddPredictionToScoreKernel<USE_INDICES>` with FULL numeric AND categorical membership math this phase (8/16/32 column-width dispatch, `FindInBitsetCUDA`), anchored to the cpu f64 fold via fixture models.
- **D-06:** Include §9's `AddPredictionToScoreKernel<USE_BAGGING>` (per-row leaf-output add via the data-index→leaf map) in THIS phase.
- **D-07:** Introduce a device-resident flat `CUDATree`; `SplitKernel` mutates it BEFORE the partition step and returns `right_leaf_index`. Host `lgbm_model::Tree` reconstructed at the end of the per-tree device run for the anchor comparison.
- **D-08:** Accept TWO small device→host transfers per split — Phase-17 8-int best-split buffer + this phase's 16-int `SplitTreeStructure` child-stats packet.
- **D-09:** The `SplitTreeStructureKernel` histogram-pool pointer swap integrates with the Phase-16 `HistArena` handle rotation — smaller/larger child roles drive the parent-minus-sibling subtraction-trick reuse. Reuse the arena's `hist_t**` handle rotation; do not rebuild it.
- **D-10:** When `LGBM_CUDA_ON_DEVICE=1`, route the partition on-device unconditionally — NO size gate this phase (spike-035 APU overhead accepted; Phase-23 DoD tunes routing).
- **D-11:** Capture NEW C++ `lib_lightgbm` goldens for this phase — partition row-order (post-scatter index array), tree-walk predict over numeric AND categorical models, and the 16-int child-stats packet. Reuse Phase-15/16 fixtures where they fit.
- **D-12:** Anchor every numeric output to the cubecl-cpu f64 fold; structure bit-exact; ROCm/CUDA f32 within ~1e-6; tie-aware where relevant; **never GPU-vs-GPU** (def-f8u-01). One `#[cube]` generic, comptime/runtime-split reduction order.
- **D-13:** `LGBM_CUDA_ON_DEVICE` OFF by default; CPU / ROCm / existing-host-CUDA paths byte-unchanged; full merge gate green with the env unset (ODL-19 hard merge gate).
- **D-14:** NO f64 per-row hot loops in new kernels (5.4× consumer-NVIDIA f64 regression, spike-052); the `double* score` accumulator and scalar gain/output math stay f64 where the reference uses it (§17).
- **D-15:** Pre-allocate the scatter scratch + block-offset buffers ONCE outside the hot loop — no per-split in-kernel device alloc (Phase-17 D-11 pattern).

### Claude's Discretion
- Exact CubeCL module placement — likely a new `data_partition.rs` + `tree.rs` (or `predict.rs`) in `kernels/`, reusing `split_info.rs` and the Phase-14 primitives for the block prefix-sum.
- Whether `AggregateBlockOffsetKernel0` (large-grid serial chunk-scan) vs `…1` (one-shot warp scan) are two kernels or one runtime-branched kernel — parity-neutral as long as the block-winner prefix order is fixed.
- Geometry tunables (§9 block sizes 1024; `SplitKernel` `<<<3,5>>>`, `SplitCategoricalKernel` `<<<3,6>>>`, `SplitTreeStructureKernel` `<<<4,5>>>` scalar fan-outs) — occupancy knobs, no parity impact; start from the faithful C++ constants; APU-aware autotune deferred.
- The exact device→host reconstruction point for the host `lgbm_model::Tree` (per-split vs per-tree) — parity-neutral as long as the final structure matches the anchor.

### Deferred Ideas (OUT OF SCOPE)
- Categorical split-FINDING / cat-feature end-to-end training → **Phase 22** (this phase builds only the membership routing).
- §11 score-updater constant scalar ops (`AddScoreConstant` / `MultiplyScoreConstant`) + §12 metrics → **Phase 20**.
- On-device objectives / `ConvertOutput` inverse-link → **Phase 19** (predict parity here uses the host objective inverse-link at the readback boundary).
- Discretized/quantized partition + `RenewDiscretizedTreeLeavesKernel` → **v2 (QGD)**.
- APU-vs-discrete partition routing tuning + size-gate (spike-035) → **Phase 23 perf/rollout DoD**.
- APU-aware autotune of §9/§10 geometry → deferred perf option (parity-neutral).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-13 | On-device data partition — `mark → prefix-sum → scatter` (never sorting) into two contiguous child ranges, the data-index→leaf-index map, and the `SplitTreeStructure` histogram-pool pointer swap; resulting row order matches the reference (§9). | Kernel Inventory Map + D-04 Gate (order-equivalence CONFIRMED) + HistArena integration below. `primitives.rs` already ships block/global prefix-sum; the gap is **integer-typed** scan launchers (u16/u32) — flagged. |
| ODL-14 | On-device tree mutation — `Split` writing the device tree arrays, ordered BEFORE partition (returns `right_leaf_index`), plus `Shrinkage`/`AddBias`, anchor-pinned to the host tree structure (§10, §1 ordering). | `SplitKernel`/`ShrinkageKernel`/`AddBiasKernel` mapped below; `split_info.rs::SplitScalars` is the `CUDASplitInfo` source. Split-before-partition invariant confirmed in reference host orchestration. |
| ODL-15 | On-device prediction — tree-walk `AddPredictionToScore` over the device columnar dataset (numeric threshold + missing/`default_left`, categorical bitset membership), within ~1e-6 + objective inverse-link (§10). | Full `AddPredictionToScoreKernel<USE_INDICES>` transcribed below (verified line-by-line); reads `column_data.rs`/`row_data.rs` 8/16/32 dispatch; inverse-link stays host-side at readback (Phase-19 boundary). |
</phase_requirements>

## Summary

Phase 18 ports three tightly-coupled CUDA translation units to CubeCL behind `LGBM_CUDA_ON_DEVICE`: the data-partition row router (`cuda_data_partition.cu` §9), the flat device tree mutation kernels (`cuda_tree.cu` §10), and the tree-walk prediction kernel. All three are **additive** — the CPU/ROCm/host-CUDA default paths stay byte-unchanged (D-13), and `on_device_growth_supported()` stays `false`. The reference source is available in-tree at `LightGBM-release-4.6.0.99/` (the AMD HIP fork, per MEMORY `rocm-baseline-amd-fork`), so every kernel was transcribed line-by-line rather than reconstructed from memory — confidence is HIGH.

The single highest-value research output was the **D-04 order-equivalence gate**, and it is **RESOLVED CONFIRMED**: the reference block-tiled `mark → prefix-sum → scatter` produces **exactly** a plain single-owner stable partition (all left rows in original relative order in `[0, to_left_total)`, all right rows in original relative order after). No escalation is needed — the cpu f64 anchor can be a simple exact stable partition. Proof from the reference source is in the D-04 section below.

This phase does **not** install any external dependencies — `cubecl` (0.10) is already the workspace compute backend; there is no Package Legitimacy Audit or Environment Availability concern beyond the already-present ROCm/cubecl-cpu toolchain. The chief *implementation* risks are: (1) integer-typed block prefix-sums (`primitives.rs` currently only instantiates f32/f64 scan launchers — u16/u32 scans are a genuine gap); (2) the histogram-pool whole-tree SWAP, which the Phase-16 `HistArena` module explicitly deferred to Phase 18; and (3) the eight-way template-flag fan-out of `GenDataToLeftBitVector` mapping cleanly to CubeCL comptime.

**Primary recommendation:** Build a new `kernels/data_partition.rs` (mark→prefix-sum→scatter) + `kernels/tree.rs` (SplitKernel/Shrinkage/AddBias) + `kernels/predict.rs` (tree-walk), reuse `primitives.rs` scans (adding u16/u32 launcher instantiations), `split_info.rs::SplitScalars` as the `CUDASplitInfo` record, and extend `HistArena` with a leaf-indexed whole-pool swap. Anchor the cpu f64 fold to a plain exact stable partition (D-04 confirmed), extend `xtask/cpp/kernel_capture.cpp` for the new goldens.

## D-04 Gate: Order-Equivalence RESOLVED — CONFIRMED (highest-value output)

**Verdict: The reference §9 block-tiled scatter is order-equivalent to a plain single-owner stable partition. NO escalation. The cpu f64 anchor D-04 design is VALID as specified.**

Evidence (read line-by-line from `LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_data_partition.cu` and `include/LightGBM/cuda/cuda_algorithms.hpp`):

1. **`ShufflePrefixSum` is INCLUSIVE.** `cuda_algorithms.hpp:33` `ShufflePrefixSum(value, …)` accumulates including the thread's own value; a separate `ShufflePrefixSumExclusive` exists at `:64`. `[VERIFIED: cuda_algorithms.hpp:33-63]`

2. **Blocks cover consecutive row chunks in current leaf-index order.** In `SplitInnerKernel` (`:918`) `global_thread_index = blockIdx.x*blockDim.x + threadIdx.x` indexes `cuda_data_indices_in_leaf` directly, so block `b` owns leaf-local rows `[b·blockDim, (b+1)·blockDim)`. `[VERIFIED: cuda_data_partition.cu:914-919]`

3. **Within a block, order is preserved on BOTH sides.** `PrepareOffset` (`:57`) writes each thread's inclusive left-rank into `block_to_left_offset[threadIdx_x]`. In `SplitInnerKernel` (`:926-932`): a left row lands at `left_out[thread_to_left_offset]` where `thread_to_left_offset = block_to_left_offset_ptr[threadIdx_x-1]` = the **exclusive** left rank (thread order preserved); a right row lands at `right_out[threadIdx.x − thread_to_left_offset]` = its exclusive right rank within the block (thread order preserved). `[VERIFIED: cuda_data_partition.cu:926-933]`

4. **Block base offsets preserve block order.** `AggregateBlockOffsetKernel{0,1}` (`:681`,`:745`) exclusive-prefix-sums the per-block left counts, so block `b`'s left base = Σ(left counts of blocks `0..b-1`); the right base is shifted by `to_left_total` so `right_base(b) = to_left_total + Σ(right counts of blocks 0..b-1)` (`:729`,`:771`). `[VERIFIED: cuda_data_partition.cu:715-783]`

5. **Composite.** Left region = `[block0 left rows | block1 left rows | …]`, each block in original row order → **all left rows in original relative order** occupying `[0, to_left_total)`. Right region after `to_left_total` = `[block0 right | block1 right | …]` → **all right rows in original relative order**. This is *exactly* `stable_partition(pred)`.

**Planning consequence:** the cpu f64 anchor is `data_indices[leaf_slice]` stably partitioned by the per-row route predicate — left-keepers first (original order), right-keepers second (original order), written back over the parent slice. Because row order fixes per-leaf f32 accumulation order (§17), this makes the *next* iteration's histogram fold bit-identical. The hip kernel runs the real 4-stage scatter; both land the same permutation. **The escalation branch (faithful block-tiled anchor with `min_num_blocks = num_data≤100 ? 1 : 80`, power-of-two `block_dim ≤ 1024`) is NOT triggered and can be dropped from planning.**

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Row route decision (bin vs threshold + missing/default) | Device kernel (`GenDataToLeftBitVector`) | — | Shared decision math also used by predict tree-walk (§/D-05); one transcription. |
| Mark → prefix-sum → scatter permutation | Device kernel (`primitives` block scan + `SplitInner`) | Host orchestration (stream sequencing) | Data-parallel; never sort (§17). |
| data-index→leaf map update | Device kernel (`UpdateDataIndexToLeafIndex`) | — | Consumed by §9 `AddPredictionToScore` leaf-map add. |
| Child-range + leaf-stats bookkeeping | Device kernel (`AggregateBlockOffset` + `SplitTreeStructure`) | Host (16-int readback → host bookkeeping) | Scalar fan-out; 16-int packet crosses back. |
| Histogram-pool pointer swap (subtraction trick) | Host `HistArena` handle rotation (D-09) | Device (leaf→slot handle table) | Correctness requirement; reuse Phase-16 arena. |
| Tree mutation (Split/Shrinkage/AddBias) | Device flat `CUDATree` (`SplitKernel`…) | Host reconstruction for anchor compare | Split BEFORE partition (§1/§10 ordering). |
| Prediction tree-walk | Device kernel (`AddPredictionToScore<USE_INDICES>`) | Host objective inverse-link at readback (Phase-19 boundary) | Reads §13 columnar dataset 8/16/32 dispatch. |
| Objective inverse-link / `ConvertOutput` | **Host (this phase)** | Phase 19 (device) | Explicitly out of scope; predict parity uses host inverse-link. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10 (in-tree) | `#[cube]` kernels, `SharedMemory`/`sync_cube()`/plane intrinsics, `client.empty` prealloc | Already the workspace compute backend; cubecl-cpu = f64 bit-exact anchor, cubecl-hip = ~1e-6 gate `[VERIFIED: crates/lgbm-compute/src/kernels/primitives.rs:50]` |
| `lgbm-compute` (`kernels/`) | workspace | Host of all device kernels + `Backend` seam | `grow_tree_on_device` seam + `LeafPartitionLayout` already stubbed `[VERIFIED: crates/lgbm-compute/src/lib.rs:1282-1314]` |

**No external packages are installed this phase.** `cubecl` and the cubecl-cpu/cubecl-hip backends are already workspace dependencies. There is no npm/PyPI/crates *addition* → the Package Legitimacy Audit is **N/A** (no new packages). `cargo 1.95.0 / rustc 1.95.0` verified present.

### Reused In-Tree Assets (DO NOT rebuild)
| Asset | Path | Role in Phase 18 |
|-------|------|------------------|
| Block/global prefix-sum + reductions | `kernels/primitives.rs` | `mark → prefix-sum → scatter` building blocks; `AggregateBlockOffset` fold. **GAP: only f32/f64 launchers exist — needs u16/u32 integer scan instantiations.** `[VERIFIED: primitives.rs:160-351]` |
| `SplitScalars` / `DeviceSplitInfo` | `kernels/split_info.rs` | The `CUDASplitInfo` record `SplitKernel` + `SplitTreeStructure` read (`default_left`, per-side sums/counts/gains/values, cat slabs) `[VERIFIED: split_info.rs:81-136]` |
| `HistArena` handle rotation | `kernels/histogram_arena.rs` | The pool pointer SWAP (D-09). Module docs: whole-tree pool SWAP is **explicitly Phase 18** `[VERIFIED: histogram_arena.rs:29-30]` |
| `CudaColumnData` / `CudaRowData` | `kernels/column_data.rs`, `row_data.rs` | §13 resident 8/16/32-width columnar store predict + partition read directly (no re-upload) |
| Host-gather `data_partition_kernel` | `kernels/partition.rs` | **Routing-decision reference ONLY** (per-row `SplitInner` `MissingType::None`) — D-01 builds the new device path parallel to it |
| Backend seam | `lib.rs` `grow_tree_on_device` + `LeafPartitionLayout` | Reached only when `LGBM_CUDA_ON_DEVICE=1`; `on_device_growth_supported` stays `false` `[VERIFIED: lib.rs:1239,1314]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New device mark→scatter (D-01) | Extend host-gather `partition.rs` | Rejected in discuss — blurs the §9-faithful anchor + block-tiled order; keep host-gather as decision reference only |
| Plain stable-partition anchor (D-04) | Faithful block-tiled scatter anchor | Not needed — order-equivalence CONFIRMED above; the block-tiled anchor is strictly more complex for identical output |

## Kernel Inventory Map (research target #1)

Every §9/§10 kernel named in CONTEXT.md, its exact reference location, what it computes, its template flags, and the closest existing CubeCL primitive to build on. Reference = `LightGBM-release-4.6.0.99/`.

| Kernel | Ref location | Computes | Template flags | CubeCL primitive to build on |
|--------|--------------|----------|----------------|------------------------------|
| `FillDataIndicesBeforeTrainKernel` | `cuda_data_partition.cu:21` | `data_indices[i]=i`, all rows→leaf 0 | — | trivial `#[cube]` init / `iota` |
| `FillDataIndexToLeafIndexKernel` | `cuda_data_partition.cu:30` | reset row→leaf map to root | — | trivial fill |
| `GenDataToLeftBitVectorKernel` | `cuda_data_partition.cu:290` | per-row `to_left` mark + block `ShufflePrefixSum<uint16_t>` → `block_to_left_offset` + per-block totals | `<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, USE_MIN_BIN, BIN_TYPE>` (D-02) | `primitives.rs` block scan (needs **u16** instantiation); route decision from `partition.rs` |
| `GenDataToLeftBitVectorKernel_Categorical` | `cuda_data_partition.cu:582` | same but membership via `CUDAFindInBitset(bitset, len, bin−min_bin+mfb_offset)` | `<BIN_TYPE, USE_MIN_BIN>` (D-03) | `FindInBitsetCUDA` port + block scan |
| `UpdateDataIndexToLeafIndexKernel` (+`_Categorical`) | `cuda_data_partition.cu:113`,`:557` | same decision, writes destination leaf id into `cuda_data_index_to_leaf_index` (stream 3, concurrent) | `<…, BIN_TYPE>` | reuse route decision; independent scatter-free write |
| `PrepareOffset` (device helper) | `cuda_data_partition.cu:52` | block-local inclusive scan + write per-block left/right totals | — | `ShufflePrefixSum<uint16_t>` = `primitives.rs` block-scan body |
| `AggregateBlockOffsetKernel0` | `cuda_data_partition.cu:681` | `grid_dim>1024`: serial chunk-scan + `ShufflePrefixSum<uint32_t>` over `num_blocks+1` per-block counts; shift right by `to_left_total`; write child `leaf_data_start/end/num_data` | `<<<1,1024>>>` | `primitives.rs` global block-totals scan (needs **u32** instantiation) |
| `AggregateBlockOffsetKernel1` | `cuda_data_partition.cu:745` | `grid_dim≤1024`: one-shot warp `ShufflePrefixSum<uint32_t>` | `<<<1, num_blocks_final_aligned>>>` | same; discretion: one runtime-branched kernel or two (parity-neutral) |
| `SplitInnerKernel` | `cuda_data_partition.cu:909` | the scatter: recover in-block exclusive left rank; left→`out[left_base+rank]`, right→`out[right_base+(tid−rank)]` in scratch `cuda_out_data_indices_in_leaf_` | — | `#[cube]` scatter reading the two offset buffers |
| `SplitTreeStructureKernel` | `cuda_data_partition.cu:787` | scalar fan-out: child leaf outputs, pack **16-int** `cuda_split_info_buffer`, assign smaller/larger roles + **swap histogram-pool pointers** (D-09) | `<<<4,5>>>` | scalar `#[cube]` + `HistArena` leaf-indexed swap |
| `CopyDataIndicesKernel` | `cuda_data_partition.cu:937` | copy reordered scratch back over parent slice (finalize contiguity) | — | trivial copy |
| `AddPredictionToScoreKernel<USE_BAGGING>` (§9) | `cuda_data_partition.cu` (launcher) | per row: look up leaf, add leaf output to `double* cuda_scores` (D-06) | `<USE_BAGGING>` (indices vs identity) | `#[cube]` gather-add; `double` accumulator (D-14) |
| `SplitKernel` (§10) | `cuda_tree.cu:48` | one numerical split: rewire parent/child links + 14 field writes from `CUDASplitInfo` (NaN→0) | `<<<3,5>>>` (15 threads) | scalar `#[cube]`; consumes `split_info.rs::SplitScalars` |
| `SplitCategoricalKernel` | `cuda_tree.cu:160` | same fan-out + `kCategoricalMask`, `num_cat`, extend `cat_boundaries`/`_inner` by bitset lengths | `<<<3,6>>>` (17 threads) | scalar `#[cube]` + cat slab append |
| `ShrinkageKernel` | `cuda_tree.cu:290` | `leaf_value *= rate` | `<<<ceil(nleaves/1024),1024>>>` | elementwise `#[cube]` (f64 scalar, D-14) |
| `AddBiasKernel` | `cuda_tree.cu:303` | `leaf_value += val` | — | elementwise `#[cube]` |
| `AddPredictionToScoreKernel<USE_INDICES>` (§10 predict) | `cuda_tree.cu:317` | tree-walk from node 0 over §13 columnar data (8/16/32 dispatch), numeric threshold + missing/default, cat bitset membership, `score[data_index] += leaf_value[~node]` | `<USE_INDICES>` | `#[cube]` tree-walk; shares route math with §9 mark; `double` score (D-14) |
| `FindInBitsetCUDA<T>` (device helper) | `cuda_tree.cu:39`, `cuda_data_partition.cu:76` | `bits[pos/32] >> (pos%32) & 1` | `<T>` | tiny `#[cube]` helper, shared by partition + predict cat paths |
| `Set/GetDecisionTypeCUDA`, `…MissingType` | `cuda_tree.cu` (`__device__`) | `int8_t` decision-type bitfield pack/unpack (`kDefaultLeftMask`, `kCategoricalMask`) | — | comptime bit helpers |

## GenDataToLeftBitVector Flag Fan-Out → CubeCL comptime (research target #2)

The reference nests compile-time flags through launcher chains (`LaunchGenDataToLeftBitVectorKernelInner0..4` etc., `cuda_data_partition.cu:~430-445`). Each flag is a **comptime `bool`** parameter on the `#[cube]` (D-02). Semantics (verified against the numeric route in `SplitInnerKernel`/`PrepareOffset` and the predict tree-walk `cuda_tree.cu:361-391`, which share the decision):

| Flag | Meaning | Route effect |
|------|---------|--------------|
| `MIN_IS_MAX` | feature has a single non-default bin (`min_bin == max_bin`) | degenerate: everything routes by default-direction / most-freq-bin logic |
| `MISSING_IS_ZERO` | `MissingType == Zero` — the zero bin is the missing sentinel | rows with `bin == default_bin` take the missing branch |
| `MISSING_IS_NA` | `MissingType == NaN` — the max bin is the NA sentinel | rows with `bin == max_bin` take the missing branch |
| `MFB_IS_ZERO` | most-freq-bin coincides with the zero/default bin | controls whether out-of-[min,max] rows fold to the default or most-freq route |
| `MFB_IS_NA` | most-freq-bin coincides with the NA bin | same, NA side |
| `MAX_TO_LEFT` | the max bin routes left | direction of the boundary/default assignment |
| `USE_MIN_BIN` | remap `bin − min_bin + offset` vs raw | must match the §13 columnar remap used by predict |
| `BIN_TYPE` | column storage width (u8/u16/u32) | the 8/16/32 dispatch (D-05); a comptime *type*, not bool |

**Shared decision (load-bearing, §specifics):** the numeric predict tree-walk (`cuda_tree.cu:376-391`) uses **exactly** the same `(missing_type==1 && bin==default_bin) || (missing_type==2 && bin==max_bin)` → `default_left ? left : right`, else `bin <= threshold ? left : right`. Transcribe this once (a `#[cube]` inline fn taking the comptime flags) and call it from both the partition mark kernel and the predict kernel. `[VERIFIED: cuda_tree.cu:376-391 vs cuda_data_partition.cu route]`

## Prediction Tree-Walk — Verified Transcription (research target, D-05)

Full numeric+categorical walk, verified line-by-line from `cuda_tree.cu:317-396`:

1. `data_index = USE_INDICES ? cuda_used_indices[inner] : inner` (`:342`).
2. `node = 0`; loop `while node >= 0`:
   - read `split_feature_inner`, `column = feature_to_column[…]`, `default_bin`, `most_freq_bin`, `max_bin`, `min_bin`, `offset` (`:345-351`).
   - **8/16/32 dispatch** on `column_bit_type` → read raw `bin` (`:354-360`).
   - **remap:** `if bin ∈ [min_bin, max_bin]: bin = bin − min_bin + offset; else bin = most_freq_bin` (`:361-365`).
   - **categorical** (`decision_type & kCategoricalMask`): `cat_idx = threshold_in_bin[node]`; `FindInBitsetCUDA(bitset_inner + cat_boundaries_inner[cat_idx], cat_boundaries_inner[cat_idx+1] − cat_boundaries_inner[cat_idx], bin)` → left else right (`:367-374`).
   - **numeric**: missing/default as above → threshold compare (`:376-391`).
3. leaf reached (`node < 0`): `score[data_index] += leaf_value[~node]` — **`double` accumulator, D-14** (`:394`).

Objective inverse-link / `ConvertOutput` is applied **host-side at the readback boundary this phase** (Phase-19 moves it on-device).

## HistArena Pool-Swap Integration (research target #3, D-09)

`SplitTreeStructureKernel` (`cuda_data_partition.cu:827-906`) does a **leaf-indexed whole-pool swap** driven by `num_data[left] < num_data[right]`:

- **Left is smaller** (`:827`): `cuda_hist_pool[right] = parent_ptr (was left's)`; `cuda_hist_pool[left] = cuda_hist + 2*right*num_total_bin` (a fresh slot). Smaller child (left) builds into the fresh slot; larger child (right) inherits the parent histogram for the `parent − smaller` subtraction. `smaller_leaf_splits->hist_in_leaf = pool[left]`, `larger->hist_in_leaf = pool[right]` (`:829-833`).
- **Right is smaller** (`:867`): mirror — `pool[right] = cuda_hist + 2*right*num_total_bin` fresh for the smaller, larger keeps parent (`:896-900`).

The existing `HistArena` (`histogram_arena.rs`) already ships the per-split `{parent, smaller, larger}` role rotation (`rotate()`, `parent_idx/smaller_idx/larger_idx`), and its module docs state the **whole-tree pool SWAP is explicitly Phase 18** (`:29-30`). Integration: extend `HistArena` with a **leaf-index → slot-handle table** and a `swap(left_leaf, right_leaf, smaller_is_left)` that reassigns handles exactly as above — do NOT rebuild the rotation. This is a **correctness** requirement (§17: building the larger child directly takes a different rounding path than `parent − smaller`). `[VERIFIED: histogram_arena.rs:16-30, 175-205]`

## Resident Dataset + Prefix-Sum Primitives (research target #4)

- **§13 columnar dataset:** `column_data.rs` / `row_data.rs` already expose the 8/16/32-width resident store both the predict and partition-route kernels read directly (Phase-15/16 hoist precedent — no per-tree re-upload). Confirmed present. `[VERIFIED: crates/lgbm-compute/src/kernels/{column_data,row_data}.rs]`
- **Block prefix-sum:** `primitives.rs` ships the 3-launch global scan (`block_scan → block-totals scan → add-back`) with a single-tile cap of 1024 blocks, generic body `block_scan_body::<N: Numeric>`, but **only `f32`/`f64` launchers are instantiated** (`prefix_sum_inclusive_f64_on`, `…_f32_on`, `:245-351`). **GAP:** §9 uses `ShufflePrefixSum<uint16_t>` (per-block, `PrepareOffset`) and `ShufflePrefixSum<uint32_t>` (`AggregateBlockOffset`). The planner must add **u16 and u32 launcher instantiations** of the existing generic bodies (the `#[cube]` bodies are already `N: Numeric` so this is instantiation, not a rewrite). The exclusive/inclusive distinction matters (§17): `PrepareOffset` uses inclusive + `[idx−1]` for exclusive rank; `AggregateBlockOffset` uses the exclusive block-totals scan. `[VERIFIED: primitives.rs:7-17, 83-135, 245-351]`
- The `>1024`-block recursive global scan is a Phase-15 concern already owned by the dataset; for typical leaf sizes `grid_dim ≤ 1024` (Kernel1 path) dominates.

## Per-Split Readback Packets (research target #5)

Two device→host transfers per split (D-08) — expected, not a regression:

**16-int `cuda_split_info_buffer` (§9, `SplitTreeStructureKernel:799-825`)** — 8 ints + 4 f64 packed into the upper 8 ints (`reinterpret_cast<double*>(buffer + 8)`):

| Index | Field |
|-------|-------|
| `[0]` | `left_leaf_index` |
| `[1]` | `leaf_num_data[left]` |
| `[2]` | `leaf_data_start[left]` |
| `[3]` | `right_leaf_index` |
| `[4]` | `leaf_num_data[right]` |
| `[5]` | `leaf_data_start[right]` |
| `[6]` | smaller-child leaf index (branch-dependent) |
| `[7]` | larger-child leaf index (branch-dependent) |
| `[8..16]` as `double[4]` | `[0]=left_sum_hessians, [1]=right_sum_hessians, [2]=left_sum_gradients, [3]=right_sum_gradients` |

Total 8·4 + 4·8 = 64 bytes = "16-int". `[VERIFIED: cuda_data_partition.cu:799-825]`

**8-int best-split "which split" buffer (Phase-17, §8 export)** — carried in from Phase 17 (`best_split_parity.rs` `SEXPORT <8 ints>`); reconcile with the 16-int packet host-side. `[VERIFIED: crates/oracle-harness/tests/best_split_parity.rs:31]`

`SplitScalars` (`split_info.rs`) is the `CUDASplitInfo` source both `SplitKernel` (via `left_value`/`right_value`/`default_left`/per-side sums, NaN→0) and `SplitTreeStructure` read; pre-allocated once (D-15, `NUM_FIELD_BUFFERS = 21`). `[VERIFIED: split_info.rs:65-136]`

## Split-Before-Partition Ordering (research target #6)

Confirmed reference host orchestration: `CUDATree.Split` runs BEFORE `DataPartition.Split`. `SplitKernel` (`cuda_tree.cu:48`) mutates the flat device tree and the host `LaunchSplitKernel` returns the `right_leaf_index` (new node index) that the partition step then consumes as `right_leaf_index` (`cuda_data_partition.cu` `SplitInnerKernel`/`AggregateBlockOffset` params). A partition running before the tree mutation would consume a stale leaf id. **Hard invariant — do not reorder** (§1/§10, §specifics). `[VERIFIED: cuda_tree.cu:48-138, cuda_data_partition.cu:909]`

## Fixture Strategy (research target #7, D-11)

Existing harness to **reuse and extend** (verified present):

- **C++ capture:** `xtask/cpp/kernel_capture.cpp` (1198 lines) already emits kernel goldens (histogram, subtract, split, and a **Phase-4 `partition.txt`**). The Phase-4 `partition.txt` is `MissingType::None` + host stable-gather only (`crates/oracle-harness/tests/fixtures/kernels/partition.txt`, format `PCASE/PBINS/PORDER/PSPLIT`). Phase 18 **extends this same file** for: (a) the full flag fan-out (missing/NA/default_left/MFB cases), (b) categorical membership routing, (c) the 16-int child-stats packet, and (d) tree-walk predict over numeric AND categorical models. `[VERIFIED: xtask/cpp/kernel_capture.cpp:5, partition.txt:1-21]`
- **Golden replay harness:** `crates/oracle-harness/tests/best_split_parity.rs` is the canonical Phase-17 pattern to mirror — `CARGO_MANIFEST_DIR` fixture path (never the untracked `LightGBM/` tree), raw-f64-bits parsing (zero rounding), graceful SKIP when the fixture is absent, `#[ignore = "Wave-0 scaffold"]` until the numeric core lands, then un-ignore. Use `oracle_harness::comparator::{compare_exact_f64_bits, …}`. `[VERIFIED: best_split_parity.rs:1-55]`
- **Real `lib_lightgbm` 4.6** builds in-tree (MEMORY `lightgbm-ref-tree-untracked`) — the new goldens can be captured against the real library, cross-checking the cpu f64 anchor (D-11).
- **Categorical fixture models already exist:** `tests/fixtures/categorical/cat_onehot.*`, `cat_manyvsmany.*` — reuse for the cat membership predict + partition anchors (D-03/D-05). Reuse Phase-15 synthetic sparse/large-bin columns + Phase-16 hist fixtures where they fit. `[VERIFIED: tests/fixtures/categorical/]`
- Capture-script precedent: `xtask/py/*_oracle_capture.py` (model/categorical/learner captures) + `xtask/src/main.rs` driver.

## Architecture Patterns

### System Architecture Diagram (per-split device flow, `LGBM_CUDA_ON_DEVICE=1`)

```
Phase-17 best split (8-int) ──► SplitScalars/CUDASplitInfo (split_info.rs, prealloc D-15)
        │
        ▼   [ORDERING INVARIANT §1/§10: tree BEFORE partition]
   SplitKernel (tree.rs) ── mutates flat CUDATree ──► returns right_leaf_index
        │
        ▼
   GenDataToLeftBitVector<flags,BIN_TYPE>  (data_partition.rs)   [+ _Categorical]
        │   per-row mark → PrepareOffset block ShufflePrefixSum<u16>
        ▼
   AggregateBlockOffsetKernel{0|1}  ── exclusive scan of per-block counts (u32)
        │   left→[0,to_left_total)  right→[to_left_total, n)
        ▼
   SplitInnerKernel ── scatter into cuda_out_data_indices_in_leaf_ (scratch, prealloc)
        │
        ▼
   CopyDataIndicesKernel ── write reordered slice back  (== stable partition, D-04)
        │
        ├─► UpdateDataIndexToLeafIndex (stream 3, concurrent) ──► row→leaf map
        │
        ▼
   SplitTreeStructureKernel ── child leaf outputs
        │   ├─ pack 16-int packet ──────────► device→host readback (D-08)
        │   └─ HistArena leaf-indexed pool SWAP (D-09, subtraction-trick reuse)
        ▼
   AddPredictionToScoreKernel<USE_BAGGING> (§9) ── double* score += leaf output

   [PREDICT PATH, D-05]  AddPredictionToScoreKernel<USE_INDICES> (predict.rs)
     tree-walk node 0 → §13 columnar (8/16/32) → numeric/cat route → double* score
     → host objective inverse-link at readback (Phase-19 boundary)
```

### Recommended Project Structure (Claude's Discretion — a suggestion)
```
crates/lgbm-compute/src/kernels/
├── data_partition.rs   # NEW: mark→prefix-sum→scatter (§9), + _Categorical
├── tree.rs             # NEW: SplitKernel / SplitCategorical / Shrinkage / AddBias (§10)
├── predict.rs          # NEW: AddPredictionToScore tree-walk (§10) + §9 leaf-map add
├── primitives.rs       # EXTEND: u16 + u32 prefix-sum launcher instantiations
├── histogram_arena.rs  # EXTEND: leaf-indexed whole-pool swap (D-09)
├── split_info.rs       # REUSE: SplitScalars = CUDASplitInfo
└── partition.rs        # REFERENCE ONLY (host-gather, MissingType::None)
```

### Anti-Patterns to Avoid
- **Sorting to partition** — §9 is strictly `mark → prefix-sum → scatter` (§17). Never sort.
- **Comparing two GPU f32 paths** — def-f8u-01. Anchor to the cpu f64 fold only.
- **f64 per-row hot loops** — D-14 / spike-052 (5.4× consumer-NVIDIA regression). f64 only in the `double* score` accumulator + scalar gain/output math.
- **Per-split device alloc** — D-15. Pre-allocate scatter scratch, block-offset buffers, 16-int packet ONCE outside the hot loop.
- **Rebuilding the histogram rotation** — D-09; extend the Phase-16 arena.
- **Building the larger child histogram directly** — must be `parent − smaller` (correctness, §17).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Block/global prefix-sum | A new scan | `primitives.rs` scan bodies (add u16/u32 launchers) | Already bit-exact vs serial Rust scan; 3-launch global structure handled |
| Split record | A new struct | `split_info.rs::SplitScalars` | Exact `CUDASplitInfo` field list incl. cat slabs; prealloc D-15 |
| Histogram pool + subtraction reuse | A new pool | `HistArena` handle rotation | Phase-16 shipped; whole-pool swap is the *only* Phase-18 addition |
| §13 columnar read (8/16/32) | Re-upload / new store | `column_data.rs` / `row_data.rs` | Resident, Phase-15/16 hoist; predict + partition read directly |
| Route decision (missing/default) | Two copies | One shared `#[cube]` inline fn (partition + predict) | §specifics: same default-direction logic; one transcription |
| Bitset membership | A new impl | Port `FindInBitsetCUDA<T>` once | Shared by partition-cat + predict-cat |

**Key insight:** this phase is ~80% *wiring proven primitives in the reference's exact order*, not new algorithm design. The reference source is in-tree and authoritative; every routing/packing rule was transcribed line-by-line. The genuinely new code is: the scatter kernel, the leaf-indexed pool swap, the comptime flag fan-out, and the u16/u32 scan instantiations.

## Common Pitfalls

### Pitfall 1: Non-stable / block-reordered scatter
**What goes wrong:** implementing the scatter with atomics or a non-order-preserving rank breaks the row permutation, changing per-leaf f32 accumulation order and drifting the *next* histogram off the anchor.
**Why:** §17 — row order = f32 accumulation order. D-04 confirmed the reference is a plain stable partition; the impl must preserve original relative order within AND across blocks.
**Avoid:** cpu anchor = exact `stable_partition`; hip kernel = inclusive block scan + exclusive block-base offsets exactly as `PrepareOffset`/`AggregateBlockOffset`/`SplitInner`.
**Warning sign:** partition golden `PORDER` mismatch, or next-iteration histogram diverges while this-iteration split matches.

### Pitfall 2: Inclusive vs exclusive scan confusion
**What goes wrong:** using an exclusive scan where §9 uses inclusive `[idx−1]`, off-by-one in the scatter destination.
**Why:** `ShufflePrefixSum` is inclusive; `SplitInner` derives the exclusive rank as `block_to_left_offset_ptr[threadIdx_x−1]`.
**Avoid:** mirror the reference exactly; add both u16 inclusive (PrepareOffset) and u32 exclusive (AggregateBlockOffset) instantiations.

### Pitfall 3: Reordering tree-mutation and partition
**What goes wrong:** partition consumes a stale `right_leaf_index`.
**Why:** §1/§10 — `CUDATree.Split` returns the new leaf id the partition needs.
**Avoid:** hard-sequence SplitKernel → partition; assert the returned `right_leaf_index` feeds the partition launch.

### Pitfall 4: Missing/default route divergence between predict and partition
**What goes wrong:** predict and partition disagree on where NA/default/out-of-range rows go.
**Why:** the two share the same decision math (`cuda_tree.cu:376-391`), but if transcribed twice they drift.
**Avoid:** one shared comptime `#[cube]` route fn (D-02/D-05, §specifics).

### Pitfall 5: Pool swap wrong side
**What goes wrong:** the larger child gets a fresh (empty) histogram instead of inheriting the parent for subtraction.
**Why:** `SplitTreeStructure` swaps by `num_data[left] < num_data[right]`; smaller builds fresh, larger inherits parent.
**Avoid:** transcribe the branch at `cuda_data_partition.cu:827` exactly into the `HistArena` leaf-indexed swap.

## Code Examples

### Reference scatter (the D-04 order-equivalence core)
```cpp
// Source: LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_data_partition.cu:926-933
const uint32_t thread_to_left_offset = (threadIdx_x == 0 ? 0 : block_to_left_offset_ptr[threadIdx_x - 1]);
const bool to_left = block_to_left_offset_ptr[threadIdx_x] > thread_to_left_offset;
if (to_left) {
  left_out_data_indices_in_leaf[thread_to_left_offset] = cuda_data_indices_in_leaf[global_thread_index];
} else {
  const uint32_t thread_to_right_offset = threadIdx.x - thread_to_left_offset;
  right_out_data_indices_in_leaf[thread_to_right_offset] = cuda_data_indices_in_leaf[global_thread_index];
}
```

### Reference predict route (share with partition mark)
```cpp
// Source: LightGBM-release-4.6.0.99/src/io/cuda/cuda_tree.cu:376-391
const uint32_t threshold_in_bin = cuda_threshold_in_bin[node];
const int8_t missing_type = GetMissingTypeCUDA(decision_type);
const bool default_left = ((decision_type & kDefaultLeftMask) > 0);
if ((missing_type == 1 && bin == default_bin) || (missing_type == 2 && bin == max_bin)) {
  node = default_left ? cuda_left_child[node] : cuda_right_child[node];
} else {
  node = (bin <= threshold_in_bin) ? cuda_left_child[node] : cuda_right_child[node];
}
```

### cpu f64 anchor (D-04 CONFIRMED — plain stable partition)
```rust
// Anchor: exact stable partition over the leaf's index slice.
// left-keepers first (original order), then right-keepers (original order).
// Order-equivalent to the §9 block-tiled scatter (proven above).
let (mut left, mut right) = (Vec::new(), Vec::new());
for &idx in &data_indices[leaf_start..leaf_end] {
    if route_left(idx) { left.push(idx) } else { right.push(idx) }
}
let to_left_total = left.len();
data_indices[leaf_start..leaf_end].copy_from_slice(&[left, right].concat());
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Host partition round-trip (shipped ROCm path) | On-device mark→prefix-sum→scatter | Phase 18 (env-gated) | Eliminates host round-trip; APU perf is *worse* (spike-035) but parity-correct; discrete-CUDA payoff is Phase-23 |
| Phase-4 `partition.txt` (MissingType::None, host gather) | Full flag fan-out + categorical + 16-int packet goldens | Phase 18 | Stronger parity gate (D-11) |
| Per-split `{parent,smaller,larger}` role rotation | Leaf-indexed whole-tree pool swap | Phase 18 | Enables multi-split tree growth reuse (Phase-16 deferred this) |

**Deprecated/outdated:** none — this is a forward port. The host-gather `partition.rs` stays as the routing-decision reference (not removed).

## Runtime State Inventory

> Additive kernel port — no rename/refactor/migration. Included for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no on-disk datastore keyed by any renamed string. | None |
| Live service config | None — no external service. | None |
| OS-registered state | None. | None |
| Secrets/env vars | `LGBM_CUDA_ON_DEVICE` (read once, OnceLock-cached, `lib.rs:1314`) — a NEW gate this phase; OFF by default. Not a rename. | None (already wired in Phase 14) |
| Build artifacts | New kernel modules compile into `lgbm-compute`; new goldens under `tests/fixtures/kernels/`. Stale build only if module names change. | `cargo build` picks up automatically |

**Nothing found requiring data migration** — verified: the phase adds device kernels + fixtures, mutates no persisted state.

## Validation Architecture

> nyquist_validation enabled + research enabled. Anchor = cubecl-cpu f64 fold (bit-exact merge gate); cubecl-hip f32 = ~1e-6, never GPU-vs-GPU (D-12/def-f8u-01).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (`oracle-harness` integration tests) + golden-fixture replay |
| Config file | none — `[[test]]` targets under `crates/oracle-harness/tests/` |
| Quick run command | `cargo test -p oracle-harness partition_parity --features cpu` (per new test module) |
| Full suite command | `cargo test --workspace` (the ODL-19 hard merge gate — must be green with `LGBM_CUDA_ON_DEVICE` unset) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-13 | Post-scatter row order == reference stable partition (numeric flag fan-out) | unit/golden | `cargo test -p oracle-harness partition_parity` | ❌ Wave 0 (extend `partition.txt` + new `partition_parity.rs`) |
| ODL-13 | Categorical membership routing (partition) | unit/golden | `cargo test -p oracle-harness partition_parity::cat` | ❌ Wave 0 |
| ODL-13 | 16-int child-stats packet fields | unit/golden | `cargo test -p oracle-harness partition_parity::packet` | ❌ Wave 0 |
| ODL-13 | HistArena leaf-indexed pool swap (subtraction reuse) | unit | `cargo test -p lgbm-compute histogram_arena::swap` | ❌ Wave 0 |
| ODL-14 | SplitKernel field writes + Split-before-partition ordering | unit/golden | `cargo test -p oracle-harness tree_mutation_parity` | ❌ Wave 0 (reuse `split.txt` cases) |
| ODL-14 | Shrinkage / AddBias | unit | `cargo test -p lgbm-compute tree::shrinkage` | ❌ Wave 0 |
| ODL-15 | Numeric tree-walk predict (8/16/32) vs f64 anchor + cross-check lib_lightgbm | integration/golden | `cargo test -p oracle-harness predict_parity::on_device` | ⚠️ extend existing `predict_parity.rs` |
| ODL-15 | Categorical membership predict (cat_onehot/cat_manyvsmany) | integration/golden | `cargo test -p oracle-harness predict_parity::cat` | ⚠️ extend |
| ODL-19 | Merge gate green with env unset; grep for no f64 per-row loops | gate | `cargo test --workspace` + grep | ✅ existing gate |
| — | hip f32 within ~1e-6 vs cpu f64 anchor (tie-aware) | integration (hip) | `cargo test -p oracle-harness --features hip kernel_parity_partition` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the relevant `partition_parity` / `tree_mutation_parity` / `predict_parity` module (`cargo test -p oracle-harness <module>`).
- **Per wave merge:** `cargo test --workspace` (env unset).
- **Phase gate:** full suite green + hip parity module (when GPU present) before `/gsd-verify-work`. Scaffold tests land `#[ignore = "Wave-0 scaffold"]` and are un-ignored when the numeric core lands (Phase-17 pattern).

### Wave 0 Gaps
- [ ] `crates/oracle-harness/tests/partition_parity.rs` — covers ODL-13 (row order, cat, packet); mirror `best_split_parity.rs` idioms.
- [ ] `crates/oracle-harness/tests/tree_mutation_parity.rs` — covers ODL-14.
- [ ] Extend `crates/oracle-harness/tests/predict_parity.rs` — on-device tree-walk (ODL-15), numeric + cat.
- [ ] Extend `xtask/cpp/kernel_capture.cpp` — full flag fan-out + categorical routing + 16-int packet + tree-walk predict goldens (D-11); regenerate `tests/fixtures/kernels/partition.txt` (+ a predict fixture).
- [ ] `HistArena` swap unit test in `lgbm-compute`.
- [ ] No new framework install — `cargo test` + golden fixtures already in place.

## Security Domain

> `security_enforcement` enabled, ASVS L1. This is an internal numerical GPU-compute library — no auth/session/network/PII surface.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (bounds) | Device buffer sizing: block-offset/scratch/packet buffers sized from `num_data`/`num_leaves`; `validate_scan_inputs` already guards scan block counts (`primitives.rs:216`). Kernel index guards (`if global_thread_index < num_data_in_leaf`) must be preserved. |
| V6 Cryptography | no | — |

### Known Threat Patterns for {device-compute}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device write in scatter | Tampering | Bounds guard on `global_thread_index` (reference does this at `:925`); size scratch `cuda_out_data_indices_in_leaf_` to full leaf `num_data` once (D-15) |
| Integer overflow in prefix-sum offsets | Tampering/DoS | u16 per-block (≤1024 fits) / u32 global counts match reference widths; assert `num_data` fits `i32` (`data_size_t`) |
| `bitset` OOB in `FindInBitsetCUDA` | Tampering | Reference guards `i1 >= n → false` (`cuda_data_partition.cu:78`); preserve the `n`-bound check |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `primitives.rs` generic scan bodies (`N: Numeric`) instantiate cleanly for `u16`/`u32` on both cubecl-cpu and cubecl-hip | Primitives target #4 | If cubecl `Numeric` excludes u16 or plane intrinsics don't lower for u16 on hip, planner needs a bespoke integer scan (adds a task). MEDIUM — verify at Wave 0 with a tiny u16/u32 scan parity test. |
| A2 | The 8-int Phase-17 best-split packet + 16-int packet reconcile without a third readback | Readback target #5 | If host bookkeeping needs a field neither packet carries, a third transfer would break the "two readbacks" premise (D-08). LOW — field lists cross-checked against reference. |
| A3 | `on_device_growth_supported()` staying `false` means these kernels are exercised only by parity tests this phase (no live grow-loop) | Backend seam | If a test path needs the full grow-loop, scope creeps toward Phase 21. LOW — CONTEXT D-07/§integration confirms per-tree device run for anchor compare only. |
| A4 | Real `lib_lightgbm` 4.6 golden capture for the 16-int packet is feasible via `kernel_capture.cpp` extension (the buffer is an internal device struct) | Fixture target #7 | The packet is populated on-device; capturing it may require an instrumented CUDA build rather than the CPU library. MEDIUM — the cpu f64 fold remains the authoritative anchor; the C++ golden is a cross-check (D-11), so a host-computed equivalent suffices if the device capture is impractical. |

**All other findings are `[VERIFIED]` against in-tree reference source or the existing `lgbm-compute` crate.**

## Open Questions

1. **u16 plane-scan lowering on cubecl-hip (A1)**
   - What we know: f32/f64 plane scans lower to ~1e-6 on cubecl-hip (`primitives.rs:50`); bodies are `N: Numeric`.
   - What's unclear: whether `u16`/`u32` `ShufflePrefixSum` analogs lower on hip 0.10.
   - Recommendation: Wave-0 spike — a 1-block u16 + u32 scan parity test on hip before committing the scatter to the generic body; fall back to a u32-widened scan if u16 doesn't lower (parity-neutral).

2. **16-int packet C++ golden capture path (A4)**
   - What we know: `kernel_capture.cpp` emits device-struct goldens; cpu f64 fold is the authoritative anchor.
   - What's unclear: whether the packet is best captured from an instrumented CUDA build or reconstructed host-side.
   - Recommendation: prefer host-reconstructed golden (compute the same fields from the C++ `Tree`/partition on CPU) — lighter and sufficient as a cross-check; escalate to instrumented capture only if a field can't be reconstructed.

3. **`AggregateBlockOffsetKernel0` vs `1` as one or two kernels (Claude's Discretion)**
   - Recommendation: start with two faithful kernels (matches reference launchers, easiest parity), collapse to one runtime-branched kernel later if desired (parity-neutral).

## Sources

### Primary (HIGH confidence — read line-by-line, in-tree)
- `LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_data_partition.cu` — §9 kernels (`PrepareOffset:52`, `SplitTreeStructure:787`, `SplitInner:909`, `AggregateBlockOffset0/1:681/745`, `CopyDataIndices:937`, categorical `:557/582`)
- `LightGBM-release-4.6.0.99/src/io/cuda/cuda_tree.cu` — §10 (`SplitKernel:48`, `SplitCategorical:160`, `Shrinkage:290`, `AddBias:303`, `AddPredictionToScore:317`, `FindInBitsetCUDA:39`)
- `LightGBM-release-4.6.0.99/include/LightGBM/cuda/cuda_algorithms.hpp` — `ShufflePrefixSum:33` (inclusive) / `…Exclusive:64`
- `docs/cuda-kernel-design.md` §9 (`:870`), §10 (`:937`), §17 (`:1171`) — port considerations
- `crates/lgbm-compute/src/kernels/{primitives.rs, histogram_arena.rs, split_info.rs, column_data.rs, row_data.rs, partition.rs}` + `lib.rs` seam
- `crates/oracle-harness/tests/best_split_parity.rs` + `tests/fixtures/kernels/partition.txt` + `xtask/cpp/kernel_capture.cpp`

### Secondary (MEDIUM confidence)
- MEMORY: `rocm-baseline-amd-fork` (use the AMD fork for GPU kernel parity), `lightgbm-ref-tree-untracked` (real lib_lightgbm 4.6 builds in-tree), `spike-035`/`partition-parallel-null` (partition is memory-bound; APU round-trip is overhead — D-10 accepts this), `def-f8u-01` (never GPU-vs-GPU), spike-052 (f64 per-row 5.4× regression — D-14).

### Tertiary (LOW confidence)
- None — no WebSearch used; this is an internal port with authoritative in-tree reference.

## Metadata

**Confidence breakdown:**
- D-04 order-equivalence gate: HIGH — proven from reference source (inclusive scan + block-ordered offsets = stable partition).
- Kernel inventory / packet layouts / predict walk: HIGH — transcribed line-by-line.
- Reused Rust assets: HIGH — signatures verified in-tree.
- u16/u32 scan lowering on hip: MEDIUM — flagged (A1, Open Q1).
- Fixture capture of the 16-int packet: MEDIUM — flagged (A4, Open Q2).

**Research date:** 2026-07-01
**Valid until:** 2026-08-01 (stable — internal port against a pinned reference tree; ~30 days)
