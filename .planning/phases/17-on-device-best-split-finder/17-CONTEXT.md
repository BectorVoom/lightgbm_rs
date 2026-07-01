# Phase 17: On-Device Best-Split Finder - Context

**Gathered:** 2026-07-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Port `cuda_best_split_finder.cu`'s **three-stage pipeline** to CubeCL, behind
`LGBM_CUDA_ON_DEVICE`, so per-feature split evaluation and the cross-feature/cross-leaf
argmax run entirely on device and return the chosen split with a **single small scalar
readback** (the 8-int buffer) and **tie-aware `default_left`** parity.

**Delivers (ODL-11, ODL-12):**
- **Stage 1 — per-`(leaf,feature)` split evaluation** (one block per task): within-feature
  block prefix-sum → cumulative left/right sums, complement-from-parent-totals, count
  recovery via `cnt_factor = num_data/sum_hessians` + `__double2int_rn`, min-data /
  min-sum-hessian guards, gain math, **forward/reverse default-bin scan**, block argmax
  (`ReduceBestGain` warp→block) → per-task `CUDASplitInfo`. Smaller-leaf task `t` writes
  `[t]`, larger-leaf task `t` writes `[t+num_tasks]`. (§8.1, numerical core.)
- **Stage 2 — cross-feature reduce per leaf** (`SyncBestSplitForLeafKernel` +
  `…AllBlocks` fold + `SetInvalidLeafSplitInfoKernel`). (§8.2)
- **Stage 3 — cross-leaf argmax + export** (`FindBestFromAllSplitsKernel`
  `ReduceBestGainForLeaves` → `PrepareLeafBestSplitInfo`): the chosen
  `(leaf, feature, threshold, default_left)` into the **single 8-int
  `cuda_best_split_info_buffer_`** — the only device→host transfer per iteration. (§8.3)
- **Tie-aware `default_left`** parity to the cpu f64 anchor (**mandatory — do NOT defer**,
  ROADMAP): a flip accepted only on a verified f32 tie (same threshold + left_count +
  f32-equal gains); a flip on any non-tie split hard-fails; empty / sparse-default-bin
  fixtures pass.

Everything **additive and off by default**; CPU / ROCm / existing-host-CUDA paths stay
byte-unchanged and the full merge gate stays green. Anchor-pinned to the cpu **f64 fold**
(structure bit-exact; values within ~1e-5 f32 envelope); **never GPU-vs-GPU** (def-f8u-01).

**Explicitly NOT in this phase:**
- The **categorical inner core** (one-hot + many-cat `BitonicArgSort` by `grad/hess`,
  `cat_threshold` list) → **Phase 22**. Phase 17 wires the `SplitFindTask` categorical
  **dispatch seam** (is_categorical / is_one_hot) but not the categorical eval math.
- Data partition / tree mutation / prediction (§9–10) → **Phase 18**
  (`CUDATree.Split` runs BEFORE `DataPartition.Split`, returning `right_leaf_index`).
- **Discretized / quantized split finder** (`FindBestSplitsDiscretizedForLeafKernel`,
  §8.1 quantized inner) → **v2 (QGD-02)**, skip (ROADMAP explicit).
- `on_device_growth_supported()` stays **false**.

</domain>

<decisions>
## Implementation Decisions

### Anchor + gain math (ODL-11, §8.1, §17)
- **D-01: New CUDA-core-faithful f64 fold is THE bit-exact anchor — NOT the host
  `split.rs` serial scan.** The existing `crates/lgbm-compute/src/kernels/split.rs` is a
  verbatim transcription of the **host** `FindBestThresholdSequentially`
  (`feature_histogram.hpp:830-1057`), which accumulates left-sums **incrementally**. The
  CUDA §8.1 core takes a **different numerical path**: block **ShufflePrefixSum** →
  cumulative sums → **complement-from-parent-totals** → **count recovery via
  `cnt_factor` + `__double2int_rn`**. Build a **single-owner `CubeDim(1)` f64 fold** that
  faithfully mirrors the CUDA accumulation (D-06 pattern: one `#[cube]` generic,
  cpu=single-owner fold, hip=block-parallel). *(Anchoring the host serial scan was rejected:
  it would mask a real CUDA-vs-host accumulation divergence, or force the device kernel to
  deviate from §8.1 to match a host path it doesn't share.)*
- **D-02: Transcribe the CUDA gain device helpers verbatim — do NOT assume host-identity.**
  Rather than reuse `crate::gain` (`get_split_gains` / `get_leaf_gain` /
  `calculate_splitted_leaf_output` / `threshold_l1`) on trust, transcribe the CUDA device
  functions (`CUDALeafSplits::GetSplitGains<USE_L1,USE_SMOOTHING>`,
  `CalculateSplittedLeafOutput`, `GetLeafGainGivenOutput`) faithfully. **Research must diff
  the CUDA gain device functions against `crate::gain` and document every delta** (epsilon
  placement, L1/smoothing branch order, output-value formula). If they prove bit-identical,
  a shared `#[cube]` is fine; the parity-conservative default is a faithful transcription.

### Within-feature scan (ODL-11, §8.1 — ROADMAP research flag)
- **D-03: Purpose-built per-task within-feature prefix-sum — NOT the generic
  `block_scan`.** The stage-1 inner scan must be faithful to the **interleaved
  `[2b]/[2b+1]`** (grad at `2b`, hess at `2b+1`) histogram layout AND the
  **forward/reverse** default-bin scan direction. Write a purpose-built per-task scan
  (may **borrow** the `primitives.rs` LDS / `SharedMemory` / `sync_cube()` idiom, but not
  the generic `block_scan`'s segment contract / output shape). Resolves the ROADMAP
  research flag (plane-sum caps at width 32/64 ≪ 256 bins) via a real block-wide LDS scan.
  *(The shipped `block_scan` primitive covers 256 ≤ 1024, but its generic (non-interleaved,
  single-direction) shape doesn't match the stage-1 inner loop — reuse rejected.)*

### Categorical + global-memory spill scope (ODL-11/12, §8.1)
- **D-04: Numerical stage-1 core only; wire the categorical dispatch seam; categorical
  eval math deferred to Phase 22.** Build the numerical `FindBestSplitsForLeafKernelInner`
  (+ REVERSE). Wire the `SplitFindTask` dispatch (the `is_categorical` / `is_one_hot`
  fields already exist in `split_info.rs`) so Phase 22 drops in
  `FindBestSplitsForLeafKernelCategoricalInner` without reshaping the 3-stage pipeline —
  mirrors Phase 16 shaping the handle contract while deferring the whole-tree swap.
- **D-05: Build the `_GlobalMemory` stage-1 spill variant this phase**, anchored by the
  **Phase-15 synthetic large-bin / global-spill column** (reuse Phase 16 D-04's fixture).
  `FindBestSplitsForLeafKernelInner_GlobalMemory` runs the same gain math over strided
  global loops (`GlobalMemoryPrefixSum`, scratch `feature_hist_{grad,hess,stat}_buffer` +
  `feature_hist_index_buffer`) for blocks with more bins than threads. *(Shared-path-only
  was considered since max_bin default 255 ≤ 256 threads, but the user chose full >256-bin
  coverage now.)* The **discretized** global-memory path is a C++ `TODO` and discretized is
  skipped anyway — not in scope.

### Kernel structure + comptime flag fan-out (ODL-11/12, §8.1–8.3)
- **D-06: Keep the 3 faithful separate stages** (stage1 per-task / stage2
  `SyncBestSplitForLeafKernel` cross-feature reduce / stage3 `FindBestFromAllSplitsKernel`
  cross-leaf argmax + `PrepareLeafBestSplitInfo` 8-int export) with the block-argmax
  reduction family (`ReduceBestGainWarp` → `…Block` → `ReduceBestGain`;
  `ReduceBestSplit`). Preserves the **single 8-int readback contract** (SC#2) and the
  reduction order the anchor pins. *(Fusing on the spoofed 8-CU APU was rejected: APU perf
  is confounded, and fusion muddies the reduction-order parity + the export contract.)*
- **D-07: Full comptime flag fan-out — wire and anchor all four**
  `<USE_RAND, USE_L1, USE_SMOOTHING, IS_LARGER>`. Includes **USE_RAND / extra-trees**
  (Phase-14 `CUDARandom` LCG ready) and **USE_SMOOTHING / `path_smooth`**, beyond
  `split.rs`'s shipped default-template host scope. **Fixture-matrix implication (research):
  requires extra-trees RNG-stream goldens** (verifying the Phase-14 `CUDARandom` bit-
  identical draw sequence) **and `path_smooth` smoothing goldens** in addition to the
  default-template anchor. `IS_LARGER` is core to the smaller/larger task split of the
  3-stage layout; `USE_L1` (`lambda_l1`) is already in the gain math.

### Carried forward from Phases 14–16 (NOT re-litigated — hard discipline)
- **D-08:** Anchor-pin every numeric output to the **cubecl-cpu f64 fold**; structure
  bit-exact; ROCm/CUDA f32 within ~1e-6; tie-aware where relevant; **never GPU-vs-GPU**
  (def-f8u-01). One `#[cube]` generic, comptime/runtime-split reduction order.
- **D-09:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA
  paths **byte-unchanged**; full merge gate green and unchanged (ODL-19 — hard merge gate).
- **D-10:** **NO f64 per-row hot loops** in new kernels (5.4× consumer-NVIDIA f64
  regression, spike-052); f64 permitted only in scalar/gain math where the reference uses
  it (the split-gain / count-recovery math is inherently f64/double in §8.1 — that stays).
- **D-11:** **Pre-allocate the split-record + scratch buffers ONCE outside the hot loop**
  (`split_info.rs` `DeviceSplitInfo::new` `client.empty` pattern; the global-memory scratch
  buffers pre-allocated once per D-05). No per-split in-kernel device alloc.

### Claude's Discretion
- Exact CubeCL module placement — likely a new `best_split.rs` (or extend `split.rs`) in
  `crates/lgbm-compute/src/kernels/`, reusing `split_info.rs` `SplitScalars` /
  `DeviceSplitInfo` for the per-task records and the 8-int export buffer shape.
- Whether stage-2's `…AllBlocks` fold is a separate `<<<1,1>>>`-analog kernel or folded
  into stage-2 when `num_blocks_per_leaf == 1` (the common small case) — parity-neutral as
  long as the block-winner reduction order is fixed.
- Geometry tunables (`NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER=256`,
  `NUM_THREADS_FIND_BEST_LEAF=256`, `NUM_TASKS_PER_SYNC_BLOCK=1024`, the smaller/larger
  stream split) are **occupancy knobs with no parity impact** — start from the faithful C++
  constants; APU-aware autotune is a deferred perf option, not a parity requirement.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §8 (§8.1–8.3) — **Best Split Finder**
  (`cuda_best_split_finder.cu`): the `SplitFindTask` struct, stage-1
  `FindBestSplitsForLeafKernel<USE_RAND,USE_L1,USE_SMOOTHING,IS_LARGER>` + its numerical /
  categorical / `_GlobalMemory` inner cores, the `ReduceBestGain` block-argmax family,
  stage-2 `SyncBestSplitForLeafKernel` (+ `…AllBlocks`, `SetInvalidLeafSplitInfoKernel`),
  stage-3 `FindBestFromAllSplitsKernel` + `PrepareLeafBestSplitInfo` (the 8-int export),
  and the setup kernels (`AllocateCatVectorsKernel`, `InitCUDARandomKernel`).
  **§8.1 discretized inner = v2 (QGD-02) — skip.**
- `docs/cuda-kernel-design.md` §7 — **Histogram Constructor** (Phase 16): the
  interleaved `[2b]/[2b+1]` `hist_in_leaf` layout stage-1 reads, and `hist_offset` /
  `mfb_offset` / `default_bin` the `SplitFindTask` carries.
- `docs/cuda-kernel-design.md` §17 — **Port considerations**: atomic-ordering
  nondeterminism → fixed-order f64 anchor; tie-aware `default_left`; template-flag →
  CubeCL comptime; the single 8-int per-split readback.
- `.planning/REFERENCE_MANIFEST.md` — v1.1 C++ port-source map + CUDA-support boundaries
  (verified against `LightGBM/` C++ source 2026-06-29, quick task 260629-djo).

### CubeCL API
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` —
  `client.empty` / `empty_tensor` once, reused (D-11 pre-allocation).
- cubecl 0.10 LDS idiom (`SharedMemory::new` / `sync_cube()` / shared atomics) as used in
  `crates/lgbm-compute/src/kernels/primitives.rs` — the D-03 purpose-built scan building
  block and the `ReduceBestGain` warp/block reduction.

### Prior-phase context (discipline carried forward)
- `.planning/phases/16-on-device-histogram-constructor/16-CONTEXT.md` — D-01/D-06/D-07/D-09
  (de-quant to `hist_t`; anchor to cpu f64; env-gated byte-unchanged; pre-allocate once) and
  the `hist_in_leaf` interleaved layout + large-bin/global-spill fixtures this phase reuses.
- `.planning/phases/15-on-device-device-dataset-row-subset-gather/15-CONTEXT.md` — §13
  partition layout + synthetic sparse/large-bin fixtures (D-05 anchor).
- `.planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-CONTEXT.md`
  — the `CUDASplitInfo` split-record + `CUDARandom` LCG (D-07 USE_RAND fixture) + the
  anchor/primitive conventions.

### Existing code to extend / reuse (already in git — DO NOT rebuild)
- `crates/lgbm-compute/src/kernels/split.rs` — the **host** `FindBestThresholdSequentially`
  transcription (the default-template CPU scan) + its documented epsilon placements
  (`2·kEpsilon` at scan entry, `kEpsilon`-seeded left/right hessian, subtracted back at
  finalization). **Reference for the gain-math epsilon contract, NOT the device anchor**
  (D-01 builds a separate CUDA-core fold).
- `crates/lgbm-compute/src/kernels/split_info.rs` — `SplitScalars` / `DeviceSplitInfo`
  (SoA `CUDASplitInfo` analog: `is_valid`, `threshold`, `default_left`, per-side
  grad/hess/count, `value`, `num_cat_threshold` + reserved cat slabs) — the per-task
  record + 8-int export buffer, pre-allocated once (D-11).
- `crates/lgbm-compute/src/kernels/primitives.rs` — `block_scan` / `ShufflePrefixSum`-class
  scan, reductions, single-block bitonic argsort (D-03 LDS idiom; the argsort feeds the
  deferred Phase-22 categorical core).
- `crates/lgbm-compute/src/kernels/random.rs` — the Phase-14 `CUDARandom` LCG (D-07
  USE_RAND / extra-trees per-task RNG, `InitCUDARandomKernel` analog).
- `crates/lgbm-compute/src/kernels/histogram*.rs` / `subtract.rs` — Phase-16 `hist_in_leaf`
  producer stage-1 reads.
- `crates/lgbm-compute/src/lib.rs` — `Backend::grow_tree_on_device` seam,
  `on_device_growth_supported` (stays **false** this phase).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`split_info.rs` `DeviceSplitInfo` / `SplitScalars`** — the SoA split-record already
  mirrors the `CUDASplitInfo` field list (incl. `is_valid`, `threshold`, `default_left`,
  per-side sums/count, `value`, cat slabs) and pre-allocates once in `new`; stage-1 writes
  it, stage-3 exports the 8-int subset (D-11).
- **`split.rs` epsilon contract** — the load-bearing `2·kEpsilon` / `kEpsilon`-seed /
  subtract-back placements are already transcribed and documented; the new CUDA-core fold
  reuses the SAME epsilon semantics (D-01/D-02).
- **`primitives.rs` LDS scan + reductions** — D-03 borrows the `SharedMemory`/`sync_cube`
  idiom for the purpose-built per-task scan and the `ReduceBestGain` block-argmax; the
  single-block bitonic argsort is the (deferred) categorical-core building block.
- **`random.rs` `CUDARandom`** — the D-07 USE_RAND extra-trees per-task RNG, already
  bit-stream-verified in Phase 14.
- **Phase-16 `hist_in_leaf` + Phase-15 large-bin fixture** — stage-1's resident input and
  the D-05 global-spill anchor.

### Established Patterns
- **Anchor to cpu f64 fold, never GPU-vs-GPU** (def-f8u-01) — D-08.
- **One `#[cube]` generic, comptime/runtime-split reduction** — cpu = single-owner fold,
  hip = block-parallel with `ReduceBestGain` — D-01/D-08.
- **Additive, env-gated, byte-unchanged default path** (ODL-19) — D-09.
- **Pre-allocate once outside the hot loop** (`split_info.rs` `new`) — D-11.
- **Template-flag → CubeCL comptime** — D-07 fan-out.

### Integration Points
- New best-split kernels live in `lgbm-compute` (`kernels/`), reading the Phase-16
  `hist_in_leaf` histograms + the Phase-15 §13 resident dataset, writing `DeviceSplitInfo`
  records, exporting the single 8-int buffer. Reached only when `LGBM_CUDA_ON_DEVICE=1`.
  **Consumed by Phase 18**: the chosen split (`right_leaf_index` via `CUDATree.Split`)
  drives `DataPartition.Split` (Split BEFORE partition). The categorical dispatch seam
  (D-04) is filled by **Phase 22**.

</code_context>

<specifics>
## Specific Ideas

- **Count recovery is a parity landmine**: `cnt_factor = num_data / sum_hessians` then
  `__double2int_rn` (round-to-nearest-even) — the new f64 fold and the hip kernel must both
  reproduce this exact rounding; a naive truncation diverges. (§8.1)
- **Complement-from-parent, not double-scan**: the CUDA core derives the right side by
  subtracting the scanned-left cumulative sum from the parent leaf totals (not a second
  reverse scan) — the REVERSE flag only changes the default-bin scan direction / recorded
  threshold offset (`t-1+offset` reverse vs `t+offset` forward). (§8.1)
- **Smaller/larger task duality**: task `t` writes `CUDASplitInfo[t]` (smaller) and
  `[t+num_tasks]` (larger); IS_LARGER is comptime, smaller on stream 0 / larger on stream 1.
- **Stage-3 self-invalidation**: `FindBestFromAllSplitsKernel` invalidates the chosen leaf
  (and the freshly created leaf slot) so it isn't re-picked next iteration — behavioral,
  must be preserved.
- **The 8-int buffer IS the only device→host transfer per iteration** — do not add
  incidental readbacks (would break SC#2 and the perf premise).

</specifics>

<deferred>
## Deferred Ideas

- **Categorical inner core** (one-hot singleton-left tests + many-cat `grad/(hess+cat_smooth)`
  `BitonicArgSort_1024` sweep to `max_num_cat`, `cat_threshold[]` list, `cat_l2`) →
  **Phase 22** (D-04 wires only the dispatch seam this phase).
- **Discretized / quantized split finder** (`FindBestSplitsDiscretizedForLeafKernel`,
  int32/int64 packed accumulator, `grad_scale`/`hess_scale`) → **v2 (QGD-02)**.
- **Data partition / tree mutation / prediction** (§9–10, Split-before-partition,
  mark→prefix-sum→scatter, pool pointer-swap, tree-walk predict) → **Phase 18**.
- **APU-aware autotune of the best-split geometry** (Phase-13 reuse) → deferred perf option;
  parity-neutral occupancy knobs (Claude's Discretion).

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 17-on-device-best-split-finder*
*Context gathered: 2026-07-01*
