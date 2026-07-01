# Phase 18: On-Device Data Partition, Tree Mutation & Prediction - Context

**Gathered:** 2026-07-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Port `cuda_data_partition.cu` (§9) and `cuda_tree.cu` (§10) to CubeCL, behind
`LGBM_CUDA_ON_DEVICE`, so **row routing**, **tree mutation**, and **prediction** run
entirely on device — eliminating the host partition round-trip.

**Delivers (ODL-13, ODL-14, ODL-15):**
- **On-device data partition (§9, ODL-13):** `mark → prefix-sum → scatter` row
  permutation (**never sorting**) into two contiguous child ranges
  (`[leaf_data_start, leaf_data_end)`), updates the data-index→leaf-index map, and
  performs the `SplitTreeStructure` **histogram-pool pointer swap** (smaller-vs-larger
  child roles → the subtraction-trick reuse). Resulting row order matches the reference
  so per-leaf f32 accumulation order is identical (§17). Kernels:
  `GenDataToLeftBitVectorKernel` (+ `_Categorical`), `UpdateDataIndexToLeafIndexKernel`
  (+ `_Categorical`), `AggregateBlockOffsetKernel{0,1}`, `SplitInnerKernel`,
  `SplitTreeStructureKernel`, `CopyDataIndicesKernel`, `AddPredictionToScoreKernel`.
- **On-device tree mutation (§10, ODL-14):** `SplitKernel` writes the device flat tree
  arrays and runs **BEFORE** the partition step, returning the `right_leaf_index` the
  partition consumes (the `CUDATree.Split` → `DataPartition.Split` ordering — §1, §10).
  Plus `ShrinkageKernel` (`leaf_value *= rate`) and `AddBiasKernel` — anchor-pinned to
  the host tree structure.
- **On-device prediction (§10, ODL-15):** the tree-walk `AddPredictionToScoreKernel<USE_INDICES>`
  from node 0 over the device columnar dataset (8/16/32 width dispatch): numeric threshold
  + missing/`default_left` handling, **categorical bitset membership** (`FindInBitsetCUDA`),
  leaf-value add to `double* score`; within ~1e-6 + objective inverse-link.

Everything **additive**; CPU / ROCm / existing-host-CUDA paths stay byte-unchanged and the
full merge gate stays green with the env unset. Anchor-pinned to the cpu **f64 fold**
(structure bit-exact; leaf values within ~1e-5 f32 envelope); **never GPU-vs-GPU**
(def-f8u-01). `on_device_growth_supported()` stays **false** this phase.

**Explicitly NOT in this phase:**
- **Categorical split-FINDING / cat-feature training end-to-end** → **Phase 22**. Phase 18
  builds the categorical bitset **membership** routing (predict + partition) fully — it is
  model-consuming and independent of how a cat split was grown — but NOT the code that
  *grows* categorical splits (that is the Phase-17 dispatch seam → Phase 22 core).
- **§11 score-updater constant scalar ops** (`AddScoreConstantKernel` /
  `MultiplyScoreConstantKernel` — init score / shrinkage / DART rescale) and **§12 metrics**
  → **Phase 20**. Note: §9's `AddPredictionToScoreKernel` (the per-row leaf-output add via
  the leaf map) IS built here, since it is part of the partition/grow chain; §11's whole-array
  scalar ops are the Phase-20 boundary.
- **On-device objectives / `ConvertOutput` inverse-link** → **Phase 19** (predict parity here
  uses the existing host objective inverse-link at the readback boundary).
- **Discretized / quantized partition + `RenewDiscretizedTreeLeavesKernel`** → **v2 (QGD)**.
- **End-to-end grow-loop driver integration + default-on rollout** → **Phase 21 / 23**.

</domain>

<decisions>
## Implementation Decisions

### Partition device path & flag fan-out (ODL-13, §9, §17)
- **D-01: Build a NEW §9-faithful on-device `mark → prefix-sum → scatter` partition —
  parallel to the shipped host-gather `data_partition_kernel`, NOT an extension of it.**
  The existing `crates/lgbm-compute/src/kernels/partition.rs` is a per-row `SplitInner`
  route decision (`MissingType::None` only) followed by a **host** two-pass stable gather —
  a different structural path than §9's block-local prefix-sum + `AggregateBlockOffset` +
  in-block-rank `SplitInner` scatter into contiguous child ranges. Mirrors Phase-17 D-01
  (the shipped host transcription is a *reference for the routing decision*, not the device
  anchor). *(Extending the host-gather kernel was rejected: it blurs the §9-faithful anchor
  and the reference block-tiled order.)*
- **D-02: Wire the FULL `GenDataToLeftBitVector` template flag fan-out now**
  — `<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT,
  USE_MIN_BIN, BIN_TYPE>` (CubeCL comptime). Missing/`default_left` handling is core to
  SC #1 partition parity and SC #3 predict, so it cannot cleanly defer. *(Numeric-no-missing-only
  was rejected — predict + partition both need default-direction routing.)*
- **D-03: Build the categorical partition routing FULLY this phase**
  (`GenDataToLeftBitVectorKernel_Categorical` / `UpdateDataIndexToLeafIndexKernel_Categorical`
  via `CUDAFindInBitset(bitset, len, bin−min_bin+mfb_offset)`) — membership only, anchored
  against a fixture model that already contains categorical splits. Categorical split-finding
  stays Phase 22.

### Row-order parity anchor (ODL-13, §9, §17 — SC #1)
- **D-04: The cpu f64 anchor reproduces the reference order via a single-owner STABLE
  partition (left rows in original relative order, then right rows in original relative
  order) — order-equivalent to §9's block-tiled scatter, NOT a faithful block-tiled
  reproduction.** Research MUST first verify order-equivalence: confirm §9's
  `AggregateBlockOffset` + `SplitInner` (left occupies `[0, to_left_total)`, right the rest,
  each side preserving original relative order) yields exactly the stable-partition order.
  Once confirmed, the anchor is a simple exact stable partition; the hip kernel runs the
  real `mark → prefix-sum → scatter`. **If research finds the reference order is NOT a plain
  stable partition, escalate — the faithful block-tiled scatter anchor becomes required**
  (block_dim/grid_dim geometry: `min_num_blocks = num_data≤100 ? 1 : 80`, power-of-two
  `block_dim ≤ 1024`). Row order must match the reference bit-for-bit because it fixes
  per-leaf f32 accumulation order (§17).

### Predict scope (ODL-15, §10 — SC #3)
- **D-05: Build the §10 tree-walk `AddPredictionToScoreKernel<USE_INDICES>` with FULL
  numeric AND categorical membership math this phase.** Numeric: threshold + missing/
  `default_left`, 8/16/32 column-width dispatch (`CUDAColumnData`), remap into the feature
  offset range (else `most_freq_bin`). Categorical: `FindInBitsetCUDA` bitset membership.
  Both anchored to the cpu f64 fold via fixture models. *(This is consistent with the
  ROADMAP note "categorical membership routing is wired here" — it is model-consuming, so it
  can be built + anchored now even though the cat split-FINDER lands in Phase 22.)*
- **D-06: Include §9's `AddPredictionToScoreKernel<USE_BAGGING>` (the per-row leaf-output
  add via the data-index→leaf map) in THIS phase**, as part of the partition/grow chain.
  Phase 20 keeps only the §11 whole-array constant scalar ops
  (`AddScoreConstant` / `MultiplyScoreConstant`).

### Device tree model, Split-before-partition & readback (ODL-14, §10, §9)
- **D-07: Introduce a device-resident flat `CUDATree`** (`cuda_leaf_value_`,
  `cuda_left_child_`, `cuda_decision_type_`, thresholds/counts/depth, categorical
  `cat_boundaries`/bitsets, …). `SplitKernel` mutates it **BEFORE** the partition step and
  returns `right_leaf_index` (the §1/§10 ordering note). The host `lgbm_model::Tree` is
  reconstructed at the end of the (per-tree) device run for the anchor comparison.
- **D-08: Accept TWO small device→host transfers per split** — the Phase-17 **8-int**
  best-split "which split" buffer + this phase's **16-int** `SplitTreeStructure` child-stats
  packet (child num_data, start, sum grad/hess refs). This matches the reference (§8 export
  + §9 `cuda_split_info_buffer_`); it does not break the "no incidental readbacks" premise.
- **D-09: The `SplitTreeStructureKernel` histogram-pool pointer swap integrates with the
  Phase-16 `HistArena` handle rotation** — smaller/larger child roles drive the
  parent-minus-sibling subtraction-trick reuse (a correctness requirement, §17). Reuse the
  Phase-16 arena's explicit `hist_t**`-style handle rotation, do not rebuild it.

### Routing gate (ODL-13, spike-035 — user override)
- **D-10: When `LGBM_CUDA_ON_DEVICE=1`, route the partition on-device unconditionally — NO
  `LGBM_RESIDENT_FORCE`-style size gate this phase.** User-chosen despite spike-035 (the
  round-trip is pure overhead on the shared-memory APU; ROCm host-partition stays the perf
  path there). Rationale accepted: the env seam already keeps the default path byte-unchanged
  (D-13), and real APU-vs-discrete routing tuning is the **Phase-23 perf/rollout DoD**
  (measure on Kaggle CUDA), not a parity concern. *(The size-gated default-off option was
  considered and explicitly declined.)*

### Fixtures (SC #1/#3/#4)
- **D-11: Capture NEW C++ `lib_lightgbm` goldens for this phase** — partition row-order
  (post-scatter index array), tree-walk predict over numeric AND categorical models, and the
  16-int child-stats packet. Heavier than pure host-model reuse but chosen for a stronger
  parity gate. Reuse Phase-15 synthetic sparse/large-bin columns + Phase-16 hist fixtures
  where they fit; the cpu f64 fold remains the bit-exact anchor, the new goldens cross-check
  it.

### Carried forward from Phases 14–17 (NOT re-litigated — hard discipline)
- **D-12:** Anchor every numeric output to the **cubecl-cpu f64 fold**; structure
  bit-exact; ROCm/CUDA f32 within ~1e-6; tie-aware where relevant; **never GPU-vs-GPU**
  (def-f8u-01). One `#[cube]` generic, comptime/runtime-split reduction order.
- **D-13:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA
  paths **byte-unchanged**; full merge gate green and unchanged (ODL-19 — hard merge gate).
- **D-14:** **NO f64 per-row hot loops** in new kernels (5.4× consumer-NVIDIA f64
  regression, spike-052); the `double* score` accumulator and scalar gain/output math stay
  f64 where the reference uses it (§17 — load-bearing).
- **D-15:** **Pre-allocate the scatter scratch + block-offset buffers ONCE outside the hot
  loop** (`cuda_out_data_indices_in_leaf_`, per-block left/right count buffers, the 16-int
  packet) — no per-split in-kernel device alloc (Phase-17 D-11 pattern).

### Claude's Discretion
- Exact CubeCL module placement — likely a new `data_partition.rs` + `tree.rs` (or
  `predict.rs`) in `crates/lgbm-compute/src/kernels/`, reusing `split_info.rs` for the split
  record and the Phase-14 primitives for the block prefix-sum.
- Whether `AggregateBlockOffsetKernel0` (large-grid serial chunk-scan) vs `…1` (one-shot warp
  scan) are two kernels or one runtime-branched kernel — parity-neutral as long as the
  block-winner prefix order is fixed.
- Geometry tunables (all §9 block sizes 1024; `SplitKernel` `<<<3,5>>>`,
  `SplitCategoricalKernel` `<<<3,6>>>`, `SplitTreeStructureKernel` `<<<4,5>>>` scalar
  fan-outs) are occupancy/fan-out knobs with no parity impact — start from the faithful C++
  constants; APU-aware autotune is a deferred perf option.
- The exact device→host reconstruction point for the host `lgbm_model::Tree` (per-split vs
  per-tree) — parity-neutral as long as the final structure matches the anchor.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §9 — **Data Partition** (`cuda_data_partition.cu`):
  `FillDataIndicesBeforeTrainKernel`, `FillDataIndexToLeafIndexKernel`,
  `GenDataToLeftBitVectorKernel<…,BIN_TYPE>` (+ `_Categorical`),
  `UpdateDataIndexToLeafIndexKernel<…>` (+ `_Categorical`),
  `AggregateBlockOffsetKernel{0,1}`, `SplitInnerKernel`, `SplitTreeStructureKernel`
  (16-int packet + pool pointer swap), `CopyDataIndicesKernel`,
  `AddPredictionToScoreKernel<USE_BAGGING>`, and the `CalcBlockDim` geometry +
  host-orchestration stream layout (mark→prefix-sum→scatter, never sort).
- `docs/cuda-kernel-design.md` §10 — **Tree Model I/O** (`cuda_tree.cu`): `CUDATree`
  flat device arrays, `SplitKernel` `<<<3,5>>>`, `SplitCategoricalKernel` `<<<3,6>>>`,
  `ShrinkageKernel`, `AddBiasKernel`, `AddPredictionToScoreKernel<USE_INDICES>` (the
  tree-walk predict), `Set/GetDecisionTypeCUDA`, `FindInBitsetCUDA<T>`.
- `docs/cuda-kernel-design.md` §1 / §16 — the **`CUDATree.Split` BEFORE `DataPartition.Split`**
  ordering note (returns `right_leaf_index`), and End-to-End sequencing.
- `docs/cuda-kernel-design.md` §17 — **Port considerations**: mark→prefix-sum→scatter
  (no sort) + block prefix-sum conventions must match so row/accumulation order is
  identical; subtraction-trick + pool pointer swap = correctness; `double` score
  accumulator load-bearing; template-flag → CubeCL comptime.
- `.planning/REFERENCE_MANIFEST.md` — v1.1 C++ port-source map + CUDA-support boundaries.

### CubeCL API
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` —
  `client.empty` / `empty_tensor` once, reused (D-15 pre-allocation).
- cubecl 0.10 LDS idiom (`SharedMemory::new` / `sync_cube()` / shared atomics) as used in
  `crates/lgbm-compute/src/kernels/primitives.rs` — the block prefix-sum
  (`ShufflePrefixSum` analog) and warp reductions the §9 scatter builds on.

### Prior-phase context (discipline carried forward)
- `.planning/phases/17-on-device-best-split-finder/17-CONTEXT.md` — the 8-int best-split
  readback (D-08 pairs with it), the `DeviceSplitInfo` record, the "new faithful anchor,
  not the shipped host transcription" pattern (D-01), the categorical dispatch seam this
  phase's cat MEMBERSHIP complements.
- `.planning/phases/16-on-device-histogram-constructor/16-CONTEXT.md` — the `HistArena`
  `hist_t**` handle rotation the §9 pool pointer swap integrates with (D-09), the
  subtraction-trick correctness contract.
- `.planning/phases/15-on-device-device-dataset-row-subset-gather/15-CONTEXT.md` —
  `CudaColumnData` / `CudaRowData` §13 layout + 8/16/32 width dispatch the predict kernel
  reads (D-05), synthetic sparse/large-bin fixtures (D-11).
- `.planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-CONTEXT.md`
  — block/global prefix-sum + reduction primitives (the §9 mark→prefix-sum building
  blocks), the anchor/primitive conventions.

### Existing code to extend / reuse (already in git — DO NOT rebuild)
- `crates/lgbm-compute/src/kernels/partition.rs` — the shipped **host-gather**
  `data_partition_kernel` (`SplitInner` `MissingType::None` route + host stable gather).
  **Reference for the per-row routing decision, NOT the §9 device anchor** (D-01 builds the
  new mark→prefix-sum→scatter path).
- `crates/lgbm-compute/src/kernels/split_info.rs` — `SplitScalars` / `DeviceSplitInfo`
  (the `CUDASplitInfo` analog `SplitKernel` reads; `default_left`, per-side sums/count,
  `value`, `num_cat_threshold` + cat slabs).
- `crates/lgbm-compute/src/kernels/histogram_arena.rs` — the Phase-16 `HistArena` handle
  rotation the pool pointer swap reuses (D-09).
- `crates/lgbm-compute/src/kernels/column_data.rs` / `row_data.rs` — the §13 resident
  columnar dataset the predict + partition-routing kernels read (D-05, 8/16/32 width).
- `crates/lgbm-compute/src/kernels/primitives.rs` — block prefix-sum / reductions
  (D-04 scatter, `AggregateBlockOffset`).
- `crates/lgbm-compute/src/lib.rs` — `Backend::grow_tree_on_device` seam +
  `LeafPartitionLayout` payload; `on_device_growth_supported` stays **false** this phase.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`partition.rs` `SplitInner` route decision** — the per-row left/right classification
  (threshold + min/max bin) is already transcribed; the new §9 kernel reuses the *decision*
  and adds the missing/default flag fan-out (D-02) + device prefix-sum/scatter (not the host
  gather).
- **`split_info.rs` `DeviceSplitInfo`** — the split record `SplitKernel` consumes (NaN→0
  weights, per-side sums, `default_left`, cat slabs); pre-allocated once (D-15).
- **`histogram_arena.rs` handle rotation** — the pool pointer swap (D-09) is the same
  smaller-into-fresh / larger-in-place rotation shipped for the subtraction trick.
- **`column_data.rs` / `row_data.rs`** — the §13 resident 8/16/32-width columnar store the
  predict + partition kernels read directly (no re-upload; Phase-15/16 hoist precedent).
- **`primitives.rs` block prefix-sum + reductions** — the `mark → prefix-sum → scatter`
  building blocks and `AggregateBlockOffset` fold.

### Established Patterns
- **New faithful device anchor, not the shipped host transcription** (Phase-17 D-01) — D-01.
- **Anchor to cpu f64 fold, never GPU-vs-GPU** (def-f8u-01) — D-12.
- **Additive, env-gated, byte-unchanged default path** (ODL-19) — D-13.
- **Pre-allocate once outside the hot loop** — D-15.
- **Template-flag → CubeCL comptime** — D-02 full fan-out.
- **Subtraction-trick + pool pointer swap as correctness** (§17) — D-09.

### Integration Points
- New partition + tree + predict kernels live in `lgbm-compute` (`kernels/`), consuming the
  Phase-17 chosen split (8-int) + `DeviceSplitInfo`, mutating the device flat `CUDATree`
  (D-07) BEFORE partition (D-07 ordering), reading the §13 resident dataset (D-05), and
  exporting the 16-int child-stats packet (D-08). Reached only when `LGBM_CUDA_ON_DEVICE=1`.
  **Consumed by Phase 19** (objectives / `ConvertOutput` inverse-link at the predict
  boundary), **Phase 20** (§11 score constant ops + §12 metrics), and **Phase 21**
  (end-to-end grow-loop driver). **Phase 22** fills the categorical split-FINDER (this phase
  already provides the categorical membership routing it will exercise).

</code_context>

<specifics>
## Specific Ideas

- **Split-before-partition is a hard ordering invariant** (§1/§10): `SplitKernel` writes the
  device tree and returns `right_leaf_index` that `DataPartition.Split` consumes. Do not
  reorder — a partition that runs before the tree mutation would consume a stale leaf id.
- **Row order = f32 accumulation order** (§17): the post-scatter index array must match the
  reference bit-for-bit, because the next iteration's per-leaf histogram folds rows in that
  order. D-04's stable-partition anchor is only valid if research confirms order-equivalence.
- **Two readbacks per split are expected, not a regression**: 8-int (which split, Phase 17) +
  16-int (child stats, §9). Anything beyond these two per split breaks the perf premise.
- **Categorical membership ≠ categorical split-finding**: predict + partition membership
  (`FindInBitsetCUDA`) is fully built + anchored here against a fixture cat model; growing a
  cat split is Phase 22. Keep the two cleanly separated so Phase 22 drops in without
  reshaping this phase's kernels.
- **`default_left` / missing routing is load-bearing** in BOTH the partition scatter (D-02)
  and the predict tree walk (D-05) — same default-direction logic, share the transcription.

</specifics>

<deferred>
## Deferred Ideas

- **Categorical split-FINDING / cat-feature end-to-end training** → **Phase 22** (this phase
  builds only the membership routing).
- **§11 score-updater constant scalar ops** (`AddScoreConstant` / `MultiplyScoreConstant`)
  + **§12 metrics** → **Phase 20**.
- **On-device objectives / `ConvertOutput` inverse-link** → **Phase 19**.
- **Discretized/quantized partition + `RenewDiscretizedTreeLeavesKernel`** → **v2 (QGD)**.
- **APU-vs-discrete partition routing tuning + size-gate** (spike-035; user chose
  unconditional default-on when env set, D-10) → **Phase 23 perf/rollout DoD** (measure on
  Kaggle discrete CUDA).
- **APU-aware autotune of §9/§10 geometry** (Phase-13 reuse) → deferred perf option;
  parity-neutral occupancy knobs (Claude's Discretion).

### Reviewed Todos (not folded)
- **establish-large-data-benchmark-fixture** — GPU-perf profiling, belongs to Phase 23 DoD,
  not this parity port (matched on generic keywords only).
- **profile-gpu-training-loop-large-data** — same; Phase 23 perf attribution.
- **spike-gpu-cpu-crossover** — GPU-vs-CPU routing, Phase 23 (relates to D-10 deferral).
- **spike-lowrow-phase-ab** — fixed-overhead localization, Phase 23 perf, not parity.

</deferred>

---

*Phase: 18-on-device-data-partition-tree-mutation-prediction*
*Context gathered: 2026-07-01*
