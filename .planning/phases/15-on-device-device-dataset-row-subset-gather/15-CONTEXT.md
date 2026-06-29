# Phase 15: On-Device Device Dataset + Row-Subset Gather - Context

**Gathered:** 2026-06-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the binned feature matrix and the bagging/GOSS row subset live **resident on
device** in the layout the on-device histogram kernel (Phase 16, §7) is built around,
and port the `CopySubrow` row-subset gather. Everything is **additive and off by
default** behind `LGBM_CUDA_ON_DEVICE`; CPU / ROCm / existing-host-CUDA paths stay
byte-unchanged and the full merge gate stays green.

**Delivers (ODL-03, ODL-04):**
- An on-device **columnar binned dataset** with u8/16/32 bin-width dispatch, dense +
  sparse CSR, carrying the **feature-partition layout** (columns grouped so one
  partition's histogram fits shared memory; an over-budget column becomes its own
  large-bin → global-memory partition). (§3, §13)
- An on-device **row-subset gather** (`CopySubrow` analog) that builds the
  bagging / GOSS subset dataset on device, anchor-pinned to the host subset-selection
  draw sequence. (§3)

**Explicitly NOT in this phase:** the histogram build/fix/subtract (Phase 16), best-split
finding (Phase 17), data partition / tree mutation / prediction *kernels* (Phase 18),
on-device objectives/score/metrics (19–20), the end-to-end driver that grows a tree
on-device (Phase 21). This phase mirrors the v1.0 binning resident in the device layout —
it does not grow anything. `on_device_growth_supported()` stays **false**.

</domain>

<decisions>
## Implementation Decisions

### Representation scope (ODL-03)
- **D-01:** **Build BOTH the §3 column store AND the §13 row + feature-partition store
  this phase**, plus both `CopySubrow` variants — fully matching the C++ dual
  representation up front. Rationale: user chose to materialize the complete C++ dataset
  surface now rather than defer the column store / prediction-side gather to Phase 18.
  The §13 row-wise feature-partition store is the direct Phase-16 histogram input; the §3
  column store (`CUDAColumnData`, column-wise + per-feature meta) is the prediction-side
  store consumed later (Phase 18 tree-walk predict).

### Feature-partition layout (ODL-03)
- **D-02:** **Adopt the C++ §13 shared-hist feature-partition grouping for the on-device
  route; leave the existing ROCm per-feature kernel untouched.** Build
  `DivideCUDAFeatureGroups`: pack columns until `max_num_bin_per_partition =
  shared_hist_size / 2` (grad+hess pair per entry) is hit; a column whose bin count
  exceeds the budget becomes its **own large-bin partition** (`NumLargeBinPartition() > 0`
  → the §7 `_GlobalMemory` path). This is the geometry Phase-16 §7 needs
  (`blockIdx.x` = partition, `threadIdx.x` = column). The shipped one-cube-per-feature +
  autotuned row-partition ROCm kernel stays byte-unchanged on its own host-driven path —
  the two coexist. Phase 16 maps the **reused Phase-11 u64 fixed-point accumulation**
  onto this new partition geometry (that mapping is a Phase-16 research detail, not here).

### Dense + sparse scope (ODL-03)
- **D-03:** **Full dense + sparse CSR parity this phase** (user chose breadth over a
  sparse skeleton). Port the complete dispatch: dense across all three bin widths
  (u8/16/32) + the large-bin/global spill case, AND the sparse `InitSparseData<BIN_TYPE,
  PTR_TYPE>` 3×3 with `GetSparseDataPartitioned` (the partition-local CSR re-lay that
  subtracts `partition_hist_start` to make bins partition-local) across all three
  `row_ptr_type` widths {16,32,64}.
- **D-04 (sparse verification anchor):** Committed corpora are dense
  (`is_enable_sparse` default false), so sparse has no real-corpus parity anchor. Validate
  the device CSR re-lay by **synthesizing in-test sparse columns sized to force each
  `row_ptr_type{16,32,64}`** (nnz crossing the 2^16 / 2^32 thresholds) **plus the
  large-bin spill**, each anchored bit-exact to the **host Rust binned values**. Covers
  the 3×3 matrix deterministically without committing a new corpus fixture. (No new C++
  capture: the v1.0 Rust binning is already C++-bit-exact, so it IS the anchor — same
  reasoning as Phase-14 D-04 for the RNG.)

### Row-subset gather + selection locus (ODL-04)
- **D-05:** **On-device row selection via the Phase-14 `CUDARandom` LCG for plain
  bagging, this phase.** Port the bagging draw on-device, **replicating the host
  `BAGGING_RAND_BLOCK` block-RNG structure bit-for-bit** vs the existing C++-bit-exact
  host bagging order. The device draw sequence must be anchor-pinned to the host
  `bag_data_indices` selection (which rows, in which order) — never GPU-vs-GPU.
- **D-06:** **GOSS *selection* defers to when on-device gradients exist (Phase 19+).**
  GOSS top-|gradient| + random-rest selection needs per-row gradients resident (Phase 19)
  and the Phase-14 percentile/sort skeleton; pulling that forward would cross into
  Phase-19 territory. The **`CopySubrow` gather itself works for ANY index set this
  phase** — GOSS subsets gather correctly from host-supplied indices; only the on-device
  GOSS *draw* is deferred.
- **D-07:** The `CopySubrow` kernel takes `cuda_used_indices` as **input** (one thread
  per selected row, gathers that row across all columns, dispatching per-column on bit
  width, `in[used_indices[local]] → out[local]`), `COPY_SUBROW_BLOCK_SIZE = 1024`,
  `<<<ceil(num_used/1024), 1024>>>`. Produces the compacted binned subset.

### Carried forward from Phase 14 (NOT re-litigated — hard discipline)
- **D-08:** Anchor-pin every numeric output to the **cubecl-cpu f64 fold**; tie-aware
  `default_left`; **never** GPU-vs-GPU (def-f8u-01). Structure bit-exact; f32 leaf/score
  numerics within the ~1e-5 envelope.
- **D-09:** **Pre-allocate once outside the hot loop** (`client.empty` / `empty_tensor`,
  reused/indexed across launches) — the CubeCL pre-allocation pattern; no per-call /
  per-row device alloc. The resident dataset is uploaded once (hoist per the shipped
  `resident_bins_uploaded` once-per-train pattern), not re-uploaded per tree.
- **D-10:** `LGBM_CUDA_ON_DEVICE` OFF by default; CPU / ROCm / existing-host-CUDA paths
  **byte-unchanged**; full merge gate (`raw_bin_train_matches_cpp_golden`,
  `learner_parity`, lgbm/treelearner/compute suites) green and unchanged.

### Claude's Discretion
- Exact CubeCL module placement (likely new `crates/lgbm-compute/src/kernels/` modules for
  the column store, row/partition store, and `CopySubrow`), and whether the device dataset
  structs live in `lgbm-compute` vs a host-layout struct in `lgbm-dataset`.
- The exact `bit_type()` / `row_ptr_bit_type()` runtime dispatch surface (the C++
  `void* const*` column-pointer table + `column_bit_type` array → a CubeCL-idiomatic
  width-dispatched enum / comptime specialization).
- Whether the §13 dense re-lay (`GetDenseDataPartitioned`) and CSR re-lay share helper code
  with the existing host binning extraction.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §3 — Dataset on device (`CUDAColumnData`): column-wise
  binned matrix, `void* const*` column-pointer table + `column_bit_type` width dispatch,
  and `CopySubrowKernel_ColumnData` (launcher `LaunchCopySubrowKernel`,
  `COPY_SUBROW_BLOCK_SIZE=1024`, signature with `cuda_used_indices` IN /
  `out_cuda_data_by_column` OUT).
- `docs/cuda-kernel-design.md` §13 — Device Row Data (`CUDARowData`): the row-wise binned
  matrix + feature-partition layout. `bit_type()∈{8,16,32}`, `row_ptr_bit_type()∈{16,32,64}`,
  the 3×3 `InitSparseData<BIN_TYPE,PTR_TYPE>` dispatch; `DivideCUDAFeatureGroups`,
  `max_num_bin_per_partition = shared_hist_size_/2`, `NumLargeBinPartition()`, the
  accessor table (`cuda_feature_partition_column_index_offsets`, `cuda_column_hist_offsets`,
  `cuda_partition_hist_offsets`, `max_num_column_per_partition`, `num_feature_partitions`,
  `shared_hist_size`), and `GetSparseDataPartitioned` (subtract `partition_hist_start`).
- `docs/cuda-kernel-design.md` §7.1 — confirms the histogram geometry this layout feeds
  (`grid_dim_x = num_feature_partitions`; `blockIdx.x` = partition, `threadIdx.x` = column).
- `docs/cuda-kernel-design.md` §14 — `CUDAColumnData` / `CUDARowData` host-class roles
  (device-buffer ownership, `InitCUDAMemoryFromHostMemory`, pinned host alloc).
- `docs/cuda-kernel-design.md` §17 — port considerations (atomic-ordering / f64 anchor;
  template-flag → comptime mapping).
- `.planning/REFERENCE_MANIFEST.md` — v1.1 C++ port-source map + CUDA-support boundaries.

### CubeCL API
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` —
  host-side pre-allocation (`client.empty` / `empty_tensor` once, reused) — the pattern
  D-09 follows for the resident dataset + subset buffers.

### Existing code to extend / reuse (already in git — DO NOT rebuild)
- `crates/lgbm-compute/src/lib.rs` — `BinColumn` (u8/u16/u32 narrowest-typed column,
  `bin()` / `iter_u32` / `to_u32_vec` / `gather(rows)` accessors); `Backend::
  grow_tree_on_device` (~1272), `on_device_growth_supported` (~1239, stays false).
- `crates/lgbm-compute/src/kernels/random.rs` — Phase-14 `CUDARandom` device LCG
  (`RandInt16`/`RandInt32`/`NextFloat`), the D-05 bagging-draw building block.
- `crates/lgbm-compute/src/kernels/partition.rs` — existing native-width
  `create_from_slice` bin upload + narrow-upload gather precedent (~248/362/373).
- `crates/lgbm-compute/src/kernels/split_info.rs` — the once-in-`new` `client.empty`
  pre-allocation pattern (D-09 reference).
- `crates/lgbm-core/src/random.rs` — host `Random` LCG (C++-bit-exact); bagging draw anchor.
- `crates/lgbm-boosting/src/lib.rs` — host `BaggingSampleStrategy` / `GossSampleStrategy`
  / `BAGGING_RAND_BLOCK` — the host subset draw sequence D-05/D-06 anchor against.
- `crates/lgbm-dataset/src/dataset.rs` — `FeatureGroup` store, `is_sparse` config path,
  `LeafPartitionLayout`; the v1.0 binning this phase mirrors resident.

### Reused shipped kernels
- Phase 11 u64 fixed-point build kernel (reused on the new partition geometry in Phase 16),
  Phase 12 sibling co-pack scan, Phase 13 autotune — `.claude/skills/spike-findings-lightgbm_rs/`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`BinColumn`** (`lgbm-compute/src/lib.rs`): already the narrowest-typed (u8/u16/u32)
  binned column with a `gather(rows)` and width accessors — the host-side seed for the
  device column store + the `CopySubrow` width dispatch (D-01/D-07).
- **Phase-14 `CUDARandom`** (`kernels/random.rs`): the device LCG that the on-device
  bagging draw (D-05) is built on; already bit-exact vs host `Random`.
- **Host bagging/GOSS** (`lgbm-boosting/src/lib.rs`): the C++-bit-exact draw sequence the
  device selection anchors against; GOSS gather reuses it for host-supplied indices (D-06).
- **`client.empty`-once pattern** (`kernels/split_info.rs`) + the `resident_bins_uploaded`
  once-per-train hoist: the lifecycle template for D-09.

### Established Patterns
- **Anchor to cpu f64 fold, never GPU-vs-GPU** (def-f8u-01) — D-08.
- **Additive, env-gated, byte-unchanged default path** — the v1.1 merge-gate discipline (D-10).
- **Host Rust binning IS the parity anchor** (already C++-bit-exact) — no new C++ capture for
  the device dataset; same reasoning as Phase-14 D-04 for RNG (D-04 here).

### Integration Points
- New device dataset structs + `CopySubrow` live in `lgbm-compute` (host-layout build may
  borrow from `lgbm-dataset` `FeatureGroup`). Consumed by Phase 16 (histogram, §13 layout)
  and Phase 18 (prediction, §3 column store). The on-device route is reached only when
  `LGBM_CUDA_ON_DEVICE=1`; the existing per-feature ROCm kernel path is untouched (D-02).

</code_context>

<specifics>
## Specific Ideas

- `max_num_bin_per_partition = shared_hist_size_ / 2` (each entry a grad+hess pair) and the
  over-budget-column → large-bin partition rule are the parity-load-bearing details of
  `DivideCUDAFeatureGroups` — port them literally (D-02).
- `GetSparseDataPartitioned` must subtract `partition_hist_start` so each partition's bins
  are partition-local — the CSR re-lay correctness crux (D-03).
- Sparse verification: synthesize in-test columns whose nnz crosses 2^16 and 2^32 so each
  `row_ptr_type{16,32,64}` buffer is actually exercised, + a column over the shared-hist
  budget for the large-bin spill (D-04).
- The device bagging draw must reproduce the host `BAGGING_RAND_BLOCK` block structure
  exactly — the block-parallel RNG seeding is the parity surface, not just the final set (D-05).

</specifics>

<deferred>
## Deferred Ideas

- **On-device GOSS *selection*** (top-|grad| + random rest) → Phase 19+ when on-device
  gradients + the percentile/sort skeleton are available (D-06). The gather works now.
- **Prediction wiring on the §3 column store** → Phase 18 (tree-walk predict); the store is
  built now (D-01) but has no consumer until then.
- **Quantized/discretized integer dataset path** (§4) → v2 (out of v1.1 scope).
- **Any actual on-device tree growth** → Phase 21.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 15-on-device-device-dataset-row-subset-gather*
*Context gathered: 2026-06-30*
