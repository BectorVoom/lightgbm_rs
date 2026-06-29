# Phase 15: On-Device Device Dataset + Row-Subset Gather - Research

**Researched:** 2026-06-30
**Domain:** CUDA→CubeCL port — on-device columnar/row binned dataset, feature-partition layout, `CopySubrow` row-subset gather, on-device bagging draw
**Confidence:** HIGH (port-source design doc + existing in-tree assets are the anchors; no external research needed)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Build BOTH the §3 column store (`CUDAColumnData`: column-wise + per-feature meta, prediction-side, Phase-18 consumer) AND the §13 row + feature-partition store (`CUDARowData`: the direct Phase-16 histogram input) this phase, plus **both** `CopySubrow` variants — fully matching the C++ dual representation up front.
- **D-02:** Adopt the C++ §13 shared-hist feature-partition grouping for the on-device route; **leave the existing ROCm per-feature kernel untouched** (the two coexist). Build `DivideCUDAFeatureGroups`: pack columns until `max_num_bin_per_partition = shared_hist_size / 2` (grad+hess pair per entry); a column whose bin count exceeds the budget becomes its **own large-bin partition** (`NumLargeBinPartition() > 0` → §7 `_GlobalMemory` path). Geometry: `blockIdx.x` = partition, `threadIdx.x` = column. Phase 16 maps the reused Phase-11 u64 fixed-point accumulation onto this geometry (NOT this phase).
- **D-03:** Full dense + sparse CSR parity this phase. Dense across all three bin widths (u8/16/32) + the large-bin/global spill case, AND the sparse `InitSparseData<BIN_TYPE, PTR_TYPE>` 3×3 with `GetSparseDataPartitioned` (subtracts `partition_hist_start` → partition-local bins) across all three `row_ptr_type` widths {16,32,64}.
- **D-04 (sparse verification anchor):** Committed corpora are dense, so sparse has no real-corpus anchor. Validate the device CSR re-lay by **synthesizing in-test sparse columns sized to force each `row_ptr_type{16,32,64}`** (nnz crossing 2^16 / 2^32) **plus the large-bin spill**, each anchored bit-exact to the **host Rust binned values** (the v1.0 Rust binning IS the C++-bit-exact anchor — same reasoning as Phase-14 D-04 for the RNG). No new C++ capture.
- **D-05:** On-device row selection via the Phase-14 `CUDARandom` LCG for plain bagging this phase, **replicating the host `BAGGING_RAND_BLOCK` block-RNG structure bit-for-bit** vs the existing C++-bit-exact host bagging order. The device draw sequence is anchor-pinned to the host `bag_data_indices` selection (which rows, in which order) — **never GPU-vs-GPU**.
- **D-06:** GOSS *selection* defers to when on-device gradients exist (Phase 19+). The **`CopySubrow` gather itself works for ANY index set this phase** — GOSS subsets gather correctly from host-supplied indices; only the on-device GOSS *draw* is deferred.
- **D-07:** The `CopySubrow` kernel takes `cuda_used_indices` as **input** (one thread per selected row, gathers that row across all columns, dispatching per-column on bit width, `in[used_indices[local]] → out[local]`), `COPY_SUBROW_BLOCK_SIZE = 1024`, `<<<ceil(num_used/1024), 1024>>>`. Produces the compacted binned subset.
- **D-08:** Anchor-pin every numeric output to the **cubecl-cpu f64 fold**; tie-aware `default_left`; **never** GPU-vs-GPU (def-f8u-01). Structure bit-exact; f32 leaf/score numerics within ~1e-5.
- **D-09:** **Pre-allocate once outside the hot loop** (`client.empty` / `empty_tensor`, reused/indexed across launches) — no per-call/per-row device alloc. The resident dataset is uploaded once (hoist per the shipped `resident_bins_uploaded` once-per-train pattern), not re-uploaded per tree.
- **D-10:** `LGBM_CUDA_ON_DEVICE` OFF by default; CPU / ROCm / existing-host-CUDA paths **byte-unchanged**; full merge gate (`raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites) green and unchanged.

### Claude's Discretion
- Exact CubeCL module placement (likely new `crates/lgbm-compute/src/kernels/` modules for the column store, row/partition store, and `CopySubrow`), and whether the device dataset structs live in `lgbm-compute` vs a host-layout struct in `lgbm-dataset`.
- The exact `bit_type()` / `row_ptr_bit_type()` runtime dispatch surface (the C++ `void* const*` column-pointer table + `column_bit_type` array → a CubeCL-idiomatic width-dispatched enum / comptime specialization).
- Whether the §13 dense re-lay (`GetDenseDataPartitioned`) and CSR re-lay share helper code with the existing host binning extraction.

### Deferred Ideas (OUT OF SCOPE)
- On-device GOSS *selection* (top-|grad| + random rest) → Phase 19+ (the gather works now).
- Prediction wiring on the §3 column store → Phase 18 (store built now, no consumer yet).
- Quantized/discretized integer dataset path (§4) → v2.
- Any actual on-device tree growth → Phase 21. `on_device_growth_supported()` stays **false**.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-03 | On-device columnar binned dataset (u8/16/32 dispatch; dense + sparse CSR) resident on device, carrying the feature-partition layout the histogram kernel is built around — features grouped so one partition's histogram fits shared memory, a too-wide column becoming its own large-bin partition (→ global-memory path). (§3, §13) | Standard Stack (existing `BinColumn` + generic-over-`Int` kernel idiom); Architecture Patterns 1–4 (the two stores, `DivideCUDAFeatureGroups` formula, the 3×3 sparse dispatch, `GetSparseDataPartitioned` partition-local CSR); Pitfalls 1–4. |
| ODL-04 | On-device row-subset gather (`CopySubrow` analog) builds the bagging/GOSS subset dataset on device, anchor-pinned to the host subset-selection draw sequence. (§3) | Architecture Pattern 5 (`CopySubrow` per-row width-dispatched gather, D-07 launch geometry); Pattern 6 (on-device bagging draw reusing Phase-14 `draw_next_float_on`, BAGGING_RAND_BLOCK block structure); Pitfalls 5–6. |
</phase_requirements>

## Summary

This is a **pure CUDA→CubeCL port phase** with all ten decisions locked. There is **no external technology research to do** — the port source is `docs/cuda-kernel-design.md` (§3, §7.1, §13, §14, §17), the parity anchor is the existing C++-bit-exact host Rust binning, and every building block already exists in git (`BinColumn`, the generic-over-`Int` kernel idiom in `partition.rs`/`histogram.rs`, the Phase-14 `CUDARandom` draw launchers, the `client.empty`-once pattern, the `resident_bins_uploaded` once-per-train hoist, and the host `BaggingSampleStrategy`). The research job is to extract the **parity-load-bearing geometry/sizing formulas** to port literally and to flag the landmines.

The phase delivers two coexisting on-device dataset representations plus a row-subset gather, all additive behind `LGBM_CUDA_ON_DEVICE` and all anchored to the host binned values (never GPU-vs-GPU). The §13 **row + feature-partition store** is the new, load-bearing artifact: it introduces a layout the existing feature-major resident-bins buffer does **not** have — columns grouped into partitions whose histogram fits `shared_hist_size/2` (grad+hess) entries, with over-budget columns spilling to their own large-bin partition, and per-partition bin offsets relaid so each partition's bins are partition-local. The §3 **column store** is a thinner prediction-side mirror with no consumer until Phase 18. The `CopySubrow` gather is a per-row, per-column-width-dispatched compaction whose index set this phase always takes from the host (the on-device bagging *draw* is the only selection ported now, anchored bit-for-bit to the host `BAGGING_RAND_BLOCK` per-block RNG stream).

**Primary recommendation:** Port the §13 sizing/offset formulas (`max_num_bin_per_partition = shared_hist_size/2`, the `DivideCUDAFeatureGroups` packing rule, the `partition_hist_start` subtraction) **as host-side `usize` integer arithmetic on the CPU side**, where they have zero parity impact beyond producing the right offset tables; express width dispatch with the proven **generic-over-`cubecl::Int` + `u32::cast_from` index-read idiom** (`partition.rs:46`, `histogram.rs:1090`) rather than a `void*`-table analog; reuse `draw_next_float_on` (`random.rs:240`) verbatim for the bagging draw, feeding it one seed per `BAGGING_RAND_BLOCK` block; and anchor every output to the host with the `cuda_random_parity.rs` test shape (host-vs-device, `to_bits()`/exact-int equality, never GPU-vs-GPU).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Feature→partition grouping, bin-offset tables (`DivideCUDAFeatureGroups`, `cuda_*_hist_offsets`) | Host (CPU, `usize` integer math) | — | Pure integer layout computation; C++ does it host-side in `CUDARowData::Init`. No kernel, no parity surface beyond producing correct offsets. |
| Dense per-partition re-lay (`GetDenseDataPartitioned`) | Host (CPU concat) → device upload | — | Row-major partition-local buffer assembled on host, uploaded once (D-09). |
| Sparse CSR per-partition re-lay (`GetSparseDataPartitioned`, subtract `partition_hist_start`) | Host (CPU) → device upload | — | The `partition_hist_start` subtraction is host integer arithmetic; correctness crux is offset bookkeeping, not a kernel. |
| `bit_type()` / `row_ptr_bit_type()` width dispatch | Device kernel (`#[cube]` generic-over-`Int`) + host launch match | Host | The kernel is monomorphized per width via `cubecl::Int`; the host matches the `BinColumn`/ptr-width enum to pick the monomorph (the `partition.rs` `launch_native!` precedent). |
| `CopySubrow` row gather | Device kernel (one unit per selected row) | Host (index set supply) | One thread per row gathers across columns; D-07. |
| Bagging row-selection draw | Device kernel (Phase-14 `draw_next_float_on`) | Host (anchor: `bag_data_indices`) | Per-block LCG draws on device; the host draw is the bit-for-bit anchor (D-05). |
| Resident-buffer lifecycle (alloc-once, upload-once) | Host (`client.empty` / `resident_bins_uploaded` hoist) | — | D-09; no per-tree re-upload. |

## Standard Stack

This phase adds **no external dependencies**. Everything is in-tree.

### Core (existing crates / modules to extend)
| Module | Purpose | Why Standard |
|--------|---------|--------------|
| `crates/lgbm-compute/src/lib.rs` — `BinColumn` (`:52`) | Narrowest-typed (u8/u16/u32) binned column with `gather(rows)` (`:134`), `bin(row)` (`:105`), `to_u32_vec` (`:145`), `len`/`is_empty` | The host-side seed for the device column store + the `CopySubrow` width dispatch (D-01/D-07). Already the project's width-narrowing primitive. |
| `crates/lgbm-compute/src/kernels/partition.rs` — `data_partition_kernel<B: Int>` (`:46`), `launch_native!` macro (`:371`) | The generic-over-`Int` `#[cube]` kernel + host width-match launch idiom (`u32::cast_from(bins[i])`) | The exact CubeCL-idiomatic answer to the C++ `void* const*` + `column_bit_type` dispatch (Claude's Discretion). Reuse the pattern verbatim. |
| `crates/lgbm-compute/src/kernels/histogram.rs` — `construct_leaf_hist_resident_kernel<B: Int>` (`:1090`), `upload_resident_columns` (`:2569`) | The `<B: Int>` resident-bins read precedent (qix) + column-upload helper | Confirms generic-over-`Int` width dispatch is the established project idiom (`u32::cast_from(resident_bins[...])`). |
| `crates/lgbm-compute/src/kernels/random.rs` — `draw_next_float_on` (`:240`), `draw_rand_int16_on` (`:165`), `cuda_next_float` (`:77`) | Phase-14 device LCG draw launchers, bit-identical to host `Random` | The D-05 bagging-draw building block — reuse directly (n=block-count seeds, k=block size). |
| `crates/lgbm-compute/src/kernels/split_info.rs` — `DeviceSplitInfo::new` (`:267`), the single `client.empty` site (`:292`) | The "allocate once in `new`, zero alloc in the hot path" SoA template | The D-09 lifecycle reference for the resident dataset + subset buffers. |
| `crates/lgbm-compute/src/lib.rs` — `GpuBackend::upload_resident_bins` (`:2364`), `resident_bin_width` (`:2056`), `ResidentBinWidth` (`:2046`), `ResidentBins` cache (`:2116`) | The once-per-train upload + interior-mutability `RefCell<Option<…>>` cache, narrowest-uniform-width concat | The D-09 once-per-train hoist template. **NOTE the layout difference (see Pitfall 3).** |
| `crates/lgbm-boosting/src/sample_strategy.rs` — `BaggingSampleStrategy` (`:100`), `bagging()` (`:239`), `BAGGING_RAND_BLOCK = 1024` (`:46`), `reset_sample_config` seeding (`:160`), `bag_data_indices()` (`:376`) | The host C++-bit-exact bagging draw the device anchors against (D-05) | The bit-for-bit parity oracle. The per-block RNG (`Random::new(bagging_seed + block_idx)`, advanced continuously) is the parity surface. |
| `crates/lgbm-dataset/src/dataset.rs` — `Dataset` (`:32`), `is_sparse`/`is_enable_sparse` (`:201`), `FeatureGroup`, `LeafPartitionLayout` (`:88`) | The v1.0 host binning this phase mirrors resident + the `is_sparse` config gate | Source of host binned values (the D-04 anchor) and the sparse-config switch. |

### Supporting
| Module | Purpose | When to Use |
|--------|---------|-------------|
| `cubecl` `client.empty(bytes)` / `empty_tensor(shape, elem)` | Host-side pre-allocation outside the hot loop (manual §13) | D-09 — every resident buffer + subset output buffer. |
| `cubecl::prelude::CubeElement::as_bytes` + `client.create_from_slice` | Native-width host→device upload (the `partition.rs:373` / `lib.rs:2394` precedent) | Uploading each width's column/row buffer without a u32 widen. |
| `client.read_one_unchecked(handle)` + `<T>::from_bytes` | Readback for the host-vs-device parity asserts | Verification only — not in any hot path. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Generic-over-`Int` `#[cube]` + host width-match | A `void*`-table / `Array<u8>` byte-buffer with in-kernel width branch | Rejected: defeats CubeCL monomorphization, reintroduces per-element width branches the project explicitly removed (spike-004); the `partition.rs`/`histogram.rs` precedent is generic-over-`Int`. Document this in the dispatch section. |
| Re-using the feature-major `resident_bins` buffer for §13 | A new row-major partition-local buffer | Rejected: §13 is row-wise partition-local (`data[idx·ncol+tx]`), the existing buffer is feature-major (`f*num_data+row`). Different layout — must be a **new** buffer (D-02: the two coexist). See Pitfall 3. |
| One device draw kernel per bagging call | Reuse `draw_next_float_on` with block-seeds | Use the existing launcher — it already threads the LCG single-owner per task; one task per `BAGGING_RAND_BLOCK` block. |

**Installation:** none — no new crates. Confirm the workspace builds with `cargo build -p lgbm-compute` after adding the new kernel modules to `kernels/mod.rs`.

## Package Legitimacy Audit

**Not applicable.** This phase installs **no external packages** — all work extends in-tree `lgbm-compute` / `lgbm-boosting` / `lgbm-dataset` modules and uses the already-vendored `cubecl` dependency. No registry verification needed.

## Architecture Patterns

### System Architecture Diagram

```
                       HOST (CPU, integer layout math — zero parity surface)
                       ┌──────────────────────────────────────────────────────┐
host binned values     │  DivideCUDAFeatureGroups:                             │
(Dataset / BinColumn   │    budget = shared_hist_size/2  (grad+hess per entry) │
 = D-04 anchor)  ──────▶  pack cols until budget hit → partition               │
                       │    col bin-count > budget → own LARGE-BIN partition   │
                       │  build offset tables:                                 │
                       │    feature_partition_column_index_offsets[]           │
                       │    column_hist_offsets[]  (per-col, partition-local)  │
                       │    partition_hist_offsets[] (global partition start)  │
                       │    max_num_column_per_partition, num_feature_partitions│
                       └───────────────┬───────────────────┬───────────────────┘
              GetDenseDataPartitioned   │                   │  GetSparseDataPartitioned
              (row-major partition-local)│                  │  (per-partition CSR;
                                         │                   │   subtract partition_hist_start
                                         ▼                   ▼   → partition-local bins)
                       ┌─────────────────────────────────────────────────────┐
                       │ client.empty / create_from_slice  (UPLOAD ONCE, D-09)│  RefCell<Option<…>> cache
                       └───────────────┬───────────────────┬──────────────────┘  (resident_* hoist)
                                       ▼                    ▼
   ┌────────────────────────────────────────┐  ┌──────────────────────────────────────┐
   │ §13 CUDARowData (row + partition store) │  │ §3 CUDAColumnData (column store)      │
   │  bit_type()∈{8,16,32}                   │  │  per-column u8/16/32 buffers +        │
   │  row_ptr_bit_type()∈{16,32,64} (sparse) │  │  per-feature meta (bit_type, min/max  │
   │  → 3×3 InitSparseData<BIN,PTR> dispatch │  │   bin, offset, most_freq/default bin) │
   │  feeds Phase-16 histogram (§7 geometry: │  │  → Phase-18 prediction consumer       │
   │   blockIdx.x=partition, threadIdx.x=col)│  │   (NO consumer this phase)            │
   └────────────────────────────────────────┘  └──────────────────────────────────────┘
                                       │                    │
                                       ▼                    ▼
                       ┌─────────────────────────────────────────────────────┐
                       │ CopySubrowKernel  (D-07): one unit per selected row,  │
                       │  gathers across cols, width-dispatched per column,    │
                       │  in[used_indices[local]] → out[local]                 │
                       │  COPY_SUBROW_BLOCK_SIZE=1024                           │
                       └───────────────┬─────────────────────────────────────-┘
                                       ▲ used_indices (input)
            ┌──────────────────────────┴──────────────────────────┐
            │ Bagging draw (D-05): reuse Phase-14 draw_next_float_on │
            │  one seed per BAGGING_RAND_BLOCK (1024) block;         │
            │  draw NextFloat per row-in-block; route < fraction     │
            │  ANCHOR: host bag_data_indices (bit-for-bit, host-vs-  │
            │  device, NEVER GPU-vs-GPU)                             │
            └───────────────────────────────────────────────────────┘
   GOSS / arbitrary index set: host-supplied → CopySubrow gathers (D-06)
```

### Recommended Project Structure
```
crates/lgbm-compute/src/kernels/
├── mod.rs                  # register the 2-3 new modules (additive, ungated like random/split)
├── row_data.rs   (new)     # §13 CUDARowData: DivideCUDAFeatureGroups host layout,
│                           #   GetDense/SparseDataPartitioned, bit_type/row_ptr_bit_type
│                           #   dispatch, the 3×3 InitSparseData<BIN,PTR>, offset tables
├── column_data.rs (new)    # §3 CUDAColumnData: per-column buffers + per-feature meta
└── copy_subrow.rs (new)    # CopySubrow #[cube] kernel (generic-over-Int) + host launcher
                            #   + the on-device bagging draw wrapper over draw_next_float_on
```
Decision (Claude's Discretion D-01): keep the device structs in **`lgbm-compute`** (where `BinColumn` and the kernel infra live); the host **layout computation** (`DivideCUDAFeatureGroups`, offset tables) can live as a plain Rust struct in `lgbm-compute` too — it is integer math over `BinColumn` widths, needs no `lgbm-dataset` types, and avoids a dependency-direction question (`lgbm-compute` does not depend on `lgbm-dataset`). The host caller passes in `&[&BinColumn]` + per-feature `num_bin`, exactly as `upload_resident_bins` already does.

### Pattern 1: Width dispatch via generic-over-`cubecl::Int` (the `void*`-table replacement)
**What:** The C++ `void* const* in_cuda_data_by_column` + `uint8_t* column_bit_type` runtime-width table maps to a `#[cube]` kernel generic over `B: Int`, monomorphized on the host by matching the `BinColumn` variant.
**When to use:** Every kernel that reads bin buffers (`CopySubrow`, and the §13 store accessors) and every CSR row-pointer read (generic over the `PTR_TYPE` width {u16,u32,u64}).
**Example (the established precedent — copy this shape):**
```rust
// Source: crates/lgbm-compute/src/kernels/partition.rs:46 (kernel) + :371 (host match)
#[cube(launch)]
pub fn data_partition_kernel<B: Int>(bins: &Array<B>, route: &mut Array<u32>, /* … */) {
    let i = ABSOLUTE_POS;
    if i < bins.len() {
        // u32::cast_from(x: u32) is the identity; <u8>/<u16> widen the index losslessly.
        let bin = u32::cast_from(bins[i]) as i32;
        // … per-row logic …
    }
}
// Host: pick the monomorph by BinColumn width, upload at native width (4× fewer bytes for u8):
macro_rules! launch_native { ($w:ty, $slice:expr) => {{
    let h = client.create_from_slice(<$w>::as_bytes($slice));
    unsafe { data_partition_kernel::launch::<$w, R>(client, count, dim,
        ArrayArg::from_raw_parts(h, n), /* … */); }
}};}
match bins { BinColumn::U8(v) => launch_native!(u8, v),
             BinColumn::U16(v) => launch_native!(u16, v),
             BinColumn::U32(v) => launch_native!(u32, v) }
```
For `CopySubrow` the kernel is generic over the **column** width; for sparse the row-pointer reads are generic over `PTR_TYPE`. The 3×3 `InitSparseData<BIN_TYPE, PTR_TYPE>` becomes a host-side `match (bit_type, row_ptr_bit_type)` selecting the `<B, P>` monomorph (or two-level dispatch). Anything outside {8,16,32}×{16,32,64} returns a `ComputeError` (the C++ `Log::Fatal` analog) — do NOT silently widen.

### Pattern 2: `DivideCUDAFeatureGroups` — the partition-packing formula (port literally)
**What:** Host integer math grouping feature columns into partitions whose histogram fits shared memory.
**The parity-load-bearing rule (§13):**
- `max_num_bin_per_partition = shared_hist_size / 2` — each histogram entry is a `(grad, hess)` pair, hence `/2`.
- `shared_hist_size = gpu_use_dp ? DP_SHARED_HIST_SIZE : SP_SHARED_HIST_SIZE` (`DP_SHARED_HIST_SIZE=6144`, `SP_=2×`; §7.0). For the f64 anchor / `gpu_use_dp` path use the DP value.
- Walk feature groups in order; accumulate each column's bin count; when adding a column would exceed `max_num_bin_per_partition`, close the current partition and start a new one.
- A single column whose own bin count **exceeds** the budget becomes its **own large-bin partition** → increments `NumLargeBinPartition()`; Phase-16 routes those to the `_GlobalMemory` kernel.
**Offset tables to build (the §13 accessor table, port the meaning exactly):**
| Table | Meaning |
|-------|---------|
| `feature_partition_column_index_offsets[i..i+1]` | partition *i* owns columns `[off[i], off[i+1])` |
| `column_hist_offsets[c]` | per-column bin offset **within its partition** (partition-local) |
| `partition_hist_offsets[i]` | global bin offset where partition *i* begins |
| `max_num_column_per_partition` | sizes the histogram kernel's `block_dim_x` / per-row nnz |
| `num_feature_partitions` | = histogram `grid_dim_x` |
**Anti-impact note (§17):** the shared-vs-global spill threshold is a *capacity* choice with **no parity impact** as long as the in-strategy reduction order is fixed. So the budget arithmetic only needs to produce the *correct* offsets — it does not itself move any float. This makes it safe to port as plain host `usize` math and unit-test against hand-computed expectations.

### Pattern 3: `GetSparseDataPartitioned` — partition-local CSR re-lay (the correctness crux)
**What:** For sparse storage, each partition gets its own CSR (`row_ptr` + bin values), and the bin indices are made **partition-local by subtracting `partition_hist_start`** (= `partition_hist_offsets[i]`).
**Why it matters:** the histogram kernel indexes shared memory by the partition-local bin; if the global bin offset is not subtracted, the partition writes out of its SMEM window. This subtraction is the single most error-prone line — port it literally and assert it in tests (synthesize a 2-partition sparse case and check the re-lay'd bins equal `global_bin - partition_hist_start`).
**`row_ptr_type` selection:** the CSR row-pointer width is chosen by max nnz: `≤ 2^16 → u16`, `≤ 2^32 → u32`, else `u64`. This is the `PTR_TYPE` axis of the 3×3 dispatch (D-03). Build the row-pointer buffer at the narrowest fitting width (mirrors `BinColumn`'s width-by-capacity rule).

### Pattern 4: §3 column store vs §13 row store (two coexisting layouts)
**What:** §3 `CUDAColumnData` is **column-major** (per-column buffers + per-feature meta: `bit_type`, `feature_{min,max}_bin`, `offset`, `most_freq_bin`, `default_bin`, missing/mfb flags, `feature_to_column`); §13 `CUDARowData` is **row-major partition-local**. They are different buffers serving different consumers (prediction vs histogram). Build both (D-01) but do not try to share their storage.
**When to use:** §13 feeds Phase-16 (histogram); §3 feeds Phase-18 (prediction). §3 has no consumer this phase — build + parity-test it, do not wire it.

### Pattern 5: `CopySubrow` row-subset gather (D-07)
**What:** One unit per selected row; the unit gathers that row across **all** columns, dispatching per-column on bit width, `in[used_indices[local]] → out[local]`. `COPY_SUBROW_BLOCK_SIZE = 1024`, launch `ceil(num_used/1024)` blocks.
**Example skeleton:**
```rust
#[cube(launch)]
pub fn copy_subrow_kernel<B: Int>(
    in_col: &Array<B>, out_col: &mut Array<B>,
    used_indices: &Array<i32>, num_used: u32,
) {
    let local = ABSOLUTE_POS;
    if local < num_used {
        let src = used_indices[local] as u32; // host-validated in [0, num_data)
        out_col[local] = in_col[src];          // same width in/out (no widen — D-07)
    }
}
```
Drive it once per column, dispatching the monomorph on the column width (Pattern 1). For the §3 column store, loop over columns; for §13 row-major, gather the row's contiguous span. The index set comes from the host (bagging or GOSS) — the kernel is index-set-agnostic (D-06).

### Pattern 6: On-device bagging draw, anchored to host `BAGGING_RAND_BLOCK` (D-05)
**What:** Reproduce the host per-block RNG stream on device, then route each row by `NextFloat() < bagging_fraction`.
**The host parity surface (must mirror exactly — `sample_strategy.rs:239-282`):**
- Blocks of `BAGGING_RAND_BLOCK = 1024` rows; block `b` seeded `Random::new(bagging_seed + b)` (`:160-162`), constructed **once** and advanced continuously across bagging iterations.
- Within a `bagging()` call, row `i` uses block `b = i / 1024`, drawing one `next_float()` in **row order**; the per-block stream advances by exactly the block's row count per call.
- Route: `draw (as f64) < bagging_fraction` → in-bag (filled left, ascending); else OOB (filled right; the OOB tail is then reversed so it reads descending). `bag_data_indices = [in-bag asc] ++ [OOB desc]`.
**Device approach (reuse `draw_next_float_on`, `random.rs:240`):** pass `n_blocks` seeds (the per-block LCG *states* at the current iteration — supply the host block states, or, for a single-iteration anchor test, the initial `bagging_seed + b`), `k = BAGGING_RAND_BLOCK`; the launcher returns row-major `[block0_draw0.., block1_draw0..]` so row `i`'s draw is `out[b*1024 + (i - b*1024)] = out[i]`. Then a route kernel (or host comparison) applies `draw < fraction`.
**Anchor (D-05/D-08):** assert the device draw stream / route decisions against the host `bagging()` `bag_data_indices` **bit-for-bit** (NextFloat via `f32::to_bits()` equality, route via exact set+order) — host-vs-device, never GPU-vs-GPU. See `cuda_random_parity.rs` for the exact assertion shape.

### Anti-Patterns to Avoid
- **A `void*`/byte-buffer width branch inside the kernel.** Use generic-over-`Int` monomorphs (Pattern 1) — the project removed per-element width branches for measured perf (spike-004) and CubeCL has no clean `void*`.
- **Re-uploading the dataset per tree.** Upload once, cache in a `RefCell<Option<…>>` (the `resident_bins` / `resident_bins_uploaded` hoist, `lib.rs:2364`). D-09.
- **Per-row / per-call `client.empty`.** Allocate the resident + subset buffers once in a `new`-style constructor (the `split_info.rs:267` template). D-09.
- **Comparing two GPU paths.** Always anchor to the host f64 fold / host binned values / host `Random` (D-08, def-f8u-01).
- **Forgetting the `partition_hist_start` subtraction** in the sparse re-lay (Pattern 3) — silent SMEM overflow / wrong histogram in Phase 16.
- **Re-using the feature-major `resident_bins` buffer for §13** — wrong layout (Pitfall 3).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-width kernel dispatch | A custom `void*` table + in-kernel byte-width switch | Generic-over-`cubecl::Int` `#[cube]` + host `match BinColumn` (`partition.rs:46`/`371`) | Established project idiom; monomorphizes cleanly; native-width upload is 4× fewer bytes for u8. |
| Device LCG draw stream | A new device RNG | `draw_next_float_on` / `draw_rand_int16_on` (`random.rs:240`/`165`) | Phase-14 shipped, already bit-identical to host `Random` and tested in `cuda_random_parity.rs`. |
| Once-per-train upload + cache | A bespoke upload-tracking flag | The `resident_bins` `RefCell<Option<ResidentBins>>` + `upload_resident_bins`/`wants_resident_bins` hooks (`lib.rs:2116`/`2364`) | Shipped once-per-train hoist; D-09 lifecycle is solved. |
| Narrowest-width column storage | A new u8/u16/u32 storage enum | `BinColumn` (`lib.rs:52`) with `gather`/`bin`/`to_u32_vec` | Already the project's width primitive; `gather(rows)` is literally the host `CopySubrow`. |
| Pre-allocated device buffer lifecycle | Ad-hoc `client.empty` in loops | The `DeviceSplitInfo::new` single-`empty`-site SoA template (`split_info.rs:267`) | Proven D-09 pattern; counts/asserts its allocations. |
| Host bagging order to anchor against | Re-derive the bagging RNG | `BaggingSampleStrategy::bagging()` + `bag_data_indices()` (`sample_strategy.rs:239`/`376`) | C++-bit-exact host oracle already exists. |

**Key insight:** This phase is ~90% **wiring proven primitives into two new layouts + one gather**. The only genuinely new code is the host-side `DivideCUDAFeatureGroups` integer layout + offset tables, the sparse CSR partition-local re-lay, and the `CopySubrow`/bagging-draw kernels — all small, all anchored to existing host truth.

## Common Pitfalls

### Pitfall 1: `max_num_bin_per_partition` off-by-factor (the `/2`)
**What goes wrong:** Sizing the partition budget by total bins instead of `shared_hist_size/2`, or forgetting that each entry is a grad+hess pair.
**Why it happens:** The `/2` is implicit in the C++ (`shared_hist_size_ / 2`); easy to drop.
**How to avoid:** Port `max_num_bin_per_partition = shared_hist_size / 2` literally; unit-test the partition count against a hand-computed example (e.g., `shared_hist_size=6144` → budget 3072 bins/partition).
**Warning signs:** Partition count differs from the C++ for a known feature set; Phase-16 SMEM overflow later.

### Pitfall 2: 3×3 dispatch silently degrading on an unsupported width
**What goes wrong:** A `row_ptr_type` outside {16,32,64} or `bit_type` outside {8,16,32} falls through to a default and corrupts indices.
**Why it happens:** Rust `match` with a catch-all arm.
**How to avoid:** Mirror the C++ `Log::Fatal` — return a `ComputeError` on any width outside the 3×3, never widen silently. Test each of the 9 cells is reachable (D-04 synthesizes columns forcing each `row_ptr_type`).
**Warning signs:** A width that "works" but produces wrong bins; a cell never exercised by tests.

### Pitfall 3: Re-using the feature-major resident buffer for the §13 row store
**What goes wrong:** The existing `resident_bins` buffer is **feature-major** (`f*num_data + row`, `lib.rs:2360`); §13 `CUDARowData` is **row-major partition-local** (`data[idx·ncol+tx]`, §7.2). Reusing the existing buffer gives the histogram kernel the wrong stride.
**Why it happens:** Both are "resident bins"; the names collide.
**How to avoid:** Build a **new** row-major partition-local buffer in `GetDenseDataPartitioned`. The two coexist (D-02). Document the layout in the struct doc-comment.
**Warning signs:** Phase-16 histograms wrong despite "resident bins uploaded"; transposed-looking bin reads.

### Pitfall 4: `partition_hist_start` not subtracted in the sparse re-lay
**What goes wrong:** Per-partition CSR bins stay global instead of partition-local → SMEM index overflow / cross-partition contamination in Phase 16.
**Why it happens:** The subtraction (`GetSparseDataPartitioned` subtracts `partition_hist_start`) is one easily-missed line.
**How to avoid:** Port it literally; assert in a 2-partition synthesized test that re-lay'd bins equal `global_bin - partition_hist_offsets[partition]`.
**Warning signs:** Single-partition tests pass, multi-partition fail.

### Pitfall 5: Bagging RNG block state not the *continuing* stream
**What goes wrong:** Re-seeding each iteration with `bagging_seed + block` instead of advancing the per-block `Random` continuously gives the right draw for iteration 0 but diverges from iteration 1 onward.
**Why it happens:** The host constructs `bagging_rands` **once** (`:160`) and advances them across `bagging()` calls (`:248` "ADVANCE across draws"). A naive device port re-seeds per call.
**How to avoid:** For the single-draw anchor test this phase ships, seed with `bagging_seed + block` (iteration-0 equivalent) AND document that multi-iteration parity requires feeding the host's current per-block states as device seeds. The on-device *bagging loop* is Phase-21 driver scope; this phase proves the single-draw block structure. Keep the seed-supply seam explicit.
**Warning signs:** Iteration-0 bag matches, later iterations drift.

### Pitfall 6: f32/f64 promotion in the bagging route comparison
**What goes wrong:** Host compares `next_float() as f64 < bagging_fraction` (f32 draw promoted to f64, `:255-257`). Comparing in f32 on device can flip a knife-edge row.
**Why it happens:** Device kernel naturally works in f32; the comparison must promote.
**How to avoid:** Promote the device `NextFloat` to f64 before the `< fraction` compare, exactly as the host does. `bagging_fraction` is f64. (NextFloat itself stays f32-exact — the divisor is exactly `32768.0f32`, `random.rs:78`.)
**Warning signs:** A handful of rows near the fraction boundary differ host-vs-device.

## Code Examples

### Reuse the Phase-14 draw launcher for the bagging draw
```rust
// Source: crates/lgbm-compute/src/kernels/random.rs:240 (draw_next_float_on)
// One "task" per BAGGING_RAND_BLOCK; k = block size; out is row-major per task.
let n_blocks = (num_data + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK;
let seeds: Vec<u32> = (0..n_blocks).map(|b| (bagging_seed + b) as u32).collect();
let draws = draw_next_float_on(&client, &seeds, BAGGING_RAND_BLOCK as u32)?; // len = n_blocks*1024
// row i draw == draws[i]  (block b = i/1024, offset i%1024 → b*1024 + i%1024 == i)
// route: draws[i] as f64 < bagging_fraction → in-bag (anchor vs host bag_data_indices)
```

### Native-width upload + monomorph launch (the dispatch idiom to copy)
```rust
// Source: crates/lgbm-compute/src/kernels/partition.rs:371 (launch_native! macro)
match bins {                       // bins: &BinColumn
    BinColumn::U8(v)  => launch_native!(u8,  v),   // uploads count×1 bytes, ::<u8> monomorph
    BinColumn::U16(v) => launch_native!(u16, v),
    BinColumn::U32(v) => launch_native!(u32, v),
}
```

### Allocate-once resident buffer (D-09 lifecycle)
```rust
// Source: crates/lgbm-compute/src/kernels/split_info.rs:267,292 — the ONLY client.empty site, in new()
pub fn new(client: &ComputeClient<R>, total_bytes: usize) -> Self {
    let handle = client.empty(total_bytes);   // once; reused across every launch
    Self { handle, /* … */ }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `void* const*` column table + `column_bit_type` byte array (C++ CUDA) | generic-over-`cubecl::Int` `#[cube]` + host `match BinColumn` monomorph | project idiom (partition.rs/histogram.rs) | No in-kernel width branch; native-width upload. |
| u32-widened bin upload | narrowest-width native upload (`resident_bin_width`, `qix`) | shipped pre-Phase-15 | 4× fewer bytes for u8 data. |
| Per-leaf / per-tree bin re-upload | once-per-train resident hoist (`resident_bins_uploaded`) | shipped (260621-p9v) | D-09 baseline; extend to the §13 store. |

**Deprecated/outdated:** none relevant — this phase is greenfield within an established idiom set.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `DP_SHARED_HIST_SIZE=6144` / `SP_=2×` and `max_num_bin_per_partition = shared_hist_size/2` are the values to port (from §7.0/§13 of the design doc, which summarizes the C++). | Pattern 2 | Wrong partition count vs C++; but Phase-16 (not this phase) is where it bites — and §17 says the spill threshold has no parity impact, so a wrong value still produces a *valid* (just differently-grouped) layout. Verify against `cuda_row_data.hpp` constants if the reference tree is fetched. |
| A2 | The new modules can be registered ungated in `kernels/mod.rs` (like `random`/`split`) rather than `#[cfg(feature="gpu")]`, so cpu-anchor tests run without a GPU. | Recommended Project Structure | If they must be gpu-gated, the cpu f64-anchor parity tests can't run on the default build — but the precedent (`random`/`split` ungated, mod.rs comment) says ungated is correct. |
| A3 | For the bagging draw, supplying `(bagging_seed + block)` as the device seed reproduces the host **iteration-0** draw exactly (the host `Random::new(seed){x = seed as u32}` = the device seed convention, confirmed in `cuda_random_parity.rs`). Multi-iteration continuity needs host block-state supply (Pitfall 5). | Pattern 6 | If wrong, even iteration-0 bag diverges — but `cuda_random_parity.rs` already proves the seed convention, so risk is low. |
| A4 | The §13 store should live in `lgbm-compute` (not `lgbm-dataset`), since `lgbm-compute` does not depend on `lgbm-dataset` and the layout math is over `BinColumn`. | Recommended Project Structure | If the planner prefers a host struct in `lgbm-dataset`, the dependency direction must be checked (`lgbm-compute` → `lgbm-dataset` would be a new edge; avoid). |

**Verification note:** A1's exact constants should be confirmed against `include/LightGBM/cuda/cuda_row_data.hpp` / `cuda_histogram_constructor.hpp` if the read-only `LightGBM/` reference tree is available locally (per MEMORY, `external_libs` CAN be fetched and the AMD fork `LightGBM-release-4.6.0.99/` is the HIP baseline). The design doc is the registered port-source map; treat its numbers as authoritative unless the reference contradicts.

## Open Questions (RESOLVED)

1. **Should the §3 column store carry the full per-feature meta now, or only the buffers Phase-18 will read?** — **RESOLVED:** build the buffers + numeric meta now; defer categorical-bitset meta to Phase-22 (categorical is ODL-22). Parity-test the binned column values + numeric meta; leave a documented TODO for categorical meta. *(Adopted by plan 15-03, which cites "Open Question 1".)*
   - What we know: §14 lists the meta (`bit_type`, `feature_{min,max}_bin`, `offset`, `most_freq_bin`, `default_bin`, missing/mfb flags, `feature_to_column`). D-01 says build it now.
   - What was unclear: whether the missing/categorical meta is needed before Phase-18/22.

2. **Single bagging-draw kernel vs host-side route after `draw_next_float_on`?** — **RESOLVED:** for this phase's *anchor* (proving the block structure), compute the route host-side from the device draw stream and assert vs host `bag_data_indices` — simplest, fully anchored. A fused device route kernel is a Phase-21 driver optimization. *(Adopted by plan 15-04, which cites "Open Question 2 recommendation".)*
   - What we know: `draw_next_float_on` returns the float stream; the route (`< fraction`) is trivial.
   - What was unclear: whether to add a device route kernel or compute the route host-side from the readback.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cubecl` (cpu + hip runtimes) | All kernels + cpu f64 anchor | ✓ (in-tree) | workspace-pinned 0.10 | — |
| `cubecl-cpu` (f64-fold anchor runtime) | The D-08 bit-exact merge gate / parity tests | ✓ | 0.10 | — (this IS the anchor; no fallback needed) |
| ROCm GPU (`cubecl-hip`) | Optional on-device f32 cross-check | ✓ but **spoofed 8-CU APU** (gfx1152, per MEMORY) | — | Parity gates valid on the APU; all *perf* numbers APU-confounded — do NOT draw perf conclusions this phase (it's a dataset/gather phase, not a perf phase). |
| `LightGBM/` C++ reference tree | Confirming A1 constants only | untracked / fetchable | 4.6 | Design doc is the registered authoritative port-source map. |

**Missing dependencies with no fallback:** none — the phase runs entirely on the cpu anchor; GPU is an optional cross-check.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test) — the project standard; no external framework |
| Config file | none — per-crate `tests/*.rs` integration tests + in-module `#[cfg(test)]` unit tests |
| Quick run command | `cargo test -p lgbm-compute --test <new_test_file>` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-03 | Resident §13 row store reproduces host binned values per-column (dense, all 3 bit widths) | unit/parity | `cargo test -p lgbm-compute --test device_dataset_parity dense_bin_parity_all_widths` | ❌ Wave 0 |
| ODL-03 | `DivideCUDAFeatureGroups` partition count + offset tables match hand-computed expectation; large-bin spill increments `NumLargeBinPartition` | unit | `cargo test -p lgbm-compute --test device_dataset_parity feature_partition_layout` | ❌ Wave 0 |
| ODL-03 | Sparse CSR re-lay: each `row_ptr_type{16,32,64}` cell exercised (synthesized nnz crossing 2^16/2^32) + `partition_hist_start` subtracted → partition-local bins | unit/parity | `cargo test -p lgbm-compute --test device_dataset_parity sparse_relay_3x3_and_partition_local` | ❌ Wave 0 |
| ODL-03 | §3 column store binned values + numeric per-feature meta match host | parity | `cargo test -p lgbm-compute --test device_dataset_parity column_store_parity` | ❌ Wave 0 |
| ODL-04 | `CopySubrow` gather: compacted subset == host `BinColumn::gather(used_indices)` across all widths, dense + row-major | parity | `cargo test -p lgbm-compute --test copy_subrow_parity gather_matches_host_all_widths` | ❌ Wave 0 |
| ODL-04 | On-device bagging draw stream + route bit-for-bit vs host `bag_data_indices` (NextFloat `to_bits()`, route set+order); spans ≥2 blocks (>1024 rows) | parity | `cargo test -p lgbm-compute --test copy_subrow_parity bagging_draw_matches_host` | ❌ Wave 0 |
| ODL-04 | `CopySubrow` works for an arbitrary host-supplied index set (GOSS-shaped, D-06) | unit | `cargo test -p lgbm-compute --test copy_subrow_parity gather_arbitrary_indices` | ❌ Wave 0 |
| D-10 | Merge gate unchanged: default-path suites byte-identical with `LGBM_CUDA_ON_DEVICE` unset | regression | `cargo test --workspace` (esp. `raw_bin_train_matches_cpp_golden`, `learner_parity`) | ✅ exists |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute --test <touched_test_file>` (the new device-dataset/copy-subrow files).
- **Per wave merge:** `cargo test -p lgbm-compute` + `cargo test -p lgbm-boosting` (bagging anchor) + `cargo clippy -p lgbm-compute --tests`.
- **Phase gate:** `cargo test --workspace` green (merge gate D-10 unchanged) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/lgbm-compute/tests/device_dataset_parity.rs` — covers ODL-03 (dense all widths, partition layout, sparse 3×3 + partition-local, §3 column store). Mirror the `cuda_random_parity.rs` host-vs-device anchor shape.
- [ ] `crates/lgbm-compute/tests/copy_subrow_parity.rs` — covers ODL-04 (`CopySubrow` vs `BinColumn::gather`, bagging draw vs host `bag_data_indices`, arbitrary index set).
- [ ] In-test sparse-column synthesizer (helper) — generates columns whose nnz forces each `row_ptr_type{16,32,64}` + a column over the shared-hist budget (D-04). Lives in the test file or a `tests/` helper module.
- [ ] No framework install needed — cargo test is in place.

## Security Domain

`security_enforcement: true`, ASVS level 1. This is a **device-buffer/index-arithmetic** phase with no auth/session/network/crypto surface; the only applicable category is **input validation** at the unsafe-launch boundary.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | Validate every `used_indices[i] ∈ [0, num_data)` and every bin/row-ptr index in-range **before** the `unsafe { create_from_slice / launch }`, returning `ComputeError` (the project's CMP-01 confined-unsafe + T-04-01 boundary-validation pattern — see `partition.rs:222-241`). The C++ raw-pointer table has no bounds check; the Rust port must add one at the host boundary. |
| V6 Cryptography | no | — (the LCG is a PRNG for sampling, not crypto — do not treat as secure RNG) |

### Known Threat Patterns for this phase
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| `CopySubrow` `used_indices` out of `[0, num_data)` → OOB device read | Tampering / Info-disclosure | Host-validate every index before launch; `ComputeError::BinIndexOutOfRange`-style error (the `partition.rs` precedent). |
| `n_blocks * BAGGING_RAND_BLOCK` or `num_used * width` length overflow in buffer sizing | Denial of Service / memory corruption | `checked_mul` at the V5 boundary (the `validate_draw_inputs` precedent, `random.rs:146`). |
| Sparse `row_ptr` width too small for actual nnz → truncated offsets → OOB | Tampering | Pick `row_ptr_type` by actual max nnz (Pattern 3); assert the chosen width fits before upload. |
| Unsafe `from_raw_parts` handle outliving its allocation | Memory corruption | Keep all `cubecl` unsafe confined to the launcher fn with a SAFETY comment proving handle sizing + lifetime (CMP-01, the established convention). |

## Sources

### Primary (HIGH confidence)
- `docs/cuda-kernel-design.md` §3 (Dataset on device / `CUDAColumnData` / `CopySubrowKernel_ColumnData`), §7.0–7.2 (histogram storage + geometry this layout feeds), §13 (`CUDARowData`, `DivideCUDAFeatureGroups`, the accessor table, `GetSparseDataPartitioned`), §14 (host-class roles), §15 (`CUDARandom`), §16 (end-to-end sequencing), §17 (port considerations) — the registered authoritative port-source map.
- In-tree code (read this session): `crates/lgbm-compute/src/lib.rs` (`BinColumn` `:52-188`, `upload_resident_bins` `:2364`, `resident_bin_width` `:2056`), `crates/lgbm-compute/src/kernels/partition.rs` (generic-over-`Int` kernel `:46` + `launch_native!` `:371`), `crates/lgbm-compute/src/kernels/random.rs` (`draw_next_float_on` `:240`, `cuda_next_float` `:77`), `crates/lgbm-compute/src/kernels/split_info.rs` (`DeviceSplitInfo::new` `:267`/`292`), `crates/lgbm-compute/src/kernels/histogram.rs` (`<B:Int>` resident precedent `:1090`), `crates/lgbm-boosting/src/sample_strategy.rs` (`bagging()` `:239`, `BAGGING_RAND_BLOCK` `:46`, seeding `:160`), `crates/lgbm-compute/tests/cuda_random_parity.rs` (the anchor-test shape).
- `cubecl_manual/manual/cubecl/13_memory_preallocation.md` — `client.empty`/`empty_tensor` once-outside-the-loop (D-09).
- `.planning/phases/15-…/15-CONTEXT.md` — the 10 locked decisions.

### Secondary (MEDIUM confidence)
- `.planning/MEMORY.md` notes: spoofed-APU GPU (perf-confounded), narrow-upload/resident-hoist shipped patterns, def-f8u-01 (never GPU-vs-GPU).

### Tertiary (LOW confidence)
- none — no web/external research was needed (pure in-tree port).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every building block is in git and read this session.
- Architecture: HIGH — formulas/geometry transcribed directly from the registered design doc + cross-checked against in-tree precedents.
- Pitfalls: HIGH — derived from §17 port considerations + the existing parity-discipline notes.
- Constant A1 (`shared_hist_size` values): MEDIUM — from the design-doc summary; confirm against `cuda_row_data.hpp` if the reference tree is fetched (no parity impact per §17, only grouping).

**Research date:** 2026-06-30
**Valid until:** stable — port-source + in-tree assets don't drift; revisit only if the design doc or `BinColumn`/`random.rs` APIs change.
