# Phase 16: On-Device Histogram Constructor - Context

**Gathered:** 2026-06-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the hot-path histogram — **build → fix → subtract** — entirely on device, behind
`LGBM_CUDA_ON_DEVICE`. Build only the **smaller** leaf from data; derive the **larger**
sibling by subtraction (`larger = parent − smaller`) via `hist_t**` pointer rotation;
repair the most-frequent-bin omission with `FixHistogram`. Everything is **additive and
off by default**; CPU / ROCm / existing-host-CUDA paths stay byte-unchanged and the full
merge gate stays green. Anchor-pinned to the cpu f64 fold (bit-exact structure; ROCm/CUDA
f32 within ~1e-6).

**Delivers (ODL-09, ODL-10):**
- On-device **histogram build** (dense + sparse × shared-memory + global-memory spill) on
  the f32 / **u64 fixed-point** accumulation path with **two-tier atomic accumulation**
  (block-local LDS then cross-block merge), reusing the shipped Phase-11 u64 fixed-point
  build kernel's accumulation primitive on the new §13 partition geometry. (§7.1–7.4)
- The **subtraction trick on device** — build-smaller-only, **`FixHistogram`**
  (most-frequent-bin repair = leaf-total minus scanned sum), **`SubtractHistogram`**
  (larger = parent − smaller) via **`hist_t**` pointer rotation** (larger inherits the
  parent buffer; smaller gets a fresh arena slot), **no bulk histogram copy**. Preserved as
  a **correctness** requirement — building the larger child directly takes a different
  rounding path. (§7.5, §17)

**Explicitly NOT in this phase:**
- The **§9 SplitTreeStructure pool pointer-SWAP** and the cross-tree histogram-pool
  management → **Phase 18**. Phase 16 demonstrates the rotation under an explicit handle
  contract but does not implement the whole-tree swap driver.
- Best-split finding (Phase 17), data partition / tree mutation / prediction (Phase 18),
  on-device objectives/score/metrics (19–20), the end-to-end on-device tree driver
  (Phase 21).
- **Quantized/discretized build kernels (§7.3)** — **v2 (QGD-02)**, skip.
- `on_device_growth_supported()` stays **false**.

</domain>

<decisions>
## Implementation Decisions

### Subtract numeric domain (ODL-09/10, §7.5, §17)
- **D-01:** **De-quant after build; Fix + Subtract run in the `hist_t` float domain.** The
  u64 fixed-point (S=2^30) stays confined to the **BUILD shared-accumulation** (its
  Phase-11 role). The two-tier merge output is de-quanted to the durable `hist_t` **once**
  (cpu anchor = f64, ROCm/CUDA = f32); **then** `FixHistogram` and `SubtractHistogram`
  operate on `hist_t` — faithful to C++'s all-`double` Fix/Subtract path, reusing the
  shipped `subtract.rs` f64/f32 subtract kernels, and keeping the anchor = cpu f64 fold
  clean. *(Integer-domain subtract was rejected: exactly reproducible on the subtract step
  but diverges from the C++ rounding path, forces a separate anchor argument, and makes Fix
  mix a float leaf-total with integer bins.)*

### Arena & handle protocol (ODL-10, §7.0, §17)
- **D-02:** **Phase 16 owns a pre-allocated-once histogram arena + an explicit handle
  contract; the §9 pool pointer-SWAP is deferred to Phase 18.** The build→fix→subtract
  entry takes `{parent_handle, smaller = fresh_arena_slot, larger = parent_buffer_alias}`
  and **demonstrates the pointer rotation** (larger derived **in-place in the parent's
  buffer**, smaller into a fresh slot — `USED_HISTOGRAM_BUFFER_NUM`-style pool, pre-allocated
  once per D-09). It is **anchor-tested in isolation** this phase. The cross-tree
  `SplitTreeStructureKernel` swap logic (which leaf becomes which across the growing tree)
  stays **Phase 18**. *(Building the full pool rotation manager now was rejected as Phase-18
  scope creep.)*

### Build-kernel structure (ODL-09, §7.1–7.4)
- **D-03:** **Net-new on-device two-tier kernel on the §13 partition geometry** —
  `blockIdx.x` = feature partition, `threadIdx.x` = column, `threadIdx.y × blockIdx.y`
  stripes the leaf's rows; **LDS block-local u64 atomics → cross-block `atomicAdd_system`
  merge**. Reuses **Phase-11's u64 fixed-point accumulation primitive** + the
  **already-landed LDS / `SharedMemory` / `sync_cube()` idiom** (cubecl 0.10). The shipped
  one-cube-per-feature + autotuned-row-partition ROCm build stays **byte-unchanged and
  coexists** (carries forward Phase-15 D-02). *(Bending the existing per-feature build onto
  the new geometry was rejected — risks the byte-unchanged ROCm path and doesn't match the
  §7 two-tier structure SC#1 mandates.)*
- **D-04:** **Build the global-memory spill path (`NumLargeBinPartition() > 0`) for real
  this phase** — the `_GlobalMemory` variant that replaces `__shared__` with a
  `cuda_hist_buffer_` slice per y-block — anchored by the **Phase-15 synthetic large-bin
  column**. *(Skeleton-and-defer was rejected; SC#1 mandates shared + global-spill both.)*

### Verification fixtures (ODL-09/10, §7.5, §17)
- **D-05:** **Anchor build/fix/subtract to the cpu f64 fold with corpora + synthetic
  columns + targeted Fix/ordering tests.** Build anchor = committed **dense corpora** +
  **Phase-15 synthetic sparse** (forcing `row_ptr_type` {16,32,64}) **& large-bin/global-
  spill** columns, all bit-exact to the cpu f64 fold. Add a **purpose-built
  `most_freq_bin ≠ 0` column** to force `FixHistogram`'s omit-and-repair path (DEF-07-02)
  and anchor the repaired default-bin value (`leaf_total − scanned Σ`). Add an explicit
  **build-smaller-before-subtract ordering-invariant test** (the 8aed100-class guard:
  parent fully built/synced before any child subtract reads it) **+ an interleaved
  `[2b]/[2b+1]` layout assert** (grad at `2b`, hess at `2b+1`). **Never GPU-vs-GPU**
  (def-f8u-01). *(Corpora-only was rejected — it leaves sparse/spill, the `most_freq_bin≠0`
  Fix path, and the ordering invariant unexercised — the exact places parity historically
  broke.)*

### Carried forward from Phases 14–15 (NOT re-litigated — hard discipline)
- **D-06:** Anchor-pin every numeric output to the **cubecl-cpu f64 fold**; structure
  bit-exact; ROCm/CUDA f32 within ~1e-6; tie-aware where relevant; **never GPU-vs-GPU**
  (def-f8u-01). The cpu anchor build is the single-owner `CubeDim::new_1d(1)` fold (atomics
  unsupported / nondeterministic on cubecl-cpu); the hip path uses the two-tier LDS kernel —
  one `#[cube]` generic, comptime/runtime-split reduction order.
- **D-07:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA paths
  **byte-unchanged**; full merge gate green and unchanged (ODL-19 — the hard merge gate).
- **D-08:** **f32 + u64 fixed-point build with NO f64 per-row hot loops** (ODL-19; verified
  by grep + per-tree-ms, not a 6× sweep; the 5.4× consumer-NVIDIA f64 regression, spike-052);
  f64 permitted only in scalar/gain math where the reference uses it.
- **D-09:** **Pre-allocate once outside the hot loop** (`client.empty` / `empty_tensor`,
  reused/indexed across launches; the `split_info.rs` once-in-`new` pattern). Resident
  dataset uploaded once (Phase-15 hoist), not re-uploaded per tree.

### Claude's Discretion
- **Geometry tunables** (`NUM_DATA_PER_THREAD=400`, `NUM_THREADS_PER_BLOCK=504`,
  `grid_dim_y` floor 160, `NUM_FEATURE_PER_THREAD_GROUP=28`,
  `DP_/SP_SHARED_HIST_SIZE`) are **occupancy knobs with no parity impact** (§17: the
  shared-vs-global threshold and grid shape don't affect the result as long as the
  in-strategy reduction order is fixed). Start from the faithful C++ constants; **APU-aware
  autotune (Phase-13 reuse) is a deferred perf option, not a parity requirement** — the
  spoofed 8-CU APU makes the NVIDIA-tuned constants ~12× off, but that's perf-only.
- Exact CubeCL module placement (likely extend `crates/lgbm-compute/src/kernels/`:
  `histogram.rs` build, `subtract.rs` math, new `fix_histogram` + arena/pool module), and
  the `hist_t**` rotation's concrete handle/enum representation.
- Whether the de-quant step (D-01) is a fused tail of the merge kernel or a separate pass.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §7 (§7.0–7.6) — **Histogram Constructor**
  (`cuda_histogram_constructor.cu`, 960 lines): `ConstructHistogramForLeaf` entry, the
  interleaved `[2b]/[2b+1]` `cuda_hist_` arena, `CalcConstructHistogramKernelDim` geometry
  (§7.1), the 4 standard build variants (dense/sparse × shared/`_GlobalMemory`, §7.2), the
  host dispatch ladder (§7.4), and **§7.5 finalization** (`FixHistogramKernel` most-freq-bin
  repair via `ShuffleReduceSum`, `SubtractHistogramKernel`). **§7.3 discretized kernels are
  v2 (QGD-02) — skip.**
- `docs/cuda-kernel-design.md` §13 — **Device Row Data / feature-partition layout** that
  feeds §7.1 geometry (`blockIdx.x` = partition, `threadIdx.x` = column;
  `max_num_column_per_partition`, `column_hist_offsets[_full]`,
  `feature_partition_column_index_offsets`, `NumLargeBinPartition()`). Built in Phase 15.
- `docs/cuda-kernel-design.md` §17 — **Port considerations**: atomic-ordering
  nondeterminism → fixed-order f64 anchor; the subtraction trick as a **correctness**
  requirement; most-freq-bin omission + `[2b]/[2b+1]` layout as behavioral; template-flag →
  CubeCL comptime; shared-vs-global spill = capacity choice with no parity impact.
- `.planning/REFERENCE_MANIFEST.md` — v1.1 C++ port-source map + CUDA-support boundaries.

### Reused shipped kernel (the BUILD accumulation primitive)
- `.planning/phases/11-gpu-fixedpoint-int-atomics/SPEC.md` — the **u64 fixed-point
  (S=2^30) build kernel** D-03 reuses on the new §13 partition geometry.

### CubeCL API
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` —
  `client.empty` / `empty_tensor` once, reused — the D-02/D-09 arena pre-allocation pattern.
- cubecl 0.10 LDS idiom (`SharedMemory::new` / `sync_cube()` / shared atomics) as used in
  `crates/lgbm-compute/src/kernels/primitives.rs` (segmented LDS scan) — the D-03 two-tier
  block-local accumulation building block.

### Prior-phase context (discipline carried forward)
- `.planning/phases/15-on-device-device-dataset-row-subset-gather/15-CONTEXT.md` — D-08/09/10
  (anchor to cpu f64 / never GPU-vs-GPU; pre-allocate once; env-gated byte-unchanged) and the
  §13 partition layout + synthetic sparse/large-bin fixtures this phase reuses.
- `.planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-CONTEXT.md`
  — the device-primitive / anchor conventions.

### Existing code to extend / reuse (already in git — DO NOT rebuild)
- `crates/lgbm-compute/src/kernels/histogram.rs` — `construct_histograms` cpu f64-fold
  anchor (single-owner `CubeDim::new_1d(1)`) + f32 mirror + native; the build kernel home
  and the cpu-anchor fold (D-06).
- `crates/lgbm-compute/src/kernels/subtract.rs` — `subtract_hist_kernel` f64/f32/vec + the
  `subtract_histograms_*` runtime-generic path; the D-01 Fix/Subtract math to reuse.
- `crates/lgbm-compute/src/kernels/primitives.rs` — `SharedMemory`/`sync_cube()` LDS scan
  (D-03 LDS building block), `ShuffleReduceSum`-class reductions (FixHistogram repair).
- `crates/lgbm-compute/src/kernels/row_data.rs` / `column_data.rs` — the Phase-15 §13/§3
  device dataset + feature-partition accessors the build kernel reads.
- `crates/lgbm-compute/src/kernels/split_info.rs` — the once-in-`new` `client.empty`
  pre-allocation pattern (D-02/D-09 arena reference).
- `crates/lgbm-compute/src/lib.rs` — `Backend::grow_tree_on_device` (~1272),
  `on_device_growth_supported` (~1239, stays **false** this phase).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`histogram.rs` cpu f64-fold anchor** — the single-owner `CubeDim::new_1d(1)` build that
  IS the bit-exact anchor; the hip two-tier LDS kernel is pinned to its output (D-06).
- **`subtract.rs`** — shipped f64/f32 subtract kernels; D-01 makes Fix/Subtract operate in
  this `hist_t` float domain so these are directly reused (no integer-domain subtract).
- **Phase-11 u64 fixed-point build primitive** — the S=2^30 scaled-integer atomic
  accumulation reused inside the new §13-geometry two-tier kernel (D-03).
- **`primitives.rs` LDS idiom** (`SharedMemory`/`sync_cube`) — block-local accumulation
  (D-03) and the `ShuffleReduceSum`-style reduce for `FixHistogram` (D-05).
- **Phase-15 §13 row/partition store + synthetic sparse/large-bin fixtures** — the build
  kernel's resident input (D-03/D-04) and the verification anchors (D-05).

### Established Patterns
- **Anchor to cpu f64 fold, never GPU-vs-GPU** (def-f8u-01) — D-06.
- **One `#[cube]` generic, comptime/runtime-split reduction** — cpu anchor = single-owner
  fold, hip = two-tier LDS atomics (D-06).
- **Additive, env-gated, byte-unchanged default path** (ODL-19) — D-07.
- **Pre-allocate once outside the hot loop** (`split_info.rs`) — D-02/D-09.
- **Subtraction trick is a CORRECTNESS requirement** (different rounding path) — D-01/D-02.

### Integration Points
- New on-device build/fix/subtract + the histogram arena live in `lgbm-compute`
  (`kernels/`), reading the Phase-15 §13 resident row/partition store. Reached only when
  `LGBM_CUDA_ON_DEVICE=1`; the shipped per-feature ROCm build path is untouched and coexists
  (D-03). Consumed by Phase 17 (best-split reads `hist_in_leaf`) and Phase 18 (the §9 pool
  pointer-swap that this phase's handle contract is shaped for).

</code_context>

<specifics>
## Specific Ideas

- **Two-tier atomics are the defining structure**: `atomicAdd_block` into LDS during the
  row sweep, then `atomicAdd_system` to merge each `blockIdx.y` block's partial into the
  global leaf histogram (disjoint row stripes → atomic global merge required). (§7.2)
- **`FixHistogram` runs one block per feature with `most_freq_bin ≠ 0`**: each thread loads
  one bin's grad/hess (0 for the most-frequent / out-of-range bin), `ShuffleReduceSum` over
  `num_bin_aligned`, thread 0 writes `feat_hist[mfb·2] = leaf_sum_grad − Σ`,
  `[mfb·2+1] = leaf_sum_hess − Σ`. `need_fix_histogram_features_` + power-of-two
  `num_bin_aligned` precomputed host-side. (§7.5)
- **`SubtractHistogram`**: one thread per element, guarded by `larger.leaf_index ≥ 0`,
  `larger_hist[i] −= smaller_hist[i]`. (§7.5)
- **`_GlobalMemory` spill** replaces `__shared__` with a `cuda_hist_buffer_` slice at
  `(blockIdx.y · num_total_bin + phs) · 2`, sized `grid_dim_y · num_total_bin · {4 DP, 2 SP}`.
- **De-quant once** (D-01) between the two-tier merge and Fix — confine u64 fixed-point to
  build accumulation only.

</specifics>

<deferred>
## Deferred Ideas

- **§9 `SplitTreeStructureKernel` histogram-pool pointer-SWAP + whole-tree pool management**
  → **Phase 18** (D-02 defers; Phase 16 only demonstrates rotation under a handle contract).
- **Discretized / quantized build + fix + subtract kernels (§7.3)** → **v2 (QGD-02)**.
- **APU-aware autotune of the build geometry** (Phase-13 reuse) → deferred perf option;
  parity-neutral occupancy knobs, not required this phase (Claude's Discretion).
- **On-device best-split** (reads `hist_in_leaf`) → Phase 17.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 16-on-device-histogram-constructor*
*Context gathered: 2026-06-30*
