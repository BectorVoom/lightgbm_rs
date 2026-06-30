# Phase 16: On-Device Histogram Constructor - Research

**Researched:** 2026-07-01
**Domain:** CubeCL 0.10 GPU compute kernels — on-device histogram build (two-tier u64 fixed-point atomics) + the subtraction trick (FixHistogram + SubtractHistogram via `hist_t**` pointer rotation), anchored to the cubecl-cpu f64 fold
**Confidence:** HIGH (the codebase already contains every building block; this phase restructures + composes them, it does not invent primitives)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** De-quant after build; **Fix + Subtract run in the `hist_t` float domain**. The u64 fixed-point (S=2^30) stays confined to the BUILD shared-accumulation (its Phase-11 role). The two-tier merge output is de-quanted to durable `hist_t` **once** (cpu anchor = f64, ROCm/CUDA = f32); then `FixHistogram` and `SubtractHistogram` operate on `hist_t` — faithful to C++'s all-`double` Fix/Subtract path, reusing the shipped `subtract.rs` f64/f32 subtract kernels. *(Integer-domain subtract rejected.)*
- **D-02:** Phase 16 owns a **pre-allocated-once histogram arena + an explicit handle contract**; the §9 pool pointer-SWAP is deferred to Phase 18. The build→fix→subtract entry takes `{parent_handle, smaller = fresh_arena_slot, larger = parent_buffer_alias}` and **demonstrates the pointer rotation** (larger derived in-place in the parent's buffer, smaller into a fresh slot — `USED_HISTOGRAM_BUFFER_NUM`-style pool, pre-allocated once per D-09). Anchor-tested in isolation this phase. Cross-tree `SplitTreeStructureKernel` swap → Phase 18.
- **D-03:** **Net-new on-device two-tier kernel on the §13 partition geometry** — `blockIdx.x`=feature partition, `threadIdx.x`=column, `threadIdx.y × blockIdx.y` stripes the leaf's rows; **LDS block-local u64 atomics → cross-block `atomicAdd_system` merge**. Reuses Phase-11's u64 fixed-point accumulation primitive + the already-landed LDS/`SharedMemory`/`sync_cube()` idiom (cubecl 0.10). The shipped one-cube-per-feature + autotuned-row-partition ROCm build stays **byte-unchanged and coexists**.
- **D-04:** **Build the global-memory spill path (`NumLargeBinPartition() > 0`) for real this phase** — the `_GlobalMemory` variant that replaces `__shared__` with a `cuda_hist_buffer_` slice per y-block — anchored by the Phase-15 synthetic large-bin column.
- **D-05:** **Anchor build/fix/subtract to the cpu f64 fold** with committed dense corpora + Phase-15 synthetic sparse (forcing `row_ptr_type` {16,32,64}) & large-bin/global-spill columns (all bit-exact); a purpose-built `most_freq_bin ≠ 0` column to force `FixHistogram`'s omit-and-repair (anchor repaired value = `leaf_total − scanned Σ`); a build-smaller-before-subtract **ordering-invariant test** (8aed100-class); an interleaved `[2b]/[2b+1]` layout assert. **Never GPU-vs-GPU** (def-f8u-01).
- **D-06:** Anchor-pin every numeric output to the cubecl-cpu f64 fold; structure bit-exact; ROCm/CUDA f32 within ~1e-6; tie-aware; never GPU-vs-GPU. cpu anchor = single-owner `CubeDim::new_1d(1)` fold (atomics unsupported/nondeterministic on cubecl-cpu); hip = two-tier LDS kernel — **one `#[cube]` generic, comptime/runtime-split reduction order**.
- **D-07:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU/ROCm/existing-host-CUDA paths **byte-unchanged**; full merge gate green (ODL-19, the hard merge gate).
- **D-08:** **f32 + u64 fixed-point build with NO f64 per-row hot loops** (verified by grep + per-tree-ms); f64 permitted only in scalar/gain math where the reference uses it.
- **D-09:** **Pre-allocate once outside the hot loop** (`client.empty` / `empty_tensor`, reused/indexed across launches; the `split_info.rs` once-in-`new` pattern). Resident dataset uploaded once (Phase-15 hoist).

### Claude's Discretion
- **Geometry tunables** (`NUM_DATA_PER_THREAD=400`, `NUM_THREADS_PER_BLOCK=504`, `grid_dim_y` floor 160, `NUM_FEATURE_PER_THREAD_GROUP=28`, `DP_/SP_SHARED_HIST_SIZE`) are occupancy knobs with NO parity impact. Start from the faithful C++ constants; APU-aware autotune (Phase-13 reuse) is a deferred perf option, not a parity requirement.
- Exact CubeCL module placement (likely extend `crates/lgbm-compute/src/kernels/`: `histogram.rs` build, `subtract.rs` math, new `fix_histogram` + arena/pool module), and the `hist_t**` rotation's concrete handle/enum representation.
- Whether the de-quant step (D-01) is a fused tail of the merge kernel or a separate pass.

### Deferred Ideas (OUT OF SCOPE)
- §9 `SplitTreeStructureKernel` histogram-pool pointer-SWAP + whole-tree pool management → **Phase 18**.
- Discretized / quantized build + fix + subtract kernels (§7.3) → **v2 (QGD-02)**.
- APU-aware autotune of the build geometry → deferred perf option (parity-neutral).
- On-device best-split (reads `hist_in_leaf`) → Phase 17.
- `on_device_growth_supported()` stays **false**.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-09 | On-device histogram build (dense + sparse × shared-memory + global-memory spill), two-tier atomic accumulation (block-local then cross-block merge), f32 / u64 fixed-point path (NO f64 per-row hot loop), anchor-pinned to the cpu f64 fold | Standard Stack + Architecture Patterns 1–3 (two-tier u64 LDS kernel on §13 geometry); reuses `construct_leaf_hist_resident_lds_kernel_u64` accumulation idiom + `FeaturePartitionLayout`; global-spill variant per §7.2; Code Examples 1–3 |
| ODL-10 | Subtraction trick on device — build-smaller-only, `FixHistogram` (most-freq-bin repair = leaf-total − scanned Σ), `SubtractHistogram` (larger = parent − smaller) via `hist_t**` pointer rotation, no bulk copy; correctness requirement (larger-direct takes a different rounding path) | Architecture Patterns 4–5 (de-quant-once → Fix (ShuffleReduceSum / serial-fold anchor) → Subtract); reuses `subtract.rs` kernels + `fix_compact_kernel` Fix logic; arena/handle contract from `split_info.rs` once-alloc pattern; Code Example 4–5; Pitfalls 1, 4, 6 |
</phase_requirements>

## Summary

Phase 16 is a **composition-and-restructuring** phase, not a from-scratch kernel phase. Every numeric primitive it needs already exists in `crates/lgbm-compute/src/kernels/` and is golden-validated: the cpu f64-fold anchor (`construct_histograms_f64_on`, `CubeDim::new_1d(1)`), the **u64 two's-complement fixed-point LDS build** (`construct_leaf_hist_resident_lds_kernel_u64`, S=2^30), the **dequant→FixHistogram fold** (`fix_compact_kernel`, ascending serial fold), the **subtract kernels** (`subtract_hist_kernel` f64/f32 + `_vec`), the **single-owner reductions** (`reduce_sum_body`), the **once-allocated arena pattern** (`DeviceSplitInfo::new` → one `client.empty` per field), and the **§13 feature-partition layout** (`FeaturePartitionLayout` from Phase 15). [VERIFIED: codebase grep]

The genuinely new work is three-fold: (1) a **net-new two-tier build kernel** mapped onto the §13 partition geometry (`blockIdx.x`=partition, `threadIdx.x`=column, `threadIdx.y × blockIdx.y`=row stripes) with **block-local LDS u64 atomics → cross-block global merge** — structurally distinct from the shipped one-cube-per-feature ROCm kernel, which stays byte-unchanged and coexists (D-03); (2) the **`_GlobalMemory` spill variant** for large-bin partitions that exceed shared capacity (D-04); (3) the **`hist_t**` pointer-rotation arena + explicit handle contract** that drives build-smaller / subtract-larger with no bulk copy (D-02). The subtraction trick is a **correctness** requirement — building the larger child directly takes a different f32 rounding path. [CITED: docs/cuda-kernel-design.md §7, §17]

The single largest landmine is sequencing: the parent histogram must be **fully built and synced** before any child subtract reads it. This exact bug shipped once (the Phase-12 co-pack scan deferral ran `subtract_resident` before the fused smaller histogram was built; fixed in debug `8aed100`) — D-05's ordering-invariant test exists precisely to guard it. [VERIFIED: spike-findings skill]

**Primary recommendation:** Extend the existing kernels in place — write ONE `#[cube]` generic whose reduction order is comptime/runtime-split (cpu = single-owner `CubeDim(1)` fold with no atomics; hip = two-tier LDS u64 atomics), confine u64 fixed-point (S=2^30) to BUILD accumulation, de-quant once to `hist_t`, then run Fix + Subtract in the float domain reusing `fix_compact_kernel`'s Fix logic and `subtract.rs`. Anchor everything to the cpu f64 fold; never compare two GPU f32 paths.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-row scatter-accumulate into bins | GPU/device compute (hip LDS u64 atomics) | cpu-anchor (`CubeDim(1)` ordered f64 fold) | The hot loop is the device kernel; the anchor is the deterministic reference, not a perf path |
| Block→global histogram merge | GPU/device compute (`atomicAdd_system`-equiv u64) | — | cubecl 0.10 has NO global grid barrier — cross-block reduction MUST be a global atomic, not a sync |
| De-quant (u64 fixed-point → `hist_t`) | GPU/device compute (one pass, D-01) | host launcher (placement decision) | Confined between merge and Fix; fused-tail-of-merge vs separate pass is Claude's discretion |
| FixHistogram (most-freq-bin repair) | GPU/device compute | cpu-anchor serial fold | Operates on `hist_t` float domain (D-01); per-feature reduce over `num_bin_aligned` |
| SubtractHistogram (larger = parent − smaller) | GPU/device compute (`subtract.rs` kernels) | cpu-anchor | Element-wise, no atomics, bit-exact-by-construction per cell |
| Arena allocation + `hist_t**` handle rotation | host orchestration (`client.empty` once) | — | Pre-allocate once outside the hot loop; rotation is host-side pointer/handle bookkeeping, not a kernel |
| Anchor verification + golden replay | host test harness (`oracle-harness`, `lgbm-compute/tests`) | — | cpu f64 fold is the bit-exact merge gate; never GPU-vs-GPU |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (workspace dep) | `#[cube]` kernels, `SharedMemory`, `Atomic<u64>`, `client.empty`/`create_from_slice`, `sync_cube()`, plane reductions | Already the project's only compute/GPU abstraction (CLAUDE.md: no raw CUDA/OpenCL) [VERIFIED: Cargo.toml line 25] |
| `thiserror` | 2.0.18 | Structured `ComputeError` at the `lgbm-compute` boundary | Project convention (CLAUDE.md error-handling) [VERIFIED: Cargo.toml] |

**No new external packages.** This phase extends `crates/lgbm-compute` (and its `oracle-harness` test crate) using dependencies already in the workspace. [VERIFIED: codebase]

### Supporting (existing in-repo modules to EXTEND, not rebuild)
| Module | Symbol | Purpose | Reuse in Phase 16 |
|--------|--------|---------|-------------------|
| `kernels/histogram.rs` | `construct_histograms_f64_on` (`CubeDim::new_1d(1)`) | The cpu f64-fold **anchor** | The two-tier hip kernel is pinned to its output (D-06) |
| `kernels/histogram.rs` | `construct_leaf_hist_resident_lds_kernel_u64<B: Int>` | u64 fixed-point LDS build (S=2^30) | The **accumulation primitive** to lift onto §13 geometry (D-03) |
| `kernels/histogram.rs` | `fix_compact_kernel` (dequant + FixHistogram fold) | Per-feature dequant→fix (serial ascending fold) | Reuse the **dequant + Fix** logic; DROP the compact (CPU-only artifact — see Pitfall 5) |
| `kernels/histogram.rs` | `construct_hist_cuda_mirror_kernel` + `rocm_cuda_mirror.rs` | Faithful CUDA-mirror dense kernel + its `cpu_anchor`/`assert_close` test scaffold | Test-harness template for the new two-tier kernel |
| `kernels/subtract.rs` | `subtract_hist_kernel` (f64), `_f32`, `_vec<F,N>` | Element-wise `out[i] = parent[i] − child[i]` | The **SubtractHistogram** math, directly (D-01) |
| `kernels/primitives.rs` | `reduce_sum_body<N>` (single-owner fold), LDS scan, `plane_sum` | Reductions for the FixHistogram repair (`ShuffleReduceSum` analog) | Fix's `leaf_total − scanned Σ` reduce |
| `kernels/split_info.rs` | `DeviceSplitInfo::new` (one `client.empty` per field, counted) | The **allocate-exactly-once arena** pattern | Template for the histogram arena + handle contract (D-02/D-09) |
| `kernels/row_data.rs` | `FeaturePartitionLayout`, `divide_cuda_feature_groups` | §13 feature-partition geometry (built in Phase 15) | The build kernel's geometry inputs (`feature_partition_column_index_offsets`, `column_hist_offsets`, `partition_hist_offsets`, `num_large_bin_partition`, `shared_hist_size`) [VERIFIED: row_data.rs] |
| `kernels/column_data.rs` | per-feature meta accessors | `most_freq_bin`, `num_bin`, `offset` for Fix | Fix per-feature scalars |

### Alternatives Considered
| Instead of | Could Use | Tradeoff (why NOT here) |
|------------|-----------|-------------------------|
| Two-tier LDS u64 atomics | f32 atomic `fetch_add` (the old shipped `construct_hist_kernel_lds_f32`) | f32 atomicAdd is a CAS-retry loop (~820 Mr/s, contention-bound) and non-deterministic; u64 `ds_add_u64` is native single-instruction + deterministic + ~3600× accuracy (spike-018/019 SHIPPED) [VERIFIED: spike-findings] |
| `Atomic<u64>` two's-complement | `Atomic<i64>` | **BROKEN in cubecl-hip 0.10** — `Atomic<i64>::store` lowers to `atomicExch(long long*)` which HIP lacks (compiles, link-fails). HARD CONSTRAINT (spike-018b) [VERIFIED: histogram.rs:1258-1264, lib.rs:1264] |
| De-quant once then float Fix/Subtract (D-01) | Integer-domain subtract | Rejected by D-01: diverges from C++ rounding, forces a separate anchor, makes Fix mix a float leaf-total with integer bins |
| Build smaller + subtract larger | Build both children directly | Subtraction is a **correctness** requirement — direct-build of the larger child takes a different f32 rounding path (§17) [CITED: §17] |

**Installation:** None — `cargo build -p lgbm-compute --features gpu` (gpu/rocm feature gates the device kernels; cpu anchor builds without it). [VERIFIED: histogram.rs `#[cfg(feature = "gpu")]` gates]

## Package Legitimacy Audit

> No external packages are installed in this phase. It extends existing in-repo crates using workspace dependencies already vetted (`cubecl` 0.10.0, `thiserror` 2.0.18 from crates.io).

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| cubecl | crates.io | pre-existing dep | — | github.com/tracel-ai/cubecl | OK | Already in workspace (no change) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                         ConstructHistogramForLeaf(smaller, larger)        [host entry, §7.0]
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │  early-return if BOTH children fail min_data/min_hess   │
              └───────────────────────────┬───────────────────────────┘
                                          │ (only the SMALLER leaf is built from data)
                                          ▼
   §13 inputs ─────────────►  [1] TWO-TIER BUILD KERNEL  (hip: u64 LDS atomics ; cpu: CubeDim(1) ordered f64 fold)
   resident bins (u8/16/32)        blockIdx.x = partition                  one #[cube] generic, D-06
   leaf row indices                threadIdx.x = column                    comptime/runtime-split reduction
   ord_g / ord_h (f32)             threadIdx.y × blockIdx.y = row stripe
   FeaturePartitionLayout          ├─ Phase 1: zero LDS (or _GlobalMemory slice)   sync_cube()
                                   ├─ Phase 2: scatter-accumulate  atomicAdd_block (LDS) — quantize round(v·2^30)
                                   │           (sparse: per-row CSR fetch ; large-bin: _GlobalMemory spill, D-04)
                                   └─ Phase 3: merge  atomicAdd_system (cross-block, global)   ← NO grid barrier exists
                                          │
                                          ▼  (smaller leaf RAW u64 fixed-point histogram)
                          [2] DE-QUANT ONCE  (bits as i64)/2^30 → hist_t   (D-01; fused-tail OR separate pass)
                                          │
                                          ▼
                          [3] FixHistogram  (per feature with most_freq_bin ≠ 0)
                                  reduce Σ over num_bin_aligned (ShuffleReduceSum / single-owner fold)
                                  feat_hist[mfb·2]   = leaf_sum_grad − Σ
                                  feat_hist[mfb·2+1] = leaf_sum_hess − Σ      [hist_t float domain]
                                          │
              ┌───────────────────────────┴────── parent histogram (already built+synced) ──────┐
              ▼                                                                                   │
   [4] hist_t** POINTER ROTATION (host, D-02)                                                    │
        larger  ← parent_buffer_alias  (in-place)                                                │
        smaller ← fresh_arena_slot                                                               │
              ▼                                                                                   ▼
   [5] SubtractHistogram  larger_hist[i] -= smaller_hist[i]   (subtract.rs, guarded larger.leaf_index ≥ 0)
              │
              ▼
        hist_in_leaf  for smaller AND larger  →  consumed by Phase 17 (best-split) / Phase 18 (pool swap)
```

### Recommended Project Structure
```
crates/lgbm-compute/src/kernels/
├── histogram.rs       # EXTEND: add two-tier-on-§13-geometry build kernel (dense+sparse, shared+global)
│                      #         + de-quant placement; reuse u64 LDS idiom + fix_compact dequant/Fix logic
├── subtract.rs        # REUSE as-is: subtract_hist_kernel f64/f32/vec for SubtractHistogram (D-01)
├── primitives.rs      # REUSE: reduce_sum_body / LDS / plane_sum for FixHistogram repair
├── row_data.rs        # REUSE: FeaturePartitionLayout (§13 geometry inputs)
├── split_info.rs      # PATTERN: copy the once-in-new client.empty arena discipline
└── histogram_arena.rs # NEW (Claude's discretion): hist_t** pool + handle-rotation contract (D-02)
crates/oracle-harness/tests/
└── kernel_parity.rs   # EXTEND: golden replay for build/fix/subtract; add most_freq_bin≠0 + ordering + [2b]/[2b+1]
crates/lgbm-compute/tests/
└── rocm_cuda_mirror.rs# PATTERN: cpu_anchor + assert_close(tol) template for the new two-tier kernel
```

### Pattern 1: One `#[cube]` generic, comptime/runtime-split reduction (D-06)
**What:** A single kernel body parameterized so the cpu anchor folds in a fixed order via a single-owner `CubeDim::new_1d(1)` lane (no atomics — cubecl-cpu atomics are nondeterministic), while the hip path uses two-tier LDS u64 atomics across all lanes.
**When to use:** Every accumulating kernel in this phase (build, and the Fix reduce).
**Example:**
```rust
// Source: crates/lgbm-compute/src/kernels/histogram.rs (existing pattern, lines 144-198 anchor / 1267-1314 hip)
// cpu anchor — single-owner ordered fold, CubeDim::new_1d(1), no atomics:
//   construct_histograms_f64_on(...)  →  hist_fold_body with `if UNIT_POS == 0 { ...ascending... }`
// hip — two-tier u64 LDS atomics:
//   let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
//   sub[c].store(0u64); sync_cube();
//   sub[ti].fetch_add(qg); sub[ti+1].fetch_add(qh); sync_cube();   // qg = round(v·2^30) bits
//   out[base + m].fetch_add(sub[m].load());                        // cross-block global merge
```

### Pattern 2: Two-tier atomics on the §13 partition geometry (D-03, the defining structure)
**What:** `blockIdx.x` selects a feature partition; `threadIdx.x` selects one column in that partition; `threadIdx.y × blockIdx.y` stripes the leaf's rows. Block-local `atomicAdd_block` into LDS during the sweep, then `atomicAdd_system` (cross-block) to merge each y-block's partial into the global leaf histogram. Many y-blocks cover disjoint row stripes of the same partition → the global merge MUST be atomic. [CITED: §7.1, §7.2]
**Why two tiers and not a barrier:** cubecl 0.10 has **no global grid barrier across cubes** — synchronize within a cube only. The cross-block reduction is therefore a global atomic, not a sync. [VERIFIED: lib.rs:1263 checklist]
**Geometry (start from faithful C++ constants — occupancy knobs, no parity impact, Claude's discretion):**
```
block_dim_x = max_num_column_per_partition          // FeaturePartitionLayout.max_num_column_per_partition
block_dim_y = NUM_THREADS_PER_BLOCK(504) / block_dim_x
grid_dim_x  = num_feature_partitions                 // FeaturePartitionLayout.num_feature_partitions
grid_dim_y  = max(160, ceil(ceil(n_leaf / 400) / block_dim_y))
```

### Pattern 3: De-quant once, between merge and Fix (D-01)
**What:** Dequantize the u64 fixed-point RAW histogram `(bits as i64) / 2^30 → hist_t` exactly once. The existing `fix_compact_kernel` already does this inline as its first pass (`hist[wbi] = f64::cast_from(i64::cast_from(h_raw[wbi])) / SCALE_F64`). [VERIFIED: histogram.rs:2363-2367]
**Decision (Claude's discretion):** fused tail of the merge kernel vs a separate pass. Recommendation: keep it a **separate pass** (or the first pass of the Fix kernel as `fix_compact_kernel` does today) so the BUILD kernel stays a clean u64-only accumulator and the anchor split (atomics on hip, ordered fold on cpu) is not entangled with the float cast.

### Pattern 4: FixHistogram — most-freq-bin omit-and-repair (ODL-10, §7.5)
**What:** During the scatter, the most-frequent bin is omitted. One block per feature with `most_freq_bin ≠ 0`: each thread loads one bin's grad/hess (0 for the most-freq / out-of-range bin), reduce Σ over `num_bin_aligned` (power-of-two, host-precomputed), then thread 0 writes `feat_hist[mfb·2] = leaf_sum_grad − Σ`, `feat_hist[mfb·2+1] = leaf_sum_hess − Σ`. [CITED: §7.5]
**Reuse:** `fix_compact_kernel` already implements this fold (`do_fix = mfb > 0 && mfb < nb`, seed with raw leaf totals, subtract every other bin in ascending order via branchless `select`). [VERIFIED: histogram.rs:2369-2398] The §7.5 CUDA shape uses `ShuffleReduceSum` over `num_bin_aligned`; the cpu anchor uses the ascending serial fold. Keep BOTH under the D-06 comptime/runtime split: cpu = serial ascending fold (bit-exact anchor); hip = `ShuffleReduceSum`/`plane_sum` (within ~1e-6).
**Critical (anchor):** the f64 fold order is **load-bearing — never reorder/parallelize** the cpu anchor path. [VERIFIED: histogram.rs:2383-2385]

### Pattern 5: `hist_t**` pointer rotation arena (D-02, ODL-10)
**What:** `larger` child inherits the parent's buffer (subtract derives it in-place); `smaller` child gets a fresh arena slot. No bulk histogram copy. The pool is `USED_HISTOGRAM_BUFFER_NUM`-style, pre-allocated once.
**Reuse the pattern:** `DeviceSplitInfo::new` is the canonical "allocate exactly once, count the allocations, assert the invariant" template — one `client.empty(len * elem_size)` per buffer in `new`, never elsewhere. Mirror it for the histogram arena (`device_allocations` counter + structural assertion). [VERIFIED: split_info.rs:286-321]
**Scope boundary:** Phase 16 demonstrates the rotation under an **explicit handle contract**, anchor-tested in isolation. The cross-tree `SplitTreeStructureKernel` swap (which leaf becomes which across the growing tree) is **Phase 18** — do NOT build the whole-tree pool manager.

### Anti-Patterns to Avoid
- **Subtract-before-build:** running SubtractHistogram before the parent (or fused smaller) histogram is fully built + synced. This is the `8aed100` bug class. D-05's ordering-invariant test guards it. [VERIFIED: spike-findings]
- **Comparing two GPU f32 paths:** never assert hip-vs-cuda or hip-vs-hip; pin to the cpu f64 fold (def-f8u-01; host-vs-device max 1.907e-6 is inherent f32 noise, NOT a bit-exact swap). [VERIFIED: spike-findings]
- **f64 per-row hot loop:** spike-052 measured a 5.4× regression from an f64 mega-kernel on consumer NVIDIA (1/32 f64 throughput). Keep BUILD in u64 fixed-point; de-quant once; f64 only in scalar/gain math. [VERIFIED: spike-findings]
- **`Atomic<i64>`:** broken on cubecl-hip 0.10 (see Alternatives). Use `Atomic<u64>` two's-complement.
- **Porting the compact step into the §7 path:** the CUDA histogram constructor does NOT compact (see Pitfall 5).
- **A global grid barrier:** does not exist in cubecl 0.10; use the two-tier atomic merge.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Element-wise larger = parent − smaller | A new subtract kernel | `subtract.rs::subtract_hist_kernel` (f64/f32/`_vec`) | Already shipped, bit-exact per cell, Vector<P,N> twin already won (spike-041) [VERIFIED] |
| u64 fixed-point accumulation idiom | New quantize/store/merge | `construct_leaf_hist_resident_lds_kernel_u64`'s `round(v·2^30) → i64 bits → wrapping fetch_add` | Phase-11 SHIPPED, deterministic, ~1e-6 parity proven [VERIFIED] |
| Dequant + FixHistogram fold | New fix kernel | `fix_compact_kernel`'s dequant + Fix logic (drop the compact) | Verbatim port of host `fix_histogram.rs`, ascending f64 fold is the anchor [VERIFIED] |
| Once-allocated device arena | Per-launch `client.empty` | `DeviceSplitInfo::new` pattern (alloc once in `new`, counted, asserted) | D-09; per-hot-loop alloc forces context switches (cubecl manual §Purpose) [CITED] |
| §13 partition geometry | Recompute partitions | `FeaturePartitionLayout` / `divide_cuda_feature_groups` (Phase 15) | Built and tested in Phase 15; the histogram launcher consumes it directly [VERIFIED] |
| Single-owner ordered reductions | New reduce | `primitives.rs::reduce_sum_body` (cpu anchor) + `plane_sum` (hip) | Anchor folds in fixed order; matched-order dot/sum already validated [VERIFIED] |

**Key insight:** This phase's risk is in **wiring and sequencing** (geometry mapping, dequant placement, build-then-subtract ordering, arena handle contract), not in numeric primitives. Treat every "new kernel" as a re-composition of a validated body under the D-06 comptime/runtime split.

## Runtime State Inventory

> Not a rename/refactor/migration phase — this section is N/A. Phase 16 is additive new device code behind `LGBM_CUDA_ON_DEVICE`, off by default; no stored data, service config, OS state, secrets, or build artifacts carry a renamed string. (None — verified: the phase adds kernels + tests, changes no existing on/by-default path per D-07.)

## Common Pitfalls

### Pitfall 1: Subtract reads the parent before the build completes
**What goes wrong:** SubtractHistogram (or a fused smaller-build + subtract) runs before the parent / smaller histogram is fully built and synced → garbage larger-child histogram.
**Why it happens:** Kernel-launch reordering / deferral for perf (the Phase-12 sibling-co-pack scan deferral did exactly this).
**How to avoid:** Enforce build-fully-synced-before-subtract; D-05's explicit ordering-invariant test (8aed100-class) is mandatory. With no global grid barrier, the "sync" is the kernel-boundary / `SynchronizeCUDADevice` between build and subtract.
**Warning signs:** larger-child cells equal to (or off by exactly) the parent; intermittent parity failures that vanish when launches are serialized. [VERIFIED: spike-findings, debug 8aed100]

### Pitfall 2: `Atomic<i64>` link failure on hip
**What goes wrong:** Code compiles, then fails to link / launch on cubecl-hip 0.10.
**Why it happens:** `Atomic<i64>::store` lowers to `atomicExch(long long*)`, which HIP lacks.
**How to avoid:** Use `Atomic<u64>` with `.store(0u64)` / `.fetch_add(qbits)`; the wrapping u64 add IS the i64 two's-complement add. (`wrapping_add` is NOT a kernel intrinsic — the atomic wraps natively.) [VERIFIED: histogram.rs:1258-1264, lib.rs:1264-1265]

### Pitfall 3: f64 in the per-row hot loop
**What goes wrong:** ~5.4× slowdown on consumer NVIDIA (and the merge-gate D-08 grep check fails).
**Why it happens:** Naively accumulating grad/hess in f64 inside the scatter.
**How to avoid:** BUILD accumulates u64 fixed-point (S=2^30); de-quant once to `hist_t` AFTER the merge. Verified by grep + per-tree-ms, not a 6× sweep (D-08). [VERIFIED: spike-findings spike-052]

### Pitfall 4: most_freq_bin handling — when to fix and what value
**What goes wrong:** Wrong default-bin value, or fixing when you shouldn't (DEF-07-02 class).
**Why it happens:** The most-freq bin is omitted during scatter and must be reconstructed; `most_freq_bin == 0` is special (bin 0 is never directly folded), and `mfb >= num_bin` is a defensive bound.
**How to avoid:** Fix only when `mfb > 0 && mfb < num_bin`; repaired value = `leaf_total − Σ(other bins)` in ascending order. Reuse `fix_compact_kernel`'s exact guard. NOTE: DEF-07-02 (most_freq_bin==0 **compaction** diverges from C++ for non-constant hessians) is a *compaction* issue — and the §7 CUDA path does NOT compact (Pitfall 5), so Phase 16 sidesteps it, but the Fix guard must still match. [VERIFIED: histogram.rs:2369-2398; MEMORY.md DEF-07-02]

### Pitfall 5: Don't port the `compact` step
**What goes wrong:** Adding the offset-compaction (shift bins down by `offset`, zero the tail) into the §7 path produces a histogram shape the CUDA reference never produces.
**Why it happens:** The existing `fix_compact_kernel` bundles dequant + Fix + **compact** because it serves the CPU learner path (`learner.rs:2838-2864`, `Dataset::FixHistogram`). The CUDA histogram constructor (§7) does **build → fix → subtract** only — compaction is a CPU-path artifact.
**How to avoid:** Reuse `fix_compact_kernel`'s dequant + Fix logic but **omit the compact block** (`if off > 0 {...}`) for the §7-faithful on-device path. Anchor the new path to the cpu f64 fold for the *uncompacted* histogram. [VERIFIED: histogram.rs:2400-2427; CITED: §7.5 lists only Fix + Subtract]

### Pitfall 6: `_GlobalMemory` spill sizing and indexing (D-04)
**What goes wrong:** Out-of-bounds or wrong partial-histogram slot when a single column's bin count exceeds shared capacity.
**Why it happens:** The `_GlobalMemory` variant replaces `__shared__` with a `cuda_hist_buffer_` slice at `(blockIdx.y · num_total_bin + phs) · 2` — one partial histogram per y-block in global memory; buffer sized `grid_dim_y · num_total_bin · {4 if DP, 2 if SP}`.
**How to avoid:** Pre-allocate the spill buffer once (D-09) sized for `grid_dim_y · num_total_bin`; gate the variant on `NumLargeBinPartition() > 0` (a comptime flag, no parity impact, §17); anchor with the Phase-15 synthetic large-bin column. [CITED: §7.2, §7.6]

### Pitfall 7: cubecl-cpu cannot do atomics
**What goes wrong:** Trying to run the two-tier atomic kernel on the cpu anchor → nondeterministic / unsupported.
**How to avoid:** The cpu anchor MUST be the single-owner `CubeDim::new_1d(1)` ordered fold (no atomics, fixed reduction order). cubecl-cpu threads the CubeDim/UNIT axis; `CubeDim(1)` = serial owner. The two-tier LDS path is hip-only (`#[cfg(feature = "gpu")]`). [VERIFIED: histogram.rs:180, MEMORY.md cubecl-cpu]

## Code Examples

### Example 1: u64 fixed-point LDS accumulation (the BUILD primitive to lift onto §13)
```rust
// Source: crates/lgbm-compute/src/kernels/histogram.rs:1267-1314 (construct_leaf_hist_resident_lds_kernel_u64)
const SCALE_F32: f32 = 1_073_741_824.0; // 2^30  (build side)
let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
// 1. zero this feature's LDS cells (u64 zero = additive identity bits)
let mut c = UNIT_POS as usize;
while c < feat_len { sub[c].store(0u64); c += cd; }
sync_cube();
// 2. scatter rows into LDS, quantizing to fixed-point i64-bits
let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE_F32)));
let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE_F32)));
sub[ti].fetch_add(qg);                 // ti = bin * 2   ([2b] grad / [2b+1] hess)
sub[ti + 1].fetch_add(qh);
sync_cube();
// 3. merge LDS -> global slot (wrapping u64 add == i64 two's-complement)
let mut m = UNIT_POS as usize;
while m < feat_len { out[base + m].fetch_add(sub[m].load()); m += cd; }
// Phase-16 change: map (f=CUBE_POS_X) → PARTITION (blockIdx.x), add threadIdx.x=column,
//                  threadIdx.y × blockIdx.y row stripes; cross-block merge = atomicAdd_system equiv.
```

### Example 2: De-quant once (fused first pass of Fix)
```rust
// Source: crates/lgbm-compute/src/kernels/histogram.rs:2348-2367 (fix_compact_kernel)
const SCALE_F64: f64 = 1_073_741_824.0; // 2^30  (dequant side, same value in f64)
for w in 0..nb {
    let wbi = base + (w as usize) * 2;
    hist[wbi]     = f64::cast_from(i64::cast_from(h_raw[wbi]))     / SCALE_F64;
    hist[wbi + 1] = f64::cast_from(i64::cast_from(h_raw[wbi + 1])) / SCALE_F64;
}
```

### Example 3: FixHistogram repair (cpu-anchor serial fold; load-bearing order)
```rust
// Source: crates/lgbm-compute/src/kernels/histogram.rs:2373-2397 (fix_compact_kernel)
let do_fix = mfb > 0 && mfb < nb;          // skip mfb==0 and out-of-range
if do_fix {
    let mut g = 0.0f64; let mut h = 0.0f64;
    g += sum_gradient;  h += sum_hessian;  // RAW leaf totals (host f64, never quantized)
    for i in 0..nb {                        // ASCENDING — never reorder on the anchor path
        let bi = base + (i as usize) * 2;
        let take = i != mfb;                // branchless exclude of the most-freq bin
        g -= select(take, hist[bi],     0.0);
        h -= select(take, hist[bi + 1], 0.0);
    }
    let mi = base + (mfb as usize) * 2;
    hist[mi] = g; hist[mi + 1] = h;         // feat_hist[mfb·2] = leaf_total − Σ
}
// hip variant (D-06): replace the ascending fold with ShuffleReduceSum/plane_sum over num_bin_aligned.
```

### Example 4: SubtractHistogram (reuse subtract.rs verbatim)
```rust
// Source: crates/lgbm-compute/src/kernels/subtract.rs:44-50
#[cube(launch)]
pub fn subtract_hist_kernel(parent: &Array<f64>, child: &Array<f64>, out: &mut Array<f64>) {
    // 1D grid-stride: each thread one independent f64 cell, no atomics, bit-exact per cell.
    // larger = parent − smaller (guard larger.leaf_index ≥ 0 at the host/launch level).
}
// f32 mirror (hip, no-f64 device): subtract_hist_kernel_f32 ; SIMD twin: subtract_hist_kernel_vec<F,N>
```

### Example 5: Allocate-exactly-once arena (the D-02/D-09 handle contract template)
```rust
// Source: crates/lgbm-compute/src/kernels/split_info.rs:286-321 (DeviceSplitInfo::new)
let mut device_allocations = 0usize;
let mut alloc = |elem_size: usize, len: usize| -> Handle {
    device_allocations += 1;
    client.empty(len * elem_size)        // the ONLY client.empty call in the module — runs only in new()
};
// ... one alloc(...) per buffer; assert device_allocations == NUM_FIELD_BUFFERS afterwards.
// Phase 16: a USED_HISTOGRAM_BUFFER_NUM-slot hist_t pool, allocated once; rotation swaps handles,
//           never reallocates. larger = parent_buffer_alias (in-place) ; smaller = fresh slot.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| f32 atomic `fetch_add` histogram build | u64 two's-complement fixed-point (S=2^30) | Phase 11 / spike-018/019 | Deterministic, ~1e-6 parity, ~3600× accuracy, native `ds_add_u64` (no CAS retry) |
| `Line<T>` | `Vector<P,N>` | cubecl 0.10 | The SIMD type name; subtract already vectorized (spike-041 SHIPPED) |
| One-cube-per-feature build (shipped ROCm path) | Net-new two-tier on §13 partition geometry (Phase 16) | this phase | Matches the §7 CUDA two-tier structure; the per-feature path stays byte-unchanged + coexists (D-03) |
| `fix_compact_kernel` (dequant+fix+**compact**) for CPU learner | dequant+fix only for the §7 on-device path | this phase | Compact is a CPU artifact; §7 = build→fix→subtract |

**Deprecated/outdated:**
- `Atomic<i64>` on cubecl-hip 0.10 — link-fails; replaced by `Atomic<u64>` two's-complement.
- The "atomic-bound build" framing (spike-015) — post-u64 the wide build is uncoalesced-bin-gather-bound (spike-030); but the stable-partition monotone `leaf_rows` order already banks ~70% of coalescing, so the build is effectively tuned on the APU (perf only, not parity).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `HIST_LDS_MAX` is large enough for the largest per-partition (not per-feature) histogram under the new two-tier geometry | Pattern 2 / Example 1 | LDS overflow on wide partitions → must fall to `_GlobalMemory` sooner; verify the constant vs `shared_hist_size` budget when mapping to partitions [ASSUMED — `HIST_LDS_MAX` was sized for per-feature; partitions pack multiple columns] |
| A2 | `ShuffleReduceSum` for the hip Fix path can be expressed with `plane_sum` (≤ one plane width) over `num_bin_aligned` without a cross-plane sum | Pattern 4 | If `num_bin_aligned` > plane width, need a multi-plane reduce or the single-owner fold; cpu anchor unaffected [ASSUMED — lib.rs:1266 warns plane-sum spans at most one plane] |
| A3 | The Phase-15 synthetic sparse + large-bin fixtures are reusable as-is for build/fix/subtract anchoring (they were built for the dataset/partition layer) | Verification fixtures | May need extension to carry ordered grad/hess + leaf totals for the Fix anchor [ASSUMED — confirm fixture shape in planning] |

**If this table is empty:** N/A — three assumptions flagged for planner/discuss confirmation.

## Open Questions (RESOLVED)

> All three are Claude's-discretion items resolved during planning: Q1 → separate de-quant pass (16-03 T3); Q2 → `HistArena` struct with `Vec<Handle>` slots + `{parent,smaller,larger}` indices (16-02); Q3 → extend the `rocm_cuda_mirror` scaffold (16-01). Recommendations below are the adopted choices.

1. **De-quant placement: fused merge-tail vs separate pass (D-01, Claude's discretion).**
   - What we know: `fix_compact_kernel` currently dequants as the Fix kernel's first pass; the BUILD merge writes u64.
   - What's unclear: whether folding dequant into the BUILD merge's global-write reduces a launch without entangling the anchor's atomics-vs-ordered-fold split.
   - Recommendation: keep dequant as a separate pass / Fix first-pass (the shipped shape) — cleanest anchor separation; revisit fusion only as a deferred perf option.

2. **`hist_t**` rotation handle representation (D-02, Claude's discretion).**
   - What we know: `DeviceSplitInfo` uses one `Handle` per field; the histogram pool is `USED_HISTOGRAM_BUFFER_NUM` slots.
   - What's unclear: enum-of-slot-index vs `Vec<Handle>` + rotation indices vs a small `HistArena { slots: Vec<Handle>, free: ..., parent_idx, smaller_idx, larger_idx }`.
   - Recommendation: a small arena struct holding `Vec<Handle>` slots + explicit `{parent, smaller, larger}` slot indices; rotation reassigns indices (larger ← parent's index), never reallocates. Anchor-test the index bookkeeping in isolation (D-02).

3. **Whether the new two-tier kernel reuses the `construct_hist_cuda_mirror_kernel` test scaffold or needs a fresh §13-aware harness.**
   - What we know: `rocm_cuda_mirror.rs` has `cpu_anchor` + `assert_close(tol)` for the dense mirror.
   - Recommendation: extend that scaffold; add partition-layout-aware cases (sparse row_ptr {16,32,64}, large-bin spill, mfb≠0, ordering, [2b]/[2b+1]).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | All compute code | ✓ | rust 1.95 (workspace) | — |
| cubecl | `#[cube]` kernels | ✓ | 0.10.0 | — |
| cubecl-cpu runtime | cpu f64-fold anchor (hard merge gate) | ✓ | via cubecl 0.10 `cpu` feature | — |
| cubecl-hip (ROCm) | the two-tier u64 LDS device kernel | ✓ (local, SPOOFED 8-CU gfx1152 APU) | cubecl 0.10 `rocm` | parity valid; ALL perf numbers APU-confounded |
| Real discrete CUDA | perf validation (deferred to Phase 23) | ✗ locally | — | Kaggle CLI A/B harness (authenticated as boomvector) |

**Missing dependencies with no fallback:** none (parity gating is fully local on cpu + the APU).
**Missing dependencies with fallback:** real discrete-GPU perf — Kaggle (deferred to Phase 23, out of scope for Phase 16; this phase is parity-only).

**APU caveat (MEMORY.md):** the local "gfx1100" is an HSA-spoofed 8-CU gfx1152 Radeon 860M APU. Parity gates are valid; rocprof counters are impossible; any perf number is APU-confounded. Phase 16 is parity-only — do not gate on perf here (D-08 verifies "no f64 hot loop" by grep + per-tree-ms, not a sweep).

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `oracle-harness` golden replay |
| Config file | per-crate `Cargo.toml`; fixtures under `crates/oracle-harness/tests/fixtures/kernels/` |
| Quick run command | `cargo test -p lgbm-compute --lib` |
| Full suite command | `cargo test -p lgbm-compute --features gpu && cargo test -p oracle-harness --test kernel_parity && cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-09 | Dense build bit-exact to cpu f64 fold (committed corpora) | golden replay | `cargo test -p oracle-harness --test kernel_parity` | ✅ extend |
| ODL-09 | Sparse build, row_ptr {16,32,64} synthetic | unit (synthetic fixture) | `cargo test -p lgbm-compute --features gpu sparse_build` | ❌ Wave 0 |
| ODL-09 | Large-bin / `_GlobalMemory` spill | unit (Phase-15 large-bin col) | `cargo test -p lgbm-compute --features gpu global_spill` | ❌ Wave 0 |
| ODL-09 | `[2b]/[2b+1]` interleave assert (grad@2b, hess@2b+1) | unit | `cargo test -p lgbm-compute --features gpu layout_interleave` | ❌ Wave 0 |
| ODL-10 | most_freq_bin≠0 Fix repair value = leaf_total − Σ | unit (purpose-built col) | `cargo test -p lgbm-compute --features gpu fix_mfb_nonzero` | ❌ Wave 0 |
| ODL-10 | SubtractHistogram larger = parent − smaller | unit | `cargo test -p lgbm-compute --features gpu subtract_trick` | ✅ subtract.rs has kernels; add anchored case |
| ODL-10 | Build-smaller-before-subtract ordering invariant (8aed100-class) | unit | `cargo test -p lgbm-compute --features gpu subtract_ordering` | ❌ Wave 0 |
| ODL-09/10 | hip path within ~1e-6 of cpu anchor (NEVER GPU-vs-GPU) | rocm parity | `cargo test -p lgbm-compute --features gpu --test rocm_cuda_mirror` | ✅ extend scaffold |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute --lib` + clippy on edited files
- **Per wave merge:** `cargo test -p lgbm-compute --features gpu && cargo test -p oracle-harness --test kernel_parity`
- **Phase gate:** `cargo test --workspace` GREEN (the D-07 merge gate: CPU/ROCm/host-CUDA byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset) before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] Synthetic sparse-column fixtures forcing `row_ptr_type` {16,32,64} (reuse/extend Phase-15 fixtures — A3)
- [ ] Synthetic large-bin / global-spill column fixture (reuse Phase-15 large-bin col)
- [ ] Purpose-built `most_freq_bin ≠ 0` column + anchored repaired default-bin value
- [ ] Ordering-invariant test harness (parent fully built+synced before child subtract; 8aed100-class)
- [ ] `[2b]/[2b+1]` interleave assert helper
- [ ] Extend `rocm_cuda_mirror.rs` `cpu_anchor`/`assert_close` scaffold to the two-tier + sparse + spill cases

## Security Domain

> security_enforcement is enabled (config.json). This phase is GPU compute kernels over already-validated, in-process binned data — no network, auth, session, or external input surface. The relevant control is **memory-safety of launch arguments**, consistent with the existing V5-style boundary checks in the codebase.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate launch-arg bounds at the host launcher BEFORE `launch_unchecked` (typed `ComputeError`): `num_bin > 0`, `2*num_bin` overflow, `slot_off + region ≤ raw.len()`, arena slot indices in range — mirrors `fix_compact_f64_on` / `DeviceSplitInfo::new` V5 checks [VERIFIED: histogram.rs:2445-2448, split_info.rs:276-284] |
| V6 Cryptography | no | — |

### Known Threat Patterns for cubecl GPU kernels
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| OOB device write via bad slot_off / arena index | Tampering | Host-side V5 length/overflow checks before any `launch_unchecked` (the existing pattern) |
| u64 fixed-point overflow on pathological leaves | Tampering / DoS | i64@2^30 safe to ~1e9 rows × |g|≤8 (Phase-11 bound); add a scale/range check or clamp + document the bound [CITED: Phase-11 SPEC item 3] |
| `launch_unchecked` invariant violation | Tampering | Uphold launch-arg invariants by hand (lib.rs checklist); keep the counted allocate-once assertion |

## Sources

### Primary (HIGH confidence)
- `docs/cuda-kernel-design.md` §7.0–7.6 (Histogram Constructor), §13 (Device Row Data / feature-partition layout), §17 (Port Considerations) — the C++ port-source spec [CITED]
- `crates/lgbm-compute/src/kernels/histogram.rs` — cpu f64 anchor, u64 fixed-point LDS build, `fix_compact_kernel`, CUDA-mirror kernel [VERIFIED: read]
- `crates/lgbm-compute/src/kernels/subtract.rs`, `primitives.rs`, `split_info.rs`, `row_data.rs` — reused primitives + arena pattern + §13 layout [VERIFIED: read]
- `.claude/skills/spike-findings-lightgbm_rs` — u64 atomics SHIPPED, `Atomic<i64>` broken, no-f64-hot-loop (spike-052), 8aed100 ordering bug, never-GPU-vs-GPU (def-f8u-01), Vector<P,N> [VERIFIED: skill]
- `.planning/phases/11-gpu-fixedpoint-int-atomics/SPEC.md` — S=2^30, u64 two's-complement, overflow bound [CITED]
- `.planning/phases/16-on-device-histogram-constructor/16-CONTEXT.md` — D-01..D-09 locked decisions [VERIFIED]

### Secondary (MEDIUM confidence)
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` — `client.empty`/`empty_tensor` once-allocation pattern [CITED]
- `crates/lgbm-compute/tests/rocm_cuda_mirror.rs`, `crates/oracle-harness/tests/kernel_parity.rs` — anchor test scaffolds [VERIFIED: read]
- MEMORY.md — APU spoof, cubecl-cpu single-owner fold, DEF-07-02 compaction note [VERIFIED]

### Tertiary (LOW confidence)
- Geometry tunables / occupancy (`NUM_DATA_PER_THREAD=400`, etc.) — faithful C++ constants, parity-neutral; APU-confounded perf [ASSUMED for perf, CITED for values from §7.0]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every primitive read directly from the codebase; cubecl 0.10 confirmed in Cargo.toml
- Architecture: HIGH — §7/§13/§17 design doc + existing kernels give the exact two-tier + subtraction structure
- Pitfalls: HIGH — all sourced from shipped spikes (018/019/052), the 8aed100 debug, and read code (Atomic<i64>, compact-is-CPU-only)
- Geometry tunables (perf): LOW — APU-confounded; parity-neutral by §17, so does not affect the gate

**Research date:** 2026-07-01
**Valid until:** 2026-07-31 (stable — internal codebase + pinned cubecl 0.10; revisit if cubecl bumps or Phase 15 fixtures change)
