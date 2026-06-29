# Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG - Research

**Researched:** 2026-06-29
**Domain:** CubeCL 0.10 device-kernel authoring (warp/plane scans + reductions, LDS block-scans, bitonic argsort), host-side memory pre-allocation (SoA device split-record), device LCG RNG parity, and a C++/HIP `__device__` fixture-capture harness.
**Confidence:** HIGH (scope locked by CONTEXT; mechanics verified against the installed cubecl-0.10 source, the in-repo prior-art kernels, and the AMD-fork C++ reference).

## Summary

This is a **NO-OP seam foundation phase**. The scope is fully locked by CONTEXT.md (D-01..D-11); the research job was to resolve the four CubeCL-0.10 gotchas, the C++ fixture-harness mechanics, the SoA split-record layout, and the validation architecture into concrete, code-shaped guidance. All four gotchas resolve cleanly, and one is materially *easier* than CONTEXT anticipated.

**Headline finding (changes the plan):** cubecl-core **0.10.0 ships first-class plane scan/reduction intrinsics** — `plane_inclusive_sum`, `plane_exclusive_sum`, `plane_sum`, `plane_max`, `plane_min`, plus `plane_shuffle_up/_down/_xor` and `PLANE_DIM`. The C++ `ShufflePrefixSum` (hand-rolled `__shfl` scan) and `ShuffleReduceSum/Max/Min` map **directly** onto these intrinsics for the within-plane level; only the cross-plane (block-wide, 1024-thread) and cross-block (global) levels need the staged LDS + multi-kernel structure. `[VERIFIED: ~/.cargo/.../cubecl-core-0.10.0/src/frontend/plane.rs]`

Three more gotchas resolve to established in-repo idioms: (1) **no global barrier** → the 3-kernel global-scan structure the C++ `ShufflePrefixSumGlobal` already uses (block-scan → block-sums-scan → add-back), one `client.empty` scratch buffer of length `num_blocks` reused across launches; (2) **`Atomic<i64>` broken / no `wrapping_add` intrinsic** → the shipped **u64 two's-complement fixed-point** idiom (`u64::cast_from(i64::cast_from(round(v*SCALE)))` + `Atomic<u64>::fetch_add`) for the i64 quantized field, and **plain `+`/`*` on `u32`** (native hardware wrap) for the LCG recurrence; (3) **plane-sum ≤ plane width** → a segmented **`SharedMemory` block-scan** (per-plane scan via `plane_inclusive_sum`, stage per-plane totals in a ≤32-slot LDS buffer, scan those, add back) for the 256-bin within-feature case. `launch_unchecked` is wrapped exactly as the existing histogram launchers do (confined `unsafe` block + a SAFETY comment discharging the V5 host-side bounds proof, CMP-01/NRW-01).

The C++ fixture harness (D-03) is **buildable locally with `hipcc` against the in-repo AMD fork** (`LightGBM-release-4.6.0.99/src/cuda/cuda_algorithms.cu`) — most multi-block primitives are already host-callable `template`-instantiated wrappers; only the pure block-level `__device__` helpers need a one-line `__global__` shim. No CUDA box / Kaggle is required for fixture *capture* (the primitives are deterministic). RNG needs no harness (D-04: pinned to the existing host `Random`).

**Primary recommendation:** Create `crates/lgbm-compute/src/kernels/primitives.rs` housing the prefix-sum, reduction, single-block bitonic argsort kernels (full depth) + percentile / multi-block-argsort / items-sort skeletons; build them on the cubecl-0.10 `plane_*` intrinsics with LDS staging for block/global levels; pre-allocate the SoA split-record with `client.empty` per field once outside the (future) grow loop; validate against committed C++/HIP fixtures (numeric) and the host `Random` stream (RNG); keep `on_device_growth_supported()==false` and `grow_tree_on_device()==Ok(None)`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Grow-loop subset at **full depth** + anchor-pinned **skeletons** for the rest. Full depth for what Phases 15–18 consume: **prefix-sum** (inclusive/exclusive, block + multi-kernel global) and **shuffle reductions** (sum/max/min, dot-product), plus **single-block bitonic argsort**.
- **D-02:** **Percentile** (weighted/unweighted), **multi-block / `…Global` argsort**, and **`BitonicArgSortItems`** ported this phase as **anchor-pinned skeletons** — correct + tested, finalized by their first consumer (percentile + items → Phase 19; multi-block argsort → Phase 19/22). ODL-01's "all primitives exist" is satisfied by skeletons; depth follows demand.
- **D-03:** **Capture C++ `lib_lightgbm` golden fixtures** for the numeric primitives via a **thin C++ harness** that launches each `__device__` primitive (`ShufflePrefixSum`, `ShuffleReduce*`, `BitonicArgSort*`, `PercentileDevice`) and dumps committed fixtures. Index-only ops (argsort) assert the **permutation** bit-exact; numeric ops assert bit-exact on the cpu f64 anchor and ~1e-6 for ROCm/CUDA f32.
- **D-04:** The **`CUDARandom` LCG** is pinned against the **existing host `Random`** (`crates/lgbm-core/src/random.rs`, already C++-bit-exact `214013·x+2531011`) — NOT a new C++ capture. Assert the device stream reproduces `RandInt16`/`RandInt32`/`NextFloat` bit-for-bit against the host draw sequence.
- **D-05:** **Struct-of-Arrays (SoA), host pre-allocated**, numeric fields + categorical reserved. One pre-allocated buffer per field sized `[num_leaf_slots]`, allocated **once outside the grow loop** (`client.empty` / `empty_tensor`), reused/indexed by leaf-slot.
- **D-06:** Numeric fields now: `is_valid`, `leaf_index`, `gain`, `inner_feature_index`, `threshold` (u32), `default_left`, per-side `{left,right}_sum_gradients/_sum_hessians` (f64), `_count` (i32), `_gain`, `_value` (f64), and quantized `_sum_of_gradients_hessians` (i64). Categorical buffers (`num_cat_threshold`, `cat_threshold`, `cat_threshold_real`) **pre-allocated reserved** now so Phase 22 fills, not restructures.
- **D-07:** The 8-int / 16-int packed readback packet is a **separate** concern; it lands with its consumer (Phase 17/18). Phase 14 defines the resident record only.
- **D-08:** **No per-split / per-`SplitInfo` device alloc** anywhere — the C++ `AllocateCatVectorsKernel` per-split alloc is exactly what CubeCL pre-allocation eliminates.
- **D-09:** **Strict no-op seam.** `on_device_growth_supported()` stays **false**; `grow_tree_on_device(..)` returns **`Ok(None)`**; no histogram/split/growth kernels. Slice-0 no-op seam tests (learner_parity.rs) must remain green.
- **D-10:** Anchor discipline (hard): anchor-pin to the **cubecl-cpu f64 fold**, tie-aware `default_left`, **never** GPU-vs-GPU (def-f8u-01). Leaf/score numerics within ~1e-5 f32 envelope; structure bit-exact.
- **D-11:** `LGBM_CUDA_ON_DEVICE` OFF by default; CPU / ROCm / existing-host-CUDA paths **byte-unchanged**; full merge gate (`raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites) green and unchanged.

### Claude's Discretion
- Exact CubeCL module placement (likely a new `crates/lgbm-compute/src/kernels/primitives.rs`) and whether they share `#[cube]` source with any CPU path — but **the CPU anchor stays a plain serial f64 reference** (cubecl-cpu lost to native on CPU hot paths; memory `unified-cpu-gpu-kernels-pref`); these device primitives are GPU-side, anchored against the C++ fixtures (D-03).
- cubecl-0.10 gotcha handling (no global barrier → multi-kernel global prefix-sum; `Atomic<i64>` broken; `wrapping_add` not an intrinsic; plane-sum ≤ plane width → segmented LDS block-scan; `launch_unchecked` unsafe) — **research, not discussion** (resolved below).

### Deferred Ideas (OUT OF SCOPE)
- Percentile / multi-block argsort / ranking item-sort depth-hardening → Phase 19 / 22.
- 8-/16-int packed split-readback packet wiring → Phase 17 / 18.
- Categorical `cat_threshold` slab fill → Phase 22 (buffers reserved now).
- Quantized/discretized integer histogram & split path (§4, §7.3, §8.1) → v2 (QGD-02), out of v1.1 entirely.
- Any actual on-device tree growth → Phase 21.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-01 | Shared device primitives as reusable CubeCL kernels: block + multi-kernel global prefix-sum (incl/excl), shuffle reductions (sum/max/min, dot-product), bitonic argsort (single/multi-block, index-only), weighted/unweighted percentile — each anchor-pinned where it carries numeric output. | Architecture Patterns §1–4 (plane intrinsics + LDS block-scan + 3-kernel global scan + bitonic). Standard Stack: cubecl 0.10 `plane_inclusive_sum`/`plane_exclusive_sum`/`plane_sum`/`plane_max`/`plane_min`/`plane_shuffle_up`. Validation Architecture maps each to a C++ fixture. |
| ODL-02 | CubeCL-safe pre-allocated device split-record (`CUDASplitInfo` analog, NO per-split device alloc) + `CUDARandom` LCG bit-identical to host `Random`. | SoA layout (§Pattern 5) via `client.empty` per field; deep-copy `operator=` → device/host leaf-slot copy. LCG via plain u32 `+`/`*` (native wrap), pinned to host `Random` (D-04). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Block/global prefix-sum, reductions, argsort, percentile | GPU device kernels (`lgbm-compute/kernels`) | cubecl-cpu f64 reference fold (anchor) | These are `__device__`/`__global__` primitives; the CPU side exists only as the deterministic verification anchor (D-10), not a perf path (cubecl-cpu lost to native). |
| Device split-record (SoA) | GPU device buffers (host pre-allocated handles) | Host indexing for leaf-slot copy | Resident record consumed by future on-device argmax (§8); host owns allocation lifetime (`client.empty` once), device kernels index it. |
| `CUDARandom` LCG | GPU device kernel (`#[cube]`) | Host `Random` (parity oracle) | Per-task seeded device stream for extra-trees threshold selection; bit-identical to the host LCG (D-04). |
| Seam discriminator + oracle | Host (`lgbm-compute` trait / `oracle-harness`) | — | `on_device_growth_supported()` / `grow_tree_on_device()` are host-side trait methods; the oracle is a host test. |
| Fixture capture | Off-tree C++/HIP harness (build-time tool) | Committed `.bin`/`.txt` fixtures | One-time golden capture; not part of the Rust build graph. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (workspace-pinned) | All device kernels (`#[cube]`, plane intrinsics, `SharedMemory`, `Atomic<u64>`, launch) | Already the project's locked compute layer; **no new dependency** this phase. `[VERIFIED: Cargo.toml:25 + Cargo.lock cubecl 0.10.0]` |
| `cubecl-cpu` (via `cubecl` `cpu` feature) | 0.10.0 | The deterministic f64 anchor runtime (`ActiveRuntime`/`CpuBackend`) | D-10 anchor; default feature. `[VERIFIED: crates/lgbm-compute/Cargo.toml:8]` |
| `cubecl-hip` (via `rocm` feature) | 0.10.0 | ROCm device execution + parity track | Existing `rocm` feature; `GpuBackend<R>`. `[VERIFIED: crates/lgbm-compute/Cargo.toml:47]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `bytemuck` | (existing) | `as_bytes`/`from_bytes` for host↔device transfer in launchers | Every host launcher / fixture-reader. `[VERIFIED: used throughout histogram.rs]` |
| `thiserror` | (existing) | `ComputeError` domain errors at the V5 boundary | New primitive launchers return `Result<_, ComputeError>`. `[CITED: CLAUDE.md error-handling]` |
| `hipcc` (ROCm 7.1) | 7.1.52802 | Build the D-03 C++ fixture harness locally | One-time fixture capture only. `[VERIFIED: /opt/rocm/bin/hipcc present]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled `__shfl`-style scan (C++ literal port) | cubecl-0.10 `plane_inclusive_sum`/`plane_exclusive_sum` | The intrinsics are simpler, backend-portable, and avoid re-implementing the warp scan; **verify lowering on cubecl-cpu + cubecl-hip** (Open Question 1) before fully committing — fall back to a `plane_shuffle_up` manual scan if a backend doesn't lower the intrinsic. |
| C++ CUDA capture on Kaggle | local `hipcc` against the AMD fork | hipcc local is faster + already available; primitives are deterministic so device choice doesn't change the golden (modulo f32 reduction-tree order, covered by ~1e-6 tol). Kaggle is the fallback if a primitive fails to hipify. |

**Installation:** No package installs. (cubecl 0.10 already vendored; hipcc already present.)

## Package Legitimacy Audit

> **Not applicable.** This phase installs **no external packages**. All compute uses the already-vetted workspace `cubecl 0.10.0` (and its `cpu`/`hip` features) plus existing deps (`bytemuck`, `thiserror`). The C++ fixture harness uses the system `hipcc`/`cmake`, not a package-manager dependency. No registry verification required.

## Architecture Patterns

### System Architecture Diagram

```text
                        ┌─────────────────────────────────────────────┐
   ODL-01 primitives    │   crates/lgbm-compute/src/kernels/           │
   (NEW primitives.rs)  │                                              │
                        │   #[cube] device kernels                     │
  host slice ──launch──▶│   ┌──────────────┐  ┌──────────────┐         │
   (Vec<T>)             │   │ prefix_sum   │  │ reduce        │         │
                        │   │  block(LDS)  │  │  sum/max/min  │──plane──▶│ plane_inclusive_sum
                        │   │  global(3kn) │  │  dotprod      │  intrins │ plane_sum/max/min
                        │   └──────┬───────┘  └──────────────┘         │ plane_shuffle_up
                        │          │ scratch=client.empty(num_blocks)  │
                        │   ┌──────▼───────┐  ┌──────────────┐         │
                        │   │ bitonic      │  │ percentile    │         │
                        │   │  argsort1024 │  │ (skeleton)    │         │
                        │   │  (idx-only)  │  │ argsort+wpsum │         │
                        │   └──────────────┘  └──────────────┘         │
                        └───────────────┬─────────────────────────────┘
                                        │ read_one → Vec<T>
            ┌───────────────────────────┴───────────────┐
            ▼                                            ▼
  ┌──────────────────────┐                  ┌──────────────────────────┐
  │ ODL-02 split-record  │                  │ ODL-02 CUDARandom (#[cube])│
  │ SoA: client.empty    │                  │ x = x*214013 + 2531011 (u32)│
  │ per field [n_slots]   │                  │ RandInt16/32/NextFloat      │
  │ alloc ONCE off-loop   │                  └────────────┬───────────────┘
  │ copy slot a→b (idx)   │                               │ pinned bit-exact
  └──────────────────────┘                               ▼
                                              crates/lgbm-core/src/random.rs (host Random)
   ─────────── VALIDATION ──────────────────────────────────────────────
   numeric primitives ──▶ C++/HIP fixture (hipcc vs AMD fork) ──┐
                                                                ├─▶ oracle-harness tests
   RNG ──────────────▶ host Random draw sequence ──────────────┘
   seam ──▶ on_device_growth_supported()==false / grow_tree_on_device()==Ok(None) (unchanged)
```

### Recommended Project Structure
```text
crates/lgbm-compute/src/
├── kernels/
│   ├── primitives.rs   # NEW (D-Discretion): prefix_sum, reduce, bitonic argsort (full);
│   │                   #      percentile / multi-block argsort / items-sort (skeleton)
│   ├── split_info.rs   # NEW: SoA device split-record (client.empty per field) + slot-copy op
│   ├── random.rs       # NEW: CUDARandom #[cube] LCG + host launcher for parity draws
│   ├── histogram.rs    # prior art (plane/LDS/u64 atomic patterns) — DO NOT modify
│   └── mod.rs          # add `pub mod primitives; pub mod split_info; pub mod random;`
├── lib.rs              # seam unchanged (on_device_growth_supported / grow_tree_on_device)
crates/oracle-harness/tests/
├── learner_parity.rs   # Slice-0 no-op seam tests STAY green (extend, don't rewrite)
└── primitive_parity.rs # NEW: fixture-driven primitive + RNG parity tests
tools/cuda-fixtures/     # NEW (off-build): hipcc harness + capture script + committed goldens
├── capture_primitives.cu
├── CMakeLists.txt (or build.sh wrapping hipcc)
└── fixtures/*.bin
```

### Pattern 1: Within-plane scan/reduction via 0.10 intrinsics (full depth, block ≤ plane)
**What:** Map the C++ `ShufflePrefixSum`/`ShuffleReduceSum/Max/Min` block helpers directly onto cubecl-0.10 plane intrinsics.
**When to use:** When the active length ≤ `PLANE_DIM` (32 on NVIDIA, 64 on AMD).
```rust
// Source: VERIFIED cubecl-core-0.10.0/src/frontend/plane.rs (plane_inclusive_sum/exclusive_sum/sum/max/min)
use cubecl::prelude::*;
#[cube]
fn plane_scan_incl<F: Float>(v: F) -> F { plane_inclusive_sum(v) }   // C++ ShufflePrefixSum
#[cube]
fn plane_scan_excl<F: Float>(v: F) -> F { plane_exclusive_sum(v) }   // C++ ShufflePrefixSumExclusive
#[cube]
fn plane_reduce_sum<F: Float>(v: F) -> F { plane_sum(v) }            // C++ ShuffleReduceSum
#[cube]
fn plane_reduce_max<F: Numeric>(v: F) -> F { plane_max(v) }          // C++ ShuffleReduceMax
#[cube]
fn plane_reduce_min<F: Numeric>(v: F) -> F { plane_min(v) }          // C++ ShuffleReduceMin
// Dot-product reduction = elementwise mul then plane_sum (C++ ShuffleReduceDotProd).
```
**Note:** `PLANE_DIM` is a runtime builtin; do NOT hard-code 32/64. The plane width difference is exactly why f32 reductions are held to ~1e-6, not bit-exact, across backends (Pitfall 3).

### Pattern 2: Block-wide scan (1024 threads > plane) via segmented LDS staging
**What:** Compose a `GLOBAL_PREFIX_SUM_BLOCK_SIZE=1024` block scan from plane scans + a `SharedMemory` staging buffer — the exact structure C++ uses ("per-warp totals staged in a 32-slot shared buffer").
**When to use:** Block-level prefix-sum, AND the **256-bin within-feature scan** (the plane-sum-≤-plane-width gotcha).
```rust
// Source: ASSUMED structure (standard block-scan); LDS/sync_cube idiom VERIFIED in histogram.rs:801-834
const N_PLANES_MAX: u32 = 32; // 1024 / 32; AMD wave64 → 16 used, 32 is the safe LDS cap
#[cube(launch_unchecked)]
fn block_inclusive_scan<F: Float>(data: &Array<F>, out: &mut Array<F>, n: u32) {
    let stage = SharedMemory::<F>::new(N_PLANES_MAX);
    let lane  = UNIT_POS % PLANE_DIM;
    let plane = UNIT_POS / PLANE_DIM;
    let i = UNIT_POS;
    let v = if i < n { data[i] } else { F::new(0.0) };
    let local = plane_inclusive_sum(v);                 // 1. scan within each plane
    if lane == PLANE_DIM - 1 { stage[plane] = local; }  //    last lane writes plane total
    sync_cube();                                        // 2. block barrier (intra-cube ONLY)
    // 3. one plane scans the per-plane totals (n_planes ≤ 32 ≤ PLANE_DIM)
    if plane == 0 {
        let t = if lane < CUBE_DIM / PLANE_DIM { stage[lane] } else { F::new(0.0) };
        stage[lane] = plane_exclusive_sum(t);
    }
    sync_cube();
    // 4. add the plane's base back
    if i < n { out[i] = local + stage[plane]; }
}
```
**Why this resolves the gotcha:** No primitive ever sums across more than `PLANE_DIM` lanes; the cross-plane combine goes through LDS + `sync_cube()` (intra-cube barrier), never a (non-existent) cross-cube barrier.

### Pattern 3: Multi-kernel global prefix-sum (no global barrier) — 3 launches
**What:** Full-array scan across many blocks. There is **no global barrier between cubes** in CubeCL, so it is split into three kernel launches with one reused scratch buffer — identical to the C++ `ShufflePrefixSumGlobal` (`…Kernel`, `…ReduceBlockKernel`, `…AddBase`).
**When to use:** Arrays longer than one block (Phase-15+ partition scans, percentile weight prefix-sums).
```text
// Source: CITED docs/cuda-kernel-design.md §2.4 (3-kernel structure) + AMD fork cuda_algorithms.cu:17-71
num_blocks = ceil(len / 1024)
scratch    = client.empty(num_blocks * size_of::<T>())   // allocated ONCE, reused (D-05 pre-alloc)
launch 1 (num_blocks × 1024): block_inclusive_scan(data→data); write block total → scratch[block]
launch 2 (1 × 1024):          block_exclusive_scan(scratch→scratch) in place   // ≤1024 blocks fits 1 block
launch 3 (num_blocks × 1024): data[i] += scratch[block]                        // add-back base
```
**Launch count:** exactly 3 (assumes `num_blocks ≤ 1024`, i.e. `len ≤ ~1M`; for larger, launch 2 recurses — out of scope this phase, note for Phase 15). Scratch sized `num_blocks`, allocated via `client.empty` once and reused across iterations (the D-05/D-08 pre-allocation rule applies to primitive scratch too).

### Pattern 4: Single-block bitonic argsort (index-only, full depth this phase)
**What:** Sort an **index array** by key, never moving the values (C++ `BitonicArgSort_1024`). `BITONIC_SORT_NUM_ELEMENTS=1024`, `BITONIC_SORT_DEPTH=11` (`2^11=2048`).
**When to use:** ≤1024 (skeleton extends to 2048 / multi-block in Phase 19/22, D-02).
```rust
// Source: CITED docs/cuda-kernel-design.md §2.4; structure is the standard bitonic network
// indices[] starts as identity (or a provided permutation); keys[] is read-only and UNMOVED.
// For each (k = 2..=n by *2) { for (j = k/2; j>0; j/=2) {
//     partner = i ^ j;
//     if partner > i {
//         ascending = (i & k) == 0;
//         if (keys[indices[i]] > keys[indices[partner]]) == ascending { swap indices[i], indices[partner]; }
//     }
//     sync_cube();   // intra-cube barrier between stages
// }}
```
**Critical:** the comparator reads `keys[indices[i]]` (indirection); only `indices` is swapped. Permutation is asserted **bit-exact** against the C++ fixture (no float tolerance — it's an index array). Tie-handling (equal keys) must match C++ comparator order exactly or the permutation diverges (Pitfall 5).

### Pattern 5: SoA device split-record, host pre-allocated once (D-05/D-06/D-08)
**What:** One `client.empty` buffer **per field**, sized `[num_leaf_slots]`, allocated once outside the (future) grow loop, indexed by leaf-slot. The C++ deep-copy `operator=` becomes a "copy leaf-slot a → b across all field buffers".
```rust
// Source: VERIFIED cubecl_manual 13_memory_preallocation.md (client.empty / empty_tensor, reuse off-loop)
pub struct DeviceSplitInfo<R: Runtime> {
    // numeric (D-06) — one handle per field, each length num_leaf_slots
    pub is_valid:            Handle, // u8/bool
    pub leaf_index:          Handle, // i32
    pub gain:                Handle, // f64
    pub inner_feature_index: Handle, // i32
    pub threshold:           Handle, // u32
    pub default_left:        Handle, // u8/bool
    pub left_sum_gradients:  Handle, pub right_sum_gradients: Handle, // f64
    pub left_sum_hessians:   Handle, pub right_sum_hessians:  Handle, // f64
    pub left_count:          Handle, pub right_count:         Handle, // i32
    pub left_gain:           Handle, pub right_gain:          Handle, // f64
    pub left_value:          Handle, pub right_value:         Handle, // f64
    pub left_sum_gh_quant:   Handle, pub right_sum_gh_quant:  Handle, // i64 (Atomic<u64> twos-complement, Pitfall 2)
    // categorical RESERVED (D-06) — pre-allocated now, FILLED Phase 22 (not restructured)
    pub num_cat_threshold:   Handle, // i32  [num_leaf_slots]
    pub cat_threshold:       Handle, // u32  [num_leaf_slots * MAX_CAT_PER_SPLIT] reserved slab
    pub cat_threshold_real:  Handle, // i32  [num_leaf_slots * MAX_CAT_PER_SPLIT] reserved slab
}
impl<R: Runtime> DeviceSplitInfo<R> {
    pub fn new(client: &ComputeClient<R>, num_leaf_slots: usize) -> Self { /* client.empty per field, ONCE */ }
}
```
**Deep-copy `operator=` (slot a → slot b):** two valid mappings — (a) **host-side index copy** (read the source slot scalar, write the dest slot) for Phase-14 correctness tests; (b) a tiny `#[cube]` "copy slot" kernel that, for each field buffer, does `buf[b] = buf[a]` — preferred for the future on-device argmax (§8) so the record never round-trips to host. Phase 14 only needs the layout + a working slot-copy (host or device); the §8 argmax consumer is Phase 17. **No per-split alloc anywhere (D-08).**

### Pattern 6: `launch_unchecked` safe-wrapper convention (CMP-01 / NRW-01)
**What:** `::launch_unchecked` is `unsafe`; the repo confines it in the host launcher behind a SAFETY comment that discharges the host-side V5 bounds proof.
```rust
// Source: VERIFIED histogram.rs:596-607, 900-911 (identical established convention)
// SAFETY: all input handles sized n / out sized out_len, each outliving the launch;
// every device index host-proven < len by V5 validation BEFORE upload; unsafe confined here (CMP-01).
unsafe {
    prefix_sum_kernel::launch_unchecked(client, count, dim, ArrayArg::from_raw_parts(h_in, n), ...);
}
```
Use plain `::launch` (checked) for the skeleton/percentile paths where perf is irrelevant; reserve `launch_unchecked` for the full-depth prefix-sum/reduction hot primitives. Both produce identical numerics.

### Anti-Patterns to Avoid
- **Cross-cube barrier.** There is no global barrier; never assume one cube sees another's writes mid-launch. Use the 3-kernel split (Pattern 3). `[VERIFIED: lib.rs:1263 checklist]`
- **`Atomic<i64>` or `.wrapping_add()` in `#[cube]`.** Broken / not an intrinsic. Use `Atomic<u64>` two's-complement + plain `+`/`*` (Pitfall 2). `[VERIFIED: lib.rs:1264-1265 + histogram.rs:1301-1311]`
- **`plane_sum` over a 256-bin span.** Exceeds plane width → wrong result. Use the LDS block-scan (Pattern 2). `[VERIFIED: lib.rs:1266]`
- **GPU-vs-GPU parity assertion.** Never compare two nondeterministic GPU f32 paths; anchor to the cpu f64 fold / C++ fixture (D-10, def-f8u-01). `[VERIFIED: memory def-f8u-01]`
- **Re-introducing a per-split device alloc** for categorical thresholds (the C++ `AllocateCatVectorsKernel` pattern). Pre-allocate the reserved slab (D-08).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Within-plane prefix sum | A `plane_shuffle_up` warp-scan loop | `plane_inclusive_sum`/`plane_exclusive_sum` | cubecl-0.10 lowers these to native scan ops; less code, fewer parity bugs. `[VERIFIED: plane.rs:242/275]` |
| Within-plane reduction | A `plane_shuffle_xor` butterfly | `plane_sum`/`plane_max`/`plane_min` | Same — native, already used on hip. `[VERIFIED: plane.rs:213/389/412]` |
| Device LCG wrap | A `wrapping_add` shim / i64 atomics | plain `u32` `*`/`+` (hardware two's-complement wrap) | `wrapping_add` is not a `#[cube]` intrinsic; native u32 arithmetic already wraps on GPU. `[VERIFIED: lib.rs:1265]` |
| i64 quantized accumulator | `Atomic<i64>` | `Atomic<u64>` storing i64 bits (`u64::cast_from(i64::cast_from(..))`) | `Atomic<i64>` is broken on cubecl-hip 0.10; the u64 two's-complement idiom is shipped + bit-exact. `[VERIFIED: histogram.rs:1301-1311]` |
| Device memory lifetime | Per-iteration alloc/free | `client.empty` once, reuse across launches | Hot-loop alloc is the bottleneck CubeCL pre-allocation eliminates (D-05/D-08). `[CITED: cubecl_manual 13]` |
| RNG golden | A new C++ RNG capture | Existing host `Random` draw sequence | D-04: host `Random` is already C++-bit-exact; it IS the oracle. `[VERIFIED: random.rs:44-77]` |

**Key insight:** The 0.10 plane intrinsics collapse most of the C++ `cuda_algorithms.cu` device-helper complexity into one-liners; the genuine porting work is the **block/global staging** (LDS + multi-kernel) and the **bitonic comparator/tie order**, not the warp primitives themselves.

## Runtime State Inventory

> This is an **additive greenfield** phase (new kernels + new structs behind an OFF-by-default flag), not a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string.
- **Stored data:** None — no datastore keys/IDs change.
- **Live service config:** None.
- **OS-registered state:** None.
- **Secrets/env vars:** One **new** read-only env var `LGBM_CUDA_ON_DEVICE` (already wired, `learner.rs:444`); stays OFF by default (D-11). No existing var renamed.
- **Build artifacts:** The off-tree `tools/cuda-fixtures/` harness produces committed fixture files (one-time); not part of the cargo build graph. No stale-artifact risk in the Rust crates.

## Common Pitfalls

### Pitfall 1: A 0.10 plane intrinsic may not lower on every backend
**What goes wrong:** `plane_inclusive_sum`/`plane_exclusive_sum`/`plane_max`/`plane_min` register IR ops, but a given backend (cubecl-cpu f64 anchor, or cubecl-hip) might not lower a specific op, producing a compile/runtime failure or a silently wrong result.
**Why it happens:** The frontend always exposes the op; backend coverage varies by version. The shipped histogram path only exercised `plane_sum`/`plane_any`/`plane_ballot`/`plane_shuffle`, NOT the scan/max/min intrinsics — so they are **unproven in this repo**.
**How to avoid:** Add a tiny smoke test per intrinsic per backend FIRST (cubecl-cpu, then hip). If one fails, fall back to a `plane_shuffle_up`-based manual scan (Pattern 1 alternative). This is the single highest-risk unknown — front-load it as a Wave-0 task.
**Warning signs:** kernel-compile panics mentioning an unsupported `Plane::*` op; reduction results equal to the input (op silently no-op'd).

### Pitfall 2: i64 quantized field via the wrong atomic/wrap
**What goes wrong:** Using `Atomic<i64>` or `.wrapping_add()` in `#[cube]` → broken codegen or a non-intrinsic error.
**Why it happens:** cubecl-hip 0.10 has no working `Atomic<i64>`; `wrapping_add` is a host method, not a device intrinsic.
**How to avoid:** Store i64 as u64 bits (`u64::cast_from(i64::cast_from(round(v*SCALE_F32)))`), accumulate with `Atomic<u64>::fetch_add` (two's-complement wrap is exactly i64 add), dequant with `(bits as i64) / SCALE_F64`. For the LCG use plain `u32` `*`/`+`. `[VERIFIED: histogram.rs:1301-1311, SCALE_F32=2^30]`
**Warning signs:** "method `wrapping_add` not found in cube scope"; i64 atomic add yielding garbage on hip.

### Pitfall 3: Expecting bit-exact f32 reductions across backends
**What goes wrong:** Asserting bit-exact equality between the cubecl-hip f32 reduction and the C++ f32 fixture, or between NVIDIA-32-lane and AMD-64-lane reductions.
**Why it happens:** Plane width (32 vs 64) changes the reduction tree → different f32 rounding; atomic/parallel order is non-deterministic.
**How to avoid:** Integer/permutation ops (prefix-sum on ints, argsort) → bit-exact. f64 ordered folds → bit-exact only if the **reduction order matches** (replicate the C++ pairwise/tree order in the cpu anchor; see Open Question 2). f32 GPU reductions → assert ~1e-6 only (D-03/D-10). `[VERIFIED: design §17 atomic-ordering; memory perf-gap / def-f8u-01]`
**Warning signs:** a reduction test green on cubecl-cpu but flaky on hip at the last ULP.

### Pitfall 4: f64 in a hot device kernel (forward-looking)
**What goes wrong:** Building a primitive on f64 device math that later sits in a CUDA hot loop tanks on consumer NVIDIA (1/32 f64 throughput) — spike-052 saw a 5.4× regression from an f64 mega-kernel.
**Why it happens:** Consumer NVIDIA f64 is heavily rate-limited.
**How to avoid:** The numeric **anchor** is f64 (cubecl-cpu) — fine, it's the CPU reference. The **device** primitives that will run in the Phase-21 grow loop (prefix-sum over partition indices, reductions over grad/hess) should stay f32/int + the u64 fixed-point trick, never f64. Percentile/argsort skeletons may use f64 (cold, Phase-19 consumer). `[VERIFIED: memory cuda-architectural-launch-bound / spike-052]`
**Warning signs:** an f64 `#[cube]` op in a path the §16 grow loop will call per-leaf.

### Pitfall 5: Bitonic tie-order divergence from C++
**What goes wrong:** Equal-key elements produce a different permutation than the C++ fixture → permutation bit-exact assertion fails even though the sort is "correct".
**Why it happens:** Bitonic comparator behavior on ties depends on the exact `>`-vs-`>=` and ascending-flag convention; any deviation reorders equal keys.
**How to avoid:** Mirror the C++ comparator exactly (`cuda_algorithms.cu` `BitonicArgCompareKernel`); use the same strict/non-strict comparison and ascending-segment parity. Capture the fixture with a tie-rich input to lock the convention.
**Warning signs:** permutation matches on distinct-key inputs but fails on inputs with duplicate keys.

### Pitfall 6: Breaking the Slice-0 no-op seam
**What goes wrong:** A refactor flips `on_device_growth_supported()` to true or makes `grow_tree_on_device` return `Some`, activating an unfinished path and failing the merge gate.
**Why it happens:** Touching `lib.rs:1239/1272` or the `GpuBackend<R>` override while "extending" the oracle.
**How to avoid:** Treat the discriminator + seam as frozen (D-09). Only ADD the primitives/structs/RNG + extend `oracle-harness` tests. The two Slice-0 tests (`learner_parity_on_device_seam_is_provable_noop_slice0`, `…_oracle_host_fallback_slice0`) are the guard. `[VERIFIED: learner_parity.rs:2451-2482]`

## Code Examples

### CUDARandom LCG (device) bit-identical to host `Random` (D-04)
```rust
// Source: VERIFIED host random.rs:44-77 (214013·x+2531011, RandInt16/32, NextFloat/32768f)
//         + AMD fork include/LightGBM/cuda/cuda_random.hpp:58-65 (same recurrence)
#[cube]
fn cuda_rand_advance(state: &mut u32) -> u32 {   // x = 214013*x + 2531011 (native u32 wrap)
    *state = *state * 214013u32 + 2531011u32;     // NOT wrapping_add — plain ops wrap on device
    *state
}
#[cube]
fn cuda_rand_int16(state: &mut u32) -> i32 { (((cuda_rand_advance(state)) >> 16) & 0x7FFFu32) as i32 }
#[cube]
fn cuda_rand_int32(state: &mut u32) -> i32 { ((cuda_rand_advance(state)) & 0x7FFFFFFFu32) as i32 }
#[cube]
fn cuda_next_float(state: &mut u32) -> f32 { (cuda_rand_int16(state) as f32) / 32768.0f32 }
// Host launcher: seed N tasks, draw K each, read back; compare to host Random::new(seed) sequence
// bit-for-bit (RandInt16/32 == i32 equality; NextFloat == f32 to_bits() equality).
```
**Parity assertion:** seed identical, draw `RandInt16`/`RandInt32` as exact `i32`, `NextFloat` via `f32::to_bits()` equality (exactly the host test at `random.rs:158-167`). This confirms the "plain `+`/`*` wraps on device" assumption end-to-end (the D-04 test IS the verification of Pitfall 2's wrap claim).

### C++/HIP fixture harness skeleton (D-03)
```cpp
// Source: VERIFIED AMD fork src/cuda/cuda_algorithms.cu (host wrappers are template-instantiated &
//         host-callable: ShufflePrefixSumGlobal<uint32_t>, ShuffleReduceSumGlobal<double,double>,
//         BitonicArgSortGlobal, GlobalInclusiveArgPrefixSum). Build with hipcc against the fork.
// For HOST wrappers: call directly, copy device→host, dump bytes.
// For pure __device__ block helpers (ShufflePrefixSum<T>, PercentileDevice, BitonicArgSort_1024):
//   add a one-line __global__ shim:
template <typename T>
__global__ void shim_ShufflePrefixSum(T* buf, int n) { /* call __device__ ShufflePrefixSum on buf */ }
// main(): for each primitive + a fixed seeded input → launch → hipMemcpy D2H → fwrite fixture .bin.
// Commit fixtures + the input generator seed so they are reproducible.
```
**Build:** `hipcc -I LightGBM-release-4.6.0.99/include -I LightGBM-release-4.6.0.99/src capture_primitives.cu LightGBM-release-4.6.0.99/src/cuda/cuda_algorithms.cu -o capture` (exact include set TBD at plan time). Runs on the local spoofed APU — adequate because these primitives are deterministic; f32 reductions captured here are compared at ~1e-6 (Pitfall 3), the bit-exact gate is the cpu f64 anchor.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hand-rolled `__shfl` warp scan (C++ literal) | `plane_inclusive_sum`/`plane_exclusive_sum` intrinsics | cubecl 0.10 | Less code; the within-plane level is a one-liner. `[VERIFIED: plane.rs]` |
| `Atomic<i64>` for quantized accum | `Atomic<u64>` two's-complement bits | shipped Phase 11 (spike-018) | Works on hip + bit-exact + deterministic. `[VERIFIED: histogram.rs]` |
| Per-split `cudaMalloc` for cat thresholds (C++) | Pre-allocated reserved SoA slab | this phase (D-08) | Eliminates the hot-loop alloc CubeCL has no clean analog for. |

**Deprecated/outdated:**
- **`wrapping_add`/`wrapping_mul` in `#[cube]`** — not device intrinsics; use plain ops (hardware wrap).
- **`Atomic<i64>` on cubecl-hip 0.10** — broken; do not use.
- **Single-kernel global scan** — impossible without a global barrier; use the 3-kernel split.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `plane_inclusive_sum`/`plane_exclusive_sum`/`plane_max`/`plane_min` lower correctly on BOTH cubecl-cpu and cubecl-hip 0.10 | Stack / Pattern 1 / Pitfall 1 | MEDIUM — if a backend doesn't lower one, fall back to a `plane_shuffle_up` manual scan; adds work but no scope change. **Mitigation: Wave-0 smoke test.** |
| A2 | Plain `u32` `*`/`+` in `#[cube]` wraps two's-complement on device (no trap), giving a bit-identical LCG stream | Pattern / Code Examples / Pitfall 2 | LOW — the D-04 parity test verifies this directly; if it fails, the only known device wrap path is via `Atomic<u64>` math, but non-atomic u32 wrap is standard GPU behavior. |
| A3 | Local `hipcc` build against the AMD fork yields goldens equivalent to the CUDA reference (deterministic primitives; f32 within ~1e-6) | Validation / Code Examples | LOW — argsort/int-scan are device-independent; f32 reductions covered by the ~1e-6 tol. Kaggle CUDA is the fallback if a primitive won't hipify. |
| A4 | `num_blocks ≤ 1024` so the global scan is exactly 3 launches this phase | Pattern 3 | LOW — Phase-14 tests use small inputs; the recursive case is a Phase-15 concern, noted. |
| A5 | The C++ f64 reduction order can be replicated in the cpu anchor for bit-exact f64 assertion | Pitfall 3 / Open Q2 | MEDIUM — if not, f64 reductions assert within a tight ULP band instead of bit-exact; argsort/int-prefix-sum remain bit-exact regardless. |
| A6 | A host-side leaf-slot copy is sufficient for the Phase-14 `operator=` test (device copy kernel deferred to the §8 consumer) | Pattern 5 | LOW — D-07 explicitly defers the readback packet/consumer to Phase 17/18. |

## Open Questions (RESOLVED)

> All three open questions are resolved by the Phase-14 plans. Resolutions are recorded inline below and carried into the owning plan's tasks + SUMMARY.

1. **Do `plane_inclusive_sum`/`plane_exclusive_sum`/`plane_max`/`plane_min` lower on cubecl-cpu AND cubecl-hip 0.10?**
   - What we know: the frontend exposes + registers IR for all of them (`plane.rs`); the repo has only exercised `plane_sum`/`plane_any`/`plane_ballot`/`plane_shuffle` on hip.
   - What's unclear: backend lowering coverage for the scan/max/min ops.
   - Recommendation: **Wave-0 smoke test per intrinsic per backend** before building the primitives; fall back to `plane_shuffle_up` manual scan if any op is missing.
   - **RESOLVED by 14-01** (Wave-0 `plane_intrinsic_smoke.rs` per-intrinsic-per-backend smoke test, with the `plane_shuffle_up` manual-scan fallback authored and recorded in 14-01-SUMMARY for 14-03 to consume).

2. **For f64 reductions/percentile, is the assertion bit-exact vs the C++ fixture, or bit-exact vs a Rust serial f64 ref + ~tol vs C++?**
   - What we know: D-03 says "numeric ops assert bit-exact on the cpu f64 anchor"; D-10 pins the anchor to the cubecl-cpu f64 fold. Bit-exactness between a GPU warp-tree f64 reduction and a serial CPU fold requires matching reduction order.
   - What's unclear: whether the C++ fixture's f64 reduction order is replicated in the Rust anchor, or whether f64 reductions get a tight-ULP tolerance.
   - Recommendation: integer prefix-sum + argsort permutation → bit-exact vs fixture; **f64 reductions/percentile → replicate the C++ pairwise order in the cpu anchor for bit-exact, else assert a documented f64 ULP band**. Decide at plan time per primitive.
   - **RESOLVED by 14-03** (the per-primitive f64 reduction-order policy is decided in-plan in 14-03 Task 2 — bit-exact-with-matched-order vs documented ULP band, recorded per reduction in code/14-03-SUMMARY).

3. **Exact `MAX_CAT_PER_SPLIT` reserved-slab width for `cat_threshold`/`cat_threshold_real`?**
   - What we know: D-06 reserves the buffers now; Phase 22 fills them.
   - What's unclear: the per-slot reserved length (C++ sizes it dynamically per split).
   - Recommendation: reserve a conservative fixed cap (e.g. matching `config_->max_cat_threshold`) and document it as Phase-22-tunable; the SoA layout doesn't change if the cap is revised, only the slab length.
   - **RESOLVED by 14-04** (the `MAX_CAT_PER_SPLIT` reserved width is defined as a documented Phase-22-tunable `usize` const in 14-04 Task 1, with the cap recorded in 14-04-SUMMARY).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cargo` | Rust build/test | ✓ | 1.95.0 | — |
| `cubecl` 0.10 (`cpu`) | f64 anchor runtime + all kernels | ✓ | 0.10.0 (vendored) | — |
| `cubecl` (`hip`/`rocm`) | ROCm parity track | ✓ | 0.10.0 + ROCm 7.1 | — |
| `hipcc` | D-03 fixture harness build | ✓ | 7.1.52802 (ROCm) | Kaggle CUDA (nvcc) |
| `cmake` | optional harness build driver | ✓ | 3.28.3 | plain `hipcc` invocation |
| AMD fork source (`cuda_algorithms.cu`, `cuda_random.hpp`, `cuda_split_info.hpp`) | fixture harness + signature reference | ✓ | `LightGBM-release-4.6.0.99/` (untracked, present) | mainline `LightGBM/` (same files) |
| `nvcc` (CUDA toolkit) | only if fixtures must come from real NVIDIA | ✗ | — | Kaggle CLI (`kaggle` 2.2.1 present, authenticated `boomvector`) |
| ROCm GPU (spoofed 8-CU APU) | run hip kernels / capture fixtures | ✓ | gfx1152 spoofed gfx1100 | cubecl-cpu for non-GPU primitive tests |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** `nvcc`/real-CUDA — fixtures captured locally via `hipcc` instead (deterministic primitives; A3). Kaggle is the escape hatch.

## Validation Architecture

> nyquist_validation is enabled. This section gates VALIDATION.md downstream.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test) + cubecl runtimes |
| Config file | none (workspace cargo) |
| Quick run command | `cargo test -p lgbm-compute primitives` and `cargo test -p lgbm-core random` |
| Full suite command | `cargo test --workspace` (merge gate: `raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites — D-11) |
| ROCm parity | `cargo test -p oracle-harness --features rocm` (hip primitive/seam tests) |

### Phase Requirements → Test Map
| Req ID | Behavior | Anchor | Assertion | Test Type | Automated Command | File Exists? |
|--------|----------|--------|-----------|-----------|-------------------|-------------|
| ODL-01 | block prefix-sum (incl/excl) | C++ fixture (`ShufflePrefixSum`) | int → bit-exact; f64 → bit-exact/ULP (Open Q2) | unit | `cargo test -p oracle-harness prefix_sum` | ❌ Wave 0 |
| ODL-01 | global 3-kernel prefix-sum | C++ fixture (`ShufflePrefixSumGlobal`) | int → bit-exact | unit | `cargo test -p oracle-harness prefix_sum_global` | ❌ Wave 0 |
| ODL-01 | reductions sum/max/min, dotprod | C++ fixture (`ShuffleReduce*Global`) | f64 bit-exact/ULP; f32 ~1e-6 (hip) | unit | `cargo test -p oracle-harness reduce` | ❌ Wave 0 |
| ODL-01 | single-block bitonic argsort | C++ fixture (`BitonicArgSort`) | **permutation bit-exact** (incl. tie input) | unit | `cargo test -p oracle-harness argsort` | ❌ Wave 0 |
| ODL-01 | percentile (skeleton, wtd/unwtd) | C++ fixture (`PercentileDevice`) | f64 bit-exact/ULP; f32 ~1e-6 | unit (skeleton) | `cargo test -p oracle-harness percentile` | ❌ Wave 0 |
| ODL-01 | multi-block argsort / items-sort (skeleton) | C++ fixture (`…Global`) | permutation bit-exact | unit (skeleton) | `cargo test -p oracle-harness argsort_global` | ❌ Wave 0 |
| ODL-02 | SoA split-record alloc + slot-copy | structural (round-trip) | field values preserved; slot a→b copy exact; no per-split alloc | unit | `cargo test -p lgbm-compute split_info` | ❌ Wave 0 |
| ODL-02 | CUDARandom LCG stream | host `Random` (D-04) | RandInt16/32 i32-exact; NextFloat `to_bits()`-exact | unit | `cargo test -p lgbm-compute cuda_random` | ❌ Wave 0 |
| D-09 | no-op seam unchanged | — | discriminator false; `Ok(None)`; Slice-0 tests green | regression | `cargo test -p oracle-harness slice0` | ✅ exists (`learner_parity.rs:2451`) |
| D-11 | merge gate byte-unchanged | C++ goldens | full suite green, default path unchanged | integration | `cargo test --workspace` | ✅ exists |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute <primitive>` (the single primitive touched).
- **Per wave merge:** `cargo test -p lgbm-compute && cargo test -p oracle-harness` (+ `--features rocm` for hip parity).
- **Phase gate:** `cargo test --workspace` green (D-11 merge gate) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tools/cuda-fixtures/capture_primitives.cu` + build script + committed `fixtures/*.bin` — the C++/HIP goldens (D-03). **Blocking** for every ODL-01 numeric test.
- [ ] `crates/oracle-harness/tests/primitive_parity.rs` — fixture loader + per-primitive parity tests.
- [ ] **Wave-0 spike (Open Q1):** smoke-test `plane_inclusive_sum`/`exclusive_sum`/`max`/`min` on cubecl-cpu + hip before authoring the primitives.
- [ ] `crates/lgbm-compute/src/kernels/primitives.rs`, `split_info.rs`, `random.rs` + `mod.rs` wiring.
- [ ] Tie-rich argsort fixture input (Pitfall 5).

## Security Domain

> `security_enforcement` enabled (ASVS L1). This is a numeric/compute foundation phase with no auth/session/network surface.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Host launchers validate slice lengths / bin ranges and return `ComputeError` (the V5 boundary) BEFORE any `launch_unchecked` — this is also the safety proof discharging the unsafe launch (CMP-01/NRW-01). |
| V6 Cryptography | yes (negative) | `CUDARandom`/`Random` are **non-cryptographic** deterministic PRNGs and MUST NEVER be used as a security RNG — already documented in `random.rs:8-14`. The device port inherits this constraint. |

### Known Threat Patterns for {cubecl device kernels}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device write via `launch_unchecked` | Tampering / DoS | Host-side V5 length/range validation proves every index in range before upload; unsafe confined to the launcher (CMP-01). `[VERIFIED: histogram.rs SAFETY blocks]` |
| Integer overflow in scratch sizing (`num_blocks * size`) | DoS | Compute sizes in `usize`, validate against input length at the V5 boundary. |
| Misuse of deterministic RNG as security entropy | Information disclosure | Module doc forbids it (V6 negative control); only the seeded constructor is ported. |

## Sources

### Primary (HIGH confidence)
- `~/.cargo/registry/.../cubecl-core-0.10.0/src/frontend/plane.rs` — VERIFIED plane intrinsics present (`plane_inclusive_sum`, `plane_exclusive_sum`, `plane_sum`, `plane_max`, `plane_min`, `plane_shuffle_up/down/xor`, `plane_ballot`, `PLANE_DIM` note).
- `crates/lgbm-compute/src/kernels/histogram.rs` — VERIFIED in-repo prior art: `SharedMemory`/`sync_cube` LDS pattern (792-835), u64 two's-complement fixed-point atomics (1265-1314), `launch_unchecked` safe-wrapper convention (596-607, 900-911), `plane_*` usage (435-558).
- `crates/lgbm-compute/src/lib.rs` — VERIFIED seam (`on_device_growth_supported` 1239, `grow_tree_on_device` 1272) + the cubecl-0.10 gotcha checklist (1262-1267).
- `crates/lgbm-core/src/random.rs` — VERIFIED host `Random` LCG (the D-04 oracle).
- `crates/oracle-harness/tests/learner_parity.rs` — VERIFIED Slice-0 no-op seam tests (2422-2482) + oracle helpers.
- `LightGBM-release-4.6.0.99/src/cuda/cuda_algorithms.cu` + `include/LightGBM/cuda/cuda_random.hpp` — VERIFIED C++ primitive/RNG signatures (host wrappers are template-instantiated + host-callable).
- `Cargo.toml` / `Cargo.lock` — VERIFIED cubecl 0.10.0 pin.

### Secondary (MEDIUM confidence)
- `docs/cuda-kernel-design.md` §2.4 / §15 / §17 — CITED primitive inventory, tunables, `CUDASplitInfo` fields, port considerations.
- `cubecl_manual/manual/cubecl/13_memory_preallocation.md` — CITED `client.empty`/`empty_tensor` pre-allocation pattern.
- `.claude/skills/spike-findings-lightgbm_rs/SKILL.md` + memory entries — CITED u64-atomic/`Atomic<i64>`-broken, plane-width, autotune, def-f8u-01, spike-052 f64-tanks findings.

### Tertiary (LOW confidence)
- Native u32 device-wrap behavior (A2) — ASSUMED standard GPU semantics; verified end-to-end by the D-04 parity test.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; cubecl 0.10 verified in lock + source.
- Architecture / gotcha resolutions: HIGH — each maps to verified in-repo prior art or verified crate source; the one MEDIUM is plane-intrinsic backend lowering (Open Q1, mitigated by Wave-0 smoke test).
- Pitfalls: HIGH — derived from shipped spike findings + the in-code gotcha checklist.
- Fixture harness: MEDIUM — hipcc + AMD-fork path verified present; exact include set + tie-order convention pinned at plan time.

**Research date:** 2026-06-29
**Valid until:** 2026-07-29 (stable — cubecl pin fixed; revisit if the workspace bumps cubecl off 0.10).
