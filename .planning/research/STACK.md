# Stack Research

**Domain:** On-device GPU decision-tree growth learner (CubeCL CUDA/ROCm), porting LightGBM's `CUDASingleGPUTreeLearner` into the existing pure-Rust `lgbm-rs` workspace (v1.1 GPU training-speed milestone).
**Researched:** 2026-06-28
**Confidence:** HIGH (cubecl version + capability API verified against `Cargo.lock` and the live `runtime::probe_capabilities`; reference architecture read directly from `LightGBM/src/treelearner/cuda/*`)

> NOTE: This file is the **v1.1 milestone** stack research (CUDA on-device tree learner). The prior v1.0 whole-project stack research is preserved in git history at this same path.

## TL;DR — the one architectural finding that governs everything

**The official `CUDASingleGPUTreeLearner` is NOT a persistent megakernel and uses NO device-side global barrier and NO device-driven growth loop.** It is **host-driven**, exactly like today's `lgbm-rs` learner — the C++ host still runs the per-leaf `for` loop. The difference is *granularity*, not *location*: each growth step dispatches a **handful of large kernels over whole-frontier device-resident state** (leaf-splits struct, data-index→leaf map, the histogram pool), and the host reads back **only a few scalars** (best feature/threshold/gain/default-left) to decide the next leaf. Phases are overlapped with **4 CUDA streams**. (Evidence: `cuda_single_gpu_tree_learner.cpp:34-90` Init; `cuda_data_partition.cu` uses `cuda_streams_[0..3]`, `<<<grid,block>>>` launches + two-stage `AggregateBlockOffsetKernel0/1` reductions; `cuda_leaf_splits.hpp` holds a device `CUDALeafSplitsStruct`.)

**Consequence for the stack: cubecl 0.10.0 — the version already pinned — is sufficient. No new crate and no new cubecl capability is required to match the reference architecture.** The milestone is *kernels + device-resident state extension + stop reading histograms back to host*, not a new compute primitive. The two genuine traps are numerical/capability, not architectural: (a) on CUDA `has_f64 == true` will tempt the existing f64 anchor kernel onto NVIDIA where f64 runs at 1/32 rate (spike-052: 5.4× slower) — the new path MUST stay on the u64 fixed-point build; (b) there is no grid barrier, so cross-leaf reductions are separate launches (which is what the reference does anyway).

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `cubecl` | **0.10.0** (pinned, `Cargo.lock`) | The compute seam; already provides every primitive the on-device learner needs | No upgrade required — `sync_cube`, `Atomic<u64/i64/f32>`, `SharedMemory`, plane ops, runtime-bound `for`/`while` loops, and the resident-`Handle` pattern are all in 0.10.0 and in production use in `kernels/histogram.rs`. Upgrading mid-milestone would churn the verified parity surface for no capability gain. |
| `cubecl-cuda` | 0.10.0 (`cubecl/cuda` → `CudaRuntime`) | The real-NVIDIA backend this milestone targets | Already wired as `CudaBackend = GpuBackend<CudaRuntime>` (lib.rs:2116). The generic `GpuBackend<R>` means CUDA inherits every kernel ROCm validates. |
| `cubecl-hip` | 0.10.0 (`cubecl/hip` → `RocmRuntime`) | The local parity-gate backend (spoofed 8-CU APU) | Hardware parity gate; bit-exact-to-anchor proofs run here before Kaggle CUDA confirms speed. |
| `cubecl-cpu` | 0.10.0 (default `cpu`) | The f64 **bit-exact deterministic anchor** — the hard merge gate | Unchanged. The on-device learner is held to ~1e-6 against this; do not touch it. |

### Supporting Libraries (already in tree — no additions)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl-hip-sys` | 7.1.5280200 | `rocm`-gated CU-count FFI for occupancy | ROCm only; irrelevant to CUDA (spike-053 refuted occupancy tuning on real NVIDIA). |
| `serde` | 1.x (`gpu`-gated) | `AutotuneKey` Serialize/Deserialize for the persistent autotune cache | Already present; spike-051/053 say **do not** add occupancy autotuning for CUDA. |
| `rayon` | 1.10 | Host-side parallelism (CPU anchor path) | Unchanged; not on the GPU growth path. |

**No new `[dependencies]` entry is needed.** Everything is a sub-feature of the already-vetted `cubecl 0.10.0` workspace dep (same rationale recorded at `Cargo.toml:43-46` for the `cuda` feature).

### CubeCL 0.10.0 device primitives — verified present and sufficient

All confirmed in active use in `crates/lgbm-compute/src/kernels/` against cubecl 0.10.0:

| Primitive | API (0.10.0) | Status | Role in the on-device learner |
|-----------|--------------|--------|-------------------------------|
| Intra-cube barrier | `sync_cube()` | ✅ `histogram.rs:814,826,1217…` | LDS histogram build; block-local prefix sums |
| Integer atomics | `Atomic<u64>`, `Atomic<i64>`, `::fetch_add` | ✅ in use | u64 fixed-point histogram build (the no-f64 path) |
| f32 atomics | `Atomic<f32>` + `AtomicUsage::Add` (probe-gated) | ✅ in use | f32 scatter build where supported |
| Shared memory (LDS) | `SharedMemory::<T>::new(CONST)` | ✅ `histogram.rs:801` | privatized sub-histograms; block prefix sums for partition |
| Plane/warp collectives | `plane_sum/ballot/any/broadcast/shuffle` | ✅ in use | warp-aggregated reductions; best-split argmax in a plane |
| **Runtime-bound loops** | `for i in 0..n {}`, `while i < n {}`, runtime `n` | ✅ `partition.rs:346`, `subtract.rs:48`, `histogram.rs:810` | **data-dependent loop bounds inside a kernel work** (answers Q1) |
| comptime specialization | `#[comptime] flag`, `#[cube(launch)]` / `launch_unchecked` | ✅ in use | feature-gated kernel variants |
| Capability probe | `client.features().supports_type(f64)`, `.features().plane.contains(Plane::Ops)`, `.properties().atomic_type_usage(...)`, `.properties().hardware.plane_size_max` | ✅ `runtime.rs:108-130` | backend dispatch (f64 vs u64-fixed; plane size) |
| Device-resident state | `cubecl::server::Handle` cached in `RefCell` across launches | ✅ `ResidentBins`, `resident_pool` (lib.rs:2009-2051) | **holds the whole growth-loop state on device** (answers Q4) |

## The five capability questions, answered head-on

### Q1 — Device-side dynamic control flow & data-dependent loop bounds: **FEASIBLE, already used**
cubecl 0.10.0 lowers ordinary Rust `for i in 0..n` / `while i < n` (runtime `n`) and runtime `if` into device control flow; the project already ships such kernels (`subtract.rs:48` `while i < n`, `histogram.rs:810` `while c < lds`, `partition.rs:346` `for i in 0..n`). `#[unroll]` is opt-in for comptime-known bounds; without it, bounds stay dynamic. **Verdict: data-dependent intra-kernel loops are fine.** Caveat: divergent data-dependent branches cost warp divergence (the project's "don't chase divergence" gate notes it's cleanly measurable but off the dominant path) — design frontier kernels branch-light, mirroring the reference's bit-vector partition.

### Q2 — Persistent / megakernel vs host launches: **host launches are correct, not a limitation**
cubecl 0.10.0 exposes **no persistent-kernel / cooperative-grid-launch API**, and you do **not** need one: the reference `CUDASingleGPUTreeLearner` is itself host-driven with per-step launches. The current ~8,570-launches/train problem is **not** "host drives the loop" — it is "host drives the loop **per-feature, per-leaf, with histogram read-back**". The fix is **fewer, whole-frontier launches over resident state**, not a megakernel. **Verdict: keep the host growth loop; collapse per-feature/per-leaf launches into whole-frontier kernels. Do NOT attempt a single device-resident megakernel** — unsupported in 0.10.0 and unnecessary.

### Q3 — No global barrier; scratch; atomics; inter-workgroup sync: **multiple-launches idiom (matches reference)**
Confirmed: cubecl 0.10.0 has `sync_cube()` (intra-cube only) and **no grid-wide barrier / cooperative-groups sync**. The idiom is the reference's: **device-resident state in `Handle`s persists across launches, and a kernel boundary IS the grid barrier.** Inter-workgroup reductions (leaf sums, prefix-sum offsets for the partition scatter) follow the reference — a block-level reduce kernel writes partials, a second small kernel combines them (`cuda_data_partition.cu` `AggregateBlockOffsetKernel0/1`). Device scratch = extra `Handle`s allocated once and reused. u32/u64 atomics: supported on CUDA and HIP. **Verdict: feasible; architecture is "resident state + a short DAG of launches per growth step."**

### Q4 — Can the resident-handle pattern hold the whole growth-loop state? **YES — already proven, just extend it**
The backend already caches device state across launches via interior-mutable `Handle`s: `ResidentBins` (binned dataset, uploaded once) and `resident_pool: RefCell<Vec<Option<Handle>>>` (per-leaf histogram pool mirror). The reference's on-device state is a small set of arrays — `CUDALeafSplitsStruct` (per-leaf sum_grad/sum_hess/data-start/count/best-split) + a `data_index_to_leaf_index` map + `data_indices` ordering + the histogram pool. **All of these are just more resident `Handle`s in the same `GpuBackend<R>` struct.** **Verdict: feasible with the existing pattern. The stack addition is a `ResidentGrowthState` (Handles for the data→leaf map, leaf-splits, data-index ordering) alongside `resident_bins`/`resident_pool`, populated once per tree and mutated in place by kernels.** Only tiny split-decision scalars are read back per step — use the idiomatic batched `client.read(vec![h])` (the memory note flags the per-handle N-read loop as a launch-bound anti-pattern).

### Q5 — cubecl-cuda vs cubecl-hip asymmetries that matter: **f64 is the big one**

| Axis | cubecl-hip (gfx1100, local) | cubecl-cuda (NVIDIA, Kaggle) | Action |
|------|------------------------------|------------------------------|--------|
| **f64** | `has_f64 == false` → already off the f64 kernel | **`has_f64 == true`** → `ReducePath::reduce_type()` would pick the f64 anchor kernel | **CRITICAL: the new on-device build must NOT use the `has_f64` f64 kernel on CUDA.** Consumer NVIDIA f64 = 1/32 f32; spike-052 measured the f64 fused kernel **5.4× slower** on real NVIDIA. Keep the **u64 fixed-point** build (spike-018). The `has_f64`-keyed `ReducePath` is a foot-gun — the CUDA learner path must select the integer build explicitly, not inherit the anchor's f64 reduce type. |
| **Plane/warp size** | wave32 (`plane_size_max == 32`) | warp = 32 | Symmetric at 32; already parameterized via `plane_size_max`. Don't hardcode. |
| **u64/i64 atomics** | supported | supported | Symmetric — fixed-point build path is portable. |
| **f32 atomics** | supported | supported | Symmetric. (WGSL/wgpu lacks them — out of scope, documented.) |
| **Streams / async overlap** | single default stream per client | reference uses 4 streams | **VERIFY at plan time** whether cubecl 0.10.0 exposes multiple streams; if not, phase overlap is unavailable and the win comes purely from launch-count reduction (still the dominant lever per spike-054). Treat multi-stream overlap as a *stretch*, not a dependency. |

## Installation

No new packages. Build the CUDA backend with the existing feature:
```bash
# local ROCm parity gate
cargo test -p oracle-harness -p lgbm-treelearner -p lgbm --features rocm
# CUDA backend compile (real-CUDA validation is via the Kaggle wheel, user boomvector)
cargo build -p lgbm-compute --features cuda
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Host-driven loop + whole-frontier resident kernels (mirror reference) | Single persistent megakernel with device-side growth loop | **Never on cubecl 0.10.0** — no cooperative-grid/persistent-kernel API, and no grid barrier. Revisit only if a future cubecl exposes cooperative launch AND profiling shows launch latency still dominates after frontier-batching. |
| u64 fixed-point on-device histogram build | f64 histogram build on CUDA | Never on consumer NVIDIA (1/32 f64). Only the cubecl-cpu anchor uses f64. |
| Extend `GpuBackend<R>` resident state + one coarse `grow_one_tree_on_device` method | A separate `CudaOnDeviceLearner` bypassing the `Backend` trait | If the fine-grained per-leaf `Backend` ops become a straitjacket. Preferred: add a coarse per-tree trait method with a default impl that falls back to today's per-leaf orchestration (keeps CPU/ROCm byte-unchanged), and a `GpuBackend<R>` override that runs the resident frontier loop. |
| Bit-vector partition + block-prefix-sum scatter (reference design) | The current host-routed `prefers_host_partition` path | Keep host-routed partition for **ROCm** (spike-035: shared-DDR5 APU makes the device round-trip pure overhead). For **CUDA discrete PCIe**, on-device partition is the milestone's point (eliminates the host round-trip the launch-bound analysis blames). |

## What NOT to Use / NOT to Attempt

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| f64 hot loops in new CUDA kernels | 1/32 f32 rate on consumer NVIDIA; spike-052 = 5.4× regression | u64 fixed-point accumulation (spike-018); convert once at the end |
| Inheriting the `has_f64`-keyed `ReducePath` on CudaBackend for the new build | `has_f64 == true` on CUDA silently routes to the slow f64 kernel | Explicitly select the integer build for the on-device CUDA path |
| A persistent/cooperative megakernel | Unsupported in cubecl 0.10.0; reference doesn't use one | Few large host-launched kernels over resident state |
| Assuming a grid-wide barrier exists | cubecl 0.10.0 has only `sync_cube()` (intra-cube) | Kernel boundary = grid barrier; two-stage reduce kernels for cross-block sums |
| Build-occupancy / row-partition `P` autotuning for CUDA | spike-053 refuted it on real NVIDIA (P=1 optimal) | Leave autotune ROCm-only |
| `read`-per-handle loops in the readback | Memory note: N-read loop masks a launch-bound win | `client.read(vec![handles])` batched readback of the tiny split scalars |
| `plane_match_any` | Codebase's own note: **absent in cubecl 0.10.0** (`histogram.rs:407`) | Manual `plane_ballot` + `plane_shuffle` leader-election loop (the shipped idiom) |
| Reading histograms back to host per leaf | This (not "host loop") is the source of the ~8,570 round-trips | Keep histograms resident (`resident_pool` already does for ROCm); extend to the CUDA frontier build |

## Stack Patterns by Variant

**If matching the reference architecture (recommended):**
- Host runs the leaf-wise best-first loop; per step it launches: (1) whole-frontier histogram build into the resident pool (u64 fixed-point), (2) histogram-subtract for the larger child, (3) frontier best-split argmax kernel, (4) bit-vector partition + block-prefix-sum + scatter updating the resident data→leaf map. Read back only the winning split scalars.
- Because cubecl has no grid barrier and no persistent kernel, and because this is exactly what the launch-bound analysis (spikes 051–054) prescribes.

**If multi-stream overlap is exposed in cubecl 0.10.0:**
- Overlap partition (stream A) with the next leaf's build prep (stream B), as the reference does with `cuda_streams_[0..3]`.
- Treat as stretch — the proven win is launch-count reduction, which is stream-independent.

**If the per-leaf `Backend` trait becomes a straitjacket:**
- Add a single coarse `grow_one_tree_on_device(...)` trait method, default impl = existing per-leaf orchestration (CPU anchor + ROCm host-routed paths byte-unchanged), `GpuBackend<R>` override = resident frontier loop. Mirrors the existing "default impl = byte-unchanged CPU path, GPU overrides" seam discipline.

## Version Compatibility

| Component | Pinned | Notes |
|-----------|--------|-------|
| `cubecl` + all sub-crates (`-cuda/-hip/-cpu/-core/-runtime`) | 0.10.0 (lockstep) | All cubecl crates move together; do not bump one. |
| `cubecl-hip-sys` | 7.1.5280200 | ROCm-only; transitive + promoted optional. Irrelevant to CUDA. |
| Rust edition | 2024 | Workspace-wide; unchanged. |

## Open items to verify at plan time (do not assume)

1. **Multi-stream support in cubecl 0.10.0** — the reference uses 4 streams; if cubecl exposes only one default stream per client, phase-overlap is off the table (acceptable; launch-count reduction is the real lever). Verify against `cubecl::server`/client API before scoping overlap work.
2. **Idiomatic readback of small scalar sets** — confirm `client.read(vec![h])` vs `read_one_unchecked` semantics for the per-step split-decision struct (memory note flags the N-read-loop anti-pattern; verify the batched form on cubecl-cuda).
3. **Resident `Handle` in-place mutation/aliasing** — confirm a kernel can take the same `Handle` as both input and output (in-place data→leaf map update) under cubecl 0.10.0 aliasing rules, or whether double-buffering (ping-pong Handles) is required (the reference double-buffers the index map).
4. **Per-leaf `Backend` trait vs a coarse per-tree method** — planner design decision; the coarse method is lower-risk for keeping the CPU bit-exact anchor untouched.

## Sources

- `Cargo.lock` (lines 649-810) — cubecl 0.10.0 + all sub-crate versions, pinned — HIGH
- `crates/lgbm-compute/src/lib.rs` (Backend trait + `GpuBackend<R>` resident state, lines 486-2300) — existing resident-Handle pattern and trait-default seam — HIGH
- `crates/lgbm-compute/src/kernels/histogram.rs` + `runtime.rs:108-130` — verified cubecl 0.10.0 device primitives (`sync_cube`, atomics, plane ops, runtime loops) and the capability-probe API — HIGH
- `LightGBM/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp`/`.cu`, `cuda_data_partition.cu`, `cuda_leaf_splits.hpp` — reference on-device architecture: host-driven, few large kernels, device-resident leaf-splits struct, 4 streams, two-stage cross-block reduce — HIGH
- `.claude/skills/spike-findings-lightgbm_rs/references/cuda-architectural-launch-bound.md` (spikes 051–054, real-NVIDIA Kaggle) — launch-bound mechanism, f64-on-NVIDIA 5.4× penalty, occupancy refuted, on-device learner is the one lever — HIGH
- MEMORY: GPU-is-spoofed-8CU-APU; spike-035 ROCm host-partition; `client.read(vec![h])` readback idiom — MEDIUM/HIGH

---
*Stack research for: CUDA on-device tree learner (v1.1 GPU training-speed milestone)*
*Researched: 2026-06-28*
