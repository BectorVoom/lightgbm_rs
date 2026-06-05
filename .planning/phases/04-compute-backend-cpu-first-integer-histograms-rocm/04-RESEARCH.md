# Phase 4: Compute Backend (CPU-first f32 histograms → ROCm) - Research

**Researched:** 2026-06-05
**Domain:** GPU/compute-kernel authoring with CubeCL 0.10.0 (cubecl-cpu + cubecl-hip/ROCm); faithful 1:1 port of LightGBM's histogram-construction / best-split / data-partition hotpath
**Confidence:** HIGH (CubeCL API verified against vendored 0.10.0 source in the local cargo registry; C++ behavior read directly from the pinned reference tree)

## Summary

Phase 4 fills the existing kernel-free `lgbm-compute::Backend` skeleton with three whole-kernel operations — `construct_histograms`, `find_best_split` (gain math inside the kernel), and `data_partition` — authored in CubeCL `#[cube]` kernels that run on the **cubecl-cpu** reference path (deterministic anchor, bit-exact vs a committed C++-transcription golden) and the **cubecl-hip** ROCm path (matches cubecl-cpu within ~1e-6). CubeCL 0.10.0 is the latest stable (published 2026-05-07, ~1 month old), so the alpha-churn-containment mandate (CMP-01) is well founded — and the entire API surface was verified against the vendored source rather than training data.

The single most consequential finding is a **direct conflict with the D-04 bit-exact-on-cubecl-cpu bet that must be validated empirically before the kernel suite is built**: the cubecl-cpu runtime (`runner.rs::execute_data`) dispatches **each unit of a cube onto a separate OS worker thread that runs concurrently** — it is NOT a single-threaded sequential executor. Bit-determinism is therefore only achievable if each histogram bin / each reduction is summed by exactly **one** unit in a fixed order (e.g. `CubeDim::new_1d(1)`, or one designated unit performing the ordered fold after `sync_cube`). Any design that accumulates a shared histogram cell from multiple units — especially via atomics — will be order-nondeterministic and will not hit bit-exact. This maps exactly onto the C++ `deterministic=true num_threads=1` reference path (`leaf_splits.hpp:180`: the OMP reduction is disabled under `deterministic_`). The second consequential finding is the **capability matrix**: `Plane::Ops` is present on cubecl-hip but ABSENT on cubecl-cpu (plane_size=1); f32 atomic-add is registered on cubecl-hip but NOT on cubecl-cpu; f64 is present on cubecl-cpu but disabled on cubecl-hip. So CMP-04's capability gate is real and load-bearing, and the deterministic sequential fallback is the default path on cubecl-cpu.

The third finding fixes a numeric-contract subtlety: although gradients/hessians are `score_t = float` (f32), LightGBM accumulates histograms in **`hist_t = double` (f64)** with an interleaved `[grad, hess]` stride-2 layout. The kernels must accumulate in f64 to match — "standard f32 accumulations" (D-03) means f32 inputs, f64 histogram cells, exactly as C++.

**Primary recommendation:** Build the cubecl-cpu kernels as single-reduction-owner (one unit folds each histogram cell / each scan in C++ bin order), accumulate in f64, transcribe the gain math from `feature_histogram.hpp:711-845` verbatim, and run a **Wave-0 spike** that proves cubecl-cpu produces bit-identical f64 histogram + f32 leaf-output across N repeated launches BEFORE committing to the full suite. If the spike fails, fall back to the D-04a plan (relax cubecl-cpu anchor to ~1e-6).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Compute / Tree-Learner Boundary (CMP-05 / Phase 4↔5 cut)**
- **D-01:** Whole-kernel ops. `lgbm-compute`'s `Backend` trait exposes coarse, complete operations matching the CMP-05 wording — `construct_histograms`, `find_best_split` (the split-gain math lives **inside** the kernel), and `data_partition`. The Phase-5 learner orchestrates these (growth, leaf-splits, constraints, subtraction-trick bookkeeping) but does not re-implement the per-bin gain scan.
- **D-01a — boundary note:** Because `find_best_split` carries the gain formula, Phase 4 effectively implements the math of Phase-5 TRL-04 early (`ThresholdL1`, `GetSplitGains`/`GetLeafGain`, `kEpsilon`/`2*kEpsilon` positions, `lambda_l1`/`lambda_l2`/`min_gain_to_split`/`min_sum_hessian_in_leaf`/`min_data_in_leaf`/`max_delta_step`/`path_smooth`, tie-breaking, `SKIP_DEFAULT_BIN`/missing routing). Research must define exactly which gain parameters flow into the kernel and confirm which `feature_histogram.hpp` routines move into the kernel vs stay in the Phase-5 learner. TRL-04 in Phase 5 then consumes this kernel rather than re-deriving gains.

**Kernel Golden / Validation Strategy (ORA-04, no learner yet)**
- **D-02:** Header-only C++ transcription for kernel goldens. Extend the `xtask` capture harness with a histogram/split/partition capture subcommand that verbatim-transcribes the C++ routines (`feature_histogram.hpp`, `dense_bin.hpp`/`sparse_bin.hpp`, the `ConstructHistograms`/`FindBestSplitsFromHistograms`/data-partition logic) header-only, emits goldens over synthetic bin + grad/hess inputs, and commits them. Human-approved, numerically identical to `lib_lightgbm`, replayable with no C++ toolchain at normal test time.
- **D-02a:** Synthetic inputs should exercise every kernel path the contract cares about: dense + sparse bin layouts, the most-frequent-bin / default-bin skip, missing/zero routing, multiple bit widths where they affect accumulation, and grad/hess sign/magnitude spread that stresses the f32 reduction. Reuse the Phase-2 binned-store forms where possible.

**ROCm Gating Posture (CMP-03 / SC#3 / SC#5)**
- **D-03:** CPU-solid now, ROCm best-effort. Make the cubecl-cpu path rock-solid and fully oracle-gated this phase (hard gate). Bring up ROCm and run the oracle on the local ROCm GPU, but if CubeCL-alpha or hardware capability gaps block ~1e-6 parity, record them as known issues (with specifics) rather than blocking phase completion.
- **D-03a:** ORA-04's literal "oracle passes on ROCm" remains the target; this decision sets the completion bar as "CPU gate green + ROCm executed with gaps documented." Surface any ROCm gap explicitly in verification (no silent pass).

**Determinism & Validation Anchor (CMP-02 / ~1e-6 contract)**
- **D-04:** cubecl-cpu IS the deterministic anchor — one impl per kernel. The single-threaded cubecl-cpu kernel must reproduce the committed C++-transcription golden bit-exact; cubecl-hip then matches cubecl-cpu within ~1e-6. No separate scalar-Rust reference is maintained — exactly one Rust implementation of each kernel.
- **D-04a — research/planning watch:** This assumes the cubecl-cpu runtime is bit-deterministic single-threaded. If bring-up shows cubecl-cpu cannot be made bit-stable against the f32 golden, the fallback is to relax the cubecl-cpu anchor to ~1e-6 against the golden. Flag this empirically early, before building the full kernel suite on the bit-exact assumption.

**Carried Forward (locked by prior phases — not re-litigated)**
- Faithful C++ mirror discipline (which child is constructed vs subtracted, default-bin skip, `kEpsilon` placement), not idiomatic redesigns. Do not "improve" the subtraction trick or reduction order.
- f32 end-to-end, ~1e-6 absolute, standard f32 accumulations; integer-quantized histograms dropped.
- `lgbm-compute` is the single CubeCL seam (CMP-01); no crate above it names a CubeCL runtime.
- CPU/ROCm are separate oracle gates; committed-golden + idempotent-regen + header-only-transcription-fallback discipline.
- Single-threaded deterministic core matching the pinned `deterministic=true force_row_wise=true num_threads=1` reference, with per-row/per-feature independence as the parallel-ready seam.

### Claude's Discretion
- The exact `Backend` trait method signatures and the `Runtime` associated-type binding; the kernel buffer/launch/allocation API shape; the `Plane`-API capability-gating mechanism and the deterministic sequential-fallback structure (bounded by CMP-04 + SC#4); the precise gain-config parameter struct passed into `find_best_split` (bounded by D-01a); the synthetic-input fixture format and the histogram/split/partition golden serialization (bounded by the oracle-harness comparator seam); the cubecl-cpu vs cubecl-hip feature-flag / runtime-selection mechanism (bounded by CMP-03 "Cargo feature and/or runtime config"). When C++ behavior is the spec, the C++ source is authoritative over any inferred default.

### Deferred Ideas (OUT OF SCOPE)
- Tree-learner orchestration (leaf-wise growth, leaf-splits, `num_leaves`/`max_depth`, subtraction-trick bookkeeping, monotone/interaction constraints, feature subsampling, `force_row_wise`/`force_col_wise` selection) — Phase 5.
- GBDT spine, objectives, metrics — Phase 6; DART/RF/GOSS — Phase 7.
- f32 transcendental (exp/log/pow/sigmoid) CPU↔ROCm parity — primarily Phase 6; note any early ROCm signal for Phase 6.
- Parallel (rayon) CPU histogram path — later, separately-validated optimization that must still match the deterministic anchor.
- Integer-quantized / discretized histograms (QNT-01) and linear-tree kernels (LIN-01) — v2.
- A residual ROCm oracle gap (if CubeCL alpha / hardware blocks ~1e-6) — tracked as a Phase-4 follow-up per D-03a.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CMP-01 | `lgbm-compute` backend trait isolating all device ops behind one crate (contains CubeCL alpha churn) | Existing `Backend { type Runtime; }` skeleton binds `Runtime: cubecl::Runtime`; kernels authored with `#[cube]` and launched via generated `::launch::<R>`; ALL cubecl type names confined to this crate. Verified: cubecl re-exports `Runtime`, `ComputeClient`, `CubeDim`, `CubeCount` from cubecl-core/cubecl-runtime. |
| CMP-02 | CPU backend (cubecl-cpu) as deterministic reference execution path | `cubecl_cpu::CpuRuntime` + `CpuDevice`; `CpuRuntime::client(&CpuDevice)`. Determinism caveat: multi-worker execution model — see Pitfall 1. f64 supported; Plane::Ops + f32-atomics NOT supported (sequential fold required anyway). |
| CMP-03 | ROCm/HIP backend (cubecl-hip) selectable via Cargo feature and/or runtime config | `cubecl_hip::HipRuntime` + `AmdDevice { index }`; gated behind a `rocm`/`hip` Cargo feature on lgbm-compute that enables `cubecl/hip`. Local GPU is gfx1100 (RDNA3, wave32). A CPU-only build must NOT enable the hip feature (SC#1). |
| CMP-04 | CUDA warp-level ops mapped onto CubeCL `Plane` API with capability gating + sequential fallback | `plane_sum`/`plane_inclusive_sum`/`plane_exclusive_sum`/`plane_max`/`plane_broadcast`/`plane_shuffle_*` verified in `cubecl-core/frontend/plane.rs`. Capability gate: `client.features().plane.contains(Plane::Ops)` (from `cubecl_ir::features::Plane`). Gate also covers f64 (`client.features().supports_type(...)`) and atomics (`client.properties().atomic_type_usage(ty).contains(AtomicUsage::Add)`). |
| CMP-05 | GPU-resident histogram construction, best-split finding, data partition kernels meeting ~1e-6 (f32) | Three `#[cube]` kernels; inputs = Phase-2 binned store + f32 ordered grad/hess; histogram cells = f64 (`hist_t`). Gain math transcribed from `feature_histogram.hpp:711-845`. Split scan from `feature_histogram.hpp:845-1000`. |
| ORA-04 | Oracle suite executes and passes on the ROCm backend | Header-only C++-transcription goldens committed to `tests/fixtures/`; cubecl-cpu compared bit-exact (`compare_exact_f64_bits`), cubecl-hip compared `compare_within(ORACLE_TOL=1e-6)`. CPU = hard gate, ROCm = bring-up + documented gaps (D-03a). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Histogram accumulation (grad/hess sum per bin) | Compute kernel (`lgbm-compute`) | Phase-2 binned store (input) | Whole-kernel op D-01; per-bin f64 accumulation is the GPU-relevant inner loop |
| Best-split finding (gain scan + threshold) | Compute kernel (`lgbm-compute`) | lgbm-core Config (gain params) | D-01: gain math lives inside the kernel; consumes Config gain surface |
| Data partition (row→leaf routing) | Compute kernel (`lgbm-compute`) | — | D-01: `data_partition` is a backend op; partition state lives in Phase-5 learner |
| Backend trait + Runtime binding | `lgbm-compute` (the ONLY cubecl seam) | — | CMP-01 containment boundary |
| Capability gate (Plane::Ops / f64 / atomics) | `lgbm-compute` (startup query) | — | CMP-04; queried once via `client.features()`/`client.properties()` |
| Runtime selection (cpu vs hip) | `lgbm-compute` (Cargo feature + runtime config) | — | CMP-03; downstream crates never name a runtime |
| Tree growth / leaf-splits / subtraction-trick orchestration | Phase-5 learner (NOT this phase) | calls `Backend` ops | D-01 boundary; out of scope here |
| Golden capture (C++ transcription) | `xtask` (capture-time only) | committed `tests/fixtures/` | D-02; no C++ toolchain at test time |
| Oracle comparison | `oracle-harness` | — | bit-exact (cpu) vs ~1e-6 (hip) split |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 (pinned) | Compute-kernel authoring + CPU/ROCm runtimes | Project mandate (CLAUDE.md); already in workspace `Cargo.toml` + `Cargo.lock` |
| `cubecl-cpu` (via `cubecl/cpu`) | 0.10.0 | Deterministic CPU reference runtime (`CpuRuntime`) | MLIR/LLVM-compiled CPU backend; the D-04 anchor |
| `cubecl-hip` (via `cubecl/hip`) | 0.10.0 | ROCm/HIP runtime (`HipRuntime`) | Targets the mandated local ROCm GPU (gfx1100) |
| `thiserror` | 2.0.18 | Structured `ComputeError` at the crate boundary | FND-04 idiom carried from prior phases |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `lgbm-core` | workspace | `Config` gain params, f32 types, `Random` | Gain-config surface into `find_best_split` |
| `lgbm-dataset` | workspace | Binned columnar store (`Bin` trait, `FeatureGroup`) | Histogram-kernel input — do NOT re-bin |
| `oracle-harness` | workspace | `compare_exact_f64_bits` (anchor) + `compare_within(ORACLE_TOL)` (ROCm) | Golden comparison seam |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cubecl-cpu as anchor | Separate scalar-Rust reference | D-04 explicitly rejects this — one impl per kernel. Only revisit if Wave-0 spike fails (D-04a). |
| f32 atomic-add accumulation | Single-owner ordered fold | Atomics are nondeterministic (order) AND unsupported on cubecl-cpu — must use ordered fold for the anchor regardless. |
| `Plane` ops on CPU path | Sequential fallback | `Plane::Ops` is absent on cubecl-cpu (plane_size=1) — the fallback IS the CPU path, not an option. |

**Installation:** No new dependencies. `lgbm-compute/Cargo.toml` already declares `cubecl.workspace = true`. Add the runtime features:
```toml
# crates/lgbm-compute/Cargo.toml
[dependencies]
cubecl = { workspace = true, features = ["cpu"] }
lgbm-core = { path = "../lgbm-core" }
lgbm-dataset = { path = "../lgbm-dataset" }
thiserror = { workspace = true }

[features]
default = ["cpu"]
cpu = []                       # cubecl/cpu always on for the reference path
rocm = ["cubecl/hip"]          # opt-in; a CPU-only build omits this (SC#1)
```
*(Exact feature names are Claude's discretion per CONTEXT.md; `cubecl/cpu` and `cubecl/hip` are the verified upstream feature names.)*

**Version verification:**
```
cubecl = 0.10.0  [VERIFIED: crates.io API — latest stable, published 2026-05-07]
cubecl-cpu 0.10.0, cubecl-hip 0.10.0, cubecl-runtime 0.10.0  [VERIFIED: Cargo.lock checksums present]
```

## Package Legitimacy Audit

> All packages are already vendored in the workspace `Cargo.lock` with checksums; no new external package is introduced this phase. slopcheck is a Python tool and does not apply to crates.io; the relevant verification is registry-presence + source-repo provenance.

| Package | Registry | Age | Source Repo | Verification | Disposition |
|---------|----------|-----|-------------|--------------|-------------|
| cubecl | crates.io | 0.10.0 published 2026-05-07 (latest stable) | github.com/tracel-ai/cubecl (Burn/Tracel team) | In Cargo.lock w/ checksum `fd203fef…`; crates.io confirms version | Approved (pinned) |
| cubecl-cpu | crates.io | 0.10.0 | github.com/tracel-ai/cubecl | Cargo.lock checksum `f572143f…` | Approved |
| cubecl-hip | crates.io | 0.10.0 | github.com/tracel-ai/cubecl | Cargo.lock checksum `3c6b510a…` | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none — cubecl is the project-mandated framework, already pinned, from a well-known maintainer (Tracel AI / Burn).

## Architecture Patterns

### System Architecture Diagram

```
                Phase-5 learner (NOT this phase)
                        │ calls Backend ops (never names cubecl)
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  lgbm-compute  (the ONLY cubecl seam — CMP-01)                    │
│                                                                    │
│  Backend trait                                                     │
│   type Runtime: cubecl::Runtime                                    │
│   fn construct_histograms(binned, ordered_grad, ordered_hess)→hist │
│   fn find_best_split(hist, gain_cfg, ...)→SplitInfo                │
│   fn data_partition(binned, threshold, ...)→left/right indices     │
│                                                                    │
│  ┌───────────────┐   capability gate (startup, once):             │
│  │ runtime select│   client.features().plane.contains(Plane::Ops)  │
│  │  cpu | rocm    │   client.features().supports_type(f64)          │
│  └──────┬─────────┘   client.properties().atomic_type_usage(...)    │
│         │                                                          │
│   ┌─────┴──────────────────────────┐                              │
│   ▼ cubecl-cpu (anchor)            ▼ cubecl-hip (ROCm, gfx1100)    │
│   plane_size=1                     plane_size=32 (wave32)          │
│   Plane::Ops: NO                   Plane::Ops: YES                 │
│   f64: YES                         f64: NO (disabled upstream)     │
│   f32 atomic-add: NO               f32 atomic-add: YES             │
│   → single-owner ordered fold      → Plane reduction OR ordered    │
│   → BIT-EXACT vs golden            → within ~1e-6 of cpu           │
└───────────────────────────────────────────────────────────────────┘
         │ inputs                                  │ goldens
         ▼                                         ▼
  lgbm-dataset (binned store)            oracle-harness ← tests/fixtures/
  lgbm-core (Config gain params)         (C++-transcription, committed)
                                                  ▲
                                          xtask kernel-capture (capture-time only)
```

Data flow for one `construct_histograms` call: binned feature column (Phase-2 store) + per-row f32 `ordered_gradients`/`ordered_hessians` → kernel reads `bin = data(row)`, accumulates `hist[bin<<1] += grad` (f64), `hist[(bin<<1)+1] += hess` (f64) → output f64 histogram `[g0,h0,g1,h1,…]`. Then `find_best_split` scans that histogram applying the gain formula → `SplitInfo`. Then `data_partition` routes rows to left/right by threshold.

### Recommended Project Structure
```
crates/lgbm-compute/src/
├── lib.rs              # Backend trait (extend existing skeleton) + ComputeError
├── runtime.rs          # cpu/rocm runtime selection + capability gate
├── kernels/
│   ├── histogram.rs    # #[cube] construct_histograms (f64 accumulation)
│   ├── split.rs        # #[cube] find_best_split (gain math, scan)
│   └── partition.rs    # #[cube] data_partition (row→leaf routing)
├── gain.rs             # ThresholdL1/GetLeafGain/GetSplitGains/CalculateSplittedLeafOutput (host-comptime helpers mirrored into the kernel)
└── error.rs            # thiserror ComputeError

xtask/
├── src/main.rs         # add "kernel-capture" subcommand
└── cpp/
    └── kernel_capture.cpp   # header-only transcription of ConstructHistogram + FindBestThreshold + Split

crates/oracle-harness/tests/
└── kernel_parity.rs    # replay committed histogram/split/partition goldens
```

### Pattern 1: `#[cube]` kernel authoring + launch
**What:** A kernel is a `#[cube(launch)]` fn; the macro generates `::launch::<R>(&client, cube_count, cube_dim, args…)`.
**When to use:** Every backend op.
**Example:**
```rust
// Source: cubecl-core-0.10.0/src/runtime_tests/atomic.rs:8-58 (verified shape)
use cubecl::prelude::*;

#[cube(launch)]
fn construct_hist_kernel(
    binned: &Array<u32>,           // per-row bin index
    grad: &Array<f32>,             // ordered gradients (score_t = f32)
    hess: &Array<f32>,             // ordered hessians
    out: &mut Array<f32>,          // f64 in practice — see Pitfall 3
) {
    // single-owner ordered fold for the deterministic anchor — see Pitfall 1
    if UNIT_POS == 0 {
        for i in 0..binned.len() {
            let ti = binned[i] * 2;
            out[ti] += grad[i];        // grad cell
            out[ti + 1] += hess[i];    // hess cell
        }
    }
}

// host launch
let client = R::client(&Default::default());
let h_bin  = client.create(u32::as_bytes(&bins));
let h_grad = client.create(f32::as_bytes(&grads));
let h_out  = client.empty(2 * num_bins * size_of::<f64>());
unsafe {
    construct_hist_kernel::launch::<R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),                       // single unit → ordered fold
        ArrayArg::from_raw_parts(h_bin, n),
        ArrayArg::from_raw_parts(h_grad, n),
        ArrayArg::from_raw_parts(h_out, 2 * num_bins),
    );
}
let bytes = client.read_one_unchecked(h_out);   // read results back
```
*(API names `client.create`/`empty`/`read_one_unchecked`, `CubeCount::new_single`/`new_1d`, `CubeDim::new_1d`, `ArrayArg::from_raw_parts` all VERIFIED in vendored 0.10.0 source.)*

### Pattern 2: Capability gate at startup (CMP-04)
**What:** Query the runtime's `Features`/`DeviceProperties` once when the backend is created; select Plane-reduction vs sequential-fold path.
**Example:**
```rust
// Source: cubecl-core-0.10.0/src/runtime_tests/plane.rs:568, atomic.rs:20-23 (verified)
use cubecl::ir::features::{Plane, AtomicUsage};

let has_plane  = client.features().plane.contains(Plane::Ops);
let has_f64    = client.features().supports_type(/* f64 storage type */);
let atomic_ty  = Type::new(StorageType::Atomic(f32_elem)).with_vector_size(1);
let has_f32_atomic = client.properties().atomic_type_usage(atomic_ty).contains(AtomicUsage::Add);

let reduce = if has_plane { ReducePath::Plane } else { ReducePath::Sequential };
```
On cubecl-cpu: `has_plane=false`, `has_f64=true`, `has_f32_atomic=false` → Sequential.
On cubecl-hip (gfx1100): `has_plane=true`, `has_f64=false`, `has_f32_atomic=true`.

### Pattern 3: Whole-kernel gain math (D-01)
**What:** `find_best_split` carries `ThresholdL1`/`GetLeafGain`/`GetSplitGains`/`CalculateSplittedLeafOutput` inside the kernel, transcribed verbatim from `feature_histogram.hpp:711-845`. Gain config flows in as a comptime/scalar struct (see Gain-Config Surface table below).

### Anti-Patterns to Avoid
- **Histogram accumulation via atomics for the anchor:** nondeterministic reduction order AND unsupported on cubecl-cpu. Use a single-owner ordered fold for the cubecl-cpu anchor.
- **Multi-unit reduction without a fixed fold order:** cubecl-cpu runs units concurrently on separate threads — order is not guaranteed.
- **Accumulating histograms in f32:** C++ uses `hist_t = double`. Match it (f64 cells).
- **"Improving" the subtraction trick or scan order:** forbidden by carried-forward faithful-mirror discipline.
- **Naming a cubecl runtime in any crate above lgbm-compute:** violates CMP-01.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU kernel codegen / launch | Custom HIP/OpenCL FFI | `cubecl` `#[cube]` + `::launch` | Project mandate; raw CUDA/OpenCL out of scope |
| Warp/plane reductions | Hand-written shuffle intrinsics | `plane_sum`/`plane_inclusive_sum`/`plane_shuffle_*` | Verified CubeCL frontend; portable across cpu/hip |
| Capability detection | Probing GPU caps manually | `client.features()` / `client.properties()` | Upstream already populates `Features`/`DeviceProperties` per device |
| Bin storage / iteration | New bin structs | Phase-2 `lgbm-dataset` `Bin` trait | D-02a: reuse the bit-faithful store |
| Float comparison harness | New comparator | `oracle-harness` `compare_exact_f64_bits` / `compare_within` | Existing seam; bit-exact vs ~1e-6 split already modeled |
| `%g` / golden serialization | New formatter | Phase-3 `format_g17`/`format_g6` if emitting floats as text | Already bit-proven |

**Key insight:** Every "hard" primitive in this phase (kernel launch, plane reduce, capability query, bin iteration, float comparison) already exists in cubecl 0.10.0 or in prior-phase crates. The phase's real work is the **faithful transcription of the C++ gain/accumulation math** and the **determinism engineering** of the cubecl-cpu fold order — not infrastructure.

## Runtime State Inventory

> Phase 4 is greenfield kernel code (no rename/migration). One inventory category is genuinely relevant: build artifacts / toolchain state for the optional ROCm feature.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — kernels are stateless; inputs come from the in-memory Phase-2 store | none |
| Live service config | None | none |
| OS-registered state | None | none |
| Secrets/env vars | `CUBECL_CPU_STACK_SIZE`/`CUBECL_CPU_STACK_MB` (optional, tune cpu worker stack); not required | none unless deep kernels overflow the 64MB default stack |
| Build artifacts | ROCm toolchain: `/opt/rocm-7.1.1` present, `hipcc` HIP 7.1.52802, `rocminfo` shows gfx1100 GPU. `cubecl-hip-sys` links `amdhip64`/`hiprtc` at build time — only when the `rocm` feature is enabled (SC#1: CPU-only build must not need it) | Verify `rocm` feature builds against ROCm 7.1; CPU-only build must compile with NO ROCm libs present |

**Nothing found in category:** Stored data, live service config, OS-registered state — verified by inspecting the kernel-free `Backend` skeleton and the stateless kernel design.

## Common Pitfalls

### Pitfall 1: cubecl-cpu is NOT single-threaded — the D-04 bit-exact bet depends on fold-ownership
**What goes wrong:** D-04 assumes the cubecl-cpu runtime is "single-threaded" and therefore bit-deterministic. The vendored source shows otherwise: `cubecl-cpu-0.10.0/src/compute/runner.rs::execute_data` spawns **one OS worker thread per cube unit** (`cube_dim.x * .y * .z` tasks dispatched to `Worker`s, joined via an mpsc channel). Units within a cube run concurrently. If multiple units accumulate into the same histogram cell (atomics or unsynchronized shared memory), the float summation ORDER is nondeterministic → not bit-exact, and atomics aren't even supported on cpu.
**Why it happens:** "Single-threaded deterministic core" (Phase-2 D-03) referred to the C++ `num_threads=1` semantic, not a guarantee about how cubecl-cpu schedules units.
**How to avoid:** Make each reduction owned by exactly ONE unit in a fixed order — either `CubeDim::new_1d(1)` (one unit folds everything, trivially matching C++ `num_threads=1` order) or a designed scheme where each output cell is summed by a single unit after `sync_cube`. The C++ `deterministic_` path (`leaf_splits.hpp:180`) does exactly this: it disables the OMP reduction and sums sequentially.
**Warning signs:** Repeated launches of the same kernel produce different low-bit f64 results; histogram cells differ by ULPs run-to-run.
**MITIGATION REQUIRED — Wave-0 spike (per D-04a):** Before building the full suite, write a minimal `construct_histograms` kernel and assert it produces **byte-identical** f64 output across N≥20 launches on cubecl-cpu, AND bit-identical to a hand-computed sequential f64 fold of the same inputs. If this fails, invoke the D-04a fallback (relax cubecl-cpu anchor to ~1e-6 and reconsider a scalar reference).

### Pitfall 2: Capability matrix is asymmetric — gate every divergent feature
**What goes wrong:** Assuming a feature present on one backend is present on the other.
**Verified matrix (cubecl 0.10.0):**
| Feature | cubecl-cpu | cubecl-hip (gfx1100) | Source |
|---------|-----------|----------------------|--------|
| `Plane::Ops` | NO (plane_size 1, not inserted) | YES (inserted in `hip/runtime.rs:172`) | features.rs:118 |
| `Plane::NonUniformControlFlow` | NO | YES | hip/runtime.rs:175 |
| f64 storage/arith | YES (`elem.rs:90`) | NO (commented out, "CUDA_ERROR_INVALID_VALUE", `cpp/shared/base.rs`) | register_supported_types |
| f32 atomic-add | NO (commented out) | YES (`AtomicUsage::Add` registered for F32) | cpp/shared/base.rs:~2120 |
| plane/warp size | 1 | 32 (RDNA3 wave32) | cpu/runtime.rs:51, hip/arch.rs:45 |
**Why it happens:** Alpha runtimes implement features unevenly; the docs don't surface this.
**How to avoid:** Query `client.features()` / `client.properties()` at startup (Pattern 2) and branch. The CPU path is sequential-fold (no Plane, no atomics, f64 OK). The HIP path can use Plane reductions but must NOT rely on f64 — accumulate in f32 on HIP, which is the ~1e-6-tolerated divergence from the f64 cpu anchor (this is exactly what the ~1e-6 contract was designed to absorb; CONCERNS.md FP-ordering note).
**Warning signs:** Kernel compile errors on HIP mentioning f64; `Plane::Ops` panics on cpu; atomic ops missing on cpu.

### Pitfall 3: Histograms accumulate in f64, not f32 (the `hist_t` subtlety)
**What goes wrong:** Treating "f32 end-to-end" (D-03) as "accumulate histograms in f32." LightGBM's `hist_t = double` (`bin.h:33`). Gradients/hessians are read as f32 (`score_t`) but summed into f64 cells, interleaved `[grad,hess]` stride-2, indexed `ti = bin << 1` (`dense_bin.hpp:120`). The C++ leaf-splits reductions also accumulate into `double` (`leaf_splits.hpp`).
**Why it happens:** The "f32" in the contract refers to inputs/scores/leaf values; the histogram is an internal f64 accumulator.
**How to avoid:** Kernel inputs are f32, histogram output cells are f64. On cubecl-cpu (f64 supported) this matches C++ exactly → bit-exact. On cubecl-hip (no f64) accumulate in f32 → within ~1e-6 of the f64 cpu result (documented, tolerated).
**Warning signs:** Histogram parity off by more than ULP on cpu (suggests f32 accumulation); large gradient-magnitude spreads amplify the error.

### Pitfall 4: `2*kEpsilon` hessian bump + `kEpsilon` scan seeds are load-bearing
**What goes wrong:** Dropping the exact epsilon placements changes which split wins.
**Verified positions:**
- `FindBestThreshold` adds `2 * kEpsilon` to `sum_hessian` at entry (`feature_histogram.hpp:172`).
- The scan seeds `sum_right_hessian = kEpsilon` (REVERSE) / `sum_left_hessian = kEpsilon` (forward) at scan start (`feature_histogram.hpp:862`, `:935`).
- `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f` (`meta.h:54,56`). Note `1e-15f` is a FLOAT literal — preserve the f32→f64 widening exactly.
**How to avoid:** Copy the constants and apply them in the exact arithmetic positions; do not "clean up" the seeds.

### Pitfall 5: Default-bin skip + offset arithmetic in the scan
**What goes wrong:** Restructuring the scan loop drops the `SKIP_DEFAULT_BIN` continue and the `offset` arithmetic, changing which thresholds are considered.
**Verified:** scan uses `offset = meta_->offset`, skips `(t + offset) == default_bin` when `SKIP_DEFAULT_BIN`, threshold recorded as `t - 1 + offset` (REVERSE) / `t + offset` (forward) (`feature_histogram.hpp:858-936`). `offset`/`default_bin_`/`most_freq_bin_` semantics in `bin.h:180-258`.
**How to avoid:** Replicate offset + default-bin skip exactly (CONCERNS.md §default-bin skip).

### Pitfall 6: Subtraction trick low-bit divergence (orchestration is Phase 5, but math defined here)
**What goes wrong:** Constructing both children directly instead of subtracting changes low bits.
**Verified:** `use_subtract = parent_leaf_histogram_array_ != nullptr` (`serial_tree_learner.cpp:398`); larger child is derived by `FeatureHistogram::Subtract` (`feature_histogram.hpp:99`). Phase 4 must expose a `subtract`-capable op or ensure `construct_histograms` outputs are subtract-compatible; the *orchestration* (which child) is Phase 5, but the subtract MATH belongs to the kernel layer. Confirm with the planner whether a `subtract_histograms` backend op is in-scope for D-01 (recommended: yes, since it is a histogram-layer op the learner orchestrates).

## Code Examples

### Histogram accumulation (C++ reference to transcribe)
```cpp
// Source: LightGBM/src/io/dense_bin.hpp:99-141 (ConstructHistogramInner, USE_HESSIAN)
hist_t* grad = out;            // hist_t = double
hist_t* hess = out + 1;
for (data_size_t i = start; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto ti = static_cast<uint32_t>(data(idx)) << 1;   // bin<<1
    grad[ti] += ordered_gradients[i];   // f32 read, f64 accumulate
    hess[ti] += ordered_hessians[i];
}
```

### Gain math (C++ reference to transcribe verbatim)
```cpp
// Source: LightGBM/src/treelearner/feature_histogram.hpp:711-734
static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
}
// GetLeafGain (no max-output, no smoothing — the common path):
//   USE_L1:  (ThresholdL1(g,l1)^2) / (h + l2)
//   else:    (g*g) / (h + l2)
// GetSplitGains = GetLeafGain(left) + GetLeafGain(right)
// CalculateSplittedLeafOutput:
//   USE_L1:  -ThresholdL1(g,l1) / (h + l2)   else  -g / (h + l2)
```

### Numerical split scan gates (C++ reference)
```cpp
// Source: LightGBM/src/treelearner/feature_histogram.hpp:862-934 (REVERSE branch)
// seed:  sum_right_gradient=0, sum_right_hessian=kEpsilon, right_count=0
// per bin t (high→low), after accumulate:
//   if (right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf) continue;
//   left_count = num_data - right_count;
//   if (left_count < min_data_in_leaf) break;
//   if (sum_left_hessian < min_sum_hessian_in_leaf) break;
//   current_gain = GetSplitGains(...);
//   if (current_gain <= min_gain_shift) continue;   // min_gain_shift = gain_shift + min_gain_to_split
//   if (current_gain > best_gain) { best_threshold = t-1+offset; best_gain = current_gain; }
```

### Plane reduction (CubeCL, HIP path)
```rust
// Source: cubecl-core-0.10.0/src/runtime_tests/plane.rs:7-15 (verified)
#[cube(launch)]
fn plane_reduce_kernel<F: Float>(out: &mut Tensor<F>) {
    let val = out[UNIT_POS as usize];
    let summed = plane_sum(val);       // warp-wide sum, gated by Plane::Ops
    if UNIT_POS == 0 { out[0] = summed; }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Raw OpenCL `histogram256.cl` (256 bins/workgroup, manual atomics) | CubeCL `#[cube]` kernels with `plane_*` + capability gate | cubecl 0.10.0 (2026-05) | Use the `.cl`/`.cu` files as *algorithm* references only, not line-by-line translations (CONCERNS.md §155) |
| Integer-quantized histogram strategy | Standard f32-input/f64-accumulate histograms | Phase 1 D-02/D-03 | Quantized path (`use_quantized_grad`) explicitly out of scope |
| CUDA warp-shuffle intrinsics | `plane_shuffle`/`plane_sum`/`plane_inclusive_sum` | cubecl 0.10.0 | Portable; gated by `Plane::Ops` |

**Deprecated/outdated:**
- Training-data CubeCL API shapes: do NOT trust. 0.10.0 is ~1 month old and the API differs from older docs.rs (0.9.0 was the last docs.rs-indexed before 0.10.0). All API claims in this doc are from the vendored 0.10.0 source.
- `cubecl-hip` 0.5.0 on docs.rs is a SEPARATE older standalone crate line — the workspace uses the integrated `cubecl 0.10.0` with the `hip` feature, not standalone `cubecl-hip 0.5`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | cubecl-cpu CAN be made bit-deterministic via single-owner ordered fold (`CubeDim::new_1d(1)`) | Pitfall 1, D-04 | HIGH — if the MLIR backend reorders fp ops or the fold still drifts, D-04 bit-exact anchor fails → D-04a fallback to ~1e-6. **Mitigated by the mandatory Wave-0 spike.** |
| A2 | cubecl-hip on gfx1100 produces f32 results within ~1e-6 of the cubecl-cpu f64 result for these kernels | D-03, Pitfall 2 | MEDIUM — ROCm is best-effort (D-03); a gap is a documented follow-up, not a blocker. f32-vs-f64 accumulation + warp reduction order may exceed 1e-6 on large/ill-conditioned inputs. |
| A3 | A `subtract_histograms` backend op is in-scope for D-01 (histogram-layer math) | Pitfall 6 | MEDIUM — if the planner scopes Subtract entirely into Phase 5, the kernel layer needn't expose it. Needs a planner/discuss decision. |
| A4 | The Phase-2 `Bin` trait exposes enough (per-row `data(idx)`, `num_data`) to drive a histogram kernel without re-binning | Don't Hand-Roll | LOW — verified `Bin::data(idx)->u32` exists; iterator-based access (`GetIterator`) may be needed for sparse, mirroring C++. |
| A5 | "f32 end-to-end" tolerates f64 histogram cells on cpu (matching C++ `hist_t=double`) | Pitfall 3 | LOW — directly verified in C++ source; this is faithful-mirror, not a deviation. |

## Open Questions

1. **Does the cubecl-cpu MLIR backend preserve fp operation order within a single unit's loop?**
   - What we know: each unit runs on its own thread; a single-owner loop has no cross-thread reduction.
   - What's unclear: whether MLIR/LLVM applies fast-math reassociation that would reorder the f64 fold (LightGBM compiles WITHOUT fast-math for the deterministic path).
   - Recommendation: The Wave-0 spike (Pitfall 1) settles this empirically. If MLIR reassociates, investigate a `CUBECL_*` flag or accept ~1e-6 anchor (D-04a).

2. **HIP path: accumulate in f32 (no f64) — does warp-reduction order stay within ~1e-6?**
   - What we know: gfx1100 is wave32; f32 atomic-add is available; `plane_sum` is available.
   - What's unclear: whether a `plane_sum`-based histogram or an atomic-based one stays within 1e-6 of the f64 cpu anchor across the D-02a stress inputs.
   - Recommendation: Bring up HIP, run the oracle, document any gap (D-03a). Prefer the same single-owner ordered fold on HIP first (simplest parity), optimize to Plane later.

3. **Scope of `data_partition` output:** indices array (C++ `indices_` reordered, `leaf_begin_`/`leaf_count_`) vs a left/right boolean mask?
   - What we know: C++ `DataPartition::Split` reorders an indices array in place and tracks `leaf_begin_`/`leaf_count_` (`data_partition.hpp:101`); `Dataset::Split` decides per-row left/right.
   - What's unclear: the exact backend op signature (Claude's discretion per CONTEXT.md).
   - Recommendation: Mirror the C++ row→{left,right} partition with stable order; let Phase 5 own the `leaf_begin_`/`leaf_count_` bookkeeping.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all | ✓ | rustc 1.95.0 (edition 2024) | — |
| cubecl 0.10.0 | all kernels | ✓ | 0.10.0 (Cargo.lock) | — |
| cubecl-cpu runtime | CMP-02 anchor | ✓ | 0.10.0 (MLIR/LLVM backend) | — |
| ROCm toolkit | CMP-03 hip feature | ✓ | ROCm 7.1.1, HIP 7.1.52802 (`/opt/rocm-7.1.1`) | CPU-only build omits `rocm` feature (SC#1) |
| AMD GPU (HIP) | ORA-04 ROCm gate | ✓ | gfx1100 (Radeon 860M, RDNA3, wave32) via `rocminfo` | If HIP build/run fails: document as D-03a gap, CPU gate still hard-passes |
| C++ toolchain (test time) | — | n/a | — | NOT used at test time — header-only transcription, committed goldens (D-02) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** A C++ build of `lib_lightgbm` is unavailable (`external_libs` unvendored) — fallback is the header-only transcription capture (D-02), carried from Phases 1/2/3.

**Note on ROCm version:** cubecl-hip 0.10.0 was developed against ROCm 6.4.0 (per upstream changelog). The local ROCm is **7.1.1** — newer. ROCm minor-version drift is a known cubecl-hip-sys sensitivity (the `cubecl-hip-sys` crate pins HIP runtime symbol versions). If the `rocm` feature fails to build/link against ROCm 7.1, that is a documented D-03a gap (CPU gate unaffected), and a candidate `cubecl-hip-sys` env override (`ROCM_PATH=/opt/rocm-7.1.1`) should be tried first.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + `oracle-harness` comparators |
| Config file | none (Cargo workspace convention) |
| Quick run command | `cargo test -p lgbm-compute` |
| Full suite command | `cargo test --workspace` |
| ROCm gate (separate) | `cargo test -p lgbm-compute --features rocm` (run on the local GPU) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CMP-01 | No crate above lgbm-compute names cubecl | unit/grep | `cargo test -p lgbm-compute` + a guard test grepping deps | ❌ Wave 0 |
| CMP-02 | cubecl-cpu histogram bit-exact vs golden | integration | `cargo test -p oracle-harness --test kernel_parity` | ❌ Wave 0 |
| CMP-03 | cubecl-hip selectable, runs | integration | `cargo test -p lgbm-compute --features rocm` | ❌ Wave 0 |
| CMP-04 | capability gate + sequential fallback | unit | `cargo test -p lgbm-compute capability` | ❌ Wave 0 |
| CMP-05 | histogram/split/partition meet contract | integration | `cargo test -p oracle-harness --test kernel_parity` | ❌ Wave 0 |
| ORA-04 | oracle passes on cpu (hard) / runs on rocm | integration | `cargo test --workspace` (cpu) + rocm-feature run | ❌ Wave 0 |
| D-04a | cubecl-cpu bit-determinism spike (N launches identical) | spike test | `cargo test -p lgbm-compute determinism_spike` | ❌ Wave 0 (do FIRST) |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute`
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** full workspace green (cpu) before `/gsd-verify-work`; ROCm run executed with gaps documented (D-03a)

### Wave 0 Gaps
- [ ] `crates/lgbm-compute/tests/determinism_spike.rs` — the D-04a bit-exact spike (RUN FIRST, before building the suite)
- [ ] `crates/lgbm-compute/src/error.rs` — `ComputeError` (thiserror)
- [ ] `crates/lgbm-compute/src/runtime.rs` — runtime selection + capability gate
- [ ] `crates/oracle-harness/tests/kernel_parity.rs` — golden replay (histogram/split/partition layers)
- [ ] `crates/oracle-harness/src/comparator.rs` — confirm `compare_exact_f64_bits` is exported (it is; line 150) and add a multi-bin helper if needed
- [ ] `xtask/cpp/kernel_capture.cpp` + `xtask` `kernel-capture` subcommand — header-only transcription of ConstructHistogram + FindBestThreshold + Split
- [ ] `tests/fixtures/kernels/` — committed synthetic-input goldens (D-02a path coverage)

## Security Domain

> `security_enforcement: true`, ASVS level 1. This is a numerical compute library with no auth/session/network surface. Most ASVS categories are N/A.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate kernel inputs at the `Backend` boundary: bin indices in range `[0, num_bin)`, array lengths consistent, num_data ≥ 0 → typed `ComputeError`, never a panic/UB. Mirrors the Phase-2 `DatasetError` discipline (Security V5 carried forward). |
| V6 Cryptography | no | — |

### Known Threat Patterns for {Rust compute / GPU}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds bin index into histogram array | Tampering/DoS | Validate `bin < num_bin` before launch; size `out` as `2*num_bin` f64; `ArrayArg` length must match |
| `unsafe` in kernel launch (`from_raw_parts`, `launch`) | Tampering | Confine all `unsafe` to lgbm-compute; document the safety invariant (handle/len correspondence) at each call site |
| Integer overflow in `bin << 1` / index math | Tampering | Use checked/`u32` widening exactly as C++ (`static_cast<uint32_t>(data(idx)) << 1`); validate num_bin fits |
| Worker-thread stack overflow on deep kernels (cubecl-cpu) | DoS | Default 64MB stack; expose `CUBECL_CPU_STACK_MB` note; keep kernels iterative not deeply recursive |

## Sources

### Primary (HIGH confidence)
- `cubecl 0.10.0` vendored source — `~/.cargo/registry/src/.../cubecl-0.10.0`, `cubecl-core-0.10.0`, `cubecl-cpu-0.10.0`, `cubecl-hip-0.10.0`, `cubecl-cpp-0.10.0`, `cubecl-ir-0.10.0`, `cubecl-runtime-0.10.0` — Plane API (`frontend/plane.rs`, `runtime_tests/plane.rs`), capability/Features (`ir/features.rs`, `runtime_tests/atomic.rs`), runtimes (`cpu/runtime.rs`, `cpu/compute/runner.rs+worker.rs`, `hip/runtime.rs`, `cpp/shared/base.rs`, `hip/arch.rs`), launch/client API (`compute/launcher.rs`, `runtime/client.rs`), `frontend/element/atomic.rs`.
- LightGBM C++ reference (read-only, pinned) — `src/treelearner/feature_histogram.hpp` (gain math 711-845, scan 845-1000, Subtract 99, `2*kEpsilon` at 172), `src/io/dense_bin.hpp` (ConstructHistogramInner 99-141), `src/treelearner/serial_tree_learner.cpp` (ConstructHistograms/use_subtract 398-475), `src/treelearner/leaf_splits.hpp` (deterministic branch 98-180), `src/treelearner/data_partition.hpp` (Split 101), `include/LightGBM/bin.h` (hist_t=double 33, offset/default_bin 180-258), `include/LightGBM/meta.h` (kEpsilon/kZeroThreshold 54-56).
- `crates.io API` — cubecl latest = 0.10.0, published 2026-05-07.
- Local environment — `rocminfo` (gfx1100 GPU), `hipcc --version` (ROCm 7.1.1 / HIP 7.1.52802), `rustc 1.95.0`, Cargo.lock (cubecl 0.10.0 + subcrate checksums).
- Project: `.planning/codebase/CONCERNS.md` (§histogram/best-split, §FP reduction ordering, §subtraction trick, §default-bin skip, §kEpsilon), existing `crates/lgbm-compute`, `crates/lgbm-dataset`, `crates/lgbm-core/config`, `crates/oracle-harness`, `xtask`.

### Secondary (MEDIUM confidence)
- WebSearch — cubecl-hip ROCm 6.4.0 baseline, separate older standalone cubecl-hip 0.5 line (informational; the workspace uses integrated cubecl 0.10.0 `hip` feature).

### Tertiary (LOW confidence)
- Training-data CubeCL API shapes — explicitly NOT relied upon; all API claims re-verified against vendored 0.10.0 source.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — cubecl 0.10.0 verified in lockfile + crates.io; no new deps.
- CubeCL API (Plane/launch/capability): HIGH — read directly from vendored 0.10.0 source, not docs/training.
- C++ gain/histogram math: HIGH — read directly from the pinned reference tree.
- Determinism bet (D-04): MEDIUM — architecture understood (multi-worker cpu model), but bit-exactness of the MLIR f64 fold is an empirical unknown (A1) gated by the mandatory Wave-0 spike.
- ROCm ~1e-6 parity: MEDIUM-LOW — best-effort per D-03; ROCm 7.1 vs cubecl-hip 6.4 baseline drift + f32-vs-f64 accumulation are real unknowns (A2), but D-03a makes any gap a documented follow-up, not a blocker.

**Research date:** 2026-06-05
**Valid until:** 2026-06-19 (14 days — cubecl is alpha and fast-moving; re-verify the API against the vendored source if the pin changes)
