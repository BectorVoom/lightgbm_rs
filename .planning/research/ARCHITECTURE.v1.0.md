# Architecture Research

**Domain:** Pure-Rust port of LightGBM (histogram-based gradient boosting) on a CubeCL compute backend, structured as a Cargo workspace, with strict 1e-12 cross-backend (CPU + ROCm) numerical parity.
**Researched:** 2026-06-05
**Confidence:** HIGH on crate decomposition and C++ mapping (grounded in `.planning/codebase/`), HIGH on CubeCL primitives (Context7 `/tracel-ai/cubecl` v0.10 + repo docs), MEDIUM on the cross-backend determinism strategy (the approach is grounded in LightGBM's own integer-quantized gradient path, but 1e-12 parity on ROCm must be empirically validated — no source guarantees it).

## Standard Architecture

This is a layered, interface-driven gradient-boosting engine. The C++ reference uses an abstract-base-class + string `Create*` factory at every seam (`.planning/codebase/ARCHITECTURE.md`). The Rust port replaces each factory with an enum + trait, and replaces device `#ifdef` branching with a single `Backend` trait boundary. The decomposition below is a one-to-one mapping of the C++ subsystems onto Cargo crates, chosen so the dependency graph is acyclic and the compute backend is swappable without touching boosting/tree-learner logic.

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BINDINGS                lgbm-python (PyO3)        [optional, top of stack] │
│  mirrors lightgbm Python API → calls lgbm-api only                         │
├──────────────────────────────────────────────────────────────────────────┤
│  PUBLIC API              lgbm-api  (Booster, Dataset, train/predict)        │
│  Rust-native facade; owns config enums; wires boosting + objective + metric │
├───────────────┬───────────────────────┬────────────────────┬──────────────┤
│               ▼                       ▼                    ▼               │
│  ┌────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐ │
│  │  lgbm-boosting     │  │  lgbm-objective      │  │  lgbm-metric       │ │
│  │  GBDT/DART/RF/GOSS │  │  grad/hess kernels   │  │  eval kernels      │ │
│  │  score updater     │  │  [GPU-RELEVANT]      │  │  [GPU-RELEVANT]    │ │
│  └─────────┬──────────┘  └──────────┬───────────┘  └─────────┬──────────┘ │
│            │ Train(grad,hess)→Tree  │ GetGradients           │ Eval        │
│            ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  lgbm-treelearner   histogram build → split scan → partition → tree │  │
│  │  serial learner; monotone/categorical; gradient discretizer  [GPU]  │  │
│  └─────────┬──────────────────────────────────────────────────────────┘  │
├────────────┼───────────────────────────────────────────────────────────────┤
│            ▼ ConstructHistograms / score update                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  lgbm-compute   Backend trait + CubeCL kernels (histogram, split-   │  │
│  │  gain scan, score update, grad/hess); CPU ↔ ROCm runtime selection  │  │
│  │  generic over R: Runtime  (cubecl-cpu / cubecl-hip)                  │  │
│  └─────────┬──────────────────────────────────────────────────────────┘  │
├────────────┼───────────────────────────────────────────────────────────────┤
│            ▼ binned columns, gradients/hessians as device buffers          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  lgbm-data    Dataset, BinMapper, FeatureGroup, (Multi)Val Bin,     │  │
│  │  Metadata, dataset loader (CSV/TSV/LibSVM). Immutable after load.   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────────┤
│  lgbm-model   Tree (nodes/splits/leaf outputs), text+JSON model I/O        │
│  lgbm-core    shared types: score_t/f64, ids, config enums, thiserror errs │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Crate | Responsibility (what it owns) | C++ reference subsystem |
|-------|-------------------------------|-------------------------|
| `lgbm-core` | Shared scalar types (`score_t`, `data_size_t`, bin ids), config **enums** (boosting/objective/metric/device parsed once), `thiserror` domain error types, deterministic-reduction utility traits. No heavy logic. | `include/LightGBM/meta.h`, `config.h`, `utils/` |
| `lgbm-data` | Binned columnar store: `BinMapper::FindBin`, `FeatureGroup`, `DenseBin`/`SparseBin`/`MultiValBin`, `Metadata` (labels/weights/query boundaries), dataset loader (CSV/TSV/LibSVM, later Arrow). Immutable after `finish_load`. | `src/io/` (`bin.cpp`, `dataset.cpp`, `dataset_loader.cpp`, `dense_bin.hpp`, `metadata.cpp`), `include/.../{bin.h,dataset.h,feature_group.h}` |
| `lgbm-model` | `Tree` model (nodes, thresholds, leaf outputs), `Tree::Predict`, text + JSON model serialize/deserialize (C++ format compatibility). | `src/io/tree.cpp`, `src/boosting/gbdt_model_text.cpp`, `include/.../tree.h` |
| `lgbm-compute` | **The backend boundary.** `Backend` trait + CubeCL kernels: histogram construction, split-gain scan, score update, grad/hess application. Owns CubeCL `Runtime`/client lifetime and CPU↔ROCm selection. Generic over `R: Runtime`. | `src/treelearner/{ocl,cuda}/*`, `src/io/{dense,sparse}_bin.hpp::ConstructHistogram*`, `src/boosting/cuda/cuda_score_updater`, `src/cuda/cuda_algorithms` |
| `lgbm-objective` | `ObjectiveFunction` trait + `get_gradients(score)→(grad,hess)`, `convert_output`, `boost_from_score`. Regression/binary/multiclass/ranking/xentropy families. Hot loops dispatch to `lgbm-compute`. | `src/objective/*.hpp`, `objective_function.cpp` |
| `lgbm-metric` | `Metric` trait + `eval(score)→Vec<f64>`; DCG/NDCG tables. | `src/metric/*.hpp`, `dcg_calculator.cpp` |
| `lgbm-treelearner` | Grow one tree: orchestrate histogram build (via compute), `find_best_splits` scan, `ArgMax` best-leaf pick, `data_partition`, monotone constraints, categorical splits, gradient discretizer. | `src/treelearner/serial_tree_learner.cpp`, `feature_histogram.{cpp,hpp}`, `data_partition.hpp`, `monotone_constraints.hpp`, `gradient_discretizer.{cpp,hpp}`, `col_sampler.hpp` |
| `lgbm-boosting` | Outer GBDT loop: `train_one_iter`, `boosting()` (call objective), bagging/GOSS sample strategy, `ScoreUpdater`, shrinkage, early stopping, prediction. Variants GBDT/DART/RF. | `src/boosting/gbdt.cpp`, `dart.hpp`, `rf.hpp`, `goss.hpp`, `bagging.hpp`, `score_updater.hpp` |
| `lgbm-api` | Rust-native facade: `Dataset`, `Booster`, `train`, `predict`. Parses config to enums (no stringly-typed factories), wires boosting + objective + metric + treelearner + chosen backend. `anyhow` at this app layer. | `src/c_api.cpp` semantics (not the C ABI itself — out of scope v1), `src/application/` |
| `lgbm-python` | PyO3 module mirroring the `lightgbm` Python surface (`Dataset`, `Booster`, `train`, `cv`, sklearn estimators later). Thin; converts numpy/Arrow → `lgbm-api`. | `python-package/lightgbm/{basic,engine,sklearn}.py` |

### Dependency direction (acyclic — strict downward)

```
lgbm-python → lgbm-api
lgbm-api    → lgbm-boosting, lgbm-objective, lgbm-metric, lgbm-treelearner, lgbm-data, lgbm-model, lgbm-core
lgbm-boosting → lgbm-treelearner, lgbm-objective, lgbm-metric, lgbm-model, lgbm-data, lgbm-compute, lgbm-core
lgbm-treelearner → lgbm-compute, lgbm-data, lgbm-model, lgbm-core
lgbm-objective   → lgbm-compute, lgbm-data, lgbm-core
lgbm-metric      → lgbm-compute, lgbm-data, lgbm-core
lgbm-compute     → lgbm-data (binned buffers), lgbm-core, cubecl runtimes
lgbm-data        → lgbm-core
lgbm-model       → lgbm-core
lgbm-core        → (nothing in workspace)
```

Two rules keep this clean: (1) **`lgbm-compute` is the only crate that names a CubeCL runtime** — everything above it talks to the `Backend` trait, so the boosting loop is device-agnostic (this directly fixes the C++ anti-pattern of `#ifdef USE_CUDA` scattered through `UpdateScore`). (2) **`lgbm-core` depends on nothing** so error/types are shareable without cycles.

## Recommended Project Structure

```
Cargo.toml                  # [workspace] members = crates/*
crates/
├── lgbm-core/              # types, config enums, thiserror errors, reduction traits
├── lgbm-data/              # BinMapper, FeatureGroup, Bin impls, Metadata, loader
├── lgbm-model/             # Tree, model text/JSON I/O
├── lgbm-compute/
│   ├── src/backend.rs      # Backend trait (the device boundary)
│   ├── src/cpu.rs          # CpuBackend  = ComputeBackend<CpuRuntime>
│   ├── src/rocm.rs         # RocmBackend = ComputeBackend<HipRuntime>  (feature "rocm")
│   └── src/kernels/        # #[cube] kernels, runtime-generic over R: Runtime
│       ├── histogram.rs    #   ConstructHistogram (grad/hess accumulation)
│       ├── split_scan.rs   #   prefix-scan split-gain over bins
│       ├── score_update.rs #   add tree output into score buffer
│       └── grad_hess.rs    #   objective gradient/hessian application
├── lgbm-objective/         # ObjectiveFunction trait + families
├── lgbm-metric/            # Metric trait + families, DCG tables
├── lgbm-treelearner/       # serial learner, feature_histogram scan, data_partition,
│                           # monotone, categorical, gradient_discretizer
├── lgbm-boosting/          # GBDT/DART/RF, sample strategy, score updater
├── lgbm-api/               # Booster, Dataset, train/predict facade
└── lgbm-python/            # PyO3 (cfg-gated; not part of default workspace build)
tests/
└── oracle/                 # Rust-vs-C++ 1e-12 parity harness (runs on CPU and ROCm)
```

### Structure Rationale

- **`crates/*` flat layout:** standard Cargo workspace; each crate is independently testable and the dependency edges are visible in each `Cargo.toml`.
- **`lgbm-compute/src/kernels/` separate from `cpu.rs`/`rocm.rs`:** kernels are written **once** generic over `R: Runtime`; `cpu.rs`/`rocm.rs` only pick the runtime and own the client. This mirrors CubeCL's own `pub fn run<R: Runtime>(device)` idiom.
- **`lgbm-python` excluded from default workspace members** (or behind a feature) so the core builds and tests without a Python toolchain; oracle parity does not depend on it.
- **`lgbm-core` deliberately thin:** it is the only universal dependency, so keeping logic out of it prevents recompilation cascades.

## Architectural Patterns

### Pattern 1: Backend trait as the single device boundary

**What:** One trait in `lgbm-compute` abstracts all GPU-relevant operations. Implemented twice (CPU, ROCm) but kernels are shared and generic over `R: Runtime`. Callers never see CubeCL types.
**When to use:** Everywhere a `[GPU-RELEVANT]` op is invoked — histogram build, split scan, score update, grad/hess.
**Trade-offs:** Trait calls add an indirection per *batch* (not per element), negligible. Forces all device data (`DeviceBuffer`) to live behind the trait, which is what we want for swappability.

```rust
// lgbm-compute/src/backend.rs
pub trait Backend: Send + Sync {
    type Buf<T: CubeElement>: DeviceBuffer<T>;

    fn upload<T: CubeElement>(&self, data: &[T]) -> Self::Buf<T>;
    fn download<T: CubeElement>(&self, buf: &Self::Buf<T>) -> Vec<T>;

    /// Accumulate (grad, hess) into per-bin histogram for a leaf's rows.
    /// `hist_bits` selects 16/32-bit integer accumulator (determinism, below).
    fn construct_histogram(
        &self, feature: &BinnedColumn,
        rows: Option<&Self::Buf<u32>>,      // None => all rows
        int_grad_hess: &Self::Buf<i32>,     // quantized gradients
        hist_bits: HistBits,
        out: &mut Self::Buf<i64>,           // integer histogram bins
    );

    fn split_gain_scan(&self, hist: &Self::Buf<i64>, /* ... */) -> Vec<SplitInfo>;
    fn update_score(&self, leaf_value: &[f64], partition: &Partition, score: &mut Self::Buf<f64>);
}

// lgbm-compute/src/rocm.rs — the ONLY place a runtime is named
pub struct ComputeBackend<R: Runtime> { client: ComputeClient<R::Server, R::Channel>, has_plane: bool }
pub type CpuBackend  = ComputeBackend<cubecl::cpu::CpuRuntime>;
pub type RocmBackend = ComputeBackend<cubecl::hip::HipRuntime>;   // ROCm
```

### Pattern 2: Runtime-generic kernels with comptime Plane fallback (CUDA-warp → CubeCL Plane)

**What:** Write each kernel once, parameterized by a `#[comptime] use_plane: bool`. When the runtime reports `client.features().plane.contains(Plane::Ops)`, the kernel emits `plane_sum`/`plane_*` (compiles to AMD wavefront / NVIDIA warp / Vulkan subgroup ops); otherwise it falls back to a deterministic sequential reduction. This is exactly how the C++ CUDA learner's warp-shuffle reductions map onto CubeCL — verified against `/tracel-ai/cubecl` docs.
**When to use:** Histogram block reductions and split-gain scans where the CUDA reference used `__shfl_*` warp ops.
**Trade-offs:** The plane path and the sequential path can produce *different* float orderings — which is the determinism hazard (see below). The resolution is to make the histogram accumulators **integer**, where order does not matter, so both paths give identical results.

```rust
// lgbm-compute/src/kernels/histogram.rs
#[cube(launch_unchecked)]
fn reduce_into_bin<I: Int>(part: &Array<I>, out: &mut Array<I>, #[comptime] use_plane: bool) {
    if use_plane {
        out[UNIT_POS] = plane_sum(part[UNIT_POS]);   // AMD wavefront / NVIDIA warp / subgroup
    } else {
        let mut acc = I::new(0);
        for i in 0..part.len() { acc += part[i]; }    // sequential (CPU runtime, plane_size = 1)
    }
}
// host: let use_plane = client.features().plane.contains(Plane::Ops);
```

### Pattern 3: Histogram subtraction trick preserved, made deterministic

**What:** The larger child's histogram = parent − smaller child (C++ `use_subtract`). Subtraction of two *integer* histograms is exact and order-independent, so it carries no float-determinism cost. Keep this optimization; it halves histogram work.
**When to use:** Every non-root split.
**Trade-offs:** Requires the parent histogram to remain resident (a fixed-size histogram pool sized by `num_leaves × total_bins`), same as C++.

### Pattern 4: Enum + match factories (replace stringly-typed `Create*`)

**What:** Parse config strings to exhaustive enums once at `lgbm-api`, then `match` to construct boosting/objective/metric/learner. Compile-time exhaustiveness; typos caught at parse, not deep in a hot loop.
**When to use:** All four C++ `Create*` seams.
**Trade-offs:** Adding an objective touches one enum + one match arm (vs. registering a string). Acceptable and safer.

## Data Flow

### Training flow (raw data → ensemble), with GPU-relevant stages flagged

```
[raw file / matrix]
   ↓  lgbm-data: sample columns → BinMapper::find_bin → pack into FeatureGroups
[Dataset: binned integer columns]            (CPU, one-time; immutable after finish_load)
   ↓  lgbm-boosting: GBDT loop, per iteration:
   │
   ├─(1) get_gradients(score) ............ lgbm-objective   [GPU-RELEVANT: grad/hess kernel]
   ├─(2) discretize gradients → int8 + scale  lgbm-treelearner::GradientDiscretizer  [det. linchpin]
   ├─(3) bagging / GOSS subsample ........ lgbm-boosting (CPU; deterministic RNG by seed)
   ├─(4) per tree, leaf-wise growth (num_leaves-1 splits):
   │       a. construct_histogram(rows, int_grad_hess) .... lgbm-compute  [GPU-RELEVANT: hottest]
   │       b. split_gain_scan(hist) ...................... lgbm-compute  [GPU-RELEVANT]
   │       c. ArgMax best leaf ........................... lgbm-treelearner (small, CPU)
   │       d. split + data_partition .................... lgbm-compute / lgbm-treelearner [GPU-RELEVANT]
   ├─(5) renew leaf outputs, shrinkage ... lgbm-treelearner + lgbm-model
   ├─(6) update_score(tree) ............. lgbm-compute   [GPU-RELEVANT: score reduction]
   └─(7) eval metrics, early stop ....... lgbm-metric    [GPU-RELEVANT for big eval sets]
   ↓
[append Tree to ensemble]
```

GPU-relevant stages (the candidate CubeCL kernels): **(1) gradient/hessian, (4a) histogram construction [hottest], (4b) split-gain scan, (4d) data partition, (6) score update, (7) metric eval.** Stages (3) and (4c) stay on CPU — small, branchy, and order-sensitive in ways better controlled host-side.

### Prediction flow

```
[features] → lgbm-model: Tree::predict per tree → sum over ensemble (ordered, fixed) →
lgbm-objective::convert_output (sigmoid/softmax) → [score]
```

### State management

- `Dataset` is **immutable** after load (matches C++ `FinishLoad`) — safe to share `&Dataset` across the loop and across backends.
- Mutable training state lives in `lgbm-boosting` (`models`, `gradients`, `hessians`, score buffers) and `lgbm-treelearner` (`data_partition`, histogram pool, `leaf_splits`).
- Device buffers (gradients, histograms, scores) live behind `Backend::Buf`; host mirrors only materialized when needed (e.g. ArgMax, final leaf outputs).

## Determinism Architecture (the critical constraint: 1e-12 on CPU **and** ROCm)

Floating-point `+` is non-associative, so a parallel/warp reduction and a sequential reduction over the same values generally differ in the last bits. With naive f32 or even f64 atomic accumulation, ROCm histograms will **not** match a C++ CPU reference at 1e-12. The architecture must remove order-dependence from the accumulation hot paths. Strategy, in priority order:

**1. Integer-quantized gradient accumulation (primary mechanism — grounded in LightGBM's own code).**
LightGBM already ships a `GradientDiscretizer` (`src/treelearner/gradient_discretizer.{cpp,hpp}`, verified): per iteration it maps `(grad, hess)` to `int8` with a stored `grad_scale`/`hess_scale`, then histograms accumulate as **integers** (adaptive 16- or 32-bit bins via `SetNumBitsInHistogramBin`). **Integer addition is associative and exact**, so the histogram is *bit-identical* regardless of reduction order, thread count, or whether the Plane path or the sequential path runs. The split-gain scan then operates on integer sums scaled back by `grad_scale/hess_scale` at the end (one deterministic conversion). This is the single most important design decision: **make integer histogram accumulation the default compute path on every backend**, so CPU and ROCm produce the same bits by construction. The 16/32-bit adaptive width (chosen by data count per leaf) bounds overflow exactly as C++ does.

**2. Deterministic bin boundaries.**
`BinMapper::find_bin` must reproduce C++ bin edges bit-for-bit (same quantile algorithm, same tie-breaking, same `max_bin`). Binning is one-time and CPU-only, so this is a faithful-port problem, not a parallelism problem — but it is upstream of everything, so any deviation here fails parity before training starts. Validate binning in isolation against the oracle first.

**3. Ordered / fixed-shape reductions for the unavoidable float sums (scores, metrics, leaf outputs).**
Some reductions cannot be integerized cheaply — final leaf-output computation, `ScoreUpdater` sums, and metric eval use f64. For these: (a) accumulate in **f64** always (never f32, even on GPU — confirm `cubecl-hip` exposes f64; if a kernel must stay f32, keep that sum on CPU); (b) use a **fixed-shape tree reduction** (deterministic pairwise) rather than atomic-add, so the addition order is a function of length only, identical across backends and thread counts; (c) never use floating `AtomicAdd` in a determinism-critical path — atomic float order is nondeterministic across runs and backends.

**4. Deterministic work decomposition.**
Histogram tiling, partition layout, and plane (wavefront) size must be derived from data shape, not from device-reported occupancy, so the *set* of partial sums and their combination order is fixed. The CPU runtime runs with `plane_size = 1` and sequential cube scheduling (confirmed) — that gives a natural reference ordering; the GPU path must reduce to the *same value*, which integerization (mechanism 1) guarantees, and which fixed-shape f64 reduction (mechanism 3) guarantees where integers do not apply.

**5. Match the C++ deterministic mode.**
LightGBM has a `deterministic` config flag and `force_row_wise/force_col_wise` controls. The oracle reference C++ must be run in its deterministic configuration, and the Rust port targets *that* output. Document the exact reference config in the oracle harness.

**Net effect on the compute layer:** the `Backend` trait's histogram signatures take/return **integers** (`Buf<i32>` gradients, `Buf<i64>` histograms) as the default path, with float histograms only as a non-deterministic fast path that is *off* under the parity contract. This shapes every kernel in `lgbm-compute/src/kernels/`.

**Residual risk (honest):** No source proves f64 transcendental functions (exp/log in objectives like binary/poisson) are bit-identical between a ROCm device library and the host libm. If objective grad/hess diverge in the last ULP *before* discretization, discretization may absorb it (quantization is a rounding step) — but this must be **empirically validated**, and if it fails, the fallback is to compute objective grad/hess on CPU (host libm) and only push the already-discretized integers to the GPU. Flag this for early validation.

## Build Order (foundational → dependent)

Driven by the dependency graph and by "what must be bit-exact before anything above it can be validated":

1. **`lgbm-core`** — types, config enums, error types, reduction traits. (No deps.)
2. **`lgbm-data`** — `BinMapper`/binning first; this is the determinism root (mechanism 2). Validate bin boundaries against the oracle before proceeding. Then `FeatureGroup`, `Bin` impls, `Metadata`, loader.
3. **`lgbm-model`** — `Tree` storage + text/JSON I/O. Enables loading a C++-trained model to validate *prediction* parity independently of training.
4. **`lgbm-compute`** — `Backend` trait + CPU runtime first, with the **integer histogram** and split-scan kernels (determinism mechanism 1). Add the ROCm runtime as a second impl of the same trait; cross-validate CPU vs ROCm on the kernels in isolation before wiring into training.
5. **`lgbm-objective`** + **`lgbm-metric`** — grad/hess and eval; can proceed in parallel once compute exists. (Objective grad/hess parity is a validation gate per residual-risk note.)
6. **`lgbm-treelearner`** — serial learner orchestrating compute kernels; gradient discretizer; split finding; partition; then monotone + categorical.
7. **`lgbm-boosting`** — GBDT loop first (simplest path to end-to-end 1e-12), then GOSS/bagging, then DART and RF.
8. **`lgbm-api`** — facade wiring everything; the surface the oracle harness drives.
9. **`lgbm-python`** — PyO3 last; it is a translation layer over a validated `lgbm-api`.

**The oracle harness** is built alongside step 2 (binning) and grows with each layer — every crate is validated against C++ as it lands, not at the end. Running the harness on **ROCm** is a gate at steps 4, 6, and 7.

## Anti-Patterns

### Anti-Pattern 1: Float atomic-add histogram accumulation
**What people do:** Port the GPU histogram as `AtomicAdd<f32>`/`AtomicAdd<f64>` into bins, matching a naive CUDA tutorial.
**Why it's wrong:** Atomic float order is nondeterministic across runs *and* differs from the CPU sequential reference — instant 1e-12 failure on ROCm.
**Do this instead:** Integer-quantized gradients + integer histogram bins (associative, exact); subtract trick stays exact. Float only for the final scaled-back gain and leaf output.

### Anti-Pattern 2: Naming a CubeCL runtime above `lgbm-compute`
**What people do:** Reach for `CudaRuntime`/`HipRuntime` inside the tree learner or boosting loop "just to launch one kernel."
**Why it's wrong:** Recreates the C++ `#ifdef USE_CUDA`-in-hot-path coupling; makes CPU-only testing impossible and breaks backend swappability.
**Do this instead:** Every device op goes through the `Backend` trait. The runtime type is named in exactly one file (`rocm.rs`/`cpu.rs`).

### Anti-Pattern 3: 1:1 template-matrix translation of `ConstructHistogram`
**What people do:** Mirror C++'s `<USE_INDICES, USE_HESSIAN, USE_QUANT_GRAD, HIST_BITS>` template explosion as Rust generics.
**Why it's wrong:** Monomorphization blowup and unreadable kernels (flagged in `.planning/codebase/ARCHITECTURE.md`).
**Do this instead:** A small set of parametric `#[cube]` kernels keyed by `#[comptime] hist_bits` and a runtime `use_indices` branch.

### Anti-Pattern 4: Device-reported occupancy driving reduction shape
**What people do:** Size tiles/partials from `client` occupancy or warp count.
**Why it's wrong:** Makes the *combination order* of partial sums hardware-dependent — different ROCm GPUs (or CPU) diverge.
**Do this instead:** Derive tiling and reduction shape from data dimensions only; keep float reductions fixed-shape and pairwise.

## Integration Points

### External (compute runtimes)

| Runtime | Integration | Notes |
|---------|-------------|-------|
| `cubecl-cpu` | `CpuBackend = ComputeBackend<CpuRuntime>` | Reference ordering (plane_size 1, sequential). The CPU parity target and CI default. |
| `cubecl-hip` (ROCm) | `RocmBackend = ComputeBackend<HipRuntime>`, behind `feature = "rocm"` | Mandated GPU test target. Plane API → AMD wavefront ops. Validate f64 + plane support via `client.features()` at startup. |
| CubeCL `Plane` API | `plane_sum`/`plane_*` under `#[comptime] use_plane` | Maps CUDA warp shuffles → portable subgroup ops. Determinism comes from integerization, not from the plane op itself. |

Backend selection: a `Device` enum in config (`Cpu` | `Rocm`) resolved once in `lgbm-api`; `feature = "rocm"` gates compiling the HIP runtime so a CPU-only build needs no ROCm toolchain. Both Cargo-feature (compile) and runtime (config) selection are supported, per the project constraint.

### Internal boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `lgbm-boosting ↔ lgbm-treelearner` | `train(grad, hess, is_first) -> Tree` | The tree-grow seam; device-agnostic. |
| `lgbm-treelearner ↔ lgbm-compute` | `Backend` trait calls (histogram, split scan, partition) | Only place hot kernels are invoked. |
| `lgbm-boosting ↔ lgbm-objective` | `get_gradients(score) -> (grad, hess)` | Objective owns grad/hess kernels; discretizer in treelearner consumes them. |
| `* ↔ lgbm-compute (device buffers)` | `Backend::Buf<T>` opaque buffers | Host materialization only at ArgMax / leaf output / final read. |
| `lgbm-python ↔ lgbm-api` | PyO3 over the Rust-native facade | No PyO3 type below `lgbm-api`. |

## Sources

- `.planning/codebase/ARCHITECTURE.md`, `STRUCTURE.md`, `INTEGRATIONS.md` — C++ reference subsystem boundaries, data flow, factories, GPU-relevant flags (HIGH).
- `.planning/PROJECT.md` — constraints: 1e-12 cross-backend, CubeCL mandate, Plane API, deterministic reductions, crate-per-responsibility (HIGH).
- CubeCL docs via Context7 `/tracel-ai/cubecl` (v0.10.0, May 2026) — runtime-generic kernels, `Plane::Ops` feature query, `plane_sum`, comptime fallback, monomorphized `#[cube]` traits, runtimes (`cubecl-cpu`/`cubecl-cuda`/`cubecl-hip`/`cubecl-wgpu`), CPU runtime plane_size 1 + sequential scheduling (HIGH).
- LightGBM `src/treelearner/gradient_discretizer.hpp` (read directly) — int8 gradient discretization + adaptive 16/32-bit integer histogram bins; the basis of the integer-accumulation determinism strategy (HIGH for the mechanism's existence; MEDIUM that it yields 1e-12 on ROCm — requires empirical validation).

---
*Architecture research for: pure-Rust LightGBM port on CubeCL with cross-backend bit-determinism*
*Researched: 2026-06-05*
