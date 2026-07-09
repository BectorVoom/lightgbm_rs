<!-- refreshed: 2026-07-09 -->
# Architecture

**Analysis Date:** 2026-07-09

## System Overview

```text
┌───────────────────────────────────────────────────────────────────────┐
│  lgbm            facade crate — public API / PyO3 target              │
│  `crates/lgbm/src/{booster,builder}.rs`                                │
└───────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│  lgbm-boosting   GBDT outer loop, DART/RF/GOSS, score updater          │
│  `crates/lgbm-boosting/src/gbdt.rs`                                    │
└───────┬───────────────────────────────┬─────────────────┬─────────────┘
        │                               │                 │
        ▼                               ▼                 ▼
┌───────────────────┐        ┌────────────────────┐  ┌──────────────────┐
│ lgbm-objective     │        │ lgbm-treelearner    │  │ lgbm-metric      │
│ grad/hess per row   │        │ SerialTreeLearner   │  │ eval loss/metric │
│ `src/{regression,   │        │ (leaf-wise growth)  │  │ `src/*.rs`       │
│  binary,...}.rs`    │        │ `src/learner.rs`     │  └──────────────────┘
└───────┬────────────┘        └─────────┬───────────┘
        │                                │
        │                                ▼
        │                     ┌─────────────────────────────┐
        │                     │ lgbm-compute (Backend trait)  │
        │                     │ CubeCL isolation seam (CMP-01)│
        │                     │ `src/kernels/*.rs`, runtime.rs│
        │                     └──────────┬────────────────────┘
        │                                │ (Backend::* ops; cubecl types
        │                                │  never leak above this crate)
        ▼                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  lgbm-dataset    binning (BinMapper), FeatureGroup, EFB, Dataset       │
│  `crates/lgbm-dataset/src/{dataset,bin_mapper,feature_group}.rs`       │
└───────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│  lgbm-model      Tree, GbdtModel ensemble, text/JSON I/O, predict      │
│  `crates/lgbm-model/src/{tree,ensemble,predict,model_text}.rs`         │
└───────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│  lgbm-core       Config, error types, shared numeric types, RNG        │
│  `crates/lgbm-core/src/{types,error,random}.rs`                        │
└───────────────────────────────────────────────────────────────────────┘

  Cross-cutting test/dev crate:
  oracle-harness (`crates/oracle-harness/`) — C++ parity replay/comparator,
  dev-dependency of every layer crate above lgbm-core.
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `lgbm-core` | Shared types (`Config`, numeric aliases), error primitives, deterministic RNG | `crates/lgbm-core/src/types.rs`, `crates/lgbm-core/src/random.rs` |
| `lgbm-dataset` | Feature binning (`BinMapper`), `FeatureGroup`/EFB (Exclusive Feature Bundling), `Dataset`/`Metadata`, raw-data ingest | `crates/lgbm-dataset/src/bin_mapper.rs`, `crates/lgbm-dataset/src/dataset.rs`, `crates/lgbm-dataset/src/efb.rs` |
| `lgbm-model` | `Tree` node/split representation, `GbdtModel` ensemble, text/JSON model format, prediction | `crates/lgbm-model/src/tree.rs`, `crates/lgbm-model/src/ensemble.rs`, `crates/lgbm-model/src/predict.rs` |
| `lgbm-compute` | The **single CubeCL isolation seam (CMP-01)**: `Backend` trait, `#[cube]` kernels (histogram, split, partition, predict, grow-driver), runtime/capability probing (CPU vs ROCm) | `crates/lgbm-compute/src/lib.rs` (`trait Backend` at line 658), `crates/lgbm-compute/src/runtime.rs`, `crates/lgbm-compute/src/kernels/*.rs` |
| `lgbm-treelearner` | `SerialTreeLearner`: leaf-wise (best-first) single-tree growth loop driving `Backend` ops; `DataPartition`, histogram pool, monotone constraints, categorical splits | `crates/lgbm-treelearner/src/learner.rs`, `crates/lgbm-treelearner/src/data_partition.rs`, `crates/lgbm-treelearner/src/histogram_pool.rs` |
| `lgbm-objective` | Per-row gradient/hessian computation (regression, binary, multiclass, rank, cross-entropy, custom) | `crates/lgbm-objective/src/{regression,binary,multiclass,rank,xentropy,custom}.rs` |
| `lgbm-metric` | Evaluation metrics (regression/binary/multiclass/rank/xentropy), shared DCG tables | `crates/lgbm-metric/src/*.rs`, `crates/lgbm-metric/src/dcg_calculator.rs` |
| `lgbm-boosting` | Outer GBDT loop: `Gbdt` struct, DART/RF/GOSS strategies, `ScoreUpdater`, early stopping, on-device resident score path | `crates/lgbm-boosting/src/gbdt.rs`, `crates/lgbm-boosting/src/score_updater.rs`, `crates/lgbm-boosting/src/sample_strategy.rs` |
| `lgbm` | Public facade crate: `Booster`, `TrainingBuilder`, `DenseCorpus`/`RawCorpus`, `train*` entry points; the PyO3 binding target | `crates/lgbm/src/booster.rs`, `crates/lgbm/src/builder.rs` |
| `lgbm-python` | PyO3 bindings mirroring the official `lightgbm` Python API surface | `crates/lgbm-python/src/{booster,dataset,params,marshal}.rs` |
| `oracle-harness` | C++ parity test infrastructure: golden fixtures, comparator, kernel/learner/boosting parity replay | `crates/oracle-harness/src/comparator.rs` |
| `xtask` | Dev-only automation (regenerate committed C++ RNG golden fixtures) | `xtask/src/main.rs` |

## Pattern Overview

**Overall:** Layered Cargo workspace mirroring the C++ LightGBM subsystem boundaries 1:1, with one deliberate architectural addition not present in the C++ original: a **single compute-isolation seam crate** (`lgbm-compute`) that owns 100% of the CubeCL dependency, so no other crate ever names a `cubecl` type or a concrete GPU runtime.

**Key Characteristics:**
- **CMP-01 containment discipline**: only `lgbm-compute` depends on `cubecl` directly; every other crate interacts with compute exclusively through the `Backend` trait, `SplitInfo`/`GainConfig`, `BinColumn`, and re-exported `ComputeClient`/`Handle` types. Enforced by a `cmp01_containment` guard test.
- **Trait-parameterized generic learner** instead of C++'s stringly-typed `Create*` factories: `SerialTreeLearner<'_, B: Backend>` and `Gbdt::train_one_iter<B: Backend>` take `&impl Backend`/generic `B`, monomorphized at compile time for `CpuBackend` vs the ROCm/HIP backend — no runtime dispatch, no string-keyed factory registry.
- **Capability-probed numeric path selection (CMP-04)**: `runtime::probe_capabilities()` queries the active CubeCL runtime once at startup (`has_plane`, `has_f64`, `has_f32_atomic`, `plane_size`) and selects a `ReducePath` (`Sequential` f64 fold on cubecl-cpu vs `Plane` collective reduction on cubecl-hip) — asymmetric backend capabilities are gated explicitly rather than assumed uniform.
- **Histogram-based split finding**, ported faithfully from C++: pre-binned integer feature data (`BinColumn`, narrowest u8/u16/u32 per feature), per-leaf histograms of (grad, hess) accumulated over bins, splits found by scanning histograms via in-kernel gain math (never re-derived host-side).
- **Histogram subtraction trick** ported: smaller child's histogram is built directly; larger child's is derived by `parent − smaller` (`subtract_histograms` in `Backend`), selected via `DataPartition::leaf_count` (`num_data_in_left < num_data_in_right`).
- **Dependency DAG is a strict DAG, no cycles**: `lgbm-core` → `lgbm-dataset` → `lgbm-model` → {`lgbm-compute`, `lgbm-objective`, `lgbm-metric`} → `lgbm-treelearner` → `lgbm-boosting` → `lgbm` → `lgbm-python`. `lgbm-compute` depends on `lgbm-model` (not vice versa) only to name `Tree` as an on-device grow-tree return type — verified acyclic and documented in-code (D-03 Option A).
- **f32-vs-f64 numerical fidelity is architecture-driving**: the CPU backend accumulates in f64 (the deterministic bit-exact anchor vs C++), the ROCm/HIP backend accumulates in f32 (~1e-6 tolerance) — this split is threaded through `runtime.rs`'s `Capabilities`/`ReducePath`/histogram-cell-type selection, not bolted on.

## Layers

**Foundation (`lgbm-core`):**
- Purpose: Single source of truth for `Config` (all training/prediction parameters, mirrors C++ `config.h`), shared numeric type aliases, error primitives, deterministic RNG.
- Location: `crates/lgbm-core/src/`
- Contains: `types.rs` (Config + numeric types), `error.rs`, `random.rs`.
- Depends on: `thiserror` only.
- Used by: every other crate in the workspace.

**Data layer (`lgbm-dataset`):**
- Purpose: Translate raw feature/label data into the binned columnar representation the rest of the engine operates on.
- Location: `crates/lgbm-dataset/src/`
- Contains: `bin_mapper.rs` (continuous→bin mapping), `feature_group.rs`, `multi_val_bin.rs` (sparse/grouped features), `efb.rs` (Exclusive Feature Bundling), `dataset.rs`/`ingest.rs`/`metadata.rs`.
- Depends on: `lgbm-core`.
- Used by: `lgbm-model`, `lgbm-compute`, `lgbm-objective`, `lgbm-metric`, `lgbm-treelearner`, `lgbm-boosting`, `lgbm`.

**Model layer (`lgbm-model`):**
- Purpose: Own the tree/ensemble data structures and model serialization/prediction, independent of training machinery.
- Location: `crates/lgbm-model/src/`
- Contains: `tree.rs` (node/split repr), `ensemble.rs` (`GbdtModel`), `format.rs`/`model_text.rs` (text/JSON I/O), `predict.rs`, `objective.rs` (`ObjectiveKind::convert_output`).
- Depends on: `lgbm-core`, `lgbm-dataset`.
- Used by: `lgbm-compute` (on-device grow-tree return type only), `lgbm-objective`, `lgbm-metric`, `lgbm-treelearner`, `lgbm-boosting`, `lgbm`, `lgbm-python`.

**Compute layer (`lgbm-compute`):**
- Purpose: The isolation seam for all CubeCL GPU/CPU-vector compute — histogram construction, split-gain evaluation, row partitioning, on-device tree growth, prediction, score updates.
- Location: `crates/lgbm-compute/src/`, kernels in `crates/lgbm-compute/src/kernels/`
- Contains: `lib.rs` (`trait Backend`, `BinColumn`, `DeviceFrontier`), `runtime.rs` (capability probe, `ReducePath`), `gain.rs` (`GainConfig`/`SplitInfo`), `kernels/{histogram,split,partition,data_partition,predict,grow_driver,score_updater,best_split,tree,...}.rs`.
- Depends on: `lgbm-core`, `lgbm-dataset`, `lgbm-model` (see DAG note above), `cubecl` (feature-gated `cpu`/`rocm`/`cuda`/`wgpu`), `rayon`.
- Used by: `lgbm-treelearner` (trait/types only, no runtime named), `lgbm-boosting` (trait bound + resident device score path), `lgbm`.

**Tree-growth layer (`lgbm-treelearner`):**
- Purpose: Grow one decision tree per call via best-first leaf-wise expansion, driving `Backend` ops for histograms/splits/partitioning.
- Location: `crates/lgbm-treelearner/src/`
- Contains: `learner.rs` (`SerialTreeLearner`), `data_partition.rs`, `histogram_pool.rs`/`resident_pool.rs`, `leaf_splits.rs`, `feature_histogram_categorical.rs`, `monotone_constraints.rs`, `col_sampler.rs`, `cost_effective_gradient_boosting.rs`, `gradient_discretizer.rs` (quantized training).
- Depends on: `lgbm-core`, `lgbm-dataset`, `lgbm-compute` (Backend trait only, CMP-01 containment preserved), `lgbm-model`, `rayon`.
- Used by: `lgbm-boosting`.

**Loss/eval layer (`lgbm-objective`, `lgbm-metric`):**
- Purpose: Compute training gradients/hessians and evaluation metrics from current scores.
- Location: `crates/lgbm-objective/src/`, `crates/lgbm-metric/src/`
- Depends on: `lgbm-core`, `lgbm-dataset`, `lgbm-model` (for `ObjectiveKind::convert_output`), `lgbm-metric` is also a dependency of `lgbm-objective` (shared `DcgCalculator` for ranking).
- Used by: `lgbm-boosting`.

**Ensemble layer (`lgbm-boosting`):**
- Purpose: Outer gradient-boosting loop — drive one `SerialTreeLearner` call per iteration, manage the ensemble, bagging/GOSS sampling, score updates, early stopping, DART/RF variants.
- Location: `crates/lgbm-boosting/src/`
- Contains: `gbdt.rs` (`Gbdt<'a>`, `train_one_iter<B: Backend>`, `train<B: Backend>`), `score_updater.rs`, `sample_strategy.rs` (bagging/GOSS), `early_stopping.rs`.
- Depends on: `lgbm-core`, `lgbm-dataset`, `lgbm-model`, `lgbm-treelearner`, `lgbm-objective`, `lgbm-metric`, `lgbm-compute` (trait bound + resident score path only).
- Used by: `lgbm`.

**Facade layer (`lgbm`):**
- Purpose: Single public import surface / user-facing API; the PyO3 binding target.
- Location: `crates/lgbm/src/`
- Contains: `booster.rs` (`Booster`, `DenseCorpus`, `RawCorpus`, `train*` functions), `builder.rs` (`TrainingBuilder`).
- Depends on: every layer crate below it, `lgbm-compute` (trait + `CpuBackend` anchor), `rayon`, optional `mimalloc`.
- Used by: `lgbm-python`, end users, examples in `crates/lgbm/examples/`.

**Binding layer (`lgbm-python`):**
- Purpose: PyO3 extension module mirroring the official `lightgbm` package API.
- Location: `crates/lgbm-python/src/`
- Contains: `booster.rs`, `dataset.rs`, `params.rs`, `callbacks.rs`, `marshal.rs`.
- Depends on: `lgbm`, `lgbm-core`, `lgbm-dataset`, `lgbm-model`, `mimalloc`, `pyo3`/`numpy` (version-pinned triangle, see crate Cargo.toml comments).

## Data Flow

### Primary Training Path

1. Raw rows/columns → `DenseCorpus`/`RawCorpus` construction (`crates/lgbm/src/booster.rs`), optionally feature-parallel binning via `build_feature_columns_from_raw*`.
2. Binning: `lgbm-dataset::BinMapper::FindBin`-equivalent produces `BinColumn` per feature (narrowest u8/u16/u32 width) (`crates/lgbm-dataset/src/bin_mapper.rs`).
3. `lgbm::train*` constructs a `Gbdt` (`crates/lgbm-boosting/src/gbdt.rs`) and calls `train_one_iter::<B: Backend>` per boosting round.
4. Per iteration: `lgbm-objective` computes gradients/hessians from current scores (`ObjectiveFunction::get_gradients`-equivalent).
5. `Gbdt` invokes `SerialTreeLearner::train` (`crates/lgbm-treelearner/src/learner.rs`) with the fixed gradients/hessians for that tree.
6. Inside the learner's best-first loop: `Backend::construct_histograms`/`subtract_histograms` (`lgbm-compute` kernels) → `fix_histogram` (raw-leaf-sum correction) → `find_best_split` (in-kernel gain scan) → `DataPartition::split` (row→leaf reorder) → `Tree::split` (grow node in `lgbm-model`).
7. `ScoreUpdater` (`crates/lgbm-boosting/src/score_updater.rs`) accumulates the new tree's contribution into running scores (optionally via the on-device resident score path when `boosting_on_cuda` is set).
8. `lgbm-metric` evaluates train/valid metrics; `early_stopping.rs` may halt the loop.
9. Result: `GbdtModel` ensemble of `Tree`s (`crates/lgbm-model/src/ensemble.rs`), wrapped in `Booster` (`crates/lgbm/src/booster.rs`).

### Prediction Path

1. `Booster::predict`/`predict_row`/`predict_raw_batch` (`crates/lgbm/src/booster.rs`) walk `GbdtModel`'s trees via `lgbm-model/src/predict.rs`.
2. Optional on-device batch prediction: `lgbm-compute/src/kernels/predict.rs` (`derive_leaf_map_device`), gated by the same `Backend` abstraction.
3. `ObjectiveKind::convert_output` (`lgbm-model/src/objective.rs`) applies the final link-function transform (sigmoid/softmax/etc.) shared by both prediction and probability-space metrics.

### Model Serialization

- Text/JSON dump and load: `crates/lgbm-model/src/model_text.rs`, `crates/lgbm-model/src/format.rs`.
- `Booster::model_to_string`/`save_model`/`model_from_string` (`crates/lgbm/src/booster.rs`) expose this at the facade layer.

**State Management:**
- Mutable training state lives in `Gbdt` (scores, models, sample-strategy state) and `SerialTreeLearner` (data partition, histogram pool, leaf splits) — mirrors the C++ split between `GBDT` and `TreeLearner` member state.
- `Dataset`/binned features are effectively immutable once built (mirrors C++ `Dataset::FinishLoad` read-only contract), though this is convention rather than an enforced Rust type-level invariant.

## Key Abstractions

**`Backend` trait (the CMP-01 seam):**
- Purpose: Every compute operation (histogram construct/subtract/fix, split-gain scan, row partition, tree grow, predict, score update) as a trait method, generic over a CubeCL runtime.
- Location: `crates/lgbm-compute/src/lib.rs:658`.
- Implementations: `CpuBackend` (cubecl-cpu, f64 deterministic anchor, default `cpu` feature) and a ROCm/HIP backend (cubecl-hip, f32, opt-in `rocm` feature).
- Pattern: compile-time generic monomorphization (`fn train<B: Backend>`), not runtime dynamic dispatch — replaces the C++ `Create*` string-keyed factory pattern.

**`BinColumn` (narrow columnar bins):**
- Purpose: Per-feature binned row values stored in the narrowest unsigned width for that feature's `num_bin` (mirrors C++ `DenseBin<uint8_t/uint16_t/uint32_t>`).
- Location: `crates/lgbm-compute/src/lib.rs` (`enum BinColumn { U8, U16, U32 }`), owned by `lgbm-compute` (the lowest crate touching the hot fold) and re-exported through `lgbm-treelearner`.
- Pattern: hot CPU fold reads the narrow type directly (monomorphic match, no per-element branch); cold paths (partition, bagging, GPU upload) widen via `bin()`/`iter_u32()`/`to_u32_vec()`.

**`GainConfig`/`SplitInfo`:**
- Purpose: Carry split-gain hyperparameters and computed best-split results across the `Backend` boundary without leaking cubecl types.
- Location: `crates/lgbm-compute/src/gain.rs`.

**`Capabilities`/`ReducePath` (CMP-03/CMP-04 capability gate):**
- Purpose: Probe the active runtime's feature support exactly once at startup and select the numerically-correct accumulation strategy.
- Location: `crates/lgbm-compute/src/runtime.rs`.
- Pattern: every backend-asymmetric feature (`has_plane`, `has_f64`, `has_f32_atomic`, `plane_size`) gated explicitly — never assumed present on both cubecl-cpu and cubecl-hip.

## Entry Points

**Facade training functions:**
- Location: `crates/lgbm/src/booster.rs` — `train`, `train_raw`, `train_with_valid`, `train_custom`, `train_custom_with_metric`, `train_custom_raw_with_metric`.
- Triggers: library consumers (Rust callers, examples in `crates/lgbm/examples/`, `lgbm-python` bindings).
- Responsibilities: construct `DenseCorpus`/`RawCorpus`, drive the boosting loop end-to-end, return a `Booster`.

**`Booster` prediction/serialization API:**
- Location: `crates/lgbm/src/booster.rs` (`impl Booster`) — `predict`, `predict_row`, `predict_raw_batch`, `refit`, `save_model`, `model_to_string`, `model_from_string`, `feature_importance_split`/`_gain`.
- Triggers: post-training inference/introspection calls.

**PyO3 module:**
- Location: `crates/lgbm-python/src/lib.rs`.
- Triggers: Python `import lightgbm`-equivalent via the built extension module (maturin).
- Responsibilities: mirror the official `lightgbm` Python API surface over the Rust facade.

**`xtask regen`:**
- Location: `xtask/src/main.rs`.
- Triggers: manual dev invocation (`cargo run -p xtask -- regen`) when C++ golden fixtures need regenerating.
- Responsibilities: the only workspace step requiring a C++ toolchain; deterministic (seeded, no wall-clock entropy).

## Architectural Constraints

- **Threading:** `rayon` data-parallelism at the CPU layer (feature-parallel binning in `lgbm`/`lgbm-objective`, block-parallel `DataPartition::split` in `lgbm-treelearner`, `Backend`-internal parallel folds in `lgbm-compute`); GPU/ROCm parallelism is expressed via CubeCL `#[cube]` kernels with `CubeDim`/`Plane` ops, confined entirely to `lgbm-compute`. No raw OS threads outside these two mechanisms.
- **Global state:** None of the module-level singletons the C++ engine relies on (`Common::Timer global_timer`, process-global `device_type`) appear to have direct Rust equivalents surfaced in the crate APIs explored; `Config` is passed explicitly rather than read from a process-global.
- **Compute containment (CMP-01):** No crate other than `lgbm-compute` may name a `cubecl` type or import `cubecl` directly — enforced by an in-crate guard test (`cmp01_containment`). This is the single most important architectural rule for anyone adding new compute-touching code: new kernels go in `lgbm-compute/src/kernels/`, and callers consume only `Backend` trait methods plus the re-exported `ComputeClient`/`Handle` types.
- **Numeric precision split:** CPU backend accumulates in f64 (deterministic bit-exact-vs-C++ anchor); ROCm/HIP backend accumulates in f32 (~1e-6 tolerance vs the CPU anchor, not directly vs C++). This split is threaded through `runtime.rs` and is a hard constraint on any new kernel — see `lgbm-compute/src/runtime.rs` capability gating.
- **Crate-cycle avoidance:** `lgbm-compute → lgbm-model` is a deliberate, explicitly-justified exception to the otherwise strictly layered DAG (needed only to name `Tree` as an on-device grow-tree return type); adding further upward edges from `lgbm-compute` risks reintroducing cycles and should be checked against the existing DAG before merging.
- **Dev-only golden fixtures:** `oracle-harness` is a dev-dependency of nearly every crate (parity testing against committed C++ goldens) but is never a runtime dependency — keep new parity fixtures/tests there, not in library crates.

## Anti-Patterns

### Depending on `cubecl` directly from a non-`lgbm-compute` crate

**What happens:** A crate above `lgbm-compute` in the DAG (e.g. `lgbm-treelearner`, `lgbm-boosting`, `lgbm`) adds `cubecl` to its own `Cargo.toml` or imports a `cubecl::*` type directly instead of going through `lgbm_compute::Backend`/re-exported types.
**Why it's wrong:** Breaks CMP-01 containment — the whole point of `lgbm-compute` is to confine the alpha-stage CubeCL API surface to one crate so it can evolve without leaking into the rest of the workspace. The `cmp01_containment` guard test exists specifically to catch this.
**Do this instead:** Add new operations as `Backend` trait methods in `crates/lgbm-compute/src/lib.rs`, implement the kernel in `crates/lgbm-compute/src/kernels/`, and consume it above via the trait or the already-re-exported `ComputeClient`/`Handle` types.

### Re-deriving gain math outside the kernel

**What happens:** Computing split gain from raw histogram sums in the tree-learner or elsewhere in host code, instead of using the value the `find_best_split` kernel already returns.
**Why it's wrong:** `learner.rs`'s own doc comments flag this explicitly (D-01a) — re-deriving gain host-side risks numerical drift from the kernel's in-GPU/in-kernel computation, threatening the ~1e-6/bit-exact parity contract that is this project's core value.
**Do this instead:** Treat `SplitInfo`/`GainConfig` values from `lgbm-compute` as the sole source of truth; only add `min_gain_to_split` back for the tree-model's stored `split_gain` field, never for split selection (see `learner.rs` "Keystone fidelity points").

## Error Handling

**Strategy:** Layered `thiserror`-based structured errors, one error enum per crate at the library boundary (`lgbm-core::error`, `lgbm-dataset::error`, `lgbm-model::error`, `lgbm-compute::error::ComputeError`, `lgbm-treelearner::error`, `lgbm-objective::error`, `lgbm-metric::error`, `lgbm-boosting::error`, `lgbm::error::LgbmError`), matching the project-wide convention (`thiserror` for domain errors, `anyhow` for app-layer ergonomics — used in `oracle-harness` and `xtask`).

**Patterns:**
- Each crate re-exports its own error type from the facade (`lgbm::error::LgbmError` aggregates `BoostingError`, `MetricError`, `ObjectiveError`, `TreeLearnerError`, etc. via `pub use` in `crates/lgbm/src/lib.rs`).
- `ComputeError` is the single error type crossing the `Backend` trait boundary, keeping compute failures typed without leaking cubecl error types upward.

## Cross-Cutting Concerns

**Logging:** Not evidenced by file layout scan; check individual crates for `log`/`tracing` usage if adding instrumentation (none of the crate `Cargo.toml`s explored show a logging dependency at the workspace level).
**Validation:** `Config` (`lgbm-core`) is the single source of truth for parameter validation, mirroring the C++ `config.h` single-source-of-truth convention noted in project docs.
**Parity/testing infrastructure:** `oracle-harness` (comparator + fixtures) is a cross-cutting dev-dependency threaded through nearly every crate — new numerically-sensitive code should add a parity test there rather than inventing a new comparison mechanism.

---

*Architecture analysis: 2026-07-09*
