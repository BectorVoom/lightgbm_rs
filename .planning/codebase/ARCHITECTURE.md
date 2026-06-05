<!-- refreshed: 2026-06-05 -->
# Architecture

**Analysis Date:** 2026-06-05

> **Scope note:** This document maps the **LightGBM C++ REFERENCE implementation** located under `LightGBM/` (Microsoft's reference C++ core), which is the system being ported to Rust + cubecl. It does NOT describe the Rust crate under development. All paths are relative to the repo root and live under the `LightGBM/` prefix. GPU-relevant subsystems (candidate cubecl kernels) are flagged inline with **[GPU-RELEVANT]**.

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          ENTRY / API LAYER                                │
├──────────────────────┬───────────────────────┬───────────────────────────┤
│   CLI Application     │      C API            │      Predictor (CLI)      │
│ `src/application/`    │   `src/c_api.cpp`     │ `src/application/         │
│ `application.cpp`     │   (Python/R/etc.)     │   predictor.hpp`          │
│ + `src/main.cpp`      │                       │                           │
└──────────┬───────────┴───────────┬───────────┴───────────┬───────────────┘
           │                       │                        │
           ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       BOOSTING LAYER (ensemble loop)                      │
│  Boosting interface `include/LightGBM/boosting.h`                         │
│  GBDT `src/boosting/gbdt.cpp` | DART `dart.hpp` | RF `rf.hpp`             │
│  SampleStrategy (bagging/GOSS) `bagging.hpp` `goss.hpp`                   │
│  ScoreUpdater `score_updater.hpp`                                         │
└──────┬──────────────────────┬──────────────────────────┬─────────────────┘
       │ gradients/hessians   │ Train() one tree         │ score updates
       ▼                      ▼                          ▼
┌──────────────────┐  ┌───────────────────────────┐  ┌────────────────────┐
│ ObjectiveFunction│  │     TreeLearner           │  │      Metric        │
│ `src/objective/` │  │  `src/treelearner/`       │  │  `src/metric/`     │
│ grad/hess  [GPU] │  │  serial / parallel / GPU  │  │ eval  [GPU]        │
└──────────────────┘  └────────────┬──────────────┘  └────────────────────┘
                                    │ histograms + split finding [GPU]
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DATA / IO LAYER                                     │
│  Dataset `src/io/dataset.cpp` | FeatureGroup `include/.../feature_group.h`│
│  Bin/BinMapper `src/io/bin.cpp` `dense_bin.hpp` `sparse_bin.hpp` [GPU]    │
│  Metadata `src/io/metadata.cpp` | DatasetLoader `src/io/dataset_loader.cpp`│
│  Tree (model) `src/io/tree.cpp`                                           │
└─────────────────────────────────────────────────────────────────────────┘
           │                                          ▲
           ▼                                          │
┌──────────────────────────┐          ┌──────────────────────────────────┐
│   Network (distributed)  │          │   Model output / prediction      │
│   `src/network/`         │          │   `src/boosting/gbdt_prediction` │
│   MPI / socket allreduce │          │   `gbdt_model_text.cpp`          │
└──────────────────────────┘          └──────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `Application` | CLI orchestration: parse params, load data, init train/predict | `src/application/application.cpp` |
| `Boosting` / `GBDT` | Ensemble gradient-boosting loop; owns models, score updaters, sample strategy | `include/LightGBM/boosting.h`, `src/boosting/gbdt.cpp` |
| `ObjectiveFunction` | Compute per-row gradients & hessians from current scores **[GPU-RELEVANT]** | `include/LightGBM/objective_function.h`, `src/objective/*.hpp` |
| `TreeLearner` | Grow one decision tree per call: histograms, split finding, partition **[GPU-RELEVANT]** | `include/LightGBM/tree_learner.h`, `src/treelearner/serial_tree_learner.cpp` |
| `Dataset` | Owns binned feature data (FeatureGroups), metadata; builds histograms | `include/LightGBM/dataset.h`, `src/io/dataset.cpp` |
| `Bin` / `BinMapper` | Feature binning (continuous→bin), per-bin histogram accumulation **[GPU-RELEVANT]** | `include/LightGBM/bin.h`, `src/io/bin.cpp`, `src/io/dense_bin.hpp` |
| `Metric` | Evaluate loss/metric on train & validation scores **[GPU-RELEVANT]** | `include/LightGBM/metric.h`, `src/metric/*.hpp` |
| `Tree` | Single decision tree model: nodes, splits, leaf outputs, prediction | `include/LightGBM/tree.h`, `src/io/tree.cpp` |
| `Predictor` | Batch prediction driver over a file/dataset | `src/application/predictor.hpp` |
| `Network` | Distributed allreduce/allgather over MPI or sockets | `include/LightGBM/network.h`, `src/network/network.cpp` |
| C API | Stable ABI exposing Dataset/Booster to bindings | `include/LightGBM/c_api.h`, `src/c_api.cpp` |

## Pattern Overview

**Overall:** Layered, interface-driven gradient boosting engine using the **Strategy / Factory** pattern at every major seam.

**Key Characteristics:**
- Every major subsystem is an **abstract base class with a static `Create*` factory** that switches on a string type + `device_type` (`cpu`/`gpu`/`cuda`). This is the primary porting seam — the Rust port replaces each factory + implementation.
- **Template-based parallel learners**: parallel tree learners (`FeatureParallelTreeLearner<T>`, `DataParallelTreeLearner<T>`, `VotingParallelTreeLearner<T>`, `LinearTreeLearner<T>`) are templated over a base serial/GPU learner — see `src/treelearner/tree_learner.cpp`.
- **Histogram-based split finding** is the core algorithm: data is pre-binned once into integer bins; per-node histograms of (sum_gradient, sum_hessian) are accumulated over bins, and splits are found by scanning histograms — not by sorting raw values.
- **Histogram subtraction trick**: the larger child's histogram is derived by subtracting the smaller child's histogram from the parent's (`use_subtract` in `ConstructHistograms`).
- **Device abstraction is compile-time + runtime**: CPU implementations are the default; `gpu` (OpenCL, `src/treelearner/ocl/*.cl`) and `cuda` (`.cu` files) are alternative backends guarded by `USE_GPU`/`USE_CUDA` macros.

## Layers

**Entry / API Layer:**
- Purpose: Translate external requests (CLI args, C ABI calls) into engine operations.
- Location: `src/application/`, `src/c_api.cpp`, `src/main.cpp`
- Contains: Argument parsing, `Application::Run()` dispatch, ~120 `LGBM_*` C functions.
- Depends on: Boosting, Dataset, Metric, Objective layers.
- Used by: CLI binary, language bindings (Python/R/etc.).

**Boosting Layer:**
- Purpose: Run the outer gradient-boosting loop, manage the ensemble, bagging, scores, early stopping.
- Location: `src/boosting/`
- Contains: `GBDT`, `DART`, `RF`, `GOSS`/bagging sample strategies, score updaters, prediction.
- Depends on: ObjectiveFunction (grad/hess), TreeLearner (grow tree), Metric (eval).
- Used by: Application, C API.

**Objective / Metric Layer:**
- Purpose: Define the loss (gradients/hessians) and evaluation metrics.
- Location: `src/objective/`, `src/metric/` (header-only `.hpp` implementations + `.cpp` factories).
- Depends on: Dataset metadata (labels, weights, query boundaries).
- Used by: Boosting layer.

**TreeLearner Layer:**
- Purpose: Grow one tree: build feature histograms, find best splits, partition data, set leaf outputs.
- Location: `src/treelearner/`
- Depends on: Dataset (binned data + histogram construction), Network (parallel learners).
- Used by: Boosting layer.

**Data / IO Layer:**
- Purpose: Load raw data, bin features, store binned representation, hold the tree model.
- Location: `src/io/`
- Depends on: utils, network (distributed loading).
- Used by: all upper layers.

**Network Layer:**
- Purpose: Collective communication for distributed training.
- Location: `src/network/` (MPI `linkers_mpi.cpp`, socket `linkers_socket.cpp`).

## Data Flow

### Training Path (raw data → ensemble)

1. **Parse & load** — `Application::LoadParameters` then `Application::LoadData` (`src/application/application.cpp:50,88`) build a `DatasetLoader` (`src/io/dataset_loader.cpp`).
2. **Bin features** — `DatasetLoader::LoadFromFile` samples columns, constructs `BinMapper`s (`src/io/bin.cpp`, `BinMapper::FindBin`), and packs raw values into integer bins inside `FeatureGroup`s (`include/LightGBM/feature_group.h`). Result: a `Dataset` of binned columns (`src/io/dataset.cpp`).
3. **Init engine** — `Application::InitTrain` (`application.cpp:168`) creates Boosting (`Boosting::CreateBoosting`), ObjectiveFunction (`ObjectiveFunction::CreateObjectiveFunction`), and Metrics; `GBDT::Init` (`src/boosting/gbdt.cpp:53`) creates the `TreeLearner` via `TreeLearner::CreateTreeLearner` (`src/treelearner/tree_learner.cpp:15`).
4. **Boosting loop** — `GBDT::Train` (`gbdt.cpp:237`) loops `num_iterations`, each calling `GBDT::TrainOneIter` (`gbdt.cpp:344`).
5. **Compute gradients/hessians** — `GBDT::Boosting()` (`gbdt.cpp:220`) calls `objective_function_->GetGradients(score, gradients, hessians)`. **[GPU-RELEVANT]**
6. **Bagging / subsampling** — `data_sample_strategy_->Bagging(...)` (`src/boosting/bagging.hpp`, GOSS `src/boosting/goss.hpp`).
7. **Grow one tree per class** — `tree_learner_->Train(grad, hess, is_first_tree)` (`gbdt.cpp:403` → `SerialTreeLearner::Train` `src/treelearner/serial_tree_learner.cpp:179`). Inside one tree:
   - `BeforeTrain()` resets histogram pool & root leaf sums (`serial_tree_learner.cpp:288`).
   - Leaf-wise growth loop (`serial_tree_learner.cpp:218`): for each split up to `num_leaves-1`:
     - `ConstructHistograms(...)` (`serial_tree_learner.cpp:404`) → `Dataset::ConstructHistograms` (`include/LightGBM/dataset.h:727`) → per-`Bin` `ConstructHistogram*` (`include/LightGBM/bin.h:350`). **[GPU-RELEVANT — hottest kernel]**
     - `FindBestSplitsFromHistograms(...)` (`serial_tree_learner.cpp:473`) → `FeatureHistogram::FindBestThreshold` (`src/treelearner/feature_histogram.cpp`, `feature_histogram.hpp`). **[GPU-RELEVANT]**
     - `ArrayArgs<SplitInfo>::ArgMax` picks the global best leaf (`serial_tree_learner.cpp:225`).
     - `Split(...)` updates the `Tree` and re-partitions data (`src/treelearner/data_partition.hpp`). **[GPU-RELEVANT]**
8. **Leaf output & shrinkage** — `RenewTreeOutput`, `Tree::Shrinkage(learning_rate)` (`gbdt.cpp:410-413`).
9. **Update scores** — `GBDT::UpdateScore` (`gbdt.cpp:491`) adds the new tree's predictions into `train_score_updater_` (and OOB / validation updaters).
10. **Eval & early stopping** — `GBDT::EvalAndCheckEarlyStopping` (`gbdt.cpp:472`) runs metrics; loop ends or continues.
11. **Append model** — tree pushed into `models_` (`gbdt.cpp:437`); ensemble grows.

### Prediction Path (model → scores)

1. `Predictor` constructed (`src/application/predictor.hpp:30`); calls `boosting_->InitPredict`.
2. Per row, `Predictor` selects a function (raw / transformed / leaf-index / contrib) and calls `GBDT::PredictRaw` / `Predict` (`src/boosting/gbdt_prediction.cpp:13,55`).
3. `GBDT::PredictRaw` iterates `models_` summing `Tree::Predict(features)` (`src/io/tree.cpp`), with optional prediction early-stop (`src/boosting/prediction_early_stop.cpp`).
4. `objective_function_->ConvertOutput` applies sigmoid/softmax for transformed predictions.

### Model Serialization

- Text/JSON dump and load: `src/boosting/gbdt_model_text.cpp`, tree text in `src/io/tree.cpp`; if-else C++ codegen in the same files.

**State Management:**
- Mutable training state lives in `GBDT` members (`models_`, `gradients_`, `hessians_`, score updaters) and in `TreeLearner` (`data_partition_`, histogram pool, `leaf_splits_`). `Dataset` is immutable after `FinishLoad`. Scores are accumulated in `ScoreUpdater` (`src/boosting/score_updater.hpp`).

## Key Abstractions

**Boosting (`include/LightGBM/boosting.h`):**
- Purpose: Ensemble strategy. `TrainOneIter`, `Predict*`, model I/O.
- Implementations: `GBDT` (`src/boosting/gbdt.cpp`), `DART` (`src/boosting/dart.hpp`), `RF` (`src/boosting/rf.hpp`); `GBDTBase` adds leaf get/set.
- Factory: `Boosting::CreateBoosting(type, filename)` (`src/boosting/boosting.cpp:34`).

**ObjectiveFunction (`include/LightGBM/objective_function.h`): [GPU-RELEVANT]**
- Purpose: `GetGradients(score, gradients, hessians)`; `ConvertOutput`, `BoostFromScore`.
- Implementations (header-only): regression (`src/objective/regression_objective.hpp`), binary (`binary_objective.hpp`), multiclass (`multiclass_objective.hpp`), ranking/lambdarank (`rank_objective.hpp`), cross-entropy (`xentropy_objective.hpp`); CUDA mirrors in `src/objective/cuda/`.
- Factory: `ObjectiveFunction::CreateObjectiveFunction(type, config)` (`src/objective/objective_function.cpp`).

**TreeLearner (`include/LightGBM/tree_learner.h`): [GPU-RELEVANT]**
- Purpose: `Train(gradients, hessians, is_first_tree) → Tree*`.
- Implementations: `SerialTreeLearner` (`src/treelearner/serial_tree_learner.cpp`), `GPUTreeLearner` (OpenCL, `src/treelearner/gpu_tree_learner.cpp`), `CUDASingleGPUTreeLearner` (`src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp`), and templated parallel wrappers.
- Factory: `TreeLearner::CreateTreeLearner(learner_type, device_type, config, boosting_on_cuda)` (`src/treelearner/tree_learner.cpp:15`).

**Dataset / Bin (`include/LightGBM/dataset.h`, `include/LightGBM/bin.h`): [GPU-RELEVANT]**
- Purpose: Binned columnar storage + histogram construction. `BinType` (numerical/categorical), `BinMapper::FindBin`, `Bin::ConstructHistogram*`, `MultiValBin` for grouped features.
- Implementations: `DenseBin` (`src/io/dense_bin.hpp`), `SparseBin` (`src/io/sparse_bin.hpp`), `MultiValDenseBin`/`MultiValSparseBin`.

**Metric (`include/LightGBM/metric.h`): [GPU-RELEVANT]**
- Purpose: `Eval(score, objective) → vector<double>`. Plus static `DCGCalculator` for NDCG.
- Implementations: regression/binary/multiclass/rank/xentropy metrics in `src/metric/*.hpp`; CUDA in `src/metric/cuda/`.

## Entry Points

**CLI binary (`src/main.cpp`):**
- Location: `src/main.cpp` → `LightGBM::Application(argc, argv).Run()`.
- Triggers: command line invocation (`task=train|predict|convert_model|refit`).
- Responsibilities: top-level exception handling, MPI finalize.

**C API (`src/c_api.cpp`, `include/LightGBM/c_api.h`):**
- Location: ~120 `LGBM_Dataset*` and `LGBM_Booster*` functions (e.g. `LGBM_DatasetCreateFromMat:1299`, `LGBM_BoosterCreate:1939`, `LGBM_BoosterUpdateOneIter`, `LGBM_BoosterPredictForMat`).
- Triggers: language bindings.
- Responsibilities: stable ABI, handle lifetime, thread-safety wrappers; the primary surface the Rust crate must reproduce or FFI-bridge.

## Architectural Constraints

- **Threading:** Shared-memory parallelism via **OpenMP** (`#pragma omp parallel for num_threads(OMP_NUM_THREADS())`) pervasively in the boosting loop, histogram construction, and prediction. `src/utils/openmp_wrapper.cpp` and `include/LightGBM/utils/openmp_wrapper.h` provide the wrapper. The Rust port must map this onto rayon / cubecl kernels.
- **Device backends:** Selected at runtime by `config_->device_type` but gated at compile time by `USE_GPU` (OpenCL) and `USE_CUDA` macros. `LGBM_config_::current_device` / `current_learner` are process-global (`src/boosting/gbdt.cpp:24`).
- **Global state:** `Common::Timer global_timer` (`src/boosting/gbdt.cpp:22`); static `DCGCalculator` tables (`include/LightGBM/metric.h:133`); process-global device config. These are module-level singletons.
- **Histogram pool memory:** Tree learners use a fixed-size histogram pool with the subtraction trick; memory is sized by `num_leaves` × total bins.
- **Immutability:** `Dataset` is read-only after `FinishLoad()`; histograms and partitions are the only per-tree mutable structures.

## Anti-Patterns

### Heavy template instantiation across device/quantization axes

**What happens:** `Dataset::ConstructHistograms` and `Bin::ConstructHistogram*` are instantiated across `<USE_INDICES, USE_HESSIAN, USE_QUANT_GRAD, HIST_BITS>` combinations (`include/LightGBM/dataset.h:727`, `src/treelearner/serial_tree_learner.cpp:404`).
**Why it's wrong here:** Direct 1:1 translation to Rust generics would explode monomorphization and obscure the kernel. 
**Do this instead:** In the cubecl port, express histogram construction as a small set of parametric kernels keyed by bit-width and a runtime `use_indices` branch, not a combinatorial template matrix.

### Stringly-typed factories

**What happens:** Subsystem selection is by raw `std::string` compares on `type`/`device_type`/`tree_learner` (`src/treelearner/tree_learner.cpp:15`, `src/boosting/boosting.cpp:34`).
**Why it's wrong here:** No compile-time exhaustiveness; typos surface only at runtime.
**Do this instead:** Use Rust enums + `match` for boosting/objective/learner/device selection, parsed once at config time.

### `#ifdef USE_CUDA` scattered through hot paths

**What happens:** Device branching is interleaved into otherwise-portable logic (e.g. `GBDT::UpdateScore` `src/boosting/gbdt.cpp:500-508`).
**Why it's wrong here:** Couples CPU and GPU control flow, hard to test in isolation.
**Do this instead:** Hide device dispatch behind the `TreeLearner`/`ScoreUpdater` trait boundary so the boosting loop is device-agnostic.

## Error Handling

**Strategy:** Fatal-on-error via `Log::Fatal(...)` (throws / aborts) rather than recoverable `Result` types. See `include/LightGBM/utils/log.h`. C API wraps calls in `API_BEGIN()/API_END()` macros that catch exceptions and return error codes (`src/c_api.cpp`).

**Patterns:**
- `CHECK_*` macros (`CHECK_EQ`, `CHECK_NOTNULL`, `CHECK_GT`) assert invariants and fatal on failure.
- `OMP_INIT_EX()/OMP_LOOP_EX_BEGIN()/OMP_THROW_EX()` propagate exceptions out of OpenMP parallel regions.

## Cross-Cutting Concerns

**Logging:** `Log::Info/Debug/Warning/Fatal` (`include/LightGBM/utils/log.h`), verbosity from config.
**Validation:** `CHECK_*` macros and `Config::Set` parameter validation (`src/io/config.cpp`, `config_auto.cpp`).
**Timing/Profiling:** `Common::Timer` + `Common::FunctionTimer` (`global_timer`) instrument every hot method.
**Parallelism:** OpenMP threads (shared mem) + Network allreduce (distributed) + optional GPU/CUDA backends.

---

*Architecture analysis: 2026-06-05*
