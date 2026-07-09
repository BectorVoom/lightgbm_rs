# LightGBM C++ Reference — Comprehensive Design Document (CPU / Non-GPU)

> **Scope.** This document describes the architecture, pipeline, file dependencies, and
> input/output types of Microsoft's **LightGBM** C++ engine as it lives under `LightGBM/`
> in this repository. It deliberately **excludes** the GPU backends (CUDA `.cu` files,
> OpenCL `.cl` kernels, and the `src/**/cuda/` trees). It documents the CPU code paths only —
> which are the deterministic reference that the Rust port must reproduce.
>
> **Audience.** Written so that a reader with **no prior knowledge of LightGBM** — and only a
> general software-engineering background — can understand the whole system. Machine-learning
> jargon is explained the first time it appears.

---

## 0. What LightGBM Is (in one page)

**LightGBM** ("Light Gradient Boosting Machine") is a library that trains a **predictive model**
from tabular data. You give it a table of rows (examples) and columns (features), plus a target
value per row (the *label* — e.g. a price to predict, or a yes/no class), and it learns a model
that predicts the label for new, unseen rows.

The model it builds is a **gradient-boosted decision tree ensemble (GBDT)**. Two ideas combine:

1. **Decision tree.** A flowchart of yes/no questions about a row's features
   ("is feature 7 < 3.2?"). Each leaf of the flowchart holds a numeric output. To predict, you
   walk the row down the tree to a leaf and read off its value.

2. **Gradient boosting.** One tree is weak. So LightGBM builds *many* trees in sequence, where
   each new tree is trained to correct the errors of all the trees built so far. The final
   prediction is the **sum** of all trees' outputs (scaled by a small *learning rate*). "Gradient"
   means each tree is fit to the **gradient** (slope) of the loss function — a mathematical measure
   of how wrong the current predictions are — so every tree takes a step that most reduces error.

What makes LightGBM *fast* is the **histogram-based** algorithm:

- Before training, every continuous feature value is **binned** once into a small integer bucket
  (typically ≤ 255 bins). A feature value like `3.14159` becomes bin `#42`. This is done a single
  time and never repeated.
- To decide where to split a tree node, LightGBM builds a **histogram**: for each of a feature's
  bins it sums up two per-row quantities — the **gradient** and the **hessian** (the first and
  second derivatives of the loss). It then scans the histogram bin-by-bin to find the split point
  that yields the largest *gain* (error reduction). Scanning ≤ 255 bins is far cheaper than sorting
  and scanning every raw value.
- A further trick: after a node is split into two children, the **larger child's histogram is
  obtained by subtracting the smaller child's histogram from the parent's** — so only the smaller
  child is built from scratch. This "histogram subtraction trick" roughly halves the work.

LightGBM grows trees **leaf-wise** (best-first): it always splits the single leaf that promises the
biggest gain next, up to a `num_leaves` budget — as opposed to the level-by-level growth used by
some other libraries. This tends to produce more accurate trees for a given number of leaves.

---

## 1. The Big Picture — Layered Architecture

LightGBM is organized as a stack of layers. Higher layers depend on lower ones; data becomes
progressively more "digested" as it moves down and results flow back up.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ENTRY / API LAYER                                                    │
│  main.cpp → Application (CLI)   |   c_api.cpp (LGBM_* C ABI)          │  §A
│  Turns CLI args / C-API calls into engine operations.                 │
└───────────────┬───────────────────────────────────────────────────────┘
                │ builds & drives
┌───────────────▼───────────────────────────────────────────────────────┐
│  BOOSTING LAYER   (src/boosting/)                                      │
│  GBDT / DART / RF — the outer loop over boosting iterations.           │  §D
│  Owns the tree ensemble, current scores, bagging, early stopping.      │
└───────┬─────────────────────┬───────────────────────┬─────────────────┘
        │ grad/hess            │ grow one tree          │ evaluate
┌───────▼────────┐   ┌─────────▼──────────┐   ┌─────────▼──────────┐
│ OBJECTIVE      │   │ TREE LEARNER       │   │ METRIC             │
│ src/objective/ │   │ src/treelearner/   │   │ src/metric/        │
│ loss → grad,   │   │ histograms →       │   │ score → quality    │
│ hessian        │   │ best split → tree  │   │ numbers (for logs, │
│ §F             │   │ §E                 │   │ early stopping) §G │
└───────┬────────┘   └─────────┬──────────┘   └─────────┬──────────┘
        │                      │ needs binned data / builds histograms
        └──────────────┬───────┴───────────────────────┘
┌──────────────────────▼─────────────────────────────────────────────────┐
│  DATA / IO LAYER   (src/io/)                                            │
│  Dataset (immutable binned columnar store) + BinMapper + FeatureGroup   │  §C
│  + Metadata (labels/weights/queries) + Parser + Tree model I/O.         │
└──────────────────────┬─────────────────────────────────────────────────┘
┌──────────────────────▼─────────────────────────────────────────────────┐
│  CROSS-CUTTING: Config (all parameters), meta.h (core types),          │  §B
│  utils/ (logging, threading, RNG, file I/O, JSON), Network (§H, distrib)│  §B,§H
└──────────────────────────────────────────────────────────────────────────┘
```

Section map for the rest of this document:

| § | Subsystem | Directory | Role |
|---|-----------|-----------|------|
| A | Entry & Public API | `src/application/`, `src/main.cpp`, `src/c_api.cpp` | CLI + stable C ABI |
| B | Config, Meta types, Utilities | `include/LightGBM/config.h`, `meta.h`, `utils/` | parameters, core typedefs, helpers |
| C | Data / Dataset / Binning / Tree model | `src/io/` | binned storage, parsing, model text |
| D | Boosting (GBDT ensemble) | `src/boosting/` | outer training loop |
| E | Tree Learner | `src/treelearner/` | grow one tree via histograms |
| F | Objective Functions | `src/objective/` | loss → gradient/hessian |
| G | Metrics | `src/metric/` | evaluate model quality |
| H | Network (distributed) | `src/network/` | collective comm (optional) |

---

## 2. Core Type Vocabulary (read this before the sections)

These type aliases (defined in `include/LightGBM/meta.h`) appear everywhere. Memorizing them makes
every signature below readable.

| Alias | Underlying type | Meaning |
|-------|-----------------|---------|
| `data_size_t` | `int32_t` (signed!) | a row index or a count of rows/examples |
| `score_t` | `float` (default) | a gradient, hessian, or model score value |
| `label_t` | `float` (default) | a training label value |
| `comm_size_t` | `int32_t` | a byte/element count in network communication |
| `hist_t` | `double` | one accumulator cell in a histogram (sum of grad or hess) |
| `BinMapper` | class | maps a raw feature value → integer bin index |
| `DatasetHandle` / `BoosterHandle` | `void*` | opaque C-API pointers ("handles") to engine objects |

- **Handle**: in the C API, engine objects are hidden behind `void*` pointers called *handles*, so
  other languages (Python, R) can hold and pass them without knowing the C++ layout.
- **ABI** (Application Binary Interface): the stable, C-callable calling convention the bindings
  rely on; provided by `c_api.cpp` / `c_api.h`.
- **Gradient / Hessian**: first and second derivatives of the loss with respect to a row's current
  prediction. The tree learner uses their per-bin sums to choose splits and set leaf values.

---

## 3. End-to-End Pipelines

### 3.1 Training path (raw data → model)

```
CLI args / C-API call
   → Config parsed (§B): strings → Config struct (num_leaves, objective, num_iterations, …)
   → DatasetLoader (§C): read file/matrix → Parser → fit BinMapper per feature
       → bin every value → immutable Dataset (columnar, binned) + Metadata (labels, weights)
   → GBDT::Init (§D): create ObjectiveFunction (§F), TreeLearner (§E), Metric[] (§G)
   → BoostFromScore: initialize scores to a sensible constant
   → for iter in 1..num_iterations:                    ← the boosting loop (§D)
        1. ObjectiveFunction::GetGradients(scores) → gradients[], hessians[]   (§F)
        2. (optional) SampleStrategy: bagging / GOSS subsamples rows           (§D)
        3. TreeLearner::Train(gradients, hessians) → one new Tree              (§E)
             · build per-feature histograms of (Σgrad, Σhess) over bins
             · scan histograms → best SplitInfo (feature, threshold, gain)
             · partition rows into children; repeat until num_leaves reached
             · set leaf output values
        4. ScoreUpdater: add new tree's predictions into running scores        (§D)
        5. Metric::Eval on train/valid → log; check early stopping             (§G)
   → GBDT::SaveModelToString / text dump (§C, §D): serialized model
```

### 3.2 Prediction path (model → scores)

```
Load model text → GBDT ensemble of Trees (§C/§D)
   → for each input row:
        raw features → (Predictor maps them; trees store split thresholds in raw units)
        walk row down each Tree to a leaf → sum leaf outputs × shrinkage   (§C Tree::Predict, §D)
        ObjectiveFunction::ConvertOutput: raw score → final output
            (e.g. sigmoid for binary probability, exp for poisson)          (§F)
   → optional prediction-early-stop to skip remaining trees when confident  (§A)
```

### 3.3 Model serialization
- **Text / JSON dump & load**: `src/boosting/gbdt_model_text.cpp` (ensemble framing) and
  `src/io/tree.cpp` (per-tree node/threshold/leaf text). This is the canonical portable format.
- **If-else C++ codegen**: the same files can emit the model as compilable C++ for embedded use.

---

## 4. Reading Guide

Each subsystem section (A–H) below follows the same template:
**Purpose → Key classes/functions (with C++ signatures) → Inputs/Outputs (concrete types) →
Dependencies → Place in the pipeline.** File references are given as `path:line` and are clickable.

---

## Table of Contents — Subsystem Sections

- [Section A — Entry Point & Public API Layer](#section-a-entry-point-public-api-layer)
- [Section B — Configuration, Meta Types & Utilities](#section-b-configuration-meta-types-utilities)
- [Section C — Data I/O, Dataset, Binning & Tree Model Storage](#section-c-data-i-o-dataset-binning-tree-model-storage)
- [Section D — Boosting — the GBDT Ensemble Layer](#section-d-boosting-the-gbdt-ensemble-layer)
- [Section E — Tree Learner (CPU) — Growing One Tree](#section-e-tree-learner-cpu-growing-one-tree)
- [Section F — Objective Functions (Loss → Gradient/Hessian)](#section-f-objective-functions-loss-gradient-hessian)
- [Section G — Metrics (Model Evaluation)](#section-g-metrics-model-evaluation)
- [Section H — Distributed Network / Collective Communication](#section-h-distributed-network-collective-communication)

---

## Section A — Entry Point & Public API Layer

This section documents the outermost layer of Microsoft's LightGBM: the code a *user* touches first. There are two distinct entry surfaces:

1. **The CLI executable** (`lightgbm`), driven by `main.cpp` → `Application`. You give it a config file / command-line flags and it trains or predicts, reading and writing files.
2. **The C API** (`c_api.cpp` / `c_api.h`), a stable set of ~110 `extern "C"` functions (`LGBM_*`) that language bindings (Python, R, Java, the Rust port) call to build datasets, train boosters, and predict *in memory*.

Both surfaces sit *on top of* the engine layers (Boosting, Dataset, Objective, Metric, TreeLearner). They translate external requests (argv strings, C pointers) into engine calls and marshal results back out.

#### Jargon primer (used throughout)

- **Handle**: an opaque `void*` the caller holds but cannot dereference. It points to a C++ object the library owns. The caller passes it back to later calls; a matching `*Free` call destroys it. This is how a C ABI hides C++ objects. LightGBM defines `DatasetHandle`, `BoosterHandle`, `FastConfigHandle`, `ByteBufferHandle` (`c_api.h:30-33`), all `typedef void*`.
- **ABI (Application Binary Interface)**: the binary-level calling contract (symbol names, argument layout) that lets a program written in one language call a compiled library written in another. `extern "C"` disables C++ name-mangling so the symbols are C-callable.
- **Booster**: the trained model object — the gradient-boosting ensemble of decision trees. Wraps the engine's `Boosting`/`GBDT`.
- **Objective (function)**: defines the loss being minimized; it produces per-row *gradients* and *Hessians* (first/second derivatives of the loss) that drive each boosting round.
- **Dataset**: the binned, columnar training/validation data plus metadata (labels, weights, groups).
- **`data_size_t`**: the project's row-index integer type, `typedef int32_t data_size_t` (deliberately signed). **`score_t`**: gradient/score type, `float` by default.

---

### `src/main.cpp` — CLI process entry point

**Purpose.** The `int main(int argc, char** argv)` of the standalone `lightgbm` executable. It is deliberately tiny: construct an `Application`, run it, and provide top-level exception handling so any thrown error prints to `stderr` and the process exits non-zero.

**Key function:**
- `int main(int argc, char** argv)` (`main.cpp:13`). Constructs `LightGBM::Application app(argc, argv)` (`main.cpp:16`) then calls `app.Run()` (`main.cpp:17`).

**Control flow / error handling.**
- A `bool success` flag defaults to `false` and is set `true` only after `Run()` returns cleanly (`main.cpp:23`).
- Three `catch` blocks handle `const std::exception&` (`main.cpp:25`), `const std::string&` (thrown by some LightGBM code paths, `main.cpp:29`), and `catch(...)` for anything else (`main.cpp:33`); each prints `"Met Exceptions:"` + the message to `std::cerr`.
- If `!success`, calls `exit(-1)` (`main.cpp:42`).
- **MPI note (out of scope but present):** guarded by `#ifdef USE_MPI`, it calls `Linkers::MpiFinalizeIfIsParallel()` after a clean run (`main.cpp:20`) and `Linkers::MpiAbortIfIsParallel()` on failure (`main.cpp:39`).

**Dependencies.** Includes only `<LightGBM/application.h>` (`main.cpp:5`), `<iostream>`, and conditionally `network/linkers.h`.

**Inputs/outputs.** In: `int argc`, `char** argv` (raw command line). Out: process exit code (`0` implicit success, `-1` on error).

---

### `include/LightGBM/application.h` + `src/application/application.cpp` — CLI orchestration

**Purpose.** `Application` is "the main entrance of LightGBM" (per its own doc comment, `application.h:22-28`). It owns the whole CLI lifecycle: parse parameters, load data, build the engine objects, and dispatch to *train*, *predict*, or *convert-model* logic. Everything it does, the C API does too, just piecemeal.

#### Class `Application` (`application.h:29`)

**Public interface:**
- `Application(int argc, char** argv)` (`application.h:31`, defined `application.cpp:31`) — constructor. Calls `LoadParameters(argc, argv)` (`application.cpp:32`), sets OpenMP thread count via `OMP_SET_NUM_THREADS(config_.num_threads)` (`application.cpp:34`), fatally errors if no data is given for a non-convert task (`application.cpp:35-37`), and sets the global CUDA device flag if `device_type == "cuda"` (`application.cpp:39-41`, out of scope).
- `~Application()` (`application.h:34`, `application.cpp:44`) — calls `Network::Dispose()` if `config_.is_parallel`.
- `inline void Run()` (`application.h:78-88`) — **the task dispatcher**, defined inline in the header:
  ```cpp
  if (task == kPredict || task == KRefitTree) { InitPredict(); Predict(); }
  else if (task == kConvertModel)             { ConvertModel(); }
  else                                        { InitTrain();  Train(); }
  ```
  `task` is `config_.task`, a `TaskType` enum from the config.

**Private methods (all `void`, defined in the .cpp):**
- `LoadParameters(int argc, char** argv)` (`application.cpp:50`). Parses each `argv[i]` via `Config::KV2Map`; if a `config=` file is given, reads it with `TextReader<size_t>`, strips `#` comments, trims, and merges. Then de-duplicates (`Config::KeepFirstValues`), applies alias transforms (`ParameterAlias::KeyAliasTransform`), and finally `config_.Set(params)` populates the `Config` struct (`application.cpp:84`).
- `LoadData()` (`application.cpp:88`). If continuing training from an existing model, builds a `Predictor` and grabs its `PredictFunction` to compute initial scores (`application.cpp:94-97`). Creates a `DatasetLoader` (`application.cpp:105`) and loads the training file via `LoadFromFile` (`application.cpp:110`/`114`, with rank/num-machines for distributed). Optionally saves a binary dataset (`application.cpp:117`), builds training `Metric`s (`application.cpp:121-129`), and loads each validation file with `LoadFromFileAlignWithOtherDataset` so its bins align with training data (`application.cpp:138-141`).
- `InitTrain()` (`application.cpp:168`). Optionally inits the network (`application.cpp:171`). Creates the boosting object via `Boosting::CreateBoosting(config_.boosting, config_.input_model.c_str())` (`application.cpp:182-184`) and the objective via `ObjectiveFunction::CreateObjectiveFunction(config_.objective, config_)` (`application.cpp:186-188`) — these are the *stringly-typed factories*. Calls `LoadData()`, then `objective_fun_->Init(...)` and `boosting_->Init(&config_, train_data_.get(), objective_fun_.get(), ...metrics...)` (`application.cpp:196-199`), and registers each validation set with `boosting_->AddValidDataset(...)` (`application.cpp:202`).
- `Train()` (`application.cpp:209`). The heart of CLI training: `boosting_->Train(config_.snapshot_freq, config_.output_model)` (`application.cpp:211`) runs the whole boosting loop, then `boosting_->SaveModelToFile(0, -1, feature_importance_type, output_model)` (`application.cpp:212`), and optionally emits C++ if-else code via `SaveModelToIfElse` (`application.cpp:216`).
- `InitPredict()` (`application.cpp:275`). Loads a saved model: `Boosting::CreateBoosting("gbdt", config_.input_model.c_str())` (`application.cpp:276-277`).
- `Predict()` (`application.cpp:221`). For the ordinary predict task: constructs a `Predictor` with the raw-score/leaf-index/contrib and early-stop flags from config (`application.cpp:264-267`) and calls `predictor.Predict(data_file, output_result_file, header, disable_shape_check, precise_float_parser)` (`application.cpp:268`). For the `KRefitTree` task it re-reads a leaf-prediction file, reloads data, and calls `boosting_->RefitTree(pred_leaf.data(), nrow, ncol)` (`application.cpp:258`).
- `ConvertModel()` (`application.cpp:281`). Loads the model and calls `boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str())`.

**Member fields (owned resources, `application.h:61-74`):** `Config config_`; `std::unique_ptr<Dataset> train_data_`; `std::vector<std::unique_ptr<Dataset>> valid_datas_`; `std::vector<std::unique_ptr<Metric>> train_metric_`; `std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_`; `std::unique_ptr<Boosting> boosting_`; `std::unique_ptr<ObjectiveFunction> objective_fun_`. Note the `unique_ptr` ownership pattern — the Application owns all engine objects.

**Dependencies (`application.cpp:5-27`):** `boosting.h`, `dataset.h`, `dataset_loader.h`, `metric.h`, `network.h`, `objective_function.h`, `prediction_early_stop.h`, `utils/common.h`, `utils/openmp_wrapper.h`, `utils/text_reader.h`, and the local `predictor.hpp`. The header forward-declares `DatasetLoader`, `Dataset`, `Boosting`, `ObjectiveFunction`, `Metric` (`application.h:16-20`).

**Pipeline fit.** This is the top of the *CLI* control flow: `main()` → `Application(argc,argv)` (parse) → `Run()` → dispatch to `InitTrain()+Train()` **or** `InitPredict()+Predict()` **or** `ConvertModel()`. Train writes a model file; Predict writes a results file.

---

### `src/application/predictor.hpp` — batch/file prediction driver

**Purpose.** `Predictor` wraps a trained `Boosting` model and turns it into a callable that maps one input row to its prediction outputs. It is used both by the CLI `Application::Predict` (file→file) and internally by the C API's in-memory predict functions. It selects, at construction time, exactly *which* kind of prediction is wanted (normal score, raw score, leaf index, or SHAP feature contributions) and builds a lambda for it.

#### Class `Predictor` (`predictor.hpp:30`)

**Constructor** (`predictor.hpp:41-142`):
```cpp
Predictor(Boosting* boosting, int start_iteration, int num_iteration,
          bool is_raw_score, bool predict_leaf_index, bool predict_contrib,
          bool early_stop, int early_stop_freq, double early_stop_margin)
```
- Builds an early-stop instance (`predictor.hpp:44-59`): default `"none"`; if `early_stop` is on and the booster doesn't need accurate prediction, uses `"binary"` (1 class) or `"multiclass"` (`CreatePredictionEarlyStopInstance`, from `prediction_early_stop.h`).
- Calls `boosting->InitPredict(start_iteration, num_iteration, predict_contrib)` (`predictor.hpp:61`), computes `num_pred_one_row_` (`predictor.hpp:63`) and `num_feature_` (`predictor.hpp:65`), and allocates a per-thread scratch buffer `predict_buf_` (aligned `double` vectors, one per OpenMP thread; `predictor.hpp:66-69`).
- Selects the prediction lambda `predict_fun_` based on the flags: leaf-index (`predictor.hpp:73`), contrib (`predictor.hpp:92`, plus a sparse variant `predict_sparse_fun_` at `predictor.hpp:101`; contrib is *not* supported for linear trees, `predictor.hpp:89-90`), raw score (`predictor.hpp:110`), or normal (`predictor.hpp:126`). Each lambda either scatters the sparse feature list into the dense per-thread buffer and calls the corresponding `boosting_->Predict*` (e.g. `Predict`, `PredictRaw`, `PredictLeafIndex`, `PredictContrib`), or, for very wide sparse inputs (`num_feature_ > 100000` and few nonzeros), uses the map-based `*ByMap` variants (`predictor.hpp:76-79`).

**Type of the lambda:** `PredictFunction` = `std::function<void(const std::vector<std::pair<int,double>>& features, double* output)>` (from `meta.h`). Each input row is a sparse list of `(feature_index, value)` pairs; output is written to a caller-provided `double*`.

**Key methods:**
- `const PredictFunction& GetPredictFunction() const` (`predictor.hpp:150`) and `const PredictSparseFunction& GetPredictSparseFunction() const` (`predictor.hpp:155`) — expose the built lambdas.
- `void Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check, bool precise_float_parser)` (`predictor.hpp:164`). Opens an output `VirtualFileWriter` (`predictor.hpp:165`), creates a `Parser` for the input format (`predictor.hpp:170`), enforces feature-count match unless `disable_shape_check` (`predictor.hpp:176-179`), optionally remaps columns by matching header names to trained feature names (`predictor.hpp:184-208`), and then reads the file in parallel blocks via `predict_data_reader.ReadAllAndProcessParallel(process_fun)` (`predictor.hpp:255`). `process_fun` (`predictor.hpp:230-254`) parses each line, calls `predict_fun_`, joins the `double` results with tabs, and writes them out. The parallel loop uses the project's OpenMP exception-propagation macros `OMP_INIT_EX()/OMP_LOOP_EX_BEGIN()/OMP_LOOP_EX_END()/OMP_THROW_EX()`.

**Private helpers:** `CopyToPredictBuffer` (`predictor.hpp:259`) scatters sparse features into the dense buffer; `ClearPredictBuffer` (`predictor.hpp:267`) zeroes it after use (memset if dense enough, else per-feature); `CopyToPredictMap` (`predictor.hpp:279`) builds a `std::unordered_map<int,double>` for the wide-sparse path.

**Members (`predictor.hpp:289-297`):** `const Boosting* boosting_`, `PredictFunction predict_fun_`, `PredictSparseFunction predict_sparse_fun_`, `PredictionEarlyStopInstance early_stop_`, `int num_feature_`, `int num_pred_one_row_`, and the per-thread `predict_buf_`.

**Dependencies (`predictor.hpp:8-13`):** `boosting.h`, `dataset.h`, `meta.h`, `utils/common.h`, `utils/openmp_wrapper.h`, `utils/text_reader.h`. Uses `Parser`, `VirtualFileWriter` (from `dataset.h`/io).

**Pipeline fit.** Sits between the entry layer and the engine's prediction path. For the CLI it is the file→file driver; for the C API it is the in-memory row-by-row driver (the C API's `Booster::Predict` builds a `Predictor`, then applies its `predict_fun_` to each row of an in-memory matrix/CSR).

---

### `include/LightGBM/export.h` — symbol-export macros

**Purpose.** A tiny header defining the macros that mark functions for export from the shared library and for C linkage. This is what makes the C API callable across the ABI boundary on every platform.

**Macros (`export.h:10-23`):**
- `LIGHTGBM_EXTERN_C` → `extern "C"` under C++, empty under C (`export.h:10-14`). Disables C++ name-mangling.
- `LIGHTGBM_EXPORT` → `__declspec(dllexport)` on MSVC, empty elsewhere (`export.h:18`, `21`).
- `LIGHTGBM_C_EXPORT` → `LIGHTGBM_EXTERN_C __declspec(dllexport)` on MSVC, else just `LIGHTGBM_EXTERN_C` (`export.h:19`, `22`). **Every public `LGBM_*` function in `c_api.h` is declared `LIGHTGBM_C_EXPORT`.**

**Dependencies.** None. Consumed by `c_api.h` and `prediction_early_stop.h`.

---

### `include/LightGBM/prediction_early_stop.h` + `src/boosting/prediction_early_stop.cpp` — prediction-time early stopping

**Purpose.** An optimization used *during prediction*: for some objectives you can stop summing tree outputs early once the answer is already decided (e.g. a binary margin is large enough), saving time. This defines the small callback abstraction and its three concrete strategies.

#### Structs (header)
- `struct PredictionEarlyStopInstance` (`prediction_early_stop.h:15`):
  - `using FunctionType = std::function<bool(const double*, int)>;` (`prediction_early_stop.h:19`) — takes the current prediction array and its length, returns `true` if prediction can stop.
  - `FunctionType callback_function;` (`prediction_early_stop.h:21`)
  - `int round_period;` — call the callback only every `round_period` iterations (`prediction_early_stop.h:22`).
- `struct PredictionEarlyStopConfig` (`prediction_early_stop.h:25`): `int round_period;` and `double margin_threshold;`.

#### Factory function
- `LIGHTGBM_EXPORT PredictionEarlyStopInstance CreatePredictionEarlyStopInstance(const std::string& type, const PredictionEarlyStopConfig& config)` (`prediction_early_stop.h:31`, defined `prediction_early_stop.cpp:75`). Dispatches on `type`:
  - `"none"` → `CreateNone` (`prediction_early_stop.cpp:16`): callback always returns `false`, `round_period = INT_MAX` so it is effectively never called.
  - `"multiclass"` → `CreateMulticlass` (`prediction_early_stop.cpp:25`): partial-sorts the class scores, stops when `(top1 - top2) > margin_threshold` (`prediction_early_stop.cpp:40-44`). Fatal if fewer than 2 predictions.
  - `"binary"` → `CreateBinary` (`prediction_early_stop.cpp:54`): stops when `2*|pred[0]| > margin_threshold` (`prediction_early_stop.cpp:63`). Fatal if length ≠ 1.
  - Unknown type → `Log::Fatal` (`prediction_early_stop.cpp:84`).

**Inputs/outputs.** In: `const std::string& type`, `const PredictionEarlyStopConfig&`. Out: a `PredictionEarlyStopInstance` (a callback + period).

**Dependencies.** Header includes `export.h`, `<string>`, `<functional>`. The .cpp includes `utils/log.h`, `<limits>`, `<algorithm>`, `<cmath>`, `<vector>`.

**Pipeline fit.** Consumed by `Predictor`'s constructor (`predictor.hpp:44-59`) and by the C API single-row predictor. The chosen instance is passed into `boosting_->Predict/PredictRaw` so the boosting loop can bail out early.

---

### `include/LightGBM/c_api.h` + `src/c_api.cpp` — the stable C ABI (the port's primary surface)

**Purpose.** The single most important file for anyone binding to LightGBM. `c_api.h` declares ~110 `extern "C"` `LGBM_*` functions; `c_api.cpp` implements them by wrapping the internal C++ `Dataset`, `Boosting`, `Metric`, `Objective` objects behind opaque handles. Every non-CLI consumer (Python `basic.py` via ctypes, R, Java/SWIG, and the Rust port) goes through here. The file opens with a note (`c_api.h:5-11`) that most interfaces accept both float32 and float64 to avoid conversion on large data, *except* gradients/Hessians and training scores.

#### Handle typedefs and constants (`c_api.h:30-49`)
- `typedef void* DatasetHandle;` (`c_api.h:30`), `BoosterHandle` (`:31`), `FastConfigHandle` (`:32`), `ByteBufferHandle` (`:33`).
- Data-type tags for the `data_type`/`indptr_type` int arguments: `C_API_DTYPE_FLOAT32 (0)`, `FLOAT64 (1)`, `INT32 (2)`, `INT64 (3)` (`c_api.h:35-38`).
- Prediction-kind tags: `C_API_PREDICT_NORMAL (0)`, `RAW_SCORE (1)`, `LEAF_INDEX (2)`, `CONTRIB (3)` (`c_api.h:40-43`).
- Sparse-matrix tags: `C_API_MATRIX_TYPE_CSR (0)`, `CSC (1)` (`c_api.h:45-46`). Feature-importance tags: `SPLIT (0)`, `GAIN (1)` (`c_api.h:48-49`).

#### Universal calling convention
- **Every** `LGBM_*` function returns `int`: `0` on success, `-1` on failure (documented on each). On `-1` the caller retrieves the message via `LGBM_GetLastError()` (`c_api.h:55`, defined `c_api.cpp:952`).
- **Output parameters** are trailing pointers marked `[out]` in the docs (e.g. `DatasetHandle* out`, `int64_t* out_len`, `double* out_result`). Callers pre-allocate output arrays; several "Get…" calls document how to size them (e.g. `LGBM_BoosterCalcNumPredict` gives the length to allocate before `LGBM_BoosterPredictForMat`).

#### Error handling machinery (`c_api.cpp:38-52`, `c_api.h:1634-1656`)
- `LastErrorMsg()` returns a **thread-local** `char[512]` buffer, initialized to `"Everything is fine"` (`c_api.h:1638`). `THREAD_LOCAL` is platform-selected: `thread_local`, `__declspec(thread)`, or `_Thread_local` (`c_api.h:1620-1632`).
- `LGBM_SetLastError(const char* msg)` (`c_api.h:1649`) writes into it via `snprintf`.
- Two `LGBM_APIHandleException` overloads (`c_api.cpp:38-45`) set the last error and return `-1`.
- The **`API_BEGIN()` / `API_END()` macro pair** (`c_api.cpp:47-52`) wraps every function body in a `try { … } catch(std::exception&){…} catch(std::string&){…} catch(...){…} return 0;`. So the pattern is: `int LGBM_Foo(...) { API_BEGIN(); …work…; API_END(); }`. This converts every C++ exception into the `-1` + last-error contract.
- `UNIQUE_LOCK(mtx)` / `SHARED_LOCK(mtx)` macros (`c_api.cpp:54-58`) take a write/read lock on a `yamc::alternate::shared_mutex` for thread-safe handle access.

#### Internal wrapper classes (in `namespace LightGBM`, not exported)

- **`class Booster`** (`c_api.cpp:163`) — the object a `BoosterHandle` actually points to. It owns `std::unique_ptr<Boosting> boosting_`, the training `Config config_`, `train_data_` (non-owning `const Dataset*`), `train_metric_`, `valid_metrics_`, `objective_fun_`, an array of `single_row_predictor_[PREDICTOR_TYPES]` (`PREDICTOR_TYPES == 4`, `c_api.cpp:60`), and a `mutable yamc::alternate::shared_mutex mutex_` (fields at `c_api.cpp:876-891`).
  - Two constructors: `Booster(const char* filename)` loads a saved GBDT model (`c_api.cpp:165-167`); `Booster(const Dataset* train_data, const char* parameters)` builds a fresh booster for training — parses params, sets threads, creates the boosting object, objective, and metrics, and calls `boosting_->Init(...)` (`c_api.cpp:169-194`). Note: **feature-parallel tree learner is rejected in the C API** (`c_api.cpp:186`), and a lone worker is forced to `"serial"` (`c_api.cpp:188-191`).
  - Key methods (each takes a lock): `TrainOneIter()` → `boosting_->TrainOneIter(nullptr, nullptr)` (`c_api.cpp:406-409`); `TrainOneIter(const score_t* gradients, const score_t* hessians)` for custom objectives (`c_api.cpp:416-419`); `RollbackOneIter()` (`c_api.cpp:421`); `Refit(const int32_t* leaf_preds, int32_t nrow, int32_t ncol)` (`c_api.cpp:411`); `AddValidData` (`c_api.cpp:392`); `ResetTrainingData` (`c_api.cpp:228`); `ResetConfig` (`c_api.cpp:354`); `Predict(...)` builds a `Predictor` and runs an OpenMP-parallel row loop (`c_api.cpp:485-511`); `PredictSingleRow` (`c_api.cpp:446`); the sparse-output variants (`c_api.cpp:513-`); `GetEvalCounts`/`GetEvalNames` (`c_api.cpp:831-856`); `GetFeatureNames` (`c_api.cpp:858`); `SaveModelToFile`/`SaveModelToString` (`c_api.cpp:782-791`); `GetBoosting()` accessor (`c_api.cpp:873`); and static `CheckDatasetResetConfig` (`c_api.cpp:239`) which fatally rejects changing bin-affecting params (`max_bin`, `min_data_in_bin`, `categorical_feature`, `header`, etc.) after a Dataset is built.
- **`class SingleRowPredictorInner`** (`c_api.cpp:63`) — caches a `Predictor` + its `predict_function` and `num_pred_in_one_row` for repeated single-row prediction; `IsPredictorEqual` (`c_api.cpp:92`) checks whether the cache can be reused.
- **`struct SingleRowPredictor`** (`c_api.cpp:117`, publicly called `FastConfig` in the API) — bundles a parsed `Config`, `data_type`, `num_cols`, a `SingleRowPredictorInner`, and locks the booster during a prediction (`c_api.cpp:133-142`). Returned as a `FastConfigHandle` by the `*FastInit` functions.
- **`class CSC_RowIterator`** (`c_api.cpp:932`) plus free helpers `RowFunctionFromDenseMatric`, `RowPairFunctionFromDenseMatric`, `RowPairFunctionFromDenseRows`, `RowFunctionFromCSR` (`c_api.cpp:917-929`) convert the various C input layouts (dense row-/column-major, CSR, CSC) into the `(feature_index, value)` row-lambda that `Predictor` consumes.

#### The exported `LGBM_*` surface — grouped

All are `LIGHTGBM_C_EXPORT int` returning 0/-1. Key inputs/outputs by group:

**Global / utility:**
- `const char* LGBM_GetLastError()` (`c_api.h:55`) — the only non-`int` export; returns the thread-local message.
- `LGBM_DumpParamAliases(int64_t buffer_len, int64_t* out_len, char* out_str)` (`c_api.h:64`); `LGBM_RegisterLogCallback(void (*callback)(const char*))` (`c_api.h:73`); `LGBM_GetSampleCount(int32_t num_total_row, const char* parameters, int* out)` (`c_api.h:82`); `LGBM_SampleIndices(...)` (`c_api.h:96`); `LGBM_SetMaxThreads(int)` / `LGBM_GetMaxThreads(int*)` (`c_api.h:1603`,`:1610`); `LGBM_ByteBufferGetAt`/`LGBM_ByteBufferFree` (`c_api.h:108`,`:115`).

**Dataset construction (each yields a `DatasetHandle* out`):**
- From file: `LGBM_DatasetCreateFromFile(const char* filename, const char* parameters, const DatasetHandle reference, DatasetHandle* out)` (`c_api.h:127`).
- From sampled columns: `LGBM_DatasetCreateFromSampledColumn(double** sample_data, int** sample_indices, int32_t ncol, const int* num_per_col, int32_t num_sample_row, int32_t num_local_row, int64_t num_dist_row, const char* parameters, DatasetHandle* out)` (`c_api.h:145`).
- By reference / serialized reference: `LGBM_DatasetCreateByReference` (`c_api.h:162`), `LGBM_DatasetCreateFromSerializedReference` (`c_api.h:195`).
- Dense matrix: `LGBM_DatasetCreateFromMat(const void* data, int data_type, int32_t nrow, int32_t ncol, int is_row_major, const char* parameters, const DatasetHandle reference, DatasetHandle* out)` (`c_api.h:409`); array-of-matrices `LGBM_DatasetCreateFromMats` (`c_api.h:431`).
- Sparse: `LGBM_DatasetCreateFromCSR` (`c_api.h:340`), `LGBM_DatasetCreateFromCSRFunc` (`c_api.h:363`), `LGBM_DatasetCreateFromCSC` (`c_api.h:385`). Arrow: `LGBM_DatasetCreateFromArrow` (`c_api.h:451`). Subset: `LGBM_DatasetGetSubset` (`c_api.h:467`).
- **Streaming push flow** (documented at `c_api.h:219-224`): `LGBM_DatasetInitStreaming` → `LGBM_DatasetPushRows`/`LGBM_DatasetPushRowsWithMetadata`/`…ByCSR…` → `LGBM_DatasetMarkFinished`; `LGBM_DatasetSetWaitForManualFinish` controls whether `FinishLoad` is called automatically.

**Dataset accessors / mutators:**
- `LGBM_DatasetSetFeatureNames(DatasetHandle, const char** feature_names, int num_feature_names)` (`c_api.h:480`); `LGBM_DatasetGetFeatureNames(...)` (`c_api.h:496`).
- `LGBM_DatasetSetField(DatasetHandle handle, const char* field_name, const void* field_data, int num_element, int type)` (`c_api.h:552`) — sets `label`/`weight`/`init_score`/`group`; note the type constraints (group=int32, label/weight=float32, init_score=float64, `c_api.h:540-544`). `LGBM_DatasetGetField(... int* out_len, const void** out_ptr, int* out_type)` (`c_api.h:586`); Arrow variant `LGBM_DatasetSetFieldFromArrow` (`c_api.h:571`).
- `LGBM_DatasetGetNumData(DatasetHandle, int* out)` (`c_api.h:607`), `LGBM_DatasetGetNumFeature` (`c_api.h:616`), `LGBM_DatasetGetFeatureNumBin` (`c_api.h:626`), `LGBM_DatasetAddFeaturesFrom` (`c_api.h:636`), `LGBM_DatasetUpdateParamChecking` (`c_api.h:598`).
- Persistence: `LGBM_DatasetSaveBinary` (`c_api.h:516`), `LGBM_DatasetSerializeReferenceToBinary` (`c_api.h:526`), `LGBM_DatasetDumpText` (`c_api.h:536`), `LGBM_DatasetFree(DatasetHandle handle)` (`c_api.h:508`).

**Booster lifecycle & training:**
- Create/load/free: `LGBM_BoosterCreate(const DatasetHandle train_data, const char* parameters, BoosterHandle* out)` (`c_api.h:656`); `LGBM_BoosterCreateFromModelfile(const char* filename, int* out_num_iterations, BoosterHandle* out)` (`c_api.h:667`); `LGBM_BoosterLoadModelFromString` (`c_api.h:678`); `LGBM_BoosterFree(BoosterHandle handle)` (`c_api.h:701`).
- Data/config management: `LGBM_BoosterAddValidData` (`c_api.h:729`), `LGBM_BoosterResetTrainingData` (`c_api.h:738`), `LGBM_BoosterResetParameter(BoosterHandle, const char* parameters)` (`c_api.h:747`), `LGBM_BoosterMerge` (`c_api.h:720`), `LGBM_BoosterShuffleModels` (`c_api.h:710`).
- **The training loop primitive:** `LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished)` (`c_api.h:765`) runs one boosting round; `is_finished=1` means no more splits possible. `LGBM_BoosterUpdateOneIterCustom(BoosterHandle, const float* grad, const float* hess, int* is_finished)` (`c_api.h:793`) injects custom gradients/Hessians (length `num_class * num_train_data`, unchecked). `LGBM_BoosterRollbackOneIter` (`c_api.h:803`), `LGBM_BoosterRefit(BoosterHandle, const int32_t* leaf_preds, int32_t nrow, int32_t ncol)` (`c_api.h:776`).
- Introspection: `LGBM_BoosterGetNumClasses` (`c_api.h:756`), `LGBM_BoosterGetCurrentIteration` (`c_api.h:811`), `LGBM_BoosterNumModelPerIteration` (`c_api.h:820`), `LGBM_BoosterNumberOfTotalModel` (`c_api.h:829`), `LGBM_BoosterGetNumFeature` (`c_api.h:896`), `LGBM_BoosterGetLinear` (`c_api.h:647`).
- Eval/predict-on-training-data: `LGBM_BoosterGetEvalCounts(BoosterHandle, int* out_len)` (`c_api.h:838`), `LGBM_BoosterGetEvalNames(...)` (`c_api.h:853`), `LGBM_BoosterGetEval(BoosterHandle, int data_idx, int* out_len, double* out_results)` (`c_api.h:910`; `data_idx` 0=train, 1..=validation sets), `LGBM_BoosterGetNumPredict` (`c_api.h:923`), `LGBM_BoosterGetPredict` (`c_api.h:937`). Feature names: `LGBM_BoosterGetFeatureNames` (`c_api.h:872`), `LGBM_BoosterValidateFeatureNames` (`c_api.h:886`).

**Prediction (in-memory & file):**
- File: `LGBM_BoosterPredictForFile(BoosterHandle, const char* data_filename, int data_has_header, int predict_type, int start_iteration, int num_iteration, const char* parameter, const char* result_filename)` (`c_api.h:958`).
- Size helper: `LGBM_BoosterCalcNumPredict(BoosterHandle, int num_row, int predict_type, int start_iteration, int num_iteration, int64_t* out_len)` (`c_api.h:981`) — call this to size `out_result` before the matrix/CSR predicts.
- Dense: `LGBM_BoosterPredictForMat(...)` (`c_api.h:1281`), `LGBM_BoosterPredictForMats(...)` (`c_api.h:1408`), single-row `LGBM_BoosterPredictForMatSingleRow` (`c_api.h:1319`) and its fast pair `…FastInit`/`…Fast` (`c_api.h:1350`,`:1379`).
- Sparse: `LGBM_BoosterPredictForCSR(...)` (`c_api.h:1024`), `LGBM_BoosterPredictForCSC(...)` (`c_api.h:1240`), single-row `LGBM_BoosterPredictForCSRSingleRow` (`c_api.h:1127`) + fast pair (`c_api.h:1162`,`:1202`), and the library-allocated `LGBM_BoosterPredictSparseOutput(... int64_t* out_len, void** out_indptr, int32_t** out_indices, void** out_data)` (`c_api.h:1068`) freed by `LGBM_BoosterFreePredictSparse` (`c_api.h:1096`). Arrow: `LGBM_BoosterPredictForArrow` (`c_api.h:1443`).
- The **Fast** variants return a `FastConfigHandle` from `*FastInit` that caches the predictor; call the matching `*Fast` per row, then `LGBM_FastConfigFree(FastConfigHandle)` (`c_api.h:994`). In all `out_result` layouts: normal/raw = `num_class*num_data`; leaf = `num_class*num_data*num_iteration`; contrib = `num_class*num_data*(num_feature+1)` (documented `c_api.h:999-1002`).

**Model I/O & inspection:**
- `LGBM_BoosterSaveModel(BoosterHandle, int start_iteration, int num_iteration, int feature_importance_type, const char* filename)` (`c_api.h:1463`), `LGBM_BoosterSaveModelToString(...)` (`c_api.h:1480`), `LGBM_BoosterDumpModel(...)` → JSON (`c_api.h:1499`).
- `LGBM_BoosterGetLoadedParam` (`c_api.h:690`), `LGBM_BoosterGetLeafValue`/`SetLeafValue` (`c_api.h:1515`,`:1528`), `LGBM_BoosterFeatureImportance` (`c_api.h:1543`), `LGBM_BoosterGetUpperBoundValue`/`GetLowerBoundValue` (`c_api.h:1554`,`:1563`).

**Network (distributed; out of scope but part of the surface):** `LGBM_NetworkInit` (`c_api.h:1574`), `LGBM_NetworkFree` (`c_api.h:1583`), `LGBM_NetworkInitWithFunctions` (`c_api.h:1593`).

#### Dependencies
`c_api.h` includes `arrow.h` and `export.h` (`c_api.h:16-17`) plus C stdint/stdio/string. `c_api.cpp` pulls in the whole engine: boosting, dataset/dataset_loader, metric, objective_function, network, prediction_early_stop, config, and utils — it `using`-imports `Booster`, `Config`, `Dataset`, `DatasetLoader`, `Network`, `Random`, `SingleRowPredictor`, `data_size_t`, etc. (`c_api.cpp:894-913`).

#### Pipeline fit — C API handle lifecycle
This mirrors `Application` but in discrete, caller-driven steps (this is exactly what Python's `Booster`/`Dataset` classes orchestrate):

1. **Build data:** `LGBM_DatasetCreateFromMat/CSR/File(...) → DatasetHandle`; optionally `LGBM_DatasetSetField` for labels/weights/groups. (Streaming: `InitStreaming` → `PushRows*` → `MarkFinished`.)
2. **Create booster:** `LGBM_BoosterCreate(train_data, params) → BoosterHandle`; add validation with `LGBM_BoosterAddValidData`.
3. **Train loop:** repeatedly `LGBM_BoosterUpdateOneIter(handle, &is_finished)` (or `…Custom` with your own grad/hess) until `is_finished` or a round budget; check progress with `LGBM_BoosterGetEval` / `GetCurrentIteration`.
4. **Persist:** `LGBM_BoosterSaveModel` / `SaveModelToString` / `DumpModel`.
5. **Predict:** `LGBM_BoosterCalcNumPredict` (size the buffer) → `LGBM_BoosterPredictForMat/CSR/File(...)`; or the Fast single-row path (`FastInit` → repeated `…Fast` → `FastConfigFree`).
6. **Free (mandatory, no GC):** `LGBM_BoosterFree(booster)` and `LGBM_DatasetFree(dataset)`; sparse-output predictions need `LGBM_BoosterFreePredictSparse`; byte buffers `LGBM_ByteBufferFree`.

Throughout, any non-zero return means call `LGBM_GetLastError()` for the message, and all handle mutations are serialized by the per-Booster `shared_mutex`.

---

#### Port implications (why this layer matters for the Rust rewrite)
- The C API is the **compatibility contract**: the Rust crate's public surface (and its Python binding) must reproduce these function signatures, the 0/-1 + thread-local-last-error convention, the handle lifecycle, and the exact output-array shapes/`num_class` layouts.
- The `Application` CLI flow and the C API flow are two front-ends over the *same* engine steps (create boosting + objective + metrics → init → per-iteration train → save/predict); both must be reproduced, but the C API is the higher-fidelity spec because it exposes each step explicitly.
- Stringly-typed factories (`Boosting::CreateBoosting`, `ObjectiveFunction::CreateObjectiveFunction`, `Metric::CreateMetric`, `CreatePredictionEarlyStopInstance`) are the porting seams surfaced here.


---

## Section B — Configuration, Meta Types & Utilities

This section documents the "plumbing" of LightGBM: the fundamental type aliases every other
file depends on (`meta.h`), the single giant parameter bag (`Config`) that carries all
user-facing settings through the whole engine, and a family of small utility headers
(logging, RNG, threading, file I/O, JSON, containers). No prior LightGBM knowledge is
assumed; jargon is explained on first use.

**Reading order for a newcomer:** start with `meta.h` (the type vocabulary), then `config.h`
(the parameter bag and how strings become a typed struct), then the utilities as needed.

---

### 1. `include/LightGBM/meta.h` — Core type vocabulary

**Purpose (plain language):** This tiny header defines the fundamental numeric *type aliases*
(`typedef`s) and constants used everywhere else in the codebase. Because LightGBM chooses
single-precision `float` for scores/labels by default, and 32-bit signed integers for row
indices, changing these one-line typedefs would ripple through the entire engine. For a Rust
port these are the load-bearing definitions: getting them wrong breaks numerical parity.

#### Type aliases (the ones to reproduce exactly)

| Alias | Underlying C++ type | Meaning | Location |
|-------|--------------------|---------|----------|
| `data_size_t` | `int32_t` | Type of a **row/sample count or index**. Deliberately *signed* (the comment says "it is better to use signed type"). | `meta.h:28` |
| `score_t` | `float` by default; `double` if the macro `SCORE_T_USE_DOUBLE` is defined | Type of model **scores and gradients/hessians**. | `meta.h:36-41` |
| `label_t` | `float` by default; `double` if `LABEL_T_USE_DOUBLE` is defined | Type of **metadata: labels and sample weights**. | `meta.h:43-48` |
| `comm_size_t` | `int32_t` | Type of **network / inter-machine communication sizes** (byte counts, block lengths). | `meta.h:59` |

Both `score_t` and `label_t` are `float` because the project's numerical contract is
single-precision end-to-end. The `SCORE_T_USE_DOUBLE` / `LABEL_T_USE_DOUBLE` macros are
commented-out compile-time switches (`meta.h:30-34`) that would promote them to `double`.

#### Constants (parity-critical thresholds)

| Constant | Value | Type | Meaning | Location |
|----------|-------|------|---------|----------|
| `kMinScore` | `-inf` | `score_t` | Sentinel "worst" score. | `meta.h:50` |
| `kMaxScore` | `+inf` | `score_t` | Sentinel "best" score. | `meta.h:52` |
| `kEpsilon` | `1e-15f` | `score_t` (float) | Tiny value used to avoid divide-by-zero and as a "is this effectively zero?" guard (e.g. hessian floors, `path_smooth` checks). | `meta.h:54` |
| `kZeroThreshold` | `1e-35f` | `double` | A much smaller "treat as zero" threshold (note: the literal is an `f`-suffixed float assigned to a `double`). Used e.g. in AUC-mu weight validation. | `meta.h:56` |
| `kAlignedSize` | `32` | `int` | Memory-alignment granularity (bytes) for aligned buffers. | `meta.h:80` |

#### Macros and function-pointer typedefs

- `PREFETCH_T0(addr)` (`meta.h:16-23`): cross-compiler CPU cache-prefetch hint (`_mm_prefetch`
  on x86 MSVC/Intel, `__builtin_prefetch` on GCC, no-op otherwise).
- `NO_SPECIFIC` = `-1` (`meta.h:78`): sentinel meaning "not set".
- `SIZE_ALIGNED(t)` (`meta.h:82`): rounds `t` up to the next multiple of `kAlignedSize` (32).
  Used by the threading block-sizer.
- Communication function-pointer types used by the distributed `Network` layer:
  - `PredictFunction` = `std::function<void(const std::vector<std::pair<int,double>>&, double* output)>` (`meta.h:61`) — the callback shape for predicting one row given its (feature-index, value) pairs.
  - `PredictSparseFunction` (`meta.h:64`) — sparse-output variant.
  - `ReduceFunction`, `ReduceScatterFunction`, `AllgatherFunction` (`meta.h:67-75`) — raw
    function-pointer types for MPI/socket collective operations, all keyed on `comm_size_t`.

**Who uses it:** essentially every header includes `meta.h` (directly or transitively).
`config.h`, `threading.h`, and the whole `src/` tree depend on these aliases.

---

### 2. `include/LightGBM/config.h` + `src/io/config.cpp` + `src/io/config_auto.cpp` — The parameter bag

This is the most important trio in the subsystem. Together they implement one design idea:
**every training/prediction knob lives as a public field in one big struct called `Config`,
and there is machine-generated glue that maps user-supplied string parameters (and their many
aliases) onto those typed fields.**

#### 2.1 The "single struct bag-of-parameters" pattern (`config.h`)

`struct Config` (`config.h:39`) is a plain struct whose *public data members* are the
parameters. Examples with their C++ types and defaults:

```cpp
std::string objective   = "regression";          // config.h:165
std::string boosting    = "gbdt";                 // config.h:176
int    num_iterations   = 100;                    // config.h:203
double learning_rate    = 0.1;                    // config.h:209
int    num_leaves       = kDefaultNumLeaves;      // = 31, config.h:216 (kDefaultNumLeaves at :37)
int    num_threads      = 0;                      // config.h:239  (0 = OpenMP default)
std::string device_type = "cpu";                  // config.h:253
int    seed             = 0;                      // config.h:261
bool   deterministic    = false;                  // config.h:269
int    max_bin          = 255;                    // (Dataset/IO section)
```

`TaskType task` (`config.h:128`) is an enum field (`enum TaskType { kTrain, kPredict,
kConvertModel, KRefitTree, kSaveBinary }`, `config.h:34-36`).

**How parameters are *declared*.** Each field is preceded by structured doc-comments that a
code generator reads. Take `num_iterations` (`config.h:199-203`):

```cpp
// alias = num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, nrounds, num_boost_round, n_estimators, max_iter
// check = >=0
// desc = number of boosting iterations
int num_iterations = 100;
```

Recognized annotations:
- `alias = ...` — alternative names users may type (e.g. `n_estimators` → `num_iterations`).
- `check = >=0`, `check = >1`, `check = <=131072` (see `num_leaves`, `config.h:213-214`) —
  value constraints enforced in the generated code.
- `type = enum` + `options = ...` — closed set of string values (e.g. `objective`, `boosting`, `device_type`, `tree_learner`, `task`).
- `default = ...` — documented default (may differ from the C++ initializer when the default is dynamic, e.g. `seed`'s documented default is `None`).
- `[no-automatically-extract]` (file header note, `config.h:8-10`) — do **not** auto-generate
  the string→field extraction for this param; it has hand-written parsing in `config.cpp`
  (used for `objective`, `boosting`, `task`, `device_type`, `tree_learner`, etc.).
- `[no-save]` (`config.h:11-14`) — do not write this param into the saved model text (used for
  CLI-only / file-path params like `data`, `output_model`).
- `#pragma region ...` blocks group params into sections: *Core Parameters*, *Learning Control
  Parameters*, *IO Parameters*, *Objective/Metric Parameters*, *Network*, *GPU* (`config.h:104-107`, `:274`, etc.).

**Non-parameter members** live after the parameter block (`config.h:1144-1163`): e.g.
`is_parallel`, `is_data_based_parallel`, derived matrices `auc_mu_weights_matrix`
(`std::vector<std::vector<double>>`) and `interaction_constraints_vector`
(`std::vector<std::vector<int>>`), plus the method declarations.

#### 2.2 String-to-typed-value helpers (`config.h`, inline)

Static helpers convert a `std::unordered_map<std::string,std::string>` entry to a typed value,
returning `true` if the key was present and non-empty:

- `Config::GetString(params, name, std::string* out)` — `config.h:1165`.
- `Config::GetInt(params, name, int* out)` — `config.h:1175`; parses via `Common::AtoiAndCheck`, `Log::Fatal` on bad format.
- `Config::GetDouble(params, name, double* out)` — `config.h:1188`; via `Common::AtofAndCheck`.
- `Config::GetBool(params, name, bool* out)` — `config.h:1201`; accepts `true/+` and `false/-` (case-insensitive).

Alias plumbing:
- `Config::SortAlias(x, y)` (`config.h:1220`) — orders aliases shortest-first, then
  alphabetically, to pick a canonical winner deterministically.
- `struct ParameterAlias::KeyAliasTransform(params)` (`config.h:1224-1261`) — rewrites every
  alias key in the map to its canonical name using `Config::alias_table()`, warning on
  duplicate aliases and on **unknown parameters** (keys not in `Config::parameter_set()`).
- Free functions `ParseObjectiveAlias` (`config.h:1263`) and `ParseMetricAlias`
  (`config.h:1290`) normalize objective/metric spelling variants (e.g. `mse`/`l2`/`rmse` →
  `regression`; `binary` metric → `binary_logloss`).

#### 2.3 The runtime flow: `Config::Set` and friends (`src/io/config.cpp`)

`Config::Set(const std::unordered_map<std::string,std::string>& params)` (`config.cpp:257`) is
the entry point that turns a parsed parameter map into a fully-populated, validated `Config`.
Order of operations:

1. **Seed derivation** (`config.cpp:258-268`): if `seed` is given, a `Random` (see §3.3) is
   used to derive `data_random_seed`, `bagging_seed`, `drop_seed`, `feature_fraction_seed`,
   `objective_seed`, `extra_seed` — so a single master seed reproducibly seeds all subsystems.
2. **Hand-written enum extraction** for the `[no-automatically-extract]` params:
   `GetTaskType`, `GetBoostingType`, `GetDataSampleStrategy`, `GetObjectiveType`,
   `GetMetricType`, `GetDeviceType`, `GetTreeLearnerType` (`config.cpp:99-217`, called at
   `:270-279`). These lowercase the value and map aliases to canonical strings, `Log::Fatal` on
   unknown values.
3. **`GetMembersFromString(params)`** (`config.cpp:281`) — the *machine-generated* bulk
   extraction (implemented in `config_auto.cpp`, see §2.4).
4. **Derived structures**: `GetAucMuWeights()` (`config.cpp:219`) builds
   `auc_mu_weights_matrix`; `GetInteractionConstraints()` (`config.cpp:249`) parses the
   nested-array string into `interaction_constraints_vector` via
   `Common::StringToArrayofArrays<int>`.
5. Sort `eval_at`, drop training file from `valid` list, force `save_binary` for the
   `save_binary` task (`config.cpp:288-304`).
6. **`CheckParamConflict(params)`** (`config.cpp:314`) — a large block of cross-parameter
   validation and auto-correction. Representative rules:
   - objective vs. metric vs. `num_class` consistency for multiclass (`:316-338`);
   - single-machine ⇒ `tree_learner="serial"`, `is_parallel=false` (`:340-352`);
   - `device_type=="gpu"` forces `force_col_wise=true` and disables quantized training
     (`:399-409`); `"cuda"` forces `force_row_wise=true` (`:410-417`);
   - `max_depth` set but `num_leaves` left default ⇒ shrink `num_leaves` to `2^max_depth`
     (`:384-398`);
   - `path_smooth>kEpsilon` forces `min_data_in_leaf>=2` (`:439-442`);
   - legacy `boosting=goss` is rewritten to `boosting=gbdt` + `data_sample_strategy=goss`
     (`:463-468`).

Other `config.cpp` functions: `Str2Map` (`:86`) is the top-level "raw string → clean map"
pipeline: `Common::Split` on whitespace → `KV2Map` (`:16`, splits `key=value`) → `SetVerbosity`
(`:42`, sets the global log level from `verbosity`/`verbose`) → `KeepFirstValues` (`:72`, first
value wins on duplicates) → `ParameterAlias::KeyAliasTransform`. `ToString()` (`:476`) and
`DumpAliases()` (`:487`) serialize config/aliases for the model file and tooling.

#### 2.4 `src/io/config_auto.cpp` — machine-generated glue

**This file is auto-generated** by `LightGBM/.ci/parameter-generator.py`, which parses the
annotated comments in `config.h` (header note at `config_auto.cpp:5-7`). It must never be
hand-edited; regenerating from `config.h` is the source of truth. It provides four
lookup tables plus two big extract/save functions:

- `Config::alias_table()` (`config_auto.cpp:10`) — `unordered_map<string,string>` mapping every
  alias → canonical name (e.g. `{"n_estimators","num_iterations"}`, `{"eta","learning_rate"}`,
  `{"num_leaf","num_leaves"}`).
- `Config::parameter_set()` (`config_auto.cpp:183`) — `unordered_set<string>` of all canonical
  parameter names (used to detect unknown params).
- `Config::parameter2aliases()` (`config_auto.cpp:792`) — reverse map, canonical → list of aliases.
- `Config::ParameterTypes()` — canonical name → type string.
- `Config::GetMembersFromString(params)` (`config_auto.cpp:329`) — one `GetInt/GetDouble/
  GetString/GetBool(params, "name", &field)` call per auto-extractable field, e.g.
  `GetInt(params,"num_iterations",&num_iterations)` (`:337`),
  `GetDouble(params,"learning_rate",&learning_rate)` (`:340`),
  `GetInt(params,"num_leaves",&num_leaves)` (`:343`),
  `GetInt(params,"max_bin",&max_bin)` (`:518`). Vector params are parsed via
  `Common::StringToArray` from a temp string.
- `Config::SaveMembersToString()` (`config_auto.cpp:~700+`) — emits `[name: value]\n` lines for
  every non-`[no-save]` param (e.g. `str_buf << "[lambda_l1: " << lambda_l1 << "]\n"`), used by
  `ToString()` to persist config into the model text.

#### 2.5 End-to-end config flow (string params → struct → every subsystem)

```
User (CLI args / C-API param string / Python kwargs)
   │  "num_leaves=31 objective=binary learning_rate=0.05 ..."
   ▼
Config::Str2Map            (config.cpp:86)   → clean unordered_map<string,string>,
   │                                            aliases canonicalized, verbosity applied
   ▼
Config::Set(map)           (config.cpp:257)
   ├─ derive seeds from `seed`                 (Random, random.h)
   ├─ hand-parse enum params                   (GetObjectiveType, GetDeviceType, ...)
   ├─ GetMembersFromString(map)                (config_auto.cpp — generated bulk extract)
   ├─ build derived matrices/vectors
   └─ CheckParamConflict(map)                  (validate + auto-correct)
   ▼
Fully-populated `Config` struct  (immutable-ish shared bag)
   ▼  consumed by every subsystem:
   Application, GBDT/Boosting, ObjectiveFunction, Metric, TreeLearner,
   Dataset/BinMapper, Network — each reads the fields it cares about.
```

The `Config` object is created once from the parameter string and passed by
pointer/reference into every factory (`Boosting::CreateBoosting`,
`ObjectiveFunction::CreateObjectiveFunction`, `TreeLearner::CreateTreeLearner`, etc.). It is the
single source of truth for behavior, which is exactly why a faithful Rust port must reproduce
the parameter names, aliases, defaults, and the `CheckParamConflict` auto-correction rules.

---

### 3. Utility headers (grouped logically)

#### 3.1 Logging & assertions — `include/LightGBM/utils/log.h`

**Purpose:** a single static logger plus `CHECK_*` assertion macros used across the codebase.

- `enum class LogLevel : int { Fatal=-1, Warning=0, Info=1, Debug=2 }` (`log.h:78-83`).
- `class Log` (`log.h:88`): static methods `Log::Debug/Info/Warning(const char* format, ...)`
  (printf-style varargs) and `Log::Fatal(...)` which formats, prints `[LightGBM] [Fatal] ...`
  to stderr, and **throws `std::runtime_error`** (`log.h:117-139`). `ResetLogLevel(LogLevel)`
  (`:95`) sets a thread-local minimum level; `ResetCallBack(Callback)` (`:97`) redirects output
  (used by language bindings). Output is prefixed `[LightGBM] [<Level>]`. Under `LGB_R_BUILD`
  it routes through R's `Rprintf`/`REprintf`.
- Assertion macros (all call `Log::Fatal` on failure): `CHECK(cond)` (`:41`), `CHECK_EQ/NE/GE/
  LE/GT/LT(a,b)` (`:47-69`), `CHECK_NOTNULL(ptr)` (`:71`). `THREAD_LOCAL` macro (`:34-38`)
  abstracts `thread_local` vs MSVC `__declspec(thread)`.

**Who uses it:** everything. It pulls in only standard headers, so it is the lowest-level util.

#### 3.2 OpenMP wrapper — `include/LightGBM/utils/openmp_wrapper.h` + `src/utils/openmp_wrapper.cpp`

**Purpose:** portable thread-count control and a scheme to safely propagate C++ exceptions out
of OpenMP parallel regions (OpenMP forbids exceptions crossing a parallel boundary).

- Two global ints (`openmp_wrapper.h:11-15`): `LGBM_MAX_NUM_THREADS` (a hard cap set only by
  `LGBM_SetMaxThreads()`) and `LGBM_DEFAULT_NUM_THREADS` (LightGBM's preferred default, set via
  `OMP_SET_NUM_THREADS`, e.g. from the `num_threads` param). Defined = `-1` in the `.cpp`
  (`openmp_wrapper.cpp:7-9`).
- `int OMP_NUM_THREADS()` (`.cpp:15-33`): returns `LGBM_DEFAULT_NUM_THREADS` if set, else
  `omp_get_max_threads()`, then clamps to `LGBM_MAX_NUM_THREADS` if that cap is set. This is the
  project-wide "how many threads should I use" function referenced in every
  `#pragma omp parallel for num_threads(OMP_NUM_THREADS())`.
- `void OMP_SET_NUM_THREADS(int)` (`.cpp:35-41`): `<=0` resets to "OpenMP default", else pins.
- Exception-safety trio: `class ThreadExceptionHelper` (`.h:49-74`) stores the first
  `std::exception_ptr` caught (mutex-guarded), and macros `OMP_INIT_EX()`, `OMP_LOOP_EX_BEGIN()`,
  `OMP_LOOP_EX_END()`, `OMP_THROW_EX()` (`.h:76-87`) wrap a parallel loop body so an exception on
  any thread is captured and re-thrown after the region.
- When `_OPENMP` is **not** defined (`.h:89-128`), stubs simulate a single thread:
  `OMP_NUM_THREADS()` returns `1`, the EX macros become no-ops.

#### 3.3 Random number generator — `include/LightGBM/utils/random.h`

**Purpose:** a small, *deterministic*, self-contained linear-congruential RNG (LCG). Determinism
here is essential for reproducibility/parity.

- `class Random` (`random.h:18`): default ctor seeds from `std::random_device`
  (`:23`); `explicit Random(int seed)` (`:32`) for reproducible runs (this is what
  `Config::Set` uses to derive sub-seeds).
- Core LCG (`random.h:101-111`): `x = 214013*x + 2531011` with `x` an `unsigned int`.
  `RandInt16()` returns bits 16..30; `RandInt32()` returns the low 31 bits.
- Public API: `int NextShort(lower, upper)` (`:41`, range via int16), `int NextInt(lower, upper)`
  (`:51`), `float NextFloat()` (`:59`, `[0,1)` = `RandInt16()/32768`), and
  `std::vector<int> Sample(int N, int K)` (`:69`) — pick K distinct ordered indices from
  `{0..N-1}` with two strategies depending on density.

#### 3.4 Threading helpers — `include/LightGBM/utils/threading.h`

**Purpose:** turn "process N items" into balanced per-thread blocks, and a reusable parallel
partition primitive. Templated on an index type `INDEX_T` (usually `data_size_t`).

- `class Threading` (`threading.h:19`):
  - `BlockInfo(cnt, min_cnt_per_block, int* out_nblock, INDEX_T* block_size)` (`:22`/`:30`) —
    compute number of blocks (`min(num_threads, ceil(cnt/min))`) and an aligned block size
    (`SIZE_ALIGNED`, i.e. rounded up to 32).
  - `BlockInfoForceSize(...)` (`:44`/`:61`) — like above but forces `block_size` to a multiple
    of `min_cnt_per_block` (used where block boundaries must land on feature/group boundaries).
  - `For(start, end, min_block_size, std::function<void(int,INDEX_T,INDEX_T)> inner_fun)`
    (`:68`) — runs `inner_fun(block_idx, inner_start, inner_end)` over blocks inside an
    OpenMP parallel region (with the OMP exception macros). Returns the number of blocks.
- `template<INDEX_T, bool TWO_BUFFER> class ParallelPartitionRunner` (`threading.h:91`) —
  parallel stable partition used by the tree learner's data-partition step: each thread
  partitions its slice via a user `func` returning the left-count, then results are compacted
  into `out` at computed write offsets. `TWO_BUFFER` selects one-buffer (reverse trick) vs
  two-buffer layout. Key method `Run<FORCE_SIZE>(cnt, func, INDEX_T* out) -> INDEX_T left_cnt`
  (`:117`).

#### 3.5 Array algorithms — `include/LightGBM/utils/array_args.h`

**Purpose:** small generic (`template<VAL_T>`) array reductions/selections, some threaded.

- `class ArrayArgs<VAL_T>` (`array_args.h:21`): `ArgMax` / `ArgMin` over `vector` or raw
  `(ptr,n)` (`:45-99`); `ArgMaxMT` (`:23`) is the multithreaded ArgMax (uses `Threading::For`,
  auto-selected for arrays > 1024, `:49`). `Partition` (`:101`) + `ArgMaxAtK` (`:131`) implement
  a quickselect; `MaxK(array, k, out)` (`:149`) returns the top-k. Helpers `Assign`,
  `CheckAllZero`, `CheckAll` (`:164-187`).

#### 3.6 File readers — `text_reader.h`, `pipeline_reader.h`

**Purpose:** stream large text data files line-by-line efficiently.

- `class PipelineReader` (`pipeline_reader.h:24`): static `Read(filename, skip_bytes,
  std::function<size_t(const char*, size_t)> process_fun)` (`:31`). Uses **two threads and two
  16 MB buffers** — one thread reads the next block while the caller's `process_fun` processes
  the current block, then swaps buffers (`:51-64`). Built on `VirtualFileReader` (§3.7).
- `template<INDEX_T> class TextReader` (`text_reader.h:26`): higher-level line reader built on
  `PipelineReader`. Ctor optionally skips a header line (`:33`). Key methods:
  `ReadAllAndProcess(process_fun)` (`:99`, splits buffer into lines honoring `\n`/`\r`/`\r\n`
  and stitches lines that straddle buffer boundaries via `last_line_`), `ReadAllLines()`
  (`:159`), `CountLine()` (`:245`), `SampleFromFile(Random*, sample_cnt, out)` (`:184`,
  reservoir sampling), `ReadAndFilterLines(filter_fun, out_indices)` (`:206`), and parallel
  variants `ReadAllAndProcessParallel[WithFilter]` (`:251`/`:322`). Holds results in
  `std::vector<std::string> lines_`.

#### 3.7 File abstraction — `file_io.h` + `src/io/file_io.cpp`, and `binary_writer.h`, `byte_buffer.h`

**Purpose:** abstract "where do bytes come from / go to" behind interfaces, so local files (and
potentially other backends) are interchangeable.

- `binary_writer.h`: `struct BinaryWriter` (`:16`) — abstract sink with pure-virtual
  `size_t Write(const void* data, size_t bytes)` (`:23`), plus `AlignedWrite(data, bytes,
  alignment=8)` (`:32`) and static `AlignedSize(bytes, alignment=8)` (`:48`).
- `file_io.h`:
  - `struct VirtualFileWriter : BinaryWriter` (`:23`) — adds `Init()` (`:30`), static factory
    `Make(filename) -> unique_ptr` (`:37`), static `Exists(filename)` (`:44`).
  - `struct VirtualFileReader` (`:50`) — `Init()`, `size_t Read(void* buffer, size_t bytes)
    const` (`:67`), static factory `Make(filename)` (`:73`).
- `src/io/file_io.cpp`: the only concrete implementation, `struct LocalFile` (`:16`) which
  derives from **both** reader and writer, wrapping a C `FILE*` (`fopen`/`fread`/`fwrite`/
  `fclose`). The `Make` factories (`:55-63`) return a `LocalFile` opened `"rb"` (reader) or
  `"wb"` (writer); `VirtualFileWriter::Exists` (`:65`) probes with a throwaway `"rb"` open.
- `byte_buffer.h`: `struct ByteBuffer final : BinaryWriter` (`:24`) — an in-memory
  auto-expanding `std::vector<char>` sink; `Write` appends bytes (`:31`), plus `Reserve`,
  `GetSize`, `GetAt`, `Data()` (`:40-54`). Used to serialize a model into memory (e.g. for the
  C-API to hand back to bindings).

#### 3.8 Dynamic chunked container — `include/LightGBM/utils/chunked_array.hpp`

**Purpose:** a growable array stored as a list of fixed-size chunks (avoids reallocating/copying
a huge contiguous buffer). Used by the C-API dataset-construction paths
(`LGBM_DatasetCreateFromMats`) where data arrives incrementally, possibly in parallel.

- `template<class T> class ChunkedArray` (`chunked_array.hpp:61`): ctor takes `chunk_size`
  (`:63`). **High-level API:** `add(value)` (`:81`) appends, allocating a new chunk when the
  current one fills; `get_add_count()` (`:95`), `get_chunks_count()` (`:102`),
  `get_last_chunk_add_count()` (`:109`), `data()` → `T**` (`:127`), `data_as_void()` → `void**`
  (`:137`), `coalesce_to(T* other, all_valid=false)` (`:150`) flattens into a contiguous buffer.
  **Low-level (parallel) API:** `new_chunk()` (`:238`, not thread-safe) then
  `setitem(chunk, idx, value)`/`getitem(...)` (`:194`/`:178`, thread-safe per address).
  `clear()`/`release()` (`:207`/`:216`) manage memory (each chunk is `new T[chunk_size]`).

#### 3.9 JSON — `include/LightGBM/utils/json11.h` + `src/io/json11.cpp`

**Purpose:** a tiny vendored third-party (Dropbox) JSON library for C++11, used for model
serialization/parsing (e.g. JSON model dump, forced-splits files, parser config). Vendored
directly rather than as a submodule.

- Namespace `json11_internal_lightgbm` (`json11.h:63`) — renamed to avoid clashing with other
  json11 copies.
- `class Json final` (`json11.h:69`): a value type representing any JSON value.
  `enum Type { NUL, NUMBER, BOOL, STRING, ARRAY, OBJECT }` (`:72`); `typedef std::vector<Json>
  array` and `typedef std::map<std::string,Json> object` (`:75-76`). Constructors for each type
  (`:79-113`, including implicit ctors for any type with a `to_json()`, and for map-like /
  vector-like containers). Accessors `type()`, `is_*()` (`:120-127`), value getters, plus static
  `Json::parse(...)` and instance `dump()` (per the header banner `:22-53`). **Design note
  (`:39-53`):** all numbers are stored internally as `double` (no int/float distinction), which
  matters for round-trip fidelity of the model file.
- `src/io/json11.cpp` (781 lines) is the implementation: the `JsonValue` class hierarchy
  (`JsonInt`, `JsonDouble`, `JsonBoolean`, `JsonString`, `JsonArray`, `JsonObject`, `JsonNull`),
  the serializer (`dump`), and a recursive-descent parser (`JsonParser`) supporting standard
  JSON and an optional comment-tolerant mode (`enum JsonParse { STANDARD, COMMENTS }`,
  `json11.h:65`).

#### 3.10 Common toolbox — `include/LightGBM/utils/common.h`

**Purpose:** the catch-all header (~1264 lines) of free functions in `namespace Common` used
everywhere: string parsing, fast number parsing, joins/splits, math helpers, bitsets, timers.
It includes `json11.h`, `log.h`, `openmp_wrapper.h`, and vendors **fast_double_parser** and
**fmt** (`common.h:31-33`, `#define FMT_HEADER_ONLY`).

Representative functions (all `inline static` in `namespace Common`):
- String: `Trim(std::string)` (`:76`), `Split(const char*, char)` / `Split(const char*, const
  char*)` (`:102`/`:172`), `Join<T>(const std::vector<T>&, const char* delim)` (`:501`, with an
  `int8_t` specialization at `:519` and locale-forcing overloads), `RemoveQuotationSymbol`,
  `StartsWith`.
- Number parsing (parity-critical): `Atoi<T>(const char* p, T* out)` (`:224`),
  `Atof(const char* p, double* out)` (`:262`), `AtoiAndCheck(const char*, int*)` (`:375`),
  `AtofAndCheck(const char*, double*)` (`:383`), `AtofPrecise` (uses fast_double_parser).
  These back `Config::GetInt/GetDouble`.
- Array parsing: `StringToArray<T>(str, delimiter)` (`:431`), `StringToArray<T>(str, int n)`
  (`:454`), `StringToArrayofArrays<T>` (nested `[..],[..]`, used for interaction constraints),
  `ArrayCast<From,To>`, `ArrayToString`.
- Math: `Pow(base, int power)` (`:248`), `RoundInt(double)` (`:904`), `Softmax(std::vector<
  double>*)` / `Softmax(const double* in, double* out, int len)` (`:571`/`:587`), `AvoidInf`,
  `ObtainMinMaxSum`, `ParallelSort`.
- Bitsets: `ConstructBitset`, `FindInBitset`, `InsertBitset`, `EmptyBitset` (categorical-split
  encoding).
- Timing/profiling: `FunctionTimer` / `Timer` (RAII scope timers), `AlignmentAllocator` (a
  custom `std::allocator` honoring `kAlignedSize`).

---

### 4. Dependency map (who includes whom)

```
meta.h                        ← (leaf; only <cstdint>, STL) included by nearly everything
log.h                         ← (leaf; only STL / R headers)
openmp_wrapper.h              ← export.h, log.h
binary_writer.h               ← (leaf)
  ├─ file_io.h                ← binary_writer.h              → file_io.cpp (LocalFile)
  └─ byte_buffer.h            ← export.h, binary_writer.h
random.h                      ← (leaf)
json11.h                      ← (leaf)                       → json11.cpp
common.h                      ← json11.h, log.h, openmp_wrapper.h, fast_double_parser, fmt
threading.h                   ← meta.h, common.h, openmp_wrapper.h
array_args.h                  ← openmp_wrapper.h, threading.h
pipeline_reader.h             ← file_io.h, log.h
text_reader.h                 ← log.h, pipeline_reader.h, random.h
chunked_array.hpp             ← log.h
config.h                      ← export.h, meta.h, common.h, log.h
  config.cpp                  ← config.h, random.h, common.h, log.h, cuda/vector_cudahost.h
  config_auto.cpp             ← config.h    (GENERATED from config.h by parameter-generator.py)
```

**Consumers (outside this subsystem):** `Config` is read by the Application, Boosting/GBDT,
Objective, Metric, TreeLearner, Dataset/BinMapper, and Network layers. `meta.h` typedefs and
`log.h`/`common.h`/`threading.h`/`random.h` utilities are used pervasively throughout `src/`.
The file/text readers feed the `Dataset` loader; `json11` and `ByteBuffer` feed model
serialization; `ChunkedArray` feeds the C-API dataset builders.

---

### 5. Port implications (why this subsystem matters for the Rust rewrite)

1. **Type aliases are the contract.** `data_size_t=i32`, `score_t=f32`, `label_t=f32`,
   `comm_size_t=i32`, and the constants `kEpsilon=1e-15`, `kZeroThreshold=1e-35`,
   `kAlignedSize=32` must match exactly for numerical parity.
2. **The `Config` bag + alias/default/check tables** must be reproduced faithfully, including
   the *auto-correction* logic in `CheckParamConflict` and the alias-resolution priority
   (`SortAlias`). `config_auto.cpp` being generated means the Rust equivalent can also be
   generated or transcribed from the annotated `config.h`.
3. **Determinism** hinges on the exact `Random` LCG (`214013*x + 2531011`) and on the
   OpenMP-block splitting in `threading.h` — the Rust port maps these onto rayon/cubecl but must
   preserve the seed-derivation order in `Config::Set` and the block boundaries where they
   affect reduction order.


---

## Section C — Data I/O, Dataset, Binning & Tree Model Storage

This section documents the part of LightGBM that reads raw training data, transforms it
into the compact integer representation the training algorithm needs, and stores the tree
models that training produces. Everything here is C++ reference code under `LightGBM/`;
the Rust port must reproduce its behavior bit-for-bit (see project fidelity contract).

---

### 0. The core concept you must understand first: histogram-based GBDT

LightGBM is a **gradient-boosting decision tree (GBDT)** library. It builds an *ensemble*
(a sum) of decision trees. Each new tree is trained to correct the errors of the trees so
far. The training loop repeatedly asks two questions of the data:

1. For every training row, what is the **gradient** and **hessian** of the loss at the
   current prediction? (Jargon: the *gradient* is the first derivative of the loss with
   respect to the model's current score for that row — it tells the tree which direction to
   move; the *hessian* is the second derivative — it tells the tree how confident/curved
   the loss is there. Both are computed by the objective function, an upstream subsystem.
   Types: `score_t` = `float` by default.)
2. Given those gradients/hessians, where is the best place to **split** a group of rows into
   two children so the loss drops the most?

Naively, finding the best split for a continuous feature means sorting the feature's raw
values and trying every possible threshold. That is expensive and must be repeated for
every node of every tree. LightGBM's central trick is to **pre-bin once**:

- **Binning (jargon: "bin").** Before training starts, each feature's raw values (doubles)
  are quantized into a small number of integer buckets called *bins* — typically at most
  255 of them (`max_bin`). A `BinMapper` learns the bucket boundaries from a sample of the
  data. After that, every raw value is replaced by a small integer bin index (`uint8_t`,
  `uint16_t`, or `uint32_t`). This happens exactly once and the result is immutable.

- **Histogram (jargon: "histogram").** To find the best split of a set of rows on a feature,
  LightGBM does *not* re-scan raw values. Instead it builds a **histogram**: an array
  indexed by bin, where each entry accumulates the *sum of gradients* and *sum of hessians*
  of all rows that fall into that bin (plus a count). With at most `max_bin` entries per
  feature, the best split is found by a single left-to-right scan of the histogram,
  accumulating the running left-side sums. This is O(#bins) instead of O(#rows·log#rows).

- **Histogram subtraction trick.** After a node splits into two children, only the *smaller*
  child's histogram is built from scratch; the *larger* child's histogram is obtained by
  subtracting the smaller child's histogram from the parent's. This halves histogram cost.

- **Feature groups / feature bundling (jargon: "feature group", "EFB").** Sparse features
  that are rarely non-zero at the same time can be packed into one shared bin array (their
  bin ranges are offset so they don't collide). This is *Exclusive Feature Bundling*. A
  `FeatureGroup` owns the actual bin storage for one or more features.

- **Dense vs sparse storage.** A feature that is non-zero for most rows is stored *dense*
  (one bin value per row, `DenseBin`). A feature that is mostly zero is stored *sparse*
  (only non-zero entries, as delta-encoded run lengths, `SparseBin`).

- **Multi-value bin (jargon: "multi-val bin").** An alternative row-wise layout used when
  many features are bundled together and the histogram is built by iterating over the
  (few) non-default values present in each row rather than column-by-column.

Keep this pipeline in mind — it is the skeleton the rest of this document hangs on:

```
raw file / matrix
   → Parser (text → (column, value) pairs + label)
   → sample rows → BinMapper::FindBin  (learn bin boundaries per feature)
   → Dataset::Construct  (decide feature groups, allocate Bin storage)
   → push every row: BinMapper::ValueToBin → FeatureGroup::PushData → Bin::Push
   → Dataset::FinishLoad  (dataset now IMMUTABLE)
   → [training loop] Dataset::ConstructHistograms  (Bin::ConstructHistogram)
        → tree learner scans histograms, picks split
        → Dataset::Split partitions row indices
        → Tree::Split records the node
   → Tree::AddPredictionToScore / Tree::Predict at inference
```

---

### 1. `include/LightGBM/bin.h` — binning interfaces (the heart of the subsystem)

**Purpose.** Declares the three central abstractions of the binned representation:
`BinMapper` (value→bin conversion + metadata), `Bin` (per-feature bin storage +
histogram construction + split partitioning), and `MultiValBin` (row-wise bundled
storage). It also defines the histogram data types.

**Histogram types** (`bin.h:33-46`):
- `typedef double hist_t;` — a histogram entry is a `double`. Gradients/hessians are `float`
  (`score_t`) but are *accumulated* into `double` histograms for precision.
- Histograms are stored as a flat `hist_t*` array where each bin occupies **two** doubles:
  `GET_GRAD(hist, i)` is `hist[i<<1]` (sum of gradients for bin i) and
  `GET_HESS(hist, i)` is `hist[(i<<1)+1]` (sum of hessians for bin i). `kHistEntrySize` =
  `2*sizeof(hist_t)` = 16 bytes. The count reuses the hessian slot when hessians are
  constant (`hist_cnt_t = uint64_t`, same width as `hist_t`).
- Integer histogram variants (`int_hist_t`, Int8/Int16/Int32) exist for the quantized-gradient
  training mode; ignore them unless you touch `use_quantized_grad`.

**Enums** (`bin.h:22-31`): `BinType { NumericalBin, CategoricalBin }`;
`MissingType { None, Zero, NaN }` — how missing values are treated for this feature.

#### `class BinMapper` (`bin.h:85-259`)

Maps one feature's raw `double` value to its integer bin, and stores the metadata needed to
interpret bins. One `BinMapper` per feature.

Key members (private, `bin.h:236-258`): `int num_bin_`, `MissingType missing_type_`,
`std::vector<double> bin_upper_bound_` (the split boundaries for numerical features —
`bin i` covers values `<= bin_upper_bound_[i]`), `BinType bin_type_`,
`std::unordered_map<int,unsigned int> categorical_2_bin_` + `std::vector<int> bin_2_categorical_`
(categorical mapping), `double min_val_/max_val_`, `uint32_t default_bin_` (the bin that
value 0 maps to), `uint32_t most_freq_bin_` (the most common bin — used as the "skip" bin).

Key methods:
- `void FindBin(double* values, int num_values, size_t total_sample_cnt, int max_bin,
  int min_data_in_bin, int min_split_data, bool pre_filter, BinType bin_type,
  bool use_missing, bool zero_as_missing, const std::vector<double>& forced_upper_bounds)`
  (`bin.h:201`) — **fits** the mapper from sampled values. See `bin.cpp` below.
- `inline uint32_t ValueToBin(double value) const` (`bin.h:173`, defined inline at
  `bin.h:612-650`) — **the hot conversion**. For numerical features it does a binary search
  over `bin_upper_bound_` to find the first bin whose upper bound is `>= value`. NaN maps to
  the last bin when `missing_type_==NaN`, else to 0. For categorical features it looks up
  `categorical_2_bin_` (negative or unseen categories → bin 0, the NaN bin).
- `inline double BinToValue(uint32_t bin) const` (`bin.h:138`) — inverse: returns the bin's
  upper bound (numerical) or the category integer (categorical). Used to write real
  thresholds into the tree model.
- `inline int num_bin()`, `missing_type()`, `is_trivial()` (feature has ≤1 useful bin, i.e.
  useless — will be dropped), `sparse_rate()`, `GetDefaultBin()`, `GetMostFreqBin()`.
- `CopyTo(char*)` / `CopyFrom(const char*)` / `SaveBinaryToFile(BinaryWriter*)` /
  `SizesInByte()` — serialization to the binary dataset format.
- `CheckAlign(const BinMapper& other)` — validates a validation-set feature bins identically
  to the training set.

#### `class BinIterator` (`bin.h:262-273`)

Abstract cursor over one feature's bins. `virtual uint32_t Get(data_size_t idx)` returns the
bin at row `idx` (remapped into the feature's `[min_bin,max_bin]` window with most-freq bin
handling); `RawGet` returns the raw stored value; `Reset(idx)` rewinds. Used by prediction
and by row-wise iteration.

#### `class Bin` (`bin.h:281-478`)

Abstract storage for one feature's (or one feature group's) bin column, in **original row
order**. This is where histograms are built and rows are partitioned. Key virtuals:

- `virtual void Push(int tid, data_size_t idx, uint32_t value)` (`bin.h:297`) — write bin
  `value` at row `idx` during load (thread id `tid` for parallel push).
- `virtual BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin,
  uint32_t most_freq_bin) const` (`bin.h:307`).
- **Histogram construction** (`bin.h:350-421`), the core training primitive. Two shapes,
  each with `float`-hessian and quantized-int variants:
  - `void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
    data_size_t end, const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const` — for the subset of rows named in `data_indices[start..end)`.
  - `void ConstructHistogram(data_size_t start, data_size_t end, const score_t*
    ordered_gradients, const score_t* ordered_hessians, hist_t* out) const` — for a
    contiguous range (all rows, no index indirection).
  - The `ordered_gradients`/`ordered_hessians` arrays are **pre-gathered** so that element
    `i` corresponds to `data_indices[i]`. This is a cache optimization: the histogram loop
    reads gradients sequentially instead of gather-scattering through `gradients[idx]`
    (explained in the doc comment at `bin.h:336-348`). Output `out` is the `hist_t*` grad/hess
    pair array described above.
- **Split / partition** (`bin.h:423-447`):
  - `virtual data_size_t Split(uint32_t min_bin, uint32_t max_bin, uint32_t default_bin,
    uint32_t most_freq_bin, MissingType missing_type, bool default_left, uint32_t threshold,
    const data_size_t* data_indices, data_size_t cnt, data_size_t* lte_indices,
    data_size_t* gt_indices) const` — partitions the `cnt` rows in `data_indices` into
    `lte_indices` (bin ≤ threshold → go left) and `gt_indices` (go right); returns the left
    count. `default_left` routes missing/default-bin rows. A second overload omits `min_bin`
    for single-feature groups; `SplitCategorical` variants take a bitset `threshold`.
- `virtual void FinishLoad()` — finalize storage after all pushes (e.g. sparse bins compress).
- Factories: `static Bin* CreateDenseBin(data_size_t num_data, int num_bin)` and
  `static Bin* CreateSparseBin(...)` (`bin.h:460-468`) — pick storage width based on
  `num_bin` (≤16 → 4-bit, ≤256 → `uint8_t`, ≤65536 → `uint16_t`, else `uint32_t`).
- `virtual Bin* Clone()`, `CopySubrow(...)` (build a bagging subset).

#### `class MultiValBin` (`bin.h:481-610`)

Row-wise counterpart of `Bin` for bundled features. `PushOneRow(tid, idx, values)` pushes a
whole row's non-default bin values at once; `ConstructHistogram(...)` (and Ordered/IntN
variants) build histograms by walking each row's short value list. Factories choose dense
(`CreateMultiValDenseBin`) vs sparse (`CreateMultiValSparseBin`) by
`multi_val_bin_sparse_threshold = 0.25`.

**Depended on by:** `feature_group.h`, `dataset.h`, `train_share_states.h`, tree learners
(upstream). **Depends on:** `meta.h` (`data_size_t`, `score_t`), `utils/common.h`, file I/O.

---

### 2. `src/io/bin.cpp` — `BinMapper::FindBin` and the bin-boundary algorithm

**Purpose.** Implements how a feature's bin boundaries are *learned* from sampled data — the
single most fidelity-sensitive routine in this subsystem (binning must be bit-exact vs C++).

#### `void BinMapper::FindBin(...)` (`bin.cpp:311-506`)

Inputs: `double* values` (sampled non-zero values for this feature, mutated in place),
`int num_sample_values`, `size_t total_sample_cnt` (= non-zeros + zeros), `int max_bin`,
`int min_data_in_bin`, `int min_split_data`, `bool pre_filter`, `BinType bin_type`,
`bool use_missing`, `bool zero_as_missing`, `const std::vector<double>& forced_upper_bounds`.

Steps:
1. Strip NaNs; decide `missing_type_` (`None` / `Zero` if `zero_as_missing` / `NaN` if NaNs
   present) (`bin.cpp:315-334`).
2. `std::stable_sort` the values, then build parallel arrays `distinct_values` +
   `counts` (count per distinct value), inserting a synthetic zero entry with the implied
   zero count in the right sorted position (`bin.cpp:343-375`).
3. **Numerical** (`bin.cpp:380-409`): call `FindBinWithZeroAsOneBin` → `GreedyFindBin` to
   compute `bin_upper_bound_`. If `missing_type_==NaN`, one bin is reserved for NaN (last bin,
   upper bound = NaN). `num_bin_ = bin_upper_bound_.size()`.
4. **Categorical** (`bin.cpp:410-476`): convert to ints, drop negatives (→NaN), sort
   categories by descending count, keep the most frequent until 99% of data is covered or
   `max_bin` reached, building `categorical_2_bin_`/`bin_2_categorical_`; bin 0 is the NaN bin.
5. Mark `is_trivial_` if `num_bin_ <= 1` or (with `pre_filter`) if a `NeedFilter` check fails
   — trivial features are later dropped by `Dataset::Construct` (`bin.cpp:480-488`).
6. Compute `default_bin_ = ValueToBin(0)`, `most_freq_bin_ = argmax(cnt_in_bin)`, and
   `sparse_rate_`. Note the subtle rule (`bin.cpp:498-500`): if the most-frequent bin differs
   from the zero/default bin but the feature isn't very sparse (`< kSparseThreshold=0.7`),
   `most_freq_bin_` is forced back to `default_bin_` to save load cost.

#### `std::vector<double> GreedyFindBin(...)` (`bin.cpp:78-155`)

The actual boundary-selection algorithm. If distinct values ≤ `max_bin`, each distinct value
(respecting `min_data_in_bin`) gets its own bin, with boundaries at midpoints
`GetDoubleUpperBound((v[i]+v[i+1])/2)`. Otherwise it targets `mean_bin_size = total/max_bin`
rows per bin, treats high-count values as their own bins, and greedily closes a bin when it
reaches the mean size. `FindBinWithZeroAsOneBin` (`bin.cpp:242-309`) wraps this to force a
dedicated zero bin, and `FindBinWithPredefinedBin` (`bin.cpp:157+`) honors user-forced split
points. **This midpoint/`GetDoubleUpperBound` arithmetic is what the Rust port must match to
the bit.**

Also implements `CopyTo`/`CopyFrom`/`SaveBinaryToFile`/`SizesInByte` for `BinMapper`.

---

### 3. `src/io/dense_bin.hpp` — dense per-feature storage

**Purpose.** `template <typename VAL_T, bool IS_4BIT> class DenseBin : public Bin`
(`dense_bin.hpp:53`) stores one bin value per row in a flat `std::vector<VAL_T>` (`data_`),
where `VAL_T` ∈ {`uint8_t`,`uint16_t`,`uint32_t`}. The `IS_4BIT` specialization packs two
4-bit bins per byte (`dense_bin.hpp:59-63,73`). This is the common case for non-sparse
features.

- `Push(tid, idx, value)` (`~dense_bin.hpp:70-81`) writes `data_[idx] = value` (or packs the
  nibble for 4-bit).
- **`ConstructHistogramInner<USE_INDICES, USE_PREFETCH, USE_HESSIAN>`** (`dense_bin.hpp:99-141`)
  — the templated inner loop. For each row it computes `ti = data(idx) << 1` (bin index times
  two, because grad and hess are interleaved) and does `grad[ti] += ordered_gradients[i];
  hess[ti] += ordered_hessians[i];` (or `++cnt[ti]` when hessians are constant). Software
  prefetch (`PREFETCH_T0`) hides memory latency. The public `ConstructHistogram` overloads
  (`dense_bin.hpp:143-171`) just instantiate this with the right template flags; the IntN
  overloads use `ConstructHistogramIntInner` (`dense_bin.hpp:175+`).
- **`SplitInner<...>`** (`dense_bin.hpp:316-394`) — partition. It computes the threshold in
  the feature's own bin window (`th = threshold + min_bin`, adjusted when `most_freq_bin==0`),
  then loops over the rows: rows equal to the zero/NaN bin go to the `default`/`missing`
  direction (chosen by `default_left`), rows with `bin > th` go to `gt_indices`, else
  `lte_indices`. Returns the left count. `Split`/`SplitCategorical` public overloads
  (`dense_bin.hpp:396-500`) dispatch to the right template instantiation based on
  missing-type flags.
- `GetColWiseData(...)` exposes the raw column for GPU upload (out of scope here).

---

### 4. `src/io/sparse_bin.hpp` — sparse per-feature storage

**Purpose.** `template <typename VAL_T> class SparseBin : public Bin`
(`sparse_bin.hpp:73`) stores only non-default entries as two parallel arrays: `vals_`
(the bin values) and `deltas_` (run-length gaps between consecutive stored row indices), so a
mostly-zero feature costs memory proportional to its non-zeros, not to `num_data`. Chosen when
`sparse_rate() >= kSparseThreshold (0.7)`.

- `Push` (`sparse_bin.hpp:92`) buffers into per-thread `push_buffers_`; `FinishLoad`
  merges/sorts them into the delta-encoded `deltas_`/`vals_`.
- `ConstructHistogram` (`sparse_bin.hpp:107+`) walks the sparse entries: it advances a
  running `cur_pos` by `deltas_[++i_delta]` to reach the next stored row, reads
  `bin = vals_[i_delta]`, and accumulates into `out`. Rows not stored are implicitly the
  default bin (they contribute to the most-freq bin, which is fixed up later by
  `Dataset::FixHistogram`).
- `SparseBinIterator` (`sparse_bin.hpp:28`) is the matching cursor. `Split`/`SplitCategorical`
  mirror the dense versions but operate over the sparse layout.

---

### 5. `src/io/multi_val_dense_bin.hpp` & `multi_val_sparse_bin.hpp` — bundled row-wise storage

**Purpose.** Implement `MultiValBin` for the row-wise histogram path (used when
`force_row_wise`, or when a bundle is dense/sparse enough). Instead of one array per feature,
these store, per row, the list of active bins across all bundled features, with per-feature
`offsets_` so bins from different features occupy disjoint ranges of the shared histogram.

- `MultiValDenseBin` (`multi_val_dense_bin.hpp:3`): `data_` is a flat
  `num_data * num_feature` array (`multi_val_dense_bin.hpp:9`); every row has exactly
  `num_feature_` entries (`num_element_per_row()` returns `num_feature_`). `PushOneRow`
  writes the row's `num_feature` bin values contiguously (`multi_val_dense_bin.hpp:27`).
  `ConstructHistogram` iterates rows, and within each row iterates the `num_feature` bins,
  adding `offsets_[j]` to place each into the shared histogram.
- `MultiValSparseBin` (`multi_val_sparse_bin.hpp`): stores a variable-length list per row
  (CSR-like), for bundles that are sparse; `num_element_per_row()` is an estimate.

Both provide the Ordered/Int8/Int16/Int32 histogram variants. Consumed by
`MultiValBinWrapper` in `train_share_states`.

---

### 6. `include/LightGBM/feature_group.h` — bundling features into one storage unit

**Purpose.** `class FeatureGroup` (`feature_group.h:26`) owns the `Bin` storage for a set of
features that share one histogram region. It is the bridge between per-feature `BinMapper`s
and the physical `Bin`/`MultiValBin` storage. `Dataset` holds a vector of these.

Key members (`feature_group.h:614-627`): `int num_feature_`;
`std::vector<std::unique_ptr<BinMapper>> bin_mappers_` (one per feature in the group);
`std::vector<uint32_t> bin_offsets_` (where each sub-feature's bins start inside the shared
range); `std::unique_ptr<Bin> bin_data_` (single-value path) **or**
`std::vector<std::unique_ptr<Bin>> multi_bin_data_` (multi-val path);
`bool is_multi_val_/is_dense_multi_val_/is_sparse_`; `int num_total_bin_`.

Key methods:
- **Constructor** `FeatureGroup(int num_feature, int8_t is_multi_val,
  std::vector<std::unique_ptr<BinMapper>>* bin_mappers, data_size_t num_data, int group_id)`
  (`feature_group.h:39-76`) — takes ownership of the bin mappers, computes `bin_offsets_`
  (each feature's bins laid end-to-end; bin 0 of the group is reserved for the most-freq/
  default bin unless dense-multi-val), sets `num_total_bin_`, and calls `CreateBinData`.
- `void CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse)`
  (`feature_group.h:586-612`) — picks `DenseBin` vs `SparseBin` (per feature's
  `sparse_rate() >= kSparseThreshold`) and allocates.
- **`inline void PushData(int tid, int sub_feature_idx, data_size_t line_idx, double value)`**
  (`feature_group.h:253-267`) — the per-value load path: converts `value`→bin via
  `bin_mappers_[sub]->ValueToBin`, **skips** it if it equals the most-freq bin (that's the
  implicit default), decrements by 1 if `most_freq_bin==0`, adds `bin_offsets_[sub]` for the
  single-value path, and calls `bin_data_->Push` (or `multi_bin_data_[sub]->Push`).
- **`inline data_size_t Split(int sub_feature, const uint32_t* threshold, int num_threshold,
  bool default_left, const data_size_t* data_indices, data_size_t cnt, data_size_t*
  lte_indices, data_size_t* gt_indices) const`** (`feature_group.h:398-444`) — resolves the
  sub-feature's `[min_bin,max_bin]` window and forwards to the underlying `Bin::Split`/
  `SplitCategorical`.
- `SubFeatureIterator`, `FeatureGroupIterator`, `BinToValue`, `feature_min_bin`/`feature_max_bin`.
- `SerializeToBinary` / `SizesInByte` / memory-constructors for the binary dataset format.
- `AddFeaturesFrom` — merge another group's features (used by `Dataset::AddFeaturesFrom`).

**Depends on:** `bin.h`. **Used by:** `Dataset`, `DatasetLoader`, `TrainingShareStates`
(declared as friends at `feature_group.h:28-31`).

---

### 7. `include/LightGBM/dataset.h` + `src/io/dataset.cpp` — the immutable binned dataset

**Purpose.** `class Dataset` (`dataset.h:487`) is *the* container the training loop reads
from. It owns all `FeatureGroup`s (hence all bin data), the `Metadata` (labels/weights/etc.),
and the feature-index bookkeeping. After `FinishLoad()` it is read-only; the only per-tree
mutable structures live in the tree learner (histograms, partitions).

#### Feature-index bookkeeping (private members, `dataset.h:1007-1051`)

Because trivial features are dropped and features are bundled, several index spaces coexist:
- **real/original feature index** — column position in the source file.
- **inner/used feature index** — dense 0..`num_features_-1` over kept features.
- `used_feature_map_[real] → inner` (or -1 if dropped); `real_feature_idx_[inner] → real`;
  `feature2group_[inner] → group`; `feature2subfeature_[inner] → position in group`;
  `group_bin_boundaries_[group]` gives each group's start bin in the global flat histogram.
Helper accessors: `RealFeatureIndex`, `InnerFeatureIndex`, `Feature2Group`,
`Feture2SubFeature`, `FeatureBinMapper(i)` (`dataset.h:796-800`), `FeatureNumBin`,
`RealThreshold`/`BinThreshold` (bin↔value for the tree model).

#### `void Dataset::Construct(...)` (`dataset.h:495`, `dataset.cpp:325-441`)

Inputs: `std::vector<std::unique_ptr<BinMapper>>* bin_mappers` (one per original feature,
already fitted), `int num_total_features`, `const std::vector<std::vector<double>>&
forced_bins`, sampled sparse data (`int** sample_non_zero_indices`, `double** sample_values`,
`const int* num_per_col`, `int num_sample_col`, `size_t total_sample_cnt`),
`const Config& io_config`. What it does:
1. Collect non-trivial features into `used_features` (`dataset.cpp:339-343`).
2. Decide grouping: `OneFeaturePerGroup` or, if `enable_bundle`, `FastFeatureBundling`
   (Exclusive Feature Bundling) which also decides per-group multi-val (`dataset.cpp:350-368`).
3. For each group, move the relevant `BinMapper`s in, fill the index maps, and construct a
   `FeatureGroup` (`dataset.cpp:387-411`). Accumulate `group_bin_boundaries_`.
4. Record config knobs (`max_bin_`, `min_data_in_bin_`, `use_missing_`, `zero_as_missing_`,
   `has_raw_` for linear trees, numeric-feature map) (`dataset.cpp:412-440`).

Note: `Construct` sets up structure only; actual bin values are pushed afterward via the
`PushOneRow`/`PushOneValue`/`FinishOneRow` inline methods (`dataset.h:556-619`), which route
each value through `feature_groups_[group]->PushData`.

#### `void Dataset::FinishLoad()` (`dataset.h:673`, `dataset.cpp:443-463`)

Calls `FinishLoad()` on every `FeatureGroup` (which compresses sparse bins etc.) and on
`metadata_`, then sets `is_finish_load_ = true`. **After this the dataset is immutable.**

#### Histogram construction (the training hot path)

- `template<bool USE_QUANT_GRAD,int HIST_BITS> void ConstructHistograms(const
  std::vector<int8_t>& is_feature_used, const data_size_t* data_indices, data_size_t num_data,
  const score_t* gradients, const score_t* hessians, score_t* ordered_gradients,
  score_t* ordered_hessians, TrainingShareStates* share_state, hist_t* hist_data) const`
  (inline, `dataset.h:726-758`) — public entry. Decides whether to use row-index indirection
  and whether hessians are constant, then dispatches to `ConstructHistogramsInner`.
- `ConstructHistogramsInner<USE_INDICES,USE_HESSIAN,...>` (`dataset.cpp:1262-1462`) — for the
  **column-wise** path: first *gathers* `gradients[data_indices[i]]` into the contiguous
  `ordered_gradients` buffer (`dataset.cpp:1310-1326`, the cache trick), then, in a
  `#pragma omp parallel for` over used dense groups, zeroes each group's histogram region
  (`hist_data + group_bin_boundaries_[group]*2`) and calls
  `feature_groups_[group]->bin_data_->ConstructHistogram(...)` (`dataset.cpp:1328-1401`).
  The **row-wise** path delegates to `ConstructHistogramsMultiVal` →
  `TrainingShareStates::ConstructHistograms`.
- `void FixHistogram(int feature_idx, double sum_gradient, double sum_hessian, hist_t* data)`
  (`dataset.cpp:1488-1506`) — because rows in the most-frequent bin were **skipped** during
  push (they're the implicit default), that bin's entry is empty. This reconstructs it by
  subtracting all other bins from the node's known totals: `hist[most_freq] = sum_total −
  Σ(other bins)`. Essential for correctness.

#### Splitting & other

- `inline data_size_t Split(int feature, const uint32_t* threshold, int num_threshold,
  bool default_left, const data_size_t* data_indices, data_size_t cnt, data_size_t*
  lte_indices, data_size_t* gt_indices) const` (`dataset.h:765-775`) — forwards to the
  feature's `FeatureGroup::Split`. This is how the tree learner physically partitions the
  row-index array after choosing a split.
- `GetShareStates<USE_QUANT_GRAD,HIST_BITS>(...)` (`dataset.h:667`, `dataset.cpp:612`) — builds
  the `TrainingShareStates` (multi-val bin, offsets, col-vs-row decision).
- `CopySubrow` / `CreateValid` / `AddFeaturesFrom` / `SaveBinaryFile` / `SerializeReference`
  / `DumpTextFile` — bagging subsets, validation alignment, and serialization.
- Field setters/getters: `SetFloatField`/`SetIntField`/`GetFloatField` etc. route label/
  weight/query/init-score into `Metadata`.

**Depends on:** `feature_group.h`, `bin.h`, `metadata` (declared in this header),
`train_share_states.h`, `config.h`, `arrow.h`. **Used by:** boosting (`GBDT`), tree learners,
metrics, the C API (`LGBM_Dataset*`).

---

### 8. `Metadata` (declared in `dataset.h:48-397`, implemented in `src/io/metadata.cpp`)

**Purpose.** Holds the non-feature training data: **labels** (required), **weights**
(optional per-row importance), **initial scores** (optional warm-start, one column per class),
and **query boundaries** (for ranking/LambdaRank — rows are grouped into "queries", and
`query_boundaries_[i]..query_boundaries_[i+1]` are the rows of query *i*). Also positions
(for position-bias) and derived `query_weights_`.

Members (`dataset.h:361-393`): `std::vector<label_t> label_` (label_t=`float`),
`std::vector<label_t> weights_`, `std::vector<data_size_t> query_boundaries_`,
`std::vector<label_t> query_weights_`, `std::vector<double> init_score_`,
`std::vector<data_size_t> queries_`, plus counts and `load_from_file_` flags.

Key methods:
- `void Init(const char* data_filename)` (`metadata.cpp:29`) — loads query/weight side files
  early (queries are needed *before* sampling because sampling must respect query groups).
- `void Init(data_size_t num_data, int weight_idx, int query_idx)` (`metadata.cpp:42`) —
  allocate label/weight/query arrays once row count is known.
- `void Init(data_size_t num_data, int32_t has_weights, int32_t has_init_scores,
  int32_t has_queries, int32_t nclasses)` (`metadata.cpp:73`) — streaming/init-by-flags.
- `void SetLabel(const label_t*, data_size_t)` / `SetWeights` / `SetQuery` / `SetInitScore`
  (+ Arrow overloads) — bulk setters used by the C API.
- `inline void SetInitScoreAt(data_size_t idx, const double* values)` (`dataset.h:172-178`) —
  init scores are stored **column-major** (`init_score_[class*num_data + row]`).
- `void CheckOrPartition(data_size_t num_all_data, const std::vector<data_size_t>&
  used_data_indices)` (`metadata.cpp:211`) — subset metadata for distributed/bagging.
- `void FinishLoad()` (`metadata.cpp:779-781`) — calls `CalculateQueryBoundaries()`.
- `CalculateQueryWeights()` (`metadata.cpp:742`) — derive per-query weights from row weights.
- Accessors: `label()`, `weights()`, `query_boundaries()`, `query_weights()`, `init_score()`,
  `num_queries()`, `num_init_score_classes()`.
- `SaveBinaryToFile` / `LoadFromMemory` (`metadata.cpp:790,823`) — binary format;
  **init_score is not persisted** (warning at `metadata.cpp:836`).

Copy is disabled (`dataset.h:317-319`). **Used by:** objective functions (labels/weights),
metrics, ranking, boosting.

---

### 9. `include/LightGBM/dataset.h` (Parser interface) + `src/io/parser.hpp` + `src/io/parser.cpp`

**Purpose.** Turn raw text lines into `(column_index, value)` pairs plus a label.

`class Parser` (abstract, `dataset.h:401-460`):
- `virtual void ParseOneLine(const char* str, std::vector<std::pair<int,double>>*
  out_features, double* out_label) const` (`dataset.h:422`) — parse one row. Note the output
  is a **sparse** list: only non-(near-)zero columns are emitted.
- `virtual int NumFeatures() const`.
- `static Parser* CreateParser(const char* filename, bool header, int num_features,
  int label_idx, bool precise_float_parser [, std::string parser_config_str])`
  (`dataset.h:436/448`) — factory that sniffs the file format.

Concrete parsers in `parser.hpp`:
- `CSVParser` (`parser.hpp:18-54`) — comma-separated; column at `label_idx_` becomes the
  label; other near-zero values are dropped (`std::fabs(val) > kZeroThreshold`), giving sparse
  output. Uses an injected `AtofFunc` (fast or precise float parsing).
- `TSVParser` (`parser.hpp:56-91`) — tab-separated, same logic.
- `LibSVMParser` (`parser.hpp:93-132`) — `label idx:val idx:val …`; label must be first
  column.

`parser.cpp` picks the format: `GetStatistic` (`parser.cpp:15`) counts commas/tabs/colons per
line and `GetLabelIdxFor{CSV,TSV,Libsvm}` heuristics decide the format and whether a label
column is present, then `CreateParser` instantiates the right subclass and chooses the atof
function. `ParserFactory`/`ParserReflector` (`dataset.h:463-482`) allow user-registered custom
parsers.

**Used by:** `DatasetLoader` only (parsing → sampling → binning → feature extraction).

---

### 10. `include/LightGBM/dataset_loader.h` + `src/io/dataset_loader.cpp` — building a Dataset from a file

**Purpose.** `class DatasetLoader` (`dataset_loader.h:17`) orchestrates the whole load: read
text, sample rows, fit `BinMapper`s, construct the `Dataset`, and push all rows. It is the
top-level driver of this subsystem for the file path.

Constructor: `DatasetLoader(const Config& io_config, const PredictFunction& predict_fun,
int num_class, const char* filename)` (`dataset_loader.h:19`). It reads config to set
`label_idx_`, `weight_idx_`, `group_idx_`, `ignore_features_`, `categorical_features_`,
`feature_names_`, `store_raw_`.

#### `Dataset* LoadFromFile(const char* filename, int rank, int num_machines)` (`dataset_loader.cpp:203-...`)

The main path (single-machine `rank=0,num_machines=1` via the inline overload
`dataset_loader.h:25`):
1. `CheckCanLoadFromBin` — if a pre-binned `.bin` exists, load it directly (skips binning).
2. Else create a `Parser` (`dataset_loader.cpp:221`) and `metadata_.Init(filename)`.
3. **One-pass** (`!two_round`): `LoadTextDataToMemory` reads all lines; `SampleTextDataFromMemory`
   picks `bin_construct_sample_cnt` sample rows; `ConstructBinMappersFromTextData` fits the
   bins and constructs the dataset; `metadata_.Init(num_data, weight_idx, group_idx)`;
   `ExtractFeaturesFromMemory` pushes every row's bins (`dataset_loader.cpp:229-247`).
4. **Two-pass** (`two_round`, for data too big to hold twice): sample from file, build bins,
   then re-read the file to extract features (`dataset_loader.cpp:248-269`). This keeps memory
   down at the cost of reading the file twice.

#### `void ConstructBinMappersFromTextData(...)` (`dataset_loader.cpp:1070-...`)

Parses the sample rows into per-column `sample_values`/`sample_indices`
(`dataset_loader.cpp:1078-1092`), determines `num_total_features_`, applies forced bins and
`max_bin_by_feature`, then (further down) calls `BinMapper::FindBin` **per feature, in
parallel**, and finally `dataset->Construct(...)` with the fitted mappers. This is the
sample→`FindBin`→`Construct` handoff.

#### `ExtractFeaturesFromMemory` / `ExtractFeaturesFromFile` (`dataset_loader.cpp:1263`, `:70`)

Iterate every row, `ParseOneLine`, and `dataset->PushOneRow(...)` (which routes to
`FeatureGroup::PushData`). Set labels/weights/queries into metadata. Then the caller invokes
`dataset->FinishLoad()`.

Other entry points: `LoadFromFileAlignWithOtherDataset` (validation set must bin identically
to train — uses the train dataset's mappers), `LoadFromSerializedReference`,
`ConstructFromSampleData` (used by the C API when data comes from an in-memory matrix rather
than a file).

**Depends on:** `Dataset`, `Parser`, `BinMapper`, `Metadata`, `Network` (distributed).
**Used by:** the C API `LGBM_DatasetCreateFromFile` and the CLI.

---

### 11. `include/LightGBM/train_share_states.h` + `src/io/train_share_states.cpp` — shared per-training scratch state

**Purpose.** `struct TrainingShareStates` (`train_share_states.h:268-362`) and
`class MultiValBinWrapper` (`train_share_states.h:20-266`) hold the reusable buffers and the
row-wise (multi-val) histogram machinery for one training run. This is where the row-wise
histogram path actually lives.

`TrainingShareStates` members: `bool is_col_wise` (column-wise vs row-wise histogram
strategy), `bool is_constant_hessian` (lets the code skip storing hessians and just count),
`const data_size_t* bagging_use_indices` + `bagging_indices_cnt`, a
`std::unique_ptr<MultiValBinWrapper> multi_val_bin_wrapper_`, an aligned scratch
`hist_buf_` (`std::vector<hist_t, AlignmentAllocator<hist_t,kAlignedSize>>`), and
`feature_hist_offsets_` (where each feature's bins live in the global histogram).

Key methods:
- `void CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
  std::vector<uint32_t>* offsets, bool is_col_wise)` (`train_share_states.cpp`) — compute the
  flat histogram layout offsets.
- `void SetMultiValBin(MultiValBin* bin, data_size_t num_data, ..., int num_grad_quant_bins)`.
- `template<...> void ConstructHistograms(const data_size_t* data_indices, data_size_t
  num_data, const score_t* gradients, const score_t* hessians, hist_t* hist_data)`
  (`train_share_states.h:310-320`) — the row-wise entry, delegates to the wrapper.

`MultiValBinWrapper::ConstructHistograms<USE_INDICES,ORDERED,USE_QUANT_GRAD,HIST_BITS>`
(`train_share_states.h:48-101`) splits the rows into blocks (`Threading::BlockInfo`), builds a
partial histogram per block in `hist_buf` in parallel (`ConstructHistogramsForBlock`,
`train_share_states.h:103-205`, which calls the `MultiValBin::ConstructHistogram*` methods),
then `HistMerge` sums the per-block partials and `HistMove` copies the result into the
canonical `origin_hist_data_`. Subset column/row support (`is_use_subcol_`/`is_use_subrow_`)
handles bagging and feature subsampling.

**Depends on:** `bin.h`, `feature_group.h`, `utils/threading.h`. **Used by:** `Dataset`
(holds/produces it) and the serial tree learner (owns one, passes it into
`Dataset::ConstructHistograms`).

---

### 12. `include/LightGBM/tree.h` + `src/io/tree.cpp` — the decision-tree model

**Purpose.** `class Tree` (`tree.h:26`) is the trained decision tree: an array-of-structs
representation of nodes, splits, and leaf outputs, plus prediction and (de)serialization. The
tree learner *produces* a `Tree`; boosting *stores* many and *sums* their predictions.

#### Node representation (private members, `tree.h:478-536`)

LightGBM uses a compact parallel-array layout, **not** pointer nodes. A tree with `L` leaves
has `L-1` internal (split) nodes indexed `0..L-2`. Internal-node arrays:
`left_child_[node]`, `right_child_[node]` (an index ≥0 is another internal node; a **negative**
value `~leaf` encodes a leaf — see `data_count` at `tree.h:181`), `split_feature_[node]`
(original feature index), `split_feature_inner_[node]` (inner index),
`threshold_in_bin_[node]` (`uint32_t` bin threshold), `threshold_[node]` (`double` real-value
threshold), `decision_type_[node]` (`int8_t` bitfield: categorical? default-left? missing
type — via masks `kCategoricalMask`, `kDefaultLeftMask` at `tree.h:20-21` and the
`GetDecisionType`/`GetMissingType` helpers `tree.h:262-281`), `split_gain_[node]`,
`internal_value_/weight_/count_[node]`. Leaf arrays (indexed by leaf id):
`leaf_value_[leaf]` (the **output** added to the score), `leaf_weight_`, `leaf_count_`,
`leaf_parent_`, `leaf_depth_`. Linear-tree extras: `leaf_const_`, `leaf_coeff_`,
`leaf_features_`.

#### Growing the tree

- **`int Split(int leaf, int feature, int real_feature, uint32_t threshold_bin,
  double threshold_double, double left_value, double right_value, int left_cnt, int right_cnt,
  double left_weight, double right_weight, float gain, MissingType missing_type,
  bool default_left)`** (`tree.h:63`, `tree.cpp:61-75`) — turns leaf `leaf` into an internal
  node with two new leaves. It calls the protected `Split(...)` core (`tree.h:543-585`) which
  rewires the parent pointer, creates the new internal node at index `num_leaves_-1`, sets the
  two children to `~leaf` and `~num_leaves_`, copies counts/weights/values, then this outer
  overload records `decision_type_`, `threshold_in_bin_`, `threshold_`, and returns the new
  leaf id. Returns the index of the new right leaf.
- **`int SplitCategorical(...)`** (`tree.h:86`, `tree.cpp:77-98`) — same but stores a category
  bitset in `cat_threshold_`/`cat_boundaries_` and marks `kCategoricalMask`.
- `SetLeafOutput(int leaf, double output)` (`tree.h:94`) — set a leaf's prediction (rounds
  tiny values to zero via `MaybeRoundToZero`).
- `Shrinkage(double rate)` (`tree.h:188`) — multiply all outputs by the learning rate;
  `AddBias(double)`; `AsConstantTree(double val)` (a stump, e.g. when no useful split found).

#### Prediction

- `inline double Predict(const double* feature_values) const` (`tree.h:134`, defined
  `tree.h:587-615`) — walk from the root: at each node `GetLeaf` (`tree.h:701-713`) follows
  `NumericalDecision`/`CategoricalDecision` (`tree.h:337-407`) — compare
  `feature_values[split_feature_[node]]` to `threshold_[node]` (with missing-value routing)
  until it reaches a leaf (`node<0`), then returns `LeafOutput(~node)`. Linear trees add the
  per-leaf linear model.
- `PredictByMap` (sparse feature map), `PredictLeafIndex`, `PredictContrib`
  (SHAP feature attributions via `TreeSHAP`, `tree.h:457`).
- **`virtual void AddPredictionToScore(const Dataset* data, data_size_t num_data,
  double* score) const`** (`tree.h:104`, `tree.cpp:153-...`) — the batch training-time
  predictor: adds this tree's leaf output for every row to the running `score` array. It uses
  `BinIterator`s over the dataset's bins and the *inner* decision path
  (`NumericalDecisionInner` on bin values, `tree.h:357-372`) rather than raw feature values,
  which is faster and matches how the tree was learned. A stump (`num_leaves_<=1`) just adds
  `leaf_value_[0]` to every score (`tree.cpp:154-161`).

#### Serialization

- `std::string ToString() const` (`tree.cpp:339-...`) — the text model format: writes
  `num_leaves=`, `split_feature=`, `threshold=`, `decision_type=`, `left_child=`,
  `right_child=`, `leaf_value=`, `leaf_count=`, `internal_value=`, category tables if any,
  and linear-model fields. This is exactly the block you see in a saved LightGBM model file.
- `Tree(const char* str, size_t* used_len)` (`tree.h:41`, `tree.cpp:685-...`) — the inverse:
  reconstruct a `Tree` from that text block (and advance `used_len`).
- `ToJSON()`, `ToIfElse()` (C++ codegen), `LinearModelToJSON()`.

**Depends on:** `dataset.h` (for `AddPredictionToScore`/`BinIterator`), `meta.h`.
**Used by:** boosting (`GBDT` owns `std::vector<std::unique_ptr<Tree>>`), the tree learners
(construct and grow), the predictor, and model I/O.

---

### 13. `include/LightGBM/arrow.h` — Apache Arrow zero-copy ingestion

**Purpose.** Lets LightGBM read columns directly from Apache Arrow arrays (used by the Python
`pyarrow`/pandas path) without copying. It vendors the Arrow **C Data Interface** structs
`ArrowSchema` and `ArrowArray` (`arrow.h:34-65`) and wraps a list of chunks in
`class ArrowChunkedArray` (`arrow.h:80-...`), which precomputes `chunk_offsets_` so a global
row index can be located in the right chunk. There is a further `ArrowTable` (multiple
columns) later in the header. Templated iterators (elsewhere in the file) yield typed values.

Consumed by `Metadata::SetLabel/SetWeights/SetQuery/SetInitScore(const ArrowChunkedArray&)`
(`dataset.h:114-129`) and `Dataset::SetFieldFromArrow` (`dataset.h:675`). It is purely an
input adapter — once values are read they flow through the same `BinMapper`/`Bin` pipeline.

**Depends on:** nothing internal (self-contained C interface + STL). **Used by:** `Metadata`,
`Dataset`, and the C API's Arrow entry points.

---

### 14. End-to-end data flow (tying it together)

1. **Read.** `DatasetLoader::LoadFromFile` reads the text file; `Parser::ParseOneLine`
   converts each line to `(column,value)` pairs + label (`parser.hpp`).
2. **Sample & fit bins.** A sample of rows is collected; `ConstructBinMappersFromTextData`
   calls `BinMapper::FindBin` per feature (`bin.cpp`) to learn `bin_upper_bound_` /
   categorical maps. Trivial features are flagged.
3. **Construct structure.** `Dataset::Construct` drops trivial features, groups the rest
   (Exclusive Feature Bundling), and builds one `FeatureGroup` per group, each allocating
   `DenseBin`/`SparseBin`/`MultiValBin` storage (`dataset.cpp`, `feature_group.h`, `bin.h`).
4. **Push binned values.** Every row is re-parsed and `Dataset::PushOneRow` →
   `FeatureGroup::PushData` → `BinMapper::ValueToBin` → `Bin::Push`. Most-frequent-bin values
   are skipped (implicit default). `Metadata` receives labels/weights/queries.
5. **Freeze.** `Dataset::FinishLoad` finalizes every `Bin` and `Metadata`. The dataset is now
   **immutable**.
6. **Train (per tree, per node).** The objective produces `gradients`/`hessians`
   (`score_t*`). The tree learner calls `Dataset::ConstructHistograms`, which gathers ordered
   gradients and calls `Bin::ConstructHistogram` to fill the `hist_t*` (grad,hess) arrays;
   `Dataset::FixHistogram` repairs the skipped most-freq bin. The learner scans histograms to
   pick the best (feature, bin threshold, gain), then `Dataset::Split` →
   `FeatureGroup::Split` → `Bin::Split` partitions the row-index array into left/right, and
   `Tree::Split` records the node and sets leaf outputs. The subtraction trick reuses the
   parent histogram for the larger child.
7. **Predict / persist.** `Tree::AddPredictionToScore` accumulates leaf outputs during
   training; `Tree::Predict` scores new rows; `Tree::ToString`/`Tree(const char*,…)`
   serialize/deserialize the model.

**Fidelity hotspots for the Rust port:** `GreedyFindBin`/`FindBin` boundary arithmetic
(bit-exact binning), `BinMapper::ValueToBin` binary search + NaN/missing rules, the
histogram accumulation order and `FixHistogram` subtraction, the `Split` missing/default
routing, and the exact `Tree::ToString` field order/formatting.


---

## Section D — Boosting — the GBDT Ensemble Layer

This section documents the **boosting layer** of Microsoft's LightGBM C++ reference implementation — the code that runs the outer gradient-boosting loop, owns the ensemble of trees, computes gradients, updates scores, does bagging/GOSS/DART/RF, evaluates metrics, applies early stopping, and serializes the model. It is the orchestration layer that sits above the tree learner, objective function, and metric subsystems.

All file paths below are relative to `LightGBM/`.

---

### 0. Background: what "gradient boosting" is (read this first)

If you have never seen gradient-boosted decision trees (GBDT), here is the whole idea in a few sentences.

- We want a model `F(x)` that maps a feature row `x` to a prediction (a number). We build it as a **sum of many small decision trees**: `F(x) = tree_1(x) + tree_2(x) + ...`. This sum is called the **ensemble**. Each individual tree is a "weak sub-model."
- We build the trees **one at a time**, in a loop. Each pass of the loop is one **boosting iteration**.
- A **loss function** (a.k.a. *objective*) measures how wrong the current predictions are versus the true labels (e.g. squared error for regression, log-loss for classification).
- At each iteration we ask: "in which direction should each prediction move to reduce the loss?" The answer is given by calculus:
  - The **gradient** = first derivative of the loss with respect to the current prediction of a single row. It points in the direction of *increasing* loss, so we move opposite to it. Informally, for squared error the gradient is `prediction - label`, i.e. the current error. LightGBM's type for a gradient is `score_t` (a `float` by default).
  - The **hessian** = second derivative of the loss w.r.t. the current prediction. It measures the local curvature and is used to size the step and to weight the histogram bins. Also `score_t`.
- We fit a **new tree to the gradients and hessians** (not to the raw labels). The tree learns to predict the "correction" that reduces the loss. This is the key trick: boosting reduces a hard learning problem to a sequence of easy ones.
- We multiply the new tree's outputs by a small **shrinkage rate** (a.k.a. *learning rate*, e.g. 0.1) so each tree only takes a small step — this regularizes and improves generalization.
- We add the shrunk tree's predictions into a running **score** for every training row (the job of the **ScoreUpdater**), then repeat.
- **Bagging** = train each tree on a random *subset of rows* (bootstrap-style sampling) for speed and regularization.
- **GOSS** (Gradient-based One-Side Sampling) = a smarter subsampling: keep all rows with large gradients (the "hard" examples) and randomly sample from the small-gradient rows, re-weighting them.
- **DART** = "Dropouts meet Multiple Additive Regression Trees": at each iteration, temporarily *drop* (mute) a random set of already-built trees before fitting the next one, to prevent later trees from dominating.
- **RF** = Random Forest: not boosting at all — build independent trees on bootstrap samples and *average* them. LightGBM reuses the GBDT machinery to also offer RF.
- **Early stopping** = stop the loop when a validation metric hasn't improved for N consecutive iterations, and discard those last N (non-improving) trees.

With that vocabulary, the rest of this section is readable.

---

### 1. `include/LightGBM/boosting.h` — the `Boosting` abstract interface + factory

**Purpose.** Declares the abstract base class `Boosting` (`boosting.h:27`) that every ensemble strategy (GBDT, DART, RF, GOSS) implements. It is the single porting seam and the surface that the application layer and the C API talk to. Everything is pure-virtual; the header pulls in almost no implementation. It also declares `GBDTBase` (`boosting.h:323`), a thin extension that adds per-leaf value get/set.

**Key methods (all `virtual ... = 0` unless noted), grouped by purpose:**

*Lifecycle / setup:*
- `virtual void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function, const std::vector<const Metric*>& training_metrics)` (`boosting.h:39`) — wire up config, training data, the loss function, and the training metrics.
- `virtual void ResetTrainingData(...)` (`:57`), `virtual void ResetConfig(const Config*)` (`:60`) — swap data/config for continued training.
- `virtual void AddValidDataset(const Dataset* valid_data, const std::vector<const Metric*>& valid_metrics)` (`:69`) — register a validation set + its metrics (used for early stopping and reporting).

*Training:*
- `virtual void Train(int snapshot_freq, const std::string& model_output_path)` (`:72`) — run the full loop of iterations.
- `virtual bool TrainOneIter(const score_t* gradients, const score_t* hessians)` (`:85`) — do exactly one boosting iteration. If `gradients`/`hessians` are `nullptr` the built-in objective computes them; otherwise a *custom* (user-supplied) objective is used. Returns `true` when training can no longer continue.
- `virtual void RollbackOneIter()` (`:90`), `virtual void RefitTree(const int* tree_leaf_prediction, size_t nrow, size_t ncol)` (`:77`), `virtual int GetCurrentIteration() const` (`:95`).

*Evaluation / score access:*
- `virtual std::vector<double> GetEvalAt(int data_idx) const` (`:102`) — metric values for dataset `data_idx` (0 = train, 1 = first validation set, ...).
- `virtual const double* GetTrainingScore(int64_t* out_len)` (`:109`) — raw running scores for the training set.
- `virtual void GetPredictAt(int data_idx, double* result, int64_t* out_len)` (`:124`) — objective-converted predictions.

*Prediction of one record* (all take `const double* features` and write into `double* output`):
- `virtual void PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const` (`:134`) — raw sum-of-trees, no link function.
- `virtual void Predict(...)` (`:147`) — same but applies the objective's output transform (e.g. sigmoid).
- `virtual void PredictLeafIndex(...)` (`:159`), `virtual void PredictContrib(...)` (`:170`) — leaf indices and SHAP-style per-feature contributions. `*ByMap` variants (`:137,150,162,172`) take a sparse `std::unordered_map<int,double>` feature vector instead of a dense array.

*Model I/O:*
- `virtual std::string DumpModel(int start_iteration, int num_iteration, int feature_importance_type) const` (`:182`) — JSON.
- `virtual std::string SaveModelToString(...) const` (`:216`) / `virtual bool SaveModelToFile(...) const` (`:207`) / `virtual bool LoadModelFromString(const char* buffer, size_t len)` (`:224`) — the text model format.
- `virtual std::string ModelToIfElse(int num_iteration) const` (`:189`) / `virtual bool SaveModelToIfElse(...) const` (`:197`) — emit C++ if-else code for the model.
- `static bool LoadFileToBoosting(Boosting* boosting, const char* filename)` (`:304`).

*Introspection:* `FeatureImportance` (`:232`), `MaxFeatureIdx` (`:250`), `FeatureNames` (`:256`), `NumberOfTotalModel` (`:268`), `NumModelPerIteration` (`:274`), `NumberOfClasses` (`:280`), `GetUpperBoundValue`/`GetLowerBoundValue` (`:238,244`).

**The factory** (`boosting.h:314`):
```cpp
static Boosting* CreateBoosting(const std::string& type, const char* filename);
```
This is the stringly-typed factory that maps a `type` string ("gbdt"/"dart"/"goss"/"rf") to a concrete subclass. Returns a raw owning pointer the caller wraps in a `unique_ptr`.

**`GBDTBase`** (`boosting.h:323`) adds:
- `virtual double GetLeafValue(int tree_idx, int leaf_idx) const` and `virtual void SetLeafValue(int tree_idx, int leaf_idx, double val)` — needed by refit / leaf editing.

**Dependencies:** forward-declares `Dataset`, `ObjectiveFunction`, `Metric`, `PredictionEarlyStopInstance`. **Consumers:** `src/application/application.cpp` (CLI) and `src/c_api.cpp` (bindings) hold a `Boosting*` and drive it.

---

### 2. `src/boosting/boosting.cpp` — factory implementation

**Purpose.** Implements the two static functions declared in the header, plus a small helper.

- `std::string GetBoostingTypeFromModelFile(const char* filename)` (`boosting.cpp:13`) — reads the first line of a saved model file, which stores the sub-model type string (e.g. `"tree"`).
- `bool Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename)` (`:19`) — reads the whole file into a buffer via `TextReader<size_t>` and calls `boosting->LoadModelFromString(buffer, len)`.
- `Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename)` (`:34`) — the dispatch:
  - No filename → construct fresh: `"gbdt"` → `new GBDT()`, `"dart"` → `new DART()`, `"goss"` → `new GBDT()` (note: GOSS is *not* a separate Boosting subclass — it is a GBDT whose **sample strategy** is GOSS; see §8), `"rf"` → `new RF()`, else `nullptr` (`:36-46`).
  - With filename → verify the file's first line is `"tree"` (`:49`), construct the matching subclass, then `LoadFileToBoosting` to continue training from a saved model.

**Return type:** raw owning `Boosting*`. **Depends on:** `dart.hpp`, `gbdt.h`, `rf.hpp` (goss handled inside GBDT). **Called by:** application/C-API when creating a booster.

---

### 3. `src/boosting/gbdt.h` — the `GBDT` class declaration (the heart)

**Purpose.** Declares `class GBDT : public GBDTBase` (`gbdt.h:37`) — the concrete gradient-boosting implementation. DART and RF derive from `GBDT`. This header holds all the member state and inline methods; the heavy logic lives in the three `.cpp` files.

**Key inline/declared methods** (signatures echo the base class):
- `void Init(const Config*, const Dataset*, const ObjectiveFunction*, const std::vector<const Metric*>&) override` (`:57`).
- `bool TrainOneIter(const score_t* gradients, const score_t* hessians) override` (`:154`) — see §4 for the body.
- `void MergeFrom(const Boosting* other) override` (`:70`, inline) — prepend another GBDT's trees to this one's `models_` (used to concatenate models). It deep-copies each `Tree` via `new Tree(*tree)`.
- `void ShuffleModels(int start_iter, int end_iter) override` (`:89`, inline) — randomly permute whole iterations of trees (a fixed `Random(17)`), used by DART-style prediction randomization.
- `int GetCurrentIteration() const override` (`:164`) → `models_.size() / num_tree_per_iteration_`.
- `int NumPredictOneRow(...) const override` (`:281`, inline) — compute output length per row for the raw/leaf/contrib prediction modes.
- `void InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) override` (`:426`, inline) — set the `[start_iteration_for_pred_, num_iteration_for_pred_)` window used by all predict paths, and lazily `RecomputeMaxDepth()` on every tree (mutex-guarded, once) when contributions are requested.
- `double GetLeafValue(int, int) const` / `void SetLeafValue(int, int, double)` (`:451,457`) — delegate to the underlying `Tree`.
- Protected virtuals overridden by DART/RF: `virtual void Boosting()` (`:493`), `virtual void UpdateScore(const Tree*, int)` (`:500`), `virtual bool EvalAndCheckEarlyStopping()` (`:483`), `double BoostFromAverage(int class_id, bool update_scorer)` (`:515`), `void ResetGradientBuffers()` (`:520`), `bool GetIsConstHessian(const ObjectiveFunction*)` (`:473`).

**Key member state** (`gbdt.h:522-620`) — this is the mutable training state referenced throughout:
- `int iter_` — current iteration counter.
- `const Dataset* train_data_` — read-only training data (immutable after `FinishLoad`).
- `std::unique_ptr<Config> config_` — owned copy of the config.
- `std::unique_ptr<TreeLearner> tree_learner_` — grows one tree per call (§Dependencies).
- `const ObjectiveFunction* objective_function_` — computes gradients/hessians (non-owning; `nullptr` when a custom objective is used from Python/R).
- `std::unique_ptr<ScoreUpdater> train_score_updater_` and `std::vector<std::unique_ptr<ScoreUpdater>> valid_score_updater_` — running scores for train and each validation set (§7).
- `std::vector<const Metric*> training_metrics_` and `std::vector<std::vector<const Metric*>> valid_metrics_` — metrics per dataset.
- Early-stopping state: `int early_stopping_round_`, `double early_stopping_min_delta_`, `bool es_first_metric_only_`, and `best_iter_ / best_score_ / best_msg_` (nested vectors indexed by [validation set][metric]).
- `std::vector<std::unique_ptr<Tree>> models_` — **the ensemble itself.** Trees are stored flat; iteration `i`, class `k` lives at index `i * num_tree_per_iteration_ + k`.
- Gradient buffers: `std::vector<score_t, AlignmentAllocator<...>> gradients_`, `hessians_`, and raw pointers `score_t* gradients_pointer_ / hessians_pointer_` (may point to CPU or GPU memory; GPU branches are out of scope here).
- `int num_tree_per_iteration_` — 1 for regression/binary, `num_class` for multiclass (one tree per class each iteration). `int num_class_`, `data_size_t label_idx_`, `int max_feature_idx_`.
- `int num_iteration_for_pred_`, `int start_iteration_for_pred_` — the prediction window.
- `double shrinkage_rate_` — the learning rate applied to each new tree.
- `int num_init_iteration_` — how many iterations were loaded from a pre-existing model (continued training).
- `bool average_output_` — `true` for RF (average instead of sum). `bool is_constant_hessian_`, `bool linear_tree_`.
- `std::unique_ptr<SampleStrategy> data_sample_strategy_` — the bagging/GOSS engine (§8).
- `Json forced_splits_json_` — optional user-forced split structure.

`GetLoadedParam()` (`:169`, inline) reconstructs a JSON object from the stored `loaded_parameter_` string using `Config::ParameterTypes()` to type each value.

**Depends on:** `objective_function.h`, `tree_learner.h` (via includes), `sample_strategy.h`, `score_updater.hpp`, `Tree`, `Config`.

---

### 4. `src/boosting/gbdt.cpp` — GBDT training / boosting logic

**Purpose.** Implements the outer boosting loop, gradient computation, score updates, metric evaluation, early stopping, refit, and the various reset paths. This is the file that "runs" boosting.

Module-level globals (`gbdt.cpp:22-25`): `Common::Timer global_timer` (used everywhere for `FunctionTimer` profiling), and the process-global device selectors `LGBM_config_::current_device` / `current_learner`.

**`GBDT::Init(...)`** (`gbdt.cpp:53`):
- Copies the config, reads `num_class_`, `learning_rate` → `shrinkage_rate_`, early-stopping params.
- Loads a forced-splits JSON file if configured (`:84`).
- Sets `num_tree_per_iteration_` = `objective_function_->NumModelPerIteration()` (multiclass → `num_class`) (`:93-96`).
- Creates the sample strategy: `data_sample_strategy_.reset(SampleStrategy::CreateSampleStrategy(...))` (`:101`).
- Creates the tree learner: `tree_learner_ = TreeLearner::CreateTreeLearner(config_->tree_learner, config_->device_type, config_.get(), boosting_on_gpu_)` and `tree_learner_->Init(train_data_, is_constant_hessian_)` (`:107-111`).
- Creates `train_score_updater_` (`:126`), records feature names/infos/label index, validates forced-split features (`CheckForcedSplitFeatures`, `:164`), sizes gradient buffers via `ResetGradientBuffers()` (`:149`).
- Fills `class_need_train_` — for multiclass with some empty classes, skip training those (`:151-157`).

**`GBDT::Boosting()`** (`gbdt.cpp:220`) — computes gradients & hessians for the *current* scores by calling the objective:
```cpp
objective_function_->GetGradients(GetTrainingScore(&num_score), gradients_pointer_, hessians_pointer_);
```
(There is a `bagging_by_query` variant at `:227-231` that first samples queries then computes gradients only for sampled queries — used for ranking.) If `objective_function_` is `nullptr` it fatals ("No objective function provided").

**`GBDT::Train(int snapshot_freq, const std::string& model_output_path)`** (`gbdt.cpp:237`) — the **full outer loop**:
```cpp
for (int iter = 0; iter < config_->num_iterations && !is_finished; ++iter) {
  is_finished = TrainOneIter(nullptr, nullptr);
  if (!is_finished) is_finished = EvalAndCheckEarlyStopping();
  ... log seconds/iteration ...
  if (snapshot_freq > 0 && (iter+1) % snapshot_freq == 0) SaveModelToFile(...);
}
```
So `Train` just calls `TrainOneIter` up to `num_iterations` times, checks early stopping each round, and periodically snapshots the model.

**`GBDT::TrainOneIter(const score_t* gradients, const score_t* hessians)`** (`gbdt.cpp:344`) — one boosting iteration, the core method. Step by step:
1. If `gradients==nullptr` (the normal, built-in-objective case): for each class/tree call `BoostFromAverage(cur_tree_id, true)` to get an initial bias score (`:349-351`), then call `Boosting()` to fill `gradients_pointer_`/`hessians_pointer_` from the objective (`:352`), and point `gradients`/`hessians` at those buffers.
   Else (custom objective supplied by the caller): assert `objective_function_==nullptr`; if the strategy changes hessians (GOSS) copy the caller's grads/hess into the internal buffers (`:359-371`).
2. **Bagging:** `data_sample_strategy_->Bagging(iter_, tree_learner_.get(), gradients_.data(), hessians_.data())` (`:376`). Read back `is_use_subset`, `bag_data_cnt`, and `bag_data_indices` (the in-bag row indices).
3. For each `cur_tree_id` in `[0, num_tree_per_iteration_)` (`:387`):
   - Allocate a fresh `Tree`.
   - If this class needs training and there are features: slice grads/hess at `offset = cur_tree_id * num_data_`. If using a subset, gather the in-bag grads/hess into contiguous positions (`:394-401`). Then **grow the tree**: `new_tree.reset(tree_learner_->Train(grad, hess, is_first_tree))` (`:403`).
   - If the tree actually split (`num_leaves() > 1`) (`:406`):
     - `should_continue = true`.
     - `tree_learner_->RenewTreeOutput(...)` — optionally recompute leaf outputs for objectives like L1 that need the median of residuals (uses a `residual_getter` lambda = `label[i] - score[i]`) (`:409-411`).
     - `new_tree->Shrinkage(shrinkage_rate_)` — apply the learning rate (`:413`).
     - `UpdateScore(new_tree.get(), cur_tree_id)` — add the tree's predictions into the running scores (`:415`).
     - If there was a non-zero init bias, `new_tree->AddBias(init_scores[cur_tree_id])` (`:416-418`).
   - Else (no useful split): emit a constant tree carrying the init score (first iteration only) or zero (`:419-435`).
   - `models_.push_back(std::move(new_tree))` (`:437`).
4. If no class produced a real split (`!should_continue`), warn, pop the just-added trees, and **return `true`** (training is finished) (`:440-448`).
5. Otherwise `++iter_` and return `false` (`:450`).

**`GBDT::UpdateScore(const Tree* tree, int cur_tree_id)`** (`gbdt.cpp:491`):
- If not using a subset: `train_score_updater_->AddScore(tree_learner_.get(), tree, cur_tree_id)` (fast path using the tree learner's known data partition), then separately score the **out-of-bag** rows via `AddScore(tree, indices+bag_data_cnt, num_data_-bag_data_cnt, ...)` (`:494-509`).
- If using a subset: `train_score_updater_->AddScore(tree, cur_tree_id)` over all data (`:512`).
- Always update every validation `score_updater` (`:517-519`).

**`GBDT::BoostFromAverage(int class_id, bool update_scorer)`** (`gbdt.cpp:319`) — computes the model's global starting score so boosting begins from a sensible constant (e.g. the mean label for regression, `log(p/(1-p))` for binary). Only fires for the empty model when `boost_from_average` is set (or there are zero features). It calls `ObtainAutomaticInitialScore` (`:308`) → `objective_function_->BoostFromScore(class_id)` (synced across machines via `Network::GlobalSyncUpByMean` in distributed mode), and if non-negligible adds it to the train + valid score updaters. Returns the init score (used later as tree bias).

**`GBDT::EvalAndCheckEarlyStopping()`** (`gbdt.cpp:472`) — calls `OutputMetric(iter_)`; if that returns a non-empty "best message," early stopping fired, so it logs the best iteration, pops the last `early_stopping_round_ * num_tree_per_iteration_` trees, and returns `true`.

**`GBDT::OutputMetric(int iter)`** (`gbdt.cpp:551`) — evaluates training and validation metrics (via `EvalOneMetric`), logs them, and (when `early_stopping_round_ > 0`) tracks per-metric best score/iteration. A validation metric "improves" only if `factor_to_bigger_better() * score - best > early_stopping_min_delta_` (`:594`); if no improvement for `early_stopping_round_` iterations it returns the stored best message. `es_first_metric_only_` restricts the check to the first metric.

**`GBDT::EvalOneMetric(const Metric* metric, const double* score, data_size_t num_data)`** (`gbdt.cpp:523`) → `metric->Eval(score, objective_function_)` (CPU path; CUDA copy-shuffling is out of scope).

**`GBDT::GetTrainingScore(int64_t* out_len)`** (`gbdt.cpp:635`) → returns `train_score_updater_->score()` and sets `*out_len = num_data * num_class_`.

**`GBDT::GetPredictAt(int data_idx, double* out_result, int64_t* out_len)`** (`gbdt.cpp:665`) — takes the raw running scores for train (idx 0) or a validation set, and, if an objective exists, runs `objective_function_->ConvertOutput(tree_pred, tmp_result)` per row to produce the final (link-transformed) predictions. Note multiclass scores are laid out class-major: `raw_scores[j * num_data + i]`.

**`GBDT::RefitTree(const int* tree_leaf_prediction, size_t nrow, size_t ncol)`** (`gbdt.cpp:258`) — given, for every row, which leaf of every existing tree it falls into, recompute leaf outputs against fresh gradients (used for `refit` task / model updating on new data). Calls `tree_learner_->FitByExistingTree(...)` per tree.

**`GBDT::RollbackOneIter()`** (`gbdt.cpp:454`) — undoes the last iteration: subtract each last tree's predictions from the scores (`Shrinkage(-1.0)` then `AddScore`), pop the trees, `--iter_`.

**Reset paths:** `ResetTrainingData` (`:726`), `ResetConfig` (`:795`), `ResetGradientBuffers` (`:841`) resize/rebuild buffers and score updaters when data or config changes for continued training; `CheckAlign` ensures new data has identical bin mappers.

**Dependencies wired here:** `ObjectiveFunction::GetGradients`/`BoostFromScore`/`ConvertOutput`, `TreeLearner::Train`/`RenewTreeOutput`/`AddPredictionToScore`/`FitByExistingTree`, `Metric::Eval`, `Network` (distributed sync), `SampleStrategy::Bagging`.

**Types recap:** gradients/hessians are `const score_t*` (`float*`); scores are `double*`; row counts are `data_size_t` (`int32_t`); the ensemble is `std::vector<std::unique_ptr<Tree>>`.

---

### 5. `src/boosting/gbdt_prediction.cpp` — single-record prediction

**Purpose.** Implements the prediction methods that turn the stored ensemble into outputs for one feature row. Used at inference time (after training or after loading a model).

- `void GBDT::PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const` (`gbdt_prediction.cpp:13`):
  - Zeroes `output` (length `num_tree_per_iteration_`).
  - Loops over iterations `[start_iteration_for_pred_, start+num_iteration_for_pred_)`, and within each iteration over the `num_tree_per_iteration_` classes, accumulating `output[k] += models_[i*ntpi + k]->Predict(features)` (`:18-22`).
  - **Prediction early stopping:** every `early_stop->round_period` iterations it calls `early_stop->callback_function(output, ...)`; if it returns true (predictions already confident), it stops early (`:24-30`). This is a speed optimization distinct from *training* early stopping.
- `void GBDT::PredictRawByMap(...)` (`:34`) — same but for sparse feature maps (`Tree::PredictByMap`).
- `void GBDT::Predict(const double* features, double* output, ...)` (`:55`) — calls `PredictRaw`, then if `average_output_` (RF) divides by `num_iteration_for_pred_` (`:57-61`), then applies `objective_function_->ConvertOutput(output, output)` to get the final transformed prediction (`:62-64`).
- `void GBDT::PredictByMap(...)` (`:67`) — sparse variant of `Predict`.
- `void GBDT::PredictLeafIndex(const double* features, double* output) const` (`:79`) — for each tree in the prediction window, writes `models_ptr[i]->PredictLeafIndex(features)` (which leaf the row lands in). `PredictLeafIndexByMap` (`:88`) is the sparse variant.

(`PredictContrib`/`PredictContribByMap` live in `gbdt.cpp:640,653` — they zero an output of size `num_tree_per_iteration_ * (num_features+1)` and accumulate per-tree feature contributions.)

**Types:** dense features `const double*`, sparse `const std::unordered_map<int,double>&`, output `double*`.

---

### 6. `src/boosting/gbdt_model_text.cpp` — model serialization

**Purpose.** Implements dumping/loading the model in the human-readable text format and the JSON format, plus feature-importance computation and if-else C++ codegen. `kModelVersion = "v4"` (`gbdt_model_text.cpp:19`).

- `std::string GBDT::DumpModel(int start_iteration, int num_iteration, int feature_importance_type) const` (`:21`) — builds a **JSON** string: header (`name`, `version`, `num_class`, `num_tree_per_iteration`, `label_index`, `max_feature_idx`, `objective`, `average_output`, `feature_names`, `monotone_constraints`, `feature_infos`), then a `tree_info` array where each entry is `models_[i]->ToJSON()`, then a `feature_importances` object.
- `std::string GBDT::SaveModelToString(int start_iteration, int num_iteration, int feature_importance_type) const` (`:311`) — the **canonical text format** LightGBM reads back:
  - Header lines: sub-model name (`tree`), `version=v4`, `num_class=`, `num_tree_per_iteration=`, `label_index=`, `max_feature_idx=`, `objective=`, optional `average_output`, `feature_names=`, optional `monotone_constraints=`, `feature_infos=`.
  - Each tree serialized in parallel (`#pragma omp parallel for`) as `"Tree=<idx>\n" + models_[i]->ToString()` into `tree_strs`, with a `tree_sizes=` line giving byte lengths for fast parallel re-loading (`:354-372`).
  - Then `feature_importances:` (sorted desc), a `parameters:` block (`config_->ToString()`), and an optional `parser:` block.
- `bool GBDT::SaveModelToFile(int, int, int, const char* filename) const` (`:410`) — writes the string via `VirtualFileWriter`.
- `bool GBDT::LoadModelFromString(const char* buffer, size_t len)` (`:421`) — the inverse:
  - Clears `models_`, parses the header key/value lines into a map (`:428-454`).
  - Reads `num_class`, `num_tree_per_iteration` (defaults to num_class), `label_index`, `max_feature_idx`, `average_output`, `feature_names`, `monotone_constraints`, `feature_infos` — fataling on missing required fields (`:456-521`).
  - If an `objective` line exists, reconstructs a `loaded_objective_` via `ObjectiveFunction::CreateObjectiveFunction(ParseObjectiveAlias(str))` and points `objective_function_` at it (`:523-527`) — this is why a loaded model can still transform its outputs.
  - Reconstructs trees: if `tree_sizes` present, it uses the byte boundaries to build all trees **in parallel** (`new Tree(cur_p, &used_len)`) (`:546-572`); otherwise sequentially (`:529-545`).
  - Sets `num_iteration_for_pred_ = num_init_iteration_ = models_.size()/num_tree_per_iteration_`, `iter_ = 0`, then parses the `parameters:` block (also detecting `linear_tree`) and the `parser:` block (`:573-621`).
- `std::string GBDT::ModelToIfElse(int num_iteration) const` (`:124`) / `bool GBDT::SaveModelToIfElse(...)` (`:286`) — emit compilable C++ (`PredictTree<i>` functions and dispatch tables) so a model can be baked into native code.
- `std::vector<double> GBDT::FeatureImportance(int num_iteration, int importance_type) const` (`:627`) — importance_type 0 = **split count** (increment per split with positive gain), 1 = **total gain** (sum of `split_gain`), accumulated across all trees and features.

**Types:** returns `std::string`; consumes `const char* buffer, size_t len`; trees round-trip through `Tree::ToString()` / `Tree(const char*, size_t*)`.

---

### 7. `src/boosting/score_updater.hpp` — the running-score accumulator

**Purpose.** `class ScoreUpdater` (`score_updater.hpp:21`) stores and updates the **running prediction score** for every row of a dataset. There is one for the training set and one per validation set. "Score" here = the raw sum-of-trees value before any objective transform. It is the boosting loop's accumulator: each new tree's contribution is *added* here.

**Construction** (`:27`):
```cpp
ScoreUpdater(const Dataset* data, int num_tree_per_iteration);
```
Allocates `score_` of size `num_data * num_tree_per_iteration` (aligned `std::vector<double>`), zero-initialized. If the dataset carries an `init_score` (metadata), it copies that in and sets `has_init_score_ = true` (`:34-46`).

**Key methods** (all `virtual inline`; the CUDA subclass overrides them):
- `void AddScore(double val, int cur_tree_id)` (`:54`) — add a constant to all rows of one class (used for init/bias scores). OpenMP-parallel over rows.
- `void MultiplyScore(double val, int cur_tree_id)` (`:63`) — scale all rows of one class (used by RF averaging and rollback).
- `void AddScore(const Tree* tree, int cur_tree_id)` (`:76`) — `tree->AddPredictionToScore(data_, num_data_, score_+offset)`; re-predicts the tree over the whole dataset. Used for validation data (and training when using a subset).
- `void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id)` (`:88`) — `tree_learner->AddPredictionToScore(tree, score_+offset)`; the fast path that reuses the tree learner's existing data partition (which leaf each *training* row is in), avoiding re-traversal.
- `void AddScore(const Tree* tree, const data_size_t* data_indices, data_size_t data_cnt, int cur_tree_id)` (`:101`) — add predictions for a subset of rows (the out-of-bag rows).
- `const double* score() const` (`:108`) and `data_size_t num_data() const` (`:110`).

`offset = num_data_ * cur_tree_id` everywhere: scores are stored **class-major** (all rows of class 0, then all rows of class 1, ...).

**Depends on:** `Dataset`, `Tree`, `TreeLearner`. **Used by:** `GBDT::UpdateScore`, `BoostFromAverage`, `RollbackOneIter`, DART/RF, and metric evaluation (which reads `score()`).

---

### 8. Sample strategies: bagging vs GOSS

Sampling (which rows each tree sees) is factored out of GBDT into a `SampleStrategy` hierarchy so that "bagging" and "GOSS" are pluggable.

#### 8a. `include/LightGBM/sample_strategy.h` — `SampleStrategy` base

**Purpose.** Abstract base (`sample_strategy.h:23`) for row-subsampling engines.

- `static SampleStrategy* CreateSampleStrategy(const Config*, const Dataset*, const ObjectiveFunction*, int num_tree_per_iteration)` (`:29`) — factory.
- `virtual void Bagging(int iter, TreeLearner* tree_learner, score_t* gradients, score_t* hessians) = 0` (`:31`) — produce the in-bag row set for this iteration and hand it to the tree learner (`tree_learner->SetBaggingData(...)`). GOSS may also *modify* the passed gradients/hessians (re-weighting).
- `virtual void ResetSampleConfig(const Config*, bool is_change_dataset) = 0` (`:33`).
- `virtual bool IsHessianChange() const = 0` (`:54`) — does this strategy rewrite gradients/hessians? (GOSS = true, bagging = false). GBDT uses this to decide GPU eligibility and buffer copying.
- Accessors: `is_use_subset()`, `bag_data_cnt()`, `bag_data_indices()` (an aligned `std::vector<data_size_t>` of in-bag row indices), `NeedResizeGradients()`, plus ranking helpers `num_sampled_queries()` / `sampled_query_indices()`.

**Members** include `ParallelPartitionRunner<data_size_t,false> bagging_runner_` (parallel partition of rows into in-bag/out-of-bag), `std::vector<Random> bagging_rands_` (block-wise RNGs for reproducible sampling), `std::unique_ptr<Dataset> tmp_subset_` (a compacted copy of just the in-bag rows, when `is_use_subset_`).

#### 8b. `src/boosting/sample_strategy.cpp` — the factory

`SampleStrategy::CreateSampleStrategy` (`sample_strategy.cpp:12`): if `config->data_sample_strategy == "goss"` → `new GOSSStrategy(...)`, else → `new BaggingSampleStrategy(...)`. (So `boosting=goss` and `data_sample_strategy=goss` both route here; the former is a GBDT + GOSS strategy.)

#### 8c. `src/boosting/bagging.hpp` — `BaggingSampleStrategy`

**Purpose.** Classic random row subsampling. `class BaggingSampleStrategy : public SampleStrategy` (`bagging.hpp:14`).

- `void Bagging(int iter, TreeLearner* tree_learner, ...)` (`:30`): only re-bags when `bag_data_cnt_ < num_data_ && iter % bagging_freq == 0` (or a re-bag was forced). It runs `bagging_runner_.Run<true>(...)` with a per-block helper to partition rows into in-bag (front) / out-of-bag (back) of `bag_data_indices_`, giving `bag_data_cnt_` in-bag rows. Then either passes indices to the tree learner (`SetBaggingData(nullptr, indices, cnt)`) or, if `is_use_subset_`, materializes a compacted `tmp_subset_` dataset and passes that (`:107-135`). A `bagging_by_query` branch (`:52-104`) samples whole *queries* (for ranking) and rebuilds `sampled_query_boundaries_`.
- `BaggingHelper(start, cnt, buffer)` (`:230`): for each row, keep it with probability `bagging_fraction` using the block RNG.
- `BalancedBaggingHelper(...)` (`:248`): separate keep-probabilities for positive vs negative labels (`pos_bagging_fraction` / `neg_bagging_fraction`).
- `ResetSampleConfig(...)` (`:139`): decides `bag_data_cnt_` from the fractions, whether to use the compacted subset (`average_bag_rate <= 0.5` and few feature groups → `is_use_subset_ = true`), seeds the RNGs, and flags `need_re_bagging_`.
- `IsHessianChange()` → `false` (`:217`).

#### 8d. `src/boosting/goss.hpp` — `GOSSStrategy`

**Purpose.** Gradient-based One-Side Sampling. `class GOSSStrategy : public SampleStrategy` (`goss.hpp:18`). Idea: rows with large `|grad*hess|` matter most, so **keep the top `top_rate` fraction** and **randomly sample `other_rate` fraction** of the rest, then **up-weight the sampled small-gradient rows** so the loss estimate stays unbiased.

- `void Bagging(int iter, TreeLearner*, score_t* gradients, score_t* hessians)` (`:30`): skips sampling for the first `1/learning_rate` iterations (`:33`), then partitions via `Helper(...)`.
- `Helper(start, cnt, buffer, gradients, hessians)` (`:116`) — the actual algorithm:
  - Compute per-row importance `sum_k |grad*hess|` (`:120-126`).
  - `top_k = cnt*top_rate`, `other_k = cnt*other_rate`; find the top-k threshold via `ArrayArgs::ArgMaxAtK` (`:127-131`).
  - Rows `>= threshold` are always kept; among the rest, keep each with probability `other_k/rest_all` and **multiply its grad & hess by `(cnt-top_k)/other_k`** to compensate (`:144-161`).
- `ResetSampleConfig(...)` (`:76`) — validates `top_rate + other_rate <= 1` and both `> 0`, forbids combining with bagging, sets up subset if the kept fraction `<= 0.5`.
- `IsHessianChange()` → `true` (`:111`) — GOSS rewrites gradients/hessians, so GBDT copies them into internal buffers and disables GPU boosting.

**Note:** GOSS is *not* a `Boosting` subclass; a `boosting=goss` model is a `GBDT` whose `data_sample_strategy_` is a `GOSSStrategy`.

---

### 9. `src/boosting/dart.hpp` — DART strategy

**Purpose.** `class DART : public GBDT` (`dart.hpp:23`). DART ("Dropouts meet Multiple Additive Regression Trees") reduces over-specialization of later trees by temporarily dropping a random subset of existing trees before each iteration and re-normalizing weights afterward.

- `Init` / `ResetConfig` (`:41,49`) seed `random_for_drop_ = Random(drop_seed)` and reset `sum_weight_`.
- `bool TrainOneIter(const score_t* gradient, const score_t* hessian) override` (`:58`): sets `is_update_score_cur_iter_ = false`, calls `GBDT::TrainOneIter` (which internally calls the overridden `GetTrainingScore`), then `Normalize()` and appends the new tree's weight (`shrinkage_rate_`) to `tree_weight_`.
- `const double* GetTrainingScore(int64_t* out_len) override` (`:78`): the first time per iteration it calls `DroppingTrees()` (drops happen exactly once per iteration, right before gradients are computed).
- `DroppingTrees()` (`:97`): with probability `skip_drop` skip entirely; else select drop indices by `drop_rate` (weighted by per-tree weight unless `uniform_drop`), capped at `max_drop`. For each dropped tree it *subtracts* its contribution from the training score (`Shrinkage(-1.0)` then `AddScore`). Then it recomputes `shrinkage_rate_` for the new tree based on how many were dropped (`k`): `learning_rate/(k+1)` (or the xgboost-mode formula) (`:138-146`).
- `Normalize()` (`:158`): rescales the dropped trees so the total contribution is `k/(k+1)` of the original (the three-step shrink sequence documented at `:150-156`), updating both validation and training scores and the `tree_weight_`/`sum_weight_` bookkeeping.
- `EvalAndCheckEarlyStopping() override` (`:88`) — DART just outputs metrics and **never early-stops** (returns `false`), because the score changes each iteration due to dropping.

State: `tree_weight_`, `sum_weight_`, `drop_index_`, `random_for_drop_`, `is_update_score_cur_iter_`.

---

### 10. `src/boosting/rf.hpp` — Random Forest strategy

**Purpose.** `class RF : public GBDT` (`rf.hpp:25`). Reuses GBDT's tree-growing plumbing but builds **independent, averaged** trees instead of a boosted sequence. Sets `average_output_ = true` in the constructor (`:28`), so prediction divides by the number of iterations.

- `Init` (`:33`): requires either bagging or feature subsampling (RF needs randomness), or GOSS. Sets `shrinkage_rate_ = 1.0` (no shrinkage in RF), and calls `Boosting()` once up front (RF computes gradients from a *constant* score, not the accumulating ensemble score).
- `void Boosting() override` (`:90`): computes gradients/hessians against a constant `init_scores_` (the average label per class) rather than the current ensemble score — this is what makes the trees independent rather than additive.
- `bool TrainOneIter(const score_t*, const score_t*) override` (`:111`): does bagging, grows one tree per class with `tree_learner_->Train(grad, hess, false)`, renews leaf outputs, then maintains the **running average** of scores via the pattern `MultiplyScore(cur_tree_id, iter+init)` → `UpdateScore(...)` → `MultiplyScore(cur_tree_id, 1/(iter+init+1))` (`:157-159`). It requires a built-in objective (custom grads not supported — asserts `gradients==nullptr`). Always returns `false` (RF doesn't self-terminate).
- `RollbackOneIter` (`:184`) and `AddValidDataset` (`:212`) maintain the same averaging invariant when trees are removed or a validation set is added.
- `NeedAccuratePrediction()` → `true` (`:222`) — disables prediction-time early stopping.

State: `tmp_grad_`, `tmp_hess_` (subset buffers), `init_scores_`.

---

### 11. How it all fits together (the pipeline)

**Training path** (driven by `Application::Run` or the C API `LGBM_BoosterUpdateOneIter`):
1. `Boosting::CreateBoosting(type, filename)` builds a `GBDT`/`DART`/`RF`.
2. `Init(config, train_data, objective, training_metrics)` creates the `TreeLearner`, `SampleStrategy`, and `ScoreUpdater`, sizes gradient buffers.
3. `AddValidDataset(...)` for each validation set.
4. `Train(snapshot_freq, path)` loops `num_iterations` times calling `TrainOneIter`:
   - `BoostFromAverage` (first iter) → `Boosting()` calls `objective->GetGradients(scores, grad, hess)`.
   - `SampleStrategy::Bagging` picks in-bag rows (and, for GOSS, reweights grad/hess).
   - `TreeLearner::Train(grad, hess, is_first_tree)` grows one tree per class.
   - `RenewTreeOutput` → `Tree::Shrinkage(learning_rate)` → `UpdateScore` adds predictions to the `ScoreUpdater`.
   - `EvalAndCheckEarlyStopping` → `OutputMetric` → `Metric::Eval(scores, objective)`; stop and pop trees if a validation metric stalls for `early_stopping_round_`.
5. `SaveModelToFile` writes the text model.

**Prediction path** (after training or `LoadModelFromString`):
- `InitPredict(start_iteration, num_iteration, is_pred_contrib)` sets the tree window.
- `PredictRaw` sums the windowed trees; `Predict` additionally averages (RF) and applies `objective->ConvertOutput`. `PredictLeafIndex` / `PredictContrib` for leaf indices / SHAP contributions.

**Serialization:** `SaveModelToString` (text, `v4`) / `DumpModel` (JSON); `LoadModelFromString` reconstructs `models_`, the objective (for output transforms), and parameters. `ModelToIfElse` bakes the model to C++.

**Dependency summary:**
- **Boosting → ObjectiveFunction:** `GetGradients` (grad/hess), `BoostFromScore` (init score), `ConvertOutput` (link), `NumModelPerIteration`, `NeedAccuratePrediction`.
- **Boosting → TreeLearner:** `Train` (grow tree), `RenewTreeOutput`, `AddPredictionToScore`, `SetBaggingData`, `FitByExistingTree`.
- **Boosting → Metric:** `Eval` for train/valid metrics and early stopping.
- **Boosting → SampleStrategy → TreeLearner:** `Bagging` selects rows and calls `SetBaggingData`.
- **Callers of Boosting:** `src/application/application.cpp` (CLI train/predict) and `src/c_api.cpp` (Python/R/etc. bindings) via the abstract `Boosting*` handle.

**Data layout conventions to remember for a port:**
- `models_` is flat; tree for iteration `i`, class `k` is at index `i * num_tree_per_iteration_ + k`.
- Scores and gradients are **class-major**: element for class `j`, row `i` is at `j * num_data + i`.
- `num_tree_per_iteration_` = 1 (regression/binary) or `num_class` (multiclass).
- Gradients/hessians are `score_t` (`float`); scores are `double`; row indices/counts are `data_size_t` (`int32_t`); the ensemble is `std::vector<std::unique_ptr<Tree>>`.


---

## Section E — Tree Learner (CPU) — Growing One Tree

This section documents how LightGBM grows **one** decision tree. It assumes no
prior knowledge of LightGBM or of gradient boosting.

### 0. Background jargon (read this first)

LightGBM is a **gradient boosting** library: it builds an ensemble of many small
decision trees, one at a time. On each boosting iteration a "driver" (the `GBDT`
class, outside this subsystem) computes, for every training row, a **gradient**
and a **hessian** — the first and second derivatives of the loss with respect to
the model's current prediction for that row. It then asks a `TreeLearner` to grow
exactly one tree that best fits those gradients/hessians. This document is about
that single-tree step.

Vocabulary used throughout:

- **Split / node / leaf.** A tree is a set of internal *nodes* and terminal
  *leaves*. Each internal node holds a **split**: "if feature *f* of this row is
  `<= threshold` go left, else go right." Growing the tree = repeatedly turning a
  leaf into an internal node with two child leaves.
- **Bin.** LightGBM does not work with raw feature values. Before training, the
  `Dataset` (the `io/` subsystem) **bins** each feature: continuous values are
  bucketed into small integers `0, 1, 2, …` (typically up to 255 bins). A
  "threshold" is therefore an integer bin index, not a real number. This is what
  makes the algorithm fast.
- **Histogram.** For a given leaf and a given feature, the *histogram* is an array
  indexed by bin. Each entry stores two accumulated sums: the **sum of gradients**
  and the **sum of hessians** of all rows in that leaf whose feature value falls
  in that bin. Histograms are the central data structure of the whole subsystem.
- **Gain.** A number measuring how much a candidate split improves the loss.
  Higher is better. The learner always picks the split with the largest gain.
- **Leaf-wise vs level-wise growth.** Classic decision trees grow *level-wise*
  (split every node at depth *d* before going to depth *d+1*). LightGBM grows
  **leaf-wise**: at each step it splits the single leaf, anywhere in the tree,
  that yields the highest gain. It keeps doing this until it has `num_leaves`
  leaves. This produces deeper, more accurate (but more overfit-prone) trees.
- **Threshold, monotone constraint, CEGB, quantized gradients, linear tree** — all
  explained where they first appear below.

#### The core algorithm in one paragraph

To grow a tree, the learner starts with all data in one root leaf. It builds, for
each feature, a histogram over that leaf's rows. It then **scans** each histogram
bin-by-bin: sweeping a candidate threshold across the bins, it accumulates the
left-side gradient/hessian sums and derives the right-side sums by subtraction
from the leaf total, computing the gain of splitting there. The best (feature,
threshold) over all features becomes that leaf's candidate split. Across all
current leaves, the learner picks the leaf with the globally best gain, **splits**
it (physically partitioning its rows into a left and right child), and repeats.
Two accelerations dominate: (1) histograms are built over binned integers, not raw
values; (2) the **histogram subtraction trick** — after splitting a parent whose
histogram is known, only the *smaller* child's histogram is built from scratch; the
*larger* child's histogram is obtained for free as `parent - smaller`.

---

### 1. `include/LightGBM/tree_learner.h` — the abstract interface

Purpose: defines the abstract base class `TreeLearner`
(`tree_learner.h:27`) that every concrete learner implements, plus the factory
that constructs the right one. This is the porting seam: the Rust port replaces
this interface + its implementations.

Key virtual methods (all pure-virtual unless noted):

- `virtual void Init(const Dataset* train_data, bool is_constant_hessian)`
  (`:37`) — one-time setup with the training data. `is_constant_hessian` is a fast
  path: some losses (e.g. plain L2 regression) give every row the same hessian, so
  the per-row hessian array can be skipped in sums.
- `virtual Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree)`
  (`:68`) — **the main entry point**. Given per-row gradients and hessians for
  this iteration, grow and return one tree. Note `score_t` is `float` by default
  (see project `meta.h`), so `gradients`/`hessians` are `const float*`. Returns a
  raw owning `Tree*` (caller wraps it). `is_first_tree` matters only for linear
  trees.
- `virtual void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data)`
  (`:84`) — restrict training to a bagged subset of rows. `data_size_t` is
  `int32_t` (a signed row index/count type).
- `virtual void AddPredictionToScore(const Tree* tree, double* out_score) const`
  (`:92`) — after growing, add the new tree's per-row leaf outputs into the running
  score vector.
- `virtual Tree* FitByExistingTree(...)` (`:73`,`:75`) — reuse an existing tree's
  structure but recompute leaf outputs for new gradients (used by `refit`).
- `RenewTreeOutput`, `ResetConfig`, `ResetTrainingData`, `SetForcedSplit`,
  `ResetIsConstantHessian`, `InitLinear`, `ResetBoostingOnGPU` — reset/refresh
  hooks.

The factory (`tree_learner.h:110`):

```cpp
static TreeLearner* CreateTreeLearner(const std::string& learner_type,
                                      const std::string& device_type,
                                      const Config* config,
                                      const bool boosting_on_cuda);
```

Dependencies: forward-declares `Tree`, `Dataset`, `ObjectiveFunction`. Consumes a
`Dataset` (produced by `io/`), produces a `Tree` (defined in `io/tree.*`).

---

### 2. `src/treelearner/tree_learner.cpp` — the factory implementation

Purpose: the single function `TreeLearner::CreateTreeLearner`
(`tree_learner.cpp:15`). It is a **stringly-typed factory** switching on two
strings, `device_type` (`"cpu"`/`"gpu"`/`"cuda"`) and `learner_type`
(`"serial"`/`"feature"`/`"data"`/`"voting"`), plus `config->linear_tree` and
`config->num_gpu`.

For `device_type == "cpu"` (`:17`) the mapping is:

| `learner_type` | returned object |
|----------------|-----------------|
| `serial` + `linear_tree` | `new LinearTreeLearner<SerialTreeLearner>(config)` (`:20`) |
| `serial` | `new SerialTreeLearner(config)` (`:22`) |
| `feature` | `new FeatureParallelTreeLearner<SerialTreeLearner>(config)` (`:25`) |
| `data` | `new DataParallelTreeLearner<SerialTreeLearner>(config)` (`:27`) |
| `voting` | `new VotingParallelTreeLearner<SerialTreeLearner>(config)` (`:29`) |

The `gpu` and `cuda` branches (`:31`,`:45`) are out of scope here. Note the key
port-relevant fact: **all CPU learners are either `SerialTreeLearner` itself or a
template wrapping it** — the parallel learners are `Xxx<SerialTreeLearner>`, i.e.
they inherit from and override a few methods of the serial learner. Understand
`SerialTreeLearner` and you understand 90% of the subsystem.

---

### 3. `src/treelearner/serial_tree_learner.{h,cpp}` — the heart of the subsystem

This is the single-machine, single-thread-of-control (but OpenMP-parallel-inside)
tree learner. Everything else specializes it.

#### 3.1 State (members, `serial_tree_learner.h:185`–`239`)

- `const Dataset* train_data_` — the immutable binned dataset.
- `const score_t* gradients_`, `const score_t* hessians_` — this iteration's
  per-row grads/hessians (`float*`), set at the top of `Train`.
- `std::unique_ptr<DataPartition> data_partition_` — tracks which rows are in which
  leaf (see §4).
- `FeatureHistogram* parent_leaf_histogram_array_`,
  `smaller_leaf_histogram_array_`, `larger_leaf_histogram_array_` — pointers into
  the histogram pool for the three leaves involved in a split step (§5).
- `std::vector<SplitInfo> best_split_per_leaf_` — the current best candidate split
  for every leaf; the growth loop `ArgMax`es over this (§6).
- `std::unique_ptr<LeafSplits> smaller_leaf_splits_`, `larger_leaf_splits_` —
  per-leaf running sum_gradient/sum_hessian and the leaf's row-index list (§7).
- `ordered_gradients_`, `ordered_hessians_` — scratch buffers holding the leaf's
  gradients/hessians reordered to match the leaf's row order (cache-friendly
  histogram build).
- `HistogramPool histogram_pool_` — a fixed-size cache of histogram arrays, sized
  by `num_leaves` × total-bins, reused across leaves (the subtraction trick relies
  on this).
- `ColSampler col_sampler_` — feature subsampling (§8).
- `std::unique_ptr<CostEfficientGradientBoosting> cegb_` — optional CEGB (§9).
- `std::unique_ptr<GradientDiscretizer> gradient_discretizer_` — optional quantized
  training (§10).
- `std::unique_ptr<LeafConstraintsBase> constraints_` — monotone constraints (§11).
- `HistogramPool`-sized caches, forced-split JSON, share-state, etc.

#### 3.2 `Init` (`serial_tree_learner.cpp:30`)

Computes the histogram-pool cache size (`max_cache_size`, clamped between 2 and
`num_leaves`), allocates `best_split_per_leaf_`, the two `LeafSplits`, the
`DataPartition`, the ordered-gradient buffers, optionally the
`GradientDiscretizer`, then sizes the `histogram_pool_` via
`DynamicChangeSize(...)`. Also enables CEGB if configured.

#### 3.3 `Train` — the growth loop (`serial_tree_learner.cpp:179`)

Signature: `Tree* Train(const score_t* gradients, const score_t* hessians, bool)`.
Steps:

1. Stash `gradients_`/`hessians_` (`:181`). If quantized training is on, discretize
   the gradients now (`:193`).
2. `BeforeTrain()` (`:197`) — reset the histogram pool, resample features for this
   tree (`col_sampler_.ResetByTree()`), reset the `DataPartition` to put all rows in
   leaf 0, and compute the **root leaf's** total sum_gradient / sum_hessian into
   `smaller_leaf_splits_` (`:305`–`327`). It uses the full-data path when no bagging,
   else the partial-data path.
3. Allocate the empty `Tree` (`:200`) and set the root leaf's output by hand
   (`:205`, via `FeatureHistogram::CalculateSplittedLeafOutput`), since no split
   produced it.
4. `ForceSplits(...)` (`:216`) applies any user-forced splits from JSON (§3.7) and
   returns how many splits were forced.
5. **The leaf-wise loop** (`:218`–`236`), running until the tree has `num_leaves`
   leaves:
   - `BeforeFindBestSplit(tree, left_leaf, right_leaf)` (`:220`) — depth/min-data
     gating and histogram-pool bookkeeping (§3.5).
   - `FindBestSplits(tree)` (`:222`) — build histograms for the affected leaves and
     find each leaf's best split (§3.6).
   - `ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_)` (`:225`) — pick the leaf
     with the globally maximum gain. (`ArgMax` uses `SplitInfo::operator>`, §12.)
   - If the best gain `<= 0` (`:229`) stop: no further useful split.
   - `Split(tree, best_leaf, &left_leaf, &right_leaf)` (`:234`) — physically split
     the chosen leaf (§3.8). The returned `left_leaf`/`right_leaf` feed the next
     iteration's `BeforeFindBestSplit`.
6. If quantized + `quant_train_renew_leaf`, recompute leaf outputs from exact
   gradients (`:238`).
7. `return tree.release();` — hand ownership to the caller (`GBDT`).

#### 3.4 `BeforeTrain` (`serial_tree_learner.cpp:288`)

Resets per-tree state: `histogram_pool_.ResetMap()`, `col_sampler_.ResetByTree()`
(re-draw the per-tree feature subset), `data_partition_->Init()` (all rows to leaf
0), reset every `best_split_per_leaf_[i]`, and initialize the root leaf's sums in
`smaller_leaf_splits_`.

#### 3.5 `BeforeFindBestSplit` (`serial_tree_learner.cpp:340`)

Returns `false` (skip) when the leaf cannot be split:

- Depth check (`:343`): if `leaf_depth(left_leaf) >= config->max_depth`, disable both
  new leaves by setting their gain to `kMinScore`.
- Min-data check (`:356`): if both children have fewer than `2 * min_data_in_leaf`
  rows, skip.

Otherwise it wires up the **histogram-pool pointers for the subtraction trick**
(`:364`–`378`). This is the crucial bit:

- Root case (`right_leaf < 0`, `:366`): just fetch a histogram slot for the root
  into `smaller_leaf_histogram_array_`; there is no larger leaf or parent.
- General case: the just-split parent leaf is `left_leaf`. Compare child row counts.
  The **smaller** child (fewer rows) gets a fresh histogram (`smaller_leaf_...`),
  the **larger** child inherits the parent's histogram slot (which becomes
  `larger_leaf_...` **and** `parent_leaf_histogram_array_`). If left is smaller, the
  parent slot is *moved* to the right leaf id via `histogram_pool_.Move` (`:372`).
  Because `parent_leaf_histogram_array_` is now non-null, `FindBestSplits` will use
  subtraction for the larger child.

#### 3.6 `FindBestSplits` → `ConstructHistograms` → `FindBestSplitsFromHistograms`

`FindBestSplits(tree, force_features)` (`serial_tree_learner.cpp:386`):

1. Build `is_feature_used` (a `std::vector<int8_t>`, per inner-feature flag)
   respecting the column sampler and pruning features the parent already found
   unsplittable (`:388`–`397`).
2. `use_subtract = parent_leaf_histogram_array_ != nullptr` (`:398`).
3. `ConstructHistograms(is_feature_used, use_subtract)` (`:400`).
4. `FindBestSplitsFromHistograms(is_feature_used, use_subtract, tree)` (`:401`).

`ConstructHistograms` (`:404`): delegates the actual per-row accumulation to
`train_data_->ConstructHistograms<...>(...)` (in the `io/` subsystem — that is where
the hot OpenMP loop over rows lives). It always builds the **smaller** leaf's
histogram (`:452`–`459` non-quantized path). It builds the larger leaf's histogram
**only if `!use_subtract`** (`:460`); when subtraction is possible, the larger leaf
is filled later by subtracting. The `hist_t* ... - kHistOffset` pointer arithmetic
(`:454`) positions the raw buffer; `hist_t` is the histogram accumulator element
type (a `double` per gradient/hessian slot by default). The quantized branch
(`:409`–`451`) is the same idea with 16- or 32-bit integer histogram entries.

`FindBestSplitsFromHistograms` (`serial_tree_learner.cpp:473`) — this is where the
subtraction trick and the per-feature scan happen:

- Allocate per-thread best-split arrays `smaller_best`, `larger_best` (one
  `SplitInfo` per OpenMP thread, `:477`).
- Compute each leaf's used-feature mask via `col_sampler_.GetByNode` (per-node
  feature sampling, `:479`) and its `parent_output` (for path smoothing, `:481`).
- The big OpenMP loop over features (`:511`–`606`):
  - `FixHistogram(...)` (`:531`) reconstructs the bin-0 / most-frequent-bin entry
    for the smaller leaf so the histogram totals equal the leaf totals (a numeric
    correction; quantized uses `FixHistogramInt`, `:522`).
  - `ComputeBestSplitForFeature(smaller_leaf_histogram_array_, …)` (`:538`) — scan
    this feature's histogram for its best threshold (§3.9).
  - If a larger leaf exists and `use_subtract` (`:551`): derive its histogram with
    `larger_leaf_histogram_array_[f].Subtract<...>(smaller_leaf_histogram_array_[f])`
    (`:574` non-quantized) — **the subtraction trick**. Otherwise `FixHistogram` the
    larger leaf directly (it was built fresh).
  - `ComputeBestSplitForFeature(larger_leaf_histogram_array_, …)` (`:598`).
- Reduce the per-thread bests via `ArrayArgs<SplitInfo>::ArgMax` and store into
  `best_split_per_leaf_[smaller_leaf]` and `[larger_leaf]` (`:608`–`617`).

#### 3.7 `ForceSplits` (`serial_tree_learner.cpp:620`)

Applies a user-supplied JSON tree of `{feature, threshold, left, right}` forced
splits, breadth-first, using `FeatureHistogram::GatherInfoForThreshold` to evaluate
a *specific* threshold rather than searching. Feeds `best_split_per_leaf_` then
calls `Split`. Returns the count of forced splits so the main loop resumes after
them.

#### 3.8 `Split` / `SplitInner` (`serial_tree_learner.cpp:160`,`762`)

`Split` just calls `SplitInner(..., update_cnt=true)`. `SplitInner` physically
performs the chosen split:

1. Look up the winning `SplitInfo` in `best_split_per_leaf_[best_leaf]` (`:765`).
2. `constraints_->BeforeSplit(...)` for monotone bookkeeping (`:776`).
3. If numerical (`:782`): `data_partition_->Split(best_leaf, train_data_, feature,
   &threshold, 1, default_left, next_leaf_id)` — reorder the leaf's rows into left
   (`<= threshold`) and right (`> threshold`) partitions (§4). Then
   `tree->Split(...)` (`:794`) records the node in the `Tree` model: feature,
   integer threshold, real-valued threshold, left/right outputs, counts, hessian
   sums, gain (`gain + min_gain_to_split`, `:804`), missing-value handling and
   `default_left`.
4. Categorical splits (`:807`) use a bitset of category bins and
   `tree->SplitCategorical`.
5. Re-seed `smaller_leaf_splits_`/`larger_leaf_splits_` for the two new children
   (`:850`–`898`): the child with fewer rows becomes the "smaller" leaf so the *next*
   step builds the smaller histogram and subtracts for the larger — this is what
   keeps the subtraction trick going down the tree.
6. Monotone-constraint propagation may force `RecomputeBestSplitForLeaf` on other
   leaves (`:909`–`917`).

Types flowing here: `best_split_info.threshold` is `uint32_t` (a bin index);
`left_count`/`right_count` are `data_size_t` (`int32_t`); outputs/hessian sums are
`double`.

#### 3.9 `ComputeBestSplitForFeature` (`serial_tree_learner.cpp:960`)

Per-feature glue that calls into `FeatureHistogram`:

- Non-quantized: `histogram_array_[f].FindBestThreshold(sum_gradient, sum_hessian,
  num_data, constraint, parent_output, &new_split)` (`:983`).
- Quantized: `FindBestThresholdInt(...)` (`:974`).
- Applies CEGB gain penalty (`:989`) and monotone-penalty multiplier (`:996`).
- Keeps `new_split` only if `new_split > *best_split && is_feature_used` (`:1000`) —
  note the tie-break rule is in `SplitInfo::operator>` (§12).

#### 3.10 `AddPredictionToScore` (`serial_tree_learner.h:100`), `RenewTreeOutput`
(`.cpp:920`), `FitByExistingTree` (`.cpp:247`)

- `AddPredictionToScore`: for each leaf, add `LeafOutput(leaf)` to `out_score[row]`
  for every row in that leaf (row lists from `data_partition_`).
- `RenewTreeOutput`: for objectives that override leaf outputs (e.g. some ranking /
  robust losses), recompute each leaf's output via `obj->RenewTreeOutput`, with an
  MPI GlobalSum across machines.
- `FitByExistingTree`: keep an old tree's structure, recompute each leaf's output
  from new gradients (used by `refit`), blending with `refit_decay_rate`.

---

### 4. `src/treelearner/data_partition.hpp` — who-is-in-which-leaf

Purpose: `DataPartition` (`data_partition.hpp:21`) stores, as one flat
`indices_` array, all row indices grouped by leaf: `[rows of leaf0 | rows of leaf1
| …]`. Two parallel arrays `leaf_begin_[leaf]` and `leaf_count_[leaf]` give each
leaf's slice. This is the mutable per-tree structure that lets the learner know
which rows to sum when building a leaf's histogram.

Key methods:

- `Init()` (`:49`): put every row into leaf 0 (or the bagged subset).
- `const data_size_t* GetIndexOnLeaf(int leaf, data_size_t* out_len) const` (`:87`):
  return a pointer to that leaf's row-index slice and its length. Used everywhere a
  leaf's rows are iterated.
- `void Split(int leaf, const Dataset* dataset, int feature, const uint32_t* threshold, int num_threshold, bool default_left, int right_leaf)`
  (`:101`): the physical partition. It runs `dataset->Split(...)` over the leaf's
  rows via a `ParallelPartitionRunner` (`runner_`, `:109`) that stably reorders
  `indices_` so that rows going left come first; then it updates `leaf_count_[leaf]`
  (= left count) and sets `leaf_begin_[right_leaf]` / `leaf_count_[right_leaf]`
  (`:117`–`119`). Note: the actual "which side does row *r* go" test lives in
  `Dataset::Split` (the `io/` subsystem), because it depends on the binned column
  storage.
- `leaf_count`, `leaf_begin`, `indices`, `SetUsedDataIndices` (bagging),
  `ResetByLeafPred` (refit).

The MEMORY note that `DataPartition::split` is single-threaded is the known CPU-vs-
C++ perf gap; the `runner_` is `ParallelPartitionRunner` but memory-bandwidth-bound.

---

### 5. `src/treelearner/feature_histogram.{hpp,cpp}` — histograms + best-split scan

Purpose: `FeatureHistogram` (`feature_histogram.hpp:43`) owns one feature's
histogram buffer for one leaf and contains the **split-finding scan** — the
algorithmic core. `FeatureMetainfo` (`:25`) holds the per-feature metadata the scan
needs: `num_bin`, `missing_type`, `offset`, `default_bin`, `monotone_type`,
`penalty`, `bin_type`, and the `Config*`.

#### 5.1 Storage

- `hist_t* data_` (`:61`) — the raw histogram: interleaved
  `[grad_bin0, hess_bin0, grad_bin1, hess_bin1, …]`. `GET_GRAD(data_, t)` /
  `GET_HESS(data_, t)` (macros) read bin `t`. Default `hist_t` is `double`.
- `int16_t* data_int16_` — the 16-bit integer histogram for quantized mode. 32-bit
  entries reuse `data_` reinterpreted (`RawDataInt32`, `:88`).

#### 5.2 `Subtract` — the histogram subtraction trick (`:99`)

```cpp
template <bool USE_DIST_GRAD=false, …>
void Subtract(const FeatureHistogram& other, const int32_t* buffer=nullptr);
```

Non-quantized path (`:140`–`144`): `for i: data_[i] -= other.data_[i];` — element-
wise `this = this - other`. In use, `this` is the larger child (initialized to the
parent's histogram) and `other` is the freshly built smaller child, so afterwards
`this` holds the larger child's histogram without ever scanning its rows. The
quantized branches handle mixed 16/32-bit packings.

#### 5.3 `FindBestThreshold` (`:165`) and the dispatch tower

`FindBestThreshold(double sum_gradient, double sum_hessian, data_size_t num_data,
const FeatureConstraint* constraints, double parent_output, SplitInfo* output)` is
the public entry. It sets `output->gain = kMinScore`, then calls a stored
`std::function` `find_best_threshold_fun_`, then multiplies gain by the feature's
`penalty`.

That `std::function` is compiled once by the `FuncForNumrical` /
`FuncForNumricalL1/L2/L3` template tower (`:231`–`447`). This tower is
LightGBM's way of turning many boolean config flags into a specialized,
branch-free inner loop. Each level picks one compile-time flag:

- `USE_RAND` — extremely-randomized-trees mode (random threshold).
- `USE_MC` — monotone constraints present.
- `USE_L1` — L1 regularization (`lambda_l1 > 0`).
- `USE_MAX_OUTPUT` — `max_delta_step > 0`.
- `USE_SMOOTHING` — path smoothing (`path_smooth > 0`).

plus missing-value handling (`MissingType::Zero`/`NaN`/`None`), which decides
whether the scan runs forward, reverse, or both (to try putting missing values on
the default side). The quantized twin is `FuncForNumricalL3`'s
`use_quantized_grad` branch producing `int_find_best_threshold_fun_`.

#### 5.4 `FindBestThresholdSequentially` — the actual scan (`:830`)

This is the algorithm you must reproduce bit-exactly. Signature (template flags
omitted): it takes `sum_gradient`, `sum_hessian`, `num_data`, `constraints`,
`min_gain_shift`, `SplitInfo* output`, `rand_threshold`, `parent_output`.

The `REVERSE` branch (`:854`, used to make missing/large bins default-left)
iterates bins from right to left; the forward branch (`:937`) left to right. Take
the forward branch as the model:

- Running left-side accumulators: `sum_left_gradient`, `sum_left_hessian`,
  `left_count` (`:938`).
- For each bin `t` from `0` to `num_bin-2-offset` (`:963`):
  - Add bin `t`'s grad/hess to the left sums; derive `left_count` from
    `hess * cnt_factor` where `cnt_factor = num_data / sum_hessian` (`:970`–`973`).
    (Counts are recovered from hessian sums because the histogram stores sums, not
    counts.)
  - Enforce `min_data_in_leaf` and `min_sum_hessian_in_leaf` on both sides
    (`:976`,`:982`,`:988`) — `continue`/`break` if violated.
  - Right sums are the leaf total minus the left sums (`:986`,`:992`) — this is the
    same "subtract from parent" idea, one level down, applied to left/right of a
    threshold.
  - `current_gain = GetSplitGains<...>(left…, right…, parent_output)` (`:999`).
  - If `current_gain > min_gain_shift`, mark `is_splittable_ = true`; if it beats
    `best_gain`, record `best_threshold = t + offset`, best left sums, best count
    (`:1013`–`1027`).
- After the scan, if splittable and better than the incoming `output->gain`
  (`:1031`), fill `output`: `threshold`, `left_output`/`right_output` (via
  `CalculateSplittedLeafOutput`), counts, gradient/hessian sums, `gain = best_gain
  - min_gain_shift`, and `default_left = REVERSE` (`:1055`).

The integer twin `FindBestThresholdSequentiallyInt` (`:1062`) mirrors this over
packed int histograms (further in the file, beyond the excerpt shown).

#### 5.5 The gain/output math (`:711`–`828`) — reproduce these exactly

- `ThresholdL1(s, l1) = sign(s) * max(0, |s| - l1)` (`:711`) — soft-threshold for L1.
- `CalculateSplittedLeafOutput` (`:717`): a leaf's optimal output is
  `-ThresholdL1(sum_grad, l1) / (sum_hess + l2)` (or `-sum_grad/(sum_hess+l2)`
  without L1), optionally clamped to `±max_delta_step`, optionally blended with the
  parent via path smoothing: `ret*(n/s)/(n/s+1) + parent_output/(n/s+1)`.
- `GetLeafGain` (`:800`): with no max-output/smoothing, `gain = sg_l1^2 /
  (sum_hess + l2)` (or `sum_grad^2/(sum_hess+l2)`), i.e. the standard XGBoost/LightGBM
  leaf gain; otherwise it derives gain from the computed output via
  `GetLeafGainGivenOutput` (`:818`): `-(2*sg_l1*output + (sum_hess+l2)*output^2)`.
- `GetSplitGains` (`:759`): `= GetLeafGain(left) + GetLeafGain(right)`. With
  monotone constraints it computes clamped outputs and returns `0` if the split
  would violate monotonicity (`:788`).
- The split's reported gain is `left_gain + right_gain - (parent_gain +
  min_gain_to_split)`; `min_gain_shift` (`:207`) folds in `parent_gain +
  min_gain_to_split`.

#### 5.6 `GatherInfoForThreshold` (`:474`)

Evaluate a *specific* given `threshold` (not a search) and fill a `SplitInfo`. Used
by `ForceSplits`. Numerical version at `:486`, categorical at `:590`.

`feature_histogram.cpp` contains the out-of-line categorical scan
(`FuncForCategorical`, `FindBestThresholdCategoricalInner`, and the int variant),
which handles one-hot and many-vs-many categorical splits analogously.

---

### 6. `src/treelearner/split_info.hpp` — the candidate-split record

Purpose: `SplitInfo` (`split_info.hpp:22`) is the plain struct describing one
candidate or chosen split. Fields (with types):

- `int feature = -1` — **real** (raw-data) feature index; `-1` means "no split".
- `uint32_t threshold = 0` — the bin-index threshold.
- `data_size_t left_count`, `right_count` — rows per side (`int32_t`).
- `int num_cat_threshold`, `std::vector<uint32_t> cat_threshold` — categorical split
  bitset.
- `double left_output`, `right_output` — the two children's leaf outputs.
- `double gain = kMinScore` — the split gain; `kMinScore` = -inf sentinel.
- `double left_sum_gradient`, `left_sum_hessian`, `right_sum_gradient`,
  `right_sum_hessian` — the per-side accumulated sums.
- `int64_t left_sum_gradient_and_hessian`, `right_sum_gradient_and_hessian` —
  packed integer sums for quantized mode.
- `bool default_left = true` — which way missing values go.
- `int8_t monotone_type = 0`.

`CopyTo`/`CopyFrom` (`:59`,`:95`) serialize it for network sync (distributed
learning). `Reset()` (`:132`) sets `feature=-1, gain=kMinScore`.

The comparison operators are load-bearing for reproducibility:

- `operator>` (`:138`): compare by `gain`; **on a gain tie, prefer the smaller
  feature index** (treating `-1` as `INT32_MAX`). This deterministic tie-break is
  what `ArrayArgs::ArgMax` relies on when selecting the best leaf/feature.

`LightSplitInfo` (`:198`) is a slimmed version (feature, gain, counts only) used in
the distributed voting learner to cut network traffic.

---

### 7. `src/treelearner/leaf_splits.hpp` — per-leaf running sums

Purpose: `LeafSplits` (`leaf_splits.hpp:22`) caches, for one leaf, its total
`sum_gradients_`, `sum_hessians_`, `num_data_in_leaf_`, `leaf_index_`, its row-index
pointer `data_indices_` (borrowed from `DataPartition`), its `weight_` (= the
leaf's output, used as `parent_output` for children when smoothing), and the packed
`int_sum_gradients_and_hessians_` for quantized mode.

The many `Init(...)` overloads (`:47`–`219`) cover: init from a data-partition leaf
(re-summing over the leaf's rows in an OpenMP reduction, `:146`), init from
precomputed sums (used after a split when the parent already computed each child's
sums, `:47`), init from the full dataset for the root (`:92`), and the quantized
twins. The reduction loops guard with `if (num_data_in_leaf_ >= 1024 &&
!deterministic_)` so that deterministic mode forces serial summation for bit-exact
reproducibility.

Accessors: `leaf_index()`, `num_data_in_leaf()`, `sum_gradients()`,
`sum_hessians()`, `int_sum_gradients_and_hessians()`, `data_indices()`, `weight()`.

---

### 8. `src/treelearner/col_sampler.hpp` — feature subsampling

Purpose: `ColSampler` (`col_sampler.hpp:20`) implements `feature_fraction`
(per-tree, aka `colsample_bytree`) and `feature_fraction_bynode` (per-node) random
feature subsampling, plus **interaction constraints** (restrict which features may
co-occur on a branch).

- `SetTrainingData` (`:39`) computes `used_cnt_bytree_` = round(`num_features *
  feature_fraction`) and calls `ResetByTree`.
- `ResetByTree` (`:74`): if per-tree sampling is active, randomly sample
  `used_cnt_bytree_` features (`random_.Sample`) and set the `is_feature_used_`
  byte-mask. Called at the start of each tree.
- `is_feature_used_bytree()` (`:181`): the per-tree mask that `FindBestSplits`
  consults.
- `GetByNode(const Tree* tree, int leaf)` (`:91`): returns a fresh per-node
  `std::vector<int8_t>` mask, honoring `feature_fraction_bynode` and interaction
  constraints for this leaf's branch. Called by `FindBestSplitsFromHistograms`.

Determinism note: the RNG is seeded from `feature_fraction_seed`, so feature subsets
are reproducible.

---

### 9. `src/treelearner/cost_effective_gradient_boosting.hpp` — CEGB

Purpose: `CostEfficientGradientBoosting` (`:23`) implements **CEGB** —
cost-effective gradient boosting, which penalizes the split gain by the *cost* of
using a feature (encouraging cheaper models for prediction-time budgets).

- `IsEnable(config)` (`:27`): CEGB is on unless `cegb_tradeoff >= 1`,
  `cegb_penalty_split <= 0`, and both `cegb_penalty_feature_*` lists are empty.
- `DeltaGain(feature_index, real_fidx, leaf_index, num_data_in_leaf, split_info)`
  (`:79`): the gain penalty subtracted in `ComputeBestSplitForFeature` (`:989`).
  Components: `cegb_tradeoff * cegb_penalty_split * num_data_in_leaf`, plus a
  one-time per-feature "coupled" penalty the first time a feature is used, plus a
  per-row "lazy" on-demand cost computed in `CalculateOndemandCosts` (`:139`), which
  charges only for rows that touch the feature for the first time (tracked in the
  `feature_used_in_data_` bitset).
- `UpdateLeafBestSplits` (`:99`): after a split, once a feature becomes "used", it
  refunds the coupled penalty on other leaves' cached splits and re-picks their
  bests.

---

### 10. `src/treelearner/gradient_discretizer.{hpp,cpp}` — quantized training

Purpose: `GradientDiscretizer` (`gradient_discretizer.hpp:22`) implements
**quantized gradient training** (`use_quantized_grad`): gradients and hessians are
rounded to small integers (e.g. 4–8 bit "grad quant bins") so histograms can
accumulate in `int16`/`int32` instead of `double`, which is faster and, on GPUs,
enables integer atomics. It is an *approximate* mode.

- `DiscretizeGradients(num_data, gradients, hessians)` (`:35`): quantize the float
  grads/hessians into `discretized_gradients_and_hessians_vector_` (packed
  `int8`), with `stochastic_rounding_` for unbiased rounding. Computes
  `gradient_scale_`/`hessian_scale_` (multipliers to recover the doubles).
- `grad_scale()`/`hess_scale()` (`:44`,`:48`): the recovery scale factors passed to
  `FindBestThresholdInt`.
- `SetNumBitsInHistogramBin<IS_GLOBAL>` (`:56`): choose, per leaf/node, whether a
  16- or 32-bit histogram bin suffices (small leaves fit in 16 bits). This drives
  all the `<…,16,16>` vs `<…,32,32>` template selection back in
  `serial_tree_learner.cpp` and `feature_histogram.hpp`.
- `GetHistBitsInLeaf`/`GetHistBitsInNode` (`:61`,`:70`): query those bit widths.
- `ordered_int_gradients_and_hessians()` (`:79`), `GetChangeHistBitsBuffer` (`:88`):
  scratch buffers for the histogram build and for widening a 16-bit histogram to
  32-bit when subtracting.
- `RenewIntGradTreeOutput` (`:83`): after growing, recompute each leaf's output from
  the *exact* (non-quantized) gradients so the final leaf values aren't degraded by
  quantization (`quant_train_renew_leaf`).

`gradient_discretizer.cpp` holds the implementations (the actual rounding loops,
scale computation, and the per-leaf bit-width heuristic).

---

### 11. `src/treelearner/monotone_constraints.hpp` — monotone constraints

Purpose: enforce **monotone constraints** — the requirement that the model's
prediction be non-decreasing (`+1`) or non-increasing (`-1`) in a chosen feature.

Key types:

- `BasicConstraint { double min, max; }` (`:24`) — an allowed output interval.
- `FeatureConstraint` (`:33`) — abstract: `LeftToBasicConstraint()`,
  `RightToBasicConstraint()`, `ConstraintDifferentDependingOnThreshold()`,
  plus `InitCumulativeConstraints`/`Update` (for the "advanced/intermediate"
  constraint method that tightens as the scan sweeps thresholds).
- `ConstraintEntry` (`:42`) — per-leaf mutable min/max with `UpdateMin`/`UpdateMax`.
- `BasicConstraintEntry` (`:58`) — the simple implementation (constraints don't vary
  with threshold).
- `LeafConstraintsBase` (forward-declared; factory `Create` used at
  `serial_tree_learner.cpp:51`) — owns all leaves' constraints; its `Update` after a
  split (`serial_tree_learner.cpp:909`) may require recomputing sibling leaves'
  bests (that is why `RecomputeBestSplitForLeaf` exists).

In the scan, `GetSplitGains` (feature_histogram `:759`) clamps child outputs to the
constraint intervals and returns `0` gain if the split violates the required
direction (`:788`), so monotonicity is enforced by killing offending splits.
`ComputeMonotoneSplitGainPenalty` additionally scales gain by `monotone_penalty`.

---

### 12. Selection helpers (used but defined elsewhere)

- `ArrayArgs<SplitInfo>::ArgMax(...)` (in `utils/array_args.h`) — returns the index
  of the maximum `SplitInfo` using `operator>` (so with the deterministic
  smaller-feature-index tie-break). Used to pick the best leaf
  (`serial_tree_learner.cpp:225`) and to reduce per-thread bests (`:608`).
- `HistogramPool` (in `io/`) — the reused pool of histogram arrays; `Get`, `Move`,
  `ResetMap`, `DynamicChangeSize`. Central to the subtraction trick's memory reuse.

---

### 13. `src/treelearner/linear_tree_learner.{h,cpp}` — linear trees

Purpose: `LinearTreeLearner<TREE_LEARNER_TYPE>`
(`linear_tree_learner.h:20`) is a template that **inherits from** the base learner
(`SerialTreeLearner` on CPU) and adds **linear leaves**: instead of each leaf
predicting a single constant, it predicts a *linear regression* of the leaf's rows
on their numeric features, `output = const + Σ coeff_k * feature_k`. This is the
`linear_tree=true` model.

- It reuses the base learner's whole split-finding machinery for tree *structure*;
  it only overrides the leaf-output computation.
- `InitLinear` (`.cpp:20`): detect which features contain NaN (`contains_nan_`,
  `any_nan_`) and preallocate the per-leaf normal-equation matrices `XTHX_`
  (upper-triangular `XᵀHX`) and `XTg_` (`Xᵀg`), plus per-thread copies for parallel
  accumulation. `max_num_feat = min(max_leaves, num_numeric_features)`.
- `Train` (`.cpp:65`): identical growth loop to `SerialTreeLearner::Train`
  (BeforeTrain → ForceSplits → leaf-wise split loop), but with a *3-argument* `Tree`
  ctor flagged linear (`.cpp:81`), and after growth it builds `leaf_map_` (row→leaf,
  `GetLeafMap`) and calls `CalculateLinear<HAS_NAN>(...)` (`.cpp:126`) to solve the
  per-leaf ridge-regularized least-squares (via Eigen, `#include <Eigen/Dense>`) for
  each leaf's `const` + coefficients.
- `AddPredictionToScore`/`AddPredictionToScoreInner<HAS_NAN>` (`.h:41`,`:62`):
  prediction adds `leaf_const + Σ coeff*feature` per row (falling back to the plain
  leaf output when a required feature is NaN).

Dependency: Eigen (linear algebra) for the least-squares solve — a licensing/port
consideration.

---

### 14. Distributed (parallel) learners — `parallel_tree_learner.h` +
`feature_parallel_tree_learner.cpp`, `data_parallel_tree_learner.cpp`,
`voting_parallel_tree_learner.cpp`

Purpose: three `template <typename TREELEARNER_T> class Xxx : public TREELEARNER_T`
wrappers (`parallel_tree_learner.h:26`,`:53`,`:126`) that add **multi-machine**
training via the `Network` subsystem (MPI or sockets). They override only a handful
of the base learner's hooks; the per-machine split math is unchanged. Each row/
feature is `int32_t`-indexed; `comm_size_t` is `int32_t` for network sizes.

- **`FeatureParallelTreeLearner`** (`parallel_tree_learner.h:26`): each machine has
  *all* rows but is responsible for finding the best split among *different
  features*; machines then sync the single global best split. Recommended when #data
  is small, #features large. Overrides `BeforeTrain` and
  `FindBestSplitsFromHistograms` (which ends by broadcasting the best split via
  `SyncUpGlobalBestSplit`).

- **`DataParallelTreeLearner`** (`parallel_tree_learner.h:53`): each machine holds a
  *shard of the rows*, builds *local* histograms, then all machines
  **reduce-scatter** the histograms so that each machine owns the *global* histogram
  for a subset of features, finds its features' best splits, and all-reduces the
  global best. Recommended when #data is large, #features small. Key overrides:
  `BeforeTrain` (`data_parallel_tree_learner.cpp:124` — assigns features to machines
  by balancing total bins, and all-reduces the root's global grad/hess sums),
  `FindBestSplits` (`:222` — builds local histograms then reduce-scatters them),
  `FindBestSplitsFromHistograms`, and `Split`. It keeps a separate
  `global_data_count_in_leaf_` because each machine only sees local row counts;
  `GetGlobalDataCountInLeaf` (`parallel_tree_learner.h:66`) is overridden to read it.
  `PrepareBufferPos` lays out the reduce-scatter byte blocks (with 16-bit vs 32-bit
  histogram variants for quantized mode).

- **`VotingParallelTreeLearner`** (`parallel_tree_learner.h:126`): like data-parallel
  but avoids all-reducing *every* feature's histogram. Each machine locally votes
  for its top-`top_k_` features (`GlobalVoting`, `:153`), only the globally selected
  features' histograms are aggregated (`CopyLocalHistogram`, `:160`), cutting network
  cost when both #data and #features are large. Keeps its own global `LeafSplits` and
  global `FeatureHistogram[]` buffers (`:195`–`205`).

Shared helper: `SyncUpGlobalBestSplit(...)` (`parallel_tree_learner.h:209`)
serializes the smaller/larger best `SplitInfo` into buffers and `Network::Allreduce`s
them with a custom reducer that keeps the winning split per the `LightSplitInfo`
`operator>` (same deterministic tie-break as §6).

The `.cpp` files also contain each learner's constructor/`Init`/`ResetConfig`
(allocating network buffers sized from `Network::num_machines()` and `rank_`).

---

### 15. How the subsystem fits the training pipeline

1. Once per run: `GBDT::Init` calls `TreeLearner::CreateTreeLearner(...)` (§2) then
   `learner->Init(dataset, is_constant_hessian)` (§3.2).
2. Once per boosting iteration: the objective function computes `gradients` /
   `hessians` (`float*`); `GBDT` calls
   `Tree* t = learner->Train(gradients, hessians, is_first_tree)` (§3.3), which grows
   and returns one tree (leaf-wise, histogram-based, subtraction-accelerated).
3. `GBDT` calls `learner->AddPredictionToScore(t, score)` to update the running
   per-row scores, optionally `RenewTreeOutput`, and appends `t` to the model.
4. Under distributed training the same calls run on every machine, but the parallel
   wrapper (§14) inserts `Network` reduce/allreduce steps so all machines agree on
   the same tree.

Inputs: `const Dataset*` (binned columnar data + histogram construction, from
`io/`), `const score_t* gradients/hessians` (`float`), `const Config*`. Output: an
owning `Tree*` (from `io/tree.*`) recording the split structure and leaf outputs.
The `Dataset` is read-only during training; the only mutable per-tree structures are
the `DataPartition`, the `HistogramPool`, and the `LeafSplits`/`best_split_per_leaf_`
bookkeeping.


---

## Section F — Objective Functions (Loss → Gradient/Hessian)

This section documents the **objective function** subsystem of LightGBM: the code
that turns "how wrong the model currently is" into the numeric signal the tree
learner uses to grow the next tree. All file references point into
`LightGBM/` (the read-only C++ reference).

Files covered:

- `include/LightGBM/objective_function.h` — abstract interface.
- `src/objective/objective_function.cpp` — the string-keyed factory.
- `src/objective/regression_objective.hpp` — L2/L1/Huber/Fair/Poisson/Quantile/MAPE/Gamma/Tweedie.
- `src/objective/binary_objective.hpp` — binary logloss.
- `src/objective/multiclass_objective.hpp` — softmax and one-vs-all.
- `src/objective/rank_objective.hpp` — lambdarank and rank_xendcg.
- `src/objective/xentropy_objective.hpp` — cross-entropy and its lambda variant.

---

### 1. What an objective function is (plain language)

Gradient boosting builds an ensemble (a sum) of decision trees. It works
**iteratively**: it already has a current model that produces a **score** for
every training row, and on each round it adds one more tree that nudges those
scores toward the correct answer.

To decide *which direction* and *how hard* to nudge, boosting needs to know, for
each row, how the **loss** would change if the score changed. Some jargon,
defined once:

- **Loss** — a number measuring how wrong a prediction is for one row. Example:
  squared error `(score - label)²`. Smaller is better. "Label" = the true target
  value.
- **Gradient** — the *first derivative* of the loss with respect to the score.
  Intuitively: "if I increase this row's score a tiny bit, does the loss go up or
  down, and how steeply?" It is the direction/steepness of the error. For squared
  error the gradient is `score - label`, i.e. the signed residual (how far off you
  are). The tree is fit to **push scores in the direction that reduces this**.
- **Hessian** — the *second derivative* of the loss with respect to the score.
  Intuitively: "how fast does the gradient itself change?" — the curvature. It
  tells boosting how big a step is safe: a high hessian means the loss curves
  sharply so take small steps; a low hessian means take bigger steps. LightGBM
  uses it as a per-row weight and to compute optimal leaf values (a Newton step).

So the objective function's core job, `GetGradients`, is: **given the current
scores and the true labels, produce one gradient and one hessian per training
row.** The tree learner then builds histograms of these gradient/hessian pairs
and finds the split that best reduces the loss. This is why the header comment
(`objective_function.h:32`) calls `GetGradients` "calculating first order
derivative of loss function."

The other job is **output conversion** (`ConvertOutput`): the raw score the trees
sum to is often not the final answer a user wants. For binary classification the
trees produce an unbounded number; `ConvertOutput` runs it through a **sigmoid**
(a squashing function `1/(1+e^-x)` that maps any real number to a probability in
(0,1)) to get a probability. For Poisson regression it applies `exp`. For plain
regression it is the identity.

---

### 2. The abstract interface and the factory

#### `class ObjectiveFunction` (`objective_function.h:19`)

An abstract base class (pure interface). Every concrete loss subclasses it.
Copy is disabled (`objective_function.h:89-91`). Key virtual methods:

| Method | Signature (file:line) | Purpose |
|---|---|---|
| `Init` | `virtual void Init(const Metadata& metadata, data_size_t num_data) = 0` (`:29`) | Cache label/weight pointers and precompute per-objective state. Called once before training. |
| `GetGradients` | `virtual void GetGradients(const double* score, score_t* gradients, score_t* hessians) const = 0` (`:37`) | **The core call.** Read current `score`, write per-row `gradients` and `hessians`. |
| `GetGradients` (sampled) | `virtual void GetGradients(const double* score, const data_size_t num_sampled_queries, const data_size_t* sampled_query_indices, score_t*, score_t*) const` (`:48`) | Query-bagging overload used only by lambdarank; default just forwards to the plain overload. |
| `GetName` | `virtual const char* GetName() const = 0` (`:51`) | String name, e.g. `"regression"`, `"binary"`. |
| `IsConstantHessian` | `virtual bool IsConstantHessian() const { return false; }` (`:53`) | True when every hessian equals a constant (e.g. L2 with no weights → always 1). Lets the tree learner skip storing hessians. |
| `IsRenewTreeOutput` / `RenewTreeOutput` | (`:55`, `:57`) | For losses where the optimal leaf value is not the gradient/hessian ratio (L1, quantile, MAPE), recompute the leaf output after the tree structure is fixed — e.g. the median of residuals. |
| `BoostFromScore` | `virtual double BoostFromScore(int class_id) const { return 0.0; }` (`:65`) | The constant initial score to start boosting from (the "intercept"), e.g. the mean label for L2. |
| `ClassNeedTrain` | `virtual bool ClassNeedTrain(int) const { return true; }` (`:67`) | Skip training a class that has only one label value. |
| `SkipEmptyClass` | (`:69`) | Related multiclass/binary optimization flag. |
| `NumModelPerIteration` | `virtual int NumModelPerIteration() const { return 1; }` (`:71`) | Trees grown per boosting round. `1` for most; `num_class` for multiclass. |
| `NumPredictOneRow` | (`:73`) | Number of output values per row (again `num_class` for multiclass). |
| `NeedAccuratePrediction` | (`:76`) | If false, prediction may use early stopping. |
| `NumPositiveData` | (`:79`) | For binary: count of positive samples. |
| `ConvertOutput` | `virtual void ConvertOutput(const double* input, double* output) const { output[0] = input[0]; }` (`:81`) | Map raw summed score to final output. Default is identity. |
| `ToString` | `virtual std::string ToString() const = 0` (`:85`) | Serialize config into the model file (round-tripped by the string constructor). |

Inputs/outputs use these concrete types (from `include/LightGBM/meta.h`, summarized
in project CLAUDE.md):

- `score` — `const double*`, length `num_data_` (or `num_data_ * num_class`). The
  current model scores. **Note it is `double`**, higher precision than the
  gradient output.
- `gradients`, `hessians` — `score_t*` = `float*` by default. Written in place.
- `num_data_` — `data_size_t` = `int32_t`, the number of training rows.
- `label_t` = `float` by default. Labels/weights are `const label_t*`.

#### The factory (`objective_function.cpp:20`)

`ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config)`
is a **stringly-typed factory**: a long `if/else` chain on the `type` string that
`new`s the matching class (`objective_function.cpp:68-102` is the CPU branch).
The mapping:

| `type` string | Class |
|---|---|
| `regression` | `RegressionL2loss` |
| `regression_l1` | `RegressionL1loss` |
| `quantile` | `RegressionQuantileloss` |
| `huber` | `RegressionHuberLoss` |
| `fair` | `RegressionFairLoss` |
| `poisson` | `RegressionPoissonLoss` |
| `mape` | `RegressionMAPELOSS` |
| `gamma` | `RegressionGammaLoss` |
| `tweedie` | `RegressionTweedieLoss` |
| `binary` | `BinaryLogloss` |
| `multiclass` | `MulticlassSoftmax` |
| `multiclassova` | `MulticlassOVA` |
| `lambdarank` | `LambdarankNDCG` |
| `rank_xendcg` | `RankXENDCG` |
| `cross_entropy` | `CrossEntropy` |
| `cross_entropy_lambda` | `CrossEntropyLambda` |
| `custom` | `nullptr` (user supplies gradients externally) |
| unknown | `Log::Fatal` (`objective_function.cpp:106`) |

There is a `USE_CUDA` branch (`:21-66`) selecting `CUDA*` mirror classes; it is
out of scope here. A second factory overload,
`CreateObjectiveFunction(const std::string& str)` (`:110`), rebuilds an objective
from its serialized `ToString()` form when loading a model, by splitting the
string on spaces and dispatching on the first token.

---

### 3. Each concrete family: math and implemented formulas

Convention below: `f` = score for a row, `y` = its label, `w` = its weight
(if present). Where two branches exist (weighted / unweighted), the weighted one
multiplies gradient and hessian by `w`.

#### Regression family (`regression_objective.hpp`)

All inherit from `RegressionL2loss`, which holds the shared `label_`, `weights_`,
`num_data_` members and the `sqrt_`/`deterministic_` flags (`:192-201`). `Init`
(`:113`) caches `metadata.label()` and `metadata.weights()`; with `reg_sqrt` it
replaces labels with `sign(y)·√|y|` (`:116-123`).

**L2 (squared error), `regression`** — loss `½(f − y)²`.
`GetGradients` (`:127`): `gradient = f − y`, `hessian = 1` (`:132-133`); weighted:
`gradient = (f−y)·w`, `hessian = w` (`:138-139`). `IsConstantHessian` returns true
when unweighted (`:165`). `BoostFromScore` (`:173`) = weighted mean of labels.
`ConvertOutput` is identity unless `sqrt_`, then `sign(f)·f²` (`:148-154`).

**L1 (absolute error), `regression_l1`** (`:207`) — loss `|f − y|`.
`gradient = sign(f − y)`, `hessian = 1` (`:222-224`). Because the true second
derivative of `|·|` is zero, the leaf value cannot come from grad/hess; so
`IsRenewTreeOutput` is true and `RenewTreeOutput` (`:253`) recomputes each leaf as
the (weighted) **median** of residuals via the `PercentileFun` /
`WeightedPercentileFun` macros (`:18-88`). `BoostFromScore` = median of labels
(alpha=0.5, `:236`).

**Huber, `huber`** (`:293`) — a blend: squared error when `|f−y| ≤ alpha`, linear
beyond. `gradient = (f−y)` inside the band, `sign(f−y)·alpha` outside; `hessian = 1`
(`:313-337`). `alpha_` from `config.alpha` (`:296`). `sqrt` transform is disabled.

**Fair, `fair`** (`:351`) — a smooth L1 approximation with constant `c`.
`gradient = c·x/(|x|+c)`, `hessian = c²/(|x|+c)²` where `x = f−y`
(`:368-369`). Hessian is not constant (`:385`).

**Poisson, `poisson`** (`:398`) — for count targets. The internal score `f` is a
log-rate; loss `= e^f − y·f`. `gradient = e^f − y`,
`hessian = e^f · e^{max_delta_step}` (`:446-448`); `max_delta_step` safeguards the
step. `Init` (`:413`) rejects negative labels and all-zero labels.
`ConvertOutput = exp(f)` (`:460`). `BoostFromScore = log(mean label)` (`:468`).

**Quantile, `quantile`** (`:481`) — predicts the `alpha` quantile (e.g. median at
0.5). `gradient = (1−alpha)` if `f ≥ y` else `−alpha`; `hessian = 1`
(`:498-504`). Like L1 it renews leaf outputs to the weighted `alpha`-percentile
(`:540`). Requires `0 < alpha < 1` (`:485`).

**MAPE, `mape`** (`:579`, extends L1) — mean absolute percentage error. Precomputes
`label_weight_[i] = 1/max(1,|y_i|)` in `Init` (`:599-610`) so
`gradient = sign(f−y)·label_weight_[i]` (`:619`); hessian 1 (unweighted).
Constant hessian (`:667`), renews leaves to a weighted median (`:643`).

**Gamma, `gamma`** (`:680`, extends Poisson) — for positive skewed targets, log-link.
With `exp_score = e^{−f}`: `gradient = 1 − y·e^{−f}`, `hessian = y·e^{−f}`
(`:695-697`). Inherits Poisson's `ConvertOutput = exp(f)`.

**Tweedie, `tweedie`** (`:717`, extends Poisson) — compound Poisson-Gamma with
variance power `rho` (`config.tweedie_variance_power`). With
`e1 = e^{(1−rho)f}`, `e2 = e^{(2−rho)f}`:
`gradient = −y·e1 + e2`, `hessian = −y·(1−rho)·e1 + (2−rho)·e2` (`:733-737`).

#### Binary logloss (`binary_objective.hpp`)

`class BinaryLogloss` (`:21`). **Logloss** = the negative log-likelihood of a
sigmoid classifier: it penalizes confident wrong probabilities heavily. Labels are
mapped to `{−1, +1}` internally (`label_val_`, `:87-88`) via the `is_pos_`
predicate (default `label > 0`, `:37`).

`GetGradients` (`:105`) computes, with sigmoid slope `sigmoid_` (config param):

```
response = −label · sigmoid_ / (1 + exp(label · sigmoid_ · f))   // :117
gradient = response · label_weight                                // :119
hessian  = |response| · (sigmoid_ − |response|) · label_weight    // :120
```

`label_weight` handles class imbalance: `is_unbalance` reweights the rarer class
(`:93-101`), and `scale_pos_weight` scales positives (`:102`). `Init` (`:59`)
counts positives/negatives (across machines if distributed) and disables training
if only one class is present (`need_train_`, `:80-84`).
`ConvertOutput = 1/(1+exp(−sigmoid_·f))` → probability (`:175`).
`BoostFromScore` = `log(pavg/(1−pavg))/sigmoid_`, the log-odds of the positive rate
(`:139-164`).

#### Multiclass (`multiclass_objective.hpp`)

**Softmax, `multiclass`** (`class MulticlassSoftmax`, `:24`). **Softmax** turns a
vector of `K` raw scores into a probability distribution (each in (0,1), summing to
1): `p_k = e^{f_k} / Σ_j e^{f_j}`. This objective trains `K` trees per round
(`NumModelPerIteration = num_class`, `:149`); the `score`, `gradients`, `hessians`
buffers are laid out class-major, so class `k`, row `i` lives at index
`num_data_·k + i` (`:93`).

`GetGradients` (`:86`) per row: gather the `K` scores, apply `Common::Softmax`
(`:96`), then for each class `k` (`:97-106`):

```
gradient[k] = p_k − 1   if k == true class,  else  p_k     // :101-104
hessian[k]  = factor_ · p_k · (1 − p_k)                    // :105
```

`factor_ = K/(K−1)` (`:31`) rescales the redundant K-class form to the
non-redundant form (Friedman GBDT paper). `Init` (`:53`) validates labels are
integers in `[0, K)` and computes per-class base probabilities.
`ConvertOutput` applies softmax (`:132`). `BoostFromScore = log(class_init_prob)`
(`:155`).

**One-vs-all, `multiclassova`** (`class MulticlassOVA`, `:186`). Holds `K`
independent `BinaryLogloss` objects (`binary_loss_`, `:191`), each with an
`is_pos_` predicate testing `label == i`. `GetGradients` (`:228`) just calls each
binary loss on its own slice of the buffers (offset `num_data_·i`). Outputs are
`K` independent sigmoids, **not** a normalized distribution (`:239-243`).

#### Ranking (`rank_objective.hpp`)

Ranking optimizes the *order* of documents within a **query group** (e.g. the set
of results for one search query), not each score in isolation. The abstract
`RankingObjective` (`:25`) reads query group boundaries from metadata
(`query_boundaries_`, and fatals if absent, `:51`), plus optional position data
for position-bias correction. Its `GetGradients` (`:59`) parallelizes over
queries, calling the subclass `GetGradientsForOneQuery` per group and multiplying
by weights afterward (`:75-82`).

**LambdaRank with NDCG, `lambdarank`** (`class LambdarankNDCG`, `:138`).
**NDCG** (Normalized Discounted Cumulative Gain) is a ranking quality metric that
rewards putting high-relevance documents near the top, with a positional discount.
**LambdaRank** does not differentiate NDCG directly (it is non-smooth); instead it
forms **pairwise** gradients ("lambdas"): for every pair of documents with
different labels it computes a **sigmoid pairwise** gradient — how much swapping
that pair would change NDCG, times a logistic term on their score difference.

In `GetGradientsForOneQuery` (`:180`): sort documents by score (`:196`); for each
pair `(i, j)` within the truncation level with different labels (`:208-213`):

```
delta_pair_NDCG = |gain_i − gain_j| · |discount_i − discount_j| · inverse_max_dcg  // :240
p_lambda  = sigmoid(delta_score)              // logistic on score gap, :246
p_hessian = p_lambda · (1 − p_lambda)         // :247
p_lambda  *= −sigmoid_ · delta_pair_NDCG      // :249
p_hessian *= sigmoid_² · delta_pair_NDCG      // :250
```

These are accumulated onto the high/low document (`:251-254`). `inverse_max_dcg`
(precomputed per query in `Init`, `:165-174`) normalizes by the best achievable
DCG. With `norm_` the lambdas are optionally rescaled by score distance (`:242`)
and by `log2(1+Σλ)/Σλ` (`:259-265`). `GetSigmoid` (`:268`) uses a precomputed
1M-entry lookup table (`ConstructSigmoidTable`, `:281`) for speed.
`UpdatePositionBiasFactors` (`:296`) does a Newton-Raphson update of learned
position-bias factors when position metadata is present.

**XE_NDCG, `rank_xendcg`** (`class RankXENDCG`, `:378`, arxiv 1911.09798). A
cross-entropy-style listwise ranking loss. `GetGradientsForOneQuery` (`:394`):
softmax the scores into `rho` (`:409`); build a randomized ground-truth
distribution via `Phi(label, uniform_rand) = 2^label − g` (`:448`); then compute
first-, second-, and third-order gradient terms (`:425-443`) and
`hessian = rho·(1−rho)` (`:444`). Per-query RNGs seeded from `objective_seed`
(`:389-391`) make it stochastic (hence a `mutable rands_`).

#### Cross-entropy (`xentropy_objective.hpp`)

For targets `y` anywhere in `[0,1]` (soft labels / probabilities), unlike binary
which expects 0/1.

**`cross_entropy`** (`class CrossEntropy`, `:44`). Probability `p = 1/(1+e^{−f})`;
loss `= −(1−y)log(1−p) − y·log(p)`. `Init` (`:55`) checks labels lie in `[0,1]`.
`GetGradients` (`:77`) uses a numerically stable, branch-prediction-friendly
formulation:

```
if f > −37:  exp_tmp = e^{−f};  gradient = ((1−y) − y·exp_tmp)/(1+exp_tmp);  hessian = exp_tmp/(1+exp_tmp)²
else:        exp_tmp = e^{f};   gradient = exp_tmp − y;                       hessian = exp_tmp
```

(`:103-111`). Mathematically `gradient = p − y`, `hessian = p(1−p)`.
`ConvertOutput = 1/(1+e^{−f})` (`:135`). `BoostFromScore = log(pavg/(1−pavg))`
(`:146`).

**`cross_entropy_lambda`** (`class CrossEntropyLambda`, `:185`). An alternative
parameterization where the boosted quantity feeds `lambda = log(1+e^f)` and the
per-point probability is `p = 1 − e^{−lambda·w}` (see file header comment,
`:29-36`). Unweighted it is identical to `CrossEntropy` (`:227-231`); weighted it
uses the more elaborate derivatives at `:236-248`.
`ConvertOutput = log1p(exp(f))` → the lambda, **not** a probability (`:266`; note
the warning at `:257-264`). `BoostFromScore = log(expm1(havg))` (`:276`). Requires
strictly positive weights (`:208`).

---

### 4. Inputs and outputs (concrete C++ types)

- **Entry point:** `void GetGradients(const double* score, score_t* gradients, score_t* hessians) const`
  (`objective_function.h:37`). `score` length = `num_data_` for single-output
  objectives, `num_data_ * num_class` (class-major) for multiclass.
- `score_t` = `float` (default), `label_t` = `float`, `data_size_t` = `int32_t`.
  Scores come in as `double` (accumulated in higher precision), gradients/hessians
  go out as `float`.
- **Labels/weights** are never passed to `GetGradients`; they are cached as
  `const label_t* label_` / `const label_t* weights_` during `Init` via
  `metadata.label()` and `metadata.weights()` (e.g. `regression_objective.hpp:115,124`;
  `binary_objective.hpp:61-62`).
- **Ranking** additionally caches `query_boundaries_` (`const data_size_t*`),
  `positions_`, `position_ids_`, and `num_queries_` from `Metadata`
  (`rank_objective.hpp:40-54`).
- **Output:** `gradients[i]` and `hessians[i]` written in place. `ConvertOutput`
  writes final prediction(s) into `output[]`.

---

### 5. Dependencies

- **`Dataset` / `Metadata`** — the sole data dependency. `Init(const Metadata&, data_size_t)`
  pulls labels, weights, and (for ranking) query boundaries / positions. The
  objective never reads binned feature data — only labels and the current scores.
- **`Config`** — constructor argument; supplies parameters like `sigmoid`, `alpha`,
  `fair_c`, `num_class`, `poisson_max_delta_step`, `tweedie_variance_power`,
  `lambdarank_truncation_level`, `objective_seed`, `deterministic`.
- **`Network`** — binary/multiclass sum class counts across machines in distributed
  training (`binary_objective.hpp:75-77`, `multiclass_objective.hpp:75-79`).
- **`Common` / `DCGCalculator` / `Random`** utilities — `Common::Softmax`,
  `Common::Sign`, `Common::SafeLog`, percentile macros, the DCG discount tables and
  `ArrayArgs` selection used by ranking and the L1/quantile leaf-renewal.
- **Called by `GBDT`** (`src/boosting/gbdt.cpp`): the boosting loop owns the
  objective and invokes it (see §6).

---

### 6. How it fits the training/prediction pipeline

**Training (each boosting round, inside `GBDT::TrainOneIter`):**

1. The GBDT already holds current scores for all rows (from previously added trees,
   initialized at `BoostFromScore`).
2. GBDT calls `objective->GetGradients(scores, gradients, hessians)` to fill the
   per-row gradient/hessian arrays.
3. GBDT hands those arrays to the `TreeLearner`, which builds gradient/hessian
   **histograms** and finds the splits that best reduce the loss, producing one
   tree (or `NumModelPerIteration` trees for multiclass).
4. For objectives with `IsRenewTreeOutput()` true (L1, quantile, MAPE), GBDT calls
   `RenewTreeOutput` to replace each leaf's value with the correct estimator
   (e.g. median) once the tree structure is fixed.
5. Scores are updated by adding the new tree's outputs, and the loop repeats.

`IsConstantHessian` lets the learner skip hessian storage when all hessians are
identical (e.g. unweighted L2), a memory/speed optimization.

**Prediction:** the model sums the raw tree outputs into `input`, then
`objective->ConvertOutput(input, output)` maps that raw score to the user-facing
value — identity for regression, `sigmoid` for binary, `softmax` for multiclass,
`exp` for Poisson/Gamma, `log1p(exp())` for cross_entropy_lambda. The objective is
serialized into the model via `ToString()` so the correct `ConvertOutput` is
restored on load (`CreateObjectiveFunction(const std::string&)`).


---

## Section G — Metrics (Model Evaluation)

All paths below are relative to `LightGBM/`.

### 1. Purpose in Plain Language

A **metric** measures how good the model currently is on a dataset. After every boosting iteration (every tree added to the ensemble), LightGBM asks each configured metric to score the model's predictions against the true labels. Those scores are used for two things:

1. **Logging** — printing a progress line such as `[LightGBM] [Info] ... valid_0's l2: 0.0432`.
2. **Early stopping** — a training-control heuristic: if the metric on a validation set stops improving for `early_stopping_round` iterations in a row, training halts to avoid wasting time and overfitting.

A metric is **distinct from the objective**. The **objective function** (a separate subsystem, `src/objective/`) drives training: it produces the gradients and hessians that tell each new tree which direction to move. The metric only *judges* the result; it never feeds back into how trees are built (except by triggering early stopping).

- Some metrics **coincide** with an objective. For example, `l2` (mean squared error) is both a training objective and a reporting metric; `binary_logloss` likewise.
- Some metrics have **no matching objective** and exist purely for evaluation: **AUC** (Area Under the ROC Curve), **NDCG** (Normalized Discounted Cumulative Gain), **MAP** (Mean Average Precision), `average_precision`, `auc_mu`. These measure *ranking quality* or *classification ranking* in ways that are not differentiable, so they can be evaluated but not directly optimized by gradient boosting.

Jargon glossary (all defined in detail in their family sections below):
- **logloss** (logarithmic loss / cross-entropy): penalizes confident wrong probability predictions.
- **AUC**: probability that a random positive example is scored above a random negative one; ranking quality for binary classification.
- **DCG / NDCG**: ranking-quality scores that reward putting high-relevance items near the top of a ranked list.
- **MAP**: Mean Average Precision, a ranking metric for binary relevance.
- **higher-is-better vs lower-is-better**: losses (l2, logloss) improve as they *shrink*; ranking/AUC scores improve as they *grow*. The `factor_to_bigger_better()` method encodes which direction is "better" so the early-stopping code can compare iterations uniformly.

### 2. The Abstract Base `Metric` Interface

Defined in `include/LightGBM/metric.h:24-63`. Every concrete metric derives from this pure-virtual base:

```cpp
class Metric {
 public:
  virtual ~Metric() {}

  // One-time setup: capture label/weight pointers and per-metric name(s).
  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;   // :35

  // The metric's display name(s). A vector because one metric object can emit
  // several columns (e.g. ndcg@1, ndcg@3, ndcg@5).
  virtual const std::vector<std::string>& GetName() const = 0;             // :37

  // +1.0 => higher score is better (AUC, NDCG); -1.0 => lower is better (losses).
  virtual double factor_to_bigger_better() const = 0;                      // :39

  // The workhorse: score `score` against stored labels, return one value per name.
  virtual std::vector<double> Eval(const double* score,
                                   const ObjectiveFunction* objective) const = 0;  // :44

  // Factory (see below).
  static Metric* CreateMetric(const std::string& type, const Config& config);      // :57

  virtual bool IsCUDAMetric() const { return false; }                      // :62
};
```

Copy construction/assignment is explicitly deleted (`metric.h:48-50`) — metric objects are heap-allocated and owned via raw/`unique_ptr` by the boosting layer.

Note the parallel structure between `GetName()` (returns N names) and `Eval()` (returns N doubles): a single metric instance can report multiple related numbers. Ranking metrics use this to report at several cutoff positions in one pass.

#### Factory: `Metric::CreateMetric`

Implemented in `src/metric/metric.cpp:19-133`. This is a **stringly-typed factory**: it switches on the metric-name string and `new`s the matching class, returning `nullptr` for an unknown type (`metric.cpp:132`). The CPU branch (`metric.cpp:82-128`) maps:

| `type` string | Class | Family |
|---|---|---|
| `l2` | `L2Metric` | regression |
| `rmse` | `RMSEMetric` | regression |
| `l1` | `L1Metric` | regression |
| `quantile` | `QuantileMetric` | regression |
| `huber` | `HuberLossMetric` | regression |
| `fair` | `FairLossMetric` | regression |
| `poisson` | `PoissonMetric` | regression |
| `mape` | `MAPEMetric` | regression |
| `gamma` | `GammaMetric` | regression |
| `gamma_deviance` | `GammaDevianceMetric` | regression |
| `tweedie` | `TweedieMetric` | regression |
| `binary_logloss` | `BinaryLoglossMetric` | binary |
| `binary_error` | `BinaryErrorMetric` | binary |
| `auc` | `AUCMetric` | binary |
| `average_precision` | `AveragePrecisionMetric` | binary |
| `auc_mu` | `AucMuMetric` | multiclass |
| `multi_logloss` | `MultiSoftmaxLoglossMetric` | multiclass |
| `multi_error` | `MultiErrorMetric` | multiclass |
| `ndcg` | `NDCGMetric` | ranking |
| `map` | `MapMetric` | ranking |
| `cross_entropy` | `CrossEntropyMetric` | xentropy |
| `cross_entropy_lambda` | `CrossEntropyLambdaMetric` | xentropy |
| `kullback_leibler` | `KullbackLeiblerDivergence` | xentropy |

(There is a parallel `#ifdef USE_CUDA` branch at `metric.cpp:20-79` that swaps in CUDA implementations for a subset; per instructions, GPU/CUDA code is out of scope.)

### 3. Metric Families

Most families use a shared **template + point-wise-loss** pattern: a template base class does the per-row loop, weight handling, and averaging, while a small "loss calculator" class plugged in as the template parameter supplies just three static functions — `Name()`, `LossOnPoint(...)`, and (optionally) `AverageLoss(...)`. This is the Curiously Recurring Template Pattern (CRTP): e.g. `class L2Metric : public RegressionMetric<L2Metric>`.

#### 3a. Regression — `src/metric/regression_metric.hpp`

Base template `RegressionMetric<PointWiseLossCalculator>` (`regression_metric.hpp:21-116`):
- `factor_to_bigger_better()` returns `-1.0` (`:34-36`) — all regression losses are lower-is-better.
- `Init` (`:38-56`) grabs `label_`/`weights_` from `Metadata`, precomputes `sum_weights_` (= `num_data` when unweighted), and runs `CheckLabel` on every row.
- `Eval` (`:58-95`) sums `LossOnPoint(label, score)` over all rows (weighted if weights present), then divides by `sum_weights_` via `AverageLoss`. If an `objective` is supplied, each raw score is first passed through `objective->ConvertOutput` (e.g. exp-link for Poisson) before the loss is computed (`:79-81`).

Per-metric formulas (loss of one point; `y` = label, `ŷ` = score):

| Metric | `LossOnPoint` | Notes / line |
|---|---|---|
| `l2` | `(ŷ − y)²` | mean squared error; `:142-144` |
| `rmse` | `(ŷ − y)²`, then `AverageLoss = sqrt(sum/Σw)` | root MSE; `:123-130` |
| `l1` | `|ŷ − y|` | mean absolute error; `:177-179` |
| `quantile` | `α·δ` if `δ≥0` else `(α−1)·δ`, where `δ = y − ŷ` | pinball loss at quantile `α` (`config.alpha`); `:157-164` |
| `huber` | `0.5·d²` if `|d|≤α` else `α·(|d|−0.5α)`, `d = ŷ−y` | quadratic near 0, linear in tails; `:191-198` |
| `fair` | `c·x − c²·log1p(x/c)`, `x=|ŷ−y|`, `c=fair_c` | smooth robust loss; `:212-215` |
| `poisson` | `ŷ − y·log(ŷ)` (ŷ floored at 1e-10) | Poisson deviance-ish; `:229-235` |
| `mape` | `|y − ŷ| / max(1, |y|)` | mean absolute percentage error; `:248-250` |
| `gamma` | negative log-likelihood of Gamma (`:261-268`); requires `label>0` (`CheckLabel`, `:273-275`) |
| `gamma_deviance` | `t − log(t) − 1`, `t=y/(ŷ+ε)`; `AverageLoss` multiplies by 2 (`:292-294`); requires `label>0` |
| `tweedie` | Tweedie deviance with variance power `ρ=tweedie_variance_power`; `:305-313` |

#### 3b. Binary classification — `src/metric/binary_metric.hpp`

Two sub-patterns here.

**Point-wise losses** via `BinaryMetric<Calc>` (`binary_metric.hpp:23-110`), same shape as regression (lower-is-better, `:56-58`). When an objective is present, `ConvertOutput` turns the raw margin into a probability `prob` before loss (`:80-83`).
- `binary_logloss` (`BinaryLoglossMetric`, `:115-135`): `−log(prob)` if label positive, `−log(1−prob)` if negative, clipped by `kEpsilon` to avoid `log(0)`. This is the standard binary cross-entropy.
- `binary_error` (`BinaryErrorMetric`, `:139-154`): 0/1 misclassification — counts a point as wrong when `prob≤0.5` but label is positive, or `prob>0.5` but label is non-positive. Reports the error *rate* after averaging.

**AUC** — `AUCMetric` (`:159-264`), a standalone `Metric` (not templated). **AUC** = Area Under the ROC Curve = the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative one. `factor_to_bigger_better()` returns `+1.0` (`:171-173`) — higher is better. `Eval` (`:194-251`):
1. Sort all row indices by descending score (`Common::ParallelSort`, `:200`).
2. Sweep in score order, accumulating, at each distinct score threshold, `cur_neg × (0.5·cur_pos + sum_pos)` — i.e. for every negative, count how many positives outrank it (ties count as half) (`:218`, `:244`).
3. Normalize by `sum_pos × (total_weight − sum_pos)` = (#positives × #negatives) (`:246-249`). Weighted variant multiplies counts by per-row weight. AUC does not use the `objective` argument at all (it is rank-based, invariant to monotone transforms of the score).

**Average Precision** — `AveragePrecisionMetric` (`:270-385`), also standalone, `+1.0` bigger-better (`:282-284`). It computes the area under the precision-recall curve by the same descending-score sweep, accumulating `cur_actual_pos × running_precision` and normalizing by total positives (`:305-372`).

#### 3c. Multiclass — `src/metric/multiclass_metric.hpp`

Here each row has `num_class` raw scores laid out **column-major**: the score for class `k` at row `i` is `score[num_data * k + i]` (`multiclass_metric.hpp:70`, `:98`). Base template `MulticlassMetric<Calc>` (`:21-135`) gathers the `num_class` values per row into a vector, optionally runs them through `objective->ConvertOutput` (softmax → probabilities), and feeds `LossOnPoint(label, &probs, config)`.

- `multi_logloss` (`MultiSoftmaxLoglossMetric`, `:163-180`): `−log(prob[true_class])`, clipped by `kEpsilon`. Lower-is-better (`:52-54`).
- `multi_error` (`MultiErrorMetric`, `:138-160`): top-k error. A row is wrong (contributes 1.0) if more than `multi_error_top_k` classes have score ≥ the true class's score (`:142-149`); with the default `k=1` this is ordinary "did the argmax match the label" error. Name becomes `multi_error@k` when `k≠1` (`:153-159`).
- `auc_mu` (`AucMuMetric`, `:183-365`): a multiclass generalization of AUC (Kleiman & Page 2019). Standalone, `+1.0` bigger-better (`:194`). `Init` (`:196-236`) sorts rows by true class and records per-class sizes/weights. `Eval` (`:238-340`) loops over every ordered class pair `(i,j)`, projects each row's score vector onto the class-difference weight vector `curr_v` (built from the `auc_mu_weights_matrix`, `:249`), sorts by distance to the separating hyperplane, computes a pairwise AUC `S[i][j]` (ties → 0.5), and averages over all `num_class·(num_class−1)/2` pairs (`:328-338`).

#### 3d. Ranking — `src/metric/rank_metric.hpp` and `src/metric/map_metric.hpp`

Ranking metrics operate **per query group**. In learning-to-rank, rows are grouped into "queries" (e.g. all documents shown for one search query); `Metadata::query_boundaries()` gives the `[start,end)` row range of each query, and quality is measured *within* each query then averaged. Both metrics are `+1.0` bigger-better.

**NDCG** — `NDCGMetric` (`rank_metric.hpp:19-165`). **DCG** (Discounted Cumulative Gain) scores a ranked list by summing each item's *gain* (a function of its relevance label) times a positional *discount* (later positions count less). **NDCG** (Normalized DCG) divides a query's DCG by that query's *maximum possible* DCG (the ideal ordering), giving a 0–1 score comparable across queries.
- Constructor (`:21-29`) initializes the shared `DCGCalculator` tables from `config.eval_at` (the cutoff positions, e.g. 1/3/5) and `config.label_gain`.
- `Init` (`:33-76`) emits one name per cutoff (`ndcg@k`, `:34-35`), reads labels + query boundaries + optional query weights, validates them (`DCGCalculator::CheckMetadata`/`CheckLabel`), and **precomputes the inverse max-DCG** for every query at every cutoff (`:58-75`) so `Eval` only needs the actual DCG. A query whose ideal DCG is 0 (all-zero labels) is flagged with `-1` and later scored as NDCG=1 (`:70-73`, `:99-102`).
- `Eval` (`:86-144`) computes actual DCG per query via `DCGCalculator::CalDCG`, multiplies by the cached inverse-max, sums across queries (with per-thread buffers), and divides by total query weight. Returns one value per cutoff.

**MAP** — `MapMetric` (`map_metric.hpp:20-162`). **MAP** = Mean Average Precision: for each query, walk down the score-sorted list and average the precision measured at each position where a *relevant* (positive-label) item appears; then average across queries.
- `Init` (`:31-64`) emits `map@k` names, reads query boundaries/weights, and precounts positives per query `npos_per_query_` (`:56-63`; a doc is "positive" when `label > 0.5`).
- `CalMapAtK` (`:74-104`) stable-sorts a query's docs by descending score, and for each cutoff `k` accumulates `num_hit / rank` at every relevant hit, normalizing by `min(npos, k)` (`:97-102`).
- `Eval` (`:105-143`) runs `CalMapAtK` per query (guided-schedule parallel loop) and averages by query weight. Returns one value per cutoff.

**The static `DCGCalculator`** — declared in `include/LightGBM/metric.h:68-139`, implemented in `src/metric/dcg_calculator.cpp`. It is a "static class": all state lives in **static (process-global) tables** shared by all NDCG instances:
- `label_gain_` (`dcg_calculator.cpp:15`) — the gain assigned to each integer relevance label.
- `discount_` (`dcg_calculator.cpp:16`) — the positional discount table.
- `kMaxPosition = 10000` (`dcg_calculator.cpp:17`) — the largest supported query size / discount table length.

Key methods:
- `DefaultEvalAt` (`:20-31`): if the user gave no cutoffs, default to positions 1..5.
- `DefaultLabelGain` (`:33-41`): default gain for label `i` is `2^i − 1` (capped at 31 labels to avoid overflow).
- `Init` (`:43-52`): copies the label-gain table and **precomputes the discount table** `discount_[i] = 1 / log2(2 + i)` for every position up to `kMaxPosition` — the classic DCG log-discount, computed once so `CalDCG` is just table lookups.
- `CalDCG` (`:109-132`): sort docs by score, then sum `label_gain_[label] · discount_[position]` up to each cutoff.
- `CalMaxDCG` / `CalMaxDCGAtK` (`:54-107`): the *ideal* DCG — greedily place the highest-gain labels first, giving the normalizer.
- `CheckMetadata` (`:134-144`): fails if any query has more than `kMaxPosition` rows.
- `CheckLabel` (`:147-163`): ranking labels must be non-negative integers within the label-gain table.
- `GetDiscount(k)` (`metric.h:130`): inline accessor into the precomputed table.

#### 3e. Cross-entropy family — `src/metric/xentropy_metric.hpp`

For continuous labels in `[0, 1]` (probabilistic targets). All three are lower-is-better and share a helper `XentLoss(label, prob) = −[y·log(p) + (1−y)·log(1−p)]` with epsilon clipping (`xentropy_metric.hpp:35-50`).
- `cross_entropy` (`CrossEntropyMetric`, `:71-160`): mean weighted `XentLoss`. `Init` (`:76-104`) validates labels lie in `[0,1]` and weights are non-negative with positive sum.
- `cross_entropy_lambda` (`CrossEntropyLambdaMetric`, `:166-244`): a reparameterized variant using `XentLambdaLoss` where the weight enters through `1 − exp(−weight·ĥ)` (`:53-55`); intended to pair with the `xentlambda` objective. Averages by `num_data` (not weight sum) (`:224`).
- `kullback_leibler` (`KullbackLeiblerDivergence`, `:249-354`): cross-entropy plus a precomputed constant offset — the (negative) entropy of the labels `YentLoss` (`:60-66`, presummed in `Init` at `:281-292`) — so the reported value is the KL divergence between labels and predictions rather than raw cross-entropy.

### 4. Inputs and Outputs (Concrete C++ Types)

- **Setup:** `void Init(const Metadata& metadata, data_size_t num_data)`. `data_size_t` is `int32_t` (row count). `Metadata` (from `include/LightGBM/dataset.h`) supplies:
  - `const label_t* label()` — true labels; `label_t` is `float` by default.
  - `const label_t* weights()` — optional per-row weights (`nullptr` if none).
  - `const data_size_t* query_boundaries()` and `data_size_t num_queries()` — for ranking.
  - `const label_t* query_weights()` — optional per-query weights.
  Metrics store these as raw non-owning pointers; the `Dataset`/`Metadata` outlives the metric.
- **Evaluation:** `std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const`.
  - `score` — the current model output, a raw `double*`. For single-output tasks it is one value per row; for multiclass it is column-major `num_class × num_data`.
  - `objective` — may be `nullptr`. When non-null, the metric calls `objective->ConvertOutput(raw, converted)` to turn raw margins into the space the loss expects (probabilities for logloss, `exp(x)` for Poisson, softmax for multiclass). Rank/AUC metrics ignore it.
  - **Return type is always `std::vector<double>`**, length = `GetName().size()`. Point-wise metrics return a 1-element vector (`std::vector<double>(1, loss)`); ranking metrics return one entry per cutoff position.
- **Direction:** `double factor_to_bigger_better()` — `+1.0` for AUC/NDCG/MAP/average_precision/auc_mu; `-1.0` for all losses.

Internally, all accumulation is done in `double` even though labels/scores are `float`, and per-row loops use OpenMP with `reduction(+:sum_loss)` (e.g. `regression_metric.hpp:62`) for deterministic-ish parallel summation.

### 5. Dependencies

- **Reads from `Dataset` `Metadata`** — labels, weights, query boundaries/weights. It never modifies the dataset (the dataset is immutable after `FinishLoad`).
- **Uses the `ObjectiveFunction`** only through `ConvertOutput` / `NumModelPerIteration` / `NumPredictOneRow` (`multiclass_metric.hpp:61-62`) — a read-only collaboration to interpret raw scores.
- **`Config`** — supplies metric parameters (`alpha`, `fair_c`, `tweedie_variance_power`, `eval_at`, `label_gain`, `multi_error_top_k`, `num_class`, `auc_mu_weights_matrix`).
- **`Common` utilities** — `ParallelSort`, `SafeLog`, `CheckElementsIntervalClosed`, `ObtainMinMaxSum`.
- **Consumed by the boosting layer (`GBDT`)** — `GBDT` owns the training-set and each validation-set metric list, calls `Eval` each iteration, prints the results, and passes them to the early-stopping logic. `factor_to_bigger_better()` is what lets that logic decide "did this iteration improve?" regardless of whether the metric is a loss or a score.

### 6. How It Fits the Pipeline

Training loop (in `GBDT`, `src/boosting/gbdt.cpp`), each iteration:

1. Objective computes gradients/hessians from the current scores.
2. The tree learner grows one tree; scores are updated.
3. **Metrics evaluate**: for the training set and each validation set, `Metric::Eval(current_scores, objective)` is called, producing `std::vector<double>` results.
4. Results are logged (`[LightGBM] [Info] ... valid_0's <name>: <value>`).
5. Early stopping inspects the validation metric: using `factor_to_bigger_better()` to normalize direction, if the metric has not improved for `early_stopping_round` consecutive iterations, training stops and the best iteration is retained.

So metrics are a **read-only observer** at the end of each boosting step — they judge model quality for humans and for the stopping rule, but (unlike the objective) never alter how the next tree is grown.


---

## Section H — Distributed Network / Collective Communication

### 1. Purpose in plain language

LightGBM can train a single model across several machines at once. When it does, each
machine ("worker") owns only a slice of the data, but every worker still needs the *global*
picture — for example, the sum of gradient/hessian histograms computed over *all* the data,
not just the local slice. This subsystem is the plumbing that lets the workers exchange and
combine those partial results.

It provides three classic **collective communication operations** — operations where every
worker participates and every worker ends up with a coordinated result:

- **Allreduce** — every worker starts with an array of numbers; the operation combines
  (e.g. sums) the arrays element-by-element across all workers, and hands the *same* combined
  array back to *every* worker. This is the workhorse of distributed histogram merging.
- **Allgather** — every worker contributes a block of bytes; afterwards every worker holds the
  *concatenation* of all workers' blocks (no combining, just collecting).
- **ReduceScatter** — combine (reduce) all workers' arrays element-by-element, but instead of
  giving the whole combined array to everyone, *split* (scatter) it so each worker keeps only
  its assigned chunk. Allreduce is literally implemented as ReduceScatter followed by Allgather.

These operations are built on top of one of two interchangeable low-level transports:

- **MPI** (Message Passing Interface) — an industry-standard library/runtime for HPC clusters
  that handles process launch, ranks, and message passing for you.
- **raw TCP sockets** — LightGBM's own hand-rolled transport when MPI is not available; it opens
  TCP connections between every pair of machines itself.

Jargon primer (used throughout):
- **worker / rank** — one participating machine/process. Its **rank** is its integer id,
  `0 .. num_machines-1`. `rank 0` is not special here; there is no central coordinator.
- **communication topology** — the fixed pattern of "who talks to whom, and in what order"
  used to complete a collective operation in as few rounds as possible.
- **Bruck algorithm** — an allgather schedule that finishes in O(log n) rounds by having each
  worker exchange doubling-sized chunks with a partner `2^i` ranks away, then locally rotating
  the result into place.
- **recursive halving** — a reduce-scatter schedule that also finishes in O(log n) rounds: at
  each step a worker exchanges half of the still-unreduced range with a partner and reduces the
  half it received.

### 2. Key classes and functions

#### `Network` — the static collective-operations class
Declared in `include/LightGBM/network.h:89`, implemented in `src/network/network.cpp`.
It is a **static/singleton** class (all members `static THREAD_LOCAL`, defined at
`network.cpp:17-27`) — there is one network context per thread, no instances.

Lifecycle:
- `static void Init(Config config)` — `network.h:95`, `network.cpp:30`. Builds a `Linkers`
  object (opens connections), copies out `rank_`, `num_machines_`, the two topology maps, and
  allocates a 1 MB scratch `buffer_`. **No-op when `config.num_machines <= 1`** (single machine).
- `static void Init(int num_machines, int rank, ReduceScatterFunction, AllgatherFunction)` —
  `network.h:99`, `network.cpp:45`. Alternate init that installs *external* function pointers
  for reduce-scatter/allgather instead of using the built-in socket/MPI implementations (used
  when an embedding application supplies its own transport, e.g. the bindings). Also no-op when
  `num_machines <= 1`.
- `static void Dispose()` — `network.h:101`, `network.cpp:60`. Resets to single-machine state
  (`num_machines_ = 1`, `rank_ = 0`), tears down linkers, clears the external function pointers.
- `static int rank()` / `static int num_machines()` — `network.cpp:320`/`324`.

Collective operations:
- `static void Allreduce(char* input, comm_size_t input_size, int type_size, char* output, const ReduceFunction& reducer)`
  — `network.h:116`, `network.cpp:68`. **Adaptive dispatch**: if the payload is small
  (`count < num_machines_ || input_size < 4096`) it routes to `AllreduceByAllGather`; otherwise
  it partitions the array into per-rank blocks and does `ReduceScatter` then `Allgather`
  (`network.cpp:74-92`).
- `static void AllreduceByAllGather(...)` — `network.h:127`, `network.cpp:95`. Small-payload
  path: allgather everyone's input into `buffer_`, then locally fold all blocks with `reducer`
  (`network.cpp:114-116`). Chosen to save round-trips when data is tiny.
- `static void Allgather(char* input, comm_size_t send_size, char* output)` — `network.h:138`,
  `network.cpp:121`. Equal-size variant: every worker sends exactly `send_size` bytes.
- `static void Allgather(char* input, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t all_size)`
  — `network.h:150`, `network.cpp:137`. Variable-size variant (blocks differ per rank).
  **Dispatch** (`network.cpp:141-153`): if an external allgather fn is installed, call it; else
  pick `AllgatherRing` (data > 10 MB and < 64 machines), `AllgatherRecursiveDoubling`
  (machine count is a power of 2), or `AllgatherBruck` (fallback).
- `static void ReduceScatter(char* input, comm_size_t input_size, int type_size, const comm_size_t* block_start, const comm_size_t* block_len, char* output, comm_size_t output_size, const ReduceFunction& reducer)`
  — `network.h:164`, `network.cpp:232`. **Dispatch** (`network.cpp:238-246`): external fn if
  installed; else `ReduceScatterRecursiveHalving` (power-of-2 machines or < 10 MB) or
  `ReduceScatterRing`.

Private algorithm implementations (`network.h:278-290`):
`AllgatherBruck` (`network.cpp:156`), `AllgatherRecursiveDoubling` (`network.cpp:188`),
`AllgatherRing` (`network.cpp:216`), `ReduceScatterRecursiveHalving` (`network.cpp:249`),
`ReduceScatterRing` (`network.cpp:303`). All of them drive the actual byte movement through
`linkers_->SendRecv(...)` / `Send` / `Recv`.

Convenience templated helpers (header-only, `network.h:168-275`) wrap `Allreduce`/`Allgather`
with a ready-made `reducer` lambda so callers can sync scalars/vectors without writing the
byte-fiddling reduce function themselves:
- `T GlobalSyncUpByMin<T>(T)` / `GlobalSyncUpByMax<T>` / `GlobalSyncUpBySum<T>` /
  `GlobalSyncUpByMean<T>` — `network.h:169/192/216/238`.
- `std::vector<T> GlobalSum<std::vector<T>*>` — `network.h:243`.
- `std::vector<T> GlobalArray<T>(T local)` — `network.h:265` (each rank contributes one value;
  returns the length-`num_machines` array of everyone's value, via `Allgather`).

#### `Linkers` — the transport wrapper (topology + send/recv)
Declared in `src/network/linkers.h:37`. It "wraps low level communication methods, e.g. mpi,
socket" (`linkers.h:32-34`). It owns the connections and exposes uniform blocking primitives:
- `void Recv(int rank, char* data, int len)` / `Recv(int rank, char* data, int64_t len)` —
  `linkers.h:57/59`. Block until `len` bytes arrive from `rank`.
- `void Send(int rank, char* data, int len)` / `int64_t` overload — `linkers.h:67/69`.
- `void SendRecv(int send_rank, char* send_data, int send_len, int recv_rank, char* recv_data, int recv_len)`
  — `linkers.h:79`. Send and receive simultaneously (the core primitive every collective uses;
  it overlaps a send to one partner with a receive from another to hide latency).
- Accessors `rank()`, `num_machines()`, `bruck_map()`, `recursive_halving_map()`
  (`linkers.h:87-99`).

The 64-bit `Send`/`Recv`/`SendRecv` overloads (`linkers.h:207-238`) chunk transfers larger than
`INT32_MAX` into `int`-sized pieces. `SendRecv` runs the send on a separate `std::thread` while
receiving on the calling thread, then joins (`linkers.h:225-238`) — needed because a large
blocking send can deadlock against the peer's blocking send.

There are **two mutually exclusive implementations** of the same `Linkers` class, selected at
compile time (see §4):
- socket build: `Recv/Send/SendRecv` at `linkers.h:242-281` (loop over `TcpSocket::Recv/Send`).
- MPI build: `Recv/Send/SendRecv` at `linkers.h:287-324` (`MPI_Recv`, `MPI_Isend`+`MPI_Wait`,
  all `MPI_BYTE` over `MPI_COMM_WORLD`).

#### `BruckMap` — precomputed allgather topology
`network.h:22`, built by `BruckMap::Construct(int rank, int num_machines)`
(`network.h:38`, `linker_topo.cpp:29`). Holds `int k` (number of communication rounds =
`ceil(log2(n))`) and, for each round `j`, `in_ranks[j] = (rank + 2^j) % n` and
`out_ranks[j] = (rank - 2^j + n) % n` (`linker_topo.cpp:37-45`). These are the partner ranks
for the doubling-exchange schedule that `AllgatherBruck` walks.

#### `RecursiveHalvingMap` — precomputed reduce-scatter topology
`network.h:56`, built by `RecursiveHalvingMap::Construct(int rank, int num_machines)`
(`network.h:85`, `linker_topo.cpp:68`). Fields: `int k` (rounds), `RecursiveHalvingNodeType type`,
`bool is_power_of_2`, `int neighbor`, and per-round vectors `ranks`, `send_block_start`,
`send_block_len`, `recv_block_start`, `recv_block_len` (`network.h:59-73`).

Recursive halving assumes the machine count is a power of 2. When it is not, the algorithm groups
the "extra" machines into pairs so the surviving group count *is* a power of 2
(`linker_topo.cpp:99-176`). Each machine is tagged (`enum RecursiveHalvingNodeType`,
`network.h:49-53`):
- `Normal` — a group of one machine, participates directly.
- `GroupLeader` — the machine that represents a 2-machine group in the main halving rounds
  (it first absorbs its partner's data, then later hands the result back).
- `Other` — the non-leader in a 2-machine group; it just ships its data to its `neighbor`
  leader and receives the final chunk back (see the pre/post steps in
  `ReduceScatterRecursiveHalving`, `network.cpp:252-298`).

#### Function-pointer typedefs (`include/LightGBM/meta.h`)
These define the *signatures* the collectives are parameterized over:
- `typedef int32_t comm_size_t;` — `meta.h:59`. Signed 32-bit; the type of all
  sizes/offsets/counts in this subsystem.
- `typedef void(*ReduceFunction)(const char* input, char* output, int type_size, comm_size_t array_size);`
  — `meta.h:67`. The element-wise combine step (e.g. sum). Given two byte buffers it folds
  `input` into `output` in place, `type_size` bytes per element, over `array_size` bytes total.
  The `GlobalSyncUp*` helpers pass in lambdas implementing min/max/sum (`network.h:174-235`).
- `typedef void(*ReduceScatterFunction)(char* input, comm_size_t input_size, int type_size, const comm_size_t* block_start, const comm_size_t* block_len, int num_block, char* output, comm_size_t output_size, const ReduceFunction& reducer);`
  — `meta.h:70`. Signature for an *external* reduce-scatter plugged in via `Network::Init`.
- `typedef void(*AllgatherFunction)(char* input, comm_size_t input_size, const comm_size_t* block_start, const comm_size_t* block_len, int num_block, char* output, comm_size_t output_size);`
  — `meta.h:74`. Signature for an *external* allgather.

#### Socket backend specifics
`TcpSocket` (`socket_wrapper.hpp:94`) is a thin RAII wrapper over a BSD/Winsock socket
(`Bind`/`Connect`/`Listen`/`Accept`/`Send`/`Recv`/`Close`, `socket_wrapper.hpp:242-314`), with
cross-platform IP enumeration (`GetLocalIpList`, `socket_wrapper.hpp:169-232`) and tuning
constants (`SocketConfig::kSocketBufferSize = 100000`, `kMaxReceiveSize`, `kNoDelay`,
`socket_wrapper.hpp:88-92`). The socket-build `Linkers` constructor (`linkers_socket.cpp:24`)
parses the machine list (`ParseMachineList`, `linkers_socket.cpp:81` — file or comma-separated
`ip port` / `ip:port` lines, optional `rank=` line), discovers its own rank by matching local
IPs+port, binds a listener, builds the two topology maps, then `Construct()`
(`linkers_socket.cpp:169`) opens a full mesh: it listens on a background thread for
lower-ranked peers while actively connecting to higher-ranked peers (with retry/backoff),
so exactly one connection is formed per pair.

#### MPI backend specifics
The MPI-build `Linkers` constructor (`linkers_mpi.cpp:11`) calls `MPI_Init_thread`
(if not already initialized), reads `num_machines_`/`rank_` from `MPI_Comm_size`/`MPI_Comm_rank`,
barriers, and builds the topology maps. The destructor deliberately does **not** call
`MPI_Finalize` (`linkers_mpi.cpp:29-32`) — a single-node exception must not hang the others;
finalize/abort are handled in `main()` via the statics `IsMpiInitialized`,
`MpiFinalizeIfIsParallel`, `MpiAbortIfIsParallel` (`linkers.h:146-156`, `linkers_mpi.cpp:34-58`).

### 3. Inputs and outputs (concrete C++ types)

Everything is byte-buffer oriented so a single implementation works for any element type:
- Payloads are always `char*` (`input`, `output`) — raw byte pointers.
- Sizes/offsets are `comm_size_t` (= `int32_t`); block descriptors are
  `const comm_size_t* block_start` / `const comm_size_t* block_len` (one entry per rank).
- `int type_size` is the size of one logical element in bytes — the `ReduceFunction` steps
  over the buffer `type_size` bytes at a time so it knows element boundaries.
- The combine step is a `const ReduceFunction&` (the `meta.h:67` function pointer).
- Example concrete call (`GlobalSyncUpBySum`, `network.h:218`):
  `Allreduce((char*)&local, sizeof(local), sizeof(local), (char*)&global, sum_lambda)`.
  Output `global` receives the all-worker sum; `input_size == type_size == sizeof(local)`.
- Rank/machine identifiers are plain `int`.

### 4. Dependencies and backend selection

**Who uses it (upstream):** the *parallel* tree learners — `DataParallelTreeLearner`,
`FeatureParallelTreeLearner`, `VotingParallelTreeLearner` (in `src/treelearner/`) — call
`Network::Allreduce` / `Allgather` / `ReduceScatter` (and the `GlobalSyncUp*`/`GlobalSum`
helpers) to merge per-worker histograms and split statistics into global ones. Higher up,
`GBDT`/boosting and the CLI `Application` call `Network::Init` at startup and `Network::Dispose`
at teardown. `Config` (`num_machines`, `local_listen_port`, `machines`,
`machine_list_filename`, `time_out`) feeds the socket backend.

**What it depends on (downstream):** either MPI (`<mpi.h>`) or the OS socket API
(`socket_wrapper.hpp`), plus `Config`, `meta.h` typedefs, and `utils` (logging, `TextReader`,
`Common::Split/Trim/Atoi`).

**Backend chosen at compile time.** The transport is a preprocessor switch, not runtime:
- `USE_SOCKET` → compiles `linkers_socket.cpp` + `socket_wrapper.hpp` and the socket
  `Send/Recv/SendRecv` (`linkers.h:21-23, 240-283`).
- `USE_MPI` → compiles `linkers_mpi.cpp` and the MPI `Send/Recv/SendRecv`
  (`linkers.h:25-28, 285-326`), pulling in `MPI_SAFE_CALL` (`linkers.h:27`).
Only one is active per build; the `Linkers` class has two exclusive definitions. (These map to
the CMake `USE_MPI` option described in the project's build config — MPI ON selects the MPI
backend, otherwise the socket backend.)

**A second, runtime seam** exists on top of that: `Network::Init(num_machines, rank,
reduce_scatter_ext_fun, allgather_ext_fun)` (`network.cpp:45`) lets an embedder inject its own
collective implementations via the `ReduceScatterFunction`/`AllgatherFunction` pointers, which
`ReduceScatter`/`Allgather` prefer over the built-ins when non-null
(`network.cpp:141-143, 238-239`).

### 5. How it fits the pipeline — and when it does NOT

This layer is **only active in distributed (multi-machine) training.** Every public entry point
guards on machine count:
- `Network::Init` does nothing unless `num_machines > 1` (`network.cpp:31, 47`), so on a single
  machine the network is never constructed and no ports/MPI are touched.
- The default statics leave `num_machines_ = 1`, `rank_ = 0` (`network.cpp:17-18`).
- Every collective op fatally errors if called while `num_machines_ <= 1`
  (`network.cpp:69-71, 96-97, 122-124, 138-140, 235-237`) — i.e. they are *only* meant to run
  in a properly-initialized distributed session.

**Newcomer takeaway:** if you are training on one machine (the overwhelmingly common case, and
the only case exercised by this Rust port's single-machine CPU/ROCm paths), this entire
subsystem is dormant — the boosting loop, tree learner, dataset, objective, and metric code all
run without ever calling into `Network`. It matters only for the parallel/distributed tree
learners on a cluster. In the pipeline it sits between the *tree learner* (which produces local
histograms/statistics) and the *rest of the workers* (whose local results must be merged): each
boosting iteration, the parallel learner builds local histograms, calls `Network::Allreduce`/
`ReduceScatter`/`Allgather` to obtain the global histograms, then finds splits on the merged
data exactly as the serial learner would.
