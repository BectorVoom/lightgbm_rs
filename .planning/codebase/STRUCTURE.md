# Codebase Structure

**Analysis Date:** 2026-06-05

> **Scope note:** This document maps the **LightGBM C++ REFERENCE implementation** under the `LightGBM/` directory (Microsoft's reference C++ core), the system being ported to Rust + cubecl. It does NOT describe the Rust crate under development. Exploration was restricted to `LightGBM/src/` and `LightGBM/include/`. All paths below are relative to the repo root. Subsystems that are GPU-execution candidates (cubecl kernels) are flagged **[GPU-RELEVANT]**.

## Directory Layout

```
LightGBM/
├── include/LightGBM/          # Public headers: interfaces + factories
│   ├── boosting.h             # Boosting interface (GBDT/DART/RF)
│   ├── tree_learner.h         # TreeLearner interface + factory
│   ├── objective_function.h   # ObjectiveFunction interface
│   ├── metric.h               # Metric interface + DCGCalculator
│   ├── dataset.h              # Dataset + Metadata (binned columnar store)
│   ├── bin.h                  # Bin / BinMapper / MultiValBin / BinIterator
│   ├── feature_group.h        # FeatureGroup (group of binned features)
│   ├── tree.h                 # Tree model (nodes, splits, leaf outputs)
│   ├── config.h               # Config struct (all hyperparameters)
│   ├── c_api.h                # Stable C ABI (LGBM_* declarations)
│   ├── network.h              # Distributed comm interface
│   ├── sample_strategy.h      # Bagging/GOSS strategy interface
│   ├── train_share_states.h   # Shared per-thread training buffers
│   ├── arrow.h / meta.h / export.h / prediction_early_stop.h
│   ├── cuda/                  # CUDA-specific headers          [GPU-RELEVANT]
│   │   ├── cuda_algorithms.hpp, cuda_tree.hpp, cuda_row_data.hpp
│   │   ├── cuda_column_data.hpp, cuda_split_info.hpp, cuda_metadata.hpp
│   │   └── cuda_objective_function.hpp, cuda_metric.hpp, vector_cudahost.h
│   └── utils/                 # Header-only utilities
│       ├── common.h, log.h, threading.h, random.h, array_args.h
│       ├── text_reader.h, pipeline_reader.h, file_io.h, json11.h
│       ├── openmp_wrapper.h, chunked_array.hpp, byte_buffer.h
│       └── yamc/              # vendored shared-mutex library
└── src/
    ├── main.cpp               # CLI entry point (main())
    ├── c_api.cpp              # ~120 LGBM_* C API functions (2986 lines)
    ├── application/           # CLI orchestration
    │   ├── application.cpp    # Application: parse, load, train, predict
    │   └── predictor.hpp      # Predictor: batch prediction driver
    ├── boosting/              # Ensemble loop (gradient boosting)
    │   ├── gbdt.cpp / gbdt.h  # GBDT core training loop
    │   ├── gbdt_prediction.cpp        # GBDT prediction
    │   ├── gbdt_model_text.cpp        # model serialize/deserialize + codegen
    │   ├── boosting.cpp       # Boosting::CreateBoosting factory
    │   ├── dart.hpp, rf.hpp   # DART, Random Forest variants
    │   ├── goss.hpp, bagging.hpp      # SampleStrategy implementations
    │   ├── sample_strategy.cpp        # SampleStrategy::CreateSampleStrategy
    │   ├── score_updater.hpp          # accumulates ensemble scores
    │   ├── prediction_early_stop.cpp
    │   └── cuda/              # cuda_score_updater.{cpp,cu,hpp}   [GPU-RELEVANT]
    ├── treelearner/           # Tree growing: histograms + split finding
    │   ├── serial_tree_learner.cpp / .h   # single-machine baseline
    │   ├── tree_learner.cpp   # TreeLearner::CreateTreeLearner factory
    │   ├── feature_histogram.cpp / .hpp   # histogram split-gain scan  [GPU-RELEVANT]
    │   ├── data_partition.hpp # row→leaf partitioning
    │   ├── leaf_splits.hpp, split_info.hpp        # split candidates
    │   ├── col_sampler.hpp    # feature subsampling
    │   ├── monotone_constraints.hpp, cost_effective_gradient_boosting.hpp
    │   ├── gradient_discretizer.cpp / .hpp        # quantized gradients
    │   ├── data_parallel_tree_learner.cpp         # distributed variants
    │   ├── feature_parallel_tree_learner.cpp
    │   ├── voting_parallel_tree_learner.cpp
    │   ├── parallel_tree_learner.h
    │   ├── linear_tree_learner.cpp / .h           # linear-leaf trees
    │   ├── gpu_tree_learner.cpp / .h              # OpenCL learner   [GPU-RELEVANT]
    │   ├── ocl/               # OpenCL histogram kernels           [GPU-RELEVANT]
    │   │   └── histogram16.cl, histogram64.cl, histogram256.cl
    │   └── cuda/              # CUDA learner + kernels             [GPU-RELEVANT]
    │       ├── cuda_single_gpu_tree_learner.{cpp,cu,hpp}
    │       ├── cuda_histogram_constructor.{cpp,cu,hpp}
    │       ├── cuda_best_split_finder.{cpp,cu,hpp}
    │       ├── cuda_data_partition.{cpp,cu,hpp}
    │       ├── cuda_leaf_splits.{cpp,cu,hpp}
    │       └── cuda_gradient_discretizer.{cu,hpp}
    ├── io/                    # Data loading, binning, model I/O
    │   ├── dataset.cpp        # Dataset: ConstructHistograms, FinishLoad
    │   ├── dataset_loader.cpp # DatasetLoader: file→binned Dataset
    │   ├── bin.cpp            # BinMapper::FindBin, bin construction
    │   ├── dense_bin.hpp, sparse_bin.hpp          # Bin impls         [GPU-RELEVANT]
    │   ├── multi_val_dense_bin.hpp, multi_val_sparse_bin.hpp
    │   ├── metadata.cpp       # labels, weights, query boundaries
    │   ├── tree.cpp           # Tree model storage + prediction
    │   ├── train_share_states.cpp     # per-thread histogram buffers
    │   ├── config.cpp, config_auto.cpp            # parameter parsing
    │   ├── parser.cpp / .hpp, file_io.cpp, json11.cpp
    │   └── cuda/              # cuda_row_data, cuda_column_data, cuda_tree, cuda_metadata  [GPU-RELEVANT]
    ├── metric/               # Evaluation metrics
    │   ├── metric.cpp         # Metric::CreateMetric factory
    │   ├── regression_metric.hpp, binary_metric.hpp
    │   ├── multiclass_metric.hpp, rank_metric.hpp, map_metric.hpp
    │   ├── xentropy_metric.hpp, dcg_calculator.cpp
    │   └── cuda/              # cuda_*_metric.{cpp,cu,hpp}          [GPU-RELEVANT]
    ├── objective/            # Loss functions (gradients/hessians)  [GPU-RELEVANT]
    │   ├── objective_function.cpp     # CreateObjectiveFunction factory
    │   ├── regression_objective.hpp, binary_objective.hpp
    │   ├── multiclass_objective.hpp, rank_objective.hpp, xentropy_objective.hpp
    │   └── cuda/              # cuda_*_objective.{cpp,cu,hpp}       [GPU-RELEVANT]
    ├── network/              # Distributed training transport
    │   ├── network.cpp        # allreduce/allgather collectives
    │   ├── linkers.h, linker_topo.cpp
    │   ├── linkers_mpi.cpp, linkers_socket.cpp, socket_wrapper.hpp
    ├── cuda/                 # Shared CUDA helpers                  [GPU-RELEVANT]
    │   ├── cuda_algorithms.cu, cuda_utils.cpp
    └── utils/
        └── openmp_wrapper.cpp # OpenMP thread-count helpers
```

## Directory Purposes

**`include/LightGBM/` (top level):**
- Purpose: Public C++ interfaces and factories — the abstraction contracts to port.
- Key files: `boosting.h`, `tree_learner.h`, `objective_function.h`, `metric.h`, `dataset.h`, `bin.h`, `tree.h`, `config.h`, `c_api.h`.

**`include/LightGBM/cuda/` & `include/LightGBM/utils/`:**
- `cuda/`: CUDA-host data structures and algorithm headers **[GPU-RELEVANT]**.
- `utils/`: header-only helpers (logging, threading, RNG, JSON, text/file readers); `yamc/` is a vendored shared-mutex.

**`src/boosting/`:**
- Purpose: Outer gradient-boosting ensemble loop and variants.
- Key files: `gbdt.cpp` (training loop), `boosting.cpp` (factory), `goss.hpp`/`bagging.hpp` (sampling).

**`src/treelearner/`:**
- Purpose: Inner tree-growing algorithm — histogram construction, split finding, partitioning. Core compute hotspot.
- Key files: `serial_tree_learner.cpp` (baseline), `feature_histogram.{cpp,hpp}`, `tree_learner.cpp` (factory). Subdirs `ocl/` and `cuda/` hold GPU kernels **[GPU-RELEVANT]**.

**`src/io/`:**
- Purpose: Raw data → binned `Dataset`; bin mapping; tree model storage; config parsing.
- Key files: `dataset.cpp`, `dataset_loader.cpp`, `bin.cpp`, `dense_bin.hpp`/`sparse_bin.hpp`, `tree.cpp`.

**`src/objective/` & `src/metric/`:**
- Purpose: Loss gradients/hessians and evaluation metrics. Mostly header-only `.hpp` implementations selected by a `.cpp` factory; `cuda/` subdirs mirror each for GPU **[GPU-RELEVANT]**.

**`src/network/`:**
- Purpose: Distributed-training transport (MPI and raw sockets) and collective ops.

**`src/cuda/` & `src/application/` & `src/utils/`:**
- `cuda/`: shared device algorithms/utilities **[GPU-RELEVANT]**.
- `application/`: CLI orchestration (`application.cpp`) and batch predictor (`predictor.hpp`).
- `utils/`: OpenMP wrapper.

## Key File Locations

**Entry Points:**
- `src/main.cpp`: CLI `main()`.
- `src/c_api.cpp` + `include/LightGBM/c_api.h`: stable C ABI for bindings.
- `src/application/application.cpp`: `Application::Run` train/predict dispatch.

**Configuration:**
- `include/LightGBM/config.h`: all hyperparameters (`Config` struct).
- `src/io/config.cpp`, `src/io/config_auto.cpp`: parameter parsing/aliasing (auto-generated).

**Core Logic:**
- Ensemble loop: `src/boosting/gbdt.cpp` (`TrainOneIter`, `Boosting`, `UpdateScore`).
- Tree growth: `src/treelearner/serial_tree_learner.cpp` (`Train`, `ConstructHistograms`, `FindBestSplits`).
- Histogram split gain: `src/treelearner/feature_histogram.cpp`/`.hpp`.
- Binning & histogram accumulation: `src/io/bin.cpp`, `src/io/dense_bin.hpp`, `src/io/sparse_bin.hpp`.
- Dataset: `src/io/dataset.cpp`; Tree model: `src/io/tree.cpp`.

**Interfaces (porting seams):**
- `include/LightGBM/{boosting,tree_learner,objective_function,metric}.h`, `dataset.h`, `bin.h`.

**Testing:**
- No tests under `LightGBM/src/` or `LightGBM/include/`. The reference repo's tests live in Python bindings / separate `tests/` not in scope here. Not applicable to this subtree.

## Naming Conventions

**Files:**
- Snake_case: `serial_tree_learner.cpp`, `feature_histogram.hpp`, `dataset_loader.cpp`.
- Implementation pairs: interface in `include/LightGBM/<name>.h`, impl in `src/<area>/<name>.cpp`.
- Header-only strategy implementations use `.hpp` (e.g. `binary_objective.hpp`, `dense_bin.hpp`); compiled units use `.cpp`.
- CUDA: `.cu` (device kernels) + `.cpp` (host glue) + `.hpp` (declarations), prefixed `cuda_*`.
- OpenCL kernels: `.cl` under `treelearner/ocl/`.

**Directories:**
- Lowercase single-word per subsystem: `boosting/`, `treelearner/`, `io/`, `objective/`, `metric/`, `network/`, `application/`, `utils/`, `cuda/`.
- Device backends nested as `<subsystem>/cuda/` and `treelearner/ocl/`.

**Symbols (C++):**
- Classes: `PascalCase` (`SerialTreeLearner`, `BinMapper`, `FeatureHistogram`).
- Interface base classes named after the concept (`Boosting`, `Metric`, `ObjectiveFunction`, `TreeLearner`).
- Factories: static `Create<Concept>(...)` methods.
- Member variables: trailing underscore (`train_data_`, `gradients_`, `tree_learner_`).
- C API: `LGBM_` prefix, `PascalCase` after (`LGBM_BoosterCreate`, `LGBM_DatasetCreateFromMat`).
- Macros: `UPPER_SNAKE` (`CHECK_EQ`, `OMP_NUM_THREADS`, `USE_CUDA`).

## Where to Add New Code

> For the Rust + cubecl port, mirror these seams as Rust modules/traits. The mapping below indicates the C++ origin for each Rust target.

**New objective (loss):**
- C++ origin: add `.hpp` in `src/objective/`, register in `src/objective/objective_function.cpp`.
- Rust target: implement the `ObjectiveFunction` trait (from `include/LightGBM/objective_function.h`); `GetGradients` is a cubecl kernel candidate.

**New metric:**
- C++ origin: add `.hpp` in `src/metric/`, register in `src/metric/metric.cpp`.

**New tree learner / device backend:**
- C++ origin: subclass `TreeLearner`, register in `src/treelearner/tree_learner.cpp`. GPU kernels go in `src/treelearner/ocl/` (OpenCL) or `src/treelearner/cuda/` (CUDA).
- Rust target: implement the `TreeLearner` trait; histogram construction and best-split finding are the primary cubecl kernels.

**New boosting variant:**
- C++ origin: subclass `Boosting`/`GBDTBase` in `src/boosting/`, register in `src/boosting/boosting.cpp`.

**New bin/storage type:**
- C++ origin: implement `Bin`/`MultiValBin` (`include/LightGBM/bin.h`) as a `.hpp` in `src/io/`.

**New C API function:**
- C++ origin: declare in `include/LightGBM/c_api.h`, implement in `src/c_api.cpp` wrapped in `API_BEGIN/API_END`.

**Utilities / shared helpers:**
- C++ origin: `include/LightGBM/utils/` (header-only) or `src/utils/`.

## Special Directories

**`include/LightGBM/utils/yamc/`:**
- Purpose: vendored third-party shared-mutex (read-write lock) library.
- Generated: No. Committed: Yes (vendored dependency — likely replaced by `std`/`parking_lot` in Rust).

**`src/io/config_auto.cpp` (+ generated config glue):**
- Purpose: parameter-to-field mapping for `Config`.
- Generated: Yes (auto-generated from a parameter spec in the upstream repo). Committed: Yes. Treat as a data source, not hand-edited logic, when porting.

**`src/treelearner/ocl/` (`.cl`) and all `cuda/` subdirs (`.cu`):**
- Purpose: GPU kernel sources (OpenCL / CUDA). **[GPU-RELEVANT]** — these are the direct references for cubecl kernel design (histogram construction, best-split finding, data partition).
- Generated: No. Committed: Yes.

---

*Structure analysis: 2026-06-05*
