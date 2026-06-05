# External Integrations

> **Scope note:** This document maps the **LightGBM C++ REFERENCE implementation** under `LightGBM/` (the system being ported), not the Rust crate under development. It frames the **surface area** — the public C ABI, language bindings, CLI, I/O formats, distributed and GPU integration points — that a Rust+cubecl port must reproduce to be a drop-in replacement. All paths relative to `LightGBM/`.

**Analysis Date:** 2026-06-05

## The C API — primary integration surface

The single source of truth every binding sits on top of is the C ABI in **`include/LightGBM/c_api.h`** (~85 KB), implemented in **`src/c_api.cpp`**. Symbols are exported via `LIGHTGBM_C_EXPORT` (`extern "C"` + dllexport, see `include/LightGBM/export.h`). **95 exported functions.** Everything below (Python, R, Java, Dask) calls these.

**Opaque handle types** (`include/LightGBM/c_api.h`): `DatasetHandle`, `BoosterHandle`, `FastConfigHandle`, `ByteBufferHandle` (all `void*`).

**ABI constants a port MUST keep numerically identical:**
- dtype: `C_API_DTYPE_FLOAT32=0`, `FLOAT64=1`, `INT32=2`, `INT64=3`.
- predict type: `C_API_PREDICT_NORMAL=0`, `RAW_SCORE=1`, `LEAF_INDEX=2`, `CONTRIB=3` (SHAP).
- sparse matrix type: `C_API_MATRIX_TYPE_CSR=0`, `CSC=1`.
- feature importance: `C_API_FEATURE_IMPORTANCE_SPLIT=0`, `GAIN=1`.

**Calling convention / contract:** functions return `int` (`0` success, `-1` failure); error text retrieved via `LGBM_GetLastError()` / set via `LGBM_SetLastError()`. Outputs written through caller-allocated buffers with an `out_len` pre-flight pattern (e.g. `LGBM_BoosterSaveModelToString`). Both float32 and float64 inputs are accepted everywhere except gradients/Hessians and current scores (header preamble note).

**Function groups (surface a port reproduces):**

*Global / utility:* `LGBM_GetLastError`, `LGBM_SetLastError`, `LGBM_RegisterLogCallback`, `LGBM_DumpParamAliases`, `LGBM_GetSampleCount`, `LGBM_SampleIndices`, `LGBM_GetMaxThreads`, `LGBM_SetMaxThreads`, `LGBM_ByteBufferGetAt`, `LGBM_ByteBufferFree`, `LGBM_FastConfigFree`.

*Dataset construction (in-memory):* `LGBM_DatasetCreateFromFile`, `...FromMat`, `...FromMats`, `...FromCSR`, `...FromCSRFunc`, `...FromCSC`, `...FromSampledColumn`, `...CreateByReference`, `...FromSerializedReference`, `...CreateFromArrow`, `LGBM_DatasetGetSubset`, `LGBM_DatasetAddFeaturesFrom`.

*Dataset streaming / push API:* `LGBM_DatasetInitStreaming`, `LGBM_DatasetPushRows`, `...PushRowsByCSR`, `...PushRowsWithMetadata`, `...PushRowsByCSRWithMetadata`, `LGBM_DatasetSetWaitForManualFinish`, `LGBM_DatasetMarkFinished`.

*Dataset fields / metadata / persistence:* `LGBM_DatasetGetField`, `LGBM_DatasetSetField`, `LGBM_DatasetSetFieldFromArrow`, `LGBM_DatasetGetNumData`, `...GetNumFeature`, `...GetFeatureNumBin`, `...GetFeatureNames`, `...SetFeatureNames`, `...UpdateParamChecking`, `LGBM_DatasetSaveBinary`, `LGBM_DatasetDumpText`, `LGBM_DatasetSerializeReferenceToBinary`, `LGBM_DatasetFree`.

*Booster lifecycle:* `LGBM_BoosterCreate`, `...CreateFromModelfile`, `...LoadModelFromString`, `...Free`, `...Merge`, `...AddValidData`, `...ResetTrainingData`, `...ResetParameter`, `...GetLoadedParam`.

*Training loop:* `LGBM_BoosterUpdateOneIter`, `...UpdateOneIterCustom` (custom grad/hess), `...RollbackOneIter`, `...GetCurrentIteration`, `...RefitTree` (`LGBM_BoosterRefit`), `...ShuffleModels`, `...SetLeafValue`, `...GetLeafValue`.

*Introspection:* `LGBM_BoosterGetNumClasses`, `...GetNumFeature`, `...GetFeatureNames`, `...ValidateFeatureNames`, `...GetEval`, `...GetEvalCounts`, `...GetEvalNames`, `...GetNumPredict`, `...GetPredict`, `...NumModelPerIteration`, `...NumberOfTotalModel`, `...FeatureImportance`, `...GetLinear`, `...GetUpperBoundValue`, `...GetLowerBoundValue`.

*Prediction:* `LGBM_BoosterPredictForFile`, `...ForMat`, `...ForMats`, `...ForCSR`, `...ForCSC`, `...ForArrow`, single-row fast paths (`...ForMatSingleRow`, `...ForMatSingleRowFast(+Init)`, `...ForCSRSingleRow`, `...ForCSRSingleRowFast(+Init)`), sparse output (`...PredictSparseOutput`, `...FreePredictSparse`), `...CalcNumPredict`.

*Model persistence:* `LGBM_BoosterSaveModel` (to file), `...SaveModelToString`, `...DumpModel` (JSON).

*Distributed:* `LGBM_NetworkInit`, `LGBM_NetworkInitWithFunctions`, `LGBM_NetworkFree` (see Distributed section).

## APIs & External Services

LightGBM is a self-contained library — **no network/cloud SaaS calls**. Its "integrations" are language bindings and data interchange.

**Language bindings (all wrap the C API above):**
- **Python** - `python-package/lightgbm/`. ctypes binding in `python-package/lightgbm/basic.py` (loads native lib via `python-package/lightgbm/libpath.py`, function `_find_lib_path` → `ctypes.cdll.LoadLibrary`, dll names `lib_lightgbm.{so,dll,dylib}`). Public modules: `engine.py` (train/cv), `sklearn.py` (scikit-learn estimators), `dask.py` (distributed), `callback.py`, `plotting.py`, `compat.py`, `__init__.py`.
- **R** - `R-package/`. C++ shim `R-package/src/lightgbm_R.cpp` exposing **58** `LGBM_*_R` `.Call` entry points (e.g. `LGBM_BoosterUpdateOneIter_R`, `LGBM_DatasetCreateFromMat_R`, `LGBM_BoosterPredictForCSR_R`). Built into native `lightgbm.{so,dll,dylib}` (note R strips the `lib_` prefix — see `CMakeLists.txt __BUILD_FOR_R`). `.def` for Windows: `R-package/src/lightgbm-win.def`. R API surface in `R-package/R/`, exports declared in `R-package/NAMESPACE`.
- **Java / JVM (SWIG)** - `swig/lightgbmlib.i` generates `com.microsoft.ml.lightgbm` JNI bindings, packaged as `lightgbmlib.jar` (`CMakeLists.txt USE_SWIG` block). Helper SWIG interfaces: `swig/StringArray.i`, `swig/StringArray_API_extensions.i`, `swig/ChunkedArray_API_extensions.i`, `swig/pointer_manipulation.i`, with C++ helper `swig/StringArray.hpp`. This is the binding consumed by SynapseML/MMLSpark.

**A Rust port should expose the same `LGBM_*` C ABI** so these existing wrappers (and any consumer linking `lib_lightgbm`) keep working unchanged.

## Data Storage

LightGBM has **no database/object-store integration**. All persistence is file-based or in-memory.

**Model persistence formats (must be byte/text compatible for a port):**
- Text model file - human-readable LightGBM format, written/read by `src/boosting/gbdt_model_text.cpp`, C API `LGBM_BoosterSaveModel` / `LGBM_BoosterCreateFromModelfile` / `LGBM_BoosterLoadModelFromString` / `LGBM_BoosterSaveModelToString`.
- JSON model dump - `LGBM_BoosterDumpModel` (via vendored json11, `src/io/json11.cpp`).

**Dataset persistence:**
- Binary dataset (`.bin`) - LightGBM's pre-binned columnar cache. Saved via `LGBM_DatasetSaveBinary`, loaded transparently when `LGBM_DatasetCreateFromFile` sees a binary file (`src/io/dataset_loader.cpp`, "Load from binary file"). Companion `.bin.init` initial-score file.
- Serialized reference - `LGBM_DatasetSerializeReferenceToBinary` / `LGBM_DatasetCreateFromSerializedReference` (carries bin mappers so multiple datasets share binning).
- Text dump - `LGBM_DatasetDumpText`.

**File I/O abstraction:** `src/io/file_io.cpp` + `include/LightGBM/utils/file_io.h` (local filesystem; also a pipe/HDFS-style `VirtualFileWriter`/`Reader` abstraction).

## Supported Input Data Formats (port must parse identically)

Detected and parsed in **`src/io/parser.cpp`** / `parser.hpp` (`DataType` enum: `CSV`, `TSV`, `LIBSVM`):
- **CSV** (comma-delimited, optional header).
- **TSV** (tab-delimited).
- **LIBSVM / SVMLight** sparse text (`label idx:value ...`), column count via `GetNumColFromLIBSVMFile`.
- Auto-detection of delimiter and label column (`GetLabelIdxForCSV/TSV/Libsvm`). Numeric parsing uses fast_double_parser.

In-memory dataset ingestion paths (via C API): dense matrix (row- or col-major), CSR sparse, CSC sparse, sampled-column construction, and **Apache Arrow** C Data Interface (`ArrowArray`/`ArrowSchema`) — see `include/LightGBM/arrow.h`, `arrow.tpp`, `LGBM_DatasetCreateFromArrow`, `LGBM_DatasetSetFieldFromArrow`, `LGBM_BoosterPredictForArrow`. Arrow is the closest thing to a third-party data-interchange "integration."

## Authentication & Identity

Not applicable. LightGBM performs no auth; it is a compute library.

## Monitoring & Observability

**Logging:** in-house logger `include/LightGBM/utils/log.h` (levels Fatal/Warning/Info/Debug). Redirectable by the host application via `LGBM_RegisterLogCallback` (C API) — the main hook a Rust port must provide for embedders.

**Timing:** optional `USE_TIMETAG` build flag (`-DTIMETAG`) emits per-routine time costs.

**Error tracking:** thread-local last-error string surfaced through `LGBM_GetLastError`.

## Distributed / Parallel Integration

Distributed (multi-machine) training is a first-class surface. Backend selected at build time (`CMakeLists.txt`):
- **Socket backend** (default, `-DUSE_SOCKET`) - `src/network/linkers_socket.cpp`, `src/network/socket_wrapper.hpp`. TCP-based all-reduce/all-gather; Windows links `ws2_32`/`iphlpapi`.
- **MPI backend** (`USE_MPI=ON`, `-DUSE_MPI`) - `src/network/linkers_mpi.cpp`; `MPI_Finalize`/`MPI_Abort` driven from `src/main.cpp`.
- Topology / collective ops: `src/network/network.cpp`, `src/network/linker_topo.cpp`, `include/LightGBM/network.h`.

**C API entry points:** `LGBM_NetworkInit` (machine list + local listen port + rank), `LGBM_NetworkInitWithFunctions` (inject custom `reduce_scatter`/`allgather` callbacks — used by Dask/Spark to bridge their own comm layer), `LGBM_NetworkFree`. Collective function typedefs (`ReduceScatterFunction`, `AllgatherFunction`, `ReduceFunction`) in `include/LightGBM/meta.h`.

**Distributed tree-learner strategies** (`tree_learner` config, registered in `src/treelearner/tree_learner.cpp`): `serial`, `feature` (feature-parallel), `data` (data-parallel), `voting` (voting-parallel).

**Higher-level orchestration:** `python-package/lightgbm/dask.py` builds the network over Dask workers; the SWIG/Java binding powers Spark (SynapseML). Examples in `examples/parallel_learning/`.

## GPU / Device Integration

Selected at runtime via the `device_type` config (`include/LightGBM/config.h`, default `"cpu"`) and at build time via `USE_GPU`/`USE_CUDA`:
- **`cpu`** - default, OpenMP-parallel.
- **`gpu`** (OpenCL) - requires `USE_GPU=ON`. Boost.Compute + OpenCL ICD. Kernels: `src/treelearner/ocl/histogram16.cl`, `histogram64.cl`, `histogram256.cl`; host learner `src/treelearner/gpu_tree_learner.cpp`. 32-bit float histogram sums by default; `gpu_use_dp=true` forces fp64. Device selection via `gpu_platform_id` / `gpu_device_id` config (`include/LightGBM/config.h`).
- **`cuda`** - requires `USE_CUDA=ON` (Toolkit >= 11.0). Sources under `src/**/cuda/` and `src/cuda/` (`.cu` kernels: histogram constructor, data partition, best-split finder, leaf splits, gradient discretizer, single-GPU tree learner, plus CUDA objective/metric/score-updater). Headers `include/LightGBM/cuda/*.hpp`/`.hu`. `num_gpu` config for multi-GPU. CUDA implementation is double-precision only.

These device paths are the primary target the **cubecl** port replaces — a Rust port must reproduce the histogram-construction and best-split kernels and the `device_type` dispatch.

## Algorithm "plugin" registries (extension surface a port reproduces)

String-keyed factories that a port must match name-for-name:
- **Objectives** (`src/objective/objective_function.cpp`): `regression`, `regression_l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`, `binary`, `multiclass`, `multiclassova`, `cross_entropy`, `cross_entropy_lambda`, `lambdarank`, `rank_xendcg`. Plus user custom objective via `LGBM_BoosterUpdateOneIterCustom`.
- **Boosting types** (`src/boosting/boosting.cpp`): `gbdt`, `rf`, `dart`, `goss`.
- **Metrics** (`src/metric/metric.cpp`): `l1`, `l2`, `rmse`, `quantile`, `mape`, `huber`, `fair`, `poisson`, `ndcg`, `map`, `auc`, `average_precision`, `auc_mu`, `binary_logloss`, `binary_error`, `multi_logloss`, `multi_error`, `cross_entropy`, `cross_entropy_lambda`, `kullback_leibler`, `gamma`, `gamma_deviance`, `tweedie`.

## CLI Integration

Standalone executable `lightgbm` (`BUILD_CLI=ON`, default). Entry `src/main.cpp` → `LightGBM::Application` (`include/LightGBM/application.h`, `src/application/application.cpp`). Two tasks: **train** and **predict** (also model conversion via `ConvertModel`). Configuration from command-line `key=value` pairs and/or a config file, parsed by `src/io/config.cpp`. Example configs/datasets in `examples/` (`binary_classification/`, `regression/`, `multiclass_classification/`, `lambdarank/`, `xendcg/`, `parallel_learning/`).

## Environment Configuration

LightGBM is configured by **parameters**, not environment variables (no `.env`). Behavior knobs all live in `include/LightGBM/config.h` (e.g. `objective`, `boosting`, `tree_learner`, `device_type`, `num_machines`, `gpu_platform_id`, `gpu_device_id`, `num_gpu`, `num_threads`, `seed`, `deterministic`). Parameter aliases are introspectable at runtime via `LGBM_DumpParamAliases`.

## Webhooks & Callbacks

**Incoming:** none.
**Outgoing:** none.
**Host-supplied callbacks (the real "callback surface" a port must support):**
- Log redirect: `LGBM_RegisterLogCallback`.
- Custom training objective: `LGBM_BoosterUpdateOneIterCustom` (caller supplies grad/hess arrays).
- Streaming CSR data pull: `LGBM_DatasetCreateFromCSRFunc` (caller-supplied row-fetch function).
- Distributed collective injection: `LGBM_NetworkInitWithFunctions` (caller-supplied reduce-scatter/all-gather).

---

*Integration audit: 2026-06-05*
