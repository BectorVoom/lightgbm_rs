# Codebase Concerns

**Analysis Date:** 2026-06-05

> **Scope note:** This document maps porting concerns about the **LightGBM C++ reference implementation** under `LightGBM/`, *not* concerns in the Rust crate under development (which is still greenfield). Every concern is framed from the perspective of a Rust + cubecl GPU port that must reproduce the reference's outputs. All file paths are relative to the `LightGBM/` directory.

---

## Tech Debt

This section reframes "tech debt" as **porting-risk areas**: subsystems that are large, complex, tightly-coupled, or duplicated, and therefore expensive/hazardous to port faithfully.

**Histogram construction + best-split finding (the core hotpath):**
- Issue: This is the algorithmic heart of LightGBM and is the single largest, most-coupled subsystem. The split-gain math, leaf-output formulas, and threshold search are spread across templated headers with many compile-time flag specializations (`USE_L1`, `USE_MAX_OUTPUT`, `USE_SMOOTHING`, `USE_RAND`, `USE_MC`, `REVERSE`, `SKIP_DEFAULT_BIN`, `NA_AS_MISSING`).
- Files: `src/treelearner/feature_histogram.hpp` (1597 lines), `src/treelearner/feature_histogram.cpp` (739 lines), `src/treelearner/serial_tree_learner.cpp` (1110 lines), `src/treelearner/serial_tree_learner.h`, `src/treelearner/split_info.hpp`, `src/treelearner/leaf_splits.hpp`.
- Impact: Any deviation in the gain formula, epsilon handling, or bin iteration order changes which split is chosen, which cascades into a completely different tree. Bit-for-bit divergence here invalidates the whole port.
- Fix approach: Port `CalculateSplittedLeafOutput`, `GetLeafGain`, `GetSplitGains`, and `ThresholdL1` (in `feature_histogram.hpp` ~lines 711-790) as exact scalar Rust first; add a golden-output test harness against the C++ build before attempting cubecl kernelization of histogram accumulation.

**Multiple tree-learner variants sharing one base via templates:**
- Issue: `SerialTreeLearner` is subclassed/wrapped by six variants through CRTP-style templating (`TREELEARNER_T`, `TREE_LEARNER_TYPE`). The factory `TreeLearner::CreateTreeLearner` (`src/treelearner/tree_learner.cpp:15-57`) instantiates: serial, feature-parallel, data-parallel, voting-parallel, linear, GPU (OpenCL), and CUDA-single-GPU. Each parallel variant overrides only a few hooks but inherits deep behavior.
- Files: `src/treelearner/serial_tree_learner.{h,cpp}`, `src/treelearner/parallel_tree_learner.h`, `src/treelearner/data_parallel_tree_learner.cpp` (467), `src/treelearner/feature_parallel_tree_learner.cpp`, `src/treelearner/voting_parallel_tree_learner.cpp` (506), `src/treelearner/linear_tree_learner.{h,cpp}` (403).
- Impact: The C++ template-inheritance pattern does not map cleanly to Rust traits; choosing the wrong abstraction up front (trait vs enum dispatch vs generics) forces a rewrite. Scope decision needed: which variants does the Rust port even target?
- Fix approach: Recommend porting `SerialTreeLearner` only as the v1 target, modeling shared behavior as a trait with a concrete struct, and explicitly deferring/dropping the distributed (feature/data/voting) variants.

**Bin storage: five distinct bin implementations with templated split functions:**
- Issue: Dense, sparse, multi-val dense, and multi-val sparse bins each implement their own iterators and `Split`/`ConstructHistogram` with template parameters over bit width (4/8/16/32-bit packing).
- Files: `src/io/dense_bin.hpp` (649), `src/io/sparse_bin.hpp` (857), `src/io/multi_val_dense_bin.hpp` (359), `src/io/multi_val_sparse_bin.hpp` (448), `src/io/bin.cpp` (1072), `include/LightGBM/bin.h` (654).
- Impact: Large surface area of bit-packing and pointer arithmetic to reproduce; the multi-val sparse path is the one most exercised by GPU histogram kernels and is the hardest to get right.
- Fix approach: Port dense 8-bit bin first (covers most datasets), defer 4-bit packing and the sparse/multi-val variants until correctness is established.

**Quantized / discretized gradient path (low-bit training):**
- Issue: A parallel "quantized gradients" code path exists across CPU and CUDA, with int16/int32/int64 histogram accumulation and a specialized `Subtract` template (8 bit-width combinations) in `feature_histogram.hpp`.
- Files: `src/treelearner/gradient_discretizer.{cpp,hpp}` (390 lines total), `src/treelearner/feature_histogram.hpp:96-200` (`Subtract`, `CopyToBuffer`, `CopyFromInt16ToInt32`), `src/treelearner/serial_tree_learner.cpp:490-580`.
- Impact: This doubles the histogram code surface. It is an opt-in feature (`use_quantized_grad`) and is a candidate to *exclude* from the initial port scope.
- Fix approach: Explicitly scope out quantized gradients for v1; gate it behind a config flag that errors if requested.

---

## Numerical-Fidelity Hazards

These are the changes most likely to silently produce *different trained models* even when the code "works".

**Floating-point reduction ordering under OpenMP:**
- Risk: 50 source files use `#pragma omp`; many do `reduction(+:...)` over gradients/hessians (e.g. `src/treelearner/leaf_splits.hpp:98,123,152,180`, `src/treelearner/gradient_discretizer.cpp:219,245`, `src/objective/regression_objective.hpp:177`, `src/objective/xentropy_objective.hpp:150,280`). FP addition is non-associative, so the *order* of summation affects the low bits of `sum_gradients`/`sum_hessians`, which feed directly into split gains.
- Files: see `grep -rn "reduction(" src include`.
- Current mitigation: LightGBM has a `deterministic_` config flag that switches several loops to ordered summation (the reductions are gated `if (... && !deterministic_)`; `leaf_splits.hpp:180` is the deterministic branch).
- Recommendations: The Rust port must replicate the **deterministic** summation order to be comparable, and must decide whether cubecl GPU reductions can match the CPU deterministic order at all. Tree-walk reductions on GPU will not reproduce CPU FP results; expect to validate against `deterministic=true` CPU runs, not arbitrary multi-thread runs.

**Histogram subtraction trick:**
- Risk: LightGBM computes the smaller child's histogram directly, then derives the larger child by subtracting from the parent (`use_subtract` in `serial_tree_learner.cpp:398-580`, `FeatureHistogram::Subtract` in `feature_histogram.hpp:96-145`). This is both a major speedup *and* a source of accumulated FP/integer error that the reference deliberately tolerates.
- Files: `src/treelearner/feature_histogram.hpp:96-145`, `src/treelearner/serial_tree_learner.cpp:398-580`.
- Current mitigation: Reference uses the subtraction result as-is; the integer (quantized) path uses exact int subtraction with bit-packing.
- Recommendations: A faithful port must reproduce *which* child is constructed vs subtracted, since the resulting histograms differ in the low bits from direct construction. Do not "improve" this by always constructing directly.

**Missing-value / NaN bin placement:**
- Risk: `BinMapper::ValueToBin` (`include/LightGBM/bin.h:612-652`) encodes subtle rules: NaN maps to bin 0 for categorical, to `num_bin_ - 1` for `MissingType::NaN` numerical, or to value `0.0` (then binary-searched) for `MissingType::Zero`/`None`. The binary search uses `m = (r + l - 1) / 2` and `value <= bin_upper_bound_[m]` — an off-by-one-sensitive convention.
- Files: `include/LightGBM/bin.h:612-652`, `enum MissingType` at `bin.h:27`, `src/io/bin.cpp` (bin boundary construction).
- Current mitigation: N/A (this is the defined behavior).
- Recommendations: Port `ValueToBin` and the bin-boundary finder *exactly*, including the `(r + l - 1) / 2` rounding and the `<=` comparison. Test with NaN, +0/-0, and values exactly on bin boundaries.

**Default-bin / most-frequent-bin / zero-bin special handling:**
- Risk: Histograms store an `offset` and treat the most-frequent bin specially (`default_bin_`, `most_freq_bin_`, `SKIP_DEFAULT_BIN`). The "default left" direction and the skipping of the default bin during the threshold scan affect both speed and which thresholds are even considered.
- Files: `include/LightGBM/bin.h:180-258`, split functions in `feature_histogram.hpp` with `SKIP_DEFAULT_BIN`/`NA_AS_MISSING` template flags (`feature_histogram.hpp` numerical/categorical split functions ~lines 480-700).
- Recommendations: Replicate the offset arithmetic and default-bin skip exactly; these are easy to drop accidentally when restructuring the scan loop into a kernel.

**`kEpsilon` / `kZeroThreshold` constants threaded through gain math:**
- Risk: `kEpsilon` (and `2 * kEpsilon` added to hessians at `feature_histogram.hpp:172`) appears throughout leaf-output and gain computation (`feature_histogram.hpp:265,489,514,571-662`). `sum_hessian + 2 * kEpsilon`, `sum_left_hessian = kEpsilon`, etc. are load-bearing — they prevent division blow-ups and slightly shift outputs.
- Files: `include/LightGBM/meta.h` (constant definitions), `src/treelearner/feature_histogram.hpp` (usage).
- Recommendations: Copy the exact constant values from `meta.h` and apply them in the exact same arithmetic positions.

**Custom numeric helpers (`Pow`, `Atof`, `AtofPrecise`, sigmoid):**
- Risk: `include/LightGBM/utils/common.h` (1264 lines) ships hand-rolled `Pow` (recursive, `common.h:248`), `Atof`/`AtofPrecise` (uses bundled `fast_double_parser`, `common.h:262,355`), and other math. Parsing affects how text datasets are read; `Pow` affects objective math.
- Files: `include/LightGBM/utils/common.h`, `external_libs/fast_double_parser/`.
- Recommendations: For dataset parsing, match `AtofPrecise` semantics (it parses then validates) so the same rows produce the same bins. Rust's `str::parse::<f64>` may round differently than `fast_double_parser` on edge cases.

---

## Performance Bottlenecks (cubecl-kernel candidates)

These justify GPU kernels; they are where the reference spends its time and where the existing GPU/CUDA code already lives.

**Histogram accumulation over all data in a leaf:**
- Problem: For each split, gradient/hessian sums are accumulated per bin across millions of rows — the dominant cost of training.
- Files: `ConstructHistograms` in `src/treelearner/serial_tree_learner.cpp:405-470`; per-bin-type accumulators in `src/io/dense_bin.hpp`, `src/io/multi_val_*_bin.hpp`; reference GPU kernels in `src/treelearner/ocl/histogram256.cl`, `histogram64.cl`, `histogram16.cl`.
- Cause: O(num_data × num_feature) memory-bound reduction.
- Improvement path: This is the primary cubecl kernel target. Mirror the OpenCL `histogram256` workgroup design (256 bins per workgroup) as the porting reference.

**Best-split scan over bins for every feature:**
- Problem: After histograms are built, a left-to-right (and reverse) prefix scan over bins computes gains for every feature and leaf.
- Files: `FindBestSplitsFromHistograms` (`src/treelearner/serial_tree_learner.cpp:474-580`), `feature_histogram.cpp:739`, CUDA reference `src/treelearner/cuda/cuda_best_split_finder.cu` (2239 lines — the largest .cu file).
- Improvement path: Second cubecl kernel target; the CUDA `cuda_best_split_finder.cu` is the closest reference for a GPU split-finder.

**Data partitioning after a split:**
- Problem: Re-bucketing every row to left/right child leaves on each split.
- Files: `src/treelearner/data_partition.hpp`, CUDA reference `src/treelearner/cuda/cuda_data_partition.cu` (1121 lines).
- Improvement path: Third kernel target; partition is inherently scatter-heavy and benefits from GPU.

**Score updating / prediction:**
- Problem: Adding new-tree leaf outputs to per-row scores each iteration.
- Files: `src/boosting/score_updater.hpp`, CUDA reference `src/boosting/cuda/cuda_score_updater.cu`.
- Improvement path: Simple elementwise cubecl kernel.

**Gradient/hessian computation per objective:**
- Problem: Per-row gradient/hessian recompute each boosting iteration.
- Files: `src/objective/*_objective.hpp`, CUDA references under `src/objective/cuda/*.cu`.
- Improvement path: Elementwise kernels; the `src/objective/cuda/` files are direct references.

---

## Fragile Areas (C/C++ idioms that resist a clean Rust port)

**Pervasive `reinterpret_cast` and raw-pointer histogram aliasing:**
- Files: `src/treelearner/feature_histogram.hpp:88-160` casts the same buffer between `hist_t*`, `int32_t*`, `int16_t*`, `int64_t*` for the quantized path; 584 occurrences of cast/`memcpy`/aligned-alloc idioms across `src include` (`grep -rn "reinterpret_cast\|memcpy\|aligned"`).
- Why fragile: Type-punning a single allocation between float and packed-int views is undefined-behavior-adjacent in Rust and cannot be expressed with safe slices; needs explicit `bytemuck`/union modeling or separate typed buffers.
- Safe modification: Model histogram storage as a tagged buffer with explicit, audited reinterpretation functions; never alias the same `Vec` as two types simultaneously.

**Template + compile-time-flag explosion in the tree learner:**
- Files: `src/treelearner/feature_histogram.hpp` split functions take 5-7 `bool`/`int` template params; `Subtract` has 8 specialized branches (`feature_histogram.hpp:96-145`).
- Why fragile: Rust monomorphization can replicate this but the combinatorial surface makes exhaustive testing hard; const-generics help but cubecl kernels cannot be generic over all of it.
- Safe modification: Collapse the flag matrix into runtime config where the GPU path is concerned; keep monomorphized fast paths only for the proven-hot combinations.

**Conditional compilation (`#ifdef USE_GPU` / `USE_CUDA` / `USE_MPI`):**
- Files: GPU/CUDA gated throughout — e.g. `include/LightGBM/bin.h:603` (`#ifdef USE_CUDA` virtual method), `src/treelearner/cuda/cuda_single_gpu_tree_learner.hpp:12`, network `#ifdef USE_MPI` in `src/network/`. `src/main.cpp`, `src/io/multi_val_sparse_bin.hpp`, `src/boosting/bagging.hpp`, `src/boosting/goss.hpp` all branch on compile flags.
- Why fragile: The reference selects backend at *compile time*; the Rust+cubecl design likely selects at *runtime*. The C API and class hierarchies carry `#ifdef USE_CUDA` virtual methods that change the vtable shape, so the "same" class behaves differently per build.
- Safe modification: Treat cubecl as a single runtime-selected backend; do not try to mirror the compile-time split. Audit every `#ifdef USE_CUDA` virtual to decide whether the Rust trait needs that method at all.

**Global / static / thread-local state:**
- Files: `Log` is a static class with thread-local level/callback (`include/LightGBM/utils/log.h:37,86-99`); OpenMP thread-count default is process-global (`src/utils/openmp_wrapper.cpp`); `Random` instances are held `mutable` inside metainfo (`src/treelearner/feature_histogram.hpp:37`).
- Why fragile: Static logging and global thread config map poorly to Rust's ownership model and complicate determinism/testability.
- Safe modification: Replace the static `Log` with an injected logger or `tracing`; thread per-instance RNG state explicitly rather than via `mutable`.

**`Log::Fatal` as control flow (no exceptions/`Result`):**
- Files: `include/LightGBM/utils/log.h:43,74`, used pervasively (e.g. `tree_learner.cpp:50,53`).
- Why fragile: The reference aborts the process on errors instead of returning errors. A Rust port should convert these to `Result`/`panic!` deliberately; silently mirroring `Fatal` loses error recovery the C API callers may expect.

**Hand-rolled threading/blocking utilities:**
- Files: `include/LightGBM/utils/threading.h` (`BlockInfo`, `BlockInfoForceSize`, `Threading::For`), `include/LightGBM/utils/openmp_wrapper.h`.
- Why fragile: These encode the exact data-block partitioning that, combined with FP reductions, determines numerical results. Replacing with `rayon` changes block boundaries and thus low-bit sums.
- Safe modification: Reproduce `BlockInfo` block-sizing logic exactly if bit-comparability with multi-threaded CPU runs is required; otherwise standardize on the deterministic single-order path.

---

## Existing GPU Code (reference and divergence risk)

The reference ships **two independent, mature GPU backends**. Both are references *and* divergence hazards — the cubecl port is a *third* backend that must match CPU outputs, not necessarily either GPU backend.

**OpenCL backend (Boost.Compute):**
- Files: `src/treelearner/gpu_tree_learner.{cpp,h}` (1124 lines), OpenCL kernels `src/treelearner/ocl/histogram256.cl`, `histogram64.cl`, `histogram16.cl`, vendored `external_libs/compute/` and `external_libs/boost`.
- Risk: Uses `boost::compute::vector`, `enqueue_1d_range_kernel`, kernel-arg binding by index (`gpu_tree_learner.cpp:131-176`). Kernels are selected by `histogram256/64/16` based on max bin count. This is the closest structural template for a cubecl histogram kernel, but its FP accumulation order differs from CPU.
- Divergence: Reference GPU results are *not* bit-identical to CPU; the OpenCL path only handles histogram construction on GPU and falls back to CPU for split-finding. cubecl should follow the same "GPU histograms, validate against CPU split decisions" boundary initially.

**CUDA backend (full single-GPU pipeline):**
- Files: `src/treelearner/cuda/` — `cuda_single_gpu_tree_learner.{cpp,cu,hpp}`, `cuda_histogram_constructor.cu` (960), `cuda_best_split_finder.cu` (2239), `cuda_data_partition.cu` (1121), `cuda_leaf_splits.*`, `cuda_gradient_discretizer.*`; plus `src/cuda/cuda_algorithms.cu` (512) and `include/LightGBM/cuda/cuda_algorithms.hpp` (581). 66 `__global__` kernels total across the CUDA sources.
- Risk: This is a *complete* GPU training pipeline (histogram + split + partition + score), unlike the OpenCL partial offload. It is the richest reference for kernel design but is deeply tied to CUDA-specific primitives (warp shuffles, `cudaMemcpy`, block-reduce in `cuda_algorithms.cu`).
- Divergence: cubecl's abstractions differ from raw CUDA (no direct warp-shuffle/atomics parity guaranteed). Porting kernel logic 1:1 risks importing CUDA-specific numeric behavior. Use these `.cu` files as *algorithm* references, not line-by-line translation targets.
- Coupling concern: The CUDA path threads `boosting_on_gpu_`/`boosting_on_cuda` flags through GBDT (`src/boosting/gbdt.cpp:104,122-123,192,382`), score updaters (`src/boosting/cuda/cuda_score_updater.*`), objectives, and metrics (`src/objective/cuda/`, `src/metric/cuda/`). Enabling a GPU backend touches the entire training stack, not just the tree learner. The Rust port must decide how far GPU residency extends (gradients/scores on device vs only histograms).

---

## Scaling Limits (distributed / network code)

**Distributed training subsystem — recommend out of scope:**
- Files: `src/network/network.cpp` (328), `src/network/linkers_socket.cpp` (239), `src/network/linkers_mpi.cpp` (61), `src/network/linker_topo.cpp` (179), `src/network/linkers.h` (328), `src/network/socket_wrapper.hpp`; consumers `data_parallel_tree_learner.cpp`, `feature_parallel_tree_learner.cpp`, `voting_parallel_tree_learner.cpp`.
- Risk: Dual transport (`#ifdef USE_MPI` vs socket), AllReduce/ReduceScatter topology, and the three parallel tree-learner variants exist solely for multi-machine training.
- Recommendation: Explicitly scope distributed training *out* of the initial Rust+cubecl port. It is large, orthogonal to GPU acceleration, and rarely the reason to choose a Rust port. Document this as an intentional non-goal.

---

## Dependencies at Risk (port-blocking external libs)

**Vendored C++ libraries the reference depends on:**
- Files/dirs: `external_libs/eigen` (linear tree solver), `external_libs/compute` (Boost.Compute, OpenCL backend), `external_libs/fmt`, `external_libs/fast_double_parser` (used by `common.h:355` `AtofPrecise`).
- Risk: Eigen is required for `linear_tree` (`src/treelearner/linear_tree_learner.cpp` solves least-squares per leaf). Boost.Compute is the entire OpenCL backend. `fast_double_parser` governs dataset parsing fidelity.
- Migration plan: `nalgebra`/`faer` replaces Eigen *if* linear trees are in scope (recommend deferring linear trees); cubecl replaces Boost.Compute; Rust `f64::from_str` or a parity-checked parser replaces `fast_double_parser` (verify edge-case rounding, see Numerical-Fidelity Hazards).

---

## Missing Critical Features (scope/surface decisions for the port)

**C API surface — how much to reproduce:**
- Files: `include/LightGBM/c_api.h` (1658 lines, 95 `LIGHTGBM_C_EXPORT` functions), `src/c_api.cpp` (2986 lines — second-largest file in the repo).
- Problem: The C API is the contract used by the Python and R bindings. It covers dataset construction (from file, CSR, CSC, Arrow, mat), training, prediction (multiple modes incl. leaf index, SHAP contrib), model serialization (text + binary), and config parsing. Reproducing all 95 entry points is a large undertaking with behavior (e.g. predictor early-stop, `src/boosting/prediction_early_stop.cpp`) that must match.
- Blocks: Any goal of being a drop-in replacement for the Python/R bindings requires most of this surface.
- Recommendation: Decide explicitly whether the Rust port targets (a) a native Rust API only, (b) the C API as a compatibility shim, or (c) both. If (b/c), prioritize the dataset+train+predict core (~20 functions) and defer Arrow ingestion (`include/LightGBM/arrow.h`, `arrow.tpp`), chunked arrays (`include/LightGBM/utils/chunked_array.hpp`), and binary model load/save compatibility.

**Model serialization format compatibility:**
- Files: `src/boosting/gbdt_model_text.cpp` (663), `src/io/tree.cpp` (1055 — text serialization of trees), `src/io/json11.cpp` (781 — JSON model dump), `include/LightGBM/utils/json11.h`.
- Problem: The text/JSON model formats are how trained models interoperate with the reference. Reproducing them exactly is required if the port must load reference models or be loaded by reference tools.
- Recommendation: Treat model-format compatibility as a separate, explicit deliverable; it is independent of GPU work and easy to under-scope.

**Sample strategies (bagging / GOSS) coupling:**
- Files: `include/LightGBM/sample_strategy.h`, `src/boosting/sample_strategy.cpp`, `src/boosting/bagging.hpp`, `src/boosting/goss.hpp`; boosting variants `src/boosting/dart.hpp`, `src/boosting/rf.hpp`, `src/boosting/gbdt.{cpp,h}`.
- Problem: GOSS changes the hessian (`IsHessianChange()`, referenced at `gbdt.cpp:382`) and bagging reuses subsets; both interact with the GPU-residency decision and with RNG determinism.
- Recommendation: Port plain GBDT + bagging first; defer GOSS, DART, and RF as separate boosting variants behind config flags.

---

## Test Coverage Gaps (for the port's golden-test strategy)

**No fidelity oracle exists yet — must be built:**
- What's not tested: There is no existing harness comparing Rust output to C++ output (the Rust crate is greenfield). The reference's own tests live under `tests/cpp_tests/`, `tests/c_api_test/`, and `tests/python_package_test/` but exercise the C++/Python paths, not a Rust port.
- Files: `tests/cpp_tests/`, `tests/c_api_test/`, `tests/python_package_test/`, `tests/distributed/`.
- Risk: Without a deterministic golden-output comparison, numerical-fidelity drift (FP ordering, histogram subtraction, bin placement) will go undetected until models silently diverge.
- Priority: **High.** Before any cubecl work, stand up a harness that runs the reference C++ build with `deterministic=true` on fixed seeds/datasets and snapshots: bin boundaries, per-split gains, leaf outputs, and final predictions. Use these as Rust regression fixtures.

**Edge-case inputs most likely to diverge:**
- What to test: NaN features (both `MissingType::NaN` and `Zero`), values exactly on bin boundaries, categorical features with unseen/negative categories (`bin.h:640-650`), single-row leaves, zero-hessian rows, and the histogram-subtraction child vs directly-constructed child.
- Files driving these: `include/LightGBM/bin.h:612-652`, `src/treelearner/feature_histogram.hpp` (subtract + split), `src/io/bin.cpp`.
- Priority: **High** — these are the documented numerical-fidelity hazards above.

---

*Concerns audit: 2026-06-05*
