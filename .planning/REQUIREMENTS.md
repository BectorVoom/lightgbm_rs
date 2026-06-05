# Requirements: LightGBM-rs

**Defined:** 2026-06-05
**Core Value:** For identical inputs and config, reproduce C++ LightGBM outputs to within 1e-12 absolute difference on every backend (CPU and ROCm).

> **Scope:** v1 = full single-machine parity with C++ LightGBM (GBDT/DART/RF/GOSS, all objectives + metrics, categorical, monotone, SHAP), exposed via a Rust-native API and Python bindings, on a switchable CubeCL CPU/ROCm backend. The 1e-12 oracle is a hard merge gate on every backend. Distributed training, C ABI, CLI, and raw CUDA/OpenCL are out of scope.

## v1 Requirements

### Foundations & Determinism

- [ ] **FND-01**: Port LightGBM's `Random` PRNG (32-bit LCG, `NextFloat`, `Sample(N,K)`) bit-for-bit, unit-tested against a captured C++ draw sequence
- [ ] **FND-02**: Establish workspace crate structure (loosely-coupled crates by responsibility) building under edition 2024
- [ ] **FND-03**: Define deterministic reduction strategy (integer-quantized histograms / ordered f64 accumulation) so structural results are bit-identical across CPU and ROCm
- [ ] **FND-04**: `thiserror` domain error types at every crate boundary; `anyhow` propagation in application/test layers

### Configuration

- [ ] **CFG-01**: Config struct accepting the ~110 in-scope single-machine hyperparameters
- [ ] **CFG-02**: Parameter alias resolution (e.g. `num_iteration`/`n_estimators`/`num_boost_round`) as a data table matching `config_auto.cpp`
- [ ] **CFG-03**: Parameter validation mirroring C++ `Config::Set` CHECK constraints, surfaced as typed `Result` errors

### Dataset, Binning & I/O

- [ ] **DAT-01**: `BinMapper` continuous→bin mapping (`FindBin`) producing bit-identical bin boundaries vs C++ (`max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, `data_random_seed`)
- [ ] **DAT-02**: Binned columnar dataset store (DenseBin + SparseBin) immutable after finish-load
- [ ] **DAT-03**: Missing-value handling (`use_missing`, `zero_as_missing`, `MissingType`) with C++-matching default-direction routing
- [ ] **DAT-04**: Categorical feature encoding (category→bin mapping, low-frequency folding)
- [ ] **DAT-05**: Exclusive Feature Bundling (`enable_bundle`) reproducing C++ feature grouping
- [ ] **DAT-06**: Metadata support (labels, weights, init_score, query/group boundaries)
- [ ] **DAT-07**: In-memory matrix ingestion (dense + CSR/CSC sparse) via the Rust API
- [ ] **DAT-08**: LightGBM model text format read — load a C++-trained model and predict identically
- [ ] **DAT-09**: LightGBM model text format write — emit the exact text schema (trees, leaf values, bin mappers, feature metadata) including `%.17g` float formatting

### Tree Learner

- [ ] **TRL-01**: Histogram-based serial tree learner (`ConstructHistograms` → `FindBestSplitsFromHistograms` → `Split`)
- [ ] **TRL-02**: Histogram subtraction trick producing the byte-identical FP path the model is defined against
- [ ] **TRL-03**: Leaf-wise (best-first) growth with `num_leaves` and `max_depth` caps
- [ ] **TRL-04**: Split-gain scan with exact gain formula and tie-breaking (`lambda_l1`, `lambda_l2`, `min_gain_to_split`, `min_sum_hessian_in_leaf`, `min_data_in_leaf`, `max_delta_step`, `path_smooth`)
- [ ] **TRL-05**: Numerical threshold splits with C++-matching missing/zero routing
- [ ] **TRL-06**: Categorical splits (`SplitCategorical`/`FindBestThresholdCategorical`: `max_cat_threshold`, `cat_smooth`, `min_data_per_group`, `max_cat_to_onehot`, `cat_l2`)
- [ ] **TRL-07**: Data partition (row→leaf routing) feeding histogram subtraction
- [ ] **TRL-08**: Feature subsampling per-tree and per-node (`feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed`)
- [ ] **TRL-09**: `force_row_wise` / `force_col_wise` histogram build strategies, both output-matching

### Boosting & Sample Strategies

- [ ] **BST-01**: GBDT training loop (`TrainOneIter`, `UpdateScore`, per-class trees, shrinkage, `boost_from_average`)
- [ ] **BST-02**: Score updater accumulation with deterministic reduction ordering
- [ ] **BST-03**: Bagging / row subsampling (`bagging_fraction`/`bagging_freq`/`bagging_seed`, pos/neg, `bagging_by_query`) with RNG-matching sequence
- [ ] **BST-04**: GOSS sample strategy (`top_rate`/`other_rate`) with matching gradient-magnitude sort and amplification factor
- [ ] **BST-05**: DART boosting (`drop_rate`, `max_drop`, `skip_drop`, `uniform_drop`, `xgboost_dart_mode`, `drop_seed`)
- [ ] **BST-06**: Random Forest boosting (averaged trees, mandatory bagging, no shrinkage accumulation)
- [ ] **BST-07**: Early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`)

### Objective Functions

- [ ] **OBJ-01**: Core objectives — `regression` (l2), `regression_l1`, `binary`, `multiclass` (softmax), `multiclassova`
- [ ] **OBJ-02**: `custom` objective (user-supplied grad/hess pass-through) for Python parity
- [ ] **OBJ-03**: Objective machinery — `GetGradients`, `ConvertOutput` (sigmoid/softmax/exp), `BoostFromScore`, `reg_sqrt` — exact to 1e-12
- [ ] **OBJ-04**: Remaining regression objectives — `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`
- [ ] **OBJ-05**: Cross-entropy objectives — `cross_entropy`, `cross_entropy_lambda`
- [ ] **OBJ-06**: Ranking objectives — `lambdarank`, `rank_xendcg` (query boundaries, DCGCalculator, `objective_seed`)

### Metrics

- [ ] **MET-01**: Core metrics — `l1`, `l2`, `rmse`, `binary_logloss`, `binary_error`, `auc`, `multi_logloss`
- [ ] **MET-02**: Metric infrastructure — multi-metric lists, `metric_freq`, `is_provide_training_metric`, training-metric eval
- [ ] **MET-03**: Extended regression/xentropy metrics — `quantile`, `huber`, `fair`, `poisson`, `mape`, `gamma`, `gamma_deviance`, `tweedie`, `multi_error`, `cross_entropy`, `cross_entropy_lambda`, `kullback_leibler`, `average_precision`, `auc_mu`
- [ ] **MET-04**: Ranking metrics — `ndcg`, `map` (DCGCalculator static tables, `eval_at`/`ndcg_eval_at`, per-query)

### Prediction

- [ ] **PRD-01**: Raw score prediction (sum of tree outputs)
- [ ] **PRD-02**: Transformed prediction (`ConvertOutput` sigmoid/softmax)
- [ ] **PRD-03**: Leaf index prediction (`pred_leaf`)
- [ ] **PRD-04**: Feature contributions / TreeSHAP (`predict_contrib`) over full tree node/cover structure
- [ ] **PRD-05**: Prediction early stopping (`pred_early_stop`, `_freq`, `_margin`)
- [ ] **PRD-06**: Sub-range prediction (`start_iteration` / `num_iteration`)

### Constraints & Advanced Parity

- [ ] **ADV-01**: Monotone constraints (basic, intermediate, advanced; `monotone_penalty`)
- [ ] **ADV-02**: Interaction constraints (`interaction_constraints`)
- [ ] **ADV-03**: Forced splits / forced bins (JSON-driven)
- [ ] **ADV-04**: Extra trees (`extra_trees`, `extra_seed`) randomized thresholds
- [ ] **ADV-05**: CEGB cost-effective gradient boosting (`cegb_tradeoff`, penalties)
- [ ] **ADV-06**: Refit / continue training (`refit_decay_rate`, `input_model`) for `Booster.refit()`
- [ ] **ADV-07**: Feature importance reporting (split/gain, `saved_feature_importance_type`)

### Compute Backend (CubeCL)

- [ ] **CMP-01**: `lgbm-compute` backend trait isolating all device ops behind one crate (contains CubeCL alpha churn)
- [ ] **CMP-02**: CPU backend (cubecl-cpu) as the deterministic reference execution path
- [ ] **CMP-03**: ROCm/HIP backend (cubecl-hip) selectable via Cargo feature and/or runtime config
- [ ] **CMP-04**: CUDA warp-level operations mapped onto CubeCL's `Plane` API with capability gating and sequential fallback
- [ ] **CMP-05**: GPU-resident histogram construction, best-split finding, and data partition kernels meeting the 1e-12 contract

### Oracle & Validation

- [ ] **ORA-01**: Oracle harness comparing Rust vs C++ LightGBM outputs at ≤1e-12 absolute
- [ ] **ORA-02**: Pinned C++ reference build/config manifest (threads, deterministic settings, `score_t` width) for valid comparison
- [ ] **ORA-03**: Per-stage parity tests (bin → histogram → per-split-gain → leaf-output → prediction), not just final outputs
- [ ] **ORA-04**: Oracle suite executes and passes on the ROCm backend (mandated test environment)

### APIs

- [ ] **API-01**: Rust-native API — `Dataset`, `Booster`, `train`, `predict` mirroring LightGBM semantics
- [ ] **PYB-01**: Python bindings (PyO3 + maturin) mirroring the official `lightgbm` `Booster`/`Dataset` API
- [ ] **PYB-02**: NumPy interop (rust-numpy) for dense/sparse input and array outputs
- [ ] **PYB-03**: Python sklearn-style wrapper API (`LGBMClassifier`/`LGBMRegressor`/`LGBMRanker`) parity
- [ ] **PYB-04**: Python `custom` objective/metric callbacks and `Booster.refit()` support

## v2 Requirements

Deferred — parallel code paths or convenience features not on the v1 parity spine.

### Performance / Training Paths

- **QNT-01**: Quantized / discretized gradient training (`use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`, `stochastic_rounding`)
- **LIN-01**: Linear-tree leaves (`linear_tree`, `linear_lambda`, per-leaf least squares)

### Data Ingestion

- **ING-01**: Text-file ingestion (CSV/TSV/LibSVM)
- **ING-02**: Binary dataset cache (`save_binary`)
- **ING-03**: Arrow zero-copy columnar ingestion

## Out of Scope

| Feature | Reason |
|---------|--------|
| Distributed / network (MPI) training | Large surface (collectives, allreduce determinism); not needed for single-machine parity. Design `TreeLearner` trait so a parallel impl can slot in later |
| C ABI parity (`LGBM_*`, c_api.h) | Rust-native + Python cover v1 consumers; stable-ABI handle lifetime is its own project |
| CLI application (`lightgbm` binary) | Library-first; thin CLI can wrap the Rust API later |
| Raw CUDA backend (`.cu` kernels) | Superseded by CubeCL mandate; CUDA sources retained as kernel-design reference only |
| Raw OpenCL backend (`.cl` kernels) | Superseded by CubeCL; retained as design reference |
| R bindings | Out of scope per project goals |
| NVIDIA-CUDA-specific tuning | ROCm is the mandated GPU target; CUDA tuning would fork the codebase |

## Traceability

Populated during roadmap creation (each requirement maps to exactly one phase).

| Requirement | Phase | Status |
|-------------|-------|--------|
| _(filled by roadmapper)_ | — | Pending |

**Coverage:**
- v1 requirements: 58 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 58 ⚠️

---
*Requirements defined: 2026-06-05*
*Last updated: 2026-06-05 after initial definition*
