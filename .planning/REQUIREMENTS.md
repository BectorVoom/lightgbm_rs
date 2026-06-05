# Requirements: LightGBM-rs

**Defined:** 2026-06-05
**Core Value:** For identical inputs and config, reproduce C++ LightGBM outputs to within ~1e-6 absolute difference on every backend (CPU and ROCm), using `f32` (single-precision) data types end-to-end to match the C++ reference defaults (`score_t`/`label_t` = `float`).

> **Scope:** v1 = full single-machine parity with C++ LightGBM (GBDT/DART/RF/GOSS, all objectives + metrics, categorical, monotone, SHAP), exposed via a Rust-native API and Python bindings, on a switchable CubeCL CPU/ROCm backend. The ~1e-6 (f32) oracle is a hard merge gate on every backend. Distributed training, C ABI, CLI, and raw CUDA/OpenCL are out of scope.

## v1 Requirements

### Foundations & Determinism

- [x] **FND-01**: Port LightGBM's `Random` PRNG (32-bit LCG, `NextFloat`, `Sample(N,K)`) bit-for-bit, unit-tested against a captured C++ draw sequence
- [x] **FND-02**: Establish workspace crate structure (loosely-coupled crates by responsibility) building under edition 2024
- [x] **FND-03**: Use `f32` (single-precision) data types end-to-end (gradients, hessians, leaf values, scores) matching C++ defaults, with standard `f32` histogram/score accumulations on CPU and ROCm; outputs match the C++ reference within ~1e-6 (no integer-quantized reduction strategy)
- [x] **FND-04**: `thiserror` domain error types at every crate boundary; `anyhow` propagation in application/test layers

### Configuration

- [x] **CFG-01**: Config struct accepting the ~110 in-scope single-machine hyperparameters
- [x] **CFG-02**: Parameter alias resolution (e.g. `num_iteration`/`n_estimators`/`num_boost_round`) as a data table matching `config_auto.cpp`
- [x] **CFG-03**: Parameter validation mirroring C++ `Config::Set` CHECK constraints, surfaced as typed `Result` errors

### Dataset, Binning & I/O

- [x] **DAT-01**: `BinMapper` continuous→bin mapping (`FindBin`) producing bit-identical bin boundaries vs C++ (`max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, `data_random_seed`)
- [x] **DAT-02**: Binned columnar dataset store (DenseBin + SparseBin) immutable after finish-load
- [x] **DAT-03**: Missing-value handling (`use_missing`, `zero_as_missing`, `MissingType`) with C++-matching default-direction routing
- [x] **DAT-04**: Categorical feature encoding (category→bin mapping, low-frequency folding)
- [ ] **DAT-05**: Exclusive Feature Bundling (`enable_bundle`) reproducing C++ feature grouping
- [x] **DAT-06**: Metadata support (labels, weights, init_score, query/group boundaries)
- [x] **DAT-07**: In-memory matrix ingestion (dense + CSR/CSC sparse) via the Rust API
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
- [ ] **OBJ-03**: Objective machinery — `GetGradients`, `ConvertOutput` (sigmoid/softmax/exp), `BoostFromScore`, `reg_sqrt` — within ~1e-6 (f32)
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
- [ ] **CMP-05**: GPU-resident histogram construction, best-split finding, and data partition kernels meeting the ~1e-6 (f32) contract

### Oracle & Validation

- [x] **ORA-01**: Oracle harness comparing Rust vs C++ LightGBM outputs at ≤~1e-6 absolute (f32 single-precision)
- [x] **ORA-02**: Pinned C++ reference build/config manifest (threads, deterministic settings, default `float` `score_t`/`label_t` width) for valid comparison
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

Each v1 requirement maps to exactly one phase (see `.planning/ROADMAP.md`).

| Requirement | Phase | Status |
|-------------|-------|--------|
| FND-01 | Phase 1 | Complete |
| FND-02 | Phase 1 | Complete |
| FND-03 | Phase 1 | Complete |
| FND-04 | Phase 1 | Complete |
| CFG-01 | Phase 1 | Complete |
| CFG-02 | Phase 1 | Complete |
| CFG-03 | Phase 1 | Complete |
| ORA-01 | Phase 1 | Complete |
| ORA-02 | Phase 1 | Complete |
| DAT-01 | Phase 2 | Complete (02-01) |
| DAT-02 | Phase 2 | Complete (02-02) |
| DAT-03 | Phase 2 | Complete (02-03) |
| DAT-04 | Phase 2 | Complete (02-03) |
| DAT-05 | Phase 2 | Pending |
| DAT-06 | Phase 2 | Complete |
| DAT-07 | Phase 2 | Complete |
| ORA-03 | Phase 2 | Pending |
| DAT-08 | Phase 3 | Pending |
| DAT-09 | Phase 3 | Pending |
| PRD-01 | Phase 3 | Pending |
| PRD-02 | Phase 3 | Pending |
| PRD-03 | Phase 3 | Pending |
| PRD-06 | Phase 3 | Pending |
| CMP-01 | Phase 4 | Pending |
| CMP-02 | Phase 4 | Pending |
| CMP-03 | Phase 4 | Pending |
| CMP-04 | Phase 4 | Pending |
| CMP-05 | Phase 4 | Pending |
| ORA-04 | Phase 4 | Pending |
| TRL-01 | Phase 5 | Pending |
| TRL-02 | Phase 5 | Pending |
| TRL-03 | Phase 5 | Pending |
| TRL-04 | Phase 5 | Pending |
| TRL-05 | Phase 5 | Pending |
| TRL-07 | Phase 5 | Pending |
| TRL-08 | Phase 5 | Pending |
| TRL-09 | Phase 5 | Pending |
| BST-01 | Phase 6 | Pending |
| BST-02 | Phase 6 | Pending |
| BST-03 | Phase 6 | Pending |
| BST-07 | Phase 6 | Pending |
| OBJ-01 | Phase 6 | Pending |
| OBJ-02 | Phase 6 | Pending |
| OBJ-03 | Phase 6 | Pending |
| MET-01 | Phase 6 | Pending |
| MET-02 | Phase 6 | Pending |
| API-01 | Phase 6 | Pending |
| BST-04 | Phase 7 | Pending |
| BST-05 | Phase 7 | Pending |
| BST-06 | Phase 7 | Pending |
| TRL-06 | Phase 7 | Pending |
| OBJ-04 | Phase 7 | Pending |
| OBJ-05 | Phase 7 | Pending |
| OBJ-06 | Phase 7 | Pending |
| MET-03 | Phase 7 | Pending |
| MET-04 | Phase 7 | Pending |
| PRD-04 | Phase 7 | Pending |
| PRD-05 | Phase 7 | Pending |
| ADV-01 | Phase 7 | Pending |
| ADV-02 | Phase 7 | Pending |
| ADV-03 | Phase 7 | Pending |
| ADV-04 | Phase 7 | Pending |
| ADV-05 | Phase 7 | Pending |
| ADV-06 | Phase 7 | Pending |
| ADV-07 | Phase 7 | Pending |
| PYB-01 | Phase 8 | Pending |
| PYB-02 | Phase 8 | Pending |
| PYB-03 | Phase 8 | Pending |
| PYB-04 | Phase 8 | Pending |

**Coverage:**

- v1 requirements: 69 total (the prior "58" headline was a stale count; the enumerated REQ-ID list contains 69 distinct IDs)
- Mapped to phases: 69 ✓
- Unmapped: 0 ✓
- v2 (not mapped): QNT-01, LIN-01, ING-01, ING-02, ING-03

**Per-phase counts:** P1=9, P2=8, P3=6, P4=6, P5=8, P6=10, P7=18, P8=4 (= 69).

---
*Requirements defined: 2026-06-05*
*Last updated: 2026-06-05 — numerical contract revised to f32 / ~1e-6 (Phase 1 discuss)*
