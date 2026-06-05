# Feature Research

**Domain:** Gradient-boosting decision-tree library (pure-Rust port of Microsoft LightGBM, single-machine parity)
**Researched:** 2026-06-05
**Confidence:** HIGH (grounded directly in the C++ reference subsystems under `LightGBM/src/` and `LightGBM/include/`)

> **Scope frame:** v1 = full single-machine parity with C++ LightGBM. "Table stakes" = features required to claim faithful single-machine parity (without them, a LightGBM user's workflow breaks or outputs diverge). "Differentiators" = capabilities that distinguish this port from a naive GBDT crate but are not strictly the day-one critical path. "Anti-features" = subsystems deliberately excluded from v1 (distributed, C ABI, CLI, raw CUDA/OpenCL, R). All findings cite the actual reference subsystem, not generic GBDT knowledge.
>
> **Parity contract reminder:** every in-scope feature must reproduce C++ output to within 1e-12. That raises the effective complexity of *every* numeric feature one notch above what a from-scratch implementation would need, because bit-level reduction ordering and transform formulas must match exactly.

---

## Feature Landscape

### Table Stakes (Required for Single-Machine Parity)

These are the features a LightGBM user assumes exist. Missing any of them means the port cannot load a model, reproduce a training run, or evaluate a standard config — i.e. parity fails.

#### Boosting / ensemble layer (`src/boosting/`)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **GBDT loop** (`gbdt.cpp` `TrainOneIter`, `Boosting`, `UpdateScore`) | The default `boosting=gbdt`; the spine everything else hangs off | HIGH | Owns models, score updaters, sample strategy, per-class trees, shrinkage, `boost_from_average`, early stopping. Build this first; all other boosting variants subclass it. |
| **DART** (`dart.hpp`) | `boosting=dart`; common in tuned models | MEDIUM | Drops trees each iter (`drop_rate`, `max_drop`, `skip_drop`, `uniform_drop`, `xgboost_dart_mode`), renormalizes. Subclass of GBDT; needs drop RNG (`drop_seed`) to match exactly. |
| **Random Forest** (`rf.hpp`) | `boosting=rf` | LOW–MEDIUM | GBDT with no shrinkage accumulation, averaged trees, mandatory bagging. Thin subclass; cheap once GBDT + bagging exist. |
| **Bagging / row subsampling** (`bagging.hpp`, `sample_strategy.cpp`) | `bagging_fraction`/`bagging_freq`; ubiquitous in real configs | MEDIUM | Per-iteration row sampling with `bagging_seed`; also `pos_/neg_bagging_fraction` (binary) and `bagging_by_query` (ranking). RNG sequence must match C++ to keep parity. |
| **GOSS sample strategy** (`goss.hpp`) | `data_sample_strategy=goss` (and legacy `boosting=goss`) | MEDIUM–HIGH | Keeps `top_rate` largest-|gradient| rows + samples `other_rate` of the rest with gradient amplification. **Depends on gradient magnitude sorting** each iteration — ordering and the amplification factor `(1-top_rate)/other_rate` must match bit-for-bit. |
| **Score updater** (`score_updater.hpp`) | Internal accumulator for ensemble scores | LOW | Add-tree-into-score; a primary 1e-12 reduction-ordering risk on GPU. |
| **Shrinkage / learning rate** (`Tree::Shrinkage`) | `learning_rate` | LOW | Per-tree leaf scaling; trivial but must apply in the same place as C++. |
| **Early stopping** (`EvalAndCheckEarlyStopping`) | `early_stopping_round`, `first_metric_only`, `early_stopping_min_delta` | LOW–MEDIUM | Needs metric eval on validation sets; depends on Metric layer. |

#### Tree learner (`src/treelearner/serial_tree_learner.cpp`)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Histogram-based serial tree learner** | The core algorithm; defines LightGBM | VERY HIGH | `ConstructHistograms` → `FindBestSplitsFromHistograms` → `Split` loop. The single hardest, highest-risk subsystem for 1e-12 parity (histogram accumulation order, f64 vs f32). Everything else is comparatively easy. |
| **Leaf-wise (best-first) growth** | LightGBM's signature vs level-wise | MEDIUM | `ArrayArgs::ArgMax` over leaf split candidates up to `num_leaves-1`; `max_depth` cap. Logic is simple; correctness rides on split-gain values. |
| **Split-gain scan** (`feature_histogram.{cpp,hpp}` `FindBestThreshold`) | Decides every split | HIGH | Gain formula with `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `min_sum_hessian_in_leaf`, `min_data_in_leaf`, `max_delta_step`, `path_smooth`. Exact formula + tie-breaking must match. |
| **Data partition** (`data_partition.hpp`) | Row→leaf routing after each split | MEDIUM | Re-partitions row indices per split; feeds histogram subtraction trick. |
| **Histogram subtraction trick** | Performance parity + identical FP path | MEDIUM | Larger child = parent − smaller child. Must be byte-identical to the direct path it replaces (it is *not* numerically identical to recomputing, so the choice itself is part of parity). |
| **Categorical splits** (`SplitCategorical`, `FindBestThresholdCategorical`, `BinType::CategoricalBin`) | `categorical_feature`; LightGBM's native categorical handling is a headline feature | HIGH | One-vs-rest many-vs-many split on sorted category gradient stats; governed by `max_cat_threshold`, `cat_smooth`, `min_data_per_group`, `max_cat_to_onehot`, `cat_l2`. Distinct gain code path from numerical. |
| **Numerical threshold splits + missing routing** | Default behavior | MEDIUM | Default-left/right based on `use_missing`, `zero_as_missing`, `MissingType`. Routing of NaN/zero must match. |
| **Feature subsampling** (`col_sampler.hpp`) | `feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed` | LOW–MEDIUM | Per-tree and per-node column sampling; RNG must match. |
| **`min_data_in_leaf` / `min_sum_hessian_in_leaf` / `min_gain_to_split`** | Standard regularizers | LOW | Constraint checks inside split finding. |
| **`max_depth` + `num_leaves`** | Core capacity controls | LOW | Depth tracked during leaf-wise growth. |

#### Objective functions (`src/objective/`, factory `objective_function.cpp`)

Full registered set (verified from `CreateObjectiveFunction`). Common = appears in typical workflows; Rare = present for completeness.

| Objective | Family | Frequency | Complexity | Notes |
|-----------|--------|-----------|------------|-------|
| `regression` (l2) | regression | **Common** | LOW | Default objective. Baseline grad/hess. |
| `regression_l1` (l1/MAE) | regression | Common | LOW–MEDIUM | L1 uses approximate hessian; `RenewTreeOutput` refit. |
| `huber` | regression | Rare | LOW | `alpha` param. |
| `fair` | regression | Rare | LOW | `fair_c` param. |
| `poisson` | regression | Rare | MEDIUM | log-link, `poisson_max_delta_step`. |
| `quantile` | regression | Rare | MEDIUM | `alpha`; leaf-value renewal (quantile of residuals). |
| `mape` | regression | Rare | MEDIUM | mean absolute percentage error. |
| `gamma` | regression | Rare | LOW | deviance-based. |
| `tweedie` | regression | Rare | MEDIUM | `tweedie_variance_power`. |
| `binary` (logistic) | classification | **Common** | LOW–MEDIUM | `sigmoid`, `is_unbalance`, `scale_pos_weight`; `BoostFromScore` = log-odds init; `ConvertOutput`=sigmoid. |
| `multiclass` (softmax) | classification | **Common** | MEDIUM | `num_class` trees per iter; softmax `ConvertOutput`. |
| `multiclassova` (one-vs-all) | classification | Common | MEDIUM | `num_class` independent binary objectives. |
| `cross_entropy` | classification | Rare | MEDIUM | continuous [0,1] labels. |
| `cross_entropy_lambda` | classification | Rare | MEDIUM | alternative parameterization. |
| `lambdarank` | ranking | Common (in ranking domain) | HIGH | Needs query/group boundaries, `lambdarank_truncation_level`, `lambdarank_norm`, `lambdarank_position_bias_regularization`, sigmoid; pairwise lambda gradients depend on per-query NDCG deltas (DCGCalculator). |
| `rank_xendcg` | ranking | Rare | HIGH | Cross-entropy NDCG variant; stochastic, RNG-sensitive (`objective_seed`). |
| `custom` (user-supplied grad/hess) | any | Common (Python users) | LOW | Objective passes through externally provided gradients; needed for Python API parity. |

Cross-cutting objective machinery (table stakes regardless of which objective): `GetGradients`, `ConvertOutput` (sigmoid/softmax/exp links), `BoostFromScore` (`boost_from_average`), `reg_sqrt`. These three hooks must be exact for 1e-12.

#### Metrics (`src/metric/`, factory `metric.cpp`)

Full registered set (verified from `CreateMetric`):

| Metric | Family | Frequency | Complexity | Notes |
|--------|--------|-----------|------------|-------|
| `l2`, `rmse`, `l1` | regression | **Common** | LOW | rmse = sqrt(l2). |
| `quantile`, `huber`, `fair`, `poisson`, `mape`, `gamma`, `gamma_deviance`, `tweedie` | regression | Rare | LOW–MEDIUM | Mirror objective math. |
| `binary_logloss` | binary | **Common** | LOW | Default for `binary`. |
| `binary_error` | binary | Common | LOW | threshold 0.5. |
| `auc` | binary | **Common** | MEDIUM | rank-based; tie handling must match. |
| `average_precision` | binary | Rare | MEDIUM | |
| `auc_mu` | multiclass | Rare | HIGH | multiclass AUC generalization; matrix-weighted. |
| `multi_logloss` | multiclass | **Common** | LOW | |
| `multi_error` | multiclass | Common | LOW | `multi_error_top_k`. |
| `cross_entropy`, `cross_entropy_lambda`, `kullback_leibler` | xentropy | Rare | LOW–MEDIUM | |
| `ndcg` | ranking | **Common (ranking)** | HIGH | `DCGCalculator` static gain/discount tables (`include/LightGBM/metric.h`); `eval_at` / `ndcg_eval_at` positions; per-query. |
| `map` | ranking | Rare | MEDIUM–HIGH | `map_metric.hpp`; per-query average precision. |

Metric infra (table stakes): multi-`metric` lists, per-`eval_at` cutoffs, query-group awareness, `metric_freq`, `is_provide_training_metric`. Static `DCGCalculator` tables must reproduce identical gains/discounts.

#### Dataset / IO (`src/io/`)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Binned columnar store** (`dataset.cpp`, `FeatureGroup`) | The in-memory model everything reads | HIGH | Dataset immutable after `FinishLoad`; feature grouping/bundling (EFB). |
| **BinMapper** (`bin.cpp` `FindBin`) | Continuous→bin mapping defines every split threshold | VERY HIGH | `max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, sampling RNG (`data_random_seed`). **Binning must be bit-identical** or every downstream split diverges. Highest-leverage parity risk after histograms. |
| **DenseBin / SparseBin** (`dense_bin.hpp`, `sparse_bin.hpp`) | Storage + `ConstructHistogram` | HIGH | The hottest kernels; templated over bit-width in C++. GPU-relevant. |
| **Missing-value handling** (`MissingType`, `use_missing`, `zero_as_missing`) | NaN/zero semantics | MEDIUM | Must match default-direction logic in splits. |
| **Categorical encoding** (`BinType::CategoricalBin`) | `categorical_feature` | MEDIUM | Category→bin mapping, low-frequency folding. |
| **Exclusive Feature Bundling** (`enable_bundle`) | On by default; affects feature grouping → histogram layout | HIGH | Sparse features bundled into one group. Bundling decisions change histogram construction; must reproduce grouping to keep splits identical. |
| **Model text format read/write** (`gbdt_model_text.cpp`, `tree.cpp`) | Load a C++-trained model and predict identically; save Rust-trained model | HIGH | Explicit project requirement. Must parse/emit exact LightGBM text schema (tree structure, leaf values, bin mappers, feature names, pandas categorical metadata). |
| **Metadata** (`metadata.cpp`) | labels, weights, init_score, query/group boundaries | MEDIUM | Ranking + weighted training need these. |
| **In-memory matrix ingestion** | Python/ndarray is the primary input | MEDIUM | Dense `mat`, CSR/CSC sparse construction (mirrors `LGBM_DatasetCreateFromMat/CSR/CSC` semantics but via Rust API, not C ABI). |

> CSV / LibSVM / TSV file parsing (`parser.cpp`, `dataset_loader.cpp`) is **lower priority** — Python/ndarray input is primary. Treat text-file parsing as a differentiator, not table stakes (see below).

#### Prediction (`src/boosting/gbdt_prediction.cpp`, `tree.cpp`)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Raw score prediction** (`PredictRaw`) | Sum of tree outputs | LOW–MEDIUM | Iterate `models_`, sum `Tree::Predict`. |
| **Transformed prediction** (`ConvertOutput`) | sigmoid/softmax outputs | LOW | Objective-dependent. |
| **Leaf index prediction** (`predict_leaf_index`) | `pred_leaf` in Python | LOW | Returns leaf id per tree. |
| **Feature contributions / TreeSHAP** (`tree.cpp` `TreeSHAP`, `UnwoundPathSum`) | `predict_contrib`; widely used for explainability | HIGH | Independent SHAP path-dependent algorithm over tree structure. **Depends on full Tree node/split/cover structure** being stored. Algorithmically intricate; exact float parity needed. |
| **Prediction early stopping** (`prediction_early_stop.cpp`) | `pred_early_stop`, `_freq`, `_margin` | LOW–MEDIUM | Stop summing trees when margin is decisive. |
| **`start_iteration` / `num_iteration` prediction** | Predict with a sub-range of trees | LOW | Slice of `models_`. |

#### Config surface (`include/LightGBM/config.h`, `config_auto.cpp`)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Config struct (~110 in-scope hyperparameters)** | Compatibility: configs from existing users must be accepted | HIGH (breadth) | Single-machine in-scope fields enumerated in the appendix below. Includes core (`num_iterations`, `learning_rate`, `num_leaves`, `max_depth`), tree (`min_data_in_leaf`, `lambda_l1/l2`, `min_gain_to_split`, `max_bin`), sampling (bagging/feature/GOSS), objective/metric params (`alpha`, `sigmoid`, `tweedie_variance_power`, `num_class`...), DART, categorical, monotone, CEGB, prediction. |
| **Parameter aliasing** | LightGBM accepts many aliases (`num_iteration`/`n_estimators`/`num_boost_round`) | MEDIUM | `config_auto.cpp` is auto-generated alias map; port as a data table, not hand logic. Python parity requires aliases. |
| **Parameter validation** (`Config::Set`) | Reject invalid combos as C++ does | MEDIUM | `CHECK_*` constraints; mirror to Rust `Result`/`thiserror`. |

---

### Differentiators (Distinguish This Port; Not Day-One Critical Path)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Monotone constraints** (`monotone_constraints.hpp`, `monotone_type`) | Domain-required in finance/risk; a headline LightGBM feature | MEDIUM–HIGH | `monotone_constraints`, `monotone_constraints_method` (basic/intermediate/advanced), `monotone_penalty`. Constrains split selection; intermediate/advanced track per-feature min/max bounds through the tree. In-scope for "full parity" but isolable. |
| **Interaction constraints** (`interaction_constraints`) | Controls which features may co-occur in a tree path | MEDIUM | Restricts feature set per node based on path history. |
| **Forced splits / forced bins** (`forcedsplits_filename`, `forcedbins_filename`) | Reproduce models trained with forced structure | MEDIUM | JSON-driven; needed only for configs that use them. |
| **Extra trees** (`extra_trees`, `extra_seed`) | Randomized thresholds (ExtraTrees-style) | LOW–MEDIUM | Random split threshold instead of best; RNG must match. |
| **Quantized / discretized gradient training** (`gradient_discretizer.{cpp,hpp}`, `use_quantized_grad`) | Faster training; recent LightGBM feature | HIGH | `num_grad_quant_bins`, `quant_train_renew_leaf`, `stochastic_rounding`. Adds an entire int-histogram code path (16/32-bit) parallel to the float path. **Depends on histogram learner** existing first; large surface; defer until float path is bit-exact. |
| **Linear-tree leaves** (`linear_tree_learner.{cpp,h}`, `linear_tree`) | Piecewise-linear leaf models | HIGH | Fits a linear model per leaf (`linear_lambda`, `leaf_coeff`). Separate learner subclass + per-leaf least-squares. Isolable; many users never enable it. |
| **CEGB (cost-effective gradient boosting)** (`cost_effective_gradient_boosting.hpp`) | Feature-cost-aware splitting | MEDIUM | `cegb_tradeoff`, `cegb_penalty_split`, per-feature/per-split penalties. Niche but part of full parity. |
| **Refit / continue training** (`refit_decay_rate`, `task=refit`, `input_model`) | Update leaf values on new data; warm start | MEDIUM | `GBDTBase` leaf get/set hooks. Needed for `Booster.refit()` Python parity. |
| **Feature importance** (`saved_feature_importance_type`) | split/gain importance reporting | LOW | Aggregate over trees; cheap once trees exist. |
| **CubeCL CPU↔ROCm backend** | The actual product differentiator vs C++ (memory-safe, portable GPU) | VERY HIGH | Cross-cutting; not a "feature" in the catalog sense but the project's reason to exist. Histogram construction, best-split finding, data partition are the kernels. |
| **`force_row_wise` / `force_col_wise` / `histogram_pool_size`** | Histogram build strategy + memory control | MEDIUM | Affects performance + the FP reduction path; must support both to match outputs under each setting. |
| **Text-file ingestion (CSV/TSV/LibSVM)** (`parser.cpp`) | Convenience parity with C++ data loading | MEDIUM | De-prioritized (Python/ndarray primary), so it's a differentiator rather than table stakes for v1. |

---

### Anti-Features (Deliberately NOT in v1)

| Feature | Why Requested | Why Excluded from v1 | Alternative |
|---------|---------------|----------------------|-------------|
| **Distributed / network training** (`src/network/`, MPI/socket allreduce, `data_parallel_/feature_parallel_/voting_parallel_tree_learner`) | Scale to clusters | Large surface (collectives, topology, allreduce determinism); not needed to prove the architecture or single-machine parity | Defer to post-v1; design `TreeLearner` trait so a parallel impl can slot in later. |
| **C ABI parity** (`c_api.cpp`, `c_api.h`, ~120 `LGBM_*` functions) | C/R/other-language clients | Rust-native + Python bindings cover v1 consumers; stable-ABI handle lifetime is its own project | Provide Rust-native API + PyO3 bindings; revisit C ABI only if external clients demand it. |
| **CLI application** (`src/main.cpp`, `application.cpp`, `task=train/predict/convert_model`, config-file driven) | Command-line workflows | Not required for v1 compatibility goals; config-file parsing + I/O orchestration is large | Library-first; a thin CLI can wrap the Rust API later. |
| **Raw CUDA backend** (`src/**/cuda/`, `.cu` kernels) | NVIDIA GPU speed | Superseded by CubeCL mandate; ROCm is the GPU test target | CubeCL kernels (CPU + ROCm) replace both CUDA and OpenCL. CUDA sources remain *reference* for kernel design only. |
| **Raw OpenCL backend** (`gpu_tree_learner.cpp`, `ocl/*.cl`) | AMD/Intel GPU via OpenCL | Superseded by CubeCL | Same as above — `.cl` histogram kernels are design references for CubeCL. |
| **R bindings** | R users | Out of scope per PROJECT.md | None in v1. |
| **NVIDIA-CUDA-specific tuning** | Squeeze NVIDIA perf | ROCm is the mandated GPU target; tuning to CUDA hardware would fork the codebase | Tune CubeCL for ROCm; rely on portability for other backends. |
| **`save_binary` / binary dataset format** (`save_binary`) | Faster reload of binned data | Nice-to-have, not parity-critical; can defer | Recompute binning from source data in v1; add binary cache later if needed. |
| **Arrow ingestion** (`arrow.h`) | Zero-copy columnar input | Not required for v1; Python/ndarray covers input | Add later behind a feature flag. |

> **Important:** the CUDA `.cu` and OpenCL `.cl` sources are anti-features *as build targets* but **primary references** for CubeCL kernel design (histogram construction, best-split finding, data partition). Do not delete them from the reference tree.

---

## Feature Dependencies

```
Config struct + aliasing
    └──gates──> everything (validated parameters drive all subsystems)

Dataset / BinMapper (binning)        [VERY HIGH risk]
    └──requires──> DenseBin / SparseBin storage
    └──requires──> Missing-value + categorical encoding
    └──enables──> Exclusive Feature Bundling (EFB)
    └──feeds──> Histogram construction

Histogram-based serial tree learner  [VERY HIGH risk — core]
    ├──requires──> Dataset/Bin (ConstructHistogram)
    ├──requires──> Split-gain scan (feature_histogram)
    │                  └──requires──> categorical split path (separate)
    │                  └──enhanced by──> monotone constraints
    │                  └──enhanced by──> interaction constraints
    │                  └──enhanced by──> CEGB penalties
    ├──requires──> Data partition (row→leaf)
    ├──requires──> Histogram subtraction trick
    └──enhanced by──> feature subsampling (col_sampler)

GBDT loop
    ├──requires──> ObjectiveFunction.GetGradients (grad/hess)
    ├──requires──> TreeLearner.Train (one tree)
    ├──requires──> Metric.Eval (for early stopping)
    ├──requires──> ScoreUpdater
    ├──requires──> boost_from_average (BoostFromScore)
    └──subclassed by──> DART, RF

Sample strategies
    ├── Bagging ──requires──> RNG parity (bagging_seed)
    └── GOSS    ──requires──> per-iteration gradient-magnitude SORT
                              (depends on gradients already computed)

Ranking objectives (lambdarank / rank_xendcg)
    ├──requires──> query/group boundaries (Metadata)
    └──requires──> DCGCalculator (shared with ndcg/map metrics)

Prediction
    ├── raw/transformed ──requires──> trained models_ + ConvertOutput
    ├── leaf index ──requires──> Tree structure
    ├── SHAP/contrib ──requires──> full Tree node+cover structure
    └── prediction early stop ──requires──> raw prediction loop

Model text I/O
    ├──requires──> Tree serialization
    ├──requires──> BinMapper serialization
    └──requires──> Config/feature metadata serialization

Quantized gradient training  [defer]
    └──requires──> float histogram learner FIRST (parallel int code path)

Linear tree  [defer]
    └──requires──> serial tree learner + per-leaf least-squares
```

### Dependency Notes (build-order implications)

- **BinMapper before everything compute:** binning thresholds determine every split. If binning isn't bit-identical, no downstream parity is achievable. This is the first thing to lock, with its own oracle test against C++ `FindBin`.
- **Histogram learner is the keystone:** the most complex and highest-FP-risk subsystem. GBDT, all objectives, all metrics are comparatively cheap; budget the bulk of effort here. The histogram-subtraction trick is a *correctness* dependency, not just performance — its FP result is what the model is defined against.
- **GOSS depends on gradient ordering:** it sorts rows by |gradient| each iteration after `GetGradients`. Requires a deterministic sort matching C++ ties, and the amplification constant must match exactly. Build after GBDT + ObjectiveFunction.
- **SHAP depends on Tree structure:** `TreeSHAP` walks node split features, thresholds, and per-node cover (hessian sums). The Tree model must persist this metadata even if only used at predict time. Build after Tree serialization and basic prediction.
- **Ranking objectives + ndcg/map metrics share `DCGCalculator`:** build the DCG tables once; both consume them. Both also require query-group metadata, so Metadata must support `group_column`/query boundaries before ranking.
- **Quantized-gradient and linear-tree are parallel code paths**, not extensions of the float learner — they roughly double the learner surface. Defer both until the float path passes the 1e-12 oracle, then port each against C++ with `use_quantized_grad`/`linear_tree` enabled.
- **DART/RF are thin subclasses of GBDT** — cheap once GBDT + bagging exist, but DART's drop RNG (`drop_seed`) must match.
- **`custom` objective is required for Python parity** (users pass their own grad/hess); it's nearly free (pass-through) but unlocks a large fraction of real Python workflows.

---

## MVP Definition

> "MVP" here means the smallest slice that demonstrates *faithful single-machine parity* on a real LightGBM workflow, not a reduced-feature product. v1 ultimately requires the full table-stakes set; the staging below is the recommended build order.

### Launch With (v1 core — the parity spine)

- [ ] **Config struct + aliasing + validation** — gates everything; must accept existing configs.
- [ ] **Dataset + BinMapper + DenseBin/SparseBin** — bit-identical binning is the foundation of all parity.
- [ ] **Missing-value handling + numerical splits** — correct default routing.
- [ ] **Histogram-based serial tree learner** (construct, split-gain scan, data partition, subtraction trick, leaf-wise growth, `num_leaves`/`max_depth`/`min_data_in_leaf`/`lambda_l1/l2`/`min_gain_to_split`).
- [ ] **GBDT loop** + ScoreUpdater + shrinkage + `boost_from_average`.
- [ ] **Core objectives:** `regression`, `regression_l1`, `binary`, `multiclass`, `multiclassova`, `custom`.
- [ ] **Core metrics:** `l1`, `l2`, `rmse`, `binary_logloss`, `binary_error`, `auc`, `multi_logloss`.
- [ ] **Prediction:** raw, transformed, leaf index.
- [ ] **Model text format read/write** — load a C++-trained model and predict identically (explicit requirement).
- [ ] **Bagging + feature subsampling** — present in most real configs.
- [ ] **Early stopping.**
- [ ] **Oracle harness** comparing Rust vs C++ at 1e-12.

### Add After Core Is Bit-Exact (still v1 — completes parity)

- [ ] **GOSS** sample strategy (after gradients/sort are deterministic).
- [ ] **DART** and **RF** boosting variants.
- [ ] **Categorical splits + encoding** (`categorical_feature`, `max_cat_threshold`, `cat_smooth`...).
- [ ] **Exclusive Feature Bundling** (`enable_bundle`).
- [ ] **Remaining regression objectives:** `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`.
- [ ] **Cross-entropy objectives** + `cross_entropy`/`kullback_leibler` metrics.
- [ ] **Ranking:** `lambdarank`, `rank_xendcg` + `ndcg`, `map`, `average_precision`, `auc_mu` (DCGCalculator + query metadata).
- [ ] **Monotone constraints** (basic → intermediate → advanced).
- [ ] **SHAP / feature contributions** + prediction early stopping.
- [ ] **Feature importance**, **refit/continue training**, **interaction constraints**, **forced splits/bins**, **extra trees**, **CEGB**.
- [ ] **Python bindings** mirroring the official `lightgbm` API.

### Future Consideration (post-v1)

- [ ] **Quantized/discretized gradient training** (`use_quantized_grad`) — large parallel code path; defer until float path is locked.
- [ ] **Linear-tree leaves** (`linear_tree`) — separate learner subclass.
- [ ] **Text-file ingestion** (CSV/TSV/LibSVM) — Python/ndarray covers v1 input.
- [ ] **Binary dataset cache** (`save_binary`), **Arrow ingestion**.
- [ ] **Distributed training, C ABI, CLI, R bindings, raw CUDA/OpenCL** — explicit anti-features (see above).

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Config + aliasing + validation | HIGH | MEDIUM | P1 |
| Dataset + BinMapper + Dense/Sparse bins | HIGH | HIGH | P1 |
| Histogram serial tree learner | HIGH | HIGH | P1 |
| GBDT loop + score updater + shrinkage | HIGH | HIGH | P1 |
| Core objectives (l2/l1/binary/multiclass/custom) | HIGH | MEDIUM | P1 |
| Core metrics (l2/rmse/l1/logloss/auc/multi_logloss) | HIGH | MEDIUM | P1 |
| Raw/transformed/leaf prediction | HIGH | LOW | P1 |
| Model text read/write | HIGH | HIGH | P1 |
| Bagging + feature subsampling | HIGH | MEDIUM | P1 |
| Early stopping | HIGH | LOW | P1 |
| GOSS | MEDIUM | MEDIUM | P2 |
| DART / RF | MEDIUM | LOW | P2 |
| Categorical splits + EFB | HIGH | HIGH | P2 |
| Remaining regression objectives | MEDIUM | MEDIUM | P2 |
| Ranking (lambdarank/xendcg + ndcg/map) | MEDIUM | HIGH | P2 |
| Monotone constraints | MEDIUM | MEDIUM | P2 |
| SHAP / contributions | MEDIUM | HIGH | P2 |
| Python bindings | HIGH | MEDIUM | P2 |
| Feature importance / refit / interaction / forced / extra-trees / CEGB | LOW–MEDIUM | MEDIUM | P2 |
| Quantized gradient training | LOW | HIGH | P3 |
| Linear tree | LOW | HIGH | P3 |
| Text-file ingestion | LOW | MEDIUM | P3 |
| Distributed / C ABI / CLI / CUDA / OpenCL / R | (out of scope) | — | P3 (excluded) |

**Priority key:** P1 = parity spine, must come first. P2 = completes full single-machine parity (all still required for v1). P3 = post-v1 / excluded.

---

## Competitor Feature Analysis

| Feature | C++ LightGBM (reference) | XGBoost / generic GBDT crates | This Port (v1 plan) |
|---------|--------------------------|-------------------------------|---------------------|
| Histogram tree learner | Yes (templated, OpenMP) | Yes | Yes — CubeCL kernels, bit-deterministic |
| Native categorical splits | Yes (`SplitCategorical`) | Partial (XGBoost recent) | Yes (P2) |
| GOSS | Yes | No | Yes (P2) |
| DART | Yes | Yes | Yes (P2) |
| Monotone constraints | Yes (3 methods) | Yes (XGBoost) | Yes (P2) |
| SHAP/contrib | Yes (`TreeSHAP`) | Yes | Yes (P2) |
| Ranking (lambdarank/ndcg) | Yes | Yes | Yes (P2) |
| Quantized gradients | Yes | No | Defer (P3) |
| GPU backend | CUDA + OpenCL | CUDA | CubeCL CPU+ROCm (differentiator) |
| Memory safety | No (C++) | No | Yes (Rust) — core differentiator |
| 1e-12 cross-backend determinism | Not guaranteed across devices | Not guaranteed | **Guaranteed contract** (core value) |
| Distributed training | Yes | Yes | Out of scope (v1) |

The genuine differentiators vs the C++ reference are **memory safety**, a **portable CubeCL CPU↔ROCm backend** (replacing two separate CUDA/OpenCL codebases), and a **hard 1e-12 determinism contract on every backend** — not new ML capabilities. Everything else is parity.

---

## Appendix: In-Scope Config Surface (single-machine)

From `include/LightGBM/config.h` (~110 fields after excluding distributed/GPU-vendor/CLI-only). Grouped for roadmap reference. Out-of-scope fields (`num_machines`, `local_listen_port`, `time_out`, `machine_list_filename`, `machines`, `gpu_platform_id`, `gpu_device_id`, `gpu_use_dp`, `num_gpu`, `is_parallel`, `is_data_based_parallel`, `convert_model*`, `tree_learner` distributed values) are excluded.

- **Core:** `objective`, `boosting`, `data_sample_strategy`, `num_iterations`, `learning_rate`, `num_leaves`, `num_threads`, `device_type`, `seed`, `deterministic`.
- **Tree / regularization:** `max_depth`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `min_gain_to_split`, `lambda_l1`, `lambda_l2` (`reg_alpha`/`reg_lambda` aliases), `max_delta_step`, `path_smooth`, `histogram_pool_size`, `force_col_wise`, `force_row_wise`.
- **Sampling:** `bagging_fraction`, `pos_bagging_fraction`, `neg_bagging_fraction`, `bagging_freq`, `bagging_seed`, `bagging_by_query`, `feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed`, `extra_trees`, `extra_seed`.
- **GOSS:** `top_rate`, `other_rate`.
- **DART:** `drop_rate`, `max_drop`, `skip_drop`, `xgboost_dart_mode`, `uniform_drop`, `drop_seed`.
- **Categorical:** `min_data_per_group`, `max_cat_threshold`, `cat_l2`, `cat_smooth`, `max_cat_to_onehot`, `categorical_feature`.
- **Constraints / CEGB:** `monotone_constraints`, `monotone_constraints_method`, `monotone_penalty`, `interaction_constraints`, `forcedsplits_filename`, `cegb_tradeoff`, `cegb_penalty_split`, `top_k`.
- **Binning / dataset:** `max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, `data_random_seed`, `is_enable_sparse`, `enable_bundle`, `use_missing`, `zero_as_missing`, `feature_pre_filter`, `pre_partition`, `header`, `label_column`, `weight_column`, `group_column`, `ignore_column`, `forcedbins_filename`, `precise_float_parser`, `two_round`, `save_binary`.
- **Quantized / linear (defer):** `use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`, `stochastic_rounding`, `linear_tree`, `linear_lambda`.
- **Objective params:** `num_class`, `is_unbalance`, `scale_pos_weight`, `sigmoid`, `boost_from_average`, `reg_sqrt`, `alpha`, `fair_c`, `poisson_max_delta_step`, `tweedie_variance_power`, `lambdarank_truncation_level`, `lambdarank_norm`, `lambdarank_position_bias_regularization`, `objective_seed`.
- **Metric:** `metric`, `metric_freq`, `is_provide_training_metric`, `eval_at`/`ndcg_eval_at`, `multi_error_top_k`, `early_stopping_round`, `early_stopping_min_delta`, `first_metric_only`.
- **Prediction:** `start_iteration_predict`, `num_iteration_predict`, `predict_raw_score`, `predict_leaf_index`, `predict_contrib`, `predict_disable_shape_check`, `pred_early_stop`, `pred_early_stop_freq`, `pred_early_stop_margin`.
- **Model I/O:** `input_model`, `output_model`, `saved_feature_importance_type`, `snapshot_freq`, `refit_decay_rate`, `verbosity`.

---

## Sources

- `LightGBM/src/objective/objective_function.cpp` — full objective registry (`CreateObjectiveFunction`).
- `LightGBM/src/metric/metric.cpp` — full metric registry (`CreateMetric`).
- `LightGBM/src/boosting/{gbdt.cpp,dart.hpp,rf.hpp,goss.hpp,bagging.hpp,score_updater.hpp,prediction_early_stop.cpp}` — boosting/sample strategies.
- `LightGBM/src/treelearner/{serial_tree_learner.cpp,feature_histogram.{cpp,hpp},data_partition.hpp,col_sampler.hpp,monotone_constraints.hpp,cost_effective_gradient_boosting.hpp,gradient_discretizer.{cpp,hpp},linear_tree_learner.h}` — tree learning + variants.
- `LightGBM/src/io/{dataset.cpp,bin.cpp,dense_bin.hpp,sparse_bin.hpp,tree.cpp,metadata.cpp}` — dataset/binning/SHAP/model I/O.
- `LightGBM/include/LightGBM/{config.h,bin.h,boosting.h,tree.h}` — config surface, bin types, interfaces.
- `LightGBM/src/io/tree.cpp` `TreeSHAP`/`UnwoundPathSum` — SHAP implementation.
- `.planning/PROJECT.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` — scope boundaries and subsystem map.

---
*Feature research for: pure-Rust LightGBM single-machine port*
*Researched: 2026-06-05*
