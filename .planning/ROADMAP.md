# Roadmap: LightGBM-rs

## Overview

A pure-Rust, parity-faithful port of Microsoft LightGBM on a CubeCL CPU/ROCm backend, built bottom-up along a dependency-forced spine so numerical fidelity is provable at every layer. Data types are `f32` (single-precision) end-to-end to match the C++ reference defaults, and the oracle tolerance is ~1e-6 absolute. The journey starts by pinning the oracle contract and the foundations (bit-exact RNG, f32 numerical strategy, config) that everything downstream is validated against, then locks the binning determinism root, then proves prediction parity against a C++-trained model *before* training exists. It next builds the f32 compute backend (the CubeCL-churn containment boundary), the histogram tree learner (the keystone FP-parity subsystem), and finally the GBDT loop with core objectives/metrics — the first moment a full train→predict run hits ~1e-6 (f32) parity. The remaining boosting variants, objectives, metrics, constraints, and SHAP are thin additions on the proven spine, and Python bindings land last as a translation layer over a validated Rust facade. Each phase is a vertical, oracle-validated slice: working numerical parity widens outward from binning → prediction → training rather than being deferred to the end.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Oracle Contract + Foundations** - f32 ~1e-6 oracle, pinned C++ reference, bit-exact RNG, config, f32 numerical strategy, workspace
- [x] **Phase 2: Dataset + Binning (determinism root)** - Bit-identical BinMapper, columnar bin store, missing/categorical encoding, EFB, metadata, ingestion *(7/7 plans executed incl. gap-closure 02-06 + 02-07; GAP-1/GAP-2 + CR-01 + WR-01 closed — default ingest unified onto the faithful single C++ Dataset::Construct, trivial features dropped, grouping verified; ready to re-verify)* (completed 2026-06-05)
- [ ] **Phase 3: Tree Model + Model Text I/O + Predict Parity** - Load a C++-trained model and predict identically (parity before training exists)
- [ ] **Phase 4: Compute Backend (CPU-first f32 histograms → ROCm)** - Backend trait, f32 histogram/split/score kernels, CPU then ROCm, both at ~1e-6
- [ ] **Phase 5: Tree Learner + Split Finding** - Histogram serial learner, subtraction trick, leaf-wise growth, split-gain scan with per-split parity
- [ ] **Phase 6: GBDT Spine + Core Objectives/Metrics** - First end-to-end ~1e-6 (f32) train→predict with bagging, early stopping, Rust-native API
- [ ] **Phase 7: Parity-Completing Variants** - GOSS/DART/RF, categorical/EFB splits, remaining objectives/metrics, ranking, SHAP, monotone, refit, importance
- [ ] **Phase 8: Python Bindings** - PyO3 + numpy bindings mirroring the official `lightgbm` Booster/Dataset/sklearn API

## Phase Details

### Phase 1: Oracle Contract + Foundations

**Goal**: A falsifiable, f32 single-precision oracle contract (~1e-6 absolute) and the foundations (bit-exact RNG, f32 numerical strategy, config, workspace, pinned reference) that every later phase is validated against.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: FND-01, FND-02, FND-03, FND-04, CFG-01, CFG-02, CFG-03, ORA-01, ORA-02
**Success Criteria** (what must be TRUE):

  1. The oracle harness compares Rust output against a pinned, deterministic C++ LightGBM 4.6 reference (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, fixed seed, default `float` `score_t`/`label_t` width) at ~1e-6 absolute tolerance, and the reference build/config manifest is checked in and regenerates goldens idempotently.
  2. A user can run the ported `Random` LCG and reproduce a captured 100k-draw C++ sequence (`RandInt16`/`RandInt32`/`NextFloat`/`NextInt`/`Sample(N,K)` across the branch boundary) bit-for-bit, with `u32` wraparound and `f32` `NextFloat`.
  3. The Cargo workspace (loosely-coupled crates by responsibility) builds under edition 2024 with `Cargo.lock` and `rust-toolchain.toml` committed; `thiserror` domain errors exist at crate boundaries and `anyhow` propagates at app/test layers.
  4. A config struct accepts the ~110 in-scope hyperparameters, resolves aliases (`num_iteration`/`n_estimators`/`num_boost_round`, etc.) via a data table matching `config_auto.cpp`, and rejects invalid combos with typed `Result` errors mirroring C++ `Config::Set` CHECK constraints.
  5. The f32 single-precision data-type contract and ~1e-6 oracle tolerance (standard f32 histogram/score accumulations, no integer-quantized reduction strategy) is documented as a Key Decision in PROJECT.md so no later phase targets an unfalsifiable invariant.**Plans**: 3 plans (incl. 1 gap-closure)

**Wave 1**

  - [x] 01-01-PLAN.md — Walking-skeleton spine: virtual workspace + f32 types/errors + bit-exact Random LCG + oracle comparator + pinned C++ RNG golden/manifest

**Wave 2** *(blocked on Wave 1 completion)*

  - [x] 01-02-PLAN.md — Hand-ported flat Config: struct/defaults + verbatim alias table + seed derivation + typed CHECK validation + drift-checker

**Wave 3** *(gap closure — blocked on Wave 2; closes SC#4 / CFG-02 + CFG-03)*

  - [x] 01-03-PLAN.md — Gap closure: deterministic SortAlias alias-collision resolution (CR-02) + present()-routed seed/enum empty-is-absent (CR-01), each with regression tests

### Phase 2: Dataset + Binning (determinism root)

**Goal**: A binned, immutable columnar dataset whose bin boundaries and bin assignments are bit-identical to C++ — the determinism root every downstream split inherits.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: DAT-01, DAT-02, DAT-03, DAT-04, DAT-05, DAT-06, DAT-07, ORA-03
**Success Criteria** (what must be TRUE):

  1. `BinMapper::ValueToBin` (literal `(r+l-1)/2` + `<=` search, `max_bin`/`min_data_in_bin`/`bin_construct_sample_cnt`/`data_random_seed`) produces `bin_upper_bound_` arrays and edge-case bin indices (NaN per `MissingType`, `+0.0`/`-0.0`, on-boundary, out-of-range categorical) that match C++ golden snapshots exactly.
  2. A user can ingest dense and CSR/CSC sparse in-memory matrices and metadata (labels, weights, init_score, query/group boundaries) into a Dense/Sparse-bin columnar store that is immutable after finish-load.
  3. Missing-value handling (`use_missing`, `zero_as_missing`, `MissingType`) and categorical encoding (category→bin, low-frequency folding) route exactly as C++.
  4. Exclusive Feature Bundling (`enable_bundle`) reproduces C++ feature grouping bit-for-bit.
  5. Per-stage parity tests cover the bin granularity (bin boundaries + per-row bin assignment), localizing any divergence to binning before histograms exist.

**Plans**: 6 plans (incl. 1 gap-closure)

**Wave 1**

  - [x] 02-01-PLAN.md — Crate scaffold + golden-capture harness + numeric BinMapper (FindBin/ValueToBin) golden layers 1+2

**Wave 2** *(blocked on Wave 1)*

  - [x] 02-02-PLAN.md — Bin trait + DenseBin (incl. 4-bit) + SparseBin + FeatureGroup offsets/PushData + Dataset finish_load immutability

**Wave 3** *(blocked on Wave 2)*

  - [x] 02-03-PLAN.md — Categorical folding (category→bin) + missing-value routing golden parity (layers 1+3)

**Wave 4** *(blocked on Wave 3)*

  - [x] 02-04-PLAN.md — Metadata + from_mat/from_csr/from_csc ingestion + dense/CSR/CSC equivalence + example-dataset parity

**Wave 5** *(blocked on Wave 4; EFB sequenced last per MEDIUM-risk capture flag, has checkpoint)*

  - [x] 02-05-PLAN.md — Exclusive Feature Bundling (MultiValBin + FastFeatureBundling) group/offset golden parity (layer 3)

**Wave 6** *(gap closure — closes SC#1 / DAT-01 + SC#5 / ORA-03 bin stage / DAT-07; default-config scaled filter_cnt divergence, see 02-VERIFICATION.md)*

  - [x] 02-06-PLAN.md — Gap closure: scaled `filter_cnt = (min_data_in_leaf * total_sample_cnt) / num_rows` in a single source-of-truth helper (CR-01/IN-02) + default feature_pre_filter=true ingest parity golden that fails-before/passes-after (CR-02) — DONE; GAP-1/GAP-2 closed, workspace green, capture idempotent

**Wave 7** *(gap closure — closes CR-01 (default-ingest Construct divergence) + WR-01; restores DAT-07 / ORA-03 / DAT-01/02/05 at the determinism root, see 02-06-REVIEW.md / re-verification gaps_found)*

  - [x] 02-07-PLAN.md — Gap closure: default ingest unified onto the faithful single C++ `Dataset::Construct` (`construct_bundled`) — trivial features DROPPED (`used_feature_map_[real]=-1`), EfbSamples built to the exact c_api.cpp:1352-1374 sampled-set convention (no second RNG draw); golden emitter models C++ Construct (trivial dropped, per-non-trivial group/subfeature via in-file FastFeatureBundling, is_sparse=true); parity test asserts trivial-exclusion + per-non-trivial group/subfeature parity + bit-exact stored bins, panics on missing golden (WR-01), HARD fails-before/passes-after — DONE; CR-01 + EFB parity hole + masking + WR-01 closed, workspace green, capture idempotent

### Phase 3: Tree Model + Model Text I/O + Predict Parity

**Goal**: Load a C++-trained model and predict identically — prediction parity proven independently of (and before) any training code.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DAT-08, DAT-09, PRD-01, PRD-02, PRD-03, PRD-06
**Success Criteria** (what must be TRUE):

  1. A user can load a C++-trained LightGBM `.txt` model and produce raw-score predictions within ~1e-6 (f32) of the C++ reference on the deterministic CPU path.
  2. Transformed predictions (`ConvertOutput` sigmoid/softmax) and leaf-index predictions (`pred_leaf`) match the C++ reference.
  3. The Rust writer emits the exact LightGBM text schema (tree structure, leaf values, bin mappers, feature metadata) including `%.17g` float formatting, and a load→predict→write→reload round-trip is byte-stable.
  4. Sub-range prediction (`start_iteration` / `num_iteration`) returns the C++-matching slice of the ensemble.

**Plans**: 4 plans

**Wave 1** *(enabling slice — crate + %.17g formatter + golden-capture pipeline)*

  - [x] 03-01-PLAN.md — lgbm-model crate skeleton + ModelError + `%.17g`/`{:g}` formatter (the DAT-09 linchpin, built FIRST) + `xtask model-capture` committed golden corpus (capture-path decision gate)

**Wave 2** *(blocked on Wave 1)*

  - [ ] 03-02-PLAN.md — Regression slice: load→raw-predict (dense/CSR/CSC, f64 accumulate)→write byte-exact→reload (DAT-08/DAT-09/PRD-01) — faithful array Tree + GbdtModel + model_text envelope + predict driver

**Wave 3** *(blocked on Wave 2)*

  - [ ] 03-03-PLAN.md — Transform + leaf-index slice: core ConvertOutput (sigmoid/softmax/ova/identity) + multiclass per-class stride + pred_leaf + categorical-split parity (PRD-02/PRD-03)

**Wave 4** *(blocked on Wave 3)*

  - [ ] 03-04-PLAN.md — Sub-range slice: InitPredict `start_iteration`/`num_iteration` (`-1==all`) parity (PRD-06) — full D-06 layered battery (1-5) green

**UI hint**: no

### Phase 4: Compute Backend (CPU-first integer histograms → ROCm)

**Goal**: An isolated `lgbm-compute` backend whose f32 histogram, split-scan, and score-update kernels produce results matching CPU and ROCm within ~1e-6 — the CubeCL-churn containment boundary.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: CMP-01, CMP-02, CMP-03, CMP-04, CMP-05, ORA-04
**Success Criteria** (what must be TRUE):

  1. All CubeCL usage lives behind one `lgbm-compute` `Backend` trait; no crate above it names a CubeCL runtime, and a CPU-only build needs no ROCm toolchain.
  2. Standard f32 histogram construction, best-split-finding, and data-partition kernels run on the cubecl-cpu reference path and produce results matching a sequential f32 CPU reference within ~1e-6.
  3. The same kernels run on the cubecl-hip (ROCm) backend, selectable by Cargo feature and/or runtime config, and produce results matching the CPU backend within ~1e-6 (f32).
  4. CUDA warp-level reductions are expressed via CubeCL's `Plane` API with startup capability-gating (`Plane::Ops`, f64, atomics) and a deterministic sequential fallback when a capability is absent.
  5. The oracle suite executes and passes on the ROCm backend for the histogram/split/partition kernels (mandated test environment), with CPU-runtime and ROCm treated as separate gates.

**Plans**: TBD

### Phase 5: Tree Learner + Split Finding

**Goal**: A histogram-based serial tree learner that grows the exact same tree as C++ — the keystone, highest-FP-risk subsystem, validated at per-split granularity.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: TRL-01, TRL-02, TRL-03, TRL-04, TRL-05, TRL-07, TRL-08, TRL-09
**Success Criteria** (what must be TRUE):

  1. Given fixed gradients/hessians, the learner (`ConstructHistograms` → `FindBestSplitsFromHistograms` → `Split`) selects the same split feature, split bin/threshold, and missing-direction as C++ for every split, validated against per-split candidate-gain snapshots (not just the winner).
  2. The histogram-subtraction trick reproduces the C++ smaller-child selection and derived-child histogram (matching the C++ f32 path within ~1e-6), and the default-bin-skip scan considers the same candidate-threshold set.
  3. Leaf-wise (best-first) growth respects `num_leaves`/`max_depth`, and the split-gain formula matches C++ (`kEpsilon` positions, `lambda_l1`/`lambda_l2`/`min_gain_to_split`/`min_sum_hessian_in_leaf`/`min_data_in_leaf`/`max_delta_step`/`path_smooth`).
  4. Numerical threshold splits route missing/zero exactly as C++; data partition (row→leaf) feeds the subtraction trick correctly.
  5. Per-tree/per-node feature subsampling (`feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed`) selects the same features via RNG parity, and both `force_row_wise`/`force_col_wise` strategies produce matching trees.

**Plans**: TBD

### Phase 6: GBDT Spine + Core Objectives/Metrics

**Goal**: The first end-to-end ~1e-6 (f32) train→predict run — the simplest boosting variant proves the full spine before any variant is added.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: BST-01, BST-02, BST-03, BST-07, OBJ-01, OBJ-02, OBJ-03, MET-01, MET-02, API-01
**Success Criteria** (what must be TRUE):

  1. A user can call the Rust-native API (`Dataset`, `Booster`, `train`, `predict`) to train a GBDT model and predict, with outputs within ~1e-6 (f32) of the C++ reference and a same-tree structural match on every backend.
  2. The GBDT loop (`TrainOneIter`, `UpdateScore`, per-class trees, shrinkage, `boost_from_average`) and score updater accumulate with deterministic reduction ordering.
  3. Core objectives (`regression`, `regression_l1`, `binary`, `multiclass`, `multiclassova`, `custom`) compute grad/hess, `ConvertOutput`, `BoostFromScore`, and `reg_sqrt` to within ~1e-6 (f32) of the reference.
  4. Core metrics (`l1`, `l2`, `rmse`, `binary_logloss`, `binary_error`, `auc`, `multi_logloss`) plus multi-metric infrastructure (`metric_freq`, training-metric eval) match the reference, and early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`) fires identically.
  5. Bagging / row subsampling (`bagging_fraction`/`bagging_freq`/`bagging_seed`, pos/neg, `bagging_by_query`) selects the same rows via RNG-matching sequence and call order.

**Plans**: TBD

### Phase 7: Parity-Completing Variants

**Goal**: Complete full single-machine parity — every remaining boosting variant, objective, metric, constraint, and prediction mode lands as a thin, oracle-validated addition on the proven spine.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: BST-04, BST-05, BST-06, TRL-06, OBJ-04, OBJ-05, OBJ-06, MET-03, MET-04, PRD-04, PRD-05, ADV-01, ADV-02, ADV-03, ADV-04, ADV-05, ADV-06, ADV-07
**Success Criteria** (what must be TRUE):

  1. GOSS (`top_rate`/`other_rate` with matching gradient-magnitude sort + amplification), DART (`drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`/`drop_seed`), and Random Forest (averaged trees, mandatory bagging) each train models within parity of the C++ reference.
  2. Categorical splits (`SplitCategorical`: `max_cat_threshold`/`cat_smooth`/`min_data_per_group`/`max_cat_to_onehot`/`cat_l2`) produce matching category bitsets, gains, and model-text round-trip.
  3. Remaining regression objectives (`huber`/`fair`/`poisson`/`quantile`/`mape`/`gamma`/`tweedie`), cross-entropy objectives, and ranking objectives (`lambdarank`/`rank_xendcg` with query boundaries + DCGCalculator + `objective_seed`) match the reference; extended + ranking metrics (`ndcg`/`map`/`average_precision`/`auc_mu`/...) match per-query.
  4. SHAP/feature contributions (`predict_contrib` over full node/cover structure) and prediction early stopping (`pred_early_stop`/`_freq`/`_margin`) produce C++-matching outputs.
  5. Monotone constraints (basic/intermediate/advanced + `monotone_penalty`), interaction constraints, forced splits/bins, extra trees, CEGB, refit/continue training (`Booster.refit()`), and feature importance reporting each reproduce the C++ behavior.

**Plans**: TBD

### Phase 8: Python Bindings

**Goal**: A Python interface mirroring the official `lightgbm` package, layered over the validated Rust facade.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: PYB-01, PYB-02, PYB-03, PYB-04
**Success Criteria** (what must be TRUE):

  1. A Python user can train and predict through PyO3 + maturin bindings whose `Booster`/`Dataset` API mirrors the official `lightgbm` package, releasing the GIL (`allow_threads`) around training and returning owned arrays.
  2. NumPy interop (rust-numpy) accepts both f32 and f64 dense/sparse input and returns array outputs, with contiguity/dtype handled explicitly so results match the C++ Python package for either width.
  3. The sklearn-style wrapper API (`LGBMClassifier`/`LGBMRegressor`/`LGBMRanker`) matches the official wrappers' semantics.
  4. Python `custom` objective/metric callbacks and `Booster.refit()` work and reproduce reference outputs.

**Plans**: TBD
**UI hint**: no

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Oracle Contract + Foundations | 3/3 | Plans complete | 2026-06-05 |
| 2. Dataset + Binning | 7/7 | Complete    | 2026-06-05 |
| 3. Tree Model + Model Text I/O + Predict Parity | 1/4 | In Progress|  |
| 4. Compute Backend (CPU-first → ROCm) | 0/TBD | Not started | - |
| 5. Tree Learner + Split Finding | 0/TBD | Not started | - |
| 6. GBDT Spine + Core Objectives/Metrics | 0/TBD | Not started | - |
| 7. Parity-Completing Variants | 0/TBD | Not started | - |
| 8. Python Bindings | 0/TBD | Not started | - |
