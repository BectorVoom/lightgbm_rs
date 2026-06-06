# Phase 6: GBDT Spine + Core Objectives/Metrics - Research

**Researched:** 2026-06-07
**Domain:** Gradient-boosting orchestration loop (GBDT), objective grad/hess math, metric reductions, bagging RNG, early stopping, and the first Rust-native train→predict API — all as a faithful 1:1 mirror of LightGBM C++ 4.6 below the API boundary.
**Confidence:** HIGH (every FP-load-bearing claim is read directly from the in-tree `LightGBM/` C++ source, not training memory; capture mechanics verified against the existing Phase-5 pip-wheel oracle pipeline)

## Summary

Phase 6 wraps the bit-exact Phase-5 `SerialTreeLearner` in the GBDT boosting loop. The loop itself is small and almost entirely orchestration — the numerical risk lives in (a) the objective `GetGradients` formulas, (b) the f64 score accumulation order, (c) the bagging RNG draw/call sequence, and (d) the metric reductions feeding the early-stopping decision. Every one of these is read verbatim from `LightGBM/src/boosting/gbdt.cpp`, `src/boosting/score_updater.hpp`, `src/boosting/bagging.hpp`, `src/objective/*_objective.hpp`, and `src/metric/*_metric.hpp` below. The C++ source is authoritative over any inferred default (CONTEXT D-discretion clause).

The decisive control-flow facts: `TrainOneIter` (i) calls `BoostFromAverage` per class which adds the init score to **both** train and valid score updaters BEFORE any tree, (ii) calls `Boosting()` which calls `objective->GetGradients(train_score, grad, hess)`, (iii) calls `Bagging(iter, …)` (RNG draw), (iv) loops per `cur_tree_id` growing one tree via the Phase-5 learner, calls `RenewTreeOutput` (no-op for L2/binary/multiclass, active for regression_l1), applies `Shrinkage(learning_rate)`, then `UpdateScore`, then `AddBias(init_score)`. Score accumulation is **f64** throughout (`std::vector<double> score_`); the training path uses the learner's data-partition-based `AddPredictionToScore` (per-leaf scatter, not per-row predict), which is bit-exact-relevant. The CPU `cubecl-cpu` f64-fold anchor is the hard merge gate; this is the first phase where a full multi-iteration train→predict run is compared to the real `lib_lightgbm` 4.6 wheel.

**Primary recommendation:** Build a new `lgbm-boosting` crate (the GBDT loop + score updater + sample strategy) plus `lgbm-objective` and `lgbm-metric` crates (factory-shaped enums mirroring `CreateObjectiveFunction`/`CreateMetric`), and a thin umbrella `lgbm` facade crate hosting the D-01 builder + `Booster`/`Dataset`/`train`/`predict`. Resolve the builder to `lgbm-core::Config` (D-02). Sequence strictly spine-first per D-14→D-17. Capture the layered D-10..D-13 goldens from the pinned pip `lightgbm==4.6.0` wheel via an extended Python capture (custom `fobj`/`feval` callbacks expose grad/hess and per-round metrics; `predict(raw_score=True)` exposes accumulated scores; bagged indices require a custom-objective interception trick — see Validation Architecture).

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Rust-native API shape (API-01, OBJ-02)**
- **D-01:** Builder-pattern public API (`Booster::builder()…build()` / a training builder), NOT a verbatim `lgb.train(params,…)` free function. Idiomatic Rust ergonomics on the OUTSIDE; faithful C++ mirror below the API boundary.
- **D-02:** The training-params builder resolves to `lgbm-core::Config` internally — Config remains the single source of truth. No forked defaults/aliases/param semantics.
- **D-03:** Full param surface — a method per in-scope parameter PLUS a `from_config(Config)` / raw-set escape hatch so the oracle can drive any parity-relevant parameter.
- **D-04:** `custom` objective is a closure mirroring the Python `fobj` contract. Signature shaped as `Fn(preds: &[f32], dataset: &Dataset) -> (grad: Vec<f32>, hess: Vec<f32>)`. (Exact borrow/return ownership is Claude's discretion bounded by that contract.)
- **D-05:** Eval history + early-stopping outcome surfaced as `Booster` fields (`best_iteration` + per-valid-set/per-metric eval history), mirroring Python's `best_iteration_`/`best_score_`/`record_evaluation`.

**Oracle corpus matrix (carries P5 D-08 — real `lib_lightgbm` 4.6 oracle)**
- **D-06:** All 5 core objectives + one custom get committed end-to-end (train→model-text→predict) real-binary goldens. `custom` validated against a Python `fobj` reference.
- **D-07:** Full cross-product per objective: `{bagging on/off} × {early_stopping on/off} × {boost_from_average on/off}` → ~40 committed end-to-end cells (5 × 2 × 2 × 2) + custom. Researcher MAY collapse a cell to another ONLY if provably byte-identical, documented — never silent truncation. (See "Cross-Product Collapse Analysis" below.)
- **D-08:** Small per-objective synthetic corpora, committed + idempotently regenerable, objective-appropriate labels, reusing the Phase-2 binning path.
- **D-09:** Modest multi-iteration depth (~10–20 iters), small trees (low `num_leaves`).

**Validation granularity (max-diagnostic — analog of P5 D-06)**
- **D-10:** Per-row grad/hess golden snapshots at iteration 1 AND a later iteration (scores no longer zero) for each objective, from C++ `GetGradients`.
- **D-11:** Per-iteration accumulated raw-score snapshot.
- **D-12:** Per-eval-round metric-value snapshots (l1/l2/rmse/binary_logloss/binary_error/auc/multi_logloss).
- **D-13:** Bagging RNG parity is a dedicated golden — the exact bagged row-index set per round, bit-matching the C++ `Random` draw sequence + call order.

**Spine-first sequencing (analog of P5 D-04)**
- **D-14:** Minimal end-to-end spine = `regression`(L2) + `l2`/`rmse`.
- **D-15:** The minimal spine INCLUDES `boost_from_average` (the C++ regression default).
- **D-16:** Multiclass per-class trees enter AFTER the single-output spine is proven.
- **D-17:** Addition order: objectives → multiclass → bagging → early-stop, one axis at a time.

**Carried Forward (locked by prior phases — not re-litigated):** Faithful 1:1 C++ mirror below the API boundary; f32 end-to-end, ~1e-6 absolute, standard f32 accumulations; CPU `cubecl-cpu` f64-fold is the bit-exact hard merge gate, ROCm a separate ~1e-6 gate; real `lib_lightgbm` 4.6 oracle built deterministically (`deterministic=true force_row_wise=true num_threads=1` fixed seed), committed goldens + idempotent regen, `LightGBM/` NEVER `git add`ed; `lgbm-compute` is the single CubeCL seam (boosting stays above it); single-threaded deterministic core; the Phase-5 `SerialTreeLearner` is the per-tree engine driven via `train(grad, hess, is_first_tree) → Tree`, not modified.

### Claude's Discretion
- Crate placement/structure for the boosting layer (new `lgbm-boosting` + umbrella `lgbm` facade vs folding in) and the boosting↔learner wiring.
- Where objectives + metrics live (new crates vs modules) and their internal trait shape — bounded by C++ `ObjectiveFunction`/`Metric` factory semantics.
- Exact ownership/borrow shape of the `custom` objective closure (D-04), bounded by the Python `fobj` contract.
- The golden serialization/layering format for grad/hess, per-iteration scores, per-round metrics, and bagged-index fixtures — bounded by the oracle-harness comparator + Phase-3 `%.17g` machinery.
- AUC tie-handling / sort determinism, `metric_freq`/`first_metric_only` cadence specifics, per-class score memory layout — bounded by "must match the C++ reference."
- The captured-g/h capture path config + which iteration counts as "a later iteration" for D-10.
- When C++ behavior is the spec, the C++ source is authoritative over any inferred default.

### Deferred Ideas (OUT OF SCOPE)
- GOSS / DART / Random Forest (Phase 7, BST-04/05/06).
- Categorical / EFB splits (TRL-06) — Phase 7.
- Remaining objectives (huber/fair/poisson/quantile/mape/gamma/tweedie, cross-entropy, ranking) — Phase 7 (OBJ-04/05/06).
- Extended + ranking metrics (ndcg/map/average_precision/auc_mu/…) and per-query eval — Phase 7 (MET-03/04).
- SHAP/`predict_contrib`, prediction early stopping, monotone/interaction constraints, forced splits/bins, extra-trees, CEGB, refit/continue-training, feature importance — Phase 7.
- Python/PyO3 bindings — Phase 8.
- Parallel (rayon) CPU / multi-GPU boosting path — post-MVP.
- ROCm cross-check of the full train→predict loop — research/planning call; CPU bit-exact is the hard gate.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BST-01 | GBDT training loop (`TrainOneIter`, `UpdateScore`, per-class trees, shrinkage, `boost_from_average`) | "GBDT Control Flow" — exact `TrainOneIter` ordering (gbdt.cpp:344-452), per-class loop, `Shrinkage` (tree.h:188), `BoostFromAverage` (gbdt.cpp:319-342) |
| BST-02 | Score updater accumulation with deterministic reduction ordering | "Score Updater" — f64 `score_` (score_updater.hpp:27-128); training-path data-partition scatter (serial_tree_learner.h:100-115); valid/OOB predict-side path |
| BST-03 | Bagging / row subsampling with RNG-matching sequence | "Bagging RNG" — `bagging_rand_block_=1024`, per-block `Random(bagging_seed+i)`, `BaggingHelper` NextFloat loop (bagging.hpp:230-274), one-buffer `std::reverse` (threading.h:152-155) |
| BST-07 | Early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`) | "Early Stopping" — exact `factor_to_bigger_better*score.back()` vs `best_score_` + `min_delta` (gbdt.cpp:591-608), `kMinScore` init (gbdt.cpp:215) |
| OBJ-01 | Core objectives: regression(l2), regression_l1, binary, multiclass, multiclassova | "Objective Formulas" — all five read verbatim from the headers |
| OBJ-02 | `custom` objective pass-through | "Custom Objective" — `CreateObjectiveFunction` returns `nullptr` (objective_function.cpp); `TrainOneIter(grad,hess)` non-null branch (gbdt.cpp:355-372); maps to D-04 closure / Python `fobj` |
| OBJ-03 | `GetGradients`, `ConvertOutput`, `BoostFromScore`, `reg_sqrt` | "Objective Formulas" + "ConvertOutput already in lgbm-model" — training-side g/h is the net-new work |
| MET-01 | Core metrics: l1, l2, rmse, binary_logloss, binary_error, auc, multi_logloss | "Metric Formulas" — all read verbatim; AUC tie-invariance proven |
| MET-02 | Metric infra: multi-metric, `metric_freq`, `is_provide_training_metric`, training-metric eval | "Metric Infrastructure" — `OutputMetric` cadence (gbdt.cpp:551-609), `metric_freq` gate, training-metric list |
| API-01 | Rust-native API: Dataset, Booster, train, predict | "Crate Placement & Public API" — builder→Config (D-01/02/03), Booster fields (D-05) |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GBDT loop orchestration (`TrainOneIter`, per-class loop, iteration count) | Boosting (`lgbm-boosting`) | — | Mirrors C++ `GBDT` in `src/boosting/`; owns `models_`, `iter_`, sample strategy |
| Gradient/hessian computation | Objective (`lgbm-objective`) | — | Mirrors `ObjectiveFunction::GetGradients`; consumes Dataset labels/weights, produces g/h |
| Score accumulation (`score_`) | Boosting (ScoreUpdater) | TreeLearner (fast train-path add) | `ScoreUpdater` owns the f64 buffer; the train-path add delegates to the learner's data_partition scatter |
| Bagging row selection (RNG) | Boosting (SampleStrategy) | Core (`Random` LCG) | Mirrors `BaggingSampleStrategy`; RNG is the Phase-1 `lgbm-core::Random` |
| Metric evaluation | Metric (`lgbm-metric`) | Objective (`ConvertOutput`) | Mirrors `Metric::Eval`; calls objective's `ConvertOutput` for prob-space metrics |
| Early-stopping decision | Boosting | Metric (`factor_to_bigger_better`) | `GBDT::OutputMetric`/`EvalAndCheckEarlyStopping` consume metric values |
| Public train/predict + builder | Facade (`lgbm`) | Core (`Config`) | D-01 builder resolves to `lgbm-core::Config` (D-02) |
| Per-tree growth | TreeLearner (`lgbm-treelearner`, Phase 5) | Compute (`lgbm-compute`) | Unchanged — driven by the loop via `train(grad,hess,is_first_tree)` |

## Standard Stack

This phase introduces **no new external dependencies**. Every component is either an in-tree Rust crate (Phases 1–5) or a faithful hand-port of C++ logic. The only external artifact is the already-pinned `lightgbm==4.6.0` pip wheel used by the xtask capture pipeline (not a runtime/library dependency of the deliverable).

### Core (in-tree Rust foundations to build ON)
| Crate/Module | Source | Purpose | Why Standard |
|---------|--------|---------|--------------|
| `lgbm-treelearner` | Phase 5 | `SerialTreeLearner::train(grad,hess,is_first_tree)→Tree`, bit-exact | The per-tree engine the loop drives; DO NOT modify [CITED: crates/lgbm-treelearner/src/learner.rs:265] |
| `lgbm-model` | Phase 3 | `Tree`, `GbdtModel`(check name), `%.17g` formatter, `ObjectiveKind::convert_output` | Ensemble container + end-to-end model-text golden; predict-side ConvertOutput already done [CITED: crates/lgbm-model/src/objective.rs] |
| `lgbm-core` | Phase 1 | `Config` (~110 params + alias table + CHECK), `Random` LCG, `f32` types, `thiserror` errors | Config = single source of truth (D-02); `Random` = bagging RNG parity source [CITED: crates/lgbm-core] |
| `lgbm-dataset` | Phase 2 | Immutable binned store + metadata (label/weights/query) | Objective/metric/bagging input; do NOT re-bin [CITED: REQUIREMENTS DAT-06] |
| `lgbm-compute` | Phase 4 | `Backend` trait (single CubeCL seam) | Boosting stays ABOVE it (CMP-01); learner already sits on it |
| `oracle-harness` | Phase 1 | `compare_exact_f64_bits`/`compare_within(ORACLE_TOL)`, committed-golden/regen | Bit-exact CPU anchor + ~1e-6 comparator; extend `REFERENCE_MANIFEST.md` |
| `xtask` + `xtask/py/` | Phases 3/5 | `learner-oracle-capture` pip-wheel capture pattern | Extend with a boosting/objective/metric/bagging capture subcommand |

### Supporting (Rust std/ecosystem already in workspace)
| Crate | Version | Purpose | When to Use |
|-------|---------|---------|-------------|
| `thiserror` | 2.0.18 (workspace) | Domain error types at each new crate boundary | `BoostingError`, `ObjectiveError`, `MetricError` (FND-04 idiom) |
| `anyhow` | 1.0.102 (workspace) | Ergonomic propagation in xtask/tests | Capture pipeline + test harness only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `lgbm-boosting`+`lgbm-objective`+`lgbm-metric`+`lgbm` crates | Fold all into `lgbm-model` | Folding bloats the model crate (which is currently predict/serialize only) and muddies the Phase-8 binding seam; separate crates mirror the C++ `src/boosting`/`src/objective`/`src/metric` directory split and the `lgbm` facade is the natural PyO3 target. RECOMMEND separate crates. |
| Enum-dispatch objective/metric (`enum Objective { L2, L1, Binary, … }`) | `Box<dyn ObjectiveFunction>` trait objects | C++ uses a string-keyed factory returning a base-class pointer. Rust enum dispatch is the idiomatic, allocation-free mirror and matches the small fixed set (5 + custom). RECOMMEND enum + a `custom` variant holding the D-04 closure. Trait object is fine too but adds dynamic dispatch with no benefit at this set size. |
| Custom objective as closure (D-04, LOCKED) | Trait object | LOCKED to closure by D-04 (matches Python `fobj`). |

**Installation:** None — `cargo build` over the existing workspace. No `npm`/`pip`/`cargo add` of external packages.

## Package Legitimacy Audit

> Not applicable. This phase installs **no external packages**. All work is in-tree Rust hand-ports plus the already-pinned `lightgbm==4.6.0` pip wheel (a test-time oracle, captured via the Phase-5 `learner-oracle-capture` pipeline, version-asserted at capture time — `assert lgb.__version__ == version`). The wheel is not a dependency of the shipped crate. slopcheck/registry verification has no applicable target.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────────┐
  user builder (D-01) ──▶│  lgbm (facade): Booster / Dataset / train /  │
   .objective(...)       │  predict ; builder resolves to Config (D-02) │
   .num_iterations(...)  └───────────────┬─────────────────────────────┘
   .build()                              │ Config + Dataset(s) + custom closure?
                                         ▼
                       ┌─────────────────────────────────────────────┐
                       │           lgbm-boosting :: GBDT              │
                       │                                             │
   iter loop ────────▶ │  for iter in 0..num_iterations:             │
                       │   1. BoostFromAverage(class) ──┐ (iter 0)   │
                       │      adds init_score to TRAIN  │ to score_  │
                       │      + ALL VALID updaters      │            │
                       │   2. Boosting():               ▼            │
                       │      objective.GetGradients(train_score ────┼──▶ lgbm-objective
                       │                  , grad, hess) ◀────────────┼──  (g/h from labels)
                       │   3. Bagging(iter): RNG draw ──┐            │
                       │      bag_data_indices ◀────────┼────────────┼──▶ lgbm-core::Random
                       │   4. per cur_tree_id in 0..K:  ▼            │
                       │       (copy bagged g/h subset)             │
                       │       tree = learner.train(g,h,first) ─────┼──▶ lgbm-treelearner (P5)
                       │       RenewTreeOutput (l1 only)            │
                       │       tree.Shrinkage(learning_rate)       │
                       │       UpdateScore(tree, cur_tree_id) ─────┐│
                       │       tree.AddBias(init_score)           ││
                       │       models_.push(tree)                 ▼│
                       │   5. OutputMetric(iter) ────────┐  ScoreUpdater.score_ (f64)
                       │      (metric_freq gate)         │   train: learner.AddPredictionToScore
                       │   6. EvalAndCheckEarlyStopping ─┼──▶ lgbm-metric.Eval(score, obj)
                       └─────────────────────────────────┼──────────┘  ▲ uses ConvertOutput
                                                         ▼
                                          best_iteration / eval history (D-05 Booster fields)
                                                         │
   predict(X) ────────────────────────────────────────▶ GbdtModel.predict (P3) → ConvertOutput
```

A reader can trace the spine: builder→Config→GBDT loop→(objective g/h)→(bagging draw)→(per-class tree)→(shrinkage)→(score accumulate)→(metric/early-stop)→model→predict.

### Recommended Project Structure
```
crates/
├── lgbm-objective/         # NEW: ObjectiveFunction mirror
│   └── src/lib.rs          #   enum Objective + GetGradients/BoostFromScore/ConvertOutput(training side)
├── lgbm-metric/            # NEW: Metric mirror
│   └── src/lib.rs          #   enum Metric + Eval + factor_to_bigger_better + AUC
├── lgbm-boosting/          # NEW: GBDT loop
│   └── src/
│       ├── gbdt.rs         #   TrainOneIter, Boosting, BoostFromAverage, UpdateScore, early-stop
│       ├── score_updater.rs#   f64 score_ + AddScore variants
│       └── sample_strategy.rs #  BaggingSampleStrategy (RNG)
└── lgbm/                   # NEW: umbrella facade + public API
    └── src/
        ├── builder.rs      #   D-01 builder → lgbm-core::Config (D-02/D-03)
        ├── booster.rs      #   Booster (best_iteration, eval history — D-05)
        └── lib.rs          #   train() / predict() / Dataset re-export
```

### Pattern 1: GBDT Control Flow (BST-01) — the exact `TrainOneIter` order
**What:** The per-iteration sequence. Order is FP- and RNG-load-bearing.
**When to use:** The `lgbm-boosting::GBDT::train_one_iter` mirror.
**Source:** [CITED: LightGBM/src/boosting/gbdt.cpp:344-452]
```text
TrainOneIter(gradients=null, hessians=null):           // null ⇒ built-in objective
  init_scores[K] = 0
  // (1) boost-from-average FIRST, per class — only when models_ empty (iter 0)
  for cur_tree_id in 0..K:
    init_scores[cur_tree_id] = BoostFromAverage(cur_tree_id, update_scorer=true)
      // BoostFromAverage (gbdt.cpp:319): if models_ empty && !has_init_score && obj!=null
      //   && (boost_from_average || num_features==0):
      //     init = ObtainAutomaticInitialScore = obj->BoostFromScore(class_id)
      //     if |init| > kEpsilon:  AddScore(init,class) to TRAIN updater
      //                            AND to EVERY valid score updater
      //     return init   (else return 0)
  // (2) compute gradients/hessians on CURRENT train score
  Boosting():  obj->GetGradients(GetTrainingScore(), grad_ptr, hess_ptr)   // gbdt.cpp:220-235
  // (3) bagging (unless bagging_by_query): RNG draw, sets bag_data_indices_/bag_data_cnt_
  if !bagging_by_query: data_sample_strategy_->Bagging(iter_, learner, grad_.data(), hess_.data())
  is_use_subset, bag_data_cnt, bag_data_indices = strategy state
  // (4) per-class tree loop
  should_continue = false
  for cur_tree_id in 0..K:
    offset = cur_tree_id * num_data
    new_tree = Tree(2,false,false)                       // empty
    if class_need_train_[cur_tree_id] && num_features>0:
      grad = grad_ptr + offset ; hess = hess_ptr + offset
      if is_use_subset && bag_data_cnt < num_data && !gpu:   // compact bagged g/h to front
        for i in 0..bag_data_cnt: grad_ptr[offset+i]=grad[bag_idx[i]]; hess_ptr[offset+i]=hess[bag_idx[i]]
        grad = grad_ptr+offset; hess = hess_ptr+offset
      is_first_tree = models_.size() < K
      new_tree = learner->Train(grad, hess, is_first_tree)   // ← Phase-5 learner
    if new_tree.num_leaves > 1:
      should_continue = true
      // residual_getter closure: label[i] - train_score[offset+i]  (used only by l1 RenewTreeOutput)
      learner->RenewTreeOutput(new_tree, obj, residual_getter, num_data, bag_idx, bag_cnt, train_score)
      new_tree.Shrinkage(learning_rate)                       // tree.h:188 — multiply leaf/internal values
      UpdateScore(new_tree, cur_tree_id)                      // accumulate into score_
      if |init_scores[cur_tree_id]| > kEpsilon: new_tree.AddBias(init_scores[cur_tree_id])  // tree.h:213
    else:
      // degenerate: no split. constant tree. (only the boost_from_average==false branch adds init here)
      AsConstantTree(init_scores[cur_tree_id], num_data)
    models_.push(new_tree)
  if !should_continue: pop the K just-pushed trees; return true (finished)
  ++iter_; return false
```
**Critical ordering note:** `Shrinkage` is applied to leaf/internal values BEFORE `UpdateScore`, and `AddBias` is applied AFTER `UpdateScore`. The init score is added to `score_` during step (1) via `BoostFromAverage`, NOT via `AddBias` — `AddBias` only rewrites the **stored tree values** (for model text), it does NOT touch `score_`. This is subtle: the very first tree gets `AddBias(init)` so that when the model is later re-predicted from scratch (e.g. `GbdtModel.predict`), the init score is folded into tree 0's leaves and there is no separate "init score" term in the serialized model. Mirror exactly.

### Pattern 2: Score Updater — f64 accumulation (BST-02)
**What:** `score_` is `std::vector<double>` (NOT f32). All accumulation is f64. Layout is `[class0 rows | class1 rows | …]` (offset `= num_data * cur_tree_id`).
**Source:** [CITED: LightGBM/src/boosting/score_updater.hpp:27-128]
- **Init (iter 0):** zero, unless `Dataset` has `init_score` metadata (then copy it). [score_updater.hpp:30-46]
- **`AddScore(double val, cur_tree_id)`** (used by BoostFromAverage): `score_[offset+i] += val` for all i. [hpp:54-61]
- **Training-path `AddScore(tree_learner, tree, cur_tree_id)`** (the hot path in `UpdateScore`): delegates to `tree_learner->AddPredictionToScore(tree, score_+offset)` which scatters **per leaf** via the data partition — `for each leaf: out_score[row] += LeafOutput(leaf)` over the leaf's rows. [score_updater.hpp:88-92 → serial_tree_learner.h:100-115]. **This is bit-exact-relevant**: it is NOT the same float-op order as a per-row tree walk. The Rust learner already owns `data_partition` (Phase 5) — expose an `add_prediction_to_score(&tree, &mut [f64])` on it.
- **OOB rows (training, when `bag_data_cnt < num_data`):** the out-of-bag rows are scored via the tree's predict-side `AddScore(tree, indices+bag_cnt, num_data-bag_cnt, cur_tree_id)`. [gbdt.cpp:499-509]
- **Valid score updaters:** always the tree predict-side `AddScore(tree, cur_tree_id)` (full predict). [gbdt.cpp:517-519]
**`UpdateScore` full logic** (gbdt.cpp:491-520): if NOT use_subset → train fast-path add + OOB add; if use_subset → `AddScore(tree, cur_tree_id)` (predict over the subset dataset). Then valid updaters.

### Pattern 3: Bagging RNG (BST-03 / D-13) — the exact draw/call sequence
**What:** The single most RNG-order-sensitive path. Determines `bag_data_indices_`.
**Source:** [CITED: LightGBM/src/boosting/bagging.hpp:30-274, sample_strategy.h:24/75, threading.h:91-184]
- **Per-block RNG seeding** (`ResetSampleConfig`, bagging.hpp:177-181): `bagging_rand_block_ = 1024` (const, sample_strategy.h:75). One `Random` per 1024-row block: `for i in 0..ceil(num_data/1024): bagging_rands_.push(Random(bagging_seed + i))`. `bagging_seed` default = 3.
- **When bagging fires** (bagging.hpp:33): `(bag_data_cnt_ < num_data_ && iter % bagging_freq == 0) || need_re_bagging_`. So with `bagging_freq=k`, re-bag on iters 0,k,2k,… The `need_re_bagging_` flag is set true once by `ResetSampleConfig` so iter 0 always bags.
- **`bag_data_cnt_`** target (ResetSampleConfig:161): `static_cast<data_size_t>(bagging_fraction * num_data)` (truncated). Balanced (pos/neg) variant at :158-159.
- **The draw** (`BaggingHelper`, bagging.hpp:230-246), called once with `(start=0, cnt=num_data)` under `num_threads=1` (nblock=1, inner_size=cnt):
  ```text
  cur_left_cnt = 0 ; cur_right_pos = cnt
  for i in 0..cnt:
    cur_idx = start + i
    if bagging_rands_[cur_idx / 1024].NextFloat() < bagging_fraction:
      buffer[cur_left_cnt++] = cur_idx          // in bag, appended left in row order
    else:
      buffer[--cur_right_pos] = cur_idx         // out of bag, filled from right
  return cur_left_cnt
  ```
  **The NextFloat draw is consumed for EVERY row, in row order**, even out-of-bag rows. The block index `cur_idx/1024` selects which `Random` instance draws (each block's RNG advances independently). For corpora < 1024 rows, only `bagging_rands_[0]` is ever used — but the seeding loop still constructs `ceil(num_data/1024)=1` instance seeded `bagging_seed+0`.
- **One-buffer reverse** (`ParallelPartitionRunner<data_size_t, false>` ⇒ TWO_BUFFER=false, threading.h:152-155): after `BaggingHelper`, `std::reverse(left_ptr + cur_left_count, left_ptr + cur_cnt)` reverses the out-of-bag tail. So `bag_data_indices_ = [in-bag rows in ascending order] ++ [out-of-bag rows in DESCENDING order]`. The D-13 golden must assert this exact ordering, not just the in-bag SET.
- **Balanced (pos/neg) bagging** (`BalancedBaggingHelper`, :248-274): identical loop but draws against `pos_bagging_fraction` or `neg_bagging_fraction` depending on `label[start+i] > 0`. Activated when `(pos_bagging_fraction<1 || neg_bagging_fraction<1) && num_pos_data>0`. The draw is still per-row in order.
**D-13 golden capture:** the bagged index array is internal to LightGBM — see Validation Architecture for the capture trick (intercept via a custom objective that records which rows received nonzero gradients, OR — cleaner — replicate the RNG in the Rust test and assert against a C++-captured `Random` draw sequence golden, which Phase 1 already proved bit-exact via FND-01).

### Pattern 4: Per-class score layout (multiclass, D-16)
**What:** For K classes, `score_`, `gradients_`, `hessians_` are all length `num_data * K`, laid out **class-major**: class `k`'s rows occupy `[k*num_data, (k+1)*num_data)`. [CITED: gbdt.cpp:388 `offset = cur_tree_id * num_data`; multiclass_objective.hpp:93 `idx = num_data*k + i`]. The softmax in `GetGradients` gathers across classes for each row (`rec[k] = score[num_data*k + i]`), so the objective reads strided. Mirror this layout exactly — a row-major `[row][class]` layout would diverge.

### Anti-Patterns to Avoid
- **Accumulating scores in f32.** `score_` is f64; only g/h and leaf values are f32. Mixing precision in the accumulator diverges. [score_updater.hpp:123]
- **Per-row tree-walk for the training score update.** The C++ training path uses the data-partition per-leaf scatter (`AddPredictionToScore`), a different float-op order than predict. Use the learner's partition. [serial_tree_learner.h:100]
- **Adding the init score via `AddBias` into `score_`.** Init score enters `score_` via `BoostFromAverage→AddScore`; `AddBias` only rewrites stored tree leaf values. [gbdt.cpp:327 vs :417]
- **Skipping the OOB-row score update.** When bagging, OOB rows still get the tree's prediction added to their score (so the next iteration's gradients are correct). [gbdt.cpp:499-505]
- **Reordering the bagging NextFloat draws or dropping OOB draws.** Every row draws, in order; OOB rows are reversed. [bagging.hpp:237-243]
- **Idiomatic redesign of the loop below the API boundary.** D-01 idiom applies to the builder only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-tree growth | A new tree learner | `lgbm-treelearner::SerialTreeLearner` (P5) | Bit-exact vs real binary; re-implementing forfeits Phase-5's proof |
| Bagging RNG | A new PRNG | `lgbm-core::Random` LCG (P1, FND-01) | Bit-exact draw sequence already proven; bagging just needs the per-block seeding + NextFloat loop |
| Config defaults/aliases/validation | A builder-local param store | `lgbm-core::Config` (D-02) | Single source of truth; forking defaults reintroduces drift the alias table closed |
| predict-side ConvertOutput | New transforms | `lgbm-model::ObjectiveKind::convert_output` (P3) | sigmoid/softmax/ova/identity already ported + tested |
| Model text emission | New serializer | `lgbm-model` `%.17g` formatter (P3) | End-to-end model-text golden reuses it |
| Golden comparison | New comparator | `oracle-harness::compare_exact_f64_bits` / `compare_within` | Bit-exact + ~1e-6 already standardized |
| Real-binary capture | A from-source C++ build | the pinned `lightgbm==4.6.0` pip wheel via `xtask/py/` | `external_libs` are empty in-tree; the wheel IS the authoritative real binary (P5 D-08 precedent) |

**Key insight:** Phase 6 is overwhelmingly **wiring proven components together in the exact C++ order**, not building new numerical kernels. The genuinely net-new numerical code is small: the five objectives' `GetGradients`/`BoostFromScore` (training side), the seven metrics' `Eval`, and the bagging draw loop. Everything else is orchestration that must match an ordering, not a new formula.

## Objective Formulas (OBJ-01/03) — verbatim from the C++ headers

All `score` is the f64 accumulated raw score; `gradients`/`hessians` are written as `score_t = f32`. `kEpsilon = 1e-15f`. `Sign(x) = (x>0) - (x<0)` (returns int, [common.h:872]).

**regression (L2)** [CITED: regression_objective.hpp:127-190]
- `GetGradients` (no weights): `grad[i] = (score_t)(score[i] - label[i])` ; `hess[i] = 1.0f`. (With weights: `grad = (score_t)((score_t)(score-label) * w)`, `hess = (score_t)w` — note the inner f32 cast.)
- `ConvertOutput`: identity (or `Sign(x)*x*x` if `reg_sqrt`). [hpp:148]
- `BoostFromScore`: `suml = Σ label[i]` (f64); `sumw = num_data`; return `suml/sumw` (the label mean). With `deterministic=true` the reduction is ordered (the `if(!deterministic_)` strips the OpenMP reduction). [hpp:173-190]
- `reg_sqrt`: if set, `Init` transforms `label_[i] = Sign(label)*sqrt(|label|)` and `ConvertOutput` inverts it. [hpp:116-122]
- `IsConstantHessian` = true (no weights).

**regression_l1** [CITED: regression_objective.hpp:207-288]
- `GetGradients`: `diff = score[i]-label[i]` (f64); `grad[i] = (score_t)Sign(diff)` ; `hess[i] = 1.0f`.
- `BoostFromScore`: weighted/unweighted **percentile** at `alpha=0.5` (the median) via `PercentileFun`. [hpp:236-249] — NOT a simple mean. This is a sort-based reduction (see Pitfall: percentile determinism).
- `IsRenewTreeOutput` = **true**: after each tree, `RenewTreeOutput` recomputes each leaf's output as the **median residual** of its rows (`PercentileFun` over `residual_getter`). [hpp:251-283, serial_tree_learner.cpp:920-940]. The L2/binary/multiclass objectives return false (no-op renew). This is the one objective whose leaf values are NOT the learner's Newton output — they're overwritten post-growth.

**binary** [CITED: binary_objective.hpp:105-177]
- `sigmoid_` default 1.0 (`config.sigmoid`).
- Per row: `is_pos = (label[i] > 0)`; `label_val = {-1,+1}[is_pos]`; `label_weight = label_weights_[is_pos]` (1.0 each unless `is_unbalance`/`scale_pos_weight`).
  `response = -label_val * sigmoid_ / (1 + exp(label_val * sigmoid_ * score[i]))` (f64);
  `abs_response = |response|`;
  `grad[i] = (score_t)(response * label_weight)` ;
  `hess[i] = (score_t)(abs_response * (sigmoid_ - abs_response) * label_weight)`.
- `BoostFromScore`: `pavg = (Σ is_pos)/num_data` clamped to `[kEpsilon, 1-kEpsilon]`; return `log(pavg/(1-pavg)) / sigmoid_`. [hpp:139-165]
- `ConvertOutput`: `1/(1+exp(-sigmoid_*input))`. [hpp:175]
- `ClassNeedTrain` = false if only one class present (then no tree trained — degenerate constant). [hpp:80-84,167]

**multiclass (softmax)** [CITED: multiclass_objective.hpp:24-181, common.h:571-600]
- `num_class_ = config.num_class`; `factor_ = num_class / (num_class - 1.0)` (Friedman redundant-form rescale).
- `GetGradients`: for each row, gather `rec[k] = score[num_data*k + i]` across classes, `Common::Softmax(&rec)` (max-subtraction: `wmax = max rec`, `rec[k] = exp(rec[k]-wmax)`, normalize by `Σ`), then per class:
  `p = rec[k]`; `grad[idx] = (score_t)(p - 1.0f)` if `label==k` else `(score_t)p`; `hess[idx] = (score_t)(factor_ * p * (1-p))`.
- `BoostFromScore(class_id)`: `log(max(kEpsilon, class_init_probs_[class_id]))` where `class_init_probs_[k] = (count of label==k)/num_data`. [hpp:155-157, Init:53-84]
- `ConvertOutput`: `Softmax(input, output, num_class)`. [hpp:132]
- `NumModelPerIteration` = `num_class` (⇒ `num_tree_per_iteration_ = num_class`). [hpp:149]

**multiclassova** [CITED: multiclass_objective.hpp:186-276]
- Holds `num_class_` independent `BinaryLogloss` objectives, each with `is_pos = (label == i)`.
- `GetGradients`: for each class i, call `binary_loss_[i]->GetGradients(score+offset, grad+offset, hess+offset)` with `offset = num_data*i`. [hpp:228-233]
- `BoostFromScore(class_id)` = `binary_loss_[class_id]->BoostFromScore(0)`.
- `ConvertOutput`: per-class sigmoid `1/(1+exp(-sigmoid_*input[i]))`.

### Custom objective (OBJ-02) — the D-04 closure
**Source:** [CITED: objective_function.cpp `CreateObjectiveFunction` returns nullptr for "custom"; gbdt.cpp:355-372]
- C++: `objective=custom` ⇒ `objective_function_ == nullptr`. The user supplies grad/hess each iteration; `TrainOneIter(gradients, hessians)` is called with non-null pointers. The non-null branch (gbdt.cpp:356) does `CHECK(objective_function_ == nullptr)` then (for GOSS only) copies into `gradients_`; for plain GBDT the supplied pointers are used directly. BoostFromAverage is skipped (`obj==null`).
- Python `fobj(preds, train_data) -> (grad, hess)` is called BY the Python `train()` wrapper each round, which then calls `Booster.__boost(grad, hess)` → `LGBM_BoosterUpdateOneIterCustom`. `preds` is the **raw** accumulated score (Python passes the current raw margin), shape `(num_data * num_class)` class-major (Python reshapes for multiclass).
- **D-04 Rust mapping:** the closure `Fn(preds: &[f32], dataset: &Dataset) -> (Vec<f32>, Vec<f32>)` is called by the Rust loop at step (2) IN PLACE of `objective.GetGradients`, passing the current `score_` cast to f32 (mirror Python's f32 preds). The closure's output replaces grad/hess. `boost_from_average` is forced off / ignored when custom (mirror `obj==null` skipping BoostFromAverage). For the D-06 custom golden, validate against a Python `fobj` reference run (no distinct C++ objective exists to diff).

### ConvertOutput already exists (predict side)
`lgbm-model::ObjectiveKind` already implements `ConvertOutput` (sigmoid/softmax/ova/identity) and is tested (Phase 3). Phase 6 adds the **training** side (`GetGradients`/`BoostFromScore`). The metric `Eval` calls `ConvertOutput` for prob-space metrics — reuse the existing `lgbm-model` impl (or move it to `lgbm-objective` and re-export; recommend keeping it in `lgbm-objective` as the canonical owner and having `lgbm-model` depend on it, but that is Claude's discretion bounded by not breaking Phase-3 callers).

## Metric Formulas (MET-01) — verbatim

All metrics divide by `sum_weights_` (= `num_data` unweighted). `factor_to_bigger_better`: -1 for losses, +1 for AUC. `Eval` calls `objective->ConvertOutput` when an objective is present (gbdt passes `objective_function_`, which is null for custom). [CITED: regression_metric.hpp, binary_metric.hpp, multiclass_metric.hpp]

| Metric | `LossOnPoint` (per row) | Aggregate | Source |
|--------|------------------------|-----------|--------|
| `l2` | `(score-label)^2` | `Σ/sumw` | regression_metric.hpp:142 |
| `rmse` | `(score-label)^2` | `sqrt(Σ/sumw)` | regression_metric.hpp:123-130 |
| `l1` | `|score-label|` | `Σ/sumw` | regression_metric.hpp:177 |
| `binary_logloss` | `label<=0: -log(1-p)` (if `1-p>kEps`); `label>0: -log(p)` (if `p>kEps`); else `-log(kEps)` | `Σ/sumw` | binary_metric.hpp:119-130 |
| `binary_error` | `p<=0.5: (label>0)` ; else `(label<=0)` | `Σ/sumw` | binary_metric.hpp:143-149 |
| `multi_logloss` | `rec[label]>kEps: -log(rec[label])` else `-log(kEps)` (rec = softmax/ova ConvertOutput) | `Σ/sumw` | multiclass_metric.hpp:167-174 |
| `auc` | (sorted accumulation — below) | see below | binary_metric.hpp:194-251 |

**For prob-space metrics** (`binary_logloss`, `binary_error`, `multi_logloss`) the metric calls `objective->ConvertOutput(&score[i], &prob)` first, so `p`/`rec` is the transformed probability, not the raw score. [binary_metric.hpp:80-83, multiclass_metric.hpp:74]

**AUC** [CITED: binary_metric.hpp:194-251]:
```text
sorted_idx = argsort_descending by score   (Common::ParallelSort, comparator score[a] > score[b])
cur_pos=cur_neg=sum_pos=accum=0 ; threshold = score[sorted_idx[0]]
for i in 0..n:
  if score[sorted_idx[i]] != threshold:
    threshold = score[sorted_idx[i]]
    accum += cur_neg * (cur_pos*0.5 + sum_pos)
    sum_pos += cur_pos ; cur_neg = cur_pos = 0
  cur_neg += (label[sorted_idx[i]] <= 0)
  cur_pos += (label[sorted_idx[i]] >  0)
accum += cur_neg * (cur_pos*0.5 + sum_pos) ; sum_pos += cur_pos
auc = (sum_pos>0 && sum_pos!=sum_weights) ? accum/(sum_pos*(sum_weights-sum_pos)) : 1.0
```

## State of the Art / Determinism Notes

| Concern | C++ behavior | Rust port implication |
|---------|--------------|------------------------|
| Score accumulator width | f64 (`std::vector<double>`) | Accumulate in f64; cast to f32 only when feeding g/h |
| OpenMP reductions in `BoostFromScore` | gated `if(!deterministic_)` — ordered when `deterministic=true` | Oracle runs `deterministic=true`; use a single ordered sequential sum |
| AUC sort | `ParallelSort` → `std::sort` (UNSTABLE) when `num_threads<=1` | See Pitfall 1: AUC is tie-ORDER-INVARIANT, so unstable sort is safe |
| `MaybeRoundToZero` in Shrinkage/AddBias | `IsZero(fval) ? 0 : fval` (snaps tiny values to +0) | Mirror in the Rust Tree shrinkage/add_bias (already partly present from P5 — confirm) |
| `kEpsilon` guards | BoostFromScore clamps, metric logloss floor | Copy `kEpsilon=1e-15f` from `lgbm-core` (already defined) |

## Cross-Product Collapse Analysis (D-07 — provable byte-identical collapses only)

D-07 allows collapsing a cell ONLY if provably byte-identical. The legitimate collapses, with proof:

1. **`bagging=off + early_stop=off + bfa=on` IS the per-objective spine golden** (D-07 explicitly names this). The spine end-to-end golden (D-14/D-15) already covers this cell for regression. Do NOT re-capture it as a separate "matrix" cell — reference it. (Explicitly allowed by D-07.)
2. **`early_stop=off`** cells differ from `early_stop=on` cells ONLY if early stopping actually FIRES. With `early_stopping_round` set but enough non-improving rounds never reached in ~10-20 iters, the two are byte-identical. **Do NOT collapse blindly** — a properly-constructed early-stop cell must use a valid set + a corpus where the metric plateaus so stopping fires (D-07 planning note). If a cell is constructed so early stopping never fires, it is byte-identical to its `early_stop=off` sibling and SHOULD be collapsed with that note, otherwise it is testing nothing. RECOMMEND: construct early-stop cells to genuinely fire (don't collapse).
3. **`bfa=off`** is NOT collapsible with `bfa=on` for any objective whose `BoostFromScore` is non-zero (all five core objectives have non-trivial init scores), so all `bfa` cells are distinct. EXCEPTION: if a corpus's label mean (regression) or pavg (binary) yields `|init_score| <= kEpsilon`, then `BoostFromAverage` returns 0 and the cell collapses — avoid such corpora so the bfa axis is exercised.

**Net:** the only safe automatic collapse is (1) (spine == bagging-off/es-off/bfa-on). Roughly 39 distinct cells remain + the custom run. Document every collapse in `REFERENCE_MANIFEST.md`; never silently drop a cell.

## Runtime State Inventory

> N/A for greenfield numerical code. This phase adds new crates and consumes existing in-tree state; it does not rename or migrate stored data, live-service config, OS-registered state, secrets, or build artifacts. The only persisted artifacts are the committed golden fixtures under `crates/oracle-harness/tests/fixtures/` (new files, idempotently regenerated — additive, no migration). **Nothing found in any category — verified: no rename/refactor of existing identifiers, and the new crates are net-additions to the workspace `members` list.**

## Common Pitfalls

### Pitfall 1: AUC tie-ordering vs unstable sort
**What goes wrong:** Assuming AUC requires a stable sort to match C++ bit-for-bit.
**Why it happens:** C++ `ParallelSort` falls back to `std::sort` (UNSTABLE) when `num_threads<=1` (the oracle config). [common.h:687-689]
**How to avoid:** The AUC algorithm groups all rows of equal score together (`if cur_score != threshold`) before accumulating, so the **relative order of tied scores never affects the result**. Any sort (stable or unstable) that produces descending-by-score order yields identical AUC. The Rust port can use `slice::sort_by` (unstable) or `sort_by` with the `score[a] > score[b]` comparator and match bit-for-bit. Verify with a corpus containing tied scores.
**Warning signs:** AUC golden mismatch ONLY on tied-score corpora ⇒ the grouping logic was dropped, not the sort.

### Pitfall 2: regression_l1 leaf values are NOT the learner's output
**What goes wrong:** Asserting the l1 model-text leaf values against the Newton leaf outputs from the learner.
**Why it happens:** `regression_l1` has `IsRenewTreeOutput()==true`; `RenewTreeOutput` overwrites every leaf with the **median residual** of its rows (`PercentileFun`, alpha=0.5) AFTER growth, BEFORE shrinkage. [regression_objective.hpp:251-283]
**How to avoid:** Wire `RenewTreeOutput` into the loop (step between Train and Shrinkage). For L2/binary/multiclass it's a no-op (return false / unchanged). Port `PercentileFun` (a sort + index pick) faithfully — and its determinism (Pitfall 3).
**Warning signs:** l1 spine golden diverges on leaf values but matches tree topology.

### Pitfall 3: Percentile/median determinism (regression_l1 BoostFromScore + RenewTreeOutput)
**What goes wrong:** `PercentileFun`/`WeightedPercentileFun` sort the data; tie order + interpolation must match C++.
**Why it happens:** Median of an even-count set, and weighted percentile interpolation, are sensitive to the exact algorithm. [regression_objective.hpp macros + common.h percentile helpers]
**How to avoid:** Read and port `PercentileFun`/`WeightedPercentileFun` from `common.h` verbatim (the exact index formula and `std::sort`/`nth_element` choice). This is the highest-risk piece of new numerical code in the phase. Capture a per-leaf renew golden (D-10/D-11 style) for the l1 corpus.
**Warning signs:** l1 init score or l1 leaf values off by one rank position.

### Pitfall 4: Bagging draws every row (including OOB) and reverses the tail
**What goes wrong:** Drawing NextFloat only for in-bag rows, or keeping OOB rows in ascending order.
**Why it happens:** The loop draws for all `cnt` rows; one-buffer mode reverses `[cur_left_count, cnt)`. [bagging.hpp:237-243, threading.h:152-155]
**How to avoid:** Mirror the exact loop: draw per row in order, append in-bag left, fill OOB from the right, then reverse the OOB tail. Assert the FULL `bag_data_indices` array (in-bag asc ++ OOB desc), not just the in-bag set. (D-13.)
**Warning signs:** in-bag set matches but the index array order differs ⇒ OOB handling wrong ⇒ OOB score updates land on wrong rows.

### Pitfall 5: boost_from_average init score path (two distinct entry points)
**What goes wrong:** Adding init score in the wrong place, or double-adding.
**Why it happens:** Init score enters `score_` ONCE via `BoostFromAverage→AddScore` (iter 0, before trees). Separately, `AddBias` folds it into tree-0's stored leaf values (for model serialization) but does NOT touch `score_`. A second `bfa=false` path adds init to a degenerate constant tree (gbdt.cpp:422-430). [gbdt.cpp:319-342, 416-430]
**How to avoid:** Mirror exactly: `BoostFromAverage` updates `score_` + valid updaters and returns init; the per-tree loop applies `AddBias(init)` to the tree (model text only) when `|init|>kEpsilon`. Do not add init to `score_` twice.
**Warning signs:** per-iteration score golden (D-11) off by exactly the init score at iter 0; or model-text leaf values off by init but predictions correct (means AddBias missing).

### Pitfall 6: `class_need_train_` / degenerate trees
**What goes wrong:** Forcing a tree when the objective says a class needs no training (e.g. single-class binary, or a multiclass class with prob ~0/~1).
**Why it happens:** `class_need_train_[k] = obj->ClassNeedTrain(k)`; if false, a constant (1-leaf) tree is pushed. [gbdt.cpp:155, 390, 419-434]
**How to avoid:** Honor `ClassNeedTrain`; push a constant tree for untrained classes. Keep the model's tree count == `iter * K` regardless.

## Code Examples (verified patterns from the C++ source)

### Spine GetGradients (regression L2, the D-14 starting point)
```rust
// Source: LightGBM/src/objective/regression_objective.hpp:127-142 (no weights)
// score: &[f64] (accumulated raw score), label: &[f32]
for i in 0..num_data {
    gradients[i] = (score[i] - label[i] as f64) as f32; // score_t cast
    hessians[i] = 1.0f32;
}
```

### Score updater training-path add (the bit-exact hot path)
```rust
// Source: serial_tree_learner.h:100-115 — per-leaf scatter via data_partition
// score is f64, offset = num_data * cur_tree_id
if tree.num_leaves() <= 1 { return; }
for leaf in 0..tree.num_leaves() {
    let out = tree.leaf_output(leaf);                 // f64
    for &row in data_partition.index_on_leaf(leaf) {
        score[offset + row as usize] += out;
    }
}
```

### Early-stopping decision (BST-07)
```rust
// Source: gbdt.cpp:591-608 — per (valid_set i, metric j)
// best_score_[i][j] initialized to kMinScore (gbdt.cpp:215); skip j>0 if first_metric_only
let cur = metric.factor_to_bigger_better() * eval_scores.last().unwrap();
if cur - best_score[i][j] > early_stopping_min_delta {
    best_score[i][j] = cur;
    best_iter[i][j] = iter;       // improvement
} else if iter - best_iter[i][j] >= early_stopping_round {
    // STOP: best iteration = best_iter[i][j]; pop last early_stopping_round*K trees (gbdt.cpp:484)
}
```

### Bagging draw (BST-03 / D-13)
```rust
// Source: bagging.hpp:230-246 + threading.h:152-155 (one-buffer reverse)
let mut left = 0usize; let mut right = cnt;            // cnt = num_data
let mut buf = vec![0i32; cnt];
for i in 0..cnt {
    let idx = i as i32;
    if bagging_rands[i / 1024].next_float() < bagging_fraction {  // f64 < f64
        buf[left] = idx; left += 1;
    } else {
        right -= 1; buf[right] = idx;
    }
}
buf[left..cnt].reverse();   // one-buffer reverse of the OOB tail
// buf = bag_data_indices ; left = bag_data_cnt
```

## Validation Architecture

> nyquist_validation is enabled (no `workflow.nyquist_validation: false` found — see Environment note). This section derives the layered D-10..D-13 + end-to-end goldens into a test map.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cargo test` (workspace convention, all prior phases) |
| Config file | none — `cargo test --workspace` |
| Quick run command | `cargo test -p lgbm-boosting` (per-crate, < 30s) |
| Full suite command | `cargo test --workspace` |
| Oracle capture (human-gated, NOT in routine test) | `cargo run -p xtask -- boosting-oracle-capture` (extends `learner-oracle-capture`) |

### Layered golden battery (D-10..D-13 + end-to-end), mirroring the Phase-5 layered discipline

| Layer | Golden | Asserts | Capture source |
|-------|--------|---------|----------------|
| L1 grad/hess (D-10) | `<obj>_gh_iter1.txt`, `<obj>_gh_iterN.txt` | per-row g/h at iter 1 (score=init) AND a later iter (score≠0), bit-exact f32 | Python `fobj` interception: run real `lgb.train` with a wrapping objective that records `(grad,hess)` each round before forwarding to the built-in formula; OR re-derive g/h from the built-in objective applied to captured per-iter scores (L2 below). RECOMMEND the score-derivation route (cleaner, no fobj override of a built-in). |
| L2 per-iter score (D-11) | `<obj>_scores.txt` | accumulated raw `score_` after each iteration, bit-exact f64 | `booster.predict(X, raw_score=True, num_iteration=k)` for k in 1..N (Python exposes cumulative raw margin) — class-major for multiclass |
| L3 per-round metric (D-12) | `<obj>_metrics.txt` | each metric value at each eval round | `record_evaluation` callback / `evals_result` dict from `lgb.train` with `valid_sets` + `feval`/built-in metric |
| L4 bagged indices (D-13) | `bag_indices_seed<S>_frac<F>.txt` | full `bag_data_indices` array per bagging round (in-bag asc ++ OOB desc) | Two options — see below |
| L5 end-to-end (D-06/D-07) | `<obj>_<cell>_model.txt`, `<obj>_<cell>_pred.txt` | model text (`%.17g`) + predictions, ~40 cells | `booster.save_model()` + `booster.predict()` (Phase-3/5 mechanism) |

**D-13 bagged-index capture — the hard one (internal state, not a Python API):**
- **Option A (RECOMMEND): RNG-replay golden, not a bag capture.** Phase-1 FND-01 already proved the `Random` LCG bit-exact against a captured C++ draw sequence. The bagging bag is a *pure function* of (`bagging_seed`, `bagging_fraction`, `num_data`, row order, block size 1024). The Rust test reproduces `BaggingHelper` over `lgbm-core::Random` and asserts the result against a **C++-captured `Random.NextFloat` sequence golden** (capture `ceil(num_data/1024)` Randoms seeded `bagging_seed+i`, dump their first `num_data` NextFloats). Then the bag is deterministically derived and self-checked. This avoids needing LightGBM to expose its internal bag.
- **Option B (corroboration): infer the bag from a custom-objective row-mask.** Run `lgb.train` with `bagging_freq=1` and a custom `fobj` that records WHICH rows had their gradient *consumed* — but LightGBM applies bagging by passing a subset to the tree learner, not by zeroing gradients, so the custom objective sees ALL rows; the bag is not directly observable this way. Option B is unreliable; prefer Option A. If a direct bag capture is required, the only faithful route is building `lib_lightgbm` from source with a debug dump (the P5 05-09 "build the real binary" precedent) — flag as a checkpoint:human-verify if the planner deems Option A insufficient.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BST-01 | GBDT loop grows the same trees + iteration count | integration (end-to-end model text, L5) | `cargo test -p oracle-harness --test boosting_parity` | ❌ Wave 0 |
| BST-02 | per-iter score accumulation bit-exact | integration (L2) | same, `..::score_accumulation` | ❌ Wave 0 |
| BST-03 | bagged rows match RNG sequence | unit (L4 RNG-replay) | `cargo test -p lgbm-boosting bagging_rng` | ❌ Wave 0 |
| BST-07 | early stopping fires at same iter | integration (L3 + L5 best_iteration) | `..::early_stopping` | ❌ Wave 0 |
| OBJ-01/03 | per-row g/h bit-exact (5 objectives) | unit/integration (L1) | `cargo test -p lgbm-objective gradients` | ❌ Wave 0 |
| OBJ-02 | custom closure == Python fobj reference | integration | `..::custom_objective` | ❌ Wave 0 |
| MET-01 | 7 metrics match per round | integration (L3) | `cargo test -p lgbm-metric eval` | ❌ Wave 0 |
| MET-02 | metric_freq / training-metric cadence | unit | `..::metric_infra` | ❌ Wave 0 |
| API-01 | builder→Config→train→predict end-to-end | integration (L5) | `cargo test -p lgbm public_api` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate-under-edit>` (the touched crate's tests).
- **Per wave merge:** `cargo test --workspace`.
- **Phase gate:** full workspace green + all layered goldens (L1-L5) replay bit-exact on the cubecl-cpu anchor before `/gsd-verify-work`. Capture is idempotent (`git diff` empty after regen).

### Wave 0 Gaps
- [ ] `crates/lgbm-objective/` — crate scaffold + `ObjectiveError` (thiserror) + the 5 objectives + custom closure type
- [ ] `crates/lgbm-metric/` — crate scaffold + `MetricError` + the 7 metrics + `factor_to_bigger_better`
- [ ] `crates/lgbm-boosting/` — `GBDT`, `ScoreUpdater`, `BaggingSampleStrategy`, early-stop
- [ ] `crates/lgbm/` — builder + `Booster` (D-05 fields) + `train`/`predict`
- [ ] `lgbm-treelearner`: expose `add_prediction_to_score(&tree, &mut [f64])` (data-partition scatter) + `renew_tree_output` hook (BST-02, l1)
- [ ] `lgbm-model::Tree`: confirm/add `shrinkage(rate)`, `add_bias(val)` with `MaybeRoundToZero` (some present from P5 — audit)
- [ ] `xtask boosting-oracle-capture` + extended `xtask/py/` capture (L1-L5) + `Random.NextFloat` sequence dumper for L4
- [ ] `oracle-harness/tests/fixtures/boosting/` golden corpus + `REFERENCE_MANIFEST.md` entries
- [ ] `tests/boosting_parity.rs` (or `-p oracle-harness --test boosting_parity`) layered replay

## Security Domain

> `security_enforcement` status: no `.planning/config.json` security flag was located in this session (see Environment note). This phase is a numerical library with no auth/session/network/untrusted-input surface — the only external input is the user's training data/config through the builder. Applying ASVS:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | library, no auth |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes | Builder + crate boundaries return typed `Result` (thiserror, FND-04). Validate: `num_iterations>0`, `learning_rate>0`, `0<bagging_fraction<=1`, `num_class>=1`, label range for multiclass (`[0,num_class)` — multiclass_objective.hpp:62 `Log::Fatal` → Rust `Result`), `sigmoid>0`. Mirror C++ `CHECK`/`Log::Fatal` as `Result` errors, never a panic (Phase-1 idiom). |
| V6 Cryptography | no | the `Random` LCG is a determinism tool, NOT cryptographic; do not treat as secure RNG |

| Threat Pattern | STRIDE | Standard Mitigation |
|----------------|--------|---------------------|
| Out-of-range multiclass label → OOB index | Tampering/DoS | Validate `label ∈ [0,num_class)` at objective Init, return `ObjectiveError` (C++ does `Log::Fatal`) |
| `num_data` mismatch grad/hess vs dataset | Tampering | Length checks at crate boundaries (the learner already does this — extend to the loop) |
| Builder param that violates a `CHECK` (e.g. `bagging_fraction=0`) | DoS | Route through `Config` validation (CFG-03) — already centralized (D-02) |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (edition 2024) | all crates | ✓ | rust-version 1.95 (workspace) | — |
| `cubecl` 0.10.0 (cpu default) | learner/backend (below boosting) | ✓ | 0.10.0 (workspace, alpha-pinned) | — |
| `lightgbm==4.6.0` pip wheel | xtask capture only (test-time oracle) | assumed (P5 used it) | 4.6.0 (version-asserted at capture) | If absent: capture step blocks (human-gated); routine `cargo test` replays committed goldens with NO wheel needed |
| ROCm gfx1100 GPU | ROCm cross-check (deferred this phase) | ✓ (per MEMORY) | ROCm 7.1.1 | CPU bit-exact is the hard gate; ROCm loop re-run is a deferred research call |
| C++ toolchain / `lib_lightgbm` from source | NOT needed (wheel is the oracle) | n/a | — | Only needed if D-13 Option A is rejected for a source-built debug bag dump |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** the pip wheel is only needed at capture (regen) time; the committed goldens make routine test runs self-contained (Phase-1..5 discipline).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `lightgbm==4.6.0` pip wheel is installed/available for capture (carried from P5) | Environment / Validation | Capture (regen) blocks; routine tests unaffected (committed goldens). Planner should gate capture as human-verify (P5 precedent). |
| A2 | nyquist_validation is enabled (no explicit `false` found this session) | Validation Architecture | If actually disabled, the Validation section is informational, not a gate — no harm; the layered goldens are wanted anyway per D-10..D-13. |
| A3 | `security_enforcement` defaults to enabled (no config flag located) | Security Domain | If disabled, the section is informational; the V5 input-validation it prescribes is already required by FND-04/CFG-03 regardless. |
| A4 | D-11 per-iter raw score is observable via Python `predict(raw_score=True, num_iteration=k)` matching internal `score_` bit-for-bit (f64) | Validation L2 | If the Python predict path re-accumulates in a different float order than the internal train `score_`, the L2 golden could differ from the *internal* accumulator by ULPs. Mitigation: the train-path `score_` IS what predict re-derives (both are tree-output sums); validate the END score equals `predict(raw_score)` and use the model-text + final prediction (L5) as the authoritative end-to-end gate. Flag for the planner: verify L2 capture route equals internal accumulator on the spine before relying on it for all cells. |
| A5 | `GbdtModel` is the model container name in `lgbm-model` (CONTEXT says `GbdtModel`; could be a different ident) | Standard Stack | Cosmetic; planner confirms the actual type name when wiring. |
| A6 | The Rust `Tree` already has `shrinkage`/`add_bias` (CONTEXT P5 mentions MaybeRoundToZero/Shrinkage finalize) | Wave 0 | If absent, add them (small); if present, reuse. Audit in Wave 0. |

## Open Questions

1. **Where should `ConvertOutput` canonically live?**
   - What we know: `lgbm-model::ObjectiveKind::convert_output` exists (predict side, P3); metrics + objectives both need it.
   - What's unclear: whether to move it to `lgbm-objective` (canonical objective owner) and have `lgbm-model` depend on it, or keep it in `lgbm-model` and have `lgbm-objective`/`lgbm-metric` depend on `lgbm-model`.
   - Recommendation: Keep predict-side in `lgbm-model` to avoid churning Phase-3 callers; `lgbm-metric` depends on `lgbm-model` for `ConvertOutput`. `lgbm-objective` owns the TRAINING side (`GetGradients`/`BoostFromScore`) and may re-export or duplicate the small transform. Claude's discretion (CONTEXT) — bounded by not breaking P3 tests.

2. **D-11 capture route fidelity (see A4).**
   - What we know: Python `predict(raw_score=True)` gives cumulative raw margin.
   - What's unclear: whether it equals the internal training `score_` (f64) bit-for-bit, or only within ULPs (different accumulation order).
   - Recommendation: On the spine cell, assert `predict(raw_score, k)` against the Rust internal `score_` after k iters; if bit-exact, use it for all cells; if only ~1e-6, treat L2 as a ~1e-6 (not bit-exact) layer and lean on L5 (model text, bit-exact) as the hard gate. Resolve in the spine wave before scaling to 40 cells.

3. **regression_l1 `PercentileFun` exact algorithm.**
   - What we know: it's a sort+index median (alpha=0.5), used in both `BoostFromScore` and `RenewTreeOutput`.
   - What's unclear: the exact index/interpolation convention in `common.h` (not yet read line-by-line this session).
   - Recommendation: Read `common.h` `PercentileFun`/`WeightedPercentileFun` macros verbatim before implementing the l1 cells; capture a per-leaf renew golden. Highest new-numerical-code risk.

## Sources

### Primary (HIGH confidence — read directly from in-tree C++ this session)
- `LightGBM/src/boosting/gbdt.cpp` (lines 155-218, 219-235, 237-342, 344-520, 551-638) — TrainOneIter, Boosting, BoostFromAverage, UpdateScore, OutputMetric, early-stop, AddValidDataset
- `LightGBM/src/boosting/score_updater.hpp` (full) — f64 score_, AddScore variants
- `LightGBM/src/boosting/bagging.hpp` (full) — bagging draw, ResetSampleConfig, BalancedBaggingHelper
- `LightGBM/src/boosting/sample_strategy.h` (full) — bagging_rand_block_=1024, runner type
- `LightGBM/include/LightGBM/utils/threading.h` (20-200) — BlockInfo, ParallelPartitionRunner::Run, one-buffer reverse
- `LightGBM/src/objective/regression_objective.hpp` (93-392) — L2/L1 GetGradients/BoostFromScore/ConvertOutput/RenewTreeOutput
- `LightGBM/src/objective/binary_objective.hpp` (full) — sigmoid grad/hess/BoostFromScore/ConvertOutput
- `LightGBM/src/objective/multiclass_objective.hpp` (full) — softmax + ova, per-class layout
- `LightGBM/src/objective/objective_function.cpp` (full) — CreateObjectiveFunction factory, custom=nullptr
- `LightGBM/include/LightGBM/utils/common.h` (556-600 Softmax, 653-675 AvoidInf, 682-727 ParallelSort, 872-875 Sign)
- `LightGBM/src/metric/regression_metric.hpp`, `binary_metric.hpp` (incl. AUC), `multiclass_metric.hpp`, `metric.cpp` (factory)
- `LightGBM/include/LightGBM/tree.h` (188 Shrinkage, 213 AddBias, 232 AsConstantTree, 258 MaybeRoundToZero)
- `LightGBM/src/treelearner/serial_tree_learner.h` (100-118 AddPredictionToScore + RenewTreeOutput)
- `LightGBM/include/LightGBM/config.h` (defaults: num_iterations=100, learning_rate=0.1, bagging_fraction=1.0, bagging_freq=0, bagging_seed=3, early_stopping_round=0, first_metric_only=false, early_stopping_min_delta=0.0, num_class=1, sigmoid=1.0, boost_from_average=true, reg_sqrt=false, metric_freq=1, is_provide_training_metric=false)
- `xtask/py/learner_oracle_capture.py` — the pip-wheel capture pattern to extend

### Secondary (HIGH — in-repo planning/code, read this session)
- `.planning/phases/06-gbdt-spine-core-objectives-metrics/06-CONTEXT.md` (D-01..D-17, canonical refs)
- `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` (Phase 6 goal + 5 SC), `.planning/STATE.md` (P5 complete)
- `.planning/codebase/CONCERNS.md` (FP reduction ordering, subtraction trick, kEpsilon hazards)
- `crates/lgbm-treelearner/src/learner.rs` (train signature), `crates/lgbm-model/src/{objective.rs,tree.rs}` (ConvertOutput, Tree fields), `Cargo.toml` (workspace members/deps)

### Tertiary (LOW — none)
- No unverified web sources used; all claims trace to in-tree source.

## Metadata

**Confidence breakdown:**
- GBDT control flow / score updater / objectives / metrics / bagging: **HIGH** — read verbatim from the in-tree C++ 4.6 source this session.
- Crate placement / public API shape: **HIGH** for the C++-mirror constraints, **MEDIUM** for the exact crate split (Claude's discretion per CONTEXT; recommendation is grounded in the C++ directory structure + Phase-8 binding seam).
- D-13 bagged-index capture mechanism: **MEDIUM** — Option A (RNG-replay) is HIGH-confidence (builds on proven FND-01); a direct LightGBM bag dump is NOT exposed by the Python API (LOW for Option B).
- D-11 per-iter score capture fidelity: **MEDIUM** — flagged as Open Question 2 / Assumption A4 to resolve in the spine wave.

**Research date:** 2026-06-07
**Valid until:** 2026-07-07 (stable — the C++ 4.6 reference is pinned/read-only; the only moving piece is the Rust workspace, which the planner re-checks against live code)
