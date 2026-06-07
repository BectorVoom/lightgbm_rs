# Phase 7: Parity-Completing Variants - Research

**Researched:** 2026-06-07
**Domain:** Gradient-boosting parity completion (boosting variants, objectives, metrics, categorical splits, ranking, SHAP, advanced constraints) — pure-Rust faithful port of C++ LightGBM 4.6, f32 end-to-end, `cubecl-cpu` f64-fold bit-exact hard gate.
**Confidence:** HIGH for stack/architecture/pitfalls (the entire spec is the read-only C++ tree under `LightGBM/`, read directly at file:line); MEDIUM for the D-05 root-cause outcome branch (requires the gating diagnostic wave to confirm which branch fires).

## Summary

Phase 7 is large by requirement count (18) but structurally simple: every item is an **addition** on the already-bit-exact Phase 1→6 spine. The GBDT loop, the serial tree learner, the score updater, model-text I/O, the objective/metric trait+factory seams, the `Config` bag, the `Random` LCG, and the public builder are all shipped and locked. The C++ source is the authoritative spec; nearly every Phase-7 datum in this document is cited at `LightGBM/...:line` and is therefore HIGH confidence — there is no external package to discover (the only new crates are zero, see Package Legitimacy Audit).

Three items carry real risk and shape the wave DAG. (1) The **bagged-subset split-gain knife-edge** (DEF-06-01 + the typed-rejected `regression_l1+bagging`) gates everything that bags — GOSS and RF *must* bag — so it runs FIRST as a diagnostic wave. (2) **Categorical splits** (TRL-06) re-open the bit-exact serial learner; the numeric-spine real-binary goldens must stay bit-exact after the re-open (D-06 hard invariant). (3) The **ranking stack** (OBJ-06 + MET-04 + `bagging_by_query` + DCGCalculator) couples objectives, metrics, query infrastructure, and a third RNG-replay golden, and lands as one coherent group. Everything else (objectives/metrics breadth, DART, SHAP, pred-early-stop, monotone/interaction/forced-splits/extra-trees/CEGB/refit/importance) is a thin vertical slice: Config param (mostly already present) → factory wiring → faithful math → builder setter → real-`lib_lightgbm`-4.6 oracle golden.

**Primary recommendation:** Plan as ~11–13 dependency-ordered waves in one phase (D-01): Wave 0 = bagged-subset determinism diagnostic (D-05) → objectives+metrics breadth (lowest risk, builds confidence) → DART/GOSS/RF → categorical re-open → ranking stack → prediction modes → advanced features, each a one-axis-at-a-time validated slice against a committed real-binary golden, with the full per-subsystem oracle axis matrix below. **Capture blocker:** the `lightgbm==4.6.0` wheel is NOT installed here (and 6 Phase-6 goldens already await capture); plan every golden as capture-gated with a `checkpoint:human-verify` capture step, exactly as Phase 5/6 did.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GOSS sampling (BST-04) | `lgbm-boosting` sample strategy | `lgbm-core::Random` | GOSS is a `SampleStrategy` sibling of bagging (`goss.hpp` extends `SampleStrategy`); modifies grad/hess (IsHessianChange) before the loop |
| DART (BST-05) | `lgbm-boosting` (boosting variant) | `lgbm-model` (predict-side normalize) | DART subclasses GBDT, overrides TrainOneIter/GetTrainingScore + tree-weight normalize (`dart.hpp`) |
| Random Forest (BST-06) | `lgbm-boosting` (boosting variant) | `lgbm-model` | RF subclasses GBDT (`rf.hpp`): averaged not accumulated, `MultiplyScore` rescale, mandatory bagging, no shrinkage |
| Categorical splits (TRL-06) | `lgbm-treelearner` (additive split branch) | `lgbm-model` (SplitCategorical node + model-text), `lgbm-dataset` (cat binning, already done) | The ONLY Phase-7 item that re-opens the bit-exact learner (D-06) |
| Remaining objectives (OBJ-04/05/06) | `lgbm-objective` | `lgbm-boosting::BoostObjective` enum, `lgbm-dataset` (query boundaries for rank) | New objective types behind the `GetGradients`/`BoostFromScore`/`ConvertOutput` trait |
| Metrics (MET-03/04) | `lgbm-metric` | `lgbm-metric::DCGCalculator` (new), `lgbm-dataset` query boundaries | New metric types behind `Metric::Eval`; rank metrics add per-query + DCGCalculator |
| SHAP / contrib (PRD-04) | `lgbm-model` (Tree node/cover + ensemble) | — | TreeSHAP recursion over node structure + `leaf_count`/`internal_count` cover (`tree.h:668`) |
| Pred early stop (PRD-05) | `lgbm-model::predict` driver | `lgbm-boosting` (NeedAccuratePrediction) | Prediction-side accumulation hook (`predictor.hpp`/`prediction_early_stop.cpp`) |
| Monotone/interaction/forced/extra/CEGB (ADV-01..05) | `lgbm-treelearner` (split-path constraints) | `lgbm-core::Config` | Constraints alter which split is chosen — all in the learner |
| Refit / continue (ADV-06) | `lgbm-boosting` + `lgbm-model` | `lgbm` public API | RefitTree + `input_model` continue reuse Phase-3 model I/O |
| Feature importance (ADV-07) | `lgbm-model` (Tree/ensemble) | `lgbm` public API | split/gain counts over the grown trees |

---

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** One phase, many waves. Phase 7 stays a single phase planned as a long sequence of small, dependency-ordered waves (the Phase-5 ~9-plan model), with ONE verification gate at the end. NOT split into sub-phases.
- **D-02:** Wave ordering is dependency-forced, low-risk-first (spine-first). Indicative order: (1) bagged-subset split-gain determinism (early gating wave) → (2) objective+metric breadth (OBJ-04/05, MET-03) → (3) boosting variants (GOSS/DART/RF) → (4) categorical splits (TRL-06) → (5) ranking stack (OBJ-06 + MET-04 + `bagging_by_query` + DCGCalculator) → (6) prediction modes (PRD-04/05) → (7) advanced features (ADV-01..07). Researcher proposes the EXACT DAG; plan-checker verifies.
- **D-03:** Six work-groups + the early determinism wave: (1) boosting variants, (2) categorical, (3) objectives+metrics breadth, (4) ranking stack, (5) prediction modes, (6) advanced features. Each group is a sequence of one-axis-at-a-time validated additions, never big-bang.
- **D-04:** Full cross-product (Phase-6 maximal-fidelity ethos) over per-subsystem RELEVANT axes — exhaustive committed-real-binary-golden discipline, crossed only over axes that actually change each subsystem's output, no provably-redundant cells. Planner may collapse a cell ONLY when provably byte-identical to another, documented (never silent truncation).
- **D-05:** Make the bagged-subset split-gain determinism a dedicated EARLY diagnostic wave (runs BEFORE GOSS/RF which MUST bag, and before un-deferring L1+bagging). Two outcome branches: faithful fix → un-defer `regression_l1+bagging` (remove the `BoostingError::UnsupportedConfig` typed-reject) + clear DEF-06-01; OR genuine f32/accumulation-order artifact → document as bounded known-divergence with a hard structural-divergence cap (carry `struct_divergent <= 1`).
- **D-06:** Categorical splits are an ADDITIVE branch in the Phase-5 serial learner; the numeric spine stays byte-untouched + bit-exact. **HARD INVARIANT:** the existing numeric-spine real-`lib_lightgbm`-4.6 goldens (`spine_real.txt`, `mfb_pos_real.txt`, growth-path/subtract gates) MUST still pass bit-exact after the re-open.
- **D-07:** Categorical gets its own real `lib_lightgbm` 4.6 corpus (synthetic dataset with categorical features, reusing the Phase-2 bit-exact categorical binning path) + per-split layered diagnostics: per-category gain arrays, chosen category bitset, split decision_type, and model-text round-trip of the `||`-separated category set in the `.txt` schema.

### Claude's Discretion

- The exact wave DAG / plan boundaries within the 6 groups + early determinism wave (bounded by dependency-forced, low-risk-first).
- The precise per-subsystem axis enumeration (bounded by "every axis that can change the subsystem's output, no provably-redundant cells").
- Crate placement for new variants/objectives/metrics (bounded by the existing factory seams).
- Whether GOSS sampling and DART drop-selection each get a dedicated RNG-replay golden (strongly recommended; exact golden shape is the researcher's call).
- The ranking-stack internal grouping (bounded by "shared query infrastructure lands together").
- The refit/continue-training (ADV-06) boundary (`input_model` continue vs `refit_decay_rate` leaf-refit) and which model-I/O hooks it reuses from Phase 3.
- Whether any Phase-7 subsystem warrants a ROCm cross-check vs CPU-bit-exact-only (CPU bit-exact is the hard gate; ROCm re-check is a research/planning call).
- When C++ behavior is the spec, the C++ source is authoritative over any inferred default.

### Deferred Ideas (OUT OF SCOPE)

- Python/PyO3 bindings — Phase 8 (Phase-7 builder API shaped to map 1:1).
- Parallel (rayon) CPU / multi-GPU boosting path — post-MVP optimization; must still match the deterministic anchor when added.
- ROCm cross-check of Phase-7 subsystems — research/planning call; CPU bit-exact is the hard gate.
- Out-of-milestone subsystems (distributed/network learners, linear-tree leaves, GPU-specific tree learner) — not Phase 7.

---

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BST-04 | GOSS sample strategy (`top_rate`/`other_rate`, grad-magnitude sort + amplification) | `goss.hpp` ported; sample-strategy seam in `lgbm-boosting`; RNG-replay golden spec below |
| BST-05 | DART (`drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`/`drop_seed`) | `dart.hpp` ported; boosting-variant seam + predict-side normalize; RNG-replay golden spec below |
| BST-06 | Random Forest (averaged trees, mandatory bagging, no shrinkage) | `rf.hpp` ported; `MultiplyScore` rescale + `average_output` + per-tree renew |
| TRL-06 | Categorical splits (`SplitCategorical`/`FindBestThresholdCategorical`) | `feature_histogram.cpp:144-382` + `serial_tree_learner.cpp:807-843` hook + `tree.cpp` cat model-text — full axis matrix below |
| OBJ-04 | `huber`/`fair`/`poisson`/`quantile`/`mape`/`gamma`/`tweedie` | `regression_objective.hpp` class-per-objective; axis matrix below |
| OBJ-05 | `cross_entropy`/`cross_entropy_lambda` | `xentropy_objective.hpp` |
| OBJ-06 | `lambdarank`/`rank_xendcg` (query boundaries, DCGCalculator, `objective_seed`) | `rank_objective.hpp` + DCGCalculator + query boundaries from `lgbm-dataset` |
| MET-03 | Extended regression/xentropy/multiclass metrics | `regression_metric.hpp`/`xentropy_metric.hpp`/`multiclass_metric.hpp` |
| MET-04 | `ndcg`/`map` (DCGCalculator, `eval_at`/`ndcg_eval_at`, per-query) | `rank_metric.hpp` + `dcg_calculator.cpp` |
| PRD-04 | TreeSHAP `predict_contrib` over node/cover | `tree.h:668` PredictContrib + TreeSHAP; `Tree` already has `leaf_count`/`internal_count`/`leaf_weight` cover |
| PRD-05 | Pred early stopping (`pred_early_stop`/`_freq`/`_margin`) | `predictor.hpp`/`prediction_early_stop.cpp`; params already in Config |
| ADV-01 | Monotone constraints (basic/intermediate/advanced + `monotone_penalty`) | `monotone_constraints.hpp` + `serial_tree_learner.cpp`; GetSplitGains has the `monotone_constraint` arg already wired in the kernel |
| ADV-02 | Interaction constraints (`interaction_constraints`) | `serial_tree_learner.cpp` feature-allowed gate; `interaction_constraints_vector` in Config |
| ADV-03 | Forced splits / forced bins (JSON-driven) | `serial_tree_learner.cpp` ForceSplits + `GatherInfoForThreshold*` (`feature_histogram.hpp:474`) |
| ADV-04 | Extra trees (`extra_trees`/`extra_seed`, randomized thresholds) | the `USE_RAND` template branch in `FindBestThresholdSequentially` (`feature_histogram.hpp:895,1151`) + categorical `rand_threshold` |
| ADV-05 | CEGB (`cegb_tradeoff`, penalties) | `cost_effective_gradient_boosting.hpp` |
| ADV-06 | Refit / continue training (`refit_decay_rate`, `input_model`) | `gbdt.cpp` RefitTree + Phase-3 model load |
| ADV-07 | Feature importance (split/gain, `saved_feature_importance_type`) | `tree.cpp`/`gbdt.cpp` FeatureImportance (CR-02 Phase-3 follow-up: `split_gain>0` guard) |
| bagging_by_query | query-grouped row subsampling (deferred from BST-03) | `bagging.hpp:52-104` query path; ships with the ranking stack |

---

## Standard Stack

### Core

**No new external dependencies.** Phase 7 is implemented entirely with crates and patterns already in the workspace. The "stack" is the existing internal crate set extended through its established seams.

| Crate | Role in Phase 7 | Extends Via |
|-------|-----------------|-------------|
| `lgbm-core` | Config params (most already present) + `Random` LCG (GOSS/DART/bagging_by_query RNG) | `config/set.rs` + `config/alias.rs` + `random.rs` |
| `lgbm-boosting` | GOSS strategy, DART/RF boosting variants, `bagging_by_query` | `sample_strategy.rs`, `gbdt.rs`, new variant modules; `BoostObjective` enum (`objective.rs`) for new objectives |
| `lgbm-objective` | huber/fair/poisson/quantile/mape/gamma/tweedie/xentropy/rank | new modules behind the existing objective trait |
| `lgbm-metric` | extended + ranking metrics + `DCGCalculator` | new modules behind `Metric::Eval`; new `dcg_calculator.rs` |
| `lgbm-treelearner` | categorical split branch + monotone/interaction/forced/extra/cegb constraints | additive branch in `learner.rs` split path |
| `lgbm-model` | categorical SplitCategorical node + model-text, TreeSHAP, feature importance, refit | `tree.rs` (cat fields already present), `predict.rs`, `model_text.rs` |
| `lgbm-dataset` | query/group boundaries (ranking + bagging_by_query); cat binning (already bit-exact) | `metadata.rs` (already has query_boundaries) |
| `lgbm` | one builder setter per new param/variant | `builder.rs` + `booster.rs` |
| `oracle-harness` + `xtask` | real-`lib_lightgbm`-4.6 capture + replay per subsystem | new capture subcommands + parity test files |

[VERIFIED: codebase grep] Most Phase-7 Config params already exist in `crates/lgbm-core/src/config/set.rs` (`top_rate`, `drop_rate`, `cat_smooth`, `monotone_penalty`, `extra_trees`, `data_sample_strategy`, `boosting` with the `goss`→`gbdt`+`data_sample_strategy=goss` alias-expansion at `set.rs:472-475`). The `Tree` struct already carries `num_cat`/`cat_boundaries`/`cat_threshold`/`leaf_weight`/`leaf_count`/`internal_count` and a working categorical `Decision`/`get_leaf` (Phase-3 DAT-08 model load). [VERIFIED: `crates/lgbm-model/src/tree.rs:64-93,191-193`]

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `Dart`/`Rf` structs subclassing GBDT logic | Single `Gbdt` with a `BoostingVariant` enum + variant hooks | C++ uses subclassing (`dart.hpp`/`rf.hpp : public GBDT`). Rust enum-dispatch on a `variant` field inside `Gbdt::train_one_iter` is the faithful-and-idiomatic choice; avoids trait-object churn. RECOMMEND: enum field on `Gbdt` + branch in `train_one_iter`/`get_training_score`, mirroring each C++ override 1:1. |
| `DCGCalculator` static tables | Recompute discounts per query | C++ precomputes a static discount table (`dcg_calculator.cpp`); recompute drifts FP. RECOMMEND: faithful static table init (matches `DCGCalculator::Init`). |
| Sigmoid recompute per pair (lambdarank) | Sigmoid lookup table | C++ uses a `_sigmoid_bins=1024` lookup table (`rank_objective.hpp:269-292`). MUST mirror the table (recompute diverges). |

**Installation:** none — `cargo build` over the existing workspace. No `npm`/`pip`/`cargo add`.

## Package Legitimacy Audit

**No external packages are installed in Phase 7.** Every requirement is satisfied by extending existing workspace crates. slopcheck / registry verification is **N/A** (no new dependency surface). If a planner later proposes a crate (e.g. a JSON parser for ADV-03 forced-splits), gate it behind a `checkpoint:human-verify` and run the Package Legitimacy Gate then — but the C++ reference uses a vendored `json11` and the Rust side can hand-roll the small forced-splits JSON schema or reuse `serde_json` only if already a workspace dep (verify before adding).

**Packages removed due to slopcheck [SLOP] verdict:** none (no packages).
**Packages flagged as suspicious [SUS]:** none.

---

## Architecture Patterns

### System Architecture Diagram

```
                       ┌──────────────────────────────────────────────┐
  TrainingBuilder ───▶ │ lgbm::Booster  (public API, one setter/param) │
  (one setter/variant) └───────────────────────┬──────────────────────┘
                                                │ Config (lgbm-core: params+alias+CHECK)
                                                ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ lgbm-boosting::Gbdt  +  BoostingVariant {Gbdt,Dart,Rf}      │
                    │   train_one_iter:                                           │
                    │   ① SampleStrategy.bagging(iter, grad, hess) ──────────┐    │
                    │      {Bagging | GOSS(modifies grad/hess) | by_query}   │    │
                    │   ② BoostObjective.get_gradients(score,label)→grad,hess│    │
                    │   ③ per class: TreeLearner.train(grad,hess)            │    │
                    │   ④ RenewTreeOutput (L1/quantile/mape) over partition  │    │
                    │   ⑤ ScoreUpdater += shrinkage·tree   (RF: rescale)     │    │
                    │   ⑥ DART: drop+normalize tree weights (drop_seed RNG)  │    │
                    └───────────┬───────────────────────────────────────┬───┘    │
                                │                                        │        │
              grad/hess ◀───────┘  lgbm-objective                        │        │
            ┌──────────────────────────────────────────┐                │        │
            │ huber/fair/poisson/quantile/mape/gamma/   │                │        │
            │ tweedie/xentropy/lambdarank/rank_xendcg   │                │        │
            │  (GetGradients/BoostFromScore/Convert)    │   query boundaries (lgbm-dataset)
            │  rank → DCGCalculator + objective_seed    │◀───────────────┘        │
            └──────────────────────────────────────────┘                         │
                                ▼ Tree*                                           │
            ┌───────────────────────────────────────────────────────────────────▼─┐
            │ lgbm-treelearner::SerialTreeLearner   find_best_split per feature:    │
            │   if bin_type==Numerical → FindBestThresholdSequentially (SPINE,      │
            │                            bit-exact, BYTE-UNTOUCHED — D-06 invariant) │
            │   else                   → FindBestThresholdCategorical (NEW branch)   │
            │   + monotone / interaction-allowed / forced-splits / extra-rand /cegb  │
            └───────────────────────────────────┬───────────────────────────────────┘
                                                 ▼
            ┌──────────────────────────────────────────────────────────────────────┐
            │ lgbm-model::Tree   Split | SplitCategorical (cat_threshold bitset)      │
            │   model-text (cat_boundaries/cat_threshold, `||` set) · feature import. │
            │ lgbm-model::predict   raw | transform | leaf | CONTRIB(TreeSHAP) | early│
            └──────────────────────────────────────────────────────────────────────┘
                                                 ▼
            oracle-harness: per-subsystem real-lib_lightgbm-4.6 golden + RNG-replay
```

### Pattern 1: Boosting variant as an enum field on `Gbdt` (BST-05/06)
**What:** Add a `variant: BoostingVariant` field to `Gbdt`; branch inside `train_one_iter`/`get_training_score`/`rollback` mirroring each C++ override.
**When to use:** DART and RF (which subclass GBDT in C++).
**Key C++ behaviors to mirror exactly:**
- **RF** (`rf.hpp`): `average_output_=true`; `shrinkage_rate_=1.0`; per-tree `MultiplyScore(cur_tree_id, iter+num_init); UpdateScore; MultiplyScore(1/(iter+num_init+1))` running average (`rf.hpp:157-159`); `Boosting()` re-derives grad/hess from a constant init-score buffer once (`rf.hpp:90-109`); `RenewTreeOutput` with a `residual_getter` of `label-pred` (`rf.hpp:150-152`); RF requires `objective != null` (`rf.hpp:91-93`); the CHECK that bagging OR feature_fraction is active (`rf.hpp:35-40`).
- **DART** (`dart.hpp`): `TrainOneIter` calls `GBDT::TrainOneIter` then `Normalize()` then pushes `tree_weight_` (`dart.hpp:58-71`); `GetTrainingScore` triggers `DroppingTrees()` once per iter (`dart.hpp:78-86`); `DroppingTrees` uses `random_for_drop_ = Random(drop_seed)` with the EXACT draw order: first `NextFloat()<skip_drop`, then per-tree `NextFloat()<drop_rate*tree_weight*inv_avg` (uniform_drop=false) or `<drop_rate` (uniform) (`dart.hpp:99-128`); `Normalize()` 3-step shrinkage with the xgboost_dart_mode branch (`dart.hpp:158-197`).

### Pattern 2: GOSS as a SampleStrategy sibling (BST-04)
**What:** Add a `GossSampleStrategy` alongside `BaggingSampleStrategy`; `Gbdt` selects on `data_sample_strategy`.
**Key C++ behaviors (`goss.hpp`):** `IsHessianChange()==true` (grad/hess get modified in place → `need_resize_gradients_` when `objective==null`); skip subsampling for `iter < 1/learning_rate` (`goss.hpp:33`); per-block `Random(bagging_seed+i)` block 1024 (same as bagging, `goss.hpp:95-98`); `Helper` computes `top_k=max(1, cnt*top_rate)`, `other_k=cnt*other_rate`, `ArgMaxAtK` to find the top-k threshold, then per-row: if `|grad*hess|>=threshold` keep (left), else `NextFloat()<prob` keep + **amplify grad AND hess by `multiply=(cnt-top_k)/other_k`** (`goss.hpp:127-162`). The `prob` is the running `rest_need/rest_all` — **draw order is load-bearing** (RNG-replay golden).
**Anti-pattern:** Do NOT pre-sort and pick top-k by full sort; C++ uses `ArgMaxAtK` (nth_element semantics) which can differ in ties — mirror it.

### Pattern 3: Additive categorical branch in the serial learner (TRL-06, D-06)
**What:** In `learner.rs` split path, dispatch on the feature's `bin_type`: `Numerical` → the existing bit-exact `FindBestThresholdSequentially` (BYTE-UNTOUCHED); `Categorical` → a NEW `find_best_threshold_categorical`.
**Hook point (C++):** `serial_tree_learner.cpp:779-781` decides `is_numerical_split` from `FeatureBinMapper(...)->bin_type()`; categorical takes the `else` branch (`:807-843`) building `cat_bitset_inner`/`cat_bitset` via `Common::ConstructBitset` and calling `tree->SplitCategorical` + `data_partition_->Split` with the bitset. The dispatch in find-best is in `feature_histogram.hpp:165` `FindBestThreshold` → numerical vs `FindBestThresholdCategoricalInner`.
**Categorical gain algorithm (`feature_histogram.cpp:144-382`):**
- `gain_shift` uses the ORIGINAL l2 for `min_gain_to_split`, but the per-category gain uses `l2 += cat_l2` (`:166-168,248`) — a deliberate asymmetry (`:163-165` comment).
- `use_onehot = num_bin <= max_cat_to_onehot` (`:179`).
- **one-hot path** (`:184-238`): scan each bin as a single-category split (one-vs-rest), gate `min_data_in_leaf`/`min_sum_hessian_in_leaf`, gain via `GetSplitGains`, best → `cat_threshold=[best_bin+offset]`, `num_cat_threshold=1`.
- **many-vs-many path** (`:239-339`): filter bins where `RoundInt(hess*cnt_factor) >= cat_smooth` into `sorted_idx`; `stable_sort` by `ctr_fun = sum_grad/(sum_hess + cat_smooth)` ascending (`:250-257`); try BOTH directions (`dir=+1` from front, `dir=-1` from back, `:259-262`); `max_num_cat = min(max_cat_threshold, (used_bin+1)/2)` (`:263-264`); accumulate groups respecting `min_data_per_group` (`:276-313`); best → `cat_threshold` = first `best_threshold+1` entries of `sorted_idx` (forward) or `sorted_idx[used_bin-1-i]` (backward) (`:364-378`).
**HARD INVARIANT (D-06):** the numeric `Numerical` branch must remain identical — gate the new code behind the `bin_type` dispatch so `spine_real.txt`/`mfb_pos_real.txt`/growth-path/subtract goldens replay bit-exact unchanged.

### Pattern 4: Faithful objective math, one class per objective (OBJ-04/05/06)
**What:** Each objective is a struct mirroring its C++ class's `GetGradients`/`BoostFromScore`/`ConvertOutput`/`RenewTreeOutput`. Most regression objectives subclass `RegressionL2loss` in C++ (`regression_objective.hpp:207,293,351,398,481,579,680`); mirror the inheritance as shared helpers, not literal inheritance.
**Objective-specific load-bearing detail:**
- `huber` (`:293`): grad clipped to `±alpha` (the `alpha` param at `config.h:958` is the huber δ — SHARED with quantile's α; the C++ comment says "used only in huber and quantile").
- `fair` (`:351`): `fair_c` (`config.h:963`).
- `poisson` (`:398`): `poisson_max_delta_step` (`config.h:968`); grad=`exp(score)-label`, hess=`exp(score)*max_delta_step`-ish; `BoostFromScore=SafeLog(L2 boost)` (`:469`).
- `quantile` (`:481`): `alpha` (`config.h:958`); RenewTreeOutput uses the weighted/unweighted percentile macro (`:18-88`) — `lgbm-objective::percentile.rs` already exists from Phase 6.
- `mape` (`:579`): subclasses L1; needs label-magnitude weighting.
- `gamma` (`:680`): subclasses Poisson.
- `tweedie`: `tweedie_variance_power` (`config.h:976`, range [1,2)).
- `cross_entropy`/`cross_entropy_lambda` (`xentropy_objective.hpp`): label in [0,1]; `BoostFromScore` + sigmoid Convert.
- `lambdarank`/`rank_xendcg` (`rank_objective.hpp`): pairwise lambdas over `query_boundaries`, `DCGCalculator` discounts/gains, `objective_seed` (`config.h:920`, default 5) for rank_xendcg's per-query random gamma draws, sigmoid lookup table (`:269-292`), `lambdarank_truncation_level`/`lambdarank_norm`/`label_gain`.

### Anti-Patterns to Avoid
- **Re-sorting where C++ uses nth_element/ArgMaxAtK** (GOSS top-k, percentile). Tie behavior differs.
- **Recomputing the sigmoid/DCG table per call** instead of the precomputed table — FP drift.
- **Touching the numeric split scan** while adding categorical (D-06 violation).
- **Always-direct histogram** on the bagged subset instead of the subtraction trick where C++ subtracts — low-bit drift (carried CONCERNS.md hazard).
- **Skipping the OOB rows in scoring** — Phase-6 Pitfall 4: OOB rows are still scored.
- **Using `f64` literals where C++ casts an `f32` constant** (e.g. the zero-sentinel `1e-35f as f64`, Phase-5 Fix D).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-block RNG for GOSS/bagging_by_query | A fresh RNG abstraction | `lgbm-core::Random` (FND-01, bit-exact) + the block-1024 `bagging_seed+i` pattern from `BaggingSampleStrategy` | Already proven bit-exact; GOSS/by_query reuse the identical block convention (`goss.hpp:95-98`, `bagging.hpp:178-181`) |
| DART drop RNG | New RNG | `Random(drop_seed)` (FND-01 LCG) | Same LCG, distinct seed; draw ORDER is the spec |
| Categorical bitset | Custom bitvec | Mirror `Common::ConstructBitset` + `FindInBitset` | `tree.rs` already has `find_in_bitset` (Phase-3); the bitset block layout is the model-text contract |
| Percentile / median residual (quantile/L1/mape renew) | New selection algo | `lgbm-objective::percentile.rs` (Phase-6) + the `PercentileFun`/`WeightedPercentileFun` macros (`regression_objective.hpp:18-88`) | Already ported for L1; quantile/mape reuse |
| Model-text float formatting | New formatter | `lgbm-model::format` `%.17g` (Phase-3) | Categorical thresholds, leaf values, importance all use it |
| Query boundary handling | New metadata path | `lgbm-dataset::metadata` query_boundaries (DAT-06, already present) | ranking + bagging_by_query consume it |
| TreeSHAP cover | Recompute counts | `Tree.leaf_count`/`internal_count`/`leaf_weight` (already on the struct) | `tree.h:181` `data_count` reads these; SHAP `ExpectedValue` weights by cover |

**Key insight:** Phase 7 is almost entirely *wiring existing proven machinery to new faithful math*. The single genuinely-new algorithm with FP risk is the categorical split (and the D-05 knife-edge resolution). Everything else is "port the C++ class verbatim, wire it to the existing factory seam, capture a golden."

## Runtime State Inventory

> Not a rename/refactor/migration phase. **N/A — Phase 7 is additive feature work on a greenfield Rust crate; no stored data, live-service config, OS-registered state, secrets, or stale build artifacts are renamed or migrated.** The only persistent artifacts are committed golden fixtures under `crates/oracle-harness/tests/fixtures/`, which are ADDED (new per-subsystem goldens), never migrated.

## Common Pitfalls

### Pitfall 1: The bagged-subset split-gain knife-edge (D-05 — gates GOSS/RF)
**What goes wrong:** On a bagged SUBSET, one early tree's borderline split is accepted by C++ but rounded out by the Rust `cubecl-cpu` f64-fold (or vice-versa), flipping leaf STRUCTURE (DEF-06-01: `binary_bag1_es0_bfa1` tree 0 → rust 2 vs cpp 4 leaves; `regression_l1_bag1_es0_bfa0` tree 0 → rust:0.0 vs cpp:11.0).
**Why it happens:** The gain comparison is `current_gain <= min_gain_shift` (`feature_histogram.hpp:1169,1281,562,643`). On the subset, `cnt_factor = num_data/sum_hessian` and per-bin `cnt = RoundInt(hess*cnt_factor)` (`:872-873,1122`) are sensitive to the subset's `sum_hessian` fold order. A 1-ULP difference in `sum_hessian` shifts `cnt_factor`, flips a `min_data_in_leaf`/`min_sum_hessian_in_leaf` gate or the gain tie, and changes which split wins.
**Root-cause hypothesis (MEDIUM confidence — the diagnostic wave confirms):** The Rust subset path (`learner.rs:305-336`) re-gathers `f.bins[r]` per in-bag row and folds f32 grad/hess **in in-bag order**, then the f64 histogram fold runs over that order. C++ builds the subset histogram via `tmp_subset_->CopySubrow` then `ConstructHistogram` over the subset's own row order (`goss.hpp:58-59`, `bagging.hpp:120-121`) — and for binary+bfa, `BoostFromAverage` shifts the iter-0 init score which shifts every per-row gradient. The likely root cause is **fold-ORDER faithfulness on the subset**, not a true f32 artifact: the in-bag row ORDER the Rust learner folds may differ from the C++ `CopySubrow` order, OR the `boost_from_average` init-score is applied at a different point. If so → **faithful fix branch** (un-defer L1+bagging, clear DEF-06-01). If the order is already identical and the divergence is pure f32 accumulation in the histogram → **bounded-divergence branch** (`struct_divergent <= 1` cap).
**How to investigate (the Wave-0 diagnostic plan):** (1) Capture a real-`lib_lightgbm`-4.6 FP trace of the subset histogram `sum_hessian`/`sum_gradient` per bin AND the per-split `current_gain`/`min_gain_shift` for `binary_bag1_es0_bfa1` tree 0 (Phase-5 used exactly this real-binary FP-trace technique to close the 2-ULP). (2) Compare the Rust subset histogram + gain cell-for-cell. (3) Localize: bin-order vs init-score-timing vs genuine f32. (4) Branch per D-05.
**Warning signs:** any `*_bag1_*` cell where `rust_leaves != cpp_leaves` on a single tree.

### Pitfall 2: Categorical re-open breaking the numeric spine (D-06)
**What goes wrong:** Restructuring `find_best_split` to add the categorical branch accidentally perturbs the numeric scan (loop bounds, offset, kEpsilon position), and `spine_real.txt`/`mfb_pos_real.txt` stop being bit-exact.
**How to avoid:** Gate the categorical code behind a pure `bin_type` dispatch at the TOP of per-feature find-best (mirroring `serial_tree_learner.cpp:779`); leave the numeric path byte-identical. Add a guard test that asserts `spine_real`/`mfb_pos_real`/`growth_path_subtract` still pass FIRST in the categorical wave (fails-before is impossible here — it's a no-regression gate; assert it green after every categorical commit).

### Pitfall 3: RNG draw-order divergence (GOSS/DART/bagging_by_query)
**What goes wrong:** The right number of draws but the wrong ORDER → different in-bag set / dropped trees, silently producing a plausible-but-wrong model.
**How to avoid:** Dedicated RNG-replay goldens (see RNG-Replay Golden Specs). Phase-6 D-13 proved bagging this way; the executor's CRITICAL fix there was "build `bagging_rands_` once + advance across draws" (STATE.md 06-05) — GOSS reuses the SAME `bagging_rands_` block array, DART uses a single `random_for_drop_`.

### Pitfall 4: `cnt_factor`/RoundInt count reconstruction (categorical + subset)
**What goes wrong:** `data_size_t cnt = RoundInt(hess * cnt_factor)` reconstructs per-bin counts from hessian; with non-constant hessian (binary/multiclass/poisson) the reconstructed count can be off by ±1 vs the true partition count, flipping `min_data_per_group`/`min_data_in_leaf` gates.
**How to avoid:** Use `Common::RoundInt` semantics exactly; the tree's stored `leaf_count`/`internal_count` come from the ACTUAL `data_partition` count (Phase-5 05-04 Rule-1 fix), NOT the reconstructed count — keep that distinction.

### Pitfall 5: Multiclass exp-libm horizon (carried)
**What goes wrong:** softmax/sigmoid `exp` (Rust libm vs the C++ wheel `std::exp`) differs ~1 ULP and flips a knife-edge split at iter ~5-6 (STATE.md 06-04 deviation). Affects poisson/gamma/tweedie/xentropy/rank too (all use exp/log).
**How to avoid:** Where an objective uses `exp`/`log`, expect a documented horizon cap (bit-exact for the early iters, within ORACLE_TOL after) rather than weakening the assertion — the carried Phase-6 carve-out ("bit-exact where the algorithm permits"). Capture goldens at a horizon where trees stay bit-exact.

## Code Examples

### Categorical many-vs-many threshold (port target)
```cpp
// Source: LightGBM/src/treelearner/feature_histogram.cpp:239-339 (CITED)
for (int i = bin_start; i < bin_end; ++i)
  if (Common::RoundInt(GET_HESS(data_, i) * cnt_factor) >= meta_->config->cat_smooth)
    sorted_idx.push_back(i);
used_bin = sorted_idx.size();
l2 += meta_->config->cat_l2;                                   // cat_l2 ONLY here
auto ctr_fun = [&](double g, double h){ return g/(h + meta_->config->cat_smooth); };
std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
  [&](int i,int j){ return ctr_fun(GET_GRAD(data_,i),GET_HESS(data_,i))
                         < ctr_fun(GET_GRAD(data_,j),GET_HESS(data_,j)); });
// two directions (+1 from 0, -1 from used_bin-1); max_num_cat = min(max_cat_threshold,(used_bin+1)/2)
// accumulate groups, respect min_data_per_group/min_data_in_leaf/min_sum_hessian_in_leaf,
// GetSplitGains, keep best; cat_threshold = first best_threshold+1 of sorted_idx (or reversed)
```

### GOSS amplification (port target)
```cpp
// Source: LightGBM/src/boosting/goss.hpp:127-162 (CITED)
data_size_t top_k = std::max(1, (data_size_t)(cnt * config_->top_rate));
data_size_t other_k = (data_size_t)(cnt * config_->other_rate);
ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, tmp_gradients.size(), top_k - 1);
score_t threshold = tmp_gradients[top_k - 1];
score_t multiply  = (score_t)(cnt - top_k) / other_k;     // amplification factor
// per row: grad=Σ|g*h|; if grad>=threshold keep; else if rand<prob keep AND g*=multiply,h*=multiply
```

### DART drop selection (port target — draw order is the spec)
```cpp
// Source: LightGBM/src/boosting/dart.hpp:97-128 (CITED)
bool is_skip = random_for_drop_.NextFloat() < config_->skip_drop;   // draw #0
if (!is_skip) for (int i = 0; i < iter_; ++i)
  if (random_for_drop_.NextFloat() < drop_rate * tree_weight_[i] * inv_average_weight) // draws #1..iter
    drop_index_.push_back(num_init_iteration_ + i);
```

### TreeSHAP entry (port target — cover from leaf_count/internal_count)
```cpp
// Source: LightGBM/include/LightGBM/tree.h:668-677 (CITED)
output[num_features] += ExpectedValue();
if (num_leaves_ > 1) {
  int max_path_len = max_depth_ + 1;
  std::vector<PathElement> unique_path_data(max_path_len*(max_path_len+1)/2);
  TreeSHAP(feature_values, output, 0, 0, unique_path_data.data(), 1, 1, -1);
}
```

---

## Per-Subsystem Oracle Axis Matrix (D-04)

> Full cross-product over RELEVANT axes only. "Loop axes" = `{bagging on/off} × {early_stop on/off} × {boost_from_average on/off}` (the Phase-6 D-07 pattern). A cell is INCLUDED unless provably byte-identical to another (planner documents any collapse). Capture each via real `lib_lightgbm` 4.6 + replay; assert bit-exact on CPU f64-fold where the algorithm permits, ORACLE_TOL where exp/log/horizon caps apply (Pitfall 5).

### Objectives (OBJ-04/05/06)
Each objective × `{bag on/off} × {es on/off} × {bfa on/off}` (8 cells) + objective-specific params:

| Objective | Loop axes | Objective-specific axis | Notes |
|-----------|-----------|------------------------|-------|
| huber | 8 (bag×es×bfa) | `alpha` ∈ {0.9 default, 0.5} | grad clip at ±alpha |
| fair | 8 | `fair_c` ∈ {1.0, 2.0} | |
| poisson | 8 | `poisson_max_delta_step` ∈ {0.7, 0.1} | exp → horizon cap (Pitfall 5); `boost_from_average` is `SafeLog` |
| quantile | 8 | `alpha` ∈ {0.9, 0.1} | RenewTreeOutput percentile; **bag cells gate on D-05** (renew+bag) |
| mape | 8 | — | subclasses L1; **bag cells gate on D-05** (renew+bag) |
| gamma | 8 | — | exp → horizon cap |
| tweedie | 8 | `tweedie_variance_power` ∈ {1.5, 1.1, 1.9} | exp → horizon cap |
| cross_entropy | 8 | — (label in [0,1]) | sigmoid Convert |
| cross_entropy_lambda | 8 | — | |
| lambdarank | `{bag(by_query) on/off} × {es on/off}` (bfa N/A) | `lambdarank_truncation_level`, `lambdarank_norm`, `label_gain`, `sigmoid` | needs query corpus; ndcg eval |
| rank_xendcg | same | `objective_seed` ∈ {5 default, 7} (per-query random draw — RNG-replay) | |

**Provably-collapsible:** for objectives that subclass L2 with identical loop behavior (gamma⊂poisson, mape⊂L1), the `{es,bfa}` interaction is identical to the parent's — but the GRADIENT differs, so do NOT collapse grad/hess goldens; you MAY share the loop-structure assertion. RECOMMEND: keep all 8 loop cells per objective for the layered grad/hess + per-iter score + model goldens (Phase-6 ethos), document any collapse.

### Boosting variants (BST-04/05/06)
| Variant | Defining-param axes | Loop axes crossed | RNG-replay |
|---------|--------------------|--------------------|-----------|
| GOSS | `top_rate` ∈ {0.2,0.1} × `other_rate` ∈ {0.1,0.05} (constraint top+other≤0.5 for subset path, ≤1.0 always) | × `{es on/off}` (GOSS forbids bagging — `goss.hpp:87-89`); × `{bfa on/off}` | YES (sampled+amplified indices) |
| DART | `drop_rate` ∈ {0.1,0.3} × `max_drop` ∈ {50,2} × `skip_drop` ∈ {0.5,0.0} × `uniform_drop` {T,F} × `xgboost_dart_mode` {T,F} | × `{bag on/off}` × `{es: DART overrides EvalAndCheckEarlyStopping}` | YES (drop indices per iter) |
| RF | mandatory bagging: `bagging_fraction` ∈ {0.7} × `bagging_freq` ∈ {1} OR `feature_fraction`<1 | × `{single vs multiclass}` (RF requires obj≠null) | inherits bagging RNG golden |

**DART cross-product note:** `uniform_drop × xgboost_dart_mode` = 4 normalize branches (`dart.hpp:158-197`) — all 4 must be covered (the Normalize 3-step differs per branch).

### Categorical (TRL-06)
Cross: `max_cat_to_onehot` ∈ {4 default (→ one-vs-rest for ≤4 cats), 1 (→ force many-vs-many)} × `cat_smooth` ∈ {10.0, 0.0} × `cat_l2` ∈ {10.0, 0.0} × `min_data_per_group` ∈ {100, 1} × `max_cat_threshold` ∈ {32, 2}. Both directions of the many-vs-many sort are exercised by data shape. Layered diagnostics (D-07): per-category gain array, chosen `cat_threshold` bitset, `decision_type` categorical bit, model-text `||` round-trip. **+ the no-regression numeric-spine gate (spine_real/mfb_pos_real bit-exact).**

### Monotone (ADV-01)
Cross: `monotone_constraints_method` ∈ {basic, intermediate, advanced} × `monotone_penalty` ∈ {0.0, 5.0} × constraint vector {+1, -1, mixed}. (`GetSplitGains` already has the `monotone_constraint` arg + the `left_output`/`right_output` direction check at `feature_histogram.hpp:788-790`.)

### Other advanced (ADV-02..07) — cross OWN params only (NOT loop axes)
| Subsystem | Axes | Layered diagnostic |
|-----------|------|--------------------|
| Interaction (ADV-02) | `interaction_constraints` ∈ {one group, two groups} | allowed-feature set per node |
| Forced splits (ADV-03) | JSON {single forced split, nested left/right} | forced node threshold/feature + the GatherInfoForThreshold gain |
| Extra trees (ADV-04) | `extra_trees` {T} × `extra_seed` ∈ {6, 9} | the random threshold chosen per feature (RNG draw) — RNG-replay candidate |
| CEGB (ADV-05) | `cegb_tradeoff` ∈ {1.0, 0.5} × `cegb_penalty_split` × `cegb_penalty_feature_coupled`/`_lazy` | penalized gain per split |
| Refit (ADV-06) | `refit_decay_rate` ∈ {0.9, 0.0} (leaf-refit) ; `input_model` continue | refit leaf_output = decay·old + (1-decay)·new (`config.h:551`) |
| Importance (ADV-07) | `saved_feature_importance_type` ∈ {0=split, 1=gain} | per-feature split-count / gain-sum (mind CR-02 `split_gain>0` guard) |

### Prediction modes (PRD-04/05) — cross own params
| Mode | Axes | Diagnostic |
|------|------|-----------|
| SHAP (PRD-04) | numeric-only tree × categorical tree × multiclass | per-feature contrib + `ExpectedValue` base; sum(contrib)+base == raw score |
| Pred early stop (PRD-05) | `pred_early_stop_freq` ∈ {10,1} × `pred_early_stop_margin` ∈ {10.0, 2.0} | iters-evaluated count + final score |

---

## RNG-Replay Golden Specs (D-13 pattern, carried)

Each stochastic draw gets a dedicated golden that replays the exact `Random`/LCG draw + call order, asserted bit-exact (i32 indices / f32 floats), exactly like Phase-6 `bag_indices_seed3_frac0.7.txt`.

| Golden | RNG source | Draw/call order to match (C++ cite) | Asserts |
|--------|-----------|--------------------------------------|---------|
| `goss_sampled_seed{S}_top{T}_other{O}.txt` | block `Random(bagging_seed+i)`, block 1024 | per-row in order: `bagging_rands_[idx/1024].NextFloat() < prob` where `prob=rest_need/rest_all` running (`goss.hpp:152`) | in-bag indices + which rows were amplified |
| `dart_drop_seed{S}_iter{N}.txt` | single `Random(drop_seed)` | draw #0 `NextFloat()<skip_drop`, then per-tree `NextFloat()<drop_rate*w*inv_avg` (`dart.hpp:99-128`) | dropped tree indices per iteration |
| `bagging_by_query_seed{S}.txt` | block `Random(bagging_seed+i)` over QUERIES not rows | `BaggingHelper` over `num_queries_`, then expand each in-bag query to its row range (`bagging.hpp:53-103`) | sampled query indices + expanded row indices + `sampled_query_boundaries` |
| (optional) `extra_trees_seed{S}.txt` | `meta_->rand.NextInt` per feature | the random threshold index chosen in the `USE_RAND` branch (`feature_histogram.hpp:895-898,1268-1271`, categorical `:187,268`) | chosen threshold per feature |

**Implementation note (carried CRITICAL fix):** build the `bagging_rands_` block array ONCE and advance it across draws (do not re-seed per draw) — STATE.md 06-05 records this was the fix that made bagging bit-exact. GOSS and bagging_by_query reuse this exact pattern; DART uses a single advancing `random_for_drop_`.

---

## Categorical Model-Text + Crate Placement (D-06/D-07)

**Node representation** (`tree.h:495-499`, already in Rust `tree.rs:90-93`):
- `num_cat_` (count), `cat_boundaries_` (len num_cat+1, prefix offsets into the bitset), `cat_threshold_` (uint32 bitset blocks), plus `cat_boundaries_inner_`/`cat_threshold_inner_` (the bin-space bitset for fast prediction). `decision_type_` bit0 = `kCategoricalMask`.
- `SplitCategorical` (`tree.h:86`, `tree.cpp:62-95`): sets the categorical decision bit, `threshold_in_bin_ = threshold_ = num_cat_`, pushes both `cat_threshold_` (real) and `cat_threshold_inner_` (bin), appends to `cat_boundaries_`/`cat_boundaries_inner_`, `++num_cat_`.

**Model-text schema** (`tree.cpp:346,371-378`): `num_cat=N`, `cat_boundaries=...`, `cat_threshold=...` lines; the human-readable if-else uses `||`-separated category sets (`tree.cpp:550-557` `CategoricalDecisionIfElse`). [VERIFIED: codebase grep] Rust `model_text.rs`/`tree.rs::to_string` currently emits numeric-only; Phase 7 ADDS the `num_cat`/`cat_boundaries`/`cat_threshold` lines and a `Tree::split_categorical` constructor.

**Crate placement per item** (Claude's-discretion seam map):

| Item | Crate · seam |
|------|-------------|
| GOSS | `lgbm-boosting::sample_strategy` (new `GossSampleStrategy`); selected in `Gbdt` on `data_sample_strategy=="goss"` |
| DART | `lgbm-boosting::gbdt` (new `BoostingVariant::Dart` field + branch); predict-side normalize touches `lgbm-model` tree weights |
| RF | `lgbm-boosting::gbdt` (`BoostingVariant::Rf`: `average_output`, MultiplyScore rescale) |
| bagging_by_query | `lgbm-boosting::sample_strategy` (query branch in `BaggingSampleStrategy`; remove the typed `BaggingByQueryDeferred` reject) |
| Categorical find-best | `lgbm-treelearner::learner` (additive `bin_type` branch) + new `feature_histogram`-analog cat module |
| Categorical node/model-text | `lgbm-model::tree` (`split_categorical`) + `model_text` |
| Objectives OBJ-04/05/06 | `lgbm-objective` (new modules) + `lgbm-boosting::objective::BoostObjective` enum variants |
| Metrics MET-03/04 | `lgbm-metric` (new modules) + new `lgbm-metric::dcg_calculator` |
| DCGCalculator | shared by OBJ-06 + MET-04 → `lgbm-metric::dcg_calculator` (C++ `DCGCalculator` lives in `src/metric/`) |
| SHAP | `lgbm-model::predict` (TreeSHAP) + `lgbm-model::tree` (ExpectedValue/cover) |
| Pred early stop | `lgbm-model::predict` driver |
| Monotone/interaction/forced/extra/cegb | `lgbm-treelearner::learner` split path |
| Refit | `lgbm-boosting` (RefitTree) + `lgbm-model` (model load, Phase-3) + `lgbm` API |
| Importance | `lgbm-model` (Tree/ensemble) + `lgbm` API |
| Config params | `lgbm-core::config` (set.rs + alias.rs); most already present |
| Builder setters | `lgbm::builder` (one per new param/variant) |
| Capture/replay | `oracle-harness` + `xtask` (new subcommands + parity tests) |

---

## Config Param Tables (mirror verbatim from config.h)

[VERIFIED: codebase grep] Items marked **(present)** already exist in `crates/lgbm-core/src/config/set.rs`. Items marked **(ADD)** must be added with the exact name/default/CHECK from `config.h`/`config.cpp`.

| Param | Default | CHECK | config.h | Status |
|-------|---------|-------|----------|--------|
| top_rate | 0.2 | [0,1] | :469 | present |
| other_rate | 0.1 | [0,1]; top+other≤1 | :475 | present |
| drop_rate | 0.1 | [0,1] | :440 | present |
| max_drop | 50 | — | :445 | ADD (verify) |
| skip_drop | 0.5 | [0,1] | :451 | ADD (verify) |
| xgboost_dart_mode | false | — | :455 | ADD (verify) |
| uniform_drop | false | — | :459 | ADD (verify) |
| drop_seed | 4 | — | :463 | ADD (verify) |
| min_data_per_group | 100 | >0 | :480 | ADD (verify) |
| max_cat_threshold | 32 | >0 | :486 | ADD (verify) |
| cat_l2 | 10.0 | ≥0 | :491 | ADD (verify) |
| cat_smooth | 10.0 | ≥0 | :496 | present |
| max_cat_to_onehot | 4 | >0 | :501 | ADD (verify) |
| monotone_constraints | [] | — | :515 | ADD (verify) |
| monotone_constraints_method | "basic" | enum | :525 | present |
| monotone_penalty | 0.0 | ≥0 | :532 | present |
| interaction_constraints | "" | — | :591 / vector :1153 | ADD |
| forced_splits_filename | "" | — | :541 (alias fs/forced_splits) | ADD |
| refit_decay_rate | 0.9 | [0,1] | :553 | ADD |
| cegb_tradeoff | 1.0 | ≥0 | :557 | ADD |
| cegb_penalty_split | 0.0 | ≥0 | :561 | ADD |
| cegb_penalty_feature_lazy | [] | — | :567 | ADD |
| cegb_penalty_feature_coupled | [] | — | :573 | ADD |
| extra_trees | false | — | :390 | present |
| extra_seed | 6 | — | :393 | ADD (verify) |
| saved_feature_importance_type | 0 | — | :615 | ADD |
| pred_early_stop | false | — | :871 | ADD (verify) |
| pred_early_stop_freq | 10 | — | :876 | ADD (verify) |
| pred_early_stop_margin | 10.0 | — | :881 | ADD (verify) |
| objective_seed | 5 | — | :920 | ADD (verify) |
| alpha | 0.9 | >0 (huber & quantile) | :958 | ADD (verify) |
| fair_c | 1.0 | >0 | :963 | ADD (verify) |
| poisson_max_delta_step | 0.7 | >0 | :968 | ADD (verify) |
| tweedie_variance_power | 1.5 | [1,2) | :976 | ADD (verify) |
| eval_at | [] | alias ndcg_eval_at/ndcg_at/map_eval_at/map_at | :1057-1060 | ADD |
| lambdarank_truncation_level | (verify config.h) | | (rank) | ADD |
| lambdarank_norm | (verify) | | (rank) | ADD |
| label_gain | (verify) | | (rank/DCG) | ADD |
| sigmoid | (present? verify) | >0 | | verify |

**Action for planner:** the FIRST plan of each work-group must verify/add its params against `config.h`/`config.cpp`/`config_auto.cpp` (the alias table is `config_auto.cpp`-derived per CFG-02). Do NOT trust this table's "present/ADD" without re-grepping `set.rs` — it is a starting map.

---

## Proposed Wave DAG (D-02 made concrete)

> One phase (D-01). ~12 plans. Each plan = a thin vertical slice (Config → factory wiring → faithful math → builder setter → real-binary golden). Capture-gated (wheel absent — see Environment). Dependencies are forced; low-risk-first within each tier.

```
W0  Bagged-subset determinism DIAGNOSTIC (D-05)   ── GATES W3,W5(renew+bag),un-defer L1+bag
        │  FP-trace binary_bag1_es0_bfa1 + regression_l1_bag1; branch: fix | bounded-cap
        ▼
W1  Objectives breadth A: huber,fair,quantile,mape   (regression family; renew where L1-like)
W2  Objectives breadth B: poisson,gamma,tweedie,xentropy  (exp/log → horizon caps)
        │  (W1,W2 independent of each other; both depend only on the locked spine)
W3  Metrics breadth: MET-03 (regression/xentropy/multiclass metrics)   (needs W1,W2 objectives to score against)
        ▼
W4  GOSS (BST-04)        ── depends W0 (samples), RNG-replay golden
W5  DART (BST-05)        ── depends spine; predict-side normalize; RNG-replay golden
W6  Random Forest (BST-06) ── depends W0 (mandatory bag), W1/W2 (renew)
        ▼
W7  Categorical splits (TRL-06, D-06/D-07)   ── additive learner re-open; numeric-spine no-regression gate; cat corpus
        ▼
W8  Ranking stack: OBJ-06 + MET-04 + DCGCalculator + bagging_by_query   ── shared query infra lands together; RNG-replay (by_query, rank_xendcg seed)
        ▼
W9  Prediction modes: PRD-04 SHAP + PRD-05 pred-early-stop
        ▼
W10 Advanced learner constraints: ADV-01 monotone, ADV-02 interaction, ADV-03 forced, ADV-04 extra-trees, ADV-05 CEGB
W11 Advanced model ops: ADV-06 refit/continue, ADV-07 importance
        ▼
        END: single phase verification gate (D-01)
```

**What-unblocks-what:** W0 unblocks all bagging-dependent variants (W4 GOSS, W6 RF) AND the renew+bag objective cells (W1/W5 quantile/mape on bagged subsets) AND the L1+bagging un-defer decision. W1/W2 unblock W3 (metrics need objectives to score). W7 (categorical) is independent of the boosting variants but must run after the spine is otherwise stable to keep the no-regression gate meaningful. W8 (ranking) needs query boundaries (already present) + DCGCalculator (new, lands in W8). W9/W10/W11 are leaf nodes (nothing depends on them) and can reorder by risk.

**Slice boundaries (MVP mode):** each objective/metric/variant/constraint is its own slice; group small related ones per plan (e.g. W1 = 4 regression objectives as 4 slices in one plan). The natural slice cut is exactly the C++ class boundary.

---

## State of the Art

| Old Approach | Current Approach | When | Impact |
|--------------|------------------|------|--------|
| 1e-12 parity framing | f32 / ~1e-6, CPU f64-fold bit-exact gate | Phase-1 discuss 2026-06-05 | The contract Phase 7 inherits; categorical/objectives held to the same bit-exact-where-possible bar |
| `regression_l1+bagging` shippable | typed-rejected (`UnsupportedConfig`) pending D-05 | Phase-6 06-06 Task 2b | W0 revisits; faithful fix un-defers it |
| Self-transcription oracle | real `lib_lightgbm` 4.6 committed goldens | Phase-5 05-06 | Every Phase-7 golden uses the real binary (capture-gated) |

**Deprecated/outdated:** none relevant — the C++ 4.6 source is the live spec.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | D-05 root cause is fold-ORDER/init-score-timing (faithful-fix branch likely), not pure f32 artifact | Pitfall 1 | If wrong → bounded-divergence branch; the W0 diagnostic plan resolves this definitively before any dependent wave — low risk because it's gated FIRST |
| A2 | Several DART/objective params still need ADDING to `lgbm-core::Config` (table marks "ADD (verify)") | Config Param Tables | Over/under-counting param work; mitigated by "re-grep set.rs first" instruction to planner |
| A3 | No new external crate is needed (forced-splits JSON can be hand-rolled or use an existing workspace dep) | Standard Stack | If a JSON dep is added → run Package Legitimacy Gate at that plan |
| A4 | `lambdarank_truncation_level`/`lambdarank_norm`/`label_gain`/`sigmoid` defaults are as in `rank_objective.hpp` usage | Config table | Verify against config.h at W8 plan time (I cited usage sites, not the config.h default lines) |
| A5 | The 6 Phase-6 reg_sqrt/mf2es goldens + all Phase-7 goldens require a `lightgbm==4.6.0` wheel that is NOT installed here | Environment | Capture steps must be human-gated checkpoints; tests skip-pass until captured (matches Phase-5/6 pattern) |

## Open Questions

1. **Which D-05 branch fires?**
   - Known: the divergence is a single-tree leaf-structure flip on bagged subsets with non-constant gradient (binary+bfa, L1).
   - Unclear: fold-order/init-timing (fixable) vs genuine f32 histogram accumulation (cap it).
   - Recommendation: W0 is a dedicated FP-trace diagnostic plan (Phase-5 real-binary-trace technique); it MUST conclude with a branch decision recorded before W4/W6 start.

2. **Does ROCm need any Phase-7 cross-check?**
   - Known: ROCm gfx1100 is available; CPU f64-fold is the hard gate; ROCm is best-effort ~1e-6.
   - Recommendation: default CPU-bit-exact only for Phase 7 (carried Phase-6 deferral); optionally re-run the categorical + one objective on ROCm as a smoke at the end if time permits — planner's call, not a gate.

3. **Refit boundary (ADV-06):** `input_model` continue-training vs `refit_decay_rate` leaf-refit are two distinct features sharing the `refit` umbrella.
   - Recommendation: treat as two slices in W11; leaf-refit reuses Phase-3 model load + a new per-leaf decay; continue-training reuses `num_init_iteration_` accounting (RF/DART already reference it).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all build/test | ✓ | rustc 1.95.0 (edition 2024) | — |
| `lightgbm==4.6.0` Python wheel | every real-binary golden CAPTURE (xtask) | ✗ | — | Capture is human-gated checkpoint; tests skip-pass until captured (Phase-5/6 pattern). NO fallback for enforcing parity — capture is mandatory before a golden enforces. |
| ROCm / gfx1100 | optional ROCm cross-check (Open Q2) | ✓ | rocminfo present (ROCm 7.x per STATE.md) | CPU f64-fold is the hard gate; ROCm optional |
| `external_libs` (eigen/fmt/...) | building real `lib_lightgbm` 4.6 from source for FP traces | fetchable (memory: external_libs CAN be fetched) | — | W0 FP-trace may need a from-source single-thread build (Phase-5 used this) |

**Missing dependencies with no fallback:** `lightgbm==4.6.0` wheel for golden CAPTURE — but this is the established Phase-5/6 posture: capture is a human-gated checkpoint, replay tests skip-pass cleanly until the golden exists. NOT a blocker for planning or for code-side implementation; only blocks the enforce-parity moment. **Plan every golden with an explicit `checkpoint:human-verify` capture task.**

**Missing dependencies with fallback:** none additional.

## Validation Architecture

> `.planning/config.json` not separately read for this key; STATE shows the project uses real-binary layered parity throughout. Treating nyquist_validation as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cargo test --workspace` (oracle-harness integration tests) |
| Config file | per-crate `Cargo.toml`; fixtures in `crates/oracle-harness/tests/fixtures/` |
| Quick run command | `cargo test -p oracle-harness --test <subsystem>_parity` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req | Behavior | Test type | Command | Exists? |
|-----|----------|-----------|---------|---------|
| BST-04 | GOSS sample+amplify | parity + RNG-replay | `cargo test -p oracle-harness --test boosting_parity goss` | ❌ Wave 0/4 |
| BST-05 | DART drop+normalize | parity + RNG-replay | `... boosting_parity dart` | ❌ W5 |
| BST-06 | RF averaged trees | parity | `... boosting_parity rf` | ❌ W6 |
| TRL-06 | categorical split | parity + layered + no-regression | `cargo test -p oracle-harness --test learner_parity categorical` | ❌ W7 |
| OBJ-04/05 | objective grad/hess+model | layered parity | `... boosting_parity <obj>` | ❌ W1/W2 |
| OBJ-06/MET-04 | ranking+ndcg/map | per-query parity + RNG | `... rank_parity` (new) | ❌ W8 |
| MET-03 | extended metrics | parity | `... metric_parity` (new) | ❌ W3 |
| PRD-04 | SHAP contrib | parity (sum==raw) | `... predict_parity contrib` | ❌ W9 |
| PRD-05 | pred early stop | parity | `... predict_parity early_stop` | ❌ W9 |
| ADV-01..07 | constraints/refit/importance | parity per axis | new parity files | ❌ W10/W11 |

### Sampling Rate
- **Per task commit:** the slice's own `cargo test -p oracle-harness --test <subsystem>_parity`.
- **Per wave merge:** `cargo test --workspace` (catches numeric-spine regression — critical for W7).
- **Phase gate:** full suite green + the D-06 no-regression goldens bit-exact before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] W0 diagnostic harness: real-binary FP-trace capture for `binary_bag1_es0_bfa1` tree-0 subset histogram + per-split gain (new xtask subcommand).
- [ ] New parity test files: `metric_parity.rs`, `rank_parity.rs` (or extend existing); cat cases in `learner_parity.rs`; goss/dart/rf cases in `boosting_parity.rs`; contrib/early-stop in a predict parity file.
- [ ] RNG-replay goldens: goss/dart/bagging_by_query (new fixtures).
- [ ] Capture pipeline: `lightgbm==4.6.0` wheel install (human-gated) before any golden enforces.

## Security Domain

> `security_enforcement` not explicitly false in surfaced config; including. This is an offline numerical library (no network/auth/session). The only relevant ASVS category is input validation at crate boundaries (already the established `thiserror` typed-error discipline, FND-04 / Security V5 cited throughout Phase 2-6).

### Applicable ASVS Categories
| Category | Applies | Standard Control |
|----------|---------|------------------|
| V2 Authentication | no | — |
| V3 Session | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Typed `*Error` (thiserror) at every crate boundary; validate `num_cat`/`cat_boundaries` bounds before bitset indexing (already done in `tree.rs` parse); validate query_boundaries monotonic; validate JSON forced-splits schema (ADV-03); validate `label` range for objectives (the C++ `LabelOutOfRange` guards — poisson/gamma require label≥0, xentropy label∈[0,1], rank label≥0). |
| V6 Cryptography | no | RNG is a deterministic LCG (FND-01) — NOT cryptographic; correct (parity), never use for security |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Mitigation |
|---------|--------|-----------|
| Malformed model-text (cat_threshold OOB) on load | Tampering | bounds-check `cat_boundaries`/`cat_threshold` before bitset index (`tree.rs` already validates; extend for cat in W7) |
| Untrusted forced-splits JSON (ADV-03) | Tampering/DoS | validate schema + feature/threshold ranges before use; typed error on malformed |
| Out-of-range labels feeding exp/log objectives | Tampering → NaN | mirror C++ `LabelOutOfRange` Init guards (poisson/gamma/tweedie/xentropy/rank) as typed errors |

## Sources

### Primary (HIGH confidence) — read directly at file:line
- `LightGBM/src/boosting/goss.hpp` (full), `dart.hpp` (full), `rf.hpp` (full), `bagging.hpp` (full) — boosting variants + sample strategies + bagging_by_query.
- `LightGBM/src/treelearner/feature_histogram.cpp:144-382` + `feature_histogram.hpp:165,458-666,711-828,830-1056,1059-1290` — categorical + numeric split finding, gain math.
- `LightGBM/src/treelearner/serial_tree_learner.cpp:779-843,960-1056` — numeric/categorical split dispatch hook (D-06 re-open point).
- `LightGBM/src/objective/regression_objective.hpp:18-88,93-680+` — percentile macros + per-objective classes.
- `LightGBM/src/objective/rank_objective.hpp:25-292` — ranking objective, DCG usage, sigmoid table, objective_seed.
- `LightGBM/include/LightGBM/tree.h:20,86,140,181,495-518,668-727` — categorical node, SplitCategorical, PredictContrib/TreeSHAP, cover.
- `LightGBM/src/io/tree.cpp:45-95,346-378,468-475,550-557,715` — categorical model-text + JSON + if-else.
- `LightGBM/include/LightGBM/config.h` (param lines cited inline in Config table).
- Codebase: `crates/lgbm-*/src/*` (existing seams), `crates/lgbm-model/src/tree.rs:64-93,191-193,276` (categorical fields already present), `crates/lgbm-boosting/src/objective.rs` (BoostObjective enum), `crates/lgbm-core/src/config/set.rs` (existing params).
- `.planning/codebase/CONCERNS.md` (FP reduction order, kEpsilon, subtraction trick, RNG hazards), `.planning/phases/06.../deferred-items.md` (DEF-06-01), `06-06-SUMMARY.md` (typed-reject), `.planning/STATE.md` (Phase-5/6 history + the bagging RNG-reuse fix).

### Secondary (MEDIUM confidence)
- D-05 root-cause hypothesis (A1): inferred from the C++ subset-build path (`CopySubrow`) vs the Rust `train_on_subset` in-bag fold order — confirmed only by the W0 diagnostic.

### Tertiary (LOW confidence)
- None — no WebSearch was needed; the spec is the local C++ tree.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no external deps; all seams verified by codebase grep.
- Architecture/wave DAG: HIGH — dependency forces are explicit (bagging gates GOSS/RF; categorical re-opens the learner; ranking couples query infra).
- Categorical/objective/variant math: HIGH — ported from the cited C++ file:line.
- Pitfalls: HIGH (carried Phase-5/6 + CONCERNS.md) except D-05 outcome branch: MEDIUM (W0 confirms).
- Config "present/ADD" status: MEDIUM — planner must re-grep `set.rs` (A2).

**Research date:** 2026-06-07
**Valid until:** 2026-07-07 (stable — the C++ 4.6 reference is pinned/read-only; only the workspace seams could shift, and only via Phase-7's own plans).

## RESEARCH COMPLETE

**Phase:** 7 - Parity-Completing Variants
**Confidence:** HIGH (MEDIUM only on the D-05 outcome branch, which the gating Wave 0 resolves before any dependent wave)

### Key Findings
- No new external dependencies — Phase 7 is entirely additive wiring of existing proven crates to faithful C++-ported math; the only FP-risk algorithms are categorical splits (TRL-06) and the D-05 bagged-subset knife-edge.
- The C++ spec for every subsystem was read at file:line (goss/dart/rf .hpp, feature_histogram cat path, rank objective, tree SHAP/categorical, config.h) — the document is prescriptive, not exploratory.
- The wave DAG is dependency-forced: W0 (D-05 diagnostic) gates GOSS/RF/L1+bag; categorical re-open (W7) is the only learner touch and carries a hard numeric-spine no-regression invariant; ranking (W8) lands as one query-coupled group.
- Most Config params already exist in `lgbm-core`; the `Tree` struct already carries categorical fields + working categorical prediction (Phase-3) — Phase 7 adds the SPLIT-FINDING + model-text-emit side.
- Capture blocker: `lightgbm==4.6.0` wheel is NOT installed; every golden is capture-gated (human checkpoint), tests skip-pass until captured — the established Phase-5/6 posture.

### File Created
`.planning/phases/07-parity-completing-variants/07-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | no external deps; seams grep-verified |
| Architecture / wave DAG | HIGH | dependencies explicit and forced |
| Pitfalls | HIGH | carried Phase-5/6 + CONCERNS.md; D-05 branch MEDIUM (W0 resolves) |

### Open Questions
- Which D-05 branch (faithful-fix vs bounded-cap) fires — resolved by the W0 diagnostic plan before W4/W6.
- ROCm cross-check scope (default: CPU-bit-exact only, carried deferral).
- Refit (ADV-06) two-slice boundary (continue vs leaf-refit).

### Ready for Planning
Research complete. The planner has the concrete wave DAG, per-subsystem oracle axis matrix, D-05 root-cause + outcome branches, RNG-replay golden specs, categorical re-open hook points + model-text schema, crate/factory placement per item, and the config param map. Recommend the planner author Wave 0 (D-05 diagnostic) first and make every golden capture-gated.
