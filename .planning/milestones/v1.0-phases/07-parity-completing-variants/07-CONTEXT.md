# Phase 7: Parity-Completing Variants - Context

**Gathered:** 2026-06-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 completes **full single-machine parity**: every remaining boosting variant, objective, metric, constraint, and prediction mode lands as a thin, oracle-validated addition on the proven Phase-1→6 spine. This is the largest phase by requirement count (18 reqs across 5 success criteria), but every item is an *addition* — the GBDT loop, the bit-exact serial learner, the score updater, the model-text I/O, and the Rust-native API are all already shipped and locked.

In scope (BST-04/05/06, TRL-06, OBJ-04/05/06, MET-03/04, PRD-04/05, ADV-01..07):
- **BST-04** — GOSS sample strategy (`top_rate`/`other_rate`, gradient-magnitude sort + amplification).
- **BST-05** — DART boosting (`drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`/`drop_seed`).
- **BST-06** — Random Forest boosting (averaged trees, mandatory bagging, no shrinkage accumulation).
- **TRL-06** — Categorical splits (`SplitCategorical`/`FindBestThresholdCategorical`: `max_cat_threshold`/`cat_smooth`/`min_data_per_group`/`max_cat_to_onehot`/`cat_l2`).
- **OBJ-04** — Remaining regression objectives: `huber`/`fair`/`poisson`/`quantile`/`mape`/`gamma`/`tweedie`.
- **OBJ-05** — Cross-entropy objectives: `cross_entropy`/`cross_entropy_lambda`.
- **OBJ-06** — Ranking objectives: `lambdarank`/`rank_xendcg` (query boundaries, DCGCalculator, `objective_seed`).
- **MET-03** — Extended regression/xentropy metrics: `quantile`/`huber`/`fair`/`poisson`/`mape`/`gamma`/`gamma_deviance`/`tweedie`/`multi_error`/`cross_entropy`/`cross_entropy_lambda`/`kullback_leibler`/`average_precision`/`auc_mu`.
- **MET-04** — Ranking metrics: `ndcg`/`map` (DCGCalculator static tables, `eval_at`/`ndcg_eval_at`, per-query).
- **PRD-04** — Feature contributions / TreeSHAP (`predict_contrib`) over full node/cover structure.
- **PRD-05** — Prediction early stopping (`pred_early_stop`/`_freq`/`_margin`).
- **ADV-01** — Monotone constraints (basic/intermediate/advanced, `monotone_penalty`).
- **ADV-02** — Interaction constraints (`interaction_constraints`).
- **ADV-03** — Forced splits / forced bins (JSON-driven).
- **ADV-04** — Extra trees (`extra_trees`/`extra_seed`, randomized thresholds).
- **ADV-05** — CEGB cost-effective gradient boosting (`cegb_tradeoff`, penalties).
- **ADV-06** — Refit / continue training (`refit_decay_rate`, `input_model`) for `Booster.refit()`.
- **ADV-07** — Feature importance reporting (split/gain, `saved_feature_importance_type`).
- **bagging_by_query** — query-grouped row subsampling (deferred from Phase 6, BST-03) — ships here alongside the ranking objectives that exercise it.

Out of scope (explicitly deferred):
- **Phase 8**: Python/PyO3 bindings (this phase's Rust API surface is shaped so they map 1:1).
- Parallel (rayon) CPU / multi-GPU boosting paths — post-MVP optimization on the per-tree/per-row seam; must still match the deterministic anchor when added.
- Anything not enumerated in the 18 reqs above (distributed/network learners, linear-tree leaves, etc. are not in the v1.0 milestone scope).

</domain>

<decisions>
## Implementation Decisions

### Phase structure & sequencing (Decomposition / sequencing area)
- **D-01:** **One phase, many waves.** Phase 7 stays a single phase, planned as a long sequence of small, dependency-ordered waves/plans (the Phase-5 9-plan model), with one verification gate at the end. NOT split into sub-phases. Accepted trade-off: a long-running phase, in exchange for a simpler roadmap and a single coherent parity-completion gate.
- **D-02:** **Wave ordering is dependency-forced, low-risk-first** — the same spine-first ethos as every prior phase. Order roughly by what-unblocks-what + FP risk. Indicative ordering (researcher proposes the exact DAG in RESEARCH.md, reviewed at plan time):
  1. **Bagged-subset split-gain determinism** (early diagnostic/gating wave — see D-05) — gates everything that bags.
  2. **Objective + metric breadth** (OBJ-04/05, MET-03) — thin additions on the proven loop, lowest structural risk.
  3. **Boosting variants** (BST-04 GOSS / BST-05 DART / BST-06 RF) — build on bagging + the loop.
  4. **Categorical splits** (TRL-06) — the tree-learner re-open (see D-07).
  5. **Ranking stack** (OBJ-06 + MET-04 + `bagging_by_query` + DCGCalculator) — lands together (query infrastructure is shared).
  6. **Prediction modes** (PRD-04 SHAP / PRD-05 pred-early-stop).
  7. **Advanced features** (ADV-01..07).
- **D-03:** The 6 work-groups are: (1) boosting variants, (2) categorical, (3) objectives+metrics breadth, (4) ranking stack, (5) prediction modes, (6) advanced features — plus the early determinism wave. Each group is a sequence of one-axis-at-a-time validated additions, never a big-bang.

### Oracle validation matrix (Oracle matrix scale area)
- **D-04:** **Full cross-product (Phase-6 maximal-fidelity ethos) over per-subsystem RELEVANT axes.** Continue Phase 6's exhaustive committed-real-binary-golden discipline, but crossed only over the axes that actually change each subsystem's output — no provably-redundant cells:
  - **Objectives** (OBJ-04/05/06) → full `{bagging on/off} × {early_stop on/off} × {boost_from_average on/off}` per objective (the Phase-6 D-07 pattern), plus objective-specific params where they have parity-relevant effect (e.g. `huber` δ, `tweedie` variance power, `fair` c, `quantile` α, `poisson` max_delta_step).
  - **Boosting variants** (BST-04/05/06) → cross their own defining params (GOSS `top_rate`/`other_rate`; DART `drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`; RF mandatory-bagging) × the relevant loop axes.
  - **Categorical** (TRL-06) → cross `max_cat_to_onehot` (one-hot vs many-vs-many), `cat_smooth`/`cat_l2`, `min_data_per_group`, `max_cat_threshold`.
  - **Monotone** (ADV-01) → basic × intermediate × advanced × `monotone_penalty`.
  - **SHAP / predict-modes / importance / refit / extra-trees / CEGB / interaction / forced-splits** → cross their OWN relevant params (e.g. importance split-vs-gain × `saved_feature_importance_type`); do NOT cross against `{bagging×ES×bfa}` where those axes can't change the output.
  - Researcher enumerates the exact meaningful axis set per subsystem in RESEARCH.md; the planner may collapse a cell ONLY when provably byte-identical to another, documented (never silent truncation) — carried Phase-6 D-07 rule.

### Bagged-subset split-gain knife-edge (Deferred knife-edges area)
- **D-05:** **Make the bagged-subset split-gain determinism a dedicated EARLY diagnostic wave** (runs before GOSS/RF, which MUST bag, and before un-deferring L1+bagging). Root-cause the `cubecl-cpu` f64-fold vs C++ split-gain divergence on the bagged subset that flips leaf STRUCTURE (DEF-06-01: `binary_bag1_es0_bfa1` tree 0 → 2 vs 4 leaves; and the typed-rejected `regression_l1 + bagging`). Outcome branches:
  - **If a faithful fix exists** (an FP-order / formula faithfulness gap, not a true f32 artifact): apply it, **un-defer `regression_l1 + bagging`** (remove the `BoostingError::UnsupportedConfig` typed-reject from Phase 6 06-06 Task 2b), and **clear DEF-06-01**.
  - **If it is a genuine f32/accumulation-order artifact** with no faithful fix: document it as a bounded known-divergence with a hard structural-divergence cap (carry the Phase-6 `struct_divergent <= 1` posture), so a *growing* divergence still fails as a regression.
  - Either way, the determinism posture is **decided before** the bagging-dependent variants build on it — they inherit a settled answer, not an open question.

### Categorical tree-learner re-open (Categorical learner re-open area)
- **D-06:** **Categorical splits are an ADDITIVE branch in the Phase-5 serial learner, and the numeric spine stays byte-untouched + bit-exact.** Add `FindBestThresholdCategorical` (category-bitset threshold finding: `cat_smooth`/`cat_l2` regularized per-category gain, one-hot fallback below `max_cat_to_onehot`, many-vs-many bitset above it, `min_data_per_group`/`max_cat_threshold` gates) as a NEW branch alongside the numeric scan — exactly the additive-boundary-re-open discipline 05-01 used for `skip_default_bin`/`na_as_missing`. **HARD INVARIANT:** the existing numeric-spine real-`lib_lightgbm` 4.6 goldens (`spine_real.txt`, `mfb_pos_real.txt`, the growth-path/subtract gates) MUST still pass bit-exact after the re-open.
- **D-07:** **Categorical gets its own real `lib_lightgbm` 4.6 corpus** — a synthetic dataset with categorical features (reusing the Phase-2 categorical binning path, which is already bit-exact) — plus per-split layered diagnostics: per-category gain arrays, the chosen category bitset, the split decision_type, and model-text round-trip of the categorical threshold representation (the `||`-separated category set in the `.txt` schema).

### Claude's Discretion
- The exact wave DAG / plan boundaries within the 6 groups + early determinism wave (D-02 is indicative; researcher proposes, plan-checker verifies) — bounded by dependency-forced, low-risk-first.
- The precise per-subsystem axis enumeration for the full cross-product (D-04) — bounded by "every axis that can change the subsystem's output, no provably-redundant cells."
- Crate placement for the new variants/objectives/metrics (extend `lgbm-boosting`/`lgbm-objective`/`lgbm-metric` vs new modules) — bounded by the existing factory seams (`CreateBoosting`/`CreateObjectiveFunction`/`CreateMetric`).
- Whether GOSS sampling and DART drop-selection each get a dedicated RNG-replay golden à la Phase-6 D-13 (bagging) — strongly recommended by the carried "stochastic draws proven explicitly, not implicitly" discipline, but the exact golden shape is the researcher's call.
- The ranking-stack internal grouping (lambdarank vs rank_xendcg vs DCGCalculator vs ndcg/map vs `bagging_by_query`) — bounded by "shared query infrastructure lands together."
- The refit/continue-training (ADV-06) boundary (`input_model` continue vs `refit_decay_rate` leaf-refit) and which model-I/O hooks it reuses from Phase 3.
- Whether any Phase-7 subsystem warrants a ROCm cross-check vs CPU-bit-exact-only — CPU bit-exact is the hard gate; ROCm re-check is a research/planning call (carried Phase-6 deferral).
- When C++ behavior is the spec, the C++ source is authoritative over any inferred default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ reference source (read-only port target — authoritative for all Phase-7 behavior)

**Boosting variants (BST-04/05/06, bagging_by_query)**
- `LightGBM/src/boosting/goss.hpp` — GOSS sampling: gradient-magnitude sort, `top_rate`/`other_rate` selection, the `1-top_rate)/other_rate` amplification factor, the `Random` draw sequence (BST-04).
- `LightGBM/src/boosting/dart.hpp` — DART: drop selection (`drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`), `drop_seed` RNG, normalization of kept+dropped trees, `Boosting`/`Predict` overrides (BST-05).
- `LightGBM/src/boosting/rf.hpp` — Random Forest: averaged (not accumulated) trees, mandatory bagging, no shrinkage accumulation, `BoostFromScore` differences (BST-06).
- `LightGBM/src/boosting/sample_strategy.cpp`, `LightGBM/src/boosting/bagging.hpp` — the query-grouped bagging branch (`num_sampled_queries`/`sampled_query_indices`), `gbdt.cpp:227` query path, the `Random` draw/call order the `bagging_by_query` golden must match.
- `LightGBM/src/boosting/boosting.cpp` — `CreateBoosting` factory (gbdt/dart/rf/goss dispatch).

**Categorical splits (TRL-06)**
- `LightGBM/src/treelearner/feature_histogram.hpp` — `FindBestThresholdCategorical`: per-category gain with `cat_smooth`/`cat_l2`, one-hot vs many-vs-many (`max_cat_to_onehot`), `min_data_per_group`/`max_cat_threshold` gates, the category-sort-by-gradient order, bitset construction.
- `LightGBM/src/treelearner/serial_tree_learner.cpp` — where the categorical branch hooks into `FindBestSplitsFromHistograms`/`Split` (the additive re-open point, D-06).
- `LightGBM/src/io/tree.cpp`, `LightGBM/include/LightGBM/tree.h` — categorical split node representation (`decision_type` categorical bit, `cat_boundaries_`/`cat_threshold_` bitset), `SplitCategorical`, model-text serialization of the category set (D-07).

**Remaining objectives (OBJ-04/05/06)**
- `LightGBM/src/objective/regression_objective.hpp` — `huber`/`fair`/`poisson`/`quantile`/`mape`/`gamma`/`tweedie` `GetGradients`/`BoostFromScore`/`ConvertOutput`/`RenewTreeOutput` + their params (δ, c, α, variance power, max_delta_step) (OBJ-04).
- `LightGBM/src/objective/xentropy_objective.hpp` — `cross_entropy`/`cross_entropy_lambda` grad/hess, `BoostFromScore`, `ConvertOutput` (OBJ-05).
- `LightGBM/src/objective/rank_objective.hpp` — `lambdarank`/`rank_xendcg`: pairwise lambda computation, DCGCalculator usage, `objective_seed`, query-boundary iteration, sigmoid table (OBJ-06).
- `LightGBM/src/objective/objective_function.cpp` — `CreateObjectiveFunction` factory (all new objective string keys).

**Metrics (MET-03/04)**
- `LightGBM/src/metric/regression_metric.hpp` — `quantile`/`huber`/`fair`/`poisson`/`mape`/`gamma`/`gamma_deviance`/`tweedie` (MET-03).
- `LightGBM/src/metric/xentropy_metric.hpp` — `cross_entropy`/`cross_entropy_lambda`/`kullback_leibler` (MET-03).
- `LightGBM/src/metric/multiclass_metric.hpp` — `multi_error`/`auc_mu`/`average_precision` (MET-03).
- `LightGBM/src/metric/rank_metric.hpp` — `ndcg`/`map`, `eval_at`/`ndcg_eval_at`, per-query eval (MET-04).
- `LightGBM/src/metric/dcg_calculator.cpp`, `LightGBM/include/LightGBM/metric.h` — `DCGCalculator` static gain/discount tables (shared by OBJ-06 + MET-04).
- `LightGBM/src/metric/metric.cpp` — `CreateMetric` factory.

**Prediction modes (PRD-04/05)**
- `LightGBM/src/treelearner/*` SHAP / `LightGBM/src/io/tree.cpp` `PredictContrib`, `TreeSHAP` (and the `expected_value`/cover structure), `LightGBM/include/LightGBM/tree.h` — TreeSHAP over node/cover structure (PRD-04).
- `LightGBM/src/application/predictor.hpp` — `predict_contrib` driver, prediction early stopping (`pred_early_stop`/`_freq`/`_margin`) (PRD-05).
- `LightGBM/src/boosting/gbdt_prediction.cpp` — predict-side hooks for contrib + early stop.

**Advanced features (ADV-01..07)**
- `LightGBM/src/treelearner/monotone_constraints.hpp`, `serial_tree_learner.cpp` — monotone basic/intermediate/advanced + `monotone_penalty` (ADV-01); interaction constraints (`interaction_constraints`, ADV-02); forced splits/bins (JSON, ADV-03); extra-trees randomized thresholds (`extra_trees`/`extra_seed`, ADV-04); CEGB penalties (`cegb_*`, ADV-05).
- `LightGBM/src/treelearner/cost_effective_gradient_boosting.hpp` — CEGB (ADV-05).
- `LightGBM/src/boosting/gbdt.cpp` `RefitTree`/continue-training (`input_model`, `refit_decay_rate`), `LightGBM/src/c_api.cpp` `LGBM_BoosterRefit` — refit (ADV-06).
- `LightGBM/src/io/tree.cpp` / `gbdt.cpp` feature-importance (`FeatureImportance`, split vs gain, `saved_feature_importance_type`) (ADV-07).
- `LightGBM/src/io/config.cpp` / `config_auto.cpp` + `LightGBM/include/LightGBM/config.h` — all new param names/aliases/defaults/CHECK constraints the `lgbm-core::Config` mirror must add.
- `LightGBM/include/LightGBM/meta.h` — `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`, `score_t`/`label_t = float` (load-bearing).

### Foundations to build on (Phase 1–6 deliverables)
- `crates/lgbm-treelearner/` — the bit-exact `SerialTreeLearner` (Phase 5, real-binary parity on both corpora). The additive categorical branch (D-06) re-opens THIS crate; numeric spine goldens MUST stay bit-exact.
- `crates/lgbm-boosting/` — the GBDT loop, `ScoreUpdater`, `BaggingSampleStrategy` (Phase 6). GOSS/DART/RF extend the boosting layer + its `CreateBoosting`-analog factory; `bagging_by_query` extends the existing bagging strategy.
- `crates/lgbm-objective/` — Phase-6 core objectives (`regression`/`regression_l1`/`binary`/`multiclass`/`multiclassova`/`custom`) + the `GetGradients`/`ConvertOutput`/`BoostFromScore` interface OBJ-04/05/06 extend.
- `crates/lgbm-metric/` — Phase-6 core metrics + the `Metric::Eval`/multi-metric infra MET-03/04 extend (MET-04 adds `DCGCalculator` + per-query eval).
- `crates/lgbm-model/` — `GbdtModel`/`Tree` + the `%.17g` formatter; extend for categorical-split model-text (D-07), SHAP node/cover structure (PRD-04), feature importance (ADV-07), and refit/continue-training model I/O (ADV-06).
- `crates/lgbm-core/` — `Config` (the single source of truth; add all Phase-7 params via the verbatim alias table + CHECK validation), `Random` LCG (GOSS/DART/`bagging_by_query` RNG parity source).
- `crates/lgbm-dataset/` — the immutable binned store + categorical binning (already bit-exact, Phase 2) + query/group boundaries (consumed by ranking + `bagging_by_query`); metadata.
- `crates/lgbm` — the public builder API; add a setter per new param + variant/objective/metric selection (mapping 1:1 to the Phase-8 bindings).
- `crates/oracle-harness/` + `xtask` capture pipeline — extend `REFERENCE_MANIFEST.md` + add real-`lib_lightgbm`-4.6 capture subcommands for every Phase-7 subsystem's full-cross-product cells + layered diagnostics + RNG-replay goldens.

### Project-level contract & prior context
- `.planning/PROJECT.md` — Core Value (f32, ~1e-6, both backends), Constraints, Key Decisions; PROJECT.md also records DEF-06-01 + the regression_l1+bagging typed-reject that D-05 revisits.
- `.planning/REQUIREMENTS.md` — BST-04/05/06, TRL-06, OBJ-04/05/06, MET-03/04, PRD-04/05, ADV-01..07 (all Phase 7); the DEF-06-01 note.
- `.planning/ROADMAP.md` §"Phase 7" — goal + 5 success criteria (the SC→requirement mapping).
- `.planning/phases/06-gbdt-spine-core-objectives-metrics/06-CONTEXT.md` — the locked carried-forward discipline (faithful mirror, real-binary oracle, layered diagnostics, builder API D-01..D-05, bagging RNG D-13, spine-first sequencing) Phase 7 inherits wholesale.
- `.planning/phases/06-gbdt-spine-core-objectives-metrics/deferred-items.md` — **DEF-06-01** full root-cause (the bagged-subset split-gain knife-edge D-05 must resolve).
- `.planning/phases/06-gbdt-spine-core-objectives-metrics/06-06-SUMMARY.md` — the regression_l1+bagging typed-reject decision (Task 2b) D-05 revisits.
- `.planning/phases/05-tree-learner-split-finding/05-CONTEXT.md` — the tree-learner re-open discipline (05-01 additive boundary) D-06 follows; the real-binary corpus pattern D-07 follows.
- `.planning/phases/03-tree-model-model-text-i-o-predict-parity/03-CONTEXT.md` — `%.17g` formatter + model I/O reused for categorical model-text (D-07), SHAP, importance, refit.
- `.planning/phases/02-dataset-binning-determinism-root/02-CONTEXT.md` — the bit-exact categorical binning path the TRL-06 corpus (D-07) reuses.

### Codebase maps (reference C++ architecture & porting concerns)
- `.planning/codebase/CONCERNS.md` — FP reduction ordering, `kEpsilon`/`kZeroThreshold`, RNG call-order hazards (directly relevant to D-05's knife-edge + GOSS/DART RNG).
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` — the Boosting/Objective/Metric/TreeLearner factory seams the new variants plug into.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/lgbm-treelearner::SerialTreeLearner` — bit-exact numeric learner; categorical is an ADDITIVE branch on it (D-06), numeric goldens stay bit-exact.
- `crates/lgbm-boosting` GBDT loop + `ScoreUpdater` + `BaggingSampleStrategy` — GOSS/DART/RF and `bagging_by_query` extend the boosting layer + bagging strategy; the loop/score-updater stay as-is.
- `crates/lgbm-objective` / `crates/lgbm-metric` — Phase-6 objective/metric trait + factory seams that OBJ-04/05/06 + MET-03/04 extend (ranking adds `DCGCalculator` + per-query eval).
- `crates/lgbm-model` `GbdtModel`/`Tree` + `%.17g` formatter — extended for categorical bitset model-text, SHAP node/cover, importance, refit model I/O.
- `crates/lgbm-core::Config` + alias table + `Random` LCG — params source of truth (add all Phase-7 params) + the RNG parity source for GOSS/DART/`bagging_by_query`.
- `crates/lgbm-dataset` — bit-exact categorical binning (Phase 2) reused for the TRL-06 corpus; query/group boundaries for ranking + `bagging_by_query`.
- `crates/oracle-harness` + `xtask` capture pipeline — the committed-real-binary-golden + idempotent-regen harness extended per subsystem (full cross-product + layered diagnostics + RNG-replay goldens).

### Established Patterns
- Faithful 1:1 C++ hand-port guarded by a parity test, below the API boundary (P1–P6) — applies to every Phase-7 variant/objective/metric/constraint math + every RNG draw/call order.
- Real `lib_lightgbm` 4.6 oracle, committed fixtures, idempotent C++-regen; no C++ toolchain at normal test time; **`LightGBM/` untracked — never `git add`** (memory: lightgbm-ref-tree-untracked).
- Layered, maximally-diagnostic goldens (P2–P6 ethos) — per-row grad/hess + per-iteration scores + per-round metrics + per-round stochastic-draw indices + end-to-end model/predict, so a failure localizes to the exact subsystem.
- Additive tree-learner boundary re-open (05-01) — new branch alongside the existing path, existing goldens stay bit-exact (D-06).
- Stochastic draws proven EXPLICITLY, not implicitly (Phase-6 D-13 bagging) — GOSS sampling + DART drop selection + `bagging_by_query` each warrant a dedicated RNG-replay golden.
- CPU `cubecl-cpu` f64-fold is the bit-exact hard merge gate; ROCm is a separate ~1e-6 best-effort gate.
- C++ constants in play: `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`; `score_t`/`label_t = float`.

### Integration Points
- Categorical (D-06) re-opens `lgbm-treelearner`'s split path — the only Phase-7 item that modifies the bit-exact learner; everything else is additive above it.
- GOSS/DART/RF plug into the boosting-factory seam; DART also overrides predict-side normalization (model-I/O touch).
- Ranking couples objectives + metrics + `DCGCalculator` + query boundaries + `bagging_by_query` (the one query-grouped bagging path) — lands as a coherent group.
- SHAP/importance/refit touch `lgbm-model` (node/cover structure, importance counts, model I/O).
- Everything stays CubeCL-free above the `lgbm-compute` seam (CMP-01).
- The public builder (`crates/lgbm`) grows one setter per new param/variant — shaped to map 1:1 onto the Phase-8 Python bindings.

</code_context>

<specifics>
## Specific Ideas

- **One coherent parity-completion gate (D-01):** the user wants Phase 7 as a single phase with many small dependency-ordered waves and one end verification — not fragmented into sub-phases — so "full single-machine parity" is a single provable milestone.
- **Maximal fidelity continued (D-04):** full cross-product over each subsystem's *relevant* axes — the user explicitly chose the exhaustive Phase-6 ethos again, refined only to exclude provably-meaningless cells. No validation blind spots.
- **Settle determinism before building on it (D-05):** the bagged-subset split-gain knife-edge (DEF-06-01 + L1+bagging) is an early gating wave because GOSS/RF *must* bag — the variants inherit a decided answer (faithful fix → un-defer L1+bagging + clear DEF-06-01, or bounded documented divergence with a hard cap), never an open question.
- **Numeric spine stays sacred (D-06):** categorical is purely additive; the Phase-5 real-binary numeric goldens must still pass bit-exact after the re-open — the highest-FP-risk subsystem is protected by an explicit invariant.

</specifics>

<deferred>
## Deferred Ideas

- **Python/PyO3 bindings** — Phase 8 (this phase's builder API + variant/objective/metric surface is shaped to map 1:1).
- **Parallel (rayon) CPU / multi-GPU boosting path** — post-MVP optimization on the per-tree/per-row independence seam; must still match the deterministic anchor when added.
- **ROCm cross-check of Phase-7 subsystems** — CPU bit-exact is the hard gate; whether any Phase-7 subsystem is re-run on ROCm vs deferred to a later parity sweep is a research/planning call (carried Phase-6 deferral).
- **Out-of-milestone subsystems** (distributed/network learners, linear-tree leaves, GPU-specific tree learner) — not in the v1.0 requirement set; not Phase 7.

None other — discussion stayed within Phase 7 scope.

</deferred>

---

*Phase: 7-parity-completing-variants*
*Context gathered: 2026-06-07*
