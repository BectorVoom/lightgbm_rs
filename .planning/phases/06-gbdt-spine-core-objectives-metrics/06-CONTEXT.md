# Phase 6: GBDT Spine + Core Objectives/Metrics - Context

**Gathered:** 2026-06-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 6 delivers the **first end-to-end ~1e-6 (f32) train→predict run** — the GBDT boosting spine wrapped around the Phase-5 bit-exact `SerialTreeLearner`, plus the core objectives that produce gradients/hessians at runtime, the core metrics that evaluate scores, row-subsampling (bagging), early stopping, and the **Rust-native `Dataset`/`Booster`/`train`/`predict` API** that ties it together. The simplest boosting variant (plain GBDT) proves the full spine before any variant is added in Phase 7.

In scope (BST-01, BST-02, BST-03, BST-07, OBJ-01, OBJ-02, OBJ-03, MET-01, MET-02, API-01):
- **BST-01** — GBDT training loop: `TrainOneIter`, `UpdateScore`, per-class trees, shrinkage (`learning_rate`), `boost_from_average`.
- **BST-02** — score updater accumulation with deterministic reduction ordering.
- **BST-03** — bagging / row subsampling (`bagging_fraction`/`bagging_freq`/`bagging_seed`, pos/neg, `bagging_by_query`) selecting the **same rows** via RNG-matching draw sequence + call order.
- **BST-07** — early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`).
- **OBJ-01** — core objectives: `regression` (l2), `regression_l1`, `binary`, `multiclass` (softmax), `multiclassova`.
- **OBJ-02** — `custom` objective (user-supplied grad/hess pass-through).
- **OBJ-03** — objective machinery: `GetGradients`, `ConvertOutput` (sigmoid/softmax/exp), `BoostFromScore`, `reg_sqrt`.
- **MET-01** — core metrics: `l1`, `l2`, `rmse`, `binary_logloss`, `binary_error`, `auc`, `multi_logloss`.
- **MET-02** — metric infrastructure: multi-metric lists, `metric_freq`, `is_provide_training_metric`, training-metric eval.
- **API-01** — Rust-native API: `Dataset`, `Booster`, `train`, `predict` mirroring LightGBM semantics.

Out of scope (explicitly deferred):
- **Phase 7 variants**: GOSS/DART/RF; categorical/EFB splits (TRL-06); remaining objectives (huber/fair/poisson/quantile/mape/gamma/tweedie/cross-entropy/ranking); extended/ranking metrics (ndcg/map/average_precision/auc_mu); SHAP/`predict_contrib`; prediction early stopping; monotone/interaction constraints; forced splits/bins; extra-trees; CEGB; refit/continue-training; feature importance.
- **Phase 8**: Python/PyO3 bindings (this phase's Rust API is shaped so the bindings map 1:1).
- The bit-exact serial tree learner itself (shipped Phase 5); prediction over a loaded model (shipped Phase 3).
- Parallel (rayon) CPU / multi-GPU boosting paths — post-MVP optimization on the per-tree/per-row seam; must still match the deterministic anchor when added.

</domain>

<decisions>
## Implementation Decisions

### Rust-native API shape (API-01, OBJ-02)
- **D-01:** **Builder-pattern public API.** The user-facing surface is an ergonomic Rust builder (e.g. `Booster::builder()…build()` / a training builder), NOT a verbatim `lgb.train(params, …)` free function. This is the project's first net-new user-facing surface — idiomatic Rust ergonomics are wanted **on the outside**, while every FP-load-bearing internal stays a faithful C++ mirror (the locked discipline applies below the API boundary, not to the public ergonomics).
- **D-02:** **The training-params builder resolves to `lgbm-core::Config` internally — Config remains the single source of truth.** The ergonomic builder is a thin front-end that produces a `Config`; the engine and the oracle harness both consume `Config`. No forked defaults/aliases/param semantics — the verbatim C++ alias table + CHECK validation (Phase 1) is not duplicated or bypassed.
- **D-03:** **Full param surface.** The builder exposes a method per in-scope parameter (objective, num_iterations, learning_rate, num_leaves, bagging_*, early_stopping_*, metric, seeds, determinism flags, boost_from_average, …) PLUS a `from_config(Config)` / raw-set escape hatch so the oracle can drive **any** parity-relevant parameter. (Chosen over a curated subset — consistent with the project's maximal-fidelity ethos.)
- **D-04:** **`custom` objective is a closure mirroring the Python `fobj` contract.** Signature shaped as `Fn(preds: &[f32], dataset: &Dataset) -> (grad: Vec<f32>, hess: Vec<f32>)`, matching Python's `fobj(preds, train_data) -> (grad, hess)` so the Phase-8 bindings map 1:1. (Chosen over a `CustomObjective` trait object — the closure matches the binding contract; the exact borrow/return ownership shape is Claude's discretion bounded by that contract.)
- **D-05:** **Eval history + early-stopping outcome surfaced as Booster fields, mirroring Python.** The `Booster` exposes `best_iteration` + per-valid-set / per-metric eval history (mirroring Python's `best_iteration_`/`best_score_` and the `record_evaluation` shape). (Chosen over a separately-returned `EvalResults` struct.)

### Oracle corpus matrix (carries Phase-5 D-08 — real `lib_lightgbm` 4.6 oracle)
- **D-06:** **All 5 core objectives + one custom get committed end-to-end (train→model-text→predict) real-binary goldens.** `regression`(L2), `regression_l1`, `binary`, `multiclass`(softmax), `multiclassova` each get an end-to-end golden; `custom` gets one run validated against a Python `fobj` reference (the closure forwards user grad/hess — its parity is against the Python contract, since there is no distinct C++ objective to diff). Full OBJ-01/02/03 surface.
- **D-07:** **Full cross-product of the config axes per objective.** Each objective is crossed over `{bagging on/off} × {early_stopping on/off} × {boost_from_average on/off}` → ~40 committed end-to-end real-binary cells (5 objectives × 2 × 2 × 2), plus the custom run. **Deliberately exhaustive** (user choice, maximal fidelity). _Planning note:_ early-stopping cells require a valid set + enough iterations to fire; bagging cells require `bagging_freq>0`. The researcher MAY collapse a cell to another ONLY if it is provably byte-identical (e.g. `bagging off + early-stop off + bfa on` already IS the per-objective spine golden) and documents the equivalence — never silent truncation.
- **D-08:** **Small per-objective synthetic corpora**, committed + idempotently regenerable, with objective-appropriate labels (continuous for regression, 0/1 for binary, K-class for multiclass), reusing the Phase-2 binning path. (Chosen over reusing Phase-2/5 corpora — labels must fit each objective cleanly.)
- **D-09:** **Modest multi-iteration depth (~10–20 iters), small trees (low `num_leaves`).** Enough to genuinely exercise score accumulation, shrinkage, `boost_from_average`, and a real early-stopping trigger, while keeping goldens reviewable.

### Validation granularity (max-diagnostic — analog of Phase-5 D-06)
- **D-10:** **Per-row grad/hess golden snapshots at iteration 1 AND a later iteration** (scores no longer zero) for each objective — captured from the C++ objective's `GetGradients`. Localizes any objective-math divergence to a specific row before it propagates into trees.
- **D-11:** **Per-iteration accumulated raw-score snapshot.** After each boosting iteration the score updater's accumulated raw scores are snapshotted vs the reference, so a loop/shrinkage/`boost_from_average` divergence localizes to the exact iteration.
- **D-12:** **Per-eval-round metric-value snapshots.** Each metric's value at each eval round is committed vs the reference (l1/l2/rmse/binary_logloss/binary_error/auc/multi_logloss) — this also directly validates the early-stopping decision input.
- **D-13:** **Bagging RNG parity is a dedicated golden.** The exact bagged row-index set chosen at each bagging round is snapshotted and asserted to bit-match the C++ `Random` draw sequence + call order (ROADMAP SC#5 / BST-03: "same rows via RNG-matching sequence"). NOT left implicit in end-to-end parity — a wrong-but-similar bag must not be able to mask an RNG mismatch.

### Spine-first sequencing (analog of Phase-5 D-04)
- **D-14:** **Minimal end-to-end spine = `regression`(L2) + `l2`/`rmse`.** Simplest single-output objective (constant hess=1, identity `ConvertOutput`) + its natural metrics — proves the full train→score→predict→metric loop with the least objective complexity.
- **D-15:** **The minimal spine INCLUDES `boost_from_average`.** `boost_from_average=true` is the C++ regression default and a load-bearing `BoostFromScore`/init-score FP path; including it makes the spine match the real binary's **default** run with no special-casing. (Chosen over deferring bfa — the spine should match the C++ default config.)
- **D-16:** **Multiclass per-class trees enter AFTER the single-output spine is proven.** `multiclass`/`multiclassova` grow `num_class` trees per iteration and need the per-class score layout — added as a later structural addition once regression→binary single-output is locked end-to-end.
- **D-17:** **Addition order after the single-output spine: objectives → multiclass → bagging → early-stop.** Widen objective/metric breadth on the proven loop first, then per-class trees, then row-subsampling RNG, then early stopping — each a thin, one-axis-at-a-time validated addition.

### Carried Forward (locked by prior phases — not re-litigated)
- **Faithful 1:1 C++ mirror** discipline (P1 D-11/D-12, P2 D-01, P3 D-04, P4 D-04, P5 D-08): reproduce the GBDT loop order, the score-updater reduction order, the `BoostFromScore` init, the objective grad/hess formulas, the metric reductions, and the bagging RNG draw/call order verbatim — never an idiomatic redesign **below the API boundary**. (D-01's idiomatic builder applies to the public ergonomics only.)
- **f32 end-to-end, ~1e-6 absolute, standard f32 accumulations** (`score_t`/`label_t` = `float`); integer-quantized strategy dropped (P1 D-02/D-03).
- **CPU is the bit-exact hard merge gate; ROCm is a separate ~1e-6 gate** (P4 D-03/D-04, P5): the cubecl-cpu f64-fold path is the deterministic anchor.
- **Real `lib_lightgbm` 4.6 oracle**, built deterministically (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, fixed seed); committed goldens + idempotent C++-regen; **`LightGBM/` is untracked — never `git add` it** (memory: lightgbm-ref-tree-untracked). (P5 D-08.)
- **`lgbm-compute` is the single CubeCL seam** (P1 D-09, CMP-01): the boosting layer depends on the learner/`Backend`, never on a `cubecl` runtime type.
- **Single-threaded deterministic core** matching the pinned reference (P2 D-03); per-tree/per-row independence is the parallel/GPU seam, not exercised this phase.
- **The Phase-5 `SerialTreeLearner` is the per-tree engine** — bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora; Phase 6 drives it via `train(grad, hess, is_first_tree) → Tree`, it does not modify the learner.

### Claude's Discretion
- Crate placement/structure for the boosting layer (new `lgbm-boosting` crate + an umbrella/facade `lgbm` crate for the public API, vs folding into existing crates) and the boosting↔learner wiring.
- Where objectives + metrics live (new `lgbm-objective` / `lgbm-metric` crates vs modules) and their internal trait shape — bounded by the C++ `ObjectiveFunction`/`Metric` factory semantics.
- The exact ownership/borrow shape of the `custom` objective closure (D-04), bounded by the Python `fobj` contract.
- The golden serialization/layering format for grad/hess, per-iteration scores, per-round metrics, and bagged-index fixtures — bounded by the oracle-harness comparator + Phase-3 `%.17g` machinery.
- AUC tie-handling / sort determinism, `metric_freq`/`first_metric_only` evaluation cadence specifics, and per-class score memory layout — bounded by "must match the C++ reference."
- The captured-g/h capture path config + which iteration counts as "a later iteration" for D-10, bounded by the layered-golden contract.
- When C++ behavior is the spec, the C++ source below is authoritative over any inferred default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ reference source (read-only port target — authoritative for all Phase-6 behavior)
- `LightGBM/src/boosting/gbdt.cpp` — **the primary port target.** `TrainOneIter`, `Boosting`/`UpdateScore`, `Bagging`, `BoostFromScore`/`boost_from_average`, per-class tree loop, shrinkage, early-stopping hookup, the deterministic reduction ordering (BST-01, BST-02, BST-07).
- `LightGBM/src/boosting/gbdt.h`, `LightGBM/include/LightGBM/boosting.h` — `GBDT`/`Boosting` member state + `TrainOneIter`/`Predict`/model-I/O interface the Rust boosting layer mirrors.
- `LightGBM/src/boosting/score_updater.hpp` — `ScoreUpdater::AddScore`/`MultiplyScore` accumulation + reduction order (BST-02).
- `LightGBM/src/boosting/sample_strategy.cpp`, `LightGBM/src/boosting/bagging.hpp` — `bagging_fraction`/`bagging_freq`/`bagging_seed`, pos/neg bagging, `bagging_by_query`, the `Random` draw sequence + call order the index golden must match (BST-03, D-13).
- `LightGBM/src/objective/regression_objective.hpp` — `regression`(L2) + `regression_l1` `GetGradients`/`BoostFromScore`/`ConvertOutput`, `reg_sqrt`, `RenewTreeOutput` (OBJ-01/03; the D-14 spine).
- `LightGBM/src/objective/binary_objective.hpp` — `binary` sigmoid grad/hess, `BoostFromScore` (base rate), `sigmoid_` param, `ConvertOutput` (OBJ-01/03).
- `LightGBM/src/objective/multiclass_objective.hpp` — `multiclass`(softmax) + `multiclassova` per-class grad/hess, `num_class_`, `ConvertOutput`, per-class score layout (OBJ-01/03, D-16).
- `LightGBM/src/objective/objective_function.cpp` + `LightGBM/include/LightGBM/objective_function.h` — `CreateObjectiveFunction` factory, the `custom` pass-through path (OBJ-02), `GetGradients`/`ConvertOutput`/`BoostFromScore` interface.
- `LightGBM/include/LightGBM/utils/common.h` — `Softmax` (with `wmax` max-subtraction), `Sign`, sigmoid helpers (load-bearing in objectives/ConvertOutput).
- `LightGBM/src/metric/regression_metric.hpp` (`l1`/`l2`/`rmse`), `binary_metric.hpp` (`binary_logloss`/`binary_error`), `multiclass_metric.hpp` (`multi_logloss`), `auc_metric` source / `metric.cpp` — metric reductions + `CreateMetric` factory (MET-01).
- `LightGBM/include/LightGBM/metric.h` — `Metric::Eval` interface, multi-metric handling, `DCGCalculator` (out of scope here but in the file).
- `LightGBM/src/io/config.cpp` / `config_auto.cpp` + `LightGBM/include/LightGBM/config.h` — `metric_freq`, `is_provide_training_metric`, `early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`, `boost_from_average`, `bagging_*` semantics/defaults (MET-02, BST-07, the source of truth `lgbm-core::Config` mirrors).
- `LightGBM/include/LightGBM/meta.h` — `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`, `score_t`/`label_t = float` (load-bearing).

### Foundations to build on (Phase 1–5 deliverables)
- `crates/lgbm-treelearner/` — `SerialTreeLearner::train(grad: &[f32], hess: &[f32], is_first_tree) → Tree`, bit-exact vs real `lib_lightgbm` 4.6 (Phase 5). The per-tree engine the GBDT loop drives.
- `crates/lgbm-model/` — `GbdtModel` (ensemble), `Tree`, the model-text `%.17g`/`{:g}` formatter (Phase 3) for the end-to-end model-text golden; `objective::ObjectiveKind` (predict-side `ConvertOutput` only — Phase 6 adds the **training** grad/hess side).
- `crates/lgbm-core/` — `Config` (the ~110-param bag + verbatim C++ alias table + CHECK validation; D-02/D-03 resolve to this), `src/types.rs` (f32 types), `src/error.rs` (`thiserror` boundary idiom), `Random` LCG (the bagging-RNG parity source for BST-03/D-13).
- `crates/lgbm-dataset/` — the immutable binned columnar store + metadata (labels, weights, query boundaries) consumed by objectives/metrics/bagging; do NOT re-bin.
- `crates/lgbm-compute/` — the `Backend` trait the learner sits on (the single CubeCL seam, CMP-01); the boosting layer stays above it.
- `crates/oracle-harness/` — `compare_exact_*` (bit-exact CPU anchor) + f32 ~1e-6 comparator + committed-golden/idempotent-regen seam; extend `REFERENCE_MANIFEST.md` with the Phase-6 grad/hess, per-iteration-score, per-round-metric, bagged-index, and end-to-end model/predict fixtures.
- `xtask` `bin-capture`/`model-capture`/kernel-capture/learner-capture pattern + `xtask/cpp/` — extend with a boosting/objective/metric capture subcommand (real `lib_lightgbm` 4.6 per D-06/D-07; Python `fobj` reference for the custom run).

### Project-level contract & prior context
- `.planning/PROJECT.md` — Core Value (f32, ~1e-6, both backends), Constraints, Key Decisions (standard f32 accumulations; faithful mirror; Rust-native API + Python bindings, no C-ABI/CLI in v1; CubeCL `Plane` mandate).
- `.planning/REQUIREMENTS.md` — BST-01/02/03/07, OBJ-01/02/03, MET-01/02, API-01 (Phase 6); deferred BST-04/05/06, OBJ-04/05/06, MET-03/04 (Phase 7).
- `.planning/ROADMAP.md` §"Phase 6" — goal + 5 success criteria.
- `.planning/STATE.md` — Phase 5 COMPLETE (learner bit-exact); blockers (CubeCL alpha pin; ROCm gaps — relevant only if ROCm parity is re-checked here).
- `.planning/phases/05-tree-learner-split-finding/05-CONTEXT.md` — D-08 (real `lib_lightgbm` 4.6 oracle), the faithful-mirror + committed-golden + layered-diagnostic discipline this phase inherits; the `TreeLearner::Train(grad, hess, is_first_tree) → Tree` seam.
- `.planning/phases/03-tree-model-model-text-i-o-predict-parity/03-CONTEXT.md` — the `%.17g` formatter + tree/model serialization reused for the end-to-end model-text golden; predict-side `ConvertOutput`.
- `.planning/phases/01-oracle-contract-foundations/01-CONTEXT.md` — `Random` RNG parity (bagging), the f32 strategy, the Config alias table + seed derivation.

### Codebase maps (reference C++ architecture & porting concerns)
- `.planning/codebase/CONCERNS.md` — FP reduction ordering, `kEpsilon`/`kZeroThreshold`, score accumulation, RNG call-order hazards in the boosting/objective/metric hotpaths.
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` — the Boosting layer (`GBDT::TrainOneIter` → ObjectiveFunction → TreeLearner → Metric), the `CreateObjectiveFunction`/`CreateMetric`/`CreateBoosting` factory seams, score-updater/sample-strategy layout.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/lgbm-treelearner::SerialTreeLearner::train(grad, hess, is_first_tree) → Tree` — the bit-exact per-tree engine; the GBDT loop feeds it runtime grad/hess and collects `Tree`s into `GbdtModel`.
- `crates/lgbm-model` `GbdtModel` + `Tree` + `%.17g` formatter — the ensemble container + the end-to-end model-text golden comparator; `ObjectiveKind` already implements the predict-side `ConvertOutput` (Phase 6 adds the symmetric training grad/hess side).
- `crates/lgbm-core::Config` + alias table + `Random` LCG — the params source of truth the builder resolves to (D-02), and the deterministic RNG the bagging parity (D-13) is built on.
- `crates/lgbm-dataset` binned store + metadata (labels/weights) — objective/metric/bagging input.
- `crates/oracle-harness` + `xtask` capture pipeline — committed-golden + real-`lib_lightgbm`-4.6 regen harness to extend with boosting/objective/metric/bagging capture subcommands.

### Established Patterns
- Faithful 1:1 C++ hand-port guarded by a parity test (P1–P5) — applies to the GBDT loop, score updater, objective grad/hess, metric reductions, `BoostFromScore`, and bagging RNG draw/call order (below the API boundary; the public builder is idiomatic Rust per D-01).
- Real `lib_lightgbm` 4.6 oracle, committed fixtures, idempotent C++-regen; **no C++ toolchain at normal test time**; `LightGBM/` untracked (never `git add`).
- Layered, maximally-diagnostic goldens (P2–P5 D-06 ethos, here D-10..D-13): per-row grad/hess + per-iteration scores + per-round metrics + per-round bagged indices + end-to-end model/predict so a failure localizes to objective vs loop vs metric vs bagging vs final model.
- Bit-exact comparison for the deterministic CPU anchor; ~1e-6 for any ROCm cross-check.
- C++ constants in play: `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`; `score_t`/`label_t = float`.

### Integration Points
- The Phase-6 boosting layer is the dependency consumer of the Phase-5 learner (`TreeLearner::Train(grad, hess, is_first_tree) → Tree`) and the Phase-3 model/predict; it produces the public `Booster` consumed by the Phase-8 Python bindings (which is why D-04/D-05 mirror the Python `fobj`/Booster surface).
- The `custom` objective couples a runtime user closure into the loop (D-04) — distinct from the fixture-only captured-g/h of Phase 5; the Phase-6 objectives produce g/h at runtime.
- Must remain CubeCL-free above the `lgbm-compute` seam (CMP-01).

</code_context>

<specifics>
## Specific Ideas

- **Idiomatic on the outside, faithful on the inside (deliberate split):** the public API is an ergonomic Rust builder (D-01) and the eval/custom-objective surfaces mirror Python so Phase-8 bindings map 1:1 (D-04/D-05) — but every FP-load-bearing internal (GBDT loop, score updater, objective/metric math, bagging RNG) stays a verbatim C++ mirror. The single source of truth `Config` is never forked: the builder resolves down to it (D-02).
- **Maximal fidelity continued from Phase 5:** full cross-product oracle matrix (~40 cells, D-07), full per-row grad/hess + per-iteration score + per-round metric + per-round bagged-index goldens (D-10..D-13). The user wants no validation blind spots in the first end-to-end run.
- **Spine-first vertical slice:** prove `regression`(L2)+`l2`/`rmse` end-to-end **with** `boost_from_average` (the C++ default, D-14/D-15) before widening to binary → multiclass → bagging → early-stop (D-16/D-17), one axis at a time.
- **Bagging RNG is proven explicitly, not implicitly** (D-13): the bagged-index sequence is its own golden because ROADMAP SC#5 demands "same rows via RNG-matching sequence" and a wrong-but-similar bag could otherwise hide an RNG mismatch behind a near-matching model.

</specifics>

<deferred>
## Deferred Ideas

- **GOSS / DART / Random Forest** boosting variants — Phase 7 (BST-04/05/06). This phase builds the plain-GBDT spine only.
- **Categorical / EFB splits** (TRL-06) — Phase 7.
- **Remaining objectives** (huber/fair/poisson/quantile/mape/gamma/tweedie, cross-entropy, ranking/lambdarank/rank_xendcg) — Phase 7 (OBJ-04/05/06).
- **Extended + ranking metrics** (ndcg/map/average_precision/auc_mu/...) and per-query metric eval — Phase 7 (MET-03/04).
- **SHAP/`predict_contrib`, prediction early stopping, monotone/interaction constraints, forced splits/bins, extra-trees, CEGB, refit/continue-training, feature importance** — Phase 7.
- **Python/PyO3 bindings** — Phase 8 (this phase's API is shaped to map 1:1).
- **Parallel (rayon) CPU / multi-GPU boosting path** — post-MVP optimization on the per-tree/per-row independence seam; must still match the deterministic anchor when added.
- **ROCm cross-check of the full train→predict loop** — kernels are ROCm-gated (Phase 4); whether the orchestrated boosting loop is re-run on ROCm here vs deferred to a later parity sweep is a research/planning call (CPU bit-exact is the hard gate).

None other — discussion stayed within Phase 6 scope.

</deferred>

---

*Phase: 6-gbdt-spine-core-objectives-metrics*
*Context gathered: 2026-06-07*
