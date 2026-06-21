# Phase 7: Parity-Completing Variants - Pattern Map

**Mapped:** 2026-06-07
**Files analyzed:** ~40 new/modified symbols across 8 workspace crates + oracle-harness/xtask
**Analogs found:** 18 / 18 subsystems (every Phase-7 item has a Phase-1→6 Rust analog already in-tree)

> **Scope note:** Every excerpt below is from the **existing Rust crates** (the established Phase-6 shape the planner must mirror), NOT the C++ reference. The C++ file:line citations for the new math live in `07-RESEARCH.md` §"Canonical References" / §"Code Examples"; this document maps each new symbol to the *Rust analog* it should be modeled on.

---

## File Classification

| New/Modified File · Symbol | Role | Data Flow | Closest Rust Analog | Match Quality |
|----------------------------|------|-----------|---------------------|---------------|
| `lgbm-objective/src/regression.rs` (+huber/fair/poisson/quantile/mape/gamma/tweedie variants on `enum Objective`) | algorithm/math | transform | `lgbm-objective::regression::Objective` (L2/L1) | exact |
| `lgbm-objective/src/xentropy.rs` (`cross_entropy`/`cross_entropy_lambda`) | algorithm/math | transform | `lgbm-objective::binary::Binary` (sigmoid grad/hess) | exact |
| `lgbm-objective/src/rank.rs` (`lambdarank`/`rank_xendcg`) | algorithm/math | transform (per-query) | `lgbm-objective::multiclass::MulticlassSoftmax` (strided per-group dispatch) + `Binary` (sigmoid table) | role-match |
| `lgbm-objective/src/lib.rs` (re-exports) | factory wiring | — | `lgbm-objective::lib` re-export block | exact |
| `lgbm-boosting/src/objective.rs` (`BoostObjective` new variants) | factory wiring | request-response | `lgbm-boosting::objective::BoostObjective` enum | exact |
| `lgbm-metric/src/regression.rs` (+quantile/huber/fair/poisson/mape/gamma/gamma_deviance/tweedie on `enum Metric`) | algorithm/math | transform | `lgbm-metric::regression::Metric` (L2/RMSE/L1) | exact |
| `lgbm-metric/src/xentropy.rs` (`cross_entropy`/`cross_entropy_lambda`/`kullback_leibler`) | algorithm/math | transform | `lgbm-metric::binary::BinaryMetric` | role-match |
| `lgbm-metric/src/multiclass.rs` (`multi_error`/`auc_mu`/`average_precision`) | algorithm/math | transform | `lgbm-metric::multiclass::MultiLogloss` | role-match |
| `lgbm-metric/src/rank.rs` (`ndcg`/`map`) | algorithm/math | transform (per-query) | `lgbm-metric::regression::Metric::eval` ordered fold + new dcg | role-match |
| `lgbm-metric/src/dcg_calculator.rs` (NEW) | algorithm/math | batch (static tables) | `lgbm-core::random::Random` (static-init-once analog) | partial |
| `lgbm-boosting/src/sample_strategy.rs` (`GossSampleStrategy`) | algorithm/math | event-driven (RNG) | `lgbm-boosting::sample_strategy::BaggingSampleStrategy` | exact |
| `lgbm-boosting/src/sample_strategy.rs` (`bagging_by_query` branch) | algorithm/math | event-driven (RNG) | `BaggingSampleStrategy::bagging` (per-block RNG loop) | exact |
| `lgbm-boosting/src/gbdt.rs` (`BoostingVariant {Gbdt,Dart,Rf}` field + `train_one_iter` branches) | algorithm/math | request-response | `lgbm-boosting::gbdt::Gbdt::train_one_iter` | exact |
| `lgbm-treelearner/src/learner.rs` (`bin_type` dispatch + `find_best_threshold_categorical`) | algorithm/math | CRUD (histogram scan) | `learner::scan_leaf_histogram` + `per_bin_gains` (numeric scan) | role-match (HARD INVARIANT) |
| `lgbm-treelearner/src/learner.rs` (monotone/interaction/forced/extra/cegb gates) | algorithm/math | CRUD | `learner::scan_leaf_histogram` ColSampler gate | role-match |
| `lgbm-model/src/tree.rs` (`split_categorical` ctor + cat model-text emit) | model | file-I/O | `Tree::split` (numeric) + `Tree::categorical_decision` (already present) | exact |
| `lgbm-model/src/predict.rs` (TreeSHAP `predict_contrib`) | algorithm/math | request-response | `predict.rs::predict_row_transformed` / `Tree::get_leaf` | role-match |
| `lgbm-model/src/predict.rs` (pred-early-stop) | algorithm/math | streaming (accumulate) | `predict.rs::predict_raw_mat_range` accumulation loop | role-match |
| `lgbm-model/src/ensemble.rs` (feature importance gain) | model | batch | `GbdtModel::feature_importance_split_count` | exact |
| `lgbm-boosting/src/gbdt.rs` (refit / continue-training) | model | request-response | `Gbdt::into_model` / `pop_trailing_trees` + `with_objective` | role-match |
| `lgbm-core/src/config/set.rs` (+ new params) | config param | — | `config/set.rs` typed-getter + `try_finalize` block | exact |
| `lgbm-core/src/config/alias.rs` (+ new aliases) | config param | — | `config/alias.rs::resolve_alias` | exact |
| `lgbm/src/builder.rs` (one setter / new param) | builder setter | — | `builder.rs::objective` / `reg_sqrt` / `bagging_fraction` | exact |
| `oracle-harness/tests/*_parity.rs` (new cells/files) | test/oracle golden | file-I/O | `oracle-harness/tests/boosting_parity.rs` | exact |
| `xtask/src/main.rs` (new capture subcommands) | test/oracle golden | file-I/O | `xtask::boosting_oracle_capture` | exact |

---

## Pattern Assignments

### Objectives OBJ-04/05 — `lgbm-objective/src/regression.rs` (+xentropy.rs) (algorithm/math, transform)

**Analog:** `crates/lgbm-objective/src/regression.rs` (the `enum Objective` with `parse`/`from_config`/`get_gradients`/`boost_from_score`/`renew_leaf_output`/`is_renew_tree_output`/`is_constant_hessian`/`transform_labels`). Add each new objective as either (a) a new `enum Objective` variant (regression family) or (b) a sibling struct in a new module mirroring `binary.rs`'s struct shape.

**Enum-variant + parse + flags pattern** (`regression.rs:40-133`): each objective adds a variant, an alias arm in `parse`, and an arm in every flag method (`is_constant_hessian`, `is_renew_tree_output`). Note quantile/mape are `is_renew_tree_output() == true` like `RegressionL1` — they reuse `renew_leaf_output` + `percentile::percentile_fun`.

**Per-row grad/hess pattern with V5 length-validation gate** (`regression.rs:174-221`):
```rust
pub fn get_gradients(&self, score: &[f64], label: &[f32],
    gradients: &mut [f32], hessians: &mut [f32]) -> Result<(), ObjectiveError> {
    let n = score.len();
    if label.len() != n { return Err(ObjectiveError::LengthMismatch { expected: n, actual: label.len() }); }
    if gradients.len() != n { /* … */ }
    if hessians.len() != n { /* … */ }
    match self {
        Objective::Regression { .. } => for i in 0..n {
            gradients[i] = (score[i] - label[i] as f64) as f32;   // f64 subtract, single f32 cast
            hessians[i] = 1.0f32;
        },
        // new objectives: huber clips grad to ±alpha; poisson/gamma/tweedie use .exp() in f64 then cast
    }
    Ok(())
}
```
**Load-bearing op order (mirror exactly for the new exp/log objectives):** arithmetic in `f64`, narrow to `score_t = f32` *exactly once* per write (the `(… ) as f32` at the end). `kEpsilon` via `lgbm_core::types::K_EPSILON` (`regression.rs:24`), never a fresh literal.

**`BoostFromScore` ordered-fold pattern** (`regression.rs:229-252`): single sequential f64 fold over labels in row order (the deterministic anchor — C++ `if(!deterministic_)` strips OpenMP). Poisson/tweedie wrap it in `SafeLog`; binary/xentropy clamp to `[kEpsilon, 1-kEpsilon]` (see `binary.rs:112-116`).

**Sigmoid/exp pattern + typed-error ctor (for xentropy/rank):** `binary.rs:38-96` — the `Binary::new(sigmoid)` typed-reject of `sigmoid <= 0` (`binary.rs:45-52`, mirrors C++ `Log::Fatal`) and the f64 `response = -label_val * sigmoid / (1.0 + (label_val * sigmoid * score[i]).exp())` then `as f32` cast (`binary.rs:87-94`).

---

### Ranking objectives OBJ-06 — `lgbm-objective/src/rank.rs` (algorithm/math, transform per-query)

**Analog:** `crates/lgbm-objective/src/multiclass.rs` (`MulticlassSoftmax`, `multiclass.rs:55-205`) for the **strided per-group dispatch** shape — ranking iterates `query_boundaries` instead of class strides, but the structure (struct holds metadata captured at construction, `get_gradients` gathers per-group, `num_model_per_iteration`/`class_need_train` analogs) is identical. Reuse `binary.rs`'s sigmoid f64→f32 pattern for the lambda sigmoid table.

**Construction-captures-metadata pattern** (`multiclass.rs:76`): `MulticlassSoftmax::new(num_class, labels)` captures labels at construction; ranking's analog captures `query_boundaries` + `label_gain` + `objective_seed`. `query_boundaries` is consumed from `lgbm-dataset::metadata::Metadata::query_boundaries` (`metadata.rs:58` — already prefix-summed, present since DAT-06).

**DCGCalculator (NEW, `lgbm-metric/src/dcg_calculator.rs`):** no exact analog; closest is the static-init discipline. Init the discount/gain tables ONCE (do not recompute per query — RESEARCH Anti-Pattern). Shared by OBJ-06 and MET-04, so it lives in `lgbm-metric` and OBJ-06 depends on it.

---

### Metrics MET-03/04 — `lgbm-metric/src/regression.rs` (+rank.rs) (algorithm/math, transform)

**Analog:** `crates/lgbm-metric/src/regression.rs` (the `enum Metric` with `parse`/`name`/`factor_to_bigger_better`/`eval`).

**Enum + parse + factor + ordered-eval pattern** (`regression.rs:27-111`):
```rust
pub fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, MetricError> {
    let n = scores.len();
    if labels.len() != n { return Err(MetricError::LengthMismatch { expected: n, actual: labels.len() }); }
    if n == 0 { return Ok(0.0); }
    let mut sum_loss = 0.0f64;                 // ordered sequential f64 fold (deterministic anchor)
    for i in 0..n {
        let diff = scores[i] - labels[i] as f64;
        sum_loss += match self { Metric::L2 | Metric::Rmse => diff * diff, Metric::L1 => diff.abs() };
    }
    let sum_weights = n as f64;
    Ok(match self { Metric::L2 | Metric::L1 => sum_loss / sum_weights, Metric::Rmse => (sum_loss / sum_weights).sqrt() })
}
```
Each new metric adds: an `enum Metric` variant, a `parse` alias arm, a `name` arm, a `factor_to_bigger_better` arm (`-1` for losses, `+1` for AUC/AP — `regression.rs:65-69`), and a `LossOnPoint`/`AverageLoss` arm. Prob-space metrics call `lgbm_model::ObjectiveKind::convert_output` first (do NOT re-port the transform — `lib.rs:11-14`).

**Per-query eval (ndcg/map):** model on `regression.rs::eval`'s ordered fold but iterate `query_boundaries` groups, averaging per-query DCG via the new `dcg_calculator`. `factor_to_bigger_better = +1`.

---

### Boosting variants BST-04/05/06 — `lgbm-boosting/src/gbdt.rs` + `sample_strategy.rs`

#### GOSS (BST-04) — `sample_strategy.rs::GossSampleStrategy` (algorithm/math, event-driven RNG)

**Analog:** `crates/lgbm-boosting/src/sample_strategy.rs::BaggingSampleStrategy` — the SAME per-block RNG window + draw-every-row-in-order discipline.

**CRITICAL RNG pattern — build rands ONCE, advance across draws** (`sample_strategy.rs:113-118, 142-147, 187-216`):
```rust
pub const BAGGING_RAND_BLOCK: i32 = 1024;
// reset: construct per-block Random ONCE (NOT per draw — re-seeding re-draws the SAME bag)
let bagging_rands: Vec<Random> = (0..n_blocks).map(|i| Random::new(config.bagging_seed + i)).collect();
// bagging(): each row draws IN ORDER from the continuing stream
for i in 0..cnt {
    let block = (i / BAGGING_RAND_BLOCK) as usize;
    let draw = rands[block].next_float() as f64;        // f32 next_float promoted to f64 for compare
    if draw < threshold { /* in-bag */ } else { /* OOB */ }
}
```
GOSS reuses this identical `bagging_rands_` block array (`goss.hpp:95-98`); the new bit is the `ArgMaxAtK` top-k threshold + `multiply = (cnt-top_k)/other_k` grad/hess amplification (RESEARCH §"GOSS amplification"). GOSS modifies grad/hess in place (`IsHessianChange`), so it runs INSIDE `train_one_iter` after `get_gradients`, before the learner.

**RNG-replay golden pattern** (`sample_strategy.rs:272-340` test `reference_bag` + `bag_indices_match_rng_replay_golden`): re-derive the expected draw sequence with a verbatim re-impl over `lgbm_core::Random`, assert `compare_exact`. GOSS/DART/bagging_by_query each get this (RESEARCH §"RNG-Replay Golden Specs").

#### DART/RF (BST-05/06) — `gbdt.rs::BoostingVariant` field + branches (algorithm/math, request-response)

**Analog:** `crates/lgbm-boosting/src/gbdt.rs::Gbdt::train_one_iter` (`gbdt.rs:209+`) and the constructor family (`new`/`with_objective`/`with_bagging`, `gbdt.rs:98-159`). Add a `variant: BoostingVariant {Gbdt, Dart, Rf}` field (RESEARCH Pattern 1 — enum field, NOT trait objects) and branch inside `train_one_iter`/`get_training_score`. RF uses `average_output` + `MultiplyScore` rescale via the existing `ScoreUpdater`; DART overrides predict-side normalize (touches `lgbm-model` tree weights) and uses a single advancing `Random(drop_seed)`.

**Builder-style chained-config pattern** (`gbdt.rs:150-159` `with_bagging`): add `with_variant(...)` chaining the same way. `bagging_by_query` removes the typed reject at `sample_strategy.rs:83-85` (`BoostingError::BaggingByQueryDeferred`) and adds the query-grouped branch.

---

### Categorical splits TRL-06 — `lgbm-treelearner/src/learner.rs` (algorithm/math, CRUD) — **HARD INVARIANT**

**Analog:** `crates/lgbm-treelearner/src/learner.rs::scan_leaf_histogram` (`learner.rs:974-1063`) and `per_bin_gains` (`learner.rs:1068-1200+`). Add a `bin_type` dispatch at the TOP of the per-feature loop (mirroring `serial_tree_learner.cpp:779`), routing `Numerical` to the existing byte-untouched scan and `Categorical` to a new `find_best_threshold_categorical`.

**Per-feature dispatch + cross-feature argmax pattern** (`learner.rs:1004-1061`):
```rust
for (fpos, f) in features.iter().enumerate() {
    if let Some(mask) = used_features {           // ColSampler / interaction-allowed gate goes here
        if mask.get(fpos).copied().unwrap_or(1) == 0 { continue; }
    }
    let cells = 2 * f.num_bin as usize;
    let hist = &buf[slot_off[fpos]..slot_off[fpos] + cells];
    // NEW: if f.bin_type == Categorical -> find_best_threshold_categorical(hist, f, …)
    //      else (BYTE-UNTOUCHED) -> backend.find_best_split(…)  ← D-06 numeric spine
    let split = self.backend.find_best_split(self.client, hist, &self.cfg, f.num_bin, f.offset, …)?;
    if split.gain > K_MIN_SCORE
        && split_gt(&split, f.real_feature_index, &leaf_best, leaf_best_feature) {
        leaf_best = split; leaf_best_feature = f.real_feature_index;
    }
}
```

**Count-reconstruction pattern (reuse exactly, do NOT re-derive)** (`per_bin_gains`, `learner.rs:1090-1093`): `cnt_factor = num_data / (sum_hessian + 2*eps)`, `RoundInt(x) = (x + 0.5f32 as f64) as i32`. Categorical's `cat_smooth` filter uses the SAME `RoundInt(hess * cnt_factor)` (RESEARCH Pitfall 4). The `l2 += cat_l2` asymmetry (cat_l2 only in the per-category gain, original l2 in `gain_shift`) is the deliberate divergence.

> **MUST-NOT-REGRESS analogs (D-06 hard invariant):** the bit-exact numeric-spine goldens — `spine_real.txt`, `mfb_pos_real.txt`, and the growth-path/subtract gates in `crates/oracle-harness/tests/learner_parity.rs` — MUST still replay bit-exact after the categorical re-open. Gate the new code behind a pure `bin_type` branch; assert these goldens green after EVERY categorical commit (RESEARCH Pitfall 2). See `## Shared Patterns → No-Regression Gate`.

**Monotone/interaction/forced/extra/cegb (ADV-01..05):** all hook the same per-feature loop / cross-feature argmax (`learner.rs:1004-1061`). Interaction = the `used_features` mask gate (already present, `learner.rs:1008-1012`). Monotone alters the `split_gt` direction check. Extra-trees adds a `meta.rand.next_int` randomized threshold branch (RNG-replay candidate, reuse the `Random` pattern).

---

### Categorical node + model-text TRL-06/D-07 — `lgbm-model/src/tree.rs` (model, file-I/O)

**Analog:** `Tree::split` (numeric growth, `tree.rs:276-350`) for the new `split_categorical` constructor, and `Tree::categorical_decision` (`tree.rs:180-198`, ALREADY PRESENT from Phase-3 DAT-08) + `find_in_bitset` (`tree.rs:143-150`) for the prediction side — these already work; Phase 7 adds the SPLIT-FINDING + model-text-EMIT side.

**Parallel-array growth pattern** (`tree.rs:276-350`): `split_categorical` mirrors `split` but sets the `CATEGORICAL_MASK` bit (`tree.rs:46`) in `decision_type`, sets `threshold = num_cat as f64`, pushes the bitset into `cat_threshold` (`tree.rs:93`) + the offset into `cat_boundaries` (`tree.rs:91`), and `num_cat += 1`.

**Model-text emit pattern** (`tree.rs:363-381` `to_string`): the `num_cat > 0` branch (`tree.rs:379-381`) already emits `cat_boundaries=`/`cat_threshold=`; the parse side (`tree.rs:533-553`) already round-trips them. Phase 7 wires the GROWN (not just loaded) categorical tree through this path. `%.17g`/`%g` formatters via `format::format_g17`/`format_g6` (`tree.rs:43`) — never a fresh formatter.

---

### Prediction modes PRD-04/05 — `lgbm-model/src/predict.rs` (algorithm/math)

**Analog (SHAP):** `predict.rs::predict_row_transformed` (`predict.rs:328+`) + `Tree::get_leaf`/`predict` (`tree.rs:212-237`) for tree traversal; the cover comes from `Tree.leaf_count`/`internal_count`/`leaf_weight` (already on the struct, `tree.rs:81-89`). TreeSHAP recurses the parallel-array node structure (`left_child`/`right_child`/`split_feature`/`threshold`).

**Analog (pred-early-stop):** `predict.rs::predict_raw_mat_range` accumulation loop (`predict.rs:89-129`) — the per-row f64 accumulator. Pred-early-stop adds a margin check every `pred_early_stop_freq` iterations on the running accumulator.

**Driver wrapper pattern** (`predict.rs:75-129`): thin `predict_*_mat` → `predict_*_mat_range` wrappers with a `check_cols` V5 gate (`predict.rs:42`). SHAP/early-stop add sibling drivers in this same shape.

---

### Feature importance ADV-07 + Refit ADV-06 — `lgbm-model/src/ensemble.rs` + `lgbm-boosting/src/gbdt.rs`

**Analog (importance):** `GbdtModel::feature_importance_split_count` (`ensemble.rs:108`). Gain-based importance adds a sibling summing `split_gain` (mind the CR-02 `split_gain > 0` guard, RESEARCH). The `model_text.rs::save` already recomputes the `feature_importances:` block (`model_text.rs:264-276`) — extend for `saved_feature_importance_type`.

**Analog (refit):** `Gbdt::into_model` (`gbdt.rs:557`), `pop_trailing_trees` (`gbdt.rs:546`), and `with_objective` continue-from-model. Leaf-refit reuses Phase-3 model load (`model_text.rs::load`); continue-training reuses `num_init_iteration` accounting (DART/RF already reference it).

---

### Config params (all items) — `lgbm-core/src/config/set.rs` + `alias.rs`

**Analog:** `crates/lgbm-core/src/config/set.rs`. Typed getters (`get_int`/`get_double`/…, `set.rs:494+`) and the `try_finalize` derived-default / alias-expansion block (`set.rs:440-483`). Note the GOSS alias-expansion already present (`set.rs:472-476`: `boosting == "goss"` → `gbdt` + `data_sample_strategy = "goss"`) and the `bagging_by_query` gate (`set.rs:478-481`).

**Alias pattern** (`alias.rs:192` `resolve_alias`): each new param's aliases (e.g. `forced_splits`/`fs`, `ndcg_eval_at`/`ndcg_at`) add arms here.
**Action (RESEARCH A2):** re-grep `set.rs` per param before adding — many are already present (`top_rate`/`other_rate`/`drop_rate`/`cat_smooth`/`monotone_*`/`extra_trees`). Never trust the RESEARCH "present/ADD" table without re-grep.

---

### Builder setters (all items) — `lgbm/src/builder.rs`

**Analog:** `builder.rs::objective` (`builder.rs:52-55`), `reg_sqrt` (`builder.rs:121-124`), `bagging_fraction` (`builder.rs:152-156`). Every new param is one chained setter inserting into `self.params` (the string map routed through `Config::from_params`). Bool setters mirror `reg_sqrt`/`boost_from_average` (`builder.rs:108-124`).
```rust
pub fn drop_rate(mut self, v: f64) -> Self { self.params.insert("drop_rate".into(), v.to_string()); self }
```
Shaped 1:1 for the Phase-8 Python bindings.

---

### Oracle goldens (all items) — `oracle-harness/tests/*_parity.rs` + `xtask/src/main.rs`

**Analog:** `crates/oracle-harness/tests/boosting_parity.rs` — the corpus-builder + `cell_builder(objective, bfa)` cross-product (`boosting_parity.rs:44-176`), the layered assert helpers (`assert_model_and_pred`/`assert_scores`/`assert_gradients` `boosting_parity.rs:199-301`), and the **capture-gated skip-pass** `read_golden` (`boosting_parity.rs:316-330` — a fresh checkout without the wheel-capture still builds, tests skip-pass until the golden exists).

**Capture-subcommand analog:** `xtask::boosting_oracle_capture` (registered `xtask/src/main.rs:115`), the version-pinned capture pattern (`main.rs:84-97` — assert installed `lightgbm==4.6.0` before training so a wrong version can never silently emit a golden). Each Phase-7 subsystem adds a sibling subcommand + a `tests/fixtures/<subsystem>/` dir. Comparators: `oracle_harness::comparator::ORACLE_TOL = 1e-6` (`comparator.rs:15`), `compare_exact_*` for bit-exact (RNG indices / f64 bits / bytes, `comparator.rs:125-172`).

---

## Shared Patterns

### V5 length-validation gate
**Source:** `lgbm-objective/src/regression.rs:181-199`, `lgbm-metric/src/regression.rs:82-88`
**Apply to:** every new `get_gradients` / `eval` / objective ctor. Validate all slice lengths and typed-reject (`ObjectiveError::LengthMismatch` / `MetricError::LengthMismatch` / `Unsupported` for bad params) BEFORE any per-row write — never a panic, mirroring the C++ `Log::Fatal` as a typed error (`binary.rs:45-52`).

### f64-compute / single-f32-cast op order
**Source:** `lgbm-objective/src/regression.rs:203`, `binary.rs:90-93`
**Apply to:** all objective grad/hess + metric loss math. Arithmetic in `f64`, narrow to `score_t = f32` exactly once at the write. `kEpsilon`/`kZeroThreshold` via `lgbm_core::types::{K_EPSILON, K_ZERO_THRESHOLD}` (`regression.rs:24`, `tree.rs:40`) — never a fresh literal (RESEARCH Anti-Pattern: `1e-35f as f64`).

### Ordered sequential fold (deterministic anchor)
**Source:** `lgbm-objective/src/regression.rs:235-240`, `lgbm-metric/src/regression.rs:92-102`
**Apply to:** every `BoostFromScore` / metric reduction / histogram-adjacent sum. Single in-order f64 fold (C++ `if(!deterministic_)` strips OpenMP) — the bit-exact CPU gate.

### RNG: build-once / advance-across-draws + replay golden
**Source:** `lgbm-boosting/src/sample_strategy.rs:113-118, 187-216` (impl) and `:272-340` (replay test)
**Apply to:** GOSS, DART drop, bagging_by_query, extra-trees threshold. Construct `Random(seed + block)` ONCE; advance the SAME instances across draws (re-seeding re-draws the same bag — the documented CRITICAL fix). Each stochastic draw gets a dedicated `compare_exact` RNG-replay golden (RESEARCH §"RNG-Replay Golden Specs").

### Enum-dispatch factory mirroring C++ string-keyed Create*
**Source:** `lgbm-objective/src/regression.rs::Objective` (`parse`), `lgbm-metric/src/regression.rs::Metric` (`parse`), `lgbm-boosting/src/objective.rs::BoostObjective`
**Apply to:** every new objective/metric/variant. Add a variant + a `parse`/factory arm with the C++ aliases; reject unknown names with a typed `Unsupported` error (never a silent default).

### Capture-gated skip-pass oracle test
**Source:** `oracle-harness/tests/boosting_parity.rs:316-330` (`read_golden` returns `Option`, skips when absent) + `xtask/src/main.rs:84-97` (version-pinned capture)
**Apply to:** every Phase-7 golden. Test skip-passes until the `lightgbm==4.6.0` wheel capture (human-gated checkpoint) writes the fixture; capture asserts the pinned version first.

### No-Regression Gate (D-06 HARD INVARIANT)
**Source:** `oracle-harness/tests/learner_parity.rs` — `spine_real.txt` / `mfb_pos_real.txt` / growth-path / subtract goldens
**Apply to:** the categorical re-open (TRL-06) and ANY learner touch (monotone/interaction/forced/extra/cegb). These numeric-spine real-`lib_lightgbm`-4.6 goldens MUST replay BIT-EXACT after every commit. Run `cargo test --workspace` per wave-merge to catch any numeric-spine regression (RESEARCH §"Sampling Rate").

---

## No Analog Found

| File · Symbol | Role | Data Flow | Reason |
|---------------|------|-----------|--------|
| `lgbm-metric/src/dcg_calculator.rs` (static gain/discount tables) | algorithm/math | batch | No precomputed-static-table subsystem exists yet in the Rust crates. Closest discipline is `lgbm-core::random::Random` (init-once state). Use RESEARCH §C++ `dcg_calculator.cpp` `DCGCalculator::Init` as the spec; build tables once, never recompute per query. |
| TreeSHAP recursion (`predict.rs::predict_contrib`) | algorithm/math | request-response | No tree-recursion-with-path-weighting analog exists (existing predict is a flat leaf walk). Cover fields (`leaf_count`/`internal_count`) are present; the `PathElement`/`unique_path` recursion is genuinely new — port from RESEARCH §"TreeSHAP entry" (`tree.h:668-727`). |

> Everything else maps cleanly to a Phase-1→6 Rust analog above. The two genuinely-new algorithms (DCGCalculator, TreeSHAP) plus the categorical split math (which has a *structural* analog in the numeric scan but new gain math) and the D-05 knife-edge are the only FP-risk net-new code; the rest is wiring proven machinery to faithful C++-ported math.

---

## Metadata

**Analog search scope:** `crates/lgbm-objective/src`, `crates/lgbm-metric/src`, `crates/lgbm-boosting/src`, `crates/lgbm-treelearner/src`, `crates/lgbm-model/src`, `crates/lgbm-core/src/config` + `random.rs`, `crates/lgbm-dataset/src/metadata.rs`, `crates/lgbm/src`, `crates/oracle-harness`, `xtask/src`
**Files scanned:** ~30 Rust source files (read in full where ≤ ~430 lines; targeted reads for `learner.rs`/`tree.rs`/`gbdt.rs`/`set.rs`)
**Pattern extraction date:** 2026-06-07

## PATTERN MAPPING COMPLETE

**Phase:** 7 - Parity-Completing Variants
**Files classified:** 25 new/modified symbol-groups across 8 crates + oracle-harness/xtask
**Analogs found:** 18 / 18 subsystems (23 / 25 symbol-groups have an exact-or-role Rust analog; 2 net-new algorithms documented)

### Coverage
- Symbol-groups with exact analog: 14
- Symbol-groups with role-match / partial analog: 9
- Symbol-groups with no analog: 2 (DCGCalculator static tables, TreeSHAP recursion)

### Key Patterns Identified
- All objectives/metrics extend an existing `enum` factory (`Objective`/`Metric`/`BoostObjective`) with a variant + `parse` alias arm + per-method arms; grad/hess and loss math follow the f64-compute / single-f32-cast / ordered-fold discipline with a V5 length gate.
- All stochastic variants (GOSS/DART/bagging_by_query/extra-trees) reuse the `BaggingSampleStrategy` build-rands-ONCE / advance-across-draws RNG pattern + a dedicated `compare_exact` RNG-replay golden.
- The categorical re-open is a `bin_type` dispatch branch grafted onto `learner.rs::scan_leaf_histogram`'s existing per-feature loop; the numeric spine stays byte-untouched and its `spine_real`/`mfb_pos_real` learner goldens are the must-not-regress D-06 invariant.
- Every golden mirrors `boosting_parity.rs`'s capture-gated skip-pass + version-pinned `xtask` capture subcommand; the categorical/SHAP node + model-text reuse the already-present `Tree` categorical fields, `find_in_bitset`, parallel-array `split`, and `%.17g`/`%g` formatters.

### File Created
`.planning/phases/07-parity-completing-variants/07-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. The planner can reference each analog file:line in its plan action sections — every Phase-7 subsystem has a concrete Rust shape to mirror, the two net-new algorithms (DCGCalculator, TreeSHAP) are flagged for direct C++ port, and the D-06 numeric-spine no-regression goldens are flagged as must-not-regress.
