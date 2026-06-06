# Phase 6: GBDT Spine + Core Objectives/Metrics - Pattern Map

**Mapped:** 2026-06-07
**Files analyzed:** 18 (4 new crates × scaffold+modules, 2 extended crates, xtask, oracle-harness)
**Analogs found:** 18 / 18 (every Phase-6 file has a strong in-workspace analog from Phases 1–5)

This phase is, per RESEARCH §"Don't Hand-Roll", **overwhelmingly wiring proven components together in the exact C++ order** — so nearly every new file has a precise existing analog to mirror. The C++ source named in 06-CONTEXT `<canonical_refs>` is authoritative for the *numerical formulas*; the analogs below are authoritative for the *Rust idiom, crate shape, error pattern, golden layout, and capture pipeline* the executor must copy.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-objective/Cargo.toml` | config | — | `crates/lgbm-model/Cargo.toml` | exact (crate-with-deps scaffold) |
| `crates/lgbm-objective/src/lib.rs` | service (objective factory) | transform | `crates/lgbm-model/src/objective.rs` (`ObjectiveKind` enum + `parse`) | exact (enum-dispatch factory mirror) |
| `crates/lgbm-objective/src/error.rs` | utility (error enum) | — | `crates/lgbm-core/src/error.rs` / `lgbm-treelearner/src/error.rs` | exact (thiserror idiom) |
| `crates/lgbm-metric/Cargo.toml` | config | — | `crates/lgbm-model/Cargo.toml` | exact |
| `crates/lgbm-metric/src/lib.rs` | service (metric factory) | transform / batch-reduce | `crates/lgbm-model/src/objective.rs` (enum + per-variant math) | role-match (reduction vs transform) |
| `crates/lgbm-metric/src/error.rs` | utility (error enum) | — | `crates/lgbm-core/src/error.rs` | exact |
| `crates/lgbm-boosting/Cargo.toml` | config | — | `crates/lgbm-treelearner/Cargo.toml` (multi-dep, dev-dep on oracle-harness) | exact |
| `crates/lgbm-boosting/src/gbdt.rs` | service (orchestration loop) | event-driven (per-iter) | `crates/lgbm-treelearner/src/learner.rs` (`train` driver + V5 validation) | role-match (loop vs per-tree driver) |
| `crates/lgbm-boosting/src/score_updater.rs` | service (f64 accumulator) | streaming / accumulate | `crates/lgbm-treelearner/src/data_partition.rs` (per-leaf scatter bookkeeping) | role-match |
| `crates/lgbm-boosting/src/sample_strategy.rs` | service (bagging RNG) | event-driven (per-round draw) | `crates/lgbm-core/src/random.rs` (`Random::sample` draw-loop idiom) | role-match (RNG consumer) |
| `crates/lgbm-boosting/src/error.rs` | utility (error enum) | — | `crates/lgbm-treelearner/src/error.rs` | exact |
| `crates/lgbm/Cargo.toml` | config (umbrella facade) | — | `crates/lgbm-treelearner/Cargo.toml` | exact |
| `crates/lgbm/src/builder.rs` | controller (public builder) | request-response | *no direct analog* — bounded by `lgbm-core::Config` (D-02) | no-analog (see §No Analog Found) |
| `crates/lgbm/src/booster.rs` | controller (public Booster) | request-response | `crates/lgbm-model/src/ensemble.rs` (`GbdtModel` container + accessors) | role-match |
| `crates/lgbm/src/lib.rs` | controller (train/predict re-export) | request-response | `crates/lgbm-model/src/lib.rs` (re-export surface) | exact (facade lib.rs shape) |
| `crates/lgbm-treelearner` ext: `add_prediction_to_score` + `renew_tree_output` | service (method addition) | streaming / transform | `crates/lgbm-treelearner/src/data_partition.rs` `indices_in_leaf` | exact (already-owned partition) |
| `crates/lgbm-model::Tree` ext: `shrinkage(rate)` / `add_bias(val)` | model (method addition) | transform | `crates/lgbm-model/src/tree.rs` `Tree` (has `shrinkage` *field*, no *method*) | exact (same struct) |
| `xtask` ext: `boosting-oracle-capture` + `xtask/py/` + `Random.NextFloat` dump | service (capture subcommand) | file-I/O / batch | `xtask/src/main.rs` `learner_oracle_capture` + `xtask/py/learner_oracle_capture.py` | exact |
| `crates/oracle-harness/tests/boosting_parity.rs` + fixtures + manifest | test (golden replay) | file-I/O / batch | `oracle-harness/tests/learner_parity.rs` + `comparator.rs` + `REFERENCE_MANIFEST.md` | exact |

---

## Pattern Assignments

### `crates/lgbm-objective/Cargo.toml` + `lgbm-metric` + `lgbm-boosting` + `lgbm` (crate scaffolds)

**Analog:** `crates/lgbm-treelearner/Cargo.toml` (multi-dep), `crates/lgbm-model/Cargo.toml` (lean dep + dev-dep).

**Scaffold shape to copy** (`lgbm-treelearner/Cargo.toml`):
```toml
[package]
name = "lgbm-treelearner"
version = "0.1.0"
edition.workspace = true          # always inherit edition + rust-version from workspace
rust-version.workspace = true

[dependencies]
lgbm-core = { path = "../lgbm-core" }       # path deps, NOT versioned
lgbm-dataset = { path = "../lgbm-dataset" }
thiserror = { workspace = true }            # workspace dep, never a literal version

[dev-dependencies]
oracle-harness = { path = "../oracle-harness" }   # parity tests
```

**Concrete dep map per new crate (from RESEARCH §Architectural Responsibility Map):**
- `lgbm-objective`: deps `lgbm-core` (types/`K_EPSILON`/`Config`), `lgbm-dataset` (labels/weights), `lgbm-model` (reuse `ObjectiveKind::convert_output` per Open-Q1 recommendation), `thiserror`. dev-dep `oracle-harness`.
- `lgbm-metric`: deps `lgbm-core`, `lgbm-dataset`, `lgbm-model` (ConvertOutput for prob-space metrics), `thiserror`. dev-dep `oracle-harness`.
- `lgbm-boosting`: deps `lgbm-core`, `lgbm-dataset`, `lgbm-model`, `lgbm-treelearner`, `lgbm-objective`, `lgbm-metric`, `thiserror`. **MUST NOT depend on `cubecl`/`lgbm-compute` runtime types** (CMP-01 — see `lgbm-treelearner/Cargo.toml` comment block; the learner already isolates the compute seam). dev-dep `oracle-harness`.
- `lgbm` (facade): deps all of the above + `lgbm-core::Config`. This is the Phase-8 PyO3 target.

**Workspace registration** — add all four crates to the `members` list in the root `Cargo.toml`:
```toml
members = [
    "crates/lgbm-core", "crates/lgbm-compute", "crates/lgbm-dataset",
    "crates/lgbm-model", "crates/lgbm-treelearner", "crates/oracle-harness",
    "xtask",
    # ADD: "crates/lgbm-objective", "crates/lgbm-metric",
    #      "crates/lgbm-boosting", "crates/lgbm",
]
```
(RESEARCH Runtime State Inventory: these are *net-additions* — no rename/migration.)

---

### `crates/lgbm-objective/src/lib.rs` (service, transform)

**Analog:** `crates/lgbm-model/src/objective.rs` — the existing `ObjectiveKind` enum is the *exact* idiom for the C++ string-keyed `CreateObjectiveFunction` factory (RESEARCH "Alternatives Considered": enum-dispatch is the recommended mirror, + a `custom` variant holding the D-04 closure).

**Enum-factory pattern** (objective.rs:43-72) — mirror the variant-per-objective + parsed-params shape:
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveKind {
    Regression { sqrt: bool },
    Binary { sigmoid: f64 },
    Multiclass { num_class: i32 },
    MulticlassOva { num_class: i32, sigmoid: f64 },
}
```
For Phase 6 this enum gains the **training** side. Add `regression_l1` (RESEARCH: distinct `BoostFromScore`=median + `IsRenewTreeOutput`=true) and a `Custom(closure)` variant. Each variant implements `get_gradients`, `boost_from_score`, and (for l1) `renew_tree_output`. The predict-side `convert_output` already lives here (objective.rs:198 `softmax(...)`) — **reuse it, do not re-port** (Open-Q1 recommendation: keep ConvertOutput in `lgbm-model`, `lgbm-objective` owns only the training side).

**Doc-citation idiom to copy** (objective.rs:1-31) — every objective module header cites the exact C++ file+line it mirrors (`regression_objective.hpp:148`, `common.h:587`). Phase-6 objective files must carry the same citation discipline against the RESEARCH §"Objective Formulas" line refs.

**Sign helper** (objective.rs:34-38) — already ported, reuse for `regression_l1` grad:
```rust
fn sign(x: f64) -> f64 { ((x > 0.0) as i32 - (x < 0.0) as i32) as f64 }
```

**Spine GetGradients (L2, the D-14 starting point)** — from RESEARCH §Code Examples:
```rust
for i in 0..num_data {
    gradients[i] = (score[i] - label[i] as f64) as f32;  // score_t cast; score is f64
    hessians[i]  = 1.0f32;
}
```

---

### `crates/lgbm-metric/src/lib.rs` (service, batch-reduce)

**Analog:** `crates/lgbm-model/src/objective.rs` (enum + per-variant math + C++ citations) for the *shape*; `crates/oracle-harness/src/comparator.rs` for the *reduction-over-slice + first-divergence* idiom.

**Enum factory** mirroring C++ `CreateMetric` — one variant per metric (`L1`, `L2`, `Rmse`, `BinaryLogloss`, `BinaryError`, `Auc`, `MultiLogloss`), each with `eval(&self, scores: &[f64], dataset) -> f64` and a `factor_to_bigger_better() -> f64` (-1 losses, +1 AUC). Formulas verbatim from RESEARCH §"Metric Formulas" table.

**Prob-space metrics call ConvertOutput first** (RESEARCH binary_metric.hpp:80-83) — `lgbm-metric` depends on `lgbm-model::ObjectiveKind::convert_output` (objective.rs:198) for `binary_logloss`/`binary_error`/`multi_logloss`; do NOT duplicate the transform.

**`K_EPSILON` guard** — use the already-defined constant (types.rs:31), NOT a fresh literal:
```rust
use lgbm_core::types::K_EPSILON;   // 1e-15f — the logloss floor (RESEARCH binary_metric.hpp:119-130)
```

**AUC sort** — RESEARCH Pitfall 1: unstable sort is bit-safe (tie-group-invariant). Use `slice::sort_by(|a,b| score[b].partial_cmp(&score[a]))` then the grouped accumulation loop (RESEARCH binary_metric.hpp:194-251).

---

### `crates/lgbm-objective/src/error.rs`, `lgbm-metric/src/error.rs`, `lgbm-boosting/src/error.rs` (utility, thiserror)

**Analog:** `crates/lgbm-treelearner/src/error.rs` (best — shows `#[from]` wrapping of an upstream crate's error) and `crates/lgbm-core/src/error.rs`.

**thiserror enum idiom** (treelearner/error.rs:23-73) — copy structure exactly: per-variant `#[error("…{field}…")]`, struct-style variants with doc comments, and `#[from]` for upstream errors:
```rust
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ObjectiveError {
    #[error("array length mismatch: expected `{expected}`, got `{actual}`")]
    LengthMismatch { expected: usize, actual: usize },

    // RESEARCH §Security V5: multiclass label out of [0, num_class) — C++ does
    // Log::Fatal (multiclass_objective.hpp:62); we return a typed error, never panic.
    #[error("multiclass label `{label}` out of range (num_class = {num_class})")]
    LabelOutOfRange { label: i32, num_class: i32 },
}
```
**Mandate (CLAUDE.md + every prior error.rs header):** `thiserror` derive only — never hand-roll `impl std::error::Error`. Map C++ `CHECK`/`Log::Fatal` sites to `Result` variants, never a panic (Security V5). `lgbm-boosting::BoostingError` should `#[from]`-wrap `TreeLearnerError`, `ObjectiveError`, `MetricError` (the treelearner/error.rs:66-72 `#[from] lgbm_compute::ComputeError` pattern).

---

### `crates/lgbm-boosting/src/gbdt.rs` (service, event-driven per-iteration loop)

**Analog:** `crates/lgbm-treelearner/src/learner.rs` (the `train`/`train_inner` driver: thin public method → shared inner driver, with V5 boundary validation up front).

**Driver-method layering** (learner.rs:265-312) — copy the "thin public `train` → shared `train_inner` with snapshot variants" shape. The GBDT `train_one_iter` should likewise expose a plain public method plus a snapshot-returning variant for the D-10/D-11 layered goldens (the learner already does this with `train_with_snapshots`):
```rust
pub fn train(&mut self, g: &[f32], h: &[f32], is_first_tree: bool) -> Result<Tree, TreeLearnerError> {
    Ok(self.train_with_snapshots(g, h, is_first_tree)?.0)   // thin → inner
}
```

**V5 validation-first idiom** (learner.rs:329-338) — validate slice lengths / counts before any FP work, returning typed errors:
```rust
if hessians.len() != gradients.len() {
    return Err(TreeLearnerError::LengthMismatch {
        expected: gradients.len(), actual: hessians.len(),
    });
}
```

**The loop body is a verbatim C++ mirror** — `gbdt.rs` is the port of `TrainOneIter` per RESEARCH §Pattern 1 (BoostFromAverage → Boosting/GetGradients → Bagging → per-class tree loop → RenewTreeOutput → Shrinkage → UpdateScore → AddBias). Drive the Phase-5 learner via its existing `train(grad, hess, is_first_tree)` (learner.rs:265) — DO NOT modify the learner's growth. The critical ordering (Shrinkage before UpdateScore, AddBias after, init-score via BoostFromAverage→AddScore not AddBias) is RESEARCH Pitfall 5 + Pattern 1 "Critical ordering note".

---

### `crates/lgbm-boosting/src/score_updater.rs` (service, f64 streaming accumulator)

**Analog:** `crates/lgbm-treelearner/src/data_partition.rs` — the `score_` train-path add IS a per-leaf scatter over the data partition the learner already owns (RESEARCH §Pattern 2 / Anti-Patterns).

**Use the partition's leaf-row accessor** (data_partition.rs:83 `indices_in_leaf`) for the bit-exact training-path add (RESEARCH §Code Examples "Score updater training-path add"):
```rust
// score: &mut [f64], offset = num_data * cur_tree_id  (class-major layout, Pattern 4)
for leaf in 0..tree.num_leaves() {
    let out = tree.leaf_value[leaf as usize];          // f64 leaf output
    for &row in data_partition.indices_in_leaf(leaf) { // ← existing accessor
        score[offset + row as usize] += out;
    }
}
```
**Wave-0 extension (RESEARCH Wave 0 Gaps):** add `SerialTreeLearner::add_prediction_to_score(&tree, &mut [f64])` that performs exactly this scatter over its owned `DataPartition` — the boosting crate calls it rather than re-walking the tree per row.

**f64 accumulator (Anti-Pattern):** `score_` is `Vec<f64>`, NOT `Vec<f32>` (RESEARCH score_updater.hpp:123). Note this is the one place f64 is intentional despite the f32 contract in types.rs (types.rs forbids f64 *score/label aliases* — the accumulator is a local buffer, not an aliased type; cast to f32 only when feeding g/h).

---

### `crates/lgbm-boosting/src/sample_strategy.rs` (service, per-round RNG draw)

**Analog:** `crates/lgbm-core/src/random.rs` — `Random::sample` (random.rs:87-120) is the existing draw-loop idiom over the bit-exact LCG; the bagging draw is the same `next_float()`-per-element pattern.

**RNG-consumer idiom** (random.rs:96-104) — per-element `next_float()` draw with f64-compared probability:
```rust
for i in 0..n {
    if (self.next_float() as f64) < prob { ret.push(i); }   // f32 draw, f64 compare
}
```

**Bagging draw to port** (RESEARCH §Pattern 3 + §Code Examples + Pitfall 4) — per-block `Random(bagging_seed + i)` (block size 1024), draw `next_float()` for EVERY row in order, in-bag appended left / OOB filled from right, then reverse the OOB tail:
```rust
let mut left = 0usize; let mut right = cnt;
for i in 0..cnt {
    if bagging_rands[i / 1024].next_float() < bagging_fraction { buf[left] = i as i32; left += 1; }
    else { right -= 1; buf[right] = i as i32; }
}
buf[left..cnt].reverse();    // bag_data_indices = in-bag asc ++ OOB desc (D-13 golden asserts FULL array)
```
**Reuse, do not re-port the PRNG** (RESEARCH §Don't Hand-Roll): construct `lgbm_core::random::Random::new(bagging_seed + i)` per block; `next_float()` (random.rs:75) is already bit-exact (FND-01).

---

### `crates/lgbm/src/booster.rs` (controller, request-response)

**Analog:** `crates/lgbm-model/src/ensemble.rs` (`GbdtModel`) — the model container with public fields + accessor methods. The `Booster` wraps a trained `GbdtModel` plus the D-05 eval-history/early-stopping fields.

**Container + accessor idiom** (ensemble.rs:35-71):
```rust
pub struct GbdtModel {
    pub trees: Vec<Tree>,
    pub num_class: i32,
    pub num_tree_per_iteration: i32,   // stride: trees[i*ntpi + k]  ← class-major, Pattern 4
    ...
}
impl GbdtModel {
    pub fn num_iteration(&self) -> i32 { self.trees.len() as i32 / self.num_tree_per_iteration }
}
```
**D-05 fields to add on `Booster`** (mirror Python `best_iteration_`/`best_score_`/`record_evaluation`): `best_iteration: i32` + per-valid-set / per-metric eval history. Predict delegates to the existing `GbdtModel::predict_raw` (ensemble.rs:90) → `ObjectiveKind::convert_output` (RESEARCH: predict path already shipped in Phase 3).

---

### `crates/lgbm/src/lib.rs` (controller, facade re-export)

**Analog:** `crates/lgbm-model/src/lib.rs` — the lean facade with `pub mod` + curated `pub use` re-exports:
```rust
pub mod ensemble; pub mod error; pub mod objective; pub mod tree; ...
pub use ensemble::GbdtModel;
pub use error::ModelError;
pub use objective::ObjectiveKind;
pub use tree::Tree;
```
The `lgbm` facade re-exports `Dataset`, `Booster`, `train`, `predict` and the builder so downstream (Phase-8 PyO3) has one import surface.

---

### `crates/lgbm-model::Tree` extension: `shrinkage(rate)` / `add_bias(val)`

**Analog:** `crates/lgbm-model/src/tree.rs` — the `Tree` struct already has the `shrinkage: f64` *field* (tree.rs:95) and `leaf_value: Vec<f64>` (tree.rs:79) but **no `shrinkage(rate)` / `add_bias(val)` methods** (confirmed: grep found field, not method). RESEARCH Wave-0 Gap + Assumption A6: audit & add.

**Add as `impl Tree` methods** mirroring C++ `tree.h:188`/`tree.h:213` with `MaybeRoundToZero` (RESEARCH State-of-the-Art: `IsZero(fval) ? 0 : fval`, snap tiny→+0). Use the existing `K_ZERO_THRESHOLD` (types.rs:35) for the round-to-zero test. `shrinkage(rate)` multiplies all `leaf_value` + `internal_value` and sets `self.shrinkage`; `add_bias(val)` adds to leaf/internal values (model-text only — does NOT touch `score_`, RESEARCH Pitfall 5).

---

### `xtask` extension: `boosting-oracle-capture` (service, file-I/O capture subcommand)

**Analog:** `xtask/src/main.rs` `learner_oracle_capture` (main.rs:739) + `xtask/py/learner_oracle_capture.py` — the pip-wheel (`lightgbm==4.6.0`) capture pattern. This is the EXACT precedent (D-06/D-07 carry P5 D-08).

**Subcommand dispatch** (main.rs:96-108) — add a `match` arm:
```rust
match args.next().as_deref() {
    Some("regen") => regen(),
    ...
    Some("learner-oracle-capture") => learner_oracle_capture(),
    // ADD: Some("boosting-oracle-capture") => boosting_oracle_capture(),
    ...
}
```

**Capture-fn skeleton** (main.rs:739-810) — copy verbatim: resolve `$LGBM_CAPTURE_PYTHON`, **assert the wheel version** before training (so a wrong version can't emit a divergent golden), write fixtures under the TRACKED `crates/oracle-harness/tests/fixtures/` dir (NEVER the untracked `LightGBM/`), then refresh `REFERENCE_MANIFEST.md` idempotently:
```rust
run(Command::new(&python).arg("-c").arg(format!(
    "import lightgbm,sys; assert lightgbm.__version__=='{ver}', ...",
    ver = LEARNER_ORACLE_LIGHTGBM_VERSION)), "lightgbm version check")?;
```

**Python capture script** — model `xtask/py/learner_oracle_capture.py`. Per RESEARCH §Validation Architecture, the new script emits the L1–L5 layers: per-row g/h (D-10, via score-derivation route), per-iter `predict(raw_score=True, num_iteration=k)` (D-11), `record_evaluation`/`evals_result` (D-12), and `save_model()`+`predict()` (D-13/L5). For D-13 bagged indices use **Option A (RNG-replay)**: a small `Random.NextFloat` sequence dumper (extend the existing `rng_capture.cpp` precedent / reuse the proven `rng_sequence.txt` fixture format) — the bag is derived in-Rust and self-checked, no internal-bag capture needed.

---

### `crates/oracle-harness/tests/boosting_parity.rs` + fixtures + manifest (test, golden replay)

**Analog:** `crates/oracle-harness/tests/learner_parity.rs` (59KB layered replay) + `src/comparator.rs` + `fixtures/REFERENCE_MANIFEST.md`.

**Comparator selection** (comparator.rs:92/125/150) — match the layer's precision contract:
- bit-exact f64 (per-iter scores D-11, model-text leaf values L5): `compare_exact_f64_bits`
- exact integers (bagged indices D-13): `compare_exact_u32`
- ~1e-6 f32 (g/h D-10, metrics D-12 if not bit-exact, ROCm cross-check): `compare_within(.., ORACLE_TOL)` (`ORACLE_TOL = 1e-6`, comparator.rs:15)

The `Mismatch` enum (comparator.rs:20-56) reports first-divergence index — copy the learner-parity replay's "localize to one index" assertion style.

**Fixture layout** — new subdir `crates/oracle-harness/tests/fixtures/boosting/` (alongside existing `learner/`, `kernels/`), with per-objective layered `.txt` goldens (L1–L5). Mirror the existing `fixtures/learner/{spine.txt, spine_real.txt, real_gh.txt}` naming and the `REFERENCE_MANIFEST.md` per-set section structure (manifest has dedicated "Learner Golden Set" / "REAL Learner Oracle Set" sections at lines 277/381 — add a "Boosting / Objective / Metric Golden Set (Phase 6)" section documenting every D-07 cell + the one allowed collapse, per RESEARCH §Cross-Product Collapse Analysis).

**Test-crate dev-deps** — `oracle-harness/Cargo.toml` already dev-deps `lgbm-model`, `lgbm-treelearner`, `lgbm-compute` (cubecl pulled in `[dev-dependencies]` ONLY, keeping the library crate cubecl-free). Add `lgbm-boosting`, `lgbm-objective`, `lgbm-metric`, `lgbm` as dev-deps the same way.

---

## Shared Patterns

### thiserror domain-error boundary (FND-04, CLAUDE.md mandate)
**Source:** `crates/lgbm-treelearner/src/error.rs:15-73`, `crates/lgbm-core/src/error.rs:10-69`
**Apply to:** every new crate's `error.rs` (`ObjectiveError`, `MetricError`, `BoostingError`).
- `#[derive(Debug, Error, Clone, PartialEq)]`, struct-style variants, `#[error("…{field}…")]`, `#[from]` to wrap upstream crate errors. Map C++ `CHECK`/`Log::Fatal` → typed `Result`, never panic (Security V5). Never hand-roll `impl Error`.

### Workspace-inherited Cargo metadata + path deps
**Source:** every `crates/*/Cargo.toml`
**Apply to:** all four new crate manifests.
- `edition.workspace = true`, `rust-version.workspace = true`, `thiserror = { workspace = true }`, `anyhow.workspace = true`; intra-workspace deps are `{ path = "../…" }`, never versioned. Add each crate to root `members`.

### f32 numerical contract + named constants
**Source:** `crates/lgbm-core/src/types.rs:11-35`
**Apply to:** all objective/metric/boosting math.
- `ScoreT`/`LabelT` = f32; g/h/labels are f32. Use `K_EPSILON` (1e-15f, types.rs:31) for objective clamps + metric logloss floor; `K_ZERO_THRESHOLD` (types.rs:35) for `MaybeRoundToZero`. The score *accumulator* is a local `Vec<f64>` (intentional, RESEARCH Pattern 2) — distinct from the forbidden f64 *aliases*. Comparison tolerance `ORACLE_TOL` (1e-6) lives in oracle-harness, NOT `K_EPSILON`.

### C++ source-citation doc headers (faithful-mirror discipline)
**Source:** `crates/lgbm-model/src/objective.rs:1-31`, `crates/lgbm-core/src/random.rs:1-21`
**Apply to:** every new objective/metric/boosting module.
- Module doc-comment cites the exact `LightGBM/src/...:line` it ports (RESEARCH §"Objective/Metric Formulas" provide the line refs). This is how a reviewer verifies the 1:1 mirror below the API boundary (CONTEXT carried-forward discipline).

### Enum-dispatch factory (mirror of C++ string-keyed `Create*`)
**Source:** `crates/lgbm-model/src/objective.rs:43-72` (`ObjectiveKind` + `parse`)
**Apply to:** `lgbm-objective` and `lgbm-metric`.
- One enum variant per kind carrying parsed params; a `parse`/`from_config` constructor mirrors `CreateObjectiveFunction`/`CreateMetric`. Allocation-free dispatch over a small fixed set (RESEARCH "Alternatives Considered"). Add a `Custom(closure)` objective variant (D-04).

### Committed-golden + idempotent real-binary capture (P5 D-08 precedent)
**Source:** `xtask/src/main.rs:739` (`learner_oracle_capture`) + `xtask/py/learner_oracle_capture.py` + `fixtures/REFERENCE_MANIFEST.md`
**Apply to:** the new `boosting-oracle-capture` subcommand + fixtures.
- Version-assert the pip wheel before capture; write fixtures under the tracked oracle-harness dir; refresh the manifest idempotently (`git diff` empty on re-capture); `cargo test` replays committed goldens with NO wheel needed. NEVER `git add LightGBM/`.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/lgbm/src/builder.rs` | controller (public builder) | request-response | This is the project's **first net-new user-facing surface** (CONTEXT D-01) — an idiomatic Rust builder, deliberately NOT a C++ mirror. No prior phase shipped a public builder. **Constraint, not analog:** every `.method(val)` resolves into `lgbm-core::Config` (D-02 — Config is the single source of truth; `crates/lgbm-core/src/config/` is the field reference) plus a `from_config(Config)` escape hatch (D-03). The *resolution target* (`Config`) is the anchor; the *ergonomics* are greenfield. Planner: use the standard Rust owned-builder idiom (`Self`-returning setters → `build() -> Result<…, BoostingError>`), validating through `Config` (never forking defaults/aliases). |

> Note: every other Phase-6 file has a strong in-workspace analog. `lgbm/src/builder.rs` is the single genuinely net-new shape — and even it is *bounded* by `Config` rather than fully unconstrained.

---

## Metadata

**Analog search scope:** `crates/lgbm-core/`, `crates/lgbm-treelearner/`, `crates/lgbm-model/`, `crates/oracle-harness/`, `xtask/` (Phase 1–5 deliverables), root `Cargo.toml`.
**Files scanned:** 14 source files read (error.rs ×2, types.rs, lib.rs ×3, random.rs, learner.rs train region, objective.rs, comparator.rs, ensemble.rs, data_partition.rs, tree.rs grep, format.rs grep, main.rs capture region) + 6 Cargo.toml + manifest + fixtures listing.
**Key C++ port targets (formulas, not Rust analogs):** see 06-CONTEXT `<canonical_refs>` and 06-RESEARCH §"Objective/Metric Formulas" / §"GBDT Control Flow" / §"Bagging RNG" — authoritative over any inferred default.
**Pattern extraction date:** 2026-06-07
