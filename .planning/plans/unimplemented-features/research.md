# Research: Unimplemented C++ LightGBM Features in the Rust Port

**Date:** 2026-07-16
**Branch inspected:** `main` (HEAD `27665a4`) — the live perf-campaign branch, which
also carries the merged linear-tree + quantized-grad work.
**Goal:** Identify features present in Microsoft's C++ LightGBM reference that are NOT
yet implemented in `lightgbm_rs`, and recommend the best next functional gaps to close.

> **Evidence-tag legend:** `[VERIFIED: CODEGRAPH …]` codegraph result ·
> `[VERIFIED: LOCAL <path>]` direct read/grep this session · `[INFERRED: …]`
> reasoned from Rust doc-comments citing C++ · `[UNVERIFIED: …]` could not confirm.

---

## 0. Executive Summary (read this first)

The Rust port is **far more complete** than the two prior research docs
(`.planning/plans/cpp-feature-parity/research.md`, `.planning/plans/parity-gap-closure/research.md`)
imply. Both were written on **2026-07-09 to -12 against older branches** and are now
**stale on two of their top items**:

- **Linear tree (`linear_tree=true`) is DONE on `main`** — `fit_linear_leaves` is
  called from the production boosting loop (`gbdt.rs:1037,1204`), scored via
  `ScoreUpdater::add_linear_tree_train_path` (`score_updater.rs:336`), and gated
  correctly against the resident-score device path. **No longer a gap.**
  `[VERIFIED: LOCAL grep gbdt.rs, score_updater.rs]`
- **Quantized-gradient plumbing is DONE on `main`** — `use_quantized_grad`,
  `num_grad_quant_bins`, `quant_train_renew_leaf`, `stochastic_rounding` are all
  `IN_SCOPE_PARAMS` (`scope.rs:93-98`), present as `Config` fields with both
  deterministic AND stochastic rounding implemented (`config/mod.rs:108-120`), and
  exercised by `oracle-harness/tests/quantized_parity.rs`. **No longer a gap.**
  `[VERIFIED: LOCAL grep scope.rs, config/mod.rs; ls oracle-harness/tests]`

After removing those, **four functional C++→Rust gaps remain open on `main`** — and
they are exactly the four already specced (but not yet implemented) in
`.planning/plans/parity-gap-closure/{SPEC,PLAN}.md`:

| ID | Gap | Rust status on `main` | Best next? |
|----|-----|-----------------------|------------|
| **G2** | Model **JSON dump** (`GBDT::DumpModel` / `Tree::ToJSON`) | **ABSENT** | ✅ #1 |
| **G1** | **If-else C++ codegen** (`convert_model` / `ModelToIfElse`) | **ABSENT (config parsed, inert)** | ✅ #2 |
| **G4** | **`na_as_missing`** forward routing (serial host path) | **Deferred (typed error)** | ✅ #3 |
| **G5** | Split-kernel **gain params** (`path_smooth`, `max_delta_step`, feature `penalty`) | **Gated (typed error)** | ✅ #4 |

Plus one **GPU-only** gap that is larger/riskier and needs GPU parity infra:
**G3 categorical-feature GPU kernels** (stubbed). Everything else in the C++
audit surface is either implemented or explicitly out of scope (C-API, MPI/socket
distributed, OpenCL `gpu` device type, fully-resident grow loop).

**All in-flight commits on `main` (T-01..T-13 DRGL, dead-toggle refactor) are
PERFORMANCE work, not functional gaps** — the deferred-sync grow loop was found
~1.20× *slower* and is architecturally shelved (T-11 verdict, `27665a4`).

---

## 1. Environment & Method Constraints

- **The read-only C++ reference tree `LightGBM/` is ABSENT in this sandbox.**
  `find / -maxdepth 4 -iname LightGBM -type d` and `ls LightGBM*` return nothing.
  `[VERIFIED: LOCAL find/ls]` Consequently **every C++ file:line citation below comes
  from the Rust source's own doc-comments** (which cite C++ paths), NOT from reading
  the C++ this session — tagged `[INFERRED: Rust doc-comment cites C++ …]`. The
  `oracle-harness/tests/config_drift.rs` test that mechanically diffs against
  `LightGBM/src/io/config_auto.cpp` cannot run here; **an implementer MUST re-verify
  C++ algorithm details against a checked-out `LightGBM/` 4.6 before the Green step of
  G1/G2/G4/G5** (already flagged as blocker P-1 in the existing PLAN).
- CodeGraph index is present (`.codegraph/codegraph.db`, ~34 MB) and daemon live.
- No PageIndex documents indexed (prior sessions confirmed empty library).
- Method: codegraph + targeted `grep`/`sed`/`ls` on `crates/` (the deliverable), and
  full reads of the four `.planning/` artifacts + `codebase/TESTING.md`.

---

## 2. Current Rust Port Surface (what IS implemented)

### 2.1 Objectives — `crates/lgbm-objective/` — COMPLETE
`[VERIFIED: LOCAL ls + grep regression.rs, rank.rs, xentropy.rs, percentile.rs]`

| C++ (`src/objective/*.hpp`) | Rust | Status |
|---|---|---|
| Regression: l2, l1 | `regression.rs` `Objective::{RegressionL2,RegressionL1}` | ✅ |
| huber, fair, quantile, mape, poisson, gamma, tweedie | `regression.rs` `Objective::{Huber,Fair,Quantile,Mape,Poisson,Gamma,Tweedie}` | ✅ (with `alpha`/`fair_c`/`poisson_max_delta_step`/`tweedie_variance_power` params) |
| binary | `binary.rs` | ✅ |
| multiclass softmax / ova | `multiclass.rs` | ✅ |
| cross_entropy / cross_entropy_lambda | `xentropy.rs` `XentropyKind::{CrossEntropy,CrossEntropyLambda}` | ✅ |
| lambdarank / rank_xendcg | `rank.rs` | ✅ |
| custom | `custom.rs` | ✅ |
| percentile helpers (`PercentileFun`/`WeightedPercentileFun`) | `percentile.rs` | ✅ (supports L1/quantile) |

No objective gap found.

### 2.2 Metrics — `crates/lgbm-metric/` — COMPLETE (one sub-feature gap)
`[VERIFIED: LOCAL grep *.rs]`

Implemented: l2, rmse, l1, quantile, huber, fair, poisson, mape, gamma, tweedie
(`regression.rs`); binary_logloss, binary_error, auc, average_precision
(`binary.rs`); multi_logloss/multi_error + `auc_mu` (`multiclass.rs`); ndcg, map
(`rank.rs`, via `dcg_calculator.rs`); xentropy/xentlambda + kldiv
(`xentropy.rs`).

- **Minor sub-gap:** `auc_mu_weights` is **not implemented** — `auc_mu` uses the
  default ones-off-diagonal matrix; a user-supplied `auc_mu_weights` matrix is
  ignored (`multiclass.rs:117` "auc_mu_weights is not yet…"). `[VERIFIED: LOCAL]`
  Small, self-contained, but low-value (rare param).

### 2.3 Boosting strategies — `crates/lgbm-boosting/` — COMPLETE
`[VERIFIED: LOCAL grep gbdt.rs, sample_strategy.rs]`
GBDT, DART, RF (`BoostingVariant`), GOSS + bagging sample strategies
(`sample_strategy.rs` `GOSSStrategy`/`BaggingSampleStrategy`, ported verbatim from
`goss.hpp`), early stopping (`early_stopping.rs`). No gap.

### 2.4 Tree learner — `crates/lgbm-treelearner/` — serial COMPLETE; parallel out-of-scope
`[VERIFIED: LOCAL ls]` `SerialTreeLearner` (`learner.rs`, ~5100 lines, bit-exact vs
4.6), monotone constraints, CEGB (`cost_effective_gradient_boosting.rs`), forced
splits (`forced_splits.rs`), col sampler, gradient discretizer (quantized),
**linear tree (`linear.rs`, wired)**. Feature/data/voting **parallel** learners are
**out of scope (locked, PROJECT.md — single-node only)**.

### 2.5 Dataset / binning — `crates/lgbm-dataset/` — COMPLETE (CPU)
BinMapper, FeatureGroup, EFB, categorical folding, missing-value handling on CPU.
Categorical **GPU** kernels are stubbed (see G3, §4).

### 2.6 Prediction — `crates/lgbm-model/predict.rs` — COMPLETE
`[VERIFIED: LOCAL grep predict.rs, booster.rs]` Raw score, normal (transformed),
**leaf index** (`predict_leaf_index_{mat,csr,csc}`), **SHAP/contrib**
(`predict_contrib`, TreeSHAP, mirrors `GBDT::PredictContrib`), and **prediction
early stopping** (`pred_early_stop_margin`). Contrib is correctly *unsupported* for
`linear_tree=true` (matches C++, which also never wires PredictContrib for linear
trees, `predictor.hpp:89-90`). No gap.

### 2.7 Model I/O — `crates/lgbm-model/` — TEXT complete, **JSON absent**, **codegen absent**
- Text save/load: `model_text.rs`, `tree.rs::{to_string,from_str}` with `%.17g`/`%g`
  formatters (`format.rs`). Round-trip + linear-tree blocks supported. ✅
- **JSON dump: MISSING** (G2 — §4). `[VERIFIED: LOCAL grep — zero hits for
  to_json/dump_model/NodeToJSON in lgbm-model & facade; only "json" hit is
  `forced_splits_filename("forced.json")`]` — **contradicts PROJECT.md's Validated
  claim of "text & JSON serialization"; JSON is text-only in reality.**
- **If-else codegen: MISSING** (G1 — §4).
- Refit / continued training: `Booster::{refit,refit_data,model_from_string}`
  present. ✅ (semantics not re-verified vs C++ this session.)

### 2.8 Config / params & the "unimplemented raises" gate
`[VERIFIED: LOCAL grep scope.rs, params.rs]`
- `IN_SCOPE_PARAMS` (`scope.rs:33`) now includes `linear_tree`, `linear_lambda`,
  `use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`,
  `stochastic_rounding`, `path_smooth`, `max_delta_step`, `convert_model`,
  `convert_model_language`, monotone/CEGB/forced-split params.
- `OUT_OF_SCOPE_PARAMS` (`scope.rs:179`) now contains **only** distributed
  (`num_machines`, `local_listen_port`, `time_out`, `machine_list_filename`,
  `machines`) and GPU/OpenCL (`num_gpu`, `gpu_platform_id`, `gpu_device_id`,
  `gpu_use_dp`). The quantized-grad and linear-tree groups were **removed** since the
  older research — confirming those features graduated to in-scope.
- Python `reject_unimplemented` (`params.rs:150`) raises `ValueError` only for
  `OUT_OF_SCOPE_PARAMS`; `build_config_rejects_unimplemented` test at `params.rs:293`.

### 2.9 Python bindings — `crates/lgbm-python/` — present, mirrors official API
`booster.rs`, `dataset.rs`, `params.rs`, `callbacks.rs`, `marshal.rs`. Note: this
crate historically fails to link in some sandboxes (`python3.14`); Python-level
validation of new features may not run locally.

---

## 3. In-flight work that is NOT a functional gap (do not confuse)

`[VERIFIED: LOCAL git log; .planning artifacts]`

- **Device-Resident Grow Loop (DRGL) T-01..T-13** (`27665a4`…`7672d0e`) — a
  **performance** campaign (deferred-sync grow loop). **T-11 verdict: byte-identical
  but ~1.20× SLOWER on P100 → shelved.** `.planning/plans/device-resident-grow-loop/`.
- **Dead-toggle refactor** (`refactor/remove-dead-toggles` branch) — removing dead
  A/B algorithm toggles; keeps profiling toggles. Performance/hygiene, not features.
- **`on_device_default()` = false** (locked decision) — fully GPU-resident path is
  1.12–2.2× slower; opt-in only. Out of scope as a parity gap.

These consume the recent git history but produce **zero** new C++-parity surface.

---

## 4. Prioritized Gap List (the actual deliverable)

Ranked by self-containment, testability against the 4.6 oracle, low numerical risk,
and independence from the shelved perf campaign. **These four are already specced in
`.planning/plans/parity-gap-closure/SPEC.md` (SPEC-G1/G2/G4/G5) — that plan is
drafted but UNEXECUTED on `main`.** This research confirms they remain the right set.

### G2 — Model JSON dump (`Booster::dump_model`) — **RECOMMEND #1**
- **C++ ref:** `GBDT::DumpModel`, `Tree::ToJSON`/`NodeToJSON` (`src/io/tree.cpp`,
  `gbdt_model_text.cpp`) `[INFERRED: SPEC-G2 doc]`.
- **Rust status:** **ABSENT.** No `json.rs`, no `to_json`/`dump_model`. `[VERIFIED: LOCAL]`
- **Size/complexity:** Small–Medium. Pure serialization; reuses `format::{format_g17,
  format_g6}` (`format.rs:43,52`), `Tree` node arrays (`tree.rs`), `GbdtModel`
  ensemble fields, and `model_text::save` field-presence rules as the layout template.
- **Prereqs:** none new (locked DEC-1: hand-emit with `%g`, **no `serde_json`**).
- **Numerical fidelity:** floats emitted via `%g` byte-exact vs C++; the JSON is a
  presentation format, not a training path — no f32/f64 accumulation risk. Test =
  byte-exact string compare vs `lightgbm==4.6.0` `.dump_model()`.
- **API:** `impl Booster { pub fn dump_model(&self) -> String }` mirroring
  `model_to_string` (`booster.rs:728`).

### G1 — If-else C++ codegen (`Booster::model_to_cpp`) — **RECOMMEND #2**
- **C++ ref:** `GBDT::ModelToIfElse`/`SaveModelToIfElse`, `Tree::NodeToIfElse`
  (`gbdt_model_text.cpp`, `src/io/tree.cpp`) `[INFERRED: SPEC-G1 doc]`.
- **Rust status:** **ABSENT.** `Config.convert_model` (`config/mod.rs:262`, default
  `"gbdt_prediction.cpp"`) and `convert_model_language` are **parsed but inert** — a
  silent no-op. No `model_to_cpp`/`to_if_else` anywhere. `[VERIFIED: LOCAL grep]`
- **Size/complexity:** Small–Medium. Shares the tree-walk / `decision_type` decode /
  `%g` substrate with G2 (do G2 first, reuse helpers).
- **Prereqs:** locked DEC-2 — public entry `Booster::model_to_cpp() -> String`, an
  in-memory method (no file side effect in v1). No new crate.
- **Numerical fidelity:** codegen emits thresholds/leaf values via `%g`; correctness
  = generated branch structure + comparison direction identical to `Tree::predict`
  (`tree.rs:269`). Test = byte-exact vs `convert_model` golden.

### G4 — `na_as_missing` forward routing, serial host path — **RECOMMEND #3**
- **C++ ref:** `serial_tree_learner.cpp`, `feature_histogram.hpp`, dataset bin
  handling `[INFERRED: SPEC-G4 doc]`.
- **Rust status:** **Deferred behind a typed error** at `learner.rs:1113-1121`
  (`"na_as_missing feature (num_bin>2 && missing_type==NaN) is deferred"`). A model
  with a genuine-NaN `use_missing=true` feature **fails to train**. `[VERIFIED: LOCAL]`
- **Size/complexity:** Medium, and it touches the **hot serial histogram/partition
  path** — higher risk than G1/G2. Requires: (a) NA rows accumulate into the missing
  bin during histogram build, (b) NA routed down the split's default branch during
  partition, (c) remove the gate.
- **Prereqs:** **MUST check out `LightGBM/` 4.6 first** to fix the two `TBD`s
  (missing-bin index convention, default-direction rule). Do NOT infer.
- **Numerical fidelity:** the CPU f64-fold path must stay **bit-exact** — keep NA
  accumulation order identical to the non-NA path. Test = bin-for-bin histogram +
  prediction parity vs a NaN-bearing 4.6 golden, under `LGBM_CUDA_ON_DEVICE=0`.
- **Note:** the **GPU-resident** NA scan (`histogram.rs:2482,2778`, `partition.rs:9`)
  stays out of scope — it is coupled to the shelved on-device grow loop.

### G5 — Split-kernel gain params: `path_smooth`, `max_delta_step`, feature `penalty` — **RECOMMEND #4**
- **C++ ref:** `feature_histogram.hpp` `GetLeafGain`/`CalculateSplittedLeafOutput`;
  `output->gain *= penalty` `[INFERRED: SPEC-G5 doc + split.rs:442]`.
- **Rust status:** **Gated.** `find_best_split_cpu` (`split.rs`) returns
  `ComputeError::Runtime` when `cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0`
  (`split.rs:551-553`); feature `penalty` is hard-coded `1.0` (`split.rs:442-443`).
  `GainConfig` already **carries** `max_delta_step`/`path_smooth` plumbed from
  `Config` — they're just rejected downstream. `path_smooth` is IN_SCOPE, so a user
  setting it hits the rejection. `[VERIFIED: LOCAL grep split.rs]`
- **Size/complexity:** Medium, touches the **hot split-gain kernel**. Three
  mutually-independent sub-tasks (penalty / max_delta_step / path_smooth).
- **Prereqs:** check out `LightGBM/` for the exact smoothing/clamp/penalty formulas;
  resolve OQ (per-feature penalty source: CEGB `meta_->penalty` vs config; whether
  parent output is available at the `find_best_split_cpu` call site for path_smooth).
- **Numerical fidelity:** must be CPU f64-fold **bit-exact** vs three single-param 4.6
  goldens.
- **Scheduling:** **do NOT run G5 concurrently with G4** — both touch hot
  histogram/split kernels; isolate parity regressions (SPEC §7 note).

### G3 — Categorical-feature GPU kernels — audit-list, later phase
- **C++ ref:** `src/treelearner/{ocl,cuda}/*` categorical variants.
- **Rust status:** **Stubbed.** `column_data.rs:28` (`TODO categorical bitset meta`),
  `best_split.rs` `_GlobalMemory` categorical variants "allocated but unused"
  `[INFERRED: prior research §4 + cubecl_kernel_gaps.md]`. CPU categorical path IS
  complete (`feature_histogram_categorical.rs`).
- **Why not next:** largest/riskiest, needs new `#[cube]` kernel math + GPU-specific
  parity infra the other four don't. Recommend a dedicated later phase.

### Non-gaps confirmed out of scope (locked, PROJECT.md — do NOT scope)
C-API (`LGBM_*`), distributed/MPI/socket networking + parallel tree learners,
OpenCL `gpu` device-type knobs (superseded by CubeCL), fully GPU-resident grow loop.

---

## 5. Test / Oracle Infrastructure (what an implementer must run)

`[VERIFIED: LOCAL codebase/TESTING.md; ls oracle-harness/tests]`

- **Oracle:** real `lightgbm==4.6.0` (install via uv wheel, per memory
  `cpp-linear-tree-oracle.md`). Goldens captured with
  `deterministic=true force_row_wise=true num_threads=1`, documented seeds (e.g.
  `BOOSTING_ORACLE_SEED = 0x60057000`), recorded in `REFERENCE_MANIFEST.md`.
- **Anchor & tolerance:** `cubecl-cpu` f64-fold path is the **bit-exact** hard merge
  gate (integers/bin-indices/RNG compared exact); ROCm f32 held to
  `ORACLE_TOL = 1e-6` (`oracle-harness/src/comparator.rs`). Comparator returns
  first-divergence index (`Mismatch` enum), not aggregate diff.
- **Parity test idiom:** `oracle-harness/tests/*_parity.rs` load a committed golden
  (SKIP-gracefully when absent — `predict_parity.rs:read_golden`), drive the real
  Rust path, assert via `compare_within`/`Mismatch`. New goldens go under
  `oracle-harness/tests/fixtures/<area>/` + documented in `REFERENCE_MANIFEST.md`.
- **Existing suites** (~38 files): `predict_parity`, `learner_parity`,
  `boosting_parity`, `metric_parity`, `objective_parity_*`, `quantized_parity`,
  `linear_parity`, `rank_parity`, `config_drift`, `best_split_parity`,
  `partition_parity`, ROCm-gated `on_device_*` (require `--features rocm`).
- **Commands:**
  ```bash
  cargo test --workspace                                   # CPU default features
  cargo test -p oracle-harness                             # parity suite
  cargo test -p lgbm-model                                 # G1/G2 unit specs
  LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner     # G4 unit (avoid on-device regression mask)
  cargo test -p lgbm-compute                               # G5 split-kernel unit
  # per-gap parity tests to ADD (per PLAN.md):
  cargo test -p oracle-harness --test json_dump_parity        # G2
  cargo test -p oracle-harness --test ifelse_codegen_parity   # G1
  LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test na_missing_parity   # G4
  LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test gain_params_parity  # G5
  cargo test -p oracle-harness --test config_drift         # ONLY works with LightGBM/ checked out
  ```
- **AGENTS.md rule:** confirm dependencies first; record dependency confirmation in
  the commit message. No new external crate is required for G1/G2/G4/G5 (DEC-1: no
  `serde_json`).

---

## 6. Planning Guidance & Open Questions

- **Do G2 → G1 first** (self-contained, additive, low numerical risk, share `%g`
  tree-walk substrate). Then **G4 and G5 separately** (hot path, need `LightGBM/`
  checkout, must not be scheduled concurrently).
- **Reconcile PROJECT.md:** its "Validated: text & JSON serialization" line is
  **inaccurate** — JSON is unimplemented. Update when G2 lands.
- **Blocking prereq for G4/G5:** `LightGBM/` 4.6 must be checked out to verify the
  NA routing rule, missing-bin index, and `path_smooth`/`max_delta_step`/`penalty`
  formulas before coding. `config_drift.rs` should be run first as a mechanical
  "which params are unaccounted for" check.
- **Open questions (from SPEC §9, still open):** (OQ-1) per-feature `penalty` source
  for G5-1 (CEGB `meta_->penalty` vs config); (OQ-2) is the parent output available
  at the `find_best_split_cpu` call site for `path_smooth`?; whether the Python
  binding should expose `dump_model`/`model_to_cpp` (out of v1 scope unless asked).
- **`auc_mu_weights`** (§2.2) is a tiny optional extra gap — mention only.

---

## 7. Sources

- **Project docs (LOCAL):** `.planning/PROJECT.md`, `.planning/STATE.md`,
  `.planning/plans/cpp-feature-parity/research.md` (stale on linear-tree/quantized),
  `.planning/plans/parity-gap-closure/{SPEC,PLAN,research,SOURCES}.md`,
  `.planning/plans/device-resident-grow-loop/*`, `.planning/codebase/TESTING.md`,
  `CLAUDE.md`, `AGENTS.md`, memory index.
- **CODEGRAPH:** `.codegraph/` index present; used to confirm crate/symbol layout.
- **LOCAL verification this session:** `git log/branch`; `ls crates/*/src`;
  `grep` sweeps for objective/metric variants, `to_json`/`dump_model`,
  `fit_linear_leaves`/`add_linear_tree_train_path`, `na_as_missing`,
  `path_smooth`/`max_delta_step`/`penalty`, `use_quantized_grad`, scope arrays,
  `convert_model`, prediction modes; `sed` of `gbdt.rs:720-780`, `scope.rs:179-210`.
- **Context7 / Web:** not needed — no new-library decision arose; all findings from
  local repo evidence.

---

## 8. Confidence Assessment

**HIGH** — directly verified this session:
- Linear-tree and quantized-grad are DONE on `main` (production call sites +
  IN_SCOPE params).
- JSON dump ABSENT; if-else codegen ABSENT; `na_as_missing` deferred (typed error
  `learner.rs:1113`); split gain params gated (`split.rs:553`).
- Objectives/metrics/prediction-modes/boosting-strategies complete.
- `OUT_OF_SCOPE_PARAMS` now only distributed + GPU/OpenCL.
- Test/oracle infrastructure and commands.

**MEDIUM** — inferred from Rust doc-comments citing C++ (not re-read this session):
- Exact C++ algorithm details for G4 (NA routing) and G5 (gain formulas) — **must be
  re-verified against a checked-out `LightGBM/` before implementation.**
- Categorical-GPU-kernel stub status (carried from prior research + kernel-gaps doc).

**LOW** — not verified this session:
- Refit/continued-training semantics vs C++ (present, not re-checked).
- Whether Python binding links in the target execution environment.
- `auc_mu_weights` is the only metric sub-gap (didn't exhaustively diff every C++
  metric param).
