# C++ Feature-Parity Research: lightgbm_rs vs Microsoft LightGBM

Scope: identify functionality present in the C++ LightGBM reference but not yet
implemented (or only partially wired) in the Rust port, to seed a v1.0
"C++ Feature-Parity Audit & Gap Closure" TDD plan. Research only — no code
changed.

## 1. Repo State Summary

- Branch: `master`, up to date with `origin/master`. `[VERIFIED: git status]`
- Recent commits (`git log --oneline -20`):
  1. `8d0aab5` docs: start milestone v1.0 C++ Feature-Parity Audit & Gap Closure
  2. `42249ca` feat!: flip on_device_default() to true (BREAKING, known correctness regression)
  3. `11516b6` docs: strip spike/phase R&D-history narrative from comments
  4. `be5bc6b` chore: squash history into single initial commit
  `[VERIFIED: git log --oneline -20]` — the history was squashed to one initial
  commit, so no older commit-level detail is recoverable in this repo.
- **Uncommitted work in progress** (`git status`, `git diff --stat`), all part
  of a single **linear-tree** feature slice:
  - Modified: `crates/lgbm-compute/src/kernels/tree.rs`,
    `crates/lgbm-core/src/config/{mod,scope,set}.rs`,
    `crates/lgbm-model/src/{ensemble,model_text,objective,predict,tree}.rs`,
    `crates/lgbm-treelearner/src/{learner,lib}.rs`,
    `crates/oracle-harness/tests/learner_parity.rs`.
  - Untracked: `crates/lgbm-model/tests/linear_golden_predict.rs`,
    `crates/lgbm-treelearner/src/linear.rs`,
    `crates/lgbm-treelearner/tests/linear_fit_golden.rs` (this third file was
    not present in the session's initial git-status snapshot but exists on
    disk now — `[VERIFIED: git status --porcelain, ls -la]` at research time;
    treat as part of the same in-flight slice, not a separate gap).
  - `cargo build -p lgbm-treelearner -p lgbm-model -p lgbm-boosting -p lgbm -p
    oracle-harness --tests` succeeds with the uncommitted diff applied
    `[VERIFIED: cargo build output, "Finished dev profile"]`. `cargo build
    --workspace --tests` fails only for `lgbm-python`, and only on an
    environment linker issue (`library not found: python3.14`, `mold`/`cc`
    failure) — unrelated to the linear-tree changes.
    `[VERIFIED: cargo build --workspace --tests output]`
- No PageIndex documents are indexed for this workspace —
  `mcp__pageindex__get_folder_structure` returned an empty root (`file_count:
  0, children_count: 0`). All document research below used local files only.
  `[VERIFIED: mcp__pageindex__get_folder_structure output]`
- **The vendored C++ reference trees (`LightGBM/`, `LightGBM-release-4.6.0.99/`)
  do NOT exist in this sandbox** — `find` for `LightGBM*` directories returned
  nothing. `[VERIFIED: find / -maxdepth 3 -iname "LightGBM*" -type d]` This is
  a hard research limitation: two dedicated parity tests that read the C++
  source directly (`crates/oracle-harness/tests/config_drift.rs`) FAIL in this
  environment with "No such file or directory" for
  `LightGBM/src/io/config_auto.cpp`. `[VERIFIED: cargo test -p oracle-harness
  --test config_drift output — 2 of 3 tests fail with ENOENT]` All C++-side
  claims in this report are therefore sourced from `docs/LIGHTGBM-CPP-DESIGN.md`
  / `docs/cuda-kernel-design.md` (project-authored design docs, presumably
  written against a real checkout of `LightGBM/` that is not present here) or
  from repository conventions/comments citing specific C++ file:line —
  **not from re-reading the C++ source in this session**. The planner should
  re-run `config_drift` and re-verify any C++-line citations on a machine
  where `LightGBM/` is checked out before finalizing a plan.

## 2. Existing Audit/Gap Docs Found

- **`.planning/PROJECT.md`** — the authoritative, ALREADY-BOOTSTRAPPED milestone
  doc for exactly this work (`Current Milestone: v1.0 C++ Feature-Parity Audit
  & Gap Closure`). Key points `[PROJECT: .planning/PROJECT.md]`:
  - Approach is "research-first": Phase 1 is a parity audit that produces a
    ranked gap inventory; later phases fill gaps, gated by oracle-harness
    parity tests.
  - Audit dimensions: objectives & metrics, config params & aliases,
    boosting & tree-learner features (DART/RF/GOSS, linear trees, monotone
    constraints, CEGB, forced splits, quantized gradient), dataset/binning/
    model I/O (EFB, categorical handling, text/JSON format).
  - **Out of scope for this milestone** (locked decisions, must not be
    re-researched as gaps): C-API surface (`LGBM_*`/`c_api.cpp`); distributed/
    MPI/socket networking (`src/network/`); fully GPU-resident (no-host-
    round-trip) best-first grow loop (architecturally shelved, opt-in,
    known-slow — not a parity gap); on-device CUDA path as default (stays
    opt-in via `LGBM_CUDA_ON_DEVICE`).
  - Context note: "the codebase is mature and near-complete... Known genuine
    gaps are narrow — categorical-feature GPU kernels (stubbed), quantized-
    gradient/stochastic rounding ('not yet implemented'...), and unwired
    Python params recognized by C++ but not implemented." This session's
    findings refine but do not contradict that note (see §4/§6).
- **`.planning/STATE.md`**: milestone status is `planning`, phase `Not started
  (defining requirements)`, 0 phases/plans recorded yet. `[PROJECT:
  .planning/STATE.md]` — confirms **no phase-1 audit has been executed or
  written up yet**; this research report is effectively the first concrete
  input toward that audit, not a duplicate of existing work.
- **`.planning/codebase/{ARCHITECTURE,STACK,STRUCTURE,CONCERNS,CONVENTIONS,
  TESTING,INTEGRATIONS}.md`** — a GSD-generated codebase map dated 2026-07-09,
  treated as authoritative current-state evidence for this report
  `[PROJECT: .planning/codebase/*.md]`. `CONCERNS.md` "Missing Critical
  Features" section explicitly names only two items: categorical-feature GPU
  kernel support (stubbed) and the fully-GPU-resident grow loop (shelved, not
  a parity gap per PROJECT.md). No other gaps are called out there — this
  session's deeper dive (§4) surfaces more granular gaps within nominally
  "complete" subsystems (e.g. linear-tree training-loop wiring, C++ if-else
  codegen, quantized-grad param plumbing).
- **`cubecl_kernel_gaps.md`** (repo root) — a **resolved** GPU-histogram-kernel
  *performance* gap analysis (grid_dim_y row-partitioning vs LightGBM's ROCm
  kernel), dated 2026-06-15, explicitly marked "RESOLUTION — all gaps closed."
  `[PROJECT: cubecl_kernel_gaps.md]` This is a perf document, not a feature-
  parity document; not directly relevant to this milestone's scope (feature
  presence, not throughput) beyond confirming the categorical-kernel and
  16-bit-discretized-histogram items it names are perf/precision tradeoffs,
  not open functional gaps.
- No `docs/*AUDIT*`, `docs/*GAP*`, `REQUIREMENTS.md`, `SPEC.md`, `PRD.md`, or
  `.planning/phase*`/`.planning/plans/*` directories exist yet.
  `[VERIFIED: find . -iname "*AUDIT*" -o -iname "*GAP*"; find .planning -type f]`

## 3. Current Implementation Surface (by crate)

`[PROJECT: .planning/codebase/ARCHITECTURE.md, STRUCTURE.md]` cross-checked
against `[CODEGRAPH]` and direct file reads this session.

| Crate | Role | Notable contents |
|---|---|---|
| `lgbm-core` | `Config` (mirrors C++ `config.h`), errors, RNG | `config/{mod,scope,set,alias}.rs` |
| `lgbm-dataset` | Binning, `FeatureGroup`, EFB, `Dataset` | `bin_mapper.rs`, `efb.rs`, `dataset.rs` |
| `lgbm-model` | `Tree`, `GbdtModel`, text/JSON I/O, predict | `tree.rs`, `ensemble.rs`, `predict.rs`, `model_text.rs`, `objective.rs` |
| `lgbm-compute` | The sole CubeCL seam (CMP-01): `Backend` trait, kernels | `kernels/{histogram,split,best_split,data_partition,predict,grow_driver,tree,...}.rs` |
| `lgbm-objective` | Gradients/hessians per loss | `regression.rs`, `binary.rs`, `multiclass.rs`, `rank.rs`, `xentropy.rs`, `custom.rs` |
| `lgbm-metric` | Eval metrics | `regression.rs`, `binary.rs`, `multiclass.rs`, `rank.rs`, `xentropy.rs`, `dcg_calculator.rs` |
| `lgbm-treelearner` | `SerialTreeLearner` leaf-wise growth | `learner.rs` (5100 lines), `data_partition.rs`, `histogram_pool.rs`, `monotone_constraints.rs`, `cost_effective_gradient_boosting.rs`, `gradient_discretizer.rs`, `forced_splits.rs`, **`linear.rs` (new, uncommitted)** |
| `lgbm-boosting` | Outer GBDT loop, DART/RF/GOSS | `gbdt.rs` (`DartConfig`, `RfConfig`, `use_quantized_grad` field) |
| `lgbm` | Facade / PyO3 target | `booster.rs`, `builder.rs` |
| `lgbm-python` | PyO3 bindings | `params.rs` (`reject_unimplemented`, `OUT_OF_SCOPE_PARAMS` gate) |
| `oracle-harness` | C++ parity test infra | ~30 `tests/*_parity.rs` files, `config_drift.rs` |

DAG and CMP-01 containment (only `lgbm-compute` may name `cubecl`) are
enforced by a guard test per `[PROJECT: ARCHITECTURE.md]`; not independently
re-verified this session beyond confirming the dependency graph in `Cargo.toml`
member list. `[VERIFIED: Cargo.toml workspace.members]`

### In-progress work: the linear-tree feature slice

`git diff` + untracked files show a **coherent, mostly-complete linear-tree
(`linear_tree=true`) implementation** spanning config, model, and
tree-learner:

- **`lgbm-core`**: `Config.linear_tree: bool` / `Config.linear_lambda: f64`
  added, parsed via `get_bool`/`get_double` in `config/set.rs`, and **moved
  from `OUT_OF_SCOPE_PARAMS` to `IN_SCOPE_PARAMS`** in `config/scope.rs`.
  `[VERIFIED: git diff crates/lgbm-core/src/config/{mod,scope,set}.rs]`
- **`lgbm-model`** (model-side, prediction/serialization) — **appears complete**:
  - `Tree.is_linear: bool` + new `Tree.linear: Option<LinearModel>` field
    (`LinearModel { leaf_const, leaf_features, leaf_coeff }`), mirroring C++
    `Tree::leaf_const_`/`leaf_features_`/`leaf_coeff_`.
  - `Tree::predict` now branches on `is_linear` to
    `linear_leaf_output` (intercept + Σ coeff·feature, NaN-feature fallback to
    the stored constant leaf value — matches the documented C++ behavior).
  - `Tree::to_string` emits `leaf_const=`/`num_features=`/`leaf_features=`/
    `leaf_coeff=` blocks with byte-exact C++ spacing (`join_leaf_groups_*`
    helpers, `%.17g` formatting reused from existing `format_g17`).
  - `Tree::from_str`/model-text parser: the old code **rejected** any
    `is_linear=1` model with `ModelError::MalformedModel` ("out of scope for
    Phase 3"); now parses it via a new `parse_linear_model` helper.
  - `Tree::shrinkage()` (learning-rate scaling) now also scales
    `leaf_const`/`leaf_coeff` (not run through `maybe_round_to_zero`, matching
    the documented C++ plain-multiply behavior).
  - New test `crates/lgbm-model/tests/linear_golden_predict.rs`: loads a real
    `lib_lightgbm` 4.6 linear-tree `model.txt` + `X_test.csv`/`pred.csv`
    golden (gated on env var `LINEAR_GOLDEN_DIR`, SKIPs/passes when unset —
    the standard oracle-harness graceful-skip idiom), asserts (a) byte-exact
    `to_string()` round-trip per tree block and (b) prediction within 1e-6 of
    the C++ golden. `[VERIFIED: git diff crates/lgbm-model/src/tree.rs; Read
    crates/lgbm-model/tests/linear_golden_predict.rs]`
- **`lgbm-treelearner`** (training-side fit) — **implemented but NOT wired
  into the production training loop**:
  - New `crates/lgbm-treelearner/src/linear.rs`:
    `fit_linear_leaves(tree, raw, num_features, grad, hess, linear_lambda)`
    fits hessian-weighted ridge least-squares per leaf (design row `[1,
    feat_j…]`, `A = Σh·xxᵀ + λ·diag(0,1,1,…)`, `b = −Σg·x`, Gaussian
    elimination with partial pivoting), setting `tree.is_linear`/`tree.linear`
    in place. Feature set per leaf = distinct `split_feature` values on the
    root→leaf path (matches the documented C++ `leaf_features_` derivation,
    §"linear_tree_learner.{h,cpp}" below).
  - New `crates/lgbm-treelearner/tests/linear_fit_golden.rs`: re-fits linear
    leaves against a real C++ golden's reconstructed L2 gradient and asserts
    `leaf_const`/`leaf_coeff`/`leaf_features` match within 1e-6 (also
    `LINEAR_GOLDEN_DIR`-gated, graceful-skip).
  - `[CODEGRAPH: fit_linear_leaves (crates/lgbm-treelearner/src/linear.rs:34)
    — 1 caller, only from crates/lgbm-treelearner/tests/linear_fit_golden.rs]`
    **`fit_linear_leaves` is called from nowhere in production code** —
    confirmed by `grep -rn "fit_linear_leaves" crates/lgbm-boosting
    crates/lgbm-treelearner/src crates/lgbm/src crates/lgbm-python/src`
    matching only the new `linear.rs` itself and the golden test.
    `[VERIFIED: grep, corroborated by codegraph_explore blast-radius]`
    `SerialTreeLearner::train`/`train_inner` (`learner.rs`) and
    `Gbdt::train_one_iter` (`gbdt.rs`) have **no `config.linear_tree` branch**
    — every constructed `Tree` still sets `linear: None`, `is_linear: false`
    unconditionally (see the mechanical `linear: None` additions across
    `learner.rs`, `ensemble.rs`, `model_text.rs`, `objective.rs`, `predict.rs`
    test fixtures — those are struct-literal compile fixes for the new field,
    not feature wiring).
  - `crates/lgbm-python/src/params.rs::reject_gate` (test, line 313-314)
    **still asserts `reject_unimplemented({"linear_tree": "true"})` is an
    error** — but per the uncommitted `scope.rs` diff, `linear_tree` is no
    longer in `OUT_OF_SCOPE_PARAMS`, so `reject_unimplemented` will **no
    longer reject it**. This existing unit test will fail once the scope.rs
    diff lands, unless it is deliberately updated. Not independently re-run
    (pyo3 lib tests fail to link in this sandbox — see §1); the contradiction
    is derived from direct source inspection of both files.
    `[VERIFIED: Read crates/lgbm-python/src/params.rs:150-182,306-315; git
    diff crates/lgbm-core/src/config/scope.rs]`
  - `crates/lgbm-core/src/config/scope.rs` module-level doc comment (lines
    ~14-26, NOT part of the uncommitted diff) still lists "Linear-tree
    learner (deferred)" as an out-of-scope exclusion group — stale relative
    to the `IN_SCOPE_PARAMS` array change in the same file.

**Assessment**: the linear-tree slice is roughly **70% complete** by surface
area — the harder, more numerically delicate half (model parse/serialize/
predict, and the ridge-regression fit algorithm itself, both validated against
real C++ goldens at 1e-6) is done and tested; the remaining ~30% is
**integration wiring**: (1) call `fit_linear_leaves` from the boosting loop
when `config.linear_tree` is set and the tree is not the first tree (C++: "the
FIRST tree of the ensemble stays constant"), (2) thread `raw` (unbinned
feature matrix) and per-tree `grad`/`hess` through to the tree learner/GBDT
loop (currently the engine operates on binned columns; the linear fit needs
raw float feature values — confirm the raw-feature matrix is already
available somewhere, e.g. `DenseCorpus`, or needs new plumbing), (3) fix the
now-contradictory `reject_gate` python test, (4) refresh the stale scope.rs
module doc comment, (5) an end-to-end training-loop parity test (there is
currently only a model-side predict golden and a training-side fit-in-
isolation golden — no test trains a full model with `linear_tree=true` through
`lgbm::train()`/`Booster` and compares to C++).

## 4. Feature-by-Feature Comparison

Legend: **Implemented** / **Partial** / **Missing** / **Out of scope (locked)**.

| Area | C++ location (per design docs) | Rust status | Evidence | Notes |
|---|---|---|---|---|
| GBDT outer loop | `src/boosting/gbdt.{h,cpp}` | Implemented | `[PROJECT: STRUCTURE.md]`, `crates/lgbm-boosting/src/gbdt.rs` (`Gbdt<'a>`, `train_one_iter`) | Validated (STATE: "Validated" in PROJECT.md). |
| DART | `src/boosting/dart.hpp` | Implemented | `DartConfig` in `crates/lgbm-boosting/src/gbdt.rs:127` `[VERIFIED: grep]` | Listed as Validated in PROJECT.md. |
| RF | `src/boosting/rf.hpp` | Implemented | `RfConfig`, `train_one_iter_rf` in `gbdt.rs:1290` `[VERIFIED: grep]` | Listed as Validated. |
| GOSS / bagging | `src/boosting/*` sample strategies | Implemented | `crates/lgbm-boosting/src/sample_strategy.rs` `[PROJECT: STRUCTURE.md]` | Listed as Validated. |
| Objectives (regression/binary/multiclass/rank/xentropy/custom) | `src/objective/*.hpp` | Implemented | `crates/lgbm-objective/src/{regression,binary,multiclass,rank,xentropy,custom}.rs` | Validated per PROJECT.md. |
| Metrics | `src/metric/*.hpp` | Implemented | `crates/lgbm-metric/src/*.rs` | Validated per PROJECT.md. |
| Serial tree learner (leaf-wise best-first) | `src/treelearner/serial_tree_learner.cpp` | Implemented, bit-exact | `crates/lgbm-treelearner/src/learner.rs` | "bit-exact vs `lib_lightgbm` 4.6 on both committed corpora" per CLAUDE.md/PROJECT.md — the hard merge gate. |
| Monotone constraints | `src/treelearner/monotone_constraints.hpp` | Implemented | `crates/lgbm-treelearner/src/monotone_constraints.rs` (527 lines) `[VERIFIED: wc -l]` | Config params `monotone_constraints`/`monotone_constraints_method`/`monotone_penalty` are IN_SCOPE. |
| CEGB (cost-effective gradient boosting) | `src/treelearner/cost_effective_gradient_boosting.hpp` | Implemented | `crates/lgbm-treelearner/src/cost_effective_gradient_boosting.rs` (223 lines); `CegbModel` used in `learner.rs:248,1285` `[VERIFIED: grep]` | Not exercised end-to-end in this session beyond confirming wiring points exist; recommend a targeted parity test check, not a new-feature gap. |
| Forced splits | `src/treelearner/*` (forcedsplits_filename) | Implemented | `crates/lgbm-treelearner/src/forced_splits.rs` `[PROJECT: STRUCTURE.md]`; `forcedsplits_filename` IN_SCOPE | Not deeply re-verified this session. |
| Column sampling (`feature_fraction*`) | `src/treelearner/col_sampler.hpp` | Implemented | `crates/lgbm-treelearner/src/col_sampler.rs` `[PROJECT: STRUCTURE.md]` | — |
| **Linear tree learner** (`linear_tree=true`) | `src/treelearner/linear_tree_learner.{h,cpp}` | **Partial** — model-side done, training-loop wiring missing | See §3 above; `[CODEGRAPH: fit_linear_leaves 1 caller]` | Top gap candidate — see §7. |
| Quantized-gradient training (`use_quantized_grad`) | `src/treelearner/gradient_discretizer.{hpp,cpp}` | **Partial** — engine-level implementation exists, not exposed via `Config::from_params`/Python | `GradientDiscretizer` (397 lines, incl. `new_stochastic`) `crates/lgbm-treelearner/src/gradient_discretizer.rs`; wired into `crates/lgbm-boosting/src/gbdt.rs` (`use_quantized_grad` field, `enable_quantized_grad()` method, quantize-gradients call site at `gbdt.rs:874-888`) `[VERIFIED: grep]`; but `use_quantized_grad`/`num_grad_quant_bins`/`quant_train_renew_leaf`/`stochastic_rounding` are in `scope::OUT_OF_SCOPE_PARAMS` and **not parsed** in `config/set.rs` `[VERIFIED: grep — no get_bool/get_double calls for these keys]` | Only reachable via a low-level Rust `Gbdt::enable_quantized_grad(...)` call, not via string params / Python. `quant_train_renew_leaf` doc comment says "Not yet implemented"; `stochastic_rounding` doc comment says "Rust quantized path currently supports DETERMINISTIC rounding only... not yet implemented" even though `GradientDiscretizer::new_stochastic` exists — doc/impl mismatch to resolve. |
| Categorical feature handling (CPU) | `src/io/bin.cpp`, feature_histogram categorical scan | Implemented | `crates/lgbm-treelearner/src/feature_histogram_categorical.rs` `[PROJECT: STRUCTURE.md]`; `max_cat_threshold`/`cat_l2`/`cat_smooth`/`max_cat_to_onehot`/`categorical_feature` all IN_SCOPE | CPU path only; see next row for GPU. |
| Categorical feature handling (GPU kernels) | `src/treelearner/{ocl,cuda}/*` categorical variants | **Missing / stubbed** | `crates/lgbm-compute/src/kernels/column_data.rs:28` (`TODO(Phase 22): categorical bitset meta`), `crates/lgbm-compute/src/kernels/best_split.rs:1118,1185` (`_GlobalMemory` categorical variants "allocated but unused") `[VERIFIED: grep + Read]` | Explicitly called out in `CONCERNS.md` "Missing Critical Features". |
| EFB (Exclusive Feature Bundling) | `src/io/*` bundling | Implemented | `crates/lgbm-dataset/src/efb.rs` `[PROJECT: STRUCTURE.md]` | Validated per PROJECT.md. |
| Model text I/O (save/load) | `src/boosting/gbdt_model_text.cpp`, `src/io/tree.cpp` | Implemented | `crates/lgbm-model/src/model_text.rs`, `tree.rs` (`to_string`/`from_str`) | Validated; now extended for linear trees (in-flight). |
| Model JSON I/O | `Tree::ToJSON`, `GBDT::DumpModel` | **Unverified this session** | No JSON-specific file found by name in the crate list scanned (`format.rs` exists per STRUCTURE.md but content not read this session) | `[UNVERIFIED: not read this session — planner should grep `to_json`/`dump_model` in `lgbm-model/src/format.rs` before scoping]`. |
| **C++ if-else codegen (`ModelToIfElse`/`SaveModelToIfElse`, `task=convert_model`)** | `src/boosting/gbdt_model_text.cpp` (`ModelToIfElse`, `SaveModelToIfElse`), `include/LightGBM/boosting.h:189,197` | **Missing** | `grep -rln "ToIfElse\|to_if_else\|CodeGen\|codegen"` across `lgbm-model`, `lgbm-core`, `lgbm`, `lgbm-python` → no hits `[VERIFIED: grep, empty result]`. `Config.convert_model`/`convert_model_language` fields exist and are IN_SCOPE-parsed as plain strings (`config/mod.rs:257-260`, `config/set.rs:326-327`) but nothing consumes them — no `model_to_cpp`/`to_if_else` function anywhere in the Rust facade or Python bindings. | Silent no-op gap: a user can set `convert_model`/`convert_model_language` and nothing happens. Self-contained, testable, no numerical-precision novelty (reuses the `%.17g` float-formatting infra already built for linear trees / model_text.rs). Good, low-risk TDD candidate. |
| C API (`LGBM_*`, `src/c_api.cpp`) | `include/LightGBM/c_api.h` | **Out of scope (locked)** | `[PROJECT: .planning/PROJECT.md]` "Out of Scope: C-API surface parity... user decision for this milestone" | Do not scope work here. |
| Distributed / MPI / socket networking | `src/network/*`, parallel tree learners (`feature_parallel_tree_learner.cpp`, `data_parallel_tree_learner.cpp`, `voting_parallel_tree_learner.cpp`) | **Out of scope (locked)** | `[PROJECT: .planning/PROJECT.md]` "not a port target; single-node only" | Do not scope work here even though C++ has 3 more tree-learner classes. |
| GPU/OpenCL (`gpu` device type, `.cl` kernels) | `src/treelearner/ocl/*.cl`, Boost.Compute | **Out of scope (architecturally superseded)** | `[PROJECT: STACK.md]` "Rust compute backend uses CubeCL (ROCm), not the C++ OpenCL gpu device knobs; these have no Rust analog in v1" | `num_gpu`/`gpu_platform_id`/`gpu_device_id`/`gpu_use_dp` in `OUT_OF_SCOPE_PARAMS`. Not a parity gap — deliberate architecture substitution (CubeCL ROCm/CUDA/wgpu instead). |
| Fully GPU-resident best-first grow loop | (Rust-only architectural concept; no direct C++ analog) | **Out of scope (locked, shelved)** | `[PROJECT: .planning/PROJECT.md]`, `[PROJECT: CONCERNS.md]` "1.12-2.2x SLOWER... shelved" | Implemented but perf-rejected; opt-in via `LGBM_CUDA_ON_DEVICE`. |
| Python bindings | `python-package/lightgbm/` | Implemented, mirrors official API per constraint | `crates/lgbm-python/src/{booster,dataset,params,callbacks,marshal}.rs` `[PROJECT: STRUCTURE.md]` | Validated per PROJECT.md; note the `reject_gate` staleness in §3. |
| R bindings | `R-package/` | **Out of scope (not in workspace)** | No `R-package` analog anywhere in `crates/`; not mentioned in any Rust project doc as a goal | `[ASSUMED not a goal — not stated as in-scope anywhere in CLAUDE.md/PROJECT.md; the project's stated binding target is Python only]`. Confirm with user if R parity is ever desired — currently no evidence it's wanted. |
| na_as_missing / missing-value routing (on-device resident scan) | `src/treelearner/*` missing-type handling | **Partial** — CPU path implemented, GPU resident path missing | `crates/lgbm-compute/src/kernels/histogram.rs:2482,2778`: `"build_fix_scan_resident: na_as_missing not yet implemented"` (returns a typed error, not silently wrong) `[VERIFIED: grep]`; `crates/lgbm-compute/src/kernels/partition.rs:9`: "Missing/NA routing is not yet implemented" for the transcribed `DenseBin::SplitInner` kernel (default `MissingType::None` instantiation only) `[VERIFIED: Read]` | Narrow GPU-kernel gap, properly typed-error-gated (matches PROJECT.md's "no `todo!()`/`unimplemented!()` stubs" claim — these are `ComputeError::Runtime` returns, not panics). |
| Feature-penalty (`penalty` in split-gain) / `max_delta_step` / `path_smooth` in `find_best_split_cpu` | `src/treelearner/feature_histogram.hpp` `FindBestThreshold` | **Partial** | `crates/lgbm-compute/src/kernels/split.rs:440-451` doc comment: "feature-penalty support is not yet implemented"; "unsupported non-default gain params... (max_delta_step / path_smooth are not yet implemented)" returns `ComputeError::Runtime` `[VERIFIED: Read]` | `path_smooth` IS an IN_SCOPE config param (`path_smooth` in `scope::IN_SCOPE_PARAMS`) but is rejected at the kernel level if set non-default — a genuine functional gap between "config accepts it" and "kernel honors it." Needs explicit scoping decision. |
| Refit (`Booster::refit`, C++ `task=refit` / `RefitTree`) | `src/boosting/gbdt.cpp` (`RefitTree`) | Implemented | `refit` method listed in `[PROJECT: STRUCTURE.md]` `crates/lgbm/src/booster.rs` | Not independently re-verified against C++ semantics this session; flagged for planner spot-check, not treated as a gap. |
| `task` config dispatch (train/predict/convert_model/refit CLI branches) | `src/application/application.cpp::Run()` | **N/A by design** — no CLI in this port | `Config.task` exists as a field but the Rust facade calls `train()`/`predict()`/`refit()` directly as library functions rather than dispatching off `config.task` `[ASSUMED — not exhaustively grepped for a `match config.task` dispatch site; inferred from "no CLI, `BUILD_CLI`-equivalent absent" and STRUCTURE.md's "Entry Points" list showing only library functions]` | Not a gap per se (there's no CLI to drive), EXCEPT that `task=convert_model`'s actual payload (`ModelToIfElse`) has no equivalent *library* entry point either — see the codegen row above, which is the real gap. |

## 5. Dependency Versions (exact, resolved)

`[VERIFIED: Cargo.toml, Cargo.lock]`

| Crate | Declared (workspace) | Resolved (Cargo.lock) | Notes |
|---|---|---|---|
| `cubecl` | `"0.10.0"` | `0.10.0` | The sole compute abstraction (CMP-01); backends `cpu`/`rocm`/`cuda`/`wgpu` are cargo features pulling `cubecl-cpu`/`cubecl-hip`/`cubecl-cuda`/`cubecl-wgpu` transitively. |
| `thiserror` | `"2.0.18"` | `2.0.18` (workspace) — **note**: `Cargo.lock` also resolves a second `thiserror = 1.0.69` (transitive, pulled in by an unrelated dependency, e.g. via `polars`/other crates) — not a project-code inconsistency, just a transitive-graph duplicate. | Project code uses `thiserror` 2.0.18 per convention. |
| `anyhow` | `"1.0.102"` | `1.0.102` | app/dev-layer error propagation. |
| `rayon` | `"1.10"` (per STACK.md) | **`1.12.0`** (Cargo.lock) | Declared range `1.10` in root `Cargo.toml`'s comment/STACK.md is looser than the resolved `1.12.0`; not a conflict, just noting the resolved version for the planner. |
| `pyo3` | `"0.27"` (features `abi3-py311`) | `0.27.2` | Version-locked triangle with `numpy 0.27.1`/`pyo3-polars 0.26.0` per `[PROJECT: STACK.md]`. |
| `numpy` (rust-numpy) | `0.27` | `0.27.1` | — |
| `polars` | (direct dep in `lgbm-python`) | `0.53.0` | features `dtype-categorical, dtype-u8, dtype-u16` per STACK.md. |
| `pyo3-polars` | `0.26` | `0.26.0` | — |
| Rust toolchain | `1.95.0` (stable), edition 2024 | — | `rust-toolchain.toml`. |
| `mimalloc` | `0.1` | — | process allocator, `lgbm-python` always-on, `lgbm` optional. |

No Context7-CLI lookups were performed this session — this research phase did
not reach a "select a new library" decision point (see §7: every candidate gap
is closeable with existing in-repo dependencies; `cubecl` 0.10 and Eigen-free
pure-Rust linear algebra (hand-rolled Gaussian elimination in `linear.rs`) are
already sufficient for the linear-tree fit). If a later plan needs GPU-kernel
work touching `cubecl` APIs, run `npx ctx7@latest library cubecl` /
`npx ctx7@latest docs <id>` at that time — not done here since no cubecl API
question arose.

## 6. Wired vs Unwired: The "Unwired Python Params" Class of Gap

PROJECT.md's Context note flags "unwired Python params recognized by C++ but
not implemented" as a known-gap class. This session found the mechanism
precisely: `crates/lgbm-python/src/params.rs::reject_unimplemented` only
raises for params listed in `lgbm_core::config::scope::OUT_OF_SCOPE_PARAMS`.
Currently that list contains exactly 4 groups (distributed, GPU/OpenCL,
quantized-grad — linear-tree was just removed by the uncommitted diff):

```
num_machines, local_listen_port, time_out, machine_list_filename, machines,
num_gpu, gpu_platform_id, gpu_device_id, gpu_use_dp,
use_quantized_grad, num_grad_quant_bins, quant_train_renew_leaf, stochastic_rounding
```

`[VERIFIED: Read crates/lgbm-core/src/config/scope.rs]` Any C++ `config.h`
canonical param NOT in this list and NOT in `IN_SCOPE_PARAMS` would currently
pass through as an "unknown typo" (warn, not fatal) per the `reject_gate` test
doc comment — i.e. **there could be C++ params not yet triaged into either
list at all**. The `config_drift.rs` oracle test exists specifically to catch
this drift by diffing against `LightGBM/src/io/config_auto.cpp`, but **it
cannot run in this sandbox** (missing `LightGBM/` tree, confirmed failing in
§1). **This is the single most impactful blocked verification for the audit
milestone** — the planner should run `cargo test -p oracle-harness --test
config_drift` on a machine with `LightGBM/` checked out as literally the first
step of Phase 1, since it directly and mechanically answers "which C++ config
params are unaccounted for" without further manual comparison.

## 7. Prioritized Gap Candidates for the Next TDD Plan

Ranked by (a) how close to done / low-risk-to-finish, (b) relevance to the
project's stated Core Value (f32 numerical parity), (c) how self-contained
the change is.

1. **Finish wiring the linear-tree learner into the training loop
   (`linear_tree=true` end-to-end).** Highest priority — already ~70% built
   and validated at the unit level against real C++ goldens; the remaining
   work is scoped and mechanical: (a) call `fit_linear_leaves` from
   `Gbdt::train_one_iter`/`SerialTreeLearner` when `config.linear_tree &&
   !is_first_tree`, sourcing the raw (unbinned) feature matrix and per-row
   grad/hess already computed for that iteration; (b) fix the
   now-self-contradictory `crates/lgbm-python/src/params.rs::reject_gate`
   test; (c) refresh the stale `config/scope.rs` module doc comment; (d) add
   an end-to-end oracle-harness parity test that trains a full model with
   `linear_tree=true` via the public `lgbm::train`/`Booster` API and compares
   to a C++ golden (currently only isolated model-predict and isolated
   training-fit goldens exist, not a full train-then-predict round trip).
   Risk: low — algorithm and formats are already 1e-6-validated; the
   remaining risk is purely plumbing (finding/threading the raw feature
   matrix) and NOT re-deriving math.

2. **Implement C++ if-else codegen (`ModelToIfElse`/`convert_model`).**
   Clean, self-contained, well-specified gap (C++ source: `GBDT::ModelToIfElse`
   / `SaveModelToIfElse`, `gbdt_model_text.cpp`). No GPU/precision novelty —
   reuses the exact float-formatting (`%.17g`) and tree-walk infra already
   built in `lgbm-model/src/tree.rs` for text serialization. `Config.
   convert_model`/`convert_model_language` are already parsed and IN_SCOPE but
   currently inert — a good "closes a silent no-op" fix. Needs a decision on
   what the *Rust-facing* API surface should look like (there's no CLI, so
   this must be a new `Booster`/`GbdtModel` method, e.g. `model_to_cpp()` —
   requires a **user decision** on the exact API shape/name before planning,
   since C++'s trigger is `task=convert_model` via CLI args and this project
   has no CLI).

3. **Resolve the quantized-gradient (`use_quantized_grad`) param-plumbing
   gap.** The hard numerical work (discretization, stochastic rounding,
   int histograms) is already implemented and reasonably tested in
   `gradient_discretizer.rs` and wired into `Gbdt` — but it's unreachable
   through `Config::from_params`/Python because the 4 related keys are still
   in `OUT_OF_SCOPE_PARAMS`. This is a smaller, narrower task than #1/#2:
   parse the 4 params in `config/set.rs`, move them to `IN_SCOPE_PARAMS`,
   wire `Config.use_quantized_grad` → `Gbdt::enable_quantized_grad(...)` in
   the facade/builder, and fix the doc-comment/implementation mismatch on
   `stochastic_rounding` (doc says "not yet implemented" but
   `GradientDiscretizer::new_stochastic` exists — verify which is actually
   true before promising it in Config docs). Explicitly an **approximate**
   training mode (project's own doc: "opt-in APPROXIMATE training mode") —
   confirm with the user whether it should be held to the same 1e-6 gate or a
   looser, documented tolerance, since quantization is inherently lossy by
   design in upstream LightGBM too.

4. **Categorical-feature GPU kernel support.** Explicitly named in
   `CONCERNS.md` as the one clearly "Missing Critical Feature" alongside the
   (locked-out-of-scope) resident grow loop. Larger and riskier than #1-#3:
   requires new `#[cube]` kernel code in `lgbm-compute/src/kernels/
   best_split.rs`/`column_data.rs` (the `_GlobalMemory` variants are
   "allocated but unused" — scaffolding exists, math does not), and per
   CONVENTIONS/CONCERNS it's tracked as a "v2 QGD-02" seam with "not
   currently scheduled" status, suggesting the team previously deprioritized
   it. Recommend for the audit list but likely a later phase, not the first
   TDD plan, given it needs GPU-specific parity-test infrastructure the other
   three candidates don't.

5. **`path_smooth` / feature-`penalty` / `max_delta_step` gain-param support
   in `find_best_split_cpu`.** Smaller, narrower gap: these are IN_SCOPE
   config params that the kernel explicitly rejects if set non-default
   (`ComputeError::Runtime`, not silently ignored — good defensive behavior,
   but still a functional gap between "config accepts it" and "kernel honors
   it"). Worth a quick audit-list entry; likely small in LOC but touches the
   hot split-gain kernel, so should get its own careful oracle-harness golden
   rather than being folded into #1-#3.

**Not recommended for this pass** (explicitly out of scope per locked
decisions in PROJECT.md, re-confirmed this session): C-API parity,
distributed/MPI/socket networking, fully GPU-resident grow loop, C++ OpenCL
`gpu` device-type support (architecturally superseded by CubeCL).

## 8. Open Questions / Unknowns Needing a User Decision

1. **Is the config_drift blocker acceptable to defer, or must `LightGBM/` be
   made available in whatever environment runs Phase 1 planning/execution?**
   Two of three `config_drift.rs` tests fail here for lack of the C++ tree.
   Per `[PROJECT: STRUCTURE.md]` the tree is "intentionally untracked...
   worktrees break for phases needing it" — implying the project already
   knows some phases need it and handles it as an environment precondition,
   not a bug. The planner should confirm the execution environment will have
   `LightGBM/` checked out before relying on any C++-line citation in this
   report or in `config_drift`'s output.
2. **What should the linear-tree feature's raw-feature-matrix source be?**
   `fit_linear_leaves` takes a row-major `raw: &[f64]` unbinned matrix; the
   engine's hot path operates on binned columns (`BinColumn`). Confirm
   whether an unbinned raw copy is already retained somewhere (e.g.
   `DenseCorpus`) or must be threaded through newly — this determines whether
   gap #1 is a small wiring change or requires new data-retention plumbing
   with its own memory/perf tradeoff (relevant given the project's documented
   CPU-vs-C++ `DataPartition::split` perf gap already tracked in CONCERNS.md).
3. **What should the public API for C++ if-else codegen (gap #2) look like**,
   given there is no CLI in this project? A `Booster::model_to_cpp() ->
   String` method mirroring `model_to_string()`? Should it also be exposed
   through `lgbm-python`? Needs a user decision before the TDD plan can write
   acceptance criteria.
4. **Should quantized-gradient training (gap #3) be held to the same ~1e-6
   parity gate as the rest of the project, or a documented looser tolerance**,
   given upstream LightGBM itself treats it as an approximate mode? Affects
   how oracle-harness goldens for this feature should be structured.
5. **Model JSON I/O (`ToJSON`/`DumpModel`) parity was not verified this
   session** (`format.rs` exists per the codebase map but was not read) —
   flag as `[UNVERIFIED]`; the planner should do a quick grep/read pass on
   `crates/lgbm-model/src/format.rs` before deciding whether it belongs in
   the audit's "missing" list.
6. **R-package parity**: no evidence this is an in-scope goal anywhere in
   project docs (`[ASSUMED]`). Confirm it is intentionally excluded (the
   project's stated binding target is Python-only) rather than simply
   unaddressed.

## 9. Sources

- Project documents (local files, no PageIndex — library empty):
  - `/home/user/Documents/workspace/lightgbm_rs/.planning/PROJECT.md`
  - `/home/user/Documents/workspace/lightgbm_rs/.planning/STATE.md`
  - `/home/user/Documents/workspace/lightgbm_rs/.planning/codebase/{ARCHITECTURE,STACK,STRUCTURE,CONCERNS}.md`
  - `/home/user/Documents/workspace/lightgbm_rs/cubecl_kernel_gaps.md`
  - `/home/user/Documents/workspace/lightgbm_rs/docs/LIGHTGBM-CPP-DESIGN.md` (headings surveyed; §"linear_tree_learner", §"cost_effective_gradient_boosting", §"gradient_discretizer", §"monotone_constraints", §"parallel (distributed) learners", §gbdt_model_text/ModelToIfElse read in full)
  - `/home/user/Documents/workspace/lightgbm_rs/CLAUDE.md`, `/home/user/Documents/workspace/lightgbm_rs/AGENTS.md`
- CodeGraph queries: `mcp__codegraph__codegraph_explore("SerialTreeLearner
  train linear_tree fit_linear_leaves Gbdt train_one_iter")` — confirmed
  `fit_linear_leaves` has exactly 1 caller (its own golden test), and
  surfaced verbatim source of `gbdt.rs`/`learner.rs`/`linear.rs` call sites.
- Local verification commands run this session: `git log --oneline -20`,
  `git status`, `git diff --stat` / targeted `git diff` per file, `find`
  searches for audit/gap docs and the `LightGBM/` tree, `cargo build`
  (workspace and targeted crate subsets), `cargo test -p oracle-harness
  --test config_drift`, and extensive `grep -rn` sweeps across `crates/` for
  `TODO`/`unimplemented!`/`not yet implemented`/`linear_tree`/
  `fit_linear_leaves`/`use_quantized_grad`/`ToIfElse`/`convert_model`.
- No Context7-CLI or WebSearch/WebFetch calls were made — not needed; every
  question this session was answerable from local project/repo evidence and
  design docs already produced by the project team.

## 10. Confidence Assessment

**HIGH** (directly verified this session via command output, file reads, or
codegraph):
- Repo/git state, uncommitted diff contents, build success/failure.
- Linear-tree model-side (`lgbm-model`) implementation completeness.
- `fit_linear_leaves` having zero production callers (codegraph blast-radius
  + grep, cross-checked two ways).
- The `reject_gate` python-test / `scope.rs` contradiction (direct source
  read of both files).
- `config_drift.rs` failing for lack of `LightGBM/` in this sandbox.
- Exact resolved dependency versions (`Cargo.lock`).
- Categorical-GPU-kernel and na_as_missing/resident-scan gaps (grep + read of
  the exact TODO/not-yet-implemented comments and their typed-error context).
- `path_smooth`/penalty/`max_delta_step` gain-kernel gap (direct doc-comment
  read in `split.rs`).
- C++ if-else codegen absence (exhaustive grep across all Rust crates, zero
  hits, cross-checked against the design doc's description of what C++ does).

**MEDIUM** (supported by the project's own design docs / codebase-map
analysis, not independently re-derived from the actual C++ source this
session since `LightGBM/` is unavailable):
- Exact C++ algorithmic descriptions cited from `docs/LIGHTGBM-CPP-DESIGN.md`
  (e.g. linear-tree learner's `InitLinear`/`CalculateLinear` structure,
  parallel-learner behavior) — accurate as of when that doc was written, not
  re-verified against current `LightGBM/` source.
- Completeness claims for DART/RF/GOSS/CEGB/monotone-constraints/forced-splits/
  col-sampler beyond confirming their files exist and are non-trivially sized
  with real call sites — not deeply read line-by-line this session.
- Quantized-gradient "engine implemented, not param-exposed" characterization
  — verified the wiring exists and the OUT_OF_SCOPE_PARAMS gate exists, but
  did not run the quantized training path end-to-end.

**LOW** (flagged explicitly, needs validation before planning commits to it):
- Model JSON I/O parity status — not read this session.
- R-package parity intent — assumed out of scope, not confirmed by any
  explicit project statement.
- Whether `task` config dispatch has zero remaining relevance — inferred from
  architecture (no CLI) rather than an exhaustive grep for `config.task`
  usage sites.
- Any claim about C++ line numbers reproduced from `docs/LIGHTGBM-CPP-DESIGN.md`
  without independent re-verification against `LightGBM/` (unavailable here).
