# Parity Gap-Closure Research: lightgbm_rs vs Microsoft LightGBM 4.6

> Milestone: v1.0 "C++ Feature-Parity Audit & Gap Closure". Goal of the parent
> task: identify which C++ LightGBM features/behaviors remain UNIMPLEMENTED or
> STUBBED in the Rust port and gather everything a TDD planner needs to close
> them. **Research only — no production code changed.**
>
> Branch at research time: `feat/linear-tree`. This is a *fresh* audit that
> supersedes `.planning/plans/cpp-feature-parity/research.md` (written on
> `master` before linear-tree was committed): two of that document's top-3 gaps
> (linear-tree wiring, quantized-grad param plumbing) are now **CLOSED** on this
> branch, re-baselining the gap list. See §2.

---

## 1. Research Summary

- **Goal**: enumerate the remaining C++→Rust parity gaps and prioritize them for
  a TDD gap-closure plan, respecting the AGENTS.md rule "confirm dependencies
  first."
- **Recommended first target**: **Model JSON serialization (`GBDT::DumpModel` /
  `Tree::ToJSON`)** — a workspace-wide grep finds **zero** JSON model-dump code,
  yet `.planning/PROJECT.md` lists "text & JSON serialization" as *Validated*.
  This is both a real MISSING feature and a documentation contradiction; it is
  self-contained, reuses the existing `%g` float-formatting infrastructure
  (`lgbm-model/src/format.rs`), and needs no new dependency. `[VERIFIED: LOCAL
  grep across crates/*/src — no to_json/dump_model/serde_json/"tree_info"]`
- **Most important constraints**: every gap-fill needs an oracle-harness parity
  test against a committed `lib_lightgbm` 4.6 golden; CPU f64-fold path is the
  bit-exact hard merge gate, ROCm/CUDA f32 held to ~1e-6. `[PROJECT:
  .planning/PROJECT.md, CLAUDE.md]`
- **Highest-risk findings**:
  1. The read-only C++ reference tree (`LightGBM/`) is **absent** in this
     sandbox, so `config_drift.rs` (the mechanical "which C++ config params are
     unaccounted for" checker) **cannot run** — the single most impactful
     blocked verification. `[VERIFIED: LOCAL ls -d LightGBM → NO LightGBM/ dir]`
  2. `on_device_default()` was flipped to `true` (commit `42249ca`, self-labeled
     "BREAKING, known correctness regression"); 6 `lgbm-treelearner` learner
     tests FAIL by default and only pass under `LGBM_CUDA_ON_DEVICE=0`. Any
     parity work must run tests with that env set, or the regression will mask
     results. `[PROJECT: memory/resident-score-host-update-gotcha.md; git log]`
  3. `stochastic_rounding=true` (a C++ `config.h` **default**) is now reachable
     from `Config::from_params` but has **no C++ oracle golden** (QGP-08 blocked
     on a missing `lightgbm==4.6` install). `[VERIFIED: LOCAL
     .planning/plans/quantized-grad-param-plumbing/SPEC.md §QGP-08]`

---

## 2. What Changed Since the Prior Audit (re-baseline)

The prior research doc (`.planning/plans/cpp-feature-parity/research.md`, written
on `master`) ranked its top gaps as (1) linear-tree wiring, (2) if-else codegen,
(3) quantized-grad param plumbing. On the current `feat/linear-tree` branch:

- **Linear tree (`linear_tree=true`) — NOW CLOSED.** `fit_linear_leaves` is
  called from production at `crates/lgbm-boosting/src/gbdt.rs:1030` and `:1197`;
  raw-feature plumbing threads through `crates/lgbm/src/booster.rs:1010,1060,1209`
  (`with_linear_tree`); row bagging is wired (`add_linear_tree_train_path`,
  `remap_partition_to_full`); end-to-end oracle test exists
  (`crates/oracle-harness/tests/linear_parity.rs` — "model-side + fit + end-to-end"
  per memory). `[VERIFIED: LOCAL grep fit_linear_leaves/linear_tree; memory/cpp-linear-tree-oracle.md]`
- **Quantized-grad param plumbing — EFFECTIVELY CLOSED** (has its own committed
  SPEC/PLAN at `.planning/plans/quantized-grad-param-plumbing/`). All 4 keys are
  now parsed in `config/set.rs:258-263`, and `OUT_OF_SCOPE_PARAMS` no longer
  contains them (now only distributed + GPU/OpenCL groups remain). `[VERIFIED:
  LOCAL grep config/set.rs:258-263, config/scope.rs:179-191]` Residual: QGP-06
  (Python `reject_gate` test) and QGP-08 (stochastic golden) are BLOCKED on
  environment, not code — see §5 gap G6.

So the two remaining *known* items from the prior audit (if-else codegen; the
GPU/kernel PARTIAL items) plus **two newly-surfaced gaps** (JSON model dump;
NA-forward serial branch) form this pass's gap list.

---

## 3. Crate / Workspace Map

`[VERIFIED: LOCAL Cargo.toml workspace.members; ls crates/]`

| Crate | Role | Where new gap-fills land |
|---|---|---|
| `lgbm-core` | `Config` (mirrors C++ `config.h`), errors, RNG, param scope/alias | config parsing, param scope |
| `lgbm-dataset` | Binning (`bin_mapper.rs`), `FeatureGroup`, EFB (`efb.rs`), `Dataset` | binning/categorical data |
| `lgbm-model` | `Tree`, `GbdtModel`, text I/O, predict, `format.rs` (`%g`) | **JSON dump (G2), if-else codegen (G1)** |
| `lgbm-compute` | Sole CubeCL seam (CMP-01); kernels under `src/kernels/` | **categorical GPU (G3), NA-resident (G4), gain-params (G5)** |
| `lgbm-objective` | Gradients/hessians per loss | (complete) |
| `lgbm-metric` | Eval metrics | (complete) |
| `lgbm-treelearner` | `SerialTreeLearner`, `linear.rs`, `gradient_discretizer.rs`, `monotone_constraints.rs`, `cost_effective_gradient_boosting.rs`, `forced_splits.rs` | **NA-forward serial branch (G4)** |
| `lgbm-boosting` | Outer GBDT loop, DART/RF/GOSS, `score_updater.rs` | (linear/quant wiring done) |
| `lgbm` | Facade: `train_raw`/`train`/`predict`/`refit`, `Booster`, `TrainingBuilder` | public API surface for G1/G2 |
| `lgbm-python` | PyO3 bindings; `params.rs::reject_unimplemented` scope gate | G6 residual |
| `oracle-harness` | C++ parity test infra (~35 `tests/*_parity.rs` + `config_drift.rs`) | all new goldens/tests |
| `xtask` | `regen` + oracle-capture subcommands (RNG/goss/dart/rf goldens) | golden regeneration |

- **CMP-01**: only `lgbm-compute` may name `cubecl` types (guard test).
  `[PROJECT: .planning/PROJECT.md Constraints]`
- **CubeCL backends**: `cubecl-cpu` (f64-fold deterministic anchor = hard merge
  gate) and `cubecl-hip`/`cubecl-cuda`/`cubecl-wgpu` (f32, ~1e-6) are cargo
  features. `[PROJECT: CLAUDE.md, .planning/codebase/STACK.md]`

---

## 4. Gap Inventory Table

Legend: **MISSING** / **PARTIAL** / **IMPLEMENTED** / **OUT OF SCOPE (locked)**.
"C++ location" citations derive from project design docs
(`docs/LIGHTGBM-CPP-DESIGN.md`) and code comments — `[UNVERIFIED against C++
source: LightGBM/ tree absent this sandbox]` unless separately labeled.

| # | Feature | C++ location (per design docs) | Rust state | Rust location / absence-evidence | Notes / blockers |
|---|---|---|---|---|---|
| G1 | If-else codegen (`task=convert_model`) | `gbdt_model_text.cpp` `ModelToIfElse`/`SaveModelToIfElse` | **MISSING** | `Config.convert_model`/`convert_model_language` parsed (`config/set.rs:333`, `mod.rs:260-263`) but **no consumer**: grep for `to_if_else`/`ModelToIfElse`/`model_to_cpp` → zero hits in `*/src` `[VERIFIED: LOCAL]` | Silent no-op: user sets `convert_model`, nothing happens. Reuses `%g` infra. Needs a **user decision** on Rust API shape (no CLI). |
| G2 | Model **JSON** serialization | `Tree::ToJSON`, `GBDT::DumpModel` (`gbdt_model_text.cpp`) | **MISSING** | Workspace grep `to_json`/`dump_model`/`serde_json`/`"tree_info"` → **zero hits** `[VERIFIED: LOCAL grep crates/*/src]`. `lgbm-model/src/format.rs` is `%g` float formatting only; `model_text.rs` is text-format only. | **Contradicts** PROJECT.md "Validated: text & JSON serialization." Top candidate — self-contained, `%g` infra exists. |
| G3 | Categorical-feature **GPU** kernels | `src/treelearner/{ocl,cuda}/*` categorical variants | **MISSING / stubbed** | `crates/lgbm-compute/src/kernels/column_data.rs:28` `TODO(Phase 22): categorical bitset meta`; `best_split.rs` `_GlobalMemory` categorical variants allocated-but-unused `[VERIFIED: LOCAL grep]` | Named in `.planning/codebase/CONCERNS.md` as the one "Missing Critical Feature." CPU categorical path IS implemented. Larger/riskier; needs GPU parity infra. |
| G4 | `na_as_missing` routing (GPU-resident + serial forward branch) | `src/treelearner/*` missing-type handling | **PARTIAL** | Typed-error gated (not silent): `histogram.rs:2482,2778` `"build_fix_scan_resident: na_as_missing not yet implemented"`; `learner.rs:1113` serial `"NA_AS_MISSING forward branch not implemented"`; `partition.rs:9` "Missing/NA routing is not yet implemented" `[VERIFIED: LOCAL grep + Read]` | The serial `learner.rs:1113` branch is a **host-path** gap, distinct from the GPU-resident one. CPU default-missing path works. |
| G5 | `path_smooth` / feature-`penalty` / `max_delta_step` in split-gain kernel | `feature_histogram.hpp` `FindBestThreshold` | **PARTIAL** | `crates/lgbm-compute/src/kernels/split.rs:443-452`: `penalty` defaults 1.0 "not yet implemented"; `find_best_split_cpu` returns `ComputeError::Runtime` if `max_delta_step`/`path_smooth` non-default `[VERIFIED: LOCAL Read]` | `path_smooth` IS an IN_SCOPE config param → "config accepts, kernel rejects" gap. Touches the hot split kernel; needs its own careful golden. |
| G6 | `stochastic_rounding=true` oracle coverage + Python gate test | `src/treelearner/gradient_discretizer.*` | **PARTIAL (residual)** | Math implemented + reachable (`config/set.rs:263`); QGP-08 (C++ golden for default-`true` path) **BLOCKED** — no `lightgbm==4.6` install; QGP-06 `reject_gate` test **BLOCKED** — `lgbm-python` fails to link (`python3.14`) `[VERIFIED: LOCAL SPEC.md §QGP-06/08]` | Environment blockers, not code. Also a **pre-existing `linear_tree` `reject_gate` bug** flagged in SPEC §9 Risk 1. |
| — | Linear tree (`linear_tree=true`) | `linear_tree_learner.{h,cpp}` | **IMPLEMENTED** | `gbdt.rs:1030,1197`; `booster.rs:1010-1211`; `oracle-harness/tests/linear_parity.rs` | Closed on this branch (was prior audit's #1). |
| — | Quantized-grad param plumbing | `gradient_discretizer.{hpp,cpp}` | **IMPLEMENTED** | `config/set.rs:258-263`; `scope.rs` (removed from OUT_OF_SCOPE); `oracle-harness/tests/quantized_parity.rs` | Closed except G6 residual. |
| — | Prediction: raw / leaf-index / **SHAP contrib** | `GBDT::Predict*`, `PredictContrib` | **IMPLEMENTED** | `predict.rs:369` leaf-index, `:462-509` `predict_leaf_index_{mat,csr,csc}`, `:608` "TreeSHAP feature-contribution (PRD-04)"; config flags `mod.rs:242-247` | Contradicts a prior "unverified" flag — SHAP IS present. |
| — | GBDT/DART/RF/GOSS, objectives, metrics, EFB, monotone, CEGB, forced splits, col-sampler, model **text** I/O, refit | (various) | **IMPLEMENTED** | per crate map §3; `[PROJECT: PROJECT.md "Validated"]` | Not re-derived line-by-line this session. |
| — | C-API (`LGBM_*`) | `c_api.cpp` | **OUT OF SCOPE (locked)** | `[PROJECT: PROJECT.md Out of Scope]` | Do not scope. |
| — | Distributed / MPI / socket; parallel tree learners | `src/network/*`, `*_parallel_tree_learner.cpp` | **OUT OF SCOPE (locked)** | `[PROJECT: PROJECT.md]` | Single-node only. |
| — | C++ OpenCL `gpu` device knobs; fully GPU-resident grow loop | `ocl/*.cl`; (Rust-only concept) | **OUT OF SCOPE (locked)** | `[PROJECT: PROJECT.md, STACK.md]` | CubeCL supersedes OpenCL; resident loop shelved (perf). |

---

## 5. Per-Gap Detail (top gaps)

Ordered by recommended priority: (a) self-contained + low-risk, (b) relevance to
the f32-parity Core Value, (c) closeable with existing dependencies (AGENTS.md
"confirm dependencies first").

### G2 — Model JSON serialization (`DumpModel` / `Tree::ToJSON`) — RECOMMENDED FIRST

- **What C++ does**: `GBDT::DumpModel` emits a JSON document (`"name"`,
  `"version"`, `"num_class"`, `"feature_names"`, `"tree_info"` array with
  per-tree `"tree_structure"` recursion of `split_index`/`split_feature`/
  `threshold`/`decision_type`/`left_child`/`right_child`/`leaf_value`/…), via
  `Tree::ToJSON`. `[UNVERIFIED against C++ source; per design docs]`
- **What Rust lacks**: no JSON emitter anywhere. `format.rs` (`%g` formatters
  `format_g17`/`format_g6`) and `model_text.rs` (text format) exist and are the
  reusable substrate. `[VERIFIED: LOCAL grep; Read format.rs:1-40]`
- **Files/symbols to touch**: `crates/lgbm-model/src/` (new `json.rs` or extend
  `model_text.rs`); a public `Booster::dump_model() -> String` (mirroring the
  existing `model_to_string()`) in `crates/lgbm/src/booster.rs`; optionally
  expose in `crates/lgbm-python/src/booster.rs`.
- **Dependencies (confirm first)**: JSON can be hand-emitted (byte-exact control,
  like `format.rs`) OR use `serde_json` — but **`serde_json` is NOT currently a
  workspace dependency** `[VERIFIED: LOCAL grep serde_json → zero hits]`, so
  adding it needs a decision. Hand-emission avoids a new dep and guarantees
  `%g` byte-parity; recommended.
- **How parity is tested**: generate a `DumpModel` JSON golden from
  `lightgbm==4.6` (uv wheel, per memory), commit under
  `oracle-harness/tests/fixtures/`, assert byte-exact (or structural + `%g`
  field) equality in a new `oracle-harness/tests/*_parity.rs`.
- **Prerequisite decision**: is JSON dump actually in v1 scope, given PROJECT.md
  already (incorrectly) marks it Validated? Confirm with user — this may be why
  it was overlooked.

### G1 — If-else codegen (`convert_model` / `ModelToIfElse`)

- **What C++ does**: `SaveModelToIfElse`/`ModelToIfElse` emit a C++ source file
  of nested if-else predicting the ensemble; triggered by `task=convert_model`
  (`convert_model_language="cpp"`, output path `convert_model`). `[UNVERIFIED
  against C++ source]`
- **What Rust lacks**: `Config.convert_model`/`convert_model_language` are parsed
  (`config/set.rs:333`, `alias.rs:160` maps `convert_model_file`) but **inert** —
  no function consumes them. `[VERIFIED: LOCAL grep — zero to_if_else/ModelToIfElse]`
- **Files/symbols to touch**: new emitter in `crates/lgbm-model/src/` (reuses
  `%g` + tree-walk from `tree.rs`/`predict.rs`); new public `Booster` method.
- **Dependencies**: none new — pure string generation over existing `Tree`.
- **Parity test**: golden `.cpp` from `lightgbm==4.6` `convert_model`; byte-exact
  compare. Lower value than G2 (niche feature).
- **Prerequisite decision (USER)**: there is no CLI in this port, so the trigger
  and API shape must be decided (`Booster::model_to_cpp() -> String`?). Blocks
  writing acceptance criteria.

### G4 — `na_as_missing` forward routing (serial host branch + GPU-resident)

- **What C++ does**: when a numerical feature has `missing_type==NaN` and
  `num_bin>2`, NA values route down a dedicated "forward"/"default" branch during
  histogram build and split application. `[UNVERIFIED against C++ source]`
- **What Rust lacks**: two distinct sub-gaps, both **typed-error-gated** (not
  silently wrong — good): (i) **serial host** path `learner.rs:1113` rejects
  `na_as_missing` features outright; (ii) **GPU-resident** scan `histogram.rs:2482,2778`
  and partition `partition.rs:9` not implemented. `[VERIFIED: LOCAL Read]`
- **Files/symbols**: `crates/lgbm-treelearner/src/learner.rs` (serial),
  `crates/lgbm-compute/src/kernels/{histogram,partition}.rs` (resident).
- **Dependencies**: none new; touches hot histogram/partition kernels.
- **Parity test**: corpus with genuine NaN features + `use_missing=true`; compare
  bin histograms and predictions to `lightgbm==4.6` golden. CPU f64-fold must be
  bit-exact.
- **Note**: the serial branch (i) is the more broadly-impactful, backend-agnostic
  gap; the resident (ii) is coupled to the (locked-out-of-scope-by-default)
  on-device path.

### G5 — `path_smooth` / feature-`penalty` / `max_delta_step` in split kernel

- **What Rust lacks**: `find_best_split_cpu` (`split.rs:443-452`) hard-codes
  `penalty=1.0` and returns `ComputeError::Runtime` if `max_delta_step` or
  `path_smooth` is non-default. `path_smooth` is an **IN_SCOPE** config param —
  so `Config` accepts it but the kernel rejects it. `[VERIFIED: LOCAL Read
  split.rs:435-455]`
- **Files/symbols**: `crates/lgbm-compute/src/kernels/split.rs`
  (`find_best_split_cpu` and its resident sibling), + config threading.
- **Dependencies**: none new; hot split-gain path — needs a dedicated golden.
- **Parity test**: golden trained with each of `path_smooth`/`max_delta_step`
  set; bit-exact split-gain + threshold comparison.

### G3 — Categorical-feature GPU kernels

- **What Rust lacks**: GPU (`cubecl-hip`/`cuda`) categorical split-finding —
  scaffolding (`_GlobalMemory` variants) allocated but math absent; categorical
  bitset meta TODO (`column_data.rs:28`). CPU categorical path is complete.
  `[VERIFIED: LOCAL grep best_split.rs/column_data.rs; CONCERNS.md]`
- **Files/symbols**: `crates/lgbm-compute/src/kernels/{best_split,column_data}.rs`
  — new `#[cube]` kernels.
- **Dependencies**: `cubecl` 0.10 kernel APIs — **run `npx ctx7@latest library
  cubecl` / `docs` before implementing** (no cubecl API question arose this
  session, so not fetched). Needs GPU parity infra the other gaps don't.
- **Parity test**: categorical corpus on ROCm vs CPU f64 anchor within ~1e-6.
- **Priority**: later phase — largest, riskiest; historically deprioritized
  ("v2 QGD-02 … not currently scheduled" per prior audit reading of CONCERNS).

### G6 — `stochastic_rounding=true` oracle golden + Python gate (residual)

- Fully specified already in
  `.planning/plans/quantized-grad-param-plumbing/{SPEC,PLAN}.md` (QGP-06, QGP-08,
  T-06/T-08). **Both BLOCKED on environment**: (a) no `lightgbm==4.6` Python
  install to generate the stochastic golden; (b) `lgbm-python` fails to link
  (`mold: library not found: python3.14`) so `reject_gate` can't run.
  `[VERIFIED: LOCAL SPEC.md §QGP-06/08]`
- Also carries a **pre-existing defect** (SPEC §9 Risk 1): the `linear_tree`
  sub-case of `reject_gate` (`params.rs:312-314`) still asserts rejection, but
  `linear_tree` is no longer in `OUT_OF_SCOPE_PARAMS` → the test is now wrong.
- **Action**: not new code — unblock environment, generate golden, fix the two
  stale test sub-cases.

---

## 6. Standard Stack (dependencies — resolved, verified)

`[VERIFIED: LOCAL Cargo.lock, Cargo.toml, rust-toolchain.toml]`

| Component | Workspace declared | Resolved (Cargo.lock) | Notes |
|---|---|---|---|
| Rust toolchain / edition | `1.95.0` / `2024` | — | `rust-toolchain.toml`, `resolver = "3"` |
| `cubecl` | `0.10.0` | `0.10.0` | Sole compute seam (CMP-01). For G3 kernels, fetch ctx7 docs first. |
| `thiserror` | `2.0.18` | **`1.0.69`** is the top-resolved node; workspace code uses `2.0.18` (both present — 1.0.69 is a transitive dup) | Not a project inconsistency. |
| `anyhow` | `1.0.102` | `1.0.102` | app/dev-layer errors. |
| `pyo3` | `0.27` (abi3-py311) | `0.27.2` | Version triangle w/ numpy/pyo3-polars. |
| `numpy` (rust-numpy) | `0.27` | `0.27.1` | — |
| `polars` | direct in `lgbm-python` | `0.53.0` | dtype-categorical/u8/u16 features. |
| `pyo3-polars` | `0.26` | `0.26.0` | — |
| `rayon` | `~1.10` (comment) | `1.12.0` | CPU parallelism (OpenMP analog). |
| `ndarray` | — | `0.17.2` | — |
| `mimalloc` | `0.1` | `0.1.52` | allocator; `lgbm-python` always-on. |
| `serde_json` | **absent** | **absent** | Relevant to G2 decision — hand-emit preferred. `[VERIFIED: LOCAL grep]` |

- **C++ reference version**: `lib_lightgbm` **4.6** (bit-exact CPU anchor);
  oracle wheel is `pip lightgbm==4.6.0` installed via `uv`. `[PROJECT: CLAUDE.md;
  memory/cpp-linear-tree-oracle.md]`
- **No new external crate is required for G1/G2/G4/G5.** G3 stays within `cubecl`
  0.10. `[INFERRED from gap analysis above]`

---

## 7. Build / Test / Parity Command Reference

`[VERIFIED: LOCAL — commands assembled from README.md, xtask, SPEC/PLAN, and
directory listings; NOT all executed this session (see caveats)]`

```bash
# --- Rust build / test (CPU anchor) ---
cargo build --workspace --tests           # NOTE: lgbm-python may fail to LINK
                                          # (mold: library not found: python3.14) — env issue
cargo test -p lgbm-core                    # config parsing/scope/alias suites
cargo test -p oracle-harness --test linear_parity      # linear-tree end-to-end
cargo test -p oracle-harness --test quantized_parity   # 4 tests (SPEC §QGP-09)

# --- CRITICAL: run learner/on-device tests with the regression gated OFF ---
LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner   # else 6 learner tests FAIL
                                                       # (commit 42249ca regression)

# --- Config-drift mechanical param audit (REQUIRES LightGBM/ checkout) ---
cargo test -p oracle-harness --test config_drift       # 2/3 tests FAIL here: ENOENT
                                                       # LightGBM/src/io/config_auto.cpp absent

# --- Golden regeneration (deterministic) ---
cargo run -p xtask -- regen                # RNG goldens; other oracle-capture subcommands exist
# quantized golden generator (needs real lightgbm==4.6):
.venv/bin/python crates/oracle-harness/tests/fixtures/quantized/gen_golden.py

# --- Python bindings (needs working Python dev env) ---
cd crates/lgbm-python && maturin develop --release
pytest crates/lgbm-python/python/tests/    # test_booster_parity.py, test_custom_refit_parity.py,
                                           # test_sklearn_parity.py, test_persistence.py, ...

# --- Oracle wheel install (per memory) ---
uv ...  # install lightgbm==4.6.0 (ships lib_lightgbm.so) — network + ~/.cargo/bin/uv available
```

Caveats verified this session: `LightGBM/` absent → `config_drift` fails;
`lgbm-python` link failure is environmental; `LGBM_CUDA_ON_DEVICE=0` requirement
is from `memory/resident-score-host-update-gotcha.md`.

---

## 8. Common Pitfalls and Risks

| Risk | Trigger | Consequence | Prevention | Verification |
|---|---|---|---|---|
| On-device score staleness | `on_device_default()==true` (default) with a host-only per-tree score update | Gradients computed from a **stale resident** score buffer → silent training divergence after the first such tree | Gate `resident_active` off for host-only paths (`&& !self.linear_tree` pattern) OR sync resident buffer | Run learner tests with `LGBM_CUDA_ON_DEVICE=0`; compare `Σg²` fed to learner vs reference `[PROJECT: memory]` |
| C++ tree absent | `config_drift`/any C++-source-reading test | ENOENT failures; can't mechanically confirm param coverage | Ensure `LightGBM/` checked out in execution env (it's intentionally gitignored) | `ls LightGBM/src/io/config_auto.cpp` |
| Stochastic RNG mismatch | Asserting bit-exact `stochastic_rounding=true` vs C++ | False failures — Rust xorshift64 ≠ C++ mt19937 | Use magnitude-regime delta gate, not 1e-6 (SPEC QGP-08 precedent) | `quantized_parity.rs` delta-gate methodology |
| JSON/if-else float drift | Emitting floats with `ryu`/`to_string()` instead of `%g` | Byte-diff vs C++ golden (`0.1` vs `0.10000000000000001`) | Reuse `format.rs` `format_g17`/`format_g6` | byte-exact golden compare |
| Python link failure | `cargo test -p lgbm-python` in sandbox | `mold: library not found: python3.14` | Run in an env with matching Python dev libs / maturin venv | `maturin develop` succeeds |
| Config accepts, kernel rejects | Setting `path_smooth` (IN_SCOPE) with the compute path | `ComputeError::Runtime` at train time | Implement G5 or document the limitation explicitly | golden with `path_smooth != 0` |

---

## 9. Testing and Verification Strategy

- **Unit**: `lgbm-core` config parsing/scope (`config_validation.rs`,
  `scope_classification.rs`); per-crate `#[cfg(test)]`.
- **Integration / contract (oracle-harness)**: every gap-fill needs a
  `tests/<feature>_parity.rs` comparing to a committed `lib_lightgbm` 4.6 golden;
  CPU f64-fold path bit-exact where the algorithm permits, else ~1e-6.
  `[PROJECT: PROJECT.md Constraints]`
- **Golden generation**: `xtask regen` / oracle-capture subcommands assert the
  installed C++ version matches before capture; quantized/linear goldens via
  `gen_golden.py` against the `lightgbm==4.6` uv wheel.
- **Config-drift**: `config_drift.rs` is the mechanical "which C++ params are
  unaccounted for" gate — **run FIRST** on a machine with `LightGBM/`.
- **Regression discipline**: run `LGBM_CUDA_ON_DEVICE=0` for treelearner tests
  until commit `42249ca`'s regression is resolved.
- **Python-level**: `pytest crates/lgbm-python/python/tests/` (mirrors official
  `lightgbm` API surface) — needs a working maturin/venv env.

---

## 10. Planning Guidance

- **Suggested ordering**: G2 (JSON dump) → G1 (if-else codegen) [both reuse `%g`,
  no new deps, self-contained, need only a user API-shape decision] → G4-serial
  (NA forward branch) → G5 (gain params) → G6 residual (env-unblock) → G3
  (categorical GPU, last — largest/riskiest).
- **Dependencies between tasks**: G1 and G2 share the tree-walk + `%g` emitter —
  consider a shared serialization module. G4-serial (host) is independent of
  G4-resident (GPU) and higher-value. G5 and G4-resident both touch hot compute
  kernels — schedule apart to isolate parity regressions.
- **Decisions the planner must preserve (locked)**: C-API, distributed/MPI, GPU
  OpenCL knobs, and the fully-resident grow loop are OUT OF SCOPE — do not scope
  work there. `[PROJECT: PROJECT.md]`
- **Confirm dependencies first (AGENTS.md rule)**: no new crate needed for
  G1/G2/G4/G5; explicitly decide serde_json vs hand-emit for G2; fetch cubecl
  ctx7 docs before G3.
- **Spikes / user decisions needed before implementation**: G1 API shape; G2
  scope confirmation; G3 GPU parity-infra bring-up; G6 environment provisioning.

---

## 11. Open Questions (materially block/alter planning)

1. **JSON dump scope (G2)**: PROJECT.md marks "JSON serialization" *Validated*,
   but it is absent. Is it genuinely in v1 scope, and was the "validated" claim
   erroneous? (Decides whether G2 is the first target.)
2. **If-else codegen API (G1)**: with no CLI, what is the Rust/Python entry point
   (`Booster::model_to_cpp() -> String`? file-writing to `convert_model` path?)?
   Blocks acceptance criteria.
3. **`LightGBM/` availability**: will the execution environment have the C++ tree
   checked out (required for `config_drift` and any C++-line re-verification)?
4. **On-device regression (`42249ca`)**: should closing it be a prerequisite
   phase, or is `LGBM_CUDA_ON_DEVICE=0` an acceptable standing test workaround
   for the duration of gap-closure?
5. **G3 categorical GPU priority**: defer to a later phase (recommended) or
   include now? It needs GPU parity infrastructure the other gaps don't.
6. **serde_json vs hand-emit for G2** — adding a dependency vs guaranteed `%g`
   byte-parity. Recommend hand-emit; confirm.

---

## 12. Sources

- **Project docs (PageIndex library empty — local files only)**:
  `.planning/PROJECT.md`, `.planning/STATE.md`,
  `.planning/plans/cpp-feature-parity/research.md` (prior audit, superseded),
  `.planning/plans/quantized-grad-param-plumbing/{SPEC,PLAN}.md`,
  `.planning/codebase/{ARCHITECTURE,STRUCTURE,STACK,CONCERNS,CONVENTIONS,TESTING,INTEGRATIONS}.md`,
  `CLAUDE.md`, `AGENTS.md`, `docs/LIGHTGBM-CPP-DESIGN.md`, `cubecl_kernel_gaps.md`.
  `[PageIndex `get_folder_structure` not used this session — prior audit recorded
  the library as empty for this workspace.]`
- **Memory files**: `memory/cpp-linear-tree-oracle.md`,
  `memory/resident-score-host-update-gotcha.md`, `memory/MEMORY.md`.
- **CodeGraph**: `.codegraph/` index present. Cross-checked linear-tree wiring
  and `fit_linear_leaves` callers (prior audit's codegraph blast-radius, now
  re-confirmed via grep showing 2 production callers at `gbdt.rs:1030,1197`).
- **Local verification (this session)**: `git log`, `git status`,
  `ls crates/`/`Cargo.toml`, `ls -d LightGBM` (absent), grep sweeps for
  `unimplemented!`/`todo!`/`not yet implemented`/`fit_linear_leaves`/`linear_tree`/
  `use_quantized_grad`/`convert_model`/`to_if_else`/`to_json`/`dump_model`/
  `serde_json`/`shap`/`pred_leaf`/`on_device`, `sed`/`Read` of `split.rs:435-455`,
  `learner.rs:1110-1122`, `format.rs:1-40`, `config/set.rs`, `config/scope.rs`,
  version extraction from `Cargo.lock`, `rust-toolchain.toml`.
- **Context7 CLI**: not invoked — no library-API question arose (every G1/G2/G4/G5
  gap closeable with existing deps). Fetch `cubecl` docs before G3.
- **Web**: none — all answerable from local evidence.

---

## 13. Confidence Assessment

**HIGH** (directly verified this session by command output / file read):
- Linear-tree and quantized-grad now wired on this branch (grep of production
  call sites + committed SPEC/PLAN).
- JSON model dump absent (workspace-wide grep, zero hits) — and the contradiction
  with PROJECT.md's "Validated" claim.
- If-else codegen absent (grep, zero consumers of parsed `convert_model`).
- G4/G5 typed-error gates (direct Read of the exact comments/error strings).
- SHAP/leaf-index/raw prediction present (`predict.rs` symbols).
- Resolved dependency versions and toolchain.
- `LightGBM/` absent → `config_drift` blocked; `on_device_default` regression.

**MEDIUM** (project design docs / codebase-map, not re-derived from C++ source
because `LightGBM/` is unavailable):
- Exact C++ algorithmic descriptions for `ModelToIfElse`, `DumpModel`,
  `na_as_missing` routing, and `FindBestThreshold` gain params.
- Completeness of DART/RF/GOSS/CEGB/monotone/forced-splits (files exist with real
  call sites; not read line-by-line).

**LOW** (needs validation before planning commits):
- Whether JSON dump is truly in v1 scope (vs the erroneous "Validated" flag).
- Whether the `linear_tree` `reject_gate` sub-case is currently failing (couldn't
  run `lgbm-python` — link failure).
- Any C++ file:line cited from design docs without `LightGBM/` re-verification.
</content>
</invoke>
