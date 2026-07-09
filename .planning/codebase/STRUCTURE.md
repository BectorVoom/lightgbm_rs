# Codebase Structure

**Analysis Date:** 2026-07-09

## Directory Layout

```
lightgbm_rs/
├── Cargo.toml                    # Workspace root: 12 members, shared deps, release/profiling profiles
├── Cargo.lock
├── rust-toolchain.toml
├── crates/
│   ├── lgbm-core/                # Foundation: Config, error, RNG, shared types
│   ├── lgbm-dataset/              # Binning, FeatureGroup, EFB, Dataset/Metadata
│   ├── lgbm-model/                 # Tree, GbdtModel ensemble, model text/JSON I/O, predict
│   ├── lgbm-compute/                # CubeCL isolation seam: Backend trait + kernels/
│   │   └── src/kernels/              # #[cube] kernels: histogram, split, partition, predict, grow_driver, ...
│   ├── lgbm-objective/              # Per-row gradient/hessian (regression/binary/multiclass/rank/xentropy/custom)
│   ├── lgbm-metric/                 # Evaluation metrics + shared DCG tables
│   ├── lgbm-treelearner/           # SerialTreeLearner: leaf-wise tree growth loop
│   ├── lgbm-boosting/              # Gbdt outer loop, DART/RF/GOSS, ScoreUpdater, early stopping
│   ├── lgbm/                       # Facade crate: Booster, TrainingBuilder, train*() entry points
│   │   └── examples/                 # bench_real.rs, bench_train.rs
│   ├── lgbm-python/                # PyO3 bindings mirroring the official `lightgbm` Python API
│   │   └── python/lightgbm_rs/       # Python package wrapper + python/tests/
│   └── oracle-harness/             # C++ parity test infra: comparator + committed golden fixtures
│       ├── fixtures/                 # REFERENCE_MANIFEST.md, rng_sequence.txt, primitives/
│       └── tests/                    # ~30 parity test files (kernel/learner/boosting/on_device/...)
├── xtask/                         # Dev automation: regen C++ golden fixtures (only step needing C++ toolchain)
├── docs/                          # LIGHTGBM-CPP-DESIGN.md, cuda-kernel-design.md (project design docs)
├── scripts/                       # export_highdim_dataset.py and similar helper scripts
├── LightGBM/                      # READ-ONLY C++ reference (mainline) — porting source, not part of the Rust build
├── LightGBM-release-4.6.0.99/     # READ-ONLY C++ reference (AMD ROCm/HIP fork) — GPU kernel parity/perf baseline
└── .planning/                     # GSD planning artifacts (phases, codebase docs, notes)
```

## Directory Purposes

**`crates/lgbm-core/`:**
- Purpose: Foundation crate every other crate depends on.
- Contains: `Config` struct (all training/prediction params, mirrors C++ `config.h`), error primitives, deterministic RNG, shared numeric type aliases.
- Key files: `src/types.rs`, `src/error.rs`, `src/random.rs`, `src/lib.rs`.

**`crates/lgbm-dataset/`:**
- Purpose: Turn raw feature/label data into the binned columnar representation the engine trains on.
- Contains: `BinMapper` (continuous→bin), `FeatureGroup`, `MultiValBin` (sparse/grouped), EFB (Exclusive Feature Bundling), `Dataset`/`Metadata`, raw ingest.
- Key files: `src/bin_mapper.rs`, `src/dataset.rs`, `src/feature_group.rs`, `src/multi_val_bin.rs`, `src/efb.rs`, `src/ingest.rs`.
- Subdirectory: `src/bin/` (bin-type-specific implementations).

**`crates/lgbm-model/`:**
- Purpose: Own tree/ensemble data structures, serialization, and prediction — independent of training machinery.
- Contains: `Tree` (node/split repr), `GbdtModel` (ensemble), text/JSON model format, prediction walk, `ObjectiveKind::convert_output`.
- Key files: `src/tree.rs`, `src/ensemble.rs`, `src/predict.rs`, `src/model_text.rs`, `src/format.rs`, `src/objective.rs`.

**`crates/lgbm-compute/`:**
- Purpose: **The single CubeCL isolation seam (CMP-01)** — the only crate permitted to name `cubecl` types or a concrete GPU/CPU compute runtime.
- Contains: `Backend` trait, `BinColumn`, `DeviceFrontier`, `runtime.rs` (capability probe/`ReducePath`), `gain.rs` (`GainConfig`/`SplitInfo`).
- Key files: `src/lib.rs` (`trait Backend` at line 658), `src/runtime.rs`, `src/gain.rs`, `src/device_metric.rs`, `src/device_objective.rs`, `src/fusion_prof.rs`.
- Subdirectory: `src/kernels/` — 25 `#[cube]` kernel modules: `histogram.rs`, `histogram_arena.rs`, `split.rs`, `split_info.rs`, `best_split.rs`, `data_partition.rs`, `partition.rs`, `subtract.rs`, `predict.rs`, `tree.rs`, `grow_driver.rs`, `score_updater.rs`, `row_data.rs`, `column_data.rs`, `copy_subrow.rs`, `categorical_split.rs`, `objective_{binary,multiclass,rank,regression}.rs`, `metric_pointwise.rs`, `random.rs`, `primitives.rs`, `autotune.rs`, `mod.rs`.

**`crates/lgbm-objective/`:**
- Purpose: Compute per-row gradients/hessians from current scores for each supported loss.
- Contains: `regression.rs`, `binary.rs`, `multiclass.rs`, `rank.rs`, `xentropy.rs`, `custom.rs`, `percentile.rs`.

**`crates/lgbm-metric/`:**
- Purpose: Evaluate loss/metric values on train/validation scores.
- Contains: `regression.rs`, `binary.rs`, `multiclass.rs`, `rank.rs`, `xentropy.rs`, `dcg_calculator.rs` (shared with `lgbm-objective`'s ranking code).

**`crates/lgbm-treelearner/`:**
- Purpose: Grow one decision tree per call via best-first (leaf-wise) expansion, driving `Backend` ops.
- Contains: `SerialTreeLearner` (`learner.rs`), `DataPartition` (`data_partition.rs`), `HistogramPool`/`ResidentPool`, `LeafSplits`, categorical-split handling, monotone constraints, column sampling, CEGB (cost-effective gradient boosting), gradient discretizer (quantized training), forced splits.
- Key files: `src/learner.rs`, `src/data_partition.rs`, `src/histogram_pool.rs`, `src/resident_pool.rs`, `src/leaf_splits.rs`, `src/feature_histogram_categorical.rs`, `src/monotone_constraints.rs`, `src/col_sampler.rs`, `src/cost_effective_gradient_boosting.rs`, `src/gradient_discretizer.rs`, `src/forced_splits.rs`, `src/split_info.rs`, `src/phase_prof.rs`.

**`crates/lgbm-boosting/`:**
- Purpose: Outer gradient-boosting loop — one `SerialTreeLearner` call per iteration, ensemble management, bagging/GOSS, scores, early stopping, DART/RF variants.
- Key files: `src/gbdt.rs` (`Gbdt<'a>`, `train_one_iter`, `train`, `DartConfig`, `RfConfig`), `src/score_updater.rs`, `src/sample_strategy.rs`, `src/early_stopping.rs`, `src/objective.rs`.

**`crates/lgbm/`:**
- Purpose: Public facade crate — the single import surface for the whole engine, and the PyO3 binding target.
- Key files: `src/booster.rs` (`Booster`, `DenseCorpus`, `RawCorpus`, `train*` functions), `src/builder.rs` (`TrainingBuilder`), `src/lib.rs` (curated re-exports), `src/error.rs` (`LgbmError`).
- Subdirectory: `examples/` — `bench_real.rs`, `bench_train.rs` (performance benchmark harnesses).

**`crates/lgbm-python/`:**
- Purpose: PyO3 extension module mirroring the official `lightgbm` Python package API surface.
- Key files: `src/booster.rs`, `src/dataset.rs`, `src/params.rs`, `src/callbacks.rs`, `src/marshal.rs`, `src/lib.rs`.
- Subdirectory: `python/lightgbm_rs/` — the pure-Python side of the wrapper package; `python/tests/` — Python-level test suite; `pyproject.toml` — maturin build config (`extension-module` enabled only via maturin, not `cargo test`).

**`crates/oracle-harness/`:**
- Purpose: C++ parity test infrastructure — dev-dependency of nearly every library crate, never a runtime dependency of any of them.
- Contains: `src/comparator.rs` (value comparison utilities), `tests/` (~30 parity test files: kernel, learner, boosting, objective, metric, on-device, rng, predict, partition parity, etc.), `fixtures/` (`REFERENCE_MANIFEST.md`, `rng_sequence.txt`, `primitives/` — committed C++ golden data).
- Generated: fixture regeneration is manual via `xtask regen`, not automatic.
- Committed: yes — goldens are committed so `cargo test` never needs a C++ toolchain.

**`xtask/`:**
- Purpose: Dev-only automation, invoked via `cargo run -p xtask -- <subcommand>`.
- Key files: `src/main.rs` — `regen` subcommand regenerates committed C++ RNG golden fixtures deterministically (seeded, no wall-clock entropy).

**`LightGBM/` and `LightGBM-release-4.6.0.99/`:**
- Purpose: READ-ONLY C++ reference implementations. `LightGBM/` is mainline (porting source for algorithm/API fidelity); `LightGBM-release-4.6.0.99/` is the AMD ROCm/HIP fork (hipified CUDA, used as the GPU kernel parity/perf baseline instead of mainline for GPU work).
- **Never edit or git-add these trees** — per project memory, `LightGBM/` is intentionally untracked; treat both as external reference-only, not part of the Rust architecture.

**`docs/`:**
- Purpose: Design documents referenced by CLAUDE.md/AGENTS.md.
- Key files: `LIGHTGBM-CPP-DESIGN.md` (C++ reference architecture notes), `cuda-kernel-design.md` (GPU kernel design contract).

## Key File Locations

**Entry Points:**
- `crates/lgbm/src/booster.rs`: `train`, `train_raw`, `train_with_valid`, `train_custom*` — primary training entry points.
- `crates/lgbm/src/booster.rs` (`impl Booster`): `predict`, `predict_row`, `predict_raw_batch`, `refit`, `save_model`, `model_from_string` — post-training entry points.
- `crates/lgbm-python/src/lib.rs`: PyO3 module registration — the Python-facing entry point.
- `xtask/src/main.rs`: dev-tool entry point (`regen`).

**Configuration:**
- `Cargo.toml` (workspace root): member list, shared workspace dependencies, `release`/`profiling` build profiles.
- `crates/lgbm-core/src/types.rs`: `Config` struct — the single source of truth for all training/prediction parameters.
- Per-crate `Cargo.toml` files carry extensive load-bearing comments documenting dependency-choice rationale (e.g. CMP-01 containment, PyO3 version triangle) — read these before adding a dependency.

**Core Logic:**
- `crates/lgbm-treelearner/src/learner.rs`: the tree-growth spine (`SerialTreeLearner`).
- `crates/lgbm-boosting/src/gbdt.rs`: the outer boosting loop (`Gbdt`).
- `crates/lgbm-compute/src/lib.rs`: the `Backend` trait — the compute abstraction every numeric operation flows through.
- `crates/lgbm-compute/src/kernels/`: all `#[cube]` kernel implementations.

**Testing:**
- Per-crate `src/` unit tests (inline `#[cfg(test)]` modules, not separately located — verify per crate).
- `crates/oracle-harness/tests/`: cross-crate C++ parity integration tests (the primary correctness gate for numerical fidelity).
- `crates/lgbm-python/python/tests/`: Python-level pytest suite for the PyO3 bindings.

## Naming Conventions

**Files:**
- `snake_case.rs`, one module per file, matching Rust convention (e.g. `bin_mapper.rs`, `data_partition.rs`, `score_updater.rs`).
- Error types live in a dedicated `error.rs` per crate (e.g. `lgbm-treelearner/src/error.rs` → `TreeLearnerError`).
- `lib.rs` per crate is a thin facade of `pub mod`/`pub use` declarations, not implementation.

**Crates:**
- `lgbm-<subsystem>` naming pattern throughout the workspace (`lgbm-core`, `lgbm-dataset`, `lgbm-model`, `lgbm-compute`, `lgbm-objective`, `lgbm-metric`, `lgbm-treelearner`, `lgbm-boosting`), with the bare `lgbm` crate reserved for the public facade and `lgbm-python` for the binding layer.
- `oracle-harness` and `xtask` break the `lgbm-*` pattern deliberately — they are dev/test infrastructure, not part of the shipped library surface.

**Directories:**
- All library crates under `crates/`; the only workspace member outside it is `xtask/` (dev tooling convention: build/dev tools live at the repo root, not under `crates/`).
- GPU/compute kernels are grouped under a `kernels/` subdirectory only inside `lgbm-compute` — no other crate has a kernels-style subdirectory, reinforcing that this is the sole compute seam.

## Where to Add New Code

**New objective/loss function:**
- Implementation: `crates/lgbm-objective/src/<name>.rs`, following the pattern of `regression.rs`/`binary.rs`.
- If it needs GPU support: add a device-side kernel in `crates/lgbm-compute/src/kernels/objective_<name>.rs` and gate it through `device_objective.rs`'s `DeviceObjectiveKind`.

**New metric:**
- Implementation: `crates/lgbm-metric/src/<name>.rs`, following `regression.rs`/`binary.rs`.

**New compute kernel (histogram/split/partition/predict variant):**
- Implementation: `crates/lgbm-compute/src/kernels/<name>.rs`, exposed as a new `Backend` trait method in `crates/lgbm-compute/src/lib.rs`.
- **Never** add a `cubecl` dependency to any other crate to implement this — CMP-01 containment is enforced by a guard test.

**New boosting variant (beyond GBDT/DART/RF/GOSS):**
- Implementation: `crates/lgbm-boosting/src/`, following `sample_strategy.rs`'s pattern for a new `SampleStrategy`-like abstraction, wired into `gbdt.rs`.

**New model format / serialization target:**
- Implementation: `crates/lgbm-model/src/format.rs` or a new sibling module, following `model_text.rs`.

**Public API surface change:**
- Add the `pub use` re-export in `crates/lgbm/src/lib.rs`; keep new user-facing types/functions in `crates/lgbm/src/booster.rs` or `builder.rs`.
- Mirror any new public API into `crates/lgbm-python/src/` to preserve API-surface parity with the official Python `lightgbm` package (see project constraint in CLAUDE.md).

**Parity/correctness test:**
- Add to `crates/oracle-harness/tests/`, following the naming pattern `<subsystem>_parity.rs` (e.g. `histogram` → look at `kernel_parity.rs`, `partition_parity.rs`).
- If a new C++ golden fixture is needed, extend `xtask/src/main.rs`'s `regen` subcommand and commit the regenerated fixture under `crates/oracle-harness/fixtures/`.

## Special Directories

**`LightGBM/`:**
- Purpose: Read-only mainline C++ reference implementation (the porting source).
- Generated: No (external upstream source).
- Committed: **No** — deliberately untracked in git (per project memory: "never git-add LightGBM/; worktrees break for phases needing it").

**`LightGBM-release-4.6.0.99/`:**
- Purpose: Read-only AMD ROCm/HIP fork, the real HIP histogram baseline for GPU kernel parity/perf work.
- Generated: No.
- Committed: Not verified in this pass — treat as external reference like `LightGBM/`.

**`crates/oracle-harness/fixtures/`:**
- Purpose: Committed C++ golden output used for bit-exact/tolerance parity assertions.
- Generated: Yes, via `xtask regen` (deterministic, seeded — reruns produce byte-identical output).
- Committed: Yes.

**`target/`:**
- Purpose: Cargo build output.
- Generated: Yes.
- Committed: No (standard `.gitignore`).

**`.planning/`:**
- Purpose: GSD workflow planning artifacts (phases, codebase maps, notes) — not part of the Rust build or runtime.
- Generated: Partially (by GSD tooling).
- Committed: Yes.

---

*Structure analysis: 2026-07-09*
