# Coding Conventions

**Analysis Date:** 2026-07-09

Scope: the Rust workspace crates under `crates/` and `xtask/`. `LightGBM/` and
`LightGBM-release-4.6.0.99/` are read-only C++ reference trees and are excluded —
their C++ style is documented separately and does not apply to this project's Rust code.

## Naming Patterns

**Crates:**
- One crate per subsystem, all prefixed `lgbm-*`: `lgbm-core`, `lgbm-compute`,
  `lgbm-dataset`, `lgbm-model`, `lgbm-treelearner`, `lgbm-objective`, `lgbm-metric`,
  `lgbm-boosting`, `lgbm` (top-level facade), `lgbm-python` (PyO3 binding),
  `oracle-harness` (parity test harness, dev-only). Declared in the workspace
  `Cargo.toml` members list.

**Files:**
- `snake_case.rs`, one module per file, named after the C++ subsystem it ports
  (e.g. `crates/lgbm-treelearner/src/data_partition.rs` mirrors `data_partition.hpp`).
- Every crate has an `src/error.rs` defining that crate's domain error enum
  (`crates/lgbm-core/src/error.rs`, `crates/lgbm-dataset/src/error.rs`,
  `crates/lgbm-treelearner/src/error.rs`, etc.) — this is a structural convention,
  not incidental.

**Types:**
- `PascalCase`, mirrors the C++ class it ports where one exists (`DataPartition`,
  `BinMapper`, `GradientDiscretizer`, `TreeLearnerError`).
- Error enums are named `<Crate>Error` (`CoreError`, `DatasetError`,
  `TreeLearnerError`, `ComputeError`) with domain sub-errors named `<Subsystem>Error`
  (`ConfigError`, `ForcedSplitError`).

**Functions:**
- `snake_case`, verbs matching the C++ method they port (`find_bin`,
  `construct_bitset`, `find_best_threshold_categorical`, `fix_histogram`).

**Constants:**
- `SCREAMING_SNAKE_CASE`. Tolerance/algorithm constants document their provenance
  in a doc comment, e.g. `pub const ORACLE_TOL: f32 = 1e-6;`
  (`crates/oracle-harness/src/comparator.rs`) vs `lgbm_core::types::K_EPSILON`
  (`1e-15`, an *algorithm* constant, explicitly distinguished from comparison
  tolerance in the doc comment).
- Tunable perf/behavior knobs read once via `OnceLock` from an env var with a
  documented default, e.g. `LGBM_PAR_PARTITION_MIN` (default 65536,
  `crates/lgbm-treelearner/src/data_partition.rs`).

## Code Style

**Formatting:**
- No `rustfmt.toml` or `clippy.toml` present at the repo root — default `rustfmt`
  and `clippy` settings apply (pinned toolchain `1.95.0`,
  `rust-toolchain.toml`, components `rustfmt` + `clippy`).

**Edition/Rust version:**
- `edition = "2024"`, `rust-version = "1.95"` set once in `[workspace.package]`
  (`Cargo.toml`) and inherited via `edition.workspace = true` /
  `rust-version.workspace = true` in every crate's `Cargo.toml`.

**Doc comments are load-bearing:**
- Every module (`lib.rs`) opens with a `//!` doc block stating: (1) what C++
  subsystem/file this is a "faithful 1:1 port" of, (2) crate-boundary rules (e.g.
  "this crate has NO `cubecl` dependency"), and (3) current plan/phase status.
  See `crates/lgbm-treelearner/src/lib.rs`, `crates/lgbm-dataset/src/lib.rs`.
- Non-obvious constants and thresholds carry a doc comment tracing the spike/phase
  that derived their value and the tradeoff considered (see `LGBM_PAR_PARTITION_MIN`
  above) — do not add a magic-number constant without this provenance comment.

**Dependency versions:**
- Shared dependencies (`thiserror = "2.0.18"`, `anyhow = "1.0.102"`,
  `cubecl = "0.10.0"`) are pinned once in `[workspace.dependencies]`
  (`Cargo.toml`) and referenced via `thiserror.workspace = true` etc. in each
  crate — do not add a crate-local version override.

## Import Organization

- Standard `use` grouping: external crate imports, then intra-workspace crate
  imports (`lgbm_core::...`, `lgbm_compute::...`), commented when the import
  exists specifically to preserve a crate-boundary contract, e.g.:
  ```rust
  use lgbm_compute::error::ComputeError;
  use lgbm_compute::Backend;
  use lgbm_compute::BinColumn;
  // `ComputeClient` is re-exported by the compute seam (CMP-01) so this crate names
  // the Backend ops' client argument without ever depending on `cubecl` directly.
  use lgbm_compute::ComputeClientReexport as ComputeClient;
  ```
  (`crates/lgbm-treelearner/src/data_partition.rs`)
- Crate `lib.rs` files declare `pub mod` for every submodule, then re-export the
  public surface with `pub use` (see `crates/lgbm-dataset/src/lib.rs`,
  `crates/lgbm-treelearner/src/lib.rs`). New public types must be added to both
  the `pub mod` list and the `pub use` re-export block.

## Error Handling

**Two-tier scheme mandated by CLAUDE.md and consistently followed:**
- **`thiserror`** for structured domain errors at every library crate boundary.
  Every crate depends on `thiserror` (`grep -rl thiserror crates --include=Cargo.toml`
  hits all 9 non-harness library crates) and defines its own `src/error.rs`.
  Pattern (`crates/lgbm-core/src/error.rs`):
  ```rust
  #[derive(Debug, Error, Clone, PartialEq, Eq)]
  pub enum ConfigError {
      #[error("parameter `{param}` has invalid value `{value}` for its expected type")]
      InvalidType { param: String, value: String },
      ...
  }

  #[derive(Debug, Error)]
  pub enum CoreError {
      #[error(transparent)]
      Config(#[from] ConfigError),
  }
  ```
  - Domain-specific leaf errors (e.g. `ConfigError`) are wrapped into a single
    crate-level top error (e.g. `CoreError`) via `#[error(transparent)]` +
    `#[from]`, so callers match one type per crate.
  - Doc comments on error enums explicitly map each variant back to the C++
    `Log::Fatal`/`CHECK_*` site it replaces — e.g. `ConfigError::OutOfRange`
    documents which `CHECK_GT/GE/LE/LT` it stands in for. When porting new C++
    fatal-error sites, follow this mapping-comment convention.
  - Explicit design rule stated in-code: "never hand-roll `impl
    std::error::Error`" and "never panic on user input" (Security V5) — invalid
    user/config input must surface as a typed `Result`, not a panic.
- **`anyhow`** is used only in ergonomic/app-level or test-harness layers, not
  inside library crate public APIs: `crates/oracle-harness/Cargo.toml`,
  `crates/lgbm-dataset/Cargo.toml` (internal helper paths), `crates/lgbm-model/Cargo.toml`.

**Crate isolation via error `#[from]` wrapping:**
- Cross-crate failures are absorbed into the calling crate's own error type via
  `#[from]`, not propagated as the foreign crate's raw error type, keeping each
  crate's public error surface self-contained (stated pattern in
  `crates/lgbm-treelearner/src/lib.rs`: "backend failures are wrapped via `#[from]`").

## Module Design / Crate Boundaries

- **CMP-01 containment rule**: only `lgbm-compute` (and its `[dev-dependencies]`
  test-only pulls) may depend on `cubecl`. Downstream crates (`lgbm-treelearner`,
  `lgbm-dataset`, etc.) interact with compute only through the `Backend` trait /
  `ComputeClientReexport` seam and must never add a direct `cubecl` dependency.
  This is enforced by convention + tests (`crates/lgbm-compute/tests/cmp01_containment.rs`)
  rather than a lint — do not add `cubecl` to any crate's `[dependencies]` except
  `lgbm-compute`.
- `oracle-harness` intentionally keeps its `[dependencies]` (library surface) free
  of GPU/cubecl deps and pulls every crate under test only as `[dev-dependencies]`,
  documented inline in `crates/oracle-harness/Cargo.toml` — preserves the "harness
  library itself never forces a GPU runtime on downstream consumers" property.
- Feature-gated backends: `rocm` is an additive Cargo feature
  (`oracle-harness/Cargo.toml`: `rocm = ["lgbm-compute/rocm"]`) — a default
  (feature-less) build is CPU-only; ROCm/HIP tests require `--features rocm`
  and are excluded from CI by default.

## Function & Module Design

- Functions/algorithms are ported 1:1 from a named C++ source file/method — module
  doc comments cite the exact `.cpp`/`.h`/`.hpp` file being ported so behavior can
  be diffed against the reference (e.g. "Faithful 1:1 port of LightGBM's
  `SerialTreeLearner` (`src/treelearner/serial_tree_learner.cpp` +
  `serial_tree_learner.h`)").
- Public API surface of each crate is minimized: `pub mod` for implementation
  files, but only the specific types/functions needed downstream are re-exported
  via `pub use` at the crate root (`crates/lgbm-dataset/src/lib.rs`).
- In-code `#[cfg(test)] mod tests { ... }` unit tests live alongside the
  implementation in the same file as a standard pattern (`crates/lgbm-core/src/error.rs`,
  `crates/lgbm-treelearner/src/data_partition.rs`), in addition to the crate-level
  `tests/` integration-test directories (see TESTING.md).

---

*Convention analysis: 2026-07-09*
