# Testing Patterns

**Analysis Date:** 2026-07-09

Scope: the Rust workspace (`crates/`, `xtask/`). `LightGBM/` and
`LightGBM-release-4.6.0.99/` are the read-only C++ reference used to *generate*
parity goldens, not part of this project's own test suite.

## Test Framework

**Runner:**
- Standard Rust built-in test harness (`cargo test`), no external test framework
  crate. `edition = "2024"`, toolchain pinned `1.95.0` (`rust-toolchain.toml`).

**Assertion style:**
- Plain `assert!` / `assert_eq!` / `matches!`; no assertion-library dependency
  (no `pretty_assertions`, no `assert2`, etc. detected).

**Run commands:**
```bash
cargo test --workspace                       # all crates, CPU-only default features
cargo test -p oracle-harness                  # parity suite only
cargo test -p oracle-harness --features rocm  # + ROCm/HIP hardware-only parity layer
cargo test -p lgbm-compute --features rocm    # ROCm backend-level tests
```
There is no committed CI workflow that runs `cargo test` (`.github/workflows/`
only contains `release-python.yml`); parity/perf gating is run manually/via
Kaggle harnesses per the project's spike workflow (see `AGENTS.md` / project
memory notes on Kaggle CLI usage).

## Test File Organization

**Two layers per crate:**
1. **In-file unit tests**: `#[cfg(test)] mod tests { use super::*; ... }` at the
   bottom of the implementation file being tested, e.g.
   `crates/lgbm-core/src/error.rs`, `crates/lgbm-treelearner/src/data_partition.rs`
   (`mod tests` at line 639). Used for small, self-contained unit checks of a
   single function/type.
2. **Crate-level integration tests**: `crates/<crate>/tests/*.rs`, one file per
   test concern, e.g.:
   - `crates/lgbm-core/tests/`: `alias_resolution.rs`, `config_defaults.rs`,
     `config_validation.rs`, `seed_derivation.rs`
   - `crates/lgbm-dataset/tests/`: `bin_mapper_internals.rs`,
     `bin_storage_layout.rs`, `categorical_folding.rs`, `efb_grouping.rs`,
     `numeric_assignment.rs`, plus `fixtures/` and `golden/` subdirs
   - `crates/lgbm-model/tests/`: `model_text_roundtrip.rs`,
     `predict_leaf_parity.rs`, `predict_raw_parity.rs`, `predict_subrange.rs`,
     `predict_transform.rs`, plus `fixtures/`, `golden/`
   - `crates/lgbm-compute/tests/`: `capability.rs`, `cmp01_containment.rs`,
     `copy_subrow_parity.rs`, `cuda_on_device.rs`, `cuda_random_parity.rs`,
     `device_dataset_parity.rs`, `plane_intrinsic_smoke.rs`,
     `primitives_self.rs`, `rocm_backend_parity.rs`, `split_info.rs`
   - `crates/lgbm-treelearner/tests/`: `quantized_pipeline.rs`
   - `crates/oracle-harness/tests/`: the largest suite (~30 files) — see
     "Parity Testing" below.

**Naming:**
- Test file names describe the concern under test, not the module (`*_parity.rs`
  for cross-implementation comparisons, `*_roundtrip.rs` for serialize/deserialize
  symmetry, `*_internals.rs` for white-box structural checks).
- Test function names are descriptive sentences in `snake_case`, e.g.
  `abs_diff_within_boundary`, `compare_within_reports_first_offending_index`,
  `oracle_tol_is_1e_6` (`crates/oracle-harness/tests/comparator.rs`).

## Test Structure

```rust
#[test]
fn compare_within_reports_first_offending_index() {
    let rust = [1.0_f32, 2.0, 3.0, 4.0];
    let cpp = [1.0_f32, 2.0, 3.5, 4.5]; // index 2 is the first to diverge
    let err = compare_within(&rust, &cpp, ORACLE_TOL).unwrap_err();
    match err {
        Mismatch::ValueMismatch { index, .. } => assert_eq!(index, 2),
        other => panic!("expected ValueMismatch, got {other:?}"),
    }
}
```
(`crates/oracle-harness/tests/comparator.rs`)

- Tests favor explicit `match`/`matches!` on structured error/result enums over
  string matching, so a mismatch's *kind* (length vs value vs exact) is asserted,
  not just pass/fail.
- Hardware-gated tests use `#[cfg(feature = "rocm")]` on individual `#[test]` fns
  (`crates/oracle-harness/tests/on_device_admissibility.rs:390`) or a whole-file
  `#![cfg(feature = "rocm")]` gate (`on_device_e2e_ab_corpus.rs`,
  `on_device_subtract_residue.rs`) rather than `#[ignore]`, so ROCm tests simply
  don't compile/run without `--features rocm` instead of needing `--ignored`.

## Mocking

- No mocking framework is used. This is a numerics-heavy port; "mocking" the unit
  under test would defeat the parity contract. Instead, tests replay against
  **committed golden fixtures** captured from the real C++ reference (see below).

## Fixtures and Goldens — Numerical Parity Testing (core theme)

This is the project's defining testing pattern: prove the Rust implementation
reproduces real `lib_lightgbm` (C++) output, not just internal self-consistency.

**Tolerance contract (`crates/oracle-harness/src/comparator.rs`):**
```rust
/// The locked oracle comparison tolerance (D-02): `~1e-6` absolute, f32.
pub const ORACLE_TOL: f32 = 1e-6;
```
- `ORACLE_TOL` is explicitly distinguished from `lgbm_core::types::K_EPSILON`
  (`1e-15`), which is an *algorithm* constant, not a comparison tolerance — do
  not conflate the two when writing new parity assertions.
- Float comparisons use `abs_diff_within(a, b, tol)` (inclusive `<= tol`) via
  `compare_within(&rust_slice, &cpp_slice, tol)`, which returns
  `Result<(), Mismatch>`.
- Integer draws and exact-bit `f32` RNG draws are compared for **exact**
  equality, never within tolerance (see RNG parity test) — the CPU
  (`cubecl-cpu` f64-fold) path is the deterministic bit-exact anchor per
  project convention (CLAUDE.md); only the ROCm f32 path is held to the ~1e-6
  gate.

**`Mismatch` enum — first-divergence reporting, not aggregate diff:**
```rust
pub enum Mismatch {
    LengthMismatch { rust_len: usize, cpp_len: usize },
    ValueMismatch { index: usize, rust: f32, cpp: f32, abs_diff: f32, tol: f32 },
    ExactMismatch { index: usize, rust: String, cpp: String },
}
```
Every parity comparator returns the **first offending index**, not a
statistical summary — this is deliberate (localizes a divergence within a
large randomized golden set, D-14) and should be preserved in any new
comparator.

**Golden fixture layout:**
- Golden files live in `tests/golden/` and `tests/fixtures/` subdirectories per
  crate: `crates/lgbm-dataset/tests/golden/`, `crates/lgbm-dataset/tests/fixtures/`,
  `crates/lgbm-model/tests/golden/`, `crates/lgbm-model/tests/fixtures/`,
  `crates/oracle-harness/tests/fixtures/` (largest set — `advanced/`, `boosting/`,
  `categorical/`, `constraints/`, `dart/`, `determinism/`, `goss/`, `kernels/`,
  `learner/`, `metric/`, `predict_modes/`, `quantized/`, `rank/`, `rf/`).
- A generated `REFERENCE_MANIFEST.md` documents each committed golden set: what
  it covers, the exact training config used to capture it (seed, iteration
  count, thread count, determinism flags), and the file-by-file encoding
  (`crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md`). New golden
  captures must be documented in this manifest with the same level of detail
  (config used, encoding, known caveats/pitfalls).
- Goldens are captured against **real `lightgbm==4.6.0`** with a fixed, documented
  seed (e.g. `BOOSTING_ORACLE_SEED = 0x60057000`) and deterministic flags
  (`deterministic=true force_row_wise=true num_threads=1`) — reproducibility of
  the golden generation itself is part of the contract.
- Encodings used per data type, stated explicitly in the manifest table:
  - Model text: `save_model()` `%.17g` full-precision text dump.
  - Scores/metrics: raw `f64` bits, one value/line.
  - Gradients/hessians: `f32` bits (matches the project's `score_t = float`
    single-precision contract).
  - Bin indices / storage bytes: exact-bit comparison (`ExactMismatch`), never
    tolerance-based.

**Golden replay pattern (`crates/oracle-harness/tests/*_parity.rs`):**
- Each `*_parity.rs` file drives a real Rust component (learner, objective,
  metric, booster, kernel) against inputs recorded in the golden fixtures, then
  calls `compare_within` (or an exact-equality variant) against the committed
  C++ output, asserting `.is_ok()` or matching on `Mismatch`.
- `oracle-harness`'s own `[dependencies]` (library surface) stays free of any
  GPU/cubecl or downstream-crate coupling; every crate under test
  (`lgbm-core`, `lgbm-compute`, `lgbm-dataset`, `lgbm-model`, `lgbm-treelearner`,
  `lgbm-objective`, `lgbm-metric`, `lgbm-boosting`, `lgbm`) is pulled in only as
  `[dev-dependencies]`, each with an inline comment explaining why (isolation
  idiom, CMP-01 containment) — see `crates/oracle-harness/Cargo.toml`.

**Test taxonomy present in `oracle-harness/tests/`:**
- Kernel-level: `kernel_parity.rs`, `primitive_parity.rs`, `rng_parity.rs`.
- Split/partition: `best_split_parity.rs`, `partition_parity.rs`,
  `tree_mutation_parity.rs`.
- Objective/metric: `objective_parity_{binary,multiclass,rank,regression}.rs`,
  `metric_parity.rs`, `objective_common/` (shared helper module).
- Learner/boosting/predict: `learner_parity.rs`, `boosting_parity.rs`,
  `predict_parity.rs`, `score_updater_parity.rs`, `advanced_parity.rs`.
- Config/ingest: `config_drift.rs`, `rawcorpus_binning_config.rs`,
  `raw_bin_train_parity.rs`.
- Quantized-training mode: `quantized_parity.rs`.
- ROCm/on-device (hardware-only, `#[cfg(feature = "rocm")]`):
  `on_device_admissibility.rs`, `on_device_e2e_ab_corpus.rs`,
  `on_device_float_envelope_500k.rs`, `on_device_integer_anchor.rs`,
  `on_device_subtract_residue.rs`, `on_device_sync_count.rs`,
  `on_device_tie_break_parity.rs`, `on_device_tripwire_canary.rs`,
  `phase31_grad_score_ab.rs`, `resident_score_ab.rs`.
- Cross-cutting: `comparator.rs` (tests the comparator utility itself, not a
  parity test).

**Crate-boundary isolation testing:**
- `crates/lgbm-compute/tests/cmp01_containment.rs` is a dedicated test enforcing
  the CMP-01 rule (only `lgbm-compute` may depend on `cubecl`) — treat this file
  as the canonical place to add a new crate-boundary invariant check, alongside
  documenting the rule in the offending crate's `lib.rs` doc comment.

## Coverage

- No coverage tool/config detected (no `tarpaulin.toml`, `.codecov.yml`, or
  `cargo-llvm-cov` config at the repo root). Coverage is driven by the parity
  contract instead: a phase/plan is not considered done until its golden-file
  parity tests pass at the locked tolerance, per CLAUDE.md's "hard merge gate"
  framing (`cubecl-cpu` bit-exact; ROCm ~1e-6).

## Test Types

**Unit tests:** in-file `#[cfg(test)] mod tests` — narrow, single-function checks
(e.g. tolerance-boundary math, error-variant display formatting).

**Integration/parity tests:** crate `tests/*.rs` — the dominant test type in this
project (~1073 `#[test]` functions across the workspace). Most exercise a real
component against a committed C++ golden, not synthetic mocks.

**Hardware-gated tests:** `#[cfg(feature = "rocm")]` (file- or fn-level) — require
a real ROCm/HIP-capable GPU; excluded from a default `cargo test` run. Do not
convert these to `#[ignore]`; follow the existing feature-gate convention so they
are excluded at compile time, not just at run time.

## Common Patterns

**Tolerance-based float comparison:**
```rust
assert!(compare_within(&rust, &cpp, ORACLE_TOL).is_ok());
```

**Structured mismatch inspection:**
```rust
let err = compare_within(&rust, &cpp, ORACLE_TOL).unwrap_err();
match err {
    Mismatch::ValueMismatch { index, .. } => assert_eq!(index, 2),
    other => panic!("expected ValueMismatch, got {other:?}"),
}
```

**Error-boundary testing (construct + display):**
```rust
#[test]
fn config_error_constructs_and_displays() {
    let e = ConfigError::InvalidType { param: "num_leaves".to_string(), value: "abc".to_string() };
    // assert on e.to_string() / Debug formatting
}
```
(`crates/lgbm-core/src/error.rs`)

When adding a new parity test: add the golden fixture under the appropriate
crate's `tests/golden/` or `tests/fixtures/` directory, document it in the
relevant `REFERENCE_MANIFEST.md`, drive the real Rust path (not a mock), and
assert via `compare_within`/`Mismatch` at `ORACLE_TOL` (or exact-equality for
integer/bin-index data) — never introduce an ad-hoc tolerance constant.

---

*Testing analysis: 2026-07-09*
