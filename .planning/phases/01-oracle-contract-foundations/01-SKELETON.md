# Walking Skeleton — LightGBM-rs

**Phase:** 1
**Generated:** 2026-06-05

## Capability Proven End-to-End

> The smallest user-visible capability that exercises the full numerical-fidelity stack.

A developer can run the ported `Random` LCG (and construct a validated `Config`) and have the oracle harness prove the output matches a pinned C++ LightGBM 4.6 reference bit-for-bit across a RANDOMIZED, diverse set of inputs (many LCG seeds + randomized `N,K` straddling the `Sample` branch boundary, plus randomized config inputs), reading committed goldens with no C++ toolchain — i.e. the parity-proving spine (workspace → RNG/config → oracle comparison) is real, falsifiable, and validated over varied distributions rather than a single point.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Workspace shape | Virtual Cargo workspace (root `Cargo.toml` = `[workspace]`, no `[package]`), members under `crates/` + `xtask/`, `lgbm-*` naming | D-08; root is not a publishable crate, members split by responsibility |
| Edition / resolver / toolchain | edition 2024, `resolver = "3"`, pinned `rust-toolchain.toml` (channel 1.95.0) | FND-02; edition 2024 stable since 1.85, resolver 3 is its default |
| Initial crate set | `lgbm-core` (types/errors/RNG/config), `lgbm-compute` (Backend trait skeleton, no kernels), `oracle-harness` (comparator/fixtures/manifest/drift), `xtask` (dev-only regen) | D-09; minimal — dataset/model/treelearner/boosting/objective/metric/api/python crates added in the phases that introduce them |
| Numerical contract | `f32` (single-precision) end-to-end for scores/gradients/hessians/leaf-values; oracle tolerance ~1e-6 absolute on every backend; standard f32 accumulations (no integer-quantized / ordered-f64 strategy) | D-01/D-02/D-03/D-04; matches C++ `score_t`/`label_t` = `float` defaults — faithfulness over out-precisioning |
| RNG | Bit-exact port of C++ `Random` LCG over `u32` (`x = 214013*x + 2531011`, `wrapping_*`), `NextFloat = RandInt16()/32768.0f`, `Sample` both branches (streaming + `BTreeSet`) | FND-01; the single most parity-critical foundation — seed derivation, bagging, feature sampling, GOSS/DART all inherit it |
| Config | Single flat `Config` struct mirroring C++ `Config` 1:1 (same field names/defaults); verbatim alias table; `from_params` pipeline (seeds → members → CHECK validation → conflict mutations); hand-ported, drift-checker test (NOT codegen) | D-11/D-12/D-13; readable, cross-checkable, guarded against upstream drift |
| Errors | `thiserror` domain errors at the `lgbm-core` boundary; `anyhow` in harness/xtask/tests | FND-04; C++ `Log::Fatal` sites become typed `Result` errors, never panics on user input |
| Compute seam | All CubeCL usage confined behind one `lgbm-compute` `Backend` trait skeleton (no kernels this phase) | CMP-01; isolates CubeCL alpha churn from every crate above it |
| C++ reference + goldens | Built from the in-repo `LightGBM/` submodule (commit `195c26fc...`, VERSION 4.6.0.99) via CMake with `deterministic=true force_row_wise=true num_threads=1` + default `float` width; oracle inputs are a RANDOMIZED, diverse set derived from ONE recorded master seed (committed in the manifest, re-rollable) — many LCG seeds + randomized `N,K` straddling the `Sample` boundary for RNG, randomized in-scope/boundary/invalid params for config; the C++ reference is run ONCE over the set, the `(input → output)` pairs committed as fixtures; normal `cargo test` reads fixtures with NO C++ toolchain; regen is idempotent (same master seed → identical fixtures) | D-05/D-06/D-07/D-14, ORA-01/ORA-02; reproducible from the repo, version-locked, fidelity proven across varied distributions not a single point |

## Stack Touched in Phase 1

> Adapted for a numerical library (no UI/DB/HTTP). The "full stack" here is: workspace build → parity primitive (RNG/config) → oracle comparison against a pinned reference.

- [x] Project scaffold — virtual workspace, edition 2024, pinned toolchain, `Cargo.lock` committed, lint via `cargo build/test`
- [x] Parity primitive — bit-exact `Random` LCG (real arithmetic, not a stub)
- [x] Configuration layer — flat `Config` with alias resolution, seed derivation via the RNG, and typed validation across randomized in-scope/boundary/invalid inputs (D-14)
- [x] Oracle comparison — abs-diff ~1e-6 comparator (float comparisons) + RNG bit-for-bit golden comparison against the pinned C++ reference over a randomized, diverse golden set (D-14)
- [x] Reproducible reference — committed randomized golden set + pinned reference manifest (commit + flags + master seed + case count) + idempotent regen `xtask` deriving the set from one recorded master seed (documented full-stack regen command: `cargo run -p xtask -- regen`)

## Out of Scope (Deferred to Later Slices)

> Explicit so later phases do not re-litigate Phase 1's minimalism.

- Any dataset, binning, `BinMapper`, columnar store (Phase 2)
- Model text I/O and prediction (Phase 3)
- Any CubeCL kernels — `lgbm-compute` is a trait skeleton only (Phase 4)
- Tree learning, histograms, split finding (Phase 5)
- GBDT loop, objectives, metrics, boosting variants (Phases 6-7)
- Python bindings (Phase 8)
- Distributed/GPU/linear-tree/quantized-grad config params — present in the alias table but excluded from `IN_SCOPE_PARAMS` validation/exposure (v2 / out of scope)
- A CLI/example binary (the removed hello-world); optional, later
- Empirical CPU↔ROCm f32 transcendental parity validation (Phase 4/6)

## Subsequent Slice Plan

Each later phase adds one vertical, oracle-validated slice on top of this skeleton without altering its architectural decisions:

- Phase 2: bit-identical `BinMapper` + immutable columnar binned dataset (determinism root) — validated through the Phase 1 oracle harness
- Phase 3: load a C++-trained model and predict identically (predict parity before training exists)
- Phase 4: fill the `lgbm-compute` `Backend` trait with f32 histogram/split/score kernels (CPU then ROCm), both at ~1e-6
- Phase 5: histogram serial tree learner with per-split parity
- Phase 6: first end-to-end f32 ~1e-6 train→predict (GBDT spine + core objectives/metrics)
- Phase 7: parity-completing variants (GOSS/DART/RF, categorical, remaining objectives/metrics, SHAP, monotone)
- Phase 8: PyO3 Python bindings over the validated Rust facade
</content>
