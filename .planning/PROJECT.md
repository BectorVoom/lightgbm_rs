# LightGBM-rs — Pure Rust LightGBM with CubeCL

## What This Is

A pure-Rust rewrite of Microsoft's LightGBM gradient-boosting library, built as a Cargo workspace and using the `cubecl` crate for compute and GPU acceleration. It targets ML practitioners and LightGBM users who want a memory-safe Rust implementation that runs on both CPU and AMD ROCm GPUs while remaining numerically faithful to the original. The Microsoft C++ implementation under `LightGBM/` is the read-only reference being ported; the deliverable is the Rust crate(s).

## Core Value

For identical inputs and configuration, the Rust implementation must reproduce the C++ LightGBM's outputs to within an absolute difference of **1e-12 on every backend (CPU and ROCm)**. Numerical fidelity is the non-negotiable contract; everything else serves it.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. The Rust deliverable currently contains only a hello-world scaffold, so nothing is validated yet. -->

(None yet — ship to validate)

### Active

<!-- v1 = full single-machine parity. All hypotheses until shipped and oracle-validated. -->

- [ ] Cargo workspace split into loosely-coupled crates (core data, boosting, tree-learner, objectives/metrics, compute backend, Python bindings)
- [ ] Dataset + binned columnar store (BinMapper, FeatureGroup, MultiValBin) matching C++ binning bit-for-bit
- [ ] GBDT training loop (gradient boosting) with 1e-12 oracle parity
- [ ] DART, Random Forest, and GOSS boosting/sample strategies
- [ ] Histogram-based serial tree learner with split-gain scan, data partition, leaf splits
- [ ] Full objective-function set (regression, binary, multiclass, ranking, etc.)
- [ ] Full metric set (l1/l2/rmse, auc, ndcg, logloss, etc.)
- [ ] Monotone constraints and categorical-feature support
- [ ] Model train/predict producing outputs within 1e-12 of C++ reference
- [ ] LightGBM model text format read/write compatibility (load a C++-trained model, predict identically)
- [ ] CubeCL compute backend with switchable CPU ↔ ROCm (feature flag and/or runtime selection)
- [ ] CUDA warp-level operations mapped onto CubeCL's `Plane` API
- [ ] Bit-deterministic reductions (ordered / f64-accumulated histogram + score updates) so ROCm also meets 1e-12
- [ ] Rust-native train/predict/Dataset/Booster API
- [ ] Python bindings mirroring the official `lightgbm` Python interface
- [ ] Oracle test harness comparing Rust vs C++ LightGBM outputs at 1e-12, executed on ROCm
- [ ] `thiserror` domain error types at crate boundaries; `anyhow` in application/test layers

### Out of Scope

<!-- Explicit boundaries with reasoning. -->

- Distributed / network (MPI-style) training — deferred beyond v1; large surface, not needed to prove the architecture
- C ABI (`c_api.h` / `LGBM_*`) parity — Rust-native + Python cover v1 consumers; revisit if C/R clients are needed
- CLI app (`lightgbm` command-line, config-file driven) — not required for v1 compatibility goals
- Raw CUDA / OpenCL backends — superseded by CubeCL per project mandate
- R bindings — out of scope
- NVIDIA-CUDA-specific tuning — ROCm is the mandated GPU test target

## Context

- **Reference codebase mapped:** `.planning/codebase/` documents the Microsoft C++ LightGBM core (ARCHITECTURE, STRUCTURE, STACK, CONVENTIONS, CONCERNS, INTEGRATIONS, TESTING). GPU-relevant subsystems are flagged there. The C++ tree is read-only reference, not a build target.
- **Current Rust crate:** greenfield — `src/main.rs` is hello-world, `Cargo.toml` declares only `cubecl = "0.10.0"`, edition 2024.
- **Key C++ subsystems to port:** `boosting/` (GBDT/DART/RF/GOSS), `treelearner/` (histograms, split finding, gradient discretizer), `io/` (Dataset, Bin, FeatureGroup, Metadata), `objective/`, `metric/`, model text serialization.
- **Numerical fidelity is the hardest part:** floating-point reductions are non-associative, so hitting 1e-12 on ROCm vs a C++ CPU reference requires deliberate determinism in binning and reduction ordering — not an afterthought.

## Constraints

- **Tech stack**: Pure Rust, Cargo workspace, `cubecl` for compute — no raw CUDA/OpenCL. Use latest available crate versions.
- **Compatibility**: 100% behavioral compatibility with C++ LightGBM for in-scope APIs, configs, and internal specifications (binning, split logic, model format).
- **Numerical**: Absolute output difference ≤ 1e-12 vs C++ reference on **both** CPU and ROCm backends.
- **Hardware**: Tests validated on a **local ROCm GPU**; CubeCL `Plane` API used for warp-level ops.
- **Backends**: CPU and ROCm must be switchable (Cargo features and/or runtime configuration).
- **Error handling**: `thiserror` for structured domain errors at library boundaries; `anyhow` for ergonomic propagation in app/high-level layers.
- **Bindings**: Python interface must mirror the official `lightgbm` package API surface.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v1 = full single-machine parity (no distributed) | Cover all boosting types/objectives/metrics while deferring the large distributed surface | — Pending |
| Strict 1e-12 oracle on **every** backend incl. ROCm | Numerical fidelity is the product's core value; no backend gets a relaxed tolerance | — Pending |
| Bit-deterministic reductions to meet GPU tolerance | FP non-associativity otherwise breaks 1e-12 on ROCm; needs ordered/f64 accumulation by design | — Pending |
| Rust-native API + Python bindings (no C ABI / CLI in v1) | Covers v1 consumers without the C-ABI/CLI surface | — Pending |
| CubeCL `Plane` API for CUDA warp operations | Project mandate; portable across CPU/ROCm without raw CUDA | — Pending |
| Cargo workspace with crate-per-responsibility | Maintainability, loose coupling, clear separation of concerns | — Pending |
| `thiserror` at boundaries, `anyhow` at app layer | Precise error matching for the library, ergonomic propagation for apps/tests | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-05 after initialization*
