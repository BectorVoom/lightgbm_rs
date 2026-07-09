# LightGBM-rs — Pure Rust LightGBM with CubeCL

## What This Is

A pure-Rust rewrite of Microsoft's LightGBM gradient-boosting library, built as a
Cargo workspace on the `cubecl` compute abstraction (CPU + AMD ROCm/CUDA GPU). It
targets ML practitioners and LightGBM users who want a memory-safe Rust
implementation that stays numerically faithful to the C++ reference. The Microsoft
C++ tree under `LightGBM/` is the read-only porting source; the deliverable is the
Rust crate(s) and a Python binding that mirrors the official `lightgbm` package.

## Core Value

For identical inputs and configuration, the Rust implementation must reproduce C++
LightGBM's outputs to within ~1e-6 absolute on every backend, using `f32`
end-to-end to match the C++ `score_t`/`label_t` = `float` defaults. The `cubecl-cpu`
f64-fold path is the deterministic anchor and hard merge gate (bit-exact vs real
`lib_lightgbm` 4.6 where the algorithm permits); ROCm/CUDA f32 is held to ~1e-6
against that anchor. Numerical fidelity at single precision is non-negotiable.

## Current Milestone: v1.0 C++ Feature-Parity Audit & Gap Closure

**Goal:** Systematically inventory every C++ LightGBM capability the Rust port
lacks (excluding the C-API), then implement the missing functionality to the
project's existing numerical-fidelity contract.

**Approach (research-first):** Phase 1 is a parity audit — parallel researchers
diff the read-only C++ reference (`LightGBM/`) against the Rust `crates/` and
produce a ranked gap inventory. Every later phase fills gaps the audit surfaces,
each gated by oracle-harness parity tests.

**Audit dimensions (C++ → Rust coverage):**
- Objectives & metrics (`src/objective`, `src/metric` → `lgbm-objective`, `lgbm-metric`)
- Config params & aliases (`include/LightGBM/config.h` → `lgbm-core` `Config`)
- Boosting & tree-learner features (DART/RF/GOSS, linear trees, monotone
  constraints, CEGB, forced splits, quantized gradient)
- Dataset / binning / model I/O (EFB, categorical handling, text/JSON format)

## Requirements

### Validated

<!-- Inferred from the existing mature codebase; confirmed working & relied upon. -->

- ✓ Histogram-based leaf-wise (best-first) serial tree learner — bit-exact vs `lib_lightgbm` 4.6 on both committed corpora
- ✓ GBDT outer boosting loop with DART / RF / GOSS variants and early stopping
- ✓ Core objectives (regression, binary, multiclass, rank/lambdarank, xentropy, custom)
- ✓ Feature binning (`BinMapper`), FeatureGroup, EFB, `Dataset`/`Metadata`
- ✓ Tree/ensemble model with text & JSON serialization and prediction
- ✓ CubeCL compute seam with CPU (f64-fold anchor) + ROCm/CUDA/wgpu backends
- ✓ PyO3 Python bindings mirroring the official `lightgbm` API surface
- ✓ oracle-harness C++ parity test infrastructure with committed goldens

### Active

<!-- v1.0 scope — populated from the parity audit in Phase 1, then filled below. -->

- [ ] Complete C++-vs-Rust feature-parity gap inventory (excluding C-API)
- [ ] Close audit-identified gaps in objectives & metrics
- [ ] Close audit-identified gaps in config params & aliases
- [ ] Close audit-identified gaps in boosting & tree-learner features
- [ ] Close audit-identified gaps in dataset / binning / model I/O

### Out of Scope

<!-- Explicit boundaries with reasoning. -->

- C-API surface parity (`src/c_api.cpp`, `LGBM_*` functions) — user decision for this milestone; the Rust facade + Python binding are the shipping surface, not a C ABI
- Distributed / MPI / socket networking (`src/network/`) — not a port target; single-node only
- Fully GPU-resident (no-host-round-trip) best-first grow loop — architecturally shelved (per-leaf sync floor), opt-in and known-slow; not a parity gap
- On-device CUDA path as default — remains opt-in via `LGBM_CUDA_ON_DEVICE` (slower than host-orchestrated)

## Context

- The codebase is **mature and near-complete** (2026-07-09 map): no `todo!()`/
  `unimplemented!()` stubs; most "unsupported" markers are proper typed-error
  branches. Known genuine gaps are narrow — categorical-feature GPU kernels
  (stubbed), quantized-gradient/stochastic rounding ("not yet implemented" in
  `config/mod.rs`), and unwired Python params recognized by C++ but not implemented.
- This milestone's value is a **systematic** sweep: without an inventory diffing
  the C++ surface against the Rust surface, "all unimplemented functions" cannot be
  scoped — so the audit is the first deliverable.
- Two read-only C++ references: `LightGBM/` (mainline, algorithm/API fidelity) and
  `LightGBM-release-4.6.0.99/` (AMD ROCm/HIP fork, GPU kernel parity baseline).
  Never edit or `git-add` either tree.
- No PROJECT.md existed before this milestone; only `.planning/codebase/` map docs.
  This PROJECT.md was bootstrapped from the codebase map + CLAUDE.md as the first
  formal GSD milestone.

## Constraints

- **Tech stack**: Pure Rust (edition 2024, toolchain 1.95.0), Cargo workspace,
  `cubecl` 0.10 for all compute — no raw CUDA/OpenCL. CMP-01: `lgbm-compute` is the
  only crate permitted to name `cubecl` types (enforced by a guard test).
- **Numerical**: `f32` end-to-end; ≤ ~1e-6 absolute vs C++ on CPU and ROCm/CUDA. CPU
  f64-fold path is the bit-exact anchor and hard merge gate.
- **Compatibility**: 100% behavioral compatibility with C++ LightGBM for in-scope
  APIs, configs, and internal specs (binning, split logic, model format).
- **Testing**: every new feature needs an oracle-harness parity test against a
  committed C++ golden; fixtures regenerated deterministically via `xtask regen`.
- **Error handling**: `thiserror` for structured domain errors at crate boundaries;
  `anyhow` for ergonomic propagation in app/dev layers.
- **Bindings**: Python interface mirrors the official `lightgbm` package API surface.
- **Hardware**: local GPU is a spoofed 8-CU APU — valid for parity gates, NOT for
  perf numbers; real discrete-CUDA perf validated remotely via Kaggle.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Parity-audit first, then fill gaps | "All unimplemented functions" is unscopeable without an inventory diffing C++ vs Rust surfaces | — Pending |
| Exclude C-API from this milestone | Rust facade + Python binding are the shipping surface; C ABI parity is not a user goal now | — Pending |
| Hold new features to the existing fidelity bar | Numerical fidelity is the non-negotiable Core Value; no reason to relax it for gap-fills | — Pending |
| Numerical contract is ~1e-6 vs f32 C++, not 1e-12 | 1e-12 is unachievable/meaningless against an f32 reference (Phase 1 discuss, 2026-06-05) | ✓ Good |
| `on_device_default()` stays false | Real-CUDA A/B found the fully-resident path 1.12–2.2× slower (per-leaf sync floor) | ✓ Good |

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
*Last updated: 2026-07-09 after bootstrapping v1.0 milestone (C++ Feature-Parity Audit & Gap Closure)*
