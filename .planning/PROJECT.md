# LightGBM-rs — Pure Rust LightGBM with CubeCL

## What This Is

A pure-Rust rewrite of Microsoft's LightGBM gradient-boosting library, built as a Cargo workspace and using the `cubecl` crate for compute and GPU acceleration. It targets ML practitioners and LightGBM users who want a memory-safe Rust implementation that runs on both CPU and AMD ROCm GPUs while remaining numerically faithful to the original. The Microsoft C++ implementation under `LightGBM/` is the read-only reference being ported; the deliverable is the Rust crate(s).

## Core Value

For identical inputs and configuration, the Rust implementation must reproduce the C++ LightGBM's outputs to within an absolute difference of **~1e-6 on every backend (CPU and ROCm)**, using `f32` (single-precision) data types end-to-end to match the C++ reference defaults (`score_t`/`label_t` = `float`). The CPU path — the `cubecl-cpu` f64-fold deterministic anchor — is the hard merge gate and, where the algorithm permits, achieves **bit-exact** parity with the C++ reference (e.g. binning, and the serial tree learner is bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora); the ROCm path (`cubecl-hip`, f32) is held to ~1e-6 against that anchor, with residual f32-vs-f64 accumulation gaps documented per phase. Numerical fidelity at single precision is the non-negotiable contract; everything else serves it.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] Dataset + binned columnar store (BinMapper, FeatureGroup, MultiValBin) matching C++ binning bit-for-bit — *Validated in Phase 2: dataset-binning-determinism-root (bit-exact ValueToBin, dense/CSR/CSC ingest, missing/categorical routing, EFB grouping, per-stage parity; default ingest unified onto the faithful single C++ `Dataset::Construct`)*
- [x] Model text format read/write + predict parity (Tree, GBDT model I/O, `%.17g`/`%g` formatter, raw/transformed/leaf/sub-range predict) — *Validated in Phase 3: tree-model-and-predict (PRD-01/02/03/06, DAT-08/09 all PASS within ~1e-6)*
- [x] CubeCL compute backend with switchable CPU ↔ ROCm (feature flag + runtime selection) — *Validated in Phase 4: compute-backend-cpu-first-integer-histograms-rocm. Backend trait with whole-kernel `construct_histograms`/`find_best_split`/`data_partition`/`subtract_histograms`; cubecl-cpu f64 fold is the BIT-EXACT deterministic anchor (D-04a proven across 25 launches + vs C++-order fold); cubecl-hip f32 path on real gfx1100 within a separate ~1e-6 gate vs the cpu anchor (best-effort D-03a, one documented ULP-scale gap in 04-ROCM-GAPS.md). CMP-01..05, ORA-04 satisfied. CUDA warp-level ops mapped onto CubeCL `Plane` API via the startup capability gate (asymmetric cpu/hip matrix).*

### Active

<!-- v1 = full single-machine parity. All hypotheses until shipped and oracle-validated. -->

- [ ] Cargo workspace split into loosely-coupled crates (core data, boosting, tree-learner, objectives/metrics, compute backend, Python bindings)
- [ ] GBDT training loop (gradient boosting) with ~1e-6 (f32) oracle parity
- [ ] DART, Random Forest, and GOSS boosting/sample strategies
- [ ] Histogram-based serial tree learner with split-gain scan, data partition, leaf splits
- [ ] Full objective-function set (regression, binary, multiclass, ranking, etc.)
- [ ] Full metric set (l1/l2/rmse, auc, ndcg, logloss, etc.)
- [ ] Monotone constraints and categorical-feature support
- [ ] Model train/predict producing outputs within ~1e-6 (f32) of C++ reference
- [ ] LightGBM model text format read/write compatibility (load a C++-trained model, predict identically)
- [ ] CubeCL compute backend with switchable CPU ↔ ROCm (feature flag and/or runtime selection)
- [ ] CUDA warp-level operations mapped onto CubeCL's `Plane` API
- [ ] Standard `f32` histogram + score-update accumulations on CPU and ROCm (no integer-quantized reduction strategy)
- [ ] Rust-native train/predict/Dataset/Booster API
- [ ] Python bindings mirroring the official `lightgbm` Python interface
- [ ] Oracle test harness comparing Rust vs C++ LightGBM outputs at ~1e-6 (f32), executed on ROCm
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
- **Current Rust crate:** Cargo workspace underway. Phases 1–4 complete (4/8 phases, 50%). Phase 1 (oracle-contract foundations), Phase 2 (`crates/lgbm-dataset`: bit-exact BinMapper, dense/sparse columnar bin store, missing/categorical encoding, EFB, metadata, dense/CSR/CSC ingestion), Phase 3 (`crates/lgbm-model`: Tree + GBDT model-text I/O, `%g` formatter, predict parity), and Phase 4 (`crates/lgbm-compute`: Backend trait + histogram/split/partition/subtract kernels, bit-exact on the cubecl-cpu f64 anchor, best-effort cubecl-hip f32 on gfx1100) — all backed by the `oracle-harness` + `xtask` C++ golden-capture pipeline. Next: Phase 5 (histogram serial tree learner + split finding, which consumes the Phase-4 compute kernels).
- **Key C++ subsystems to port:** `boosting/` (GBDT/DART/RF/GOSS), `treelearner/` (histograms, split finding, gradient discretizer), `io/` (Dataset, Bin, FeatureGroup, Metadata), `objective/`, `metric/`, model text serialization.
- **Numerical fidelity is the hardest part:** even at `f32` / ~1e-6, floating-point reductions are non-associative, so matching the C++ reference on ROCm vs a C++ CPU reference still requires care in binning and accumulation — not an afterthought. The target is single-precision parity, matching the C++ `float` defaults rather than a double-precision 1e-12 bound.

## Constraints

- **Tech stack**: Pure Rust, Cargo workspace, `cubecl` for compute — no raw CUDA/OpenCL. Use latest available crate versions.
- **Compatibility**: 100% behavioral compatibility with C++ LightGBM for in-scope APIs, configs, and internal specifications (binning, split logic, model format).
- **Numerical**: `f32` (single-precision) data types end-to-end, matching C++ `score_t`/`label_t` = `float` defaults; absolute output difference ≤ ~1e-6 vs C++ reference on **both** CPU and ROCm backends.
- **Hardware**: Tests validated on a **local ROCm GPU**; CubeCL `Plane` API used for warp-level ops.
- **Backends**: CPU and ROCm must be switchable (Cargo features and/or runtime configuration).
- **Error handling**: `thiserror` for structured domain errors at library boundaries; `anyhow` for ergonomic propagation in app/high-level layers.
- **Bindings**: Python interface must mirror the official `lightgbm` package API surface.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v1 = full single-machine parity (no distributed) | Cover all boosting types/objectives/metrics while deferring the large distributed surface | — Pending |
| `f32` single-precision end-to-end + ~1e-6 oracle on **every** backend incl. ROCm | Matches the C++ reference defaults (`score_t`/`label_t` = `float`); 1e-12 is unachievable/meaningless against an f32 reference, and f32 is the most faithful baseline | — Decided 2026-06-05 (Phase 1 discuss) |
| Standard `f32` accumulations (drop integer-quantized histograms) | At f32 / ~1e-6 the integer-quantization complexity buys nothing; standard f32 reductions keep the CubeCL CPU/ROCm path simple | — Decided 2026-06-05 (Phase 1 discuss) |
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
*Last updated: 2026-06-05 — Phase 2 (dataset + binning, determinism root) complete; dataset/binning requirement validated*
