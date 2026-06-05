# Project Research Summary

**Project:** LightGBM-rs — Pure-Rust LightGBM port on CubeCL (CPU + AMD ROCm) with Python bindings
**Domain:** Histogram-based gradient-boosting decision-tree library (faithful single-machine port of Microsoft LightGBM 4.6) with a switchable CPU/ROCm compute backend and a strict numerical-parity contract
**Researched:** 2026-06-05
**Confidence:** MEDIUM-HIGH (HIGH on stack/features/C++ subsystem mapping; MEDIUM on the achievability of literal 1e-12 parity on ROCm — the central project risk)

## Executive Summary

This is a faithful re-implementation of LightGBM's single-machine training/prediction engine in pure Rust, structured as a Cargo workspace, with the C++ LightGBM 4.6 tree under `LightGBM/` as a read-only reference. The genuine novelty is not ML capability — every feature is parity — but three cross-cutting properties: **memory safety** (Rust), a **portable CubeCL CPU↔ROCm backend** that replaces LightGBM's two separate CUDA and OpenCL codebases, and a **bit-determinism contract** across backends. Experts build this layer-by-layer with an abstract seam at every subsystem boundary; the Rust port maps each C++ abstract-base-class + string factory onto an `enum` + `trait`, and collapses the C++ `#ifdef USE_CUDA` device branching into a single `Backend` trait boundary so the boosting/tree-learner logic is device-agnostic.

The single most important finding cuts across all four research dimensions and must shape every phase: **the headline "1e-12 absolute parity on ROCm" contract is, taken literally, almost certainly unachievable for any code path that touches GPU floating-point reductions or transcendental functions** (`exp`/`log`/`pow`/`sigmoid`). LightGBM's *own* `deterministic=true` mode does not reproduce results across compilers, CPU instruction sets (FMA/AVX), or machines — the maintainers state this explicitly — and a GPU is a more divergent environment than "a different CPU." Yet the architecture research found a powerful resolution: LightGBM already ships an **integer-quantized gradient histogram path** (`GradientDiscretizer`), and **integer addition is associative and exact**, so histogram accumulation — and therefore the *tree structure* (split feature, split bin, topology, RNG-selected rows/features) — can be made **bit-identical across CPU and ROCm by construction**, independent of reduction order, thread count, or whether the CubeCL Plane path or sequential path runs. The recommended resolution is a **tiered oracle**: Tier A (bit-exact structural/RNG/bin/split parity on *all* backends), Tier B (≤1e-12 numeric on the deterministic single-threaded CPU path), Tier C (documented relaxed tolerance, e.g. ~1e-6 relative, on ROCm numeric outputs, *always paired with a Tier A same-tree structural check* so "numbers differ slightly" can never hide "a different model was trained"). This tiering must be signed off as a Key Decision in PROJECT.md before any kernel work — otherwise the project has an unfalsifiable acceptance criterion that will be cited as a failed requirement at every milestone.

The second cross-cutting theme is a **dependency-forced build order** that is non-negotiable because each layer must be bit-exact before anything above it can be validated: Config → BinMapper/Dataset → RNG → histogram tree-learner → GBDT loop → objectives/metrics → prediction → model text I/O → boosting/feature variants → Python bindings. Two foundations dominate risk — **binning** (`BinMapper::ValueToBin`, where one off-by-one mis-bins a row and cascades into a different tree) and the **histogram tree-learner** (the keystone, where FP summation order and the histogram-subtraction trick decide every split). A standalone, bit-exact port of LightGBM's hand-rolled 32-bit LCG `Random` (with `u32` wraparound and `f32` sampling) must precede any sampling code, since one wrong draw selects different rows and fails the oracle on iteration 1. Finally, **CubeCL is alpha (v0.10.0) with imminent breaking churn**; the mitigation is to isolate *all* CubeCL usage behind one `lgbm-compute` crate's `Backend` trait so an upgrade touches exactly one crate, pin every `cubecl-*` version exactly, commit `Cargo.lock`, and schedule upgrades as discrete tasks.

## Key Findings

### Recommended Stack

Pure Rust (edition 2024, toolchain ≥1.85), Cargo workspace, with all compute behind CubeCL. The numeric design is dominated by the parity contract: **f64 for every accumulation** (LightGBM accumulates histograms/scores in `double` even when gradients are `float`), a **custom columnar/bit-packed bin store** (ndarray cannot represent the 4/8/16-bit packed bin layout faithfully), and a **hand-written parser/writer for the LightGBM `.txt` model format** (bespoke line-oriented format with parity-critical field order and `%.17g`-style float formatting — serde cannot model it). See [STACK.md](STACK.md).

**Core technologies:**
- **`cubecl =0.10.0`** (pin exactly): single kernel source compiles to CPU/CUDA/HIP/wgpu; latest *stable* on crates.io (the book's `0.11.0` is unreleased `main`). Alpha — the #1 churn risk.
- **`cubecl-hip` via the `hip`/`rocm` feature** (NOT wgpu): the dedicated ROCm runtime; supports **f64 in hardware**, which wgpu/WebGPU does not — a structural reason ROCm must not route through wgpu.
- **f64 accumulators + custom bit-packed bin store**: hard requirements for 1e-12; raw gradients/labels stored at C++ width (`float` default), accumulated in f64.
- **`pyo3 0.28.3` + `numpy 0.28.0` (lockstep minors) + `maturin 1.13.3`**: Python bindings; the official `lightgbm` package is pure-Python-over-ctypes, so the thin sklearn wrapper layer can be reimplemented in Python over a Rust `Booster`, minimizing PyO3 surface. Avoid `pyo3 0.28.0/0.28.1` (yanked).
- **`thiserror 2.0.18`** at crate boundaries, **`anyhow 1.0.102`** at app/test/binding layers (never leak `anyhow` across the PyO3 boundary); **`proptest`/`criterion`/`approx`** for the oracle suite.
- **Oracle source**: fixture-based via the official Python `lightgbm` package (canonical reference, decouples from C++ build); shell out to CLI for config-file cases; FFI to `lib_lightgbm.so` only to probe internal intermediates.

### Expected Features

v1 = full single-machine parity, not a reduced product; "table stakes" = what a LightGBM user's workflow assumes exists. Every numeric feature carries the parity tax (bit-level reduction order + exact transform formulas). See [FEATURES.md](FEATURES.md).

**Must have (table stakes — the parity spine):**
- Config struct (~110 in-scope hyperparameters) + alias map + validation — gates everything.
- Dataset + BinMapper + Dense/Sparse bins — bit-identical binning is the foundation of all parity.
- Histogram-based serial tree-learner (construct → split-gain scan → partition → leaf-wise growth) — the keystone; highest FP-parity risk.
- GBDT loop + score updater + shrinkage + `boost_from_average` + early stopping.
- Core objectives (`regression`, `regression_l1`, `binary`, `multiclass`, `multiclassova`, `custom`) and core metrics (`l1`/`l2`/`rmse`/`binary_logloss`/`binary_error`/`auc`/`multi_logloss`).
- Prediction (raw / transformed / leaf index) + model text format read/write (load a C++-trained model, predict identically).
- Bagging + feature subsampling.

**Should have (completes parity / differentiators, P2):**
- GOSS, DART, RF; categorical splits + Exclusive Feature Bundling; remaining regression + cross-entropy objectives; ranking (`lambdarank`/`rank_xendcg` + `ndcg`/`map`, sharing `DCGCalculator`); monotone & interaction constraints; SHAP/contributions; feature importance; refit/continue; Python bindings mirroring the official API.
- The real differentiators vs C++: memory safety, the portable CubeCL CPU↔ROCm backend, and the bit-determinism contract.

**Defer (post-v1):**
- Quantized/discretized gradient *training* as a user feature (the int path is reused internally for determinism — see below), linear-tree leaves, text-file ingestion (CSV/TSV/LibSVM), binary dataset cache, Arrow.

**Explicit anti-features (excluded from v1):** distributed/network training, C ABI, CLI app, raw CUDA/OpenCL backends (kept as *design references* for CubeCL kernels, not build targets), R bindings, NVIDIA-specific tuning.

### Architecture Approach

A layered, acyclic Cargo workspace where dependencies flow strictly downward and **`lgbm-compute` is the only crate that names a CubeCL runtime** — everything above it talks to a `Backend` trait, so the boosting loop is device-agnostic and CPU-only testing is always possible. Kernels are written *once* generic over `R: Runtime`; `cpu.rs`/`rocm.rs` only pick the runtime. See [ARCHITECTURE.md](ARCHITECTURE.md).

**Major components (downward dependency order):**
1. **`lgbm-core`** — shared types, config enums, `thiserror` errors, reduction traits (depends on nothing; kept thin to avoid recompilation cascades).
2. **`lgbm-data`** — BinMapper/binning (the determinism root), FeatureGroup, Dense/Sparse/MultiVal bins, Metadata, loader; immutable after load.
3. **`lgbm-model`** — Tree storage + text/JSON model I/O (enables validating prediction parity independently of training).
4. **`lgbm-compute`** — the backend boundary: `Backend` trait + `#[cube]` kernels (histogram, split scan, score update, grad/hess), CPU↔ROCm selection. **The CubeCL churn containment boundary.**
5. **`lgbm-objective` / `lgbm-metric`** — grad/hess and eval kernels + DCG tables.
6. **`lgbm-treelearner`** — serial learner orchestrating compute kernels; gradient discretizer; monotone/categorical splits.
7. **`lgbm-boosting`** — GBDT/DART/RF loop, bagging/GOSS, score updater, early stopping.
8. **`lgbm-api`** — Rust-native `Booster`/`Dataset`/`train`/`predict` facade; enum+match factories replace C++ string `Create*`.
9. **`lgbm-python`** — PyO3 over the validated facade (cfg-gated; not in the default workspace build).

**The determinism architecture (the keystone design decision):** make **integer-quantized gradient accumulation the default compute path on every backend**. The `Backend` trait's histogram signatures take/return integers (`Buf<i32>` gradients → `Buf<i64>` histograms); because integer addition is associative and exact, CPU and ROCm produce the same bits by construction, and the histogram-subtraction trick stays exact. Float histograms exist only as a non-deterministic fast path that is *off* under the parity contract. Unavoidable float sums (final leaf outputs, score updates, metrics) use **f64 with fixed-shape pairwise reductions** (never floating atomic-add, whose order is nondeterministic). **Residual risk:** no source proves f64 transcendentals (exp/log) are bit-identical between a ROCm device math library and host libm; discretization (a rounding step) *may* absorb the last-ULP divergence, but this must be empirically validated, with the fallback of computing objective grad/hess on CPU and pushing only discretized integers to the GPU.

### Critical Pitfalls

The top theme is that parity is brittle and discontinuous — a 1-ULP difference in a gradient sum can flip a split and cascade into an entirely different tree — so parity must be validated at the granularity of bins → histograms → per-split gains → leaf outputs, never just final predictions. See [PITFALLS.md](PITFALLS.md).

1. **Literal 1e-12-on-ROCm is unachievable for GPU reduction/transcendental paths** — resolve with the **tiered oracle** (A: bit-exact structural on all backends; B: ≤1e-12 on deterministic CPU; C: documented relaxed tolerance on ROCm + Tier A same-tree check). Sign off as a Key Decision in Phase 0, before any kernel work.
2. **Non-bit-exact RNG port** — LightGBM's `Random` is a 32-bit LCG relying on `u32` wraparound, with `f32` `NextFloat` and a `std::set`-ordered `Sample` branch. Port standalone *first*, unit-test a 100k-draw sequence against C++ before any sampling code; verify call *order*, not just values.
3. **FP summation order in histogram/leaf reductions** — always compare against the C++ reference built/run with `deterministic=true, force_row_wise=true, num_threads=1`; accumulate CPU sums in the same sequential f64 order; integerize the GPU path. Don't parallelize the reduction until parity is locked.
4. **Histogram subtraction trick** — must reproduce *which* child is constructed directly vs derived (smaller-by-count child); integer subtraction makes it exact. Constructing both children directly diverges from the reference.
5. **`BinMapper::ValueToBin` off-by-ones** — port the `(r+l-1)/2` + `<=` search and NaN/`±0`/on-boundary placement *literally* (not Rust's `binary_search`); snapshot `bin_upper_bound_` and edge-case bin indices. Match the float parser too.
6. **Mis-configured oracle reference** — pin LightGBM version (4.6), compiler/flags (FMA/`-march`), `score_t`/`label_t` width, and the deterministic config as a checked-in manifest; generate multi-granularity goldens; stand up the harness *before* any CubeCL work.

(Further: split-gain `kEpsilon` positions, default-bin-skip in the scan, transcendental divergence across libm/HIP, subnormal/FTZ on ROCm, CPU-runtime-vs-HIP divergence within CubeCL, categorical bitset encoding, PyO3 dtype/contiguity/GIL hazards, and Cargo.lock/toolchain pinning.)

## Implications for Roadmap

Based on the research, the suggested phase structure follows the **dependency-forced build order** — each layer must be bit-exact before the next can be validated. Phase 0 is non-optional and unusually load-bearing because the oracle definition and RNG/lockfile foundations gate *all* later success criteria.

### Phase 0: Oracle Contract + Foundations
**Rationale:** Every later phase's acceptance criterion depends on this. The 1e-12-on-ROCm contract must be re-tiered before anyone writes a kernel against an unfalsifiable target.
**Delivers:** Tiered-oracle Key Decision (A/B/C) signed into PROJECT.md; pinned, containerized C++ reference build manifest (version 4.6, compiler/flags, `score_t` width, `deterministic=true force_row_wise=true num_threads=1` config); multi-granularity golden-snapshot harness; committed `Cargo.lock` + `rust-toolchain.toml`; `lgbm-core` (types, config enums, errors); standalone bit-exact `Random` LCG with a 100k-draw + `Sample(N,K)` conformance test.
**Avoids:** Pitfalls 1, 2, 10, 15 (unachievable contract, RNG mismatch, mis-configured reference, dependency churn).

### Phase 1: Dataset + Binning (the determinism root)
**Rationale:** Binning thresholds determine every split; if `BinMapper::find_bin` isn't bit-identical, no downstream parity is achievable.
**Delivers:** `lgbm-data` — `BinMapper::ValueToBin` (literal `(r+l-1)/2`+`<=`), Dense/Sparse/MultiVal bins, FeatureGroup, Metadata, missing/categorical encoding; validated against `bin_upper_bound_` + edge-case (NaN/±0/on-boundary) golden snapshots.
**Addresses:** Binned columnar store, BinMapper, missing-value handling (table stakes).
**Avoids:** Pitfall 5 (binning off-by-ones), float-parser divergence.

### Phase 2: Tree Model + Model Text I/O
**Rationale:** Storing `Tree` and parsing/emitting the C++ model text lets prediction parity be validated *independently of training* — a cheap early win and an explicit project requirement.
**Delivers:** `lgbm-model` — Tree nodes/splits/leaf outputs, `Tree::Predict`, hand-written `.txt` parser/writer with `%.17g` float formatting, JSON dump.
**Addresses:** Model text read/write, raw/leaf prediction.
**Research flag:** the `.txt` float-formatting + categorical-bitset serialization is fiddly; flag for `--research-phase`.

### Phase 3: Compute Backend (CPU-first, integer histograms)
**Rationale:** The `Backend` trait and integer-histogram kernels are the determinism linchpin and the CubeCL-churn containment boundary; build and cross-validate CPU vs ROCm on kernels *in isolation* before wiring into training.
**Delivers:** `lgbm-compute` — `Backend` trait, integer histogram construction, split-gain scan, score update, grad/hess; CPU runtime first, then HIP/ROCm as a second impl; capability-gating (`Plane::Ops`, f64, atomics) at startup; fixed-shape f64 reductions where integers don't apply.
**Uses:** `cubecl =0.10.0` / `cubecl-hip` / `cubecl-cpu` (STACK.md).
**Implements:** the determinism architecture (integer accumulation = bit-identical across backends).
**Research flag:** HIGH — CubeCL alpha API churn, ROCm f64/atomic capability gaps, CPU-runtime-vs-HIP divergence (Pitfalls 9, 11, 12). Needs `--research-phase` and a ROCm validation gate.

### Phase 4: Tree Learner + Split Finding
**Rationale:** The keystone algorithm; the hardest FP-parity subsystem. Lock CPU summation order and split-gain math before kernelizing further.
**Delivers:** `lgbm-treelearner` — serial learner orchestrating compute kernels, split-gain scan (`kEpsilon` positions, L1/L2/smoothing branches), data partition, histogram-subtraction trick (smaller-child selection), default-bin-skip, gradient discretizer, leaf-wise growth.
**Addresses:** Histogram serial tree-learner, leaf-wise growth, numerical splits (table stakes).
**Avoids:** Pitfalls 3, 4, 6, 7 (summation order, subtraction trick, split-gain constants, default-bin skip).
**Research flag:** moderate — split-gain compile-time-flag matrix is a silent-divergence surface.

### Phase 5: GBDT Loop + Core Objectives/Metrics + Bagging
**Rationale:** First end-to-end 1e-12 path; the simplest boosting variant proves the spine before adding GOSS/DART/RF.
**Delivers:** `lgbm-boosting` (GBDT loop, score updater, shrinkage, `boost_from_average`, early stopping, bagging + feature subsampling) + `lgbm-objective`/`lgbm-metric` core sets.
**Addresses:** GBDT loop, core objectives/metrics, bagging, early stopping (table stakes).
**Avoids:** Pitfall 8 (transcendentals — decide CPU-vs-GPU objective residency; default objectives on CPU/host libm).
**Research flag:** moderate — transcendental (`exp`/`log`/`Pow`/`sigmoid`) parity and compounding drift over rounds.

### Phase 6: Parity-Completing Variants
**Rationale:** Each is a thin or isolable addition once the spine is bit-exact; ordered by RNG/structure dependency.
**Delivers:** GOSS (after deterministic gradient sort), DART + RF (thin GBDT subclasses; DART needs `drop_seed`), categorical splits + EFB, remaining objectives/metrics, ranking (`lambdarank`/`ndcg`/`map` + DCGCalculator + query metadata), monotone/interaction constraints, SHAP/contributions, feature importance, refit.
**Avoids:** Pitfall 13 (categorical bitset encoding).

### Phase 7: Python Bindings
**Rationale:** A translation layer over a validated `lgbm-api`; build last.
**Delivers:** `lgbm-api` facade + `lgbm-python` (PyO3) mirroring the official `lightgbm` surface; sklearn wrappers reimplemented in Python over the Rust `Booster`.
**Avoids:** Pitfall 14 (numpy copies/contiguity/GIL/dtype; handle both f32 and f64; `allow_threads` around training; return owned arrays).

### Phase Ordering Rationale
- **Dependency-forced order** (Config → BinMapper/Dataset → RNG → histogram learner → GBDT loop → objectives/metrics → prediction → model I/O → variants → Python): each layer must be bit-exact before anything above it can be validated, and the oracle harness grows with each layer rather than being deferred to the end.
- **Binning and the histogram learner are isolated as their own phases** because they are the two highest-leverage parity risks; everything downstream inherits their bits.
- **The compute backend is CPU-first and isolated in one crate** so most logic is validated without a GPU and CubeCL alpha churn touches exactly one crate.
- **Variants (GOSS/DART/RF/categorical/ranking/constraints) come after the GBDT spine is green** because they are thin subclasses or isolable paths — cheap once the foundation holds — while quantized-gradient *training* and linear-tree are deferred post-v1 as parallel learner code paths.

### Research Flags

Phases likely needing `--research-phase` during planning:
- **Phase 3 (Compute Backend):** HIGH — CubeCL is alpha with imminent breaking changes; ROCm f64/atomic capability gaps, FTZ/subnormal behavior, and CPU-runtime-vs-HIP divergence are sparsely documented and must be empirically validated on the local ROCm GPU.
- **Phase 2 (Model Text I/O):** `.txt` float formatting (`%.17g`) and categorical-bitset serialization are exacting and under-documented.
- **Phase 4 (Tree Learner)** and **Phase 5 (Objectives):** moderate — the split-gain compile-time-flag matrix and transcendental parity each hide silent-divergence surfaces worth a focused pass.

Phases with standard, well-mapped patterns (lighter research):
- **Phase 0/1 (Foundations/Binning):** the C++ reference (`Random`, `BinMapper`) is read directly and ported literally — well-specified, not novel.
- **Phase 6 (Variants):** thin subclasses of an already-validated spine, each with a clear C++ reference.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Crate versions verified against crates.io/PyPI/Context7 on 2026-06-05; only MEDIUM on CubeCL determinism/f64 behavior on ROCm (alpha software). |
| Features | HIGH | Grounded directly in the C++ reference subsystems under `LightGBM/src` and `include/`; objective/metric registries verified from `Create*` factories. |
| Architecture | MEDIUM-HIGH | HIGH on crate decomposition + C++ mapping + CubeCL primitives; MEDIUM on the cross-backend determinism strategy — integer-quantized approach is grounded in LightGBM's own `GradientDiscretizer`, but 1e-12 on ROCm must be empirically validated. |
| Pitfalls | HIGH | HIGH on numerical/RNG/PyO3 pitfalls (verified against C++ source + issue tracker); MEDIUM on CubeCL/ROCm specifics (alpha, evolving docs). |

**Overall confidence:** MEDIUM-HIGH. The *port* is well-understood and well-referenced; the *risk* is concentrated entirely in whether ROCm can meet the parity bar, which the tiered-oracle + integer-histogram strategy is designed to make tractable but cannot guarantee without empirical validation on hardware.

### Gaps to Address

- **f64 transcendental parity (CPU↔ROCm):** No source proves bit-identical `exp`/`log` between ROCm device math and host libm. *Handle:* empirically test in Phase 3/5; if discretization doesn't absorb the divergence, fall back to CPU-resident objective grad/hess and push only discretized integers to the GPU.
- **Literal 1e-12 on ROCm numeric outputs:** Likely unachievable for GPU reduction/transcendental paths. *Handle:* the tiered-oracle Key Decision in Phase 0 (Tier A bit-exact structural on all backends; Tier C documented relaxed tolerance + same-tree check on ROCm).
- **CubeCL API stability:** Alpha v0.10.0 with an unreleased 0.11 already in the book. *Handle:* pin exactly, isolate behind `lgbm-compute`, commit `Cargo.lock`, schedule upgrades as discrete spikes at milestone boundaries.
- **Reference `score_t` width:** `float` by default but `double` under a build flag; structural mismatch if assumed wrong. *Handle:* pin and document in the Phase 0 reference manifest; match the reference's actual typedef where it rounds.
- **CubeCL CPU-runtime vs HIP divergence within the toolchain:** "kernel correct on CPU runtime" does not imply correct on ROCm. *Handle:* treat CPU-runtime parity and ROCm parity as separate test gates; run the oracle on both backends.

## Sources

### Primary (HIGH confidence)
- `LightGBM/src/{boosting,treelearner,objective,metric,io}/`, `include/LightGBM/{config.h,bin.h,meta.h,tree.h,utils/random.h}` — direct C++ reference: objective/metric registries, `Random` LCG, `BinMapper::ValueToBin`, `kEpsilon`, histogram-subtraction, `GradientDiscretizer`, model text format.
- `.planning/codebase/{ARCHITECTURE,STRUCTURE,STACK,CONVENTIONS,CONCERNS,INTEGRATIONS,TESTING}.md` — mapped C++ subsystem boundaries, GPU-relevant flags, numerical-fidelity hazards.
- crates.io / PyPI (2026-06-05) — `cubecl 0.10.0`, `pyo3 0.28.3` (0.28.0/0.28.1 yanked), `numpy 0.28.0`, `ndarray 0.17.2`, `proptest 1.11.0`, `criterion 0.8.2`, `thiserror 2.0.18`, `anyhow 1.0.102`, `maturin 1.13.3`.
- Context7 `/tracel-ai/cubecl` (v0.10) — runtime-generic kernels, `Plane::Ops` feature query, `plane_sum`, comptime fallback, `Feature::Type(Elem::Float(FloatKind::F64))` capability check, feature→crate mapping.
- LightGBM issues #6683 (cross-machine non-reproducibility), #3372 (compounding FP error), #6320 + Parameters docs (`deterministic`/`force_*_wise`).
- PyO3 / rust-numpy docs — contiguity/`as_slice`, `allow_threads` GIL release, dtype handling.

### Secondary (MEDIUM confidence)
- CubeCL book/README and `cubecl-hip(-sys)` releases — alpha "expect breaking changes," HIP build-script churn, ROCm 7.x version coupling.
- f64-on-wgpu limitation (WebSearch + CubeCL platform notes); GPU-vs-CPU libm precision studies (LLVM GSoC 2025, ACM C-math-on-GPUs, arXiv 2408.05148); FTZ/denormal notes (NVIDIA, AMD ROCm blogs).

### Tertiary (LOW confidence — needs validation)
- Empirical 1e-12 achievability on ROCm for f64 reductions/transcendentals — *no source guarantees it; validate on hardware in Phase 3/5.*

---
*Research completed: 2026-06-05*
*Ready for roadmap: yes*
