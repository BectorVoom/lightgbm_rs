# Stack Research

**Domain:** Pure-Rust rewrite of LightGBM (gradient-boosted decision trees) with a switchable CPU / AMD ROCm compute backend via CubeCL, plus Python bindings. Hard constraint: 1e-12 absolute output parity vs C++ LightGBM 4.6 on every backend.
**Researched:** 2026-06-05
**Confidence:** HIGH for crate versions (verified against crates.io / Context7 / official docs, 2026-06-05). MEDIUM for CubeCL determinism/f64 behavior on ROCm (alpha software, version churn risk — see warnings). MEDIUM for Python-API mirroring effort estimate.

---

## Executive Summary (read this first)

1. **`cubecl = "0.10.0"` is correct and current.** 0.10.0 is the latest *published stable* (crates.io, released 2026-05-07; the entire 0.10 line before it was `0.10.0-pre.N` prereleases). There is **no 0.11 on crates.io yet** — the CubeCL *book* already shows `version = "0.11.0"` in examples, which is the unreleased `main` branch running ahead of the registry. Pin exactly and treat upgrades as breaking events.
2. **AMD ROCm = the `cubecl-hip` crate via the `hip` (alias `rocm`) feature — NOT wgpu.** An earlier ambiguous source suggested routing ROCm through `wgpu`; that is wrong. CubeCL has a *dedicated* HIP runtime that compiles to AMD's HIP/ROCm C++ compiler, separate from the Vulkan-via-wgpu path. Use `hip` for the mandated ROCm test target.
3. **CubeCL is officially in alpha.** This is the single largest risk to the project. Pin every CubeCL crate to an exact version, commit `Cargo.lock`, and budget for breaking API changes on each minor bump.
4. **The 1e-12 oracle dominates the numeric design.** Use **f64 everywhere accumulation happens** (LightGBM's C++ uses `double` for histogram sums and scores even when gradients are `float`). f64 on GPU is the determinism risk: CUDA and HIP support f64 in hardware; **wgpu/WebGPU does not reliably support f64**, which is a second reason ROCm must go through `cubecl-hip`, not wgpu. Determinism requires *ordered/fixed-layout* reductions, not just f64 width.
5. **Don't use `cubecl`'s built-in `plane_sum` for the oracle-critical reductions without a determinism wrapper.** `plane_sum` maps to CUDA `warpReduceSum` / HIP `__shfl` / wgpu `subgroupAdd`, whose summation *order* is implementation-defined. For 1e-12 parity you need a reduction whose addition order is fixed and identical to the C++ reference, or you must reproduce C++'s exact ordering. Plane ops are still useful where the result is bit-stable (e.g. ballots/counts) — see Architecture notes.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Rust (edition 2024) | toolchain ≥ 1.85 | Implementation language | Mandated. Edition 2024 already in `Cargo.toml`. Note: `proptest 1.11` needs rustc ≥ 1.85, `numpy 0.28` needs ≥ 1.83 — edition 2024 implies a new enough toolchain. |
| `cubecl` | `=0.10.0` | Compute kernel language + multi-backend runtime (CPU, CUDA, HIP/ROCm, wgpu) | Mandated. Latest *stable* on crates.io (2026-05-07). Single kernel source compiles to all backends; `Plane` API exposes warp/subgroup ops portably. **Pin exactly** — alpha, fast-moving. |
| `cubecl-hip` (via `cubecl` `hip`/`rocm` feature) | matches `cubecl` 0.10.0 | AMD ROCm/HIP GPU runtime — the mandated GPU test target | Dedicated HIP runtime compiling to AMD's HIP C++ compiler. Supports f64 in hardware (unlike wgpu), required for the 1e-12 contract. |
| `cubecl-cpu` (via `cubecl` `cpu` feature) | matches `cubecl` 0.10.0 | CPU runtime (LLVM/Rust-compiled, SIMD where available) | First-class CubeCL target. On CPU `PLANE_DIM == 1` (no warps), so plane-based kernels degrade to scalar — convenient for a deterministic CPU reference path. |
| `serde` + `serde_json` | `serde 1.x` (latest 1.0.x), `serde_json 1.x` | Config (de)serialization, internal state | Standard. **Not** for the LightGBM model text format (see below) — that is a bespoke line-oriented format, hand-written parser/writer required. |

### Numeric / Data Representation

| Technology | Version | Purpose | Why / Determinism note |
|------------|---------|---------|------------------------|
| **Custom columnar / binned store** | n/a (build it) | `Dataset`, `BinMapper`, `FeatureGroup`, `MultiValBin` | **Recommended over ndarray/nalgebra for the core.** LightGBM's data layout is bespoke: bit-packed bins (4/8/16-bit), feature groups, multi-val sparse bins. To match C++ binning *bit-for-bit* you must control memory layout directly. ndarray's dense `Array2<T>` cannot represent the packed bin layout faithfully. |
| `f64` (not `f32`) for all accumulators | — | Histogram sums, gradient/hessian sums, leaf output, score updates | **Hard requirement for 1e-12.** C++ LightGBM accumulates histograms and scores in `double` even though `score_t`/`label_t` default to `float`. Mirror this exactly: store gradients/labels as the same width C++ uses (`float` by default per `meta.h`), but accumulate in `f64`. |
| `bytemuck` | latest 1.x | Zero-copy reinterpret of packed bins / GPU byte buffers | CubeCL kernels exchange `&[u8]`; `bytemuck` gives safe `cast_slice` for the columnar store ↔ device transfer. |
| `ndarray` | `0.17.2` | **Only** at the Python/NumPy boundary (predict I/O, feature matrices) | Use ndarray *narrowly* for interop with `rust-numpy` (which speaks `ndarray`), not as the internal training data structure. |
| `nalgebra` | (avoid) | — | **Not needed.** LightGBM is not linear-algebra-heavy; the only dense LA is linear-tree leaf fitting (C++ uses Eigen). A small hand-rolled normal-equations solve, or `nalgebra` *scoped to that one module*, suffices. Do not adopt it project-wide. |

### Python Bindings

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `pyo3` | `0.28.3` | Rust↔CPython bindings | Latest stable (2026-04-02). Note 0.28.0/0.28.1 were **yanked** — use ≥ 0.28.2. ABI3/stable-ABI wheels supported. |
| `numpy` (rust-numpy) | `0.28.0` | NumPy array interop (`PyArray`, `PyReadonlyArray`) | **Minor version MUST track PyO3's** — `numpy 0.28` is built for `pyo3 0.28`. This lockstep is the #1 binding-version pitfall. |
| `maturin` | `1.13.3` | Build/publish the extension wheel | De-facto standard for PyO3 wheels; `maturin develop` for local iteration, `maturin build --release` for distribution. Configure via `pyproject.toml` + `[tool.maturin]`. |

To mirror the official `lightgbm` Python API: expose `Dataset`, `Booster`, `train()`, `cv()`, and the `LGBMClassifier`/`LGBMRegressor`/`LGBMRanker` scikit-learn wrappers. The official package is **pure-Python over ctypes** (no compiled extension of its own), so its public surface is plain Python — you can reimplement the thin sklearn-wrapper layer *in Python* on top of your Rust `Booster`, and only push the hot train/predict/Dataset path into Rust via PyO3. This minimizes PyO3 surface and maximizes API fidelity.

### Error Handling

| Technology | Version | Purpose | Boundary pattern |
|------------|---------|---------|------------------|
| `thiserror` | `2.0.18` | Structured domain errors at library/crate boundaries | One `#[derive(Error)]` enum per crate (e.g. `DatasetError`, `BoostingError`, `BackendError`). `2.x` is current and stable; no API change needed vs 1.x for typical use. |
| `anyhow` | `1.0.102` | Ergonomic propagation in app/test/binding layers | Use in the oracle harness, examples, and `main`. At the PyO3 boundary, convert your `thiserror` enums into `PyErr` (impl `From<MyError> for PyErr`) — do **not** leak `anyhow::Error` across the FFI boundary. |

### Testing / Oracle / Benchmarks

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `proptest` | `1.11.0` | Property-based testing (binning invariants, split-gain monotonicity, serialization round-trip) | Standard Rust property tester. Use to fuzz Dataset construction and model-format round-trips against invariants. |
| `criterion` | `0.8.2` | Statistical benchmarks (histogram build, split scan, predict) | Standard. Bench CPU vs ROCm backends; track regressions. |
| `approx` | latest 0.5.x | `assert_abs_diff_eq!` with `epsilon = 1e-12` | The literal oracle assertion macro. Cleaner than hand-rolled tolerance checks. |
| Oracle harness → see below | — | Compare Rust output vs real C++ LightGBM at 1e-12 | **Recommended approach: shell out to the official Python `lightgbm` package** (or the CLI) to generate reference outputs/models, stored as fixtures, then compare. See rationale below. |

**Oracle invocation — recommended approach (in priority order):**

1. **Fixture-based via Python `lightgbm` (RECOMMENDED).** In a `build.rs`-free test setup, run the real `lightgbm` PyPI package (the C++ lib under ctypes) to train models and dump predictions + model text files into `tests/fixtures/`. Rust tests load the fixtures and assert ≤ 1e-12. *Why:* decouples the oracle from the C++ build, gives reproducible golden files, and the Python package is the canonical reference API you're mirroring anyway.
2. **Shell out to the `lightgbm` CLI** for cases needing exact CLI/config-file behavior. Capture stdout/model files; compare. *Why:* avoids FFI complexity; the CLI is a stable text interface.
3. **FFI to `lib_lightgbm.so` via `bindgen`/`cc`** — *only if* you need to probe internal intermediate values (e.g. raw histogram sums) that neither the CLI nor Python expose. *Why last:* requires building the C++ lib (CMake ≥ 3.28, vendored submodules), adds a heavy native build dependency to the test pipeline, and the public surface usually suffices. Reserve for deep-dive parity debugging, not the default loop.

### Serialization — LightGBM Model Text Format

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Hand-written parser/writer** | n/a (build it) | Read/write LightGBM's `.txt` model format (and predict identically from a C++-trained model) | The model format is a **bespoke line-oriented key=value + per-tree block format**, not JSON/serde. C++ writes it manually in `gbdt_model_text.cpp`/`tree.cpp`. You must reproduce field order, float formatting, and precision exactly. serde cannot model this; write an explicit parser/serializer. |
| `serde_json` | 1.x | The *separate* JSON dump (`dump_model`) | LightGBM also offers a JSON model dump (the C++ uses vendored `json11`). If you implement that, `serde_json` is fine since JSON structure is well-defined. The primary `.txt` format is the parity-critical one. |
| Float formatting | — | Match C++ `%.17g`-style output | Parity of the *written* model requires matching C++'s float-to-string. Rust's default `{}`/`ryu` formatting differs from C++ `printf`. Plan a dedicated float-formatting routine (likely `%.17g` via a `format!`-equivalent) validated against fixtures. **Flagged as a real pitfall.** |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo` workspace | Multi-crate layout | Crate-per-responsibility: `lgbm-core` (Dataset/Bin), `lgbm-boosting` (GBDT/DART/RF/GOSS), `lgbm-treelearner`, `lgbm-objective`, `lgbm-metric`, `lgbm-backend` (CubeCL), `lgbm-io` (model text), `lgbm-py` (PyO3). |
| `Cargo.lock` (committed) | Pin transitive deps | **Mandatory** given CubeCL alpha churn. Commit it for the lib too (not just bins). |
| `cargo-nextest` | Faster test runs for the large oracle suite | Optional but recommended for fixture-heavy suites. |
| ROCm toolkit | HIP backend builds/tests | `cubecl-hip` requires a local ROCm/HIP install to compile+run the AMD path. CI must have a ROCm GPU (matches the project's "tests validated on local ROCm GPU" constraint). |

## Installation

```toml
# Cargo.toml (workspace root or lgbm-backend crate)
[dependencies]
# Compute — pin exactly (alpha, fast-moving). Pick features per build.
cubecl = { version = "=0.10.0", default-features = false }

# Numeric / interop
bytemuck   = "1"
ndarray    = "0.17.2"   # only at the NumPy boundary

# Error handling
thiserror  = "2.0.18"
anyhow     = "1.0.102"  # app/test/binding layers only

# Serialization (JSON dump path only; .txt format is hand-written)
serde      = { version = "1", features = ["derive"] }
serde_json = "1"

[features]
# Backend selection is COMPILE-TIME via features that forward to cubecl.
cpu  = ["cubecl/cpu"]
rocm = ["cubecl/hip"]      # AMD ROCm — alias of cubecl's `hip` feature
cuda = ["cubecl/cuda"]     # optional / dev convenience, not the mandated target
wgpu = ["cubecl/wgpu"]     # NOT for the f64 oracle path — see warnings

[dev-dependencies]
proptest  = "1.11.0"
criterion = "0.8.2"
approx    = "0.5"
```

```toml
# lgbm-py/Cargo.toml (Python bindings crate)
[dependencies]
pyo3  = { version = "0.28.3", features = ["extension-module", "abi3-py39"] }
numpy = "0.28.0"          # MUST track pyo3 minor version
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.13,<2.0"]
build-backend = "maturin"
```

```bash
# Python dev loop
pip install maturin==1.13.3
maturin develop --release
```

## Backend Selection: How It Works (compile-time, with a runtime-generic core)

CubeCL backends are selected at **compile time via Cargo features** (`cpu`, `hip`/`rocm`, `cuda`, `wgpu`). There is no single binary that flips backends purely at runtime *by default*. The idiomatic pattern:

- Write all kernels and host code **generic over the `Runtime` trait**: `fn build_histogram<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>, ...)`.
- Each enabled feature gives you a concrete runtime type (`CpuRuntime`, `HipRuntime`, `CudaRuntime`, `WgpuRuntime`).
- To support "switchable at runtime," compile **multiple** runtimes in (enable several features) and `match` on a user config enum that dispatches to the monomorphized generic function per backend. This satisfies the project's "feature flag and/or runtime configuration" requirement: features gate which backends are *available*; a runtime enum picks among the compiled-in ones.

`Plane` API surface (warp/subgroup ops), confirmed via Context7:
- `plane_sum(x)` → CUDA `warpReduceSum` / HIP `__shfl`-based reduce / wgpu `subgroupAdd`.
- Feature-gate at runtime with `client.features().plane.contains(Plane::Ops)` before using plane ops; fall back to a scalar loop otherwise (CPU has `PLANE_DIM == 1`).
- Use `#[comptime] use_plane: bool` to monomorphize plane vs non-plane kernel variants without GPU-side branching.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Custom columnar/binned store | `ndarray` `Array2` as core store | Never for the bin store — can't bit-pack. Fine as a *prediction input* adapter only. |
| `cubecl-hip` (`hip` feature) for ROCm | `cubecl-wgpu` (Vulkan on AMD) | Only if you needed a portable Vulkan path AND could tolerate **no f64** — disqualified by the 1e-12 contract. HIP is the correct ROCm path. |
| f64 accumulators | f32 accumulators | Never for the oracle path; f32 cannot hold 1e-12 parity on summed histograms. f32 only where C++ stores f32 (raw gradient/label storage). |
| Hand-written `.txt` model parser | serde + a derive | Never — format isn't serde-shaped; field order/float formatting are parity-critical. |
| Python-`lightgbm` fixtures for oracle | FFI to `lib_lightgbm.so` | Use FFI only to inspect internal intermediates not exposed by Python/CLI. |
| `pyo3` + `numpy` lockstep | manual ctypes (like upstream) | Upstream is ctypes-over-C-ABI; we have no stable C ABI in v1, so PyO3 over the Rust API is correct. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Raw CUDA / OpenCL / HIP C++ kernels | Project mandate excludes them; defeats portability | CubeCL kernels (`#[cube]`) over the `Runtime` trait |
| `cubecl-wgpu` for the ROCm oracle path | wgpu/WebGPU lacks reliable f64; can't meet 1e-12 | `cubecl-hip` (`hip`/`rocm` feature) |
| `cubecl 0.11.x` from the book examples | Not published on crates.io as of 2026-06-05 — book runs ahead of registry | `cubecl =0.10.0` (pinned) |
| `pyo3 0.28.0` / `0.28.1` | **Yanked** on crates.io | `pyo3 0.28.3` (or ≥ 0.28.2) |
| Mismatched `numpy` / `pyo3` minors | rust-numpy is built against a specific pyo3 minor; mismatch = compile errors | Keep both on `0.28.x` |
| serde for the `.txt` model format | Format is bespoke line-oriented; serde can't reproduce field order/float formatting | Hand-written parser/writer |
| f32 histogram accumulation | Breaks 1e-12 parity (non-associative + precision loss) | f64 accumulators, fixed reduction order |
| Unordered `plane_sum` for oracle reductions | Warp/subgroup add order is implementation-defined → non-deterministic LSBs | Fixed-order reduction matching C++; reserve plane ops for bit-stable results |
| `anyhow::Error` across the PyO3 boundary | Loses typed error info; awkward `PyErr` mapping | `thiserror` enums with `impl From<E> for PyErr` |

## Stack Patterns by Variant

**If targeting the mandated ROCm parity test:**
- Build with `--features rocm` (→ `cubecl-hip`), require a local ROCm toolkit + AMD GPU.
- Force f64 accumulators; use fixed-order reductions, not raw `plane_sum`, for histogram/score sums.
- Because CPU and HIP share the same `#[cube]` source, validate CPU parity first (cheaper), then HIP.

**If building the CPU reference path:**
- Build with `--features cpu` (→ `cubecl-cpu`). `PLANE_DIM == 1` means plane kernels run scalar — a clean, deterministic reference to compare the GPU path against.

**If you need both backends switchable in one binary:**
- Enable `cpu` + `rocm` features together; dispatch via a runtime `Backend` enum that calls the same `Runtime`-generic kernels.

**If shipping Python wheels:**
- Use `abi3-py39` for a single wheel across CPython 3.9+ (matches upstream's broad version support). `maturin build --release`.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `cubecl =0.10.0` | `cubecl-hip` / `cubecl-cpu` / `cubecl-cuda` / `cubecl-wgpu` 0.10.0 | Forwarded via `cubecl` features; keep all CubeCL crates on the **same exact** version. |
| `pyo3 0.28.x` | `numpy 0.28.0` | **Lockstep minor** required. |
| `pyo3 0.28.x` | `maturin 1.13.x` | maturin 1.x supports current pyo3; `requires = "maturin>=1.13,<2.0"`. |
| `proptest 1.11.0` | rustc ≥ 1.85 | Edition 2024 implies a new-enough toolchain. |
| `numpy 0.28.0` | rustc ≥ 1.83 | Satisfied by edition-2024 toolchain. |
| `thiserror 2.0.18` | `anyhow 1.0.102` | Independent; standard pairing, no conflict. |

## Determinism / 1e-12 Oracle — Explicit Callouts

- **Width:** f64 for every accumulation (histogram bins, gradient/hessian sums, leaf values, raw scores). Store raw gradients/labels at the C++ width (`float` by default per `meta.h`) so binning/quantization matches before accumulation.
- **Order:** FP addition is non-associative. 1e-12 parity requires the *same addition order* as C++. On GPU this means avoiding implementation-defined warp/atomic reduction orders. Prefer deterministic tree-reductions with a fixed traversal, or replicate C++'s exact accumulation sequence. The project's own "bit-deterministic reductions" requirement is the correct framing — design it in, don't retrofit.
- **f64 on GPU:** CUDA and HIP support f64 in hardware (slower, but present). **wgpu does not reliably** — a structural reason ROCm must use `cubecl-hip`. Verify f64 availability at runtime via `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))` before launching f64 kernels.
- **Plane ops:** Safe for *count/ballot* style results that are order-independent; unsafe for *summation* where LSBs depend on order. Gate behind a determinism review per kernel.
- **Model-format float formatting:** Matching C++'s `%.17g`-style text output is a parity surface in its own right — Rust's default float formatting differs. Validate against fixtures.

## CubeCL Version-Churn Risk (flagged for roadmap)

- CubeCL is **officially alpha**; the README states "not all platforms support the same features" and the project is pre-1.0.
- The book already references an unreleased `0.11.0`, so a breaking minor is likely imminent. **Mitigation:** pin `=0.10.0`, commit `Cargo.lock`, isolate ALL CubeCL usage behind the `lgbm-backend` crate's own trait so an upgrade touches one crate, and schedule a "CubeCL upgrade" spike per roadmap milestone boundary.

## Sources

- crates.io API (verified 2026-06-05): `cubecl` 0.10.0 (2026-05-07, latest stable; prior were `0.10.0-pre.N`); `pyo3` 0.28.3 (0.28.0/0.28.1 yanked); `numpy` (rust-numpy) 0.28.0; `ndarray` 0.17.2; `proptest` 1.11.0; `criterion` 0.8.2; `thiserror` 2.0.18; `anyhow` 1.0.102 — HIGH
- PyPI (verified 2026-06-05): `maturin` 1.13.3 — HIGH
- Context7 `/tracel-ai/cubecl` — Plane API (`plane_sum`, `features().plane.contains(Plane::Ops)`), comptime feature specialization, `Feature::Type(Elem::Float(FloatKind::F16/F64))` runtime checks — HIGH
- GitHub `tracel-ai/cubecl` `crates/cubecl/Cargo.toml` — feature→crate mapping: `cpu`→cubecl-cpu, `cuda`→cubecl-cuda, `hip`(alias `rocm`)→cubecl-hip, `wgpu`/`wgpu-spirv`(`vulkan`)/`wgpu-msl`(`metal`)→cubecl-wgpu — HIGH
- GitHub `tracel-ai/cubecl` README support table — CUDA/HIP/wgpu/CPU all "supported"; project "in alpha" — HIGH
- CubeCL book `installation.md` — install pattern; book example pins `0.11.0` (ahead of registry) — MEDIUM (book on `main`)
- `.planning/codebase/STACK.md` — C++ reference: `score_t/label_t = float` default (double-accumulation), bespoke bin layout, json11/manual model text, Eigen for linear trees — HIGH (direct repo analysis)
- f64-on-wgpu limitation — WebSearch + CubeCL platform notes; wgpu/WebGPU f64 gap is well established, exact CubeCL behavior not byte-verified — MEDIUM

---
*Stack research for: pure-Rust LightGBM port with CubeCL CPU/ROCm backend + Python bindings, 1e-12 oracle*
*Researched: 2026-06-05*
