# Technology Stack

**Analysis Date:** 2026-07-09

## Languages

**Primary:**
- Rust (edition 2024, `rust-version = 1.95`) — the entire deliverable: `crates/lgbm-core`, `crates/lgbm-compute`, `crates/lgbm-dataset`, `crates/lgbm-model`, `crates/lgbm-treelearner`, `crates/lgbm-objective`, `crates/lgbm-metric`, `crates/lgbm-boosting`, `crates/lgbm` (facade), `crates/lgbm-python` (PyO3 bindings), `crates/oracle-harness` (parity test harness), `xtask` (repo tooling).

**Secondary:**
- Python 3.11+ — thin pure-Python wrapper package under `crates/lgbm-python/python/` (import name `lightgbm_rs`), mirrors the official `lightgbm` package API surface per project constraint. Built via maturin into a compiled extension `lightgbm_rs._core`.
- CUDA C++ / OpenCL C / C++11 — **reference only**, not part of the Rust deliverable. `LightGBM/` (mainline C++ reference, read-only per project convention — never git-added) and `LightGBM-release-4.6.0.99/` (AMD ROCm/HIP fork, used as the real HIP histogram-kernel baseline for GPU parity/perf work, per `rocm-baseline-amd-fork` project memory).

## Runtime

**Environment:**
- Rust toolchain pinned via `rust-toolchain.toml`: channel `1.95.0` (stable), components `rustfmt`, `clippy`. Edition 2024 requires this floor.
- CPython 3.11+ for the Python bindings (`abi3-py311` stable ABI — one wheel covers 3.11+).

**Package Manager:**
- Cargo (workspace resolver `"3"`), lockfile `Cargo.lock` present and committed (5846 lines).
- Python: `maturin` build backend (`crates/lgbm-python/pyproject.toml`, `requires = ["maturin>=1.13,<2.0"]`); no separate Python lockfile — the wrapper only declares `numpy`, `scipy` as install deps plus optional extras.

## Frameworks

**Core compute:**
- `cubecl` 0.10.0 (workspace dependency, `crates/lgbm-compute/Cargo.toml`) — the compute/GPU abstraction layer per project constraint ("no raw CUDA/OpenCL"). Backend features: `cpu` (default, `cubecl-cpu` — the deterministic f64-fold parity anchor), `rocm` (`cubecl/hip` + `cubecl-hip-sys` FFI), `cuda` (`cubecl/cuda`), `wgpu` (`cubecl/wgpu`, WGSL target — noted risk: no f32 atomics for the histogram kernel).
- `rayon` 1.10 — CPU-side data/thread parallelism (histogram build, partition split, feature-parallel binning, feature-parallel objective gradients). Used in `lgbm-compute`, `lgbm-treelearner`, `lgbm-objective`, `lgbm` facade.

**Python bindings:**
- `pyo3` 0.27 (`features = ["abi3-py311"]`) — Rust/CPython FFI, `crates/lgbm-python`.
- `numpy` (rust-numpy) 0.27 — array marshalling; version-locked to the pyo3 0.27 / pyo3-polars 0.26 triangle (breaking to mix with numpy 0.28).
- `pyo3-polars` 0.26 (`features = ["dtype-categorical"]`) + direct `polars` 0.53 dep (`features = ["dtype-categorical", "dtype-u8", "dtype-u16"]`) — Arrow-stream DataFrame ingest with categorical/Enum column support.
- `mimalloc` 0.1 — process global allocator for the compiled extension module (always-on in `lgbm-python`; optional feature-gated in the `lgbm` facade crate).

**Testing:**
- `oracle-harness` (in-house crate, `crates/oracle-harness`) — parity replay test harness comparing Rust output against committed C++ goldens (kernel/histogram, learner/tree, boosting-loop layers); consumed as a dev-dependency by nearly every crate.
- Cargo's built-in test runner (`cargo test`) for unit/integration tests; no external test framework.
- Real `lib_lightgbm` 4.6 built locally from `LightGBM/` (via `external_libs`) used for floating-point-trace parity debugging (per `lightgbm-ref-tree-untracked` memory).

**Build/Dev:**
- `xtask` crate (`anyhow` only dep) — custom repo-tooling pattern (`cargo run -p xtask`) instead of shell scripts/Makefiles.
- `maturin` — builds/publishes the Python extension wheel.
- Kaggle CLI (`kaggle` Python package, authenticated) — used to run real discrete-CUDA benchmarks (`poll_kaggle.sh`, `continue_benchmark.py`) since the local GPU is a spoofed 8-CU APU, not real discrete hardware (see `rocm-gfx1100-available` / `kaggle-cli-cuda-bench` memory).

## Key Dependencies

**Critical:**
- `thiserror` 2.0.18 (workspace dep) — structured domain errors at every crate's library boundary, per project constraint.
- `anyhow` 1.0.102 (workspace dep) — ergonomic error propagation in app/high-level layers (`oracle-harness`, `xtask`, dev-dependencies).
- `cubecl-hip-sys` (version "7", optional, `rocm` feature only) — direct FFI to `hipGetDevicePropertiesR0600()` for real device Compute-Unit count, avoiding a hardcoded phantom CU value.
- `serde` 1.x (`features = ["derive"]`, optional, gated by the `gpu` feature) — required by cubecl's `AutotuneKey` trait (`Serialize`/`DeserializeOwned`) for the persistent autotune disk cache; only compiled for GPU backends.

**Infrastructure:**
- `cubecl-cpu`, `cubecl-hip`, `cubecl-cuda`, `cubecl-wgpu` — pulled in transitively via `cubecl` 0.10.0 feature flags (`cpu`/`hip`/`cuda`/`wgpu`), never named directly in crate code (isolation is via the `cubecl::*::*Runtime` re-exports).

## Configuration

**Environment:**
- No `.env` files present in the repo.
- Kaggle credentials at `~/.kaggle/access_token` (outside repo, per `AGENTS.md`).
- Runtime tuning via env vars read at process start, e.g. `LGBM_PHASE_PROF`, `LGBM_BENCH_SWEEP`, `LGBM_SCAN_CUBEDIM`, `LGBM_UNIFIED_BFS_THRESHOLD`, `LGBM_UNIFIED_SUBSCAN_THRESHOLD` (perf/profiling knobs documented in project memory, not `.env`-backed).

**Build:**
- Root `Cargo.toml` — workspace member list (11 crates + xtask), `[workspace.dependencies]`, and tuned `[profile.release]` (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1` — parity-neutral codegen flags, no fast-math) plus a `[profile.profiling]` (LTO off, `codegen-units = 16`, debug symbols, for callgrind/profiler attribution).
- `crates/lgbm-python/pyproject.toml` — maturin config: `python-source = "python"`, `module-name = "lightgbm_rs._core"`, `bindings = "pyo3"`.
- `.github/workflows/release-python.yml` — GitHub Actions release pipeline (Linux `manylinux_2_34`/x86_64, macOS arm64, Windows x64, sdist) building CPU-only wheels via `PyO3/maturin-action`, publishing to PyPI via OIDC Trusted Publishing on `v*` tags. GPU wheels (`--features cuda/rocm/wgpu`) are intentionally excluded from the automated release pipeline (built manually, e.g. on Kaggle/Colab).

## Platform Requirements

**Development:**
- Rust 1.95.0 stable (rustup-managed via `rust-toolchain.toml`).
- Local AMD ROCm GPU available but is a spoofed 8-CU `gfx1152` APU masquerading as `gfx1100` (HSA_OVERRIDE) — valid for parity gates, NOT valid for GPU perf numbers (see `rocm-gfx1100-available` memory). Real discrete-GPU (CUDA) benchmarking is done remotely via Kaggle kernels.
- Python 3.11+ with `maturin` for building/testing the bindings (`uv` venv at repo root per `phase8-python-venv` memory).

**Production:**
- Distribution: PyPI wheel `lightgbm-rs` (default build CPU-only; GPU builds are opt-in cargo features `rocm`/`cuda`/`wgpu` forwarded from `lgbm-python` → `lgbm` → `lgbm-compute`).
- Backend selection cascade at compile/runtime: `rocm > cuda > wgpu > cpu` (see `crates/lgbm/src/booster.rs` per `lgbm-python`'s Cargo.toml comments).
- The `cpu` backend (`cubecl-cpu`) is always available and is the deterministic f64-fold parity anchor / hard merge gate; GPU backends are opt-in and held to a looser ~1e-6 parity bound.

---

*Stack analysis: 2026-07-09*
