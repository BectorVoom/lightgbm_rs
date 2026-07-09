# External Integrations

**Analysis Date:** 2026-07-09

## APIs & External Services

**Package Registries:**
- crates.io — Rust dependency source for all workspace crates (`Cargo.lock` entries `source = "registry+https://github.com/rust-lang/crates.io-index"`).
- PyPI — target for the published `lightgbm-rs` Python distribution; published via OIDC Trusted Publishing (no stored API token) from `.github/workflows/release-python.yml`.

**Benchmark/Compute Infrastructure:**
- Kaggle — used to run real discrete-GPU (CUDA) benchmarks/kernels, since the local ROCm GPU is a spoofed 8-CU APU (`gfx1152` masquerading as `gfx1100`), not real discrete hardware. Interaction is via the `kaggle` CLI:
  - `poll_kaggle.sh` polls `kaggle kernels status boomvector/lgb-rs-cuda-bench` until `COMPLETE`/`ERROR`/`CANCELLED`, then downloads output via `kaggle kernels output ... -p kaggle_out`.
  - `continue_benchmark.py` — companion driver script for continuing/re-running Kaggle benchmark kernels.
  - Auth: token file at `~/.kaggle/access_token` (outside repo, per `AGENTS.md`); an OAuth `credentials.json` flow under account `yensen2` is also referenced in project memory (superseding an earlier `boomvector` auth).
  - No production runtime dependency — this is a development/benchmarking integration only, not something the shipped library calls.

## Data Storage

**Databases:**
- None. No SQL/NoSQL database client dependencies anywhere in the workspace `Cargo.lock` (no `sqlx`, `diesel`, `postgres`, `rusqlite`, etc.).

**File Storage:**
- Local filesystem only. Training/prediction data is read from local files (CSV, binary datasets) and LightGBM-format model files. No cloud object storage (S3/GCS/Azure Blob) client integrations.
- `crates/lgbm-python` accepts in-memory `numpy` arrays and `polars`/Arrow DataFrames (via `pyo3-polars` + direct `polars` dep) as data sources — this is in-process data marshalling, not an external storage service.

**Caching:**
- `cubecl::tune` autotune persistent disk cache (local filesystem, gated by `serde` + the `gpu` feature) — caches GPU kernel autotune results (e.g. row-partition Compute-Unit sizing) to disk between runs. Not a distributed/external cache.

## Authentication & Identity

**Auth Provider:**
- None in the shipped library/bindings — no user-facing auth surface (this is a compute library, not a service).
- Kaggle CLI token (`~/.kaggle/access_token`) and GitHub Actions OIDC Trusted Publishing (for PyPI) are the only credentialed integrations, both development/CI-only.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Datadog/etc.). Errors surface via `thiserror`-typed domain errors at crate boundaries and `anyhow` propagation in harness/tooling code.

**Logs:**
- No structured logging framework dependency detected (no `tracing`/`log` crate in `Cargo.lock` inspection so far for the core crates); ad-hoc `println!`/env-var-gated profiling counters (e.g. `LGBM_PHASE_PROF`) are used for perf instrumentation instead of a logging service.

## CI/CD & Deployment

**Hosting:**
- No application hosting — this is a library/CLI/Python-package deliverable, not a deployed service.

**CI Pipeline:**
- GitHub Actions — single workflow `.github/workflows/release-python.yml` ("Release Python package"):
  - Triggers: push of a `v*` tag (build + publish to PyPI) or manual `workflow_dispatch` (build only, artifacts uploaded for review).
  - Jobs: `linux` (manylinux_2_34/x86_64 via `PyO3/maturin-action`), `macos` (macOS 14 arm64 only — Intel excluded, no cross-compile), `windows` (x64), `sdist`, `release` (OIDC publish to PyPI, gated on `pypi` GitHub Environment).
  - Uses `sccache` for build caching in each maturin-action job.
  - No separate lint/test CI workflow detected at `.github/workflows/` — only the release pipeline exists.

## Environment Configuration

**Required env vars:**
- None required for a default CPU build/test run.
- Perf/debug env vars (optional, read at process start): `LGBM_PHASE_PROF`, `LGBM_BENCH_SWEEP`, `LGBM_SCAN_CUBEDIM`, `LGBM_UNIFIED_BFS_THRESHOLD`, `LGBM_UNIFIED_SUBSCAN_THRESHOLD`.
- Kaggle: token resolved by the `kaggle` CLI from `~/.kaggle/access_token` (or `credentials.json`), not an env var read by this repo's own code.

**Secrets location:**
- Kaggle token: `~/.kaggle/access_token` (outside the repo, never committed).
- PyPI publishing: no stored secret — GitHub Actions OIDC Trusted Publishing issues a short-lived token at publish time via the `pypi` environment's `id-token: write` permission.
- No `.env` file or `credentials.*`/`secrets.*` files present in the repository.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None. (GitHub Actions job-to-job artifact passing via `actions/upload-artifact`/`download-artifact` is CI-internal, not a webhook.)

---

*Integration audit: 2026-07-09*
