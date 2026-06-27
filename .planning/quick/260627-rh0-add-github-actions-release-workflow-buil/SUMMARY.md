---
quick_id: 260627-rh0
slug: add-github-actions-release-workflow-buil
date: 2026-06-27
status: complete
commits:
  - 94542de ci(quick-260627-rh0): add GitHub Actions PyPI release workflow for the Python package
---

# Summary: GitHub Actions PyPI release workflow

## Delivered

`.github/workflows/release-python.yml` (the repo had no CI before). Builds maturin RELEASE
wheels for Linux (manylinux_2_28 x86_64) + macOS (x86_64 + aarch64) + Windows (x64) + sdist,
then publishes `lightgbm-rs` to PyPI via OIDC Trusted Publishing on a `v*` tag.
`workflow_dispatch` builds artifacts only (validate the pipeline before tagging).

abi3-py311 ⇒ ONE wheel per platform covers Python 3.11+ (no per-Python matrix).

## Verified locally (CI build proxy)

- `maturin build --release --manifest-path crates/lgbm-python/Cargo.toml` builds end-to-end
  in ~3m27s and emits `lightgbm_rs-0.1.0-cp311-abi3-manylinux_2_38_x86_64.whl`.
- **No system LLVM needed:** the build succeeds with no `llvm-config` on PATH and no
  `LLVM_SYS_*` env — `cubecl-cpu → tracel-llvm → tracel-llvm-bundler` self-provisions LLVM.
  So CI needs no LLVM install step.
- Wheel interior valid: `lightgbm_rs/_core.abi3.so` + `dist-info/METADATA` + `RECORD`.
- maturin compiles only the Rust crates; the untracked C++ `LightGBM/` tree is not needed.

## ⚠️ YOU must do before the first release

### 1. One-time PyPI Trusted Publishing setup
On https://pypi.org (logged in as the project owner):
1. Create/claim the project `lightgbm-rs` — or add a **pending** publisher (works before the
   first upload): Account → Publishing → "Add a pending publisher".
2. Trusted Publisher values:
   - PyPI Project Name: `lightgbm-rs`
   - Owner: `BectorVoom`
   - Repository name: `lightgbm_rs`
   - Workflow name: `release-python.yml`
   - Environment name: `pypi`
3. (Recommended) In the GitHub repo: Settings → Environments → create `pypi`.

No API token is stored — auth is OIDC (`id-token: write` in the release job).

### 2. Bump the version before tagging (GOTCHA)
The wheel version is `dynamic` from `crates/lgbm-python/Cargo.toml` `version` — **NOT** the
git tag. Before each release, set that `version` to the release number, commit, THEN tag:
```
# edit crates/lgbm-python/Cargo.toml: version = "0.2.0"
git commit -am "release: v0.2.0"
git tag v0.2.0 && git push origin v0.2.0
```
If the tag's wheel version already exists on PyPI, the upload is rejected.

### 3. Recommended first run
Run the workflow manually (Actions → Release Python package → Run workflow) to validate all
platform builds BEFORE pushing a real tag.

## Residual risks (couldn't be tested without running CI)

- **manylinux glibc vs the LLVM bundler:** the local wheel tagged `manylinux_2_38` (host
  glibc 2.38); CI uses the `manylinux_2_28` container for broader compatibility. If the
  Linux job fails fetching/linking the bundled LLVM under glibc 2.28, bump the `linux` job's
  `manylinux:` to `2_34` (or `auto`).
- macOS/Windows builds of the LLVM bundler + polars + cubecl-cpu are unverified locally
  (Linux-only host). If a platform fails, the abi3 wheels from the others still publish
  independently (jobs are not fail-fast on the macOS matrix; a hard failure in one OS job
  blocks the `release` step since it `needs` all — drop a failing OS from `needs` if you
  want partial publishes).

## Out of scope
- GPU wheels (`--features cuda/rocm/wgpu`) — need a toolkit/hardware; not in this pipeline.
- Linux aarch64 (cross-compile + bundler cross-availability unknown).
