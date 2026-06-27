---
quick_id: 260627-rh0
slug: add-github-actions-release-workflow-buil
date: 2026-06-27
status: in-progress
---

# Quick Task: GitHub Actions release workflow for the Python package

## Goal

CI that builds maturin RELEASE wheels and publishes the `lightgbm-rs` package to PyPI.
User choices: trigger on `v*` tag → publish to **PyPI via Trusted Publishing (OIDC)**;
platforms **Linux + macOS + Windows**.

## Design (`.github/workflows/release-python.yml`)

- Triggers: `push: tags: v*` (build + publish) and `workflow_dispatch` (build-only, for
  validating before tagging).
- Jobs: `linux` (manylinux_2_28, x86_64), `macos` (macos-13 x86_64 + macos-14 aarch64),
  `windows` (x64), `sdist`; then `release` (needs all) → `pypa/gh-action-pypi-publish`
  with `id-token: write`, `environment: pypi`, gated `if: startsWith(ref,'refs/tags/v')`.
- Each build: `PyO3/maturin-action@v1`, `args: --release --out dist --manifest-path
  crates/lgbm-python/Cargo.toml`, `rust-toolchain: 1.95.0` (matches rust-toolchain.toml,
  edition 2024), `sccache: true`. abi3-py311 → ONE wheel per platform covers Python 3.11+
  (no per-Python matrix). Artifacts all named `wheels-*`; the release job downloads them
  with `pattern: wheels-*, merge-multiple: true`.

## Key findings (de-risked locally)

- `cubecl-cpu → tracel-llvm → tracel-llvm-bundler` self-provisions LLVM ⇒ **NO system-LLVM
  install step needed** in CI. Proven: `maturin build --release` succeeds locally with no
  `llvm-config`/`LLVM_SYS_*` env, producing `lightgbm_rs-0.1.0-cp311-abi3-manylinux_2_38_x86_64.whl`
  in ~3.5 min. Wheel interior valid (`lightgbm_rs/_core.abi3.so` + METADATA + RECORD).
- maturin only compiles the Rust crates — the untracked C++ `LightGBM/` tree is NOT needed.
- Version is `dynamic` from `crates/lgbm-python/Cargo.toml` `version`, NOT the git tag.

## Required one-time PyPI setup (user, documented in SUMMARY)

1. pypi.org → create/claim project `lightgbm-rs` (pending publisher OK pre-first-upload).
2. Trusted Publisher: owner `BectorVoom`, repo `lightgbm_rs`, workflow `release-python.yml`,
   environment `pypi`.
3. (Recommended) GitHub repo Environment named `pypi`.

## Residual risks (flagged, can't fully test without CI)
- The `tracel-llvm-bundler` prebuilt LLVM must work inside the manylinux_2_28 (glibc 2.28)
  container. If the Linux job fails fetching/linking LLVM, bump `manylinux` to `2_34`/`auto`.
- Before each tag, bump `crates/lgbm-python/Cargo.toml` `version` to match (else wheel
  version ≠ tag and PyPI rejects re-upload of an existing version).

## Out of scope
- GPU wheels (cuda/rocm/wgpu) — need toolkits/hardware; not part of the release pipeline.
- Linux aarch64 (cross + LLVM-bundler cross-availability unknown); x86_64 only on Linux.
