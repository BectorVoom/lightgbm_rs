---
quick_id: 260626-o9d
description: Python bindings need at least numpy 2.x.x (and require Python >=3.11)
status: complete
date: 2026-06-26
---

# Quick Task 260626-o9d — Summary

## What changed

Raised the `lightgbm-rs` Python-binding floors:

- **numpy**: `numpy>=1.17.0` → `numpy>=2.0.0` (`crates/lgbm-python/pyproject.toml`)
- **Python**: `requires-python = ">=3.9"` → `">=3.11"` (`crates/lgbm-python/pyproject.toml`)
- **abi3 ABI floor**: `abi3-py39` → `abi3-py311` on both the `pyo3` dependency and
  dev-dependency (`crates/lgbm-python/Cargo.toml`), so the wheel's stable-ABI tag
  becomes `cp311-abi3`, consistent with the new Python floor. D-13 comment updated
  in `pyproject.toml`.

## Why

User requirement: the Python bindings must require at least numpy 2.x, and Python
≥ 3.11. Aligning the abi3 floor (user-confirmed) keeps the documented D-13
single-abi3-wheel decision internally consistent — the ABI tag no longer
undersells the real Python floor.

## Verification

- `cargo verify-project --manifest-path crates/lgbm-python/Cargo.toml` → `{"success":"true"}`.
- `cargo metadata` confirms `abi3-py311` is a real feature of the resolved pyo3 0.27.2.
- numpy 2.0 supports Python 3.9–3.12; the `>=3.11` floor is compatible.
- Project `.venv` is Python 3.12 — satisfies `>=3.11`.

## Notes / not touched

- The Rust `numpy = "0.27"` (rust-numpy) crate is the build-time binding layer and
  is independent of the runtime numpy floor — left unchanged.
- No version classifiers were added to `pyproject.toml` (none existed; out of scope).
- A full `cargo build`/maturin wheel build was not run (requires the maturin
  toolchain); manifest + feature resolution validated instead.
