---
status: complete
phase: 01-oracle-contract-foundations
source: [01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md]
started: 2026-06-06T22:37:34Z
updated: 2026-06-06T22:37:34Z
---

## Current Test

[testing complete]

<!--
Note: Phase 1 is a technical foundations phase (Cargo workspace, bit-exact RNG
port, f32 oracle comparator, Config + alias/drift machinery, error layering).
It has no user-facing UI/workflow. Per the chosen UAT mode, deliverables were
verified by running the actual acceptance commands (cargo build/test + the
oracle parity & config-drift suites). Each checkpoint records the real result.
-->

## Tests

### 1. Clean Workspace Build (cold-start smoke)
expected: `cargo build --workspace` compiles every crate (lgbm-core, lgbm-compute, oracle-harness, xtask) under edition 2024 with no errors.
command: cargo build --workspace
result: pass

### 2. RNG Bit-Exact Parity
expected: Rust `Random` LCG reproduces every committed C++ golden case (512 cases: 256 RNG seed sequences + 256 Sample (N,K) cases) bit-for-bit. `rng_parity` 1/1.
command: cargo test -p oracle-harness --test rng_parity
result: pass

### 3. Oracle Comparator (~1e-6)
expected: abs-diff comparator at locked ORACLE_TOL reports first-offending index correctly; unit suite 5/5.
command: cargo test -p oracle-harness --test comparator
result: pass

### 4. f32 Type/Constant Contract + RNG Unit
expected: lgbm-core unit tests (f32 ScoreT/LabelT + meta.h constants, thiserror enums, Random LCG + both Sample branches) 14/14.
command: cargo test -p lgbm-core --lib
result: pass

### 5. Config Defaults Match config.h
expected: flat `Config` defaults mirror C++ config.h member initializers 1:1; config_defaults 5/5.
command: cargo test -p lgbm-core --test config_defaults
result: pass

### 6. Deterministic Alias Resolution + empty==absent
expected: KeyAliasTransform/SortAlias collision resolution is deterministic (canonical beats alias; tie-break by key len then lexicographic); empty-string reads are no-ops. alias_resolution 8/8.
command: cargo test -p lgbm-core --test alias_resolution
result: pass

### 7. Seed Derivation (six sub-seeds, exact order)
expected: the six sub-seeds derive from `Random::new(seed)` via six draws in exact config.cpp order; seed_derivation 4/4.
command: cargo test -p lgbm-core --test seed_derivation
result: pass

### 8. Config Validation (typed errors, no panic, fuzz)
expected: in-scope CHECK constraints surface typed `ConfigError`; randomized (2000 numeric) + hostile-string fuzz (3000) never panic; empty==absent no-ops; config_validation 16/16.
command: cargo test -p lgbm-core --test config_validation
result: pass

### 9. Config Drift-Checker (Rust tables ⊇ C++)
expected: drift-checker parses config_auto.cpp alias_table()/parameter_set() and proves Rust IN_SCOPE_PARAMS/ALIAS_TABLE are a faithful superset of every in-scope C++ entry; config_drift 3/3.
command: cargo test -p oracle-harness --test config_drift
result: pass

### 10. Full Workspace Gate
expected: `cargo test --workspace` green across all crates and suites — 0 failed.
command: cargo test --workspace
result: pass

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
