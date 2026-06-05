---
phase: 1
slug: oracle-contract-foundations
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-05
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (libtest) + `cargo test` |
| **Config file** | none — standard cargo layout (Wave 0 installs nothing; libtest is built in) |
| **Quick run command** | `cargo test -p lgbm-core` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate>` (the crate touched)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite green + golden regen idempotency check
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| FND-01 | — | LCG reproduces 100k C++ draws bit-for-bit (RandInt16/32, NextFloat, NextInt, Sample across branch) | golden/unit | `cargo test -p oracle-harness rng_parity` | ❌ W0 | ⬜ pending |
| FND-01 | — | Seed derivation order matches C++ `Config::Set` | unit | `cargo test -p lgbm-core seed_derivation` | ❌ W0 | ⬜ pending |
| FND-02 | — | Workspace builds under edition 2024 | smoke | `cargo build --workspace` | ❌ W0 | ⬜ pending |
| FND-03 | — | f32 type aliases + constants match meta.h | unit | `cargo test -p lgbm-core types` | ❌ W0 | ⬜ pending |
| FND-04 | — | thiserror errors at boundary; anyhow in harness | compile/unit | `cargo test -p lgbm-core error` | ❌ W0 | ⬜ pending |
| CFG-01 | — | Config struct holds in-scope params with C++ defaults | unit | `cargo test -p lgbm-core config_defaults` | ❌ W0 | ⬜ pending |
| CFG-02 | — | Alias resolution matches `alias_table()` | unit | `cargo test -p lgbm-core alias_resolution` | ❌ W0 | ⬜ pending |
| CFG-01/CFG-02 | — | Drift-checker: Rust covers all in-scope params/aliases in config_auto.cpp | unit | `cargo test -p oracle-harness config_drift` | ❌ W0 | ⬜ pending |
| CFG-03 | T-1-01 | Each CHECK_* constraint returns typed Err on violation (no panic on hostile input) | unit | `cargo test -p lgbm-core config_validation` | ❌ W0 | ⬜ pending |
| ORA-01 | — | abs-diff comparator flags > ~1e-6 | unit | `cargo test -p oracle-harness comparator` | ❌ W0 | ⬜ pending |
| ORA-02 | — | Reference manifest (commit hash, flags) checked in and regen is idempotent | golden/script | `cargo test -p oracle-harness reference_manifest` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/oracle-harness/fixtures/rng_sequence.*` — committed 100k-draw golden (FND-01) — requires C++ harness + regen
- [ ] `crates/oracle-harness/tests/rng_parity.rs` — FND-01
- [ ] `crates/lgbm-core/src/...` + unit tests — FND-01/03/04, CFG-01/02/03
- [ ] `crates/oracle-harness/tests/config_drift.rs` — CFG drift (D-11)
- [ ] `crates/oracle-harness/tests/comparator.rs` — ORA-01
- [ ] Reference manifest file (commit hash + deterministic flags) — ORA-02
- [ ] Framework install: none (libtest is built in)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Initial golden capture from C++ reference build | FND-01 / ORA-02 | Requires building the pinned C++ LightGBM 4.6 submodule and running the capture harness once on the developer's machine | Build `lib_lightgbm` at pinned commit, run capture xtask/CMake target, commit emitted fixtures + manifest |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
