---
phase: 3
slug: tree-model-model-text-i-o-predict-parity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-05
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 03-RESEARCH.md § Validation Architecture (D-05/D-06 layered goldens).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cargo test` (workspace convention from Phases 1–2) |
| **Config file** | none — standard Cargo test layout (`crates/lgbm-model/tests/*.rs` + inline `#[cfg(test)]`) |
| **Quick run command** | `cargo test -p lgbm-model` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds (replay only; no C++ toolchain at test time) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-model`
- **After every plan wave:** Run `cargo test --workspace` (no regression in dataset/core/oracle)
- **Before `/gsd-verify-work`:** Full suite green + `model-capture` idempotent (committed goldens regenerate byte-identical)
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

> D-06 layered goldens — a mismatch localizes the divergence to parse vs predict-math vs convert vs serialize.

| Layer (D-06) | Requirement | Threat Ref | Behavior | Test Type | Automated Command | File Exists | Status |
|--------------|-------------|------------|----------|-----------|-------------------|-------------|--------|
| 0. `%.17g`/`{:g}` formatter | DAT-09 | — | printf-`%g` parity for threshold/leaf/gain fields (linchpin gate) | unit | `cargo test -p lgbm-model format::` | ❌ W0 | ⬜ pending |
| 1. model-text round-trip bytes | DAT-08, DAT-09 | — | Rust-written `.txt` byte-identical to committed C++ `.txt` (`compare_exact_bytes`) | integration | `cargo test -p lgbm-model --test model_text_roundtrip` | ❌ W0 | ⬜ pending |
| 2. raw score | PRD-01 | — | per-row f32 raw scores within ORACLE_TOL (`compare_within`) | integration | `cargo test -p lgbm-model --test predict_raw_parity` | ❌ W0 | ⬜ pending |
| 3. transformed output | PRD-02 | — | ConvertOutput sigmoid/softmax/ova/identity within ORACLE_TOL | integration | `cargo test -p lgbm-model --test predict_transform` | ❌ W0 | ⬜ pending |
| 4. leaf index | PRD-03 | — | per-(row×tree×class) u32 leaf ids exact (`compare_exact_u32`) | integration | `cargo test -p lgbm-model --test predict_leaf_parity` | ❌ W0 | ⬜ pending |
| 5. sub-range raw | PRD-06 | — | raw scores for `(start_iteration, num_iteration)` slices incl. `-1 == all` | integration | `cargo test -p lgbm-model --test predict_subrange` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**D-06 layer → comparator → fixture map (comparators already exist in oracle-harness):**

| D-06 Layer | Comparator | Fixtures (D-05 corpus) |
|-----------|-----------|------------------------|
| 1. round-trip bytes | `compare_exact_bytes` | all 5 corpora |
| 2. raw score | `compare_within` (ORACLE_TOL) | regression, binary, multiclass |
| 3. transformed | `compare_within` | binary (sigmoid), multiclass (softmax), regression (identity) |
| 4. leaf index | `compare_exact_u32` | regression, multiclass (per-class stride check) |
| 5. sub-range raw | `compare_within` | sub-range corpus |

---

## Wave 0 Requirements

- [ ] `crates/lgbm-model/` crate skeleton + add to root `Cargo.toml` `members`
- [ ] `crates/lgbm-model/src/format.rs` — `%.17g` + `{:g}` formatter + unit parity test (the linchpin; build & verify FIRST)
- [ ] `xtask` `model-capture` subcommand + capture-path decision (RESEARCH Open Q2) — emits committed `.txt` models + predict-vector goldens into `tests/fixtures/models/`
- [ ] `tests/fixtures/models/{regression,binary,multiclass,categorical,subrange}/` — committed C++ `.txt` + raw/transformed/leaf/subrange golden vectors
- [ ] Extend `REFERENCE_MANIFEST.md` (ORA-02) with model/predict fixture provenance + exact `lightgbm` version & train params
- [ ] Five integration test files (one per D-06 layer 1–5) wired to oracle-harness comparators

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Golden-capture provenance (C++ `.txt` numerically identical to `lib_lightgbm`) | DAT-08/09 | One-time human approval of capture path (pip-`lightgbm` vs verbatim transcription) since full lib is unbuildable | Run `cargo xtask model-capture`, confirm committed goldens match a known-good reference; record version/params in `REFERENCE_MANIFEST.md` |

*All other phase behaviors have automated verification via the layered goldens above.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (crate skeleton, formatter, capture, fixtures)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
