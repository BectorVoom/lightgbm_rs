---
phase: 2
slug: dataset-binning-determinism-root
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-05
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust built-in) + committed golden-replay fixtures (oracle-harness) |
| **Config file** | none — workspace `Cargo.toml`; `lgbm-dataset` crate added in Wave 0 |
| **Quick run command** | `cargo test -p lgbm-dataset` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30–60 seconds (golden replay; no C++ toolchain at test time) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-dataset`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

> Filled by the planner / gsd-nyquist-auditor from PLAN.md tasks. Three-layer golden granularity (D-07): (1) BinMapper internals — `bin_upper_bound_`, `bin_type`, `missing_type`, `default_bin`, `most_freq_bin`, `num_bin`; (2) full per-row bin-index assignment vector per feature; (3) categorical category→bin maps + EFB bundle/offset layout.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | DAT-01 | — | N/A | golden-parity | `cargo test -p lgbm-dataset` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/lgbm-dataset/` — new crate scaffold + workspace member registration
- [ ] Golden-fixture capture harness (reuse oracle-harness committed-golden + idempotent-regen pattern; extend `REFERENCE_MANIFEST.md` for binning master seed + tolerance)
- [ ] Copy/derive `LightGBM/examples/` + EFB fixtures into the committed fixture dir (never reference the untracked `LightGBM/` tree at test time)

*Existing oracle-harness infrastructure covers the comparator + replay seam; per-stage binning goldens plug in here.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| EFB golden capture mechanism | DAT-05 | Open feasibility question (whether `dataset.cpp` compiles in a focused harness vs CLI-dump fallback) — sequence EFB last | Capture EFB bundle/offset goldens via focused harness or `lightgbm` CLI dump; commit, then replay |

*All numeric/categorical/metadata behaviors have automated golden-parity verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
