---
phase: 22
slug: on-device-categorical-splits-feature-coverage
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-02
---

# Phase 22 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust) + oracle-harness integration tests |
| **Config file** | none — workspace `Cargo.toml` |
| **Quick run command** | `cargo test -p lgbm-compute categorical` |
| **Full suite command** | `cargo test --workspace` (default cubecl-cpu f64 lane = hard merge gate) |
| **Estimated runtime** | ~TBD by planner |

---

## Sampling Rate

- **After every task commit:** Run `{quick run command}`
- **After every plan wave:** Run `{full suite command}`
- **Before `/gsd-verify-work`:** Full suite must be green on the default cubecl-cpu lane
- **Max feedback latency:** TBD

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 22-01-01 | 01 | 1 | ODL-22 | — | N/A (compute lib) | integration | `cargo test -p oracle-harness learner_parity` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Planner fills the full map. Key anchor gates (D-01): (a) real 4.6 categorical goldens bit-exact — bitset / decision-type bit / `num_cat` / `cat_threshold_real`; (b) cubecl-cpu f64 structure gate routes categorical rows correctly, tie-aware on `default_left`.*

---

## Wave 0 Requirements

- [ ] Categorical corpus cases added to `crates/oracle-harness/tests/learner_parity.rs` structure gate (one-hot + many-vs-many)
- [ ] `max_cat_threshold > 32` fixture/unit-test to prove D-03 slab sizing non-vacuously (both committed goldens use 32)
- [ ] Confirm existing `fixtures/categorical/{cat_onehot,cat_manyvsmany}` goldens load

*Planner refines against the RESEARCH.md Validation Architecture section.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-ROCm (`cubecl-hip`, f32) categorical smoke | ODL-22 | Local GPU is a spoofed 8-CU APU; best-effort, non-blocking (D-04) | Run on real hardware if available; pin to cpu anchor, informative not gating |

*Full real-hardware validation is Phase 23's Kaggle DoD.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
