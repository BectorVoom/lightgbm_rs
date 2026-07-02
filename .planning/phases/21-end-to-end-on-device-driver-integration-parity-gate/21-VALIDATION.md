---
phase: 21
slug: end-to-end-on-device-driver-integration-parity-gate
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-02
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (workspace) |
| **Config file** | none — workspace `Cargo.toml` |
| **Quick run command** | `cargo test -p lgbm-compute histogram_arena` |
| **Full suite command** | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness learner_parity && cargo test --workspace` |
| **Estimated runtime** | ~60–180 seconds |

---

## Sampling Rate

- **After every task commit:** Run the relevant quick command for the touched crate
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite must be green (default lane AND `LGBM_CUDA_ON_DEVICE=1` lane)
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | ODL-18H | — | N/A | unit | `cargo test -p lgbm-compute swap_multileaf_never_aliases_live_sibling_slot swap_errors_when_pool_exhausted` | ✅ | ⬜ pending |
| 21-03-01 | 03 | 1 | ODL-18, ODL-19 | — | N/A | doc/grep | `grep -q ODL-18H .planning/REQUIREMENTS.md` | ✅ | ⬜ pending |
| 21-02-01 | 02 | 2 | ODL-18H | — | N/A | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness learner_parity_on_device` | ✅ | ⬜ pending |

*Wave column matches plan frontmatter: Wave 1 = 21-01 + 21-03 (parallel, disjoint files_modified); Wave 2 = 21-02 (depends_on 21-01's `_with_cfg`).*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* The WR-01 repro tests (`swap_multileaf_never_aliases_live_sibling_slot`, `swap_errors_when_pool_exhausted`) already exist in `lgbm-compute`; the STRUCTURE gate (`learner_parity_on_device_structure_gate`) already exists in `oracle-harness`. This phase extends existing test files — no new framework install.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-ROCm (`cubecl-hip`, f32, ~1e-6) smoke pinned to cpu-f64 anchor | ODL-18H | Requires ROCm hardware; spoofed 8-CU APU + f32 non-determinism makes it informative-not-blocking (D-04) | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --features hip` on a real ROCm box; compare STRUCTURE to cpu-f64 anchor, never GPU-vs-GPU |

*All merge-gating behaviors have automated verification on the default cubecl-cpu lane.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (none — existing infra covers all)
- [x] No watch-mode flags
- [x] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-02
