---
phase: 16
slug: on-device-histogram-constructor
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-01
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (workspace) |
| **Config file** | none — Cargo workspace; tests live in `crates/lgbm-compute/tests/` + `#[cfg(test)]` modules |
| **Quick run command** | `cargo test -p lgbm-compute` |
| **Full suite command** | `cargo test` (the full merge gate — must stay green, ODL-19) |
| **Estimated runtime** | ~quick: tens of seconds; full: minutes |

*Note: on-device (`LGBM_CUDA_ON_DEVICE`) parity tests requiring the cubecl-hip backend run only where the ROCm GPU is present; the cpu f64-fold anchor tests run everywhere and are the hard gate. Never GPU-vs-GPU (def-f8u-01).*

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute`
- **After every plan wave:** Run `cargo test` (full merge gate)
- **Before `/gsd-verify-work`:** Full suite must be green; default paths byte-unchanged
- **Max feedback latency:** ~60 seconds (quick crate-scoped run)

---

## Per-Task Verification Map

> Filled by the planner from PLAN.md tasks. Anchor every numeric output to the cubecl-cpu f64 fold (bit-exact structure; ROCm/CUDA f32 within ~1e-6).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 0 | ODL-09/10 | — | N/A (internal compute) | fixture | `cargo test -p lgbm-compute` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

> From RESEARCH.md Validation Architecture — test gaps to land before/with the build kernel:

- [ ] Synthetic **sparse** fixture (forces `row_ptr_type` {16,32,64}) — reuse Phase-15 synthetic sparse column
- [ ] Synthetic **large-bin / global-spill** fixture (`NumLargeBinPartition() > 0`) — reuse Phase-15 synthetic large-bin column
- [ ] Purpose-built **`most_freq_bin ≠ 0`** column to force `FixHistogram`'s omit-and-repair path (DEF-07-02); anchor the repaired default-bin value (`leaf_total − scanned Σ`)
- [ ] **Build-smaller-before-subtract ordering-invariant** harness (the 8aed100-class guard: parent fully built/synced before any child subtract reads it)
- [ ] **Interleaved `[2b]/[2b+1]` layout** assert (grad at `2b`, hess at `2b+1`)
- [ ] Committed dense corpora build anchor (existing) — bit-exact to cpu f64 fold

*If none new: "Existing infrastructure covers all phase requirements." — NOT the case here; Wave 0 fixtures above are required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm/CUDA f32 parity within ~1e-6 | ODL-09/10 | Requires physical ROCm GPU backend | Run `LGBM_CUDA_ON_DEVICE=1 cargo test -p lgbm-compute --features hip` on the ROCm host; compare to cpu f64 anchor (never GPU-vs-GPU) |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
