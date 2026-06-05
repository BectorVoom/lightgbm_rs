---
phase: 4
slug: compute-backend-cpu-first-integer-histograms-rocm
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-05
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + `oracle-harness` comparators |
| **Config file** | none — Cargo workspace convention |
| **Quick run command** | `cargo test -p lgbm-compute` |
| **Full suite command** | `cargo test --workspace` |
| **ROCm gate (separate)** | `cargo test -p lgbm-compute --features rocm` (run on the local ROCm GPU) |
| **Estimated runtime** | ~30–60 seconds (CPU); ROCm run separate |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full workspace must be green on the CPU (cubecl-cpu) path — the hard gate (D-03)
- **ROCm gate (separate, D-03a):** `cargo test -p lgbm-compute --features rocm` executed on the local GPU; residual ~1e-6 gaps documented as known issues, not silent passes
- **Max feedback latency:** ~60 seconds (CPU)

---

## Per-Task Verification Map

| Task ID | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-W0-spike | 0 | D-04a | — | N/A | spike | `cargo test -p lgbm-compute determinism_spike` | ❌ W0 (RUN FIRST) | ⬜ pending |
| 04-CMP-01 | — | CMP-01 | — | N/A | unit/grep | `cargo test -p lgbm-compute` + dep-guard test grepping no `cubecl` above lgbm-compute | ❌ W0 | ⬜ pending |
| 04-CMP-02 | — | CMP-02 | T-04 V5 | bin indices validated `[0,num_bin)` → typed `ComputeError` | integration | `cargo test -p oracle-harness --test kernel_parity` | ❌ W0 | ⬜ pending |
| 04-CMP-03 | — | CMP-03 | — | N/A | integration | `cargo test -p lgbm-compute --features rocm` | ❌ W0 | ⬜ pending |
| 04-CMP-04 | — | CMP-04 | — | N/A | unit | `cargo test -p lgbm-compute capability` | ❌ W0 | ⬜ pending |
| 04-CMP-05 | — | CMP-05 | T-04 V5 | array lengths consistent; `out` sized `2*num_bin` f64 | integration | `cargo test -p oracle-harness --test kernel_parity` | ❌ W0 | ⬜ pending |
| 04-ORA-04 | — | ORA-04 | — | N/A | integration | `cargo test --workspace` (cpu, hard gate) + rocm-feature run (documented) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Wave/Task IDs are provisional — the planner assigns final plan/wave numbers.*

---

## Wave 0 Requirements

- [ ] `crates/lgbm-compute/tests/determinism_spike.rs` — the **D-04a bit-determinism spike** (RUN FIRST, before building the kernel suite; validates the cubecl-cpu single-owner ordered-fold bit-exactness bet)
- [ ] `crates/lgbm-compute/src/error.rs` — `ComputeError` (thiserror), boundary input validation (V5)
- [ ] `crates/lgbm-compute/src/runtime.rs` — runtime selection (cubecl-cpu / cubecl-hip) + startup capability gate (`Plane::Ops`, f64, atomics)
- [ ] `crates/oracle-harness/tests/kernel_parity.rs` — golden replay (layered histogram / best-split / data-partition goldens)
- [ ] `crates/oracle-harness/src/comparator.rs` — confirm `compare_exact_f64_bits` exported; add multi-bin helper if needed
- [ ] `xtask/cpp/kernel_capture.cpp` + `xtask` `kernel-capture` subcommand — header-only transcription of `ConstructHistogram` + `FindBestThreshold*` + `Split`
- [ ] `tests/fixtures/kernels/` — committed synthetic-input goldens (D-02a path coverage: dense+sparse, default-bin skip, missing/zero routing, bit-width variants, grad/hess sign/magnitude spread)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm ~1e-6 parity on physical GPU | CMP-03, ORA-04 | Requires the local ROCm GPU (gfx1100 / ROCm 7.1); not available in CI / CPU-only builds | On the ROCm host: `cargo test -p lgbm-compute --features rocm`; record any residual gap vs the cubecl-cpu anchor in VERIFICATION.md (D-03a — no silent pass) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (incl. the D-04a spike FIRST)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s (CPU)
- [ ] `nyquist_compliant: true` set in frontmatter (after planner maps every task)

**Approval:** pending
