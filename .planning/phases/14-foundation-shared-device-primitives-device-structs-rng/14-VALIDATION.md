---
phase: 14
slug: foundation-shared-device-primitives-device-structs-rng
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-29
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> The detailed per-primitive anchor/assertion map lives in 14-RESEARCH.md `## Validation Architecture`; the planner lifts it into per-task `<acceptance_criteria>` and `must_haves`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust workspace) |
| **Config file** | none — workspace `Cargo.toml` |
| **Quick run command** | `cargo test -p lgbm-compute` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~{TBD by planner — fill from observed} seconds |

---

## Sampling Rate

- **After every task commit:** Run the relevant crate's quick test (`cargo test -p <crate>`)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full merge gate green (`raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites)
- **Max feedback latency:** {TBD} seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {planner fills from RESEARCH §Validation Architecture} | | | ODL-01 / ODL-02 | — | N/A | unit | `cargo test ...` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Per-intrinsic-per-backend plane-op smoke test (Open Q1 from RESEARCH) — prove `plane_inclusive_sum`/`plane_exclusive_sum`/`plane_max`/`plane_min` lower on cubecl-cpu + cubecl-hip before authoring the primitives; fall back to `plane_shuffle_up` manual scan if any fails.
- [ ] C++ `__device__` fixture-capture harness (D-03) — `hipcc` shim over the in-repo AMD fork, committed golden fixtures for ShufflePrefixSum / ShuffleReduce* / BitonicArgSort* / PercentileDevice.

*Planner finalizes the exact Wave 0 test stubs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm/CUDA f32 ~1e-6 parity on real GPU | ODL-01 | local GPU is a spoofed APU; discrete-CUDA numbers need Kaggle | see memory `kaggle-cli-cuda-bench` |

*Numeric/permutation anchoring is otherwise automated against the cpu f64 fold + committed C++ fixtures.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (plane-op smoke test, fixture harness)
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
