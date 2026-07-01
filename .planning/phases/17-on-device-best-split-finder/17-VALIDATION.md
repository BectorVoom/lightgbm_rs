---
phase: 17
slug: on-device-best-split-finder
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-01
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (`oracle-harness` integration tests + `lgbm-compute` unit tests) |
| **Config file** | none (Cargo built-in) |
| **Quick run command** | `cargo test -p lgbm-compute --lib best_split` |
| **Full suite command** | `cargo test -p oracle-harness --test best_split_parity && cargo test --workspace` |
| **Estimated runtime** | ~120 seconds (workspace); quick cpu-fold unit tests ~5s |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute --lib best_split` (fast cpu-fold unit tests)
- **After every plan wave:** Run `cargo test -p oracle-harness --test best_split_parity`
- **Before `/gsd-verify-work`:** Full suite green — esp. `learner_parity`, `kernel_parity`, `raw_bin_train_matches_cpp_golden` unregressed + hip `rocm_backend_parity` within ~1e-6
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| W0 | 00 | 0 | ODL-11/12 | — | N/A | harness | `cargo test -p oracle-harness --test best_split_parity` | ❌ W0 | ⬜ pending |
| stage1 | — | 1 | ODL-11 | — | N/A | integration (golden) | `cargo test -p oracle-harness --test best_split_parity stage1` | ❌ W0 | ⬜ pending |
| count-recovery | — | 1 | ODL-11 | T-17-V5 | reject bad `num_bin`/`num_tasks` at launch | unit | `cargo test -p lgbm-compute --lib count_recovery_ties_even` | ❌ W0 | ⬜ pending |
| flag-goldens | — | 1 | ODL-11 | — | N/A | integration | `cargo test -p oracle-harness --test best_split_parity flags` | ❌ W0 | ⬜ pending |
| globalmem | — | 1 | ODL-11 | T-17-V5 | size-validate spill scratch | integration | `cargo test -p oracle-harness --test best_split_parity globalmem` | ❌ W0 | ⬜ pending |
| stage3-export | — | 2 | ODL-12 | — | N/A | integration | `cargo test -p oracle-harness --test best_split_parity stage3_export` | ❌ W0 | ⬜ pending |
| default_left-tie | — | 2 | ODL-12 | — | N/A | integration (hip) | `cargo test -p lgbm-compute --test rocm_backend_parity default_left_tie` | ⚠️ extend | ⬜ pending |
| merge-gate | — | 2 | ODL-19 | T-17-perf | no f64 per-row loop (grep) | gate | `cargo test --workspace` + grep audit | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/oracle-harness/tests/best_split_parity.rs` — the golden anchor harness (mirror `kernel_parity.rs` parse+assert shape) — covers ODL-11/ODL-12
- [ ] `crates/oracle-harness/tests/fixtures/kernels/best_split.txt` — the 6 golden categories (§D-07: default-template, USE_L1, USE_SMOOTHING, USE_RAND, empty/sparse-default-bin, global-memory spill)
- [ ] `crates/lgbm-compute/src/kernels/best_split.rs` unit tests — count-recovery ties-even, epsilon placement, task-gen `assume_out_default_left` table
- [ ] Extend `crates/lgbm-compute/tests/rocm_backend_parity.rs` — tie-aware default_left on hip
- [ ] Confirm `InitCUDARandomKernel` seed formula before the USE_RAND golden (Open Q1)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| hip f32 mirror within ~1e-6 of cpu f64 anchor | ODL-12 | Requires a ROCm device (local spoofed 8-CU APU); parity gate valid, perf numbers APU-confounded | `cargo test -p lgbm-compute --test rocm_backend_parity` on the ROCm host |

*All other phase behaviors have automated verification against the cubecl-cpu f64 fold.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (best_split_parity harness + fixtures)
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
