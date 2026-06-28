---
phase: 14
slug: scaffold-oracle-slice-0
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-28
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 14-RESEARCH.md §"Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + the `oracle-harness` integration crate |
| **Config file** | per-crate `Cargo.toml`; tests under `crates/*/tests/` and inline `#[test]` |
| **Quick run command** | `cargo test -p lgbm-treelearner && cargo test -p oracle-harness --test learner_parity` |
| **Full suite command** | `cargo test --workspace` (the full bit-exact merge gate) |
| **Estimated runtime** | quick ~30–90s; full workspace several minutes |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-treelearner && cargo test -p oracle-harness --test learner_parity`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full `cargo test --workspace` green AND byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset
- **Max feedback latency:** ~90 seconds (quick command)

---

## Per-Task Verification Map

> Task IDs (`14-NN-MM`) are assigned by the planner. Rows below map each phase
> requirement / success criterion to its observable test; the planner binds each
> to the concrete task that produces it.

| Req / SC | Behavior | Test Type | Automated Command | File Exists | Status |
|----------|----------|-----------|-------------------|-------------|--------|
| ODL-01 / SC#1 | Merge gate byte-unchanged, `LGBM_CUDA_ON_DEVICE` unset: CPU/ROCm/host-CUDA grow identical trees | integration (existing gate) | `cargo test -p oracle-harness --test raw_bin_train_parity` (`raw_bin_train_matches_cpp_golden`) + `--test learner_parity` | ✅ existing | ⬜ pending |
| ODL-01 / SC#2 | `grow_tree_on_device` seam + default-false `on_device_growth_supported()` exist; `train_inner` fork reachable; `GpuBackend<R>` override returns `Ok(None)`/no-op; default path untouched | new test + existing gate | `cargo test -p oracle-harness --test learner_parity` (new: force-eligible ⇒ fork returns the host-fallback tree) | ❌ W0 | ⬜ pending |
| ODL-02 / SC#3 | `assert_on_device_tree_matches_cpu_anchor` tie-aware oracle compiles + passes against the host-fallback tree (structure bit-exact + ~1e-5 leaf envelope + tie-aware `default_left`) | new test | `cargo test -p oracle-harness --test learner_parity` (new oracle test) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] New oracle test in `crates/oracle-harness/tests/learner_parity.rs` — exercises the seam via host-fallback (`grow_tree_on_device(..)?.unwrap_or_else(host_grow)`, D-01) and the tie-aware comparator (covers ODL-02 / SC#3)
- [ ] New SC#2 test proving the `train_inner` fork is reachable and returns the host tree when forced eligible (covers ODL-01 / SC#2)
- [ ] Tie-aware comparator: generalize `assert_gpu_tree_matches_cpu_anchor` — decode `default_left` as bit1 of `decision_type` (mask=2) and lift `kernel_parity.rs:1597-1620` near-tie acceptance to per-node (covers ODL-02)
- [ ] Framework install: none — built-in harness present

*Existing infrastructure (oracle-harness, cpu f64 anchor, near-tie logic) covers the comparison machinery; only the three new tests above are net-new.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Byte-identical-to-master trees with `LGBM_CUDA_ON_DEVICE` unset | ODL-01 / SC#1 | "byte-identical to master" is a cross-commit diff, not a single in-tree assertion | Run `cargo test --workspace` on master and on this branch with the env unset; confirm the merge-gate suites (`raw_bin_train_matches_cpp_golden`, `learner_parity`, `kernel_parity`) produce identical pass sets. Automated suites give the regression signal; the byte-identity claim is confirmed by the green unchanged gate. |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
