---
phase: 23
slug: perf-validation-default-on-rollout-dod
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-02
---

# Phase 23 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `23-RESEARCH.md` § Validation Architecture. Real-CUDA numbers (SC-1/SC-2/SC-3, ODL-20/21) are **Kaggle-only** by design — the local GPU is a spoofed 8-CU APU. Everything else is locally testable on the `cpu`/`rocm` feature builds.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + `oracle-harness` integration tests |
| **Config file** | Workspace `Cargo.toml`; tests in `crates/oracle-harness/tests/*.rs`, `crates/lgbm-compute/tests/*.rs` |
| **Quick run command** | `cargo test -p lgbm-compute cuda_on_device` |
| **Full suite command** | `cargo test --workspace` (CPU f64 merge gate — hard gate, SC-4) |
| **Real-CUDA gate** | Kaggle harness (Python), `boomvector` — **not runnable locally** |
| **Estimated runtime** | ~local suite minutes; Kaggle kernel run out-of-band |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute cuda_on_device` (fast resolver/default checks)
- **After every plan wave:** Run `cargo test --workspace` (full CPU f64 merge gate — SC-4)
- **Before `/gsd-verify-work`:** Full suite green locally + Kaggle A/B results artifact committed
- **Max feedback latency:** local suite runtime (Kaggle A/B is asynchronous, out-of-band)

---

## Per-Task Verification Map

| Req / SC | Behavior | Test Type | Automated Command | Locally testable? |
|----------|----------|-----------|-------------------|-------------------|
| D-01 tri-state | unset→default, `"0"`→off, `"1"`→on; read-once OnceLock | unit | `cargo test -p lgbm-compute cuda_on_device_override` | ✅ (env-set in-process) |
| D-02/D-03 CUDA-only default | cpu/rocm feature ⇒ `on_device_default()==false`; cuda ⇒ true | unit + cfg | `cargo test -p lgbm-compute on_device_default` | ✅ cpu/rocm arm; ⚠️ cuda arm needs `-F cuda` build |
| SC-4 byte-unchanged | env-unset merge gate identical; ROCm/CPU unchanged | integration | `cargo test --workspace` (`learner_parity`, `score_updater_parity`, `resident_score_ab`) | ✅ |
| L-2 single-source | learner + compute + boosting agree on the toggle | unit/integration | `cargo test -p oracle-harness resident_score_ab score_updater` | ✅ |
| D-11 parity (structural) | on-device tree bit-exact to cpu-f64 anchor | integration | `cargo test -p oracle-harness learner_parity` (`LGBM_CUDA_ON_DEVICE=1` guard) | ✅ (cpu anchor lane) |
| L-1 launch capture | on-device driver emits non-zero `device_launches` | unit/integration | new test asserting driver bumps the launch counter | ✅ (cpu build exercises driver) |
| ODL-20 / SC-1 / SC-2 | real-CUDA `device_launches/tree` < 8,570/100 + wall-clock ratios both shapes | e2e (manual, Kaggle) | Kaggle kernel → `poll_kaggle.sh` | ❌ Kaggle only |
| ODL-21 / SC-3 | default flip contingent on ≤5%/both-shapes + parity | e2e (manual, Kaggle) + separate commit (D-09) | harness PASS verdict in results.md | ❌ Kaggle only |
| D-11 parity (real-CUDA) | on-device preds vs host-CUDA ≤1e-6 on actual data | e2e (manual, Kaggle) | assertion embedded in harness | ❌ Kaggle only |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] **L-1:** Instrument `grow_driver.rs` device launches into a phase_prof counter surfaced in the `COUNTS:` line — **blocks SC-2 measurement.** Add launch-count + a unit test.
- [ ] **L-2:** Reconcile `learner.rs:471 cuda_on_device_env()` with the compute resolver (single source of truth) — test that all three toggle sites agree.
- [ ] Unit tests for the tri-state resolver (`unset`/`"0"`/`"1"`/malformed) and `on_device_default()` per-feature — likely a new test file under `crates/lgbm-compute/tests/` or `#[cfg(test)]` in `lib.rs`.
- [ ] Committed Kaggle harness script + `results.{md,json}` schema — validated locally by a dry-run parse over a captured stderr fixture.

*Existing infrastructure already covers the merge gate, learner/score parity, and the `LGBM_CUDA_ON_DEVICE=1`-guarded on-device parity test.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-CUDA `device_launches/tree` + wall-clock A/B at 500k×50 and 100k×500 | ODL-20 / SC-1 / SC-2 | Requires real discrete NVIDIA GPU; local GPU is a spoofed 8-CU APU | Push Kaggle kernel as `boomvector`, `poll_kaggle.sh`, capture `results.{md,json}` |
| Default-on flip verdict (≤5%, both shapes, parity) | ODL-21 / SC-3 | Depends on the Kaggle A/B result; flip is a separate verdict-gated commit (D-09) | Read PASS/FAIL from committed results artifact; flip only on PASS |
| Real-CUDA end-to-end ≤1e-6 parity | D-11 | Needs real CUDA device | Assertion embedded in harness against host-CUDA/official predictions |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies (Kaggle-gated items documented as manual)
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers L-1 + L-2 + tri-state/default unit tests
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
