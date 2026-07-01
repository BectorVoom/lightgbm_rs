---
phase: 20
slug: on-device-score-updater-metrics
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-02
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `oracle-harness` integration tests |
| **Config file** | none (Cargo default) |
| **Quick run command** | `cargo test -p lgbm-compute` |
| **Full suite command** | `cargo test --workspace` (the merge gate, env unset) |
| **On-device gated run** | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness` |
| **Estimated runtime** | ~90 seconds (quick) / several min (full workspace) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute` (fast kernel units)
- **After every plan wave:** Run `cargo test --workspace` (env unset — proves byte-unchanged) AND `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness`
- **Before `/gsd-verify-work`:** Full workspace green (env unset) AND all three D-06 parity layers green (env set)
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 20-W0-01 | 00 | 0 | ODL-17 | — | length/bounds guards before launch | capture | `python xtask/py/metric_oracle_capture.py` (rmse/l2/l1/binary_logloss) | ❌ W0 | ⬜ pending |
| 20-01-xx | 01 | 1 | ODL-16 | T-20-01 | bounds-guard launcher; `usize` offset | unit + A/B | `cargo test -p oracle-harness score_updater` | ❌ W0 | ⬜ pending |
| 20-01-xx | 01 | 2 | ODL-16 | — | resident vs host mirror parity | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness resident_score_ab` | ❌ W0 | ⬜ pending |
| 20-02-xx | 02 | 1 | ODL-17 | T-20-01 | bounds-guard launcher | parity | `cargo test -p oracle-harness --test metric_parity` | ⚠ 8/12 (W0 fills 4) | ⬜ pending |
| 20-02-xx | 02 | 1 | ODL-17 | — | discriminator routes unsupported → host | unit | `cargo test -p lgbm-compute metric_supported` | ❌ W0 | ⬜ pending |
| 20-03-xx | 03 | 2 | ODL-18 | T-20-01 | STRUCTURE bit-exact to cpu f64 anchor | parity | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity on_device` | ✅ oracle exists | ⬜ pending |
| 20-03-xx | 03 | 2 | ODL-19 | — | no f64 per-row hot loop; env-unset byte-unchanged | grep + gate | `cargo test --workspace` (env unset) + `rg 'f64' <new kernels>` | ✅ merge gate | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Task IDs are provisional until the planner assigns them.*

---

## Wave 0 Requirements

- [ ] `xtask/py/metric_oracle_capture.py` — capture `rmse` / `l2` / `l1` / `binary_logloss` goldens (Pitfall 1: only 8/12 exist on disk)
- [ ] `crates/oracle-harness/tests/metric_parity.rs` — add on-device metric cells (currently host-only replay) covering ODL-17
- [ ] `crates/oracle-harness/tests/learner_parity.rs` — activate the STRUCTURE gate cell with a real on-device tree (currently Slice-0 host-fallback stand-in ~line 2445)
- [ ] New resident-score A/B test (ODL-16 D-06 layer 2) — no existing file
- [ ] New `metric_supported` discriminator unit test (mirror `device_objective.rs:114` supported/unsupported assertions)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-CUDA (non-APU) parity of the resident loop | ODL-18/19 | Local GPU is a spoofed 8-CU APU; discrete-CUDA numbers only via Kaggle | Deferred to Phase 23 perf-validation DoD; parity here validates on ROCm/cpu-anchor |

*All in-scope parity behaviors have automated verification anchored to the cpu f64 fold + real `lib_lightgbm` metric goldens.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (4 metric goldens + on-device cells + A/B + discriminator)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
