---
phase: 22
slug: on-device-categorical-splits-feature-coverage
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-02
---

# Phase 22 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Filled from the 5 produced plans (22-01..22-05, 4 waves, 11 tasks) after plan verification.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust built-in harness) + `oracle-harness` integration crate |
| **Config file** | none (Cargo default; workspace `Cargo.toml`) |
| **Quick run command** | `cargo test -p lgbm-compute --lib categorical_split` (new-module unit tests) |
| **Full suite command** | `cargo test --workspace` (env unset) + `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity` |
| **Merge gate** | cubecl-cpu f64 lane; `cargo test --workspace` env-unset stays byte-green (SC #4) |
| **Estimated runtime** | workspace suite dominated by oracle-harness parity (minutes) |

---

## Sampling Rate

- **After every task commit:** Run that task's `<automated>` command (per-task map below)
- **After every plan wave:** Run `cargo test --workspace` (env unset) + the gated `LGBM_CUDA_ON_DEVICE=1` parity subset for that wave
- **Before `/gsd-verify-work`:** Full suite green on the default cubecl-cpu f64 lane (the hard merge gate)
- **Max feedback latency:** single-crate `--lib` tests are fast (seconds); full parity is the pre-verify gate

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 22-01-01 | 01 | 1 | ODL-22 | T-22-04 | No OOB write on width>32 slab; no silent truncation | unit | `cargo test -p lgbm-compute --lib split_info` (`cat_slab_width_gt_32_no_truncation`) | ❌ W0 | ⬜ pending |
| 22-01-02 | 01 | 1 | ODL-22 | — | Honest host-fallback (D-06), no silent wrong answer | unit | `cargo test -p lgbm-treelearner --lib on_device_eligible_false_for_categorical_plus_quantized` | ❌ W0 | ⬜ pending |
| 22-02-01 | 02 | 1 | ODL-22 | T-22-04 | Crate-cycle-safe (native primitives only) | build+unit | `cargo build -p lgbm-compute && cargo test -p lgbm-compute --lib grow_driver` | ✅ | ⬜ pending |
| 22-02-02 | 02 | 1 | ODL-22 | — | N/A | build+integration | `cargo build -p oracle-harness --tests && LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- learner_parity_on_device_structure_gate` | ✅ | ⬜ pending |
| 22-03-01 | 03 | 2 | ODL-22 | — | Bitset bits match reference (§6.3) | unit | `cargo test -p lgbm-compute --lib categorical_split` (`construct_bitset`, `set_real_threshold`) | ❌ W0 | ⬜ pending |
| 22-03-02 | 03 | 2 | ODL-22 | — | Eval matches host f64 anchor (one-hot + many-vs-many) | unit (tdd) | `cargo test -p lgbm-compute --lib categorical_split` (`onehot`, `manyvsmany`) | ❌ W0 | ⬜ pending |
| 22-04-01 | 04 | 3 | ODL-22 | — | Numeric best-split unchanged; kEpsilon pass-through | build+unit | `cargo build -p lgbm-compute && cargo test -p lgbm-compute --lib best_split` | ✅ | ⬜ pending |
| 22-04-02 | 04 | 3 | ODL-22 | T-22-12 | Numeric spine untouched; single kEpsilon bump site | build+unit | `cargo build -p lgbm-compute && cargo test -p lgbm-compute --lib grow_driver` | ✅ | ⬜ pending |
| 22-04-03 | 04 | 3 | ODL-22 | — | Routing-convention correct (Open Q1) | unit | `cargo test -p lgbm-compute --lib categorical_partition_counts_match_host_stable` | ❌ W0 | ⬜ pending |
| 22-05-01 | 05 | 4 | ODL-22 | T-22-15 | Device grow routes categorical rows (cpu-f64 structure gate, tie-aware default_left) | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- learner_parity_on_device_structure_gate` | ❌ W0 | ⬜ pending |
| 22-05-02 | 05 | 4 | ODL-22 | T-22-15 | Real 4.6 golden bit-exact (bitset/dt&1/num_cat/cat_threshold) + predict-through + numeric no-regression | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- learner_parity_categorical` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**D-01 dual anchor:** golden bit-exactness lives in 22-05-02; the cubecl-cpu f64 structure gate (tie-aware `default_left`) lives in 22-05-01. Both are required — neither alone.

---

## Wave 0 Requirements

- [x] Categorical corpus cases added to `crates/oracle-harness/tests/learner_parity.rs` structure gate (one-hot + many-vs-many) — planned in 22-02-02 (scaffold) / 22-05-01 (cases)
- [x] `max_cat_threshold > 32` unit test to prove D-03 slab sizing non-vacuously — planned in 22-01-01 (`cat_slab_width_gt_32_no_truncation`; both committed goldens use 32)
- [x] Confirm existing `fixtures/categorical/{cat_onehot,cat_manyvsmany}` goldens load — consumed in 22-05-02 (already committed; SKIP-passes if absent, learner_parity.rs:1387)
- [x] Open-Q1 partition-count isolation test (`categorical_partition_counts_match_host_stable`) — planned in 22-04-03

*Boxes reflect coverage by the produced plans, not execution completion (`wave_0_complete` flips true once Wave 1–0 test scaffolding is committed green).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-ROCm (`cubecl-hip`, f32) categorical smoke | ODL-22 | Local GPU is a spoofed 8-CU APU; best-effort, non-blocking (D-04) | Run on real hardware if available; pin to cpu anchor, informative not gating |

*Full real-hardware validation is Phase 23's Kaggle DoD.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (all 11 tasks carry an `<automated>` command)
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency acceptable (per-task `--lib` tests are fast; full parity is the pre-verify gate)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-02
