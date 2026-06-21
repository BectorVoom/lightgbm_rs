---
phase: 05
slug: tree-learner-split-finding
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-06
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: rendered from `05-RESEARCH.md` § Validation Architecture (the authoritative per-task contract).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + `oracle-harness` golden comparators (`compare_exact_f64_bits`, `%.17g` model-text compare) |
| **Config file** | none — workspace `cargo test`; committed fixtures under `tests/fixtures/learner/` |
| **Quick run command** | `cargo test -p lgbm-treelearner` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds (golden replay is bit-compare, not retraining) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-treelearner` (unit level: leaf queue, gates, partition, col_sampler, fix_histogram)
- **After every plan wave:** Run `cargo test --workspace` (full golden replay + cross-workspace regression)
- **Before `/gsd-verify-work`:** Full suite green AND per-split AND per-tree goldens replay bit-exact
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | TRL-05 | — | N/A (internal numeric) | unit/build | `cargo build -p lgbm-compute` (asserts `cfg_skip_default_bin` removed) | ✅ extends lgbm-compute | ⬜ pending |
| 05-01-02 | 01 | 1 | TRL-05 | — | N/A (internal numeric) | integration (golden) | `cargo test -p oracle-harness --test kernel_parity kernel_parity_split_bit_exact_on_cpu` | ✅ extends kernel_parity | ⬜ pending |
| 05-02-01 | 02 | 1 | TRL-01 | — | N/A (internal numeric) | unit | `cargo test -p lgbm-model tree_split` | ✅ extends tree.rs | ⬜ pending |
| 05-02-02 | 02 | 1 | TRL-01, TRL-04 | T-05-02 (g/h/config ingest boundary) | typed `TreeLearnerError` on malformed input, no panic | unit | `cargo test -p lgbm-treelearner` | ❌ W0 (new crate) | ⬜ pending |
| 05-02-03 | 02 | 1 | TRL-01, TRL-04 | T-05-02 (fixture file IO) | capture reads read-only `LightGBM/`; never `git add`-ed | integration (harness) | `cargo test -p oracle-harness --test learner_parity` | ❌ W0 (new harness) | ⬜ pending |
| 05-03-01 | 03 | 2 | TRL-02, TRL-07 | — | N/A (internal numeric) | unit | `cargo test -p lgbm-treelearner` | ❌ W0 (depends 05-02) | ⬜ pending |
| 05-03-02 | 03 | 2 | TRL-01, TRL-02, TRL-03, TRL-04, TRL-05, TRL-07 | — | N/A (internal numeric) | unit + integration | `cargo test -p lgbm-treelearner leaf_wise_caps` | ❌ W0 (depends 05-02) | ⬜ pending |
| 05-03-03 | 03 | 2 | TRL-04 / D-02a, D-07 | — | N/A (internal numeric) | integration (per-split + per-tree golden, cross-check) | `cargo test -p oracle-harness --test learner_parity` | ❌ W0 (depends 05-02) | ⬜ pending |
| 05-04-01 | 04 | 3 | TRL-08, TRL-09 | — | N/A (internal numeric) | unit | `cargo test -p lgbm-treelearner col_sampler` | ❌ W0 (depends 05-03) | ⬜ pending |
| 05-04-02 | 04 | 3 | TRL-08, TRL-09 / D-03 | T-05-04-02 (force_col_wise divergence STOP-and-flag) | divergent tree → STOP + SUMMARY flag, never silent-ship | integration (tree equality + RNG parity golden) | `cargo test -p oracle-harness --test learner_parity` | ❌ W0 (depends 05-03) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Sampling continuity: no run of 3 consecutive implementation tasks lacks an automated verify — each plan's final task is a golden-replay integration gate.*

---

## Wave 0 Requirements

The learner has zero pre-existing test infrastructure; all rows below are created in Wave 1 (chiefly 05-02 Task 3) and consumed by Waves 2–3.

- [ ] `crates/lgbm-treelearner/` crate + `Cargo.toml` (new workspace member) — covers all TRL-* (05-02 Task 2)
- [ ] `xtask learner-capture` subcommand + `xtask/cpp/learner_capture.cpp` (header-only D-01/D-02 transcription) — emits per-split + per-tree goldens (05-02 Task 3, completed 05-03 Task 3)
- [ ] `tests/fixtures/learner/` committed goldens (per-split per-bin gain arrays + full-tree model-text) for synthetic + captured-g/h corpora
- [ ] `crates/oracle-harness/tests/learner_parity.rs` — the per-split + per-tree replay harness (05-02 Task 3)
- [ ] `REFERENCE_MANIFEST.md` extension — pin the learner fixture set + capture config (D-04 row/col, D-03 g/h source)
- [ ] `lgbm-model::Tree::split(...)` mutation method + growth-time arrays (`leaf_depth`, `leaf_parent`, `split_feature_inner`, `threshold_in_bin`) — D-07 full-tree golden enabler (05-02 Task 1)

*Comparators (`oracle-harness`), the `%.17g` formatter (`lgbm-model`), the capture pattern (`xtask`), and the Backend ops (`lgbm-compute`) all already exist and are reused.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| One-time golden regeneration | D-01 / D-04 | Requires the read-only `LightGBM/` reference on disk; not run in CI | `cargo xtask learner-capture` then assert `git diff tests/fixtures/learner/` is empty (idempotent regen) |
| ROCm gfx1100 learner re-check | TRL-01..09 (deferrable) | Needs the local ROCm GPU; CPU bit-exact is the hard gate (P4 D-03) | Run `cargo test --workspace --features rocm` on the gfx1100 host |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-06
