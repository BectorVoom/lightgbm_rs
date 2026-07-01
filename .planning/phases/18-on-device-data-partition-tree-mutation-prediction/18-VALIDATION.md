---
phase: 18
slug: on-device-data-partition-tree-mutation-prediction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-07-01
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Anchor = cubecl-cpu f64 fold (bit-exact merge gate); cubecl-hip f32 = ~1e-6, tie-aware, **never GPU-vs-GPU** (D-12 / def-f8u-01). Merge gate must stay green with `LGBM_CUDA_ON_DEVICE` unset (D-13 / ODL-19).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (`oracle-harness` integration tests) + golden-fixture replay |
| **Config file** | none — `[[test]]` targets under `crates/oracle-harness/tests/` |
| **Quick run command** | `cargo test -p oracle-harness partition_parity --features cpu` (per new module) |
| **Full suite command** | `cargo test --workspace` (ODL-19 hard merge gate — green with `LGBM_CUDA_ON_DEVICE` unset) |
| **Estimated runtime** | ~90–180 seconds (workspace) |

---

## Sampling Rate

- **After every task commit:** Run the relevant `partition_parity` / `tree_mutation_parity` / `predict_parity` module (`cargo test -p oracle-harness <module>`).
- **After every plan wave:** Run `cargo test --workspace` (env unset).
- **Before `/gsd-verify-work`:** Full suite green + hip parity module (`--features hip`) when a GPU is present.
- **Max feedback latency:** ~180 seconds.

Scaffold tests land `#[ignore = "Wave-0 scaffold"]` and are un-ignored when the numeric core lands (Phase-17 pattern).

---

## Per-Task Verification Map

> Task IDs are illustrative until PLAN.md files are written; the planner binds each requirement to concrete tasks.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 18-00-01 | 00 | 0 | ODL-13 | T-18-01 | u16/u32 scan bounds guarded (`validate_scan_inputs`) | unit | `cargo test -p lgbm-compute primitives::int_scan` | ❌ W0 | ⬜ pending |
| 18-01-01 | 01 | 1 | ODL-13 | T-18-01 | scatter `global_thread_index < num_data` guard preserved | unit/golden | `cargo test -p oracle-harness partition_parity` | ❌ W0 | ⬜ pending |
| 18-01-02 | 01 | 1 | ODL-13 | — | membership `n`-bound guard in `FindInBitsetCUDA` | unit/golden | `cargo test -p oracle-harness partition_parity::cat` | ❌ W0 | ⬜ pending |
| 18-01-03 | 01 | 1 | ODL-13 | — | 16-int packet fields match reference | unit/golden | `cargo test -p oracle-harness partition_parity::packet` | ❌ W0 | ⬜ pending |
| 18-01-04 | 01 | 1 | ODL-13 | — | leaf-indexed pool swap = subtraction reuse | unit | `cargo test -p lgbm-compute histogram_arena::swap` | ❌ W0 | ⬜ pending |
| 18-02-01 | 02 | 1 | ODL-14 | — | SplitKernel field writes + Split-before-partition ordering | unit/golden | `cargo test -p oracle-harness tree_mutation_parity` | ❌ W0 | ⬜ pending |
| 18-02-02 | 02 | 1 | ODL-14 | — | Shrinkage / AddBias | unit | `cargo test -p lgbm-compute tree::shrinkage` | ❌ W0 | ⬜ pending |
| 18-03-01 | 03 | 2 | ODL-15 | — | numeric tree-walk (8/16/32) vs f64 anchor + lib_lightgbm cross-check | integration/golden | `cargo test -p oracle-harness predict_parity::on_device` | ⚠️ extend | ⬜ pending |
| 18-03-02 | 03 | 2 | ODL-15 | — | categorical membership predict | integration/golden | `cargo test -p oracle-harness predict_parity::cat` | ⚠️ extend | ⬜ pending |
| 18-xx-99 | — | gate | ODL-19 | — | merge gate green env-unset; no f64 per-row loops (grep) | gate | `cargo test --workspace` + grep | ✅ existing | ⬜ pending |
| 18-hip-01 | — | gate | ODL-13/15 | — | hip f32 within ~1e-6 vs cpu f64 (tie-aware) | integration (hip) | `cargo test -p oracle-harness --features hip kernel_parity_partition` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/oracle-harness/tests/partition_parity.rs` — ODL-13 (row order, categorical, 16-int packet); mirror `best_split_parity.rs` idioms (raw-f64-bits parse, graceful SKIP, `#[ignore]` scaffold).
- [ ] `crates/oracle-harness/tests/tree_mutation_parity.rs` — ODL-14 (SplitKernel field writes, ordering).
- [ ] Extend `crates/oracle-harness/tests/predict_parity.rs` — on-device tree-walk (ODL-15), numeric + categorical.
- [ ] Extend `xtask/cpp/kernel_capture.cpp` — full flag fan-out + categorical routing + 16-int packet + tree-walk predict goldens (D-11); regenerate `tests/fixtures/kernels/partition.txt` (+ predict fixture).
- [ ] `HistArena` leaf-indexed pool-swap unit test in `lgbm-compute`.
- [ ] Wave-0 u16/u32 scan-lowering parity spike on hip (Open Q1 / A1) — fall back to u32-widen if u16 doesn't lower (parity-neutral).
- [ ] No new framework install — `cargo test` + golden fixtures already in place.

---

## Golden Provenance Limitation (IN-01)

> Raised by the phase-18 code review (`18-REVIEW.md`, IN-01). Recorded here so the
> numerical-fidelity contract is read with the right caveat.

The partition (`SplitRouteFanout`), categorical (`SplitCategoricalRoute` /
`FindInBitsetHost`) and predict (`PredWalk`) goldens in `xtask/cpp/kernel_capture.cpp`
(≈`:512`, `:585`, `:598`, `:1437`) are **hand-written host re-transcriptions** of
`dense_bin.hpp` / `cuda_tree.cu`, **not** captures from the compiled reference
(`lib_lightgbm` 4.6 / the AMD-fork CUDA path). The Rust kernels (`route_to_left`,
`route_left_host`, the predict walk) are independent transcriptions of the *same* C++
source lines.

**Implication:** the phase-18 partition / tree-mutation / predict parity cells prove the
two transcriptions **agree with each other** — not that either matches the *compiled*
reference. A transcription error shared by both sides (e.g. a remapped-`bin`-vs-raw
`default_bin` comparison that both omit, `predict.rs:130` / `kernel_capture.cpp:1462`)
would pass green. This differs from the project's binning / serial-tree-learner goldens,
which ARE captured from real `lib_lightgbm` 4.6 and are therefore reference-faithful.
The CLAUDE.md fidelity contract is stated against the *compiled* reference, so these
phase-18 goldens are a weaker (transcription-consistency) guarantee until pinned.

**Remediation (deferred, not blocking — kernels are OFF behind `LGBM_CUDA_ON_DEVICE` /
`on_device_growth_supported() == false`):** where the AMD-fork CUDA path can be built on
the Kaggle CUDA harness, capture at least one PORDER / predict golden from the compiled
kernel to pin the transcription against the real reference. Tracked below under
Manual-Only Verifications ("Real `lib_lightgbm` 4.6 golden capture").

See also MEMORY: `on-device-kernel-goldens-are-retranscriptions`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| hip f32 ~1e-6 parity | ODL-13/15 | Requires a ROCm GPU present; CI anchor is cpu f64 | Run `cargo test -p oracle-harness --features hip kernel_parity_partition` on the local ROCm box |
| Real `lib_lightgbm` 4.6 golden capture | ODL-13/15 (D-11) | Needs the untracked reference tree built locally | Build lib_lightgbm 4.6 in-tree, run `xtask` capture, commit regenerated fixtures |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
