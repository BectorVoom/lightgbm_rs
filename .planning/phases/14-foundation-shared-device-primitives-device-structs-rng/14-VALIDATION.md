---
phase: 14
slug: foundation-shared-device-primitives-device-structs-rng
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-29
approved: 2026-06-29
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
| **Estimated runtime** | observed full-suite wall-time recorded by 14-06 Task 3 in 14-06-SUMMARY (backfills this field) |

---

## Sampling Rate

- **After every task commit:** Run the relevant crate's quick test (`cargo test -p <crate>`)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full merge gate green (`raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites)
- **Max feedback latency:** per-task quick command is a single crate/test target (seconds); full-suite latency recorded by 14-06 Task 3 in 14-06-SUMMARY.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 14-01-T1 plane-intrinsic smoke (Open Q1) | 14-01 | 1 | ODL-01 | T-14-01-01 (accept) | V5: checked `::launch`, fixed host slices, no OOB surface | unit (Wave-0 de-risk) | `cargo test -p lgbm-compute --test plane_intrinsic_smoke` | ❌ W0 (created here) | ⬜ pending |
| 14-01-T2 scaffold 3 kernel modules + barrel | 14-01 | 1 | ODL-01, ODL-02 | — | N/A (compile-only stubs) | build | `cargo build -p lgbm-compute && cargo test -p lgbm-compute --no-run` | ❌ W0 (created here) | ⬜ pending |
| 14-02-T1 C++/HIP primitive_capture.cu harness | 14-02 | 1 | ODL-01 | T-14-02-02 (accept) | off-build dev tool; seeded synthetic inputs only | infra (harness) | `test -f xtask/cpp/primitive_capture.cu && grep -q "__global__" xtask/cpp/primitive_capture.cu` | ❌ W0 (created here) | ⬜ pending |
| 14-02-T2 CMake target + xtask subcommand + committed goldens | 14-02 | 1 | ODL-01 | T-14-02-01 (mitigate) | byte-idempotent capture (one MASTER_SEED); re-run → empty `git diff` | infra (fixtures) | `cargo build -p xtask && cargo run -p xtask -- primitive-capture && git diff --quiet crates/oracle-harness/fixtures/primitives` | ❌ W0 (created here) | ⬜ pending |
| 14-03-T1 block + global prefix-sum (incl/excl) | 14-03 | 2 | ODL-01 | T-14-03-01/02 (mitigate) | V5 length validation + scratch sizing in `usize` before `launch_unchecked` | unit (tdd) | `cargo test -p lgbm-compute --test primitives_self prefix_sum` | ✅ self-test created here | ⬜ pending |
| 14-03-T2 shuffle reductions sum/max/min + dotprod | 14-03 | 2 | ODL-01 | T-14-03-01 (mitigate) | V5 length validation; anchor to serial f64 (Open Q2 policy) | unit (tdd) | `cargo test -p lgbm-compute --test primitives_self reduce` | ✅ self-test created here | ⬜ pending |
| 14-03-T3 single-block index-only bitonic argsort | 14-03 | 2 | ODL-01 | T-14-03-01 (mitigate) | index-only (values unmoved); permutation bit-exact, tie-locked | unit (tdd) | `cargo test -p lgbm-compute --test primitives_self argsort` | ✅ self-test created here | ⬜ pending |
| 14-04-T1 SoA device split-record (alloc-once + slot-copy) | 14-04 | 2 | ODL-02 | T-14-04-01/03 (mitigate) | host slot-index validation; slab sizing in `usize`; no per-split device alloc | unit | `cargo test -p lgbm-compute --test split_info` | ✅ self-test created here | ⬜ pending |
| 14-04-T2 CUDARandom #[cube] LCG (D-04 parity) | 14-04 | 2 | ODL-02 | T-14-04-02 (mitigate) | V6 negative control: non-crypto PRNG, never security entropy | unit (tdd) | `cargo test -p lgbm-compute --test cuda_random_parity` | ✅ test created here | ⬜ pending |
| 14-05-T1 percentile skeleton (wtd/unwtd) | 14-05 | 3 | ODL-01 | T-14-05-01 (mitigate) | checked `::launch`; V5 length validation before launch | unit (skeleton) | `cargo test -p lgbm-compute --test primitives_self percentile` | ✅ self-test extended here | ⬜ pending |
| 14-05-T2 multi-block / global argsort skeleton | 14-05 | 3 | ODL-01 | T-14-05-01 (mitigate) | checked `::launch`; index-only; no cross-cube barrier assumption | unit (skeleton) | `cargo test -p lgbm-compute --test primitives_self argsort_global` | ✅ self-test extended here | ⬜ pending |
| 14-05-T3 per-segment items-sort skeleton (golden deferred → Phase 19) | 14-05 | 3 | ODL-01 | T-14-05-01 (mitigate) | checked `::launch`; V5 segment validation; convention locked via 14-02/14-03 single-block fixture | unit (skeleton) | `cargo test -p lgbm-compute --test primitives_self items_sort` | ✅ self-test extended here | ⬜ pending |
| 14-06-T1 no-op seam + anchor-pinned oracle (D-09/D-10) | 14-06 | 4 | ODL-01, ODL-02 | T-14-06-01 (mitigate) | seam FROZEN (`Ok(None)`); anchor to cpu f64, never GPU-vs-GPU | regression | `cargo test -p oracle-harness --test learner_parity` | ✅ exists (`learner_parity.rs`) | ⬜ pending |
| 14-06-T2 primitive fixture-parity replay vs C++ goldens (D-03/D-10) | 14-06 | 4 | ODL-01 | T-14-06-02 (mitigate) | skip-if-absent logged; int/perm bit-exact, no silent tolerance widening | unit (parity) | `cargo test -p oracle-harness --test primitive_parity` | ❌ W0 (created here, replays 14-02 fixtures) | ⬜ pending |
| 14-06-T3 full merge-gate verification (D-11) | 14-06 | 4 | ODL-01, ODL-02 | T-14-06-01 (mitigate) | `LGBM_CUDA_ON_DEVICE` OFF by default; existing paths byte-unchanged | integration | `cargo test --workspace` | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*File Exists: ❌ W0 = the test/harness is created within this phase (its absence is closed by the listed Wave-0 / owning task); ✅ = the gate already exists in-repo.*

---

## Wave 0 Requirements

The two MISSING references the downstream numeric tests depend on are both produced in **Wave 1** (this phase has no separate Wave 0 — the de-risk + fixture-capture work is the first wave, and every Wave-2+ test depends on it):

- [ ] **Per-intrinsic-per-backend plane-op smoke test (Open Q1)** — owned by **14-01 Task 1** (`crates/lgbm-compute/tests/plane_intrinsic_smoke.rs`). Proves `plane_inclusive_sum` / `plane_exclusive_sum` / `plane_max` / `plane_min` lower on cubecl-cpu (and cubecl-hip under `--features rocm`) before any primitive is authored; falls back to a `plane_shuffle_up` manual scan if any op fails to lower, recording the chosen path in 14-01-SUMMARY for 14-03 to consume. Gate: `cargo test -p lgbm-compute --test plane_intrinsic_smoke`.
- [ ] **C++ `__device__` fixture-capture harness (D-03)** — owned by **14-02 Tasks 1+2** (`xtask/cpp/primitive_capture.cu` + the `primitive-capture` xtask subcommand). A self-contained `hipcc` shim over the in-repo AMD fork emitting byte-idempotent committed golden fixtures for ShufflePrefixSum (incl/excl, block+global) / ShuffleReduce sum·max·min·dotprod / single+multi-block BitonicArgSort (tie-rich) / PercentileDevice into `crates/oracle-harness/fixtures/primitives/`. These goldens are replayed by 14-06 Task 2. Gate: `cargo run -p xtask -- primitive-capture && git diff --quiet crates/oracle-harness/fixtures/primitives`.

Every ODL-01 numeric parity assertion (14-06 Task 2) depends on the harness; 14-03/14-05 primitive authoring depends on the smoke-test outcome.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm/CUDA f32 ~1e-6 parity on real GPU | ODL-01 | local GPU is a spoofed APU; discrete-CUDA numbers need Kaggle | see memory `kaggle-cli-cuda-bench` |

*Numeric/permutation anchoring is otherwise automated against the cpu f64 fold + committed C++ fixtures.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — every task in 14-01..14-06 carries an `<automated>` command (see Per-Task Verification Map).
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — all 15 tasks have automated verify; continuity is unbroken.
- [x] Wave 0 covers all MISSING references (plane-op smoke test, fixture harness) — 14-01 T1 (smoke test) + 14-02 T1/T2 (fixture harness) close both before any dependent Wave-2+ test runs.
- [x] No watch-mode flags — all commands are single-shot `cargo test` / `cargo build` / `cargo run`; no `--watch`.
- [x] Feedback latency bounded — per-task gates are single crate/test targets (seconds); full-suite wall-time is recorded by 14-06 Task 3 in 14-06-SUMMARY.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-06-29
