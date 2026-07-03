---
phase: 23-perf-validation-default-on-rollout-dod
verified: 2026-07-03T21:00:00Z
status: gaps_found
score: 7/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "The real-CUDA A/B numbers are captured — device_launches/tree (SC-2), the lgb_rs/official wall-clock ratios at 500k×50 AND the wide shape (SC-1), and the real-CUDA end-to-end parity number (D-11 max_abs_on_host)"
    status: partial
    reason: "The Kaggle A/B harness DID run on real discrete NVIDIA CUDA and the FAIL verdict is definitive (>>> A/B VERDICT: FAIL <<< in evidence/kaggle-ab-run.log). But the run's /kaggle/working exceeded Kaggle's output-size cap, so the tiny results.{md,json} were dropped and only the log survived. The exact per-shape wall-clock ratios, device_launches/tree, and the D-11 real-CUDA parity number were NOT recorded this run (results.json: medians/launches/parity = null). This is a documented capture gap, not a functional defect — the harness fix (results.json echoed to stdout between <<<AB_RESULTS_JSON sentinels) makes a re-run self-evidencing. Deferred to a v2 re-run."
    artifacts:
      - path: ".planning/phases/23-perf-validation-default-on-rollout-dod/results.json"
        issue: "medians=null, launches=null, parity=null, numbers_captured=false, capture_gap_reason=kaggle-output-size-cap-dropped-results-files"
    missing:
      - "Re-run ab_harness.py on Kaggle (boomvector) with the sentinel-echo fix to capture the exact device_launches/tree (SC-2), the wall-clock ratios at both shapes (SC-1), and the D-11 real-CUDA parity number into a numeric results.md"
  - truth: "The on-device learner is the DEFAULT CUDA tree-learner path (ODL-21 literal deliverable / milestone DoD)"
    status: partial
    reason: "The default-on flip was INTENTIONALLY WITHHELD by design. ODL-21 is CONTINGENT on the A/B being not-slower; the real-CUDA A/B FAILED (launch-bound, ~7.0h matrix / ~23 min per 100-tree arm), so per the plan's D-09 FAIL branch the behavior-changing flip was correctly not committed. on_device_default() returns literal false (crates/lgbm-compute/src/lib.rs:1378-1380); on-device stays opt-in via LGBM_CUDA_ON_DEVICE=1 with the =0/unset off-switch retained. The audit-before-wire CONTINGENCY MECHANISM worked exactly as designed — but the LITERAL end-state (on-device is the default) is unmet. The perf work to make on-device not-slower is the primary v2 carry-forward."
    artifacts:
      - path: "crates/lgbm-compute/src/lib.rs"
        issue: "on_device_default() { false } — flip to cfg!(feature=\"cuda\") never made (verdict-gated, A/B FAIL)"
    missing:
      - "On-device CUDA perf work to close the per-leaf launch-bound slowdown, then a re-audit A/B PASS, before on_device_default() can flip to cfg!(feature=\"cuda\")"
deferred: []
---

# Phase 23: Perf-Validation + Default-On Rollout (DoD) Verification Report

**Phase Goal:** Measure the on-device win on real CUDA and make the on-device learner the DEFAULT CUDA tree-learner path — contingent on parity AND not-slower — with the host path retained as an off-switch. This is the milestone Definition of Done.
**Verified:** 2026-07-03T21:00:00Z
**Status:** gaps_found (2 partial — both intentional, documented deferrals to v2)
**Re-verification:** No — initial verification (phase ended on the D-09 human-verify FAIL branch; no prior VERIFICATION.md)

## Executive Summary

Phase 23 is a proof-gated rollout phase. Its core value — an **audit-before-wire A/B gate that decides whether to flip the CUDA default to the on-device learner** — was delivered and exercised end-to-end. The gate fired **FAIL** on real discrete NVIDIA CUDA and, exactly as designed (D-09), **withheld** the behavior-changing default-on flip.

This is **NOT a clean pass**. Two of the phase's must-haves are **PARTIAL**:

- **ODL-20** (SC-1/SC-2): the A/B harness ran on real CUDA and the FAIL verdict is definitive, but Kaggle's output-size cap **dropped the exact numbers** (wall-clock ratios, device_launches/tree, real-CUDA parity). A re-run with the committed capture fix is needed to record them.
- **ODL-21** (SC-3): the default-on flip was **correctly withheld** on the failing A/B. The contingency logic worked as designed; the **literal deliverable (on-device is the default) is unmet**, and on-device stays opt-in.

Everything that COULD be verified locally is solid: the tri-state env toggle, the launch-count instrumentation, the A/B harness parse/verdict/parity logic (8/8 unit tests), the no-flip decision + DoD evidence artifact, and the byte-unchanged merge gate (SC-4). These are framed below as **documented, intentional deferrals to v2 (the on-device CUDA perf work)** — not broken functionality.

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | `LGBM_CUDA_ON_DEVICE` is a tri-state resolver (unset⇒default, `"0"`⇒force-off, `"1"`⇒force-on), single source of truth | ✓ VERIFIED | `cuda_on_device_override_from` closed-enum match (lib.rs:1345-1351); `cuda_on_device_enabled` = override.unwrap_or_else(on_device_default) (lib.rs:1330-1332); learner duplicate parse removed. `cargo test -p lgbm-compute --test cuda_on_device` → 3 passed |
| 2 | `on_device_default()` returns `false`; on-device is opt-in only (pre-verdict safe state) | ✓ VERIFIED | `fn on_device_default() -> bool { false }` (lib.rs:1378-1380); SC-4 anchor `assert!(!cuda_on_device_enabled())` (lib.rs:3639) passes; crate dir git-clean (flip never committed) |
| 3 | On-device launch-count instrumentation surfaces a non-zero, sub-baseline `device_launches=` (with `on_device=`) only under `LGBM_PHASE_PROF`, inert otherwise | ✓ VERIFIED | Compute-owned `ON_DEVICE_LAUNCH_CNT` + gated `bump_launch()` (grow_driver.rs); folded into phase_prof COUNTS line; `cargo test -p lgbm-compute --test on_device_launch_count` → 1 passed (per-leaf collapse bound, post-WR-01/WR-02 fix) |
| 4 | A committed, credential-free Kaggle A/B harness with importable parse/median/verdict/parity-envelope logic | ✓ VERIFIED | `ab_harness.py` ast-parse-clean, no credential literal (grep-clean); `python3 test_ab_harness_parse.py` → 8/8 pass, exit 0 (SHORT regex, launches/tree, median, PASS + FAIL-slowdown + FAIL-parity branches, near-tie envelope) |
| 5 | The A/B harness RAN on real discrete NVIDIA CUDA and produced a definitive verdict | ✓ VERIFIED | `>>> A/B VERDICT: FAIL <<<` present in `evidence/kaggle-ab-run.log` (103KB run log); provenance = real discrete NVIDIA CUDA (not the spoofed APU), `maturin -F cuda` + official lightgbm rebuilt USE_CUDA=ON from source; ~7.0h matrix |
| 6 | The real-CUDA A/B **numbers** are captured — device_launches/tree (SC-2), wall-clock ratios at both shapes (SC-1), and the D-11 real-CUDA parity number | ✗ FAILED | Kaggle output-size cap dropped `results.{md,json}`; `results.json` records medians=null, launches=null, parity=null, `numbers_captured=false`. Verdict definitive, numeric breakdown NOT recorded this run. Harness fixed (stdout sentinel echo) for a self-evidencing re-run |
| 7 | The on-device learner is the **DEFAULT** CUDA tree-learner path (ODL-21 literal / milestone DoD) | ✗ FAILED | Default-on flip WITHHELD by design (D-09 FAIL branch): `on_device_default()` stays `false`, on-device opt-in via `LGBM_CUDA_ON_DEVICE=1`. Contingency mechanism worked; literal end-state unmet. v2 perf carry-forward |
| 8 | ROCm + CPU routing stay host-driven / byte-unchanged; CPU f64 merge gate green (SC-4) | ✓ VERIFIED | No crate code changed in plan 04 (flip withheld); crate dirs git-clean; tri-state + counter inert with env unset; targeted compute tests green; per SUMMARY 969-test workspace suite green with env unset |
| 9 | DoD evidence artifact + no-flip decision are recorded | ✓ VERIFIED | `results.md` (verdict FAIL + numbers-not-captured caveat + NO-FLIP decision), `results.json` (`flip_performed=false`, `dod_complete=true`), `evidence/kaggle-ab-run.log`, `evidence/kernel-metadata.json` all present |

**Score:** 7/9 truths verified (0 present, behavior-unverified). The 2 FAILED truths are documented, intentional deferrals to v2 — see Gaps Summary.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-compute/src/lib.rs` | Tri-state resolver + `on_device_default()==false` | ✓ VERIFIED | Resolver at :1330-1363; default stub at :1378-1380 returns `false` (flip withheld) |
| `crates/lgbm-compute/tests/cuda_on_device.rs` | Tri-state mapping unit tests | ✓ VERIFIED | 3 tests pass (closed-enum, `"0"` force-off distinctness, cpu-build default-off) |
| `crates/lgbm-compute/src/kernels/grow_driver.rs` | Launch counter + per-leaf `bump_launch()` | ✓ VERIFIED | Counter + gated bump; per-leaf granularity post WR-01 fix |
| `crates/lgbm-treelearner/src/phase_prof.rs` | Fold on_device into `device_launches=` COUNTS | ✓ VERIFIED | Fold + `on_device=`/`launch_unit=` annotation; SHORT regex preserved |
| `crates/lgbm-compute/tests/on_device_launch_count.rs` | Non-zero sub-baseline launch-count test | ✓ VERIFIED | 1 test pass; exact per-leaf collapse bound post WR-02 fix |
| `.planning/.../ab_harness.py` | Committed credential-free A/B harness | ✓ VERIFIED | ast-clean, credential-free, sentinel capture fix present (:236-238) |
| `.planning/.../test_ab_harness_parse.py` | Local dry-run parse/verdict test | ✓ VERIFIED | 8/8 pass, exit 0 |
| `.planning/.../results.md` + `results.json` | DoD evidence (verdict FAIL) | ⚠️ PRESENT, numbers dropped | Verdict FAIL recorded; exact numbers null (Kaggle output-cap) — honest caveat documented |
| `.planning/.../evidence/kaggle-ab-run.log` | Authoritative surviving run evidence | ✓ VERIFIED | 103KB; contains the FAIL verdict + build/matrix timeline |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `cuda_on_device_enabled()` | `on_device_default()` | `override.unwrap_or_else(on_device_default)` | ✓ WIRED | lib.rs:1331 — single source of truth; learner reads through `on_device_growth_supported()` |
| on-device driver dispatch | phase_prof COUNTS | `on_device_launch_count_take()` | ✓ WIRED | Compute-owned atomic consumed by phase_prof::dump (crate-cycle-safe) |
| `results.md` verdict FAIL | default-on flip | D-09 verdict gate | ✓ WIRED (as no-flip) | FAIL verdict → flip withheld; `on_device_default()` unchanged. Contingency correctly blocked the flip |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Tri-state resolver mapping + cpu default-off | `cargo test -p lgbm-compute --test cuda_on_device` | 3 passed | ✓ PASS |
| On-device launch-count non-zero + per-leaf collapse bound | `cargo test -p lgbm-compute --test on_device_launch_count` | 1 passed | ✓ PASS |
| A/B harness parse/median/verdict/parity | `python3 test_ab_harness_parse.py` | 8/8 pass, exit 0 | ✓ PASS |
| Harness importable anywhere (no heavy deps at module scope) | `python3 -c "import ast; ast.parse(...)"` | parse-ok | ✓ PASS |
| No Kaggle credential literal committed | grep credential patterns in `ab_harness.py` | no match | ✓ PASS |
| Real-CUDA A/B verdict definitive | grep `A/B VERDICT: FAIL` in evidence log | present | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| ODL-20 | 23-01/02/03 | Real-CUDA Kaggle A/B harness measures device_launches/tree + wall-clock ratio at 500k×50 and a wide shape | ⚠️ PARTIAL | Harness built + RAN on real CUDA; FAIL verdict definitive; but exact numbers dropped by Kaggle output-cap (SC-1/SC-2 not recorded this run). Re-run needed |
| ODL-21 | 23-01/04 | On-device becomes DEFAULT CUDA path contingent on parity AND not-slower; `LGBM_CUDA_ON_DEVICE=0` off-switch retained | ⚠️ PARTIAL | Contingency FAILED (A/B FAIL) → default flip correctly WITHHELD; off-switch + opt-in retained. Literal deliverable (on-device is default) unmet |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (none blocking) | — | — | — | Code-review WR-01/02/03 + IN-01/02 all fixed (23-REVIEW-FIX.md, 5/5, commits 617ea76/1527b4e/6bb27cc). No debt markers (TBD/FIXME/XXX) in phase files. `on_device_default(){false}` is intentional pre-verdict state, not a stub defect |

### Human Verification Required

None outstanding. The one human-verify checkpoint (the real-CUDA A/B verdict, D-09) was already exercised — it returned FAIL and is recorded in 23-UAT.md (5/5 UAT tests passed, committed 49c55bc). No behavior-dependent truths remain unexercised.

### Gaps Summary

Both gaps are **documented, intentional deferrals to v2**, not broken functionality:

1. **Real-CUDA A/B numbers not captured (ODL-20 / SC-1 / SC-2).** The harness ran on real discrete NVIDIA CUDA and the FAIL verdict is definitive, but Kaggle's output-size cap dropped `results.{md,json}` (only the log survived). The exact wall-clock ratios, device_launches/tree, and the D-11 real-CUDA parity number are unrecorded this run. The harness was fixed (results.json echoed to stdout between sentinels) so a re-run is self-evidencing. **Action: re-run on Kaggle to record the numbers.**

2. **On-device is not the default (ODL-21 / SC-3).** The default-on flip was correctly WITHHELD because the A/B FAILED the not-slower contingency (launch-bound, ~23 min per 100-tree arm). The audit-before-wire mechanism worked exactly as designed; the literal end-state is unmet. **Action (v2): close the per-leaf launch-bound slowdown, then re-audit before flipping `on_device_default()` to `cfg!(feature="cuda")`.**

**On the DoD framing:** Per D-09, a failing A/B with a committed evidence artifact + a withheld flip + a follow-up note is an explicitly valid DoD-complete outcome — the phase's core value (a proof-gated rollout DECISION) was delivered. This verifier records that the DECISION mechanism is sound and verified, while being honest that the two literal deliverables (recorded numbers, on-device-as-default) are unmet and carried to v2. This matches the v1.1 milestone audit's assessment (ODL-20/ODL-21 = partial).

**Optional override path.** If the team wishes to formally accept ODL-21's by-design withholding as milestone-complete (rather than a gap), add to this file's frontmatter:

```yaml
overrides:
  - must_have: "The on-device learner is the DEFAULT CUDA tree-learner path"
    reason: "Default-on flip is CONTINGENT (D-09) on a not-slower A/B; the real-CUDA A/B FAILED, so withholding the flip is the correct designed outcome. On-device retained as opt-in; perf work deferred to v2."
    accepted_by: "{name}"
    accepted_at: "{ISO timestamp}"
```

Re-running verification with that override would move truth #7 to PASSED (override). The numbers-capture gap (truth #6) should remain a real gap until a re-run records them.

---

_Verified: 2026-07-03T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
