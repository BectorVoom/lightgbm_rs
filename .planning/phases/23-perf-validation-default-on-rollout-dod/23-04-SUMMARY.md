---
phase: 23-perf-validation-default-on-rollout-dod
plan: 04
subsystem: infra
tags: [cuda, on-device, rollout, dod, ab-verdict, audit-before-wire, default-off, no-flip]

# Dependency graph
requires:
  - phase: 23-perf-validation-default-on-rollout-dod (plan 01)
    provides: on_device_default()==false stub + LGBM_CUDA_ON_DEVICE tri-state resolver (the off-switch/opt-in that stays authoritative on FAIL)
  - phase: 23-perf-validation-default-on-rollout-dod (plan 03)
    provides: the committed Kaggle A/B harness whose real-CUDA run produced the FAIL verdict this plan gates on
provides:
  - D-07 evidence artifact (results.md + results.json) recording the real-CUDA A/B verdict FAIL + the numbers-not-captured caveat
  - The audit-before-wire DECISION: NO default-on flip (behavior-changing flip withheld on failing proof, D-09)
  - Harness capture fix so a future re-run is self-evidencing (results.json echoed to the Kaggle log)
affects: [ODL-21, SC-4]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Audit-before-wire gate exercised to its FAIL terminus: a committed evidence artifact + a WITHHELD behavior-changing commit (the crate flip is intentionally never made) is itself a valid DoD-complete phase outcome (D-09)"
    - "Log-borne durable capture: when a CI/notebook host may drop output files under a size cap, echo the machine-readable result to stdout inside unique sentinels so the numbers survive in the log"

key-files:
  created:
    - .planning/phases/23-perf-validation-default-on-rollout-dod/results.md
    - .planning/phases/23-perf-validation-default-on-rollout-dod/results.json
    - .planning/phases/23-perf-validation-default-on-rollout-dod/evidence/kaggle-ab-run.log
    - .planning/phases/23-perf-validation-default-on-rollout-dod/evidence/kernel-metadata.json
    - .planning/phases/23-perf-validation-default-on-rollout-dod/23-04-SUMMARY.md
  modified:
    - .planning/phases/23-perf-validation-default-on-rollout-dod/ab_harness.py

key-decisions:
  - "A/B verdict FAIL on real discrete NVIDIA CUDA → NO FLIP: on_device_default() stays false; CUDA on-device learner stays opt-in via LGBM_CUDA_ON_DEVICE=1 (D-09)"
  - "P-3 mono-feature-vs-dual-feature mechanism choice is N/A — no flip was performed, so on_device_default() is unchanged from its 23-01 false stub"
  - "Kaggle dropped results.{md,json} under its output-size cap; exact per-shape ratios / parity numbers are UNRECOVERABLE this run — the verdict (FAIL) is definitive, the breakdown is not captured; surviving evidence is evidence/kaggle-ab-run.log"

requirements-completed: []

# Metrics
duration: ~15min
completed: 2026-07-03
status: complete
---

# Phase 23 Plan 04: Verdict-Gated Default-On Flip — A/B FAIL, NO Flip (DoD-complete) Summary

**The real-discrete-CUDA Kaggle A/B run (`yensen2/lgb-rs-phase23-ab`) returned `>>> A/B VERDICT: FAIL <<<`: the on-device path is NOT within the D-04 `<=1.05x` not-slower bar vs host-CUDA (the ~7.0h matrix / ~23 min per 100-tree arm signals a severe per-leaf launch-bound slowdown). Per the plan's blocking human-verify checkpoint FAIL branch (D-09), the behavior-changing ODL-21 default-on flip is intentionally NOT performed — `on_device_default()` stays `false`, the CUDA on-device learner stays opt-in via `LGBM_CUDA_ON_DEVICE=1`, and the phase is closed DoD-complete with a committed evidence artifact + a follow-up deferral note. The audit-before-wire gate did its job: the launch-bound risk was caught before it shipped as a default.**

## Performance
- **Duration:** ~15 min
- **Completed:** 2026-07-03
- **Tasks:** 3 (evidence artifact, harness capture fix, this summary) — plus tracking updates
- **Files created:** 5 (results.md, results.json, evidence log + kernel metadata, this summary); 1 modified (ab_harness.py)
- **Crate files touched:** 0 — `crates/lgbm-compute/src/lib.rs` and `tests/cuda_on_device.rs` are deliberately UNCHANGED (the flip is withheld on failing proof)

## Decision: NO FLIP (A/B FAIL)
- **Verdict:** FAIL — `>>> A/B VERDICT: FAIL <<<` in `evidence/kaggle-ab-run.log` (line 1069).
- **`on_device_default()`** remains `false` (its pre-verdict value from plan 23-01). CUDA builds do NOT default to the on-device learner.
- **Off-switch / opt-in:** the ODL-21 tri-state resolver + host-CUDA fallback delivered in 23-01/23-02 remain authoritative — set `LGBM_CUDA_ON_DEVICE=1` to opt in to the on-device path; unset/`0` keeps host-CUDA. Default-on is DEFERRED.
- **P-3 mechanism choice:** **N/A.** The mono-feature `cfg!(feature="cuda")` vs per-runtime sealed-const decision only arises when the flip is made. No flip was performed, so the question does not apply this plan.

## Evidence (D-07)
- **`results.md`** — human-readable evidence doc: verdict FAIL; provenance (kernel `yensen2/lgb-rs-phase23-ab`, real discrete NVIDIA CUDA — NOT the local spoofed 8-CU APU; `maturin -F cuda` from source; official lightgbm rebuilt `--no-binary` USE_CUDA=ON after the pip binary lacked CUDA); the build/matrix timeline (maturin done ~688s, official rebuilt ~1577s, matrix ~1577→26874s = **~7.0h**); the severe-slowdown interpretation (D-04 not-slower bar failed); and the explicit **numbers-not-captured caveat**.
- **`results.json`** — machine-readable FAIL stub (`numbers_captured=false`, `artifact_capture="partial-log-only"`, evidence pointer, `flip_performed=false`).
- **`evidence/kaggle-ab-run.log`** — the full ~103KB JSON-stream run log incl. the FAIL verdict + timeline (the authoritative surviving evidence).
- **`evidence/kernel-metadata.json`** — the kernel spec used.
- **Kernel terminal status ERROR is BY DESIGN:** the harness prints the verdict then `sys.exit(1)` on FAIL; Kaggle maps a non-zero exit to ERROR. Not a build/crash failure.

## Numbers-not-captured caveat
Kaggle did NOT commit `results.{md,json}` from this run: the large `/kaggle/working` (cloned repo + Rust `target/` + many 500k-row `.npy` pred files) exceeded Kaggle's output-size cap, so the tiny results files were dropped — only the log + `_ab_worker.py` survived. Therefore the **exact per-shape wall-clock ratios, the real-CUDA parity number (D-11), and `device_launches/tree` (SC-2) are UNRECOVERABLE from this run.** The verdict (FAIL) is definitive; the numeric breakdown behind it is not recorded. This capture gap is fixed for future re-runs (see below).

## Harness capture fix
`ab_harness.py` `_emit_results` now also **echoes the full `results.json` to stdout** bracketed by `<<<AB_RESULTS_JSON` / `AB_RESULTS_JSON>>>` sentinels after writing the files, so a re-run's numbers survive in the Kaggle **log** even if Kaggle again drops the output files under its size cap. Pure-function signatures are unchanged; `test_ab_harness_parse.py` stays 8/8 green and the module still `ast.parse`-cleanly. (No attempt was made to change Kaggle's output-size behavior — the stdout echo is the durable fix.)

## SC-4 (cpu/rocm byte-unchanged)
**Trivially upheld** — no crate code changed this plan. Since `on_device_default()` was not flipped, the cpu default-off merge gate and the rocm build are byte-identical to their pre-plan state. There is nothing to re-gate.

## Follow-up note (deferral)
On-device-as-default is **DEFERRED** pending on-device CUDA perf work: the per-leaf launch-bound slowdown (the ~23 min/arm signal) must be closed before a re-audit. A future re-run should use the **fixed harness** (numbers now echo to the log, so the exact ratios/parity/launches survive the output cap) to produce a full numeric `results.md` for a re-gated D-09 decision. Until then, the on-device path ships as opt-in only.

## DoD status (D-09)
Phase 23 is **DoD-complete** on the FAIL branch: the audit-before-wire gate was exercised end-to-end, the committed evidence artifact landed, and the behavior-changing flip was correctly WITHHELD on a failing proof. A failing A/B with documented numbers + a follow-up note is an explicitly valid DoD outcome (D-09) — the phase's core value (a proof-gated rollout decision) is delivered.

## Deviations from Plan
The plan's Tasks 1 & 2 (the `on_device_default()` flip + the cuda-arm test) were **intentionally NOT executed** — this is the plan's own FAIL-branch instruction (D-09, `must_haves` truth: "If the A/B FAILS, this plan is NOT committed; the default stays OFF"), not a deviation from intent. In their place, per the checkpoint FAIL branch and the orchestrator's continuation directive, this plan committed the D-07 evidence artifact, fixed the harness capture bug, and closed the phase DoD-complete.

## Self-Check: PASSED
All artifacts present (results.md, results.json, evidence/kaggle-ab-run.log, evidence/kernel-metadata.json, 23-04-SUMMARY.md, ab_harness.py); both task commits (2487512, 63ab70f) in git history; `crates/lgbm-compute/` is CLEAN (flip withheld — no crate edits); `test_ab_harness_parse.py` 8/8 green; `ab_harness.py` `ast.parse`-clean.

---
*Phase: 23-perf-validation-default-on-rollout-dod*
*Completed: 2026-07-03*
