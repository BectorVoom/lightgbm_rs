---
phase: 20-on-device-score-updater-metrics
plan: 00
subsystem: on-device-metrics-scaffolding
tags: [metric-discriminator, oracle-goldens, kernel-stubs, ODL-17, D-05]
status: complete
requires:
  - "device_objective.rs discriminator pattern (Phase 19)"
  - "metric_oracle_capture.py + xtask metric-oracle-capture (Phase 7)"
  - "pinned lightgbm 4.6 .venv at repo root"
provides:
  - "device_metric::metric_supported + DeviceMetricKind (the ODL-17 host-fallback classifier)"
  - "4 real lib_lightgbm 4.6 metric goldens: rmse/l2/l1/binary_logloss triplets"
  - "registered empty kernel stubs kernels/score_updater.rs (Plan 01) + kernels/metric_pointwise.rs (Plan 02)"
affects:
  - "Plan 20-01 (score updater kernel) — fills kernels/score_updater.rs"
  - "Plan 20-02 (metric evaluator kernel) — fills kernels/metric_pointwise.rs, consumes metric_supported"
tech_stack:
  added: []
  patterns:
    - "mirror device_objective_supported: enum + from_name -> Option<Kind> + supported = is_some()"
    - "one-file-per-plan Wave-0 stub registration to avoid same-wave kernels/mod.rs conflicts (D-08)"
key_files:
  created:
    - crates/lgbm-compute/src/device_metric.rs
    - crates/lgbm-compute/src/kernels/score_updater.rs
    - crates/lgbm-compute/src/kernels/metric_pointwise.rs
    - crates/oracle-harness/tests/fixtures/metric/rmse_{labels,scores,value}.txt
    - crates/oracle-harness/tests/fixtures/metric/l2_{labels,scores,value}.txt
    - crates/oracle-harness/tests/fixtures/metric/l1_{labels,scores,value}.txt
    - crates/oracle-harness/tests/fixtures/metric/binary_logloss_{labels,scores,value}.txt
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-compute/src/kernels/mod.rs
    - xtask/py/metric_oracle_capture.py
    - xtask/src/main.rs
decisions:
  - "Captured the 4 real lib_lightgbm 4.6 goldens (the venv was available) — NOT the cpu-fold fallback; honors D-03 fidelity intent (A1 default branch)."
  - "binary_logloss captured with objective binary (not regression) so its raw score is on the pre-sigmoid scale the binary metric inverse-link consumes."
  - "Renamed the device_metric tests to include the substring metric_supported so the plan's exact verify command actually exercises the assertions (was matching 0 tests)."
metrics:
  tasks_completed: 2
  files_created: 15
  files_modified: 4
  duration_minutes: 18
  completed_date: 2026-07-02
---

# Phase 20 Plan 00: On-Device Score-Updater & Metrics Wave-0 Scaffolding Summary

Front-loaded the Wave-0 scaffolding for the two Phase-20 kernel plans: captured the 4 missing device-supported metric goldens (rmse/l2/l1/binary_logloss) as real `lib_lightgbm` 4.6 triplets, registered two empty kernel stub modules so Plans 01/02 fill them without a same-wave `kernels/mod.rs` conflict, and implemented the `metric_supported` discriminator (D-05) that gates the host fallback for the non-pointwise metric set.

## What Was Built

### Task 1 — 4 missing metric goldens (commit da40f29)
Extended `xtask/py/metric_oracle_capture.py` with `train_and_capture` calls for the 4 base pointwise losses that had never been captured (Pitfall 1): `rmse`, `l2`, `l1` via objective `regression`, and `binary_logloss` via objective `binary`. Ran `cargo run -p xtask -- metric-oracle-capture` against the repo-root pinned `lightgbm 4.6.0` `.venv`, writing the 12 golden files (`{name}_{labels,scores,value}.txt`) into `crates/oracle-harness/tests/fixtures/metric/` in the same one-value-per-line f64/f32-bit shape as the existing `poisson_*.txt`. Added the 4 metrics to the xtask capture-verification loop. Captured values: rmse=3.6401, l2=13.2503, l1=3.0509, binary_logloss=0.40064. Nothing added under `LightGBM/`.

**Goldens-present path taken (A1 default).** The pinned venv was available, so the real-golden capture ran — the `CPU-FOLD-FALLBACK: rmse/l2/l1/binary_logloss` marker is intentionally NOT present (fallback was not needed).

### Task 2 — discriminator + kernel stub registration (commit af70c3b)
- `crates/lgbm-compute/src/device_metric.rs`: `pub fn metric_supported(name: &str) -> bool` + a `DeviceMetricKind` classifier mirroring `device_objective.rs` exactly (enum + `from_name -> Option<Kind>` + `supported = is_some()`). Returns `true` for exactly the 12 pointwise losses (rmse, l2/mse/regression, l1/mae, quantile, huber, fair, poisson, mape, gamma, gamma_deviance, tweedie, binary_logloss/binary) and `false` for auc, auc_mu, ndcg, map, average_precision, multi_error, multi_logloss, cross_entropy/xentropy, cross_entropy_lambda/xentlambda, kullback_leibler, and any unknown name. Keyed on the METRIC name list, independent of `device_objective_supported` (D-05).
- Two empty compiling stubs `kernels/score_updater.rs` (Plan 01) and `kernels/metric_pointwise.rs` (Plan 02), registered in `kernels/mod.rs`; `pub mod device_metric;` added to `lib.rs`.
- Tests mirror `device_objective.rs`'s structure: one rejecting the full unsupported set, one accepting all 12, and an explicit asymmetry test proving `metric_supported("gamma") && !device_objective_supported("gamma")` for mape/gamma/gamma_deviance/tweedie.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Verify command matched 0 tests**
- **Found during:** Task 2 verification
- **Issue:** The plan's exact verify command `cargo test -p lgbm-compute metric_supported` matched 0 tests because no test path contained the substring `metric_supported` (the mirrored `device_objective.rs` test names are `supported_*`/`unsupported_*`). The command exited 0 vacuously, under-testing the acceptance criteria.
- **Fix:** Renamed the three tests to `metric_supported_rejects_unsupported`, `metric_supported_accepts_all_twelve`, `metric_supported_objective_asymmetry_holds` so the plan's exact command now runs all 3 assertions (including the D-05 asymmetry).
- **Files modified:** crates/lgbm-compute/src/device_metric.rs
- **Commit:** af70c3b

**2. [Rule 2 - Missing coverage] Added 4 metrics to the xtask verify loop**
- **Issue:** The xtask `metric_oracle_capture` fn post-capture existence check listed only the original 14 metrics; the 4 new goldens would be written but not verified by the capture command.
- **Fix:** Added rmse/l2/l1/binary_logloss to the xtask verification loop so a missing capture aborts.
- **Files modified:** xtask/src/main.rs
- **Commit:** da40f29

## Verification

- `cargo test -p lgbm-compute metric_supported` — 3 passed (rejects-unsupported, accepts-all-twelve, objective-asymmetry).
- `cargo build --workspace` — green with `LGBM_CUDA_ON_DEVICE` unset (byte-unchanged host path; the new modules are ungated compiling stubs + a pure classifier).
- 4 golden triplets present on disk (real `lib_lightgbm` 4.6 capture; no cpu-fold fallback needed).
- `grep 'pub mod score_updater'` / `'pub mod metric_pointwise'` (kernels/mod.rs) and `'pub mod device_metric'` (lib.rs) each match.
- No clippy findings in the 3 new files (pre-existing warnings in other crates are out of scope).
- `git log` confirms nothing added under `LightGBM/`.

## Threat Mitigations Applied

- **T-20-00-01** (capture writing outside fixtures dir): capture wrote only under `crates/oracle-harness/tests/fixtures/metric/`; verified `git log` adds nothing under `LightGBM/`.
- **T-20-00-02** (discriminator misclassification): unit tests assert the full unsupported set returns `false`; the discriminator is a pure classifier with no device dispatch.
- **T-20-00-SC** (installs): no package installs; capture ran against the pre-provisioned pinned lightgbm 4.6 venv.

## Self-Check: PASSED

- All 3 created source files + the 4 `_value.txt` goldens FOUND on disk.
- Commits da40f29 and af70c3b FOUND in git history.
- No `LightGBM/` paths in either commit.
