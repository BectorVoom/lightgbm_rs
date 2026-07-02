---
phase: 20-on-device-score-updater-metrics
plan: 02
subsystem: on-device-metric-evaluator
tags: [metric, ODL-17, D-03, D-04, D-08, D-10, D-11, EvalKernel, ConvertOutput]
status: complete
requires:
  - "kernels/metric_pointwise.rs empty stub registered by Plan 20-00"
  - "primitives.rs reduce_sum_f64_on ordered f64 fold (D-10, do not rebuild)"
  - "objective_regression.rs convert_output_on + CONVERT_EXP launcher"
  - "objective_binary.rs sigmoid_convert_output_on launcher"
  - "device_metric.rs DeviceMetricKind + metric_supported discriminator (Plan 20-00)"
  - "lib_lightgbm 4.6 metric goldens captured by Plan 20-00 (rmse/l2/l1/binary_logloss)"
provides:
  - "metric_on_point<F> 12-branch comptime §12.1 table + EvalKernel<metric, use_weight>"
  - "eval_pointwise_on two-stage f64 fold + per-metric AverageLoss finalizers"
  - "eval_metric_on: ConvertOutput compose (D-04) into pre-allocated score_convert_buffer BEFORE EvalKernel"
  - "MetricEvalParams (alpha/fair_c/tweedie_variance_power/sigmoid) = C++ Config defaults"
  - "12 on-device metric parity cells anchored to lib_lightgbm goldens on the cpu f64 anchor"
affects:
  - "Plan 20-03 (device tree-learner driver) — the device metric path completes the boosting-layer eval; unsupported metrics route to host via the Plan-00 discriminator"
tech_stack:
  added: []
  patterns:
    - "comptime-generic EvalKernel<#[comptime] metric, #[comptime] use_weight> — one thread/row, ShuffleReduceSum per-block partials folded by reduce_sum_f64_on (D-10)"
    - "ConvertOutput composed into Eval flow (D-04): inverse-link into pre-allocated buffer BEFORE the kernel, keyed off the ORIGINAL metric name, never DeviceObjectiveKind"
    - "point/param f64 confined to the reference-blessed per-row loss (D-08), no f64 in a grow/build hot loop"
key_files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/metric_pointwise.rs
    - crates/oracle-harness/tests/metric_parity.rs
decisions:
  - "The 12-way comptime match compiled on cubecl 0.10 — the A2 fallback (split regression/binary EvalKernels) was NOT needed; a single EvalKernel<metric, use_weight> handles all 12 losses."
  - "The convert MODE is routed off DeviceMetricKind (the ORIGINAL metric name), NEVER DeviceObjectiveKind — honoring the device_objective.rs:33-39 warning that the objective kind is a support classifier, not a ConvertOutput key. A mis-route would be caught by the goldens."
  - "gamma_deviance's epsilon is a host f64 literal (1e-9) threaded through the kernel param arg for bit-exactness; the non-param arms ignore MetricEvalParams entirely."
  - "score_convert_buffer is materialized once per eval_metric_on call (D-11): pass-through metrics reuse raw_scores.to_vec(); only the exp/sigmoid metrics run a convert launcher."
metrics:
  tasks_completed: 2
  files_created: 0
  files_modified: 2
  duration_minutes: 25
  completed_date: 2026-07-02
---

# Phase 20 Plan 02: On-Device Pointwise Metric Evaluator (§12/§12.1, ODL-17) Summary

Implemented the on-device pointwise metric evaluator: a single comptime-generic
`EvalKernel<metric, use_weight>` whose per-row `metric_on_point(label, score, param, #[comptime] metric)`
transcribes the 12-row §12.1 table (8 regression arms + gamma/gamma_deviance/tweedie +
binary_logloss), folded to `sum_loss` / `sum_weight` by the reused ordered f64 fold
(`reduce_sum_f64_on`, D-10) with the correct `AverageLoss` finalizers (RMSE `sqrt`,
Gamma-deviance `×2`, else Σloss/Σweight). ConvertOutput is composed into the Eval flow (D-04):
the inverse-link runs into a pre-allocated `score_convert_buffer` BEFORE the kernel, keyed off
the ORIGINAL metric name. Every supported metric is anchored to its real `lib_lightgbm` 4.6
golden (D-03) on the cpu f64 anchor.

## What Was Built

### Task 1 — EvalKernel comptime table + two-stage f64 fold (commit 3ea6745)
`crates/lgbm-compute/src/kernels/metric_pointwise.rs`: a `METRIC_*` constant block (one per
§12.1 row, mirroring the `TAG_*` convention) + a `metric_tag(DeviceMetricKind)` map, and a
`#[cube] fn metric_on_point<F: Float>(label, score, param, #[comptime] metric: u32) -> F` whose
12 branches transcribe `lgbm-metric`'s `loss_on_point` verbatim (the 8 regression arms +
gamma/gamma_deviance/tweedie + the binary_logloss arm with its `kEpsilon = 1e-15f` guards).
Wrapped in a comptime-generic `EvalKernel<metric, use_weight>` (one thread/row) emitting
`(loss, weight)` per-block partials, and a host launcher `eval_pointwise_on` folding the
partials to scalar `sum_loss` / `sum_weight` through `reduce_sum_f64_on` (D-10) then applying
the correct finalizer per metric (RMSE `sqrt`, Gamma-deviance `×2`, else mean). `unsafe`
confined to the launch site; bounds-guarded `i < num_data`. Five cpu-anchor unit tests: all 12
metrics bit-exact vs the host fold, finalizer parity (RMSE sqrt / gamma-deviance ×2),
uniform-weight equivalence, typed length-mismatch error, empty-is-zero.

### Task 2 — ConvertOutput compose (D-04) + on-device parity cells (D-03) (commit 73383c5)
`crates/lgbm-compute/src/kernels/metric_pointwise.rs` extended with:
- `MetricEvalParams { alpha, fair_c, tweedie_variance_power, sigmoid }` = the C++ `Config`
  defaults (0.9 / 1.0 / 1.5 / 1.0) — the per-metric scalars the parametrized §12.1 arms read.
- `eval_metric_on<R>(client, kind, raw_scores, labels, weights, params)`: composes ConvertOutput
  into the Eval flow (D-04) into a `score_convert_buffer` materialized once (D-11) BEFORE the
  kernel, keyed off the ORIGINAL metric name (`DeviceMetricKind`), NEVER `DeviceObjectiveKind`:
  poisson/gamma/gamma_deviance/tweedie → `convert_output_on` `CONVERT_EXP`; binary_logloss →
  `objective_binary::sigmoid_convert_output_on`; l2/rmse/l1/quantile/huber/fair/mape →
  pass-through. Then `eval_pointwise_on` folds and finalizes.

`crates/oracle-harness/tests/metric_parity.rs`: an `assert_device_metric(name, kind, params)`
helper that `load_triplet`s the golden, runs the composed device Eval on the cpu f64 anchor, and
asserts against the captured `lib_lightgbm` value within `ORACLE_TOL` (SKIP-gated on absent
goldens). 12 device cells added (l2/rmse/l1/quantile/huber/fair/mape pass-through +
poisson/gamma/gamma_deviance/tweedie exp-convert + binary_logloss sigmoid-convert). No device
cell for the unsupported metrics (cross_entropy/multi_error/auc_mu/average_precision/
kullback_leibler stay host-fallback replay per D-05).

## Deviations from Plan

### Implementation choices (within plan scope)

- **The 12-way comptime match compiled — A2 fallback not needed.** RESEARCH Assumption A2
  allowed splitting into regression/binary EvalKernels if the 12-way `#[comptime]` match hit a
  cubecl 0.10 limit. It compiled cleanly, so a single `EvalKernel<metric, use_weight>` handles
  all 12 losses (parity-neutral either way).
- **gamma_deviance epsilon threaded as a param.** Its `1e-9` guard is a host f64 literal routed
  through the kernel's `param` arg for bit-exactness, rather than a `#[comptime]` constant — the
  non-param arms ignore `MetricEvalParams` entirely.

### Resume note
Plan 20-02 was executed across two sessions: Task 1 committed as 3ea6745, Task 2 completed in
the working tree but left uncommitted with no SUMMARY. Closed out manually — Task 2's two files
committed as 73383c5 (only the 20-02 files staged; unrelated working-tree churn left untouched),
this SUMMARY written, both gates re-verified green before the commit.

## Verification

- `cargo test -p lgbm-compute metric_pointwise` — 5 passed (all 12 metrics bit-exact vs host
  fold, finalizer parity, uniform-weight equivalence, typed length error, empty-is-zero).
- `cargo test -p oracle-harness --test metric_parity` — 27 passed, including all 12 device cells
  (device_l2/rmse/l1/quantile/huber/fair/mape/poisson/gamma/gamma_deviance/tweedie/binary_logloss).
- `grep -n 'reduce_sum_f64_on' crates/lgbm-compute/src/kernels/metric_pointwise.rs` — matches
  (two-stage fold reuses the D-10 reducer, no new reducer written).
- `unsafe` only at the `launch_unchecked` sites; point/param f64 confined to the reference-blessed
  per-row loss (D-08), no f64 in a grow/build hot loop.

## Threat Mitigations Applied

- **T-20-02-01** (OOB device read/write in EvalKernel / convert launchers): bounds-guarded
  `i < num_data` inside each `#[cube]`; buffers sized exactly with `create_from_slice`; `unsafe`
  confined to the launch site.
- **T-20-02-02** (reduction partial buffer sized wrong): per-block partials sized to the exact
  block count; folded via the single-owner `reduce_sum_f64_on`.
- **T-20-02-03** (wrong ConvertOutput mode leaking an un-inverse-linked score): the convert mode
  is routed off the ORIGINAL metric name (`DeviceMetricKind`), never `DeviceObjectiveKind`; the
  goldens catch a mis-routed transform (poisson/gamma/tweedie assert with exp applied first,
  binary_logloss with sigmoid).
- **T-20-02-SC** (installs): no package installs this plan.

## Self-Check: PASSED

- Both source files (kernels/metric_pointwise.rs, oracle-harness/tests/metric_parity.rs) FOUND on disk.
- Commits 3ea6745 (Task 1) and 73383c5 (Task 2) FOUND in git history.
- `metric_pointwise.rs` contains `fn metric_on_point` and `reduce_sum_f64_on`; the 12 `device_*_parity` cells present in `metric_parity.rs`.
