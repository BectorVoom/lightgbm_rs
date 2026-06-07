---
phase: 07-parity-completing-variants
plan: 04
subsystem: metric
tags: [metric, parity, lightgbm, quantile, huber, fair, poisson, gamma, tweedie, cross_entropy, kullback_leibler, multi_error, auc_mu, average_precision]

# Dependency graph
requires:
  - phase: 06-gbdt-spine-core-objectives-metrics
    provides: "Metric::Eval seam (regression/binary/multiclass metric layer), ObjectiveKind::convert ConvertOutput shim"
  - phase: 07-parity-completing-variants (07-02/07-03)
    provides: "alpha/fair_c/tweedie_variance_power config params + exp/log family objectives + boosting capture infra"
provides:
  - "Extended regression metrics: quantile/huber/fair/poisson/mape/gamma/gamma_deviance/tweedie on the Metric::Eval seam"
  - "Xentropy metrics: cross_entropy/cross_entropy_lambda/kullback_leibler (new xentropy.rs)"
  - "Multiclass metrics: multi_error (top-k error), auc_mu (default equal-weight matrix)"
  - "Binary metric: average_precision (faithfully placed in binary family per C++ metric.cpp)"
  - "metric-oracle-capture xtask + metric_oracle_capture.py (real-binary metric golden capture)"
  - "crates/oracle-harness/tests/metric_parity.rs (capture-gated per-metric parity)"
affects: [07-09 (MET-04 ranking metrics ndcg/map), early-stopping metric routing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Metric eval applies ObjectiveKind ConvertOutput inline per family (exp for poisson/gamma/tweedie; sigmoid/log1p for xentropy)"
    - "Metric parity via (scores, labels, captured-value) triplet replay rather than per-round metric-history alignment"
key-files:
  created:
    - crates/lgbm-metric/src/xentropy.rs
    - crates/oracle-harness/tests/metric_parity.rs
    - xtask/py/metric_oracle_capture.py
    - crates/oracle-harness/tests/fixtures/metric/*.txt (14 metrics × {scores,labels,value})
  modified:
    - crates/lgbm-metric/src/regression.rs
    - crates/lgbm-metric/src/multiclass.rs
    - crates/lgbm-metric/src/binary.rs
    - crates/lgbm-metric/src/lib.rs
    - xtask/src/main.rs

key-decisions:
  - "average_precision is a BINARY-task metric (AveragePrecisionMetric, binary_metric.hpp) per the C++ metric.cpp factory — placed in BinaryMetric, not multiclass, despite the plan grouping it loosely under multiclass. Faithful to C++."
  - "poisson/gamma/tweedie regression metrics apply exp ConvertOutput (objective == metric family in capture); quantile/huber/fair/mape use the identity-ConvertOutput regression objective (score the raw score directly)."
  - "Metric enum dropped Eq derive (now carries f64 alpha/fair_c/tweedie params); no consumer relied on Metric: Eq."
  - "Metric parity uses a (scores,labels,captured-value) triplet replay (Rust eval over captured real-binary scores == captured metric value within ORACLE_TOL) — cleanly independent of the DEF-07-02 learner knife-edge."

patterns-established:
  - "RegressionMetricParams binds the C++ config_ fields (alpha/fair_c/tweedie_variance_power) the parametrized metric arms read via Metric::parse_with_params."
  - "AucMu constructs the C++ default equal-weight class matrix (ones off-diagonal, zero diagonal) per Config::GetAucMuWeights."

requirements-completed: [MET-03]

# Metrics
duration: 11min
completed: 2026-06-07
---

# Phase 7 Plan 04: Extended Metrics (MET-03) Summary

**14 extended regression/xentropy/multiclass/binary evaluation metrics ported 1:1 from the C++ `*_metric.hpp` and validated to faithful parity (ORACLE_TOL) against real lib_lightgbm 4.6, with correct bigger-better signs and ConvertOutput routing.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-06-07T07:05:30Z
- **Completed:** 2026-06-07T07:16:01Z
- **Tasks:** 3 completed (Task 3 human-verify checkpoint pre-satisfied: wheel present)
- **Files modified:** 5 source/test/xtask + 1 new python script + 42 golden fixtures

## Accomplishments

- **Task 1 (7bf207e):** Extended metric math behind the enum-dispatch factory.
  - regression.rs: `quantile`/`huber`/`fair`/`poisson`/`mape`/`gamma`/`gamma_deviance`/`tweedie` LossOnPoint arms in the ordered f64 fold; `RegressionMetricParams` + `parse_with_params`; exp ConvertOutput for poisson/gamma/tweedie; gamma_deviance `sum_loss * 2` average; `safe_log` (kZeroThreshold floor) for gamma.
  - xentropy.rs (new): `cross_entropy`/`cross_entropy_lambda`/`kullback_leibler` with sigmoid / log1p(exp) ConvertOutput, the 1e-12 `XentLoss` clip, and the KL `YentLoss` label-entropy offset.
  - multiclass.rs: `MultiError` (top-k argmax error) + `AucMu` (full Eval port with default equal-weight matrix).
  - binary.rs: `AveragePrecision` (PR-AUC, sort-based), factor +1.
  - 41 unit tests (parse aliases, factor signs, hand-computed values, ConvertOutput routing) GREEN.
- **Task 2 (dfad4ae):** Capture emitter + capture-gated parity.
  - `metric-oracle-capture` xtask subcommand + `metric_oracle_capture.py`: trains a tiny model per metric family with a compatible objective on real lib_lightgbm 4.6, dumps per-metric (raw scores, labels, real-binary value); version-pinned 4.6.0.
  - `crates/oracle-harness/tests/metric_parity.rs`: 14 capture-gated cells + a builder-routing test; skip-pass without goldens.
- **Task 3 (e2dd2cf):** Real-binary goldens captured + committed (byte-idempotent; cells GREEN).

## Parity outcome

All 14 `metric_parity` cells GREEN within ORACLE_TOL vs real lib_lightgbm 4.6, including the transcendental-bearing fair/poisson/gamma/gamma_deviance/tweedie metrics. Captured values (round 5): quantile 1.5254, huber 2.3890, fair 1.8140, mape 0.5737, poisson -6.0095, gamma 2.7246, gamma_deviance 1.4155, tweedie 9.9358, cross_entropy 0.5873, cross_entropy_lambda 0.5931, kullback_leibler 0.1203, average_precision 1.0, multi_error 0.0, auc_mu 1.0. Capture is byte-idempotent (sha256 of all fixtures identical across two runs).

## Deviations from Plan

### Auto-fixed / faithful-placement adjustments

**1. [Rule 2 - Faithful correctness] average_precision placed in BinaryMetric, not multiclass.**
- **Found during:** Task 1 (reading C++ refs).
- **Issue:** The plan grouped `average_precision` under "multiclass," but `AveragePrecisionMetric` lives in `binary_metric.hpp` and the C++ `metric.cpp` factory creates it as a binary-task metric.
- **Fix:** Added it as a `BinaryMetric::AveragePrecision` arm (factor +1). Faithful to C++; the parity test exercises it identically.
- **Files modified:** crates/lgbm-metric/src/binary.rs
- **Commit:** 7bf207e

**2. [Rule 1 - Type correctness] Removed `Eq` from `Metric` derive.**
- **Issue:** The new `Quantile`/`Huber`/`Fair`/`Tweedie` arms carry f64 params, which are not `Eq`.
- **Fix:** Dropped `Eq` (kept `PartialEq`); confirmed no consumer (booster.rs `EvalMetric`, parity tests) relied on `Metric: Eq` — full `cargo build -p lgbm` + workspace tests pass.
- **Commit:** 7bf207e

**3. set.rs not modified.** The plan listed `crates/lgbm-core/src/config/set.rs` in files_modified, but re-grep confirmed `alpha`/`fair_c`/`tweedie_variance_power`/`multi_error_top_k` are already parsed + validated (added in 07-02/07-03) and visible to the metric layer. No edit needed (the action said "confirm they are visible" — they are).

## Authentication gates

None. The Task 3 `checkpoint:human-verify` (gate="blocking-human") was pre-satisfied: the `lightgbm==4.6.0` wheel is installed at `/tmp/lgbm-capture-venv`, so the capture ran non-interactively, was version-asserted, byte-idempotent, and flipped all cells to GREEN.

## Known Stubs

None. `MultiError::name` returns the canonical `"multi_error"` for the in-scope default `top_k==1` (the C++ `@k` suffix is dynamic and only exercised at non-default top-k, which is out of scope here). Documented inline, not a data stub.

## Verification

- `cargo test -p lgbm-metric` — 41 passed.
- `cargo test -p oracle-harness --test metric_parity` — 15 passed (14 metric cells GREEN + builder routing).
- `cargo test --workspace` — GREEN; boosting_parity 58 passed / 13 ignored (DEF-07-02) / 0 failed (spine NOT regressed).
- `cargo build --workspace --tests` — exit 0.
- `cargo clippy` — clean on all edited files (regression.rs / xentropy.rs / multiclass.rs / binary.rs / lib.rs / xtask main.rs / metric_parity.rs).
- Capture byte-idempotent (sha256 identical across two runs); `LightGBM/` never git-added.

## Self-Check: PASSED

All created files exist on disk (xentropy.rs, metric_parity.rs, metric_oracle_capture.py, fixtures/metric/*.txt, 07-04-SUMMARY.md) and all three task commits (7bf207e, dfad4ae, e2dd2cf) are present in git history.
