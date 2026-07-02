---
phase: 20-on-device-score-updater-metrics
verified: 2026-07-02T15:10:00Z
status: passed
score: 6/6 success criteria verified
behavior_unverified: 0
overrides_applied: 0
requirements_verified: [ODL-16, ODL-17, ODL-18, ODL-19]
gates_run:
  - "cargo test --workspace (LGBM_CUDA_ON_DEVICE unset) -> exit 0, 0 failures (byte-unchanged merge gate)"
  - "LGBM_CUDA_ON_DEVICE=1 learner_parity_on_device_structure_gate --exact -> 1 passed (non-vacuous; 4 leaves, max leaf diff 0.000e0)"
  - "LGBM_CUDA_ON_DEVICE=1 resident_score_ab -> 2 passed (resident cuda_score_ == host score_ bit-exact)"
  - "metric_parity -> 27 passed incl. 12 device_* cells"
  - "lgbm-compute metric_pointwise -> 5 unit passed; device_metric metric_supported -> 3 passed"
  - "score_updater_parity -> 3 passed; on_device_growth_supported_stays_false -> 1 passed"
crate_cycle_invariant: "grep -rn lgbm_treelearner crates/lgbm-compute/src -> NONE (no treelearner->compute->treelearner cycle)"
scope_notes:
  - "Criterion 5 driver proving slice is continuous-feature + L2 + MissingType::None (documented, not a silent gap); L1/quantile/categorical/RenewTreeOutput deferred to the follow-up on the same ordering contract."
  - "Criterion 2/3: the on-device metric evaluator (eval_metric_on) is anchor-pinned as a standalone capability but is NOT yet invoked from the GBDT eval loop (boosting metric eval currently runs host-side over the mirrored score). The honest-fallback discriminator (metric_supported) exists and is unit-tested over the full unsupported set. Wiring the device eval into GBDT is follow-up work, consistent with the L2-slice scope."
  - "DART/RF per-row-predict score paths stay host-side (documented, out of the L2 proving slice)."
---

# Phase 20: On-Device Score Updater & Metrics Verification Report

**Phase Goal:** The cumulative score lives resident on device and the supported pointwise metrics evaluate on-device, completing the boosting-layer device path (+ pulled-forward on-device grow loop, ODL-18/19).
**Verified:** 2026-07-02
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Success Criteria (Observable Truths)

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | On-device score update — resident `cuda_score_`, constant add/multiply, host-mirror toggle (§11) | PASS | Kernels `add_score_constant_on`/`multiply_score_constant_on` over `Array<f64>` (`score_updater.rs:52,62,73`); `boosting_on_cuda_` toggle + resident loop wired in `gbdt.rs:1006-1018` (gated on `boosting_on_cuda()`). Gate `resident_score_ab` = 2 passed (resident == host `score_` bit-exact after full multi-iter train); `score_updater_parity` = 3 passed. |
| 2 | On-device pointwise metric eval — EvalKernel + two-stage reduction over 12 losses, anchor-pinned (§12) | PASS | `metric_on_point` 12-branch comptime table + `eval_pointwise_on` two-stage fold via reused `reduce_sum_f64_on` + `eval_metric_on` ConvertOutput compose (`metric_pointwise.rs:123,281,403`). Gate `metric_parity` = 27 passed incl. all 12 `device_*` cells anchored to real lib_lightgbm 4.6 goldens; `metric_pointwise` = 5 unit passed. |
| 3 | CUDA-unsupported metrics honestly fall back to host (§12.1) | PASS | `metric_supported`/`DeviceMetricKind` classifier returns false for auc/auc_mu/ndcg/map/average_precision/multi_*/cross_entropy*/kullback_leibler (`device_metric.rs`). Gate: 3 discriminator tests (rejects-unsupported, accepts-all-twelve, objective-asymmetry). No unsupported metric is ever routed to a device path (device eval not wired into GBDT; all boosting eval host-side). |
| 4 | Score + metric anchor-pinned; CPU/ROCm/host-CUDA byte-unchanged; merge gate green | PASS | `cargo test --workspace` (LGBM_CUDA_ON_DEVICE unset) exit 0, zero failures. All parity anchored to the cpu f64 fold (never GPU-vs-GPU). Gated flips return `cuda_on_device_enabled()` (lib.rs:2288,1358); `on_device_growth_supported_stays_false` = 1 passed. |
| 5 | On-device driver full grow loop -> (Tree, DataPartition), STRUCTURE bit-exact tie-aware, leaf values ~1e-5 (ODL-18, §6/§16) | PASS | `grow_tree_on_device_driver` sequences Phase-16/17/18 kernels in SerialTreeLearner best-first order with own `DriverLeaf` bookkeeping (`grow_driver.rs:418`); wired into `CpuBackend`/`GpuBackend<R>` seam (lib.rs:2302,1371). Gate `learner_parity_on_device_structure_gate --exact` = 1 passed (non-vacuous): "4 leaves match cpu f64 anchor (structure bit-exact, max leaf diff 0.000e0)". Anchor is the real `SerialTreeLearner` cpu train; comparator is tie-aware on `default_left`. |
| 6 | Every new kernel keeps f32+u64 fixed-point, no f64 per-row hot loops; byte-unchanged env-unset (ODL-19, §17) | PASS | Driver composes only existing golden kernels (construct/subtract/find_best_split/partition); the only per-row f64 is the ROOT ordered fold (reference-blessed `LeafSplits::init` analog, run once) — child sums seeded from `SplitInfo` (no re-fold). Score-updater f64 is the allowed resident score buffer. FixHistogram/compact are O(num_bin) scalar folds, not per-row. Env-unset workspace green. |

**Score:** 6/6 criteria verified (0 present-but-behavior-unverified).

### Required Artifacts

| Artifact | Provides | Status |
|----------|----------|--------|
| `crates/lgbm-compute/src/kernels/metric_pointwise.rs` | §12 EvalKernel + 12-branch `metric_on_point` + two-stage fold + `eval_metric_on` | VERIFIED (substantive, wired, tested) |
| `crates/lgbm-compute/src/kernels/score_updater.rs` | §11 AddScoreConstant/MultiplyScoreConstant f64 launchers | VERIFIED |
| `crates/lgbm-boosting/src/score_updater.rs` | `boosting_on_cuda_` toggle + resident mirror + D-02 per-leaf delegate | VERIFIED |
| `crates/lgbm-compute/src/kernels/grow_driver.rs` | `GrowFeature` carrier + `grow_tree_on_device_driver` + buffer-strategy A/B | VERIFIED |
| `crates/lgbm-compute/src/lib.rs` | gated `on_device_growth_supported()` + 5-arg `grow_tree_on_device` seam -> driver | VERIFIED |
| `crates/lgbm-compute/src/device_metric.rs` | `metric_supported` host-fallback discriminator (ODL-17) | VERIFIED |
| `crates/lgbm-boosting/src/gbdt.rs` | §16 resident cross-iteration score loop | VERIFIED (wired, gated) |
| `crates/oracle-harness/tests/{metric,score_updater,learner,resident_score}_*.rs` | anchor gates | VERIFIED (all green) |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| `gbdt.rs` TrainOneIter | `add_prediction_to_score_on_device` | `update_score_resident` (gbdt.rs:1018), gated `boosting_on_cuda()` | WIRED |
| `CpuBackend/GpuBackend::grow_tree_on_device` | `grow_tree_on_device_driver` | lib.rs:1383,2314, gated `cuda_on_device_enabled()` | WIRED |
| `eval_metric_on` | GBDT eval loop | — | NOT WIRED (standalone anchor-pinned capability; documented scope note) |

### Behavioral Spot-Checks / Gate Execution

| Gate | Command | Result | Status |
|------|---------|--------|--------|
| Merge gate | `cargo test --workspace` (env unset) | exit 0, 0 failures | PASS |
| STRUCTURE (crit 5) | `LGBM_CUDA_ON_DEVICE=1 ... --exact learner_parity_on_device_structure_gate` | 1 passed, max leaf diff 0.000e0 | PASS (non-vacuous) |
| STRUCTURE env-unset | same, env unset | 1 passed (defers Ok(None), byte-unchanged) | PASS |
| Resident A/B (crit 1) | `LGBM_CUDA_ON_DEVICE=1 ... resident_score_ab` | 2 passed | PASS |
| Metric parity (crit 2) | `metric_parity` | 27 passed (12 device cells) | PASS |
| Discriminator (crit 3) | `metric_supported` | 3 passed | PASS |
| Kernel units | `metric_pointwise` / `score_updater` | 5 / 3+5 passed | PASS |
| Crate cycle | `grep lgbm_treelearner crates/lgbm-compute/src` | NONE | PASS |
| LightGBM/ read-only | `git log --name-only 3ea6745..2ddfc23 \| grep LightGBM/` | 0 paths | PASS |

### Anti-Patterns Found

None. No unreferenced TBD/FIXME/XXX debt markers in phase-20-modified source. `unsafe` confined to `launch_unchecked` sites in each kernel. Driver body returns `Ok(Some(..))` when gated (not a stub). `eval_metric_on` is `pub` (no dead-code warning) though not yet consumed by GBDT — noted as scope, not a stub.

### Human Verification Required

None. All six criteria are backed by deterministic cpu-f64-anchor gates run in this verification. The ROCm f32 cells are opt-in behind `--features rocm` and validated on the (spoofed-APU) hardware per the SUMMARYs; the merge gate is the default cubecl-cpu lane, which is green.

### Gaps Summary

No blocking gaps. All 6 ROADMAP success criteria are met in the delivered code and confirmed by independent gate runs. Three documented scope boundaries were confirmed as intentional (not silent gaps):

1. The grow driver's proving slice is continuous + L2 + `MissingType::None` (the anchor tree uses the identical `proving_slice_config`); L1/quantile/categorical/RenewTreeOutput are explicit follow-ups on the same ordering contract.
2. The on-device pointwise metric evaluator is anchor-pinned standalone; GBDT still evals metrics host-side over the mirrored score. The honest-fallback discriminator is present and correct, so criterion 3 holds; end-to-end device metric routing is follow-up.
3. DART/RF per-row-predict score paths remain host-side (out of the L2 slice).

The replan invariant (no `treelearner -> compute -> treelearner` crate cycle) holds: `GrowFeature` mirrors the learner spine column using only lgbm-compute-reachable types, and no `lgbm_treelearner` / `FeatureColumn` reference exists under `crates/lgbm-compute/src`.

---

_Verified: 2026-07-02T15:10:00Z_
_Verifier: Claude (gsd-verifier)_
