# Phase 19: On-Device Objectives - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-01
**Phase:** 19-on-device-objectives
**Areas discussed:** Anchor & fixture strategy, Boosting-layer integration depth, Ranking objective scope, Renew/Convert target surface

---

## Anchor & fixture strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Host f64 anchor + real C++ captures | cpu f64 lgbm-objective is the deterministic anchor; ALSO capture real compiled-lib_lightgbm goldens for representative objectives (L2, binary, softmax, lambdarank) as a fidelity cross-check | ✓ |
| Host f64 anchor only | Reuse existing lgbm-objective f64 as sole anchor; no new C++ captures (transcription-agrees-with-transcription) | |
| Fresh C++ goldens per objective | Capture compiled-lib_lightgbm grad/hess for all 11 objectives (heaviest) | |

**User's choice:** Host f64 anchor + real C++ captures (representative per family)
**Notes:** Chosen to answer the `on-device-kernel-goldens-are-retranscriptions` memory caveat cheaply — objectives are pure elementwise math so real captures are low-cost. Four representatives (L2, binary, softmax, lambdarank) are the floor, not the ceiling.

---

## Boosting-layer integration depth

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone kernels, wired in Phase 21 | Device-objective kernels + thin device module, anchor-pinned standalone; boosting_on_cuda wiring deferred to Phase 21 | ✓ |
| Wire boosting_on_cuda now | Add device GetGradients/BoostFromScore into the GBDT loop this phase | |

**User's choice:** Standalone kernels, wired in Phase 21
**Notes:** Keeps the default boosting path byte-unchanged; matches the Phase 14–18 chain (kernels behind the seam, driver phase wires them). `on_device_growth_supported()` stays false.

---

## Ranking objective scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both variants now | shared + >2048 global for both LambdaRank-NDCG and RankXENDCG | ✓ |
| Shared-bounded first, defer >2048 | Ship shared-memory path; defer the >2048 _Sorted/GlobalMemory variant; host fallback for large queries | |

**User's choice:** Both variants now
**Notes:** The Phase-14 bitonic_argsort_global/items skeletons already de-risk the global path; completing >2048 now avoids a host-fallback hole for large queries.

---

## Renew/Convert target surface

| Option | Description | Selected |
|--------|-------------|----------|
| Device kernels, standalone; integrate in P21 | RenewTreeOutput/ConvertOutput as standalone device kernels over device buffers; not swapped into live GBDT/CUDATree yet | ✓ |
| Wire onto Phase-18 CUDATree + score now | Operate on the Phase-18 device tree leaf array + resident score directly this phase | |

**User's choice:** Device kernels, standalone; integrate in P21
**Notes:** Consistent with the integration-depth choice — avoids coupling Phase 19 to Phase 18's device tree ahead of the Phase-21 driver.

---

## Claude's Discretion

- CubeCL module placement (single `objective.rs` vs per-family files) + the device-objective trait/enum shape.
- Comptime-generic-vs-per-kernel for the six regression grad kernels.
- Whether MulticlassOVA literally reuses the binary kernel or is a softmax-off variant.
- Block/geometry constants (start from faithful C++; autotune deferred).
- Which extra objectives (beyond the four representatives) also get a real C++ capture if cheap.

## Deferred Ideas

- Wiring the device objective path into the live GBDT loop → Phase 21.
- §11 score-updater constant ops + §12 metrics → Phase 20.
- CUDA-unsupported objectives (MAPE/Gamma/Tweedie/xentropy/MAP/rank-MAP) → host fallback, never ported.
- Discretized/quantized objective path (RenewDiscretizedTreeLeavesKernel) → v2 (QGD).
- APU-aware autotune of objective/rank block geometry → deferred perf option.
- Real C++ captures for the remaining 7 objectives → opportunistic hardening pass.
