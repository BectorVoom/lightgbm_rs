# Phase 18: On-Device Data Partition, Tree Mutation & Prediction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-01
**Phase:** 18-on-device-data-partition-tree-mutation-prediction
**Areas discussed:** Partition device path & flag fan-out, Row-order anchor, Predict scope, Device tree + readback, Categorical scope, Routing gate, Fixtures

---

## Partition device path & flag fan-out

| Option | Description | Selected |
|--------|-------------|----------|
| New §9 device kernel, full flag fan-out | New faithful mark→prefix-sum→scatter parallel to the shipped host-gather kernel; wire the full GenDataToLeftBitVector flag set now | ✓ |
| New §9 device kernel, numeric-no-missing only | Build the §9 scatter but wire only default numeric routing; defer missing/NA flags | |
| Extend the existing host-gather kernel | Reuse `data_partition_kernel` route + add device prefix-sum/scatter on top | |

**User's choice:** New §9 device kernel, full flag fan-out
**Notes:** → CONTEXT D-01/D-02. The shipped host-gather kernel is a routing-decision reference, not the device anchor (Phase-17 D-01 pattern). Missing/default_left can't cleanly defer given SC #1/#3.

---

## Row-order anchor (SC #1)

| Option | Description | Selected |
|--------|-------------|----------|
| Stable partition anchor (order-equivalent) | cpu anchor = single-owner stable partition (order-equivalent to §9 block-tiled scatter); hip runs the real scatter; verify equivalence in research | ✓ |
| Faithful block-tiled scatter anchor | Anchor reproduces block-tiled prefix-sum + in-block rank scatter with block/grid geometry | |

**User's choice:** Stable partition anchor (order-equivalent)
**Notes:** → CONTEXT D-04. Conditional: research must confirm the reference order IS a plain stable partition; if not, escalate to the faithful block-tiled anchor.

---

## Predict scope (SC #3)

| Option | Description | Selected |
|--------|-------------|----------|
| Numeric predict + cat seam; include §9 leaf-map add | Numeric tree-walk + cat dispatch seam (defer cat math to Phase 22); include §9 AddPredictionToScore | |
| Numeric predict + cat seam; defer §9 leaf-map add to Phase 20 | Same predict, push §9 leaf-map add to Phase 20 | |
| Full predict incl. categorical math now | Build numeric AND categorical membership eval in the predict kernel this phase | ✓ |

**User's choice:** Full predict incl. categorical math now
**Notes:** → CONTEXT D-05/D-06. Clarified in follow-up: "full cat math" = full bitset MEMBERSHIP (model-consuming), consistent with the ROADMAP "membership routing is wired here"; cat split-FINDING stays Phase 22. §9 leaf-map AddPredictionToScore included here (part of the grow chain).

---

## Device tree + readback

| Option | Description | Selected |
|--------|-------------|----------|
| Device flat tree + 16-int packet, reconcile w/ 8-int | Device-resident flat CUDATree, Split before partition, two transfers/iter (8-int + 16-int), pool swap via Phase-16 arena | ✓ |
| Keep tree on host, only partition+score on-device | Host owns Tree; device only partition + predict — contradicts SC #2 | |
| You decide | Let research determine layout/round-trip/reconciliation | |

**User's choice:** Device flat tree + 16-int packet, reconcile w/ 8-int
**Notes:** → CONTEXT D-07/D-08/D-09.

---

## Categorical scope (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — full membership in BOTH predict + partition | Full FindInBitsetCUDA membership in §10 predict AND §9 partition routing; split-finding stays Phase 22 | ✓ |
| Predict membership only; partition cat = seam | Full cat predict, partition cat routing stays a seam until Phase 22 | |
| You decide | Research decides how far cat membership can go now | |

**User's choice:** Yes — full membership in BOTH predict + partition
**Notes:** → CONTEXT D-03/D-05.

---

## Routing gate

| Option | Description | Selected |
|--------|-------------|----------|
| Clone LGBM_RESIDENT_FORCE size-gate, off by default | Size-gated + default-off; ROCm keeps host path; env override for benching | |
| On-device partition default-on when env set | With LGBM_CUDA_ON_DEVICE=1, route partition on-device unconditionally (no size gate) | ✓ |

**User's choice:** On-device partition default-on when env set
**Notes:** → CONTEXT D-10. Overrides spike-035 (APU round-trip overhead); accepted because the env seam keeps the default path byte-unchanged and APU-vs-discrete tuning is the Phase-23 DoD.

---

## Fixtures

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse host-trained golden models + Phase 15/16 synthetic cols | Anchor against existing host models + Phase-15/16 fixtures; no new C++ goldens | |
| Capture new C++ goldens for this phase | Dedicated lib_lightgbm captures for partition row-order + cat-model predict | ✓ |
| You decide | Research picks the minimal fixture set | |

**User's choice:** Capture new C++ goldens for this phase
**Notes:** → CONTEXT D-11. Reuse Phase-15/16 fixtures where they fit; new goldens cross-check the cpu f64 anchor.

---

## Claude's Discretion

- CubeCL module placement (likely new `data_partition.rs` + `tree.rs`/`predict.rs`).
- `AggregateBlockOffsetKernel0`/`1` as two kernels vs one runtime-branched kernel.
- §9/§10 geometry constants (block sizes, scalar fan-out grids) — occupancy knobs, parity-neutral.
- Device→host host-`Tree` reconstruction point (per-split vs per-tree).

## Deferred Ideas

- Categorical split-FINDING / cat-feature end-to-end → Phase 22.
- §11 score constant scalar ops + §12 metrics → Phase 20.
- On-device objectives / ConvertOutput inverse-link → Phase 19.
- Discretized/quantized partition + RenewDiscretizedTreeLeaves → v2 (QGD).
- APU-vs-discrete partition routing tuning + size-gate → Phase 23 perf/rollout DoD.
- APU-aware autotune of §9/§10 geometry → deferred perf option.
- Reviewed-not-folded todos (4 GPU-perf profiling todos) → Phase 23, keyword-only matches.
