# Phase 20: On-Device Score Updater & Metrics - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-02
**Phase:** 20-on-device-score-updater-metrics
**Areas discussed:** Score-updater integration depth, Metric anchor strategy, ConvertOutput boundary, Unsupported-metric fallback gate, Residency scope & Phase 20/21 boundary, Score parity gate

---

## Score-updater integration depth

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone kernels, defer wiring | Standalone anchor-pinned constant-op kernels + host-mirror toggle; boosting_on_cuda_ wiring → Phase 21 (Phase 19 D-02 style) | |
| Wire resident score now | Make cuda_score_ resident across iterations in the boosting layer this phase (touch boosting_on_cuda_ seam) | ✓ |

**User's choice:** Wire resident score now.
**Notes:** Chose the more integrated path over the conservative chain-discipline default. Refined further below (Residency scope).

---

## Metric anchor strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Real C++ goldens (all 12) | Anchor every device metric kernel to the existing real compiled-lib_lightgbm captures in oracle-harness/fixtures/metric/ | ✓ |
| cpu-f64 anchor + per-family C++ cross-check | lgbm-metric f64 primary + one representative C++ cross-check per family (Phase 19 D-01) | |

**User's choice:** Real C++ goldens (all 12).
**Notes:** Fixtures already on disk → near-zero cost for genuine reference fidelity. → CONTEXT D-03.

---

## ConvertOutput boundary

| Option | Description | Selected |
|--------|-------------|----------|
| EvalKernel takes pre-converted scores | EvalKernel operates on already-converted scores; ConvertOutput→score_convert_buffer_ compose deferred to Phase 21 | |
| Compose ConvertOutput into Eval now | Wire the Phase-19 device ConvertOutput into the Eval flow this phase, end-to-end, anchor-pinned | ✓ |

**User's choice:** Compose ConvertOutput into Eval now.
**Notes:** Consistent with building the full path this phase. → CONTEXT D-04.

---

## Unsupported-metric fallback gate

| Option | Description | Selected |
|--------|-------------|----------|
| Metric-supported discriminator | Default-false-style discriminator routing AUC/NDCG/MAP/multiclass/xentropy to host; device-evaluates the 12 pointwise losses; notes MAPE/Gamma/Tweedie metric-vs-objective asymmetry | ✓ |
| Let planner decide the gate shape | Lock only the supported/unsupported sets; leave code shape to planner | |

**User's choice:** Metric-supported discriminator.
**Notes:** Exact supported/unsupported sets locked; asymmetry with Phase-19 objective fallback captured. → CONTEXT D-05.

---

## Residency scope (follow-up on "wire resident score now")

| Option | Description | Selected |
|--------|-------------|----------|
| Resident buffer + constant ops + per-tree device AddScore | Resident cuda_score_ across iterations over host-grown trees; on_device_growth_supported() stays false; grow stays Phase 21 | |
| Full end-to-end resident loop | Score never leaves device across the whole train, fed by on-device grow; front-runs Phase 21's driver | ✓ |

**User's choice:** Full end-to-end resident loop.
**Notes:** Escalated to the Phase 20/21 boundary question below.

---

## Score parity gate

| Option | Description | Selected |
|--------|-------------|----------|
| Anchor-pinned kernels + resident-score A/B vs host score_ | Anchor constant-op kernels to cpu f64 AND assert resident cuda_score_ matches host score_ after a full multi-iteration run | ✓ |
| Kernel-level anchor only | Anchor individual kernels only; defer full-run A/B to Phase 21 | |

**User's choice:** Anchor-pinned kernels + resident-score A/B vs host score_.
**Notes:** Combined with the structure gate pulled forward from ODL-18. → CONTEXT D-06.

---

## Phase 20/21 boundary (tension resolution)

| Option | Description | Selected |
|--------|-------------|----------|
| Resident score, host-grown trees (keep boundary) | Resident cuda_score_ over host-grown trees; on_device_growth_supported() false; grow loop stays Phase 21 (Recommended) | |
| Pull Phase 21's on-device grow into Phase 20 | Merge ODL-18 forward: full on-device driver + STRUCTURE-bit-exact gate in Phase 20; Phase 21 becomes categorical/hardening | ✓ |

**User's choice:** Pull Phase 21's on-device grow into Phase 20.
**Notes:** Deliberate roadmap re-scoping against the recommended option. Phase 20 becomes the end-to-end driver phase; Phase 21 shrinks to hardening/slack. ROADMAP.md Phase 20/21 entries should be re-scoped via /gsd-phase before planning. → CONTEXT D-01.

---

## Claude's Discretion

- EvalKernel as one comptime-generic `#[cube]` vs 12 concrete kernels (parity-neutral).
- Code shape of the metric-supported discriminator (enum method vs match) — only the sets are locked.
- Module placement (score_updater.rs / metric.rs in lgbm-compute; boosting_on_cuda_ wiring in lgbm-boosting).
- Block/geometry constants (NUM_DATA_PER_EVAL_THREAD=1024, num_threads_per_block_) — faithful C++ defaults; autotune deferred.
- Driver Handle in-place-alias vs ping-pong double-buffer; batched client.read readback semantics — verify at plan time.

## Deferred Ideas

- On-device categorical splits → Phase 22.
- Perf-validation + default-ON rollout DoD → Phase 23.
- CUDA-unsupported metrics (AUC/NDCG/MAP/multiclass/xentropy/KL) → permanent host fallback.
- APU-aware autotune of EvalKernel / grow-loop geometry → deferred perf option.
- Phase 21 re-scope to hardening/slack (or fold into 22/23) via /gsd-phase.
- Four GPU-perf-profiling todos (loop profiling, low-row A/B, GPU-vs-CPU crossover, large-data fixture) → Phase 23.
