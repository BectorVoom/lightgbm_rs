# Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-29
**Phase:** 14-foundation-shared-device-primitives-device-structs-rng
**Areas discussed:** Primitive scope, Verify anchor, Split-record representation, Seam boundary

---

## Primitive scope this phase (ODL-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Full ODL-01 now | Port every §2.4 primitive at full multi-block/global depth, even those with no consumer yet | |
| Grow-loop subset + stubs | Full depth for what Phases 15-18 consume (prefix-sum, reductions, single-block argsort); percentile + argsort-items + multi-block-global as anchor-pinned skeletons finalized by consumer phase | ✓ |
| Consumer-deferred | Only prefix-sum + reductions now; defer all argsort + percentile to first consumer | |

**User's choice:** Grow-loop subset + stubs
**Notes:** Honors §17 "port these first" without front-loading YAGNI depth onto a foundation phase whose downstream consumers (19/22) may refine signatures.

---

## Primitive & RNG verification anchor (ODL-01, ODL-02)

| Option | Description | Selected |
|--------|-------------|----------|
| In-test serial Rust f64 ref | Compute expected with serial f64 reference in each test | |
| cubecl-cpu f64-fold anchor | Run each primitive through the cubecl-cpu f64-fold path as anchor | |
| Capture C++ lib_lightgbm fixtures | Build real lib_lightgbm and capture C++ primitive outputs as committed goldens | ✓ |

**Follow-up (fixture capture mechanism):**

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: C++ where callable, else f64 ref | C++ goldens where host-callable; serial-f64 ref for pure __device__ helpers | |
| Write thin C++ harness to expose them | Small C++ test driver launches each __device__ primitive on real lib_lightgbm, dumps fixtures | ✓ |
| Reconsider → cubecl-cpu f64-fold anchor | Drop C++ capture; pin to existing cubecl-cpu f64-fold anchor | |

**User's choice:** Capture C++ lib_lightgbm fixtures via a thin C++ harness wrapping each `__device__` primitive in a launchable `__global__` shim.
**Notes:** The `__device__` helpers (ShufflePrefixSum/PercentileDevice/…) are not host-callable standalone, so a small CUDA driver is required. RNG (CUDARandom) is pinned to the existing host `Random` (already C++-bit-exact), not a new C++ capture.

---

## Device split-record representation (ODL-02)

| Option | Description | Selected |
|--------|-------------|----------|
| SoA, numeric-only, cat deferred | One buffer per numeric field, sized [num_leaf_slots]; categorical deferred entirely to Phase 22 | |
| SoA, numeric + cat reserved | Same SoA layout but also pre-allocate categorical field buffers (reserved) so Phase 22 fills not restructures | ✓ |
| Defer layout to research | Lock only the principle; researcher pins exact field set/typing | |

**User's choice:** SoA, numeric + cat reserved
**Notes:** Grounded in the CubeCL memory-preallocation manual — host-side `empty`/`empty_tensor` once outside the hot loop, reused across launches, indexed by leaf-slot. The resident record (argmax copies) is distinct from the small 8-/16-int packed readback packet, which lands with its consumer (Phase 17/18). No per-split device alloc.

---

## Seam wiring boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Strict no-op seam | Primitives + split-record + RNG + extended oracle only; growth stays Ok(None), discriminator false | ✓ |
| Wire a primitive smoke-path | Keep growth no-op but wire one trivial end-to-end on-device path behind the env flag | |

**User's choice:** Strict no-op seam
**Notes:** Real on-device growth begins at Phase 21. Slice-0 no-op seam tests must remain green.

---

## Claude's Discretion

- CubeCL module placement of new primitives (likely `kernels/primitives.rs`); CPU anchor stays serial f64 (cubecl-cpu lost to native on CPU hot paths).
- cubecl-0.10 gotcha handling in primitive design (no global barrier, broken `Atomic<i64>`, plane-sum ≤ plane width → segmented LDS scan for 256-bin, `launch_unchecked` unsafe) — research, not discussion.

## Deferred Ideas

- Percentile / multi-block argsort / ranking item-sort depth-hardening → Phase 19 / 22.
- 8-/16-int packed split-readback packet wiring → Phase 17 / 18.
- Categorical cat_threshold slab fill → Phase 22 (buffers reserved now).
- Quantized/discretized integer histogram & split path → v2 (QGD-02).
- Any actual on-device tree growth → Phase 21.
