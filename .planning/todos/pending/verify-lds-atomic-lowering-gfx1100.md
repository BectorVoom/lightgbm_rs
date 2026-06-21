---
title: Verify cubecl-hip lowers SharedMemory<Atomic<f32>> to LDS-scoped atomics on gfx1100
date: 2026-06-21
priority: high
type: todo
context: /gsd-explore "investigate gpu kernel has bottleneck(speed) of learning large data"
---

# Todo: verify LDS-scoped atomic lowering on gfx1100

The entire LDS-histogram speedup depends on the per-workgroup sub-histogram atomics actually
landing in **LDS (shared memory)**, not silently lowering to **global** atomics. The AMD HIP fork
gets its win from `atomicAdd_block` (LDS-scoped) + a single `atomicAdd_system` global merge
(`LightGBM-release-4.6.0.99/src/treelearner/cuda/cuda_histogram_constructor.cu:33,61-68`).

CubeCL 0.10 exposes `Atomic<f32>` over both global memory and `SharedMemory<Atomic<f32>>`, but
**has no explicit memory-scope qualifier** — the backend must infer LDS scope. Our LDS kernel
uses it at `crates/lgbm-compute/src/kernels/histogram.rs:770,791`.

**Task:** confirm the cubecl-hip gfx1100 backend emits LDS-scoped atomics (e.g. `ds_add` /
shared-memory atomic ISA) for `SharedMemory<Atomic<f32>>`, not a global atomic. Approaches:
inspect generated HIP/ISA, profile LDS atomic counters (rocprof), or A/B the LDS vs global kernel
and confirm the speedup magnitude matches an LDS win. If it lowers to global atomics, the
"LDS kernel" buys nothing and the lever is dead until the backend supports scoped atomics.

**Do this AFTER** [[reconcile-gpu-hist-prod-vs-bench-kernel]] confirms there's a real gap and the
LDS kernel is the intended fix.

Known CubeCL-0.10 limits (from explorer): no f64 on gfx1100 (SP f32 LDS path only, ~1e-6 gate,
never bit-exact GPU — already accepted); no `plane_match_any` (worked around via ballot/leader
loop, `histogram.rs:555-613`).
