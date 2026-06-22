---
title: The "gfx1100 GPU" is a spoofed 8-CU APU (Radeon 860M / gfx1152) — recontextualizes the whole GPU campaign
date: 2026-06-22
context: rocprof investigation of "why the atomic build sits at 820 Mr/s"
---

# The GPU is a spoofed 8-CU APU, not a discrete gfx1100

## What the rocprof investigation found

Trying to profile the f32-atomic histogram build with rocprof (ROCm 7.1.1) surfaced the
real hardware identity instead of micro-counters:

- `rocprofv3 --pmc` **aborts**: `rocprofiler_iterate_agent_supported_counters failed for
  agent 1 (gfx1152) :: Agent HW architecture is not supported, no counter metrics found.`
  → the TRUE silicon is **gfx1152** (RDNA 3.5); `HSA_OVERRIDE_GFX_VERSION=11.0.0` only
  spoofs the runtime, not the profiler. **Hardware counters are unavailable on this box.**
- `rocprofv3 --kernel-trace` captures only `__amd_rocclr_copyBuffer` — cubecl-hip's JIT
  compute dispatches don't surface (compounded by cubecl-hip-sys `hipconfig` panics under
  the profiler's exec intercept). So per-kernel HW counters are doubly blocked here.
- `rocminfo`: Device = **AMD Radeon 860M** (Ryzen AI 7 350, Strix Point APU),
  **Compute Unit: 8**, 2 SIMDs/CU, wavefront 32, 3 GHz, **Memory Properties: APU**
  (16 GB GPU pool carved from 32 GB system DDR5 — no dedicated VRAM).
- CPU = same chip: Ryzen AI 7 350, 8 cores / **16 threads**, sharing the **same DDR5 bus**.

## Why this is the answer to "why 820 Mr/s"

The atomic build is an **8-CU integrated GPU doing uncoalesced atomic-scatter on system
DDR5 shared with the 16-thread CPU**. ~820 Mr/s (spike-007) / 234 Mr/s naive (spike-006)
is roughly what that hardware class sustains — it was never a 96-CU discrete gfx1100
(which has ~192 SIMDs and ~960 GB/s dedicated GDDR6).

## The code is calibrated for hardware that isn't present

`crates/lgbm-compute/src/kernels/histogram.rs`:
- `ROWPART_TARGET_CUBES = 768` with the comment "~8 workgroups × 96 CUs (gfx1100)" —
  the real device has **8 CUs**, so 768 cubes **over-subscribes by 12×**. The row-partition
  gate `768/nf` is mis-tuned for this APU (should be ~64 = 8 wkgrps × 8 CU).
- spike-007's "one-cube-per-feature starves the 96-CU GPU / P=16 sweet spot / P=32
  over-partitions" is all relative to a phantom 96-CU device.

## Implications (correct the campaign's framing)

1. **"GPU loses to the 16-core CPU at every shape / crossover erased"
   ([[wide-tall-two-backend-root-cause]]) is HARDWARE-CONFOUNDED.** It's an 8-CU iGPU vs a
   16-thread CPU on ONE shared DDR5 bus. This is expected and **must NOT be generalized to
   a discrete gfx1100** (96 CU, GDDR6) — there the GPU could well win. All "GPU is
   parity-maintenance not a speed path" conclusions are true *for this APU only*.
2. **Per-kernel HW profiling is impossible on this box** (rocprofiler has no gfx1152
   counter tables; cubecl JIT kernels also don't trace). Future GPU micro-optimization
   that needs rocprof/occupancy/roofline data requires either a rocprofiler build with
   gfx1152 support, a real discrete gfx1100/gfx1101, or a different profiling approach
   (e.g. cubecl's own kernel dump + analytical occupancy).
3. If discrete-GPU performance matters, **re-benchmark on real gfx1100/gfx110x hardware**
   before trusting any GPU-vs-CPU routing conclusion; and **re-tune `ROWPART_TARGET_CUBES`
   to the actual CU count** (query `rocminfo` Compute Unit at runtime instead of hardcoding 96).

## Reproduce

```
rocminfo | grep -A12 'Name: *gfx1100'   # shows Compute Unit: 8, Memory Properties: APU
rocprofv3 --pmc SQ_WAVES -- <any rocm binary>   # aborts: gfx1152 not supported
```
