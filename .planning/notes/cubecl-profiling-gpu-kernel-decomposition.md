---
title: CubeCL built-in profiling — true per-kernel GPU device-time decomposition (+ the single-threaded subtract lever)
date: 2026-06-22
context: GPU investigation continued — rocprof was dead on gfx1152, CubeCL's own profiler works
---

# CubeCL profiling unblocked the GPU kernel decomposition

## The tool (works where rocprof failed)

rocprof/rocprofv3 cannot profile this box (gfx1152 APU — no rocprofiler counter tables;
cubecl JIT kernels don't surface in `--kernel-trace`; see
[[gpu-is-spoofed-8cu-apu-not-gfx1100]]). **CubeCL's built-in profiler instruments the
host/JIT side and needs no GPU counters.** Recipe (TEMPORARY — do NOT commit the toml,
it logs on every GPU run):

```toml
# cubecl.toml at repo root — levels: disabled|basic|minimal|medium|full
[profiling]
logger = { level = "full", file = "/tmp/cubecl_profiling.log" }   # per-kernel device time
[compilation]
logger = { level = "basic", file = "/tmp/cubecl_compile.log" }    # JIT compile log
```
No Cargo feature needed (the file logger is built-in; `profile-tracy`/`tracing` features
are only for the Tracy GUI / rust-tracing integrations). Run any `--features rocm` binary
from the repo root; parse `/tmp/cubecl_profiling.log` lines `| <dur><unit> | <kernel>::...`.

## Per-kernel device-time breakdown (1M×50, steady-state, JIT first-calls excluded)

| kernel | device-time share | launch shape |
|--------|-------------------|--------------|
| `construct_leaf_hist_resident_lds` (BUILD, f32-atomic) | **52.6%** | `(F,P)` LDS multi-thread — atomic/memory-bound |
| `subtract_hist_kernel_f32` (SUBTRACT trick) | **31.6%** | **`(1,1,1) × 1` SINGLE THREAD** |
| `find_best_splits_fused` (SCAN) | 11.3% | `(n,1,1) × 1` |
| `data_partition` | 2.7% | — |
| `fix_compact` | 1.9% | — |

JIT first-call per kernel ≈ 120–153ms ONE-TIME (warmup-amortized; bites process-per-train).

## This CORRECTS the earlier "build = 86%" claim

The spike-015 `SCAN_DRAIN` experiment forced a readback that drained the WHOLE async queue
(build + subtract + fix + partition) into one "build" bucket → overstated build at 86–92%.
CubeCL profiles each kernel separately: **build is ~53% of device-kernel time, and the
SUBTRACTION-trick kernel is a co-dominant ~32%** — invisible to phase_prof/scan_prof because
they attributed at the wrong seam.

## NEW LEVER: parallelize `subtract_hist_kernel_f32` (bit-exact, ~32% of device time)

The kernel body is literally:
```rust
#[cube(launch)] pub fn subtract_hist_kernel_f32(parent, child, out: &mut Array<f32>) {
    if UNIT_POS == 0 { for i in 0..parent.len() { out[i] = parent[i] - child[i]; } }
}
```
**Only thread 0** runs the entire elementwise `parent − child` over ~25k elements (50 feat ×
256 bins × 2) on ONE of the GPU's ~512 lanes. Heritage: *"single-owner to keep the cpu
launch shape consistent"* — it inherited the CPU f64-anchor's serial shape and was never
re-parallelized for the device.

**Fix:** launch a real workgroup and stride by `ABSOLUTE_POS` (each thread owns disjoint
elements). **BIT-EXACT** — each `out[i]` is independent, same f32 subtract, no atomics, no
ordering, no contention (unlike the BUILD, which is f32-atomic ~1e-6). The cleanest GPU
device-time win available: cuts ~32% of per-iteration device time to near-zero, with zero
parity risk. See [[parallelize-subtract-hist-kernel]].

Caveat: on the 8-CU APU, GPU still loses to CPU overall ([[wide-tall-two-backend-root-cause]])
— this is a GPU-internal cleanup, valuable mainly on real discrete hardware where the GPU
path matters. But it's correctness-shaped (no kernel should run 1-lane) and bit-exact-safe.
