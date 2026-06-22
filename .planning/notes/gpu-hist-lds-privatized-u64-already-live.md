---
title: "u64 fixed-point in LDS" is ALREADY the live GPU BUILD kernel — do not re-explore as novel
date: 2026-06-22
context: explore session — "attack the f32-atomic BUILD bottleneck with a new kernel design"; survey concluded the chosen design is already shipped
supersedes_framing_in: gpu-build-atomic-contention-research.md
---

# The LDS-privatized u64 fixed-point histogram BUILD is already shipped (Phase 11)

## TL;DR for a future explorer

If you arrive here intending to "port LightGBM's CUDA LDS-privatized sub-histogram"
or "switch the GPU histogram BUILD to fixed-point integer atomics to kill f32-atomic
contention" — **stop, it is already the live production kernel.** This note exists
because an explore session (2026-06-22) re-derived that design from a CUDA survey
before discovering it was already built, wired, and verified-in-code as Phase 11.

## What the live kernel already does

- Live ROCm resident BUILD = `construct_leaf_hist_resident_lds_kernel_u64<B: Int>`
  (`crates/lgbm-compute/src/kernels/histogram.rs:1246`), dispatched by
  `resident_raw_build_into` under the `fixed_point` flag (`histogram.rs:1823-1864`).
- **LDS-privatized:** `SharedMemory::<Atomic<u64>>` per-cube sub-histogram
  (`histogram.rs:1261`), `fetch_add` into LDS, then LDS→global merge
  (`histogram.rs:1282-1291`). This is exactly the CUDA LightGBM
  `cuda_histogram_constructor.cu` pattern (shared sub-hist + single global flush).
- **u64 two's-complement fixed-point @ S=2^30** accumulation
  (`SCALE_F32 = 2^30`, `histogram.rs:630`); quantize
  `u64::cast_from(i64::cast_from(f32::round(g*SCALE_F32)))` (`:1280-81`).
  Integer atomic adds are **order-independent ⇒ deterministic** (the parity win
  over f32 atomics), dequantized back to f64 at the fix-compact seam
  (`hist = (bits as i64)/2^30`, `histogram.rs:2003-04`).
- Typed overflow guard at the resident-build boundary (one-pass `max_abs`,
  `worst = rows*max_abs*2^30 >= i64::MAX` → `ComputeError::Runtime`,
  `histogram.rs:2271-2295`).

## Why it beats f32 atomics (validated, spike-018/019)

- **Speed:** ~1.3–1.7× device-time on the HEAVY wide / large-leaves regime
  (sign-stable across 2 process runs); null-but-not-regressed in the light regime.
  Root cause: on RDNA, f32 `atomicAdd` is a `ds_cmpst` CAS-retry loop that explodes
  under high contention; integer `ds_add_u64` is native single-instruction.
  The win **composes with row-partition** (survives P=16). Determinant = total
  atomic load, not occupancy.
- **Accuracy:** within ~1e-6 of the f64 anchor (EXACT in the cancelling regime),
  ~10^3–10^4× more accurate than the f32 path. (i32 overflows → i64/u64 required;
  int16 was too coarse — that was the old spike-008 null.)
- Parity gate re-pinned to the **CPU f64 anchor** at `FIXEDPOINT_REL_GATE = 1e-7`
  with a `to_bits()` two-run determinism assert
  (`crates/oracle-harness/tests/kernel_parity.rs:1748,1841,1864`).

## What is genuinely still open (NOT the kernel design)

1. **Hardware verification unrun** — Phase 11 status is `human_needed`: 3
   `#[cfg(feature=rocm)]` truths (parity gate ≤1e-7+determinism, A/B not-slower
   sign ≥2 runs, unchanged-path regression pins) must execute on the real ROCm GPU
   and be *observed*. Persisted as `11-UAT.md` (commit `7f182f9`).
2. **WR-05 coverage gap** — the live-path parity gate only checks a 10-row P=1
   leaf, so the multi-cube **P>1 row-partition merge** the phase enables is never
   anchor-checked at P>1. → see todo `verify-u64-resident-build-parity-at-p-gt-1`.
3. **Optional composition** — stack spike-017 per-warp LDS replication on top
   (SPEC scope item 5, explicitly deferred; honest measurement needs discrete
   gfx110x — the local box is a spoofed 8-CU APU, all throughput APU-confounded).

## Process lesson

A research subagent surveying the *reference* (LightGBM CUDA) correctly described
the target design but **incorrectly inferred our kernel does per-row global
atomics**. The existing note `gpu-build-atomic-contention-research.md` already
recorded "Already LDS-privatized at the workgroup/cube level" — check our own
artifacts/code before treating a reference-derived design as a gap.
