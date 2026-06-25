---
spike: 018
name: fixedpoint-int-atomics
type: standard
validates: "Given the wide f32-atomic histogram BUILD, when each grad/hess is accumulated as wide fixed-point i64 (scale 2^30) via integer LDS atomics instead of f32, then it (a) stays within the ~1e-6 ROCm gate of the f64 anchor, (b) is order-independent/deterministic, and (c) is not slower — resolving research Q2 (finding #3)"
verdict: VALIDATED (strong) — fixed-point u64 atomics are ~1.9× FASTER + ~3600× more accurate + deterministic at the wide P=1 regime; the strongest GPU build lever found. Wiring is a major change (parity re-pin) — disposition pending.
related: [008, 006, 007, 015, 017]
tags: [performance, gpu, rocm, histogram, fixed-point, integer-atomics, determinism, parity, finding-3, wide-shape]
---

# Spike 018: Fixed-point integer atomics (research Q2 / finding #3)

## What This Validates

Research finding #3: accumulate the histogram with INTEGER atomics instead of f32.
Q2's open question: under the *contention* goal (not quantized training), does
fixed-point pay off vs the ~1e-6 ROCm budget — given the GPU int kernel was already
"null (W5)"? Two sub-questions, risk-ordered (parity gate first, spike-008 discipline):

- **018a (CPU host probe — the GATE):** can wide i64 fixed-point (q=round(g·S), exact
  integer accumulation, dequantize sum/S) stay ≤~1e-6 of the f64 anchor incl. the
  cancelling regime, AND be order-independent (deterministic)?
- **018b (GPU, only if 018a passes):** does cubecl-hip 0.10 support i64/u64 atomics, and
  is integer-atomic throughput ≥ f32? Reconcile the W5 null.

## Investigation Trail

1. **Prior-art framing.** W5 (`gpu_packed_int.rs`) found packed-int16 (one i32 atomic)
   NULL vs f32 (1.00–1.01×); spike-008 INVALIDATED int16 *quantization* for parity
   (3.2e-4). Both are about COARSE int16 packing. Q2 is WIDE i64 fixed-point of the
   EXACT grads — different mechanism. The live rationale after W5 is determinism +
   parity-quality, NOT (predicted) speed.
2. **018a parity probe (CPU, `examples/fixedpoint_parity_probe.rs`).** Decisive PASS —
   see Results. The gate OPENS (opposite of spike-008): wide i64 clears ~1e-6 with
   margin and is *exact* in the cancelling regime.
3. **018b GPU feasibility — first wall.** `Atomic<i64>` compiles but FAILS at runtime:
   cubecl-hip 0.10 lowers `Atomic<i64>::store` to `atomicExch(long long*, …)`, and HIP
   has NO `atomicExch` for `long long*` (only `unsigned long long*`). The `fetch_add`
   path correctly reinterpret-cast to `uint64*`; `store` did not. **cubecl-hip 0.10
   codegen limitation.**
4. **018b workaround — `Atomic<u64>` two's-complement.** Store the i64 quantized value's
   BITS as u64; wrapping u64 `atomicAdd` == signed i64 add (two's complement); host
   reinterprets final bits back to i64. Compiles AND runs. (Bias-offset was tried first
   but is wrong — each bin sums a variable row count, not N.)
5. **018b speed — surprising win, then rigorous confirm.** Raw 3-round showed i64/f32
   1.1–1.83× (noisy, launch-bound: per-call readback). Upgraded to compute-throughput
   (accumulate LAUNCHES → one read, interleaved median+p25/p75, 2 process runs):
   **i64/f32 = 1.80×/2.06×, both SEP-WIN.** Contradicts W5 — reconciled by regime
   (below).

## Results

### 018a — parity gate: PASS (CPU, N=1M, vs f64 anchor)

| case | f32-atomic envelope (GPU's current error) | i64 fixed-point S=2³⁰ | det? |
|------|------------------------------------------:|----------------------:|:----:|
| residual ±0.5 | 7.1e-4 | **2.5e-8** (PASS) | yes |
| cancelling ~0 | 7.7e-5 | **0.0 (exact)** (PASS) | yes |
| hessian +1    | 1.1e-3 | **0.0 (exact)** (PASS) | yes |

i64 @ S=2³⁰ clears ~1e-6 with margin on ALL cases, **~10³–10⁴× more accurate than the
f32 path**, exact where f32 catastrophically cancels, and order-independent. i32
OVERFLOWS the hessian case (why i64 is required). S=2³⁰ never overflows i64 for ≤~1e9
rows. (Opposite of spike-008: int16 was too coarse; wide i64 is plenty.)

### 018b — GPU feasibility + speed (gfx1152 APU, single cube = wide P=1 regime)

- **Feasibility: YES via `Atomic<u64>`** (i64 blocked by a cubecl-hip 0.10 `atomicExch`
  codegen gap; u64 two's-complement works).
- **Accuracy on hardware: i64 5.9e-9 vs f32 2.2e-5** (~3600×), matching 018a.
- **Speed: i64/f32 = 1.80× (run1) / 2.06× (run2), both SEP-WIN** (i64 p75 < f32 p25;
  sign-stable across processes). f32 73–77ms[72..81], i64 37–40ms[36..43].
  **⚠ CORRECTED by spike-019:** this 1.9× is INFLATED by the single-cube SIMPLE kernel
  (direct `binned[i]`, no `leaf_rows` indirection). In the realistic resident `build_rp`
  kernel the magnitude is **~1.3–1.7×** (still robust, sign-stable, occupancy-independent,
  composes with row-partition). The win is REAL in high-atomic-load regimes (the wide
  root/large leaves), null only at light load. See `019-int-atomic-contention-regime`.
- **Determinism: i64 deterministic by construction.** (f32 showed bit-eq here too, but
  that's *incidental* to the single-cube stable scheduling — multi-cube f32 is NOT
  deterministic; i64 always is.)

### Reconciling the W5 null (important)

W5: packed-int (1 i32 atomic) vs f32 (2 atomics), **row-partitioned P=16** → null.
018b: u64 (2 atomics) vs f32 (2 atomics), **single cube P=1** → ~1.9×. Both right *in
their regime*: under **high per-cube contention** (P=1, the WIDE shape), f32 `atomicAdd`
on RDNA lowers to a **CAS retry loop (`ds_cmpst`)** whose retries explode with
contention, while integer `ds_add_u64` is a native single-instruction LDS op that never
retries (research finding #3 / AMD atomics docs). At P=16 (low contention, tall-narrow)
the retries are rare so atomic type doesn't matter — W5's null. **018b's regime is the
wide bottleneck (spike-015), so the win lands where it matters.** (Testable prediction,
not yet run: the win should SHRINK as P rises.)

## Significance

This is the **strongest GPU build-kernel lever in the campaign** and it is **triple-aligned
with the project's parity-first contract**: ~1.9× device-time AND ~3600× more accurate
(2.2e-5 → 5.9e-9, much closer to the f64 anchor) AND deterministic. It partly reframes
the whole "atomic-contention-bound build" story: a large share of that cost is the f32
atomic being a **CAS loop**, not contention per se — switching to integer atomics
attacks the real instruction-level bottleneck.

## Disposition — pending human decision (major change)

NOT auto-wired. Unlike spike-017 (modest) / p93 (null), this is a robust, multi-axis
win worth considering for production — but it's a **large change**: a new fixed-point
build kernel (u64 two's-complement), the resident/merge buffers go i64 (2× bytes), and
the GPU f32 accumulation order changes ⇒ a full oracle-harness parity RE-PIN
(kernel/learner/boosting on gfx1100; def-f8u-01 guardrail — pin to the CPU f64 anchor).
Because the new path is *more* accurate, it should clear the ~1e-6 gate more easily.

Caveats before wiring: (1) APU-only, single-cube microbench — the regime matches wide
P=1, but discrete-gfx110x confirmation is ideal (option-ii: device-time proxy accepted);
(2) overflow guard needed for extreme leaves (>~1e8 rows × large grads) — clamp scale or
range-check; (3) the contention-regime reconciliation (win shrinks at P>1) is a
hypothesis — a P-sweep would confirm it and tell us whether to keep row-partition or
replace it with integer atomics.

Evidence: `examples/fixedpoint_parity_probe.rs` (018a), `examples/gpu_fixedpoint_i64.rs`
(018b). Reusable: the comptime-flag in-kernel A/B + median/p25/p75 harness (CONVENTIONS).
