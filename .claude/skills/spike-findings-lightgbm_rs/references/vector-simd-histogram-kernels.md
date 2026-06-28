# Vector<P,N> SIMD vectorization of the histogram pipeline (the frontier, closed)

Spikes 041–045 systematically probed whether cubecl's `Vector<P,N>` SIMD lever pays
across every streaming kernel in the histogram pipeline. **The frontier is now CLOSED:
every kernel is classified.** One win shipped; the rest are documented negatives.

## Requirements (honored)
- Bit-exact vs the CPU f64 anchor (element-wise vector ops preserve float order).
- ROCm-parity track — real payoff is on discrete gfx110x, not the spoofed APU.

## The carry-forward RULE (the durable finding)
`Vector<P,N>` pays ONLY where the kernel is **memory-bound** AND the vectorized op
**covers the bottleneck**. Vectorizing a non-bottleneck load adds register/occupancy
pressure that REGRESSES (esp. wide on the 8-CU APU). This rule classified all 5:

| Spike | Kernel | Verdict | Why |
|---|---|---|---|
| 041 | subtract (element-wise, no atomics) | ✅ **WON, SHIPPED** | pure streaming load-sub-store — all three vectorize. cpu f32-vec16 3.68×, hip f32-vec4 1.06–1.29×. Wired in quick-260627-agx. |
| 042 | split-scan pair-read | ❌ NULL | scan is a dependent prefix-sum chain + divide + argmax — NOT load-bound; vectorizing the load attacks a non-bottleneck. |
| 043 | build grad/hess input | ❌ NULL + wide REGRESSION (0.83×) | grad/hess latency already hidden behind the dominant permuted bin-gather (86–95%, structurally un-vectorizable); extract adds occupancy pressure. |
| 044 | fix-compact dequant | ✅ feasible + bit-exact, ROI-bounded | vectorizes (cpu f64-vec8 2.52×) but it's a sub-1% fused-minority fraction on a cubecl path; hip caps f64 at vec2. DON'T WIRE. |
| 045 | coalesced-build-vector (reorder→coalesced→Vector) | ❌ INVALIDATED both counts | the reorder pass IS the permuted gather (030's bottleneck) ⇒ NET loss; once coalesced the build is LDS-atomic-scatter-bound not load-bound ⇒ Vector regresses. |

## How to build it (the one win)
cubecl 0.10 has NO `Line<T>` — the type is **`Vector<P,N>`** (`Line` is a later rename).
`N:Size` is a runtime `usize` positional arg after `CubeDim`; array lengths in vector
units; widths from `io_optimized_vector_sizes`. See `subtract.rs`
`subtract_hist_kernel_vec<F,N>` + `pick_vec_width` (gate: `width>1 && n % width == 0`,
else fall back to the scalar kernel). Cross-type casts work bit-exactly
(`Vector<u64,N>→<i64,N>→<f64,N>`; divide needs broadcast `Vector::new(SCALE)`).

## What to avoid
- Don't `Vector`-ize the build bin-gather (permutation — un-vectorizable) or the scan
  (dependent chain). Both confirmed NULL/regression.
- 3 impl gotchas (spike-045): literal `Vector<_,2>` panics the cube macro (need generic
  `N:Size`); `N::value() as usize` unroll → runtime Vector index → segfault (use
  `#[unroll] for j in 0..N::value()`); reorder dest-stride is `r` not `num_data`.

## Constraints
- hip f64 caps at vec2 (vs f32-vec4); cubecl-cpu has no `Atomic<u64>`.
- Wins are size-scaling (bench WIDE: ~256k cells); ROI bounded on the APU (loses to CPU
  e2e) — reopen only on discrete gfx110x.

## Origin
Synthesized from spikes: 041, 042, 043, 044, 045. Sources in:
sources/041-line-feasibility-subtract/ … sources/045-coalesced-build-vector/.
