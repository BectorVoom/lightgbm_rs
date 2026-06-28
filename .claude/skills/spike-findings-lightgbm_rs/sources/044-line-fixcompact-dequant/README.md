---
spike: 044
name: line-fixcompact-dequant
type: standard
validates: "Given fix_compact's dequant sub-step is a streaming u64→f64 map over [g,h] pairs, when vectorized as Array<Vector<_,N>>, then the dequant falls + stays bit-exact"
verdict: VALIDATED
related: [041, 042, 043]
tags: [performance, gpu, rocm, cpu, vectorization, vector, dequant, fix-compact, bit-exact, cross-type-cast, roi-bounded]
---

# Spike 044 — vectorize the `fix_compact` DEQUANT sub-step as `Vector<_,N>`

## What This Validates

Given `fix_compact_kernel`'s dequant sub-step (`histogram.rs:2347-2351`) is a streaming
`u64→f64` map (`hist[i] = f64::cast_from(i64::cast_from(h_raw[i])) / 2^30`) over the `[g,h]`
cells, when it's vectorized as `Array<Vector<_,N>>`, then the dequant runs faster while
staying bit-exact. This is the **one untested streaming map left in the histogram pipeline**
after 041 (subtract, won), 042 (scan, null), 043 (build, immune).

## Research

Built on the spike-041 `Vector<P,N>` recipe (CONVENTIONS). Two new feasibility findings:

- **`Vector<P,N>` supports CROSS-TYPE casts in cubecl 0.10**, bit-exactly:
  `Vector::<i64,N>::cast_from(Vector<u64,N>)` then `Vector::<f64,N>::cast_from(Vector<i64,N>)`
  lowers to per-lane casts identical to the scalar `i64::cast_from`/`f64::cast_from`. This
  extends the 041 recipe (which only did same-type element-wise ops) to **type-converting
  streaming maps**. Divide-by-scalar must broadcast: `asf / Vector::<f64,N>::new(SCALE)` —
  `Vector / const-f64` fails (the `Div<$lit>` impl needs a literal token, not a `const`).
- **hip caps f64 vectorization at vec2** (`io_optimized_vector_sizes(f64) = [2,1]`: 128-bit
  load / 64-bit f64 = 2), vs vec4 for f32 (041). So the f64 dequant ceiling on hip is lower
  than the f32 subtract's.

The dequant runs **f64 on the gfx1100/gfx1152 APU despite `has_f64==false`** (histogram.rs:2310),
exactly as the live `fix_compact_kernel` does.

## How to Run

```
cargo run --release --example spike044_vector_dequant_ab
cargo run --release --features rocm --example spike044_vector_dequant_ab
```
Source: `crates/lgbm-compute/examples/spike044_vector_dequant_ab.rs`. Discipline per
CONVENTIONS: overwrite-launch ×50 into one reused out + single read, interleaved median of
11, judge sign, 2 restarts.

## Results

**VERDICT: VALIDATED — the dequant vectorizes bit-exactly and wins in isolation; ROI-bounded
in production.** Bit-exact on every cell, both backends.

| backend | 25.6k cells | 256k cells (wide) | max width |
|---|---|---|---|
| cubecl-cpu (f64) | vec8 1.16× | **vec8 2.52×** / vec4 1.99× | vec8 |
| cubecl-hip (f64, 2 restarts) | vec2 1.04–1.08× | **vec2 1.01–1.13×** (weak, noisy) | vec2 |

`vs=1` control ~1.0× both backends (harness sanity). The cpu win scales with size like the
subtract (041) — confirms the dequant is memory-bound (minimal per-element compute: one int
reinterpret + one power-of-2 divide). The hip win is sign-positive but **weak** and
**vec2-capped** (f64).

### Why ROI is bounded (the honest disposition)

1. **hip is vec2-capped + weak** — f64's 128-bit load gives only 2 lanes, and the win is
   ~1.0–1.13× within APU noise (vs the subtract's f32-vec4 1.06–1.29×).
2. **The dequant is a FUSED minority fraction.** In production the wide rocm path fuses
   dequant+fix+compact+scan into `build_fix_scan_resident`, where the dequant is a small part
   of a single-threaded-per-feature, fix-reduction-dominated cube. A ~1.1× on that fraction
   is sub-1% e2e (the 042 "non-bottleneck inside a bigger kernel" effect, once fused).
3. **The big cpu win (2.5×) doesn't apply** — the CPU anchor's fix/dequant is the NATIVE host
   path (`fix_histogram.rs` / `build_fix_scan` host analog), not this cubecl kernel — the same
   CPU-is-native caveat as 041's subtract.

### Disposition

DON'T WIRE — keep `spike044_vector_dequant_ab.rs` as rocm-gated evidence. The dequant
vectorizes cleanly and bit-exactly (the feasibility + cross-type-cast recipe is the reusable
deliverable), but the production payoff is bounded: vec2-capped weak hip win on a fused
minority fraction, and the strong cpu win is on a native-not-cubecl path. Revisit only if a
fused-kernel refactor makes vectorizing the dequant loop near-free, or on discrete gfx110x.
This is the **third** confirmation of the 041 rule's corollary: even a correctly-shaped
streaming map (memory-bound, op-covers-bottleneck) only pays e2e when it's a MAJORITY of the
kernel's work AND on a path the backend actually runs through cubecl.

### Closes the question

With 044, every streaming/element-wise kernel in the histogram pipeline has been classified:
subtract (041, won+shipped), scan (042, null), build (043, immune), dequant (044, bounded).
**The "optimise the histogram by Vector<P,N>" frontier is now fully mapped** — there is no
remaining un-probed vectorization lever in the histogram path.
