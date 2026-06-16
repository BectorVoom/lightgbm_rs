---
spike: 008
name: 16bit-discretized-hist
type: standard
validates: "Given LightGBM's int16 discretized histogram (Lever 3), when the quantization parity envelope is measured vs the f64 anchor, then it is decided whether the path can meet the project's exact-parity contract or is approximate-only"
verdict: INVALIDATED
related: [007, 006]
tags: [performance, gpu, rocm, histogram, quantization, parity, negative-result]
---

# Spike 008: 16-bit discretized histogram (INVALIDATED for exact parity)

## What This Validates

The third (and largest) lever from `cubecl_kernel_gaps.md` /
`.planning/notes/cubecl-vs-rocm-histogram-kernel-comparison.md`: LightGBM's discretized
histogram build. Each row's `(grad, hess)` is quantized to int16
(`q = round(value * scale)`, `scale = (bins/2)/abs_max`), **packed into one int32**, and
accumulated with a SINGLE integer atomic — halving atomic count + LDS width vs our two f32
atomics (`CUDAConstructDiscretizedHistogramDenseKernel`, `cuda_gradient_discretizer.cu`).

The speed mechanism is real. The question that gates it for lightgbm_rs: **can int16
quantization meet the project's non-negotiable ~1e-6 exact-parity-to-C++ contract**, or is
it irreducibly approximate?

## Research (the decisive context)

LightGBM exposes this as **`use_quantized_grad`, default `FALSE`** — an explicit opt-in.
The config docs say it outright (`config.h:624-637`):
> "gradient quantization can accelerate training, **with little accuracy drop in most
> cases** … with more bins, the quantized training will be **closer to** full precision."

i.e. it is an APPROXIMATE mode by construction — a speed/accuracy tradeoff, never exact.
`num_grad_quant_bins` default = **4**. Stochastic rounding makes it unbiased over the
*ensemble*, not per-histogram.

## Method

Cheap CPU probe (`crates/lgbm-compute/examples/quant_parity_probe.rs`, no GPU — spike-006
"probe before plumbing" discipline): realistic binary-logloss grads (g∈(-1,1), h∈(0,0.25]),
200k rows × 64 bins. Quantize at several bin counts, build the int histogram, de-quantize,
compare to the exact f64 histogram. Deterministic AND stochastic rounding.

## Results

**VERDICT: INVALIDATED for exact parity.** Even at FULL int16 the drift is ~30–300× the gate:

| quant_bins | rel err (determ.) | rel err (stochastic) | vs project gate (1e-5 / 1e-6) |
|-----------:|------------------:|---------------------:|-------------------------------|
| 4 (LGBM default) | 2.1e0 | 1.76e1 | FAIL (200%+ error) |
| 16 | 1.2e0 | 1.44e0 | FAIL |
| 256 | 6.2e-2 | 1.03e-1 | FAIL |
| 4096 | 1.4e-2 | 1.11e-2 | FAIL |
| **65536 (full int16)** | **3.2e-4** | 1.5e-3 | **FAIL (~30× over 1e-5)** |

- **The quantization floor (~3e-4 at full int16) is irreducible** and sits ~30× above the
  f32 GPU gate, ~300× above the ~1e-6 CPU-anchor contract. No bin count closes it.
- Stochastic rounding is **worse per-build** (unbiased only across many trees, not within one
  histogram) — so it does not help a per-build parity gate.

## Signal for the Build

- **DO NOT build the discretized GPU kernel as a drop-in for the exact build.** It can never
  pass the bit-exact CPU merge gate or the ~1e-6 contract that is the project's core value.
  The cheap CPU probe killed a large packed-int-atomic kernel + gradient-discretizer plumbing
  before any of it was written (same outcome class as spike-006).
- **The ONLY place it could live is a SEPARATE opt-in `use_quantized_grad` approximate mode**,
  mirroring C++'s opt-in — a PRODUCT/SCOPE decision (new config surface, its own approximate
  contract, ensemble-level accuracy validation), not a kernel optimization. Deferred to that
  decision; see the updated seed `.planning/seeds/16bit-discretized-histogram.md`.
- Lever 3 is therefore **closed as "not applicable to the exact path."** Remaining gap-closure
  effort goes to the parity-SAFE lever (multi-feature-per-cube packing), which preserves the
  exact f32 build.

Reusable: `quant_parity_probe.rs` (the quantization-envelope probe; reuse if a quantized mode
is ever scoped).
