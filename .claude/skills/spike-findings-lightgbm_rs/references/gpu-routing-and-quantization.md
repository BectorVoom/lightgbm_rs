# GPU Routing (when to use it) & Quantized Histograms (why not exact)

Implementation blueprint from spikes 001 and 008 — two "scope/feasibility" findings that
bound what the GPU and quantization tracks can claim.

## Requirements

- "GPU is faster" is claimed ONLY in the regime the data supports — never route
  small/medium datasets to the GPU.
- Quantized/discretized paths may NOT be presented as exact; the ~1e-6 / bit-exact
  contract is the project's core value.

## When the GPU wins (spike 001 — VALIDATED)

With *today's* kernels (no batched find_best_split/subtract/partition), the ROCm f64 path
crosses below the **single-thread** CPU f64 anchor at **≈700k rows** (feat=50, bins=255,
31 leaves, regression); robust GPU win ≳1M, widening to 1.45× at 2M (GPU rows/s still
climbing — not saturated).

| rows | winner | GPU speedup |
|------|--------|-------------|
| ≤200k | CPU | 0.06–0.58× |
| 700k | tie | ~0.99× |
| 1M / 1.5M / 2M | GPU | 1.12–1.24× / 1.39× / 1.45× |

**Routing rules for the build:**
- The GPU's high fixed **launch-bound floor** (~5–6s/30 iters regardless of rows ≤100k)
  is what makes small data lose. The parked batch-find_best_split/subtract/partition seed
  would *lower* that floor → move the crossover left. It's an **optimization, not a
  prerequisite** (the GPU already wins at scale without it).
- ⚠️ **The 700k crossover is vs SINGLE-thread CPU.** Spike 005 made the CPU anchor
  multi-threaded (≈16× at large), so the real crossover moves to **many millions of rows**.
  Re-measure against the multi-threaded anchor before any GPU-at-scale claim.
- Crossover shifts right with more features/bins/leaves (more launches = higher floor).
- Open gate: this measured **wall-clock, not parity** — confirm the GPU still matches the
  anchor at 1M+ rows separately.

## Why quantized histograms can't be the exact path (spike 008 — INVALIDATED)

LightGBM's int16 discretized build (`use_quantized_grad`, **default FALSE**) packs
quantized (grad,hess) into one int32 and uses a single integer atomic — real speed
mechanism. But a cheap CPU probe (`quant_parity_probe.rs`, no GPU — probe-before-plumbing)
shows the quantization floor is **irreducible**:

| quant_bins | rel err (determ.) | vs gate (1e-5/1e-6) |
|-----------:|------------------:|---------------------|
| 4 (LGBM default) | 2.1e0 | FAIL (200%+) |
| 256 | 6.2e-2 | FAIL |
| **65536 (full int16)** | **3.2e-4** | **FAIL (~30× over 1e-5, ~300× over 1e-6)** |

No bin count closes it; stochastic rounding is unbiased only across the *ensemble*, worse
per-build. **Disposition:** it can only ever be a **separate opt-in `use_quantized_grad`
approximate mode** (mirroring C++) — a product/scope decision with its own approximate
contract, never a drop-in for the exact build. (This was later built as the gated
phase-10 quantized-training mode — see project memory; it stays opt-in + gated vs C++
deterministic goldens, never the default exact path.)

## What to Avoid

- Quoting the 700k crossover without the multi-threaded-CPU caveat.
- Building the discretized GPU kernel as an exact-path optimization (it can't pass the gate).

## Harnesses

`bench_crossover.rs` (env-var size sweep, one rocm build), `quant_parity_probe.rs`.

## Origin

Spike 001 (VALIDATED — crossover), spike 008 (INVALIDATED for exact parity). Sources in
`sources/001-*`, `sources/008-*`.
