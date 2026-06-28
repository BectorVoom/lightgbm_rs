---
spike: 041
name: line-feasibility-subtract
type: standard
validates: "Given cubecl 0.10 on hip + cpu, when the element-wise histogram subtract is rewritten over Array<Vector<F,N>> (vector_size swept from io_optimized_vector_sizes) vs the scalar kernel, then it compiles + runs on AMD + CPU, stays bit-exact, and device-time falls"
verdict: VALIDATED
related: [024, 042, 043]
tags: [performance, gpu, rocm, cpu, vectorization, line, vector, subtract, kill-question, bit-exact]
---

# Spike 041 — `Vector<P,N>` feasibility on the histogram SUBTRACT (the kill question)

## What This Validates

Given cubecl 0.10 on the hip (gfx1100/gfx1152 APU) and cpu runtimes, when the
element-wise histogram subtract (`out[i] = parent[i] − child[i]`) is rewritten over
`Array<Vector<F,N>>` with `vector_size` swept from `io_optimized_vector_sizes`, then it
**(a)** compiles + runs on both backends, **(b)** stays **bit-exact** to the scalar
kernel, and **(c)** device-time falls at width > 1. This is the KILL QUESTION for the
whole "optimise the cubecl kernel by `Line<T>`/`Array<Line<T>>`" idea — subtract is the
cleanest possible target (element-wise, no atomics, no permutation), so if `Vector`
broke here it would be dead everywhere.

## Research

### The headline API finding (read from the cubecl 0.10 SOURCE, not the manual)

**cubecl 0.10 has NO `Line<T>` type.** The user asked for `Line<T>` / `Array<Line<T>>`;
the actual vectorized container on this pinned version is **`Vector<P: Scalar, N: Size>`**
(`cubecl-core-0.10.0/src/frontend/container/vector/base.rs:11`). `Line` is the name on a
later `main`/the burn book — a rename. The cubecl-main docs (context7) already show the
`Vector<F, N>` examples. This is the same class of "manual-divergence" finding spike-037
logged for autotune; capture it so future vectorization work starts from the right type.

**The launch ABI** (canonical reference: `cubecl-core-0.10.0/src/runtime_tests/vector.rs`):

| Concern | How (0.10) |
|---|---|
| kernel signature | `parent: &Array<Vector<F, N>>`, `N: Size` a generic type param |
| the `N: Size` value | a **runtime `usize`** inserted as a positional arg **right after `CubeDim`**, before the kernel's own args: `launch::<F,R>(client, count, dim, vector_size, parent, child, out, …)` |
| array length | `ArrayArg::from_raw_parts(handle, n_elements / vector_size)` — counted in **vector units** over the SAME byte buffer |
| a bare `usize` kernel param | passed **raw** (e.g. `n_vec,`), NOT wrapped in `ScalarArg` (mirrors production `num_data,`) |
| which widths to sweep | `client.io_optimized_vector_sizes(size_of::<F>())` → backend's useful widths (hip f32 → `[4,2,1]`; cpu f32 → `[16,8,4,2,1]`, f64 → `[8,4,2,1]`) |
| element ops | `Vector` impls `Add/Sub/Mul/…` element-wise (`vector/ops.rs`) + broadcasting; element read `v[i]`, element write via `RuntimeCell::store_at`, comptime size `N::value()` |

### Why subtract is the kill-question target

`subtract_hist_kernel` (`kernels/subtract.rs`) is the cleanest vectorization fit in the
codebase: pure `out[i] = parent[i] − child[i]` 1D grid-stride over the `[g,h,g,h,…]`
cells — no atomics, no permutation, no reduction. `Vector::sub` is element-wise, so the
result is **bit-exact to scalar by construction** (no float op is reordered). Confirmed:
`bit_exact=true` on **every** width × size × backend cell below.

## How to Run

```
cargo run --release --example spike041_vector_subtract_ab               # cubecl-cpu (f64 + f32)
cargo run --release --features rocm --example spike041_vector_subtract_ab   # + cubecl-hip (f32)
```
Source: `crates/lgbm-compute/examples/spike041_vector_subtract_ab.rs`. Bench discipline
(CONVENTIONS 017–019/040): `LAUNCHES=50` overwrite-launches into one reused `out` + a
single `read_one_unchecked` to force sync (subtract OVERWRITES → re-launch idempotent,
the autotune-safe class per spike-038), scalar/vector interleaved per rep, median of
`REPS=11`, **judge the sign** (the spoofed APU confounds absolute time), 2 process restarts.

## Results

**VERDICT: VALIDATED.** `Vector<P,N>` compiles, runs on hip + cpu, is **bit-exact on
every cell** (parity kill-gate PASSES by construction), and is **faster at width > 1 on
both backends**. The `vs=1` control is ~1.0× everywhere (harness sanity: width-1 = scalar).

### cubecl-cpu (n = cells; median speedup vec/scalar)

| n (cells) | f64 best | f32 best | scaling |
|---|---|---|---|
| 25,600 (50feat×256bin×2) | vs8 **1.07–1.17×** | vs16 **1.20–1.22×** | small op, overhead-bound |
| 256,000 (500feat×256bin×2) | vs8 **2.55–2.57×** | vs16 **3.53–3.68×** | monotone in width; the real win |

### cubecl-hip (gfx1100/gfx1152 APU, f32 only — hip has no f64; 2 restarts)

| n (cells) | vec4 (max width) | vec2 | vs1 control |
|---|---|---|---|
| 25,600 | **1.25–1.29×** (sign-stable) | 0.96–1.09× (noisy) | 0.93–0.99× |
| 256,000 | **1.06–1.19×** (sign-stable) | 1.08–1.09× | 0.95–1.02× |

hip caps at `io_optimized_vector_sizes(f32) = [4,2,1]` (vec4 max = 128-bit load / 32-bit
f32), so the ceiling is lower than cpu's vec16; the win is sign-positive but modest and
magnitude-noisy on the spoofed 8-CU APU (subtract is a tiny, shared-DDR5-bandwidth-bound op).

### Investigation Trail — where the demonstrated win actually lands in production

Traced the live dispatch before claiming a wire:

- **CPU production subtract = NATIVE.** `CpuBackend::subtract_histograms` →
  `subtract_histograms_cpu_native` (`lib.rs:1343`), NOT the cubecl `subtract_hist_kernel`.
  ⇒ the cubecl-cpu 2.5–3.7× vectorization win **does not touch the merge-gate CPU anchor**
  (consistent with [[unified-cpu-gpu-kernels-pref]]: "cubecl-cpu lost to native, CPU stays
  native"). The CPU win is real for the *kernel* but the production CPU path doesn't run it.
- **rocm hot subtract = `subtract_resident`** (on-device, `lib.rs:2563`) — a **different
  kernel** than the `subtract_hist_kernel` benched here. Vectorizing the rocm hot path means
  vectorizing `subtract_resident`, not this kernel.
- The benched `subtract_hist_kernel` (via `subtract_histograms_f64_on`) is live only for
  **portable backends (cuda/wgpu)** + the generic f64 reference path (`lib.rs:2090/2300`).

So the kill question PASSES strongly, but the *demonstrated* win sits on a kernel whose
production hot-path role is bounded. The real production targets are `subtract_resident`
(rocm) and — more importantly — the **scan (042)** and **build (043)** kernels, which is
exactly the rest of this spike series.

### Disposition

- **Feasibility + parity: VALIDATED, reusable.** The `Vector<P,N>` launch recipe above is
  the foundation for 042/043. `Array<Vector<F,N>>` is bit-exact for any element-wise op.
- **Wire = deferred to the consolidated decision after 042/043** (wire the cleanest
  *production-hot-path* winner). Wiring `subtract_hist_kernel` now would help only portable
  backends for bounded benefit; the disciplined call is to first learn whether the scan/build
  (the actual rocm hot kernels) vectorize, then wire the best of the three. Matches the
  017/020/024 "validated-but-ROI-gate-the-wire" pattern. The spoofed APU loses to the CPU
  overall ⇒ this is ROCm-parity-track / discrete-gfx110x payoff.

### Surprises

- The win **scales with problem size**, not just width: at 25.6k cells cpu-f32-vec16 is
  only ~1.2× but at 256k it's ~3.7×. The small histogram is launch/overhead-bound; the
  large one exposes the genuine memory-throughput win. ⇒ 042/043 must bench at the WIDE
  (500-feature) shape to see the real signal, not the small one.
- `vs=2` on hip is magnitude-noisy (0.96–1.09×) — only the **max width (vec4)** gives a
  sign-stable hip win. Carry forward: on hip, sweep to the max `io_optimized` width.
