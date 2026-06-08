---
quick_id: 260608-jyl
slug: r2-native-cpu-backend
status: complete
date: 2026-06-08
---

# Quick Task 260608-jyl — R2: native CPU backend (SUMMARY)

## Headline

Replaced the four single-unit cubecl-cpu hot-path kernels with native Rust f64
loops. **10× faster on large, 44× on small; lightgbm_rs went from ~40–80× slower
than C++ LightGBM 4.6 to ~2–4×** (small within 1.9×). Bit-exact gate GREEN
throughout — zero numeric change.

| size | M0 baseline | after R2 | speedup | vs C++ 4.6 (1-thread) |
|------|----|----|----|----|
| small  | 1.71s | 38.7ms  | **44×** | 1.9× slower |
| medium | 4.75s | 258ms   | **18×** | 3.4× slower |
| large  | 8.93s | 887ms   | **10×** | 4.3× slower |

## Root cause (probe-proven)

Every CPU-anchor op (`construct_histograms`, `find_best_split`, `subtract`,
`data_partition`) is a `CubeDim::new_1d(1)` SINGLE-UNIT sequential kernel. The
cubecl-cpu launch around it costs a fixed ~20–50µs/call (probe: 210× the native
loop at R=300) and runs per-(feature,leaf). The launch dispatch, not the
arithmetic, was the ~8s.

## What changed (all inside lgbm-compute — CMP-01 containment)

Per-op native twin + reroute `CpuBackend`:
- `construct_histograms_cpu_native` — f64 fold (T1, −32%).
- `find_best_split_cpu_native` — faithful REVERSE+FORWARD scan: same host pre-step
  (2·kEpsilon bump, min_gain_shift), gate ORDER, eps placements, operand orders,
  decode+accept-gate; `select(c,a,b)` → `if` (a no-op since `+0.0`, gain primitives
  pure) (T2, −75% — the big one).
- `subtract_histograms_cpu_native` (element-wise) + `data_partition_cpu_native`
  (integer routing + stable two-pass gather) (T3, −37% large / −71% small).

cubecl `*_cpu` paths **retained** for the kernel-parity / ROCm-mirror tests; the
hip f32/u32 paths are **untouched**. The architecture is now "native f64 anchor for
CPU, cubecl for GPU (ROCm)" — the numerical contract (bit-exact f64 ordered fold)
is unchanged.

## Parity gate — GREEN after every task

`cargo test -p oracle-harness`: kernel_parity 4, learner_parity 29, boosting_parity
75, predict_parity 5, raw_bin 2, rng 1 — all bit-exact, 0 failed. Units: lgbm 41,
compute 18, treelearner 64. **Zero numeric change.**

## Remaining gap (~2–4×) — next levers

The gap now grows with row count (small 1.9× → large 4.3×), pointing at the
per-feature row GATHER + single-thread:
- **R4 rayon over features** for histogram construction (C++ parallelizes here; the
  per-feature fold is independent — keep the within-feature ordered f64 fold). Now
  likely the biggest remaining lever.
- **R3 columnar storage + subtraction reuse** to cut the `Vec<Vec<f64>>` gather.
- R1's per_bin_gains skip (260608-jpj) now matters slightly more in relative terms.
