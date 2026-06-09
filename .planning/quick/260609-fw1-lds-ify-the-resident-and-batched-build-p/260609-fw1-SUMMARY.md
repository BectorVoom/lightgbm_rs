---
quick_id: 260609-fw1
title: LDS-ify the resident/batched build hot path + wire live
date: 2026-06-09
status: complete
type: implementation + test-fix + benchmark
parity_class: neutral (f32 ~1e-6 ROCm path; cpu anchor untouched)
commits: d82611b (gate fix), b878eb5 (LDS build)
---

# Quick Task 260609-fw1 — Summary

LDS-privatized the GPU histogram **build hot path** (resident + batched) and **wired it
live** into training. Per the user's chosen sequencing, fixed the flaky gate FIRST so
wiring could be cleanly verified.

## Part 1 — DEF-f8u-01 gate fix (d82611b)

`learner_parity_{resident,fused}_equals_host_tree_on_hip` was flaky (~4/6) — two
nondeterministic GPU f32-atomic trees compared to each other at a 1e-6 absolute
leaf-value tol (below the f32 leaf-accumulation noise floor; mutual diff to 3.1e-6).
**Fix:** pin both GPU trees to the deterministic cpu f64 anchor — structure BIT-EXACT
(verified stable: GPU structure == CPU structure), leaf values within
`ROCM_LEAF_VALUE_TOL=1e-5` (the `sqrt(R)·ε·mean|g|` f32 envelope; observed ~1.75e-6, 6×
headroom). Histogram-cell ~1e-6 contract unchanged. **12/12 green (was ~6/12).**

## Part 2 — LDS build kernels + wiring (b878eb5)

`construct_leaf_hist_{resident,batched}_lds_kernel` — ONE CUBE PER FEATURE, each owning
a ≤2 KiB LDS sub-histogram (per-feature because the concatenated multi-feature output
exceeds one cube's LDS), merged into its global slot once per cell. Per-feature global
atomic traffic `2*R` → `2*num_bin[f]`. Shared `resident_raw_build_into` helper
(LDS-or-naive, ≤256-bin gate) routes BOTH the resident-pool chain
(`build_fix_compact_resident_f64_on`) and the host path
(`build_leaf_histograms_resident_f32_on`); the batched launcher gets the same branch.
**Wired live** (unlike f8u) because the fixed gate verifies non-regression.

## Benchmark (gfx1100, 20k-row leaf, LDS vs naive build)

| feats | bins | speedup |
|---|---|---|
| 50 | 16 | **5.1×** |
| 50 | 64 | **7.4×** |
| 50 | 256 | **3.5×** |
| 20 | 16 | **9.0×** |

3.5–9× faster on the actual hot path.

## Gate (GREEN)

Default merge gate 0-failed (lgbm 41 / python 55 / compute 18 / treelearner 65 /
boosting 75 / learner_parity 29 / kernel_parity 6); hip kernel_parity 15/15 (build
oracles now LDS); hip learner_parity 31/31 end-to-end LDS-built (10/10 stable); clippy
clean.

## Files modified

- `crates/oracle-harness/tests/learner_parity.rs` (DEF-f8u-01 fix — anchor comparison)
- `crates/lgbm-compute/src/kernels/histogram.rs` (LDS build kernels + helper + wiring)

## Follow-ups

- Optionally wire `RocmBackend::construct_histograms` (single-feature) to the f8u LDS
  kernel now that the gate is reliable (smaller, non-hot-path).
- End-to-end GPU train wall-clock vs C++ (the build kernels are now fast; measure where
  the GPU train bottleneck moves next).
