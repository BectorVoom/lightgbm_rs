---
status: complete
phase: quick-260608-mc5
plan: 01
subsystem: lgbm-compute (split-finding kernels)
tags: [kernel-merge, gpu-launch-collapse, cubecl, split-finding, perf, parity]
requires: [260608-lsx, 260608-lad]
provides:
  - "shared #[cube] split_scan_body — single source of the f64 split math"
  - "find_best_splits_fused_kernel + find_best_splits_batched_fused_f64_on (one launch per leaf)"
  - "three-way fused==per-feature==native bit-exact oracle gate"
affects: [RocmBackend split path (now 1 launch/leaf), CpuBackend (unchanged, native)]
tech-stack:
  added: []
  patterns: ["fused per-leaf cube launch (CubeCount::Static(num_feats,1,1))", "shared #[cube] helper with base-offset args"]
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/oracle-harness/tests/kernel_parity.rs
decisions:
  - "CpuBackend keeps the NATIVE per-feature split path (fused cubecl-cpu materially regresses CPU; CLAUDE.md non-neg #2) — GPU fused override + shared helper kept regardless"
metrics:
  duration: "~1h"
  completed: 2026-06-08
---

# Quick 260608-mc5: Fused Per-Leaf GPU Split Scan + Kernel Collapse Summary

One-liner: Merged the forked split-finding transcriptions into ONE shared `#[cube] split_scan_body` called by both the single-feature kernel and a NEW fused per-leaf batched kernel, collapsed the GPU `find_best_splits_batched` to ONE launch per leaf (3.4–3.6x faster on gfx1100), and — on measured evidence — kept CpuBackend on its native path to avoid a silent CPU slowdown.

## What shipped

### Task 1 — THE MERGE (step 1): shared scan helper (`8520519`)
- Extracted the REVERSE+FORWARD f64 scan body of `find_best_split_kernel` into a new shared `#[cube] split_scan_body(hist, hist_base, out, out_base, ...)`. Reads index `hist[hist_base + bi]`; finalization writes `out[out_base + 0..12]`.
- `find_best_split_kernel` is now a thin `#[cube(launch)]` wrapper calling the helper with bases `0, 0` — observably identical (kernel_parity stayed bit-exact). VERBATIM transcription: no f64 op reordered, same `select`/`done`/eps/threshold arithmetic, same literal inits.
- `find_best_split_kernel_f32` (the hip f32 mirror) left UNTOUCHED.

### Task 2 — THE MERGE + THE COLLAPSE (`b3b30f0`)
- `find_best_splits_fused_kernel`: ONE launch per leaf, `CubeCount::Static(num_feats,1,1)`, `CubeDim::new_1d(1)`; cube `f = CUBE_POS_X` calls `split_scan_body(hist, slot_off[f], out, f*12, ...)` — reads only `[slot_off[f], slot_off[f]+2*num_bin[f])`, writes only `out[f*12..f*12+12]` (mirrors `construct_leaf_hist_batched_kernel`).
- `find_best_splits_batched_fused_f64_on<R>` launcher: full host V5 BEFORE the single launch (per-feature `num_bin==0`/`2*num_bin` overflow/`slot_off+2*num_bin > buf.len()` → `LengthMismatch`/`na_as_missing`/non-default smoothing → typed error; leaf-level `!(sum_hessian>0)` once; empty `feats` → `Ok(vec![])` no launch). Leaf-level scalars (`sum_hessian + 2*kEpsilon`, `min_gain_shift`) computed ONCE; per-feature device arrays uploaded; same accept-gate decode per 12-cell window, pushed in input order. All `unsafe` confined to the launcher (CMP-01).
- Both backends wired through it initially (the literal merge): `RocmBackend` and `CpuBackend`.

### Task 3 — oracle + measured CPU decision (`4575f66`)
- `kernel_parity_fused_equals_per_feature_and_native`: over the committed split fixture corpus, asserts (bit-exact f64 via `compare_exact_f64_bits` + exact int/flag) that for EVERY feature in input order `fused == per-feature(cubecl) == native`. The three-way merge gate — PASSES.
- Measured CPU before/after and made the honest call (below). The CpuBackend override was REVERTED to native; the GPU fused override + shared helper are kept.

## Performance measurements (REAL, bench_train.rs, release)

### CPU (cubecl-cpu) — measured on this HEAD
| size   | native (BEFORE, R2-equiv) | fused cubecl-cpu (AFTER) | ratio |
|--------|---------------------------|--------------------------|-------|
| small  | 42.86 ms                  | 223.92 ms                | ~5.2x slower |
| medium | 256.17 ms                 | 618.49 ms                | ~2.4x slower |
| large  | 828.95 ms                 | 1.76 s                   | ~2.1x slower |

The fused cubecl-cpu path MATERIALLY regresses CPU — the cubecl-cpu per-leaf launch dispatch dominates even batched into one launch per leaf (the same root cause R2 / 260608-jyl found: launch fixed cost, not arithmetic). Both measurements taken on this HEAD by toggling only the CpuBackend wiring (native trait default vs fused override). The native numbers match the STATE.md R2 baseline (38.7/258/887 ms), confirming the measurement is apples-to-apples.

### GPU (cubecl-hip, gfx1100) — the launch-count collapse
| size   | per-feature loop (BEFORE) | fused 1-launch/leaf (AFTER) | speedup |
|--------|---------------------------|-----------------------------|---------|
| small  | 4.95 s                    | 1.42 s                      | ~3.5x   |
| medium | 17.28 s                   | 5.10 s                      | ~3.4x   |
| large  | 43.75 s                   | 12.10 s                     | ~3.6x   |

The collapse is a clear GPU win — exactly the deferred per-leaf launch-count collapse from 260608-lsx.

## Decision (CLAUDE.md non-negotiable #2 — no silent CPU slowdown)

Took the plan's Task-3 decision branch: **CpuBackend keeps the NATIVE per-feature path** (the `Backend::find_best_splits_batched` trait default → `find_best_split_cpu_native`). The CpuBackend override is intentionally NOT defined (documented at the revert site in `lib.rs`). The GPU `RocmBackend` KEEPS the fused override (3.4–3.6x faster, f64 bit-exact on gfx1100). The shared `split_scan_body` helper (THE MERGE) stays for BOTH paths regardless — the split math is unified to a single source even though the cubecl-cpu launcher is not the production CPU path (it remains proven bit-exact via the oracle gate). Final CPU bench after the revert: 42.06 / 248.68 / 839.11 ms — back at baseline, no regression.

## Verification (REAL output)

- `cargo build --workspace` (cpu): PASS. `cargo build --workspace --features rocm`: PASS.
- `cargo test -p oracle-harness --test kernel_parity` (cpu): **6/6 GREEN** bit-exact (incl. new `kernel_parity_fused_equals_per_feature_and_native` + the existing `kernel_parity_batched_equals_per_feature_on_cpu`).
- `cargo test -p oracle-harness --test learner_parity` (cpu): **29/29 GREEN** bit-exact (tree growth, which calls `find_best_splits_batched`, unchanged).
- `cargo test -p oracle-harness --features rocm --test kernel_parity`: 9/10 pass. The 1 failure `hip::kernel_parity_split_within_tol_on_hip` is the PRE-EXISTING f32 D-03a tolerance gap (04-ROCM-GAPS.md) — **VERIFIED pre-existing**: checked out the three source files at base commit `c8ae5d2` and reproduced the identical failure (same abs_diff values, same `default_left` mismatch). OUT of scope per constraint #3; the f32 hip-split path is untouched by this task.
- bench_train CPU before/after + GPU before/after: captured above.
- clippy: my new fused code is clean; all remaining `lgbm-compute` warnings are pre-existing in untouched bin/distinct_values code and the untouched f32 hip path (split.rs:1527) / the pre-existing trait-default arg-count (lib.rs:268).

## Deviations from Plan

**1. [Rule 3 - Blocking] cubecl Array index type for `CUBE_POS_X`**
- Found during: Task 2 build.
- Issue: indexing per-feature `Array`s with the `u32` `CUBE_POS_X` directly failed cubecl lowering (`NativeExpand<u32>` vs `NativeExpand<usize>`).
- Fix: bind `let fi = f as usize;` and index `slot_off[fi]` etc. (`out_base = f * 12u32` stays u32). No numerics affected.
- Files: crates/lgbm-compute/src/kernels/split.rs. Commit: b3b30f0.

**2. [Task-3 decision branch — measured, not a defect] CpuBackend reverted to native**
- The plan explicitly authorized this branch if the fused cubecl-cpu path materially regresses CPU. It does (table above). Documented at the revert site and here. The MERGE (shared helper) and the GPU COLLAPSE both ship; only the CpuBackend production wiring stays native.

## Known Stubs
None.

## Threat Flags
None — no new network/auth/file-access surface; the fused launcher tightens the trust boundary (host V5 before a single launch; cube `f` reads only its validated region, T-mc5-01/02).

## Self-Check: PASSED
- crates/lgbm-compute/src/kernels/split.rs — FOUND (contains `split_scan_body`, `find_best_splits_fused_kernel`, `find_best_splits_batched_fused_f64_on`).
- crates/oracle-harness/tests/kernel_parity.rs — FOUND (contains `kernel_parity_fused_equals_per_feature_and_native`).
- Commits 8520519, b3b30f0, 4575f66 — all present in `git log`.
