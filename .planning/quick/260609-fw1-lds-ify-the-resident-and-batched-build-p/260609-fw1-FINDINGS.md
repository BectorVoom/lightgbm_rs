---
quick_id: 260609-fw1
type: implementation + test-fix + benchmark
title: LDS-ify the resident/batched build hot path (eo5 Finding #2 follow-up) + wire live
date: 2026-06-09
status: complete
hardware: gfx1100 (real ROCm), cubecl-hip 0.10.0
verdict: LDS build wired LIVE into the training hot path; 3.5–9× faster; DEF-f8u-01 gate fixed first to enable clean verification
commits: d82611b (gate fix), b878eb5 (LDS build)
---

# LDS-ify the resident/batched build path + wire live

User-chosen sequencing (AskUserQuestion): **fix the flaky gate first, then LDS-ify the
build path AND wire it live.** Both done.

## Part 1 — DEF-f8u-01 gate fix (commit d82611b)

`learner_parity_{resident,fused}_equals_host_tree_on_hip` compared two
independently-nondeterministic GPU f32-atomic trees to EACH OTHER at a **1e-6 absolute
leaf-value tol** → flaky ~4/6 runs (mutual leaf diff measured to 3.1e-6). Root cause:
the 1e-6 absolute bound is below the genuine f32 leaf-accumulation noise floor.

**Fix (principled, not a blanket weaken):** pin BOTH GPU trees to the **deterministic
cpu f64 anchor**. Empirically verified the enabling facts:
- GPU structure == CPU f64 structure is rock-stable (split_feature/threshold/
  decision_type/leaf_count all match across every probe run) → structural fields stay
  **BIT-EXACT** vs the anchor (the real tree-change detector).
- Single GPU path vs the f64 anchor reaches ~1.4–1.75e-6 on leaf values → leaf VALUES
  use `ROCM_LEAF_VALUE_TOL = 1e-5`, the f32 leaf-accumulation envelope
  (`~sqrt(R)·ε_f32·mean|g|`), ~6× headroom. The histogram-cell ~1e-6 GPU-vs-f64
  contract is UNCHANGED (kernel_parity oracles).

`resident/fused == host` follows transitively (both == anchor). **Verified 12/12 green
(was ~6/12).** This reliable gate is what makes wiring the build path verifiable.

## Part 2 — LDS build kernels + wiring (commit b878eb5)

`construct_leaf_hist_{resident,batched}_lds_kernel` (`histogram.rs`): **ONE CUBE PER
FEATURE**, each owning a private ≤2 KiB LDS sub-histogram (the concatenated
multi-feature output can't fit one cube's LDS, so per-feature is the design — mirrors
LightGBM's OpenCL `histogram*.cl` one-workgroup-per-feature). Units stride the leaf's
rows doing cheap LDS atomics, then merge into that feature's global slot. **Per-feature
global atomic traffic drops from `2*R` to `2*num_bin[f]`.** `slot_off` carries a
sentinel (= slot_len) so each cube reads its width `slot_off[f+1]-slot_off[f]`.

**Wiring (the key difference from f8u, which stayed unwired):** a shared
`resident_raw_build_into` helper picks LDS (every feature ≤256 bins) or the naive
kernel, and is used by BOTH `build_leaf_histograms_resident_f32_on` AND the
resident-pool chain `build_fix_compact_resident_f64_on` — so the resident-pool path and
the host path share ONE accumulation structure. `build_leaf_histograms_batched_f32_on`
gets the same LDS/naive branch. All of `RocmBackend::{build_leaf_histograms_raw,
build_resident_leaf}` now route through LDS automatically. >256-bin features fall back
to the naive kernel.

f32 atomics ⇒ same ~1e-6 ROCm gate; cpu f64 anchor untouched.

## Benchmark — LDS vs naive resident build (gfx1100, 20k-row leaf, --release)

| features | bins | naive | LDS | **speedup** |
|---|---|---|---|---|
| 50 | 16  | 11432 µs | 2223 µs | **5.1×** |
| 50 | 64  |  7550 µs | 1028 µs | **7.4×** |
| 50 | 256 |  3461 µs |  996 µs | **3.5×** |
| 20 | 16  |  4443 µs |  493 µs | **9.0×** |

LDS build is **3.5–9× faster** across the board on the actual training hot path. (Note:
LDS itself is slower at fewer bins — 16-bin LDS has higher *intra-cube* LDS-atomic
contention than 256-bin — but still crushes naive, whose *global*-atomic contention is
far worse at few bins.)

## Gate (GREEN)

- Default merge gate 0-failed: lgbm 41, python 55, compute 18, treelearner 65,
  boosting 75, learner_parity 29, kernel_parity 6.
- hip kernel_parity 15/15 — the resident/batched BUILD oracles
  (`kernel_parity_resident_build_fix_compact_equals_host_on_hip`,
  `kernel_parity_resident_gather_equals_host_gather_on_hip`) now exercise LDS.
- hip learner_parity 31/31 — resident/fused END-TO-END now LDS-built, pinned to the cpu
  f64 anchor; **10/10 stable** over repeated runs.
- rocm_parallel_histogram 7/7; clippy clean.

## Scope / status

- This LDS-ifies the RAW build (`construct_leaf_hist_*`). The f8u single-feature
  `construct_histograms_lds_f32_on` remains available; `RocmBackend::construct_histograms`
  (the simple single-feature path, not the batched/resident hot path) is still naive —
  it could now be wired too given the reliable gate, but the hot path is the
  batched/resident build done here.
- CPU native path untouched (the bit-exact merge gate + fast path).
