---
phase: 260619-ngo
plan: 01
subsystem: lgbm-compute (GPU histogram kernels)
tags: [rocm, gpu, histogram, benchmark, cuda-mirror, lds, measurement-only]
requires: [construct_histograms_cuda_mirror_resident_on, build_leaf_histograms_resident_f32_on]
provides: [mirror_vs_lds_ab_bench, wiring_recommendation]
affects: []
key-files:
  created:
    - crates/lgbm-compute/examples/mirror_vs_lds.rs
  modified: []
decisions:
  - "WIRING RECOMMENDATION: LEAVE THE CUDA-MIRROR RESIDENT KERNEL AS A PRIMITIVE (do not wire as default, do not gate-opt-in) — it is never meaningfully faster than the wired LDS path and is clearly slower at the mid/16 cell; no win regime exists."
metrics:
  duration: ~12 min
  completed: 2026-06-19
  tasks: 2
  files: 1
---

# Phase 260619 Plan ngo: Benchmark CUDA-mirror resident histogram vs wired LDS Summary

A rocm-gated A/B micro-bench (`crates/lgbm-compute/examples/mirror_vs_lds.rs`) measured the
CUDA-mirror resident histogram kernel (`construct_histograms_cuda_mirror_resident_on`)
against the production-wired LDS resident build (`build_leaf_histograms_resident_f32_on`) on
the local gfx1100 — same resident bins, same leaf rows, same sentinel-free `slot_off`,
resident upload excluded. **Verdict: the mirror is never faster and is clearly slower at the
mid/16 cell — leave it as a primitive.**

## What was measured

- FEATS=50, NUM_DATA=200_000, BIN_SWEEP={16,64,256}, two leaves: LARGE (all 200k rows) and
  MID (~50k rows, every 4th).
- WARMUP=3 discarded, median of TIMED=7 per variant, device sync (result read-back) inside
  each timed call. Run from 3 separate process invocations for drift.
- Three numbers per cell: `mirror_ms`, `lds_incl_ms` (the LDS launcher's INTERNAL host
  ord_g/ord_h gather INCLUDED — production's real per-leaf cost, learner.rs:1767),
  `lds_excl_ms` (= `lds_incl − gather_ms`, kernel + per-leaf uploads only). `speedup` =
  `lds_incl/mirror`, `speedup_kernel` = `lds_excl/mirror` (>1.0 ⇒ mirror faster).
- Same-input sanity assert (ABS 5e-6 / REL 1e-5, the f32-atomic envelope) pinned the two RAW
  histograms to each other once per cell — **passed for every cell on every run** (no panic),
  so the timing comparison is between correct, equivalent computations.

## Real gfx1100 A/B table

Run 3 (representative — medians stable across all three runs; see drift note):

```
     leaf |  bins |   mirror_ms | lds_incl_ms | lds_excl_ms | gather_ms |   speedup | speedup_kernel
----------+-------+-------------+-------------+-------------+-----------+-----------+---------------
    large |    16 |       6.253 |       6.774 |       6.643 |     0.130 |     1.08x |         1.06x
  mid~50k |    16 |       2.363 |       1.700 |       1.653 |     0.047 |     0.72x |         0.70x
    large |    64 |       6.169 |       6.247 |       6.117 |     0.130 |     1.01x |         0.99x
  mid~50k |    64 |       1.868 |       1.607 |       1.574 |     0.033 |     0.86x |         0.84x
    large |   256 |       6.078 |       6.223 |       6.091 |     0.132 |     1.02x |         1.00x
  mid~50k |   256 |       1.930 |       1.911 |       1.876 |     0.034 |     0.99x |         0.97x
```

All three runs (for cross-checking drift; mirror_ms / lds_incl_ms / speedup):

| cell | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| large/16  | 6.18 / 6.36 / 1.03x | 6.11 / 5.95 / 0.97x | 6.25 / 6.77 / 1.08x |
| mid/16    | 4.19 / 1.82 / **0.44x** | 3.67 / 1.62 / **0.44x** | 2.36 / 1.70 / **0.72x** |
| large/64  | 6.37 / 6.46 / 1.01x | 6.20 / 6.69 / 1.08x | 6.17 / 6.25 / 1.01x |
| mid/64    | 1.83 / 1.77 / 0.97x | 1.87 / 1.71 / 0.92x | 1.87 / 1.61 / 0.86x |
| large/256 | 6.20 / 6.17 / 1.00x | 6.46 / 6.36 / 0.99x | 6.08 / 6.22 / 1.02x |
| mid/256   | 1.84 / 1.84 / 1.00x | 1.96 / 1.89 / 0.97x | 1.93 / 1.91 / 0.99x |

**Drift note:** the `mid/16` mirror time drifts most (4.19 → 3.67 → 2.36 ms across the three
process restarts — a warmup/allocator settling effect on the first-built leaf), but in every
run the mirror is the LOSER there (0.44–0.72x). All other cells are stable to ~±5% and the
LARGE leaf converges to ~1.0x every run.

## Findings (honest read, honesty mandate)

1. **LARGE leaf (the GPU-relevant regime): mirror ≈ LDS, within noise.** Across all bins the
   speedup is 0.97–1.08x — a dead heat. At the full-corpus leaf the two kernels are
   statistically tied; the mirror shows no advantage.

2. **MID leaf: mirror is slower, decisively at 16 bins.** `mid/16` speedup is 0.44–0.72x
   every run (mirror 2.4–4.2ms vs LDS 1.6–1.8ms). `mid/64` is 0.86–0.97x; `mid/256` ~0.99x.
   The mirror's row-partition + in-kernel indirect gather design pays a penalty at smaller
   leaves with few bins, where the LDS one-cube-per-feature path is more efficient.

3. **The host gather is NOT the LDS path's weakness.** `gather_ms` is 0.033–0.139ms — i.e.
   `lds_incl ≈ lds_excl` (the gather is <2% of the LDS time at large, ~2% at mid). The
   `speedup_kernel` (kernel-only) tracks `speedup` almost exactly (0.70–1.06x). So there is
   no host-gather tax for the mirror's in-kernel gather to win back — the comparison is
   essentially kernel-vs-kernel, and the LDS kernel is at-worst-tied, often faster. This
   directly answers the plan's separation question: the LDS path's real cost is the kernel
   compute, not the host gather; removing the gather would change nothing material.

4. **Same-input correctness held everywhere** (ABS 5e-6 / REL 1e-5) — the mirror and LDS RAW
   histograms agree within the f32-atomic envelope on all 6 cells across all 3 runs, so the
   A/B compared correct, equivalent computations. (The real parity gate remains the CPU f64
   anchor, covered by the existing `rocm_cuda_mirror.rs` tests; this bench does not re-assert
   it.)

This is consistent with the project's standing GPU finding (SKILL / memory `gpu-hist-levers-closed`,
260619-j9t/mwr): the faithful CUDA mirror is parity-not-speed; the LDS resident path (260609-fw1)
is the already-optimized production kernel.

## WIRING RECOMMENDATION — LEAVE AS PRIMITIVE

**Recommendation: LEAVE THE CUDA-MIRROR RESIDENT KERNEL AS A ROCM-GATED PRIMITIVE.** Do NOT
wire it as the default; do NOT add a gated opt-in path.

Evidence, stated plainly per the honesty mandate (this is a negative result and is reported as
one):

- **Not faster anywhere.** At the LARGE leaf (the regime where GPU histogram build matters) the
  mirror is tied with the wired LDS path (0.97–1.08x, noise). It wins no cell decisively in any
  run.
- **Slower where it differs.** At MID/16 the mirror is 0.44–0.72x — a real, reproducible loss.
  MID/64 is 0.86–0.97x. The only direction the data points is "mirror ≤ LDS".
- **No win-regime exists**, so the user's rule resolves cleanly: faster → default (N/A);
  slower/mixed with a clear win-regime → gated (N/A, there is no win-regime); slower/mixed with
  NO win-regime → **leave as primitive**. This is that last case.
- **The host gather is negligible** (≤0.14ms), so there is no "remove the LDS host-gather tax
  later" lever that would flip the verdict; the LDS kernel itself is the at-worst-equal winner.

The mirror stays valuable as the faithful CUDA-reference primitive (parity/maintenance,
documented residual vs the CPU f64 anchor) — just not as a production build kernel.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Corrected the LDS launcher calling convention**
- **Found during:** Task 2 (first run on gfx1100).
- **Issue:** The plan's `<interface_contract>` stated the LDS launcher consumes caller-
  pre-gathered LEAF-LENGTH `ord_g`/`ord_h` (`ord_g[k]=grad[leaf_rows[k]]`). The real
  `build_leaf_histograms_resident_f32_on` takes FULL-CORPUS grad/hess and does that gather
  INTERNALLY (`resident_raw_build_into`, histogram.rs:1384). Passing a leaf-length array
  indexed out of bounds (`len 50000, index 50000`) and panicked on the MID leaf.
- **Fix:** Feed both launchers full-corpus grad/hess (matching the wired learner.rs:1767).
  Since the host gather is internal and cannot be stripped without editing the kernel
  (forbidden), `lds_excl` is derived as `lds_incl − gather_ms`, with `gather_ms` timed
  separately and printed for transparency. This honors the plan's INTENT (production cost +
  a kernel-only number) faithfully.
- **Files modified:** `crates/lgbm-compute/examples/mirror_vs_lds.rs` (header DEVIATION note
  + body).
- **Commit:** 6ea8ddf
- **Impact on result:** none adverse — the gather turned out to be ≤0.14ms, so `lds_incl ≈
  lds_excl` and the verdict is robust to how the gather is attributed.

No kernel, lib.rs, learner, or CPU f64 anchor was modified. Both launchers were reused as-is.

## Follow-up (deliberately deferred — NOT done here)

Wiring is the explicitly-deferred follow-up the user chose ("benchmark first, then decide").
This task delivered the numbers + the recommendation only; per the recommendation above, the
follow-up is a no-op (leave the primitive as-is) unless a future kernel redesign changes the
mirror's cost profile.

## Commits

- `6b9dcbd` feat(260619-ngo): add rocm-gated mirror-vs-LDS resident histogram A/B bench
- `6ea8ddf` fix(260619-ngo): correct LDS launcher calling convention in mirror-vs-LDS bench

## Self-Check: PASSED

- `crates/lgbm-compute/examples/mirror_vs_lds.rs` exists and is committed.
- Commits `6b9dcbd` and `6ea8ddf` exist on master.
- No reference tree (LightGBM/, LightGBM-release-4.6.0.99/, cuml-main/) was git-added.
- Working tree clean except the (orchestrator-owned) SUMMARY.md and untracked reference trees.
