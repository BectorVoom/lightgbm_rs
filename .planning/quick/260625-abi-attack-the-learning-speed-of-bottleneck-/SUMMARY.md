---
quick_id: 260625-abi
slug: attack-the-learning-speed-of-bottleneck
date: 2026-06-25
status: complete
result: SHIPPED — GPU scan-occupancy lever (feature-per-lane), bit-exact, isolated scan ~3×
---

# Attack the GPU learning-speed bottleneck — SUMMARY

## Outcome

Shipped a **bit-exact** GPU scan-occupancy lever (spike-021): the fused per-leaf split
scan was launched as one **single-threaded** cube per feature (`CubeDim(1)`, ~1/32 wave
ALU utilization). Repacked to **one feature per lane** (`CubeDim(W)`, default **W=64** on
rocm via `LGBM_SCAN_CUBEDIM`). Each feature's scan stays sequential ⇒ bit-identical for
every W; W=1 is byte-identical to the original.

## How we got there (evidence-first, as chosen)

1. **Grounding** — established the obvious target (histogram BUILD) is already heavily
   optimized: u64 fixed-point shipped (spike-018/19), all build micro-levers dead
   (006/009/008/017/020). So "attack the build" was the wrong frame.
2. **Re-profile (T1/T2)** — the last attribution (spike-014/15) predated the u64 win.
   `LGBM_PHASE_PROF` + `LGBM_SCAN_PROF` + `LGBM_SCAN_DRAIN` A/B showed: the per-leaf
   "scan" (~85% of train) splits **~46% build / ~54% genuine scan**; the spike-015
   constant-array-hoist candidate is **dead (0.2%)**. The genuine scan is a `CubeDim(1)`
   single-threaded-per-feature kernel — the real target.
3. **Attack (T3)** — feature-per-lane repack. Isolated scan launch+readback (build
   drained), 250k×500: **W=1 11.8s → W=64 3.99s (2.96×) → W=128 3.33s (3.54×)**,
   monotonic. Default W=64 (robust knee).
4. **Measure + gate (T4)** — end-to-end median **~1.27×** (APU-noisy; one cold rep
   spuriously showed 3.8× — discarded); `phase_prof` learner −20%, total −8%. Parity:
   `kernel_parity --features rocm` **16/16 pass** (hip split within ~1e-6 at W=64;
   cubecl-cpu fused==per-feature==native at W=1).

## Honest framing (ROI)

The cold isolated ceiling (3×) overstates the warm end-to-end (~1.2–1.3×) because the
per-leaf readback sync is also gated by the unchanged build (Amdahl). The GPU is a
spoofed 8-CU APU that loses to the multi-threaded CPU anchor at every shape — this is
**ROCm-parity-track maintenance**, not an overall-fastest win. The removed
under-utilization is even more wasteful on real discrete gfx110x (more idle lanes at
`CubeDim(1)`), where the end-to-end share should be larger.

## Files

- `crates/lgbm-compute/src/kernels/split.rs` — `scan_cube_dim()` (env, rocm default 64),
  `find_best_splits_fused_kernel` (`ABSOLUTE_POS` index + `n_feats` tail guard),
  `find_best_splits_fused_inner` (`CubeDim(W)`, `CubeCount=ceil(n/W)`).
- `.planning/spikes/021-scan-feature-per-lane-occupancy/README.md` — full spike record.

## Gate

- `cargo test -p oracle-harness --features rocm --test kernel_parity` → 16/16 ok.
- `cargo test -p lgbm-treelearner --lib`, `-p lgbm-boosting --lib` → regression (CPU path
  unaffected; non-rocm pins W=1).
- CPU f64 anchor untouched.
</content>
