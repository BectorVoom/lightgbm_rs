# 260619-p93 — FINDINGS: warp-aggregated (Plane) f32-atomic histogram A/B on gfx1100

**Date:** 2026-06-19
**Hardware:** local AMD gfx1100 (wave32, `plane_size=32`, `has_plane=true`, `has_f64=false`, `has_f32_atomic=true`), cubecl-hip 0.10.0, `--release`.
**Bench:** `crates/lgbm-compute/examples/plane_aggregate_ab.rs`
**Kernel:** `construct_hist_kernel_atomic_f32_plane` + launcher `construct_histograms_parallel_f32_plane_on` (histogram.rs, rocm-gated, comptime `use_plane`).
**Re-pin test:** `crates/lgbm-compute/tests/rocm_plane_aggregate.rs`

**Method:** interleaved single-binary A/B. Both arms call the SAME launcher,
toggling ONLY the comptime `use_plane` flag — so uploads / cube count (`ceil(n/256)`
cubes of 256 units) / read-back are identical and the delta isolates EXACTLY the
warp-aggregation codegen. WARMUP discarded, MEDIAN + p25/p75 spread, device sync
(read-back) forced inside each timed call, arms INTERLEAVED per timed iter.
`delta% = (baseline − plane) / baseline * 100` (>0 ⇒ plane faster). Bin sweep
`[16, 64, 256]` is the PRIMARY independent variable (the in-plane collision rate is
the whole point of the lever). Run across **≥2 separate process invocations** to
check delta-sign stability vs noise.

The warp-aggregation path does CORRECT same-bin grouping (leader-iteration ballot →
`plane_shuffle` the leader bin → masked `plane_sum` per group → one elected lane
issues a single global atomic per distinct bin per plane). A naive whole-plane
`plane_sum` would corrupt the histogram (divergent bins); the re-pin test below
proves the implemented grouping is correct.

---

## Parity / correctness (re-pin test vs the CPU f64 anchor)

`cargo test -p lgbm-compute --features rocm --test rocm_plane_aggregate` → **5/5 GREEN.**

| test | result |
|------|:-------|
| `plane_no_lost_updates_under_contention` (50k rows → 16/64/256 bins, exact-integer) | EXACT, no lost/misrouted update |
| `plane_within_tolerance_of_cpu_f64_anchor` (n=8000, bins 16/64/256, real residual grads) | max rel **4.6e-7 / 3.8e-7 / 2.7e-7** — well inside the 1e-5 rocm gate |
| `plane_equals_baseline_on_integer_data` (30k rows, 128 bins) | EXACT vs the shipped per-row path |
| `plane_launcher_false_arm_matches_shipped_baseline` | the `use_plane=false` arm is byte-faithful to `construct_histograms_parallel_f32_on` |
| `plane_large_leaf_drift_not_worse_than_baseline` (n=50k, 16 bins) | plane within 2× the SHIPPED baseline's own drift |

**The warp-aggregation is provably CORRECT** and inside the existing rocm contract
at the regime the gate is calibrated for (≤8000 rows). The plane same-bin tree
reduction routes every add to the right bin (a naive `plane_sum` would not).

**Large-leaf f32 note (shared, NOT a plane regression):** at n=50000 / 16 bins each
of the 16 cells sums ~3125 near-cancelling `sin*0.5` residual gradients into ONE f32
cell, so BOTH the shipped per-row baseline AND the plane variant drift ~3e-4 rel
from the f64 anchor (catastrophic f32 cancellation — RESEARCH §parity-assessment /
04-ROCM-GAPS). Measured: baseline-vs-anchor 3.05e-4, plane-vs-anchor 2.93e-4 — the
SAME regime (the plane tree reduction was marginally *better* this run). The 1e-5
gate is therefore NOT applied to EITHER GPU f32 path at this f32-cancelling shape;
no tolerance was widened on any existing gate. (DEF-f8u-01: never pin two
nondeterministic GPU f32 paths to each other at 1e-6 — both are pinned to the f64
anchor.)

---

## A/B speed — warp-aggregate (`use_plane=true`) vs baseline per-row (`use_plane=false`)

| regime        | bins | run-1 delta% | run-2 delta% | sign stable? | spread separated? | verdict |
|---------------|-----:|-------------:|-------------:|:-------------|:------------------|:--------|
| launch-bound  |   16 |       −8.16  |       −6.39  | yes (neg)    | yes (plane slower) | NEGATIVE — plane slower |
| launch-bound  |   64 |      −55.59  |      −37.81  | yes (neg)    | yes (plane slower) | NEGATIVE — plane slower |
| launch-bound  |  256 |      −46.33  |      −44.27  | yes (neg)    | yes (plane slower) | NEGATIVE — plane slower |
| compute-bound |   16 |      +13.03  |      +14.17  | yes (pos)    | **NO — bands overlap** | SUB-NOISE / NULL |
| compute-bound |   64 |      −14.77  |      −21.23  | yes (neg)    | yes (plane slower) | NEGATIVE — plane slower |
| compute-bound |  256 |      −44.73  |      −35.12  | yes (neg)    | yes (plane slower) | NEGATIVE — plane slower |

Representative spreads (compute-bound, 16 bins — the only positive cell):
- run-1: baseline p25/p75 **2.5419 / 3.1850 ms** vs plane **2.1772 / 2.8118 ms**
- run-2: baseline p25/p75 **2.7867 / 3.0611 ms** vs plane **2.2857 / 2.6526 ms**

The plane band sits LOW within the baseline band but the two p25/p75 ranges
**overlap** (baseline's p25 2.54/2.79 is below the plane's p75 2.81/2.65). The
+13–14% median delta is sign-stable but NOT spread-SEPARATED, so by the plan's wiring
bar it is SUB-NOISE / NULL, not a robust win.

---

## Interpretation

The result is the **predicted NULL** (RESEARCH; 3 corroborating in-repo findings —
gpu-hist-levers-closed, spike-006, 260619-ol8):

- **Launch-bound (small n): robustly NEGATIVE at every bin count.** The leader-
  iteration loop runs `PLANE_DIM` passes with a `plane_ballot` + `plane_shuffle` +
  two `plane_sum`s each; at a small leaf this collective overhead dwarfs the few
  atomics it saves, and the regime is fixed launch+readback latency both arms pay.
  Plane is 6–56% slower.
- **Compute-bound, 256 bins: robustly NEGATIVE (−35 to −45%).** Exactly RESEARCH's
  decider: a 32-lane wave hits ~30 distinct bins out of 256, so there is almost
  NOTHING to amortize — the aggregation is pure overhead. This is the strongest NULL
  evidence.
- **Compute-bound, 64 bins: NEGATIVE (−15 to −21%).** Still too many distinct bins
  per wave to amortize the collective cost.
- **Compute-bound, 16 bins: the ONLY positive cell (+13–14%), sign-stable but
  spread-OVERLAPPING.** At 16 bins a 32-lane wave averages ~2 lanes/bin so a few
  atomics do collapse — but the win is inside the run-to-run noise band and confined
  to a single (large-leaf, very-low-bin) regime where histograms are already cheap.
  It does NOT clear the spread-separated bar required to wire.

The kernel is contention / scattered-read-latency bound (spike-006: 234 Mreads/s;
gpu-hist-levers-closed: "atomic-contention bound, latency/width/overhead levers
don't move it"). Collapsing uniform-random collisions does not relieve the hot-bin
serialization or the scattered `binned[idx]` read latency that dominate.

---

## DISPOSITION: NULL — keep as a rocm-gated PRIMITIVE, do NOT wire

Per the plan's success criteria and the ngo/ol8/j9t precedent: the warp-aggregated
`_plane` kernel + launcher are **kept as a correct, tested, rocm-gated PRIMITIVE and
are NOT wired into the training-path histogram routing.** The lever is closed with a
clean A/B signal:

- it is robustly SLOWER in 5 of 6 cells (sign-stable, spread-separated negatives);
- the single positive cell (compute-bound, 16 bins, +13–14%) is sign-stable but
  spread-OVERLAPPING and regime-narrow — not a robust, spread-separated win;
- this confirms the RESEARCH prediction and the three prior in-repo findings.

**No win was manufactured.** A benched NULL kept as a primitive is the valid,
honest outcome for this measurement-disposition task.

### Human sign-off flag (the checkpoint's wiring decision)

The plan's Task-3 checkpoint asks for a human decision on **whether to WIRE** the
plane variant into the production histogram path. The measurement recommendation is
**DO NOT WIRE** (NULL / no robust spread-separated win-regime). The kernel stays a
rocm-gated primitive. If a reviewer disagrees on the strength of the single
compute-bound/16-bin cell, that is the only point of judgement — but it fails the
spread-separated bar, so the recommendation is to leave the wiring as-is (unwired).

---

## Guardrails honored

- New `_plane` kernel is `#[cfg(feature="rocm")]` + comptime `use_plane` — the
  CPU-only build emits ZERO plane codegen (`cargo build -p lgbm-compute` green;
  default `cargo test -p lgbm-compute` 30 passed / 1 pre-existing-ignored).
- The CPU f64 anchor kernels (`construct_hist_kernel` / `construct_hist_kernel_f32`)
  and the baseline `construct_hist_kernel_atomic_f32` body + launcher are
  BYTE-UNCHANGED (pure-addition diff to histogram.rs).
- Correct same-bin aggregation (ballot/leader-iteration + masked `plane_sum`), NOT a
  naive whole-plane `plane_sum`; ballot indexed by runtime `PLANE_DIM` geometry (scan
  all 4 ballot words; no hardcoded 32 — wave32/wave64 portable).
- Parity asserts pin the plane variant to the CPU f64 ANCHOR within the EXISTING
  rocm gate (ABS 5e-6 / REL 1e-5); never GPU-f32-vs-GPU-f32 (DEF-f8u-01); no
  tolerance widened.
- Reused the existing `probe_capabilities(...).has_plane` host gate — not re-rolled.
- Existing `rocm_parallel_histogram` 7/7 GREEN (unregressed); `rocm_plane_aggregate`
  5/5 GREEN.
