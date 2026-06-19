# 260619-ol8 — FINDINGS: launch_unchecked vs launch (bounds-check codegen) A/B on gfx1100

**Date:** 2026-06-19
**Hardware:** local AMD gfx1100 (wave32), cubecl-hip 0.10.0, `--release`.
**Bench:** `crates/lgbm-compute/examples/launch_unchecked_ab.rs`
**Method:** dual-kernel single-binary interleaved A/B. For each of the 3 hot-loop
production kernels a bench-only `_checked` twin (`#[cube(launch)]`, byte-identical
body) is launched INTERLEAVED with the shipped `#[cube(launch_unchecked)]` kernel.
WARMUP discarded, median + p25/p75 spread, device-sync (read-back) inside each timed
call. `delta% = (checked - unchecked) / checked * 100` (>0 ⇒ unchecked faster). Run
across **2 separate process invocations** to check delta-sign stability vs noise.

`launch_unchecked` ONLY removes the in-kernel per-access bounds-check codegen;
numerics are unchanged. The same-input sanity asserts (f32-atomic envelope ABS
5e-6 / REL 1e-5 for the atomic + LDS kernels, BIT-EQUAL for the f64 deterministic
fused kernel) **passed for every cell in both runs** — confirming the twins are
still byte-faithful to the shipped kernels (no drift) and the comparison is between
equivalent computations. (This is a same-input check, NOT the real
GPU-vs-CPU-f64-anchor parity gate, which stays in `rocm_cuda_mirror.rs` /
`kernel_parity`.)

---

## Kernel 1 — f32-atomic histogram (`construct_hist_kernel_atomic_f32`)

| regime        | bins | run-1 delta% | run-2 delta% | sign stable? | verdict |
|---------------|-----:|-------------:|-------------:|:-------------|:--------|
| launch-bound  |   16 |       -1.62  |   (≈noise)   | no           | SUB-NOISE / NULL |
| launch-bound  |   64 |       -7.07  |   (≈noise)   | no           | SUB-NOISE / NULL |
| launch-bound  |  256 |       +0.25  |      -10.05  | **flips**    | SUB-NOISE / NULL |
| compute-bound |   16 |       -3.13  |       -7.37  | (neg, small) | SUB-NOISE / NULL |
| compute-bound |   64 |       -3.40  |       -1.25  | (neg, small) | SUB-NOISE / NULL |
| compute-bound |  256 |       +4.49  |       +2.87  | (small)      | SUB-NOISE / NULL |

Representative spreads (run-1): launch-bound 256 bins checked 0.0359/0.0446 ms vs
unchecked 0.0388/0.0421 ms — the medians (0.0403 vs 0.0402) sit well INSIDE each
other's p25/p75 band. Every atomic delta is within the arm spread and several flip
sign across the two runs.

**Disposition: NULL.** No measurable launch_unchecked benefit for the f32-atomic
kernel in either regime. The kernel is one cheap guarded scatter per row
(`if idx < len { 2 atomic fetch_add }`); the per-access bounds branch is negligible
against atomic-contention / memory latency, and the launch-bound cells are dominated
by fixed launch+readback latency that both arms pay equally.

---

## Kernel 2 — resident-LDS leaf histogram (`construct_leaf_hist_resident_lds_kernel`, P=1)

| regime        | bins | run-1 delta% | run-2 delta% | sign stable? | verdict |
|---------------|-----:|-------------:|-------------:|:-------------|:--------|
| launch-bound  |   16 |       +0.10  |       +1.68  | (≈0)         | SUB-NOISE / NULL |
| launch-bound  |   64 |       +6.31  |       +1.84  | pos, small   | SUB-NOISE / NULL |
| launch-bound  |  256 |       +4.22  |       +5.25  | pos, small   | SUB-NOISE (marginal) |
| compute-bound |   16 |       -7.26  |       +1.46  | **flips**    | SUB-NOISE / NULL |
| compute-bound |   64 |       +0.37  |       -4.34  | **flips**    | SUB-NOISE / NULL |
| compute-bound |  256 |       +6.10  |      +11.46  | pos          | SUB-NOISE (within spread) |

Representative spreads (run-2): compute-bound 256 bins checked 6.2999/7.0780 ms vs
unchecked 5.7785/6.7617 ms — the bands OVERLAP heavily; the +11.46% median delta is
inside the combined spread. Several cells flip sign across runs.

**Disposition: SUB-NOISE / effectively NULL.** The 256-bin cells lean slightly toward
unchecked (positive in all 4 of the 256-bin samples) but every delta stays within the
arm spread and the smaller-bin cells flip sign. There is at most a marginal, not
robustly measurable, benefit. The LDS kernel's hot path is LDS atomics +
`sync_cube()` barriers, which dominate the per-access bounds branch.

---

## Kernel 3 — fused build+fix+compact+scan (`build_fix_scan_fused_kernel`, f64 deterministic)

| regime        | bins | run-1 delta% | run-2 delta% | sign stable? | verdict |
|---------------|-----:|-------------:|-------------:|:-------------|:--------|
| launch-bound  |   16 |      +15.71  |      +12.94  | **yes (pos)**| **MEASURABLE** |
| launch-bound  |   64 |       +8.61  |      +12.06  | **yes (pos)**| **MEASURABLE** |
| launch-bound  |  256 |      +12.78  |      +15.23  | **yes (pos)**| **MEASURABLE** |
| compute-bound |   16 |      +39.00  |      +45.72  | **yes (pos)**| **MEASURABLE (large)** |
| compute-bound |   64 |      +45.43  |      +42.94  | **yes (pos)**| **MEASURABLE (large)** |
| compute-bound |  256 |      +42.21  |      +46.22  | **yes (pos)**| **MEASURABLE (large)** |

Spreads are NON-OVERLAPPING in the compute-bound cells. Run-2 compute-bound 256 bins:
checked p25/p75 **74.99 / 75.44 ms** vs unchecked **39.95 / 41.50 ms** — the entire
unchecked band is below the entire checked band, a clean ~1.8× separation. The
launch-bound cells (smaller absolute times) also show a consistent, non-overlapping
positive delta (~9–16%).

**Disposition: MEASURABLE and ROBUST.** `launch_unchecked` is a real, large,
sign-stable win for the fused kernel: ~9–16% in the launch-bound regime and
~40–46% (≈1.7–1.8×) in the compute-bound regime, reproduced across both process
restarts with non-overlapping spread. Because the fused kernel is f64 and
DETERMINISTIC (one cube per feature, `CubeDim::new_1d(1)`, sequential ascending
fold, NO atomics) and the two arms are asserted BIT-IDENTICAL, this delta is PURE
bounds-check codegen — not accumulation order, not transfer, not atomic contention.
This is the expected place for the win to surface: the kernel runs long
single-unit loops (zero/build over every bin × every row, fix over every bin,
compact, then the scan), so a per-access bounds branch is in the innermost hot loop
and its removal compounds across the whole sequential body.

---

## Summary verdict (per the honesty mandate)

| Kernel | launch-bound | compute-bound | overall |
|--------|:-------------|:--------------|:--------|
| f32-atomic              | NULL | NULL | **NULL** — no measurable benefit |
| resident-LDS (P=1)      | SUB-NOISE (marginal at 256 bins) | SUB-NOISE | **effectively NULL** |
| fused build+fix+scan f64| MEASURABLE (~9–16%) | MEASURABLE (~40–46%, ≈1.8×) | **REAL, large win** |

- The mwr expectation — that the realistic regime is transfer-/latency-bound and
  masks launch overhead — **holds for the two f32-atomic-class kernels** (atomic and
  LDS): their deltas are within the arm spread and flip sign across restarts. **No
  win is manufactured for them; they are reported as NULL / sub-noise.**
- The mwr expectation does **NOT** hold for the **fused f64 kernel**: there
  `launch_unchecked` delivers a robust, sign-stable, spread-separated speedup,
  because the bottleneck there is the long single-unit sequential loop body
  (many indexed f64 accesses), not transfer or atomics — exactly where dropped
  bounds-check codegen pays off.

## Measurement caveats

- **cold-ceiling-overstates-warm** (spike-findings SKILL): all numbers are WARM
  medians (WARMUP discarded). Cold/first-launch times overstate the steady-state
  cost; do not read the absolute ms as production per-leaf cost.
- The **launch-bound cells** (small leaf, minimal transfer) are the only place a
  per-launch codegen delta could surface for the latency-bound kernels — and even
  there the atomic/LDS deltas stayed sub-noise.
- The **fused kernel delta is pure codegen**: the f64 deterministic path makes the
  two arms bit-identical, so the ~40% gap is the bounds-check codegen alone, not f32
  accumulation drift.
- Re-run across ≥2 processes (done) to confirm sign stability; the atomic/LDS signs
  flip, the fused sign does not.

## Disposition for the nrw sweep

- **The nrw `launch_unchecked` sweep stays as-is.** This task is MEASUREMENT-ONLY
  quantification/confirmation, NOT a recommendation to revert. The sweep is now
  shown to be:
  - **strongly justified for the fused kernel** (a real ~1.8× compute-bound win), and
  - **harmless (neutral) for the f32-atomic and LDS kernels** (sub-noise — no
    measurable cost or benefit, so keeping the unchecked attribute there carries no
    regression).
- **Twin-sync caveat (load-bearing):** the bench `_checked` twins duplicate the
  shipped kernel bodies for timing only. Any future edit to
  `construct_hist_kernel_atomic_f32`, `construct_leaf_hist_resident_lds_kernel`, or
  `build_fix_scan_fused_kernel` MUST be mirrored into the twins in
  `launch_unchecked_ab.rs`, or the A/B silently drifts. The same-input asserts
  (f32-envelope / bit-equal) are the runtime guard that catches such drift.
