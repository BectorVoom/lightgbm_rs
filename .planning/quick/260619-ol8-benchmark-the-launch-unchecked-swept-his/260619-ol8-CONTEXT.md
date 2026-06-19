# Quick Task 260619-ol8: Benchmark the launch_unchecked-swept histogram kernels for actual overhead reduction - Context

**Gathered:** 2026-06-19
**Status:** Ready for planning

<domain>
## Task Boundary

Measure the ACTUAL overhead reduction from quick-260619-nrw's sweep that switched the
production rocm-gated histogram kernels from `#[cube(launch)]` to `#[cube(launch_unchecked)]`
(drops in-kernel per-access bounds-check codegen). MEASUREMENT-ONLY: this task adds a
rocm-gated benchmark example and reports real gfx1100 numbers; it does NOT change any
production kernel, launcher, or the CPU f64 anchor.

</domain>

<decisions>
## Implementation Decisions (LOCKED — do not revisit)

### A/B methodology — dual-kernel, ONE binary, interleaved
- `launch_unchecked` is a COMPTIME kernel attribute → it CANNOT be toggled at runtime in a
  single binary (mwr hit this; could only report it qualitatively). So the A/B harness MUST
  define, for each benchmarked kernel, a `_checked` twin (identical body, `#[cube(launch)]`)
  beside the shipped `_unchecked` kernel, and launch BOTH in the same process.
- Honor the project measurement rule: discard warm-ups (≥2–3), report median-of-N (≥5–7),
  device-sync (result read-back) inside each timed call, interleave checked vs unchecked,
  ideally reproduce across ≥2 process restarts for drift.
- The twin kernels live in the BENCH/example (or a clearly bench-only module), NOT wired into
  production — keep production untouched. They must be byte-identical to the shipped kernels
  except the launch attribute, so the comparison isolates exactly the bounds-check codegen.

### Scope — hot-loop kernels, launch-bound regime
- Benchmark the kernels where in-kernel bounds-checks sit in a HOT scatter loop and where the
  delta could plausibly show: the atomic kernel (`construct_hist_kernel_atomic_f32`), the
  WIRED training path (`construct_leaf_hist_resident_lds_kernel`), and the fused
  build+scan kernel (`build_fix_scan_fused_kernel`). (3 representative kernels, not all 8.)
- Drive them in the LAUNCH-BOUND regime — small leaves / many launches / minimal per-call
  transfer — because mwr proved the realistic regime is TRANSFER-bound, where launch+codegen
  overhead is masked. Also include at least one larger/compute-bound point for contrast so the
  report is honest about where (if anywhere) the win is real vs sub-noise.

### Honesty mandate
- A NULL / sub-noise result is an ACCEPTABLE and likely outcome and MUST be reported plainly
  (the mwr finding: dominant overhead is transfer, not launch). Do NOT manufacture a win.
  Report the regime(s) where the delta is measurable vs where it is noise, with the median
  spread so the reader can judge significance.

</decisions>

<specifics>
## Specific Ideas

- Reuse the existing harness patterns: `crates/lgbm-compute/examples/cuda_mirror_overhead.rs`
  and `mirror_vs_lds.rs` (warm-up discard + median, MB/call, rocm-gated with a CPU-only stub
  main).
- nrw commits: d4dde2f / 609cc67 / 61aac21 (the sweep). Pre-sweep parent = c3e5b05.
- Known landmine DEF-MWR-01 (full-corpus near-zero-grad f32-atomic cancellation flake) — this
  is a TIMING bench, not a parity gate, so parity asserts are optional; if the twin kernels
  are sanity-checked against each other for equal output, use the f32-atomic envelope and note
  it is NOT the real parity gate (that stays GPU-vs-CPU-f64-anchor in rocm_cuda_mirror.rs).

</specifics>

<canonical_refs>
## Canonical References

- `crates/lgbm-compute/src/kernels/histogram.rs` — the swept production kernels (read-only here).
- `crates/lgbm-compute/examples/cuda_mirror_overhead.rs`, `mirror_vs_lds.rs` — harness patterns.
- `.planning/quick/260619-nrw-*/260619-nrw-SUMMARY.md` — what was swept + the qualitative claim.
- Skill `spike-findings-lightgbm_rs` — cold-ceiling-overstates-warm + GPU-vs-CPU-f64-anchor rules.

</canonical_refs>
</content>
