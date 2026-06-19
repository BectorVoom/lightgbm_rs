# Quick Task 260619-nrw: Refer cubecl manual and reduce overhead in GPU kernel - Context

**Gathered:** 2026-06-19
**Status:** Ready for research + planning

<domain>
## Task Boundary

Reduce non-compute / launch / codegen overhead in the **production** GPU histogram
kernels by applying current cubecl-manual techniques. This continues the GPU-overhead
campaign:
- `260619-j9t` ported the CUDA-mirror histogram kernel to cubecl.
- `260619-mwr` cut the **mirror's** overhead ~10× via `launch_unchecked` (drops in-kernel
  bounds-check codegen; host-side V5 validation discharges the contract) + an upload-once
  resident-Handle launcher.
- `260619-ngo` A/B-benched mirror vs the wired LDS resident path and concluded: **leave the
  mirror as a non-wired primitive** — no win regime; the production path is the LDS resident
  build (`build_leaf_histograms_resident_f32_on`).

KEY GAP found this session: `launch_unchecked` and the manual's launch-overhead techniques
are applied ONLY to the CUDA-mirror kernel. Every **production** kernel in
`crates/lgbm-compute/src/kernels/histogram.rs` still uses plain `#[cube(launch)]` with
in-kernel bounds-check codegen (e.g. `construct_leaf_hist_resident_lds_kernel::launch`:1403,
`construct_leaf_hist_resident_kernel::launch`:1426, LDS build:618, atomic:443,
batched:721/743, fix-compact:1657/1808).

</domain>

<decisions>
## Implementation Decisions (LOCKED — do not revisit)

### Target kernels
- **ALL production histogram kernels** in `histogram.rs` are in scope (LDS build, atomic,
  batched, resident, resident-LDS, fix-compact) — NOT just the wired resident path.
- The CUDA-mirror primitive is OUT of scope for further squeezing (ngo closed it; leave as
  primitive). It may still serve as a reference for which manual techniques transfer.

### Aggressiveness / risk
- **Compute restructuring is ALLOWED** — accumulation/reduction-order changes that may shift
  f32 results are permitted, provided they stay within the project's ~1e-6 parity envelope
  vs the CPU f64 anchor.
- This is BROADER than mwr's strictly-numerics-preserving scope. Because order changes can
  move f32 results, every restructured kernel MUST be re-pinned to the CPU f64 anchor
  (GPU-vs-CPU-f64-anchor, NEVER GPU-vs-GPU — memory DEF-f8u-01), and if any change pushes a
  cell past the existing tolerance, the tolerance review is part of this task (document the
  residual, do not silently weaken a gate without flagging it).

### Mandated input
- "refer cubecl manual" is a hard requirement: a research pass MUST pull current cubecl
  launch/codegen/overhead guidance (launch_unchecked, comptime specialization, CubeDim/
  CubeCount tuning, shared-memory/atomics patterns, redundant-upload elimination) BEFORE
  planning. Use context7/find-docs for cubecl.

### Process
- Running as quick `--research --validate`: focused researcher → planner (quick-full) →
  plan-checker loop → executor → verifier. Justified by the parity-critical, compute-
  restructuring scope on the wired training path.

</decisions>

<specifics>
## Specific Ideas

- The proven, lowest-risk lever to extend first: switch production kernels from `::launch`
  to `::launch_unchecked` with host-side bounds validation discharging the contract — exactly
  what mwr did for the mirror (commit 61b96d3), now never applied to production.
- Pre-existing flaky parity test DEF-MWR-01 (full-corpus near-zero-grad f32-atomic
  cancellation, |diff|~8.7e-6 occasionally > ABS 5e-6) is a known landmine — distinguish any
  new parity movement from this pre-existing noise.

</specifics>

<canonical_refs>
## Canonical References

- `crates/lgbm-compute/src/kernels/histogram.rs` — all production kernels + the mirror.
- `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` — GPU-vs-CPU-f64-anchor parity pattern.
- Prior summaries: `.planning/quick/260619-mwr-*/260619-mwr-SUMMARY.md`,
  `.planning/quick/260619-ngo-*/260619-ngo-SUMMARY.md`.
- Skill: `spike-findings-lightgbm_rs` (gpu-hist-levers-closed, cold-ceiling-overstates-warm,
  GPU-vs-CPU-f64-anchor parity rule).
- cubecl manual (to be fetched in research): launch_unchecked, comptime, CubeDim/CubeCount.

</canonical_refs>
</content>
</invoke>
