---
title: Anchor-check the live u64 fixed-point resident BUILD at P>1 (close WR-05)
date: 2026-06-22
priority: medium
context: .planning/phases/11-gpu-fixedpoint-int-atomics/11-VERIFICATION.md (WR-05)
related_note: .planning/notes/gpu-hist-lds-privatized-u64-already-live.md
requires: ROCm GPU (#[cfg(feature=rocm)])
---

# Anchor-check the u64 resident BUILD at P>1 (WR-05)

> **RESOLVED 2026-06-22 (quick task 260622-t4u).** Added permanent test
> `kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip`
> (`crates/oracle-harness/tests/kernel_parity.rs`): a 300k-row leaf forces the
> DEFAULT heuristic to P>1 (no env dependence), asserts `row_partition_count > 1`,
> parity ≤1e-7 vs the CPU f64 anchor, and `to_bits()` determinism. GPU-verified on
> gfx1100: `row_partition_count(3,300000)=10`, `max_rel=0.000e0` (bit-exact),
> determinism held. Commits 5a2b3fa + f5bf726. See `quick/260622-t4u-SUMMARY.md`.

**The one real correctness hole in Phase 11.** The live-path parity gate
`kernel_parity_resident_build_fix_compact_equals_host_on_hip` only exercises a
**10-row, P=1 leaf**. But the headline feature of the phase is the multi-cube
**row-partition (P>1) merge** — each cube builds a partial LDS sub-histogram and
they sum into the global u64 buffer. That merge is **never anchor-checked at P>1**.

Integer adds are order-independent, so the P>1 read-back *should* be bit-equal to
the CPU f64 anchor regardless of cube-completion order — but that is exactly the
claim a P>1 comparison would PROVE, and it is currently untested. The phase's
verifier flagged this as a WARNING (not a blocker) and recommended closing it.

## Done when

- The resident u64 build is forced through a **P>1** path (e.g. `LGBM_ROWPART_MIN=0`
  with a leaf of ≥256k rows so row-partition actually splits into multiple cubes),
  and the read-back histogram still matches the CPU f64 anchor within
  `FIXEDPOINT_REL_GATE` (1e-7), with the `to_bits()` determinism sub-assert holding.
- Either extend the existing `kernel_parity.rs` test with a P>1 case, or add a
  sibling test, so the coverage is permanent (not a one-off manual run).

## Notes

- Must run on the ROCm GPU (`--features rocm`). The local box is a spoofed 8-CU APU
  — judge PARITY (exactness vs the f64 anchor), not throughput; throughput is
  APU-confounded.
- Closing this would let a Phase 11 re-verification reach `passed` outright
  (currently `human_needed`).
