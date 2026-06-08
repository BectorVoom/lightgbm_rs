---
quick_id: 260609-aqy
title: GPU host<->device boundary optimization analysis (bytemuck / arrow-rs / boundary copies)
type: investigation
status: complete
date: 2026-06-09
tasks_completed: 1/1 (Task 2 optional impl correctly skipped)
deliverable: .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md
commit: 6781c4f
---

# Phase quick-260609-aqy Plan 01: GPU boundary optimization analysis — Summary

Surveyed the entire host<->device boundary in `lgbm-compute` and produced a ranked,
evidence-grounded analysis: bytemuck is REJECT (redundant with cubecl's
`CubeElement::as_bytes`/`from_bytes`, transitive-only in `Cargo.lock`), arrow-rs at the
boundary is REJECT (data already contiguous `&[T]`), and no boundary-copy reduction
qualifies as a zero-parity-risk implementation given the L3 finding that the round-trip is
not the GPU bottleneck — so analysis-only, no code change.

## What was done

- Read all boundary code (`histogram.rs`, `split.rs`, `subtract.rs`, `partition.rs`,
  `lib.rs`, `runtime.rs`) and the prior 260609-9nu arrow-rs dataset findings.
- Ran an exhaustive Grep sweep (`create_from_slice`, `from_bytes`, `client.empty`,
  `vec![0`, `extend_from_slice`, `as_bytes`, `map(|&x| f64::from`) so the survey covers
  ALL ~13 zero-alloc launcher sites and 3 widening-collect sites, not just the scouted
  ones.
- Re-verified every cited line against live code (the scouted line hints were confirmed;
  e.g. widening collects at `histogram.rs:392/495/606`, the zero-init comment at
  `histogram.rs:161-165`, split out-cells fully written at `split.rs:372-383`/`605-616`).
- Wrote `260609-aqy-ANALYSIS.md` with: Summary, L3 reality check, ranked opportunities
  table + per-opportunity subsections (file:line | current code | proposed change |
  L3-framed impact | parity risk | verdict), explicit bytemuck and arrow-rs verdicts, a
  Recommendation, and a Parity Verdict rollup.

## Key findings

- **bytemuck: REJECT.** The cast is already zero-copy via `CubeElement::as_bytes`/
  `from_bytes`; no struct-of-arrays / non-CubeElement / Vec-elimination niche exists at
  this boundary. Already in `Cargo.lock:310` (v1.25.0) transitively only — not a direct
  dep in any workspace `Cargo.toml`.
- **arrow-rs at boundary: REJECT.** Data reaching `create_from_slice` is already a
  contiguous `&[T]`; an arrow `Buffer` feeds it no better and duplicates the already-present
  `polars-arrow` runtime (`Cargo.lock:2927`). Dataset/ingest verdict cited from 260609-9nu,
  not re-derived (arrow-rs earmarked only for v2 `ING-*`).
- **Accumulate/atomic zero-init buffers (O3): KEEP.** Construct (`out[ti] += ...`) and
  atomic (`fetch_add`) outputs depend on zero-init; `empty()` would fold onto stale pooled
  memory -> wrong histogram -> break the bit-exact cpu merge gate. The in-code comment at
  `histogram.rs:161-165` already documents this; left untouched.
- **f32->f64 widening collects (O2): REJECT as target.** Necessary type conversions
  (4->8 byte), not reinterprets — not removable by bytemuck/arrow.
- **Split out-cells `empty()` (O1): INVESTIGATE-FURTHER, not now.** Single-feature split
  kernel fully overwrites all 12 cells (safe), but value is L3-noise (~96 bytes) and it
  introduces a sharp "must-overwrite" invariant a multi-feature refactor could break into a
  parity bug. Fails the zero-parity-risk + worth-it bars for optional Task 2.

## Deviations from plan

None. Task 2 (optional implementation) correctly skipped — the planner judged none
qualifies by default, and the live-code reading confirmed no trivially-small,
zero-parity-risk, parity-neutral win exists (O1 is closest and fails both bars).

## Parity impact

None. No recommendation weakens the f32 ~1e-6 / cubecl-cpu f64-fold bit-exact contract.
The single accumulate/atomic hazard is explicitly flagged and left as-is.

## Verification

- Plan's automated verify command: **PASSED** (doc exists; bytemuck + arrow present;
  verdict keywords present; every `histogram.rs:N` citation <= file length).
- Cross-file spot-checks (split.rs:799/1678/372, lib.rs:815, subtract.rs:99,
  partition.rs:230, Cargo.lock:310/2927): all resolve to the claimed code.

## Files

- Created/committed: `.planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md` (commit 6781c4f)

## Self-Check: PASSED
- Deliverable exists at required path (verified).
- Commit 6781c4f exists (verified).
- All file:line citations resolve to matching live code (verified).
