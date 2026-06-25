---
phase: 12-gpu-sibling-scan-copack
reviewed: 2026-06-25T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/src/resident_pool.rs
  - crates/lgbm/examples/bench_gpu_vs_cpu.rs
  - crates/oracle-harness/tests/kernel_parity.rs
findings:
  critical: 0
  warning: 4
  info: 5
  total: 9
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-25
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 12 wires a 2-slot sibling-scan co-pack GPU kernel
(`find_best_splits_fused_siblings_kernel` /
`find_best_splits_fused_siblings_from_handles_on`), a `RocmBackend::scan_resident_siblings`
override, the learner growth-loop co-pack gate (`copack_feats` / `precomputed_batched_splits`),
a `LGBM_SIBLING_COPACK` env override, a cubecl-cpu W=1 byte-identity parity test, and a
co-pack ON/OFF A/B bench section.

The central numerical-fidelity claim — that co-packing changes only *which launch* a
feature's sequential `split_scan_body` runs in, never its math — holds up under review.
The co-packed launcher computes per-sibling `min_gain_shift` / `2*kEpsilon` bump exactly as
the single-slot path does, reuses the SAME shared `split_scan_body`, and applies the SAME
12-cell decode/accept-gate per half. The eligibility gate is conservative and structural
(both siblings resident-scan-only, both scannable, identical spine membership), with a
byte-unchanged two-scan fall-back, and the bit-exact CPU anchor path is never touched
(co-pack ANDs in `resident_eligible`, always false on `CpuBackend`). I found no BLOCKER:
no parity-breaking defect, no OOB/UB, no auth/secret/injection surface. The
`subtract_resident` correctly leaves the smaller slot resident, so both sibling Handles are
live at the co-pack point as the comments claim.

The findings below are robustness, fragile-invariant, and quality issues — most notably a
set of `debug_assert!`-only guards protecting load-bearing co-pack alignment invariants that
become silent mis-mappings in release if an upstream refactor ever violates them (WR-01), and
a `sum_hessian > 0` scannability gate that is checked in the learner but is also a hard error
in the kernel, where a future caller change could turn a gate-skip into a typed-error abort
(WR-02).

## Warnings

### WR-01: Co-pack result alignment is protected only by `debug_assert!` — a release-build spine-membership skew silently mis-maps splits

**File:** `crates/lgbm-treelearner/src/learner.rs:2348-2356` (gate at `:1767-1803`)
**Issue:** When co-pack fires, the smaller and larger siblings each receive
`precomputed_batched_splits = Some(vec)`, and `scan_leaf_histogram` re-derives its OWN
`batched_feats` in Pass-1, then uses `splits` directly as `batched_splits`, indexing by
`spine_batch_index[fpos]`. The contract that `splits.len() == batched_feats.len()` and that
the two are slot-for-slot aligned is enforced only by:
```rust
debug_assert_eq!(
    splits.len(), batched_feats.len(),
    "co-packed splits must align with this leaf's spine batch"
);
```
This is correct *today* because the gate at `:1796` requires `smaller_feats == larger_feats`
(computed by `spine_batched_feats`, a deliberate duplicate of Pass-1's gate logic). But there
are now THREE copies of the identical gate sequence (Pass-1 in `scan_leaf_histogram:2260-2309`,
`spine_batched_feats:2086-2131`, and the equality check). If any one drifts (e.g. a future gate
added to Pass-1 but not to `spine_batched_feats`), the lengths can still match while the
*ordering/membership* differs, producing a wrong-but-plausible tree — and in a `--release`
build the `debug_assert!` is compiled out, so it ships silently. Per CLAUDE.md non-negotiable
#1 (numerical fidelity), an alignment skew is exactly the class of defect that must fail loudly.
**Fix:** Promote the length check to an always-on guard returning a typed error, and/or make
`scan_leaf_histogram` consume the caller's already-computed `feats` for the co-pack case rather
than re-deriving `batched_feats` from a third copy of the gate:
```rust
if splits.len() != batched_feats.len() {
    return Err(TreeLearnerError::Compute(ComputeError::Runtime {
        detail: "co-packed splits misaligned with leaf spine batch (gate drift)".into(),
    }));
}
```

### WR-02: Co-pack scannability gate and kernel `sum_hessian > 0` reject can disagree on the boundary

**File:** `crates/lgbm-treelearner/src/learner.rs:1762-1778`; `crates/lgbm-compute/src/kernels/split.rs:1612-1623`
**Issue:** The learner gates co-pack on `smaller_splits.sum_hessians > 0.0 &&
... num_data_in_leaf > 0` (same for larger). The kernel
`find_best_splits_fused_siblings_from_handles_on` *independently* HARD-ERRORS on
`!(sum_hessian_a > 0.0)` / `!(sum_hessian_b > 0.0)`. These two predicates must stay in
lock-step: the learner gate is the sole guarantee the kernel reject is never hit. They agree
today, but are expressed at two layers with no shared constant. If the learner gate is ever
relaxed (or a sibling's `sum_hessians` rounds to exactly `0.0` under a future bagging/GOSS edge),
the co-pack path aborts the WHOLE tree with a typed error (propagated via `?` at `:1832`)
instead of degrading that leaf to `none()` the way non-co-packed `scan_leaf_histogram` does at
`:2225`. A perf optimization should never convert "this leaf can't split" into a train-aborting
error.
**Fix:** Make the co-pack gate the single source of truth (document the kernel reject as an
unreachable defensive assert), OR have the sibling launcher mirror `scan_leaf_histogram`'s
graceful degradation (return `none()` vectors for a non-positive-hessian sibling) so co-pack is
never *less* robust than the fall-back it replaces.

### WR-03: Co-pack trigger overloads `larger_resident_slot` as both "scan slot" and "did-resident-subtract-happen" flag

**File:** `crates/lgbm-treelearner/src/learner.rs:1637-1666, 1774`
**Issue:** `larger_resident_slot` is `Some(larger_slot)` after EITHER the
`subtract_resident` arm (`:1644`) OR the resident direct-build else-arm (`:1654`
`build_resident_leaf_into`). Co-pack eligibility at `:1774` uses
`larger_resident_slot == larger_slot_id` as a proxy for "the larger child was derived by
resident subtract." But the directly-built resident arm also sets it to `Some(larger_slot)`, so
that comparison would ALSO be true there. Today the direct-build arm is documented "cannot happen
post-root in the spine," so it is unreachable — but the co-pack kernel was only validated against
the subtract-derived larger Handle layout, and relying on an unreachable branch to keep the gate
correct is fragile.
**Fix:** Introduce an explicit `larger_is_resident_subtract: bool` set only on the
`subtract_resident` success arm and gate co-pack on it, making "co-pack only for the
subtract-derived larger child" explicit and refactor-safe instead of implied by control flow.

### WR-04: `unsafe { std::env::set_var/remove_var }` in the A/B bench is thread-unsafe and leaks ON-state on panic

**File:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs:184, 188, 224`
**Issue:** The co-pack A/B toggles `LGBM_SIBLING_COPACK` via `unsafe { std::env::set_var(...) }`
between OFF/ON arms in-process. `set_var`/`remove_var` are `unsafe` precisely because they are
not thread-safe; `train()` (and cubecl/rayon under it) spawns workers, and any concurrent
`std::env::var` read during `set_var` is UB. It works because `sibling_copack_override()` is read
on the main growth thread before the parallel regions, but the harness is the documented operator
entry point, so the fragility ships. Also, a panic in the ON arm before `:224` leaks the var set
to `1` for every subsequent size.
**Fix:** Thread the override through `Config` (a `sibling_copack: Option<bool>` field) so the A/B
selects per-`train` call with no env mutation. If the env path must stay, scope the reset in an
RAII guard whose `Drop` runs `remove_var`, so an ON-arm panic cannot leak ON state forward.

## Info

### IN-01: Three near-duplicate copies of the per-feature spine-gate ladder

**File:** `crates/lgbm-treelearner/src/learner.rs:2086-2131` (`spine_batched_feats`), `2260-2309` (Pass-1), `1492-1505` (col-sampler combine)
**Issue:** The col-sampler-mask → parent-splittable → interaction-allowed → not-categorical →
not-monotone → not-extra-trees ladder is transcribed three times; the comments themselves note
they "MUST stay byte-identical" (`:2066`). This duplication is the structural root of WR-01.
**Fix:** Extract one `fn is_spine_feature(&self, fpos, f, leaf, mask, parent_splittable) -> bool`
(plus the shared `BatchedSplitFeature` constructor) and call it from all three sites.

### IN-02: `scan_leaf_histogram` now has five mutually-constraining mode selectors

**File:** `crates/lgbm-treelearner/src/learner.rs:2141-2206`
**Issue:** `resident_slot`, `fused_build`, `unified_build`, `subtract_inputs`, and the new
`precomputed_batched_splits` are five mode-flags whose mutual exclusion is only `debug_assert!`'d
(`:2348`), on the single most parity-critical function in the crate.
**Fix:** Replace with one `enum HistSource { HostBuf, Resident(usize), Fused(usize),
UnifiedBuild, SubtractScan(Vec<f64>, Vec<f64>), Precomputed(Vec<SplitInfo>) }` so the
mutual-exclusion is type-enforced, not asserted.

### IN-03: SC-4 verdict band hard-codes a 3% noise threshold on a single-process median

**File:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs:198-206`
**Issue:** `ratio >= 1.03` / `>= 0.97` cutoffs print `trends-faster` / `NOT-SLOWER` /
`SLOWER(noise? rerun)`. On the spoofed 8-CU APU single-process medians can swing beyond 3% (the
file's own header demands ">=2 PROCESSES for sign-stability"), so the printed verdict implies
more confidence than the data supports.
**Fix:** Widen the band, label it "(single-process, sign-only)", or compute it across the >=2
processes the methodology requires.

### IN-04: `median` returns the upper-middle and panics on empty input

**File:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs:100-103`
**Issue:** `ds[ds.len() / 2]` is the upper-median (not the mean of the two centers) and panics on
an empty slice. Harmless at `reps >= 3`, but unguarded if `reps` ever becomes operator-tunable to 0.
**Fix:** `if ds.is_empty() { return Duration::ZERO; }` and document the upper-median choice.

### IN-05: `let iters = iters;` no-op rebind

**File:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs:274`
**Issue:** `let iters = iters; // bind for the closure below` is a no-op — `iters` is a `Copy`
`i32` already in scope and the closure at `:335` captures it by copy regardless.
**Fix:** Delete the line.

---

_Reviewed: 2026-06-25_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
