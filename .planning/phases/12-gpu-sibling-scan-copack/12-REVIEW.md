---
phase: 12-gpu-sibling-scan-copack
reviewed: 2026-06-25T02:21:14Z
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
  warning: 2
  info: 5
  total: 7
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-25T02:21:14Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed the spike-024 sibling-scan co-pack wiring: a new 2-slot co-packed scan
kernel + Handle launcher (`split.rs`), the `Backend::scan_resident_siblings` trait
default + RocmBackend impl (`lib.rs`), the growth-loop reorder that defers the
smaller-child scan and co-packs both siblings (`learner.rs`), the
`LGBM_SIBLING_COPACK` env gate (`resident_pool.rs`), the A/B bench harness
(`bench_gpu_vs_cpu.rs`), and the byte-identical + ~1e-6 anchor parity cells
(`kernel_parity.rs`).

The implementation is careful and the bit-exact-by-construction contract is
well-defended. I traced the four highest-risk areas the context flagged:

- **2-slot lane mapping / OOB:** the kernel guards `g < 2*n_feats`, writes only
  `out[g*12 .. g*12+12]` inside the `2*n*12` allocation, and the
  `if g < n_feats {g} else {g - n_feats} as usize` index parses (verified) as
  `(if … else …) as usize` ⇒ correct local feature index. No OOB.
- **Shared params vs per-sibling scalars:** per-feature arrays (`slot_off`..`fwd_count`)
  are built once and shared; the `2*kEpsilon` bump + `min_gain_shift` and
  `(sum_gradient, sum_hessian, num_data)` are threaded PER SIBLING (A-suffixed and
  B-suffixed) into the kernel and into the two decode halves. No swap/aliasing.
- **Both Handles simultaneously resident:** the larger build/subtract is hoisted
  ahead of either scan; the smaller scan is deferred past `subtract_resident`; the
  RocmBackend impl borrows both slots in one `borrow()` scope. Neither slot is freed
  early.
- **Eligibility gate:** the gate ANDs `resident_eligible`, smaller-resident-only,
  larger-subtract-resident-only, both-scannable, spine-equality, and the env
  override; any miss falls back to the byte-unchanged two-scan path. The spine
  replica (`spine_batched_feats`) is byte-faithful to `scan_leaf_histogram`'s Pass-1
  (same gates, same order, same struct fields; `BatchedSplitFeature` derives
  `Eq` over integer/bool-only fields ⇒ exact equality).

The CPU f64 anchor is genuinely untouched (the co-pack gate ANDs in
`resident_eligible`, which is false on CpuBackend), the W=1 byte-identity test
passes on the default build, and the byte-identical parity cell uses `assert_eq!`
on the full `SplitInfo` while the ~1e-6 cell pins to the CPU f64 anchor
(`find_best_split_cpu_native`), honoring def-f8u-01.

No Critical findings. Two Warnings (a maintenance-fragility duplication and a
behavioral edge in the larger-resident branch that is correct today but
under-guarded) and five Info items.

## Warnings

### WR-01: `spine_batched_feats` silently duplicates `scan_leaf_histogram`'s Pass-1 with no shared-source or test guard against drift

**File:** `crates/lgbm-treelearner/src/learner.rs:2056-2120` (replica) vs `:2257-2309` (original Pass-1)

**Issue:** Co-pack correctness depends on `spine_batched_feats` producing a feature
list byte-identical to the Pass-1 gate-only pre-pass inside `scan_leaf_histogram`.
The two are hand-copied: same six gates (col-sampler mask → parent-splittable →
interaction → categorical → monotone → extra-trees) in the same order, pushing the
same eight `BatchedSplitFeature` fields. Today they match exactly. But they are two
independent copies of a load-bearing predicate. If a future edit changes one Pass-1
gate (e.g. adds a new spine-exclusion, or reorders a gate, or changes a field) and
the maintainer does not also update the replica, the co-pack path will feed a
`feats` list that is mis-aligned with the sibling's actual `batched_feats` — the
`debug_assert_eq!(splits.len(), batched_feats.len())` only catches a LENGTH
mismatch (release builds skip it entirely), not a same-length membership skew, which
would silently mis-map split results between features. There is no test that asserts
the replica equals the live Pass-1 output for a non-trivial mask.

**Why it matters:** Silent feature-to-result mis-mapping corrupts split selection
(wrong threshold/gain attributed to wrong feature) and would only surface as a
hard-to-diagnose parity drift on col-sampled / interaction-constrained workloads on
the rocm path — exactly the configs the spine-equality guard was added to handle.

**Fix:** Either (a) extract the Pass-1 gate loop into ONE shared private method that
both `scan_leaf_histogram` and the co-pack site call (single source of truth — the
project's stated preference, cf. `find_best_splits_fused_inner` as the single scan
source), or (b) add a `debug_assert_eq!` at the co-pack site comparing
`spine_batched_feats(smaller_leaf, …)` against the `batched_feats` that
`scan_leaf_histogram` actually built for the smaller leaf, plus a unit test that
drives a non-empty `used_features` mask + an interaction constraint through both
paths and asserts equality. Option (a) is preferred and eliminates the drift class
entirely.

### WR-02: Co-pack eligibility does not require the larger child to be subtract-derived, yet the contract/comments assume it is

**File:** `crates/lgbm-treelearner/src/learner.rs:1772-1788` (gate) and `:1637-1666` (larger-resident assignment)

**Issue:** The CONTEXT and the co-pack comments state the larger child is the
"resident SUBTRACT+scan-only path" and is "subtract-derived". But the gate only
checks `larger_resident_slot == larger_slot_id` (i.e. `Some(larger_slot)`).
`larger_resident_slot` is set to `Some(larger_slot)` in BOTH the subtract branch
(`:1644 subtract_resident`) AND the direct-build branch (`:1654
build_resident_leaf_into`, the "no parent retained" case the comment itself says
"cannot happen post-root in the spine"). If that direct-build branch ever does fire
(e.g. a future spine change, or a root-adjacent edge), the larger child is resident
but NOT subtract-derived — co-pack would still engage. This is numerically still
correct (the kernel reads whatever resident histogram is in the slot; the math is
identical regardless of derivation), so it is not a correctness bug TODAY, but the
gate's guarantee is weaker than the documented contract, and the discrepancy is
not asserted.

**Why it matters:** A reviewer/maintainer reading the gate trusts the comment's
"subtract+scan-only" invariant; the code does not enforce it. If a later change
makes the direct-build-resident larger child behave differently (e.g. needs a
re-fix the subtract path skips per non-negotiable #3), co-pack would silently
include a case the contract excluded.

**Fix:** Either tighten the gate to assert the larger child came through the
subtract path (e.g. carry a `larger_subtract_derived: bool` flag set only in the
`subtract_resident` arm and AND it into the co-pack predicate), or relax the
comment/CONTEXT wording to "larger child resident (subtract-derived in the live
spine; direct-build-resident is also co-packable since the kernel reads the slot
contents)" and add a `debug_assert!` documenting that the direct-build arm is
unreachable post-root. Pick one so the code and the stated invariant agree.

## Info

### IN-01: Unnecessary `#[allow(clippy::neg_cmp_op_on_partial_ord)]` on a positive comparison

**File:** `crates/lgbm-treelearner/src/learner.rs:1762-1764`

**Issue:** `smaller_scannable` is computed as `smaller_splits.sum_hessians > 0.0 &&
smaller_splits.num_data_in_leaf > 0` — both POSITIVE comparisons — but carries
`#[allow(clippy::neg_cmp_op_on_partial_ord)]`, a lint that only fires on `!(x > 0.0)`
forms. The allow is inert here (copy-pasted from the `!(sum_hessian > 0.0)`
NaN-catching sites). It also means this `smaller_scannable` does NOT catch NaN the
way the kernel's `!(sum_hessian > 0.0)` does — a NaN `sum_hessians` makes
`smaller_scannable` false (NaN > 0.0 is false), so the gate correctly falls back, so
behavior is fine; only the allow annotation is misleading.

**Fix:** Drop the `#[allow]` on the `smaller_scannable` line (it guards nothing), or
add a comment noting NaN ⇒ false ⇒ falls back to the two-scan path.

### IN-02: `larger_scannable` lacks the parallel `smaller_scannable` local + allow, slight asymmetry

**File:** `crates/lgbm-treelearner/src/learner.rs:1782-1783`

**Issue:** The smaller child's scannability is hoisted into a named
`smaller_scannable` local (`:1763`), but the larger child's identical check
(`larger_splits.sum_hessians > 0.0 && larger_splits.num_data_in_leaf > 0`) is
inlined directly into the `if` chain at `:1782-1783`. Minor readability asymmetry;
both are correct.

**Fix:** Hoist a parallel `larger_scannable` local for symmetry, or inline both.
Cosmetic.

### IN-03: Parity fixture's `buf_b` is an independent seed, not a true subtract-derived histogram

**File:** `crates/oracle-harness/tests/kernel_parity.rs:1148-1167` (`copack_two_histograms`)

**Issue:** The co-pack parity cells construct `buf_b` (the "larger-subtract-derived"
sibling) as a hand-written independent histogram, not as `parent − smaller`. This is
fine for the KERNEL parity gate (it exercises two slots with different totals over a
shared layout, which is exactly what the kernel sees), and the byte-identical
`assert_eq!` vs two single-slot scans is the strong gate. But the fixture comment
calls `buf_b` "larger-subtract-derived", which slightly overstates what is tested:
the test does not validate that a real subtract-derived child co-packs correctly
end-to-end (that is covered separately by `learner_parity` + the live growth path).

**Fix:** Reword the fixture comment to "second sibling with different leaf totals
(mimicking the smaller-built vs larger-derived asymmetry)" so it does not imply a
subtract derivation the test does not perform. No code change needed.

### IN-04: `unsafe { set_var/remove_var }` in the bench A/B is process-global and not thread-safe

**File:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs:206-249`

**Issue:** The A/B toggles `LGBM_SIBLING_COPACK` via `unsafe std::env::set_var`
between arms in-process (correct, since `sibling_copack_override()` reads the env per
query). This is a bench-only example, single-threaded at the toggle points, so it is
safe in practice. Flagged only because `set_var` is process-global: if the bench
ever trained concurrently (rayon spawns threads inside `train`, but the env is read
before the parallel region per query), a future refactor that reads the override
inside a parallel region could race. Not an issue today.

**Fix:** No action required for the bench. If `sibling_copack_override()` is ever
called from inside a parallel region, snapshot it once per train into the learner
rather than reading env per query under concurrency.

### IN-05: `cube_count` uses `2 * n as u32` where a large `n` could overflow before `div_ceil`

**File:** `crates/lgbm-compute/src/kernels/split.rs` (`find_best_splits_fused_siblings_from_handles_on`, the `let cube_count = (2 * n as u32).div_ceil(scan_w);` line)

**Issue:** `n` is `feats.len()` (`usize`). `(2 * n as u32)` casts `n` to `u32` then
doubles it. For `n > u32::MAX/2` (~2.1 billion features) this overflows; in release
it wraps, in debug it panics. Feature counts are realistically in the hundreds to
low thousands, so this is purely theoretical, but the single-slot path
(`find_best_splits_fused_inner:1467`) uses `(n as u32).div_ceil(scan_w)` (no `2*`),
so the co-pack path has half the headroom. The per-feature `out_len = 2 * n * 12`
(usize) and the device allocations are the real scale limits long before this.

**Fix:** None required (unreachable feature count). If desired for symmetry, compute
`let total = (2 * n) as u32;` from the usize product, or assert `n` fits, matching
the single-slot site's style.

---

_Reviewed: 2026-06-25T02:21:14Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
