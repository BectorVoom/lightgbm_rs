---
phase: 12-gpu-sibling-scan-copack
fixed_at: 2026-06-25T00:00:00Z
review_path: .planning/phases/12-gpu-sibling-scan-copack/12-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 7
skipped: 2
status: partial
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-06-25
**Source review:** .planning/phases/12-gpu-sibling-scan-copack/12-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (fix_scope = all — includes Info)
- Fixed: 7
- Skipped: 2

All fixes preserve the phase-12 numerical-fidelity contract: no kernel math, no
floating-point operation order, and no bit-exact CPU anchor behavior was altered.
Every touched crate was `cargo check`/`cargo build --release`-verified after its fix.
The co-pack path's parity (confirmed clean by the reviewer) is untouched.

## Fixed Issues

### WR-01: Co-pack result alignment protected only by `debug_assert!`

**Files modified:** `crates/lgbm-treelearner/src/learner.rs`
**Commit:** 98542c9
**Status:** fixed: requires human verification
**Applied fix:** Promoted the compiled-out `debug_assert_eq!(splits.len(),
batched_feats.len(), ...)` in `scan_leaf_histogram`'s `precomputed_batched_splits`
branch to an always-on guard that returns `TreeLearnerError::LengthMismatch { expected:
batched_feats.len(), actual: splits.len() }`. A release-build spine-membership skew now
fails loudly instead of silently mis-mapping splits. Used the existing `LengthMismatch`
variant (semantically a length/alignment mismatch at the learner boundary) rather than
the review's illustrative `ComputeError::Runtime`, since this is a learner-side
invariant. The mutual-exclusion `debug_assert!` above it was left as-is (separate
invariant). Behavior is unchanged on the path taken today (lengths already match via the
caller's `smaller_feats == larger_feats` gate); flagged for human verification because it
converts an assertion into a runtime error-return on a parity-critical function.

### WR-02: Co-pack scannability gate vs kernel `sum_hessian > 0` reject can disagree

**Files modified:** `crates/lgbm-treelearner/src/learner.rs`,
`crates/lgbm-compute/src/kernels/split.rs`
**Commit:** 887538b
**Status:** fixed
**Applied fix:** Documentation-only, zero behavioral change. Designated the learner
co-pack scannability gate (`sum_hessians > 0.0 && num_data_in_leaf > 0` per sibling) as
the SINGLE SOURCE OF TRUTH and documented the kernel's per-sibling `!(sum_hessian > 0.0)`
hard-error in `find_best_splits_siblings` as a DEFENSIVE / unreachable boundary check on
the production co-pack path (a non-scannable leaf degrades to `none()` before co-pack is
considered). Added a matching note at the learner gate stating that if the gate is ever
relaxed, the kernel reject must be downgraded to graceful `none()` degradation. Chose the
"single source of truth + document" option over rewiring the kernel to return `none()`
vectors, because changing the kernel's error/return surface risks the parity contract and
the reject is genuinely unreachable today.

### WR-03: Co-pack trigger overloads `larger_resident_slot` as a did-subtract-happen flag

**Files modified:** `crates/lgbm-treelearner/src/learner.rs`
**Commit:** c57d665
**Status:** fixed: requires human verification
**Applied fix:** Introduced an explicit `let mut larger_is_resident_subtract = false;`
flag set to `true` ONLY on the resident `subtract_resident` success arm, and added it to
the co-pack eligibility gate (alongside the existing `larger_resident_slot ==
larger_slot_id` check). This makes "co-pack only for the subtract-derived larger child"
explicit and refactor-safe instead of implied by an unreachable-branch invariant.
Behavior-preserving today (the direct-build resident arm is unreachable post-root, so the
set of leaves that co-pack is unchanged); flagged for human verification because it adds a
condition to the co-pack gate.

### WR-04: `unsafe set_var/remove_var` in A/B bench is thread-unsafe and leaks ON-state on panic

**Files modified:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs`
**Commit:** fbce4f1
**Status:** fixed
**Applied fix:** Added a `CopackEnvGuard` RAII type (rocm-gated) whose `set(value)`
constructor sets `LGBM_SIBLING_COPACK` and whose `Drop` always removes it. The OFF and ON
A/B arms each scope a guard in a block so a panic in `timed_run` (e.g. the ON arm) can no
longer leak the `1` state forward into later sizes. Kept the env-toggle path (the
preferred `Config`-threading fix would touch the learner's override reader and was deemed
out-of-scope/higher-risk for a bench-only file); documented that `set_var` remains sound
only because the override is read on the main growth thread before parallel regions spawn.
Verified compiling under both default and `--features rocm`.

### IN-03: SC-4 verdict band hard-codes a 3% noise threshold on a single-process median

**Files modified:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs`
**Commit:** ba7a398
**Status:** fixed
**Applied fix:** Widened the verdict band (1.03/0.97 -> 1.05/0.95) and labeled every
verdict string "(single-proc, sign-only)" / "rerun >=2 procs", with a comment pointing to
the file header's ">=2 processes for sign-stability" requirement. The verdict now reads
the trend, not a confidence-bearing pass/fail.

### IN-04: `median` returns the upper-middle and panics on empty input

**Files modified:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs`
**Commit:** 657d6c7
**Status:** fixed
**Applied fix:** Added an `if ds.is_empty() { return Duration::ZERO; }` guard so an
operator-tuned `reps == 0` cannot panic, and documented the deliberate upper-median choice
(element at `len / 2`, not the mean of the two centers).

### IN-05: `let iters = iters;` no-op rebind

**Files modified:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs`
**Commit:** 5a31c1a
**Status:** fixed
**Applied fix:** Deleted the no-op rebind line — `iters` is a `Copy` `i32` already in scope
and the downstream closure captures it by copy.

## Skipped Issues

### IN-01: Three near-duplicate copies of the per-feature spine-gate ladder

**File:** `crates/lgbm-treelearner/src/learner.rs:2086-2131`, `2260-2309`, `1492-1505`
**Reason:** skipped: fix risks the numerical-fidelity contract on the most
parity-critical function. The three sites are NOT cleanly equivalent: the `1492-1505`
col-sampler `combine` computes the `used_features` mask (`is_feature_used_bytree &&
node_selected`), a DIFFERENT computation from the categorical/monotone/interaction/
extra-trees gate ladder shared only by `spine_batched_feats` (collect-only) and Pass-1
(which ALSO writes `spine_batch_index[fpos]` and falls non-spine features through to inline
handling). Extracting a shared `is_spine_feature` helper would touch the Pass-1 loop that
defines batch membership/ordering — precisely the byte-identity that WR-01 guards. The
CRITICAL CONTEXT instructs skipping any finding that risks parity. The WR-01 always-on
length guard (commit 98542c9) already converts the dedup's core risk (silent gate drift)
into a loud failure, mitigating IN-01's stated motivation. Recommend a dedicated,
parity-gated refactor phase if this quality improvement is desired.

### IN-02: `scan_leaf_histogram` has five mutually-constraining mode selectors

**File:** `crates/lgbm-treelearner/src/learner.rs:2141-2206`
**Reason:** skipped: fix risks the numerical-fidelity contract. Replacing
`resident_slot` / `fused_build` / `unified_build` / `subtract_inputs` /
`precomputed_batched_splits` with a `HistSource` enum is a substantial restructuring of
the single most parity-critical function in the crate AND every call site that builds its
arguments. The CRITICAL CONTEXT forbids altering behavior or FP-operation order on this
path; an enum refactor of the hot dispatch carries real risk of a subtle behavioral change
for zero functional benefit (the mutual exclusion is already `debug_assert!`'d). This is an
Info-tier quality improvement best done as a dedicated refactor with a bit-exact parity
gate, not as an inline review-fix.

---

_Fixed: 2026-06-25_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
