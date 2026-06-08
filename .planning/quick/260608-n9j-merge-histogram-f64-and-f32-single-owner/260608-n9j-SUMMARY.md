---
quick_id: 260608-n9j
slug: merge-histogram-f64-and-f32-single-owner
type: execute
date: 2026-06-08
mode: quick
phase: quick-260608-n9j
plan: 01
status: complete
requirements: [HIST-merge-shared-fold-helper]
key-files:
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
commits:
  - e89b9be: "refactor(260608-n9j): merge histogram f64/f32 fold into one generic #[cube] helper"
metrics:
  duration: ~6m
  tasks: 1
  files: 1
---

# Quick Task 260608-n9j: Merge histogram f64/f32 single-owner fold into one shared #[cube] helper

Collapsed the hand-duplicated single-owner ordered histogram fold (formerly inline in both
`construct_hist_kernel` f64 and `construct_hist_kernel_f32` f32 `#[cube(launch)]` kernels) into ONE
generic-over-`Numeric` `#[cube] hist_fold_body` helper. Both launch kernels are now thin wrappers
that call `hist_fold_body::<f64>(...)` / `hist_fold_body::<f32>(...)`. Single source of truth, zero
algorithmic/perf change. The generic `#[cube]`-over-`Numeric` approach (the precedent: split.rs's
`split_scan_body`, gain.rs's `Numeric`/`cast_from` usage) compiled cleanly on both backends — no
macro/duplication fallback was needed.

## What changed

`crates/lgbm-compute/src/kernels/histogram.rs`:
- Added private `#[cube] fn hist_fold_body<N: Numeric>(binned, grad, hess, out: &mut Array<N>)` — the
  single-owner (`UNIT_POS == 0`) ascending-row-order fold, `bin<<1` stride-2 cells,
  `out[ti] += N::cast_from(grad[i])`.
- `construct_hist_kernel` (f64) and `construct_hist_kernel_f32` (f32) keep their public
  `#[cube(launch)]` signatures/names/doc comments and now delegate to the helper. The
  capability-gate routing depends on the two named launch entry points; both are preserved verbatim.

For `N = f64`, `N::cast_from(grad[i])` is the f32→f64 widening — byte-identical to the prior
`f64::cast_from(grad[i])` (bit-exact cpu anchor). For `N = f32` it is the identity cast — observably
identical to the prior `out[ti] += grad[i]` (hip mirror).

## Out-of-scope, untouched (as mandated)
- Atomic kernels `construct_hist_kernel_atomic_f32` / `construct_leaf_hist_batched_kernel`
  (`Atomic<f32>::fetch_add`, different algorithm, `#[cfg(feature="rocm")]`) — not touched.
- `construct_histograms_cpu_native` (R2 native CPU path), host launchers, V5 validation, Backend
  trait, learner, untracked `LightGBM/` tree — not touched.

## Verification (real output captured)

| Gate | Result |
|------|--------|
| `cargo build --workspace` (cpu) | exit 0 — Finished in 28.72s |
| `cargo build --workspace --features rocm` | exit 0 (lgbm-compute clean rebuild; cubecl-hip in tree) |
| `cargo test -p oracle-harness --test kernel_parity` (cpu) | 6/6 GREEN incl. `kernel_parity_histogram_bit_exact_on_cpu` |
| `cargo test -p oracle-harness --test learner_parity` (cpu) | 29/29 GREEN bit-exact |
| `cargo test -p oracle-harness --features rocm` (kernel_parity) | `hip::kernel_parity_histogram_within_tol_on_hip` GREEN |
| `grep -c "for i in 0..binned.len()"` | 1 (fold loop now appears ONCE) |
| `cargo clippy -p lgbm-compute` | no histogram warnings |

### HARD GATE — CPU f64 BIT-EXACT: PASS
`kernel_parity_histogram_bit_exact_on_cpu` and all 29 `learner_parity` cases stayed GREEN
bit-exact. No ops reordered, no assertions weakened.

### Pre-existing rocm failure (NOT a regression)
`hip::kernel_parity_split_within_tol_on_hip` FAILS — but this is the **split** path (split.rs,
untouched by this task), a documented D-03a f32-vs-f64 accumulation gap (04-ROCM-GAPS.md). Verified
PRE-EXISTING: stashing this task's diff and running the test on the BASE produces byte-identical
failures (`hip=61.250004 vs cpu=61.25`, etc.; same `default_left` mismatch). The histogram hip test
(`hip::kernel_parity_histogram_within_tol_on_hip`) — the path this task actually changed — passes.

## Deviations from Plan
None — plan executed exactly as written. The generic `#[cube]`-over-`Numeric` helper (the preferred
approach) worked on this cubecl 0.10.0; the macro/duplication fallback was not required.

## Self-Check: PASSED
- `crates/lgbm-compute/src/kernels/histogram.rs` — FOUND (modified, contains `fn hist_fold_body`)
- Commit `e89b9be` — FOUND in `git log`
