# Deferred items — quick 260619-mwr

## DEF-MWR-01 — `cuda_mirror_full_corpus_leaf_matches_anchor` is PRE-EXISTING flaky (out of scope)

**Status:** pre-existing, NOT introduced by this task. Logged, not fixed (SCOPE BOUNDARY).

**Symptom:** `cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror
cuda_mirror_full_corpus_leaf_matches_anchor` fails intermittently on gfx1100 with e.g.

```
cuda_mirror_full: cell 2 diverged beyond the f32-atomic envelope —
anchor 0.000005841255187988281 vs gpu -0.00000286102294921875
(|diff| 0.000008702278137207031 > tol 0.00000500005841255188)
```

The failing cells are GRAD cells whose TRUE sum is near zero (~5.8e-6, ~-0.25): the grad
histogram is the difference of large near-cancelling f32 partial sums, so the f32-atomic
reorder residual (run-to-run nondeterministic accumulation order) occasionally exceeds the
test's ABS 5e-6 floor (REL 1e-5 contributes ~nothing at these tiny magnitudes). Observed
|diff| reaches ~8.7e-6, ABOVE the test's documented "~2.4e-6 max" claim — i.e. the
documented bound was optimistic; the real worst-case on the full-corpus (~2000-row) leaf is
larger.

**Proven pre-existing (isolation experiment):** reverting BOTH
`crates/lgbm-compute/src/kernels/histogram.rs` AND
`crates/lgbm-compute/tests/rocm_cuda_mirror.rs` to `HEAD~1` (the `#[cube(launch)]` checked
kernel + the original 3 tests) and running the full-corpus test 6× still produced a failure
(1/6). `launch_unchecked` only removes in-kernel bounds-check codegen — it cannot change the
f32-atomic accumulation order — so this task did not introduce or worsen the flakiness.

**Why not fixed here:** the plan (260619-mwr) explicitly forbids weakening the tolerance or
changing the existing three tests ("Do NOT weaken the tolerance or change the existing three
tests"). The correct fix is a separate decision (raise the full-corpus ABS floor to ~1e-5
to match the real f32-atomic cancellation envelope on a 2000-row leaf, OR shrink the
full-corpus leaf so cancellation is bounded, OR pin the run with a fixed atomic order). Out
of scope for the overhead-reduction task.

**Scope-clean evidence for THIS task:** the new resident-Handle path
(`cuda_mirror_resident_matches_cpu_anchor_within_tol`) and the per-call `dense` / `empty`
tests are STABLE (5/5 across repeated runs). Both `launch_unchecked` launch paths agree with
the CPU f64 anchor within the envelope on the non-trivial `(7..num_data).step_by(3)` leaf
subset — the subset the plan specified. The flakiness is confined to the pre-existing
all-2000-rows cancellation-amplified cell.
