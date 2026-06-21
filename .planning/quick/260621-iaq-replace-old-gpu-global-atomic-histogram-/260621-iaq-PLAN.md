---
phase: quick-260621-iaq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/tests/rocm_parallel_histogram.rs
  - crates/lgbm-compute/tests/rocm_plane_aggregate.rs
  - crates/lgbm-compute/examples/lazy_dispatch_ab.rs
  - crates/lgbm-compute/examples/batched_read_audit_ab.rs
  - crates/lgbm-compute/examples/plane_aggregate_ab.rs
  - crates/lgbm-compute/examples/launch_unchecked_ab.rs
autonomous: false
requirements: []

must_haves:
  truths:
    - "The GPU parity seam RocmBackend::construct_histograms drives the LDS kernel, not the deleted global-atomic kernel"
    - "construct_hist_kernel_atomic_f32 and construct_histograms_parallel_f32_on no longer exist anywhere in the crate"
    - "cargo build and cargo build --features rocm both compile (no dangling refs)"
    - "cargo test --release -p lgbm-compute --features rocm passes on gfx1100"
    - "oracle-harness kernel_parity/learner_parity/boosting_parity + rocm_backend_parity pass within the ~1e-6 gate after the LDS rewire"
  artifacts:
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "RocmBackend::construct_histograms wired to construct_histograms_lds_f32_on"
      contains: "construct_histograms_lds_f32_on"
    - path: "crates/lgbm-compute/src/kernels/histogram.rs"
      provides: "old global-atomic kernel + launcher removed; LDS path is the seam"
  key_links:
    - from: "crates/lgbm-compute/src/lib.rs (RocmBackend::construct_histograms)"
      to: "kernels::histogram::construct_histograms_lds_f32_on"
      via: "direct call"
      pattern: "construct_histograms_lds_f32_on"
---

<objective>
Replace the old GPU global-atomic histogram kernel with the LDS-privatized kernel as
the GPU parity/test seam, delete the now-dead kernel, and fix every dependent so the
workspace compiles with and without `--features rocm` and the parity gate stays green.

Production GPU training already drives the LDS resident build (`build_leaf_histograms_resident_f32_on`
via `build_leaf_histograms_raw`); the global-atomic kernel (`construct_histograms_parallel_f32_on`)
is ONLY reachable through the `Backend::construct_histograms` trait method (lib.rs:1930
RocmBackend impl), whose callers are all parity tests. This task swaps that seam to the
LDS launcher (measured up to 5.26x faster, within the ~1e-6 gate, exact vs naive on
integer data this session) and removes the obsolete kernel + its A/B scaffolding.

Purpose: collapse two divergent f32 GPU accumulation paths to one (the LDS path),
removing dead code and the obsolete benchmarks that only existed to A/B the old kernel.
Output: rewired seam, deleted kernel, retargeted/removed dependents, green parity gate.

CAVEAT (must be honored, not worked around): swapping the seam from global-atomic to
LDS perturbs the seam's f32 accumulation ORDER (see the lib.rs:1943-1954 comment being
removed). If any oracle-harness parity assertion lands on the 1e-6 knife-edge
(cf. DEF-f8u-01: a prior flaky resident-vs-host f32 near-tie), SURFACE it — report the
failing cell + the two values — and do NOT loosen any tolerance. Per CLAUDE.md the
cubecl-cpu f64 anchor is the hard merge gate; the LDS path is held to ~1e-6 best-effort.

HARD CONSTRAINTS — do NOT touch:
- the `Backend::construct_histograms` trait method signature itself (lib.rs:522)
- its CPU f64 anchor impl (CpuBackend, lib.rs:1139) — the bit-exact deterministic gate
- the production resident/LDS build path (`build_leaf_histograms_resident_f32_on`,
  `build_leaf_histograms_raw`)
- the surviving `_plane` kernel + launcher (`construct_hist_kernel_atomic_f32_plane`,
  `construct_histograms_parallel_f32_plane_on`) and the `_lds` kernel/launcher
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@crates/lgbm-compute/src/lib.rs
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm-compute/tests/rocm_parallel_histogram.rs
@crates/lgbm-compute/tests/rocm_plane_aggregate.rs

# Interface facts the executor needs (already gathered — do NOT re-explore):
# - LDS launcher signature is IDENTICAL to the deleted one (drop-in):
#     construct_histograms_lds_f32_on<R>(client, binned: &[u32], grad: &[f32],
#       hess: &[f32], num_bin: u32) -> Result<Vec<f64>, ComputeError>
#   It additionally REJECTS num_bin > 256 (LDS sub-hist cap). The seam's parity
#   callers all use num_bin <= 256, so this is safe; no current caller needs >256.
# - The `_plane` launcher's `use_plane=false` arm is the BYTE-FAITHFUL twin of the
#   deleted kernel (per test `plane_launcher_false_arm_matches_shipped_baseline`),
#   so it is the correct retarget baseline for the surviving plane tests.
# - rocm_backend_parity.rs already constructs `RocmBackend::default()` (DEF-EU9-01
#   resolved) and routes through `.construct_histograms(` — it exercises the rewired seam.
# - oracle-harness parity callers of `.construct_histograms(`:
#     crates/oracle-harness/tests/{kernel_parity,learner_parity,boosting_parity}.rs
#   (they go through the Backend trait, so the rewire flows to them automatically —
#    no edits there, only re-run + verify).
</context>

<tasks>

<task type="auto">
  <name>Task 1: Rewire the seam to LDS and delete the old global-atomic kernel + launcher</name>
  <files>crates/lgbm-compute/src/lib.rs, crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
In crates/lgbm-compute/src/lib.rs RocmBackend::construct_histograms (~1930-1962):
replace the call `kernels::histogram::construct_histograms_parallel_f32_on(...)` (~1955)
with `kernels::histogram::construct_histograms_lds_f32_on(...)` — the args
(client, binned, ordered_gradients, ordered_hessians, num_bin) are unchanged (identical
signature). Replace the stale comment block at ~1938-1954 (the "kt8 ... NOT wired here"
explanation) with a concise note that the GPU parity seam now drives the LDS-privatized
sub-histogram kernel (per-cube LDS atomics then a single global merge): correct vs naive
on integer data, within the ~1e-6 ROCm gate vs the CPU f64 anchor, and faster under
contention; the CPU anchor (CpuBackend) stays the bit-exact hard merge gate. State
explicitly that swapping from global-atomic to LDS changes this seam's f32 accumulation
ORDER (the ~1e-6 best-effort GPU contract; the f64 anchor is unaffected).

In crates/lgbm-compute/src/kernels/histogram.rs:
1. Delete the `#[cube(launch_unchecked)]` kernel `construct_hist_kernel_atomic_f32`
   (~388-404) and its host launcher `construct_histograms_parallel_f32_on` (~415-469).
2. Fix the three rustdoc intra-doc links that point at the now-deleted symbols (they
   would become broken-link rustdoc warnings):
   - ~472: `_plane` VARIANT of [`construct_hist_kernel_atomic_f32`] — reword to plain
     prose (drop the intra-link), e.g. "a plane (warp-aggregated) variant of the
     per-row f32-atomic scatter".
   - ~624: "A near-verbatim copy of [`construct_histograms_parallel_f32_on`]" — reword
     to describe the structure without the dead link (e.g. "the same V5 boundary checks,
     zeroed-f32 alloc, ceil(n/256) cube count, and f32->f64 widen on read-back as the
     LDS launcher").
   - ~748: "The naive [`construct_hist_kernel_atomic_f32`] issues ..." — reword to plain
     prose describing the per-row global-atomic scatter without the dead link.
   Also fix the error-message hint at ~829 inside `construct_histograms_lds_f32_on`
   that says "(use construct_histograms_parallel_f32_on instead)" — change it to no
   longer name the deleted symbol (e.g. "(num_bin > 256 is unsupported by the LDS
   sub-hist path)").
Do NOT touch the `_plane` kernel/launcher bodies, the `_lds` kernel/launcher bodies,
the CpuBackend f64 anchor (lib.rs:1139), the Backend trait method (lib.rs:522), or any
build_leaf_histograms_* resident path.
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/lightgbm_rs && cargo build -p lgbm-compute 2>&1 | tail -5 && cargo build -p lgbm-compute --features rocm 2>&1 | tail -5 && grep -rc "construct_hist_kernel_atomic_f32\b\|construct_histograms_parallel_f32_on" crates/lgbm-compute/src/ | grep -v ':0$' || echo "SRC CLEAN: no refs to deleted symbols"</automated>
  </verify>
  <done>
RocmBackend::construct_histograms calls construct_histograms_lds_f32_on; the old kernel
+ launcher are gone from src/; no rustdoc broken-link refs to the deleted symbols remain
in src/; both `cargo build -p lgbm-compute` and `--features rocm` compile clean. (Tests
in Task 2 may still reference the deleted symbols at this point — that is expected; the
src-only build above does not compile the test/example targets.)
  </done>
</task>

<task type="auto">
  <name>Task 2: Retarget/remove dependents (tests + examples), then run the full rocm parity gate</name>
  <files>crates/lgbm-compute/tests/rocm_parallel_histogram.rs, crates/lgbm-compute/tests/rocm_plane_aggregate.rs, crates/lgbm-compute/examples/lazy_dispatch_ab.rs, crates/lgbm-compute/examples/batched_read_audit_ab.rs, crates/lgbm-compute/examples/plane_aggregate_ab.rs, crates/lgbm-compute/examples/launch_unchecked_ab.rs</files>
  <action>
TESTS — keep the LDS + plane coverage, drop only the now-dead-kernel cases:

tests/rocm_parallel_histogram.rs — remove the import of `construct_histograms_parallel_f32_on`
(~9). DELETE the tests that exclusively exercise the removed kernel:
`parallel_atomic_no_lost_updates_under_contention` (~17), `parallel_within_tolerance_of_cpu_f64_anchor`
(~43), and `parallel_is_faster_than_single_unit_on_gpu` (~182, A/Bs the removed kernel
vs the single-unit kernel). KEEP all LDS tests (`lds_no_lost_updates_under_contention`,
`lds_within_tolerance_of_cpu_f64_anchor`, `bench_lds_vs_naive_atomic_large`). For the two
that still reference the removed launcher as a baseline:
- `lds_equals_naive_atomic_on_integer_data` (~123): retarget the "naive" baseline to
  `construct_histograms_parallel_f32_plane_on(&gc, ..., num_bin, false)` (the byte-faithful
  twin of the removed kernel on integer data); update the import to bring that symbol in.
  Keep the exact-equality assertion (both are deterministic on integer data).
- `bench_lds_vs_naive_atomic_large` (~143): retarget the two `construct_histograms_parallel_f32_on`
  calls (warmup ~158, timed ~163) to `construct_histograms_parallel_f32_plane_on(..., false)`
  so the bench still A/Bs LDS vs the per-row global-atomic baseline. (It asserts nothing,
  just prints — keep it.)

tests/rocm_plane_aggregate.rs — remove the `construct_histograms_parallel_f32_on` import
(~15). Retarget its baseline uses to the surviving `_plane` launcher's false arm:
- `plane_large_leaf_drift_not_worse_than_baseline` (~125 `base = ...`) ->
  `construct_histograms_parallel_f32_plane_on(&gc, ..., num_bin, false)`.
- `plane_equals_baseline_on_integer_data` (~165 `baseline = ...`) -> same false-arm call.
- `plane_launcher_false_arm_matches_shipped_baseline` (~178): this test compared the
  plane false arm to the removed kernel; with the kernel gone it would compare the false
  arm to itself. DELETE this test (it is now degenerate). The false arm's correctness is
  already covered by `plane_equals_baseline_on_integer_data` (false vs true on integer
  data) and the LDS integer-equality test.

EXAMPLES — these exist solely to A/B the removed kernel; DELETE the files outright:
- examples/lazy_dispatch_ab.rs (A/Bs construct_hist_kernel_atomic_f32 immediate-vs-deferred read)
- examples/batched_read_audit_ab.rs (audits construct_hist_kernel_atomic_f32 batched read)
- examples/plane_aggregate_ab.rs (A/Bs the per-row atomic vs the plane variant — its
  non-plane arm is the removed kernel; the plane variant's correctness is covered by
  rocm_plane_aggregate.rs tests, so the bench example has no independent value left)
- examples/launch_unchecked_ab.rs (A/Bs construct_hist_kernel_atomic_f32 checked vs
  unchecked launch — the lever it measured is fully resolved per nrw/ol8 in STATE)
Use `rm` for each. (STATE confirms these are measurement-only with zero production callers.)

THEN run the gate (Task verify below).
  </action>
  <verify>
    <automated>cd /home/user/Documents/workspace/lightgbm_rs && grep -v '^#' -r crates/lgbm-compute/tests/ crates/lgbm-compute/examples/ | grep -c "construct_hist_kernel_atomic_f32\b\|construct_histograms_parallel_f32_on" | grep -qx 0 && echo "DEP CLEAN" && cargo build -p lgbm-compute --features rocm --tests --examples 2>&1 | tail -5 && cargo test --release -p lgbm-compute --features rocm 2>&1 | tail -30</automated>
  </verify>
  <done>
No test/example references the deleted symbols (DEP CLEAN); the four obsolete A/B example
files are removed; `cargo build -p lgbm-compute --features rocm --tests --examples`
compiles clean; `cargo test --release -p lgbm-compute --features rocm` passes on gfx1100
with the LDS + plane coverage intact. Any 1e-6 knife-edge failure is SURFACED with cell +
values, not silenced by widening a tolerance.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>
The GPU parity seam (RocmBackend::construct_histograms) now drives the LDS kernel instead
of the deleted global-atomic kernel. Because this changes the seam's f32 accumulation
order, the oracle-harness end-to-end parity tests must be re-run and confirmed green.
  </what-built>
  <how-to-verify>
Run the oracle-harness parity tests that route through the rewired seam, on gfx1100:

  cd /home/user/Documents/workspace/lightgbm_rs
  cargo test --release -p oracle-harness --features rocm kernel_parity 2>&1 | tail -20
  cargo test --release -p oracle-harness --features rocm learner_parity 2>&1 | tail -20
  cargo test --release -p oracle-harness --features rocm boosting_parity 2>&1 | tail -20
  cargo test --release -p lgbm-compute --features rocm rocm_backend_parity 2>&1 | tail -20

(Use the project's actual rocm feature/test invocation if the harness gates hip cells
differently — match how STATE describes running hip kernel_parity/learner_parity.)

Expected: all pass within the ~1e-6 ROCm gate (the CpuBackend f64 anchor cells stay
BIT-EXACT — they were not touched). If ANY cell fails on the 1e-6 knife-edge (cf.
DEF-f8u-01), the executor must report the failing cell + both values here and STOP —
do NOT loosen any tolerance to make it pass. The f64 anchor is the hard gate; the LDS
path is ~1e-6 best-effort, and a genuine knife-edge regression is a finding to surface,
not to mask.
  </how-to-verify>
  <resume-signal>Type "approved" once all four parity suites pass, or paste the failing cell + values if a knife-edge regression appears.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host to GPU kernel launch | binned/grad/hess slices uploaded to the device; the LDS launcher V5-validates lengths + bin range and rejects num_bin > 256 before any unsafe launch_unchecked |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-iaq-01 | Tampering | rewired construct_histograms seam (f32 order change) | mitigate | re-run oracle-harness kernel/learner/boosting parity + rocm_backend_parity; surface any 1e-6 knife-edge cell, never widen tolerance (Task 3 blocking checkpoint) |
| T-iaq-02 | Denial of Service | construct_histograms_lds_f32_on num_bin > 256 reject | accept | seam callers all use num_bin <= 256; the launcher already returns ComputeError::Runtime for >256 (no panic, no OOB) |
| T-iaq-03 | Information Disclosure | launch_unchecked OOB if validation skipped | accept | the LDS launcher retains the same host-side V5 length/bin-range checks that discharge the launch_unchecked obligations (unchanged code) |
</threat_model>

<verification>
- `cargo build -p lgbm-compute` and `cargo build -p lgbm-compute --features rocm` compile.
- No reference to `construct_hist_kernel_atomic_f32` / `construct_histograms_parallel_f32_on`
  remains anywhere in `crates/lgbm-compute/` (src, tests, examples).
- `cargo test --release -p lgbm-compute --features rocm` passes on gfx1100.
- oracle-harness kernel_parity / learner_parity / boosting_parity + rocm_backend_parity
  pass within the ~1e-6 gate (CpuBackend f64 anchor cells bit-exact, untouched).
</verification>

<success_criteria>
- RocmBackend::construct_histograms drives construct_histograms_lds_f32_on.
- The old global-atomic kernel + launcher are deleted; all dependents fixed or removed.
- LDS + plane test coverage preserved (no silent loss of GPU correctness coverage).
- Both builds (default and --features rocm) compile; full rocm test suite green on gfx1100.
- Parity gate green; any 1e-6 knife-edge surfaced (cell + values), never masked.
</success_criteria>

<output>
Create `.planning/quick/260621-iaq-replace-old-gpu-global-atomic-histogram-/260621-iaq-SUMMARY.md` when done.
</output>
