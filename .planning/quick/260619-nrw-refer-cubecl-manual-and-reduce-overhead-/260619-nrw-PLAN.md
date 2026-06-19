---
phase: quick-260619-nrw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/tests/rocm_parallel_histogram.rs
autonomous: true
requirements: [NRW-01, NRW-03]
must_haves:
  truths:
    - "Every rocm-gated production histogram kernel launches via ::launch_unchecked (no in-kernel bounds-check codegen)"
    - "The two f64-deterministic kernels (fix_compact, build_fix_scan_fused) stay bit-exact to the CPU f64 anchor after the switch"
    - "Every f32 production kernel still matches the CPU f64 anchor within ABS 5e-6 / REL 1e-5 (GPU-vs-CPU-f64-anchor, never GPU-vs-GPU)"
    - "The CPU f64 anchor kernels (construct_hist_kernel f64, construct_hist_kernel_f32) are byte-unchanged"
    - "CPU-only build and --features rocm build both compile"
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/histogram.rs"
      provides: "All 8 production launchers switched to launch_unchecked with per-access SAFETY enumerations"
      contains: "launch_unchecked"
    - path: "crates/lgbm-compute/tests/rocm_parallel_histogram.rs"
      provides: "GPU-vs-CPU-f64-anchor re-pin coverage for the swept production kernels"
      contains: "anchor"
  key_links:
    - from: "construct_leaf_hist_resident_lds_kernel (wired production path, :873)"
      to: "resident_raw_build_into LDS launcher (:1403)"
      via: "::launch_unchecked switch + extended SAFETY enumeration"
      pattern: "construct_leaf_hist_resident_lds_kernel::launch_unchecked"
    - from: "swept production kernels"
      to: "CPU f64 anchor"
      via: "rocm parity tests (ABS 5e-6 / REL 1e-5, bounded leaf subset)"
      pattern: "assert_close|anchor"
---

<objective>
Extend the proven mwr `launch_unchecked` lever — currently applied ONLY to the
out-of-scope CUDA-mirror primitive — to ALL eight `#[cfg(feature="rocm")]`-gated
**production** histogram kernels in `histogram.rs`, dropping the per-access
in-kernel bounds-check codegen the `#[cube(launch)]` macro emits in their scatter
hot loops. The host-side V5 validation already present in every launcher discharges
the `launch_unchecked` contract; the only new artifact per kernel is the per-access
SAFETY enumeration (copy the mirror's worked template at :1174-1189). Every changed
kernel is re-pinned to the CPU f64 anchor.

Purpose: the production training path (`construct_leaf_hist_resident_lds_kernel`,
wired at :1403) and the other rocm kernels never received the codegen-overhead cut
mwr proved on the mirror. This is the numerics-preserving, in-repo-proven win the
research names as the spine of the work.

Output: 8 production launchers on `::launch_unchecked` with enumerated SAFETY
contracts; GPU-vs-CPU-f64-anchor parity re-pinned and green on gfx1100; both build
configurations compile.

SCOPE NOTE (honest, per research + CONTEXT):
- `#[comptime]` specialization (NRW-02) is INTENTIONALLY NOT pursued. Research §Lever-2
  + Pitfall-3 show the only material candidate (bin-count `lds_len`/`feat_len`) would
  re-introduce the multi-binary cost the repo deliberately avoids (histogram.rs:458-463),
  and the remaining run-constant scalar (`num_data` stride) gates no GPU-side branch —
  comptime-ing it yields no measurable win. Adding it would be padding. Dropped, not deferred.
- Stage-3 order-changing restructure is NOT pursued: ngo's A/B already showed the wired
  LDS kernel is at/near optimal; CONTEXT permits restructuring but the research data says
  there is no win regime. Not invented just because it is permitted.
- The two CPU f64 anchor kernels `construct_hist_kernel` (:84, the merge-gate f64 fold)
  and `construct_hist_kernel_f32` (:105) are EXCLUDED — they run on the `ActiveRuntime`/
  cubecl-cpu merge gate, not the rocm path, and the constraint mandates their numerics
  stay UNTOUCHED. (launch_unchecked is numerics-preserving so it would be safe, but they
  carry the bit-exact merge gate and are not the rocm-overhead target this task scopes.)
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/quick/260619-nrw-refer-cubecl-manual-and-reduce-overhead-/260619-nrw-CONTEXT.md
@.planning/quick/260619-nrw-refer-cubecl-manual-and-reduce-overhead-/260619-nrw-RESEARCH.md
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm-compute/tests/rocm_cuda_mirror.rs
@crates/lgbm-compute/tests/rocm_parallel_histogram.rs

# Mirror SAFETY-contract template to copy: histogram.rs:1174-1203 (already in file above).
# DEF-MWR-01 landmine: full-corpus near-zero-grad f32-atomic cancellation, |diff|~8.7e-6 —
#   pre-existing, NOT a regression. launch_unchecked CANNOT change accumulation order, so any
#   parity movement on a near-zero-grad cell after the sweep is THIS, not the switch.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Sweep the two f64-deterministic kernels + the wired LDS production path to launch_unchecked</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
Switch the SAFEST and HIGHEST-VALUE kernels first, per RESEARCH §Recommended-Ordering 1(a)+1(b).

(1) f64 deterministic kernels — ZERO numeric risk (bit-exact, one cube per feature, ascending fold):
  - `fix_compact_kernel` (:1476): change `#[cube(launch)]` → `#[cube(launch_unchecked)]`. At BOTH
    launch sites (:1657 and :1808) change `fix_compact_kernel::launch(` → `::launch_unchecked(`
    (both are already inside `unsafe { }` blocks).
  - `build_fix_scan_fused_kernel` (:1913): same attribute change; launcher at :2255
    `build_fix_scan_fused_kernel::launch(` → `::launch_unchecked(`.

(2) The WIRED production training path:
  - `construct_leaf_hist_resident_lds_kernel` (:873): attribute → `#[cube(launch_unchecked)]`;
    launcher in `resident_raw_build_into` LDS branch (:1403)
    `construct_leaf_hist_resident_lds_kernel::launch(` → `::launch_unchecked(`.

For EACH of the three kernels, extend the EXISTING SAFETY comment above its `unsafe` block with a
per-device-access enumeration, copying the mirror template at :1174-1189 verbatim in style. Cite the
exact device accesses the RESEARCH kernel-table already enumerated:
  - fix_compact: `h_raw`, `h_hist`, and the per-feature `slot/numbin/offset/mfb[f]` arrays (all len `n`,
    `f < n` from the `CubeCount` of one cube per feature).
  - fused: `resident_bins[col+leaf_rows[k]]`, `leaf_rows[k] < num_data`, `ord_g/ord_h[k]`, `h_hist`,
    `h_out[f*12..]`, index arrays len `n`.
  - resident_lds: `resident_bins[col + leaf_rows[k]]` (`col = f*num_data`, `leaf_rows[k] < num_data`),
    `slot_off[f]`/`slot_off[f+1]`, the LDS `sub[bin*2+1]` within `HIST_LDS_MAX` (num_bin<=256),
    `out[base+m]` for `m < feat_len`.
Each enumeration must end with the mwr clause: "the host-side V5 checks discharge exactly the
obligations the launch_unchecked contract requires, and the launch does NOT change numerics — only
bounds-check codegen is removed; scatter order / f32-atomic accumulation is identical." Do NOT change
the kernel bodies, CubeDim, CubeCount, or any accumulation/scatter logic — only the two tokens per
kernel + the SAFETY prose. The f64 anchor kernels at :84 and :105 MUST stay `#[cube(launch)]` (untouched).
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute</automated>
    <automated>cargo build -p lgbm-compute --features rocm</automated>
    <automated>grep -n 'construct_leaf_hist_resident_lds_kernel::launch_unchecked\|fix_compact_kernel::launch_unchecked\|build_fix_scan_fused_kernel::launch_unchecked' crates/lgbm-compute/src/kernels/histogram.rs | grep -vc '^#'</automated>
    <automated>grep -c '^#\[cube(launch)\]' crates/lgbm-compute/src/kernels/histogram.rs</automated>
    <human-check>grep result for the three ::launch_unchecked names shows >=4 sites (fix_compact has 2); `construct_hist_kernel` (:84) and `construct_hist_kernel_f32` (:105) remain `#[cube(launch)]`</human-check>
  </verify>
  <done>
Both builds (CPU-only and --features rocm) compile. `fix_compact_kernel`,
`build_fix_scan_fused_kernel`, and `construct_leaf_hist_resident_lds_kernel` carry
`#[cube(launch_unchecked)]` and all their launch sites call `::launch_unchecked`, each
with an extended per-access SAFETY enumeration. The two CPU f64 anchor kernels remain
`#[cube(launch)]`, byte-unchanged.
  </done>
</task>

<task type="auto">
  <name>Task 2: Sweep the remaining atomic / lds / batched / resident kernels to launch_unchecked</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
Per RESEARCH §Recommended-Ordering 1(c), switch the five remaining rocm-gated f32 production kernels.
For each: `#[cube(launch)]` → `#[cube(launch_unchecked)]` on the kernel, `::launch` → `::launch_unchecked`
at its launcher (all already inside `unsafe { }`), and extend the existing SAFETY comment with the
per-access enumeration from the RESEARCH kernel-table:
  - `construct_hist_kernel_atomic_f32` (:389) — launcher `construct_histograms_parallel_f32_on` (:443):
    accesses `binned[idx]`, `grad/hess[idx]` (idx<num_data), `out[bin*2+1]` (bin<num_bin).
  - `construct_hist_kernel_lds_f32` (:522) — launcher `construct_histograms_lds_f32_on` (:618):
    `binned[i]`, `grad/hess[i]`, LDS `sub[ti+1]` within HIST_LDS_MAX, `out[m]` (m<2*num_bin).
  - `construct_leaf_hist_batched_lds_kernel` (:924) — launcher batched-LDS branch (:721):
    `gathered_bins[fbase+k]`, `ord_g/ord_h[k]`, `slot_off[f]`/`slot_off[f+1]`, LDS sub, `out[base+m]`.
  - `construct_leaf_hist_batched_kernel` (:651) — launcher batched-naive branch (:743):
    `gathered_bins[idx]`, `ord_g/ord_h[k]`, `slot_off[f]`, `out[cell+1]`.
  - `construct_leaf_hist_resident_kernel` (:772) — launcher resident-naive branch (:1426):
    `resident_bins[f*num_data+row]`, `leaf_rows[k]<num_data`, `ord_g/ord_h[k]`, `slot_off[f]`, `out[cell+1]`.
Each enumeration ends with the same mwr numerics-unchanged clause as Task 1. Kernel bodies, CubeDim,
CubeCount, and all scatter/atomic logic stay byte-identical — only the two tokens + SAFETY prose change.
After this task NO `#[cfg(feature="rocm")]`-gated production kernel uses `#[cube(launch)]`; the only
remaining `#[cube(launch)]` kernels are the two CPU f64 anchor kernels (:84, :105) plus any non-kernel.
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute</automated>
    <automated>cargo build -p lgbm-compute --features rocm</automated>
    <automated>grep -n 'construct_hist_kernel_atomic_f32::launch_unchecked\|construct_hist_kernel_lds_f32::launch_unchecked\|construct_leaf_hist_batched_lds_kernel::launch_unchecked\|construct_leaf_hist_batched_kernel::launch_unchecked\|construct_leaf_hist_resident_kernel::launch_unchecked' crates/lgbm-compute/src/kernels/histogram.rs | grep -vc '^#'</automated>
    <automated>grep -v '^#' crates/lgbm-compute/src/kernels/histogram.rs | grep -c '::launch('</automated>
    <human-check>Only the two CPU-anchor launch sites (`construct_hist_kernel::launch` :177, `construct_hist_kernel_f32::launch` :362) remain as `::launch(`; all 8 rocm production launchers use `::launch_unchecked`</human-check>
  </verify>
  <done>
Both builds compile. All five remaining rocm-gated production kernels carry
`#[cube(launch_unchecked)]` and launch via `::launch_unchecked` with extended SAFETY
enumerations. The only surviving `::launch(` sites are the two CPU f64 anchor launchers
(:177, :362). cargo clippy clean on the edited file.
  </done>
</task>

<task type="auto">
  <name>Task 3: Re-pin every swept kernel GPU-vs-CPU-f64-anchor on gfx1100 and document residuals</name>
  <files>crates/lgbm-compute/tests/rocm_parallel_histogram.rs, crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
Re-pin each swept kernel to the CPU f64 anchor (NEVER GPU-vs-GPU — DEF-f8u-01), reusing the
`rocm_cuda_mirror.rs` pattern: build a small corpus, gather the bounded leaf subset
`(7..num_data).step_by(3)` the stable tests use (NOT the full corpus — that triggers the
DEF-MWR-01 near-zero-grad flake), compute the CPU f64 anchor, and assert against each kernel's
output with `assert_close` (ABS 5e-6 / REL 1e-5). The f64 deterministic kernels (fix_compact,
fused) must assert BIT-EXACT (`compare_exact_f64_bits` / `f64::to_bits`), not just within-tol.

Audit the existing rocm parity tests (`rocm_parallel_histogram.rs`, `rocm_row_partition.rs`,
`rocm_cuda_mirror.rs`) FIRST: many production kernels are already covered (the atomic/lds parallel
path in `rocm_parallel_histogram.rs`, the resident path elsewhere). For any swept kernel WITHOUT an
existing anchor pin, add a `#[cfg(feature="rocm")]` test that exercises its launcher and asserts
GPU-vs-CPU-f64-anchor. Do not duplicate coverage that already exists — extend, don't bloat.

Run the full rocm parity suite on gfx1100. If any cell exceeds ABS 5e-6 / REL 1e-5:
  - FIRST distinguish DEF-MWR-01: if it is a full-corpus near-zero-grad cell, confirm it reproduces
    on HEAD before the sweep (revert-and-rerun) and document it as pre-existing, NOT a regression.
  - launch_unchecked CANNOT change accumulation order, so a within-bounded-subset cell that moves
    is a real finding: document the residual in a comment + the SUMMARY and FLAG the tolerance review
    to the user — do NOT silently widen the gate.
Capture warm medians per the cold-ceiling rule ONLY if reporting any timing (3 warm-ups discarded,
median of >=7); the numeric re-pin is the gate, timing is secondary and may be reported as "modest /
launch-overhead-only" per the research (transfer-bound at these sizes — measure, do not over-claim).
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute --features rocm --tests</automated>
    <automated>cargo test -p lgbm-compute --features rocm --test rocm_parallel_histogram</automated>
    <automated>cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror</automated>
    <automated>cargo test -p lgbm-compute --features rocm --test rocm_row_partition</automated>
    <automated>cargo clippy -p lgbm-compute --features rocm --tests 2>&1 | grep -c 'warning: ' </automated>
    <human-check>On gfx1100 every swept kernel matches the CPU f64 anchor within ABS 5e-6 / REL 1e-5 (f64 kernels bit-exact); any out-of-tol cell is explicitly attributed to DEF-MWR-01 (pre-existing, full-corpus near-zero-grad) and not to the launch_unchecked switch, or FLAGGED for tolerance review</human-check>
  </verify>
  <done>
The full rocm parity suite is green on gfx1100 with every swept production kernel
re-pinned GPU-vs-CPU-f64-anchor (never GPU-vs-GPU). The two f64 kernels are bit-exact;
the f32 kernels are within ABS 5e-6 / REL 1e-5. Any residual is documented and attributed
to DEF-MWR-01 or flagged for tolerance review — no gate silently weakened. CPU-only test
build still compiles (rocm code stays `#[cfg(feature="rocm")]`).
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host launcher → GPU kernel (`launch_unchecked`) | The `unsafe` launch removes in-kernel bounds checks; every device array index must be host-proven in range BEFORE upload, or a malformed input is UB on device. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-NRW-01 | Tampering | `launch_unchecked` SAFETY contract incompleteness — a device access not enumerated in the SAFETY prose could escape the host V5 proof | mitigate | Per-kernel enumeration copied from the mirror template (:1174-1189), checked line-by-line against the kernel body; every launcher already runs full V5 validation (no new validation logic, the bound proofs already exist) — RESEARCH Pitfall 2 |
| T-NRW-02 | Information Disclosure / Elevation | Reading/writing out of `resident_bins`/`out` past the allocation under a malformed leaf_rows or bin value with checks removed | mitigate | The existing V5 boundary validation (`validate_histogram_inputs`, per-feature bin-range, `leaf_rows[k] < num_data`) is the discharge cited in each SAFETY enumeration; unchanged from the `::launch` path, only codegen differs |
| T-NRW-03 | Repudiation | A parity movement misattributed to the switch vs the pre-existing DEF-MWR-01 flake | accept (documented) | Bounded leaf subset `(7..num_data).step_by(3)` in tests avoids the near-zero-grad full-corpus cells; any full-corpus trip is revert-and-reproduced on HEAD and logged as pre-existing — RESEARCH Pitfall 1 |
</threat_model>

<verification>
- `cargo build -p lgbm-compute` (CPU-only) and `cargo build -p lgbm-compute --features rocm` both compile.
- No `#[cfg(feature="rocm")]`-gated production kernel uses `#[cube(launch)]`; the only `::launch(`
  sites are the two CPU f64 anchor launchers (:177, :362).
- The full rocm parity suite (`rocm_parallel_histogram`, `rocm_cuda_mirror`, `rocm_row_partition`)
  is green on gfx1100; f64 kernels bit-exact, f32 within ABS 5e-6 / REL 1e-5.
- clippy clean on the edited file under both feature sets.
- CPU f64 anchor kernels byte-unchanged (the merge-gate `cargo test -p lgbm-compute` f64-fold
  determinism tests still pass — not regressed).
</verification>

<success_criteria>
- All 8 rocm-gated production histogram kernels launch via `::launch_unchecked` with complete,
  enumerated SAFETY contracts (NRW-01).
- Every swept kernel re-pinned GPU-vs-CPU-f64-anchor on gfx1100, within the existing envelope;
  residuals documented, any tolerance movement flagged not silently widened (NRW-03).
- comptime (NRW-02) and Stage-3 restructure explicitly dropped with the research rationale recorded
  (no padding).
- Both build configurations compile; CPU f64 anchor numerics untouched; rocm code stays gated.
</success_criteria>

<output>
Create `.planning/quick/260619-nrw-refer-cubecl-manual-and-reduce-overhead-/260619-nrw-SUMMARY.md` when done.
</output>
