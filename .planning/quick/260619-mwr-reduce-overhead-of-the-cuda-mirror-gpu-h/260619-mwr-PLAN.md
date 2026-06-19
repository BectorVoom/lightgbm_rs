---
phase: quick-260619-mwr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/examples/cuda_mirror_overhead.rs
autonomous: true
requirements: [MWR-01, MWR-02, MWR-03]

must_haves:
  truths:
    - "The CUDA-mirror histogram kernel launches via launch_unchecked (no per-access bounds-check codegen), with the V5 boundary validation kept BEFORE upload."
    - "A resident-Handle variant of the mirror launcher exists that uploads the full feature-major bin buffer ONCE and re-uploads only per-leaf data_indices (+ per-iter grad/hess) on each call."
    - "The rocm_cuda_mirror parity test stays GREEN (within the f32-atomic envelope vs the CPU f64 anchor) for BOTH launch paths — pinned GPU-vs-CPU-anchor, never GPU-vs-GPU."
    - "A warmed-up rocm-gated micro-bench reports before/after medians, separating the launch-overhead win from the transfer-overhead win."
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/histogram.rs"
      provides: "launch_unchecked mirror kernel + resident-Handle mirror launcher variant"
      contains: "launch_unchecked"
    - path: "crates/lgbm-compute/examples/cuda_mirror_overhead.rs"
      provides: "warmed-up before/after micro-bench for the mirror kernel"
      contains: "rocm"
    - path: "crates/lgbm-compute/tests/rocm_cuda_mirror.rs"
      provides: "parity coverage for both the per-call and resident-Handle launchers"
      contains: "resident"
  key_links:
    - from: "crates/lgbm-compute/examples/cuda_mirror_overhead.rs"
      to: "construct_histograms_cuda_mirror_on (+ resident variant)"
      via: "rocm_client launch with warm-up and median timing"
      pattern: "construct_histograms_cuda_mirror"
---

<objective>
Reduce the non-compute overhead of the CUDA-mirror GPU histogram kernel
(`construct_hist_cuda_mirror_kernel` / `construct_histograms_cuda_mirror_on` in
`crates/lgbm-compute/src/kernels/histogram.rs`) using two cubecl-manual levers, and
PROVE the win with a warmed-up before/after micro-bench on gfx1100.

Two real overheads (per the task analysis), each fixed independently so each lever's
effect is attributable:
1. **Redundant per-access bounds checks** — the kernel uses `#[cube(launch)]` + `::launch`,
   which emits in-kernel bounds-check codegen in the hot scatter loop. The launcher
   ALREADY runs full V5 validation (bin-range + lengths + data-index range) before
   upload and confines unsafe (CMP-01), so the in-kernel checks are pure redundant
   overhead → switch to `launch_unchecked` (MWR-01).
2. **Per-call full re-upload of the resident bin buffer** — the per-call launcher
   `create_from_slice`s the FULL `num_features * num_data` bin buffer (and full-corpus
   grad/hess) on EVERY call. CUDA keeps bins RESIDENT (uploaded once per train). Add a
   Handle-accepting variant that takes a pre-uploaded device `Handle` so the big bin
   buffer moves once and only the small per-leaf `data_indices` (+ per-iter grad/hess)
   re-upload — mirroring the `build_leaf_histograms_resident_f32_on` / `ResidentBins`
   pattern already in this file and `lib.rs` (MWR-02).

Purpose: make the tested CUDA-mirror primitive faithful to the CUDA upload-once model
and remove redundant launch codegen, with real measured figures (MWR-03) instead of a
claim. The mirror stays a rocm-gated PRIMITIVE — do NOT wire it into the production
histogram path, and do NOT touch the CPU f64 bit-exact anchor.

Output: an updated mirror kernel + a new resident-Handle launcher + extended parity
coverage + a warmed-up before/after micro-bench whose numbers land in the SUMMARY.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

# The target kernel + per-call launcher (switch to launch_unchecked; add resident variant)
@crates/lgbm-compute/src/kernels/histogram.rs

# The parity test that must stay green (~1e-6 vs CPU f64 anchor; never GPU-vs-GPU)
@crates/lgbm-compute/tests/rocm_cuda_mirror.rs

# Resident-Handle precedent for the upload-once lever (ResidentBins, upload_resident_bins)
@crates/lgbm-compute/src/lib.rs

# Warmed-up rocm-gated micro-bench precedent (warm-up + median, launch into freshly-zeroed out)
@crates/lgbm-compute/examples/gpu_row_partition.rs

# Load-bearing benchmark rule: cold ceiling overstates warm 3-7×; warm up before timing.
# Plane reductions do NOT apply to data-dependent histogram scatter — keep atomics.
@.claude/skills/spike-findings-lightgbm_rs/SKILL.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Switch the mirror kernel to launch_unchecked + add a resident-Handle launcher variant</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
Two changes to the CUDA-mirror code, both rocm-gated, numerics-preserving (same atomic
order / contention model — `launch_unchecked` only removes in-kernel bounds-check
codegen, it does NOT change the scatter or accumulation).

(MWR-01) Change the kernel attribute on `construct_hist_cuda_mirror_kernel` (~line 1007)
from `#[cube(launch)]` to `#[cube(launch_unchecked)]`. In the existing per-call launcher
`construct_histograms_cuda_mirror_on` (~line 1091), change the call site (~line 1174)
from `construct_hist_cuda_mirror_kernel::launch(...)` to
`construct_hist_cuda_mirror_kernel::launch_unchecked(...)`. The call is already inside an
`unsafe` block with the SAFETY comment — extend that SAFETY comment to state that the
V5 boundary validation above (bin-range per feature, every `data_indices[k] < num_data`,
length checks) is what makes dropping the in-kernel bounds checks sound: every device
access (`data[col + data_index]`, `grad[data_index]`, `out[base + m]`) is provably in
range from the host-side checks, exactly as the manual's launch_unchecked contract
requires. Do NOT change CubeCount/CubeDim/argument order — only the method name.

(MWR-02) Add a NEW resident-Handle launcher variant alongside the per-call one. Model it
on `build_leaf_histograms_resident_f32_on` (~line 811) for the Handle-accepting shape and
on `construct_histograms_cuda_mirror_on` for the per-feature V5 validation + result
widening. Suggested signature:
`construct_histograms_cuda_mirror_resident_on<R: cubecl::Runtime>(client, resident_bins: cubecl::server::Handle, num_data, num_features, data_indices: &[u32], grad: &[f32], hess: &[f32], slot_off: &[usize], slot_len, num_bin) -> Result<Vec<f64>, ComputeError>`.
The caller is responsible for having uploaded the feature-major bin buffer ONCE (length
`num_features * num_data`) and passing its `Handle` — this variant must NOT re-upload the
bin buffer. Per call it uploads ONLY `data_indices`, `grad`, `hess`, the sentinel
`slot_off`, and the zeroed `out`. Because the bins are not on the host here, it CANNOT run
the per-feature bin-range scan that the per-call variant does (that scan reads `data[...]`
from the host slice) — instead validate what it CAN: `grad.len()==num_data`,
`hess.len()==num_data`, `slot_off.len()==num_features`, `num_bin <= 256`, every
`data_indices[k] < num_data`, and `num_features != 0`. Early-return zeros on an empty leaf.
Then `launch_unchecked` the SAME `construct_hist_cuda_mirror_kernel` with the passed
`resident_bins` Handle as the `data` arg (reuse `row_partition_count`, `slot_off_sentinel`,
and the `CubeCount::Static(num_features, p, 1)` / `CubeDim::new_1d(256)` config identical to
the per-call launcher). Add a SAFETY comment noting the bin-range invariant is the CALLER's
responsibility (the resident buffer was validated at upload time) — mirror the wording in
`build_leaf_histograms_resident_f32_on`. Keep the f32→f64 widening on read-back. Add a
clear doc comment that this is the upload-once CUDA-faithful variant and is a tested
primitive, NOT wired into production. NEVER git-add the LightGBM reference trees.
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute --features rocm 2>&1 | tail -20</automated>
  </verify>
  <done>
`construct_hist_cuda_mirror_kernel` uses `#[cube(launch_unchecked)]`; both the per-call
launcher and the new `construct_histograms_cuda_mirror_resident_on` call
`::launch_unchecked` inside `unsafe` with updated SAFETY comments. The resident variant
accepts a `cubecl::server::Handle`, re-uploads only per-leaf data, and the crate compiles
under `--features rocm`. CPU-only build is unaffected (mirror code stays `#[cfg(feature = "rocm")]`).
  </done>
</task>

<task type="auto">
  <name>Task 2: Extend the parity test to cover the resident-Handle variant (GPU-vs-CPU-anchor)</name>
  <files>crates/lgbm-compute/tests/rocm_cuda_mirror.rs</files>
  <action>
Add ONE new `#[test]` (rocm-gated, the file is already `#![cfg(feature = "rocm")]`) that
exercises `construct_histograms_cuda_mirror_resident_on` and asserts it against the SAME
`cpu_anchor` used by the existing tests — pinning GPU-vs-CPU-f64-anchor, NEVER GPU-vs-GPU
(memory DEF-f8u-01). Reuse `make_corpus`, the existing `assert_close` (ABS 5e-6 / REL 1e-5
f32-atomic envelope), and a non-trivial `data_indices` leaf subset (e.g. the same
`(7..num_data).step_by(3)` subset as `cuda_mirror_dense_matches_cpu_anchor_within_tol`).
Upload the corpus's feature-major `resident` buffer ONCE via the rocm client
(`client.create_from_slice(u32::as_bytes(&corpus.resident))`, with `use cubecl::prelude::CubeElement;`
in scope as the launchers do), then call the resident variant with that `Handle`. Assert
the result matches `cpu_anchor(&corpus, &data_indices)` within `assert_close`.

This proves the upload-once path produces the same histogram as the per-call path and the
CPU anchor — i.e. switching to `launch_unchecked` and to a resident Handle did NOT change
numerics. Do NOT weaken the tolerance or change the existing three tests. (The existing
per-call tests already cover the `launch_unchecked` switch from Task 1, since they call
`construct_histograms_cuda_mirror_on` unchanged.)
  </action>
  <verify>
    <automated>cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror 2>&1 | tail -25</automated>
  </verify>
  <done>
All rocm_cuda_mirror tests pass on gfx1100, including the new resident-Handle test, within
the f32-atomic envelope vs the CPU f64 anchor. No existing test or tolerance changed.
  </done>
</task>

<task type="auto">
  <name>Task 3: Add a warmed-up before/after micro-bench and record real figures</name>
  <files>crates/lgbm-compute/examples/cuda_mirror_overhead.rs</files>
  <action>
Create a new rocm-gated example `cuda_mirror_overhead.rs` modeled on
`examples/gpu_row_partition.rs` (the `#[cfg(not(feature = "rocm"))]` stub `main` that prints
"requires --features rocm" + the `#[cfg(feature = "rocm")]` real `main`). Honor the
load-bearing warm-vs-cold rule: run >=2 discarded warm-up launches per variant before
timing, take the MEDIAN of >=5 timed launches, and force device sync by reading back `out`
(or the final accumulated buffer) so the timer captures real device work, not just queue
submission.

Build a realistic corpus per `<measurement_requirement>`: 50 features, ~200k rows, run the
sweep at 16 / 64 / 256 bins, and a LARGE leaf subset (e.g. ~half to all rows so the row
loop dominates). Generate deterministic well-spread bins (no RNG; the same hash style as the
existing example/test). Measure THREE configurations so each lever is attributable and the
before/after is clean:
  (A) BEFORE — the per-call launcher `construct_histograms_cuda_mirror_on`, which both
      re-uploads the full bin buffer per call AND (now) uses launch_unchecked. To isolate
      the LAUNCH-overhead win alone, ALSO time a checked baseline: temporarily this is hard
      since the kernel attribute changed — instead frame (A) as "per-call upload-each-time"
      and attribute the launch win via the cubecl-manual rationale + the (B)-vs-(C) split.
  (B) per-call upload path (full re-upload every launch) — the transfer-heavy path.
  (C) AFTER — `construct_histograms_cuda_mirror_resident_on` with the bin buffer uploaded
      ONCE before the timed loop (only per-leaf data_indices/grad/hess re-upload inside).
Report (B) vs (C) as the TRANSFER-overhead win (this is the dominant non-compute overhead),
and note the LAUNCH-overhead win (launch_unchecked) qualitatively from the manual rationale
plus any measured delta. Print a clear table: bins × {per-call median ms, resident median
ms, speedup×, MB transferred per call}. Compute MB-per-call for each path
(per-call moves `num_features*num_data*4` bytes of bins; resident moves only
`data_indices.len()*4 + 2*num_data*4`) so the transfer reduction is quantified, not just
timed.

The example must NOT be wired into any test gate; it is a manual rocm bench. Print
sufficient numbers that the SUMMARY can quote real before/after medians per the deliverable.
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute --features rocm --example cuda_mirror_overhead 2>&1 | tail -15</automated>
  </verify>
  <done>
`cargo run -p lgbm-compute --features rocm --example cuda_mirror_overhead` runs on gfx1100,
warms up (>=2 discard) and reports median-of->=5 figures for the per-call vs resident paths
across 16/64/256 bins at 50 features / ~200k rows, with MB-transferred-per-call for each.
The CPU-only build prints the stub and exits cleanly. Real before/after numbers captured for
the SUMMARY.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host launcher → GPU kernel | host-validated slices/handles cross into unchecked device kernel code |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-mwr-01 | Tampering/Info-disclosure | `launch_unchecked` mirror kernel (out-of-bounds device access if validation gap) | mitigate | Per-call variant keeps the full V5 scan (bin-range per feature + every data_index < num_data + length checks) BEFORE upload; resident variant validates everything reachable host-side (grad/hess/slot_off lengths, data_index range, num_bin<=256) and documents the bin-range invariant as the caller's upload-time responsibility (mirrors `build_leaf_histograms_resident_f32_on`). SAFETY comments updated to state the launch_unchecked contract. |
| T-mwr-02 | Tampering (numerics) | switching launch path could change histogram output | mitigate | launch_unchecked changes ONLY bounds-check codegen, not scatter/atomic order; resident variant launches the SAME kernel with the SAME CubeCount/CubeDim. New parity test pins the resident path GPU-vs-CPU-f64-anchor within the f32-atomic envelope (never GPU-vs-GPU, DEF-f8u-01); existing 3 tests cover the per-call launch_unchecked switch unchanged. |
| T-mwr-SC | Tampering | npm/pip/cargo installs | accept | No new dependencies — uses existing `cubecl`/`cubecl-hip` already in the workspace; no package installs in this plan. |
</threat_model>

<verification>
- `cargo build -p lgbm-compute --features rocm` compiles (kernel + both launchers).
- `cargo build -p lgbm-compute` (CPU-only) still compiles — mirror code stays rocm-gated.
- `cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror` is GREEN (existing 3 + new resident test) within the f32-atomic envelope vs the CPU f64 anchor.
- `cargo run -p lgbm-compute --features rocm --example cuda_mirror_overhead` prints warmed-up median before/after figures with per-call MB-transferred.
- The CPU f64 bit-exact anchor and the wired production histogram path are UNTOUCHED.
</verification>

<success_criteria>
- Mirror kernel uses `launch_unchecked`; both launchers call it inside `unsafe` with sound, updated SAFETY comments (MWR-01).
- A resident-Handle mirror launcher exists that uploads the bin buffer once and re-uploads only per-leaf data (MWR-02).
- rocm_cuda_mirror parity stays green for both paths, pinned GPU-vs-CPU-anchor (never GPU-vs-GPU).
- A warmed-up before/after micro-bench reports real medians separating the transfer-overhead win (per-call vs resident) from the launch-overhead win, with MB-per-call quantified (MWR-03).
- No production wiring change; CPU anchor untouched; reference trees never git-added.
</success_criteria>

<output>
Create `.planning/quick/260619-mwr-reduce-overhead-of-the-cuda-mirror-gpu-h/260619-mwr-SUMMARY.md` when done.
Quote the real before/after median figures (per-call vs resident, per bin count) and the
per-call MB-transferred reduction in the SUMMARY — the deliverable is real numbers, not a claim.
</output>
