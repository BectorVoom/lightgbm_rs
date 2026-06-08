---
phase: quick-260608-oib
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/oracle-harness/tests/kernel_parity.rs
autonomous: true
requirements: [L3-GPU-FIXCOMPACT, L3-DEVICE-RESIDENT-SCAN]

must_haves:
  truths:
    - "An on-GPU fix+compact kernel produces, for a leaf's RAW f32-atomic histogram, the SAME fixed+compacted f64 buffer the host fix_histogram+compact_histogram produces — BIT-EXACT (compare_exact_f64_bits) for mfb==0/offset==1, mfb>0/offset==0, mfb>=num_bin no-op, and offset>=num_bin degenerate-zero."
    - "The CPU merge gate is byte-unchanged: fix_histogram.rs and compact_histogram (learner.rs) are not modified; CpuBackend keeps host-side fix+compact; kernel_parity + learner_parity (cpu) stay GREEN bit-exact."
    - "DEF-07-02 semantics are REPLICATED, not changed: boosting_parity (incl. mfb_zero_offset_histogram_contract, golden [2,4,2,4]) stays GREEN."
    - "For directly-built leaves on the GPU path the fixed+compacted histogram stays DEVICE-RESIDENT from build through the split-scan — only the per-feature 12-cell SplitInfo is read back (the full-histogram read-back + re-upload to the split kernel are eliminated)."
    - "GPU-grown trees are unchanged vs the nn7 baseline; the pre-existing f32 D-03a split gap is untouched; rocm parity suite stays GREEN."
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/histogram.rs"
      provides: "On-GPU fix+compact kernel (one cube per feature, f64 ascending fold) + a resident build->fix->compact launcher returning a device Handle (and a host-readback variant for the Task-1 oracle)."
      contains: "fix_compact"
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "RocmBackend wiring that threads the resident fixed+compacted Handle from build into find_best_splits for directly-built leaves; host fallback for subtract-derived leaves."
    - path: "crates/oracle-harness/tests/kernel_parity.rs"
      provides: "GPU-fix+compact == host-fix+compact BIT-EXACT oracle (4 offset/mfb cases) under #[cfg(feature = \"rocm\")]."
      contains: "fix_compact"
  key_links:
    - from: "crates/lgbm-compute/src/kernels/histogram.rs (fix+compact kernel)"
      to: "fix_histogram.rs:50-80 + learner.rs:2838 compact_histogram"
      via: "verbatim ascending f64 subtraction + offset shift, single-owner per feature (CubeDim::new_1d(1), one cube per feature)"
      pattern: "fix_compact"
    - from: "crates/lgbm-compute/src/lib.rs (RocmBackend build path)"
      to: "crates/lgbm-compute/src/kernels/split.rs find_best_splits_batched_fused_f64_on"
      via: "device Handle of the fixed+compacted histogram passed straight into the split kernel (no host buf re-upload) for directly-built leaves"
      pattern: "resident|Handle"
---

<objective>
L3 (deferred from 260608-nn7): move FixHistogram + histogram compaction onto the GPU
and keep the fixed/compacted histogram DEVICE-RESIDENT through the split-scan, so the
per-leaf build→fix→scan chain stops bouncing the full histogram to the host. This
eliminates the dominant remaining GPU host↔device round-trip: today
`build_leaf_histograms_resident_f32_on` reads the full RAW f32 histogram back to the
host (widened to f64) so the host can run `fix_histogram` + `compact_histogram`, then
`find_best_splits_batched_fused_f64_on` re-uploads that fixed buffer to the split kernel
(`split.rs:1125 client.create_from_slice(buf)`). Both transfers go away for
directly-built leaves.

Purpose: the fix-mandated read-back is the round-trip nn7's SUMMARY explicitly assigned
to L3 (the only residual per-leaf transfer after L1/L2). Removing it is the concrete
remaining GPU win for the per-leaf flow.

Output: an on-GPU fix+compact kernel proven BIT-EXACT against the host fix+compact
(Task 1, isolates the numerically-risky kernel with no architecture change), then the
device-resident build→fix→compact→scan wiring for directly-built leaves (Task 2), with
the subtraction-trick larger child kept on the existing host path (documented deferral).

NON-NEGOTIABLE: GPU-only. The CPU `cubecl-cpu` f64-fold path is the merge gate and stays
BYTE-UNCHANGED — `fix_histogram.rs`, `compact_histogram`, and CpuBackend are untouched.
This is a PORT of the current host semantics, NOT a fix of DEF-07-02.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

# The exact host semantics being ported (read these — the kernel must reproduce them):
@crates/lgbm-treelearner/src/fix_histogram.rs
@crates/lgbm-treelearner/src/lib.rs
# learner.rs key regions: compact_histogram fn:2838-2864; build_leaf_histogram_into:1442-1497
# (build→fix→compact→pool); find_best_splits build/scan/subtract orchestration:1326-1431.
@crates/lgbm-treelearner/src/learner.rs

# Compute seam + existing GPU kernels (the precedents to reuse):
@crates/lgbm-compute/src/lib.rs
# histogram.rs: build_leaf_histograms_resident_f32_on:549 (resident gather, READS BACK at :605),
# construct_leaf_hist_resident_kernel:511 (one-unit-per (feature,row) f32-atomic build).
@crates/lgbm-compute/src/kernels/histogram.rs
# split.rs: find_best_splits_batched_fused_f64_on:1021 (consumes host buf, uploads at :1125;
# one cube per feature CubeCount::Static(n,1,1) / CubeDim::new_1d(1)); split_scan_body:144
# (the shared #[cube] per-feature sequential scan — the precedent for one-cube-per-feature
# f64 sequential kernels).
@crates/lgbm-compute/src/kernels/split.rs
@crates/lgbm-compute/src/kernels/subtract.rs

# Gates that MUST stay GREEN + the oracle pattern to mirror:
# boosting_parity.rs:2938 mfb_zero_offset_histogram_contract (golden [2,4,2,4]).
@crates/oracle-harness/tests/boosting_parity.rs
# kernel_parity.rs: compare_exact_f64_bits import:35; resident-gather rocm oracle:1464-1529
# (the EXACT template for the new GPU-fix==host-fix oracle: RocmBackend::default(),
# upload_resident_bins, build through the override, assert).
@crates/oracle-harness/tests/kernel_parity.rs

# Bench harness (nn7 baselines: small 1.43s / medium ~4.4s / large ~11.9s; --features rocm):
@crates/lgbm/examples/bench_train.rs

# nn7 deferred-L3 spec:
@.planning/quick/260608-nn7-eliminate-gpu-host-device-round-trips-de/260608-nn7-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 0: Confirm baseline GREEN + capture GPU bench BEFORE</name>
  <files>(no source edits — measurement only)</files>
  <action>
Establish the honest before-state. Run, in the MAIN tree on branch master (worktree
isolation DISABLED; do NOT touch untracked LightGBM/ or .serena/):

1. `cargo build --workspace` (cpu) and `cargo build --workspace --features rocm` — both
   must compile clean. Capture real output (warnings ok, no errors).
2. CPU bit-exact gates GREEN: run kernel_parity and learner_parity (cpu), and the full
   boosting_parity suite — CONFIRM `mfb_zero_offset_histogram_contract` is GREEN (golden
   leaf_count [2,4,2,4], split_gain ≈[5.06897, 1.8, 0.375479]). This is the DEF-07-02
   port-target the GPU kernel must REPLICATE, not change.
3. ROCm parity suite GREEN (the rocm-gated kernel_parity + learner_parity tests on
   gfx1100). Note any pre-existing f32 D-03a split gap as the accepted baseline.
4. Capture the GPU bench BEFORE: `cargo run --release --features rocm --example bench_train`
   (or the repo's exact invocation — check bench_train.rs header / STATE for the flag set).
   Record the median train times for small/medium/large VERBATIM (these are the L3
   before-numbers; do not reuse nn7's quoted 1.43/4.4/11.9 if this machine differs —
   capture fresh on THIS HEAD).

Record all four outputs in the eventual SUMMARY's evidence section. NO fabricated numbers:
if a suite cannot run (e.g. GPU busy), say so explicitly rather than inventing a result.
  </action>
  <verify>
    <automated>cargo build --workspace 2>&1 | tail -5 && cargo build --workspace --features rocm 2>&1 | tail -5</automated>
  </verify>
  <done>
Both builds compile; kernel_parity + learner_parity (cpu) + boosting_parity (incl.
mfb_zero_offset_histogram_contract) GREEN; rocm parity suite GREEN; GPU bench BEFORE
medians captured verbatim for small/medium/large.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 1: On-GPU fix+compact kernel, gated BIT-EXACT vs host fix+compact</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs, crates/oracle-harness/tests/kernel_parity.rs</files>
  <behavior>
The new kernel + launcher, given a leaf's concatenated RAW stride-2 f64 histogram
(the SAME values the host gets by widening the f32-atomic cells — Task 1 may start from
the existing host-readback RAW buffer, then re-upload it; this isolates fix+compact
numerics independent of where the RAW build lives) plus per-feature {num_bin, offset,
most_freq_bin} and the leaf's RAW sum_gradient/sum_hessian, MUST produce a buffer
BIT-IDENTICAL to applying host `fix_histogram` then `compact_histogram` per feature.

Oracle cases (compare_exact_f64_bits, #[cfg(feature="rocm")] on gfx1100):
  - Test A — mfb>0, offset==0: fix RECONSTRUCTS the mfb cell (raw leaf total − Σ other
    bins, ASCENDING order), compact is a no-op. Use a feature like the resident-oracle's
    f0/f2 with most_freq_bin>0.
  - Test B — mfb==0, offset==1: fix is a NO-OP (C++ `if (most_freq_bin>0)` false), compact
    DROPS bin 0 and shifts cell c ← bin c+1, tail zeroed. This is the DEF-07-02 case.
  - Test C — mfb>=num_bin: fix is a no-op (defensive bound), offset==0 ⇒ compact no-op.
  - Test D — offset>=num_bin degenerate: compact zeros the whole feature region.
A mixed-feature leaf (features with different mfb/offset/num_bin concatenated) must also
match the per-feature host loop cell-for-cell.
  </behavior>
  <action>
Implement an on-GPU fix+compact for ONE leaf's concatenated histogram, VERBATIM to the
host semantics in `fix_histogram.rs:50-80` and `compact_histogram` (`learner.rs:2838-2864`):

A. Add a `#[cube(launch)]` `fix_compact_kernel` (`#[cfg(feature="rocm")]`) in
   histogram.rs. ONE cube per feature — `CubeCount::Static(num_features,1,1)`,
   `CubeDim::new_1d(1)` — mirroring `construct_leaf_hist_resident_kernel` and the fused
   split kernel's one-cube-per-feature precedent (`split.rs:944-976`). Cube `f` owns
   ONLY its `[slot_off[f], slot_off[f]+2*num_bin[f])` region. Per-feature scalar params
   (num_bin, offset, most_freq_bin) arrive as `n`-length `Array<i32>`/`Array<u32>` (same
   marshalling shape as the fused split launcher's slot_off/num_bin/offset arrays). The
   leaf RAW sum_gradient/sum_hessian are leaf-level scalars shared across the batch.
   Compute in f64 (gfx1100 runs f64 despite has_f64==false — see the fused f64 split
   kernel precedent).

   In-kernel per feature, EXACT order (port, do not "improve"):
   1. FixHistogram: if most_freq_bin==0 OR most_freq_bin>=num_bin → skip (no write). Else
      seed g=sum_gradient, h=sum_hessian; for i in 0..num_bin, if i!=mfb subtract
      hist[i<<1]/hist[(i<<1)+1] in ASCENDING i (load-bearing f64 fold order — never
      reorder/parallelize); write hist[mfb<<1]=g, hist[(mfb<<1)+1]=h. Use the RAW
      sum_hessian (Pitfall 2 — NOT the +2*kEpsilon bumped value).
   2. compact: if offset<=0 → no-op. If offset>=num_bin → zero the whole feature region.
      Else for c in 0..(num_bin-offset) (ASCENDING): hist[c<<1]=hist[(c+offset)<<1];
      hist[(c<<1)+1]=hist[((c+offset)<<1)+1]; then zero the tail
      [(num_bin-offset)<<1 .. 2*num_bin). The forward in-place shift is safe (src>=dst).
   Respect the cubecl-cpu/MLIR lowering constraints noted in split.rs:36-49 if they apply
   on hip (literal-init loop-carried vars; branchless select for conditional stores) —
   reuse the same encoding workarounds the fused split kernel uses.

B. Add a host launcher `fix_compact_f64_on<R>` that takes the concatenated buf (host
   slice for Task 1's isolation), the per-feature {slot_off,num_bin,offset,most_freq_bin}
   arrays, and sum_gradient/sum_hessian; uploads, launches, reads back the fixed+compacted
   f64 buffer. (Task 2 will add the device-Handle-in/Handle-out variant; Task 1 keeps the
   readback so the kernel is proven in isolation with zero pool/architecture change.)
   V5 validation BEFORE launch (mirror the fused split launcher): num_bin==0 → typed
   error; 2*num_bin overflow → typed error; slot_off+2*num_bin > buf.len() →
   LengthMismatch; empty feats → Ok(buf unchanged) with no launch. Confine all cubecl
   `unsafe` here (CMP-01) with a SAFETY comment.

C. Add the BIT-EXACT oracle in kernel_parity.rs under `#[cfg(feature="rocm")]`, mirroring
   the resident-gather oracle template (`:1464-1529`): build the four cases above, run the
   host path (`fix_histogram` + `compact_histogram` per feature, imported from
   lgbm_treelearner) and the GPU path (`fix_compact_f64_on`), and assert with
   `compare_exact_f64_bits` (NOT assert_within — the key numerical insight is that reading
   the same f64-widened RAW cells + folding in the same ascending order yields BIT-EXACT
   output; the only ~1e-6 difference is the already-accepted f32-atomic RAW build, which
   is NOT exercised here). Include the mixed-feature leaf assertion.

Do NOT modify fix_histogram.rs or compact_histogram. Do NOT change CpuBackend. Do NOT
wire this into the live build path yet (Task 2 does the wiring).
  </action>
  <verify>
    <automated>cargo build --workspace --features rocm 2>&1 | tail -5 && cargo test -p oracle-harness --features rocm fix_compact 2>&1 | tail -25</automated>
  </verify>
  <done>
`fix_compact_kernel` + `fix_compact_f64_on` exist and compile under --features rocm; the
new oracle (4 cases + mixed-feature leaf) passes BIT-EXACT via compare_exact_f64_bits on
gfx1100; fix_histogram.rs / compact_histogram / CpuBackend byte-unchanged
(`git diff` shows no hunks touching them); cpu kernel_parity + learner_parity +
boosting_parity (incl. mfb_zero_offset_histogram_contract) still GREEN; GPU trees
unchanged (the kernel is not yet wired into the live path).
  </done>
</task>

<task type="auto">
  <name>Task 2: Keep the fixed+compacted histogram DEVICE-RESIDENT through the split-scan (directly-built leaves)</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs, crates/lgbm-compute/src/lib.rs, crates/oracle-harness/tests/kernel_parity.rs</files>
  <action>
Eliminate the per-leaf full-histogram round-trip for DIRECTLY-BUILT leaves by threading
the fixed+compacted histogram as a device `Handle` from the build kernel through fix+compact
into the split-scan kernel — the histogram VALUES never leave the device; only the small
per-feature 12-cell SplitInfo is read back.

SCOPE (honest staging — directly-built leaves only):
- The directly-built smaller leaf path is the clean target: `build_leaf_histogram_into`
  (learner.rs:1442) calls `backend.build_leaf_histograms_raw` then the host loops
  fix+compact, and `scan_leaf_histogram` calls `backend.find_best_splits_batched` which
  re-uploads the host buf. Collapse this into a single device-resident chain on RocmBackend.
- The SUBTRACTION-TRICK larger child (learner.rs:1352-1418) operates on host pool
  `Vec<f64>` buffers (parent − smaller via `subtract_histograms`, plus the T-05-07-01
  audit hook that re-reads host buf, and the use_subtract derivation that is NOT
  re-FixHistogram'd). Keeping that resident would require restructuring the HistogramPool
  + subtraction into device handles — out of scope / high-risk for this task. KEEP the
  subtract-derived larger child on the EXISTING host path (build_leaf_histograms_raw
  readback → host subtract → host find_best_splits_batched). Document this explicitly in
  the SUMMARY as the deferred remainder. The pool, subtraction trick, and host fix+compact
  fallback all remain intact and bit-faithful for that path.

Wiring (RocmBackend-only; CpuBackend + the Backend trait defaults UNCHANGED):
1. Add a device-Handle variant of the build→fix→compact chain in histogram.rs:
   `build_fix_compact_resident_f64_on<R>` that runs the resident build kernel
   (`construct_leaf_hist_resident_kernel`) into an f32-atomic device buffer, then launches
   `fix_compact_kernel` (Task 1) over that buffer (widened to an f64 device buffer — match
   the existing f32→f64 widening semantics so the values equal today's host readback),
   and RETURNS the fixed+compacted f64 device `Handle` (NOT a Vec<f64>). No readback here.
2. Modify `find_best_splits_batched_fused_f64_on` (split.rs) — or add a sibling
   `..._from_handle` — to accept a pre-existing device `Handle` for the histogram instead
   of always `create_from_slice(buf)` at :1125. When a Handle is supplied, skip the
   upload and feed it directly to the split kernel (`ArrayArg::from_raw_parts(handle,
   slot_len)`); keep the existing buf-slice entry point for the host/subtract path and the
   cpu oracle. Buf length must be carried alongside the Handle (the split kernel reads
   `[slot_off,...]` regions within it).
3. In RocmBackend (lib.rs), for the directly-built path, expose the resident chain so the
   learner's build+scan of a directly-built leaf routes build→fix→compact→scan entirely on
   device. Use the existing `RefCell` resident-state pattern (nn7 ResidentBins precedent at
   lib.rs:472-500) to hold the per-leaf fixed+compacted Handle between the build call and
   the scan call (single-threaded train loop ⇒ RefCell borrow is safe). The
   subtract-derived larger child must continue to route through the host
   `build_leaf_histograms_raw` + host fix+compact + buf-slice split entry (the fallback).

   Note the seam shape: `build_leaf_histograms_raw` currently returns `Vec<f64>` and
   `find_best_splits_batched` takes `buf: &[f64]`. Choose the LEAST-invasive wiring that
   keeps the directly-built leaf's histogram on device while leaving CpuBackend and the
   subtract path on the Vec<f64> seam unchanged — e.g. RocmBackend stashes the resident
   Handle in its RefCell during `build_leaf_histograms_raw` (still returning a Vec<f64>
   for the trait, but marking "this leaf is resident"), and consumes it in
   `find_best_splits_batched` when present, falling back to `create_from_slice(buf)` when
   absent (subtract path / cache miss). Keep the trait signatures intact; do the residency
   bookkeeping inside RocmBackend. Document the exact mechanism chosen in the SUMMARY.

4. Parity: GPU-grown trees must be UNCHANGED vs the nn7 baseline (the resident path
   produces the same fixed+compacted values the host path did — proven bit-exact in
   Task 1; the only ~1e-6 is the unchanged f32-atomic RAW build). Add/extend a rocm oracle
   asserting the resident build→fix→compact→scan SplitInfo equals the host-path SplitInfo
   for a directly-built leaf (assert_within for the f32-atomic RAW-build tolerance; the
   fix+compact step itself is bit-exact). The pre-existing f32 D-03a split gap stays
   untouched.

5. Capture the GPU bench AFTER (same invocation as Task 0) and report small/medium/large
   medians side-by-side with the BEFORE numbers to quantify the round-trip elimination.
   The win shows most on workloads where the per-leaf full-histogram transfer is
   proportionally large (cf. nn7's medium-skewed L1 result). NO fabricated numbers — if
   the win is within run-to-run noise on a size, say so.

DEFERRAL: if threading the Handle through the split kernel proves too invasive to land
safely within this task, deliver Task 1 (proven bit-exact on-GPU fix+compact) + the
resident build→fix→compact Handle (step 1) WITHOUT step 2/3, and DEFER the
split-from-Handle wiring + subtraction-trick residency with an explicit follow-up section
in the SUMMARY. Honest partial delivery over an over-stuffed risky commit. The
subtraction-trick + full pool-residency is the expected deferral either way.

CPU path BYTE-UNCHANGED throughout: fix_histogram.rs, compact_histogram, CpuBackend, and
the Backend trait DEFAULT impls must show no diff hunks affecting their behavior.
  </action>
  <verify>
    <automated>cargo build --workspace 2>&1 | tail -5 && cargo build --workspace --features rocm 2>&1 | tail -5 && cargo test -p oracle-harness 2>&1 | tail -15</automated>
  </verify>
  <done>
For directly-built leaves the fixed+compacted histogram stays device-resident from build
through scan (no full-histogram readback/re-upload; only 12-cell SplitInfo returns) — OR
the split-from-Handle wiring is explicitly deferred with the resident build→fix→compact
Handle delivered. CPU bit-exact gates GREEN byte-unchanged (kernel_parity + learner_parity
cpu); boosting_parity incl. mfb_zero_offset_histogram_contract GREEN; rocm parity suite
GREEN with GPU trees unchanged vs nn7 and D-03a untouched; the resident-path-==-host-path
SplitInfo oracle passes; GPU bench AFTER captured and compared to BEFORE (real numbers).
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host learner → GPU kernel | per-leaf histogram values + per-feature params cross to device; results cross back |
| device RAW(f32) → device fixed(f64) | the fix+compact kernel reads f32-atomic cells widened to f64 — the parity-critical seam |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-oib-01 | Tampering | fix+compact kernel numerics (mfb reconstruction / offset shift) silently diverging from host | mitigate | BIT-EXACT oracle (compare_exact_f64_bits) covering mfb==0/offset==1, mfb>0, mfb>=num_bin, offset>=num_bin, + mixed-feature leaf (Task 1) |
| T-oib-02 | Tampering | accidental edit to fix_histogram.rs / compact_histogram / CpuBackend breaking the merge gate | mitigate | `git diff` no-hunk check on those files each task; cpu kernel_parity + learner_parity + boosting_parity GREEN bit-exact each task |
| T-oib-03 | Tampering | DEF-07-02 semantics "fixed" instead of REPLICATED (offset double-apply) | mitigate | boosting_parity::mfb_zero_offset_histogram_contract (golden [2,4,2,4]) GREEN at Task 0 baseline AND after each task; explicit Test B (mfb==0/offset==1) in the oracle |
| T-oib-04 | Information disclosure | wrong/garbage values fed to split kernel from an unsynchronized or wrong-length device Handle | mitigate | carry buf length with the Handle; V5 slot_off+2*num_bin ≤ len validation before launch; resident-path-==-host-path SplitInfo oracle (Task 2) |
| T-oib-05 | Denial of service | out-of-bounds device index (slot_off / bin / offset) → UB / hang on gfx1100 | mitigate | per-feature region validation host-side before launch; in-kernel ASCENDING bounds match host; SAFETY comments on confined cubecl unsafe (CMP-01) |
| T-oib-SC | Tampering | npm/pip/cargo installs | accept | no new package installs in this plan (pure-Rust workspace edits only); no legitimacy gate needed |
</threat_model>

<verification>
Run after EACH task (executor captures REAL output, no fabrication):
- `cargo build --workspace` (cpu) AND `cargo build --workspace --features rocm` — clean.
- CPU bit-exact (merge gate): kernel_parity + learner_parity (cpu) GREEN BIT-EXACT.
- boosting_parity full suite incl. `mfb_zero_offset_histogram_contract` (golden [2,4,2,4],
  split_gain ≈[5.06897, 1.8, 0.375479]) GREEN — DEF-07-02 REPLICATED not changed.
- ROCm parity suite GREEN on gfx1100; pre-existing f32 D-03a split gap untouched.
- Task 1: the new GPU-fix+compact == host-fix+compact BIT-EXACT oracle (4 cases + mixed)
  passes via compare_exact_f64_bits.
- Task 2: resident-path-==-host-path SplitInfo oracle passes (assert_within for the
  f32-atomic RAW build); GPU bench before/after captured and compared.
- `git diff` confirms fix_histogram.rs / compact_histogram / CpuBackend / Backend-trait
  defaults have NO behavior-changing hunks.
</verification>

<success_criteria>
- On-GPU fix+compact kernel proven BIT-EXACT vs host fix_histogram+compact_histogram
  across all four mfb/offset cases and a mixed-feature leaf.
- Directly-built leaves keep the fixed+compacted histogram device-resident build→scan
  (full-histogram readback + re-upload eliminated; only 12-cell SplitInfo returns) — OR
  the split-from-Handle wiring is honestly deferred with the resident build→fix→compact
  Handle delivered and the remainder documented.
- CPU merge gate byte-unchanged and GREEN bit-exact; boosting_parity (incl.
  mfb_zero_offset_histogram_contract) GREEN; rocm parity GREEN; GPU trees unchanged vs
  nn7; D-03a untouched.
- GPU bench before/after captured with REAL medians quantifying the round-trip removal.
- Subtraction-trick larger child kept on the host fallback path, documented as the
  deferred remainder.
</success_criteria>

<output>
Create `.planning/quick/260608-oib-l3-on-gpu-fixhistogram-compaction-keep-f/260608-oib-SUMMARY.md` when done.
</output>
