---
quick_id: 260608-lsx
slug: fuse-histogram-construction-and-partition-search
type: execute
date: 2026-06-08
mode: quick
phase: quick-260608-lsx
plan: 01
wave: 1
depends_on: [260608-lad]
autonomous: false
requirements: [GPU-PERF-batched-find-best-split]
files_modified:
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/oracle-harness/tests/kernel_parity.rs
must_haves:
  truths:
    - "CpuBackend grows trees BIT-EXACT to today (kernel_parity + learner_parity unchanged)"
    - "A single Backend trait method (find_best_splits_batched) finds all of a leaf's per-feature splits, generic over <B: Backend>"
    - "CpuBackend uses the DEFAULT impl (per-feature loop, bit-exact anchor); RocmBackend OVERRIDES with one batched GPU launch per leaf"
    - "scan_leaf_histogram routes its spine find_best_split calls through the new batched method without changing the grown tree on CPU"
    - "ROCm (--features rocm) build compiles and the batched GPU path runs within ~1e-6 of the CPU f64 anchor"
  artifacts:
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "Backend::find_best_splits_batched default impl + RocmBackend override"
      contains: "fn find_best_splits_batched"
    - path: "crates/lgbm-treelearner/src/learner.rs"
      provides: "two-pass scan_leaf_histogram (gates -> batched call -> argmax)"
      contains: "find_best_splits_batched"
    - path: "crates/oracle-harness/tests/kernel_parity.rs"
      provides: "batched-vs-per-feature parity assertion on CPU"
  key_links:
    - from: "crates/lgbm-treelearner/src/learner.rs"
      to: "Backend::find_best_splits_batched"
      via: "spine scan call"
      pattern: "find_best_splits_batched"
    - from: "crates/lgbm-compute/src/lib.rs RocmBackend"
      to: "kernels::split batched GPU launch"
      via: "trait override"
      pattern: "find_best_splits_batched"
---

<objective>
Fuse histogram construction and partition search (find_best_split) into a single
fused/batched per-leaf operation, fully generic over `<B: Backend>` so the runtime
switches between CPU (the f64 bit-exact anchor) and GPU (ROCm) by backend choice.

"Partition search" here = finding the best split/partition point by SCANNING the
leaf histogram (NOT data_partition / row routing — that stays a separate stage).
This implements the already-DESIGNED-but-deferred 260608-lad **Part 2**
(`find_best_splits_batched`): a `Backend` trait method whose default impl loops
`find_best_split` in feature order (CPU byte-identical to today), and a
`RocmBackend` override that finds all of a leaf's per-feature splits in ONE GPU
launch — collapsing `num_features` per-leaf find_best_split launches to 1/leaf.

Purpose: this is the remaining GPU lever from 260608-lad (after batched histograms
got the GPU to −28%); `subtract` + `data_partition` per-split launches are the
small tail. The hist build (`build_leaf_histograms_raw`, lad Part 1/3) and the
split scan (this task) become the two halves of a leaf's fused work, both
backend-batched.

NON-NEGOTIABLE (CLAUDE.md): the CpuBackend f64-fold path stays BIT-EXACT (it is the
hard merge gate); the learned tree structure on CPU is identical to today; the ROCm
f32 path is within ~1e-6.

Output: `Backend::find_best_splits_batched` (default + RocmBackend override), a
two-pass `scan_leaf_histogram`, and a CPU kernel-parity assertion that the batched
result equals the per-feature loop cell-for-cell.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@.planning/quick/260608-lad-abstract-backend-parallel-prefix-sum-fin/SUMMARY.md

# The Backend trait + both backends (the trait method goes here)
@crates/lgbm-compute/src/lib.rs

# Existing per-feature find_best_split (CPU native, f64 cubecl, ROCm f32) — the
# batched method composes / mirrors these
@crates/lgbm-compute/src/kernels/split.rs

# The spine scan to refactor (~1493-1810) + the build_leaf_histogram_into batched
# precedent (~1430-1485) + the find_best_splits caller (~1327-1394)
@crates/lgbm-treelearner/src/learner.rs

# The oracle gate (CPU-only, EXACT f64 bit-match; does NOT need LightGBM/)
@crates/oracle-harness/tests/kernel_parity.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add Backend::find_best_splits_batched (default impl + RocmBackend override)</name>
  <files>crates/lgbm-compute/src/lib.rs, crates/lgbm-compute/src/kernels/split.rs</files>
  <action>
Add a new trait method `find_best_splits_batched` to `Backend` in
crates/lgbm-compute/src/lib.rs, modeled on the existing `build_leaf_histograms_raw`
batched seam (the lad Part 1 precedent at ~lib.rs:204-235). This is the lad Part 2
design.

Signature intent (the fused per-leaf split SCAN over the concatenated histogram
buffer): it takes the same concatenated stride-2 f64 histogram buffer layout that
`build_leaf_histograms_raw` produces (feature `fpos` occupying
`[slot_off[fpos], slot_off[fpos] + 2*num_bins[fpos])`), PLUS one entry per spine
feature of the per-feature parameters that `find_best_split` already takes. Pass
the per-feature params as parallel slices (one element per scanned feature
position), NOT a heap of separate args:
  - `client: &ComputeClient<Self::Runtime>`
  - `buf: &[f64]` — the concatenated leaf histogram (already FixHistogram'd +
    compacted by the caller, exactly as `scan_leaf_histogram` reads it today)
  - a `&[BatchedSplitFeature]` (define this small plain-data struct in split.rs and
    re-export it) carrying, per scanned feature: `slot_off: usize`, `num_bin: u32`,
    `offset: i32`, `default_bin: u32`, `most_freq_bin: u32`, `skip_default_bin: bool`,
    `na_as_missing: bool`, `run_forward: bool`. These are EXACTLY the per-feature
    args `Backend::find_best_split` takes today (lib.rs:123-138).
  - `cfg: &GainConfig`, `sum_gradient: f64`, `sum_hessian: f64`, `num_data: i32`
    (the leaf totals — shared across all features in the batch).
Returns `Result<Vec<SplitInfo>, ComputeError>` — one SplitInfo per input feature,
in the SAME order as the input slice (order-preservation is what keeps CPU
bit-exact, since the caller's cross-feature argmax is order-sensitive via
`split_gt`).

DEFAULT impl (on the trait, used by CpuBackend unchanged): loop over the feature
list in order, for each call `self.find_best_split(client, &buf[slot_off..slot_off+
2*num_bin], cfg, num_bin, offset, default_bin, most_freq_bin, skip_default_bin,
na_as_missing, run_forward, sum_gradient, sum_hessian, num_data)?` and collect into
a Vec. This composes the EXISTING per-feature `find_best_split` so CpuBackend
output is byte-identical to the current per-feature loop (the lad Part 2 design
point: "default impl loops find_best_split in feature order (CPU bit-exact)"). Do
NOT add CpuBackend-specific overrides — the default IS the anchor.

RocmBackend OVERRIDE (`#[cfg(feature = "rocm")]`, in the RocmBackend impl block):
add a batched GPU implementation in crates/lgbm-compute/src/kernels/split.rs,
e.g. `find_best_splits_batched_f64_on<R: Runtime>` (one cube/launch per feature
reading its `[slot_off, slot_off+2*num_bin)` region of `buf`, running the SAME scan
as `find_best_split_f64_on`, writing its SplitInfo; ONE launch finds all features'
splits per leaf). Reuse the existing `#[cube]` gain primitives in gain.rs and the
scan logic already in `find_best_split_f64_on` (split.rs:578) / its f32 sibling —
keep the C++ gate ORDER / eps / threshold semantics identical so the f32 path
stays within ~1e-6 of the f64 anchor. If a fully-batched single-launch kernel is
non-trivial to land safely in this task, the RocmBackend override MAY initially
delegate to the default loop (still correct + within the same gate) and the
batched kernel can be the GPU-perf follow-up — but the trait method, default impl,
and the CPU bit-exact composition MUST land here. Do NOT weaken any numerical
assertion. Do NOT touch `data_partition`.
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute --features cpu</automated>
    <automated>cargo test -p oracle-harness --test kernel_parity</automated>
  </verify>
  <done>`Backend::find_best_splits_batched` exists with a default impl that composes `find_best_split` in order; `cargo build -p lgbm-compute` (cpu) is clean; `cargo test -p oracle-harness --test kernel_parity` still passes (4/4, unchanged) because the new method is additive and not yet wired into the learner.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor scan_leaf_histogram to a two-pass batched scan (CPU bit-exact)</name>
  <files>crates/lgbm-treelearner/src/learner.rs</files>
  <action>
Refactor `scan_leaf_histogram` (learner.rs:1493-1810) to the lad Part 2 two-pass
structure WITHOUT changing the grown tree on CPU. The current single loop
interleaves load-bearing per-feature gates with the spine `find_best_split` call
(learner.rs:1696-1711). Split it:

PASS 1 (gate + classify, IN feature order): walk features applying the EXISTING
gates in the SAME order — col-sampler `used_features` mask (1548), the LOAD-BEARING
`parent_splittable` gate (1566, the GOSS-critical skip), the ADV-02 interaction gate
(1573), and the bin_type / monotone / extra-trees branch selection (1587, 1660-1694).
For each SPINE feature (the `else` branch at 1694-1711 — numeric, NOT categorical,
NOT monotone, NOT extra-trees-rand), record its `BatchedSplitFeature` params
(slot_off[fpos], f.num_bin, f.offset, f.default_bin, f.most_freq_bin,
skip_default_bin, na_as_missing, run_forward) into a Vec, remembering its `fpos` so
PASS 2 can map the result back. Categorical / monotone / extra-trees / gated-out
features are NOT batched — they keep their existing inline handling (their split is
computed in-place exactly as today). Preserve the EXACT iteration order so the
recorded spine-feature list is in ascending fpos.

ONE BATCHED CALL: after Pass 1, call
`self.backend.find_best_splits_batched(self.client, buf, &batched_feats, &self.cfg,
sum_g, sum_h, num_data_in_leaf)?` ONCE, getting back `Vec<SplitInfo>` in the same
order as `batched_feats`.

PASS 2 (post-process + argmax, IN feature order): walk features again in the SAME
order; for each spine feature pull its SplitInfo from the batched results (by the
remembered mapping), then apply the EXISTING post-processing in the EXISTING order:
`this_leaf_splittable[fpos] = split.gain > K_MIN_SCORE` (1717), the ADV-05 CEGB
penalty subtract (1721-1730), the ADV-01 monotone penalty (only for monotone
features, handled in their own branch), the D-06 snapshot `per_bin_gains` (gated by
`capture_snapshots`, 1758), push the `FeatureSplitRecord`, and the cross-feature
argmax via `split_gt` (1772-1777). The categorical/monotone/extra-trees inline
results are folded into the SAME argmax in the SAME relative order.

CRITICAL bit-exactness invariants (this is the merge gate):
  - The cross-feature argmax (`split_gt`: gain, then smaller feature) MUST see
    candidates in the IDENTICAL order as today, so the tie-break is unchanged.
  - The `feature_splittable` persistence (1803-1808) and `best_cat_threshold`
    cleanup (1782-1794) MUST be unchanged.
  - On CpuBackend, `find_best_splits_batched` is the default per-feature loop, so
    each spine feature's SplitInfo is byte-identical to today's inline
    `find_best_split` call — the grown tree MUST be identical.
The simplest safe shape: keep the single `for (fpos, f)` loop but, before it, do one
gate-only pre-pass to collect `batched_feats` + the single batched call, then INSIDE
the existing loop replace ONLY the spine `find_best_split(...)?` call
(1696-1711) with a lookup into the batched results. That minimizes churn to the
load-bearing gate/argmax/record code. Do NOT change data_partition. Do NOT alter
the categorical, monotone, or extra-trees branches' math.
  </action>
  <verify>
    <automated>cargo test -p oracle-harness --test learner_parity</automated>
    <automated>cargo test -p oracle-harness --test kernel_parity</automated>
    <automated>cargo build --workspace --tests</automated>
  </verify>
  <done>`scan_leaf_histogram` issues ONE `find_best_splits_batched` call per leaf for its spine features; `cargo test -p oracle-harness --test learner_parity` passes (12/12, trees identical — proves the fused split-finding produces the same trees); `kernel_parity` 4/4 unchanged; workspace tests build.</done>
</task>

<task type="auto">
  <name>Task 3: Confirm the oracle + add a batched-vs-per-feature CPU parity assertion; both-backend build</name>
  <files>crates/oracle-harness/tests/kernel_parity.rs</files>
  <action>
Add a focused CPU test to crates/oracle-harness/tests/kernel_parity.rs that proves
the new fused method is bit-exact to the per-feature path on the committed fixtures.
Reuse the existing split golden parsing (`parse_split`, the SplitGolden cases used
by `kernel_parity_split_bit_exact_on_cpu` at :512). For a leaf made of the golden's
feature cases concatenated into one buffer (build the `buf` by laying each case's
histogram at successive `slot_off`s, and a matching `Vec<BatchedSplitFeature>`):
  1. Call `CpuBackend.find_best_splits_batched(...)` ONCE → `Vec<SplitInfo>`.
  2. For each feature, call the existing `CpuBackend.find_best_split(...)` per
     feature (the path the oracle already asserts bit-exact to the C++ golden).
  3. Assert the batched result equals the per-feature result for every SplitInfo
     field via `compare_exact_f64_bits` on `gain`/`left_output`/`right_output`/
     `left_sum_gradient`/etc. and exact equality on the integer fields (threshold,
     default_left, feature). This proves the DEFAULT batched impl is the bit-exact
     anchor (the lad Part 2 contract).
Keep it CPU-only (no LightGBM/ dependency), matching the existing kernel_parity
style. Do NOT weaken `compare_exact_f64_bits` to a tolerance.

Then run the full required verification gate (below). For ROCm: build
`--features rocm` to confirm the RocmBackend override compiles; if the gfx1100
hardware run is feasible, run the hip-gated kernel_parity split test to confirm the
batched GPU split path is within ORACLE_TOL (~1e-6) of the CPU f64 anchor (the
existing `#[cfg(feature = "rocm")]` hip test block at :804+ is the precedent). If
the hardware run is not feasible in this session, the `--features rocm` BUILD
passing is the required floor and the hip parity run is noted for follow-up — do
NOT fake a hardware result.
  </action>
  <verify>
    <automated>cargo test -p oracle-harness --test kernel_parity</automated>
    <automated>cargo test -p oracle-harness --test learner_parity</automated>
    <automated>cargo build --workspace --features rocm</automated>
  </verify>
  <done>A new CPU test asserts `find_best_splits_batched` == per-feature `find_best_split` bit-exact on the committed split golden; `cargo test -p oracle-harness --test kernel_parity` and `--test learner_parity` both GREEN; `cargo build --workspace` passes for default (cpu) AND `--features rocm`; (if hardware available) the hip batched-split parity is within ~1e-6 of the CPU anchor.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| learner -> Backend trait | leaf histogram buffer + per-feature params cross into the fused compute op |
| CpuBackend (anchor) vs RocmBackend | the f64 bit-exact merge gate vs the ~1e-6 GPU track |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-lsx-01 | Tampering | find_best_splits_batched result ordering | mitigate | preserve input feature order in the returned Vec; argmax via split_gt sees identical order ⇒ CPU tree bit-exact (learner_parity gate) |
| T-lsx-02 | Information disclosure | buf region indexing per feature | mitigate | each feature reads only `[slot_off, slot_off+2*num_bin)`; reuse the validated slot_off layout from build_leaf_histograms_raw; ComputeError on length mismatch (no panic/UB) |
| T-lsx-03 | Denial of service | empty batch / no spine features (all gated out) | accept | empty `batched_feats` ⇒ empty Vec, no launch; categorical/monotone-only leaves handled inline as today |
| T-lsx-SC | Tampering | no new package installs | accept | this task adds no npm/pip/cargo dependencies; no legitimacy gate needed |
</threat_model>

<verification>
- `cargo test -p oracle-harness --test kernel_parity` GREEN (split/hist/partition/subtract bit-exact on CPU; + the new batched-vs-per-feature assertion).
- `cargo test -p oracle-harness --test learner_parity` GREEN (12/12 — the spine/growth parity proving the fused split-finding still grows IDENTICAL trees; this is the core merge gate for this task).
- `cargo build --workspace` (default cpu) AND `cargo build --workspace --features rocm` both exit 0.
- (If gfx1100 available) hip batched-split parity within ORACLE_TOL (~1e-6) of the CPU f64 anchor.
- `LightGBM/` never git-added; `data_partition` untouched.
</verification>

<success_criteria>
- A SINGLE `Backend::find_best_splits_batched` method finds all of a leaf's spine
  feature splits, generic over `<B: Backend>`; CpuBackend uses the default
  per-feature-order loop (bit-exact anchor), RocmBackend overrides for the GPU.
- `scan_leaf_histogram` issues ONE batched call per leaf for its spine features and
  grows trees BIT-EXACT to today on CPU (learner_parity 12/12, kernel_parity 4/4).
- Both `--features cpu` (default) and `--features rocm` build clean.
- No numerical assertion weakened; the CPU f64 merge gate holds bit-exact.
</success_criteria>

<output>
Create `.planning/quick/260608-lsx-fuse-histogram-construction-and-partitio/260608-lsx-SUMMARY.md` when done.
</output>
