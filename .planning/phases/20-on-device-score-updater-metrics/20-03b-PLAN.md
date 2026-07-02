---
phase: 20-on-device-score-updater-metrics
plan: 03b
type: execute
wave: 4
depends_on: [20-00, 20-01, 20-02, 20-03a]
files_modified:
  - crates/lgbm-compute/src/kernels/grow_driver.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/oracle-harness/tests/learner_parity.rs
autonomous: true
requirements: [ODL-18, ODL-19]
must_haves:
  truths:
    - "grow_driver.rs implements a MINIMAL, purpose-built per-leaf best-first orchestration in lgbm-compute that sequences the existing Phase-16/17/18 kernels: root init (Phase-14 split_info/random) -> per-leaf loop up to num_leaves-1 -> build smaller child hist + subtract larger (Phase-16, parent-built-before-child-subtract ordering) -> best-split finder stage1/2/3 (Phase-17) -> break if best_leaf == -1 -> DeviceCudaTree::split_on_device (Phase-18, BEFORE partition) -> partition + update_data_index_to_leaf (Phase-18, using 20-03a's locked buffer strategy) -> to_host_tree + LeafPartitionLayout"
    - "The driver reproduces SerialTreeLearner's best-first order + smaller/larger subtraction pairing + minimal slot bookkeeping WITHOUT reusing lgbm-treelearner's LeafSplits/HistogramPool (those cannot be named from lgbm-compute); it drives the kernels with its own lightweight state"
    - "grow_tree_on_device fills the Ok(None) body to call the driver and returns Some((Tree, LeafPartitionLayout)); the gated on_device_growth_supported() (from 20-03a) makes it reachable only with LGBM_CUDA_ON_DEVICE=1"
    - "The grown on-device tree is STRUCTURE bit-exact to the cpu f64 anchor (tie-aware default_left, identical threshold + child row-counts on any accepted flip), leaf values within ROCM_LEAF_VALUE_TOL, via assert_on_device_tree_matches_cpu_anchor over cpu_anchor_tree — NEVER GPU-vs-GPU (D-07, def-f8u-01)"
    - "Every new driver kernel keeps f32 + u64 fixed-point build with no f64 per-row grow/build hot loop; f64 only in the reference-blessed scalar/gain math (ODL-19, D-08); env unset = byte-unchanged"
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/grow_driver.rs"
      provides: "the per-leaf best-first grow orchestration body sequencing Phase-16/17/18 kernels"
      contains: "num_leaves"
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "grow_tree_on_device body wired to the driver (returns Some)"
      contains: "grow_tree_on_device"
    - path: "crates/oracle-harness/tests/learner_parity.rs"
      provides: "activated STRUCTURE gate cell running a REAL on-device tree vs cpu_anchor_tree (host_grow stand-in replaced)"
  key_links:
    - from: "crates/lgbm-compute/src/kernels/grow_driver.rs"
      to: "crates/lgbm-compute/src/kernels/histogram.rs"
      via: "driver build-smaller/subtract-larger composes the Phase-16 build+fix+subtract kernels"
      pattern: "histogram|subtract"
    - from: "crates/lgbm-compute/src/kernels/grow_driver.rs"
      to: "crates/lgbm-compute/src/kernels/tree.rs"
      via: "DeviceCudaTree::split_on_device runs BEFORE partition and yields right_leaf_index; to_host_tree returns lgbm_model::Tree"
      pattern: "DeviceCudaTree|split_on_device|to_host_tree"
    - from: "crates/oracle-harness/tests/learner_parity.rs"
      to: "crates/lgbm-compute/src/lib.rs"
      via: "assert_on_device_tree_matches_cpu_anchor over a real grow_tree_on_device tree grown on the cubecl-cpu runtime"
      pattern: "assert_on_device_tree_matches_cpu_anchor"
---

<objective>
Fill the load-bearing half of the D-01 pulled-forward driver: implement the minimal per-leaf
best-first grow orchestration in `crates/lgbm-compute/src/kernels/grow_driver.rs` that sequences the
already-golden Phase-16 (histogram build/fix/subtract), Phase-17 (best-split), and Phase-18 (tree
mutation + data partition) kernels into the §6/§16 loop, wire the `grow_tree_on_device` body to it
(returning `Some((Tree, LeafPartitionLayout))`), and ACTIVATE the STRUCTURE-bit-exact gate against
the cpu f64 anchor (replacing the Slice-0 `host_grow` stand-in with a real on-device tree).

The driver reproduces SerialTreeLearner's best-first order + smaller/larger subtraction pairing +
minimal histogram-slot bookkeeping using its OWN lightweight state — it does NOT reuse
lgbm-treelearner's `LeafSplits`/`HistogramPool` (unnameable from lgbm-compute; the verified crate
wall). It composes only lgbm-compute-local kernels (`FeatureMeta` derived from 20-03a's `GrowFeature`,
`BinColumn`, `DeviceCudaTree`) + the buffer strategy 20-03a locked. Scope the proving slice to
continuous features + L2 (no RenewTreeOutput refit — Pitfall 4).

Purpose: this is the STRUCTURE-bit-exact gate D-01 absorbed from Phase 21 — it activates the dormant
learner fork so a full tree grows on device, anchored to the cpu f64 fold in the DEFAULT merge-gate
lane (the on-device tree grows on the cubecl-cpu runtime, per 20-03a's gated CpuBackend flip), never
against a second GPU f32 path.
Output: the driver body in `grow_driver.rs`, the wired `grow_tree_on_device` in `lib.rs`, and the
activated STRUCTURE gate cell in `learner_parity.rs`.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/20-on-device-score-updater-metrics/20-CONTEXT.md
@.planning/phases/20-on-device-score-updater-metrics/20-RESEARCH.md
@.planning/phases/20-on-device-score-updater-metrics/20-PATTERNS.md
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm-compute/src/kernels/best_split.rs
@crates/lgbm-compute/src/kernels/data_partition.rs
@crates/lgbm-compute/src/kernels/tree.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement the per-leaf best-first grow driver in grow_driver.rs + wire the grow_tree_on_device body</name>
  <files>crates/lgbm-compute/src/kernels/grow_driver.rs, crates/lgbm-compute/src/lib.rs</files>
  <read_first>
    - crates/lgbm-compute/src/kernels/histogram.rs (Phase-16 build/fix/subtract: `construct_histograms_f32_on` / `construct_leaf_hist_on_device` / `fix_compact_f64_on` / the subtract entry; the 8aed100-class ordering guard — parent fully built before any child subtract reads it)
    - crates/lgbm-compute/src/kernels/best_split.rs (lines 119-140 FeatureMeta the driver derives from GrowFeature; 194 `build_split_find_tasks`; 632 `find_best_splits_stage1_on`; 1970 `sync_best_split_for_leaf_on`; 2025 `sync_best_split_all_blocks`; 2052 `set_invalid_leaf_split_info`; 2135 `find_best_from_all_splits_on` — the stage-1/2/3 argmax + the best_leaf == -1 sentinel; tie-aware default_left encoded in SplitScalars)
    - crates/lgbm-compute/src/kernels/tree.rs (lines 492 SplitResult; 533-640 DeviceCudaTree::new/num_leaves/num_cat; 691 `split_on_device` returning right_leaf_index BEFORE partition; 835 shrink; 840 add_bias; 898 `to_host_tree` -> lgbm_model::Tree)
    - crates/lgbm-compute/src/kernels/data_partition.rs (lines 482 `partition_leaf_stable`; 657 `partition_on_device`; 822 `update_data_index_to_leaf_on`; 896 `split_tree_structure_packet` — the mark->prefix-sum->scatter; use 20-03a's LOCKED buffer strategy)
    - crates/lgbm-compute/src/kernels/{split_info,random}.rs (Phase-14 device split/leaf-split structs + LCG for root init §6.1)
    - crates/lgbm-compute/src/kernels/grow_driver.rs (the 20-03a GrowFeature struct + the locked buffer-strategy helper this body consumes)
    - crates/lgbm-dataset/src/dataset.rs (lines 88-97 — LeafPartitionLayout the driver returns: num_data/indices/leaf_begin/leaf_count)
    - crates/lgbm-treelearner/src/learner.rs (lines 1266-2066 — the HOST per-leaf growth order to MIRROR for bit-exactness: best-first leaf selection, smaller/larger child assignment, subtraction pairing, threshold recording via bin_upper_bound; this is the reference ORDER the native driver reproduces WITHOUT importing its types)
    - MEMORY: phase18-wr01-histarena-swap-aliasing (the slot-aliasing bug this multi-leaf loop must avoid — use 20-03a's locked strategy); def-f8u-01 (never GPU-vs-GPU); spike-052 (no f64 per-row hot loops; keep u64 fixed-point build)
  </read_first>
  <action>
    Implement `pub fn grow_tree_on_device_driver<R: cubecl::Runtime>(client, gradients, hessians,
    features: &[GrowFeature], num_leaves, max_depth) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError>`
    in `grow_driver.rs`. Root init (§6.1) seeds the single root leaf's gradient/hessian sums via the
    Phase-14 reduction and the split_info/random structs. Then a per-leaf best-first loop up to
    `num_leaves - 1`: (a) derive `FeatureMeta` for each feature from `GrowFeature`, build the SMALLER
    child histogram (Phase-16) and derive the LARGER child by subtraction from the parent (enforcing
    parent-fully-built-before-child-subtract ordering), tracking the smaller/larger pairing + the
    histogram-slot bookkeeping in the driver's OWN minimal state (NOT LeafSplits/HistogramPool); (b)
    run the best-split finder stage-1/2/3 (Phase-17) and read back the chosen split (8-int SplitScalars),
    selecting the best leaf best-first; (c) break when `best_leaf == -1`; (d) call
    `DeviceCudaTree::split_on_device` (Phase-18) BEFORE partition and consume its `right_leaf_index`;
    (e) run the partition + `update_data_index_to_leaf_on` (Phase-18) using the buffer strategy 20-03a
    LOCKED (double-buffer unless alias proven safe). Record thresholds via each feature's
    `bin_upper_bound` (real f64 threshold). After the loop, `DeviceCudaTree::to_host_tree` yields the
    `lgbm_model::Tree` and the final row->leaf layout yields `LeafPartitionLayout` (num_data / indices
    grouped by leaf / leaf_begin / leaf_count). Wire the `lib.rs` `grow_tree_on_device` body: when the
    gated discriminator is active, call the driver and return `Ok(Some((tree, layout)))`; otherwise
    `Ok(None)`. Keep every new driver kernel f32 + u64 fixed-point with NO f64 per-row grow/build hot
    loop (ODL-19); f64 only where the reference gain/leaf-value math already uses it. Bounds-guard every
    launcher; confine `unsafe` to launch sites. Scope to continuous features + L2 (no RenewTreeOutput —
    Pitfall 4); document the ordering contract for L1/quantile follow-up in a code comment.
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute && cargo test --workspace</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test --workspace` is GREEN with `LGBM_CUDA_ON_DEVICE` unset — the driver is unreachable
      (gated), CPU/ROCm/host-CUDA byte-unchanged (D-09/ODL-19 hard merge gate).
    - `grep -rn 'lgbm_treelearner' crates/lgbm-compute/src` returns nothing (no `use lgbm_treelearner` /
      `lgbm_treelearner::` — the real invariant: NO treelearner dependency edge from lgbm-compute). Do NOT
      grep bare `FeatureColumn|LeafSplits|HistogramPool` — those match 8 pre-existing COMMENT lines
      (`CUDALeafSplitsStruct` x3, host `HistogramPool` doc-comments x5) and give a false positive. The
      driver reproduces the host order with its OWN state, never importing a treelearner type.
    - The `lib.rs` `grow_tree_on_device` body composes the Phase-16/17/18 kernels (no re-implemented
      histogram/split/partition) and returns `Some((Tree, LeafPartitionLayout))` when the gate is active.
    - Manual grep review of the driver's build/partition kernels confirms no f64 per-row grow/build hot
      loop was introduced (u64 fixed-point build retained); the reviewed kernel list is captured in the
      SUMMARY (ODL-19).
  </acceptance_criteria>
  <done>grow_tree_on_device grows a full continuous-feature L2 tree on device by sequencing the existing kernels with the §16 ordering and 20-03a's locked buffer strategy, using the driver's own bookkeeping, and returns (Tree, LeafPartitionLayout) with no crate cycle and no f64 per-row hot loop.</done>
</task>

<task type="auto">
  <name>Task 2: Activate the STRUCTURE bit-exact gate vs the cpu f64 anchor (ODL-18) + ODL-19 no-f64 review</name>
  <files>crates/oracle-harness/tests/learner_parity.rs</files>
  <read_first>
    - crates/oracle-harness/tests/learner_parity.rs (lines 2185-2233 `assert_on_device_tree_matches_cpu_anchor` — tie-aware default_left acceptance at 2217-2233, ROCM_LEAF_VALUE_TOL, SPLIT_GAIN_TIE_TOL; 2245 `cpu_anchor_tree`; 2245-2272 the `host_grow` stand-in to REPLACE; 2442-2500 the Slice-0 cells + the `let backend = CpuBackend; cpu_client()` construction idiom; note the rocm cell at 1965 is `#[cfg(feature="rocm")]` — the primary gate here is the DEFAULT cpu build, NOT rocm)
    - crates/lgbm-treelearner/src/learner.rs (lines 485-505 the `on_device_eligible` cache; 696-724 the fork that now reaches the REAL driver)
    - .planning/phases/20-on-device-score-updater-metrics/20-PATTERNS.md (learner_parity.rs section — activate the existing oracle with a REAL on-device tree; anchor is ALWAYS cpu_anchor_tree, never a second GPU path)
    - .planning/phases/20-on-device-score-updater-metrics/20-RESEARCH.md (D-06 layer 3 = this structure gate; Validation Architecture Test Map: the on-device gated run command)
    - MEMORY: def-f8u-01 (never compare two nondeterministic GPU f32 paths); on-device-kernel-goldens-are-retranscriptions (the anchor is the cpu f64 fold construction, which IS the merge-gate reference)
  </read_first>
  <action>
    In `learner_parity.rs`, add a NEW default-cpu-build STRUCTURE gate cell named EXACTLY
    `learner_parity_on_device_structure_gate` (NOT `#[cfg(feature="rocm")]`): with `LGBM_CUDA_ON_DEVICE=1`,
    grow a continuous-feature L2 tree through the REAL driver on the cubecl-cpu runtime (construct the
    backend that grows on-device on the cpu runtime per 20-03a's gated CpuBackend flip, or drive
    `grow_tree_on_device` directly), build the `cpu_anchor_tree` for the SAME inputs, and call
    `assert_on_device_tree_matches_cpu_anchor(&on_device_tree, &cpu_anchor_tree(..), "on-device")`. The
    anchor is ALWAYS the cubecl-cpu f64 fold — NEVER a second GPU f32 path (D-07/def-f8u-01). Do NOT reuse
    the `host_grow` stand-in — the cell exercises a GENUINE on-device tree (also delete/retire the now-dead
    `host_grow` stand-in helper at learner_parity.rs:2245-2272 if nothing else references it). Assert
    structure fields bit-exact, leaf values within `ROCM_LEAF_VALUE_TOL`, and default_left flips accepted
    ONLY on a genuine f32-vs-f64 split_gain near-tie (identical threshold + identical child row-counts); a
    non-tie flip hard-fails. If an additional rocm-hardware variant is added it goes in a SEPARATE
    `#[cfg(feature="rocm")]` cell run with `--features rocm` and is NOT the primary gate.

    ALSO (WARNING 2 — retire the now-obsolete rocm-gated Slice-0 assertions): the two
    `#[cfg(feature="rocm")] mod hip` cells `learner_parity_on_device_oracle_host_fallback_slice0` (:2442)
    and `learner_parity_on_device_seam_is_provable_noop_slice0` (:2471) — which 20-03a migrated to the
    5-arg signature but which still assert the Slice-0 `grow_tree_on_device == Ok(None)` and (for the
    noop cell) the gated-false discriminator — are now INVALID once the driver returns `Some`. Rewrite them
    to match the activated driver: replace the `Ok(None)` assertion with an assertion that, with
    `LGBM_CUDA_ON_DEVICE=1`, `grow_tree_on_device(&g,&h,&grow_features,num_leaves,max_depth)` returns
    `Ok(Some(_))` and the grown tree is STRUCTURE-bit-exact to `cpu_anchor_tree` (the same discipline as
    the default cell); KEEP the invariant that with the env unset the discriminator is `false` and the seam
    returns `Ok(None)` (byte-unchanged). Leave no known-broken cell under `cargo test --features rocm`.
    Record the ODL-19 no-f64 review: a grep-based inspection over the driver's new build/partition kernels
    confirming no f64 per-row grow/build hot loop (f64 permitted only in the reference-blessed scalar/gain
    math) — capture the reviewed kernel list + finding in the SUMMARY. Keep the gate behind the env so a
    fresh `cargo test --workspace` (env unset) is unchanged.
  </action>
  <verify>
    <automated>LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_structure_gate 2>&1 | tee /dev/stderr | grep -q 'test result: ok. 1 passed' && cargo test --workspace</automated>
  </verify>
  <acceptance_criteria>
    - The STRUCTURE gate verify uses `-- --exact learner_parity_on_device_structure_gate` piped through
      `grep -q 'test result: ok. 1 passed'`, so a zero-match (vacuous) run FAILS the gate; the cell is in
      the DEFAULT cpu build (NOT `#[cfg(feature="rocm")]`) and the command proves exactly 1 test ran.
    - The on-device tree is STRUCTURE bit-exact to `cpu_anchor_tree` with leaf values within
      `ROCM_LEAF_VALUE_TOL`; the test compares ONLY against the cpu f64 anchor (no GPU-vs-GPU); the
      `host_grow` stand-in is retired.
    - Any default_left flip is corroborated by identical threshold + identical child row-counts + a
      split_gain near-tie (a non-tie flip hard-fails).
    - Under `cargo test --features rocm`, the two Slice-0 cells are rewritten to assert `Ok(Some(_))` +
      STRUCTURE bit-exactness with the env set (and `false`/`Ok(None)` with the env unset) — no cell asserts
      the retired Slice-0 `Ok(None)`-when-active contract.
    - The SUMMARY records the reviewed driver kernels and confirms no f64 per-row grow/build hot loop
      (ODL-19); `cargo test --workspace` (env unset) stays green and byte-unchanged.
  </acceptance_criteria>
  <done>The STRUCTURE gate asserts a real on-device tree bit-exact to the cpu f64 anchor with tie-aware default_left, self-verifies non-vacuously in the default cpu merge-gate lane, the obsolete rocm Slice-0 assertions are rewritten to the activated-driver contract, and the ODL-19 no-f64 review is captured; the env-unset merge gate is unchanged.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host learner -> device grow driver | gradients/hessians + GrowFeature metadata cross to the resident device grow loop; the driver reads/writes device index/histogram/partition buffers across kernel launches |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-03b-01 | Tampering | out-of-bounds device read/write across the sequenced driver kernels | mitigate | Bounds-guard every launcher (`i < len`); size histogram/index/partition buffers exactly; keep `unsafe` confined to each `launch_unchecked` site (the Phase-16/17/18 launchers already do this) |
| T-20-03b-02 | Tampering | data->leaf map buffer aliasing corrupting partition results at num_leaves>2 | mitigate | Use 20-03a's LOCKED buffer strategy (double-buffer unless alias proven safe); the STRUCTURE gate at num_leaves>2 catches corruption |
| T-20-03b-03 | Tampering | parent-histogram read before fully built (subtract-larger ordering) yields wrong child | mitigate | Enforce parent-fully-built-before-child-subtract ordering (the 8aed100-class guard); the STRUCTURE gate catches a wrong child |
| T-20-03b-04 | Tampering | discriminator over-claim mutating the ROCm/host-CUDA/CPU path with env unset | mitigate | The gate is `cuda_on_device_enabled()` (from 20-03a); `cargo test --workspace` env-unset proves byte-unchanged (D-09) |
| T-20-03b-05 | Tampering | integer overflow in per-leaf offset / leaf-index arithmetic across the loop | mitigate | Reuse the existing `usize`/`i32` bounds-checked index patterns from data_partition.rs; validate leaf indices against num_leaves |
| T-20-03b-SC | Tampering | npm/pip/cargo installs | accept | No new package installs this plan |
</threat_model>

<verification>
- `cargo test --workspace` green with LGBM_CUDA_ON_DEVICE unset (byte-unchanged merge gate; no crate cycle).
- `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_structure_gate 2>&1 | tee /dev/stderr | grep -q 'test result: ok. 1 passed'` — self-verifying NON-vacuous STRUCTURE gate in the default cpu build (bit-exact vs cpu f64 anchor).
- No f64 per-row grow/build hot loop introduced (reviewed + recorded in SUMMARY); u64 fixed-point build retained.
- `grep -rn 'lgbm_treelearner' crates/lgbm-compute/src` finds nothing (no treelearner dependency edge — the real no-crate-cycle invariant; a bare `FeatureColumn|LeafSplits|HistogramPool` grep false-positives on 8 pre-existing comment lines).
- `cargo test --features rocm` compiles + the two rewritten Slice-0 cells pass (Ok(Some(_)) + STRUCTURE bit-exact with env set; false/Ok(None) with env unset).
</verification>

<success_criteria>
grow_tree_on_device grows a full continuous-feature L2 tree on device by composing the existing
Phase-16/17/18 kernels with the §16 ordering and 20-03a's locked buffer strategy (using the driver's
own bookkeeping, no treelearner types); the grown tree is STRUCTURE bit-exact to the cpu f64 anchor
with tie-aware default_left and leaf values within ROCM_LEAF_VALUE_TOL, proven by a non-vacuous
default-cpu-build gate; the no-f64 constraint is verified; and the env-unset merge gate is byte-unchanged.
</success_criteria>

<artifacts_produced>
## Artifacts This Plan Produces
- `crates/lgbm-compute/src/kernels/grow_driver.rs`: the per-leaf best-first grow orchestration
  (root init -> build smaller/subtract larger -> best-split -> break on best_leaf == -1 ->
  DeviceCudaTree::split_on_device -> partition/update_data_index_to_leaf, returning (Tree,
  LeafPartitionLayout)) using the driver's own bookkeeping (no LeafSplits/HistogramPool).
- `crates/lgbm-compute/src/lib.rs`: `grow_tree_on_device` body wired to the driver (returns
  `Some((Tree, LeafPartitionLayout))` when the gated discriminator is active).
- `crates/oracle-harness/tests/learner_parity.rs`: new default-cpu-build STRUCTURE gate cell
  `learner_parity_on_device_structure_gate` running a REAL on-device tree through
  `assert_on_device_tree_matches_cpu_anchor` on the cpu f64 anchor (host_grow stand-in retired);
  self-verifying non-vacuous. PLUS the two rocm-gated Slice-0 cells rewritten to the activated-driver
  contract (`Ok(Some(_))` + STRUCTURE bit-exact with env set; `false`/`Ok(None)` with env unset).
- ODL-19 no-f64 review record (reviewed driver kernel list) in the SUMMARY.
- Env behavior: on-device growth reachable only when `LGBM_CUDA_ON_DEVICE=1`; env unset =
  byte-unchanged CPU/ROCm/host-CUDA.
</artifacts_produced>

<output>
Create `.planning/phases/20-on-device-score-updater-metrics/20-03b-SUMMARY.md` when done.
</output>
</content>
</invoke>
