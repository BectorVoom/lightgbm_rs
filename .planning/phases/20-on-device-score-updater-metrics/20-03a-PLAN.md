---
phase: 20-on-device-score-updater-metrics
plan: 03a
type: execute
wave: 3
depends_on: [20-00, 20-01, 20-02]
files_modified:
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/src/kernels/grow_driver.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/oracle-harness/tests/learner_parity.rs
autonomous: true
requirements: [ODL-18, ODL-19]
must_haves:
  truths:
    - "grow_tree_on_device gains ADDITIVE feature/bin-metadata parameters expressed in ONLY lgbm-compute-reachable types (a new lgbm-compute GrowFeature struct over BinColumn + lgbm-dataset BinType/MissingType + primitive slices) — it NEVER names lgbm-treelearner's FeatureColumn/LeafSplits/HistogramPool, so no treelearner->compute->treelearner crate cycle is introduced (Option A, D-01)"
    - "on_device_growth_supported() returns cuda_on_device_enabled() on the cpu-reachable backend AND on GpuBackend<R>, so with LGBM_CUDA_ON_DEVICE unset it stays false and the CPU/ROCm/host-CUDA paths are byte-unchanged (Pitfall 2, D-09)"
    - "The learner fork at learner.rs:714 builds the Vec<GrowFeature> from self.features (field-by-field from FeatureColumn) and passes it to grow_tree_on_device; grow_tree_on_device still returns Ok(None) this plan, so with the env SET the fork safely falls through to the byte-identical host path (output still correct, on-device not yet active)"
    - "The data->leaf map Handle buffer strategy (in-place alias vs ping-pong double-buffer, Pitfall 3) is A/B-tested against the cpu f64 anchor at num_leaves>2 and LOCKED (double-buffer unless alias proven safe), so 20-03b builds on a decided strategy"
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/grow_driver.rs"
      provides: "GrowFeature additive metadata struct (reachable types only) + the Handle buffer-strategy A/B harness helper; the per-leaf driver body itself lands in 20-03b"
      contains: "struct GrowFeature"
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "expanded grow_tree_on_device signature (additive GrowFeature slice) + gated on_device_growth_supported() flip; body still Ok(None)"
      contains: "grow_tree_on_device"
    - path: "crates/lgbm-treelearner/src/learner.rs"
      provides: "call-site builds Vec<GrowFeature> from self.features and passes the additive args"
    - path: "crates/oracle-harness/tests/learner_parity.rs"
      provides: "Handle alias-vs-double-buffer A/B cell (anchor-pinned) + a cell proving the fork safely defers to host with the env set"
  key_links:
    - from: "crates/lgbm-treelearner/src/learner.rs"
      to: "crates/lgbm-compute/src/lib.rs"
      via: "the on_device_eligible fork passes the new Vec<GrowFeature> into grow_tree_on_device"
      pattern: "grow_tree_on_device"
    - from: "crates/lgbm-compute/src/kernels/grow_driver.rs"
      to: "crates/lgbm-compute/src/lib.rs"
      via: "GrowFeature + BinColumn are lgbm-compute-local; BinType/MissingType are lgbm-dataset — all reachable without a treelearner import"
      pattern: "GrowFeature|BinColumn"
---

<objective>
De-risk the D-01 pulled-forward driver by landing the SAFE, byte-unchanged plumbing slice
that the old 2-file 20-03 lacked: (1) expand `Backend::grow_tree_on_device` with ADDITIVE
feature/bin-metadata parameters using ONLY lgbm-compute-reachable types (a new `GrowFeature`
struct over `BinColumn` + lgbm-dataset `BinType`/`MissingType` + primitive slices — never
lgbm-treelearner's `FeatureColumn`), (2) flip `on_device_growth_supported()` GATED behind
`cuda_on_device_enabled()` on the cpu-reachable backend AND `GpuBackend<R>` (Pitfall 2), (3)
update the learner.rs:714 fork to build the `Vec<GrowFeature>` from `self.features` and pass it,
and (4) A/B-test and LOCK the data->leaf map `Handle` buffer strategy (Pitfall 3) against the cpu
anchor.

This resolves the verified crate-cycle blocker WITHOUT naming any treelearner type: the
Phase-16/17/18 kernels already consume `FeatureMeta`/`BinColumn`/`DeviceCudaTree` +
`BinType`/`MissingType`, never `FeatureColumn` — so a driver in lgbm-compute is fully
deliverable (Option A, honoring D-01). The `grow_tree_on_device` BODY stays `Ok(None)` this plan
(the fork safely defers to the byte-identical host path with the env set), so 20-03a is
independently verifiable NOW; the real per-leaf orchestration body + the activated STRUCTURE gate
are 20-03b.

Purpose: separate the mechanical, provably-safe signature/plumbing/gated-flip/buffer-decision work
(shippable and reviewable in isolation) from the load-bearing bit-exact driver body — so a failure
in the hard part cannot silently corrupt the safe part.
Output: expanded seam + gated flip in `lib.rs`, the `GrowFeature` struct + buffer A/B harness in
new `kernels/grow_driver.rs`, the call-site wiring in `learner.rs`, and the two safety cells in
`learner_parity.rs`.
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
@crates/lgbm-compute/src/kernels/best_split.rs
@crates/lgbm-compute/src/kernels/data_partition.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add GrowFeature metadata struct + expand grow_tree_on_device signature + gated flip + call-site wiring</name>
  <files>crates/lgbm-compute/src/kernels/grow_driver.rs, crates/lgbm-compute/src/lib.rs, crates/lgbm-treelearner/src/learner.rs</files>
  <read_first>
    - crates/lgbm-compute/src/lib.rs (lines 1225-1318 — the seam doc that itself says "Richer feature/bin inputs arrive in Slice 1 as ADDITIVE parameters"; the `Ok(None)` `grow_tree_on_device` body at 1285; the `on_device_growth_supported()` default-false at 1242; `cuda_on_device_enabled()` at 1314; the `BinColumn` enum at lib.rs:55; the `CpuBackend` impl at 1337 and the `#[cfg(feature="gpu")] GpuBackend<R>` impl at 2221)
    - crates/lgbm-treelearner/src/learner.rs (lines 87-198 — the FeatureColumn field set the GrowFeature struct mirrors 1:1: bins/num_bin/offset/min_bin/max_bin/default_bin/most_freq_bin/missing_type/bin_upper_bound/real_feature_index/bin_type; lines 714-724 — the fork call site to update; line 210 `features: Vec<FeatureColumn>`)
    - crates/lgbm-compute/src/kernels/best_split.rs (lines 119-140 — FeatureMeta, the lgbm-compute-local per-feature struct the split kernel already consumes; the driver derives FeatureMeta from GrowFeature internally in 20-03b)
    - crates/lgbm-dataset/src/bin_mapper.rs (lines 45-60 — BinType + MissingType, the lgbm-dataset enums GrowFeature reuses; confirms they are BELOW lgbm-compute and reachable)
    - crates/lgbm-compute/Cargo.toml (confirms lgbm-compute depends on lgbm-core + lgbm-dataset + lgbm-model ONLY — NOT lgbm-treelearner; naming FeatureColumn here is the crate-cycle the blocker verified)
    - .planning/phases/20-on-device-score-updater-metrics/20-PATTERNS.md (the lib.rs seam-activation section: LeafPartitionLayout payload shape; Pitfall 2 gated-flip note)
    - MEMORY: def-f8u-01 (never GPU-vs-GPU); spike-052 (no f64 per-row hot loops)
  </read_first>
  <action>
    In new `crates/lgbm-compute/src/kernels/grow_driver.rs`, define `pub struct GrowFeature` mirroring
    the fields the Phase-16/17/18 kernels need from FeatureColumn, using ONLY lgbm-compute-reachable
    types: `bins: BinColumn` (lgbm-compute), `num_bin: u32`, `offset: i32`, `min_bin: u32`,
    `max_bin: u32`, `default_bin: u32`, `most_freq_bin: u32`, `missing_type: lgbm_dataset::MissingType`,
    `bin_upper_bound: Vec<f64>`, `real_feature_index: i32`, `bin_type: lgbm_dataset::BinType`. Do NOT
    import or name `FeatureColumn` (or any lgbm-treelearner type) anywhere in lgbm-compute — that is the
    verified crate cycle. Register the module in `lib.rs` (`pub mod grow_driver;` under kernels) and
    re-export `GrowFeature`. Expand the `Backend::grow_tree_on_device` trait signature to take an
    ADDITIVE `features: &[GrowFeature]` parameter (alongside the existing gradients/hessians/num_leaves/
    max_depth); keep the return type EXACTLY `Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError>`
    (never name treelearner's DataPartition). Keep the default body `Ok(None)` on the trait and on
    every impl this plan (the real body is 20-03b). Flip `on_device_growth_supported()` to return
    `cuda_on_device_enabled()` on BOTH the cpu-reachable backend (`CpuBackend`) AND `GpuBackend<R>` —
    NOT a bare `true` — so with `LGBM_CUDA_ON_DEVICE` unset every backend reports false and the host
    path is byte-unchanged (Pitfall 2, D-09). Extending the gated flip to `CpuBackend` is deliberate:
    the STRUCTURE gate (20-03b) must grow the on-device tree on the cubecl-cpu runtime so it runs in
    the DEFAULT merge gate (the hard cpu-f64 anchor lane), not behind rocm hardware; the env gate keeps
    the merge gate (env unset) byte-unchanged. In `learner.rs`, at the fork (714-724), build a
    `Vec<GrowFeature>` from `self.features` by mapping each `FeatureColumn` field-by-field into
    `GrowFeature`, and pass `&grow_features` as the new argument to `grow_tree_on_device`. Bounds/
    length discipline stays at the boundary (V5). Confine any `unsafe` to launch sites (none added here).
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute && cargo build -p lgbm-treelearner && cargo test --workspace</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test --workspace` is GREEN with `LGBM_CUDA_ON_DEVICE` unset — every backend's
      `on_device_growth_supported()` is false, the fork is dead, and the CPU/ROCm/host-CUDA paths are
      byte-unchanged (D-09/ODL-19 hard merge gate).
    - `grep -n 'FeatureColumn' crates/lgbm-compute/src/**/*.rs` returns nothing — lgbm-compute never
      names an lgbm-treelearner type (no crate cycle); `GrowFeature` uses only BinColumn (lgbm-compute)
      + BinType/MissingType (lgbm-dataset) + primitives.
    - `on_device_growth_supported()` returns `cuda_on_device_enabled()` on CpuBackend AND GpuBackend<R>
      (verified by inspection); with the env unset both are false.
    - The learner fork passes `&grow_features` (built from `self.features`) into the expanded
      `grow_tree_on_device`; with the env SET, `cargo test --workspace` is still green because the body
      returns `Ok(None)` and the fork falls through to the byte-identical host path.
  </acceptance_criteria>
  <done>The seam carries the feature/bin metadata a bit-exact grow loop needs, the discriminator flip is gated so env-unset is byte-unchanged, no crate cycle is introduced, and the body still safely defers — the plumbing 20-03b builds on is in place and independently proven.</done>
</task>

<task type="auto">
  <name>Task 2: Lock the data->leaf map Handle buffer strategy with an anchor-pinned A/B (Pitfall 3) + prove the fork defers safely</name>
  <files>crates/oracle-harness/tests/learner_parity.rs, crates/lgbm-compute/src/kernels/grow_driver.rs</files>
  <read_first>
    - crates/lgbm-compute/src/kernels/data_partition.rs (lines 482-525 `partition_leaf_stable` / `partition_categorical_stable`; 657-686 `partition_on_device`; 822-895 `update_data_index_to_leaf_on` — the mark->prefix-sum->scatter that reads AND writes the row->leaf index arrays in one pass; this IS the aliasing surface)
    - crates/oracle-harness/tests/learner_parity.rs (lines 2185-2245 `assert_on_device_tree_matches_cpu_anchor` + `cpu_anchor_tree`; 2245-2272 the `host_grow` stand-in; 2442-2500 the Slice-0 no-op cells — the pattern for a new anchor-pinned cell; line 331/386/463 `let backend = CpuBackend;` + `cpu_client()` construction idiom)
    - .planning/phases/20-on-device-score-updater-metrics/20-RESEARCH.md (Pitfall 3 — alias vs double-buffer; Open Question 2 RESOLVED: double-buffer is the safe default, A/B-verify, num_leaves>2 STRUCTURE gate is the corruption catcher)
    - MEMORY: phase18-wr01-histarena-swap-aliasing (the latent HistArena::swap slot-aliasing bug that "will bite the Phase-21 multi-leaf on-device grow loop" — this loop; the buffer strategy this task locks prevents it)
  </read_first>
  <action>
    Add a `grow_driver` helper that exposes the two candidate data->leaf map update strategies for the
    per-split partition rewrite: (A) in-place alias (the kernel reads and writes the same `Handle`) and
    (B) ping-pong double-buffer (read source `Handle`, write a distinct destination `Handle`, swap). In
    `learner_parity.rs`, add an anchor-pinned A/B cell named EXACTLY
    `learner_parity_on_device_buffer_strategy_ab` (default cpu build, NOT rocm-gated) that drives BOTH
    strategies over the existing Phase-18 `update_data_index_to_leaf_on` / `partition_leaf_stable` kernels
    for a multi-split case (`num_leaves > 2`) on the cubecl-cpu runtime, reconstructs the resulting
    `LeafPartitionLayout`, and asserts each strategy's row->leaf assignment matches the cpu f64 anchor's
    partition (NEVER strategy-A vs strategy-B directly — anchor ONLY to the host/cpu-fold layout,
    def-f8u-01). PREFER double-buffer: lock it as the strategy 20-03b's driver uses UNLESS the alias
    strategy is proven bit-identical to the anchor across the A/B (record the decision + the evidence in
    the SUMMARY). Also add a cell named EXACTLY `learner_parity_on_device_seam_defers` (default cpu build)
    that, with `LGBM_CUDA_ON_DEVICE=1`, calls the expanded
    `grow_tree_on_device(&g,&h,&grow_features,num_leaves,max_depth)` directly on CpuBackend and asserts it
    returns `Ok(None)` this plan (the safe-defer contract), and that a full `SerialTreeLearner` train with
    the env set produces the byte-identical tree to the env-unset host path (the fork falls through).

    ALSO (WARNING 2 — the two obsolete rocm-gated Slice-0 seam cells): migrate BOTH
    `learner_parity_on_device_oracle_host_fallback_slice0` (learner_parity.rs:2442) and
    `learner_parity_on_device_seam_is_provable_noop_slice0` (:2471), which live in
    `#[cfg(feature="rocm")] mod hip` and call the OLD 4-arg `grow_tree_on_device(&g,&h,num_leaves,max_depth)`.
    Update both to the new 5-arg signature (passing `&grow_features`). Replace the now-invalid
    UNCONDITIONAL `on_device_growth_supported()==false` assertion with the GATED invariant — assert it is
    `false` when `LGBM_CUDA_ON_DEVICE` is unset (the byte-unchanged merge-gate contract), which is still
    true after 20-03a's gated flip. Keep the `grow_tree_on_device == Ok(None)` assertion valid for THIS
    plan (the body still defers); 20-03b will retire that assertion when it activates the driver. This
    keeps `cargo test --features rocm` compiling and green — leave no known-broken cell.
  </action>
  <verify>
    <automated>LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_buffer_strategy_ab 2>&1 | tee /dev/stderr | grep -q 'test result: ok. 1 passed' && LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_seam_defers 2>&1 | tee /dev/stderr | grep -q 'test result: ok. 1 passed'</automated>
  </verify>
  <acceptance_criteria>
    - The A/B cell `learner_parity_on_device_buffer_strategy_ab` runs BOTH buffer strategies over the real
      Phase-18 partition kernels at `num_leaves>2` and asserts each against the cpu f64 anchor's partition
      (not against each other); the chosen strategy (double-buffer unless alias proven safe) is recorded in
      the SUMMARY with the A/B evidence.
    - Each verify command uses `-- --exact <cell_name>` and pipes through `grep -q 'test result: ok. 1 passed'`,
      so a zero-match (vacuous) run FAILS the gate; both cells exist in the default cpu build (NOT
      `#[cfg(feature="rocm")]`) and each command proves exactly 1 test ran.
    - `learner_parity_on_device_seam_defers` proves `grow_tree_on_device` returns `Ok(None)` this plan and the
      env-set learner train equals the env-unset host tree (the fork defers safely).
    - Under `cargo test --features rocm`, the two migrated Slice-0 cells compile against the 5-arg signature
      and pass: the discriminator is asserted `false` when the env is unset (gated invariant), and the seam
      still returns `Ok(None)` this plan — no known-broken cell remains.
  </acceptance_criteria>
  <done>The data->leaf map buffer strategy is A/B-verified against the cpu anchor and LOCKED for 20-03b (self-verifying, non-vacuous), the expanded seam is proven to defer safely with the env set, and the two obsolete rocm Slice-0 cells are migrated to the 5-arg signature + gated invariant — no broken cell in any build.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host learner -> device grow seam | the learner passes gradients/hessians + the new `GrowFeature` metadata slice across to the (still-deferring) device seam; the buffer A/B exercises the device index/partition buffers across kernel launches |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-20-03a-01 | Tampering | discriminator over-claim mutating the ROCm/host-CUDA/CPU path with env unset | mitigate | Gate the flip behind `cuda_on_device_enabled()` on BOTH CpuBackend and GpuBackend<R> (false when env unset); `cargo test --workspace` env-unset proves byte-unchanged (D-09) |
| T-20-03a-02 | Tampering | data->leaf map buffer aliasing corrupting partition results at num_leaves>2 | mitigate | A/B alias-vs-double-buffer against the cpu anchor (Pitfall 3); prefer double-buffer; lock the safe strategy before 20-03b writes the driver |
| T-20-03a-03 | Tampering | out-of-bounds device read/write when the GrowFeature metadata is mis-sized vs num_data | mitigate | Validate `grow_features.len()`, per-feature `bins` length == num_data, `num_bin`/offset bounds at the boundary (V5) before any launch; keep `unsafe` confined to launch sites |
| T-20-03a-04 | Tampering | integer overflow in per-feature offset / leaf-index arithmetic | mitigate | Reuse the existing `usize`/`i32` bounds-checked index patterns from data_partition.rs; validate indices against num_leaves |
| T-20-03a-SC | Tampering | npm/pip/cargo installs | accept | No new package installs this plan |
</threat_model>

<verification>
- `cargo test --workspace` green with LGBM_CUDA_ON_DEVICE unset (byte-unchanged merge gate; no crate cycle).
- `grep -rn 'lgbm_treelearner' crates/lgbm-compute/src` finds nothing (no treelearner dependency edge from lgbm-compute — GrowFeature uses BinColumn + lgbm-dataset types only).
- `learner_parity_on_device_buffer_strategy_ab` green via `-- --exact ... | grep -q 'test result: ok. 1 passed'` (both strategies anchored to the cpu f64 partition; non-vacuous; chosen strategy recorded).
- `learner_parity_on_device_seam_defers` green via `-- --exact ... | grep -q 'test result: ok. 1 passed'` (fork defers safely; env-set tree == env-unset host tree; non-vacuous).
- `cargo test --features rocm` compiles + passes the two MIGRATED Slice-0 cells (5-arg signature; discriminator asserted false only when env unset).
</verification>

<success_criteria>
The `grow_tree_on_device` seam carries the additive feature/bin metadata a bit-exact grow loop needs
(reachable types only, no crate cycle), the discriminator flip is gated on both backends so the
env-unset path is byte-unchanged, the learner call-site is wired, and the data->leaf map buffer
strategy is A/B-locked against the cpu anchor — the safe half of the pulled-forward driver is landed
and independently verified, with the body deferring safely until 20-03b.
</success_criteria>

<artifacts_produced>
## Artifacts This Plan Produces
- `crates/lgbm-compute/src/kernels/grow_driver.rs` (NEW): `GrowFeature` additive metadata struct
  (BinColumn + lgbm-dataset BinType/MissingType + primitive slices — never FeatureColumn) + the
  data->leaf map buffer-strategy A/B helper (alias vs double-buffer).
- `crates/lgbm-compute/src/lib.rs`: `grow_tree_on_device` signature expanded with `features: &[GrowFeature]`
  (body still `Ok(None)`); `on_device_growth_supported()` flipped to `cuda_on_device_enabled()` on
  CpuBackend AND GpuBackend<R> (gated).
- `crates/lgbm-treelearner/src/learner.rs`: the fork builds `Vec<GrowFeature>` from `self.features` and
  passes it to the expanded seam.
- `crates/oracle-harness/tests/learner_parity.rs`: anchor-pinned buffer-strategy A/B cell
  (`learner_parity_on_device_buffer_strategy_ab`) + a safe-defer cell
  (`learner_parity_on_device_seam_defers`: env-set returns Ok(None) and env-set train == env-unset host
  tree), both in the default cpu build; PLUS the two migrated rocm-gated Slice-0 cells
  (`..._host_fallback_slice0`, `..._seam_is_provable_noop_slice0`) updated to the 5-arg signature +
  gated-false invariant.
- Locked decision (SUMMARY): data->leaf map buffer strategy (double-buffer unless alias proven safe) +
  A/B evidence.
- Env behavior: on-device growth still unreachable (body defers); env unset = byte-unchanged; no crate cycle.
</artifacts_produced>

<output>
Create `.planning/phases/20-on-device-score-updater-metrics/20-03a-SUMMARY.md` when done.
</output>
</content>
</invoke>
