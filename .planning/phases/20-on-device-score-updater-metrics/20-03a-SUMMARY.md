---
phase: 20-on-device-score-updater-metrics
plan: 03a
subsystem: infra
tags: [cubecl, on-device-growth, tree-learner, crate-cycle, gated-flip, buffer-strategy]

# Dependency graph
requires:
  - phase: 20-00
    provides: Phase-20 kernel module stubs + LGBM_CUDA_ON_DEVICE seam scaffolding
  - phase: 20-01
    provides: on-device score updater (§11) kernels
  - phase: 20-02
    provides: on-device pointwise-metric (§12) evaluator
  - phase: 18-on-device-best-split-finder
    provides: update_data_index_to_leaf_on + partition_leaf_stable (the §9 mark->prefix-sum->scatter row router)
provides:
  - "GrowFeature additive metadata carrier (lgbm-compute-local; no treelearner->compute->treelearner crate cycle)"
  - "expanded 5-arg Backend::grow_tree_on_device seam (additive features slice; body still Ok(None))"
  - "gated on_device_growth_supported() flip on CpuBackend AND GpuBackend<R> (cuda_on_device_enabled())"
  - "learner.rs on-device fork wires Vec<GrowFeature> from self.features"
  - "data->leaf map buffer-strategy A/B harness (Alias vs DoubleBuffer) + LOCKED decision (double-buffer)"
affects: [20-03b, on-device-growth, tree-learner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive lgbm-compute-local mirror struct (GrowFeature) to cross a seam without naming an upper-crate type (acyclic Option A)"
    - "Gated discriminator flip (env-derived cuda_on_device_enabled()) instead of a bare true — byte-unchanged when env unset"
    - "Anchor-pinned buffer-strategy A/B (both strategies vs the cpu f64 partition, never GPU-vs-GPU / strategy-vs-strategy)"

key-files:
  created:
    - crates/lgbm-compute/src/kernels/grow_driver.rs
  modified:
    - crates/lgbm-compute/src/kernels/mod.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "Data->leaf map buffer strategy LOCKED = DOUBLE-BUFFER (conservative default, no read/write aliasing of one map buffer). A/B evidence: both Alias and DoubleBuffer produced a row->leaf map byte-equal to the cpu f64 anchor partition at num_leaves>2, so alias is not proven UNSAFE — but double-buffer is locked per RESEARCH Pitfall 3 / phase18-wr01 HistArena::swap aliasing risk."
  - "GrowFeature carries the numeric-grow-loop fields only (drops the categorical bin_to_category table) — the on-device numeric grow loop does not consume it this milestone."
  - "Gated flip extended to CpuBackend (not only GpuBackend<R>) so 20-03b grows the on-device tree on the cubecl-cpu runtime INSIDE the default merge gate (the cpu-f64 anchor lane), with the env gate keeping env-unset byte-unchanged."

patterns-established:
  - "Cross-crate seam metadata without a cycle: mirror the upper-crate struct field-by-field in the lower crate using only lower-crate-reachable types."
  - "Self-verifying (non-vacuous) test gate: `-- --exact <cell> | grep -q 'test result: ok. 1 passed'` fails on a zero-match run."

requirements-completed: [ODL-18, ODL-19]

# Metrics
duration: 25 min
completed: 2026-07-02
status: complete
---

# Phase 20 Plan 03a: On-Device Grow-Driver Plumbing Slice Summary

**Landed the SAFE, byte-unchanged half of the pulled-forward on-device grow driver: the additive `GrowFeature` seam metadata (no crate cycle), the gated `on_device_growth_supported()` flip on both backends, the learner call-site wiring, and an anchor-pinned data->leaf map buffer-strategy A/B that LOCKS double-buffer — with the driver body still deferring (`Ok(None)`) until 20-03b.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-02T04:00:00Z (approx)
- **Completed:** 2026-07-02T04:24:22Z
- **Tasks:** 2 completed
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments
- New `GrowFeature` struct (lgbm-compute-local, over `BinColumn` + `lgbm-dataset` `BinType`/`MissingType` + primitives) carries the feature/bin metadata a bit-exact grow loop needs **without** naming any `lgbm-treelearner` type — the verified `treelearner -> compute -> treelearner` crate cycle is not reintroduced.
- `Backend::grow_tree_on_device` expanded to a 5-arg signature (additive `features: &[GrowFeature]`); body stays `Ok(None)` on the trait default and both explicit impls.
- `on_device_growth_supported()` flipped to `cuda_on_device_enabled()` on **CpuBackend AND GpuBackend<R>** — gated, not a bare `true`; with `LGBM_CUDA_ON_DEVICE` unset both are `false` and the CPU/ROCm/host-CUDA paths are byte-unchanged.
- `learner.rs` fork (train_inner) builds `Vec<GrowFeature>` from `self.features` and passes `&grow_features`; dead when the env is unset (never allocated).
- Data->leaf map buffer-strategy A/B harness (`build_leaf_map_on` over the real Phase-18 `update_data_index_to_leaf_on` kernel) + the anchor-pinned oracle cell that runs BOTH strategies at `num_leaves>2` and pins each to the cpu f64 partition — **double-buffer LOCKED** for 20-03b.

## Task Commits

1. **Task 1: GrowFeature seam + gated flip + call-site wiring** - `fe97e7e` (feat)
2. **Task 2: buffer-strategy A/B lock + seam-defers proof + rocm Slice-0 migration** - `8a14999` (test)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/grow_driver.rs` (created) - `GrowFeature` additive metadata struct + the `LeafMapBufferStrategy`/`LeafMapStep`/`build_leaf_map_on` buffer-strategy A/B helper.
- `crates/lgbm-compute/src/kernels/mod.rs` - registered `pub mod grow_driver`.
- `crates/lgbm-compute/src/lib.rs` - re-export `GrowFeature`; expanded `grow_tree_on_device` signature; gated `on_device_growth_supported()` overrides on `CpuBackend` + `GpuBackend<R>`.
- `crates/lgbm-treelearner/src/learner.rs` - the on-device fork builds + passes `Vec<GrowFeature>`.
- `crates/oracle-harness/tests/learner_parity.rs` - two new default-cpu cells (`learner_parity_on_device_buffer_strategy_ab`, `learner_parity_on_device_seam_defers`) + `grow_features_of` helper; migrated the two rocm Slice-0 cells to the 5-arg signature + gated-false invariant.

## Decisions Made
See `key-decisions` frontmatter. Headline: **data->leaf map buffer strategy = DOUBLE-BUFFER (locked)**; A/B evidence recorded below.

### Buffer-strategy A/B evidence (Pitfall 3)
- Scenario: 12-row corpus feature (6 bins), two splits reaching `num_leaves=3` (the corruption-catcher regime).
- Both `LeafMapBufferStrategy::Alias` (one running map read+written in place) and `LeafMapBufferStrategy::DoubleBuffer` (read source, write distinct destination, swap) drove the real Phase-18 `update_data_index_to_leaf_on` device kernel per split.
- Each strategy's final row->leaf map was asserted **byte-equal to the cpu f64 anchor partition** (`partition_leaf_stable`), never strategy-vs-strategy (def-f8u-01).
- Result: both matched the anchor. Alias is therefore not proven UNSAFE at this plumbing stage, but **double-buffer is LOCKED** as the conservative default (no read/write aliasing of a single map buffer) — the strategy 20-03b's driver will use.

## Deviations from Plan

None - plan executed exactly as written.

The only micro-adjustment: the plan's Task-1 acceptance criterion required the literal token `FeatureColumn` to be absent from `crates/lgbm-compute/src`. Initial doc comments referenced it in prose; reworded to "the learner's spine feature column" so the criterion's grep returns nothing (and no broken intra-doc links). This is a wording change within the same task, not a scope/behavior deviation.

## Issues Encountered
None.

## Verification Results

All gates run exactly as specified in the plan.

- `cargo test --workspace` (LGBM_CUDA_ON_DEVICE unset) — GREEN, zero failures (80 `test result: ok` lines; the hard merge gate is byte-unchanged, no crate cycle).
- `grep -rn 'FeatureColumn' crates/lgbm-compute/src` — NONE. `grep -rn 'lgbm_treelearner' crates/lgbm-compute/src` — NONE (no treelearner dependency edge).
- Gate 1 — `LGBM_CUDA_ON_DEVICE=1 ... --exact learner_parity_on_device_buffer_strategy_ab | grep -q 'test result: ok. 1 passed'`:
  `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 30 filtered out` — PASS (non-vacuous, 1 test).
- Gate 2 — `LGBM_CUDA_ON_DEVICE=1 ... --exact learner_parity_on_device_seam_defers | grep -q 'test result: ok. 1 passed'`:
  `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 30 filtered out` — PASS (non-vacuous, 1 test).
- Exact chained verify command (both gates `&&`-chained via `tee /dev/stderr | grep -q`): CHAINED EXIT 0.
- `cargo test -p oracle-harness --test learner_parity --features rocm --no-run` — compiles (migrated Slice-0 cells against the 5-arg signature).
- Migrated rocm Slice-0 cells RAN on the local (spoofed-APU) GPU:
  `test hip::learner_parity_on_device_seam_is_provable_noop_slice0 ... ok` (`1 passed`),
  `test hip::learner_parity_on_device_oracle_host_fallback_slice0 ... ok` (`1 passed`).
- New default-cpu cells also GREEN under the env-unset workspace run (`2 passed`).

## Next Phase Readiness
The safe plumbing 20-03b builds on is in place and independently proven: the seam carries the additive metadata, the discriminator flip is gated (env-unset byte-unchanged), the call-site is wired, and the buffer strategy is A/B-locked (double-buffer) against the cpu anchor. 20-03b can now write the per-leaf on-device grow-driver body (deriving `FeatureMeta` from `GrowFeature`) and activate the STRUCTURE gate, retiring the `Ok(None)` defer assertions.

No blockers.

---
*Phase: 20-on-device-score-updater-metrics*
*Completed: 2026-07-02*

## Self-Check: PASSED
- FOUND: crates/lgbm-compute/src/kernels/grow_driver.rs
- FOUND commit fe97e7e (feat 20-03a)
- FOUND commit 8a14999 (test 20-03a)
- FOUND cells: learner_parity_on_device_buffer_strategy_ab, learner_parity_on_device_seam_defers (default cpu build)
- FOUND migrated rocm Slice-0 cells passing (5-arg signature + gated invariant)
