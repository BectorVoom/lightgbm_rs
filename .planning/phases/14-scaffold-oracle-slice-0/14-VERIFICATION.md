---
phase: 14-scaffold-oracle-slice-0
verified: 2026-06-29T00:00:00Z
status: passed
score: 3/3 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 3/3 (1 flagged for human review — WR-03)
  gaps_closed:
    - "WR-03: tie-aware default_left guard in assert_on_device_tree_matches_cpu_anchor is no longer tautological — re-keyed on split_gain (the one node-level field the strict structural body does NOT pin bit-equal), so a default_left flip whose recorded gain gap exceeds SPLIT_GAIN_TIE_TOL (1e-3 rel) now hard-fails. Doc-comment corrected to describe the gain-based predicate."
  gaps_remaining: []
  regressions: []
---

# Phase 14: Scaffold + Oracle (Slice 0) Verification Report

**Phase Goal:** The on-device growth seam and its anchor-pinned oracle exist with ZERO behavior change — wiring risk is isolated from kernel risk, and the merge gate is proven before any kernel is written.
**Verified:** 2026-06-29
**Status:** passed
**Re-verification:** Yes — after WR-03 gap closure (commit 9609099)

## Goal Achievement

The phase goal — an additive on-device growth seam + anchor-pinned oracle with ZERO behavior change, merge gate proven before any kernel — is **achieved and now authoritative**. The byte-unchanged invariant holds both structurally (the on-device fork is gated behind an AND that short-circuits to a statically dead branch on every Slice-0 backend; the oracle change is entirely inside `#[cfg(feature = "rocm")] mod hip`, so the default/CPU build does not even compile it) and empirically (the full bit-exact merge gate is green). The sole prior open item — the inert tie-aware `default_left` guard (WR-03) — has been fixed: the guard now keys on `split_gain`, an unpinned node-level field, making it genuinely reachable-as-failing. SC#3 is therefore satisfied at BINDING-quality tie-awareness, not merely scaffold. No human-verification items remain.

### WR-03 Resolution (re-verification focus)

The fix (commit 9609099, `crates/oracle-harness/tests/learner_parity.rs`) is confirmed genuine, not cosmetic:

- **The guard can now fail.** The in-loop predicate is `gain_near_tie && same_threshold && same_child_counts` (learner_parity.rs:2207-2208). `same_threshold` and `same_child_counts` are forced true by the strict body (it pins `threshold`, `leaf_count`, `internal_count` bit-exact) — these were the entire predicate before, hence the tautology. The new `gain_near_tie` term compares `split_gain[node]` against the anchor with `SPLIT_GAIN_TIE_TOL = 1e-3` relative tolerance. Verified that `split_gain` is **NOT** asserted anywhere in `assert_tree_structure_and_leaves` (lines 2087-2126 contain no `split_gain` reference) and is a real `Vec<f32>` Tree field (`lgbm-model/src/tree.rs:88`). Because `split_gain` is unconstrained by the structural body, `gain_near_tie` can evaluate `false` on a `default_left` flip → the `&&` is `false` → the `assert!` fires. The guard is reachable-as-failing; the WR-03 tautology is eliminated.
- **The discriminator is meaningful.** A genuine f32-vs-f64 near-tie flips the missing direction precisely because the two competing missing-direction gains were within f32 rounding, so the recorded `split_gain` differs only marginally → accepted. A real wrong-direction kernel divergence records a materially different gain → rejected. This correctly lifts the per-`SplitInfo` near-tie at `kernel_parity.rs:1597` to a per-node index.
- **The doc-comment is now accurate.** The false claim "A flip on a NON-tie node hard-fails" (which was true only by accident of being unreachable) is replaced with "A flip whose `split_gain` gap exceeds tolerance hard-fails (a real wrong-direction divergence stays caught — the guard is genuinely reachable-as-failing, not tautological)" (learner_parity.rs:2160-2162). The description matches the predicate.

### Observable Truths

| #   | Truth (Success Criterion) | Status | Evidence |
| --- | ------------------------- | ------ | -------- |
| SC#1 | `LGBM_CUDA_ON_DEVICE` OFF by default; CPU/ROCm/host-CUDA grow byte-identical trees; full bit-exact merge gate green & unchanged | ✓ VERIFIED | `on_device_eligible = backend.on_device_growth_supported() && cuda_on_device_env()` (learner.rs:488); discriminator false on every Slice-0 backend → `&&` short-circuits → fork at learner.rs:704 statically dead. WR-03 fix is confined to `#[cfg(feature = "rocm")] mod hip` (module opens learner_parity.rs:1965), so the default build never compiles it. Empirical: `cargo test --workspace` all suites ok, 0 failed; `learner_parity --features rocm` 33 passed; `kernel_parity --features rocm` 21 passed. |
| SC#2 | Additive `Backend::grow_tree_on_device` + default-false `on_device_growth_supported()` discriminator, routed by a decide-once early-return fork in `train_inner`; `GpuBackend<R>` override returns no-op so default path untouched | ✓ VERIFIED | Trait default `on_device_growth_supported() -> false` (lib.rs:1239), seam `grow_tree_on_device(..) -> Ok(None)` (lib.rs:1272), explicit `GpuBackend<R>` override `Ok(None)` (lib.rs:2207). Fork at top of `train_inner` (learner.rs:704), early-returns on `Some`, falls through on `Ok(None)`. SC#2 test `learner_parity_on_device_seam_is_provable_noop_slice0` asserts discriminator `== false` AND seam `== Ok(None)` on BOTH `CpuBackend` and `GpuBackend` — green. Unchanged by the WR-03 fix. |
| SC#3 | `assert_on_device_tree_matches_cpu_anchor` oracle pins tree STRUCTURE to the cpu f64 anchor (tie-aware `default_left`, now reachable-as-failing) with leaf values within ~1e-5 f32 envelope — present BEFORE any kernel, never comparing two GPU paths | ✓ VERIFIED | Comparator at learner_parity.rs:2166 delegates the 8 bit-exact structural fields + per-leaf `ROCM_LEAF_VALUE_TOL = 1e-5` envelope to factored `assert_tree_structure_and_leaves` (:2087), always anchors to `cpu_anchor_tree` (never a 2nd GPU path, def-f8u-01). Runs LIVE GREEN before any kernel via `learner_parity_on_device_oracle_host_fallback_slice0`. **Tie-awareness now BINDING-quality:** the `default_left` (bit1) guard keys on the unpinned `split_gain` (`gain_near_tie`, `SPLIT_GAIN_TIE_TOL = 1e-3`), so a non-tie flip hard-fails (WR-03 resolved). |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-dataset/src/dataset.rs` | `LeafPartitionLayout` POD (num_data, indices, leaf_begin, leaf_count) | ✓ VERIFIED | `pub struct LeafPartitionLayout` at :88, exactly the 4 public fields, no upward crate dep |
| `crates/lgbm-dataset/src/lib.rs` | re-export | ✓ VERIFIED | `pub use dataset::{..., LeafPartitionLayout}` at :25 |
| `crates/lgbm-compute/Cargo.toml` | `lgbm-model` path dep (acyclic) | ✓ VERIFIED | `lgbm-model = { path = "../lgbm-model" }` at :15 |
| `crates/lgbm-compute/src/lib.rs` | discriminator + seam on `trait Backend` + `GpuBackend<R>` no-op override | ✓ VERIFIED | trait default-false :1239, seam default-`Ok(None)` :1272, GpuBackend override :2207; no cycle |
| `crates/lgbm-treelearner/src/learner.rs` | `cuda_on_device_env()`, `on_device_eligible` field, new()-init, fork | ✓ VERIFIED | helper :443, field :306, AND-gate init :488, fork :704 |
| `crates/lgbm-treelearner/src/data_partition.rs` | `DataPartition::from_payload` | ✓ VERIFIED | `pub fn from_payload(..)` at :74 |
| `crates/oracle-harness/tests/learner_parity.rs` | tie-aware comparator (now reachable-as-failing) + 2 tests | ✓ VERIFIED | `assert_on_device_tree_matches_cpu_anchor` :2166 (split_gain-keyed tie guard), `assert_tree_structure_and_leaves` :2087, `child_row_counts` :2071, `SPLIT_GAIN_TIE_TOL` :2063, host-fallback oracle + seam no-op tests green |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| lgbm-compute/lib.rs | lgbm-dataset/dataset.rs | seam return type names `LeafPartitionLayout` | ✓ WIRED | `Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError>` (lib.rs:1278) |
| lgbm-compute/lib.rs | lgbm-model/tree.rs | seam names `lgbm_model::Tree` (acyclic) | ✓ WIRED | dep edge added; workspace builds 0 cycles |
| lgbm-treelearner/learner.rs | lgbm-compute/lib.rs | `on_device_eligible` ANDs discriminator; fork calls `grow_tree_on_device` | ✓ WIRED | learner.rs:488, :705 |
| lgbm-treelearner/learner.rs | data_partition.rs | fork reconstructs via `from_payload` | ✓ WIRED | learner.rs:711 → data_partition.rs:74 |
| oracle-harness | lgbm-compute | oracle calls seam; SC#2 asserts discriminator false | ✓ WIRED | learner_parity.rs host-fallback + seam no-op tests |
| oracle-harness | lgbm-model | comparator decodes `default_left` = bit1 (mask 2) AND reads unpinned `split_gain` | ✓ WIRED | `DEFAULT_LEFT_MASK = 2` :2046, `split_gain[node]` :2199-2200 — the live discriminator |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Workspace byte-unchanged, env unset (SC#1) | `cargo test --workspace` | all suites ok, 0 failed | ✓ PASS |
| Oracle + SC#2/SC#3 seam tests (rocm) | `cargo test -p oracle-harness --test learner_parity --features rocm` | 33 passed, 0 failed | ✓ PASS |
| Kernel parity merge gate (rocm) | `cargo test -p oracle-harness --test kernel_parity --features rocm` | 21 passed, 0 failed | ✓ PASS |
| WR-03 guard non-tautology (static) | inspect: `split_gain` absent from `assert_tree_structure_and_leaves` (:2087-2126); present as Tree field (`tree.rs:88`) | `gain_near_tie` unconstrained → assert reachable-as-failing | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| ODL-01 | 14-01, 14-02 | Additive `grow_tree_on_device` + default-false discriminator, off behind `LGBM_CUDA_ON_DEVICE`, CPU/ROCm/host-CUDA byte-unchanged | ✓ SATISFIED | Seam + discriminator + decide-once fork; merge gate green & byte-unchanged (SC#1, SC#2). REQUIREMENTS.md marks ODL-01 Complete. |
| ODL-02 | 14-03 | Anchor-pinned oracle asserts on-device tree STRUCTURE bit-exact to cpu f64 anchor (tie-aware default_left), leaf within ~1e-5, never 2 GPU paths | ✓ SATISFIED | Oracle exists, structure-pinned, leaf-enveloped, cpu-anchored, green before any kernel. Tie-aware `default_left` now genuinely reachable-as-failing (WR-03 resolved) — BINDING-quality, not scaffold. REQUIREMENTS.md marks ODL-02 Complete. |

No orphaned requirements: REQUIREMENTS.md maps Phase 14 to exactly ODL-01, ODL-02 (both Complete), both claimed by phase plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (phase diffs) | — | No TBD/FIXME/XXX/todo!/unimplemented! in added lines | ℹ️ Info | Clean |
| learner_parity.rs | 2160-2162 | Doc-comment now accurately describes the gain-based predicate | ℹ️ Info | WR-03 RESOLVED — claim matches code |
| learner.rs | 704-714 | On-device fork ignores `capture_snapshots` and sits before V5 validation | ℹ️ Info (deferred) | WR-01/WR-02 — dormant in Slice 0 (fork statically dead); latent Slice-1 risk captured in 14-REVIEW.md; does not affect the Phase 14 zero-behavior-change goal |

### Human Verification Required

None. The sole prior `human_needed` item (WR-03) is resolved in code and re-verified above. WR-01/WR-02 are dormant Slice-1 latent risks captured in 14-REVIEW.md, explicitly out of scope for the Phase 14 zero-behavior-change goal (the fork is statically dead in Slice 0); they become live work when Slice 1 wires a real kernel.

### Gaps Summary

No gaps. The seam is additive and provably off (discriminator false + `Ok(None)` no-op on every Slice-0 backend, AND-gate short-circuit → statically dead fork). The WR-03 oracle fix is confined to the `#[cfg(feature = "rocm")] mod hip` test module, so the default/CPU build is untouched at the source level; empirically the full merge gate (`cargo test --workspace`, `learner_parity --features rocm` 33/33, `kernel_parity --features rocm` 21/21) is green and byte-unchanged. The tie-aware `default_left` guard is now genuinely conditional — keyed on the unpinned `split_gain`, it hard-fails a real wrong-direction divergence — so SC#3 is satisfied at binding tie-awareness quality. All three success criteria VERIFIED, no human-verification items remain.

---

_Verified: 2026-06-29_
_Verifier: Claude (gsd-verifier)_
_Re-verification: WR-03 closed (commit 9609099)_
