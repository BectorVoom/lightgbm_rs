---
phase: 14-scaffold-oracle-slice-0
verified: 2026-06-29T00:00:00Z
status: human_needed
score: 3/3 must-haves verified (1 flagged for human review)
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Decide whether the inert tie-aware default_left guard in assert_on_device_tree_matches_cpu_anchor is acceptable as a Slice-0 SCAFFOLD, and correct the misleading doc-comment before Phase 16 makes the assert binding."
    expected: "Maintainer accepts scaffold-grade for Slice 0 (ROADMAP defers the BINDING tie-aware default_left assert to Phase 16: 'tie-aware default_left assert lands here'), AND schedules the WR-03 fix: make the default_left tie genuinely conditional (relax the shared threshold compare to a documented near-tie tolerance OR assert default_left strictly in the structural body and fall to the tie path only on a proven near-tie). Until then, correct the doc-comment claim 'A flip on a NON-tie node hard-fails' which is false as written."
    why_human: "WR-03 is a judgment call about whether a present-but-inert oracle sub-component satisfies a scaffold-grade success criterion. The tautology is real (verified in code) but its impact is zero in Slice 0 (no kernel can produce a flip) and the binding assert is explicitly deferred to Phase 16. Whether to accept now vs. fix now is a maintainer decision."
---

# Phase 14: Scaffold + Oracle (Slice 0) Verification Report

**Phase Goal:** The on-device growth seam and its anchor-pinned oracle exist with ZERO behavior change — wiring risk is isolated from kernel risk, and the merge gate is proven before any kernel is written.
**Verified:** 2026-06-29
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

The phase goal — an additive on-device growth seam + anchor-pinned oracle with ZERO behavior change, merge gate proven before any kernel — is **achieved**. The byte-unchanged invariant is proven both structurally (the `on_device_eligible` AND-gate short-circuits to a statically dead fork) and empirically (the full bit-exact merge gate is green). One sub-component of SC#3 (the tie-aware `default_left` protection) is present but currently inert; per the ROADMAP it is a scaffold whose binding form is explicitly deferred to Phase 16, so it is flagged for a maintainer decision rather than treated as a blocker.

### Observable Truths

| #   | Truth (Success Criterion) | Status | Evidence |
| --- | ------------------------- | ------ | -------- |
| SC#1 | `LGBM_CUDA_ON_DEVICE` OFF by default; CPU/ROCm/host-CUDA grow byte-identical trees; full bit-exact merge gate green & unchanged | ✓ VERIFIED | `cuda_on_device_env()` = `matches!(env::var("LGBM_CUDA_ON_DEVICE").as_deref(), Ok("1"))` (learner.rs:443) — OFF unless exactly `"1"`. `on_device_eligible = backend.on_device_growth_supported() && cuda_on_device_env()` (learner.rs:488); discriminator false on every Slice-0 backend → `&&` short-circuits → fork at learner.rs:704 is statically dead. Empirical: `cargo test --workspace` all suites ok, 0 failed; `raw_bin_train_matches_cpp_golden` ok; `kernel_parity` 21 passed; `learner_parity --features rocm` 33 passed. |
| SC#2 | Additive `Backend::grow_tree_on_device` + default-false `on_device_growth_supported()` discriminator, routed by a decide-once-at-top early-return fork in `train_inner`; `GpuBackend<R>` override returns no-op so default path untouched | ✓ VERIFIED | Trait default `on_device_growth_supported() -> false` (lib.rs:1239) and `grow_tree_on_device(..) -> Ok(None)` (lib.rs:1272). Explicit `GpuBackend<R>` override returns `Ok(None)` (lib.rs:2207). Fork at the TOP of `train_inner` (learner.rs:704), ahead of `capture_snapshots` assignment (:718) and the resident block; early-returns on `Some`, falls through on `Ok(None)`. SC#2 test `learner_parity_on_device_seam_is_provable_noop_slice0` asserts discriminator `== false` AND `grow_tree_on_device(..) == Ok(None)` on BOTH `CpuBackend` and `GpuBackend` — green. |
| SC#3 | `assert_on_device_tree_matches_cpu_anchor` oracle scaffold pins tree STRUCTURE to the cpu f64 anchor (tie-aware `default_left`) with leaf values within ~1e-5 f32 envelope — present BEFORE any kernel, never comparing two GPU paths | ✓ VERIFIED (scaffold) — tie-awareness flagged (WR-03, see Human Verification) | Comparator exists (learner_parity.rs:2142), pins 8 structural fields bit-exact via factored `assert_tree_structure_and_leaves`, enforces per-leaf `ROCM_LEAF_VALUE_TOL = 1e-5` envelope, always anchors to `cpu_anchor_tree` (never a 2nd GPU path, def-f8u-01). Runs LIVE GREEN before any kernel via `learner_parity_on_device_oracle_host_fallback_slice0` (host-fallback `unwrap_or_else`). **Caveat:** the `default_left` (bit1) tie guard is currently tautological — the strict shared body asserts `threshold`/`leaf_count`/`internal_count` bit-exact BEFORE the tie loop, so `same_threshold && same_child_counts` is always true; a `default_left`-only flip is accepted unconditionally and the doc-comment claim "A flip on a NON-tie node hard-fails" is false as written. ROADMAP defers the BINDING tie-aware assert to Phase 16. |

**Score:** 3/3 truths verified (SC#3 verified at scaffold grade; tie-awareness sub-claim flagged for human review)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-dataset/src/dataset.rs` | `LeafPartitionLayout` POD (num_data, indices, leaf_begin, leaf_count) | ✓ VERIFIED | `pub struct LeafPartitionLayout` at :88, `#[derive(Debug, Clone)]`, exactly the 4 public fields, no methods, no upward crate dep |
| `crates/lgbm-dataset/src/lib.rs` | re-export | ✓ VERIFIED | `pub use dataset::{..., LeafPartitionLayout}` at :25 |
| `crates/lgbm-compute/Cargo.toml` | `lgbm-model` path dep (acyclic) | ✓ VERIFIED | `lgbm-model = { path = "../lgbm-model" }` at :15 |
| `crates/lgbm-compute/src/lib.rs` | discriminator + seam on `trait Backend` + `GpuBackend<R>` no-op override | ✓ VERIFIED | trait default-false :1239, seam default-`Ok(None)` :1272, GpuBackend override :2207; `grep -c 'use lgbm_treelearner'` = 0 (no cycle); cubecl-0.10 checklist baked into doc-comment |
| `crates/lgbm-treelearner/src/learner.rs` | `cuda_on_device_env()`, `on_device_eligible` field, new()-init, fork | ✓ VERIFIED | helper :443, field :306, inline AND-gate init :488, fork :704 (single assignment site, never recomputed in train_inner) |
| `crates/lgbm-treelearner/src/data_partition.rs` | `DataPartition::from_payload` | ✓ VERIFIED | `pub fn from_payload(p: lgbm_dataset::LeafPartitionLayout) -> Self` at :74 (thin 4-field move) |
| `crates/oracle-harness/tests/learner_parity.rs` | tie-aware comparator + 2 tests | ✓ VERIFIED (tie-awareness inert, WR-03) | `assert_on_device_tree_matches_cpu_anchor` :2142, `assert_tree_structure_and_leaves` :2070, `child_row_counts` :2054, host-fallback oracle :2389, seam no-op :2418 |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| lgbm-compute/lib.rs | lgbm-dataset/dataset.rs | seam return type names `LeafPartitionLayout` | ✓ WIRED | `Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError>` (lib.rs:1278) |
| lgbm-compute/lib.rs | lgbm-model/tree.rs | seam names `lgbm_model::Tree` (acyclic) | ✓ WIRED | dep edge added; workspace builds 0 cycles |
| lgbm-treelearner/learner.rs | lgbm-compute/lib.rs | `on_device_eligible` ANDs discriminator; fork calls `grow_tree_on_device` | ✓ WIRED | learner.rs:488, :705 |
| lgbm-treelearner/learner.rs | data_partition.rs | fork reconstructs via `from_payload` | ✓ WIRED | learner.rs:711 → data_partition.rs:74 |
| oracle-harness | lgbm-compute | oracle calls seam; SC#2 asserts discriminator false | ✓ WIRED | learner_parity.rs:2400, :2427-2444 |
| oracle-harness | lgbm-model | comparator decodes `default_left` = bit1 (mask 2) | ✓ WIRED | `DEFAULT_LEFT_MASK = 2` :2046, used :2160-2171 (though guard is inert, WR-03) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Workspace byte-unchanged, env unset (SC#1) | `LGBM_CUDA_ON_DEVICE= cargo test --workspace` | all suites ok, 0 failed | ✓ PASS |
| Bit-exact golden merge gate | `cargo test -p oracle-harness --test raw_bin_train_parity --features rocm` | `raw_bin_train_matches_cpp_golden` ok; 2 passed | ✓ PASS |
| Kernel parity merge gate | `cargo test -p oracle-harness --test kernel_parity --features rocm` | 21 passed, 0 failed | ✓ PASS |
| Oracle + SC#2 seam tests | `cargo test -p oracle-harness --test learner_parity --features rocm` | 33 passed (incl. both new tests) | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| ODL-01 | 14-01, 14-02 | Additive `grow_tree_on_device` + default-false discriminator, off behind `LGBM_CUDA_ON_DEVICE`, CPU/ROCm/host-CUDA byte-unchanged | ✓ SATISFIED | Seam + discriminator + decide-once fork; merge gate green & byte-unchanged (SC#1, SC#2) |
| ODL-02 | 14-03 | Anchor-pinned oracle asserts on-device tree STRUCTURE bit-exact to cpu f64 anchor (tie-aware default_left), leaf within ~1e-5, never 2 GPU paths | ✓ SATISFIED (scaffold) | Oracle exists, structure-pinned, leaf-enveloped, cpu-anchored, green before any kernel. Tie-aware default_left present but inert (WR-03) — binding assert deferred to Phase 16 per ROADMAP |

No orphaned requirements: REQUIREMENTS.md maps Phase 14 to exactly ODL-01, ODL-02 (both Complete), both claimed by phase plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (phase diffs) | — | No TBD/FIXME/XXX/todo!/unimplemented! in added lines | ℹ️ Info | Clean; all 7 task commits present |
| learner_parity.rs | 2138 | Doc-comment claim "A flip on a NON-tie node hard-fails" is false as written | ⚠️ Warning | WR-03 — see Human Verification; misleading, not a goal blocker |
| learner.rs | 704-714 | On-device fork ignores `capture_snapshots` and sits before V5 validation | ℹ️ Info (deferred) | WR-01/WR-02 — dormant in Slice 0 (fork statically dead); latent Slice-1 risk noted in 14-REVIEW.md; does not affect the zero-behavior-change goal |

### Human Verification Required

#### 1. WR-03 — Tie-aware `default_left` guard is currently inert (scaffold vs. binding)

**Test:** Inspect `assert_on_device_tree_matches_cpu_anchor` (learner_parity.rs:2142). Confirm that `assert_tree_structure_and_leaves` asserts the full `threshold`, `leaf_count`, and `internal_count` vectors bit-exact (:2080, :2083, :2084) BEFORE the `decision_type` tie loop (:2155), making `same_threshold && same_child_counts` (:2172-2174) always true so the tie guard (:2175) can never fire.
**Expected:** Maintainer decides: (a) **accept** scaffold-grade for Slice 0 — ROADMAP Phase 16 explicitly states "tie-aware `default_left` assert lands here," and no Slice-0 kernel can produce a flip, so the inertness has zero current impact (recommended); and (b) **schedule** the WR-03 fix before Phase 16 — make the tie genuinely conditional (relax the shared threshold compare to a documented near-tie tolerance, OR assert `default_left` strictly in the structural body and fall to the tie path only on a proven near-tie input) and correct the false doc-comment claim.
**Why human:** This is a judgment call about whether a present-but-inert oracle sub-component satisfies a scaffold-grade success criterion, plus a scheduling decision for the binding fix. The verifier confirms the tautology is real and that the rest of SC#3 (structural pin, leaf envelope, cpu anchor, present-before-kernel, never-2-GPU-paths) is genuinely satisfied.

### Gaps Summary

No gaps block the phase goal. The seam is additive and provably off (discriminator false + `Ok(None)` no-op on every Slice-0 backend, AND-gate short-circuit → statically dead fork). The full bit-exact merge gate (`raw_bin_train_matches_cpp_golden`, `kernel_parity`, `learner_parity`, lgbm/treelearner/compute suites, workspace) is green and byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset. The oracle scaffold exists, runs green before any kernel, and is correctly pinned to the cpu f64 anchor.

The single open item (WR-03) is the inert tie-aware `default_left` guard within an otherwise-verified oracle. Assessment: **scaffold-grade SC#3 is satisfied for Slice 0** because (1) the structural-pin, leaf-envelope, cpu-anchor, present-before-kernel, and never-2-GPU-paths claims are all genuinely met and green; (2) the ROADMAP explicitly defers the BINDING tie-aware `default_left` assert to Phase 16; and (3) no Slice-0 kernel can produce a `default_left` flip, so the inertness has zero current behavioral impact. It is routed to human verification (not a blocker) because the doc-comment overstates protection and the guard must be made genuinely conditional before Phase 16 activates a real kernel. WR-01 (fork ignores `capture_snapshots`) and WR-02 (fork precedes V5 validation) are dormant Slice-1 latent risks already captured in 14-REVIEW.md and do not affect the Phase 14 zero-behavior-change goal.

---

_Verified: 2026-06-29_
_Verifier: Claude (gsd-verifier)_
