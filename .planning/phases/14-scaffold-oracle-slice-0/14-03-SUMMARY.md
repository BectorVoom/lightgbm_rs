---
phase: 14-scaffold-oracle-slice-0
plan: 03
subsystem: oracle-harness
tags: [oracle, tie-aware-comparator, default-left, seam-noop, host-fallback, cpu-anchor, cuda]

# Dependency graph
requires:
  - phase: 14 (plan 01)
    provides: "Backend::on_device_growth_supported() (default false) + grow_tree_on_device() (default Ok(None)) + GpuBackend<R> explicit no-op override"
  - phase: 14 (plan 02)
    provides: "train_inner decide-once fork + DataPartition::from_payload (the seam this oracle exercises end-to-end)"
provides:
  - "assert_on_device_tree_matches_cpu_anchor — tie-aware tree comparator (structure bit-exact + ~1e-5 leaf envelope + per-internal-node default_left tie acceptance)"
  - "assert_tree_structure_and_leaves — the factored shared structural+leaf body (strict assert_gpu_tree_matches_cpu_anchor now delegates)"
  - "child_row_counts — per-internal-node tie predicate (the per-node analog of kernel_parity same_left_count)"
  - "learner_parity_on_device_oracle_host_fallback_slice0 — LIVE D-01 host-fallback oracle (ODL-02/SC#3), GREEN before any kernel"
  - "learner_parity_on_device_seam_is_provable_noop_slice0 — SC#2 seam no-op proof on Cpu + Gpu backends (ODL-01)"
affects: [slice-1 (the real on-device kernel will be pinned by this same tie-aware oracle once it flips the discriminator true)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Factor a shared structural-assert body so a strict and a tie-aware comparator both delegate (no duplicated 8-field block)"
    - "Lift a per-SplitInfo near-tie acceptance (kernel_parity.rs:1597) to a per-NODE index: default_left flip accepted only on threshold + child-row-count equality"
    - "LIVE seam exercise via TEST-ONLY unwrap_or_else(host_grow): Ok(None) ⇒ host stand-in, so the comparator + seam plumbing run GREEN before any kernel"
    - "Pin the on-device tree to the deterministic cpu f64 anchor, NEVER a second GPU f32 path (def-f8u-01)"

key-files:
  created: []
  modified:
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "D-04 honored: only bit1 (default_left) of decision_type is tie-aware; categorical (bit0) + missing_type (bits2-3) stay STRICT via `& !DEFAULT_LEFT_MASK`. At Tree level the kernel's net-gain tie reduces to threshold (exact f64) + child row-count equality (split_gain is predict-irrelevant)."
  - "D-01/D-02 honored: the unwrap_or_else(host_grow) host-fallback lives in the ORACLE TEST only; production never falls back (the learner uses Ok(None) ⇒ fall through). host_grow is the same cpu f64 construction as cpu_anchor_tree, so the Slice-0 stand-in trivially passes."
  - "Placed all new code inside the existing `#[cfg(feature = \"rocm\")] mod hip` so it reuses spine_corpus/cfg/cpu_anchor_tree/ROCM_LEAF_VALUE_TOL and so Task 3 can name GpuBackend (RocmBackend). The new tests run under the rocm merge gate; the cpu-only workspace gate is byte-unchanged (test-only, rocm-gated)."

patterns-established:
  - "Shared-body factoring of parallel comparators: extract the common structural+leaf asserts, let each variant own only the field it compares differently (decision_type strict vs tie-aware)."
  - "Per-node tie predicate child_row_counts(tree, node): internal child (>=0) → internal_count, leaf child (<0, ~leaf) → leaf_count — the Tree-level analog of the SplitInfo left_count tie check."

requirements-completed: [ODL-02, ODL-01]

# Metrics
duration: 18min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 03: scaffold-oracle-slice-0 Summary

**ODL-02/ODL-01 oracle — the tie-aware `assert_on_device_tree_matches_cpu_anchor` (structure bit-exact + ~1e-5 leaf envelope + per-internal-node `default_left` tie acceptance), run LIVE via the D-01 host-fallback before any kernel exists, plus the SC#2 proof that the Plan-01 seam is a provable no-op (`Ok(None)`, discriminator false) on both Cpu and Gpu backends — all pinned to the deterministic cpu f64 anchor, never a second GPU f32 path.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-29
- **Completed:** 2026-06-29
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- **`assert_on_device_tree_matches_cpu_anchor`** — the tie-aware generalization (ODL-02/D-04). Structure stays BIT-EXACT (num_leaves, split_feature, threshold, left_child, right_child, leaf_count, internal_count) and leaves within `ROCM_LEAF_VALUE_TOL` (1e-5), via the factored shared body. `decision_type` is compared strict on every bit EXCEPT `default_left` (bit1): categorical (bit0) + missing_type (bits2-3) must match via `& !DEFAULT_LEFT_MASK`, and a `default_left` flip is accepted ONLY on a genuine f32-vs-f64 near-tie — at Tree level, threshold (exact f64) + child row-count equality (`child_row_counts`). A flip on a NON-tie node hard-fails. The tie branch is dormant in Slice 0 (no kernel flips it) but compiles and is reachable.
- **Shared body factored** — `assert_tree_structure_and_leaves` carries the 8-field + leaf-envelope block; the existing strict `assert_gpu_tree_matches_cpu_anchor` now delegates to it and adds only its strict `decision_type` assert (no duplicated block; the two hip tree tests still pass).
- **LIVE D-01 host-fallback oracle** (`learner_parity_on_device_oracle_host_fallback_slice0`, ODL-02/SC#3) — obtains the tree via `backend.grow_tree_on_device(&g,&h,nl,md)?.map(|(t,_)| t).unwrap_or_else(|| host_grow(..))`; the Slice-0 seam returns `Ok(None)`, so the TEST-ONLY host stand-in supplies it, and the tie-aware comparator pins it to `cpu_anchor_tree` GREEN — proving comparator + seam signature + plumbing before any kernel.
- **SC#2 seam no-op proof** (`learner_parity_on_device_seam_is_provable_noop_slice0`, ODL-01) — on `CpuBackend` AND `GpuBackend` (`RocmBackend`): `on_device_growth_supported() == false` and `grow_tree_on_device(..) == Ok(None)` (the `GpuBackend<R>` explicit override). The default route is provably untouched. No env var set → no `FORCE_ENV_LOCK` needed.
- **Full merge gate green AND byte-unchanged** with `LGBM_CUDA_ON_DEVICE` unset.

## Task Commits

Each task was committed atomically:

1. **Task 1: tie-aware assert_on_device_tree_matches_cpu_anchor + shared body factor** - `8a4df3f` (feat)
2. **Task 2: LIVE D-01 host-fallback oracle test (ODL-02/SC#3)** - `0154274` (feat)
3. **Task 3: SC#2 seam no-op / default-route-untouched test (ODL-01)** - `e098b72` (feat)

## Files Created/Modified
- `crates/oracle-harness/tests/learner_parity.rs` - Added (inside `mod hip`): `DEFAULT_LEFT_MASK` const, `child_row_counts`, the factored `assert_tree_structure_and_leaves`, the refactored strict `assert_gpu_tree_matches_cpu_anchor` (now delegates), the tie-aware `assert_on_device_tree_matches_cpu_anchor`, the `host_grow` stand-in, and the two new `#[test]`s.

## Decisions Made
- **D-04 — only bit1 is tie-aware.** `decision_type & !DEFAULT_LEFT_MASK` is compared strictly; the `default_left` flip is the sole tolerated difference, gated on threshold + child-row-count equality. This is the Tree-level form of the kernel_parity.rs:1597 `same_threshold && same_left_count && net_gain_tie` check — split_gain is predict-irrelevant metadata, so at Tree level the net-gain tie reduces to identical threshold + child counts (the physically-identical winning split).
- **D-01/D-02 — host-fallback is test-only.** `host_grow` (= `cpu_anchor_tree`'s cpu f64 construction) is the stand-in for the not-yet-existent on-device tree and lives ONLY in the oracle test; production never falls back. Because the Slice-0 seam returns `Ok(None)`, `unwrap_or_else` supplies the host tree and the comparator trivially passes (both trees are the same cpu f64 build).
- **Placement in `mod hip`.** All new code sits in the existing `#[cfg(feature = "rocm")] mod hip` to reuse `spine_corpus`/`cfg`/`cpu_anchor_tree`/`ROCM_LEAF_VALUE_TOL` and so the SC#2 test can name `GpuBackend` (`RocmBackend`). The new tests run under the rocm merge gate (`cargo test -p oracle-harness --test learner_parity --features rocm`); the cpu-only workspace gate is unaffected (test-only additions in a rocm-gated module).

## Deviations from Plan

None - plan executed exactly as written. The three tasks landed with the exact comparator semantics, the LIVE host-fallback oracle shape, and the SC#2 dual-backend no-op proof specified. No auto-fixes (Rules 1-3) or architectural changes (Rule 4) were needed. The Task 3 OPTIONAL delegating-wrapper-backend fork exercise was skipped per executor discretion (the direct no-op proof is sufficient for SC#2, as the plan permits).

## Issues Encountered
None. (After Task 1, `assert_on_device_tree_matches_cpu_anchor` / `child_row_counts` / `DEFAULT_LEFT_MASK` were briefly unused until Tasks 2-3 wired them — an expected interim state, no warning failure.)

## Verification Evidence
- `cargo test -p oracle-harness --test learner_parity --features rocm` — **33 passed, 0 failed** (the 31 existing parity tests unregressed + the 2 new tests green).
- New tests green: `hip::learner_parity_on_device_oracle_host_fallback_slice0`, `hip::learner_parity_on_device_seam_is_provable_noop_slice0`.
- Merge gate (rocm): `--test raw_bin_train_parity` + `--test kernel_parity` — **21 + 2 passed, 0 failed**.
- Merge gate (cpu, `LGBM_CUDA_ON_DEVICE` unset): `cargo test --workspace` — all suites `ok`, 0 failed (byte-unchanged; the additions are test-only and rocm-gated, SC#1).
- Acceptance greps: `fn assert_on_device_tree_matches_cpu_anchor` present; the comparator masks bit1 via `& !DEFAULT_LEFT_MASK` and gates the per-node tie on `same_threshold && same_child_counts`; the 8 structural fields stay bit-exact + leaves use `ROCM_LEAF_VALUE_TOL` in the shared `assert_tree_structure_and_leaves`; the oracle test shows `grow_tree_on_device(..)` + `.map(|(t,_)| t).unwrap_or_else(..)`; the SC#2 test asserts `on_device_growth_supported() == false` and `matches!(.., Ok(None))` on both CpuBackend and GpuBackend.

## User Setup Required
None - no external service configuration required. `LGBM_CUDA_ON_DEVICE` stays unset in production.

## Next Phase Readiness
- The tie-aware oracle is ready to pin Slice 1's real on-device kernel: once the kernel lands and `on_device_growth_supported()` flips true, the SAME `assert_on_device_tree_matches_cpu_anchor` (with its dormant default_left tie branch now potentially live) validates the device tree against the cpu f64 anchor — no comparator change needed.
- The host-fallback `unwrap_or_else(host_grow)` in the oracle test will naturally start receiving `Some((tree, payload))` from the seam in Slice 1, exercising the real path; the SC#2 no-op test will need its `Ok(None)`/`false` assertions updated when the discriminator flips (the expected Slice-1 churn).

## Self-Check: PASSED

Modified file present; all three task commits (`8a4df3f`, `0154274`, `e098b72`) exist in git history.

---
*Phase: 14-scaffold-oracle-slice-0*
*Completed: 2026-06-29*
