---
phase: 14-foundation-shared-device-primitives-device-structs-rng
plan: 05
subsystem: compute
tags: [cubecl, primitives, percentile, bitonic-argsort, multi-block, items-sort, skeleton, anchor-pinned, gpu]

# Dependency graph
requires:
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    plan: "03"
    provides: "single-block index-only bitonic argsort (comparator/tie convention) + block/global inclusive prefix-sum — the full-depth primitives these skeletons compose"
provides:
  - "percentile_unweighted_f32_on / percentile_weighted_f32_on — anchor-pinned percentile skeletons mirroring C++ PercentileDevice (both branches), composing the 14-03 argsort + inclusive prefix-sum, f64 interp (cold path)"
  - "bitonic_argsort_global_on — multi-block/global index-only argsort skeleton (CHECKED ::launch) reusing the shared bitonic_argsort_body network, permutation bit-exact vs serial on >1024 inputs"
  - "bitonic_argsort_items_on — per-segment ranking items-sort skeleton (BitonicArgSortItems analog), V5 segment validation, local 0-based per-segment permutations"
  - "bitonic_argsort_body extracted as the single-source comparator network shared by the single-block (launch_unchecked) and global (checked ::launch) launchers"
affects: [19, 22]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Anchor-pinned skeleton (D-02): correct + serial-f64-anchored now, depth/perf hardening deferred to the named first consumer; composes the full-depth 14-03 primitives instead of reinventing"
    - "Single-source #[cube] comparator network body (bitonic_argsort_body) shared by an unchecked single-block launcher and a checked-::launch global launcher"
    - "CHECKED ::launch for skeleton paths (perf irrelevant, RESEARCH Pattern 6 / threat T-14-05-01); V5 input/segment validation to a typed ComputeError before any launch"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/primitives.rs
    - crates/lgbm-compute/tests/primitives_self.rs

key-decisions:
  - "Each skeleton COMPOSES the 14-03 full-depth primitives rather than launching its own bespoke math: percentile = argsort + (weighted) inclusive prefix-sum + host-side threshold/interp; items-sort = per-segment bitonic_argsort_on; global argsort = the shared bitonic_argsort_body network with the 1024 cap lifted. This is exactly D-02's 'compose, do not reinvent'."
  - "Skeleton launches are CHECKED (::launch) per RESEARCH Pattern 6 / threat T-14-05-01 (perf irrelevant on these cold paths); the global argsort kernel uses #[cube(launch)] and percentile/items reuse the already-SAFETY-proven full-depth primitives, each gated by V5 input validation (empty/length/segment) returning a typed ComputeError before any device alloc."
  - "Percentile interpolation + weight prefix-sum are f64 (acceptable on this cold Phase-19 consumer path, RESEARCH Pitfall 4) — NOT the f32/int+u64 discipline the grow-loop hot primitives require."
  - "The weighted-percentile edge branch mirrors the C++ values[pos] (raw, un-permuted) read VERBATIM — a reference quirk preserved for faithfulness; the Phase-19 consumer revisits it when it captures the dedicated weighted fixture (deferred to NVIDIA per 14-02)."
  - "No dedicated C++ items-sort golden captured this phase (intentional, per plan): the comparator/tie convention is already C++-locked by the 14-02 tie-rich single-block argsort fixture (asserted bit-exact in 14-03); the per-segment skeleton merely reuses that locked convention segment-by-segment, so it is anchored to the serial per-segment Rust reference here. Capturing an items-sort fixture is the Phase-19 ranking hardening task."

requirements-completed: [ODL-01]

# Metrics
duration: ~5 min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 05: Anchor-Pinned Primitive Skeletons Summary

**Authored the three remaining ODL-01 primitives — weighted/unweighted percentile, multi-block/global index-only argsort, and the per-segment ranking items-sort — as anchor-pinned SKELETONS (D-02) that COMPOSE the 14-03 full-depth argsort + prefix-sum, each correct and serial-f64-anchor-tested now with depth-hardening explicitly deferred to its named Phase-19/22 consumer.**

## Performance

- **Duration:** ~5 min
- **Completed:** 2026-06-29
- **Tasks:** 3
- **Files modified:** 2 (both extended; none created)
- **Tests:** 9 new serial-anchored self-tests (22 total in `primitives_self.rs`), all green on the cpu f64 anchor

## Accomplishments

### Task 1 — Weighted + unweighted percentile skeleton
- `percentile_unweighted_f32_on` mirrors C++ `PercentileDevice<…,USE_WEIGHT=false>`: compose the 14-03 ascending argsort, then `float_pos = (1-alpha)*len`, `pos = trunc(float_pos)`, linear interp between `sorted[pos-1]`/`sorted[pos]` (clamped to the extremes), interp in f64 (cold path, Pitfall 4), cast back to f32.
- `percentile_weighted_f32_on` mirrors the `USE_WEIGHT=true` branch: argsort, inclusive prefix-sum of the weights IN SORTED ORDER (the 14-03 `prefix_sum_inclusive_f64_on` device primitive), `threshold = total_weight*(1-alpha)`, first-crossing position, interp. The `values[pos]` edge read is mirrored verbatim from C++.
- V5: empty input rejected (`LengthMismatch`); weighted rejects `weights.len() != len`.
- Anchored bit-exact vs (a) concrete hand-computed values (median 2.5; weighted edge 4.0, interior 2.0) and (b) a serial f64 reference across `alpha ∈ {0,0.25,0.5,0.75,0.9,1}` on non-uniform inputs.

### Task 2 — Multi-block / global argsort skeleton
- Extracted the 14-03 bitonic network into a single-source `#[cube] bitonic_argsort_body`; the existing single-block launcher (`launch_unchecked`) and the new global launcher both delegate to it, so the comparator/tie convention exists exactly once.
- `bitonic_argsort_global_on` lifts the single-block 1024 cap to `MAX_GLOBAL_ARGSORT_ELEMENTS` via `bitonic_argsort_global_kernel` (`#[cube(launch)]`, CHECKED). Index-only (`keys` read-only; `keys_after` proves it). No cross-cube barrier assumption (single-owner serial network subsumes the C++ per-block-sort + cross-block-merge phases). Documented as a skeleton; the genuine multi-cube per-block+merge decomposition (and non-pow2 offset handling) is the Phase-19/22 hardening task.
- Permutation bit-exact vs the serial reference on a 1500-element distinct-descending input and an 1100-element tie-rich (3 distinct keys) input — both spanning more than one single-block tile.

### Task 3 — Per-segment ranking items-sort skeleton
- `bitonic_argsort_items_on(keys, segment_boundaries: &[i32], ascending)` sorts indices WITHIN each segment by key (index-only), composing the 14-03 single-block argsort segment-by-segment. Returns a flat `Vec<i32>` of length `keys.len()`; each segment's slice is that segment's LOCAL 0-based permutation, matching the C++ `BitonicArgSortItemsGlobalKernel` (whose `BitonicArgSortDevice` initializes per-segment-local indices).
- `validate_segments` enforces the V5 boundary BEFORE any launch: ≥2 entries, starts at 0, ends at `keys.len()`, non-decreasing, each segment ≤ the single-block cap — else a typed `ComputeError`.
- Bit-exact vs a serial per-segment reference on a 3-segment differing-length input and a tie-rich-segment input; malformed-boundary inputs rejected.

## Acceptance Criteria

| Task | Criterion | Result |
|------|-----------|--------|
| 1 | both wtd+unwtd variants correct on small inputs | PASS (`percentile` filter: 3/3) |
| 1 | documented skeleton + Phase-19 hardening owner named | PASS (doc comments) |
| 1 | anchored to serial f64 ref (no GPU-vs-GPU) | PASS |
| 2 | sorts > one single-block tile, index-only, permutation bit-exact | PASS (`argsort_global` filter: 3/3, 1500 & 1100 elems) |
| 2 | no cross-cube barrier assumption; skeleton + hardening owner named | PASS |
| 3 | sorts within each segment independently, index-only, bit-exact | PASS (`items_sort` filter: 3/3) |
| 3 | documented skeleton + Phase-19 ranking consumer named | PASS |

## Verification

- `cargo test -p lgbm-compute --test primitives_self` → 22 passed (13 from 14-03 + 9 new).
- Per-task filters: `percentile` 3/3, `argsort_global` 3/3, `items_sort` 3/3.
- `cargo build -p lgbm-compute` → clean.
- `cargo test -p lgbm-compute --lib` → 52 passed, 1 ignored, 0 failed (no regression).
- `cargo check -p lgbm-compute --features rocm` → clean (the skeletons are not rocm-gated; the rocm plane variants still compile).
- `cargo clippy -p lgbm-compute` → the only 2 primitives.rs warnings are at lines 282/367 (pre-existing 14-03 prefix-sum launch code, not the new skeletons); no new warnings introduced.

## Threat Model

- **T-14-05-01 (Tampering/DoS — OOB on skeleton launches), mitigate:** SATISFIED. The global argsort skeleton uses CHECKED `::launch`; percentile/items reuse the already-SAFETY-proven full-depth primitives. Every host entry V5-validates length/segment boundaries to a typed `ComputeError` before any device alloc/launch (`validate_segments`, percentile empty/length guards, `MAX_GLOBAL_ARGSORT_ELEMENTS` cap).
- **T-14-05-SC (package installs), accept:** N/A — no package installs (cubecl 0.10 vendored).

## Deviations from Plan

**[Process] Three tasks committed as one cohesive commit rather than three per-task commits.**
- The three skeletons were authored as one contiguous, interleaved addition across the same two shared files (`primitives.rs` + `primitives_self.rs`), with Task 2's shared-body refactor underpinning Task 2's own kernel. The additions are a single non-splittable diff hunk in `primitives.rs`, so hunk-level per-task staging was not cleanly achievable (interactive `git add -i` unsupported in this environment). All three tasks are present, individually tested (per-task filters green), and the commit body enumerates each task. No functional impact.

Otherwise: plan executed exactly as written. No auto-fixes (Rules 1–3) were needed; the composition of the already-correct 14-03 primitives worked first time.

## Known Stubs

None. The three deliverables are correct, anchor-tested SKELETONS (D-02) — not stubs: each produces correct output on its supported input regime. What is deferred is DEPTH (the `>1024` multi-cube GPU decomposition for argsort, the f32/int hardening + dedicated C++ fixtures for percentile/items), explicitly owned by the named Phase-19/22 consumers, not absent functionality.

## Issues Encountered

None.

## Next Phase Readiness

- **Phase 19 (objectives/ranking):** unblocked — `percentile_*` (regression/Huber/Fair init scores) and `bitonic_argsort_items_on` (lambdarank per-query sort) exist with correct signatures + behavior; Phase 19 extends (f32/int hardening, dedicated weighted-percentile + items-sort C++ fixtures, descending ranking default) rather than invents.
- **Phase 19/22 (on-device sort at scale):** `bitonic_argsort_global_on` exists; the genuine multi-cube per-block-sort + cross-block-merge kernels (and non-pow2 offset handling) are its hardening task.
- ODL-01's "all primitives exist" is now fully satisfied across 14-03 (full-depth) + 14-05 (skeletons).
- No blockers.

## Self-Check: PASSED
- Files modified exist on disk: `crates/lgbm-compute/src/kernels/primitives.rs`, `crates/lgbm-compute/tests/primitives_self.rs` (both FOUND).
- Commit exists: `8432331` (FOUND via `git log --grep="14-05"`).
- `cargo test -p lgbm-compute --test primitives_self` → 22 passed; per-task filters (percentile/argsort_global/items_sort) all green.

---
*Phase: 14-foundation-shared-device-primitives-device-structs-rng*
*Completed: 2026-06-29*
