---
phase: 14-foundation-shared-device-primitives-device-structs-rng
plan: 06
subsystem: oracle-harness
tags: [oracle, fixture-parity, primitives, prefix-sum, reduction, argsort, percentile, seam, merge-gate, rocm]

# Dependency graph
requires:
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    plan: "02"
    provides: "committed C++ HIP golden fixtures (fixtures/primitives/{prefix_sum,reduce,argsort,percentile}.txt); weighted percentile deferred (status=deferred_kaggle_nvcc)"
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    plan: "03"
    provides: "full-depth device primitives (block+global prefix-sum, sum/max/min/dot reductions, single-block argsort) + the f64-order policy + argsort tie convention"
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    plan: "05"
    provides: "anchor-pinned skeletons (unweighted/weighted percentile, multi-block/global argsort, items-sort)"
provides:
  - "primitive_parity.rs: fixture-replay parity of every committed C++ primitive golden vs the Rust device primitives — int/perm bit-exact, f64 ULP band, f32 rocm ~1e-6, weighted percentile gated as deferred"
  - "Extended no-op seam + tie-aware anchor-pinned oracle (doc-only) referencing the foundation modules Phase 21 will consume; discriminator FROZEN false (D-09)"
  - "Proven full merge gate: cargo test --workspace green with LGBM_CUDA_ON_DEVICE unset (816 passed / 0 failed / 3 ignored, ~44s)"
affects: [15-minimal-on-device-growth, 19, 21, 22]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fixture-replay parity test mirroring rng_parity.rs: skip-if-absent (green pre-capture), per-record tolerance class, anchored to the C++ golden / cpu f64 fold, never GPU-vs-GPU (def-f8u-01)"
    - "Documented C++ reference-quirk bridges: ShuffleReduce 0-identity fold (rust.max(0)/min(0)) + ShufflePrefixSumExclusive warp-boundary-lane 0 (idx % 32 == 0)"
    - "f64->f32 argsort key-cast: bit-exact on every f32-distinct pair + global monotonicity, with a benign equal-f32-key collision fallback"
    - "Separate #[cfg(feature=rocm)] mod hip f32 leg: ~1e-6 surfaced (no silent pass) + generous relative sanity bound that hard-fails a real kernel bug"

key-files:
  created:
    - crates/oracle-harness/tests/primitive_parity.rs
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "f64 sum/dot/block-prefix-sum cross-validated within a 1e-12 relative band (F64_ORDER_REL_TOL): the C++ warp-tree summation order differs from the cpu anchor's single-owner ascending fold; the committed inputs are same-sign similar-magnitude (well-conditioned) so the band is a few ULP. 442/~768 f64 prefix-sum elements happened to be bit-exact."
  - "f64 max/min asserted BIT-EXACT via the documented C++ 0-identity bridge: ShuffleReduceMax/Min folds a 0 identity lane, so the all-negative goldens record max=0; the Rust selection-only primitive seeds data[0], so the check compares rust.max(0.0)/rust.min(0.0) — then bit-exact."
  - "ShufflePrefixSumExclusive records the within-warp exclusive value (0) at every warp-START lane (idx % 32 == 0, idx > 0) while the rest hold the global running sum; the cpu anchor's clean exclusive scan is asserted bit-exact EVERYWHERE EXCEPT those lanes, where the golden is asserted == 0. Inclusive block scan is a clean full contiguous scan (no quirk)."
  - "Argsort permutations cross-checked bit-exact on the f32-key order (the Rust primitive is f32, the goldens are f64): any index difference must be on an EQUAL-f32-key pair (a benign f64->f32 cast collision; the clustered fixtures produce 1715 such reorders), and the output is asserted globally monotone. This is the contracted 'index-only permutation bit-exact' at f32 precision, anchored to the C++ order."
  - "Unweighted percentile cross-validated at f32 ~1e-6 (PCTL_REL_TOL); the WEIGHTED percentile goldens are SKIPPED (status=deferred_kaggle_nvcc — non-idempotent on the gfx1100 APU, deferred to NVIDIA/Kaggle per 14-02/14-05 D-02), NOT a gate failure."
  - "Seam extension is doc-only (D-09): on_device_growth_supported() stays false, grow_tree_on_device() stays Ok(None); the oracle's anchor discipline (cpu f64, tie-aware default_left, never GPU-vs-GPU) is made explicit and the foundation modules (kernels::primitives/split_info/random) Phase 21 consumes are referenced. No behavioral change to the discriminator, env gate, or LeafPartitionLayout."

requirements-completed: [ODL-01, ODL-02]

# Metrics
duration: ~35 min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 06: Primitive Fixture-Parity + Merge Gate Summary

**Cross-validated every committed C++ device-primitive golden (prefix-sum, reductions, argsort, percentile) against the Rust 14-03/14-05 device primitives in a new `primitive_parity.rs` — int/permutation bit-exact, f64 within a documented ULP band, f32 ~1e-6 on the ROCm leg, weighted percentile gated as deferred — re-established/extended the anchor-pinned tie-aware on-device seam + oracle as a STRICT no-op (D-09), and proved the full workspace merge gate green (816 passed / 0 failed) with `LGBM_CUDA_ON_DEVICE` off and every existing path byte-unchanged.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-06-29
- **Tasks:** 3
- **Files:** 4 (1 created, 3 modified)
- **Full-suite runtime:** `cargo test --workspace` ≈ **44 s wall** (816 passed, 0 failed, 3 ignored) — backfills the 14-VALIDATION TBD latency field.

## Accomplishments

### Task 1 — Extend the no-op seam + anchor-pinned oracle (D-09 / D-10) [`f0817dd`]
- **Doc-only**, no behavioral change. `crates/lgbm-compute/src/lib.rs`: a note by the `grow_tree_on_device` seam pointing at the golden-validated foundation modules (`kernels::primitives` / `split_info` / `random`) the Slice-1 on-device grow loop (Phase 21) will compose; the discriminator stays FROZEN `false`.
- `crates/lgbm-treelearner/src/learner.rs`: a note by `cuda_on_device_env` confirming `LGBM_CUDA_ON_DEVICE` stays an opt-in `"1"` AND-gate that leaves the host path byte-unchanged when unset.
- `crates/oracle-harness/tests/learner_parity.rs`: extended `assert_on_device_tree_matches_cpu_anchor` doc with the explicit anchor discipline (cpu f64 fold, tie-aware `default_left`, NEVER GPU-vs-GPU) + a reference to the new foundation primitives/structs/RNG.
- **Verified:** `on_device_growth_supported() == false` + `grow_tree_on_device() == Ok(None)` (unchanged); both Slice-0 no-op seam tests green under `--features rocm` (`..._seam_is_provable_noop_slice0`, `..._oracle_host_fallback_slice0`).

### Task 2 — Primitive fixture-parity replay vs C++ goldens (D-03 / D-10) [`3c23987`]
- New `crates/oracle-harness/tests/primitive_parity.rs` (≈430 lines) mirroring `rng_parity.rs`: skip-if-absent fixture loader, `key=value` / `;`-list parse helpers, per-record replay over the 14-03/14-05 Rust device primitives.
- **Coverage validated on the cpu f64 anchor:** 16 prefix-sum cases (10 integer bit-exact + 6 f64 ULP-band), 12 f64 reduction cases (sum/dot ULP band, max/min bit-exact via the 0-identity bridge), 13 argsort cases (3 tie-rich) bit-exact on the f32-key order, 9 unweighted percentile cases (f32 ~1e-6). 9 weighted percentile + 12 f32 reduction records correctly deferred/routed.
- **ROCm leg (`--features rocm`, mod hip):** 12 f32 reduction cases checked at ~1e-6 (surfaced, no silent pass) on the real hip GPU — no parity gap or sanity failure.

### Task 3 — Full merge-gate verification (D-11)
- `cargo test --workspace` with `LGBM_CUDA_ON_DEVICE` UNSET → **816 passed, 0 failed, 3 ignored**, ~44 s.
- Named gates confirmed green: `raw_bin_train_matches_cpp_golden` ✓, `learner_parity` (29 CPU; 33 incl. both Slice-0 no-op tests under rocm) ✓, `primitive_parity` (4 CPU; 5 incl. hip leg) ✓, plus the full lgbm / treelearner / compute suites. No source edits in this task.

## Tolerance-Band Decisions (recorded per the plan's output spec)

| Primitive | Class | Constant / bridge |
|-----------|-------|-------------------|
| integer (u32/u64) prefix-sum | BIT-EXACT | exact in f64 (sums < 2^53) |
| f64 block prefix-sum (incl) | f64 ULP band | `F64_ORDER_REL_TOL = 1e-12` (warp-tree vs serial) |
| f64 prefix-sum (excl) | bit-exact + warp-boundary 0 quirk | `idx % 32 == 0` lanes asserted == 0 |
| f64 reduction sum / dot | f64 ULP band | `F64_ORDER_REL_TOL = 1e-12` |
| f64 reduction max / min | BIT-EXACT | C++ 0-identity bridge `rust.max(0)/min(0)` |
| argsort permutation (block+global, asc/desc, tie-rich) | BIT-EXACT on f32-key order | benign equal-f32-key collision fallback + global monotonicity |
| unweighted percentile | f32 ~1e-6 | `PCTL_REL_TOL = 1e-6` |
| weighted percentile | DEFERRED (skip) | `status=deferred_kaggle_nvcc` (NVIDIA) |
| f32 reductions | ROCm ~1e-6 surfaced | `ORACLE_TOL=1e-6` + `SANITY_REL=1e-3` hard-fail |

## Deviations from Plan

None - plan executed exactly as written. No auto-fixes (Rules 1-3) were needed in the source; the only test-authoring adjustments (the exclusive warp-boundary-0 handling and the f32-cast-collision argsort fallback) are documented reference-quirk accommodations discovered while replaying the committed goldens, contained entirely within `primitive_parity.rs` (Task 2's file).

## Reference Quirks Pinned (so they are not lost)

1. **C++ `ShuffleReduceMax/Min` 0-identity fold** — the all-negative-input goldens record `max == 0`; the cross-check bridges via `rust.max(0.0)` / `rust.min(0.0)` then asserts bit-exact.
2. **C++ `ShufflePrefixSumExclusive` warp-boundary 0** — every warp-start lane (`idx % 32 == 0`, `idx > 0`) records the within-warp exclusive `0` while other lanes hold the global running sum; the cpu anchor's clean exclusive scan is bit-exact everywhere else.
3. **f64→f32 argsort key cast** — the clustered f64 fixtures collide heavily in f32 (1715 benign reorders across the 13 cases); the permutation is asserted bit-exact on every f32-distinct pair and globally monotone.

## Known Stubs

None. `primitive_parity.rs` is a complete fixture-replay gate. Weighted percentile is intentionally deferred (NVIDIA/Kaggle capture, 14-02/14-05 D-02), explicitly counted and logged — not a stub in this plan's file.

## Threat Model

- **T-14-06-01 (Tampering — accidental discriminator flip), mitigate:** SATISFIED. Seam is doc-only/FROZEN (`on_device_growth_supported()==false`, `grow_tree_on_device()==Ok(None)`); both Slice-0 no-op tests + the full merge gate guard any flip (Pitfall 6).
- **T-14-06-02 (Spoofing — fixture replay masking divergence), mitigate:** SATISFIED. Skip-if-absent is explicit + logged; int/perm asserted bit-exact (no silent tolerance widening); f64/f32 bands are documented and narrow; the two reference quirks are pinned with positive assertions (the boundary lane MUST be 0), not skipped.
- **T-14-06-SC (package installs), accept:** N/A — no package installs (cubecl 0.10 vendored).

## Issues Encountered

- The first run surfaced the `ShufflePrefixSumExclusive` warp-boundary 0 quirk (golden `0` at idx 32/64 vs the cpu anchor's clean global exclusive scan). Root-caused as a faithful reference artifact (the arrays are identical except the two warp-start lanes) and handled with a documented `idx % 32 == 0` assertion — no source change.

## Next Phase Readiness

- **Phase 15 (minimal on-device growth):** unblocked — every foundation primitive is now golden-validated against the committed C++ goldens (ODL-01 closed); the seam + tie-aware oracle are re-established and ready to receive the first real on-device tree (the oracle pins it to the cpu f64 anchor, never GPU-vs-GPU).
- **Phase 21 (on-device grow loop):** the foundation modules (`kernels::primitives`/`split_info`/`random`) it will compose are referenced from the seam doc and proven by this gate.
- Phase 14 success criteria 1 (numeric anchor), 3 (seam + oracle extended, never GPU-vs-GPU), and 4 (env off by default, byte-unchanged, merge gate green) are satisfied. No blockers.

## Self-Check: PASSED
- Files created/modified exist on disk: `crates/oracle-harness/tests/primitive_parity.rs` (FOUND), `crates/lgbm-compute/src/lib.rs`, `crates/lgbm-treelearner/src/learner.rs`, `crates/oracle-harness/tests/learner_parity.rs` (all FOUND).
- Commits exist: `f0817dd` (Task 1), `3c23987` (Task 2) (both FOUND via `git log`).
- `cargo test -p oracle-harness --test primitive_parity` → 4 passed (CPU); `--features rocm` → 5 passed (incl. hip f32 leg).
- `cargo test -p oracle-harness --test learner_parity` → 29 passed (CPU); `--features rocm` → 33 passed (both Slice-0 no-op tests green).
- `cargo test --workspace` → 816 passed, 0 failed, 3 ignored (LGBM_CUDA_ON_DEVICE unset).

---
*Phase: 14-foundation-shared-device-primitives-device-structs-rng*
*Completed: 2026-06-29*
