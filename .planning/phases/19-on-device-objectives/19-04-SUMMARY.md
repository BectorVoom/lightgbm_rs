---
phase: 19-on-device-objectives
plan: 04
subsystem: compute
tags: [cubecl, objective, lambdarank, rank_xendcg, ranking, oracle-harness, cuda-parity]

# Dependency graph
requires:
  - phase: 19-on-device-objectives
    plan: 00
    provides: ungated objective_rank stub + objective_common parity harness + lambdarank_gh goldens
  - phase: 14-foundation-shared-device-primitives
    provides: bitonic_argsort_items_on (per-segment argsort) + draw_next_float_on (bit-identical LCG)
  - phase: 07-ranking-objectives-metrics
    provides: rank.rs Lambdarank/RankXendcg host math + rank_xendcg_objseed5 RNG-replay golden
provides:
  - LambdaRank-NDCG on-device grad/hess (#[cube], shared + >2048 `_Sorted` launchers)
  - RankXENDCG on-device grad/hess (#[cube], shared + `_GlobalMemory` hessian-buffer-aliasing launchers)
  - Per-item ranking RNG composition (draw_next_float_on, seed+q per query, row-major)
  - objective_parity_rank.rs: lambdarank + rank_xendcg parity cells + device-RNG bit-exact replay
affects: [21-on-device-growth-loop]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Block-per-query ranking kernel realized as a single-owner (UNIT_POS==0) per-query f64 fold — the deterministic anchor for a CUDA atomicAdd_block layout (the random.rs determinism mandate)"
    - "f32 grad/hess accumulation with f64 intermediate math in one #[cube] kernel (mixed Array<f32>/Array<f64> signature), matching the host &mut[f32] fold bit-for-bit"
    - "Host-side sigmoid-table replication + device GetSigmoid lookup (u32::cast_from float→idx) so the device matches the rank.rs table-based golden, not a direct-exp re-derive"
    - "CUDA >2048 buffer-aliasing reproduced by binding the hessian output handle to BOTH the rho scratch and the hess arg (same-index read-then-write is race-free); cuda_params_buffer pre-allocated once"
    - "Tie canonicalization on the bitonic descending permutation (ascending-index within equal-score runs) to match the std-stable golden convention"

key-files:
  created:
    - crates/oracle-harness/tests/objective_parity_rank.rs
  modified:
    - crates/lgbm-compute/src/kernels/objective_rank.rs

key-decisions:
  - "Ranking kernels launch single-owner (deterministic f64 anchor); the CUDA NUM_QUERY_PER_BLOCK=10 / atomicAdd_block layout is recorded and documented as the GPU f32 mirror's residual, NOT realized on the anchor (D-05)"
  - "The sigmoid table (1M bins) is replicated host-side and looked up in-kernel — required for compare_within(ORACLE_TOL) vs the table-based golden (a direct-exp re-derive drifts ~1e-5 > tol)"
  - "pow2_int (2^label) is evaluated host-side (labels are known non-negative integers; repeated-multiply, bit-identical to rank.rs phi) to avoid a device integer-bit-op loop"
  - "log2 is computed as ln(x)/ln(2) — cubecl exposes ln but not log2; the <=1-ULP scales grad/hess by ~1e-16, far within ORACLE_TOL"

patterns-established:
  - "Pattern: bitonic tie canonicalization — reorder each equal-score run of the bitonic descending perm to ascending original index, matching the host std-stable sort (no-op for distinct scores)"
  - "Pattern: faithful CUDA output-buffer aliasing on the f64 anchor via a doubly-bound mutable handle with a proven same-index read-then-write access order"

requirements-completed: [ODL-08]

# Metrics
duration: 20min
completed: 2026-07-01
status: complete
---

# Phase 19 Plan 04: On-Device Ranking Objectives Summary

**LambdaRank-NDCG (shared + >2048 `_Sorted`) and RankXENDCG (shared + `_GlobalMemory`) on-device grad/hess as standalone CubeCL kernels — block-per-query single-owner f64 folds composing `bitonic_argsort_items_on` (with std-stable tie canonicalization) and `draw_next_float_on` (bit-identical per-item RNG), anchor-pinned to the `rank.rs` f64 fold, the real `lambdarank_gh` golden, and the `rank_xendcg_objseed5` RNG-replay golden.**

## Performance
- **Duration:** 20 min
- **Tasks:** 2
- **Files created/modified:** 2 (1 created, 1 modified)

## Accomplishments
- **LambdaRank-NDCG** (`lambdarank_body` + `lambdarank_kernel_{shared,sorted}_f64` + `lambdarank_get_gradients{,_sorted}_on`): per-query DESCENDING-score item ranking via `bitonic_argsort_items_on` (segment = query) with a tie canonicalization pass; pairwise λ over `truncation_level` with the replicated 1M-bin sigmoid lookup; f32 accumulation matching the host `&mut[f32]` fold; `norm` rescale. Both the shared and `>2048` `_Sorted` (`MAX_ITEM_GT_1024`) variants are built and asserted bit-identical.
- **RankXENDCG** (`rank_xendcg_body` + `rank_xendcg_kernel_{shared,global}_f64` + `rank_xendcg_get_gradients{,_global}_on`): per-query softmax → three-order cross-entropy-NDCG grad + hessian; `phi = pow2_int(label) - gamma` (host repeated-multiply `2^label`; `gamma` composed from `draw_next_float_on`, seed `+ q` per query, row-major). The `_GlobalMemory` variant faithfully reproduces the CUDA hessian-buffer aliasing (rho stashed in the hessian output buffer + a pre-allocated-once `cuda_params_buffer`); shared and global hessians are asserted identical (the Pitfall 4 guard).
- **Parity** (`objective_parity_rank.rs`): `lambdarank` cell (device shared+`_Sorted` vs the `Lambdarank` f64 anchor AND the real `lambdarank_gh_iter1` golden, `compare_within(ORACLE_TOL)`; shared==`_Sorted`; twice-run determinism; distinct-score cross-check); `rank_xendcg` cell (device shared+global vs the `RankXendcg` f64 anchor; Pitfall 4 shared==global guard; determinism); `rank_xendcg_rng_replay` cell (device `draw_next_float_on` stream `compare_exact_u32` vs the host `Random(seed+q)` stream AND the committed golden draws — never GPU-vs-GPU).

## Task Commits
1. **Task 1: LambdaRank-NDCG grad/hess (shared + >2048 `_Sorted`)** — `dce051d` (feat)
2. **Task 2: RankXENDCG grad/hess (shared + `_GlobalMemory`) + per-item RNG** — `b36975f` (feat)
3. **Clippy fix (approx_constant on the ln(2) literal)** — `1886777` (fix)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/objective_rank.rs` — filled the ODL-08 stub: both objectives' `#[cube]` grad/hess bodies + kernels + launchers, sigmoid-table replication, per-item RNG composition, tie canonicalization, buffer aliasing.
- `crates/oracle-harness/tests/objective_parity_rank.rs` — `lambdarank`, `rank_xendcg`, `rank_xendcg_rng_replay` parity cells (capture-gated skip-pass via `objective_common`).

## Decisions Made
- Single-owner deterministic f64 anchor for both ranking kernels (the `atomicAdd_block` / `NUM_QUERY_PER_BLOCK=10` CUDA layout is documented as the GPU f32 mirror's residual, D-05) — matches the proven `random.rs` mandate.
- The device replicates the host's 1M-bin sigmoid **table lookup** (not a direct-`exp` re-derive): the `lambdarank_gh` golden was produced by the `rank.rs` port, which uses the quantized table; a direct-`exp` device drifts ~1e-5 (> `ORACLE_TOL`).
- `pow2_int` (`2^label`) evaluated host-side (bit-identical repeated-multiply) to sidestep a device integer-bit-op loop; `log2` computed as `ln(x)/ln(2)` (cubecl has no `log2`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Bitonic ties diverge from the std-stable golden convention → tie canonicalization added**
- **Found during:** Task 1
- **Issue:** `bitonic_argsort_items_on`'s comparator has no index tie-break (a size-3 tie sorts to `[2,1,0]`, not `[0,1,2]`), while the host anchor + golden use a std STABLE sort (ascending-index ties). The iter-1 golden feeds all-zero scores (a full tie), and real trained scores tie constantly (rows sharing a leaf value), so the raw bitonic order produced grad diffs up to ~0.012 (>> `ORACLE_TOL`). An initial sub-ULP key-perturbation fix worked only for zero scores (the perturbation is absorbed by f32 rounding for nonzero equal scores).
- **Fix:** After the bitonic sort, canonicalize each equal-score run (bitonic places equal scores contiguously) to ascending original index — matching the std-stable convention. No-op for distinct scores; the kernel reads the original f64 scores.
- **Files modified:** crates/lgbm-compute/src/kernels/objective_rank.rs
- **Verification:** `lambdarank` cell passes on iter-1 (zeros) AND the distinct trained-score cross-check.
- **Committed in:** `dce051d`

**2. [Rule 1 - Bug] clippy::approx_constant deny on the hardcoded `LN_2` literal**
- **Found during:** post-Task-2 verification (`cargo clippy -p lgbm-compute`)
- **Issue:** the `ln(2)` denominator was a hardcoded `0.693…` literal, which trips clippy's deny-by-default `approx_constant`.
- **Fix:** compute `ln(2)` via the device `2.0f64.ln()` (same value, no literal); rank cells remain green.
- **Files modified:** crates/lgbm-compute/src/kernels/objective_rank.rs
- **Committed in:** `1886777`

---
**Total deviations:** 2 auto-fixed (2 bugs). No architectural changes; no cross-plan file contention (this plan owns both touched files).

## Known Stubs / Deferred Surfaces
- **Queries `> BITONIC_SORT_NUM_ELEMENTS`**: the `_Sorted` (`MAX_ITEM_GT_1024`) launcher is built and validated on the §5.4 corpus (proving the code path), but the underlying multi-block per-segment sort for genuinely huge queries remains the deferred `primitives.rs` hardening (per its own SKELETON note); the §5.4 corpora are zero-init or distinct-trained with small queries.
- **Exact tie-ordering for arbitrarily-large equal scores**: the tie canonicalization uses exact f64-score equality of the original scores; f64-distinct/f32-equal near-ties keep bitonic's order (delta < f32 ULP → within `ORACLE_TOL`). Documented in-file.
- **kMinScore / dropped-doc and position-bias/weights**: out of scope for the §5.4 spine (mirrors the host `rank.rs`'s own documented out-of-scope surface); the standard training path never sets a `kMinScore` input.

## Threat Model Compliance
- **T-19-04-01 (Tampering, RankXENDCG >2048 hessian aliasing):** mitigated — the aliasing is documented with a proven same-index read-then-write order; the `cuda_params_buffer` is pre-allocated once; the shared==global hessian-match assertion is the guard.
- **T-19-04-02 (Tampering, per-query index / empty query):** mitigated — `validate_query_boundaries` at the V5 boundary; `cnt <= 1` singleton skip; in-kernel `i < cnt` guards.
- **T-19-04-03 (Spoofing/Repudiation, RNG substitution):** mitigated — composes only the deterministic `draw_next_float_on` LCG; the RNG-replay cell pins the stream bit-exact vs the host `Random` and the committed golden (never a CSPRNG).

## Verification
- `cargo test -p oracle-harness --test objective_parity_rank` — 3/3 cells pass (`lambdarank`, `rank_xendcg`, `rank_xendcg_rng_replay`).
- `cargo test --workspace --lib --tests -j 4` — 68 test binaries pass, zero failures (`LGBM_CUDA_ON_DEVICE` unset).
- `cargo clippy -p lgbm-compute` — no errors (the 3 remaining `manual_slice_size_calculation` warnings in the file are the same idiom the sibling objective kernels use).

## Next Phase Readiness
- Both ranking objectives are available on-device for the Phase-21 growth loop (`lgbm_compute::kernels::objective_rank`), OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-06). All four Wave-2 family plans (19-01..04) are now complete; Phase 19 (ODL-05..08) is fully filled.

## Self-Check: PASSED
- Both files exist on disk (`objective_rank.rs` 812 lines, `objective_parity_rank.rs` 274 lines).
- All three commits (`dce051d`, `b36975f`, `1886777`) present in git history.
- Acceptance markers confirmed: `bitonic_argsort_items_on` (7×) and `draw_next_float_on` (5×) present; `lambdarank_kernel_{shared,sorted}_f64` + `rank_xendcg_kernel_{shared,global}_f64` all built; no `percentile_device` reference.

---
*Phase: 19-on-device-objectives*
*Completed: 2026-07-01*
