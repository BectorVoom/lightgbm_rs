# T-11 (SPEC-DRGL-11) — P100 perf A/B verdict of `LGBM_GROW_DEFER_SYNC`

Recorded 2026-07-15. Per the spec, the result is recorded WHATEVER the verdict; an
inconclusive/regressing result is a valid, complete outcome and does NOT block T-11 —
it blocks a later default-flip (which is out of scope). **No default flip is made.**

## Setup
- Kaggle kernel `yensen2/lgb-rs-t11-defer-ab` (patch of T-01→T-06 vs origin/main, applied
  clean, built `--features cuda`). GPU: **Tesla P100-PCIE-16GB**.
- Corpus 500k × 50, `n_estimators=100`, `num_leaves=31`, on-device CUDA
  (`LGBM_CUDA_ON_DEVICE=1`, `LGBM_PHASE_PROF=1`, `LGBM_AUTOTUNE=0`).
- Arms: `base` (`LGBM_GROW_DEFER_SYNC` unset) vs `defer` (`=1`). 4 rounds, order-alternated
  (A/B/B/A…), warm-median-of-3 (cold round dropped).

## Result
| arm | warm-median wall (s) | deferred_read_fused | num_trees |
|-----|----------------------|---------------------|-----------|
| base  | **6.94** | 0    | 100 |
| defer | **8.30** | 3000 | 100 |

- **speedup (defer vs base) = 0.836 ⇒ the deferral is ~1.20× SLOWER on P100.**
- **preds parity: bit_identical = TRUE, max_abs = 0.0** — byte-identical predictions.
- counts_ok = TRUE (tripwire fires on `defer`, absent on `base`; 3000 = 30 splits × 100 trees).
- tree_count_ok = TRUE.

## Interpretation
1. **Correctness confirmed on P100 (and resolves the flagged CUDA risk):** even though the
   deferred arm's LEFT/RIGHT scans use the LEGACY devcount kernel on CUDA (parprefix defaults
   OFF there) while `base` uses the serial-STAGED kernel, the predictions are `max_abs = 0.0`
   — serial-staged == legacy bit-for-bit on this workload. The deferral is byte-identical
   end-to-end (matches the gfx1151 + P100 unit-test results).
2. **The sync-halving does NOT translate to wall-time — it REGRESSES.** T-06 proved the per-
   grow blocking-sync count drops `2L → L+2`, but the deferral's REQUIRED compute-path changes
   cost more than the saved syncs recover on P100:
   - build-LEFT (which=0) can build the LARGER child histogram (the eager arm builds the
     SMALLER by the subtraction-trick — cheaper).
   - two SEPARATE LEFT/RIGHT scan launches instead of the eager arm's ONE co-packed sibling
     scan (2× scan launches).
   - on CUDA the deferred scan is the LEGACY kernel, not the serial-STAGED one the eager arm
     uses (legacy is the slower geometry).
   Consistent with the local gfx1151 free-run pre-check (flag-OFF 19.75 ms/tree, flag-ON
   20.83 ms/tree — ~5% slower on the APU) and the memory's warning that the sync-tail's P100
   transfer was unproven.

## Verdict
`LGBM_GROW_DEFER_SYNC` stays **DEFAULT OFF** (unchanged). The P100 verdict does NOT justify a
default flip — it regresses. The deferral is correct + the sync-count contract is met, but it
is not a perf win as built.

## If revisited (out of scope for this phase)
The deferral's overhead is the compute-path changes, not the fusion itself. A future attempt
to make it a WIN would need to remove those: (a) co-pack the LEFT/RIGHT deferred scans (use
`subtract_scan_resident_siblings_into_frontier_devcount` — already built — instead of two
separate scans), and (b) make the deferred CUDA scan use serial-staged/parprefix rather than
the legacy kernel (a `which`-aware staged/parprefix devcount for CUDA). Only then re-run the
P100 A/B. Until a measured P100 win exists, the flag ships OFF.
