---
status: complete
phase: quick-260614-p0n
plan: 01
subsystem: treelearner
tags: [performance, cpu, histogram, build, bit-exact, profiling, low-rows]

requires:
  - phase: spike-002-lowrow-phase-ab
    provides: "diagnosis — low-row gap is ~entirely histogram build (5.2x vs C++); 22% of in-main instructions are malloc/memset/memcpy inside the build"
provides:
  - "accumulate_histogram_into — a tested fold-in-place native f64 histogram primitive (folds into a caller-pre-zeroed out sub-slice; bit-identical to the allocating native path)"
  - "learner num_bins descriptor-cache — reused per-leaf Vec<u32> across builds"
  - "MEASURED VERDICT: the fold-in-place call-site rewrite is net-negative (no low-row win, ~9% large-row regression) and was reverted; the alloc churn is NOT the low-row wall-clock bottleneck at 2k rows"
affects: [R3-columnar-storage, histogram-build-perf, treelearner]

tech-stack:
  added: []
  patterns:
    - "Fold-in-place accumulator into a caller-owned pre-zeroed sub-slice (primitive kept, call-site reverted)"
    - "Measurement-gated perf change: interleaved A/B before shipping; revert on a material large-size regression"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-treelearner/src/learner.rs

key-decisions:
  - "Fold-in-place at the build_leaf_histograms_raw call site is REVERTED: it gave no measurable low-row build speedup and regressed the 200k-row train ~9% (scattered RMW across the 32KB multi-feature out buffer vs the original tight ~1KB per-feature hot scratch + streaming copy)."
  - "construct_histograms_cpu_native restored to its exact inline fold loop (NOT delegated): a shared &mut[f64] helper / a second validation pass measurably regressed the large native path; the two loop bodies are pinned byte-identical by a unit test instead."
  - "The learner num_bins descriptor-cache is KEPT: large- and small-neutral, removes one per-leaf Vec<u32> allocation with zero downside."
  - "accumulate_histogram_into is KEPT as a tested primitive for a future columnar R3 rewrite that can fold into a per-feature hot scratch (not the big out buffer), getting the alloc saving WITHOUT the large-row cache penalty."

patterns-established:
  - "Pattern: validate the spike's instruction-count hypothesis against wall-clock A/B before committing to it — 22% malloc/memset/memcpy by instruction count did NOT translate into a low-row wall-time win."

requirements-completed: []  # R3-HIST-BUILD-PERF NOT satisfied — the targeted low-row build speedup was not achieved; see verdict.

duration: ~75min
completed: 2026-06-14
---

# Phase quick-260614-p0n: R3 histogram-build perf fix Summary

**Measured the spike-002 low-row histogram-build hypothesis end-to-end: the fold-in-place rewrite delivered NO low-row build speedup (within noise) and a reproducible ~9% large-row regression, so it was reverted; the bit-exact-safe alloc-reduction primitive + the learner num_bins cache were kept as perf-neutral cleanup.**

## Performance

- **Duration:** ~75 min
- **Tasks:** 3 of 3 (Task 1 + Task 2 code, Task 3 gate + measurement)
- **Files modified:** 2 (crates/lgbm-compute/src/kernels/histogram.rs, crates/lgbm-treelearner/src/learner.rs)

## Accomplishments

- **`accumulate_histogram_into` (histogram.rs):** a fold-in-place native f64 accumulator that folds the same ascending rows / `bin<<1` cells / f32→f64 accumulation directly into a caller-pre-zeroed `out` sub-slice (no per-feature Vec, no copy), with the same V5 validation + an extra typed `LengthMismatch` on a short `out`. Proven bit-identical (`f64::to_bits` cell-by-cell on 2 shapes) to `construct_histograms_cpu_native` by a new unit test, plus a sub-slice-placement test and a short-`out` rejection test.
- **Learner `num_bins` descriptor-cache (learner.rs):** `build_leaf_histogram_into` no longer allocates a fresh `Vec<u32>` of per-feature bin counts per call — it fills a `RefCell<Vec<u32>>` lazily on first build (surviving `with_features`), re-validates length/content, and re-borrows it thereafter. Content + order are byte-identical, so the fold inputs (and the bit-exact f64 tree) are unchanged.
- **Full bit-exact merge gate GREEN** on the final keep-set (see below).

## Bit-Exact Merge Gate (Task 3) — PASS

All four gate commands green on the final HEAD (7ca89b1):

| Gate command | Result |
|---|---|
| `cargo test -p lgbm-compute` | 21 passed / 0 failed (incl. 3 new accumulate-into unit tests) |
| `cargo test -p lgbm-treelearner` | 65 passed / 0 failed |
| `cargo test -p oracle-harness --test learner_parity` | 29 passed / 0 failed (spine_real, growth_path_subtract, mfb_pos_real, subtract — bit-exact) |
| `cargo test -p oracle-harness` (full corpora) | every suite 0 failed / 0 ignored (no new failures vs HEAD; no pre-existing DEF-07-02 cells regressed) |

The f64 fold ORDER is frozen and byte-unchanged throughout. clippy clean on every edited file; `LightGBM/` never git-added.

## Measurement — before → after (the key finding)

Method: phase profiler (`LGBM_PHASE_PROF=1`), warm, **interleaved A/B** (baseline 299c6ec vs final HEAD), median of 3 interleaved rounds. Build µs/iter is the `[phase_prof] ... build=` accumulator over reps×iters.

### Small (2000 rows × 12 feat × 32 bins, 100 iters × 9 reps = 900 it)

| metric | BASELINE | FINAL (keep-set) | verdict |
|---|---|---|---|
| build (ms / 900 it) | ~214–217 (≈239 µs/iter) | ~215–223 (≈240 µs/iter) | **within noise — no speedup** |
| train_median | 41.6–42.6 ms | 41.3–41.8 ms | within noise |

### Large (200000 rows × 32 feat × 64 bins, 50 iters × 3 reps), train_median

| variant | large train_median | vs baseline |
|---|---|---|
| BASELINE (299c6ec) | **4.43 s** (very stable) | — |
| **fold-in-place at call site** (rejected) | **~4.89–4.95 s** | **~+9% REGRESSION** (3 interleaved rounds) |
| FINAL keep-set (HEAD 7ca89b1) | 4.46–4.52 s | within noise (regression gone) |

The decisive datum: an isolated interleaved A/B (corrected histogram.rs, baseline learner, fold-in-place on/off) showed the **fold-in-place alone** moved large from ~4.52 s → ~4.95 s across 3 rounds. Reverting it returns large to baseline.

## Deviations from Plan

### [Rule 1 — Perf regression] Reverted the Task-1 fold-in-place at the call site

- **Found during:** Task 3 measurement (the large-size sanity check the plan mandates).
- **Issue:** The plan's hypothesis (spike-002) was that killing the per-feature `vec![0.0; 2*num_bin]` alloc + memset + memcpy would cut the low-row build. Measurement disproved the payoff: (a) at 2k rows the fold-in-place gave NO measurable build-µs/iter or train_median improvement (within ~6% run-to-run noise) — the spike's "22% malloc/memset/memcpy" is an *instruction-count* share, and at this scale the allocator fast-path + 512-byte per-feature copies are already cheap in wall-time; (b) at 200k rows it *regressed* train ~9% (reproducible, 3 interleaved rounds), because accumulating directly into the 32KB multi-feature `out` buffer scatters the read-modify-write across cache, vs the original tight ~1KB per-feature hot scratch + streaming `copy_from_slice`.
- **Fix:** Reverted `build_leaf_histograms_raw` to the baseline per-feature `construct_histograms` + `copy_from_slice`. The success criterion "no material large-size regression" is a hard gate; the change is rejected on perf grounds (it is still bit-exact — this is NOT a fold-order rejection).
- **Files:** crates/lgbm-compute/src/lib.rs (reverted to baseline).
- **Commit:** 7ca89b1.

### [Rule 1 — Codegen regression] Restored `construct_histograms_cpu_native` to its exact inline loop

- **Found during:** Task 3 large-size isolation.
- **Issue:** The plan asked the native path to delegate to `accumulate_histogram_into` so the two "share ONE fold body." That delegation (a) ran a second O(n) bin-range validation per call, and (b) even after de-duplicating the validation, routing the hot allocate-then-fold path through a shared `&mut [f64]` helper measurably regressed the 200k-row native build ~5% (codegen/inlining difference).
- **Fix:** `construct_histograms_cpu_native` keeps its original inline fold loop (single validation, no delegation). The "can never drift" intent is satisfied more strongly by the `accumulate_into_is_bit_identical_to_native` unit test (`f64::to_bits` cell-by-cell on multiple shapes) than by textual sharing.
- **Files:** crates/lgbm-compute/src/kernels/histogram.rs.
- **Commit:** 7ca89b1.

### [Plan-authorized deferral] feature_bins lifetime reuse NOT attempted

- The plan permitted caching only `num_bins` if the `feature_bins: Vec<&[u32]>` lifetime fought the borrow checker (its `&[u32]` borrow is tied to the `features` parameter lifetime; storing it behind `&self` would infect the struct with a lifetime param — an architectural change the plan defers). `feature_bins` stays a per-call local. The columnar storage rewrite (the structural R3 lever) is likewise deferred.

## What was KEPT (perf-neutral, bit-exact)

- `accumulate_histogram_into` + its 3 unit tests (a reusable, tested primitive; a future columnar R3 rewrite can fold into a per-feature hot scratch — not the big `out` buffer — to get the alloc saving WITHOUT the large-row cache penalty observed here).
- The learner `num_bins` descriptor-cache (commit 98cd934): large- and small-neutral, removes one per-leaf `Vec<u32>` allocation with no downside.

## Requirement status

**R3-HIST-BUILD-PERF: NOT satisfied.** The targeted low-row histogram-build speedup was not achieved — measurement showed the alloc-churn removal does not move low-row wall-clock at 2k rows, and the fold-in-place variant regressed large. The real R3 lever per spike-002 ("Columnar pre-binned storage + subtraction reuse — the structural fix") remains open; this plan delivered the bit-exact-safe primitive and the measurement that re-scopes it, not the win.

## Commits

- c4fb4be — perf: fold-in-place native f64 histogram accumulator (Task 1; call-site fold-in-place later reverted)
- 98cd934 — perf: reuse per-leaf num_bin descriptor scratch in learner build (Task 2; KEPT)
- 7ca89b1 — perf: revert fold-in-place at call site (large-row regression); restore inline native loop; keep primitive + num_bins cache

## Self-Check: PASSED

- FOUND: crates/lgbm-compute/src/kernels/histogram.rs (accumulate_histogram_into present)
- FOUND: crates/lgbm-treelearner/src/learner.rs (build_num_bins cache present)
- FOUND: .planning/quick/260614-p0n-r3-histogram-build-perf-fix-columnar-pre/260614-p0n-SUMMARY.md
- FOUND: commits c4fb4be, 98cd934, 7ca89b1
