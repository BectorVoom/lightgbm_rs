---
status: complete
phase: quick-260622-ia0
plan: 01
subsystem: treelearner
tags: [rayon, data-partition, prefix-sum, cpu-perf, bit-exact, tall-narrow]

requires:
  - phase: spike-005-feature-parallel-build
    provides: the LGBM_PAR_THRESHOLD/par_build_threshold rayon-gate idiom + the bit-exact parallel-equals-serial test pattern (build_histograms_parallel_equals_serial)
provides:
  - rayon-parallel, deterministic static-chunk + exclusive prefix-sum, bit-exact DataPartition::split with a leaf-row gate (LGBM_PAR_SPLIT_THRESHOLD, default 16384)
  - split_parallel_equals_serial byte-identical merge-gate test
affects: [cpu-train-perf, tall-narrow-shape, tree-learner]

tech-stack:
  added: [rayon (already in-tree via lgbm-compute; newly added to lgbm-treelearner)]
  patterns:
    - "C++ ParallelPartitionRunner mirror: static-chunk (512) the leaf rows, per-chunk left/right gather in ascending within-chunk order, exclusive prefix-sum to disjoint write offsets, parallel scatter into [all left | all right] — byte-identical to the serial stable two-pass gather"
    - "V5 bounds validated BEFORE the parallel region (rows walked ascending) so the lowest-index offending bin surfaces regardless of thread scheduling"

key-files:
  created: []
  modified:
    - crates/lgbm-treelearner/src/data_partition.rs
    - crates/lgbm-treelearner/Cargo.toml

key-decisions:
  - "Parallel scatter uses a raw-pointer SendPtr into disjoint, prefix-summed, in-bounds regions (no atomics) — the prefix sums make every chunk's write range pairwise-disjoint, so the unsafe is sound and the output is order-deterministic."
  - "split_parallel_equals_serial drives the serial Backend path and split_numeric_parallel DIRECTLY (not via env mutation) — std::env::set_var is unsafe on edition 2024 and would race with the parallel lib-test harness; the direct-call A/B is both safe and a stricter bit-exactness proof."
  - "The win does NOT materialize reliably end-to-end (reported honestly below). Partition is a stable 29% of train at 1Mx50, and the parallel path can cut it ~15-24% when the rayon pool is idle, but it regresses to +27-40% under contention because the histogram BUILD phase (69% of train) is ITSELF rayon-parallel — the same fork/join-contention lesson from the split-scan campaign. Kept gated and opt-out-able; no end-to-end train-wall win claimed."

patterns-established:
  - "Deterministic prefix-sum parallel partition: when a serial stable two-pass gather must be parallelized bit-exactly, static-chunk + per-chunk gather + exclusive-prefix-sum-to-disjoint-slots reproduces the [left|right] concatenation byte-for-byte."

requirements-completed: [QUICK-260622-ia0]

duration: 35min
completed: 2026-06-22
---

# Phase quick-260622-ia0: Parallelize DataPartition::split Summary

**Added a deterministic rayon static-chunk + exclusive prefix-sum reorder to `DataPartition::split` that is byte-identical to the serial path and gated on a leaf-row threshold — the bit-exact merge gate holds, but the end-to-end tall-narrow win does NOT reliably materialize (parallel partition contends with the already-rayon-parallel histogram build).**

## Performance

- **Duration:** ~35 min
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

### Task 1 — Parallel prefix-sum reorder (commit 9ac07fb)

- Added `rayon = "1.10"` to `lgbm-treelearner` (the exact version already used by `lgbm-compute`).
- Rewrote the numeric branch of `DataPartition::split` to gate on a leaf-row threshold:
  - `count >= par_split_threshold()` (default 16384, env `LGBM_PAR_SPLIT_THRESHOLD`) → new `split_numeric_parallel`.
  - Below threshold → the EXISTING serial Backend `data_partition` path, verbatim and unchanged.
- `split_numeric_parallel` mirrors C++ `ParallelPartitionRunner` (`data_partition.hpp`, `schedule(static, 512)`):
  1. V5 bounds (`num_bin > 0`, `threshold < num_bin`, every per-row `bin < num_bin`) validated BEFORE the parallel region, walking rows ascending, so the lowest-index offending bin's error matches the serial op regardless of scheduling (threat T-ia0-02).
  2. Routing math (`th = threshold + min_bin`, `−1` if `most_freq_bin == 0`; out-of-`[min,max]` → `default_to_right = most_freq_bin > threshold`) is byte-identical to `data_partition_cpu_native`.
  3. `par_chunks(512)` → per-chunk left/right global-id gather in ascending within-chunk order.
  4. Exclusive prefix-sum of per-chunk left/right counts → disjoint write offsets (`total_left` = serial `split_point`).
  5. Parallel scatter into the disjoint regions (raw-pointer `SendPtr`, no atomics) → `[all left | all right]`, byte-identical to the serial stable two-pass gather.
- Added `split_parallel_equals_serial`: a 5000-row scattered leaf (U16 width, `most_freq_bin == 0` branch) asserting the parallel slice is byte-identical to the serial Backend slice.
- Categorical split, the GPU/resident path, and all public signatures untouched.

### Task 2 — Bit-exact merge gate + end-to-end bench (no code change)

- Bit-exact merge gate: GREEN. `cargo test -p lgbm-treelearner --lib` (77 passed, incl. the new test + the 2 unchanged split unit tests) and `cargo test -p oracle-harness` (all green, incl. `raw_bin_train_matches_cpp_golden` — the f64-anchor goldens vs lib_lightgbm 4.6).
- Clippy clean on `data_partition.rs` (the 16 remaining workspace warnings are all pre-existing in `learner.rs`).

## Bench: tall-narrow 1M × 50 (A/B on the SAME binary)

Baseline = serial-forced via `LGBM_PAR_SPLIT_THRESHOLD=4294967295`; After = default (parallel fires on the root + early leaves). `LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=50`. Interleaved, warm, on a thermal/load-sensitive box.

| Metric | Baseline (serial) | After (parallel) |
|--------|-------------------|------------------|
| partition phase (ms) — samples | 667, 670, 676, 677, 705, 670, 656 | 510, 853, 580, 609, 600, 931, 851 |
| partition phase **median** | **~670 ms** (tight, 656–705) | **~609 ms** (high variance, 510–931) |
| partition **share** | 29.0–29.3% | 23.5–34.7% |
| train wall **median** | ~1.30 s | ~1.28 s (range 1.26–1.46) |

### Honest read

- The **bit-exact gate is the headline pass** — the parallel reorder is byte-identical to the serial gather, proven by `split_parallel_equals_serial` AND the `raw_bin_train` f64-anchor goldens (the partition order feeds the histogram subtraction trick, so any drift would surface there).
- The **end-to-end perf win does NOT reliably materialize.** The baseline partition is rock-stable at ~670 ms (29%). The parallel partition is **bimodal**: ~510–610 ms (−15 to −24%) when the rayon pool is otherwise idle, but it **regresses to ~850–930 ms (+27 to +40%) under contention.** Median partition is ~609 ms (≈ −9%), but the train-wall medians (~1.30 s vs ~1.28 s) sit **within run-to-run noise**.
- **Root cause of the contention:** the histogram BUILD phase is 69% of train at this shape and is ITSELF rayon-parallel (≥16384 rows, spike-005). The partition runs interleaved with — and on the same global rayon pool as — that already-saturated build. This is the exact "don't parallelize a phase harder when a neighbouring phase already saturates the pool" lesson from the split-scan campaign (FUSION wins, parallelizing-harder loses). The fork/join + scheduling jitter swamps the modest, gateable gain.

## Wide-shape sanity 1M × 500 (no off-target regression)

| Metric | Baseline (serial) | After (default) |
|--------|-------------------|-----------------|
| train wall | 7.97 s | 7.94 s (within noise) |
| partition share | 5.0% | 6.2% |

Partition is only ~5% at the wide shape, so the parallel-path variance is irrelevant to the 7.9 s wall — confirms no regression off the target shape.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] edition-2024 `std::env::set_var` is unsafe**
- **Found during:** Task 1 (test compile)
- **Issue:** The plan suggested toggling `LGBM_PAR_SPLIT_THRESHOLD` (0 vs usize::MAX) inside `split_parallel_equals_serial` to force both paths. On the workspace's Rust edition, `std::env::{set_var,remove_var}` are `unsafe` and, more importantly, mutating a process-global env races with the parallel `--lib` test harness.
- **Fix:** Rewrote the test to drive the serial Backend path and `split_numeric_parallel` DIRECTLY on the same scattered leaf and compare the resulting `indices` slices. This is both safe and a stricter, env-free bit-exactness proof. The env gate itself is still exercised in production via `par_split_threshold()`.
- **Files modified:** `crates/lgbm-treelearner/src/data_partition.rs`
- **Commit:** 9ac07fb

## Known Stubs

None.

## Self-Check: PASSED

- `crates/lgbm-treelearner/src/data_partition.rs` — FOUND (contains `split_numeric_parallel`, `par_split_threshold`, `split_parallel_equals_serial`).
- `crates/lgbm-treelearner/Cargo.toml` — FOUND (contains `rayon = "1.10"`).
- Commit 9ac07fb — FOUND in `git log`.
- `cargo test -p lgbm-treelearner --lib` — 77 passed, 0 failed.
- `cargo test -p oracle-harness` — all green (incl. `raw_bin_train_matches_cpp_golden`).

---

## FINAL DISPOSITION (orchestrator + user decision): REVERTED ✗

The implementation above was **bit-exact** (independently re-verified: `split_parallel_equals_serial`,
`raw_bin_train_matches_cpp_golden`, full oracle suite all green) but an **end-to-end NULL**:
train-wall stayed within run-to-run noise and the partition phase went **bimodal** (−15–24%
when the rayon pool is idle, +27–40% under contention) because the histogram BUILD (69% of
train) is already rayon-parallel and saturates the same pool. This re-confirms the split-scan
campaign lesson: *don't parallelize a phase against an already-parallel neighbour — fuse, don't
contend.*

Per the no-regression-anywhere discipline, the change was **reverted** (commit `52d63b3`,
reverting `1ff6891` — the original code commit hash; the SUMMARY's earlier `9ac07fb`/other
hashes refer to intermediate executor commits). The rayon dependency was removed from
`lgbm-treelearner`. A NOTE was left at `DataPartition::split` documenting the null so it is not
naively re-attempted. The root-cause finding stands (partition is 29% serial at tall-narrow);
the lever to capture it is NOT independent parallelization but a future FUSE of partition with a
neighbouring phase, or reducing the build's pool pressure first.

**Net production change: none** (reverted). Value delivered: a measured negative result + the
in-code NOTE + this evidence.
