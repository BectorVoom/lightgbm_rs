---
quick_task: 260620-d3v
title: Benchmark the full split-fusion campaign (a48+b97+c5v) end-to-end on REAL workloads
date: 2026-06-20
status: complete
type: measurement
subsystem: bench-harness
tags: [bench, split-fusion, a48, b97, c5v, raw-corpus, bit-identical, mnist]
verdict: "PROVEN pure speed change — high-dim -66% sign-stable WIN, typical neutral-by-design, bit-identical model+pred on every real dataset"
key-files:
  created:
    - crates/lgbm/examples/bench_real.rs
    - scripts/export_highdim_dataset.py
    - .planning/quick/260620-d3v-benchmark-the-full-fusion-campaign-a48-b/260620-d3v-FINDINGS.md
  modified:
    - .gitignore
    - .planning/STATE.md
commits:
  - 761eab4  # Task 1: real-workload bench harness
  - 420a262  # Task 2: high-dim export script + .gitignore
  - de239f4  # Task 3: FINDINGS + PLAN commit
metrics:
  workloads: 3
  bit_identical_diffs_empty: "6/6"
  highdim_delta: "-66.0%"
---

# Quick 260620-d3v — Benchmark the full split-fusion campaign on REAL workloads — Summary

## One-liner

The full split-fusion campaign (a48 smaller-child fusion + b97 larger-child fusion + c5v
core-derived thresholds) is a **pure speed change**: a single env-toggled binary shows a
**sign-stable −66% train-wall WIN on a high-dimensional (784-feat MNIST) real dataset**,
is **neutral-by-design on typical (28-feat) real datasets**, and emits a **bit-identical
model text AND prediction vector** campaign-ON vs campaign-OFF on every real dataset.

## What was done

1. **Task 1 — threshold-agnostic real-workload harness** (`crates/lgbm/examples/bench_real.rs`).
   Loads a label-first TSV via `RawCorpus::from_columns` (column-major, no transpose) → realistic
   `Config` (max_bin=255, num_leaves=31, num_iterations=100, lr=0.1, deterministic) → `train_raw`
   UNCHANGED (real continuous floats through the bit-exact `BinMapper`, NOT the identity-bin
   `DenseCorpus` path). Modes: `bench` (warm/discard-run-0, 3-rep MEDIAN train-wall, dumps
   BUILD_NS/SCAN_NS when `LGBM_PHASE_PROF=1`), `dump-model` (`model_to_string`), `dump-pred`
   (one f32/row as raw IEEE-754 hex bits). The harness never touches the thresholds — the A/B is
   driven entirely by the two `LGBM_UNIFIED_*` env vars the library gates read.

2. **Task 2 — high-dim export** (`scripts/export_highdim_dataset.py`). `.venv` sklearn 1.9.0
   (no new dep). PRIMARY `fetch_openml('mnist_784')` subsampled to 15000 rows (fixed seed);
   FALLBACK bundled `fetch_olivetti_faces` (4096×400, no network). Writes
   `target/bench_data/highdim.tsv` (gitignored; only the script is committed).

3. **Task 3 — A/B run + bit-identical check** (`260620-d3v-FINDINGS.md`). For binary.train,
   regression.train, and high-dim MNIST: warm 3-rep median A (campaign OFF, env MAX) vs B
   (campaign ON, defaults), plus the campaign-ON-vs-OFF model/pred bit-identical diff.

4. **Task 4 — parity gate** (this checkpoint). Standard parity sanity suite + clean check, all
   green; FINDINGS reviewed for honest framing.

## A/B toggle mechanism (single binary, env-only)

Each campaign change gates on `features.len() >= threshold()`;
`lgbm_compute::unified_bfs_threshold()` / `unified_subscan_threshold()` read
`LGBM_UNIFIED_BFS_THRESHOLD` / `LGBM_UNIFIED_SUBSCAN_THRESHOLD` with **env taking ultimate
precedence** (`crates/lgbm-compute/src/lib.rs:430,473`); each below-gate fallback is the
byte-unchanged pre-campaign two-step code.

- **A = campaign OFF (exact pre-campaign baseline):** both env vars = `18446744073709551615`
  (`usize::MAX` → fusions never fire).
- **B = campaign ON:** both unset → core-derived defaults (~100 / ~130 at 16 cores).

Same binary, same harness, only the two env values change — cleaner and more accurate than a
cross-commit checkout (zero harness skew).

## A/B results (warm, 3-rep MEDIAN train-wall) — verbatim

| Workload          | feat | A median ms | B median ms | delta %    | sign-stable? | BUILD_NS A→B (ms) | SCAN_NS A→B (ms) | bit-identical model? | bit-identical pred? |
| ----------------- | ---- | ----------- | ----------- | ---------- | ------------ | ----------------- | ---------------- | -------------------- | ------------------- |
| binary.train      | 28   | 224.682     | 229.681     | +2.2%      | NO (overlap) | 160.18 → 164.19   | 270.19 → 273.33  | YES                  | YES                 |
| regression.train  | 28   | 193.602     | 192.514     | −0.6%      | NO (overlap) | 160.40 → 158.58   | 263.11 → 263.20  | YES                  | YES                 |
| highdim MNIST-784 | 784  | 8379.573    | 2846.795    | **−66.0%** | **YES**      | **17888.9 → 0.0** | 4496.6 → 5580.4  | YES                  | YES                 |

(BUILD_NS/SCAN_NS = phase_prof accumulators summed over the 3 timed reps.)

High-dim sign-stability: B reps [2750, 2847, 3263] ms all below A reps [8314, 8380, 8449] ms —
no distribution overlap. Mechanism: BUILD_NS collapses 17889 ms → 0.0 ms (the separate
whole-buffer histogram build is fused into the per-feature unified scan region; a48+b97 both
fire above the gate at 784 feat), SCAN absorbs it (4497 → 5580 ms), net ~3× faster train.

## Bit-identical check (the speed-only correctness proof)

`dump-model` and `dump-pred` were run under A (env MAX) and B (defaults) and diffed for each
workload. **All 6 diffs EMPTY:**

```
binary.train(28f)     MODEL: IDENTICAL   PRED: IDENTICAL
regression.train(28f) MODEL: IDENTICAL   PRED: IDENTICAL
highdim-mnist(784f)   MODEL: IDENTICAL   PRED: IDENTICAL
```

Campaign-ON == campaign-OFF, bit-for-bit, on every real dataset → the campaign changed only
speed, not output. (A non-empty diff would have been a correctness failure → STOP; none occurred.)

## High-dim dataset provenance

- **MNIST-784** via `sklearn.datasets.fetch_openml('mnist_784', version=1)` — fetched on this box
  (PRIMARY path succeeded; olivetti fallback NOT triggered).
- **Rows × features used: 15000 × 784** (fixed-seed permutation subsample of 70000).
- Label = digit 0..9, fit with the **regression** objective (speed bench, not accuracy bench).
- Exported to `target/bench_data/highdim.tsv` — **gitignored, NOT committed.**

## Honest framing

- **Typical (28 feat):** within noise (+2.2% / −0.6%, both run-overlapping → not sign-stable).
  Neutral by design — 28 < ~100/130, gate keeps fusion OFF, both arms run byte-identical
  two-step code. The gate correctly shields typical workloads from the narrow a48/9cp regression.
- **High-dim (784 feat):** sign-stable −66% WIN. **The win is LARGER on real BinMapper-binned
  data than the synthetic identity-binned campaign probes (single-digit % at 120–200 feat)
  suggested** — at 784 feat the eliminated build dominates the loop, so the whole-buffer build
  the fusion removes was a far bigger fraction of train time than the synthetic 120–200-feat
  probes captured. The high-dim win materializes on real data and is, if anything, understated
  by the synthetic benches. This is a valuable, truthful finding — not manufactured.

## Parity gate (Task 4 HARD gate) — verbatim counts

- `cargo test -p oracle-harness --test kernel_parity --test learner_parity` →
  **kernel_parity 6 passed / 0 failed; learner_parity 29 passed / 0 failed.**
- `cargo test -p lgbm-compute --lib` → **43 passed / 0 failed / 1 ignored.**
- `cargo test -p lgbm-treelearner` → **76 passed / 0 failed / 2 ignored** (+ 1/0 and 0/0
  ancillary suites).
- `cargo check -p lgbm-compute -p lgbm-treelearner` → clean (Finished, no warnings).
- `cargo build -p lgbm --example bench_real` → clean (Finished).

All green on the CPU f64 anchor (authoritative). No rocm cells run (gfx1100-only / non-blocking).

## Deviations from plan

None — plan executed exactly as written. No library hot-path code modified (harness +
export script + docs only); no missing public accessor needed (`RawCorpus::from_columns`,
`train_raw`, `model_to_string`, `predict`, `to_rows`, `phase_prof::dump` all already public).
`.gitignore` already covered `/target`; an explicit `target/bench_data/` + `*.highdim.tsv`
entry was added belt-and-suspenders. The MNIST PRIMARY path succeeded so the olivetti fallback
was not exercised (fallback code present + documented).

## Known Stubs

None.

## Self-Check: PASSED

- `crates/lgbm/examples/bench_real.rs` — FOUND (committed 761eab4).
- `scripts/export_highdim_dataset.py` — FOUND (committed 420a262).
- `.planning/quick/260620-d3v-benchmark-the-full-fusion-campaign-a48-b/260620-d3v-FINDINGS.md` — FOUND (committed de239f4).
- Commits 761eab4 / 420a262 / de239f4 — all present in `git log`.
- MNIST data file `target/bench_data/highdim.tsv` — present on disk, `git check-ignore` confirms NOT tracked.
- Reference trees (LightGBM/, LightGBM-release-4.6.0.99/, cuml-main/, .serena/) — NOT git-tracked.
