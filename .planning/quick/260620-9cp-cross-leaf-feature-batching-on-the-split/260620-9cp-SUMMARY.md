---
quick_id: 260620-9cp
title: Cross-leaf/feature batching on the split scan
type: quick
date: 2026-06-20
status: complete
decision: NULL (honest — serial stays effective default)
key_files:
  created:
    - .planning/quick/260620-9cp-cross-leaf-feature-batching-on-the-split/260620-9cp-FINDINGS.md
  modified:
    - crates/lgbm-compute/src/lib.rs
decisions:
  - "HONEST NULL: parallel per-feature split scan is NOT adopted as the default. The isolated SCAN_NS win on wide (−13%) does NOT translate to a warm train-wall win (+11% WORSE); narrow regresses 3.3× on SCAN / 1.8× on train-wall. Per the project's audit-before-wire value, the serial loop stays the effective default."
  - "Gate key = feats.len() (scan work ∝ features × bins, not leaf rows). Feature count alone separates the narrow/wide regimes cleanly (bin counts near-uniform across a leaf's spine), so the simpler key was kept — no need for feats.len() * mean_num_bin."
  - "Committed default LGBM_PAR_SCAN_THRESHOLD = usize::MAX (gate never trips by default); parallel path reachable only via explicit env override, and proven bit-exact FORCED-ON (LGBM_PAR_SCAN_THRESHOLD=0)."
  - "CpuBackend now has an EXPLICIT find_best_splits_batched override (was relying on the Backend trait default serial loop); RocmBackend's fused-launch override is untouched."
metrics:
  tasks: 3
  files_changed: 1
  commits: 4
requirements: [PERF-SPLIT-SCAN-01]
---

# Quick 260620-9cp: Cross-leaf/feature batching on the split scan — Summary

**One-liner:** Built an order-preserving, size-gated parallel per-feature split scan
in `CpuBackend` (ONE rayon fork/join per leaf, amortized across all features), measured
it warm A/B on narrow + wide configs, and — because the isolated SCAN win does not survive
into warm train-wall (the scan fork/join contends with the already-rayon-parallel BUILD
path) — landed an **honest NULL**: serial stays the effective default, parallel kept
env-reachable and proven bit-exact forced-on.

## Decision: NULL (serial stays the effective default)

The parallel scan is **not** adopted. `LGBM_PAR_SCAN_THRESHOLD` defaults to `usize::MAX`,
so the gate never trips in production; the parallel path is reachable only via an explicit
env override and is proven byte-identical to serial (bit-exact gate green forced-on).

## What was done

### Task 1 — Ordered, size-gated parallel split scan (commit 50ebd26)
`CpuBackend` previously used the `Backend` trait-default `find_best_splits_batched`
(serial `for f in feats`). Added an **explicit override** inside `impl Backend for
CpuBackend` that:
- Below `par_scan_threshold()`: runs the serial loop verbatim (zero behavior change;
  empty feats ⇒ empty Vec, threat T-lsx-03 preserved).
- At/above threshold: hoists the per-feature `[slot_off, slot_off+2*num_bin)` range
  validation BEFORE the parallel map (walking features in ascending index, returning the
  **lowest-index** `LengthMismatch`/`Runtime` error — so a parallel error race can never
  change WHICH error surfaces, matching the serial first-failure behavior), then builds the
  result with `feats.par_iter().zip(ranges.par_iter()).map(...).collect::<Result<Vec<_>,_>>()?`.
  `par_iter().map(...).collect()` **preserves input order**, so `out[i]` corresponds to
  `feats[i]` — byte-identical to the serial loop, which is what keeps the caller's
  cross-feature argmax (`scan_leaf_histogram`) bit-exact.
- Added `par_scan_threshold()` (env `LGBM_PAR_SCAN_THRESHOLD`) keyed on `feats.len()`
  (scan work ∝ features × bins, NOT leaf rows — a rows-based gate would be wrong).
- `RocmBackend::find_best_splits_batched` (the fused-launch override) is **untouched**.

### Task 2 — A/B measurement + threshold tuning (commit 4a533f5)
Ran the existing `crates/lgbm-treelearner/examples/bench_split_scan.rs` warm
(`LGBM_PHASE_PROF=1`, cold iteration discarded, WARM_REPS=3 internal) on narrow (10 feat)
and wide (120 feat), 3 outer runs each, A = serial (`LGBM_PAR_SCAN_THRESHOLD=1000000`)
vs B = parallel (`LGBM_PAR_SCAN_THRESHOLD=0`). Tuned the default to `usize::MAX` (NULL).

### Task 3 — HARD parity merge gate (this commit + checkpoint)
Ran all four parity/test suites in BOTH default and forced-on (`LGBM_PAR_SCAN_THRESHOLD=0`)
modes. All green in both modes — the forced-on run is the load-bearing proof that the
parallel path is byte-identical to serial (CPU f64 anchor authoritative).

## A/B numbers (verbatim, cubecl-cpu f64 anchor; warm; SCAN_NS = isolated scan, warm_wall = 3-internal-rep train wall)

### A = SERIAL (LGBM_PAR_SCAN_THRESHOLD=1000000)
```
narrow: SCAN_NS=33514816  BUILD_NS=61241479  SCAN%=35.37  warm_wall=168.842ms
wide:   SCAN_NS=353187074 BUILD_NS=438755232 SCAN%=44.60  warm_wall=1066.232ms
narrow: SCAN_NS=31442369  BUILD_NS=57299246  SCAN%=35.43  warm_wall=156.259ms
wide:   SCAN_NS=341779984 BUILD_NS=428415356 SCAN%=44.38  warm_wall=1040.916ms
narrow: SCAN_NS=31740105  BUILD_NS=57044143  SCAN%=35.75  warm_wall=154.304ms
wide:   SCAN_NS=357059589 BUILD_NS=453774998 SCAN%=44.04  warm_wall=1086.194ms
```

### B = PARALLEL (LGBM_PAR_SCAN_THRESHOLD=0)
```
narrow: SCAN_NS=99330716  BUILD_NS=72888680  SCAN%=57.68  warm_wall=279.373ms
wide:   SCAN_NS=309604222 BUILD_NS=517620544 SCAN%=37.43  warm_wall=1174.816ms
narrow: SCAN_NS=110811544 BUILD_NS=77767380  SCAN%=58.76  warm_wall=301.332ms
wide:   SCAN_NS=306883355 BUILD_NS=526413396 SCAN%=36.83  warm_wall=1184.005ms
narrow: SCAN_NS=104365565 BUILD_NS=73069866  SCAN%=58.82  warm_wall=286.682ms
wide:   SCAN_NS=308615025 BUILD_NS=525894615 SCAN%=36.98  warm_wall=1184.289ms
```

### DEFAULT-AS-TUNED before NULL (threshold=64: narrow serial, wide parallel) — confirms wide regresses even when narrow is gated out
```
narrow: SCAN_NS=33315181  BUILD_NS=60257303  warm_wall=166.403ms
wide:   SCAN_NS=316734910 BUILD_NS=523105021 warm_wall=1191.231ms
narrow: SCAN_NS=32057178  BUILD_NS=56691294  warm_wall=157.420ms
wide:   SCAN_NS=320019099 BUILD_NS=548659124 warm_wall=1231.063ms
narrow: SCAN_NS=32628296  BUILD_NS=59485009  warm_wall=162.339ms
wide:   SCAN_NS=305275701 BUILD_NS=496755887 warm_wall=1143.608ms
```

### Medians (3-run) and verdict
| config | metric    | A=serial | B=parallel | Δ            | sign-stable | verdict        |
|--------|-----------|----------|------------|--------------|-------------|----------------|
| narrow | SCAN_NS   | 31.74ms  | 104.37ms   | +229% WORSE  | yes          | regress        |
| narrow | train-wall| 156.3ms  | 286.7ms    | +84% WORSE   | yes          | regress        |
| wide   | SCAN_NS   | 353.19ms | 308.62ms   | −13% better  | yes          | scan win        |
| wide   | train-wall| 1066.2ms | 1184.3ms   | +11% WORSE   | yes          | **regress**    |

**Why the isolated scan win does not survive:** the per-leaf scan fork/join contends with
the already-rayon-parallelized BUILD path — wide `BUILD_NS` rose 438→520ms when the scan was
parallelized. The 13% SCAN_NS reduction is more than eaten by the BUILD slowdown, so warm
**train-wall regresses +11% on wide** (sign-stable: B 1175–1184ms vs A 1041–1086ms, no
overlap). Narrow is catastrophically worse on both metrics (10 features can't amortize one
fork/join). Gating at threshold=64 (narrow serial, wide parallel) still regresses wide.

**WIN/NULL criterion:** adopt only if wide shows a sign-stable train-wall gain AND narrow
shows no sign-stable regression. Wide train-wall is a sign-stable **regression** ⇒ criterion
fails ⇒ **honest NULL**. No win manufactured.

## Gate-key choice
Kept `feats.len()` (the simplest defensible scan-work proxy). Feature count alone cleanly
separates the regimes (narrow 10 vs wide 120; bin counts near-uniform across a leaf's spine
features), so the `feats.len() * mean_num_bin` work key was not needed. Documented in the
`par_scan_threshold()` doc comment.

## Tuned default threshold
`LGBM_PAR_SCAN_THRESHOLD` default = `usize::MAX` (gate never trips in production; serial is
the effective default). Parallel path reachable only via explicit env override.

## HARD parity merge gate (verbatim counts — CPU f64 anchor authoritative)

### Default path (gate in its tuned NULL state, threshold = usize::MAX)
```
kernel_parity:       6 passed; 0 failed; 0 ignored
learner_parity:     29 passed; 0 failed; 0 ignored
lgbm-compute --lib: 32 passed; 0 failed; 1 ignored
lgbm-treelearner:   76 passed; 0 failed; 2 ignored   (unittests src/lib.rs)
                     1 passed; 0 failed; 0 ignored    (tests/quantized_pipeline.rs)
```

### Parallel FORCED ON (LGBM_PAR_SCAN_THRESHOLD=0) — load-bearing bit-exactness proof
```
kernel_parity:       6 passed; 0 failed; 0 ignored
learner_parity:     29 passed; 0 failed; 0 ignored
lgbm-compute --lib: 32 passed; 0 failed; 1 ignored
lgbm-treelearner:   76 passed; 0 failed; 2 ignored   (unittests src/lib.rs)
                     1 passed; 0 failed; 0 ignored    (tests/quantized_pipeline.rs)
```

**PASS:** every command green in BOTH default and forced-on runs. The forced-on run proves
the parallel path is byte-identical to serial — the CPU f64 anchor stays bit-exact under
parallelism. rocm `--features rocm` cells are gfx1100-only / non-blocking and were not run
(CPU f64 anchor is the merge gate).

## Deviations from plan
- **CpuBackend used the trait-default `find_best_splits_batched`, not an existing override.**
  The plan described editing "the method at lib.rs:563" inside the CpuBackend block; in fact
  that method is the `Backend` trait default and CpuBackend did not override it. Implemented
  the lever as an **explicit CpuBackend override** (cleanest scoping: the trait default stays
  the serial bit-exact-anchor reference, RocmBackend's fused override is untouched). Behavior
  is identical to what the plan intended. [Rule 3 — blocking issue resolved by scoping.]
- No other deviations. RocmBackend (lib.rs:1251) untouched; untracked reference trees
  (LightGBM/, LightGBM-release-4.6.0.99/, cuml-main/, .serena/) never git-added.

## Known stubs
None.

## Self-Check: PASSED
- `crates/lgbm-compute/src/lib.rs` modified — FOUND (par_scan_threshold + CpuBackend override).
- `.planning/quick/260620-9cp-cross-leaf-feature-batching-on-the-split/260620-9cp-FINDINGS.md` — FOUND.
- Commit 50ebd26 (Task 1) — FOUND.
- Commit 4a533f5 (Task 2 NULL tune) — FOUND.
- Parity gate green in BOTH default and LGBM_PAR_SCAN_THRESHOLD=0 — verified above.
