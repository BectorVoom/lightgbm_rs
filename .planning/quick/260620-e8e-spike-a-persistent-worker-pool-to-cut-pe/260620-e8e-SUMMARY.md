---
quick_id: 260620-e8e
title: Spike — persistent worker pool to cut per-leaf dispatch cost
subsystem: cpu-treelearner
tags: [spike, fork-join-floor, persistent-pool, dispatch-latency, fusion-gate, NULL, parity-gate]
status: complete
result: NULL
date: 2026-06-20
requires: [260620-dpk]   # the per-leaf fixed-cost decomposition this spike builds on
provides: []             # negative result + reusable isolated dispatch micro-bench
affects: []              # NO library hot-path code changed
key-files:
  created:
    - crates/lgbm-compute/examples/dispatch_microbench.rs   # throwaway isolated dispatch micro-bench
    - .planning/quick/260620-e8e-spike-a-persistent-worker-pool-to-cut-pe/260620-e8e-FINDINGS.md
  modified: []
decisions:
  - "NULL at Task 1: keep rayon par_iter as the per-leaf dispatch primitive; fusion gate stays at core_scaled_threshold(100/130)."
metrics:
  tasks: 3   # Task 1 (built), Task 2 (skipped on NULL), Task 3 (NULL-confirmation checkpoint)
  files_created: 2
  files_modified: 0
---

# Quick 260620-e8e: Persistent worker pool to cut per-leaf dispatch cost — Summary

**Fail-fast SPIKE — NULL by design, NULL is the result.** An isolated dispatch micro-bench
proves a hand-rolled persistent pool is genuinely 5.2–8.5× cheaper PER DISPATCH than rayon's
`par_iter` join, but the absolute per-leaf saving is only ~0.23–0.31% of the ms-scale per-leaf
WORK wall — two orders of magnitude below the ≥6% needed to move the ~100/130-feat fusion gate.
The per-leaf cost is WORK-bound, not dispatch-bound. No library code changed; the spike died
cheaply at Task 1.

## What was done

- **Task 1 (the KILL SWITCH — where the budget went):** Built a throwaway cargo example
  `crates/lgbm-compute/examples/dispatch_microbench.rs`, NOT wired into the learner, touching
  NO library hot-path code. It times the BARE per-dispatch cost of fanning out N tiny
  per-feature tasks and joining, for four candidates:
  - **(a) rayon-baseline** — `par_iter` over the global pool (today's primitive).
  - **(b) spin-pool** — std-only hand-rolled fixed-lane pool: W lanes parked on an `AtomicU64`
    epoch, static block split, spin-wait `AtomicUsize` completion barrier (low-latency / burns-cores end).
  - **(c) block-pool** — same pool but blocking: workers `park()`, driver `unpark`s, `Mutex`+`Condvar`
    completion barrier (no-burn / higher-wake-latency end).
  - **(d) rayon-NOOP-floor** — rayon over a no-op closure (dispatch floor reference).

  Per-task work = a tight fold over a 384-element buffer (a realistic per-feature scan proxy,
  ~a few µs) so dispatch is measured against real granularity, not a no-op. N∈{20,40,80};
  ≥10k inner dispatches/cell; 1 warmup outer rep discarded; median + p25/p75. The pools are
  bound to the same `rayon::current_num_threads()` lane count for a fair barrier-vs-barrier
  comparison. A built-in correctness cross-check confirms the pools' disjoint-write output is
  **bit-identical** to rayon's (order + values preserved). **std primitives only — no
  `crossbeam`, no new external dependency; cubecl 0.10 pinned.**

- **Task 2 (gated prototype): SKIPPED — Task 1 was NO-GO/NULL.** No `LGBM_DISPATCH_POOL` gate,
  no `build_fix_scan_impl`/`subtract_scan_impl` change. Zero library hot-path modification.

- **Task 3 (parity / NULL-confirmation checkpoint):** Confirmed the example compiles + runs +
  emits a verdict; confirmed my commit's `git diff` touches ONLY the example + FINDINGS (no
  `src/`); ran the parity suite once (green); clean `cargo check`. **This is a blocking-human
  checkpoint — see "Checkpoint" below.**

## Task-1 per-dispatch micro-bench (verbatim)

```
# rayon worker threads = 16  (RAYON_NUM_THREADS=unset)
# correctness: spin bit-eq rayon = true, block bit-eq rayon = true (disjoint-write order preserved)
```

| candidate         |  N | p25 µs | med µs | p75 µs |
|-------------------|---:|-------:|-------:|-------:|
| rayon-NOOP-floor  | 20 |  3.056 |  3.767 |  8.727 |
| rayon-baseline    | 20 | 81.352 | 85.330 | 114.604 |
| spin-pool         | 20 | 15.730 | 16.471 | 17.302 |
| block-pool        | 20 | 54.962 | 77.034 | 99.095 |
| rayon-NOOP-floor  | 40 |  3.627 |  9.247 |  9.458 |
| rayon-baseline    | 40 | 155.731 | 209.973 | 216.304 |
| spin-pool         | 40 | 23.604 | 24.627 | 25.808 |
| block-pool        | 40 | 103.083 | 137.237 | 181.319 |
| rayon-NOOP-floor  | 80 |  4.378 |  4.519 |  4.649 |
| rayon-baseline    | 80 | 310.340 | 326.781 | 431.617 |
| spin-pool         | 80 | 39.424 | 40.877 | 42.439 |
| block-pool        | 80 | 278.591 | 314.808 | 353.471 |

### GO/NO-GO verdict (verbatim)

```
N=20: rayon=85.330µs spin=16.471µs(×5.18) block=77.034µs(×1.11)  best per-leaf saving≈137.72µs = 0.230% of ~60ms wall  [latency≥2×:true magnitude≥6%:false]
N=40: rayon=209.973µs spin=24.627µs(×8.53) block=137.237µs(×1.53) best per-leaf saving≈370.69µs = 0.309% of ~120ms wall [latency≥2×:true magnitude≥6%:false]
N=80: rayon=326.781µs spin=40.877µs(×7.99) block=314.808µs(×1.04) best per-leaf saving≈571.81µs = 0.238% of ~240ms wall [latency≥2×:true magnitude≥6%:false]

VERDICT: NO-GO  → VERDICT: NULL
```

**Margin:** the GO criterion is conjunctive (≥2× per-dispatch cut **AND** ≥6%-of-per-leaf-wall
absolute saving). The spin pool clears the latency half by a wide margin (×5.18–×8.53) but
**every candidate fails the magnitude half by ~20–30×** (0.23–0.31% « 6%). → **NULL.**

## Idle-burn (Task 2 did NOT run, but the axis is decided at Task 1)

The spin pool's latency win does **not** survive the idle-burn caveat even if magnitude held:
spinning on the epoch pins all worker cores at ~100% during the sequential between-leaf gaps
(argmax / data-partition / bookkeeping) and whenever the library is loaded-but-idle
(prediction-only, post-train) — a NO-SHIP for a library. The blocking pool (no-burn end) is
**not faster than rayon** in absolute median (×1.04–1.53), so the no-burn end has no win to
ship either. Both ends fail: spin = fast-but-burns, block = no-burn-but-no-win.

## Parity (NULL path — run once to confirm green)

My commit changed **zero library code** (`git diff HEAD~1 HEAD` = example + FINDINGS only), so
CPU f64 bit-exact parity is trivially untouched. Confirmed green:

- `cargo test -p oracle-harness --test kernel_parity` → **6 passed, 0 failed**.
- `cargo test -p oracle-harness --test learner_parity` → **29 passed, 0 failed**.
- `cargo check -p lgbm-compute -p lgbm-treelearner -p oracle-harness` → **clean**.
- `cargo run -p lgbm-compute --example dispatch_microbench --release` → compiles + runs + emits
  `VERDICT: NULL`.

No reference trees (`LightGBM/`, `LightGBM-release-4.6.0.99/`, `cuml-main/`, `.serena/`) or
gitignored bench data were git-added.

## Deviations from plan

- **Task-1 implementation bug (Rule 1 — auto-fixed, in the throwaway example):** the first build
  segfaulted because the spin/block pools passed `&[Vec<f64>]` to a worker by casting the slice's
  data pointer to `*const Vec<Vec<f64>>` and dereferencing — a layout mismatch (a `Vec` header is
  ptr/cap/len, not the slice element data). Fixed by storing the slice element base pointer
  (`bufs.as_ptr() as usize`) and indexing per-element (`&*bufs_base.add(i)`). Re-ran clean; the
  built-in correctness cross-check now passes bit-eq vs rayon. Confined entirely to the throwaway
  example — no library impact.
- **No `crossbeam` reached for** (the plan's STOP-and-flag clause never triggered): both
  candidates were expressible with std `thread` + `atomic` + `Mutex`/`Condvar`. No dep added.
- **Task 2 skipped by design** (Task 1 NULL) — not a deviation, the planned NULL branch.

## Known stubs

None. The micro-bench is a complete, self-contained measurement; no placeholder data paths.

## Threat flags

None. The example is a host-only measurement binary; it adds no network/auth/file/schema surface.

## Checkpoint (Task 3 — blocking-human)

Task 3 is a `checkpoint:human-verify` with `gate="blocking-human"`. NULL-path verification is
complete (example committed + runs + verdict; no library code changed; parity 6/6 + 29/29 green;
clean check; no reference trees added). **Awaiting human "approved" to accept the documented
NULL.** The fusion gate stays at `core_scaled_threshold(100/130)`; no behavioral change.

## Self-Check: PASSED

- FOUND: crates/lgbm-compute/examples/dispatch_microbench.rs
- FOUND: .planning/quick/260620-e8e-spike-a-persistent-worker-pool-to-cut-pe/260620-e8e-FINDINGS.md
- FOUND: .planning/quick/260620-e8e-spike-a-persistent-worker-pool-to-cut-pe/260620-e8e-PLAN.md
- FOUND commit 82b9a8f (test: micro-bench + FINDINGS)
- FOUND commit 9502a10 (docs: PLAN.md)
