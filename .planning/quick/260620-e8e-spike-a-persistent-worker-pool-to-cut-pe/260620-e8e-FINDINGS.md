# 260620-e8e — FINDINGS: persistent worker pool to cut per-leaf dispatch cost

**Fail-fast SPIKE — NULL by design, and NULL is the result.** A throwaway, isolated
dispatch micro-bench (`crates/lgbm-compute/examples/dispatch_microbench.rs`, NOT wired
into the learner, NO library hot-path code touched) times the BARE per-dispatch cost of
fanning out N tiny per-feature tasks and joining, for rayon's global-pool `par_iter`
(the primitive the fused per-leaf region uses today) vs two hand-rolled std-only
persistent pools (spin-wait and blocking).

Backend/box: cubecl-cpu f64 anchor, 16-core, `RAYON_NUM_THREADS=unset` (16 workers; the
hand-rolled pools bound to the same 16 lanes for a fair barrier-vs-barrier comparison).
std primitives ONLY (`std::thread` + `std::sync::atomic` + `Mutex`/`Condvar`) — **no new
external dependency added**; cubecl 0.10 pinned. Per-task work = a tight fold over a
384-element buffer (a realistic per-feature scan proxy, ~a few µs) so dispatch is
measured against real granularity, not a no-op. ≥10k inner dispatches per cell, 1
warmup outer rep discarded, median + p25/p75 reported.

---

## Task 1 — per-dispatch micro-bench (verbatim)

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

### GO/NO-GO analysis (verbatim)

```
N=20: rayon=85.330µs spin=16.471µs(×5.18) block=77.034µs(×1.11)  best per-leaf saving≈137.72µs = 0.230% of ~60ms wall  [latency≥2×:true magnitude≥6%:false]
N=40: rayon=209.973µs spin=24.627µs(×8.53) block=137.237µs(×1.53) best per-leaf saving≈370.69µs = 0.309% of ~120ms wall [latency≥2×:true magnitude≥6%:false]
N=80: rayon=326.781µs spin=40.877µs(×7.99) block=314.808µs(×1.04) best per-leaf saving≈571.81µs = 0.238% of ~240ms wall [latency≥2×:true magnitude≥6%:false]
```

**VERDICT: NO-GO → NULL.**

---

## Why NULL (the honest read)

The GO criterion is **conjunctive**: a candidate must cut rayon's per-dispatch by **≥2×
AND** the absolute per-leaf saving must be a **meaningful fraction (≥6%) of the ~100-feat
per-leaf train-wall**. Exactly ONE half holds:

1. **Latency ≥2×: the spin pool PASSES, easily.** The hand-rolled spin pool (W persistent
   lanes parked on an `AtomicU64` epoch, static block split, spin-wait completion barrier)
   dispatches **5.2–8.5× cheaper** than rayon's median `par_iter` join for N∈{20,40,80}.
   rayon's work-stealing split + join barrier genuinely carries more per-dispatch overhead
   than a fixed-lane epoch-bump + spin-barrier for these tiny, perfectly-balanced fan-outs.
   This is a real, reproducible latency win and the correctness cross-check confirms the
   pool's disjoint-write output is **bit-identical** to rayon's (order + values preserved).

2. **Magnitude ≥6%: ALL candidates FAIL, by ~20–30×.** The absolute per-dispatch delta is
   tens-to-hundreds of µs; a leaf does ~2 such dispatches (BFS build + SUB subtract), so the
   *whole* per-leaf dispatch saving is **≈0.14–0.57 ms**. The per-leaf train-wall is
   **milliseconds** (dpk: BFS+SUB per-leaf totals ~96–155 ms BFS + ~51–97 ms SUB across the
   20k-row sweep; the par region is 88–99.9% *actual fold/fix/scan WORK*, not dispatch). The
   dispatch saving is therefore **~0.23–0.31% of the per-leaf wall** at every measured N —
   **two orders of magnitude below** the ~6% needed to plausibly move a ~100-feat crossover.

This is dpk's result restated from the other side: dpk proved the per-leaf fixed cost is
~99% fork/join floor **+ parallel WORK** and <1% allocation. The "+ WORK" is the load-bearing
term. A cheaper *dispatch primitive* only attacks the dispatch sliver — which is itself a
small fraction of the already-small fixed cost. Even a **zero-cost** dispatch (infinitely
fast pool) removes <1% of the per-leaf wall and cannot move the gate.

### The spin pool's latency win does NOT survive the idle-burn caveat

Per the plan's ship/no-ship axis, the spin candidate is **not auto-GO on latency alone**.
Even if the magnitude bar were met (it is not), the spin pool **continuously burns all
worker cores** while spinning on the epoch — including the sequential between-leaf gaps
(argmax, data-partition, tree bookkeeping) and whenever the library is loaded-but-idle
(prediction-only, post-train). For a *library*, pinning N cores at ~100% while idle is a
NO-SHIP regardless of dispatch latency. The blocking pool (the no-burn end) is **not
faster than rayon** in absolute median (×1.04–1.53), confirming the classic tradeoff:
remove the burn and the condvar wake latency reintroduces ≈ the join cost we tried to cut.
So neither end of the tradeoff is shippable: spin = fast-but-burns, block = no-burn-but-no-win.

### Measurement caveat (does NOT change the verdict)

The box ran load-avg ~21 on 16 cores during this campaign, which inflates the
**rayon-baseline absolute** medians (and adds the wide p75 tails). But the conclusion is
**invariant** to this: the magnitude test compares the per-dispatch *delta* (µs) against the
*per-leaf WORK wall* (ms). Even if rayon's true uncontended per-dispatch were a small
fraction of the measured 85–327 µs, the absolute per-leaf saving stays sub-1% of the
ms-scale per-leaf work — the gate cannot move. The spin pool's *relative* ×5–8.5 advantage
likewise persists across the load (both arms ran interleaved under the same load).

---

## Decision

**NULL — STOP at Task 1. Task 2 (gated prototype) NOT built. Task 3 = NULL-confirmation path.**

- No library hot-path code changed: the deliverable is the throwaway micro-bench example +
  this FINDINGS doc. CPU f64 bit-exact parity is **trivially untouched**.
- The fusion gate stays at `core_scaled_threshold(100, cores)` BFS /
  `core_scaled_threshold(130, cores)` subscan (the dpk/c5v anchors). No primitive beats
  rayon's dispatch by enough — *and* by a margin that matters — to move it.
- No new external dependency added (std-only); no `crossbeam` reached for.
- **No manufactured win. No idle-core-burning pool shipped to shave dispatch latency.**

The 260620-e8e contribution is the negative result + a reusable isolated dispatch
micro-bench: any *future* dispatch-model idea (e.g. a fundamentally different fan-out that
also cheapens the WORK term, not just the barrier) can be measured against this baseline
rather than re-derived. The decisive lesson: **the per-leaf cost is WORK-bound, not
dispatch-bound — a cheaper barrier (even an 8× cheaper one) is immaterial.**
