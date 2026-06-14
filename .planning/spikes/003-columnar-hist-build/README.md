---
spike: 003
name: columnar-hist-build
type: standard
validates: "Given the histogram-build dominates CPU train, when the redundant per-feature grad/hess gather is hoisted to once-per-leaf, then build wall-clock drops at BOTH low and large rows without regression, bit-exactly"
verdict: VALIDATED
related: [002]
tags: [performance, cpu, histogram, build, bit-exact, gather, treelearner]
---

# Spike 003: Columnar / once-gather histogram build

## What This Validates

**Given** the histogram BUILD dominates CPU train time (spike 002: 62.9% at 2k rows,
93.7% at 200k rows), **when** the ordered gradient/hessian gather is hoisted out of the
per-feature loop to run ONCE per leaf, **then** build wall-clock drops materially at
BOTH low and large rows with no regression — holding the bit-exact f64 fold order.

## The find (root cause the quick task missed)

`Backend::build_leaf_histograms_raw` (the CPU f64 anchor build) gathered THREE arrays
per feature in its inner loop:
```
for each feature f:
  for each leaf row r:
    ord_bins.push(bins_f[r]); ord_g.push(grad[r]); ord_h.push(hess[r])
  construct_histograms(ord_bins, ord_g, ord_h, ...)
```
But `ord_g`/`ord_h` are **identical across every feature** — only the bin column
differs. The per-feature re-gather repeated the grad/hess gather `num_features` times
(12× at small, 32× at large). C++ avoids this with `ordered_gradients_`/
`ordered_hessians_` gathered once per leaf.

**Fix:** gather `ord_g`/`ord_h` once per leaf before the feature loop; re-gather only
`ord_bins` per feature. Values and fold order are byte-identical ⇒ bit-exact. ~4 lines.

This is distinct from quick-260614-p0n, which chased the *allocation* churn (fold-in-
place) — disproven (no low-row win, −9% large regression). The real cost was the
redundant *gather memory traffic*, not allocation.

## How to Run

```bash
LGBM_PHASE_PROF=1 BENCH_SIZES="small:2000:12:32"        BENCH_ITERS=100 BENCH_REPS=9 cargo run --release --example bench_crossover
LGBM_PHASE_PROF=1 BENCH_SIZES="large:200000:32:64"      BENCH_ITERS=50  BENCH_REPS=5 cargo run --release --example bench_crossover
```

## Results

**VERDICT: VALIDATED.** Big win at both scales, bit-exact, no regression. Confirmed
across 2 interleaved A/B rounds (honoring the p0n lesson: validate wall-clock, not
instruction counts).

| size | metric | baseline | once-gather | Δ |
|------|--------|----------|-------------|---|
| small 2k×12 | build (ms/900it) | 205.8 | 138.5 | **−33%** |
| small 2k×12 | train_median | 39.6–40.4 ms | 32.6–33.9 ms | **−16…−18%** |
| large 200k×32 | build (ms/250it) | 17123 | 10424 | **−39%** |
| large 200k×32 | train_median | 4.25–4.30 s | 2.89–2.90 s | **−32…−33%** |

rows/s: small 50k→59–61k, large 47k→69k.

### Bit-exact merge gate — PASS
`cargo test -p lgbm-compute --lib` 21/0; `-p lgbm-treelearner` 65/0; `-p oracle-harness
--test learner_parity` 29/0 (bit-exact); full `-p oracle-harness` every suite 0-failed/
0-ignored. f64 fold order frozen; clippy clean; GPU/rocm path untouched (RocmBackend
overrides this default impl).

### Why large improves (vs p0n regressing it)
p0n folded into the 32KB multi-feature `out` buffer → cache scatter → −9% at large.
This change touches only the gather, keeping the existing tight per-feature
`construct_histograms` + streaming copy — so it *removes* memory traffic everywhere and
helps large MORE (32 features × redundant gather eliminated).

## Signal for the Build

- **SHIP IT** — committed as the R3 perf win (the lever quick-260614-p0n was looking
  for). One-leaf grad/hess gather is the highest-ROI, lowest-risk histogram-build fix.
- Remaining headroom: build is still 52.9% (small) / 90.2% (large) of train. Further
  levers — columnar uint8 bin storage (denser bin column read), the `accumulate_
  histogram_into` per-feature hot-scratch fold to drop the per-feature alloc *without*
  the big-buffer cache penalty, subtraction-trick reuse — are now the next candidates,
  each to be A/B'd at both scales before committing.
- Closes part of [[perf-gap-vs-cpp-40-80x]] R3; the low-row gap (spike 002) shrinks
  from 5.2× toward parity (build −33% at small).
