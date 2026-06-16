---
spike: 011
name: parallel-build-scatter
type: standard
validates: "Given the rayon feature-parallel histogram build, when each task folds directly into its disjoint &mut out[slot_off..] slice (split_at_mut) instead of a per-feature Vec<f64> + reassembly copy, then the N allocations + the copy vanish and large-leaf build gets faster, byte-identically"
verdict: INVALIDATED
related: [005, 003, 002]
tags: [performance, cpu, histogram, parallelism, rayon, false-sharing, negative-result]
---

# Spike 011: Parallel-Build Scatter (eliminate the `Vec<Vec<f64>>` intermediate)

## What This Validates

Given the rayon feature-parallel build in `build_histograms_into`
(`crates/lgbm-compute/src/lib.rs`), **when** each feature folds directly into its
disjoint sub-slice of the shared output arena `out` (carved with `split_at_mut`)
instead of allocating its own per-feature `Vec<f64>` and then `copy_from_slice`-ing
it back, **then** the N per-feature heap allocations and the reassembly copy vanish
and the large-leaf build runs faster — while staying byte-identical (disjoint writes,
same fold order).

This was framed as a clean "optimise `Vec<Vec<T>>`" win on the R4 large-leaf BUILD
path (the proven-dominant cost from spikes 002/003).

## Approach

The parallel branch of `build_histograms_into` (spike 005 / R4) currently does:

```rust
let hists: Vec<Vec<f64>> = (0..nfeat).into_par_iter()
    .map(|fpos| { let mut h = vec![0.0; 2*num_bins[fpos]]; fold_one_feature(..,&mut h); h })
    .collect();
for (fpos, h) in hists.into_iter().enumerate() {
    out[slot_off[fpos]..slot_off[fpos]+h.len()].copy_from_slice(&h);   // sequential assembly
}
```

The candidate scatter replaces it with disjoint-slot writes (no intermediate, no copy):

```rust
let mut slots: Vec<&mut [f64]> = /* carve `out` via split_at_mut at slot_off boundaries */;
slots.into_par_iter().enumerate()
    .for_each(|(fpos, slot)| fold_one_feature(feature_bins[fpos], .., slot));
```

Both fold in `leaf_rows` order into disjoint regions of a pre-zeroed buffer ⇒ provably
byte-identical to serial (the existing `build_histograms_parallel_equals_serial` gate
held with the scatter live).

## How to Run

The microbench is a permanent `#[ignore]`d in-crate test (it needs the private
`fold_one_feature` / slot layout), with both strategies as self-contained local
closures so it documents the negative result regardless of which one ships:

```bash
# default 1M-row leaf
cargo test -p lgbm-compute --release --lib spike011_microbench -- --ignored --nocapture
# sweep the leaf size (16384 = the live LGBM_PAR_THRESHOLD)
SPIKE011_ROWS=16384 cargo test -p lgbm-compute --release --lib spike011_microbench -- --ignored --nocapture
```

It interleaves LIVE (Vec<Vec>+copy) and SCATTER per launch to cancel thermal/scheduler
drift, reports the median of 30 launches after 5 warmup, and asserts byte-equality.

## Investigation Trail

1. **Triaged the whole codebase's `Vec<Vec<T>>`.** Most are cold (EFB grouping,
   dataset sampling, predict-path `feature_row`) or already-flat-adjacent — the build
   output is *already* a flat `out` arena. Only two `Vec<Vec<f64>>` sit on the train
   hot path: this parallel intermediate, and `HistogramPool.buffers` (spike 010).
2. **Implemented the scatter** in production, confirmed the bit-exact gate stayed green
   (disjoint writes, same order).
3. **End-to-end `bench_train` was inconclusive** (±12% run-to-run, AFTER even appeared
   *slower* on `large`). At these shapes the parallel build is a tiny slice of train
   time and "forced-parallel" (`LGBM_PAR_THRESHOLD=0`) adds rayon dispatch overhead on
   tiny leaves that swamps the signal. Per the CONVENTIONS bench discipline, dropped to
   an **isolated in-process microbench of the one function**.
4. **Microbench at 1M-row leaf: null (0.998–1.00×).** The build is dominated by ~70ms
   of scattered gather/fold; 50 small allocs (4KB each) + a 200KB sequential copy are
   negligible next to it.
5. **Swept the leaf size — and found a REGRESSION, not a win.** At the leaf sizes where
   the parallel path actually fires (≥16384 rows, the live threshold) the scatter is
   **10–21% SLOWER**, converging to null only on huge leaves:

   | leaf rows | LIVE Vec<Vec>+copy | SCATTER | ratio live/scatter |
   |-----------|--------------------|---------|--------------------|
   | 16384     | 226µs              | 282µs   | **0.80–0.87×**     |
   | 32768     | 515µs              | 573µs   | **0.79–0.90×**     |
   | 131072    | 2.22ms             | 2.61ms  | **0.85×**          |
   | 500000    | 9.85ms             | 9.94ms  | 0.99×              |
   | 1000000   | 70.8ms             | 70.9ms  | 1.00×              |

   Confirmed across 3 process restarts (warmup-drift ruled out).
6. **Reverted the production change.** Added a `NOTE (spike 011, INVALIDATED)` at the
   call site so the `Vec<Vec<f64>>` is not "cleaned up" by a future pass.

## Results

**Verdict: INVALIDATED.** The `Vec<Vec<f64>>` intermediate is **load-bearing, not a
wart.** Each rayon task folding into its **own private, cache-hot buffer** followed by
a single sequential assembly memcpy is *faster* than 16 threads scattering writes into
disjoint regions of one shared `out` arena. The scatter trades a cheap thread-local
accumulate + streaming copy for cross-core cache-coherence / false-sharing traffic on
the shared buffer, and at the threshold leaf sizes (where build time matters and the
fold is short enough that the coherence cost is a real fraction) that costs 13–21%.

This is the classic **private-accumulator-then-merge** pattern beating shared-output
scatter — and it matters *most* precisely on the leaves the parallel path governs.

### Signal for the build

- **Keep the `Vec<Vec<f64>>` per-feature accumulators in `build_histograms_into`.**
  Not all `Vec<Vec<T>>` are pessimizations; thread-private accumulators are the right
  shape for a parallel reduction into a shared buffer.
- The histogram build remains **memory-gather-bound** (consistent with 002/003/006):
  the output histogram alloc/copy is negligible; the row gather is everything.
- General lesson for the "optimise `Vec<Vec<T>>`" idea: flattening helps only where the
  nested vec causes pointer-chasing *on the hot read/write path*. A nested vec used as
  a **per-thread scratch that is written once and copied once** is already optimal.
