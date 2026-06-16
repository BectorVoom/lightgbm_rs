---
spike: 010
name: histogram-pool-arena
type: standard
validates: "Given the per-tree HistogramPool, when buffers Vec<Vec<f64>> → one flat Vec<f64> arena (stride=hist_len), then per-tree alloc+clone-memcpy vanishes, train is faster, and the build stays bit-exact"
verdict: VALIDATED
related: [011, 002, 003, 005]
tags: [performance, cpu, histogram, allocation, arena, bit-exact, R3]
---

# Spike 010: Histogram-Pool Arena (flatten `buffers: Vec<Vec<f64>>`)

## What This Validates

Given `HistogramPool.buffers: Vec<Vec<f64>>` (`crates/lgbm-treelearner/src/histogram_pool.rs`),
constructed **once per tree** at `learner.rs:816` as `num_leaves` slot buffers,
**when** it is replaced by ONE flat `Vec<f64>` arena (slot `s` = `[s*hist_len, (s+1)*hist_len)`),
**then** the per-tree allocation churn drops and train gets faster — bit-exactly,
since slot values are unchanged.

## Research / Mechanism

The old form `vec![vec![0.0f64; hist_len]; cache_size]` is doubly wasteful per tree:
1. `cache_size` separate heap allocations (one per slot).
2. `vec![template; n]` builds one zeroed template then **clone-memcpys** it
   `cache_size−1` times — real memory traffic (~3MB on the large shape), not the
   "free" calloc zero-pages a single contiguous alloc gets.

A flat `vec![0.0; cache_size*hist_len]` is a single allocation, no clones, and gives
the subtraction trick contiguous parent/child slots. The public slot API
(`buffer`/`buffer_mut`/`get`/`move_`/`reset_map`) is byte-for-byte unchanged, so the
learner is untouched and bit-exactness is automatic.

## How to Run

Isolated per-tree ceiling microbench (permanent `#[ignore]`d in-crate test):
```bash
cargo test -p lgbm-treelearner --release --lib spike010_pool_alloc_ceiling -- --ignored --nocapture
```
End-to-end (the honest confirmation):
```bash
cargo run --release --example bench_train     # in crates/lgbm
```
Bit-exact gates: `cargo test -p lgbm-treelearner --release --lib` (76 tests),
`cargo test -p oracle-harness --release` (incl. `raw_bin_train_parity`).

## Investigation Trail

1. **Triaged hot-path `Vec<Vec<T>>`** (shared with spike 011): the pool buffers are
   the second of the two on the train hot path. Unlike spike 011's parallel
   intermediate, a slot read returns one contiguous slice, so the per-access
   indirection is NOT hot — the cost is the per-tree construction.
2. **Isolated ceiling microbench** (3 process runs, 16 cores) — current vs flat vs
   reused-arena, per "tree", as a fraction of the bench's per-tree wall:

   | shape  | cur `Vec<Vec>` | flat 1-alloc | reused arena | ceiling (cur−reuse) |
   |--------|----------------|--------------|--------------|---------------------|
   | small  | 2.8–5.0µs (1–2%) | ~1µs (0.4%) | 30ns (0%) | ~1–2% of tree |
   | medium | 103–119µs (8–9%) | 5–8µs (0.4–0.6%) | 30ns | **~8–9%** |
   | large  | 492–972µs (11–22%) | 32–50µs (0.7–1.1%) | 30ns | **~11–22%** |

   The `vec![template; n]` clone-memcpy is the bulk of the cur cost; a flat alloc is
   15–20× cheaper in isolation; reuse-across-trees → ~0.
3. **Implemented the flat arena** (Variant A, per-tree lifetime, internal-only change).
   All gates green: 76 treelearner + 41 lgbm + oracle `raw_bin_train_parity` (bit-exact
   vs lib_lightgbm 4.6).
4. **Confirmed end-to-end** (the spike-011 discipline — isolated overstates):

   | size   | BEFORE `Vec<Vec>` | AFTER flat | delta |
   |--------|-------------------|-----------|-------|
   | small  | ~27.0ms           | ~27.9ms   | ~null (noise) |
   | medium | ~134.5ms          | ~132.4ms  | **−1.6%** |
   | large  | ~410ms            | ~392ms    | **−4.3%** |

   Real and consistent (AFTER ≤ BEFORE every run) but well below the 8–22% isolated
   ceiling: the allocator reuses the freed same-size block each tree (hot, already
   page-faulted), so Variant A's per-tree realloc is not fully cold, and the flat
   form's deferred zero-page faults are paid later during `construct`. Only
   reuse-across-trees removes that entirely.

## Results

**Verdict: VALIDATED + SHIPPED (Variant A).** Flattening `buffers: Vec<Vec<f64>>` →
flat `Vec<f64>` arena is bit-exact (slot semantics unchanged), passes every gate, and
gives a measured **~4% large / ~1.6% medium / null small** end-to-end train speedup
for a trivial internal change. An R3 alloc-traffic win in the same family as
spikes 003/004.

### Contrast with spike 011

Both targeted hot-path `Vec<Vec<f64>>`, opposite verdicts — the discriminator is
*how the nested vec is used*:
- **011 (INVALIDATED):** per-thread scratch written once + copied once is already
  optimal; flattening it forced shared-buffer false sharing → regression.
- **010 (VALIDATED):** a `vec![template; n]` clone-memcpy'd structure rebuilt every
  tree is pure waste; flattening removes real allocation/memcpy traffic → win.

### Follow-on (high-value, not done here)

**Reuse the pool across trees** — hoist it from a per-tree local to a learner field,
`reset_map()` (already buffers-untouched) per tree. Microbench ceiling is the full
**8–22%** of per-tree time on medium/large. End-to-end gain is likely less (allocator
already amortizes some of it, per the Variant-A result), so it needs its own measured
spike/quick-task and a small learner refactor — but it is the larger remaining prize.
