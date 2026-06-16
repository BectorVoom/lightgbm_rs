---
spike: 012
name: reuse-pool-across-trees
type: standard
validates: "Given the HistogramPool is rebuilt per tree, when it is hoisted to a learner field and reused across all trees in a train (alloc once, reset_map per tree), then the per-tree alloc+zero+page-fault vanishes, train is faster, and output stays bit-exact"
verdict: VALIDATED
related: [010, 011, 002]
tags: [performance, cpu, histogram, allocation, reuse, bit-exact, R3]
---

# Spike 012: Reuse the Histogram Pool Across Trees

## What This Validates

The explicit follow-on flagged by spike 010. Given `HistogramPool` was still
allocated **per tree** at the top of `train_inner` (`learner.rs`), even after 010
flattened its buffers to one arena, **when** the pool is hoisted to a learner field
(`hist_pool: Option<HistogramPool>`) and reused across every tree in a train —
`take()` at the top, `reset_map()`, use, store back — **then** the per-tree
allocation + zeroing + first-touch page faults are paid once per *train* instead of
once per *tree*, and the tree is bit-exact.

## Why It's Safe (bit-exact)

`reset_map()` clears only the leaf→slot index maps (mapper / inverse_mapper / LRU),
exactly a fresh pool's state. The buffers carry the previous tree's contents, but
**every slot is fully overwritten before any read**:
- directly-built leaves zero the whole slot then fill it (`build_leaf_histogram_into`
  line 1706: `for c in buf.iter_mut() { *c = 0.0 }`),
- subtract-derived leaves `copy_from_slice` the whole slot (`larger = parent − smaller`).

This is the *same* situation as within-tree slot reuse (slots are evicted + reused
with stale data inside a single tree already — the pool docstring states "the buffer
is NOT zeroed; the learner overwrites it"). So cross-tree reuse changes nothing
observable. Confirmed by the oracle `raw_bin_train_matches_cpp_golden` gate.

## Implementation

Three edits in `crates/lgbm-treelearner/src/learner.rs`:
1. Field `hist_pool: Option<HistogramPool>` (lazily built on the first tree, survives
   `with_features`).
2. At the pool site: `take()` the cached pool if its geometry matches (`cache_size ==
   num_leaves`, `hist_len == slot_len` — always true within a train), else build fresh.
3. Before the success return: `self.hist_pool = Some(pool)`. Error early-returns drop
   the pool (fresh next time) — correctness preserved.

`train_inner` is `&mut self`, so `take()`/store-back needs no `RefCell` and leaves all
existing `&mut pool` local usage untouched.

## How to Run

```bash
cargo test -p lgbm-treelearner --release --lib          # 76 tests
cargo test -p oracle-harness --release                  # incl. raw_bin_train_matches_cpp_golden
cargo run   -p lgbm --release --example bench_train
```

## Investigation Trail

1. **Confirmed the slot-overwrite contract** before touching anything — read
   `build_leaf_histogram_into` (full zero+fill) and the subtract path (full
   copy_from_slice). Cross-tree stale data is never read ⇒ reuse is bit-exact.
2. **Implemented** the `take`/reuse/store-back. All gates green: 76 treelearner + 41
   lgbm + oracle `raw_bin_train_matches_cpp_golden` (bit-exact vs C++).
3. **Benched the INCREMENTAL win on top of spike-010** (the per-tree flat arena is the
   baseline, already on master):

   | size   | BEFORE per-tree flat (010) | AFTER reuse (012) | incremental |
   |--------|----------------------------|-------------------|-------------|
   | small  | ~28ms                      | ~27ms             | ~null (noise) |
   | medium | ~135.0ms                   | ~133.0ms          | ~−1.5% (noisy) |
   | large  | ~410.1ms (cluster 406–410) | ~397.2ms (cluster 392–400) | **−3.1%** (non-overlapping clusters) |

   3 clean process runs each (cold-rebuild first-run outlier discarded).

## Results

**Verdict: VALIDATED + SHIPPED, modest.** Reusing the pool across trees is bit-exact,
passes the C++ golden gate, removes per-tree pool allocation entirely (architecturally
cleaner — one alloc per train), and gives a real **~3% large** incremental train
speedup on top of spike 010.

**The win is well below the 8–22% isolated ceiling** — the spike-010 lesson, now
quantified: the per-tree flat-arena realloc (010) already returned the *same hot,
already-page-faulted block* from the allocator each tree, so most of the theoretical
ceiling was captured by 010's single-alloc flatten. Reuse only removes the residual
(allocator bookkeeping + the rare cold fault), worth ~3% on large. The isolated
microbench's "cur−reuse = 8–22%" measured a COLD pool each iteration; the live train
is warm.

### Combined 010 + 012

vs the original jagged `Vec<Vec<f64>>` per-tree pool: ~4% (010) + ~3% (012) ≈ **~7%
large**, fully bit-exact, no parity risk. An R3 allocation-traffic win in the family of
003/004.

### Cross-reference

Same idea as the rejected reuse-vs-realloc tension in 010/011: the allocator is a far
better amortizer of fixed-size per-iteration buffers than intuition suggests. Always
confirm end-to-end (warm) — the cold isolated ceiling overstates 3–7×.
