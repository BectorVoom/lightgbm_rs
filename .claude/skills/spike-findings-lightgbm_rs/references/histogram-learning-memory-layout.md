# Histogram & Learning-Path Memory Layout

Implementation blueprint from spikes 010–013 — how to (and how NOT to) optimise
`Vec<Vec<T>>` / per-iteration allocation in the CPU training path, with the numbers and
the bit-exact gates that prove it.

## Requirements (non-negotiable)

- **The CPU f64 anchor stays bit-exact to C++** — every change here is gated by
  `cargo test -p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`) +
  `cargo test -p lgbm-treelearner --lib` + `cargo test -p lgbm`. A speed change that
  moves a single ULP is rejected.
- **`Vec<Vec<T>>` is not categorically a pessimization.** Decide per-instance by *usage*
  (see the decision rule below). Do not blanket-flatten.
- **Ship on the end-to-end `bench_train` number, never the isolated microbench** — the
  cold isolated ceiling overstates the warm end-to-end win 3–7× (allocator amortization).

## The decision rule (the core finding)

For any nested/`vec![template; n]` structure on the hot path, ask how it's used:

| Usage pattern | Verdict | Why |
|---------------|---------|-----|
| `vec![template; n]` rebuilt per tree/leaf, **MB-scale** | **Flatten + reuse** | Clone-memcpy + N allocs are real traffic (010: pool, ~3MB/tree) |
| `vec![template; n]` rebuilt per tree, **KB-scale** | **Leave it** | Sub-noise; refactor risk > gain (013: 1.5KB bool matrix = 0.005–0.25%/tree) |
| Per-thread private accumulator, written-once + merged-once | **Leave it** | Already optimal; flattening forces shared-buffer false sharing (011: regressed 13–21%) |
| Per-leaf row lists | **Already flat** | `DataPartition` is `indices: Vec<u32>` + `leaf_begin`/`leaf_count` offsets (C++-faithful), never `Vec<Vec>` |
| Conditional/empty on the default path | **Ignore** | `branch_features`, `bynode_selected` (inner empty unless interaction/bynode active); `best_cat_threshold` (`None` on numeric spine) |

## How to Build It — the two shipped wins

Both live in `crates/lgbm-treelearner/`. Combined ≈ **~7% large** train time vs the
original jagged pool, fully bit-exact.

### 1. Flat histogram-pool arena (spike 010) — `histogram_pool.rs`

Replace per-slot jagged buffers with one contiguous arena:

```rust
// struct field:
buffers: Vec<f64>,                       // was: Vec<Vec<f64>>
// new():
buffers: vec![0.0f64; cache_size * hist_len],   // was: vec![vec![0.0; hist_len]; cache_size]
// accessors:
pub fn buffer(&self, slot: usize) -> &[f64] {
    let base = slot * self.hist_len;
    &self.buffers[base..base + self.hist_len]
}
pub fn buffer_mut(&mut self, slot: usize) -> &mut [f64] {
    let base = slot * self.hist_len;
    &mut self.buffers[base..base + self.hist_len]
}
```

The public slot API (`buffer`/`buffer_mut`/`get`/`move_`/`reset_map`) is unchanged ⇒
the learner is untouched ⇒ bit-exact by construction. Removes the `vec![template; n]`
clone-memcpy (the real cost) + N allocs. End-to-end ~4% large.

### 2. Reuse the pool across trees (spike 012) — `learner.rs`

The pool was still `HistogramPool::new(...)` **per tree** inside `train_inner`. Hoist it
to a learner field and reuse:

```rust
// struct field:
hist_pool: Option<HistogramPool>,        // new(): None
// top of train_inner (after slot_len is known):
let want_cache = self.num_leaves.max(1) as usize;
let mut pool = match self.hist_pool.take() {
    Some(p) if p.cache_size() == want_cache && p.hist_len() == slot_len => p,
    _ => HistogramPool::new(self.num_leaves, slot_len),
};
pool.reset_map();
// ... existing per-tree growth, all `&mut pool` usage unchanged ...
// before the success return:
self.hist_pool = Some(pool);
```

`train_inner` is `&mut self`, so `take()`/store-back needs no `RefCell`. End-to-end
~3% large *on top of* 010.

**Why it's bit-exact:** `reset_map()` restores a fresh pool's index-map state, and
*every slot is fully overwritten before any read* — directly-built leaves zero+fill the
whole slot (`build_leaf_histogram_into`: `for c in buf.iter_mut() { *c = 0.0 }`),
subtract-derived leaves `copy_from_slice` the whole slot. Cross-tree stale data is never
read; this is the same situation as within-tree slot reuse.

## What to Avoid

- **Don't scatter the parallel build into a shared `out` (spike 011).** The rayon
  per-feature `Vec<Vec<f64>>` accumulators in `build_histograms_into` look flattenable but
  are load-bearing: replacing them with 16 threads folding directly into disjoint slots of
  one shared buffer regressed **13–21%** at the leaf sizes the parallel path runs on
  (false sharing / cache-coherence > the alloc+copy removed). A `NOTE` marks them — leave
  them. Private-accumulator-then-merge beats shared-output scatter.
- **Don't flatten `feature_splittable` (spike 013).** Real `vec![template;n]` pattern but
  ~1.5KB → 0.005–0.25% of per-tree time. The literal flatten changes the `[leaf][feature]`
  access type at multiple sites for sub-noise gain. Keep `Vec<Vec<bool>>`.
- **Don't trust the isolated microbench number.** 010's isolated ceiling was 8–22%/tree;
  end-to-end was ~4%. 012's was the same 8–22%; end-to-end ~3%. The allocator returns the
  same hot, pre-faulted block on a fixed-size per-iteration realloc, so most of the cold
  ceiling is already paid for. **Always confirm with `bench_train`.**

## How to measure (the harness)

1. Isolated, in-crate `#[ignore]`d microbench with `before`/`after` as self-contained
   local closures (needs private fns/layout). Interleave per launch, median of N after
   warmup, `black_box` a sink, **sweep the size** (the win/regression often only shows at
   the shapes the path actually runs on). Run with `--ignored --nocapture`, 2–3 process
   restarts. Reference: `spike010_pool_alloc_ceiling`, `spike011_microbench`,
   `spike013_feature_splittable` (sources/ here).
2. **Then** `cargo run -p lgbm --release --example bench_train` — small/medium/large,
   before vs after (stash the change for the baseline; pre-build to avoid a cold first-run
   outlier; AFTER ≤ BEFORE clusters, ideally non-overlapping).
3. Bit-exact gate before claiming anything.

## Constraints

- Gains are concentrated on **medium/large** shapes (more bins × leaves = more pool
  bytes); small is allocation-noise-dominated and shows ~null.
- These are **CPU-anchor** wins. The ROCm path uses its own resident pool mirror
  (`reset_resident_pool`); these CPU-storage changes are no-ops there.
- This sweep is **complete for the learning leaf path** — no remaining hot per-leaf
  `Vec<Vec<T>>`. Adjacent (out-of-scope-for-Vec<Vec>) lever still open: reuse per-leaf
  flat scan scratch (`gather_info_for_threshold`) R3-style.

## Origin

Synthesized from spikes: 010, 011, 012, 013 (2026-06-17).
Source READMEs + microbenches in: `sources/010-*`, `sources/011-*`, `sources/012-*`,
`sources/013-*`. Shipped commits: `d9cbae4` (010), `5c8fa43` (012/013); 011 reverted with
an in-source `NOTE`.
