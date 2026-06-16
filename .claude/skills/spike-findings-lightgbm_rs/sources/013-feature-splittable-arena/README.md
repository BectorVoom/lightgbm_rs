---
spike: 013
name: feature-splittable-arena
type: standard
validates: "Given the per-tree feature_splittable = vec![vec![true; nf]; nl] bool matrix, when flattened/reused, then train is faster — or is the magnitude negligible?"
verdict: INVALIDATED
related: [012, 010, 011]
tags: [performance, cpu, allocation, negative-result, not-worth-it]
---

# Spike 013: feature_splittable Arena (the last per-tree `vec![template; n]`)

## What This Validates

`feature_splittable: RefCell<Vec<Vec<bool>>>` is rebuilt every tree as
`vec![vec![true; num_features]; num_leaves]` (`learner.rs:~891`) — the *same*
`vec![template; n]` clone-memcpy pattern that made the histogram pool worth flattening
(spike 010), but on a `[num_leaves][num_features]` bool matrix (~1.5KB for 31×50, vs
the pool's multiple MB). This spike asks whether it's worth flattening/reusing.

## How to Run

```bash
cargo test -p lgbm-treelearner --release --lib spike013_feature_splittable -- --ignored --nocapture
```

Isolated construction cost (3-way: current `Vec<Vec<bool>>` / flat `Vec<bool>` / reused
arena) as a fraction of per-tree train time.

## Investigation Trail

1. **Predicted null up front:** spike 012 showed that reusing a *multi-MB* structure
   across trees yielded only ~3% because the allocator amortizes fixed-size per-tree
   reallocs. A ~1.5KB structure can't beat that. Confirmed with a number rather than a
   refactor (the literal flatten touches the `[leaf][feature]` access type at multiple
   read/write sites — `scan_leaf_histogram`, the subtract gate — disproportionate risk).
2. **Measured** (median of 500, 2 process runs):

   | shape (nl×nf) | cur `Vec<Vec<bool>>` | % of per-tree | flat | reuse | ceiling |
   |---------------|----------------------|---------------|------|-------|---------|
   | small  31×12  | 651–661ns            | **0.25%**     | 30–60ns | 30–50ns | ~0.23% |
   | medium 31×30  | 231–331ns            | **0.02%**     | 20–30ns | 20–30ns | ~0.02% |
   | large  31×50  | 230–331ns            | **0.005–0.008%** | 40–60ns | 20–30ns | ~0.007% |

## Results

**Verdict: INVALIDATED (not worth optimizing).** The construction is **0.005–0.25%**
of per-tree time — below bench noise. The optimization is real in kind (flat/reuse is
15–20× cheaper in isolation, same as the pool) but the absolute magnitude is
negligible, and the literal flatten would change the `[leaf][feature]` access type at
several sites for ≤0.25% best case (small) / ≤0.02% (medium/large). **Keep the
`Vec<Vec<bool>>` as-is.** No production change made.

### Signal for the build

- This closes the "in learning" `Vec<Vec<T>>` sweep: the only ones that mattered were
  the histogram pool (010 flatten + 012 reuse, shipped, ~7% large combined) and the
  rejected parallel-build intermediate (011). Everything else — `feature_splittable`
  (013), `branch_features` / `bynode_selected` (conditional/empty on the default path),
  `interaction_constraints` (config-time) — is cold or sub-noise.
- **Rule of thumb confirmed:** flattening a per-tree `vec![template; n]` only pays when
  the structure is large (MB-scale: pool yes, 1.5KB bool matrix no). Size-gate the
  intuition before refactoring.
