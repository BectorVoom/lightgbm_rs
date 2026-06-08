---
quick_id: 260608-lad
slug: abstract-backend-parallel-findsplit-batched-hist
status: partial
date: 2026-06-08
---

# Quick Task 260608-lad — backend abstraction + batched histogram (+ find_best_split plan)

## Delivered

### Part 1 — backend abstraction (DONE, bit-exact)
Added `Backend::build_leaf_histograms_raw` — the batched per-leaf seam. Builds ALL
features' RAW histograms for a leaf in one call. DEFAULT impl = the per-feature
gather+construct loop (the bit-exact CPU anchor); GPU overrides it. The learner's
`build_leaf_histogram_into` now calls it once per leaf, then host-side
FixHistogram+compact. CPU path byte-identical.

### Part 3 — batched per-leaf GPU histogram (DONE, ~1e-6)
`RocmBackend` overrides `build_leaf_histograms_raw` with ONE batched kernel: a unit
per `(feature, leaf-row)`, f32 atomic-add into the concatenated output. Collapses
the per-feature construct launches → **one launch per leaf**.
- End-to-end GPU train: small 6.16s→**4.61s** (−25%), medium 22.9s→**16.55s** (−28%).
- Default CPU merge gate GREEN; facade-on-GPU (41 tests) passes.

## Part 2 — parallel/batched find_best_split (DESIGNED, NOT YET INTEGRATED)

**Honest finding (measured, R1-style):** a *per-feature* parallel find_best_split is
**launch-bound** — the GPU per-call cost is ~50µs launch vs ~1µs for the ≤256-bin
scan, so parallelizing within one call saves <1µs/call ≈ 0 end-to-end. The lever
that helps is the SAME as for histograms: **batch it** (one launch finds all
features' splits per leaf). Part 3 proved the batching approach works (−28%).

**Why it's deferred (not rushed):** batched find_best_split must integrate into
`scan_leaf_histogram`, which interleaves the spine `find_best_split` with
load-bearing per-feature gates — col-sampler, **parent-splittability (GOSS)**,
interaction constraints, the `this_leaf_splittable` propagation, and the
categorical/monotone/extra-trees branches. A rushed refactor there risks the
**bit-exact CPU merge gate** (the project's core contract). It needs its own careful
pass.

**Design (next task):**
1. `Backend::find_best_splits_batched(buf, per-feature params, scan-mask) -> Vec<SplitInfo>`
   — default impl loops `find_best_split` in feature order (CPU bit-exact);
   GPU kernel = one cube per feature, reads its histogram region from the
   concatenated `buf`, runs the scan, writes its 12-cell SplitInfo. One launch.
2. Refactor `scan_leaf_histogram`: first pass evaluates gates → list of spine
   features to scan; ONE batched call; second pass does splittable/CEGB/penalty/
   records/argmax using the precomputed results (gates + non-spine branches
   unchanged). Preserves feature order ⇒ CPU bit-exact; GPU collapses
   num_features find_best_split launches → 1/leaf.

This is the remaining lever to make the GPU competitive (after it, `subtract` +
`data_partition` per-split launches are the small tail).

## GPU progression (end-to-end train, large-ish)

| stage | small | medium |
|-------|-------|--------|
| single-unit f64 (kfu) | 8.28s | 40.8s |
| + parallel hist (kt8) | 6.16s | 22.9s |
| + batched hist (lad p3) | **4.61s** | **16.6s** |
| (next) + batched find_best_split | — | — |

CPU native stays the fast path (small 38.7ms); the GPU is a separate ~1e-6 track
being made progressively parallel.
