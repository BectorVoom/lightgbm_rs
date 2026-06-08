---
quick_id: 260609-c2l
type: technical-investigation
title: Minimizing host-device transfers for kernels in the leaf-estimation path
date: 2026-06-09
status: complete
verdict: NO-OP — leaf estimation is already host-resident scalar math; no transfer to minimize
---

# Investigation: Host-Device Transfers in the Leaf-Estimation Path

## Question

Can we minimize host↔device transfers for the **kernels** involved in LightGBM's
**leaf-estimation** part?

## Verdict (TL;DR)

**No — there is essentially nothing to minimize.** Leaf estimation in this port is
**pure host-side f64 scalar math**, not a GPU kernel. The leaf output value
(`-g/(h+l2)`, plus the L1/quantile/MAPE renewal medians) is computed on the CPU
from sums that are **already host-resident**. The only "in-kernel" leaf output is
the `left_output`/`right_output` pair computed *inside* `find_best_split` — and it
is already piggybacked onto the existing 96-byte `SplitInfo` readback at **zero
marginal transfer cost**.

The transfer question only ever had teeth in **histogram construction**, which is a
*different* part of the tree learner. That was already addressed (device-resident
histogram pool, task 260608-p90) and — critically — proven **NOT** to be the
bottleneck: the GPU path is **launch-bound, not transfer-bound** (profile 260608-profile).

This finding is consistent with, and narrows, the prior nn7→oib→p90→s2b→t3t thread.

## What "leaf estimation" maps to in this codebase

Two distinct operations, both **host-only**:

1. **Newton-step leaf output** — `calculate_splitted_leaf_output(use_l1, sum_g, sum_h, l1, l2)`
   - `crates/lgbm-compute/src/gain.rs:135-148`
   - Formula: `use_l1 ? -ThresholdL1(g,l1)/(h+l2) : -g/(h+l2)`
   - Marked `#[cube]` so it *can* run in a kernel, but in the leaf path it is called
     as a **plain host function** (see call sites below).

2. **Leaf-output renewal** (regression_l1 median / quantile percentile / MAPE weighted median)
   - `crates/lgbm-objective/src/regression.rs:627-651` (`renew_leaf_output`)
   - Driven host-side by `renew_tree_output` (`crates/lgbm-treelearner/src/learner.rs:3134-3152`),
     a pure-Rust closure dispatch over `(leaf_id, row_indices) -> f64`.
   - Inputs are residuals `label - score`, computed on host
     (`crates/lgbm-boosting/src/gbdt.rs:735-736`).

## Where the leaf-estimation inputs come from (the data-locality crux)

`LeafSplits` (`crates/lgbm-treelearner/src/leaf_splits.rs:34-52`) holds
`sum_gradients`, `sum_hessians`, `weight` — **all host-side f64 scalars**. Its
sums arrive by one of two paths, neither of which incurs a leaf-specific transfer:

- **Ordered host fold** (`leaf_splits.rs:75-96`): strictly sequential left-to-right
  f64 widening fold over the leaf's rows of the **host-native** `gradients`/`hessians`
  arrays. This is the bit-exact deterministic anchor — it MUST stay a host sequential
  fold, and the gradients it reads already live on host.
- **Subtraction-trick seed** (`leaf_splits.rs:105-122`, `init_from_sums`): sums are
  taken directly from the `SplitInfo` already returned by `find_best_split` — i.e.
  from data that **already crossed back to host** as part of the unavoidable split
  readback. No additional transfer.

Either way, `weight = calculate_splitted_leaf_output(...)` runs on the CPU over
scalars that are already in host memory.

## Transfer inventory around the kernel-bearing parts (for completeness)

These are the *only* kernels with host↔device traffic, and none of them is "leaf
estimation":

| Kernel / Backend method | Upload (host→dev) | Readback (dev→host) |
|---|---|---|
| `construct_histograms` / batched / resident | binned u32 (one-time resident), per-leaf grad/hess f32, leaf_rows u32 | f64/f32 histogram — **eliminated on resident path** |
| `subtract_histograms` (resident) | — (Handles on-device) | — (Handle stays resident) |
| `find_best_split[s]` | f64 histogram (or resident Handle) | **`SplitInfo` only** — 12 f64 = 96 B/feature (incl. left/right leaf output) |
| `data_partition` | bin indices u32 | per-row route u32 |

Resident state held in `RocmBackend` across calls: binned feature columns
(uploaded **once per `train()`**) and per-leaf histogram `Handle`s (resident pool).
Per-tree gradients/hessians are uploaded **once per tree** (they change every
iteration, so this is inherent), then only small per-leaf sub-slices move.

Refs: `crates/lgbm-compute/src/lib.rs` (Backend trait + RocmBackend, methods at
~69/124/156/178/205/268/332-471), `crates/lgbm-treelearner/src/resident_pool.rs`,
`crates/lgbm-compute/src/kernels/{histogram,split,subtract,partition}.rs`.

## Why minimizing transfers here would not help

1. **Leaf estimation isn't a kernel.** There is no leaf-output kernel in the
   backend (confirmed: no leaf-output method in `lgbm-compute`). The scalar Newton
   step on host is ~free; moving it to the device would *add* a launch + a readback,
   not remove one.
2. **The split-time leaf output is already free.** `left_output`/`right_output` are
   computed inside `find_best_split` and ride the existing 96-byte `SplitInfo`
   readback that has to happen regardless.
3. **The renewal path is intrinsically host.** L1/quantile/MAPE renewal needs a
   median/percentile of residuals, computed from host-side `label - score`. There is
   no device residency to preserve.
4. **The real cost is elsewhere and is launch-count, not transfer.** Prior profiling
   (260608-profile, gfx1100): ~202–208 launches/tree tracking `num_leaves`, ~50µs
   each; eliminating the per-leaf histogram round-trip (260608-p90) gave only a
   mixed/modest win (medium ~+10%, large ~flat, small −13% before size-gating). The
   bench workloads are small/launch-bound.

## Recommendation

**Do not pursue transfer minimization in the leaf-estimation path** — it is already
host-resident and minimal. If GPU performance is the goal, the established lever is
**launch-count reduction** in histogram/split, and even that is at diminishing
returns against the current small/launch-bound benches (see the nn7→t3t thread).
Before more GPU micro-optimization, reconsider whether GPU is the right axis at all:
the native CPU backend is already ~2–4× of C++ and is the bit-exact merge gate
(see `perf-gap-vs-cpp-40-80x`).

One adjacent, larger structural transfer exists but is **out of scope and not
recommended**: the per-tree host→device upload of freshly computed gradients/hessians
(O(num_data) f32×2, once per tree). Removing it would require running the objective
on-device and keeping scores resident — a large restructure with f32-parity risk,
for a transfer that is cheap relative to the ~200 per-tree launches. It is also not
"leaf estimation."

## Evidence trail

- `crates/lgbm-compute/src/gain.rs:135-148` — leaf-output formula (`#[cube]`, used as host fn)
- `crates/lgbm-treelearner/src/leaf_splits.rs:75-122` — host ordered fold + subtraction seed
- `crates/lgbm-treelearner/src/learner.rs:784-790, 3134-3152` — root-leaf seed + renew dispatch
- `crates/lgbm-objective/src/regression.rs:627-651` — host median/percentile renewal
- `crates/lgbm-boosting/src/gbdt.rs:582-587, 715-742` — host gradient compute + renew gating
- `crates/lgbm-compute/src/lib.rs` (Backend trait + RocmBackend) — transfer seam
- `crates/lgbm-treelearner/src/resident_pool.rs` — device-resident histogram pool (already done)
- Memory: `l3-on-gpu-fixhistogram-deferred` (launch-bound finding), `perf-gap-vs-cpp-40-80x`
