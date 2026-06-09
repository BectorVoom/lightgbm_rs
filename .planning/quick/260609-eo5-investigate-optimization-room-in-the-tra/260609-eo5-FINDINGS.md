---
quick_id: 260609-eo5
type: technical-investigation
title: Optimization room in the traversal algorithm within the kernels
date: 2026-06-09
status: complete
verdict: Two real GPU-path levers (partition parallelization = low-risk; histogram LDS privatization = constrained); split scan is inherently sequential. All gated by launch-bound reality.
---

# Investigation: Traversal-Algorithm Optimization Room in the Kernels

## Scope

"Kernels" = the cubecl `#[cube]` kernels in `crates/lgbm-compute/src/kernels/`
(`histogram.rs`, `split.rs`, `partition.rs`, `subtract.rs`). There is **no
prediction / tree-walk kernel** (predict is host-side), so "traversal within the
kernels" means the inner loops of these four kernels. The GPU path is `RocmBackend`;
`CpuBackend` uses native Rust loops (the bit-exact merge gate) and is unaffected by
everything below.

## Verdict (TL;DR)

The traversal algorithms are **already correctly bifurcated** into (a) bit-exact
sequential anchors that *cannot* be parallelized without breaking the merge gate,
and (b) parallel GPU paths. Within the parallel GPU paths there are **two genuine
traversal-optimization opportunities**, ranked:

1. **[HIGH value / LOW risk] The partition kernel runs single-unit sequential on the
   GPU** even though row routing is embarrassingly parallel and order-independent.
   Trivially parallelizable with zero parity risk.
2. **[REAL but CONSTRAINED] The parallel histogram kernel uses naive global atomics**
   — no local-memory (LDS) privatized sub-histograms, the classic GPU-histogram
   optimization that LightGBM's *own* OpenCL kernels use. Helps only when
   atomic-contention-bound (large data); needs LDS budgeting.

A third traversal (the split scan) is **inherently sequential** (prefix dependency +
single-owner f64 fold) and should not be touched.

**Overarching caveat (do not skip):** prior profiling established the GPU is
**launch-bound** on the current bench workloads (~200 launches/tree, ~50µs each;
[[l3-on-gpu-fixhistogram-deferred]]). These are *traversal/compute* wins, so they
only materialize on **large/compute-bound** data, not the small launch-bound benches.
The CPU native path (already ~2–4× of C++, [[perf-gap-vs-cpp-40-80x]]) is the fast
path and the bit-exact gate; none of this touches it.

---

## Finding 1 — Partition kernel is single-unit sequential on the GPU [HIGH / LOW-RISK]

**Code:** `crates/lgbm-compute/src/kernels/partition.rs:53` (`data_partition_kernel`)

```rust
if UNIT_POS == 0 {
    ...
    for i in 0..bins.len() {          // ALL rows, ONE lane
        let bin = bins[i] as i32;
        let is_default = bin < min_bin || bin > max_bin;
        let gt = bin > th;
        let go_right = select(is_default, default_to_right, gt);
        route[i] = select(go_right, 1u32, 0u32);
    }
}
```

Launched at `partition.rs:239` with `CubeCount::Static(1,1,1), CubeDim::new_1d(1)` —
**a single lane walks every row in the leaf.** `RocmBackend::data_partition`
(`lib.rs:763`) dispatches to this (`data_partition_on`).

**Why this is the cleanest lever:**
- The route computation is **per-row independent**: `route[i] = f(bins[i])` depends
  only on row `i`'s bin and the (scalar) split threshold/min/max/most_freq_bin. No
  prefix dependency, no cross-row carry.
- It is **integer and order-free** — `route[i]` is deterministic regardless of
  evaluation order, so parallelizing carries **ZERO parity risk** (unlike the f64
  histogram fold, which needs ascending row order for bit-exactness).
- The genuinely order-dependent part — the *stable* compaction of `route` into
  contiguous left/right index runs (C++ `DataPartition::Split`) — is **already done
  on the host** after readback; only the route *classification* is in-kernel.

**The fix (out of scope here):** rewrite as one-unit-per-row using `ABSOLUTE_POS`
with an `idx < bins.len()` bounds guard — structurally identical to the existing
`construct_hist_kernel_atomic_f32` (`histogram.rs:333-340`), launched over
`ceil(n/256)` cubes of 256. No atomics needed (each unit writes its own `route[i]`).

**Impact:** removes a fully-serial O(rows) GPU traversal that runs ~once per split
(≈ summed-over-depth ≈ num_data × tree_depth single-lane iterations per tree). On
large data this is pure waste on one lane; this is exactly the class of single-unit
kernel that made the GPU train "~214× slower as-built" ([[perf-gap-vs-cpp-40-80x]],
kfu). Low-risk because parity is unaffected.

---

## Finding 2 — Parallel histogram uses naive global atomics, no LDS privatization [REAL / CONSTRAINED]

**Code:** `crates/lgbm-compute/src/kernels/histogram.rs:327` (`construct_hist_kernel_atomic_f32`)

```rust
let idx = ABSOLUTE_POS;
if idx < binned.len() {
    let ti = binned[idx] as usize * 2;
    out[ti].fetch_add(grad[idx]);      // DIRECT global-memory atomic
    out[ti + 1].fetch_add(hess[idx]);
}
```

This is the **naive** GPU histogram: every row across all lanes/workgroups issues an
atomic `fetch_add` straight into the *global* output histogram. When many rows share
a bin (the normal case — binning concentrates mass), those atomics **serialize on
contention**. The same pattern is used by the batched/resident variants
(`construct_leaf_hist_batched_kernel:413`, `construct_leaf_hist_resident_kernel:511`).

**What's missing:** the standard GPU-histogram optimization is **privatized
sub-histograms in local memory (LDS)** — each workgroup accumulates into its own LDS
copy (often several copies to spread bank conflicts), then reduces the copies into
global memory once. LightGBM's *own* OpenCL kernels do exactly this
(`LightGBM/src/treelearner/ocl/histogram{16,64,256}.cl`). Confirmed absent here: a
grep for `SharedMemory`/`Plane`/`sub-hist`/`privat`/LDS across all kernels returns
nothing — every kernel is atomic-scatter or single-owner sequential
(`split.rs:34` even notes "no Plane dependency").

**Parity:** feasible within the existing contract. The f32-atomic path is *already*
nondeterministic-order / ~1e-6 (not bit-exact); privatize-then-reduce keeps it f32
and within the **same ~1e-6 ROCm gate** ([[l3-on-gpu-fixhistogram-deferred]] design).
The CPU f64 ordered anchor is untouched.

**Constraints (why this is "constrained", not "do it"):**
- **LDS budget.** A 256-bin feature needs `256 × 2 × 4B = 2 KiB` per sub-histogram
  copy; the *batched* kernel concatenates many features per launch, so a full
  per-workgroup privatization may exceed the ~64 KiB LDS on gfx1100 — would need
  per-feature tiling or a bin-count-gated kernel family (mirroring LightGBM's
  16/64/256 split).
- **Launch-bound reality.** Contention only bites when the kernel is
  *compute/atomic-bound* — i.e. large data. On the small launch-bound benches this is
  ~0 gain (consistent with the nn7→t3t thread's diminishing returns).
- **Complexity vs the banked wins.** This is a real kernel-family rewrite with
  bank-conflict tuning; weigh against the fact that GPU is not currently the fast
  path at all.

---

## Finding 3 — Split scan: inherently sequential, leave it [DO-NOT-TOUCH]

**Code:** `crates/lgbm-compute/src/kernels/split.rs:144` (`split_scan_body`), the
single source of truth, launched one-cube-per-feature at `CubeDim::new_1d(1)`.

- The scan is a **prefix accumulation** over bins (running `sum_right_*` / `sum_left_*`),
  so each bin depends on all prior bins — sequential by nature. A parallel
  prefix-scan (Blelloch) exists in theory but would **break the single-owner f64 fold
  ordering** that the bit-exact anchor requires (the same fundamental tension
  documented in [[l3-on-gpu-fixhistogram-deferred]]: bit-exact ⟹ single-owner
  sequential f64).
- The loop is **branchless with no early-out** (`split.rs:270` "Always COMPUTE the
  gain… then gate"): unlike C++'s `break`, it iterates all `num_bin-1` bins even
  after the monotone `done` gate trips. This is a forced consequence of cubecl-cpu's
  MLIR lowering rejecting nested-if mutation. It adds a **constant factor on a
  ≤256-iteration single loop (~1µs)** — utterly dwarfed by the ~50µs launch. Not
  worth restructuring, and any "early break" reintroduces the lowering hazard.
- **Already-optimized sub-point:** the redundant FORWARD pass is **not** run for the
  common `missing_type==None` case — the launcher sets `fwd_count = 0`
  (`split.rs:791-795`) so the FORWARD loop iterates zero times. No waste there.

---

## What is correctly NOT a problem

- **`construct_hist_kernel` (f64 single-owner fold, `histogram.rs:55`, `CubeDim 1`)** —
  this is the deterministic bit-exact *anchor* by design; its sequential traversal is
  mandatory, and the GPU hot path already uses the *parallel* f32-atomic kernel
  instead (`RocmBackend::construct_histograms` → `..._parallel_f32_on`, `lib.rs:719`).
- **`subtract_histograms`** — O(bins) elementwise parent−child; trivial, no traversal
  structure to optimize.
- **CpuBackend** — native loops (`construct_histograms_cpu_native`,
  `data_partition_cpu_native`, `find_best_split_cpu_native`); the fast path + merge
  gate, intentionally not kernels.

---

## Recommendation

If GPU performance is being pursued, **do Finding 1 first** (parallelize the
partition kernel one-unit-per-row): it is low-risk (zero parity impact, integer
order-free), small, and removes a genuinely serial GPU traversal. Treat **Finding 2**
(LDS-privatized histogram) as a larger, bin-count-gated kernel-family project to
schedule only if large-data GPU training becomes the target and after confirming
atomic contention dominates (profile first — the GPU has repeatedly proven
launch-bound, not compute-bound, on these benches). **Finding 3** should be left
as-is.

Bigger picture: every lever here is GPU-path-only and bounded by the launch-bound
finding. Before investing, reconfirm whether GPU is the right axis at all — the CPU
native backend is already the fast path and the bit-exact merge gate.

## Evidence trail

- `crates/lgbm-compute/src/kernels/partition.rs:43-76,194-242` — single-unit partition kernel + launcher
- `crates/lgbm-compute/src/kernels/histogram.rs:55-83` — f64 single-owner fold (anchor)
- `crates/lgbm-compute/src/kernels/histogram.rs:314-393,413,511` — naive global-atomic parallel histograms
- `crates/lgbm-compute/src/kernels/split.rs:142-371,775-836` — sequential scan body + `fwd_count=0` dispatch
- `crates/lgbm-compute/src/lib.rs:707-790,836-891` — RocmBackend → kernel wiring
- `crates/lgbm-compute/src/lib.rs:508-594` — CpuBackend → native loops (unaffected)
- `LightGBM/src/treelearner/ocl/histogram{16,64,256}.cl` — reference LDS-privatized histograms
- Memory: [[l3-on-gpu-fixhistogram-deferred]] (launch-bound; bit-exact⟹sequential tension), [[perf-gap-vs-cpp-40-80x]]
