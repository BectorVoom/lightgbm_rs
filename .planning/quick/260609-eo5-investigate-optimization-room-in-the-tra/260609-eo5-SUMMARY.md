---
quick_id: 260609-eo5
title: Investigate optimization room in the traversal algorithm within the kernels
date: 2026-06-09
status: complete
type: investigation
code_changes: none
---

# Quick Task 260609-eo5 — Summary

**Type:** Technical investigation (no code changes).

**Verdict:** The kernel traversals are already correctly split into bit-exact
sequential anchors (untouchable) and parallel GPU paths. Within the parallel paths
there are **two genuine levers** (GPU/`RocmBackend`-only; CPU native path untouched),
both bounded by the established **launch-bound** caveat (wins only on large/compute-bound
data, not the small benches):

1. **[HIGH value / LOW risk] Partition kernel is single-unit sequential on the GPU.**
   `data_partition_kernel` (`partition.rs:53`) runs `if UNIT_POS==0 { for i in 0..bins.len() }`,
   launched `CubeDim 1` — one lane walks every row. Row routing is per-row independent
   and integer-deterministic (no order/parity constraint; the order-dependent stable
   compaction is already host-side), so parallelizing one-unit-per-row (like the atomic
   histogram kernel) carries **zero parity risk**. This is the cleanest lever.

2. **[REAL but CONSTRAINED] Parallel histogram uses naive global atomics.**
   `construct_hist_kernel_atomic_f32` (`histogram.rs:327`) does direct
   `out[bin].fetch_add` into global memory → atomic contention on popular bins. The
   classic fix — local-memory (LDS) privatized sub-histograms + reduce, which
   LightGBM's own OpenCL kernels (`ocl/histogram{16,64,256}.cl`) use — is **absent**
   (no `SharedMemory`/`Plane`/privatization anywhere). Stays within the existing
   ~1e-6 f32 gate, but needs LDS budgeting (256-bin × features in the batched layout)
   and only helps when atomic-contention-bound.

3. **[DO-NOT-TOUCH] Split scan is inherently sequential** — prefix-sum dependency +
   single-owner f64 fold = bit-exact requirement; parallel-scan would break the
   anchor. Its branchless no-early-out adds only a ~1µs constant on a ≤256-bin loop,
   dwarfed by the ~50µs launch. (The redundant FORWARD pass is already eliminated via
   `fwd_count=0` for `missing_type==None`.)

**Recommendation:** if pursuing GPU perf, do #1 first (low-risk partition
parallelization); schedule #2 as a larger bin-count-gated kernel-family project only
after confirming contention dominates (profile — GPU has repeatedly proven
launch-bound). Reconfirm GPU is even the right axis (CPU native is the fast path +
merge gate).

**Deliverable:** `260609-eo5-FINDINGS.md` (per-kernel traversal classification,
ranked opportunities, evidence trail).

**Files modified:** none (investigation only).
