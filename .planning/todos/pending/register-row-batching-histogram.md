---
title: Register row-batching in the histogram kernel (NUM_DATA_PER_THREAD analog)
date: 2026-06-15
priority: medium
type: todo
context: /gsd-explore "Compare learning kernel in C++ hip and cubecl, then optimise cubecl kernel"
---

# Todo: register row-batching in the histogram kernel

LightGBM's ROCm histogram kernel folds `NUM_DATA_PER_THREAD = 400` rows per thread before
touching shared memory. Our `construct_leaf_hist_resident_lds_kernel` does **one row per loop
iteration**, hitting an LDS atomic on every row.

**Change:** have each lane accumulate K rows into registers (per-bin partial in private regs, or
just batch the loop so consecutive same-bin rows coalesce) before issuing the LDS `fetch_add` —
cutting LDS-atomic frequency ~K×.

- Low-risk, local to the kernel; compounds with the row-partition spike.
- **Do this AFTER** the row-partitioning spike ([[row-partitioned-histogram-build]]) lands —
  the batch factor K interacts with the row-partition stride, so tune them together.
- Parity: f32-atomic order already non-deterministic; stays in the ~1e-6 gate. Re-run the
  existing kernel-parity harness.

Reference: [[cubecl-vs-rocm-histogram-kernel-comparison]].
