---
quick_id: 260609-c2l
title: Investigate whether we can minimize host-device transfers for kernels in the leaf-estimation part
date: 2026-06-09
status: complete
type: investigation
code_changes: none
---

# Quick Task 260609-c2l — Summary

**Type:** Technical investigation (no code changes).

**Verdict: NO — there is essentially nothing to minimize.**

Leaf estimation in this port is **pure host-side f64 scalar math**, not a GPU kernel:
- The Newton-step leaf output `-g/(h+l2)` (`gain.rs:135-148`,
  `calculate_splitted_leaf_output`) runs on the CPU over sums that are already
  host-resident (`leaf_splits.rs:75-122`).
- The renewal path (L1 median / quantile / MAPE) is intrinsically host
  (`regression.rs:627-651`, `learner.rs:3134-3152`).
- The only "in-kernel" leaf output (`left_output`/`right_output`) is already
  embedded in the unavoidable 96-byte `SplitInfo` readback from `find_best_split` —
  **zero marginal transfer**.

There is **no leaf-output kernel** in `lgbm-compute` to optimize. The transfer
question only had teeth in **histogram construction** (a different part), which was
already addressed by the device-resident histogram pool (260608-p90) and proven
**launch-bound, not transfer-bound** (260608-profile: ~200 launches/tree, eliminating
the histogram round-trip gave only a mixed/modest win).

**Recommendation:** Do not pursue transfer minimization in the leaf-estimation path.
If GPU perf is the goal, the established lever is launch-count reduction in
histogram/split — itself at diminishing returns on the current small/launch-bound
benches. Reconsider whether GPU is the right axis given native CPU is already ~2–4×
of C++ and is the bit-exact gate.

**Deliverable:** `260609-c2l-FINDINGS.md` (full report with code map, transfer
inventory, and evidence trail).

**Files modified:** none (investigation only).
