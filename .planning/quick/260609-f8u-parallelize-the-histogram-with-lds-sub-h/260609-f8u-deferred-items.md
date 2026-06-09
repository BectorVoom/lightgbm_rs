---
quick_id: 260609-f8u
type: deferred-items
date: 2026-06-09
---

# Deferred Items — 260609-f8u

## DEF-f8u-01 — `learner_parity_resident_equals_host_tree_on_hip` is FLAKY (pre-existing)

**Status:** pre-existing defect surfaced during f8u; NOT introduced here. Logged for
a future fix.

**Symptom:** under `cargo test -p oracle-harness --features rocm --test learner_parity`,
`hip::learner_parity_resident_equals_host_tree_on_hip` fails intermittently:
```
resident vs host leaf 11: resident=0.7184145450592039 host=0.7184155838830129
abs_diff=0.000001038823808974243 > 0.000001 — the resident chain changed the tree
```

**Verified pre-existing + flaky:** with `RocmBackend::construct_histograms` reverted to
the naive atomic path (master-equivalent — f8u did not change it on the production
path), the test failed **4 of 6 consecutive runs** (runs: FAIL FAIL ok FAIL ok FAIL).

**Root cause:** the test asserts the resident chain and the host path agree to **1e-6**
on every leaf output. Both are GPU f32-atomic paths whose accumulation order is
**nondeterministic** (atomic adds commit in arbitrary order). Leaf 11's output sits on
the 1e-6 knife-edge, so `abs_diff` hovers ~0.9e-6…1.1e-6 and tips over the gate on ~half
the runs. This is f32-atomic nondeterminism at a tolerance boundary, not a tree-structure
change (same leaf, ~1e-6 value wobble). Consistent with the memory note about a
"known-flaky resident cell".

**Why it matters now:** it blocks cleanly verifying whether routing
`construct_histograms` to the new LDS kernel is non-regressive — the baseline itself is
~50% red. Hence the LDS kernel is landed but NOT wired (see FINDINGS).

**Fix options (future):**
1. Make the comparison robust to f32-atomic nondeterminism — e.g. a tie-aware bound at
   leaf-output knife-edges (like commit 1832206 did for `kernel_parity_split` default_left),
   or compare both GPU paths against the f64 anchor within ORACLE_TOL rather than against
   each other at 1e-6. **Do NOT blanket-weaken** — target the knife-edge case explicitly.
2. Make the resident + host GPU build paths share ONE deterministic-enough accumulation
   (the same unification needed to wire LDS live), removing the run-to-run wobble.

## Finding #2 follow-up (not a defect) — LDS-ify the BUILD hot path

`construct_histograms_lds_f32_on` LDS-privatizes only the single-feature primitive. The
training hot path is the batched/resident build kernels
(`build_leaf_histograms_raw` → `construct_leaf_hist_batched_kernel` /
`construct_leaf_hist_resident_kernel` / `build_fix_compact_resident`). Applying LDS
there (concatenated multi-feature layout, per-feature LDS budgeting) + unifying the
accumulation order is what makes the proven 4–4.6× win actually reach GPU training.
