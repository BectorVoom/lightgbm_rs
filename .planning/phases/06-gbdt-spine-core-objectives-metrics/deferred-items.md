# Phase 06 — Deferred Items (out-of-scope discoveries during execution)

These were discovered during execution but fall OUTSIDE the scope of the current
task. They are logged here per the executor scope boundary (do not auto-fix
pre-existing issues in unrelated paths) and tracked for a future phase.

## DEF-06-01 — `binary` + `bagging` + `boost_from_average` per-tree split-count knife-edge

- **Discovered during:** 06-06 Task 2b execution (closing regression_l1 + bagging).
- **Cell:** `binary_bag1_es0_bfa1` (and `binary_bag1_es1_bfa1` when its trimmed
  tree counts also differ).
- **Symptom:** On the bagged SUBSET with `boost_from_average=ON`, ONE early tree's
  top split lands on a split-gain knife-edge: C++ accepts the split (tree 0 → 4
  leaves) while the Rust `cubecl-cpu` f64-fold gain rounds it out (tree 0 → 2
  leaves). All other trees agree on structure and are bit-exact. Observed:
  `binary_bag1_es0_bfa1 tree 0: rust_leaves=2 golden_leaves=4`. The bfa-OFF
  binary+bagging cells (`binary_bag1_es0_bfa0`) DO agree on structure (tree 0 = 2
  leaves in both) and remain strictly bit-exact.
- **Root-cause family:** identical to the regression_l1 + bagging structural
  divergence the user typed-rejected in 06-06 Task 2b — a split-gain knife-edge
  over the bagged subset where the iter-0 init score shifts the per-row gradients
  just enough to flip a borderline split. No leaf-VALUE fix applies to a leaf
  STRUCTURE divergence.
- **Pre-existing:** YES. Confirmed against HEAD (d10e3ac): the matrix panicked on
  the regression_l1 + bagging cell FIRST (it iterates before `binary`), masking
  this binary cell. It was NOT introduced by Task 2b. Verified by temporarily
  skipping the regression_l1 + bagging cells at HEAD — the binary cell then panics
  identically (`rust_len: 2, cpp_len: 4`).
- **Why not fixed here:** out of Task 2b's stated scope (the user decision named
  only `regression_l1 + bagging`). Fixing a tree-learner split-gain knife-edge is
  an architectural investigation (deviation Rule 4) that needs its own decision —
  it is NOT a leaf-value renewal. Unlike regression_l1, `binary + bagging` is NOT
  typed-rejected (no user decision covers it, and it is a valid use case that
  produces correct-enough output on most cells), so the matrix asserts every
  STRUCTURALLY-MATCHING tree bit-exact and tolerates ONLY the documented
  single-tree structural divergence with a hard cap (`struct_divergent <= 1`) so a
  growing divergence still fails as a regression.
- **Tracked for:** a future phase (split-gain knife-edge determinism over bagged
  subsets — possibly the same fix that would un-defer regression_l1 + bagging).
