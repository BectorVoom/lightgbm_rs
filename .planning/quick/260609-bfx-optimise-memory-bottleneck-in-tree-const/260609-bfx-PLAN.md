---
phase: quick-260609-bfx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-treelearner/src/learner.rs
autonomous: true
requirements: [QUICK-260609-bfx]
must_haves:
  truths:
    - "Tree construction performs fewer heap allocations per leaf build than before (one fewer Vec allocation per feature per directly-built leaf)."
    - "The CPU f64-fold spine remains BIT-EXACT to the real lib_lightgbm 4.6 goldens (learner_parity unchanged)."
    - "Full workspace parity suite stays GREEN — no regression from the allocation change."
  artifacts:
    - path: "crates/lgbm-treelearner/src/learner.rs"
      provides: "build_leaf_histogram_into with in-place fix+compact (no per-feature to_vec)"
      contains: "fn build_leaf_histogram_into"
  key_links:
    - from: "build_leaf_histogram_into per-feature loop"
      to: "fix_histogram + compact_histogram"
      via: "in-place &mut slice of the owned raw buffer (no clone)"
      pattern: "fix_histogram\\(&mut raw\\[|fix_histogram\\(slice"
---

<objective>
Optimise the memory bottleneck in tree construction by eliminating the per-feature
heap allocation in the directly-built-leaf histogram path, WITHOUT touching
accumulation order or precision (numerical parity is the non-negotiable contract).

This task is intentionally narrow: ONE measured, lowest-risk, parity-neutral
allocation elimination — not a refactor of the pool/grad/hess buffers (those were
already addressed by quick task 260609-8go, which moved rather than cloned the
per-iteration grad/hess/score snapshots).

Purpose: reduce per-leaf allocation churn in the CPU spine hot path
(`build_leaf_histogram_into`), which runs `num_features` times per directly-built
leaf, for every leaf, every tree, every boosting iteration.

Output: an edited `build_leaf_histogram_into` that runs `fix_histogram` +
`compact_histogram` IN PLACE on a `&mut` sub-slice of the already-owned `raw`
buffer the backend returns, removing the per-feature `to_vec()` clone at
`learner.rs:1647` — byte-identical ops in byte-identical order.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@crates/lgbm/examples/bench_train.rs
@crates/lgbm-treelearner/src/learner.rs
@crates/lgbm-treelearner/src/fix_histogram.rs
@.planning/quick/260609-8go-reuse-per-iteration-grad-hessian-score-b/260609-8go-SUMMARY.md

## Identified allocation site (verify before relying on this)

`crates/lgbm-treelearner/src/learner.rs`, `fn build_leaf_histogram_into`
(declared at line 1601), per-feature loop at lines 1645-1655:

```
for (fpos, f) in features.iter().enumerate() {
    let cells = 2 * f.num_bin as usize;
    let mut hist = raw[slot_off[fpos]..slot_off[fpos] + cells].to_vec();   // <-- line 1647: per-feature clone
    fix_histogram(&mut hist, f.most_freq_bin, sum_g, sum_h);
    compact_histogram(&mut hist, f.offset);
    buf[slot_off[fpos]..slot_off[fpos] + cells].copy_from_slice(&hist);
}
```

- `raw` (line 1629) is a freshly-allocated owned `Vec<f64>` returned by
  `backend.build_leaf_histograms_raw(...)`. The learner OWNS it; nothing else
  reads it after this loop.
- `fix_histogram(hist: &mut [f64], ...)` (`fix_histogram.rs:50`) and
  `compact_histogram(hist: &mut [f64], offset: i32)` (`learner.rs:3150`) both
  mutate their slice IN PLACE.
- Therefore the per-feature `to_vec()` clone is redundant: the same two ops can run
  directly on `&mut raw[slot_off[fpos] .. slot_off[fpos] + cells]`, then the (now
  fixed+compacted) sub-slice is copied into `buf`. Same values, same order, same
  storage type (f64) — only the intermediate clone is removed.

`build_leaf_histogram_into` is called from learner.rs:1127, 1434, 1528, 1546 —
i.e. the directly-built smaller-child / root / audit / fallback paths, every tree
growth. The per-feature loop therefore allocates `num_features` short-lived Vecs
per directly-built leaf.

NOTE (do NOT touch): the `feature_bins`/`num_bins` collects at 1627-1628 and the
resident-path collects at 1693-1694 are `num_features`-sized metadata Vecs (slice
refs / u32), not the dominant churn, and feed the Backend signature — leave them.
The grad/hess/score per-iteration buffers were already optimised in 260609-8go;
do NOT redo that.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Measure & confirm the dominant per-leaf allocation site</name>
  <files>crates/lgbm-treelearner/src/learner.rs (read-only inspection), crates/lgbm/examples/bench_train.rs (run only)</files>
  <action>
Establish a baseline and confirm the target before editing.

1. Capture a train baseline with the existing harness:
   `cargo run --release --example bench_train` (from repo root, package lgbm).
   Record the three `train_median` numbers (small/medium/large) into the SUMMARY —
   these are the before/after instrument.

2. Confirm the allocation hot spot by code inspection (no guessing). Verify all
   three facts that make the line-1647 `to_vec()` the target:
   (a) `raw` at learner.rs:1629 is an owned `Vec<f64>` the learner exclusively owns
       and does not read after the per-feature loop;
   (b) `fix_histogram` (fix_histogram.rs:50) and `compact_histogram`
       (learner.rs:3150) both take `&mut [f64]` and mutate in place;
   (c) the loop runs once per feature per directly-built leaf (call sites
       learner.rs:1127/1434/1528/1546).
   Confirm via grep that `raw` has no other reader after line 1655 in the function
   body. If any of (a)-(c) is FALSE, STOP and report — do not proceed to Task 2
   with a wrong target.

3. Record in the SUMMARY: the named target (`build_leaf_histogram_into` per-feature
   `to_vec()` at learner.rs:1647), why it is the highest-impact lowest-risk win
   (per-feature × per-leaf × per-tree × per-iter churn), and why it is provably
   parity-neutral (in-place mutation of the same f64 storage, same op order).
  </action>
  <verify>
    <automated>cargo run --release --example bench_train 2>&1 | tail -8</automated>
  </verify>
  <done>Baseline train_median for small/medium/large captured; facts (a)-(c) confirmed by inspection; the target allocation site is named with file:line in the SUMMARY. No code changed yet.</done>
</task>

<task type="auto">
  <name>Task 2: Eliminate the per-feature to_vec() (in-place fix + compact on owned raw)</name>
  <files>crates/lgbm-treelearner/src/learner.rs</files>
  <action>
In `build_leaf_histogram_into` (learner.rs, per-feature loop ~1645-1655), make
`raw` mutable (`let mut raw = ...` at line 1629) and rewrite the loop body to run
`fix_histogram` and `compact_histogram` directly on a `&mut` sub-slice of `raw`,
then copy that sub-slice into `buf`. Remove the `let mut hist = ...to_vec()`
clone entirely.

The new loop must call exactly the SAME two functions, with exactly the SAME
arguments (`f.most_freq_bin`, `sum_g`, `sum_h`, `f.offset`), on exactly the SAME
f64 cells in the SAME order — only the backing storage changes from a per-feature
clone to the owned `raw` buffer's sub-slice. Borrow the sub-slice mutably for the
two in-place ops, end that borrow, then `buf[slot_off[fpos]..+cells]
.copy_from_slice(&raw[slot_off[fpos]..+cells])`. Watch the borrow checker:
take the `&mut raw[range]` for fix+compact in an inner scope (or via
`split_at_mut`/index), then re-borrow `raw` immutably for the copy-into-`buf`.

Do NOT change `fix_histogram`, `compact_histogram`, the Backend call, the
`feature_bins`/`num_bins` collects, or any accumulation logic. Do NOT alter the
empty/no-hessian `buildable` short-circuit. This is purely the removal of one
clone per feature.

Keep the existing explanatory comments (FixHistogram / compaction / Pitfall 2) —
they remain accurate; just attach them to the in-place ops.
  </action>
  <verify>
    <automated>cargo test -p oracle-harness --test learner_parity --test kernel_parity 2>&1 | tail -20</automated>
  </verify>
  <done>learner_parity (incl. spine_real bit-exact vs real lib_lightgbm 4.6) and kernel_parity pass with the clone removed. The per-feature `to_vec()` is gone from `build_leaf_histogram_into`.</done>
</task>

<task type="auto">
  <name>Task 3: Full parity gate + before/after measurement</name>
  <files>crates/lgbm-treelearner/src/learner.rs (no further edits expected)</files>
  <action>
Prove no parity regression across the whole gate, then re-measure.

1. Run the bit-exact / parity merge gate:
   `cargo test -p lgbm -p lgbm-treelearner -p lgbm-boosting -p oracle-harness`.
   This MUST stay GREEN with the same pass/ignore counts as before the change.
   The CPU f64-fold spine staying bit-exact is the hard merge gate — any new
   failure or any newly-required tolerance is a STOP-and-revert condition (the
   change is parity-neutral by construction; a failure means the edit altered
   storage/order incorrectly).

2. Re-run `cargo run --release --example bench_train` and record after numbers
   next to the Task-1 baseline in the SUMMARY. Report the delta honestly — if the
   single-threaded spine shows flat/noise (as 260609-8go found for allocator
   swaps), say so; the deliverable is the *allocation reduction* (one fewer Vec
   per feature per leaf build), which is correct regardless of wall-clock noise on
   this synthetic corpus. Do NOT overclaim a speedup that the bench does not show.

3. clippy clean on the edited file: `cargo clippy -p lgbm-treelearner`.
   `LightGBM/` must never be git-added.
  </action>
  <verify>
    <automated>cargo test -p lgbm -p lgbm-treelearner -p lgbm-boosting -p oracle-harness 2>&1 | tail -25</automated>
  </verify>
  <done>Full parity suite GREEN with unchanged pass/ignore counts; before/after bench numbers recorded; clippy clean; allocation reduction confirmed (per-feature clone removed). parity_class = neutral.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| none (internal refactor) | No external/untrusted input crosses this change; it is a pure in-process allocation rewrite of an existing owned buffer. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-bfx-01 | Tampering | build_leaf_histogram_into f64 fold order | mitigate | Run fix_histogram/compact_histogram in the SAME order on the SAME f64 cells; prove bit-exact via learner_parity (spine_real vs real lib_lightgbm 4.6) + full oracle-harness gate. |
| T-bfx-02 | Information disclosure | none | accept | No data crosses a trust boundary; purely internal buffer reuse. |

(No npm/pip/cargo install tasks in this plan — package-legitimacy gate N/A.)
</threat_model>

<verification>
- `cargo test -p oracle-harness --test learner_parity` — bit-exact CPU spine vs real lib_lightgbm 4.6 (the hard merge gate).
- `cargo test -p oracle-harness --test kernel_parity` — histogram kernel parity unregressed.
- `cargo test -p lgbm -p lgbm-treelearner -p lgbm-boosting -p oracle-harness` — full workspace parity gate GREEN, unchanged pass/ignore counts.
- `cargo clippy -p lgbm-treelearner` — clean.
- `cargo run --release --example bench_train` — before/after train_median recorded.
</verification>

<success_criteria>
- The per-feature `to_vec()` clone in `build_leaf_histogram_into` is removed; fix+compact run in place on the owned `raw` buffer.
- CPU f64-fold spine remains BIT-EXACT (learner_parity passes, no tolerance weakened, no test #[ignore]d to pass).
- Full parity suite GREEN with the same counts as the pre-change baseline.
- One fewer heap allocation per feature per directly-built leaf build.
- Before/after bench numbers recorded honestly (no overclaim).
- parity_class = neutral.
</success_criteria>

<output>
Create `.planning/quick/260609-bfx-optimise-memory-bottleneck-in-tree-const/260609-bfx-SUMMARY.md` when done.
</output>
