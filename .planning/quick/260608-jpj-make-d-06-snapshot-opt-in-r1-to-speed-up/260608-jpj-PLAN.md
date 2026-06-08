---
quick_id: 260608-jpj
slug: make-d-06-snapshot-opt-in-r1-to-speed-up
date: 2026-06-08
mode: quick (parity-gated; follow-up to 260608-iwj R1)
---

# Quick Task 260608-jpj — R1: make the D-06 snapshot opt-in

Production `train()` always computes the D-06 snapshot (`per_bin_gains`, a per-
feature host re-scan of every fixed histogram) and discards it. Make it opt-in so
the boosting loop skips it, while golden-replay tests still get full snapshots.

## Safety proof (from source trace)

- The live split decision is `backend.find_best_split(...)` → `best_split_per_leaf`
  (learner.rs:1672, 1749). The splittability gate is `this_leaf_splittable`
  (1693/1775). Neither depends on `per_bin_gains`.
- `per_bin_gains` (1731) feeds ONLY `FeatureSplitRecord.cand_rev/cand_fwd` →
  `SplitSnapshot` (1389), which `train()` / `train_returning_partition()` discard.
- `per_bin_gains` is a pure read (`hist`, `self.cfg`) — no side effects.
- Production boosting calls `train_returning_partition` (gbdt.rs:797,1125) →
  snapshots discarded. Golden tests call `train_with_snapshots` /
  `train_with_col_sampler_trace`.

⇒ Gating `per_bin_gains` on a capture flag is provably bit-identical for the grown
tree.

## Tasks

- **T1 — capture flag + gate.** Add `capture_snapshots: bool` field (default false).
  Thread `capture` into `train_inner`; set `self.capture_snapshots` at its top.
  Decouple `train()` from `train_with_snapshots` (today it delegates) — both call
  `train_inner` with the right flag. Gate the `per_bin_gains` call in
  `scan_leaf_histogram` on `self.capture_snapshots` (empty cand arrays when off).
  Wrappers: `train`/`train_returning_partition` = false; `train_with_snapshots`/
  `train_with_col_sampler_trace` = true.

- **T2 — parity gate + measure.** `cargo test -p oracle-harness` must stay GREEN
  (bit-exact); core unit suites green. Re-run `bench_train` (M3 vs M4). Update the
  REPORT with the R1 delta. Commit.

## Parity gate (non-negotiable)

`cargo test -p oracle-harness` GREEN — especially `learner_parity` /
`boosting_parity` (the D-06 snapshot replays must still pass with capture ON).
