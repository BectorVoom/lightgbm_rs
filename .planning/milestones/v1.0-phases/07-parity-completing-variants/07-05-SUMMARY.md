---
phase: 07-parity-completing-variants
plan: 05
subsystem: boosting-sample-strategy
tags: [goss, sample-strategy, bst-04, argmaxatk, grad-hess-amplification, rng-replay, bagging-rands, lightgbm-4.6, oracle-parity, numerical-fidelity]

# Dependency graph
requires:
  - phase: 07-01
    provides: "the D-05 faithful-fix (min_gain_shift from the 2*kEpsilon-bumped sum_hessian) that made the bagged-SUBSET split-gain deterministic — GOSS trains on a kept SUBSET, so it inherits this now-settled subset-path determinism"
  - phase: 06-05
    provides: "the BaggingSampleStrategy build-once bagging_rands_ block-1024 RNG (advanced across draws, never re-seeded) + the subset-train + OOB-score path in Gbdt::train_one_iter that GOSS reuses"
provides:
  - "GossSampleStrategy (goss.hpp 1:1): build-once bagging_rands_ block-1024 RNG, skip subsampling for iter < 1/learning_rate, ArgMaxAtK top-k threshold over |grad*hess|, per-row keep/amplify with the running rest_need/rest_all prob, amplify grad AND hess by multiply=(cnt-top_k)/other_k"
  - "ArrayArgs::ArgMaxAtK + Partition ported bit-for-bit (NOT a full sort — tie behavior matches)"
  - "data_sample_strategy=goss selection in Gbdt::train_one_iter (after get_gradients, before the learner; IsHessianChange amplifies grad/hess in place); mutually exclusive with bagging"
  - "builder setters top_rate/other_rate/data_sample_strategy/boosting + goss(top,other) convenience; booster GOSS wiring"
  - "goss-oracle-capture xtask + xtask/py/goss_oracle_capture.py; 16 real-lib_lightgbm-4.6 model parity goldens + the goss_rng_replay golden, byte-idempotent"
  - "BST-04 GOSS validated: real-binary parity (bit-exact leaf values) across top_rate×other_rate×{es}×{bfa} + the dedicated RNG-replay golden (kept/dropped indices bit-exact)"
affects: [07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GOSS as a BaggingSampleStrategy sibling: a SampleStrategy that runs INSIDE train_one_iter (after get_gradients, before the learner) because IsHessianChange mutates grad/hess in place, then reuses the bagging subset-train + dropped-row predict-side scoring path"
    - "ArgMaxAtK (quickselect nth_element) ported verbatim for the top-k threshold — never a full sort (tie behavior differs); the negated `!(a > v)` partition comparisons are a faithful port (allow neg_cmp_op_on_partial_ord with justification)"
    - "RNG-replay golden over the bit-exact C++ Random LCG re-implemented in the capture python (the wheel cannot expose internal bag indices) — identical posture to the bagging bag_indices_* golden; the golden carries the input grad/hess so the Rust replay self-derives the |g*h| magnitudes + ArgMaxAtK threshold"

key-files:
  created:
    - crates/oracle-harness/tests/fixtures/goss/.gitkeep
    - crates/oracle-harness/tests/fixtures/goss/goss_rng_replay.txt
    - "crates/oracle-harness/tests/fixtures/goss/goss_t{200,100}_o{100,50}_es{0,1}_bfa{0,1}_model.txt (16 cells)"
    - xtask/py/goss_oracle_capture.py
    - .planning/phases/07-parity-completing-variants/07-05-SUMMARY.md
  modified:
    - crates/lgbm-boosting/src/sample_strategy.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm-boosting/src/error.rs
    - crates/lgbm-boosting/src/lib.rs
    - crates/lgbm/src/builder.rs
    - crates/lgbm/src/booster.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - xtask/src/main.rs
    - .planning/REFERENCE_MANIFEST.md

key-decisions:
  - "GOSS reuses the proven build-once bagging_rands_ block-1024 RNG (STATE.md 06-05 CRITICAL fix) — constructed ONCE in reset_sample_config, advanced across draws; re-seeding per draw would re-draw the same bag."
  - "GOSS shares the bagging subset-train + dropped-row predict-side scoring path in Gbdt::train_one_iter (Pitfall 4: dropped rows STILL get the tree prediction). The selection block computes a shared (use_subset, in_bag, oob) from whichever strategy (bagging XOR goss) is active; GOSS additionally amplifies grad/hess in place BEFORE the learner."
  - "The GOSS RNG-replay golden freezes the goss.hpp Helper output over the bit-exact C++ Random LCG re-implemented in the capture python (the wheel cannot expose internal bag indices), exactly like the bagging bag_indices_* golden. The model parity goldens ARE real-lib_lightgbm-4.6 trains, so a wrong amplification/kept-set fails the parity replay (verified by corrupting a golden)."
  - "GOSS forbids bagging (goss.hpp:87-89): the booster selects GOSS XOR bagging; the GOSS parity matrix has NO bag axis."

patterns-established:
  - "A grad/hess-mutating SampleStrategy (IsHessianChange) sits between get_gradients and the learner; the kept/dropped index arrays flow into the existing subset path."
  - "ArgMaxAtK/nth_element ports for top-k selection — never a stable sort; carry a unit test that distinguishes the partition order from a sort."

requirements-completed: [BST-04]

# Metrics
duration: ~16 min (1 session: TDD strategy + math, gbdt selection, builder/booster/xtask wiring, capture via the ready 4.6.0 venv, byte-idempotent verify)
completed: 2026-06-07
---

# Phase 7 Plan 05: GOSS Sample Strategy (BST-04) Summary

**GOSS (gradient-based one-side sampling) ships end-to-end as a `BaggingSampleStrategy` sibling, faithful 1:1 to `goss.hpp`: build-once `bagging_rands_` block-1024 RNG (advanced across draws), skip subsampling for `iter < 1/learning_rate`, `ArgMaxAtK` top-k threshold over `|grad*hess|` (verbatim quickselect, NOT a full sort), per-row keep/amplify with the running `rest_need/rest_all` prob, and amplification of grad AND hess by `multiply=(cnt-top_k)/other_k`. It is selected on `data_sample_strategy=goss`, runs inside `train_one_iter` after `get_gradients` (it mutates grad/hess in place — `IsHessianChange`), and reuses the proven bagging subset-train + dropped-row predict-side scoring path. Validated against real `lib_lightgbm` 4.6 across `top_rate×other_rate×{es}×{bfa}` (16 cells, bit-exact leaf values) plus a dedicated RNG-replay golden (kept/dropped indices bit-exact).**

## Performance

- **Duration:** ~16 min, one session.
- **Completed:** 2026-06-07
- **Tasks:** 3 — (1) GossSampleStrategy + ArgMaxAtK + Gbdt selection + RNG-replay test infra (TDD); (2) builder setters + booster wiring + xtask capture emitter + capture-gated parity cells; (3) the real-binary capture (the wheel gate was already satisfied by the ready `/tmp/lgbm-capture-venv` 4.6.0 venv, so the executor completed it rather than halting).

## What shipped

1. **`GossSampleStrategy`** (`crates/lgbm-boosting/src/sample_strategy.rs`) — 1:1 port of `GOSSStrategy` (`goss.hpp`):
   - `reset_sample_config` builds the per-block `Random(bagging_seed+i)` window ONCE; CHECKs `top+other<=1` and both `>0` (`goss.hpp:85-86`) as a typed `BoostingError::GossConfig`.
   - `bagging(iter, &mut grad, &mut hess)` skips for `iter < (int)(1/lr)` (`goss.hpp:33`), else runs `helper`.
   - `helper` (`goss.hpp:116-165`): per-row `Σ|g*h|`, `top_k=max(1,(int)(cnt*top_rate))`, `other_k=(int)(cnt*other_rate)`, `ArgMaxAtK(top_k-1)` threshold, then per row IN ORDER keep-top / `NextFloat()<prob` keep+amplify (grad AND hess `*= multiply`) / drop, with `prob = rest_need/rest_all` running.
2. **`ArgMaxAtK` + `Partition`** ported verbatim (`array_args.h:101-146`) — quickselect nth_element semantics, NOT a sort.
3. **Gbdt selection** (`gbdt.rs::train_one_iter`): the sampling block now computes a shared `(use_subset, in_bag, oob)` from bagging XOR GOSS; GOSS amplifies grad/hess in place before the learner, then the existing subset-train + kept/dropped predict-side scatter runs. `with_goss(goss, features)` builder.
4. **Builder + booster** (`builder.rs`, `booster.rs`): `top_rate`/`other_rate`/`data_sample_strategy`/`boosting` setters + `goss(top,other)` convenience; booster selects `GossSampleStrategy` when `data_sample_strategy=goss` (mutually exclusive with bagging).
5. **Capture** (`xtask/src/main.rs` + `xtask/py/goss_oracle_capture.py`): `goss-oracle-capture` emits 16 real-binary model goldens + the `goss_rng_replay` golden; byte-idempotent.
6. **Parity cells** (`boosting_parity.rs`): `goss_rng_replay` (kept/dropped indices bit-exact) + `goss_parity_matrix` (per-tree leaf values bit-exact over the overlapping trees).

## Deviations from Plan

None — the plan executed as written. The Task-1 plan listed `set.rs` for confirming `top_rate`/`other_rate` + the `goss` alias-expansion; a re-grep (per 07-PATTERNS A2) confirmed both are ALREADY present (`set.rs:201-207` for the rates with `[0,1]` CHECKs, `set.rs:472-476` for the `boosting==goss` → `gbdt` + `data_sample_strategy=goss` expansion), so no `set.rs` edit was needed. The `top+other<=1` CHECK lives at the GOSS strategy boundary (`goss.hpp:85`, mirrored in `GossSampleStrategy::reset_sample_config`), not in config (matching C++, where it is a strategy-construction CHECK).

The capture step (Task 3) was a `checkpoint:human-verify` only because the wheel was historically absent; the ready `/tmp/lgbm-capture-venv` (lightgbm 4.6.0) satisfied that gate, so the capture was completed in-session (no halt) per the execution-context guidance.

## Out-of-scope (not fixed — deviation scope boundary)

- Pre-existing `clippy::ptr_arg` warnings at `gbdt.rs:490,496` (`&feature_row` in the bagging subset-path predict call) predate this plan (present in the parent commit) and are NOT in the GOSS additions — left untouched.

## Verification

- `cargo test -p lgbm-boosting` — **GREEN** (38 lib tests incl. 7 GOSS: config check, skip window, grad+hess amplification, ArgMaxAtK-not-sort, RNG-replay reference, gbdt selection, skip-window-equals-no-sampling).
- `cargo test -p oracle-harness --test boosting_parity goss` — **GREEN** (`goss_rng_replay` + `goss_parity_matrix`, both with goldens present — NOT skip-passing).
- **Teeth verified:** corrupting one model golden FAILS `goss_parity_matrix`; restored.
- **Byte-idempotent:** a second `goss-oracle-capture` run left an empty `git diff` over `fixtures/goss/`.
- `cargo test --workspace` — **GREEN** (50 test binaries, 0 failed; `boosting_parity` 40 passed / 7 ignored — the 7 ignored are the unrelated DEF-07-02 fair/quantile-bagged cells, untouched).
- **Spine NOT regressed:** `subset_determinism_diagnostic` (07-01) + the D-07 matrix (`early_stopping`) GREEN; the 07-01 bagging bit-exactness intact.
- `cargo build --workspace --tests` — exit 0; clippy clean on every edited file.
- `git status --porcelain` — `LightGBM/` never git-added.

## Task Commits

1. `d3b1c8f` — `feat(07-05)`: GossSampleStrategy (ArgMaxAtK + amplification) + Gbdt selection.
2. `9f51e1f` — `feat(07-05)`: GOSS builder setters + booster wiring + capture emitter + parity cells.
3. `fdae38c` — `test(07-05)`: capture GOSS real-lib_lightgbm-4.6 goldens (parity + RNG-replay).

## Self-Check: PASSED

- `07-05-SUMMARY.md` exists on disk; `GossSampleStrategy` + the 16 model goldens + `goss_rng_replay.txt` + `goss_oracle_capture.py` all present.
- Commits `d3b1c8f` / `9f51e1f` / `fdae38c` present in history.
- `cargo test --workspace` GREEN; `LightGBM/` never git-added.
