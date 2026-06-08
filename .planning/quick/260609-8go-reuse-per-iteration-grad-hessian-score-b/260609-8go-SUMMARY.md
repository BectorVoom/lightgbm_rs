---
quick_id: 260609-8go
title: Optimize lightgbm_rs against the optimisor/manual techniques
status: complete
parity_class: neutral
code_commit: 9b6b667
---

# Quick Task 260609-8go — Summary

## Request

*"Consider whether there are areas that can be optimized in arrow-rs. Reference:
`/home/user/Documents/workspace/optimisor/manual`."* — scoped with the user to: **analyze
`lightgbm_rs` (this project) using the manual's 9 optimization techniques, then implement the
highest-value parity-neutral win.**

## Part 1 — Analysis (the deliverable)

`lightgbm_rs` does **not** use Apache Arrow (every "arrow" hit in the tree is the substring inside
*"narrow"*). The only Arrow-using project in the workspace is the `optimisor` scratch crate. So the
manual is a **technique catalog**, and the question is which of its patterns apply to a bit-exact-
parity-gated, single-threaded-CPU-spine LightGBM port.

| Manual technique | Verdict for lightgbm_rs |
|---|---|
| **Half precision f16/bf16 (CubeCL)** | ❌ **Disqualified** — breaks the f32-end-to-end parity contract (the project's non-negotiable; CPU anchor is the bit-exact merge gate). |
| **Zero-copy Arrow→CubeCL upload** | ⚪ Low value — no Arrow path; memory note records host↔device round-trip is *not* the GPU bottleneck. |
| **bytemuck zero-copy transmutation** | ⚪ Low value — same; upload isn't the bottleneck. |
| **Arrow numeric branching / dictionary casting** | ⚫ N/A — Arrow-specific; no Arrow data path. |
| **compact_str (SSO)** | ⚪ Low value — strings aren't in the numeric training hot path. |
| **smallvec (SVO)** | 🟡 Partial — real candidates in the tree-learner hot path, but most scratch Vecs are `num_bin`-sized (~2–4 KB → too big for inline SVO) and touch the bit-exact anchor → parity-risky for a quick task. |
| **jemalloc** | 🔁 Redundant — mimalloc already wired. |
| **mimalloc** | ✅ Already wired (bench feature + forced in `lgbm-python`). **Measured here** → flat. |

### mimalloc was measured (free — already wired), result is flat/mixed

`bench_train`, system vs mimalloc:

| size | system | mimalloc | Δ |
|---|---|---|---|
| small (2k×12) | 40.10 ms | 37.06 ms | −7.6% |
| medium (8k×30) | 239.59 ms | 243.53 ms | +1.6% |
| large (20k×50) | 829.24 ms | 835.61 ms | +0.8% |

Mixed/flat — the CPU spine is **single-threaded**, so there's no allocator contention to relieve.
Confirms the existing decision to keep mimalloc **opt-in**, not default. (Matches the project's prior
"profile before assuming" findings.)

## Part 2 — Implementation (highest-value parity-neutral win)

**Change:** move (not clone) the per-iteration grad/hess/score snapshots in the training driver
(`crates/lgbm/src/booster.rs`, commit `9b6b667`).

### What & why

The initial premise — "pool grad/hess/score across iterations" — turned out **wrong** on reading the
code, and the discovery *is* the finding: `train_one_iter` **moves** those buffers out into
`IterSnapshot`, and the driver **retains every iteration's copy** in the public golden-replay history
(`Booster::iter_scores` / `iter_grad_hess`). They are not transient → cannot be pooled.

The real waste was in the caller:
- `snap.gradients` / `snap.hessians` were **cloned** despite having no use after the push → now moved.
- `snap.score` was **double-allocated** (`train_one_iter` `to_vec()`s it at `gbdt.rs:911`, then the
  caller `.clone()`d it again) → now moved, with the two training-metric eval reads redirected to the
  just-pushed `iter_scores` element.

**Effect:** removes 3 large allocations + memcpys per emitted iteration (~320 KB/iter on the 20k-row
size). Move == clone in data ⇒ **parity-neutral by construction**.

### Validation

- **Bit-exact parity gate GREEN:** `cargo test -p lgbm -p lgbm-boosting -p oracle-harness` → all pass,
  including `raw_bin_train_matches_cpp_golden`, `rng_parity_replays_every_committed_case`, and the
  rank parity suite.
- **Bench (system allocator), before → after (isolated runs):**

  | size | before | after (run1 / run2) | verdict |
  |---|---|---|---|
  | small | 40.10 ms | 40.30 / 41.52 ms | flat (noise) |
  | medium | 239.59 ms | 244.73 / 249.57 ms | flat (noise) |
  | large | 829.24 ms | 875.32 / 861.05 ms | flat (noise) |

  **Wall-clock flat** — training is tree-construction-bound (histogram build + split finding), not
  boosting-loop-alloc-bound. (The first "after" run showing a 15% regression was discarded: it ran
  concurrently with the parity-test job and was CPU-contended.)

### Disposition

**Kept.** Unlike a perf gamble that adds risk/complexity for a flat result (→ gate off), this change
*removes* work and code (clones → moves) and eliminates a genuine double-allocation, with zero parity
risk. A strict simplification + alloc-churn reduction; flat wall-clock is acceptable.

## Follow-ups (documented, not done)

- **Tree-learner scratch reuse** (`learner.rs:1647` histogram `.to_vec()`, `:2541`/`:2595`
  `cand_rev`/`cand_fwd`) — the *highest allocation count* (hundreds of thousands of small Vecs per
  train) and the only place alloc reduction could plausibly move wall-clock. Touches the bit-exact
  anchor → needs its own measured task with parity diffing. **This is where to look next for a real
  CPU win from the manual's smallvec/pooling angle.**
- **Rayon parallelism of the single-threaded CPU spine** — the actual lever for the 40–80× C++ gap;
  architectural + parity-sensitive, out of scope for a quick task. Would also finally make the
  mimalloc/jemalloc allocator choice matter (contention relief).
- **`train_score_pre` reuse** (`gbdt.rs:626`) — a genuinely transient per-iter snapshot, but reuse
  needs a field + borrow-dance inside the parity-critical `train_one_iter` with early-return handling;
  marginal (1 alloc/iter), deferred.
- **Apache Arrow ingestion for `lgbm-python`** (mirrors the official pyarrow input + zero-copy→CubeCL)
  — a feature, not a quick optimization; the only place the manual's Arrow patterns would land.
