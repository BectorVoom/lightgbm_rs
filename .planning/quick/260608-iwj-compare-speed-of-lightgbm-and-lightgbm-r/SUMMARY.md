---
quick_id: 260608-iwj
slug: compare-speed-of-lightgbm-and-lightgbm-r
status: complete
date: 2026-06-08
---

# Quick Task 260608-iwj — Compare speed & optimise (SUMMARY)

## What was done

Built a Rust-only before/after benchmark harness, measured the four parity-safe
optimization levers cumulatively, applied them, verified the bit-exact parity gate
stayed green, and timed C++ `lightgbm==4.6` on identical data as a reference point.

## Result

| | small | medium | large |
|--|--|--|--|
| **Rust train, M0 → M3** | 1.71s → 1.55s (−9.4%) | 4.75s → 4.21s (−11.4%) | 8.93s → 8.12s (−9.1%) |
| **C++ 4.6 (1 thread)** | 19.9ms | 75.9ms | 206.6ms |

**Headline:** lightgbm_rs is bit-exact-correct but **~40–80× slower** than C++
LightGBM 4.6. The parity-safe levers bought ~9–11%; the order-of-magnitude gap is
architectural (see REPORT.md roadmap R1–R4).

## Levers applied (all parity-neutral, gate verified)

1. **`[profile.release]`** — lto=fat, codegen-units=1 (workspace Cargo.toml). ~3%.
2. **mimalloc `#[global_allocator]`** — bench (feature-gated) + `lgbm-python`
   cdylib (unconditional, the shipped artifact). ~5%.
3. **Gather buffer-reuse** — `build_leaf_histogram_into` reuses 3 scratch Vecs
   across features instead of allocating per feature.
4. **Candidate pre-sizing** — `per_bin_gains` cand_rev/cand_fwd pre-sized to
   num_bin. (~1–2% combined for 3+4; mimalloc had already absorbed most churn.)

`smallvec` crate was NOT added: every hot small-Vec site escapes into stored
`FeatureSplitRecord`/splittable structures (type ripple would risk the bit-exact
gate); the one clean non-escaping site (`gains`) is CEGB-gated/cold. Buffer-reuse
achieved the allocation-churn goal on the actual hot paths instead. Documented in
the T4 commit + REPORT.

## Parity gate (non-negotiable) — GREEN

`cargo test -p oracle-harness`: boosting_parity 75, learner_parity 29,
kernel_parity 4, predict_parity 5, raw_bin_train_parity 2, rng_parity 1 — all
bit-exact, 0 failed. Core unit suites (lgbm 41, boosting 55, compute 18,
treelearner 64) green. **Zero numeric change.**

## Out of scope / explicitly NOT done

- Half-precision f16/bf16 GPU kernels (would break the parity gate — by decision).
- The R1–R4 architectural speedups (snapshot opt-in, batched histogram dispatch,
  columnar storage, feature-parallel rayon) — each is phase-sized; flagged in
  REPORT.md for follow-up.

## Artifacts

- `crates/lgbm/examples/bench_train.rs` — the benchmark harness
- `260608-iwj-REPORT.md` — full measurement table + C++ comparison + roadmap
- `bench_cpp_ref.py` — C++ reference timing script
- Commits: 74256d7 (T1) · T2 profile · T3 mimalloc · T4 learner reuse
