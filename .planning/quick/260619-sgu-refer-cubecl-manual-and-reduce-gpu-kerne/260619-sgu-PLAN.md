---
quick_id: 260619-sgu
title: "Refer cubecl manual + reduce GPU kernel overhead — wire the deferred-sync dispatch lever"
date: 2026-06-19
mode: quick (--research --validate)
status: complete
disposition: RECONCILE-ONLY (no production code change)
---

# Quick 260619-sgu — Reduce GPU kernel overhead (cubecl deferred-sync WIRE)

## Original intent

Follow up on quick-260619-q2z (commit 7edddaa+a36a643), which spiked the cubecl
manual ch.05 lazy-execution (deferred-sync) lever on the GPU histogram build and
found the **first non-NULL** result on the GPU dispatch-overhead axis
(+19–26% compute-bound, bins≥256, feats≥32) with disposition **"WIRE pending"**.

The brief: wire that deferred-drain call-ordering pattern into the production
per-feature histogram leaf loop, gated to compute-bound × bins≥256 × feats≥32,
with end-to-end parity re-validation.

## What research found (decisive — premise was stale)

A `--research` pass (260619-sgu-RESEARCH.md) consulted the cubecl 0.10.0 client
API and the production GPU path. It overturned the brief:

- **The per-feature submit→block→submit→block loop the q2z spike modeled does NOT
  exist in the production GPU path.** It was already collapsed (260608-lsx/lad/p90/
  fw1) into **ONE launch per leaf** — `construct_leaf_hist_resident_lds_kernel::launch_unchecked`
  with `CubeCount::Static(num_features, p, 1)` (histogram.rs:1709-1711) — and on the
  fused/resident path the histogram stays **device-resident in a pool Handle with no
  per-leaf read-back at all**.
- The only code that still does per-feature `launch_unchecked` + immediate
  `read_one_unchecked` is `construct_histograms_parallel_f32_on` (histogram.rs:416→467),
  reached only via `RocmBackend::construct_histograms`. **Grep confirms its only
  callers are tests** (kernel_parity, learner_parity, rocm_backend_parity,
  boosting_parity) — zero production callers. This test-only path IS the spike's slow
  "Arm A" baseline.
- The other production `read_one_unchecked` sites (`subtract.rs` 116/228, `partition.rs`
  266, `split.rs` 1078/1291) are each **one launch → one read** per call — split's
  multi-feature scan already batches all features into one launch. No
  submit-block-submit-block loop anywhere in production for deferral to overlap.

**Conclusion:** the "WIRE pending" target no longer exists in production. The q2z
win was already captured structurally by the 260608 batched/resident collapse.
Wiring a deferred drain would require *re-introducing* a per-feature loop just to
manufacture a win against a baseline production already beats — dishonest, not an
optimization. (Confirmed independently by the orchestrator via grep + file reads,
and confirmed as the disposition by the user.)

## Tasks (reconcile-only)

1. **No production code change.** Do not re-introduce a per-feature leaf loop; do not
   touch any kernel, launcher, or the CPU f64 anchor.
2. **Record the evidence** in this quick task's SUMMARY (zero production callers of the
   per-feature path; production = single batched launch + resident no-readback;
   cubecl 0.10 idiomatic single-drain is `client.read(Vec<Handle>)`, client.rs:131).
3. **Update the stale memory** `gpu-lazy-dispatch-deferred-sync-win` disposition from
   "WIRE pending" → "WIRE MOOT — superseded by the 260608 batched/resident collapse;
   do not re-spike, do not re-wire." Update the MEMORY.md index hook.
4. **Update STATE.md** Quick Tasks Completed + Last activity.

## Verification note

The `--validate` plan-check + verifier steps target *code changes against the parity
contract*. With no code change, there is nothing for them to check against the
bit-exact CPU anchor / ~1e-6 ROCm gate — verification is moot and intentionally
skipped. The substantive verification here is the grep + file-read evidence that the
production seam is already collapsed (captured in SUMMARY).

## must_haves

- truths:
  - The production GPU leaf histogram build is a single batched `launch_unchecked` per
    leaf (`CubeCount::Static(num_features, p, 1)`), not a per-feature submit-block loop.
  - The per-feature immediate-read path (`construct_histograms`) has zero production
    callers — it is test-only.
  - cubecl 0.10.0's idiomatic single deferred drain of many handles is
    `client.read(Vec<Handle>)` (cubecl-runtime-0.10.0/src/client.rs:131).
- artifacts:
  - 260619-sgu-RESEARCH.md (cubecl API + seam evidence)
  - 260619-sgu-SUMMARY.md (reconciliation record)
  - updated memory `gpu-lazy-dispatch-deferred-sync-win`
- key_links:
  - crates/lgbm-compute/src/kernels/histogram.rs:1709 (single batched production launch)
  - crates/lgbm-compute/src/kernels/histogram.rs:416 (test-only per-feature launcher)
