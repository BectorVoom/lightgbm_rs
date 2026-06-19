---
quick_id: 260619-sgu
title: "Refer cubecl manual + reduce GPU kernel overhead — deferred-sync WIRE"
date: 2026-06-19
status: complete
disposition: RECONCILE-ONLY — no production code change (deferred-sync WIRE target no longer exists in production)
outcome: NULL-WIRE (honest); stale "WIRE pending" memory reconciled
commits: docs-only
---

# Quick 260619-sgu — SUMMARY

## Outcome: RECONCILE-ONLY — the deferred-sync WIRE target no longer exists in production

The brief (follow-up to q2z) was to wire the cubecl ch.05 lazy-execution
(deferred-sync) dispatch pattern into the production per-feature GPU histogram leaf
loop. **A `--research` pass + independent orchestrator verification proved there is no
such loop in production to wire** — it was already collapsed into a single batched
launch by the 260608 batched/resident work. Per the user's decision, this task lands
as a reconciliation (no production code change) that corrects the stale memory and
records the evidence so the lever is not re-spiked.

## Evidence (grep + file:line, independently verified)

1. **Production leaf build = ONE launch per leaf (not a per-feature loop).**
   `construct_leaf_hist_resident_lds_kernel::launch_unchecked` is launched with
   `CubeCount::Static(num_features as u32, p, 1)` — all features in one dispatch
   (`crates/lgbm-compute/src/kernels/histogram.rs:1709-1711`). The fused/resident scan
   path keeps the histogram device-resident in a pool Handle with **no per-leaf
   read-back**; the larger sibling is derived on-device via `subtract_resident`.

2. **The per-feature immediate-read path is TEST-ONLY (zero production callers).**
   `construct_histograms_parallel_f32_on` (histogram.rs:416) does `launch_unchecked`
   then immediate `read_one_unchecked(h_out)` (line 467), reached only via
   `RocmBackend::construct_histograms`. Grep of `.construct_histograms(` across
   `crates/` returns only:
   - `oracle-harness/tests/kernel_parity.rs`
   - `oracle-harness/tests/learner_parity.rs`
   - `oracle-harness/tests/boosting_parity.rs`
   - `lgbm-compute/tests/rocm_backend_parity.rs`

   This test-only path **is** the q2z spike's slow "Arm A" baseline. Production never
   calls it.

3. **No other production submit-block-submit-block loop exists.** The remaining
   production `read_one_unchecked` sites are each one launch → one read per call:
   `subtract.rs:116/228`, `partition.rs:266`, `split.rs:1078/1291` (split's
   multi-feature scan already batches all features into one launch; the `for f in feats`
   loops are CPU-side result assembly, not per-feature launches). Nothing for a deferred
   drain to overlap.

4. **cubecl 0.10.0 deferred-drain API confirmed.** The idiomatic single deferred drain
   of many handles is `pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes>`
   (`cubecl-runtime-0.10.0/src/client.rs:131`) — the clean form of the spike's
   hand-rolled N×`read_one_unchecked`. Launches are non-blocking (queue on a per-StreamId
   stream); only `read*`/`sync` block, so deferral is pure call-ordering and
   numerics-preserving. (Recorded for future use; not applied to production this task.)

## Why no production change (the honest call)

The q2z win (+19–26% compute-bound) was measured **against the test-only Arm A
baseline**. Production already realizes that benefit structurally (one launch/leaf +
resident no-readback). Re-introducing a per-feature loop solely to create a deferral
seam would manufacture a win against a baseline production already beats — a regression
dressed as an optimization. The CPU f64 anchor is the bit-exact hard merge gate; the
correct action is to NOT touch it (or any kernel/launcher) for a win that is already
captured.

## Disposition

- **No production code change.** No kernel, launcher, or CPU anchor touched.
- **Memory reconciled:** `gpu-lazy-dispatch-deferred-sync-win` updated from
  "WIRE pending" → "WIRE MOOT — superseded by the 260608 batched/resident collapse;
  do not re-spike, do not re-wire." `client.read(Vec<Handle>)` recorded as the
  idiomatic single-drain for any genuinely-new multi-launch seam that might appear later.
- **No new gates run** (no code changed → existing merge gates unaffected). The
  parity-contract `--validate` checker/verifier are moot for a docs-only reconciliation
  and were intentionally skipped.

## Optional future work (NOT done here, surfaced for the record)

- The test-only per-feature path could be tidied to use `client.read(Vec<Handle>)`
  instead of N×`read_one_unchecked` to match the cubecl manual's idiom — but it is
  test-only, so this is code-style alignment with no production perf impact. Deferred.
- A genuinely-new deferred-drain win would require a NEW multi-launch production seam
  (not the histogram leaf build) + a fresh A/B + parity re-validation — treat as a new
  spike, not a "wiring" of q2z.

## Files

- `260619-sgu-PLAN.md` — reconcile-only plan
- `260619-sgu-RESEARCH.md` — cubecl 0.10 API + production seam evidence (the decisive doc)
- `260619-sgu-SUMMARY.md` — this file
