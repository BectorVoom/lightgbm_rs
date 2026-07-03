---
status: complete
phase: quick-260619-tlk
plan: 01
subsystem: gpu-compute
tags: [cubecl, gpu-overhead, audit, batched-read, deferred-sync, rocm, parity-gate]
requires:
  - "cubecl 0.10 / cubecl-runtime 0.10 (pinned)"
  - "the wired RocmBackend per-leaf resident path (build_fix_scan_resident, subtract_resident, scan_resident_leaf)"
provides:
  - "An evidence-backed CASE-A verdict: no remaining non-redundant batched-read / deferred-sync / round-trip lever in the production GPU dispatch/read path"
  - "A rocm-gated, measurement-only Case-A confirmation harness (batched_read_audit_ab.rs)"
affects:
  - "none (measurement + docs only; no production source changed)"
tech-stack:
  added: []
  patterns:
    - "audit-before-wire (faithfulness over busy-work): cite the manual, inventory production, refuse to manufacture a win"
key-files:
  created:
    - crates/lgbm-compute/examples/batched_read_audit_ab.rs
    - .planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md
  modified: []
decisions:
  - "CASE A: production already satisfies every cubecl-0.10 overhead-reduction idiom (launch_unchecked shipped; one-launch-one-consolidated-read per launcher; resident pool reused; per-leaf path = one fused launch + device-resident subtract). No lever to wire."
  - "read_one_unchecked(h) IS read(vec![h]) (cubecl-runtime-0.10.0 client.rs:131-149 — same read_async(vec![h]) -> read_sync drain); the batched idiom only collapses an N-handle read loop, which has no production caller."
metrics:
  duration: "~4 min"
  completed: "2026-06-19"
---

# Quick 260619-tlk: Refer the cubecl manual & reduce GPU-kernel overhead Summary

**Audited the production GPU dispatch/read path against the cubecl 0.10 overhead-reduction
idioms and recorded a CASE-A verdict — production already batches everything; there is no
remaining non-redundant batched-read / deferred-sync / round-trip lever to wire — backed by
manual + runtime-source citations, a launcher-by-launcher round-trip inventory, and a
rocm-gated measurement-only confirmation harness, with the CPU f64 bit-exact anchor proven
unregressed.**

## What was done

### Task 1 — cubecl 0.10 manual + production round-trip inventory
- Fetched the cubecl 0.10 manual (`ctx7 docs /tracel-ai/cubecl`) and cited each
  overhead-reduction idiom with attribution: `launch_unchecked` (drop bounds-check
  codegen), `read_one`/`read_one_unchecked` (single-handle drain), batched
  `client.read(Vec<Handle>)` (collapse an N-handle read loop), deferred/lazy execution
  (async submission; sync at `read`/`sync`/`flush`), allocation reuse (`client.empty`/
  `create_from_slice`), autotune (runtime kernel selection — orthogonal to single-launch
  overhead, recorded not pursued).
- Quoted the **load-bearing runtime fact** from `cubecl-runtime-0.10.0/src/client.rs`
  (lines 131-149): `read_one_unchecked(h)` is *defined as* `read_async(vec![h])` drained by
  `read_sync` — i.e. literally the one-element case of the batched `read(Vec<Handle>)`.
- Built the PRODUCTION round-trip inventory (grep of every `::launch*` +
  `read_one_unchecked`/`client.read` in `crates/lgbm-compute/src`, cross-referenced with the
  wired `RocmBackend` methods and the `lgbm-treelearner` per-leaf path): **every production
  launcher is one-launch -> at-most-one-consolidated-read; none leaves >=2 out-handles
  unread before a sync.** The per-leaf hot path is the single fused
  `build_fix_scan_resident_f64_on` launch (build+fix+compact+scan ALL features, one cube per
  feature, `CubeCount::Static(n,1,1)`) + a device-resident `subtract_resident` (no readback)
  + one small SplitInfo drain.
- Confirmed (by grep of `crates/lgbm-treelearner/src`) that the per-feature submit->block
  path the q2z deferred-sync spike beat (`construct_histograms_parallel_f32_on`) has **ZERO
  production callers** — it is wired only to the single-feature `Backend::construct_histograms`,
  which the learner never calls; its real callers are rocm parity tests + the example harnesses.

### Task 2 — CASE-A verdict + confirmation harness + parity gate
- Recorded the explicit CASE-A verdict in FINDINGS.md with the per-launcher evidence and the
  manual citation that the single-handle read already IS the idiomatic batched form.
- Wrote `crates/lgbm-compute/examples/batched_read_audit_ab.rs` — a rocm-gated,
  measurement-only confirmation harness that launches the shipped fused atomic kernel ONCE
  into ONE consolidated out-handle (the production shape) and times the two read idioms on
  that single handle (arm A `read_one_unchecked(h)` vs arm B `read(vec![h])`), asserting
  byte-equivalence and expecting a sub-noise delta (they are the same drain). Mirrors
  `lazy_dispatch_ab.rs`'s `#[cfg(not(feature="rocm"))]` no-op main. It deliberately does NOT
  manufacture an N-handle loop to "beat" (the forbidden q2z/sgu anti-pattern).

## Parity gate (the merge gate)
- `cargo build -p lgbm-compute --example batched_read_audit_ab` (default cpu feature): GREEN.
- `cargo test -p lgbm-compute --lib`: **GREEN — 30 passed / 0 failed / 1 ignored** (the 1
  ignored is the pre-existing rocm-gated cell, not a regression).
- `cargo test -p oracle-harness --test kernel_parity`: **GREEN — 6 passed / 0 failed / 0
  ignored** (all CPU f64 bit-exact anchor cells: histogram/split/subtract/partition
  bit-exact + the fused==per-feature==native oracle).
- **Non-rocm runner:** the `--features rocm` hip cells are feature-gated out, so the hip
  kernel_parity/learner_parity cells did NOT run here; the audit + example touch no hip
  kernel, so the 1e-6 hip envelope is unaffected by construction. To exercise the Case-A A/B
  on hardware: `cargo run --release -p lgbm-compute --features rocm --example
  batched_read_audit_ab` (gfx1100; expected delta ~0 within p25/p75).
- The CPU f64 bit-exact anchor is provably unregressed. No production kernel, launcher,
  learner, or CPU anchor was modified.

## Deviations from Plan
None - plan executed exactly as written (CASE A, the expected branch).

## Known Stubs
None.

## Threat Flags
None — no new network endpoint, auth path, file access, or schema change. The only new
surface is a `#[cfg(feature="rocm")]` measurement example.

## Commits
- `0af9709`: feat(quick-260619-tlk): add rocm-gated Case-A batched-read audit example

## Self-Check: PASSED
- `crates/lgbm-compute/examples/batched_read_audit_ab.rs` — FOUND (committed in 0af9709).
- `.planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md` — FOUND.
- Commit `0af9709` — FOUND in git log.
