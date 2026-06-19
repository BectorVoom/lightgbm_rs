# Quick 260619-sgu: Wire cubecl lazy-execution (deferred-sync) into the production GPU histogram leaf loop — Research

**Researched:** 2026-06-19
**Domain:** cubecl 0.10 client dispatch/sync API; GPU per-feature histogram launch ordering
**Confidence:** HIGH (cubecl API verified against installed crate source; production seam verified by grep + file:line read)

## Summary

The q2z spike proved a real win (+19–26% compute-bound, bins≥256, feats≥32) for **deferred-sync** dispatch: submit N per-feature histogram launches back-to-back into N distinct out-handles, then ONE drain — versus the spike's "Arm A" baseline of `launch → blocking read_one → launch → blocking read_one …`.

**The decisive finding for wiring: the per-feature submit-block-submit-block loop the spike modeled DOES NOT EXIST in the wired production GPU path.** The production ROCm leaf build was already collapsed (260608-lsx/lad/p90/fw1) into **ONE launch per leaf** (`CubeCount::Static(num_features, P, 1)` — all features in a single kernel dispatch), and on the fused/resident path the histogram stays **device-resident in a pool Handle with no per-leaf read-back at all**. The only code that still does the per-feature `launch_unchecked` + immediate `read_one_unchecked` is `construct_histograms_parallel_f32_on` (the single-feature `Backend::construct_histograms`), and grep confirms its **only callers are tests** (kernel_parity, learner_parity, rocm_backend_parity, boosting_parity) — never the production learner.

**Primary recommendation:** This is NOT a drop-in wiring of "defer the read in the per-feature leaf loop," because that loop is already a single batched launch. Two honest dispositions, in priority order:

1. **Most likely correct outcome — RECONCILE, do not re-wire the win away.** The spike's Arm A (the slow baseline) is the *test-only* per-feature path. The production path already realizes the spike's Arm B benefit structurally (one launch/leaf, resident no-readback). Confirm via a quick check that the wired path has no hidden multi-read-per-leaf seam (it does not — see §2). If so, the correct deliverable is a short FINDINGS note: "win already captured by the batched/resident collapse; no further wiring available on the wired path," plus optionally applying the cleaner `client.read(Vec<Handle>)` batch-drain to the **test-only** per-feature path so the test harness matches the manual's idiom.
2. **If a deferrable multi-launch seam is desired**, the candidate is the smaller+larger sibling builds and/or the build→scan chain within `update_leaf` — but the resident path already keeps these on-device (subtract_resident, fused build+fix+scan in one launch), so there is little host-round-trip bubble left to recover. Treat any such change as a NEW A/B, not a "wiring."

**Do not manufacture a win by re-introducing a per-feature loop just so deferral can be "wired".** The win the spike measured against was a baseline the production code already beats.

## cubecl 0.10 Lazy-Execution / Read API (VERIFIED)

cubecl/cubecl-runtime pinned at **0.10.0** [VERIFIED: Cargo.lock]. Signatures quoted from the installed crate source `cubecl-runtime-0.10.0/src/client.rs`:

| Method | Signature (file:line) | Semantics |
|--------|----------------------|-----------|
| submit (launch) | `kernel::launch_unchecked(client, count, dim, args…)` | **Non-blocking.** Queues the kernel on the client's stream; the CPU does NOT wait. This is the mechanism that makes deferral possible — N launches in a row never block. [VERIFIED: kernel macro + `submit` at client.rs:238] |
| `read_one` | `pub fn read_one(&self, handle: Handle) -> Result<Bytes, ServerError>` (client.rs:136) | Blocking single-handle drain (`read_sync(read_async(vec![handle]))`). |
| `read_one_unchecked` | `pub fn read_one_unchecked(&self, handle: Handle) -> Bytes` (client.rs:145) | Blocking single-handle drain, panics on error. The pattern used everywhere in histogram.rs today. |
| **`read`** | **`pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes>`** (client.rs:131) | **The idiomatic single deferred drain of MANY handles in ONE blocking sync.** This is the clean production form of the spike's hand-rolled "Arm B" (which did N separate `read_one_unchecked`). |
| `read_async` | `pub fn read_async(&self, handles: Vec<Handle>) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send` (client.rs:109) | The async primitive `read`/`read_one` are built on; returns a future over all handles. |
| `sync` | `pub fn sync(&self) -> DynFut<Result<(), ServerError>>` (client.rs:805) | Force-drain the whole stream without reading data back (used by resident path where data stays on device). |

**The deferred-sync pattern in canonical cubecl 0.10 form:**
```rust
// submit ALL launches first (non-blocking) into distinct out-handles
let h_outs: Vec<Handle> = (0..n).map(|_| client.create_from_slice(zeros)).collect();
for f in 0..n {
    unsafe { kernel::launch_unchecked(&client, count, dim, …, ArrayArg::from_raw_parts(h_outs[f].clone(), out_len)); }
}
// ONE deferred drain of the whole batch — replaces N blocking read_one calls
let all: Vec<Bytes> = client.read(h_outs);   // client.rs:131
```
[VERIFIED: cubecl-runtime-0.10.0/src/client.rs]. The spike (lazy_dispatch_ab.rs:264-268) emulated this with a loop of `read_one_unchecked`; `client.read(Vec<Handle>)` is the cleaner one-call equivalent and is what production code should use if any deferred-drain seam is wired.

**Stream ordering:** all ops carry a `stream_id` (client.rs:85-99); launches and the final `read` execute in submission order on that stream, so deferring the read does not reorder kernels — pure call-ordering, numerics-preserving. [VERIFIED: client.rs StreamId plumbing]

## 2. The Exact Production Seam (and why it is already collapsed)

### Per-feature immediate-read launcher — TEST-ONLY
`construct_histograms_parallel_f32_on` (histogram.rs:416) does `launch_unchecked` (line 456) then immediate `read_one_unchecked(h_out)` (line 467). It is reached only via `RocmBackend::construct_histograms` (lib.rs:1039). [VERIFIED: grep]

Callers of `.construct_histograms(` [VERIFIED: grep]:
- `oracle-harness/tests/kernel_parity.rs:206, 1215`
- `oracle-harness/tests/learner_parity.rs:340, 362, 369`
- `lgbm-compute/tests/rocm_backend_parity.rs:39, 42, 61`
- `oracle-harness/tests/boosting_parity.rs:1972`

**Zero production callers.** The production learner never calls `construct_histograms`.

### The actual WIRED production leaf build — already ONE launch/leaf
Production leaf histogram build flows: `learner.rs build_leaf_histogram_into` / `build_resident_leaf_into` → `Backend::build_leaf_histograms_raw` (lib.rs:1164) →
- resident cache populated → `build_leaf_histograms_resident_f32_on` → `resident_raw_build_into` (histogram.rs:1654), which launches `construct_leaf_hist_resident_lds_kernel` with **`CubeCount::Static(num_features as u32, p, 1)`** (histogram.rs:1709-1711) — ALL features in ONE dispatch. [VERIFIED: file read]
- fused/resident scan path keeps the histogram in a **pool Handle on-device** (`build_fix_scan_resident_f64_on`, learner.rs:2053) — NO per-leaf read-back; larger sibling derived via `subtract_resident` on-device (learner.rs:1594), also no read-back.

So across a leaf's features there is **already no submit→block→submit→block serialization** — the spike's Arm A baseline. The batched/resident collapse is the structural realization of the spike's Arm B.

### Minimal refactor — REVISED from the task brief
- **Do NOT** "collect handles across features, single drain after the loop" in the production leaf loop — there is no per-feature handle loop there anymore; it is a single batched launch.
- **If** any deferred-drain idiom is applied, apply it to the **test-only** per-feature path (replace the in-loop `read_one_unchecked` with one `client.read(Vec<Handle>)`) so tests exercise the manual's pattern — but note this changes only test timing, not production.
- **Gating** (when/if a future multi-launch production seam appears): the spike's regime gate is **compute-bound (large leaf) × bins≥256 × feats≥32**; launch-bound (small leaf) is NULL/negative (per STATE.md + memory `gpu-lazy-dispatch-deferred-sync-win`). A runtime gate would check `leaf_rows` (compute-bound proxy, e.g. ≥ the existing `RESIDENT_MIN_NUM_DATA=12_000` or higher), `num_bin ≥ 256`, and `num_features ≥ 32`, falling back to the immediate path otherwise.

## 3. Parity Risk + Re-Validation

Deferring the sync is **pure call-ordering** — same kernels, same inputs, same submission order — so numerics are unchanged (the spike's `assert_same_input_f32` confirmed A==B within the f32 envelope, lazy_dispatch_ab.rs:111-125). [VERIFIED: spike source]

**Parity gate (the #1 constraint):** the CPU f64 anchor is bit-exact and the hard merge gate; ROCm is held to ~1e-6 *vs that anchor*. Re-validation tests if any code is touched [VERIFIED: grep]:
- `oracle-harness/tests/kernel_parity.rs` — `kernel_parity_*` GPU↔CPU-anchor (f64 bit-exact: hip kernel_parity 9/9; f32 within ABS 5e-6/REL 1e-5).
- `oracle-harness/tests/learner_parity.rs` — end-to-end `learner_parity_*` incl. `learner_parity_resident_equals_host_tree_on_hip` (the resident-built tree vs host anchor).
- `lgbm-compute/tests/rocm_backend_parity.rs` — `rocm_parallel_histogram` 7/7.
- `oracle-harness/tests/boosting_parity.rs` — full boosting end-to-end.

Run on real gfx1100 with `--features rocm`. The default (CPU) merge gate must stay 0-failed.

**KNOWN FLAKY-TEST LESSON [VERIFIED: memory DEF-f8u-01]:** never compare two nondeterministic GPU f32 paths to each other at 1e-6 — `learner_parity_resident_equals_host_tree_on_hip` is pre-existing flaky for exactly this reason. Pin GPU trees to the **cpu f64 anchor** (structure bit-exact; leaf values within `ROCM_LEAF_VALUE_TOL=1e-5`), as fw1 (commit d82611b) did. Do NOT add a GPU-immediate-vs-GPU-deferred 1e-6 assertion as a gate; if a same-input drift check is wanted, use the spike's ABS 5e-6 / REL 1e-5 envelope as a sanity check only (NOT the parity gate).

## 4. Pitfalls / Gotchas

- **Handle lifetime when holding N out-handles before drain:** each `client.create_from_slice` Handle must outlive its launch and the final `read`. Holding them in a `Vec<Handle>` until `client.read(h_outs)` satisfies this. `ArrayArg::from_raw_parts(handle.clone(), len)` is the existing idiom (`Handle` is cheaply cloneable / ref-counted) — no aliasing issue since each launch writes a DISTINCT out-handle. [VERIFIED: histogram.rs usage + client.rs read signature]
- **N distinct zeroed out-buffers cost memory:** deferral REQUIRES N separate out-handles (Arm A reused one). For feats≥32 × (2·256) f64/f32 cells this is modest, but it is the trade the win pays for. Not a concern on the wired path (single launch, single out-handle).
- **Stream is in-order:** cubecl queues on a per-`StreamId` stream; deferral overlaps CPU-submit with GPU-execute but does NOT run kernels concurrently/out-of-order — so no numeric reordering, and also no extra parallelism beyond hiding the host round-trip. [VERIFIED: client.rs:85-104]
- **Autotune:** not exercised by these hand-written `#[cube(launch_unchecked)]` kernels (no `#[autotune]`); deferral does not interact with it. [ASSUMED — no autotune attributes seen in histogram.rs]
- **Keep the immediate-read path as fallback:** the launch-bound regime is NULL/negative for deferral (memory `gpu-lazy-dispatch-deferred-sync-win`, STATE.md). Any wired deferral must be gated and must retain the immediate path for small leaves.
- **`read` vs `read_one_unchecked` error handling:** `client.read(Vec<Handle>)` panics on failure (`.expect("TODO")`, client.rs:132) just like `read_one_unchecked`; behavior-equivalent for the existing unchecked call sites.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | No `#[autotune]` on the histogram kernels, so deferral does not interact with autotune | §4 | Low — visual scan of histogram.rs; if present, deferral still call-ordering-only |

## Open Questions

1. **Is the intended deliverable a code change at all, or a reconciliation FINDINGS note?**
   - What we know: the production leaf path is already one-launch/leaf + resident no-readback; the spike's slow baseline is test-only.
   - What's unclear: whether the user expects production code edits despite the win being structurally already captured.
   - Recommendation: surface this to the user before editing production kernels. The honest outcome may be "win already realized by the batched/resident collapse; optionally tidy the test-only path to use `client.read(Vec<Handle>)`." Do NOT re-introduce a per-feature loop to create a deferral seam.

## Sources

### Primary (HIGH confidence)
- `cubecl-runtime-0.10.0/src/client.rs` (installed crate source) — `read`/`read_one`/`read_one_unchecked`/`read_async`/`sync` signatures, stream/submit semantics.
- `Cargo.lock` — cubecl 0.10.0 pin.
- `crates/lgbm-compute/src/kernels/histogram.rs` (read) — production launchers, resident/batched collapse, `CubeCount::Static(num_features, p, 1)`.
- `crates/lgbm-compute/src/lib.rs` (read) — `RocmBackend::construct_histograms` / `build_leaf_histograms_raw` routing.
- grep across `crates/` — `.construct_histograms(` callers are tests-only; production read sites enumerated.
- `crates/lgbm-compute/examples/lazy_dispatch_ab.rs` (read) — the spike A/B; Arm A == test-only per-feature pattern, Arm B == deferred drain.
- `.planning/STATE.md` — q2z disposition (WIRE gated compute-bound×bins≥256×feats≥32), DEF-f8u-01 flaky lesson.

### Secondary (MEDIUM confidence)
- ctx7 `/tracel-ai/cubecl` docs — `client.read_one(output)` launch idiom, comptime specialization (corroborates the API; cubecl docs are thin on multi-handle `read`).

## Metadata

**Confidence breakdown:**
- cubecl read/sync API: HIGH — quoted from installed 0.10.0 source.
- Production seam location: HIGH — file:line verified + grep proves zero production callers of the per-feature path.
- Parity re-validation set: HIGH — test names grep-confirmed; DEF-f8u-01 lesson from project memory.

**Research date:** 2026-06-19
**Valid until:** 2026-07-19 (stable; cubecl pin frozen at 0.10.0)
