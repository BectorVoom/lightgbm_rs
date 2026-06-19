# Quick 260619-p93 — Research: Plane (warp) aggregation of the f32-atomic histogram scatter

**Researched:** 2026-06-19
**Domain:** CubeCL 0.10 Plane (warp/subgroup) API; warp-aggregated atomics for the ROCm histogram scatter kernel
**Confidence:** HIGH on the API surface (read from cubecl-core-0.10.0 source) and the parity contract; HIGH on the expected-win assessment (corroborated by 3 prior in-repo spike findings).

## Summary

The task is to warp-aggregate the global f32 `fetch_add`s in the ROCm histogram scatter kernel(s) using CubeCL's Plane API, to cut global-atomic contention. CubeCL 0.10 exposes the full primitive set required for correct warp-aggregated atomics (`plane_ballot`, `plane_elect`, `plane_broadcast`, `plane_sum`, `UNIT_POS_PLANE`, `PLANE_DIM`), and the repo already has host-side `Plane::Ops` capability gating wired (`probe_capabilities` / `Capabilities.has_plane` / `plane_size` / `ReducePath`). So the mechanism is fully available.

The crux is correctness and value. A naive `plane_sum` is **WRONG** here: each lane's destination `bin` is data-dependent and divergent across the plane, so summing all lanes' contributions and writing them to one bin corrupts the histogram. Correct warp aggregation needs **same-bin grouping within the plane** (ballot/match → reduce per group → one elected lane issues the global atomic per distinct bin).

**Primary recommendation: this is most likely NULL on the GLOBAL-atomic kernel and a MARGINAL-at-best, parity-risky change on the LDS kernel — do NOT ship it speculatively. Build the A/B harness FIRST (mirror `launch_unchecked_ab.rs`), and only wire a plane variant if the interleaved bench shows a robust, sign-stable, spread-separated win in a real bin-count regime.** Three independent prior in-repo findings already point to NULL (see "Expected win"). The valid, honest deliverable of the implementation step may well be "benched, NULL, kept as a rocm-gated primitive (not wired)" — exactly as 260619-ngo/ol8 concluded for their levers.

## User Constraints (from the task brief)

- **Numerical contract (CLAUDE.md):** the f32 ROCm path must stay within ~1e-6 of the CPU f64 anchor (the gate the rocm-atomic tests use is ABS 5e-6 / REL 1e-5). NOT bit-exact — bit-exactness is the CPU f64 anchor's job. The CPU f64 anchor kernels (`construct_hist_kernel`, `construct_hist_kernel_f32`) must stay byte-unchanged.
- **CPU-only build must never emit plane codegen** — all plane work stays behind `#[cfg(feature = "rocm")]` and comptime gating.
- **Honesty mandate:** do not manufacture a win. A credible "likely NULL, here's the bench that proves it" is a valid finding (per the task brief and the ol8/ngo precedent).
- **Twin-sync contract:** if a `_checked`/baseline twin is added to a bench, any later edit to the shipped kernel must be mirrored into the twin (the same-input assert is the runtime guard).

## CubeCL 0.10 Plane primitives (verified from source)

All in `cubecl-core-0.10.0/src/frontend/plane.rs` and `topology.rs`, re-exported via `cubecl::prelude::*`. `[VERIFIED: cubecl-core-0.10.0 source]`

| Primitive | Signature | Use in warp-aggregated histogram |
|-----------|-----------|-----------------------------------|
| `plane_sum<E>(value: E) -> E` | reduce-sum across all lanes in the plane | the naive (WRONG-alone) reduction; correct only within a same-bin group |
| `plane_ballot(pred: bool) -> Vector<u32, Const<4>>` | per-lane predicate → 128-bit mask (4×u32; vector size is always 4 even on wave32 — index by runtime `PLANE_DIM`) | same-bin grouping: ballot which lanes share my bin |
| `plane_elect() -> bool` | true for the lowest active `plane_unit_id` | elect one lane to issue the global atomic |
| `plane_broadcast<E>(value: E, index: u32) -> E` | broadcast lane `index`'s value to all lanes (const index; use `plane_shuffle` for dynamic) | broadcast the group leader's bin/sum |
| `plane_shuffle<E>(value, src_lane: u32) -> E` | each lane reads `src_lane`'s value (dynamic index) | read another lane's bin for matching |
| `plane_shuffle_xor / _up / _down<E>(value, n) -> E` | butterfly / shifted shuffles | building a manual same-bin reduction if needed |
| `plane_all(bool) / plane_any(bool) -> bool` | predicate across plane | fast "do all lanes share one bin?" check |
| `plane_inclusive_sum / exclusive_sum<E>(value) -> E` | prefix sums | segmented-scan style aggregation |
| `PLANE_DIM` (builtin u32) | lanes per plane (32 on gfx1100 wave32) | runtime plane width; index the ballot vector |
| `UNIT_POS_PLANE` (builtin u32) | lane id within the plane | per-lane identity for matching/elect |

**Notable gap:** there is **no `plane_match_any` / `plane_partition`** primitive (CUDA's `__match_any_sync`). Same-bin grouping must be built manually from `plane_ballot` + bit ops, or via an iterative `plane_broadcast`/`plane_shuffle` loop over distinct bins. This raises the implementation cost and weakens the value case.

**Host-side gate (already in the repo):** `crates/lgbm-compute/src/runtime.rs` — `probe_capabilities()` sets `has_plane = client.features().plane.contains(Plane::Ops)` and `plane_size = client.properties().hardware.plane_size_max`. `capability.rs` asserts cpu=`has_plane:false/size:1`, gfx1100=`true/32`. Reuse this; do NOT re-roll the probe.

## The correctness crux (divergent bins)

In `construct_hist_kernel_atomic_f32` (histogram.rs ~389) each lane does `out[binned[idx]*2].fetch_add(grad[idx])` where `bin` differs across adjacent rows. A whole-plane `plane_sum` is therefore **incorrect**. The correct warp-aggregated-atomic algorithm (classic CUDA pattern, e.g. NVIDIA "warp-aggregated atomics") is:

1. Each lane has its target `bin`.
2. **Group lanes by equal `bin`** within the plane. With no `match_any`, do it by leader iteration: while any lane is unclaimed, `leader_bin = plane_broadcast(my_bin, first_unclaimed_lane)`; `same = plane_ballot(my_bin == leader_bin)`; mark those lanes claimed.
3. **Reduce** each group's grad/hess (`plane_sum` masked to the group, or popcount-weighted via shuffles).
4. **One elected lane per group** (`plane_elect` among the group, or the leader lane) issues a **single** `out[bin*2].fetch_add(group_sum)` — collapsing up to `PLANE_DIM` global atomics per bin into one.

This is correct but non-trivial, and the f32 group-reduction changes accumulation order vs the sequential atomic path (parity note below).

## Expected win — HONEST assessment: **likely NULL on global-atomic, MARGINAL/risky on LDS**

Three independent prior in-repo findings converge on NULL, and the algorithm's value depends entirely on the in-plane collision rate:

- **Collision-rate reasoning (the decider):** warp aggregation only helps when many of the 32 lanes in a wave target the *same* bin. For uniformly distributed bins, the expected number of distinct bins hit by 32 lanes is ≈ `B·(1 - (1-1/B)^32)`. At **B=256** that is ~30 distinct bins out of 32 lanes → almost no collisions → the aggregation overhead (ballot loop + shuffles) is pure cost with nearly nothing to amortize → **NULL/regression**. At **B=16** lanes collide heavily (~13 distinct bins, ~2.4 lanes/bin avg) → some atomics collapse, but real bin distributions are skewed (most_freq_bin dominates), which both *helps* (the dominant bin collapses a lot) and *hurts* (every wave serializes on that one hot bin regardless). So any win is confined to the **low-bin (16/64) regime**, exactly where the histograms are already cheap.
- **gpu-hist-levers-closed memory (decisive):** "Register row-batching — NULL. **At saturating occupancy the bottleneck is LDS atomic contention, not load latency.**" and "the build is **atomic-contention bound**, so latency/width/overhead levers don't move it." Warp aggregation reduces atomic *count* but if the bottleneck is contention on the *hot bins* (most_freq_bin), collapsing 32 uniform-random adds to ~30 does not relieve the hot-bin serialization.
- **spike-006 memory:** the GPU build is "**atomic-contention/scattered-read-latency bound (234 Mreads/s), NOT bandwidth-bound**." The scattered indirect `binned[idx]` read latency is paid regardless of aggregation.
- **ol8 finding:** `launch_unchecked` on the f32-atomic kernel was NULL precisely because "global-atomic contention / memory latency dominates" — the same wall this lever runs into.

**Verdict matrix (expected, to be confirmed by bench):**

| Target | B=16/64 | B=256 | Overall expectation |
|--------|---------|-------|---------------------|
| Global-atomic kernel (`construct_hist_kernel_atomic_f32`) | possible marginal | NULL/regression | **likely NULL** |
| LDS kernel (`construct_hist_kernel_lds_f32`) — plane-aggregate before the LDS atomic | marginal at best | NULL | **likely NULL, higher parity risk** |

**Highest-value, lowest-parity-risk insertion point IF anything:** plane-aggregate *before the LDS atomic* inside `construct_hist_kernel_lds_f32` is the most defensible target only at low bin counts — but the LDS path already cut global atomics to `CUBE_COUNT*2*num_bin` and its hot path is LDS atomics + `sync_cube()`, which plane aggregation does not touch. **Recommend benching the global-atomic kernel first (simplest, no LDS interaction) to get a clean NULL/non-NULL signal before touching the wired LDS path.** Do NOT touch the wired training path (`construct_leaf_hist_resident_lds_kernel`) speculatively.

## Recommended implementation approach (only past a positive bench)

1. **Add a `_plane` kernel variant**, comptime-gated, beside `construct_hist_kernel_atomic_f32`:
   ```rust
   #[cfg(feature = "rocm")]
   #[cube(launch_unchecked)]
   pub fn construct_hist_kernel_atomic_f32_plane(
       binned: &Array<u32>, grad: &Array<f32>, hess: &Array<f32>,
       out: &mut Array<Atomic<f32>>,
       #[comptime] use_plane: bool,   // comptime → no GPU-side branch; CPU build never sees plane codegen
   ) {
       let idx = ABSOLUTE_POS;
       if idx < binned.len() {
           let bin = binned[idx] as usize;
           if use_plane {
               // same-bin group → plane_sum the group's grad/hess → plane_elect issues ONE fetch_add per bin
           } else {
               out[bin*2].fetch_add(grad[idx]);
               out[bin*2 + 1].fetch_add(hess[idx]);
           }
       }
   }
   ```
   The `#[comptime] use_plane: bool` generates a distinct kernel binary with no device branch (manual "Feature Specialization with Comptime Flags") — and the CPU-only build, which never compiles the `rocm` cfg, never emits plane codegen.
2. **Host gate:** only launch the `use_plane=true` variant when `probe_capabilities(client).has_plane` (reuse the existing probe). Fall back to the shipped non-plane launcher otherwise.
3. **Keep it a PRIMITIVE, not wired**, until the A/B proves a robust win — same disposition pattern as ngo/ol8/j9t.
4. **Do NOT make `num_bin` comptime** (gpu-hist-levers-closed: re-introduces the multi-binary cost the repo avoids).

## Parity assessment

- Plane group-reduction is an f32 **tree reduction**; the shipped path is a nondeterministic f32 atomic sequence. Both are non-deterministic f32 accumulations measured against the **CPU f64 anchor** at ABS 5e-6 / REL 1e-5 — neither is bit-exact, and the contract was *designed* for exactly this f32 reordering (D-03a, 04-ROCM-GAPS). A tree reduction is typically **more** accurate than a long sequential f32 chain, so the change is expected to stay well inside the existing envelope.
- **Risk flag:** at large leaf row counts the per-bin sum grows and f32 magnitude effects appear (the row-partition path already documents 4e-7→~2e-5 rel drift at large leaves). Pin any plane test to the **CPU f64 anchor** (never compare two non-deterministic GPU f32 paths to each other — see DEF-f8u-01: pin both to the f64 anchor, leaf values within a 1e-5 f32 envelope).

## A/B verification plan (mirror `launch_unchecked_ab.rs`)

Build `crates/lgbm-compute/examples/plane_aggregate_ab.rs`, `--features rocm`, MEASUREMENT-ONLY (production untouched):

- **Interleaved arms:** baseline `construct_hist_kernel_atomic_f32` (shipped) vs the `_plane` variant, interleaved per timed iter so thermal/clock drift hits both equally.
- **Regimes:** launch-bound (small n, e.g. 2_048) and compute-bound (200_000), each swept over **bins = [16, 64, 256]** — the collision-rate axis is the whole point, so the bin sweep is the primary independent variable.
- **Method:** WARMUP discarded, MEDIAN + p25/p75 spread, device sync (read-back) forced inside each timed call, `delta% = (baseline - plane)/baseline*100`.
- **≥2 process runs:** a delta within the p25/p75 spread or whose sign flips across runs is SUB-NOISE / NULL (the ol8 disposition rule).
- **Same-input parity assert:** assert the plane and baseline histograms agree within ABS 5e-6 / REL 1e-5 (the f32 envelope) — this also doubles as the twin-drift guard.
- **GPU-vs-CPU-f64-anchor re-pin:** separately assert the plane variant's output matches `construct_histograms_cpu` (the f64 anchor) within the rocm gate, at representative shapes, so the reordering is proven inside the contract (not just consistent with the baseline GPU path).

## Pitfalls

1. **Divergent-bin correctness:** naive `plane_sum` over the whole plane corrupts the histogram. Must group by equal bin first (ballot/elect). The #1 way to ship a silently-wrong kernel here.
2. **No `match_any` primitive in 0.10:** same-bin grouping is a manual ballot+shuffle loop; cost scales with distinct bins per plane → at 256 bins the loop overhead can exceed the atomics it saves.
3. **Hot-bin serialization persists:** aggregation collapses *uniform* collisions but every wave still serializes on `most_freq_bin`; contention-bound profile (gpu-hist-levers-closed/spike-006) means the lever may not move the wall.
4. **Parity / accumulation order:** f32 tree reduction ≠ sequential atomics; stay inside ABS 5e-6 / REL 1e-5; pin tests to the CPU f64 anchor, never GPU-f32-to-GPU-f32.
5. **CPU-only build must not break:** all plane code under `#[cfg(feature = "rocm")]`; `use_plane` is `#[comptime]` so no device branch and no CPU codegen. Verify `cargo build -p lgbm-compute` (no rocm) stays green.
6. **wave32 vs wave64:** gfx1100 is wave32 (`plane_size=32`), but `plane_ballot` always returns 4×u32 (128 bits) — index it by runtime `PLANE_DIM`, never assume 32. Do not hardcode the plane width.
7. **Twin-sync:** the bench baseline duplicates the shipped kernel body — mirror any future edit, guarded by the same-input assert (load-bearing per ol8).
8. **Don't re-explore closed levers:** gpu-hist-levers-closed already closed row-batching/packing/16-bit/launch_unchecked. This plane lever is the one genuinely un-benched contention lever — but the prior findings strongly predict NULL, so bench before building anything load-bearing.

## Assumptions Log

| # | Claim | Risk if wrong |
|---|-------|---------------|
| A1 | gfx1100 maps `plane_sum`/`plane_ballot`/`plane_elect` to real wave32 hardware instructions via cubecl-hip 0.10 (memory says Plane YES on gfx1100; not bench-confirmed for ballot/elect specifically) | If ballot/elect lower poorly on ROCm, the aggregation overhead grows and NULL becomes regression — the A/B catches it |
| A2 | Real histogram bin distributions are skewed enough that collisions exist at low bin counts but are rare at 256 | If distributions are more uniform than assumed, even the low-bin win evaporates → confirm via the bin-sweep A/B |

## Sources

### Primary (HIGH)
- `cubecl-core-0.10.0/src/frontend/plane.rs`, `topology.rs` — exact plane primitive signatures + builtins `[VERIFIED]`
- `crates/lgbm-compute/src/runtime.rs`, `tests/capability.rs` — existing `Plane::Ops` host gate `[VERIFIED]`
- `crates/lgbm-compute/src/kernels/histogram.rs` — the atomic + LDS kernels `[VERIFIED]`
- Context7 `/tracel-ai/cubecl` — comptime feature-specialization + `client.features().plane.contains(Plane::Ops)` pattern `[CITED]`

### Secondary (HIGH, in-repo evidence)
- memory `gpu-hist-levers-closed` — atomic-contention bound; row-batching NULL `[VERIFIED]`
- memory `cuda-mirror-kernel-slower-than-cpu`, spike-006 (STATE.md) — scattered-read/atomic-contention bound, not bandwidth `[VERIFIED]`
- `260619-ol8-FINDINGS.md` — launch_unchecked NULL on atomic kernel (contention dominates) `[VERIFIED]`
- `launch_unchecked_ab.rs` — interleaved A/B harness pattern to mirror `[VERIFIED]`

## Metadata
- API surface: HIGH (read from pinned 0.10.0 source). Expected-win: HIGH-confidence NULL prediction (3 corroborating findings) but UNBENCHED for this specific lever. Parity: HIGH.
- Valid until: stable while cubecl pinned at 0.10.0.
