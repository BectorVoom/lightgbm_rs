---
quick_id: 260609-aqy
title: GPU host<->device boundary optimization analysis — bytemuck, arrow-rs, and boundary-copy costs
type: investigation
status: complete
date: 2026-06-09
scope: crates/lgbm-compute (the host<->device boundary only; dataset/ingest covered by 260609-9nu)
verdict_keywords: ADOPT / INVESTIGATE-FURTHER / REJECT
---

# GPU boundary optimization analysis (lgbm-compute)

## 1. Summary

The host<->device boundary in `lgbm-compute` is **already lean for the cast itself**.
cubecl's `CubeElement` trait does the zero-copy reinterpret in both directions: upload
via `T::as_bytes(slice)` fed to `client.create_from_slice(...)` (e.g.
`f32::as_bytes(grad)` at `histogram.rs:159`) and readback via `T::from_bytes(&bytes)`
(e.g. `f64::from_bytes(&bytes).to_vec()` at `histogram.rs:189`). There is **no
intermediate `Vec<u8>` serialization** on the hot path — `as_bytes` is a slice
reinterpret. The genuine, non-trivial costs at the boundary are three, none of which
bytemuck can remove:

1. **Host zero-allocs feeding output buffers** — the recurring `vec![0; n]` + upload
   pattern (~13 live launcher sites). MOST of these feed *accumulate/atomic* buffers
   that genuinely depend on zero-init; a *minority* feed *fully-overwritten* buffers
   where `client.empty()` could in principle skip the host alloc + upload.
2. **Per-element f32->f64 widening collects on readback** (`histogram.rs:392, 495, 606`)
   — these are *type conversions*, not reinterprets. bytemuck cannot remove them; they
   are semantically required (the learner pool is f64; the GPU atomic path is f32).
3. **Host concat/gather copies before upload** — the resident-column concat
   (`lib.rs:815`, `histogram.rs:857`) and the per-leaf `ord_g`/`ord_h`/`gathered_bins`
   gathers (`histogram.rs:459, 462-463, 567-568, 907-908, 1409-1410`). These are
   *gathers/concats* (data is reshaped), not casts — neither bytemuck nor arrow changes
   their cost.

**Headline verdicts:**

- **bytemuck: REJECT.** It is redundant with `CubeElement::as_bytes`/`from_bytes` for
  every cast on the boundary, adds no niche win here, and is already present only
  transitively (`Cargo.lock` line 310, version 1.25.0 — pulled by cubecl/wgpu deps, NOT
  a direct dependency in any workspace `Cargo.toml`). Adding it as a direct dep buys
  nothing.
- **arrow-rs at the boundary: REJECT.** Data reaching `create_from_slice` is already
  contiguous `&[T]`; an arrow `Buffer` would feed it no better and adds a second Arrow
  runtime to the build graph (`polars-arrow` is already present via pyo3-polars). The
  dataset/ingest assessment was settled by 260609-9nu (cited below, not re-derived).

The single defensible micro-opportunity is `client.empty()` for the **fully-overwritten
split out-cells** (single-feature kernels) — and even that is bounded near-zero by the
L3 finding and carries an annoying parity hazard if mis-applied. Recommendation:
**analysis-only, no implementation** (see §6).

---

## 2. The L3 reality check (frames every estimate below)

From `STATE.md` / project memory (`l3-on-gpu-fixhistogram-deferred.md`), the
device-resident histogram pool work (260608-nn7/oib/p90) proved:

> **the host<->device ROUND-TRIP was NOT the GPU bottleneck** (mixed win, small inputs
> regressed) — "profile before assuming the boundary is the bottleneck."

This is decisive for this whole analysis. Every opportunity below is a *boundary-copy*
reduction. The L3 work already moved the histogram VALUES off the round-trip (resident
pool, handle-in/handle-out subtract, on-GPU fix+compact) and the net was *mixed* — small
inputs got *slower*. Therefore:

- Shaving a host `vec![0; n]` alloc or a single upload is bounded by a cost that L3
  showed is **not** dominating wall-clock. Expected impact on training time:
  **negligible to zero**, and possibly *negative* on small inputs (the empty()/memset
  trade can lose to a warm allocator).
- The real wall-clock lever for this project is elsewhere (tree-construction-bound; see
  `perf-gap-vs-cpp-40-80x.md`). Do not let a boundary micro-opt consume a parity-risk
  budget.

No estimate below is allowed to claim more than "micro / likely in the noise."

---

## 3. Ranked opportunities

Ranked by `(value × likelihood) / risk`. All "value" is L3-bounded (micro at best).

| # | Opportunity | File:line | Value | Likelihood | Parity risk | Verdict |
|---|-------------|-----------|-------|------------|-------------|---------|
| O1 | `empty()` for fully-overwritten **split out-cells** (single-feature) | `split.rs:799-800`, `split.rs:1678-1679` | micro | med | LOW-but-sharp | INVESTIGATE-FURTHER |
| O2 | Per-element f32->f64 widening collects (note as unavoidable) | `histogram.rs:392, 495, 606` | n/a | n/a | n/a (type conv) | REJECT (not removable) |
| O3 | `empty()` for **accumulate/atomic** histogram out buffers | `histogram.rs:166-167, 291-292, 368-369, 470-471, 575-576, 904-905, 949-950`; `subtract.rs:99-100, 164-165, 211-212`; `partition.rs:230-231` | micro | low | HIGH (breaks parity) | REJECT |
| O4 | Host concat (resident columns) | `lib.rs:813-815`, `histogram.rs:855-857` | micro | low | none | REJECT (one-time; already L1-optimal) |
| O5 | Per-leaf host gathers `ord_g`/`ord_h`/`gathered_bins` | `histogram.rs:459, 462-463, 567-568, 907-908, 1409-1410` | micro | low | none | REJECT (gather, not a cast) |
| O6 | Adopt `bytemuck` for the cast | (whole boundary) | none | n/a | none | REJECT (redundant w/ CubeElement) |
| O7 | Adopt `arrow-rs` Buffer feeding `create_from_slice` | (whole boundary) | none | n/a | none | REJECT (data already contiguous `&[T]`) |

### O1 — `client.empty()` for fully-overwritten split out-cells — INVESTIGATE-FURTHER

- **Evidence:** `split.rs:797-800` (f64) and `split.rs:1676-1679` (f32) allocate a
  12-cell zeroed out buffer (`let out_len = 12usize; let zeros = vec![0.0f64; out_len];
  let h_out = client.create_from_slice(...)`).
- **Current code:** the single-feature split kernel **unconditionally writes every one
  of the 12 cells** — `out[ob + 0..=11] = ...` at `split.rs:372-383` (f64) and
  `out[0..=11] = ...` at `split.rs:605-616` (f32). Every cell is assigned (the
  "not-splittable" path still writes `out[ob] = is_splittable` = 0.0 and writes the
  remaining cells with their initial/sentinel values). Because the kernel WRITES (never
  `+=`) all 12 cells, the prior zero-init is not load-bearing for THIS kernel.
- **Proposed change:** replace the host `vec![0; 12]` + `create_from_slice` with
  `client.empty(12 * size_of::<f64>())` for the single-feature split launchers ONLY.
- **Expected impact (L3-framed):** saves one 12-element host alloc + one tiny upload per
  split-finding call. This is microscopic in absolute terms and, per L3, the round-trip
  is not the bottleneck — **expected wall-clock effect ~0**. The split out buffer is
  12 cells; the upload is ~96 bytes. This is not worth a parity-risk budget on its own.
- **Parity risk (LOW but SHARP):** the *single-feature* kernel is safe (all 12 cells
  written). BUT the **multi-feature** split kernel writes only its own window
  `out[f*12 .. f*12+12]` (`split.rs:942` doc, `split.rs:1258` SAFETY note). If any
  feature window is somehow not visited, `empty()` would surface a STALE pooled buffer
  into that window — a silent wrong-split parity break. The hazard is exactly the one
  the `histogram.rs:161-165` comment documents for the accumulate buffers. Getting the
  "which buffer is fully overwritten" judgment wrong is a parity bug, not a perf
  regression.
- **Verdict: INVESTIGATE-FURTHER** — technically safe for the single-feature path, but
  the value is L3-bounded to noise and the upside does not justify introducing an
  `empty()`/overwrite invariant that a future multi-feature refactor could quietly
  violate. Not recommended for implementation now (§6).

### O2 — Per-element f32->f64 widening collects — REJECT (not removable)

- **Evidence/current code:** `histogram.rs:392`, `histogram.rs:495`, `histogram.rs:606`
  all end with `Ok(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect())`
  (and the degenerate no-launch path at `histogram.rs:771`). These appear in the
  parallel-atomic and batched/resident f32 build launchers — the GPU accumulates the
  histogram in f32 (gfx1100 has no f64 atomics), then the result is widened to f64 for
  the learner's f64 pool.
- **Why it is not a boundary cast you can reinterpret:** `f32::from_bytes(&bytes)` IS the
  zero-copy reinterpret (cubecl). The `.map(|&x| f64::from(x)).collect()` that follows is
  a **numeric widening** (4 bytes -> 8 bytes, value-preserving conversion), which by
  definition allocates a new `Vec<f64>` and touches every element. bytemuck operates on
  same-size/same-layout reinterprets; it **cannot** turn an f32 buffer into an f64 buffer
  without the same per-element conversion.
- **Parity note:** this widening is *required* and is part of the documented ROCm ~1e-6
  contract (the f32 atomic accumulation diverges from the cpu f64 anchor at ~1e-6;
  widening to f64 afterward does not recover precision but is needed for the f64 pool).
  Removing or reordering it would be a contract change, not an optimization.
- **Verdict: REJECT** — note it as a genuine allocation cost on the f32 GPU path, but it
  is semantically necessary and not addressable by bytemuck or arrow.

### O3 — `empty()` for accumulate/atomic histogram output buffers — REJECT

- **Evidence:** the `vec![0; out_len]` + upload feeding histogram CONSTRUCT outputs:
  `histogram.rs:166-167` (f64 construct), `291-292` (f32 construct), `368-369` (parallel
  f32 atomic), `470-471` (batched), `575-576` (resident), `904-905` and `949-950`
  (build_fix_compact_resident raw + f64 out); plus `subtract.rs:99-100, 164-165, 211-212`
  and `partition.rs:230-231`.
- **Why these MUST stay zeroed:** the construct kernels do `out[ti] += ...`
  (`histogram.rs:68-69`) and the parallel kernels do `out[ti].fetch_add(...)`
  (`histogram.rs:338-339, 427-428, 530-531`) — **accumulate from zero**. The in-code
  comment at `histogram.rs:161-165` already states it explicitly: *"`client.empty`
  returns UNINITIALIZED device memory from the pool — it may recycle a prior launch's
  buffer, so a fresh launch would fold on top of stale values."* Using `empty()` here
  WITHOUT a device-side zero/memset would fold gradients onto stale pooled data ->
  wrong histogram -> parity break (and the cpu f64 anchor is the bit-exact MERGE GATE).
  - subtract (`subtract.rs`) and partition (`partition.rs:230`) out buffers are
    element-wise *written* (`out[i] = parent[i]-child[i]`, `route[i] = ...`), so they are
    NOT accumulate — but their `vec![0; n]` cost is identical-micro and the value is L3-noise;
    grouping them under REJECT keeps the safe default.
- **Parity risk: HIGH** for the accumulate/atomic buffers — this is the exact hazard the
  plan warns about. Do **not** convert any accumulate-from-zero / atomic buffer to
  `empty()` without issuing a device-side zero first (which would re-introduce a cost,
  likely net-negative per L3).
- **Verdict: REJECT** — leave the explicit zero-init in place; it is correctness, not
  waste.

### O4 — Host concat of resident feature columns — REJECT

- **Evidence/current code:** `lib.rs:813-815` and `histogram.rs:855-857` build one
  feature-major buffer via `concat.extend_from_slice(col)` then upload once
  (`u32::as_bytes(&concat)`).
- **Assessment:** this is a one-time-per-train upload (the L1 win itself —
  `l3-on-gpu-fixhistogram-deferred.md`). It is a memory *reshape* (separate column
  slices -> one contiguous buffer), not a cast. bytemuck/arrow do not eliminate a
  concat; arrow's contiguity guarantee does not help because the columns arrive as
  separate `&[u32]` slices that must be gathered regardless. It is already amortized
  across the whole train.
- **Verdict: REJECT** — already optimal; nothing to do.

### O5 — Per-leaf host gathers (`ord_g`/`ord_h`/`gathered_bins`) — REJECT

- **Evidence/current code:** `histogram.rs:459` (`gathered_bins.push(bins[row])`),
  `462-463`, `567-568`, `907-908`, `1409-1410`
  (`leaf_rows.iter().map(|&r| gradients[r as usize]).collect()`).
- **Assessment:** these are *gathers* — pulling the leaf's rows out of the full
  grad/hess/bins arrays by index. The resident path (260608-nn7 L1) already moved the
  big `[num_features × rows]` bin gather ON DEVICE (the kernel gathers from the resident
  column at `histogram.rs:528`); what remains host-side is the small per-leaf
  `ord_g`/`ord_h` gather and the `leaf_rows` index upload, which L1 deliberately kept
  small. A gather is index-driven data movement; bytemuck (reinterpret) and arrow
  (contiguity) change nothing about it.
- **Verdict: REJECT** — already at the L1-optimized shape; the residual gather is small
  and the L3 finding says the round-trip is not the bottleneck.

---

## 4. Verdict: bytemuck vs CubeElement::as_bytes — REJECT

**bytemuck is REDUNDANT with cubecl for everything at this boundary.**

- **The cast is already zero-copy via cubecl.** Upload uses `T::as_bytes(&[T]) -> &[u8]`
  (`CubeElement::as_bytes`, e.g. `f32::as_bytes(grad)` at `histogram.rs:159`,
  `u32::as_bytes(&concat)` at `lib.rs:820` with the trait explicitly brought into scope).
  Readback uses `T::from_bytes(&[u8]) -> &[T]` (`histogram.rs:189`). Both are slice
  reinterprets with no allocation — exactly what `bytemuck::cast_slice` /
  `bytemuck::pod_collect_to_vec` would provide. There is no `to_vec()`-of-bytes
  intermediate to eliminate.
- **No struct-of-arrays niche here.** The boundary only ever moves flat homogeneous
  slices (`&[f32]`, `&[f64]`, `&[u32]`, `&[i32]`). There is no packed `#[repr(C)]` struct
  being reinterpreted to a field-slice where bytemuck's derive would add value.
- **No non-CubeElement type.** Every element type crossing the boundary already
  implements `CubeElement`. bytemuck would only matter for a type cubecl can't reinterpret
  — none exists here.
- **No intermediate-Vec elimination.** The one place a `Vec` is genuinely allocated on
  readback is the **f32->f64 widening** (O2), which is a *conversion*, not a reinterpret —
  bytemuck cannot remove it (different element size).
- **Presence:** bytemuck is already in `Cargo.lock` (line 310, v1.25.0) **transitively**
  (via cubecl/wgpu's dependency tree), but is **not** a direct dependency in ANY workspace
  `Cargo.toml` (verified: `grep -rn bytemuck --include=Cargo.toml` returns nothing for the
  workspace crates). Promoting it to a direct dep adds API surface and a maintenance/parity
  liability for zero functional gain.

**Verdict: REJECT** — `CubeElement::as_bytes`/`from_bytes` already deliver the zero-copy
cast; bytemuck adds nothing at this boundary and there is no struct/SoA/non-CubeElement
niche where it would help.

## 5. Verdict: arrow-rs at the GPU boundary — REJECT

NARROW assessment (boundary only). The dataset/storage/binning/ingest question was settled
by **260609-9nu** (`260609-9nu-FINDINGS-adopt-arrow-rs.md`): NO win for internal storage
(the 4-bit-packed `DenseBin` is more compact than arrow's byte-per-element arrays and is a
hard parity gate), arrow's validity-bitmap null model mismatches the project's
NaN-as-missing parity semantics, and the tree already depends on `polars-arrow` (not
`arrow-rs`) — arrow-rs is earmarked ONLY for future v2 `ING-*` Parquet/CSV ingestion. That
conclusion is **cited, not re-derived**.

For the **GPU boundary specifically**:

- The data reaching `client.create_from_slice(...)` is already a **contiguous `&[T]`** in
  every case (a `Vec<f32>`/`Vec<f64>`/`Vec<u32>` or a slice of one). `create_from_slice`
  consumes `&[u8]` from `T::as_bytes` over that contiguous slice.
- An arrow `Buffer` (64-byte aligned, contiguous) would feed `create_from_slice` **no
  better**: the existing `Vec<T>` is already contiguous, and cubecl/wgpu's own upload
  staging handles alignment — there is no measured alignment fault to fix, and the L3
  work showed the upload is not the bottleneck anyway. The arrow alignment guarantee is a
  solution to a problem this boundary does not have.
- Adopting arrow-rs would add a **second Arrow runtime** to the build graph alongside the
  `polars-arrow` already present (`Cargo.lock:2927`), for no new capability on the GPU
  path.

**Verdict: REJECT** at the GPU boundary. (arrow-rs's only legitimate future home remains
v2 file ingestion, per 260609-9nu — out of scope here.)

---

## 6. Recommendation

**Analysis-only — no implementation recommended.** This is the L3-consistent outcome.

Ranked disposition:

1. **bytemuck — REJECT (do not adopt).** Redundant with `CubeElement::as_bytes`/
   `from_bytes`; no SoA/non-CubeElement/Vec-elimination niche at this boundary; already
   transitive-only. Promoting it to a direct dep is pure liability.
2. **arrow-rs at the boundary — REJECT (do not adopt).** Data is already contiguous
   `&[T]`; an arrow `Buffer` feeds `create_from_slice` no better and duplicates the
   `polars-arrow` runtime. (Dataset/ingest verdict deferred to 260609-9nu; arrow-rs
   earmarked only for v2 `ING-*`.)
3. **f32->f64 widening collects (O2) — REJECT as an optimization target.** Necessary
   type conversions on the f32 GPU path; not removable by any cast library. Documented as
   a real-but-required allocation cost.
4. **Accumulate/atomic zero-init (O3) — KEEP AS-IS.** The explicit `vec![0; n]` + upload
   is correctness, not waste; converting to `empty()` would break the bit-exact cpu merge
   gate. The `histogram.rs:161-165` comment must stay.
5. **Concat/gather copies (O4, O5) — REJECT.** Already at the L1-optimized shape; gathers
   are not addressable by bytemuck/arrow.
6. **Split out-cell `empty()` (O1) — INVESTIGATE-FURTHER, NOT NOW.** The single-feature
   split kernel fully overwrites all 12 cells, so `empty()` is *technically* safe there —
   but the value is L3-bounded to wall-clock noise (~96-byte buffer, and the round-trip is
   not the bottleneck), while it introduces an "every cell must be written" invariant that
   a future multi-feature refactor could silently violate into a parity bug. **No
   qualifying trivially-small, zero-parity-risk, parity-neutral win exists** for the
   optional Task 2 — O1 is the closest candidate and it fails the "zero-parity-risk" bar
   (the invariant is sharp) and the "worth it" bar (L3 noise). Therefore Task 2 is
   correctly skipped.

**Bottom line:** nothing here threatens the f32 ~1e-6 / cubecl-cpu f64-fold bit-exact
contract, and nothing here is worth implementing. The boundary is already lean for the
cast; the only non-trivial costs (f32->f64 widening, gathers, zero-init) are either
required or already at their optimized shape. The real perf lever for this project is
tree-construction, not the GPU boundary (`perf-gap-vs-cpp-40-80x.md`).

---

## Parity Verdict (rollup)

Every opportunity above carries an explicit verdict: O1 INVESTIGATE-FURTHER, O2 REJECT,
O3 REJECT, O4 REJECT, O5 REJECT, O6 (bytemuck) REJECT, O7 (arrow-rs) REJECT. **No
recommendation weakens or contradicts the f32 ~1e-6 / cubecl-cpu f64-fold bit-exact merge
gate.** The one accumulate/atomic hazard (O3) is explicitly flagged as a parity break if
`empty()` were applied without a device-side zero, and is left untouched.
