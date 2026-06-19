# Quick 260619-nrw: Reduce overhead in production GPU histogram kernels — Research

**Researched:** 2026-06-19
**Domain:** cubecl 0.10.0 launch/codegen overhead levers applied to the lightgbm_rs production GPU histogram kernels
**Confidence:** HIGH (cubecl API verified via Context7 + the in-repo mirror precedent; kernel inventory read directly from source)

## Summary

cubecl is **pinned at 0.10.0** (`Cargo.toml:25`, `Cargo.lock` confirms; lgbm-compute uses `features=["cpu"]`, `rocm = ["cubecl/hip"]`). The cubecl manual documents three overhead levers relevant here: (1) `launch_unchecked` to drop in-kernel bounds-check codegen, (2) `#[comptime]` parameters to monomorphize hot scalars and eliminate runtime branches/loops, and (3) CubeDim/CubeCount occupancy tuning. `launch_unchecked` is **already applied to the CUDA-mirror kernel only** (`construct_hist_cuda_mirror_kernel`, launched at :1191/:1329 with a fully-worked SAFETY contract). **No production kernel uses it.**

The single cheapest, safest, proven-in-repo lever is to sweep the production launchers from `::launch` to `::launch_unchecked`. mwr already proved on the mirror that this is **strictly numerics-preserving** (it only removes the per-access bounds-check branch the manual emits; scatter order and f32-atomic accumulation are byte-unchanged) and supplied the host-validation contract template to copy. Eight production kernels at seven launch sites are in scope.

**Primary recommendation:** Stage 1 — sweep all production launchers to `launch_unchecked`, copying the mirror's per-kernel SAFETY contract (already-present V5 validation discharges it); re-pin each to the CPU f64 anchor. Stage 2 — `#[comptime]` the per-kernel constants that gate hot loops/branches (`lds_len`/`feat_len`, `num_data` stride, the LDS-vs-naive bin cap). Stage 3 (optional, order-changing) — only if a kernel-level bench shows a win. Re-pin to the CPU f64 anchor after EVERY stage; never GPU-vs-GPU (DEF-f8u-01).

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **ALL production histogram kernels** in `histogram.rs` are in scope (LDS build, atomic, batched, resident, resident-LDS, fix-compact, fused) — NOT just the wired resident path.
- The **CUDA-mirror primitive is OUT of scope** for further squeezing (ngo closed it; leave as primitive). It may serve as a reference for which manual techniques transfer.
- **Compute restructuring is ALLOWED** — accumulation/reduction-order changes that may shift f32 results are permitted, *provided* they stay within the ~1e-6 parity envelope vs the CPU f64 anchor. Broader than mwr's strictly-numerics-preserving scope.
- Every restructured kernel **MUST be re-pinned to the CPU f64 anchor (GPU-vs-CPU-f64-anchor, NEVER GPU-vs-GPU — DEF-f8u-01)**. If any change pushes a cell past tolerance, the tolerance review is part of this task: document the residual, do not silently weaken a gate without flagging it.
- "refer cubecl manual" is a **hard requirement**: this research pass pulled current cubecl launch/codegen/overhead guidance before planning (done — see Sources).
- Process: quick `--research --validate` → planner → plan-checker → executor → verifier.

### Claude's Discretion
- Ordering of techniques within the staged sweep; which `#[comptime]` params are worth monomorphizing; whether to attempt any Stage-3 order-changing restructure at all (the data so far says the LDS kernel is already at/near optimal — ngo).

### Deferred Ideas (OUT OF SCOPE)
- Further squeezing the CUDA-mirror primitive; wiring the mirror as a production path (ngo closed both).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NRW-01 | Sweep production launchers `::launch` → `::launch_unchecked` with host-validation SAFETY contracts | Mirror precedent (:1174-1203) + per-kernel contract table below |
| NRW-02 | `#[comptime]`-specialize hot scalar params where it removes a runtime branch/loop bound | cubecl comptime API verified (Sources); candidate params in the kernel table |
| NRW-03 | Re-pin every changed kernel to the CPU f64 anchor; document residuals; flag any tolerance change | Parity pattern from `rocm_cuda_mirror.rs` (ABS 5e-6 / REL 1e-5) |

## cubecl 0.10.0 Overhead Levers (verified API)

### Lever 1 — `launch_unchecked` (drops in-kernel bounds-check codegen)

`#[cube(launch)]` emits a per-access dynamic bounds check on every device array load/store. `#[cube(launch_unchecked)]` generates an **`unsafe`** launch fn that omits those checks — "the compiler can lower loads and stores without dynamic bounds checks in the hot loop while retaining the same safe surface" [CITED: cubecl docs]. The kernel body is identical; **only the safety wrapper changes**, so it is **numerics-preserving** (no order/precision change). The change is at TWO sites per kernel:

```rust
// 1. kernel attribute
#[cube(launch_unchecked)]              // was #[cube(launch)]
pub fn construct_hist_kernel_lds_f32(/* unchanged body */) { ... }

// 2. launch site (already inside an `unsafe { }` block in every launcher)
construct_hist_kernel_lds_f32::launch_unchecked(   // was ::launch
    client, CubeCount::Static(..), CubeDim::new_1d(256),
    /* identical args */
);
```

[VERIFIED: Context7 /tracel-ai/cubecl — `gelu_array::launch_unchecked` example shows the exact `#[cube(launch_unchecked)]` + `::launch_unchecked` pairing inside `unsafe { }`]

**The contract the caller must discharge (per kernel):** every device array index reachable in the kernel must be proven in-range by host-side validation BEFORE the launch. The mirror's worked contract (:1174-1189) is the template: enumerate each device access (`data[col+idx]`, `grad[idx]`, `out[base+m]`, `sub[bin*2+1]`) and cite the host check that bounds it. **Every production launcher already runs full V5 validation and already wraps the launch in `unsafe { }` with a SAFETY comment** — so the work is: switch the two tokens + extend the existing SAFETY comment with the per-access enumeration. No new validation logic is needed (the launchers already validate; the bound proofs already exist in prose).

### Lever 2 — `#[comptime]` specialization (monomorphize hot scalars, kill runtime branches)

A `#[comptime]` parameter is baked into the kernel binary at first-compile, enabling "loop unrolling, shape specialization … without runtime cost" and "comptime `if` blocks generate separate kernel variants without GPU-side branching" [CITED: cubecl docs]. API:

```rust
#[cube(launch_unchecked)]
fn k<F: Float>(input: &Array<F>, output: &mut Array<F>,
                #[comptime] use_plane: bool,        // baked at compile
                #[comptime] end: Option<usize>) {
    if use_plane { /* compiled only when true */ } else { for i in 0..end.unwrap() {..} }
}
// launch passes the comptime value as an ordinary trailing arg:
k::launch_unchecked::<F, R>(&client, count, dim, ArrayArg::.., ArrayArg::.., has_plane, Some(n));
```

[VERIFIED: Context7 /tracel-ai/cubecl — `sum_maybe_plane` example]

**Gotcha (the reason the LDS kernels do NOT already do this):** `SharedMemory::<_>::new(N)` needs a **comptime** `N`. The repo deliberately allocates the fixed `HIST_LDS_MAX = 512` max and drives the active length with the *runtime* `lds_len`/`feat_len` (histogram.rs:458-463 comment) precisely to avoid one binary per bin-count (LightGBM's `histogram{16,64,256}.cl` family). Making `lds_len` comptime would specialize the binary per bin-count — possible, but it re-introduces the multi-binary cost the repo chose to avoid. **Recommend: do NOT comptime the bin-count.** The lower-risk comptime candidates are scalars that are constant across a whole train run and currently passed as runtime args feeding loop bounds or a branch: e.g. `num_data` (the resident column stride — constant for the entire training run) and the LDS-vs-naive dispatch (already a host `if`, no kernel branch). Net: comptime has a **small** surface here; flag it as a low-priority Stage-2 lever, not the main win.

### Lever 3 — CubeDim / CubeCount occupancy

cubecl exposes `CubeCount::Static(x,y,z)` and `CubeDim::new_1d(n)` (used throughout). Occupancy was **already tuned** by spike-007 (`row_partition_count`, P clamped to 16; P=32 over-partitions and regresses — memory `spike-007-row-partition-occupancy`). ngo confirmed the LDS kernel is at-worst-tied with the mirror at the large leaf. **Do not re-explore occupancy** unless a new bench motivates it; the lever is closed.

## Production Kernel → Technique Map

Eight `#[cube(launch)]` production kernels (the mirror at :1007 is already `launch_unchecked` and OUT of scope). `launch_unchecked` applies to **all eight** and is numerics-preserving. `#[comptime]` candidates are narrow.

| Kernel (line) | Launcher (site) | `launch_unchecked` win | Device accesses to enumerate in SAFETY | Comptime candidate | Parity risk |
|---|---|---|---|---|---|
| `construct_hist_kernel_atomic_f32` (:389) | `construct_histograms_parallel_f32_on` (:443) | drop per-row bounds branch in `2*n` scatter | `binned[idx]`, `grad/hess[idx]`, `out[bin*2+1]` | none material | **none** (order-identical) |
| `construct_hist_kernel_lds_f32` (:522) | `construct_histograms_lds_f32_on` (:618) | drop checks in zero/scatter/merge loops | `binned[i]`,`grad/hess[i]`,`sub[ti+1]`,`out[m]` | `lds_len`→comptime (rejected: per-bin binary) | **none** |
| `construct_leaf_hist_batched_kernel` (:651) | `build_leaf_histograms_batched_f32_on` naive (:743) | drop check in `num_features*R` scatter | `gathered_bins[idx]`,`ord_g/h[k]`,`slot_off[f]`,`out[cell+1]` | none | **none** |
| `construct_leaf_hist_batched_lds_kernel` (:924) | …batched LDS (:721) | drop checks in 3 loops | `gathered_bins[fbase+k]`,`ord_g/h[k]`,`slot_off[f/f+1]`,`sub`,`out[base+m]` | `feat_len`/`num_data` | **none** |
| `construct_leaf_hist_resident_kernel` (:772) | `resident_raw_build_into` naive (:1426) | drop check in `total` scatter | `resident_bins[f*num_data+row]`,`leaf_rows[k]`,`ord_g/h[k]`,`slot_off[f]`,`out[cell+1]` | `num_data` (run-constant stride) | **none** |
| `construct_leaf_hist_resident_lds_kernel` (:873) | `resident_raw_build_into` LDS (:1403) **[WIRED production path]** | drop checks in 3 loops | `resident_bins[col+leaf_rows[k]]`,`slot_off[f/f+1]`,`sub`,`out[base+m]` | `num_data` | **none** |
| `fix_compact_kernel` (:1476) | two sites (:1657, :1808) | drop checks in per-feature fix/compact loops | `h_raw`,`h_hist`,`slot/numbin/offset/mfb[f]` all len `n` | per-feature scalars are already array-indexed | **none** (f64, deterministic) |
| `build_fix_scan_fused_kernel` (:1913) | fused launcher (:2255) | drop checks in build+fix+scan | `resident_bins`,`leaf_rows[k]`,`ord_g/h`,`h_hist`,`h_out[f*12..]`, index arrays len `n` | `num_data_stride` | **none** (f64, deterministic) |

**Key reads:**
- The WIRED training-path kernel is `construct_leaf_hist_resident_lds_kernel` (:873, launched :1403) — prioritize its sweep + re-pin.
- The `fix_compact`/`fused` kernels are **f64 and deterministic** (one cube per feature, `CubeDim::new_1d(1)`, ascending fold). `launch_unchecked` there carries **zero** numeric risk and is bit-exact — the safest of all.
- Expected win is **launch-overhead/codegen only** (removing the branch from the scatter hot loop). mwr measured the dominant cost at these sizes is **transfer, not launch** — so quantify per-kernel, do not over-claim. The win is real but modest; report measured medians, not estimates.

## Common Pitfalls

### Pitfall 1 — DEF-MWR-01 pre-existing flaky parity (do NOT attribute to this work)
The full-corpus near-zero-grad cell test (`cuda_mirror_full_corpus_leaf_matches_anchor`) intermittently shows |diff|~8.7e-6 > the ABS 5e-6 floor — f32-atomic cancellation on cells whose true sum is ~0, with nondeterministic accumulation order. **Proven pre-existing** (mwr reverted to HEAD~1 and still saw 1/6 failures). `launch_unchecked` CANNOT change accumulation order, so any parity movement on a near-zero-grad cell after the sweep is THIS landmine, **not a regression**. When verifying, use the bounded leaf subset `(7..num_data).step_by(3)` the stable tests use; if a full-corpus cell trips, distinguish it explicitly.

### Pitfall 2 — `launch_unchecked` SAFETY contract incompleteness
The unsafe contract requires EVERY device access be host-proven in-range. Missing one access in the enumeration = UB on a malformed input. Mitigation: copy the mirror's per-access enumeration style (:1174-1189) and check it against the kernel body line-by-line. Every launcher already has the V5 checks and the `unsafe` block — the only new artifact is the per-access prose.

### Pitfall 3 — comptime bin-count would regenerate the multi-binary cost
Making `lds_len`/`feat_len` comptime specializes one kernel binary per bin-count — the exact thing the repo avoids (histogram.rs:458-463). Keep bin-count runtime; comptime only run-constant scalars (e.g. `num_data` stride) if at all.

### Pitfall 4 — order-changing (Stage 3) moves f32 results
Any restructure that changes atomic scatter order or reduction shape can move f32 cells. CONTEXT permits this within the envelope, but it MUST be re-pinned GPU-vs-CPU-f64-anchor (never GPU-vs-GPU). If a cell exceeds ABS 5e-6 / REL 1e-5, document the residual and FLAG the tolerance review — do not silently widen the gate.

## Recommended Ordering (cheapest/safest first)

1. **Stage 1 — `launch_unchecked` sweep (numerics-preserving, do first).**
   Order within: (a) the two **f64 deterministic** kernels (`fix_compact_kernel`, `build_fix_scan_fused_kernel`) — zero numeric risk, bit-exact; (b) the **wired** `construct_leaf_hist_resident_lds_kernel`; (c) the remaining atomic/batched/resident kernels. Switch `#[cube(launch)]`→`#[cube(launch_unchecked)]` and `::launch`→`::launch_unchecked`, extend each SAFETY comment with the per-access enumeration from the table. **Re-pin to the CPU f64 anchor after each kernel** (the existing `rocm_cuda_mirror.rs` ABS 5e-6 / REL 1e-5 pattern; bounded leaf subset). No tolerance change expected (order-identical).

2. **Stage 2 — `#[comptime]` run-constant scalars (optional, small surface).**
   Only `num_data` (resident stride) and similar train-run-constant scalars. **Skip the bin-count** (Pitfall 3). Re-pin after each.

3. **Stage 3 — order-changing restructure (only if a kernel-level bench shows a win).**
   ngo's data says the LDS kernel is already at/near optimal, so this stage may be a no-op. If attempted, re-pin GPU-vs-CPU-f64-anchor and flag any tolerance movement (Pitfall 4).

**Re-pin gate after every stage:** GPU-vs-CPU-f64-anchor, never GPU-vs-GPU (DEF-f8u-01); honor the warm-vs-cold bench rule for any timing claim (3 warm-ups discarded, median of ≥7).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The cubecl 0.10.0 `launch_unchecked`/`#[comptime]` API matches the Context7 examples (book is on `main`; repo is pinned 0.10.0) | cubecl levers | LOW — the mirror at :1007/:1191 already compiles `launch_unchecked` against the pinned 0.10.0, proving the API shape in-repo |
| A2 | `launch_unchecked` removes only bounds-check codegen, not numerics, on each production kernel | Kernel map | LOW — proven on the mirror by mwr's A/B; re-pin confirms per kernel |
| A3 | Per-kernel `launch_unchecked` win is modest (transfer-bound at these sizes) | Summary | LOW — mwr measured transfer ≫ launch; the plan must measure, not estimate |

## Sources

### Primary (HIGH confidence)
- Context7 `/tracel-ai/cubecl` — `#[cube(launch)]`/`#[cube(launch_unchecked)]` macro, `gelu_array::launch_unchecked` example, `sum_maybe_plane` comptime example, `#[cube(comptime)]` struct fields.
- In-repo: `crates/lgbm-compute/src/kernels/histogram.rs` (all 8 production kernels + the mirror's worked `launch_unchecked` contract :1174-1203), `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` (parity pattern, ABS 5e-6 / REL 1e-5), `Cargo.toml:25` + `Cargo.lock` (cubecl 0.10.0).
- Prior summaries: `260619-mwr-SUMMARY.md` (launch_unchecked numerics-preserving proof, DEF-MWR-01), `260619-ngo-SUMMARY.md` (LDS path is the optimal production kernel; occupancy closed).

### Secondary (MEDIUM confidence)
- cubecl book (burn.dev/books/cubecl) — comptime specialization / loop-unrolling rationale.

### Tertiary (LOW confidence)
- WebSearch (arxiv "Fearless Concurrency on the GPU", github tracel-ai/cubecl) — `unchecked_accesses` / lowering loads without dynamic bounds checks (corroborates Primary).

## Metadata

**Confidence breakdown:**
- cubecl API (launch_unchecked, comptime): HIGH — verified via Context7 + in-repo mirror compiles against pinned 0.10.0.
- Kernel inventory + technique map: HIGH — read directly from source with line numbers.
- Magnitude of win: MEDIUM — mwr shows transfer-bound; per-kernel launch win must be measured.

**Research date:** 2026-06-19
**Valid until:** 2026-07-19 (cubecl 0.10.0 pinned; stable)
