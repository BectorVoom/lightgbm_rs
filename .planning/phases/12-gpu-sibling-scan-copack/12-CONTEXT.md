# Phase 12: gpu-sibling-scan-copack - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning
**Source:** Spike campaign (023 post-021-roundtrip-attribution + 024 batch-sibling-scans). The spike process WAS the research/context gathering — this phase WIRES the validated lever. Mirrors the Phase-11 spike-validated pattern.

<domain>
## Phase Boundary

In the ROCm resident GPU scan path, replace the **two separate per-sibling scan
launches + two readbacks** (today: the smaller child and the larger child are each scanned
by their own `scan_resident_leaf` call ⇒ one blocking `read_one_unchecked` sync per
leaf-node ≈ 59 syncs/tree) with **ONE co-packed 2-slot scan launch + ONE readback per
split** (≈ 30 syncs/tree). Delivers (per spike-024): a ~2.0× isolated win on the scan
launch+readback, **bit-exact** to the two-scan path; honest e2e ~10–15% at small/medium,
~1.5% at wide (per spike-023's scan-sync fractions). CPU f64 anchor UNTOUCHED; CPU/GPU
routing UNCHANGED; the wide build path (u64 atomics, Phase 11) UNTOUCHED.
</domain>

<decisions>
## Implementation Decisions (LOCKED — spike-validated)

### The 2-slot scan kernel (new)
- Add a `#[cube(launch)]` kernel that scans **two** resident histogram Handles in one launch
  — modelled on `find_best_splits_fused_kernel` (`kernels/split.rs:992`) but taking
  `hist_a` AND `hist_b` (the smaller- and larger-child slots) and writing `out` of length
  `2*n*12`. Global lane `g = ABSOLUTE_POS as u32`: `g < n_feats` ⇒ sibling-A (smaller) feature
  `g` into `out[g*12..]`; `n_feats ≤ g < 2*n_feats` ⇒ sibling-B (larger) feature `g − n_feats`
  into `out[g*12..]`. Tail-cube guard `g < 2*n_feats`.
- **Per-feature param arrays are SHARED** between siblings (both children have the same dataset
  feature layout: same `slot_off`/`num_bin`/`offset`/`default_bin`/`skip`/`rev_count`/`fwd_count`,
  length `n`). **Leaf-level scalars are PER-SIBLING** (`sum_gradient`, `sum_hessian_bumped`,
  `num_data`, `min_gain_shift` differ — smaller built vs larger subtract-derived). Reuse the
  shared `split_scan_body` (`kernels/split.rs:144`) with the right slot + per-sibling scalars.
- Feature-per-lane **W=64** (the SHIPPED spike-021 packing; `scan_cube_dim()` / `LGBM_SCAN_CUBEDIM`);
  CubeCount = `ceil(2*n_feats / W)`. `W=1` (cubecl-cpu anchor) stays byte-identical.
- New backend method `scan_resident_siblings(smaller_slot, larger_slot, …)` on the `Backend`
  trait + RocmBackend impl (analog of `scan_resident_leaf`, `lib.rs:2356`), returning
  `(Vec<SplitInfo>, Vec<SplitInfo>)` decoded from ONE `read_one_unchecked` (first `n` =
  smaller, next `n` = larger). CpuBackend impl may delegate to two `scan_resident_leaf` calls
  (the win is GPU-launch-structural; CPU has no launch floor) OR a W=1 two-slot scan — either
  is bit-exact; choose the simpler.

### Growth-loop reorder (learner.rs `find_best_splits`, ~1406–1790)
- Today: smaller child is built+scanned (~1547–1612), THEN larger child is subtracted+scanned
  (~1613–1777). To co-pack, **defer the smaller-child scan** until after `subtract_resident`
  (`learner.rs:1653`) so both slots scan together: **build smaller → subtract larger →
  scan(smaller, larger) co-packed → one readback → distribute the two SplitInfo vecs.** Both
  Handles are simultaneously resident at that point (the 023/scan-map confirms `subtract_resident`
  reads the smaller slot, so it is alive through the subtract). No build/subtract reordering needed.
- **Eligibility gate:** co-pack applies ONLY when BOTH siblings take the resident scan-only path
  (`resident_eligible`, not fused, not host) AND both are scannable (`sum_h > 0`, num_data > 0).
  Any other case (host path, the OFF-by-default fused `build_fix_scan_resident`, a single-child
  split, categorical/forced-split spine-ineligible) ⇒ **fall back to the existing two separate
  scans** (byte-unchanged). Add a `LGBM_SIBLING_COPACK` env override (`0` force-off / `1` force-on
  bypassing only a size threshold) for A/B benching, mirroring `LGBM_RESIDENT_FORCE`.

### Parity (bit-exact by construction)
- Each feature's REVERSE+FORWARD sequential scan is UNCHANGED (the same `split_scan_body`,
  same per-feature disjoint region) — **NO spike-016 reorder**. Co-packing only changes WHICH
  launch a feature's scan runs in, not its math ⇒ the co-packed SplitInfos are byte-identical to
  the two-separate-scan path (spike-024 proved this: B's two halves == A's two results, every cell).
- Leaf scalars: the `2*kEpsilon` `sum_hessian` bump + `min_gain_shift` are computed ONCE PER
  SIBLING (as the single-slot path already does), threaded per-sibling. The subtract-derived
  larger child is NOT re-FixHistogram'd (non-negotiable #3) — the kernel reads its pre-fixed Handle.
- Output order preserved (input feature `i` → output `out[i]`), per sibling.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spike evidence (the validated design + numbers)
- `.planning/spikes/024-batch-sibling-scans/README.md` — the isolated A/B (~2.0×, bit-exact), e2e bound, ROI gate.
- `.planning/spikes/023-post-021-roundtrip-attribution/README.md` — the regime-split attribution + per-tree sync counts.
- `crates/lgbm-compute/examples/spike024_sibling_scan_ab.rs` — the working 2-slot-vs-2×1-slot kernel A/B + parity gate.
- `Skill("spike-findings-lightgbm_rs")` — the campaign blueprint (bit-exact gate, cold-overstates-warm, cube authoring gotchas).

### Code to modify / mirror
- `crates/lgbm-compute/src/kernels/split.rs` — `find_best_splits_fused_kernel` (`:992`), `split_scan_body` (`:144`), `find_best_splits_fused_inner` (`:1193`), `scan_cube_dim()`.
- `crates/lgbm-compute/src/lib.rs` — `scan_resident_leaf` (`:2356`), `subtract_resident` (`:2319`), the `Backend` trait (`:943`), CpuBackend mirror.
- `crates/lgbm-treelearner/src/learner.rs` — `find_best_splits` smaller-scan (`~1593`) + larger-scan (`~1756`) + subtract (`:1653`); `phase_prof` COUNTERS.
- `crates/oracle-harness/tests/kernel_parity.rs` (`--features rocm`) — the ~1e-6 hip-vs-anchor gate to extend with a co-pack cell.

### Conventions / gotchas
- `.planning/spikes/CONVENTIONS.md` — cubecl `#[cube]` authoring gotchas (raw scalars, `f64::cast_from`, inline-not-helper, literal `0.0` sentinels), GPU device-time A/B discipline (median[p25..p75], ≥2 restarts, judge SIGN on the 8-CU APU).
</canonical_refs>

<specifics>
## Success Criteria (observable)

1. **Bit-exact parity:** a new `kernel_parity` (`--features rocm`) cell asserts the co-packed
   sibling scan returns SplitInfos byte-identical to two separate `scan_resident_leaf` calls,
   AND the rocm co-pack path stays within ~1e-6 of the CPU f64 anchor. `cubecl-cpu` W=1 byte-identical.
2. **Merge gate green (CPU anchor untouched):** `cargo test -p lgbm-treelearner --lib`,
   `-p lgbm-boosting`, `-p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`, `learner_parity`) all pass.
3. **Sync count drops:** under `LGBM_PHASE_PROF=1`, the `COUNTS:` line shows `scan_resident`
   ≈ halved per tree (≈59 → ≈30) on the eligible resident path (e.g. medium 20k×30 / large 200k×40).
4. **e2e (sign, APU-confounded):** `bench_gpu_vs_cpu` median train at small/medium is **not slower**
   and trends faster (target ~10–15%; report honestly per the cold-overstates-warm rule — do NOT
   claim the isolated 2× as e2e). Wide unaffected (~1.5%), routing unchanged.
</specifics>

<deferred>
## Out of Scope / Deferred

- CPU/GPU routing changes; the wide histogram BUILD path (Phase 11 u64 atomics).
- The host scan path and the OFF-by-default fused `build_fix_scan_resident` path — co-pack is
  resident-scan-only; everything else falls back to the unchanged two-scan path.
- **Depth-wise / frontier-batched scan** (the further sync-cut beyond sibling-merge) — spike-025
  SUPERSEDED (leaf-wise dependency + parity risk). Not in this phase.
- Discrete-gfx110x perf validation (this APU is sign-only).
</deferred>

---

*Phase: 12-gpu-sibling-scan-copack*
*Context gathered: 2026-06-25 from spikes 023 + 024 (spike-validated, research-equivalent)*
