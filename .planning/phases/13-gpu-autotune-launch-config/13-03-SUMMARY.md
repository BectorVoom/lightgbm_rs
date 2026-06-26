---
phase: 13-gpu-autotune-launch-config
plan: 03
subsystem: gpu-compute
tags: [cubecl, autotune, rocm, split-scan, scan-W, feature-per-lane, launch-config]

# Dependency graph
requires:
  - phase: 13-01
    provides: "kernels::autotune module (LaunchKey, autotune_enabled, cache_namespace_id) + serde rocm dep"
  - phase: 13-02
    provides: "the TunableSet/LocalTuner wiring convention (rebuild set per call; env-pin seam; read_one_unchecked returns Bytes; OVERWRITE-vs-ACCUMULATE generator classification)"
  - phase: spikes-021-022
    provides: "feature-per-lane fused scan (bit-exact across CubeDim W)"
  - phase: spikes-037-040
    provides: "cubecl::tune real-0.10 API, CloneInputGenerator-for-OVERWRITE classification, log2 key, vs-heuristic win"
provides:
  - "SCAN_TUNER + SCAN_WSET + scan_wset_tunable_set — the single-leaf scan-W autotune machinery (CloneInputGenerator)"
  - "SCAN_SIBLINGS_TUNER + scan_wset_siblings_tunable_set — the co-pack 2-slot sibling-scan twin (separate cache namespace)"
  - "both fused split launchers (find_best_splits_fused_inner + find_best_splits_fused_siblings_from_handles_on) default-on autotuned for CubeDim W; scan_cube_dim() kept as the LGBM_AUTOTUNE=0 / explicit-LGBM_SCAN_CUBEDIM fallback + parity seam"
affects: [13-04-parity-bench]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CloneInputGenerator for OVERWRITE-class autotune kernels (the scan writes fresh out windows) — vs FreshOutGenerator for the ACCUMULATING build (13-02), per spike-038"
    - "two kernel families (single-leaf vs 2-slot co-pack) get SEPARATE LocalTuner namespaces so their identical (0,feats,bins) LaunchKey never collides in one cache"
    - "autotune-or-fallback guard as `let autotuned = autotune_enabled() && env_unset(LGBM_SCAN_CUBEDIM)` + cfg(not rocm) ⇒ false, so the non-rocm oracle path stays on scan_cube_dim()==1"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/split.rs

key-decisions:
  - "WIRED BOTH production scan paths (lib.rs:2642 single-leaf + lib.rs:2680 co-pack siblings), not just the single-leaf 1466 consumer — the co-pack is the dominant steady-state scan path, leaving it on the heuristic would miss the main lever"
  - "SCAN_WSET = {32,64,128,256} (W=1 excluded — it is the degenerate the lever exists to avoid; still reachable via LGBM_SCAN_CUBEDIM=1 / non-rocm oracle)"
  - "LaunchKey.bucket = 0 (scan W tracks feature/bin shape, not row count) so the key is stable across a train — no per-leaf tuning storm (spike-039)"
  - "LaunchKey.bins = max(num_bin) (widest feature's bin count) — the per-feature slot-width driver, matching 13-02"

patterns-established:
  - "Pattern: an OVERWRITE-class GPU kernel under cubecl autotune uses the built-in CloneInputGenerator (re-running a rep recomputes the identical store); only ACCUMULATING kernels need a fresh-output generator"
  - "Pattern: distinct kernel families that share a LaunchKey shape get distinct local_tuner!(name) namespaces to keep their on-disk caches isolated"

requirements-completed: [AT-SCAN-W]

# Metrics
duration: 4min
completed: 2026-06-26
status: complete
---

# Phase 13 Plan 03: GPU Autotune Scan-W Summary

**CubeCL autotune is now the default-on rocm selector for the split-scan `CubeDim` width `W`, wired at BOTH fused split launchers — the single-leaf `find_best_splits_fused_inner` (`SCAN_TUNER`) and the co-pack 2-slot sibling scan `find_best_splits_fused_siblings_from_handles_on` (`SCAN_SIBLINGS_TUNER`) — each tuner sweeping `SCAN_WSET = {32,64,128,256}` with a `CloneInputGenerator` (the scan OVERWRITES fresh `out` windows, spike-038 OVERWRITE class), keyed on the stable `(0, num_features, max_num_bin)` shape, with `scan_cube_dim()` / `LGBM_SCAN_CUBEDIM` kept as the `LGBM_AUTOTUNE=0` fallback and the 13-04 all-W parity seam.**

## Performance

- **Duration:** ~4 min
- **Tasks:** 2
- **Files modified:** 1 (`crates/lgbm-compute/src/kernels/split.rs`)

## Accomplishments

- **Task 1 — scan-W autotune machinery (commit `eeec02d`):** Added `SCAN_WSET = {32,64,128,256}` (each clamped `[1,256]`; `W=1` intentionally excluded — it is the one-cube-per-feature degenerate the spike-021 lever exists to avoid, still reachable via `LGBM_SCAN_CUBEDIM=1` / the non-rocm oracle), `SCAN_TUNER` (`local_tuner!("scan")`) and `SCAN_SIBLINGS_TUNER` (`local_tuner!("scan_siblings")`), `launch_scan_at<R>` / `scan_wset_tunable_set<R>` (one `Tunable` per `W` launching `find_best_splits_fused_kernel`, `CloneInputGenerator`, key `LaunchKey{bucket:0, feats:n, bins:max_num_bin}`), and the co-pack twin `launch_scan_siblings_at<R>` / `scan_wset_siblings_tunable_set<R>` (wrapping `find_best_splits_fused_siblings_kernel`). The `CloneInputGenerator` choice carries an inline comment naming the spike-038 OVERWRITE class so a future reader does not "fix" it to a fresh-output generator.
- **Task 2 — wiring both fused launchers (commit `3f50c4c`):** Replaced the lone `scan_cube_dim()` pick at each of the two scan launch sites with an `autotune-or-fallback` guard: `let autotuned = autotune::autotune_enabled() && std::env::var_os("LGBM_SCAN_CUBEDIM").is_none()` (`#[cfg(not(feature="rocm"))] ⇒ false`). When `autotuned`, the launch is driven through the tuner (`SCAN_TUNER` / `SCAN_SIBLINGS_TUNER`) whose winner writes the real `h_out`; else the EXISTING `scan_cube_dim()` direct launch runs byte-for-byte unchanged (covering `LGBM_AUTOTUNE=0`, an explicit `LGBM_SCAN_CUBEDIM`, and the non-rocm `scan_cube_dim()==1` bit-exact oracle path). The `h_out` read-back + per-feature 12-cell decode/accept-gate is unchanged in both branches.

## Scope decision (non-silent — the second `scan_cube_dim()` consumer)

The plan gave discretion on the second consumer (the phase-12 co-pack sibling-scan launcher at the former line ~1746): wire it the same way OR defer with a recorded reason. **Decision: WIRED.** Both scan launchers are live production paths (`lib.rs:2642` → single-leaf `find_best_splits_fused_inner`; `lib.rs:2680` → co-pack `find_best_splits_fused_siblings_from_handles_on`), and the co-pack is the dominant steady-state scan (it scans both children of an interior split in one launch), so leaving it on the heuristic would have left the main scan lever un-autotuned. The sibling kernel is the SAME OVERWRITE class (each lane writes a fresh 12-cell window), so its tunable set also uses `CloneInputGenerator`. It runs under a SEPARATE `SCAN_SIBLINGS_TUNER` namespace: its `LaunchKey` would otherwise collide on `(0, feats, bins)` with the single-leaf scan yet benchmark a different kernel — the distinct `local_tuner!("scan_siblings")` keeps the two on-disk caches isolated while both reuse the shared `SCAN_WSET` (same `W` ordering ⇒ same `fastest_index`→`W` mapping).

## Verification

- **oracle-harness `kernel_parity` (rocm) — all three modes green, 18 passed each:**
  - Default-on autotune: `kernel_parity_fused_equals_per_feature_and_native` (single-leaf) + `hip::kernel_parity_sibling_copack_equals_two_scans_on_hip` (co-pack) pass — both wired paths produce correct per-feature split windows under the tuner.
  - `LGBM_AUTOTUNE=0`: 18 passed (heuristic `scan_cube_dim()` fallback reproduces the prior path).
  - `LGBM_SCAN_CUBEDIM=128`: 18 passed (explicit override wins over autotune).
- **CPU merge gate untouched:** `cargo test -p lgbm-treelearner --lib` → 77 passed (rocm-gated wiring only; the non-rocm `scan_cube_dim()==1` oracle path is unchanged).
- **`cargo test -p lgbm-compute --features rocm --lib`** → 60 passed.
- Default (cpu, no-rocm) build of `lgbm-compute` compiles (the new machinery is `#[cfg(feature = "rocm")]`; the wiring's `autotuned` resolves to `false`).

## Deviations from Plan

None for the core machinery. One scope choice (documented above, plan-sanctioned discretion): the second consumer was WIRED rather than deferred, adding `SCAN_SIBLINGS_TUNER` + `scan_wset_siblings_tunable_set` + `launch_scan_siblings_at` beyond the three artifacts the plan listed for the single-leaf path. No production behavior changed except the intended scan-W selection; all parity gates hold.

## Threat mitigations applied

- **T-13-03-01 (tamper `LGBM_SCAN_CUBEDIM`):** the existing `scan_cube_dim()` clamps `[1,256]` and parse-fails to the default; an explicit (even garbage) `LGBM_SCAN_CUBEDIM` makes `var_os(...).is_none()` false ⇒ `autotuned=false` ⇒ the override-honoring fallback runs (the documented escape hatch wins over autotune).
- **T-13-03-02 (poisoned scan cache picks a different W):** accepted — every `SCAN_WSET` variant is bit-exact (each feature's scan stays sequential, no spike-016 reorder), so a wrong pick is perf-only; full all-W bit-exactness is the 13-04 gate.

## Known limitations / honest bound

- Absolute wall-clock is APU-confounded (spoofed 8-CU APU); the durable deliverable is the SELECTION method (measure-don't-model) + portability to real discrete GPUs, not a local e2e train-time win (the 16-core CPU still beats this GPU scan end-to-end here). The autotune SELECTION axis is the spoof-robust deliverable.
- The cold tune on first call per `(0, feats, bins)` key writes a small `target/autotune/0.10.0/<device>/...scan*.json.log`; warm hits are ~µs.

## Next Phase Readiness

- 13-04 (all-variants parity bench) can pin each `W` via `LGBM_SCAN_CUBEDIM=k` and assert every `SCAN_WSET` variant against the CPU f64 anchor, and toggle `LGBM_AUTOTUNE=0` to exercise the `scan_cube_dim()` fallback — for BOTH the single-leaf and co-pack scan launchers.

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/split.rs` present on disk.
- All scan-autotune symbols verified by grep: `SCAN_TUNER`, `SCAN_WSET`, `scan_wset_tunable_set`, `SCAN_SIBLINGS_TUNER`, `scan_wset_siblings_tunable_set`, the two `*.execute` call sites, the preserved `scan_cube_dim()` fallback.
- Task commits verified in git history: `eeec02d` (Task 1), `3f50c4c` (Task 2).

---
*Phase: 13-gpu-autotune-launch-config*
*Completed: 2026-06-26*
