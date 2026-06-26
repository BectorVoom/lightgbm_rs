---
phase: 13-gpu-autotune-launch-config
verified: 2026-06-26T13:30:00Z
status: passed
score: 13/13 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 13: Autotuned GPU launch-config selection (`cubecl::tune`) Verification Report

**Phase Goal:** Replace the hand-tuned/env GPU launch-config heuristics with CubeCL runtime autotuning, default-on for ALL GPU (rocm) selection — autotuning BOTH the histogram-BUILD row-partition `P` and the split-SCAN `CubeDim` width `W`. Must NOT change CPU routing or the f64 anchor; stays within the ~1e-6 ROCm parity gate.
**Verified:** 2026-06-26T13:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Verification Note on ROCm Runtime Results

Several truths assert runtime behavior on the ROCm device (the build/scan tuner winners producing correct histograms/windows; all-PSET/all-WSET anchor parity; the sign-stable e2e A/B). Per the verification brief, the ROCm tests were executed green by the phase executors and are NOT re-run here. This report verifies that the **code and test artifacts exist and are correctly wired** (which I confirmed directly from the source), and **trusts the executor-attested ROCm pass** for the on-device numeric results. Items resting on executor attestation are marked in the Evidence column. The CPU-side claims (default build, merge gate, off-switch, key math) were verified directly.

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | Default (cpu, no-rocm) build pulls no autotune/serde-derive codegen on the hot path | ✓ VERIFIED | `cargo build -p lgbm-compute` exit 0; `kernels/mod.rs:12` gates `pub mod autotune` behind `#[cfg(feature="rocm")]`; `serde` is `optional=true` + only in the `rocm` feature list (Cargo.toml:27,41) |
| 2 | `autotune_enabled()` default-on; false iff `LGBM_AUTOTUNE=0`; read fresh per call | ✓ VERIFIED | `autotune.rs:81-83` (`!matches!(env::var(...), Ok("0"))`); unit test `autotune_enabled_flips_with_env` (:130) |
| 3 | `Backend::prefers_autotune_launch_config` default-false; RocmBackend delegates to `autotune_enabled()`; CpuBackend inherits false | ✓ VERIFIED | trait default `lib.rs:915-917` (`{ false }`); RocmBackend override `lib.rs:2202-2204` → `autotune::autotune_enabled()`; exactly 2 occurrences (CpuBackend does not override) |
| 4 | `LaunchKey` serde-derives + `cubecl::tune::AutotuneKey` + `Display` | ✓ VERIFIED | `autotune.rs:33-49` (derive Serialize/Deserialize/Hash/Eq; `impl AutotuneKey`; Display `LaunchKey(b..,f..,b..)`); test :142 |
| 5 | `size_band(rows)` returns the `log2` occupancy bucket (not exact rows), never panics | ✓ VERIFIED | `autotune.rs:60-67`; tests `size_band_never_panics_on_zero`, `_is_monotonic_non_decreasing`, `_buckets_same_decade_together` (:104-127) |
| 6 | Build P autotuned via `BUILD_TUNER` over `BUILD_PSET={1,4,8,16,32}`; winner writes the real `out` once (grad-conservation); uses FreshOutGenerator (accumulating kernel), never CloneInputGenerator | ✓ VERIFIED | `histogram.rs:1782,1787,1892-1920` (FreshOutGenerator swaps slot 5 with fresh zeroed u64/f32; GAT spelled `<Vec<Handle> as TuneInputs>::At<'a>`); three-way pick :2166-2200; grad-conservation test `:3498+` (fresh vs clone) — **rocm runtime pass executor-attested** |
| 7 | Build AutotuneKey = `(size_band(rows), num_features, num_bins)`; `LGBM_AUTOTUNE=0` → `row_partition_count` byte-for-byte; `LGBM_AUTOTUNE_FORCE_P=k` pins single P | ✓ VERIFIED | key gen `histogram.rs:1943`; fallback branch :2196-2199; `force_row_partition()` :1795 + pin branch :2166-2169 |
| 8 | Both live resident build classes (f32-resident + u64 fixed-point) route through `resident_raw_build_into`; all four `row_partition_count` sites accounted for non-silently | ✓ VERIFIED | wired site :2013/2166; explicit scope comment naming sites 982/1543/1692 as cold-fallback/test-only :2148-2165; sites 982/1543/1692 textually unchanged (`grep` confirms still `row_partition_count`) |
| 9 | Scan W autotuned via `SCAN_TUNER` (+ `SCAN_SIBLINGS_TUNER`) over `SCAN_WSET={32,64,128,256}`; CloneInputGenerator (OVERWRITE class); key `(0,feats,bins)` | ✓ VERIFIED | `split.rs:1002,1008,1016,1117,1249` (CloneInputGenerator both sets); key `bucket:0` |
| 10 | Both fused split launchers wired (single-leaf + co-pack siblings); `LGBM_AUTOTUNE=0` / explicit `LGBM_SCAN_CUBEDIM` fall back to `scan_cube_dim()`; non-rocm `autotuned=false` | ✓ VERIFIED | guard `split.rs:1777-1778` + `.execute` :1814; sibling guard :2111-2112 + `.execute` :2151; `#[cfg(not rocm)] let autotuned=false` :1780,:2114; fallback `scan_cube_dim()` preserved :1824,:2156 |
| 11 | All-PSET build parity pinned to the CPU f64 anchor (u64 fixed-point ~1e-7), at least one P>1; all-WSET scan bit-exact to W=1 — never GPU-vs-GPU | ✓ VERIFIED | `kernel_parity.rs:2813` builds anchor via `construct_histograms_cpu`+`fix_histogram`+`host_compact` (NOT a 2nd GPU launch), loops `LGBM_AUTOTUNE_FORCE_P` over PSET, P>1 guard :2823; `:2935` loops `LGBM_SCAN_CUBEDIM`, `assert_eq!` vs W=1; panic-safe `EnvVarGuard` :2770 — **rocm runtime pass executor-attested** |
| 12 | e2e A/B (`bench_gpu_vs_cpu`) shows autotune ≥ heuristic at production width, recovers P=1; default bench unchanged (opt-in) | ✓ VERIFIED | A/B arm `bench_gpu_vs_cpu.rs:289-507`, gated `LGBM_BENCH_AUTOTUNE_AB=1` :638; reads winner `build_P{n}` at `fastest_index` from cache :336-364; `ScopedEnv` :317 — **2-restart sign-stable NOT-SLOWER executor-attested (13-04 SUMMARY)** |
| 13 | CPU f64 merge gate untouched; first-tune cost + honest-bound recorded | ✓ VERIFIED | `lgbm-treelearner --lib` 77 passed (orchestrator-confirmed) + default `lgbm-compute` build green; first-tune cost + spoof-confounded honest bound recorded verbatim in 13-04 SUMMARY (:90-100) |

**Score:** 13/13 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-compute/src/kernels/autotune.rs` | LaunchKey/size_band/autotune_enabled/cache_namespace_id, rocm-gated | ✓ VERIFIED | 147 lines, all symbols `pub`, 5 unit tests |
| `crates/lgbm-compute/Cargo.toml` | serde real rocm-gated optional dep | ✓ VERIFIED | line 27 `optional=true`; line 41 `dep:serde` in `rocm` feature; absent from `[dev-dependencies]` |
| `crates/lgbm-compute/src/lib.rs` | `prefers_autotune_launch_config` seam | ✓ VERIFIED | default :915, override :2202 |
| `crates/lgbm-compute/src/kernels/histogram.rs` | FreshOutGenerator, BUILD_TUNER, BUILD_PSET, build_pset_tunable_set, force_row_partition, wired pick | ✓ VERIFIED | all symbols present (grep) |
| `crates/lgbm-compute/src/kernels/split.rs` | SCAN_TUNER(+siblings), SCAN_WSET, tunable-set builders, wired guards | ✓ VERIFIED | all symbols present (grep) |
| `crates/oracle-harness/tests/kernel_parity.rs` | all-PSET build + all-WSET scan anchor parity | ✓ VERIFIED | both test fns present (:2813,:2935) + EnvVarGuard |
| `crates/lgbm/examples/bench_gpu_vs_cpu.rs` | opt-in autotune A/B arm | ✓ VERIFIED | A/B arm present, cache-winner parser |

### Key Link Verification

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| `resident_raw_build_into` | `kernels::autotune` | `autotune_enabled()` + `size_band` + `cache_namespace_id()` + `BUILD_TUNER.execute` | ✓ WIRED (histogram.rs:2170-2195) |
| `BUILD_TUNER` TunableSet | LDS build kernels | each PSET P via `launch_build_at` + FreshOutGenerator | ✓ WIRED |
| fused split launchers | `kernels::autotune` | `autotune_enabled()` + `SCAN_TUNER/SCAN_SIBLINGS_TUNER.execute` | ✓ WIRED (split.rs:1814,2151) |
| `kernel_parity.rs` | histogram.rs + split.rs | `LGBM_AUTOTUNE_FORCE_P` / `LGBM_SCAN_CUBEDIM` sweep vs CPU f64 anchor | ✓ WIRED |
| `bench_gpu_vs_cpu.rs` | autotune launch paths | `LGBM_AUTOTUNE` A/B + cache `fastest_index` read | ✓ WIRED |

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
| ----------- | ----------- | ------ | -------- |
| AT-SERDE | 13-01 | ✓ SATISFIED | serde promoted to real rocm-gated optional dep (Cargo.toml:27,41) |
| AT-DEFAULT-ON | 13-01 | ✓ SATISFIED | `prefers_autotune_launch_config` default-on for rocm (lib.rs:2202); free-fn paths consult `autotune_enabled()` |
| AT-BUILD-P | 13-02 | ✓ SATISFIED | build-P autotune wired into `resident_raw_build_into` with FORCE_P + `LGBM_AUTOTUNE=0` fallback |
| AT-SCAN-W | 13-03 | ✓ SATISFIED | scan-W autotune wired into both fused launchers with `scan_cube_dim()`/`LGBM_SCAN_CUBEDIM` fallback |
| AT-PARITY | 13-04 | ✓ SATISFIED | all-PSET/all-WSET anchor parity tests + e2e A/B + CPU merge gate green |

All 5 requirement IDs accounted for across plan frontmatter, SUMMARYs, and code. No orphaned requirements (project has no `.planning/REQUIREMENTS.md`; IDs cross-referenced against plan frontmatter as instructed).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| histogram.rs | 3084 | `"...na_as_missing not yet implemented"` | ℹ️ Info | Pre-existing runtime error string in `build_fix_scan_resident` for an unsupported option; not a phase-13 artifact and unrelated to autotune scope. Not a debt marker for this phase. |

No `TBD`/`FIXME`/`XXX`/`todo!`/`unimplemented!` introduced by phase 13. All 13 commits are atomic and well-formed (b45b245 … 7849c7c).

### Notable observations (non-blocking)

- `kernel_parity.rs` defines `BUILD_PSET_MIRROR` / `SCAN_WSET_MIRROR` local copies (the real consts are private to the kernel modules). They currently match the real `BUILD_PSET={1,4,8,16,32}` / `SCAN_WSET={32,64,128,256}` and the test carries shape guards (non-empty + P>1). Minor future-drift risk if the real sets change without updating the mirror — informational only.

### Human Verification Required

None. The ROCm on-device numeric results (build/scan tuner correctness, all-variant anchor parity, sign-stable e2e A/B) were executed green by the phase executors and are trusted per the verification brief; the corresponding test/bench artifacts exist and are correctly wired in the codebase. The CPU merge gate and default-build no-leak claims were verified directly.

### Gaps Summary

No gaps. Every phase-13 must-have is satisfied by code that exists, is substantive, and is wired. The autotune machinery is uniformly `#[cfg(feature="rocm")]`-gated and does not alter the default CPU build or the f64 anchor (default `lgbm-compute` build green; trait method default-false so CpuBackend is byte-unchanged). Both GPU launch knobs (build `P`, scan `W`) are default-on autotuned on rocm with documented `LGBM_AUTOTUNE=0` / `LGBM_AUTOTUNE_FORCE_P` / `LGBM_SCAN_CUBEDIM` escape hatches, and every PSET/WSET variant is parity-gated to the CPU f64 anchor (never GPU-vs-GPU).

---

_Verified: 2026-06-26T13:30:00Z_
_Verifier: Claude (gsd-verifier)_
