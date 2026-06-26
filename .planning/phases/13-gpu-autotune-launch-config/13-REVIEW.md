---
phase: 13-gpu-autotune-launch-config
reviewed: 2026-06-26T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - crates/lgbm-compute/Cargo.toml
  - crates/lgbm-compute/src/kernels/autotune.rs
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm/examples/bench_gpu_vs_cpu.rs
  - crates/oracle-harness/tests/kernel_parity.rs
findings:
  critical: 0
  warning: 2
  info: 4
  total: 6
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-06-26
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the phase-13 net changes that wire CubeCL runtime autotuning (`cubecl::tune`)
default-on for the rocm GPU launch-config knobs: the histogram-build row-partition `P`
(`histogram.rs`) and the split-scan `CubeDim` width `W` (`split.rs`), plus the shared
`kernels::autotune` module, the `serde` dependency promotion, the all-variant parity gate
(`kernel_parity.rs`), and the e2e A/B bench arm (`bench_gpu_vs_cpu.rs`).

**The rocm `#[cfg]` gating is correct.** Every new autotune symbol, the `serde` dep, the
`kernels::autotune` module, and both launch-path branches are `#[cfg(feature = "rocm")]`;
the non-rocm `autotuned` binding resolves to `false`, so the default CPU build and the f64
deterministic anchor pull no autotune codegen and are byte-unchanged. I verified the spike
examples that `use serde` are cfg-gated (`#[cfg(not(feature = "rocm"))]` stub `main` +
serde behind `#[cfg(feature = "rocm")]`), so the dev→optional serde move does not break the
default `cargo build --examples`. The `FreshOutGenerator` (accumulating build) vs
`CloneInputGenerator` (overwriting scan) classification matches spike-038, argument lists at
both scan tuner call sites line up with the function signatures, the env seams clamp/fall
through safely, and the panic-safe `EnvVarGuard`/`ScopedEnv` restore-on-drop is sound for
the single-threaded test/bench usage.

Two Warnings concern the **parity gate's coverage of the f32 production build path** and a
**hand-copied mirror of private candidate-set consts**. Neither blocks: the project's own
contract frames the rocm f32 path as best-effort ~1e-6 with documented residual gaps, and
the CPU merge gate is untouched. The Info items are dead-code and diagnostic-fragility notes.

## Warnings

### WR-01: f32-resident default build is autotuned across `P` with no all-variant parity gate, introducing run-to-run nondeterminism

**File:** `crates/lgbm-compute/src/kernels/histogram.rs:2168-2200` (autotune branch in `resident_raw_build_into`); reachability via `crates/lgbm-compute/src/lib.rs:2429-2441` and `histogram.rs:1136-1148`
**Issue:** The default (non-quantized) GPU training histogram path is
`Backend::build_leaf_histograms_raw` → `build_leaf_histograms_resident_f32_on` →
`resident_raw_build_into(..., fixed_point=false)`, which now flows through the same default-on
autotune branch and lets `BUILD_TUNER` pick any `P ∈ {1,4,8,16}`. Per spike-007 / 13-CONTEXT,
`P>1` on the **f32** path widens GPU-vs-CPU divergence to ~2e-5 (f32 reduction-order), so:
(a) the histogram output of the default path is now **nondeterministic across runs** — which
`P` wins depends on cold-tune benchmark timing on the device — whereas before phase 13 the
heuristic deterministically chose `P=1` at the 50-feature production width; and (b) the
13-04 all-variant gate `kernel_parity_resident_build_all_pset_p_equals_anchor_on_hip` exercises
**only the u64 fixed-point path** (`build_fix_compact_resident_readback_f64_on`, gated at
1e-7). There is no test pinning the f32-resident build across all `P` to the CPU f64 anchor,
despite CONTEXT's instruction to "establish which build path is the autotuned default and
gate accordingly." Both paths are autotuned; only one is gated. A future kernel change that
widened f32 cross-`P` divergence would pass CI undetected.

This is a Warning, not a Blocker, because CLAUDE.md explicitly frames the `cubecl-hip` f32
path as a ~1e-6 *best-effort* gate with "residual f32-vs-f64 accumulation gaps documented per
phase," and spike-007's 2e-5 is one such documented gap — so it is not a contract violation.
The defect is the missing test gate plus newly-introduced, undocumented-at-the-call-site
nondeterminism on the default path.

**Fix:** Pick one:
```text
1. Add an all-PSET f32 parity test pinned to the CPU f64 anchor at the documented
   best-effort tolerance (e.g. ~1e-4/1e-5), mirroring the u64 all-PSET test, so the
   runtime-reachable f32 P set is gated.
2. Or restrict autotune-P to the order-independent u64 fixed-point path and keep the
   f32-resident path at deterministic P=1 (preserves run-to-run reproducibility).
3. Or, at minimum, add an inline note at the f32 autotune call site documenting that
   the f32 build output is P-dependent (~2e-5) and thus run-nondeterministic, so a
   future reader does not assume determinism.
```

### WR-02: parity gate hand-copies private candidate-set consts — silent drift risk

**File:** `crates/oracle-harness/tests/kernel_parity.rs:296,298` (`BUILD_PSET_MIRROR`, `SCAN_WSET_MIRROR`)
**Issue:** The all-variant parity tests hardcode `BUILD_PSET_MIRROR = &[1,4,8,16,32]` and
`SCAN_WSET_MIRROR = &[32,64,128,256]` as local copies of the **private** `histogram::BUILD_PSET`
and `split::SCAN_WSET`. Because the production consts are not exported, the test cannot
reference them, so if a maintainer adds or changes a `P`/`W` in the production set, the
"every autotune candidate is anchor-pinned" gate silently stops covering the new variant —
with no compile error. The whole point of the gate (autotune may pick any candidate at
runtime) is undermined by the copy going stale.
**Fix:** Export the candidate sets and import them so the gate tracks the source of truth:
```rust
// histogram.rs / split.rs
#[cfg(feature = "rocm")] pub const BUILD_PSET: &[u32] = &[1, 4, 8, 16, 32];
#[cfg(feature = "rocm")] pub const SCAN_WSET:  &[u32] = &[32, 64, 128, 256];
// kernel_parity.rs
use lgbm_compute::kernels::histogram::BUILD_PSET;
use lgbm_compute::kernels::split::SCAN_WSET;
```
If keeping them private is preferred, add a `const _: () = assert!(...)` equality check or a
test that compares lengths/contents against a re-exported accessor.

## Info

### IN-01: `Backend::prefers_autotune_launch_config` is dead code (no callers)

**File:** `crates/lgbm-compute/src/lib.rs:914-916` (default), `2198-2204` (RocmBackend override)
**Issue:** The trait method is defined (default-false + RocmBackend delegating to
`autotune_enabled()`), but a grep finds no call site — every launch path consults
`autotune::autotune_enabled()` directly. The 13-01 summary describes it as "the discoverable
seam," but as written it is an unused public trait method that every `Backend` implementor
must now nominally consider. Harmless (defaulted) but dead.
**Fix:** Either route the launch-time decision through `backend.prefers_autotune_launch_config()`
(making it load-bearing), or drop the method and keep `autotune_enabled()` as the single seam.

### IN-02: the "all-variant" parity gates bypass the actual tuner machinery

**File:** `crates/oracle-harness/tests/kernel_parity.rs:309,431`
**Issue:** `kernel_parity_resident_build_all_pset...` forces `LGBM_AUTOTUNE_FORCE_P`, which
short-circuits the tuner to a direct launch; `kernel_parity_fused_scan_all_wset...` forces
`LGBM_AUTOTUNE=0`+`LGBM_SCAN_CUBEDIM`, which takes the fallback direct launch. So neither
"all-variant" gate exercises `LocalTuner::execute` + `FreshOutGenerator`/`CloneInputGenerator`
— the code that actually runs by default in production. Tuner-path numerics are instead
covered by separate tests (`build_tuner_grad_conservation_fresh_vs_clone`,
`kernel_parity_fused_equals_per_feature_and_native` default-on). Coverage exists, but the
gate name overstates what these two tests verify.
**Fix:** Add one default-on (tuner-driven) run asserted against the CPU f64 anchor per knob,
so the production selection path itself is anchor-pinned, not only the per-variant kernel.

### IN-03: `read_autotune_build_picks` fragile cache parsing + hardcoded path

**File:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs:62-113`
**Issue:** Bench-only diagnostic: hand-rolled substring parsing of the cubecl cache `.json.log`
and a hardcoded `target/autotune/0.10.0/rocm_0` path (version literal + `rocm_0` derived from
`cache_namespace_id()="rocm:0"` with the colon mangled to an underscore). Brittle to any
cubecl version bump or cache-layout change; degrades silently to "<none persisted>". Acceptable
for an example, but the coupling to the on-disk format is undocumented as a maintenance hazard.
**Fix:** Note the version/path coupling in a comment, or gate the version string off a cubecl
constant if one is exposed.

### IN-04: `LaunchKey` Display reuses the `b` prefix for both `bucket` and `bins`

**File:** `crates/lgbm-compute/src/kernels/autotune.rs:46-48`
**Issue:** `write!(f, "LaunchKey(b{},f{},b{})", bucket, feats, bins)` renders both `bucket`
and `bins` with a `b` prefix (e.g. `LaunchKey(b10,f50,b256)`), which is ambiguous when
eyeballing logs. Cosmetic.
**Fix:** Use distinct prefixes, e.g. `LaunchKey(bucket={},feats={},bins={})`.

---

_Reviewed: 2026-06-26_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
