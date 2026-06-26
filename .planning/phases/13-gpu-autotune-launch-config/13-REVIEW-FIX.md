---
phase: 13-gpu-autotune-launch-config
fixed_at: 2026-06-27T00:00:00Z
review_path: .planning/phases/13-gpu-autotune-launch-config/13-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 13: Code Review Fix Report

**Fixed at:** 2026-06-27
**Source review:** .planning/phases/13-gpu-autotune-launch-config/13-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (2 Warning, 4 Info — `fix_scope: all`)
- Fixed: 6
- Skipped: 0

All fixes preserve the project's numerical contract: every change is either
`#[cfg(feature = "rocm")]`-gated or doc/comment-only, so the default CPU build and the
f64 deterministic anchor are byte-unchanged. The CPU merge gate
(`cargo test -p lgbm-treelearner --lib`) stays green at **77 passed / 0 failed**, the
default `cargo build -p lgbm-compute` compiles, and the rocm build
(`cargo build -p lgbm-compute --features rocm`) plus the parity test crate
(`cargo test -p oracle-harness --features rocm --test kernel_parity --no-run`) compile.

## Fixed Issues

### WR-01: f32-resident default build autotuned across `P` with no all-variant parity gate

**Files modified:** `crates/lgbm-compute/src/kernels/histogram.rs`, `crates/oracle-harness/tests/kernel_parity.rs`
**Commit:** d6a1ddb
**Status:** fixed: requires human verification (on-device rocm parity run is a follow-up)
**Applied fix:** Chose guardrail option (a) — extend the 13-04 gate rather than disable
f32 autotune. Added `kernel_parity_resident_build_all_pset_f32_p_equals_anchor_on_hip`,
which forces every runtime-reachable `P` in `BUILD_PSET` on the f32 RAW resident build
(`build_leaf_histograms_resident_f32_on`) and pins each to the CPU f64 anchor
(`construct_histograms_cpu`, def-f8u-01 — anchored to the f64 fold, never GPU-vs-GPU) at
a documented best-effort envelope `F32_BUILD_REL_GATE = 1e-4` (~5× over spike-007's
observed ~2e-5 f32 cross-`P` reduction-order gap; the u64 path keeps its 1e-7 bit-exact
gate, untouched). Also added an inline note at the f32 autotune call site documenting
that the f32 build output is `P`-dependent (~2e-5) and therefore run-to-run
nondeterministic — within the contract's ~1e-6 best-effort f32 gap, not bit-exact.
**Verification:** Compiles under `--features rocm` (`--no-run`). Because the test runs
on the rocm GPU, its tolerance could not be exercised on-device here — on-device parity
re-validation is the documented follow-up. Logic/tolerance choice flagged for human
confirmation.

### WR-02: parity gate hand-copied private candidate-set consts (silent drift risk)

**Files modified:** `crates/lgbm-compute/src/kernels/histogram.rs`, `crates/lgbm-compute/src/kernels/split.rs`, `crates/oracle-harness/tests/kernel_parity.rs`
**Commit:** a7e1806
**Applied fix:** Promoted `histogram::BUILD_PSET` and `split::SCAN_WSET` from private to
`pub` (still `#[cfg(feature = "rocm")]`-gated) and replaced the `BUILD_PSET_MIRROR` /
`SCAN_WSET_MIRROR` hand-copies in `kernel_parity.rs` with aliased `use` imports of the
production consts, so the gate now tracks the source of truth — adding a `P`/`W`
automatically extends the sweep instead of silently leaving a variant ungated.
**Verification:** rocm build + parity test crate compile clean.

### IN-01: `Backend::prefers_autotune_launch_config` is dead code

**Files modified:** `crates/lgbm-compute/src/lib.rs`
**Commit:** 065bc55
**Applied fix:** Confirmed zero callers repo-wide (only the trait default + the
RocmBackend override existed). Removed both; `autotune::autotune_enabled()` remains the
single seam every launch path already consulted.
**Verification:** default + rocm builds compile.

### IN-02: "all-variant" parity gate names overstate tuner-path coverage

**Files modified:** `crates/oracle-harness/tests/kernel_parity.rs`
**Commit:** 309348c
**Applied fix:** Low-risk doc-accuracy fix (per guardrails). Added SCOPE notes to both
`kernel_parity_resident_build_all_pset_*` and `kernel_parity_fused_scan_all_wset_*`
clarifying they FORCE each variant (direct launch / explicit-env fallback) and pin
per-variant KERNELS — not `LocalTuner::execute` — and pointing at the default-on tests
that do cover the live tuner-selection path
(`build_tuner_grad_conservation_fresh_vs_clone`,
`kernel_parity_fused_equals_per_feature_and_native`). Did not add a new on-device
tuner-driven test (could not be validated here); documented as the residual follow-up.
**Verification:** parity test crate compiles under rocm.

### IN-03: `read_autotune_build_picks` fragile cache parsing + hardcoded path

**Files modified:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs`
**Commit:** 046ecb2
**Applied fix:** Added a MAINTENANCE HAZARD comment documenting the coupling of the
hardcoded `target/autotune/0.10.0/rocm_0` path to the cubecl version literal, the
`rocm:0`→`rocm_0` namespace mangling, and the cubecl `.json.log` line shape — all of
which degrade silently to `<none persisted>` on drift. cubecl exposes no public constant
for the version/namespace today, so the literal stays hand-maintained (bench-only
diagnostic, never gates pass/fail).
**Verification:** example compiles under `--features rocm`.

### IN-04: `LaunchKey` Display reused the `b` prefix for `bucket` and `bins`

**Files modified:** `crates/lgbm-compute/src/kernels/autotune.rs`
**Commit:** ef10ef5
**Applied fix:** Changed `LaunchKey(b{},f{},b{})` to
`LaunchKey(bucket={},feats={},bins={})`. Display is log-only (the persisted cache key
uses serde and the bench parses the JSON object), so this is purely cosmetic and does
not affect cache identity.
**Verification:** rocm build compiles.

## Follow-ups (not blocking)

- **On-device rocm re-validation** of the new f32 all-PSET parity gate (WR-01): run
  `cargo test -p oracle-harness --features rocm --test kernel_parity
  kernel_parity_resident_build_all_pset_f32_p_equals_anchor_on_hip` on the ROCm GPU and
  confirm the `1e-4` envelope holds (tighten toward `~2e-5` if the device run is
  comfortably inside it).
- **Optional (IN-02):** add one default-on, tuner-driven (`LocalTuner::execute`) run
  asserted against the CPU f64 anchor per knob, so the production selection path itself
  is anchor-pinned rather than only the per-variant kernels.

---

_Fixed: 2026-06-27_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
