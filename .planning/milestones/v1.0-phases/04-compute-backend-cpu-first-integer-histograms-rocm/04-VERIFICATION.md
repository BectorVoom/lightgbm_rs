---
phase: 04-compute-backend-cpu-first-integer-histograms-rocm
verified: 2026-06-06T00:00:00Z
status: passed
score: 5/5 success-criteria verified (+ all 4 plans' must-haves)
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
follow_ups:  # NOT phase blockers — latent issues confirmed for Phase-5 attention
  - id: CR-01/CR-02
    summary: "Oracle test host-side L1 sign helper (kernel_parity.rs:501,1019) uses f64::signum() (±1 at zero) instead of C++ Sign() (0 at zero). Production gain::threshold_l1 is CORRECT (uses select-based Sign). Latent: no committed golden has a zero-gradient L1 case, so the oracle's divergent sign is never exercised. Risk: a future zero-gradient L1 case could falsely fail OR mask a real kernel regression."
    affects_goal: false
    recommended_for: "Phase 5 (or a quick test-fidelity fix): make leaf_gain/leaf_gain_f32 in kernel_parity.rs use (x>0)-(x<0) or delegate to lgbm_compute::gain::get_leaf_gain; add a sum_gradient==0 && lambda_l1>0 golden case."
  - id: WR-02
    summary: "REVERSE split scan uses rev_count = num_bin-1 with t = t_start - k (t_start = num_bin-1-offset). For offset>=2 the final t values go negative and (t as usize) wraps to a huge index, reading hist[bi] out of bounds (backend-dependent UB on cubecl-cpu). Latent: the only offset>0 golden is default_bin_skip (offset=1, num_bin=5 → min t=0), so negative t is never reached."
    affects_goal: false
    recommended_for: "Phase 5 (BEFORE real offset>=2 feature-group layouts appear): clamp the REVERSE loop bound or guard the body on t>=0; add a golden with offset>=2."
---

# Phase 4: Compute Backend (CPU-first integer histograms → ROCm) Verification Report

**Phase Goal:** An isolated `lgbm-compute` backend whose f32 histogram, split-scan, and data-partition kernels produce results matching CPU and ROCm within ~1e-6 — the CubeCL-churn containment boundary.
**Verified:** 2026-06-06
**Status:** passed
**Re-verification:** No — initial verification
**Mode:** mvp (goal is a capability statement with 5 explicit Success Criteria; verified against those + all four plans' must_haves)

## Goal Achievement

This verifier did NOT trust SUMMARY.md claims. The load-bearing gates were re-executed
independently: `cargo test --workspace` (cpu bit-exact hard gate), and — because this
verifier host happens to be the same class of gfx1100 ROCm 7.1.1 machine documented in
04-ROCM-GAPS.md — the ROCm half (`rocm_smoke` + the hip parity layer) was re-run on the
REAL GPU rather than accepted from the SUMMARY.

### Observable Truths (ROADMAP Success Criteria — the contract)

| # | Truth (SC) | Status | Evidence |
|---|------------|--------|----------|
| 1 | All CubeCL usage behind one `lgbm-compute` `Backend` trait; no crate above it names a CubeCL runtime; CPU-only build needs no ROCm toolchain | ✓ VERIFIED | `Backend` trait in `lib.rs:33` binds `type Runtime: cubecl::Runtime`; `cmp01_containment` test (2 sub-tests) green; `cargo tree -p lgbm-core/-dataset/-model` shows NO cubecl above the seam; `cargo build -p lgbm-compute --no-default-features --features cpu` exits 0 (no ROCm); cubecl pulled into oracle-harness only via `[dev-dependencies]` |
| 2 | f32 histogram / best-split / data-partition kernels run on cubecl-cpu reference and match a sequential reference within ~1e-6 | ✓ VERIFIED (exceeds — BIT-EXACT) | The cpu anchor is the f64 single-owner ordered fold proven bit-stable (D-04a spike, 25 launches + vs C++-order sequential fold). `kernel_parity` 4/4 green: histogram/split/subtract `compare_exact_f64_bits` (BIT-EXACT, not merely 1e-6), partition `compare_exact_u32`. Re-run by this verifier: all pass |
| 3 | Same kernels run on cubecl-hip (ROCm), selectable by Cargo feature/runtime config, matching the CPU backend within ~1e-6 (f32) | ✓ VERIFIED (re-run on real gfx1100) | `cargo build -p lgbm-compute --features rocm` exits 0 (no ROCM_PATH override). `rocm_smoke` 2/2 pass on the GPU. Hip parity 4/4 pass: partition bit-exact, subtract within tol, histogram/split within the two-tier gate. f32-vs-f64 accumulation gaps surfaced per-case (max abs-diff single_bin_pileup 9.77e-4 ≈1 f32 ULP, split fwd 3.81e-6 / rev 7.63e-6) — documented in 04-ROCM-GAPS.md, best-effort per D-03/D-03a, NOT a phase blocker |
| 4 | CUDA warp-level reductions via CubeCL `Plane` API with startup capability-gating (`Plane::Ops`, f64, atomics) + deterministic sequential fallback | ✓ VERIFIED | `runtime.rs` `probe_capabilities` queries `client.features().plane.contains(Plane::Ops)`, `supports_type` (f64), `atomic_type_usage(...)` (f32-atomic); `capability` test asserts cpu matrix (Plane=false → `ReducePath::Sequential`); re-run `rocm_smoke` asserts the real gfx1100 matrix (Plane YES / f64 NO / atomic YES / plane_size 32) → `ReducePath::Plane` + `AccumulateType::F32`. (See IN-03 note below — Plane reduce is plumbed/gated but kernels currently launch single-owner; this is forward-looking, the gate + fallback are real and tested) |
| 5 | Oracle suite executes and passes on the ROCm backend for histogram/split/partition (separate gates) | ✓ VERIFIED (re-run on real gfx1100) | CPU gate (hard bar): `kernel_parity` 4/4 bit-exact. ROCm gate (separate): `cargo test -p oracle-harness --features rocm --test kernel_parity hip::` → 4/4 pass on the GPU, each `HIP PARITY GAP` line printed to stderr (no silent pass). CPU and ROCm are distinct test gates; the CPU gate is unaffected by the rocm-gated code |

**Score:** 5/5 Success Criteria verified.

### Plan Must-Haves (per-plan truths — all VERIFIED)

| Plan | Must-have | Status | Evidence |
|------|-----------|--------|----------|
| 04-01 | D-04a bit-determinism (N≥20 launches byte-identical + bit-exact vs sequential fold) | ✓ | `determinism_spike` N_LAUNCHES=25; `determinism_spike_cpu_bit_exact` green |
| 04-01 | Capability gate selects Sequential vs Plane via features()/properties() | ✓ | `capability_cpu_matrix` green; runtime.rs queries confirmed |
| 04-01 | OOB bin / length-mismatch → typed ComputeError, never panic | ✓ | `boundary_validation_returns_typed_errors` green; ComputeError 4 variants |
| 04-01 | CMP-01 containment guard passes | ✓ | `cmp01_containment` 2/2 green |
| 04-02 | construct_histograms coarse whole-kernel op (f64 stride-2 cells) | ✓ | `Backend::construct_histograms` + `CpuBackend` impl in lib.rs |
| 04-02 | cubecl-cpu histogram bit-exact vs committed C++ golden (18 D-02a cases) | ✓ | `kernel_parity_histogram_bit_exact_on_cpu` green; histogram.txt COUNTS=18 (dense+sparse+defbin+w4/8/16/32+pileup) |
| 04-02 | kernel-capture idempotent; golden covers D-02a paths | ✓ | golden committed; coverage confirmed by case labels (regen requires C++ toolchain, not test) |
| 04-03 | find_best_split: in-kernel gain math, kEpsilon/2*kEpsilon, both REVERSE (t-1+offset) & FORWARD (t+offset) branches, exact gate order | ✓ | split.rs:229 (t-1+offset) & 286/466 (t+offset); split.txt has reverse_winner + forward_winner cases; `kernel_parity_split_bit_exact_on_cpu` green |
| 04-03 | subtract_histograms (FeatureHistogram::Subtract) + data_partition (stable reorder + split_point) | ✓ | Backend methods + kernels present; `kernel_parity_subtract_*` + `kernel_parity_partition_*` green |
| 04-03 | split/partition/subtract bit-exact vs goldens; Sequential on cpu | ✓ | 3 parity layers green; capability gate Sequential on cpu |
| 04-04 | rocm runtime selectable; hip kernels run on gfx1100 | ✓ | rocm build + smoke + parity re-run on real GPU by this verifier |
| 04-04 | hip capability gate detects hip matrix, f32 path on hip / f64 on cpu | ✓ | `rocm_capability_matrix_gfx1100` green; AccumulateType F32/F64 gate |
| 04-04 | hip f32 vs cpu f64 anchor within ORACLE_TOL (separate gate, compare_within) | ✓ | hip parity layer uses compare_within(ORACLE_TOL); two-tier gate |
| 04-04 | residual ROCm gap documented, no silent pass (D-03a) | ✓ | 04-ROCM-GAPS.md records G-04-01/G-04-02 + per-case abs-diff; stderr gap lines confirmed on re-run |
| 04-04 | CPU-only build still passes with no ROCm toolchain (SC#1) | ✓ | `cargo test --workspace` green; no-default-features cpu build exits 0 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-compute/src/error.rs` | ComputeError 4-variant thiserror enum | ✓ VERIFIED | 4 variants; unit tests green; no anyhow |
| `crates/lgbm-compute/src/runtime.rs` | runtime selection + capability gate | ✓ VERIFIED | Capabilities/ReducePath/AccumulateType/probe_capabilities; cpu+rocm paths |
| `crates/lgbm-compute/src/gain.rs` | ThresholdL1/leaf_gain/split_gains + f32 mirrors + GainConfig/SplitInfo | ✓ VERIFIED | `Sign`-correct via select; verbatim C++ citations |
| `crates/lgbm-compute/src/kernels/{histogram,split,partition,subtract}.rs` | the 4 #[cube] kernels + cpu launchers + f32 mirrors | ✓ VERIFIED | all present, wired to CpuBackend, parity-green |
| `crates/lgbm-compute/tests/{determinism_spike,capability,cmp01_containment,rocm_smoke}.rs` | the gate tests | ✓ VERIFIED | all green (rocm_smoke cfg-gated, runs only under --features rocm; re-run 2/2 on GPU) |
| `crates/oracle-harness/tests/kernel_parity.rs` | cpu bit-exact + hip ~1e-6 parity layers | ✓ VERIFIED | 4 cpu layers green; hip layer re-run 4/4 on GPU |
| `crates/oracle-harness/tests/fixtures/kernels/{histogram,split,partition,subtract}.txt` | committed C++ goldens | ✓ VERIFIED | all committed; histogram 18-case, split 5-case (reverse/forward/defbin/L1/no-split) |
| `xtask/cpp/kernel_capture.cpp` + CMakeLists | header-only C++ transcription harness | ✓ VERIFIED | present; transcribes ConstructHistogram/FindBestThreshold/Subtract/SplitInner |
| `04-ROCM-GAPS.md` | D-03a documented-gap ledger | ✓ VERIFIED | real-hardware results; numbers reproduced by this verifier's re-run |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| kernel_parity.rs | Backend::construct_histograms/find_best_split/data_partition/subtract_histograms | drive on goldens, compare | ✓ WIRED (4/4 cpu green) |
| kernel_parity.rs | compare_exact_f64_bits / compare_exact_u32 (full module path) | bit-exact cpu gate | ✓ WIRED |
| kernel_parity.rs (hip) | compare_within(ORACLE_TOL) | hip-f32-vs-cpu-f64-anchor ~1e-6 separate gate | ✓ WIRED (re-run on GPU) |
| runtime.rs | cubecl_hip::HipRuntime + AmdDevice{0} | #[cfg(feature=rocm)] selection | ✓ WIRED (rocm build + smoke green) |
| CpuBackend | kernels::*::*_cpu launchers | trait dispatch | ✓ WIRED |

### Behavioral Spot-Checks (executed by this verifier, not trusted from SUMMARY)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| cpu bit-exact gate | `cargo test --workspace` | all green incl. kernel_parity 4/4 | ✓ PASS |
| lgbm-compute unit + integration | `cargo test -p lgbm-compute` | 17 unit + capability + cmp01(2) + determinism(2) green | ✓ PASS |
| CPU-only build (SC#1) | `cargo build -p lgbm-compute --no-default-features --features cpu` | exit 0 | ✓ PASS |
| containment (no cubecl above seam) | `cargo tree -p lgbm-core/-dataset/-model \| grep cubecl` | none | ✓ PASS |
| rocm build | `cargo build -p lgbm-compute --features rocm` | exit 0, no ROCM_PATH override | ✓ PASS |
| rocm smoke on gfx1100 | `cargo test -p lgbm-compute --features rocm --test rocm_smoke` | 2/2 pass | ✓ PASS |
| hip parity on gfx1100 | `cargo test -p oracle-harness --features rocm --test kernel_parity hip::` | 4/4 pass; gaps surfaced to stderr | ✓ PASS |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| CMP-01 | lgbm-compute backend trait isolating all device ops | ✓ SATISFIED | Backend seam + cmp01_containment green; no upper crate names cubecl |
| CMP-02 | cubecl-cpu deterministic reference path | ✓ SATISFIED | D-04a anchor proven; 4 kernels bit-exact on cpu |
| CMP-03 | cubecl-hip selectable via Cargo feature | ✓ SATISFIED | rocm feature → HipRuntime; re-run on gfx1100 |
| CMP-04 | Plane API + capability gating + sequential fallback | ✓ SATISFIED | probe_capabilities + gate; cpu→Sequential, hip matrix asserted (re-run) |
| CMP-05 | histogram/split/partition kernels at ~1e-6 (f32) | ✓ SATISFIED | full kernel set bit-exact cpu + hip within tol/documented gap |
| ORA-04 | Oracle suite passes on ROCm (separate gates) | ✓ SATISFIED | cpu hard gate + ROCm gate both re-run green |

No orphaned requirements: all 6 IDs declared in plan frontmatter map to Phase 4 in REQUIREMENTS.md and are marked `[x]` with evidence.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No TBD/FIXME/XXX in any phase-modified src | ✓ none | — |
| — | — | No TODO/HACK/PLACEHOLDER/unimplemented (production) | ✓ none | — |
| kernels/{partition,subtract}.rs | 162-201 | `.unwrap()` | ℹ️ Info | All inside `#[cfg(test)]` modules — not production launch paths |

### Code-Review Critical Findings Assessment (per task request)

Both 04-REVIEW.md Critical findings were independently confirmed and assessed against the
phase GOAL/requirements. **Neither blocks the Phase-4 goal as verified** — both are LATENT
(not exercised by the current committed golden corpus; the suite is green and the cpu
bit-exact gate — the hard completion bar — passes).

- **CR-01 / CR-02 (oracle `signum` ≠ C++ `Sign`):** CONFIRMED at `kernel_parity.rs:501` and
  `:1019` — the *oracle test* host-side L1 sign helper uses `f64::signum()` (±1 at zero).
  The *production* `gain::threshold_l1` is CORRECT — it uses the `select`-based
  `(s>0)-(s<0)` form (0 at zero); `grep signum crates/lgbm-compute/src` → none. Latent
  because no committed golden has a zero-gradient L1 case (the lone L1 case `l1_forward`
  has `sum_gradient=-2.0`). Does NOT affect the verified goal. Recorded as a Phase-5
  follow-up (the oracle could false-fail or mask a regression on a future zero-gradient L1
  case; the fix is trivial and the gain MATH contract should be made trustworthy before
  Phase-5 consumes it).

- **WR-02 (unbounded negative-t reverse scan index):** CONFIRMED — `split.rs` REVERSE uses
  `rev_count = (num_bin-1).max(0)` with `t = t_start - k` (`t_start = num_bin-1-offset`). For
  `offset>=2` the trailing `t` goes negative and `(t as usize)` (lines 186/251) wraps to a
  huge index → potential OOB read of `hist[bi]` (backend-dependent on cubecl-cpu). Latent:
  the only `offset>0` golden is `default_bin_skip` (offset=1, num_bin=5 → min t=0). Does NOT
  affect the verified goal for the current corpus. Recorded as a Phase-5 follow-up — this is
  a real correctness landmine the moment real `offset>=2` feature-group layouts appear, and
  should be fixed (clamp the loop bound / guard on `t>=0`) before Phase-5 tree growth drives
  these inputs.

### Human Verification Required

None outstanding. The 04-04 Task-3 `checkpoint:human-verify` (run the ROCm oracle on the
physical gfx1100) was already executed and documented in 04-ROCM-GAPS.md — AND this verifier
independently re-ran it on the same-class GPU and reproduced the documented per-case
abs-diffs. No further human verification is required for Phase-4 goal achievement.

### Gaps Summary

No goal-blocking gaps. All 5 ROADMAP Success Criteria are observably true in the codebase;
all four plans' must_haves are verified; all 6 requirement IDs are satisfied with reproduced
evidence. The cpu bit-exact gate (the non-negotiable hard completion bar) is green, and the
best-effort ROCm gate runs on real gfx1100 hardware with its f32-vs-f64 accumulation gap
explicitly documented (no silent pass) per D-03a — an acceptable follow-up, not a blocker.

Two latent code-review findings (CR-01/CR-02 oracle sign-helper fidelity; WR-02 negative-t
reverse-scan bound) are confirmed real but unexercised by the current corpus and therefore do
not block the Phase-4 goal. They are recorded in the `follow_ups` frontmatter for Phase-5
attention, where real `offset>=2` layouts and zero-gradient L1 cases will exercise both paths.

---

_Verified: 2026-06-06_
_Verifier: Claude (gsd-verifier) — cpu gate + ROCm gate both re-executed independently_
