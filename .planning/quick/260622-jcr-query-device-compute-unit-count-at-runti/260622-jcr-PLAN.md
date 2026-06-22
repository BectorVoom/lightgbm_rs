---
phase: quick-260622-jcr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/Cargo.toml
  - crates/lgbm-compute/src/kernels/histogram.rs
autonomous: true
requirements: [QUICK-260622-jcr]
must_haves:
  truths:
    - "On the rocm build, the row-partition target_cubes is derived from the device's actual Compute Unit count (8 here → ~64), not the hardcoded 768."
    - "row_partition_count queries the CU count at most ONCE per process (cached), not per leaf."
    - "Resolution order is env-override → num_streaming_multiprocessors(Some) → hipGetDevicePropertiesR0600().multiProcessorCount → documented safe fallback (never silent 768)."
    - "The default (non-rocm) CPU build is byte-unchanged: no new code path, no new dependency compiled."
    - "The CPU f64 anchor bit-exact merge gate stays green (rocm-gated change cannot touch it)."
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/histogram.rs"
      provides: "rocm-gated runtime CU-count query + cached target_cubes feeding row_partition_count"
      contains: "hipGetDevicePropertiesR0600"
    - path: "crates/lgbm-compute/Cargo.toml"
      provides: "optional cubecl-hip-sys dep wired into the rocm feature"
      contains: "cubecl-hip-sys"
  key_links:
    - from: "row_partition_count"
      to: "rowpart_target_cubes()"
      via: "cached OnceLock CU-count query replaces the ROWPART_TARGET_CUBES const"
      pattern: "rowpart_target_cubes"
    - from: "rowpart_target_cubes()"
      to: "cubecl_hip_sys::hipGetDevicePropertiesR0600"
      via: "FFI read of hipDeviceProp_tR0600.multiProcessorCount"
      pattern: "multiProcessorCount"
---

<objective>
Replace the hardcoded `ROWPART_TARGET_CUBES = 768` ("~8 workgroups × 96 CUs (gfx1100)") in the
rocm row-partition occupancy tuning with a runtime query of the device's ACTUAL Compute Unit
count. The real device is an 8-CU APU (Radeon 860M / gfx1152 spoofed as gfx1100), so 768
over-provisions the occupancy target by 12×: `row_partition_count` computes `P = clamp(768/nf, 1, 16)`
calibrated for phantom 96-CU hardware.

The fix derives `target_cubes = num_cu × CUBES_PER_CU` (CUBES_PER_CU = 8, preserving spike-007's
"~8 workgroups/CU" intent) → 64 on this device, not 768.

Purpose: Stop calibrating GPU occupancy for hardware that isn't present. Primary deliverable is
CORRECTNESS (no phantom-hardware assumption), not a guaranteed speedup.
Output: A cached, rocm-gated CU-count query feeding `row_partition_count`; CPU build untouched.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/notes/gpu-is-spoofed-8cu-apu-not-gfx1100.md

# The target: row-partition tuning block + the existing unit test.
@crates/lgbm-compute/src/kernels/histogram.rs
# How the project already reads device props (Capabilities probe pattern).
@crates/lgbm-compute/src/runtime.rs
# Feature wiring (cpu default, rocm opt-in = cubecl/hip).
@crates/lgbm-compute/Cargo.toml

# VERIFIED API FACTS (do not re-derive; confirm they compile):
# - `cubecl-hip-0.10.0/src/runtime.rs:70` already calls
#   `cubecl_hip_sys::hipGetDevicePropertiesR0600(&mut props, device.index as hipDevice_t)`
#   then at line 147 sets `num_streaming_multiprocessors: None` — so the cubecl path returns
#   None on HIP. Use it FIRST when Some (forward-compatible), but it will be None here.
# - The FFI symbol is `hipGetDevicePropertiesR0600` (R0600 suffix, NOT bare hipGetDeviceProperties).
#   Struct = `hipDeviceProp_tR0600`, field `multiProcessorCount: c_int` (= 8 on this device).
# - cubecl-hip-sys is already a transitive dep (lockfile: cubecl-hip-sys 7.1.5280200); its own
#   build.rs selects the right bindings_NNNNN for the installed ROCm. Add it as a DIRECT optional
#   dep pinned to the same major (7.x) and gate it on the `rocm` feature so the cpu build never
#   pulls it.
# - The rocm device ordinal is 0 (`rocm_client()` binds `AmdDevice::new(0)`); pass device index 0.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add optional cubecl-hip-sys dep + cached runtime CU-count query</name>
  <files>crates/lgbm-compute/Cargo.toml, crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
In `crates/lgbm-compute/Cargo.toml`: add `cubecl-hip-sys = { version = "7", optional = true }`
under `[dependencies]` (matches the transitive 7.1.x already in Cargo.lock; its build.rs auto-selects
ROCm bindings). Wire it into the `rocm` feature: change `rocm = ["cubecl/hip"]` to
`rocm = ["cubecl/hip", "dep:cubecl-hip-sys"]`. The default `cpu` feature MUST NOT reference it, so a
plain `cargo build --release` neither compiles nor links cubecl-hip-sys (keeps SC#1: cpu build
toolchain-free).

In `crates/lgbm-compute/src/kernels/histogram.rs`, within the existing `#[cfg(feature = "rocm")]`
tuning block (around lines 622-659): introduce `const CUBES_PER_CU: u32 = 8;` (documents spike-007's
"~8 workgroups/CU" intent) and `const ROWPART_TARGET_CUBES_FALLBACK: u32 = 64;` (the documented safe
small default for an APU-class device when every query fails — explicitly NOT 768; comment that 768
was the phantom-96-CU value). Keep `ROWPART_P_MAX = 16` and `ROWPART_MIN_LEAF = 256_000` unchanged.

Add a cached query function (rocm-gated) `fn rowpart_target_cubes() -> u32` backed by a
`static TARGET: std::sync::OnceLock<u32> = OnceLock::new();` so the CU count is queried at most ONCE
per process (row_partition_count runs per leaf). Resolution order INSIDE the OnceLock init closure:
  (a) `LGBM_ROWPART_TARGET_CUBES` env var, parsed to u32 — explicit override for benching/A-B; if set
      and >0, return it verbatim (do NOT multiply by CUBES_PER_CU — it is the literal target).
  (b) else query the device CU count via a helper `query_num_cu() -> Option<u32>` and return
      `num_cu * CUBES_PER_CU` when Some.
  (c) else `ROWPART_TARGET_CUBES_FALLBACK`.

`query_num_cu()`:
  1. First try cubecl's reported value: `crate::runtime::rocm_client().properties().hardware
     .num_streaming_multiprocessors` (forward-compatible — returns None on cubecl-hip 0.10 today, but
     populated on cuda and possibly future hip). If `Some(n)` with n>0, return it.
  2. else FFI fallback: in an `unsafe` block, zero a `cubecl_hip_sys::hipDeviceProp_tR0600` (via
     `std::mem::zeroed()`), call `cubecl_hip_sys::hipGetDevicePropertiesR0600(&mut props, 0)` (device
     ordinal 0, matching `rocm_client`), and if the status == `cubecl_hip_sys::HIP_SUCCESS` and
     `props.multiProcessorCount > 0`, return `Some(props.multiProcessorCount as u32)`. Mirror the
     SAFETY-comment style of `cubecl-hip/src/runtime.rs:65-94` (props initialized by the call on
     success). On any non-success or non-positive count, return None.

Add a doc comment on `rowpart_target_cubes` recording: device is an 8-CU APU → target ≈ 64 (was 768
for a phantom 96-CU gfx1100); changing P alters the f32 partial-sum grouping (spike-007: P≥2 widens
GPU-vs-P=1 divergence to ~2e-5, WITHIN the ~1e-6-best-effort ROCm gate — the GPU path was never
bit-exact; the cpu f64 anchor is untouched).
  </action>
  <verify>
    <automated>cargo build --release 2>&1 | tail -3 && cargo build --release --features rocm 2>&1 | tail -5</automated>
  </verify>
  <done>Default `cargo build --release` succeeds and does NOT pull cubecl-hip-sys (cpu build
  unchanged). `cargo build --release --features rocm` compiles + links — the FFI call to
  `hipGetDevicePropertiesR0600` and the `cubecl-hip-sys` dep resolve. `rowpart_target_cubes()` and
  `query_num_cu()` exist, rocm-gated, with a OnceLock cache and the (a)→(b)→(c) resolution order.</done>
</task>

<task type="auto">
  <name>Task 2: Wire row_partition_count to the runtime target + update the unit test</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
Replace the two `ROWPART_TARGET_CUBES` (the const) references in `row_partition_count` (lines ~655-658)
with calls to `rowpart_target_cubes()`. Bind `let target = rowpart_target_cubes();` once at the top of
the function (after the min_leaf gate), then use `target` for both the `nf >= target → return 1` guard
and the `(target / nf).clamp(1, ROWPART_P_MAX)` result. Remove the now-unused
`const ROWPART_TARGET_CUBES` (it is superseded by the runtime query + the FALLBACK const). Keep
`ROWPART_MIN_LEAF`, `LGBM_ROWPART_MIN` override, the `num_features == 0`/`leaf_rows < min_leaf` early
returns, and `ROWPART_P_MAX` clamp byte-identical.

Update the existing `#[cfg(feature = "rocm")]` unit test `row_partition_count_heuristic` (lines
~2806-2824): it imports and references `ROWPART_TARGET_CUBES` which no longer exists. Rewrite it to be
robust to a runtime target by binding `let target = super::rowpart_target_cubes();` at the test top and
expressing all assertions in terms of `target` instead of the deleted const (e.g. saturated-feature
case uses `target as usize`; the few-features P-value asserts `(target / 50).clamp(1, ROWPART_P_MAX)`).
To make the test deterministic regardless of host hardware, set `LGBM_ROWPART_TARGET_CUBES` to a known
value at the START of the test via `std::env::set_var` BEFORE the first `rowpart_target_cubes()` call —
BUT note the OnceLock caches the first read process-wide. Two options, pick the cleaner: (i) factor the
resolution logic into a pure `fn resolve_target_cubes(env: Option<u32>, queried: Option<u32>) -> u32`
that the OnceLock wrapper and the test BOTH call (test passes explicit args, no env/OnceLock races —
PREFERRED, mirrors the "pure CPU logic, unit-testable without a GPU" doc on row_partition_count); or
(ii) keep the env approach but accept the test asserts against whatever target the cache resolved.
Prefer (i): add `resolve_target_cubes` (pure, takes the env override and the queried CU count, applies
the (a)→(b)→(c) order with CUBES_PER_CU/FALLBACK) and unit-test IT directly plus the heuristic against
a forced target. Keep the test asserting: small leaf → 1; degenerate nf=0 → 1; saturated nf≥target → 1;
large leaf + few features → clamped tuned P in [2, P_MAX].
  </action>
  <verify>
    <automated>cargo test -p lgbm-compute --lib --features rocm row_partition 2>&1 | tail -15 && cargo test -p lgbm-compute --lib 2>&1 | tail -5</automated>
  </verify>
  <done>`row_partition_count` reads `rowpart_target_cubes()` (no `ROWPART_TARGET_CUBES` const
  remains). The rocm-gated unit test compiles and passes against the runtime/forced target. The
  default `cargo test -p lgbm-compute --lib` (cpu) compiles and passes — no rocm symbols leak into the
  cpu build.</done>
</task>

<task type="auto">
  <name>Task 3: Prove the bit-exact gate + confirm CU=8 / P change on hardware</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
This task is verification-only (no production code change beyond an optional tiny rocm-gated test
asserting the queried CU count). Run, in order:

1. BIT-EXACT MERGE GATE (rocm-gated change must leave it trivially green):
   `cargo test -p lgbm-treelearner --lib` and `cargo test -p oracle-harness`. Both must stay GREEN —
   the change is `#[cfg(feature = "rocm")]`-only and the default test build never compiles it, so the
   CPU f64 anchor is byte-unchanged. If oracle-harness has a known pre-existing failure (e.g.
   goss_parity_matrix per STATE DEF-08-OOS-01), note it as pre-existing and not introduced here.

2. CONFIRM CU=8 ON THIS DEVICE: add a small `#[cfg(feature = "rocm")] #[test] fn queried_cu_count_is_8()`
   (or a one-off `eprintln!` in the heuristic test) that calls `query_num_cu()` and asserts it returns
   `Some(8)` on this box (Radeon 860M, 8 CUs per the spoofed-APU note), and that with no env override
   `rowpart_target_cubes()` ≈ 64 (8 × CUBES_PER_CU), NOT 768. Run
   `cargo test -p lgbm-compute --lib --features rocm queried_cu_count_is_8 -- --nocapture`. If the FFI
   query is environment-flaky, downgrade the hard assert to an `eprintln!` + a soft `>0` check and
   record the printed value in the SUMMARY. Document the observed CU count and resulting target_cubes.

3. GPU A/B WHERE P ACTUALLY CHANGES (1M×50): at 1M×50 the OLD gate gives `P=clamp(768/50)=15`, the NEW
   gives `clamp(64/50)=1`. Run the bench harness before/after by env-overriding the target to compare
   the two regimes on the SAME binary (avoids a rebuild between arms):
     - OLD regime: `LGBM_ROWPART_TARGET_CUBES=768 LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide \
       LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=50 cargo run --release --features rocm \
       --example bench_gpu_vs_cpu`
     - NEW regime: same command with `LGBM_ROWPART_TARGET_CUBES=64`.
   Machine is thermal-sensitive — warm up first, interleave arms (run each ≥3×, alternating), report
   medians. HONEST reporting: state whether reducing over-subscription on the 8-CU APU helps the GPU
   build, is a wash, or regresses. The PRIMARY deliverable is correctness (no phantom-hardware
   assumption) — a speedup is NOT required; a wash is an acceptable, expected outcome on an 8-CU iGPU
   sharing DDR5 with a 16-thread CPU. Record numbers + verdict in the SUMMARY.
  </action>
  <verify>
    <automated>cargo test -p lgbm-treelearner --lib 2>&1 | tail -5 && cargo test -p oracle-harness 2>&1 | tail -8</automated>
  </verify>
  <done>Bit-exact gate (`lgbm-treelearner` lib + `oracle-harness`) is GREEN (any failure is
  pre-existing per STATE.md, explicitly noted, not introduced). The queried CU count is recorded as 8
  (target_cubes ≈ 64) on this device. The 1M×50 GPU A/B (target 768 vs 64) is run interleaved and its
  result + honest verdict (help / wash / regress) is recorded in the SUMMARY. ROCm parity note: P
  change alters f32 partial-sum grouping within the ~1e-6 best-effort gate; no bit-exactness claimed
  for the GPU path.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| process → HIP runtime (FFI) | `hipGetDevicePropertiesR0600` reads device properties via the rocm driver; only reached on the opt-in `rocm` build, device ordinal 0 (matches `rocm_client`). |
| env → tuning param | `LGBM_ROWPART_TARGET_CUBES` is an operator-set occupancy knob; bounds-clamped downstream by `ROWPART_P_MAX`. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-jcr-01 | Tampering | env `LGBM_ROWPART_TARGET_CUBES` | accept | Parsed to u32; non-parse/≤0 falls through to query path; result clamped by `ROWPART_P_MAX`. Bench-only knob, no security surface. |
| T-jcr-02 | Denial of Service | per-leaf CU query | mitigate | `OnceLock` caches the FFI query → at most one `hipGetDevicePropertiesR0600` call per process, not per leaf. |
| T-jcr-03 | Information disclosure / crash | unsafe FFI on zeroed `hipDeviceProp_tR0600` | mitigate | Mirror cubecl-hip's own SAFETY pattern: props initialized by the call on success; status checked == HIP_SUCCESS before reading `multiProcessorCount`; non-success → None → documented FALLBACK (never UB, never silent 768). |
| T-jcr-SC | Tampering | cubecl-hip-sys dep add | mitigate | cubecl-hip-sys is already a transitive dep of cubecl-hip (lockfile 7.1.5280200, checksum-pinned); promoting it to a direct OPTIONAL `rocm`-gated dep adds no new supply-chain surface to the default cpu build. No new package install — already vendored in the registry cache. |
</threat_model>

<verification>
- `cargo build --release` (cpu) succeeds AND does not compile cubecl-hip-sys (default build unchanged).
- `cargo build --release --features rocm` compiles + links the FFI call.
- `cargo test -p lgbm-compute --lib` (cpu) and `... --features rocm` pass (heuristic + resolution unit tests).
- Bit-exact merge gate: `cargo test -p lgbm-treelearner --lib` + `cargo test -p oracle-harness` GREEN (pre-existing failures noted, not introduced).
- Runtime confirmation: `query_num_cu()` returns 8 on this device → `rowpart_target_cubes()` ≈ 64 (not 768).
- GPU A/B at 1M×50 (target 768 vs 64) run interleaved; honest verdict recorded.
</verification>

<success_criteria>
- The hardcoded 768 phantom-96-CU assumption is gone; the row-partition target derives from the
  device's ACTUAL CU count at runtime (8 → ~64 here), cached once per process.
- Resolution order env → num_streaming_multiprocessors(Some) → hipGetDevicePropertiesR0600 → documented
  FALLBACK (never silent 768).
- CPU build byte-unchanged (no new dep, no new code path); bit-exact merge gate trivially green.
- ROCm parity: GPU path's P change stays within the ~1e-6 best-effort gate; no bit-exactness claimed
  (it was never bit-exact). CPU f64 anchor untouched.
</success_criteria>

<output>
Create `.planning/quick/260622-jcr-query-device-compute-unit-count-at-runti/260622-jcr-SUMMARY.md` when done.
</output>
