---
phase: quick-260626-igc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/Cargo.toml
  - crates/lgbm-compute/src/runtime.rs
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm/Cargo.toml
  - crates/lgbm/src/booster.rs
autonomous: true
requirements: ["260626-igc"]

must_haves:
  truths:
    - "`cargo check -p lgbm-compute --features cuda` and `cargo check -p lgbm --features cuda` build clean (CUDA expected to compile per locked decision #3)."
    - "`cargo check` (default cpu) and `cargo check --features rocm` (both crates) remain unbroken — byte-identical behavior on the rocm path."
    - "CudaBackend and WgpuBackend dispatch the SAME runtime-generic GPU kernels RocmBackend uses (construct_histograms_lds_f32_on, find_best_split_f64_on, data_partition_on, subtract_histograms_f64_on) via the `*_on<R: cubecl::Runtime>` launchers — no new/forked kernels."
    - "wgpu check outcome is recorded honestly: `--features wgpu` MAY fail to compile (WGSL has no f32 atomics, locked decision #3); if it fails, the SUMMARY documents what/where/why instead of swapping kernels."
  artifacts:
    - path: "crates/lgbm-compute/src/runtime.rs"
      provides: "CudaRuntime/WgpuRuntime type aliases + cuda_client()/wgpu_client() ctors, mirroring rocm_client()"
      contains: "CudaRuntime"
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "CudaBackend + WgpuBackend impl Backend with type Runtime = runtime::CudaRuntime|WgpuRuntime"
      contains: "CudaBackend"
    - path: "crates/lgbm/src/booster.rs"
      provides: "cfg cascade selecting RocmBackend > CudaBackend > WgpuBackend > CpuBackend"
      contains: "CudaBackend"
  key_links:
    - from: "crates/lgbm/src/booster.rs"
      to: "crates/lgbm-compute/src/runtime.rs"
      via: "cfg-gated `use lgbm_compute::runtime::cuda_client` + `let backend = CudaBackend; let client = cuda_client();`"
      pattern: "cuda_client"
    - from: "crates/lgbm-compute/src/lib.rs"
      to: "crates/lgbm-compute/src/kernels/histogram.rs"
      via: "CudaBackend::construct_histograms dispatches construct_histograms_lds_f32_on::<CudaRuntime>"
      pattern: "construct_histograms_lds_f32_on"
---

<objective>
Add `cuda` (cubecl-cuda / `cubecl::cuda::CudaRuntime`) and `wgpu` (cubecl-wgpu /
`cubecl::wgpu::WgpuRuntime`) compute backends to `lgbm-compute`, wired exactly like the
existing ROCm backend and REUSING the runtime-generic `#[cube]` GPU kernels. Surface
both backends through the `lgbm` facade's feature-switched backend selection.

Scope is COMPILE-GATED WIRING ONLY (locked decision #2). The deliverable bar is the
`cargo check` matrix below — NO runtime tests, NO parity tests, NO benchmarks (this
machine has no NVIDIA GPU; the AMD GPU is out of scope for this task).

Purpose: let downstream builds target CUDA and WGPU runtimes through the same Backend
seam, without forking the kernel code.

Output:
- `lgbm-compute` features `gpu` (umbrella), `cuda`, `wgpu`; `CudaBackend`/`WgpuBackend`.
- `lgbm` features `cuda`, `wgpu`; booster backend-selection cascade extended.
- A SUMMARY recording the four-way `cargo check` outcomes (esp. the wgpu result).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

# The seams being mirrored (read these regions before editing):
@crates/lgbm-compute/Cargo.toml
@crates/lgbm-compute/src/runtime.rs
@crates/lgbm/Cargo.toml
@crates/lgbm/src/booster.rs

# RocmBackend reference impl + the 4 core kernel dispatch bodies to mirror:
#   crates/lgbm-compute/src/lib.rs
#     - Backend trait ~486 (required methods: construct_histograms, find_best_split,
#       data_partition, subtract_histograms — all others have trait defaults)
#     - CpuBackend ~1219 (the minimal-impl template: unit struct, overrides the 4 + a few)
#     - RocmBackend struct ~2001 / impl ~2055 (the 4 core method bodies live at
#       construct_histograms ~2068, find_best_split ~2098, data_partition ~2132,
#       subtract_histograms ~2181)
#   crates/lgbm-compute/src/kernels/histogram.rs
#     - construct_hist_kernel_lds_f32 #[cube] ~772 (rocm-gated) and
#       construct_histograms_lds_f32_on launcher ~827 (rocm-gated)

# VERIFIED FACTS (confirmed during planning — do not re-research):
#   - cubecl 0.10.0 exposes cargo features `cuda` and `wgpu`; module paths
#     `cubecl::cuda` (re-export of cubecl-cuda) and `cubecl::wgpu` (cubecl-wgpu).
#   - cubecl::cuda::CudaRuntime + cubecl::cuda::CudaDevice (impl Default; also ::new(index)).
#   - cubecl::wgpu::WgpuRuntime + cubecl::wgpu::WgpuDevice (impl Default => DefaultDevice).
#   - The Runtime::client(&device) entry mirrors rocm's HipRuntime::client(&AmdDevice::new(0)).
#   - find_best_split_f64_on / data_partition_on / subtract_histograms_f64_on are already
#     runtime-generic and UNGATED. Only construct_histograms_lds_f32_on (+ its #[cube]
#     kernel) is rocm-gated and must be widened.
#   - histogram.rs ALSO contains a rocm-ONLY cubecl_hip_sys FFI (query_num_cu ~685 →
#     rowpart_target_cubes ~723). construct_histograms_lds_f32_on does NOT call these
#     (its cube_count is a local div_ceil, not a CU-count query). Leave the hip-sys
#     path on `#[cfg(feature = "rocm")]`.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add gpu/cuda/wgpu features + CudaBackend/WgpuBackend in lgbm-compute</name>
  <files>crates/lgbm-compute/Cargo.toml, crates/lgbm-compute/src/runtime.rs, crates/lgbm-compute/src/kernels/histogram.rs, crates/lgbm-compute/src/lib.rs</files>
  <action>
Wire the two new GPU backends into `lgbm-compute`, reusing the existing generic kernels.

1. `crates/lgbm-compute/Cargo.toml` `[features]`: introduce an umbrella `gpu = []`
   feature that every GPU backend enables, then add the two new backends:
   - add `gpu = []`
   - add `cuda = ["cubecl/cuda", "gpu"]`
   - add `wgpu = ["cubecl/wgpu", "gpu"]`
   - append `"gpu"` to the existing `rocm` feature list, so it becomes
     `rocm = ["cubecl/hip", "dep:cubecl-hip-sys", "gpu"]`.
   (Feature names `cuda`/`wgpu` on the `cubecl` dep are VERIFIED present in cubecl
   0.10.0 — do not change the version pin; `cubecl` is `0.10.0` from the workspace.)
   Do NOT add any new `[dependencies]` entry — `cuda`/`wgpu` are sub-features of the
   already-vetted `cubecl` dep (no package-legitimacy gate applies).

2. `crates/lgbm-compute/src/runtime.rs`: mirror the existing `rocm_client()` /
   `RocmRuntime` block (the `#[cfg(feature = "rocm")]` items at the bottom of the file)
   with cuda and wgpu equivalents:
   - `#[cfg(feature = "cuda")] pub type CudaRuntime = cubecl::cuda::CudaRuntime;`
   - `#[cfg(feature = "cuda")] #[must_use] pub fn cuda_client() -> ComputeClient<cubecl::cuda::CudaRuntime>`
     returning `cubecl::cuda::CudaRuntime::client(&cubecl::cuda::CudaDevice::new(0))`
     (mirrors rocm's `AmdDevice::new(0)`).
   - `#[cfg(feature = "wgpu")] pub type WgpuRuntime = cubecl::wgpu::WgpuRuntime;`
   - `#[cfg(feature = "wgpu")] #[must_use] pub fn wgpu_client() -> ComputeClient<cubecl::wgpu::WgpuRuntime>`
     returning `cubecl::wgpu::WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::default())`.
   Keep the same doc-comment style as `rocm_client`; note in the wgpu doc-comment that
   the f32-atomic kernels may not compile for WGSL (locked decision #3 known risk).

3. `crates/lgbm-compute/src/kernels/histogram.rs`: widen ONLY the two LDS-construct
   items so they compile under any GPU backend. Change the `#[cfg(feature = "rocm")]`
   attribute on `construct_hist_kernel_lds_f32` (~772) AND on
   `construct_histograms_lds_f32_on` (~827) from `feature = "rocm"` to
   `feature = "gpu"`. Do NOT touch the other rocm gates in this file — in particular
   leave `query_num_cu` / `rowpart_target_cubes` and any `cubecl_hip_sys` code on
   `#[cfg(feature = "rocm")]` (those are rocm-only FFI). If `cargo check --features cuda`
   later reports a missing helper that one of these two widened items genuinely calls,
   widen that specific helper to `feature = "gpu"` too — UNLESS it references
   `cubecl_hip_sys` or `rocm_client`, in which case stop and record it as a finding
   (do not work around by swapping kernels). Based on the verified body, no such helper
   is expected.

4. `crates/lgbm-compute/src/lib.rs`: add `CudaBackend` and `WgpuBackend`. They are
   unit structs (like `CpuBackend`), gated `#[cfg(feature = "cuda")]` /
   `#[cfg(feature = "wgpu")]`. Each `impl Backend` sets
   `type Runtime = runtime::CudaRuntime` / `runtime::WgpuRuntime` and overrides EXACTLY
   the four required methods, dispatching the SAME kernels RocmBackend dispatches —
   copy RocmBackend's bodies verbatim for these four:
     - `construct_histograms` → `kernels::histogram::construct_histograms_lds_f32_on(client, binned, ordered_gradients, ordered_hessians, num_bin)`
     - `find_best_split` → `kernels::split::find_best_split_f64_on(...)` (forward all args, signature identical to RocmBackend's)
     - `data_partition` → `kernels::partition::data_partition_on(...)` (forward all args)
     - `subtract_histograms` → `kernels::subtract::subtract_histograms_f64_on(client, parent, child)`
   Inherit the trait defaults for every other method (resident-pool, batched, native,
   build_fix_scan, subtract_scan) — exactly as CpuBackend leaves them undefined. Do NOT
   add ResidentBins state or override the resident-pool overrides: those drag in the
   rocm-only CU-count FFI and are out of the compile-only scope (decision #2).
   To avoid pasting the four-method body twice, define a
   `#[cfg(any(feature = "cuda", feature = "wgpu"))] macro_rules! gpu_core_backend`
   taking `($name:ident, $rt:ty)` that emits `pub struct $name;` plus the
   `impl Backend for $name { type Runtime = $rt; <four methods> }`, then invoke it with
   `#[cfg(feature = "cuda")] gpu_core_backend!(CudaBackend, runtime::CudaRuntime);` and
   `#[cfg(feature = "wgpu")] gpu_core_backend!(WgpuBackend, runtime::WgpuRuntime);`.
   (The `#[cfg(any(...))]` on the macro definition avoids an `unused_macros` warning in
   the default cpu build.) Leave `RocmBackend` and its impl entirely UNTOUCHED — the
   proven rocm path must stay byte-identical (decision #2).

Run `cargo fmt -p lgbm-compute` after editing.
  </action>
  <verify>
    <automated>cargo check -p lgbm-compute 2>&1 | tail -5 && cargo check -p lgbm-compute --features rocm 2>&1 | tail -5 && cargo check -p lgbm-compute --features cuda 2>&1 | tail -5</automated>
  </verify>
  <done>
- `cargo check -p lgbm-compute` (default cpu) passes (hard gate).
- `cargo check -p lgbm-compute --features rocm` passes — rocm path unbroken (hard gate).
- `cargo check -p lgbm-compute --features cuda` passes (expected per decision #3). If it
  fails only inside the `cubecl-cuda` dependency build because the CUDA toolkit is absent
  on this machine (not in our wiring code), record that as an environment limitation in
  the SUMMARY — do NOT alter kernels to work around it.
- `cargo check -p lgbm-compute --features wgpu` is RUN and its outcome recorded; a
  compile failure in the f32-atomic kernel monomorphization is an ACCEPTED discovered
  outcome (decision #3), documented (what/where/why), NOT worked around.
- `RocmBackend` impl is unchanged; no new `[dependencies]` added.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add lgbm cuda/wgpu features + booster backend-selection cascade, run full check matrix</name>
  <files>crates/lgbm/Cargo.toml, crates/lgbm/src/booster.rs</files>
  <action>
Surface the two new backends through the `lgbm` facade and extend the feature-switched
backend selection so exactly one backend is chosen.

1. `crates/lgbm/Cargo.toml` `[features]`: mirror the existing
   `rocm = ["lgbm-compute/rocm"]` line with
   - `cuda = ["lgbm-compute/cuda"]`
   - `wgpu = ["lgbm-compute/wgpu"]`

2. `crates/lgbm/src/booster.rs`: extend the two cfg sites that currently switch only
   between `rocm` and `not(rocm)`.
   - Imports (~lines 20-27): replace the two-arm `#[cfg(feature = "rocm")]` /
     `#[cfg(not(feature = "rocm"))]` import block with a FOUR-arm mutually-exclusive
     cascade (priority rocm > cuda > wgpu > cpu):
       - `#[cfg(feature = "rocm")]` → `use lgbm_compute::runtime::rocm_client; use lgbm_compute::RocmBackend;`
       - `#[cfg(all(feature = "cuda", not(feature = "rocm")))]` → `use lgbm_compute::runtime::cuda_client; use lgbm_compute::CudaBackend;`
       - `#[cfg(all(feature = "wgpu", not(feature = "rocm"), not(feature = "cuda")))]` → `use lgbm_compute::runtime::wgpu_client; use lgbm_compute::WgpuBackend;`
       - `#[cfg(not(any(feature = "rocm", feature = "cuda", feature = "wgpu")))]` → `use lgbm_compute::runtime::cpu_client; use lgbm_compute::CpuBackend;`
   - Instantiation (~lines 1097-1108): replace the two `let backend` / `let client`
     arms with the SAME four-arm cascade:
       - rocm: `let backend = RocmBackend::default(); let client = rocm_client();`
       - cuda (and not rocm): `let backend = CudaBackend; let client = cuda_client();`
       - wgpu (and not rocm/cuda): `let backend = WgpuBackend; let client = wgpu_client();`
       - else (cpu): `let backend = CpuBackend; let client = cpu_client();`
   Use the exact `not(...)` guards above so any combination of enabled features still
   selects exactly one backend (the `learner` / `Gbdt` loop below is already generic over
   `B: Backend`, so only these two sites change). Keep the existing explanatory comment,
   updating it to mention the cuda/wgpu arms.

Run `cargo fmt -p lgbm` after editing.
  </action>
  <verify>
    <automated>cargo check 2>&1 | tail -5 && cargo check --features rocm 2>&1 | tail -5 && cargo check --features cuda 2>&1 | tail -5 && cargo check -p lgbm --features wgpu 2>&1 | tail -8</automated>
  </verify>
  <done>
- `cargo check` (default workspace, cpu) passes (hard gate — cpu anchor unbroken).
- `cargo check --features rocm` passes — rocm path unbroken (hard gate).
- `cargo check -p lgbm --features cuda` passes (expected per decision #3; or documented
  environment limitation if cubecl-cuda's own build needs an absent CUDA toolkit).
- `cargo check -p lgbm --features wgpu` is RUN; outcome (pass or the accepted f32-atomic
  compile failure) is recorded in the SUMMARY with what/where/why — NOT worked around.
- Exactly one backend is selected for every feature combination (the `not(...)` cfg
  guards guarantee this).
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | This task is compile-gated backend wiring only — no runtime, no new external input, no new I/O surface. CudaBackend/WgpuBackend dispatch the same already-reviewed generic kernels; the f32-atomic/parity contract is unchanged. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-igc-01 | Tampering | supply chain (`cubecl/cuda`, `cubecl/wgpu` features) | accept | No new crate dependency is added — `cuda`/`wgpu` are sub-features of the already-vetted `cubecl 0.10.0` workspace dep; no install task, package-legitimacy gate does not apply. |
| T-igc-02 | Denial of Service | rocm path regression | mitigate | RocmBackend impl + runtime.rs rocm block left byte-untouched; `gpu` is added to the `rocm` feature set so the two widened LDS items stay compiled under rocm; `cargo check --features rocm` is a hard gate in both tasks. |
</threat_model>

<verification>
Full check matrix (run from repo root):
- `cargo check`                                  (default cpu — HARD GATE)
- `cargo check --features rocm`                  (lgbm rocm — HARD GATE)
- `cargo check -p lgbm-compute --features cuda`  (expected pass)
- `cargo check -p lgbm --features cuda`          (expected pass)
- `cargo check -p lgbm-compute --features wgpu`  (may fail — accepted, document)
- `cargo check -p lgbm --features wgpu`          (may fail — accepted, document)

The wgpu failure mode (if it occurs) is expected to surface inside the f32-atomic
kernel monomorphization (WGSL has no f32 atomics). Record the exact error and location
in the SUMMARY; do NOT swap kernels or add an f64/portable fallback to work around it
(locked decision #3).
</verification>

<success_criteria>
- `cargo check` (cpu) and `cargo check --features rocm` pass — existing paths unbroken.
- `cuda` builds clean for both `lgbm-compute` and `lgbm` (or a documented environment
  limitation in the cubecl-cuda dependency build, not in our wiring).
- `wgpu` build attempted; outcome documented honestly in the SUMMARY.
- CudaBackend/WgpuBackend reuse the runtime-generic kernels (no forked/portable kernels);
  RocmBackend untouched.
</success_criteria>

<output>
Create `.planning/quick/260626-igc-support-cuda-and-wgpu-by-using-rocm-kern/260626-igc-SUMMARY.md` when done.
The SUMMARY MUST include a "cargo check matrix" section recording the pass/fail result of
each of the six checks above, and — if wgpu (or cuda) failed — exactly what failed, in
which file/symbol, and why (the discovered-outcome record required by locked decision #3).
</output>
