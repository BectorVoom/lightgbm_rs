# Phase 23: Perf-Validation + Default-On Rollout (DoD) - Research

**Researched:** 2026-07-02
**Domain:** CUDA tree-learner routing flip + real-discrete-CUDA A/B benchmarking (Kaggle) + DoD evidence
**Confidence:** HIGH (all findings VERIFIED against the working tree; no external packages, no new deps)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `LGBM_CUDA_ON_DEVICE` becomes **tri-state**: unset ⇒ follow device default, `"0"` ⇒ force OFF (the off-switch fallback), `"1"` ⇒ force ON. This replaces the current binary parse (`cuda_on_device_enabled()` at `crates/lgbm-compute/src/lib.rs:1324`, today: unset/anything⇒off, `"1"`⇒on). The OnceLock-cached read pattern stays.
- **D-02:** The CUDA-only default lives at the **routing seam** as `device_type==cuda AND enabled`, NOT baked into the env parse. ROCm/CPU never see default-on because their device gate is false — this is how SC-4 (ROCm + CPU byte-unchanged) is upheld.
- **D-03:** "This is CUDA vs ROCm/HIP" is decided by the **compiled cubecl runtime binding (cargo feature)** — a `cubecl-cuda` binding ⇒ default-on eligible; `cubecl-hip` ⇒ default-off. No runtime device_type-string sniffing. A ROCm build literally cannot default-on. (Matches how `runtime::ActiveRuntime` is already selected by cargo feature.)
- **D-04:** Pass bar = on-device **median wall-clock ≤ 5% slower** than the current host-CUDA path ("within noise" — not-measurably-slower, not a strict win).
- **D-05:** Sign-stability = **3 in-session runs per config, take the median.** The default flips only if **BOTH** shapes (500k×50 AND the wide shape) pass the ≤5% bar. A regression on either shape blocks the global default flip (conservative — no shape-aware routing heuristic).
- **D-06:** The wide shape = **100k × 500**. Tree count 100 (baseline convention).
- **D-07:** Harness is a **committed, reusable script** (extends the existing `benchmark.py` / `continue_benchmark.py` family) that emits a **structured results file (MD + JSON)** — `device_launches`, wall-clock medians, and ratios — committed under the phase dir as the first-class DoD evidence artifact.
- **D-08:** Comparison set to report: on-device **vs host-CUDA** (the not-slower gate, D-04/D-05), plus context ratios vs **official LightGBM** (the ~4.46× pre-on-device bar), plus `device_launches/tree` vs the **8,570 / 100-trees** baseline (SC-2 launch-collapse confirmation).
- **D-09:** Phase 23 **always** lands the committed harness + results artifact. The **default-ON flip is a SEPARATE commit gated on the pass verdict.** If the A/B fails the ≤5%/both-shapes bar, the default stays **OFF** (opt-in via `LGBM_CUDA_ON_DEVICE=1`) and the phase is still DoD-complete with documented numbers + a follow-up note. Honors the audit-before-wire / fused-kernel-default-off precedent.
- **D-10:** At default-ON, unsupported configs (`use_quantized_grad`, or any case where `grow_tree_on_device` returns `Ok(None)`) fall back to the host-CUDA path with the existing **silent `Ok(None)`** behavior — no log noise, results still correct.
- **D-11:** Parity proof backing the flip = the CPU f64 merge gate (on-device tree bit-exact on the cubecl-cpu anchor lane) + the already-green per-phase ~1e-6 anchor gates (14–22) **PLUS a real-CUDA end-to-end ~1e-6 parity assertion in the Kaggle harness** (on-device predictions vs host-CUDA / official on the actual datasets), committed as a hard check.

### Claude's Discretion
- The exact mechanism for capturing `device_launches` on Kaggle (parsing the existing `[phase_prof:…] COUNTS: device_launches=…` line emitted under `LGBM_PHASE_PROF`, at `crates/lgbm-treelearner/src/phase_prof.rs:197`) is a planner/implementation detail.
- Results-file schema fields beyond the required metrics (D-07/D-08); Kaggle GPU-quota budgeting across the 3-run × 2-shape × 2-path matrix.

### Deferred Ideas (OUT OF SCOPE)
- **Multi-stream overlap** — a stretch spike ONLY if the launch-count reduction underdelivers on wall-clock. Not in scope unless the A/B shows launch-collapse without a wall-clock win.
- **Shape-aware routing** (default-on per-shape rather than globally) — rejected for this phase (needs a routing heuristic); revisit only if one shape consistently regresses.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-20 | A real-CUDA **Kaggle A/B harness** measures the on-device path's `device_launches/tree` (target well below 8,570/100-trees) and the lgb_rs/official wall-clock ratio at 500k×50 and a wide shape. | §5 Kaggle harness (existing `benchmark.py`/`continue_benchmark.py`/`poll_kaggle.sh` flow), §4 `device_launches` capture via `phase_prof::dump("train")` + the `COUNTS:` regex. **Landmine L-1: the on-device driver does not currently bump those counters** (Wave-0 gap). |
| ODL-21 | The on-device learner becomes the **DEFAULT** CUDA tree-learner path — contingent on ~1e-6 parity AND not-slower on the Kaggle A/B — with the host path retained as the `LGBM_CUDA_ON_DEVICE=0` off-switch fallback. | §1 tri-state parse (D-01), §2 CUDA-only default discriminant (D-02/D-03), §3 boosting-layer toggle agreement, §6 real-CUDA parity assertion (D-11). Flip is a **separate verdict-gated commit** (D-09). |
</phase_requirements>

## Summary

This is a **routing-flip + measurement** phase, not a kernel/algorithm phase. The on-device grow driver already exists, is wired behind an env gate on all three backends (`CpuBackend`, `GpuBackend<R>`), and is bit-exact to the cubecl-cpu f64 anchor. What Phase 23 does: (1) convert the **two** binary env parses to **tri-state**, (2) add a compile-time **CUDA-only device-default** at the routing seam so an unset env defaults ON only under a `cubecl-cuda` binding, (3) build a committed Kaggle A/B harness that measures `device_launches` + wall-clock on real NVIDIA hardware, and (4) flip the default **in a separate commit gated on the A/B verdict**.

The load-bearing correctness invariant is **SC-4**: CPU and ROCm must stay byte-unchanged. This is upheld structurally: the merge gate and every local `cargo test` run compile the **default `cpu` feature** (cubecl-cpu), where `cfg!(feature = "cuda")` is `false`, so the tri-state device-default resolves to OFF and the env-unset lane is byte-identical. The default-on can only fire in a `-F cuda` wheel build (the Kaggle path). ROCm (`-F rocm`) likewise has `cfg!(feature = "cuda") == false`.

The single biggest **implementation gap** (Landmine L-1): the `device_launches` counters (`BUILD_RESIDENT_CNT`/`SUBTRACT_RESIDENT_CNT`/`SCAN_RESIDENT_CNT`/`FUSED_CNT`) are bumped **only in the host-driven per-leaf path in `learner.rs`**. When on-device growth engages, the learner returns early at `learner.rs:817` *before* that code, and `grow_driver.rs` bumps none of them — so the `COUNTS:` line would report `0` (in fact it is suppressed by the `> 0` guard). SC-2 ("device_launches drops below 8,570") therefore requires **instrumenting the on-device driver's real device launches** into a phase_prof counter, or the harness has no non-zero on-device number to compare. Plan a Wave-0 task for this.

**Primary recommendation:** Land the tri-state parse + CUDA-only-default discriminant + on-device launch instrumentation + committed Kaggle A/B harness with the D-11 parity assertion as the always-landed value; gate the actual `default = ON` flip behind a separate commit contingent on a green ≤5%/both-shapes/parity verdict from the harness output.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Env tri-state parse (`LGBM_CUDA_ON_DEVICE`) | `lgbm-compute` (`cuda_on_device_enabled()`) | `lgbm-treelearner` (`cuda_on_device_env()`) — **duplicate parse, must agree** | Compute layer owns the canonical read; the learner has a *separate* read that also gates growth (L-2). |
| CUDA-only device default | `lgbm-compute` routing seam (`on_device_growth_supported()` / a `cfg!(feature="cuda")` helper) | — | D-02: default is a routing property, not an env property; D-03: keyed on the compiled cubecl binding. |
| Boosting-layer resident-score toggle | `lgbm-boosting` (`ScoreUpdater::boosting_on_cuda`, keyed off `cuda_on_device_enabled()` at construction) | `gbdt.rs` driver override (`set_boosting_on_cuda`) | Must observe the same tri-state + CUDA-only default so compute & boosting layers agree. |
| `device_launches` measurement | `lgbm-treelearner::phase_prof` (counters + `dump("train")`) | `lgbm-compute::kernels::grow_driver` (**needs new launch counting**, L-1) | Emitter exists; the on-device driver is currently uninstrumented. |
| Real-CUDA A/B + parity assertion | Kaggle harness (Python, committed under phase dir) | `lgbm/src/booster.rs` (`dump("train")` emit point) | Kaggle is the only real-discrete-CUDA path; local GPU is a spoofed APU. |

## Standard Stack

No new libraries. Phase 23 reuses existing crates and the already-vetted `cubecl 0.10.0` workspace dep.

### Core (existing, in-tree)
| Component | Location | Purpose |
|-----------|----------|---------|
| `cuda_on_device_enabled()` | `crates/lgbm-compute/src/lib.rs:1324` | Canonical env gate (OnceLock). Convert to tri-state (D-01). |
| `cuda_on_device_env()` | `crates/lgbm-treelearner/src/learner.rs:471` | **Second, duplicate** env parse (learner-local). Must also become tri-state (L-2). |
| `Backend::on_device_growth_supported()` | `lib.rs:1249` (default), `:1358` (Cpu), `:2288` (Gpu) | The routing discriminator; the CUDA-only default rides here (D-02). |
| `Backend::grow_tree_on_device()` | `lib.rs:1294` (default `Ok(None)`), `:1371` (Cpu), `:2302` (Gpu) | The on-device grow seam; already active, returns `Ok(None)` for unsupported. |
| `kernels::grow_driver::grow_tree_on_device_driver` | `crates/lgbm-compute/src/kernels/grow_driver.rs` | The actual on-device per-leaf best-first driver. **Site for launch instrumentation (L-1).** |
| `ScoreUpdater::boosting_on_cuda` | `crates/lgbm-boosting/src/score_updater.rs:51,75,167,175` | Resident-score toggle, keyed off `cuda_on_device_enabled()` at `:75`. |
| `phase_prof` COUNTS emitter | `crates/lgbm-treelearner/src/phase_prof.rs:194-199` | Emits `device_launches=…` under `LGBM_PHASE_PROF=1`. |
| `phase_prof::dump("train")` | `crates/lgbm/src/booster.rs:1480` | The emit point on the SHIPPED Python-wheel train path (harness parses this). |
| `runtime::ActiveRuntime` / `CudaRuntime` / `RocmRuntime` | `crates/lgbm-compute/src/runtime.rs:138,185,178` | cfg-gated runtime type aliases — the D-03 precedent for "runtime binding = cargo feature". |

### Cargo feature map (VERIFIED `crates/lgbm-compute/Cargo.toml`)
- `default = ["cpu"]` → `runtime::ActiveRuntime = cubecl::cpu::CpuRuntime`. **The merge gate / local test build.** `cfg!(feature="cuda")` is `false` here.
- `cuda = ["cubecl/cuda", "gpu"]` → `CudaRuntime`, `CudaBackend = GpuBackend<CudaRuntime>`. The Kaggle wheel is built `-F cuda`.
- `rocm = ["cubecl/hip", "dep:cubecl-hip-sys", "gpu"]` → `RocmRuntime`, `RocmBackend`. `cfg!(feature="cuda")` is `false`.
- `wgpu`, `gpu` (umbrella). All GPU backends share **one generic** `impl<R: cubecl::Runtime> Backend for GpuBackend<R>` (`lib.rs:2270`).

### Alternatives Considered (for the D-03 discriminant)
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cfg!(feature = "cuda")` helper (recommended, simplest) | Per-runtime sealed-trait const (`trait DeviceDefault { const ON_DEVICE_DEFAULT: bool; }` impl'd on `CudaRuntime=true`, `RocmRuntime/WgpuRuntime=false`, added as an `R:` bound) | The cfg helper is correct for **mono-feature builds** (the real case; Kaggle builds `-F cuda` only). The per-runtime const is correct even in a pathological `-F cuda,rocm` build because it keys on the concrete `R`, not a global cfg (see Pitfall P-3). Recommend the cfg helper unless the planner wants the dual-feature guarantee. |

**No install step.** `cuda`/`wgpu` are sub-features of the already-present `cubecl 0.10.0` workspace dep — no `[dependencies]` entry is added.

## Package Legitimacy Audit

**Not applicable — Phase 23 adds no external packages.** The `cuda` feature is a sub-feature of the pre-vetted `cubecl 0.10.0` workspace dependency (`crates/lgbm-compute/Cargo.toml` `[features] cuda = ["cubecl/cuda", "gpu"]`). The Kaggle harness `pip install`s only already-used tooling (`numpy`, `scipy`, `scikit-learn`, `maturin`, official `lightgbm`) — no new library is introduced into the shipped crate.

## Architecture Patterns

### System Architecture Diagram

```
                      LGBM_CUDA_ON_DEVICE (env)          compiled cubecl feature
                       unset / "0" / "1"                  (cpu | cuda | rocm | wgpu)
                              │                                    │
                              ▼                                    ▼
        ┌──────────────────────────────────┐        ┌──────────────────────────┐
        │ cuda_on_device_enabled()  [tri]   │        │ on_device_default()       │
        │ lib.rs:1324  (OnceLock, read once)│        │ = cfg!(feature="cuda")    │  ◄── D-03
        └──────────────────────────────────┘        └──────────────────────────┘
                              │  (also: learner.rs:471 cuda_on_device_env, DUP)   │
                              └───────────────────┬───────────────────────────────┘
                                                  ▼  resolve: "1"→on, "0"→off, unset→default
                    ┌───────────────────────────────────────────────┐
                    │ Backend::on_device_growth_supported()          │  ◄── D-02 routing seam
                    │  Cpu(lib.rs:1358) / Gpu<R>(lib.rs:2288)        │
                    └───────────────────────────────────────────────┘
                              │  ANDed at learner.rs:539 / :3844
                              ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ SerialTreeLearner.on_device_eligible                          │
        │   = supported() && env() && !(cat && quantized)  (gate:481)   │
        └──────────────────────────────────────────────────────────────┘
               │ eligible                                    │ not eligible / Ok(None)
               ▼                                             ▼
    grow_tree_on_device (lib.rs:1371/2302)          host per-leaf path (learner.rs)
     → grow_driver (returns early @817)              build/subtract/scan/partition
       [L-1: bump device-launch counter HERE]        [bumps *_RESIDENT_CNT counters]
               └──────────────────────┬──────────────────────┘
                                       ▼
                        phase_prof::dump("train")  (booster.rs:1480)
                        emits "[phase_prof:train] COUNTS: device_launches=N …"
                                       ▼
                        Kaggle harness parses stderr → MD + JSON evidence
```

### Pattern 1: Tri-state env parse preserving read-once OnceLock (D-01)
**What:** Convert the binary `bool` OnceLock to a tri-state that distinguishes unset / `"0"` / `"1"`, then resolve against the device default at the routing seam.
**When to use:** Both `cuda_on_device_enabled()` (`lib.rs:1324`) and `cuda_on_device_env()` (`learner.rs:471`).
**Example:**
```rust
// crates/lgbm-compute/src/lib.rs — replace the binary parse at :1324
// Tri-state: None = unset (follow device default), Some(false) = "0" force off,
// Some(true) = "1" force on. Read ONCE, OnceLock-cached (mirrors split_2lane_enabled).
fn cuda_on_device_override() -> Option<bool> {
    use std::sync::OnceLock;
    static E: OnceLock<Option<bool>> = OnceLock::new();
    *E.get_or_init(|| match std::env::var("LGBM_CUDA_ON_DEVICE").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None, // unset / empty / malformed ⇒ follow device default (D-01)
    })
}

// D-02/D-03: the CUDA-only device default — compile-time, keyed on the cubecl binding.
#[must_use]
fn on_device_default() -> bool {
    // ROCm/WGPU/CPU builds have cfg!(feature="cuda")==false ⇒ default OFF (SC-4).
    // NOTE: for a pathological -F cuda,rocm build see Pitfall P-3.
    cfg!(feature = "cuda")
}

// The resolved gate the routing seam calls (replaces the old bool fn).
#[must_use]
pub fn cuda_on_device_enabled() -> bool {
    cuda_on_device_override().unwrap_or_else(on_device_default)
}
```
Because `cuda_on_device_enabled()` keeps the same name + `-> bool` signature, `ScoreUpdater::new` (`score_updater.rs:75`) and `on_device_growth_supported()` (`lib.rs:1358`,`:2288`) pick up the new semantics with no call-site change — **except** the learner's own duplicate (L-2, below).

### Pattern 2: The learner's duplicate parse must be reconciled (L-2)
**What:** `learner.rs:471 cuda_on_device_env()` is a *second, independent* binary parse (`matches!(var, Ok("1"))`), ANDed at `:539` and `:3844` as `backend.on_device_growth_supported() && cuda_on_device_env()`. Since `on_device_growth_supported()` already returns `cuda_on_device_enabled()`, this is a **double gate on two different parses**.
**When to use:** Must be updated in lockstep with Pattern 1, or the two layers disagree (e.g. `"0"` off-switch works in compute but the learner still treats unset-under-cuda as off).
**Recommendation:** Delete `cuda_on_device_env()` and make the learner's `base` gate be `backend.on_device_growth_supported()` alone (it already encodes the resolved tri-state + default), OR make `cuda_on_device_env()` call the compute-layer resolver. Simplest: drop the redundant AND so there is a **single source of truth** (`lgbm_compute::cuda_on_device_enabled()`).

### Pattern 3: Boosting-layer toggle agreement (D-02)
**What:** `ScoreUpdater::new` sets `boosting_on_cuda: lgbm_compute::cuda_on_device_enabled()` (`score_updater.rs:75`). Since the resolver name/signature is unchanged, the boosting layer automatically inherits tri-state + CUDA-only default. The GBDT driver override `set_boosting_on_cuda` (`gbdt.rs:1485`) and the resident-score branch (`gbdt.rs:1006`) stay as-is.
**Verification hook:** Existing tests `resident_score_ab.rs:187` and `score_updater_parity.rs:108` assert "unset ⇒ boosting_on_cuda default OFF". Under the cpu-feature test build `on_device_default()==false`, so these **still pass unchanged** — the merge gate is byte-identical (SC-4).

### Pattern 4: On-device device-launch instrumentation (L-1, SC-2)
**What:** Add launch counting inside `grow_tree_on_device_driver` so an on-device train produces a non-zero, much-smaller `device_launches` for the harness to compare against 8,570/100-trees.
**Where:** Bump a phase_prof counter (reuse `FUSED_CNT`/`BUILD_RESIDENT_CNT` semantics or add a dedicated `ON_DEVICE_LAUNCH_CNT` field surfaced in the `COUNTS:` line) at each real `client.execute`/`launch_unchecked` in the driver's per-leaf loop.
**Why it's mandatory:** Without it the on-device `COUNTS:` line is suppressed (`phase_prof.rs:194` guards on `sum > 0`), so ODL-20/SC-2 cannot be measured. This is a **Wave-0 gap**.

### Anti-Patterns to Avoid
- **Baking the CUDA-only default into the env parse.** D-02 forbids this — the default is a routing property. Keep `cuda_on_device_override()` device-agnostic; resolve against `on_device_default()` at the seam.
- **Runtime `device_type` string sniffing.** D-03 forbids it. Use the compile-time cargo feature / runtime-type discriminant only.
- **Flipping the crate `default` feature and the `LGBM_CUDA_ON_DEVICE` semantics in the same commit as the harness.** D-09 requires the flip be a *separate, verdict-gated* commit.
- **Comparing two nondeterministic GPU f32 paths at 1e-6** (def-f8u-01): the parity assertion (D-11) must anchor on-device predictions to host-CUDA / official, and structurally to the cpu-f64 merge gate — not on-device-f32 vs host-f32 directly without the anchor.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reading the on-device toggle | A new env var / new parse site | Extend the existing `cuda_on_device_enabled()` resolver | There are already **two** parses (L-2); adding a third worsens the drift. Collapse to one source of truth. |
| Measuring device launches | New instrumentation harness | The existing `phase_prof` COUNTS emitter + `dump("train")` at `booster.rs:1480` | Already wired into the shipped Python train path; only the on-device driver counter is missing (L-1). |
| Kaggle push/poll/build | A fresh CI script | Extend `benchmark.py`/`continue_benchmark.py` + `poll_kaggle.sh` | The clone→maturin `-F cuda`→install→official-lightgbm→`make_classification` flow already works and is authenticated as `boomvector`. |
| Runtime selection by device | `if device_type == "cuda"` at runtime | The cfg-gated `ActiveRuntime`/`CudaRuntime` type aliases (`runtime.rs`) | Established precedent (D-03); compile-time, no sniffing. |

**Key insight:** Almost everything Phase 23 needs already exists and is proven. The net-new code is: tri-state resolver (~15 lines), a `cfg!(feature="cuda")` default helper (~3 lines), on-device launch counting (L-1), and the committed A/B harness (Python). The risk is in **coherence** (three toggle sites must agree) and **evidence discipline** (separate verdict-gated flip commit), not in novel algorithms.

## Runtime State Inventory

> This phase changes routing/config semantics (env parse) — reviewed for hidden runtime state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — the flip touches no datastore, model format, or serialized field. Model output is unchanged (on-device tree is structure-bit-exact to the cpu anchor). | None |
| Live service config | None on the shipped library. The **Kaggle kernel** `boomvector/lgb-rs-cuda-bench` is live service config that lives on Kaggle, not in git (referenced by `poll_kaggle.sh`). The committed harness script is the git-tracked source; the Kaggle-side kernel is (re)pushed from it. | Harness push step re-creates the kernel from the committed script. |
| OS-registered state | None. | None |
| Secrets/env vars | `LGBM_CUDA_ON_DEVICE` **semantics change** (binary→tri-state) — same var name, new meaning of `"0"` (now an explicit force-off) and unset (now device-default). Kaggle CLI auth (`boomvector`) is an existing credential, unchanged. `LGBM_PHASE_PROF=1` is read by the harness to surface COUNTS. | Document the new tri-state contract in code + results MD; no key rename. |
| Build artifacts | The Kaggle wheel is built `-F cuda` per run (`maturin build --release -F cuda`); `continue_benchmark.py` already `rm -rf target/wheels/` before rebuild. If the crate `default` feature is later flipped, confirm the local cpu merge-gate build is unaffected (it is: `cfg!(feature="cuda")==false`). | None beyond existing rebuild step. |

**The canonical question — after every repo file is updated, what still has the old string cached?** The `OnceLock` caches read the env *once per process*; the merge-gate tests that set/unset `LGBM_CUDA_ON_DEVICE` in-process already account for this (`resident_score_ab.rs:168` skips the default-check when `=1`). No cross-process cache.

## Common Pitfalls

### Pitfall P-1: OnceLock read-once determinism vs the merge gate
**What goes wrong:** The tri-state resolver is `OnceLock`-cached; the first read in a process freezes the value. Tests that toggle `LGBM_CUDA_ON_DEVICE` per-arm cannot flip it mid-process.
**Why it happens:** `cuda_on_device_enabled()`/`cuda_on_device_env()`/`phase_prof::enabled()` all `get_or_init` once.
**How to avoid:** Keep the existing test seams (`Gbdt::set_boosting_on_cuda`, `SerialTreeLearner` overrides) for in-process A/B; do NOT rely on env re-read. The merge gate runs env-unset under the cpu feature → resolver returns `on_device_default()==false` → byte-unchanged. **Warning sign:** a test that expects flipping env between two `.fit()` calls in one process to change routing.

### Pitfall P-2: The `device_launches` COUNTS line is emitted from the HOST path only
**What goes wrong:** On-device train emits `device_launches=0` (line suppressed), so the harness sees nothing and SC-2 is unmeasurable.
**Why it happens:** Counters bump in `learner.rs` (`:1873,:2060,:2281,:2725,:2742`); the on-device path returns early at `learner.rs:817` and `grow_driver.rs` bumps none. Confirmed by grep: no `phase_prof`/`CNT`/`bump` in `grow_driver.rs`.
**How to avoid:** L-1 Wave-0 task — instrument the driver's launches. **Warning sign:** an on-device `[phase_prof:train]` block with no `COUNTS:` line.

### Pitfall P-3: The generic `GpuBackend<R>` impl and a dual-feature build
**What goes wrong:** `impl<R: cubecl::Runtime> Backend for GpuBackend<R>` is ONE impl for cuda/rocm/wgpu. A `cfg!(feature = "cuda")` default helper is a **global** compile-time flag: in a `-F cuda,rocm` build, `cfg!(feature="cuda")==true` for the `RocmBackend<HipRuntime>` instance too, so ROCm could default-on — violating D-03/SC-4.
**Why it happens:** cfg cannot vary per-`R` inside a generic impl.
**How to avoid:** For the real mono-feature Kaggle/ROCm builds this cannot happen (a `-F rocm` build has `cfg!(feature="cuda")==false`). If the dual-feature guarantee is wanted, use the per-runtime sealed-trait const (Alternatives table) keyed on the concrete `R`. **Recommend:** cfg helper + a code comment asserting mono-feature builds; upgrade to the const only if the planner wants belt-and-suspenders. **Warning sign:** anyone building `-F cuda,rocm`.

### Pitfall P-4: Flipping the default without the verdict (D-09)
**What goes wrong:** Auto-engaging the default before the A/B proves ≤5%/both-shapes/parity — the exact anti-pattern the "audit-before-wire / fused-kernel default-off" precedent guards against.
**How to avoid:** Land harness + evidence + instrumentation in the main phase commits; make `on_device_default()` return `true` (or flip the crate `default` feature per the chosen mechanism) in a **separate commit** whose message cites the results artifact's PASS verdict. If FAIL, that commit is simply not made — phase is still DoD-complete (D-09).

### Pitfall P-5: `Ok(None)` silent host-fallback must survive default-on (D-10)
**What goes wrong:** At default-on, `use_quantized_grad` / categorical+quantized / any `grow_tree_on_device → Ok(None)` must silently run the host-CUDA path with correct results and no log noise.
**How to avoid:** The fallback is already structural: `on_device_eligible_gate` (`learner.rs:481`) ANDs `!(cat && quantized)`, and the driver returns `Ok(None)` for unsupported configs → learner falls through at `:826`. Verify no new `eprintln!` fires on the default-on unsupported path (the `cat_quant_fallback_logged` guard already logs at most once). **Warning sign:** log spam on a quantized default-on run.

## Code Examples

### The harness `device_launches` parse (D-08, Claude's Discretion)
```python
import re
# stderr line (phase_prof.rs:197):
# [phase_prof:train] COUNTS: device_launches=8570 (build_resident=... subtract_resident=...
#   scan_resident=... fused=...) | scan_roundtrips(syncs)=...
COUNTS = re.compile(
    r"\[phase_prof:\w+\] COUNTS: device_launches=(?P<launches>\d+) "
    r"\(build_resident=(?P<build>\d+) subtract_resident=(?P<sub>\d+) "
    r"scan_resident=(?P<scan>\d+) fused=(?P<fused>\d+)\)"
)
def parse_launches(stderr: str):
    m = COUNTS.search(stderr)
    return None if not m else {k: int(v) for k, v in m.groupdict().items()}
# Run each arm with env LGBM_PHASE_PROF=1 (+ LGBM_CUDA_ON_DEVICE=1 for the on-device arm),
# capture stderr, parse, divide launches by n_estimators=100 for device_launches/tree.
```

### Kaggle A/B arm skeleton (extends `continue_benchmark.py`)
```python
# Per D-05: 3 runs × 2 shapes {(500000,50),(100000,500)} × 2 paths {host-cuda, on-device}.
# host-cuda arm:  env LGBM_CUDA_ON_DEVICE=0  (explicit off-switch, new tri-state)
# on-device arm:  env LGBM_CUDA_ON_DEVICE=1
# Take the MEDIAN of 3 wall-clocks per (shape,path). PASS iff for BOTH shapes:
#   median(on_device) <= 1.05 * median(host_cuda)   (D-04 ≤5% "within noise")
# Emit: results.json (raw runs + medians + ratios + launches) and results.md (verdict table).
```

### Real-CUDA end-to-end parity assertion (D-11)
```python
# After training BOTH lgb_rs arms on the SAME (X, y), assert on-device predictions match
# host-CUDA (and report vs official) within ~1e-6 absolute on the actual dataset.
p_on   = on_device_model.predict(X, raw_score=True)
p_host = host_cuda_model.predict(X, raw_score=True)
max_abs = float(np.max(np.abs(p_on - p_host)))
assert max_abs <= 1e-6, f"D-11 real-CUDA parity FAILED: max_abs={max_abs}"
# Context (not a hard gate at the same tol — f32 reference): report np.max(|p_on - p_official|).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `LGBM_CUDA_ON_DEVICE` binary (`"1"`⇒on, else off) | Tri-state (unset⇒device-default, `"0"`⇒off, `"1"`⇒on) | This phase (D-01) | `"0"` becomes a first-class off-switch; unset can default-on under cuda. |
| On-device default OFF everywhere | Default ON under a `cubecl-cuda` binding (verdict-gated) | This phase (D-02/D-03/D-09) | CUDA users get the on-device path by default; ROCm/CPU unchanged. |
| Two independent env parses (compute + learner) | Single source of truth (L-2 reconciliation) | This phase | Removes drift risk between layers. |

**Deprecated/outdated:** The binary `cuda_on_device_env()` in `learner.rs:471` — fold into the compute-layer resolver.

## Validation Architecture

> `workflow.nyquist_validation: true` — this section drives the VALIDATION.md.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + `oracle-harness` integration tests |
| Config file | Workspace `Cargo.toml`; tests in `crates/oracle-harness/tests/*.rs` and `crates/lgbm-compute/tests/*.rs` |
| Quick run command | `cargo test -p lgbm-compute cuda_on_device` (unit: tri-state resolver + default helper) |
| Full suite command | `cargo test --workspace` (the CPU f64 merge gate — hard gate, must be green, SC-4) |
| Real-CUDA gate | Kaggle harness (Python) — **not runnable locally** (local GPU is a spoofed 8-CU APU) |

### Phase Requirements → Test Map
| Req / SC | Behavior | Test Type | Automated Command | Locally testable? |
|----------|----------|-----------|-------------------|-------------------|
| D-01 tri-state | unset→default, `"0"`→off, `"1"`→on; read-once OnceLock | unit | `cargo test -p lgbm-compute cuda_on_device_override` | ✅ (env-set within test process; mind P-1) |
| D-02/D-03 CUDA-only default | cpu/rocm feature ⇒ `on_device_default()==false`; cuda ⇒ true | unit + cfg | `cargo test -p lgbm-compute on_device_default` (cpu build asserts false) | ✅ cpu/rocm arm locally; ⚠️ cuda arm needs `-F cuda` build (compiles locally even without a GPU, but growth needs the device) |
| SC-4 byte-unchanged | env-unset merge gate identical; ROCm/CPU unchanged | integration | `cargo test --workspace` (incl. `learner_parity`, `score_updater_parity`, `resident_score_ab`) | ✅ |
| L-2 single-source | learner + compute + boosting agree on the toggle | unit/integration | `cargo test -p oracle-harness resident_score_ab score_updater` | ✅ |
| D-11 parity (structural) | on-device tree bit-exact to cpu-f64 anchor | integration | `cargo test -p oracle-harness learner_parity` (with `LGBM_CUDA_ON_DEVICE=1` guard, `learner_parity.rs:1594`) | ✅ (cpu anchor lane) |
| L-1 launch capture | on-device driver emits non-zero `device_launches` | unit/integration | new test asserting the driver bumps the launch counter | ✅ (cpu build exercises the driver) |
| ODL-20 / SC-1 / SC-2 | real-CUDA `device_launches/tree` < 8,570/100 + wall-clock ratios at both shapes | e2e (manual, Kaggle) | Kaggle kernel `boomvector/lgb-rs-cuda-bench` → `poll_kaggle.sh` | ❌ Kaggle only |
| ODL-21 / SC-3 | default flip contingent on ≤5%/both-shapes + parity | e2e (manual, Kaggle) + separate commit (D-09) | harness PASS verdict in results.md | ❌ Kaggle only |
| D-11 parity (real-CUDA) | on-device preds vs host-CUDA ≤1e-6 on actual data | e2e (manual, Kaggle) | assertion embedded in harness | ❌ Kaggle only |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute cuda_on_device` (fast resolver/default checks)
- **Per wave merge:** `cargo test --workspace` (full CPU f64 merge gate — SC-4)
- **Phase gate:** Full suite green locally + Kaggle A/B results artifact committed (harness landed regardless of verdict, D-09); default-flip commit only if PASS.

### Wave 0 Gaps
- [ ] **L-1:** Instrument `grow_driver.rs` device launches into a phase_prof counter surfaced in the `COUNTS:` line — **blocks SC-2 measurement.** Add `crates/lgbm-compute/.../grow_driver.rs` launch-count + a unit test.
- [ ] **L-2:** Reconcile `learner.rs:471 cuda_on_device_env()` with the compute resolver (single source of truth) — add/adjust a test that all three toggle sites agree.
- [ ] Unit tests for the tri-state resolver (`unset`/`"0"`/`"1"`/malformed) and `on_device_default()` per-feature — likely a **new test file** under `crates/lgbm-compute/tests/` or `#[cfg(test)]` in `lib.rs`.
- [ ] Committed Kaggle harness script + `results.{md,json}` schema (no local test; validated by a dry-run parse over a captured stderr fixture).

*(Existing infrastructure already covers the merge gate, learner/score parity, and the `LGBM_CUDA_ON_DEVICE=1`-guarded on-device parity test.)*

## Security Domain

> `security_enforcement: true`, ASVS level 1. This phase is compute/config + a CI benchmark script; attack surface is small.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Kaggle CLI uses existing `boomvector` credential (out of band). |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `LGBM_CUDA_ON_DEVICE` parse matches on **exact** strings (`"1"`/`"0"`), all else ⇒ default. No path/format/injection surface (documented as T-14-02). Keep this pattern for tri-state — do NOT parse into a path or command. |
| V6 Cryptography | no | — |

### Known Threat Patterns for {env-parse + CI harness}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Env var used to influence a code path | Tampering | Exact-match parse, closed enum (`Some(true)/Some(false)/None`); no eval/exec of the value. |
| Kaggle harness clones + builds from a public repo | Tampering / Supply chain | Harness pins the repo (`BectorVoom/lightgbm_rs`) and builds from source with `maturin`; no third-party binary wheel of lgb_rs. Official `lightgbm` installed from source (`--no-binary`). No new package added to the shipped crate. |
| Secrets in the committed harness | Info disclosure | The harness must NOT embed Kaggle tokens; auth stays in the CLI's out-of-band config. Verify no credential literals in the committed script. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 8,570/100-trees `device_launches` baseline was measured on the **host-CUDA** per-leaf path (the counters that bump live there). | §4, SC-2 | If the baseline was captured differently, the SC-2 comparison denominator changes. Confirmed structurally: only `learner.rs` bumps those counters; sourced from `Skill("spike-findings-lightgbm_rs")` / CONTEXT D-08. |
| A2 | `grow_driver.rs` bumps **none** of the launch counters (verified by grep: no `phase_prof`/`CNT`/`bump`). | §4, P-2, L-1 | If some launch counting exists elsewhere in the on-device path, L-1 shrinks. Grep was over `grow_driver.rs` only; planner should re-confirm across the driver's kernel callees. |
| A3 | The wide shape 100k×500 (float64 ≈ 400 MB) fits Kaggle T4 (16 GB) memory. | §5, D-06 | If OOM, reduce rows; D-06 already chose fewer rows "to fit T4 memory". |
| A4 | A `-F cuda` wheel builds and links on Kaggle without local-GPU-only build steps (existing `benchmark.py` already does `maturin build -F cuda`). | §5 | Existing scripts prove the build path; toolkit drift on Kaggle could break it. |
| A5 | The official-vs-lgb_rs ~4.46× pre-on-device bar is the correct context ratio to report. | §D-08 | Context-only (not a gate); sourced from CONTEXT/skill. |

## Open Questions

1. **Which D-03 mechanism — `cfg!(feature="cuda")` helper vs per-runtime const?**
   - What we know: mono-feature builds (Kaggle `-F cuda`, ROCm `-F rocm`) are correct with the simple cfg helper.
   - What's unclear: whether the project ever builds `-F cuda,rocm` (would mis-default ROCm on, P-3).
   - Recommendation: cfg helper + a mono-feature comment; upgrade to the per-runtime sealed const only if a dual-feature build is a real target.

2. **Where does the launch counter for L-1 surface — reuse an existing `*_CNT` field or add `ON_DEVICE_LAUNCH_CNT`?**
   - What we know: the harness parses `device_launches=N`; reusing the sum keeps the regex stable.
   - Recommendation: add a dedicated counter but include it in the existing `device_launches=` total so the harness regex is unchanged, and add an `on_device=` sub-field for clarity.

3. **Kaggle GPU-quota budget for the 3×2×2 = 12-run matrix (D-05) plus warmups.**
   - What we know: each arm is a full 100-tree train at up to 500k×50 / 100k×500; Kaggle GPU quota is weekly-limited.
   - Recommendation: one kernel run does all 12 arms sequentially (single build), discards a warmup tree-batch per arm; poll via `poll_kaggle.sh`. Budget one full kernel run + one retry.

## Environment Availability

| Dependency | Required By | Available (local) | Version | Fallback |
|------------|------------|-------------------|---------|----------|
| Rust + cargo | All builds/tests | ✓ | workspace toolchain | — |
| cubecl (cpu) | Merge gate | ✓ | 0.10.0 | — |
| cubecl (cuda) | Kaggle wheel | build-only locally (no real device) | 0.10.0 | Kaggle T4 |
| Real discrete NVIDIA GPU | SC-1/SC-2/SC-3 real numbers | ✗ (local GPU = spoofed 8-CU APU) | — | **Kaggle `boomvector` (only path)** |
| Kaggle CLI (authed) | Push/poll A/B | ✓ (`boomvector`) | — | — |
| Python (numpy/scipy/sklearn/maturin/lightgbm) | Harness | provided by Kaggle image / `pip install` | — | uv `.venv` at repo root for local wheel builds |

**Missing dependencies with no fallback:** none — the real-CUDA numbers require Kaggle, which is available.
**Missing dependencies with fallback:** local discrete NVIDIA GPU → Kaggle T4.

## Sources

### Primary (HIGH confidence — VERIFIED against working tree)
- `crates/lgbm-compute/src/lib.rs:1230-1500, 2240-2420` — `cuda_on_device_enabled`, `on_device_growth_supported`, `grow_tree_on_device`, generic `GpuBackend<R>` impl.
- `crates/lgbm-compute/src/runtime.rs` — `ActiveRuntime`/`CudaRuntime`/`RocmRuntime` cfg-gated aliases (D-03 precedent).
- `crates/lgbm-compute/Cargo.toml` — feature map (`cpu`/`gpu`/`cuda`/`rocm`/`wgpu`).
- `crates/lgbm-treelearner/src/learner.rs:463-544, 805-826, 3835-3860` — `cuda_on_device_env`, eligibility gate, on-device fork, `refresh_on_device_eligibility`.
- `crates/lgbm-treelearner/src/phase_prof.rs:79-231` — counters, `enabled()`, `dump()`, COUNTS emitter.
- `crates/lgbm-boosting/src/score_updater.rs:44-176` and `gbdt.rs:1006,1485` — `boosting_on_cuda` toggle.
- `crates/lgbm/src/booster.rs:1480` — `dump("train")` on the shipped Python path.
- `crates/lgbm-compute/src/kernels/grow_driver.rs` — on-device driver (grep-confirmed no launch counting → L-1).
- `crates/oracle-harness/tests/{resident_score_ab,score_updater_parity,learner_parity}.rs` — existing merge-gate/default-off assertions.
- `benchmark.py`, `continue_benchmark.py`, `benchmark_cpu_gpu.py`, `poll_kaggle.sh` (repo root) — existing Kaggle A/B flow.

### Secondary (MEDIUM confidence)
- `Skill("spike-findings-lightgbm_rs")` / CONTEXT.md — 8,570/100-trees baseline, ~4.46× official bar, spike-048 attribution.

## Metadata

**Confidence breakdown:**
- Standard stack / seams: HIGH — all read directly from the current tree.
- Routing/tri-state mechanics: HIGH — exact call sites and signatures verified.
- `device_launches` capture: HIGH on the emitter/regex; the L-1 gap is a VERIFIED absence (grep).
- Real-CUDA numbers: N/A locally — Kaggle-only by design.

**Research date:** 2026-07-02
**Valid until:** 2026-08-01 (stable — internal code seams; re-verify line numbers if the crates change before planning)

## RESEARCH COMPLETE
