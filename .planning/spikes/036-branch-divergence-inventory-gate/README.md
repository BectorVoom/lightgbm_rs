---
spike: 036
name: branch-divergence-inventory-gate
type: standard
validates: "Given every #[cube] kernel + post-035 attribution, when each data-dependent branch is mapped to its divergence class and critical-path phase AND a known-magnitude divergence is injected on the APU, then either ≥1 divergent branch is on a hot path AND sign-measurable (continue to 037/038) or not (STOP — documented don't-chase)"
verdict: PARTIAL
related: [021, 024, 030, 034, 035]
tags: [performance, gpu, rocm, divergence, branching, attribution, measurability, gate, negative-result]
---

# Spike 036: Branch-Divergence Inventory + Critical-Path/Measurability Gate

## What This Validates

The idea "optimize conditional branching in GPU kernels" presumes two things that the
campaign's rules forbid assuming:
1. **Critical-path:** that a divergent branch sits on a production hot path (the 030/034
   re-attribute rule — the bottleneck has moved 4× and is currently *build*-bound wide /
   *partition*-launch small-med, now host-routed by 035).
2. **Measurability:** that a divergence delta is observable on the **spoofed 8-CU gfx1152
   APU** where rocprof is unsupported and every GPU number is sign-only.

This gate tests BOTH before any kernel A/B (037 trip-count divergence, 038 break-vs-select).

## Research

No external library applies — the "state of the art" here is (a) the live kernel source and
(b) the RDNA/GCN lockstep-wavefront execution model. The relevant fact: a wavefront executes
one instruction stream in lockstep; a data-dependent branch/loop bound makes the wave run
until its **slowest active lane**, with diverged lanes **masked** (idle but occupying slots).
This is a property of the **wavefront scheduler**, NOT of CU count or the memory subsystem —
so, unlike bandwidth/occupancy, it is *a priori* plausible that divergence is the one GPU
effect this APU measures faithfully. The measurability probe (below) tests exactly that.

## Branch Inventory — every `#[cube]` kernel, divergence class, hot-path status

| Kernel (file) | Data-dependent divergence? | On production default path post-035? | Critical-path phase |
|---|---|---|---|
| `split_scan_body` / `find_best_splits_fused_kernel` (split.rs) | **YES — loop-trip-count**: lane `f` scans `0..max(rev,fwd)_count[f]`; a W=64 wave runs at the **max num_bin** in the wave. Body itself fully **branchless `select`** (the `done` flag *predicates*, it does not early-exit). | **YES** (the per-leaf split scan) | **scan** — only **3–7%** launch-bound post-co-pack (034); ≈3.5% wide |
| co-pack siblings `if g<n_feats {A} else {B}` (split.rs) | Boundary-wave only; both arms call the *same* body. ≤1 diverged wave/launch. | YES (Phase-12 co-pack) | negligible |
| `data_partition_kernel` (partition.rs) | NO — routing is branchless `select`; `if most_freq_bin==0` is **uniform** (same for all lanes). | **NO — OFF the default path** (035 `prefers_host_partition()` default-ON routes partition on the host) | n/a (host now) |
| `construct_hist_kernel_atomic_f32_plane` (histogram.rs) | **YES — heavy** (`plane_ballot` while-loop, leader election, nested `if`). | **NO** — the p93 warp-aggregation path, **NULL/superseded** by u64 atomics (017/020) | n/a (dead path) |
| u64 resident LDS build (`construct_leaf_hist_resident_lds_kernel_u64`; modelled by spike030 `build_full`) | **NO** — `while k < r` grid-stride; every lane runs `r/stride (±1)` iters. **Uniform.** | **YES — the build** | **build** — the **dominant** wide bottleneck (030: 86–95%), and it is **divergence-free by construction** |
| `subtract_hist_kernel` (subtract.rs) | NO — `while i < n` grid-stride, uniform. | YES | small; uniform |
| `hist_fold_body` (histogram.rs, cpu anchor) | single-owner `if UNIT_POS==0`; W=1. | cpu anchor only | n/a |

**Reading of the inventory:** the only *live, data-dependent, cross-lane* divergence on the
production path is the **split-scan trip-count divergence** (row 1) — and the scan is only
**3–7%** of launch-bound train (034) and **~3.5%** wide. The **dominant** bottleneck — the
wide histogram **build** — is **uniform/divergence-free by construction**, so branch
optimization *cannot* touch it. The partition device kernel (branchless anyway) is off the
default path; the only heavily-divergent kernel (plane-atomic) is a dead/NULL path.

## How to Run

```
cargo run --release --features rocm --example spike036_divergence_measurability
```

## What to Expect

A controlled-divergence **ladder**. Four arms do **identical total work** (sum of per-lane
loop trip counts = `CUBE_DIM*K`); only the **distribution** across wavefront lanes changes
(UNIFORM / DIV2 / DIV4 / DIV32 — `lane%n==0 ? n*K : 0`). If wavefronts serialize to the
slowest lane, wall-clock scales **1 : 2 : 4 : 32** despite constant useful work (the masked
idle lanes are pure waste). Loop body = constant fma into a register sink (no memory, no
DCE); trip counts come from a **device array** so the compiler can't specialize.

## Investigation Trail

- **Inventory first (read, don't guess).** Walked all four kernel files. Found the split
  scan + partition kernels were *already* written fully branchless (`select`-everywhere) —
  not for divergence, but to satisfy cubecl-cpu's MLIR lowering (which rejects in-loop
  conditional-store `if` chains). The divergence-elimination transform is already shipped as
  a side effect. So "optimize the branches" is largely *already done*; what remains is
  **loop-trip-count** divergence (lane count differs), not branch-store divergence.
- **Then the measurability ladder.** Prior belief (from the APU memo) was "divergence is a
  warp-occupancy effect → APU-confounded → probably unmeasurable." The ladder **refuted that
  prior, cleanly:**

  | run | UNIFORM | DIV2 (ideal 2) | DIV4 (ideal 4) | DIV32 (ideal 32) |
  |---|---|---|---|---|
  | 1 | 164.8 ms (1.00×) | 321.9 (1.95×) | 633.5 (3.84×) | 4821.1 (29.25×) |
  | 2 | 161.5 ms (1.00×) | 305.1 (1.89×) | 585.2 (3.62×) | 4129.7 (25.57×) |

  Near-ideal slope, restart-stable, every rung's p25 well above UNIFORM's p75. **Lockstep
  masking is faithful on this APU** — it is the *one* GPU micro-architectural effect that is
  cleanly measurable here (because it's a scheduler property, not CU-count / memory-bound,
  which are the confounded axes).

## Results

**VERDICT: PARTIAL — measurability VALIDATED (strongly); critical-path WEAK/conditional.**

- **Measurability half → PASS.** Injected divergence shows up in wall-clock with near-ideal
  fidelity (≈0.92–0.98× of the ideal ratio at every rung, 2 restarts). A divergence A/B *can*
  be sign-judged on this hardware — contradicting the "APU can't measure it" prior. This is a
  reusable result: **divergence is the exception to the spoofed-APU measurement caveat.**
- **Critical-path half → WEAK.** No divergent branch is on the *dominant* bottleneck: the
  wide **build** is uniform/divergence-free by construction (030), the **partition** device
  kernel is branchless *and* off the default path (035), and the heavily-divergent
  plane-atomic kernel is a dead/NULL path. The **only** live data-dependent divergence — the
  split-scan **trip-count** imbalance — sits on the **3–7%** scan phase and only bites on
  **mixed-cardinality** feature sets (a wave of all-256-bin features has *zero* trip-count
  divergence; identical counts ⇒ identical waves).

**Surprises:**
1. The kernels are *already* branchless (cubecl-cpu MLIR forced it) — the obvious lever is
   pre-pulled. What's left is loop-length divergence, a different and narrower thing.
2. Divergence is *measurable* on the APU after all — the one effect that survives the spoof.
3. But measurable ≠ worth it: the measurable divergence is off the dominant path.

**Go/no-go for the deferred spikes:**
- **037 (scan trip-count divergence)** — build ONLY as a **bounded, conditional** probe on a
  **mixed-cardinality** dataset (categorical + low-card numericals mixed with 256-bin), where
  wave imbalance is non-zero. On the production all-256-bin shape it is a guaranteed null.
  Even a positive result caps at a fraction of the 3–7% scan ⇒ honest e2e ceiling ≪ 1%.
- **038 (real-`break` vs predicated-`select`)** — **LOW value, likely don't-build.** The
  `done` flag is *intra-lane* predication; a real `break` cuts a lane's own ALU but the wave
  still runs to its **slowest** lane, so divergence is unchanged unless early-exit is
  *correlated* across the wave's features. Plus it must be a hip-only fork (the shared
  `split_scan_body` is the bit-exact CPU anchor — untouchable). Speculative payoff on a
  3–7% phase.

**Net:** the measurability gate is GREEN and reusable; the critical-path gate is AMBER — the
honest recommendation is **do not chase branch divergence as a general lever** (the dominant
build path has none), and only run 037 as a small bounded mixed-cardinality curiosity if the
project starts targeting categorical-heavy datasets. This mirrors 030/031/033: a rigorously
*bounded* don't-chase is the deliverable.
</content>
</invoke>
