---
spike: 051
name: real-cuda-hist-reattribution
type: standard
validates: "Given the instrumented CUDA wheel on Kaggle real NVIDIA (500k x 50), when the GPU hist+split phase is probed by an LGBM_AUTOTUNE_FORCE_P occupancy sweep {1,4,8,16,32,64,128} vs default-autotune, then the dominant reclaimable sub-phase is localized — occupancy-starved build COMPUTE (=> 053 lift the P/BUILD_PSET ceiling) vs launch/orchestration overhead (=> 052 fuse launches)"
verdict: VALIDATED
related: [040, 048, 049, 030, 037]
tags: [gpu, cuda, kaggle, profiling, re-attribution, occupancy, row-partition, build, narrow-shape]
---

# Spike 051: Real-CUDA histogram-phase re-attribution (occupancy sweep)

## What This Validates

The 001–040 GPU perf campaign tuned the build kernel on a **spoofed 8-CU APU**.
Spikes 048/049 gave the first real-NVIDIA wall map (post-metric-fix, ~11s, 500k×50):
**GPU histogram phases = 53%** (hist+split 26% / partition 10% / in_learner_other 15%).
The campaign's iron rule — *re-profile after every change; the bottleneck has moved 4×* —
plus the SKILL's explicit note that the 030 build attribution "reopens only on discrete
hardware — re-run the probe there" — means the GPU build must be **re-attributed on real
CUDA before any optimisation**.

Decisive question: is the `hist+split` (build+subtract) device-time **occupancy-starved
COMPUTE** (lever = lift the row-partition `P` / `BUILD_PSET` ceiling, spike-053) or
**launch/orchestration-bound** (lever = fuse the 8,570 launches, spike-052)?

## Research

### What's already in production (read from source — no assumptions)
- The production CUDA path is `CudaBackend = GpuBackend<CudaRuntime>` (one generic impl,
  `lib.rs:2116/2126`). The per-leaf resident build funnels through
  `resident_raw_build_into` (`histogram.rs:2034`), launching
  `construct_leaf_hist_resident_lds_kernel_u64` as `CubeCount::Static(num_features, P, 1)`
  × `CubeDim(256)`. **At 50 features, P=1 ⇒ only 50 workgroups** on a GPU with thousands
  of cores.
- **Row-partition `P` is chosen three ways** (`histogram.rs:2187`): `LGBM_AUTOTUNE_FORCE_P=k`
  pins P=k (NO clamp — bypasses `ROWPART_P_MAX=16`); else **autotune (default-ON,
  `autotune_enabled()` = true unless `LGBM_AUTOTUNE=0`)** picks P over
  `BUILD_PSET = [1,4,8,16,32]` keyed on `size_band(rows)`; else the
  `row_partition_count` heuristic (P=1 at 50 feat — the spike-040 latent mis-tune).
- So on Kaggle the build **already autotunes** P∈{1..32}. Two open holes this probe targets:
  1. **`BUILD_PSET` ceiling = 32.** A big NVIDIA GPU (T4 40 SM, P100 56 SM) may want
     P>32 (P=32 ⇒ 1600 workgroups; could still under-occupy). If build keeps dropping
     past 32, lifting the ceiling is the spike-053 win — bigger on NVIDIA than the APU's 10%.
  2. **Does autotune even work on cubecl-cuda?** It was validated on cubecl-hip 0.10
     (037–040). If default-autotune ≠ the best forced P, autotune mis-fires on cuda.

### Method — "remove the suspect" via existing toggles (zero code change)
Sweep `LGBM_AUTOTUNE_FORCE_P ∈ {1,4,8,16,32,64,128}` vs `default(autotune)` vs
`LGBM_AUTOTUNE=0(heuristic)`, all under `LGBM_PHASE_PROF=1`, one fresh subprocess per arm
(phase_prof atomics are process-global). Read the `hist+split` / `build` device-time per
arm from the dump. Modeled on the proven 046/048 Kaggle harness; absolute walls are NOT
cross-session comparable (Kaggle assigns T4/P100/T4×2) ⇒ trust **in-session deltas**.

| Read-out | Conclusion | Next |
|---|---|---|
| build keeps DROPPING past P=32 | ceiling too low on NVIDIA | **053 GREEN** (lift BUILD_PSET) |
| build saturates at P≤32, default(autotune) matches it | autotune works; build is tuned | residual ⇒ **052** (fusion) |
| default(autotune) ≫ best forced P | autotune mis-fires on cubecl-cuda | fix autotune-on-cuda first |
| build ≈ flat vs P | NOT occupancy-bound — launch/orchestration | **052** (fuse the 8,570 launches) |

**Chosen approach:** self-contained Kaggle driver `spike051_kaggle.py` (inner bench
inlined; no git push — FORCE_P/autotune are existing toggles on current master 3975ca6).
Kernel `boomvector/lgb-rs-cuda-spike051`.

## How to Run

```bash
kaggle kernels push -p kaggle_push_051
kaggle kernels status  boomvector/lgb-rs-cuda-spike051   # poll to COMPLETE
kaggle kernels output  boomvector/lgb-rs-cuda-spike051 -p kaggle_out_051
```

## What to Expect

A per-arm table: `wall_s | histsplit | build | scan | partn | launches` for each forced P,
default-autotune, and heuristic, plus the official-CUDA wall for the gap. The `build`
column vs P is the verdict signal.

## Investigation Trail

- Mapped the production launch path (Explore agent + source read): build → `resident_raw_build_into`
  three-way P pick; confirmed `LGBM_AUTOTUNE_FORCE_P` flows into the live u64 build and is
  unclamped; confirmed `BUILD_PSET` ceiling = 32 and autotune default-ON.
- Built the zero-code occupancy-sweep probe (existing toggles) and pushed kernel v1.
- **Parser bug caught:** my summary table read the `%:` percentage line (`hist+split=73.8`)
  as the ms value. The RAW dumps (`phase_prof_dumps.txt`) are the truth and **reconcile
  exactly with 048/049** (timed = the `device_launches=8570` dump). Each arm emits a warmup
  dump (445 launches — absorbs the cold CUDA-context/kernel-JIT, ~4.4s) then the timed
  100-tree dump (8570 launches). Always read the absolute-ms line, not the `%:` line.

## Results

**VERDICT: VALIDATED.** Real-CUDA timed dumps (100 trees, 500k×50; `device_launches=8570`):

| arm | hist+split | scan | partition | phases | in_lrn_other | learner | train_one_iter |
|---|---|---|---|---|---|---|---|
| force_p=1     | 4897 | 477 | 1742 | 6639 | 2553 | 9192 | **10004** |
| force_p=4     | 5181 | 300 | 1733 | 6913 | 2552 | 9466 | 10274 |
| force_p=8     | 5348 | 292 | 1730 | 7078 | 2560 | 9639 | 10457 |
| force_p=16    | 5481 | 286 | 1718 | 7199 | 2546 | 9745 | 10546 |
| force_p=32    | 5478 | 283 | 1758 | 7236 | 2592 | 9828 | 10646 |
| force_p=64    | 5330 | 284 | 1737 | 7067 | 2570 | 9637 | 10454 |
| force_p=128   | 5261 | 285 | 1702 | 6963 | 2485 | 9448 | 10242 |
| default(autotune) | 5188 | 287 | 1712 | 6901 | 2531 | 9432 | 10245 |
| **autotune=0 (P=1)** | 4844 | 473 | 1720 | 6564 | 2492 | **9056** | **9862** |

(ms; build=0 in every arm — see async note. COUNTS identical across arms:
device_launches=8570 build_resident=2890 subtract_resident=2790 scan_resident=2890
**fused=0** | syncs=2890.)

### Finding 1 — build occupancy is NOT the bottleneck ⇒ **spike-053 REFUTED on real CUDA**
The `LGBM_AUTOTUNE_FORCE_P` sweep is **flat-to-slightly-worse**: `train_one_iter` is BEST at
**P=1** (10004ms) and monotonically *worse* to P=16 (10546ms), recovering slightly by P=128.
Forcing more build parallelism on a real NVIDIA GPU does **nothing good** here. Lifting the
`BUILD_PSET` ceiling (>32) — the spike-053 hypothesis — is **refuted**: there is no occupancy
headroom to reclaim at the narrow 50-feature shape. The APU's P-sensitivity (spike-040, ~10%)
**does not transfer** to discrete CUDA. (FORCE_P verifiably took effect: `autotune=0` and
`force_p=1` match within noise; scan/histsplit shift with P.)

### Finding 2 — autotune slightly UNDERPERFORMS plain P=1 on cuda
`autotune=0` (heuristic, pins P=1) is the FASTEST arm (9862ms t1iter, 9056ms learner) —
~4% better than `default(autotune)` (10245ms). On the spoofed APU autotune *won* ~10%
(040); on real CUDA narrow-shape it adds cold-tune + per-regime overhead for **no** gain
(the optimum is just P=1). Autotune's value is APU-specific; consider `LGBM_AUTOTUNE=0` as
the CUDA narrow-shape default (or skip the cold-tune when BUILD_PSET's optimum is P=1).

### Finding 3 — the cost is LAUNCH/SYNC-LATENCY-bound ⇒ **spike-052 GREEN**
`build=0` because the build kernel is **async-issued** (no readback); its device compute
drains at the **2890 scan syncs**, so `hist+split` conflates build-compute + subtract +
sync-latency. The occupancy-*insensitivity* (Finding 1) disambiguates it: a compute-bound
kernel on a big GPU would speed up with P; this one doesn't ⇒ it is **launch/sync-latency-bound**,
not compute-throughput-bound. The narrow 50-feature shape makes each of the 8570 launches do
tiny work, so per-launch dispatch + the 2890 round-trip syncs dominate. **And `fused=0`** —
the prototyped fusion (spike-024 sibling co-pack, `build_fix_scan` directly-built child) is
**OFF on the production CUDA path**. Cutting launches/syncs via fusion (**spike-052**) is the
real lever, and is testable next with existing toggles (`LGBM_SIBLING_COPACK=1`,
`LGBM_FUSED_FORCE=1`).

### Reconciliation with 048/049
Default(autotune) here: hist+split 5188 / scan 287 / partition 1712 / in_lrn_other 2531 /
learner 9432 — matches 048 (hist+split 4508 / scan 286 / partition 1710 / ilo 2573) and 049
within cross-session GPU drift. The 53% "GPU histogram phases" map holds; this spike adds the
mechanism: **launch/sync-bound, not occupancy-bound**.

### Signal for the build
- **Drop spike-053** (build occupancy/PSET-ceiling) — refuted on real CUDA. Repurpose its slot.
- **Promote spike-052** (launch/sync fusion) to the primary lever. First probe: flip the
  existing `LGBM_SIBLING_COPACK` / `LGBM_FUSED_FORCE` toggles on CUDA, confirm syncs 2890↓ and
  wall↓; then wire fusion default-on for cuda if it wins.
- **Minor CUDA win available now:** `LGBM_AUTOTUNE=0` (~4% t1iter) on the narrow shape.
- Evidence: `kaggle-run.log`, `phase_prof_dumps.txt`. APU caveat lifted — this is real NVIDIA.
