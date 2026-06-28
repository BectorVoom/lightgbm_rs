---
spike: 052
name: cuda-launch-fusion
type: standard
validates: "Given 051's finding that real-CUDA hist+split is launch/sync-latency-bound (8570 launches / 2890 syncs, fused=0), when the existing build_fix_scan fusion (LGBM_FUSED_FORCE=1) + copack toggles are flipped on real NVIDIA, then launches/syncs/wall drop (=> wire fused default-on for cuda) OR stay flat (=> the per-leaf host-orchestration is the wall; need the architectural on-device learner)"
verdict: VALIDATED
related: [051, 048, 024, 023, 035]
tags: [gpu, cuda, kaggle, launch-fusion, syncs, round-trip, narrow-shape, orchestration]
---

# Spike 052: CUDA launch fusion — cut the 8,570 launches / 2,890 syncs

## What This Validates

Spike-051 localized the real-CUDA `hist+split` cost as **launch/sync-latency-bound**
(occupancy-insensitive; `build=0` async; 8570 launches / 2890 syncs / 100 trees at 500k×50)
and found **`fused=0`** — the prototyped directly-built-child fusion (`build_fix_scan`) is OFF
on the production CUDA path. This spike tests whether cutting launches/syncs via the **existing**
fusion levers pays on **real NVIDIA** (where PCIe sync latency is real), unlike the spoofed APU
where it benched flat-to-negative.

## Research

### Gating (read from source)
- **`sibling_copack` is DEFAULT-ON** (spike-024, `sibling_copack_override()` returns None ⇒
  `copack_override != Some(false)` ⇒ engages when `resident_eligible`). So the 2890 syncs are
  ALREADY the co-packed count (both siblings' scans share one launch+readback). `LGBM_SIBLING_COPACK=0`
  force-OFF (more syncs) is the control that confirms copack is currently helping.
- **`build_fix_scan` fusion is DEFAULT-OFF** (`fused_directly_built_eligible`,
  `FUSED_MAX_NUM_DATA = -1` ⇒ false for all real workloads — "flat-to-negative at every band"
  *on the APU*). `LGBM_FUSED_FORCE=1` forces it ON, fusing build+fix+scan for the directly-built
  (smaller) child into ONE launch (3→1 for that child), cutting `build_resident` + `scan_resident`
  launches and their drains.
- 051 micro-win available to stack: `LGBM_AUTOTUNE=0` (~4% t1iter on the narrow CUDA shape).

### Method — zero-code toggle sweep (existing master)
Arms under `LGBM_PHASE_PROF=1`: `baseline` / `fused=1` / `fused=1+AT0` / `copack=0` /
`fused=1+copack=1`. Read `launches / build / subtr / scan / fused / syncs / t1iter / wall`.
In-session deltas (absolute Kaggle walls drift across sessions). Kernel
`boomvector/lgb-rs-cuda-spike052`.

| Read-out | Conclusion | Next |
|---|---|---|
| `fused=1` cuts launches+syncs AND drops t1iter | per-leaf fusion pays on real CUDA | wire `fused` default-ON for cuda (a real code win; parity-gate it) |
| `fused=1` cuts launches but t1iter flat/worse | dispatch isn't the wall; the host round-trip-per-leaf design is | the lever is the architectural on-device multi-leaf learner (milestone) |
| `copack=0` raises syncs+wall | confirms copack is load-bearing on CUDA | keep copack default-on (already is) |

## How to Run

```bash
kaggle kernels push  -p kaggle_push_052
kaggle kernels status boomvector/lgb-rs-cuda-spike052
kaggle kernels output boomvector/lgb-rs-cuda-spike052 -p kaggle_out_052
```

## Investigation Trail

- 051 redirect: occupancy refuted, launch-bound, `fused=0`/`copack default-on`.
- Read the fusion gating; built the zero-code toggle sweep; pushed kernel v1.
- Result: BOTH fusion levers refuted. Verified the `fused=1` catastrophe is systematic
  (not a cold-JIT artifact): steady-state ~571ms/tree across all 3 forced-on arms vs
  ~95ms/tree separate; warmup dumps show the same direction.

## Results

**VERDICT: VALIDATED measurement — both cheap fusion levers REFUTED on real CUDA.**
Timed dumps (100 trees, 500k×50; `device_launches`=timed):

| arm | wall_s | t1iter | learner | phases | launches | build | subtr | scan | fused | syncs |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | **10.696** | **9479** | 8684 | 6192 | 8570 | 2890 | 2790 | 2890 | 0 | 2890 |
| fused=1 | 58.292 | 57122 | 56263 | 53611 | 8470 | 0 | 2790 | 2790 | 2890 | 2790 |
| fused=1+AT0 | 57.356 | 56195 | 55344 | 52722 | 8470 | 0 | 2790 | 2790 | 2890 | 2790 |
| copack=0 | 11.059 | 9868 | 9086 | 6639 | 11360 | 2890 | 2790 | 5680 | 0 | 5680 |
| fused=1+copack=1 | 57.580 | 56393 | 55533 | 52912 | 8470 | 0 | 2790 | 2790 | 2890 | 2790 |

### Finding 1 — `build_fix_scan` fusion is CATASTROPHIC on CUDA (5.4× worse) ⇒ keep OFF
Forcing fusion replaced 2890 separate builds with 2890 fused launches (−100 net launches)
but blew `phases` from 6192→53611ms (**t1iter 9479→57122ms, ~6×/tree: 571ms vs 95ms**).
Systematic across all 3 forced-on arms; not a cold artifact. **Mechanism:** the fused
kernel (`build_fix_scan_resident_f64_on`) does **f64 build+scan together**, and consumer
NVIDIA (T4/P100) f64 throughput is **1/32 of f32** — so the giant f64 fused kernel is
f64-throttled. The separate path is fast because it uses **u64 fixed-point** integer-atomic
build (spike-018) + f64 scan only on the spine. The spike-024 "flat-to-negative on the APU"
becomes "**catastrophic on real CUDA**" — `FUSED_MAX_NUM_DATA=-1` (default-off) is **correct
and now validated on real hardware**. Do NOT wire fusion on for cuda. (A *non-f64* fused
kernel might recover this, but that's a new kernel, not a toggle.)

### Finding 2 — syncs are CHEAP (~0.14ms) ⇒ 051's "sync-latency" framing REFINED
`copack=0` doubled syncs (2890→5680) for only **+390ms** (+3.6% t1iter) ⇒ **~0.14ms/sync**;
the 2890 baseline syncs cost only ~400ms of the 9479ms. **Readback-sync latency is NOT the
wall.** This refines spike-051: the cost is the **8570 small SERIAL kernel launches**
(~0.72ms each = `phases`/launches), dependency-chained per node (build→subtract→scan before
the next split is picked in best-first growth) — not the readback syncs. Sibling co-pack
(default-on) IS load-bearing (saves 2790 launches/syncs for ~390ms — keep it on), but there
is no further sync-reduction headroom worth chasing.

### Signal for the build — the cheap-CUDA-win search is CLOSED
051 (occupancy) + 052 (fusion + sync) **exhaust the cheap levers** — all refuted on real
NVIDIA. The narrow-shape lgb_rs-CUDA gap (~5–6× vs official) is **architectural**: a
host-driven, per-leaf growth loop issuing 8570 small launches gated by the best-first
build→subtract→scan dependency chain, vs official's `CUDASingleGPUTreeLearner` that grows the
whole tree **on-device** with far fewer, bigger kernels. **The only real lever is the
on-device multi-leaf tree learner — a milestone, not a spike.** Near-term: route narrow
shapes per the 054 crossover; the `f64`-throttle insight also flags that any future CUDA
kernel work should avoid f64 hot loops on consumer GPUs (prefer the u64 fixed-point path).
- Evidence: `kaggle-run.log`. Both fusion-on arms reproduce within 1.2% of each other.
