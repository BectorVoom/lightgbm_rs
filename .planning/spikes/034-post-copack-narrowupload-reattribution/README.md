---
spike: 034
name: post-copack-narrowupload-reattribution
type: measurement
validates: "Given the GPU train with spike-024 (sibling-scan co-pack) AND spike-029 (narrow-upload partition) BOTH now SHIPPED since the last full attribution (spike-023), when re-profiled across both regimes (small/medium launch-bound + wide 250k–1M×500 compute-bound) with the whole-train BUDGET + per-tree COUNTS + LGBM_SCAN_DRAIN build-drain + the LGBM_BENCH_COPACK_AB ON/OFF A/B, then the CURRENT dominant reclaimable residual is named and we learn whether the bottleneck moved a 4th time (014→015→023→?)"
verdict: PENDING
related: [023, 024, 029, 015, 014, 030, 021]
tags: [performance, gpu, rocm, profiling, attribution, re-profile, regime-split, post-wire, measurement]
---

# Spike 034: Post-(024+029) GPU Bottleneck Re-Attribution

## What This Validates

Given/When/Then — see frontmatter. The campaign's iron rule (CONVENTIONS.md, MANIFEST
Requirements): **re-profile after EVERY build change — the bottleneck has moved three times
(014→015→023).** Two build changes shipped since spike-023's attribution:

- **Spike-024 sibling-scan co-pack** (Phase 12, default-ON via `LGBM_SIBLING_COPACK`):
  co-packs the two siblings of every split into ONE 2-slot scan launch + ONE readback →
  per-tree `scan_resident` syncs ~59 → ~30. Targets the **launch-bound (small/medium)** floor.
- **Spike-029 narrow-upload partition fuse** (quick-260625-j1l, default-ON, structural):
  ROCm partition uploads leaf bins at native width (count×u8) not u32-widened → ~4× fewer
  upload bytes + 4× fewer device bin reads on the production U8 case. Targets the **GPU
  partition device round-trip**.

So the spike-023 attribution is now STALE in exactly the two phases these levers touched.
This spike re-runs the 023 matrix and diffs.

## Research / Prior Art

This is a re-run of the spike-023 attribution harness — no new instrumentation needed; all
env knobs already exist in `crates/lgbm/examples/bench_gpu_vs_cpu.rs` +
`crates/lgbm-treelearner/src/phase_prof.rs`:

- `LGBM_PHASE_PROF=1` → whole-train **BUDGET** line (`binning/grad/learner/score`, with
  `in_learner_other` = `learner − phases` and its `resident_bin_upload` drill-down) + the
  spike-023 **COUNTS** line (`device_launches`, `build_resident/subtract_resident/scan_resident/fused`,
  `scan_roundtrips(syncs)`).
- `LGBM_SCAN_DRAIN=1` → forces a pre-scan build drain so build-compute is separated from the
  scan launch+readback (defeats the async-artifact that made spike-015's "scan=96%" wrong).
- `LGBM_BENCH_COPACK_AB=1` (needs PHASE_PROF + rocm) → co-pack ON vs OFF, median train +
  sync-count delta. This is the **direct measurement of 024's production effect**.
- `LGBM_BENCH_SWEEP=wide` → 250k/500k/1M × 500 feat, bins=128 (compute-bound regime).
- default sizes → small/medium/large (launch-bound regime).

**HARDWARE CAVEAT (load-bearing):** this box is a spoofed 8-CU gfx1152 APU (Radeon 860M) on
shared DDR5, NOT a discrete gfx1100. Absolute perf is APU-confounded; rocprof HW counters are
unavailable. Judge **SIGN + counts + fractions**, not absolute Mr/s. ≥2 process restarts for
sign-stability (CONVENTIONS device-time discipline).

## Spike-023 Baseline (what we diff against)

Per-tree COUNTS @ num_leaves=31 (shape-independent):
- ~30 build_resident + ~29 subtract_resident + **~59 scan_resident SYNCS** (one per leaf-node;
  both siblings of every split scanned in TWO separate launches+readbacks) = **~118 launches/tree**.

DRAIN attribution (build-compute vs genuine scan-sync):
- **launch-bound (small/medium):** scan-sync = **~48% / ~35%** of the scan round-trip
  (~44µs/sync ≈ pure fixed latency at small); host partition **~13%**.
- **compute-bound (large/wide):** build-compute DOMINATES and GROWS with rows
  (**68% → 96.5% @1M×500**, undiminished by u64); scan-sync collapses to **3.2%**; host
  `partition` (single-threaded `DataPartition::split`) grows to **~23%** — a CPU-track residual
  no GPU lever touches.

**Predictions to test (H):**
- H1: co-pack ON → `scan_resident` syncs/tree ≈ **30** (was ~59); co-pack OFF reproduces ~59.
- H2: launch-bound regime — the scan-sync fraction ~halves; the NEW dominant reclaimable
  residual in small/medium is something ELSE (build-compute? host partition? loop overhead?).
- H3: compute-bound regime — build-compute still dominates (~unchanged; neither lever touches
  build); partition's GPU device-round-trip share drops (029), but the HOST single-threaded
  `DataPartition::split` residual may be unchanged (029 narrowed the device upload, not the host gather).

## How to Run

```bash
# Build once
cargo build --release --features rocm --example bench_gpu_vs_cpu

# A. Launch-bound regime (small/medium/large) — full attribution
LGBM_PHASE_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu 2>&1 | tee runA.log
# A-drain: separate build-compute from scan-sync
LGBM_PHASE_PROF=1 LGBM_SCAN_DRAIN=1 cargo run --release --features rocm --example bench_gpu_vs_cpu 2>&1 | tee runA_drain.log
# A-copack: the direct 024 ON/OFF measurement (sync count + e2e)
LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu 2>&1 | tee runA_copack.log

# B. Compute-bound regime (250k/500k/1M × 500)
LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu 2>&1 | tee runB.log
LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1 LGBM_SCAN_DRAIN=1 cargo run --release --features rocm --example bench_gpu_vs_cpu 2>&1 | tee runB_drain.log

# Repeat A-copack and B across >=2 process restarts for sign-stability.
```

## Investigation Trail

**Shapes (this box):** launch-bound = small 2k×12×32 / medium 20k×30×64 / large 200k×40×128
(iters=50, reps=5 ⇒ 250 timed trees); wide = 250k/500k/1M × 500 × 128 (iters=8, reps=3 ⇒ 24
timed trees). Backend `rocm(gfx1100)` = spoofed 8-CU gfx1152 APU. `rowpart_min=256000` ⇒
row-partition INACTIVE at every launch-bound size and at 250k wide.

1. **Run A (launch-bound, PHASE_PROF).** COUNTS: medium `scan_resident=7500/250 = 30/tree`,
   large `7450/250 = 29.8/tree` (was ~59 in spike-023). Co-pack confirmed LIVE. % split:
   medium `hist+split=62.0 (scan=3.0) partition=38.0`; large `hist+split=70.3 (scan=6.9)
   partition=29.7`; small `hist+split=86.3 (build=17.8 scan=53.8) partition=13.7`. **Surprise:
   partition is now the #1–#2 phase at medium/large; scan collapsed.** small uses the LEGACY
   build path (build≠0, no COUNTS); medium/large use the RESIDENT path (build=0 in the legacy
   sub-timer, COUNTS fire).
2. **Run A-DRAIN.** `LGBM_SCAN_DRAIN=1` did NOT move medium/large (`build=0` persists) — the
   spike-023 drain hook no longer re-attributes build-vs-scan on the resident+co-pack path.
   Tooling note, not a blocker (the BUDGET + COUNTS still attribute at phase granularity).
3. **Run B (wide, PHASE_PROF) + B-DRAIN.** All wide sizes `scan_resident=720/24 = 30/tree`
   (co-pack live at wide too). `%: hist+split≈91 (scan 15.6→22→28.1 GROWS w/ rows) partition≈9`.
   The "scan" timer GROWING with rows = async build-compute draining inside the scan readback
   (spike-015 artifact; DRAIN ineffective here too). True scan-sync is tiny; **build dominates,
   partition only ~9%.** `in_learner_other ≈ 24–28% of learner` (resident_bin_upload ≈ ¼ of it;
   binning is a per-rep bench artifact that amortizes in bin-once-train-many).
4. **Co-pack A/B (direct 024 proof, ×2 restarts).** medium/large `syncs_off≈2930–2950,
   syncs_on≈1490–1500` ⇒ exact ~59→~30/tree (1.97×). e2e `off/on`: large **1.09 / 1.11**
   (sign-stable WIN ~9–11%); medium 0.95 (neutral — scan is only 3% there); small inactive
   (syncs=0). The e2e win tracks the scan fraction, as predicted.
5. **Restart #2 (Run A).** partition medium 38.1 / large 30.2 / small 13.5; scan medium 3.2 /
   large 7.0 — all within ~0.5pp of restart #1. **Sign-stable.**

## Results

**VERDICT: ✅ VALIDATED (measurement) — the bottleneck MOVED a 4th time (014→015→023→**034**),
and the move is REGIME-SPECIFIC.**

### Both shipped levers confirmed live and working
- **Spike-024 co-pack:** per-tree `scan_resident` syncs **59 → 30** (counter-exact, 1.97×, both
  restarts, both medium & large). e2e **~9–11% faster at large** (sign-stable), neutral at
  medium/small where scan is already <4% of train. Working exactly as designed.
- **Spike-029 narrow-upload:** the rocm partition runs the native-width device path (no u32
  widen). Its isolated delta isn't separable from this whole-train run, but the path is live.

### Launch-bound regime (small/medium/large) — THE BOTTLENECK MOVED
| Phase | spike-023 (pre-wire) | spike-034 (post-wire) |
|-------|----------------------|------------------------|
| scan-sync | ~48% / ~35% of round-trip (the bottleneck) | **3.0% (med) / 7.0% (large)** — co-pack closed it |
| **partition** | ~13% | **38% (med) / 30% (large)** — NEW #1 reclaimable |
| hist+split (build+subtract+fix+splitfind) | — | ~62% (med) / ~70% (large), build-dominated |

The per-leaf scan-sync floor that 023 named as the small/medium bottleneck is **closed by
co-pack**. The new dominant reclaimable phase is **partition** — the device `data_partition_native`
round-trip (host gather → narrow upload → route kernel → blocking readback), ~30 splits/tree
each a separate sync. 029 cut its *bytes*, not its *per-split round-trip structure*.

### Wide / compute-bound regime (×500) — UNCHANGED
`hist+split ≈ 91%` of phases (build-compute dominated, uncoalesced-gather-bound per spike-030),
partition only **~9%**, scan-sync tiny. Neither lever targets this regime (024 predicted ~1.5%;
029 only touches the 9% partition) — confirmed. Still closed on this APU (031); re-run on
discrete gfx110x.

### Tooling note (parked)
`LGBM_SCAN_DRAIN` no longer separates build-compute from scan on the resident+co-pack path
(`build=0` in both modes, all resident shapes). The spike-023 drain hook needs re-wiring after
Phase 12 if a future spike needs the build-vs-scan split inside resident `hist+split`.

### Next-lever hypothesis (NOT built here — gates a future spike)
The new launch-bound bottleneck (**partition 30–38%**) points at: **route partition to the HOST
on the rocm backend** via the SHIPPED spike-027 fused-gather host partition (1.3–2.7×, bit-exact,
no device round-trip / no per-split sync). On this shared-DDR5 APU the device round-trip buys
little. Caveat: the resident build reads indices on-device, so host partition trades the route
round-trip for an index re-upload — a transfer tradeoff to MEASURE, not assume (029 proved
transfer is not free even on an APU). ROI-gated as ever: CPU wins overall on this 8-CU APU; real
payoff on discrete gfx110x. **This is the recommended 035.**

### Caveats
Spoofed 8-CU gfx1152 APU; absolute ms APU-confounded; SIGN + counts + fractions only. Phase
fractions are stable to ~0.5pp across 2 restarts. Bit-exact gate untouched (instrumentation-only
run; no code changed).
