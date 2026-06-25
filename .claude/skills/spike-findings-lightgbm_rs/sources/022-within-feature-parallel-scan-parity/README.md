---
spike: 022
name: within-feature-parallel-scan-parity
type: standard
validates: "Given a within-feature parallel best-split scan (reordered f64 prefix-sum, the cubecl-hip plane_inclusive_sum order), when its default_left tie-flips are classified by modelling the default-bin (most_freq_bin) semantics spike-016 omitted, then are the flips COSMETIC (missing-routing-only, present-data predictions unchanged) or REAL (a populated default bin routed differently → tree divergence)?"
verdict: VALIDATED (parity GATE resolved) — PARITY-SAFE within ~1e-6; every default_left flip is COSMETIC, threshold flips are equal-gain plateaus. Closes spike-016's deferred question. ROI still low (dominated by spike-021 at wide; narrow-only).
related: [016, 021, 015]
tags: [performance, gpu, rocm, scan, split, parity, default-left, default-bin, reorder, host-probe, cheap-probe, plane-scan]
---

# Spike 022a: within-feature parallel-scan parity — the default_left resolution

## What This Validates

spike-016 probed parallelizing the within-feature best-split scan (which reorders the f64
prefix-sum) and found: **0 threshold flips** on realistic data (partition parity-safe), but
**~34% default_left flips** at equal-gain reverse-vs-forward ties — and **could not classify
them** because its host probe modelled only the `offset=0` / no-skip path, omitting the
default-bin (`most_freq_bin`) semantics. It deferred the question to "the actual GPU kernel
run against `kernel_parity_split_within_tol_on_hip`."

This spike resolves it **on the host** (cheaper than a GPU kernel, the scope chosen) by
closing spike-016's two stated limitations:
1. **Models the default-bin semantics** (`offset`, `skip_default_bin`, default-bin mass) so
   each flip can be classified COSMETIC vs REAL.
2. **Uses the EXACT reorder cubecl-hip 0.10 emits** for `plane_inclusive_sum` — a
   Hillis-Steele `__shfl_up` loop (`reduce_inclusive` in `cubecl-cpp/.../warp.rs`), not
   spike-016's "pairwise-tree" proxy.

## Research (grounded in the installed crate)

- **cubecl-hip 0.10 supports `plane_inclusive_sum`** — it lowers (no HIP override) to the
  default `reduce_inclusive`: a Hillis-Steele loop `for offset=1,2,4…: tmp=__shfl_up(acc,offset);
  if(lane>=offset) acc+=tmp;`. This **reorders the f64 adds** (each prefix becomes a balanced
  tree), confirming feasibility AND giving the faithful reorder order to model.
- **Architectural caveat (feasibility, not parity):** `num_bin` reaches 256 ≫ `PLANE_DIM`
  (32/64), so a real within-feature scan needs a **segmented/LDS block-scan**, not a single
  plane scan — a substantial kernel. (Relevant to the deferred 022b perf question, not to
  this parity gate.)

## How to Run

```
cargo run --release -p lgbm-compute --example spike022_default_bin_parity_probe
SPIKE022_TIESTRESS=0.3 cargo run --release -p lgbm-compute --example spike022_default_bin_parity_probe
```

`examples/spike022_default_bin_parity_probe.rs` — pure host f64, calls the REAL
`gain::{get_split_gains, calculate_splitted_leaf_output}`. 240k random histograms (bins
16/64/128/256 × offset∈{0,1} × default-bin mass {empty,populated}) compared seq-vs-HS, plus
a direct default-bin mass-sweep mechanism demo.

## Investigation Trail

1. **Decompose / reframe (ultra-think).** spike-016's ROI was assessed *before* spike-021
   shipped feature-per-lane scan packing. Post-021 the device is throughput-saturated at wide
   shapes, so within-feature parallelism only helps NARROW (where the GPU is least competitive
   vs the CPU anchor). Chose to first resolve the **parity gate** (the genuine open knowledge),
   deferring the (likely-low-ROI) perf prototype.
2. **First probe run — bug + signal.** Classifier reported `inf` leaf-output divergence and
   "REAL" for cosmetic cases: the bin-0-move reconstruction was wrongly applied to `offset=0`
   flips (which move only MISSING values, no histogram mass). BUT the key column was already
   decisive: `flip_defFULL = 0` — no flip ever had a populated default bin.
3. **Fixed classification** into three buckets (missing-only / empty-default / FULL-default)
   with per-offset-correct leaf-output divergence. Result: all flips cosmetic, max present-data
   leaf-output Δ = **exactly 0.0**, threshold flips are ~1e-13 gain plateaus.
4. **Chased the surprise.** `offset=1` produced *zero* flips — so the random sweep never
   directly landed a populated-default near-tie. Added a **direct mechanism demo**: at a fixed
   split, sweep the default-bin mass and compare the reverse-vs-forward gain gap to the reorder
   noise floor (ε≈1e-12). Proved the gap is **linear in mass**, so only mass ≲ 1e-12 (empty)
   can be flipped by the reorder — and there the routing moves no data (cosmetic).

## Results

### Random sweep (240k histograms; TIESTRESS 0.0 and 0.3 both)

| metric | realistic | near-tie stress |
|--------|-----------|-----------------|
| threshold flips | 201 / 240k, worst gain reldiff **1.4e-13** | 340 / 240k, **2.0e-13** |
| default_left flips | 34322 — **all missing-only (offset 0)** | 34237 — all missing-only |
| flips with EMPTY default bin | 0 | 0 |
| flips with **POPULATED** default bin | **0** | **0** |
| max present-data leaf-output Δ over ALL flips | **0.000e0** | **0.000e0** |
| REAL flips (present Δ > 1e-6) | **0** | **0** |

### Mechanism demo (default-bin mass sweep at a fixed split)

| default-bin mass (h) | gain gap (rev−fwd) | gap < ε≈1e-12? | leaf-output Δ |
|----------------------|--------------------|----------------|---------------|
| 0 (empty) | 0.0 | yes → **COSMETIC** | 0.0 |
| 1e-9 | 7.4e-10 | no → stable | 2.5e-12 |
| 1e-6 | 7.4e-7 | no → stable | 2.5e-9 |
| 1e-3 … 20 | 7.4e-4 … 7.7 | no → stable | 2.5e-6 … 2.5e-2 |

**The gain gap is linear in default-bin mass.** A flip needs gap < reorder noise (~1e-12),
i.e. mass ≲ 1e-12 — an (essentially) empty bin, which moves no present data. Any real
populated default bin (mass ≥ 1e-9) yields a gap ≫ noise ⇒ argmax stable ⇒ no flip.

**VERDICT: PARITY-SAFE within ~1e-6.** Resolves spike-016's deferred question:
- **Threshold** (the partition of present data): stable; the rare flips are equal-gain
  plateaus (~1e-13) — arbitrary tie-breaks between equally-good splits, the same class the hip
  split parity test is already tie-aware for (def-hip-split, 1832206).
- **default_left**: every flip is COSMETIC — it only reroutes MISSING values (no histogram
  mass) or an empty default bin; present-data leaf outputs are unchanged (Δ = 0.0). A populated
  default bin can never flip (gain gap ≫ reorder noise).

A within-feature parallel scan with a **tie-aware argmax** (reverse-first / lowest-`t`, per
spike-016 rec #3) would therefore choose the **same splits and same present-data leaf values**
within the ~1e-6 hip gate.

## Signal for the build (wire / don't wire)

- **Parity is NO LONGER the blocker** — spike-016's deferred risk is retired (safe within
  ~1e-6, present-data bit-structure preserved). The residual default_left flips are cosmetic
  (missing-value routing), themselves an f32-tie residual the tie-aware hip test tolerates.
- **ROI is still LOW (the real blocker).** Post-spike-021 (feature-per-lane, shipped), the
  scan already saturates the device with feature-level parallelism at wide shapes; within-
  feature parallelism (one feature per plane, segmented LDS block-scan) trades that for shorter
  per-feature latency — a win **only at narrow shapes** where spike-021 under-fills, and narrow
  is exactly where the GPU is least competitive vs the multi-threaded CPU anchor. Plus it is a
  substantial kernel (256-bin block-scan + tie-aware plane-argmax + default-bin handling).
- **Recommendation: DON'T WIRE now; the gate is GREEN for later.** Revisit only if narrow-shape
  training perf on **real discrete gfx110x** (not the spoofed 8-CU APU) becomes a priority. The
  next step would be 022b (the deferred perf prototype): isolated cooperative-scan vs
  feature-per-lane A/B across feature counts to confirm the narrow-wins/wide-regresses crossover.

## Limitations (honest)

- Host probe, not the GPU kernel — but the reorder order is now the **exact** Hillis-Steele
  cubecl-hip emits, and the parity question is a property of the f64 argmax under reordering,
  which the host reproduces faithfully (the GPU f32-vs-f64 envelope is a separate, larger,
  already-documented residual that only *widens* the cosmetic tie band, never creates a real
  partition change the host doesn't see).
- The `offset=1` random path under-generated populated-default near-ties; the **direct
  mechanism demo** covers that scenario explicitly (gap linear in mass), so the conclusion does
  not rest on the random sweep alone.
</content>
