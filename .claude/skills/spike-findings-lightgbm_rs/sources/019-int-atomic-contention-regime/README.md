---
spike: 019
name: int-atomic-contention-regime
type: standard
validates: "Given the spike-018 fixed-point u64-atomic win (~1.9× at single-cube P=1), when row-partition P and cube-occupancy and rows-per-cube are swept, then we learn whether the win is a measurement artifact (under-occupancy) or a real regime effect, and whether integer atomics SUBSTITUTE for or COMPOSE with row-partition"
verdict: VALIDATED + CORRECTS 018 — the win is REAL and sign-stable (~1.3–1.7×) in HIGH-atomic-load regimes (the wide root/large leaves), COMPOSES with row-partition; 018b's 1.9× was single-cube-inflated; null only at light load
related: [018, 007, 015]
tags: [performance, gpu, rocm, histogram, integer-atomics, row-partition, contention, regime, occupancy, wide-shape]
---

# Spike 019: Integer-atomic win — which regime is it real in?

## What This Validates

Spike-018b measured u64 fixed-point atomics ~1.9× faster than f32 at single-cube P=1.
This spike stress-tests that: is it an UNDER-OCCUPANCY artifact (1 cube on 8 CUs = 7
idle) or a real ROWS-PER-CUBE/contention effect? And does row-partition (spike-007)
KILL the win (substitutes) or not (composes)? Both kernels are the 007 `build_rp`
layout (CubeCount=(feats,P)); the f32 and u64 twins differ ONLY in atomic type, so the
per-cell ratio isolates atomic-type cost. Harness: interleaved median+p25/p75 over 9
reps, 2 process runs (CONVENTIONS).

## Results (2 process runs, sign-stable)

`i64/f32` = f32_median / u64_median (>1 ⇒ u64 atomics faster). SEP = bands separated.

| config (cubes×rows/cube @P1) | total load | P=1 | across P=1..16 | verdict |
|------------------------------|-----------:|----:|----------------|---------|
| **16×1M** (occupied, heavy)  | 16M rows | **1.57–1.70× SEP** | **all P SEP, 1.23–1.70×** | robust win |
| **64×200k** (well-occupied)  | 12.8M    | 1.16–1.21× SEP | 1.09–1.28× | win |
| **16×200k** (occupied, light)| 3.2M     | 1.03–1.05× | ~1.0× (overlap) | **NULL** |
| **1×1M** (=018b under-occ.)  | 1M       | 1.31–1.39× SEP | noisy 1.1–1.4× | modest |

## Diagnosis (corrects spike-018)

1. **NOT a pure under-occupancy artifact.** The win persists at 16 and 64 cubes
   (occupied) — 16×1M is the strongest cell (1.6×), not the 1-cube one.
2. **The determinant is TOTAL ATOMIC LOAD / rows-per-cube, not occupancy.** Heavy load
   (16×1M, 64×200k) → robust SEP win; light load (16×200k = 3.2M rows) → null. f32
   `atomicAdd` on RDNA lowers to a CAS retry loop (`ds_cmpst`); under heavy atomic
   pressure the retries serialize and dominate, while integer `ds_add_u64` is a native
   single-instruction op that never retries (finding #3 / AMD atomics docs). Light load
   doesn't saturate the atomic units, so f32 keeps up.
3. **Integer atomics COMPOSE with row-partition (not pure substitutes).** In 16×1M the
   win survives all the way to P=16 (1.23×). Row-partition splits rows across MORE cubes
   but the device-wide atomic pressure is unchanged, so f32 still pays the CAS-retry tax
   — integer atomics still help on top.
4. **018b's 1.9× was inflated** by its single-cube + SIMPLE kernel (direct `binned[i]`,
   no `leaf_rows` indirection). The realistic resident `build_rp` kernel (double
   indirection, shared by both arms) dilutes the ratio to **~1.3–1.7×** in the heavy
   regime. Still a robust, real win — just not 1.9×.

## Mapping to wide training (why this matters)

The wide shape (1M×500) at the ROOT leaf launches ~500 cubes × ~1M rows = ~500M
atomic-pairs/launch — FAR into the heavy-load regime where the win is ~1.3–1.7×. The
top tree levels (most rows, most work) sit in the winning regime; only small deep leaves
fall into the null light-load regime, and those are cheap. So the aggregate wide-train
device-time win is real and weighted toward the expensive leaves — on top of the
unconditional ~3600× accuracy and determinism wins from spike-018.

## Disposition

CONFIRMS the spike-018 direction with a corrected magnitude (~1.3–1.7×, not 1.9×) and
removes the "artifact?" doubt: the lever is real in the regime wide-training large leaves
live in, occupancy-independent, and composes with row-partition. The wiring decision
(deferred from spike-018) is now de-risked on the measurement axis. Remaining wiring work
is unchanged: new u64 two's-complement kernel + i64 buffers (2× bytes) + oracle parity
re-pin (def-f8u-01) + overflow guard for extreme leaves; APU-measured (option-ii proxy),
discrete-gfx110x confirmation still ideal.

Evidence: `examples/gpu_int_vs_f32_psweep.rs` (4 occupancy×rows configs × P-sweep).
