---
spike: 020
name: perwarp-replication-on-u64
type: standard
validates: "Given the LIVE u64 fixed-point resident BUILD kernel, when each 32-lane wave gets its own LDS sub-histogram replica (R8) instead of 256 threads sharing one, then per-cube atomic contention drops further and BUILD device-time falls OVER the non-replicated u64 baseline — or fixed-point already took the contention win (NULL)"
verdict: PARTIAL
related: [017, 018, 019]
tags: [performance, gpu, rocm, histogram, lds, replication, fixed-point, integer-atomics, contention, composition, device-time-proxy, wide-shape, conditional-win]
---

# Spike 020: Per-warp LDS replication ON TOP of the u64 fixed-point BUILD

## What This Validates

Given the LIVE `construct_leaf_hist_resident_lds_kernel_u64` (Phase 11, u64 two's-complement
fixed-point @ S=2³⁰ in `SharedMemory<Atomic<u64>>`), when each 32-lane wave gets its own LDS
sub-histogram replica (the spike-017 lever: R = waves/cube), then does per-cube atomic
contention drop *further* and device-time fall **over the non-replicated u64 baseline** — or
has the f32→u64 fixed-point switch (spike-018/019) already captured the contention win, making
replication redundant?

This is the composition spike-018's follow-up flagged ("orthogonal and complementary") but never
actually measured together. Spike-017 measured replication on the **f32** kernel (~1.1×, kept as
evidence, not wired); this measures it on the **u64** kernel that has since shipped.

## How to Run

```bash
# CPU-only stub compiles (default build unaffected):
cargo build -p lgbm-compute --example gpu_u64_lds_replication_ab
# rocm build:
cargo build -p lgbm-compute --features rocm --example gpu_u64_lds_replication_ab
# Device-time A/B — run the WHOLE PROCESS >=2x, check the SIGN is stable:
cargo run --release -p lgbm-compute --features rocm --example gpu_u64_lds_replication_ab
```

Example file: `crates/lgbm-compute/examples/gpu_u64_lds_replication_ab.rs` (rocm-gated kernel +
A/B harness; CPU-only stub `main`). Example-only — the live kernel + CPU f64 anchor are untouched
(`git diff --stat -- crates/lgbm-compute/src/` empty).

## Method (spike-017 replication × spike-018b/019 measurement discipline)

- Candidate = the u64 kernel with `#[comptime] replicas` and per-warp indexing
  `replica = (UNIT_POS / PLANE_DIM) % nrep`, deterministic ascending replica-merge → one global
  atomic per cell. R=1 reduces byte-identically to the live non-replicated kernel.
- Baseline (R1) vs candidate (R8) share the IDENTICAL `build_rp` layout
  (`CubeCount=(num_features,P)`, `CubeDim=256`), differing ONLY in `replicas` ⇒ the ratio isolates
  replication.
- Compute-throughput timing: ~20 launches accumulated into ONE reused buffer, single readback;
  REPS=9 interleaved, `median[p25..p75]`; SEP-WIN iff `R8_p75 < R1_p25` else overlap; ≥2 process
  runs for sign-stability.
- Regimes: HEAVY wide 16×1M (contention bites) + LIGHT 16×200k (control). R ∈ {1,2,4,8}; P ∈ {1,16}.
- Spoofed 8-CU gfx1152 APU on shared DDR5 ⇒ absolute Mr/s is APU-confounded; judge the SIGN.

## Results — VERDICT: ⚠️ PARTIAL (MODEST, NULL-leaning — conditional on P=16, regresses at production P=1)

**The headline:** R8 SEP-WINs **~1.17–1.20×** at **P=16** (both regimes, both process runs —
sign-stable). But at **P=1 the win evaporates and inverts**: HEAVY-wide R8 **regresses to ~0.90×**
(overlap, never SEP, both runs). So replication does NOT add a contention win at the
**production-relevant P=1 wide regime** — there the u64 fixed-point switch already took it, and the
2× LDS / halved occupancy of R8 makes it a net loss.

### Captured A/B — both process runs (median[p25..p75] ms, ratio = R1/R8)

| Regime | P | R | Run1 R1 ms | Run1 R8 ms | Run1 R1/R8 | Run1 | Run2 R1 ms | Run2 R8 ms | Run2 R1/R8 | Run2 |
|--------|---|---|------------|------------|------------|------|------------|------------|------------|------|
| HEAVY 16×1M | 1 | 8 | 492[484..498] | 542[527..547] | **0.91×** | overlap (regress) | 487[478..489] | 541[531..591] | **0.90×** | overlap (regress) |
| HEAVY 16×1M | 16 | 8 | 399[392..405] | 333[333..338] | **1.20×** | **SEP-WIN** | 385[384..387] | 331[329..334] | **1.17×** | **SEP-WIN** |
| LIGHT 16×200k | 1 | 8 | 88[87..89] | 85[85..86] | 1.04× | SEP* | 89[87..89] | 85[83..87] | 1.05× | overlap |
| LIGHT 16×200k | 16 | 8 | 71[70..71] | 60[60..61] | **1.19×** | **SEP-WIN** | 72[71..74] | 61[60..62] | **1.19×** | **SEP-WIN** |

\* LIGHT/P=1 SEP flickers to overlap in run 2 → noise, treat as null.

### Parity (HARD gate)
- **R8 == R1 BIT-EXACT** (raw u64 cells) in every regime × P, both runs. Integer replica-merge is
  order-independent ⇒ exact, as required. No regression.
- CPU integer-reference sanity (feature 0): `max_unit_gap ≈ 6.5e-9` (HEAVY) / `2.79e-9` (LIGHT) in
  grad units — a host-vs-device f32-quantize-multiply ULP artifact, not a structural bug
  (`cpu_ref ok=true`). Correctness confirmed beyond just R1==R8.

### R2/R4 vs R8 — all-or-nothing at the warp boundary (spike-017 reproduced under u64)
R2/R4 are flat/overlap (~0.99–1.05×, no stable SEP) everywhere; only **R8** (replicas == 8 waves,
so each wave32 gets a private sub-hist) produces a stable SEP. The win is a STEP at R=num_waves,
not a gradient — exactly spike-017's f32 finding, now confirmed for u64.

### LDS budget / occupancy
R8 LDS/cube = `8 × HIST_LDS_MAX(512) × 8 B = 32 KB ≤ 64 KB`. Fits, but is 2× spike-017's f32 16 KB
and **halves the LDS-limited cubes/CU vs the R1 4 KB sub-hist**. That occupancy cost is exactly the
mechanism: at P=1 (few cubes, no row-partition) the halved occupancy is unrelieved ⇒ regress; at
P=16 (row-partition already supplies cubes) the per-replica contention drop (256→32 threads/atomic)
dominates ⇒ win. R16 = 64 KB (occupancy 1) — not swept, would over-pressure.

## Investigation Trail

1. Built the u64 replicated kernel by grafting spike-017's comptime per-warp-replica structure onto
   the live u64 kernel body (kept the S=2³⁰ quantize). R=1 verified byte-behaviour-identical to the
   live kernel via R8==R1 bit-exact at replicas=1 trivially and the baseline path.
2. First HEAVY run showed P=16 SEP-WIN but P=1 *regression* — the inverse of spike-018/019's
   fixed-point pattern (which won AT P=1 and improved with P). Re-ran a 2nd process to confirm the
   sign: P=16 win and P=1 regress BOTH held.
3. Swept R2/R4 to test whether the win was gradual or boundary-stepped → confirmed boundary
   (only R8), matching spike-017 → the contention relieved is inter-wave LDS-atomic serialization,
   not collision count.
4. Cross-checked the mechanism against the LDS budget: 32 KB/cube at R8 halves occupancy, which
   explains why the lever needs P=16's externally-supplied occupancy to pay off and regresses without it.

## Disposition

**DO NOT WIRE.** The R8 win is real and sign-stable but **conditional on row-partition P=16 being
active** — and the production wide regime (×500 features) runs at **P=1** (`target_cubes/500 → 1`),
which is exactly where R8 **regresses ~0.90×**. So on the path that matters, this lever is a loss.
The win it does show (P=16, ~1.18×) lands only on shapes that already row-partition (large leaves /
fewer features), where the u64 fixed-point win (spike-018/019) is already collecting most of the
benefit. Net: the f32→u64 fixed-point switch **already captured the contention win** at the
production regime; per-warp replication on top is a conditional, occupancy-gated extra that does not
generalize to wide P=1.

Kept as **rocm-gated evidence** (`examples/gpu_u64_lds_replication_ab.rs`). Revisit only if (a)
discrete gfx110x silicon becomes the perf target (the APU's 8-CU occupancy confounds every cell),
AND (b) a row-partition policy lands that keeps P>1 at wide — only then would the P=16 win be
reachable in production. Until both hold, this is closed as MODEST/NULL-leaning.

## Sources / prior art
- spike-017 (per-warp f32 replication, ~1.1×, R8 boundary) — the lever this composes.
- spike-018/019 (u64 fixed-point ~1.3–1.7×, contention-regime, composes with P) — the baseline this
  sits on top of and which already took the P=1 contention win.
