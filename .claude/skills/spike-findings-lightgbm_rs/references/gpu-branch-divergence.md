# GPU Branch / Warp Divergence

Blueprint from spike-036 (the "optimize conditional branching in GPU kernels" gate).
**TL;DR: do not chase branch divergence as a general lever — the kernels are already
branchless and the only live divergence is off the dominant path. BUT divergence is the one
GPU effect the spoofed APU measures faithfully, so when you DO have a divergent hot path, you
can sign-judge it.**

## Requirements

- The CPU f64 anchor stays **bit-exact** — `split_scan_body` is the SHARED single source of
  the split math (CPU anchor + GPU). Any divergence experiment that changes its control flow
  MUST be a **separate hip-only kernel fork**, never an edit to `split_scan_body`.
- GPU numbers are **sign-only** (spoofed 8-CU gfx1152 APU). Divergence is the *exception* that
  is cleanly measurable — but the magnitude is still APU-confounded; judge the ladder SHAPE.

## How to Build It (the measurability ladder — reuse before any divergence A/B)

Before A/B-ing a real divergent kernel on this hardware, first confirm the hardware resolves
divergence at the magnitude you care about. The probe: N arms doing **identical total work**,
differing only in how the loop trip count is **distributed across wavefront lanes**.

```rust
// crates/lgbm-compute/examples/spike036_divergence_measurability.rs
#[cube(launch_unchecked)]
fn diverge(counts: &Array<u32>, out: &mut Array<f32>, w: f32) {
    let lane = UNIT_POS as usize;       // within-cube lane
    let gpos = ABSOLUTE_POS;            // unique global lane id (usize)
    let n = counts[lane];              // data-dependent trip count (drives divergence)
    let mut acc = 0.0f32;              // loop-carried mutable MUST init from a literal
    let mut i = 0u32;
    while i < n {                      // wave runs to its SLOWEST active lane
        acc = acc * 1.0000001f32 + w; // constant ALU/iter, w runtime ⇒ no closed-form fold
        i += 1u32;
    }
    out[gpos] = acc;                  // register sink written once ⇒ defeats DCE
}
```

- **Patterns** (each sums to `CUBE_DIM*K` ⇒ identical work): `UNIFORM=K`, `DIV2=lane%2?0:2K`,
  `DIV4=lane%4?0:4K`, `DIV32=lane%32?0:32K`. Interleaving `lane%n` keeps the imbalance
  *within* any contiguous 32- or 64-lane group ⇒ robust to wave32 vs wave64.
- **Timing:** accumulate `LAUNCHES≈20` into one reused `out` buffer, force sync with a final
  `read_one_unchecked`, interleaved median + p25/p75 over ~11 reps, ≥2 process restarts
  (the CONVENTIONS GPU-A/B discipline). Report each arm's ratio vs UNIFORM and a SEP test
  (rung p25 > UNIFORM p75).
- **Read:** ratios ≈ 1:2:4:32 ⇒ divergence resolvable. Collapse →1 ⇒ not resolvable.

**Measured result (2 restarts, near-ideal & stable):**

| arm | run 1 | run 2 | ideal |
|---|---|---|---|
| UNIFORM | 1.00× | 1.00× | 1 |
| DIV2 | 1.95× | 1.89× | 2 |
| DIV4 | 3.84× | 3.62× | 4 |
| DIV32 | 29.3× | 25.6× | 32 |

⇒ **Wavefront lockstep-masking is faithful on the spoofed APU.** Divergence is a
scheduler property (not CU-count / memory-bound — the axes the APU confounds), so it is the
ONE GPU micro-arch effect that survives the spoof and is cleanly sign-measurable. **Reusable
carve-out from the "APU numbers are unmeasurable" caveat.**

## What to Avoid (the don't-chase, with reasons)

- **Don't "convert branches to branchless" — it's already done.** `split_scan_body` and
  `data_partition_kernel` are fully `select(cond,new,old)`-encoded already (cubecl-cpu's MLIR
  lowering rejects in-loop conditional-store `if` chains, so the divergence-elimination
  transform shipped as a side-effect). There is no branch-store divergence left to remove.
- **Don't optimize divergence on the dominant path — it has none.** The wide **build** (the
  030/034 bottleneck) is a `while k < r` grid-stride loop → uniform trip count → **zero
  divergence by construction**. Branch work cannot touch it.
- **The only live data-dependent cross-lane divergence** is the split-scan **loop-trip-count**
  imbalance (lane `f` scans `0..max(rev,fwd)_count[f]`; a W=64 wave runs at the max num_bin).
  It is **zero on the production all-256-bin shape** (identical counts ⇒ identical waves) and
  sits on the **3–7% scan phase** (034). Real only on **mixed-cardinality** feature sets;
  honest e2e ceiling **≪1%**.
- **`done`-predication is intra-lane, not cross-lane.** Replacing the sticky `done` flag with
  a real divergent `break` cuts a lane's OWN ALU but the wave still runs to its slowest lane —
  so it does NOT reduce divergence unless early-exit is *correlated* across the wave's
  features. Speculative; and it forks the bit-exact anchor. Likely don't-build (038).
- **The heavily-divergent kernel is dead.** `construct_hist_kernel_atomic_f32_plane`
  (ballot/leader-election loops) is the p93 warp-aggregation path — NULL, superseded by u64
  atomics (017/020). Not on any production path.

## Branch inventory (every `#[cube]` kernel)

| Kernel | Divergent? | On default path post-035? | Phase |
|---|---|---|---|
| `split_scan_body` / fused / co-pack siblings (split.rs) | YES — loop-trip-count (body is branchless `select`) | YES | scan (3–7%); zero at uniform cardinality |
| `data_partition_kernel` (partition.rs) | NO (branchless; `if most_freq_bin==0` is uniform) | **NO** (035 host-routes partition) | n/a |
| `construct_hist_kernel_atomic_f32_plane` | YES — heavy | **NO** (p93 NULL/dead) | n/a |
| u64 resident LDS build (`construct_leaf_hist_resident_lds_kernel_u64`) | NO — uniform grid-stride | YES | **build (dominant) — divergence-free** |
| `subtract_hist_kernel` | NO — uniform | YES | small |

## Constraints

- Spoofed 8-CU gfx1152 APU; rocprof unsupported; magnitude APU-confounded (sign/shape only).
- cubecl-cpu MLIR: loop-carried mutables init from a **literal**; in-loop conditional stores
  must be **branchless `select`** (this is *why* the scan/partition kernels are already
  branchless). casts inside a cube body use `f32::cast_from(x)`, not `x as f32`.
- `ABSOLUTE_POS` is `usize`; per-axis `UNIT_POS` is `u32` (cast before u32 arithmetic).

## Origin

Synthesized from spike: 036 (PARTIAL — measurability VALIDATED, critical-path WEAK).
Source files in: sources/036-branch-divergence-inventory-gate/.
Related: 021/024 (scan occupancy & co-pack — where the trip-count divergence lives), 030
(build is uncoalesced-gather-bound, not divergence), 034/035 (the post-co-pack attribution
that put scan at 3–7% and host-routed partition).
