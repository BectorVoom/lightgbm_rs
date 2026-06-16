# CPU Histogram Build — the shipped R3/R4 speed campaign

Implementation blueprint from spikes 002–005. The histogram BUILD is the CPU train
bottleneck; these four spikes localized it and stacked four bit-exact wins.

## Requirements

- Every change is **bit-exact to C++** — gate with `cargo test -p lgbm-compute --lib`,
  `-p lgbm-treelearner`, `-p oracle-harness` (`learner_parity` 29/0). f64 fold order frozen.
- **A/B both scales** (small ~2k and large ~200k) before shipping — several levers help
  one scale and regress the other.
- **Measure wall-clock, not instruction counts** (the p0n lesson: callgrind on a
  cubecl-cpu binary is dominated by bundled-LLVM static init — useless for attribution).

## The diagnosis (spike 002)

Per-phase A/B of Rust vs `lib_lightgbm` 4.6 (built `-DUSE_TIMETAG`) at 2k×12, single
thread, localized the low-row gap: **histogram build is 187.5µs of the ~188µs/iter gap**
(Rust 232µs vs C++ 44.5µs = 5.2×); split-scan (1.4×) and partition (2.1×) are near parity.
Tooling kept in-tree: env-gated `lgbm_treelearner::phase_prof` (`LGBM_PHASE_PROF=1`,
zero-overhead off), `[profile.profiling]`, `bench_crossover.rs`. Reproduce the C++ side
with `sources/002-lowrow-phase-ab/gen_data.py` + `train.conf`.

## The four shipped wins (apply in this order — each builds on the last)

1. **Once-per-leaf grad/hess gather (spike 003, R3).** The CPU build re-gathered
   `ord_g`/`ord_h` *inside* the per-feature loop, but they're identical across features —
   only the bin column differs. Hoist the grad/hess gather to once per leaf; re-gather
   only `ord_bins` per feature. ~4 lines, byte-identical fold ⇒ bit-exact. **build −33%
   small / −39% large; train −16–18% / −32–33%.** (Distinct from the disproven p0n
   alloc-churn lever — the cost was redundant gather *memory traffic*, not allocation.)

2. **Fused branchless build-from-column (spike 003b/r4o).** Fold the bin directly from
   the column into reused hot scratch (no `ord_bins` materialization). Wins big — **but
   ONLY branchless**: any per-element `bin < num_bin` check serializes the loop and
   regresses large +3–8%. Unlock by **relocating bin validation to a once-per-train
   upstream check** (validate each column's range when `with_features` is set; the hot
   fold then trusts the invariant, like C++ `dense_bin.hpp`). train −17% small / −6.6%
   large on top of 003.

3. **Narrow bin columns u32→u16→u8 (spike 004, R3).** Store each `FeatureColumn`'s bins
   in the narrowest unsigned type for its `num_bin` (u8 ≤256, u16 ≤65536, else u32) — an
   enum `BinColumn{U8,U16,U32}` with a widening `bin(row)->u32` accessor (no memory
   doubling, faithful to C++ `DenseBin<uintN_t>`). Isolated gather+fold **−58% (u8)**: the
   u32 column (781KB/feat) overflows L2, u8 (195KB) fits → the random `bins[leaf_rows[i]]`
   gather hits L2. Full impl gave **large train −49%** (2.74→1.40s). Small: neutral
   (already L1-resident). Default `max_bin=255` ⇒ u8 covers the common case.

4. **Feature-parallel build (spike 005, R4).** Above a leaf-size threshold, rayon
   `into_par_iter` over features (each folds its own histogram from shared read-only
   ord_g/ord_h, sequential copy into `out`). Bit-exact (per-feature fold order unchanged,
   disjoint outputs → thread-count-independent). **Gate at `leaf_rows ≥ 16384`
   (`LGBM_PAR_THRESHOLD`)**: unconditional parallel regresses small 5× (rayon dispatch >
   tiny folds); crossover ≈12k rows. Large train **−26%** at 16 cores, zero small/medium
   regression. NOTE: this makes the anchor multi-threaded — the 1-core-vs-C++ basis no
   longer applies at large, and it pushes the spike-001 GPU crossover far up.

   ⚠️ This per-feature `Vec<Vec<f64>>` intermediate is the one spike 011 (see
   histogram-learning-memory-layout.md) proved **load-bearing** — do NOT "flatten" it
   into a shared-buffer scatter.

## What to Avoid

- Chasing split-scan or partition for low rows (already ~1.4–2.1×, <20% of the gap).
- Allocation-churn micro-opts as the *primary* lever (p0n: no low-row win, −9% large).
  The win was memory traffic (gather) and cache density (bin width), not malloc.
- Per-element safety checks in the hot fold (serializes; relocate to once-per-train).
- Trusting instruction-count profilers on cubecl-cpu binaries.

## Constraints

- Wins concentrate at **large rows** (build is ~90% of large train); small is alloc-noise.
- Harnesses (in-crate): `bench_crossover.rs`, `bin_width_microbench.rs`, `phase_prof`.

## Origin

Spikes 002, 003 (+003b/r4o), 004, 005 — all VALIDATED + SHIPPED. Sources in
`sources/002-*`…`sources/005-*`. Part of perf-gap-vs-cpp-40-80x R3+R4.
