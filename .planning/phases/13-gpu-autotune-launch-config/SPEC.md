# Phase 13 — Autotuned GPU launch-config selection (`cubecl::tune`)

**Status:** scoping (spike-validated, ready to plan)
**Origin:** spikes 037–040 (`autotune-*`). Continues the `09/11/12` GPU perf line; acts on
the latent `row_partition_count` mis-tune that spike-040 surfaced.

## Goal

Make **CubeCL runtime autotuning** the **default GPU launch-config selector** (rocm),
replacing the hand-tuned/env heuristics for both GPU launch knobs — histogram-build
row-partition `P` and split-scan `CubeDim` `W`. Autotune benchmarks the candidate variants
once per occupancy regime, caches the winner (in-process + persistent disk), and runs it on
every subsequent launch — beating the shipped heuristic **~10%** on the 8-CU APU and
self-calibrating on any future GPU with zero re-tuning, all within the ~1e-6 ROCm parity
contract vs the CPU f64 anchor.

## Why (spike evidence)

- **Feasible end-to-end on cubecl-hip 0.10** (spike-037): compile / run-on-device /
  benchmark-both / pick-winner / in-proc cache ~6µs / persistent disk cache across processes.
  Independently re-derived spike-007's P=16. The `cubecl_manual` doc is wrong on 3 load-bearing
  points — code from the source.
- **Correctness-safe with the right generator** (spike-038): the accumulating build kernel
  corrupts 27× under `CloneInputGenerator`; a **fresh-output `InputGenerator`** restores
  `rel_err 0`. OVERWRITE kernels (scan) are safe as-is.
- **Cache amortizes with the right key** (spike-039): exact-`rows` keying is a per-leaf tuning
  storm (975ms/tree); `log2(rows)` bucketing → ~3× fewer tunes, keeps the per-regime crossover.
- **Beats the heuristic ~10%, never loses** (spike-040, 3 restarts sign-stable): the shipped
  `row_partition_count(50,n)` under-partitions to **P=1** (the slowest sweep point) at the
  production width; autotune picks P∈{4,8,16}.

## Scope (what to build)

1. **Autotune the histogram-build `P`** in the resident-build launch path. Wrap the build
   kernel at a `PSET` (e.g. {1,4,8,16,32}) as `Tunable`s; key on `(log2(rows), num_features,
   num_bins)`; **fresh-output `InputGenerator`** (accumulating kernel). Replace the
   `row_partition_count` call at the resident launch sites with the autotuned pick;
   keep `row_partition_count` as the documented cold-start/fallback bound.
2. **Autotune the split-scan `CubeDim` W** in the scan launch path. Wrap at a `WSET`
   (e.g. {32,64,128,256}); `CloneInputGenerator` ok (scan overwrites). Replace the
   `LGBM_SCAN_CUBEDIM` default with the autotuned pick.
3. **Default-on rocm discriminator** — a backend seam (cf `prefers_host_partition`, 035) that
   routes GPU launch-config selection through the tuner by default; an off-switch
   (`LGBM_AUTOTUNE=0`) falls back to the heuristic.
4. **`serde` as a real dep** of the crate defining the `AutotuneKey` (was dev-only in spikes).
5. **Static `LocalTuner`s + AutotuneKey types**, cache namespaced by the device id.

## Hard gates / constraints

- **Parity holds for EVERY PSET/WSET variant**, pinned to the CPU f64 anchor within ~1e-6
  (def-f8u-01 — never GPU-vs-GPU). Because autotune may pick any variant at runtime, the
  oracle parity tests must cover the whole variant set, not just the current default.
  - VERIFY: the **u64 fixed-point build** (phase 11) is order-independent ⇒ `P` is
    **bit-identical across P** there (clean); the **f32 build** is NOT (P≥2 → ~2e-5 rel,
    still in-gate). Scan feature-per-lane is bit-exact across `W`. Establish the autotuned
    default's build path and gate to its parity class.
- **CPU f64 anchor UNTOUCHED** (bit-exact merge gate): `cargo test -p lgbm-treelearner --lib`
  + `-p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`) stay green.
- **CPU routing unchanged** — this is rocm-only launch-config selection; the GPU still loses
  to the 16-core CPU end-to-end here (the win is local-relative + portability).
- Feature-gated to `rocm`; the CPU-only build pulls no autotune/serde-derive codegen on the
  hot path (serde dep is fine; gate the tuner statics behind `rocm`).
- Measurement is the **device-time proxy** (8-CU APU; wall-clock confounded). Report the SIGN
  (autotune ≥ heuristic) + selection, not absolute Mr/s; ≥2 process restarts.

## Out of scope

- Recalibrating `row_partition_count` as a standalone fix (the user chose autotune over the
  cheap heuristic fix; the heuristic remains only as the fallback bound).
- Autotuning structural choices (sibling co-pack, host-vs-device partition) — not launch-dim
  sweeps; their existing gates (024/035) stay.
- Changing CPU routing or making GPU the default training backend.
- Discrete-GPU wall-clock validation of the portability claim (no hardware).

## Success criteria

- Autotune is the default rocm selector for both `P` and `W`; `LGBM_AUTOTUNE=0` falls back
  to the heuristic and reproduces prior behavior.
- An e2e A/B (`bench_gpu_vs_cpu` or equivalent) shows autotune ≥ the heuristic at the
  production width (sign-stable, ≥2 restarts) — i.e. it does NOT regress, and recovers the
  ~10% P-under-partition spike-040 measured.
- Oracle rocm parity green for every PSET/WSET variant, pinned to the CPU f64 anchor (~1e-6);
  CPU merge gate green.
- First-tune cost is bounded and documented (synchronous cold tune per new key; warm hits ~µs).

## Evidence / reference

`.claude/skills/spike-findings-lightgbm_rs/references/gpu-kernel-autotuning.md` (blueprint),
`.planning/spikes/037..040/README.md`, examples `spike037..040_*.rs`.
