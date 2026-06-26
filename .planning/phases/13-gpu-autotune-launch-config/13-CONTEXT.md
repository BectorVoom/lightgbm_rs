# Phase 13: gpu-autotune-launch-config - Context

**Gathered:** 2026-06-26
**Status:** Ready for planning
**Source:** Spike campaign 037–040 (autotune feasibility / correctness / key-granularity / vs-heuristic). The spike process WAS the context/research — this phase wires the validated lever. Plus a user scope decision (see Implementation Decisions).

<domain>
## Phase Boundary

Replace the hand-tuned / env-var GPU launch-config heuristics with **CubeCL runtime
autotuning** (`cubecl::tune`), **default-on for all GPU (rocm) launch-config selection**.
Two launch knobs come under autotune control:

1. **Histogram-BUILD row-partition `P`** — currently `row_partition_count(num_features,
   leaf_rows)` (`histogram.rs:744`, called at the resident-build launch sites ~982/1543/
   1692/1791). Spike-040 found it under-partitions to **P=1 at the production 50-feature
   width** (~10% slow on the 8-CU APU).
2. **Split-SCAN `CubeDim` W** — currently the `LGBM_SCAN_CUBEDIM` env (default 64,
   spike-021 feature-per-lane).

Delivers: autotune picks the measured-fastest variant per occupancy regime, cached, on
every GPU — beating the heuristic ~10% locally and self-calibrating on any future GPU
(discrete gfx110x / NVIDIA) with zero re-tuning. **CPU f64 anchor UNTOUCHED; CPU routing
unchanged; stays within the ~1e-6 ROCm parity gate.**
</domain>

<decisions>
## Implementation Decisions (LOCKED — user + spike-validated)

### Scope (user decision, 2026-06-26)
- **Autotune wiring**, not a heuristic recalibration (user picked "Autotune wiring (as asked)").
- **Default-on for ALL GPU selection** ("All gpu use autotune") — autotune REPLACES the
  heuristics as the default rocm selection path. NOT an opt-in default-false gate. The old
  heuristic (`row_partition_count` / `LGBM_SCAN_CUBEDIM`) becomes only the cold-start /
  cache-miss fallback bound, not the steady-state selector.
- **Both** GPU launch knobs: histogram-build `P` AND split-scan `CubeDim`.

### Autotune wiring (LOCKED — spike-validated, see gpu-kernel-autotuning.md)
- **Code from the SOURCE, not the `cubecl_manual` doc** (it is wrong on 3 points, spike-037):
  (1) `TunableSet::new`'s 1st closure is the KeyGenerator → returns the `AutotuneKey` (not a
  String); (2) `LocalTuner::execute(id, …)`'s 1st arg is the cache-namespace ID (e.g. the
  device id), NOT the key; (3) the `AutotuneKey` trait alias requires `serde::{Serialize,
  DeserializeOwned}` under `std_io` (always on linux) ⇒ a `serde` dep is mandatory in the
  crate that defines the key.
- **Fresh-output `InputGenerator` for the BUILD tunable** (spike-038) — the build kernel
  ACCUMULATES (`fetch_add`); `CloneInputGenerator` shares the real `out` handle ⇒ 27×
  corruption. The split-scan kernel OVERWRITES slots ⇒ `CloneInputGenerator` is safe there
  (the 038 OVERWRITE/ACCUMULATE/RMW classification governs each tunable).
- **`log2(rows)` occupancy-regime AutotuneKey** (spike-039) — exact-`rows` keying = a per-leaf
  tuning STORM (every node a cold ~40ms tune). Key on `(log2(rows) | size-band, num_features,
  num_bins)`; the variant choice tracks the occupancy regime, not the exact count.
- **Read the winner from the persisted cache** (`target/autotune/0.10.0/<device>/*.json.log`,
  `fastest_index` → PSET[idx]); persistent disk cache works across processes (spike-037).

### Parity (LOCKED — hard gate)
- The parity gate must hold for **EVERY variant in each PSET**, pinned to the **CPU f64
  anchor** (never GPU-vs-GPU, def-f8u-01) — because autotune may pick any variant at runtime.
- **VERIFY the production resident-build path:** the u64 fixed-point build
  (`construct_leaf_hist_resident_lds_kernel_u64`, phase 11) is order-independent ⇒ `P` is
  **parity-neutral (bit-identical across P)** on that path; the f32 path
  (`construct_leaf_hist_resident_lds_kernel`) is NOT (spike-007: P≥2 widens GPU-vs-CPU f32
  divergence to ~2e-5 rel, still inside the ~1e-6-best-effort gate). The split-scan
  feature-per-lane kernel is bit-exact across `CubeDim` W (each feature stays sequential,
  spike-021/022). Establish which build path is the autotuned default and gate accordingly.

### Claude's Discretion
- The exact `PSET` for each knob (e.g. P ∈ {1,4,8,16,32}, W ∈ {32,64,128,256}) and whether to
  derive bounds from `row_partition_count` / CU count rather than hardcode.
- AutotuneKey struct shape + size-band bucketing function; where the key/tuner statics live.
- Backend-discriminator mechanics for "default-on rocm" (a trait method that returns the
  autotuned P/W vs the heuristic; keep the heuristic callable as the documented fallback).
- First-tune latency handling (synchronous cold tune on first call per key) and any warm-up.
- Whether to retire `LGBM_SCAN_CUBEDIM` / `LGBM_ROWPART_TARGET_CUBES` envs or keep them as
  override/disable escape hatches (an off-switch like `LGBM_AUTOTUNE=0` is reasonable).
- Test/bench harness shape for the parity re-pin (all-variants) + the e2e A/B vs the heuristic.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The validated blueprint (READ FIRST)
- `.claude/skills/spike-findings-lightgbm_rs/references/gpu-kernel-autotuning.md` — the
  implementation blueprint (real 0.10 API, fresh-output generator, log2 key, measure-don't-model).
- `.planning/spikes/037-autotune-hip-feasibility/README.md` — feasibility + the 3 manual-API corrections.
- `.planning/spikes/038-autotune-inplace-correctness/README.md` — the fresh-output InputGenerator + kernel-safety classification.
- `.planning/spikes/039-autotune-key-cache-thrash/README.md` — the `log2(rows)` key + thrash data.
- `.planning/spikes/040-autotune-vs-heuristic/README.md` — autotune beats the heuristic ~10%; the latent `row_partition_count` mis-tune.

### Working spike harnesses (proven, compiling against the real kernels)
- `crates/lgbm-compute/examples/spike037_autotune_hip_feasibility.rs` — minimal LocalTuner/TunableSet on hip.
- `crates/lgbm-compute/examples/spike038_autotune_inplace_correctness.rs` — `FreshOutGenerator` impl.
- `crates/lgbm-compute/examples/spike039_autotune_key_cache_thrash.rs` — log2 keying harness.
- `crates/lgbm-compute/examples/spike040_autotune_vs_heuristic.rs` — P-sweep + read-winner-from-cache.

### Production code to modify
- `crates/lgbm-compute/src/kernels/histogram.rs` — `row_partition_count` (744), the resident
  build launchers (~982/1543/1692/1791), the u64 build kernel (1251) + f32 (1179).
- The split-scan launcher (`LGBM_SCAN_CUBEDIM` consumer; spike-021 feature-per-lane scan).
- `crates/lgbm-compute/src/runtime.rs` — `rocm_client` / `RocmRuntime` (device id for the cache ID).
- `crates/lgbm-compute/Cargo.toml` — add `serde` (derive) as a real dep (was dev-only in the spikes).
- The rocm Backend trait impl (the default-on discriminator seam; cf `prefers_host_partition` 035).

### Parity gates (hard)
- oracle-harness `kernel_parity` / `learner_parity` (rocm) — pin EVERY PSET variant to the CPU
  f64 anchor within ~1e-6; def-f8u-01 (never GPU-vs-GPU).
- CPU merge gate: `cargo test -p lgbm-treelearner --lib` + `-p oracle-harness` (esp.
  `raw_bin_train_matches_cpp_golden`) — must stay green (CPU anchor untouched).
</canonical_refs>

<specifics>
## Specific Ideas

- The autotune SELECTION is the spoof-robust axis (relative within-device) — it re-derived
  spike-007's P=16 and beat the heuristic on the confounded 8-CU APU. So the parity/selection
  logic is trustworthy here even though absolute wall-clock is APU-confounded.
- Honest bound (state in the SUMMARY): ~10% is on the GPU build, which the 16-core CPU beats
  end-to-end on this hardware — the durable deliverable is the METHOD (measure-don't-model) +
  portability to real discrete GPUs, not a local e2e train-time win.
- Pre-warming / shipping `autotune_cache.json` alongside a binary is a documented CubeCL option
  for instant deployment (spike-037 §cache) — note as a follow-on, not this phase.
</specifics>

<deferred>
## Deferred Ideas

- Autotuning structural choices (sibling co-pack on/off, host-vs-device partition) — those are
  not launch-dim sweeps; keep their existing gates (035/024).
- Discrete-gfx110x / NVIDIA wall-clock validation of the portability claim (no hardware here).
- Retiring the heuristic functions entirely (keep them as the documented cold-start fallback).
</deferred>

---

*Phase: 13-gpu-autotune-launch-config*
*Context gathered: 2026-06-26 via spike campaign 037–040 + user scope decision*
