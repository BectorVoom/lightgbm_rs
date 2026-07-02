# Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG - Pattern Map

**Mapped:** 2026-06-29
**Files analyzed:** 9 new/extended artifacts (3 NEW kernel modules, 1 NEW C++ harness + fixtures, 1 NEW test file, 4 EXISTING seam extensions)
**Analogs found:** 9 / 9 (every artifact has an in-repo analog — this is a strict "extend existing patterns" phase)

> **Planner orientation:** This phase is ADDITIVE, env-gated (`LGBM_CUDA_ON_DEVICE`), strict no-op seam (D-09). Every new file mirrors an existing file's structure. The seam files (#5) **already exist** — scope them as "extend the doc/test surface, do NOT flip the discriminator". Nothing here invents a new pattern; all five cubecl-0.10 gotchas already have shipped prior art in `histogram.rs`.

## File Classification

| New/Extended File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/primitives.rs` (NEW) | kernel (`#[cube]`) | transform / reduction | `crates/lgbm-compute/src/kernels/histogram.rs` | exact (same crate, same `#[cube]` + launcher + LDS conventions) |
| `crates/lgbm-compute/src/kernels/split_info.rs` (NEW) | device-struct / resident pool | batch (pre-alloc, index-by-slot) | `split.rs` `client.empty` pre-alloc + `histogram.rs` resident-pool | role-match |
| `crates/lgbm-compute/src/kernels/random.rs` (NEW) | kernel (`#[cube]`) + host launcher | request-response (seed→draw→readback) | `crates/lgbm-core/src/random.rs` (host `Random`) + histogram launcher | exact (port reference) |
| `crates/lgbm-compute/src/kernels/mod.rs` (EXTEND) | config / barrel | — | existing `mod.rs` (lines 1-19) | exact |
| `xtask/cpp/primitive_capture.cu` (NEW) + CMake target + committed fixtures | test-harness (off-build) | file-I/O (golden dump) | `xtask/cpp/kernel_capture.cpp` + `xtask/cpp/CMakeLists.txt` + `regen()` | role-match (C++→CUDA/HIP; same capture shape) |
| `crates/oracle-harness/tests/primitive_parity.rs` (NEW) | test | request-response (fixture replay) | `crates/oracle-harness/tests/rng_parity.rs` | exact (fixture-loader + skip-if-absent + bit/ULP asserts) |
| `crates/lgbm-compute/src/lib.rs` seam (EXTEND-ONLY) | service (trait) | request-response | EXISTS (`on_device_growth_supported` 1239, `grow_tree_on_device` 1272/2207) | self (frozen, D-09) |
| `crates/lgbm-treelearner/src/learner.rs` seam (EXTEND-ONLY) | service | request-response | EXISTS (`cuda_on_device_env` 443, `on_device_eligible` 488) | self (frozen) |
| `crates/lgbm-dataset/src/dataset.rs` `LeafPartitionLayout` (EXTEND-ONLY) | model (POD) | — | EXISTS (line 88) | self (frozen) |
| `crates/oracle-harness/tests/learner_parity.rs` oracle (EXTEND-ONLY) | test | — | EXISTS (`assert_on_device_tree_matches_cpu_anchor` 2166, Slice-0 tests 2422/2452) | self (frozen, must stay green) |

---

## Pattern Assignments

### `crates/lgbm-compute/src/kernels/primitives.rs` (NEW — kernel, transform/reduction)

**Analog:** `crates/lgbm-compute/src/kernels/histogram.rs` (the canonical `#[cube]` kernel + safe-launcher module in this crate).

**Imports pattern** (`histogram.rs:28-31`):
```rust
use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::runtime::ActiveRuntime;
```

**Shared generic `#[cube]` helper, thin per-type launch wrappers** (`histogram.rs:55-113`) — the SINGLE-SOURCE-OF-TRUTH idiom. Author the prefix-sum/reduction math ONCE in a generic `#[cube] fn` and emit thin `#[cube(launch)]`/`#[cube(launch_unchecked)]` wrappers per cell type (the f64 cpu-anchor vs f32 hip-mirror split):
```rust
#[cube]
fn hist_fold_body<N: Numeric>(binned: &Array<u32>, /* ... */ out: &mut Array<N>) { /* math once */ }

#[cube(launch)]
pub fn construct_hist_kernel(/* ... */ out: &mut Array<f64>) { hist_fold_body::<f64>(/* ... */); }

#[cube(launch)]
pub fn construct_hist_kernel_f32(/* ... */ out: &mut Array<f32>) { hist_fold_body::<f32>(/* ... */); }
```
→ Apply to: prefix-sum (incl/excl), reductions (sum/max/min, dotprod). Use plane intrinsics inside the body (RESEARCH Pattern 1: `plane_inclusive_sum`/`plane_exclusive_sum`/`plane_sum`/`plane_max`/`plane_min`).

**LDS block-scan staging pattern** (`histogram.rs:792-835`, the `SharedMemory` + `sync_cube()` LDS prior art for RESEARCH Pattern 2 segmented block-scan):
```rust
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn construct_hist_kernel_lds_f32(/* ... */ lds_len: u32) {
    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX); // COMPTIME-sized to max
    let cd = CUBE_DIM as usize;
    // 1. zero active LDS cells strided by UNIT_POS; sync_cube();
    // 2. scatter strided rows into LDS; sync_cube();
    // 3. merge LDS -> global (one atomic/cell)
}
```
Note the **comptime-sized-to-max LDS const** (`HIST_LDS_MAX = 512`, `histogram.rs:613-625`) driven by a runtime active-length arg — the exact pattern for the 256-bin within-feature block-scan (`N_PLANES_MAX` staging buffer).

**Safe `launch_unchecked` wrapper convention** (`histogram.rs:581-607`, also `split.rs:805-836`) — confined `unsafe` + SAFETY comment discharging the V5 host-side bounds proof (CMP-01/NRW-01):
```rust
// SAFETY: every input handle sized n / out sized out_len, each outliving the launch;
// every device index host-proven < len by V5 validation BEFORE upload; unsafe confined here (CMP-01).
unsafe {
    construct_hist_kernel_atomic_f32_plane::launch_unchecked(
        client, CubeCount::Static(cube_count, 1, 1), CubeDim::new_1d(cube_dim),
        ArrayArg::from_raw_parts(h_bin, n), /* ... */ use_plane,
    );
}
let bytes = client.read_one_unchecked(h_out);
```
Use plain checked `::launch` for the cold skeleton/percentile paths (`split.rs:814` precedent); reserve `launch_unchecked` for full-depth hot prefix-sum/reduction.

**V5 boundary validation BEFORE launch** (`histogram.rs:151-167`): validate lengths/ranges → `Result<_, ComputeError>`, then `create_from_slice` inputs, then `create_from_slice(&zeros)` for any `+=`-accumulated output (NOT `client.empty` for accumulators — see the load-bearing comment at `histogram.rs:161-167`).

**Capability-gated f64/f32 routing** (`runtime.rs:84-95` `accumulate_type()`): route the cpu-anchor (`has_f64`) to the f64 kernel and hip (`!has_f64`) to the f32 mirror — the numeric primitives that carry output follow this exact gate. For the i64 quantized accumulator, use the `Atomic<u64>` two's-complement idiom (`histogram.rs:1300-1311`, `SCALE_F32 = 2^30`), NEVER `Atomic<i64>`.

---

### `crates/lgbm-compute/src/kernels/split_info.rs` (NEW — device-struct, batch/pre-alloc)

**Analog:** the `client.empty` pre-allocation idiom (`split.rs:797-803`) + the resident-pool "allocate once, reuse across launches" discipline (`histogram.rs:161-167` comment contrasting `empty` vs zeroed).

**Pre-allocate-once-then-reuse pattern** (`split.rs:797-803` — `empty` is correct ONLY when the kernel WRITES every cell, never `+=`):
```rust
let out_len = 12usize;
// kernel WRITES (never `+=`) all cells unconditionally, so `out` needs no zero-init.
// empty() skips the host zero-alloc + upload. (Contrast accumulate buffers which MUST be zeroed.)
let h_out = client.empty(out_len * core::mem::size_of::<f64>());
```

**SoA struct = one `Handle` per field, sized `[num_leaf_slots]`, allocated once in a `new(client, num_leaf_slots)` constructor** (RESEARCH Pattern 5 / D-05/D-06). Each field is its own `client.empty(num_leaf_slots * size_of::<T>())`. Categorical buffers (`num_cat_threshold`, `cat_threshold`, `cat_threshold_real`) are pre-allocated reserved slabs now (D-06 — Phase 22 fills, not restructures). **No per-split device alloc anywhere** (D-08 — the C++ `AllocateCatVectorsKernel` anti-pattern).

**Slot-copy `operator=` analog** (deep-copy slot a→b): for Phase-14 correctness, a host-side index copy is sufficient (RESEARCH A6 / D-07 defers the device copy kernel + readback packet to the Phase-17/18 consumer). A tiny `#[cube]` "copy slot" kernel (`buf[b] = buf[a]` per field) is the preferred future form; mirror the thin-launch-wrapper convention from `histogram.rs:84-92`.

**Reserved-slab const convention** (mirror `HIST_LDS_MAX`, `histogram.rs:613-625`): define `MAX_CAT_PER_SPLIT` as a documented plain `usize` const, Phase-22-tunable (RESEARCH Open Q3).

---

### `crates/lgbm-compute/src/kernels/random.rs` (NEW — `#[cube]` LCG + host launcher)

**Analog (parity oracle, MUST reproduce bit-for-bit):** `crates/lgbm-core/src/random.rs` — the host `Random` LCG.

**The exact recurrence + draw methods to port** (`random.rs:44-77`):
```rust
fn rand_int16(&mut self) -> i32 {
    self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
    ((self.x >> 16) & 0x7FFF) as i32
}
fn rand_int32(&mut self) -> i32 {
    self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
    (self.x & 0x7FFF_FFFF) as i32
}
fn next_float(&mut self) -> f32 { (self.rand_int16() as f32) / 32768.0_f32 } // divisor EXACTLY 32768.0, f32 end-to-end
```

**Device-side wrap rule (RESEARCH Pitfall 2 / `lib.rs:1265` checklist):** in `#[cube]` use plain `u32` `*`/`+` (native hardware two's-complement wrap) — `wrapping_add`/`wrapping_mul` are NOT cube intrinsics:
```rust
#[cube]
fn cuda_rand_advance(state: &mut u32) -> u32 {
    *state = *state * 214013u32 + 2531011u32; // plain ops wrap on device
    *state
}
```

**Host launcher:** seed N tasks, draw K each, `read_one_unchecked` back — same launcher shape as `histogram.rs:176-189` (`create_from_slice` seeds → `launch` → `read_one_unchecked` → `from_bytes`).

**Security V6 negative control:** carry forward the `random.rs:8-14` module doc — deterministic non-crypto PRNG, never a security RNG.

---

### `xtask/cpp/primitive_capture.cu` + CMake target + committed fixtures (NEW — off-build golden harness)

**Analog:** `xtask/cpp/kernel_capture.cpp` (the existing self-contained C++ transcription harness) + `xtask/cpp/CMakeLists.txt` + the `regen()` driver in `xtask/src/main.rs:331-413`.

**Capture-harness conventions to mirror** (from `kernel_capture.cpp:1-45`):
- **Self-contained transcription, not a `lib_lightgbm` link** — the existing harnesses compile against pinned headers only (the LightGBM `external_libs/` submodules are empty; see project memory `lightgbm-ref-tree-untracked`). The primitive harness wraps each `__device__` helper from the AMD fork `cuda_algorithms.cu` in a one-line `__global__` shim (RESEARCH Code Examples / D-03 — they are not host-callable as-is).
- **Determinism / idempotency (D-14):** derive all synthetic inputs from one recorded `MASTER_SEED` passed on argv (header-only `LightGBM::Random`); re-running `regen` produces byte-identical fixtures (empty `git diff`).

**Build/run driver pattern** (`xtask/src/main.rs:331-413` `regen()`): add a new `regen`-style subcommand that (1) cmake-configures `xtask/cpp`, (2) builds the new target, (3) runs the exe writing committed fixtures into `crates/oracle-harness/fixtures/`, (4) refreshes `REFERENCE_MANIFEST.md`. Build with `hipcc` against `LightGBM-release-4.6.0.99/` (RESEARCH §C++/HIP harness; local APU adequate — primitives deterministic, f32 reductions held to ~1e-6).

**Fixture format:** key=value text lines (mirror `rng_sequence.txt` / `rng_parity.rs:75-83` parsing) — one record per primitive case with input seed + expected output (int permutation as `;`-list; f64 as bits or value; f32 as `to_bits()` for ~1e-6).

---

### `crates/oracle-harness/tests/primitive_parity.rs` (NEW — fixture-replay test)

**Analog:** `crates/oracle-harness/tests/rng_parity.rs` (fixture loader + skip-if-absent + bit/ULP asserts) — and for the RNG sub-case, this IS the D-04 home (host `Random` stream is the oracle, no C++ capture).

**Skip-if-fixture-absent so `cargo test` stays green pre-capture** (`rng_parity.rs:53-61`):
```rust
let Ok(text) = std::fs::read_to_string(&path) else {
    eprintln!("SKIP — fixture {} not found. Run `cargo run -p xtask -- <regen>` ...", path.display());
    return;
};
```

**Fixture path + parse helpers** (`rng_parity.rs:22-49`): `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/<name>")`, `;`-split list parsers, `key=value` field extractor.

**Assertion discipline** (`rng_parity.rs:91-123` + RESEARCH Pitfall 3 / D-03):
- Index/permutation ops (argsort) + integer prefix-sum → **bit-exact** (`assert_eq!`).
- f32 outputs → `to_bits()` equality only where exact (RNG), else ~1e-6 relative for GPU f32 reductions (Pitfall 3).
- f64 reductions/percentile → bit-exact vs cpu f64 anchor where reduction order matches, else documented ULP band (RESEARCH Open Q2).
- **NEVER GPU-vs-GPU** — pin to the cpu f64 anchor / C++ fixture (def-f8u-01, D-10).

**CUDARandom device-stream replay (D-04):** mirror `rng_parity.rs:107-115` `to_bits()` comparison, but draw from the device launcher (this file's new `random.rs`) vs `Random::new(seed)` host sequence.

**ROCm/cubecl smoke-test idiom** (for the Wave-0 plane-intrinsic probe, RESEARCH Open Q1 / Pitfall 1): `crates/lgbm-compute/tests/rocm_plane_aggregate.rs` — `#![cfg(feature = "rocm")]`, `probe_capabilities(&gc).has_plane` guard, pin GPU result to `cpu_client()` anchor within tolerance.

---

## Shared Patterns

### Capability-gated f64-anchor vs f32-mirror routing
**Source:** `crates/lgbm-compute/src/runtime.rs:84-131` (`Capabilities`, `accumulate_type()`, `probe_capabilities`)
**Apply to:** every numeric primitive that carries output (prefix-sum f64 cells, reductions). cpu (`has_f64`) → f64 kernel (bit-exact anchor); hip (`!has_f64`) → f32 mirror (~1e-6).
```rust
pub fn accumulate_type(&self) -> AccumulateType {
    if self.has_f64 { AccumulateType::F64 } else { AccumulateType::F32 }
}
```

### Safe `launch_unchecked` wrapper (CMP-01 / NRW-01)
**Source:** `crates/lgbm-compute/src/kernels/histogram.rs:581-607`, `split.rs:805-836`
**Apply to:** every host launcher in `primitives.rs` / `random.rs`. Confined `unsafe` + SAFETY comment discharging the V5 host-side bounds proof; `ArrayArg::from_raw_parts(handle, len)`; `read_one_unchecked` readback.

### V5 input validation at the Backend boundary
**Source:** `histogram.rs:144-167` (validate → `Result<_, ComputeError>` → upload)
**Apply to:** all new host launchers — never panic/UB on caller input; return `ComputeError` (`crate::error`).

### `client.empty` once, reuse / index-by-slot (pre-allocation)
**Source:** `split.rs:797-803` (`empty` only when kernel WRITES every cell), `histogram.rs:161-167` (zeroed `create_from_slice` when kernel `+=`-accumulates)
**Apply to:** the SoA split-record (D-05/D-08) and all primitive scratch (3-kernel global scan scratch sized `num_blocks`, allocated once — RESEARCH Pattern 3).

### cubecl-0.10 gotcha checklist (all five have shipped prior art)
**Source:** `lib.rs:1262-1267` checklist + `histogram.rs` implementations
**Apply to:** every `#[cube]` in this phase:
- No cross-cube barrier → 3-kernel global scan (RESEARCH Pattern 3).
- `Atomic<i64>` broken → `Atomic<u64>` two's-complement (`histogram.rs:1300-1311`).
- `wrapping_add` not an intrinsic → plain `u32` `*`/`+` (`random.rs` LCG).
- plane-sum ≤ plane width → segmented LDS block-scan (`histogram.rs:792-835`).
- `launch_unchecked` unsafe → confined wrapper (above).

### Anchor discipline — never GPU-vs-GPU
**Source:** `learner_parity.rs:2166-2222` (`assert_on_device_tree_matches_cpu_anchor`), `rocm_plane_aggregate.rs:67-96`
**Apply to:** every primitive test that carries numeric output — pin to cpu f64 anchor / C++ fixture, tie-aware where needed (D-10, def-f8u-01).

---

## No Analog Found

None. Every artifact maps to an existing in-repo pattern (this is the explicit goal of a foundation/seam phase).

---

## Seam Extension Scope (files that ALREADY exist — DO NOT rebuild)

The planner must scope these as **extend doc/test surface only**; flipping any discriminator breaks the merge gate (D-09, RESEARCH Pitfall 6):

| File | Symbol (line) | Frozen invariant |
|------|---------------|------------------|
| `crates/lgbm-compute/src/lib.rs` | `on_device_growth_supported` (1239), `grow_tree_on_device` (1272 trait default, 2207 `GpuBackend<R>` explicit override) | stays `false` / `Ok(None)` in Slice 0 |
| `crates/lgbm-treelearner/src/learner.rs` | `cuda_on_device_env` (443), `on_device_eligible` (488) | `LGBM_CUDA_ON_DEVICE` only `"1"`; AND-gate keeps host path byte-unchanged when unset |
| `crates/lgbm-dataset/src/dataset.rs` | `LeafPartitionLayout` (88) | POD; no dep on lgbm-treelearner/compute (crate-cycle guard) |
| `crates/oracle-harness/tests/learner_parity.rs` | `assert_on_device_tree_matches_cpu_anchor` (2166), `..._noop_slice0` (2452), `..._oracle_host_fallback_slice0` (2422) | both Slice-0 tests MUST stay green |

---

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/{kernels,runtime,lib}.rs`, `crates/lgbm-core/src/random.rs`, `crates/lgbm-treelearner/src/learner.rs`, `crates/lgbm-dataset/src/dataset.rs`, `crates/oracle-harness/tests/`, `crates/lgbm-compute/tests/`, `xtask/{src,cpp}/`
**Files scanned:** ~16 (full reads of the 9 analog sources + supporting structure greps)
**Pattern extraction date:** 2026-06-29
</content>
</invoke>
