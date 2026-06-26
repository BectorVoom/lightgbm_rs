---
spike: 037
name: autotune-hip-feasibility
type: standard
validates: "Given cubecl-hip 0.10, when a LocalTuner/TunableSet wraps two CubeDim/P variants of a production-shaped histogram-build kernel, then it compiles, runs on the AMD device, benchmarks both, caches a winner, and the cache persists across processes"
verdict: VALIDATED
related: [007, 021, 038, 039, 040]
tags: [gpu, rocm, autotune, cubecl, tune, feasibility, kill-question]
---

# Spike 037: Autotune-on-HIP Feasibility (the kill question)

## What This Validates

Given cubecl-hip 0.10 (`cubecl::hip::HipRuntime`, our live ROCm backend), when a
`LocalTuner` + `TunableSet` wraps two launch configs of a faithful row-partitioned
LDS-atomic histogram-build kernel (`build_rp` at `P=1` == production one-cube-per-feature,
vs `P=16` == the spike-007 occupancy win), then the whole autotune pipeline **compiles,
runs on the AMD device, benchmarks both variants, selects the fastest, and caches it**.

This is the KILL question for the entire idea: the autotune manual
(`cubecl_manual/manual/cubecl/12_autotuning.md`) only ever instantiates the tuner on
`CpuRuntime`. If `cubecl::tune` didn't work on `hip`, every downstream spike is moot.

## Research

`cubecl 0.10` re-exports `cubecl_runtime::tune` as **`cubecl::tune`** (`cubecl-core`
`lib.rs:50`). The module is real and backend-agnostic (it drives benchmarking through the
generic `ComputeClient`). Persistent caching is gated by the `std_io` cfg
(`std + linux/macos/windows`) — **always active here**.

**Three divergences from the manual** (the manual is idealized / internally inconsistent;
these only surface against the real 0.10 source — verified in
`~/.cargo/registry/.../cubecl-runtime-0.10.0/src/tune/`):

| # | Manual says | Real 0.10 API | Impact |
|---|-------------|---------------|--------|
| 1 | `TunableSet::new`'s first closure returns a `String` (`"axpy-tune"`) | First arg is the **KeyGenerator** `for<'a> Fn(&I::At<'a>) -> AutotuneKey` — must return the **key type** | Wrong key ⇒ won't compile / wrong cache bucketing |
| 2 | `TUNER.execute(&key, …)` passes the `AutotuneKey` | `execute(id, …)`'s first arg is the **cache-namespace ID** (`Display`); the key is generated **internally** from inputs via the KeyGenerator | Conflating them mis-namespaces the on-disk cache |
| 3 | `impl AutotuneKey for K {}` (marker) | Same — BUT the trait alias requires `serde::{Serialize, DeserializeOwned}` under `std_io` | Must add a `serde` dep + `#[derive(Serialize, Deserialize)]` on the key |

`local_tuner!("name")` exists (the manual's form) **and** `local_tuner!()` (no-arg, the
inline-doc form) — both work; the name is appended to `module_path!()`.

## How to Run

```bash
cargo run --release --features rocm --example spike037_autotune_hip_feasibility
# inspect the persisted cache:
cat target/autotune/0.10.0/rocm_0/spike037_autotune_hip_feasibility-tune_impl-hist037.json.log
```

## What to Expect

- Cold first run: `[cold] returned in ~300–500ms` (JIT-compiles + benchmarks **both**
  variants), then `[warm] … ~6µs` (in-process cache hit).
- A persisted cache file under `target/autotune/0.10.0/rocm_0/…json.log`.
- After the file exists, a fresh process returns `[cold] … ~800µs` (persistent disk hit —
  no re-benchmark).

## Investigation Trail

1. **Source audit first** (no GPU needed): confirmed `cubecl::tune` is re-exported, read
   the real `LocalTuner::execute` / `TunableSet::new` / `Tunable::new` / `AutotuneKey`
   signatures. Found the three manual divergences above — wrote the example against the
   **real** API, not the manual's.
2. **`serde` is not in the workspace.** The `AutotuneKey` bound forced adding
   `serde = { version="1", features=["derive"] }` to lgbm-compute **dev-dependencies**
   (examples-only — production build, which compiles no examples, never pulls it). This is
   a real, if small, integration cost finding.
3. **Compiled `--features rocm`** — the entire `tune_impl` module (LocalTuner/TunableSet/
   Tunable/execute wrapping the real `#[cube(launch)]` kernel) built clean on the hip
   backend. Only my post-read helper had trivial errors (Bytes vs Vec<u8>).
4. **Ran on the device:**
   - Cold execute **490ms** → both variants benchmarked (json shows both timings).
   - Warm in-process execute **6.26µs** (~78,000× faster) = cache hit.
   - The persisted `json.log`: `fastest_index:1` = **`build_rp_P16`** won (median 4.79ms)
     vs `build_rp_P1` (median 5.79ms) — **~1.2×**, independently re-deriving spike-007's
     row-partition occupancy win. The tuner picked the right kernel with zero hand-tuning.
   - 3rd process (cache file present): cold execute **828µs** = persistent disk hit
     (re-derived the winner from disk; the manual's "warm-start across processes" claim
     holds on hip, contra my mid-spike doubt — the first re-run was still settling the
     write).
5. **In-place hazard observed (lead-in to 038):** the output buffer is ACCUMULATED garbage
   — `max grad-cell` = 25508 after a cold tune (both variants × many benchmark reps into
   the SAME shared `d_out` handle), and 1822 after a persistent hit (winner runs ~twice).
   The result depends on *how many times tuning launched the kernel*. `CloneInputGenerator`
   clones the **handle ref**, not the buffer, so every rep mutates one buffer. This is
   exactly the manual's §3 caveat — spike-038 quantifies it and finds the fix.

## Results

**VERDICT: VALIDATED.** Autotune is feasible on cubecl-hip 0.10 end-to-end:
compile ✓, run-on-device ✓, benchmark-both ✓, pick-winner ✓ (matched spike-007),
in-process cache ✓ (~6µs), persistent cross-process cache ✓ (~800µs cold-with-cache).

**Carry-forward requirements for the real build:**
- Add `serde` derive to whatever crate defines the `AutotuneKey` (dev-only if the key
  lives in an example; a real `[dependencies]` entry if it ships).
- Write tunable closures against the **real** API (KeyGenerator returns the key;
  `execute`'s id is the cache namespace, e.g. the device id).
- **The accumulating histogram/partition kernels are NOT autotune-safe with
  `CloneInputGenerator`** — the benchmark reps corrupt the output buffer. Resolve in 038
  before wiring anything.

**Surprises:** (1) the tuner re-derived the spike-007 P=16 winner unaided — evidence the
*selection* is sound even on the spoofed APU (it's a relative within-device comparison, the
one thing the spoof doesn't confound — cf the 036 divergence-measurability carve-out).
(2) The manual is wrong/inconsistent on the two most load-bearing API points (key-gen
return type, execute's id) — do not code from it; code from the source.
