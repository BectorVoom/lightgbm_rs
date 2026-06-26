# GPU Kernel Autotuning (CubeCL `cubecl::tune` on cubecl-hip 0.10)

Blueprint for replacing the hand-tuned/env-var launch-config heuristics (row-partition `P`,
scan `CubeDim`) with CubeCL's runtime autotuner — a measured, self-calibrating, cached
selection. Synthesized from spikes 037–040 (all VALIDATED, 2026-06-26).

## Requirements (non-negotiable, from MANIFEST + project)

- **CPU f64 anchor stays bit-exact** — autotuning is rocm-only kernel plumbing; gate any wire
  with `cargo test -p lgbm-treelearner --lib` + `-p oracle-harness`. (These spikes are
  example-only + a dev-dep; the default build + treelearner tests stay green.)
- **Backend-specific paths gate on a default-false trait method**, never a global env/flag
  (the `prefers_host_partition`/`data_partition_native` idiom — 027/029/035). An autotune wire
  is a rocm-backend override; every other backend stays byte-unchanged.
- **Ship on end-to-end `bench_train`, not the isolated microbench.** Autotune's ~10% P-pick win
  (040) is on the spoofed-APU GPU build, which the 16-core CPU beats end-to-end here
  (`perf-gap-vs-cpp-40-80x`) — the durable deliverable is the METHOD + portability, not a local
  e2e number on this box.

## How to Build It

### 1. The real 0.10 API — code from the SOURCE, not the manual

`cubecl::tune::*` (re-exported via `cubecl-core` `lib.rs:50` from `cubecl-runtime/src/tune/`).
The `cubecl_manual/.../12_autotuning.md` is idealized AND internally inconsistent — it is wrong
on its three most load-bearing points. Verified against
`~/.cargo/registry/.../cubecl-runtime-0.10.0/src/tune/`:

| The manual says | The real 0.10 API |
|-----------------|-------------------|
| `TunableSet::new`'s 1st closure returns a `String` (`"axpy-tune"`) | It's the **KeyGenerator** `for<'a> Fn(&I::At<'a>) -> AutotuneKey` — returns the **key type** |
| `TUNER.execute(&key, …)` passes the `AutotuneKey` | `execute(id, client, set, inputs)` — `id` is the cache-namespace **ID** (`Display`, e.g. `"rocm:0"`); the key is generated INTERNALLY from inputs |
| `impl AutotuneKey for K {}` (marker only) | Same — but the trait alias requires `serde::{Serialize, DeserializeOwned}` under `std_io` (always on linux ⇒ persistent cache active) |

Minimal working shape (proven in `sources/037-*/spike037_autotune_hip_feasibility.rs`):

```rust
use cubecl::tune::{local_tuner, LocalTuner, Tunable, TunableSet};

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct HistKey { bucket: u32, feats: usize }   // see §3 for "bucket"
impl cubecl::tune::AutotuneKey for HistKey {}
impl std::fmt::Display for HistKey { /* required */ }

static TUNER: LocalTuner<HistKey, String> = local_tuner!("hist");   // <K, ID>

let set = TunableSet::new(
    move |_inputs: &Vec<cubecl::server::Handle>| HistKey { bucket, feats },  // KeyGenerator → KEY
    FreshOutGenerator { client, slot_len },                                  // InputGenerator (§2)
)
.with(Tunable::new("P1",  move |inp: Vec<Handle>| { launch(&c1,  &inp, 1);  Ok::<(),String>(()) }))
.with(Tunable::new("P16", move |inp: Vec<Handle>| { launch(&c16, &inp, 16); Ok::<(),String>(()) }));

TUNER.execute(&"rocm:0".to_string(), &client, std::sync::Arc::new(set), handles);
```

- Inputs type = `Vec<cubecl::server::Handle>` (blanket `TuneInputs`; `At<'a> = Self`).
- Tunable closures return `Result<(), String>`; `()` implements `AutotuneOutput`.
- Persistent cache writes to `target/autotune/0.10.0/<device>/<modpath>-<name>.json.log`
  (`fastest_index` + both variants' medians). In-proc hit ~6µs; cross-process disk hit ~800µs
  vs ~300–500ms cold-tune. **Add `serde` (derive)** to the crate defining the key (dev-dep if
  the key lives in an example; a real dep if it ships).

### 2. Accumulating kernels need a FRESH-OUTPUT InputGenerator (038 — the correctness gate)

The histogram BUILD `fetch_add`s into a resident `out`; `Handle::clone` is a ref-count bump
(NOT a buffer copy). So `CloneInputGenerator` makes **every benchmark rep accumulate into the
caller's REAL `out`** → measured **27× corruption** (N = the whole sample budget, not +1; the
manual badly undersells "accumulate during the cold run"). The winner's FINAL run uses the
ORIGINAL inputs (`tuner.rs:183` generates inputs for benchmarks; `local.rs:170` runs the winner
on the originals), so isolating the benchmark output buffer fixes it cleanly:

```rust
struct FreshOutGenerator { client: ComputeClient<RocmRuntime>, slot_len: usize }
impl InputGenerator<HistKey, Vec<Handle>> for FreshOutGenerator {
    // GAT method: spell the return through `…::At<'a>` or you get E0195.
    fn generate<'a>(&self, _k: &HistKey, inputs: &<Vec<Handle> as TuneInputs>::At<'a>)
        -> <Vec<Handle> as TuneInputs>::At<'a> {
        let mut v = inputs.clone();
        v[5] = self.client.create_from_slice(f32::as_bytes(&vec![0.0f32; self.slot_len])); // fresh out
        v
    }
}
```

Result: real `out` touched exactly once → `rel_err 0` by grad-conservation
(`Σ grad cells == feats × Σ(ord_g)`, order-independent so f32-atomic noise is irrelevant).

**Kernel-safety classification** (decide per kernel before wrapping it):
| Class | Example | Generator needed |
|-------|---------|------------------|
| OVERWRITE (`store`) | scan / split writing fresh slots | `CloneInputGenerator` is safe |
| ACCUMULATE (`fetch_add`) | histogram BUILD (`build_rp`) | **fresh-output** (this spike) |
| in-place read-modify-write | partition `indices` | **deep-COPY** generator — but partition is host-routed on rocm (035), so not a live GPU-autotune target |

### 3. Key on the occupancy REGIME, not exact dims (039 — the cache-amortization gate)

Per-leaf row counts essentially never repeat ⇒ keying `AutotuneKey` on exact `rows` = a tuning
STORM (25/25 tree nodes cold, ~975ms for ONE shallow tree; scales to minutes on real trees).
Key on **`log2(rows)` (or a small set of size-bands) + feats + bins**:

| Strategy | distinct keys (25 nodes) | total tuning wall | variant selection |
|----------|--------------------------|-------------------|-------------------|
| EXACT `rows` | 25/25 (STORM) | 975ms | best per-size (but noisy near ties) |
| **BUCKET `log2(rows)`** | **5/25** | **325ms (~3×)** | **per-regime, crossover preserved** |
| FIXED `feats` | 1/25 | 158ms | root's variant mis-applied to small leaves |

The variant choice tracks the occupancy REGIME (is the leaf big enough to saturate the CUs?),
not the exact count — so `log2(rows)` amortizes the cache to ~one tune per size-decade while
keeping per-regime selection. Warm hits are ~3µs (free). Clear `target/autotune` to force true
cold tunes when measuring tuning cost.

### 4. Read the winner / measure-don't-model (037/040 — selection is spoof-robust)

Absolute Mr/s is APU-confounded, but the tuner's RELATIVE within-device pick is sound (the one
axis the spoof doesn't break — cf the 036 divergence-measurability carve-out). Read the chosen
variant from the persisted log: find the line containing the key, parse `"fastest_index":N`,
index into your PSET. Autotune independently re-derived spike-007's P=16 winner AND **beat the
shipped 8-CU `row_partition_count` heuristic ~10%** (040).

## What to Avoid

- **Don't code from the manual** — its key-gen return type and `execute` id are both wrong; you
  will not compile (or will mis-namespace the cache).
- **Don't use `CloneInputGenerator` on an accumulating kernel** — silent 27× wrong output.
- **Don't key on exact `rows`** (or any per-leaf-unique dim) — tuning storm; the cache never
  amortizes.
- **Don't rely on `tuner.init`'s memoization to vary per-call inputs** — `init` caches the FIRST
  set per closure-type (pins the first node's dims). Build a fresh `Arc<TunableSet>` per call;
  the cache lives in the `LocalTuner` keyed by AutotuneKey + a structure checksum, so a fresh set
  with a seen key still hits.
- **Don't over-fine-key to chase the exact best P** — the P4–P16 curve is FLAT (all ~10% faster
  than P1); the exact winner wanders run-to-run on the noisy APU. You only need to AVOID P1,
  which any reasonable key does. Paying a cold tune per node to resolve a runtime-irrelevant
  near-tie is pure waste.

## Constraints

- **cubecl 0.10**: `cubecl::tune` available; `local_tuner!("name")` and `local_tuner!()` both
  work. `std_io` cfg (std + linux/mac/win) is always on here ⇒ persistent disk cache active ⇒
  serde-derive mandatory on the key.
- **Spoofed 8-CU APU** (`rocm-gfx1100-available`): autotune SELECTION is valid (relative); any
  absolute timing is confounded; the GPU build loses to the 16-core CPU end-to-end here, so the
  real payoff is portability (discrete gfx110x / NVIDIA self-calibrate with zero re-tuning).
- **Latent production mis-tune surfaced by 040:** `row_partition_count(50, n)` resolves
  `target_cubes = 8 CU × 8 = 64`, `MIN_LEAF = 256_000`, `clamp(64/50) = 1` → **P=1 at the
  production 50-feature width** (the slowest sweep point; P4–16 is ~10% faster). The 8-CU
  correction over-corrected from the old phantom-96-CU `P≈16`. Fix = recalibrate the heuristic
  (raise `CUBES_PER_CU`/lower `MIN_LEAF`) OR adopt autotune (robust + portable).

## Origin

Synthesized from spikes: 037, 038, 039, 040.
Source files available in: `sources/037-autotune-hip-feasibility/`,
`sources/038-autotune-inplace-correctness/`, `sources/039-autotune-key-cache-thrash/`,
`sources/040-autotune-vs-heuristic/`.
