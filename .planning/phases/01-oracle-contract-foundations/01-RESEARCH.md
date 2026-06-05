# Phase 1: Oracle Contract + Foundations - Research

**Researched:** 2026-06-05
**Domain:** Rust workspace foundations, bit-exact PRNG porting, hand-ported config schema, deterministic C++ oracle + golden harness, f32 numerical contract
**Confidence:** HIGH (RNG, config, numerical contract grounded in actual in-repo C++ source; ecosystem versions verified on registry)

## Summary

Phase 1 is a foundations phase with **zero algorithmic ML** — no binning, trees, boosting, objectives, metrics, or kernels. Its deliverables are (1) a Cargo virtual workspace skeleton (`lgbm-core`, `lgbm-compute`, oracle harness crate), (2) a **bit-for-bit port of LightGBM's `Random` LCG**, (3) a **hand-ported `Config`** mirroring C++ 1:1 with alias resolution and CHECK validation, (4) the **f32 / ~1e-6 numerical contract** documented as a Key Decision, and (5) a **pinned, regenerable C++ 4.6 reference + committed goldens** that every later phase validates against.

The highest-value research output is the **exact LCG arithmetic** and **exact config schema/constraints**, both of which I extracted directly from the read-only `LightGBM/` source rather than memory. The LCG is the classic MSVC `rand` recurrence `x = 214013*x + 2531011` over **`unsigned int` (u32 wraparound)** with seed default `123456789`; `NextFloat = RandInt16()/32768.0f`; and `Sample(N,K)` branches at `K > 1 && K > (N / log2(K))` (note `log2` is computed in **`double`**). The config has **131 auto-extracted parameters**, an **alias table**, **60 inline `CHECK_*` range constraints**, and a `CheckParamConflict` pass with mutating side-effects. The seed→sub-seed derivation itself uses the LCG, so the RNG port is a prerequisite for config parity.

**Primary recommendation:** Port the LCG over Rust `u32` with `wrapping_mul`/`wrapping_add` (NOT `i32`/`i64`), drive a C++ harness linking `lib_lightgbm` to emit a committed 100k-draw golden, hand-port the flat `Config` struct with a static alias table + a drift-checker test that parses `config_auto.cpp`, and build the pinned C++ 4.6.0.99 reference from the in-repo submodule via CMake with `deterministic=true force_row_wise=true num_threads=1`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Numerical Contract (supersedes prior strict-1e-12 direction):**
- **D-01:** Data types are **`f32` (single-precision) end-to-end** — gradients, hessians, leaf values, scores — matching the C++ reference defaults (`score_t`/`label_t` = `float`). No `SCORE_T_USE_DOUBLE`.
- **D-02:** Oracle tolerance is **~1e-6 absolute on every backend (CPU and ROCm)**, not 1e-12.
- **D-03:** Use **standard `f32` histogram and score-update accumulations** on CPU and ROCm. The integer-quantized histogram strategy and the ordered-f64-accumulation strategy are **both dropped**.
- **D-04:** This f32 / ~1e-6 contract is documented as a **Key Decision in PROJECT.md** (satisfies SC#5).
- **Downstream note:** f32 transcendental (exp/log/pow/sigmoid) parity CPU↔ROCm at ~1e-6 is unproven — validate in Phase 4/6; fallback is CPU-resident objective grad/hess. Binning still aims for exact bin-index reproduction (integer concern, unaffected).

**C++ Reference & Goldens:**
- **D-05:** Build the C++ reference **from the in-repo `LightGBM/` submodule** (CMake) at golden-generation time with pinned deterministic flags.
- **D-06:** **Committed fixtures + idempotent regen script** (xtask/script). Normal test runs read committed fixtures and do NOT require the C++ toolchain.
- **D-07:** Use a **small C++ harness program linking `lib_lightgbm`** to emit the 100k-draw RNG sequence and per-stage intermediates; use the `lightgbm` CLI for end-to-end train/predict goldens.

**Workspace Structure:**
- **D-08:** **Virtual workspace** (root `Cargo.toml` virtual manifest, no root package); members under `crates/` with **`lgbm-*`** naming.
- **D-09:** **Minimal initial crate set** — only `lgbm-core`, `lgbm-compute` (CubeCL `Backend` trait skeleton, no kernels), and the **oracle harness crate**.
- **D-10:** **Remove the hello-world `src/main.rs`**; commit `Cargo.lock` and `rust-toolchain.toml` (SC#3).

**Config Modeling:**
- **D-11:** **Hand-port** the Rust `Config`, **guarded by a drift-checker test** that parses C++ `config.h`/`config_auto.cpp` and asserts the Rust table covers every in-scope param/alias/default. (NOT a build-time code generator.)
- **D-12:** **Single flat `Config` struct mirroring C++ `Config` 1:1** (same field names, same defaults). No grouped/nested sub-structs.
- **D-13:** Validation mirrors C++ `Config::Set` CHECK constraints as typed `Result` errors (CFG-03), using `thiserror` domain error types at the `lgbm-core` boundary (FND-04).

### Claude's Discretion
- Exact fixture file formats (RNG sequence, per-stage snapshots)
- The precise `rust-toolchain` version to pin
- The internal error-type taxonomy
- The CMake invocation details

### Deferred Ideas (OUT OF SCOPE)
- Full crate-per-responsibility split (dataset/model/treelearner/boosting/objective/metric/api/python) — added incrementally in later phases.
- A CLI/example binary to replace removed hello-world — optional, later.
- Empirical CPU↔ROCm f32 transcendental parity validation — Phase 4/6.
- Any dataset/binning, prediction, compute kernels, tree learning, boosting, objectives, metrics, Python.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-01 | Port `Random` PRNG (32-bit LCG, `NextFloat`, `Sample(N,K)`) bit-for-bit, unit-tested against captured C++ sequence | Exact LCG arithmetic extracted from `random.h` (see Standard Stack + Code Examples + Pitfalls 1-4). Harness design D-07 emits 100k golden. |
| FND-02 | Workspace crate structure (loosely-coupled crates by responsibility) under edition 2024 | Virtual workspace layout + `[workspace]` manifest pattern (Architecture Patterns). edition 2024 stable since 1.85; toolchain is 1.95. |
| FND-03 | `f32` data types end-to-end; standard `f32` accumulations; outputs within ~1e-6 | `meta.h` confirms `score_t`/`label_t` = `float` by default; `kEpsilon=1e-15f`, `kZeroThreshold=1e-35f`. Documented as Key Decision (D-04). |
| FND-04 | `thiserror` domain errors at crate boundaries; `anyhow` in app/test | `thiserror 2.0.18`, `anyhow 1.0.102` verified. Error taxonomy pattern in Architecture Patterns. |
| CFG-01 | Config struct accepting ~110 in-scope hyperparameters | 131 auto-extracted params in `config_auto.cpp` `GetMembersFromString`; flat struct 1:1 (D-12). |
| CFG-02 | Alias resolution as data table matching `config_auto.cpp` | `Config::alias_table()` is a `unordered_map<string,string>` — port verbatim as a static map. |
| CFG-03 | Validation mirroring C++ `Config::Set` CHECK constraints as typed `Result` | 60 inline `CHECK_*` constraints enumerated + `CheckParamConflict` side-effects (see Code Examples). |
| ORA-01 | Oracle harness comparing Rust vs C++ at ≤~1e-6 absolute (f32) | Harness crate (D-09); abs-diff comparator; fixtures-only at test time (D-06). |
| ORA-02 | Pinned C++ reference build/config manifest (threads, deterministic, `float` width) | Reference is `LightGBM/` submodule @ commit stable-32 (VERSION 4.6.0.99); CMake build with deterministic flags (see Environment + Code Examples). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Bit-exact RNG (`Random` LCG) | `lgbm-core` (library logic) | — | Pure integer/float arithmetic, no device dependency; every later phase's sampling depends on it |
| Config struct + alias table + validation | `lgbm-core` (library logic) | — | Shared config bag imported by every crate; mirrors C++ single-`Config`-everywhere pattern |
| Domain error types (`thiserror`) | `lgbm-core` boundary | app/test layer (`anyhow`) | Structured errors at lib boundary; ergonomic propagation in harness/tests |
| f32 numeric type aliases / constants | `lgbm-core` | — | `score_t`/`label_t`/`data_size_t` equivalents shared everywhere |
| CubeCL `Backend` trait skeleton | `lgbm-compute` | — | Single seam isolating CubeCL alpha churn (CMP-01); no kernels this phase |
| Oracle comparison harness | oracle harness crate (app/test tier) | C++ reference (external build) | Validation seam; reads committed fixtures, regenerates via xtask |
| Pinned C++ reference build | external (CMake / `LightGBM/` submodule) | xtask regen script | Golden generation only; not a runtime dependency of the Rust crate |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `thiserror` | 2.0.18 | Derive domain error enums at `lgbm-core` boundary | Mandated by CLAUDE.md + FND-04; the de-facto Rust library-error crate `[VERIFIED: crates.io]` |
| `anyhow` | 1.0.102 | Ergonomic error propagation in harness/tests/xtask | Mandated by CLAUDE.md + FND-04; standard app-layer error crate `[VERIFIED: crates.io]` |
| `cubecl` | 0.10.0 | Compute backend trait substrate (`Backend` skeleton only this phase) | Project mandate; already pinned in current `Cargo.toml` `[VERIFIED: crates.io]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `libm` | latest | Portable `log2` for the `Sample` branch boundary if `std` float behavior needs pinning | Only if cross-platform `f64 log2` reproducibility is a concern; std `f64::log2` is typically sufficient `[ASSUMED]` |
| (dev) C++ toolchain + CMake ≥ 3.28 | system | Build pinned reference for golden generation | xtask regen only; NOT a normal test-run dependency (D-06) |

**No serialization crate is strictly required.** Fixture formats are Claude's discretion (D); plain text or a tiny hand-rolled format avoids a dependency. If a structured format is preferred, `serde` + a text format is the conventional choice, but weigh it against the "minimal deps" spirit of the phase.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-ported config | Build-time codegen from `config.h` | Explicitly rejected by D-11 (hand-port + drift-checker test instead) — codegen is harder to read and review for parity |
| std `f64::log2` in `Sample` | `libm::log2` | std is fine on CPU; pin to `libm` only if a platform shows divergence (unlikely for this phase, CPU-only) |
| Custom fixture text format | `serde_json` / `bincode` | serde adds deps; for an integer/float sequence, line-delimited text is simplest and diff-friendly |

**Installation:**
```bash
# In each member crate's Cargo.toml (versions pinned in workspace Cargo.lock):
# lgbm-core:    thiserror = "2.0.18"
# oracle/xtask: anyhow = "1.0.102"
# lgbm-compute: cubecl = "0.10.0"
```

**Version verification (performed this session):**
- `thiserror 2.0.18` — `cargo search` confirmed latest `[VERIFIED: crates.io]`
- `anyhow 1.0.102` — `cargo search` confirmed latest `[VERIFIED: crates.io]`
- `cubecl 0.10.0` — `cargo search` confirmed; already in repo `Cargo.toml` `[VERIFIED: crates.io]`
- Rust toolchain `1.95.0` (2026-04-14); edition 2024 stable since 1.85 `[VERIFIED: rustc --version]`

## Package Legitimacy Audit

> slopcheck was not installable in this sandbox; all packages below are well-established and additionally verified directly on crates.io via `cargo search`. Per protocol, packages not slopcheck-cleared are tagged `[ASSUMED]` — but these three are first-party Rust ecosystem staples (dtolnay/tracel-ai) with millions of downloads, and the planner already has them pinned. No new/obscure packages are introduced.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `thiserror` | crates.io | ~6 yrs | >300M total | github.com/dtolnay/thiserror | unavailable → `[ASSUMED]` | Approved (CLAUDE.md-mandated, registry-confirmed) |
| `anyhow` | crates.io | ~6 yrs | >300M total | github.com/dtolnay/anyhow | unavailable → `[ASSUMED]` | Approved (CLAUDE.md-mandated, registry-confirmed) |
| `cubecl` | crates.io | ~2 yrs | (alpha) | github.com/tracel-ai/cubecl | unavailable → `[ASSUMED]` | Approved (project mandate, already pinned) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*slopcheck was unavailable; per protocol the planner may optionally gate installs behind a `checkpoint:human-verify`, though these three are CLAUDE.md-mandated/already-pinned and low-risk.*

## Architecture Patterns

### System Architecture Diagram

```
                         ┌──────────────────────────────────────────────┐
                         │  LightGBM/ (read-only C++ 4.6.0.99 submodule) │
                         │  random.h · config.h · config_auto.cpp        │
                         └───────────────┬──────────────────────────────┘
                                         │ (build via CMake, deterministic flags)
                                         ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  xtask / regen script  (requires C++ toolchain — DEV ONLY)   │
        │   ├─ C++ harness linking lib_lightgbm → emits RNG draws       │
        │   │     + per-stage intermediates                             │
        │   └─ lightgbm CLI → end-to-end train/predict goldens          │
        └───────────────┬─────────────────────────────────────────────┘
                        │ writes once, committed to git
                        ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Committed fixture files  (.planning-independent test data)  │
        │   rng_sequence.txt · config_goldens · stage_snapshots        │
        └───────────────┬─────────────────────────────────────────────┘
                        │ read at normal test time (NO C++ toolchain needed)
                        ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                     Cargo virtual workspace                        │
   │                                                                    │
   │   crates/lgbm-core ──────┐   (types, errors, RNG, Config)          │
   │     · Random (u32 LCG)   │                                         │
   │     · Config + aliases   │  imported by →  crates/lgbm-compute     │
   │     · thiserror errors   │                  · Backend trait        │
   │     · f32 type aliases   │                    (CubeCL skeleton)    │
   │                          │                                         │
   │   crates/oracle-harness ─┘  ← reads fixtures, abs-diff ≤ ~1e-6,    │
   │     (anyhow, test tier)      drives drift-checker + RNG parity     │
   └───────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
lightgbm_rs/
├── Cargo.toml              # virtual manifest: [workspace] members = ["crates/*"], no [package]
├── Cargo.lock              # committed (D-10)
├── rust-toolchain.toml     # committed, pins channel = "1.95.0" (or pinned stable), edition gate
├── crates/
│   ├── lgbm-core/          # types, errors, RNG, config — dependency root
│   │   ├── Cargo.toml      # thiserror = "2.0.18"
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs    # thiserror enum(s): CoreError / ConfigError
│   │       ├── random.rs   # Random LCG (u32)
│   │       ├── types.rs    # score_t/label_t/data_size_t equiv aliases + constants
│   │       └── config/     # flat Config struct, alias table, validation
│   ├── lgbm-compute/       # CubeCL Backend trait skeleton (no kernels)
│   │   ├── Cargo.toml      # cubecl = "0.10.0"
│   │   └── src/lib.rs
│   └── oracle-harness/     # validation crate (anyhow); reads fixtures
│       ├── Cargo.toml      # anyhow = "1.0.102"
│       ├── fixtures/       # committed goldens (rng, config, stage snapshots)
│       ├── src/
│       └── tests/          # RNG parity, config drift-checker, abs-diff comparator
└── LightGBM/               # read-only C++ reference (existing)
```
*(xtask convention: either a `xtask/` member crate or a `crates/oracle-harness` bin target for regen. xtask-as-member is the common Rust pattern.)*

### Pattern 1: Virtual Workspace Manifest
**What:** Root `Cargo.toml` is `[workspace]`-only (no `[package]`), members globbed under `crates/`.
**When to use:** Multi-crate repo where the root is not itself a publishable crate (D-08).
**Example:**
```toml
# Cargo.toml (root, virtual)
[workspace]
resolver = "3"            # edition 2024 default resolver
members = ["crates/*"]

[workspace.package]
edition = "2024"
rust-version = "1.95"

[workspace.dependencies]
thiserror = "2.0.18"
anyhow = "1.0.102"
cubecl = "0.10.0"
```
Each member then does `thiserror.workspace = true` etc. `[CITED: doc.rust-lang.org/cargo/reference/workspaces.html]` (resolver "3" is the edition-2024 default `[ASSUMED]` — verify against current cargo docs).

### Pattern 2: Bit-exact LCG over `u32`
**What:** The state `x` is C++ `unsigned int` → port as Rust `u32` with `wrapping_*`. The recurrence and bit-extractions are exact.
**When to use:** FND-01 — this is the single most parity-critical port in the phase.
**Example:** see Code Examples below.

### Pattern 3: Flat Config + static alias map + sequential validation
**What:** One `struct Config` with public fields named identically to C++; a `&'static` alias→canonical map; a `Set(params)`/`from_params` that (1) derives seeds, (2) extracts members, (3) runs `CHECK_*` as `Result`, (4) runs conflict resolution.
**When to use:** CFG-01/02/03 (D-12).

### Pattern 4: Drift-checker test (not codegen)
**What:** A `#[test]` that reads `LightGBM/include/LightGBM/config.h` + `LightGBM/src/io/config_auto.cpp` from the repo, extracts the param/alias/default set, and asserts the Rust tables are a superset of the in-scope set. Fails CI if C++ adds a param the Rust port doesn't cover.
**When to use:** D-11 guard. Lighter than codegen, catches drift.

### Anti-Patterns to Avoid
- **Porting the LCG over `i32`/`i64`:** C++ `x` is `unsigned int`; signed overflow in Rust panics in debug and is UB-equivalent reasoning in C++. Must be `u32` `wrapping_mul`/`wrapping_add`.
- **Using `f64` for scores "for safety":** Explicitly rejected (D-01/D-03). Out-precisioning the f32 reference breaks parity, not improves it.
- **Nested/grouped config sub-structs:** Rejected by D-12; defeats 1:1 cross-checking.
- **Build-time config codegen:** Rejected by D-11.
- **Making normal `cargo test` depend on the C++ toolchain:** Rejected by D-06; only the regen xtask may need it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Error enums + `Display`/`Error` impls | Manual `impl std::error::Error` | `thiserror` derive | CLAUDE.md mandate; correct source-chaining |
| Error propagation in harness/xtask | Custom result wrappers | `anyhow::Result` + `?` | CLAUDE.md mandate; context-rich |
| Workspace dependency dedup | Per-crate version strings | `[workspace.dependencies]` + `.workspace = true` | Single source of version truth, matches `Cargo.lock` discipline |
| Device abstraction | Raw HIP/CUDA/OpenCL | `cubecl` `Backend` trait | Project mandate; isolates alpha churn (CMP-01) |

**Key insight:** The RNG and config porting are *deliberately* hand-written — they are the parity contract, not boilerplate. Everything else (errors, deps, device layer) should lean on standard crates. Do not invert this.

## Common Pitfalls

### Pitfall 1: Signedness of the LCG state
**What goes wrong:** Porting `x` as `i32`/`i64` produces different `>> 16` and `& mask` results and can panic on overflow.
**Why it happens:** C++ `unsigned int x` is easy to miss; the multiply `214013 * x` overflows u32 by design.
**How to avoid:** `let mut x: u32 = 123456789;` then `x = x.wrapping_mul(214013).wrapping_add(2531011);`. `RandInt16` returns `((x >> 16) & 0x7FFF) as i32`; `RandInt32` returns `(x & 0x7FFF_FFFF) as i32`.
**Warning signs:** First few draws diverge from the golden; debug-build panic "attempt to multiply with overflow".

### Pitfall 2: `NextFloat` precision and the `/32768.0f` divisor
**What goes wrong:** Using `65536.0` or `f64` division changes the value.
**Why it happens:** `RandInt16()` is a 15-bit value `[0, 32767]`; C++ divides by `32768.0f` in **`float`**.
**How to avoid:** `(rand_int16 as f32) / 32768.0_f32` — keep it f32 throughout. Range is `[0.0, ~0.99997)`.
**Warning signs:** `NextFloat` golden mismatch in the last few ULPs.

### Pitfall 3: `Sample` branch boundary uses `double` log2 and `size_t` arithmetic
**What goes wrong:** Choosing the wrong branch (streaming vs set-based) produces a different draw sequence even if each branch is individually correct.
**Why it happens:** The condition is `K > 1 && K > (N / std::log2(K))` where `N/log2(K)` is `int / double → double`. The streaming branch uses `prob = (K - ret.size()) / static_cast<double>(N - i)` with `ret.size()` a `size_t`. The set-based branch uses `NextInt(0, r+1)` and a `std::set<int>` (ordered) with a collision-reinsert rule.
**How to avoid:** Replicate the exact condition in `f64`; replicate both branches; in the set branch use a `BTreeSet<i32>` to match `std::set` ordering, and replicate the `insert(v)` / on-collision `insert(r)` logic precisely. Golden-test **across the boundary** (pick N,K pairs straddling the threshold, per CONTEXT specifics).
**Warning signs:** `Sample` matches for small/large K but diverges near the crossover.

### Pitfall 4: `NextShort`/`NextInt` modulo on a signed result
**What goes wrong:** `(RandInt16()) % (upper - lower) + lower` — `RandInt16()` is non-negative (`& 0x7FFF`), so `%` is well-defined, but the seed-derivation in `Config::Set` calls `NextShort(0, int16_max)`. Getting the range arithmetic wrong shifts all derived sub-seeds.
**Why it happens:** Config `seed` derives `data_random_seed`, `bagging_seed`, `drop_seed`, `feature_fraction_seed`, `objective_seed`, `extra_seed` **in that exact order** via `rand.NextShort(0, INT16_MAX)`.
**How to avoid:** Port `NextShort`/`NextInt` exactly and derive the six sub-seeds in the same order from a fresh `Random(seed)`. Golden-test the seed derivation.
**Warning signs:** Config goldens match except the derived seed fields.

### Pitfall 5: Pinning the wrong C++ version
**What goes wrong:** Building against released `4.6.0` instead of the in-repo snapshot diverges if behavior changed.
**Why it happens:** `VERSION.txt` is **`4.6.0.99`** (a dev snapshot, `git describe` = `stable-32-g195c26fc`), not the 4.6.0 tag.
**How to avoid:** Pin the reference to the **in-repo submodule commit** (D-05), not an external release. Record the commit hash in the reference manifest (ORA-02).
**Warning signs:** Goldens regenerate differently on a teammate's machine that pulled upstream LightGBM.

### Pitfall 6: Determinism flags incomplete
**What goes wrong:** Non-reproducible goldens.
**Why it happens:** `deterministic=true` alone is insufficient; `CheckParamConflict` warns that you must also set `force_col_wise=true` or `force_row_wise=true`, and threading must be pinned.
**How to avoid:** Reference manifest sets `deterministic=true force_row_wise=true num_threads=1` + fixed `seed` (per ORA-02 / SC#1). Default `score_t`/`label_t` = `float` (do NOT define `SCORE_T_USE_DOUBLE`).
**Warning signs:** Same inputs, different goldens across runs.

## Code Examples

### Random LCG (verified port target)
```rust
// Source: LightGBM/include/LightGBM/utils/random.h (read-only reference)
// C++: unsigned int x = 123456789;
//      RandInt16: x = (214013*x + 2531011); return (x >> 16) & 0x7FFF;
//      RandInt32: x = (214013*x + 2531011); return x & 0x7FFFFFFF;
//      NextFloat: (float)RandInt16() / 32768.0f
pub struct Random { x: u32 }

impl Random {
    pub fn new(seed: i32) -> Self { Random { x: seed as u32 } }   // explicit Random(int seed){ x = seed; }
    // default ctor uses std::random_device — NOT reproducible; only the seeded ctor is ported for parity.

    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        ((self.x >> 16) & 0x7FFF) as i32
    }
    fn rand_int32(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        (self.x & 0x7FFF_FFFF) as i32
    }
    pub fn next_short(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int16() % (upper - lower) + lower
    }
    pub fn next_int(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int32() % (upper - lower) + lower
    }
    pub fn next_float(&mut self) -> f32 {
        (self.rand_int16() as f32) / 32768.0_f32
    }
    // Sample(N, K): branch at  K > 1 && K > (N / (K as f64).log2())
    //   streaming branch: prob = (K - ret.len()) as f64 / (N - i) as f64; if next_float() < prob push i
    //   set branch: BTreeSet<i32>; for r in (N-K)..N { v = next_int(0, r+1); if !set.insert(v) { set.insert(r); } }
}
```

### Config constants & types (verified)
```rust
// Source: LightGBM/include/LightGBM/meta.h
// typedef int32_t data_size_t;   typedef float score_t;   typedef float label_t;
pub type DataSizeT = i32;
pub type ScoreT = f32;     // f32 contract (D-01) — do NOT use SCORE_T_USE_DOUBLE
pub type LabelT = f32;
pub const K_EPSILON: f32 = 1e-15;        // kEpsilon = 1e-15f
pub const K_ZERO_THRESHOLD: f64 = 1e-35; // kZeroThreshold = 1e-35f stored as double
pub const K_MIN_SCORE: f32 = f32::NEG_INFINITY;
pub const K_MAX_SCORE: f32 = f32::INFINITY;
```

### Seed derivation (verified — must run in this exact order)
```rust
// Source: LightGBM/src/io/config.cpp Config::Set
// if seed provided: Random rand(seed); int int_max = INT16_MAX;
//   data_random_seed = rand.NextShort(0, int_max);  bagging_seed; drop_seed;
//   feature_fraction_seed; objective_seed; extra_seed;   // THIS ORDER
let mut rand = Random::new(seed);
let int_max = i16::MAX as i32;          // 32767
cfg.data_random_seed     = rand.next_short(0, int_max);
cfg.bagging_seed         = rand.next_short(0, int_max);
cfg.drop_seed            = rand.next_short(0, int_max);
cfg.feature_fraction_seed= rand.next_short(0, int_max);
cfg.objective_seed       = rand.next_short(0, int_max);
cfg.extra_seed           = rand.next_short(0, int_max);
```

### Config CHECK constraints (the 60 inline ones — port as typed Result)
```text
// Source: LightGBM/src/io/config_auto.cpp Config::GetMembersFromString (verbatim list)
num_iterations >= 0          learning_rate > 0          num_leaves > 1 && <= 131072
min_data_in_leaf >= 0        min_sum_hessian_in_leaf >= 0
bagging_fraction (0,1]       pos_bagging_fraction (0,1]  neg_bagging_fraction (0,1]
feature_fraction (0,1]       feature_fraction_bynode (0,1]
early_stopping_min_delta >= 0
lambda_l1 >= 0  lambda_l2 >= 0  linear_lambda >= 0  min_gain_to_split >= 0
drop_rate [0,1]  skip_drop [0,1]  top_rate [0,1]  other_rate [0,1]
min_data_per_group > 0  max_cat_threshold > 0  cat_l2 >= 0  cat_smooth >= 0
max_cat_to_onehot > 0  top_k > 0  monotone_penalty >= 0
refit_decay_rate [0,1]  cegb_tradeoff >= 0  cegb_penalty_split >= 0  path_smooth >= 0
max_bin > 1  min_data_in_bin > 0  bin_construct_sample_cnt > 0
num_class > 0  scale_pos_weight > 0  sigmoid > 0  alpha > 0  fair_c > 0
poisson_max_delta_step > 0  tweedie_variance_power [1,2)
lambdarank_truncation_level > 0  lambdarank_position_bias_regularization >= 0
metric_freq > 0  multi_error_top_k > 0
num_machines > 0  local_listen_port > 0  time_out > 0  num_gpu > 0
// (several apply to out-of-scope distributed/gpu params — include the in-scope subset; the
//  drift-checker test asserts coverage of in-scope params.)
```

### Type-coercion behavior (verified — error semantics to mirror)
```text
// Source: LightGBM/src/io/config.cpp
// GetInt → Common::AtoiAndCheck: on parse failure → Log::Fatal("Parameter %s should be of type int...")
// In Rust this becomes a typed ConfigError::InvalidType { param, value } returned as Err (CFG-03),
// rather than a fatal log. Enum-valued params (boosting, objective, device_type, tree_learner, task)
// each have their own "Unknown X type" fatal → map to ConfigError::UnknownValue.
```

### CheckParamConflict side-effects (verified — these MUTATE config, not just validate)
```text
// Source: LightGBM/src/io/config.cpp Config::CheckParamConflict — in-scope (single-machine) subset:
// - multiclass objective vs num_class mismatch → error
// - objective/metric multiclass mismatch → error
// - max_depth>0 && num_leaves not explicitly set → may REDUCE num_leaves to 2^max_depth (a mutation)
// - device_type gpu/cuda → forces force_col_wise/force_row_wise (out-of-scope device this phase, but
//   the deterministic CPU reference relies on force_row_wise=true being honored)
// - path_smooth > kEpsilon && min_data_in_leaf < 2 → sets min_data_in_leaf = 2 (mutation)
// - min_data_in_leaf<=0 && min_sum_hessian_in_leaf<=kEpsilon → sets min_data_in_leaf=1 (mutation)
// Port the in-scope mutations; warnings → log/no-op; fatals → typed Err.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| edition 2021, resolver "2" | edition 2024, resolver "3" (default) | Rust 1.85 (2025) | Workspace manifest uses `resolver = "3"`; toolchain 1.95 supports it |
| thiserror 1.x | thiserror 2.x | 2024 | Minor API/MSRV bumps; 2.0.18 is current — use 2.x |
| Strict 1e-12 / tiered oracle | f32 / ~1e-6 single-precision | 2026-06-05 (this phase's discuss) | The numerical contract — all later phases validate against f32 reference |

**Deprecated/outdated:**
- `SCORE_T_USE_DOUBLE` / `LABEL_T_USE_DOUBLE`: confirmed *off* by default in `meta.h`; the port must NOT enable them (D-01).
- Prior `cubecl 0.9` docs on docs.rs: the repo pins `0.10.0` — use 0.10 APIs.

## Runtime State Inventory

> This is a greenfield Rust phase that also *removes* the hello-world. Two small refactor-adjacent items exist (file removal + manifest restructure); no external runtime state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastores exist yet | None (verified: greenfield Rust crate) |
| Live service config | None | None (verified: no services) |
| OS-registered state | None | None (verified: no scheduled tasks/daemons) |
| Secrets/env vars | None | None (verified: no secrets in repo) |
| Build artifacts | Current root `src/main.rs` hello-world + single-package `Cargo.toml`; existing `Cargo.lock` (103KB) will be regenerated when the workspace is restructured | D-10: remove `src/main.rs`, convert root `Cargo.toml` to virtual `[workspace]`, regenerate & commit `Cargo.lock`. The existing lock references the old single-package layout. |

## Common Pitfalls (cross-check summary)

The six pitfalls above are the parity-critical ones. The meta-pitfall: **every numeric and ordering detail in the RNG and config is load-bearing** because all downstream phases inherit these foundations. A wrong sub-seed order or LCG signedness silently corrupts every later parity test.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Resolver "3" is the edition-2024 workspace default | Architecture Pattern 1 | Low — cargo emits a warning if mismatched; easy fix |
| A2 | `cubecl 0.10` exposes a `cpu`/`hip` runtime feature suitable for a `Backend` trait skeleton | Standard Stack | Low this phase (skeleton only, no kernels); real risk is Phase 4 |
| A3 | std `f64::log2` is reproducible enough for the `Sample` branch boundary on the CPU reference | Pitfalls 3 | Low — CPU-only this phase; if divergence appears, pin `libm` |
| A4 | The "~110 in-scope params" maps to the 131 auto-extracted minus out-of-scope (distributed/gpu/linear/quantized) params | Phase Requirements / CFG-01 | Medium — exact in-scope count needs the planner to enumerate against REQUIREMENTS scope; drift-checker test (D-11) mitigates |
| A5 | thiserror/anyhow/cubecl exact patch versions (2.0.18 / 1.0.102 / 0.10.0) are current | Standard Stack | Low — verified via cargo search this session; pin in Cargo.lock |
| A6 | xtask-as-member-crate is the chosen regen pattern | Architecture | Low — Claude's discretion (D); bin target in harness crate is equivalent |

## Open Questions

1. **Exact "in-scope" parameter count (~110 vs 131 auto-extracted)**
   - What we know: `config_auto.cpp` auto-extracts 131 params via `GetMembersFromString`; alias_table maps many more aliases.
   - What's unclear: which exact params are "in-scope single-machine" — distributed (`num_machines`, `local_listen_port`, `time_out`), GPU (`num_gpu`, `gpu_*`), linear-tree, and quantized-grad params are out of v1 scope.
   - Recommendation: Planner enumerates the in-scope set from REQUIREMENTS v1 scope; the drift-checker test (D-11) asserts coverage and will flag any miss. Port the full alias table verbatim (cheap) but only validate/expose in-scope params.

2. **Fixture file format**
   - What we know: must hold the 100k RNG draw sequence + per-stage intermediates + config goldens; Claude's discretion.
   - What's unclear: text vs binary vs serde.
   - Recommendation: line-delimited text (diff-friendly, no extra deps) for the RNG sequence; keep config goldens as a simple key=value or JSON-ish text. Avoid serde unless a strong need emerges.

3. **C++ harness linkage mechanics**
   - What we know: D-07 wants a small C++ program linking `lib_lightgbm` including `Random`/`BinMapper` headers; CLI for end-to-end.
   - What's unclear: build it inside the LightGBM CMake tree (add an executable target) vs a standalone CMakeLists linking the built lib.
   - Recommendation: standalone tiny CMake target linking the built `lib_lightgbm` + including `LightGBM/include` — keeps the read-only submodule untouched. Planner to confirm.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc / cargo | All Rust crates | ✓ | 1.95.0 (edition 2024 OK) | — |
| C++ compiler (gcc/clang) | C++ reference + harness build (regen only) | ✓ (assumed; needed for LightGBM build) | per system | Goldens are committed (D-06) — regen optional |
| CMake ≥ 3.28 | LightGBM reference build | needs verification | — | Required only for golden regen, not normal tests |
| `cubecl` 0.10 + CPU runtime | `lgbm-compute` trait skeleton | ✓ (crates.io) | 0.10.0 | Skeleton compiles without a live GPU |
| ROCm GPU | (Phase 4+, not this phase) | n/a this phase | — | Not needed for Phase 1 (no kernels) |

**Missing dependencies with no fallback:** None for normal test runs (fixtures committed).
**Missing dependencies with fallback:** CMake/C++ toolchain — only the regen xtask needs them; normal `cargo test` reads committed goldens (D-06). The planner should add a verification step (`cmake --version`, `c++ --version`) inside the regen task and a clear error if absent.

## Validation Architecture

> nyquist_validation is enabled (config). Phase 1 is foundations, so most tests are unit/golden, not ML behavior.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (libtest) + `cargo test` |
| Config file | none — standard cargo layout (Wave 0: none needed) |
| Quick run command | `cargo test -p lgbm-core` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-01 | LCG reproduces 100k C++ draws bit-for-bit (RandInt16/32, NextFloat, NextInt, Sample across branch) | golden/unit | `cargo test -p oracle-harness rng_parity` | ❌ Wave 0 |
| FND-01 | Seed derivation order matches C++ `Config::Set` | unit | `cargo test -p lgbm-core seed_derivation` | ❌ Wave 0 |
| FND-02 | Workspace builds under edition 2024 | smoke | `cargo build --workspace` | ❌ Wave 0 |
| FND-03 | f32 type aliases + constants match meta.h | unit | `cargo test -p lgbm-core types` | ❌ Wave 0 |
| FND-04 | thiserror errors at boundary; anyhow in harness | compile/unit | `cargo test -p lgbm-core error` | ❌ Wave 0 |
| CFG-01 | Config struct holds in-scope params with C++ defaults | unit | `cargo test -p lgbm-core config_defaults` | ❌ Wave 0 |
| CFG-02 | Alias resolution matches `alias_table()` | unit | `cargo test -p lgbm-core alias_resolution` | ❌ Wave 0 |
| CFG-02/CFG-01 | Drift-checker: Rust covers all in-scope params/aliases in config_auto.cpp | unit | `cargo test -p oracle-harness config_drift` | ❌ Wave 0 |
| CFG-03 | Each CHECK_* constraint returns typed Err on violation | unit | `cargo test -p lgbm-core config_validation` | ❌ Wave 0 |
| ORA-01 | abs-diff comparator flags > ~1e-6 | unit | `cargo test -p oracle-harness comparator` | ❌ Wave 0 |
| ORA-02 | Reference manifest (commit hash, flags) is checked in and regen is idempotent | golden/script | `cargo test -p oracle-harness reference_manifest` (or xtask check) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate>` (the crate touched)
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** `cargo test --workspace` green + golden regen idempotency check before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `crates/oracle-harness/fixtures/rng_sequence.*` — committed 100k-draw golden (covers FND-01) — requires C++ harness + regen
- [ ] `crates/oracle-harness/tests/rng_parity.rs` — covers FND-01
- [ ] `crates/lgbm-core/src/...` + unit tests — covers FND-01/03/04, CFG-01/02/03
- [ ] `crates/oracle-harness/tests/config_drift.rs` — covers CFG drift (D-11)
- [ ] `crates/oracle-harness/tests/comparator.rs` — covers ORA-01
- [ ] Reference manifest file (commit hash + deterministic flags) — covers ORA-02
- [ ] Framework install: none (libtest is built in)

## Security Domain

> security_enforcement is enabled (ASVS level 1). Phase 1 is an offline numerical/build-foundations phase with no network, auth, user input over the wire, or persistence. Most ASVS categories are N/A.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access control surface |
| V5 Input Validation | yes (narrow) | Config param parsing → typed `Result` errors (CFG-03), not panics; reject malformed input cleanly |
| V6 Cryptography | no | The LCG is a *reproducibility* PRNG, NOT cryptographic — never represent it as secure randomness |
| V12 Files/Resources | yes (narrow) | Drift-checker reads in-repo `config.h`/`config_auto.cpp`; regen reads/writes fixtures under the repo only — no path traversal outside workspace |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted config string causing panic (overflow/parse) | Denial of Service | Return typed `ConfigError` via `Result`; use `wrapping_*` for LCG; no `unwrap` on user-facing parse |
| Supply-chain (slopsquat) on added crates | Tampering | Only CLAUDE.md-mandated, registry-verified crates added; `Cargo.lock` committed and pinned |
| Misrepresenting LCG as secure RNG | Information Disclosure | Document `Random` as deterministic/non-crypto in rustdoc |

## Sources

### Primary (HIGH confidence)
- `LightGBM/include/LightGBM/utils/random.h` (read in-session) — exact LCG, `NextFloat`, `Sample` branch
- `LightGBM/include/LightGBM/meta.h` (read in-session) — `score_t`/`label_t`/`data_size_t`, `kEpsilon`, `kZeroThreshold`
- `LightGBM/src/io/config.cpp` (read in-session) — `Config::Set`, seed derivation, `CheckParamConflict`, type-coercion fatals
- `LightGBM/src/io/config_auto.cpp` (read in-session) — `alias_table()`, `parameter_set()`, `GetMembersFromString`, 60 `CHECK_*` constraints
- `LightGBM/include/LightGBM/config.h` (read in-session) — param defaults, aliases, annotations
- `LightGBM/CMakeLists.txt` + `LightGBM/VERSION.txt` + `git describe` — version 4.6.0.99 (commit stable-32-g195c26fc), CMake ≥ 3.28, CXX 11
- `cargo search` (in-session) — thiserror 2.0.18, anyhow 1.0.102, cubecl 0.10.0
- `rustc --version` (in-session) — 1.95.0, edition 2024 stable

### Secondary (MEDIUM confidence)
- https://github.com/tracel-ai/cubecl — CubeCL runtimes (cpu/cuda/wgpu/hip); alpha, expect breaking changes
- https://burn.dev/blog/release-0.20.0/ — CubeCL CPU runtime maturity
- https://lib.rs/crates/cubecl-hip — ROCm HIP runtime existence (relevant Phase 4, noted here)

### Tertiary (LOW confidence)
- General CubeCL 0.10 feature-flag details (cpu/cuda/wgpu/hip) from WebSearch — verify exact API against docs.rs when `lgbm-compute` is filled in (Phase 4)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified on crates.io; CLAUDE.md-mandated
- RNG port: HIGH — exact arithmetic read from source, not memory
- Config schema/validation: HIGH — alias table, 131 getters, 60 CHECKs, conflict logic all read from source
- Numerical contract: HIGH — meta.h confirms f32 defaults; contract is a locked decision
- Architecture/workspace: MEDIUM-HIGH — standard cargo patterns; resolver "3" default is [ASSUMED]
- CubeCL skeleton: MEDIUM — 0.10 confirmed but APIs evolving; skeleton-only this phase limits risk

**Research date:** 2026-06-05
**Valid until:** 2026-07-05 for stack (stable); CubeCL details ~7 days (fast-moving alpha) — re-verify at Phase 4
