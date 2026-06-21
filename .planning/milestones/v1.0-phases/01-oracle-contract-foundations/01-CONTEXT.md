# Phase 1: Oracle Contract + Foundations - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 delivers a **falsifiable f32 single-precision oracle contract (~1e-6 absolute)** plus the bit-exact foundations every later phase is validated against: the ported `Random` LCG, the f32 numerical strategy, the workspace skeleton, the hand-ported `Config`, and a pinned, regenerable C++ reference + golden harness.

In scope: oracle harness + pinned C++ reference build + committed goldens (ORA-01, ORA-02); bit-exact RNG port (FND-01); workspace structure (FND-02); f32 data-type/accumulation contract documented as a Key Decision (FND-03); `thiserror`/`anyhow` error layering (FND-04); `Config` struct + alias resolution + validation (CFG-01/02/03).

Out of scope: any dataset/binning, prediction, compute kernels, tree learning, boosting, objectives, metrics, or Python — those are later phases. No GPU kernels are written here (the compute crate is a trait skeleton only).

</domain>

<decisions>
## Implementation Decisions

### Numerical Contract (MAJOR REVISION — supersedes prior strict-1e-12 direction)
- **D-01:** Data types are **`f32` (single-precision) end-to-end** — gradients, hessians, leaf values, scores — matching the C++ reference defaults (`score_t`/`label_t` = `float`). No `SCORE_T_USE_DOUBLE`.
- **D-02:** Oracle tolerance is **~1e-6 absolute on every backend (CPU and ROCm)**, not 1e-12. 1e-12 is meaningless against an f32 reference; f32 is the most faithful baseline.
- **D-03:** Use **standard `f32` histogram and score-update accumulations** on CPU and ROCm. The integer-quantized histogram strategy and the ordered-f64-accumulation strategy are **both dropped** — they buy nothing at f32 / ~1e-6 and add complexity.
- **D-04:** This f32 / ~1e-6 contract is documented as a **Key Decision in PROJECT.md** (satisfies SC#5) — already written during this discussion, along with REQUIREMENTS.md, ROADMAP.md, and STATE.md updates so no later phase targets an unfalsifiable invariant.
- **Note for downstream:** f32 transcendental (exp/log/pow/sigmoid) parity CPU↔ROCm at ~1e-6 is unproven and must be empirically validated in Phase 4/6; if a gap appears, the fallback is CPU-resident objective grad/hess. Binning still aims for exact bin-index reproduction (an integer concern, unaffected by the f32 score contract).

### C++ Reference & Goldens
- **D-05:** **Build the C++ reference from the in-repo `LightGBM/` submodule** (CMake) at golden-generation time with the pinned deterministic flags (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, fixed seed, default `float` width). Fully reproducible from the repo; version locked to checked-in source.
- **D-06:** **Committed fixtures + idempotent regen script** (an xtask/script). Goldens are generated once, committed as fixture files, and regenerable on demand; normal test runs read committed fixtures and do NOT require the C++ toolchain. CI can verify regeneration matches.
- **D-07:** Use a **small C++ harness program linking `lib_lightgbm`** (including `Random`/`BinMapper` headers) to emit the RNG draws and per-stage intermediates as fixtures; use the `lightgbm` CLI for end-to-end train/predict goldens. (CLI alone cannot expose RNG draws or pre-final intermediates needed by SC#2 and ORA-03.)
- **D-14:** **Randomized-at-capture oracle inputs (supersedes single-fixed-seed capture).** Oracle test inputs are a *randomized, diverse* set rather than one fixed value: the regen harness derives many test cases from a single recorded **master seed** (committed in the manifest so the set is re-rollable and reproducible) — for RNG: many random LCG seeds plus randomized `N,K` pairs straddling the `Sample` `K > N/log2(K)` branch boundary; for config: randomized in-scope param combinations including boundary/invalid values. The C++ reference is run **once** over this set at golden-generation time and the `(input → output)` pairs are committed as fixtures (preserves D-06: no C++ toolchain at normal test time). The Rust suite replays **every** committed case and asserts parity — exact for integer/`f32` RNG draws, ≤ ~1e-6 absolute for float comparisons — so fidelity is validated **across varied random distributions**, not a single point. Tolerance and master seed are recorded in `REFERENCE_MANIFEST.md`.

### Workspace Structure
- **D-08:** **Virtual workspace** (root `Cargo.toml` is a virtual manifest, no root package); member crates live under `crates/` with an **`lgbm-*`** naming convention (e.g. `lgbm-core`, `lgbm-compute`).
- **D-09:** **Minimal initial crate set** — Phase 1 creates only `lgbm-core` (shared types, errors, RNG, config), `lgbm-compute` (CubeCL `Backend` trait skeleton — no kernels yet, isolates CubeCL per CMP-01), and the **oracle harness crate**. Dataset/model/treelearner/boosting/objective/metric/api/python crates are added in the phases that introduce them.
- **D-10:** **Remove the hello-world `src/main.rs`**; a CLI/example can be reintroduced later. Commit `Cargo.lock` and `rust-toolchain.toml` (SC#3).

### Config Modeling
- **D-11:** **Hand-port** the Rust `Config` (struct + alias table + defaults + CHECK validation) for readable, idiomatic Rust, **guarded by a drift-checker test** that parses C++ `config.h`/`config_auto.cpp` and asserts the Rust table covers every in-scope param/alias/default. (Not a build-time code generator.)
- **D-12:** **Single flat `Config` struct mirroring C++ `Config` 1:1** (same field names, same defaults) — easiest to cross-check for parity and matches how C++ passes one config bag everywhere. No grouped/nested sub-structs.
- **D-13:** Validation mirrors C++ `Config::Set` CHECK constraints surfaced as typed `Result` errors (CFG-03), using `thiserror` domain error types at the `lgbm-core` boundary (FND-04).

### Claude's Discretion
- Exact fixture file formats (RNG sequence, per-stage snapshots), the precise rust-toolchain version to pin, the internal error-type taxonomy, and the CMake invocation details are left to research/planning, consistent with the decisions above.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level contract (UPDATED this session — read for the revised f32/~1e-6 baseline)
- `.planning/PROJECT.md` — Core Value, Constraints, and Key Decisions table (f32 / ~1e-6 contract, standard f32 accumulations)
- `.planning/REQUIREMENTS.md` — Phase 1 requirements FND-01..04, CFG-01..03, ORA-01/02 (revised to f32 / ~1e-6)
- `.planning/ROADMAP.md` §"Phase 1" — goal + 5 success criteria (revised)
- `.planning/STATE.md` — decisions log + blockers (records this pivot)

### C++ reference source (read-only port target)
- `LightGBM/include/LightGBM/utils/random.h` — the `Random` LCG to port bit-for-bit (`RandInt16`/`RandInt32`/`NextFloat`/`NextInt`/`Sample(N,K)`, `u32` wraparound, `f32 NextFloat = RandInt16()/32768.0f`)
- `LightGBM/include/LightGBM/config.h` — single source of truth for the ~110 params, defaults, aliases, and doc-comment annotations
- `LightGBM/src/io/config_auto.cpp` — generated alias map + parameter set the Rust alias table and drift checker must match
- `LightGBM/.ci/parameter-generator.py` — reference for how config.h is parsed (informs the drift-checker, not used to codegen)
- `LightGBM/CMakeLists.txt` — build flags / `score_t` width handling for the pinned deterministic reference build

### Codebase maps (reference C++ architecture)
- `.planning/codebase/STACK.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/STRUCTURE.md` — C++ stack, conventions, layout

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None in the Rust crate yet — greenfield. `src/main.rs` is hello-world and will be removed; `Cargo.toml` declares only `cubecl = "0.10.0"` (edition 2024).

### Established Patterns
- The C++ reference patterns to mirror: single flat `Config` bag passed everywhere; string-keyed factories (not relevant to Phase 1); `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f` constants live in C++ headers (will matter from Phase 2 onward).

### Integration Points
- `lgbm-core` becomes the dependency root (types/errors/RNG/config) every later crate imports.
- `lgbm-compute`'s `Backend` trait is the single seam isolating CubeCL alpha churn (CMP-01); Phase 4 fills it in.
- The oracle harness crate is the validation seam that every later phase's parity tests plug into.

</code_context>

<specifics>
## Specific Ideas

- The numerical pivot to f32 / ~1e-6 was a deliberate mid-discussion project revision by the user: "Align the data types entirely with the original LightGBM C++ source code by shifting from f64 to f32." Faithfulness to the C++ `float` defaults is the driving intent — match the original, don't out-precision it.
- C++ RNG detail to preserve exactly: `NextFloat()` = `static_cast<float>(RandInt16()) / 32768.0f`; `Sample` branches on `K > (N / log2(K))` between the streaming and set-based algorithms — both branches must be ported and golden-tested across the branch boundary.

</specifics>

<deferred>
## Deferred Ideas

- Full crate-per-responsibility split (dataset/model/treelearner/boosting/objective/metric/api/python crates) — added incrementally in the phases that introduce each, not in Phase 1 (per D-09).
- A CLI/example binary to replace the removed hello-world — optional, later.
- Empirical CPU↔ROCm f32 transcendental parity validation — Phase 4/6.

None other — discussion stayed within Phase 1 scope.

</deferred>

---

*Phase: 1-oracle-contract-foundations*
*Context gathered: 2026-06-05*
