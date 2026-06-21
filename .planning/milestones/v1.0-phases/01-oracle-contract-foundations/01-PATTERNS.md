# Phase 1: Oracle Contract + Foundations - Pattern Map

**Mapped:** 2026-06-05
**Files analyzed:** 14 new + 2 removed/restructured
**Analogs found:** 11 / 14 (C++ source analogs; the remaining 3 are Rust-idiom/greenfield with no C++ analog)

> **Greenfield note:** There is no pre-existing Rust code in this repo except a hello-world
> `src/main.rs` (removed this phase) and a single-package root `Cargo.toml` (converted to a
> virtual workspace this phase). The authoritative analogs for nearly every new Rust file live
> in the **read-only** `LightGBM/` C++ reference. Excerpts below are extracted verbatim from
> that source so the planner can ground each task at line-level fidelity. Rust-idiom analogs
> (workspace manifest, thiserror enums) have no C++ counterpart — the idiomatic pattern is noted instead.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `Cargo.toml` (root, restructure) | config | transform | RESEARCH §Pattern 1 (Rust idiom) | rust-idiom (no C++ analog) |
| `rust-toolchain.toml` (new) | config | — | RESEARCH §Recommended Structure (Rust idiom) | rust-idiom (no C++ analog) |
| `src/main.rs` (REMOVE) | — | — | — | n/a (deletion, per D-10) |
| `crates/lgbm-core/src/types.rs` | model/types | transform | `LightGBM/include/LightGBM/meta.h` | exact (typedefs + constants) |
| `crates/lgbm-core/src/random.rs` | utility | transform | `LightGBM/include/LightGBM/utils/random.h` | exact (whole class) |
| `crates/lgbm-core/src/error.rs` | utility (error types) | — | `config.cpp` `Log::Fatal` sites (semantics only) | rust-idiom + C++ error semantics |
| `crates/lgbm-core/src/config/mod.rs` (struct + defaults) | config/model | transform | `LightGBM/include/LightGBM/config.h` | exact (field list + defaults) |
| `crates/lgbm-core/src/config/alias.rs` (alias table) | config | transform | `config_auto.cpp` `alias_table()` | exact (static map) |
| `crates/lgbm-core/src/config/set.rs` (`from_params`/Set: seeds + members + validation + conflicts) | config | transform/validation | `config.cpp` `Config::Set` + `config_auto.cpp` `GetMembersFromString` + `CheckParamConflict` | exact |
| `crates/lgbm-core/src/lib.rs` | config (module wiring) | — | (Rust idiom) | rust-idiom |
| `crates/lgbm-compute/src/lib.rs` (`Backend` trait skeleton) | provider/trait | — | none (CubeCL idiom; CMP-01 seam) | no-analog (skeleton only) |
| `crates/lgbm-compute/Cargo.toml` | config | — | (Rust idiom) | rust-idiom |
| `crates/oracle-harness/src/lib.rs` + `comparator` | utility (test tier) | transform | `meta.h` `kEpsilon` + abs-diff (ORA-01) | partial (tolerance constant only) |
| `crates/oracle-harness/tests/*` + `fixtures/` + reference manifest | test | file-I/O | `LightGBM/CMakeLists.txt` + `VERSION.txt` (build flags) | role-match (build config) |
| `xtask`/regen (C++ harness + CMake driver) | script (dev-only) | batch/file-I/O | `LightGBM/CMakeLists.txt` (deterministic flags) | role-match |

## Pattern Assignments

### `crates/lgbm-core/src/random.rs` (utility, transform)

**Analog:** `LightGBM/include/LightGBM/utils/random.h` (entire class, lines 18-112) — **the single most parity-critical port in the phase.** Port the WHOLE class; every constant is load-bearing.

**State + constructor** (random.h lines 32-34, 111):
```cpp
explicit Random(int seed) { x = seed; }   // seeded ctor only — the default ctor uses std::random_device, NOT reproducible, DO NOT port
unsigned int x = 123456789;               // u32, default seed 123456789
```
Rust: `pub struct Random { x: u32 }` ; `Random::new(seed: i32) -> Self { Random { x: seed as u32 } }`.

**Core LCG (the two private generators)** (random.h lines 101-109):
```cpp
inline int RandInt16() {
  x = (214013 * x + 2531011);
  return static_cast<int>((x >> 16) & 0x7FFF);   // 15-bit, [0, 32767]
}
inline int RandInt32() {
  x = (214013 * x + 2531011);
  return static_cast<int>(x & 0x7FFFFFFF);        // 31-bit non-negative
}
```
Rust: `self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);` then `((self.x >> 16) & 0x7FFF) as i32` / `(self.x & 0x7FFF_FFFF) as i32`. **MUST be `u32` `wrapping_*`** — porting over `i32`/`i64` panics in debug and diverges (RESEARCH Pitfall 1).

**Public range/float draws** (random.h lines 41-62):
```cpp
inline int NextShort(int lower, int upper) { return (RandInt16()) % (upper - lower) + lower; }
inline int NextInt(int lower, int upper)   { return (RandInt32()) % (upper - lower) + lower; }
inline float NextFloat() { return static_cast<float>(RandInt16()) / (32768.0f); }  // f32 divide by 32768.0f, NOT 65536/f64
```
`NextFloat` must stay f32 end-to-end: `(self.rand_int16() as f32) / 32768.0_f32` (RESEARCH Pitfall 2). Range `[0.0, ~0.99997)`.

**`Sample(N, K)` — both branches, branch boundary uses `double` log2** (random.h lines 69-98):
```cpp
inline std::vector<int> Sample(int N, int K) {
  std::vector<int> ret; ret.reserve(K);
  if (K > N || K <= 0) { return ret; }
  else if (K == N) { for (int i=0;i<N;++i) ret.push_back(i); }
  else if (K > 1 && K > (N / std::log2(K))) {                 // BRANCH BOUNDARY: N/log2(K) computed in double
    for (int i=0;i<N;++i) {
      double prob = (K - ret.size()) / static_cast<double>(N - i);  // ret.size() is size_t
      if (NextFloat() < prob) ret.push_back(i);
    }
  } else {
    std::set<int> sample_set;                                  // ORDERED set → BTreeSet<i32> in Rust
    for (int r = N - K; r < N; ++r) {
      int v = NextInt(0, r + 1);
      if (!sample_set.insert(v).second) { sample_set.insert(r); }  // collision → insert r instead
    }
    for (auto it = sample_set.begin(); it != sample_set.end(); ++it) ret.push_back(*it);
  }
  return ret;
}
```
Replicate the condition in `f64` (`K as f64 > (N as f64 / (K as f64).log2())` guarded by `K > 1`); use `BTreeSet<i32>` to match `std::set` ordering; golden-test N,K pairs **straddling the threshold** (RESEARCH Pitfall 3, CONTEXT specifics line 90).

**Error handling:** none — pure arithmetic, no fallible paths. Document in rustdoc as deterministic / NON-cryptographic (RESEARCH Security V6).

---

### `crates/lgbm-core/src/types.rs` (model/types, transform)

**Analog:** `LightGBM/include/LightGBM/meta.h` (lines 27-57, entire type/constant block).

**Type aliases** (meta.h lines 27-48):
```cpp
typedef int32_t data_size_t;          // signed by design
#ifndef SCORE_T_USE_DOUBLE
typedef float score_t;                // f32 contract (D-01) — DO NOT enable SCORE_T_USE_DOUBLE
#endif
#ifndef LABEL_T_USE_DOUBLE
typedef float label_t;                // f32
#endif
typedef int32_t comm_size_t;
```
Rust: `pub type DataSizeT = i32; pub type ScoreT = f32; pub type LabelT = f32; pub type CommSizeT = i32;`

**Constants** (meta.h lines 50-56, 78-80):
```cpp
const score_t kMinScore = -std::numeric_limits<score_t>::infinity();
const score_t kMaxScore =  std::numeric_limits<score_t>::infinity();
const score_t kEpsilon = 1e-15f;
const double  kZeroThreshold = 1e-35f;   // NOTE: literal is 1e-35f but stored as double
#define NO_SPECIFIC (-1)
const int kAlignedSize = 32;
```
Rust: `K_MIN_SCORE = f32::NEG_INFINITY`, `K_MAX_SCORE = f32::INFINITY`, `K_EPSILON: f32 = 1e-15`, `K_ZERO_THRESHOLD: f64 = 1e-35`, `NO_SPECIFIC: i32 = -1`, `K_ALIGNED_SIZE: i32 = 32`. **Unit-test exact constant values** (FND-03 test map). `kEpsilon` is reused by `CheckParamConflict` (path_smooth, min_sum_hessian) — keep it as the single source.

**Error handling:** none (constants only).

---

### `crates/lgbm-core/src/config/mod.rs` (config/model struct + defaults) (config, transform)

**Analog:** `LightGBM/include/LightGBM/config.h` (the ~131 documented fields with default values) + `config_auto.cpp` `GetMembersFromString` (lines 329+) for the canonical member list.

**Pattern (D-12):** ONE flat `struct Config` with public fields named identically to C++ (`num_iterations`, `learning_rate`, `num_leaves`, `min_data_in_leaf`, `bagging_fraction`, `feature_fraction`, `lambda_l1`, `lambda_l2`, `max_depth`, `seed`, the six derived seeds, etc.). No nested sub-structs. Defaults come from `config.h` doc-comment `default =` annotations. `impl Default for Config` mirrors those defaults 1:1.

**Member-extraction order is canonical** — `GetMembersFromString` (config_auto.cpp lines 331-end) defines the exact field set the struct must hold and that the drift-checker asserts coverage of:
```cpp
GetString(params, "data", &data);
GetInt(params, "num_iterations", &num_iterations);
GetDouble(params, "learning_rate", &learning_rate);
GetInt(params, "num_leaves", &num_leaves);
GetInt(params, "num_threads", &num_threads);
GetBool(params, "deterministic", &deterministic);
GetBool(params, "force_row_wise", &force_row_wise);   // determinism-critical: ORA-02 reference sets this true
... (131 total via Get{Int,Double,String,Bool})
```
**In-scope subset (Open Question 1 / A4):** exclude distributed (`num_machines`, `local_listen_port`, `time_out`), GPU (`num_gpu`, `gpu_*`), linear-tree, quantized-grad params from validation/exposure, but port the full alias table verbatim (cheap). Planner enumerates the in-scope set against v1 REQUIREMENTS; drift-checker (below) guards coverage.

**Error handling:** struct itself is infallible; fallibility lives in `set.rs`.

---

### `crates/lgbm-core/src/config/alias.rs` (config, transform)

**Analog:** `config_auto.cpp` `Config::alias_table()` (lines 10-182) — a `static std::unordered_map<std::string,std::string>` mapping alias → canonical name. Port **verbatim** as a Rust `&'static` map (e.g. `phf` map or a `match`/`HashMap` built once).

**Excerpt** (config_auto.cpp lines 10-49):
```cpp
const std::unordered_map<std::string, std::string>& Config::alias_table() {
  static std::unordered_map<std::string, std::string> aliases({
  {"config_file", "config"}, {"task_type", "task"}, {"objective_type", "objective"},
  {"app", "objective"}, {"application", "objective"}, {"loss", "objective"},
  {"boosting_type", "boosting"}, {"boost", "boosting"},
  {"num_iteration", "num_iterations"}, {"n_iter", "num_iterations"}, {"num_tree", "num_iterations"},
  {"n_estimators", "num_iterations"}, ...
  {"shrinkage_rate", "learning_rate"}, {"eta", "learning_rate"},
  {"num_leaf", "num_leaves"}, {"max_leaves", "num_leaves"}, {"max_leaf", "num_leaves"},
  {"num_thread", "num_threads"}, ...
```
Resolution: an alias maps to its canonical; canonical names map to themselves (or are absent and passed through). CFG-02 test asserts the Rust map equals `alias_table()` for in-scope entries.

**Error handling:** unknown param → `Log::Warning("Unknown parameter %s", kv)` in C++ (config.cpp line 28). In Rust this is a warning/no-op or (Claude's discretion) a soft diagnostic — NOT a hard error, matching C++.

---

### `crates/lgbm-core/src/config/set.rs` (`from_params` / `Set`) (config, transform + validation)

**Analog:** `config.cpp` `Config::Set` (lines 257-308) + `config_auto.cpp` `GetMembersFromString` CHECK constraints + `config.cpp` `CheckParamConflict` (lines 314-474). This file orchestrates the four-stage pipeline.

**Stage 1 — seed derivation (EXACT ORDER, depends on `random.rs`)** (config.cpp lines 259-268):
```cpp
if (GetInt(params, "seed", &seed)) {
  Random rand(seed);
  int int_max = std::numeric_limits<int16_t>::max();   // 32767
  data_random_seed      = rand.NextShort(0, int_max);
  bagging_seed          = rand.NextShort(0, int_max);
  drop_seed             = rand.NextShort(0, int_max);
  feature_fraction_seed = rand.NextShort(0, int_max);
  objective_seed        = rand.NextShort(0, int_max);
  extra_seed            = rand.NextShort(0, int_max);   // THIS EXACT ORDER
}
```
Six sub-seeds derived from a fresh `Random(seed)` in this order; a wrong order silently corrupts every later sampling phase (RESEARCH Pitfall 4). Golden-test the derivation (FND-01 seed_derivation test).

**Stage 2 — enum/type parse with typed errors** (config.cpp `GetInt` lines 33-40, `GetBoostingType` lines 99-112):
```cpp
// GetInt: on parse failure → Log::Fatal("Parameter %s should be of type int, got \"%s\"", key, candidate);
// GetBoostingType / GetObjectiveType / GetDeviceType / GetTreeLearnerType / GetTaskType:
//   unknown string → Log::Fatal("Unknown boosting type %s", value);   (and analogous messages)
```
**Rust mapping (CFG-03, FND-04):** C++ `Log::Fatal` → typed `Err`. Recommended taxonomy:
`ConfigError::InvalidType { param, value }` for parse failures; `ConfigError::UnknownValue { param, value }` for unknown enum strings (boosting/objective/device_type/tree_learner/task). Never `Log::Fatal`/panic on user input (RESEARCH Security V5/DoS).

**Stage 3 — CHECK_* range constraints (60 inline)** (config_auto.cpp lines 337-376, representative):
```cpp
GetInt(params, "num_iterations", &num_iterations);  CHECK_GE(num_iterations, 0);
GetDouble(params, "learning_rate", &learning_rate); CHECK_GT(learning_rate, 0.0);
GetInt(params, "num_leaves", &num_leaves);          CHECK_GT(num_leaves, 1); CHECK_LE(num_leaves, 131072);
GetInt(params, "min_data_in_leaf", &min_data_in_leaf); CHECK_GE(min_data_in_leaf, 0);
GetDouble(params, "min_sum_hessian_in_leaf", &min_sum_hessian_in_leaf); CHECK_GE(min_sum_hessian_in_leaf, 0.0);
GetDouble(params, "bagging_fraction", &bagging_fraction); CHECK_GT(bagging_fraction, 0.0); CHECK_LE(bagging_fraction, 1.0);
GetDouble(params, "pos_bagging_fraction", &pos_bagging_fraction); CHECK_GT(..., 0.0); CHECK_LE(..., 1.0);
GetDouble(params, "neg_bagging_fraction", &neg_bagging_fraction); CHECK_GT(..., 0.0); CHECK_LE(..., 1.0);
```
Each `CHECK_GE/GT/LE/LT` → a typed `Err` (e.g. `ConfigError::OutOfRange { param, value, bound }`) instead of fatal. Full constraint list enumerated in RESEARCH §"Config CHECK constraints" (the 60 inline ones; the drift-checker asserts in-scope coverage). One `Result`-returning validator per field or a table-driven validator.

**Stage 4 — `CheckParamConflict` (these MUTATE the config, not just validate)** (config.cpp lines 314-474). In-scope (single-machine) mutations/errors to port:
```cpp
// multiclass objective requires num_class > 1, else Fatal → Err                       (lines 319-327)
// objective/metric multiclass mismatch → Fatal → Err                                  (lines 328-338)
// max_depth>0 && num_leaves NOT explicitly set → if 2^max_depth < num_leaves:
//   num_leaves = 2^max_depth;                       // MUTATION                        (lines 384-398)
// path_smooth > kEpsilon && min_data_in_leaf < 2 → min_data_in_leaf = 2; (warn)        (lines 439-442)
// min_data_in_leaf <= 0 && min_sum_hessian_in_leaf <= kEpsilon → min_data_in_leaf = 1; (lines 457-462)
// boosting=="goss" → boosting="gbdt", data_sample_strategy="goss"; (warn)              (lines 463-468)
// bagging_by_query && data_sample_strategy!="bagging" → bagging_by_query=false; (warn) (lines 470-473)
```
Determinism-relevant for ORA-02: `device_type=="cuda"` forces `force_row_wise=true` (lines 410-413) — the pinned CPU reference relies on `force_row_wise=true` being honored. Port the in-scope mutations exactly; warnings → log/no-op; fatals → typed `Err`. (RESEARCH §CheckParamConflict side-effects.)

**Error handling:** the whole `from_params` returns `Result<Config, ConfigError>`; mutations applied in-place before returning `Ok`.

---

### `crates/lgbm-compute/src/lib.rs` (`Backend` trait skeleton) (provider/trait)

**Analog:** NONE — this is the CubeCL isolation seam (CMP-01). No kernels this phase (D-09); a trait skeleton only. Use the CubeCL `0.10.0` `Backend`/`Runtime` idiom; planner fills it in Phase 4. Keep CubeCL types confined to this crate so alpha churn cannot leak into `lgbm-core`. See RESEARCH §"No Analog" rationale (A2: 0.10 APIs evolving, skeleton-only limits risk).

---

### `crates/oracle-harness/` (test tier: comparator + fixtures + reference manifest)

**Comparator analog (ORA-01):** tolerance constant `kEpsilon`/contract is f32 ~1e-6 (D-02), NOT `meta.h`'s `kEpsilon=1e-15`. The comparator is an abs-diff check `|rust - cpp| <= 1e-6`. No direct C++ analog for the comparator itself; the tolerance is the locked Phase-1 decision.

**Reference manifest analog (ORA-02):** `LightGBM/CMakeLists.txt` (build flags, `score_t` width handling) + `LightGBM/VERSION.txt` (= `4.6.0.99`, commit `stable-32-g195c26fc`). Manifest must pin: the submodule commit hash, CMake invocation, and deterministic flags `deterministic=true force_row_wise=true num_threads=1` + fixed `seed` + default `float` width (do NOT define `SCORE_T_USE_DOUBLE`). RESEARCH Pitfalls 5 & 6.

**Fixtures (Claude's discretion, D):** line-delimited text for the 100k RNG draw sequence (diff-friendly, no serde dep); simple key=value/text for config goldens. Committed once via the regen xtask; normal `cargo test` reads them with NO C++ toolchain (D-06).

**Drift-checker test (CFG-02/CFG-01, D-11):** a `#[test]` that reads `LightGBM/include/LightGBM/config.h` + `LightGBM/src/io/config_auto.cpp` from the repo, extracts the param/alias/default set, and asserts the Rust tables are a **superset** of the in-scope set. Reference for parsing approach: `LightGBM/.ci/parameter-generator.py` (informs the parser, NOT used to codegen). Reads only in-repo files — no path traversal (RESEARCH Security V12).

**Error handling:** harness/tests/xtask use `anyhow::Result` + `?` (FND-04, app tier).

---

### `Cargo.toml` (root, restructure) + `rust-toolchain.toml`

**Analog:** Rust idiom (RESEARCH §Pattern 1) — no C++ analog. Convert root to a **virtual** `[workspace]` manifest (no `[package]`), `members = ["crates/*"]`, `resolver = "3"` (edition-2024 default, A1 — verify), `[workspace.package] edition = "2024" rust-version = "1.95"`, and `[workspace.dependencies] thiserror = "2.0.18" / anyhow = "1.0.102" / cubecl = "0.10.0"`. Each member uses `thiserror.workspace = true`. Remove `src/main.rs` (D-10); commit `Cargo.lock` (the existing 103KB lock references the old single-package layout — regenerate) and `rust-toolchain.toml`.

---

## Shared Patterns

### Error Handling (FND-04)
**Source:** `config.cpp` `Log::Fatal` sites (lines 37, 112, 180, 196, 214, 321, 325, 336, 429, 432) → typed errors.
**Apply to:** all of `lgbm-core` (boundary). Harness/xtask/tests use `anyhow`.
```text
C++ Log::Fatal("Parameter %s should be of type int...")  →  Err(ConfigError::InvalidType{param,value})
C++ Log::Fatal("Unknown boosting type %s")               →  Err(ConfigError::UnknownValue{param,value})
C++ CHECK_GT/GE/LE/LT(field, bound)                      →  Err(ConfigError::OutOfRange{param,value,bound})
```
Use `thiserror` derive (CLAUDE.md mandate + RESEARCH "Don't Hand-Roll"); never hand-roll `impl std::error::Error`; never panic/`unwrap` on user-facing parse (Security V5).

### f32 Numerical Contract (FND-03, D-01/02/03)
**Source:** `meta.h` lines 36-56 (`score_t`/`label_t` = `float`).
**Apply to:** every type alias, the RNG `NextFloat`, the comparator tolerance, and all later phases.
- Scores/gradients/hessians/leaf-values are `f32`. Do NOT use `f64` "for safety" (RESEARCH Anti-Pattern).
- `NextFloat` divides by `32768.0f` in `f32`.
- Oracle tolerance is `~1e-6` abs (D-02) — distinct from `meta.h kEpsilon=1e-15f` (an algorithm constant, not the oracle tolerance).

### Bit-exact integer arithmetic
**Source:** `random.h` lines 101-108 (`unsigned int x`, `214013*x + 2531011`).
**Apply to:** `random.rs` (and any future LCG-derived sampling). Always `u32` + `wrapping_mul`/`wrapping_add`; bit-extractions `>> 16 & 0x7FFF` / `& 0x7FFF_FFFF` exactly. (RESEARCH Pitfall 1 / Anti-Pattern.)

### Determinism flags (ORA-02)
**Source:** `config.cpp` `CheckParamConflict` (force_row_wise honoring, lines 410-413) + `LightGBM/CMakeLists.txt` + `VERSION.txt`.
**Apply to:** the reference manifest and the regen xtask. `deterministic=true force_row_wise=true num_threads=1` + fixed seed + default `float` width; pin the in-repo submodule commit, not an upstream release (RESEARCH Pitfalls 5 & 6).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/lgbm-compute/src/lib.rs` | provider/trait | — | CubeCL `Backend` seam (CMP-01); no C++ analog, skeleton only, no kernels this phase |
| `crates/oracle-harness` comparator | utility/test | transform | abs-diff at ~1e-6 is a Phase-1 decision (D-02); no C++ counterpart |
| `Cargo.toml` / `rust-toolchain.toml` | config | — | Rust workspace idiom (RESEARCH §Pattern 1); no C++ analog |

> For these, the planner should use RESEARCH.md patterns (virtual workspace manifest, CubeCL
> 0.10 `Backend` idiom, abs-diff comparator) rather than a C++ analog.

## Metadata

**Analog search scope:** `LightGBM/include/LightGBM/` (`utils/random.h`, `meta.h`, `config.h`), `LightGBM/src/io/` (`config.cpp`, `config_auto.cpp`), `LightGBM/CMakeLists.txt`, `LightGBM/VERSION.txt`; current Rust `Cargo.toml` + `src/main.rs`.
**Files scanned:** 7 C++ reference files (read fully or targeted) + 2 existing Rust files.
**Pattern extraction date:** 2026-06-05
