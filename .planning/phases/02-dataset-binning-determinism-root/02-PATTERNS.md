# Phase 2: Dataset + Binning (determinism root) - Pattern Map

**Mapped:** 2026-06-05
**Files analyzed:** 13 new (`lgbm-dataset` crate) + 4 cross-cutting (workspace/xtask/oracle-harness) modified
**Analogs found:** 17 / 17 (every new file maps to a proven Phase 1 analog in the Rust workspace)

> **Provenance note.** Every Rust file below is a transcription of a specific read-only C++
> source (`LightGBM/` tree, git-untracked — never `git add`). The "C++ source-of-truth" column
> names the authoritative spec; the "Closest Rust analog" column names the *existing Phase 1 file
> whose structure/style the new file should copy*. Copy STRUCTURE + DOC-STYLE + TEST-STYLE from
> the Rust analog; copy BEHAVIOR from the C++ source-of-truth. The two are orthogonal and both
> mandatory.

---

## File Classification

| New/Modified File | Role | Data Flow | C++ source-of-truth | Closest Rust analog | Match Quality |
|-------------------|------|-----------|---------------------|---------------------|---------------|
| `crates/lgbm-dataset/Cargo.toml` | config | — | — | `crates/lgbm-core/Cargo.toml` | exact |
| `crates/lgbm-dataset/src/lib.rs` | module-root | — | — | `crates/lgbm-core/src/lib.rs` | exact |
| `crates/lgbm-dataset/src/error.rs` | error-model | request-response (validation) | C++ `Log::Fatal`/`CHECK_*` sites | `crates/lgbm-core/src/error.rs` | exact |
| `crates/lgbm-dataset/src/bin_mapper.rs` | model (port) | transform | `src/io/bin.cpp` `FindBin`+helpers; `bin.h` `ValueToBin` | `crates/lgbm-core/src/random.rs` | role-match (1:1 C++ mirror + drift test) |
| `crates/lgbm-dataset/src/bin/mod.rs` | model (trait + factory) | transform | `bin.h` `Bin`/`BinIterator`; `bin.cpp` `CreateDenseBin`/`CreateSparseBin` | `crates/lgbm-core/src/config/mod.rs` | role-match (trait/factory vs struct) |
| `crates/lgbm-dataset/src/bin/dense_bin.rs` | model (storage) | file-I/O (byte layout) | `src/io/dense_bin.hpp` | `crates/lgbm-core/src/random.rs` | role-match (bit-exact port) |
| `crates/lgbm-dataset/src/bin/sparse_bin.rs` | model (storage) | file-I/O (byte layout) | `src/io/sparse_bin.hpp` | `crates/lgbm-core/src/random.rs` | role-match (bit-exact port) |
| `crates/lgbm-dataset/src/multi_val_bin.rs` | model (storage) | file-I/O (byte layout) | `bin.cpp` `CreateMultiValBin` | `crates/lgbm-core/src/random.rs` | role-match (bit-exact port) |
| `crates/lgbm-dataset/src/feature_group.rs` | service (offset packing + push) | transform | `feature_group.h` (offsets, `PushData`, `CreateBinData`) | `crates/lgbm-core/src/config/set.rs` | role-match (ordered pipeline port) |
| `crates/lgbm-dataset/src/efb.rs` | service (grouping) | event-driven (RNG-driven greedy) | `dataset.cpp` `FastFeatureBundling`/`FindGroups`/`GetConflictCount` | `crates/lgbm-core/src/random.rs` + `config/set.rs` | role-match (RNG-consuming sequential port) |
| `crates/lgbm-dataset/src/metadata.rs` | model (owned vectors) | CRUD | `dataset.h` `Metadata` + `FinishLoad` | `crates/lgbm-core/src/config/mod.rs` | role-match (flat owned struct) |
| `crates/lgbm-dataset/src/dataset.rs` | service (construct + immutability) | batch | `dataset.cpp` `Construct`/`FinishLoad` | `crates/lgbm-core/src/config/set.rs` | role-match (multi-stage pipeline) |
| `crates/lgbm-dataset/src/ingest.rs` | controller (public-internal API) | request-response (CRUD ingest) | `c_api.cpp` `SampleCount`/`CreateSampleIndices`; `LGBM_DatasetCreateFromMat/CSR/CSC` | `crates/lgbm-core/src/config/set.rs` (`from_params`) | role-match (validated entry point) |
| `crates/lgbm-dataset/tests/*.rs` (golden) | test | request-response (golden replay) | — (replays committed goldens) | `crates/oracle-harness/tests/rng_parity.rs` | exact |
| **Modified:** root `Cargo.toml` | config | — | — | (add `members` entry, established list) | exact |
| **Modified:** `xtask/src/main.rs` + `xtask/cpp/` | utility (capture harness) | batch (fixture regen) | `bin.cpp`/`dataset.cpp` capture | `xtask/src/main.rs` `regen` + `xtask/cpp/CMakeLists.txt` | exact |
| **Modified:** `crates/oracle-harness/` | utility (comparator + manifest) | transform | — | `crates/oracle-harness/src/comparator.rs` + `REFERENCE_MANIFEST.md` | exact (add exact-equality variants) |

---

## Pattern Assignments

### `crates/lgbm-dataset/Cargo.toml` (config)

**Analog:** `crates/lgbm-core/Cargo.toml` (lines 1-9). Add the `lgbm-core` path dep + workspace `thiserror`. Per RESEARCH line 100: `lgbm-core = { path = "../lgbm-core" }`, `thiserror.workspace = true`.

```toml
[package]
name = "lgbm-dataset"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true

[dependencies]
lgbm-core = { path = "../lgbm-core" }
thiserror.workspace = true

[dev-dependencies]
# (golden-replay tests; add anyhow.workspace if test plumbing needs it)
```

> Note: workspace deps are declared once in root `Cargo.toml` (`thiserror = "2.0.18"`, `anyhow = "1.0.102"`); reference them with `.workspace = true`, never re-pin versions (matches `lgbm-core/Cargo.toml` and `oracle-harness/Cargo.toml`).

---

### `crates/lgbm-dataset/src/lib.rs` (module-root)

**Analog:** `crates/lgbm-core/src/lib.rs` (lines 1-13). Crate-level `//!` doc naming the subsystem + provenance, `pub mod` declarations in dependency order, and a small set of `pub use` re-exports of the primary types (`Dataset`, `BinMapper`, `DatasetError`).

```rust
//! `lgbm-dataset` — deterministic feature binning + immutable columnar dataset.
//!
//! Faithful 1:1 port of LightGBM's `src/io/` binning subsystem (D-04). Bin
//! boundaries and per-row bin indices are bit-identical to C++ (the determinism
//! root every downstream split inherits). Depends only on `lgbm-core`.

pub mod bin;
pub mod bin_mapper;
pub mod dataset;
pub mod efb;
pub mod error;
pub mod feature_group;
pub mod ingest;
pub mod metadata;
pub mod multi_val_bin;

pub use dataset::Dataset;
pub use error::DatasetError;
```

---

### `crates/lgbm-dataset/src/error.rs` (error-model, request-response)

**Analog:** `crates/lgbm-core/src/error.rs` (lines 1-69) — copy the `thiserror` idiom EXACTLY.

**Doc-header pattern** (lines 1-10): map C++ `Log::Fatal`/`CHECK_*` sites to typed `Result` errors; cite "Security V5: never panic on user input".

**Enum pattern** (lines 18-58): `#[derive(Debug, Error, Clone, PartialEq, Eq)]`, one `#[error("...")]`-annotated variant per failure class, each field doc-commented.

```rust
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// Matrix/vector shape mismatch (e.g. labels.len() != num_rows).
    #[error("shape mismatch: {detail}")]
    ShapeMismatch { detail: String },

    /// Malformed CSR/CSC (non-monotone indptr, out-of-range indices).
    #[error("malformed sparse matrix: {detail}")]
    MalformedSparse { detail: String },
    // ... one variant per V5 validation class (RESEARCH §Security: dims,
    //     indptr monotonicity/bounds, label/weight length, query-sum, max_bin>=0)
}
```

**Top-level wrapper pattern** (lines 64-69): mirror `CoreError` — a crate boundary enum with `#[error(transparent)] #[from]` over the domain error, so callers match one type.

**Cross-cutting:** validation targets from RESEARCH §Security (V5 row 584): validate matrix dims, CSR/CSC `indptr` monotonicity & bounds, `indices` bounded, label/weight length == num_rows, query boundaries sum == num_rows, non-negative `max_bin`/`min_data_in_bin`. Return typed errors, never panic; no `unsafe` indexing on caller data.

---

### `crates/lgbm-dataset/src/bin_mapper.rs` (model port, transform) — THE DETERMINISM KERNEL

**Analog:** `crates/lgbm-core/src/random.rs` (entire file) — the gold-standard "bit-for-bit C++-mirror transcription + parity test" pattern.

**Doc-header pattern** (random.rs lines 1-20): open with `//! Bit-for-bit port of the C++ ... from <exact path>`, then an `# Arithmetic notes` section calling out every load-bearing type/precision decision. For `bin_mapper.rs` the load-bearing notes are: f64 boundary math (NOT f32), `f64::next_up()` ≡ `std::nextafter(a, +INF)`, the literal `(r+l-1)/2` midpoint, `0.99f` float-literal cut, stable descending categorical sort.

**Method-doc pattern** (random.rs lines 41-77): every fn carries the verbatim C++ source line as its doc (e.g. `/// C++ NextFloat: ...`). Apply to `find_bin`, `greedy_find_bin`, `find_bin_with_zero_as_one_bin`, `find_bin_with_predefined_bin`, `value_to_bin`, `need_filter`.

**Wrapping/precision discipline** (random.rs lines 44-47, 75-77): random.rs uses `wrapping_mul`/`wrapping_add` and keeps `next_float` strictly f32. `bin_mapper.rs` keeps ALL boundary math in `f64` and uses `f64::next_up()`:

```rust
// Source: LightGBM/include/LightGBM/utils/common.h:845-852
#[inline] fn get_double_upper_bound(a: f64) -> f64 { a.next_up() }
#[inline] fn check_double_equal_ordered(a: f64, b: f64) -> bool { b <= a.next_up() }
```

**`ValueToBin` verbatim transcription** (RESEARCH §5; do NOT use `partition_point`/`binary_search`):

```rust
// Source: LightGBM/include/LightGBM/bin.h:612-650
let (mut l, mut r) = (0i32, self.num_bin - 1);
if self.missing_type == MissingType::NaN { r -= 1; }
while l < r {
    let m = (r + l - 1) / 2;                  // literal -1 midpoint, load-bearing
    if value <= self.bin_upper_bound[m as usize] { r = m; } else { l = m + 1; }
}
return l as u32;
```

**RNG reuse** (random.rs is the dependency): sampling for `FindBin` routes through `lgbm_core::random::Random::new(data_random_seed).sample(n, k)` — see `ingest.rs` below. Do NOT add a new RNG.

**Inline-test pattern** (random.rs lines 123-230): a `#[cfg(test)] mod tests` with an independent reference recomputation (`ref_rand_int16`) and exact-bit asserts. For `bin_mapper.rs`, add unit tests for the on-boundary `value_to_bin` tie direction and the de-dup edge cases (cheap, no C++ needed); the heavyweight cross-C++ parity lives in `tests/` golden files.

---

### `crates/lgbm-dataset/src/bin/mod.rs` (Bin trait + BinValue trait + factory)

**Analog:** `crates/lgbm-core/src/config/mod.rs` (lines 19-25) for the module-organization + `pub use` re-export pattern (`pub mod alias; pub use alias::resolve_alias;`).

The factory mirrors C++ `CreateDenseBin`/`CreateSparseBin` width selection (RESEARCH §6, bin.cpp:613-633): pick `u8` (<256) / `u16` (<65536) / `u32` from `num_bins`, plus the `IS_4BIT` (≤16) path. D-01 locks `Box<dyn Bin>` trait-object dispatch:

```rust
pub fn create_dense_bin(num_data: i32, num_bin: i32) -> Box<dyn Bin> {
    if num_bin <= 16      { Box::new(DenseBin::<u8, true>::new(num_data)) }   // IS_4BIT (D-02)
    else if num_bin <= 256{ Box::new(DenseBin::<u8, false>::new(num_data)) }
    else if num_bin <= 65536 { Box::new(DenseBin::<u16, false>::new(num_data)) }
    else                  { Box::new(DenseBin::<u32, false>::new(num_data)) }
}
```

`BinValue` trait bound set is Claude's discretion (CONTEXT D-discretion) — bounded by "faithful mirror": minimally `Copy + Into<u32> + TryFrom<u32>`-style ops the hot path needs.

---

### `crates/lgbm-dataset/src/bin/dense_bin.rs` (storage, file-I/O byte layout)

**Analog:** `crates/lgbm-core/src/random.rs` (bit-exact port discipline) for the doc/transcription style.

**C++ source-of-truth:** `src/io/dense_bin.hpp:56-82, 510-565` (RESEARCH §6). Const-generic `DenseBin<T, const IS_4BIT: bool>`. The even/odd `buf_` split + OR-merge at `finish_load` is the EXACT byte layout Phase 4 reads — golden-test raw `data_` bytes (`tests/bin_storage_layout`).

```rust
// 4-bit Push: i1 = idx>>1; i2 = (idx&1)<<2; v = (value as u8) << i2;
//             if i2==0 { data[i1] = v } else { buf[i1] = v }
// finish_load (4-bit): for i in 0..len { data[i] |= buf[i]; } buf.clear();
// data(idx): IS_4BIT ? (data[idx>>1] >> ((idx&1)<<2)) & 0xf : data[idx]
```

---

### `crates/lgbm-dataset/src/bin/sparse_bin.rs` (storage, file-I/O byte layout)

**Analog:** `crates/lgbm-core/src/random.rs` (bit-exact port discipline).

**C++ source-of-truth:** `src/io/sparse_bin.hpp:598-659, 661-687` (RESEARCH §7). Delta-encode on `finish_load`: `std::sort` by index (non-stable OK — unique post `cur_delta==0` skip), 255-run-length deltas, then `GetFastIndex` power-of-two-strided lookup. Golden-test `deltas_`/`vals_`/`fast_index` (`tests/bin_storage_layout`).

---

### `crates/lgbm-dataset/src/multi_val_bin.rs` (storage, file-I/O byte layout)

**Analog:** `crates/lgbm-core/src/random.rs` (bit-exact port). **C++ source-of-truth:** `bin.cpp` `CreateMultiValBin` (635-706). Dense/sparse per-sub-feature selection (`sparse_rate() >= 0.7`), offset layout. Consumed by `feature_group.rs` multi-val push (`+1` convention, RESEARCH §10).

---

### `crates/lgbm-dataset/src/feature_group.rs` (service, transform)

**Analog:** `crates/lgbm-core/src/config/set.rs` (lines 1-40) — the "ordered multi-stage pipeline that mirrors a specific C++ function, stage-by-stage, with the C++ line ranges in the doc header" pattern. set.rs lists its 4 stages "in the exact C++ order" with line cites; `feature_group.rs` does the same for offset packing → `PushData`.

**C++ source-of-truth:** `feature_group.h:39-76` (offset packing, RESEARCH §9) and `:253-267` (`PushData`, RESEARCH §10).

```rust
// Offset packing (§9): offset=1; if sum_sparse_rate<0.25 && is_multi_val {offset=0; dense_multi_val=true}
//   num_total_bin = offset; bin_offsets=[num_total_bin]
//   for each feature: num_bin = mapper.num_bin(); if most_freq_bin==0 { num_bin -= offset }
//                     num_total_bin += num_bin; bin_offsets.push(num_total_bin)
// PushData (§10):
//   let bin = mapper.value_to_bin(value);
//   if bin == mapper.most_freq_bin() { return; }      // skip most-freq (implicit)
//   if mapper.most_freq_bin() == 0 { bin -= 1; }
//   if is_multi_val { multi_bin_data[sub].push(row, bin+1); }
//   else { bin += bin_offsets[sub]; bin_data.push(row, bin); }
```

**Integer-overflow guard** (RESEARCH §Security): accumulate `num_total_bin_` in `u64` (mirrors C++ `uint64_t`, dataset.cpp:383).

---

### `crates/lgbm-dataset/src/efb.rs` (service, event-driven RNG-greedy)

**Analog (primary):** `crates/lgbm-core/src/random.rs` for the RNG-consumption discipline; **(secondary)** `config/set.rs` for the "transcribe the ordered pipeline verbatim" doc style.

**C++ source-of-truth:** `dataset.cpp:60-323` (RESEARCH §11). Build EFB **last** (RESEARCH Pitfall 5) after numeric+categorical binning is golden-proven. All randomness through `lgbm_core::random::Random::new(num_data)`, calling `.sample(...)` + `.next_short(i+1, num_group)`; stable sort on non-zero-count descending; element-wise swap of the two parallel vectors in the same loop iteration (RESEARCH Anti-pattern, §11). Transcribe `GetConflictCount`/`FindGroups`/`FixSampleIndices`/the shuffle verbatim.

> EFB golden capture (layer 3) is the one MEDIUM-risk feasibility item (RESEARCH Q1/A3): a focused `dataset.cpp` capture may need a fuller `lib_lightgbm` build or a CLI-dump fallback. Sequence EFB last so a group mismatch is unambiguously an EFB bug.

---

### `crates/lgbm-dataset/src/metadata.rs` (model, CRUD)

**Analog:** `crates/lgbm-core/src/config/mod.rs` (lines 26-90) — the "ONE flat `pub struct` with public fields named identically to C++, every field doc-commented with its C++ type + default" pattern.

**C++ source-of-truth:** `dataset.h` `Metadata` + `FinishLoad`/`CalculateQueryWeights` (RESEARCH DAT-06). Use the f32 type aliases from `lgbm_core::types`: `LabelT`/`ScoreT` (= f32) for `label_`/`weights_`, `f64` for `init_score_`, `i32` (`DataSizeT`) for `query_boundaries_`. Do NOT widen labels/weights to f64 (types.rs lines 5-8: out-precisioning breaks parity).

```rust
pub struct Metadata {
    /// C++ `std::vector<label_t> label_` (f32 contract).
    pub label: Vec<lgbm_core::types::LabelT>,
    /// C++ `std::vector<label_t> weights_`.
    pub weights: Vec<lgbm_core::types::LabelT>,
    /// C++ `std::vector<double> init_score_`.
    pub init_score: Vec<f64>,
    /// C++ `std::vector<data_size_t> query_boundaries_`.
    pub query_boundaries: Vec<lgbm_core::types::DataSizeT>,
}
```

---

### `crates/lgbm-dataset/src/dataset.rs` (service, batch)

**Analog:** `crates/lgbm-core/src/config/set.rs` (multi-stage pipeline, ordered, C++-line-cited doc header).

**C++ source-of-truth:** `dataset.cpp` `Construct` (325-441) + `FinishLoad` (443-463). `FinishLoad()` is the **immutability boundary** (RESEARCH diagram): after it, the store is read-only — model this as a type-state or a `finished: bool` guard so post-finish mutation is impossible/`Err`. `Construct` calls `CreateBinData(..., force_dense=true, force_sparse=false)` (RESEARCH §8) — common MVP path is DENSE.

---

### `crates/lgbm-dataset/src/ingest.rs` (controller, request-response CRUD)

**Analog:** `crates/lgbm-core/src/config/set.rs` `from_params` (lines 34-40) — the "single validated public entry point that runs a fixed-order pipeline and returns `Result<_, DomainError>`, never panics" pattern. This is the ONLY externally-callable surface this phase exposes (D-05): `from_mat` / `from_csr` / `from_csc`.

**C++ source-of-truth:** `c_api.cpp:974-982` sampling (RESEARCH §12) — route through Phase 1 RNG:

```rust
// Source: LightGBM/src/c_api.cpp:974-982
fn sample_count(total_nrow: i32, cfg: &Config) -> i32 {
    if total_nrow < cfg.bin_construct_sample_cnt { total_nrow } else { cfg.bin_construct_sample_cnt }
}
fn create_sample_indices(total_nrow: i32, cfg: &Config) -> Vec<i32> {
    let mut rand = lgbm_core::random::Random::new(cfg.data_random_seed);  // Phase 1 RNG
    rand.sample(total_nrow, sample_count(total_nrow, cfg))
}
```

Validate caller input at the boundary (return `DatasetError`, never panic) before any indexing — see `error.rs` cross-cutting list.

---

### `crates/lgbm-dataset/tests/*.rs` (golden replay tests)

**Analog:** `crates/oracle-harness/tests/rng_parity.rs` (entire file) — the committed-golden replay pattern.

Copy these load-bearing behaviors:
- **Fixture path via `CARGO_MANIFEST_DIR`** (rng_parity.rs lines 22-24): `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/...")` — never an absolute or untracked-`LightGBM/` path at test time.
- **Graceful SKIP when fixture absent** (rng_parity.rs lines 54-61): `let Ok(text) = read_to_string(&path) else { eprintln!("SKIP — run xtask regen ..."); return; };` keeps `cargo test` green pre-capture (D-06).
- **Exact, NOT tolerance, comparisons** (rng_parity.rs lines 91-114, and RESEARCH §Validation line 558): bin indices, group offsets, storage bytes compared with `assert_eq!`; `bin_upper_bound_` f64 array compared **bit-exact** (`.to_bits()` per element, mirroring the `got.to_bits() == exp_bits` float pattern), NOT the `~1e-6` oracle tolerance.
- **Localizing assert messages** (rng_parity.rs lines 93-97): include seed/feature/row index in the message to pinpoint the diverging stage+feature (D-07 three-layer diagnosis).

Test files map to RESEARCH §Validation rows: `bin_mapper_internals`, `numeric_assignment`, `missing_edge_cases`, `categorical_folding`, `bin_storage_layout`, `efb_grouping`, `metadata`, `ingest_equivalence`, `example_dataset_parity`.

---

## Shared Patterns

### Bit-exact C++-mirror transcription (D-11/D-12 discipline)
**Source:** `crates/lgbm-core/src/random.rs` (lines 1-77).
**Apply to:** `bin_mapper.rs`, `bin/dense_bin.rs`, `bin/sparse_bin.rs`, `multi_val_bin.rs`, `feature_group.rs`, `efb.rs`.
- Doc-header: `//! Bit-for-bit port of <C++ symbol> from <exact path>` + an `# Arithmetic notes` section naming every load-bearing type/precision/ordering decision.
- Per-method doc: the verbatim C++ source line.
- Keep C++ widths exactly (`double`→`f64`, `int`→`i32`, `uint32_t`→`u32`, `uint8_t`→`u8`); use `wrapping_*` where C++ relies on unsigned overflow; keep f64 boundary math out of f32.

### thiserror domain errors at the crate boundary (FND-04 / Security V5)
**Source:** `crates/lgbm-core/src/error.rs` (lines 10-69).
**Apply to:** `error.rs` (and every `ingest.rs` validation site).
- `#[derive(Debug, Error, ...)]` enum, one `#[error("...")]` variant per failure class, fields doc-commented.
- A top-level `transparent`/`#[from]` wrapper (`DatasetError` mirrors `CoreError`).
- Never panic on caller input; return `Result`.

### f32 numerical contract (do not over-precision)
**Source:** `crates/lgbm-core/src/types.rs` (lines 10-41).
**Apply to:** `metadata.rs` (labels/weights = `LabelT`/`ScoreT` = f32), `bin_mapper.rs` (use `K_ZERO_THRESHOLD` = `1e-35` f64, `K_EPSILON` = `1e-15` f32 from `lgbm_core::types`).
- Reuse the existing aliases; do NOT redefine. Note the dual: binning *boundaries* are f64; *labels/weights/scores* are f32.

### RNG parity precondition (reuse, never re-implement)
**Source:** `crates/lgbm-core/src/random.rs` (`Random::new`/`sample`/`next_short`).
**Apply to:** `ingest.rs` (bin-construction sampling, seed = `data_random_seed`), `efb.rs` (group search/shuffle, seed = `num_data`).
- Construct `Random::new(seed)` and call `.sample(n,k)` / `.next_short(lo,hi)` exactly where C++ does. Any other RNG breaks parity (RESEARCH Don't-Hand-Roll).

### Committed-golden replay harness (D-06 / D-14 / ORA-03)
**Source:** `crates/oracle-harness/tests/rng_parity.rs` + `crates/oracle-harness/src/comparator.rs` + `REFERENCE_MANIFEST.md` + `xtask/src/main.rs` `regen` + `xtask/cpp/CMakeLists.txt`.
**Apply to:** all `lgbm-dataset/tests/*.rs`, the new comparator exact-equality variants, the new `xtask` `bin-capture` step, and the extended manifest.
- **Comparator extension** (`comparator.rs` is f32-tolerance-only, lines 68-94): add exact-equality variants — `compare_exact_u32(&[u32])` (bin-index vectors), `compare_exact_f64_bits(&[f64])` (boundary arrays), `compare_exact_bytes(&[u8])` (storage layout). Mirror the `Mismatch { index, .. }` first-divergence-reporting shape (lines 20-42).
- **xtask capture** (`main.rs` lines 82-166, `cpp/CMakeLists.txt`): add a `bin-capture` subcommand and a standalone `bin_capture.cpp` that compiles `bin.cpp` (+ later `dataset.cpp`) against on-disk `LightGBM/include` and `external_libs/` (present, untracked) — NEVER `add_subdirectory` the submodule, NEVER modify it. Drive from the same `MASTER_SEED` constant for idempotent regen (empty `git diff`).
- **Manifest** (`REFERENCE_MANIFEST.md`, written by `main.rs` `write_manifest`): extend with binning master seed, the four-source corpus parameters (max_bin / min_data_in_bin / bin_construct_sample_cnt / data_random_seed sweeps), and the **exact-comparison** note (binning is exact-integer/exact-bytes, not the ~1e-6 tolerance).
- **Fixture provenance:** copy chosen `LightGBM/examples/*` datasets into a committed `crates/lgbm-dataset/tests/fixtures/` (or extend `oracle-harness/fixtures/`); NEVER reference the untracked `LightGBM/` path at test time (project memory `lightgbm-ref-tree-untracked`).

### Workspace crate registration
**Source:** root `Cargo.toml` (lines 3-8).
**Apply to:** root `Cargo.toml`.
- Add `"crates/lgbm-dataset"` to `members` (between `lgbm-compute` and `oracle-harness`, per RESEARCH lines 92-98). Workspace deps already declared (`thiserror`/`anyhow`); no new entries.

---

## No Analog Found

None. Every new `lgbm-dataset` file maps to a proven Phase 1 Rust analog (the bit-exact `random.rs` mirror, the `error.rs` thiserror idiom, the `config/` flat-struct + ordered-pipeline patterns, and the `oracle-harness` + `xtask` golden-replay harness). The C++ port targets in `LightGBM/` are read-only source-of-truth, not files this phase modifies.

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | Phase 1 deliverables cover every structural and validation pattern this phase needs. |

---

## Metadata

**Analog search scope:** `crates/lgbm-core/src/` (random, types, error, config/*), `crates/oracle-harness/src/` + `tests/`, `xtask/src/` + `xtask/cpp/`, root `Cargo.toml`.
**Files scanned:** 13 Rust source/test files + 3 manifests + 2 C++ harness files.
**C++ source-of-truth (read-only, untracked):** `LightGBM/{include/LightGBM/bin.h, src/io/bin.cpp, src/io/dense_bin.hpp, src/io/sparse_bin.hpp, include/LightGBM/dataset.h, src/io/dataset.cpp, include/LightGBM/feature_group.h, src/c_api.cpp, include/LightGBM/utils/common.h, include/LightGBM/meta.h}` — line refs in 02-RESEARCH.md §Code Examples.
**Pattern extraction date:** 2026-06-05
