# Phase 3: Tree Model + Model Text I/O + Predict Parity - Pattern Map

**Mapped:** 2026-06-05
**Files analyzed:** 16 (new + modified)
**Analogs found:** 15 / 16 (1 module has no direct analog: `format.rs` `%.17g` formatter)

> **HARD CONSTRAINT (project memory: `lightgbm-ref-tree-untracked`):** The `LightGBM/`
> reference tree is git-UNTRACKED and its `external_libs/{fmt,fast_double_parser}` are
> empty — `lib_lightgbm` is UNBUILDABLE here. It may NEVER be referenced as a
> runtime/test dependency. Goldens are C++-generated (via pip `lightgbm` or
> header-only verbatim transcription, Open Q2) then **copied/committed** into
> `tests/fixtures/models/`. Tests resolve fixture paths via `CARGO_MANIFEST_DIR`,
> never an absolute path and never a path under `LightGBM/`.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-model/Cargo.toml` | config | n/a | `crates/lgbm-dataset/Cargo.toml` | exact |
| `crates/lgbm-model/src/lib.rs` | module-root | n/a | `crates/lgbm-dataset/src/lib.rs` | exact |
| `crates/lgbm-model/src/error.rs` | model (error boundary) | request-response | `crates/lgbm-dataset/src/error.rs` + `crates/lgbm-core/src/error.rs` | exact |
| `crates/lgbm-model/src/tree.rs` | model (repr + traversal) | transform | `crates/lgbm-dataset/src/bin_mapper.rs` (parallel-array struct + verbatim C++ kernel) | role+flow match |
| `crates/lgbm-model/src/ensemble.rs` | model (ensemble repr) | transform | `crates/lgbm-dataset/src/bin_mapper.rs` (struct mirror) | role match |
| `crates/lgbm-model/src/model_text.rs` | service (serde) | file-I/O / transform | `crates/lgbm-dataset/tests/golden/mod.rs` (keyed text parse) + `xtask` `write_manifest` (text emit) | partial (no exact serde analog) |
| `crates/lgbm-model/src/predict.rs` | service | transform / request-response | `crates/lgbm-dataset/src/ingest.rs` (raw-row materialize + validated entry point) | role+flow match |
| `crates/lgbm-model/src/format.rs` | utility | transform | — (NO analog) | none |
| `crates/lgbm-model/tests/model_text_roundtrip.rs` | test | file-I/O | `crates/lgbm-dataset/tests/example_dataset_parity.rs` | exact |
| `crates/lgbm-model/tests/predict_raw_parity.rs` | test | request-response | `crates/lgbm-dataset/tests/example_dataset_parity.rs` | exact |
| `crates/lgbm-model/tests/predict_transform.rs` | test | request-response | `crates/lgbm-dataset/tests/example_dataset_parity.rs` | exact |
| `crates/lgbm-model/tests/predict_leaf_parity.rs` | test | request-response | `crates/lgbm-dataset/tests/example_dataset_parity.rs` | exact |
| `crates/lgbm-model/tests/predict_subrange.rs` | test | request-response | `crates/lgbm-dataset/tests/example_dataset_parity.rs` | exact |
| `xtask/src/main.rs` (`model-capture` subcmd) | utility (codegen/capture) | batch | `xtask/src/main.rs` `bin_capture()` | exact |
| `Cargo.toml` (workspace members) | config | n/a | existing `[workspace].members` block | exact |
| `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` (extend) | config (manifest) | n/a | `xtask` `write_manifest()` | exact |

## Pattern Assignments

### `crates/lgbm-model/Cargo.toml` (config)

**Analog:** `crates/lgbm-dataset/Cargo.toml` (verbatim)

Copy the dependency shape exactly. `lgbm-model` depends on `lgbm-core` AND
`lgbm-dataset` (predict inputs reuse `from_mat`/`from_csr`/`from_csc`); dev-dep on
`oracle-harness`. Keep `edition.workspace`/`rust-version.workspace`. There are no
binaries (`tree.rs` etc. are library modules) but `lgbm-model` has no `src/bin/`
dir, so the `autobins = false` line from the analog is NOT needed here.

```toml
[package]
name = "lgbm-model"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true

[dependencies]
lgbm-core = { path = "../lgbm-core" }
lgbm-dataset = { path = "../lgbm-dataset" }
thiserror.workspace = true

[dev-dependencies]
anyhow.workspace = true
oracle-harness = { path = "../oracle-harness" }
```

---

### `crates/lgbm-model/src/lib.rs` (module-root)

**Analog:** `crates/lgbm-dataset/src/lib.rs` (lines 1-32)

Copy the doc-comment + `pub mod` + flat `pub use` re-export idiom exactly. The
analog declares every submodule with `pub mod` then re-exports the public types in
one flat block. Mirror for the RESEARCH module layout (`tree`, `ensemble`,
`model_text`, `predict`, `format`, `error`).

```rust
//! `lgbm-model` — Tree + GBDT ensemble model, model-text I/O, predictor.
//!
//! Faithful 1:1 port of LightGBM's `tree.h`/`tree.cpp` (Tree),
//! `gbdt_model_text.cpp` (envelope), `gbdt_prediction.cpp` (predict loop) (D-03/D-04).
//! Depends on `lgbm-core` (Config/types/error) and `lgbm-dataset` (predict inputs).

pub mod ensemble;
pub mod error;
pub mod format;
pub mod model_text;
pub mod predict;
pub mod tree;

pub use error::ModelError;
pub use ensemble::GbdtModel;
pub use tree::Tree;
```

---

### `crates/lgbm-model/src/error.rs` (model — error boundary)

**Analog:** `crates/lgbm-dataset/src/error.rs` (whole file) + `crates/lgbm-core/src/error.rs` (lines 18-58 for the `#[derive]` + per-variant doc + `#[cfg(test)]` Display tests)

`thiserror` derive is a CLAUDE.md mandate (never hand-roll `impl Error`). Copy the
`#[derive(Debug, Error, Clone, PartialEq, Eq)]` enum + one-variant-per-V5-input-class
shape. The Security section of RESEARCH (V5, lines 535-544) enumerates the exact
fatal sites to mirror as `ModelError` variants: malformed model text (missing
`num_leaves`, array-length inconsistency), `feature_infos`/`feature_names` count
mismatch, out-of-bounds node index, `tree_sizes` overflow. NEVER `panic!`/unchecked
`[]` on parsed model bytes.

Derive header + variant idiom to copy (from `lgbm-dataset/src/error.rs:28-66`):
```rust
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Malformed model text: a required key is absent, or a parsed array length
    /// is inconsistent with `num_leaves`/`num_cat` (would index OOB). Mirrors the
    /// C++ `Log::Fatal` parse checks (gbdt_model_text.cpp:494,514; tree.cpp).
    #[error("malformed model text: {detail}")]
    MalformedModel {
        detail: String,
    },
    /// Prediction input shape mismatch (feature count != max_feature_idx+1, etc.).
    #[error("predict shape mismatch: {detail}")]
    ShapeMismatch {
        detail: String,
    },
}
```

Also copy the `#[cfg(test)] mod tests` Display-assertion idiom verbatim from the
analog (`lgbm-dataset/src/error.rs:68-116`): construct each variant, assert
`!to_string().is_empty()` and that it contains the detail substring.

---

### `crates/lgbm-model/src/tree.rs` (model — repr + traversal)

**Analog:** `crates/lgbm-dataset/src/bin_mapper.rs` (lines 1-120) — the canonical
"parallel-array struct that 1:1-mirrors a C++ class, guarded by a bit-exact parity
test" pattern (D-04 mandate).

**Faithful-mirror struct idiom** (copy from `bin_mapper.rs:97-120`): trailing-
underscore field names kept verbatim to make the C++ correspondence unambiguous;
each field carries a `/// C++ <type> <name>` doc line. RESEARCH Pattern 1
(`03-RESEARCH.md:178-212`) gives the exact `Tree` field set already typed in Rust
— use it as the field list. Note the analog uses `pub` fields with underscore
suffixes (e.g. `pub num_bin_: i32`); follow that convention OR private fields with
accessors (Claude's discretion per CONTEXT D-04), but keep the C++ names.

```rust
/// Feature value → bin mapper + bin meta information.
///
/// Public fields mirror the C++ `BinMapper` private members (`bin.h:235-`); the
/// trailing-underscore names are kept verbatim to make the C++ correspondence
/// unambiguous for the port.
#[derive(Debug, Clone, PartialEq)]
pub struct BinMapper {
    /// C++ `int num_bin_` — number of bins.
    pub num_bin_: i32,
    ...
}
```

**Verbatim-C++-kernel idiom** (copy from `bin_mapper.rs:1-37` + 63-95): the module
doc explicitly states "transcribed line-for-line from the C++ source," names the
exact C++ source lines, and documents every arithmetic/precision decision (f64 vs
f32, `next_up()` == `std::nextafter`, tie direction). Apply the same discipline to
`Tree::Predict`/`GetLeaf`/`NumericalDecision`/`CategoricalDecision` — RESEARCH
Patterns 2-3 (`03-RESEARCH.md:214-246`) give the decision-type bit decode and
`find_in_bitset` already in Rust with cited C++ lines. Small `#[inline] fn` helpers
mirroring single C++ functions (one-to-one) — copy the `get_double_upper_bound` /
`check_double_equal_ordered` idiom (`bin_mapper.rs:65-75`):

```rust
/// C++ `Common::GetDoubleUpperBound` (`common.h:850-852`):
/// `std::nextafter(a, INFINITY)`.
#[inline]
fn get_double_upper_bound(a: f64) -> f64 {
    a.next_up()
}
```

**Reuse, don't redefine** (`bin_mapper.rs:41`): pull `K_ZERO_THRESHOLD` (1e-35) and
other constants from `lgbm_core::types` — `is_zero` uses `K_ZERO_THRESHOLD`.
```rust
use lgbm_core::types::K_ZERO_THRESHOLD;
```

**Anti-pattern guard:** RESEARCH §Anti-Patterns (`03-RESEARCH.md:263-267`) — use
`split_feature` (original index) NOT `_inner_`; accumulate in f64 not f32; no enum
tree.

---

### `crates/lgbm-model/src/ensemble.rs` (model — ensemble repr)

**Analog:** `crates/lgbm-dataset/src/bin_mapper.rs` (struct-mirror idiom, same as `tree.rs`)

`GbdtModel` mirrors C++ `models_` (flat `Vec<Tree>`) + `num_class`,
`num_tree_per_iteration`, and `InitPredict` sub-range state. RESEARCH Pattern 4
(`03-RESEARCH.md:248-261`) gives the exact `InitPredict`/`PredictRaw` stride math in
Rust. Keep the faithful-mirror field-naming idiom from the analog. The
`num_tree_per_iteration` stride (`models[i*ntpi + k]`) is the load-bearing index —
document it with the C++ source line as `bin_mapper.rs` documents its arithmetic.

---

### `crates/lgbm-model/src/model_text.rs` (service — serde)

**Analog (PARSE side):** `crates/lgbm-dataset/tests/golden/mod.rs` (lines 70-187) —
the keyed, order-independent `key=value` text parser. RESEARCH Pitfall 4
(`03-RESEARCH.md:311-315`) mandates a keyed map + pull-with-default parser (NOT
positional) to handle conditional fields (`cat_*`, single-leaf early return).

The analog's `field()`/`parse_i32()`/`parse_u32()`/`parse_*_list()` helpers
(`golden/mod.rs:148-186`) are the exact tokenizer idiom to copy:
```rust
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}
```
**BUT note divergence:** the golden loader splits on whitespace and uses `key=` prefix
matching; the model-text format is `key=<space-joined values>` per line (RESEARCH
`Tree::ToString` order, `03-RESEARCH.md:372-394`). The parse-with-fallback shape
carries over; the exact split differs (split on first `=`, then whitespace-split the
value). For float arrays, use Rust `f64::from_str` (RESEARCH Pitfall 6 — correctly
rounded, meets/exceeds both C++ parsers).

**Analog (WRITE side):** `xtask/src/main.rs` `write_manifest()` (lines 386-560) —
the idiom for emitting an exact multi-section text artifact with a fixed section
order via `format!`. The model-text envelope order (RESEARCH `SaveModelToString`,
`03-RESEARCH.md:396-422`) and per-tree `ToString` order
(`03-RESEARCH.md:372-394`) are the byte-exact spec.

**KEY de-risk (RESEARCH Summary line 50 + Don't-Hand-Roll line 277):** on a
load→write round-trip, ALL ensemble-level metadata strings (`feature_infos=`,
`feature_names=`, the entire `parameters:` block) are stored verbatim as `String`
and re-emitted unchanged — never reformatted. Only per-tree float arrays round-trip
through parse→format. So `model_text.rs` keeps captured metadata as opaque
`String`s; the write contract reduces to (a) preserve those strings and (b) call
`format.rs` `%.17g`/`{:g}` on the per-tree float arrays.

**Tree boundary slicing:** RESEARCH Pitfall 5 (`03-RESEARCH.md:317-321`) — honor the
`tree_sizes=` byte boundaries on read; compute `tree_sizes` from each serialized
tree string's byte length on write. Use checked `usize` arithmetic (Security:
`sum(tree_sizes) <= buffer.len()`).

---

### `crates/lgbm-model/src/predict.rs` (service — transform)

**Analog:** `crates/lgbm-dataset/src/ingest.rs` (lines 1-49, 217-318) — the
"validated single public entry point that reuses the dense/CSR/CSC input forms"
pattern.

**Validated-entry idiom** (`ingest.rs:15-18` doc + 217-246 body): each public predict
fn VALIDATES caller input at the boundary FIRST (return typed `ModelError`, never
panic), then runs the fixed pipeline. Copy the shape-check idiom:
```rust
let expected = (num_rows as i64) * (num_cols as i64);
if data.len() as i64 != expected {
    return Err(DatasetError::ShapeMismatch { detail: format!(...) });
}
```

**Single-widen-site idiom** (`ingest.rs:43-49`): the predict path materializes a
dense `f64` row buffer of width `max_feature_idx+1` from the caller's raw input
(D-02a: RAW values, NO re-binning). RESEARCH Open Q1 (`03-RESEARCH.md:443-446`)
confirms: mirror the C++ `Predictor::CopyToPredictBuffer` — reuse `lgbm-dataset`'s
dense/CSR/CSC iteration SHAPE for row extraction but feed RAW values to
`Tree::Predict`, do NOT route through `Dataset::construct`/`BinMapper`. A thin
raw-row materializer lives here (the dataset exposes binned columns, not raw rows).

**Reuse the existing input signatures:** `lgbm_dataset::{from_mat, from_csr, from_csc}`
exist (`ingest.rs:217,301,364`) but produce BINNED datasets — the predict path
needs the raw values, so either accept the same `(&[f32], num_rows, num_cols)` /
CSR/CSC argument shapes directly (recommended) or add a raw-row helper. Match the
argument-tuple shape of `from_mat` for consistency.

**ConvertOutput shim:** RESEARCH Code Examples (`03-RESEARCH.md:342-370`) gives the
four core transforms in Rust; parse `sigmoid_`/`num_class_` from the model's
`objective=` line (NOT a training Config). Softmax MUST use max-subtraction
(`common.h:587`).

**Anti-pattern guard:** accumulate `output[k]` in f64; cast to f32 only at the
comparison boundary (RESEARCH `03-RESEARCH.md:264`).

---

### `crates/lgbm-model/src/format.rs` (utility — `%.17g` formatter) — NO ANALOG

**No existing analog in the workspace.** This is genuinely new (the serialization
linchpin, RESEARCH Pitfall 1 `03-RESEARCH.md:295-299`). The planner should point
executors at the RESEARCH spec, not a precedent file:
- `{:.17g}` for `threshold`/`leaf_value`/`leaf_weight` (printf `%g`, 17 sig digits,
  strip trailing zeros, exponent only when exp `< -4` or `>= 17`).
- `{:g}` (6 sig digits) for `split_gain`/`internal_value`/`internal_weight`.
- ostream-default for `shrinkage` (likely == `{:g}`; golden is arbiter).
- Do NOT use `ryu`/`{}`/`{:.17e}` (shortest-round-trip ≠ `%g`).

**Test idiom to borrow:** the `format::` unit-test gate (RESEARCH test map line 497)
follows the inline `#[cfg(test)] mod tests` style used in `comparator.rs:191-226`
and `error.rs` — a battery of doubles (subnormals, 1e300, 1e-300, 0.1, integers,
values needing exactly 17 digits) compared against captured C++ output. Build &
verify this FIRST (Wave 0), before the tree writer.

---

### `crates/lgbm-model/tests/*.rs` (5 integration tests)

**Analog (ALL FIVE):** `crates/lgbm-dataset/tests/example_dataset_parity.rs` (whole file)

This is the canonical committed-golden replay test. Copy verbatim:
- **`CARGO_MANIFEST_DIR` fixture-path resolution** (`example_dataset_parity.rs:38-44`):
  ```rust
  fn fixtures_dir() -> PathBuf {
      PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
  }
  ```
  NEVER an absolute path, NEVER `LightGBM/`.
- **Graceful SKIP pre-capture** (`example_dataset_parity.rs:184-193`): if the golden
  file is absent, `eprintln!` a SKIP message naming the regen command and `return`
  (test passes). This is how the crate compiles+tests green before fixtures land.
  ```rust
  let Ok(text) = std::fs::read_to_string(&gpath) else {
      eprintln!("...: SKIP — golden {} not found. Run `cargo run -p xtask -- model-capture` ...", gpath.display());
      return;
  };
  ```
- **Localizing assert messages** naming fixture + row/tree/class (`...:232-241`).
- **Keyed golden parser** reuse the `field`/`parse_*` helpers from
  `tests/golden/mod.rs` (consider a `tests/golden/mod.rs` in the new crate, mirroring
  the dataset crate's shared test loader).

**Per-test comparator** (each maps to a D-06 layer; RESEARCH lines 503-510):

| Test file | Comparator | Import |
|-----------|-----------|--------|
| `model_text_roundtrip.rs` | `compare_exact_bytes` | `oracle_harness::comparator::compare_exact_bytes` |
| `predict_raw_parity.rs` | `compare_within` (ORACLE_TOL) | `oracle_harness::{compare_within, ORACLE_TOL}` |
| `predict_transform.rs` | `compare_within` | `oracle_harness::{compare_within, ORACLE_TOL}` |
| `predict_leaf_parity.rs` | `compare_exact_u32` | `oracle_harness::comparator::compare_exact_u32` |
| `predict_subrange.rs` | `compare_within` | `oracle_harness::{compare_within, ORACLE_TOL}` |

> **Import nuance (verified):** `compare_within`, `ORACLE_TOL`, `Mismatch`,
> `abs_diff_within` are re-exported at the `oracle_harness` crate root
> (`oracle-harness/src/lib.rs:10`). The `compare_exact_u32` / `compare_exact_bytes`
> / `compare_exact_f64_bits` functions are NOT re-exported at the root — they must be
> imported via the full module path `oracle_harness::comparator::compare_exact_*`
> (as `example_dataset_parity.rs:32` does). Either extend the lib re-export OR use
> the module path; the analog uses the module path.

---

### `xtask/src/main.rs` — new `model-capture` subcommand

**Analog:** `xtask/src/main.rs` `bin_capture()` (lines 186-318) — verbatim shape.

Copy the entire `bin_capture` body structure for `model_capture`:
1. `workspace_root()` + `verify_toolchain()` (`main.rs:187-188`).
2. Resolve fixture dir under the TRACKED crate (`crates/lgbm-model/tests/fixtures/models/`),
   `create_dir_all`, build per-fixture paths (`main.rs:206-223`).
3. Configure + build the C++ capture target via CMake (`main.rs:224-246`) — OR shell
   out to pip `lightgbm` (RESEARCH Open Q2 path B, `03-RESEARCH.md:448-454`; the
   capture-path decision is the FIRST planning gate).
4. Run the capture, assert each output file was written (`main.rs:248-293`).
5. `write_manifest()` refresh (`main.rs:296-300`), then print the idempotency check
   reminder (`main.rs:313-316`).

Add `Some("model-capture") => model_capture(),` to the `match` dispatch
(`main.rs:53-63`) and update the usage/`bail!` strings to list the new subcommand.

Recorded-constants idiom (`main.rs:22-49`): add a `MODEL_*` master-seed/version
constant block above `main()` if the capture is randomized, kept in sync with the
manifest (the pattern the existing `MASTER_SEED`/`BIN_MASTER_SEED` follow).

> **Capture-feasibility note (RESEARCH Open Q2, `03-RESEARCH.md:448-454`):**
> `lib_lightgbm` is unbuildable (empty `external_libs`) AND no `lightgbm` is
> installed. The existing `bin_capture` resolved this via HEADER-ONLY VERBATIM
> TRANSCRIPTION (see `REFERENCE_MANIFEST.md` "Capture-harness resolution"). Phase 3
> needs a TRAINED model `.txt` as input, which transcription alone cannot produce —
> so the recommended path is (B) install pip `lightgbm` to TRAIN + emit reference
> `.txt`/predict vectors, commit them. This is a planning gate, NOT an established
> analog. The `xtask/cpp/{rng_capture,bin_capture}.cpp` transcription harnesses are
> the analog ONLY if the writer is transcribed (path A).

---

### `Cargo.toml` (workspace members) — modify

**Analog:** the existing `[workspace].members` array (root `Cargo.toml`).

Append `"crates/lgbm-model",` to the `members` list (RESEARCH line 96-98, CONTEXT
integration point line 100). No other root change — `thiserror`/`anyhow` are already
in `[workspace.dependencies]`.

---

### `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` — extend

**Analog:** `xtask/src/main.rs` `write_manifest()` (lines 386-560) — the manifest is
GENERATED, not hand-edited.

The manifest is written by `write_manifest()` as a `format!` string keyed off the
recorded constants. To extend it for the model/predict corpus, ADD a new section to
that `format!` string (mirroring the existing "Numeric Binning Golden Set" /
"EFB Grouping Golden Set" sections, `main.rs:463-547`) documenting: the model corpus
(D-05: regression/binary/multiclass/categorical/subrange), the exact `lightgbm`
version + train params (Open Q2 path B), and the capture-harness resolution note.
Pin via constants like the existing `LIGHTGBM_COMMIT`/`LIGHTGBM_VERSION`
(`main.rs:46-49`). Do NOT edit the `.md` by hand — regen owns it (idempotency: empty
`git diff` after re-running `xtask model-capture`).

## Shared Patterns

### Faithful 1:1 C++ parallel-array mirror (D-04)
**Source:** `crates/lgbm-dataset/src/bin_mapper.rs:1-37, 97-120`
**Apply to:** `tree.rs`, `ensemble.rs` (and the array-layout fields the parser/writer fill)
- Module doc names the exact C++ source file + line ranges and states "transcribed line-for-line."
- Struct fields keep trailing-underscore C++ names, each with a `/// C++ <type> <name>` doc.
- Every precision/arithmetic decision (f64 vs f32, tie direction) is documented inline.
- Small `#[inline] fn` helpers map one-to-one onto single C++ functions, each citing its `common.h`/`tree.h` line.

### thiserror boundary errors, never panic (CLAUDE.md mandate, Security V5)
**Source:** `crates/lgbm-dataset/src/error.rs` (whole) + `crates/lgbm-core/src/error.rs:18-58`
**Apply to:** `error.rs`, and every fallible path in `model_text.rs` (parse) + `predict.rs` (shape validation)
```rust
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelError {
    #[error("malformed model text: {detail}")]
    MalformedModel { detail: String },
}
```
- One variant per V5 input-validation class. Map each C++ `Log::Fatal`/`CHECK_*` to a `Result::Err`, never `panic!`/unchecked `[]`.
- Inline `#[cfg(test)] mod tests` asserting each variant's Display is non-empty + contains its detail.

### Reuse workspace constants/types — never redefine
**Source:** `crates/lgbm-dataset/src/bin_mapper.rs:40-41`
**Apply to:** `tree.rs` (`K_ZERO_THRESHOLD` for `is_zero`), `predict.rs` (`ScoreT`=f32 at the boundary)
```rust
use lgbm_core::types::K_ZERO_THRESHOLD;
```
`K_ZERO_THRESHOLD` (1e-35 f64) and `K_EPSILON` (1e-15 f32) live in `lgbm_core::types` (`types.rs:31,35`).

### Committed-golden replay test (CARGO_MANIFEST_DIR + graceful SKIP)
**Source:** `crates/lgbm-dataset/tests/example_dataset_parity.rs:38-44, 182-193`
**Apply to:** all five `tests/*.rs`
- Fixture paths via `env!("CARGO_MANIFEST_DIR")`, never absolute, never under `LightGBM/`.
- `let Ok(text) = read_to_string(...) else { eprintln!("SKIP ..."); return; }` so tests pass pre-capture.
- Localizing assert messages naming fixture + row/tree/class index.

### Keyed text parse with pull-with-default (NOT positional)
**Source:** `crates/lgbm-dataset/tests/golden/mod.rs:70-186`
**Apply to:** `model_text.rs` (parser), and the goldens' test loader
- Parse into tokens, pull fields by `key=` prefix (`field()` helper), default absent conditional fields.
- `parse_f64_bits_list`/`parse_u32_list` idiom for `;`-separated array fields (the predict-vector goldens).

### Bit/byte-exact vs ~1e-6 split (D-06 layered goldens)
**Source:** `crates/oracle-harness/src/comparator.rs` (whole) + `example_dataset_parity.rs:32`
**Apply to:** all five tests
- Discrete artifacts (model-text bytes → `compare_exact_bytes`; leaf indices → `compare_exact_u32`) are bit/byte-exact.
- Continuous scores → `compare_within(.., ORACLE_TOL)` (1e-6 f32, `comparator.rs:15`).
- `compare_exact_*` are NOT root-re-exported — import via `oracle_harness::comparator::compare_exact_*`.

### Idempotent C++-regen capture subcommand (no toolchain at test time)
**Source:** `xtask/src/main.rs` `bin_capture()` (186-318) + `write_manifest()` (386-560)
**Apply to:** the new `model-capture` subcommand
- `workspace_root()` + `verify_toolchain()` → configure/build/run capture → assert outputs written → refresh manifest → print idempotency reminder.
- Recorded-constant master seeds; goldens are a pure function of constants (empty `git diff` on re-run).
- Fixtures written under the TRACKED crate dir, COPIED from C++ output — never referencing the untracked `LightGBM/` tree.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/lgbm-model/src/format.rs` | utility | transform | No `%.17g`/`{:g}` printf-style float formatter exists anywhere in the workspace. It is the serialization-parity linchpin and is genuinely new. Use the RESEARCH spec (`03-RESEARCH.md:295-299, 461-463`) + the inline-`#[cfg(test)]` battery-test idiom from `comparator.rs:191-226`. Build & verify FIRST (Wave 0 gate). |

> The `model-capture` C++ harness itself (path B: pip `lightgbm` training) also has
> no in-repo analog — the existing `xtask/cpp/*.cpp` are transcription harnesses, and
> the trained-model-input requirement (RESEARCH Open Q2) means the capture path is a
> planning gate, not a copyable precedent.

## Metadata

**Analog search scope:** `crates/lgbm-core/`, `crates/lgbm-dataset/`,
`crates/oracle-harness/`, `xtask/`, root `Cargo.toml`.
**Files scanned:** ~14 (Cargo.toml ×3, lib.rs ×2, error.rs ×2, bin_mapper.rs,
ingest.rs, comparator.rs, golden/mod.rs, example_dataset_parity.rs, xtask main.rs,
types.rs, config/mod.rs grep).
**Pattern extraction date:** 2026-06-05
**Hard constraint recorded:** `LightGBM/` is git-untracked + unbuildable; goldens are
C++-generated then committed under `tests/fixtures/`, resolved via `CARGO_MANIFEST_DIR`.
