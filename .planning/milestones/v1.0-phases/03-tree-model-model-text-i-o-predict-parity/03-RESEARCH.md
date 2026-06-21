# Phase 3: Tree Model + Model Text I/O + Predict Parity - Research

**Researched:** 2026-06-05
**Domain:** Model-text (de)serialization + tree/GBDT prediction-math faithful C++ port (pure Rust)
**Confidence:** HIGH (all behavior verified against the pinned in-repo C++ source; no library-version uncertainty — this is a hand-port, not a dependency-selection problem)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** The Rust model-text writer must be **byte-identical to C++ `GBDT::SaveModelToString`** for the same model. Key ordering, whitespace/newlines, `tree_sizes=` block, `feature_infos=`, and `%.17g` float formatting all match exactly. Makes load→predict→write→reload **trivially byte-stable**.
- **D-01a:** Because the write contract is byte-exact, the SC#3 "bin mappers / feature metadata" wording is subsumed: the writer emits **whatever C++ emits** — notably `feature_infos=` per-feature min/max-range metadata, NOT full `BinMapper` bin-boundary arrays. The byte-identical golden is the arbiter. *(VERIFIED below: `feature_infos=` is per-feature `[min:max]` for numerical / `cat:cat:...` for categorical — confirmed in `bin.h:224` `bin_info_string()`.)*
- **D-02:** Phase 3 predicts on **dense + CSR/CSC** input matrices, reusing the Phase-2 ingest forms. **Single-row** prediction deferred to Phase 8; SHAP/early-stop are Phase 7.
- **D-02a:** Prediction runs on **raw feature values** using the model's stored real thresholds / categorical bitsets (the loaded-model path), NOT by re-binning through a `BinMapper`. Deterministic CPU path is **single-threaded** (`num_threads=1` reference).
- **D-03:** **One new `lgbm-model` crate** holds the entire subsystem: `Tree` + GBDT-ensemble representation, model-text load/save, and the predictor.
- **D-04:** The in-memory `Tree` is a **faithful 1:1 mirror of C++ `tree.h`**: parallel arrays (`split_feature_`, `threshold_`, `decision_type_`, `left_child_`/`right_child_`, `leaf_value_`, `leaf_count_`, `internal_value_`, `cat_boundaries_`/`cat_threshold_`, …) rather than an idiomatic Rust node enum. GBDT mirrors `models_` (flat tree list with per-iteration/per-class indexing).
- **D-05:** Committed golden corpus covers **regression + binary + multiclass + categorical-split + sub-range** C++-trained models. Scope-bounded to **core** objectives only.
- **D-06:** **Layered golden granularity per fixture:** (1) model-text round-trip bytes; (2) raw score f32 ~1e-6; (3) transformed output f32 ~1e-6; (4) leaf-index exact integer; (5) sub-range raw scores.
- **D-07:** Fixtures produced by **C++ training a model and emitting its `.txt`** via the golden-capture xtask pipeline; generate once, **commit**, replay with no C++ toolchain at test time. If full-lib linkage remains infeasible (`external_libs` unvendored), fall back to **header-only / verbatim transcription** capture, human-approved.

### Claude's Discretion
- Exact `lgbm-model` internal module layout; the precise `Tree`/ensemble field set and accessor shape (bounded by "faithful `tree.h` mirror"); the model-text parser's tokenizer strategy (bounded by "byte-identical writer + exact-parse round-trip"); golden file formats; the predictor's dense-vs-sparse iteration structure; how `decision_type_` bit flags are decoded — all bounded by "faithful C++ mirror, ~1e-6 f32 scores, byte-exact model text." **When C++ behavior is the spec, the C++ source is authoritative over any inferred default.**

### Deferred Ideas (OUT OF SCOPE)
- **Single-row prediction** (per-row `PredictFunction`) — Phase 8.
- **SHAP / `predict_contrib` (PRD-04)** and **prediction early stopping (PRD-05)** — Phase 7.
- **Non-core objective `ConvertOutput`** (huber/fair/poisson/quantile/mape/gamma/tweedie, cross-entropy, ranking) — Phase 7.
- **`DumpModel` JSON + `ModelToIfElse` C++ codegen** — out of scope.
- **Feature-importance reporting (ADV-07)** — Phase 7. *(NOTE: see Pitfall 7 — `SaveModelToString` DOES emit a `feature_importances:` block, so the WRITER must reproduce it; only the public reporting API is deferred.)*
- **Parallel (rayon) prediction** — Phase 3 ships single-threaded deterministic.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DAT-08 | Model text **read** — parse a C++-trained `.txt` into the in-memory model | `GBDT::LoadModelFromString` (`gbdt_model_text.cpp:421`) + `Tree(const char* str, size_t* used_len)` (`tree.cpp:685`). Full schema + parse semantics enumerated below. |
| DAT-09 | Model text **write** — byte-identical to C++ `SaveModelToString` incl. `%.17g` | `GBDT::SaveModelToString` (`gbdt_model_text.cpp:311`) + `Tree::ToString` (`tree.cpp:339`) + `CommonC::ArrayToString`/`{:.17g}` (`common.h:1239`). Section order + formatting fully specified below. |
| PRD-01 | Raw-score prediction (sum of tree outputs) | `GBDT::PredictRaw` (`gbdt_prediction.cpp:13`) + `Tree::Predict`/`GetLeaf` (`tree.h:587,701`). |
| PRD-02 | Transformed prediction (`ConvertOutput`) | `GBDT::Predict` (`gbdt_prediction.cpp:55`) + core `ConvertOutput` (regression identity/sqrt, binary sigmoid, multiclass softmax, multiclassova per-class sigmoid). |
| PRD-03 | Leaf-index prediction (`pred_leaf`) | `GBDT::PredictLeafIndex` (`gbdt_prediction.cpp:79`) + `Tree::PredictLeafIndex` (`tree.h:650`). |
| PRD-06 | Sub-range prediction (`start_iteration`/`num_iteration`) | `GBDT::InitPredict` (`gbdt.h:426`) sets `start_iteration_for_pred_`/`num_iteration_for_pred_`; `-1 == all` semantics. |
</phase_requirements>

## Summary

Phase 3 is a **pure hand-port problem**, not a library-selection problem: there is no external crate to choose — the entire subsystem is a faithful 1:1 mirror of three C++ files (`tree.h`/`tree.cpp` for the per-tree representation + serialization + predict-math, `gbdt_model_text.cpp` for the ensemble-level model-text envelope, `gbdt_prediction.cpp` for the prediction loop) plus the core `ConvertOutput` transforms from three objective headers. All behavior was verified directly against the pinned in-repo C++ source (commit-pinned under `LightGBM/`, VERSION 4.6.0.99). There are zero `[ASSUMED]` claims in this research.

The single highest-risk surface is **DAT-09 byte-exact write** (SC#3). It hinges on `%.17g` float formatting produced by the C++ `fmt` library (`CommonC::ArrayToString` → `__TToStringHelper<T,true,true>` → `fmt::format_to_n(buf, len, "{:.17g}", value)`). Rust's `{:e}`/`{}`/`ryu` do **not** byte-match C `%.17g`/`fmt {:.17g}` in general (shortest-round-trip vs fixed-17-significant-digit semantics differ). The plan must reproduce `%.17g` exactly. **The key de-risking insight (verified below): for a load→write round-trip, the ensemble-level metadata strings (`feature_infos=`, `feature_names=`, the entire `parameters:` block) are loaded verbatim as strings and re-emitted unchanged — only the per-tree float arrays (`leaf_value`, `threshold`, `leaf_weight`, `internal_value`, `internal_weight`, `leaf_const`, `leaf_coeff`, `split_gain`) actually round-trip through parse(double)→format. So byte-stability of the round-trip reduces to: (a) preserve metadata strings verbatim, and (b) reproduce `%.17g` for the per-tree float arrays only.**

The second-highest risk is **predict-math fidelity**: `Tree::Predict` traverses on `double` feature values using stored `double` thresholds and `double` leaf values, accumulated in a `double` per-class output — the f32 (~1e-6) contract applies only at the comparison boundary (output is cast/compared as f32). Missing-value routing (None/Zero/NaN via the `decision_type_` 2-bit `missing_type` field), default-left routing (bit 1), and categorical bitset lookup (`FindInBitset`) are all small but exact-match-critical decode paths.

**Primary recommendation:** Build `lgbm-model` as four faithful modules — `tree` (array repr + `Predict`/`GetLeaf`/decision decode + `ToString`/parse), `ensemble` (GBDT `models_` + per-iteration/per-class indexing + `InitPredict` sub-range), `model_text` (the `SaveModelToString`/`LoadModelFromString` envelope, preserving metadata strings verbatim on round-trip), and `predict` (batch dense/CSR/CSC driver + the four core `ConvertOutput`s). Implement a Rust `%.17g` formatter first (it is the linchpin; write a dedicated parity test against captured doubles). Capture goldens via **header-only / verbatim transcription** of `SaveModelToString`+`Tree::ToString` (the full lib is unbuildable here — `external_libs/{fmt,fast_double_parser}` are empty), OR install an external `lightgbm` (pip/CLI) to emit reference `.txt` models — neither is present today (verified), so the capture-path decision is the first planning gate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Model-text parse (DAT-08) | `lgbm-model::model_text` + `::tree` | `lgbm-core` (Config/error) | Envelope keys parsed at ensemble level; per-tree block parsed by the `Tree` parse-ctor. |
| Model-text write byte-exact (DAT-09) | `lgbm-model::model_text` + `::tree` | — | `SaveModelToString` order at ensemble level; `Tree::ToString` per tree; both own their float formatting (`%.17g`). |
| `%.17g` float formatting | `lgbm-model` util (new) | — | Serialization-parity linchpin; no existing crate provides it. |
| Raw-score predict (PRD-01) | `lgbm-model::predict` + `::tree` | — | Ensemble loops trees; `Tree::Predict` does the traversal. Pure CPU, single-threaded. |
| Decision/missing/categorical decode | `lgbm-model::tree` | — | Hot inner traversal; mirrors `NumericalDecision`/`CategoricalDecision`. |
| `ConvertOutput` transforms (PRD-02) | `lgbm-model::predict` (objective shim) | `lgbm-core::Config` (objective string, sigmoid, num_class) | Phase 3 needs only the *output transform*, parsed from the model's `objective=` line — NOT the full objective machinery (grad/hess is Phase 6). |
| Leaf-index predict (PRD-03) | `lgbm-model::predict` + `::tree` | — | `PredictLeafIndex` returns the `~node` leaf id. |
| Sub-range slicing (PRD-06) | `lgbm-model::ensemble` | — | `InitPredict` computes `start_iteration_for_pred_`/`num_iteration_for_pred_`. |
| Prediction input ingest (dense/CSR/CSC) | `lgbm-dataset` (reuse) | `lgbm-model::predict` | Reuse Phase-2 `from_mat`/`from_csr`/`from_csc` row materialization; D-02a uses RAW values, no re-binning. |

## Standard Stack

**This phase introduces NO new external runtime dependencies.** It is a hand-port. The "stack" is the existing workspace crates plus one internal formatting utility.

### Core (existing workspace crates — reuse, do not re-add)
| Crate | Version | Purpose | Why Standard |
|-------|---------|---------|--------------|
| `lgbm-core` | workspace (path) | `Config` (objective, num_class, sigmoid, start_iteration), `types.rs` (f32), `error.rs` (`thiserror` idiom), `Random` (not needed for predict, but available) | Single config/types/error source (Phase 1 lock). Extend `error.rs` with a `ModelError` variant family. |
| `lgbm-dataset` | workspace (path) | `from_mat`/`from_csr`/`from_csc` (predict inputs), `Metadata`, `BinMapper`/`FinishedDataset` | Phase-2 ingest forms reused verbatim for D-02 prediction inputs. |
| `oracle-harness` | workspace (path) | `compare_within` (f32 ~1e-6 via `ORACLE_TOL`), `compare_exact_u32` (leaf index), `compare_exact_bytes` (model text), `Mismatch` | The exact comparator seam each D-06 layer plugs into — all three comparators already exist (`comparator.rs:92,125,172`). |

### Supporting (Rust std only)
| Item | Purpose | When to Use |
|------|---------|-------------|
| `std::fmt` / custom `%.17g` formatter | Byte-exact float serialization | DAT-09 writer for per-tree float arrays. **Must NOT use `ryu`/`{}`/`{:e}` — they produce shortest-round-trip, not 17-sig-digit `%.17g`.** |
| `std::str::FromStr` for `f64`/`f32`/`i32` | Parse-back of arrays | DAT-08 parser. Rust `f64::from_str` round-trips `%.17g` output exactly (17 sig digits is > the 17 needed for IEEE-754 double round-trip), so parse is low-risk. See Pitfall 6. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Faithful parallel-array `Tree` (D-04) | Idiomatic Rust node enum / `Box`-linked tree | REJECTED by D-04: array layout + traversal order is part of the parity surface and maps 1:1 onto the model text. An enum would force a translation layer at parse/write that re-introduces divergence risk. |
| Custom `%.17g` formatter | `ryu` crate, or `format!("{:.17e}")` | `ryu` = shortest round-trip (e.g. `0.1` not `0.10000000000000001`); `{:.17e}` = always-exponential + always-17-after-decimal. Neither matches C `%g` semantics (strip trailing zeros, switch to exponent only outside `[1e-4, 1e17)`, 17 *significant* digits). Must hand-roll or find a verified `printf`-compatible `%g`. |
| Re-bin predict inputs through `BinMapper` | (the inner/bin predict path `NumericalDecisionInner`) | REJECTED by D-02a: load-from-model predict uses the *real-value* path (`Tree::Predict`→`NumericalDecision` on `double` thresholds), NOT the bin path. The bin path is a training-time optimization (Phase 5). |

**Installation:** None. Add the crate to the workspace:
```toml
# root Cargo.toml [workspace].members  — append:
    "crates/lgbm-model",
```

**Version verification:** N/A — no external packages. Verified `cargo` workspace edition 2024, rust-version 1.95 (`[VERIFIED: crates/lgbm-core/Cargo.toml + root Cargo.toml]`).

## Package Legitimacy Audit

> Not applicable. This phase installs **no external packages** — it is a pure intra-workspace hand-port. All dependencies are `path = "../..."` workspace crates already present and audited in Phases 1–2. slopcheck/registry verification is moot.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none) | — | No external installs in this phase |

## Architecture Patterns

### System Architecture Diagram

```
                          model.txt  (C++-trained, committed fixture)
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │  lgbm-model::model_text::load (DAT-08)            │
        │  GBDT::LoadModelFromString envelope:              │
        │    parse header keys (num_class, objective, …)    │
        │    parse feature_infos / feature_names VERBATIM ──┼──► keep as Strings
        │    parse tree_sizes → per-tree byte boundaries    │     (round-trip)
        │    for each "Tree=i" block:                       │
        │       Tree(str,&used_len) ─► parallel arrays      │
        │    capture loaded_parameter_ block VERBATIM ──────┼──► keep as String
        └───────────────┬─────────────────────────────────┘
                        │  GbdtModel { trees[], num_class, num_tree_per_iteration,
                        │              objective_string, feature_infos[], … }
        ┌───────────────┴───────────────┬───────────────────────────┐
        ▼                                ▼                           ▼
┌──────────────────┐      ┌──────────────────────────┐   ┌────────────────────┐
│ predict (PRD-01) │      │ model_text::save (DAT-09) │   │ predict_leaf (PRD-3)│
│ InitPredict      │      │ SaveModelToString order:  │   │ PredictLeafIndex    │
│  → start/num_pred│      │  header → feature_infos=  │   │  → ~node leaf id    │
│ for iter in range│      │  (verbatim) → tree_sizes= │   │  (exact integer)    │
│  for k in class: │      │  → "" → Tree=i+ToString() │   └────────────────────┘
│   out[k]+=Tree   │      │  → "end of trees"         │
│        .Predict()│      │  → feature_importances:   │   inputs: dense/CSR/CSC
│ avg if RF        │      │  → parameters: (verbatim) │   (lgbm-dataset, RAW vals,
│ ConvertOutput    │      │  reproduce %.17g per tree │    D-02a: no re-binning)
│  (PRD-02)        │      └──────────────────────────┘
└──────────────────┘
        │                              │
        ▼ f32 ~1e-6                     ▼ byte-exact
  oracle-harness::compare_within  oracle-harness::compare_exact_bytes
```

### Recommended Project Structure
```
crates/lgbm-model/
├── Cargo.toml              # deps: lgbm-core, lgbm-dataset (path); dev: oracle-harness
├── src/
│   ├── lib.rs              # pub re-exports; ModelError (thiserror)
│   ├── tree.rs             # Tree: parallel arrays (D-04), Predict/GetLeaf,
│   │                       #   NumericalDecision/CategoricalDecision, decision_type
│   │                       #   bit decode, ToString + parse-from-str ctor
│   ├── ensemble.rs         # GbdtModel: models_ (Vec<Tree>), num_class,
│   │                       #   num_tree_per_iteration, InitPredict sub-range state
│   ├── model_text.rs       # load (LoadModelFromString) + save (SaveModelToString)
│   │                       #   envelope; verbatim metadata strings on round-trip
│   ├── predict.rs          # batch dense/CSR/CSC driver, raw + leaf-index;
│   │                       #   objective shim (core ConvertOutput only)
│   └── format.rs           # %.17g + {:g} formatter (the serialization linchpin)
└── tests/
    ├── model_text_roundtrip.rs   # D-06 layer 1 (compare_exact_bytes)
    ├── predict_raw_parity.rs     # D-06 layer 2 (compare_within)
    ├── predict_transform.rs      # D-06 layer 3 (compare_within)
    ├── predict_leaf_parity.rs    # D-06 layer 4 (compare_exact_u32)
    └── predict_subrange.rs       # D-06 layer 5 (compare_within)
```
*(Module names are Claude's discretion per CONTEXT; this is one faithful layout.)*

### Pattern 1: Faithful parallel-array Tree (D-04)
**What:** Mirror `tree.h` member arrays exactly; predict walks node indices, leaves encoded as `~node` (bitwise-NOT) negatives.
**When to use:** The whole `Tree` type.
**Example (verified from C++):**
```rust
// Source: LightGBM/include/LightGBM/tree.h:478-541 (member layout)
//         + tree.h:701-713 (GetLeaf), 587-615 (Predict)
pub struct Tree {
    num_leaves: i32,
    num_cat: i32,
    left_child: Vec<i32>,      // len num_leaves-1; negative = ~leaf
    right_child: Vec<i32>,
    split_feature: Vec<i32>,   // ORIGINAL feature index (split_feature_, not _inner_)
    threshold: Vec<f64>,       // real-value threshold (predict path)
    decision_type: Vec<i8>,    // bit0=categorical, bit1=default_left, bits2-3=missing_type
    split_gain: Vec<f32>,
    leaf_value: Vec<f64>,      // len num_leaves
    leaf_weight: Vec<f64>,
    leaf_count: Vec<i32>,
    internal_value: Vec<f64>,  // len num_leaves-1
    internal_weight: Vec<f64>,
    internal_count: Vec<i32>,
    cat_boundaries: Vec<i32>,  // len num_cat+1
    cat_threshold: Vec<u32>,   // bitset blocks
    shrinkage: f64,
    is_linear: bool,           // Phase 3: always false for in-scope models
}

// GetLeaf: node starts at 0; while node >= 0, descend; return ~node.
fn get_leaf(&self, fv: &[f64]) -> i32 {
    let mut node = 0i32;
    if self.num_cat > 0 {
        while node >= 0 { node = self.decision(fv[self.split_feature[node as usize] as usize], node); }
    } else {
        while node >= 0 { node = self.numerical_decision(fv[self.split_feature[node as usize] as usize], node); }
    }
    !node  // C++ ~node
}
```

### Pattern 2: `decision_type_` bit decode (the routing linchpin)
**What:** Per-node `int8` packs three facts. Verified from `tree.h:20-21,262-281,337-355`.
```rust
// Source: tree.h  #define kCategoricalMask (1)  #define kDefaultLeftMask (2)
const CATEGORICAL_MASK: i8 = 1;
const DEFAULT_LEFT_MASK: i8 = 2;
fn get_decision_type(dt: i8, mask: i8) -> bool { (dt & mask) > 0 }
fn get_missing_type(dt: i8) -> u8 { ((dt >> 2) & 3) as u8 }  // 0=None,1=Zero,2=NaN

// NumericalDecision (tree.h:337-355), feature value is f64:
fn numerical_decision(&self, mut fval: f64, node: usize) -> i32 {
    let mt = get_missing_type(self.decision_type[node]);
    if fval.is_nan() && mt != MISSING_NAN { fval = 0.0; }          // NaN→0 unless NaN-type
    if (mt == MISSING_ZERO && is_zero(fval)) || (mt == MISSING_NAN && fval.is_nan()) {
        return if get_decision_type(self.decision_type[node], DEFAULT_LEFT_MASK)
               { self.left_child[node] } else { self.right_child[node] };
    }
    if fval <= self.threshold[node] { self.left_child[node] } else { self.right_child[node] }
}
// is_zero: tree.h:254  fval >= -kZeroThreshold && fval <= kZeroThreshold ; kZeroThreshold=1e-35
```
**Anti-pattern:** Decoding `missing_type` as a single bit, or forgetting the `NaN→0.0` coercion when `missing_type != NaN`. Both silently mis-route a fraction of rows → ~1e-6 fails only on NaN/zero-bearing fixtures.

### Pattern 3: CategoricalDecision via bitset (verified tree.h:374-390, common.h:836)
```rust
// fval<0 or NaN → right_child. cat_idx = threshold[node] as i32.
// FindInBitset over cat_threshold[cat_boundaries[cat_idx] .. cat_boundaries[cat_idx+1]]
fn find_in_bitset(bits: &[u32], pos: i32) -> bool {     // common.h:836
    let i1 = (pos / 32) as usize;
    if i1 >= bits.len() { return false; }
    (bits[i1] >> (pos % 32)) & 1 == 1
}
```

### Pattern 4: Ensemble predict loop + sub-range (PRD-01/06)
```rust
// Source: gbdt_prediction.cpp:13-32 (PredictRaw) + gbdt.h:426-436 (InitPredict)
// InitPredict (call once per predict request):
//   total = models.len() / num_tree_per_iteration
//   start = clamp(start_iteration, 0, total)
//   num_for_pred = if num_iteration > 0 { min(num_iteration, total - start) } else { total - start }
//   => start_iteration_for_pred = start
// PredictRaw: output[0..num_tree_per_iteration] = 0
//   for i in start .. start+num_for_pred:
//     for k in 0..num_tree_per_iteration:
//        output[k] += models[i*num_tree_per_iteration + k].predict(features)   // f64 accumulate
```
**Key:** `num_tree_per_iteration_` = 1 for regression/binary, = num_class for multiclass — drives BOTH per-class output indexing AND the `models_[i*ntpi + k]` stride.

### Anti-Patterns to Avoid
- **Accumulating predictions in f32.** C++ accumulates `output[k]` in `double` (`gbdt_prediction.cpp:16` `double* output`). Use `f64` internally, cast to `f32` only at the comparison boundary, or the ~1e-6 contract drifts on deep ensembles.
- **Using `split_feature_inner_` for the real-value predict path.** `Tree::GetLeaf` (real-value) uses `split_feature_` (original index); `_inner_` is for the bin path only (training). The model text emits `split_feature` = `split_feature_` (original). (tree.cpp:347-348, tree.h:705.)
- **Reformatting metadata strings on write.** `feature_infos`, `feature_names`, and the `parameters:` block round-trip as opaque strings — reformatting them risks byte divergence (and they use ostream `setprecision(17)`, a DIFFERENT formatter than the per-tree fmt `{:.17g}`). Preserve verbatim.
- **Idiomatic enum tree.** Forbidden by D-04.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense/CSR/CSC row materialization | A new sparse reader | `lgbm-dataset` `from_mat`/`from_csr`/`from_csc` (D-02) | Already bit-identical to C++; re-implementing re-opens ingest parity. **BUT note** the predict path needs RAW feature values per row (D-02a), not binned — confirm what the dataset exposes (see Open Question 1). |
| f32 ~1e-6 / leaf-int / byte comparison | New asserts | `oracle-harness::{compare_within, compare_exact_u32, compare_exact_bytes}` | All three already exist (`comparator.rs`). D-06's five layers map onto exactly these three. |
| Softmax | Naive `exp/sum` | Port `Common::Softmax` **with the max-subtraction** (`common.h:587`) | C++ subtracts `wmax` before `exp` for stability; skipping it diverges at ~1e-6 and on overflow. |
| C-locale float→string | `format!("{}")` / `ryu` | A `%.17g`/`{:g}`-faithful formatter (Pitfall 1) | Shortest-round-trip ≠ printf `%g`. This is THE serialization-parity risk. |
| Config `parameters:` block reproduction | Re-deriving from `lgbm-core::Config` | Preserve `loaded_parameter_` verbatim (C++ does exactly this: `gbdt_model_text.cpp:397-401`) | The write path emits `config_->ToString()` only when training; on a *loaded* model it re-emits the captured string unchanged. Phase 3 only loads, so verbatim preservation is both correct AND avoids porting `Config::SaveMembersToString` (~110 params). |

**Key insight:** The byte-exact-write contract is far less work than it appears *because Phase 3 never trains*. Everything above the per-tree blocks is preserved verbatim from the loaded bytes. The genuine new work is (a) one `%.17g` formatter and (b) the per-tree `ToString` float arrays.

## Runtime State Inventory

> Not a rename/refactor/migration phase — this is greenfield crate creation. Section included only to confirm no hidden runtime state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 3 reads committed `.txt` fixtures only; writes test-scratch output | None |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | None | None |
| Build artifacts | New `crates/lgbm-model` added to workspace `members`; `cargo` recompiles workspace. No stale artifacts. | Add member to root `Cargo.toml`; `cargo build` |

## Common Pitfalls

### Pitfall 1: `%.17g` ≠ Rust default / `ryu` / `{:.17e}` (THE risk for DAT-09)
**What goes wrong:** Rust `f64::to_string()` / `ryu` emit shortest round-trip (`0.1`), `{:.17}` emits 17 *fractional* digits, `{:.17e}` always uses exponent form. C `%g` (== C++ `fmt {:.17g}`) emits **17 significant digits**, strips trailing zeros, and uses exponent form only when the exponent `< -4` or `>= precision(17)`.
**Why it happens:** `CommonC::ArrayToString<true>` → `__TToStringHelper<T,true,true>` → `fmt::format_to_n(buf, len, "{:.17g}", value)` (`common.h:1227`). `fmt`'s `{:.17g}` follows printf `%g`.
**How to avoid:** Implement a `%.17g`-faithful formatter (and a `{:g}` = 6-sig-digit variant for `split_gain`, which uses `__TToStringHelper<T,true,false>` → `{:g}`, `common.h:1220`). Write a dedicated parity test feeding a battery of doubles (subnormals, 1e300, 1e-300, integers-as-double, 0.1, values needing exactly 17 digits) and compare against captured C++ output BEFORE building the tree writer.
**Warning signs:** Round-trip test fails on a single byte mid-array; the differing value is always a "messy" float.

### Pitfall 2: Two different float formatters in one file
**What goes wrong:** `feature_infos=` uses ostream `<< setprecision(digits10+2=17) << min_val_` (`bin.h:229`), while per-tree `threshold`/`leaf_value` use fmt `{:.17g}` (`tree.cpp:352,360`). These can differ for some doubles (ostream default float format vs `%g`).
**Why it happens:** Two independent code paths in C++.
**How to avoid:** On a load→write round-trip, `feature_infos_` is preserved verbatim (never reformatted) — so this difference is invisible to Phase 3. Only matters if the writer ever *generates* `feature_infos` from scratch (it doesn't, until training). Document loudly so Phase 6 doesn't trip on it.

### Pitfall 3: `decision_type` written as int8-cast-to-int
**What goes wrong:** Emitting raw `i8` (e.g. `-126`) instead of the C++ form.
**Why it happens:** `tree.cpp:353-354` writes `ArrayToString(Common::ArrayCast<int8_t,int>(decision_type_), ...)` — casts each `int8_t` to `int` first, then formats as integer. So a `decision_type` of `2` writes `2`; values stay small positive (mask bits 0-3 → 0..15). Parse-back reads `StringToArrayFast<int8_t>` (`tree.cpp:806`).
**How to avoid:** Format `decision_type` as plain integers (the `int` cast), parse back into `i8`. Range is 0..15 in practice.

### Pitfall 4: Tree parse is keyed, order-independent, max 22 header lines
**What goes wrong:** Writing a strict positional parser that breaks if a conditional field (`cat_boundaries`, `leaf_const`) is absent.
**Why it happens:** `Tree(const char* str, size_t* used_len)` (`tree.cpp:685`) reads up to `max_num_line=22` `key=value` lines into a map, stops at blank line, then pulls fields by key with per-field fallbacks (e.g. `split_gain` defaults to zeros if absent). `cat_*` only present when `num_cat>0`; `leaf_*`/linear only when `is_linear`. The early-return `if (num_leaves_<=1 && !is_linear_) return;` (`tree.cpp:747`) means a single-leaf tree has ONLY `num_leaves`/`num_cat`/`leaf_value`/`shrinkage`/`is_linear`/`leaf_count`.
**How to avoid:** Parse into a key→value map first (like C++), then pull-with-default. Handle the single-leaf early return. **Crucially, `*used_len` = bytes consumed up to and including the blank line that terminates the header** — the ensemble loader advances by `tree_sizes[i]` instead (when present), so the per-tree `used_len` is only the fallback path.
**Warning signs:** Multiclass/categorical fixtures fail to parse while regression passes.

### Pitfall 5: `tree_sizes` drives parse boundaries (parallel in C++, sequential here)
**What goes wrong:** Computing tree byte boundaries wrong → mis-sliced trees.
**Why it happens:** When `tree_sizes=` is present (always, for v4 models), C++ computes `tree_boundaries[i+1] = tree_boundaries[i] + tree_sizes[i]` and parses tree `i` starting at `p + tree_boundaries[i]` (`gbdt_model_text.cpp:546-572`). `tree_sizes[i]` = byte length of the `"Tree=i\n" + ToString() + "\n"` string (`gbdt_model_text.cpp:360-362`). The `+'\n'` after `ToString()` (which itself ends in `\n\n`) matters for the byte count.
**How to avoid:** On WRITE, compute each tree's serialized string exactly (`"Tree=" + idx + "\n" + tree.to_string() + "\n"`) and its byte length → `tree_sizes`. On READ, honor `tree_sizes` boundaries. Single-threaded is fine (D-02a) — the C++ `#pragma omp` is just speed.
**Warning signs:** Round-trip `tree_sizes=` line differs by a few bytes per tree.

### Pitfall 6: Parse precision — `StringToArray` (precise) vs `StringToArrayFast` (lossy)
**What goes wrong:** Using one parser for all arrays.
**Why it happens:** C++ deliberately uses `StringToArray<double>` (precise `Atof`) for `threshold`, `leaf_value`, `leaf_weight`, `leaf_const`, `leaf_coeff` (`tree.cpp:718,770,800,813,835`) but `StringToArrayFast` (less-precise `Atof`) for `split_gain`, `internal_*`, `decision_type` (`tree.cpp:776,782,788,794,806`). For Phase 3 prediction, only `threshold`/`leaf_value` affect scores — both use the PRECISE parser. Rust `f64::from_str` is correctly-rounded (≥ precise `Atof`), so parse-back is safe for the predict-relevant fields.
**How to avoid:** Use Rust `f64::from_str`/`f32::from_str` (correctly rounded) everywhere — it meets or exceeds both C++ parsers' precision. The only risk is byte-exact *re-write*, which is handled by the `%.17g` formatter, not the parser. Document that `internal_value`/`split_gain` are NOT predict-relevant (they're metadata) so a tiny parse difference there cannot break PRD-* (and `%.17g` re-write from the precise-parsed value reproduces the original string).
**Warning signs:** N/A for predict; only relevant if a write of a `Fast`-parsed field diverges — verify on round-trip.

### Pitfall 7: `feature_importances:` block IS in the write output
**What goes wrong:** Treating ADV-07 "feature importance deferred to Phase 7" as meaning the writer omits it. `SaveModelToString` ALWAYS emits a `feature_importances:` block (`gbdt_model_text.cpp:389-392`), computed via `FeatureImportance(num_iteration, type)`, sorted descending by count, only features with importance > 0.
**Why it happens:** The block is co-located in `gbdt_model_text.cpp` but is part of the model text envelope, not the deferred reporting API.
**How to avoid:** For byte-exact write of a *loaded* model: the loader does NOT capture/recompute feature importances (it's derived from the trees). The simplest faithful path is to **recompute** `FeatureImportance` (split-count or gain) over the loaded trees on write — OR, since the round-trip golden is the arbiter, verify whether the committed reference `.txt` was produced with the same `feature_importance_type` and reproduce that computation. **This is an Open Question (see below) — confirm the importance type and whether load preserves vs recomputes.** Note `FeatureImportance` itself is a small loop over `split_feature_`/`split_gain_` (`gbdt_model_text.cpp:627`), portable, but it counts splits per ORIGINAL feature — verify against the golden.
**Warning signs:** Round-trip diverges in the trailing `feature_importances:` block while trees match.

### Pitfall 8: `models_.size() % num_tree_per_iteration` and `num_iteration_for_pred_`
**What goes wrong:** Off-by-stride on multiclass leaf-index output sizing.
**Why it happens:** `NumPredictOneRow` for leaf index = `num_class * num_iterations_in_range` (`gbdt.h:281-291`). The output vector for `pred_leaf` is `num_tree_per_iteration * num_iteration_for_pred` long, laid out `[iter0_class0, iter0_class1, ..., iter1_class0, ...]` (`gbdt_prediction.cpp:79-86`).
**How to avoid:** Size and index leaf-index output exactly per `gbdt.h`/`gbdt_prediction.cpp`. Cover with the multiclass fixture's layer-4 golden.

## Code Examples

### Core `ConvertOutput` transforms (PRD-02) — verified
```rust
// Source: regression_objective.hpp:148, binary_objective.hpp:175,
//         multiclass_objective.hpp:132 + 239, common.h:587 (Softmax)

// regression (l2/l1): identity, OR sqrt variant (objective string "regression sqrt")
fn convert_regression(input: f64, sqrt: bool) -> f64 {
    if sqrt { sign(input) * input * input } else { input }   // Common::Sign
}

// binary: single output, sigmoid with sigmoid_ param (parsed from objective line "sigmoid:X")
fn convert_binary(input: f64, sigmoid: f64) -> f64 {
    1.0 / (1.0 + (-sigmoid * input).exp())
}

// multiclass softmax: in-place over num_class with max-subtraction
fn softmax(input: &[f64], output: &mut [f64]) {           // common.h:587
    let wmax = input.iter().copied().fold(input[0], f64::max);
    let mut wsum = 0.0;
    for i in 0..input.len() { output[i] = (input[i] - wmax).exp(); wsum += output[i]; }
    for o in output.iter_mut() { *o /= wsum; }
}

// multiclassova: per-class sigmoid (num_class outputs)
fn convert_multiclassova(input: &[f64], output: &mut [f64], sigmoid: f64) {
    for i in 0..input.len() { output[i] = 1.0 / (1.0 + (-sigmoid * input[i]).exp()); }
}
```
**The objective string is parsed from the model's `objective=` line** (e.g. `objective=binary sigmoid:1`, `objective=multiclass num_class:3`, `objective=regression`). The objective shim reads `sigmoid_`/`num_class_` from that line — NOT from a training Config. (binary_objective.hpp:43-53 string-ctor; multiclass_objective.hpp:34-47.) `objective=regression` → identity unless ` sqrt` token present (regression_objective.hpp:string-ctor).

### `Tree::ToString` exact section order (DAT-09) — verified
```
num_leaves=<int>\n
num_cat=<int>\n
split_feature=<int int ...>\n                 (n=num_leaves-1)   ArrayToString (int → "{}")
split_gain=<float ...>\n                       {:g}  (low precision!)
threshold=<double ...>\n                       {:.17g}  (ArrayToString<true>)
decision_type=<int ...>\n                      int8 cast→int → "{}"
left_child=<int ...>\n
right_child=<int ...>\n
leaf_value=<double ...>\n     (n=num_leaves)    {:.17g}
leaf_weight=<double ...>\n    (n=num_leaves)    {:.17g}
leaf_count=<int ...>\n        (n=num_leaves)
internal_value=<double ...>\n                  {:g}  (NOT high precision — ArrayToString default)
internal_weight=<double ...>\n                 {:g}
internal_count=<int ...>\n
[ cat_boundaries=<int ...>\n  cat_threshold=<uint ...>\n ]   only if num_cat>0
is_linear=<0|1>\n
[ leaf_const / num_features / leaf_features / leaf_coeff ]    only if is_linear (OUT OF SCOPE)
shrinkage=<double>\n          (ostream << shrinkage_, see note)
\n                                              ← trailing blank line (tree.cpp:406)
```
**CRITICAL nuance just discovered:** `internal_value`/`internal_weight` use `ArrayToString(...)` WITHOUT the `<true>` template arg (tree.cpp:365-368) → `{:g}` (6 sig digits), whereas `threshold`/`leaf_value`/`leaf_weight` use `ArrayToString<true>` → `{:.17g}`. `shrinkage=` uses bare `str_buf << shrinkage_` on a `C_stringstream` (tree.cpp:405) → ostream default (NOT fmt). The plan needs BOTH formatters (`{:g}` and `{:.17g}`) plus the ostream-default path for `shrinkage`. The byte-exact golden is the arbiter — but flag all three formatting modes.

### `SaveModelToString` ensemble envelope order (DAT-09) — verified
```
tree\n                                  (SubModelName, gbdt.h:466 → "tree")
version=v4\n                            (kModelVersion, gbdt_model_text.cpp:19)
num_class=<int>\n
num_tree_per_iteration=<int>\n
label_index=<int>\n
max_feature_idx=<int>\n
objective=<objective string>\n          (only if objective_function_ != null)
[ average_output\n ]                     (only if average_output_; RF — bare line, no '=')
feature_names=<space-joined>\n
[ monotone_constraints=<space-joined>\n ]  (only if non-empty)
feature_infos=<space-joined [min:max] or cat:cat>\n
tree_sizes=<space-joined sizes>\n
\n                                       ← blank line
Tree=0\n<tree0 ToString>\nTree=1\n<tree1 ToString>\n ...
end of trees\n
\n
feature_importances:\n
<feature_name>=<count>\n ...             (desc by count, >0 only, stable_sort)
\n
parameters:\n
<config ToString OR loaded_parameter_ verbatim>\n
end of parameters\n
[ \nparser:\n<parser_config>\nend of parser\n ]   (only if parser_config_str_ non-empty)
```
*(All `\n`; `C_stringstream` forces C locale. Verified gbdt_model_text.cpp:311-407.)*

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ArrayToStringFast` / `Common::ArrayToString` (old printf-based) | `CommonC::ArrayToString` + `fmt::format_to_n("{:.17g}")` | LightGBM ≥ v4 (model `version=v4`) | The writer MUST match `fmt`'s `{:.17g}`, not C `snprintf` (they agree for `%g` but the buffer-size guard differs). The committed fixtures are v4 — confirmed `kModelVersion="v4"`. |
| Sequential tree parse via `used_len` | `tree_sizes=` byte-boundary parallel parse | v4 added `tree_sizes=` | Read path keys off `tree_sizes` (the `used_len` path is legacy fallback for pre-`tree_sizes` models). |

**Deprecated/outdated:** Pre-v4 models without `tree_sizes=` use the sequential `used_len` parse (`gbdt_model_text.cpp:529-545`). Phase 3 fixtures are v4, so implement the `tree_sizes` path; the `used_len` fallback is optional (note it for completeness, low priority).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| — | (none) | — | All claims verified against pinned in-repo C++ source. The only unresolved items are Open Questions below (genuine gaps requiring a planning decision or a fixture-capture experiment), not assumptions stated as fact. |

**This table is empty of `[ASSUMED]` claims** — every behavioral statement cites a specific C++ source line. The capture-feasibility and feature-importance items are surfaced as Open Questions, not assumptions.

## Open Questions (RESOLVED)

> All four open questions are closed by the Phase-3 plans (planning gate, not an execution blocker):
> - **Q1 (raw-row predict path)** — RESOLVED: 03-02-T3 + PATTERNS `predict.rs` materialize raw `f64` rows directly, bypassing `Dataset::construct` (mirror C++ `Predictor`).
> - **Q2 (golden-capture path)** — RESOLVED: 03-01-T3 `checkpoint:decision` (blocking-human) selects path B (pip `lightgbm` train+dump) with transcription fallback; provenance recorded in `REFERENCE_MANIFEST.md`.
> - **Q3 (`feature_importances:` round-trip)** — RESOLVED: 03-02-T2 recomputes `FeatureImportance` (split-count default) with the byte-exact golden as arbiter.
> - **Q4 (three formatters)** — RESOLVED: 03-01-T2 implements `format_g17` (`{:.17g}`) + `format_g6` (`{:g}`) with the shrinkage/ostream case settled against the golden.

1. **Does `lgbm-dataset` expose RAW per-row feature values for the predict path (D-02a)?**
   - What we know: D-02a requires prediction on raw `f64` feature values against stored real thresholds — NOT the binned representation. The Phase-2 `from_mat`/`from_csr`/`from_csc` produce a `FinishedDataset` of *binned* columns.
   - What's unclear: Whether the predict path should (a) keep the caller's raw `&[f32]`/CSR/CSC and materialize per-row `f64` vectors directly (bypassing `Dataset` binning entirely, matching the C++ `Predictor` which parses raw values into a `predict_buf_`), or (b) read raw values back from the dataset. The C++ `Predictor` (`predictor.hpp:259` `CopyToPredictBuffer`) builds a raw `double` buffer from the input pairs — it does NOT go through `BinMapper`. 
   - Recommendation: **Mirror C++ exactly** — the predictor takes raw input rows (dense `&[f32]`/`&[f64]` or CSR/CSC), materializes a dense `f64` row buffer of width `max_feature_idx+1`, and feeds `Tree::Predict`. Reuse `lgbm-dataset`'s CSR/CSC iteration *shape* for row extraction but feed raw values, not bins. Do NOT route through `Dataset::construct`. Confirm at planning time which `lgbm-dataset` helpers (if any) expose raw row iteration, else add a thin raw-row materializer in `lgbm-model::predict`.

2. **Golden-capture feasibility (D-07) — which path produces the committed `.txt` fixtures?** *(FIRST planning gate.)*
   - What we know (VERIFIED): The full `lib_lightgbm` is **unbuildable here** — `gbdt_model_text.cpp` and `tree.cpp` `#include common.h`, which pulls `fmt/format.h` + `fast_double_parser.h` from `external_libs/{fmt,fast_double_parser}`, which are **empty directories** (confirmed: `ls external_libs/fmt/include/fmt/format.h` → not found). No system `lightgbm` is installed (confirmed: `python3 -c import lightgbm` fails; no `lightgbm` CLI). This is the SAME blocker Phase 1 (RNG) and Phase 2 (binning) hit.
   - What's unclear: The cleanest capture path. Three candidates:
     - **(A) Verbatim transcription** (Phase 1/2 precedent): transcribe `SaveModelToString` + `Tree::ToString` + the `fmt {:.17g}` formatting into the `xtask/cpp` harness, compile header-only (it needs a trained model as input though — so it ALSO needs a trainer, which doesn't exist yet). **Problem:** Phase 3 has no Rust trainer (Phase 5/6) and can't build the C++ trainer either. A model `.txt` must come from *somewhere*.
     - **(B) Install an external `lightgbm`** (pip wheel or prebuilt binary) purely to TRAIN and emit reference `.txt` models + reference predictions, then COMMIT them. This is the most likely viable path: pip `lightgbm` ships a prebuilt `lib_lightgbm` with fmt baked in, so its `.txt` output IS the authoritative `version=v4` format with correct `%.17g`. The committed `.txt` + committed prediction vectors become the goldens; no toolchain at test time.
     - **(C) Hand-author tiny `.txt` models** by transcribing a known-good small tree. Fragile, not faithful — only as a last resort for unit-level tests.
   - Recommendation: **Pursue (B)** — add an xtask `model-capture` subcommand that shells out to a pip-installed `lightgbm` (or a one-off documented script) to train the D-05 corpus (regression/binary/multiclass/categorical/sub-range), dump `.txt` models + predict vectors (raw, transformed, leaf-index, sub-range) on the reused Phase-2 input matrices, and COMMIT all of it into `tests/fixtures/models/`. Document the exact `lightgbm` version + train params in `REFERENCE_MANIFEST.md` (extends ORA-02). The byte-exact write golden = the pip-`lightgbm`-written `.txt`. **Confirm the pip wheel's model text uses `fmt {:.17g}` identically** (it should — same source) by a spot round-trip. If pip install is disallowed in the environment, fall back to (A)+(B-hybrid): transcribe the model writer AND a minimal GBDT-train-from-fixed-grad stub — significantly more work; flag to user.

3. **`feature_importances:` block on round-trip — recompute or preserve?** (see Pitfall 7)
   - What we know: `SaveModelToString` recomputes `FeatureImportance(num_iteration, type)` every write (`gbdt_model_text.cpp:373`). The loader does NOT store importances. So a faithful writer must RECOMPUTE them from the loaded trees.
   - What's unclear: Which `feature_importance_type` the reference `.txt` was written with (split vs gain) and whether `num_iteration` passed at save time equals the loaded count. For a `model.txt` written by `Booster.save_model()`, the default is split-count over all iterations.
   - Recommendation: Port `FeatureImportance` (split-count default, `gbdt_model_text.cpp:627`) and the descending stable-sort; verify the round-trip golden's trailing block matches. If it diverges, capture the exact save-time `importance_type` from the reference and match. Low predict-risk (not used by PRD-*), pure write-parity.

4. **`%.17g` vs `{:g}` vs ostream-default — three formatters needed.** (see Tree::ToString nuance)
   - What we know: `threshold`/`leaf_value`/`leaf_weight` → `{:.17g}`; `split_gain`/`internal_value`/`internal_weight` → `{:g}`; `shrinkage` → ostream `<<`. All confirmed.
   - Recommendation: Implement `{:.17g}` and `{:g}` (printf-`%g` with precision 17 and 6). For `shrinkage`, ostream default float is typically `{:g}`-like with precision 6 (`std::defaultfloat`, precision 6) — verify against golden; likely identical to `{:g}` for the usual `shrinkage` values (0.1, 1, etc.). The byte-exact golden settles it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust / cargo (edition 2024) | Building the crate | ✓ | rust-version 1.95 (workspace) | — |
| `lgbm-core`, `lgbm-dataset`, `oracle-harness` | All Phase-3 work | ✓ | workspace path deps (Phase 1/2 complete) | — |
| Full `lib_lightgbm` build (`external_libs/fmt`, `fast_double_parser`) | Ideal golden capture | ✗ | — | empty submodule dirs → **unbuildable** (verified). Use verbatim transcription OR external `lightgbm`. |
| System/pip `lightgbm` (to TRAIN + emit reference `.txt`) | D-07 golden capture (path B) | ✗ | — | pip install in xtask capture step (one-off, documented), OR transcription fallback (path A) |
| C++ toolchain + CMake ≥ 3.28 | xtask cpp capture (transcription path) | ✓ (used by Phase 1/2 `xtask/cpp`) | — | — |

**Missing dependencies with no fallback:** None hard-blocking — but the golden-capture path (Open Q2) is the first planning gate and MUST be resolved before fixtures can be produced.

**Missing dependencies with fallback:**
- Full lib build → header-only transcription (Phase 1/2 precedent) or external `lightgbm` install for training/dumping.

## Validation Architecture

> Nyquist validation is ENABLED (config has no `workflow.nyquist_validation:false`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cargo test` (workspace convention from Phases 1–2) |
| Config file | none — standard Cargo test layout (`crates/lgbm-model/tests/*.rs` + inline `#[cfg(test)]`) |
| Quick run command | `cargo test -p lgbm-model` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DAT-08 | Parse C++ `.txt` → in-memory model (then predict matches) | integration | `cargo test -p lgbm-model --test predict_raw_parity` | ❌ Wave 0 |
| DAT-09 | Write byte-identical to C++ `SaveModelToString` (round-trip) | integration | `cargo test -p lgbm-model --test model_text_roundtrip` | ❌ Wave 0 |
| DAT-09 | `%.17g`/`{:g}` formatter parity (pre-tree-writer gate) | unit | `cargo test -p lgbm-model format::` | ❌ Wave 0 |
| PRD-01 | Raw-score f32 ~1e-6 (`compare_within`) | integration | `cargo test -p lgbm-model --test predict_raw_parity` | ❌ Wave 0 |
| PRD-02 | Transformed output f32 ~1e-6 (sigmoid/softmax/ova/identity) | integration | `cargo test -p lgbm-model --test predict_transform` | ❌ Wave 0 |
| PRD-03 | Leaf-index exact integer (`compare_exact_u32`) | integration | `cargo test -p lgbm-model --test predict_leaf_parity` | ❌ Wave 0 |
| PRD-06 | Sub-range raw scores f32 ~1e-6 | integration | `cargo test -p lgbm-model --test predict_subrange` | ❌ Wave 0 |

**D-06 layer → comparator map (all comparators already exist):**
| D-06 Layer | Artifact | Comparator | Fixtures (D-05) |
|-----------|----------|-----------|-----------------|
| 1. model-text round-trip bytes | Rust-written `.txt` vs committed C++ `.txt` | `compare_exact_bytes` | all 5 corpora |
| 2. raw score | per-row `f32` raw scores | `compare_within` (ORACLE_TOL) | regression, binary, multiclass |
| 3. transformed output | per-row `f32` after ConvertOutput | `compare_within` | binary (sigmoid), multiclass (softmax), regression (identity) |
| 4. leaf index | per-(row×tree×class) `u32` leaf ids | `compare_exact_u32` | regression, multiclass (stride check) |
| 5. sub-range raw | raw scores for `(start_iteration,num_iteration)` slices incl. `-1==all` | `compare_within` | sub-range corpus |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-model` (crate-local layers)
- **Per wave merge:** `cargo test --workspace` (no regression in dataset/core/oracle)
- **Phase gate:** Full suite green + bin-capture/model-capture idempotent before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `crates/lgbm-model/` crate skeleton + add to root `Cargo.toml` `members`
- [ ] `crates/lgbm-model/src/format.rs` — `%.17g` + `{:g}` formatter + unit parity test (the linchpin; build & verify FIRST)
- [ ] `xtask` `model-capture` subcommand + capture path decision (Open Q2) — emits committed `.txt` models + predict-vector goldens into `tests/fixtures/models/`
- [ ] `tests/fixtures/models/{regression,binary,multiclass,categorical,subrange}/` — committed C++ `.txt` + raw/transformed/leaf/subrange golden vectors
- [ ] Extend `REFERENCE_MANIFEST.md` (ORA-02) with model/predict fixture provenance + exact `lightgbm` version & train params
- [ ] Five integration test files (one per D-06 layer mapping above)

## Security Domain

> `security_enforcement` not disabled in config → enabled. This phase has a narrow attack surface: it parses **committed, trusted, C++-generated `.txt` fixtures** (not untrusted user input in v1) and predicts on in-memory matrices. The eventual public surface (Phase 8 Python) is out of scope here.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Library code |
| V5 Input Validation | **yes** | The model-text parser and the predict-input ingest are the validation boundary. Malformed model text / mismatched feature counts must return a typed `thiserror` `ModelError`, **never panic** (Phase 2 D-05 / FND-04 precedent). Mirror C++ `Log::Fatal` checks (e.g. `num_leaves` missing, `feature_names` size ≠ `max_feature_idx+1`, `feature_infos` size mismatch) as `Result::Err`. Validate array-length consistency (`split_feature.len()==num_leaves-1`, etc.) before indexing. |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for {Rust model parser}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds index from malformed `split_feature`/`left_child` (a node index ≥ array len) | Tampering / DoS | Validate all parsed array lengths against `num_leaves`/`num_cat` at parse time; return `ModelError`, never `panic!`/unchecked `[]`. C++ uses raw `[]` (UB on bad input) — the Rust port should be STRICTER (typed error) while observably identical on valid input. |
| Integer overflow in `tree_sizes` byte-boundary arithmetic | Tampering | Use checked/`usize` arithmetic; validate `sum(tree_sizes) <= buffer.len()`. |
| `feature_infos`/`feature_names` count mismatch | Tampering | Already a C++ `Log::Fatal` (`gbdt_model_text.cpp:494,514`) — port as `ModelError`. |
| Panic on NaN/inf feature value at predict | DoS | C++ handles NaN explicitly in `NumericalDecision`; the Rust port must too (NaN→0 coercion / NaN-type routing) — no `unwrap` on float comparisons. |

## Sources

### Primary (HIGH confidence) — pinned in-repo C++ (commit-pinned, VERSION 4.6.0.99)
- `LightGBM/include/LightGBM/tree.h` — Tree member layout (478-541), `Predict`/`GetLeaf` (587-727), `NumericalDecision`/`CategoricalDecision`/`Decision` (337-415), `GetDecisionType`/`GetMissingType` bit helpers (262-281), masks (20-21)
- `LightGBM/src/io/tree.cpp` — `ToString` (339-409, section order + formatting modes), parse ctor `Tree(const char*, size_t*)` (685-866, keyed map + per-field fallbacks + single-leaf early return)
- `LightGBM/src/boosting/gbdt_model_text.cpp` — `SaveModelToString` (311-407, envelope order), `LoadModelFromString` (421-625, key parse + tree_sizes boundaries + parameters/parser blocks), `kModelVersion="v4"` (19), `FeatureImportance` (627)
- `LightGBM/src/boosting/gbdt_prediction.cpp` — `PredictRaw`/`Predict`/`PredictLeafIndex` (13-95, accumulation + ConvertOutput + leaf-index layout)
- `LightGBM/src/boosting/gbdt.h` — `InitPredict` (426-436, sub-range state), `NumPredictOneRow` (281-291), `SubModelName="tree"` (466)
- `LightGBM/include/LightGBM/utils/common.h` — `CommonC::ArrayToString` + `__TToStringHelper` `{:.17g}`/`{:g}` (1210-1256), `Join` int8 specialization (518-534), `Softmax` (587), `FindInBitset` (836), `AvoidInf` (653-665)
- `LightGBM/src/objective/{regression,binary,multiclass}_objective.hpp` — `ConvertOutput` (regression 148, binary 175, multiclass 132, multiclassova 239) + string-ctors parsing `sigmoid:`/`num_class:`/`sqrt`
- `LightGBM/src/application/predictor.hpp` — batch predict driver, raw-value `CopyToPredictBuffer` (259, confirms D-02a raw path)
- `LightGBM/include/LightGBM/bin.h` — `bin_info_string` (224, confirms `feature_infos` = `[min:max]`/`cat:cat`, ostream `setprecision(17)`)
- `LightGBM/src/io/config.cpp` — `Config::ToString` (476, the `parameters:` block — preserved verbatim on load, not re-derived)
- Workspace: `crates/oracle-harness/src/comparator.rs` (comparators), `crates/lgbm-dataset/src/ingest.rs` (from_mat/csr/csc sigs), `crates/lgbm-dataset/src/dataset.rs` (FinishedDataset), root `Cargo.toml`, `xtask/cpp/{CMakeLists.txt,bin_capture.cpp}` (capture-infeasibility precedent)

### Secondary / Tertiary
- None used. No WebSearch/Context7 needed — this phase is a hand-port of a fully-available, pinned C++ reference.

## Metadata

**Confidence breakdown:**
- Standard stack (no new deps; reuse + 1 formatter): HIGH — verified against workspace + C++ source
- Architecture (faithful array mirror, envelope/predict separation): HIGH — directly from `tree.h`/`gbdt_model_text.cpp`
- Pitfalls (%.17g, dual formatters, decision-type bits, tree_sizes, parse precision, feature_importances): HIGH — each cites exact source lines
- Golden-capture path (D-07): MEDIUM — infeasibility of full-lib build VERIFIED; the *chosen* capture path (pip `lightgbm` vs transcription) is an open planning decision (Open Q2), not a verified fact

**Research date:** 2026-06-05
**Valid until:** Stable — the C++ reference is commit-pinned and read-only; findings do not expire until the pinned LightGBM version changes. Re-verify Open Q2 (environment `lightgbm` availability) at plan time.
