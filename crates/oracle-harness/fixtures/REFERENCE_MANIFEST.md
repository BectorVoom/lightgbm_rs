# Reference Manifest — LightGBM-rs Oracle (Phases 1-2)

This file pins the C++ reference build used to generate the committed RNG
golden set (`rng_sequence.txt`). It records everything needed to reproduce the
fixtures deterministically (ORA-02, D-05, D-14). Normal `cargo test` reads the
committed fixtures and needs NONE of this; only `cargo run -p xtask -- regen`
does (D-06).

## Pinned C++ Reference

- **Submodule:** `LightGBM/` (in-repo, read-only)
- **Commit:** `195c26fc7b00eb0fec252dfe841e2e66d6833954`
- **Version (`VERSION.txt`):** `4.6.0.99`

## Deterministic Build / Capture Flags

- `deterministic=true`
- `force_row_wise=true`
- `num_threads=1`
- default `float` width — `SCORE_T_USE_DOUBLE` / `LABEL_T_USE_DOUBLE` NOT defined (D-01)
- CPU-only build: `USE_GPU=OFF USE_CUDA=OFF USE_MPI=OFF USE_SWIG=OFF BUILD_CLI=OFF`

> The RNG (`LightGBM::Random`) is a self-contained, header-only LCG, so its draws
> do not depend on the threading/row-wise/build flags above. The RNG golden is
> therefore captured by compiling `rng_capture` DIRECTLY against the pinned
> `include/LightGBM/utils/random.h` (default f32 width) — no `lib_lightgbm` build
> or link (the in-repo submodule's `external_libs/` are not vendored). The
> deterministic CPU-only flags above are recorded because the same pinned
> reference build is the source of truth for all later (training) goldens; this
> manifest is the single source of truth for that reference configuration.

## Exact Regeneration Command

```bash
cargo run -p xtask -- regen
```

which internally runs (standalone CMake, never modifying the submodule tree):

```bash
cmake -S xtask/cpp -B target/xtask-cpp-build \
-DLIGHTGBM_DIR=<repo>/LightGBM -DCMAKE_BUILD_TYPE=Release
cmake --build target/xtask-cpp-build --target rng_capture --config Release
target/xtask-cpp-build/rng_capture \
crates/oracle-harness/fixtures/rng_sequence.txt 1592594996 256 256
```

## Randomized-at-Capture Case Set (D-14)

The golden set is derived deterministically from ONE recorded master seed (no
wall-clock / OS entropy), so regeneration is idempotent (empty `git diff`).

- **Master seed:** `1592594996` (`0x5EED1234`)
- **RNG cases:** `256` (many random LCG seeds; each emits NextShort / NextInt /
NextFloat / NextInt draw sequences in a fixed order)
- **Sample cases:** `256` (randomized `(N, K)` pairs straddling the
`K > N / log2(K)` branch boundary — small-K set branch, large-K streaming
branch, and near-boundary)
- **Total generated cases:** `512`

## Fixture Format (`rng_sequence.txt`)

Line-delimited text (diff-friendly, no serde). `#`-prefixed lines are comments.

```
MASTER_SEED <seed>
COUNTS rng=<n> sample=<n>
RNG seed=<s> int16=<a;b;...> int32=<...> float=<bits;...> int=<...>
SAMPLE seed=<s> N=<n> K=<k> result=<v0;v1;...>
```

`float` values are the raw little-endian f32 bit pattern (a decimal `u32`) so the
Rust parity test asserts exact-bit f32 equality; integer draws are compared
exactly; `Sample` output is compared as an exact ordered sequence.

## Numeric Binning Golden Set (Phase 2, layers 1+2)

Captured by `cargo run -p xtask -- bin-capture` into
`crates/lgbm-dataset/tests/fixtures/numeric_binning.txt`. Covers the NUMERIC
`BinMapper::FindBin` (layer 1: `bin_upper_bound_`, `num_bin`, `bin_type`,
`missing_type`, `default_bin`, `most_freq_bin`, `is_trivial`) and per-row
`ValueToBin` (layer 2). Categorical folding and EFB are OUT OF SCOPE here
(categorical -> Plan 03, EFB -> Plan 05).

- **Binning master seed:** `185712367` (`0x0B11BEEF`) —
the SINGLE source of randomness for the binning corpus (idempotent regen).
- **Corpus (four-source, D-06; numeric subset):**
1. synthetic randomized distributions sweeping `max_bin` (2/16/64/255),
`min_data_in_bin` (1/3/20), and `bin_construct_sample_cnt` (64/256/100000),
each with a randomized `data_random_seed`;
2. curated numeric edge battery: NaN-as-missing, +0.0/-0.0 signed zeros,
on-boundary ties, all-missing, single-value, all-zero, zero-as-missing,
a pre-filter-triggering column, and a dense 500-value column.
(LightGBM example datasets and the categorical/EFB corpus land in later plans.)

### EXACT comparison discipline (NOT the ~1e-6 oracle tolerance)

Binning goldens are compared **bit-exact**, never within the `~1e-6` oracle
tolerance: per-row bin indices via `compare_exact_u32`, the f64
`bin_upper_bound_` array via `compare_exact_f64_bits` (`.to_bits()` per element),
and storage-layout bytes (later plans) via `compare_exact_bytes`. A 1-ULP
boundary drift is a real divergence, so exact f64-bit equality is mandatory.

### Capture-harness note (external_libs unavailable)

The authoritative `BinMapper::FindBin`/`ValueToBin` in `src/io/bin.cpp` pull in
`common.h` -> `fast_double_parser.h` + `fmt/format.h` from `external_libs/`,
which are present here only as EMPTY directories (the LightGBM tree is
git-untracked and its submodules are not vendored). `bin.cpp` is therefore
unbuildable in this environment. `xtask/cpp/bin_capture.cpp` VERBATIM-transcribes
the numeric FindBin family from the pinned `bin.cpp`/`bin.h` (commit `195c26fc7b00eb0fec252dfe841e2e66d6833954`,
version `4.6.0.99`) using the genuine `std::nextafter` (== `GetDoubleUpperBound`)
and the asymmetric `b <= nextafter(a)` dedup — so it emits goldens byte-identical
to lib_lightgbm — and links only the header-only reference `Random` for sampling.
This mirrors the Phase-1 header-only `rng_capture` discipline.

## EFB Grouping Golden Set (Phase 2, layer 3, DAT-05)

Captured by `cargo run -p xtask -- bin-capture` into
`crates/lgbm-dataset/tests/fixtures/efb_grouping.txt`. Covers Exclusive Feature
Bundling (layer 3): feature->group membership (`feature2group_` /
`feature2subfeature_`), per-group `bin_offsets_` + `num_total_bin_` + the
`group_is_multi_val` flag, and the per-row bundled bin index per single-value
group. Corpus = D-06 number 4: two mutually-exclusive sparse feature sets (which EFB
bundles into one group each) plus a control where no features are mutually
exclusive (one single-feature group per feature — proves the `enable_bundle`
dispatch boundary).

### Capture-harness resolution: VERBATIM TRANSCRIPTION (external_libs unvendored)

The plan flagged a MEDIUM-risk feasibility choice between (a) a focused harness
compiling `src/io/dataset.cpp` directly, and (b) a full-CLI `enable_bundle=true`
dump. **Both nominal options are provably infeasible in this environment:**

- **(a) focused `dataset.cpp` build — INFEASIBLE.** `dataset.cpp` transitively
includes `common.h` -> `fast_double_parser.h` + `fmt/format.h` from
`external_libs/`, which are present here only as EMPTY directories (the
LightGBM tree is git-untracked and its submodules are unvendored). The build
fails with `fast_double_parser.h: No such file or directory`.
- **(b) full-CLI dump — INFEASIBLE.** Building `lib_lightgbm` / the `lightgbm`
CLI requires the same unvendored `external_libs` (`fast_double_parser`, `fmt`,
`eigen`, `compute`), so the CLI cannot be built either.

**Resolution (human-approved):** EFB is captured by a HEADER-ONLY VERBATIM
TRANSCRIPTION of the EFB pipeline (`GetConflictCount`/`FindGroups`/
`FastFeatureBundling`/`FixSampleIndices` + the bundled `FeatureGroup` /
`bin_offsets_` / `num_total_bin_` group layout) from the pinned `dataset.cpp`
(commit `195c26fc7b00eb0fec252dfe841e2e66d6833954`, version `4.6.0.99`) and `feature_group.h`, compiled against
only `-I LightGBM/include` plus the header-only `LightGBM::Random` (sampling +
group shuffle). This is the SAME discipline plans 02-01..02-04 used for every
prior golden layer (numeric / storage / categorical / missing / metadata): no
`external_libs`, no `lib_lightgbm` link, output byte-identical to what
lib_lightgbm would emit because the transcribed code is the authoritative
reference source.

### Exact bin-capture command

```bash
cargo run -p xtask -- bin-capture
```

## Model / Predict Golden Set (Phase 3, D-05 / PRD-01..PRD-06)

Captured by `cargo run -p xtask -- model-capture` into
`crates/lgbm-model/tests/fixtures/models/{regression,binary,multiclass,categorical,subrange}/`.
Each corpus directory holds the authoritative C++ `version=v4` `model.txt`
(`Booster.save_model()`) plus per-corpus predict-vector goldens:
`raw.txt` (PRD-01 raw scores), `transformed.txt` (PRD-02 — sigmoid for binary,
softmax for multiclass, identity for regression), `leaf.txt` (PRD-03 leaf
indices), and (for `subrange`) `subrange.txt` (PRD-06 raw scores for
representative `(start_iteration, num_iteration)` slices incl. `-1 == all
remaining`). The fixed-double `%g` battery for the `format.rs` DAT-09 formatter
is `models/format_golden.txt` (G17 = `{:.17g}`, G6 = `{:g}`). Float golden
vectors are `;`-separated raw f64 bit patterns (decimal `u64`) for bit-exact
replay; leaf indices are `;`-separated decimal `u32`.

- **Training tool (capture-time only):** pip `lightgbm` `4.6.0`
(RESEARCH Open Q2 path B). NOT a dependency of the shipped crate and NEVER read
at `cargo test` time — the fixtures are committed.
- **Train seed:** `2147483647` (`0x7FFFFFFF`).
- **Deterministic train params:** `deterministic=true force_row_wise=true
num_threads=1 bagging_freq=0 bagging_fraction=1.0 feature_fraction=1.0
num_boost_round=10 num_leaves=31 min_data_in_leaf=20` (NO data subsampling), so
re-running `model-capture` is byte-idempotent (empty `git diff`).
- **Corpora:** regression (`objective=regression`), binary
(`objective=binary`), multiclass (`objective=multiclass num_class=3`, label
derived deterministically by tertile-bucketing a stable feature), categorical
(`objective=binary` with 4 integerized `categorical_feature` columns),
subrange (a regression model exercising the PRD-06 sub-range slices). The
regression/binary inputs are the COPIED Phase-2 example matrices under
`crates/lgbm-dataset/tests/fixtures/examples/` — NEVER the untracked
`LightGBM/` tree.

### Capture-path resolution: PATH B (pip lightgbm train + dump), human-approved

RESEARCH Open Q2 (the FIRST planning gate) offered (A) verbatim transcription of
`SaveModelToString` + a train stub vs (B) pip `lightgbm` train + dump. **Path A
is infeasible standalone here** (Phase 3 has no Rust trainer and the C++ trainer
is unbuildable — `external_libs/{fmt,fast_double_parser,...}` are empty), so a
trained `.txt` must come from a prebuilt `lib_lightgbm`. **Path B was selected
and approved:** the pip wheel ships `lib_lightgbm` with `fmt` baked in, so its
`save_model()` IS the authoritative v4 format with correct `%.17g`. The exact
tool version + train params are pinned above; the produced fixtures were
human-approved as numerically identical to `lib_lightgbm` (03-VALIDATION.md
Manual-Only Verifications). The capture interpreter is resolved from
`$LGBM_CAPTURE_PYTHON` (a venv with `pip install lightgbm`).

### Exact model-capture command

```bash
LGBM_CAPTURE_PYTHON=/path/to/venv/bin/python cargo run -p xtask -- model-capture
```

## Kernel Golden Set (Phase 4, D-02 / D-02a)

Captured by `cargo run -p xtask -- kernel-capture` into
`crates/oracle-harness/tests/fixtures/kernels/histogram.txt`. Covers the D-01
whole-kernel `construct_histograms` op: the stride-2 `[g0,h0,g1,h1,...]` f64
histogram (`hist_t = double`) accumulated from f32 (`score_t = float`) ordered
gradients/hessians over a feature column's per-row bin indices
(`ti = bin << 1`). The cubecl-cpu kernel reproduces this BIT-EXACT (the D-04
deterministic anchor); `crates/oracle-harness/tests/kernel_parity.rs` replays it
via `compare_exact_f64_bits`.

- **Kernel master seed:** `1096282125` (`0x4157F00D`) —
the SINGLE source of randomness for the histogram corpus (idempotent regen).
- **D-02a path coverage:** dense + sparse bin layouts; the most-frequent /
default-bin (lowest-bin) routing; multiple bin-store bit widths
(`DenseBin<u8,4bit>` / `u8` / `u16` / `u32` and the matching `SparseBin`
widths, selected by `num_bin` per `Bin::CreateDenseBin`/`CreateSparseBin`); an
all-rows-on-one-bin pileup; an empty-sparse-stream (all-bin-0) round-trip; and
a grad/hess sign+magnitude spread (~1e-3 .. ~1e3, mixed signs) that stresses
the non-associative f64 reduction order.

### Capture-harness note (external_libs unbuildable)

The authoritative `ConstructHistogram` lives in `src/io/dense_bin.hpp` /
`sparse_bin.hpp`, which (via `<LightGBM/bin.h>` -> `common.h`) transitively pull
in `fast_double_parser.h` + `fmt/format.h` from `external_libs/` — present here
only as EMPTY directories. `xtask/cpp/kernel_capture.cpp` therefore VERBATIM-
transcribes the `ConstructHistogram` accumulation bodies from the pinned
`dense_bin.hpp:130-141` / `sparse_bin.hpp:138-152` (commit `195c26fc7b00eb0fec252dfe841e2e66d6833954`, version
`4.6.0.99`), reusing the `DenseBin`/`SparseBin` bin-storage forms, and emits
goldens byte-identical to lib_lightgbm. Synthetic inputs use the genuine
header-only `LightGBM::Random`. Same discipline as `rng_capture`/`bin_capture`:
no `external_libs`, no `lib_lightgbm` link, no C++ toolchain at `cargo test` time
(the golden is committed).

### 04-03 split / partition / subtract goldens

`kernel-capture` also emits three more goldens under the same kernels dir
(`split.txt`, `partition.txt`, `subtract.txt`), each a VERBATIM transcription of
the pinned reference (commit `195c26fc7b00eb0fec252dfe841e2e66d6833954`, version `4.6.0.99`):

- **`split.txt`** — `FindBestThresholdSequentially` + the gain math
(`feature_histogram.hpp:711-1057`, default CPU template). Each case emits the
PER-CANDIDATE gains (REVERSE + FORWARD, NaN where a candidate is gated) AND the
winning `SplitInfo`, so a divergence localizes to the gain scan, not just the
winner. Covers a REVERSE-branch winner (`default_left=1`, threshold `t-1+offset`),
a FORWARD-branch winner (`t+offset`), a default-bin-skip case, an L1-regularized
case, and a no-admissible-split case.
- **`partition.txt`** — `DataPartition::Split` row routing via `SplitInner`
(`dense_bin.hpp:314-394`, `MissingType::None`) + the stable two-pass gather;
emits the reordered index array + `split_point`.
- **`subtract.txt`** — `FeatureHistogram::Subtract` (`feature_histogram.hpp:99-145`,
default `USE_DIST_GRAD=false`): `derived[i] = parent[i] - child[i]`.

`crates/oracle-harness/tests/kernel_parity.rs` replays all four layers BIT-EXACT
on the cubecl-cpu anchor via `compare_exact_f64_bits` / `compare_exact_u32`.

### Exact kernel-capture command

```bash
cargo run -p xtask -- kernel-capture
```

## Learner Golden Set (Phase 5, D-06 per-split / D-07 per-tree)

Captured by `cargo run -p xtask -- learner-capture` into
`crates/oracle-harness/tests/fixtures/learner/`. Covers the serial tree-learner
growth: PER-SPLIT snapshots (D-06 — the full per-bin gain array + winning split,
so a divergence localizes to the gain scan) and PER-TREE goldens (D-07 — the
grown tree's `Tree::to_string()` text, compared via the Phase-3 `%.17g` machinery
as a `String`). `crates/oracle-harness/tests/learner_parity.rs` replays them
bit-exact on the cubecl-cpu anchor (`compare_exact_f64_bits` per-split) / string
equality (per-tree).

- **Learner master seed:** `514219757` (`0x1EA65EED`) —
recorded for format continuity; the Plan-05-03 corpus is hand-crafted (fixed
synthetic g/h), NOT RNG-derived, so the capture is byte-idempotent regardless.
- **Plan 05-03 status: REAL SPINE GOLDEN (`spine.txt`).** The full verbatim
leaf-wise-loop transcription grows a tree over a FIXED 12-row / 2-feature
synthetic g/h corpus (`force_row_wise`, `feature_fraction=1.0`,
`missing_type=None` per RESEARCH A5 — NA_AS_MISSING deferred). It emits 10
PSPLIT records (per-bin REVERSE+FORWARD gain arrays per candidate feature at
every split decision, D-06) + 1 PTREE record (the grown 4-leaf tree's field set
as raw bits, D-07). `learner_parity.rs` replays per-split bit-exact, full-tree
via the shared `%.17g` formatter, the subtraction trick, missing/zero routing,
and the D-02a kernel-vs-learner cross-check.
- **Plan 05-04 status: parity ADDITIONS (`col_wise.txt`, `col_sampler.txt`,
`real_gh.txt`).** Three goldens layered on the proven spine:
- **`col_wise.txt` (TRL-09).** The SAME spine corpus grown under `force_col_wise`.
The transcription is strategy-agnostic (row- vs column-major histogram build
differ ONLY in accumulation ORDER, not result — Pitfall 5), so on the
single-thread cubecl-cpu anchor the grown tree is bit-identical to `spine.txt`.
`learner_parity_row_vs_col` grows the corpus under BOTH `BuildStrategy::RowWise`
and `ColWise` and asserts `row_tree.to_string() == col_tree.to_string() ==`
this golden (String equality). **Open Q2 RESOLVED: `force_col_wise` is a config
FLAG (a no-op) over the shared `construct_histograms` Backend op on the
deterministic anchor — NOT a distinct compute path** (A1 confirmed; a divergence
would fail the row==col gate loudly rather than ship a divergent tree).
- **`col_sampler.txt` (TRL-08).** A `feature_fraction=1.0` /
`feature_fraction_bynode=0.5` config over a 4-feature corpus, drawing the
GENUINE header-only reference `Random::Sample` (`col_sampler.hpp` transcription).
Emits `CS_BYTREE` (the per-tree `ResetByTree` selection) + `CS_NODE` lines (each
per-node `GetByNode` selection, in DRAW ORDER: root first, then smaller-leaf
then larger-leaf per split). The Rust `ColSampler` reproduces the EXACT selected
REAL-feature indices via `train_with_col_sampler_trace`; a wrong draw sequence
fails the parity gate (threat T-05-04-01) rather than silently selecting
different features. The growth is col-sampler-GATED so the draw count/order
matches the Rust learner's trace exactly.
- **`real_gh.txt` (D-03).** Captured iteration-1 g/h from two REAL objectives
(regression-l2 `grad=score-label`, `hess=1`; binary-logloss
`response=-label*sigmoid/(1+exp(label*sigmoid*score))`), `boost_from_average=
false` (score=0), `score_t=float`, over fixed real labels (a realistic gradient
distribution). Each `GH_CORPUS` block emits the captured g/h (raw f32 bits) +
the per-feature bin layout (`GH_FEATURE`) + the grown reference tree (PSPLIT +
PTREE). `learner_parity_real_gh_full_tree` grows from the captured g/h and
asserts the full tree `to_string()` is byte-identical to the C++ reference
(D-07 under a realistic distribution, `missing_type=None` — A5). Regression
grows a clean 3-leaf tree; binary (fractional 0.25 hessians) a clean 2-leaf
tree — `num_leaves` per corpus chosen so every split's ACTUAL children are
non-degenerate.
- **Faithfulness fix (this plan):** the tree's `leaf_count`/`internal_count` record
the ACTUAL `data_partition_->leaf_count(...)` after the row partition
(`serial_tree_learner.cpp:788-791`, `update_cnt=true`), NOT the SplitInfo
`round_int(hess*cnt_factor)` reconstructed counts (which can disagree by +/-1 for
fractional hessians). This corrected the spine's `spine.txt` leaf counts to the
faithful actual-partition values (summing to num_data) and is applied in both the
Rust `split_inner` and the C++ transcription.

### Record format (`spine.txt`)

```
LEARNER_MASTER_SEED <seed>
COUNTS splits=<n> trees=<n>
PSPLIT split=<i> leaf=<l> feature=<f> num_bin=<n> rev=<f64bits;...> fwd=<f64bits;...> winner=<f64bits>
PTREE name=<id> num_leaves=<n>
PT_SPLIT_FEATURE <i...>  PT_THRESHOLD_BITS <u64...>  PT_DECISION_TYPE <i...>
PT_SPLIT_GAIN_BITS <u32...>  PT_LEFT_CHILD <i...>  PT_RIGHT_CHILD <i...>
PT_LEAF_VALUE_BITS <u64...>  PT_LEAF_WEIGHT_BITS <u64...>  PT_LEAF_COUNT <i...>
PT_INTERNAL_VALUE_BITS <u64...>  PT_INTERNAL_COUNT <i...>
ENDTREE
```

`rev`/`fwd`/`winner` + the PT_*_BITS lines are raw little-endian f64/f32 bit
patterns (decimal `u64`/`u32`) for bit-exact replay; the Rust side reconstructs
the reference `Tree` from the PT_* fields and serializes it via the shared
`lgbm-model` `%.17g` formatter for the D-07 String compare.

### Capture-harness note (external_libs unbuildable)

The authoritative `SerialTreeLearner` lives in
`src/treelearner/serial_tree_learner.cpp`, which (via `<LightGBM/dataset.h>` ->
`common.h`) transitively #includes `fast_double_parser.h` + `fmt/format.h` from
`external_libs/` — present here only as EMPTY directories. `learner_capture.cpp`
therefore VERBATIM-transcribes the learner growth loop (Plan 05-03) from the
pinned `serial_tree_learner.cpp` (commit `195c26fc7b00eb0fec252dfe841e2e66d6833954`, version `4.6.0.99`),
reusing `kernel_capture.cpp`'s already-transcribed gain/split math (D-02a
cross-check), and includes the header-only `LightGBM/include` only for the genuine
reference `Random`. Same discipline as `rng_capture`/`bin_capture`/`kernel_capture`:
no `external_libs`, no `lib_lightgbm` link, no C++ toolchain at `cargo test` time
(the golden is committed).

### Exact learner-capture command

```bash
cargo run -p xtask -- learner-capture
```

## REAL Learner Oracle Set (Phase 5, plan 05-06 / D-08 — CR-02 closure)

Captured by `cargo run -p xtask -- learner-oracle-capture` into
`crates/oracle-harness/tests/fixtures/learner/{spine_real.txt,mfb_pos_real.txt}`.
These REPLACE the pre-D-09 self-transcription learner goldens (`spine.txt` /
`real_gh.txt`, which shared the port's offset/`--th` conventions and so validated
the port against ITSELF — CR-02) with model text dumped from the REAL prebuilt
`lib_lightgbm` `4.6.0` (the pip wheel's `save_model()`, exactly the
Phase-3 `model-capture` mechanism — human-approved). Building `lib_lightgbm` from
source is INFEASIBLE here (the in-repo submodule's `external_libs` are empty), so
the pip wheel is the authoritative real binary.

- **`spine_real.txt`** — a `most_freq_bin==0` corpus (offset==1 scan+partition
path) trained on the real binary.
- **`mfb_pos_real.txt`** — a `most_freq_bin > 0` corpus (offset==0 path); the
FIRST bit-exact real-binary anchor for the offset==1-vs-offset==0 convention
fixed in plan 05-05.

- **Training tool (capture-time only):** pip `lightgbm` `4.6.0` —
NOT a crate dependency and NEVER read at `cargo test` time (the goldens are
committed). The version is asserted before training (threat T-05-06-03).
- **Oracle seed:** `97913454` (`0x05D60A6E`).
- **Deterministic train params:** `deterministic=true force_row_wise=true
num_threads=1 bagging_fraction=1.0 feature_fraction=1.0` + identity binning
(`max_bin >= K`, `min_data_in_bin=1`, `bin_construct_sample_cnt >= n_rows`,
`feature_pre_filter=false`, `min_data_in_leaf=1`), so `binned_value == raw_value`
and the dump is byte-idempotent.
- **Binning-pinning (MANDATORY):** the python dumper forces identity binning
(distinct consecutive integers `0..K-1` as raw values) and ASSERTS the realized
per-feature bin count + `most_freq_bin` match the harness corpus layout
(`most_freq_bin > 0` for the mfb>0 corpus), ABORTING the capture on any
mismatch — so a golden can only ever be trained on the exact bin layout the
Rust learner consumes (a binning mismatch can never masquerade as a learner
divergence).

### Exact learner-oracle-capture command

```bash
LGBM_CAPTURE_PYTHON=/path/to/venv/bin/python cargo run -p xtask -- learner-oracle-capture
```
