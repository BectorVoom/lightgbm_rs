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
