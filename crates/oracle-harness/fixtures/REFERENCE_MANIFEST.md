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

### Exact bin-capture command

```bash
cargo run -p xtask -- bin-capture
```
