//! `xtask` — dev-only developer automation for LightGBM-rs.
//!
//! The `regen` subcommand regenerates the committed C++ RNG golden set
//! (`crates/oracle-harness/fixtures/rng_sequence.txt`) and refreshes the pinned
//! reference manifest. It is the ONLY step that needs a C++ toolchain; normal
//! `cargo test` reads the committed fixtures (D-06).
//!
//! Determinism / idempotency (D-14): the randomized case set is derived solely
//! from a recorded [`MASTER_SEED`] constant — there is NO wall-clock or OS
//! entropy source — so re-running `regen` produces byte-identical fixtures and
//! an empty `git diff` (ORA-02).
//!
//! File-I/O safety (T-1-04 / Security V12): every path this tool reads or writes
//! is resolved relative to the workspace root; it never traverses outside the
//! repo.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

// ---------------------------------------------------------------------------
// Recorded randomized-capture parameters (D-14). These are the SINGLE source of
// randomness for the golden set. Changing them re-rolls the set; keep them in
// sync with REFERENCE_MANIFEST.md.
// ---------------------------------------------------------------------------

/// The recorded master seed from which every randomized case (LCG seeds and
/// `Sample` `(N, K)` pairs) is derived deterministically. Committed in the
/// manifest so the set is re-rollable and reproducible.
pub const MASTER_SEED: i32 = 0x5EED_1234;

/// Number of randomized RNG cases (each = many seeds × multi-method draws).
pub const N_RNG_CASES: u32 = 256;

/// Number of randomized `Sample` `(N, K)` cases straddling the branch boundary.
pub const N_SAMPLE_CASES: u32 = 256;

/// The recorded master seed for the Phase-2 numeric binning golden corpus. Like
/// [`MASTER_SEED`] it is the SINGLE source of randomness for the binning cases
/// (synthetic distributions + curated edge battery), so `bin-capture` is
/// idempotent (empty `git diff`). Recorded in REFERENCE_MANIFEST.md.
pub const BIN_MASTER_SEED: i32 = 0x0B11_BEEF;

/// Pinned LightGBM submodule commit (recorded in the manifest, ORA-02 / D-05).
pub const LIGHTGBM_COMMIT: &str = "195c26fc7b00eb0fec252dfe841e2e66d6833954";

/// Pinned LightGBM version (`LightGBM/VERSION.txt`).
pub const LIGHTGBM_VERSION: &str = "4.6.0.99";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("regen") => regen(),
        Some("bin-capture") => bin_capture(),
        Some(other) => {
            bail!("unknown subcommand `{other}` (try: regen | bin-capture)");
        }
        None => {
            eprintln!("usage: cargo run -p xtask -- <regen | bin-capture>");
            Ok(())
        }
    }
}

/// Resolve the workspace root (the directory containing the root `Cargo.toml`).
///
/// `CARGO_MANIFEST_DIR` for the xtask crate is `<root>/xtask`, so the workspace
/// root is its parent. We never construct paths from arbitrary CWD/argv.
fn workspace_root() -> Result<PathBuf> {
    let manifest_dir = env_path("CARGO_MANIFEST_DIR")?;
    let root = manifest_dir
        .parent()
        .context("xtask manifest dir has no parent (unexpected layout)")?
        .to_path_buf();
    // Sanity: the root must contain the virtual workspace manifest.
    if !root.join("Cargo.toml").is_file() {
        bail!("workspace root {} has no Cargo.toml", root.display());
    }
    Ok(root)
}

fn env_path(key: &str) -> Result<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .with_context(|| format!("environment variable {key} is not set"))
}

/// Regenerate the golden set + refresh the manifest.
fn regen() -> Result<()> {
    let root = workspace_root()?;
    verify_toolchain()?;

    let lightgbm_dir = root.join("LightGBM");
    if !lightgbm_dir.join("include/LightGBM/utils/random.h").is_file() {
        bail!(
            "LightGBM submodule not found at {} (expected include/LightGBM/utils/random.h)",
            lightgbm_dir.display()
        );
    }

    let cpp_dir = root.join("xtask/cpp");
    let build_dir = root.join("target/xtask-cpp-build");
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("creating build dir {}", build_dir.display()))?;

    let fixtures_dir = root.join("crates/oracle-harness/fixtures");
    std::fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("creating fixtures dir {}", fixtures_dir.display()))?;
    let fixture_path = fixtures_dir.join("rng_sequence.txt");
    let manifest_path = fixtures_dir.join("REFERENCE_MANIFEST.md");

    // 1. Configure the standalone, header-only capture build (compiles
    //    rng_capture directly against the pinned LightGBM headers; the
    //    header-only `Random` needs no lib_lightgbm build/link).
    eprintln!("xtask regen: configuring C++ capture build ...");
    run(
        Command::new("cmake")
            .arg("-S")
            .arg(&cpp_dir)
            .arg("-B")
            .arg(&build_dir)
            .arg(format!("-DLIGHTGBM_DIR={}", lightgbm_dir.display()))
            .arg("-DCMAKE_BUILD_TYPE=Release"),
        "cmake configure",
    )?;

    // 2. Build rng_capture.
    eprintln!("xtask regen: building rng_capture ...");
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("rng_capture")
            .arg("--config")
            .arg("Release"),
        "cmake build",
    )?;

    // 3. Run the capture over the master-seed-derived randomized set.
    let exe = locate_capture_exe(&build_dir)?;
    eprintln!("xtask regen: running capture ({}) ...", exe.display());
    run(
        Command::new(&exe)
            .arg(&fixture_path)
            .arg(MASTER_SEED.to_string())
            .arg(N_RNG_CASES.to_string())
            .arg(N_SAMPLE_CASES.to_string()),
        "rng_capture",
    )?;

    if !fixture_path.is_file() {
        bail!(
            "capture completed but {} was not written",
            fixture_path.display()
        );
    }

    // 4. Refresh the reference manifest (idempotent — fixed content).
    write_manifest(&manifest_path)?;

    eprintln!(
        "xtask regen: done. Wrote {} and {}.",
        fixture_path.display(),
        manifest_path.display()
    );
    eprintln!(
        "Re-run `cargo run -p xtask -- regen` and confirm \
         `git diff --stat crates/oracle-harness/fixtures/` is empty (idempotent)."
    );
    Ok(())
}

/// Regenerate the Phase-2 numeric binning golden corpus (layers 1+2).
///
/// Mirrors [`regen`]: configures + builds the standalone `bin_capture` C++ target
/// (header-only against the pinned `LightGBM/include` for the reference Random;
/// the numeric `FindBin`/`ValueToBin` are verbatim-transcribed in `bin_capture.cpp`
/// because the submodule's `external_libs/` are not vendored here — see that
/// file's header), then runs it over the [`BIN_MASTER_SEED`]-derived corpus and
/// writes `crates/lgbm-dataset/tests/fixtures/numeric_binning.txt`. Idempotent.
fn bin_capture() -> Result<()> {
    let root = workspace_root()?;
    verify_toolchain()?;

    let lightgbm_dir = root.join("LightGBM");
    if !lightgbm_dir.join("include/LightGBM/utils/random.h").is_file() {
        bail!(
            "LightGBM submodule not found at {} (expected include/LightGBM/utils/random.h)",
            lightgbm_dir.display()
        );
    }

    let cpp_dir = root.join("xtask/cpp");
    let build_dir = root.join("target/xtask-cpp-build");
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("creating build dir {}", build_dir.display()))?;

    let fixtures_dir = root.join("crates/lgbm-dataset/tests/fixtures");
    std::fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("creating fixtures dir {}", fixtures_dir.display()))?;
    let fixture_path = fixtures_dir.join("numeric_binning.txt");
    let storage_fixture_path = fixtures_dir.join("bin_storage_layout.txt");

    eprintln!("xtask bin-capture: configuring C++ capture build ...");
    run(
        Command::new("cmake")
            .arg("-S")
            .arg(&cpp_dir)
            .arg("-B")
            .arg(&build_dir)
            .arg(format!("-DLIGHTGBM_DIR={}", lightgbm_dir.display()))
            .arg("-DCMAKE_BUILD_TYPE=Release"),
        "cmake configure",
    )?;

    eprintln!("xtask bin-capture: building bin_capture ...");
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("bin_capture")
            .arg("--config")
            .arg("Release"),
        "cmake build",
    )?;

    let exe = locate_exe(&build_dir, "bin_capture")?;
    eprintln!("xtask bin-capture: running capture ({}) ...", exe.display());
    run(
        Command::new(&exe)
            .arg(&fixture_path)
            .arg(BIN_MASTER_SEED.to_string())
            .arg(&storage_fixture_path),
        "bin_capture",
    )?;

    if !fixture_path.is_file() {
        bail!(
            "capture completed but {} was not written",
            fixture_path.display()
        );
    }
    if !storage_fixture_path.is_file() {
        bail!(
            "capture completed but {} was not written",
            storage_fixture_path.display()
        );
    }

    // Refresh the shared reference manifest (regen + bin-capture write the same
    // file; content is a pure function of the recorded constants, idempotent).
    let manifest_path = root
        .join("crates/oracle-harness/fixtures")
        .join("REFERENCE_MANIFEST.md");
    write_manifest(&manifest_path)?;

    eprintln!(
        "xtask bin-capture: done. Wrote {} and {}.",
        fixture_path.display(),
        storage_fixture_path.display()
    );
    eprintln!(
        "Re-run `cargo run -p xtask -- bin-capture` and confirm \
         `git diff --stat crates/lgbm-dataset/tests/fixtures/` is empty (idempotent)."
    );
    Ok(())
}

/// Verify a C++ toolchain and CMake are present, returning a clear (non-panic)
/// error if absent (RESEARCH §Environment).
fn verify_toolchain() -> Result<()> {
    check_tool("cmake", &["--version"]).context(
        "cmake is required for `regen` (CMake >= 3.28). Install it and retry; \
         normal `cargo test` does NOT need a C++ toolchain (fixtures are committed).",
    )?;
    // Prefer c++, fall back to checking via cmake-detected compiler if absent.
    check_tool("c++", &["--version"]).context(
        "a C++ compiler (`c++`) is required for `regen`. Install gcc/clang and retry; \
         normal `cargo test` does NOT need a C++ toolchain (fixtures are committed).",
    )?;
    Ok(())
}

fn check_tool(tool: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(tool)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .with_context(|| format!("`{tool}` not found on PATH"))?;
    if !status.success() {
        bail!("`{tool} {}` exited with {status}", args.join(" "));
    }
    Ok(())
}

fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn {what}"))?;
    if !status.success() {
        bail!("{what} failed with {status}");
    }
    Ok(())
}

/// Find the built `rng_capture` executable under the build dir (handles
/// single- and multi-config generators).
fn locate_capture_exe(build_dir: &Path) -> Result<PathBuf> {
    locate_exe(build_dir, "rng_capture")
}

/// Find a built capture executable `name` under the build dir (handles single-
/// and multi-config generators).
fn locate_exe(build_dir: &Path, name: &str) -> Result<PathBuf> {
    let candidates = [
        build_dir.join(name),
        build_dir.join(format!("{name}.exe")),
        build_dir.join(format!("Release/{name}")),
        build_dir.join(format!("Release/{name}.exe")),
    ];
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!(
        "could not locate the built {name} executable under {}",
        build_dir.display()
    );
}

/// Write the pinned reference manifest (ORA-02 / D-05 / D-14). Content is a pure
/// function of the recorded constants, so this is idempotent.
fn write_manifest(path: &Path) -> Result<()> {
    let content = format!(
        "# Reference Manifest — LightGBM-rs Oracle (Phases 1-2)\n\
\n\
This file pins the C++ reference build used to generate the committed RNG\n\
golden set (`rng_sequence.txt`). It records everything needed to reproduce the\n\
fixtures deterministically (ORA-02, D-05, D-14). Normal `cargo test` reads the\n\
committed fixtures and needs NONE of this; only `cargo run -p xtask -- regen`\n\
does (D-06).\n\
\n\
## Pinned C++ Reference\n\
\n\
- **Submodule:** `LightGBM/` (in-repo, read-only)\n\
- **Commit:** `{commit}`\n\
- **Version (`VERSION.txt`):** `{version}`\n\
\n\
## Deterministic Build / Capture Flags\n\
\n\
- `deterministic=true`\n\
- `force_row_wise=true`\n\
- `num_threads=1`\n\
- default `float` width — `SCORE_T_USE_DOUBLE` / `LABEL_T_USE_DOUBLE` NOT defined (D-01)\n\
- CPU-only build: `USE_GPU=OFF USE_CUDA=OFF USE_MPI=OFF USE_SWIG=OFF BUILD_CLI=OFF`\n\
\n\
> The RNG (`LightGBM::Random`) is a self-contained, header-only LCG, so its draws\n\
> do not depend on the threading/row-wise/build flags above. The RNG golden is\n\
> therefore captured by compiling `rng_capture` DIRECTLY against the pinned\n\
> `include/LightGBM/utils/random.h` (default f32 width) — no `lib_lightgbm` build\n\
> or link (the in-repo submodule's `external_libs/` are not vendored). The\n\
> deterministic CPU-only flags above are recorded because the same pinned\n\
> reference build is the source of truth for all later (training) goldens; this\n\
> manifest is the single source of truth for that reference configuration.\n\
\n\
## Exact Regeneration Command\n\
\n\
```bash\n\
cargo run -p xtask -- regen\n\
```\n\
\n\
which internally runs (standalone CMake, never modifying the submodule tree):\n\
\n\
```bash\n\
cmake -S xtask/cpp -B target/xtask-cpp-build \\\n\
  -DLIGHTGBM_DIR=<repo>/LightGBM -DCMAKE_BUILD_TYPE=Release\n\
cmake --build target/xtask-cpp-build --target rng_capture --config Release\n\
target/xtask-cpp-build/rng_capture \\\n\
  crates/oracle-harness/fixtures/rng_sequence.txt {master_seed} {n_rng} {n_sample}\n\
```\n\
\n\
## Randomized-at-Capture Case Set (D-14)\n\
\n\
The golden set is derived deterministically from ONE recorded master seed (no\n\
wall-clock / OS entropy), so regeneration is idempotent (empty `git diff`).\n\
\n\
- **Master seed:** `{master_seed}` (`0x{master_seed_hex:08X}`)\n\
- **RNG cases:** `{n_rng}` (many random LCG seeds; each emits NextShort / NextInt /\n\
  NextFloat / NextInt draw sequences in a fixed order)\n\
- **Sample cases:** `{n_sample}` (randomized `(N, K)` pairs straddling the\n\
  `K > N / log2(K)` branch boundary — small-K set branch, large-K streaming\n\
  branch, and near-boundary)\n\
- **Total generated cases:** `{total}`\n\
\n\
## Fixture Format (`rng_sequence.txt`)\n\
\n\
Line-delimited text (diff-friendly, no serde). `#`-prefixed lines are comments.\n\
\n\
```\n\
MASTER_SEED <seed>\n\
COUNTS rng=<n> sample=<n>\n\
RNG seed=<s> int16=<a;b;...> int32=<...> float=<bits;...> int=<...>\n\
SAMPLE seed=<s> N=<n> K=<k> result=<v0;v1;...>\n\
```\n\
\n\
`float` values are the raw little-endian f32 bit pattern (a decimal `u32`) so the\n\
Rust parity test asserts exact-bit f32 equality; integer draws are compared\n\
exactly; `Sample` output is compared as an exact ordered sequence.\n\
\n\
## Numeric Binning Golden Set (Phase 2, layers 1+2)\n\
\n\
Captured by `cargo run -p xtask -- bin-capture` into\n\
`crates/lgbm-dataset/tests/fixtures/numeric_binning.txt`. Covers the NUMERIC\n\
`BinMapper::FindBin` (layer 1: `bin_upper_bound_`, `num_bin`, `bin_type`,\n\
`missing_type`, `default_bin`, `most_freq_bin`, `is_trivial`) and per-row\n\
`ValueToBin` (layer 2). Categorical folding and EFB are OUT OF SCOPE here\n\
(categorical -> Plan 03, EFB -> Plan 05).\n\
\n\
- **Binning master seed:** `{bin_master_seed}` (`0x{bin_master_seed_hex:08X}`) —\n\
  the SINGLE source of randomness for the binning corpus (idempotent regen).\n\
- **Corpus (four-source, D-06; numeric subset):**\n\
  1. synthetic randomized distributions sweeping `max_bin` (2/16/64/255),\n\
     `min_data_in_bin` (1/3/20), and `bin_construct_sample_cnt` (64/256/100000),\n\
     each with a randomized `data_random_seed`;\n\
  2. curated numeric edge battery: NaN-as-missing, +0.0/-0.0 signed zeros,\n\
     on-boundary ties, all-missing, single-value, all-zero, zero-as-missing,\n\
     a pre-filter-triggering column, and a dense 500-value column.\n\
  (LightGBM example datasets and the categorical/EFB corpus land in later plans.)\n\
\n\
### EXACT comparison discipline (NOT the ~1e-6 oracle tolerance)\n\
\n\
Binning goldens are compared **bit-exact**, never within the `~1e-6` oracle\n\
tolerance: per-row bin indices via `compare_exact_u32`, the f64\n\
`bin_upper_bound_` array via `compare_exact_f64_bits` (`.to_bits()` per element),\n\
and storage-layout bytes (later plans) via `compare_exact_bytes`. A 1-ULP\n\
boundary drift is a real divergence, so exact f64-bit equality is mandatory.\n\
\n\
### Capture-harness note (external_libs unavailable)\n\
\n\
The authoritative `BinMapper::FindBin`/`ValueToBin` in `src/io/bin.cpp` pull in\n\
`common.h` -> `fast_double_parser.h` + `fmt/format.h` from `external_libs/`,\n\
which are present here only as EMPTY directories (the LightGBM tree is\n\
git-untracked and its submodules are not vendored). `bin.cpp` is therefore\n\
unbuildable in this environment. `xtask/cpp/bin_capture.cpp` VERBATIM-transcribes\n\
the numeric FindBin family from the pinned `bin.cpp`/`bin.h` (commit `{commit}`,\n\
version `{version}`) using the genuine `std::nextafter` (== `GetDoubleUpperBound`)\n\
and the asymmetric `b <= nextafter(a)` dedup — so it emits goldens byte-identical\n\
to lib_lightgbm — and links only the header-only reference `Random` for sampling.\n\
This mirrors the Phase-1 header-only `rng_capture` discipline.\n\
\n\
### Exact bin-capture command\n\
\n\
```bash\n\
cargo run -p xtask -- bin-capture\n\
```\n",
        commit = LIGHTGBM_COMMIT,
        version = LIGHTGBM_VERSION,
        master_seed = MASTER_SEED,
        master_seed_hex = MASTER_SEED as u32,
        n_rng = N_RNG_CASES,
        n_sample = N_SAMPLE_CASES,
        total = N_RNG_CASES + N_SAMPLE_CASES,
        bin_master_seed = BIN_MASTER_SEED,
        bin_master_seed_hex = BIN_MASTER_SEED as u32,
    );
    std::fs::write(path, content)
        .with_context(|| format!("writing manifest {}", path.display()))?;
    Ok(())
}
