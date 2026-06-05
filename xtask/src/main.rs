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

/// Pinned LightGBM submodule commit (recorded in the manifest, ORA-02 / D-05).
pub const LIGHTGBM_COMMIT: &str = "195c26fc7b00eb0fec252dfe841e2e66d6833954";

/// Pinned LightGBM version (`LightGBM/VERSION.txt`).
pub const LIGHTGBM_VERSION: &str = "4.6.0.99";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("regen") => regen(),
        Some(other) => {
            bail!("unknown subcommand `{other}` (try: regen)");
        }
        None => {
            eprintln!("usage: cargo run -p xtask -- regen");
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

    // 1. Configure the standalone capture build (builds lib_lightgbm too).
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
    let candidates = [
        build_dir.join("rng_capture"),
        build_dir.join("rng_capture.exe"),
        build_dir.join("Release/rng_capture"),
        build_dir.join("Release/rng_capture.exe"),
    ];
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!(
        "could not locate the built rng_capture executable under {}",
        build_dir.display()
    );
}

/// Write the pinned reference manifest (ORA-02 / D-05 / D-14). Content is a pure
/// function of the recorded constants, so this is idempotent.
fn write_manifest(path: &Path) -> Result<()> {
    let content = format!(
        "# Reference Manifest — LightGBM-rs Oracle (Phase 1)\n\
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
> The RNG (`LightGBM::Random`) is a header-only LCG, so its draws do not depend\n\
> on the threading/row-wise flags above; those flags are recorded because the\n\
> same pinned, deterministic build is the reference for all later (training)\n\
> goldens, and the manifest is the single source of truth for the reference\n\
> build configuration.\n\
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
exactly; `Sample` output is compared as an exact ordered sequence.\n",
        commit = LIGHTGBM_COMMIT,
        version = LIGHTGBM_VERSION,
        master_seed = MASTER_SEED,
        master_seed_hex = MASTER_SEED as u32,
        n_rng = N_RNG_CASES,
        n_sample = N_SAMPLE_CASES,
        total = N_RNG_CASES + N_SAMPLE_CASES,
    );
    std::fs::write(path, content)
        .with_context(|| format!("writing manifest {}", path.display()))?;
    Ok(())
}
