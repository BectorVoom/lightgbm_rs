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

use anyhow::{Context, Result, bail};

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

/// The recorded master seed for the Phase-4 compute-kernel histogram golden
/// corpus (D-02a). Like the other master seeds it is the SINGLE source of
/// randomness for the synthetic histogram cases (bin layouts, grad/hess spread),
/// so `kernel-capture` is byte-idempotent (empty `git diff`). Recorded in
/// REFERENCE_MANIFEST.md.
pub const KERNEL_MASTER_SEED: i32 = 0x4157_F00D;

/// The recorded master seed for the Phase-5 serial tree-learner golden corpus
/// (D-06 per-split / D-07 per-tree). Like the other master seeds it is the SINGLE
/// source of randomness for the learner cases, so `learner-capture` is
/// byte-idempotent (empty `git diff`). Recorded in REFERENCE_MANIFEST.md. Plan
/// 05-03 emits the real `spine.txt` from the fixed (hand-crafted, NOT
/// RNG-derived) synthetic corpus; the seed is recorded for format continuity.
pub const LEARNER_MASTER_SEED: i32 = 0x1EA6_5EED;

/// Pinned LightGBM submodule commit (recorded in the manifest, ORA-02 / D-05).
pub const LIGHTGBM_COMMIT: &str = "195c26fc7b00eb0fec252dfe841e2e66d6833954";

/// Pinned LightGBM version (`LightGBM/VERSION.txt`).
pub const LIGHTGBM_VERSION: &str = "4.6.0.99";

/// The recorded train seed for the Phase-3 model corpus (D-05). Like the other
/// master seeds it is the SINGLE source of randomness for `model-capture`
/// (combined with `deterministic=true force_row_wise=true num_threads=1` and NO
/// data subsampling), so the captured `.txt` models + predict goldens are
/// byte-idempotent (empty `git diff` on re-run). Recorded in REFERENCE_MANIFEST.md.
pub const MODEL_TRAIN_SEED: i32 = 0x7FFF_FFFF;

/// The pinned pip-`lightgbm` version used to TRAIN + dump the Phase-3 model
/// corpus (RESEARCH Open Q2 path B). The prebuilt wheel ships `lib_lightgbm`
/// with `fmt` baked in, so its `save_model()` output IS the authoritative
/// `version=v4` model text with correct `%.17g` floats. Pinned here + in the
/// manifest; `model-capture` asserts the installed version matches.
pub const MODEL_LIGHTGBM_VERSION: &str = "4.6.0";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("regen") => regen(),
        Some("bin-capture") => bin_capture(),
        Some("model-capture") => model_capture(),
        Some("kernel-capture") => kernel_capture(),
        Some("learner-capture") => learner_capture(),
        Some(other) => {
            bail!(
                "unknown subcommand `{other}` \
                 (try: regen | bin-capture | model-capture | kernel-capture | \
                 learner-capture)"
            );
        }
        None => {
            eprintln!(
                "usage: cargo run -p xtask -- \
                 <regen | bin-capture | model-capture | kernel-capture | \
                 learner-capture>"
            );
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
    if !lightgbm_dir
        .join("include/LightGBM/utils/random.h")
        .is_file()
    {
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
    if !lightgbm_dir
        .join("include/LightGBM/utils/random.h")
        .is_file()
    {
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
    let categorical_fixture_path = fixtures_dir.join("categorical_folding.txt");
    let missing_fixture_path = fixtures_dir.join("missing_edge_cases.txt");
    let metadata_fixture_path = fixtures_dir.join("metadata.txt");
    let efb_fixture_path = fixtures_dir.join("efb_grouping.txt");
    let default_cfg_fixture_path = fixtures_dir.join("default_config_ingest.txt");
    let example_fixture_path = fixtures_dir.join("example_dataset_binning.txt");
    // The COPIED example datasets live under the committed fixtures dir (never the
    // untracked LightGBM/ tree); the C++ harness reads these exact paths.
    let example_inputs = [
        fixtures_dir.join("examples/regression.train"),
        fixtures_dir.join("examples/binary.train"),
    ];

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
    // Verify the copied example fixtures exist before capture (they are committed,
    // never read from the untracked LightGBM/ tree).
    for ex in &example_inputs {
        if !ex.is_file() {
            bail!(
                "example fixture {} not found — copy it from LightGBM/examples/ into the \
                 committed fixtures dir first",
                ex.display()
            );
        }
    }

    let mut cmd = Command::new(&exe);
    cmd.arg(&fixture_path)
        .arg(BIN_MASTER_SEED.to_string())
        .arg(&storage_fixture_path)
        .arg(&categorical_fixture_path)
        .arg(&missing_fixture_path)
        .arg(&metadata_fixture_path)
        .arg(&efb_fixture_path)
        // FIXED positional slot (argv[8]), BEFORE the variadic example tail so the
        // example inputs stay strictly last and the new golden is never consumed
        // as an example input.
        .arg(&default_cfg_fixture_path)
        .arg(&example_fixture_path);
    for ex in &example_inputs {
        cmd.arg(ex);
    }
    run(&mut cmd, "bin_capture")?;

    for p in [
        &fixture_path,
        &storage_fixture_path,
        &categorical_fixture_path,
        &missing_fixture_path,
        &metadata_fixture_path,
        &efb_fixture_path,
        &default_cfg_fixture_path,
        &example_fixture_path,
    ] {
        if !p.is_file() {
            bail!("capture completed but {} was not written", p.display());
        }
    }

    // Refresh the shared reference manifest (regen + bin-capture write the same
    // file; content is a pure function of the recorded constants, idempotent).
    let manifest_path = root
        .join("crates/oracle-harness/fixtures")
        .join("REFERENCE_MANIFEST.md");
    write_manifest(&manifest_path)?;

    eprintln!(
        "xtask bin-capture: done. Wrote {}, {}, {}, {}, {}, {}, {}, and {}.",
        fixture_path.display(),
        storage_fixture_path.display(),
        categorical_fixture_path.display(),
        missing_fixture_path.display(),
        metadata_fixture_path.display(),
        efb_fixture_path.display(),
        default_cfg_fixture_path.display(),
        example_fixture_path.display()
    );
    eprintln!(
        "Re-run `cargo run -p xtask -- bin-capture` and confirm \
         `git diff --stat crates/lgbm-dataset/tests/fixtures/` is empty (idempotent)."
    );
    Ok(())
}

/// Regenerate the Phase-4 compute-kernel histogram golden corpus (D-02 / D-02a).
///
/// Mirrors [`bin_capture`]: configures + builds the standalone `kernel_capture`
/// C++ target (header-only against the pinned `LightGBM/include` for the
/// reference `Random`; the `ConstructHistogram` accumulation bodies are
/// verbatim-transcribed in `kernel_capture.cpp` because the submodule's
/// `external_libs/` are not vendored here — see that file's header), then runs
/// it over the [`KERNEL_MASTER_SEED`]-derived synthetic corpus and writes
/// `crates/oracle-harness/tests/fixtures/kernels/histogram.txt`. Byte-idempotent.
fn kernel_capture() -> Result<()> {
    let root = workspace_root()?;
    verify_toolchain()?;

    let lightgbm_dir = root.join("LightGBM");
    if !lightgbm_dir
        .join("include/LightGBM/utils/random.h")
        .is_file()
    {
        bail!(
            "LightGBM submodule not found at {} (expected include/LightGBM/utils/random.h)",
            lightgbm_dir.display()
        );
    }

    let cpp_dir = root.join("xtask/cpp");
    let build_dir = root.join("target/xtask-cpp-build");
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("creating build dir {}", build_dir.display()))?;

    // Fixtures live under the TRACKED oracle-harness crate dir — NEVER the
    // untracked LightGBM/ tree.
    let fixtures_dir = root.join("crates/oracle-harness/tests/fixtures/kernels");
    std::fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("creating fixtures dir {}", fixtures_dir.display()))?;
    let fixture_path = fixtures_dir.join("histogram.txt");
    // 04-03 additional goldens (split / partition / subtract).
    let split_path = fixtures_dir.join("split.txt");
    let partition_path = fixtures_dir.join("partition.txt");
    let subtract_path = fixtures_dir.join("subtract.txt");

    eprintln!("xtask kernel-capture: configuring C++ capture build ...");
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

    eprintln!("xtask kernel-capture: building kernel_capture ...");
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("kernel_capture")
            .arg("--config")
            .arg("Release"),
        "cmake build",
    )?;

    let exe = locate_exe(&build_dir, "kernel_capture")?;
    eprintln!(
        "xtask kernel-capture: running capture ({}) ...",
        exe.display()
    );
    run(
        Command::new(&exe)
            .arg(&fixture_path)
            .arg(KERNEL_MASTER_SEED.to_string())
            .arg(&split_path)
            .arg(&partition_path)
            .arg(&subtract_path),
        "kernel_capture",
    )?;

    for p in [&fixture_path, &split_path, &partition_path, &subtract_path] {
        if !p.is_file() {
            bail!("capture completed but {} was not written", p.display());
        }
    }

    // Refresh the shared reference manifest (idempotent — pure function of the
    // recorded constants).
    let manifest_path = root
        .join("crates/oracle-harness/fixtures")
        .join("REFERENCE_MANIFEST.md");
    write_manifest(&manifest_path)?;

    eprintln!(
        "xtask kernel-capture: done. Wrote {}, {}, {}, {} and refreshed {}.",
        fixture_path.display(),
        split_path.display(),
        partition_path.display(),
        subtract_path.display(),
        manifest_path.display()
    );
    eprintln!(
        "Re-run `cargo run -p xtask -- kernel-capture` and confirm \
         `git diff --stat crates/oracle-harness/tests/fixtures/kernels/` \
         is empty (byte-idempotent)."
    );
    Ok(())
}

/// Regenerate the Phase-5 serial tree-learner golden corpus (D-06 / D-07).
///
/// Mirrors [`kernel_capture`]: configures + builds the standalone `learner_capture`
/// C++ target (header-only against the pinned `LightGBM/include` for the reference
/// `Random`; the learner growth loop is verbatim-transcribed in
/// `learner_capture.cpp` because the submodule's `external_libs/` are not vendored
/// here — see that file's header), then runs it over the [`LEARNER_MASTER_SEED`]-
/// derived corpus and writes the goldens under
/// `crates/oracle-harness/tests/fixtures/learner/`. Byte-idempotent.
///
/// Plan 05-03 emits the real `spine.txt`: the full verbatim leaf-wise-loop
/// transcription over a fixed synthetic g/h corpus, carrying per-split (D-06)
/// per-bin gain arrays + the per-tree (D-07) grown-tree field set.
fn learner_capture() -> Result<()> {
    let root = workspace_root()?;
    verify_toolchain()?;

    let lightgbm_dir = root.join("LightGBM");
    if !lightgbm_dir
        .join("include/LightGBM/utils/random.h")
        .is_file()
    {
        bail!(
            "LightGBM submodule not found at {} (expected include/LightGBM/utils/random.h)",
            lightgbm_dir.display()
        );
    }

    let cpp_dir = root.join("xtask/cpp");
    let build_dir = root.join("target/xtask-cpp-build");
    std::fs::create_dir_all(&build_dir)
        .with_context(|| format!("creating build dir {}", build_dir.display()))?;

    // Fixtures live under the TRACKED oracle-harness crate dir — NEVER the
    // untracked LightGBM/ tree.
    let fixtures_dir = root.join("crates/oracle-harness/tests/fixtures/learner");
    std::fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("creating fixtures dir {}", fixtures_dir.display()))?;
    // Plan 05-03 emits the real per-split (D-06) + per-tree (D-07) spine golden.
    let spine_path = fixtures_dir.join("spine.txt");

    eprintln!("xtask learner-capture: configuring C++ capture build ...");
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

    eprintln!("xtask learner-capture: building learner_capture ...");
    run(
        Command::new("cmake")
            .arg("--build")
            .arg(&build_dir)
            .arg("--target")
            .arg("learner_capture")
            .arg("--config")
            .arg("Release"),
        "cmake build",
    )?;

    let exe = locate_exe(&build_dir, "learner_capture")?;
    eprintln!(
        "xtask learner-capture: running capture ({}) ...",
        exe.display()
    );
    run(
        Command::new(&exe)
            .arg(&spine_path)
            .arg(LEARNER_MASTER_SEED.to_string()),
        "learner_capture",
    )?;

    if !spine_path.is_file() {
        bail!(
            "capture completed but {} was not written",
            spine_path.display()
        );
    }

    // Refresh the shared reference manifest (idempotent — pure function of the
    // recorded constants).
    let manifest_path = root
        .join("crates/oracle-harness/fixtures")
        .join("REFERENCE_MANIFEST.md");
    write_manifest(&manifest_path)?;

    eprintln!(
        "xtask learner-capture: done. Wrote {} and refreshed {}.",
        spine_path.display(),
        manifest_path.display()
    );
    eprintln!(
        "Re-run `cargo run -p xtask -- learner-capture` and confirm \
         `git diff --stat crates/oracle-harness/tests/fixtures/learner/` \
         is empty (byte-idempotent)."
    );
    Ok(())
}

/// Regenerate the Phase-3 model + predict golden corpus (D-05, PRD-01..PRD-06).
///
/// RESEARCH Open Q2 capture-path resolution: **path B (human-approved)**. The
/// full `lib_lightgbm` is unbuildable here (`external_libs/{fmt,...}` are empty),
/// and Phase 3 has no Rust trainer yet, so the authoritative reference model
/// `.txt` is produced by a pip-installed `lightgbm` (its prebuilt wheel ships
/// `lib_lightgbm` with `fmt` baked in → `save_model()` IS the authoritative
/// `version=v4` text with correct `%.17g` floats). This subcommand shells out to
/// `xtask/py/model_capture.py`, which TRAINS the 5-corpus D-05 set
/// (regression / binary / multiclass / categorical / subrange) on the reused
/// Phase-2 example matrices with `deterministic=true force_row_wise=true
/// num_threads=1 seed=MODEL_TRAIN_SEED` and NO subsampling, then dumps each
/// `model.txt` + per-corpus predict-vector goldens (raw / transformed / leaf /
/// sub-range) and the `format_golden.txt` `%g` battery. Byte-idempotent.
///
/// The pip `lightgbm` is a CAPTURE-time tool only — never a dependency of the
/// shipped crate and never read at `cargo test` time (the fixtures are
/// committed). The interpreter is resolved from `$LGBM_CAPTURE_PYTHON` (or a few
/// common venv locations / `python3`); it must have `lightgbm` importable.
fn model_capture() -> Result<()> {
    let root = workspace_root()?;

    let python = resolve_capture_python()?;
    let script = root.join("xtask/py/model_capture.py");
    if !script.is_file() {
        bail!("capture script {} not found", script.display());
    }

    // Fixtures live under the TRACKED crate dir — NEVER the untracked LightGBM/ tree.
    let models_dir = root.join("crates/lgbm-model/tests/fixtures/models");
    std::fs::create_dir_all(&models_dir)
        .with_context(|| format!("creating models fixtures dir {}", models_dir.display()))?;

    // The reused Phase-2 example matrices (COPIED into the committed fixtures dir).
    let reg_train = root.join("crates/lgbm-dataset/tests/fixtures/examples/regression.train");
    let bin_train = root.join("crates/lgbm-dataset/tests/fixtures/examples/binary.train");
    for ex in [&reg_train, &bin_train] {
        if !ex.is_file() {
            bail!(
                "example fixture {} not found (it is the committed Phase-2 input matrix)",
                ex.display()
            );
        }
    }

    // Verify the interpreter has the recorded lightgbm version before training.
    eprintln!(
        "xtask model-capture: using python {} (lightgbm {} expected) ...",
        python.display(),
        MODEL_LIGHTGBM_VERSION
    );
    run(
        Command::new(&python).arg("-c").arg(format!(
            "import lightgbm,sys; \
             assert lightgbm.__version__=='{ver}', \
             'lightgbm '+lightgbm.__version__+' != recorded {ver}'",
            ver = MODEL_LIGHTGBM_VERSION
        )),
        "lightgbm version check",
    )
    .context(
        "the capture interpreter must have lightgbm importable at the recorded version. \
         Set $LGBM_CAPTURE_PYTHON to a python (e.g. a venv) with \
         `pip install lightgbm` of that version. `cargo test` does NOT need this.",
    )?;

    eprintln!("xtask model-capture: training the D-05 corpus + dumping goldens ...");
    run(
        Command::new(&python)
            .arg(&script)
            .arg(&models_dir)
            .arg(&reg_train)
            .arg(&bin_train)
            .arg(MODEL_TRAIN_SEED.to_string())
            .arg(MODEL_LIGHTGBM_VERSION),
        "model_capture.py",
    )?;

    // Assert every required output landed.
    for corpus in ["regression", "binary", "multiclass", "categorical", "subrange"] {
        let cdir = models_dir.join(corpus);
        for f in ["model.txt", "raw.txt", "transformed.txt", "leaf.txt"] {
            let p = cdir.join(f);
            if !p.is_file() {
                bail!("capture completed but {} was not written", p.display());
            }
        }
    }
    let subrange = models_dir.join("subrange/subrange.txt");
    if !subrange.is_file() {
        bail!("capture completed but {} was not written", subrange.display());
    }
    let format_golden = models_dir.join("format_golden.txt");
    if !format_golden.is_file() {
        bail!("capture completed but {} was not written", format_golden.display());
    }

    // Refresh the shared reference manifest (idempotent — pure function of the
    // recorded constants).
    let manifest_path = root
        .join("crates/oracle-harness/fixtures")
        .join("REFERENCE_MANIFEST.md");
    write_manifest(&manifest_path)?;

    eprintln!(
        "xtask model-capture: done. Wrote 5 corpora + format_golden.txt under {} and refreshed {}.",
        models_dir.display(),
        manifest_path.display()
    );
    eprintln!(
        "Re-run `cargo run -p xtask -- model-capture` and confirm \
         `git diff --stat crates/lgbm-model/tests/fixtures/models \
         crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` is empty (byte-idempotent)."
    );
    Ok(())
}

/// Resolve a python interpreter that can run the capture (lightgbm importable).
///
/// Order: `$LGBM_CAPTURE_PYTHON`, then a few common venv locations, then
/// `python3` on PATH. Returns a clear (non-panic) error naming the override.
fn resolve_capture_python() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("LGBM_CAPTURE_PYTHON") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        bail!(
            "$LGBM_CAPTURE_PYTHON points at {} which is not a file",
            path.display()
        );
    }
    let candidates = [
        PathBuf::from("/tmp/lgbm-capture-venv/bin/python"),
        PathBuf::from("python3"),
    ];
    for c in &candidates {
        // A bare `python3` (no `/`) is resolved via PATH by Command; accept it.
        if c.components().count() == 1 || c.is_file() {
            return Ok(c.clone());
        }
    }
    bail!(
        "no capture python found. Set $LGBM_CAPTURE_PYTHON to a python with \
         `pip install lightgbm` (e.g. a venv)."
    );
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
## EFB Grouping Golden Set (Phase 2, layer 3, DAT-05)\n\
\n\
Captured by `cargo run -p xtask -- bin-capture` into\n\
`crates/lgbm-dataset/tests/fixtures/efb_grouping.txt`. Covers Exclusive Feature\n\
Bundling (layer 3): feature->group membership (`feature2group_` /\n\
`feature2subfeature_`), per-group `bin_offsets_` + `num_total_bin_` + the\n\
`group_is_multi_val` flag, and the per-row bundled bin index per single-value\n\
group. Corpus = D-06 number 4: two mutually-exclusive sparse feature sets (which EFB\n\
bundles into one group each) plus a control where no features are mutually\n\
exclusive (one single-feature group per feature — proves the `enable_bundle`\n\
dispatch boundary).\n\
\n\
### Capture-harness resolution: VERBATIM TRANSCRIPTION (external_libs unvendored)\n\
\n\
The plan flagged a MEDIUM-risk feasibility choice between (a) a focused harness\n\
compiling `src/io/dataset.cpp` directly, and (b) a full-CLI `enable_bundle=true`\n\
dump. **Both nominal options are provably infeasible in this environment:**\n\
\n\
- **(a) focused `dataset.cpp` build — INFEASIBLE.** `dataset.cpp` transitively\n\
  includes `common.h` -> `fast_double_parser.h` + `fmt/format.h` from\n\
  `external_libs/`, which are present here only as EMPTY directories (the\n\
  LightGBM tree is git-untracked and its submodules are unvendored). The build\n\
  fails with `fast_double_parser.h: No such file or directory`.\n\
- **(b) full-CLI dump — INFEASIBLE.** Building `lib_lightgbm` / the `lightgbm`\n\
  CLI requires the same unvendored `external_libs` (`fast_double_parser`, `fmt`,\n\
  `eigen`, `compute`), so the CLI cannot be built either.\n\
\n\
**Resolution (human-approved):** EFB is captured by a HEADER-ONLY VERBATIM\n\
TRANSCRIPTION of the EFB pipeline (`GetConflictCount`/`FindGroups`/\n\
`FastFeatureBundling`/`FixSampleIndices` + the bundled `FeatureGroup` /\n\
`bin_offsets_` / `num_total_bin_` group layout) from the pinned `dataset.cpp`\n\
(commit `{commit}`, version `{version}`) and `feature_group.h`, compiled against\n\
only `-I LightGBM/include` plus the header-only `LightGBM::Random` (sampling +\n\
group shuffle). This is the SAME discipline plans 02-01..02-04 used for every\n\
prior golden layer (numeric / storage / categorical / missing / metadata): no\n\
`external_libs`, no `lib_lightgbm` link, output byte-identical to what\n\
lib_lightgbm would emit because the transcribed code is the authoritative\n\
reference source.\n\
\n\
### Exact bin-capture command\n\
\n\
```bash\n\
cargo run -p xtask -- bin-capture\n\
```\n\
\n\
## Model / Predict Golden Set (Phase 3, D-05 / PRD-01..PRD-06)\n\
\n\
Captured by `cargo run -p xtask -- model-capture` into\n\
`crates/lgbm-model/tests/fixtures/models/{{regression,binary,multiclass,categorical,subrange}}/`.\n\
Each corpus directory holds the authoritative C++ `version=v4` `model.txt`\n\
(`Booster.save_model()`) plus per-corpus predict-vector goldens:\n\
`raw.txt` (PRD-01 raw scores), `transformed.txt` (PRD-02 — sigmoid for binary,\n\
softmax for multiclass, identity for regression), `leaf.txt` (PRD-03 leaf\n\
indices), and (for `subrange`) `subrange.txt` (PRD-06 raw scores for\n\
representative `(start_iteration, num_iteration)` slices incl. `-1 == all\n\
remaining`). The fixed-double `%g` battery for the `format.rs` DAT-09 formatter\n\
is `models/format_golden.txt` (G17 = `{{:.17g}}`, G6 = `{{:g}}`). Float golden\n\
vectors are `;`-separated raw f64 bit patterns (decimal `u64`) for bit-exact\n\
replay; leaf indices are `;`-separated decimal `u32`.\n\
\n\
- **Training tool (capture-time only):** pip `lightgbm` `{model_lgbm_version}`\n\
  (RESEARCH Open Q2 path B). NOT a dependency of the shipped crate and NEVER read\n\
  at `cargo test` time — the fixtures are committed.\n\
- **Train seed:** `{model_train_seed}` (`0x{model_train_seed_hex:08X}`).\n\
- **Deterministic train params:** `deterministic=true force_row_wise=true\n\
  num_threads=1 bagging_freq=0 bagging_fraction=1.0 feature_fraction=1.0\n\
  num_boost_round=10 num_leaves=31 min_data_in_leaf=20` (NO data subsampling), so\n\
  re-running `model-capture` is byte-idempotent (empty `git diff`).\n\
- **Corpora:** regression (`objective=regression`), binary\n\
  (`objective=binary`), multiclass (`objective=multiclass num_class=3`, label\n\
  derived deterministically by tertile-bucketing a stable feature), categorical\n\
  (`objective=binary` with 4 integerized `categorical_feature` columns),\n\
  subrange (a regression model exercising the PRD-06 sub-range slices). The\n\
  regression/binary inputs are the COPIED Phase-2 example matrices under\n\
  `crates/lgbm-dataset/tests/fixtures/examples/` — NEVER the untracked\n\
  `LightGBM/` tree.\n\
\n\
### Capture-path resolution: PATH B (pip lightgbm train + dump), human-approved\n\
\n\
RESEARCH Open Q2 (the FIRST planning gate) offered (A) verbatim transcription of\n\
`SaveModelToString` + a train stub vs (B) pip `lightgbm` train + dump. **Path A\n\
is infeasible standalone here** (Phase 3 has no Rust trainer and the C++ trainer\n\
is unbuildable — `external_libs/{{fmt,fast_double_parser,...}}` are empty), so a\n\
trained `.txt` must come from a prebuilt `lib_lightgbm`. **Path B was selected\n\
and approved:** the pip wheel ships `lib_lightgbm` with `fmt` baked in, so its\n\
`save_model()` IS the authoritative v4 format with correct `%.17g`. The exact\n\
tool version + train params are pinned above; the produced fixtures were\n\
human-approved as numerically identical to `lib_lightgbm` (03-VALIDATION.md\n\
Manual-Only Verifications). The capture interpreter is resolved from\n\
`$LGBM_CAPTURE_PYTHON` (a venv with `pip install lightgbm`).\n\
\n\
### Exact model-capture command\n\
\n\
```bash\n\
LGBM_CAPTURE_PYTHON=/path/to/venv/bin/python cargo run -p xtask -- model-capture\n\
```\n\
\n\
## Kernel Golden Set (Phase 4, D-02 / D-02a)\n\
\n\
Captured by `cargo run -p xtask -- kernel-capture` into\n\
`crates/oracle-harness/tests/fixtures/kernels/histogram.txt`. Covers the D-01\n\
whole-kernel `construct_histograms` op: the stride-2 `[g0,h0,g1,h1,...]` f64\n\
histogram (`hist_t = double`) accumulated from f32 (`score_t = float`) ordered\n\
gradients/hessians over a feature column's per-row bin indices\n\
(`ti = bin << 1`). The cubecl-cpu kernel reproduces this BIT-EXACT (the D-04\n\
deterministic anchor); `crates/oracle-harness/tests/kernel_parity.rs` replays it\n\
via `compare_exact_f64_bits`.\n\
\n\
- **Kernel master seed:** `{kernel_master_seed}` (`0x{kernel_master_seed_hex:08X}`) —\n\
  the SINGLE source of randomness for the histogram corpus (idempotent regen).\n\
- **D-02a path coverage:** dense + sparse bin layouts; the most-frequent /\n\
  default-bin (lowest-bin) routing; multiple bin-store bit widths\n\
  (`DenseBin<u8,4bit>` / `u8` / `u16` / `u32` and the matching `SparseBin`\n\
  widths, selected by `num_bin` per `Bin::CreateDenseBin`/`CreateSparseBin`); an\n\
  all-rows-on-one-bin pileup; an empty-sparse-stream (all-bin-0) round-trip; and\n\
  a grad/hess sign+magnitude spread (~1e-3 .. ~1e3, mixed signs) that stresses\n\
  the non-associative f64 reduction order.\n\
\n\
### Capture-harness note (external_libs unbuildable)\n\
\n\
The authoritative `ConstructHistogram` lives in `src/io/dense_bin.hpp` /\n\
`sparse_bin.hpp`, which (via `<LightGBM/bin.h>` -> `common.h`) transitively pull\n\
in `fast_double_parser.h` + `fmt/format.h` from `external_libs/` — present here\n\
only as EMPTY directories. `xtask/cpp/kernel_capture.cpp` therefore VERBATIM-\n\
transcribes the `ConstructHistogram` accumulation bodies from the pinned\n\
`dense_bin.hpp:130-141` / `sparse_bin.hpp:138-152` (commit `{commit}`, version\n\
`{version}`), reusing the `DenseBin`/`SparseBin` bin-storage forms, and emits\n\
goldens byte-identical to lib_lightgbm. Synthetic inputs use the genuine\n\
header-only `LightGBM::Random`. Same discipline as `rng_capture`/`bin_capture`:\n\
no `external_libs`, no `lib_lightgbm` link, no C++ toolchain at `cargo test` time\n\
(the golden is committed).\n\
\n\
### 04-03 split / partition / subtract goldens\n\
\n\
`kernel-capture` also emits three more goldens under the same kernels dir\n\
(`split.txt`, `partition.txt`, `subtract.txt`), each a VERBATIM transcription of\n\
the pinned reference (commit `{commit}`, version `{version}`):\n\
\n\
- **`split.txt`** — `FindBestThresholdSequentially` + the gain math\n\
  (`feature_histogram.hpp:711-1057`, default CPU template). Each case emits the\n\
  PER-CANDIDATE gains (REVERSE + FORWARD, NaN where a candidate is gated) AND the\n\
  winning `SplitInfo`, so a divergence localizes to the gain scan, not just the\n\
  winner. Covers a REVERSE-branch winner (`default_left=1`, threshold `t-1+offset`),\n\
  a FORWARD-branch winner (`t+offset`), a default-bin-skip case, an L1-regularized\n\
  case, and a no-admissible-split case.\n\
- **`partition.txt`** — `DataPartition::Split` row routing via `SplitInner`\n\
  (`dense_bin.hpp:314-394`, `MissingType::None`) + the stable two-pass gather;\n\
  emits the reordered index array + `split_point`.\n\
- **`subtract.txt`** — `FeatureHistogram::Subtract` (`feature_histogram.hpp:99-145`,\n\
  default `USE_DIST_GRAD=false`): `derived[i] = parent[i] - child[i]`.\n\
\n\
`crates/oracle-harness/tests/kernel_parity.rs` replays all four layers BIT-EXACT\n\
on the cubecl-cpu anchor via `compare_exact_f64_bits` / `compare_exact_u32`.\n\
\n\
### Exact kernel-capture command\n\
\n\
```bash\n\
cargo run -p xtask -- kernel-capture\n\
```\n\
\n\
## Learner Golden Set (Phase 5, D-06 per-split / D-07 per-tree)\n\
\n\
Captured by `cargo run -p xtask -- learner-capture` into\n\
`crates/oracle-harness/tests/fixtures/learner/`. Covers the serial tree-learner\n\
growth: PER-SPLIT snapshots (D-06 — the full per-bin gain array + winning split,\n\
so a divergence localizes to the gain scan) and PER-TREE goldens (D-07 — the\n\
grown tree's `Tree::to_string()` text, compared via the Phase-3 `%.17g` machinery\n\
as a `String`). `crates/oracle-harness/tests/learner_parity.rs` replays them\n\
bit-exact on the cubecl-cpu anchor (`compare_exact_f64_bits` per-split) / string\n\
equality (per-tree).\n\
\n\
- **Learner master seed:** `{learner_master_seed}` (`0x{learner_master_seed_hex:08X}`) —\n\
  recorded for format continuity; the Plan-05-03 corpus is hand-crafted (fixed\n\
  synthetic g/h), NOT RNG-derived, so the capture is byte-idempotent regardless.\n\
- **Plan 05-03 status: REAL SPINE GOLDEN (`spine.txt`).** The full verbatim\n\
  leaf-wise-loop transcription grows a tree over a FIXED 12-row / 2-feature\n\
  synthetic g/h corpus (`force_row_wise`, `feature_fraction=1.0`,\n\
  `missing_type=None` per RESEARCH A5 — NA_AS_MISSING deferred). It emits 10\n\
  PSPLIT records (per-bin REVERSE+FORWARD gain arrays per candidate feature at\n\
  every split decision, D-06) + 1 PTREE record (the grown 4-leaf tree's field set\n\
  as raw bits, D-07). `learner_parity.rs` replays per-split bit-exact, full-tree\n\
  via the shared `%.17g` formatter, the subtraction trick, missing/zero routing,\n\
  and the D-02a kernel-vs-learner cross-check.\n\
\n\
### Record format (`spine.txt`)\n\
\n\
```\n\
LEARNER_MASTER_SEED <seed>\n\
COUNTS splits=<n> trees=<n>\n\
PSPLIT split=<i> leaf=<l> feature=<f> num_bin=<n> rev=<f64bits;...> fwd=<f64bits;...> winner=<f64bits>\n\
PTREE name=<id> num_leaves=<n>\n\
PT_SPLIT_FEATURE <i...>  PT_THRESHOLD_BITS <u64...>  PT_DECISION_TYPE <i...>\n\
PT_SPLIT_GAIN_BITS <u32...>  PT_LEFT_CHILD <i...>  PT_RIGHT_CHILD <i...>\n\
PT_LEAF_VALUE_BITS <u64...>  PT_LEAF_WEIGHT_BITS <u64...>  PT_LEAF_COUNT <i...>\n\
PT_INTERNAL_VALUE_BITS <u64...>  PT_INTERNAL_COUNT <i...>\n\
ENDTREE\n\
```\n\
\n\
`rev`/`fwd`/`winner` + the PT_*_BITS lines are raw little-endian f64/f32 bit\n\
patterns (decimal `u64`/`u32`) for bit-exact replay; the Rust side reconstructs\n\
the reference `Tree` from the PT_* fields and serializes it via the shared\n\
`lgbm-model` `%.17g` formatter for the D-07 String compare.\n\
\n\
### Capture-harness note (external_libs unbuildable)\n\
\n\
The authoritative `SerialTreeLearner` lives in\n\
`src/treelearner/serial_tree_learner.cpp`, which (via `<LightGBM/dataset.h>` ->\n\
`common.h`) transitively #includes `fast_double_parser.h` + `fmt/format.h` from\n\
`external_libs/` — present here only as EMPTY directories. `learner_capture.cpp`\n\
therefore VERBATIM-transcribes the learner growth loop (Plan 05-03) from the\n\
pinned `serial_tree_learner.cpp` (commit `{commit}`, version `{version}`),\n\
reusing `kernel_capture.cpp`'s already-transcribed gain/split math (D-02a\n\
cross-check), and includes the header-only `LightGBM/include` only for the genuine\n\
reference `Random`. Same discipline as `rng_capture`/`bin_capture`/`kernel_capture`:\n\
no `external_libs`, no `lib_lightgbm` link, no C++ toolchain at `cargo test` time\n\
(the golden is committed).\n\
\n\
### Exact learner-capture command\n\
\n\
```bash\n\
cargo run -p xtask -- learner-capture\n\
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
        kernel_master_seed = KERNEL_MASTER_SEED,
        kernel_master_seed_hex = KERNEL_MASTER_SEED as u32,
        learner_master_seed = LEARNER_MASTER_SEED,
        learner_master_seed_hex = LEARNER_MASTER_SEED as u32,
        model_train_seed = MODEL_TRAIN_SEED,
        model_train_seed_hex = MODEL_TRAIN_SEED as u32,
        model_lgbm_version = MODEL_LIGHTGBM_VERSION,
    );
    std::fs::write(path, content)
        .with_context(|| format!("writing manifest {}", path.display()))?;
    Ok(())
}
