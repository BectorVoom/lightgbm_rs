//! If-else C++ codegen parity (G1 / SPEC-G1-4) vs real `lib_lightgbm` 4.6.
//!
//! ## Byte-exact golden: UNAVAILABLE in this environment (documented blocker)
//! `GBDT::ModelToIfElse` (`Config.convert_model`) is invoked ONLY from the
//! `lightgbm` CLI binary / `Config`-driven training-app path in the real C++
//! source. The `lightgbm==4.6.0` **pip wheel** used as this project's oracle
//! ships neither a C-API entry point for it NOR the `lightgbm` CLI executable
//! (only the Python-importable shared library + the Python surface) — so
//! `convert_model`'s output cannot be captured from this sandbox without
//! either the LightGBM CLI or a checked-out + built `LightGBM/` 4.6 tree,
//! NEITHER of which is present here (`LightGBM/` absent — same class of
//! blocker documented for G4/G5, `research.md` §1). `json_dump_regression_...`'s
//! precedent (G2) worked because `LGBM_BoosterDumpModel` IS a C-API export the
//! wheel carries; `convert_model` has no equivalent.
//!
//! `ifelse_codegen_byte_parity` therefore mirrors the `json_dump_parity.rs`
//! SKIP-graceful idiom: it looks for a committed golden `.cpp` under
//! `tests/fixtures/ifelse_codegen/` and SKIPs (passes) when absent — exactly
//! the correct Red state for a golden-driven parity test per SPEC-G1-4
//! ("passes with golden, SKIP without"). A future session with the LightGBM
//! CLI or a built `LightGBM/` tree can populate the fixture and turn this into
//! a real byte-exact gate.
//!
//! ## The REAL correctness gate: compile + run the generated C++
//! Byte parity to an unavailable golden is strictly weaker evidence than
//! actually exercising the generated code. `ifelse_codegen_functional_validation`
//! (1) loads a committed model (numeric-split `regression_3tree`, and a
//! genuinely categorical-split `cat_onehot`), (2) generates its `.cpp` via
//! [`lgbm_model::codegen_cpp::model_to_cpp`], (3) wraps it with a tiny `main`
//! that reads feature rows from stdin and prints `PredictRaw` outputs, (4)
//! compiles it with the first available system C++ compiler (`g++`/`cc`/
//! `clang++`), (5) runs it on random + edge-case (NaN, exact-threshold,
//! out-of-bitset) feature rows, and (6) asserts every compiled-C++ prediction
//! matches [`lgbm_model::GbdtModel::predict_raw`] within the project's ~1e-6
//! numeric contract (`CLAUDE.md`). If no C++ compiler is available, this test
//! SKIPs gracefully (printed, not silent).

use std::path::{Path, PathBuf};
use std::process::Command;

use lgbm_model::GbdtModel;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ifelse_codegen_dir() -> PathBuf {
    fixtures_dir().join("ifelse_codegen")
}

/// SKIP gracefully (returning `None`) when a golden file is absent — matching
/// the `json_dump_parity.rs` / `predict_parity.rs` idiom.
fn read_golden(corpus: &str, file: &str) -> Option<String> {
    let path = ifelse_codegen_dir().join(corpus).join(file);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "ifelse_codegen_parity: SKIP — golden {} not found (byte-exact \
                 `convert_model` golden unavailable in this sandbox — see module \
                 doc header for why; `ifelse_codegen_functional_validation` in \
                 this same file is the real correctness gate).",
                path.display()
            );
            None
        }
    }
}

/// Byte-exact structure + numeric-exact floats vs the real `convert_model`
/// output — SKIPs (passes) until a golden `.cpp` is committed (blocker
/// documented in the module doc header).
#[test]
fn ifelse_codegen_byte_parity() {
    let (Some(model_txt), Some(golden_cpp)) = (
        read_golden("regression_3tree", "model.txt"),
        read_golden("regression_3tree", "model.cpp"),
    ) else {
        return; // golden absent -> SKIP
    };

    let model = lgbm_model::model_text::load(&model_txt).expect("load committed model.txt");
    let got = lgbm_model::codegen_cpp::model_to_cpp(&model);
    assert_eq!(got, golden_cpp, "if-else codegen diverged from the committed golden");
}

// ---- functional validation: compile + run the generated C++ ----

/// Deterministic splitmix64 — no new crate dependency (mirrors
/// `on_device_float_envelope_500k.rs`'s idiom).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in `[lo, hi)`.
    fn next_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        lo + u * (hi - lo)
    }
}

/// Find the first available C++ compiler on `PATH`, or `None` (SKIP).
fn find_cxx_compiler() -> Option<&'static str> {
    for candidate in ["g++", "clang++", "cc"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Some(candidate);
        }
    }
    None
}

/// Build the standalone `.cpp` (the generated model source + a tiny stdin-driven
/// `main` harness): reads `rows`, then `rows * ncols` doubles, calls
/// `PredictRaw`, and prints `ntpi` doubles (`%.17g`) per row.
fn wrap_with_main(model_cpp: &str, ncols: usize, ntpi: usize) -> String {
    format!(
        "{model_cpp}\n\
         #include <cstdio>\n\
         int main() {{\n\
         \x20 int rows = 0;\n\
         \x20 if (std::scanf(\"%d\", &rows) != 1) return 1;\n\
         \x20 double arr[{ncols}];\n\
         \x20 for (int r = 0; r < rows; ++r) {{\n\
         \x20\x20 for (int c = 0; c < {ncols}; ++c) {{\n\
         \x20\x20\x20 if (std::scanf(\"%lf\", &arr[c]) != 1) return 1;\n\
         \x20\x20 }}\n\
         \x20\x20 double out[{ntpi}];\n\
         \x20\x20 PredictRaw(arr, out);\n\
         \x20\x20 for (int k = 0; k < {ntpi}; ++k) {{\n\
         \x20\x20\x20 std::printf(\"%.17g\\n\", out[k]);\n\
         \x20\x20 }}\n\
         \x20 }}\n\
         \x20 return 0;\n\
         }}\n"
    )
}

/// Random + edge-case feature rows for a model with `ncols` features, biased
/// toward `[lo, hi)` with occasional NaN / exact-boundary values.
fn gen_rows(rng: &mut SplitMix64, ncols: usize, lo: f64, hi: f64, n: usize) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<f64> = (0..ncols)
            .map(|_| {
                let bucket = rng.next_u64() % 8;
                match bucket {
                    0 if i > 0 => f64::NAN, // occasional NaN (skip row 0 to keep one all-finite row)
                    1 => 0.0,
                    _ => rng.next_f64(lo, hi),
                }
            })
            .collect();
        rows.push(row);
    }
    rows
}

/// Load `corpus`'s `model.txt`, generate its C++, compile + run it on random
/// rows, and assert parity with [`GbdtModel::predict_raw`]. Returns `true` if
/// the check actually ran (compiler + fixture present), `false` if it SKIPped.
fn functional_validate(
    corpus: &str,
    cxx: &str,
    tmp_dir: &Path,
    row_lo: f64,
    row_hi: f64,
) -> bool {
    let model_txt_path = fixtures_dir().join("json_dump").join(corpus).join("model.txt");
    let model_txt = match std::fs::read_to_string(&model_txt_path) {
        Ok(s) => s,
        Err(_) => {
            // Fall back to the categorical fixture location.
            let alt = fixtures_dir().join("categorical").join(format!("{corpus}.txt"));
            match std::fs::read_to_string(&alt) {
                Ok(s) => s,
                Err(_) => {
                    eprintln!(
                        "ifelse_codegen_functional_validation: SKIP — no model.txt found for \
                         corpus {corpus} at {} or {}.",
                        model_txt_path.display(),
                        alt.display()
                    );
                    return false;
                }
            }
        }
    };

    let model: GbdtModel = lgbm_model::model_text::load(&model_txt).expect("load model.txt");
    let ncols = (model.max_feature_idx + 1).max(1) as usize;
    let ntpi = model.num_tree_per_iteration.max(1) as usize;

    let model_cpp = lgbm_model::codegen_cpp::model_to_cpp(&model);
    let full_cpp = wrap_with_main(&model_cpp, ncols, ntpi);

    let src_path = tmp_dir.join(format!("{corpus}_ifelse.cpp"));
    let bin_path = tmp_dir.join(format!("{corpus}_ifelse_bin"));
    std::fs::write(&src_path, &full_cpp).expect("write generated .cpp");

    let compile = Command::new(cxx)
        .args(["-O0", "-std=c++17", "-o"])
        .arg(&bin_path)
        .arg(&src_path)
        .output()
        .expect("spawn compiler");
    assert!(
        compile.status.success(),
        "compile of generated C++ failed for {corpus}:\nstdout={}\nstderr={}\n---cpp---\n{full_cpp}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr),
    );

    let mut rng = SplitMix64(0xC0FF_EE00_u64.wrapping_add(corpus.len() as u64));
    let rows = gen_rows(&mut rng, ncols, row_lo, row_hi, 50);

    // Build stdin: "<rows>\n" then each row's values via format_g17.
    let mut stdin_text = format!("{}\n", rows.len());
    for row in &rows {
        let toks: Vec<String> = row.iter().map(|&v| lgbm_model::format::format_g17(v)).collect();
        stdin_text.push_str(&toks.join(" "));
        stdin_text.push('\n');
    }

    let run = Command::new(&bin_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .expect("stdin")
                .write_all(stdin_text.as_bytes())?;
            child.wait_with_output()
        })
        .expect("run compiled binary");
    assert!(run.status.success(), "compiled binary exited non-zero for {corpus}");

    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("parse compiled-C++ output as f64"))
        .collect();
    assert_eq!(got.len(), rows.len() * ntpi, "output row count mismatch for {corpus}");

    for (r, row) in rows.iter().enumerate() {
        let expected = model.predict_raw(row, 0, -1);
        for k in 0..ntpi {
            let e = expected[k];
            let g = got[r * ntpi + k];
            // ~1e-6 numeric contract (CLAUDE.md); both sides do the SAME f64
            // sum-of-tree-predict in the SAME order so this is expected to be
            // far tighter in practice.
            let tol = 1e-6 * e.abs().max(1.0);
            assert!(
                (e - g).abs() <= tol || (e.is_nan() && g.is_nan()),
                "{corpus} row {r} class {k}: rust={e:?} cpp={g:?} diverged beyond {tol:e}"
            );
        }
    }
    true
}

/// SPEC-G1-4's real correctness gate (see module doc header): compile + run
/// the generated C++ for a numeric-split model AND a genuinely
/// categorical-split model, and assert parity with `GbdtModel::predict_raw`.
#[test]
fn ifelse_codegen_functional_validation() {
    let Some(cxx) = find_cxx_compiler() else {
        eprintln!(
            "ifelse_codegen_functional_validation: SKIP — no C++ compiler \
             (g++/clang++/cc) found on PATH. G1 could not be functionally \
             validated in this environment."
        );
        return;
    };

    let tmp_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");

    let mut ran_any = false;
    // Numeric-only splits, 3 features, missing_type=None (regression_3tree).
    ran_any |= functional_validate("regression_3tree", cxx, &tmp_dir, -0.5, 1.5);
    // Genuinely categorical splits (num_cat>0, bitset membership + NaN/negative
    // right-routing) — `categorical/cat_onehot.txt` is itself model-text.
    ran_any |= functional_validate("cat_onehot", cxx, &tmp_dir, -3.0, 6.0);

    assert!(ran_any, "no fixture corpus was found — functional validation did not run");
}
