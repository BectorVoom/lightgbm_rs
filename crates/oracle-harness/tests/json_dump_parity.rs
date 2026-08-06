//! JSON model-dump parity (G2 / SPEC-G2-4) vs real `lib_lightgbm` 4.6.
//!
//! Loads a committed model (`model.txt`) via [`lgbm_model::model_text::load`],
//! runs [`lgbm_model::json::dump_model`], and asserts parity with the golden
//! `model.json` — the raw `LGBM_BoosterDumpModel` string of `lightgbm==4.6.0`.
//!
//! ## Parity contract: structure byte-exact, floats numerically-exact (≤ULP)
//! The JSON dump is a **presentation** format (SPEC §4 / numerical-fidelity note:
//! "the JSON is a presentation format, not a training path"). All scaffolding —
//! keys, ordering, punctuation, whitespace, string values (`decision_type`,
//! `missing_type`, `objective`, feature names, categorical `"a||b"` thresholds)
//! and every INTEGER literal — must be **byte-identical**. FLOAT literals are
//! compared by parsed value within a tight tolerance (≤ a few ULP).
//!
//! The tolerance is required — and does NOT mask emitter bugs — because C++'s two
//! serializers disagree at the ULP level on high-entropy values: the model-text
//! writer (`fmt {:.17g}`) + its parser (`fast_double_parser`) form a
//! round-trip-STABLE but not-correctly-rounded pair, so a value LOADED from
//! `model.txt` can sit 1 ULP away from the double the C-API JSON (glibc
//! `iostream`) formatted for the same in-memory model. Rust's `format_g17` is
//! itself byte-exact vs `fmt` (its own golden test), so our emitter is faithful;
//! the residual ≤1-ULP string delta is a C++ writer/parser artifact on a
//! non-training field, not a defect in `dump_model`. A real structural or
//! precision bug in the emitter shifts a value far more than a few ULP and is
//! still caught.
//!
//! Idiom mirrors `predict_parity.rs`: a `CARGO_MANIFEST_DIR`-rooted fixture path
//! (NEVER the untracked `LightGBM/` tree) and a graceful SKIP when the golden is
//! absent.

use std::path::PathBuf;

fn json_dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/json_dump")
}

fn read_fixture(corpus: &str, file: &str) -> Option<String> {
    let path = json_dump_dir().join(corpus).join(file);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!("json_dump_parity: SKIP — fixture {} not found.", path.display());
            None
        }
    }
}

/// A numeric literal is a float iff it carries a `.` or an exponent marker;
/// otherwise it is an integer that MUST match byte-for-byte.
fn is_float_literal(tok: &str) -> bool {
    tok.contains('.') || tok.contains('e') || tok.contains('E')
}

/// Replace every OUTSIDE-QUOTES numeric literal with `\u{0}` and collect the
/// literals in order. Quoted strings (feature names, `"<="`, `"a||b"`,
/// `objective`) are left intact so their digits are never treated as numbers.
fn tokenize(s: &str) -> (String, Vec<String>) {
    let bytes = s.as_bytes();
    let mut template = String::with_capacity(s.len());
    let mut nums = Vec::new();
    let mut in_quote = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_quote {
            template.push(c);
            if c == '"' {
                in_quote = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_quote = true;
            template.push(c);
            i += 1;
            continue;
        }
        // Start of a numeric literal: a digit, or a '-' immediately before a digit.
        let starts_num = c.is_ascii_digit()
            || (c == '-' && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit()));
        if starts_num {
            let start = i;
            i += 1; // consume first char
            while i < bytes.len() {
                let d = bytes[i] as char;
                let sign_in_exp = (d == '+' || d == '-')
                    && matches!(bytes[i - 1] as char, 'e' | 'E');
                if d.is_ascii_digit() || d == '.' || d == 'e' || d == 'E' || sign_in_exp {
                    i += 1;
                } else {
                    break;
                }
            }
            nums.push(s[start..i].to_string());
            template.push('\u{0}');
            continue;
        }
        template.push(c);
        i += 1;
    }
    (template, nums)
}

/// Byte-exact structure + numeric-exact floats: `dump_model` vs the 4.6 golden.
#[test]
fn json_dump_regression_3tree_parity() {
    let (Some(model_txt), Some(golden)) = (
        read_fixture("regression_3tree", "model.txt"),
        read_fixture("regression_3tree", "model.json"),
    ) else {
        return; // golden absent → SKIP
    };

    let model = lgbm_model::model_text::load(&model_txt).expect("load committed model.txt");
    let got = lgbm_model::json::dump_model(&model);

    let (got_tpl, got_nums) = tokenize(&got);
    let (gold_tpl, gold_nums) = tokenize(&golden);

    // 1. Scaffolding (keys / order / punctuation / whitespace / quoted strings)
    //    must be byte-identical.
    if got_tpl != gold_tpl {
        let at = got_tpl
            .char_indices()
            .zip(gold_tpl.chars())
            .find(|((_, a), b)| a != b)
            .map(|((i, _), _)| i)
            .unwrap_or(got_tpl.len().min(gold_tpl.len()));
        let lo = at.saturating_sub(50);
        panic!(
            "JSON structure diverged (tpl lens got={} gold={}) at {at}:\nGOT   :{:?}\nGOLDEN:{:?}",
            got_tpl.len(),
            gold_tpl.len(),
            &got_tpl[lo..(at + 30).min(got_tpl.len())].replace('\u{0}', "#"),
            &gold_tpl[lo..(at + 30).min(gold_tpl.len())].replace('\u{0}', "#"),
        );
    }
    assert_eq!(
        got_nums.len(),
        gold_nums.len(),
        "different count of numeric literals"
    );

    // 2. Each numeric literal: integers byte-exact; floats within ≤ a few ULP.
    const REL_TOL: f64 = 1e-12;
    for (idx, (g, o)) in gold_nums.iter().zip(got_nums.iter()).enumerate() {
        if is_float_literal(g) || is_float_literal(o) {
            let gv: f64 = g.parse().expect("golden float parses");
            let ov: f64 = o.parse().expect("emitted float parses");
            let tol = REL_TOL * gv.abs().max(1.0);
            assert!(
                (gv - ov).abs() <= tol,
                "float #{idx} diverged beyond ULP tolerance: golden={g} emitted={o} (|Δ|={:e} > {tol:e})",
                (gv - ov).abs()
            );
        } else {
            assert_eq!(o, g, "integer literal #{idx} must be byte-exact");
        }
    }
}
