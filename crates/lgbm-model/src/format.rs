//! Printf-faithful `%g` float formatters — the DAT-09 serialization linchpin.
//!
//! LightGBM's model-text writer formats every float through `fmt`'s `{:.17g}`
//! (high precision) or `{:g}` (6 sig digits), via `CommonC::ArrayToString` →
//! `__TToStringHelper<T,true,true>` → `fmt::format_to_n(buf, len, "{:.17g}", v)`
//! (`common.h:1227`) and `__TToStringHelper<T,true,false>` → `{:g}`
//! (`common.h:1220`). `fmt`'s `{:g}` follows C printf `%g` semantics.
//!
//! A faithful byte-exact write therefore REQUIRES a `%g` formatter, NOT Rust's
//! `f64::to_string()` / `ryu` (shortest round-trip — emits `0.1`, not the C++
//! `0.10000000000000001`) and NOT `{:.17e}` (always exponential). This module
//! hand-rolls the `%g` algorithm at precision 17 ([`format_g17`]) and 6
//! ([`format_g6`]). See RESEARCH Pitfall 1 (`03-RESEARCH.md:295-299`).
//!
//! ## Which model field uses which formatter
//! - `{:.17g}` ([`format_g17`]): `threshold`, `leaf_value`, `leaf_weight`
//!   (`tree.cpp` `ArrayToString<true>`).
//! - `{:g}` ([`format_g6`]): `split_gain`, `internal_value`, `internal_weight`
//!   (`tree.cpp` `ArrayToString` without the `<true>` arg).
//! - `shrinkage` uses C++ ostream-default (`std::defaultfloat`, precision 6,
//!   `tree.cpp:405` `str_buf << shrinkage_`), which is `{:g}`-equivalent for the
//!   typical shrinkage values that appear in real models. [`format_g6`] covers
//!   it pending the byte-exact `format_golden.txt` as the final arbiter
//!   (RESEARCH Open Q4, `03-RESEARCH.md:461-463`).
//!
//! ## `%g` rule (C / printf / `fmt`)
//! Let `P` = precision (>= 1). Format the value with `P` significant digits.
//! Let `X` be the decimal exponent of the value rounded to `P` significant
//! digits (the power of ten of the leading digit). Use:
//! - scientific notation (`d.ddde±XX`) iff `X < -4` or `X >= P`;
//! - fixed notation otherwise.
//! Then strip trailing zeros in the fractional part (and a trailing `.`).
//! C locale: `.` decimal point, lowercase `e`, exponent sign always present,
//! at least 2 exponent digits (matching `fmt`'s output).

/// Format an `f64` reproducing C printf `%.17g` / `fmt` `{:.17g}` byte-for-byte
/// (17 significant digits, `%g` fixed/scientific selection, trailing zeros
/// stripped, C locale).
///
/// Used for the predict-relevant model fields `threshold`, `leaf_value`,
/// `leaf_weight`. Round-trips bit-exact: `f64::from_str(&format_g17(x)) == Ok(x)`
/// for every finite `x` (Rust parse is correctly rounded, RESEARCH Pitfall 6).
pub fn format_g17(value: f64) -> String {
    format_g(value, 17)
}

/// Format an `f64` reproducing C printf `%g` / `fmt` `{:g}` at the default
/// precision 6 (6 significant digits).
///
/// Used for the metadata model fields `split_gain`, `internal_value`,
/// `internal_weight`, and (pending the golden arbiter) `shrinkage`.
pub fn format_g6(value: f64) -> String {
    format_g(value, 6)
}

/// Core `%g` formatter at an arbitrary precision `precision` (>= 1).
fn format_g(value: f64, precision: usize) -> String {
    debug_assert!(precision >= 1);

    // Non-finite: fmt prints `nan` / `inf` / `-inf`. These never appear in valid
    // model floats, but we must not panic.
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }

    // Sign handling — preserve the sign of zero (C `%g` prints `-0` for -0.0).
    let negative = value.is_sign_negative();
    let abs = value.abs();

    if abs == 0.0 {
        return if negative { "-0".to_string() } else { "0".to_string() };
    }

    // Obtain `precision` significant digits + the decimal exponent via Rust's
    // correctly-rounded scientific formatter. `{:.*e}` with `precision-1`
    // fractional digits yields exactly `precision` significant digits, e.g.
    // precision 17 -> "{:.16e}". Rust emits `d.ffffe±E` (mantissa always has a
    // leading non-zero digit for a non-zero value; `E` has the sign and the
    // minimal digit count, which we re-pad below).
    let sci = format!("{:.*e}", precision - 1, abs);
    let (mantissa, exp_str) = sci
        .split_once('e')
        .expect("Rust {:e} output always contains 'e'");
    let exp: i32 = exp_str
        .parse()
        .expect("Rust {:e} exponent is a valid i32");

    // `digits` = the `precision` significant decimal digits with NO point.
    // mantissa is like "1" (precision 1) or "1.0000000000000001"; strip the dot.
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    // Defensive: ensure exactly `precision` digits (Rust always emits this many
    // for the requested precision, but guard against any rounding-carry change
    // in digit count — `{:e}` keeps `precision` sig digits, the exponent absorbs
    // a 9.99..->1.00.. carry, so length is stable).
    debug_assert_eq!(digits.len(), precision, "sci mantissa digit count");

    // `%g` decision: scientific iff exp < -4 or exp >= precision.
    let body = if exp < -4 || exp >= precision as i32 {
        format_scientific(&mut digits, exp)
    } else {
        format_fixed(&mut digits, exp)
    };

    if negative {
        format!("-{body}")
    } else {
        body
    }
}

/// Build scientific `%g` form `d.ddde±EE` from the significant `digits` and the
/// decimal `exp` of the leading digit, stripping trailing zeros in the fraction.
fn format_scientific(digits: &mut String, exp: i32) -> String {
    // First digit before the point, the rest after.
    let mut chars = digits.chars();
    let lead = chars.next().expect("at least one significant digit");
    let mut frac: String = chars.collect();
    strip_trailing_zeros(&mut frac);

    let mantissa = if frac.is_empty() {
        lead.to_string()
    } else {
        format!("{lead}.{frac}")
    };

    // Exponent: lowercase `e`, explicit sign, min 2 digits (fmt / printf C locale).
    let sign = if exp < 0 { '-' } else { '+' };
    let mag = exp.unsigned_abs();
    format!("{mantissa}e{sign}{mag:02}")
}

/// Build fixed `%g` form from the significant `digits` and the decimal `exp`,
/// stripping trailing zeros (and a trailing `.`).
fn format_fixed(digits: &mut String, exp: i32) -> String {
    // `exp` is the power of ten of the leading digit. With `digits` significant
    // figures `d0 d1 d2 ...`, the value is `d0.d1d2... * 10^exp`.
    let n = digits.len() as i32;

    let result = if exp >= 0 {
        // Decimal point sits after `exp + 1` digits.
        let int_len = (exp + 1) as usize;
        if (exp + 1) >= n {
            // All significant digits are integer digits; pad with zeros.
            let zeros = (exp + 1 - n) as usize;
            let mut s = digits.clone();
            s.push_str(&"0".repeat(zeros));
            s
        } else {
            let (int_part, frac_part) = digits.split_at(int_len);
            let mut frac = frac_part.to_string();
            strip_trailing_zeros(&mut frac);
            if frac.is_empty() {
                int_part.to_string()
            } else {
                format!("{int_part}.{frac}")
            }
        }
    } else {
        // exp in [-4, -1]: leading "0." then `-exp-1` zeros then the digits.
        let lead_zeros = (-exp - 1) as usize;
        let mut frac = "0".repeat(lead_zeros);
        frac.push_str(digits);
        strip_trailing_zeros(&mut frac);
        if frac.is_empty() {
            "0".to_string()
        } else {
            format!("0.{frac}")
        }
    };

    result
}

/// Remove trailing `'0'` characters from a fractional-digit string in place.
fn strip_trailing_zeros(frac: &mut String) {
    while frac.ends_with('0') {
        frac.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::str::FromStr;

    /// The fixed double battery, paired with the expected C printf `%g` output.
    /// These literals are the C++ `fmt` `{:.17g}` / `{:g}` reference values;
    /// Task 4's `model-capture` emits a `format_golden.txt` produced by the real
    /// pip-`lightgbm` `fmt` for the SAME battery, and [`golden_matches_battery`]
    /// asserts the committed golden agrees with these literals (so the literals
    /// are never a hand-guess that drifts from the authoritative formatter).
    fn g17_battery() -> Vec<(f64, &'static str)> {
        vec![
            (0.1, "0.10000000000000001"),
            (1.0, "1"),
            (1e300, "1.0000000000000001e+300"),
            (1e-300, "1e-300"),
            (5e-324, "4.9406564584124654e-324"), // smallest subnormal
            (42.0, "42"),
            (0.30000000000000004, "0.30000000000000004"),
            (0.0, "0"),
            (-0.0, "-0"),
            (-1.5, "-1.5"),
            (123456789.0, "123456789"),
            (1234567890123456.0, "1234567890123456"),
            (12345678901234567.0, "12345678901234568"), // 17 int digits, exp=16 < 17 -> fixed
            (123456789012345678.0, "1.2345678901234568e+17"), // exp=17 >= 17 -> sci
            (0.0001, "0.0001"),         // exp=-4 -> fixed
            (0.00001, "1.0000000000000001e-05"), // exp=-5 -> sci
            (100.0, "100"),
        ]
    }

    fn g6_battery() -> Vec<(f64, &'static str)> {
        vec![
            (123456.789, "123457"),
            (1.0, "1"),
            (0.1, "0.1"),
            (1e300, "1e+300"),
            (1e-300, "1e-300"),
            (0.0, "0"),
            (-0.0, "-0"),
            (3.14159265, "3.14159"),
            (0.0001234567, "0.000123457"),
            (1234567.0, "1.23457e+06"),
            (100.0, "100"),
        ]
    }

    #[test]
    fn format_g17_battery() {
        for (x, expected) in g17_battery() {
            let got = format_g17(x);
            assert_eq!(
                got, expected,
                "format_g17({x:?}) = {got:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn format_g6_battery() {
        for (x, expected) in g6_battery() {
            let got = format_g6(x);
            assert_eq!(
                got, expected,
                "format_g6({x:?}) = {got:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn format_g17_explicit_cases() {
        // The headline DAT-09 case: 17 sig digits, NOT shortest-round-trip "0.1".
        assert_eq!(format_g17(0.1), "0.10000000000000001");
        assert_eq!(format_g17(1.0), "1");
        assert_eq!(format_g17(42.0), "42");
    }

    #[test]
    fn round_trip_bit_exact_g17() {
        // Rust f64::from_str is correctly rounded, so format_g17 must round-trip
        // bit-exact for every finite battery value (RESEARCH Pitfall 6).
        let mut values: Vec<f64> = g17_battery().into_iter().map(|(x, _)| x).collect();
        // A few extra messy floats.
        values.extend_from_slice(&[
            std::f64::consts::PI,
            std::f64::consts::E,
            1.0 / 3.0,
            2.0_f64.powi(-1074), // smallest subnormal
            f64::MIN_POSITIVE,
            f64::MAX,
            -f64::MAX,
            9.999999999999999e15,
            1.7976931348623157e308,
        ]);
        for x in values {
            if x == 0.0 {
                continue; // -0.0 / 0.0 round-trip handled separately below
            }
            let s = format_g17(x);
            let back = f64::from_str(&s)
                .unwrap_or_else(|_| panic!("format_g17({x:?}) = {s:?} did not parse"));
            assert_eq!(
                back.to_bits(),
                x.to_bits(),
                "round-trip failed: {x:?} -> {s:?} -> {back:?}"
            );
        }
        // signed-zero round-trip
        assert_eq!(f64::from_str(&format_g17(0.0)).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            f64::from_str(&format_g17(-0.0)).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn no_panic_on_extremes() {
        // Must not panic on these; values cannot appear in valid model floats but
        // the formatter is defensive.
        let _ = format_g17(f64::NAN);
        let _ = format_g17(f64::INFINITY);
        let _ = format_g17(f64::NEG_INFINITY);
        let _ = format_g17(5e-324);
        let _ = format_g17(1e300);
        let _ = format_g17(1e-300);
        let _ = format_g6(f64::NAN);
        assert_eq!(format_g17(f64::INFINITY), "inf");
        assert_eq!(format_g17(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_g17(f64::NAN), "nan");
    }

    fn golden_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/models/format_golden.txt")
    }

    /// If the committed `format_golden.txt` (emitted by `xtask model-capture`
    /// from the authoritative pip-`lightgbm` `fmt`) is present, assert that BOTH
    /// our `format_g17`/`format_g6` AND the in-test expected literals agree with
    /// it byte-for-byte. This is the arbiter that the hand-written battery
    /// literals are the genuine C++ `fmt` output. Pre-capture (golden absent),
    /// it SKIPs (the crate tests stay green before Task 4 lands the fixture).
    ///
    /// Golden format (line-delimited; `#` comments):
    /// ```text
    /// G17 bits=<u64> out=<string>
    /// G6  bits=<u64> out=<string>
    /// ```
    #[test]
    fn golden_matches_formatter() {
        let path = golden_path();
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!(
                "golden_matches_formatter: SKIP — golden {} not found. \
                 Run `cargo run -p xtask -- model-capture` to emit it.",
                path.display()
            );
            return;
        };
        let mut checked = 0usize;
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let toks: Vec<&str> = line.split_whitespace().collect();
            let kind = toks[0];
            let bits: u64 = toks
                .iter()
                .find_map(|t| t.strip_prefix("bits="))
                .expect("bits= field")
                .parse()
                .expect("u64 bits");
            // out= is the rest of the line after "out=" (may be empty / contain
            // no spaces — our formatter never emits spaces, so single token).
            let out = line
                .split_once("out=")
                .expect("out= field")
                .1
                .to_string();
            let x = f64::from_bits(bits);
            let got = match kind {
                "G17" => format_g17(x),
                "G6" => format_g6(x),
                other => panic!("unknown golden kind {other:?}"),
            };
            assert_eq!(
                got, out,
                "{kind} mismatch for f64::from_bits({bits}) = {x:?}: \
                 formatter gave {got:?}, C++ fmt golden gave {out:?}"
            );
            checked += 1;
        }
        assert!(checked > 0, "format_golden.txt contained no G17/G6 records");
    }
}
