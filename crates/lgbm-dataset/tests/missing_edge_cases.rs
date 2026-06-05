//! Missing-value routing golden replay (DAT-03): for each `(use_missing,
//! zero_as_missing)` config and edge column, rebuild the numeric Rust `BinMapper`
//! and assert layer-1 `missing_type`/`num_bin`/`default_bin` + the per-row
//! `value_to_bin` index vector are BIT-IDENTICAL to the C++ golden.
//!
//! Covers the three `MissingType` resolutions (None/Zero/NaN), `+0.0`/`-0.0`
//! signed zeros routing identically, all-missing collapsing to a valid single-bin
//! mapper, and single-value columns (SC#3 missing routing, SC#5 per-row index).
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs`: `CARGO_MANIFEST_DIR`
//! fixture path (never the untracked `LightGBM/` tree), graceful SKIP pre-capture,
//! and localizing assert messages (config + row).
//!
//! Fixture written by `cargo run -p xtask -- bin-capture` into
//! `tests/fixtures/missing_edge_cases.txt`. Record format:
//! ```text
//! MCASE name=<id> max_bin=<m> use_missing=<0|1> zero_as_missing=<0|1> num_rows=<n>
//! VALUES <f64bits;...>          # full per-row column, raw f64 bits (u64 dec)
//! MAPPER num_bin=<n> bin_type=<0|1> missing_type=<0|1|2> default_bin=<u32> \
//!        most_freq_bin=<u32> is_trivial=<0|1>
//! ASSIGN <u32;...>              # per-row value_to_bin over the column
//! ```

use std::path::PathBuf;

use lgbm_dataset::bin_mapper::{BinMapper, MissingType};
use oracle_harness::comparator::compare_exact_u32;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/missing_edge_cases.txt")
}

fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_i32(tokens: &[&str], key: &str) -> i32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing i32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad i32 field `{key}`"))
}

fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 field"))
        .collect()
}

fn missing_from_code(code: i32) -> MissingType {
    match code {
        0 => MissingType::None,
        1 => MissingType::Zero,
        2 => MissingType::NaN,
        other => panic!("unknown missing_type code {other}"),
    }
}

#[derive(Debug)]
struct MissGolden {
    name: String,
    column: Vec<f64>,
    max_bin: i32,
    use_missing: bool,
    zero_as_missing: bool,
    g_num_bin: i32,
    g_missing_type: MissingType,
    g_default_bin: u32,
    g_assign: Vec<u32>,
}

fn parse(text: &str) -> Vec<MissGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "MCASE" {
            continue; // MASTER_SEED / COUNTS
        }
        let name = field(&t, "name").expect("MCASE name").to_string();
        let max_bin = parse_i32(&t, "max_bin");
        let use_missing = parse_i32(&t, "use_missing") != 0;
        let zero_as_missing = parse_i32(&t, "zero_as_missing") != 0;

        let vt: Vec<&str> = lines.next().expect("VALUES").split_whitespace().collect();
        assert_eq!(vt[0], "VALUES", "expected VALUES after MCASE `{name}`");
        let column = parse_f64_bits_list(vt.get(1).copied().unwrap_or(""));

        let mt: Vec<&str> = lines.next().expect("MAPPER").split_whitespace().collect();
        assert_eq!(mt[0], "MAPPER", "expected MAPPER for `{name}`");
        let g_num_bin = parse_i32(&mt, "num_bin");
        let g_missing_type = missing_from_code(parse_i32(&mt, "missing_type"));
        let g_default_bin = parse_i32(&mt, "default_bin") as u32;

        let at: Vec<&str> = lines.next().expect("ASSIGN").split_whitespace().collect();
        assert_eq!(at[0], "ASSIGN", "expected ASSIGN for `{name}`");
        let g_assign = parse_u32_list(at.get(1).copied().unwrap_or(""));

        out.push(MissGolden {
            name,
            column,
            max_bin,
            use_missing,
            zero_as_missing,
            g_num_bin,
            g_missing_type,
            g_default_bin,
            g_assign,
        });
    }
    out
}

#[test]
fn missing_edge_cases_replays_every_committed_case() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "missing_edge_cases: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse(&text);
    assert!(!cases.is_empty(), "fixture present but parsed zero cases");

    let mut nan_seen = false;
    let mut zero_as_missing_seen = false;
    let mut signed_zero_seen = false;
    let mut all_missing_seen = false;

    for c in &cases {
        // Rebuild the numeric mapper over the FULL column (sample_cnt = num_rows).
        // NaN-strip + missing-type derivation happens inside find_bin_numeric.
        let mapper = BinMapper::find_bin_numeric(
            c.column.clone(),
            c.max_bin,
            /*min_data_in_bin=*/ 1,
            /*min_split_data=*/ 1,
            /*pre_filter=*/ false,
            c.use_missing,
            c.zero_as_missing,
            c.column.len(),
            &[],
        );

        let cfg = format!(
            "(use_missing={}, zero_as_missing={})",
            c.use_missing, c.zero_as_missing
        );

        assert_eq!(
            mapper.missing_type_, c.g_missing_type,
            "case `{}` {cfg}: missing_type mismatch",
            c.name
        );
        assert_eq!(
            mapper.num_bin_, c.g_num_bin,
            "case `{}` {cfg}: num_bin mismatch",
            c.name
        );
        assert_eq!(
            mapper.default_bin_, c.g_default_bin,
            "case `{}` {cfg}: default_bin mismatch",
            c.name
        );

        // Per-row value_to_bin index vector, bit-exact (covers NaN -> top-bin vs
        // treat-as-0, +0.0/-0.0 identical routing, all-missing single bin).
        let rust_assign: Vec<u32> = c.column.iter().map(|&v| mapper.value_to_bin(v)).collect();
        compare_exact_u32(&rust_assign, &c.g_assign).unwrap_or_else(|m| {
            panic!("case `{}` {cfg}: per-row value_to_bin {m}", c.name);
        });

        if c.g_missing_type == MissingType::NaN {
            nan_seen = true;
        }
        if c.zero_as_missing {
            zero_as_missing_seen = true;
        }
        if c.name.contains("signed_zero") {
            signed_zero_seen = true;
        }
        if c.name.contains("all_missing") {
            all_missing_seen = true;
        }
    }

    assert!(nan_seen, "corpus must include a MissingType::NaN routing case");
    assert!(
        zero_as_missing_seen,
        "corpus must include a zero_as_missing sweep"
    );
    assert!(
        signed_zero_seen,
        "corpus must include a +0.0/-0.0 signed-zero case"
    );
    assert!(
        all_missing_seen,
        "corpus must include an all-missing column case"
    );
}
