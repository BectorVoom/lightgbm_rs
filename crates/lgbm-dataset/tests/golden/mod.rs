//! Shared fixture loader for the numeric binning golden-replay tests
//! (`bin_mapper_internals` = layer 1, `numeric_assignment` = layer 2).
//!
//! Parses `tests/fixtures/numeric_binning.txt` (written by
//! `cargo run -p xtask -- bin-capture`). Fixture path is resolved via
//! `CARGO_MANIFEST_DIR` — NEVER an absolute path and NEVER the untracked
//! `LightGBM/` tree (project memory `lightgbm-ref-tree-untracked`).
//!
//! Record format (line-delimited; `#`-prefixed lines are comments):
//! ```text
//! MASTER_SEED <seed>
//! COUNTS cases=<n>
//! CASE name=<id> max_bin=<m> min_data_in_bin=<d> min_split_data=<s> \
//!       pre_filter=<0|1> use_missing=<0|1> zero_as_missing=<0|1> \
//!       num_rows=<n> seed=<data_random_seed> sample_cnt=<c>
//! VALUES <f64bits;...>            # full per-row column, raw f64 bits (u64 dec)
//! GOLDEN num_bin=<n> bin_type=<0|1> missing_type=<0|1|2> default_bin=<u32> \
//!        most_freq_bin=<u32> is_trivial=<0|1> upper=<f64bits;...>
//! ASSIGN <u32;...>               # per-row ValueToBin over the column
//! ```

#![allow(dead_code)] // each test binary uses a subset of the fields.

use std::path::PathBuf;

/// C++ `MissingType` enum code (0=None, 1=Zero, 2=NaN).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingTypeCode {
    None,
    Zero,
    NaN,
}

/// One committed numeric binning case: the full input column + config + the
/// C++ golden (layer 1 internals + layer 2 per-row assignment).
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub column: Vec<f64>,
    pub max_bin: i32,
    pub min_data_in_bin: i32,
    pub min_split_data: i32,
    pub pre_filter: bool,
    pub use_missing: bool,
    pub zero_as_missing: bool,
    pub data_random_seed: i32,
    pub sample_cnt: i32,
    // golden layer 1
    pub golden_num_bin: i32,
    pub golden_bin_type: i32,
    pub golden_missing_type: MissingTypeCode,
    pub golden_default_bin: u32,
    pub golden_most_freq_bin: u32,
    pub golden_is_trivial: bool,
    pub golden_upper: Vec<f64>,
    // golden layer 2
    pub golden_assign: Vec<u32>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/numeric_binning.txt")
}

/// Load every committed case, or `None` if the fixture is absent (pre-capture).
pub fn load_cases() -> Option<Vec<Case>> {
    let text = std::fs::read_to_string(fixture_path()).ok()?;
    Some(parse(&text))
}

fn parse(text: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens[0] != "CASE" {
            // MASTER_SEED / COUNTS metadata — skip.
            continue;
        }
        // CASE line
        let name = field(&tokens, "name").expect("CASE name").to_string();
        let max_bin = parse_i32(&tokens, "max_bin");
        let min_data_in_bin = parse_i32(&tokens, "min_data_in_bin");
        let min_split_data = parse_i32(&tokens, "min_split_data");
        let pre_filter = parse_i32(&tokens, "pre_filter") != 0;
        let use_missing = parse_i32(&tokens, "use_missing") != 0;
        let zero_as_missing = parse_i32(&tokens, "zero_as_missing") != 0;
        let data_random_seed = parse_i32(&tokens, "seed");
        let sample_cnt = parse_i32(&tokens, "sample_cnt");

        // VALUES line
        let values_line = lines.next().expect("VALUES line after CASE").trim();
        let vt: Vec<&str> = values_line.split_whitespace().collect();
        assert_eq!(vt[0], "VALUES", "expected VALUES after CASE `{name}`");
        let column = parse_f64_bits_list(vt.get(1).copied().unwrap_or(""));

        // GOLDEN line
        let golden_line = lines.next().expect("GOLDEN line").trim();
        let gt: Vec<&str> = golden_line.split_whitespace().collect();
        assert_eq!(gt[0], "GOLDEN", "expected GOLDEN for case `{name}`");
        let golden_num_bin = parse_i32(&gt, "num_bin");
        let golden_bin_type = parse_i32(&gt, "bin_type");
        let golden_missing_type = match parse_i32(&gt, "missing_type") {
            0 => MissingTypeCode::None,
            1 => MissingTypeCode::Zero,
            2 => MissingTypeCode::NaN,
            other => panic!("unknown missing_type code {other} in case `{name}`"),
        };
        let golden_default_bin = parse_u32(&gt, "default_bin");
        let golden_most_freq_bin = parse_u32(&gt, "most_freq_bin");
        let golden_is_trivial = parse_i32(&gt, "is_trivial") != 0;
        let golden_upper =
            parse_f64_bits_list(field(&gt, "upper").expect("GOLDEN upper field"));

        // ASSIGN line
        let assign_line = lines.next().expect("ASSIGN line").trim();
        let at: Vec<&str> = assign_line.split_whitespace().collect();
        assert_eq!(at[0], "ASSIGN", "expected ASSIGN for case `{name}`");
        let golden_assign = parse_u32_list(at.get(1).copied().unwrap_or(""));

        cases.push(Case {
            name,
            column,
            max_bin,
            min_data_in_bin,
            min_split_data,
            pre_filter,
            use_missing,
            zero_as_missing,
            data_random_seed,
            sample_cnt,
            golden_num_bin,
            golden_bin_type,
            golden_missing_type,
            golden_default_bin,
            golden_most_freq_bin,
            golden_is_trivial,
            golden_upper,
            golden_assign,
        });
    }
    cases
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

fn parse_u32(tokens: &[&str], key: &str) -> u32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing u32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad u32 field `{key}`"))
}

/// Parse a `;`-separated list of raw f64 bit patterns (each a decimal `u64`)
/// back into `f64` via `from_bits` — lossless, exact.
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
