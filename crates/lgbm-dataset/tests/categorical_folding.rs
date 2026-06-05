//! Categorical folding golden replay (DAT-04): for every committed categorical
//! case, rebuild the Rust categorical `BinMapper` from the full column and assert
//! its category→bin map (`categorical_2_bin_` + `bin_2_categorical_`) and layer-1
//! internals are BIT-IDENTICAL to the C++ golden, and the per-row `value_to_bin`
//! vector matches via `compare_exact_u32`.
//!
//! This proves the categorical descending-count fold (stable `SortForPair`
//! is_reverse), the `0.99f` cut, the `min_data_in_bin` fold-break, and the NaN
//! dummy bin 0 are reproduced exactly (SC#3 layer 1+3, SC#5 per-row index).
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs`: `CARGO_MANIFEST_DIR`
//! fixture path (never the untracked `LightGBM/` tree), graceful SKIP pre-capture,
//! and localizing assert messages (case + category/row).
//!
//! Fixture written by `cargo run -p xtask -- bin-capture` into
//! `tests/fixtures/categorical_folding.txt`. Record format:
//! ```text
//! CCASE name=<id> max_bin=<m> min_data_in_bin=<d> use_missing=<0|1> \
//!       zero_as_missing=<0|1> pre_filter=<0|1> min_split_data=<s> num_rows=<n>
//! VALUES <f64bits;...>          # full per-row column, raw f64 bits (u64 dec)
//! MAPPER num_bin=<n> bin_type=<0|1> missing_type=<0|1|2> default_bin=<u32> \
//!        most_freq_bin=<u32> is_trivial=<0|1>
//! C2B <cat:bin;...>             # categorical_2_bin_, sorted by category key
//! B2C <cat;...>                 # bin_2_categorical_ (bin 0 == -1 NaN dummy)
//! ASSIGN <u32;...>              # per-row value_to_bin over the column
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use lgbm_dataset::bin_mapper::{BinMapper, BinType, MissingType};
use oracle_harness::comparator::compare_exact_u32;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/categorical_folding.txt")
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

fn parse_i32_list(s: &str) -> Vec<i32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<i32>().expect("i32 field"))
        .collect()
}

/// Parse `C2B` payload `cat:bin;cat:bin;...` into a `HashMap<i32, u32>`.
fn parse_c2b(s: &str) -> HashMap<i32, u32> {
    let mut m = HashMap::new();
    if s.is_empty() {
        return m;
    }
    for pair in s.split(';') {
        let (k, v) = pair.split_once(':').expect("C2B pair `cat:bin`");
        m.insert(
            k.parse::<i32>().expect("C2B category key"),
            v.parse::<u32>().expect("C2B bin value"),
        );
    }
    m
}

#[derive(Debug)]
struct CatGolden {
    name: String,
    column: Vec<f64>,
    max_bin: i32,
    min_data_in_bin: i32,
    min_split_data: i32,
    pre_filter: bool,
    use_missing: bool,
    zero_as_missing: bool,
    g_num_bin: i32,
    g_missing_type: MissingType,
    g_default_bin: u32,
    g_most_freq_bin: u32,
    g_is_trivial: bool,
    g_c2b: HashMap<i32, u32>,
    g_b2c: Vec<i32>,
    g_assign: Vec<u32>,
}

fn missing_from_code(code: i32) -> MissingType {
    match code {
        0 => MissingType::None,
        1 => MissingType::Zero,
        2 => MissingType::NaN,
        other => panic!("unknown missing_type code {other}"),
    }
}

fn parse(text: &str) -> Vec<CatGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "CCASE" {
            continue; // MASTER_SEED / COUNTS
        }
        let name = field(&t, "name").expect("CCASE name").to_string();
        let max_bin = parse_i32(&t, "max_bin");
        let min_data_in_bin = parse_i32(&t, "min_data_in_bin");
        let use_missing = parse_i32(&t, "use_missing") != 0;
        let zero_as_missing = parse_i32(&t, "zero_as_missing") != 0;
        let pre_filter = parse_i32(&t, "pre_filter") != 0;
        let min_split_data = parse_i32(&t, "min_split_data");

        let vt: Vec<&str> = lines.next().expect("VALUES").split_whitespace().collect();
        assert_eq!(vt[0], "VALUES", "expected VALUES after CCASE `{name}`");
        let column = parse_f64_bits_list(vt.get(1).copied().unwrap_or(""));

        let mt: Vec<&str> = lines.next().expect("MAPPER").split_whitespace().collect();
        assert_eq!(mt[0], "MAPPER", "expected MAPPER for `{name}`");
        let g_num_bin = parse_i32(&mt, "num_bin");
        let g_missing_type = missing_from_code(parse_i32(&mt, "missing_type"));
        let g_default_bin = parse_i32(&mt, "default_bin") as u32;
        let g_most_freq_bin = parse_i32(&mt, "most_freq_bin") as u32;
        let g_is_trivial = parse_i32(&mt, "is_trivial") != 0;

        let ct: Vec<&str> = lines.next().expect("C2B").split_whitespace().collect();
        assert_eq!(ct[0], "C2B", "expected C2B for `{name}`");
        let g_c2b = parse_c2b(ct.get(1).copied().unwrap_or(""));

        let bt: Vec<&str> = lines.next().expect("B2C").split_whitespace().collect();
        assert_eq!(bt[0], "B2C", "expected B2C for `{name}`");
        let g_b2c = parse_i32_list(bt.get(1).copied().unwrap_or(""));

        let at: Vec<&str> = lines.next().expect("ASSIGN").split_whitespace().collect();
        assert_eq!(at[0], "ASSIGN", "expected ASSIGN for `{name}`");
        let g_assign = parse_u32_list(at.get(1).copied().unwrap_or(""));

        out.push(CatGolden {
            name,
            column,
            max_bin,
            min_data_in_bin,
            min_split_data,
            pre_filter,
            use_missing,
            zero_as_missing,
            g_num_bin,
            g_missing_type,
            g_default_bin,
            g_most_freq_bin,
            g_is_trivial,
            g_c2b,
            g_b2c,
            g_assign,
        });
    }
    out
}

#[test]
fn categorical_folding_replays_every_committed_case() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "categorical_folding: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse(&text);
    assert!(!cases.is_empty(), "fixture present but parsed zero cases");

    let mut foldbreak_seen = false;
    let mut negative_seen = false;

    for c in &cases {
        // Rebuild the categorical mapper over the FULL column (sample_cnt =
        // num_rows; the C++ harness builds without sub-sampling so the map is
        // self-contained). NaN-strip happens inside find_bin_categorical.
        let mapper = BinMapper::find_bin_categorical(
            c.column.clone(),
            c.max_bin,
            c.min_data_in_bin,
            c.min_split_data,
            c.pre_filter,
            c.use_missing,
            c.zero_as_missing,
            c.column.len(),
        );

        assert_eq!(
            mapper.bin_type_,
            BinType::Categorical,
            "case `{}`: must be a categorical mapper",
            c.name
        );

        // Layer 1 internals.
        assert_eq!(
            mapper.num_bin_, c.g_num_bin,
            "case `{}`: num_bin mismatch",
            c.name
        );
        assert_eq!(
            mapper.missing_type_, c.g_missing_type,
            "case `{}`: missing_type mismatch",
            c.name
        );
        assert_eq!(
            mapper.default_bin_, c.g_default_bin,
            "case `{}`: default_bin mismatch",
            c.name
        );
        assert_eq!(
            mapper.most_freq_bin_, c.g_most_freq_bin,
            "case `{}`: most_freq_bin mismatch",
            c.name
        );
        assert_eq!(
            mapper.is_trivial_, c.g_is_trivial,
            "case `{}`: is_trivial mismatch",
            c.name
        );

        // Layer 3: category→bin map (order-independent) + bin→category vector.
        assert_eq!(
            mapper.categorical_2_bin_, c.g_c2b,
            "case `{}`: categorical_2_bin_ map mismatch",
            c.name
        );
        assert_eq!(
            mapper.bin_2_categorical_, c.g_b2c,
            "case `{}`: bin_2_categorical_ mismatch",
            c.name
        );

        // Per-row value_to_bin index vector, bit-exact.
        let rust_assign: Vec<u32> = c.column.iter().map(|&v| mapper.value_to_bin(v)).collect();
        compare_exact_u32(&rust_assign, &c.g_assign).unwrap_or_else(|m| {
            panic!("case `{}`: per-row value_to_bin {m}", c.name);
        });

        // Coverage flags for the required fixture properties.
        if c.name.contains("foldbreak") {
            foldbreak_seen = true;
        }
        if c.name.contains("negative") {
            negative_seen = true;
        }
    }

    assert!(
        foldbreak_seen,
        "corpus must include a min_data_in_bin fold-break (rare-level) case"
    );
    assert!(
        negative_seen,
        "corpus must include a negative/out-of-range category case"
    );
}
